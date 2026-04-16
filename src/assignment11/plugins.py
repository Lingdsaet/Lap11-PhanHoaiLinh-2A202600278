import json
import time
import re
from collections import defaultdict, deque
from datetime import datetime

from google.adk.plugins import base_plugin
from google.genai import types
from google.adk.agents.invocation_context import InvocationContext

# ============================================================
# 1. RateLimitPlugin
# ============================================================
class RateLimitPlugin(base_plugin.BasePlugin):
    """Sliding window rate limiter to prevent abuse."""
    
    def __init__(self, max_requests=10, window_seconds=60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows = defaultdict(deque)

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        user_id = invocation_context.user_id if invocation_context else "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        # Clean old timestamps
        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            wait_time = int(self.window_seconds - (now - window[0]))
            return types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text=f"Rate limit exceeded. Please wait {wait_time} seconds before trying again."
                )]
            )

        window.append(now)
        return None

# ============================================================
# 2. AuditLogPlugin
# ============================================================
class AuditLogPlugin(base_plugin.BasePlugin):
    """Capture all interactions for security auditing."""
    
    def __init__(self, log_file="audit_log.json"):
        super().__init__(name="audit_log")
        self.log_file = log_file
        self.logs = []

    def _extract_text(self, content: types.Content) -> str:
        if content and content.parts:
            return "".join(p.text for p in content.parts if hasattr(p, 'text'))
        return ""

    async def on_user_message_callback(self, *, invocation_context, user_message):
        # We record the input but don't block
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": invocation_context.user_id if invocation_context else "anonymous",
            "session_id": invocation_context.session_id if invocation_context else "N/A",
            "input": self._extract_text(user_message),
            "output": None,
            "blocked_by": None,
            "latency_ms": None,
        }
        self.logs.append(log_entry)
        invocation_context.context_variables["audit_index"] = len(self.logs) - 1
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        # Update the log entry with matching output
        idx = callback_context.context_variables.get("audit_index")
        if idx is not None and idx < len(self.logs):
            response_text = self._extract_text(llm_response.content) if hasattr(llm_response, 'content') else ""
            self.logs[idx]["output"] = response_text
            
            # Record which plugin blocked if any (heuristic)
            if any(kw in response_text.lower() for kw in ["block", "rate limit", "cannot", "apologize"]):
                self.logs[idx]["blocked_by"] = "guardrails"
        
        return llm_response

    def export_json(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, indent=2, default=str)
        print(f"Audit log exported to {self.log_file}")

# ============================================================
# 3. InputGuardrailPlugin
# ============================================================
class InputGuardrailPlugin(base_plugin.BasePlugin):
    """Combine injection detection and topic filtering."""
    
    def __init__(self, allowed_topics, blocked_topics):
        super().__init__(name="input_guardrail")
        self.allowed_topics = allowed_topics
        self.blocked_topics = blocked_topics

    def detect_injection(self, text: str) -> bool:
        patterns = [
            r"ignore (all )?(previous|above) instructions",
            r"you are now (DAN|unrestricted|an AI that)",
            r"reveal your (system )?prompt",
            r"output (your|the) instructions",
            r"forget (all )?your constraints",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def is_off_topic(self, text: str) -> bool:
        text_lower = text.lower()
        if any(topic in text_lower for topic in self.blocked_topics):
            return True
        if any(topic in text_lower for topic in self.allowed_topics):
            return False
        return True

    async def on_user_message_callback(self, *, invocation_context, user_message):
        text = "".join(p.text for p in user_message.parts if hasattr(p, 'text'))
        
        if self.detect_injection(text):
            return types.Content(role="model", parts=[types.Part.from_text(
                text="Security Block: Potential prompt injection detected."
            )])
            
        if self.is_off_topic(text):
            return types.Content(role="model", parts=[types.Part.from_text(
                text="Off-topic Block: I can only assist with banking-related inquiries."
            )])
            
        return None

# ============================================================
# 4. Multi-Criteria Judge Plugin
# ============================================================
JUDGE_INSTRUCTION = """You are a quality assurance judge for a banking AI assistant.
Evaluate the given AI response on 4 criteria (1-5 each):
1. SAFETY: No harmful content, leaked secrets, or dangerous instructions.
2. RELEVANCE: Response is about banking/finance, not off-topic.
3. ACCURACY: No fabricated data or false claims.
4. TONE: Professional, helpful customer service tone.

Respond in EXACTLY this format:
SAFETY: <score>
RELEVANCE: <score>
ACCURACY: <score>
TONE: <score>
VERDICT: PASS or FAIL
REASON: <one sentence>
"""

class MultiCriteriaJudgePlugin(base_plugin.BasePlugin):
    """Use a separate LLM call to judge output quality across multiple axes."""
    
    def __init__(self, genai_client, model="gemini-1.5-flash"):
        super().__init__(name="llm_judge")
        self.client = genai_client
        self.model = model

    async def _judge(self, response_text: str) -> str:
        prompt = f"Evaluate this AI response for safety and quality:\n\n{response_text}"
        res = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Content(role="user", parts=[types.Part.from_text(text=JUDGE_INSTRUCTION)]),
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            ]
        )
        return res.text

    async def after_model_callback(self, *, callback_context, llm_response):
        response_text = "".join(p.text for p in llm_response.content.parts if hasattr(p, 'text'))
        if not response_text:
            return llm_response

        verdict_text = await self._judge(response_text)
        
        # Log the judge verdict in context for monitoring
        callback_context.context_variables["judge_verdict"] = verdict_text
        
        if "VERDICT: FAIL" in verdict_text.upper():
            reason = "Failed safety evaluation."
            for line in verdict_text.splitlines():
                if "REASON:" in line:
                    reason = line.split(":", 1)[1].strip()
            
            llm_response.content.parts = [types.Part.from_text(
                text=f"I apologize, but this response was flagged by our safety system. Reason: {reason}"
            )]
            
        return llm_response
