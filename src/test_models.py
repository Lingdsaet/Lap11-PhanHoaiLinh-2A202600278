import os
from google import genai

def list_models():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set")
        return
    
    client = genai.Client(api_key=api_key)
    print("Available models:")
    for model in client.models.list():
        print(f"- {model.name}")

if __name__ == "__main__":
    list_models()
