import requests
import os
from app.core.config import settings

GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Use LLaMA model for cultural analysis
MODEL_NAME = "llama3-70b-8192"

SYSTEM_PROMPT = "You are an expert ornamentologist with deep knowledge of Central Asian cultural symbols and their meanings."

def get_grok_response(prompt: str) -> str:
    if not settings.GROK_API_KEY:
        raise RuntimeError("GROK_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {settings.GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    try:
        response = requests.post(GROK_API_URL, headers=headers, json=data)
        if response.status_code != 200:
            raise RuntimeError(f"Grok API error: {response.text}")
        response_data = response.json()
        if 'choices' not in response_data or not response_data['choices']:
            raise RuntimeError("Unexpected response format from Grok API")
        return response_data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling Grok API: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error calling Grok API: {str(e)}")
