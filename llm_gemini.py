# llm_gemini.py

from google import genai
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

client = genai.Client(api_key=API_KEY)

def translate_explanation(text: str, lang: str) -> str:
    """
    Translate text into the specified language.
    """
    if not text:
        return ""

    prompt = f"""
Translate the following text into '{lang}'.

Rules:
- Only return the translated text
- Do NOT explain
- Preserve medical meaning

TEXT:
{text}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={"temperature": 0.2}
    )

    return response.text.strip()


def generate_med_explanation(meds, interactions):
    """
    Explain meds + interactions in simple language.
    """
    prompt = f"""
You are a medical safety assistant giving general, non-diagnostic explanations.

Meds detected: {meds}
Interactions found: {interactions}

Write a clear and friendly explanation including:
- What each medication is commonly used for
- Basic safety information
- Common side effects
- Serious warning signs to watch for
- If interactions exist, explain what the risk is
- Keep the tone simple and helpful

Rules:
- Do NOT give doses or medical instructions.
- Do NOT mention the patient.
- Keep it general, max 8 sentences.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={"temperature": 0.2}
    )

    return response.text.strip()
