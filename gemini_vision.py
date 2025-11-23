"""
Gemini Vision extraction helper (google-genai SDK).

- Client initialized once.
- Model name read from .env (GEMINI_MODEL).
- Always returns dict with:
    raw_text: str
    meds: list[str]
    explanation: optional str
"""

import os
import json
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=API_KEY)

PROMPT = """
You are an assistant that extracts medication names from prescription images.

Return ONLY valid JSON with this schema:
{
  "raw_text": "<the key text you see>",
  "meds": ["<medication name 1>", "<medication name 2>", ...]
}

Rules:
- meds must be a list of drug/medication names only.
- Do not include dosage, instructions, addresses, dates, or random words.
- Preserve proper capitalization where possible.
- If uncertain, include best guess but keep meds short.
"""


def _safe_json_parse(text: str) -> Dict[str, Any]:
    # try strict json first
    try:
        return json.loads(text)
    except Exception:
        pass

    # salvage: find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return {"raw_text": text, "meds": []}


def gemini_extract_drugs_from_image(img_bytes: bytes) -> Dict[str, Any]:
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_bytes}},
            ],
        }
    ]

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )

    text = (getattr(response, "text", None) or "").strip()
    data = _safe_json_parse(text)

    raw_text = str(data.get("raw_text", "")).strip()
    meds = data.get("meds", [])
    if not isinstance(meds, list):
        meds = [str(meds)]
    meds = [str(m).strip() for m in meds if str(m).strip()]

    return {"raw_text": raw_text, "meds": meds}
