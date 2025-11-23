"""
Fast, Gemini-only OCR + medication extraction API
with DrugDB interaction checking + Gemini explanation + TTS.
"""

import io
import os
import json
import base64
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import Response
from PIL import Image
from dotenv import load_dotenv

from gemini_vision import gemini_extract_drugs_from_image
from llm_gemini import generate_med_explanation, translate_explanation
from drugs import DrugDB
from interaction_db import InteractionDB
from tts_gemini import text_to_speech

# Load env
load_dotenv()

# Initialize DBs
drugdb = DrugDB("drugs.csv")
interactiondb = InteractionDB("interactions_clean.csv")

app = FastAPI(title="Med Local API (Gemini Vision)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------

class CheckImageResponse(BaseModel):
    raw_text: str
    gemini_meds: List[str]
    candidate_meds: List[str]
    checked_pairs: int
    dangerous_combinations: List[dict]
    explanation: str

class TTSRequest(BaseModel):
    text: str

# ---------- Helpers ----------

def compress_image(img_bytes: bytes, quality: int = 38, max_side: int = 1600) -> bytes:
    try:
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)

    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def normalize_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(i).strip() for i in value if str(i).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("\n", ",").split(",")]
        return [p for p in parts if p]
    return [str(value).strip()]

# ---------- Routes ----------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ocr/check-image", response_model=CheckImageResponse)
async def ocr_check_image_route(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    img_bytes = await file.read()
    img_small = compress_image(img_bytes)

    g = gemini_extract_drugs_from_image(img_small)
    raw_text = g.get("raw_text", "")
    gemini_meds = normalize_list(g.get("meds", []))

    normalized_meds = drugdb.normalize_many(gemini_meds)
    checked_pairs, interactions = interactiondb.check_list(normalized_meds)

    explanation = generate_med_explanation(normalized_meds, interactions)

    return CheckImageResponse(
        raw_text=raw_text,
        gemini_meds=normalized_meds,
        candidate_meds=normalized_meds,
        checked_pairs=checked_pairs,
        dangerous_combinations=interactions,
        explanation=explanation
    )


@app.post("/ocr/check-image/{lang}")
async def ocr_check_image_lang(lang: str, file: UploadFile = File(...)):
    lang = lang.lower().strip() or "en"

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    img_bytes = await file.read()

    g = gemini_extract_drugs_from_image(img_bytes)
    meds = g.get("meds", [])
    raw_text = g.get("raw_text", "")

    # ❗ FIXED: use correct variable names
    normalized_meds = drugdb.normalize_many(meds)
    checked_pairs, interactions = interactiondb.check_list(normalized_meds)

    explanation_en = generate_med_explanation(normalized_meds, interactions)

    if lang != "en":
        explanation = translate_explanation(explanation_en, lang)
        raw_text_translated = translate_explanation(raw_text, lang)
    else:
        explanation = explanation_en
        raw_text_translated = raw_text

    # audio = text_to_speech(explanation, lang)

    return {
        "lang": lang,
        "raw_text": raw_text_translated,
        "gemini_meds": normalized_meds,
        "candidate_meds": normalized_meds,
        "checked_pairs": checked_pairs,
        "dangerous_combinations": interactions,
        "explanation": explanation,
        # "audio_base64": base64.b64encode(audio).decode()
    }


@app.post("/tts/{lang}")
async def generate_audio(lang: str, req: TTSRequest):
    try:
        audio_bytes = text_to_speech(req.text, lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

    return Response(content=audio_bytes, media_type="audio/mpeg")
