# ocr.py
import easyocr

# Create reader once
reader = easyocr.Reader(["en"], gpu=False)

def extract_text_from_image(img_bytes: bytes) -> str:
    """Run OCR and return raw text."""
    import numpy as np
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(image)

    lines = reader.readtext(img_np, detail=0)
    return "\n".join(lines)


def find_medications_in_text(text: str):
    """
    Simple extraction: return alphabetic tokens >= 3 chars.
    """
    import re
    tokens = re.split(r"[^a-zA-Z]+", text)
    meds = [t.lower() for t in tokens if len(t) >= 3]
    return list(set(meds))
