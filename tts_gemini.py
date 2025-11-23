import base64
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

def text_to_speech(text: str, lang: str = "en") -> bytes:
    """
    Working TTS for google-genai >= 1.50.0
    Uses content parts for both text + audio config
    """

    voice = "en-US-Neural2-F"

    response = client.models.generate_content(
        model="gemini-tts-1",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": text},
                    {
                        "voiceConfig": {"voiceName": voice},
                    },
                    {
                        "audioConfig": {
                            "audioEncoding": "MP3"
                        }
                    }
                ]
            }
        ]
    )

    # Extract audio data
    for part in response.candidates[0].content.parts:
        if hasattr(part, "audio") and hasattr(part.audio, "data"):
            return base64.b64decode(part.audio.data)

    raise RuntimeError("Gemini TTS returned no audio data")