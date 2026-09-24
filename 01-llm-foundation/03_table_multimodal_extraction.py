"""Turn a photographed table into JSON with a single vision-language request.

Demonstrates that an image travels inside an ordinary chat message:
    1. Pick Gemini when its key is set, otherwise OpenAI.
    2. Put a text part and an image_url part into the content of one user message.
    3. Ask for the table as JSON and print what comes back, unchecked.
"""

import os
import sys
from openai import OpenAI

# Automatically load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Client Setup: Google Gemini (Primary / Free) with OpenAI Fallback
# ---------------------------------------------------------------------------
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

api_key = gemini_key or openai_key

if not api_key:
    raise RuntimeError("No API key found! Please set GEMINI_API_KEY (Google AI Studio) or OPENAI_API_KEY in .env file.")

if gemini_key:
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    default_model = "gemini-2.5-flash"
    provider_name = "Google Gemini"
else:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model = "gpt-4o-mini"
    provider_name = "OpenAI"

client = OpenAI(api_key=api_key, base_url=base_url)

IMAGE_URL = "https://aiwucai.oss-cn-huhehaote.aliyuncs.com/pdf_table.jpg"


def extract_table_from_image(image_url: str = IMAGE_URL, model: str = default_model) -> str:
    """Send image URL + prompt to VLM and extract JSON data."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "This is a table image. Please extract all the table content and output it in a structured JSON format.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(f"=== Multimodal Table Extraction ===")
    print(f"Provider: {provider_name}")
    print(f"Model   : {default_model}")
    print(f"Image   : {IMAGE_URL}\n")
    try:
        result = extract_table_from_image()
        print("Extracted JSON Result:")
        print(result)
    except Exception as e:
        print(f"Call failed: {e}")
