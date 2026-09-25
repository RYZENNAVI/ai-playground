"""This script draws a small order table, sends the picture to a vision-language model
and asks for the table as JSON. The picture goes into an ordinary chat message. The
message has two parts: a text part with the instruction and an image_url part with the
picture encoded as base64.

The script uses Gemini when GEMINI_API_KEY is set, and OpenAI otherwise. The run prints
two parts:
    1. The rows of the table, drawn with Pillow and saved to outputs/order_table.png.
    2. The model's reply. The script does not parse it, so it may come wrapped in a
       Markdown code block.
"""

import base64
import os
import sys
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# Read the keys from the .env file at the repository root, if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Ensure UTF-8 output on the Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

api_key = gemini_key or openai_key

if not api_key:
    raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY in .env and retry.")

if gemini_key:
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    default_model = "gemini-3.1-flash-lite"
    provider_name = "Google Gemini"
else:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model = "gpt-4o-mini"
    provider_name = "OpenAI"

client = OpenAI(api_key=api_key, base_url=base_url)

IMAGE_PATH = Path(__file__).parent / "outputs" / "order_table.png"

TITLE = "Office supply order"
ROWS = [
    ("Item", "Quantity", "Unit price (EUR)", "Amount (EUR)"),
    ("Printer paper, A4", "10", "4.50", "45.00"),
    ("Ballpoint pens, box of 20", "3", "6.20", "18.60"),
    ("Stapler", "2", "12.90", "25.80"),
    ("Whiteboard markers", "5", "3.40", "17.00"),
    ("Total", "", "", "106.40"),
]
COLUMN_WIDTHS = (280, 110, 170, 150)
ROW_HEIGHT = 44

PROMPT = "Extract every field in this table and return it as JSON."


def load_font(size, bold=False):
    """Return a truetype face when the system has one, else the bundled default."""
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_table(path: Path) -> Path:
    """Draw TITLE and ROWS as a ruled table, save it as a PNG and return the path.
    The header row and the total row are bold."""
    margin, top = 20, 70
    img = Image.new("RGB", (sum(COLUMN_WIDTHS) + 2 * margin,
                            top + ROW_HEIGHT * len(ROWS) + margin), "white")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 22), TITLE, font=load_font(22, bold=True), fill="black")

    for r, row in enumerate(ROWS):
        font = load_font(16, bold=r in (0, len(ROWS) - 1))
        y = top + r * ROW_HEIGHT
        x = margin
        for text, width in zip(row, COLUMN_WIDTHS):
            draw.rectangle([x, y, x + width, y + ROW_HEIGHT], outline="black")
            draw.text((x + 10, y + 13), text, font=font, fill="black")
            x += width

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def extract_table_from_image(image_path: Path, model: str = default_model) -> str:
    """Send the picture and the instruction in one user message, and return the model's
    reply as text."""
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
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
    print("=== Table extraction from an image ===")
    print(f"Provider: {provider_name}")
    print(f"Model   : {default_model}\n")

    print("--- 1. Table image ---")
    path = draw_table(IMAGE_PATH)
    print(TITLE)
    for row in ROWS:
        print("  " + " | ".join(row))
    print(f"Saved to {path.relative_to(Path(__file__).parent)}")

    print("\n--- 2. Model reply ---")
    print(extract_table_from_image(path))
