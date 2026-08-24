"""Score a vision model field by field on forms whose every value was recorded as it was drawn.

Demonstrates how to tell a working extractor from one that merely returns JSON:
    1. Render claim forms locally, keeping the value written into each field.
    2. Plant five traps that a reader gets right and an extractor often does not.
    3. Render the same form in three languages, then again as a phone photograph of it.
    4. Ask the model for one strict JSON object per form.
    5. Score every field against the value that was drawn, not against a reading of the output.
    6. Sort the mistakes by kind and report which condition carries them.

Module 06: Multimodal Vision - Field-Level Extraction Audit.
"""

import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

OUT_DIR = Path(__file__).parent / "outputs" / "claim_forms"
MODEL = os.getenv("VISION_MODEL", "gemini-3.1-flash-lite")
FORM_WIDTH, FORM_HEIGHT = 720, 520

FIELDS = (
    "policy_number",
    "vehicle_model",
    "severity",
    "claim_amount",
    "driver_name",
    "road_surface",
)

# The traps, one per row. Each is a thing a person reads correctly off the page
# and an extractor gets wrong in a way that still looks like a valid answer.
TRAPS = {
    "policy_number": "characters that share a shape: the letter I against the digit 1",
    "vehicle_model": "a badge whose last character decides the model",
    "severity": "three boxes, one of them ticked",
    "driver_name": "a field blacked out on the page",
    "road_surface": "a field left blank, with a filled neighbour to borrow from",
}

# Labels only. Every value on the form stays identical across the three renders,
# so a difference in the score is a difference in reading the layout, not the data.
LABELS = {
    "english": {
        "title": "MOTOR CLAIM RECORD",
        "policy_number": "Policy number",
        "vehicle_model": "Vehicle",
        "severity": "Severity",
        "claim_amount": "Amount claimed",
        "driver_name": "Driver",
        "road_surface": "Road surface",
        "weather": "Weather",
        "options": ("Minor", "Moderate", "Severe"),
    },
    "french": {
        "title": "DECLARATION DE SINISTRE AUTO",
        "policy_number": "Numero de police",
        "vehicle_model": "Vehicule",
        "severity": "Gravite",
        "claim_amount": "Montant reclame",
        "driver_name": "Conducteur",
        "road_surface": "Etat de la chaussee",
        "weather": "Meteo",
        "options": ("Leger", "Modere", "Grave"),
    },
    "german": {
        "title": "KFZ-SCHADENSMELDUNG",
        "policy_number": "Versicherungsnummer",
        "vehicle_model": "Fahrzeug",
        "severity": "Schwere",
        "claim_amount": "Geforderter Betrag",
        "driver_name": "Fahrer",
        "road_surface": "Fahrbahnzustand",
        "weather": "Wetter",
        "options": ("Gering", "Mittel", "Schwer"),
    },
}

# What the form says, decided here and never read back off the picture.
TRUTH = {
    "policy_number": "IF-4821-77",
    "vehicle_model": "A6",
    "severity": None,  # per language: the option printed beside the ticked box
    "claim_amount": "1240.50",
    "driver_name": "REDACTED",
    "road_surface": "BLANK",
}

# Which of the three boxes is filled. The expected answer is the option printed
# beside it, which differs by language - asking for a translated word instead
# would score the model on its vocabulary rather than on which box it saw.
TICKED = 1

PROMPT = (
    "This image is a motor insurance claim form. Extract exactly these fields and "
    "return one JSON object with these keys and no others: "
    + ", ".join(FIELDS)
    + ". Rules: copy values exactly as printed, in the language they are printed in; "
    "for the severity field return the option label printed beside the ticked box, "
    "copied character for character and not translated; for a value that is blacked "
    "out return the string REDACTED; for a field left empty return the string BLANK; "
    "for the amount return digits and a decimal point only. Return the JSON and "
    "nothing else."
)

# How a form arrives when somebody photographs it on a desk rather than exporting
# it: a little rotation, a soft focus, a lighting gradient, and JPEG compression.
PHOTO_ROTATION = 1.4
PHOTO_BLUR = 0.8
PHOTO_QUALITY = 55


# The free tier caps requests per minute rather than per day, so a batch that
# fires as fast as the network allows will trip it. These two numbers pace it.
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 8


def load_font(size, bold=False):
    """Return a truetype face when the system has one, else the bundled default."""
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_form(language):
    """Draw one claim form and return its path. The values are the same every time."""
    labels = LABELS[language]
    img = Image.new("RGB", (FORM_WIDTH, FORM_HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    title_font = load_font(24, bold=True)
    label_font = load_font(15)
    value_font = load_font(17, bold=True)

    draw.rectangle([0, 0, FORM_WIDTH - 1, 58], fill=(28, 46, 74))
    draw.text((24, 18), labels["title"], font=title_font, fill="white")
    draw.rectangle([0, 0, FORM_WIDTH - 1, FORM_HEIGHT - 1], outline=(28, 46, 74), width=2)

    left, top, row_height = 34, 92, 62

    def row(index, key, value, value_colour=(15, 15, 15)):
        y = top + index * row_height
        draw.text((left, y), labels[key], font=label_font, fill=(90, 90, 90))
        draw.line([left, y + 44, FORM_WIDTH - left, y + 44], fill=(205, 205, 205))
        if value is not None:
            draw.text((left + 300, y + 2), value, font=value_font, fill=value_colour)
        return y

    row(0, "policy_number", TRUTH["policy_number"])
    row(1, "vehicle_model", f"Audi {TRUTH['vehicle_model']} Avant")
    row(2, "claim_amount", "1,240.50 EUR")

    # The driver's name is on the form and covered over, which is not the same as
    # the field being empty. An extractor that treats the two alike loses the
    # distinction the redaction was made to preserve.
    y = row(3, "driver_name", None)
    draw.rectangle([left + 298, y, left + 470, y + 26], fill=(20, 20, 20))

    # One field is filled and the one under it is not. The filled neighbour is
    # what an over-eager reader borrows from.
    y = row(4, "road_surface", None)
    draw.text((left + 300, y + 2), "", font=value_font, fill=(15, 15, 15))
    draw.text((left, y + 26), labels["weather"], font=label_font, fill=(90, 90, 90))
    draw.text((left + 300, y + 24), "Rain", font=value_font, fill=(15, 15, 15))

    # Three boxes, one ticked. Nothing is written next to the ticked box, so the
    # answer is carried by which box is filled and not by any text.
    y = top + 5 * row_height + 20
    draw.text((left, y), labels["severity"], font=label_font, fill=(90, 90, 90))
    box_x = left + 300
    for index, option in enumerate(labels["options"]):
        draw.rectangle([box_x, y, box_x + 16, y + 16], outline=(40, 40, 40), width=2)
        if index == TICKED:
            draw.rectangle([box_x + 4, y + 4, box_x + 12, y + 12], fill=(20, 20, 20))
        draw.text((box_x + 24, y - 1), option, font=label_font, fill=(15, 15, 15))
        box_x += 24 + int(draw.textlength(option, font=label_font)) + 30

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"claim_{language}.png"
    img.save(path)
    return path


def photograph(path):
    """Return a degraded copy of a form, standing in for a picture taken of it.

    Nothing about the content changes. Every value is still on the page, and a
    person still reads all six. What changes is how much work the characters take,
    which is where the difference between the two columns of the score comes from.
    """
    img = Image.open(path).convert("RGB")
    img = img.rotate(PHOTO_ROTATION, resample=Image.BICUBIC, expand=True, fillcolor="white")
    img = img.filter(ImageFilter.GaussianBlur(PHOTO_BLUR))

    # A lighting gradient across the page, brighter on one side than the other.
    width, height = img.size
    gradient = Image.linear_gradient("L").resize((width, height)).rotate(90)
    img = Image.composite(img, ImageEnhance.Brightness(img).enhance(0.62), gradient)
    img = ImageEnhance.Contrast(img).enhance(0.88)

    out = path.with_name(path.stem + "_photo.jpg")
    img.save(out, quality=PHOTO_QUALITY)
    return out


def call_with_retry(client, **kwargs):
    """Send one request, waiting out the per-minute request limit if it is hit.

    The free tier allows a fixed number of requests a minute, and a script that
    sends its whole batch as fast as it can will reach that limit part way through.
    Backing off and retrying is what keeps a run reproducible for someone else.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(BACKOFF_SECONDS * (attempt + 1))


def build_client():
    """Return an OpenAI-compatible client pointed at whichever key is present."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return OpenAI(
            api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY first.")


def ask(client, image_path):
    """Send one form and return the raw reply text."""
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    media = "jpeg" if image_path.suffix == ".jpg" else "png"
    response = call_with_retry(
        client,
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{media};base64,{encoded}"},
                    },
                ],
            }
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_json(text):
    """Pull the first JSON object out of a reply, fenced or not."""
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalise(field, value):
    """Reduce a reply to the form the truth table is written in.

    Every rule here is about presentation rather than content: casing, thousands
    separators, a trailing currency code. Normalising these away keeps the score
    on what was read rather than on how it was typed.
    """
    if value is None:
        return "NULL"
    text = str(value).strip()
    if field == "claim_amount":
        digits = re.sub(r"[^0-9.]", "", text.replace(",", ""))
        return digits.rstrip(".") or "NULL"
    if field == "vehicle_model":
        match = re.search(r"\b([A-Z]\d)\b", text.upper())
        return match.group(1) if match else text.upper()
    return text.upper()


def classify(field, expected, got):
    """Name the kind of mistake, using the trap that field was built around."""
    if got == "NULL":
        return "dropped"
    if field == "driver_name":
        return "redaction read as content" if got != "BLANK" else "redaction read as empty"
    if field == "road_surface":
        return "empty field filled in from elsewhere"
    if field == "severity":
        return "wrong option taken as ticked"
    if field in ("policy_number", "vehicle_model"):
        same_length = len(got) == len(expected)
        return "character misread" if same_length else "value not found"
    return "wrong value"


def expected_value(field, language):
    """Return the value drawn on that language's form, normalised for comparison."""
    if field == "severity":
        return normalise(field, LABELS[language]["options"][TICKED])
    return normalise(field, TRUTH[field])


def main():
    print("=" * 78)
    print("--- 1. Forms whose values are known because they were written here ---")
    for field in FIELDS:
        if field == "severity":
            printed = ", ".join(LABELS[lang]["options"][TICKED] for lang in LABELS)
            print(f"  {field:<16} box {TICKED + 1} of 3, printed as {printed}")
        else:
            print(f"  {field:<16} {TRUTH[field]}")

    print()
    print(f"--- 2. The {len(TRAPS)} traps on the page ---")
    for field, description in TRAPS.items():
        print(f"  {field:<16} {description}")

    print()
    print("--- 3. Three languages, then a photograph of each ---")
    sources = {}
    for language in LABELS:
        clean = render_form(language)
        photo = photograph(clean)
        sources[(language, "render")] = clean
        sources[(language, "photo")] = photo
        print(f"  {language:<9} {clean.name} -> {photo.name} "
              f"({clean.stat().st_size // 1024} KB, {photo.stat().st_size // 1024} KB)")
    print(f"saved under {OUT_DIR}")
    print(f"the photograph is the same page rotated {PHOTO_ROTATION} degrees, blurred, "
          f"lit unevenly and saved at JPEG quality {PHOTO_QUALITY}")

    print()
    print("--- 4. Asking for one JSON object per image ---")
    client = build_client()
    print(f"model: {MODEL}")
    replies = {}
    for key, path in sources.items():
        raw = ask(client, path)
        parsed = parse_json(raw)
        replies[key] = parsed
        state = "parsed" if parsed else f"unparseable: {raw[:56]!r}"
        print(f"  {key[0]:<9} {key[1]:<7} {state}")

    print()
    print("--- 5. Scoring every field against the value that was drawn ---")
    conditions = ("render", "photo")
    mistakes = []
    correct = {key: 0 for key in sources}
    for condition in conditions:
        print(f"{condition}:")
        print(f"  {'field':<16}" + "".join(f"{lang:<16}" for lang in LABELS))
        for field in FIELDS:
            row = f"  {field:<16}"
            for language in LABELS:
                expected = expected_value(field, language)
                got = normalise(field, (replies[(language, condition)] or {}).get(field))
                if got == expected:
                    correct[(language, condition)] += 1
                    row += f"{'ok':<16}"
                else:
                    row += f"{got[:14]:<16}"
                    mistakes.append((language, condition, field, expected, got))
            print(row)
    print()
    for condition in conditions:
        scored = sum(correct[(lang, condition)] for lang in LABELS)
        total = len(FIELDS) * len(LABELS)
        print(f"  {condition:<7} {scored}/{total} fields = {scored / total:.0%}")
    for language in LABELS:
        parts = "  ".join(
            f"{condition} {correct[(language, condition)]}/{len(FIELDS)}"
            for condition in conditions
        )
        print(f"  {language:<9} {parts}")

    print()
    print("--- 6. What kind of mistakes they are ---")
    if not mistakes:
        print("  none: every field on every image came back as drawn")
    else:
        kinds = {}
        for language, condition, field, expected, got in mistakes:
            kind = classify(field, expected, got)
            kinds.setdefault(kind, []).append(f"{language}/{condition}")
            print(f"  {language:<9} {condition:<7} {field:<16} wanted {expected!r}, "
                  f"got {got!r}  -> {kind}")
        print()
        for kind, entries in sorted(kinds.items(), key=lambda item: -len(item[1])):
            print(f"  {len(entries)}x {kind}: {', '.join(entries)}")
        by_condition = {
            condition: sum(1 for m in mistakes if m[1] == condition) for condition in conditions
        }
        print(f"  by condition: " + ", ".join(f"{k} {v}" for k, v in by_condition.items()))
    print()
    print("a reply that parses as JSON with every key present is not evidence of anything; "
          "the score above needed the drawn values to exist")
    print("=" * 78)


if __name__ == "__main__":
    main()
