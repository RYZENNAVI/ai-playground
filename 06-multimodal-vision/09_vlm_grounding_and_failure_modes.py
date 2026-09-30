"""This script asks a vision-language model to box a wheel, to read a dense departures
board and to answer follow-ups about an image, testing visual grounding, dense text
reading and image context across turns. The script draws its own images, so every answer
has a known value. The wheel's box is scored by intersection over union (IoU). The board
has 26 small rows, and the reply is checked for degenerate repetition. The follow-up
questions show whether the model still sees the image once it is dropped from the chat
history.

The run prints six parts:
    1. A side view of a vehicle, with the box of its front wheel recorded as drawn.
    2. The model's box for that wheel, scored by IoU under every coordinate convention.
    3. A departures board of 26 rows, and the airlines with a flight in zone A.
    4. The model's list of those airlines, against the drawn rows.
    5. Whether that reply collapsed into repetition.
    6. Four follow-up questions, with the image kept in the history and dropped from it.
"""

import base64
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from PIL import Image, ImageDraw, ImageFont

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

OUT_DIR = Path(__file__).parent / "outputs" / "grounding"
SEED = 3407

SCENE_SIZE = (640, 420)
BOARD_SIZE = (760, 620)
ROW_COUNT = 26
ZONES = ("A", "B", "C")

AIRLINES = (
    "Northwind Air", "Cedar Jet", "Blue Harbour", "Kestrel Airways", "Vantage Air",
    "Orchard Express", "Halcyon Air", "Tidewater Jet", "Ridgeline Air", "Marlin Airways",
)

# A collapsed reply repeats one entry again and again. It counts as degenerate once
# one entry appears this many times.
REPETITION_LIMIT = 4


# Retry a rate-limited request up to five times, waiting 8, 16, 24 and 32 seconds.
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


def draw_scene():
    """Draw a side view of a vehicle and return the image path and the box of its front wheel.

    The box comes from the drawing, so the overlap is scored against a known coordinate.
    """
    width, height = SCENE_SIZE
    img = Image.new("RGB", SCENE_SIZE, (208, 214, 220))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, height - 90, width, height], fill=(120, 124, 130))
    draw.rectangle([120, 170, 520, 300], fill=(46, 86, 148))
    draw.polygon([(190, 170), (300, 110), (420, 110), (470, 170)], fill=(70, 116, 180))
    draw.rectangle([300, 118, 415, 165], fill=(180, 205, 230))

    front = (168, 268, 240, 340)
    rear = (410, 268, 482, 340)
    for box in (front, rear):
        draw.ellipse(box, fill=(28, 28, 30))
        inset = 14
        draw.ellipse(
            [box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset], fill=(196, 198, 202)
        )
    draw.text((26, 24), "SIDE VIEW", font=load_font(20, bold=True), fill=(40, 40, 40))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "scene.png"
    img.save(path)
    return path, front


def draw_board(seed=SEED):
    """Draw a departures board of small dense rows and return what each row says."""
    rng = random.Random(seed)
    width, height = BOARD_SIZE
    img = Image.new("RGB", BOARD_SIZE, (16, 18, 24))
    draw = ImageDraw.Draw(img)
    header_font = load_font(17, bold=True)
    row_font = load_font(12)

    draw.text((20, 16), "DEPARTURES", font=header_font, fill=(240, 200, 60))
    columns = (20, 110, 330, 430, 530, 640)
    for label, x in zip(("FLIGHT", "AIRLINE", "GATE", "ZONE", "TIME", "STATUS"), columns):
        draw.text((x, 48), label, font=row_font, fill=(150, 156, 170))

    rows = []
    y = 70
    for index in range(ROW_COUNT):
        airline = AIRLINES[index % len(AIRLINES)]
        row = {
            "flight": f"{airline[:2].upper()}{rng.randint(100, 999)}",
            "airline": airline,
            # The gate letter is drawn independently of the zone, so it cannot
            # stand in for the zone column.
            "gate": f"{rng.choice(ZONES)}{rng.randint(1, 24)}",
            "zone": ZONES[index % len(ZONES)],
            "time": f"{rng.randint(6, 22):02d}:{rng.choice(('05', '20', '35', '50'))}",
            "status": rng.choice(("On time", "Boarding", "Delayed")),
        }
        rows.append(row)
        values = (row["flight"], row["airline"], row["gate"], row["zone"], row["time"], row["status"])
        for value, x in zip(values, columns):
            draw.text((x, y), value, font=row_font, fill=(226, 230, 238))
        y += 20

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "board.png"
    img.save(path)
    return path, rows


def call_with_retry(client, **kwargs):
    """Send one request, retrying with a longer wait each time the rate limit is hit."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(BACKOFF_SECONDS * (attempt + 1))


def build_client():
    """Return a client and model name: Gemini when GEMINI_API_KEY is set, else OPENAI_API_KEY."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return OpenAI(
            api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        ), os.getenv("VISION_MODEL", "gemini-3.1-flash-lite")
    if os.getenv("OPENAI_API_KEY"):
        # OPENAI_BASE_URL belongs to OPENAI_API_KEY only, so a Gemini key is never
        # sent to whatever endpoint that variable points at.
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ), os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY first.")


def image_part(path):
    """Return the OpenAI content part that carries a local file as a data URL."""
    encoded = base64.b64encode(path.read_bytes()).decode()
    media = "jpeg" if path.suffix == ".jpg" else "png"
    return {"type": "image_url", "image_url": {"url": f"data:image/{media};base64,{encoded}"}}


def ask(client, model, messages):
    """Send a prepared message list and return the reply text."""
    response = call_with_retry(client, model=model, messages=messages, temperature=0)
    return response.choices[0].message.content or ""


def extract_numbers(text):
    """Pull the four coordinates out of a reply, whether it is JSON or prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            for value in payload.values():
                if isinstance(value, (list, tuple)) and len(value) == 4:
                    return [float(v) for v in value]
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    found = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [float(v) for v in found[:4]] if len(found) >= 4 else None


def interpretations(numbers, size):
    """Return the four numbers read as pixels, a 0-1 fraction and a 0-1000 grid, each in both axis orders.

    A reply does not say which convention it used, so every reading gets scored.
    """
    width, height = size
    a, b, c, d = numbers
    scales = {
        "pixels": (1.0, 1.0),
        "0-1 fraction": (width, height),
        "0-1000 grid": (width / 1000, height / 1000),
    }
    # Numbers this large cannot be fractions, so that reading is dropped.
    if max(numbers) > 1.5:
        scales.pop("0-1 fraction")
    out = {}
    for name, (sx, sy) in scales.items():
        out[f"{name} as x1 y1 x2 y2"] = (a * sx, b * sy, c * sx, d * sy)
        out[f"{name} as y1 x1 y2 x2"] = (b * sx, a * sy, d * sx, c * sy)
    return out


def iou(box_a, box_b):
    """Return intersection over union for two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def repetition(items, text):
    """Return the count of the most repeated whole entry, that entry, and the reply length.

    Four of these airlines end in the word Air, so counting words would flag a correct answer.
    """
    if not items:
        return 0, "", len(text)
    entry, count = Counter(items).most_common(1)[0]
    return count, entry, len(text)


def main():
    print("=" * 78)
    client, model = build_client()
    print(f"model: {model}")

    # 1. The scene
    print()
    print("--- 1. A scene with the target's box recorded as it was drawn ---")
    scene_path, target = draw_scene()
    print(f"{scene_path.name}  {SCENE_SIZE[0]}x{SCENE_SIZE[1]}")
    print(f"the front wheel occupies {target}, which is "
          f"{(target[2] - target[0]) * (target[3] - target[1]) / (SCENE_SIZE[0] * SCENE_SIZE[1]):.1%} "
          f"of the frame")

    # 2. Pointing at the wheel
    print()
    print("--- 2. Asking the model to point at it ---")
    grounding_prompt = (
        "Look at this image of a vehicle. Return the bounding box of the FRONT wheel, "
        "the one on the left of the picture. Answer with one JSON object of the form "
        '{"box": [x1, y1, x2, y2]} and nothing else. State the coordinates in pixels '
        "of this image."
    )
    reply = ask(client, model, [{"role": "user", "content": [
        {"type": "text", "text": grounding_prompt}, image_part(scene_path)]}])
    print(f"reply: {reply.strip()[:120]!r}")
    numbers = extract_numbers(reply)
    if numbers is None:
        print("  no four numbers came back, so there is nothing to score")
        print("  a reply can acknowledge the request in full sentences and still point at nothing")
    else:
        print(f"  four numbers: {numbers}")
        print(f"  the wheel was drawn at ({target[0]}, {target[1]}, {target[2]}, {target[3]})")
        scored = sorted(
            ((name, box, iou(box, target)) for name, box in
             interpretations(numbers, SCENE_SIZE).items()),
            key=lambda item: -item[2],
        )
        for name, box, score in scored:
            marker = "  <-- lands on it" if score >= 0.5 else ""
            print(f"  {name:<26} ({box[0]:>6.0f},{box[1]:>6.0f},{box[2]:>6.0f},{box[3]:>6.0f})  "
                  f"IoU {score:.3f}{marker}")
        best_name, best_box, best_score = scored[0]
        as_pixels = next(s for n, _, s in scored if n == "pixels as x1 y1 x2 y2")
        print(f"  best reading is {best_name} at IoU {best_score:.3f}; the same four numbers "
              f"read as pixels score {as_pixels:.3f}")

        # One number can be right while another is wrong, and a single overlap
        # figure hides which. Splitting it by axis says where the model went.
        span_x = (max(0.0, min(best_box[2], target[2]) - max(best_box[0], target[0]))
                  / (target[2] - target[0]))
        span_y = (max(0.0, min(best_box[3], target[3]) - max(best_box[1], target[1]))
                  / (target[3] - target[1]))
        area_ratio = (((best_box[2] - best_box[0]) * (best_box[3] - best_box[1]))
                      / ((target[2] - target[0]) * (target[3] - target[1])))
        inside = span_x > 0.999 and span_y > 0.999
        print(f"  it holds {span_x:.0%} of the wheel's width and {span_y:.0%} of its height"
              + (", so the wheel is entirely inside the box" if inside else ""))
        print(f"  the box is {area_ratio:.1f} times the wheel's area, "
              f"{best_box[3] - best_box[1]:.0f} pixels tall against {target[3] - target[1]}"
              + (", so all of the missing overlap is surplus box" if inside else ""))
        if as_pixels == 0 and best_score > 0:
            print("  the reply did not say which convention it used, though the prompt asked "
                  "for pixels; read as pixels, this partial hit scores zero")

    # 3. The board
    print()
    print("--- 3. A dense board whose every row is known ---")
    board_path, rows = draw_board()
    zone = "A"
    in_zone = sorted({row["airline"] for row in rows if row["zone"] == zone})
    print(f"{board_path.name}  {BOARD_SIZE[0]}x{BOARD_SIZE[1]}, {len(rows)} rows of 12 px text")
    print(f"zone {zone} holds {sum(1 for r in rows if r['zone'] == zone)} flights "
          f"from {len(in_zone)} airlines")
    print(f"  {', '.join(in_zone)}")

    # 4. Airlines in zone A
    print()
    print("--- 4. Asking a question that needs several of those rows ---")
    board_prompt = (
        f"This is a departures board. List every distinct airline that has a flight in "
        f"zone {zone}. Answer with a JSON array of airline names and nothing else."
    )
    board_reply = ask(client, model, [{"role": "user", "content": [
        {"type": "text", "text": board_prompt}, image_part(board_path)]}])
    named = re.findall(r'"([^"]+)"', board_reply) or re.findall(r"[A-Z][a-z]+ [A-Za-z]+", board_reply)
    predicted = {name.strip() for name in named}
    truth_set = set(in_zone)
    hits = predicted & truth_set
    print(f"reply length {len(board_reply)} characters, {len(predicted)} distinct names")
    print(f"  correct   {len(hits)}/{len(truth_set)}: {', '.join(sorted(hits)) or 'none'}")
    missed = truth_set - predicted
    invented = predicted - truth_set
    print(f"  missed    {', '.join(sorted(missed)) or 'none'}")
    print(f"  not in zone {zone}: {', '.join(sorted(invented)) or 'none'}")

    # 5. Repetition
    print()
    print("--- 5. Checking whether the reply collapsed ---")
    count, entry, length = repetition(named, board_reply)
    expected_length = sum(len(name) + 4 for name in truth_set) + 4
    print(f"most repeated entry: {entry!r} appears {count} time(s)")
    print(f"reply is {length} characters against about {expected_length} for a bare list")
    if count >= REPETITION_LIMIT or length > expected_length * 4:
        print(f"  past the limit of {REPETITION_LIMIT} repeats or four times the expected "
              f"length, so this reply degenerated rather than answered")
    else:
        print("  inside both limits, so the reply held together")
    words = Counter(re.findall(r"[A-Za-z]+", board_reply)).most_common(1)
    if words and words[0][1] >= REPETITION_LIMIT > count:
        print(f"  counted by word instead, {words[0][0]!r} appears {words[0][1]} times, which "
              f"reaches the limit and would call this answer degenerate")

    # 6. Follow-ups with the image kept and dropped
    print()
    print("--- 6. Follow-up questions, with the image left in the history and dropped from it ---")
    first_prompt = (
        "What is the flight number in the first row of this board? Answer with just the code."
    )
    first_reply = ask(client, model, [{"role": "user", "content": [
        {"type": "text", "text": first_prompt}, image_part(board_path)]}])
    print(f"turn 1, image attached: {first_reply.strip()[:40]!r}, drawn as {rows[0]['flight']!r}")
    print("the follow-ups all ask for something the first reply never mentioned:")

    probes = [
        ("gate of row 1", "What is the gate for that same flight? Answer with just the gate.",
         rows[0]["gate"]),
        ("time of row 1", "What is the departure time of that same flight? Answer with just "
                          "the time.", rows[0]["time"]),
        ("airline of row 4", "What airline operates the flight in the fourth row? Answer with "
                             "just the name.", rows[3]["airline"]),
        ("status of row 2", "What is the status of the flight in the second row? Answer with "
                            "just the status.", rows[1]["status"]),
    ]
    # Two histories, differing in one thing. The first keeps the message the image
    # was attached to. The second replaces that message with its text, which is
    # what a client does when it stores a transcript as strings.
    kept = [
        {"role": "user", "content": [{"type": "text", "text": first_prompt},
                                     image_part(board_path)]},
        {"role": "assistant", "content": first_reply},
    ]
    dropped = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_reply},
    ]

    tally = {"kept": 0, "dropped": 0}
    blind_hits = []
    print(f"  {'probe':<18}{'drawn':<16}{'image kept':<20}{'image dropped':<20}")
    for label, question, expected in probes:
        answers = {
            "kept": ask(client, model, kept + [{"role": "user", "content": question}]),
            "dropped": ask(client, model, dropped + [{"role": "user", "content": question}]),
        }
        cells = {}
        for key, answer in answers.items():
            hit = expected.lower() in answer.lower()
            tally[key] += hit
            if key == "dropped" and hit:
                blind_hits.append(label)
            cells[key] = f"{answer.strip()[:12]!r} {'ok' if hit else 'no'}"
        print(f"  {label:<18}{expected:<16}{cells['kept']:<20}{cells['dropped']:<20}")

    print(f"  image kept in the history: {tally['kept']}/{len(probes)} correct; "
          f"dropped from it: {tally['dropped']}/{len(probes)}")
    if "status of row 2" in blind_hits:
        print("  status is one of three values, so a blind guess lands on it about a third of "
              "the time; the hit there without the image is weaker evidence than a hit on "
              "gate, time or airline, which have far more possible values")
    print("  the image does not have to ride on the newest message, but it does have to "
          "still be on one of them")
    print("  a transcript stored as plain strings loses it silently, and the next answer "
          "comes back in the same confident shape as the ones that could see")
    print("=" * 78)


if __name__ == "__main__":
    main()
