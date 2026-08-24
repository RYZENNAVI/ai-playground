"""Answer questions about a clip with an image model, and measure what the sampling costs.

Demonstrates a frame-sampling stand-in for a video model, and its two blind spots:
    1. Synthesise a clip whose events happen at times chosen here, not observed.
    2. Encode it, so the rest of the script reads a video file like any other.
    3. Sample keyframes on a fixed stride and record which events each one can reach.
    4. Ask the vision model one question per sampled frame.
    5. Assemble the answers into a timeline and locate the event from it.
    6. Re-sample the same answers at coarser strides and price each one.

Module 06: Multimodal Vision - Video by Keyframe Sampling.
"""

import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

OUT_DIR = Path(__file__).parent / "outputs" / "clip"
VIDEO_PATH = OUT_DIR / "approach.mp4"
MODEL = os.getenv("VISION_MODEL", "gemini-3.1-flash-lite")

WIDTH, HEIGHT = 480, 270
FPS = 30
DURATION_SECONDS = 4
TOTAL_FRAMES = FPS * DURATION_SECONDS

# The two events, decided here. One lasts for the rest of the clip, the other for
# a tenth of a second.
IMPACT_FRAME = 78
FLASH_FRAME = 45
FLASH_LENGTH = 3

SAMPLE_STRIDE = 10
COARSER_STRIDES = (10, 20, 30, 40)

QUESTION = (
    "This is one frame from a dashcam clip of a car passing a barrier. Answer with "
    "one word, either DAMAGED if the car body carries a visible scrape, or CLEAN if "
    "it does not. Answer with that word and nothing else."
)


# The free tier caps requests per minute rather than per day, so a batch that
# fires as fast as the network allows will trip it. These two numbers pace it.
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 8


def load_font(size):
    """Return a truetype face when the system has one, else the bundled default."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(index):
    """Draw frame `index` of the clip. The car moves right and is scraped at IMPACT_FRAME."""
    img = Image.new("RGB", (WIDTH, HEIGHT), (176, 194, 214))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 170, WIDTH, HEIGHT], fill=(96, 100, 106))
    for mark in range(-40, WIDTH + 40, 80):
        x = mark + (index * 4) % 80
        draw.rectangle([x, 214, x + 40, 220], fill=(226, 226, 226))

    # The barrier the car passes, fixed in the frame.
    draw.rectangle([352, 96, 372, 200], fill=(198, 176, 60))
    draw.rectangle([352, 96, 372, 118], fill=(40, 40, 40))

    # The car, entering from the left and crossing the barrier partway through.
    x = int(28 + index * 2.6)
    body = [x, 126, x + 150, 186]
    draw.rectangle(body, fill=(52, 92, 156))
    draw.polygon(
        [(x + 32, 126), (x + 62, 98), (x + 118, 98), (x + 134, 126)], fill=(80, 124, 190)
    )
    for wheel_x in (x + 26, x + 112):
        draw.ellipse([wheel_x, 172, wheel_x + 30, 200], fill=(26, 26, 28))

    # The scrape appears at the moment of impact and stays for the rest of the clip.
    if index >= IMPACT_FRAME:
        draw.line([x + 96, 150, x + 146, 162], fill=(232, 232, 236), width=5)
        draw.line([x + 104, 162, x + 144, 170], fill=(214, 214, 220), width=4)

    # The brake lamp, on for a tenth of a second and off again.
    if FLASH_FRAME <= index < FLASH_FRAME + FLASH_LENGTH:
        draw.rectangle([x, 140, x + 10, 156], fill=(236, 48, 40))

    draw.text((10, 8), f"t={index / FPS:5.2f}s", font=load_font(16), fill=(20, 20, 20))
    return img


def write_video(path):
    """Encode every frame into an mp4 and return how many frames were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit("no mp4 encoder available to OpenCV on this machine")
    for index in range(TOTAL_FRAMES):
        frame = cv2.cvtColor(np.asarray(render_frame(index)), cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    return TOTAL_FRAMES


def sample_keyframes(path, stride):
    """Read the encoded video back and return (index, jpeg bytes) for every stride-th frame.

    The frames are read out of the file rather than kept from the renderer, so
    everything downstream works from the video the way it would from any other.
    """
    capture = cv2.VideoCapture(str(path))
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                frames.append((index, buffer.tobytes()))
        index += 1
    capture.release()
    return frames



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


def ask_frame(client, jpeg):
    """Send one frame and return its one-word verdict."""
    encoded = base64.b64encode(jpeg).decode()
    response = call_with_retry(
        client,
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": QUESTION},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            }
        ],
    )
    reply = (response.choices[0].message.content or "").strip().upper()
    return "DAMAGED" if "DAMAG" in reply else "CLEAN"


def locate_transition(timeline):
    """Return the first sampled frame answered DAMAGED, or None if none was."""
    for index, verdict in timeline:
        if verdict == "DAMAGED":
            return index
    return None


def main():
    print("=" * 78)
    print("--- 1. A clip whose events are known because they were scheduled here ---")
    print(f"{TOTAL_FRAMES} frames at {FPS} fps, {WIDTH}x{HEIGHT}, {DURATION_SECONDS} seconds")
    print(f"  scrape appears at frame {IMPACT_FRAME} (t={IMPACT_FRAME / FPS:.2f}s) "
          f"and stays for the rest of the clip")
    print(f"  brake lamp is on for frames {FLASH_FRAME}-{FLASH_FRAME + FLASH_LENGTH - 1} "
          f"(t={FLASH_FRAME / FPS:.2f}s), {FLASH_LENGTH / FPS * 1000:.0f} ms in all")

    print()
    print("--- 2. Encoding it ---")
    written = write_video(VIDEO_PATH)
    print(f"wrote {VIDEO_PATH.name}, {written} frames, "
          f"{VIDEO_PATH.stat().st_size / 1024:.0f} KB")
    print("  nothing after this point uses the renderer; the frames come back out of the file")

    print()
    print("--- 3. Sampling keyframes ---")
    frames = sample_keyframes(VIDEO_PATH, SAMPLE_STRIDE)
    indices = [index for index, _ in frames]
    print(f"stride {SAMPLE_STRIDE}: {len(frames)} frames of {TOTAL_FRAMES} "
          f"({len(frames) / TOTAL_FRAMES:.0%}), one every {SAMPLE_STRIDE / FPS:.2f}s")
    print(f"  sampled frames: {indices}")
    reachable = [i for i in indices if FLASH_FRAME <= i < FLASH_FRAME + FLASH_LENGTH]
    print(f"  frames landing inside the brake lamp window: {reachable or 'none'}")
    print(f"  an event shorter than the stride is not hard to see, it is not sampled")

    print()
    print("--- 4. One question per sampled frame ---")
    client = build_client()
    print(f"model: {MODEL}")
    timeline = []
    for index, jpeg in frames:
        verdict = ask_frame(client, jpeg)
        timeline.append((index, verdict))
    print(f"{len(timeline)} calls, one per frame, each one blind to the others")

    print()
    print("--- 5. The timeline, and the event read off it ---")
    for index, verdict in timeline:
        truth = "DAMAGED" if index >= IMPACT_FRAME else "CLEAN"
        flag = "" if verdict == truth else "   <- disagrees with the frame as drawn"
        print(f"  frame {index:>4}  t={index / FPS:5.2f}s  {verdict:<8}{flag}")
    agree = sum(
        1 for index, verdict in timeline
        if verdict == ("DAMAGED" if index >= IMPACT_FRAME else "CLEAN")
    )
    print(f"per-frame agreement {agree}/{len(timeline)}")
    found = locate_transition(timeline)
    if found is None:
        print("  no sampled frame came back DAMAGED, so the timeline holds no event")
    else:
        error = (found - IMPACT_FRAME) / FPS
        print(f"  first DAMAGED frame: {found} (t={found / FPS:.2f}s) against the scrape "
              f"at frame {IMPACT_FRAME} (t={IMPACT_FRAME / FPS:.2f}s)")
        print(f"  the estimate is late by {error:.2f}s, and it can only ever be late, "
              f"because the change is invisible until the next sample")

    print()
    print("--- 6. The same answers re-read at coarser strides ---")
    answered = dict(timeline)
    print(f"  {'stride':>7}{'calls':>7}{'window':>9}{'estimate':>11}{'error':>9}")
    for stride in COARSER_STRIDES:
        subset = [(i, answered[i]) for i in sorted(answered) if i % stride == 0]
        found = locate_transition(subset)
        estimate = f"{found / FPS:.2f}s" if found is not None else "not seen"
        error = f"{(found - IMPACT_FRAME) / FPS:.2f}s" if found is not None else "-"
        print(f"  {stride:>7}{len(subset):>7}{stride / FPS:>8.2f}s{estimate:>11}{error:>9}")
    print(f"  the window is what a stride guarantees; the error in any one run is wherever "
          f"the samples happened to fall inside it")
    print(f"  stride {COARSER_STRIDES[-1]} lands closer here than stride {COARSER_STRIDES[-2]} "
          f"does, on a third of the calls - which is luck, not a reason to sample less")
    print()
    print("this reads each frame on its own and stitches the answers together afterwards; "
          "a model built for video sees the frames together, which is how motion, order and "
          "duration become answerable at all")
    print("=" * 78)


if __name__ == "__main__":
    main()
