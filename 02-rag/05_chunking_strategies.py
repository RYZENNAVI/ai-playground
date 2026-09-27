"""This script tries five chunking strategies on one short document and compares
the chunk sizes. The document is a theme park ticket guide in three paragraphs:
ticket types, buying a ticket, and discounts. It is about 1,300 characters long,
and the target chunk size is 800. Only sizes are measured. The script does not test
retrieval.

The run prints six parts:
    1. Fixed length. The guide cut every 800 characters and moved back to the last
       sentence end. The next chunk starts 150 characters earlier, so it can begin
       in the middle of a sentence.
    2. Sentence packing. Whole sentences packed into chunks of up to 800
       characters. No sentence is cut, but paragraph breaks are ignored.
    3. LLM. A chat model chooses the break points, and the run counts how many of
       its chunks appear word for word in the guide. Without an API key, or when
       the call fails, part 2 runs in its place and the row is marked.
    4. Hierarchical. The same paragraphs under headings, with a new chunk at every
       heading. It only splits at headings: no chunk keeps its parent heading, and
       the title, followed directly by a subheading, becomes a chunk of its own.
    5. Sliding window. An 800-character window moved 450 characters at a time, so
       neighbouring chunks share 350 characters. Chunks start and end mid-word.
    6. Side by side. Count, average, minimum, maximum and spread (maximum minus
       minimum) for every strategy.
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

CHUNK_SIZE = 800
OVERLAP = 150
# How far the fixed-length splitter may rewind to land on a sentence end.
SENTENCE_LOOKBACK = 200
TERMINATORS = ".!?"

SAMPLE_PARAGRAPHS = [p.strip() for p in """The park sells several ticket types to suit different visitors. A one-day ticket is the most basic option: the date is chosen at purchase, and the price moves with the season. A two-day ticket must be used on two consecutive days and costs about ten percent less than two separate one-day tickets. Festival tickets cover selected event periods, so the validity window printed on the ticket matters more than usual.

Tickets are sold mainly through official channels: the park website, the mobile app, and the in-app store. Authorised resellers also sell them, but only listings carrying the official partner badge are genuine. Every electronic ticket is bound to an identity document. Residents use a national ID card, overseas visitors use a passport, and a child ticket additionally requires a birth certificate or an equivalent proof of age.

Discounts have to be registered before the visit. A birthday visitor who registers through an official channel receives a badge and a dessert voucher. Holders of a marriage certificate issued within the last six months can buy a special package that includes dinner for two at the banquet hall. Serving and retired military personnel get twenty percent off on presentation of a valid card, but the request has to be filed at least three days in advance.""".split("\n\n")]

SAMPLE_TEXT = "\n\n".join(SAMPLE_PARAGRAPHS)

# The same three paragraphs under headings, so only the headings differ from
# SAMPLE_TEXT.
STRUCTURED_TEXT = "\n\n".join([
    "# Ticket Guide",
    "## Ticket Types",
    SAMPLE_PARAGRAPHS[0],
    "## Buying a Ticket",
    SAMPLE_PARAGRAPHS[1],
    "## Discounts",
    SAMPLE_PARAGRAPHS[2],
])


def pick_provider():
    """Return (api_key, base_url, model) for the first key found, DeepSeek first.
    Any one key is enough for part 3."""
    if os.getenv("DEEPSEEK_API_KEY"):
        return (os.getenv("DEEPSEEK_API_KEY"),
                "https://api.deepseek.com", "deepseek-chat")
    if os.getenv("GEMINI_API_KEY"):
        return (os.getenv("GEMINI_API_KEY"),
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "gemini-3.1-flash-lite")
    if os.getenv("OPENAI_API_KEY"):
        return (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"),
                "gpt-4o-mini")
    return None


def fixed_length_chunks(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Strategy 1: cut at chunk_size, then move back to a sentence end within
    SENTENCE_LOOKBACK. The next chunk starts overlap earlier, often mid-sentence."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for i in range(end - 1, max(start, end - 1 - SENTENCE_LOOKBACK), -1):
                if text[i] in TERMINATORS:
                    end = i + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Stop at the end of the text. Otherwise the overlap keeps pulling start
        # back, and the loop emits 150 more chunks, each one character shorter.
        if end >= len(text):
            break
        # Step back by the overlap, but never far enough to stall or move backwards.
        start = max(end - overlap, start + 1)
    return chunks


def split_sentences(text):
    """Split into sentences and keep each terminator.
    re.split would drop every full stop."""
    pattern = rf"[^{re.escape(TERMINATORS)}\n]+[{re.escape(TERMINATORS)}]*"
    return [s.strip() for s in re.findall(pattern, text) if s.strip()]


def semantic_chunks(text, max_size=CHUNK_SIZE):
    """Strategy 2: pack whole sentences into chunks of up to max_size characters.
    Line breaks are dropped, so a chunk can span two paragraphs."""
    chunks = []
    current = ""
    for sentence in split_sentences(text):
        # The + 1 counts the space that joins the sentence to the chunk.
        if current and len(current) + 1 + len(sentence) > max_size:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def llm_chunks(text, max_size=CHUNK_SIZE):
    """Strategy 3: let a chat model choose the break points, one API call per text.
    Returns None without a key or when the call fails, so the caller can fall back."""
    from openai import OpenAI

    provider = pick_provider()
    if not provider:
        print("  (no API key configured, falling back to sentence packing)")
        return None
    api_key, base_url, model = provider
    print(f"  (splitting with {model})")

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = (
        f"Split the text below into chunks of at most {max_size} characters.\n"
        "Rules: keep each chunk semantically complete, break at natural "
        'boundaries, and reply with JSON shaped as {"chunks": ["...", "..."]}.\n\n'
        f"Text:\n{text}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You split text and reply with strict "
                                              "JSON only, no markdown fences."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # Some models wrap the JSON in fences even when told not to.
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        chunks = json.loads(raw).get("chunks", [])
        if chunks:
            # A model can reword the text while splitting it.
            verbatim = sum(chunk in text for chunk in chunks)
            print(f"  {verbatim} of {len(chunks)} chunks appear word for word "
                  "in the text")
            return chunks
        print("  (empty chunk list, falling back to sentence packing)")
    except Exception as exc:
        print(f"  (LLM split failed: {exc}, falling back to sentence packing)")
    return None


def hierarchical_chunks(text, target_size=CHUNK_SIZE):
    """Strategy 4: start a new chunk at every heading or past target_size. All
    heading levels are treated alike, so no chunk keeps its parent heading."""
    heading = ("# ", "## ", "### ")
    chunks = []
    current = ""
    for line in (ln.strip() for ln in text.split("\n")):
        if not line:
            continue
        starts_section = line.startswith(heading)
        too_long = len(current) + len(line) > target_size and current.strip()
        if (starts_section or too_long) and current.strip():
            chunks.append(current.strip())
            current = ""
        current = f"{current}\n{line}" if current else line
    if current.strip():
        chunks.append(current.strip())
    return chunks


def sliding_window_chunks(text, window=CHUNK_SIZE, step=OVERLAP * 3):
    """Strategy 5: move a fixed window forward by step characters. Neighbouring
    chunks share window minus step characters, so the index holds repeated text."""
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + window].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def describe(name, produce, fallback=None, preview=60):
    """Print size statistics and a preview of every chunk. Takes a callable so a
    strategy's own messages print under its heading."""
    print(f"\n--- {name} ---")
    chunks = produce()
    if chunks is None and fallback:
        chunks = fallback()
        name = f"{name} (fell back)"
    if not chunks:
        print("  no chunks produced")
        return None

    sizes = [len(c) for c in chunks]
    stats = {
        "name": name,
        "count": len(chunks),
        "avg": sum(sizes) / len(sizes),
        "min": min(sizes),
        "max": max(sizes),
    }
    print(f"  chunks={stats['count']}  avg={stats['avg']:.0f}  "
          f"min={stats['min']}  max={stats['max']}  spread={stats['max'] - stats['min']}")
    for i, chunk in enumerate(chunks, 1):
        flat = chunk.replace("\n", " ")
        tail = "..." if len(flat) > preview else ""
        print(f"    [{i}] {len(chunk):4} chars | {flat[:preview]}{tail}")
    return stats


def summarise(all_stats):
    """Part 6: one row per strategy."""
    print("\n--- 6. Side by side ---")
    print(f"  {'strategy':<22}{'chunks':>8}{'avg':>8}{'min':>8}{'max':>8}{'spread':>9}")
    for s in (s for s in all_stats if s):
        print(f"  {s['name']:<22}{s['count']:>8}{s['avg']:>8.0f}"
              f"{s['min']:>8}{s['max']:>8}{s['max'] - s['min']:>9}")
    print("\n  Spread is max minus min, so lower means more even sizes.")
    print("  Size is all this script measures. It does not test retrieval.")


def main():
    print(f"Source text: {len(SAMPLE_TEXT)} chars "
          f"({len(STRUCTURED_TEXT)} with headings), target chunk size {CHUNK_SIZE}")

    stats = [
        describe("1. Fixed length", lambda: fixed_length_chunks(SAMPLE_TEXT)),
        describe("2. Sentence packing", lambda: semantic_chunks(SAMPLE_TEXT)),
        describe("3. LLM", lambda: llm_chunks(SAMPLE_TEXT),
                 fallback=lambda: semantic_chunks(SAMPLE_TEXT)),
        # Hierarchical needs headings, so it gets the structured variant.
        describe("4. Hierarchical", lambda: hierarchical_chunks(STRUCTURED_TEXT)),
        describe("5. Sliding window", lambda: sliding_window_chunks(SAMPLE_TEXT)),
    ]

    # 6. Side by side
    summarise(stats)


if __name__ == "__main__":
    main()
