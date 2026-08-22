"""Split one document five different ways and compare what each strategy produces.

Demonstrates the trade-offs behind chunking, the step that decides retrieval quality:
    1. Fixed-length splitting that backs off to a sentence boundary.
    2. Semantic splitting on sentence units, no overlap.
    3. LLM-driven splitting that picks its own break points.
    4. Hierarchical splitting that follows heading structure.
    5. Sliding window splitting with deliberate overlap.
    6. Score every strategy side by side on the same text.

Module 02: RAG - Chunking Strategies.
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

CHUNK_SIZE = 800
OVERLAP = 150
# How far the fixed-length splitter may rewind to land on a sentence end.
SENTENCE_LOOKBACK = 200
TERMINATORS = ".!?"

SAMPLE_PARAGRAPHS = [p.strip() for p in """The park sells several ticket types to suit different visitors. A one-day ticket is the most basic option: the date is chosen at purchase, and the price moves with the season. A two-day ticket must be used on two consecutive days and costs about ten percent less than two separate one-day tickets. Festival tickets cover selected event periods, so the validity window printed on the ticket matters more than usual.

Tickets are sold mainly through official channels: the park website, the mobile app, and the in-app store. Authorised resellers also sell them, but only listings carrying the official partner badge are genuine. Every electronic ticket is bound to an identity document. Residents use a national ID card, overseas visitors use a passport, and a child ticket additionally requires a birth certificate or an equivalent proof of age.

Discounts have to be registered before the visit. A birthday visitor who registers through an official channel receives a badge and a dessert voucher. Holders of a marriage certificate issued within the last six months can buy a special package that includes dinner for two at the banquet hall. Serving and retired military personnel get twenty percent off on presentation of a valid card, but the request has to be filed at least three days in advance.""".split("\n\n")]

SAMPLE_TEXT = "\n\n".join(SAMPLE_PARAGRAPHS)

# The same three paragraphs under headings. Keeping the prose identical is what
# makes the final table a fair comparison: only the headings differ, so a row
# measured on this variant can still be read against the rows above it.
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
    """Return (api_key, base_url, model) for whichever key is configured.

    Only chat completion is needed here, so DeepSeek comes first; Gemini and
    OpenAI follow, which means a single key of any kind is enough to run step 3.
    """
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
    """Strategy 1: cut at chunk_size, but rewind to the nearest sentence end.

    Plain fixed-length slicing severs sentences mid-word. Scanning backwards for
    punctuation keeps chunks readable while staying near the target size; the
    overlap carries a little context across the seam.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for i in range(end, max(start, end - SENTENCE_LOOKBACK), -1):
                if text[i] in TERMINATORS:
                    end = i + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Once the window reaches the end there is nothing left to overlap into;
        # without this the loop crawls forward one character at a time and emits
        # a long tail of single-character chunks.
        if end >= len(text):
            break
        # Step back by the overlap, but never far enough to stall or move backwards.
        start = max(end - overlap, start + 1)
    return chunks


def split_sentences(text):
    """Split into sentences while keeping the terminating punctuation.

    A plain re.split on the terminators drops them, which silently strips every
    full stop from the output; matching a sentence body together with its
    terminator keeps the chunks quotable.
    """
    pattern = rf"[^{re.escape(TERMINATORS)}\n]+[{re.escape(TERMINATORS)}]*"
    return [s.strip() for s in re.findall(pattern, text) if s.strip()]


def semantic_chunks(text, max_size=CHUNK_SIZE):
    """Strategy 2: group whole sentences, never splitting one.

    Every chunk is a complete thought, which is what makes retrieval accurate.
    The cost is uneven length: a trailing sentence can end up alone in a tiny chunk.
    """
    chunks = []
    current = ""
    for sentence in split_sentences(text):
        if current and len(current) + len(sentence) > max_size:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def llm_chunks(text, max_size=CHUNK_SIZE):
    """Strategy 3: hand the job to an LLM and let it choose the break points.

    Produces the most even, most semantically clean chunks, but costs an API call
    per document. Falls back to the semantic splitter when no key is configured or
    the model returns something unparseable, so the comparison can still run.
    """
    from openai import OpenAI

    provider = pick_provider()
    if not provider:
        print("  (no API key configured, falling back to semantic splitting)")
        return semantic_chunks(text, max_size)
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
        # Models add ```json fences even when told not to; strip them before parsing.
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        chunks = json.loads(raw).get("chunks", [])
        if chunks:
            return chunks
        print("  (empty chunk list, falling back to semantic splitting)")
    except Exception as exc:
        print(f"  (LLM split failed: {exc}, falling back to semantic splitting)")
    return semantic_chunks(text, max_size)


def hierarchical_chunks(text, target_size=CHUNK_SIZE):
    """Strategy 4: start a new chunk whenever a heading appears.

    Ideal for manuals and specs, where a section is the natural unit of meaning.
    The weakness shows up as size control: a lone heading becomes its own tiny chunk.
    """
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
    """Strategy 5: slide a fixed window forward in fixed steps.

    Because step < window, consecutive chunks share text, so a fact sitting on a
    boundary still appears whole somewhere. The price is duplicated content in the
    index, and a stub chunk at the tail.
    """
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + window].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def describe(name, produce, preview=60):
    """Print size statistics plus a short preview of every chunk.

    The producer is a callable rather than a ready list because Python evaluates
    arguments before the call: passing chunks directly would run the strategy
    first, so any message it prints while working would land under the previous
    strategy's heading instead of its own.
    """
    print(f"\n--- {name} ---")
    chunks = produce()
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
    """Step 6: put every strategy on one line so the trade-offs are visible."""
    print("\n=== Side-by-side comparison ===")
    print(f"  {'strategy':<22}{'chunks':>8}{'avg':>8}{'min':>8}{'max':>8}{'spread':>9}")
    for s in (s for s in all_stats if s):
        print(f"  {s['name']:<22}{s['count']:>8}{s['avg']:>8.0f}"
              f"{s['min']:>8}{s['max']:>8}{s['max'] - s['min']:>9}")
    print("\n  Lower spread means more even chunks. Even chunks embed more")
    print("  predictably, which is why LLM splitting usually wins on quality")
    print("  and loses on cost.")


def main():
    print(f"Source text: {len(SAMPLE_TEXT)} chars "
          f"({len(STRUCTURED_TEXT)} with headings), target chunk size {CHUNK_SIZE}")

    stats = [
        describe("1. fixed length", lambda: fixed_length_chunks(SAMPLE_TEXT)),
        describe("2. semantic", lambda: semantic_chunks(SAMPLE_TEXT)),
        describe("3. llm", lambda: llm_chunks(SAMPLE_TEXT)),
        # Hierarchical needs headings, so it gets the structured variant.
        describe("4. hierarchical", lambda: hierarchical_chunks(STRUCTURED_TEXT)),
        describe("5. sliding window", lambda: sliding_window_chunks(SAMPLE_TEXT)),
    ]
    summarise(stats)


if __name__ == "__main__":
    main()
