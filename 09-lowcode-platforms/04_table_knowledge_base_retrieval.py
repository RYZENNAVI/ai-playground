"""This script builds a table knowledge base from two CSV tables and one prose file.
Each table row becomes one chunk with its column names attached, and the bge-small
sentence encoder embeds every chunk for retrieval by cosine similarity. A question
with three conditions in it is then answered a second way, by structured filtering
on the columns, and the two answers are compared row for row.

The run prints 7 parts:
    1. Two tables and one prose document.
    2. One row, one chunk, headers attached.
    3. Which column can single a row out.
    4. A pricing question, against whole rows and against the family column.
    5. A question with three conditions in it.
    6. The same question, answered by filtering the columns.
    7. A question only the prose document answers, and its cost.
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
MODEL_ID = "BAAI/bge-small-en-v1.5"

TOP_K = 4
# A prose chunk is capped by length; a table row is a chunk because a row is the
# unit a question is asked about. These two numbers only govern the prose file.
CHUNK_CHARS = 420
CHUNK_OVERLAP = 60


def ensure_model(model_id=MODEL_ID):
    """Return a local path for the weights, checking every module's weights/ first
    so an encoder another module downloaded is reused."""
    vendor, _, name = model_id.partition("/")
    for candidate in sorted(MODULE_DIR.parent.glob(
            f"*/weights/models/{vendor}--{name}/snapshots/*")):
        if (candidate / "config.json").exists():
            return str(candidate)
    from modelscope import snapshot_download
    weights = MODULE_DIR / "weights"
    weights.mkdir(exist_ok=True)
    return snapshot_download(model_id, cache_dir=str(weights))


_ENCODER = None


def encode(texts):
    """Embed a list of strings, loading the encoder once per process."""
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ENCODER = SentenceTransformer(ensure_model())
    return _ENCODER.encode(list(texts), normalize_embeddings=True,
                           show_progress_bar=False)


def read_table(name):
    """Read one CSV into a list of dictionaries."""
    with (DATA_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_to_chunk(row):
    """Render one row as text with its headers, because a bare '19.00' means nothing."""
    return "; ".join(f"{key}: {value}" for key, value in row.items())


def chunk_prose(text, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Split prose on length, rewinding to a sentence end only past the halfway mark.
    Both guards keep the cursor moving forward; without them the loop can run forever."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text.rfind(". ", start, end)
            if window > start + size // 2:
                end = window + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def search(query, chunks, vectors, top_k=TOP_K):
    """Return the top_k chunks by cosine similarity, with their scores."""
    scores = encode([query])[0] @ vectors.T
    ranked = np.argsort(-scores)[:top_k]
    return [(chunks[i], float(scores[i])) for i in ranked]


def index_selectivity(rows, column):
    """Return how many rows share each value of one column."""
    counts = {}
    for row in rows:
        counts[row[column]] = counts.get(row[column], 0) + 1
    return counts


def parse_conditions(question, rows):
    """Turn a question into exact conditions over named columns, from the table's values.
    A user id or event type the table lacks is left out silently; a date is kept."""
    conditions = {}
    user_ids = {row["user_id"] for row in rows}
    for user_id in user_ids:
        if user_id.lower() in question.lower():
            conditions["user_id"] = user_id
    date = re.search(r"\d{4}-\d{2}-\d{2}", question)
    if date:
        conditions["date"] = date.group(0)
    event_types = {row["event_type"] for row in rows}
    # Substring tests with no word boundary: 'research' in a question reads as 'Search'.
    for event_type in event_types:
        if event_type.lower() in question.lower():
            conditions["event_type"] = event_type
        elif loosen(event_type) in loosen(question):
            conditions.setdefault("event_type_loose", event_type)
    return conditions


def loosen(text):
    """Lower a string and keep only letters and digits: 'sign in' matches 'Sign-in'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def apply_conditions(rows, conditions):
    """Return the rows that satisfy every parsed condition."""
    selected = rows
    if "user_id" in conditions:
        selected = [r for r in selected if r["user_id"] == conditions["user_id"]]
    if "date" in conditions:
        selected = [r for r in selected if r["event_time"].startswith(conditions["date"])]
    event_type = conditions.get("event_type") or conditions.get("event_type_loose")
    if event_type:
        selected = [r for r in selected if r["event_type"] == event_type]
    return selected


def main():
    plans = read_table("commission_plans.csv")
    events = read_table("user_behavior_event.csv")
    notes = (DATA_DIR / "service_notes.txt").read_text(encoding="utf-8")

    # 1. Two tables and one prose document
    print("--- 1. Two tables and one prose document ---")
    print(f"  commission_plans     {len(plans)} rows x {len(plans[0])} columns  "
          f"{list(plans[0])}")
    print(f"  user_behavior_event  {len(events)} rows x {len(events[0])} columns  "
          f"{list(events[0])}")
    prose_chunks = chunk_prose(notes)
    print(f"  service_notes.txt    {len(notes)} characters -> {len(prose_chunks)} chunks")

    # 2. One row, one chunk, headers attached
    print("\n--- 2. One row, one chunk, headers attached ---")
    plan_chunks = [row_to_chunk(row) for row in plans]
    event_chunks = [row_to_chunk(row) for row in events]
    for chunk in plan_chunks[:2]:
        print(f"  {chunk}")
    print(f"  {len(plan_chunks)} plan chunks and {len(event_chunks)} event chunks, "
          f"none of them split mid-row")
    plan_vectors = np.asarray(encode(plan_chunks))
    event_vectors = np.asarray(encode(event_chunks))
    prose_vectors = np.asarray(encode(prose_chunks))
    print(f"  encoded with {MODEL_ID}, {plan_vectors.shape[1]} dimensions")

    # 3. Which column can single a row out
    print("\n--- 3. Which column can single a row out ---")
    for column in ("family", "plan"):
        counts = index_selectivity(plans, column)
        worst = max(counts.values())
        unique = sum(1 for c in counts.values() if c == 1)
        print(f"  {column:<7} {len(counts)} distinct value(s), "
              f"{unique}/{len(plans)} rows uniquely identified, "
              f"worst case {worst} row(s) share a value")
    print("  the index column has to be both what the customer says out loud and")
    print("  selective enough to leave one row standing")

    # 4. A pricing question, against whole rows and against the family column
    print("\n--- 4. A pricing question, against whole rows and against the family "
          "column ---")
    question = "What does the Momentum plan cost per trade?"
    print(f"  {question!r}")
    for chunk, score in search(question, plan_chunks, plan_vectors):
        print(f"    {score:.3f}  {chunk}")
    family_chunks = [f"family: {row['family']}" for row in plans]
    family_vectors = np.asarray(encode(family_chunks))
    top_family = search(question, family_chunks, family_vectors, top_k=1)[0][0]
    top_family = top_family.split(": ", 1)[1]
    tied = index_selectivity(plans, "family")[top_family]
    print("  the same question against an index built on the family column only:")
    for chunk, score in search(question, family_chunks, family_vectors, top_k=tied):
        print(f"    {score:.3f}  {chunk}")
    print(f"  {tied} rows carry the value '{top_family}', "
          f"so that index cannot separate them")

    # 5. A question with three conditions in it
    print("\n--- 5. A question with three conditions in it ---")
    question = "Did user U-100241 sign in on 2026-05-04?"
    print(f"  {question!r}")
    hits = search(question, event_chunks, event_vectors)
    for chunk, score in hits:
        print(f"    {score:.3f}  {chunk}")
    def fits(chunk):
        return "U-100241" in chunk and "2026-05-04" in chunk and "Sign-in" in chunk

    wanted = [h for h in hits if fits(h[0])]
    print(f"  {len(wanted)} of the {len(hits)} rows returned satisfy all three")
    ranked = search(question, event_chunks, event_vectors, top_k=len(event_chunks))
    for rank, (chunk, score) in enumerate(ranked, 1):
        if fits(chunk):
            print(f"  the one row that does sits at rank {rank} of {len(ranked)} "
                  f"({score:.3f})")
    print("  similarity ranks by resemblance, and every row about signing in")
    print("  resembles this one, whichever user or day it belongs to")

    # 6. The same question, answered by filtering the columns
    print("\n--- 6. The same question, answered by filtering the columns ---")
    conditions = parse_conditions(question, events)
    literal = {k: v for k, v in conditions.items() if k != "event_type_loose"}
    print(f"  conditions matched literally: {literal}")
    literal_rows = apply_conditions(events, literal)
    for row in literal_rows:
        print(f"    {row['event_time']}  {row['event_type']:<15} {row['event_detail']}")
    print(f"  {len(literal_rows)} row(s) satisfy those: the question writes 'sign in'")
    print("  and the column stores 'Sign-in', so the third condition matched nothing")
    print("  and was dropped from the filter without a word")
    print(f"  conditions matched after folding case and punctuation: {conditions}")
    selected = apply_conditions(events, conditions)
    for row in selected:
        print(f"    {row['event_time']}  {row['event_type']:<15} {row['event_detail']}")
    truth = [r for r in events if r["user_id"] == "U-100241"
             and r["event_time"].startswith("2026-05-04") and r["event_type"] == "Sign-in"]
    print(f"  {len(selected)} row(s) satisfy the filter; scanning the table directly")
    verdict = "agree" if selected == truth else "disagree"
    print(f"  finds {len(truth)}, so the filter and the table {verdict}")

    # 7. A question only the prose document answers, and its cost
    print("\n--- 7. A question only the prose document answers, and its cost ---")
    question = "Why does face unlock stop working after changing the password?"
    print(f"  {question!r}")
    for chunk, score in search(question, prose_chunks, prose_vectors, top_k=2):
        print(f"    {score:.3f}  {chunk[:150]}...")
    print("  no column holds this answer, so no filter can be written for it")
    for k in (2, TOP_K, len(prose_chunks)):
        chars = sum(len(c) for c, _ in
                    search(question, prose_chunks, prose_vectors, top_k=k))
        print(f"  top_k={k:<2} recalls {chars} characters, roughly {chars // 4} "
              f"tokens, paid on every question before the model answers")
    if chars > len(notes):
        print(f"  top_k={len(prose_chunks)} is all {len(prose_chunks)} chunks, more "
              f"than the {len(notes)}-character file because chunks overlap")
    print("  the chunks are not all the same length, so the bill rises with the")
    print("  recall count without tracking it exactly")


if __name__ == "__main__":
    main()
