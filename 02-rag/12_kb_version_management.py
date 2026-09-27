"""This script versions a knowledge base and benchmarks retrieval on each version,
with a regression check before release. The base describes an invented theme park.
Version 1 has 3 entries. Version 2 adds 2 entries and extends the other 3. Both are
embedded with gemini-embedding-001 at 1024 dimensions into FAISS inner-product
indexes. Five test questions each name a string the answer must contain, and a
question counts as answered when that string appears in the top 3 entries. Version
1 has only 3 entries, so for it the top 3 is the whole base and its score says only
whether the answer exists.

The run prints six parts:
    1. Version fingerprints. A hash, the entry count and the length of each version.
    2. What changed. Added, removed and modified entries, found without a model.
    3. Indexing both versions. The embeddings are rescaled to unit length first,
       because at 1024 dimensions they come back shorter.
    4. Scoring both versions. Each question against each index, with the rank of
       the first entry that holds the answer.
    5. What the difference measures. Which questions each version gained or lost,
       and whether version 1 held the answer at all.
    6. Regression check. Whether every question version 1 answered still passes.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

EMBED_MODEL = "gemini-embedding-001"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBED_DIM = 1024
TOP_K = 3

VERSION_1 = [
    {"id": "kb_001", "text": "Riverbend Park is on the east bank of the river and "
                             "opened on 16 June 2016."},
    {"id": "kb_002", "text": "A weekday adult ticket is 399; weekends and public "
                             "holidays are 499."},
    {"id": "kb_003", "text": "The park opens at 08:00 and closes at 20:00."},
]

VERSION_2 = [
    {"id": "kb_001", "text": "Riverbend Park is on the east bank of the river and "
                             "opened on 16 June 2016. It covers 390 hectares across "
                             "seven themed zones."},
    {"id": "kb_002", "text": "A weekday adult ticket is 399; weekends and public "
                             "holidays are 499. Children between 1.0 and 1.4 metres "
                             "pay 299 on weekdays and 374 at weekends. Under 1.0 "
                             "metres is free."},
    {"id": "kb_003", "text": "The park opens at 08:00 and closes at 20:00, every day "
                             "of the year. Check the app before travelling."},
    {"id": "kb_004", "text": "Metro line 11 stops at the park station, and a shuttle "
                             "bus runs from the central terminal."},
    {"id": "kb_005", "text": "The headline rides are the launch coaster in Tomorrow "
                             "Quarter, the mine train in Dream Valley and the indoor "
                             "boat ride in Treasure Cove."},
]

# Two of these five have no answer anywhere in version 1, on purpose. Version 1 has
# only TOP_K entries, so it returns all of them for every question: its score can
# only show whether the answer exists. Part 5 checks that this is all the gap measures.
TEST_CASES = [
    {"query": "Where is the park?", "expect": "east bank"},
    {"query": "How much is an adult ticket on a Tuesday?", "expect": "399"},
    {"query": "What time does it close?", "expect": "20:00"},
    {"query": "How do I get there by public transport?", "expect": "line 11"},
    {"query": "Which rides should I not miss?", "expect": "launch coaster"},
]


def gemini():
    """Return a client for Gemini, which supplies the embedding model used here."""
    from openai import OpenAI

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set. Add it to .env and retry.")
    return OpenAI(api_key=key, base_url=GEMINI_BASE)


def embed(api, texts):
    """Return the texts' embeddings rescaled to unit length, and their mean length before.
    Below full width this model's vectors are shorter than 1, so the inner product needs it."""
    import numpy as np

    response = api.embeddings.create(model=EMBED_MODEL, input=texts,
                                     dimensions=EMBED_DIM)
    matrix = np.array([item.embedding for item in response.data], dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-12), float(norms.mean())


def version_stats(entries):
    """Part 1: hash and measure one version. Sorting first makes the hash depend on
    the content only, so one edited character changes it and nothing else does."""
    payload = json.dumps(sorted((e["id"], e["text"]) for e in entries),
                         ensure_ascii=False)
    lengths = [len(e["text"]) for e in entries]
    return {
        "entries": len(entries),
        "hash": hashlib.md5(payload.encode("utf-8")).hexdigest()[:12],
        "mean_chars": sum(lengths) / len(lengths) if lengths else 0,
        "total_chars": sum(lengths),
    }


def diff_versions(old, new):
    """Part 2: added and removed ids by set operations, modified ones by exact comparison.
    No model is needed, because whether a text changed has an exact answer."""
    old_map = {e["id"]: e["text"] for e in old}
    new_map = {e["id"]: e["text"] for e in new}
    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    modified = sorted(i for i in set(old_map) & set(new_map)
                      if old_map[i] != new_map[i])
    return added, removed, modified, old_map, new_map


def build_index(api, entries):
    """Part 3: embed one version and put it in a FAISS index."""
    import faiss

    vectors, mean_norm = embed(api, [e["text"] for e in entries])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, entries, mean_norm


def evaluate(api, index, entries, cases, k=TOP_K):
    """Part 4: retrieve the top k for every case and find the first entry holding the answer.
    A substring test only shows the answer was retrieved, not that any reply would be right."""
    import numpy as np

    query_vectors, _ = embed(api, [c["query"] for c in cases])

    # One throwaway search first. The first call into the library pays a one-off
    # setup cost, and timing it made the smaller index look a hundred times slower.
    index.search(np.array([query_vectors[0]]), min(k, len(entries)))

    results, elapsed = [], []
    for case, vector in zip(cases, query_vectors):
        started = time.perf_counter()
        _, indexes = index.search(np.array([vector]), min(k, len(entries)))
        elapsed.append((time.perf_counter() - started) * 1000)
        retrieved = [entries[i] for i in indexes[0] if i >= 0]
        rank = next((position for position, entry in enumerate(retrieved, 1)
                     if case["expect"].lower() in entry["text"].lower()), None)
        results.append({"query": case["query"], "expect": case["expect"],
                        "hit": rank is not None, "rank": rank})
    accuracy = sum(r["hit"] for r in results) / len(results)
    return results, accuracy, sum(elapsed) / len(elapsed)


def short(text, width=62):
    """Trim text to one printable line."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[:width - 1] + "…"


def cell(result):
    """Show a result as 'rank 1' or 'MISS'."""
    return f"rank {result['rank']}" if result["hit"] else "MISS"


def main():
    api = gemini()

    # 1. Version fingerprints

    print("=" * 92)
    print("--- 1. Version fingerprints ---")
    stats = {"v1.0": version_stats(VERSION_1), "v2.0": version_stats(VERSION_2)}
    print(f"  {'version':<8} {'entries':>8} {'mean chars':>11} {'total':>8}  hash")
    for name, s in stats.items():
        print(f"  {name:<8} {s['entries']:>8} {s['mean_chars']:>11.0f} "
              f"{s['total_chars']:>8}  {s['hash']}")

    # 2. What changed

    print("\n--- 2. What changed ---")
    added, removed, modified, old_map, new_map = diff_versions(VERSION_1, VERSION_2)
    print(f"  added {len(added)}, removed {len(removed)}, modified {len(modified)}")
    for entry_id in added:
        print(f"    + {entry_id}: {short(new_map[entry_id])}")
    for entry_id in removed:
        print(f"    - {entry_id}: {short(old_map[entry_id])}")
    for entry_id in modified:
        grew = len(new_map[entry_id]) - len(old_map[entry_id])
        print(f"    ~ {entry_id}: {grew:+d} chars")
    print("\n  No model was called for this. Whether two strings differ has an exact")
    print("  answer, and an exact answer is cheaper, faster and identical on every run.")

    # 3. Indexing both versions

    print("\n--- 3. Indexing both versions ---")
    index_1, entries_1, norm_1 = build_index(api, VERSION_1)
    index_2, entries_2, norm_2 = build_index(api, VERSION_2)
    print(f"  v1.0: {index_1.ntotal} vectors, mean raw norm before scaling {norm_1:.3f}")
    print(f"  v2.0: {index_2.ntotal} vectors, mean raw norm before scaling {norm_2:.3f}")
    if abs(norm_1 - 1.0) > 0.01:
        print(f"  Truncated to {EMBED_DIM} dimensions these are not unit vectors, so they")
        print("  are rescaled before they go into the index.")

    # 4. Scoring both versions

    print(f"\n--- 4. Scoring both versions, top {TOP_K} ---")
    results_1, accuracy_1, ms_1 = evaluate(api, index_1, entries_1, TEST_CASES)
    results_2, accuracy_2, ms_2 = evaluate(api, index_2, entries_2, TEST_CASES)
    print(f"  {'query':<44} {'v1.0':>8} {'v2.0':>8}")
    print("  " + "-" * 62)
    for r1, r2 in zip(results_1, results_2):
        print(f"  {short(r1['query'], 42):<44} {cell(r1):>8} {cell(r2):>8}")
    print("  " + "-" * 62)
    print(f"  {'accuracy':<44} {accuracy_1:>8.0%} {accuracy_2:>8.0%}")
    print("\n  The rank is where the first entry holding the answer came.")
    print(f"  Version 1 returns all {len(VERSION_1)} of its entries for every question, "
          "so a MISS there")
    print("  means the answer is missing, not that retrieval failed.")
    for name, results in (("version 1", results_1), ("version 2", results_2)):
        for r in results:
            if r["hit"] and r["rank"] > 1:
                print(f"  In {name}, '{short(r['query'], 40)}' counts as answered, "
                      f"but the answer came at rank {r['rank']}.")

    # 5. What the difference measures

    print("\n--- 5. What the difference measures ---")
    gained = [r2 for r1, r2 in zip(results_1, results_2) if r2["hit"] and not r1["hit"]]
    lost = [r2["query"] for r1, r2 in zip(results_1, results_2)
            if r1["hit"] and not r2["hit"]]
    print(f"  accuracy {accuracy_1:.0%} -> {accuracy_2:.0%}")
    absent = 0
    for r2 in gained:
        in_v1 = any(r2["expect"].lower() in text.lower() for text in old_map.values())
        absent += not in_v1
        where = "held by an entry" if in_v1 else "in no entry"
        print(f"    gained: {short(r2['query'], 44):<46} '{r2['expect']}' was {where} "
              "of version 1")
    for query in lost:
        print(f"    lost  : {short(query, 70)}")
    if gained and absent == len(gained) and not lost:
        print("\n  Every gain comes from an answer version 1 did not contain. The")
        print("  benchmark is measuring coverage, not retrieval quality: version 2")
        print("  does not search better, it has more to find.")

    print(f"\n  Mean search time was {ms_1:.3f} ms on {len(VERSION_1)} vectors and "
          f"{ms_2:.3f} ms on {len(VERSION_2)}.")
    print("  An exact search over this few vectors costs the same either way, so any")
    print("  difference here is noise. Report it as no measurable change.")

    # 6. Regression check

    print("\n--- 6. Regression check ---")
    print("  A release needs one question answered: did anything that used to work")
    print("  stop working?")
    # A regression check has one denominator and it is not the test set. Counting
    # against every case folds three different outcomes into one number: cases
    # that passed and still pass, cases that never passed and still do not, and
    # cases the new version fixed. Only the first is what "still pass" claims.
    was_passing = [(r1, r2) for r1, r2 in zip(results_1, results_2) if r1["hit"]]
    regressions = [r1["query"] for r1, r2 in was_passing if not r2["hit"]]
    still_passing = len(was_passing) - len(regressions)

    if was_passing:
        print(f"  {still_passing}/{len(was_passing)} previously passing cases still pass "
              f"({len(was_passing)} of {len(TEST_CASES)} passed on version 1)")
    else:
        print(f"  no case passed on version 1, so there is nothing to regress "
              f"(0 of {len(TEST_CASES)})")
    for query in regressions:
        print(f"    REGRESSION: {short(query, 66)}")
    if gained:
        print(f"  Cases version 2 fixed: {len(gained)}, the gains listed in part 5.")
    if was_passing and not regressions:
        print("  No regressions on this test set.")
    print("\n  'On this test set' is the whole claim. Five cases cannot certify a")
    print("  release; they can only catch the breakages the five cases cover.")


if __name__ == "__main__":
    main()
