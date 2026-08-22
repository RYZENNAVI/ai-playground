"""Treat a knowledge base as versioned code: diff it, benchmark it, regression-test it.

Demonstrates the release discipline a knowledge base needs once it starts changing:
    1. Stamp each version with a hash and a set of statistics.
    2. Diff two versions with set operations and exact comparison, no model involved.
    3. Embed each version into its own FAISS index.
    4. Retrieve and score each version against the same test set.
    5. Compare the two runs and read what the difference actually measures.
    6. Run the old test cases against the new version as a regression check.

Module 02: RAG - Versioning and Benchmarking.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

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

# Two of these five have no answer anywhere in version 1. That is deliberate, and
# step 5 is about noticing that the accuracy gap measures exactly that and nothing
# more subtle.
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
    """Embed texts and return unit-length vectors.

    Normalising is not optional. This model is only unit-length at its full width;
    ask for fewer dimensions and the vectors come back with varying magnitude, so
    an unnormalised inner product would rank partly by length instead of purely by
    direction. Normalising first makes the inner product a cosine again.
    """
    import numpy as np

    response = api.embeddings.create(model=EMBED_MODEL, input=texts,
                                     dimensions=EMBED_DIM)
    matrix = np.array([item.embedding for item in response.data], dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-12), float(norms.mean())


def version_stats(entries):
    """Step 1: hash and measure one version.

    The hash covers ids and text in a stable order, so two builds of the same
    content produce the same fingerprint and a single edited character produces
    a different one.
    """
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
    """Step 2: what changed between two versions.

    Set operations on the ids give added and removed; the intersection is then
    compared character by character. There is no model in this function and no
    reason for one - "did this text change" has an exact answer, and an exact
    answer is cheaper, faster and reproducible.
    """
    old_map = {e["id"]: e["text"] for e in old}
    new_map = {e["id"]: e["text"] for e in new}
    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    modified = sorted(i for i in set(old_map) & set(new_map)
                      if old_map[i] != new_map[i])
    return added, removed, modified, old_map, new_map


def build_index(api, entries):
    """Step 3: embed one version and put it in a FAISS index."""
    import faiss

    vectors, mean_norm = embed(api, [e["text"] for e in entries])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index, entries, mean_norm


def evaluate(api, index, entries, cases, k=TOP_K):
    """Step 4: retrieve for every case and score it.

    Scoring is a substring test: did the expected string appear anywhere in the
    retrieved text. That is enough to tell retrieval apart from silence, and it
    is all it is enough for - it cannot tell a correct answer from a passage that
    merely contains the right characters, and it says nothing about the wording
    of any answer generated afterwards.
    """
    import numpy as np

    query_vectors, _ = embed(api, [c["query"] for c in cases])

    # One throwaway search first. The very first call into the library pays a
    # one-off setup cost, and timing it alongside the rest made the smaller index
    # look a hundred times slower than the larger one - an artefact, not a result.
    index.search(np.array([query_vectors[0]]), min(k, len(entries)))

    results, elapsed = [], []
    for case, vector in zip(cases, query_vectors):
        started = time.perf_counter()
        scores, indexes = index.search(np.array([vector]), min(k, len(entries)))
        elapsed.append((time.perf_counter() - started) * 1000)
        retrieved = [entries[i] for i in indexes[0] if i >= 0]
        joined = " ".join(e["text"] for e in retrieved)
        results.append({
            "query": case["query"],
            "expect": case["expect"],
            "hit": case["expect"].lower() in joined.lower(),
            "top_id": retrieved[0]["id"] if retrieved else None,
            "top_score": float(scores[0][0]) if len(scores[0]) else 0.0,
        })
    accuracy = sum(r["hit"] for r in results) / len(results)
    return results, accuracy, sum(elapsed) / len(elapsed)


def short(text, width=62):
    """Trim text to one printable line."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[:width - 1] + "…"


def main():
    api = gemini()

    print("=" * 92)
    print("--- 1. Version fingerprints ---")
    stats = {"v1.0": version_stats(VERSION_1), "v2.0": version_stats(VERSION_2)}
    print(f"  {'version':<8} {'entries':>8} {'mean chars':>11} {'total':>8}  hash")
    for name, s in stats.items():
        print(f"  {name:<8} {s['entries']:>8} {s['mean_chars']:>11.0f} "
              f"{s['total_chars']:>8}  {s['hash']}")

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

    print("\n--- 3. Indexing both versions ---")
    index_1, entries_1, norm_1 = build_index(api, VERSION_1)
    index_2, entries_2, norm_2 = build_index(api, VERSION_2)
    print(f"  v1.0: {index_1.ntotal} vectors, mean raw norm before scaling {norm_1:.3f}")
    print(f"  v2.0: {index_2.ntotal} vectors, mean raw norm before scaling {norm_2:.3f}")
    if abs(norm_1 - 1.0) > 0.01:
        print(f"  Truncated to {EMBED_DIM} dimensions these are not unit vectors, which")
        print("  is why they are normalised before indexing rather than after.")

    print("\n--- 4. Scoring both against the same cases ---")
    results_1, accuracy_1, ms_1 = evaluate(api, index_1, entries_1, TEST_CASES)
    results_2, accuracy_2, ms_2 = evaluate(api, index_2, entries_2, TEST_CASES)
    print(f"  {'query':<44} {'v1.0':>6} {'v2.0':>6}")
    print("  " + "-" * 60)
    for r1, r2 in zip(results_1, results_2):
        print(f"  {short(r1['query'], 42):<44} {'ok' if r1['hit'] else 'MISS':>6} "
              f"{'ok' if r2['hit'] else 'MISS':>6}")
    print("  " + "-" * 60)
    print(f"  {'accuracy':<44} {accuracy_1:>5.0%} {accuracy_2:>6.0%}")
    print(f"  {'mean search time (ms)':<44} {ms_1:>6.3f} {ms_2:>6.3f}")

    print("\n--- 5. What the difference measures ---")
    gained = [r2["query"] for r1, r2 in zip(results_1, results_2)
              if r2["hit"] and not r1["hit"]]
    lost = [r2["query"] for r1, r2 in zip(results_1, results_2)
            if r1["hit"] and not r2["hit"]]
    print(f"  accuracy {accuracy_1:.0%} -> {accuracy_2:.0%}")
    for query in gained:
        print(f"    gained: {short(query, 70)}")
    for query in lost:
        print(f"    lost  : {short(query, 70)}")
    if gained and not lost:
        print("\n  Every gain here comes from an entry that version 1 simply did not")
        print("  contain. The benchmark is measuring coverage, not retrieval quality:")
        print("  version 2 does not search better, it has more to find. A version")
        print("  comparison will usually be measuring this, so it is worth saying out")
        print("  loud before anyone reads the number as a search improvement.")

    delta_ms = ms_2 - ms_1
    print(f"\n  search time moved by {delta_ms:+.3f} ms going from {len(VERSION_1)} to "
          f"{len(VERSION_2)} vectors.")
    print("  At this scale that figure is measurement noise, not a trend - an exact")
    print("  search over five vectors and over three costs effectively the same. Any")
    print("  reported speed difference this small should be quoted as 'no measurable")
    print("  change' rather than as a number.")

    print("\n--- 6. Regression check ---")
    print("  The question a release needs answered is not 'is the new version better'")
    print("  but 'did anything that used to work stop working'.")
    # A regression check has one denominator and it is not the test set. Counting
    # against every case folds three different outcomes into one number: cases
    # that passed and still pass, cases that never passed and still do not, and
    # cases the new version fixed. Only the first is what "still pass" claims.
    was_passing = [(r1, r2) for r1, r2 in zip(results_1, results_2) if r1["hit"]]
    regressions = [r1["query"] for r1, r2 in was_passing if not r2["hit"]]
    still_passing = len(was_passing) - len(regressions)
    fixed = [r2["query"] for r1, r2 in zip(results_1, results_2)
             if not r1["hit"] and r2["hit"]]

    if was_passing:
        print(f"  {still_passing}/{len(was_passing)} previously passing cases still pass "
              f"({len(was_passing)} of {len(TEST_CASES)} passed on version 1)")
    else:
        print(f"  no case passed on version 1, so there is nothing to regress "
              f"(0 of {len(TEST_CASES)})")
    for query in regressions:
        print(f"    REGRESSION: {short(query, 66)}")
    for query in fixed:
        print(f"    fixed by v2: {short(query, 64)}")
    if was_passing and not regressions:
        print("  no regressions - version 2 is safe to ship on this test set")
    print("\n  'On this test set' is the whole claim. Five cases cannot certify a")
    print("  release; they can only catch the breakages the five cases cover.")


if __name__ == "__main__":
    main()
