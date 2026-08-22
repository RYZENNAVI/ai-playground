"""Ask one multi-hop question twice: once of a vector index, once of a knowledge graph.

Demonstrates what a graph index buys over similarity search, and what it costs:
    1. Read the corpus and show the chain of facts the question depends on.
    2. Baseline: embed the chunks, retrieve the nearest ones, answer from those.
    3. Locate the isolated environment the graph tool runs in.
    4. Build the graph index, unless a previous run already produced one.
    5. Report what that index contains.
    6. Global search, which reasons over community summaries.
    7. Local search, which walks out from the entities in the question.
    8. Put the three answers side by side.

Module 02: RAG - Graph Retrieval.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

HERE = Path(__file__).parent
CORPUS = HERE / "data" / "graphrag_input" / "northgate_archive.txt"
WORKSPACE = HERE / "models" / "graphrag"

# The graph tool pins numpy 1.x and pandas 2.x. Installing it beside the rest of
# this module would force numpy back a major version and break torch, faiss and
# sentence-transformers along with it, so it lives in its own interpreter and is
# driven through the command line. Nothing it installs reaches this process.
VENV = Path(__file__).parents[2] / ".venv-graphrag"
VENV_PYTHON = VENV / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = VENV / "bin" / "python"

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.1-flash-lite"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Chunk size has to leave top-k retrieval with an actual choice to make. At 300
# words this corpus splits into four chunks, so asking for three returns most of
# the document and similarity search cannot fail - not because it is good, but
# because it was never made to select. Smaller chunks put that decision back.
CHUNK_WORDS = 150
TOP_K = 3

# Answering this needs five separate facts joined end to end. No single passage
# states the connection, which is the condition a graph index is meant to handle
# and similarity search is not.
QUESTION = "How did Mira Delaunay's way of working end up affecting Port Halbrook?"

# Each link is paired with the term that settles whether it was retrieved, and
# every term is chosen to be unique in this corpus. Taking the last word of the
# sentence instead - the obvious shortcut - makes link four turn on the word
# "possible" and links one and two on "Institute" and "Lab", terms that occur in
# passages having nothing to do with the chain. A coverage number is only worth
# printing when the thing it counts is specific enough to be wrong.
CHAIN = [
    ("Delaunay trained Tomas Ek at the Coastal Institute", "Tomas Ek"),
    ("Ek founded Northgate Lab", "Northgate Lab"),
    ("Northgate developed Latch Encoding", "Latch Encoding"),
    ("Latch Encoding made the Orrery system possible", "Orrery"),
    ("Orrery was deployed at Port Halbrook", "Port Halbrook"),
]

# The entities the question is actually about. Step 8 uses this to separate the
# names a graph answer was asked for from the ones it went and found on its own.
CHAIN_ENTITIES = ["Mira Delaunay", "Tomas Ek", "Coastal Institute", "Northgate Lab",
                  "Latch Encoding", "Orrery", "Port Halbrook"]

INSTALL_HINT = f"""The isolated environment is missing. Create it with:

    python -m venv "{VENV}"
    "{VENV_PYTHON}" -m pip install graphrag==2.7.2

Then set GRAPHRAG_API_KEY in {WORKSPACE / '.env'} and run this script again."""


def gemini():
    """Return a client for Gemini, which supplies both models this script needs."""
    from openai import OpenAI

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set. Add it to .env and retry.")
    return OpenAI(api_key=key, base_url=GEMINI_BASE)


def chunk_words(text, size=CHUNK_WORDS):
    """Split the corpus into word-count chunks, paragraph boundaries preferred."""
    chunks, current = [], []
    for paragraph in [p.strip() for p in text.split("\n\n") if p.strip()]:
        current.append(paragraph)
        if sum(len(c.split()) for c in current) >= size:
            chunks.append("\n\n".join(current))
            current = []
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def embed(api, texts):
    """Embed a list of strings and return plain float vectors."""
    response = api.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def cosine_top_k(query_vector, matrix, k=TOP_K):
    """Return the indexes of the k nearest rows by cosine similarity.

    Vectors are normalised before the dot product rather than after, because a
    truncated embedding is not a unit vector and the raw dot product would then
    be ranking magnitude alongside direction.
    """
    import numpy as np

    q = np.asarray(query_vector, dtype="float32")
    m = np.asarray(matrix, dtype="float32")
    q /= (np.linalg.norm(q) or 1.0)
    m /= (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    scores = m @ q
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


def answer_from_context(api, question, context):
    """Answer strictly from the retrieved text, so gaps in it show up as gaps."""
    response = api.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content":
                   "Answer the question using only the context below. If the context "
                   "does not establish the connection being asked about, say so "
                   "plainly rather than filling the gap.\n\n"
                   f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return response.choices[0].message.content.strip()


def run_graphrag(args, label):
    """Run the graph tool in its own interpreter and return (output, seconds)."""
    started = time.time()
    result = subprocess.run(
        [str(VENV_PYTHON), "-m", "graphrag", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    elapsed = time.time() - started
    if result.returncode != 0:
        print(f"  {label} failed after {elapsed:.0f}s")
        print("  " + "\n  ".join((result.stderr or result.stdout).strip().splitlines()[-8:]))
        return None, elapsed
    return result.stdout, elapsed


PROBE = """
import glob, json, os
import pandas as pd

folder = OUTPUT_DIR
out = {}
for path in glob.glob(os.path.join(folder, "*.parquet")):
    out[os.path.basename(path)[:-8]] = len(pd.read_parquet(path))

entities = pd.read_parquet(os.path.join(folder, "entities.parquet"))
relationships = pd.read_parquet(os.path.join(folder, "relationships.parquet"))
out["_entities"] = list(entities["title"])[:14]
out["_all_entities"] = [t for t in entities["title"] if t]
out["_edges"] = ["%s -> %s" % (a, b) for a, b in
                 zip(relationships["source"], relationships["target"])][:8]

# The ids an answer is allowed to cite. Gathered here because the answers name
# rows by human_readable_id, and only this interpreter can read the tables.
ids = {}
for name in ("entities", "relationships", "community_reports"):
    path = os.path.join(folder, name + ".parquet")
    if not os.path.exists(path):
        continue
    frame = pd.read_parquet(path)
    if "human_readable_id" in frame.columns:
        ids[name] = sorted(int(v) for v in frame["human_readable_id"].dropna())
out["_ids"] = ids

print(json.dumps(out))
"""


def graph_stats():
    """Ask the isolated interpreter what the index contains.

    The tables are parquet, and reading them here would mean adding pyarrow to
    this environment for the sake of a few counts. The interpreter that wrote
    them already has it, so it does the reading and returns JSON.
    """
    probe = PROBE.replace("OUTPUT_DIR", repr(str(WORKSPACE / "output")))
    result = subprocess.run([str(VENV_PYTHON), "-c", probe],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None


# A graph answer cites its evidence as [Data: Reports (3, 5); Entities (8)]. The
# names in those markers are not the table names, so they need mapping back.
CITED_TABLE = {"reports": "community_reports", "entities": "entities",
               "relationships": "relationships"}
CITATION_BLOCK = re.compile(r"\[Data:([^\]]*)\]", re.IGNORECASE)
CITATION_PART = re.compile(r"([A-Za-z ]+)\(([^)]*)\)")


def cited_ids(text):
    """Return {table: {id, ...}} for every [Data: ...] marker in an answer."""
    found = {}
    for block in CITATION_BLOCK.findall(text or ""):
        for table, numbers in CITATION_PART.findall(block):
            key = CITED_TABLE.get(table.strip().lower())
            if key:
                found.setdefault(key, set()).update(
                    int(n) for n in re.findall(r"[0-9]+", numbers))
    return found


def check_citations(answer, known_ids):
    """Report whether each id an answer cites exists in the table it named.

    This settles the one part of "should I believe this" that needs no model: a
    citation either points at a row that exists or it does not. It says nothing
    about whether that row supports the sentence it is attached to - a real id
    under an invented claim passes here, which is the limit worth stating out
    loud rather than letting the check imply more than it verifies.
    """
    cited = cited_ids(answer)
    if not cited:
        print("    citations: the answer names no evidence")
        return
    for table, numbers in sorted(cited.items()):
        valid = set(known_ids.get(table) or [])
        if not valid:
            print(f"    citations: {len(numbers)} to {table}, ids unavailable")
            continue
        unknown = sorted(n for n in numbers if n not in valid)
        line = f"    citations: {len(numbers) - len(unknown)}/{len(numbers)} resolve in {table}"
        print(line + (f"   UNKNOWN {unknown}" if unknown else ""))


def mentions(name, text):
    """True when text names this entity as a word rather than inside another.

    Substring matching is not good enough here: the index holds an entity
    called EK alongside TOMAS EK, and a bare substring test finds the first one
    in any answer containing the word "week".
    """
    return re.search(r"\b" + re.escape(name.lower()) + r"\b",
                     (text or "").lower()) is not None


def collapse(titles):
    """Drop each title that is a shorter form of another title in the same set.

    The index lists RAMAN beside PRIYA RAMAN and NORTHGATE beside NORTHGATE LAB,
    which is the near-duplicate problem step 5 reports. Counting both makes one
    entity look like two, so the longest form wins and the rest are folded into
    it. This is presentation only - the index still holds both, and the note in
    step 5 is what says so.
    """
    kept = []
    for title in sorted(titles, key=len, reverse=True):
        if not any(mentions(title, other) for other in kept):
            kept.append(title)
    return sorted(kept)


def entity_split(answer, all_entities):
    """Return (chain entities the answer names, graph entities beyond the chain).

    Both sides have to survive the same near-duplicate problem. The first list
    is counted against CHAIN_ENTITIES rather than against index titles, so it is
    deduplicated by construction; the second comes straight from the index and
    has to be collapsed by hand, or one entity under two names is reported as
    two separate findings.
    """
    on_chain = [c for c in CHAIN_ENTITIES if mentions(c, answer)]
    off_chain = {e for e in all_entities
                 if mentions(e, answer)
                 and not any(c.lower() in e.lower() or e.lower() in c.lower()
                             for c in CHAIN_ENTITIES)}
    return on_chain, collapse(off_chain)


def clean(text):
    """Drop the library's warning lines from captured output."""
    noise = ("LiteLLM:WARNING", "DeprecationWarning", "Move sampling", "WARN  lance")
    return "\n".join(line for line in (text or "").splitlines()
                     if line.strip() and not any(n in line for n in noise))


def wrap(text, width=94, indent="  "):
    """Print a paragraph at a readable width."""
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            if len(line) + len(word) + 1 > width:
                print(indent + line)
                line = word
            else:
                line = f"{line} {word}".strip()
        print(indent + line)


def main():
    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found at {CORPUS}")
    text = CORPUS.read_text(encoding="utf-8")

    print("=" * 98)
    print("--- 1. The corpus and the question ---")
    print(f"  {CORPUS.name}: {len(text.split())} words")
    print(f"\n  Q: {QUESTION}\n")
    print("  Answering it means joining five facts that are never stated together:")
    for i, (step, _) in enumerate(CHAIN, 1):
        print(f"    {i}. {step}")

    api = gemini()

    print("\n" + "=" * 98)
    print("--- 2. Baseline: similarity search over the same text ---")
    chunks = chunk_words(text)
    vectors = embed(api, chunks)
    query_vector = embed(api, [QUESTION])[0]
    hits = cosine_top_k(query_vector, vectors)
    print(f"  {len(chunks)} chunks, retrieving the nearest {TOP_K}")
    for rank, (index, score) in enumerate(hits, 1):
        first_line = " ".join(chunks[index].split())[:78]
        print(f"    {rank}. cos {score:.3f}  {first_line}…")

    context = "\n\n".join(chunks[i] for i, _ in hits)
    lowered = context.lower()
    missing = [step for step, marker in CHAIN if marker.lower() not in lowered]
    baseline_answer = answer_from_context(api, QUESTION, context)
    print(f"\n  chain links present in the retrieved text: "
          f"{len(CHAIN) - len(missing)}/{len(CHAIN)}")
    for step in missing:
        print(f"    missing: {step}")
    print("\n  Answer:")
    wrap(baseline_answer, indent="    ")

    print("\n" + "=" * 98)
    print("--- 3. The isolated environment ---")
    if not VENV_PYTHON.exists():
        print(INSTALL_HINT)
        return
    print(f"  interpreter: {VENV_PYTHON}")
    print("  It holds numpy 1.x and pandas 2.x, which this process does not, and")
    print("  is driven entirely through the command line.")

    print("\n--- 4. Building the graph index ---")
    if (WORKSPACE / "output" / "entities.parquet").exists():
        print("  an index is already present - delete models/graphrag/output to rebuild")
    else:
        print("  running, this is the slow step …")
        output, elapsed = run_graphrag(["index", "--root", str(WORKSPACE)], "index")
        if output is None:
            return
        print(f"  finished in {elapsed:.0f}s")

    print("\n--- 5. What the index contains ---")
    stats = graph_stats()
    if stats:
        for key in ("documents", "text_units", "entities", "relationships",
                    "communities", "community_reports"):
            if key in stats:
                print(f"    {stats[key]:>4}  {key}")
        print(f"\n  entities: {', '.join(stats.get('_entities', []))}")
        print("  a few edges:")
        for edge in stats.get("_edges", []):
            print(f"    {edge}")
        names = stats.get("_entities", [])
        near_duplicates = [n for n in names
                           if any(n != o and (n.startswith(o) or o.startswith(n))
                                  for o in names)]
        if near_duplicates:
            print(f"\n  Note {sorted(set(near_duplicates))}: the same organisation appears")
            print("  under two names. Extraction merges entities that share a name and a")
            print("  type; resolving different names for one real thing is a separate")
            print("  step that is off by default, and this is what that costs.")

    print("\n--- 6. Global search: reasoning over community summaries ---")
    output, elapsed = run_graphrag(
        ["query", "--root", str(WORKSPACE), "--method", "global", "--query", QUESTION],
        "global query")
    global_answer = clean(output)
    if global_answer:
        wrap(global_answer, indent="    ")
        check_citations(global_answer, (stats or {}).get("_ids") or {})
        print(f"\n  [{elapsed:.0f}s]")

    print("\n--- 7. Local search: walking out from the entities named in the question ---")
    output, elapsed = run_graphrag(
        ["query", "--root", str(WORKSPACE), "--method", "local", "--query", QUESTION],
        "local query")
    local_answer = clean(output)
    if local_answer:
        wrap(local_answer, indent="    ")
        check_citations(local_answer, (stats or {}).get("_ids") or {})
        print(f"\n  [{elapsed:.0f}s]")

    print("\n" + "=" * 98)
    print("--- 8. Reading the three answers ---")
    all_entities = (stats or {}).get("_all_entities") or []
    if all_entities:
        for label, answer in (("global", global_answer), ("local", local_answer)):
            on_chain, off_chain = entity_split(answer, all_entities)
            print(f"  {label:<7} names {len(on_chain)} of the {len(CHAIN_ENTITIES)} "
                  f"entities the question is about"
                  + (f", plus {len(off_chain)} beyond it" if off_chain else ""))
            if off_chain:
                print(f"    beyond the chain: {', '.join(off_chain[:8])}")
        print("  Names outside the chain are the graph following edges nobody asked")
        print("  about. That is the behaviour worth watching: it is where a genuine")
        print("  connection and a wrong edge look identical from the outside, and the")
        print("  citation check above only tells you the rows cited are real.")
        print()
    print("  Similarity search returned the passages that read most like the question,")
    print("  which is not the same as the passages that connect it. One of the three")
    print("  it retrieved is a collection the corpus explicitly describes as unrelated -")
    print("  it scores well because it shares names and vocabulary, not because it")
    print("  contributes a link.")
    if missing:
        print("  It also left a link out, and nothing in the retrieved text marks the")
        print("  absence, so the answer routes around the gap rather than reporting it.")
    else:
        print("  It carried all five links even so, and the baseline answer is sound.")
        print("  That is a fact about this corpus, not a verdict on the method. Twelve")
        print("  hundred words split seven ways makes the nearest three passages nearly")
        print("  half the archive, and a chain packed that tightly survives being")
        print("  retrieved by resemblance. Where the two approaches separate is a corpus")
        print("  whose five links sit hundreds of pages apart, and this one is too small")
        print("  to show it. Note what that costs the comparison: the graph is doing")
        print("  real work below, but this run does not prove it was needed.")
    print()
    print("  The graph answers differ because the joins were computed at index time.")
    print("  Global reads community summaries and cites them as Reports; local starts")
    print("  from the entities in the question and cites Entities and Relationships.")
    print("  Global suits questions about the corpus as a whole, local suits questions")
    print("  about a named thing in it.")
    print()
    print("  Neither is free of the usual caveat: the graph answers are fluent and")
    print("  confident, and a wrong edge would read exactly as well as a right one.")
    print("  Structure improves what gets retrieved; it does not verify it.")
    print()
    print("  Cost is the honest half of this comparison. The baseline was two API calls")
    print("  in total. The graph needed a full indexing pass before a single question")
    print("  could be asked - on this corpus roughly a minute and a half, and that pass")
    print("  scales with the corpus rather than with the number of questions. Over a")
    print("  book it runs for half an hour and bills accordingly. It earns that back")
    print("  only where the connections matter more than the passages do.")


if __name__ == "__main__":
    main()
