"""This script compares GraphRAG with plain vector retrieval on one multi-hop
question. The corpus is a 1,220-word invented archive about Northgate Lab. The
question asks how Mira Delaunay's way of working affected Port Halbrook. The
answer needs five facts joined end to end, and no passage states them together.
The baseline splits the text into 150-word chunks, embeds them with
gemini-embedding-001, retrieves the nearest 3 and has gemini-3.1-flash-lite answer
from them. GraphRAG (Microsoft's graphrag 2.7.2) extracts entities and
relationships when it builds the index, groups them into communities and
summarises each one. It then answers twice: a global search over the community
summaries, and a local search that starts from the entities in the question.
graphrag pins numpy 1.x, so it runs in its own virtual environment and this
script drives it from the command line.

The run prints eight parts:
    1. The corpus and the question. The five links the answer needs.
    2. Baseline. The 3 nearest chunks, which chain terms they contain, and the
       answer.
    3. The isolated environment. Where the graphrag interpreter is.
    4. Building the graph index. Skipped when a previous run left one.
    5. What the index contains. Table sizes, some entities and edges, and names
       that appear in two forms.
    6. Global search. The answer from community summaries, and whether its
       citations point at rows that exist.
    7. Local search. The same for the search that starts from the question's
       entities.
    8. Reading the three answers. How many of the question's entities each answer
       names, and which names it adds beyond them.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

HERE = Path(__file__).parent
CORPUS = HERE / "data" / "graphrag_input" / "northgate_archive.txt"
WORKSPACE = HERE / "models" / "graphrag"

# graphrag pins numpy 1.x. Installing it beside the rest of this module would force
# numpy back a major version and break torch, faiss and sentence-transformers along
# with it, so it lives in its own interpreter and is driven through the command
# line. Nothing it installs reaches this process.
VENV = Path(__file__).parents[2] / ".venv-graphrag"
VENV_PYTHON = VENV / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    VENV_PYTHON = VENV / "bin" / "python"

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.1-flash-lite"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# At 300 words this corpus splits into four chunks, so the top 3 would return most
# of it and similarity search could not fail. 150 words leaves it a real choice.
CHUNK_WORDS = 150
TOP_K = 3

# Answering this needs five separate facts joined end to end. No single passage
# states the connection, which is the condition a graph index is meant to handle
# and similarity search is not.
QUESTION = "How did Mira Delaunay's way of working end up affecting Port Halbrook?"

# Each link is paired with a term that marks it in retrieved text. The terms are
# specific to the chain: the last word of each sentence would not do, because
# "possible", "Institute" and "Lab" also occur in passages unrelated to it. A term
# can still appear in a passage that denies the link, so a count of terms is an
# upper bound on the links retrieved.
CHAIN = [
    ("Delaunay trained Tomas Ek at the Coastal Institute", "Tomas Ek"),
    ("Ek founded Northgate Lab", "Northgate Lab"),
    ("Northgate developed Latch Encoding", "Latch Encoding"),
    ("Latch Encoding made the Orrery system possible", "Orrery"),
    ("Orrery was deployed at Port Halbrook", "Port Halbrook"),
]

# The entities the question is about. Step 8 counts them in each answer, and
# separates them from the names an answer went and found on its own.
CHAIN_ENTITIES = ["Mira Delaunay", "Tomas Ek", "Coastal Institute", "Northgate Lab",
                  "Latch Encoding", "Orrery", "Port Halbrook"]

# The archive's own note on the Broch collection, which researchers confuse with the
# Northgate material. A retrieved chunk holding it contributes no link.
UNRELATED_MARKER = "warning against this"

INSTALL_HINT = f"""The isolated environment is missing. Create it with:

    python -m venv "{VENV}"
    "{VENV_PYTHON}" -m pip install graphrag==2.7.2

Then set GRAPHRAG_API_KEY in {WORKSPACE / '.env'} and run this script again."""


def gemini():
    """Return a client for Gemini, which supplies both models the baseline needs."""
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
    """Return (index, score) for the k rows nearest the query by cosine similarity."""
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
for name in ("entities", "relationships", "community_reports", "text_units"):
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
    """Return (what the index holds, None), or (None, the error) when it cannot be read.
    The graphrag interpreter reads the parquet tables, so this one needs no pyarrow."""
    probe = PROBE.replace("OUTPUT_DIR", repr(str(WORKSPACE / "output")))
    result = subprocess.run([str(VENV_PYTHON), "-c", probe],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        return None, lines[-1] if lines else f"exit code {result.returncode}"
    try:
        return json.loads(result.stdout.strip().splitlines()[-1]), None
    except (json.JSONDecodeError, IndexError):
        return None, "the probe printed no JSON"


# A graph answer cites its evidence as [Data: Reports (3, 5); Entities (8)]. The
# names in those markers are not the table names, so they need mapping back.
CITED_TABLE = {"reports": "community_reports", "entities": "entities",
               "relationships": "relationships", "sources": "text_units"}
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
    """Print whether each id an answer cites exists in the table it names.
    A real id under an invented claim still passes: this checks existence, not support."""
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
    """True when text names this entity as a whole word. A plain substring test would
    find the index entity EK inside the word "week"."""
    return re.search(r"\b" + re.escape(name.lower()) + r"\b",
                     (text or "").lower()) is not None


def collapse(titles):
    """Keep only the longest of titles that name the same thing, such as NORTHGATE and
    NORTHGATE LAB. This affects the printout only; the index still holds both."""
    kept = []
    for title in sorted(titles, key=len, reverse=True):
        if not any(mentions(title, other) for other in kept):
            kept.append(title)
    return sorted(kept)


def entity_split(answer, all_entities):
    """Return (chain entities the answer names, other index entities it names).
    The second list is collapsed, so one entity under two names counts once."""
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

    # 1. The corpus and the question

    print("=" * 98)
    print("--- 1. The corpus and the question ---")
    print(f"  {CORPUS.name}: {len(text.split())} words")
    print(f"\n  Q: {QUESTION}\n")
    print("  Answering it means joining five facts that are never stated together:")
    for i, (step, _) in enumerate(CHAIN, 1):
        print(f"    {i}. {step}")

    api = gemini()

    # 2. Baseline

    print("\n" + "=" * 98)
    print("--- 2. Baseline ---")
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
    print(f"\n  chain terms present in the retrieved text: "
          f"{len(CHAIN) - len(missing)}/{len(CHAIN)}")
    for step in missing:
        print(f"    missing: {step}")
    print("  A term can appear in a passage that denies the link, so this is an")
    print("  upper bound on the links retrieved.")
    print("\n  Answer:")
    wrap(baseline_answer, indent="    ")

    # 3. The isolated environment

    print("\n" + "=" * 98)
    print("--- 3. The isolated environment ---")
    if not VENV_PYTHON.exists():
        print(INSTALL_HINT)
        return
    print(f"  interpreter: {VENV_PYTHON}")
    print("  It holds the numpy 1.x that graphrag pins, and this script drives it")
    print("  entirely through the command line.")

    # 4. Building the graph index

    print("\n--- 4. Building the graph index ---")
    index_seconds = None
    if (WORKSPACE / "output" / "entities.parquet").exists():
        print("  an index is already present; delete models/graphrag/output to rebuild")
    else:
        print("  running, this is the slow step …")
        output, index_seconds = run_graphrag(["index", "--root", str(WORKSPACE)], "index")
        if output is None:
            return
        print(f"  finished in {index_seconds:.0f}s")

    # 5. What the index contains

    print("\n--- 5. What the index contains ---")
    stats, error = graph_stats()
    if error:
        print(f"  could not read the index: {error}")
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

    # 6. Global search

    print("\n--- 6. Global search ---")
    output, global_seconds = run_graphrag(
        ["query", "--root", str(WORKSPACE), "--method", "global", "--query", QUESTION],
        "global query")
    global_answer = clean(output)
    if global_answer:
        wrap(global_answer, indent="    ")
        check_citations(global_answer, (stats or {}).get("_ids") or {})
        print(f"\n  [{global_seconds:.0f}s]")

    # 7. Local search

    print("\n--- 7. Local search ---")
    output, local_seconds = run_graphrag(
        ["query", "--root", str(WORKSPACE), "--method", "local", "--query", QUESTION],
        "local query")
    local_answer = clean(output)
    if local_answer:
        wrap(local_answer, indent="    ")
        check_citations(local_answer, (stats or {}).get("_ids") or {})
        print(f"\n  [{local_seconds:.0f}s]")

    # 8. Reading the three answers

    print("\n" + "=" * 98)
    print("--- 8. Reading the three answers ---")
    all_entities = (stats or {}).get("_all_entities") or []
    named = {}
    for label, answer in (("baseline", baseline_answer), ("global", global_answer),
                          ("local", local_answer)):
        if not answer:
            continue
        on_chain, off_chain = entity_split(answer, all_entities)
        named[label] = on_chain
        print(f"  {label:<9} names {len(on_chain)} of the {len(CHAIN_ENTITIES)} "
              f"entities the question is about"
              + (f", plus {len(off_chain)} beyond it" if off_chain else ""))
        absent = [c for c in CHAIN_ENTITIES if c not in on_chain]
        if absent:
            print(f"    not named: {', '.join(absent)}")
        if off_chain:
            print(f"    beyond the chain: {', '.join(off_chain[:8])}")
    if not all_entities:
        print("  (the index could not be read, so names beyond the chain are not counted)")
    print("  In a graph answer, a name outside the chain is the graph following an edge")
    print("  nobody asked about. A genuine connection and a wrong edge look the same")
    print("  from outside, and the citation checks only show that the cited rows exist.")

    print()
    if any(UNRELATED_MARKER in chunks[i] for i, _ in hits):
        print("  One of the chunks the baseline retrieved is the Broch collection, which")
        print("  the archive itself warns researchers against. It scored well because it")
        print("  shares names and vocabulary, not because it holds a link.")
    base_named = len(named.get("baseline", []))
    if not missing and base_named < len(CHAIN_ENTITIES):
        print(f"  The retrieved text held all {len(CHAIN)} chain terms, yet the baseline "
              f"answer names only")
        print(f"  {base_named} of the {len(CHAIN_ENTITIES)} entities. Retrieving the "
              "passages is not the same as joining them.")
    print(f"  This corpus is small: {len(text.split())} words in {len(chunks)} chunks, so "
          f"the top {TOP_K} is")
    print(f"  {TOP_K / len(chunks):.0%} of it. The two approaches separate more clearly "
          "when the links sit")
    print("  far apart, which this corpus is too small to show.")

    print()
    print("  The graph answers can join the chain because the joins were computed when")
    print("  the index was built. Global reads community summaries and cites them as")
    print("  Reports. Local starts from the entities in the question and cites")
    print("  Entities, Relationships and Sources as well. The answers are fluent either")
    print("  way, and a wrong edge would read as well as a right one: structure")
    print("  improves what gets retrieved, it does not verify it.")

    print()
    print("  Cost. The baseline made three API calls: embeddings for the chunks,")
    print("  an embedding for the question, and the answer.", end=" ")
    print(f"The global search took {global_seconds:.0f}s")
    print(f"  and the local search {local_seconds:.0f}s on this run.", end=" ")
    if index_seconds is not None:
        print(f"Building the index took {index_seconds:.0f}s")
        print("  before any question could be asked.")
    else:
        print("The index was already")
        print("  built, so its cost is not in this run. Building it is a full pass over the")
        print("  corpus, and it grows with the corpus, not with the number of questions.")
    print("  It pays off only where the connections matter more than the passages.")


if __name__ == "__main__":
    main()
