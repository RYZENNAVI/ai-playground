"""This script answers questions about an invented insurer's policy wording.
The eight documents are written into the script, and each question names the
document that answers it, so retrieval can be scored rather than judged. The
same chunks go to a keyword index (BM25) and a vector index. The two rankings
are also fused two ways. Reciprocal rank fusion (RRF) adds 1 / (60 + rank) from
each backend and never looks at the scores. The weighted fusion rescales each
backend's scores to 0 to 1 for the question and adds them half and half. Raw
scores cannot be added: BM25 has no upper bound and cosine does.

The run prints six parts:
    1. The corpus, chunked. Windows of 60 words with 15 overlapping.
    2. Two indexes and two fusions. The four backends the rest of the run uses.
    3. Retrieval scores. Each question through each backend, and the rank of the
       expected document if it is in the top 3.
    4. Two ways to cut the context. The same retrieved chunks cut to three, and
       to an estimated budget of 220 tokens.
    5. Answers. Gemini answers each question from the keyword backend's chunks
       and from the vector backend's.
    6. Peeling a failure back. The first question a backend missed, traced one
       layer at a time from the answer down to the cause.

Run with --ui to serve the same backends behind a small web interface.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

CHUNK_WORDS = 60
CHUNK_OVERLAP = 15
TOP_K = 3
TOKEN_BUDGET = 220
CHARS_PER_TOKEN = 4
# RRF's usual constant: it keeps the top rank of one backend from outweighing the other.
RRF_K = 60
KEYWORD_WEIGHT = 0.5

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 768
CHAT_MODEL = "gemini-3.1-flash-lite"
MAX_ATTEMPTS = 5
RETRY_BACKOFF = 8

# An invented insurer, invented products, invented clause numbers. The corpus is
# written here rather than loaded so the whole script runs with no data files, and
# so every question below has a known correct source document.
CORPUS = {
    "travel-delay": """
        Clause 4.1 Trip Delay Benefit. Meridian Assurance reimburses reasonable
        additional accommodation and meal expenses when a scheduled departure is
        delayed by more than six consecutive hours. The daily limit is 180 units and
        the aggregate limit per trip is 720 units. A written confirmation from the
        carrier stating the length of and reason for the delay must be submitted
        within thirty days of the delayed departure. Delays caused by a strike
        announced before the policy start date are excluded from this benefit.
    """,
    "baggage-loss": """
        Clause 4.2 Baggage Benefit. Meridian Assurance covers checked baggage that is
        permanently lost, stolen or damaged while in the custody of a common carrier.
        The limit is 1,200 units per trip and 400 units for any single article. Items
        left unattended in a public place are not covered. Claims require the property
        irregularity report issued by the carrier and receipts or other proof of value
        for any article claimed above 150 units.
    """,
    "medical-abroad": """
        Clause 5.1 Emergency Medical Expenses Abroad. Meridian Assurance pays for
        emergency treatment, hospital admission and prescribed medication required
        while the insured is outside their country of residence. The limit is 500,000
        units. Treatment that could reasonably be postponed until the insured returns
        home is not an emergency under this clause. Pre-existing conditions declared
        and accepted at underwriting remain covered; undeclared conditions do not.
    """,
    "medical-evacuation": """
        Clause 5.2 Repatriation and Evacuation. Where the treating physician and the
        Meridian Assurance assistance centre jointly determine that local facilities
        are inadequate, transport to the nearest suitable facility or to the country
        of residence is arranged and paid for directly. Arrangements made without the
        prior agreement of the assistance centre are reimbursed only up to the cost
        the centre would have incurred.
    """,
    "employer-liability": """
        Clause 7.3 Employer Liability. The policy indemnifies the insured employer
        against sums they become legally liable to pay as damages for bodily injury
        sustained by an employee arising out of and in the course of employment. The
        limit of indemnity is 5,000,000 units in the aggregate for the period of
        insurance. Liability assumed under contract beyond what would exist at common
        law is excluded unless endorsed.
    """,
    "public-liability": """
        Clause 7.4 Public Liability. The policy indemnifies the insured against sums
        payable as damages for accidental bodily injury to a third party or accidental
        damage to third party property occurring at the insured premises. The limit of
        indemnity is 2,000,000 units for any one occurrence. Damage to property in the
        insured's own custody or control is excluded.
    """,
    "property-allrisks": """
        Clause 9.1 Property All Risks. Insured property is covered against accidental
        physical loss or damage other than by an excluded cause. The sum insured is
        stated in the schedule and represents the reinstatement value. Where the sum
        insured is less than the reinstatement value at the time of loss, any claim is
        reduced in the same proportion. Wear, tear and gradual deterioration are
        excluded throughout.
    """,
    "claims-process": """
        Clause 11.2 Notification and Settlement. Notice of any event likely to give
        rise to a claim must reach Meridian Assurance within fourteen days. The
        insurer acknowledges receipt within three working days and issues a decision
        within twenty working days of receiving a complete file. Where a decision
        cannot be reached in that period the insurer states in writing what further
        evidence is required and when a decision is expected.
    """,
}

# The two groups are meant to favour opposite backends: exact terms the keyword
# index, paraphrases the vector index. Step 3 shows whether they do.
QUESTIONS = [
    ("What does Clause 7.3 cover?", "employer-liability", "exact term"),
    ("What is the aggregate limit under Clause 4.1?", "travel-delay", "exact term"),
    ("Someone stole my suitcase at the airport. Am I covered?", "baggage-loss", "paraphrase"),
    ("I got sick on holiday and had to be flown home. Who pays?",
     "medical-evacuation", "paraphrase"),
    ("How long does the insurer have to decide on my claim?", "claims-process", "paraphrase"),
]


def chunk_documents() -> list:
    """Split every document into overlapping windows of whole words, each tagged with its document.
    The overlap keeps a sentence that straddles a boundary whole in one of the windows."""
    chunks = []
    for name, text in CORPUS.items():
        words = " ".join(text.split()).split(" ")
        start = 0
        while start < len(words):
            window = words[start:start + CHUNK_WORDS]
            chunks.append({
                "document": name,
                "position": len(chunks),
                "text": " ".join(window),
            })
            if start + CHUNK_WORDS >= len(words):
                break
            start += CHUNK_WORDS - CHUNK_OVERLAP
    return chunks


def tokenize(text: str) -> list:
    """Lowercase and split into words, keeping the dot inside clause numbers such as 7.3.
    Every sentence-final dot is kept too; step 6 shows what that costs."""
    return re.findall(r"[a-z0-9.]+", text.lower())


class KeywordBackend:
    """Score chunks by BM25, where rare words such as a clause number weigh most.
    A word the question never uses adds nothing, which makes it precise and brittle."""

    name = "keyword"

    def __init__(self, chunks: list, split=tokenize):
        self.chunks = chunks
        self.split = split
        self.index = BM25Okapi([split(chunk["text"]) for chunk in chunks])

    def search(self, query: str, limit: int) -> list:
        scores = self.index.get_scores(self.split(query))
        order = np.argsort(scores)[::-1][:limit]
        return [dict(self.chunks[i], score=float(scores[i])) for i in order]


class VectorBackend:
    """Score chunks by the cosine similarity of their embeddings to the question's,
    so a question can find an answer that shares none of its words."""

    name = "vector"

    def __init__(self, chunks: list, client: OpenAI):
        self.chunks = chunks
        self.client = client
        self.matrix = self._embed([chunk["text"] for chunk in chunks])

    def _embed(self, texts: list) -> np.ndarray:
        vectors = []
        for text in texts:
            response = call_with_retry(
                self.client, kind="embedding",
                model=EMBED_MODEL, input=text, dimensions=EMBED_DIMENSIONS,
            )
            vectors.append(response.data[0].embedding)
        matrix = np.asarray(vectors, dtype=float)
        # Truncated embeddings are not unit length, so cosine has to be taken
        # explicitly rather than read off a dot product.
        return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    def search(self, query: str, limit: int) -> list:
        vector = self._embed([query])[0]
        scores = self.matrix @ vector
        order = np.argsort(scores)[::-1][:limit]
        return [dict(self.chunks[i], score=float(scores[i])) for i in order]


class HybridBackend:
    """Fuse the keyword and vector rankings of every chunk, by RRF or by a weighted sum."""

    def __init__(self, keyword: KeywordBackend, vector: VectorBackend, method: str):
        self.chunks = keyword.chunks
        self.backends = ((keyword, KEYWORD_WEIGHT), (vector, 1 - KEYWORD_WEIGHT))
        self.name = method

    def search(self, query: str, limit: int) -> list:
        total = len(self.chunks)
        fused = np.zeros(total)
        for backend, weight in self.backends:
            hits = backend.search(query, total)
            if self.name == "rrf":
                for rank, hit in enumerate(hits, start=1):
                    fused[hit["position"]] += 1 / (RRF_K + rank)
            else:
                scores = np.array([hit["score"] for hit in hits])
                spread = scores.max() - scores.min()
                scaled = (scores - scores.min()) / spread if spread else np.zeros(total)
                for hit, value in zip(hits, scaled):
                    fused[hit["position"]] += weight * value
        order = np.argsort(fused)[::-1][:limit]
        return [dict(self.chunks[i], score=float(fused[i])) for i in order]


def call_with_retry(client, kind: str = "chat", **kwargs):
    """Send one request, backing off when the provider answers with a rate limit.
    Indexing sends one request per chunk in a burst, which trips per-minute limits."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if kind == "embedding":
                return client.embeddings.create(**kwargs)
            return client.chat.completions.create(**kwargs)
        except Exception as error:
            retriable = any(token in str(error).lower()
                            for token in ("429", "rate", "exhausted", "timeout", "503"))
            if not retriable or attempt == MAX_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF * attempt
            print(f"    provider pushed back ({type(error).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def by_count(hits: list, limit: int) -> list:
    """Keep a fixed number of chunks, which is the cutoff a search call usually exposes."""
    return hits[:limit]


def by_token_budget(hits: list, budget: int) -> list:
    """Keep chunks until the estimated tokens would pass the budget, but always keep the first.
    Tokens are characters / CHARS_PER_TOKEN, an estimate rather than a tokenizer count."""
    kept, spent = [], 0
    for hit in hits:
        cost = max(1, len(hit["text"]) // CHARS_PER_TOKEN)
        if kept and spent + cost > budget:
            break
        kept.append(hit)
        spent += cost
    return kept


def estimate_tokens(hits: list) -> int:
    """Estimate the token cost of a set of chunks, using the same rule as the budget."""
    return sum(max(1, len(hit["text"]) // CHARS_PER_TOKEN) for hit in hits)


def answer(client, question: str, hits: list) -> str:
    """Answer one question from the retrieved chunks alone."""
    context = "\n\n".join(f"[{hit['document']}] {hit['text']}" for hit in hits)
    response = call_with_retry(
        client, kind="chat",
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content":
                "Answer only from the context provided. Name the clause you used. "
                "If the context does not contain the answer, reply exactly: "
                "not in the retrieved context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


def pick_client() -> OpenAI:
    """Return a Gemini client for both embeddings and chat, or None without a key.
    One provider means one key and one quota to check when something fails."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key,
                  base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def locate(hits: list, expected: str) -> tuple:
    """Return the rank and score of the expected document's best chunk, or (None, None)."""
    for rank, hit in enumerate(hits, start=1):
        if hit["document"] == expected:
            return rank, hit["score"]
    return None, None


def score_backends(backends: list) -> dict:
    """Run every question through every backend and record whether the source came back."""
    results = {}
    for backend in backends:
        rows = []
        for question, expected, kind in QUESTIONS:
            hits = backend.search(question, TOP_K)
            found = [hit["document"] for hit in hits]
            rows.append({
                "question": question, "expected": expected, "kind": kind,
                "returned": found, "hit": expected in found,
                "rank": found.index(expected) + 1 if expected in found else None,
                "hits": hits,
            })
        results[backend.name] = rows
    return results


def build_ui(backends: dict, client):
    """Build a small web interface over the same backends, without launching it."""
    import gradio as gr

    def respond(question: str, backend_name: str, cutoff: str):
        backend = backends[backend_name]
        hits = backend.search(question, 8)
        kept = (by_token_budget(hits, TOKEN_BUDGET) if cutoff == "token budget"
                else by_count(hits, TOP_K))
        retrieved = "\n\n".join(
            f"[{hit['document']}] score {hit['score']:.4f}\n{hit['text']}" for hit in kept
        )
        reply = answer(client, question, kept) if client else "no API key configured"
        return retrieved, f"{estimate_tokens(kept)} estimated tokens", reply

    with gr.Blocks(title="Policy search") as demo:
        gr.Markdown("## Policy search\nAsk a question against the policy corpus.")
        with gr.Row():
            question = gr.Textbox(label="Question", scale=3,
                                  value=QUESTIONS[2][0])
            backend_name = gr.Radio(list(backends), value="keyword", label="Backend")
            cutoff = gr.Radio(["fixed count", "token budget"], value="token budget",
                              label="Context cutoff")
        ask = gr.Button("Search", variant="primary")
        with gr.Row():
            retrieved = gr.Textbox(label="Retrieved chunks", lines=14)
            reply = gr.Textbox(label="Answer", lines=14)
        spent = gr.Textbox(label="Context size")
        ask.click(respond, [question, backend_name, cutoff], [retrieved, spent, reply])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ui", action="store_true", help="serve the web interface")
    parser.add_argument("--share", action="store_true", help="expose the interface publicly")
    args = parser.parse_args()

    client = pick_client()
    if client is None:
        print("(no GEMINI_API_KEY; this script needs it for both the embeddings and the answers)")
        return

    # 1. The corpus, chunked

    chunks = chunk_documents()
    print("--- 1. The corpus, chunked ---")
    print(f"    documents {len(CORPUS)}, chunks {len(chunks)}, "
          f"window {CHUNK_WORDS} words with {CHUNK_OVERLAP} overlapping")
    lengths = [len(chunk["text"]) for chunk in chunks]
    print(f"    chunk length in characters: min {min(lengths)}, max {max(lengths)}, "
          f"mean {sum(lengths) / len(lengths):.0f}")
    print(f"    estimated tokens in the whole corpus: {estimate_tokens(chunks):,}")

    # 2. Two indexes and two fusions

    print("\n--- 2. Two indexes and two fusions ---")
    keyword = KeywordBackend(chunks)
    print(f"    keyword index built over {len(chunks)} chunks")
    vector = VectorBackend(chunks, client)
    print(f"    vector index built with {EMBED_MODEL} at {EMBED_DIMENSIONS} dimensions")
    backends = {"keyword": keyword, "vector": vector,
                "rrf": HybridBackend(keyword, vector, "rrf"),
                "weighted": HybridBackend(keyword, vector, "weighted")}
    print(f"    rrf fuses the two rankings with k={RRF_K}; weighted rescales both scores")
    print(f"    to 0 to 1 per query and gives keyword a weight of {KEYWORD_WEIGHT}")

    # 3. Retrieval scores

    print(f"\n--- 3. Retrieval scores, {len(QUESTIONS)} questions through each backend, "
          f"top {TOP_K} ---")
    scored = score_backends(list(backends.values()))
    print(f"    {'question':<42}{'kind':<12}" + "".join(f"{name:>10}" for name in backends))
    for i, (question, expected, kind) in enumerate(QUESTIONS):
        cells = [f"rank {scored[name][i]['rank']}" if scored[name][i]["hit"] else "missed"
                 for name in backends]
        print(f"    {question[:40]:<42}{kind:<12}" + "".join(f"{cell:>10}" for cell in cells))
    exact = [i for i, question in enumerate(QUESTIONS) if question[2] == "exact term"]
    para = [i for i, question in enumerate(QUESTIONS) if question[2] == "paraphrase"]
    for label, group in (("all", range(len(QUESTIONS))), ("exact term", exact),
                         ("paraphrase", para)):
        counts = [f"{sum(1 for i in group if scored[name][i]['hit'])}/{len(group)}"
                  for name in backends]
        print(f"    {'found, ' + label:<54}" + "".join(f"{count:>10}" for count in counts))

    # Fusion can only add a hit that one backend has and the other lacks.
    missed = {name: {i for i, row in enumerate(scored[name]) if not row["hit"]}
              for name in backends}
    if missed["keyword"] - missed["vector"] and missed["vector"] - missed["keyword"]:
        print("\n    The two backends miss different questions, which is the case fusion is for.")
    else:
        strong, weak = (("vector", "keyword") if len(missed["vector"]) <= len(missed["keyword"])
                        else ("keyword", "vector"))
        print(f"\n    The {strong} backend finds everything the {weak} backend finds, so fusion")
        print("    has no hit to add. It can only pull a hit down:")
        for name in ("rrf", "weighted"):
            lost = missed[name] - missed[strong]
            for i in sorted(lost):
                print(f"      {name}: {QUESTIONS[i][0]}")
            if not lost:
                print(f"      {name}: nothing lost on this run")
        print("    Fusion pays only when each backend finds what the other misses.")
        if missed[weak]:
            print(f"    Step 6 traces the {weak} miss to its cause.")

    # 4. Two ways to cut the context

    print("\n--- 4. Two ways to cut the context ---")
    print(f"    {'question':<40}{'fixed count':>22}{'token budget':>24}")
    print(f"    {'':<40}{'chunks':>10}{'tokens':>12}{'chunks':>12}{'tokens':>12}")
    largest_counted, largest_budgeted = 0, 0
    for question, _, _ in QUESTIONS:
        hits = vector.search(question, 8)
        counted = by_count(hits, TOP_K)
        budgeted = by_token_budget(hits, TOKEN_BUDGET)
        largest_counted = max(largest_counted, estimate_tokens(counted))
        largest_budgeted = max(largest_budgeted, estimate_tokens(budgeted))
        print(f"    {question[:38]:<40}{len(counted):>10}{estimate_tokens(counted):>12}"
              f"{len(budgeted):>12}{estimate_tokens(budgeted):>12}")
    print(f"\n    Tokens are estimated as characters / {CHARS_PER_TOKEN}, not counted by a tokenizer.")
    print(f"    Budget {TOKEN_BUDGET}: the largest budgeted context is {largest_budgeted}, "
          f"the largest fixed-count one {largest_counted}.")
    print(f"    Chunks here run {min(lengths)} to {max(lengths)} characters, so three chunks cost")
    print("    whatever those three happen to hold. The count does not track the unit")
    print("    being limited.")

    # 5. Answers

    print("\n--- 5. Answers from the keyword and vector backends ---")
    replies = {}
    for i, (question, expected, kind) in enumerate(QUESTIONS):
        print(f"\n    Q: {question}   (expected source: {expected})")
        for name in ("keyword", "vector"):
            hits = by_token_budget(scored[name][i]["hits"], TOKEN_BUDGET)
            replies[name, i] = " ".join(answer(client, question, hits).split())
            sources = ", ".join(dict.fromkeys(hit["document"] for hit in hits))
            print(f"        {name:<9} retrieved [{sources}]")
            print(f"        {'':<9} {replies[name, i][:150]}")

    # 6. Peeling a failure back

    print("\n--- 6. Peeling a failure back one layer at a time ---")
    failing = next(
        (i for i in range(len(QUESTIONS))
         if not scored["keyword"][i]["hit"] or not scored["vector"][i]["hit"]),
        None,
    )
    if failing is None:
        print("    Both backends returned the expected document for every question here,")
        print("    so there is no failure to locate on this run.")
        return
    question, expected, kind = QUESTIONS[failing]
    broken = "keyword" if not scored["keyword"][failing]["hit"] else "vector"
    print(f"    Taking the {broken} backend on: {question}")
    print(f"    Layer 1, the answer      : {replies[broken, failing][:60]}")
    print(f"    Layer 2, the retrieval   : expected {expected}, "
          f"got {scored[broken][failing]['returned']}")
    hits = backends[broken].search(question, len(chunks))
    rank, best = locate(hits, expected)
    if rank is None:
        print(f"    Layer 3, the raw scoring : no chunk of {expected} is in the index at all")
        print("\n    Scoring cannot be the problem when the document never reached the index.")
        print("    The repair is upstream of retrieval: ingestion, chunking or parsing.")
        return
    print(f"    Layer 3, the raw scoring : the expected document's best chunk sits at "
          f"rank {rank} of {len(hits)}")
    print(f"                               top score {hits[0]['score']:.4f}, "
          f"expected document's best {best:.4f}")
    stuck = []
    if broken == "keyword":
        question_words = set(tokenize(question))
        stuck = sorted({word for word in tokenize(hits[rank - 1]["text"])
                        if word.endswith(".") and word[:-1] in question_words})
    if not stuck:
        print("\n    The answer was never the problem. The document was in the index the")
        print("    whole time and the scoring put it below the cutoff, which is a different")
        print("    repair from anything that could be done to the prompt.")
        return
    plain = KeywordBackend(chunks, split=lambda text: [w.rstrip(".") for w in tokenize(text)])
    plain_rank, _ = locate(plain.search(question, len(chunks)), expected)
    print(f"    Layer 4, the tokens      : the question has "
          f"{', '.join(repr(word[:-1]) for word in stuck)}, the chunk has "
          f"{', '.join(repr(word) for word in stuck)}")
    print(f"                               without sentence-final dots the chunk ranks "
          f"{plain_rank} of {len(chunks)}")
    print("\n    Neither the answer nor BM25 was the problem. The tokenizer keeps the dot")
    print("    in clause numbers such as 7.3, and with it every sentence-final dot, so")
    print("    these words never match. Nothing raised an error; the chunk just ranked lower.")


if __name__ == "__main__":
    parsed = argparse.ArgumentParser(add_help=False)
    parsed.add_argument("--ui", action="store_true")
    parsed.add_argument("--share", action="store_true")
    known, _ = parsed.parse_known_args()
    if known.ui:
        client = pick_client()
        chunks = chunk_documents()
        ui_backends = {"keyword": KeywordBackend(chunks)}
        if client is not None:
            ui_backends["vector"] = VectorBackend(chunks, client)
            for method in ("rrf", "weighted"):
                ui_backends[method] = HybridBackend(
                    ui_backends["keyword"], ui_backends["vector"], method)
        build_ui(ui_backends, client).launch(share=known.share)
    else:
        main()
