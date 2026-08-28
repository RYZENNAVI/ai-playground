"""Serve the same corpus from two retrieval backends, cut the context two ways, and answer from each.

Demonstrates that a retrieval failure has to be located before it can be fixed:
    1. Build a policy corpus and split it into chunks of a stated size.
    2. Index those chunks once for keyword scoring and once for vector scoring.
    3. Run a query set through both backends and score the documents they return.
    4. Cut the retrieved context to a fixed number of chunks, and measure what that costs.
    5. Cut it to a token budget instead, and measure the same thing again.
    6. Answer each query from each backend, and read the answers against the source.
    7. Peel the same failing query back a layer at a time until the cause is visible.

Run with --ui to serve the same two backends behind a small web interface.

Module 10: Applied Projects - Retrieval Backends and Context Budgets.
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

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

CHUNK_WORDS = 60
CHUNK_OVERLAP = 15
TOP_K = 3
TOKEN_BUDGET = 220
CHARS_PER_TOKEN = 4

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

# Each question names the document that answers it, so retrieval can be scored
# rather than judged. The two groups are chosen to pull in opposite directions.
QUESTIONS = [
    ("What does Clause 7.3 cover?", "employer-liability", "exact term"),
    ("What is the aggregate limit under Clause 4.1?", "travel-delay", "exact term"),
    ("Someone stole my suitcase at the airport. Am I covered?", "baggage-loss", "paraphrase"),
    ("I got sick on holiday and had to be flown home. Who pays?",
     "medical-evacuation", "paraphrase"),
    ("How long does the insurer have to decide on my claim?", "claims-process", "paraphrase"),
]


def chunk_documents() -> list:
    """Split every document into overlapping windows of whole words.

    Windows overlap so a sentence that straddles a boundary still appears intact in
    one of them. Every chunk carries the name of the document it came from, which is
    what makes a retrieved chunk scoreable against the expected source.
    """
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
    """Lowercase and split into word characters, which is what the keyword index scores."""
    return re.findall(r"[a-z0-9.]+", text.lower())


class KeywordBackend:
    """Score chunks by term overlap, the way a text index does.

    Rare terms carry the most weight, so an exact clause number or an unusual noun
    ranks its chunk highly. A word the query never uses contributes nothing, which
    is the property that makes this backend precise and brittle at the same time.
    """

    name = "keyword"

    def __init__(self, chunks: list):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    def search(self, query: str, limit: int) -> list:
        scores = self.index.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:limit]
        return [dict(self.chunks[i], score=float(scores[i])) for i in order]


class VectorBackend:
    """Score chunks by cosine similarity between embeddings of the query and the chunk.

    Embeddings place text that means similar things near each other, so a question
    that shares no words with its answer can still find it. An identifier is scored
    the same way, by what it resembles rather than by matching it, so it can still be
    found but is not guaranteed the top position the way an exact match is.
    """

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


def call_with_retry(client, kind: str = "chat", **kwargs):
    """Send one request, backing off when the provider answers with a rate limit.

    Indexing a corpus sends one request per chunk in a burst, which is exactly the
    shape that trips a per-minute limit even when the daily quota is untouched.
    """
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
    """Keep chunks until the estimated token count would exceed the budget.

    The unit here is the one the model is actually limited by. A count of chunks only
    stands in for it while every chunk is the same size, and stops standing in for it
    the moment the corpus holds documents of different lengths.
    """
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
    """Return a client for the one provider this script uses for both roles.

    Retrieval needs embeddings and answering needs chat, and keeping both on one
    provider means one key, one base URL and one quota to reason about when
    something starts failing.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key,
                  base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


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
    """Build a small web interface over the same two backends, without launching it."""
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
        print("No GEMINI_API_KEY found. This script uses one provider for both the")
        print("embeddings and the answers, so that key is required.")
        return

    chunks = chunk_documents()
    print("--- 1. The corpus, chunked ---")
    print(f"    documents {len(CORPUS)}, chunks {len(chunks)}, "
          f"window {CHUNK_WORDS} words with {CHUNK_OVERLAP} overlapping")
    lengths = [len(chunk["text"]) for chunk in chunks]
    print(f"    chunk length in characters: min {min(lengths)}, max {max(lengths)}, "
          f"mean {sum(lengths) / len(lengths):.0f}")
    print(f"    estimated tokens in the whole corpus: {estimate_tokens(chunks):,}")

    print("\n--- 2. Two indexes over the same chunks ---")
    keyword = KeywordBackend(chunks)
    print(f"    keyword index built over {len(chunks)} chunks")
    vector = VectorBackend(chunks, client)
    print(f"    vector index built with {EMBED_MODEL} at {EMBED_DIMENSIONS} dimensions")
    backends = {"keyword": keyword, "vector": vector}

    print(f"\n--- 3. {len(QUESTIONS)} questions through both backends, top {TOP_K} ---")
    scored = score_backends([keyword, vector])
    print(f"    {'question':<52}{'kind':<12}{'keyword':>9}{'vector':>9}")
    for i, (question, expected, kind) in enumerate(QUESTIONS):
        left = scored["keyword"][i]
        right = scored["vector"][i]
        left_text = f"rank {left['rank']}" if left["hit"] else "missed"
        right_text = f"rank {right['rank']}" if right["hit"] else "missed"
        print(f"    {question[:50]:<52}{kind:<12}{left_text:>9}{right_text:>9}")
    for name in ("keyword", "vector"):
        hits = sum(1 for row in scored[name] if row["hit"])
        print(f"    {name:<10} found the expected document for {hits} of {len(QUESTIONS)}")

    exact = [i for i, question in enumerate(QUESTIONS) if question[2] == "exact term"]
    para = [i for i, question in enumerate(QUESTIONS) if question[2] == "paraphrase"]
    for label, group in (("exact term", exact), ("paraphrase", para)):
        left = sum(1 for i in group if scored["keyword"][i]["hit"])
        right = sum(1 for i in group if scored["vector"][i]["hit"])
        left_ranks = [scored["keyword"][i]["rank"] for i in group if scored["keyword"][i]["hit"]]
        right_ranks = [scored["vector"][i]["rank"] for i in group if scored["vector"][i]["hit"]]
        left_mean = f"{np.mean(left_ranks):.1f}" if left_ranks else "-"
        right_mean = f"{np.mean(right_ranks):.1f}" if right_ranks else "-"
        print(f"    on {label:<12} questions: keyword {left} of {len(group)} "
              f"(mean rank {left_mean}), vector {right} of {len(group)} "
              f"(mean rank {right_mean})")
    print("\n    Neither backend is the better one. They fail on different questions, and")
    print("    the split above is the reason a system keeps both rather than choosing.")

    print(f"\n--- 4-5. Two ways to cut the context ---")
    print(f"    {'question':<40}{'fixed count':>22}{'token budget':>24}")
    print(f"    {'':<40}{'chunks':>10}{'tokens':>12}{'chunks':>12}{'tokens':>12}")
    for question, _, _ in QUESTIONS:
        hits = vector.search(question, 8)
        counted = by_count(hits, TOP_K)
        budgeted = by_token_budget(hits, TOKEN_BUDGET)
        print(f"    {question[:38]:<40}{len(counted):>10}{estimate_tokens(counted):>12}"
              f"{len(budgeted):>12}{estimate_tokens(budgeted):>12}")
    print(f"\n    The budget is {TOKEN_BUDGET} tokens and the token column stays under it.")
    print("    A fixed count does not track tokens at all; it happens to here because")
    print("    every chunk in this corpus is nearly the same length. Add one long")
    print("    document and the count stops standing in for the thing being limited.")

    print("\n--- 6. Answers from each backend ---")
    for i, (question, expected, kind) in enumerate(QUESTIONS):
        print(f"\n    Q: {question}   (expected source: {expected})")
        for name in ("keyword", "vector"):
            hits = by_token_budget(scored[name][i]["hits"], TOKEN_BUDGET)
            reply = answer(client, question, hits)
            sources = ", ".join(dict.fromkeys(hit["document"] for hit in hits))
            print(f"        {name:<9} retrieved [{sources}]")
            print(f"        {'':<9} {reply[:150]}")

    print("\n--- 7. Peeling a query back one layer at a time ---")
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
    print(f"    Layer 1, the answer      : built from "
          f"{scored[broken][failing]['returned']}")
    print(f"    Layer 2, the retrieval   : expected {expected}, "
          f"got {scored[broken][failing]['returned']}")
    hits = backends[broken].search(question, len(chunks))
    ranks = [i for i, hit in enumerate(hits, start=1) if hit["document"] == expected]
    print(f"    Layer 3, the raw scoring : the expected document's best chunk sits at "
          f"rank {min(ranks)} of {len(hits)}")
    print(f"                               top score {hits[0]['score']:.4f}, "
          f"expected document's best {max(h['score'] for h in hits if h['document'] == expected):.4f}")
    print("\n    The answer was never the problem. The document was in the index the")
    print("    whole time and the scoring put it below the cutoff, which is a different")
    print("    repair from anything that could be done to the prompt.")


if __name__ == "__main__":
    parsed = argparse.ArgumentParser(add_help=False)
    parsed.add_argument("--ui", action="store_true")
    known, _ = parsed.parse_known_args()
    if known.ui:
        client = pick_client()
        chunks = chunk_documents()
        ui_backends = {"keyword": KeywordBackend(chunks)}
        if client is not None:
            ui_backends["vector"] = VectorBackend(chunks, client)
        build_ui(ui_backends, client).launch()
    else:
        main()
