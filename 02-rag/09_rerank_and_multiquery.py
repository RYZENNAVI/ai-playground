"""Recall wide with a cheap scorer, then rerank narrow with an expensive one.

Demonstrates the two-stage retrieval pattern and the query expansion that feeds it:
    1. Load the knowledge base and keep each heading attached to its own text.
    2. Stage one: recall candidate paragraphs with BM25, which is keyword matching only.
    3. Stage two: rerank the sentences inside those paragraphs with a cross-encoder.
    4. Show why the unit fed to the reranker decides whether it works at all.
    5. Read the score scale, which is unbounded and not comparable across corpora.
    6. Expand one question into several phrasings and measure what that buys.

Module 02: RAG - Reranking and Query Expansion.
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

DATA_DIR = Path(__file__).parent / "data" / "disney_kb"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

# Stage one casts a wide net and stage two narrows it. RECALL_K well above
# FINAL_K is the whole point: if they were equal the reranker would only be
# reordering a set it can no longer change the membership of.
RECALL_K = 8
FINAL_K = 3
EXPANSION_COUNT = 4
# Fragments shorter than this are list markers and sentence debris, not answers.
MIN_SENTENCE_WORDS = 6

QUESTIONS = [
    "Can I move my visit to a different day after buying?",
    "My father is 68 - does he pay less?",
    "How do I skip the queue on the busiest rides?",
]


def load_chunks():
    """Read the .docx knowledge base into paragraph chunks.

    Each file opens with a heading. Indexing that heading as a chunk of its own
    is a trap: a five-word title matches almost any question on the topic and
    pushes the paragraph holding the real answer out of the top results. It is
    kept as a separate field instead - BM25 gets it as searchable context, and
    the reranker never sees it, for the reason step 4 measures.
    """
    from docx import Document

    chunks = []
    for path in sorted(DATA_DIR.glob("*.docx")):
        paragraphs = [p.text.strip() for p in Document(path).paragraphs if p.text.strip()]
        if not paragraphs:
            continue
        heading, body = paragraphs[0], paragraphs[1:]
        for i, text in enumerate(body):
            chunks.append({
                "id": f"{path.stem}#{i}",
                "heading": heading,
                "text": text,
                "indexed": f"{heading}. {text}",
            })
    return chunks


def split_sentences(text):
    """Break a paragraph into sentence units long enough to stand alone."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.split()) >= MIN_SENTENCE_WORDS]


def tokenize(text):
    """Lowercase and split on word characters, which is all BM25 needs here."""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_recall(index, chunks, query, k=RECALL_K):
    """Return the top k chunks by BM25 score, cheapest pass first.

    BM25 improves on TF-IDF in two places: term frequency saturates, so a word
    appearing a hundred times does not count ten times more than ten times, and
    scores are normalised by document length, so long chunks lose the advantage
    they get from simply containing more words.
    """
    scores = index.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], float(scores[i])) for i in order]


def rerank_sentences(encoder, query, candidates, k=FINAL_K):
    """Score every sentence in the recalled paragraphs, keep the best k.

    BM25 scores query and document apart and compares the two results. The
    cross-encoder reads them together in one forward pass, so it judges whether
    this text answers this question rather than whether it shares words with it.
    That is also why it is too slow for the whole corpus - hence stage one.

    Sentences rather than paragraphs, because a cross-encoder trained on short
    search passages loses the signal when one relevant clause is buried in four
    hundred characters of neighbouring policy. Step 4 measures that difference.
    """
    units = []
    for chunk, bm25_score in candidates:
        for sentence in split_sentences(chunk["text"]):
            units.append((chunk, bm25_score, sentence))
    if not units:
        return []
    scores = encoder.predict([(query, sentence) for _, _, sentence in units])
    ranked = sorted(zip(units, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, bm25_score, sentence, float(score))
            for (chunk, bm25_score, sentence), score in ranked[:k]]


def expand_query(api, query, count=EXPANSION_COUNT):
    """Ask the model for several phrasings of the same question.

    One phrasing is one throw of the dice: if the asker's wording misses the
    vocabulary the document used, BM25 returns nothing useful. Several phrasings
    are several throws, and the union of their hits is what gets reranked.
    """
    prompt = (
        f"Rewrite the question below as {count} alternative phrasings that a "
        "search index might match better. Vary the vocabulary - use synonyms a "
        "policy document would plausibly use. Keep the meaning identical.\n\n"
        f"Question: {query}\n\n"
        "Reply with a JSON array of strings and nothing else."
    )
    response = api.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        variants = json.loads(text)
    except json.JSONDecodeError:
        return []
    return [v for v in variants if isinstance(v, str)]


def multi_query_recall(index, chunks, queries, k=RECALL_K):
    """Recall for every phrasing, then merge on chunk id keeping the best score.

    Deduplication belongs here rather than after reranking: the same chunk
    surfacing under three phrasings would otherwise occupy three of the final
    slots and crowd out everything else.
    """
    best = {}
    for query in queries:
        for chunk, score in bm25_recall(index, chunks, query, k):
            if chunk["id"] not in best or score > best[chunk["id"]][1]:
                best[chunk["id"]] = (chunk, score)
    return sorted(best.values(), key=lambda x: x[1], reverse=True)


def short(text, width=70):
    """Trim text to one printable line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width - 1] + "…"


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Knowledge base not found at {DATA_DIR}")

    print("--- 1. Load the knowledge base ---")
    chunks = load_chunks()
    index = BM25Okapi([tokenize(c["indexed"]) for c in chunks])
    sentence_total = sum(len(split_sentences(c["text"])) for c in chunks)
    headings = sorted({c["heading"] for c in chunks})
    print(f"{len(chunks)} paragraph chunks ({sentence_total} sentence units) "
          f"from {len(headings)} documents:")
    for heading in headings:
        count = sum(1 for c in chunks if c["heading"] == heading)
        print(f"  {count:>2} x {heading}")

    print(f"\nloading {CROSS_ENCODER} …")
    from sentence_transformers import CrossEncoder
    encoder = CrossEncoder(CROSS_ENCODER, max_length=512)

    print("\n--- 2/3. Recall paragraphs with BM25, rerank sentences with the cross-encoder ---")
    for question in QUESTIONS:
        candidates = bm25_recall(index, chunks, question)
        final = rerank_sentences(encoder, question, candidates)
        recalled_ids = [c["id"] for c, _ in candidates]

        print(f"\n  Q: {question}")
        print(f"    stage 1 - BM25 kept {len(candidates)} of {len(chunks)} paragraphs")
        for rank, (chunk, score) in enumerate(candidates[:FINAL_K], 1):
            print(f"      {rank}. bm25 {score:6.2f}  {short(chunk['text'])}")
        print("    stage 2 - cross-encoder ranked the sentences inside them")
        for rank, (chunk, _, sentence, score) in enumerate(final, 1):
            was = recalled_ids.index(chunk["id"]) + 1
            print(f"      {rank}. cross {score:7.2f}  (from bm25 paragraph {was})")
            print(f"         {short(sentence, 74)}")

    print("\n--- 4. Why the unit matters ---")
    target = next((c["text"] for c in chunks if "date can be changed" in c["text"]), None)
    question = QUESTIONS[0]
    if target:
        sentence = next(s for s in split_sentences(target) if "date can be changed" in s)
        variants = [
            ("heading + whole paragraph", f"Shanghai Disney Resort Ticket Rules. {target}"),
            ("whole paragraph", target),
            ("the answering sentence", sentence),
        ]
        scores = encoder.predict([(question, text) for _, text in variants])
        print(f"  same question, same answer, three different units fed to the model:")
        for (label, text), score in zip(variants, scores):
            print(f"    {score:8.2f}  {label:<28} ({len(text)} chars)")
        print("  The answer never moved; only the amount of unrelated text around it")
        print("  did. Feed the reranker a paragraph and the one relevant clause is")
        print("  diluted by everything beside it, which is how the correct passage")
        print("  ends up ranked below a wrong one.")

    print("\n--- 5. Reading the scores ---")
    probe = [
        (question, "The date can be changed once, free of charge, up to 48 hours "
                   "before the visit."),
        (question, "The Eiffel Tower is a wrought-iron lattice tower in Paris."),
    ]
    for (_, passage), score in zip(probe, encoder.predict(probe)):
        print(f"  {score:8.2f}  {short(passage, 62)}")
    print("  These are unbounded logits, not probabilities. Note that the correct")
    print("  passage also scores negative here - so the sign is not a relevance")
    print("  threshold, and a raw value cannot be read as 'relevant' or 'not'.")
    print("  Only the ordering within one query on one corpus carries meaning.")
    print("  Scores from a different model, or a different chunk size, are not")
    print("  comparable to these.")

    print("\n--- 6. Query expansion, and what it is worth ---")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("  DEEPSEEK_API_KEY is not set - skipping expansion.")
        return
    api = OpenAI(api_key=api_key, base_url=BASE_URL)
    print("  Question 2 above failed, and it failed in stage one: the asker says")
    print("  'father', '68' and 'pay less'; the policy says 'aged 65 or over' and")
    print("  'senior rate'. Not one word in common, so BM25 never handed the right")
    print("  paragraph to the reranker. Expansion is the fix for exactly that.")

    for question in QUESTIONS:
        variants = expand_query(api, question)
        single = bm25_recall(index, chunks, question)
        multi = multi_query_recall(index, chunks, [question] + variants)
        gained = {c["id"] for c, _ in multi} - {c["id"] for c, _ in single}

        print(f"\n  Q: {question}")
        for v in variants:
            print(f"    + {short(v, 72)}")
        print(f"    recall: {len(single)} paragraphs -> {len(multi)} "
              f"({len(gained)} newly reachable)")
        for label, candidates in (("single", single), ("expanded", multi)):
            best = rerank_sentences(encoder, question, candidates, k=1)
            if best:
                _, _, sentence, score = best[0]
                print(f"    {label:<9} best answer {score:7.2f}  {short(sentence, 58)}")

    print("\n" + "=" * 76)
    print("Takeaway: the two stages are not interchangeable. BM25 is fast enough to")
    print("score every chunk and too shallow to rank the survivors; the cross-encoder")
    print("ranks well and is far too slow to run over everything. Expansion widens")
    print("what stage one can see, which pays off exactly when the asker's vocabulary")
    print("differs from the document's. And the reranker's granularity is a setting,")
    print("not a detail: paragraphs in, noise out.")
    print()
    print("One caveat worth keeping: on question 2 the correct sentence wins by")
    print("well under a tenth of a point over an unrelated one. That is a")
    print("coin-flip margin, not a verdict - a reranker that puts the right answer")
    print("first is not the same as a reranker that is confident about it.")


if __name__ == "__main__":
    main()
