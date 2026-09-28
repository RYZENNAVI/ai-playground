"""This script answers three visitor questions from a small knowledge base of
Disney ticket rules, in two stages: recall, then reranking. Stage one, BM25,
scores every paragraph by the words it shares with the question. BM25 is TF-IDF
with two fixes: repeats of a word soon stop adding score, and long paragraphs
lose the edge of simply holding more words. It is cheap, so it keeps a wide set
of 8 paragraphs. Stage two, a cross-encoder, reads the question together with
each sentence of those paragraphs and keeps the best 3. It judges meaning rather
than shared words, but it costs one model pass per sentence, which does not scale
to a large corpus.

The run prints five parts:
    1. Loading the knowledge base. Each .docx file becomes paragraph chunks that
       carry the file's heading.
    2. Recall and rerank. BM25 keeps 8 paragraphs, and the cross-encoder ranks
       their sentences.
    3. Why the unit matters. Question 1 scored against the answering sentence,
       its paragraph, and the paragraph with its heading, then against the 8
       recalled paragraphs whole.
    4. Reading the scores. Question 1 against two hand-written sentences, one
       correct and one about the Eiffel Tower. The cross-encoder returns raw
       logits, so the correct sentence can score below zero.
    5. Query expansion, and what it is worth. The model rewrites each question
       four ways, BM25 recalls for all of them, and the reranker runs again.
       Needs DEEPSEEK_API_KEY or OPENAI_API_KEY.
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

DATA_DIR = Path(__file__).parent / "data" / "disney_kb"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# DeepSeek when its key is set, otherwise OpenAI. OPENAI_BASE_URL and
# OPENAI_MODEL point the OpenAI key at another compatible vendor.
if os.getenv("DEEPSEEK_API_KEY"):
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-chat"
else:
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Stage one keeps 8 paragraphs, stage two picks 3 sentences from all of their sentences.
RECALL_K = 8
FINAL_K = 3
EXPANSION_COUNT = 4
# Fragments shorter than this are list markers and sentence debris, not answers.
MIN_SENTENCE_WORDS = 6

QUESTIONS = [
    "Can I move my visit to a different day after buying?",
    "My father is 68. Does he pay less?",
    "How do I skip the queue on the busiest rides?",
]


def load_chunks():
    """Read each .docx file into paragraph chunks that carry the file's heading.
    A heading is never a chunk of its own, since a short title matches almost any question."""
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
    """Return the top k chunks by BM25 score."""
    scores = index.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(chunks[i], float(scores[i])) for i in order]


def rerank_sentences(encoder, query, candidates, k=FINAL_K):
    """Score every sentence of the recalled paragraphs against the query, keep the best k.
    Sentences, not paragraphs, because step 3 shows a paragraph dilutes the answering clause."""
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
    """Ask the model for several phrasings of the question, in words a policy document might use."""
    prompt = (
        f"Rewrite the question below as {count} alternative phrasings that a "
        "search index might match better. Vary the vocabulary: use synonyms a "
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
    """Recall for every phrasing and merge on chunk id, keeping the best score.
    Without the merge, a chunk found twice would repeat its sentences in the final slots."""
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

    # 1. Loading the knowledge base

    print("--- 1. Loading the knowledge base ---")
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

    # 2. Recall and rerank

    print("\n--- 2. Recall and rerank ---")
    for question in QUESTIONS:
        candidates = bm25_recall(index, chunks, question)
        final = rerank_sentences(encoder, question, candidates)
        recalled_ids = [c["id"] for c, _ in candidates]

        print(f"\n  Q: {question}")
        print(f"    stage 1: BM25 kept {len(candidates)} of {len(chunks)} paragraphs")
        for rank, (chunk, score) in enumerate(candidates[:FINAL_K], 1):
            print(f"      {rank}. bm25 {score:6.2f}  {short(chunk['text'])}")
        print("    stage 2: cross-encoder ranked the sentences inside them")
        for rank, (chunk, _, sentence, score) in enumerate(final, 1):
            was = recalled_ids.index(chunk["id"]) + 1
            print(f"      {rank}. cross {score:7.2f}  (from bm25 paragraph {was})")
            print(f"         {short(sentence, 74)}")

    # 3. Why the unit matters

    print("\n--- 3. Why the unit matters ---")
    target = next((c for c in chunks if "date can be changed" in c["text"]), None)
    question = QUESTIONS[0]
    if target:
        sentence = next(s for s in split_sentences(target["text"]) if "date can be changed" in s)
        variants = [
            ("heading + whole paragraph", target["indexed"]),
            ("whole paragraph", target["text"]),
            ("the answering sentence", sentence),
        ]
        scores = encoder.predict([(question, text) for _, text in variants])
        print("  same question, same answer, three different units fed to the model:")
        for (label, text), score in zip(variants, scores):
            print(f"    {score:8.2f}  {label:<28} ({len(text)} chars)")

        candidates = bm25_recall(index, chunks, question)
        scores = encoder.predict([(question, c["text"]) for c, _ in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        print(f"  if the reranker scored the {len(candidates)} recalled paragraphs whole "
              "instead of by sentence:")
        for (chunk, _), score in ranked[:FINAL_K]:
            mark = "  <- holds the answer" if chunk is target else ""
            print(f"    {score:8.2f}  {short(chunk['text'], 52)}{mark}")
        print("  The answer never moved; only the amount of unrelated text around it")
        print("  did. Feed the reranker a paragraph and the one relevant clause is")
        print("  diluted by everything beside it, which is how the paragraph holding")
        print("  the answer can end up below a wrong one.")

    # 4. Reading the scores

    print("\n--- 4. Reading the scores ---")
    probe = [
        (question, "The date can be changed once, free of charge, up to 48 hours "
                   "before the visit."),
        (question, "The Eiffel Tower is a wrought-iron lattice tower in Paris."),
    ]
    for (_, passage), score in zip(probe, encoder.predict(probe)):
        print(f"  {score:8.2f}  {short(passage, 62)}")
    print("  These are unbounded logits, not probabilities. The correct passage")
    print("  also scores negative here, so the sign is not a relevance threshold,")
    print("  and a raw value cannot be read as 'relevant' or 'not'. Only the")
    print("  ordering within one query on one corpus carries meaning. Scores from")
    print("  a different model, or a different chunk size, are not comparable to")
    print("  these.")

    # 5. Query expansion, and what it is worth

    print("\n--- 5. Query expansion, and what it is worth ---")
    api_key = API_KEY
    if not api_key:
        print("  (no DEEPSEEK_API_KEY or OPENAI_API_KEY, query expansion is skipped)")
        return
    api = OpenAI(api_key=api_key, base_url=BASE_URL)
    print("  Question 2 above failed, and it failed in stage one: the asker says")
    print("  'father', '68' and 'pay less'; the policy says 'aged 65 or over' and")
    print("  'senior rate'. No word that matters in common, so BM25 never handed")
    print("  the right paragraph to the reranker. Expansion is the fix for that.")

    leads = {}
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
            top = rerank_sentences(encoder, question, candidates, k=2)
            if top:
                _, _, sentence, score = top[0]
                print(f"    {label:<9} best answer {score:7.2f}  {short(sentence, 58)}")
        if len(top) == 2:
            leads[question] = top[0][3] - top[1][3]
            print(f"    {'':<9} ahead of the next sentence by {leads[question]:.2f}")

    print("\n" + "=" * 76)
    print("Takeaway: the two stages are not interchangeable. BM25 is fast enough to")
    print("score every chunk and too shallow to rank the survivors. The cross-encoder")
    print("ranks well, but it reads every pair in full, which is too slow for a large")
    print("corpus. Expansion widens what stage one can see, which pays off when the")
    print("asker's vocabulary differs from the document's. And the unit fed to the")
    print("reranker is a setting, not a detail.")
    if QUESTIONS[1] in leads:
        print()
        print(f"On question 2 the best sentence leads the next one by only "
              f"{leads[QUESTIONS[1]]:.2f}.")
        print("Ranking the right answer first is not the same as being confident")
        print("about it.")


if __name__ == "__main__":
    main()
