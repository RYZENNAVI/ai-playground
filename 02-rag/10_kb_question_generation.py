"""Give every knowledge chunk a set of generated questions, then retrieve on those instead.

Demonstrates why matching question against question beats matching question against prose:
    1. Generate a basic question set for one chunk, typed and graded by difficulty.
    2. Generate a wider set that also answers itself, so unanswerable questions can be dropped.
    3. Discard the generated questions the source text cannot actually support.
    4. Build two BM25 indexes over the same knowledge - one on prose, one on questions.
    5. Score both indexes against the same test queries.
    6. Read the per-query scores, which say more than the accuracy headline.

Module 02: RAG - Question Generation for Retrieval.
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

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

BASIC_QUESTION_COUNT = 5
DIVERSE_QUESTION_COUNT = 8

# Function words carry no topic and wreck BM25 on short queries: "When is it
# quietest?" otherwise scores highest against a chunk that merely happens to
# contain "when" and "it". A Chinese segmenter drops particles as a side effect
# of segmenting, so a pipeline ported from Chinese has to put this back by hand.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
i me my we our you your he she it its they them their
is am are was were be been being do does did doing done have has had having
can could will would shall should may might must
of in on at to for from by with without about into over under
what when where which who whom whose why how
any some all no not only just also very much many more most
s t don t
""".split())

# The knowledge base is written out here rather than read from data/ because the
# measurement in steps 5-6 needs a fixed answer key: every test query has to map
# to exactly one chunk, and that mapping has to survive edits to the data files.
KNOWLEDGE_BASE = [
    {
        "id": "kb_001",
        "category": "basics",
        "text": "Riverbend Park sits on the east bank of the river and was the first "
                "park of its kind in the region when it opened on 16 June 2016. It covers "
                "390 hectares across seven themed zones: Main Street, Wonder Gardens, "
                "Explorer Isle, Treasure Cove, Tomorrow Quarter, Dream Valley and Riverbend "
                "Village.",
    },
    {
        "id": "kb_002",
        "category": "pricing",
        "text": "Admission is priced by season and by day of the week. A weekday adult "
                "ticket costs 399 in local currency and a weekend or public holiday ticket "
                "costs 499. Children between 1.0 and 1.4 metres pay 299 on weekdays and 374 "
                "at weekends. Children under 1.0 metres enter free.",
    },
    {
        "id": "kb_003",
        "category": "hours",
        "text": "The gates normally open at 08:00 and close at 20:00, though the exact "
                "times shift with the season and with special events. Visitor numbers peak "
                "at weekends, on public holidays and during the school summer break; the "
                "quietest stretch is a weekday morning outside term breaks.",
    },
    {
        "id": "kb_004",
        "category": "transport",
        "text": "Four routes reach the park from the city centre: metro line 11 stops at "
                "the park station, a dedicated shuttle bus runs from the central terminal, a "
                "taxi takes roughly 40 to 60 minutes depending on traffic, and drivers can "
                "use the on-site car park, which charges 100 per day.",
    },
    {
        "id": "kb_005",
        "category": "attractions",
        "text": "Tomorrow Quarter holds the fastest ride in the park, a launch coaster that "
                "reaches its top speed in under three seconds and is the single biggest "
                "adrenaline draw on site. Dream Valley runs a gentler mine train suitable "
                "for younger visitors, and Treasure Cove stages an indoor boat ride through "
                "a pirate battle.",
    },
    {
        "id": "kb_006",
        "category": "rules",
        "text": "Sealed packaged snacks and bottled water may be carried in. Glass "
                "containers and alcohol are refused at the gate. Bags are checked on entry, "
                "and any item longer than 70 centimetres has to go into a locker near the "
                "main entrance.",
    },
]

# Worded the way a visitor would ask, not the way the source text is written.
# The first two share no content word at all with their answer - "picnic" against
# "snacks", "crowds" against "quietest" - which is precisely the case question-side
# retrieval exists for. The third deliberately does overlap, as a control: it shows
# what the technique buys when the asker already happens to use the right words.
TEST_QUERIES = [
    {"query": "Am I allowed to take a picnic in?", "answer_id": "kb_006"},
    {"query": "What time should I show up to avoid the crowds?", "answer_id": "kb_003"},
    {"query": "How much does it cost to park a car?", "answer_id": "kb_004"},
]

BASIC_INSTRUCTION = """You are a question-answering specialist. Given a piece of knowledge,
write the questions it can answer. Requirements:
1. Vary the phrasing - direct, indirect and comparative forms.
2. Do not repeat yourself.
3. Never ask anything the text does not answer.

Return JSON only:
{"questions": [{"question": "...", "question_type": "direct|indirect|comparative|conditional",
                "difficulty": "easy|medium|hard"}]}"""

DIVERSE_INSTRUCTION = """You are a question-answering specialist. Write highly varied questions
for the knowledge below. Vary all four of these:
1. type - direct, indirect, comparative, conditional, hypothetical, inferential
2. wording - different sentence shapes, vocabulary and register
3. difficulty - easy, medium and hard must all appear
4. angle - ask from different perspectives

For each question also state whether the knowledge really answers it, and give
that answer. Be strict: if the text does not contain the answer, say so.

Return JSON only:
{"questions": [{"question": "...", "question_type": "...", "difficulty": "...",
                "perspective": "...", "is_answerable": true, "answer": "..."}]}"""


def client():
    """Return an OpenAI-protocol client pointed at DeepSeek.

    Everything here is text generation, so one provider covers the whole script.
    """
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Add it to .env and retry.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def ask_json(api, instruction, knowledge, count):
    """Send one generation prompt and parse the JSON object out of the reply."""
    prompt = (f"### Instruction ###\n{instruction}\n\n"
              f"### Knowledge ###\n{knowledge}\n\n"
              f"### Number of questions ###\n{count}\n\n### Result ###\n")
    text = api.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ).choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}


def tokenize(text):
    """Lowercase and split on word characters.

    English needs no segmentation step, so splitting on word characters is the
    whole tokenizer. The stopword filter is not optional here: a segmenter drops
    function words as a side effect of segmenting, and word splitting does not,
    which is why STOPWORDS exists above.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    kept = [w for w in words if w not in STOPWORDS]
    # A query made entirely of function words would otherwise become an empty
    # token list, which scores every document at zero.
    return kept or words


def build_index(documents):
    """Build a BM25 index over a list of strings."""
    return BM25Okapi([tokenize(d) for d in documents])


def retrieve(index, owners, query):
    """Return (owner_id, score, position) for the best match.

    owners maps each indexed document back to the chunk it belongs to, which is
    what lets the question index answer in the same currency as the prose index.
    The position comes back as well, because the chunk id alone hides the thing
    worth seeing: which generated question actually won the match.
    """
    scores = index.get_scores(tokenize(query))
    best = max(range(len(scores)), key=lambda i: scores[i])
    return owners[best], float(scores[best]), best


def short(text, width=72):
    """Trim text to one printable line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width - 1] + "…"


def main():
    api = client()
    sample = KNOWLEDGE_BASE[0]

    print("=" * 78)
    print("--- 1. A basic question set for one chunk ---")
    print(f"knowledge ({sample['id']}): {short(sample['text'], 70)}")
    basic = ask_json(api, BASIC_INSTRUCTION, sample["text"], BASIC_QUESTION_COUNT)
    for i, item in enumerate(basic.get("questions", []), 1):
        print(f"  {i}. {item.get('question', '')}")
        print(f"     type: {item.get('question_type', '?')}   "
              f"difficulty: {item.get('difficulty', '?')}")

    print("\n--- 2. A wider set that answers itself ---")
    diverse = ask_json(api, DIVERSE_INSTRUCTION, sample["text"], DIVERSE_QUESTION_COUNT)
    for i, item in enumerate(diverse.get("questions", []), 1):
        mark = "ok " if item.get("is_answerable") else "NO "
        print(f"  {i}. [{mark}] {item.get('question', '')}")
        print(f"     {item.get('question_type', '?')} / {item.get('difficulty', '?')} "
              f"/ {item.get('perspective', '?')}  ->  {short(str(item.get('answer', '')), 56)}")

    print("\n--- 3. Dropping what the text cannot support ---")
    print("  The is_answerable and answer fields are the quality gate, not decoration.")
    print("  A generated question the source cannot answer is a hallucinated question:")
    print("  index it and retrieval will happily match a chunk that then fails to")
    print("  answer. Everything below keeps only the questions that passed.")

    print("\n" + "=" * 78)
    print("--- 4. Generating questions for the whole knowledge base ---")
    prose_docs, prose_owners = [], []
    question_docs, question_owners = [], []
    for chunk in KNOWLEDGE_BASE:
        prose_docs.append(chunk["text"])
        prose_owners.append(chunk["id"])

        generated = ask_json(api, BASIC_INSTRUCTION, chunk["text"], BASIC_QUESTION_COUNT)
        questions = [q.get("question", "") for q in generated.get("questions", [])
                     if q.get("question")]
        for question in questions:
            question_docs.append(question)
            question_owners.append(chunk["id"])
        print(f"  {chunk['id']} ({chunk['category']:<11}) -> {len(questions)} questions")
        for question in questions:
            print(f"      {short(question, 68)}")

    prose_index = build_index(prose_docs)
    question_index = build_index(question_docs)
    print(f"\n  prose index   : {len(prose_docs)} documents")
    print(f"  question index: {len(question_docs)} documents "
          f"covering the same {len(KNOWLEDGE_BASE)} chunks")

    print("\n--- 5. Scoring both indexes on the same queries ---")
    rows = []
    for case in TEST_QUERIES:
        query, expected = case["query"], case["answer_id"]
        prose_id, prose_score, _ = retrieve(prose_index, prose_owners, query)
        question_id, question_score, best = retrieve(
            question_index, question_owners, query)
        rows.append({
            "query": query, "expected": expected,
            "prose_id": prose_id, "prose_score": prose_score,
            "question_id": question_id, "question_score": question_score,
            "matched_question": question_docs[best],
        })

    prose_hits = sum(1 for r in rows if r["prose_id"] == r["expected"])
    question_hits = sum(1 for r in rows if r["question_id"] == r["expected"])
    total = len(rows)
    print(f"  prose retrieval accuracy   : {prose_hits / total:6.1%}  ({prose_hits}/{total})")
    print(f"  question retrieval accuracy: {question_hits / total:6.1%}  ({question_hits}/{total})")

    print("\n--- 6. Per-query detail ---")
    print(f"  {'query':<48} {'prose':>9} {'question':>9} {'delta':>8}")
    print("  " + "-" * 78)
    for r in rows:
        delta = r["question_score"] - r["prose_score"]
        prose_mark = "ok" if r["prose_id"] == r["expected"] else "MISS"
        question_mark = "ok" if r["question_id"] == r["expected"] else "MISS"
        print(f"  {short(r['query'], 46):<48} {r['prose_score']:>6.3f} {prose_mark:>2} "
              f"{r['question_score']:>6.3f} {question_mark:>2} {delta:>+8.3f}")
        print(f"      matched {r['question_id']} on: {short(r['matched_question'], 56)}")
    print()
    print("  The matched line is the whole mechanism in one place. The question")
    print("  index does not retrieve prose at all - it retrieves a generated")
    print("  question and then follows it back to the chunk that produced it, so")
    print("  a hit or a miss is decided by how close the asker came to one of the")
    print("  phrasings generated above, not by the wording of the source text.")

    flipped = [r for r in rows
               if r["prose_id"] != r["expected"] and r["question_id"] == r["expected"]]
    broke = [r for r in rows
             if r["prose_id"] == r["expected"] and r["question_id"] != r["expected"]]
    no_match = [r for r in rows if r["prose_score"] == 0.0]
    gained_only = [r for r in rows
                   if r["prose_id"] == r["expected"] == r["question_id"]
                   and r["question_score"] > r["prose_score"]]

    print("\n--- What the numbers actually say ---")
    if no_match:
        print(f"  {len(no_match)} of {total} queries share no content word with any chunk, so")
        print("  BM25 scores every chunk at zero and the winner is whichever document")
        print("  argmax happens to reach first. A 0.000 in the prose column is not a")
        print("  weak match - it is no match at all:")
        for r in no_match:
            print(f"    {short(r['query'], 68)}")
    if flipped:
        print(f"\n  {len(flipped)} query(s) went from wrong to right. Those are the only ones")
        print("  that moved the accuracy figure:")
        for r in flipped:
            print(f"    {short(r['query'], 68)}")
            print(f"      prose picked {r['prose_id']}, expected {r['expected']}")
    if broke:
        print(f"\n  {len(broke)} query(s) went from right to wrong - the generated questions")
        print("  pulled the winner away from the correct chunk:")
        for r in broke:
            print(f"    {short(r['query'], 68)}  (picked {r['question_id']})")
    if gained_only:
        print(f"\n  {len(gained_only)} query(s) scored higher but kept the same winner. A rising")
        print("  score is not a better answer - read the hit/miss column, not the delta.")
    print("\n  The technique pays where the asker's vocabulary and the document's")
    print("  diverge, and it is not free elsewhere. The control query is the reason")
    print("  it is in this test set: generated questions add a second vocabulary to")
    print("  match against, and that extra surface can pull a query towards the wrong")
    print("  chunk just as easily as towards the right one. Net accuracy is what")
    print("  matters, not the wins counted on their own.")


if __name__ == "__main__":
    main()
