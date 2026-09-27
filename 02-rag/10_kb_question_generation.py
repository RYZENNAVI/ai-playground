"""This script tries Doc2Query on a small knowledge base. Doc2Query writes the
questions each chunk can answer. Retrieval then matches a visitor's question
against those generated questions, not against the chunk text. Both sides are
searched with BM25, so the comparison is about wording alone.
The knowledge base is six chunks about an invented theme park: basics, prices,
opening hours, transport, rides and park rules. Three visitor questions each map
to one chunk. Two of them share no content word with any chunk. The third uses
the chunk's own words, as a control.

The run prints seven parts:
    1. A basic question set. DeepSeek writes 5 questions for the first chunk,
       each typed and graded by difficulty. This part is for comparison only and
       nothing later uses it: the questions come without answers, so none of
       them can be checked.
    2. A wider set that answers itself. 8 questions for the same chunk, each with
       an answer and a flag saying whether the text supports it. Parts 2 and 3
       show on one chunk what part 4 does to every chunk. Part 4 generates the
       questions again, so its count for the first chunk can differ from part 3's.
    3. Dropping what the text cannot support. The flagged questions from part 2.
    4. Questions for the whole knowledge base. The wider set for every chunk, with
       the flagged ones dropped, then two BM25 indexes: one on the six chunks and
       one on the kept questions.
    5. Scoring both indexes. The three visitor questions through each index.
    6. Per-query detail. The top score in each index, and the generated question
       each visitor question matched.
    7. What the numbers say. Which questions went from wrong to right, and which
       from right to wrong.
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

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

BASIC_QUESTION_COUNT = 5
DIVERSE_QUESTION_COUNT = 8

# Function words carry no topic and wreck BM25 on short queries: "When is it
# quietest?" otherwise scores highest against the park overview, which merely
# contains "when" and "it", and not against the opening hours.
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

# Every test query below maps to exactly one of these chunks. That mapping is the
# answer key steps 5 to 7 score against.
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
# The first two share no content word with their answer: "picnic" against
# "snacks", "crowds" against "quietest". The third is a control. It uses the
# chunk's own words, so it shows whether the technique hurts a query that
# already matches.
TEST_QUERIES = [
    {"query": "Am I allowed to take a picnic in?", "answer_id": "kb_006"},
    {"query": "What time should I show up to avoid the crowds?", "answer_id": "kb_003"},
    {"query": "How much does it cost to park a car?", "answer_id": "kb_004"},
]

BASIC_INSTRUCTION = """You write the questions a piece of knowledge can answer. Requirements:
1. Vary the phrasing: direct, indirect and comparative forms.
2. Do not repeat yourself.
3. Never ask anything the text does not answer.

Return JSON only:
{"questions": [{"question": "...", "question_type": "direct|indirect|comparative|conditional",
                "difficulty": "easy|medium|hard"}]}"""

DIVERSE_INSTRUCTION = """You write the questions a piece of knowledge can answer. Make them
highly varied. Vary all four of these:
1. type: direct, indirect, comparative, conditional, hypothetical, inferential
2. wording: different sentence shapes, vocabulary and register
3. difficulty: easy, medium and hard must all appear
4. angle: ask from different perspectives

For each question also state whether the knowledge really answers it, and give
that answer. Be strict: if the text does not contain the answer, say so.

Return JSON only:
{"questions": [{"question": "...", "question_type": "...", "difficulty": "...",
                "perspective": "...", "is_answerable": true, "answer": "..."}]}"""


def client():
    """Return an OpenAI-protocol client pointed at DeepSeek, the one provider this script uses."""
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
    """Lowercase, split into words and drop the stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    kept = [w for w in words if w not in STOPWORDS]
    # A query made entirely of function words would otherwise become an empty
    # token list, which scores every document at zero.
    return kept or words


def build_index(documents):
    """Build a BM25 index over a list of strings."""
    return BM25Okapi([tokenize(d) for d in documents])


def retrieve(index, owners, query):
    """Return (chunk id, score, position) of the best match. owners maps entries to chunks.
    The position says which generated question won; the chunk id alone would hide it."""
    scores = index.get_scores(tokenize(query))
    best = max(range(len(scores)), key=lambda i: scores[i])
    return owners[best], float(scores[best]), best


def short(text, width=72):
    """Trim text to one printable line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width - 1] + "…"


def plural(count):
    """Return "1 query" or "N queries"."""
    return f"{count} {'query' if count == 1 else 'queries'}"


def main():
    api = client()
    sample = KNOWLEDGE_BASE[0]

    # 1. A basic question set

    print("=" * 78)
    print("--- 1. A basic question set for one chunk ---")
    print(f"knowledge ({sample['id']}): {short(sample['text'], 70)}")
    basic = ask_json(api, BASIC_INSTRUCTION, sample["text"], BASIC_QUESTION_COUNT)
    for i, item in enumerate(basic.get("questions", []), 1):
        print(f"  {i}. {item.get('question', '')}")
        print(f"     type: {item.get('question_type', '?')}   "
              f"difficulty: {item.get('difficulty', '?')}")

    # 2. A wider set that answers itself

    print("\n--- 2. A wider set that answers itself ---")
    diverse = ask_json(api, DIVERSE_INSTRUCTION, sample["text"], DIVERSE_QUESTION_COUNT)
    items = diverse.get("questions", [])
    for i, item in enumerate(items, 1):
        mark = "ok " if item.get("is_answerable") else "NO "
        print(f"  {i}. [{mark}] {item.get('question', '')}")
        print(f"     {item.get('question_type', '?')} / {item.get('difficulty', '?')} "
              f"/ {item.get('perspective', '?')}  ->  {short(str(item.get('answer', '')), 56)}")

    # 3. Dropping what the text cannot support

    print("\n--- 3. Dropping what the text cannot support ---")
    flagged = sum(1 for item in items if not item.get("is_answerable"))
    print(f"  {flagged} of {len(items)} questions above are marked NO and are dropped.")
    print("  Indexed, such a question would send a visitor to a chunk that cannot")
    print("  answer it. Part 4 applies the same check to every chunk.")

    # 4. Questions for the whole knowledge base

    print("\n" + "=" * 78)
    print("--- 4. Questions for the whole knowledge base ---")
    prose_docs, prose_owners = [], []
    question_docs, question_owners = [], []
    for chunk in KNOWLEDGE_BASE:
        prose_docs.append(chunk["text"])
        prose_owners.append(chunk["id"])

        generated = ask_json(api, DIVERSE_INSTRUCTION, chunk["text"], DIVERSE_QUESTION_COUNT)
        generated = [q for q in generated.get("questions", []) if q.get("question")]
        questions = [q["question"] for q in generated if q.get("is_answerable")]
        for question in questions:
            question_docs.append(question)
            question_owners.append(chunk["id"])
        print(f"  {chunk['id']} ({chunk['category']:<11}) -> {len(generated)} generated, "
              f"{len(questions)} kept")
        for question in questions:
            print(f"      {short(question, 68)}")

    prose_index = build_index(prose_docs)
    question_index = build_index(question_docs)
    print(f"\n  prose index   : {len(prose_docs)} documents")
    print(f"  question index: {len(question_docs)} documents "
          f"covering the same {len(KNOWLEDGE_BASE)} chunks")

    # 5. Scoring both indexes

    print("\n--- 5. Scoring both indexes ---")
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

    # 6. Per-query detail

    print("\n--- 6. Per-query detail ---")
    print(f"  {'query':<48} {'prose':<11} {'question':<11}")
    print("  " + "-" * 72)
    for r in rows:
        prose_mark = "ok" if r["prose_id"] == r["expected"] else "MISS"
        question_mark = "ok" if r["question_id"] == r["expected"] else "MISS"
        print(f"  {short(r['query'], 46):<48} {r['prose_score']:6.3f} {prose_mark:<4} "
              f"{r['question_score']:6.3f} {question_mark:<4}")
        print(f"      matched {r['question_id']} on: {short(r['matched_question'], 56)}")
    print()
    print("  Each matched line is a generated question, not source text. The question")
    print("  index finds the closest generated question and returns the chunk it came")
    print("  from. A hit depends on how close the visitor came to one of those phrasings.")

    # 7. What the numbers say

    flipped = [r for r in rows
               if r["prose_id"] != r["expected"] and r["question_id"] == r["expected"]]
    broke = [r for r in rows
             if r["prose_id"] == r["expected"] and r["question_id"] != r["expected"]]
    no_match = [r for r in rows if r["prose_score"] == 0.0]

    print("\n--- 7. What the numbers say ---")
    if no_match:
        print(f"  {len(no_match)} of {total} queries share no content word with any chunk, so BM25")
        print("  scores every chunk at zero and max() returns the first one. A 0.000 in")
        print("  the prose column means nothing matched:")
        for r in no_match:
            print(f"    {short(r['query'], 68)}")
    if flipped:
        print(f"\n  {plural(len(flipped))} went from wrong to right:")
        for r in flipped:
            print(f"    {short(r['query'], 68)}")
            print(f"      prose picked {r['prose_id']}, expected {r['expected']}")
    if broke:
        print(f"\n  {plural(len(broke))} went from right to wrong. A generated question from")
        print("  another chunk matched better:")
        for r in broke:
            print(f"    {short(r['query'], 68)}  (picked {r['question_id']})")
    print("\n  Question retrieval pays where the visitor's words and the document's differ.")
    print("  The generated questions also bring words of their own, and those can match")
    print("  the wrong chunk too. Count the net change, not the wins alone.")


if __name__ == "__main__":
    main()
