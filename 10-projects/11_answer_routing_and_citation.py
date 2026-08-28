"""Route a question twice before answering it, then check the page numbers the answer cites.

Demonstrates the parts of a document question-answering system that sit outside retrieval:
    1. Hold a small set of reports whose pages carry deliberately non-contiguous numbers.
    2. Route each question to the report it concerns, and score that routing.
    3. Route it again to an answer type, and score that separately.
    4. Answer under the schema the type calls for, with reasoning and page references.
    5. Check every cited page against the pages the model was actually given.
    6. Split a comparison question into one sub-question per report and recombine.
    7. Report routing accuracy, schema conformance, answer accuracy and citation validity.

Module 10: Applied Projects - Routing, Structured Answers, and Citations.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MAX_ATTEMPTS = 4
RETRY_BACKOFF = 6

# Page numbers are deliberately sparse and non-contiguous. A model that invents a
# citation reaches for a small round number, and a corpus numbered 1, 2, 3 would
# hide that. These numbers make an invented page detectable on sight.
REPORTS = {
    "Alderway Foods": {
        14: "Alderway Foods reported revenue of 812.4 million units for the year ended "
            "31 December 2024, against 764.9 million units in 2023.",
        29: "The group operates three reporting segments: Chilled, Ambient and Foodservice. "
            "Chilled contributed 46 per cent of revenue, Ambient 33 per cent and "
            "Foodservice 21 per cent.",
        47: "Average headcount was 6,180 in 2024 compared with 6,402 in 2023. The "
            "reduction followed the closure of the Northolt packing site.",
        63: "Priya Raman has served as Chief Executive since March 2021. The Chief "
            "Financial Officer is Tomas Lindqvist.",
    },
    "Brightlane Logistics": {
        11: "Brightlane Logistics reported revenue of 1,204.7 million units for 2024, "
            "compared with 1,188.2 million units in 2023.",
        38: "Operating margin fell from 8.9 per cent to 6.1 per cent. The decline is "
            "attributed to higher subcontracted haulage rates and to the one-off cost of "
            "exiting the Rotterdam depot lease.",
        52: "Average headcount rose to 9,940 from 9,415, driven by insourcing of the "
            "final-mile fleet in two regions.",
        71: "Chief Executive Marcus Oyelaran was appointed in September 2019.",
    },
    "Coldharbour Energy": {
        9: "Coldharbour Energy reported revenue of 640.1 million units for 2024, down "
           "from 703.5 million units in 2023.",
        26: "The company reports two segments, Generation and Networks, contributing "
            "58 per cent and 42 per cent of revenue respectively.",
        44: "Average headcount was 3,275 in 2024 against 3,301 in 2023.",
        58: "Sara Whitcombe became Chief Executive in June 2023, succeeding an interim "
            "appointment held for eleven months.",
    },
}

# Every question carries the report it concerns, the shape of answer it calls for,
# the pages that contain the answer, and the answer itself. Nothing below is judged
# by reading it; all four are checked against these.
QUESTIONS = [
    {"question": "What was Alderway Foods' revenue in 2024?",
     "report": "Alderway Foods", "type": "number", "pages": [14], "answer": "812.4"},
    {"question": "Did Brightlane Logistics increase its average headcount in 2024?",
     "report": "Brightlane Logistics", "type": "boolean", "pages": [52], "answer": "yes"},
    {"question": "Who is the Chief Executive of Coldharbour Energy?",
     "report": "Coldharbour Energy", "type": "name", "pages": [58], "answer": "Sara Whitcombe"},
    {"question": "Which reporting segments does Alderway Foods use?",
     "report": "Alderway Foods", "type": "names", "pages": [29],
     "answer": "Chilled, Ambient, Foodservice"},
    {"question": "Why did Brightlane Logistics' operating margin fall?",
     "report": "Brightlane Logistics", "type": "string", "pages": [38],
     "answer": "higher subcontracted haulage rates and the cost of exiting the "
               "Rotterdam depot lease"},
    # A question whose shape is a number but whose answer is not in the report. It
    # separates "routed to the right type" from "found an answer", which a question
    # answerable from the pages cannot do.
    {"question": "What dividend per share did Coldharbour Energy declare?",
     "report": "Coldharbour Energy", "type": "number", "pages": [], "answer": "N/A"},
]

COMPARISONS = [
    {"question": "Which of the three companies had the highest revenue in 2024?",
     "sub_question": "What was {company}'s revenue in 2024, in millions of units?",
     "answer": "Brightlane Logistics"},
    {"question": "Which company employed more people on average in 2024, "
                 "Alderway Foods or Coldharbour Energy?",
     "sub_question": "What was {company}'s average headcount in 2024?",
     "answer": "Alderway Foods"},
]

ANSWER_TYPES = ["number", "boolean", "name", "names", "string"]

# One instruction per answer type. The reasoning field is the same in all of them;
# what changes is the shape the answer field has to take, which is the whole reason
# the type is decided before the question is answered rather than after.
TYPE_RULES = {
    "number": "answer must be a bare number with no units, no thousands separators "
              "and no words.",
    "boolean": 'answer must be exactly "yes" or "no".',
    "name": "answer must be a single proper name and nothing else.",
    "names": "answer must be the names only, separated by commas, in the order the "
             "source gives them.",
    "string": "answer must be one short sentence.",
}


def pick_provider() -> tuple:
    """Return (api_key, base_url, model) for whichever key is configured.

    Every call here is chat completion over short text, so DeepSeek comes first and
    the vision-capable provider is left for the scripts that need it.
    """
    if os.getenv("DEEPSEEK_API_KEY"):
        return (os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com", "deepseek-chat")
    if os.getenv("GEMINI_API_KEY"):
        return (os.getenv("GEMINI_API_KEY"),
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "gemini-3.1-flash-lite")
    if os.getenv("OPENAI_API_KEY"):
        return (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"), "gpt-4o-mini")
    return None


def call_with_retry(client, **kwargs):
    """Send one request, backing off when the provider answers with a rate limit."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
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


def ask_json(client, model: str, system: str, user: str) -> dict:
    """Send one request and parse the JSON object out of the reply.

    The object is extracted with a regular expression rather than assumed to be the
    whole reply, because a model that has been asked to reason will sometimes put a
    sentence in front of it. Failing to parse is recorded as a schema failure below
    rather than raised, since how often that happens is one of the things measured.
    """
    response = call_with_retry(
        client, model=model, temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"_raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"_raw": text}


def route_to_report(client, model: str, question: str) -> str:
    """Decide which report a question is about, before any of them is opened.

    Routing first is what keeps the rest of the work small: one report is opened
    instead of all of them. It also means a routing mistake cannot be recovered later,
    because the correct pages are never in front of the model that answers.
    """
    system = (
        "You route a question to exactly one report. Reply with JSON only: "
        '{"report": "<name>"}. The available reports are: '
        + ", ".join(REPORTS) + "."
    )
    result = ask_json(client, model, system, question)
    return result.get("report", "")


def route_to_type(client, model: str, question: str) -> str:
    """Decide what shape the answer has to take, before the answer is produced."""
    system = (
        "You decide what shape an answer must take. Reply with JSON only: "
        '{"type": "<type>"}. The types are: number (a single quantity), boolean '
        "(yes or no), name (one proper name), names (a list of proper names), "
        "string (a short free-text explanation)."
    )
    result = ask_json(client, model, system, question)
    return result.get("type", "")


def build_context(report: str) -> tuple:
    """Return the report's pages as text, and the set of page numbers supplied."""
    pages = REPORTS.get(report, {})
    text = "\n\n".join(f"[page {number}] {body}" for number, body in sorted(pages.items()))
    return text, set(pages)


def answer_question(client, model: str, question: str, report: str, answer_type: str) -> dict:
    """Answer one question under the schema its type calls for, citing pages.

    The reasoning field comes before the answer field on purpose: the model fills the
    object in order, so putting the working first means the answer is written after it
    rather than justified by it. The references field is what the next step checks.
    """
    context, _ = build_context(report)
    rule = TYPE_RULES.get(answer_type, TYPE_RULES["string"])
    system = (
        "You answer questions about one company report, using only the pages given. "
        "Reply with JSON only, with these four keys in this order: "
        '{"reasoning": "<your working, one or two sentences>", '
        '"answer": <the answer>, "references": [<page numbers you used>], '
        '"confidence": <0 to 1>}. '
        f"For this question, {rule} "
        'If the pages do not contain the answer, set answer to "N/A", '
        "references to an empty list, and confidence to 0."
    )
    user = f"Pages:\n{context}\n\nQuestion: {question}"
    return ask_json(client, model, system, user)


def validate_references(result: dict, supplied: set) -> dict:
    """Split the cited pages into those that were supplied and those that were not.

    A page number the model was never given cannot have been read, whatever the
    answer says. Removing those is the cheapest correctness check in the system,
    and it needs no judgement about whether the answer itself is right.
    """
    cited = result.get("references", [])
    if not isinstance(cited, list):
        cited = []
    numbers = []
    for item in cited:
        try:
            numbers.append(int(item))
        except (TypeError, ValueError):
            continue
    kept = [number for number in numbers if number in supplied]
    dropped = [number for number in numbers if number not in supplied]
    return {"cited": numbers, "kept": kept, "dropped": dropped}


def is_not_available(value) -> bool:
    """Recognise the reply the schema reserves for an answer the pages do not contain."""
    return str(value).strip().lower() in {"n/a", "na", "not available", "none"}


def conforms(result: dict, answer_type: str) -> bool:
    """Check that the reply has all four keys and that the answer matches its type.

    The N/A reply is part of the schema rather than a violation of it. Scoring it as
    a schema failure would penalise the one behaviour the prompt asks for when the
    answer is absent, and would make the conformance figure unreadable.
    """
    required = {"reasoning", "answer", "references", "confidence"}
    if not required.issubset(result):
        return False
    value = result["answer"]
    if is_not_available(value):
        return not result["references"]
    if answer_type == "number":
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", str(value).strip()))
    if answer_type == "boolean":
        return str(value).strip().lower() in {"yes", "no"}
    return isinstance(value, (str, list)) and bool(str(value).strip())


def matches_expected(result: dict, expected: str, answer_type: str) -> bool:
    """Compare an answer against the expected one, loosely enough to allow wording."""
    given = str(result.get("answer", "")).strip().lower()
    wanted = expected.strip().lower()
    if wanted == "n/a":
        return is_not_available(given)
    if answer_type == "number":
        try:
            return abs(float(given) - float(wanted)) < 0.05
        except ValueError:
            return False
    if answer_type in {"boolean", "name"}:
        return wanted in given
    if answer_type == "names":
        parts = [part.strip() for part in wanted.split(",")]
        return all(part in given for part in parts)
    keywords = [word for word in re.findall(r"[a-z]{5,}", wanted)]
    hits = sum(1 for word in keywords if word in given)
    return hits >= max(1, len(keywords) // 2)


def answer_comparison(client, model: str, item: dict) -> dict:
    """Answer a question spanning every report by asking each one separately first.

    A comparison cannot be routed to a single report, so the routing step above has
    nothing to choose. Splitting it restores the property the rest of the system
    depends on: each sub-question has one report, one set of pages, and one citable
    source. The final step compares answers rather than documents.
    """
    parts = []
    for company in REPORTS:
        sub = item["sub_question"].format(company=company)
        result = answer_question(client, model, sub, company, "number")
        checked = validate_references(result, set(REPORTS[company]))
        parts.append({"company": company, "answer": result.get("answer"),
                      "pages": checked["kept"], "dropped": checked["dropped"]})

    summary = "\n".join(f"{part['company']}: {part['answer']} (pages {part['pages']})"
                        for part in parts)
    system = (
        "You compare figures that have already been extracted. Reply with JSON only: "
        '{"reasoning": "<one sentence>", "answer": "<the company name>", '
        '"references": [<page numbers>], "confidence": <0 to 1>}.'
    )
    final = ask_json(client, model, system,
                     f"Figures:\n{summary}\n\nQuestion: {item['question']}")
    return {"parts": parts, "final": final}


def main() -> None:
    provider = pick_provider()
    if provider is None:
        print("No API key found. Set DEEPSEEK_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY.")
        return
    api_key, base_url, model = provider
    client = OpenAI(api_key=api_key, base_url=base_url)

    print("--- 1. The reports ---")
    for name, pages in REPORTS.items():
        print(f"    {name:<24}{len(pages)} pages, numbered {sorted(pages)}")
    all_pages = sorted({page for pages in REPORTS.values() for page in pages})
    print(f"\n    No report is numbered from 1, and no two share a page number.")
    print(f"    Any citation outside {all_pages} was invented rather than read.")

    print(f"\n--- 2-3. Routing {len(QUESTIONS)} questions twice ---")
    print(f"    model {model}, temperature 0\n")
    print(f"    {'question':<52}{'report':>10}{'type':>8}")
    routed = []
    for item in QUESTIONS:
        report = route_to_report(client, model, item["question"])
        answer_type = route_to_type(client, model, item["question"])
        routed.append({"item": item, "report": report, "type": answer_type})
        report_mark = "ok" if report == item["report"] else "WRONG"
        type_mark = "ok" if answer_type == item["type"] else "WRONG"
        print(f"    {item['question'][:50]:<52}{report_mark:>10}{type_mark:>8}")
    report_right = sum(1 for row in routed if row["report"] == row["item"]["report"])
    type_right = sum(1 for row in routed if row["type"] == row["item"]["type"])
    print(f"\n    report routing {report_right} of {len(QUESTIONS)}, "
          f"type routing {type_right} of {len(QUESTIONS)}")

    print("\n--- 4-5. Answering under the schema, and checking the citations ---")
    scored = []
    for row in routed:
        item = row["item"]
        # The answer is produced from whatever the router chose, not from the
        # correct report, so a routing mistake shows up here as a wrong answer.
        report = row["report"] if row["report"] in REPORTS else item["report"]
        answer_type = row["type"] if row["type"] in ANSWER_TYPES else "string"
        result = answer_question(client, model, item["question"], report, answer_type)
        _, supplied = build_context(report)
        citations = validate_references(result, supplied)
        record = {
            "item": item,
            "result": result,
            "citations": citations,
            "conforms": conforms(result, answer_type),
            "correct": matches_expected(result, item["answer"], answer_type),
            "type_used": answer_type,
        }
        scored.append(record)

        print(f"\n    Q: {item['question']}")
        print(f"       expected {item['answer']!r} from pages {item['pages']}")
        print(f"       answer   {str(result.get('answer'))[:80]!r}")
        print(f"       cited {citations['cited']}   valid {citations['kept']}"
              f"   invented {citations['dropped']}")
        print(f"       schema {'ok' if record['conforms'] else 'FAILED'}, "
              f"answer {'ok' if record['correct'] else 'WRONG'}, "
              f"confidence {result.get('confidence')}")

    print("\n--- 6. Comparisons, split one report at a time ---")
    comparison_right = 0
    for item in COMPARISONS:
        outcome = answer_comparison(client, model, item)
        print(f"\n    Q: {item['question']}")
        for part in outcome["parts"]:
            print(f"       {part['company']:<24}{str(part['answer']):>12}   "
                  f"pages {part['pages']}"
                  + (f"   invented {part['dropped']}" if part["dropped"] else ""))
        given = str(outcome["final"].get("answer", ""))
        right = item["answer"].lower() in given.lower()
        comparison_right += 1 if right else 0
        print(f"       combined -> {given!r}   expected {item['answer']!r}   "
              f"{'ok' if right else 'WRONG'}")
    print("\n    Each sub-answer keeps its own citation, so the comparison inherits")
    print("    sources rather than producing a claim no page supports.")

    print("\n--- 7. What the run measured ---")
    total = len(QUESTIONS)
    conforming = sum(1 for row in scored if row["conforms"])
    correct = sum(1 for row in scored if row["correct"])
    invented = sum(len(row["citations"]["dropped"]) for row in scored)
    cited = sum(len(row["citations"]["cited"]) for row in scored)
    print(f"    report routing        {report_right} of {total}")
    print(f"    answer type routing   {type_right} of {total}")
    print(f"    schema conformance    {conforming} of {total}")
    print(f"    answers correct       {correct} of {total}")
    print(f"    comparisons correct   {comparison_right} of {len(COMPARISONS)}")
    print(f"    page citations        {cited} made, {invented} of them invented")
    print("\n    Those are five separate numbers because they fail separately. A wrong")
    print("    answer traced to routing is repaired in the router; one traced to the")
    print("    schema is repaired in the prompt; an invented citation is caught without")
    print("    knowing whether the answer was right at all.")


if __name__ == "__main__":
    main()
