"""This script answers questions about three company reports with a hosted chat model: query
routing sends each question to a report and to an answer type, the model answers in
structured JSON with page citations, and every cited page is checked against the pages it
was given.

It covers the parts of a document question-answering system that sit outside retrieval:
    1. Hold a small set of reports whose pages carry deliberately non-contiguous numbers.
    2. Route each question to a report and to an answer type, and score the two separately.
    3. Answer under the schema the type calls for, and check every cited page.
    4. Feed the checks a wrong route and an invented page, and see which one catches each.
    5. Split a comparison question into one sub-question per report and recombine.
    6. Report the six scores: two routings, schema, answers, comparisons and citations.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

MAX_ATTEMPTS = 4
RETRY_BACKOFF = 6

# Page numbers are deliberately sparse and non-contiguous. In a corpus numbered 1,
# 2, 3 a guessed page number would very likely exist and pass as read; here most
# guesses land on a page that was never supplied, which the check below catches.
# The check proves a cited page was given to the model, not that it supports the answer.
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


def pick_provider() -> tuple | None:
    """Return (api_key, base_url, model) for whichever key is configured, DeepSeek first."""
    if os.getenv("DEEPSEEK_API_KEY"):
        return (os.getenv("DEEPSEEK_API_KEY"), "https://api.deepseek.com", "deepseek-chat")
    if os.getenv("GEMINI_API_KEY"):
        return (os.getenv("GEMINI_API_KEY"),
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "gemini-3.1-flash-lite")
    if os.getenv("OPENAI_API_KEY"):
        return (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"),
                os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    return None


def call_with_retry(client, **kwargs):
    """Send one request, backing off when the provider is rate-limited, busy or timing out."""
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
    """Send one request and parse the JSON object in the reply.

    A reply that does not parse comes back as {"_raw": text} and scores as a schema failure.
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
    """Decide which report a question is about, so only that report is opened.

    A wrong pick cannot be repaired later: the answering call never sees the right pages.
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

    Reasoning comes before answer, so the model writes the answer after its working.
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

    A page the model was never given cannot have been read, whether or not the answer is right.
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
    """Check that the reply has all four keys and that the answer has its type's basic shape.

    For name, names and string the check is only that the answer is non-empty.
    """
    required = {"reasoning", "answer", "references", "confidence"}
    if not required.issubset(result):
        return False
    value = result["answer"]
    # N/A is what the prompt asks for when the pages lack the answer, so it conforms.
    if is_not_available(value):
        return not result["references"]
    if answer_type == "number":
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", str(value).strip()))
    if answer_type == "boolean":
        return str(value).strip().lower() in {"yes", "no"}
    if isinstance(value, list):
        return bool(value)
    return isinstance(value, str) and bool(value.strip())


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
    if answer_type == "boolean":
        return given == wanted
    if answer_type == "name":
        return wanted in given
    if answer_type == "names":
        parts = [part.strip() for part in wanted.split(",")]
        return all(part in given for part in parts)
    keywords = [word for word in re.findall(r"[a-z]{5,}", wanted)]
    hits = sum(1 for word in keywords if word in given)
    return hits >= max(1, len(keywords) // 2)


def answer_comparison(client, model: str, item: dict) -> dict:
    """Answer a question spanning the reports by asking each report separately first.

    A comparison has no single report to route to; each sub-question has one, and its own pages.
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
    # The combined reply cites pages of its own. The only pages it can have read are
    # the ones the sub-answers kept, so that is what its citations are checked against.
    supplied = {page for part in parts for page in part["pages"]}
    return {"parts": parts, "final": final,
            "final_citations": validate_references(final, supplied)}


def main() -> None:
    provider = pick_provider()
    if provider is None:
        print("No API key found. Set DEEPSEEK_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY.")
        return
    api_key, base_url, model = provider
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 1. The reports
    print("--- 1. The reports ---")
    for name, pages in REPORTS.items():
        print(f"    {name:<24}{len(pages)} pages, numbered {sorted(pages)}")
    all_pages = sorted({page for pages in REPORTS.values() for page in pages})
    print("\n    No report is numbered from 1, and no two share a page number.")
    print(f"    Any citation outside {all_pages} was invented rather than read.")

    # 2. Routing each question to a report and to an answer type
    print(f"\n--- 2. Routing {len(QUESTIONS)} questions twice ---")
    print(f"    model {model}, temperature 0\n")
    print(f"    {'question':<66}{'report':>8}{'type':>8}")
    routed = []
    for item in QUESTIONS:
        report = route_to_report(client, model, item["question"])
        answer_type = route_to_type(client, model, item["question"])
        routed.append({"item": item, "report": report, "type": answer_type})
        report_mark = "ok" if report == item["report"] else "WRONG"
        type_mark = "ok" if answer_type == item["type"] else "WRONG"
        print(f"    {item['question']:<66}{report_mark:>8}{type_mark:>8}")
    report_right = sum(1 for row in routed if row["report"] == row["item"]["report"])
    type_right = sum(1 for row in routed if row["type"] == row["item"]["type"])
    print(f"\n    report routing {report_right} of {len(QUESTIONS)}, "
          f"type routing {type_right} of {len(QUESTIONS)}")

    # 3. Answering under the schema, and checking the citations
    print("\n--- 3. Answering under the schema, and checking the citations ---")
    scored = []
    for row in routed:
        item = row["item"]
        # The answer is produced from whatever the router chose, not from the
        # correct report, so a routing mistake propagates into answering instead of
        # being silently corrected. It usually surfaces as a wrong answer, though not
        # always: an N/A question can still come back N/A from the wrong pages. A
        # reply naming no known report opens no pages rather than falling back to
        # the expected report, which would feed the answer key into the system.
        report = row["report"]
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
        }
        scored.append(record)

        print(f"\n    Q: {item['question']}")
        print(f"       expected {item['answer']!r} from pages {item['pages']}")
        print(f"       answer   {str(result.get('answer'))!r}")
        print(f"       cited {citations['cited']}   valid {citations['kept']}"
              f"   invented {citations['dropped']}")
        print(f"       schema {'ok' if record['conforms'] else 'FAILED'}, "
              f"answer {'ok' if record['correct'] else 'WRONG'}, "
              f"confidence {result.get('confidence')}")

    # 4. A wrong route and an invented page, fed in on purpose
    print("\n--- 4. A wrong route and an invented page, fed in on purpose ---")
    probe = QUESTIONS[0]
    wrong_report = next(name for name in REPORTS if name != probe["report"])
    result = answer_question(client, model, probe["question"], wrong_report, probe["type"])
    _, supplied = build_context(wrong_report)
    misrouted = validate_references(result, supplied)
    misrouted_right = matches_expected(result, probe["answer"], probe["type"])
    print(f"    Q: {probe['question']}")
    print(f"       answered from the pages of {wrong_report}")
    print(f"       answer {str(result.get('answer'))!r}, "
          f"answer {'ok' if misrouted_right else 'WRONG'}, "
          f"invented {misrouted['dropped']}")
    # One page the model was given, one from another report, one from no report.
    invented_reply = {"references": [14, 11, 3]}
    invented = validate_references(invented_reply, set(REPORTS[probe["report"]]))
    print(f"\n    a reply citing {invented['cited']} for {probe['report']}: "
          f"valid {invented['kept']}, invented {invented['dropped']}")
    if not misrouted_right and not misrouted["dropped"]:
        print("\n    The wrong route is caught by the answer key, not by the citation check.")
    if invented["dropped"]:
        print("    The invented pages are caught by the citation check, with no answer key.")

    # 5. Comparisons, split one report at a time
    print("\n--- 5. Comparisons, split one report at a time ---")
    comparison_right = 0
    sourced = True
    for item in COMPARISONS:
        outcome = answer_comparison(client, model, item)
        print(f"\n    Q: {item['question']}")
        for part in outcome["parts"]:
            print(f"       {part['company']:<24}{str(part['answer']):>12}   "
                  f"pages {part['pages']}"
                  + (f"   invented {part['dropped']}" if part["dropped"] else ""))
            sourced = sourced and bool(part["pages"]) and not part["dropped"]
        given = str(outcome["final"].get("answer", ""))
        right = item["answer"].lower() in given.lower()
        comparison_right += 1 if right else 0
        print(f"       combined -> {given!r}   expected {item['answer']!r}   "
              f"{'ok' if right else 'WRONG'}")
        final_cites = outcome["final_citations"]
        print(f"       combined cites {final_cites['cited']}   valid {final_cites['kept']}"
              f"   invented {final_cites['dropped']}")
    if sourced:
        print("\n    Each sub-answer keeps its own citation, so the comparison inherits")
        print("    sources rather than producing a claim no page supports.")

    # 6. What the run measured
    print("\n--- 6. What the run measured ---")
    total = len(QUESTIONS)
    conforming = sum(1 for row in scored if row["conforms"])
    correct = sum(1 for row in scored if row["correct"])
    invented_count = sum(len(row["citations"]["dropped"]) for row in scored)
    cited = sum(len(row["citations"]["cited"]) for row in scored)
    print(f"    report routing        {report_right} of {total}")
    print(f"    answer type routing   {type_right} of {total}")
    print(f"    schema conformance    {conforming} of {total}")
    print(f"    answers correct       {correct} of {total}")
    print(f"    comparisons correct   {comparison_right} of {len(COMPARISONS)}")
    print(f"    page citations        {cited} made, {invented_count} of them invented")
    print("\n    Six numbers, because each one points to a different fix. A wrong answer")
    print("    traced to routing is repaired in the router; one traced to the schema is")
    print("    repaired in the prompt; an invented citation is caught without knowing")
    print("    whether the answer was right at all.")


if __name__ == "__main__":
    main()
