"""Grow a knowledge base out of support conversations, then audit what it became.

Demonstrates the two halves of keeping a knowledge base alive:
    1. Extract structured knowledge points from a single conversation.
    2. Drop the points that record what someone wanted rather than what is true.
    3. Merge the survivors by type into fewer, fuller entries.
    4. Audit for coverage: which test questions the base cannot answer.
    5. Audit for staleness: which entries have gone out of date.
    6. Audit for contradictions: where the base disagrees with itself.
    7. Combine the three into one report.

Module 02: RAG - Knowledge Base Curation.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

# These two carry a visitor's wants, not facts about the park. Indexing them
# means a later question about ticket prices can retrieve "the visitor wanted to
# know the ticket price", which answers nothing and displaces something that would.
TRANSIENT_TYPES = {"need", "question"}

CONVERSATIONS = [
    [("user", "I want to visit Riverbend Park. What does a ticket cost?"),
     ("assistant", "A weekday adult ticket is 399 and a weekend or public holiday "
                   "ticket is 499. Children between 1.0 and 1.4 metres pay 299 on "
                   "weekdays and 374 at weekends. Under 1.0 metres is free."),
     ("user", "Do I need to book ahead?"),
     ("assistant", "Booking ahead is advisable, especially at weekends and on public "
                   "holidays. Book through the park website or an authorised reseller."),
     ("user", "How do I get there from the airport?"),
     ("assistant", "Take metro line 2 to Riverside, change to line 11, and get off at "
                   "the park station - about an hour in total. A taxi takes about 40 "
                   "minutes.")],
    [("user", "Is the park open today and how busy is it?"),
     ("assistant", "The park opens daily from 08:00 to 20:00. Crowds peak at weekends, "
                   "on public holidays and during the school summer break."),
     ("user", "Which rides are worth queueing for?"),
     ("assistant", "The launch coaster in Tomorrow Quarter, the mine train in Dream "
                   "Valley, and the indoor boat ride in Treasure Cove."),
     ("user", "Anything I should know before I go?"),
     ("assistant", "Check the opening hours on the app before travelling, and use the "
                   "app to see live queue times.")],
    [("user", "What does parking cost?"),
     ("assistant", "The on-site car park charges 100 per day. It fills up early, so "
                   "the metro is often the easier option."),
     ("user", "Can I take food in?"),
     ("assistant", "Sealed packaged snacks and bottled water are fine. Glass and "
                   "alcohol are refused at the gate. Food inside the park is expensive, "
                   "so bringing your own is worth it."),
     ("user", "I'm taking a toddler - anything else?"),
     ("assistant", "Pushchairs can be hired near the main entrance, and some rides "
                   "have a minimum height.")],
]

# The audit half runs against a base seeded with three planted defects, because a
# checker that reports nothing proves nothing. kb_002 and kb_005 disagree about
# the parking charge, kb_004 describes an event that has long since finished, and
# nothing at all covers the pet question in AUDIT_QUERIES.
SEEDED_BASE = [
    {"id": "kb_001", "text": "Riverbend Park opens daily from 08:00 to 20:00."},
    {"id": "kb_002", "text": "The on-site car park charges 100 per day."},
    {"id": "kb_003", "text": "A weekday adult ticket is 399; weekends and public "
                             "holidays are 499."},
    {"id": "kb_004", "text": "The winter lantern festival runs from 1 December 2023 "
                             "to 5 January 2024, with late opening until 22:00."},
    {"id": "kb_005", "text": "Parking is charged at 150 per day at the main car park."},
    {"id": "kb_006", "text": "Sealed snacks and bottled water may be brought in. Glass "
                             "and alcohol are refused at the gate."},
]

AUDIT_QUERIES = [
    "How much is parking?",
    "Can I bring my dog?",
    "What time does the park close?",
]

EXTRACT_INSTRUCTION = """You are a knowledge extraction specialist. Pull the reusable
knowledge out of the conversation below. Cover:
1. factual information - places, times, prices, rules
2. what the visitor wanted
3. questions asked and answered
4. procedures and step-by-step routes
5. cautions and reminders

Classify each point as exactly one of: fact, need, question, process, caution.

Return JSON only:
{"extracted_knowledge": [{"knowledge_type": "...", "content": "...",
                          "confidence": 0.0, "source": "user|assistant",
                          "keywords": ["..."], "category": "..."}],
 "conversation_summary": "...", "user_intent": "..."}"""

MERGE_INSTRUCTION = """You are a knowledge editor. Merge the {kind} points below into one
fuller point. Rules:
1. keep every piece of information - lose nothing
2. remove duplication and fold similar phrasings together
3. stay accurate and complete
4. keep the result readable
5. take the highest confidence of the inputs

Return JSON only:
{{"knowledge_type": "{kind}", "content": "...", "confidence": 0.0,
  "keywords": ["..."], "category": "...", "frequency": {count}}}"""

COVERAGE_INSTRUCTION = """You are a knowledge base completeness auditor. Given the entries
and the test questions, decide which questions the entries cannot answer.

Judge only whether an answer exists somewhere in the entries. Do not judge
whether that answer is correct, current or agreed upon. In particular:
1. an answer that another entry contradicts still counts as present - that is a
   consistency problem, and the consistency check owns it
2. an answer that looks out of date still counts as present - that is a
   freshness problem, and the freshness check owns it

Report a question as missing only when no entry answers it at all.

Return JSON only:
{"missing_knowledge": [{"query": "...", "missing_aspect": "...",
                        "importance": "high|medium|low", "suggested_content": "..."}],
 "coverage_score": 0.0}"""

FRESHNESS_INSTRUCTION = """You are a knowledge base freshness auditor. Today's date is
{today}. Decide which entries have gone out of date - expired dates and periods,
prices likely to have moved, superseded rules, finished events.

Return JSON only:
{{"outdated_knowledge": [{{"chunk_id": "...", "outdated_aspect": "...",
                          "severity": "high|medium|low", "suggested_update": "..."}}],
  "freshness_score": 0.0}}"""

CONSISTENCY_INSTRUCTION = """You are a knowledge base consistency auditor. Find entries that
contradict each other. Every conflict must name at least two entry ids.

Return JSON only:
{"conflicting_knowledge": [{"conflict_type": "...", "chunk_ids": ["...", "..."],
                            "conflicting_content": ["..."],
                            "severity": "high|medium|low",
                            "resolution_suggestion": "..."}],
 "consistency_score": 0.0}"""


def client():
    """Return an OpenAI-protocol client pointed at DeepSeek.

    Every step here is text in, structured text out, so one provider covers it.
    """
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Add it to .env and retry.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def ask_json(api, prompt):
    """Send one prompt and parse the JSON object out of the reply."""
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


def render(turns):
    """Format a conversation for the prompt."""
    return "\n".join(f"{speaker}: {text}" for speaker, text in turns)


def extract(api, turns):
    """Step 1: turn one conversation into typed knowledge points."""
    return ask_json(api, f"### Instruction ###\n{EXTRACT_INSTRUCTION}\n\n"
                         f"### Conversation ###\n{render(turns)}\n\n### Result ###\n")


def merge_group(api, kind, points):
    """Step 3: fold one type's points into a single entry."""
    if len(points) == 1:
        return points[0]
    body = "\n".join(f"- {p.get('content', '')}" for p in points)
    prompt = (MERGE_INSTRUCTION.format(kind=kind, count=len(points))
              + f"\n\n### Points to merge ###\n{body}\n\n### Result ###\n")
    merged = ask_json(api, prompt)
    return merged or points[0]


def audit(api, entries, queries, today):
    """Steps 4-6: three independent checks over the same entries.

    They are separate calls because their inputs genuinely differ. Coverage needs
    the test questions, and cannot be derived from the entries alone - a gap is
    only a gap relative to something someone wanted to ask. Freshness needs the
    current date, because the model has no clock. Consistency needs neither: a
    contradiction is visible inside the entries themselves.

    Separate inputs do not by themselves keep the three verdicts apart. A
    coverage check shown a base that contradicts itself will report the
    contradicted topic as a gap unless it is told not to, which is why
    COVERAGE_INSTRUCTION spends four lines saying what is not a gap. That
    boundary lives in the prompt, so it holds most of the time rather than
    always - the overlap warning further down stays in for the rest.
    """
    body = "\n".join(f"{e['id']}: {e['text']}" for e in entries)
    questions = "\n".join(f"- {q}" for q in queries)

    coverage = ask_json(api, f"### Instruction ###\n{COVERAGE_INSTRUCTION}\n\n"
                             f"### Entries ###\n{body}\n\n"
                             f"### Test questions ###\n{questions}\n\n### Result ###\n")
    freshness = ask_json(api, "### Instruction ###\n"
                              + FRESHNESS_INSTRUCTION.format(today=today)
                              + f"\n\n### Entries ###\n{body}\n\n### Result ###\n")
    consistency = ask_json(api, f"### Instruction ###\n{CONSISTENCY_INSTRUCTION}\n\n"
                                f"### Entries ###\n{body}\n\n### Result ###\n")
    return coverage, freshness, consistency


def as_score(value):
    """Read a score that may arrive as a number or as a string."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def short(text, width=86):
    """Trim text to one printable line."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[:width - 1] + "…"


def main():
    api = client()

    print("=" * 92)
    print("--- 1. Extracting knowledge from one conversation ---")
    first = extract(api, CONVERSATIONS[0])
    points = first.get("extracted_knowledge", [])
    for i, point in enumerate(points, 1):
        print(f"  {i}. [{point.get('knowledge_type', '?'):<8}] "
              f"conf {as_score(point.get('confidence')) or 0:.2f}  "
              f"{short(point.get('content', ''), 62)}")
    print(f"\n  summary: {short(first.get('conversation_summary', ''))}")
    print(f"  intent : {short(first.get('user_intent', ''))}")

    print("\n--- Extracting from the rest ---")
    harvested = list(points)
    for turns in CONVERSATIONS[1:]:
        found = extract(api, turns).get("extracted_knowledge", [])
        harvested.extend(found)
        print(f"  +{len(found)} points")
    print(f"  {len(harvested)} points in total")

    print("\n" + "=" * 92)
    print("--- 2. Dropping what is not knowledge ---")
    keep, drop = [], []
    for point in harvested:
        (drop if point.get("knowledge_type") in TRANSIENT_TYPES else keep).append(point)
    print(f"  {len(harvested)} -> {len(keep)}  ({len(drop)} dropped)")
    for point in drop:
        print(f"    drop [{point.get('knowledge_type')}] {short(point.get('content'), 66)}")
    print("\n  These record what somebody wanted, not what is true. Left in the index,")
    print("  a later question about ticket prices can retrieve 'the visitor wanted to")
    print("  know the ticket price' - which answers nothing and takes a slot that")
    print("  something useful would have filled.")

    print("\n--- 3. Merging by type ---")
    grouped = defaultdict(list)
    for point in keep:
        grouped[point.get("knowledge_type", "other")].append(point)
    merged = []
    for kind, group in sorted(grouped.items()):
        entry = merge_group(api, kind, group)
        merged.append(entry)
        print(f"\n  {kind}: {len(group)} points -> 1 "
              f"(confidence {as_score(entry.get('confidence')) or 0:.2f})")
        print(f"    {short(entry.get('content', ''), 84)}")
    print(f"\n  {len(keep)} points became {len(merged)} entries. Short fragments answer")
    print("  a question only partly; one entry that holds the whole topic answers it")
    print("  in a single retrieval.")

    print("\n" + "=" * 92)
    today = date.today().isoformat()
    print(f"--- 4-6. Auditing a knowledge base (today is {today}) ---")
    print("  The base below has three defects planted in it on purpose, because an")
    print("  auditor that finds nothing has demonstrated nothing:")
    for entry in SEEDED_BASE:
        print(f"    {entry['id']}: {short(entry['text'], 78)}")

    coverage, freshness, consistency = audit(api, SEEDED_BASE, AUDIT_QUERIES, today)

    print("\n  4. Coverage")
    missing = coverage.get("missing_knowledge", [])
    print(f"     score {as_score(coverage.get('coverage_score')) or 0:.2f}, "
          f"{len(missing)} gap(s)")
    for gap in missing:
        print(f"     - {short(gap.get('query'), 46):<48} "
              f"[{gap.get('importance', '?')}] {short(gap.get('missing_aspect'), 34)}")

    print("\n  5. Freshness")
    stale = freshness.get("outdated_knowledge", [])
    print(f"     score {as_score(freshness.get('freshness_score')) or 0:.2f}, "
          f"{len(stale)} stale entry(s)")
    for item in stale:
        print(f"     - {item.get('chunk_id', '?'):<8} [{item.get('severity', '?')}] "
              f"{short(item.get('outdated_aspect'), 62)}")

    print("\n  6. Consistency")
    clashes = consistency.get("conflicting_knowledge", [])
    print(f"     score {as_score(consistency.get('consistency_score')) or 0:.2f}, "
          f"{len(clashes)} conflict(s)")
    for clash in clashes:
        ids = clash.get("chunk_ids", [])
        print(f"     - {', '.join(ids):<18} [{clash.get('severity', '?')}] "
              f"{short(clash.get('conflict_type'), 50)}")
        if len(ids) < 2:
            print("       [suspect] a contradiction needs two sides; this names one")

    print("\n--- 7. Report ---")
    scores = {"coverage": as_score(coverage.get("coverage_score")),
              "freshness": as_score(freshness.get("freshness_score")),
              "consistency": as_score(consistency.get("consistency_score"))}
    present = {k: v for k, v in scores.items() if v is not None}
    for name, value in scores.items():
        print(f"    {name:<12} {value if value is not None else float('nan'):.2f}")
    if present:
        overall = sum(present.values()) / len(present)
        print(f"    {'overall':<12} {overall:.2f}")

    found = {"coverage": bool(missing), "freshness": bool(stale),
             "consistency": bool(clashes)}
    print(f"\n  planted defects detected: "
          f"{sum(found.values())}/3  ({', '.join(k for k, v in found.items() if v) or 'none'})")
    missed = [k for k, v in found.items() if not v]
    if missed:
        print(f"  not detected: {', '.join(missed)} - the check ran and found nothing,")
        print("  which on a base with a known defect means the check missed it.")

    if len(stale) > len(SEEDED_BASE) / 2:
        print(f"\n  Note the freshness check flagged {len(stale)} of {len(SEEDED_BASE)} entries.")
        print("  Read those findings before trusting the count: some will be entries")
        print("  that merely could change one day, and some will be the conflicting")
        print("  pair reported again under a heading that belongs to the consistency")
        print("  check. The three audits overlap in practice even though their inputs")
        print("  do not, so a low freshness score can be measuring the wrong thing.")

    if len(set(present.values())) == 1 and len(present) > 1:
        print("\n  Note all three scores came back identical. Three independent checks")
        print("  agreeing to two decimal places is a property of how the model picks")
        print("  numbers, not a measurement of three separate things. Use these to")
        print("  rank one base against itself over time, never as a target to hit.")
    print("\n  What the checks can and cannot do: they are the only part of this")
    print("  script with an answer key, and even here the scores are impressions")
    print("  while the findings are checkable. Trust the findings list; treat the")
    print("  numbers beside it as decoration.")


if __name__ == "__main__":
    main()
