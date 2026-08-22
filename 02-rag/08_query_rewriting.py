"""Rewrite a user's spoken-style question into something a retriever can actually match.

Demonstrates query rewriting, the step that decides whether retrieval starts on target:
    1. Rewrite a question that only makes sense against the conversation before it.
    2. Rewrite a comparison whose two sides were never named.
    3. Resolve pronouns that point at something said earlier.
    4. Split one turn that packs several independent questions.
    5. Strip the emotion out of a rhetorical question.
    6. Let the model classify the type and rewrite in a single call.
    7. Decide whether a question needs live data instead of the local index.
    8. Rewrite that question into search-engine form.
    9. Turn it into a concrete search plan.

Module 02: RAG - Query Rewriting.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

# Every rewriter below shares this frame; only the instruction changes. Keeping
# the frame fixed is what makes the five types comparable - any difference in
# the output comes from the instruction, not from how the context was fed in.
PROMPT_FRAME = """### Instruction ###
{instruction}

### Conversation history ###
{history}

### Current question ###
{query}

### Rewritten question ###
"""

# A short exchange the first five steps all rewrite against. Three turns is
# enough to create every ambiguity the types are meant to fix.
HISTORY = """User: I want to hear about the newest area at Riverbend Park.
Assistant: Riverbend Park just opened the Wildwood area, with a ranger station and a training camp.
User: What rides does that area have?
Assistant: Wildwood currently has the ranger station, the training camp and an ice cream parlour."""

COMPARISON_HISTORY = """User: I want to hear about the newest areas at Riverbend Park.
Assistant: Riverbend Park just opened Wildwood, and there is also the Skyline area."""

PRONOUN_HISTORY = """User: Tell me about the fireworks at Riverbend Park and Harbour Park.
Assistant: Both Riverbend Park and Harbour Park run a fireworks show."""

RHETORICAL_HISTORY = """User: I would like to book tickets for next Saturday.
Assistant: Checking now - next Saturday is sold out.
User: Sold out? A friend of mine walked up and bought one last week."""


def client():
    """Return an OpenAI-protocol client pointed at DeepSeek.

    One provider for the whole script: every step here is plain text work, so
    there is no reason to juggle a second key and a second rate limit.
    """
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Add it to .env and retry.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def ask(api, prompt, temperature=0):
    """Send one prompt and return the text, with temperature pinned to 0.

    Rewriting is a transformation, not a creative task; a stable temperature
    means two runs on the same question produce the same rewrite.
    """
    response = api.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def parse_json(text):
    """Pull a JSON object out of a reply that may be wrapped in a code fence.

    Models fence their JSON often enough that stripping the fence here is
    cheaper than fighting it in the prompt on every single call.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        cleaned = cleaned.rsplit("```", 1)[0]
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        return None


def rewrite(api, instruction, query, history=""):
    """Fill the shared frame with one instruction and return the rewrite."""
    return ask(api, PROMPT_FRAME.format(
        instruction=instruction, history=history or "(none)", query=query))


# --- 1. Context-dependent -------------------------------------------------
CONTEXT_INSTRUCTION = (
    "You are a query optimisation assistant. Read the current question together "
    "with the conversation before it and decide whether the question depends on "
    "that context. If it does, rewrite it as a standalone question carrying every "
    "piece of context it needs. If it does not, return the question unchanged. "
    "Answer with the rewritten question only."
)

# --- 2. Comparative -------------------------------------------------------
COMPARATIVE_INSTRUCTION = (
    "You are a query analyst. Identify the items the user is comparing, using the "
    "conversation for anything left unsaid, then rewrite the question so both "
    "sides of the comparison are named explicitly. Answer with the rewritten "
    "question only."
)

# --- 3. Ambiguous reference ----------------------------------------------
PRONOUN_INSTRUCTION = (
    "You are a disambiguation expert. Find what the pronouns and vague words in "
    "the question actually refer to, using the conversation history, and replace "
    "each of them with the concrete name. Answer with the rewritten question only."
)

# --- 4. Multi-intent ------------------------------------------------------
SPLIT_INSTRUCTION = (
    "You are a task splitter. Break the question into independent questions that "
    "can each be answered on their own. Reply with a JSON array of strings and "
    "nothing else, for example [\"question 1\", \"question 2\"]."
)

# --- 5. Rhetorical --------------------------------------------------------
RHETORICAL_INSTRUCTION = (
    "You are an intent reader. The user is asking rhetorically or venting. Work "
    "out the factual question underneath and restate it as a neutral question "
    "suitable for searching a knowledge base. Answer with the rewritten question only."
)

# --- 6. One call that does classification and rewriting together ----------
CLASSIFY_INSTRUCTION = """You are a query analyst. Classify the user's question as exactly one of:
1. context_dependent - leans on the previous turns, e.g. "any others", "what else"
2. comparative       - asks which is better or how two things differ
3. ambiguous_pronoun - contains "it", "they", "this", "both" pointing at something unnamed
4. multi_intent      - packs several independent questions into one turn
5. rhetorical        - phrased as a challenge or complaint rather than a question

When a question fits both multi_intent and ambiguous_pronoun, choose multi_intent.

Return JSON only:
{"query_type": "...", "rewritten_query": "...", "confidence": 0.0}"""

# --- 7-9. Live data -------------------------------------------------------
WEB_NEED_INSTRUCTION = """You are a query analyst. Decide whether answering needs a live web search
rather than a static knowledge base. A search is needed for:
1. time-sensitive wording - latest, today, now, currently
2. prices - how much, fare, fee
3. opening state - opening hours, closing time, whether it is open
4. events - shows, performances, festivals
5. weather
6. travel routes and transit
7. booking policy and availability
8. live conditions - queues, crowd levels

Return JSON only:
{"need_web_search": true, "search_reason": "...", "confidence": 0.0}"""

WEB_REWRITE_INSTRUCTION = """You are a search query specialist. Rewrite the question into the form a
search engine handles best:
1. name the place explicitly
2. state the time window
3. break the sentence into keywords
4. make the intent explicit
5. drop conversational filler
6. add close synonyms

Return JSON only:
{"rewritten_query": "...", "search_keywords": ["..."], "search_intent": "...",
 "suggested_sources": ["..."]}"""

WEB_STRATEGY_INSTRUCTION = """You are a search strategist. Build a search plan for the question.
Give at least three distinct extended keywords that do not repeat the primary
keywords, and name concrete site types rather than search engines.

Return JSON only:
{"primary_keywords": ["..."], "extended_keywords": ["..."],
 "search_platforms": ["..."], "time_range": "..."}"""

# The confidence the model reports is its own impression, not a distance in
# vector space, so it cannot carry a fine-grained decision. It is used here only
# as an on/off gate, which is the one job a soft score can still do honestly.
WEB_SEARCH_THRESHOLD = 0.7


def show(title, before, after):
    """Print one rewrite as a before/after pair."""
    print(f"\n{title}")
    print(f"  before: {before}")
    print(f"  after : {after}")


def main():
    api = client()

    print("=" * 78)
    print("--- 1. Context-dependent question ---")
    q = "Are there any other rides?"
    show("depends on three turns of history", q,
         rewrite(api, CONTEXT_INSTRUCTION, q, HISTORY))
    print("  why  : on its own this matches any chunk containing the word 'rides';")
    print("         naming the area and the three known rides narrows the target.")

    print("\n--- 2. Comparative question ---")
    q = "Which one takes longer and is more fun?"
    show("both sides were never named", q,
         rewrite(api, COMPARATIVE_INSTRUCTION, q, COMPARISON_HISTORY))

    print("\n--- 3. Ambiguous reference ---")
    q = "When do both of them start?"
    show("'both of them' points backwards", q,
         rewrite(api, PRONOUN_INSTRUCTION, q, PRONOUN_HISTORY))

    print("\n--- 4. Multi-intent question ---")
    q = "How much is a ticket? Do I need to book ahead? What does parking cost?"
    raw = rewrite(api, SPLIT_INSTRUCTION, q)
    parts = parse_json(raw) or [raw]
    print("\nsplit into independent questions")
    print(f"  before: {q}")
    for i, part in enumerate(parts, 1):
        print(f"  after {i}: {part}")
    print("  why  : this type alone returns a list, not a string. Everything")
    print("         downstream has to retrieve each part and merge the answers,")
    print("         so it changes the shape of the pipeline, not just the wording.")

    print("\n--- 5. Rhetorical question ---")
    q = "Don't tell me I have to book a month ahead as well?"
    show("emotion carries the sentence, not the request", q,
         rewrite(api, RHETORICAL_INSTRUCTION, q, RHETORICAL_HISTORY))
    print("  why  : vectorising the original spends most of its length on the")
    print("         complaint; the bookable fact is only a few words of it.")

    print("\n" + "=" * 78)
    print("--- 6. Classify and rewrite in one call ---")
    print(f"{'question':<46} {'type':<20} {'conf':>5}")
    print("-" * 78)
    samples = [
        ("Are there any other rides?", HISTORY),
        ("Which Riverbend Park area is more fun?", COMPARISON_HISTORY),
        ("Are they all suitable for small children?", PRONOUN_HISTORY),
        ("Which restaurants are there? What do they cost?", ""),
        ("Don't tell me this is another two-hour queue?", ""),
    ]
    rewrites = []
    for query, history in samples:
        parsed = parse_json(ask(api, PROMPT_FRAME.format(
            instruction=CLASSIFY_INSTRUCTION,
            history=history or "(none)", query=query))) or {}
        kind = parsed.get("query_type", "?")
        conf = parsed.get("confidence", 0)
        rewrites.append(parsed.get("rewritten_query", ""))
        print(f"{query:<46} {kind:<20} {float(conf):>5.2f}")
    print("-" * 78)
    for query, text in zip([s[0] for s in samples], rewrites):
        print(f"  {query}\n    -> {text}")
    print("\n  Watch what happens to the multi-intent sample. This schema declares")
    print("  rewritten_query as one string, so the two questions packed into that")
    print("  turn come back flattened into a single line - the structure has no")
    print("  room for the list step 4 produced. The classification is right and")
    print("  the rewrite is still wrong, because one output shape cannot serve")
    print("  five query types. That is the cost of folding both jobs into one")
    print("  call, and the reason step 4 handles this type on its own.")
    print("\n  Read the confidence column with care: the model produces it from")
    print("  its own impression of the answer, so a wrong classification can")
    print("  still come back at the same score as a right one. It ranks; it does")
    print("  not measure.")

    print("\n" + "=" * 78)
    print("--- 7. Does this question need live data? ---")
    live_queries = [
        "Is Riverbend Park open today, and how busy is it right now?",
        "How much is a Riverbend Park ticket next Saturday, and how far ahead must I book?",
    ]
    accepted = []
    for query in live_queries:
        verdict = parse_json(ask(api, PROMPT_FRAME.format(
            instruction=WEB_NEED_INSTRUCTION, history="(none)", query=query))) or {}
        conf = float(verdict.get("confidence", 0))
        needed = bool(verdict.get("need_web_search")) and conf >= WEB_SEARCH_THRESHOLD
        print(f"\n  {query}")
        print(f"    search: {needed}   confidence: {conf:.2f} "
              f"(gate at {WEB_SEARCH_THRESHOLD})")
        print(f"    reason: {verdict.get('search_reason', '')}")
        if needed:
            accepted.append(query)

    print("\n--- 8. Rewrite for a search engine ---")
    print("  Note this is a different target from steps 1-5: those produce a full")
    print("  sentence for vector retrieval, this produces keywords for a crawler.")
    strategies = []
    for query in accepted:
        rewritten = parse_json(ask(api, PROMPT_FRAME.format(
            instruction=WEB_REWRITE_INSTRUCTION, history="(none)", query=query))) or {}
        print(f"\n  {query}")
        print(f"    query   : {rewritten.get('rewritten_query', '')}")
        print(f"    keywords: {rewritten.get('search_keywords', [])}")
        print(f"    sources : {rewritten.get('suggested_sources', [])}")
        strategies.append(query)

    print("\n--- 9. Build a search plan ---")
    for query in strategies:
        plan = parse_json(ask(api, PROMPT_FRAME.format(
            instruction=WEB_STRATEGY_INSTRUCTION, history="(none)", query=query))) or {}
        primary = plan.get("primary_keywords", [])
        extended = plan.get("extended_keywords", [])
        print(f"\n  {query}")
        print(f"    primary : {primary}")
        print(f"    extended: {extended}")
        print(f"    sites   : {plan.get('search_platforms', [])}")
        print(f"    window  : {plan.get('time_range', '')}")
        # Guard rails, not decoration: asked to decompose a sentence into
        # keywords a model will happily hand the whole sentence back and leave
        # the extended list empty. Demanding "at least three distinct extended
        # keywords" in the instruction is what keeps that from happening, so the
        # checks below verify the instruction is still doing its job.
        if len(primary) == 1 and primary[0].strip().rstrip("?") == query.strip().rstrip("?"):
            print("    [weak] primary keywords are the original sentence verbatim")
        if not extended:
            print("    [weak] no extended keywords were produced")

    print("\n" + "=" * 78)
    print("Takeaway: rewriting is not free - each step above is one model call.")
    print("Short, literal questions retrieve fine without it, so gate it on the")
    print("value of the query rather than running it on everything.")


if __name__ == "__main__":
    main()
