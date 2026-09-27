"""This script rewrites questions a visitor might ask a park assistant into
questions a retriever can use. Parts 1 to 6 use a made-up park, Riverbend Park,
and most of their questions come with the short conversation they depend on.
Parts 7 to 9 ask about a real park, Shanghai Disneyland, so that part 8 can run
a real web search. DeepSeek does every rewrite, and all prompts share one frame:
instruction, conversation history, current question. Only part 8 retrieves
anything: it sends the original question and its rewrite to the Tavily search
API, when TAVILY_API_KEY is set.

The run prints nine parts:
    1. Context-dependent question. "Are there any other rides?" rewritten with the
       area and the rides named earlier in the conversation.
    2. Comparative question. The two areas being compared, named.
    3. Ambiguous reference. "both of them" replaced by the two fireworks shows.
    4. Multi-intent question. Three questions in one turn, split into a JSON list.
    5. Rhetorical question. The booking question inside a complaint.
    6. Classify and rewrite in one call. Five samples, one of each type, with the
       type and confidence the model reports. The multi-intent sample comes back
       as one string, because this output has room for only one.
    7. Live data. Whether each of three questions needs a web search, gated at a
       confidence of 0.7. One of them can be answered from the knowledge base.
    8. Search engine rewrite. Each question that passes part 7, turned into a
       keyword query with suggested sources. Tavily searches the original
       question and the rewrite, and the top three results of each are printed
       side by side.
    9. Search plan. The same questions, from the original wording, turned into
       primary and extended keywords, site types and a time window, with checks
       that the keywords are not just the original sentence.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
TAVILY_URL = "https://api.tavily.com/search"
SEARCH_RESULTS = 3

# Every rewriter shares this frame and only the instruction changes, so a
# difference in output comes from the instruction.
PROMPT_FRAME = """### Instruction ###
{instruction}

### Conversation history ###
{history}

### Current question ###
{query}

### Rewritten question ###
"""

# The conversation for part 1 and the first sample in part 6.
HISTORY = """User: I want to hear about the newest area at Riverbend Park.
Assistant: Riverbend Park just opened the Wildwood area, with a ranger station and a training camp.
User: What rides does that area have?
Assistant: Wildwood currently has the ranger station, the training camp and an ice cream parlour."""

COMPARISON_HISTORY = """User: I want to hear about the newest areas at Riverbend Park.
Assistant: Riverbend Park just opened Wildwood, and there is also the Skyline area."""

# "both shows" is the last plural before the question. When the reply read "Both
# Riverbend Park and Harbour Park run a fireworks show", the rewrite often named
# the two parks instead of the shows.
PRONOUN_HISTORY = """User: Tell me about the fireworks at Riverbend Park and Harbour Park.
Assistant: Riverbend Park and Harbour Park each run a fireworks show, and both shows are popular."""

RHETORICAL_HISTORY = """User: I would like to book tickets for next Saturday.
Assistant: Checking now - next Saturday is sold out.
User: Sold out? A friend of mine walked up and bought one last week."""


def client():
    """Return an OpenAI-protocol client for DeepSeek, which does every step."""
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in .env and retry.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def ask(api, prompt, temperature=0):
    """Send one prompt at temperature 0 and return the text. Rewriting is not a
    creative task."""
    response = api.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def parse_json(text):
    """Return the JSON in a reply, with any code fence removed, or None."""
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


def web_search(query, key):
    """Return (title, url) for Tavily's top results. Plain HTTP, so no package is needed."""
    body = json.dumps({"query": query, "max_results": SEARCH_RESULTS}).encode()
    request = urllib.request.Request(TAVILY_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        results = json.loads(response.read())["results"]
    return [(result["title"], result["url"]) for result in results]


def rewrite(api, instruction, query, history=""):
    """Fill the shared frame with one instruction and return the rewrite."""
    return ask(api, PROMPT_FRAME.format(
        instruction=instruction, history=history or "(none)", query=query))


# Instructions for parts 1 to 5, one per question type.
CONTEXT_INSTRUCTION = (
    "You are a query optimisation assistant. Read the current question together "
    "with the conversation before it and decide whether the question depends on "
    "that context. If it does, rewrite it as a standalone question carrying every "
    "piece of context it needs. If it does not, return the question unchanged. "
    "Answer with the rewritten question only."
)

COMPARATIVE_INSTRUCTION = (
    "You are a query analyst. Identify the items the user is comparing, using the "
    "conversation for anything left unsaid, then rewrite the question so both "
    "sides of the comparison are named explicitly. Answer with the rewritten "
    "question only."
)

PRONOUN_INSTRUCTION = (
    "You are a disambiguation expert. Find what the pronouns and vague words in "
    "the question actually refer to, using the conversation history, and replace "
    "each of them with the concrete name. Answer with the rewritten question only."
)

SPLIT_INSTRUCTION = (
    "You are a task splitter. Break the question into independent questions that "
    "can each be answered on their own. Reply with a JSON array of strings and "
    "nothing else, for example [\"question 1\", \"question 2\"]."
)

RHETORICAL_INSTRUCTION = (
    "You are an intent reader. The user is asking rhetorically or venting. Work "
    "out the factual question underneath and restate it as a neutral question "
    "suitable for searching a knowledge base. Answer with the rewritten question only."
)

# Part 6: one call that classifies and rewrites.
CLASSIFY_INSTRUCTION = """You are a query analyst. Classify the user's question as exactly one of:
1. context_dependent - leans on the previous turns, e.g. "any others", "what else"
2. comparative       - asks which is better or how two things differ
3. ambiguous_pronoun - contains "it", "they", "this", "both" pointing at something unnamed
4. multi_intent      - packs several independent questions into one turn
5. rhetorical        - phrased as a challenge or complaint rather than a question

When a question fits both multi_intent and ambiguous_pronoun, choose multi_intent.

Return JSON only:
{"query_type": "...", "rewritten_query": "...", "confidence": 0.0}"""

# Parts 7 to 9: questions that may need live data, about a real park.
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

# The confidence is the model's own impression, not a measurement, so it is used
# only as an on/off gate.
WEB_SEARCH_THRESHOLD = 0.7


def show(title, before, after):
    """Print one rewrite as a before/after pair."""
    print(f"\n{title}")
    print(f"  before: {before}")
    print(f"  after : {after}")


def main():
    api = client()

    # 1. Context-dependent question
    print("=" * 78)
    print("--- 1. Context-dependent question ---")
    q = "Are there any other rides?"
    show("depends on the conversation before it", q,
         rewrite(api, CONTEXT_INSTRUCTION, q, HISTORY))
    print("  why  : on its own this does not say which area; the rewrite names it")
    print("         and the rides already listed.")

    # 2. Comparative question
    print("\n--- 2. Comparative question ---")
    q = "Which one takes longer and is more fun?"
    show("both sides were never named", q,
         rewrite(api, COMPARATIVE_INSTRUCTION, q, COMPARISON_HISTORY))

    # 3. Ambiguous reference
    print("\n--- 3. Ambiguous reference ---")
    q = "When do both of them start?"
    show("'both of them' points backwards", q,
         rewrite(api, PRONOUN_INSTRUCTION, q, PRONOUN_HISTORY))

    # 4. Multi-intent question
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

    # 5. Rhetorical question
    print("\n--- 5. Rhetorical question ---")
    q = "Don't tell me I have to book a month ahead as well?"
    show("emotion carries the sentence, not the request", q,
         rewrite(api, RHETORICAL_INSTRUCTION, q, RHETORICAL_HISTORY))
    print("  why  : the sentence is a complaint; the rewrite keeps the question")
    print("         inside it.")

    # 6. Classify and rewrite in one call
    print("\n" + "=" * 78)
    print("--- 6. Classify and rewrite in one call ---")
    print(f"{'question':<48} {'type':<20} {'conf':>5}")
    print("-" * 78)
    samples = [
        ("Are there any other rides?", HISTORY),
        ("Which Riverbend Park area is more fun?", COMPARISON_HISTORY),
        ("Are they all suitable for small children?", PRONOUN_HISTORY),
        ("Which restaurants are there? What do they cost?", ""),
        ("Don't tell me this is another two-hour queue?", ""),
    ]
    rewrites, scores = [], []
    for query, history in samples:
        parsed = parse_json(ask(api, PROMPT_FRAME.format(
            instruction=CLASSIFY_INSTRUCTION,
            history=history or "(none)", query=query))) or {}
        kind = parsed.get("query_type", "?")
        conf = parsed.get("confidence", 0)
        rewrites.append(parsed.get("rewritten_query", ""))
        scores.append(float(conf))
        print(f"{query:<48} {kind:<20} {float(conf):>5.2f}")
    print("-" * 78)
    for query, text in zip([s[0] for s in samples], rewrites):
        print(f"  {query}\n    -> {text}")
    print("\n  Look at the multi-intent sample. This schema declares rewritten_query")
    print("  as one string, so the two questions in that turn come back as a single")
    print("  line. The output has no room for the list part 4 produced, which is")
    print("  why part 4 handles this type on its own.")
    print(f"\n  Confidence runs from {min(scores):.2f} to {max(scores):.2f}. The model "
          "reports it itself;")
    print("  it is not measured.")

    # 7. Live data
    print("\n" + "=" * 78)
    print("--- 7. Does this question need live data? ---")
    live_queries = [
        "Is Shanghai Disneyland open today, and how busy is it right now?",
        "How much is a Shanghai Disneyland ticket next Saturday, and how far ahead must I book?",
        # A question the knowledge base answers, so part 7 can also show a no.
        "Which themed lands does Shanghai Disneyland have?",
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

    # 8. Search engine rewrite
    print("\n--- 8. Rewrite for a search engine ---")
    print("  A different target from parts 1 to 5: those produce a full sentence for")
    print("  vector retrieval, this produces keywords for a search engine.")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("  (no TAVILY_API_KEY, the web searches are skipped)")
    for query in accepted:
        rewritten = parse_json(ask(api, PROMPT_FRAME.format(
            instruction=WEB_REWRITE_INSTRUCTION, history="(none)", query=query))) or {}
        search_query = rewritten.get("rewritten_query", "")
        print(f"\n  {query}")
        print(f"    query   : {search_query}")
        print(f"    keywords: {rewritten.get('search_keywords', [])}")
        print(f"    sources : {rewritten.get('suggested_sources', [])}")
        if not tavily_key or not search_query:
            continue
        for label, text in (("original", query), ("rewrite", search_query)):
            print(f"    Tavily, {label}:")
            for title, url in web_search(text, tavily_key):
                print(f"      {title[:60]}  {url}")
    if tavily_key:
        print("\n  Search results change from day to day, so a rerun can list other pages.")

    # 9. Search plan
    print("\n--- 9. Build a search plan ---")
    for query in accepted:
        plan = parse_json(ask(api, PROMPT_FRAME.format(
            instruction=WEB_STRATEGY_INSTRUCTION, history="(none)", query=query))) or {}
        primary = plan.get("primary_keywords", [])
        extended = plan.get("extended_keywords", [])
        print(f"\n  {query}")
        print(f"    primary : {primary}")
        print(f"    extended: {extended}")
        print(f"    sites   : {plan.get('search_platforms', [])}")
        print(f"    window  : {plan.get('time_range', '')}")
        # Checks that the instruction still works: asked for keywords, a model can
        # hand back the whole sentence and no extended keywords.
        if len(primary) == 1 and primary[0].strip().rstrip("?") == query.strip().rstrip("?"):
            print("    [weak] primary keywords are the original sentence verbatim")
        if not extended:
            print("    [weak] no extended keywords were produced")

    print("\n" + "=" * 78)
    print("Each rewrite above is one model call, so rewriting has a cost per question.")


if __name__ == "__main__":
    main()
