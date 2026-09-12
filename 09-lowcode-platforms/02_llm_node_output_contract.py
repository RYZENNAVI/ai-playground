"""Put a real model behind a workflow node and watch the next node compare strings.

Demonstrates where a model node and a code node meet, and what leaks through:
    1. Run the node on a plain instruction, the way a first draft states it.
    2. Run it again with an output example pinned to the prompt.
    3. Run it a third time with a written JSON constraint and the format switch on.
    4. Score all three against the exact vocabulary the downstream node expects.
    5. Route every row through that downstream node and count what disappears.
    6. Normalise each label before comparing it, and route the same rows again.
    7. Hand the model a deterministic edit and check it against the same edit in code.

Module 09: Low-Code Platforms - Model Node Output Contract.
"""

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MAX_ATTEMPTS = 4
RETRY_BACKOFF = 6

# The vocabulary the downstream code node compares against, character for character.
EXPECTED = ("positive", "neutral", "negative")

# Variant 1: everything a first draft usually says. A role, a task, no shape.
PLAIN_PROMPT = """You are a review sentiment analyst for a trading application.
Read the review, say how the reviewer feels, and give a short digest of what
they report."""

# Variant 2: the same node with an output example appended, which is how these
# prompts tend to look once someone has read a reply they could not parse.
EXAMPLE_PROMPT = PLAIN_PROMPT + """

# Output example
"verdict": "positive/neutral/negative",
"digest": "..."
"""

# Variant 3: the example, plus the platform switch that makes the whole reply JSON.
SCHEMA_PROMPT = EXAMPLE_PROMPT + """
# Constraint
- Reply with a single JSON object holding exactly the keys "verdict" and "digest".
- "verdict" must be one of: positive, neutral, negative.
"""

VARIANTS = (
    ("plain", PLAIN_PROMPT, False),
    ("example", EXAMPLE_PROMPT, False),
    ("json mode", SCHEMA_PROMPT, True),
)

REVIEWS = [
    {"title": "Fast and stable",
     "body": "Order entry is quick and the charts finally load on a weak connection."},
    {"title": "Login keeps failing",
     "body": "Face unlock fails every morning and the password screen rejects a valid password."},
    {"title": "Fine for the price",
     "body": "It does what it says. Nothing about it stands out either way."},
    {"title": "Lost my watchlist",
     "body": "The update wiped my watchlist and support has not replied in four days."},
    {"title": "Good research tab",
     "body": "The research tab is genuinely useful, though it buries the export button."},
    {"title": "Charts are unreadable at night",
     "body": "Dark mode turns the candles grey on grey, so I trade from a laptop instead."},
]

# A deterministic edit: drop these words, keep everything else exactly as it was.
STOPWORDS = ["broker", "brokerage", "application", "app", "user", "users", "not"]

STOPWORD_PROMPT = """Remove every word in this list from the text: {words}.
Keep all remaining words and their order unchanged. Make sure the word "not" is
removed. Reply with the edited text only."""

STOPWORD_TEXT = (
    "The broker app is not slow, but users report the brokerage research tab is "
    "not reachable when the application reconnects. Not one user mentioned fees."
)


def pick_provider():
    """Return (api_key, base_url, model) for whichever key is configured.

    Only chat completion is needed here, so DeepSeek comes first; Gemini and
    OpenAI follow, so a single key of any kind is enough to run the script.
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
    """Send one request, backing off when the provider answers with a rate limit.

    A node that fires once per element sends as many requests as there are
    elements, so a burst is the normal case here rather than the exception.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as error:
            retriable = any(token in str(error).lower()
                            for token in ("429", "rate", "exhausted", "timeout"))
            if not retriable or attempt == MAX_ATTEMPTS:
                raise
            wait = RETRY_BACKOFF * attempt
            print(f"    provider pushed back ({type(error).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def run_node(client, model, review, prompt, json_mode):
    """Run one model node over one review and return its raw reply."""
    kwargs = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Title: {review['title']}\nBody: {review['body']}"},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return call_with_retry(client, **kwargs).choices[0].message.content.strip()


def read_verdict(raw):
    """Pull a verdict out of a reply, the way a platform's parser tries to.

    A JSON body is read as JSON. Anything else falls back to a regular
    expression over a labelled line, and if that finds nothing the parser is
    reduced to taking the first word. Each step down is a step further from a
    value the next node can rely on.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "verdict" in parsed:
            return str(parsed["verdict"]), "json"
    except json.JSONDecodeError:
        pass
    match = re.search(r'"?(?:verdict|sentiment)"?\**\s*[:=]\s*\**\s*"?([A-Za-z]+)', raw,
                      re.IGNORECASE)
    if match:
        return match.group(1), "regex"
    first = raw.strip().strip("*# ").split()
    return (first[0] if first else ""), "guess"


def split_by_verdict(rows):
    """Split rows by an exact string comparison, as the code node in 01 does."""
    positive = [r for r in rows if r["verdict"] == "positive"]
    neutral = [r for r in rows if r["verdict"] == "neutral"]
    negative = [r for r in rows if r["verdict"] == "negative"]
    return positive, neutral, negative


def normalise(verdict):
    """Fold a label onto the expected vocabulary before comparing it.

    The match is on the whole cleaned string, not a substring. A substring test
    would be shorter and wrong: 'not positive' contains 'positive', so it would
    fold onto the opposite of what the model said. Tidying case and punctuation
    is the job here; guessing at a label the model did not give is not.
    """
    return verdict.strip().strip("*#.\"' ").lower()


def strip_words_in_code(text, words):
    """Remove whole words from text, which is what the instruction actually asks."""
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b",
                         re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", pattern.sub("", text)).strip()


def word_counts(text):
    """Count the words in a piece of text, ignoring case and punctuation."""
    return Counter(re.findall(r"[A-Za-z']+", text.lower()))


def surviving_stopwords(text, words):
    """Return the listed words still present in a piece of text."""
    lowered = text.lower()
    return [w for w in words if re.search(rf"\b{re.escape(w)}\b", lowered)]


def main():
    provider = pick_provider()
    if provider is None:
        print("No API key found. Set DEEPSEEK_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY.")
        return
    api_key, base_url, model = provider
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"model node backed by {model}, temperature 0, {len(REVIEWS)} reviews per variant")

    runs = {}
    for step, (name, prompt, json_mode) in enumerate(VARIANTS, start=1):
        heading = {"plain": "a plain instruction",
                   "example": "an output example pinned to the prompt",
                   "json mode": "a written constraint and the format switch, both"}[name]
        print(f"\n--- {step}. The node with {heading} ---")
        rows = []
        for review in REVIEWS:
            raw = run_node(client, model, review, prompt, json_mode)
            verdict, how = read_verdict(raw)
            rows.append({"title": review["title"], "raw": raw,
                         "verdict": verdict, "how": how})
            print(f"  {review['title'][:30]:<32} read by {how:<5} -> {verdict!r}")
        runs[name] = rows
        first = rows[0]["raw"].replace("\n", " ")
        print(f"  first reply began: {first[:72]!r}")

    print("\n--- 4. Scored against the vocabulary the next node compares against ---")
    for name, rows in runs.items():
        parsed = sum(1 for r in rows if r["how"] == "json")
        exact = sum(1 for r in rows if r["verdict"] in EXPECTED)
        print(f"  {name:<10} {parsed}/{len(rows)} replies parsed as JSON, "
              f"{exact}/{len(rows)} verdicts match the vocabulary exactly")
        off = sorted({r["verdict"] for r in rows if r["verdict"] not in EXPECTED})
        if off:
            print(f"             values outside it: {off}")

    print("\n--- 5. What the downstream code node does with those rows ---")
    routed_raw = {}
    for name, rows in runs.items():
        pos, neu, neg = split_by_verdict(rows)
        routed = len(pos) + len(neu) + len(neg)
        routed_raw[name] = routed
        print(f"  {name:<10} routed {routed}/{len(rows)}  "
              f"(positive {len(pos)}, neutral {len(neu)}, negative {len(neg)})")
        lost = [r["title"] for r in rows if r["verdict"] not in EXPECTED]
        if lost:
            print(f"             dropped without an error: {lost}")
    print("  a row that matches no branch raises nothing anywhere; it is simply gone")

    print("\n--- 6. The same rows, with each label normalised first ---")
    stubborn = set()
    folded_labels = {}
    for name, rows in runs.items():
        folded = [{**r, "verdict": normalise(r["verdict"])} for r in rows]
        folded_labels[name] = [r["verdict"] for r in folded]
        pos, neu, neg = split_by_verdict(folded)
        routed = len(pos) + len(neu) + len(neg)
        print(f"  {name:<10} routed {routed}/{len(rows)}  "
              f"(positive {len(pos)}, neutral {len(neu)}, negative {len(neg)}), "
              f"recovering {routed - routed_raw[name]}")
        still_off = sorted({r["verdict"] for r in folded if r["verdict"] not in EXPECTED})
        stubborn.update(still_off)
        if still_off:
            print(f"             still outside the vocabulary: {still_off}")
    print("  folding costs one line and recovers every label that differs from the")
    print("  vocabulary only in case or punctuation. It cannot recover a word the")
    print(f"  model picked for itself: {sorted(stubborn) or 'nothing here'} came back")
    print("  unroutable either way, which is the enum's job in the prompt, not the")
    print("  parser's job after the fact")
    reference = folded_labels["json mode"]
    for name in ("plain", "example"):
        agree = sum(1 for a, b in zip(folded_labels[name], reference) if a == b)
        print(f"  {name:<10} agrees with json mode on {agree}/{len(reference)} rows "
              f"once folded")
    print("  there is no hand-labelled sentiment anywhere in this script, so that is")
    print("  agreement between variants, not accuracy. Steps 4 and 5 score the output")
    print("  contract; nothing here scores the judgement behind the label")

    print("\n--- 7. A deterministic edit, asked of the model and written in code ---")
    print(f"  words to remove: {STOPWORDS}")
    raw = call_with_retry(
        client,
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": STOPWORD_PROMPT.format(words=", ".join(STOPWORDS))},
            {"role": "user", "content": STOPWORD_TEXT},
        ],
    ).choices[0].message.content.strip()
    in_code = strip_words_in_code(STOPWORD_TEXT, STOPWORDS)
    left_model = surviving_stopwords(raw, STOPWORDS)
    left_code = surviving_stopwords(in_code, STOPWORDS)
    print(f"  model node : {raw[:96]!r}")
    print(f"               {len(left_model)} listed word(s) survive: {left_model}")
    print(f"  code node  : {in_code[:96]!r}")
    print(f"               {len(left_code)} listed word(s) survive: {left_code}")
    # Counting only the residue answers half the instruction. The other half is
    # 'keep all remaining words unchanged', and a model can satisfy the first
    # while quietly failing the second.
    # Word counts alone cannot see order: 'users trust brokers' and 'brokers trust
    # users' have identical counts. The instruction says the remaining words keep
    # their order, so the sequence is the test and the counts explain a failure.
    model_tokens, code_tokens = raw.split(), in_code.split()
    dropped = word_counts(in_code) - word_counts(raw)
    added = word_counts(raw) - word_counts(in_code)
    print(f"  the two results match token for token, order included: "
          f"{'yes' if model_tokens == code_tokens else 'no'}")
    if model_tokens != code_tokens:
        first = next((i for i, (a, b) in enumerate(zip(model_tokens, code_tokens))
                      if a != b), min(len(model_tokens), len(code_tokens)))
        print(f"               they first part at token {first}: model wrote "
              f"{' '.join(model_tokens[first:first + 3])!r}, rule wrote "
              f"{' '.join(code_tokens[first:first + 3])!r}")
    print(f"               words the rule keeps that the model dropped: "
          f"{sorted(dropped.elements())}")
    print(f"               words the model wrote that the rule does not: "
          f"{sorted(added.elements())}")
    print("  the edit is defined by a rule, so the node that can apply a rule owns it")


if __name__ == "__main__":
    main()
