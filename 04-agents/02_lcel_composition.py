"""This script joins LangChain runnables into chains with LCEL, the LangChain
Expression Language, and runs them on a product review and a few local inputs.
In LCEL every step is a runnable: a template, a model, a parser, or a plain
function wrapped in RunnableLambda. The pipe operator | joins runnables into
one, and the result has the same invoke, stream and batch methods as each step.
RunnableParallel runs steps side by side, RunnableBranch picks one by a
condition, and .with_retry() repeats a step that fails.

The run prints six parts:
    1. Sequential chain. The review is translated into French, its main
       complaint stated in French, and that translated back: three model
       calls joined by |.
    2. Retry. A step that fails twice, retried by a hand-written loop and by
       .with_retry(), which also waits between attempts. No model is called.
    3. Local functions as runnables. Three plain functions wrapped in
       RunnableLambda and invoked like a model. No model is called.
    4. Parallel branches. Five prompts about the review, run at once with
       RunnableParallel and then one after another, timed.
    5. Conditional routing. RunnableBranch sends three inputs to different
       steps by their shape: CSV is converted to JSON, a multi-line note has
       its lines counted, and short text is left as is. No model is called.
       Then which uses of | accept a plain function and which raise
       TypeError.
    6. Streaming the composed chain. Time to the first streamed chunk against
       a blocking call on the same prompt.
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

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

REVIEW = (
    "The battery lasts a full weekend and the case feels solid, but the app "
    "drops the connection every few hours and support never replied to me."
)

CSV_SAMPLE = "name,age,comment\nAlice,25,Works exactly as described\nBen,30,Support was slow\nChloe,28,Great value for the price"

MULTILINE_TEXT = "First line of the note\nSecond line of the note\nThird line of the note"


def build_model(streaming: bool = False) -> ChatOpenAI:
    """Return the chosen chat model through LangChain's OpenAI client."""
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0,
        streaming=streaming,
    )


# Plain functions used in parts 3 and 5. RunnableLambda turns each into a runnable.

POSITIVE_WORDS = ("good", "great", "solid", "love", "excellent", "works")
NEGATIVE_WORDS = ("bad", "slow", "drops", "never", "broken", "poor")


def analyse_text(payload: dict) -> str:
    """Count words and characters and take a crude sentiment reading from two word lists.
    The lists are naive on purpose: the step costs nothing and never calls a model."""
    text = payload["text"].lower()
    positive = sum(word in text for word in POSITIVE_WORDS)
    negative = sum(word in text for word in NEGATIVE_WORDS)
    if positive > negative:
        sentiment = "positive"
    elif negative > positive:
        sentiment = "negative"
    else:
        sentiment = "mixed"
    return (
        f"words={len(payload['text'].split())} "
        f"characters={len(payload['text'])} "
        f"sentiment={sentiment} (hits {positive}+/{negative}-)"
    )


def convert_data(payload: dict) -> str:
    """Convert CSV text into a JSON list with one object per row."""
    lines = payload["data"].strip().split("\n")
    headers = lines[0].split(",")
    rows = [dict(zip(headers, line.split(","))) for line in lines[1:] if line.count(",") == len(headers) - 1]
    return json.dumps(rows, indent=2)


def process_text(payload: dict) -> str:
    """Count the lines in `content`."""
    return f"{len(payload['content'].splitlines())} lines"


LOCAL_STEPS = {
    "analyse": RunnableLambda(analyse_text),
    "convert": RunnableLambda(convert_data),
    "process": RunnableLambda(process_text),
}


def make_flaky_step() -> RunnableLambda:
    """Return a step that fails on its first two calls, then succeeds.
    It stands in for a rate limit or a dropped connection, without a network call."""
    calls = {"count": 0}

    def flaky(payload: dict) -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError(f"transient failure on attempt {calls['count']}")
        return payload["input"]

    return RunnableLambda(flaky)


def run_sequential_chain(model: ChatOpenAI) -> None:
    """Step 1. Join three model calls with |. Each dict such as {"french": ...}
    renames the output to the next template's variable."""
    print("--- 1. Sequential chain ---")
    to_french = ChatPromptTemplate.from_template("Translate to French, output the translation only:\n{input}") | model | StrOutputParser()
    review_it = ChatPromptTemplate.from_template("In French, in two sentences, state the main complaint in this review:\n{french}") | model | StrOutputParser()
    back_to_english = ChatPromptTemplate.from_template("Translate to English, output the translation only:\n{summary}") | model | StrOutputParser()

    chain = to_french | {"french": lambda text: text} | review_it | {"summary": lambda text: text} | back_to_english
    result = chain.invoke({"input": REVIEW})
    print(f"  source:  {REVIEW}")
    print(f"  result:  {result}")


def run_retry() -> None:
    """Step 2. Retry a step that fails twice, by hand and with .with_retry()."""
    print("\n--- 2. Retry ---")
    print("  retrying a flaky step by hand:")
    flaky = make_flaky_step()
    attempts, outcome = 0, None
    while attempts < 3 and outcome is None:
        attempts += 1
        try:
            outcome = flaky.invoke({"input": "step succeeded"})
        except RuntimeError as exc:
            print(f"    attempt {attempts} failed: {exc}")
    print(f"    manual loop: {outcome!r} after {attempts} attempts")

    print("  retrying the same step with .with_retry():")
    retrying_flaky = make_flaky_step().with_retry(stop_after_attempt=3)
    started = time.perf_counter()
    outcome = retrying_flaky.invoke({"input": "step succeeded"})
    waited = time.perf_counter() - started
    print(f"    .with_retry(): {outcome!r} after {waited:.2f}s, no loop or except clause written")


def run_local_steps() -> None:
    """Step 3. Invoke three plain functions through RunnableLambda, the same interface a model has."""
    print("\n--- 3. Local functions as runnables ---")
    print(f"  analyse: {LOCAL_STEPS['analyse'].invoke({'text': REVIEW})}")
    rows = json.loads(LOCAL_STEPS["convert"].invoke({"data": CSV_SAMPLE}))
    print(f"  convert: {rows[0]} ({len(rows)} rows)")
    print(f"  process: {LOCAL_STEPS['process'].invoke({'content': MULTILINE_TEXT})}")


def run_parallel_branches(model: ChatOpenAI) -> None:
    """Step 4. Run five prompts on the review at once, then one after another, and time both.
    RunnableParallel gives each branch its own thread."""
    print("\n--- 4. Parallel branches ---")
    branches = {
        "summary": ChatPromptTemplate.from_template("Summarise in one sentence:\n{input}") | model | StrOutputParser(),
        "area": ChatPromptTemplate.from_template("Reply with exactly one word, the product area this is about:\n{input}") | model | StrOutputParser(),
        "sentiment": ChatPromptTemplate.from_template("Reply with exactly one word (positive, negative or mixed):\n{input}") | model | StrOutputParser(),
        "urgency": ChatPromptTemplate.from_template("Reply with exactly one word (low, medium or high) for how urgently this needs a reply:\n{input}") | model | StrOutputParser(),
        "language": ChatPromptTemplate.from_template("Reply with exactly one word, the language this text is written in:\n{input}") | model | StrOutputParser(),
    }

    parallel = RunnableParallel(**branches)
    started = time.perf_counter()
    result = parallel.invoke({"input": REVIEW})
    parallel_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for branch in branches.values():
        branch.invoke({"input": REVIEW})
    serial_seconds = time.perf_counter() - started

    for name, value in result.items():
        print(f"  {name}: {value.strip()}")
    print(f"  parallel {parallel_seconds:.2f}s vs serial {serial_seconds:.2f}s ({len(branches)} branches)")


def route_by_shape(payload: dict) -> str:
    """The router's logic as a plain function. It is never called; it only shows when | accepts one."""
    content = payload["content"]
    if "," in content:
        return convert_data({"data": content})
    if content.count("\n") >= 2:
        return process_text({"content": content})
    return f"short text, {len(content)} characters, left as is"


def run_branching() -> None:
    """Step 5. Route each input by its shape with RunnableBranch, and run the matching step. A plain
    function also pipes when the other side is a runnable; two plain functions raise TypeError."""
    print("\n--- 5. Conditional routing ---")
    router = RunnableBranch(
        (lambda payload: "," in payload["content"], RunnableLambda(lambda payload: convert_data({"data": payload["content"]}))),
        (lambda payload: payload["content"].count("\n") >= 2, LOCAL_STEPS["process"]),
        RunnableLambda(lambda payload: f"short text, {len(payload['content'])} characters, left as is"),
    )
    for content in [CSV_SAMPLE, MULTILINE_TEXT, "hello"]:
        result = " ".join(router.invoke({"content": content}).split())
        if len(result) > 70:
            result = result[:70] + "..."
        print(f"  {content.splitlines()[0][:28]:30} -> {result}")

    print("\n  a plain function piped into the router still works (LangChain coerces it):")
    coerced = route_by_shape | router
    print(f"    {type(coerced).__name__}, no TypeError")

    print("  but two plain functions piped together have no Runnable to coerce through:")
    try:
        route_by_shape | route_by_shape
    except TypeError as exc:
        print(f"    TypeError: {exc}")


def run_streaming(model: ChatOpenAI) -> None:
    """Step 6. Stream the chain, and time its first chunk against a blocking call. A plain
    function in RunnableLambda would wait for the whole input and undo the streaming."""
    print("\n--- 6. Streaming the composed chain ---")
    chain = ChatPromptTemplate.from_template("List three short bullet points about {topic}.") | model | StrOutputParser()
    topic = "keeping a laptop battery healthy"

    chunks, first_chunk_seconds = 0, None
    started = time.perf_counter()
    print("  ", end="", flush=True)
    for chunk in chain.stream({"topic": topic}):
        if first_chunk_seconds is None:
            first_chunk_seconds = time.perf_counter() - started
        print(chunk.replace("\n", "\n  "), end="", flush=True)
        chunks += 1
    print(f"\n  streamed: first chunk after {first_chunk_seconds:.2f}s, {chunks} chunks total")

    started = time.perf_counter()
    chain.invoke({"topic": topic})
    blocking_seconds = time.perf_counter() - started
    print(f"  blocking: nothing visible until {blocking_seconds:.2f}s, when the full answer arrives at once")


def main() -> None:
    if not API_KEY:
        print("No DEEPSEEK_API_KEY or OPENAI_API_KEY in .env. Only steps 2, 3 and 5 run.")
        run_retry()
        run_local_steps()
        run_branching()
        return

    model = build_model()
    run_sequential_chain(model)
    run_retry()
    run_local_steps()
    run_parallel_branches(model)
    run_branching()
    run_streaming(build_model(streaming=True))


if __name__ == "__main__":
    main()
