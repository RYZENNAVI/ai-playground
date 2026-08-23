"""Compose runnables with the pipe operator instead of writing glue code.

Demonstrates what the pipe operator actually buys you:
    1. Chain three model calls so each one consumes the previous answer, then
       retry a flaky step by hand and again with `.with_retry()`.
    2. Wrap plain Python functions as chain steps and dispatch to them by name.
    3. Run five independent branches over one input and time them against serial.
    4. Route an input to a different chain depending on a condition, then pipe
       the router's own output into a further step.
    5. Stream the final chain and compare it against a blocking call.

Module 04: Agents - LCEL Composition.
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

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

REVIEW = (
    "The battery lasts a full weekend and the case feels solid, but the app "
    "drops the connection every few hours and support never replied to me."
)

CSV_SAMPLE = "name,age,comment\nAlice,25,Works exactly as described\nBen,30,Support was slow\nChloe,28,Great value for the price"

MULTILINE_TEXT = "First line of the note\nSecond line of the note\nThird line of the note"


def build_model(streaming: bool = False) -> ChatOpenAI:
    """Return a chat model reached through the OpenAI request format."""
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
        streaming=streaming,
    )


# 2. Plain Python functions, later lifted into the chain by RunnableLambda.

POSITIVE_WORDS = ("good", "great", "solid", "love", "excellent", "works")
NEGATIVE_WORDS = ("bad", "slow", "drops", "never", "broken", "poor")


def analyse_text(payload: dict) -> str:
    """Count words and characters and take a crude sentiment reading.

    The word lists are deliberately naive. Their job is to be a step in a chain
    that costs nothing and always returns, so the composition being demonstrated
    stays visible instead of hiding behind another model call.
    """
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
    """Convert between CSV and JSON in whichever direction is asked for."""
    source, target = payload["source_format"], payload["target_format"]
    if source == "csv" and target == "json":
        lines = payload["data"].strip().split("\n")
        headers = lines[0].split(",")
        rows = [dict(zip(headers, line.split(","))) for line in lines[1:] if line.count(",") == len(headers) - 1]
        return json.dumps(rows, indent=2)
    if source == "json" and target == "csv":
        rows = json.loads(payload["data"])
        headers = sorted({key for row in rows for key in row})
        body = "\n".join(",".join(str(row.get(header, "")) for header in headers) for row in rows)
        return ",".join(headers) + "\n" + body
    return f"unsupported conversion: {source} -> {target}"


def process_text(payload: dict) -> str:
    """Count lines, find a substring, or replace one, depending on `operation`."""
    operation, content = payload["operation"], payload["content"]
    if operation == "count_lines":
        return f"{len(content.splitlines())} lines"
    if operation == "find":
        hits = [f"line {i}: {line}" for i, line in enumerate(content.splitlines(), 1) if payload["needle"] in line]
        return "\n".join(hits) if hits else f"no line contains {payload['needle']!r}"
    if operation == "replace":
        return content.replace(payload["needle"], payload["replacement"])
    return f"unsupported operation: {operation}"


LOCAL_STEPS = {
    "analyse": RunnableLambda(analyse_text),
    "convert": RunnableLambda(convert_data),
    "process": RunnableLambda(process_text),
}


def make_flaky_step() -> RunnableLambda:
    """Return a step that fails on its first two calls, then succeeds.

    The failure count lives in a closure instead of coming from a real network
    call, so the retry demo below is deterministic and free. What it stands in
    for is any step that can fail transiently - a rate limit, a dropped
    connection - where the fix is "try again", not "handle the error".
    """
    calls = {"count": 0}

    def flaky(payload: dict) -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError(f"transient failure on attempt {calls['count']}")
        return payload["input"]

    return RunnableLambda(flaky)


def run_sequential_chain(model: ChatOpenAI) -> None:
    """Step 1. Feed each model call the previous call's output.

    Three separate chains are joined by the pipe operator, so the composed
    object takes one dict in and returns one string out. What makes it a chain
    rather than three calls is the key handoff: stage one is wrapped in
    {"french": ...} because stage two declares {french} as its variable, and the
    same trick carries stage two into stage three. Mismatch those key names and
    the chain fails at the boundary, not inside the model.

    The retry comparison at the end is not about this translation chain - it
    reuses the flaky step above to show what surviving a transient failure
    costs in each style: a hand-written loop with its own attempt counter and
    except clause, against one method call that does the same thing.
    """
    print("--- 1. Sequential chain ---")
    to_french = ChatPromptTemplate.from_template("Translate to French, output the translation only:\n{input}") | model | StrOutputParser()
    review_it = ChatPromptTemplate.from_template("In French, in two sentences, state the main complaint in this review:\n{french}") | model | StrOutputParser()
    back_to_english = ChatPromptTemplate.from_template("Translate to English, output the translation only:\n{summary}") | model | StrOutputParser()

    chain = to_french | {"french": lambda text: text} | review_it | {"summary": lambda text: text} | back_to_english
    result = chain.invoke({"input": REVIEW})
    print(f"  source:  {REVIEW}")
    print(f"  result:  {result}")

    print("\n  retrying a flaky step by hand:")
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
    outcome = retrying_flaky.invoke({"input": "step succeeded"})
    print(f"    .with_retry(): {outcome!r}, no loop or except clause written")


def run_local_steps() -> None:
    """Step 2. Dispatch to a plain function through the same runnable interface.

    Nothing here calls a model. The point is that RunnableLambda gives an
    ordinary function the same invoke/stream/batch surface a model has, so local
    work and model work can sit in one chain without adapters between them.
    """
    print("\n--- 2. Local functions as chain steps ---")
    print(f"  analyse: {LOCAL_STEPS['analyse'].invoke({'text': REVIEW})}")
    converted = LOCAL_STEPS["convert"].invoke({"data": CSV_SAMPLE, "source_format": "csv", "target_format": "json"})
    print(f"  convert: {converted.splitlines()[0]} ... ({len(json.loads(converted))} rows)")
    print(f"  process: {LOCAL_STEPS['process'].invoke({'operation': 'count_lines', 'content': MULTILINE_TEXT})}")


def run_parallel_branches(model: ChatOpenAI) -> None:
    """Step 3. Send one input down five branches at once and time the difference.

    RunnableParallel starts every branch on its own thread and returns a dict
    keyed by branch name. All five branches call the same model with the same
    input, so the only variable is whether they wait for each other. Five
    branches instead of two makes the gap between the two timings scale with
    branch count rather than sit inside normal network jitter.
    """
    print("\n--- 3. Parallel branches ---")
    branches = {
        "summary": ChatPromptTemplate.from_template("Summarise in one sentence:\n{input}") | model | StrOutputParser(),
        "area": ChatPromptTemplate.from_template("Reply with exactly one word, the product area this is about:\n{input}") | model | StrOutputParser(),
        "sentiment": ChatPromptTemplate.from_template("Reply with exactly one word - positive, negative, or mixed:\n{input}") | model | StrOutputParser(),
        "urgency": ChatPromptTemplate.from_template("Reply with exactly one word - low, medium, or high - for how urgently this needs a reply:\n{input}") | model | StrOutputParser(),
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
    """The same routing logic as `router` below, written as a plain function.

    It exists only to show where the pipe operator stops working. LangChain
    coerces a bare function into a Runnable when the *other* side of `|` is
    already one - that is why `route_by_shape | explain` below succeeds. What
    a bare function cannot do is start a pipe with another bare function:
    neither side defines `__or__`, so there is nothing to trigger coercion,
    and `route_by_shape | route_by_shape` raises TypeError. The branch below
    is built from RunnableLambda specifically so it never hits that case.
    """
    content = payload["content"]
    if content.count("\n") >= 2:
        return LOCAL_STEPS["process"].invoke({"operation": "count_lines", "content": content})
    if "," in content:
        return "looks like tabular data, use the converter"
    return f"short text, {len(content)} characters, left as is"


def run_branching(model: ChatOpenAI) -> None:
    """Step 4. Pick a different chain depending on what the input looks like.

    RunnableBranch is an if/elif written as data instead of control flow. That
    matters because the branch stays a runnable: it can be piped into, streamed
    from, and inspected, which a bare Python if statement in the middle of a
    chain cannot - demonstrated below by piping the router into a further step,
    then trying the same thing with `route_by_shape` and watching it fail.
    """
    print("\n--- 4. Conditional routing ---")
    router = RunnableBranch(
        (lambda payload: payload["content"].count("\n") >= 2, LOCAL_STEPS["process"]),
        (lambda payload: "," in payload["content"], RunnableLambda(lambda payload: "looks like tabular data, use the converter")),
        RunnableLambda(lambda payload: f"short text, {len(payload['content'])} characters, left as is"),
    )
    for content in [MULTILINE_TEXT, "name,age", "hello"]:
        verdict = router.invoke({"operation": "count_lines", "content": content})
        print(f"  {content.splitlines()[0][:28]:30} -> {verdict}")

    print("\n  piping the router into a further step:")
    explain = ChatPromptTemplate.from_template("In one short sentence, explain why this routing verdict makes sense for the input {content!r}:\n{verdict}") | model | StrOutputParser()
    full_chain = router | RunnableLambda(lambda verdict: {"content": MULTILINE_TEXT, "verdict": verdict}) | explain
    print(f"    {full_chain.invoke({'operation': 'count_lines', 'content': MULTILINE_TEXT})}")

    print("  a plain function piped into that same Runnable still works (LangChain coerces it):")
    coerced = route_by_shape | explain
    print(f"    {type(coerced).__name__}, no TypeError")

    print("  but two plain functions piped together have no Runnable to coerce through:")
    try:
        route_by_shape | route_by_shape
    except TypeError as exc:
        print(f"    TypeError: {exc}")


def run_streaming(model: ChatOpenAI) -> None:
    """Step 5. Print the answer while the model is still writing it.

    Streaming is a property of the composed chain, not only of the model: every
    step in this chain can pass chunks along, so calling stream on the outermost
    object yields text as soon as the first tokens exist. A step that must see
    the whole input, such as a JSON parser, would silently turn this back into a
    single block at the end. The blocking call below times how long that first
    visible output takes without streaming, for the same prompt.
    """
    print("\n--- 5. Streaming the composed chain ---")
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
    has_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    if not has_key:
        print("DEEPSEEK_API_KEY is not set; steps 1, 3, 4 and 5 are skipped.\n")

    model = build_model() if has_key else None

    if model is not None:
        run_sequential_chain(model)
    run_local_steps()
    if model is not None:
        run_parallel_branches(model)
    if model is not None:
        run_branching(model)
    if model is not None:
        run_streaming(build_model(streaming=True))


if __name__ == "__main__":
    main()
