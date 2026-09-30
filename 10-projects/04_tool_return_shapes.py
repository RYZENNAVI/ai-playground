"""This script asks a hosted chat model the same question about the price database five
times, changing only the shape of the tool output in the prompt (head, head and tail, a
describe() summary, endpoints, and endpoints with the change computed), and scores each
reply against the answer computed from the database.

A tool's return value sets the ceiling on what a model can answer:
    1. Compute the answer directly from the database, so every reply can be scored.
    2. Run one query and turn its result into five different return shapes.
    3. Send each shape to the model with the same question and prompt, and read out the
       instrument and the change each reply gives.
    4. Score every reply against the computed answer, naming and number apart, next to
       the size of the shape it was given.
    5. Check which instruments the two ten-row shapes reach, and whether either holds an
       instrument's first and last close together.
"""

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

DATA = Path(__file__).parent / "data"
DB_PATH = DATA / "market.sqlite"

MAX_ATTEMPTS = 4
RETRY_BACKOFF = 6

TARGET_YEAR = 2024
QUESTION = (
    f"Of the instruments in this data, which one moved the furthest over {TARGET_YEAR} "
    f"in percentage terms, measured as the largest absolute percentage change? A fall "
    f"counts as a move, so a drop of 40 percent is a larger move than a rise of 30. "
    f"Compare each instrument's first close of the year with its last close of the year, "
    f"and report the signed change for the instrument you pick."
)

# The query a text-to-SQL step would produce for that question. It is correct SQL:
# it selects exactly the rows the question is about.
QUERY = f"""
    SELECT ticker, trade_date, close
    FROM daily_price
    WHERE trade_date LIKE '{TARGET_YEAR}%'
    ORDER BY ticker, trade_date
"""

SYSTEM_PROMPT = (
    "You answer questions about market data using only the tool output given to you. "
    "Do not use any outside knowledge about real companies. "
    "Reply with JSON only, in the form "
    '{"ticker": "<ticker>", "change_pct": <number>, "basis": "<one short sentence>"}. '
    "change_pct is a percentage, so a rise of nine and a half percent is 9.5. "
    "If the data you were given cannot answer the question, still give your best "
    "estimate from what you have."
)


def truth_table(connection: sqlite3.Connection) -> pd.DataFrame:
    """Compute each instrument's first and last close of the year, and the change between them.
    It reads the database rather than any shape, so every reply is scored against it."""
    frame = pd.read_sql_query(QUERY, connection)
    rows = []
    for ticker, group in frame.groupby("ticker"):
        group = group.sort_values("trade_date")
        first, last = group.iloc[0], group.iloc[-1]
        rows.append({
            "ticker": ticker,
            "first_date": first["trade_date"],
            "first_close": first["close"],
            "last_date": last["trade_date"],
            "last_close": last["close"],
            "change_pct": 100.0 * (last["close"] - first["close"]) / first["close"],
            "rows": len(group),
        })
    return pd.DataFrame(rows)


def shape_head(frame: pd.DataFrame) -> str:
    """Return the first ten rows, the usual preview.
    Sorted by ticker, all ten belong to the first instrument."""
    return frame.head(10).to_markdown(index=False)


def shape_head_and_tail(frame: pd.DataFrame) -> str:
    """Return the first five rows and the last five.
    Sorted by ticker, no instrument appears with both its first and its last close."""
    return pd.concat([frame.head(5), frame.tail(5)]).to_markdown(index=False)


def shape_head_tail_describe(frame: pd.DataFrame) -> str:
    """Add summary statistics of the close column to the previous shape.
    They pool every instrument's closes, so no per-instrument change can be read from them."""
    digest = pd.concat([frame.head(5), frame.tail(5)]).to_markdown(index=False)
    stats = frame[["close"]].describe().round(4).to_markdown()
    return f"{digest}\n\nSummary statistics over all rows:\n{stats}"


def shape_endpoints(frame: pd.DataFrame) -> str:
    """Return the first and last row of every instrument.
    The rows are chosen per instrument, not off the ends of one flat table."""
    parts = []
    for ticker, group in frame.groupby("ticker"):
        group = group.sort_values("trade_date")
        parts.append(pd.concat([group.head(1), group.tail(1)]))
    return pd.concat(parts).to_markdown(index=False)


def shape_computed(frame: pd.DataFrame) -> str:
    """Return the endpoints with the percentage change already worked out.
    The model then only has to pick the largest absolute value."""
    rows = []
    for ticker, group in frame.groupby("ticker"):
        group = group.sort_values("trade_date")
        first, last = group.iloc[0], group.iloc[-1]
        rows.append({
            "ticker": ticker,
            "first_date": first["trade_date"],
            "first_close": first["close"],
            "last_date": last["trade_date"],
            "last_close": last["close"],
            "change_pct": round(100.0 * (last["close"] - first["close"]) / first["close"], 2),
        })
    return pd.DataFrame(rows).to_markdown(index=False)


SHAPES = [
    ("head(10)", shape_head),
    ("head(5) + tail(5)", shape_head_and_tail),
    ("head(5) + tail(5) + describe()", shape_head_tail_describe),
    ("first and last row per ticker", shape_endpoints),
    ("endpoints with change computed", shape_computed),
]


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
    """Send one request, backing off when the provider is rate-limited or timing out."""
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


def ask(client, model: str, tool_output: str) -> dict:
    """Send one shape to the model and parse the JSON it replies with."""
    response = call_with_retry(
        client,
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {QUESTION}\n\nTool output:\n{tool_output}"},
        ],
    )
    text = response.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"ticker": None, "change_pct": None, "basis": text[:120]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"ticker": None, "change_pct": None, "basis": text[:120]}


def main() -> None:
    if not DB_PATH.exists():
        raise SystemExit(f"Missing {DB_PATH.name}. Run 01_build_project_datasets.py first.")

    provider = pick_provider()
    if provider is None:
        print("No API key found. Set DEEPSEEK_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY.")
        return
    api_key, base_url, model = provider
    client = OpenAI(api_key=api_key, base_url=base_url)

    with sqlite3.connect(DB_PATH) as connection:
        frame = pd.read_sql_query(QUERY, connection)
        truth = truth_table(connection)

    # 1. The answer, computed from the database
    print("--- 1. The answer, computed from the database ---")
    print(truth.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    best = truth.loc[truth["change_pct"].abs().idxmax()]
    print(f"\n    largest absolute move: {best['ticker']} at {best['change_pct']:+.2f}%")

    # 2. One query, five return shapes
    print("\n--- 2. One query, five return shapes ---")
    print(f"    the query returns {len(frame):,} rows across {frame['ticker'].nunique()} "
          f"instruments, sorted by ticker then date")
    rendered = [(name, builder(frame)) for name, builder in SHAPES]
    for name, text in rendered:
        print(f"    {name:<34}{len(text):>7,} characters")

    # 3. The same question against each shape
    print("\n--- 3. The same question against each shape ---")
    print(f"    model {model}, temperature 0\n")
    results = []
    for step, (name, text) in enumerate(rendered, start=1):
        reply = ask(client, model, text)
        results.append((name, text, reply))
        claimed = reply.get("change_pct")
        claimed_text = f"{claimed:+.2f}%" if isinstance(claimed, (int, float)) else "none"
        print(f"    {step}. {name}")
        print(f"       says {str(reply.get('ticker')):<6} {claimed_text:>9}   "
              f"{str(reply.get('basis'))[:70]}")

    # 4. Scored against the computed answer
    print("\n--- 4. Scored against the computed answer ---")
    print(f"    {'shape':<32}{'ticker':>7}{'named':>7}{'claimed':>9}"
          f"{'truth':>9}{'error':>8}{'chars':>7}")
    truth_by_ticker = truth.set_index("ticker")["change_pct"].to_dict()
    for name, text, reply in results:
        ticker = reply.get("ticker")
        claimed = reply.get("change_pct")
        named_right = "yes" if ticker == best["ticker"] else "no"
        if isinstance(claimed, (int, float)) and ticker in truth_by_ticker:
            actual = truth_by_ticker[ticker]
            error = f"{abs(claimed - actual):.2f}"
            actual_text = f"{actual:+.2f}"
            claimed_text = f"{claimed:+.2f}"
        else:
            error, actual_text, claimed_text = "-", "-", "-"
        print(f"    {name:<32}{str(ticker):>7}{named_right:>7}{claimed_text:>9}"
              f"{actual_text:>9}{error:>8}{len(text):>7,}")
    print("\n    'named' asks whether the reply picked the instrument that actually moved")
    print("    most. 'error' is measured against the truth for whichever instrument the")
    print("    reply named, so a reply can be precise about the wrong instrument. 'chars'")
    print("    is the size of the shape the reply was given.")

    # 5. The two shapes that both fit in ten rows
    print("\n--- 5. The two shapes that both fit in ten rows ---")
    row_key = re.compile(r"^\| ([A-Z]{3}) +\| (\d{4}-\d{2}-\d{2})", re.M)
    any_complete = False
    for name, text in rendered[:2]:
        rows = set(row_key.findall(text))
        reached = sorted({ticker for ticker, _ in rows})
        complete = [r.ticker for r in truth.itertuples()
                    if (r.ticker, r.first_date) in rows and (r.ticker, r.last_date) in rows]
        any_complete = any_complete or bool(complete)
        print(f"    {name:<20}reaches {reached}, with first and last close: "
              f"{complete or 'none'}")
    if any_complete:
        print("    Both are ten rows, and at least one holds an instrument's first and last "
              "close.")
    else:
        print("    Both are ten rows, and neither holds any instrument's first and last close "
              "together.")


if __name__ == "__main__":
    main()
