"""Ask a model the same question five times, changing only what its tool hands back.

Demonstrates that a tool's return value sets the ceiling on what a model can answer:
    1. Compute the answer directly from the database, so every reply can be scored.
    2. Run one query and turn its result into five different return shapes.
    3. Send each shape to the model behind the same question and the same prompt.
    4. Read out which instrument each reply names and what change it claims.
    5. Score every reply against the computed answer, naming and number apart.
    6. Compare the two shapes that both fit in ten rows but carry different rows.
    7. Print how many characters each shape spent to reach its score.

Module 10: Applied Projects - Tool Return Shapes.
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

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

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
# it selects exactly the rows the question is about. Everything that follows is
# about what happens to those rows on the way back to the model.
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

    This runs against the whole year rather than any digest of it, so it is the
    reference every reply is scored against. It is deliberately computed in SQL
    rather than from the frame the model sees, so the two cannot drift together.
    """
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
    """Return the first ten rows, which is the shape a result preview usually takes.

    Ten rows of a 2000-row result is a preview, and a preview of a table sorted by
    ticker is entirely the first ticker. Nothing in it is wrong; it is just the
    wrong ten rows for this question.
    """
    return frame.head(10).to_markdown(index=False)


def shape_head_and_tail(frame: pd.DataFrame) -> str:
    """Return the first five rows and the last five, the usual fix for the shape above.

    This looks like the strict improvement it is normally taken for: the digest now
    reaches both ends of the result. But the result is sorted by ticker first, so
    the head is one instrument's January and the tail is a different instrument's
    December. The two halves describe different instruments, and nothing in the
    output says so.
    """
    return pd.concat([frame.head(5), frame.tail(5)]).to_markdown(index=False)


def shape_head_tail_describe(frame: pd.DataFrame) -> str:
    """Add summary statistics over the numeric columns to the previous shape.

    Statistics are the usual answer to "the model has no overview". They do give an
    overview, of the wrong thing: the spread of every close price of every
    instrument pooled together, which no per-instrument question can be read out of.
    """
    digest = pd.concat([frame.head(5), frame.tail(5)]).to_markdown(index=False)
    stats = frame[["close"]].describe().round(4).to_markdown()
    return f"{digest}\n\nSummary statistics over all rows:\n{stats}"


def shape_endpoints(frame: pd.DataFrame) -> str:
    """Return the first and last row of every instrument, and nothing else.

    This is the smallest shape that contains the answer. The change from the shapes
    above is not size; it is that the rows were chosen per group rather than off the
    ends of one flat table.
    """
    parts = []
    for ticker, group in frame.groupby("ticker"):
        group = group.sort_values("trade_date")
        parts.append(pd.concat([group.head(1), group.tail(1)]))
    return pd.concat(parts).to_markdown(index=False)


def shape_computed(frame: pd.DataFrame) -> str:
    """Return the endpoints with the percentage change already worked out.

    The arithmetic moves out of the model and into the tool. Everything the model
    still has to do is read a table and pick the largest absolute value.
    """
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


def pick_provider() -> tuple:
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
    """Send one request, backing off when the provider answers with a rate limit."""
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

    print("--- 1. The answer, computed from the database ---")
    print(truth.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    best = truth.loc[truth["change_pct"].abs().idxmax()]
    print(f"\n    largest absolute move: {best['ticker']} at {best['change_pct']:+.2f}%")

    print(f"\n--- 2. One query, five return shapes ---")
    print(f"    the query returns {len(frame):,} rows across {frame['ticker'].nunique()} "
          f"instruments, sorted by ticker then date")
    rendered = [(name, builder(frame)) for name, builder in SHAPES]
    for name, text in rendered:
        print(f"    {name:<34}{len(text):>7,} characters")

    print(f"\n--- 3-4. The same question against each shape ---")
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

    print("\n--- 5. Scored against the computed answer ---")
    print(f"    {'shape':<34}{'ticker':>8}{'named right':>13}{'claimed':>10}"
          f"{'truth':>9}{'error':>9}")
    truth_by_ticker = truth.set_index("ticker")["change_pct"].to_dict()
    for name, _, reply in results:
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
        print(f"    {name:<34}{str(ticker):>8}{named_right:>13}{claimed_text:>10}"
              f"{actual_text:>9}{error:>9}")
    print("\n    'named right' asks whether the reply picked the instrument that actually")
    print("    moved most. 'error' is measured against the truth for whichever instrument")
    print("    the reply named, so a reply can be precise about the wrong instrument.")

    print("\n--- 6. The two shapes that both fit in ten rows ---")
    ticker_pattern = re.compile(r"^\| ([A-Z]{3})", re.M)
    head_only = sorted(set(ticker_pattern.findall(rendered[0][1])))
    head_tail = sorted(set(ticker_pattern.findall(rendered[1][1])))
    print(f"    head(10) covers tickers       : {head_only}")
    print(f"    head(5)+tail(5) covers tickers: {head_tail}")
    print("    Both are ten rows. The second reaches two instruments and neither of them")
    print("    completely, which is why its answer can look reasonable and still be built")
    print("    from one instrument's January and another's December.")

    print("\n--- 7. Characters spent per shape ---")
    print(f"    {'shape':<34}{'characters':>12}{'named right':>13}")
    for (name, text, reply) in results:
        named_right = "yes" if reply.get("ticker") == best["ticker"] else "no"
        print(f"    {name:<34}{len(text):>12,}{named_right:>13}")
    print("\n    The shape that answers the question is not the largest one. Choosing rows")
    print("    per group rather than off the ends of the table is what changed the answer.")


if __name__ == "__main__":
    main()
