"""Let the model pick tools to answer database questions, via function calling.

This script gives DeepSeek four tools through function calling (also called
tool calling): run a query, draw a bar chart, fit a linear regression, and
rank factors with a decision tree. The system message holds the schema, what
each stored code means and three settled questions. A hand-written loop sends
the conversation, runs whatever tools the model asks for and sends back the
results, until the model answers in words or ten turns pass.

Two of the tools reach numbers no column holds. daily_sales stores how many
new, renewal and upgrade policies sold each day and the day's total premium,
but not what one policy of each kind costs. A linear regression without an
intercept recovers those prices as its coefficients:

    total_premium = w_new * new + w_renewal * renewal + w_upgrade * upgrade

A day with no sales takes no premium, so the line goes through the origin and
each coefficient reads as a price. Script 01 generated the table from fixed
prices and campaign multipliers, so parts 5 and 6 check the tools against them.

The run prints six parts:
    1. Tools declared. Each tool's name and description.
    2. System message. Its length and what it holds.
    3. One query. "How many customers do we have?", which a settled question
       answers as 35, not the 40 rows in the table.
    4. A query and then a chart. Total claimed by claim type, saved as a PNG
       in data/charts.
    5. A number no column holds. New against renewal premium in the yearend
       campaign, checked against the prices that generated the data.
    6. Which factors move the total. The decision tree's ranking, checked
       against what the generator actually used.
"""

import json
import os
import sqlite3
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

_db = import_module("01_build_insurance_db")

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

CHART_DIR = Path(__file__).parent / "data" / "charts"

# Ten turns is well past what any of these questions need, and short enough that
# a model going in circles stops rather than billing indefinitely.
MAX_TURNS = 10

# run_sql sends back at most this many rows, so a large result cannot flood the
# conversation.
MAX_ROWS = 50

SYSTEM_TEMPLATE = """You answer questions about an insurance database by calling
tools. Write SQLite-compatible SQL only.

Schema:
{schema}

Stored codes:
- policies.policy_status: IF in force, LP lapsed, TM terminated
- policies.payment_status: P paid, NP not paid
- claims.claim_status: APP approved, PND pending, PAY paid, DEN denied
- customers.customer_status: A active, L lapsed, C closed

The daily_sales table holds one row per day with the number of policies sold to
each customer segment and the total premium taken that day. It does not hold the
average premium per segment. Use fit_segment_premium when that is asked for.

Settled questions. These are house rules, not facts the schema can tell you, so
follow them rather than deriving your own:

- "How many customers do we have?"
  SELECT COUNT(*) FROM customers WHERE customer_status IN ('A', 'L')
  Closed accounts are not counted as customers.

- "How much have we paid out in claims?"
  SELECT SUM(claim_amount) FROM claims WHERE claim_status = 'PAY'
  Only settled payouts count. Approved-but-unpaid claims are a liability, not
  an outgoing.

- "What is a policy worth per year?"
  premium * CASE payment_frequency WHEN 'Monthly' THEN 12
                                   WHEN 'Quarterly' THEN 4
                                   WHEN 'Annual' THEN 1 END
  Premium is held on products, not on policies, and it is per payment period
  rather than per year.

Answer in one or two sentences once you have the numbers."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a read-only SELECT and return the rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT statement.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_chart",
            "description": "Draw a bar chart from a query and save it as a PNG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A SELECT returning a label column then "
                                       "a numeric column.",
                    },
                    "title": {"type": "string"},
                },
                "required": ["sql", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fit_segment_premium",
            "description": "Recover the average premium per customer segment "
                           "from daily totals by fitting a linear model. Use "
                           "this whenever per-segment premium is asked for, "
                           "because no column stores it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "campaign": {
                        "type": "string",
                        "description": "Restrict to one campaign: none, spring, "
                                       "autumn, yearend, or all.",
                    }
                },
                "required": ["campaign"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_drivers",
            "description": "Rank which factors move daily premium the most, "
                           "using a decision tree.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def make_client():
    """Return an OpenAI-protocol client pointed at DeepSeek."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def tool_run_sql(connection, sql):
    """Execute a SELECT and return the rows, or the error so the model can fix
    its query instead of the conversation ending."""
    try:
        cursor = connection.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
    except sqlite3.Error as error:
        return {"error": str(error)}
    result = {
        "columns": columns, "rows": rows[:MAX_ROWS], "row_count": len(rows)
    }
    if len(rows) > MAX_ROWS:
        result["note"] = f"only the first {MAX_ROWS} rows are included"
    return result


def tool_plot_chart(connection, sql, title):
    """Render a two-column result as a bar chart and save it."""
    import matplotlib

    # Agg draws to files only, so no window opens.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        rows = connection.execute(sql).fetchall()
    except sqlite3.Error as error:
        return {"error": str(error)}
    if not rows or len(rows[0]) < 2:
        return {"error": "the query must return a label column and a number"}

    labels = [str(row[0]) for row in rows]
    values = [float(row[1]) for row in rows]

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() else "_" for c in title.lower())[:50]
    path = CHART_DIR / f"{safe_name}.png"

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color="#4c72b0")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return {"saved_to": str(path), "bars": len(labels)}


def tool_fit_segment_premium(connection, campaign="all"):
    """Fit total premium on the three daily counts, with no intercept, and
    return the coefficients as per-segment prices with the fit's R^2."""
    from sklearn.linear_model import LinearRegression

    sql = ("SELECT new_count, renewal_count, upgrade_count, total_premium "
           "FROM daily_sales")
    params = ()
    if campaign and campaign != "all":
        sql += " WHERE campaign = ?"
        params = (campaign,)
    rows = connection.execute(sql, params).fetchall()
    if len(rows) < 10:
        return {"error": f"only {len(rows)} days match; too few to fit"}

    features = [[row[0], row[1], row[2]] for row in rows]
    target = [row[3] for row in rows]
    model = LinearRegression(fit_intercept=False).fit(features, target)

    return {
        "campaign": campaign,
        "days_used": len(rows),
        "average_premium": {
            "new": round(float(model.coef_[0]), 2),
            "renewal": round(float(model.coef_[1]), 2),
            "upgrade": round(float(model.coef_[2]), 2),
        },
        "fit_quality_r2": round(float(model.score(features, target)), 4),
    }


def driver_importances(connection):
    """Return every factor's decision-tree importance, highest first. A tree
    mixes counts with categories, and copes with a lift that multiplies."""
    from sklearn.tree import DecisionTreeRegressor

    rows = connection.execute(
        "SELECT new_count, renewal_count, upgrade_count, campaign, "
        "is_month_end, weekday, total_premium FROM daily_sales"
    ).fetchall()

    campaigns = ["none", "spring", "autumn", "yearend"]
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    names = ["new_count", "renewal_count", "upgrade_count", "is_month_end"]
    names += [f"campaign={c}" for c in campaigns]
    names += [f"weekday={d}" for d in weekdays]

    features, target = [], []
    for new, renewal, upgrade, campaign, month_end, weekday, total in rows:
        row = [new, renewal, upgrade, month_end]
        row += [1 if campaign == c else 0 for c in campaigns]
        row += [1 if weekday == d else 0 for d in weekdays]
        features.append(row)
        target.append(total)

    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(features, target)
    ranked = sorted(
        zip(names, model.feature_importances_), key=lambda p: p[1], reverse=True
    )
    return len(rows), [(name, float(score)) for name, score in ranked]


def tool_rank_drivers(connection):
    """Return the five most important factors that have any importance."""
    days, ranked = driver_importances(connection)
    return {
        "days_used": days,
        "top_factors": [
            {"factor": name, "importance": round(score, 4)}
            for name, score in ranked[:5]
            if score > 0
        ],
    }


def dispatch(connection, name, arguments):
    """Route one tool call to its implementation."""
    if name == "run_sql":
        return tool_run_sql(connection, arguments["sql"])
    if name == "plot_chart":
        return tool_plot_chart(connection, arguments["sql"], arguments["title"])
    if name == "fit_segment_premium":
        return tool_fit_segment_premium(connection, arguments.get("campaign", "all"))
    if name == "rank_drivers":
        return tool_rank_drivers(connection)
    return {"error": f"no tool named {name}"}


def converse(client, connection, system, question):
    """Run the tool-calling loop until the model answers in words. Tool results
    go back as messages, so the model can chain one call into the next."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, temperature=0.0
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return message.content

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            arguments = json.loads(call.function.arguments or "{}")
            preview = json.dumps(arguments)
            if len(preview) > 88:
                preview = preview[:85] + "..."
            print(f"    -> {call.function.name}({preview})")
            result = dispatch(connection, call.function.name, arguments)
            if "saved_to" in result:
                here = Path(__file__).parent
                saved = Path(result["saved_to"]).relative_to(here)
                print(f"       saved to {saved.as_posix()}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str),
            })

    return "stopped after the turn limit"


def main():
    db_path = _db.ensure_database()
    schema = _db.load_schema()
    client = make_client()
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    system = SYSTEM_TEMPLATE.format(schema=schema)

    try:
        # 1. Tools declared
        print("--- 1. Tools declared ---")
        for tool in TOOLS:
            function = tool["function"]
            print(f"  {function['name']}")
            for line in textwrap.wrap(function["description"], 68):
                print(f"      {line}")

        # 2. System message
        print("\n--- 2. System message ---")
        print(f"  {len(system)} characters: schema, code meanings, settled")
        print("  questions, and one note telling the model which question the")
        print("  fitting tool answers.")

        # 3. One query
        print("\n--- 3. One query ---")
        # Governed by a settled question: closed accounts are not customers, so
        # the expected answer is 35 rather than the 40 rows in the table.
        question = "How many customers do we have?"
        print(f"  Q: {question}")
        print(converse(client, connection, system, question))

        # 4. A query and then a chart
        print("\n--- 4. A query and then a chart ---")
        question = ("Chart the total amount claimed by claim type, then tell me "
                    "which type is largest.")
        print(f"  Q: {question}")
        print(converse(client, connection, system, question))

        # 5. A number no column holds
        print("\n--- 5. A number no column holds ---")
        question = ("What does a new customer pay on average compared with a "
                    "renewal, during the yearend campaign?")
        print(f"  Q: {question}")
        print(converse(client, connection, system, question))
        # The fit is deterministic, so running it again gives the numbers the
        # model received. The generator priced yearend days at base x lift.
        fit = tool_fit_segment_premium(connection, "yearend")
        lift = _db.CAMPAIGN_LIFT["yearend"]
        print(f"\n  Check: the fit used {fit['days_used']} yearend days, "
              f"R^2 {fit['fit_quality_r2']}.")
        for segment, base in _db.SEGMENT_PREMIUM.items():
            recovered = fit["average_premium"][segment]
            true = base * lift
            print(f"    {segment:<9}fit {recovered:>8.2f}   generator "
                  f"{base} x {lift} = {true:.1f}   "
                  f"({(recovered - true) / true:+.1%})")

        # 6. Which factors move the total
        print("\n--- 6. Which factors move the total ---")
        question = "Which factors drive daily premium the most?"
        print(f"  Q: {question}")
        print(converse(client, connection, system, question))
        # The generator priced each day from the three counts and the campaign
        # only, so month end and weekday should get nothing.
        days, ranked = driver_importances(connection)
        unused = sum(s for n, s in ranked
                     if n == "is_month_end" or n.startswith("weekday="))
        campaign = sum(s for n, s in ranked if n.startswith("campaign="))
        yearend_days = connection.execute(
            "SELECT COUNT(*) FROM daily_sales WHERE campaign = 'yearend'"
        ).fetchone()[0]
        print(f"\n  Check: is_month_end and weekday get {unused:.2f} together.")
        if unused == 0:
            print("  The generator never used them, and the tree agrees.")
        else:
            print("  The generator never used them, so this is noise.")
        print(f"  The campaign factors get {campaign:.2f} together. Yearend")
        print(f"  days are priced {lift - 1:.0%} higher, but they are "
              f"{yearend_days} of {days}")
        print("  days, so a low importance is not a small effect.")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
