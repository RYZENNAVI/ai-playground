"""Hand the model a set of tools and let it decide which ones a question needs.

Demonstrates the step from writing SQL to acting on a database:
    1. Declare four tools: run a query, draw a chart, fit a model, rank drivers.
    2. Put the schema, the stored-code meanings and the settled questions into
       the system message.
    3. Loop: send the conversation, run whatever tool the model asks for, repeat.
    4. Answer a question that needs one query.
    5. Answer one that needs a query and then a chart.
    6. Answer one whose numbers are not in any column, only implied by totals.
    7. Answer one asking which factors move a figure the most.

Module 03: Text2SQL - Tool-Calling Agent.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

_db = import_module("01_build_insurance_db")

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

CHART_DIR = Path(__file__).parent / "data" / "charts"

# Ten turns is well past what any of these questions need, and short enough that
# a model going in circles stops rather than billing indefinitely.
MAX_TURNS = 10

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
average premium per segment - use fit_segment_premium when that is asked for.

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
    """Execute a SELECT and hand back rows the model can read.

    Errors are returned rather than raised. The model wrote this SQL, so a
    failure is information it can act on - raising would end the conversation
    instead of letting it fix the query.
    """
    try:
        cursor = connection.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()[:50]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}
    except sqlite3.Error as error:
        return {"error": str(error)}


def tool_plot_chart(connection, sql, title):
    """Render a two-column result as a bar chart and save it."""
    import matplotlib

    # Pick the non-interactive backend before pyplot is imported, or a machine
    # with no display will fail on import rather than on draw.
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
    """Recover per-segment premium from daily totals with a linear fit.

    Nothing in the database stores what one new customer pays on average. What
    it does store is how many of each segment signed on a given day and what
    came in that day, so the per-segment figures sit in the coefficients of

        total_premium = w_new * new + w_renewal * renewal + w_upgrade * upgrade

    Fitting that model is the only way to get them, which is what makes this a
    tool rather than a query. No intercept is fitted: a day with no sales takes
    no premium, and forcing the line through the origin keeps each coefficient
    interpretable as a price rather than a price plus a share of some constant.
    """
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


def tool_rank_drivers(connection):
    """Rank what moves daily premium, using a decision tree's split importance.

    A tree is used rather than a correlation because the candidate factors are a
    mix of counts and categories, and because a split-based ranking survives the
    fact that the campaign lift is multiplicative rather than additive.
    """
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
    return {
        "days_used": len(rows),
        "top_factors": [
            {"factor": name, "importance": round(float(score), 4)}
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
    """Run the tool-calling loop until the model answers in words.

    Each turn either produces a final message or a batch of tool calls. Tool
    results go back in as messages of their own, which is what lets the model
    chain one call into the next - query first, then chart what the query found.
    """
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
            preview = json.dumps(arguments)[:88]
            print(f"    -> {call.function.name}({preview})")
            result = dispatch(connection, call.function.name, arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str)[:3000],
            })

    return "stopped after the turn limit"


def main():
    db_path = _db.ensure_database()
    schema = _db.load_schema()
    client = make_client()
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    system = SYSTEM_TEMPLATE.format(schema=schema)

    try:
        print("--- 1. Tools declared ---")
        for tool in TOOLS:
            function = tool["function"]
            print(f"  {function['name']:<22}{function['description'][:60]}")

        print("\n--- 2. System message ---")
        print(f"  {len(system)} characters: schema, code meanings, settled")
        print("  questions, and one note telling the model which question the")
        print("  fitting tool answers.")

        print("\n--- 3./4. One query ---")
        # Governed by a settled question: closed accounts are not customers, so
        # the expected answer is 35 rather than the 40 rows in the table.
        question = "How many customers do we have?"
        print(f"  Q: {question}")
        print(converse(client, connection, system, question))

        print("\n--- 5. A query and then a chart ---")
        question = ("Chart the total amount claimed by claim type, then tell me "
                    "which type is largest.")
        print(f"  Q: {question}")
        print(converse(client, connection, system, question))

        print("\n--- 6. A number no column holds ---")
        question = ("What does a new customer pay on average compared with a "
                    "renewal, during the yearend campaign?")
        print(f"  Q: {question}")
        print(converse(client, connection, system, question))

        print("\n--- 7. Which factors move the total ---")
        question = "Which factors drive daily premium the most?"
        print(f"  Q: {question}")
        print(converse(client, connection, system, question))

        print("\n--- Ground truth for step 6 ---")
        # The generator used fixed prices, so the fit can be checked rather than
        # taken on faith. Anything close to these three numbers means the model
        # recovered figures the database never stored.
        print(f"  prices used to generate the data: {_db.SEGMENT_PREMIUM}")
        print(f"  campaign multipliers:             {_db.CAMPAIGN_LIFT}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
