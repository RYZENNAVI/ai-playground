"""Turn a question into SQL by prompting alone, then measure which prompt wins.

Demonstrates the two things that decide accuracy before any framework is added:
    1. Load the schema and a set of questions with known-correct answers.
    2. Build the same question three ways: prose schema, prose plus an explicit
       instruction, and raw CREATE TABLE text ending in a code fence.
    3. Ask the model for SQL under each of the three prompts.
    4. Execute what comes back and compare it against the known answer.
    5. Score the three styles side by side.
    6. Retrieve similar past questions and their verified SQL.
    7. Re-ask the losing questions with those examples pasted in.

Module 03: Text2SQL - Prompt Styles and Retrieved Examples.
"""

import os
import re
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

# A plain-English paragraph naming the same four tables and columns. It carries
# no types and no allowed values - that absence is exactly what separates it from
# the CREATE TABLE text style C uses. Nothing else about the three prompts
# changes, so any score gap traces back to those two missing pieces.
PROSE_SCHEMA = """Table customers: customer_id, name, gender, date_of_birth,
marital_status, occupation, phone_number, email, city, registered_on,
customer_status.
Table products: product_id, product_name, product_type, coverage_amount,
coverage_years, premium, payment_frequency, sales_region, product_status.
Table policies: policy_number, customer_id, product_id, policy_status,
beneficiary, relationship, start_date, end_date, payment_status, payment_method.
Table claims: claim_number, policy_number, claim_date, claim_type, claim_amount,
claim_status, handler, review_date, denial_reason."""

# Style A - prose schema, question wrapped in a comment block.
PROMPT_A = """# language: SQL
/*
{question} First decide which tables and columns you need, then write the SQL.
The database has these tables:
=====
{schema}
*/
# {question}"""

# Style B - same prose schema, but the instruction to produce one query is
# spelled out at the end instead of being implied.
PROMPT_B = """-- language: SQL
/*{question}
Here are the tables
=====
{schema}
=====
Write one SQL query: {question}
*/"""

# Style C - the real CREATE TABLE text, laid out as a completion the model
# finishes. The trailing fence is deliberate: these models were trained on code,
# and an unclosed ```sql block is the strongest possible signal that SQL comes
# next rather than an explanation.
PROMPT_C = """-- language: SQL
### Question: {question}
### Input: {schema}
### Response:
Here is the SQL query I have generated to answer the question `{question}`:
```sql
"""

# Each style is paired with the schema text it is allowed to see. A and B get
# the paragraph, C gets the CREATE TABLE statements. Pairing them here rather
# than at the call site is what stops the two from being mixed up.
STYLES = {
    "A prose": (PROMPT_A, "prose"),
    "B prose+ask": (PROMPT_B, "prose"),
    "C create-table": (PROMPT_C, "ddl"),
}

# Each question ships with three things: the SQL a human wrote for it, and the
# stored literal the query cannot do without (None when the question needs no
# literal). The generated query is never compared as text - only the rows it
# returns are compared, so a different but equivalent query still counts as
# correct. The literal is checked separately, because a strong model sometimes
# guesses the right value; scoring the guess as knowledge would hide the very
# gap this experiment is measuring.
BENCHMARK = [
    # The first two need only column names, so every style should get them.
    # That control only holds if each question has one defensible answer. The
    # second one joins through claims, where a customer with two large claims
    # comes back twice, so "list each customer once" is spelled out rather than
    # left implied - without it a query that omits DISTINCT answers the question
    # as asked and still scores as wrong, and the resulting gap between styles
    # would be about de-duplication rather than about the schema text.
    #
    # "only" earns its place the same way. Rows are compared as whole tuples, so
    # a query that also selects customer_id is scored wrong even though it found
    # the right customers. Some phrasings pull the model towards adding a key
    # column; on a control question that has to be closed off in the wording.
    (
        "List the name and phone number of every customer.",
        "SELECT name, phone_number FROM customers",
        None,
    ),
    (
        "Which customers filed claims over 10000? List each customer once, "
        "showing only their name and phone number.",
        "SELECT DISTINCT c.name, c.phone_number FROM customers c "
        "JOIN policies p ON c.customer_id = p.customer_id "
        "JOIN claims cl ON p.policy_number = cl.policy_number "
        "WHERE cl.claim_amount > 10000",
        None,
    ),
    # The rest hinge on information only the CREATE TABLE text carries: the exact
    # stored values behind a status column, and the fact that dates are stored as
    # ISO strings. The prose schema lists the same column names but says nothing
    # about what goes in them, so a model reading it has to guess.
    (
        "Which claims were turned down? Show the claim number and the reason.",
        "SELECT claim_number, denial_reason FROM claims WHERE claim_status = 'DEN'",
        'DEN',
    ),
    (
        "Which customers have lapsed? Show their id and name.",
        "SELECT customer_id, name FROM customers WHERE customer_status = 'L'",
        'L',
    ),
    (
        "How many policies are still running? Count them.",
        "SELECT COUNT(*) FROM policies WHERE policy_status = 'IF'",
        'IF',
    ),
    (
        "Which customers signed up during 2023? Show id, name and sign-up date.",
        "SELECT customer_id, name, registered_on FROM customers "
        "WHERE registered_on >= '2023-01-01' AND registered_on < '2024-01-01'",
        None,
    ),
    (
        "For each product type, give the average premium and how many policies "
        "were sold under it.",
        "SELECT pr.product_type, AVG(pr.premium) AS average_premium, "
        "COUNT(po.policy_number) AS policy_count FROM policies po "
        "JOIN products pr ON po.product_id = pr.product_id "
        "GROUP BY pr.product_type",
        None,
    ),
]

# Questions already answered correctly at some point, paired with the SQL that
# worked. Step 6 retrieves from this list; step 7 pastes what it finds into the
# prompt. Only verified SQL belongs here - a wrong example teaches the model to
# repeat the mistake.
VERIFIED_EXAMPLES = [
    (
        "How many customers do we have?",
        "SELECT COUNT(*) FROM customers",
    ),
    (
        "Which customers are married?",
        "SELECT customer_id, name FROM customers WHERE marital_status = 'Married'",
    ),
    (
        "List every claim that is still pending, with its handler.",
        "SELECT claim_number, handler, claim_date FROM claims "
        "WHERE claim_status = 'PND'",
    ),
    (
        "Show each customer together with the policies they hold.",
        "SELECT c.name, p.policy_number, p.policy_status FROM customers c "
        "JOIN policies p ON c.customer_id = p.customer_id",
    ),
    (
        "What is the total claim amount per claim type?",
        "SELECT claim_type, SUM(claim_amount) FROM claims GROUP BY claim_type",
    ),
    (
        "Which products are no longer sold?",
        "SELECT product_name, product_type FROM products "
        "WHERE product_status = 'Retired'",
    ),
]


def make_client():
    """Return an OpenAI-protocol client pointed at DeepSeek."""
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def extract_sql(text):
    """Pull a bare SQL statement out of whatever the model returned.

    Style C ends on an open code fence, so the reply often starts mid-block and
    has no opening fence at all. Handling both shapes here keeps the scoring
    about the SQL rather than about who formatted their answer more tidily.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        text = text.split("```")[0]
    text = text.strip()
    # Drop any prose the model put before the statement.
    match = re.search(r"\b(SELECT|WITH)\b", text, re.I)
    if match:
        text = text[match.start():]
    return text.rstrip().rstrip(";").strip()


def ask_for_sql(client, prompt):
    """Send one prompt and return the SQL it produced."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return extract_sql(response.choices[0].message.content)


def run_sql(connection, sql):
    """Execute a query and return its rows, or None if it will not run.

    A query that raises is simply wrong for scoring purposes, so the failure is
    swallowed here and surfaces as a miss in the score table.
    """
    try:
        return connection.execute(sql).fetchall()
    except sqlite3.Error:
        return None


def rows_match(left, right):
    """Compare two result sets ignoring row order and float noise.

    Row order is not part of the question unless the question asked for it, and
    averages come back with different trailing digits depending on how the query
    was written, so both are normalised away before comparing.
    """
    if left is None or right is None:
        return False
    if len(left) != len(right):
        return False

    def normalise(rows):
        out = []
        for row in rows:
            out.append(tuple(
                round(value, 4) if isinstance(value, float) else value
                for value in row
            ))
        return sorted(out, key=repr)

    return normalise(left) == normalise(right)


def score_styles(client, connection, schema):
    """Run every question under all three prompt styles and tally two scores.

    The first score is whether the rows came back right. The second is whether
    the query used the literal the database actually stores, which is the piece
    of information only the CREATE TABLE text supplies. Both are needed: a
    strong model sometimes guesses a value correctly, and scoring that guess as
    knowledge would hide the gap this experiment exists to measure.
    """
    rows_ok = {name: [] for name in STYLES}
    literal_ok = {name: [] for name in STYLES}
    schemas = {"prose": PROSE_SCHEMA, "ddl": schema}
    for question, gold, literal in BENCHMARK:
        expected = run_sql(connection, gold)
        print(f"\n  Q: {question}")
        for name, (template, schema_kind) in STYLES.items():
            sql = ask_for_sql(
                client,
                template.format(question=question, schema=schemas[schema_kind]),
            )
            actual = run_sql(connection, sql)
            ok = rows_match(actual, expected)
            rows_ok[name].append(ok)
            note = ""
            if literal is not None:
                used = re.search(r"['\"]" + re.escape(literal.strip("'")) + r"['\"]", sql) is not None
                literal_ok[name].append(used)
                note = f"   literal {literal}: {'used' if used else 'GUESSED'}"
            status = "pass" if ok else ("error" if actual is None else "wrong rows")
            print(f"    {name:<16} {status:<11}{note}")
    return rows_ok, literal_ok


def build_retriever():
    """Return a function that finds the most similar verified examples.

    Similarity here is lexical, not semantic: the questions are short, share a
    vocabulary, and live in one domain, so term overlap separates them cleanly.
    It also keeps this script on a single provider, since the chat endpoint used
    above has no embedding counterpart.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    questions = [q for q, _ in VERIFIED_EXAMPLES]
    vectoriser = TfidfVectorizer(stop_words="english").fit(questions)
    matrix = vectoriser.transform(questions)

    def retrieve(query, top_k=2, threshold=0.05):
        scores = cosine_similarity(vectoriser.transform([query]), matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [
            (VERIFIED_EXAMPLES[i][0], VERIFIED_EXAMPLES[i][1], score)
            for i, score in ranked[:top_k]
            if score >= threshold
        ]

    return retrieve


def prompt_with_examples(question, schema, examples):
    """Build style C again, with retrieved question/SQL pairs pasted in front."""
    block = "\n".join(
        f"-- Q: {q}\n{sql};" for q, sql, _ in examples
    )
    return f"""-- language: SQL
### Similar questions answered before:
{block}

### Question: {question}
### Input: {schema}
### Response:
Here is the SQL query I have generated to answer the question `{question}`:
```sql
"""


def main():
    db_path = _db.ensure_database()
    schema = _db.load_schema()
    client = make_client()
    connection = sqlite3.connect(db_path)

    try:
        print("--- 1. Benchmark ---")
        print(f"  {len(BENCHMARK)} questions, each with a hand-written answer")
        print(f"  database: {db_path.name}")

        print("\n--- 2. Three prompt styles ---")
        print("  A prose         schema as a paragraph, question in a comment")
        print("  B prose+ask     same paragraph, explicit 'write one SQL query'")
        print("  C create-table  raw CREATE TABLE text, reply opens inside ```sql")

        print("\n--- 3./4. Generating and executing ---")
        rows_ok, literal_ok = score_styles(client, connection, schema)

        print("\n--- 5. Scores ---")
        total = len(BENCHMARK)
        literal_total = sum(1 for _, _, lit in BENCHMARK if lit is not None)
        print(f"  {'style':<16} {'rows right':<13}stored literal used")
        for name in STYLES:
            print(f"  {name:<16} {f'{sum(rows_ok[name])}/{total}':<13}"
                  f"{sum(literal_ok[name])}/{literal_total}")
        print("\n  Rows-right can be reached by guessing a value that happens to")
        print("  match. The literal column cannot - it only rises when the prompt")
        print("  actually told the model what the column stores.")

        print("\n--- 6. Retrieving similar verified examples ---")
        retrieve = build_retriever()
        failed = [
            BENCHMARK[i]
            for i, ok in enumerate(rows_ok["C create-table"])
            if not ok
        ]
        if not failed:
            print("  style C answered everything; showing retrieval on one question anyway")
            failed = BENCHMARK[-1:]

        for question, _, _ in failed:
            found = retrieve(question)
            print(f"\n  Q: {question}")
            for example_q, _, score in found:
                print(f"    {score:.3f}  {example_q}")

        print("\n--- 7. Re-asking with those examples ---")
        for question, gold, _ in failed:
            examples = retrieve(question)
            if not examples:
                print(f"  no example passed the threshold for: {question}")
                continue
            sql = ask_for_sql(client, prompt_with_examples(question, schema, examples))
            ok = rows_match(run_sql(connection, sql), run_sql(connection, gold))
            print(f"  {'pass' if ok else 'still wrong'}: {question}")
            print(f"    {sql}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
