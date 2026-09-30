"""This script asks a chat model to write SQL for 7 questions about the insurance
database from script 01, and scores the queries by running them. Turning a
question into SQL is called Text2SQL. The questions are asked under three prompt
styles, and the last two parts add few-shot examples found by retrieval. Styles A
and B describe four of the tables in a paragraph that names the columns only.
Style C pastes the CREATE TABLE text, with its column comments, into a completion
template that ends inside an open ```sql block. C changes both the template and
the schema, while A and B differ only in wording.

Scoring is by execution accuracy: the generated query runs, and its rows are
compared with the rows of a hand-written query. Three questions filter on a
status column, and the table stores each status as a code, such as 'DEN' for a
denied claim. Only the column comments in style C give those codes, so the run
also checks whether each query used the code the table stores, and prints the
value it wrote instead.

The last two parts try few-shot retrieval. Six hand-written question and SQL
pairs are searched by TF-IDF, and the closest ones are pasted in front of a
question the schema cannot answer: how many customers Kingsford has. The house
rule is that closed accounts are not customers, and only the first example
follows it. Part 6 asks three times: without examples, with the examples as
bare SQL, and with the reason for the rule written above the SQL.

The run prints six parts:
    1. Benchmark. The 7 questions.
    2. Three prompt styles.
    3. Generating and executing. Each question under each style, with the
       result and, on the three status questions, what the table stores and
       what the query wrote.
    4. Scores. Rows right and table codes used, per style.
    5. Retrieving similar verified examples. The Kingsford question, the words
       it shares with the example questions, and the closest examples.
    6. Asking with and without the examples. The count from each of the three
       prompts, against the hand-written answer.
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

# A paragraph naming four of the five tables and their columns. It leaves out
# daily_sales, which no question uses, and it carries no types, comments or
# status codes. Style C changes the template as well as the schema, so only A
# against B is a one-change comparison.
PROSE_SCHEMA = """Table customers: customer_id, name, gender, date_of_birth,
marital_status, occupation, phone_number, email, city, registered_on,
customer_status.
Table products: product_id, product_name, product_type, coverage_amount,
coverage_years, premium, payment_frequency, sales_region, product_status.
Table policies: policy_number, customer_id, product_id, policy_status,
beneficiary, relationship, start_date, end_date, payment_status, payment_method.
Table claims: claim_number, policy_number, claim_date, claim_type, claim_amount,
claim_status, handler, review_date, denial_reason."""

# Style A: prose schema, question wrapped in a comment block.
PROMPT_A = """# language: SQL
/*
{question} First decide which tables and columns you need, then write the SQL.
The database has these tables:
=====
{schema}
*/
# {question}"""

# Style B: same prose schema, but the instruction to produce one query is
# spelled out at the end instead of being implied.
PROMPT_B = """-- language: SQL
/*{question}
Here are the tables
=====
{schema}
=====
Write one SQL query: {question}
*/"""

# Style C: the real CREATE TABLE text, laid out as a completion the model
# finishes. The prompt ends inside an open ```sql block, so the reply starts
# with SQL rather than an explanation. extract_sql() handles that shape.
PROMPT_C = """-- language: SQL
### Question: {question}
### Input: {schema}
### Response:
Here is the SQL query I have generated to answer the question `{question}`:
```sql
"""

# Each style is paired here with the schema text it may see: A and B get the
# paragraph, C gets the CREATE TABLE statements.
STYLES = {
    "A prose": (PROMPT_A, "prose"),
    "B prose+ask": (PROMPT_B, "prose"),
    "C create-table": (PROMPT_C, "ddl"),
}

# Each entry holds the question, the SQL a human wrote for it, and the status
# code the table stores and the query cannot do without (None when it needs
# none). The generated query is never compared as text. Only the rows it returns
# are compared, so a different but equivalent query still counts as correct. The
# code is checked separately to show why a query failed. A lucky guess of the
# code would still count as used.
BENCHMARK = [
    # The first two need only column names, so every style should get them.
    # That control only holds if each question has one defensible answer. The
    # second one joins through claims, where a customer with two large claims
    # comes back twice, so "list each customer once" is spelled out rather than
    # left implied. Without it a query that omits DISTINCT answers the question
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
    # The next three need a status code that only the CREATE TABLE comments give.
    # The prose schema names the same columns but not what goes in them, so a
    # model reading it has to guess.
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
    # The last two need no status code either: a date range, and a join with an
    # average.
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

# Hand-written question and SQL pairs, each checked against the database, with
# the reason for any house rule the SQL follows. Part 5 retrieves from this list
# and part 6 pastes what it finds into the prompt. Only checked SQL belongs here,
# because a wrong example teaches the model the mistake.
VERIFIED_EXAMPLES = [
    (
        "How many customers do we have?",
        "SELECT COUNT(*) FROM customers WHERE customer_status IN ('A', 'L')",
        "Closed accounts are not counted as customers.",
    ),
    (
        "Which customers are married?",
        "SELECT customer_id, name FROM customers WHERE marital_status = 'Married'",
        None,
    ),
    (
        "List every claim that is still pending, with its handler.",
        "SELECT claim_number, handler, claim_date FROM claims "
        "WHERE claim_status = 'PND'",
        None,
    ),
    (
        "Show each customer together with the policies they hold.",
        "SELECT c.name, p.policy_number, p.policy_status FROM customers c "
        "JOIN policies p ON c.customer_id = p.customer_id",
        None,
    ),
    (
        "What is the total claim amount per claim type?",
        "SELECT claim_type, SUM(claim_amount) FROM claims GROUP BY claim_type",
        None,
    ),
    (
        "Which products are no longer sold?",
        "SELECT product_name, product_type FROM products "
        "WHERE product_status = 'Retired'",
        None,
    ),
]

# Kingsford has six customers, one of them closed. The schema does not say that
# closed accounts do not count, so the answer depends on the first example.
RETRIEVAL_QUESTION = "How many customers do we have in Kingsford?"
RETRIEVAL_ANSWER = (
    "SELECT COUNT(*) FROM customers "
    "WHERE city = 'Kingsford' AND customer_status IN ('A', 'L')"
)


def make_client():
    """Return an OpenAI-protocol client for the provider chosen above."""
    key = API_KEY
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY or OPENAI_API_KEY in .env and retry.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def extract_sql(text):
    """Pull a bare SQL statement out of the reply, fenced or not.
    Style C's reply usually starts inside the block and has no opening fence."""
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
    The caller prints None as 'error'."""
    try:
        return connection.execute(sql).fetchall()
    except sqlite3.Error:
        return None


def rows_match(left, right):
    """Compare two result sets ignoring row order and float noise.
    No question asks for an order, and averages differ in the last digits."""
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
    """Run every question under all three styles. Tally whether the rows came
    back right and, on the status questions, whether the table's code was used."""
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
                used = re.search(r"['\"]" + re.escape(literal) + r"['\"]", sql) is not None
                literal_ok[name].append(used)
                if used:
                    note = f"   table stores '{literal}', query used it"
                else:
                    wrote = ", ".join(f"'{v}'" for v in re.findall(r"'([^']*)'", sql))
                    note = f"   table stores '{literal}', query wrote {wrote or 'no quoted value'}"
            status = "pass" if ok else ("error" if actual is None else "wrong rows")
            print(f"    {name:<16} {status:<11}{note}")
    return rows_ok, literal_ok


def build_retriever():
    """Return a TF-IDF search over the example questions. The vocabulary comes
    from the six examples only, so words new to them are dropped from a query."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    questions = [example[0] for example in VERIFIED_EXAMPLES]
    vectoriser = TfidfVectorizer(stop_words="english").fit(questions)
    matrix = vectoriser.transform(questions)
    analyse = vectoriser.build_analyzer()

    def retrieve(query, top_k=2, threshold=0.05):
        """Return the query's words that are in the vocabulary, and the hits."""
        kept = [word for word in analyse(query) if word in vectoriser.vocabulary_]
        scores = cosine_similarity(vectoriser.transform([query]), matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        hits = [
            (VERIFIED_EXAMPLES[i], score)
            for i, score in ranked[:top_k]
            if score >= threshold
        ]
        return kept, hits

    return retrieve


def prompt_with_examples(question, schema, hits, reasons):
    """Build style C again, with retrieved question/SQL pairs pasted in front.
    With reasons, an example's house rule goes above its SQL as a comment."""
    lines = []
    for (example_q, sql, reason), _ in hits:
        lines.append(f"-- Q: {example_q}")
        if reasons and reason:
            lines.append(f"-- {reason}")
        lines.append(f"{sql};")
    block = "\n".join(lines)
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
        # 1. Benchmark
        print("--- 1. Benchmark ---")
        print(f"  {len(BENCHMARK)} questions, each with a hand-written answer")
        print(f"  database: {db_path.name}")

        # 2. Three prompt styles
        print("\n--- 2. Three prompt styles ---")
        print("  A prose         schema as a paragraph, question in a comment")
        print("  B prose+ask     same paragraph, explicit 'write one SQL query'")
        print("  C create-table  raw CREATE TABLE text, reply opens inside ```sql")

        # 3. Generating and executing
        print("\n--- 3. Generating and executing ---")
        rows_ok, literal_ok = score_styles(client, connection, schema)

        # 4. Scores
        print("\n--- 4. Scores ---")
        total = len(BENCHMARK)
        literal_total = sum(1 for _, _, lit in BENCHMARK if lit is not None)
        print(f"  {'style':<16} {'rows right':<13}table's code used")
        for name in STYLES:
            print(f"  {name:<16} {f'{sum(rows_ok[name])}/{total}':<13}"
                  f"{sum(literal_ok[name])}/{literal_total}")
        print("\n  The last column counts queries that used the code the table stores,")
        print("  such as 'DEN'. A lucky guess of that code would count too.")

        # 5. Retrieving similar verified examples
        print("\n--- 5. Retrieving similar verified examples ---")
        retrieve = build_retriever()
        kept, hits = retrieve(RETRIEVAL_QUESTION)
        print(f"  Q: {RETRIEVAL_QUESTION}")
        print(f"  words it shares with the example questions: "
              f"{', '.join(kept) or 'none'}")
        for (example_q, _, _), score in hits:
            print(f"    {score:.3f}  {example_q}")

        # 6. Asking with and without the examples
        print("\n--- 6. Asking with and without the examples ---")
        expected = run_sql(connection, RETRIEVAL_ANSWER)
        print(f"  hand-written SQL: {expected[0][0]}")
        prompts = [
            ("no examples",
             PROMPT_C.format(question=RETRIEVAL_QUESTION, schema=schema)),
            ("examples, SQL only",
             prompt_with_examples(RETRIEVAL_QUESTION, schema, hits, reasons=False)),
            ("examples with reason",
             prompt_with_examples(RETRIEVAL_QUESTION, schema, hits, reasons=True)),
        ]
        for label, prompt in prompts:
            sql = ask_for_sql(client, prompt)
            rows = run_sql(connection, sql)
            if rows is None:
                shown = "error"
            elif len(rows) == 1 and len(rows[0]) == 1:
                shown = str(rows[0][0])
            else:
                shown = f"{len(rows)} rows"
            verdict = "right" if rows_match(rows, expected) else "wrong"
            print(f"  {label:<22}{shown:<8}{verdict}")
            print(f"      {' '.join(sql.split())}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
