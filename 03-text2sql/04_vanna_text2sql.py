"""Generate SQL with Vanna, which retrieves the schema, notes and examples it needs.

Vanna applies retrieval-augmented generation to Text2SQL. It stores three kinds of
training material in a vector store (Chroma, embedded locally): the five CREATE
TABLE statements, five written notes that explain the status codes and where
premium lives, and three question and SQL pairs. For each question it retrieves
related items of each kind and pastes them into the prompt for DeepSeek.

Vanna returns up to 10 items of each kind by default, and this store holds 5, 5
and 3, so every question gets all of them. Retrieval here only orders them.

The run prints seven parts:
    1. Composing the client. A Chroma store and the DeepSeek chat model.
    2. Connecting to the database.
    3. Training. The counts of each kind of material.
    4. What the question retrieves. The lapsed-policy question's top items of
       each kind. This part only shows them: part 5 retrieves again by itself.
    5. Generating and running. The SQL and its result, checked against a
       hand-written query that converts each premium to a yearly amount.
    6. Correcting a wrong answer. "How many customers do we have?" before
       correction, and the settled SQL stored as a new pair.
    7. Asking again. The same question, then the same question reworded, and
       whether each one used the stored correction.
"""

import contextlib
import io
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

_db = import_module("01_build_insurance_db")

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

STORE_PATH = Path(__file__).parent / "data" / "vanna_store"

# Written notes are the piece the other two approaches have no room for. Script
# 02 could only carry what fits in one prompt; script 03 lost the comments to
# reflection. Here the meanings live in the store and are retrieved by question.
# This store is small enough that every question gets all five.
DOCUMENTATION = [
    "policy_status is stored as a two-letter code. IF means the policy is in "
    "force, LP means it has lapsed, TM means it was terminated.",
    "claim_status is stored as a three-letter code. APP means approved, PND "
    "means pending review, PAY means paid out, DEN means denied.",
    "customer_status is stored as a single letter. A means active, L means "
    "lapsed, C means closed.",
    "payment_status on a policy is P when the premium was paid and NP when it "
    "was not.",
    "Premium is held on the products table, not on the policy. To total or "
    "average premiums for policies, join policies to products on product_id.",
]

# Question and SQL pairs. Retrieval matches on the question text, so these teach
# by example rather than by rule.
TRAINING_PAIRS = [
    (
        "Which claims were denied, and why?",
        "SELECT claim_number, denial_reason FROM claims WHERE claim_status = 'DEN'",
    ),
    (
        "List the claims that are still pending review.",
        "SELECT claim_number, handler, claim_date FROM claims "
        "WHERE claim_status = 'PND'",
    ),
    (
        "What is the total premium across all policies that are in force?",
        "SELECT SUM(pr.premium) FROM policies po "
        "JOIN products pr ON po.product_id = pr.product_id "
        "WHERE po.policy_status = 'IF'",
    ),
]

QUESTION = "How many policies have lapsed, and what do they cost in premium per year?"
# premium is the price per payment period, so a yearly figure multiplies it by
# the number of periods in a year.
REFERENCE_SQL = (
    "SELECT COUNT(*), SUM(pr.premium * CASE pr.payment_frequency "
    "WHEN 'Monthly' THEN 12 WHEN 'Quarterly' THEN 4 ELSE 1 END) "
    "FROM policies po JOIN products pr ON po.product_id = pr.product_id "
    "WHERE po.policy_status = 'LP'"
)

CORRECTION_QUESTION = "How many customers do we have?"
REWORDED_QUESTION = "How many customers are there in total?"
# The answer a reviewer settled on for the question above. Counting customers
# looks unambiguous until someone asks whether closed accounts still count. The
# house rule here is that they do not, and no amount of schema reading would
# reveal that. It is a decision, not a fact about the data.
CORRECTED_SQL = (
    "SELECT COUNT(*) FROM customers WHERE customer_status IN ('A', 'L')"
)


def quiet(call, *args, **kwargs):
    """Run a Vanna call without its own prints (full DDL on training, token counts)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args, **kwargs)


def build_client(fresh=False):
    """Compose a Vanna client from a Chroma store and an OpenAI-protocol chat model.
    In vanna 2 these classes live under vanna.legacy."""
    from openai import OpenAI
    from vanna.legacy.chromadb import ChromaDB_VectorStore
    from vanna.legacy.openai import OpenAI_Chat

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Add it to your .env file.")

    if fresh and STORE_PATH.exists():
        shutil.rmtree(STORE_PATH)
    STORE_PATH.mkdir(parents=True, exist_ok=True)

    class LocalVanna(ChromaDB_VectorStore, OpenAI_Chat):
        """Vector store on disk, chat model over the network. Each parent gets only
        its own config keys, because the chat half rejects the store's."""

        def __init__(self, config):
            ChromaDB_VectorStore.__init__(
                self, config={"path": config["path"]}
            )
            OpenAI_Chat.__init__(
                self, client=config["client"], config={"model": config["model"]}
            )

        def log(self, message, title="Info"):
            """Drop Vanna's log calls, which print the whole prompt each time.
            Its plain prints are silenced by quiet() instead."""

    client = OpenAI(api_key=key, base_url=BASE_URL)
    return LocalVanna({"path": str(STORE_PATH), "client": client, "model": MODEL})


def train(vanna, schema):
    """Store each CREATE TABLE separately, then the notes and the pairs. Separate
    statements let a larger store return only the tables a question needs."""
    counts = {"ddl": 0, "documentation": 0, "pairs": 0}
    for statement in schema.split(";"):
        statement = statement.strip()
        if statement.startswith("CREATE TABLE"):
            quiet(vanna.train, ddl=statement + ";")
            counts["ddl"] += 1
    for note in DOCUMENTATION:
        quiet(vanna.train, documentation=note)
        counts["documentation"] += 1
    for question, sql in TRAINING_PAIRS:
        quiet(vanna.train, question=question, sql=sql)
        counts["pairs"] += 1
    return counts


def show_retrieval(vanna, question):
    """Print what the question pulls out of the store before any SQL exists."""
    ddl = vanna.get_related_ddl(question)
    docs = vanna.get_related_documentation(question)
    pairs = vanna.get_similar_question_sql(question)

    print(f"  related DDL: {len(ddl)} statement(s)")
    for item in ddl[:2]:
        print(f"    {item.splitlines()[0][:70]}")
    print(f"  related notes: {len(docs)}")
    for item in docs[:3]:
        print(f"    {item[:78]}")
    print(f"  similar question/SQL pairs: {len(pairs)}")
    for item in pairs[:2]:
        text = item.get("question") if isinstance(item, dict) else str(item)
        print(f"    {str(text)[:78]}")


def rows(frame):
    """Return a result frame's rows as tuples, rounded, for comparison."""
    return [
        tuple(round(v, 2) if isinstance(v, float) else v for v in row)
        for row in frame.itertuples(index=False)
    ]


def flat(sql):
    """Collapse a query onto one line."""
    return " ".join(sql.split())


def main():
    db_path = _db.ensure_database()
    schema = _db.load_schema()

    # 1. Composing the client
    print("--- 1. Composing the client ---")
    vanna = build_client(fresh=True)
    print(f"  vector store: {STORE_PATH}")
    print(f"  chat model:   {MODEL}")
    print("  Embeddings run locally inside the store, so nothing but the chat")
    print("  call leaves the machine.")

    # 2. Connecting to the database
    print("\n--- 2. Connecting to the database ---")
    vanna.connect_to_sqlite(str(db_path))
    print(f"  connected: {db_path.name}")

    # 3. Training
    print("\n--- 3. Training ---")
    counts = train(vanna, schema)
    print(f"  {counts['ddl']} CREATE TABLE statements")
    print(f"  {counts['documentation']} written notes")
    print(f"  {counts['pairs']} question/SQL pairs")

    # 4. What the question retrieves
    print(f"\n--- 4. What '{QUESTION}' retrieves ---")
    show_retrieval(vanna, QUESTION)

    # 5. Generating and running
    print("\n--- 5. Generating and running ---")
    sql = quiet(vanna.generate_sql, QUESTION)
    print(f"  SQL:\n    {flat(sql)}\n")
    frame = vanna.run_sql(sql)
    print(f"  Result:\n{frame.to_string(index=False)}")
    expected = vanna.run_sql(REFERENCE_SQL)
    count, yearly = rows(expected)[0]
    print(f"\n  Hand-written SQL: {count} policies, {yearly} in premium per year.")
    if rows(frame) == rows(expected):
        print("  The generated result matches it.")
    else:
        print("  The generated result does not match it.")
        if "payment_frequency" not in sql:
            print("  The query never reads payment_frequency, so it adds up premiums")
            print("  per payment period instead of per year.")

    # 6. Correcting a wrong answer
    print("\n--- 6. Correcting a wrong answer ---")
    print(f"  Q: {CORRECTION_QUESTION}")
    before = quiet(vanna.generate_sql, CORRECTION_QUESTION)
    print(f"  before correction:\n    {flat(before)[:150]}")
    # A reviewer settles the ambiguity once and the settled pair goes back in.
    # Only verified SQL belongs here, because a wrong pair teaches the mistake
    # as readily as a right one teaches the fix. Part 7 shows how far it reaches.
    quiet(vanna.train, question=CORRECTION_QUESTION, sql=CORRECTED_SQL)
    print(f"  stored correction:\n    {CORRECTED_SQL}")

    # 7. Asking again
    print("\n--- 7. Asking again ---")
    for question in (CORRECTION_QUESTION, REWORDED_QUESTION):
        after = quiet(vanna.generate_sql, question)
        used = "customer_status" in after
        print(f"  Q: {question}")
        print(f"    {flat(after)[:150]}")
        print(f"    uses the stored correction: {'yes' if used else 'no'}, "
              f"result {rows(vanna.run_sql(after))[0][0]}")


if __name__ == "__main__":
    main()
