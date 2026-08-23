"""Keep the schema, the notes and the past answers in a vector store instead of a prompt.

Demonstrates the retrieval-trained approach to Text2SQL:
    1. Compose a client from a vector store class and a chat class.
    2. Point it at the local database.
    3. Train it on three kinds of material: DDL, written notes, question/SQL pairs.
    4. Inspect what a question retrieves before any SQL is written.
    5. Generate SQL and run it.
    6. Feed a corrected answer back in, the way a wrong-answer notebook works.
    7. Re-ask the question the correction was meant to fix.

Module 03: Text2SQL - Retrieval-Trained Client.
"""

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

_db = import_module("01_build_insurance_db")

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

STORE_PATH = Path(__file__).parent / "data" / "vanna_store"

# Written notes are the piece the other two approaches have no room for. Script
# 02 could only carry what fits in one prompt; script 03 lost the comments to
# reflection. Here the meanings live in the store and are retrieved only when a
# question needs them.
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
CORRECTION_QUESTION = "How many customers do we have?"
# The answer a reviewer settled on for the question above. Counting customers
# looks unambiguous until someone asks whether closed accounts still count. The
# house rule here is that they do not, and no amount of schema reading would
# reveal that - it is a decision, not a fact about the data.
CORRECTED_SQL = (
    "SELECT COUNT(*) FROM customers WHERE customer_status IN ('A', 'L')"
)


def build_client(fresh=False):
    """Compose a Vanna client from a vector store and a chat model.

    Vanna ships the store and the model as separate mixins so either half can be
    swapped. The import path moved in version 2: the classes the 0.x examples
    use now live under vanna.legacy, while the top-level package became an agent
    framework with a different shape entirely.
    """
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
        """Vector store on disk, chat model over the network.

        The two parents read different keys out of the same config dict, and the
        chat half chokes on the store's keys, so each is initialised with only
        what it understands.
        """

        def __init__(self, config):
            ChromaDB_VectorStore.__init__(
                self, config={"path": config["path"]}
            )
            OpenAI_Chat.__init__(
                self, client=config["client"], config={"model": config["model"]}
            )

        def log(self, message, title="Info"):
            """Swallow the library's running commentary.

            The default implementation prints the whole assembled prompt on
            every call, which buries this script's own output. The prompt is
            still worth seeing once, so step 4 prints the retrieved pieces
            deliberately instead.
            """

    client = OpenAI(api_key=key, base_url=BASE_URL)
    return LocalVanna({"path": str(STORE_PATH), "client": client, "model": MODEL})


def train(vanna, schema):
    """Load the three kinds of material into the store.

    Splitting the schema per statement matters: retrieval returns whole records,
    so one giant blob would drag every table into every prompt and undo the point
    of retrieving at all.
    """
    counts = {"ddl": 0, "documentation": 0, "pairs": 0}
    for statement in schema.split(";"):
        statement = statement.strip()
        if statement.startswith("CREATE TABLE"):
            vanna.train(ddl=statement + ";")
            counts["ddl"] += 1
    for note in DOCUMENTATION:
        vanna.train(documentation=note)
        counts["documentation"] += 1
    for question, sql in TRAINING_PAIRS:
        vanna.train(question=question, sql=sql)
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


def main():
    db_path = _db.ensure_database()
    schema = _db.load_schema()

    print("--- 1. Composing the client ---")
    vanna = build_client(fresh=True)
    print(f"  vector store: {STORE_PATH}")
    print(f"  chat model:   {MODEL}")
    print("  Embeddings run locally inside the store, so nothing but the chat")
    print("  call leaves the machine.")

    print("\n--- 2. Connecting to the database ---")
    vanna.connect_to_sqlite(str(db_path))
    print(f"  connected: {db_path.name}")

    print("\n--- 3. Training ---")
    counts = train(vanna, schema)
    print(f"  {counts['ddl']} CREATE TABLE statements")
    print(f"  {counts['documentation']} written notes")
    print(f"  {counts['pairs']} question/SQL pairs")

    print(f"\n--- 4. What '{QUESTION}' retrieves ---")
    show_retrieval(vanna, QUESTION)

    print("\n--- 5. Generating and running ---")
    sql = vanna.generate_sql(QUESTION)
    print(f"  SQL:\n{sql}\n")
    frame = vanna.run_sql(sql)
    print(f"  Result:\n{frame.to_string(index=False)}")

    print("\n--- 6. Correcting a wrong answer ---")
    print(f"  Q: {CORRECTION_QUESTION}")
    before = vanna.generate_sql(CORRECTION_QUESTION)
    print(f"  before correction:\n    {' '.join(before.split())[:150]}")
    # This is the loop that makes the store worth keeping: a reviewer settles
    # the ambiguity once, the settled pair goes back in, and the next asker
    # inherits the decision. Only verified SQL belongs here - a wrong pair
    # teaches the mistake just as efficiently as a right one teaches the fix.
    vanna.train(question=CORRECTION_QUESTION, sql=CORRECTED_SQL)
    print(f"  stored correction:\n    {CORRECTED_SQL}")

    print("\n--- 7. Asking again ---")
    after = vanna.generate_sql(CORRECTION_QUESTION)
    print(f"  after correction:\n    {' '.join(after.split())[:150]}")
    changed = " ".join(before.split()) != " ".join(after.split())
    print(f"\n  answer changed: {changed}")
    result = vanna.run_sql(after)
    print(f"  Result:\n{result.to_string(index=False)}")


if __name__ == "__main__":
    main()
