"""Let a toolkit introspect the database instead of pasting a schema into a prompt.

Demonstrates what a database-aware toolkit gains, and what it quietly loses:
    1. Open the database through a wrapper that can read its own structure.
    2. Compare the wrapper's schema text against what the database really stores.
    3. List the tools the agent was handed.
    4. Answer a question that needs only column names and sample rows.
    5. Answer one that needs a stored code, and watch the agent hunt for meaning.
    6. Ask about a table that does not exist, to see the framework's failure mode.
    7. Weigh the round trips against the single call used in script 02.

Module 03: Text2SQL - Database Toolkit Agent.
"""

import os
import sqlite3
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

# Answerable from column names plus the sample rows the toolkit attaches.
PLAIN_QUESTION = (
    "What is the average premium for each product type? Round to two decimals."
)
# Answerable only if you know that policy_status stores 'IF' for in force.
CODED_QUESTION = "How many policies are still in force?"
MISSING_TABLE_QUESTION = "Describe the PolicyHolderDetails table."

REFERENCE_SQL = "SELECT COUNT(*) FROM policies WHERE policy_status = 'IF'"

# Without a cap the agent will keep re-running the same aggregate while it tries
# to work out what a status code means. Five steps is enough to answer a
# well-posed question and short enough to make a stuck one obvious.
MAX_ITERATIONS = 5


def build_agent(db_uri):
    """Wire up the toolkit agent.

    The imports sit inside the function for two reasons: the LangChain stack is
    heavy enough that the rest of the module should not pay to import it, and
    these paths move between releases. In langchain 1.3 the agent constructor
    lives in langchain_community, not in langchain.agents where older code
    looks for it.
    """
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase
    from langchain_openai import ChatOpenAI

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set. Add it to your .env file.")

    database = SQLDatabase.from_uri(db_uri)
    # Near-zero temperature because the task has one right answer. Sampling
    # variety helps prose and hurts SQL.
    llm = ChatOpenAI(model=MODEL, temperature=0.01, api_key=key, base_url=BASE_URL)
    toolkit = SQLDatabaseToolkit(db=database, llm=llm)
    agent = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, max_iterations=MAX_ITERATIONS
    )
    return database, toolkit, agent


def compare_schema_sources(db_path, database):
    """Show that reflection rebuilds the DDL and drops the column comments.

    SQLite keeps the exact CREATE TABLE text it was given, comments included, in
    sqlite_master. The wrapper does not read that text - it reflects the table
    through SQLAlchemy and prints a reconstruction, which carries types and keys
    but no comments. That gap is the whole point of this step: the toolkit
    recovers the structure automatically and loses the meaning automatically.
    """
    connection = sqlite3.connect(db_path)
    try:
        stored = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'policies'"
        ).fetchone()[0]
    finally:
        connection.close()

    reflected = database.get_table_info(["policies"])
    marker = "IF = in force"
    return stored, reflected, marker in stored, marker in reflected


def ask(agent, question, label):
    """Run one question, returning the answer or the exception it raised."""
    print(f"  Q: {question}\n")
    try:
        result = agent.invoke({"input": question})
        print(f"\n  {label}: {result['output']}")
        return result["output"]
    except Exception as error:
        print(f"\n  {label} raised {type(error).__name__}: {str(error)[:150]}")
        return None


def main():
    db_path = _db.ensure_database()
    db_uri = f"sqlite:///{db_path}"

    print("--- 1. Opening the database through the wrapper ---")
    database, toolkit, agent = build_agent(db_uri)
    print(f"  uri: {db_uri}")
    print(f"  tables discovered: {database.get_usable_table_names()}")
    print("  Nobody pasted a schema in - the wrapper read the structure itself.")

    print("\n--- 2. What the wrapper's schema text keeps, and what it drops ---")
    stored, reflected, in_stored, in_reflected = compare_schema_sources(db_path, database)
    print(f"  stored CREATE TABLE contains 'IF = in force': {in_stored}")
    print(f"  wrapper's schema text contains it:           {in_reflected}")
    print("  The wrapper reflects the table and rebuilds the DDL, so types and")
    print("  keys survive but the column comments do not. It does attach three")
    print("  sample rows, which is the only clue left about stored values.")

    print("\n--- 3. Tools handed to the agent ---")
    for tool in toolkit.get_tools():
        print(f"  {tool.name:<22} {tool.description.splitlines()[0][:66]}")

    print("\n--- 4. A question the sample rows can answer ---")
    ask(agent, PLAIN_QUESTION, "Answer")

    print("\n--- 5. A question that needs the meaning of a stored code ---")
    print("  Watch for repeated queries: the agent can see 'IF', 'LP' and 'TM'")
    print("  in the data but has nothing telling it which one means in force.")
    coded = ask(agent, CODED_QUESTION, "Answer")

    print("\n--- 6. A table that does not exist ---")
    # The parser expects every reply to stay in the Action / Action Input shape.
    # Noticing a missing table pushes the agent towards plain prose, which the
    # parser may or may not accept depending on how the sentence lands - the
    # same question raises on one run and returns on the next. That instability
    # is the point worth seeing, so both outcomes are handled rather than one
    # being presented as the behaviour.
    ask(agent, MISSING_TABLE_QUESTION, "Answer")

    print("\n--- 7. Verdict ---")
    connection = sqlite3.connect(db_path)
    try:
        truth = connection.execute(REFERENCE_SQL).fetchone()[0]
    finally:
        connection.close()
    print(f"  Hand-written SQL says {truth} policies are in force.")
    print(f"  Agent answered: {coded if coded else 'no answer within the step cap'}")
    print()
    print("  The toolkit removes the work of describing a schema and adds sample")
    print("  rows for free, but it reaches the model through reflection, which is")
    print("  exactly where the comments are lost. Script 02 pastes the raw CREATE")
    print("  TABLE text instead: one call, no introspection, comments intact.")


if __name__ == "__main__":
    main()
