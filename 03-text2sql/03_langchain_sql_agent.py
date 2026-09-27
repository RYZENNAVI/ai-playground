"""Answer questions with LangChain's SQL agent, which reads the schema itself.

Script 02 pasted the schema into the prompt. Here LangChain's SQLDatabaseToolkit
reads it from the database by reflection, and create_sql_agent builds a ReAct
agent on DeepSeek that calls the toolkit's four tools until it can answer.
Reflection rebuilds each CREATE TABLE from the table structure. Types and keys
survive, but the column comments that explain the status codes are lost. The
toolkit attaches three sample rows per table instead.

The agent gets three questions: one that column names can answer, one that needs
the meaning of the code 'IF', and one about a table that does not exist.

The run prints seven parts:
    1. Opening the database through the wrapper. The five tables it found.
    2. What the wrapper's schema text keeps, and what it drops. Whether the
       comment 'IF = in force' survives reflection.
    3. Tools handed to the agent.
    4. A question the sample rows can answer. The average premium per
       product type.
    5. A question that needs the meaning of a stored code. How many policies
       are in force. No comment tells the agent what 'IF' means, so it lists
       the codes and guesses from the letters.
    6. A table that does not exist. The ReAct output parser raises an error
       when the model answers in plain prose.
    7. Verdict. The agent's answer to part 5 against the hand-written SQL, and
       the tool calls each question took.
"""

import os
import sqlite3
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

# Answerable from column names plus the sample rows the toolkit attaches.
PLAIN_QUESTION = (
    "What is the average premium for each product type? Round to two decimals."
)
# Answerable only if you know that policy_status stores 'IF' for in force.
CODED_QUESTION = "How many policies are still in force?"
MISSING_TABLE_QUESTION = "Describe the PolicyHolderDetails table."

REFERENCE_SQL = "SELECT COUNT(*) FROM policies WHERE policy_status = 'IF'"

# The cap only stops an agent that cannot settle on an answer. The coded
# question takes six steps (five tool calls and the final answer), so 5 cut it
# off just before it answered.
MAX_ITERATIONS = 8


def build_agent(db_uri):
    """Build the database wrapper, the toolkit and a ReAct SQL agent on DeepSeek.
    In langchain 1.3 create_sql_agent lives in langchain_community."""
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
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        max_iterations=MAX_ITERATIONS,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    return database, toolkit, agent


def compare_schema_sources(db_path, database):
    """Return the stored and the reflected CREATE TABLE for policies. SQLite keeps
    the original text with its comments; the wrapper rebuilds it without them."""
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
    """Run one question. Return (answer, tool calls), or (None, None) if it raised."""
    print(f"  Q: {question}\n")
    try:
        result = agent.invoke({"input": question})
        print(f"\n  {label}: {result['output']}")
        return result["output"], len(result["intermediate_steps"])
    except Exception as error:
        print(f"\n  {label} raised {type(error).__name__}: {str(error)[:150]}")
        return None, None


def main():
    db_path = _db.ensure_database()
    db_uri = f"sqlite:///{db_path}"

    # 1. Opening the database through the wrapper
    print("--- 1. Opening the database through the wrapper ---")
    database, toolkit, agent = build_agent(db_uri)
    print(f"  uri: {db_uri}")
    print(f"  tables discovered: {database.get_usable_table_names()}")
    print("  Nobody pasted a schema in. The wrapper read the structure itself.")

    # 2. What the wrapper's schema text keeps, and what it drops
    print("\n--- 2. What the wrapper's schema text keeps, and what it drops ---")
    _, _, in_stored, in_reflected = compare_schema_sources(db_path, database)
    print(f"  stored CREATE TABLE contains 'IF = in force': {in_stored}")
    print(f"  wrapper's schema text contains it:           {in_reflected}")
    print("  The wrapper reflects the table and rebuilds the DDL, so types and")
    print("  keys survive but the column comments do not. It attaches three")
    print("  sample rows, but neither they nor the DDL say what a code means.")

    # 3. Tools handed to the agent
    print("\n--- 3. Tools handed to the agent ---")
    for tool in toolkit.get_tools():
        print(f"  {tool.name:<22} {tool.description.splitlines()[0][:66]}")

    # 4. A question the sample rows can answer
    print("\n--- 4. A question the sample rows can answer ---")
    _, plain_calls = ask(agent, PLAIN_QUESTION, "Answer")

    # 5. A question that needs the meaning of a stored code
    print("\n--- 5. A question that needs the meaning of a stored code ---")
    print("  The sample rows show only 'IF'. Nothing in the schema text says what")
    print("  'IF', 'LP' or 'TM' mean, so the agent has to guess from the letters.")
    coded, coded_calls = ask(agent, CODED_QUESTION, "Answer")

    # 6. A table that does not exist
    print("\n--- 6. A table that does not exist ---")
    # The parser expects every reply in the Action / Action Input shape. Noticing
    # a missing table pushes the model towards plain prose, and so far that has
    # raised on every run. ask() still handles an answer too, because whether
    # the parser accepts the prose depends on how the model words it.
    missing, missing_calls = ask(agent, MISSING_TABLE_QUESTION, "Answer")

    # 7. Verdict
    print("\n--- 7. Verdict ---")
    connection = sqlite3.connect(db_path)
    try:
        truth = connection.execute(REFERENCE_SQL).fetchone()[0]
    finally:
        connection.close()
    print(f"  Hand-written SQL says {truth} policies are in force.")
    if coded is None or coded.startswith("Agent stopped"):
        print(f"  The agent gave no answer within the cap of {MAX_ITERATIONS} steps.")
    elif str(truth) in coded:
        print(f"  The agent matched {truth}, but only by guessing that 'IF' means in")
        print("  force. The comment that says so never reached it.")
    else:
        print(f"  The agent's answer does not contain {truth}: {coded}")

    part6 = "raised before finishing" if missing is None else f"took {missing_calls}"
    print(f"\n  Tool calls: part 4 took {plain_calls}, part 5 took {coded_calls},"
          f" part 6 {part6}.")
    print("  The agent makes a model call to choose each tool and one more for")
    print("  the final answer, and sql_db_query_checker calls the model as well.")
    print("  Script 02 used one model call per question.")


if __name__ == "__main__":
    main()
