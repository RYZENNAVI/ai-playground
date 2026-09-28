"""Run a tool-calling agent with LangChain's create_agent on a simulated incident.

create_agent runs the ReAct loop from script 03 over the model's native
function calling: each tool's name, docstring and argument types become a
schema, the model returns tool calls instead of text to parse, and the
framework runs them and sends the results back. The four tools probe a small
simulated network: resolve a hostname, ping a host, check a local interface
and search the service log.

The run prints four parts:
    1. Tool schemas handed to the model. The name, arguments and description
       the framework read from each function.
    2. Diagnosing an incident. The question names a symptom, not a host, so
       the agent finds billing.internal in the log before it resolves and
       pings it. The trace lists every call and the first line it returned.
    3. The same incident with a step limit of 4. The agent is stopped after
       its tool calls, before it writes an answer.
    4. The same tools with vague descriptions. Only the docstrings change, to
       "Do a lookup." and similar, and the agent needs more calls to reach the
       same answer.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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

# The simulated network. billing.internal resolves but does not answer, which
# the agent can only find by reading the log first.
HOSTS = {
    "shop.internal": {"address": "10.0.4.21", "reachable": True, "latency_ms": 12},
    "billing.internal": {"address": "10.0.4.37", "reachable": False, "latency_ms": None},
    "cache.internal": {"address": "10.0.4.55", "reachable": True, "latency_ms": 8},
}

INTERFACES = {
    "eth0": {"state": "up", "address": "10.0.4.9", "gateway": "10.0.4.1"},
    "eth1": {"state": "down", "address": None, "gateway": None},
}

LOG_LINES = [
    "14:02:11 ERROR pool: connection to billing.internal:5432 refused",
    "14:02:12 WARN  pool: retrying billing.internal:5432 (attempt 2)",
    "14:03:40 ERROR pool: connection to billing.internal:5432 refused",
    "14:05:02 INFO  cache: 1840 keys evicted",
    "14:06:19 WARN  resolver: slow response from 10.0.4.1 (812 ms)",
]

INCIDENT_QUESTION = "Checkout keeps failing with connection errors since 14:00. What is broken?"


@tool
def resolve_host(hostname: str) -> str:
    """Resolve a hostname to an address. Use this before assuming a host exists."""
    record = HOSTS.get(hostname)
    if record is None:
        return f"{hostname} does not resolve: no such name."
    return f"{hostname} resolves to {record['address']}."


@tool
def ping_host(hostname: str) -> str:
    """Check whether a host answers on the network, and report the round trip time."""
    record = HOSTS.get(hostname)
    if record is None:
        return f"Cannot ping {hostname}: the name does not resolve."
    if not record["reachable"]:
        return f"{hostname} ({record['address']}) does not answer: request timed out."
    return f"{hostname} ({record['address']}) answers in {record['latency_ms']} ms."


@tool
def check_interface(name: str) -> str:
    """Report the state of one local network interface, such as eth0 or eth1."""
    record = INTERFACES.get(name)
    if record is None:
        return f"No interface named {name}. Known interfaces: {', '.join(INTERFACES)}."
    if record["state"] == "down":
        return f"{name} is administratively down and has no address."
    return f"{name} is up, address {record['address']}, gateway {record['gateway']}."


@tool
def search_logs(keyword: str) -> str:
    """Search the recent service log for a keyword and return the matching lines."""
    hits = [line for line in LOG_LINES if keyword.lower() in line.lower()]
    if not hits:
        return f"No log line contains {keyword!r}."
    return "\n".join(hits)


TOOLS = [resolve_host, ping_host, check_interface, search_logs]
CLEAR_TOOLS = {item.name: item for item in TOOLS}

SYSTEM_PROMPT = (
    "You diagnose network incidents. Investigate with the tools before concluding. "
    "Name the failing component and the evidence you based that on. Be brief."
)


def build_model() -> ChatOpenAI:
    """Return the chosen chat model through LangChain's OpenAI client."""
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0,
    )


def describe_tools() -> None:
    """Step 1. Print the schema the @tool decorator built from each function's signature and docstring."""
    print("--- 1. Tool schemas handed to the model ---")
    for item in TOOLS:
        argument_names = ", ".join(item.args_schema.model_json_schema()["properties"])
        print(f"  {item.name}({argument_names}): {item.description}")


def run_agent(question: str, tools: list, recursion_limit: int = 12) -> dict:
    """Run the agent and return its messages, and whether it hit the recursion limit.
    Streaming keeps the messages made before the limit."""
    agent = create_agent(build_model(), tools, system_prompt=SYSTEM_PROMPT)
    messages: list = []
    try:
        for state in agent.stream(
            {"messages": [HumanMessage(question)]},
            config={"recursion_limit": recursion_limit},
            stream_mode="values",
        ):
            messages = state["messages"]
        return {"messages": messages, "stopped": False}
    except Exception as error:
        if "recursion" in str(error).lower():
            return {"messages": messages, "stopped": True}
        raise


def print_trace(result: dict) -> None:
    """Print every tool the agent called and the first line each one returned."""
    calls = 0
    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                calls += 1
                print(f"    call {calls}: {call['name']}({list(call['args'].values())})")
        elif isinstance(message, ToolMessage):
            print(f"      -> {message.content.splitlines()[0]}")
    print(f"    tool calls: {calls}")
    if result["stopped"]:
        print("  answer: none, the agent hit its step limit first")
        return
    print(f"  answer: {result['messages'][-1].text}")


def demo_incident() -> None:
    """Step 2. The question names a symptom, not a host, so the agent has to find the host in the log first."""
    print("\n--- 2. Diagnosing an incident ---")
    print(f"  question: {INCIDENT_QUESTION}")
    print_trace(run_agent(INCIDENT_QUESTION, TOOLS))


def demo_step_limit() -> None:
    """Step 3. Run the same incident with a recursion limit of 4. The limit stops
    the agent after its tool calls, before it writes an answer."""
    print("\n--- 3. The same incident with a step limit of 4 ---")
    print_trace(run_agent(INCIDENT_QUESTION, TOOLS, recursion_limit=4))


def demo_description_matters() -> None:
    """Step 4. Give the same functions vague docstrings and re-run. Each vague tool
    calls the clear one, so only the description the model reads changes."""
    print("\n--- 4. The same tools with vague descriptions ---")

    @tool
    def resolve_host(hostname: str) -> str:
        """Do a lookup."""
        return CLEAR_TOOLS["resolve_host"].func(hostname)

    @tool
    def ping_host(hostname: str) -> str:
        """Do a check."""
        return CLEAR_TOOLS["ping_host"].func(hostname)

    @tool
    def check_interface(name: str) -> str:
        """Do a check."""
        return CLEAR_TOOLS["check_interface"].func(name)

    @tool
    def search_logs(keyword: str) -> str:
        """Do a search."""
        return CLEAR_TOOLS["search_logs"].func(keyword)

    print_trace(run_agent(INCIDENT_QUESTION, [resolve_host, ping_host, check_interface, search_logs]))


def main() -> None:
    describe_tools()

    if not API_KEY:
        print("\nNo DEEPSEEK_API_KEY or OPENAI_API_KEY in .env. Parts 2 to 4 need one, so the run stops here.")
        return

    demo_incident()
    demo_step_limit()
    demo_description_matters()


if __name__ == "__main__":
    main()
