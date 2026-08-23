"""Hand the same loop to a framework and see what it takes over.

Demonstrates the agent a diagnosis workflow ends up needing:
    1. Declare four probes as typed tools the model can read the signature of.
    2. Let the framework run reason, act and observe over a real incident.
    3. Read the step trace to see which probe ran, in what order, and why.
    4. Cap the step budget and watch a deliberately unsolvable case hit the cap.
    5. Blur the tool descriptions and re-run, to show selection is text-driven.

Module 04: Agents - Tool-Using Diagnosis Agent.
"""

import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

# Fixed seed so the simulated latencies are the same on every run. A diagnosis
# demo whose numbers move between runs cannot be discussed after the fact.
random.seed(7)

# The simulated network. Each host is either reachable or not, and the resolver
# knows a different set of names than the ping table does - that mismatch is
# what makes multi-step diagnosis necessary instead of one lookup.
HOSTS = {
    "shop.internal": {"address": "10.0.4.21", "reachable": True},
    "billing.internal": {"address": "10.0.4.37", "reachable": False},
    "cache.internal": {"address": "10.0.4.55", "reachable": True},
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
    return f"{hostname} ({record['address']}) answers in {random.randint(2, 40)} ms."


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

SYSTEM_PROMPT = (
    "You diagnose network incidents. Investigate with the tools before concluding. "
    "Name the failing component and the evidence you based that on. Be brief."
)


def build_model() -> ChatOpenAI:
    """Return a chat model reached through the OpenAI request format."""
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
    )


def describe_tools() -> None:
    """Step 1. Show the schema the framework derived from each function.

    The decorator reads the signature and the docstring and turns them into a
    tool schema. That schema, not the Python function, is what the model sees,
    so an argument with no type hint or a docstring that omits what the input
    should look like degrades tool selection while the code still runs fine.
    """
    print("--- 1. Tool schemas handed to the model ---")
    for item in TOOLS:
        argument_names = ", ".join(item.args_schema.model_json_schema()["properties"])
        print(f"  {item.name}({argument_names}): {item.description}")


def run_agent(question: str, tools: list, recursion_limit: int = 12) -> dict:
    """Run the agent and return both its answer and the trace of what it called.

    create_agent compiles to a graph, so the return value is the whole message
    list rather than one string. Reading that list back is the only way to tell
    a correct answer that was investigated from a correct answer that was
    guessed, which is why every demo below prints the trace instead of the text.
    """
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
            # Streaming keeps the messages produced before the cap was reached,
            # so a stopped run can still be inspected instead of vanishing.
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
    """Steps 2 and 3. A real incident that needs more than one probe.

    The question names a symptom, not a host, so the agent has to find the host
    in the log before it can test it. That ordering is not scripted anywhere -
    it falls out of the tool descriptions and the observations coming back.
    """
    print("\n--- 2-3. Diagnosing a live incident ---")
    question = "Checkout keeps failing with connection errors since 14:00. What is broken?"
    print(f"  question: {question}")
    print_trace(run_agent(question, TOOLS))


def demo_step_limit() -> None:
    """Step 4. Ask something no tool can settle, with a tight step budget.

    The simulated network has no traffic-volume data at all, so no sequence of
    calls can answer this. A capped budget turns that into a bounded failure
    instead of a loop that bills forever, which is the practical reason the cap
    exists rather than a theoretical one.
    """
    print("\n--- 4. An unanswerable question against a tight budget ---")
    question = "Which host consumed the most bandwidth this week, and by how many gigabytes?"
    print(f"  question: {question}")
    print_trace(run_agent(question, TOOLS, recursion_limit=4))


def demo_description_matters() -> None:
    """Step 5. Change only the wording of a description and re-run.

    Nothing about the four functions changes here. The vague copies keep the
    same names and the same behaviour, and only their descriptions lose the
    detail about what each probe is for. Both runs reach the same conclusion,
    but the vague one spends roughly twice the tool calls getting there, most of
    them wasted guessing hostnames such as "checkout", "api" and "db" that the
    clear run never tried. The description is the variable, and it is billed
    per wrong guess.
    """
    print("\n--- 5. The same tools with vague descriptions ---")

    @tool
    def resolve_host(hostname: str) -> str:
        """Do a lookup."""
        return HOSTS.get(hostname, {}).get("address") or f"{hostname} does not resolve."

    @tool
    def ping_host(hostname: str) -> str:
        """Do a check."""
        record = HOSTS.get(hostname)
        if record is None:
            return f"Cannot ping {hostname}."
        return "answers" if record["reachable"] else "request timed out"

    @tool
    def check_interface(name: str) -> str:
        """Do a check."""
        record = INTERFACES.get(name)
        return f"{name} is {record['state']}" if record else f"no interface {name}"

    @tool
    def search_logs(keyword: str) -> str:
        """Do a search."""
        hits = [line for line in LOG_LINES if keyword.lower() in line.lower()]
        return "\n".join(hits) if hits else "nothing found"

    question = "Checkout keeps failing with connection errors since 14:00. What is broken?"
    print_trace(run_agent(question, [resolve_host, ping_host, check_interface, search_logs]))


def main() -> None:
    describe_tools()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\nDEEPSEEK_API_KEY is not set; steps 2 to 5 need it. Stopping here.")
        return

    demo_incident()
    demo_step_limit()
    demo_description_matters()


if __name__ == "__main__":
    main()
