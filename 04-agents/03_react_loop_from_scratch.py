"""This script answers questions about a small compliance rule book with a
ReAct agent loop written by hand, with no agent framework. ReAct (reason and
act) has the model write a Thought, then an Action and its input. The program
runs that tool, adds the result as an Observation, and calls the model again,
until the model writes a Final Answer. A framework hides four jobs that this
script does itself. It renders the tool names and descriptions into the prompt.
It stops the model at "Observation:", so the model cannot invent the tool's
result. It parses each reply into an action or a final answer. It sends the
real result back.

The tools search a four-rule compliance rule book, list one category, or read
one rule by id.

The run prints five parts:
    1. Tools rendered into the prompt. The descriptions and names as the
       model sees them.
    2. The loop with tools. A question about the minimum a securities fund
       must raise, answered from rule R-002.
    3. The same tools, but the prompt never lists them. The model guesses
       tool names, every guess misses, and it gives up.
    4. A question the rule book does not cover. A tax question, which the
       model searches for before saying the rule book has no answer.
    5. A question that names a rule id. The only part that calls read_rule.
"""

import os
import re
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

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

MAX_STEPS = 6

# A small compliance rule book. Every entry carries a category, so the model can
# search text or list a whole category.
RULES = [
    {
        "id": "R-001",
        "category": "eligibility",
        "question": "Who qualifies as an eligible investor?",
        "answer": (
            "An eligible investor must show both the capacity to assess risk and the capacity to absorb "
            "loss, commit at least 1,000,000 to a single fund, and meet one of: net assets of at least "
            "10,000,000 for an entity, or financial assets of at least 3,000,000 for an individual."
        ),
    },
    {
        "id": "R-002",
        "category": "eligibility",
        "question": "What is the minimum size a fund must raise?",
        "answer": (
            "A securities fund may not close below 10,000,000 in committed capital. Venture and growth "
            "funds are governed by the fund agreement instead of a fixed floor."
        ),
    },
    {
        "id": "R-003",
        "category": "supervision",
        "question": "What risk reserve must a manager hold?",
        "answer": (
            "A securities fund manager sets aside 10 percent of management fee income as a risk reserve, "
            "used only to compensate investors for losses caused by the manager's own breach or error."
        ),
    },
    {
        "id": "R-004",
        "category": "supervision",
        "question": "How often must a manager report to investors?",
        "answer": (
            "A quarterly report is due within 15 business days of quarter end, and an audited annual "
            "report within four months of year end."
        ),
    },
]

CATEGORIES = sorted({rule["category"] for rule in RULES})


# The tools. Each one is an ordinary function returning a string.


def search_rules(keywords: str) -> str:
    """Return every rule whose question or answer contains one of the keywords."""
    terms = [term.lower() for term in keywords.replace(",", " ").split() if len(term) > 2]
    hits = [
        rule
        for rule in RULES
        if any(term in (rule["question"] + rule["answer"]).lower() for term in terms)
    ]
    if not hits:
        return f"No rule matches {keywords!r}."
    return "\n".join(f"[{rule['id']}] {rule['question']} {rule['answer']}" for rule in hits)


def list_category(category: str) -> str:
    """Return the questions filed under one category, or the valid category names."""
    wanted = category.strip().lower()
    hits = [rule for rule in RULES if rule["category"] == wanted]
    if not hits:
        return f"Unknown category {category!r}. Valid categories: {', '.join(CATEGORIES)}."
    return "\n".join(f"[{rule['id']}] {rule['question']}" for rule in hits)


def read_rule(rule_id: str) -> str:
    """Return the full text of one rule by its identifier."""
    wanted = rule_id.strip().upper()
    for rule in RULES:
        if rule["id"] == wanted:
            return rule["answer"]
    return f"No rule with id {rule_id!r}."


TOOLS = {
    "search_rules": (search_rules, "Search the rule book by keywords. Input: two or three keywords."),
    "list_category": (list_category, f"List the rules in one category. Input: one of {', '.join(CATEGORIES)}."),
    "read_rule": (read_rule, "Read the full text of one rule. Input: a rule id such as R-001."),
}


PROMPT_TEMPLATE = """You answer questions about a fund compliance rule book.

You can use these tools:
{tools}

Use exactly this format:

Question: the question you must answer
Thought: what you need to do next
Action: one of [{tool_names}]
Action Input: the input for that tool
Observation: the result the tool returned
... (Thought/Action/Action Input/Observation may repeat)
Thought: I now know the final answer
Final Answer: the answer for the user

If the rule book does not cover the question, say so plainly in the Final Answer
instead of inventing a rule.

Question: {question}"""

# The same template with the tool listing removed entirely. Part 3 swaps this in
# to show what the loop degrades into when the tool names never reach the model.
BLIND_TEMPLATE = PROMPT_TEMPLATE.replace(
    "You can use these tools:\n{tools}\n\n", ""
).replace("Action: one of [{tool_names}]", "Action: the tool to use")


def render_tools() -> tuple[str, str]:
    """Turn the tool registry into the descriptions and names the prompt needs.
    A model only knows a tool exists if its name is in the text; part 3 leaves them out."""
    descriptions = "\n".join(f"- {name}: {description}" for name, (_, description) in TOOLS.items())
    names = ", ".join(TOOLS)
    return descriptions, names


def parse_reply(reply: str) -> tuple[str, str]:
    """Return ("final", text), ("action", "name||input") or ("error", reply)."""
    if "Final Answer:" in reply:
        return "final", reply.split("Final Answer:")[-1].strip()

    match = re.search(r"Action\s*:\s*(.*?)\n+Action\s*Input\s*:\s*(.*)", reply, re.DOTALL)
    if match:
        tool_name = match.group(1).strip()
        # DOTALL lets group 2 run across lines. Only the first line is the input.
        tool_input = match.group(2).strip().strip('"').splitlines()[0].strip()
        return "action", f"{tool_name}||{tool_input}"

    return "error", reply.strip()


def call_model(client: OpenAI, messages: list[dict]) -> str:
    """Send the transcript and stop the model at the first "Observation:". Without the
    stop, the model writes the Observation itself and invents the result."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        stop=["Observation:"],
    )
    return response.choices[0].message.content


def run_loop(client: OpenAI, question: str, template: str = PROMPT_TEMPLATE, show_transcript: bool = False) -> str:
    """Run Thought, Action and Observation until a Final Answer or MAX_STEPS. Each reply
    and each observation go in as their own assistant and user messages."""
    descriptions, names = render_tools()
    messages = [
        {
            "role": "user",
            "content": template.format(tools=descriptions, tool_names=names, question=question),
        }
    ]
    tool_calls = 0

    for step in range(1, MAX_STEPS + 1):
        reply = call_model(client, messages)
        kind, payload = parse_reply(reply)

        if kind in ("final", "error"):
            if show_transcript:
                print(f"    transcript: {len(messages)} messages")
            print(f"    passes: {step}, tool calls: {tool_calls}")
            return payload if kind == "final" else f"[unparsable reply] {payload}"

        tool_name, tool_input = payload.split("||", 1)
        function = TOOLS.get(tool_name, (None, None))[0]
        observation = function(tool_input) if function else f"No tool named {tool_name!r}."
        tool_calls += 1
        # With several rules in one result, the first line alone would hide the others.
        rule_ids = re.findall(r"^\[(R-\d+)\]", observation, re.MULTILINE)
        shown = f"{len(rule_ids)} rules: {', '.join(rule_ids)}" if len(rule_ids) > 1 else observation.splitlines()[0][:70]
        print(f"    step {step}: {tool_name}({tool_input!r}) -> {shown}")

        messages.append({"role": "assistant", "content": reply.strip()})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "[stopped] the loop hit its step limit without a final answer."


def demo_full_loop(client: OpenAI) -> None:
    """Step 2. One question that cannot be answered without a tool call."""
    print("\n--- 2. The loop with tools rendered into the prompt ---")
    question = "What is the minimum a securities fund must raise before it closes?"
    print(f"  question: {question}")
    answer = run_loop(client, question, show_transcript=True)
    print(f"  answer:   {answer}")


def demo_missing_tool_names(client: OpenAI) -> None:
    """Step 3. Keep the tools registered but leave them out of the prompt. The model guesses names
    such as search_rule_book, every guess misses, and it then answers that it has no working tool."""
    print("\n--- 3. Same tools, but the prompt never lists them ---")
    question = "What is the minimum a securities fund must raise before it closes?"
    answer = run_loop(client, question, template=BLIND_TEMPLATE)
    print(f"  answer:   {answer}")


def demo_outside_knowledge(client: OpenAI) -> None:
    """Step 4. Ask about something the rule book does not contain. The model searches, lists
    both categories, and then says in a Final Answer that the rule book has no answer."""
    print("\n--- 4. A question the rule book does not cover ---")
    question = "What tax rate applies to carried interest for this fund?"
    print(f"  question: {question}")
    answer = run_loop(client, question)
    print(f"  answer:   {answer}")


def demo_read_rule_path(client: OpenAI) -> None:
    """Step 5. Name a rule id in the question, the only part that calls read_rule."""
    print("\n--- 5. A question phrased to trigger read_rule directly ---")
    question = "Please read rule R-004 in full and tell me the exact reporting deadlines."
    print(f"  question: {question}")
    answer = run_loop(client, question)
    print(f"  answer:   {answer}")


def main() -> None:
    descriptions, names = render_tools()
    print("--- 1. Tools rendered into the prompt ---")
    print(textwrap.indent(descriptions, "  "))
    print(f"  tool_names rendered as: {names}")

    if not API_KEY:
        print("\nNo DEEPSEEK_API_KEY or OPENAI_API_KEY in .env. The loop needs one, so the run stops here.")
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    demo_full_loop(client)
    demo_missing_tool_names(client)
    demo_outside_knowledge(client)
    demo_read_rule_path(client)


if __name__ == "__main__":
    main()
