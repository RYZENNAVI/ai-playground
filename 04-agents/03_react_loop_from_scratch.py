"""Build the reason-and-act loop by hand, with no agent framework involved.

Demonstrates the four things a framework does for you when it runs an agent:
    1. Render the tool names and descriptions into the prompt as plain text.
    2. Stop generation at "Observation:" so the model cannot invent tool output.
    3. Parse the reply into either an action to run or a finished answer.
    4. Append the real tool result and call the model again with the longer text.
    5. Re-run with the tool list left out, and watch the model invent names.
    6. Ask something the knowledge base does not cover, to exercise the fallback.
    7. Name a rule id directly, to exercise the one tool the other demos never call.

Module 04: Agents - Hand-Written ReAct Loop.
"""

import os
import re
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

MAX_STEPS = 6

# A small compliance knowledge base. Every entry carries a category so that one
# tool can search text while another can list a whole category, which is what
# forces the model to choose between tools rather than always taking the first.
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


# 1. The tools. Each one is an ordinary function returning a string.


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

# The same template with the tool listing removed entirely. Step 5 swaps this in
# to show what the loop degrades into when the tool names never reach the model.
BLIND_TEMPLATE = PROMPT_TEMPLATE.replace(
    "You can use these tools:\n{tools}\n\n", ""
).replace("Action: one of [{tool_names}]", "Action: the tool to use")


def render_tools() -> tuple[str, str]:
    """Turn the tool registry into the two strings the prompt needs.

    This is the step that is easiest to skip and hardest to notice skipping. The
    functions are already registered in TOOLS, so the code looks complete, but a
    model only knows a tool exists if its name and description appear in the
    text it is given. Step 5 runs the loop without this and nothing gets called.
    """
    descriptions = "\n".join(f"- {name}: {description}" for name, (_, description) in TOOLS.items())
    names = ", ".join(TOOLS)
    return descriptions, names


def parse_reply(reply: str) -> tuple[str, str]:
    """Turn one model reply into either ("final", text) or ("action", name, input).

    Three fallbacks sit in front of the regex, and each one exists because a
    model really does reply this way. A reply that reaches the regex and still
    does not match is the only genuine parse failure.
    """
    if "Final Answer:" in reply:
        return "final", reply.split("Final Answer:")[-1].strip()

    # The model decided mid-format that it cannot help, and dropped the format.
    boundary_phrases = ("the rule book does not", "is not covered", "no information about", "i do not have")
    lowered = reply.lower()
    if any(phrase in lowered for phrase in boundary_phrases) and "action:" not in lowered:
        return "final", reply.strip()

    match = re.search(r"Action\s*:\s*(.*?)\n+Action\s*Input\s*:\s*(.*)", reply, re.DOTALL)
    if match:
        tool_name = match.group(1).strip()
        # DOTALL makes group 2 greedy across newlines, so if the model keeps
        # writing after the input line and before the stop sequence cuts it
        # off, that trailing text would otherwise ride along as part of the
        # tool input. Only the first line is ever the actual input.
        tool_input = match.group(2).strip().strip('"').splitlines()[0].strip()
        return "action", f"{tool_name}||{tool_input}"

    # A long reply with no Action and no Final Answer is prose, not a malformed
    # step; treating it as an answer beats raising on the user's behalf.
    if len(reply.strip()) > 80:
        return "final", reply.strip()

    return "error", reply.strip()


def call_model(client: OpenAI, messages: list[dict]) -> str:
    """Send the transcript and stop the model at the first "Observation:".

    Without the stop sequence the model happily continues past its own Action
    and writes an Observation too, hallucinating the tool result and never
    letting real code run. The stop sequence is what turns one long piece of
    generated text into a loop the program controls.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        stop=["Observation:"],
    )
    return response.choices[0].message.content


def run_loop(client: OpenAI, question: str, template: str = PROMPT_TEMPLATE, show_transcript: bool = False) -> str:
    """Run reason, act and observe until the model produces a final answer.

    The transcript is carried as alternating turns - the model's Thought/Action
    as an assistant message, the tool result as the user message that follows -
    rather than as one growing block of text. The distinction is not cosmetic:
    handed a half-finished block, the model treats it as a document to continue
    and restarts it, re-typing the original question and its own first Thought.
    That stale Thought then sits after the newest Observation, so the model
    reads it as the latest line and reissues the action it already ran. Splitting
    the same content into turns removes the ambiguity about which line is newest.
    """
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
        print(f"    step {step}: {tool_name}({tool_input!r}) -> {observation.splitlines()[0][:70]}")

        messages.append({"role": "assistant", "content": reply.strip()})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "[stopped] the loop hit its step limit without a final answer."


def demo_full_loop(client: OpenAI) -> None:
    """Steps 1 to 4. One question that cannot be answered without a tool call."""
    print("--- 1-4. The loop with tools rendered into the prompt ---")
    question = "What is the minimum a securities fund must raise before it closes?"
    print(f"  question: {question}")
    answer = run_loop(client, question, show_transcript=True)
    print(f"  answer:   {answer}")


def demo_missing_tool_names(client: OpenAI) -> None:
    """Step 5. Keep the tools registered but leave them out of the prompt.

    The registry is unchanged and every function still works when called
    directly. Only the text changed. What the model does with that gap is the
    point: rather than stopping, it guesses names that sound right for the job -
    SearchRuleBook, LookupRuleBook, GetRuleBook - and every guess misses, until
    the step budget runs out. Registering a tool and offering a tool are two
    different acts, and a registry that is never rendered fails silently.
    """
    print("\n--- 5. Same tools, but the prompt never lists them ---")
    question = "What is the minimum a securities fund must raise before it closes?"
    answer = run_loop(client, question, template=BLIND_TEMPLATE)
    print(f"  answer:   {answer}")


def demo_outside_knowledge(client: OpenAI) -> None:
    """Step 6. Ask about something the rule book does not contain.

    The interesting part is not the answer but the shape of the reply: the model
    abandons the Action format once it decides no tool can help, which is the
    exact case the boundary fallback in parse_reply was written for.
    """
    print("\n--- 6. A question the rule book does not cover ---")
    question = "What tax rate applies to carried interest for this fund?"
    print(f"  question: {question}")
    answer = run_loop(client, question)
    print(f"  answer:   {answer}")


def demo_read_rule_path(client: OpenAI) -> None:
    """Step 7. Name a rule id directly, to exercise read_rule.

    The other three demos all resolve through search_rules or list_category,
    so read_rule's id lookup and its "no such id" message have never actually
    run. Naming an id in the question is what tips tool selection toward it.
    """
    print("\n--- 7. A question phrased to trigger read_rule directly ---")
    question = "Please read rule R-004 in full and tell me the exact reporting deadlines."
    print(f"  question: {question}")
    answer = run_loop(client, question)
    print(f"  answer:   {answer}")


def main() -> None:
    descriptions, names = render_tools()
    print("--- tools available to the loop ---")
    print(textwrap.indent(descriptions, "  "))
    print(f"  tool_names rendered as: {names}\n")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is not set; the loop needs it. Stopping here.")
        return

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)
    demo_full_loop(client)
    demo_missing_tool_names(client)
    demo_outside_knowledge(client)
    demo_read_rule_path(client)


if __name__ == "__main__":
    main()
