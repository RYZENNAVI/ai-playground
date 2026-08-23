"""Run the same five nodes as a fixed pipeline and as a graph that can skip it.

Demonstrates that an agent's behaviour is its topology, not its prompts:
    1. Declare one state object that every node reads from and writes back to.
    2. Wire five nodes into a straight pipeline and run a question through it.
    3. Print the state after each node to see which field each one contributed.
    4. Force a shallow question through that same pipeline, to see what running
       every node costs when none of them were needed.
    5. Add a triage node whose verdict decides which path the next edge takes.
    6. Send shallow, deep and ambiguous questions through that graph and compare,
       tracking the token cost of each path.
    7. Print both topologies so the structural difference is visible, not implied.

Module 04: Agents - LangGraph Topologies.
"""

import operator
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

# What triage falls back to when the model's verdict names neither "shallow"
# nor "deep" - leaning toward the expensive path keeps quality the default
# when the classifier itself is unclear, rather than quietly saving money.
FALLBACK_DEPTH = "deep"


class ReviewState(TypedDict):
    """The one object every node reads from and writes back into.

    Nodes never call each other and never pass arguments. Each returns a partial
    dict that is merged into this state, so adding a node means adding a field
    rather than rewiring a call chain. The visited list is annotated with a
    reducer because it is the only field several nodes append to; without that
    annotation the last writer would overwrite the earlier entries.
    """

    question: str
    depth: Optional[Literal["shallow", "deep"]]
    verdict_raw: Optional[str]
    facts: Optional[str]
    framing: Optional[str]
    options: Optional[str]
    choice: Optional[str]
    answer: Optional[str]
    visited: Annotated[list[str], operator.add]
    tokens: Annotated[int, operator.add]


def build_model() -> ChatOpenAI:
    """Return a chat model reached through the OpenAI request format."""
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
    )


def ask(model: ChatOpenAI, instruction: str, **values: str) -> tuple[str, int]:
    """Send one templated instruction and return the reply plus its token cost."""
    chain = ChatPromptTemplate.from_template(instruction) | model
    reply = chain.invoke(values)
    tokens = reply.usage_metadata["total_tokens"] if reply.usage_metadata else 0
    return reply.content.strip(), tokens


def require(state: ReviewState, field: str, node: str) -> str:
    """Read a field an earlier node in the graph is supposed to have set.

    A bare state[field] fails with a KeyError that says nothing about why - and
    the whole point of this script is inviting readers to rewire the edges, so
    a node running before the one that fills its input is a real mistake to
    make, not a hypothetical one. This names the missing field and the node
    that needed it instead.
    """
    value = state.get(field)
    if not value:
        raise ValueError(f"node {node!r} needs '{field}', but no earlier node produced it - check the edges")
    return value


# 1-2. The five nodes. Each one adds exactly one field, which is what lets the
# same functions be reused under two different topologies further down.


def make_nodes(model: ChatOpenAI) -> dict:
    """Build the node functions, closing over the model they call."""

    def gather(state: ReviewState) -> dict:
        """Collect the raw considerations before any judgement is applied."""
        facts, tokens = ask(
            model,
            "List three concrete factors that bear on this question, one per line, no preamble:\n{question}",
            question=state["question"],
        )
        return {"facts": facts, "visited": ["gather"], "tokens": tokens}

    def frame(state: ReviewState) -> dict:
        """Turn loose factors into a single sentence stating the real trade-off."""
        framing, tokens = ask(
            model,
            "In one sentence, name the central trade-off these factors describe:\n{facts}",
            facts=require(state, "facts", "frame"),
        )
        return {"framing": framing, "visited": ["frame"], "tokens": tokens}

    def propose(state: ReviewState) -> dict:
        """Generate candidate courses of action against that trade-off."""
        options, tokens = ask(
            model,
            "Given this trade-off, propose two opposing courses of action, one per line, no preamble:\n{framing}",
            framing=require(state, "framing", "propose"),
        )
        return {"options": options, "visited": ["propose"], "tokens": tokens}

    def choose(state: ReviewState) -> dict:
        """Pick one candidate and say what would have to be true for it to hold."""
        choice, tokens = ask(
            model,
            "Pick one of these and state the condition under which it stops being right. Two sentences:\n{options}",
            options=require(state, "options", "choose"),
        )
        return {"choice": choice, "visited": ["choose"], "tokens": tokens}

    def report(state: ReviewState) -> dict:
        """Compose the pieces into the answer the caller actually receives."""
        answer, tokens = ask(
            model,
            "Write a three-sentence answer to '{question}' using this reasoning:\n{choice}",
            question=state["question"],
            choice=require(state, "choice", "report"),
        )
        return {"answer": answer, "visited": ["report"], "tokens": tokens}

    def triage(state: ReviewState) -> dict:
        """Decide whether the question needs the pipeline or a direct answer.

        A verdict that names neither word means the model did not follow the
        format, not that it picked "shallow" - falling back to deep is the
        quality-over-cost default for that case, and the raw reply is kept so a
        wrong call can be diagnosed against what the model actually said.
        """
        verdict, tokens = ask(
            model,
            "Reply with exactly one word, shallow or deep. A question is shallow if a "
            "single factual sentence answers it, and deep if it needs weighing "
            "trade-offs.\nQuestion: {question}",
            question=state["question"],
        )
        verdict = verdict.lower()
        if "deep" in verdict:
            depth = "deep"
        elif "shallow" in verdict:
            depth = "shallow"
        else:
            depth = FALLBACK_DEPTH
        return {"depth": depth, "verdict_raw": verdict, "visited": ["triage"], "tokens": tokens}

    def answer_directly(state: ReviewState) -> dict:
        """Answer in one pass, skipping every analysis node."""
        answer, tokens = ask(model, "Answer in one short sentence:\n{question}", question=state["question"])
        return {"answer": answer, "visited": ["answer_directly"], "tokens": tokens}

    return {
        "gather": gather,
        "frame": frame,
        "propose": propose,
        "choose": choose,
        "report": report,
        "triage": triage,
        "answer_directly": answer_directly,
    }


def build_pipeline(nodes: dict):
    """Step 2. Wire the five analysis nodes in a fixed order, with no branches.

    Every question pays for all five calls, whether or not it needed them. That
    is the honest trade of a linear pipeline: predictable and easy to reason
    about, and wasteful the moment the inputs vary in difficulty.
    """
    builder = StateGraph(ReviewState)
    for name in ["gather", "frame", "propose", "choose", "report"]:
        builder.add_node(name, nodes[name])
    builder.add_edge(START, "gather")
    builder.add_edge("gather", "frame")
    builder.add_edge("frame", "propose")
    builder.add_edge("propose", "choose")
    builder.add_edge("choose", "report")
    builder.add_edge("report", END)
    return builder.compile()


def build_router(nodes: dict):
    """Step 5. Put a triage node in front and let its verdict pick the path.

    The five analysis nodes are the same objects the pipeline uses; only the
    edges differ. A conditional edge is a function from state to the name of the
    next node, so the decision is data the graph produced rather than a flag the
    caller had to set in advance.
    """
    builder = StateGraph(ReviewState)
    for name in ["triage", "gather", "frame", "propose", "choose", "report", "answer_directly"]:
        builder.add_node(name, nodes[name])

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        lambda state: "gather" if state["depth"] == "deep" else "answer_directly",
        {"gather": "gather", "answer_directly": "answer_directly"},
    )
    builder.add_edge("gather", "frame")
    builder.add_edge("frame", "propose")
    builder.add_edge("propose", "choose")
    builder.add_edge("choose", "report")
    builder.add_edge("report", END)
    builder.add_edge("answer_directly", END)
    return builder.compile()


FIELDS = ["depth", "facts", "framing", "options", "choice", "answer"]


def run_and_trace(graph, question: str, show_fields: bool = False) -> dict:
    """Step 3. Stream the run and print what each node added to the state."""
    final: dict = {}
    for state in graph.stream({"question": question, "visited": [], "tokens": 0}, stream_mode="values"):
        final = state
        if show_fields and state.get("visited"):
            filled = [name for name in FIELDS if state.get(name)]
            print(f"    after {state['visited'][-1]:<16} state holds: {', '.join(filled)}")
    return final


def describe(graph, label: str) -> None:
    """Step 7. Print the compiled edges so the two shapes can be compared."""
    print(f"  {label}")
    for edge in sorted(graph.get_graph().edges, key=lambda item: (item.source, item.target)):
        marker = " (conditional)" if edge.conditional else ""
        print(f"    {edge.source} -> {edge.target}{marker}")


def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is not set; every step needs it. Stopping here.")
        return

    nodes = make_nodes(build_model())
    pipeline = build_pipeline(nodes)
    router = build_router(nodes)

    deep_question = "Should a small team run its own database server instead of paying for a managed one?"
    shallow_question = "What port does PostgreSQL listen on by default?"
    ambiguous_question = "Is PostgreSQL better than MySQL?"

    print("--- 1-3. Fixed pipeline, every node runs ---")
    print(f"  question: {deep_question}")
    deep_pipeline = run_and_trace(pipeline, deep_question, show_fields=True)
    print(f"    nodes run: {len(deep_pipeline['visited'])}, tokens: {deep_pipeline['tokens']}"
          f" -> {' -> '.join(deep_pipeline['visited'])}")
    print(f"  answer: {deep_pipeline['answer']}")

    print("\n--- 4. The shallow question, forced through the same fixed pipeline ---")
    print(f"  question: {shallow_question}")
    shallow_pipeline = run_and_trace(pipeline, shallow_question)
    print(f"    nodes run: {len(shallow_pipeline['visited'])}, tokens: {shallow_pipeline['tokens']}"
          f" -> {' -> '.join(shallow_pipeline['visited'])}")
    print(f"    the pipeline's 'framing' step invented a trade-off anyway: {shallow_pipeline['framing']}")
    print(f"  answer: {shallow_pipeline['answer']}")

    print("\n--- 5-6. Same nodes, triage decides the path ---")
    router_runs = {}
    for question in [shallow_question, deep_question, ambiguous_question]:
        result = run_and_trace(router, question)
        router_runs[question] = result
        print(f"  question: {question}")
        print(f"    triage said {result['depth']} (model replied {result['verdict_raw']!r}),"
              f" nodes run: {len(result['visited'])}, tokens: {result['tokens']}"
              f" -> {' -> '.join(result['visited'])}")
        print(f"    answer: {result['answer']}")

    print("\n  what routing actually costs or saves, in measured tokens:")
    shallow_saving = shallow_pipeline["tokens"] - router_runs[shallow_question]["tokens"]
    deep_overhead = router_runs[deep_question]["tokens"] - deep_pipeline["tokens"]
    print(f"    shallow question: pipeline {shallow_pipeline['tokens']}, router {router_runs[shallow_question]['tokens']}"
          f" (saves {shallow_saving})")
    print(f"    deep question:    pipeline {deep_pipeline['tokens']}, router {router_runs[deep_question]['tokens']}"
          f" (costs {deep_overhead} extra for triage)")
    # Each call is an independent generation, so ordinary length variance in
    # gather/choose/etc. can be bigger than the one extra triage call - when
    # that happens deep_overhead comes out negative and a break-even percentage
    # would be noise dressed up as a number, not a real threshold.
    if deep_overhead <= 0:
        print("    triage's overhead was smaller than this run's normal generation variance"
              " - routing measured no real cost on the deep path here")
    elif shallow_saving + deep_overhead > 0:
        breakeven = deep_overhead / (shallow_saving + deep_overhead)
        print(f"    break-even share of shallow traffic: {breakeven:.0%}"
              f" (routing wins once more than that share of questions is shallow)")

    print("\n--- 7. The two topologies ---")
    describe(pipeline, "fixed pipeline")
    describe(router, "conditional router")


if __name__ == "__main__":
    main()
