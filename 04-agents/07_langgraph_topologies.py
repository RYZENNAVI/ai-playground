"""Run the same nodes as a fixed pipeline and as a routed graph with LangGraph.

LangGraph builds an agent as a StateGraph. Each node is a function that reads
one shared state and returns the fields it adds, and edges decide which node
runs next. A conditional edge picks the next node from the state, which is how
the graph routes. Here five model calls (gather, frame, propose, choose,
report) form an analysis pipeline. The router puts a triage call in front: a
shallow question gets one direct answer, and a deep one goes through all five.

The run prints four parts:
    1. Fixed pipeline. A deep question through the five nodes, with the
       fields the state holds after each one.
    2. The shallow question through the same pipeline. All five nodes run,
       and frame names a trade-off the question does not have.
    3. Triage decides the path. A shallow, a deep and an ambiguous question
       through the router. Then what routing saves on the shallow question,
       what the triage call adds on the deep one, and the share of shallow
       questions above which routing pays.
    4. The two topologies. The edges of both compiled graphs.
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

# Triage's path when the verdict names neither word. Deep keeps quality the
# default when the classifier is unclear.
FALLBACK_DEPTH = "deep"


class ReviewState(TypedDict):
    """The state every node reads and returns fields into. visited and tokens
    have a reducer, so each node's value is added instead of replacing the last."""

    question: str
    depth: Optional[Literal["shallow", "deep"]]
    verdict_raw: Optional[str]
    triage_tokens: Optional[int]
    facts: Optional[str]
    framing: Optional[str]
    options: Optional[str]
    choice: Optional[str]
    answer: Optional[str]
    visited: Annotated[list[str], operator.add]
    tokens: Annotated[int, operator.add]


def build_model() -> ChatOpenAI:
    """Return the chosen chat model through LangChain's OpenAI client."""
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0,
    )


def ask(model: ChatOpenAI, instruction: str, **values: str) -> tuple[str, int]:
    """Send one templated instruction and return the reply plus its token cost."""
    chain = ChatPromptTemplate.from_template(instruction) | model
    reply = chain.invoke(values)
    tokens = reply.usage_metadata["total_tokens"] if reply.usage_metadata else 0
    return reply.content.strip(), tokens


def require(state: ReviewState, field: str, node: str) -> str:
    """Read a field an earlier node should have set, and name it if the edges skipped that node."""
    value = state.get(field)
    if not value:
        raise ValueError(f"node {node!r} needs '{field}', but no earlier node produced it. Check the edges.")
    return value


# The nodes. Each adds one field, so the same functions serve both topologies.


def make_nodes(model: ChatOpenAI | None) -> dict:
    """Build the node functions, closing over the model they call. With no model
    the graphs still build, but the nodes cannot run."""

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
            "Question: {question}\nIn one sentence, name the central trade-off these factors describe:\n{facts}",
            question=state["question"],
            facts=require(state, "facts", "frame"),
        )
        return {"framing": framing, "visited": ["frame"], "tokens": tokens}

    def propose(state: ReviewState) -> dict:
        """Generate candidate courses of action against that trade-off."""
        options, tokens = ask(
            model,
            "Question: {question}\nGiven this trade-off, propose two opposing courses of action, "
            "one per line, no preamble:\n{framing}",
            question=state["question"],
            framing=require(state, "framing", "propose"),
        )
        return {"options": options, "visited": ["propose"], "tokens": tokens}

    def choose(state: ReviewState) -> dict:
        """Pick one candidate and say what would have to be true for it to hold."""
        choice, tokens = ask(
            model,
            "Question: {question}\nPick one of these and state the condition under which it stops "
            "being right. Two sentences:\n{options}",
            question=state["question"],
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
        """Classify the question as shallow or deep. The raw reply and the call's own
        token cost are kept, the cost because it is exactly what routing adds."""
        verdict, tokens = ask(
            model,
            "Reply with exactly one word, shallow or deep. A question is shallow if a "
            "single factual sentence answers it, and deep if it needs weighing "
            "trade-offs.\nQuestion: {question}",
            question=state["question"],
        )
        if "deep" in verdict.lower():
            depth = "deep"
        elif "shallow" in verdict.lower():
            depth = "shallow"
        else:
            depth = FALLBACK_DEPTH
        return {
            "depth": depth,
            "verdict_raw": verdict,
            "triage_tokens": tokens,
            "visited": ["triage"],
            "tokens": tokens,
        }

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
    """Wire the five analysis nodes in a fixed order, with no branches. Every
    question pays for all five calls."""
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
    """Put a triage node in front of the same five nodes. A conditional edge reads
    the depth triage wrote and sends the question to gather or answer_directly."""
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
    """Stream the run and, if asked, print which fields the state holds after each node."""
    final: dict = {}
    for state in graph.stream({"question": question, "visited": [], "tokens": 0}, stream_mode="values"):
        final = state
        if show_fields and state.get("visited"):
            filled = [name for name in FIELDS if state.get(name)]
            print(f"    after {state['visited'][-1]:<16} state holds: {', '.join(filled)}")
    return final


def describe(graph, label: str) -> None:
    """Print one compiled graph's edges, sorted so the two shapes can be compared."""
    print(f"  {label}")
    for edge in sorted(graph.get_graph().edges, key=lambda item: (item.source, item.target)):
        marker = " (conditional)" if edge.conditional else ""
        print(f"    {edge.source} -> {edge.target}{marker}")


def print_topologies(pipeline, router) -> None:
    """Part 4. Print the edges of both graphs. No model is called."""
    print("\n--- 4. The two topologies ---")
    describe(pipeline, "fixed pipeline")
    describe(router, "conditional router")


def main() -> None:
    nodes = make_nodes(build_model() if API_KEY else None)
    pipeline = build_pipeline(nodes)
    router = build_router(nodes)

    if not API_KEY:
        print_topologies(pipeline, router)
        print("\nNo DEEPSEEK_API_KEY or OPENAI_API_KEY in .env. Parts 1 to 3 need one, so only part 4 ran.")
        return

    deep_question = "Should a small team run its own database server instead of paying for a managed one?"
    shallow_question = "What port does PostgreSQL listen on by default?"
    ambiguous_question = "Is PostgreSQL better than MySQL?"

    print("--- 1. Fixed pipeline ---")
    print(f"  question: {deep_question}")
    deep_pipeline = run_and_trace(pipeline, deep_question, show_fields=True)
    print(f"    nodes run: {len(deep_pipeline['visited'])}, tokens: {deep_pipeline['tokens']}"
          f" -> {' -> '.join(deep_pipeline['visited'])}")
    print(f"  answer: {deep_pipeline['answer']}")

    print("\n--- 2. The shallow question through the same pipeline ---")
    print(f"  question: {shallow_question}")
    shallow_pipeline = run_and_trace(pipeline, shallow_question)
    print(f"    nodes run: {len(shallow_pipeline['visited'])}, tokens: {shallow_pipeline['tokens']}"
          f" -> {' -> '.join(shallow_pipeline['visited'])}")
    print(f"    the pipeline's 'framing' step invented a trade-off anyway: {shallow_pipeline['framing']}")
    print(f"  answer: {shallow_pipeline['answer']}")

    print("\n--- 3. Triage decides the path ---")
    router_runs = {}
    for question in [shallow_question, deep_question, ambiguous_question]:
        result = run_and_trace(router, question)
        router_runs[question] = result
        print(f"  question: {question}")
        print(f"    triage said {result['depth']} (model replied {result['verdict_raw']!r}),"
              f" nodes run: {len(result['visited'])}, tokens: {result['tokens']}"
              f" -> {' -> '.join(result['visited'])}")
        print(f"    answer: {result['answer']}")

    print("\n  what routing saves or adds, in measured tokens:")
    shallow_saving = shallow_pipeline["tokens"] - router_runs[shallow_question]["tokens"]
    # The triage call is the only node the router adds on the deep path. Two whole
    # runs of the deep question also differ by generation length, so they are not compared.
    triage_cost = router_runs[deep_question]["triage_tokens"]
    print(f"    shallow question: pipeline {shallow_pipeline['tokens']}, router {router_runs[shallow_question]['tokens']}"
          f" (saves {shallow_saving})")
    print(f"    deep question:    the triage call adds {triage_cost}")
    breakeven = triage_cost / (shallow_saving + triage_cost)
    print(f"    break-even share of shallow traffic: {breakeven:.0%}"
          f" (routing wins once more than that share of questions is shallow)")

    print_topologies(pipeline, router)


if __name__ == "__main__":
    main()
