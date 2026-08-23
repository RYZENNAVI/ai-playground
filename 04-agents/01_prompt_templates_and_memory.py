"""Render prompts from templates and carry a conversation across turns.

Demonstrates the two pieces every chat application assembles by hand:
    1. Fill a single-variable template and read the exact string it produced.
    2. Split the same instruction into a system role and a human role.
    3. Pipe a template into a model and a parser to get plain text back.
    4. Replay stored messages so the model can resolve "it" and "that one".
    5. Print the stored history to see what the later turns actually sent.
    6. Drop the history and ask the same follow-up, to show what memory buys.

Module 04: Agents - Prompt Templates and Conversation Memory.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"


def build_model() -> ChatOpenAI:
    """Return a chat model that speaks the OpenAI wire format.

    Every provider in this repository is reached the same way: the OpenAI
    request shape plus a base_url. Swapping providers is a two-line change and
    never touches the prompts or the chain built on top.
    """
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0,
    )


def render_single_variable_template() -> None:
    """Step 1. Fill one slot in a template and show the resulting string.

    A template is string formatting with a declared list of inputs. The value of
    declaring them is that a missing variable fails here, while the prompt is
    still a local object, instead of arriving at the model as the literal text
    "{product}" and coming back as a confidently wrong answer.
    """
    print("--- 1. Single-variable template ---")
    template = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    for product in ["colorful socks", "noise-cancelling headphones"]:
        print(f"  input:  {product}")
        print(f"  render: {template.format(product=product)}")

    missing = template.input_variables
    print(f"  declared variables: {missing}")


def render_role_split_template() -> None:
    """Step 2. Put the standing instruction and the payload in separate roles.

    The single-variable template above mixes the instruction and the user's text
    into one blob. Splitting them means the instruction stays fixed while only
    the human message changes, and the model treats the two differently: the
    system message describes the job, the human message is the thing to act on.
    """
    print("\n--- 2. System and human roles ---")
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You translate {source_language} into {target_language}. Reply with the translation only."),
            ("human", "{text}"),
        ]
    )
    rendered = chat_template.format_messages(
        source_language="English",
        target_language="French",
        text="I love programming.",
    )
    for message in rendered:
        print(f"  [{message.type}] {message.content}")


def run_template_model_parser_chain(model: ChatOpenAI) -> None:
    """Step 3. Compose template, model, and parser into one callable.

    The pipe operator wires three objects into a single runnable: the template
    turns a dict into messages, the model turns messages into a response object,
    and the parser pulls the text out of that object. Without the parser the
    result is a message wrapper, not a string, which is the usual surprise when
    the output is passed straight into the next step.
    """
    print("\n--- 3. Template to model to parser ---")
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "You translate {source_language} into {target_language}. Reply with the translation only."),
                ("human", "{text}"),
            ]
        )
        | model
        | StrOutputParser()
    )
    payload = {
        "source_language": "English",
        "target_language": "French",
        "text": "I love programming.",
    }
    answer = chain.invoke(payload)
    print(f"  with parser:    {answer!r}")

    unparsed = (chain.steps[0] | chain.steps[1]).invoke(payload)
    print(f"  without parser: {type(unparsed).__name__} carrying {unparsed.content!r}")


def build_conversation(model: ChatOpenAI):
    """Wrap a chain in a graph that stores its own messages between calls.

    The model is stateless: each request is judged only on the messages it
    carries. Memory is therefore not a model feature but a caller habit - keep
    the transcript, and prepend it next time. Here a checkpointer does the
    keeping, MessagesPlaceholder marks the slot the transcript is poured into,
    and a thread id decides which transcript this call belongs to.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise assistant. Answer in one short sentence."),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model

    def respond(state: MessagesState) -> dict:
        return {"messages": [chain.invoke({"messages": state["messages"]})]}

    builder = StateGraph(MessagesState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    return builder.compile(checkpointer=InMemorySaver())


def run_conversation(model: ChatOpenAI) -> None:
    """Steps 4 and 5. Ask a follow-up that only works if the past is replayed.

    The second question says "it" and never names the subject. A sensible answer
    proves the first exchange was resent, because nothing else in the request
    identifies what "it" refers to.
    """
    print("\n--- 4. Multi-turn conversation ---")
    conversation = build_conversation(model)
    config = {"configurable": {"thread_id": "demo"}}

    turns = [
        "I am building a small tool that renames photo files by date.",
        "What should I call it?",
    ]
    for turn in turns:
        state = conversation.invoke({"messages": [HumanMessage(turn)]}, config=config)
        print(f"  user: {turn}")
        print(f"  bot:  {state['messages'][-1].text}")

    print("\n--- 5. What the checkpointer holds ---")
    stored = conversation.get_state(config).values["messages"]
    for index, message in enumerate(stored, start=1):
        preview = message.text.replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:90] + "..."
        print(f"  {index}. [{message.type}] {preview}")
    print(f"  stored messages: {len(stored)}")


def run_without_memory(model: ChatOpenAI) -> None:
    """Step 6. Send the follow-up alone, with no transcript in front of it.

    Same model, same question, one difference: the first turn is gone. The
    answer stops being about a photo-renaming tool, which is the clearest
    evidence that memory lives in the request payload and nowhere else.
    """
    print("\n--- 6. The same follow-up without history ---")
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", "You are a concise assistant. Answer in one short sentence."),
                ("human", "{input}"),
            ]
        )
        | model
        | StrOutputParser()
    )
    answer = chain.invoke({"input": "What should I call it?"})
    print("  user: What should I call it?")
    print(f"  bot:  {answer}")


def main() -> None:
    render_single_variable_template()
    render_role_split_template()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\nDEEPSEEK_API_KEY is not set; steps 3 to 6 need it. Stopping here.")
        return

    model = build_model()
    run_template_model_parser_chain(model)
    run_conversation(model)
    run_without_memory(model)


if __name__ == "__main__":
    main()
