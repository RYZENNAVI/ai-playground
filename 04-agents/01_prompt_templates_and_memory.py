"""Build prompts with LangChain prompt templates, and give a chat model memory
with a LangGraph checkpointer.

A prompt template is text with named slots that LangChain fills in. A chat
template also splits the text into a system message and a human message. The
model remembers nothing between requests. The checkpointer stores each
thread's messages, and the graph sends all of them again with the next
question.

The run prints six parts:
    1. Single-variable template. One template filled with two products, and
       the variable names it read from its text.
    2. System and human roles. A translation instruction as the system
       message and the text to translate as the human message. Nothing is
       sent to the model yet.
    3. Template to model to parser. The same translation template piped into
       the model and StrOutputParser. Run again without the parser, the chain
       returns an AIMessage instead of a string.
    4. Multi-turn conversation. A second question, "What should I call it?",
       that only makes sense after the first.
    5. What the checkpointer holds. The four messages stored for the thread
       in part 4. The system message is not stored; the graph adds it on
       every call.
    6. The same follow-up without history. The second question sent alone.
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


def build_model() -> ChatOpenAI:
    """Return the chosen chat model through LangChain's OpenAI client."""
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0,
    )


def render_single_variable_template() -> None:
    """Step 1. Fill one template with two products and print the variables it read from its text."""
    print("--- 1. Single-variable template ---")
    template = PromptTemplate.from_template(
        "What is a good name for a company that makes {product}?"
    )
    for product in ["colorful socks", "noise-cancelling headphones"]:
        print(f"  input:  {product}")
        print(f"  render: {template.format(product=product)}")

    variables = template.input_variables
    print(f"  variables read from the text: {variables}")


def render_role_split_template() -> None:
    """Step 2. Render the instruction as a system message and the text as a human message."""
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
    """Step 3. Pipe template, model and parser into one chain, then run it again
    without the parser, which returns an AIMessage."""
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
    """Wrap the chain in a graph whose checkpointer stores each thread's messages.
    The model keeps nothing between requests, so the graph resends them every turn."""
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
    """Steps 4 and 5. Ask a follow-up that names nothing, then print what the checkpointer stored."""
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
    """Step 6. Send the follow-up alone, with no stored messages in front of it."""
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

    if not API_KEY:
        print("\nNo DEEPSEEK_API_KEY or OPENAI_API_KEY in .env. Steps 3 to 6 need one, so the run stops here.")
        return

    model = build_model()
    run_template_model_parser_chain(model)
    run_conversation(model)
    run_without_memory(model)


if __name__ == "__main__":
    main()
