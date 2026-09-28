"""Serve tools over the Model Context Protocol (MCP) and call them from a model.

MCP is a standard way to publish tools that any client can discover and call.
This file holds both halves. Run normally, it starts a second copy of itself
with --serve as a subprocess. That copy is the server, and it speaks JSON-RPC
over stdin and stdout. The parent is the client. It asks the server for its
tools, hands their schemas to the chat model, and sends each tool call the
model makes across to the server. The three tools work on a folder of notes:
list them, read one, and count its words.

The run prints five parts:
    1. Tools this file publishes. The three functions registered on the
       server, and the notes directory they read.
    2. Handshake. The server's name and the protocol version both sides
       agreed on.
    3. What the server advertises. Each tool's name, required arguments and
       description, as the client received them.
    4. The same schemas in chat-API form. read_note's schema, with
       input_schema renamed to parameters.
    5. Tool-calling loop. The model answers a question about the notes, and
       every call it makes goes to the server. The first call is also shown
       as it crossed: the arguments sent and the full result that came back.

Run `python 05_mcp_client_and_server.py --serve` to start only the server.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

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

NOTES_DIR = Path(__file__).parent / "data" / "notes"

MAX_ROUNDS = 6


# The server half. Only the three functions registered in build_server are
# exposed to the client.


def resolve_note_path(filename: str) -> Path | None:
    """Return the path for a model-supplied filename, or None if it leaves
    NOTES_DIR or is not a .txt file."""
    path = (NOTES_DIR / filename).resolve()
    if not path.is_relative_to(NOTES_DIR.resolve()) or path.suffix != ".txt":
        return None
    return path


def build_server():
    """Register the three note tools on a server and return it. Nothing in this
    half prints, because stdout carries the protocol."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("notes")

    @server.tool()
    def list_notes() -> str:
        """List the note files available, one filename per line."""
        names = sorted(path.name for path in NOTES_DIR.glob("*.txt"))
        return "\n".join(names) if names else "No notes found."

    @server.tool()
    def read_note(filename: str) -> str:
        """Read one note file in full. Input: a filename from list_notes."""
        path = resolve_note_path(filename)
        if path is None or not path.is_file():
            raise ValueError(f"No note named {filename!r}.")
        return path.read_text(encoding="utf-8")

    @server.tool()
    def count_words(filename: str) -> int:
        """Count the words in one note file. Input: a filename from list_notes."""
        path = resolve_note_path(filename)
        if path is None or not path.is_file():
            raise ValueError(f"No note named {filename!r}.")
        return len(path.read_text(encoding="utf-8").split())

    return server


def serve() -> None:
    """Run the server on stdin and stdout until the parent process closes them."""
    build_server().run("stdio")


# The client half.


def to_chat_tools(mcp_tools: list) -> list[dict]:
    """Rewrite advertised tool schemas into the chat API's shape. Both use JSON
    Schema, so only the field name changes, from input_schema to parameters."""
    return [
        {
            "type": "function",
            "function": {
                "name": item.name,
                "description": item.description or "",
                "parameters": item.input_schema,
            },
        }
        for item in mcp_tools
    ]


async def run_client(question: str) -> None:
    """Connect to the server, hand its tools to the model, and run the loop."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from openai import OpenAI

    parameters = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).resolve()), "--serve"])

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            handshake = await session.initialize()
            print("--- 2. Handshake ---")
            print(f"  server: {handshake.server_info.name}, protocol {handshake.protocol_version}")

            listed = (await session.list_tools()).tools
            print("\n--- 3. What the server advertises ---")
            for item in listed:
                required = item.input_schema.get("required", [])
                print(f"  {item.name}({', '.join(required)}): {item.description}")

            chat_tools = to_chat_tools(listed)
            print("\n--- 4. The same schemas in chat-API form ---")
            print(f"  {json.dumps(chat_tools[1], indent=2)}")

            if not API_KEY:
                print("\nNo DEEPSEEK_API_KEY or OPENAI_API_KEY in .env. Part 5 needs one, so the run stops here.")
                return

            print("\n--- 5. Tool-calling loop across the process boundary ---")
            print(f"  question: {question}")
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            messages = [{"role": "user", "content": question}]
            shown_wire_format = False

            for round_number in range(1, MAX_ROUNDS + 1):
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=chat_tools,
                    temperature=0,
                )
                choice = response.choices[0].message
                if not choice.tool_calls:
                    print(f"  rounds: {round_number}")
                    print(f"  answer: {choice.content}")
                    return

                messages.append(choice.model_dump(exclude_none=True))
                calls = choice.tool_calls
                arguments_by_call = [json.loads(call.function.arguments or "{}") for call in calls]
                # The model can return several tool_calls in one round. gather sends
                # them to the server together instead of one after another.
                results = await asyncio.gather(
                    *(session.call_tool(call.function.name, arguments) for call, arguments in zip(calls, arguments_by_call))
                )
                for call, arguments, result in zip(calls, arguments_by_call, results):
                    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
                    first_line = (text.splitlines() or [""])[0]

                    print(f"    {call.function.name}({arguments}) -> {first_line[:60]}")
                    if not shown_wire_format:
                        wire_result = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                        print("      first call as it crossed the boundary:")
                        print(f"      request  : tools/call {json.dumps({'name': call.function.name, 'arguments': arguments})}")
                        print(f"      response : {json.dumps(wire_result)}")
                        shown_wire_format = True

                    messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

            print("  the loop hit its round limit without a final answer")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return

    # Print UTF-8 even when the output is piped or redirected on Windows.
    sys.stdout.reconfigure(encoding="utf-8")

    print("--- 1. Tools this file publishes ---")
    for name in ["list_notes", "read_note", "count_words"]:
        print(f"  {name}")
    print(f"  notes directory: {NOTES_DIR.relative_to(Path(__file__).parent)}\n")

    asyncio.run(
        run_client(
            "Which note explains why the checkout outage took so long to diagnose, "
            "and what follow-up did it recommend?"
        )
    )


if __name__ == "__main__":
    main()
