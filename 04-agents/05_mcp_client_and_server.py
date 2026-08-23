"""Serve tools over the Model Context Protocol and drive them from a model.

Demonstrates both halves of a protocol that is usually only shown from one side:
    1. Publish three local functions as tools on a server that speaks stdio.
    2. Start that server as a subprocess and complete the protocol handshake.
    3. Ask the server what it offers and print the schema it advertises.
    4. Translate those schemas into the tool array the chat API expects.
    5. Run a tool-calling loop where every call crosses the process boundary.
    6. Print the arguments and the raw result of one call as they cross it.

Run `python 05_mcp_client_and_server.py --serve` to start only the server.

Module 04: Agents - MCP Client and Server.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

NOTES_DIR = Path(__file__).parent / "data" / "notes"

MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"

MAX_ROUNDS = 6


# 1. The server. Only these three functions are exposed; anything else in this
# file stays invisible to the client, which is the point of the boundary.


def resolve_note_path(filename: str) -> Path | None:
    """Resolve a model-supplied filename to a path, or None if it escapes NOTES_DIR.

    `NOTES_DIR / filename` alone does not confine anything: `..` segments walk
    back out of the directory, and an absolute path (`C:/Windows/...`) replaces
    NOTES_DIR entirely rather than being appended to it - that is how `Path`'s
    `/` operator is defined, not a bug in this code. filename comes straight
    from the model's tool call, so resolving the path and checking it is still
    inside NOTES_DIR is what actually limits reads to this directory.
    """
    path = (NOTES_DIR / filename).resolve()
    if not path.is_relative_to(NOTES_DIR.resolve()) or path.suffix != ".txt":
        return None
    return path


def build_server():
    """Register the three note tools on a server object and return it.

    Every print statement is deliberately absent from this half of the file. A
    stdio server speaks the protocol on stdout, so one stray print corrupts the
    stream and the client fails at the handshake with a parse error that says
    nothing about the print. Servers log to stderr or to a file, never stdout.
    """
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
            return f"No note named {filename!r}."
        return path.read_text(encoding="utf-8")

    @server.tool()
    def count_words(filename: str) -> int:
        """Count the words in one note file. Input: a filename from list_notes."""
        path = resolve_note_path(filename)
        if path is None or not path.is_file():
            return -1
        return len(path.read_text(encoding="utf-8").split())

    return server


def serve() -> None:
    """Run the server on stdin and stdout until the parent process closes them."""
    build_server().run("stdio")


# 2-6. The client.


def to_chat_tools(mcp_tools: list) -> list[dict]:
    """Rewrite advertised tool schemas into the shape the chat API accepts.

    Both sides already speak JSON Schema, so this is a rename rather than a
    translation: the protocol calls the field input_schema and the chat API
    calls it parameters. Seeing how little happens here is the useful part -
    a server written for one client works with any model that takes tools.
    """
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

            print("\n--- 5. Tool-calling loop across the process boundary ---")
            print(f"  question: {question}")
            client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)
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
                # The model can issue several tool_calls in one round (step 5's
                # demo does), and each one crosses the process boundary on its
                # own round trip - gather runs them concurrently instead of
                # paying for that boundary once per call in sequence.
                results = await asyncio.gather(
                    *(session.call_tool(call.function.name, arguments) for call, arguments in zip(calls, arguments_by_call))
                )
                for call, arguments, result in zip(calls, arguments_by_call, results):
                    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
                    first_line = (text.splitlines() or [""])[0]

                    print(f"    {call.function.name}({arguments}) -> {first_line[:60]}")
                    if not shown_wire_format:
                        print("\n    --- 6. One call, as it crossed the boundary ---")
                        print(f"    request  : {json.dumps({'name': call.function.name, 'arguments': arguments})}")
                        print(f"    response : content={first_line[:48]!r}")
                        print(f"               structured={result.structured_content}\n")
                        shown_wire_format = True

                    messages.append({"role": "tool", "tool_call_id": call.id, "content": text})

            print("  the loop hit its round limit without a final answer")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return

    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    load_dotenv(Path(__file__).parents[1] / ".env")
    load_dotenv(Path(__file__).parents[2] / ".env")

    print("--- 1. Tools this file publishes ---")
    for name in ["list_notes", "read_note", "count_words"]:
        print(f"  {name}")
    print(f"  notes directory: {NOTES_DIR.relative_to(Path(__file__).parent)}\n")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("DEEPSEEK_API_KEY is not set; the client loop needs it. Stopping here.")
        return

    asyncio.run(
        run_client("Which note explains why the checkout outage took so long to diagnose, and what was the fix?")
    )


if __name__ == "__main__":
    main()
