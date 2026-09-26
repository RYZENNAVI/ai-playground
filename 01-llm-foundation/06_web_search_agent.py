"""This script asks a chat model what DeepSeek and OpenAI have announced recently. The
model's knowledge stops at its training cutoff, so it gets a web search tool, and the
script runs each search the model asks for. The search is simulated on purpose: the
tool queries the Wikipedia search API, which needs no key, and the model is told it
searches the web.

Nothing in the tool loop itself stops the model from searching forever, so the script
caps it at three rounds. After the third round it adds a message telling the model to
stop, and calls it once more with tool_choice="none", which rules out another tool call.

The script uses DeepSeek when DEEPSEEK_API_KEY is set, Gemini when GEMINI_API_KEY is
set, and OpenAI otherwise. The run prints three parts:
    1. The question.
    2. Each search, round by round, and the start of its result. One round can hold
       several searches. Each search returns up to three results. A failed request
       gives the model an error message instead of raising. If the model is still
       searching after three rounds, a notice says so.
    3. The final answer.
"""

import html
import json
import os
import sys
import urllib.parse
import urllib.request
from openai import OpenAI

# Read the keys from the .env file at the repository root, if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if not (deepseek_key or gemini_key or openai_key):
    raise SystemExit("Set DEEPSEEK_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY in .env and retry.")

if deepseek_key:
    api_key = deepseek_key
    base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"
    provider = "DeepSeek"
elif gemini_key:
    api_key = gemini_key
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    default_model = "gemini-3.1-flash-lite"
    provider = "Google Gemini"
else:
    api_key = openai_key
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model = "gpt-4o-mini"
    provider = "OpenAI"

client = OpenAI(api_key=api_key, base_url=base_url)


def web_search_tool(query: str) -> str:
    """Stand in for a web search by searching English Wikipedia. Return the top three
    results as text. A failed request returns a message for the model instead of raising."""
    encoded = urllib.parse.quote(query)
    # MediaWiki Action API: list=search runs a full-text search and returns titles and snippets as JSON.
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json"

    # The Wikimedia API expects a User-Agent that names the project and a contact.
    headers = {"User-Agent": "AIPlaygroundAgent/1.0 (https://github.com/ryzennavi/ai-playground)"}

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("query", {}).get("search", [])[:3]
            snippets = []
            for r in results:
                title = r.get("title", "")
                snippet_text = r.get("snippet", "").replace("<span class='searchmatch'>", "").replace("</span>", "").replace('<span class="searchmatch">', "")
                snippets.append(f"Title: {title}\nSnippet: {html.unescape(snippet_text)}")
            return "\n\n".join(snippets) if snippets else f"No results for {query!r}."
    except Exception as e:
        print(f"[Search failed] {e}")
        return f"The search request failed ({e}). Answer from what you already know."


SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "Call this tool to search the web when encountering questions about real-time news, recent events, or information beyond model knowledge cutoffs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords for live web search",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def search_agent(user_query: str, model: str = default_model, max_iterations: int = 3) -> str:
    """Let the model search for at most max_iterations rounds, then force a text answer
    with tool_choice="none". Return the answer."""
    print("=== Wikipedia search agent ===")
    print(f"Provider  : {provider}")
    print(f"Model     : {model}")
    print(f"Max rounds: {max_iterations}\n")

    print("--- 1. Question ---")
    print(f"Query: {user_query!r}\n")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Search the web when a question needs information you do not have.",
        },
        {"role": "user", "content": user_query},
    ]

    print("--- 2. Searches ---")
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=SEARCH_TOOL_SCHEMA,
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            search_query = args.get("query", user_query)
            fn_name = tool_call.function.name
            print(f"[Round {iteration + 1}/{max_iterations}] Search: {search_query!r}")
            search_result = web_search_tool(search_query)
            preview = search_result[:150] + ("..." if len(search_result) > 150 else "")
            print(f"[Round {iteration + 1}/{max_iterations}] Result: {preview}\n")
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": search_result,
                }
            )

    # tool_choice="none" rules out another tool call, so the reply is text.
    print(f"[Cap reached] {max_iterations} rounds used. Asking for the final answer without tools.")
    messages.append({
        "role": "user",
        "content": "You have used all your searches. Answer now from the results you have."
    })

    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=SEARCH_TOOL_SCHEMA,
        tool_choice="none"
    )
    return final_response.choices[0].message.content


if __name__ == "__main__":
    query = "What have DeepSeek and OpenAI announced recently?"

    # Three rounds keep the run short.
    answer = search_agent(query, max_iterations=3)
    print(f"\n--- 3. Final answer ---\n{answer}")
