"""Universal Web Search Agent using OpenAI SDK format.

Demonstrates web-grounded generation via standard Function Calling (Tools):
    1. Send user query + search tool definitions (`web_search_tool`).
    2. Model identifies real-time query and requests tool call.
    3. Execute live web search locally and return snippets as `role: "tool"`.
    4. Bounded Agent loop (`max_iterations=3`) preventing runaway infinite tool calls.
    5. Uses `tool_choice="none"` on forced exit to guarantee pure text summary.

Supported Providers: DeepSeek, Google Gemini, OpenAI (Universal).
Module 01: LLM Foundation — Web Search Grounding.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from openai import OpenAI

# Automatically load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Client Setup: DeepSeek (Primary) / Gemini / OpenAI Fallback
# ---------------------------------------------------------------------------
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if deepseek_key:
    api_key = deepseek_key
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    default_model = "deepseek-chat"
    provider = "DeepSeek"
elif gemini_key:
    api_key = gemini_key
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    default_model = "gemini-2.5-flash"
    provider = "Google Gemini"
else:
    api_key = openai_key or "dummy"
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model = "gpt-4o-mini"
    provider = "OpenAI"

if not (deepseek_key or gemini_key or openai_key):
    raise RuntimeError("No API key found! Please set DEEPSEEK_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY in .env file.")

client = OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Universal Web Search Tool (No API Key Required)
# ---------------------------------------------------------------------------
def web_search_tool(query: str) -> str:
    """Universal search tool querying live web/Wikipedia API."""
    encoded = urllib.parse.quote(query)
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json"

    # Wikimedia MediaWiki API requires a compliant User-Agent with project/contact info
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
                snippets.append(f"Title: {title}\nSnippet: {snippet_text}")
            return "\n\n".join(snippets) if snippets else f"No relevant live search results found for: {query!r}."
    except Exception as e:
        print(f"[Search Engine Notice] Search API request failed: {e}")
        return f"Live search engine request encountered an issue ({e}). Please answer based on available knowledge."


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


# ---------------------------------------------------------------------------
# Universal Bounded Web Search Agent Loop
# ---------------------------------------------------------------------------
def universal_web_search(user_query: str, model: str = default_model, max_iterations: int = 3) -> str:
    """Universal web search agent with bounded iterations (max_iterations) and a tool_choice='none' circuit breaker."""
    print(f"=== Universal Bounded Web Search Agent ===")
    print(f"Provider      : {provider}")
    print(f"Model         : {model}")
    print(f"Max Search Limit: {max_iterations} rounds\n")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that answers questions using live web search when necessary. Call web_search_tool if needed.",
        },
        {"role": "user", "content": user_query},
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=SEARCH_TOOL_SCHEMA,
        )

        msg = response.choices[0].message
        messages.append(msg)

        # Exit loop early when model generates final text answer without tool calls
        if not msg.tool_calls:
            return msg.content

        # Execute requested search tool calls
        for tool_call in msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            search_query = args.get("query", user_query)
            fn_name = getattr(tool_call.function, "name", "web_search_tool")
            print(f"[Round {iteration + 1}/{max_iterations}] Tool Request -> Searching: {search_query!r}")
            search_result = web_search_tool(search_query)
            print(f"[Round {iteration + 1}/{max_iterations}] Snippet -> {search_result[:150]}...\n")
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": search_result,
                }
            )

    # Hard stop protection with tool_choice="none": Guarantee pure text answer
    print(f"[Notice] Maximum search limit ({max_iterations} rounds) reached. Force generating final summary...")
    messages.append({
        "role": "user",
        "content": "Search iteration limit reached. Do not perform any more tool calls or searches. Please provide your best complete final answer now using all search information gathered so far."
    })

    try:
        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=SEARCH_TOOL_SCHEMA,
            tool_choice="none"
        )
    except Exception:
        final_response = client.chat.completions.create(
            model=model,
            messages=messages
        )

    return final_response.choices[0].message.content


if __name__ == "__main__":
    query = "What are the latest major news announcements from DeepSeek or OpenAI recently?"
    print(f"Query: {query!r}\n")

    # Set max_iterations=3 to keep execution fast and prevent runaway searches
    answer = universal_web_search(query, max_iterations=3)
    print(f"Final Answer:\n{answer}\n")
