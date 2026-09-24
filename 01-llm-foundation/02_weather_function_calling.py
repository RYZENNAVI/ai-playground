"""This script asks a chat model about the weather in two cities. The model has no live
weather data, so it cannot answer on its own. Instead it asks the script to call a local
function, get_current_weather, and the script sends the results back.

The script uses DeepSeek when DEEPSEEK_API_KEY is set, and OpenAI otherwise. The run
prints three steps:
    1. The question, sent together with the JSON schema of the weather function.
    2. Each tool call the model returns instead of an answer, usually one per city, and
       the result of running it. The function reads a fixed table of temperatures, so
       the numbers are made up.
    3. The final answer, which the model writes from those results.
"""

import json
import os
import sys
from openai import OpenAI

# Read the keys from the .env file at the repository root, if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Ensure UTF-8 output on the Windows terminal (model replies may contain emoji)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    raise SystemExit("Set DEEPSEEK_API_KEY or OPENAI_API_KEY in .env and retry.")

if os.getenv("DEEPSEEK_API_KEY"):
    base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"
else:
    # OPENAI_BASE_URL belongs to OPENAI_API_KEY only, so a DeepSeek key is never
    # sent to whatever endpoint that variable points at.
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    default_model = "gpt-4o-mini"

client = OpenAI(api_key=api_key, base_url=base_url)


def get_current_weather(location: str) -> str:
    """Return a made-up weather report for one city as a JSON string. Cities that are
    not in the table get 20 degrees Celsius."""
    temperatures = {
        "Dalian": 10,
        "Shanghai": 36,
        "Shenzhen": 37,
        "Beijing": 25,
        "San Francisco": 18,
    }
    temp = temperatures.get(location, 20)
    return json.dumps(
        {
            "location": location,
            "temperature": temp,
            "unit": "celsius",
            "forecast": ["Sunny", "Light breeze"],
        },
        ensure_ascii=False,
    )


AVAILABLE_TOOLS = {
    "get_current_weather": get_current_weather
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather and temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. Dalian, Shanghai, or San Francisco",
                    },
                },
                "required": ["location"],
            },
        },
    }
]


def run_tool_conversation(query: str = "What is the weather like in Dalian right now?", model: str = default_model):
    """Ask one question, run the tool calls the model requests, and return its final answer."""
    # 1. Question
    print("--- 1. Question ---")
    print(f"Query: {query!r}")
    messages = [{"role": "user", "content": query}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 2. Tool calls and results
    print("\n--- 2. Tool calls and results ---")
    if not tool_calls:
        print("Model did not request any tool call.")
        return response_message.content

    # The tool results must follow the assistant message that requested them.
    messages.append(response_message)

    for tool_call in tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        print(f"[Tool Requested] {fn_name}({fn_args})")

        tool_func = AVAILABLE_TOOLS.get(fn_name)
        if tool_func:
            tool_output = tool_func(**fn_args)
            print(f"[Tool Result   ] {tool_output}")

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_output,
                }
            )

    # 3. Final answer
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return final_response.choices[0].message.content


if __name__ == "__main__":
    print(f"=== Weather function calling ({default_model}) ===")
    final_answer = run_tool_conversation("How is the weather in Shanghai and Shenzhen today?")
    print(f"\n--- 3. Final answer ---\n{final_answer}")
