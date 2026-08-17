"""Function Calling (Tools) using standard OpenAI SDK format.

Demonstrates the 4-step Tool Call lifecycle:
    1. Send user query + tool definitions (OpenAI `tools` parameter format).
    2. Model detects tool requirement and returns `tool_calls` with arguments.
    3. Execute the function locally and get the result.
    4. Pass the result back with `role: "tool"` to get the final grounded answer.

Supported Providers: DeepSeek (`deepseek-chat`), OpenAI (`gpt-4o-mini`).
"""

import json
import os
import sys
from openai import OpenAI

# Automatically load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# Ensure UTF-8 output on Windows terminal (model replies may contain emoji)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Client Setup: DeepSeek (Primary) with OpenAI Fallback
# ---------------------------------------------------------------------------
api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("No API key found! Please set DEEPSEEK_API_KEY or OPENAI_API_KEY.")

if os.getenv("DEEPSEEK_API_KEY"):
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"
else:
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"

base_url = os.getenv("OPENAI_BASE_URL", default_base_url)
client = OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Local Tool Implementation (Stub)
# ---------------------------------------------------------------------------
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Mock weather service returning JSON string."""
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
            "unit": unit,
            "forecast": ["Sunny", "Light breeze"],
        },
        ensure_ascii=False,
    )


AVAILABLE_TOOLS = {
    "get_current_weather": get_current_weather
}

# ---------------------------------------------------------------------------
# OpenAI Tools Specification
# ---------------------------------------------------------------------------
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
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["location"],
            },
        },
    }
]


def run_tool_conversation(query: str = "What is the weather like in Dalian right now?", model: str = default_model):
    """Run a complete function calling loop."""
    print(f"Query: {query!r}")
    messages = [{"role": "user", "content": query}]

    # Step 1: Send request with tools schema
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_SCHEMA,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Step 2: Check if model wants to call a tool
    if not tool_calls:
        print("Model did not request any tool call.")
        return response_message.content

    # Append assistant's response to message history
    messages.append(response_message)

    # Step 3: Execute tool calls locally
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

    # Step 4: Send tool results back to model for final natural answer
    final_response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return final_response.choices[0].message.content


if __name__ == "__main__":
    print(f"=== Universal Function Calling Demo ({default_model}) ===")
    final_answer = run_tool_conversation("How is the weather in Shanghai and Shenzhen today?")
    print(f"\nFinal Response:\n{final_answer}")
