"""This script hands a database alert to a chat model. The system message tells the
model to check the server first. The model cannot read the server itself, so it calls
a local function, get_current_status, and the script sends the result back. The script
keeps calling the model until a reply comes without a tool call. There is no limit on
rounds; script 06 adds one.

The script uses DeepSeek when DEEPSEEK_API_KEY is set, and OpenAI otherwise. The run
prints three parts:
    1. The alert.
    2. Each tool call and its result. The function draws connections, CPU and memory at
       random on every call, so the numbers change from run to run. The connection
       count always stays above the threshold of 80 in the alert.
    3. The model's diagnosis and action plan.
"""

import json
import os
import random
import sys
from openai import OpenAI

# Read the keys from the .env file at the repository root, if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

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


def get_current_status() -> str:
    """Return made-up metrics for the database server as a JSON string. Each call draws
    new random values."""
    status_info = {
        "connections": random.randint(81, 150),
        "cpu_usage": f"{round(random.uniform(50, 98), 1)}%",
        "memory_usage": f"{round(random.uniform(60, 95), 1)}%",
    }
    return json.dumps(status_info, ensure_ascii=False)


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_status",
            "description": "Query the monitoring system to retrieve database server connections, CPU usage, and memory usage.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def handle_ops_alert(alert_text: str, model: str = default_model):
    """Call the model until it answers without a tool call, run each tool it asks for,
    and print the diagnosis."""
    print("=== Database alert diagnosis ===")
    print("--- 1. Alert ---")
    print(f"{alert_text}\n")

    messages = [
        {
            "role": "system",
            "content": "You are a senior AIOps engineer. Upon receiving a database alert, call get_current_status first to fetch real-time server performance metrics, then give a root-cause analysis and an action plan.",
        },
        {"role": "user", "content": alert_text},
    ]

    print("--- 2. Tool calls and results ---")
    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            print("--- 3. Diagnosis ---")
            print(message.content)
            break

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            print(f"[Tool Requested] {fn_name}()")
            if fn_name == "get_current_status":
                status_result = get_current_status()
                print(f"[Tool Result   ] {status_result}\n")
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": fn_name,
                        "content": status_result,
                    }
                )


if __name__ == "__main__":
    alert = "[CRITICAL] Database connections above the threshold of 80. Time: 2026-08-06 15:30:00"
    handle_ops_alert(alert)
