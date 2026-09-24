"""Diagnose a database alert with a model that fetches the server metrics it needs.

Demonstrates a tool loop that stops only when the model stops asking:
    1. Give the model an alert and a system message telling it to check the server first.
    2. Offer one tool that returns connections, CPU and memory, drawn at random per call.
    3. Keep calling the model and running the tools it asks for, with no limit on rounds.
    4. Print the diagnosis once a reply arrives without a tool call.

Script 06 adds the round limit this loop does not have.
"""

import json
import os
import random
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
# Mock Infrastructure Monitoring API
# ---------------------------------------------------------------------------
def get_current_status() -> str:
    """Mock monitoring system retrieving database server metrics."""
    status_info = {
        "connections": random.randint(50, 150),
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
    """Run AIOps multi-step agent loop."""
    print(f"=== AIOps Incident Handler ===")
    print(f"Alert: {alert_text}\n")

    messages = [
        {
            "role": "system",
            "content": "You are a senior AIOps engineer. Upon receiving a database alert, call get_current_status first to fetch real-time server performance metrics, then provide root-cause analysis and action plan.",
        },
        {"role": "user", "content": alert_text},
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            print("--- Final Diagnosis & Action Plan ---")
            print(message.content)
            break

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            print(f"[AIOps Tool Request] Calling {fn_name}()")
            if fn_name == "get_current_status":
                status_result = get_current_status()
                print(f"[AIOps Tool Response] {status_result}\n")
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": fn_name,
                        "content": status_result,
                    }
                )


if __name__ == "__main__":
    alert = "[CRITICAL ALERT] Database connection count exceeded threshold (Current limit: 80). Time: 2026-08-06 15:30:00"
    handle_ops_alert(alert)
