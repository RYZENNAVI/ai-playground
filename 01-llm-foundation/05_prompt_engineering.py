"""Prompt engineering patterns via standard OpenAI SDK (DeepSeek / OpenAI).

Demonstrates four key prompt engineering techniques:
    1. Structured Template — `# Objective` / `# Output Format` / `# User Input`
    2. JSON Mode          — `response_format={"type": "json_object"}`
    3. Chain-of-Thought   — Step-by-step reasoning & rule checking
    4. Prompt-Tuning      — Meta-prompting: LLM optimizing its own prompt
"""

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


def get_completion(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = default_model,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Generate completion with optional JSON mode."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# --- Shared Example Domain: Mobile Plan Extraction ---------------------------
INSTRUCTION = """
Your task is to identify user preferences for mobile data plans.
Each plan has three attributes: Plan Name, Monthly Price, and Monthly Data.
Identify the user's requirements for these three attributes from the input.
"""

INPUT_TEXT = "Help me subscribe to a 100GB plan with a budget under $30/month."


def compose_prompt(instruction: str, user_input: str, output_format: str = None, cot: bool = False) -> str:
    """Compose structured prompt components."""
    blocks = [f"# Objective\n{instruction}"]
    if cot:
        blocks.append("# Thinking Requirement\nPlease analyze the conversation details step by step.")
    if output_format:
        blocks.append(f"# Output Format\n{output_format}")
    blocks.append(f"# User Input\n{user_input}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    print(f"=== Prompt Engineering Demo ({default_model}) ===\n")

    print("--- 1. Structured Template ---")
    prompt1 = compose_prompt(INSTRUCTION, INPUT_TEXT)
    print(get_completion(prompt1))

    print("\n--- 2. JSON Mode ---")
    prompt2 = compose_prompt(
        INSTRUCTION,
        INPUT_TEXT,
        output_format="Output directly as a JSON object containing keys: name, price_limit, data_gb",
    )
    print(get_completion(prompt2, json_mode=True))

    print("\n--- 3. Chain of Thought (Rule Compliance Evaluation) ---")
    rulebook = """
Evaluate whether customer support reply complies with rules:
1. Must be polite.
2. Must accurately mention plan name, price, and data allowance.
3. Must not end the conversation prematurely.

Available plans:
- Starter: $10/mo, 10GB
- Pro: $30/mo, 100GB
- Unlimited: $50/mo, 1000GB
"""
    conversation = """
Customer: What high-data plans do you have?
Support: Hi! We recommend our Unlimited Plan ($50/mo for 1000GB). Would you like to subscribe?
"""
    prompt3 = compose_prompt(rulebook, conversation, output_format="Reason step by step, then output verdict: Compliant or Non-Compliant", cot=True)
    print(get_completion(prompt3, temperature=0.1))

    print("\n--- 4. Meta-Prompting (Prompt Self-Tuning) ---")
    meta_instruction = "You are a senior Prompt Engineer. Help me optimize the following system prompt to be more rigorous and effective."
    raw_prompt = "You are a mobile plan support agent named Melon. Help users pick plans ($10 for 10GB, $30 for 100GB)."
    prompt = compose_prompt(meta_instruction, raw_prompt)
    print(get_completion(prompt, temperature=0.7))
