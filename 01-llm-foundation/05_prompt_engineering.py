"""This script runs four prompts about mobile data plans and prints each reply. Parts 1
and 2 send the same request. Part 2 adds an output format and turns on JSON mode. Parts
3 and 4 are separate tasks that show two more techniques.

The script uses DeepSeek when DEEPSEEK_API_KEY is set, and OpenAI otherwise. The run
prints four parts:
    1. Structured template. The prompt has an objective and the user input, each under a
       Markdown heading. The model reports the plan preferences it finds.
    2. JSON mode. The same prompt gets an output format, and the reply is a JSON object
       that parses without cleanup.
    3. Chain of thought. The model checks a support reply against three rules step by
       step, then gives a verdict. The reply quotes the wrong price, so the right verdict
       is Non-Compliant.
    4. Meta-prompting. The model rewrites a weak system prompt for a support agent.
"""

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


def get_completion(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = default_model,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Send one system message and one user message, and return the reply. With
    json_mode, the API is asked for a JSON object."""
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


# Parts 1 and 2 use this instruction and input.
INSTRUCTION = """
Your task is to identify user preferences for mobile data plans.
Each plan has three attributes: Plan Name, Monthly Price, and Monthly Data.
Identify the user's requirements for these three attributes from the input.
"""

INPUT_TEXT = "Help me subscribe to a 100GB plan with a budget under $30/month."


def compose_prompt(instruction: str, user_input: str, output_format: str = None, cot: bool = False) -> str:
    """Join the prompt sections under Markdown headings: the objective, a thinking
    requirement when cot is set, the output format when one is given, and the user input."""
    blocks = [f"# Objective\n{instruction}"]
    if cot:
        blocks.append("# Thinking Requirement\nPlease analyze the conversation details step by step.")
    if output_format:
        blocks.append(f"# Output Format\n{output_format}")
    blocks.append(f"# User Input\n{user_input}")
    return "\n\n".join(blocks)


if __name__ == "__main__":
    print(f"=== Prompt engineering ({default_model}) ===\n")

    print("--- 1. Structured template ---")
    prompt1 = compose_prompt(INSTRUCTION, INPUT_TEXT)
    print(get_completion(prompt1))

    print("\n--- 2. JSON mode ---")
    prompt2 = compose_prompt(
        INSTRUCTION,
        INPUT_TEXT,
        output_format="Output directly as a JSON object containing keys: name, price_limit, data_gb",
    )
    print(get_completion(prompt2, json_mode=True))

    print("\n--- 3. Chain of thought ---")
    rulebook = """
Check whether the support reply follows these rules:
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
Support: Hi! We recommend our Unlimited Plan ($40/mo for 1000GB). Would you like to subscribe?
"""
    prompt3 = compose_prompt(rulebook, conversation, output_format="Give the verdict at the end: Compliant or Non-Compliant", cot=True)
    print(get_completion(prompt3, temperature=0.1))

    print("\n--- 4. Meta-prompting ---")
    meta_instruction = "You are a senior Prompt Engineer. Help me optimize the following system prompt to be more rigorous and effective."
    raw_prompt = "You are a mobile plan support agent named Melon. Help users pick plans ($10 for 10GB, $30 for 100GB)."
    prompt = compose_prompt(meta_instruction, raw_prompt)
    print(get_completion(prompt, temperature=0.7))
