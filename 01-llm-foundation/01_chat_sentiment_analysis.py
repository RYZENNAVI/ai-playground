"""Chat completion & sentiment analysis using standard OpenAI SDK (DeepSeek / OpenAI).

Demonstrates universal chat completion:
    - Environment-driven API key & base URL configuration (Primary: DeepSeek, Fallback: OpenAI).
    - Standard system / user / assistant message protocol.
    - Zero-shot NLP classification (sentiment analysis) via prompt engineering.
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
    raise RuntimeError("No API key found! Please set DEEPSEEK_API_KEY or OPENAI_API_KEY in environment or .env file.")

if os.getenv("DEEPSEEK_API_KEY"):
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-chat"
else:
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"

base_url = os.getenv("OPENAI_BASE_URL", default_base_url)
client = OpenAI(api_key=api_key, base_url=base_url)


def chat(user_prompt: str, system_prompt: str = "You are a helpful assistant", model: str = default_model) -> str:
    """One-shot chat completion using the universal OpenAI SDK."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def sentiment_analysis(review: str, model: str = default_model) -> str:
    """Classify a product review as positive, negative, or neutral using a system prompt."""
    messages = [
        {
            "role": "system",
            "content": "You are a professional sentiment analyst. Classify the sentiment of the product review. Reply with only one word: Positive, Negative, or Neutral.",
        },
        {"role": "user", "content": review},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,  # Low temperature for deterministic classification
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print(f"=== Universal OpenAI SDK Demo ===")
    print(f"Using Endpoint: {base_url}")
    print(f"Default Model : {default_model}\n")

    print("--- 1. Basic Chat ---")
    print(chat("Hello, please introduce yourself in one sentence."))

    print("\n--- 2. Sentiment Analysis ---")
    reviews = [
        "The audio quality of this speaker is amazing, giving you unexpected sound!",
        "Battery life is terrible, dies in half a day. Strongly not recommended.",
        "Secure packaging, fast shipping, quality is average.",
    ]
    for r in reviews:
        print(f"  Review: {r!r} -> Result: {sentiment_analysis(r)}")
