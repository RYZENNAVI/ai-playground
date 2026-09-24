"""This script sends two kinds of request to a chat model. The first is an ordinary
question. The second asks the model to label a product review as positive, negative
or neutral. Both use the same API call but different system messages.

The script uses DeepSeek when DEEPSEEK_API_KEY is set, and OpenAI otherwise. The run
prints two parts:
    1. A plain question and the model's reply.
    2. Three product reviews and the label the model gives each one. The labelling rule
       is in the system message, and the temperature is 0.1, so a review gets the same
       label almost every time.
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


# 1. Plain question

def chat(user_prompt: str, system_prompt: str = "You are a helpful assistant", model: str = default_model) -> str:
    """Send one system message and one user message, and return the reply."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


# 2. Sentiment labels

def sentiment_analysis(review: str, model: str = default_model) -> str:
    """Ask the model to label a product review as positive, negative or neutral."""
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
        temperature=0.1,  # low, so a review usually gets the same label on every run
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("=== Chat and sentiment classification ===")
    print(f"Using Endpoint: {base_url}")
    print(f"Default Model : {default_model}\n")

    print("--- 1. Plain question ---")
    print(chat("Hello, please introduce yourself in one sentence."))

    print("\n--- 2. Sentiment labels ---")
    reviews = [
        "This speaker sounds far better than I expected.",
        "The battery dies within half a day. I would not buy it again.",
        "It arrived quickly and was well packed, but the quality is only average.",
    ]
    for r in reviews:
        print(f"  Review: {r!r} -> Label: {sentiment_analysis(r)}")
