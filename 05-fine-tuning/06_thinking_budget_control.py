"""Thinking budget control: cap or extend how long a reasoning model thinks, with no training.

This is test-time compute control. The script decodes one token at a time, so it can act in
the middle of the thinking phase. To cap thinking, it writes </think> into the stream once
the budget is spent. To extend it, it bans </think> and appends a nudge, "Wait, let me check
that again.", each time the model is about to stop.

The run prints four parts:
    1. The ids of <think> and </think>. Each is one token, so </think> can be banned in the logits.
    2. Eight questions with one numeric answer, under caps of 24, 64, 160 and 400 thinking
       tokens and under the 400 cap with two nudges: accuracy, mean thinking tokens, and how
       many answers stopped on their own.
    3. The first question (raspberry) under every setting.
    4. Accuracy and thinking tokens from the tightest cap to the widest, a summary of part 2.
"""

import os
import re
import sys
import time
from pathlib import Path

import torch

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
SIBLING_CACHE = Path(__file__).parent.parent / "01-llm-foundation" / "weights"
CACHE_DIR = os.getenv("HF_CACHE_DIR", str(SIBLING_CACHE))

OPEN_DELIMITER = "<think>"
CLOSE_DELIMITER = "</think>"
NUDGE = "\nWait, let me check that again.\n"
CLOSING_HINT = "\nFinal answer: "
BUDGETS = (24, 64, 160, 400)
EXTENSIONS = 2
ANSWER_TOKENS = 48
TEMPERATURE = 0.0

# Questions whose answers are decidable, so accuracy is counted rather than judged.
QUESTIONS = (
    {"prompt": "How many times does the letter r appear in the word raspberry? "
               "Reply with just the number.", "answer": "3"},
    {"prompt": "How many times does the letter s appear in the word possessions? "
               "Reply with just the number.", "answer": "5"},
    {"prompt": "How many times does the letter a appear in the word banana? "
               "Reply with just the number.", "answer": "3"},
    {"prompt": "A tray holds 24 cups. Nine are removed and then six are added back. "
               "How many cups are on the tray? Reply with just the number.",
     "answer": "21"},
    {"prompt": "A shelf has 7 rows of 8 tins. Five tins are sold. How many tins "
               "remain? Reply with just the number.", "answer": "51"},
    {"prompt": "What is 17 times 13 minus 20? Reply with just the number.",
     "answer": "201"},
    {"prompt": "How many times does the letter e appear in the word beekeeper? "
               "Reply with just the number.", "answer": "5"},
    {"prompt": "A box weighs 4 kilograms empty and holds 12 bags of 3 kilograms "
               "each. What is the total weight in kilograms? Reply with just the "
               "number.", "answer": "40"},
)


def load_model(model_id, cache_dir):
    """Load the checkpoint and tokenizer, reusing whatever is already on disk."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer, device


def find_delimiters(tokenizer):
    """Resolve both thinking delimiters to token ids.

    A single id can be banned in the logits; if </think> spans several tokens, the script bans the first.
    """
    ids = {}
    for name in (OPEN_DELIMITER, CLOSE_DELIMITER):
        encoded = tokenizer(name, add_special_tokens=False)["input_ids"]
        ids[name] = encoded
        print(f"  {name!r} -> {len(encoded)} token(s), ids {encoded}")
    close_ids = ids[CLOSE_DELIMITER]
    if len(close_ids) != 1:
        print("  The closing delimiter is not a single token; the first id will be")
        print("  the one banned when thinking is extended.")
    return ids[OPEN_DELIMITER], close_ids


def build_prefix(tokenizer, prompt, device):
    """Render the question so the thinking phase starts right after the prompt.

    This template already ends with <think>; the delimiter is added by hand only if it did not.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True)
    if not text.rstrip().endswith(OPEN_DELIMITER):
        text = text + OPEN_DELIMITER + "\n"
    return tokenizer(text, return_tensors="pt").to(device)


def pick_token(logits, temperature):
    """Choose the next token, greedily when the temperature is zero."""
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / temperature
    probabilities = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def run_with_budget(model, tokenizer, prompt, device, budget, extensions,
                    close_ids, answer_tokens, temperature):
    """Decode one token at a time, banning </think> while nudges remain and forcing it at the budget.

    A single generate() call would finish the sequence before either intervention could apply.
    """
    inputs = build_prefix(tokenizer, prompt, device)
    ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=ids, use_cache=True)
    cache = outputs.past_key_values
    logits = outputs.logits[0, -1]

    thinking_tokens = 0
    nudges_used = 0
    thinking_pieces = []
    stopped_on_its_own = False

    while thinking_tokens < budget:
        if extensions - nudges_used > 0:
            logits = logits.clone()
            logits[close_ids[0]] = float("-inf")
        token = pick_token(logits, temperature)

        if token == close_ids[0]:
            stopped_on_its_own = True
            break
        if token == tokenizer.eos_token_id:
            stopped_on_its_own = True
            break

        thinking_pieces.append(token)
        thinking_tokens += 1
        with torch.no_grad():
            outputs = model(input_ids=torch.tensor([[token]], device=device),
                            past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1]

        wants_to_stop = int(torch.argmax(logits).item()) == close_ids[0]
        if wants_to_stop and nudges_used < extensions:
            nudge_ids = tokenizer(NUDGE, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                outputs = model(
                    input_ids=torch.tensor([nudge_ids], device=device),
                    past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[0, -1]
            thinking_pieces.extend(nudge_ids)
            thinking_tokens += len(nudge_ids)
            nudges_used += 1

    # Close the thinking phase by hand and let the model produce its answer.
    forced = close_ids + tokenizer(CLOSING_HINT, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([forced], device=device),
                        past_key_values=cache, use_cache=True)
    cache = outputs.past_key_values
    logits = outputs.logits[0, -1]

    answer_pieces = []
    for _ in range(answer_tokens):
        token = pick_token(logits, temperature)
        if token == tokenizer.eos_token_id:
            break
        answer_pieces.append(token)
        with torch.no_grad():
            outputs = model(input_ids=torch.tensor([[token]], device=device),
                            past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1]

    return {
        "thinking_tokens": thinking_tokens,
        "nudges": nudges_used,
        "stopped_on_its_own": stopped_on_its_own,
        "thinking": tokenizer.decode(thinking_pieces, skip_special_tokens=True),
        "answer": tokenizer.decode(answer_pieces, skip_special_tokens=True),
    }


def first_number(text):
    """Read the integer in \\boxed{} if there is one, otherwise the first integer.

    The model often writes the working first, as in 4 kilograms + 36 kilograms = \\boxed{40}.
    """
    boxed = re.findall(r"\\boxed\{(-?\d+)\}", text)
    if boxed:
        return boxed[0]
    found = re.findall(r"-?\d+", text)
    return found[0] if found else None


def sweep(model, tokenizer, questions, device, budgets, extensions, close_ids,
          answer_tokens, temperature):
    """Answer every question under every setting and print one row per setting."""
    settings = [(f"cap {budget}", budget, 0) for budget in budgets]
    settings.append((f"cap {budgets[-1]} + {extensions} nudges", budgets[-1], extensions))

    print(f"\n{'setting':>26} {'accuracy':>9} {'mean thinking tokens':>21} "
          f"{'stopped on its own':>19} {'seconds':>8}")
    results = {}
    for label, budget, nudge_count in settings:
        started = time.time()
        correct = 0
        tokens = 0
        self_stopped = 0
        details = []
        for question in questions:
            outcome = run_with_budget(model, tokenizer, question["prompt"], device,
                                      budget, nudge_count, close_ids, answer_tokens,
                                      temperature)
            predicted = first_number(outcome["answer"])
            hit = predicted == question["answer"]
            correct += int(hit)
            tokens += outcome["thinking_tokens"]
            self_stopped += int(outcome["stopped_on_its_own"])
            details.append((question, outcome, predicted, hit))
        elapsed = time.time() - started
        results[label] = {
            "accuracy": correct / len(questions),
            "tokens": tokens / len(questions),
            "details": details,
        }
        print(f"{label:>26} {correct}/{len(questions):<7} "
              f"{tokens / len(questions):>21.1f} "
              f"{self_stopped:>13}/{len(questions):<5} {elapsed:>8.1f}")
    return results


def show_one_question(results, index=0):
    """Print one question under every setting."""
    for label, payload in results.items():
        question, outcome, predicted, hit = payload["details"][index]
        thinking = " ".join(outcome["thinking"].split())
        print(f"\n  {label}")
        print(f"    thinking tokens spent: {outcome['thinking_tokens']}, "
              f"nudges: {outcome['nudges']}, "
              f"stopped on its own: {outcome['stopped_on_its_own']}")
        print(f"    thinking (first 200 chars): {thinking[:200]!r}")
        print(f"    answer: {' '.join(outcome['answer'].split())[:80]!r}")
        print(f"    expected {question['answer']}, read {predicted}, correct {hit}")


def main():
    torch.manual_seed(3407)
    model, tokenizer, device = load_model(MODEL_ID, CACHE_DIR)
    print(f"[Device] {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"[Model] {MODEL_ID}, dtype={model.dtype}")
    print("\nNo weight changes in this script; only the decoding loop does.")

    print("\n--- 1. Locate the thinking delimiters ---")
    _, close_ids = find_delimiters(tokenizer)

    print("\n--- 2. Answer every question under each thinking budget ---")
    print(f"Questions: {len(QUESTIONS)}, all with a single decidable numeric answer.")
    results = sweep(model, tokenizer, QUESTIONS, device, BUDGETS, EXTENSIONS,
                    close_ids, ANSWER_TOKENS, TEMPERATURE)

    print("\n--- 3. The first question under each setting ---")
    show_one_question(results)

    print("\n--- 4. From the tightest cap to the widest ---")
    labels = list(results)
    tightest, widest = results[labels[0]], results[labels[-2]]
    print(f"Accuracy "
          f"{tightest['accuracy']:.1%} -> {widest['accuracy']:.1%}, "
          f"mean thinking tokens {tightest['tokens']:.1f} -> {widest['tokens']:.1f}")
    if torch.cuda.is_available():
        print(f"Peak VRAM reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
