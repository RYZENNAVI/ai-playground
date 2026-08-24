"""Teach a base model a fixed answer schema with a low-rank adapter and score the result.

Demonstrates supervised fine-tuning end to end on one machine:
    1. Generate a labelled set from deterministic rules, so answers can be marked.
    2. Wrap every example in an instruction template and terminate it properly.
    3. Mask the prompt tokens out of the loss and count what remains supervised.
    4. Attach a low-rank adapter and report how little of the model is trainable.
    5. Score the untouched base model, which supplies the before number.
    6. Train, then score again on inputs the model never saw during training.
    7. Save the adapter, reload it onto a fresh base, and merge it into the weights.

Module 05: Fine-Tuning - Supervised Fine-Tuning with a Low-Rank Adapter.
"""

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
SIBLING_CACHE = Path(__file__).parent.parent / "01-llm-foundation" / "weights"
CACHE_DIR = os.getenv("HF_CACHE_DIR", str(SIBLING_CACHE))
ADAPTER_DIR = Path(__file__).parent / "outputs" / "sft_adapter"
DATA_FILE = Path(__file__).parent / "data" / "underwriting_triage.jsonl"

RANK = 8
ALPHA = 16
DROPOUT = 0.05
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
STEPS = 120
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 128
MAX_NEW_TOKENS = 24
EVAL_CASES = 24
SEED = 3407

INSTRUCTION = (
    "Classify the policy application into a tier and an action. "
    "Answer with one line in the form: TIER: <A|B|C> | ACTION: <accept|refer|decline>"
)

TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"
    "### Response:\n"
)

ACTIONS = {"A": "accept", "B": "refer", "C": "decline"}


def classify(age, claims):
    """The rule the model has to absorb.

    The labels come from a rule rather than from a person, which is what makes
    the evaluation in steps 5 and 6 a measurement instead of an impression: every
    held-out input has exactly one correct answer, and a wrong answer is wrong
    without argument. A dataset scraped from free text could not be scored this
    way, and "the answers look better" would be the only available verdict.
    """
    if age < 25 or claims >= 3:
        tier = "C"
    elif claims == 0 and age >= 30:
        tier = "A"
    else:
        tier = "B"
    return tier, ACTIONS[tier]


def build_dataset(seed, eval_cases, path):
    """Step 1. Enumerate every combination, then split so evaluation inputs are unseen.

    The split is over inputs, not over rendered strings. If an evaluation input
    also appeared in training, a high score would only prove the model memorised
    it, and the whole comparison would say nothing about whether the rule was
    learned.
    """
    combinations = [
        (age, claims, value)
        for age in range(18, 71, 2)
        for claims in range(0, 5)
        for value in (8000, 15000, 30000, 60000)
    ]
    rng = random.Random(seed)
    rng.shuffle(combinations)

    records = []
    for age, claims, value in combinations:
        tier, action = classify(age, claims)
        records.append({
            "input": f"age={age}; claims={claims}; vehicle_value={value}",
            "output": f"TIER: {tier} | ACTION: {action}",
            "tier": tier,
        })

    evaluation = records[:eval_cases]
    training = records[eval_cases:]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in training:
            handle.write(json.dumps(record) + "\n")

    distribution = {tier: sum(1 for r in training if r["tier"] == tier)
                    for tier in "ABC"}
    print(f"Training examples: {len(training)}, evaluation examples: {len(evaluation)}")
    print(f"Tier distribution in training: {distribution}")
    print(f"Wrote the training split to {path}")
    print("\nTwo training examples as the model sees them:")
    for record in training[:2]:
        print(f"  input : {record['input']}")
        print(f"  output: {record['output']}")
    return training, evaluation


def load_base(model_id, cache_dir, dtype):
    """Load tokenizer and base weights, reusing whatever is already cached."""
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
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def encode(tokenizer, record, max_length):
    """Step 2-3. Render one example and mask everything the model should not learn.

    Two details decide whether this works at all. The end-of-sequence token has
    to be appended to the answer, or nothing ever tells the model to stop and
    generation runs until it hits the token limit. And the prompt tokens have to
    carry the ignore label, or the model spends its capacity learning to write
    the questions back out instead of answering them.
    """
    prompt = TEMPLATE.format(instruction=INSTRUCTION, input=record["input"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(record["output"], add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return input_ids, labels


def report_masking(tokenizer, record, max_length):
    """Print the token accounting for a single example."""
    input_ids, labels = encode(tokenizer, record, max_length)
    supervised = sum(1 for label in labels if label != -100)
    print(f"Tokens in the rendered example: {len(input_ids)}")
    print(f"Tokens carrying a label: {supervised} "
          f"({supervised / len(input_ids):.1%} of the sequence)")
    print(f"Supervised text: {tokenizer.decode([l for l in labels if l != -100])!r}")


def collate(tokenizer, batch, max_length, device):
    """Pad a batch to a common length and build the attention mask."""
    encoded = [encode(tokenizer, record, max_length) for record in batch]
    width = max(len(ids) for ids, _ in encoded)
    input_ids, labels, attention = [], [], []
    for ids, label in encoded:
        padding = width - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * padding)
        labels.append(label + [-100] * padding)
        attention.append([1] * len(ids) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "labels": torch.tensor(labels, device=device),
        "attention_mask": torch.tensor(attention, device=device),
    }


def attach_adapter(model, rank, alpha, dropout, target_modules):
    """Step 4. Wrap the chosen projections in low-rank adapters and freeze the rest.

    `print_trainable_parameters` is worth reading rather than skipping: it is the
    number that explains why this fits on one consumer card. The frozen weights
    still need memory for the forward pass, but no gradients and no optimiser
    state are kept for them, and those two are what usually exhaust the card.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapted = get_peft_model(model, config)
    print(f"Rank {rank}, alpha {alpha}, dropout {dropout}")
    print(f"Attached to: {', '.join(target_modules)}")
    adapted.print_trainable_parameters()
    return adapted


def generate(model, tokenizer, records, device, max_new_tokens):
    """Answer every evaluation input with greedy decoding."""
    model.eval()
    answers = []
    for record in records:
        prompt = TEMPLATE.format(instruction=INSTRUCTION, input=record["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = output[0][inputs["input_ids"].shape[1]:]
        answers.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return answers


def score(records, answers, label, show=3):
    """Step 5-6. Mark the answers on two separate counts.

    Schema compliance and correctness are tracked apart on purpose. A model can
    produce the right tier inside a paragraph of prose, which is useless to a
    caller parsing the line, and it can produce a perfectly shaped line with the
    wrong tier in it. Collapsing both into a single accuracy number would hide
    whichever of the two failed.
    """
    exact = 0
    schema = 0
    per_tier = {tier: [0, 0] for tier in "ABC"}
    for record, answer in zip(records, answers):
        first_line = answer.splitlines()[0].strip() if answer else ""
        expected = record["output"]
        shaped = first_line.startswith("TIER: ") and " | ACTION: " in first_line
        schema += int(shaped)
        correct = int(first_line == expected)
        exact += correct
        per_tier[record["tier"]][0] += correct
        per_tier[record["tier"]][1] += 1

    print(f"\n{label}")
    print(f"  answers in the required shape: {schema}/{len(records)} "
          f"({schema / len(records):.1%})")
    print(f"  answers exactly correct:       {exact}/{len(records)} "
          f"({exact / len(records):.1%})")
    # Which branch of the rule was learned, and which was not. An aggregate score
    # of eighty-something percent could mean small errors everywhere or one whole
    # branch missed, and those two call for different fixes.
    breakdown = "  ".join(
        f"{tier}: {hits}/{total}" for tier, (hits, total) in per_tier.items() if total)
    print(f"  correct by expected tier:      {breakdown}")
    for record, answer in list(zip(records, answers))[:show]:
        collapsed = answer.replace("\n", " ")[:90]
        print(f"    input    {record['input']}")
        print(f"    expected {record['output']}")
        print(f"    produced {collapsed!r}")
    return schema / len(records), exact / len(records)


def train(model, tokenizer, records, device, steps, batch_size, learning_rate,
          max_length, seed):
    """Step 6. Run the optimiser over sampled batches and report loss and cost."""
    rng = random.Random(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()

    started = time.time()
    losses = []
    for step in range(1, steps + 1):
        batch = collate(tokenizer, rng.sample(records, batch_size), max_length, device)
        optimiser.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == 1:
            window = losses[-20:]
            print(f"  step {step:>3d}  loss {loss.item():.4f}  "
                  f"mean of last {len(window)} {sum(window) / len(window):.4f}")

    elapsed = time.time() - started
    print(f"Loss: {losses[0]:.4f} -> {sum(losses[-10:]) / 10:.4f} "
          f"(mean of the last ten steps)")
    print(f"Wall clock: {elapsed:.1f} s for {steps} steps "
          f"({elapsed / steps * 1000:.0f} ms per step)")
    if torch.cuda.is_available():
        print(f"Peak VRAM reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


def directory_size(path):
    """Total bytes of every file under a directory."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def save_reload_merge(model, tokenizer, records, device, adapter_dir, dtype,
                      max_new_tokens):
    """Step 7. Save the adapter, put it back on a clean base, then fold it in.

    Three separate facts get checked here. The saved directory holds only the
    adapter, which is why it is measured in megabytes next to a multi-gigabyte
    checkpoint. Reloading it onto a freshly loaded base has to reproduce the same
    answers, otherwise the saved artefact is not the thing that was trained. And
    merging writes the adapter into the frozen matrices, after which the model is
    an ordinary model again with no adapter left to load - convenient to deploy,
    but no longer swappable for a different adapter.
    """
    from peft import PeftModel

    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    model.save_pretrained(adapter_dir)
    files = sorted(file.name for file in adapter_dir.iterdir())
    print(f"Adapter directory: {adapter_dir}")
    print(f"  files: {', '.join(files)}")
    print(f"  size: {directory_size(adapter_dir) / 1e6:.2f} MB")

    trained_answers = generate(model, tokenizer, records[:4], device, max_new_tokens)

    fresh_base, _, _ = load_base(MODEL_ID, CACHE_DIR, dtype)
    reloaded = PeftModel.from_pretrained(fresh_base, adapter_dir)
    reloaded_answers = generate(reloaded, tokenizer, records[:4], device, max_new_tokens)
    identical = trained_answers == reloaded_answers
    print(f"\nReloaded adapter reproduces the trained answers: {identical}")
    if not identical:
        for trained, restored in zip(trained_answers, reloaded_answers):
            print(f"  trained  {trained!r}")
            print(f"  reloaded {restored!r}")

    merged = reloaded.merge_and_unload()
    merged_answers = generate(merged, tokenizer, records[:4], device, max_new_tokens)
    print(f"Merged model reproduces the same answers: "
          f"{merged_answers == reloaded_answers}")
    print(f"Adapter modules left after merging: "
          f"{sum(1 for name, _ in merged.named_modules() if 'lora' in name)}")


def main():
    # Adapter dropout draws from the global torch generator, so without this the
    # same script prints a different accuracy on every run and no number in the
    # output can be quoted.
    torch.manual_seed(SEED)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("--- 1. Generate a labelled set from deterministic rules ---")
    training, evaluation = build_dataset(SEED, EVAL_CASES, DATA_FILE)

    base, tokenizer, device = load_base(MODEL_ID, CACHE_DIR, dtype)
    print(f"\n[Device] {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"[Model] {MODEL_ID}, dtype={base.dtype}")

    print("\n--- 2-3. Render one example and mask the prompt out of the loss ---")
    report_masking(tokenizer, training[0], MAX_LENGTH)

    print("\n--- 5. Score the untouched base model first ---")
    base_answers = generate(base, tokenizer, evaluation, device, MAX_NEW_TOKENS)
    base_schema, base_exact = score(evaluation, base_answers, "Base model, before training:")

    print("\n--- 4. Attach the adapter ---")
    model = attach_adapter(base, RANK, ALPHA, DROPOUT, TARGET_MODULES)

    print("\n--- 6. Train, then score on the held-out inputs ---")
    train(model, tokenizer, training, device, STEPS, BATCH_SIZE, LEARNING_RATE,
          MAX_LENGTH, SEED)
    tuned_answers = generate(model, tokenizer, evaluation, device, MAX_NEW_TOKENS)
    tuned_schema, tuned_exact = score(evaluation, tuned_answers, "Adapted model, after training:")

    print(f"\nSchema compliance {base_schema:.1%} -> {tuned_schema:.1%}")
    print(f"Exact correctness {base_exact:.1%} -> {tuned_exact:.1%}")

    print("\n--- 7. Save, reload, and merge the adapter ---")
    save_reload_merge(model, tokenizer, evaluation, device, ADAPTER_DIR, dtype,
                      MAX_NEW_TOKENS)


if __name__ == "__main__":
    main()
