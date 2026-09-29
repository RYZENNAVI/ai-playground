"""Supervised fine-tuning (SFT) with LoRA: teach a base model a fixed answer line.

Every example is rendered in the Alpaca instruction template, and only the answer
tokens carry a label. PEFT attaches rank-8 adapters to the four attention
projections, so 0.12% of the weights train. The labels come from a three-branch
rule on age and claims, so every answer can be marked right or wrong.
vehicle_value takes no part in the rule.

The run prints six parts:
    1. The data. 540 applications, split by (age, claims) pair so that no pair
       in the 60 evaluation cases appears in training.
    2. One example rendered and masked, with the count of supervised tokens.
    3. The base model scored twice: on the instruction alone, and with the rule
       written into the prompt. Always answering the most common tier is printed
       as a baseline.
    4. The adapter attached, and the share of the model it trains.
    5. Training for 120 steps, then the same evaluation cases scored again.
    6. The adapter saved, reloaded onto a fresh base and merged into the
       weights, each checked on the first four evaluation cases.
"""

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch

# Print UTF-8 even when the output is piped or redirected on Windows.
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
VEHICLE_VALUES = (8000, 15000, 30000, 60000)
# Sixty evaluation cases: 15 (age, claims) pairs, each with the four vehicle values.
EVAL_CASES = 60
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

# The same rule classify() applies, written out. Part 3 scores the base model with
# it, because a rule this small fits in a prompt and prompting is cheaper to try.
RULE_HINT = (
    " Use exactly this rule: if age < 25 or claims >= 3 then TIER C and ACTION "
    "decline; else if claims == 0 and age >= 30 then TIER A and ACTION accept; "
    "otherwise TIER B and ACTION refer."
)

PROMPTED_INSTRUCTION = INSTRUCTION + RULE_HINT

ACTIONS = {"A": "accept", "B": "refer", "C": "decline"}


def classify(age, claims):
    """The rule the model has to absorb. It gives every input exactly one correct
    answer, so parts 3 and 5 can mark each answer right or wrong."""
    if age < 25 or claims >= 3:
        tier = "C"
    elif claims == 0 and age >= 30:
        tier = "A"
    else:
        tier = "B"
    return tier, ACTIONS[tier]


def build_dataset(seed, eval_cases, path):
    """Part 1. Enumerate every application and hold out whole (age, claims) pairs.
    The rule reads only age and claims, so a split by input would leak every pair."""
    pairs = [(age, claims) for age in range(18, 71, 2) for claims in range(0, 5)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    held_out = set(pairs[:eval_cases // len(VEHICLE_VALUES)])

    records = []
    for age in range(18, 71, 2):
        for claims in range(0, 5):
            for value in VEHICLE_VALUES:
                tier, action = classify(age, claims)
                records.append(((age, claims), {
                    "input": f"age={age}; claims={claims}; vehicle_value={value}",
                    "output": f"TIER: {tier} | ACTION: {action}",
                    "tier": tier,
                }))
    rng.shuffle(records)

    evaluation = [record for pair, record in records if pair in held_out]
    training = [record for pair, record in records if pair not in held_out]
    # Read the pairs back from the rendered inputs, so the check does not trust the split.
    def pair_of(record):
        return tuple(part.split("=")[1] for part in record["input"].split("; ")[:2])

    overlap = {pair_of(r) for r in evaluation} & {pair_of(r) for r in training}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in training:
            handle.write(json.dumps(record) + "\n")

    def distribution(rows):
        return {tier: sum(1 for r in rows if r["tier"] == tier) for tier in "ABC"}

    print(f"Training examples: {len(training)}, evaluation examples: {len(evaluation)}")
    print(f"Tier distribution in training: {distribution(training)}")
    print(f"Tier distribution in evaluation: {distribution(evaluation)}")
    print(f"Evaluation (age, claims) pairs also in training: "
          f"{len(overlap)} of {len(held_out)}")
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
    """Part 2. Render one example, append the end token so generation learns to stop,
    and give the prompt tokens the ignore label so only the answer is learned."""
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
    """Part 4. Wrap the chosen projections in LoRA adapters and freeze the rest.
    Frozen weights keep no gradients and no optimiser state, which is what saves memory."""
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


def generate(model, tokenizer, records, device, max_new_tokens,
             instruction=INSTRUCTION):
    """Answer every input with greedy decoding, so the same model and input always give
    the same line. Part 3 also passes PROMPTED_INSTRUCTION."""
    model.eval()
    answers = []
    for record in records:
        prompt = TEMPLATE.format(instruction=instruction, input=record["input"])
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
    """Parts 3 and 5. Count answers in the required shape and exactly correct answers
    separately, since a right tier inside prose and a clean line with a wrong tier both fail."""
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
    # An aggregate score could mean small errors everywhere or one whole branch
    # of the rule missed, and those two call for different fixes.
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
    """Part 5. Run the optimiser over sampled batches and report loss and cost."""
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
    """Part 6. Save the adapter, reload it onto a fresh base, then merge it into the
    weights. After merging there is no adapter left to swap for another one."""
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
    print(f"\nReloaded adapter reproduces the trained answers "
          f"on the first four evaluation cases: {identical}")
    if not identical:
        for trained, restored in zip(trained_answers, reloaded_answers):
            print(f"  trained  {trained!r}")
            print(f"  reloaded {restored!r}")

    merged = reloaded.merge_and_unload()
    merged_answers = generate(merged, tokenizer, records[:4], device, max_new_tokens)
    print(f"Merged model reproduces the same answers on those four cases: "
          f"{merged_answers == reloaded_answers}")
    print(f"Adapter modules left after merging: "
          f"{sum(1 for name, _ in merged.named_modules() if 'lora' in name)}")


def main():
    # Adapter dropout draws from the global torch generator, so without this the
    # same script prints a different accuracy on every run.
    torch.manual_seed(SEED)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("--- 1. The data ---")
    training, evaluation = build_dataset(SEED, EVAL_CASES, DATA_FILE)

    base, tokenizer, device = load_base(MODEL_ID, CACHE_DIR, dtype)
    print(f"\n[Device] {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    print(f"[Model] {MODEL_ID}, dtype={base.dtype}")

    print("\n--- 2. One example rendered and masked ---")
    report_masking(tokenizer, training[0], MAX_LENGTH)

    print("\n--- 3. The base model, with and without the rule ---")
    base_answers = generate(base, tokenizer, evaluation, device, MAX_NEW_TOKENS)
    base_schema, base_exact = score(evaluation, base_answers,
                                    "Base model, instruction only:")
    prompted_answers = generate(base, tokenizer, evaluation, device, MAX_NEW_TOKENS,
                                instruction=PROMPTED_INSTRUCTION)
    prompted_schema, prompted_exact = score(evaluation, prompted_answers,
                                            "Base model, rule written into the prompt:")
    counts = {tier: sum(1 for r in evaluation if r["tier"] == tier) for tier in "ABC"}
    majority = max(counts, key=counts.get)
    majority_share = counts[majority] / len(evaluation)
    prompted_tiers = {tier: sum(1 for answer in prompted_answers
                                if answer.startswith(f"TIER: {tier}"))
                      for tier in "ABC"}

    print("\n--- 4. The adapter ---")
    model = attach_adapter(base, RANK, ALPHA, DROPOUT, TARGET_MODULES)

    print("\n--- 5. Train, then score the same evaluation cases ---")
    train(model, tokenizer, training, device, STEPS, BATCH_SIZE, LEARNING_RATE,
          MAX_LENGTH, SEED)
    tuned_answers = generate(model, tokenizer, evaluation, device, MAX_NEW_TOKENS)
    tuned_schema, tuned_exact = score(evaluation, tuned_answers, "Adapted model, after training:")

    print(f"\n{'setting':>40} {'schema':>8} {'exact':>8}")
    print(f"{f'always TIER {majority}':>40} {'':>8} {majority_share:>7.1%}")
    print(f"{'base, instruction only':>40} {base_schema:>7.1%} {base_exact:>7.1%}")
    print(f"{'base, rule written into the prompt':>40} "
          f"{prompted_schema:>7.1%} {prompted_exact:>7.1%}")
    print(f"{'adapter, rule learned from examples':>40} "
          f"{tuned_schema:>7.1%} {tuned_exact:>7.1%}")
    print(f"With the rule in the prompt, the base model answered TIER A {prompted_tiers['A']}, "
          f"B {prompted_tiers['B']} and C {prompted_tiers['C']} times.")
    print(f"Always answering TIER {majority} scores {majority_share:.1%}, so compare "
          f"each row with that line, not with zero.")
    if prompted_tiers[majority] == len(evaluation):
        print(f"Written into the prompt, the rule turned every answer into TIER {majority},")
        print(f"which scores the same as always answering {majority}.")
    if tuned_exact > majority_share and prompted_exact <= majority_share:
        print("The adapter is the only setting that uses the rule.")

    print("\n--- 6. Save, reload, and merge the adapter ---")
    save_reload_merge(model, tokenizer, evaluation, device, ADAPTER_DIR, dtype,
                      MAX_NEW_TOKENS)


if __name__ == "__main__":
    main()
