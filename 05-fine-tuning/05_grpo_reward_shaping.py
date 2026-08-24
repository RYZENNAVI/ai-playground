"""Train on scored samples instead of written answers, using group-relative policy optimisation.

Demonstrates reinforcement learning from rules rather than from labelled outputs:
    1. Generate arithmetic problems whose answers can be checked automatically.
    2. Define five reward functions, from tag counting up to answer correctness.
    3. Sample a group of answers per problem so the group can grade itself.
    4. Turn raw rewards into advantages by centring them inside each group.
    5. Hold the policy near the frozen base with a penalty on the log ratio.
    6. Update the adapter from sampled text, and watch each reward term move.
    7. Score format compliance and accuracy before and after on unseen problems.

Module 05: Fine-Tuning - Group Relative Policy Optimisation.
"""

import os
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
SIBLING_CACHE = Path(__file__).parent.parent / "01-llm-foundation" / "weights"
CACHE_DIR = os.getenv("HF_CACHE_DIR", str(SIBLING_CACHE))

RANK = 16
ALPHA = 32
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
GROUP_SIZE = 6
PROMPTS_PER_STEP = 2
STEPS = 24
LEARNING_RATE = 1e-5
KL_COEFFICIENT = 0.02
MAX_NEW_TOKENS = 160
CHUNK_SIZE = 2
TEMPERATURE = 0.9
TOP_P = 0.95
EVAL_PROBLEMS = 12
SEED = 3407

PREFILL = "<reasoning>\n"

SYSTEM_PROMPT = (
    "Answer in exactly this format and nothing else:\n"
    "<reasoning>\none or two short sentences\n</reasoning>\n"
    "<answer>\na single integer\n</answer>"
)

STRICT_PATTERN = re.compile(
    r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>", re.DOTALL)
SOFT_PATTERN = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)


def make_problems(count, seed):
    """Step 1. Build problems from arithmetic that is verifiable by construction.

    The answer is computed here, not written by a person, so the correctness
    reward is a fact rather than a judgement. This is what separates a reward
    that can be optimised from one that only looks like it can: a scorer that
    needs an opinion cannot be run thousands of times inside a training loop.
    """
    rng = random.Random(seed)
    problems = []
    while len(problems) < count:
        total = rng.randrange(120, 400, 10)
        first = rng.randrange(10, 60)
        second = rng.randrange(10, 60)
        third = rng.randrange(10, 60)
        if first + second + third >= total:
            continue
        question = (
            f"A jar holds {total} beads in four colours. "
            f"There are {first} blue, {second} purple and {third} orange beads. "
            "How many red beads are there?"
        )
        problems.append({"question": question, "answer": total - first - second - third})
    return problems


def extract_answer(text):
    """Pull the integer out of the answer tags, or return None."""
    if "<answer>" not in text:
        return None
    body = text.split("<answer>")[-1].split("</answer>")[0]
    found = re.findall(r"-?\d+", body)
    return found[0] if found else None


def reward_tag_count(completion, expected):
    """Partial credit for each tag that appears exactly once.

    This is the only reward that pays out for a half-formed answer, and it exists
    to keep the early steps from being flat. If every reward were all-or-nothing,
    a model that never once produced the full shape would receive an identical
    score for every sample, the advantages inside the group would all be zero,
    and there would be no gradient to learn from at all.
    """
    score = 0.0
    for tag in ("<reasoning>", "</reasoning>", "<answer>", "</answer>"):
        if completion.count(tag) == 1:
            score += 0.125
    return score


def reward_soft_format(completion, expected):
    """Credit for having the tags in the right order, whitespace ignored."""
    return 0.5 if SOFT_PATTERN.search(completion) else 0.0


def reward_strict_format(completion, expected):
    """Credit for the exact line layout, newlines included."""
    return 0.5 if STRICT_PATTERN.match(completion.strip()) else 0.0


def reward_integer(completion, expected):
    """Credit for putting a bare integer inside the answer tags."""
    answer = extract_answer(completion)
    return 0.5 if answer is not None and answer.lstrip("-").isdigit() else 0.0


def reward_correct(completion, expected):
    """The reward that actually matters, and the largest one."""
    answer = extract_answer(completion)
    return 2.0 if answer is not None and answer == str(expected) else 0.0


REWARDS = (
    ("tags", reward_tag_count),
    ("soft", reward_soft_format),
    ("strict", reward_strict_format),
    ("integer", reward_integer),
    ("correct", reward_correct),
)


def score_completion(completion, expected):
    """Return the total reward and the contribution of each term."""
    parts = {name: function(completion, expected) for name, function in REWARDS}
    return sum(parts.values()), parts


def load_policy(model_id, cache_dir, rank, alpha, target_modules):
    """Load the base model and attach the adapter that will be trained.

    The frozen base is not loaded a second time to act as the reference policy.
    Switching the adapter off inside a context manager turns the same weights
    back into the original model, which halves the memory this loop needs.
    """
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
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
    base = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, config)
    model.print_trainable_parameters()
    return model, tokenizer, device


def build_prompt(tokenizer, question):
    """Render one problem through the chat template and open the first tag.

    The opening tag is written into the prompt instead of being left for the
    model to produce. Without it this loop stalls at reward zero: a base model
    asked for tagged output writes a paragraph of prose instead, no sample in the
    group earns anything, every advantage is zero and no gradient exists. Handing
    over the first token of the structure is the cheapest way to make some
    samples better than others, which is the only condition under which a
    group-relative method can start.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    return rendered + PREFILL


def sample_group(model, tokenizer, question, device, group_size, max_new_tokens,
                 temperature, top_p):
    """Step 3. Draw several answers to the same problem from the current policy.

    Sampling has to be on, and the temperature has to be high enough that the
    group actually differs. Identical samples all earn identical rewards, and the
    centring step that follows would then hand back nothing but zeros.
    """
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    completions = [
        PREFILL + tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
        for sequence in outputs
    ]
    return outputs, prompt_length, completions


def advantages_from_rewards(rewards):
    """Step 4. Centre and scale the rewards inside the group.

    A group that grades itself needs no value network: the mean of the group is
    the baseline, and each sample is judged on how much better or worse it is
    than its siblings on the very same problem. That is what "group relative"
    names. When every sample in a group scores the same, the standard deviation
    is zero, the advantages are zero, and the step is correctly a no-op.
    """
    tensor = torch.tensor(rewards, dtype=torch.float32)
    spread = tensor.std()
    if spread < 1e-6:
        return torch.zeros_like(tensor), float(tensor.mean()), 0.0
    return (tensor - tensor.mean()) / (spread + 1e-6), float(tensor.mean()), float(spread)


def sequence_log_probabilities(model, sequences, prompt_length, pad_token_id):
    """Per-token log probabilities of the sampled continuation, with padding masked.

    The log probability of the chosen token is taken as its logit minus the log
    sum of all logits, rather than by normalising the whole distribution first.
    Both give the same number, but a full log-softmax allocates another array the
    size of the logits - sequences by positions by the entire vocabulary - and on
    a vocabulary of a hundred and fifty thousand entries that second copy is
    hundreds of megabytes that the card does not have to spare.
    """
    attention = (sequences != pad_token_id).long()
    attention[:, :prompt_length] = 1
    logits = model(input_ids=sequences, attention_mask=attention).logits[:, :-1]
    targets = sequences[:, 1:]
    chosen = logits.gather(2, targets.unsqueeze(-1)).squeeze(-1).float()
    normaliser = torch.logsumexp(logits.float(), dim=-1)
    gathered = chosen - normaliser
    mask = torch.zeros_like(gathered)
    mask[:, prompt_length - 1:] = 1.0
    mask = mask * (targets != pad_token_id).float()
    return gathered, mask


def policy_step(model, sequences, prompt_length, advantages, pad_token_id,
                kl_coefficient, loss_scale, chunk_size):
    """Step 5-6. Push probability toward the samples that beat their group.

    Two terms are added. The first multiplies each sampled token's log
    probability by the advantage of the sample it came from, so text that scored
    above its siblings becomes more likely. The second penalises drifting away
    from the frozen base, measured on the same tokens with the adapter switched
    off. Without that second term the policy is free to collapse onto whatever
    quirk the reward functions reward, and fluency is not among the things they
    check.

    The group is walked in chunks and each chunk is backpropagated as it is
    computed, so only one chunk of logits is alive at a time. Gradients add up in
    the parameters exactly as they would from one large batch; what changes is
    the peak memory, which is what decides whether this runs at all on a single
    consumer card.
    """
    group_size = sequences.shape[0]
    policy_total = 0.0
    kl_total = 0.0
    for start in range(0, group_size, chunk_size):
        stop = min(start + chunk_size, group_size)
        piece = sequences[start:stop]
        piece_advantages = advantages[start:stop]

        log_probabilities, mask = sequence_log_probabilities(
            model, piece, prompt_length, pad_token_id)
        with torch.no_grad():
            with model.disable_adapter():
                reference, _ = sequence_log_probabilities(
                    model, piece, prompt_length, pad_token_id)

        token_counts = mask.sum(dim=1).clamp(min=1.0)
        mean_log_probability = (log_probabilities * mask).sum(dim=1) / token_counts
        policy_loss = -(piece_advantages * mean_log_probability).sum() / group_size
        log_ratio = (log_probabilities - reference) * mask
        kl = (log_ratio.sum(dim=1) / token_counts).sum() / group_size

        (loss_scale * (policy_loss + kl_coefficient * kl)).backward()
        policy_total += float(policy_loss.detach())
        kl_total += float(kl.detach())
        del log_probabilities, reference, mask
    return policy_total, kl_total


def evaluate(model, tokenizer, problems, device, max_new_tokens, label, show=2):
    """Step 7. Score greedy answers on format compliance and correctness."""
    model.eval()
    shaped = 0
    correct = 0
    samples = []
    for problem in problems:
        prompt = build_prompt(tokenizer, problem["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = PREFILL + tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        shaped += int(bool(SOFT_PATTERN.search(completion)))
        correct += int(extract_answer(completion) == str(problem["answer"]))
        samples.append((problem, completion))

    print(f"\n{label}")
    print(f"  answers holding the tag structure: {shaped}/{len(problems)} "
          f"({shaped / len(problems):.1%})")
    print(f"  answers with the right integer:    {correct}/{len(problems)} "
          f"({correct / len(problems):.1%})")
    for problem, completion in samples[:show]:
        collapsed = " ".join(completion.split())[:160]
        print(f"    expected {problem['answer']}, produced {collapsed!r}")
    return shaped / len(problems), correct / len(problems)


def train(model, tokenizer, problems, device, steps, prompts_per_step, group_size,
          learning_rate, kl_coefficient, max_new_tokens, temperature, top_p, seed):
    """Run the sampling and update loop, reporting every reward term as it moves."""
    rng = random.Random(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=learning_rate)

    header = "  ".join(f"{name:>7}" for name, _ in REWARDS)
    print(f"\n{'step':>5} {'reward':>7} {'spread':>7} {'kl':>8} {header}")
    started = time.time()
    history = []
    for step in range(1, steps + 1):
        batch_loss = 0.0
        batch_reward = 0.0
        batch_spread = 0.0
        batch_kl = 0.0
        totals = {name: 0.0 for name, _ in REWARDS}
        optimiser.zero_grad(set_to_none=True)

        for problem in rng.sample(problems, prompts_per_step):
            sequences, prompt_length, completions = sample_group(
                model, tokenizer, problem["question"], device, group_size,
                max_new_tokens, temperature, top_p)
            rewards = []
            for completion in completions:
                total, parts = score_completion(completion, problem["answer"])
                rewards.append(total)
                for name, value in parts.items():
                    totals[name] += value / (group_size * prompts_per_step)
            advantages, mean_reward, spread = advantages_from_rewards(rewards)
            advantages = advantages.to(device)

            model.train()
            policy_loss, kl = policy_step(
                model, sequences, prompt_length, advantages,
                tokenizer.pad_token_id, kl_coefficient,
                loss_scale=1.0 / prompts_per_step, chunk_size=CHUNK_SIZE)
            batch_loss += policy_loss / prompts_per_step
            batch_reward += mean_reward / prompts_per_step
            batch_spread += spread / prompts_per_step
            batch_kl += kl / prompts_per_step

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimiser.step()
        history.append(batch_reward)
        parts = "  ".join(f"{totals[name]:>7.3f}" for name, _ in REWARDS)
        print(f"{step:>5} {batch_reward:>7.3f} {batch_spread:>7.3f} "
              f"{batch_kl:>8.4f} {parts}")

    elapsed = time.time() - started
    early = sum(history[:5]) / 5
    late = sum(history[-5:]) / 5
    print(f"\nMean reward, first five steps {early:.3f}, last five steps {late:.3f}")
    print(f"Wall clock: {elapsed:.1f} s for {steps} steps "
          f"({elapsed / steps:.1f} s per step)")
    if torch.cuda.is_available():
        print(f"Peak VRAM reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


def main():
    torch.manual_seed(SEED)

    print("--- 1. Build problems that can be checked automatically ---")
    problems = make_problems(60, SEED)
    evaluation = problems[:EVAL_PROBLEMS]
    training = problems[EVAL_PROBLEMS:]
    print(f"Problems: {len(training)} for training, {len(evaluation)} held out")
    print(f"  example question: {training[0]['question']}")
    print(f"  example answer:   {training[0]['answer']}")

    print("\n--- 2. The five reward terms, scored on two hand-written samples ---")
    good = "<reasoning>\nSubtract the three known counts.\n</reasoning>\n<answer>\n42\n</answer>"
    poor = "The answer is probably around forty something."
    for label, sample in (("well formed, correct", good), ("prose, no tags", poor)):
        total, parts = score_completion(sample, 42)
        detail = ", ".join(f"{name} {value:.3f}" for name, value in parts.items())
        print(f"  {label:>22}: total {total:.3f}  ({detail})")

    print("\n--- 3. Load the policy and attach the adapter ---")
    model, tokenizer, device = load_policy(MODEL_ID, CACHE_DIR, RANK, ALPHA,
                                           TARGET_MODULES)
    print(f"[Device] {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    print("\n--- 7. Score the policy before any updates ---")
    before_shape, before_correct = evaluate(
        model, tokenizer, evaluation, device, MAX_NEW_TOKENS, "Before training:")

    print("\n--- 4-6. Sample groups, centre the rewards, update the adapter ---")
    train(model, tokenizer, training, device, STEPS, PROMPTS_PER_STEP, GROUP_SIZE,
          LEARNING_RATE, KL_COEFFICIENT, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, SEED)

    after_shape, after_correct = evaluate(
        model, tokenizer, evaluation, device, MAX_NEW_TOKENS, "After training:")
    print(f"\nTag structure {before_shape:.1%} -> {after_shape:.1%}")
    print(f"Correct integer {before_correct:.1%} -> {after_correct:.1%}")


if __name__ == "__main__":
    main()
