"""This script checks LoRA's premise, that a full fine-tuning update is nearly low
rank, by training a model's weights with no rank limit and reading the singular
values of the update. LoRA trains two thin matrices beside a frozen weight
instead of the weight itself, which only works if the update it stands in for
is nearly low rank. Here two projections per layer train in full.

The run prints eight parts:
    1. A LoRA layer by hand. It starts as a no-op because `up` starts at zero.
    2. Adapter parameters against a 1536x1536 matrix, for ranks 1 to 128.
    3. The projections an adapter can attach to, and what three selections cost.
    4. A full-rank update. AdamW trains q_proj and v_proj in the last four layers
       for 40 steps on eight fixed pairs.
    5. The spectrum of each of the eight updates: how many directions 50, 90 and
       99 percent of the energy need.
    6. Two controls for layer 24's q_proj: the frozen weight and random noise of
       the same shape and scale.
    7. Layer 24's q_proj update truncated to rank r: the energy kept, the relative
       error left and the adapter parameters.
    8. The same reading as the training pool covers 1, 3 and 10 kinds of task,
       with pool size, batch, steps and learning rate held fixed. The final
       losses differ, so a tier may need more directions because it is less
       converged, not only because its work is broader.

What the measurement covers. The update comes from eight pairs trained for 40
steps, and its loss ends at 0.05, closer to memorisation than to a skill. Only
q_proj and v_proj in layers 24 to 27 are measured. k_proj, o_proj and the three
feed forward projections hold most of the parameters and are never measured, and
neither is any lower layer. So the reading supports "this kind of fine-tuning
concentrates its update in few directions", not "weight updates are low rank".
"""

import os
import sys
from pathlib import Path

import torch

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
SIBLING_CACHE = Path(__file__).parent.parent / "01-llm-foundation" / "weights"
CACHE_DIR = os.getenv("HF_CACHE_DIR", str(SIBLING_CACHE))

TARGET_SUFFIXES = ("q_proj", "v_proj")
TRAINED_LAYERS = 4
STEPS = 40
LEARNING_RATE = 1e-4
MAX_LENGTH = 96
RANKS = (1, 2, 4, 8, 16, 32, 64, 128)
ENERGY_TARGETS = (0.5, 0.9, 0.99)
TASK_TIERS = (1, 3, 10)
TASK_POOL = 48
TASK_BATCH = 8

# Short, repetitive supervision. What matters is a consistent gradient signal, so
# that the update comes from real optimisation and not from random perturbation.
TRAINING_PAIRS = (
    ("Summarise the risk: driver aged 19, three claims.", "Risk level: high."),
    ("Summarise the risk: driver aged 45, no claims.", "Risk level: low."),
    ("Summarise the risk: driver aged 22, one claim.", "Risk level: elevated."),
    ("Summarise the risk: driver aged 60, no claims.", "Risk level: low."),
    ("Summarise the risk: driver aged 30, two claims.", "Risk level: elevated."),
    ("Summarise the risk: driver aged 18, one claim.", "Risk level: high."),
    ("Summarise the risk: driver aged 52, one claim.", "Risk level: low."),
    ("Summarise the risk: driver aged 25, four claims.", "Risk level: high."),
)


class LowRankAdapter(torch.nn.Module):
    """Part 1. A frozen linear layer plus up @ down, scaled by alpha / r.
    `up` starts at zero, so training begins from the frozen layer's exact output."""

    def __init__(self, base: torch.nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.down = torch.nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.up = torch.nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = alpha / rank

    def forward(self, hidden):
        return self.base(hidden) + (hidden @ self.down.T @ self.up.T) * self.scale

    def effective_update(self):
        """The dense matrix the adapter is equivalent to, for inspection only."""
        return (self.up @ self.down) * self.scale


def adapter_is_identity_at_start():
    """Part 1. Show the adapter changes nothing before training and something after."""
    torch.manual_seed(3407)
    base = torch.nn.Linear(256, 256, bias=False)
    adapter = LowRankAdapter(base, rank=8, alpha=16.0)
    sample = torch.randn(4, 256)

    with torch.no_grad():
        frozen_output = base(sample)
        adapted_output = adapter(sample)
    print(f"down shape {tuple(adapter.down.shape)}, up shape {tuple(adapter.up.shape)}")
    print(f"up starts at zero: {bool(torch.all(adapter.up == 0))}")
    print(f"max difference from the frozen layer: "
          f"{(adapted_output - frozen_output).abs().max():.2e}")
    print(f"rank of the equivalent update matrix: "
          f"{torch.linalg.matrix_rank(adapter.effective_update()).item()} of 256")

    with torch.no_grad():
        adapter.up.normal_(0.0, 0.02)
        changed_output = adapter(sample)
    print(f"after up is given values, max difference: "
          f"{(changed_output - frozen_output).abs().max():.4f}")
    print(f"rank of the equivalent update matrix: "
          f"{torch.linalg.matrix_rank(adapter.effective_update()).item()} of 256")


def parameter_accounting(out_features, in_features, ranks):
    """Part 2. Compare adapter parameters against the frozen matrix they sit beside."""
    dense = out_features * in_features
    print(f"\nFrozen matrix: {out_features} x {in_features} = {dense:,} parameters")
    print(f"{'rank':>6} {'adapter params':>16} {'share of matrix':>17}")
    for rank in ranks:
        params = rank * (out_features + in_features)
        print(f"{rank:>6} {params:>16,} {params / dense:>16.2%}")


def survey_projections(model, ranks):
    """Part 3. Report every linear projection and price three selections of them.
    An adapter attaches to a chosen list of module names, and the list sets its cost."""
    groups = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            groups.setdefault(suffix, []).append(
                (name, module.out_features, module.in_features))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params / 1e9:.2f}B")
    print(f"\n{'projection':>12} {'count':>7} {'shape':>18} {'dense params':>15}")
    for suffix, entries in groups.items():
        name, out_features, in_features = entries[0]
        dense = sum(out * inp for _, out, inp in entries)
        print(f"{suffix:>12} {len(entries):>7} {f'{out_features}x{in_features}':>18} "
              f"{dense:>15,}")

    selections = {
        "q_proj + v_proj": ("q_proj", "v_proj"),
        "all four attention projections": ("q_proj", "k_proj", "v_proj", "o_proj"),
        # The seven per-layer projections. lm_head is left out, as in PEFT's all-linear.
        "all seven projections": tuple(key for key in groups if key != "lm_head"),
    }
    print(f"\n{'selection':>32} {'modules':>9} " +
          " ".join(f"{f'r={rank}':>11}" for rank in ranks[:5]))
    for label, suffixes in selections.items():
        entries = [entry for suffix in suffixes for entry in groups.get(suffix, [])]
        counts = []
        for rank in ranks[:5]:
            params = sum(rank * (out + inp) for _, out, inp in entries)
            counts.append(f"{params / total_params:>10.3%}")
        print(f"{label:>32} {len(entries):>9} " + " ".join(counts))
    return groups


def load_model():
    """Load the checkpoint in float32, because in bfloat16 (about three decimal digits)
    an update far smaller than the weights would be partly rounded away."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    # The loader draws a progress bar with carriage returns, which turns into
    # thousands of lines when the output is piped to a file instead of a terminal.
    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    print(f"[Weights] {MODEL_ID}")
    path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[Model] dtype={model.dtype}, "
          f"VRAM allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB"
          if device == "cuda" else f"[Model] dtype={model.dtype}")
    return model, tokenizer, device


def select_trained_modules(model, suffixes, layer_count):
    """Pick the projections to update, taking the last few layers only."""
    layers = model.model.layers
    chosen = {}
    for index in range(len(layers) - layer_count, len(layers)):
        for name, module in layers[index].named_modules():
            if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in suffixes:
                chosen[f"layer{index}.{name}"] = module
    return chosen


def build_batch(tokenizer, pairs, device, max_length):
    """Tokenise the pairs and mask the prompt tokens out of the loss, so only the
    answer tokens produce the gradient behind the measured update."""
    input_ids, labels = [], []
    for prompt, answer in pairs:
        prompt_ids = tokenizer(f"{prompt}\n", add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [tokenizer.eos_token_id]
        ids = (prompt_ids + answer_ids)[:max_length]
        label = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
        padding = max_length - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * padding)
        labels.append(label + [-100] * padding)
    return (torch.tensor(input_ids, device=device),
            torch.tensor(labels, device=device))


def train_full_rank(model, tokenizer, modules, device, steps, learning_rate,
                    max_length, pairs=None, batch_size=None, verbose=True):
    """Part 4. Train the chosen projections with AdamW and no rank limit, and return
    the update. Part 8 passes a larger pool and a fixed batch size."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for module in modules.values():
        module.weight.requires_grad_(True)
        trainable.append(module.weight)

    before = {name: module.weight.detach().clone()
              for name, module in modules.items()}
    trainable_count = sum(p.numel() for p in trainable)
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Updating {len(modules)} matrices, {trainable_count:,} parameters "
              f"({trainable_count / total:.3%} of the model)")

    optimiser = torch.optim.AdamW(trainable, lr=learning_rate)
    all_ids, all_labels = build_batch(tokenizer, pairs or TRAINING_PAIRS, device,
                                      max_length)
    pool = all_ids.shape[0]
    width = batch_size or pool
    model.train()
    first_loss = None
    for step in range(1, steps + 1):
        start = ((step - 1) * width) % pool
        rows = [(start + offset) % pool for offset in range(width)]
        input_ids, labels = all_ids[rows], all_labels[rows]
        optimiser.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimiser.step()
        if first_loss is None:
            first_loss = loss.item()
        if verbose and (step % 10 == 0 or step == 1):
            print(f"  step {step:>3d}  loss {loss.item():.4f}")
    final_loss = loss.item()
    if verbose:
        print(f"Loss moved {first_loss:.4f} -> {final_loss:.4f}")
        if torch.cuda.is_available():
            print(f"Peak VRAM reserved: "
                  f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
    train_full_rank.final_loss = final_loss

    updates = {name: (module.weight.detach() - before[name]).float().cpu()
               for name, module in modules.items()}
    return updates, {name: tensor.float().cpu() for name, tensor in before.items()}


def spectrum_summary(matrix, targets):
    """Return the singular values and the rank needed for each share of the energy."""
    values = torch.linalg.svdvals(matrix)
    energy = values**2
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    needed = {target: int(torch.searchsorted(cumulative, target).item()) + 1
              for target in targets}
    return values, needed


def inspect_updates(updates, weights, targets, ranks):
    """Parts 5 to 7. Decompose each update, then compare and truncate layer 24's q_proj."""
    generator = torch.Generator().manual_seed(3407)
    print(f"\n{'matrix':>22} {'shape':>14} {'full rank':>10} " +
          " ".join(f"{f'{target:.0%}':>7}" for target in targets))
    for name, update in updates.items():
        values, needed = spectrum_summary(update, targets)
        full_rank = min(update.shape)
        print(f"{name:>22} {f'{update.shape[0]}x{update.shape[1]}':>14} "
              f"{full_rank:>10} " +
              " ".join(f"{needed[target]:>7d}" for target in targets))

    name, update = next(iter(updates.items()))
    values, needed = spectrum_summary(update, targets)
    print(f"\nTaking {name} as the example:")
    print(f"  largest singular value  {values[0]:.6f}")
    print(f"  singular value 64       {values[63]:.6f}")
    print(f"  ratio                   {values[0] / values[63]:.1f}x")

    print("\n--- 6. Two controls for layer 24's q_proj ---")
    weight = weights[name]
    weight_values, weight_needed = spectrum_summary(weight, targets)
    noise = torch.randn(update.shape, generator=generator) * update.std()
    noise_values, noise_needed = spectrum_summary(noise, targets)
    print(f"{'matrix':>22} " + " ".join(f"{f'{t:.0%}':>7}" for t in targets) +
          f" {'top/64':>9}")
    for label, vals, need in (("trained update", values, needed),
                              ("frozen weight", weight_values, weight_needed),
                              ("random noise", noise_values, noise_needed)):
        print(f"{label:>22} " + " ".join(f"{need[t]:>7d}" for t in targets) +
              f" {vals[0] / vals[63]:>8.1f}x")
    print("\nThe update concentrates in a few directions; noise of the same size")
    print("spreads across all of them, and the frozen weight sits in between.")
    print("Only the first of those three is compressible, and it is the one an")
    print("adapter has to represent.")

    print("\n--- 7. The update truncated to rank r ---")
    left, singular, right = torch.linalg.svd(update, full_matrices=False)
    total_norm = torch.linalg.norm(update)
    print(f"{'rank':>6} {'energy kept':>12} {'rel. error':>11} {'adapter params':>16}")
    errors = {}
    for rank in ranks:
        if rank > len(singular):
            continue
        approximation = (left[:, :rank] * singular[:rank]) @ right[:rank]
        error = torch.linalg.norm(update - approximation) / total_norm
        kept = 1 - error**2
        errors[rank] = (kept, error)
        params = rank * (update.shape[0] + update.shape[1])
        print(f"{rank:>6} {kept:>11.2%} {error:>10.2%} {params:>16,}")
    kept, error = errors[16]
    print(f"At rank 16 the update keeps {kept:.0%} of its energy, but {error:.0%} of its magnitude")
    print("is still wrong. These shares belong to one narrow task, and part 8 shows the")
    print("count of directions moving as the training pool widens.")


# Ten short input-output tasks used by part 8. Its tiers are nested prefixes of
# TASK_BUILDERS, so moving up a tier only adds kinds of work.
CAPITALS = (("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"),
            ("Egypt", "Cairo"), ("Norway", "Oslo"), ("Chile", "Santiago"))
ANTONYMS = (("increase", "decrease"), ("wide", "narrow"), ("early", "late"),
            ("heavy", "light"), ("open", "closed"), ("strong", "weak"))
PAST_TENSE = (("run", "ran"), ("write", "wrote"), ("bring", "brought"),
              ("keep", "kept"), ("send", "sent"), ("teach", "taught"))
CATEGORIES = (("sparrow", "bird"), ("salmon", "fish"), ("beetle", "insect"),
              ("otter", "mammal"), ("gecko", "reptile"), ("toad", "amphibian"))
WORDS = ("banana", "engine", "cactus", "rocket", "planet", "silver")
SENTIMENTS = (("The service was terrible.", "negative"),
              ("The parcel arrived early.", "positive"),
              ("The seat would not recline.", "negative"),
              ("The room was spotless.", "positive"),
              ("The call was cut off twice.", "negative"),
              ("The refund cleared the same day.", "positive"))
DISTANCES = (3, 7, 12, 25, 40, 60)


def task_risk(count):
    pairs = []
    for index in range(count):
        age = 18 + (index * 7) % 48
        claims = index % 5
        if claims >= 3 or (age < 25 and claims >= 1):
            level = "high"
        elif claims >= 1:
            level = "elevated"
        else:
            level = "low"
        noun = "claim" if claims == 1 else "claims"
        pairs.append((f"Summarise the risk: driver aged {age}, {claims} {noun}.",
                      f"Risk level: {level}."))
    return pairs


def task_sum(count):
    pairs = []
    for index in range(count):
        left = 11 + (index * 13) % 80
        right = 3 + (index * 7) % 40
        pairs.append((f"Add: {left} and {right}.", f"Result: {left + right}."))
    return pairs


def task_larger(count):
    pairs = []
    for index in range(count):
        left = 17 + (index * 23) % 80
        right = 9 + (index * 31) % 80
        right = right + 1 if right == left else right
        pairs.append((f"Larger: {left} or {right}.", f"Larger: {max(left, right)}."))
    return pairs


def cycle_pairs(table, count, prompt, answer):
    return [(prompt.format(*table[index % len(table)]),
             answer.format(*table[index % len(table)]))
            for index in range(count)]


def task_capital(count):
    return cycle_pairs(CAPITALS, count, "Capital: {0}.", "Capital: {1}.")


def task_antonym(count):
    return cycle_pairs(ANTONYMS, count, "Opposite: {0}.", "Opposite: {1}.")


def task_past(count):
    return cycle_pairs(PAST_TENSE, count, "Past tense: {0}.", "Past tense: {1}.")


def task_category(count):
    return cycle_pairs(CATEGORIES, count, "Category: {0}.", "Category: {1}.")


def task_letters(count):
    return [(f"Letters: {WORDS[index % len(WORDS)]}.",
             f"Letters: {len(WORDS[index % len(WORDS)])}.")
            for index in range(count)]


def task_sentiment(count):
    return cycle_pairs(SENTIMENTS, count, "Sentiment: {0}", "Sentiment: {1}.")


def task_metres(count):
    return [(f"Convert: {DISTANCES[index % len(DISTANCES)]} kilometres to metres.",
             f"Result: {DISTANCES[index % len(DISTANCES)] * 1000} metres.")
            for index in range(count)]


TASK_BUILDERS = (task_risk, task_sum, task_larger, task_capital, task_antonym,
                 task_past, task_category, task_letters, task_sentiment,
                 task_metres)


def build_task_pool(kinds, size):
    """Draw `size` examples spread evenly over the first `kinds` task builders."""
    per_kind = -(-size // kinds)
    columns = [TASK_BUILDERS[index](per_kind) for index in range(kinds)]
    pool = [column[row] for row in range(per_kind) for column in columns]
    return tuple(pool[:size])


def rank_versus_task_variety(model, tokenizer, modules, baseline, device, tiers,
                             pool_size, steps, batch_size, learning_rate,
                             max_length, targets):
    """Part 8. Retrain from the original weights with 1, 3 and 10 kinds of task, same
    number of examples each; tier 1 uses the same kind of risk prompt as part 4."""
    originals = {name: tensor.to(device) for name, tensor in baseline.items()}
    short = [name.replace("self_attn.", "").replace("layer", "l")
                 .replace("_proj", "") for name in modules]
    print(f"Pool held at {pool_size} examples, batch at {batch_size}, "
          f"{steps} steps, lr {learning_rate}.")
    print("\n" + f"{'task kinds':>11} {'final loss':>11} " +
          " ".join(f"{f'{target:.0%} mean':>10}" for target in targets))
    detail = []
    for kinds in tiers:
        with torch.no_grad():
            for name, module in modules.items():
                module.weight.copy_(originals[name])
        pool = build_task_pool(kinds, pool_size)
        updates, _ = train_full_rank(model, tokenizer, modules, device, steps,
                                     learning_rate, max_length, pairs=pool,
                                     batch_size=batch_size, verbose=False)
        needed = {target: [] for target in targets}
        for update in updates.values():
            _, counts = spectrum_summary(update, targets)
            for target in targets:
                needed[target].append(counts[target])
        print(f"{kinds:>11d} {train_full_rank.final_loss:>11.4f} " +
              " ".join(f"{sum(needed[t]) / len(needed[t]):>10.1f}" for t in targets))
        detail.append((kinds, needed[0.9]))
    with torch.no_grad():
        for name, module in modules.items():
            module.weight.copy_(originals[name])

    print("\n" + "Directions needed for 90% of the energy, per matrix:")
    print(f"{'task kinds':>11} " + " ".join(f"{label:>8}" for label in short))
    for kinds, counts in detail:
        print(f"{kinds:>11d} " + " ".join(f"{count:>8d}" for count in counts))


def main():
    print("--- 1. A LoRA layer by hand ---")
    adapter_is_identity_at_start()

    print("\n--- 2. Adapter parameters against the frozen matrix ---")
    parameter_accounting(1536, 1536, RANKS)

    model, tokenizer, device = load_model()

    print("\n--- 3. The projections an adapter can attach to ---")
    survey_projections(model, RANKS)

    print("\n--- 4. A full-rank update ---")
    modules = select_trained_modules(model, TARGET_SUFFIXES, TRAINED_LAYERS)
    print(f"Selected: {', '.join(modules)}")
    updates, weights = train_full_rank(model, tokenizer, modules, device, STEPS,
                                       LEARNING_RATE, MAX_LENGTH)

    print("\n--- 5. The spectrum of each update ---")
    inspect_updates(updates, weights, ENERGY_TARGETS, RANKS)

    print("\n--- 8. The same reading as the task pool widens ---")
    rank_versus_task_variety(model, tokenizer, modules, weights, device,
                             TASK_TIERS, TASK_POOL, STEPS, TASK_BATCH,
                             LEARNING_RATE, MAX_LENGTH, ENERGY_TARGETS)


if __name__ == "__main__":
    main()
