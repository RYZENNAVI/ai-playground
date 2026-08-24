"""Test the assumption a low-rank adapter rests on: that a weight update is nearly low rank.

Demonstrates where the rank budget of an adapter comes from:
    1. Build a low-rank adapter by hand and confirm it starts as a no-op.
    2. Count adapter parameters against the frozen matrix for a range of ranks.
    3. List which projections an adapter would attach to, and what each choice costs.
    4. Update two projections per layer with unconstrained full-rank gradients.
    5. Decompose the resulting update and read off how many terms it really uses.
    6. Compare that spectrum against the frozen weight and against random noise.
    7. Truncate the update to rank r and measure how much of it survives.

Module 05: Fine-Tuning - The Low-Rank Update Hypothesis.
"""

import os
import sys
from pathlib import Path

import torch

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

# Short, repetitive supervision. The task is not the point here; a consistent
# gradient signal is, because the update has to come from real optimisation
# rather than from random perturbation for the decomposition to mean anything.
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
    """Step 1. A frozen matrix plus a trainable pair of thin matrices.

    The forward pass adds two things: the frozen projection, and the input pushed
    through a narrow bottleneck of width r and back out. `down` starts at random
    and `up` starts at zero, so the added term is exactly zero at the start and
    the adapted layer answers identically to the frozen one. That is what makes
    attaching an adapter safe: training begins from the original behaviour rather
    than from a perturbed version of it.

    The scale factor alpha / r keeps the size of the added term roughly constant
    when r changes, so raising the rank adds capacity without also multiplying
    the effective learning rate.
    """

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
    """Step 1. Show the adapter changes nothing before training and something after."""
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
    """Step 2. Compare adapter parameters against the frozen matrix they sit beside."""
    dense = out_features * in_features
    print(f"\nFrozen matrix: {out_features} x {in_features} = {dense:,} parameters")
    print(f"{'rank':>6} {'adapter params':>16} {'share of matrix':>17}")
    for rank in ranks:
        params = rank * (out_features + in_features)
        print(f"{rank:>6} {params:>16,} {params / dense:>16.2%}")


def survey_projections(model, ranks):
    """Step 3. Report every attachable projection and price the usual selections.

    An adapter is not paired one-to-one with every weight in the network. It is
    attached to a chosen list of module names, and that list is a decision with a
    cost. Attaching to the two attention projections most commonly chosen touches
    a small fraction of the network; attaching to all seven linear projections
    multiplies both the parameter count and the memory the optimiser needs.
    """
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
        "attention and feed forward": tuple(groups.keys()),
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
    """Load the checkpoint in float32 so a small weight change is not lost to rounding.

    bfloat16 carries about three decimal digits, and the update measured in step 4
    is far smaller than the weights it is added to. Storing the weights in float32
    keeps the difference between before and after meaningful; the cost is roughly
    double the memory, which a 1.5B checkpoint can still afford here.
    """
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
    """Tokenise the pairs and mask the prompt tokens out of the loss.

    Only the answer tokens carry a label. Leaving the prompt tokens in the loss
    would train the model to generate the questions as well, which is a different
    objective than the one intended and dilutes the gradient that produces the
    update being measured.
    """
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


def train_full_rank(model, tokenizer, modules, device, steps, learning_rate, max_length):
    """Step 4. Update the chosen projections with no rank constraint at all.

    Every entry of these matrices is free to move, so the update that comes out
    is whatever plain gradient descent wanted. If its singular values still decay
    steeply, that decay is a property of the learning problem rather than
    something an adapter imposed - which is the only way this measurement can
    support the low-rank choice instead of assuming it.
    """
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
    print(f"Updating {len(modules)} matrices, {trainable_count:,} parameters "
          f"({trainable_count / total:.3%} of the model)")

    optimiser = torch.optim.AdamW(trainable, lr=learning_rate)
    input_ids, labels = build_batch(tokenizer, TRAINING_PAIRS, device, max_length)
    model.train()
    first_loss = None
    for step in range(1, steps + 1):
        optimiser.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimiser.step()
        if first_loss is None:
            first_loss = loss.item()
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>3d}  loss {loss.item():.4f}")
    print(f"Loss moved {first_loss:.4f} -> {loss.item():.4f}")
    if torch.cuda.is_available():
        print(f"Peak VRAM reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

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
    """Step 5-7. Decompose each update, compare it against controls, and truncate it."""
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

    print("\nStep 6. The same reading for two controls of identical shape:")
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

    print("\nStep 7. Share of the update kept when it is truncated to rank r:")
    left, singular, right = torch.linalg.svd(update, full_matrices=False)
    total_norm = torch.linalg.norm(update)
    print(f"{'rank':>6} {'kept':>9} {'dropped':>9} {'adapter params':>16}")
    for rank in ranks:
        if rank > len(singular):
            continue
        approximation = (left[:, :rank] * singular[:rank]) @ right[:rank]
        kept = 1 - (torch.linalg.norm(update - approximation) / total_norm) ** 2
        params = rank * (update.shape[0] + update.shape[1])
        print(f"{rank:>6} {kept:>8.2%} {1 - kept:>8.2%} {params:>16,}")


def main():
    print("--- 1. Build a low-rank adapter by hand ---")
    adapter_is_identity_at_start()

    print("\n--- 2. Count adapter parameters against the frozen matrix ---")
    parameter_accounting(1536, 1536, RANKS)

    model, tokenizer, device = load_model()

    print("\n--- 3. Survey the projections an adapter could attach to ---")
    survey_projections(model, RANKS)

    print("\n--- 4. Update two projections per layer with full-rank gradients ---")
    modules = select_trained_modules(model, TARGET_SUFFIXES, TRAINED_LAYERS)
    print(f"Selected: {', '.join(modules)}")
    updates, weights = train_full_rank(model, tokenizer, modules, device, STEPS,
                                       LEARNING_RATE, MAX_LENGTH)

    print("\n--- 5. Decompose the update and read its effective rank ---")
    inspect_updates(updates, weights, ENERGY_TARGETS, RANKS)


if __name__ == "__main__":
    main()
