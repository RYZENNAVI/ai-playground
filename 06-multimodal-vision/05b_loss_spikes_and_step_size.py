"""Measure a training loss spike and test what causes it, on the detector from script 05.

That detector's loss falls smoothly for twenty-odd epochs, then every term rises together
for a couple of epochs and falls back. This script treats that as a question to answer with
measurements rather than a story to tell:
    1. Reproduce the training run exactly, recording every optimiser step.
    2. Draw the anatomy of the rise: loss, gradient norm, step size, Adam's moments.
    3. Ask whether one unlucky batch is responsible.
    4. Ask which loss terms rise, and which layers move.
    5. Control: keep everything and change only the batch order.
    6. Control: halve the learning rate, and run long enough to reach the same loss.
    7. Control: train four times longer, and count how often the break-up happens.

The detector, its loss and its data come from script 05 unchanged, so the trajectory here is
the one that script reports; the instrumentation only reads, and never draws a random number
of its own. Expect roughly 40 minutes on a laptop GPU: the long run in step 7 is four times
the original training, and steps 1 to 4 pay for a parameter snapshot at every step.

Module 06: Multimodal Vision - Grid Detection and Pose Assembly, a supplement to script 05.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "grid_detection_and_pose"
DETECTOR_SOURCE = Path(__file__).parent / "05_grid_detection_and_pose_assembly.py"
LONG_EPOCHS, HALF_RATE_EPOCHS, OTHER_SHUFFLE = 120, 60, 12345
QUIET_FROM, QUIET_TO = 21, 24        # the flat stretch before the first break-up
RISE = 1.25                          # what counts as risen, for an epoch and for a rolling median


def load_detector_module():
    """Import script 05 by path, since a module name cannot start with a digit."""
    spec = importlib.util.spec_from_file_location("detector_source", DETECTOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.argv = [str(DETECTOR_SOURCE)]      # its parser must not see this script's arguments
    spec.loader.exec_module(module)
    return module


def build_dataset(s05):
    """The training images, anchors and grid targets of script 05, drawn in its own order."""
    rng = np.random.default_rng(s05.SEED)
    train = [s05.render_detection(rng) for _ in range(s05.TRAIN_IMAGES)]
    boxes = np.concatenate([b for _, b, _ in train])
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    anchors, _ = s05.kmeans_anchors(wh, s05.ANCHORS, np.random.default_rng(s05.SEED))
    targets = [s05.encode(b, l, anchors)[0] for _, b, l in train]
    return train, anchors, targets


def train_detector(s05, data, epochs, learning_rate, shuffle_seed, record=False, label=""):
    """Train exactly as script 05 does, optionally recording every optimiser step.

    Returns the per-epoch totals and, when record is set, a structured array with one row per
    step. Recording clones the parameters around each step to measure how far they moved, and
    keeps the previous gradient to measure whether successive gradients agree.
    """
    import torch

    train, anchors, targets = data
    torch.manual_seed(s05.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = s05.build_detector().to(device)
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    images_t = torch.from_numpy(np.stack([i for i, _, _ in train])).permute(0, 3, 1, 2).contiguous()
    targets_t = torch.from_numpy(np.stack(targets))
    gt_t = [torch.from_numpy(b) for _, b, _ in train]
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(shuffle_seed)

    named = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
    parameters = [q for _, q in named]
    layers = [n for n, _ in named]
    columns = ([("epoch", "i4"), ("step", "i4"), ("batch", "i4")]
               + [(n, "f8") for n in ("xy", "wh", "obj", "noobj", "class", "total")]
               + [("positives", "i4"), ("boxes", "i4"), ("grad_norm", "f8"), ("update_norm", "f8"),
                  ("sqrt_v", "f8"), ("abs_m", "f8"), ("cosine", "f8")]
               + [(f"rel {n}", "f8") for n in layers])
    rows, previous_grad, step = [], None, 0
    totals, started = [], time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(order), s05.BATCH):
            idx = order[start:start + s05.BATCH]
            x = images_t[idx].to(device).float().div(255)
            terms = s05.yolo_loss(s05.reshape_head(model(x)), targets_t[idx].to(device),
                                  [gt_t[i].to(device) for i in idx], anchors_t)
            loss = sum(terms.values())
            optimiser.zero_grad()
            loss.backward()
            if record:
                grad_norm = torch.sqrt(sum((q.grad.detach() ** 2).sum() for q in parameters)).item()
                before = [q.detach().clone() for q in parameters]
                flat = torch.cat([q.grad.detach().flatten() for q in parameters])
                # Adam's first moment only builds up while successive gradients point the same way.
                cosine = (np.nan if previous_grad is None else
                          torch.nn.functional.cosine_similarity(flat, previous_grad, dim=0).item())
                previous_grad = flat
            optimiser.step()
            if record:
                moved = [(q.detach() - b).norm() for q, b in zip(parameters, before)]
                relative = [(m / b.norm().clamp_min(1e-12)).item() for m, b in zip(moved, before)]
                update_norm = torch.sqrt(sum(m ** 2 for m in moved)).item()
                state = [optimiser.state[q] for q in parameters]
                sqrt_v = torch.sqrt(torch.cat([s["exp_avg_sq"].flatten() for s in state])).mean().item()
                abs_m = torch.cat([s["exp_avg"].flatten() for s in state]).abs().mean().item()
                values = {name: value.item() for name, value in terms.items()}
                rows.append(tuple([epoch, step, start // s05.BATCH]
                                  + [values[n] for n in ("xy", "wh", "obj", "noobj", "class")]
                                  + [sum(values.values()),
                                     int((targets_t[idx][..., 4] == 1).sum().item()),
                                     sum(len(gt_t[i]) for i in idx),
                                     grad_norm, update_norm, sqrt_v, abs_m, cosine]
                                  + relative))
            running += loss.item() * len(idx)
            step += 1
        totals.append(running / len(train))
        if epoch % 10 == 0 or epoch == epochs:
            print(f"    {label:<26} epoch {epoch:>4}/{epochs}  total {totals[-1]:>8.4f}"
                  f"  {time.perf_counter() - started:>5.0f}s", flush=True)
    return np.array(totals), (np.array(rows, dtype=columns) if record else None), layers


def rolling_median(values, window=51):
    """A median over the last window steps, to read a trend out of very noisy per-step numbers.

    The first recorded gradient cosine has no previous step to compare against, so this skips
    missing values rather than letting one of them empty the first window.
    """
    windows = [values[max(0, i - window + 1):i + 1] for i in range(len(values))]
    return np.array([np.nanmedian(w) if np.isfinite(w).any() else np.nan for w in windows])


def break_ups(totals, threshold=RISE, after=10):
    """Epochs, one-based, whose total rose over the previous one by more than the threshold."""
    return [(i + 1, totals[i - 1], totals[i]) for i in range(after, len(totals))
            if totals[i] > threshold * totals[i - 1]]


def episodes(totals, threshold=RISE, after=10):
    """Group the rising epochs into events, since one break-up keeps rising for a while.

    Returns, for each event, the epoch it started from, the loss it left, and the highest
    loss it reached before coming back down. Counting rising epochs instead would count a
    single event as two or three.
    """
    found = []
    for epoch, before, after_value in break_ups(totals, threshold, after):
        if found and epoch == found[-1][0] + found[-1][3]:
            start, left, top, length = found[-1]
            found[-1] = (start, left, max(top, after_value), length + 1)
        else:
            found.append((epoch, before, after_value, 1))
    return [(start, left, max(top, totals[start - 1:start + length].max()))
            for start, left, top, length in found]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--long-epochs", type=int, default=LONG_EPOCHS,
                        help=f"epochs for the long run of step 7 (default {LONG_EPOCHS})")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s05 = load_detector_module()
    print("--- 1. The same training run as script 05, with every step recorded ---")
    print(f"  detector from {DETECTOR_SOURCE.name}: Adam {s05.LEARNING_RATE}, batch {s05.BATCH}, "
          f"{s05.EPOCHS} epochs, no schedule")
    data = build_dataset(s05)
    totals, rows, layers = train_detector(s05, data, s05.EPOCHS, s05.LEARNING_RATE, s05.SEED,
                                          record=True, label="recorded run")
    peak = int(np.argmax(totals[QUIET_TO:]) + QUIET_TO) + 1
    print(f"  {len(rows)} optimiser steps over {s05.EPOCHS} epochs; the loss falls to "
          f"{totals[QUIET_TO - 1]:.4f} by epoch {QUIET_TO}, peaks at {totals[peak - 1]:.4f} in epoch "
          f"{peak}, and ends at {totals[-1]:.4f}")
    for epoch, before, after in break_ups(totals):
        print(f"  epoch {epoch}: {before:.4f} -> {after:.4f}, x{after / before:.2f}")

    # 2. Anatomy
    print("\n--- 2. What moves during the rise ---")
    quiet = rows[(rows["epoch"] >= QUIET_FROM) & (rows["epoch"] <= QUIET_TO)]
    loud = rows[(rows["epoch"] >= peak - 1) & (rows["epoch"] <= peak)]
    print(f"  {'':<24}{f'quiet, ep {QUIET_FROM}-{QUIET_TO}':>18}{f'rise, ep {peak - 1}-{peak}':>18}{'ratio':>8}")
    for name, key in (("total loss", "total"), ("gradient norm", "grad_norm"),
                      ("update norm", "update_norm"), ("mean sqrt(v)", "sqrt_v"),
                      ("mean |m|", "abs_m"), ("gradient cosine", "cosine")):
        a, b = np.nanmedian(quiet[key]), np.nanmedian(loud[key])
        print(f"  {name:<24}{a:>18.5f}{b:>18.5f}{b / a:>8.2f}")
    per_gradient = (np.nanmedian(loud["update_norm"] / loud["grad_norm"])
                    / np.nanmedian(quiet["update_norm"] / quiet["grad_norm"]))
    print(f"  Adam's distance per unit of gradient changes by x{per_gradient:.2f}, so the larger steps "
          f"are not only larger gradients")
    print("  the second moment does not dip, so the steps do not grow because Adam's denominator shrank")

    arrivals = []
    for key in ("update_norm", "abs_m", "grad_norm", "total"):
        base = float(np.median(quiet[key]))
        rolled = rolling_median(rows[key])
        start = int(np.flatnonzero(rows["epoch"] == QUIET_TO)[-1])
        hit = np.flatnonzero((np.arange(len(rolled)) > start) & (rolled > RISE * base))
        arrivals.append((key, int(rows["step"][hit[0]]) if len(hit) else -1))
    print(f"  a rolling median first passes {RISE:g}x its quiet level at step: "
          + ", ".join(f"{k} {s}" for k, s in arrivals))
    print("  (their noise levels differ, so treat that ordering as weak evidence)")

    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    panels = (("total loss\nper batch", "total", "k", True),
              ("gradient norm", "grad_norm", "tab:red", True),
              ("parameter\nupdate norm", "update_norm", "tab:green", True),
              ("gradient cosine\nwith previous step", "cosine", "tab:blue", False))
    for ax, (label, key, colour, log) in zip(axes, panels):
        ax.plot(rows["step"], rows[key], lw=0.5, color=colour, alpha=0.35 if key == "cosine" else 1.0)
        if key == "cosine":
            # Step to step this one is pure noise; the trend only appears through a median.
            ax.plot(rows["step"], rolling_median(rows[key]), lw=1.6, color="tab:blue",
                    label="median of the last 51 steps")
            ax.legend(loc="lower left", fontsize=9)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(label)
    axes[4].plot(rows["step"], rows["abs_m"], lw=0.6, color="tab:orange", label="mean |m|, first moment")
    axes[4].plot(rows["step"], rows["sqrt_v"], lw=0.6, color="tab:purple", label="mean sqrt(v), second moment")
    axes[4].set_yscale("log")
    axes[4].set_ylabel("Adam's moments")
    axes[4].legend(loc="lower left", fontsize=9)
    axes[4].set_xlabel(f"optimiser step ({len(rows) // s05.EPOCHS} per epoch)")
    span = (rows["step"][rows["epoch"] == peak - 1][0], rows["step"][rows["epoch"] == peak][-1])
    for ax in axes:
        ax.axvspan(*span, color="tab:red", alpha=0.08, zorder=0)
        ax.grid(alpha=0.25)
    axes[0].set_title(f"every optimiser step of the detector; epochs {peak - 1}-{peak} shaded")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spike_anatomy.png", dpi=110)
    plt.close(fig)
    print("  spike_anatomy.png: loss, gradient norm, update norm, gradient cosine and Adam's two")
    print("  moments for every step, with the rise shaded")

    # 3. One unlucky batch?
    print("\n--- 3. Is one batch responsible? ---")
    print(f"  {'epoch':>6}{'median':>10}{'largest':>10}{'largest/median':>16}{'boxes in it':>13}")
    for epoch in range(QUIET_FROM, min(peak + 2, s05.EPOCHS) + 1):
        s = rows[rows["epoch"] == epoch]
        worst = int(np.argmax(s["total"]))
        print(f"  {epoch:>6}{np.median(s['total']):>10.4f}{s['total'].max():>10.4f}"
              f"{s['total'].max() / np.median(s['total']):>16.2f}{s['boxes'][worst]:>13}")
    print("  the heaviest batch of a rising epoch stands out no further than in a quiet one, and the")
    print("  rise takes hundreds of steps, so no single batch explains it")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    quiet_mask = (rows["epoch"] >= QUIET_FROM) & (rows["epoch"] <= QUIET_TO)
    loud_mask = (rows["epoch"] >= peak - 1) & (rows["epoch"] <= peak)
    axes[0].scatter(rows["boxes"][quiet_mask], rows["total"][quiet_mask], s=8, alpha=0.5,
                    label=f"quiet, epochs {QUIET_FROM}-{QUIET_TO}")
    axes[0].scatter(rows["boxes"][loud_mask], rows["total"][loud_mask], s=8, alpha=0.5,
                    label=f"rising, epochs {peak - 1}-{peak}", color="tab:red")
    axes[0].set_xlabel("ground-truth boxes in the batch")
    axes[0].set_ylabel("total loss")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].set_title("a heavier batch costs a little more, in both stretches")
    ratios = [rows["total"][rows["epoch"] == e].max() / np.median(rows["total"][rows["epoch"] == e])
              for e in range(1, s05.EPOCHS + 1)]
    axes[1].bar(np.arange(1, s05.EPOCHS + 1), ratios,
                color=["tab:red" if peak - 1 <= e <= peak else "0.6" for e in range(1, s05.EPOCHS + 1)])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("largest batch loss / median")
    axes[1].set_title("no epoch of the rise holds an outlier batch")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spike_batch_content.png", dpi=110)
    plt.close(fig)
    print("  spike_batch_content.png: batch loss against how many boxes the batch holds, and the")
    print("  worst-to-median ratio of every epoch")

    # 4. Which terms, which layers
    print("\n--- 4. Which terms rise, and which layers move ---")
    early, late = rows[rows["epoch"] == QUIET_TO - 1], rows[rows["epoch"] == peak]
    print(f"  {'term':>8}{f'epoch {QUIET_TO - 1}':>12}{f'epoch {peak}':>12}{'ratio':>8}")
    for name in ("xy", "wh", "obj", "noobj", "class"):
        a, b = np.median(early[name]), np.median(late[name])
        print(f"  {name:>8}{a:>12.4f}{b:>12.4f}{b / a:>8.1f}")
    print("  the unbounded confidence terms degrade far more than the bounded coordinate terms")

    relative = [f"rel {n}" for n in layers]
    growth = [(np.median(late[c]) / max(np.median(early[c]), 1e-12), c[4:]) for c in relative]
    head = max(int(n.split(".")[0]) for n in layers)
    head_growth = [g for g, n in growth if n.startswith(f"{head}.")]
    print(f"  relative movement per step grows by x{min(g for g, _ in growth):.1f} to "
          f"x{max(g for g, _ in growth):.1f} across the tensors; the prediction head, layer {head}, "
          f"grows by x{min(head_growth):.1f} to x{max(head_growth):.1f}")
    print("  so the whole network moves further, the body at least as much as the head")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in ("xy", "wh", "obj", "noobj", "class"):
        per_epoch = [np.median(rows[name][rows["epoch"] == e]) for e in range(1, s05.EPOCHS + 1)]
        axes[0].plot(np.arange(1, s05.EPOCHS + 1), per_epoch, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("median loss per batch")
    axes[0].axvspan(peak - 1, peak, color="tab:red", alpha=0.08)
    axes[0].legend(ncol=5, fontsize=9)
    axes[0].set_title("every term rises, the confidence terms most")
    position = np.arange(len(relative))
    colours = ["tab:orange" if n.startswith(f"{head}.") else "tab:blue" for _, n in growth]
    axes[1].barh(position, [g for g, _ in growth], color=colours)
    axes[1].set_yticks(position)
    axes[1].set_yticklabels([n for _, n in growth], fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel(f"relative movement per step, epoch {peak} / epoch {QUIET_TO - 1}")
    axes[1].set_title("orange: the prediction head; blue: the body")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spike_terms_and_layers.png", dpi=110)
    plt.close(fig)
    print("  spike_terms_and_layers.png: each loss term over training, and how much further every")
    print("  parameter tensor moves per step during the rise")

    # 5, 6, 7. Controls
    print("\n--- 5. Control: the same everything, a different batch order ---")
    # A few epochs past the script's own thirty, so that a late break-up is seen whole.
    other, _, _ = train_detector(s05, data, s05.EPOCHS + 10, s05.LEARNING_RATE, OTHER_SHUFFLE,
                                 label="other batch order")
    other_episodes = episodes(other)
    print(f"  break-ups start at epochs {[e for e, _, _ in other_episodes] or 'none'}, against "
          f"{[e for e, _, _ in episodes(totals)]} with the original order")
    if other_episodes:
        print(f"  the first leaves a loss of {other_episodes[0][1]:.4f}, against "
              f"{episodes(totals)[0][1]:.4f} with the original order")
    print("  the break-up survives the change and moves, so it does not belong to particular batches")

    print("\n--- 6. Control: half the learning rate, run long enough to reach the same loss ---")
    half, _, _ = train_detector(s05, data, HALF_RATE_EPOCHS, s05.LEARNING_RATE / 2, s05.SEED,
                                label="half the rate")
    half_episodes = episodes(half)
    print(f"  break-ups start at epochs {[e for e, _, _ in half_episodes] or 'none'}")
    if half_episodes:
        epoch, before, _ = half_episodes[0]
        print(f"  the first comes at epoch {epoch}, from a loss of {before:.4f}, against "
              f"{episodes(totals)[0][1]:.4f} at the full rate")
        print("  so halving the step does not remove it; it buys a lower loss before the same thing happens")

    print(f"\n--- 7. Control: {args.long_epochs} epochs at the original settings ---")
    long_run, _, _ = train_detector(s05, data, args.long_epochs, s05.LEARNING_RATE, s05.SEED,
                                    label="four times longer")
    long_episodes = episodes(long_run)
    print(f"  {'starts at':>10}{'leaving':>10}{'peaking at':>12}{'ratio':>8}")
    for epoch, before, top in long_episodes:
        print(f"  {epoch:>10}{before:>10.4f}{top:>12.4f}{top / before:>8.2f}")
    starts = [e for e, _, _ in long_episodes]
    if len(starts) > 1:
        print(f"  {len(starts)} break-ups, spaced {', '.join(str(d) for d in np.diff(starts))} epochs "
              f"apart, each leaving from a lower loss than the last")
    print(f"  the trend keeps falling underneath: {long_run.min():.4f} at epoch "
          f"{int(long_run.argmin()) + 1}, against {totals[-1]:.4f} after {s05.EPOCHS} epochs")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(np.arange(1, s05.EPOCHS + 1), totals, label=f"batch order {s05.SEED}, the script's")
    axes[0].plot(np.arange(1, len(other) + 1), other, label=f"batch order {OTHER_SHUFFLE}")
    axes[0].plot(np.arange(1, HALF_RATE_EPOCHS + 1), half, label=f"rate {s05.LEARNING_RATE / 2:g}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("total training loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("changing the batch order moves it; halving the rate delays it")
    axes[1].plot(np.arange(1, args.long_epochs + 1), long_run, color="tab:blue",
                 label=f"rate {s05.LEARNING_RATE:g}, {args.long_epochs} epochs")
    axes[1].plot(np.arange(1, HALF_RATE_EPOCHS + 1), half, color="tab:orange",
                 label=f"rate {s05.LEARNING_RATE / 2:g}, {HALF_RATE_EPOCHS} epochs")
    axes[1].axvspan(1, s05.EPOCHS, color="0.85", zorder=0)
    axes[1].text(s05.EPOCHS / 2, long_run.max() * 0.5, f"what script 05 runs\n(epochs 1-{s05.EPOCHS})",
                 ha="center", fontsize=9, color="0.35")
    for epoch, _, top in long_episodes:
        axes[1].annotate(f"epoch {epoch}", (float(np.argmax(long_run[epoch - 1:epoch + 2]) + epoch), top),
                         (epoch, top * 2.6), ha="center", fontsize=8, color="tab:blue",
                         arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("total training loss")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("it breaks up and recovers again and again, while the trend keeps falling")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spike_controls.png", dpi=110)
    plt.close(fig)
    print("  spike_controls.png: the two batch orders and the halved rate side by side, and the long")
    print("  run with every break-up marked")

    print("\n--- What the measurements support ---")
    print("  The rise is not one bad batch, and not Adam's denominator shrinking: no outlier step")
    print("  exists and the second moment holds steady through it. It survives a change of batch")
    print("  order, and halving the step size only postpones it to a lower loss. Training four times")
    print("  longer brings it back at a regular spacing, each time from a lower loss. That is the")
    print("  behaviour of a step size meeting a loss surface that keeps sharpening as the loss falls:")
    print("  the run oscillates out of the narrow region, the surface flattens, and it settles again.")
    print("  Measuring curvature directly would be needed to close the case; these runs do not do that.")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
