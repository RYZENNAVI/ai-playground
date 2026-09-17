"""Measure the curvature of the detector's loss surface, epoch by epoch, through the break-up.

Script 05b established what the rise near epoch 25 is not: not one unlucky batch, not Adam's
denominator shrinking, not particular batches at all, and not something a smaller step removes.
It ended on an inference - that a fixed step is meeting a loss surface which keeps sharpening -
that its own measurements could not check, because it never measured curvature. This script does:
    1. Reproduce the training run of script 05 exactly, and fix one probe set to measure on.
    2. After every epoch, estimate the sharpest curvature of the loss there, without ever
       building the Hessian: Hessian-vector products driven by power iteration.
    3. Measure two curvatures beside it - along the direction the parameters actually moved,
       and after Adam's own preconditioner, which is the one its step size answers to.
    4. Report what the numbers do and do not support.

Nothing here trains differently from script 05. The measurement takes gradients but never a
step, restores the batch-norm statistics its forward passes would otherwise move, and draws its
random vectors from a generator of its own, so the trajectory is the one 05 and 05b report.

Module 06: Multimodal Vision - Grid Detection and Pose Assembly, a supplement to script 05.
"""

import argparse
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "grid_detection_and_pose"
DETECTOR_SOURCE = Path(__file__).parent / "05_grid_detection_and_pose_assembly.py"
PROBE_SIZE, POWER_ITERATIONS = 128, 12
QUIET_EPOCHS = 4        # how many settled epochs before a break-up to read as its baseline
# A break-up takes a few hundred steps inside one or two epochs, so measuring only at epoch
# boundaries cannot see whether curvature moves while it happens. These epochs are measured
# inside as well, which is where the question is actually decided.
FINE_WINDOW, FINE_EVERY = (22, 27), 25
# Script 05b found the half-rate run breaks up near epoch 56, so it has to run past that.
HALF_RATE_EPOCHS = 60


def load_detector_module():
    """Import script 05 by path, since a module name cannot start with a digit."""
    spec = importlib.util.spec_from_file_location("detector_source", DETECTOR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.argv = [str(DETECTOR_SOURCE)]
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


def batch_norm_buffers(model):
    """Every running statistic, so a measurement can put them back exactly as it found them."""
    return {name: buffer.detach().clone() for name, buffer in model.named_buffers()}


def restore_buffers(model, saved):
    import torch

    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(saved[name])


def probe_gradient(model, loss_of, parameters, create_graph=False):
    """The gradient of the probe loss, summed over the probe's batches.

    The probe is split into the same batch size the detector trains with, both to keep the
    memory of a double backward pass reasonable and so that batch normalisation sees batches
    of the size it saw in training.
    """
    import torch

    total = None
    for make_loss in loss_of:
        grads = torch.autograd.grad(make_loss(), parameters, create_graph=create_graph)
        flat = torch.cat([g.flatten() for g in grads])
        total = flat if total is None else total + flat
    return total / len(loss_of)


def hessian_vector_product(model, loss_of, parameters, vector):
    """H times a vector, by differentiating the gradient's projection onto it.

    Never forms the Hessian. The first backward pass is taken with the graph kept, so the
    second one can differentiate through it; each probe batch is released before the next.
    """
    import torch

    total = None
    for make_loss in loss_of:
        grads = torch.autograd.grad(make_loss(), parameters, create_graph=True)
        flat = torch.cat([g.flatten() for g in grads])
        product = torch.autograd.grad((flat * vector).sum(), parameters)
        piece = torch.cat([p.flatten() for p in product])
        total = piece if total is None else total + piece
        del grads, flat, product
    return total / len(loss_of)


def power_iteration(model, loss_of, parameters, iterations, generator, scale=None):
    """The largest eigenvalue of H, or of a symmetrically scaled version of it.

    Power iteration converges to the eigenvalue of largest magnitude; the Rayleigh quotient of
    the final vector gives it with its sign, so a negative return would mean the sharpest
    direction here curves downward. With scale set to a per-parameter vector s, this measures
    diag(s) H diag(s) instead, which is symmetric and has the eigenvalues of the preconditioned
    operator diag(s^2) H.
    """
    import torch

    width = sum(p.numel() for p in parameters)
    vector = torch.randn(width, generator=generator).to(parameters[0].device)
    vector /= vector.norm()
    estimate, history = 0.0, []
    for _ in range(iterations):
        product = hessian_vector_product(
            model, loss_of, parameters, vector if scale is None else vector * scale)
        if scale is not None:
            product = product * scale
        estimate = float(torch.dot(vector, product))
        history.append(estimate)
        norm = product.norm()
        if norm < 1e-12:
            break
        vector = product / norm
    # How far the last step still moved the estimate, as a check that it settled.
    drift = abs(history[-1] - history[-2]) / (abs(history[-1]) + 1e-12) if len(history) > 1 else float("nan")
    return estimate, drift, vector


def measure(s05, model, probe, anchors_t, optimiser, parameters, direction, generator, iterations):
    """Curvature at the current parameters, leaving the model and the training stream untouched."""
    import torch

    saved = batch_norm_buffers(model)
    was_training = model.training
    model.train()      # the surface the optimiser walks on is the one batch statistics define
    images, targets, boxes = probe
    loss_of = [(lambda i=i: sum(s05.yolo_loss(
        s05.reshape_head(model(images[i])), targets[i], boxes[i], anchors_t).values()))
        for i in range(len(images))]

    with torch.no_grad():
        losses = [float(make_loss()) for make_loss in loss_of]
    gradient = probe_gradient(model, loss_of, parameters)
    sharpest, drift, _ = power_iteration(model, loss_of, parameters, iterations, generator)

    # Adam's own scaling. Its step is lr * m / (sqrt(v_hat) + eps), so the curvature its step
    # size actually meets is that of the surface seen through diag(1 / sqrt(v_hat)).
    state = [optimiser.state[p] for p in parameters]
    step = max(int(state[0]["step"]) if "step" in state[0] else 1, 1)
    beta2 = optimiser.param_groups[0]["betas"][1]
    correction = 1 - beta2 ** step
    scale = torch.cat([(s["exp_avg_sq"].flatten() / correction).sqrt().add(
        optimiser.param_groups[0]["eps"]).reciprocal().sqrt() for s in state])
    preconditioned, pre_drift, _ = power_iteration(
        model, loss_of, parameters, iterations, generator, scale=scale)

    along = float("nan")
    if direction is not None:
        product = hessian_vector_product(model, loss_of, parameters, direction)
        along = float(torch.dot(direction, product))

    restore_buffers(model, saved)
    model.train(was_training)
    return {"probe loss": float(np.mean(losses)), "gradient norm": float(gradient.norm()),
            "lambda max": sharpest, "drift": drift, "preconditioned": preconditioned,
            "preconditioned drift": pre_drift, "along update": along}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe-size", type=int, default=PROBE_SIZE,
                        help=f"training images the curvature is measured on (default {PROBE_SIZE})")
    parser.add_argument("--power-iters", type=int, default=POWER_ITERATIONS,
                        help=f"power iterations per estimate (default {POWER_ITERATIONS})")
    parser.add_argument("--epochs", type=int, default=None,
                        help="epochs to train (default: the same as script 05)")
    parser.add_argument("--fine-window", type=int, nargs=2, default=list(FINE_WINDOW),
                        metavar=("FIRST", "LAST"),
                        help=f"epochs measured inside as well as at their end (default {FINE_WINDOW[0]} "
                             f"{FINE_WINDOW[1]})")
    parser.add_argument("--half-rate-epochs", type=int, default=HALF_RATE_EPOCHS,
                        help=f"epochs for the half-step-size control, 0 to skip "
                             f"(default {HALF_RATE_EPOCHS})")
    parser.add_argument("--fine-every", type=int, default=FINE_EVERY,
                        help=f"steps between measurements inside that window (default {FINE_EVERY})")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    s05 = load_detector_module()
    epochs = args.epochs or s05.EPOCHS
    print("--- 1. The training run of script 05, and one fixed probe set ---")
    print(f"  detector from {DETECTOR_SOURCE.name}: Adam {s05.LEARNING_RATE}, batch {s05.BATCH}, "
          f"{epochs} epochs, no schedule")
    train, anchors, targets = build_dataset(s05)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_t = torch.from_numpy(np.stack([i for i, _, _ in train])).permute(0, 3, 1, 2).contiguous()
    targets_t = torch.from_numpy(np.stack(targets))
    gt_t = [torch.from_numpy(b) for _, b, _ in train]
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)

    # The probe set is fixed once, before any training, and every measurement in either run sees
    # exactly these images: otherwise a change in curvature could not be told from a change of batch.
    probe_index = np.arange(args.probe_size)
    digest = hashlib.sha256(probe_index.tobytes()).hexdigest()[:16]
    probe = ([images_t[probe_index[s:s + s05.BATCH]].to(device).float().div(255)
              for s in range(0, len(probe_index), s05.BATCH)],
             [targets_t[probe_index[s:s + s05.BATCH]].to(device)
              for s in range(0, len(probe_index), s05.BATCH)],
             [[gt_t[i].to(device) for i in probe_index[s:s + s05.BATCH]]
              for s in range(0, len(probe_index), s05.BATCH)])
    steps_per_epoch = -(-len(train) // s05.BATCH)
    print(f"  probe set: training images {probe_index[0]}-{probe_index[-1]}, {len(probe_index)} of "
          f"{s05.TRAIN_IMAGES}, in {len(probe[0])} batches of {s05.BATCH}, sha256 {digest}")
    width = sum(p.numel() for p in s05.build_detector().parameters() if p.requires_grad)
    print(f"  curvature by Hessian-vector products and {args.power_iters} power iterations; the "
          f"Hessian is never formed")
    print(f"  {width} parameters, so the Hessian it stands for would hold "
          f"{width ** 2 / 1e12:.1f} trillion entries")

    def run(epochs, learning_rate, fine_window, label):
        """Train as script 05 does, measuring curvature at every epoch and inside a window."""
        torch.manual_seed(s05.SEED)
        model = s05.build_detector().to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(s05.SEED)
        # Measurements draw random vectors; a generator of their own keeps them out of the
        # training stream, so each trajectory is the one script 05 and 05b report.
        measure_generator = torch.Generator().manual_seed(s05.SEED)
        parameters = [p for p in model.parameters() if p.requires_grad]
        fine_first, fine_last = fine_window
        rows, fine_rows, started = [], [], time.perf_counter()
        print(f"\n  {label}: Adam {learning_rate}, batch {s05.BATCH}, {epochs} epochs, "
              f"{steps_per_epoch} steps to the epoch, no schedule")
        if fine_first <= fine_last:
            print(f"  epochs {fine_first} to {fine_last} are measured every {args.fine_every} "
                  f"steps as well")
        print(f"  {'epoch':>5}{'train loss':>12}{'probe loss':>12}{'lambda max':>12}{'drift':>9}"
              f"{'along step':>12}{'Adam-scaled':>13}{'|grad|':>9}{'|update|':>10}{'time':>8}")
        for epoch in range(1, epochs + 1):
            model.train()
            before = torch.cat([p.detach().flatten().clone() for p in parameters])
            inside = fine_first <= epoch <= fine_last
            recent = before.clone() if inside else None
            running = 0.0
            order = torch.randperm(len(train), generator=generator)
            for step, start in enumerate(range(0, len(order), s05.BATCH)):
                idx = order[start:start + s05.BATCH]
                x = images_t[idx].to(device).float().div(255)
                terms = s05.yolo_loss(s05.reshape_head(model(x)), targets_t[idx].to(device),
                                      [gt_t[i].to(device) for i in idx], anchors_t)
                loss = sum(terms.values())
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running += loss.item() * len(idx)
                if inside and (step + 1) % args.fine_every == 0:
                    now = torch.cat([p.detach().flatten() for p in parameters])
                    moved_recently = now - recent
                    recent = now.clone()
                    measured = measure(s05, model, probe, anchors_t, optimiser, parameters,
                                       moved_recently / moved_recently.norm().clamp_min(1e-12),
                                       measure_generator, args.power_iters)
                    measured.update({"epoch": epoch - 1 + (step + 1) / steps_per_epoch,
                                     "batch loss": loss.item(),
                                     "update norm": float(moved_recently.norm()) / args.fine_every})
                    fine_rows.append(measured)
            moved = torch.cat([p.detach().flatten() for p in parameters]) - before
            update_norm = float(moved.norm())
            measured = measure(s05, model, probe, anchors_t, optimiser, parameters,
                               moved / moved.norm().clamp_min(1e-12), measure_generator,
                               args.power_iters)
            measured.update({"epoch": epoch, "train loss": running / len(train),
                             "update norm": update_norm})
            rows.append(measured)
            print(f"  {epoch:>5}{measured['train loss']:>12.4f}{measured['probe loss']:>12.4f}"
                  f"{measured['lambda max']:>12.1f}{measured['drift']:>9.1e}"
                  f"{measured['along update']:>12.1f}{measured['preconditioned']:>13.1f}"
                  f"{measured['gradient norm']:>9.2f}{update_norm:>10.4f}"
                  f"{time.perf_counter() - started:>7.0f}s")
        return rows, fine_rows

    print(f"\n--- 2. Training, measuring curvature after every epoch ---")
    rows, fine_rows = run(epochs, s05.LEARNING_RATE, args.fine_window, "full rate")

    # 3. What the numbers say
    print("\n--- 3. Curvature against the break-up ---")
    column = lambda key: np.array([row[key] for row in rows])
    train_loss, lam = column("train loss"), column("lambda max")
    rise = [i + 1 for i in range(10, len(train_loss)) if train_loss[i] > 1.25 * train_loss[i - 1]]
    if not rise:
        print("  no epoch rose by more than a quarter over the one before it in this run")
        first = None
    else:
        first = rise[0]
        print(f"  the loss breaks up at epoch {first}, rising from {train_loss[first - 2]:.4f} "
              f"to {train_loss[first - 1]:.4f}")
    # The baseline is the settled stretch immediately before the break-up, so it stays meaningful
    # whatever epoch the break-up lands on and however long the run is.
    quiet_to = (first - 2) if first else len(rows)
    quiet_from = max(1, quiet_to - QUIET_EPOCHS + 1)
    quiet = slice(quiet_from - 1, quiet_to)
    print(f"  {'':<26}{f'quiet, ep {quiet_from}-{quiet_to}':>20}{'at the break-up':>18}{'ratio':>9}")
    for label, key in (("sharpest curvature", "lambda max"), ("curvature along the step", "along update"),
                       ("Adam-scaled curvature", "preconditioned"), ("probe loss", "probe loss")):
        values = column(key)
        before, at = float(np.median(values[quiet])), (float(values[first - 1]) if first else float("nan"))
        print(f"  {label:<26}{before:>20.3f}{at:>18.3f}{at / before if before else float('nan'):>9.2f}")
    if first:
        climb = lam[first - 1] / np.median(lam[quiet])
        print(f"  the sharpest curvature at the break-up is {climb:.2f}x its level over the quiet epochs")
        print(f"  its highest value before the break-up is {lam[:first - 1].max():.1f} at epoch "
              f"{int(lam[:first - 1].argmax()) + 1}, and it ends the run at {lam[-1]:.1f}")
    print(f"  the largest drift left in a power-iteration estimate is {column('drift')[1:].max():.1e} "
          f"relative, over {args.power_iters} iterations")

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axis = np.arange(1, epochs + 1)
    axes[0].plot(axis, train_loss, "o-", color="k", label="training loss")
    axes[0].plot(axis, column("probe loss"), "o-", color="0.6", label="probe-set loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[1].plot(axis, lam, "o-", color="tab:red")
    axes[1].set_ylabel("sharpest curvature\nlambda max(H)")
    axes[2].plot(axis, column("along update"), "o-", color="tab:purple")
    axes[2].set_ylabel("curvature along\nthe step taken")
    axes[3].plot(axis, column("preconditioned"), "o-", color="tab:green")
    axes[3].set_ylabel("curvature after\nAdam's scaling")
    axes[3].set_xlabel("epoch")
    for ax in axes:
        if first:
            ax.axvspan(quiet_from, quiet_to, color="tab:blue", alpha=0.07)
            ax.axvspan(first - 1, first + 1, color="tab:red", alpha=0.1)
        ax.grid(alpha=0.3)
    if first:
        axes[0].text(first, axes[0].get_ylim()[1] * 0.5, f"break-up\nepoch {first}", ha="center",
                     fontsize=8, color="tab:red")
    axes[0].legend(fontsize=9)
    axes[0].set_title("the loss surface under the detector, measured on one fixed probe set "
                      "(blue: quiet epochs, red: the break-up)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "curvature_over_training.png", dpi=110)
    plt.close(fig)
    print("  curvature_over_training.png: loss, sharpest curvature, curvature along the step taken,")
    print("  and curvature after Adam's scaling, epoch by epoch")

    if fine_rows:
        print(f"\n--- 4. Inside the break-up, every {args.fine_every} steps ---")
        fine = lambda key: np.array([row[key] for row in fine_rows])
        where = fine("epoch")
        print(f"  {'epoch':>8}{'batch loss':>12}{'probe loss':>12}{'lambda max':>12}"
              f"{'along step':>12}{'Adam-scaled':>13}{'step size':>11}")
        for row in fine_rows:
            print(f"  {row['epoch']:>8.2f}{row['batch loss']:>12.4f}{row['probe loss']:>12.4f}"
                  f"{row['lambda max']:>12.1f}{row['along update']:>12.2f}"
                  f"{row['preconditioned']:>13.1f}{row['update norm']:>11.5f}")
        if first:
            settled = where < first - 1
            during = (where >= first - 1) & (where <= first + 1)
            for label, key in (("sharpest curvature", "lambda max"),
                               ("curvature along the step", "along update"),
                               ("Adam-scaled curvature", "preconditioned")):
                level = float(np.median(fine(key)[settled])) if settled.any() else float("nan")
                peak = float(np.max(fine(key)[during])) if during.any() else float("nan")
                print(f"  {label:<26} before {level:>10.1f}, highest during the break-up "
                      f"{peak:>10.1f}, ratio {peak / level if level else float('nan'):>5.2f}")

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(where, fine("batch loss"), ".-", color="k", lw=0.9)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("loss of the batch\njust trained on")
        axes[1].plot(where, fine("lambda max"), ".-", color="tab:red", lw=0.9, label="lambda max(H)")
        axes[1].plot(where, fine("preconditioned"), ".-", color="tab:green", lw=0.9,
                     label="after Adam's scaling")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("curvature")
        axes[1].legend(fontsize=9)
        axes[2].plot(where, fine("along update"), ".-", color="tab:purple", lw=0.9)
        axes[2].set_ylabel("curvature along\nthe steps just taken")
        axes[2].set_xlabel("epoch")
        for ax in axes:
            if first:
                ax.axvspan(first - 1, first + 1, color="tab:red", alpha=0.1)
            ax.grid(alpha=0.3)
        axes[0].set_title(f"inside the break-up: a measurement every {args.fine_every} steps, "
                          f"epochs {args.fine_window[0]} to {args.fine_window[1]}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "curvature_inside_the_break_up.png", dpi=110)
        plt.close(fig)
        print(f"  curvature_inside_the_break_up.png: the same three curvatures at "
              f"{args.fine_every}-step resolution through the epochs where the loss leaves and returns")

    if args.half_rate_epochs:
        print(f"\n--- 5. The same measurement at half the step size ---")
        print("  Script 05b found that halving the rate does not remove the break-up but postpones")
        print("  it to about half the loss. If a smaller step can tolerate a sharper surface, the")
        print("  curvature it breaks up at should be the higher of the two.")
        half_rows, _ = run(args.half_rate_epochs, s05.LEARNING_RATE / 2, (1, 0), "half rate")
        half = lambda key: np.array([row[key] for row in half_rows])
        half_loss, half_lam = half("train loss"), half("lambda max")
        half_rise = [i + 1 for i in range(10, len(half_loss)) if half_loss[i] > 1.25 * half_loss[i - 1]]
        half_first = half_rise[0] if half_rise else None
        print(f"\n  {'run':<14}{'break-up':>10}{'loss it left':>14}{'lambda max before':>19}"
              f"{'Adam-scaled before':>20}")
        pairs = [("full rate", s05.LEARNING_RATE, rows, first, column),
                 ("half rate", s05.LEARNING_RATE / 2, half_rows, half_first, half)]
        summary = {}
        for label, rate, these, at_epoch, get in pairs:
            if at_epoch is None:
                print(f"  {label:<14}{'none':>10}")
                continue
            # The last reading before the rise, which is the state the break-up started from.
            before_lam = float(get("lambda max")[at_epoch - 2])
            before_pre = float(get("preconditioned")[at_epoch - 2])
            summary[label] = (at_epoch, float(get("train loss")[at_epoch - 2]), before_lam, before_pre)
            print(f"  {label:<14}{at_epoch:>10}{get('train loss')[at_epoch - 2]:>14.4f}"
                  f"{before_lam:>19.1f}{before_pre:>20.1f}")
        if len(summary) == 2:
            full_v, half_v = summary["full rate"], summary["half rate"]
            print(f"  halving the step buys {full_v[1] / half_v[1]:.2f}x lower loss before the "
                  f"break-up, and it starts from")
            print(f"  {half_v[2] / full_v[2]:.2f}x the sharpest curvature and "
                  f"{half_v[3] / full_v[3]:.2f}x the Adam-scaled curvature")
            # A stability boundary of the form (step size x curvature) would hold this product
            # fixed across step sizes. It is the gradient-descent analysis of a quadratic, and
            # Adam is neither, so it is read here as a diagnostic proxy and nothing stronger.
            print(f"\n  step size x curvature at the break-up, a GD-style stability proxy:")
            print(f"  {'run':<14}{'rate':>9}{'x lambda max':>15}{'x Adam-scaled':>16}")
            for label, rate, _, _, _ in pairs:
                if label in summary:
                    print(f"  {label:<14}{rate:>9.1e}{rate * summary[label][2]:>15.3f}"
                          f"{rate * summary[label][3]:>16.3f}")
            raw = (s05.LEARNING_RATE / 2 * half_v[2]) / (s05.LEARNING_RATE * full_v[2])
            scaled = (s05.LEARNING_RATE / 2 * half_v[3]) / (s05.LEARNING_RATE * full_v[3])
            print(f"  halving the rate leaves that product at {raw:.2f}x of its full-rate value on "
                  f"the raw curvature")
            print(f"  and {scaled:.2f}x on the Adam-scaled one, against a loss that differs by "
                  f"{full_v[1] / half_v[1]:.2f}x")

        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        full_x, half_x = np.arange(1, len(rows) + 1), np.arange(1, len(half_rows) + 1)
        both = ((full_x, column, "tab:blue", s05.LEARNING_RATE),
                (half_x, half, "tab:orange", s05.LEARNING_RATE / 2))
        for x, get, colour, rate in both:
            axes[0].plot(x, get("train loss"), "o-", ms=3, color=colour, label=f"rate {rate:g}")
            axes[1].plot(x, get("lambda max"), "o-", ms=3, color=colour)
            axes[2].plot(x, get("preconditioned"), "o-", ms=3, color=colour)
            axes[3].plot(x, rate * get("preconditioned"), "o-", ms=3, color=colour)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("training loss")
        axes[1].set_ylabel("sharpest curvature\nlambda max(H)")
        axes[2].set_ylabel("curvature after\nAdam's scaling")
        axes[3].set_ylabel("step size x that\ncurvature, a proxy")
        for ax in axes:
            for at_epoch, colour in ((first, "tab:blue"), (half_first, "tab:orange")):
                if at_epoch:
                    ax.axvline(at_epoch, color=colour, linestyle="--", lw=1)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=9)
        axes[0].set_title("the same detector at two step sizes, dashed where each breaks up",
                          fontsize=11)
        axes[3].set_xlabel("epoch   (the bottom panel is a diagnostic proxy, "
                           "not Adam's stability condition)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "curvature_lr_control.png", dpi=110)
        plt.close(fig)
        print("  curvature_lr_control.png: loss and sharpest curvature for both step sizes, with")
        print("  each run's break-up marked")

    print(f"\n--- What the curvature measurements support ---")
    if first:
        quiet_level, before_it, at_it = np.median(lam[quiet]), lam[first - 2], lam[first - 1]
        print(f"  The sharpest curvature sits near {quiet_level:.0f} over the quiet epochs, is "
              f"{before_it:.0f} by the epoch before the break-up and {at_it:.0f} at it, so it does "
              f"{'rise' if before_it > quiet_level else 'not rise'} into the event. The finer")
        print("  sampling above says the same from inside it. A version of the explanation in which")
        print("  curvature climbs until the step no longer fits is not what this run does.")
    if args.half_rate_epochs:
        print("  What the two step sizes show is the other half of it. Each run breaks up at its own")
        print("  loss, far apart, and the curvature it breaks up from is the higher one for the")
        print("  smaller step. Read as a step-size-times-curvature boundary, the two agree much")
        print("  better on the Adam-scaled curvature than on the raw one, and the Adam-scaled one is")
        print("  what Adam's own step size answers to. That is the prediction such a boundary makes,")
        print("  from two runs.")
    print("  Three things this does not establish. The Hessian here is of the smooth piece of the")
    print("  loss the detector sits on: the ignore mask is a threshold, so it holds one setting")
    print("  while the Hessian is taken and can jump between epochs. A boundary of this form is the")
    print("  analysis of gradient descent on a quadratic, and Adam on this loss is neither, so the")
    print("  product above is a diagnostic proxy and not Adam's stability condition. And two step")
    print("  sizes agreeing is a correlation across two runs; turning the step size down when the")
    print("  proxy climbs, and seeing the break-up not happen, is the experiment that would settle")
    print("  it, and this script does not run it.")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
