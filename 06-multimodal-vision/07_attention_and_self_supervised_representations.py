"""Write attention out by hand, then learn image representations with no labels and score them.

Demonstrates the transformer block and three label-free training objectives:
    1. Compute scaled dot-product attention from Q, K and V, and check it against PyTorch.
    2. Put it inside an encoder block with residual connections and layer normalisation.
    3. Cut an image into tokens two ways, by patches and by convolution, and classify with labels.
    4. Train a supervised convolutional baseline as the reference the label-free methods are read against.
    5. Train an autoencoder on reconstruction alone and probe its representation with one linear layer.
    6. Train a masked autoencoder that rebuilds the patches it was not shown, and probe it the same way.
    7. Train a contrastive encoder on pairs of augmentations, with and without a projection head.
    8. Put every representation on the same scale: one linear layer, the same probe, the same data.

Module 06: Multimodal Vision - Attention and Self-Supervised Representations.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "attention_and_representations"
SEED = 3407

SIZE = 32
PATCH = 4
CLASSES = ("bar", "box", "cross", "disk", "ring", "triangle", "wedge", "zigzag")
TRAIN_IMAGES, TEST_IMAGES = 12000, 2000

EMBED, HEADS, DEPTH, MLP_RATIO = 96, 4, 3, 2
SUPERVISED_EPOCHS, BATCH, LEARNING_RATE = 8, 128, 1e-3
AE_EPOCHS, MAE_EPOCHS, CONTRASTIVE_EPOCHS = 12, 30, 20
PROBE_EPOCHS, PROBE_BATCH, PROBE_LEARNING_RATE = 5, 256, 1e-3
MASK_RATIO, TEMPERATURE, CONTRASTIVE_BATCH = 0.5, 0.2, 256


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def render_object(label, rng):
    """One 32x32 RGB image of the given class, on textured ground, in a random colour and pose."""
    noise = cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE, 3)).astype(np.float32), (0, 0), 2.0)
    img = np.clip(110 + 35 * noise / noise.std(), 0, 255)
    colour = tuple(float(v) for v in rng.integers(20, 256, 3))
    cx, cy = rng.integers(12, SIZE - 12, 2)
    size = int(rng.integers(12, 20))
    angle = float(rng.uniform(0, 180))
    name = CLASSES[label]
    if name == "bar":
        cv2.fillPoly(img, [np.rint(cv2.boxPoints(((float(cx), float(cy)), (float(size * 1.6), 4.0), angle))).astype(np.int32)], colour)
    elif name == "box":
        cv2.rectangle(img, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), colour, -1)
    elif name == "cross":
        cv2.line(img, (cx - size // 2, cy), (cx + size // 2, cy), colour, 3)
        cv2.line(img, (cx, cy - size // 2), (cx, cy + size // 2), colour, 3)
    elif name == "disk":
        cv2.circle(img, (int(cx), int(cy)), size // 2, colour, -1)
    elif name == "ring":
        cv2.circle(img, (int(cx), int(cy)), size // 2, colour, 2)
    elif name == "triangle":
        cv2.fillPoly(img, [np.array([(cx, cy - size // 2), (cx - size // 2, cy + size // 2),
                                     (cx + size // 2, cy + size // 2)], np.int32)], colour)
    elif name == "wedge":
        cv2.ellipse(img, (int(cx), int(cy)), (size // 2, size // 2), angle, 0, 120, colour, -1)
    else:
        points = [(cx - size // 2 + i * size // 4, cy + (-1) ** i * size // 4) for i in range(5)]
        cv2.polylines(img, [np.array(points, np.int32)], False, colour, 2)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def synthetic_dataset(count, rng):
    """Balanced labels and images for the rendered classes."""
    labels = np.arange(count) % len(CLASSES)
    rng.shuffle(labels)
    return np.stack([render_object(int(label), rng) for label in labels]), labels


def load_cifar10(root, train_count, test_count, rng):
    """CIFAR-10 images as 32x32 BGR arrays, from the python batches.

    Both splits are drawn from the first two training batches; the official test_batch
    is not read. The held-out split is fine for comparing the methods here with each
    other, but its accuracies are not CIFAR-10 test-set figures.
    """
    def read(name):
        with open(Path(root) / name, "rb") as handle:
            batch = pickle.load(handle, encoding="bytes")
        images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)[..., ::-1]
        return np.ascontiguousarray(images), np.array(batch[b"labels"], np.int64)

    images, labels = read("data_batch_1")
    more, more_labels = read("data_batch_2")
    images, labels = np.concatenate([images, more]), np.concatenate([labels, more_labels])
    order = rng.permutation(len(images))
    train, test = order[:train_count], order[train_count:train_count + test_count]
    return images[train], labels[train], images[test], labels[test]


# ---------------------------------------------------------------------------
# 1-2. Attention by hand
# ---------------------------------------------------------------------------

def attention_by_hand(q, k, v, heads):
    """Scaled dot-product attention, split across heads, written out in matrix form.

    Every token's query is compared with every token's key by a dot product, which
    is large when the two point the same way. Dividing by the square root of the
    head dimension keeps those dot products from growing with the dimension and
    pushing the softmax into a corner. The softmax turns each row into weights that
    sum to one, and the output of a token is that weighted sum of the values.
    """
    import torch

    b, n, d = q.shape
    head_dim = d // heads
    split = lambda t: t.view(b, n, heads, head_dim).transpose(1, 2)
    scores = split(q) @ split(k).transpose(-2, -1) / np.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    out = (weights @ split(v)).transpose(1, 2).reshape(b, n, d)
    return out, weights


def build_blocks():
    """A transformer encoder block whose attention is the function above."""
    import torch
    import torch.nn as nn

    class SelfAttention(nn.Module):
        """One linear layer produces Q, K and V; another mixes the heads' outputs back together."""

        def __init__(self, dim=EMBED, heads=HEADS):
            super().__init__()
            self.heads = heads
            self.qkv = nn.Linear(dim, 3 * dim)
            self.project = nn.Linear(dim, dim)

        def forward(self, x, return_weights=False):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            out, weights = attention_by_hand(q, k, v, self.heads)
            out = self.project(out)
            return (out, weights) if return_weights else out

    class EncoderBlock(nn.Module):
        """Attention and a feed-forward network, each added back onto its input and normalised."""

        def __init__(self, dim=EMBED, heads=HEADS, ratio=MLP_RATIO):
            super().__init__()
            self.attention = SelfAttention(dim, heads)
            self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, dim))

        def forward(self, x):
            x = self.norm1(x + self.attention(x))
            return self.norm2(x + self.mlp(x))

    return SelfAttention, EncoderBlock


# ---------------------------------------------------------------------------
# 3. Tokenisers and classifiers
# ---------------------------------------------------------------------------

def build_models(classes):
    """Every network this script trains, all reading 32x32x3 images."""
    import torch
    import torch.nn as nn

    SelfAttention, EncoderBlock = build_blocks()

    class PatchTokens(nn.Module):
        """Cut the image into PATCH x PATCH squares and project each one to a token."""

        def __init__(self, dim=EMBED):
            super().__init__()
            self.project = nn.Conv2d(3, dim, PATCH, stride=PATCH)
            self.positions = nn.Parameter(torch.zeros(1, (SIZE // PATCH) ** 2, dim))
            nn.init.normal_(self.positions, std=0.02)

        def forward(self, x):
            return self.project(x).flatten(2).transpose(1, 2) + self.positions

    class ConvolutionalTokens(nn.Module):
        """Three strided convolutions instead of one cut, so neighbouring tokens overlap."""

        def __init__(self, dim=EMBED):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, dim // 4, 3, stride=1, padding=1), nn.BatchNorm2d(dim // 4), nn.ReLU(),
                nn.Conv2d(dim // 4, dim // 2, 3, stride=2, padding=1), nn.BatchNorm2d(dim // 2), nn.ReLU(),
                nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1), nn.BatchNorm2d(dim), nn.ReLU())
            self.positions = nn.Parameter(torch.zeros(1, (SIZE // PATCH) ** 2, dim))
            nn.init.normal_(self.positions, std=0.02)

        def forward(self, x):
            return self.stem(x).flatten(2).transpose(1, 2) + self.positions

    class Transformer(nn.Module):
        """Tokens, DEPTH encoder blocks, then attention pooling into one vector per image."""

        def __init__(self, tokeniser, dim=EMBED, depth=DEPTH):
            super().__init__()
            self.tokeniser = tokeniser
            self.blocks = nn.Sequential(*[EncoderBlock(dim) for _ in range(depth)])
            self.pool = nn.Linear(dim, 1)
            self.head = nn.Linear(dim, classes)

        def features(self, x):
            tokens = self.blocks(self.tokeniser(x))
            weights = torch.softmax(self.pool(tokens), dim=1)
            return (weights * tokens).sum(1)

        def forward(self, x):
            return self.head(self.features(x))

    class ConvNet(nn.Module):
        """Five convolutions and a classifier: the supervised baseline."""

        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                nn.Flatten())
            self.head = nn.Linear(256, classes)

        def features(self, x):
            return self.body(x)

        def forward(self, x):
            return self.head(self.body(x))

    class Autoencoder(nn.Module):
        """Convolutional encoder and decoder trained to put the image back together."""

        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1))

        def features(self, x):
            return self.encoder(x).mean(dim=(2, 3))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class MaskedAutoencoder(nn.Module):
        """Encode the visible patches only, then rebuild the hidden ones from their positions.

        The encoder never sees a masked patch, so it cannot copy it; the only route
        to a low loss is a representation of the visible patches that says enough
        about the picture to predict the rest of it.
        """

        def __init__(self, dim=EMBED, depth=DEPTH):
            super().__init__()
            self.tokens = (SIZE // PATCH) ** 2
            self.project = nn.Linear(3 * PATCH * PATCH, dim)
            self.positions = nn.Parameter(torch.zeros(1, self.tokens, dim))
            nn.init.normal_(self.positions, std=0.02)
            self.encoder = nn.Sequential(*[EncoderBlock(dim) for _ in range(depth)])
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.decoder = nn.Sequential(*[EncoderBlock(dim) for _ in range(2)])
            self.predict = nn.Linear(dim, 3 * PATCH * PATCH)

        def to_patches(self, x):
            b = x.shape[0]
            side = SIZE // PATCH
            patches = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
            return patches.permute(0, 2, 3, 1, 4, 5).reshape(b, side * side, -1)

        def encode(self, patches, keep=None):
            tokens = self.project(patches) + self.positions
            return self.encoder(tokens if keep is None else tokens.gather(
                1, keep[..., None].expand(-1, -1, tokens.shape[-1])))

        def features(self, x):
            return self.encode(self.to_patches(x)).mean(1)

        def forward(self, x, keep, masked):
            patches = self.to_patches(x)
            encoded = self.encode(patches, keep)
            full = self.mask_token.expand(x.shape[0], self.tokens, -1) + self.positions
            full = full.scatter(1, keep[..., None].expand(-1, -1, encoded.shape[-1]), encoded)
            predicted = self.predict(self.decoder(full))
            target = patches.gather(1, masked[..., None].expand(-1, -1, patches.shape[-1]))
            return predicted.gather(1, masked[..., None].expand(-1, -1, predicted.shape[-1])), target

    class ContrastiveNet(nn.Module):
        """The same convolutional body with a projection head used only during training."""

        def __init__(self, projection=True):
            super().__init__()
            self.body = ConvNet().body
            self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128)) if projection \
                else nn.Identity()

        def features(self, x):
            return self.body(x)

        def forward(self, x):
            return self.head(self.body(x))

    return {"patch tokens": lambda: Transformer(PatchTokens()),
            "convolutional tokens": lambda: Transformer(ConvolutionalTokens()),
            "supervised ConvNet": ConvNet, "autoencoder": Autoencoder,
            "masked autoencoder": MaskedAutoencoder,
            "contrastive with head": lambda: ContrastiveNet(True),
            "contrastive without head": lambda: ContrastiveNet(False)}


# ---------------------------------------------------------------------------
# 4-7. Training objectives
# ---------------------------------------------------------------------------

def batches(count, batch, generator):
    """Shuffled index batches for one epoch."""
    import torch

    order = torch.randperm(count, generator=generator)
    return [order[start:start + batch] for start in range(0, count, batch)]


def train_supervised(model, x, y, epochs, device, generator):
    """Cross-entropy training, used for the classifiers and the baseline."""
    import torch
    import torch.nn.functional as F

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for idx in batches(len(x), BATCH, generator):
            loss = F.cross_entropy(model(x[idx].to(device)), y[idx].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    return time.perf_counter() - started


def train_autoencoder(model, x, epochs, device, generator):
    """Mean squared error between the reconstruction and the standardised input."""
    import torch
    import torch.nn.functional as F

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    started, last = time.perf_counter(), 0.0
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in batches(len(x), BATCH, generator):
            images = x[idx].to(device)
            loss = F.mse_loss(model(images), images)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        last = total / len(x)
    return time.perf_counter() - started, last


def train_masked_autoencoder(model, x, epochs, device, generator):
    """Reconstruct only the patches the encoder was not shown."""
    import torch
    import torch.nn.functional as F

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tokens = model.tokens
    visible = int(round(tokens * (1 - MASK_RATIO)))
    started, last = time.perf_counter(), 0.0
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in batches(len(x), BATCH, generator):
            images = x[idx].to(device)
            shuffle = torch.argsort(torch.rand(len(idx), tokens, device=device), dim=1)
            keep, masked = shuffle[:, :visible], shuffle[:, visible:]
            predicted, target = model(images, keep, masked)
            loss = F.mse_loss(predicted, target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        last = total / len(x)
    return time.perf_counter() - started, last


def augment(images):
    """A random view: crop and resize, horizontal flip, colour jitter, sometimes greyscale.

    The draws come from torch's global generator, seeded once in main, on the images' device.
    """
    import torch
    import torch.nn.functional as F

    b = images.shape[0]
    scale = 0.6 + 0.4 * torch.rand(b, device=images.device)
    offset = (1 - scale)[:, None] * (2 * torch.rand(b, 2, device=images.device) - 1)
    flip = (torch.rand(b, device=images.device) < 0.5).float() * -2 + 1
    theta = torch.zeros(b, 2, 3, device=images.device)
    theta[:, 0, 0] = scale * flip
    theta[:, 1, 1] = scale
    theta[:, :, 2] = offset
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    out = F.grid_sample(images, grid, align_corners=False, padding_mode="reflection")
    out = out * (0.6 + 0.8 * torch.rand(b, 1, 1, 1, device=images.device))
    out = out + 0.2 * (2 * torch.rand(b, 3, 1, 1, device=images.device) - 1)
    grey = out.mean(1, keepdim=True).expand_as(out)
    take_grey = (torch.rand(b, 1, 1, 1, device=images.device) < 0.2).float()
    return take_grey * grey + (1 - take_grey) * out


def nt_xent(z1, z2, temperature=TEMPERATURE):
    """Normalised temperature-scaled cross-entropy: each view must pick its partner out of the batch.

    Both views are L2-normalised, so a dot product is a cosine. Every view is scored
    against all 2N - 1 others, its partner is the single positive, and the loss is
    the cross-entropy of that choice. Dividing by a small temperature sharpens the
    distribution, which puts the pressure on the hardest negatives.
    """
    import torch
    import torch.nn.functional as F

    z = F.normalize(torch.cat([z1, z2]), dim=1)
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(-torch.inf)
    n = len(z1)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(similarity, targets)


def train_contrastive(model, x, epochs, device, generator):
    """Two augmentations per image per step, pulled together against everything else in the batch."""
    import torch

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    started, last = time.perf_counter(), 0.0
    for _ in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for idx in batches(len(x), CONTRASTIVE_BATCH, generator):
            if len(idx) < 4:
                continue
            images = x[idx].to(device)
            loss = nt_xent(model(augment(images)), model(augment(images)))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        last = total / max(seen, 1)
    return time.perf_counter() - started, last


def representations(model, x, device, batch=500):
    """Features from a frozen encoder."""
    import torch

    model.eval()
    with torch.no_grad():
        return torch.cat([model.features(x[start:start + batch].to(device)).cpu()
                          for start in range(0, len(x), batch)])


def linear_probe(train_features, train_labels, test_features, test_labels, classes, device, generator):
    """One linear layer on frozen features: how linearly separable the representation is."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    mean, std = train_features.mean(0, keepdim=True), train_features.std(0, keepdim=True) + 1e-6
    train_features = ((train_features - mean) / std).to(device)
    test_features = ((test_features - mean) / std).to(device)
    probe = nn.Linear(train_features.shape[1], classes).to(device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=PROBE_LEARNING_RATE)
    for _ in range(PROBE_EPOCHS):
        for idx in batches(len(train_features), PROBE_BATCH, generator):
            loss = F.cross_entropy(probe(train_features[idx]), train_labels[idx].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    with torch.no_grad():
        return (probe(test_features).argmax(1).cpu() == test_labels).float().mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cifar10-root", help="folder holding the CIFAR-10 python batches")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    # cuDNN chooses a convolution algorithm per shape and some of them accumulate in a
    # non-deterministic order, which moves a probe accuracy by a point or two between runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(SEED)

    # 1. Attention
    print("--- 1. Scaled dot-product attention ---")
    torch.manual_seed(SEED)
    q, k, v = (torch.randn(2, 4, EMBED) for _ in range(3))
    out, weights = attention_by_hand(q, k, v, HEADS)
    print(f"  {q.shape[1]} tokens of {EMBED} values, {HEADS} heads of {EMBED // HEADS}: "
          f"Q, K, V {tuple(q.shape)} -> attention weights {tuple(weights.shape)} -> output {tuple(out.shape)}")
    print(f"  every row of the weights sums to {weights.sum(-1).min().item():.6f}; "
          f"first head, first token attends {np.round(weights[0, 0, 0].numpy(), 3).tolist()}")
    show = lambda t: np.round(t[0, 0, :6].numpy(), 3).tolist()
    print(f"  first six values of the first token: Q {show(q)}")
    print(f"                                       K {show(k)}")
    print(f"                                       V {show(v)}")
    print(f"                                  output {show(out)}")
    reference = nn.MultiheadAttention(EMBED, HEADS, batch_first=True, bias=False)
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.eye(EMBED).repeat(3, 1))
        reference.out_proj.weight.copy_(torch.eye(EMBED))
        theirs, _ = reference(q, k, v, need_weights=False)
    print(f"  against nn.MultiheadAttention with identity projections: largest gap "
          f"{(out - theirs).abs().max().item():.2e}")
    print("  Dividing by the square root of the head dimension keeps the dot products from")
    print("  growing with the dimension; without it the softmax saturates and the gradient")
    print("  through it vanishes.")

    # 2. Encoder block
    print("\n--- 2. The encoder block around it ---")
    _, EncoderBlock = build_blocks()
    block = EncoderBlock()
    x = torch.randn(2, 4, EMBED)
    y = block(x)
    print(f"  input {tuple(x.shape)} -> output {tuple(y.shape)}, "
          f"{sum(p.numel() for p in block.parameters())} parameters")
    print(f"  input mean {x.mean().item():+.4f} sd {x.std().item():.4f}; "
          f"output mean {y.mean().item():+.4f} sd {y.std().item():.4f} (layer normalisation sets these)")
    identity = EncoderBlock()
    with torch.no_grad():
        for parameter in identity.mlp.parameters():
            parameter.zero_()
        for parameter in identity.attention.project.parameters():
            parameter.zero_()
    print(f"  with both sub-layers zeroed, the block returns its normalised input: largest gap "
          f"{(identity(x) - identity.norm2(identity.norm1(x))).abs().max().item():.2e}.")
    print("  The residual connections keep a direct path for the input through each sub-layer.")
    print("  With the normalisation after each addition, that path is LayerNorm(LayerNorm(x)),")
    print("  not x itself: this block does not start as the identity function.")

    # Data
    print("\n--- 3. Tokenising an image two ways ---")
    if args.cifar10_root:
        train_images, train_labels, test_images, test_labels = load_cifar10(
            args.cifar10_root, TRAIN_IMAGES, TEST_IMAGES, rng)
        classes, source = 10, (f"CIFAR-10: {len(train_images)} training and {len(test_images)} held-out images, "
                               f"both from the official training batches")
    else:
        train_images, train_labels = synthetic_dataset(TRAIN_IMAGES, rng)
        test_images, test_labels = synthetic_dataset(TEST_IMAGES, rng)
        classes = len(CLASSES)
        source = (f"rendered objects in {classes} classes: {len(train_images)} training and "
                  f"{len(test_images)} test images (pass --cifar10-root for CIFAR-10)")
    cv2.imwrite(str(OUT_DIR / "dataset.png"), np.hstack(list(train_images[:16])))
    mean = train_images.reshape(-1, 3).mean(0) / 255
    std = train_images.reshape(-1, 3).std(0) / 255
    to_tensor = lambda images: ((torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
                                 - torch.tensor(mean, dtype=torch.float32)[:, None, None])
                                / torch.tensor(std, dtype=torch.float32)[:, None, None])
    x_train, x_test = to_tensor(train_images), to_tensor(test_images)
    y_train, y_test = torch.from_numpy(train_labels).long(), torch.from_numpy(test_labels).long()
    print(f"  {source}")

    models = build_models(classes)
    print(f"  {'tokeniser':<24}{'tokens':>8}{'parameters':>12}{'time':>8}{'test accuracy':>15}")
    for name in ("patch tokens", "convolutional tokens"):
        model = models[name]().to(device)
        with torch.no_grad():
            tokens = model.tokeniser(x_test[:1].to(device)).shape[1]
        elapsed = train_supervised(model, x_train, y_train, SUPERVISED_EPOCHS, device, generator)
        model.eval()
        with torch.no_grad():
            accuracy = (torch.cat([model(x_test[s:s + 500].to(device)).argmax(1).cpu()
                                   for s in range(0, len(x_test), 500)]) == y_test).float().mean().item()
        print(f"  {name:<24}{tokens:>8}{sum(p.numel() for p in model.parameters()):>12}"
              f"{elapsed:>7.0f}s{accuracy:>15.2%}")
    print(f"  {SUPERVISED_EPOCHS} epochs each, {DEPTH} encoder blocks of {EMBED} values and {HEADS} heads.")
    print("  Cutting the image into squares gives every token one patch and nothing of its")
    print("  neighbours; a convolutional stem overlaps them, so a token already carries local")
    print("  structure before attention starts relating tokens to each other.")

    # 4-7. Representations
    print("\n--- 4. A supervised baseline, and three objectives that use no labels ---")
    results = {}
    baseline = models["supervised ConvNet"]().to(device)
    elapsed = train_supervised(baseline, x_train, y_train, SUPERVISED_EPOCHS, device, generator)
    results["supervised ConvNet"] = (elapsed, None, linear_probe(
        representations(baseline, x_train, device), y_train,
        representations(baseline, x_test, device), y_test, classes, device, generator))
    print(f"  supervised ConvNet trained in {elapsed:.0f}s")

    autoencoder = models["autoencoder"]().to(device)
    elapsed, loss = train_autoencoder(autoencoder, x_train, AE_EPOCHS, device, generator)
    results["autoencoder"] = (elapsed, loss, linear_probe(
        representations(autoencoder, x_train, device), y_train,
        representations(autoencoder, x_test, device), y_test, classes, device, generator))
    print(f"  autoencoder: {AE_EPOCHS} epochs, final reconstruction MSE {loss:.4f}")
    with torch.no_grad():
        sample = x_test[:8].to(device)
        rebuilt = autoencoder(sample).cpu()
    restore = lambda t: np.clip(255 * (t.permute(0, 2, 3, 1).numpy() * std + mean), 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "autoencoder.png"),
                np.vstack([np.hstack(list(restore(sample.cpu()))), np.hstack(list(restore(rebuilt)))]))

    masked = models["masked autoencoder"]().to(device)
    elapsed, loss = train_masked_autoencoder(masked, x_train, MAE_EPOCHS, device, generator)
    results["masked autoencoder"] = (elapsed, loss, linear_probe(
        representations(masked, x_train, device), y_train,
        representations(masked, x_test, device), y_test, classes, device, generator))
    print(f"  masked autoencoder: {MAE_EPOCHS} epochs at mask ratio {MASK_RATIO}, "
          f"{int((SIZE // PATCH) ** 2 * (1 - MASK_RATIO))} of {(SIZE // PATCH) ** 2} patches encoded, "
          f"final MSE on the hidden patches {loss:.4f}")

    for name in ("contrastive with head", "contrastive without head"):
        model = models[name]().to(device)
        elapsed, loss = train_contrastive(model, x_train, CONTRASTIVE_EPOCHS, device, generator)
        results[name] = (elapsed, loss, linear_probe(
            representations(model, x_train, device), y_train,
            representations(model, x_test, device), y_test, classes, device, generator))
        print(f"  {name}: {CONTRASTIVE_EPOCHS} epochs, batch {CONTRASTIVE_BATCH}, "
              f"temperature {TEMPERATURE}, final NT-Xent {loss:.4f}")
    cv2.imwrite(str(OUT_DIR / "augmentations.png"), np.vstack([
        np.hstack(list(restore(x_test[:8]))),
        np.hstack(list(restore(augment(x_test[:8].to(device)).cpu()))),
        np.hstack(list(restore(augment(x_test[:8].to(device)).cpu())))]))

    # 8. The same probe on every representation
    print("\n--- 8. One linear layer on every frozen representation ---")
    chance = 1 / classes
    print(f"  {PROBE_EPOCHS} epochs of a single linear layer on standardised features, "
          f"{len(x_train)} training and {len(x_test)} held-out images; the probe itself uses the labels")
    print(f"  {'representation':<26}{'labels used':>12}{'training time':>15}{'probe accuracy':>16}")
    for name, (elapsed, _, accuracy) in results.items():
        used = "all" if name == "supervised ConvNet" else "none"
        print(f"  {name:<26}{used:>12}{elapsed:>14.0f}s{accuracy:>16.2%}")
    print(f"  {'guessing':<26}{'-':>12}{'-':>15}{chance:>16.2%}")
    print("  'labels used' refers to training the encoder; every probe is trained on the labels.")
    print("  The three label-free encoders differ in architecture, feature size and epochs as well")
    print("  as in objective, so the ordering is of these configurations, not of the objectives alone.")
    print("  It is consistent with reconstruction rewarding whatever fills the most pixels, while")
    print("  the other two ask for something harder: predicting patches the encoder never saw, and")
    print("  telling two views of one image apart from every other image.")
    print("  The two contrastive runs share the body architecture and every training setting, and")
    print("  the projection head is their one design difference; each still starts from its own")
    print("  random weights, augmentations and batch order, so it is not a strict single-variable")
    print("  ablation. In this setup the head substantially improves the features underneath it: the loss")
    print("  pulls the head's output onto a sphere and discards what it does not need, and the")
    print("  body is one layer removed from that.")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
