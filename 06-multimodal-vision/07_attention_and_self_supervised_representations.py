"""Write attention out by hand, then learn image representations with no labels and score them.

Demonstrates the transformer block and three label-free training objectives:
    1. Compute scaled dot-product attention from Q, K and V, and check it against PyTorch.
    2. Put it inside an encoder block with residual connections and layer normalisation.
    3. Cut an image into tokens two ways, by patches and by convolution, classify with labels,
       and probe what each tokeniser's frozen representation is worth on its own.
    4. Train a supervised convolutional baseline as the reference the label-free methods are read against.
    5. Train an autoencoder on reconstruction alone and probe its representation with one linear layer.
    6. Train a masked autoencoder that rebuilds the patches it was not shown, and probe it the same way.
    7. Train a contrastive encoder on pairs of augmentations, with and without a projection head.
    8. Score six encoders two ways: a frozen linear probe, and the same fine-tuning for all of them.

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

OUT_BASE = Path(__file__).parent / "outputs" / "attention_and_representations"
# A run on photographs and a run on the rendered objects write the same file names, so
# each gets its own folder and neither overwrites the other.
OUT_DIR = OUT_BASE / "synthetic"
SEED = 3407

SIZE = 32
PATCH = 4
CLASSES = ("bar", "box", "cross", "disk", "ring", "triangle", "wedge", "zigzag")
CIFAR_CLASSES = ("aeroplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
TRAIN_IMAGES, TEST_IMAGES = 12000, 2000

EMBED, HEADS, DEPTH, MLP_RATIO = 96, 4, 3, 2
SUPERVISED_EPOCHS, BATCH, LEARNING_RATE = 8, 128, 1e-3
AE_EPOCHS, MAE_EPOCHS, CONTRASTIVE_EPOCHS = 12, 30, 20
PROBE_EPOCHS, PROBE_BATCH, PROBE_LEARNING_RATE = 5, 256, 1e-3
# Fine-tuning deliberately borrows the supervised settings, so that the six encoders meet the
# downstream task on exactly the terms the supervised baseline was trained on.
FINETUNE_EPOCHS, FINETUNE_BATCH, FINETUNE_LEARNING_RATE = SUPERVISED_EPOCHS, BATCH, LEARNING_RATE
MASK_RATIO, TEMPERATURE, CONTRASTIVE_BATCH = 0.5, 0.2, 256
# The six encoders compared head to head in step 8. The projection-head ablation of step 7 is
# deliberately left out: it is a second run of one of these six, not a seventh design.
COMPARED = ("patch tokens", "convolutional tokens", "supervised ConvNet", "autoencoder",
            "masked autoencoder", "contrastive with head")
LABELLED_PRETRAINING = ("patch tokens", "convolutional tokens", "supervised ConvNet")


# Data

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


# 1-2. Attention by hand

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


# 3. Tokenisers and classifiers

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


# 4-7. Training objectives

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
    started, history = time.perf_counter(), []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for idx in batches(len(x), BATCH, generator):
            loss = F.cross_entropy(model(x[idx].to(device)), y[idx].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        history.append(total / len(x))
    return time.perf_counter() - started, history


def train_autoencoder(model, x, epochs, device, generator):
    """Mean squared error between the reconstruction and the standardised input."""
    import torch
    import torch.nn.functional as F

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    started, history = time.perf_counter(), []
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
        history.append(total / len(x))
    return time.perf_counter() - started, history[-1], history


def train_masked_autoencoder(model, x, epochs, device, generator):
    """Reconstruct only the patches the encoder was not shown."""
    import torch
    import torch.nn.functional as F

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tokens = model.tokens
    visible = int(round(tokens * (1 - MASK_RATIO)))
    started, history = time.perf_counter(), []
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
        history.append(total / len(x))
    return time.perf_counter() - started, history[-1], history


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
    started, history = time.perf_counter(), []
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
        history.append(total / max(seen, 1))
    return time.perf_counter() - started, history[-1], history


def labelled(image, text):
    """A BGR uint8 image with its label on a black strip in the top-left corner."""
    image = image.copy()
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(image, (2, 2), (10 + text_w, 10 + text_h), (0, 0, 0), -1)
    cv2.putText(image, text, (6, 6 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return image


def image_rows(rows, scale=4):
    """Rows of 32x32 BGR images, enlarged, each row labelled on its first image, grey dividers between rows."""
    stacked = []
    for name, images in rows:
        line = np.hstack([cv2.resize(image, (SIZE * scale, SIZE * scale), interpolation=cv2.INTER_NEAREST)
                          for image in images])
        stacked += [labelled(line, name), np.full((4, line.shape[1], 3), 128, np.uint8)]
    return np.vstack(stacked[:-1])


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


def fine_tune(encoder, x_train, y_train, x_test, y_test, classes, device, generator):
    """Unfreeze a trained encoder, put a fresh linear classifier on it, and train the two together.

    This asks a different question from the probe. The probe freezes the encoder and measures how
    linearly separable its representation already is; this lets the encoder keep learning from the
    labels and measures what the whole system reaches after adapting to the task.

    The encoder is deep-copied, so the representation the probe scored is left exactly as it was,
    and the classifier is a new layer rather than the task head the encoder may already carry. Every
    encoder gets the same labelled data, epochs, batch size, optimiser, learning rate and objective;
    only the width of its representation differs. Parts that the representation does not use - an
    autoencoder's decoder, a contrastive projection head, an original classification head - stay in
    the copy but receive no gradient, so they never move.
    """
    import copy

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    model = copy.deepcopy(encoder)
    model.eval()
    with torch.no_grad():
        width = model.features(x_train[:1].to(device)).shape[1]
    head = nn.Linear(width, classes).to(device)
    optimiser = torch.optim.Adam(list(model.parameters()) + list(head.parameters()),
                                 lr=FINETUNE_LEARNING_RATE)
    started, history = time.perf_counter(), []
    for _ in range(FINETUNE_EPOCHS):
        model.train()
        total = 0.0
        for idx in batches(len(x_train), FINETUNE_BATCH, generator):
            loss = F.cross_entropy(head(model.features(x_train[idx].to(device))), y_train[idx].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        history.append(total / len(x_train))
    model.eval()
    with torch.no_grad():
        predicted = torch.cat([head(model.features(x_test[start:start + 500].to(device))).argmax(1).cpu()
                               for start in range(0, len(x_test), 500)])
    return time.perf_counter() - started, history, (predicted == y_test).float().mean().item(), width


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cifar10-root", help="folder holding the CIFAR-10 python batches")
    args = parser.parse_args()
    global OUT_DIR
    OUT_DIR = OUT_BASE / ("cifar10" if args.cifar10_root else "synthetic")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    import torch
    import torch.nn as nn
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

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
    fig, axes = plt.subplots(1, HEADS, figsize=(3.3 * HEADS, 3.9))
    for head, ax in enumerate(axes):
        grid = weights[0, head].numpy()
        ax.imshow(grid, cmap="viridis", vmin=0, vmax=1)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        ax.set_title(f"head {head + 1}", fontsize=10)
        ax.set_xlabel("key token")
        ax.set_ylabel("query token")
        ax.set_xticks(range(grid.shape[1]))
        ax.set_yticks(range(grid.shape[0]))
    fig.suptitle("attention weights of the first sequence, per head: every row sums to 1")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attention_weights.png", dpi=110)
    plt.close(fig)
    print("  attention_weights.png: the 4x4 weight matrix of every head for the first sequence")

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
    with torch.no_grad():
        zeroed_out = identity(x)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].hist(x.flatten().numpy(), bins=60, color="grey")
    axes[0].set_title(f"input values: mean {x.mean().item():+.3f}, sd {x.std().item():.3f}", fontsize=10)
    axes[1].hist(y.detach().flatten().numpy(), bins=60, color="tab:blue")
    axes[1].set_title(f"block output: mean {y.mean().item():+.3f}, sd {y.std().item():.3f}", fontsize=10)
    axes[2].scatter(x.flatten().numpy(), zeroed_out.flatten().numpy(), s=3, alpha=0.4)
    axes[2].plot([-3, 3], [-3, 3], "k--", linewidth=1, label="identity")
    axes[2].set_xlabel("input value")
    axes[2].set_ylabel("output value")
    axes[2].set_title("sub-layers zeroed: LN(LN(x)) is not x", fontsize=10)
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "encoder_block.png", dpi=110)
    plt.close(fig)
    print("  encoder_block.png: input and output value distributions, and the zeroed block against the identity")

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
    names_of = CIFAR_CLASSES if args.cifar10_root else CLASSES
    fig, axes = plt.subplots(2, 8, figsize=(12, 4.3))
    for ax, image, label in zip(axes.ravel(), train_images[:16], train_labels[:16]):
        ax.imshow(image[..., ::-1], interpolation="nearest")
        ax.set_title(names_of[int(label)], fontsize=9)
        ax.axis("off")
    fig.suptitle("the first 16 training images: " + ("CIFAR-10" if args.cifar10_root else "rendered objects"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset.png", dpi=110)
    plt.close(fig)
    mean = train_images.reshape(-1, 3).mean(0) / 255
    std = train_images.reshape(-1, 3).std(0) / 255
    to_tensor = lambda images: ((torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
                                 - torch.tensor(mean, dtype=torch.float32)[:, None, None])
                                / torch.tensor(std, dtype=torch.float32)[:, None, None])
    x_train, x_test = to_tensor(train_images), to_tensor(test_images)
    y_train, y_test = torch.from_numpy(train_labels).long(), torch.from_numpy(test_labels).long()
    print(f"  {source}")

    models = build_models(classes)
    print(f"  {'tokeniser':<24}{'tokens':>8}{'parameters':>12}{'time':>8}{'end-to-end':>13}{'linear probe':>14}")
    token_accuracy, loss_histories, encoders = {}, {}, {}
    probe_accuracy, pretrain_time = {}, {}
    for name in ("patch tokens", "convolutional tokens"):
        model = models[name]().to(device)
        with torch.no_grad():
            tokens = model.tokeniser(x_test[:1].to(device)).shape[1]
        elapsed, loss_histories[name] = train_supervised(model, x_train, y_train, SUPERVISED_EPOCHS, device, generator)
        model.eval()
        with torch.no_grad():
            accuracy = (torch.cat([model(x_test[s:s + 500].to(device)).argmax(1).cpu()
                                   for s in range(0, len(x_test), 500)]) == y_test).float().mean().item()
        # The probe added here must not disturb the random stream the later objectives draw from,
        # so it runs on a forked generator of its own and the outer stream resumes untouched.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            probe_generator = torch.Generator().manual_seed(SEED)
            probe_accuracy[name] = linear_probe(
                representations(model, x_train, device), y_train,
                representations(model, x_test, device), y_test, classes, device, probe_generator)
        print(f"  {name:<24}{tokens:>8}{sum(p.numel() for p in model.parameters()):>12}"
              f"{elapsed:>7.0f}s{accuracy:>13.2%}{probe_accuracy[name]:>14.2%}")
        token_accuracy[name] = accuracy
        encoders[name], pretrain_time[name] = model, elapsed
    print(f"  {SUPERVISED_EPOCHS} epochs each, {DEPTH} encoder blocks of {EMBED} values and {HEADS} heads.")
    print("  Cutting the image into squares gives every token one patch and nothing of its")
    print("  neighbours; a convolutional stem overlaps them, so a token already carries local")
    print("  structure before attention starts relating tokens to each other.")
    print("  Two different questions sit in those last two columns. End-to-end accuracy is what the")
    print("  whole classifier reaches on its own task, its original head included. The linear probe")
    print(f"  throws that head away, freezes everything else, and trains one fresh Linear({EMBED} -> "
          f"{classes})")
    print("  on the pooled features: how linearly separable the representation already is. They are")
    print("  not the same measurement and should not be read as one number being a corrected version")
    print("  of the other.")
    # A 3x3 convolution at stride 1 and two at stride 2 give each output token a 9x9 input
    # window centred on 4 * (its row or column), so neighbouring windows overlap by 5 pixels.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, name in zip(axes, ("patch tokens", "convolutional tokens")):
        ax.imshow(test_images[0][..., ::-1], extent=(0, SIZE, SIZE, 0), interpolation="nearest")
        for k in range(0, SIZE + 1, PATCH):
            ax.axhline(k, color="w", linewidth=0.4)
            ax.axvline(k, color="w", linewidth=0.4)
        for column, colour in ((3, "red"), (4, "yellow")):
            if name == "patch tokens":
                corner, side = (column * PATCH, 3 * PATCH), PATCH
            else:
                corner, side = (column * PATCH - 4, 3 * PATCH - 4), 9
            ax.add_patch(Rectangle(corner, side, side, fill=False, edgecolor=colour, linewidth=2.5))
        window = PATCH if name == "patch tokens" else 9
        ax.set_title(f"{name}: two neighbouring tokens see {window}x{window} px each\n"
                     f"test accuracy {token_accuracy[name]:.2%}", fontsize=10)
        ax.set_xlim(0, SIZE)
        ax.set_ylim(SIZE, 0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tokenisers.png", dpi=110)
    plt.close(fig)
    print("  tokenisers.png: the input window of two neighbouring tokens under each tokeniser;")
    print("  dataset.png: the first 16 training images with their classes")

    # 4-7. Representations
    print("\n--- 4. A supervised baseline, and three objectives that use no labels ---")
    results = {}
    baseline = models["supervised ConvNet"]().to(device)
    elapsed, loss_histories["supervised ConvNet"] = train_supervised(baseline, x_train, y_train, SUPERVISED_EPOCHS,
                                                                     device, generator)
    results["supervised ConvNet"] = (elapsed, None, linear_probe(
        representations(baseline, x_train, device), y_train,
        representations(baseline, x_test, device), y_test, classes, device, generator))
    encoders["supervised ConvNet"] = baseline
    print(f"  supervised ConvNet trained in {elapsed:.0f}s")

    autoencoder = models["autoencoder"]().to(device)
    elapsed, loss, loss_histories["autoencoder"] = train_autoencoder(autoencoder, x_train, AE_EPOCHS, device,
                                                                     generator)
    results["autoencoder"] = (elapsed, loss, linear_probe(
        representations(autoencoder, x_train, device), y_train,
        representations(autoencoder, x_test, device), y_test, classes, device, generator))
    encoders["autoencoder"] = autoencoder
    print(f"  autoencoder: {AE_EPOCHS} epochs, final reconstruction MSE {loss:.4f}")
    with torch.no_grad():
        sample = x_test[:8].to(device)
        rebuilt = autoencoder(sample).cpu()
    restore = lambda t: np.clip(255 * (t.permute(0, 2, 3, 1).numpy() * std + mean), 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / "autoencoder.png"), image_rows([
        ("input", list(restore(sample.cpu()))), ("reconstruction", list(restore(rebuilt)))]))
    print("  autoencoder.png: eight test images above their reconstructions")

    masked = models["masked autoencoder"]().to(device)
    elapsed, loss, loss_histories["masked autoencoder"] = train_masked_autoencoder(masked, x_train, MAE_EPOCHS,
                                                                                   device, generator)
    results["masked autoencoder"] = (elapsed, loss, linear_probe(
        representations(masked, x_train, device), y_train,
        representations(masked, x_test, device), y_test, classes, device, generator))
    encoders["masked autoencoder"] = masked
    print(f"  masked autoencoder: {MAE_EPOCHS} epochs at mask ratio {MASK_RATIO}, "
          f"{int((SIZE // PATCH) ** 2 * (1 - MASK_RATIO))} of {(SIZE // PATCH) ** 2} patches encoded, "
          f"final MSE on the hidden patches {loss:.4f}")
    # The mask for the picture comes from a generator of its own, so drawing it takes
    # nothing from the global stream that the contrastive augmentations draw from next.
    mask_generator = torch.Generator().manual_seed(SEED)
    side = SIZE // PATCH
    to_image = lambda p: p.view(p.shape[0], side, side, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(
        p.shape[0], 3, SIZE, SIZE)
    with torch.no_grad():
        sample = x_test[:8].to(device)
        visible = int(round(masked.tokens * (1 - MASK_RATIO)))
        shuffle = torch.argsort(torch.rand(8, masked.tokens, generator=mask_generator), dim=1).to(device)
        keep, hidden = shuffle[:, :visible], shuffle[:, visible:]
        predicted, _ = masked(sample, keep, hidden)
        patches = masked.to_patches(sample)
        index = hidden[..., None].expand(-1, -1, patches.shape[-1])
        shown = patches.scatter(1, index, torch.zeros_like(predicted))   # zero is the mean colour
        rebuilt = patches.scatter(1, index, predicted)
    cv2.imwrite(str(OUT_DIR / "masked_autoencoder.png"), image_rows([
        ("input", list(restore(sample.cpu()))),
        (f"what the encoder sees: {visible} of {masked.tokens} patches", list(restore(to_image(shown).cpu()))),
        ("hidden patches filled in by the decoder", list(restore(to_image(rebuilt).cpu())))]))
    print("  masked_autoencoder.png: eight test images, the half the encoder is shown, and the")
    print("  hidden half as the decoder predicts it")

    for name in ("contrastive with head", "contrastive without head"):
        model = models[name]().to(device)
        elapsed, loss, loss_histories[name] = train_contrastive(model, x_train, CONTRASTIVE_EPOCHS, device, generator)
        results[name] = (elapsed, loss, linear_probe(
            representations(model, x_train, device), y_train,
            representations(model, x_test, device), y_test, classes, device, generator))
        encoders[name] = model
        print(f"  {name}: {CONTRASTIVE_EPOCHS} epochs, batch {CONTRASTIVE_BATCH}, "
              f"temperature {TEMPERATURE}, final NT-Xent {loss:.4f}")
    cv2.imwrite(str(OUT_DIR / "augmentations.png"), image_rows([
        ("original", list(restore(x_test[:8]))),
        ("view 1", list(restore(augment(x_test[:8].to(device)).cpu()))),
        ("view 2", list(restore(augment(x_test[:8].to(device)).cpu())))]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in ("patch tokens", "convolutional tokens", "supervised ConvNet"):
        axes[0].plot(np.arange(1, len(loss_histories[name]) + 1), loss_histories[name], "o-", label=name)
    axes[0].set_title("supervised: cross-entropy", fontsize=10)
    for name in ("autoencoder", "masked autoencoder"):
        axes[1].plot(np.arange(1, len(loss_histories[name]) + 1), loss_histories[name], "o-", label=name)
    axes[1].set_title("reconstruction: MSE (masked autoencoder on hidden patches only)", fontsize=10)
    for name in ("contrastive with head", "contrastive without head"):
        axes[2].plot(np.arange(1, len(loss_histories[name]) + 1), loss_histories[name], "o-", label=name)
    axes[2].set_title(f"contrastive: NT-Xent over batches of {CONTRASTIVE_BATCH}", fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_losses.png", dpi=110)
    plt.close(fig)
    print("  augmentations.png: eight test images and two random views of each;")
    print("  training_losses.png: every training loss per epoch, grouped by objective")

    # 8. The same two evaluations on every representation
    print("\n--- 8. Six encoders, two evaluations ---")
    chance = 1 / classes
    probe_accuracy.update({name: accuracy for name, (_, _, accuracy) in results.items()})
    pretrain_time.update({name: elapsed for name, (elapsed, _, _) in results.items()})
    widths = {name: representations(model, x_train[:1], device).shape[1] for name, model in encoders.items()}

    print("\n  A. Frozen linear probe: freeze the encoder, train only a new linear classifier.")
    print("     It asks how linearly separable the representation the encoder already learned is.")
    print(f"     {PROBE_EPOCHS} epochs of a single Linear(width -> {classes}) on features standardised by the")
    print(f"     training mean and standard deviation, {len(x_train)} training and {len(x_test)} held-out images.")
    print(f"  {'representation':<26}{'labels in pretraining':>23}{'feature width':>15}{'probe accuracy':>16}")
    for name in COMPARED:
        used = "all" if name in LABELLED_PRETRAINING else "none"
        print(f"  {name:<26}{used:>23}{widths[name]:>15}{probe_accuracy[name]:>16.2%}")
    print(f"  {'guessing':<26}{'-':>23}{'-':>15}{chance:>16.2%}")

    print("\n  B. Fine-tuning: unfreeze the encoder, put a fresh linear classifier on it, train both.")
    print("     It asks what the whole system reaches once it is allowed to adapt to the labelled task.")
    print(f"     {FINETUNE_EPOCHS} epochs, batch {FINETUNE_BATCH}, Adam {FINETUNE_LEARNING_RATE}, cross-entropy, "
          f"the same split for all six.")
    # Fine-tuning runs last and on a generator of its own, so nothing above it moves.
    finetune_generator = torch.Generator().manual_seed(SEED)
    fine_tuned, finetune_histories = {}, {}
    print(f"  {'representation':<26}{'feature width':>15}{'fine-tuning time':>18}{'accuracy':>12}")
    for name in COMPARED:
        elapsed, finetune_histories[name], accuracy, width = fine_tune(
            encoders[name], x_train, y_train, x_test, y_test, classes, device, finetune_generator)
        fine_tuned[name] = accuracy
        print(f"  {name:<26}{width:>15}{elapsed:>17.0f}s{accuracy:>12.2%}")

    print("\n  Side by side, with what each encoder's own pretraining cost:")
    print(f"  {'representation':<26}{'pretraining':>13}{'probe':>10}{'fine-tuned':>13}{'gain, points':>14}")
    for name in COMPARED:
        difference = 100 * (fine_tuned[name] - probe_accuracy[name])
        print(f"  {name:<26}{pretrain_time[name]:>12.0f}s{probe_accuracy[name]:>10.2%}"
              f"{fine_tuned[name]:>13.2%}{difference:>+14.2f}")
    print("\n  Three numbers in this script must not be run together into one ranking:")
    print("    end-to-end accuracy (step 3) - what a classifier reaches on its own task, its own head included")
    print("    probe accuracy             - how linearly separable a frozen representation already is")
    print("    fine-tuned accuracy        - what the encoder reaches after adapting to the labels")
    print("  The six encoders meet the same downstream protocol, but their pretraining objectives,")
    print("  architectures, feature widths, epochs and exposure to labels all differ, so this compares")
    print("  these configurations, not the objectives in isolation, and it is not a causal ranking of")
    print("  training methods. Three of the six saw the class labels during pretraining; the other three")
    print("  did not, and only meet a label at the probe or the fine-tuning head.")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    names = list(COMPARED) + ["contrastive without head"]
    values = [100 * probe_accuracy[name] for name in names]
    colours = ["tab:grey" if name in LABELLED_PRETRAINING else "tab:blue" for name in names]
    bars = ax.bar(names, values, color=colours)
    bars[-1].set_hatch("//")        # the ablation, not one of the six compared
    for i, value in enumerate(values):
        ax.text(i, value + 1, f"{value:.1f}%", ha="center")
    ax.axhline(100 * chance, color="tab:red", linestyle="--", label=f"guessing {100 * chance:.1f}%")
    ax.set_ylabel("linear probe accuracy on held-out images, %")
    ax.set_ylim(0, 110)
    ax.set_title("one linear layer on each frozen representation (grey: labels seen in pretraining; "
                 "hatched: the step 7 ablation)", fontsize=10)
    ax.legend()
    plt.setp(ax.get_xticklabels(), fontsize=8, rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "probe_accuracy.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    position = np.arange(len(COMPARED))
    probe_values = [100 * probe_accuracy[name] for name in COMPARED]
    tuned_values = [100 * fine_tuned[name] for name in COMPARED]
    ax.bar(position - 0.2, probe_values, 0.4, color="tab:blue", label="frozen linear probe")
    ax.bar(position + 0.2, tuned_values, 0.4, color="tab:orange", label="end-to-end fine-tuning")
    for x, value in zip(position - 0.2, probe_values):
        ax.text(x, value + 1, f"{value:.1f}", ha="center", fontsize=8)
    for x, value in zip(position + 0.2, tuned_values):
        ax.text(x, value + 1, f"{value:.1f}", ha="center", fontsize=8)
    ax.axhline(100 * chance, color="tab:red", linestyle="--",
               label=f"guessing {100 * chance:.1f}%")
    ax.set_xticks(position)
    ax.set_xticklabels([f"{name}\n{widths[name]}-d" for name in COMPARED], fontsize=8)
    ax.set_ylabel("accuracy on held-out images, %")
    ax.set_ylim(0, 132)          # headroom for the legend, since two bars reach 100
    ax.set_title("the same six encoders under both evaluations; the probe scores the representation, "
                 "fine-tuning scores the adapted model", fontsize=10)
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "representation_evaluation.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in COMPARED:
        ax.plot(np.arange(1, FINETUNE_EPOCHS + 1), finetune_histories[name], "o-", label=name)
    ax.set_xlabel("fine-tuning epoch")
    ax.set_ylabel("cross-entropy on the training set")
    ax.set_yscale("log")
    ax.set_title(f"the same downstream fine-tuning for all six encoders, {FINETUNE_EPOCHS} epochs", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fine_tuning_losses.png", dpi=110)
    plt.close(fig)
    print("\n  probe_accuracy.png: the probe accuracy of all seven configurations against guessing")
    print("  representation_evaluation.png: the six compared encoders under the probe and under fine-tuning")
    print("  fine_tuning_losses.png: the downstream cross-entropy of all six, epoch by epoch")
    print("\n  'labels in pretraining' refers to training the encoder; the probe and the fine-tuning head")
    print("  are trained on labels in every row, and the held-out split is only ever read for scoring:")
    print("  the standardising mean and standard deviation come from the training features alone, and")
    print("  no setting here was chosen by looking at a held-out number.")
    print("  The three label-free training families differ in architecture, feature size and epochs as well")
    print("  as in objective, so the ordering is of these configurations, not of the objectives alone.")
    print("  It is consistent with reconstruction rewarding whatever fills the most pixels, while")
    print("  the other two ask for something harder: predicting patches the encoder never saw, and")
    print("  telling two views of one image apart from every other image.")
    print("\n  The projection-head ablation of step 7 stays out of the six above, being a second run of")
    print("  one of them rather than a seventh design; its probe accuracy is the hatched bar:")
    print(f"  contrastive with a projection head {probe_accuracy['contrastive with head']:.2%} against "
          f"{probe_accuracy['contrastive without head']:.2%} without one, a gap of "
          f"{100 * (probe_accuracy['contrastive with head'] - probe_accuracy['contrastive without head']):.1f} "
          f"points.")
    print("  The two contrastive runs share the body architecture and every training setting, and")
    print("  the projection head is their one design difference; each still starts from its own")
    print("  random weights, augmentations and batch order, so it is not a strict single-variable")
    print("  ablation. In this setup the head substantially improves the features underneath it: the loss")
    print("  pulls the head's output onto a sphere and discards what it does not need, and the")
    print("  body is one layer removed from that.")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
