"""Run two embedding models locally and show why pooling choice matters.

Demonstrates what a wrapper hides when you encode text yourself:
    1. Score queries against documents with a CLS-pooling model.
    2. Score the same pairs with a mean-pooling model.
    3. Rebuild one of those scores by hand, from raw model outputs.
    4. Check the hand-built vectors match the wrapper's.
    5. Show what happens when the pooling strategy is wrong.

Module 02: RAG - Embedding Model Comparison.
"""

import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

# Two small models, chosen because they pool differently. Pooling is a property of
# the model, fixed when its authors trained it, so it lives here beside the name
# rather than being passed in at the call site.
PRIMARY, PRIMARY_POOLING = "BAAI/bge-small-en-v1.5", "cls"                    # 255 MB
SECONDARY, SECONDARY_POOLING = "iic/nlp_gte_sentence-embedding_english-small", "mean"  # 64 MB

MAX_LENGTH = 512

# Two questions and two passages, deliberately crossed: query 1 belongs to
# document 1, query 2 to document 2. Every step below is judged against that.
QUERIES = [
    "can I get a refund on a theme park ticket",
    "what perks does the annual pass include",
]

DOCUMENTS = [
    "Theme park tickets are normally non-refundable once sold. In exceptional "
    "cases, such as a park closure caused by severe weather, guests may "
    "reschedule or request a refund through the official channel.",
    "The annual pass comes in three tiers priced from 2399 to 4399 yuan. Holders "
    "get early park entry, discounts on merchandise and dining, and access to "
    "members-only events.",
]


def ensure_model(model_id):
    """Return a local path for the weights, downloading them only if missing.

    Kept in the module's own weights/ directory so repeated runs are free and
    the download never lands in a global cache you forget about.
    """
    import logging

    from modelscope import snapshot_download

    # The downloader logs a progress bar on every call, even when the files are
    # already present and nothing is transferred. Quiet it so the actual results
    # stay readable.
    logging.getLogger("modelscope_hub.download").setLevel(logging.WARNING)

    weights = Path(__file__).parent / "weights"
    weights.mkdir(exist_ok=True)
    return snapshot_download(model_id, cache_dir=str(weights))


def show_scores(title, scores):
    """Print a query-by-document grid, flagging the pair that should match."""
    print(f"\n{title}")
    for i, _ in enumerate(QUERIES):
        for j, _ in enumerate(DOCUMENTS):
            marker = "  <-- expected match" if i == j else ""
            print(f"  q{i + 1} x d{j + 1}: {scores[i][j]:7.4f}{marker}")


def encode_with_wrapper(model_id):
    """Steps 1-2: let SentenceTransformer handle tokenising, pooling, normalising.

    The wrapper reads each model's own config to pick the right pooling, which is
    exactly the detail step 3 has to reproduce by hand.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ensure_model(model_id), trust_remote_code=True)
    model.max_seq_length = MAX_LENGTH
    vectors = model.encode(QUERIES + DOCUMENTS, normalize_embeddings=True)
    n = len(QUERIES)
    return vectors[:n] @ vectors[n:].T


def pool(hidden, mask, how):
    """Reduce per-token states to one vector per text.

    A transformer emits one vector per token; a sentence embedding needs exactly
    one per text, and how you collapse them is model-specific:
      cls   - take token 0, the [CLS] slot the model was trained to summarise into
      mean  - average the real tokens, ignoring padding
      last  - take the final real token, used by decoder-only embedding models
    Using the wrong one still returns a plausible-looking vector, which is what
    makes this failure so easy to miss. Step 5 shows the damage.
    """
    import torch

    if how == "cls":
        return hidden[:, 0]
    if how == "mean":
        weights = mask.unsqueeze(-1).float()
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
    if how == "last":
        if mask[:, -1].sum() == mask.shape[0]:  # left padded
            return hidden[:, -1]
        lengths = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
    raise ValueError(f"unknown pooling: {how}")


_LOADED = {}


def load_model(model_id):
    """Load a tokenizer and model once, then hand out the same pair every time.

    encode_by_hand() runs three times (step 3, and twice in step 5), and without
    this each call would re-read the weights from disk. Harmless for a 255 MB
    model, painful for a multi-gigabyte one.
    """
    from modelscope import AutoModel, AutoTokenizer

    if model_id not in _LOADED:
        model_dir = ensure_model(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        model.eval()
        _LOADED[model_id] = (tokenizer, model)
    return _LOADED[model_id]


def encode_by_hand(model_id, how):
    """Step 3: tokenise, run the model, pool, and normalise, with nothing hidden."""
    import torch
    import torch.nn.functional as F

    tokenizer, model = load_model(model_id)

    batch = tokenizer(QUERIES + DOCUMENTS, max_length=MAX_LENGTH, padding=True,
                      truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state

    vectors = pool(hidden, batch["attention_mask"], how)
    # Normalising to unit length is what turns a dot product into cosine similarity.
    vectors = F.normalize(vectors, p=2, dim=1).numpy()
    n = len(QUERIES)
    return vectors[:n] @ vectors[n:].T


def compare(wrapper_scores, manual_scores):
    """Step 4: confirm the hand-built pipeline reproduces the wrapper.

    Agreement here is the proof that the pooling choice was right. If these
    diverge, the pooling strategy is the first thing to suspect.
    """
    print("\n--- 4. Does the hand-built pipeline match the wrapper? ---")
    largest = float(np.abs(np.array(wrapper_scores) - np.array(manual_scores)).max())
    print(f"  Largest absolute difference: {largest:.6f}")
    if largest < 0.01:
        print("  Match. The wrapper only saved us the tokenising and pooling code.")
    else:
        print("  Mismatch. Check the pooling strategy and the normalisation step.")


def show_wrong_pooling(model_id, right, wrong, reference):
    """Step 5: pool the same model the wrong way and see what actually changes.

    Nothing crashes, and the scores stay in a believable range. On a small sample
    the wrong pooling can even look better, which is the whole danger: the output
    alone cannot tell you it is wrong. The only dependable signal is the one from
    step 4, agreement with the model's own configuration.

    The two printed numbers serve opposite purposes:
      deviation - a correctness check against a trusted reference. It answers
                  "did I pool this model correctly", and nothing else.
      margin    - looks like a quality score but is not one: two samples is far
                  too few to be stable, and the scale differs from model to
                  model. Printed only to show that a familiar-looking number can
                  point the wrong way.

    Neither measures retrieval quality. That takes a few hundred labelled
    query-document pairs scored with Recall@K, MRR, or NDCG - or the published
    results at https://huggingface.co/spaces/mteb/leaderboard.
    """
    print("\n--- 5. What if the pooling is wrong? ---")
    correct_scores = encode_by_hand(model_id, right)
    wrong_scores = encode_by_hand(model_id, wrong)

    def deviation(scores):
        # Distance from the reference the wrapper produced.
        return float(np.abs(np.array(scores) - np.array(reference)).max())

    def margin(scores):
        # How far the correct document beats the wrong one, averaged over queries.
        return float(np.mean([scores[i][i] - scores[i][1 - i] for i in range(len(QUERIES))]))

    print(f"  {right:<4} pooling: deviation {deviation(correct_scores):.6f}, "
          f"margin {margin(correct_scores):+.4f}")
    print(f"  {wrong:<4} pooling: deviation {deviation(wrong_scores):.6f}, "
          f"margin {margin(wrong_scores):+.4f}")
    print("  Deviation is the reliable tell: the wrong pooling always drifts from")
    print("  the reference. Margin is not - it can go either way on a few samples,")
    print("  so a plausible-looking score is no evidence the pooling was right.")


def main():
    print(f"  primary  : {PRIMARY} ({PRIMARY_POOLING} pooling)")
    print(f"  secondary: {SECONDARY} ({SECONDARY_POOLING} pooling)")

    cls_scores = encode_with_wrapper(PRIMARY)
    show_scores(f"--- 1. {PRIMARY} ({PRIMARY_POOLING} pooling) ---", cls_scores)

    mean_scores = encode_with_wrapper(SECONDARY)
    show_scores(f"--- 2. {SECONDARY} ({SECONDARY_POOLING} pooling) ---", mean_scores)

    manual = encode_by_hand(PRIMARY, PRIMARY_POOLING)
    show_scores("--- 3. Same model, encoded by hand ---", manual)

    compare(cls_scores, manual)
    other = "mean" if PRIMARY_POOLING != "mean" else "cls"
    show_wrong_pooling(PRIMARY, PRIMARY_POOLING, other, cls_scores)


if __name__ == "__main__":
    main()
