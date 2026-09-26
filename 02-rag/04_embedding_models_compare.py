"""This script scores two questions, one about a ticket refund and one about annual
pass perks, against two passages that answer them, using three embedding models that
pool differently. A transformer returns one vector per token, and pooling turns
them into one vector per text. Each model is trained with its own pooling. CLS pooling
takes the first token, and mean pooling averages the real tokens. Last-token pooling
takes the final real token: in a decoder-only model each token sees only the tokens
before it, so only the last one has seen the whole text. The wrong pooling raises no
error, so the script rebuilds every score by hand and checks it against the
SentenceTransformer wrapper.

The run prints five parts:
    1. CLS pooling. BAAI/bge-small-en-v1.5 through the wrapper. Each question scores
       highest against its own passage.
    2. Mean pooling. A small GTE model through the wrapper, with the same result.
    3. Last-token pooling. Qwen3-Embedding-0.6B, a decoder-only model, through the
       wrapper, with the same result.
    4. By hand. Tokenising, pooling and normalising written out for each model, and
       the largest difference from the wrapper's scores. Qwen3 runs with right and
       with left padding, because the last real token sits in a different place.
       Every difference stays below 0.01.
    5. Wrong pooling. bge with mean pooling. The scores move away from the wrapper's,
       yet the gap between the right and wrong passage grows, so the scores alone
       cannot show the mistake. Judging retrieval quality takes hundreds of labelled
       pairs and metrics such as Recall@K, MRR or NDCG.
"""

import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

# Three models that pool differently. Pooling is fixed when a model is trained, so
# it sits beside the model name.
PRIMARY, PRIMARY_POOLING = "BAAI/bge-small-en-v1.5", "cls"
SECONDARY, SECONDARY_POOLING = "iic/nlp_gte_sentence-embedding_english-small", "mean"
DECODER, DECODER_POOLING = "Qwen/Qwen3-Embedding-0.6B", "last"

MAX_LENGTH = 512

# Query 1 belongs to document 1 and query 2 to document 2. Every step is judged
# against that.
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
    """Return a local path to the weights in weights/, downloading them from
    ModelScope only if missing."""
    import logging

    from modelscope import snapshot_download

    # Hide the INFO line the downloader logs on every call, even when every file is
    # cached. Its tqdm progress bar still shows.
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
    """Score every query against every document with SentenceTransformer, which
    reads the pooling from the model's config."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ensure_model(model_id), trust_remote_code=True)
    model.max_seq_length = MAX_LENGTH
    vectors = model.encode(QUERIES + DOCUMENTS, normalize_embeddings=True)
    n = len(QUERIES)
    return vectors[:n] @ vectors[n:].T


def pool(hidden, mask, how):
    """Reduce per-token states to one vector per text with CLS, mean or last-token
    pooling."""
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


def load_model(model_id, padding_side="right"):
    """Return a tokenizer that pads on the given side, and the model, which is loaded
    only once since encode_by_hand runs several times."""
    import torch
    from modelscope import AutoModel, AutoTokenizer

    if model_id not in _LOADED:
        model_dir = ensure_model(model_id)
        # float32, because numpy cannot read the bfloat16 that Qwen3 is stored in.
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                          dtype=torch.float32)
        model.eval()
        _LOADED[model_id] = (model_dir, model)
    model_dir, model = _LOADED[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True,
                                              padding_side=padding_side)
    return tokenizer, model


def encode_by_hand(model_id, how, padding_side="right"):
    """Tokenise, run the model, pool and normalise by hand, and return the score grid."""
    import torch
    import torch.nn.functional as F

    tokenizer, model = load_model(model_id, padding_side)

    batch = tokenizer(QUERIES + DOCUMENTS, max_length=MAX_LENGTH, padding=True,
                      truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state

    vectors = pool(hidden, batch["attention_mask"], how)
    # Normalising to unit length is what turns a dot product into cosine similarity.
    vectors = F.normalize(vectors, p=2, dim=1).numpy()
    n = len(QUERIES)
    return vectors[:n] @ vectors[n:].T


def compare(label, wrapper_scores, manual_scores):
    """Print the largest difference between the wrapper's scores and the hand-built ones."""
    largest = float(np.abs(np.array(wrapper_scores) - np.array(manual_scores)).max())
    verdict = "match" if largest < 0.01 else "mismatch, check the pooling and normalisation"
    print(f"  {label}: largest difference {largest:.6f} ({verdict})")


def show_wrong_pooling(model_id, right, wrong, reference):
    """Score the same model with the right and the wrong pooling, and print how far
    each moves from the wrapper (deviation) and how well it separates the passages
    (margin)."""
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
    print("  Deviation shows the mistake: the wrong pooling moves away from the wrapper.")
    print("  Margin does not: here the wrong pooling even looks better.")


def main():
    # 1. CLS pooling
    cls_scores = encode_with_wrapper(PRIMARY)
    show_scores(f"--- 1. CLS pooling ({PRIMARY}) ---", cls_scores)

    # 2. Mean pooling
    mean_scores = encode_with_wrapper(SECONDARY)
    show_scores(f"--- 2. Mean pooling ({SECONDARY}) ---", mean_scores)

    # 3. Last-token pooling
    last_scores = encode_with_wrapper(DECODER)
    show_scores(f"--- 3. Last-token pooling ({DECODER}) ---", last_scores)

    # 4. By hand
    print("\n--- 4. By hand ---")
    for model_id, how, side, reference in [
        (PRIMARY, PRIMARY_POOLING, "right", cls_scores),
        (SECONDARY, SECONDARY_POOLING, "right", mean_scores),
        (DECODER, DECODER_POOLING, "right", last_scores),
        (DECODER, DECODER_POOLING, "left", last_scores),
    ]:
        manual = encode_by_hand(model_id, how, side)
        compare(f"{model_id} ({how}, {side} padding)", reference, manual)

    # 5. Wrong pooling
    print("\n--- 5. Wrong pooling ---")
    other = "mean" if PRIMARY_POOLING != "mean" else "cls"
    show_wrong_pooling(PRIMARY, PRIMARY_POOLING, other, cls_scores)


if __name__ == "__main__":
    main()
