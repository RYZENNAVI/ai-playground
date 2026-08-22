"""Train Word2Vec on a Chinese novel and explore the geometry of its word vectors.

Demonstrates the jump from counting words to learning dense vectors:
    1. Segment the raw Chinese text into words with jieba.
    2. Train a baseline Word2Vec model on the segmented corpus.
    3. Measure similarity between character names.
    4. Solve analogies with vector arithmetic.
    5. Retrain with tuned hyper-parameters and save the model to disk.
    6. Reload the saved model and confirm it answers the same way.
    7. Reproduce the textbook king/queen analogy on a large English corpus.

Module 02: RAG - Word2Vec Word Embeddings.
"""

import io
import multiprocessing
import sys
import time
from pathlib import Path

import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent
SOURCE_FILE = BASE_DIR / "data" / "journey_to_the_west.txt"
SEGMENTED_FILE = BASE_DIR / "data" / "journey_to_the_west.segmented.txt"
MODEL_FILE = BASE_DIR / "models" / "word2vec_journey.model"
TEXT8_MODEL_FILE = BASE_DIR / "models" / "word2vec_text8.model"

# The corpus is Chinese, so the query terms have to be Chinese too.
MONKEY_KING = "孙悟空"   # Sun Wukong, the Monkey King
PIGSY = "猪八戒"         # Zhu Bajie, the pig disciple
PILGRIM_SUN = "孙行者"   # Pilgrim Sun, an alias of the Monkey King
MONK = "唐僧"            # Tang Seng, the monk
SANDY = "沙僧"           # Sha Seng, the third disciple
MONSTER = "妖怪"         # monster

GLOSS = {
    MONKEY_KING: "Sun Wukong",
    PIGSY: "Zhu Bajie",
    PILGRIM_SUN: "Pilgrim Sun",
    MONK: "Tang Seng",
    SANDY: "Sha Seng",
    MONSTER: "monster",
}


def label(word):
    """Render a Chinese token with its English gloss when one is known."""
    return f"{word} ({GLOSS[word]})" if word in GLOSS else word


def detect_encoding(path):
    """Guess the text encoding of a Chinese corpus file.

    Public-domain Chinese texts are still commonly distributed as GB18030 rather
    than UTF-8, and decoding one as the other fails on the very first character.
    """
    for encoding in ("utf-8", "gb18030"):
        try:
            with open(path, encoding=encoding) as handle:
                handle.read(4096)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} as UTF-8 or GB18030")


def segment_corpus(source, target):
    """Cut the raw text into space-separated words, one output line per input line.

    Chinese has no spaces between words, so every downstream tool that expects
    tokens needs this step first. The result is cached because segmenting the
    whole novel takes a while and never changes.
    """
    if target.exists():
        print(f"Reusing cached segmentation: {target.name}")
        return target

    encoding = detect_encoding(source)
    print(f"Segmenting {source.name} ({encoding}) with jieba, this takes a moment...")
    started = time.time()
    words = 0
    # Write to a temporary file first: a crash midway must not leave behind a
    # truncated cache that later runs would happily reuse.
    partial = target.with_suffix(target.suffix + ".partial")
    with open(source, encoding=encoding) as reader, \
            open(partial, "w", encoding="utf-8") as writer:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            tokens = [token for token in jieba.cut(line) if token.strip()]
            words += len(tokens)
            writer.write(" ".join(tokens) + "\n")
    partial.replace(target)
    print(f"Wrote {words} tokens in {time.time() - started:.1f}s")
    return target


def train(sentences_path, vector_size, window, min_count, workers=1):
    """Train a Word2Vec model and report its vocabulary size.

    Word2Vec never stores a sentence: it learns one vector per word by predicting
    which words share a context window. The trained model is therefore just a
    lookup table from token to vector, which is why inference is instant.
    """
    started = time.time()
    model = Word2Vec(LineSentence(str(sentences_path)), vector_size=vector_size,
                     window=window, min_count=min_count, workers=workers)
    print(f"  vector_size={vector_size} window={window} min_count={min_count} "
          f"-> vocab={len(model.wv):d}, trained in {time.time() - started:.1f}s")
    return model


def report_similarity(model, pairs):
    """Print the cosine similarity for each pair of words."""
    for left, right in pairs:
        if left not in model.wv or right not in model.wv:
            print(f"  {label(left)} vs {label(right)}: missing from vocabulary")
            continue
        print(f"  {label(left)} vs {label(right)}: {model.wv.similarity(left, right):.4f}")


def report_analogy(model, positive, negative, top_k=5):
    """Print the nearest words to positive vectors minus negative vectors."""
    missing = [w for w in positive + negative if w not in model.wv]
    if missing:
        print(f"  Skipped, missing from vocabulary: {missing}")
        return
    plus = " + ".join(label(w) for w in positive)
    minus = " - ".join(label(w) for w in negative)
    print(f"  {plus} - {minus}:")
    for word, score in model.wv.most_similar(positive=positive, negative=negative,
                                             topn=top_k):
        print(f"    [{score:.4f}] {label(word)}")


def require_data_file(path, origin):
    """Fail early with a usable message when a required data file is absent.

    The data directory is git-ignored, so a fresh clone has the scripts but not
    the corpora. The message names where the file is expected.
    """
    if not path.exists():
        raise SystemExit(
            f"Missing data file: {path}\n"
            f"Expected source: {origin}"
        )
    return path

def train_text8_model():
    """Train, or reload, a Word2Vec model on the English text8 corpus.

    A single novel is a small corpus: nearly every word ends up similar to every
    other one. text8 is a cleaned 17-million-token Wikipedia dump, large enough to
    reproduce the analogy Word2Vec is famous for. It downloads about 31 MB on
    first use, caches under ~/gensim-data, and the trained model is saved here so
    later runs skip both the download and the training.
    """
    if TEXT8_MODEL_FILE.exists():
        print("  Reusing cached text8 model")
        return Word2Vec.load(str(TEXT8_MODEL_FILE))

    import contextlib
    import gensim.downloader  # heavy and network-bound, so imported on demand
    from gensim.models.word2vec import Text8Corpus

    print("  Downloading text8 (about 31 MB on first run) and training...")
    started = time.time()
    # Ask for the path instead of the corpus object: gensim-data ships a loader
    # shim that still does "from smart_open import smart_open", which no longer
    # exists in smart_open 2+. Reading the archive with Text8Corpus avoids it.
    # The downloader also writes a carriage-return progress bar, which floods the
    # log with thousands of lines when stdout is a pipe rather than a terminal.
    if sys.stdout.isatty():
        corpus_path = gensim.downloader.load("text8", return_path=True)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            corpus_path = gensim.downloader.load("text8", return_path=True)
    model = Word2Vec(Text8Corpus(corpus_path), vector_size=100, window=5,
                     min_count=5, workers=multiprocessing.cpu_count())
    print(f"  vocab={len(model.wv):d}, trained in {time.time() - started:.1f}s")
    TEXT8_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(TEXT8_MODEL_FILE))
    return model


def main():
    print("--- 1. Segment the raw Chinese text ---")
    require_data_file(SOURCE_FILE,
                      "4-Embeddings/word2vec/journey_to_the_west/source/journey_to_the_west.txt")
    segmented = segment_corpus(SOURCE_FILE, SEGMENTED_FILE)

    print("\n--- 2. Train a baseline Word2Vec model ---")
    baseline = train(segmented, vector_size=100, window=3, min_count=1)

    print("\n--- 3. Measure similarity between character names ---")
    report_similarity(baseline, [
        (MONKEY_KING, PIGSY),
        (MONKEY_KING, PILGRIM_SUN),
        (MONKEY_KING, MONSTER),
    ])
    print(f"  Vector for {label(MONKEY_KING)}: shape={baseline.wv[MONKEY_KING].shape}, "
          f"first values={baseline.wv[MONKEY_KING][:5]}")

    print("\n--- 4. Solve analogies with vector arithmetic ---")
    report_analogy(baseline, positive=[MONKEY_KING, MONK], negative=[PILGRIM_SUN])

    print("\n--- 5. Retrain with tuned hyper-parameters and save ---")
    tuned = train(segmented, vector_size=128, window=5, min_count=5,
                  workers=multiprocessing.cpu_count())
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    tuned.save(str(MODEL_FILE))
    print(f"  Saved to {MODEL_FILE}")
    report_similarity(tuned, [
        (MONKEY_KING, PIGSY),
        (MONKEY_KING, PILGRIM_SUN),
        (MONK, SANDY),
    ])

    print("\n--- 6. Reload the saved model and confirm ---")
    reloaded = Word2Vec.load(str(MODEL_FILE))
    print(f"  Reloaded vocab size: {len(reloaded.wv):d}")
    report_similarity(reloaded, [(MONKEY_KING, PIGSY)])
    report_analogy(reloaded, positive=[MONKEY_KING, MONK], negative=[PILGRIM_SUN])
    print("  Note: every pair above scores above 0.9. That is not a sign the model")
    print("  is working unusually well -- a single novel is a small, stylistically")
    print("  repetitive corpus, so most character names end up in near-identical")
    print("  local contexts (the same dialogue tags and narrative patterns) and")
    print("  get pushed toward the same region of the vector space. Step 7 repeats")
    print("  the same recipe on a bigger, more varied corpus for contrast.")

    print("\n--- 7. Reproduce the textbook analogy on an English corpus ---")
    english = train_text8_model()
    report_analogy(english, positive=["king", "woman"], negative=["man"])
    report_analogy(english, positive=["paris", "italy"], negative=["france"])
    report_similarity(english, [("king", "queen"), ("king", "banana")])


if __name__ == "__main__":
    main()
