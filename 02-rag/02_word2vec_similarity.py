"""This script trains Word2Vec on Journey to the West and tests what the word vectors
have learned. Chinese has no spaces between words, so jieba first cuts the text into
words. Word2Vec then learns one vector per word by predicting which words appear near
each other.

The run prints seven parts:
    1. Segmentation. The raw text is GB18030, and the segmented copy is cached.
    2. Baseline model. 100-dimensional vectors, a window of 3, every word kept.
    3. Name similarity. Cosine similarity between character names.
    4. Analogy. Sun Wukong is to Pilgrim Sun (his alias) as Tang Seng is to what? The
       top answer is 长老 (elder), the way other characters address Tang Seng.
    5. Second model. 128-dimensional vectors, a window of 5, and words seen fewer than
       five times dropped, which shrinks the vocabulary from about 46,000 to 7,700.
       The model is saved to models/.
    6. Reload. The reloaded model scores the same as before it was saved. Its analogy
       answer can change between runs, because the second model trains on several
       threads (长老 and 菩萨 have both come first). Every pair scores above 0.8, even
       Sun Wukong and "monster". A single novel is small and repetitive, so the names
       share the same contexts and their vectors crowd together.
    7. English corpus. The same recipe on text8, 17 million words of cleaned
       Wikipedia. king - man + woman gives queen, and king vs banana scores near 0.
       text8 downloads once (about 31 MB), and the trained model is cached.
"""

import io
import multiprocessing
import sys
import time
from pathlib import Path

import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Print UTF-8 even when the output is piped or redirected on Windows.
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
    """Return "utf-8" or "gb18030", whichever decodes the start of the file. Many
    Chinese text files still use GB18030."""
    for encoding in ("utf-8", "gb18030"):
        try:
            with open(path, encoding=encoding) as handle:
                handle.read(4096)
            return encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} as UTF-8 or GB18030")


def segment_corpus(source, target):
    """Cut the text into space-separated words, one line per input line, and cache
    the result."""
    if target.exists():
        print(f"Reusing cached segmentation: {target.name}")
        return target

    encoding = detect_encoding(source)
    print(f"Segmenting {source.name} ({encoding}) with jieba. This takes a moment...")
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
    """Train a Word2Vec model and print its vocabulary size and training time."""
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


def train_text8_model():
    """Train a Word2Vec model on text8, or reload the cached one."""
    if TEXT8_MODEL_FILE.exists():
        print("  Reusing cached text8 model")
        return Word2Vec.load(str(TEXT8_MODEL_FILE))

    import contextlib
    import gensim.downloader  # heavy and network-bound, so imported on demand
    from gensim.models.word2vec import Text8Corpus

    print("  Downloading text8 (about 31 MB on first run) and training...")
    started = time.time()
    # Ask for the path only: the gensim-data loader imports smart_open.smart_open,
    # which smart_open 2 removed. Hide the downloader's \r progress bar on a pipe,
    # where it floods the log.
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
    # 1. Segmentation
    print("--- 1. Segmentation ---")
    segmented = segment_corpus(SOURCE_FILE, SEGMENTED_FILE)

    # 2. Baseline model
    print("\n--- 2. Baseline model ---")
    baseline = train(segmented, vector_size=100, window=3, min_count=1)

    # 3. Name similarity
    print("\n--- 3. Name similarity ---")
    report_similarity(baseline, [
        (MONKEY_KING, PIGSY),
        (MONKEY_KING, PILGRIM_SUN),
        (MONKEY_KING, MONSTER),
    ])
    print(f"  Vector for {label(MONKEY_KING)}: shape={baseline.wv[MONKEY_KING].shape}, "
          f"first values={baseline.wv[MONKEY_KING][:5]}")

    # 4. Analogy
    print("\n--- 4. Analogy ---")
    report_analogy(baseline, positive=[MONKEY_KING, MONK], negative=[PILGRIM_SUN])

    # 5. Second model
    print("\n--- 5. Second model ---")
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

    # 6. Reload
    print("\n--- 6. Reload ---")
    reloaded = Word2Vec.load(str(MODEL_FILE))
    print(f"  Reloaded vocab size: {len(reloaded.wv):d}")
    report_similarity(reloaded, [(MONKEY_KING, PIGSY)])
    report_analogy(reloaded, positive=[MONKEY_KING, MONK], negative=[PILGRIM_SUN])
    print('  Note: every pair scores above 0.8, even Sun Wukong and "monster".')
    print("  A single novel is too small and repetitive to separate the names.")

    # 7. English corpus
    print("\n--- 7. English corpus ---")
    english = train_text8_model()
    report_analogy(english, positive=["king", "woman"], negative=["man"])
    report_analogy(english, positive=["paris", "italy"], negative=["france"])
    report_similarity(english, [("king", "queen"), ("king", "banana")])


if __name__ == "__main__":
    main()
