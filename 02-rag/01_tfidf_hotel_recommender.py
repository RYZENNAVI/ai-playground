"""Recommend similar hotels from their descriptions using TF-IDF and cosine similarity.

Demonstrates classical statistical text features before neural embeddings:
    1. Load the Seattle hotel dataset.
    2. Inspect a single hotel description.
    3. Rank the most frequent n-grams after removing stop words.
    4. Clean the raw descriptions into a normalized bag of words.
    5. Build a TF-IDF matrix over the cleaned descriptions.
    6. Score every hotel pair with cosine similarity.
    7. Recommend the ten closest hotels for a given hotel name.

Module 02: RAG - TF-IDF Content-Based Recommendation.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.stdout.reconfigure(encoding="utf-8")

DATA_FILE = Path(__file__).parent / "data" / "Seattle_Hotels.csv"

# Minimal English stop-word list, inlined so the script needs no NLTK download.
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y",
}

PUNCTUATION_RE = re.compile(r"[/(){}\[\]\|@,;]")
NON_ALPHANUMERIC_RE = re.compile(r"[^0-9a-z #+_]")


def load_hotels(path):
    """Read the hotel CSV and report how many rows it holds."""
    frame = pd.read_csv(path, encoding="latin-1")
    print(f"Hotels in dataset: {len(frame)}")
    print(f"Columns: {list(frame.columns)}")
    return frame


def top_n_grams(corpus, n, k):
    """Return the k most frequent n-grams across the corpus.

    A raw count over n-grams is the simplest possible text feature: it captures
    which phrases appear often, but treats every occurrence as equally important
    no matter how common the phrase is across the whole collection.
    """
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=list(STOP_WORDS))
    counts = vectorizer.fit_transform(corpus)
    totals = counts.sum(axis=0)
    frequencies = [(word, totals[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    frequencies.sort(key=lambda pair: pair[1], reverse=True)
    return frequencies[:k]


def clean_text(text):
    """Lowercase the text, drop punctuation and stop words."""
    text = text.lower()
    text = PUNCTUATION_RE.sub(" ", text)
    text = NON_ALPHANUMERIC_RE.sub("", text)
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


def build_similarity_matrix(descriptions):
    """Turn cleaned descriptions into TF-IDF vectors and score every pair.

    TF-IDF fixes the weakness of raw counts by dividing each term's frequency by
    how many documents contain it, so words that appear in every hotel blurb
    ("seattle", "hotel") stop dominating the vector. Because the vectors are
    L2-normalized, a linear kernel is exactly the cosine similarity.
    """
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=0.01,
                                 stop_words=list(STOP_WORDS))
    matrix = vectorizer.fit_transform(descriptions)
    print(f"TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"TF-IDF matrix shape: {matrix.shape}")
    similarities = linear_kernel(matrix, matrix)
    print(f"Similarity matrix shape: {similarities.shape}")
    return similarities


def recommend(name, names, similarities, top_k=10):
    """Return the top_k hotels closest to the given hotel, excluding itself."""
    matches = names[names == name].index
    if len(matches) == 0:
        raise ValueError(f"Hotel not found: {name}")
    index = matches[0]
    ranked = pd.Series(similarities[index]).sort_values(ascending=False)
    neighbours = list(ranked.iloc[1:top_k + 1].index)
    return [(names[i], float(similarities[index][i])) for i in neighbours]


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

def main():
    print("--- 1. Load the Seattle hotel dataset ---")
    require_data_file(DATA_FILE, "4-Embeddings/hotel_recommendation/Seattle_Hotels.csv")
    frame = load_hotels(DATA_FILE)

    print("\n--- 2. Inspect a single hotel description ---")
    sample = frame.iloc[10]
    print(f"Name: {sample['name']}")
    print(f"Description: {sample['desc'][:300]}...")

    print("\n--- 3. Rank the most frequent n-grams ---")
    for phrase, count in top_n_grams(frame["desc"], n=3, k=20):
        print(f"  {count:4d}  {phrase}")

    print("\n--- 4. Clean the raw descriptions ---")
    frame["desc_clean"] = frame["desc"].apply(clean_text)
    print(f"Before: {frame['desc'].iloc[10][:120]}...")
    print(f"After : {frame['desc_clean'].iloc[10][:120]}...")

    print("\n--- 5. Build a TF-IDF matrix ---")
    print("--- 6. Score cosine similarity between every hotel pair ---")
    similarities = build_similarity_matrix(frame["desc_clean"])
    names = pd.Series(frame["name"])

    print("\n--- 7. Recommend the ten closest hotels ---")
    for query in ["Hilton Seattle Airport & Conference Center",
                  "The Bacon Mansion Bed and Breakfast"]:
        print(f"\nHotels similar to: {query}")
        for rank, (hotel, score) in enumerate(recommend(query, names, similarities), start=1):
            print(f"  {rank:2d}. [{score:.4f}] {hotel}")


if __name__ == "__main__":
    main()
