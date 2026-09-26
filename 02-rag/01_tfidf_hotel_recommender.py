"""This script recommends hotels by comparing their descriptions. The data is 152
Seattle hotels, each with a name, an address and a marketing description. Every
description becomes a TF-IDF vector, and hotels whose vectors point in similar
directions count as similar. No neural model is involved: the features are word
counts, weighted down for words that many descriptions share.

The run prints six parts:
    1. Dataset. The number of hotels and the three columns: name, address and desc.
    2. One description. The raw description of one hotel, W Seattle, to show what the
       text looks like before any processing.
    3. Frequent phrases. The 20 most common three-word phrases across all 152
       descriptions, with stop words removed. "pike place market" comes first. Raw
       counts rank a phrase high whether or not it tells hotels apart, which is the
       problem TF-IDF addresses in part 5.
    4. Cleaning. Every description is lowercased and stripped of punctuation and stop
       words. W Seattle's description is shown before and after.
    5. TF-IDF and cosine similarity. The cleaned descriptions become TF-IDF vectors
       over terms of one to three words (3348 terms). TF-IDF weights each term down by
       how many descriptions contain it, so terms that most hotels use count less. The
       vectors have unit length, so their dot product (the linear kernel) is the
       cosine similarity. The result is a 152 x 152 matrix holding the similarity of
       every pair of hotels.
    6. Recommendations. For two hotels, Hilton Seattle Airport & Conference Center and
       The Bacon Mansion Bed and Breakfast, the script sorts their rows of the matrix
       and prints the ten most similar other hotels with their scores. The Hilton gets
       mostly airport hotels, and the Bacon Mansion mostly bed and breakfasts.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Print UTF-8 even when the output is piped or redirected on Windows.
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


# 1. Dataset

def load_hotels(path):
    """Read the hotel CSV and print its size and columns."""
    frame = pd.read_csv(path, encoding="latin-1")
    print(f"Hotels in dataset: {len(frame)}")
    print(f"Columns: {list(frame.columns)}")
    return frame


# 3. Frequent phrases

def top_n_grams(corpus, n, k):
    """Return the k most frequent n-grams across the corpus."""
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words=list(STOP_WORDS))
    counts = vectorizer.fit_transform(corpus)
    totals = counts.sum(axis=0)
    frequencies = [(word, totals[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    frequencies.sort(key=lambda pair: pair[1], reverse=True)
    return frequencies[:k]


# 4. Cleaning

def clean_text(text):
    """Lowercase the text, drop punctuation and stop words."""
    text = text.lower()
    text = PUNCTUATION_RE.sub(" ", text)
    text = NON_ALPHANUMERIC_RE.sub("", text)
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


# 5. TF-IDF and cosine similarity

def build_similarity_matrix(descriptions):
    """Turn the cleaned descriptions into TF-IDF vectors and return the cosine
    similarity of every pair."""
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=0.01,
                                 stop_words=list(STOP_WORDS))
    matrix = vectorizer.fit_transform(descriptions)
    print(f"TF-IDF vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"TF-IDF matrix shape: {matrix.shape}")
    similarities = linear_kernel(matrix, matrix)
    print(f"Similarity matrix shape: {similarities.shape}")
    return similarities


# 6. Recommendations

def recommend(name, names, similarities, top_k=10):
    """Return the top_k hotels closest to the given hotel, excluding itself."""
    matches = names[names == name].index
    if len(matches) == 0:
        raise ValueError(f"Hotel not found: {name}")
    index = matches[0]
    ranked = pd.Series(similarities[index]).sort_values(ascending=False)
    neighbours = list(ranked.iloc[1:top_k + 1].index)
    return [(names[i], float(similarities[index][i])) for i in neighbours]


def main():
    print("--- 1. Dataset ---")
    frame = load_hotels(DATA_FILE)

    print("\n--- 2. One description ---")
    sample = frame.iloc[10]
    print(f"Name: {sample['name']}")
    print(f"Description: {sample['desc'][:300]}...")

    print("\n--- 3. Frequent phrases ---")
    for phrase, count in top_n_grams(frame["desc"], n=3, k=20):
        print(f"  {count:4d}  {phrase}")

    print("\n--- 4. Cleaning ---")
    frame["desc_clean"] = frame["desc"].apply(clean_text)
    print(f"Before: {frame['desc'].iloc[10][:120]}...")
    print(f"After : {frame['desc_clean'].iloc[10][:120]}...")

    print("\n--- 5. TF-IDF and cosine similarity ---")
    similarities = build_similarity_matrix(frame["desc_clean"])
    names = pd.Series(frame["name"])

    print("\n--- 6. Recommendations ---")
    for query in ["Hilton Seattle Airport & Conference Center",
                  "The Bacon Mansion Bed and Breakfast"]:
        print(f"\nHotels similar to: {query}")
        for rank, (hotel, score) in enumerate(recommend(query, names, similarities), start=1):
            print(f"  {rank:2d}. [{score:.4f}] {hotel}")


if __name__ == "__main__":
    main()
