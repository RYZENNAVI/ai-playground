"""This script builds a small vector search over four Disney FAQ entries: two about
ticket refunds, one about the annual pass and one about a ride closed for
maintenance. The question "I want to understand the refund process for Disney
tickets" and the four entries are embedded with Gemini's gemini-embedding-001, and
FAISS finds the entries closest to the question. FAISS stores only vectors and integer
ids, so the entries' text and metadata live in a dict under the same ids.

gemini-embedding-001 returns 3072 values per text, and the leading values also work as
a shorter embedding (Matryoshka representation learning). The script uses 768. A
shortened vector is no longer unit length, so every vector is rescaled to length 1.
For unit vectors, FAISS's squared L2 distance equals 2 - 2 x cosine similarity, so
both rank the entries the same way.

The run prints seven parts:
    1. One embedding. The refund question as a 768-value vector of length 1.
    2. Matryoshka dimensions. The same question embedded at 3072, 1536 and 768
       dimensions. The first values match, because each shorter vector is the start
       of the longer one.
    3. Ranking per dimension. The whole search (embed the four entries and the
       question, build an index, search) runs once at each of the three sizes, to
       check that shorter vectors still put the same entries first. The top three are
       the same at every size.
    4. Document embeddings. The four FAQ entries embedded at 768 dimensions.
    5. FAISS index. The four entry vectors go into a flat L2 index wrapped in
       IndexIDMap, so a search returns each entry's own id (doc3 as 3) instead of its
       position in the index.
    6. Search. The refund question is searched against the index. The three nearest
       entries come back with their squared L2 distance, and their text and metadata
       are looked up by id. The two refund entries come first.
    7. Save and reload. The index is written to models/ and read back, and the same
       search gives the same ranking. The metadata dict is not part of the index and
       would have to be saved alongside it.
"""

import os
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

EMBED_MODEL = "gemini-embedding-001"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
FULL_DIM = 3072
WORKING_DIM = 768
TOP_K = 3
INDEX_FILE = Path(__file__).parent / "models" / "disney_faq.index"

DOCUMENTS = [
    {
        "id": "doc1",
        "text": "Disneyland tickets are in principle non-refundable once sold. Under "
                "special circumstances, such as a park closure caused by severe "
                "weather, guests may reschedule or request a refund by following "
                "official guidance.",
        "metadata": {"source": "official_faq_v1.pdf", "category": "refund policy",
                     "author": "Admin"},
    },
    {
        "id": "doc2",
        "text": "Holders of the Magic Annual Pass may enter the park any number of "
                "times within one year, and receive discounts on dining and "
                "merchandise.",
        "metadata": {"source": "annual_pass_rules.docx", "category": "membership",
                     "author": "MarketingDept"},
    },
    {
        "id": "doc3",
        "text": "For Disney tickets bought online, a refund request must be submitted "
                "through the original purchase channel at least 48 hours before the "
                "date printed on the ticket, and a service fee may apply.",
        "metadata": {"source": "online_policy.html", "category": "refund policy",
                     "author": "E-commerceTeam"},
    },
    {
        "id": "doc4",
        "text": "The Pirates of the Caribbean attraction will be closed next week for "
                "annual maintenance.",
        "metadata": {"source": "maintenance_notice.txt", "category": "park notice",
                     "author": "OpsDept"},
    },
]

QUERY = "I want to understand the refund process for Disney tickets"


def build_client():
    """Create an OpenAI client pointed at Gemini's OpenAI-compatible endpoint."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in .env and retry.")
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


def embed(client, texts, dimensions=WORKING_DIM):
    """Embed texts and return float32 vectors rescaled to length 1, since shortened
    Matryoshka vectors are not."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts,
                                        dimensions=dimensions)
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def build_index(vectors, ids):
    """Wrap a flat L2 index in IndexIDMap so each vector keeps a document id."""
    index = faiss.IndexIDMap(faiss.IndexFlatL2(vectors.shape[1]))
    index.add_with_ids(vectors, np.array(ids, dtype="int64"))
    return index


def compare_retrieval_across_dimensions(client, documents, query, dimensions_list, top_k):
    """Run the whole embed, index and search path once per dimension and report
    whether the top-k ranking changes."""
    rankings = {}
    for dimensions in dimensions_list:
        vectors = embed(client, [doc["text"] for doc in documents], dimensions=dimensions)
        index = build_index(vectors, range(len(documents)))
        query_vector = embed(client, [query], dimensions=dimensions)
        distances, ids = index.search(query_vector, top_k)
        ranking = [documents[i]["id"] for i in ids[0] if i != -1]
        rankings[dimensions] = ranking
        hits = ", ".join(f"{doc_id}[squared L2 {dist:.4f}]" for doc_id, dist in zip(ranking, distances[0]))
        print(f"  dimensions={dimensions:5d} -> top-{top_k}: {hits}")

    baseline = next(iter(rankings.values()))
    identical = all(ranking == baseline for ranking in rankings.values())
    print(f"  Same top-{top_k} ranking at every dimension: {identical}")


def show_hits(distances, ids, store):
    """Print each search hit with its distance, source text and metadata."""
    for rank, (distance, doc_id) in enumerate(zip(distances, ids), start=1):
        if doc_id == -1:
            print(f"  {rank}. No further results.")
            continue
        document = store[int(doc_id)]
        print(f"  {rank}. [squared L2 {distance:.4f}] id={doc_id} ({document['id']})")
        print(f"     text: {document['text'][:90]}...")
        print(f"     metadata: {document['metadata']}")


def main():
    client = build_client()

    # 1. One embedding
    print("--- 1. One embedding ---")
    single = embed(client, [QUERY])
    print(f"Input: {QUERY}")
    print(f"Vector shape: {single.shape}, norm: {np.linalg.norm(single[0]):.4f}")
    print(f"First 5 values: {single[0][:5]}")

    # 2. Matryoshka dimensions
    print("\n--- 2. Matryoshka dimensions ---")
    print("Shorter vectors keep the leading values, so they take less storage.")
    for dimensions in (FULL_DIM, 1536, WORKING_DIM):
        raw = client.embeddings.create(model=EMBED_MODEL, input=QUERY,
                                       dimensions=dimensions).data[0].embedding
        print(f"  dimensions={dimensions:5d} -> len={len(raw):5d}, "
              f"first 3 values={[round(v, 6) for v in raw[:3]]}")

    # 3. Ranking per dimension
    print("\n--- 3. Ranking per dimension ---")
    compare_retrieval_across_dimensions(client, DOCUMENTS, QUERY,
                                        [FULL_DIM, 1536, WORKING_DIM], TOP_K)

    # 4. Document embeddings
    print(f"\n--- 4. Document embeddings ({WORKING_DIM} dimensions) ---")
    vectors = embed(client, [document["text"] for document in DOCUMENTS])
    # FAISS ids must be integers, so doc3 is stored under id 3.
    store = {int(document["id"].removeprefix("doc")): document for document in DOCUMENTS}
    print(f"Embedded {len(DOCUMENTS)} documents into shape {vectors.shape}")

    # 5. FAISS index
    print("\n--- 5. FAISS index ---")
    index = build_index(vectors, list(store))
    print(f"Index contains {index.ntotal} vectors")

    # 6. Search
    print("\n--- 6. Search ---")
    query_vector = embed(client, [QUERY])
    distances, ids = index.search(query_vector, TOP_K)
    show_hits(distances[0], ids[0], store)

    # 7. Save and reload
    print("\n--- 7. Save and reload ---")
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Wrote index to {INDEX_FILE}")
    reloaded = faiss.read_index(str(INDEX_FILE))
    print(f"Reloaded index contains {reloaded.ntotal} vectors")
    reloaded_distances, reloaded_ids = reloaded.search(query_vector, TOP_K)
    identical = np.array_equal(ids, reloaded_ids)
    print(f"Same ranking after reload: {identical}")
    print("Note: the index stores vectors only. The metadata store lives outside it,")
    print("so a real system must persist both together or the ids lose their meaning.")


if __name__ == "__main__":
    main()
