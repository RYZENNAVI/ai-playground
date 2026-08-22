"""Embed documents with Gemini and search them in FAISS while keeping their metadata.

Demonstrates the full path from text to a queryable vector store:
    1. Embed a single sentence and inspect the resulting vector.
    2. Compare Matryoshka output dimensions on the same input.
    3. Compare retrieval rankings across those dimensions on real documents.
    4. Embed a document set together with its metadata.
    5. Build a FAISS index that maps each vector to a custom id.
    6. Search the index with an embedded query.
    7. Resolve the returned ids back to documents and metadata.
    8. Persist the index to disk and reload it.

Module 02: RAG - Embeddings and Vector Databases.
"""

import os
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

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
    """Create an OpenAI-compatible client pointed at Google AI Studio.

    Gemini exposes an OpenAI-compatible endpoint, so the same SDK and the same
    embeddings.create call work by swapping base_url. No vendor-specific SDK is
    needed anywhere in this script.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)


def embed(client, texts, dimensions=WORKING_DIM):
    """Embed a list of texts and return L2-normalized float32 vectors.

    Truncating a Matryoshka vector breaks its unit length, so the vectors are
    renormalized here. With unit vectors, FAISS L2 distance ranks results in
    exactly the same order as cosine similarity, just on a different scale.
    """
    response = client.embeddings.create(model=EMBED_MODEL, input=texts,
                                        dimensions=dimensions)
    vectors = np.array([item.embedding for item in response.data], dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def build_index(vectors, ids):
    """Wrap a flat L2 index in an IndexIDMap so vectors carry our own ids.

    IndexFlatL2 alone would only return positions in insertion order. IndexIDMap
    lets each vector keep an application-level id, which is what connects a search
    hit back to the document and its metadata.
    """
    index = faiss.IndexIDMap(faiss.IndexFlatL2(vectors.shape[1]))
    index.add_with_ids(vectors, np.array(ids, dtype="int64"))
    return index


def compare_retrieval_across_dimensions(client, documents, query, dimensions_list, top_k):
    """Rerun the whole embed-index-search path once per dimension and compare rankings.

    Truncating a Matryoshka vector only pays off if it does not change which
    documents come back on top. This is the difference between step 2, which
    only checks that the raw numbers line up, and an actual retrieval-quality
    comparison on real documents.
    """
    rankings = {}
    for dimensions in dimensions_list:
        vectors = embed(client, [doc["text"] for doc in documents], dimensions=dimensions)
        index = build_index(vectors, range(len(documents)))
        query_vector = embed(client, [query], dimensions=dimensions)
        distances, ids = index.search(query_vector, top_k)
        ranking = [documents[i]["id"] for i in ids[0] if i != -1]
        rankings[dimensions] = ranking
        hits = ", ".join(f"{doc_id}[{dist:.4f}]" for doc_id, dist in zip(ranking, distances[0]))
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
        print(f"  {rank}. [distance {distance:.4f}] id={doc_id} ({document['id']})")
        print(f"     text: {document['text'][:90]}...")
        print(f"     metadata: {document['metadata']}")


def main():
    client = build_client()

    print("--- 1. Embed a single sentence ---")
    single = embed(client, [QUERY])
    print(f"Input: {QUERY}")
    print(f"Vector shape: {single.shape}, norm: {np.linalg.norm(single[0]):.4f}")
    print(f"First 5 values: {single[0][:5]}")

    print("\n--- 2. Compare Matryoshka output dimensions ---")
    print("The same model can return shorter vectors that keep the leading values,")
    print("so storage cost is a dial rather than a fixed property of the model.")
    for dimensions in (FULL_DIM, 1536, WORKING_DIM):
        raw = client.embeddings.create(model=EMBED_MODEL, input=QUERY,
                                       dimensions=dimensions).data[0].embedding
        print(f"  dimensions={dimensions:5d} -> len={len(raw):5d}, "
              f"first 3 values={[round(v, 6) for v in raw[:3]]}")

    print("\n--- 3. Compare retrieval rankings across dimensions ---")
    compare_retrieval_across_dimensions(client, DOCUMENTS, QUERY,
                                        [FULL_DIM, 1536, WORKING_DIM], TOP_K)

    print(f"\n--- 4. Embed the document set (dimensions={WORKING_DIM}) ---")
    vectors = embed(client, [document["text"] for document in DOCUMENTS])
    store = {index: document for index, document in enumerate(DOCUMENTS)}
    print(f"Embedded {len(DOCUMENTS)} documents into shape {vectors.shape}")

    print("\n--- 5. Build a FAISS index with custom ids ---")
    index = build_index(vectors, list(store))
    print(f"Index contains {index.ntotal} vectors")

    print("\n--- 6. Search the index with an embedded query ---")
    query_vector = embed(client, [QUERY])
    distances, ids = index.search(query_vector, TOP_K)
    print(f"Query: {QUERY}")

    print("\n--- 7. Resolve ids back to documents and metadata ---")
    show_hits(distances[0], ids[0], store)

    print("\n--- 8. Persist the index and reload it ---")
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
