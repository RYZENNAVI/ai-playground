"""Answer questions about a PDF with LangChain, FAISS, and page-level citations.

Demonstrates the end-to-end path LangChain automates for you:
    1. Extract text from a PDF while remembering which page every character came from.
    2. Split the text into overlapping chunks.
    3. Map each chunk back to the page it mostly came from.
    4. Embed the chunks and build a FAISS store.
    5. Persist the store and reload it.
    6. Retrieve the chunks nearest a question.
    7. Answer through a QA chain and cite the source pages.
    8. Retrieve again under several phrasings and see what the first pass missed.

Module 02: RAG - ChatPDF with LangChain.
"""

import os
import pickle
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

PDF_FILE = Path(__file__).parent / "data" / "bank_kpi_policy.pdf"
INDEX_DIR = Path(__file__).parent / "models" / "bank_kpi_faiss"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
# How many alternative phrasings step 8 asks for.
MULTI_QUERY_COUNT = 3

QUESTIONS = [
    "How many points are deducted for each customer complaint?",
    "When do account managers apply for their annual appointment review?",
]


def pick_provider():
    """Return (api_key, base_url, embed_model, chat_model) for whichever key works.

    Any OpenAI-compatible endpoint works. One provider covers both roles, so
    there is a single key and a single quota to reason about. Gemini is tried
    first because it also supplies the embedding model this script needs;
    DashScope and OpenAI are checked after, so setting a single key is enough.
    """
    if os.getenv("GEMINI_API_KEY"):
        return (os.getenv("GEMINI_API_KEY"),
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "gemini-embedding-001", "gemini-3.1-flash-lite")
    if os.getenv("DASHSCOPE_API_KEY"):
        return (os.getenv("DASHSCOPE_API_KEY"),
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "text-embedding-v4", "deepseek-v3")
    if os.getenv("OPENAI_API_KEY"):
        return (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"),
                "text-embedding-3-small", "gpt-4o-mini")
    raise SystemExit("Set GEMINI_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY first.")


def extract_text_with_pages(pdf_path):
    """Step 1: read the PDF, recording a page number for every character.

    Tracking pages per character rather than per line is what makes step 3 work:
    the splitter later cuts the text at arbitrary offsets, and only a character
    level mapping survives that.
    """
    from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    text = ""
    char_pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text:
            print(f"  page {page_number}: no extractable text")
            continue
        text += page_text
        char_pages.extend([page_number] * len(page_text))
    print(f"  extracted {len(text)} characters from {len(reader.pages)} pages")
    return text, char_pages


def split_text(text):
    """Step 2: cut the document into overlapping chunks.

    The separator list is tried in order, so the splitter prefers paragraph
    breaks over sentence breaks over raw character cuts. The overlap keeps a
    fact that straddles a boundary readable in at least one chunk.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:  # LangChain < 0.2 kept it in the main package
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    print(f"  split into {len(chunks)} chunks")
    return chunks


def map_chunks_to_pages(text, chunks, char_pages):
    """Step 3: decide which page each chunk belongs to.

    A chunk can straddle a page break, so there is no single right answer. Taking
    the most common page among the chunk's characters gives the page that
    contributed most of it. Chunks are located with str.find because the splitter
    strips whitespace and the offsets no longer line up exactly.
    """
    page_info = {}
    cursor = 0
    for chunk in chunks:
        position = text.find(chunk, cursor)
        if position == -1:
            position = text.find(chunk)
        if position == -1:
            page_info[chunk] = "unknown"
            continue
        span = char_pages[position:position + len(chunk)]
        page_info[chunk] = Counter(span).most_common(1)[0][0] if span else "unknown"
        cursor = position + len(chunk)
    return page_info


def make_embeddings(api_key, base_url, embed_model):
    """Build the embedding client.

    check_embedding_ctx_length=False matters for non-OpenAI endpoints: by default
    LangChain tokenises the text and posts integer arrays, which OpenAI accepts
    but Gemini's compatibility layer rejects with a 501. Turning it off sends
    plain strings, which every provider understands.
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=embed_model, api_key=api_key, base_url=base_url,
                            check_embedding_ctx_length=False)


def build_store(chunks, page_info, api_key, base_url, embed_model):
    """Steps 4-5: embed the chunks, build FAISS, and save it next to its page map."""
    # FAISS still ships inside langchain-community, which upstream has marked as
    # sunset. No official standalone replacement exists yet, so the deprecation
    # warning printed here is expected rather than a misconfiguration. Module 07
    # builds the same pipeline directly on faiss-cpu, with no framework at all.
    from langchain_community.vectorstores import FAISS

    embeddings = make_embeddings(api_key, base_url, embed_model)
    store = FAISS.from_texts(chunks, embeddings)
    store.page_info = page_info
    print("  knowledge base built")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    with open(INDEX_DIR / "page_info.pkl", "wb") as handle:
        pickle.dump(page_info, handle)
    print(f"  saved to {INDEX_DIR}")
    return store, embeddings


def load_store(embeddings):
    """Step 5 (reverse): reload a saved index instead of paying to embed again.

    allow_dangerous_deserialization is required because the store is a pickle;
    only ever point this at a file you created yourself.
    """
    from langchain_community.vectorstores import FAISS

    store = FAISS.load_local(str(INDEX_DIR), embeddings,
                             allow_dangerous_deserialization=True)
    page_info_path = INDEX_DIR / "page_info.pkl"
    if page_info_path.exists():
        with open(page_info_path, "rb") as handle:
            store.page_info = pickle.load(handle)
    else:
        store.page_info = {}
        print("  warning: page map missing, citations will read 'unknown'")
    return store


def ask(store, question, api_key, base_url, chat_model):
    """Steps 6-7: retrieve, answer, and report which pages the answer came from.

    This is the "stuff" strategy: concatenate every retrieved chunk into a single
    prompt and call the model once. It is the cheapest of the four document
    strategies and the right default while the retrieved set fits the context
    window. The alternatives cost more calls: map_reduce summarises each chunk
    separately then merges, refine walks chunks in sequence carrying an answer
    forward, and map_rerank scores each chunk and keeps the best.

    LangChain 1.x removed load_qa_chain along with the rest of langchain.chains,
    so the LCEL pipeline below expresses the same idea with the pieces that
    remain: format a prompt, call the model, parse the text out.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    print(f"\n--- Q: {question}")
    docs = store.similarity_search(question, k=TOP_K)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below. If the context does "
        "not contain the answer, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOpenAI(model=chat_model, api_key=api_key, base_url=base_url, temperature=0)
    chain = prompt | llm | StrOutputParser()

    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"A: {chain.invoke({'context': context, 'question': question})}")

    pages = []
    for doc in docs:
        page = store.page_info.get(doc.page_content.strip(), "unknown")
        if page not in pages:
            pages.append(page)
    print(f"Sources: pages {pages}")


def ask_multi_query(store, question, api_key, base_url, chat_model):
    """Step 8: retrieve under several phrasings of the question, then merge.

    A single phrasing is a single throw: if the asker's wording misses the
    vocabulary the document used, similarity search returns neighbours of the
    wrong region and the answer is never in the context at all. Asking the model
    for alternative phrasings and taking the union of their hits widens what the
    retriever can see, at the cost of one extra model call plus one embedding
    call per variant.

    LangChain ships this as MultiQueryRetriever. In 1.x it moved out of
    langchain.retrievers into the separate langchain_classic package, so it is
    written out here instead - the mechanism is a dozen lines and this keeps the
    dependency list to packages that are actively maintained.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=chat_model, api_key=api_key, base_url=base_url, temperature=0)
    expand = ChatPromptTemplate.from_template(
        "Write {n} alternative phrasings of the question below that a document "
        "search might match better. Vary the vocabulary. Keep the meaning "
        "identical. Reply with one phrasing per line and nothing else.\n\n"
        "Question: {question}"
    ) | llm | StrOutputParser()

    variants = [line.strip(" -0123456789.") for line
                in expand.invoke({"n": MULTI_QUERY_COUNT, "question": question}).splitlines()
                if line.strip()]

    print(f"\n--- Q: {question}")
    for variant in variants:
        print(f"    + {variant}")

    # Deduplicate on chunk text: the same chunk surfacing under three phrasings
    # would otherwise fill three of the TOP_K slots and crowd out everything else.
    seen, merged = set(), []
    for phrasing in [question] + variants:
        for doc in store.similarity_search(phrasing, k=TOP_K):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

    single_pages = page_numbers(store, store.similarity_search(question, k=TOP_K))
    multi_pages = page_numbers(store, merged)
    print(f"  single phrasing : {len(single_pages)} pages {single_pages}")
    print(f"  {len(variants) + 1} phrasings   : {len(multi_pages)} pages {multi_pages}")
    gained = [p for p in multi_pages if p not in single_pages]
    print(f"  newly reachable : {gained if gained else 'nothing - the first wording already covered it'}")


def page_numbers(store, docs):
    """Return the distinct source pages behind a set of retrieved chunks."""
    pages = []
    for doc in docs:
        page = store.page_info.get(doc.page_content.strip(), "unknown")
        if page not in pages:
            pages.append(page)
    return pages


def main():
    if not PDF_FILE.exists():
        raise SystemExit(f"Missing {PDF_FILE}")

    api_key, base_url, embed_model, chat_model = pick_provider()
    print(f"Provider endpoint: {base_url}")

    print("\n--- 1-3. Reading and chunking the PDF ---")
    text, char_pages = extract_text_with_pages(PDF_FILE)
    chunks = split_text(text)
    page_info = map_chunks_to_pages(text, chunks, char_pages)

    print("\n--- 4-5. Building and saving the vector store ---")
    store, embeddings = build_store(chunks, page_info, api_key, base_url, embed_model)

    print("\n--- 5. Reloading from disk ---")
    store = load_store(embeddings)
    print(f"  reloaded, {len(store.page_info)} chunks carry a page number")

    print("\n--- 6-7. Question answering ---")
    for question in QUESTIONS:
        ask(store, question, api_key, base_url, chat_model)

    print("\n--- 8. Retrieving under several phrasings ---")
    for question in QUESTIONS:
        ask_multi_query(store, question, api_key, base_url, chat_model)


if __name__ == "__main__":
    main()
