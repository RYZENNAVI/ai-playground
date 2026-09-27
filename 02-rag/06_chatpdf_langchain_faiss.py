"""This script is a small RAG pipeline. It answers two questions about a
nine-page PDF, a bank's rules for assessing retail account managers, and cites
the pages each answer came from. One question asks how many points a customer
complaint costs, the other when the yearly appointment review opens. LangChain
splits the text, embeds the chunks and builds a FAISS store. The page citations
are the script's own work. The splitter cuts the text wherever it likes, so the
script records a page number for every character and gives each chunk the page
most of its characters came from.

The run prints five parts:
    1. Reading and chunking. The text of the PDF, split into chunks of up to 1,000
       characters with 200 overlapping, each mapped to a page.
    2. Vector store. The chunks embedded and saved to disk with their page map.
    3. Reload. The saved store loaded back.
    4. Question answering. For each question, the 4 nearest chunks go into one
       prompt, and the answer is printed with the pages of those chunks.
    5. Several phrasings. The model rewrites each question three ways. The pages
       found by all four phrasings are compared with the pages found by the
       question alone, and the run shows where the page holding the answer ranks
       in each. A new page helps only if the answer page was missing before.
"""

import os
import pickle
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

PDF_FILE = Path(__file__).parent / "data" / "bank_kpi_policy.pdf"
INDEX_DIR = Path(__file__).parent / "models" / "bank_kpi_faiss"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
# How many alternative phrasings part 5 asks for.
MULTI_QUERY_COUNT = 3

# Each question with the PDF page that holds its answer.
QUESTIONS = {
    "How many points are deducted for each customer complaint?": 5,
    "When do account managers apply for their annual appointment review?": 6,
}


def pick_provider():
    """Return (api_key, base_url, embed_model, chat_model) for the first key set.
    The provider must offer embeddings too, so one key covers both roles."""
    if os.getenv("GEMINI_API_KEY"):
        return (os.getenv("GEMINI_API_KEY"),
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "gemini-embedding-001", "gemini-3.1-flash-lite")
    if os.getenv("OPENAI_API_KEY"):
        return (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"),
                "text-embedding-3-small", "gpt-4o-mini")
    raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY first.")


def extract_text_with_pages(pdf_path):
    """Read the PDF and record a page number for every character. The splitter
    cuts at any offset, so only a per-character map survives it."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    text = ""
    char_pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text:
            print(f"  page {page_number}: no extractable text")
            continue
        # Each page ends on its footer ("- 5 -"). A plain newline keeps the next
        # page's first line off it. Not "\n\n": the text has none, so the pages
        # would become the splitter's first choice and chunks would stop
        # crossing them.
        page_text += "\n"
        text += page_text
        char_pages.extend([page_number] * len(page_text))
    print(f"  extracted {len(text)} characters from {len(reader.pages)} pages")
    return text, char_pages


def split_text(text):
    """Cut the text into overlapping chunks, trying the separators in order:
    paragraph, line, full stop, space, then any character."""
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
    """Give each chunk the page most of its characters came from. Chunks are found
    with str.find, because the splitter strips whitespace and returns no offsets."""
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
    """Build the embedding client. check_embedding_ctx_length=False sends plain
    strings: the default posts token arrays, which Gemini rejects with a 501."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=embed_model, api_key=api_key, base_url=base_url,
                            check_embedding_ctx_length=False)


def build_store(chunks, page_info, api_key, base_url, embed_model):
    """Embed the chunks, build FAISS, and save it next to its page map."""
    # langchain-community prints a DeprecationWarning here. Script 07 builds the
    # same pipeline on faiss-cpu without LangChain.
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
    """Reload the saved store and its page map. The store is a pickle, so load only
    files you created yourself."""
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
    """Answer from the TOP_K nearest chunks in one prompt (the "stuff" approach)
    and print the pages they came from."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    print(f"\n  Q: {question}")
    docs = store.similarity_search(question, k=TOP_K)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below. If the context does "
        "not contain the answer, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOpenAI(model=chat_model, api_key=api_key, base_url=base_url, temperature=0)
    chain = prompt | llm | StrOutputParser()

    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"  A: {chain.invoke({'context': context, 'question': question})}")
    print(f"  Sources: pages {page_numbers(store, docs)}")


def ask_multi_query(store, question, answer_page, api_key, base_url, chat_model):
    """Retrieve under the question and its rephrasings, merge, and show where the
    answer page ranks. Written out because MultiQueryRetriever left langchain."""
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

    # Strip list markers such as "1. " or "- " from the left only, so a phrasing
    # that ends in a number keeps it.
    variants = [line.lstrip(" -0123456789.").strip() for line
                in expand.invoke({"n": MULTI_QUERY_COUNT, "question": question}).splitlines()
                if line.strip()]

    print(f"\n  Q: {question}")
    for variant in variants:
        print(f"    + {variant}")

    # Deduplicate on chunk text, so a chunk found by several phrasings counts once.
    seen, merged = set(), []
    for phrasing in [question] + variants:
        for doc in store.similarity_search(phrasing, k=TOP_K):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

    single_pages = page_numbers(store, store.similarity_search(question, k=TOP_K))
    multi_pages = page_numbers(store, merged)
    label = f"{len(variants) + 1} phrasings"
    for name, pages in (("single phrasing", single_pages), (label, multi_pages)):
        where = (f"position {pages.index(answer_page) + 1}" if answer_page in pages
                 else "missing")
        print(f"    {name:<16}: {len(pages)} pages {pages}, "
              f"answer page {answer_page} at {where}")
    gained = [p for p in multi_pages if p not in single_pages]
    if not gained:
        print("    new pages       : none")
    elif answer_page in single_pages:
        print(f"    new pages       : {gained}, not needed, the answer page was "
              "already found")
    else:
        print(f"    new pages       : {gained}")


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

    # 1. Reading and chunking
    print("\n--- 1. Reading and chunking ---")
    text, char_pages = extract_text_with_pages(PDF_FILE)
    chunks = split_text(text)
    page_info = map_chunks_to_pages(text, chunks, char_pages)

    # 2. Vector store
    print("\n--- 2. Vector store ---")
    store, embeddings = build_store(chunks, page_info, api_key, base_url, embed_model)

    # 3. Reload
    print("\n--- 3. Reload ---")
    store = load_store(embeddings)
    with_page = sum(page != "unknown" for page in store.page_info.values())
    print(f"  reloaded, {with_page} of {len(store.page_info)} chunks carry a page number")

    # 4. Question answering
    print("\n--- 4. Question answering ---")
    for question in QUESTIONS:
        ask(store, question, api_key, base_url, chat_model)

    # 5. Several phrasings
    print("\n--- 5. Several phrasings ---")
    for question, answer_page in QUESTIONS.items():
        ask_multi_query(store, question, answer_page, api_key, base_url, chat_model)


if __name__ == "__main__":
    main()
