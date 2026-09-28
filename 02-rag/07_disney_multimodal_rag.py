"""This script builds a small multimodal RAG assistant for a Disney park by hand,
without LangChain. The knowledge base is four Word files (ticket rules, senior
tickets, a visit guide, hotel and membership services) and two event posters.
Text and images are embedded by different models, so they sit in two FAISS
indexes whose distances cannot be compared. Both kinds of vector are normalised,
so the distances look alike, but the two models spread their scores differently.
Every question searches the text index. The image index is searched only when
the question mentions a poster, a picture or a similar word, and then it returns
the single nearest image.

An image can reach a text-only prompt in three ways, and the script shows all
three. CLIP embeds the picture, so a text query can find it. OCR reads the words
printed on it, and those words go into the prompt when the image is found. A
vision model describes the picture itself. Its description is printed for
comparison and is not used in the answers.

The run prints three parts:
    1. Indexing. Each paragraph keeps its heading in front, each table becomes
       Markdown, and 26 text blocks and 2 images go into the two indexes.
    2. Retrieval and answers. Three questions: ticket refunds, what the Halloween
       poster looks like, and annual pass discounts. The poster question finds
       only unrelated text, so its answer rests on the poster's OCR text and
       cannot describe the picture.
    3. Vision model. The Halloween poster described by the vision model, which
       sees the picture as well as the words.
"""

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).parents[1] / ".env")

DOCS_DIR = Path(__file__).parent / "data" / "disney_kb"
IMG_DIR = DOCS_DIR / "images"

CLIP_MODEL = "openai/clip-vit-base-patch32"
TEXT_DIM = 1024
IMAGE_DIM = 512
TOP_K = 3

# A question only triggers the image index when it actually asks about a picture.
IMAGE_KEYWORDS = ("poster", "picture", "image", "photo", "look like", "show me")

QUESTIONS = [
    "What is the refund process for Disney tickets?",
    "What does the recent Halloween event poster look like?",
    "What discounts does the Disney annual pass offer?",
]


def pick_provider():
    """Return (api_key, base_url, embed_model, chat_model, vision_model) for the
    first key set. One key covers embeddings, chat and vision."""
    if os.getenv("GEMINI_API_KEY"):
        return (os.getenv("GEMINI_API_KEY"),
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "gemini-embedding-001",
                "gemini-3.1-flash-lite", "gemini-3.1-flash-lite")
    if os.getenv("OPENAI_API_KEY"):
        return (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"),
                "text-embedding-3-small",
                os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY first.")


def parse_docx(path):
    """Return a Word file's paragraphs, each under its heading, and its tables as
    Markdown, in document order. A heading alone would match almost any question."""
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(str(path))
    tables = iter(doc.tables)
    blocks = []
    heading = ""

    for element in doc.element.body:
        tag = element.tag.split("}")[-1]
        if tag == "p":
            text = "".join(node.text or "" for node in element.iter()
                           if node.tag.endswith("}t")).strip()
            if not text:
                continue
            properties = element.find(qn("w:pPr"))
            style = properties.find(qn("w:pStyle")) if properties is not None else None
            if style is not None and str(style.get(qn("w:val"))).startswith("Heading"):
                heading = text
                continue
            blocks.append(f"{heading}\n{text}" if heading else text)
        elif tag == "tbl":
            table = next(tables, None)
            if table is None or not table.rows:
                continue
            header = [c.text.strip() for c in table.rows[0].cells]
            rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            rows += ["| " + " | ".join(c.text.strip() for c in row.cells) + " |"
                     for row in table.rows[1:]]
            blocks.append("\n".join(rows))
    return blocks


def ocr_image(path):
    """Return the words printed on an image, read by RapidOCR (pip only, no system
    install). Returns "" when RapidOCR is missing, and the run goes on without it."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return ""
    try:
        result, _ = RapidOCR()(str(path))
        return " ".join(line[1] for line in result).strip() if result else ""
    except Exception as exc:
        print(f"    OCR failed on {path.name}: {exc}")
        return ""


class Embedder:
    """The text embedding model and CLIP, which puts pictures and text in one space.
    CLIP is loaded on first use."""

    def __init__(self, api_key, base_url, embed_model):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.embed_model = embed_model
        self._clip = None

    def _load_clip(self):
        if self._clip is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            print("  loading CLIP...")
            self._clip = (CLIPModel.from_pretrained(CLIP_MODEL),
                          CLIPProcessor.from_pretrained(CLIP_MODEL), torch)
        return self._clip

    def text(self, text):
        """Embed text for the text index as a unit vector. A vector truncated to
        TEXT_DIM comes back shorter than 1, and L2 distance would count its length."""
        response = self.client.embeddings.create(
            model=self.embed_model, input=text, dimensions=TEXT_DIM)
        vector = np.array(response.data[0].embedding, dtype="float32")
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    @staticmethod
    def _to_vector(features):
        """Return CLIP's projected vector as a unit vector. transformers 5 wraps it in
        pooler_output, and indexing the wrapper would return per-patch states instead."""
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        vector = features[0].numpy()
        # CLIP is trained on cosine similarity, so only the direction means anything.
        # Unnormalised, the longer poster vector loses on L2 even when its direction
        # matches better.
        return vector / np.linalg.norm(vector)

    def image(self, path):
        """Embed a picture into CLIP's space for the image index."""
        from PIL import Image

        model, processor, torch = self._load_clip()
        inputs = processor(images=Image.open(path), return_tensors="pt")
        with torch.no_grad():
            return self._to_vector(model.get_image_features(**inputs))

    def clip_text(self, text):
        """Embed a question into CLIP's space to search the image index. It is not
        interchangeable with text(): another model, 512 dimensions, another space."""
        model, processor, torch = self._load_clip()
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            return self._to_vector(model.get_text_features(**inputs))


def build_indexes(embedder):
    """Build the text index and the image index. Both use IndexIDMap with ids from
    one metadata list, so a hit in either index leads back to its record."""
    import faiss

    metadata, text_vectors, image_vectors = [], [], []
    next_id = 0

    for path in sorted(DOCS_DIR.glob("*.docx")):
        print(f"  {path.name}")
        for block in parse_docx(path):
            metadata.append({"id": next_id, "kind": "text",
                             "source": path.name, "content": block})
            text_vectors.append(embedder.text(block))
            next_id += 1

    for path in sorted(IMG_DIR.glob("*")):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
            continue
        print(f"  {path.name}")
        metadata.append({"id": next_id, "kind": "image", "source": path.name,
                         "path": str(path), "ocr": ocr_image(path)})
        image_vectors.append(embedder.image(path))
        next_id += 1

    text_index = faiss.IndexIDMap(faiss.IndexFlatL2(TEXT_DIM))
    if text_vectors:
        ids = np.array([m["id"] for m in metadata if m["kind"] == "text"])
        text_index.add_with_ids(np.array(text_vectors).astype("float32"), ids)

    image_index = faiss.IndexIDMap(faiss.IndexFlatL2(IMAGE_DIM))
    if image_vectors:
        ids = np.array([m["id"] for m in metadata if m["kind"] == "image"])
        image_index.add_with_ids(np.array(image_vectors).astype("float32"), ids)

    print(f"  indexed {len(text_vectors)} text blocks and {len(image_vectors)} images")
    return metadata, text_index, image_index


def retrieve(query, embedder, metadata, text_index, image_index):
    """Return the TOP_K text hits plus the nearest image when the question asks for
    one. The two distance scales differ, so the hits are not merged by score."""
    by_id = {m["id"]: m for m in metadata}
    hits = []

    vector = np.array([embedder.text(query)]).astype("float32")
    distances, ids = text_index.search(vector, TOP_K)
    for distance, doc_id in zip(distances[0], ids[0]):
        if doc_id != -1:
            hits.append(by_id[doc_id])
            print(f"    text hit id={doc_id} squared L2={distance:.4f}")

    if any(word in query.lower() for word in IMAGE_KEYWORDS) and image_index.ntotal:
        vector = np.array([embedder.clip_text(query)]).astype("float32")
        distances, ids = image_index.search(vector, 1)
        for distance, doc_id in zip(distances[0], ids[0]):
            if doc_id != -1:
                hits.append(by_id[doc_id])
                # Also a squared L2 between unit vectors, but from another model:
                # a strong CLIP match sits near cosine 0.3, so do not compare this
                # number with the text ones.
                print(f"    image hit id={doc_id} squared L2={distance:.4f}")
    return hits


def describe_image(path, api_key, base_url, vision_model):
    """Ask the vision model to describe a picture. The description is only printed,
    not indexed or used in the answers."""
    import base64

    from openai import OpenAI

    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    suffix = Path(path).suffix.lstrip(".").replace("jpg", "jpeg")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=vision_model,
        messages=[{"role": "user", "content": [
            {"type": "text",
             "text": "Describe what this poster shows in two sentences: the "
                     "picture and its colours, not only the text."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/{suffix};base64,{encoded}"}},
        ]}],
    )
    return response.choices[0].message.content


def answer(query, hits, api_key, base_url, chat_model):
    """Answer from the hits only. Each passage is labelled with its source file, and
    an image contributes its OCR text."""
    from openai import OpenAI

    context, image_path = "", None
    for i, hit in enumerate(hits, 1):
        if hit["kind"] == "image":
            image_path = hit["path"]
            body = f"An image named {hit['source']}. Text found in it: {hit['ocr'] or 'none'}"
        else:
            body = hit["content"]
        context += f"[Source {i}: {hit['source']}]\n{body}\n\n"

    prompt = ("You are a Disney park assistant. Answer using only the background "
              "knowledge below, in a friendly and professional tone. If the answer "
              "is not there, say so rather than inventing one.\n\n"
              f"[Background knowledge]\n{context}[Question]\n{query}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = response.choices[0].message.content
    if image_path:
        reply += f"\n\n(Related image: {image_path})"
    return reply


def main():
    if not DOCS_DIR.exists():
        raise SystemExit(f"Missing {DOCS_DIR}")

    api_key, base_url, embed_model, chat_model, vision_model = pick_provider()
    print(f"Provider endpoint: {base_url}")

    # 1. Indexing
    print("\n--- 1. Indexing ---")
    embedder = Embedder(api_key, base_url, embed_model)
    metadata, text_index, image_index = build_indexes(embedder)

    # 2. Retrieval and answers
    print("\n--- 2. Retrieval and answers ---")
    for question in QUESTIONS:
        print(f"\n  Q: {question}")
        hits = retrieve(question, embedder, metadata, text_index, image_index)
        print(answer(question, hits, api_key, base_url, chat_model))

    # 3. Vision model
    poster = IMG_DIR / "02_halloween.jpeg"
    if poster.exists():
        print("\n--- 3. Vision model ---")
        print(describe_image(poster, api_key, base_url, vision_model))


if __name__ == "__main__":
    main()
