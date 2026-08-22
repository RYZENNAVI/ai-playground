"""Build a multimodal RAG assistant over Word documents and images, without LangChain.

Demonstrates what a framework hides, by assembling the same pipeline by hand:
    1. Parse .docx files, keeping headings as context and tables as Markdown.
    2. Read images, optionally lifting their text with OCR.
    3. Embed text with a text model and images with CLIP.
    4. Index the two modalities into two separate FAISS indexes.
    5. Retrieve text always, and images only when the question asks for one.
    6. Describe an image with a vision model as a third, text-only route.
    7. Assemble a grounded prompt and generate the answer.

Module 02: RAG - Multimodal Disney Assistant.
"""

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
load_dotenv(Path(__file__).parents[1] / ".env")
load_dotenv(Path(__file__).parents[2] / ".env")

DOCS_DIR = Path(__file__).parent / "data" / "disney_kb"
IMG_DIR = DOCS_DIR / "images"

# Three routes exist for getting an image into a text-only prompt. All three are
# implemented below, but they are not three alternatives of equal standing:
#
#   1. ocr_image()      RapidOCR reads the text printed on the picture and
#                       stores it on the image's metadata record. This string is
#                       what actually reaches the prompt.
#   2. Embedder.image() CLIP encodes the picture itself into the image index, so
#                       a text query can locate it. It yields an id, never
#                       content - which is why routes 1 and 2 are really two
#                       halves of one path: route 2 finds the image, route 1
#                       says what is on it. Together they carry the main flow.
#   3. describe_image() A vision model looks at the picture and writes a
#                       description of it. The only route that sees the artwork
#                       rather than the words on it. Standalone here: its output
#                       is printed for comparison, not fed back into retrieval.

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
    """Return (api_key, base_url, embed_model, chat_model, vision_model).

    One provider covers all three roles, which keeps a single key and a single
    quota to reason about. Gemini comes first because it is the only candidate
    whose one model handles both chat and vision; DashScope and OpenAI follow so
    a key of any kind still gets the script running.
    """
    if os.getenv("GEMINI_API_KEY"):
        return (os.getenv("GEMINI_API_KEY"),
                "https://generativelanguage.googleapis.com/v1beta/openai/",
                "gemini-embedding-001",
                "gemini-3.1-flash-lite", "gemini-3.1-flash-lite")
    if os.getenv("DASHSCOPE_API_KEY"):
        return (os.getenv("DASHSCOPE_API_KEY"),
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "text-embedding-v4", "qwen-plus", "qwen-vl-plus")
    if os.getenv("OPENAI_API_KEY"):
        return (os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"),
                "text-embedding-3-small", "gpt-4o-mini", "gpt-4o-mini")
    raise SystemExit("Set GEMINI_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY first.")


def parse_docx(path):
    """Step 1: pull paragraphs and tables out of a Word file, in document order.

    A heading is never indexed on its own. It is short and keyword-dense, so it
    scores well against almost any question about the document and crowds a real
    answer out of the top results; carried as a prefix instead, it adds context
    to the passage it introduces. Tables become Markdown rather than flattened
    text because the row and column structure is exactly what a language model
    needs to read a price list correctly. Walking element by element preserves
    the original ordering.
    """
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
    """Step 2: read any text baked into an image.

    First of the three routes an image can take into the pipeline: lift the text
    off it and treat that text like any other passage. Cheap and local, but it
    only ever sees words that were printed on the picture.

    Uses RapidOCR, which installs from pip and needs no external binary -
    Tesseract is the better-known option but has to be installed system-wide
    first. When the package is absent this returns an empty string and the
    pipeline keeps running on CLIP vectors alone.
    """
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
    """Steps 3: text and image encoders, loaded lazily.

    Two encoders, two vector spaces. The text model is good at prose similarity;
    CLIP puts pictures and their captions in one shared space, which is the only
    reason a text query can find an image at all.
    """

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
        """Embed prose for the text index, returning a unit vector.

        Normalising is not cosmetic here. Some providers only guarantee unit
        length at the model's full dimensionality, and a truncated vector comes
        back with a norm well below 1. FAISS ranks by L2 distance, so an
        un-normalised vector lets sheer magnitude outweigh direction and the
        ranking degrades into noise.
        """
        response = self.client.embeddings.create(
            model=self.embed_model, input=text, dimensions=TEXT_DIM)
        vector = np.array(response.data[0].embedding, dtype="float32")
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    @staticmethod
    def _to_vector(features):
        """Pull a flat projected vector out of whatever CLIP returned.

        transformers <5 returned a plain tensor here. transformers 5 returns an
        output object whose pooler_output already carries the projected vector.
        Indexing [0] on the new object silently yields last_hidden_state, i.e.
        per-patch states of the wrong dimensionality, so check for the attribute.
        """
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return features[0].numpy()

    def image(self, path):
        """Embed a picture into CLIP's shared space.

        Second of the three routes: index the picture itself, so a text query
        can find it even when nothing legible is printed on it. The cost is a
        second vector space to keep separate from the text one.
        """
        from PIL import Image

        model, processor, torch = self._load_clip()
        inputs = processor(images=Image.open(path), return_tensors="pt")
        with torch.no_grad():
            return self._to_vector(model.get_image_features(**inputs))

    def clip_text(self, text):
        """Embed a query into CLIP's space so it can be compared against images.

        Note this is NOT interchangeable with .text(): different model, different
        dimensionality, different space. Mixing them up silently returns nonsense.
        """
        model, processor, torch = self._load_clip()
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            return self._to_vector(model.get_text_features(**inputs))


def build_indexes(embedder):
    """Step 4: encode the corpus into one text index and one image index.

    IndexIDMap lets us attach our own ids, so a hit in either index can be traced
    back to the same metadata list.
    """
    import faiss

    metadata, text_vectors, image_vectors = [], [], []
    next_id = 0

    print("\n--- 1-4. Indexing the knowledge base ---")
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
    """Step 5: always search text; search images only on demand.

    The two indexes hold vectors from different models, so their distances are on
    different scales and cannot be ranked against each other. That is why this
    takes all the text hits plus at most one image, rather than merging by score.
    """
    by_id = {m["id"]: m for m in metadata}
    hits = []

    vector = np.array([embedder.text(query)]).astype("float32")
    distances, ids = text_index.search(vector, TOP_K)
    for distance, doc_id in zip(distances[0], ids[0]):
        if doc_id != -1:
            hits.append(by_id[doc_id])
            print(f"    text hit id={doc_id} distance={distance:.4f}")

    if any(word in query.lower() for word in IMAGE_KEYWORDS) and image_index.ntotal:
        vector = np.array([embedder.clip_text(query)]).astype("float32")
        distances, ids = image_index.search(vector, 1)
        for distance, doc_id in zip(distances[0], ids[0]):
            if doc_id != -1:
                hits.append(by_id[doc_id])
                # Distances here run far larger than the text ones; CLIP vectors
                # are unnormalised, so do not compare this number to the above.
                print(f"    image hit id={doc_id} distance={distance:.4f}")
    return hits


def describe_image(path, api_key, base_url, vision_model):
    """Step 6: the third route to using an image, via a vision model.

    The other two are ocr_image (text printed on the picture) and
    Embedder.image (the picture as a CLIP vector). Here, instead of embedding
    the picture, ask a model to describe it and feed that text into an ordinary
    text pipeline. Slower and pricier per image, but the output is readable and
    searchable with no extra vector space to manage.
    """
    import base64

    from openai import OpenAI

    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    suffix = Path(path).suffix.lstrip(".").replace("jpg", "jpeg")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=vision_model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "What kind of poster is this? Answer briefly."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/{suffix};base64,{encoded}"}},
        ]}],
    )
    return response.choices[0].message.content


def answer(query, hits, api_key, base_url, chat_model):
    """Step 7: build a grounded prompt and generate.

    Every passage is labelled with the file it came from, and the instruction
    forbids going beyond them. Source labelling plus that instruction is the
    cheapest hallucination defence there is.
    """
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

    embedder = Embedder(api_key, base_url, embed_model)
    metadata, text_index, image_index = build_indexes(embedder)

    print("\n--- 5, 7. Retrieval and generation ---")
    for question in QUESTIONS:
        print(f"\n=== {question}")
        hits = retrieve(question, embedder, metadata, text_index, image_index)
        print(answer(question, hits, api_key, base_url, chat_model))

    poster = IMG_DIR / "02_halloween.jpeg"
    if poster.exists():
        print("\n--- 6. Vision model route (image described as text) ---")
        print(describe_image(poster, api_key, base_url, vision_model))


if __name__ == "__main__":
    main()
