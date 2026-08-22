# ai-playground

A curated portfolio of hands-on AI engineering work — LLM applications, RAG pipelines,
agents, fine-tuning, and deep learning. Every module is self-contained, documented,
and runnable.

Each module folder holds flat, numbered scripts plus a topic write-up explaining the
concepts behind them. Everything talks to providers through the OpenAI SDK protocol,
so switching vendors is a matter of changing `base_url`.

## Modules

| Module | Topics |
|--------|--------|
| [01-llm-foundation](01-llm-foundation/) | LLM fundamentals · OpenAI SDK · prompt engineering · function calling · local deployment (Ollama, Transformers) |
| [02-rag](02-rag/) | Embeddings · vector databases · RAG pipelines · rerank · query rewrite · GraphRAG |
| [03-langchain-toolchain](03-langchain-toolchain/) | LangChain · Text2SQL · tool-chain orchestration |
| [04-agents](04-agents/) | Agent architecture · MCP / A2A · multi-agent collaboration |
| [05-fine-tuning](05-fine-tuning/) | SFT / LoRA / QLoRA fine-tuning |
| [06-multimodal-vision](06-multimodal-vision/) | Vision-language models · PyTorch · YOLO object detection |
| [07-ml-dl-foundation](07-ml-dl-foundation/) | Classical ML · time series · neural networks · TensorFlow |
| [08-lowcode-platforms](08-lowcode-platforms/) | Coze workflows · Dify self-hosted deployment |
| [09-projects](09-projects/) | Capstones: enterprise knowledge base · ChatBI · AIOps assistant · AI search |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` at the project root and fill in your keys:

```bash
cp .env.example .env
```

| Provider | Environment Variable | Best For |
|----------|---------------------|----------|
| DeepSeek | `DEEPSEEK_API_KEY` | Text / reasoning (primary) |
| Google Gemini | `GEMINI_API_KEY` | Multimodal vision |
| OpenAI | `OPENAI_API_KEY` | Universal fallback |

### 3. Run any script

```bash
# Chat protocol and sentiment classification
python 01-llm-foundation/01_chat_sentiment_analysis.py

# Multimodal table extraction
python 01-llm-foundation/03_table_multimodal_extraction.py
```

### 4. Local models (optional)

Install [Ollama](https://ollama.com/). Scripts pull the model themselves on first run:

```bash
python 01-llm-foundation/07_ollama_local_chat.py
```

For raw-weight inference you also need PyTorch matching your GPU — see the notes in
`requirements.txt`.

---

## Repository Layout

```
ai-playground/
├── README.md                     ← this index
├── requirements.txt              ← unified dependencies for all modules
├── .env.example                  ← template for API keys (never commit real keys)
├── .gitignore                    ← keeps data, secrets & model weights out of git
│
├── 01-llm-foundation/            LLM basics, prompt engineering, tool calling, local deployment
│   ├── LLM-Foundation.md         topic write-up
│   └── 01..08_*.py               scripts
├── 02-rag/                       embeddings, vector DB, RAG pipelines, advanced recall
│   ├── RAG-Retrieval-Augmented-Generation.md   topic write-up
│   └── 01..13_*.py               scripts
├── 03-langchain-toolchain/       LangChain, Text2SQL, function calling
├── 04-agents/                    agent design, MCP/A2A, multi-agent
├── 05-fine-tuning/               SFT / LoRA fine-tuning
├── 06-multimodal-vision/         VLM, PyTorch, YOLO detection
├── 07-ml-dl-foundation/          classical ML, time series, TensorFlow
├── 08-lowcode-platforms/         Coze workflows, Dify deployment
└── 09-projects/                  end-to-end capstone projects
```

---

## Module 01 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_chat_sentiment_analysis.py` | Chat protocol & 3-way sentiment classification |
| 02 | `02_weather_function_calling.py` | Standard tool calling — weather service |
| 03 | `03_table_multimodal_extraction.py` | Multimodal vision — table image → JSON |
| 04 | `04_ops_incident_handler.py` | AIOps agent tool loop — incident diagnosis |
| 05 | `05_prompt_engineering.py` | Structured templates, JSON mode, CoT, meta-prompting |
| 06 | `06_web_search_agent.py` | Bounded web-search agent with a `tool_choice` circuit breaker |
| 07 | `07_ollama_local_chat.py` | Local deployment via Ollama — auto model pull, streaming, `<think>` split, FastAPI gateway |
| 08 | `08_transformers_inference.py` | Raw-weight inference with Transformers — HF download, GPU placement, chat template, throughput |

See [LLM-Foundation.md](01-llm-foundation/LLM-Foundation.md) for the concepts behind these scripts.

---

## Module 02 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_tfidf_hotel_recommender.py` | TF-IDF + n-grams — content-based recommendation over hotel descriptions |
| 02 | `02_word2vec_similarity.py` | Word2Vec training, persistence and vector arithmetic |
| 03 | `03_embedding_faiss_metadata.py` | Embeddings into FAISS with metadata, Matryoshka dimensions, persistence |
| 04 | `04_embedding_models_compare.py` | Two local models compared — pooling is a property of the model |
| 05 | `05_chunking_strategies.py` | Five chunking strategies scored side by side |
| 06 | `06_chatpdf_langchain_faiss.py` | PDF question answering with LangChain, FAISS and page-level citations |
| 07 | `07_disney_multimodal_rag.py` | Multimodal RAG by hand — dual indexes, CLIP, OCR, vision fallback |
| 08 | `08_query_rewriting.py` | Five rewrite types, single-prompt intent detection, search-engine rewriting |
| 09 | `09_rerank_and_multiquery.py` | Two-stage retrieval — BM25 recall then cross-encoder rerank, with query expansion |
| 10 | `10_kb_question_generation.py` | Doc2Query — generated questions as a second retrieval index |
| 11 | `11_kb_curation.py` | Conversation distillation and knowledge-base health auditing |
| 12 | `12_kb_version_management.py` | Version hashing, set-based diffing, A/B and regression testing |
| 13 | `13_graphrag_vs_vector.py` | One multi-hop question asked of a vector index and of a knowledge graph |

See [RAG-Retrieval-Augmented-Generation.md](02-rag/RAG-Retrieval-Augmented-Generation.md) for the concepts behind these scripts.
