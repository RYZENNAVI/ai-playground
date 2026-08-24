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
| [03-text2sql](03-text2sql/) | Natural language to SQL · schema prompting · SQL agents · query safety · result evaluation |
| [04-agents](04-agents/) | Chain orchestration · ReAct agents · MCP / A2A · LangGraph architectures |
| [05-fine-tuning](05-fine-tuning/) *(draft)* | Low-rank adaptation · supervised fine-tuning · reward-driven training · vision adapters |
| [06-multimodal-vision](06-multimodal-vision/) *(draft)* | Vision-language auditing · grounding · keyframe sampling · convolution · detection metrics |
| [07-ml-dl-foundation](07-ml-dl-foundation/) | Classical ML · gradient boosting · neural networks · TensorFlow |
| [08-time-series](08-time-series/) | Stationarity · ARIMA / Prophet · seasonal decomposition · changepoint detection |
| [09-lowcode-platforms](09-lowcode-platforms/) | Coze workflows & plugins · Dify self-hosted deployment |
| [10-projects](10-projects/) | Capstones: enterprise knowledge base · ChatBI · AIOps assistant · AI search |

Modules marked *(draft)* are still being revised.

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
├── 03-text2sql/                  natural language to SQL, SQL agents, query safety
│   ├── Text2SQL-Natural-Language-to-SQL.md     topic write-up
│   └── 01..06_*.py               scripts
├── 04-agents/                    chain orchestration, ReAct, MCP/A2A, LangGraph
│   ├── Agent-Systems-Loops-Protocols-and-Topologies.md   topic write-up
│   └── 01..07_*.py               scripts
├── 05-fine-tuning/               low-rank adaptation, SFT, reward-driven training, vision adapters
│   ├── Fine-Tuning-Low-Rank-Adaptation.md      topic write-up
│   └── 01..07_*.py               scripts
├── 06-multimodal-vision/         vision-language auditing, grounding, keyframes, detection
│   ├── Multimodal-Vision-Calling-Models-and-Training-Them.md   topic write-up
│   └── 01..07_*.py               scripts
├── 07-ml-dl-foundation/          classical ML, gradient boosting, TensorFlow
├── 08-time-series/               stationarity, ARIMA/Prophet, seasonality, changepoints
├── 09-lowcode-platforms/         Coze workflows & plugins, Dify deployment
└── 10-projects/                  end-to-end capstone projects
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

---

## Module 03 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_build_insurance_db.py` | Local SQLite from a fixed seed — five tables, commented DDL, idempotent rebuild |
| 02 | `02_prompt_to_sql.py` | Three prompt styles scored on rows *and* on stored-literal use, then retrieval-augmented |
| 03 | `03_langchain_sql_agent.py` | LangChain SQLDatabaseToolkit — what reflection gains, and the comments it drops |
| 04 | `04_vanna_text2sql.py` | Vanna over a local vector store — DDL, documentation and verified pairs, with corrections |
| 05 | `05_sql_quality_gate.py` | Screening, static rules, second-opinion review, read-only execution, benchmark by join depth |
| 06 | `06_sql_agent_with_tools.py` | Tool-calling agent — query, chart, linear fit and driver ranking in one loop |

See [Text2SQL-Natural-Language-to-SQL.md](03-text2sql/Text2SQL-Natural-Language-to-SQL.md) for the concepts behind these scripts.

---

## Module 04 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_prompt_templates_and_memory.py` | Templates, role-split messages, and a conversation that survives between calls |
| 02 | `02_lcel_composition.py` | Pipe-operator composition — retries, local steps, parallel branches, routing, streaming |
| 03 | `03_react_loop_from_scratch.py` | Reason-and-act by hand — no framework, plus the run where the tool list is withheld |
| 04 | `04_tool_agent_diagnosis.py` | The same loop inside a framework — typed tools, step budget, and vague descriptions compared |
| 05 | `05_mcp_client_and_server.py` | Both halves of the Model Context Protocol — stdio server, handshake, schema translation |
| 06 | `06_a2a_agent_protocol.py` | Agent-to-agent delegation — capability card, task submission, schema and auth rejections |
| 07 | `07_langgraph_topologies.py` | One state, five nodes, two topologies — fixed pipeline against a conditional router |

See [Agent-Systems-Loops-Protocols-and-Topologies.md](04-agents/Agent-Systems-Loops-Protocols-and-Topologies.md) for the concepts behind these scripts.

---

## Module 05 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_svd_image_compression.py` | Rank-k reconstruction, paired sign flips, storage accounting, and why energy share flatters |
| 02 | `02_als_low_rank_factorization.py` | Alternating least squares on a masked matrix — the penalised objective against the printed error |
| 03 | `03_lora_low_rank_hypothesis.py` | An unconstrained weight update, decomposed and compared against the frozen weight and noise |
| 04 | `04_lora_sft_instruction_tuning.py` | Supervised tuning end to end — rule-made labels, prompt masking, save, reload, merge |
| 05 | `05_grpo_reward_shaping.py` | Group-relative policy optimisation by hand — five rewards, advantages, KL against the base |
| 06 | `06_thinking_budget_control.py` | Capping and extending a reasoning model's deliberation at decode time, without training |
| 07 | `07_vision_lora_gauge_reading.py` | A vision-language adapter on rendered panels, scored field by field |

See [Fine-Tuning-Low-Rank-Adaptation.md](05-fine-tuning/Fine-Tuning-Low-Rank-Adaptation.md) for the concepts behind these scripts.

---

## Module 06 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_vlm_field_extraction_audit.py` | Forms rendered with five traps, scored field by field on a clean page and a photograph of it |
| 02 | `02_vlm_grounding_and_failure_modes.py` | A returned box scored under every coordinate convention, plus repetition and image-in-history checks |
| 03 | `03_video_keyframe_understanding.py` | A synthesised clip read by keyframe sampling, with the localisation error the stride buys |
| 04 | `04_document_layout_audit.py` | A PDF built from a known structure, parsed back, and its headings reconciled |
| 05 | `05_conv_kernels_and_feature_maps.py` | One kernel checked against nn.Conv2d by hand, then convolution, activation and pooling |
| 06 | `06_cnn_input_resolution_mismatch.py` | A 224-shaped stem on a 32x32 input — what the mismatch costs, and what it does not |
| 07 | `07_yolo_split_audit_and_submission.py` | A detection split audited before training, and two submission edits that move no box |

See [Multimodal-Vision-Calling-Models-and-Training-Them.md](06-multimodal-vision/Multimodal-Vision-Calling-Models-and-Training-Them.md) for the concepts behind these scripts.
