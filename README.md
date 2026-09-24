# ai-playground

A personal learning playground for hands-on AI engineering, from LLM applications and
retrieval to vision, classical ML and time series. The ten modules below each hold flat,
numbered scripts plus a topic write-up that explains the concepts behind them and records
the numbers those scripts produced.

Some scripts call a hosted model. Those all go through the OpenAI SDK, so you can move
between DeepSeek, Gemini and OpenAI by changing `base_url` and the key. The rest need no
API key at all: they run local models with Ollama or Transformers, train networks in
PyTorch on your GPU, or fit models with scikit-learn, gradient boosting and forecasting
libraries.

## Modules

| Module | Topics |
|--------|--------|
| [01-llm-foundation](01-llm-foundation/) | Chat protocol · prompt engineering · function calling · tool-loop agents · multimodal extraction · local deployment (Ollama, Transformers) |
| [02-rag](02-rag/) | Embeddings · chunking · vector databases · RAG pipelines · rerank · query rewrite · multimodal RAG · knowledge-base curation & versioning · GraphRAG |
| [03-text2sql](03-text2sql/) | Natural language to SQL · schema prompting · SQL agents · query safety · result evaluation |
| [04-agents](04-agents/) | Prompt templates & memory · chain orchestration · ReAct agents · MCP / A2A · LangGraph architectures |
| [05-fine-tuning](05-fine-tuning/) | Low-rank adaptation · supervised fine-tuning · reward-driven training · decode-time thinking budget · vision adapters |
| [06-multimodal-vision](06-multimodal-vision/) | Classical vision (colour, edges, Hough, HOG, Haar) · optical flow · training mechanics & loss behaviour · detection, segmentation and pose · attention & self-supervision · vision-language auditing · split and submission audits |
| [07-ml-dl-foundation](07-ml-dl-foundation/) | Classical ML · EDA pitfalls · gradient boosting · leakage & split discipline · thresholds · ensembling · networks from scratch up to frameworks |
| [08-time-series](08-time-series/) | Seasonal decomposition · stationarity · ARIMA / Prophet · periodic factors · LSTM windowing · rolling-origin backtesting |
| [09-lowcode-platforms](09-lowcode-platforms/) | Workflow engines from a declarative graph · node & plugin contracts · table knowledge bases · platform API protocol |
| [10-projects](10-projects/) | Join grain & aggregation · reported columns and bands · tool return shapes · chart criteria & index alignment · control-chart rules · label leakage · sample units · cohort analysis · retrieval backends · citation checks |

Where a script needs data, it either generates it with a fixed seed or reads a public
dataset that the topic write-up links to instead of committing; only small source documents
with no public download are kept in the repository. The figures and tables the scripts
produce are kept in each module's `outputs/` folder, so every write-up can be read without
running anything.

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch is deliberately left out of that file, because the right build depends on your
GPU. Modules 05 to 08 train on it, so install it before running those:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
pip install torch                                                      # CPU only
```

`requirements.txt` says which module needs what, and which packages have to match your
torch version.

### 2. Configure API keys

Only the scripts that call a hosted model need a key. Modules 05 to 08 train and run
locally and need none, so you can skip this step until you reach a module that asks.

Copy `.env.example` to `.env` at the project root and fill in your keys:

```bash
cp .env.example .env          # PowerShell: copy .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` | Text and reasoning, the primary provider |
| `GEMINI_API_KEY` | Multimodal vision and long context |
| `OPENAI_API_KEY` | Universal fallback |
| `OPENAI_BASE_URL` | Where `OPENAI_API_KEY` is sent. Point it at another vendor's OpenAI-compatible endpoint to use that vendor's models instead |

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

---

## Repository Layout

```
ai-playground/
├── README.md                     ← this index
├── requirements.txt              ← dependencies for all modules, torch excluded
├── .env.example                  ← template for API keys (never commit real keys)
├── .gitignore                    ← keeps data, secrets & model weights out of git
└── 01-llm-foundation/ ... 10-projects/
```

Every module folder has the same shape: one topic write-up and a set of flat, numbered
scripts, plus an `outputs/` folder where the modules that produce figures keep them. The
tables below list the scripts of each.

---

## Module 01 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_chat_sentiment_analysis.py` | Chat protocol & 3-way sentiment classification |
| 02 | `02_weather_function_calling.py` | Tool calling: a local weather function, called once per city |
| 03 | `03_table_multimodal_extraction.py` | Multimodal vision: table image → JSON |
| 04 | `04_ops_incident_handler.py` | AIOps agent tool loop: incident diagnosis |
| 05 | `05_prompt_engineering.py` | Structured templates, JSON mode, CoT, meta-prompting |
| 06 | `06_web_search_agent.py` | Search agent over the Wikipedia API, capped at three rounds, with a `tool_choice` circuit breaker |
| 07 | `07_ollama_local_chat.py` | Local deployment via Ollama: auto model pull, streaming, `<think>` split, FastAPI gateway |
| 08 | `08_transformers_inference.py` | Raw-weight inference with Transformers: HF download, GPU placement, chat template, throughput |

See [LLM-Foundation.md](01-llm-foundation/LLM-Foundation.md) for the concepts behind these scripts.

---

## Module 02 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_tfidf_hotel_recommender.py` | TF-IDF + n-grams: content-based recommendation over hotel descriptions |
| 02 | `02_word2vec_similarity.py` | Word2Vec training, persistence and vector arithmetic |
| 03 | `03_embedding_faiss_metadata.py` | Embeddings into FAISS with metadata, Matryoshka dimensions, persistence |
| 04 | `04_embedding_models_compare.py` | Two local models compared: pooling is a property of the model |
| 05 | `05_chunking_strategies.py` | Five chunking strategies scored side by side |
| 06 | `06_chatpdf_langchain_faiss.py` | PDF question answering with LangChain, FAISS and page-level citations |
| 07 | `07_disney_multimodal_rag.py` | Multimodal RAG by hand: dual indexes, CLIP, OCR, vision fallback |
| 08 | `08_query_rewriting.py` | Five rewrite types, single-prompt intent detection, search-engine rewriting |
| 09 | `09_rerank_and_multiquery.py` | Two-stage retrieval: BM25 recall then cross-encoder rerank, with query expansion |
| 10 | `10_kb_question_generation.py` | Doc2Query: generated questions as a second retrieval index |
| 11 | `11_kb_curation.py` | Conversation distillation and knowledge-base health auditing |
| 12 | `12_kb_version_management.py` | Version hashing, set-based diffing, A/B and regression testing |
| 13 | `13_graphrag_vs_vector.py` | One multi-hop question asked of a vector index and of a knowledge graph |

See [RAG-Retrieval-Augmented-Generation.md](02-rag/RAG-Retrieval-Augmented-Generation.md) for the concepts behind these scripts.

---

## Module 03 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_build_insurance_db.py` | Local SQLite from a fixed seed: five tables, commented DDL, idempotent rebuild |
| 02 | `02_prompt_to_sql.py` | Three prompt styles scored on rows *and* on stored-literal use, then retrieval-augmented |
| 03 | `03_langchain_sql_agent.py` | LangChain SQLDatabaseToolkit: what reflection gains, and the comments it drops |
| 04 | `04_vanna_text2sql.py` | Vanna over a local vector store: DDL, documentation and verified pairs, with corrections |
| 05 | `05_sql_quality_gate.py` | Screening, static rules, second-opinion review, read-only execution, benchmark by join depth |
| 06 | `06_sql_agent_with_tools.py` | Tool-calling agent: query, chart, linear fit and driver ranking in one loop |

See [Text2SQL-Natural-Language-to-SQL.md](03-text2sql/Text2SQL-Natural-Language-to-SQL.md) for the concepts behind these scripts.

---

## Module 04 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_prompt_templates_and_memory.py` | Templates, role-split messages, and a conversation that survives between calls |
| 02 | `02_lcel_composition.py` | Pipe-operator composition: retries, local steps, parallel branches, routing, streaming |
| 03 | `03_react_loop_from_scratch.py` | Reason-and-act by hand: no framework, plus the run where the tool list is withheld |
| 04 | `04_tool_agent_diagnosis.py` | The same loop inside a framework: typed tools, step budget, and vague descriptions compared |
| 05 | `05_mcp_client_and_server.py` | Both halves of the Model Context Protocol: stdio server, handshake, schema translation |
| 06 | `06_a2a_agent_protocol.py` | Agent-to-agent delegation: capability card, task submission, schema and auth rejections |
| 07 | `07_langgraph_topologies.py` | One state, five nodes, two topologies: fixed pipeline against a conditional router |

See [Agent-Systems-Loops-Protocols-and-Topologies.md](04-agents/Agent-Systems-Loops-Protocols-and-Topologies.md) for the concepts behind these scripts.

---

## Module 05 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_svd_image_compression.py` | Rank-k reconstruction, paired sign flips, storage accounting, and why energy share flatters |
| 02 | `02_als_low_rank_factorization.py` | Alternating least squares on a masked matrix: the penalised objective against the printed error |
| 03 | `03_lora_low_rank_hypothesis.py` | An unconstrained weight update, decomposed and compared against the frozen weight and noise, then re-measured as the task widens |
| 04 | `04_lora_sft_instruction_tuning.py` | Supervised tuning end to end: rule-made labels, prompt masking, scored against a prompted-rule baseline, save, reload, merge |
| 05 | `05_grpo_reward_shaping.py` | Group-relative policy optimisation by hand: five rewards, advantages, KL against the base |
| 06 | `06_thinking_budget_control.py` | Capping and extending a reasoning model's deliberation at decode time, without training |
| 07 | `07_vision_lora_gauge_reading.py` | A vision-language adapter on rendered panels, scored field by field |

See [Fine-Tuning-Low-Rank-Adaptation.md](05-fine-tuning/Fine-Tuning-Low-Rank-Adaptation.md) for the concepts behind these scripts.

---

## Module 06 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_color_tracking_and_optical_flow.py` | Colour thresholds, morphology, connected components, mean shift and CAMSHIFT, Harris corners, block matching and Lucas-Kanade |
| 02 | `02_edges_and_hough_voting.py` | Gaussian smoothing, Sobel, Canny by hand, then lines, circles and an arbitrary shape recovered by voting |
| 03 | `03_hog_and_haar_detectors.py` | Gradient orientation histograms into a HOG descriptor, every two-rectangle Haar feature, the integral image, and AdaBoost |
| 04 | `04_training_mechanics_xor_softmax_batchnorm.py` | XOR against a straight line, softmax cross-entropy by hand, and what a mode switch changes |
| 05 | `05_grid_detection_and_pose_assembly.py` | A detector's grid targets, loss, decoding and NMS, then people assembled from part affinity fields |
| 05b | `05b_loss_spikes_and_step_size.py` | Why that detector's loss breaks up near epoch 25, measured step by step and tested against batch order and step size |
| 05c | `05c_curvature_and_stability.py` | The loss surface's curvature by Hessian-vector products, and the step size each break-up happens at |
| 06 | `06_unet_segmentation_and_skip_connections.py` | A UNet against the same net without skips and one that never resamples, scored at the boundaries |
| 07 | `07_attention_and_self_supervised_representations.py` | Attention written out, two tokenisers, and three label-free objectives under one linear probe |
| 08 | `08_vlm_field_extraction_audit.py` | Forms rendered with five traps, scored field by field on a clean page and a photograph of it |
| 09 | `09_vlm_grounding_and_failure_modes.py` | A returned box scored under every coordinate convention, plus repetition and image-in-history checks |
| 10 | `10_video_keyframe_understanding.py` | A synthesised clip read by keyframe sampling, with the localisation error the stride buys |
| 11 | `11_document_layout_audit.py` | A PDF built from a known structure, parsed back, and its headings reconciled |
| 12 | `12_conv_kernels_and_feature_maps.py` | One kernel checked against nn.Conv2d by hand, then convolution, activation and pooling |
| 13 | `13_cnn_input_resolution_mismatch.py` | A 224-shaped stem on a 32x32 input: what the mismatch costs, and what it does not |
| 14 | `14_yolo_split_audit_and_submission.py` | A detection split audited before training, and two submission edits that move no box |

See [Multimodal-Vision-From-Pixels-to-Models.md](06-multimodal-vision/Multimodal-Vision-From-Pixels-to-Models.md) for the concepts behind these scripts.

---

## Module 07 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_build_tabular_datasets.py` | Four tables drawn from explicit formulas, with the coefficients printed so later scripts can be scored |
| 02 | `02_eda_that_silently_lies.py` | One file loaded two ways, same shape twice, and the check that tells them apart |
| 03 | `03_feature_engineering_and_boosting.py` | Seventy engineered features fed to CatBoost, then a count of how many it never used |
| 04 | `04_leakage_and_split_discipline.py` | Three leaks that improve the validation score while the model gets no better |
| 05 | `05_classifier_toolbox_and_thresholds.py` | Nine classifiers on one split, then the one number none of them chose |
| 06 | `06_ensembling_blend_vs_stack.py` | Four regressors combined four ways, traced back to the error correlation that paid for it |
| 07 | `07_neural_net_from_scratch.py` | A network in numpy alone, every gradient checked against a finite difference |
| 08 | `08_framework_abstraction_ladder.py` | The same network four times, from hand-derived gradients up to one call to fit |

See [Machine-Learning-and-Deep-Learning-Foundations.md](07-ml-dl-foundation/Machine-Learning-and-Deep-Learning-Foundations.md) for the concepts behind these scripts.

---

## Module 08 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_build_time_series_datasets.py` | Five series drawn from published mechanisms, with every factor and changepoint written to a truth file |
| 02 | `02_decompose_and_stationarity.py` | Decomposition scored at the right period and three wrong ones, and what a strong cycle does to a unit-root test |
| 03 | `03_arima_grid_search_and_forecast.py` | An AIC grid on a series of known order, and what a truncated candidate list does to the winner |
| 04 | `04_prophet_trend_seasonality_changepoints.py` | Trend, cycle and events each checked against what was planted, including a component that was not |
| 05 | `05_periodic_factor_baseline.py` | Three ways to fit a weekday and month-position effect, all scored against the planted factors |
| 06 | `06_lstm_windowed_forecast.py` | A series laid out as supervised rows, and the count of observations a random split puts on both sides |
| 07 | `07_rolling_origin_backtest.py` | Four cut-off dates, four routes, and a submission file checked after it is written |

See [Time-Series-Forecasting-Baselines-and-Backtests.md](08-time-series/Time-Series-Forecasting-Baselines-and-Backtests.md) for the concepts behind these scripts.

---

## Module 09 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_workflow_engine_from_spec.py` | Three declarative graphs validated, ordered and executed, including a batch body, a selector and two sub-workflow calls |
| 02 | `02_llm_node_output_contract.py` | A model node scored against the vocabulary the next node compares against, and the rows that vanish when it drifts |
| 03 | `03_plugin_io_contract.py` | A plugin held to its declared schema, and what a forgiving field mapper hands downstream |
| 04 | `04_table_knowledge_base_retrieval.py` | One table indexed two ways, and a three-condition question similarity cannot answer |
| 05 | `05_platform_api_protocol.py` | A local server on three endpoints, blocking against streaming, and a client that guesses its way to a wrong diagnosis |

See [Low-Code-Platforms-What-The-Canvas-Runs.md](09-lowcode-platforms/Low-Code-Platforms-What-The-Canvas-Runs.md) for the concepts behind these scripts.

---

## Module 10 Scripts

| # | Script | Feature |
|---|--------|---------|
| 01 | `01_build_project_datasets.py` | Five sources generated from an explicit specification, with the answer to every later claim printed alongside them |
| 02 | `02_join_grain_and_aggregation_audit.py` | A join at the wrong grain, and the same year total three ways: one of them 197x the truth and ranking the districts differently |
| 03 | `03_dashboard_metrics_and_cache.py` | A clamped ratio column, a band that drops 1,276 customers, and two cache-freshness rules that disagree once the source changes |
| 04 | `04_tool_return_shapes.py` | One question, five return shapes, scored against a computed answer: the shape that answers it is the smallest one |
| 05 | `05_chart_criterion_and_index_alignment.py` | A chart rule reading rows where the axis needs distinct values, and a column that arrives mostly populated and entirely misdated |
| 06 | `06_bollinger_and_spc_rules.py` | A rolling band reported with the numbers behind each flag, and eight control rules, seven of them run-based, that catch different days rather than more |
| 07 | `07_label_leakage_and_importance_views.py` | A label one column and one threshold reproduce, and four importance measures that disagree on eleven features out of twelve |
| 08 | `08_association_rules_sample_unit.py` | The same holdings mined under three sample units, one of which makes every lift exactly 1.0 by construction |
| 09 | `09_cohort_is_not_a_time_series.py` | Neighbouring points sharing none of their population, a shuffle test, and two seasonal terms with no observations under them |
| 10 | `10_search_backends_and_ui.py` | Keyword and vector retrieval over one corpus, a cutoff in tokens rather than rows, and a failure isolated one layer at a time |
| 11 | `11_answer_routing_and_citation.py` | Two routers before answering, a four-field schema, and every cited page checked against the pages actually supplied |

Run `python 10_search_backends_and_ui.py --ui` to serve the same two backends behind a small web interface.

See [Applied-Projects-The-Errors-That-Do-Not-Raise.md](10-projects/Applied-Projects-The-Errors-That-Do-Not-Raise.md) for the concepts behind these scripts.
