# Module 01: LLM Foundation — Knowledge Summary

This document is the complete knowledge summary for the "LLM Foundation" topic, covering six areas:
model fundamentals, architectural innovation, model selection and compression, API integration,
self-hosted deployment, and prompt engineering. It corresponds to the 8 scripts in
`ai-playground/01-llm-foundation/`.

---

# Part I: LLM Fundamentals

## 1. The Evolution and Taxonomy of AI

### 1.1 Four Development Stages (by parameter scale)

| Stage | Core mechanism | Typical example | Parameter scale |
| :--- | :--- | :--- | :--- |
| **Early: expert systems** | Hard-coded systems built on hand-written rules | 1970s self-driving: encoding traffic rules (stop on red, yield to oncoming traffic) | —— |
| **Machine learning era** | Machines discover patterns from data on their own | House-price prediction: learning pricing rules from area, location, amenities | tens to thousands |
| **Deep learning era** | Neural networks simulate the connectivity of brain neurons | Recommender systems, image recognition | tens of thousands to millions |
| **Large-model era** | Massive data + massive compute | General-purpose, high-performance AI models | **trillions** |

*   **The core goal of AI**: enable machines to perform tasks that normally require human intelligence. The term "artificial intelligence" was first proposed at a conference in **1956**.
*   **Limits of the early stage**: unable to handle unforeseen situations; rules are impossible to enumerate exhaustively. Fundamentally a collection of hand-encoded human experience.
*   **Three pillars of the large-model era**: **GPU compute** (e.g. the H20 chip) + **massive data** (GPT-3.5 was trained on 45TB) + **large-model architecture**.
*   **The pattern of progress**: **from hand-written rules → learning from labelled data → massive self-supervised learning**.
*   **Parameter comparison**: GPT-3 reaches 175 billion parameters, while the human brain has roughly 250 trillion neurons.

### 1.2 Two Categories of AI

| | **Analytical AI (discriminative)** | **Generative AI** |
| :--- | :--- | :--- |
| Core task | Classify, predict, or decide over existing data | Create new content (text, images, audio, etc.) |
| Strength | High precision, high efficiency | Creativity and flexibility |
| Limitation | Can only work with patterns in existing data; **cannot create new content** | Faces data-privacy and copyright challenges |
| Examples | Licence-plate recognition, credit risk control | Text, image, audio generation |

### 1.3 Four Mainstream Generative Model Families
1.  **Large Language Models (LLM)**: trained on massive text corpora. Core capabilities are text understanding and generation, coherent multi-turn dialogue, and **few-shot learning**. Representatives: the GPT series, DeepSeek, Qwen.
    *   **Applications**: intelligent customer service (e-commerce after-sales), content creation (bulk ad-copy generation).
2.  **Text-to-Image / Text-to-Video models**: learn the association between images/video and their text labels, fusing concepts, attributes and styles into original work. Representatives: DALL-E, Midjourney, Sora, Stable Diffusion.
    *   **Applications**: product design ("a futuristic streamlined running shoe made from recycled ocean plastic" → concept renderings), film pre-visualisation.
3.  **Computer Vision models**: extract features from pixels and match them against known patterns to perform classification, object detection and segmentation. Representatives: **YOLO, ResNet (invented by Kaiming He)**.
    *   **Applications**: smart manufacturing (body-panel defect detection, the SAIC Volkswagen case), medical imaging (CT lesion annotation).
4.  **Autonomous driving models**: integrate vision recognition, sensor fusion (cameras + LiDAR) and decision planning.
    *   **Technical challenges**: environmental perception accuracy (e.g. the Li Auto billboard misidentification case), safety and reliability.

---

## 2. How an LLM Is Trained (using ChatGPT as the example)

### 2.1 Model Evolution and Scale-up
`Transformer (2017) → GPT (2018.06) → GPT2 (2019.11) → GPT3 (2020.05) → InstructGPT (2022.01) → ChatGPT (2022.12)`

| Model | Release | Parameters | Pre-training data |
| :--- | :--- | :--- | :--- |
| GPT | 2018.06 | 117 million | ~5GB |
| GPT-2 | 2019.02 | 1.5 billion | 40GB |
| GPT-3 | 2020.05 | 175 billion | 45TB |

### 2.2 The Three-Stage Training Methodology
*   **Step 1: collect data, fine-tune a supervised model (SFT)**
    *   Select questions from a question bank → have humans write answers → fine-tune on GPT-3.5 to obtain a **supervised learning model**.
    *   Purpose: give the model basic instruction-following ability — getting it from **0 to 60 points**.
*   **Step 2: collect comparison data, train a reward model (RM)**
    *   Generate 4 answers for the same question → have humans **rank them** (D > C > A = B) → train a **reward model** on the ranking results.
    *   **The key design — why ranking instead of scoring**: when different annotators score the same sentence you get subjective disagreements like "5 vs 4" or "0.5 vs 1", and the model has no way to reconcile them. Switching to ranking (both annotators agree A > B) **makes it far easier for annotators to produce consistent labels**.
    *   **The human cost of RLHF**: Kenyan annotators performed the answer-ranking work at $2 per hour.
*   **Step 3: collect data, optimise with reinforcement learning (PPO)**
    *   Select new questions → the RL model generates answers → the reward model scores them → **iterate on the model continuously**.
    *   Purpose: improve generation quality on open-ended and creative tasks.

### 2.3 ChatGPT's Three Core Strengths
Understanding the user's true intent · ability to maintain context · comprehension of knowledge and logic.

### 2.4 Basic Capabilities and "Super" Capabilities
*   **Basic capabilities**: language generation, in-context learning, world knowledge.
*   **"Super" capabilities**: responding to human instructions, **generalising to unseen tasks**, code generation and code comprehension.

---

## 3. DeepSeek's Four Architectural Innovations
> **Thinking: why is DeepSeek fast to compute and cheap to run?** The answer splits into "architecture design" and "training strategy".

### 3.1 MLA (Multi-Head Latent Attention)
*   **The problem it solves**: the KV cache in traditional Multi-Head Attention (MHA) is a significant drag on computational efficiency.
*   **Core principle**: an attention mechanism using **low-rank joint compression of keys and values**, which shrinks the KV cache substantially while improving computational efficiency.
*   **Compression result**: only 5% of the storage is needed to retain 98% of the useful information; inference cost drops by 95%. Analogous to image compression — a 4MB original compressed to 200KB remains perfectly usable.

### 3.2 DeepSeek-MoE (fine-grained Mixture of Experts)
*   **Basic structure**: V3 uses 61 MoE blocks. The total parameter count is very large, but each inference pass activates only a small fraction of paths (roughly 7.3%).
*   **The core analogy (hospital triage desk)**: previously every patient had to see a general practitioner, which was highly inefficient. MoE is like adding a triage desk (the Router) that assigns patients to specialists. **DeepSeek's innovation was replacing the "security guard with no medical knowledge" with an "undergraduate who does have medical knowledge"** — training the Router with a dedicated two-layer neural network.
*   **Performance gain**: training cost drops by roughly 93%, and inference speed improves significantly.

### 3.3 Mixed-Precision Training Framework (FP8)
*   **Core idea**: higher precision means more memory and more expensive arithmetic, so modules that do not need high precision store their data in FP8.
*   **Implementation**: FP32 at critical positions, FP8 for non-critical modules, with automatic precision escalation when needed (**FP8 → BF16 → FP32**).
*   **Error control**: fine-grained quantisation strategies combined with high-precision accumulation solve the quantisation-error problem inherent in low-precision training.

### 3.4 MTP (Multi-Token Prediction)
*   During training, each position predicts multiple future tokens, increasing the density of the training signal and improving data efficiency; at inference time tokens are generated in batches rather than one at a time.

### 3.5 What the Innovations Have in Common
*   All four innovations revolve around a **"maximum value for money"** design philosophy — achieving 98 points of quality at 5% of the cost.

---

## 4. Reasoning Models and Reinforcement Learning (DeepSeek-R1)
> **Thinking: why is DeepSeek-R1 such a strong reasoner?**

### 4.1 Reasoning Models vs Traditional Models
*   **Weakness of traditional models**: they operate in "start writing immediately" mode, emitting answers without deliberation, which makes them error-prone on complex problems (e.g. college-entrance-exam maths).
*   **The reasoning-model innovation**: introduce an explicit `<think>` reasoning phase. This **scratch-pad style of thinking** noticeably improves accuracy; chains of thought can run to tens of thousands of characters.
*   **The double-edged sword**: thinking is thorough and systematic, but over-reasoning causes latency (the model keeps deducing after it already has the answer).

### 4.2 Driven by Reinforcement Learning (AlphaGo as the illustration)
*   **The limit of traditional training**: supervised learning depends on human annotation (e.g. rank lists); the roughly 1 million human game records became the ceiling on ability.
*   **The RL breakthrough**: self-play generated 30 million games, with win/loss outcomes automatically labelling data quality, ultimately producing moves no human had played (a 4:1 victory over Lee Sedol).
*   **Transferring the paradigm**: applying the same paradigm to NLP, using answer quality as automatic feedback, breaks through the data bottleneck.
*   **R1's specific mechanism**: **GRPO (Group Relative Policy Optimization)** as the reward mechanism, paired with rule-based verification for automatic scoring.

### 4.3 Three Training Routes
| Model | Method |
| :--- | :--- |
| DeepSeek-R1-Zero | Pure reinforcement learning (the first demonstration that RL alone, without SFT, can elicit long-chain reasoning and self-reflection) |
| DeepSeek-R1 | Cold-start SFT → RL → CoT + general-data SFT (800k) → full-scenario RL |
| Distilled small models | SFT directly on the 800k dataset above |

### 4.4 The Data Flywheel
*   **Turning a humanities student into a science student**: starting from V3 (the "humanities student"), train scientific reasoning ability using a few thousand seed samples generated by R1-Zero.
*   **The flywheel effect**: seed data → the model gains baseline ability → it generates 600k reasoning samples, producing a "small flame growing into a large fire" iteration.
*   **Data ratio**: within the 800k training set, a **science-to-humanities ratio of 3:1** is maintained to prevent the model from forgetting its foundational abilities.

---

## 5. Core Concept: Tokenisation

### 5.1 What a Token Is
A token is the **smallest unit** an LLM uses to process text. Models cannot read characters directly; a **tokenizer** must split text into tokens and convert them into numbers (vectors) for computation.

### 5.2 Chinese vs English Tokenisation (worked examples)
*   **English** `Hello World`: GPT-4o splits into `["Hello", "World"]` => token ids = `[13225, 5922]`
*   **Chinese** `人工智能你好啊`: DeepSeek-R1 splits into `["人工智能", "你好", "啊"]` => token ids = `[33574, 30594, 3266]`
*   **Chinese is not tokenised character by character but by semantic unit.** The tokenisation scheme directly affects both model efficiency and its grasp of linguistic detail.
*   **Cost impact**: both input and output are billed, and **Chinese has lower token efficiency**.
*   **Tool**: `https://tiktokenizer.vercel.app/` shows how different models split your text.

### 5.3 Deeper Tokenizer Details
*   **Case sensitivity**: humans read "hello" and "HELLO" as near-identical, but the machine assigns different token ids (e.g. 2002491 and 2375) because the model needs to recognise casing as a feature.
*   **Vocabulary size**: Llama has roughly **30,000** tokens, DeepSeek roughly **70,000**. Vocabulary size directly affects language coverage.
*   **Algorithmic differences**: the GPT series uses **BPE**; the BERT series uses **WordPiece**; Chinese models frequently use hybrid strategies.

### 5.4 Special Tokens (control markers)
These carry no lexical meaning; they act as "punctuation" or "commands":
*   **Separators**: distinguish text segments or roles, e.g. `<|user|>`, `<|assistant|>`, `<|im_start|>`.
*   **End-of-sequence (EOS)**: tell the model the text is finished, e.g. `[EOS]`, `<|endoftext|>`. **Essential for keeping answers complete without rambling.**
*   **Start tokens**: mark the beginning of a sequence, e.g. `[CLS]`, `[BOS]`.

---

## 6. Parameter Tuning: Temperature and Top P
> **Thinking: what are Temperature and Top P, and what do they do?** => Both control the diversity of generated text, but through different mechanisms.

### 6.1 Temperature
*   **Principle**: after the model computes the probability distribution over the next token, Temperature adjusts the **"smoothness"** of that distribution.
*   **High Temperature (1.0+)**: low-probability tokens become easier to select, making output more creative but potentially incoherent.
*   **Low Temperature (0.2)**: high-probability tokens carry more weight, making output more stable but more conservative.

### 6.2 Top P (nucleus sampling)
*   **Principle**: set a probability threshold P, accumulate token probabilities from highest to lowest until the sum exceeds P, and let the model **choose only from this "core" vocabulary**.
*   **High Top P (0.9)**: a larger candidate pool, more varied results. **Low Top P (0.1)**: a very small pool, more deterministic results.

### 6.3 Comparing the Two
For the sentence **"The weather today is really…"**, the next word might be: good (60%), decent (30%), bad (9%), cola (0.01%).
*   **High Temperature**: raises every word's probability, so even the irrelevant "cola" gets a chance.
*   **Top P (0.9)**: picks only the words whose cumulative probability reaches 90% — good (60%) + decent (30%) = 90% — **excluding "cola" outright**.
*   **Conclusion**: Top P adjusts the candidate count more dynamically and avoids extremely low-probability nonsense => higher-quality text.

### 6.4 Best Practice
Use high temperature (1.0+) for creative copy, low values (0.3-) for rigorous analysis.

---

# Part II: Model Selection and Compression

## 7. Model Compression: Distillation and Quantisation

### 7.1 Distillation
*   **Definition**: compress the knowledge a large model (the teacher) has learned into a small model (the student), so the latter is smaller and faster while retaining most of the capability.
*   **Common methods**: soft-label distillation (train the student on the teacher's probability distribution), hidden-layer distillation (align intermediate representations), data augmentation (the teacher generates synthetic data for further training).
*   **DeepSeek's practice**: six small models were distilled from R1's outputs; the 32B and 70B variants match OpenAI o1-mini on several capabilities, and overall outperform the closed-source GPT-4o and Claude-3.5-Sonnet.
*   **Cost-effectiveness**: a 7B model costs roughly 1% of the 671B model while delivering about 90% of the effect.
*   **The capability ceiling**: a student cannot exceed its teacher (8B will not beat 671B), but it will beat an undistilled model of the same size.
*   **Caveats**: teacher and student task distributions must match; do not set the temperature coefficient too high; fine-tuning is still needed after distillation to avoid catastrophic forgetting; hold out 5% of the data for validation to guard against regression.

### 7.2 Quantisation
*   **Definition**: lower the storage precision of parameters (FP32 → INT8/INT4) to reduce size and accelerate inference. FP32→INT8 cuts storage requirements by 75%.
*   **Four methods compared**:

| Method | Key characteristic | Best suited for |
| :--- | :--- | :--- |
| GPTQ | Uniform 4-bit quantisation, layer-by-layer weight optimisation | Maximum compression, VRAM-constrained environments |
| Unsloth dynamic quantisation | Mixed precision (FP16 for critical layers, 4-bit for the rest) | Preserving as much model performance as possible |
| QLoRA | 4-bit during training + LoRA, restored to FP16 for inference | Lowering training cost; deployment still needs FP16 |
| PTQ (post-training quantisation) | Permanently lower precision after training | Deployment optimisation, low-latency inference (edge devices) |

*   **Measured precision loss (FP16 vs 4-bit)**: carry errors in arithmetic, missing names in closed-book knowledge questions, close-but-wrong figures when locating information in long documents, 5-shot classification accuracy dropping from 94% to 78%, broken rhyme in creative writing.
*   **The rule of thumb**: **the lower the quantisation, the smaller and faster the model — and the more likely it is to fail on tasks requiring precise numerical reasoning or fine knowledge detail.**
*   **Enterprise practice**: companies typically just use a smaller full-precision model; quantisation is mainly for extreme compression (175B from 70GB down to 20GB) and low-VRAM scenarios.

---

## 8. Sizing LLMs: Selection and Deployment Cost

### 8.1 Trade-offs Across Three Size Tiers
*   **Small models (1.5B-14B)**: fast responses, low hardware requirements, but weak fundamentals (7B is unstable on basic text generation — sometimes outright "failing").
*   **Mid-size (32B+)**: **32B delivers roughly 90% of 671B's performance**, enough for professional domains, but needs 64GB RAM and 80GB VRAM.
*   **Full-size (671B)**: the strongest performance, suitable for AGI exploration, requires multi-node distribution, extremely expensive.

### 8.2 VRAM and Hardware Reference
```
deepseek-r1:1.5b —— 1-2 GB VRAM (no GPU required; CPU RAM suffices)
deepseek-r1:7b   —— 6-8 GB VRAM (a single 4090 can run it)
deepseek-r1:14b  —— 10-12 GB VRAM
deepseek-r1:32b  —— 24-48 GB VRAM
deepseek-r1:70b  —— 96-128 GB VRAM
deepseek-r1:671b —— 496 GB
```

### 8.3 Deployment Cost Estimates
*   **32B**: the enterprise value pick, roughly **RMB 100,000**.
*   **70B**: requires 8× L40 (RMB 65,000 each), roughly **RMB 520,000**. Suffers hallucinations caused by excessively long chains of thought.
*   **671B full model**: **RMB 2-3 million**.

### 8.4 An Important Correction
*   The publicly available 1.5B, 7B and 14B models are **"distilled" versions of Qwen/Llama tuned with R1's reasoning — they are not the real R1**. The genuine DeepSeek-R1 is the full 671B model.

---

## 9. The Industry Landscape: Applications, Rankings and Accelerators

### 9.1 DeepSeek's Market Impact
*   **Release timeline**: V3 on 2024-12-26 (a muted market reaction) → R1 on 2025-01-20, which is when it truly broke out.
*   **Training cost**: 2,788K H800 GPU-hours in total, roughly **USD 5.576 million** (at $2/hour) — about one tenth of comparable products.
*   **Pricing strategy**: the first to cut prices by 99%, triggering an industry price war.
*   **Market shock**: after R1's release, US tech stocks fell 17% in a single day on 27 January, wiping out over a trillion dollars in market capitalisation.
*   **Open-source licensing**: **DeepSeek-R1 is released under the MIT License**, permitting free use, modification and commercialisation, and explicitly allowing distillation to train other models. By contrast Meta's Llama licence restricts commercial use and requires derivative models to carry the LLaMA name. This directly **removes legal risk** and supports enterprise self-hosting.

### 9.2 Measured Impact of LLMs Across Industries

| Industry | Application | Result |
| :--- | :--- | :--- |
| Insurance | Intelligent policy-clause parsing | Text processing efficiency up **30×** |
| Finance | Intelligent credit risk control | Lending-risk judgement accuracy up **21.5%** |
| Healthcare | Automated extraction from medical records | Significantly faster record processing |
| HR | Intelligent candidate-profile classification | Model recognition accuracy reaches **99%** |
| Securities | Industry news extraction | Automated analysis of sector trends |
| Telecoms | Intelligent SMS classification and moderation | Significantly better filtering efficiency |
| E-commerce | Review sentiment analysis | Rapid build-out of review analytics systems |
| Logistics | Intelligent parsing of shipping addresses | —— |

### 9.3 Where the Reasoning Models Stand (Artificial Analysis Intelligence Index)
*   **Top of the index**: o3 scores 94 and o1 scores 90, with DeepSeek R1 close behind at 89 — the gap between the leading closed models and the strongest open-weight one is narrow enough that the licence, not the score, is often the deciding factor.
*   **Open-weight reasoning models**: by early 2025 the field included DeepSeek R1/V3, Kimi k1.5 (87), Step-R-mini (84), Baichuan M1-Preview (83) and the Qwen series — enough choice that self-hosting no longer means settling for a much weaker model.
*   **Practical recommendations**: Kimi K2 stands out on coding; Qwen offers strong value for money.

### 9.4 Accelerator Capacity and What It Means for Self-Hosting
*   **Why this matters here**: the choice of accelerator sets the ceiling on which model sizes are practical to serve, and the two figures that decide it are raw compute and memory bandwidth.

| Accelerator | Compute | Bandwidth |
| :--- | :--- | :--- |
| NVIDIA H100 | 989 TFLOPs | 3.35 TB/s |
| NVIDIA H20 | 148 TFLOPs | 4 TB/s |
| AMD MI300X | 1307 TFLOPs | 5.3 TB/s |

*   **Reading the table**: compute and bandwidth do not move together. The H20 has only about 15% of the H100's compute yet slightly more bandwidth — which matters because token generation is **bandwidth-bound rather than compute-bound**, so a part that looks far weaker on paper can still serve inference respectably while being a poor choice for training.
*   **Consequence for sizing**: pair this with the deployment-cost table in section 8 — the accelerator decides what fits, quantisation decides how much of it you need.

---

# Part III: API Integration in Practice

## 10. The Three "Superpowers" of AI Chat Products

### 10.1 Web Search
*   **Purpose**: compensate for the LLM's training-data cutoff => obtain external information.
*   **Workflow**: the user asks about recent news → the system recognises the need, automatically invokes a search tool and turns the question into several concise keywords → a search-engine API is called → the retrieved real-time information is fed back as context for the model to summarise.
*   **Example**: asked "what factors drive the price of gold?", the LLM calls a search tool and folds the results into its answer.

### 10.2 Reading Files (RAG)
*   **Technical basis**: Retrieval-Augmented Generation.
*   **Workflow**: upload a file → split the content into small **chunks** → convert them into vectors via **embeddings** and store them in a **vector database** → at question time the query is also vectorised, retrieving the most relevant chunks → the chunks plus the question go to the model to generate the answer.
*   **Example**: after uploading an annual report, asking "what was the second-quarter profit?" lets RAG pinpoint the relevant passage.
*   **Engineering parameter**: a chunk size of around **500** is recommended.

### 10.3 Memory (from "goldfish" to "companion")
*   **The premise**: an LLM is **stateless**; every conversation is a fresh interaction.
*   **Short-term memory**: the most recent few turns are sent along as background each time => the "**context window**".
*   **Long-term memory**: key facts (names, preferences) are extracted algorithmically and stored in a per-user database, then read back at the start of later conversations to provide personalised background.
    *   **Example**: tell the AI "I prefer concise answers" and it will tend to reply more tersely next time.
    *   **State of the art**: OpenAI's ChatGPT already implements long-term memory.

---

## 11. API Setup and Calling Conventions

### 11.1 API Keys and Security
*   **Obtaining one**: apply through each vendor's console. This module primarily uses DeepSeek (text reasoning) and Google Gemini (multimodal vision).
*   **Security note**: a key is usually shown in full only once at creation, after which the middle characters are masked (e.g. `sk-a****4c84`), so save it immediately.
*   **Reading it in code**: **never hard-code it**. Put everything in a `.env` at the project root, load it with `python-dotenv`, read it with `os.getenv()`, and add `.env` to `.gitignore`.

### 11.2 Standardise on the OpenAI SDK Protocol
Every major vendor offers an OpenAI-compatible endpoint, so **a single codebase can switch providers just by changing `base_url`** — there is no need to pull in each vendor's native SDK.

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",     # swap in another vendor's compatible endpoint to switch
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    temperature=0.7,      # randomness
    top_p=0.8,            # output diversity
    max_tokens=1500,      # maximum output length
    stream=False,         # whether to stream
)
result = response.choices[0].message.content
```
*   **Reading the response path**: `response.choices[0].message.content` — take the first of the model's returned candidates (choices) and read its message text.
*   **Streaming**: with `stream=True`, use `for chunk in response: print(chunk.choices[0].delta.content or "", end="")`.
*   **Common compatible endpoints**: DeepSeek `api.deepseek.com`; Gemini `generativelanguage.googleapis.com/v1beta/openai/`.

### 11.3 The Four Message Roles
*   `system` (system prompt): defines the AI's **global role, persona, behavioural rules and output-format requirements**. It should stay stable — changing it frequently destabilises output — and it consumes token budget on every call.
*   `user` (user input): the user's concrete instruction or question.
*   `assistant` (assistant reply): the model's generated content. In multi-turn dialogue it must be appended to the history manually.
*   `tool` (tool role): used in function calling to send the result of a local function back to the model.
*   **Best practice**: the system prompt and user prompt should have a **clear division of labour**.

### 11.4 Implementing Multi-Turn Dialogue
*   **Mechanism**: keep appending to the conversation history via `messages.append()`; the history must contain complete turns (user → assistant → user…).
*   **Example flow**: first judge "good sound quality" → return "positive"; append the user input "the speaker is bad" → return "negative".

### 11.5 Input and Output Token Limits
*   **Input limit (context window)**: the maximum information one call can process = **system prompt + conversation history + current user input**. Exceeding it makes the API error out.
*   **Management strategy**: manage history length yourself, **truncating or summarising older messages** to stay within the limit.
*   **Worked example**: with a 4096-token ceiling, if the system prompt and history already occupy 3500 => the user prompt cannot exceed `4096 - 3500 = 596` tokens.
*   **Output limit (`max_tokens`)**: caps the length of a single reply; set it too low and answers get cut off mid-sentence.
*   **Billing reference**: charged per token (e.g. Kimi-K2 input at $0.004 per thousand tokens); context lengths differ by model (Kimi-K2 supports 131,072 tokens).

---

## 12. Four Core Application Cases

### 12.1 Case: Sentiment Analysis (basic call)
*   **Technical point**: the `system` role turns the model into a **classifier with constrained output**.
*   **Steps**: set a system prompt defining the AI's role (public-opinion analyst) → the user supplies the text → specify the output format (return only "positive" or "negative").
*   **Core code**:
    ```python
    review = "The audio quality of this speaker is amazing, giving you unexpected sound!"
    messages = [
        {"role": "system", "content": "You are a professional sentiment analyst. Classify the sentiment of the product review. Reply with only one word: Positive, Negative, or Neutral."},
        {"role": "user",   "content": review}
    ]
    response.choices[0].message.content    # result: 'Positive'
    ```
*   **Switching roles**: change the system prompt to "you are an excellent copywriter…" and the output shifts from polarity judgement to marketing copy.
*   **Another example (coding assistant)**: `"You are a senior programmer. Provide code directly, wrapped in Markdown. No explanations, no small talk."` → the input "write binary search" yields complete code directly.
*   **Business value**: automated bulk labelling of e-commerce reviews and user feedback.

### 12.2 Case: Function Calling (weather lookup)
*   **Technical point**: the model has no real-time data and must borrow external tools.
*   **The five-step workflow**:
    1.  **Send the user query**: `query = "How is the weather in Shanghai and Shenzhen today?"`, build `messages` and make the request.
    2.  **Check whether a function call is needed**: `if hasattr(message, 'function_call') and message.function_call:`, then obtain `tool_name` and `arguments` (parsed with `json.loads`).
    3.  **Execute the function**: `tool_response = get_current_weather(location=..., unit=...)`
    4.  **Add the result to the conversation**: `tool_info = {"role": "function", "name": tool_name, "content": tool_response}`, then append it.
    5.  **Let the model produce the final answer**: make the request again.
*   **Terminology**: **function call** refers to the act of invoking a tool, while **function** refers to the concrete tool you wrote — two phrasings of the same concept.
*   **Debugging tip**: test the call flow with fixed values first and wire up the live interface later; Qwen-MAX is the recommended model.
*   **Error handling**: watch for exceptions such as `KeyError`, and always check attribute existence with `hasattr`.

### 12.3 Case: Table Extraction (multimodal Qwen-VL)
*   **Technical point**: pass an image URL plus an extraction instruction and let a vision-language model (VLM) perform complex OCR and document structure understanding, emitting JSON directly.
*   **The Qwen-VL model family**:
    *   **Qwen-VL (base)**: image captioning, visual question answering (VQA), OCR, document understanding, visual grounding.
    *   **Qwen-VL-Chat (instruction-tuned)**: SFT on top of Qwen-VL, optimised for conversational interaction.
    *   **Qwen-VL-Plus / MAX (upgraded)**: performance approaching GPT-4V, but not fully open-source.
    *   **Qwen2.5-VL (latest flagship)**: available in 3B, 7B and 72B.
*   **Output caveats**: specify JSON format; be careful with the exact form of text such as dates; **represent missing or invalid fields with `null` or an empty string**; adjust key-value pairs to the actual form structure; keep field names consistent.

### 12.4 Case: Incident Response for Operations (an early Agent)
*   **Technical point**: a `while True` loop implements multi-step diagnosis — the embryonic form of an agent.
*   **Four core stages**:
    1.  **Understand the alert**: combine third-party interface data to determine the anomaly (which object, which failure mode).
    2.  **Suggest an analysis method**: combine contingency plans, operations documentation and the model's own knowledge.
    3.  **Automatically gather analysis data**: call the performance-monitoring interface (connection count, load), the log-management interface and the incident-management interface (historical resolutions).
    4.  **Recommend and execute remediation**: form a plan and execute it after user confirmation. Typical measures include **tuning database configuration**, **hunting down abnormal sessions**, and **restarting the system or restoring from backup**.
*   **The core loop**:
    ```python
    while True:
        response = get_response(messages)
        message = response.output.choices[0].message
        messages.append(message)
        if response.output.choices[0].finish_reason == 'stop':
            break
        if message.tool_calls:
            fn_name = message.tool_calls[0]['function']['name']
            arguments_json = json.loads(message.tool_calls[0]['function']['arguments'])
            tool_response = current_locals[fn_name](**arguments_json)
            messages.append({"name": fn_name, "role": "tool", "content": tool_response})
    ```
*   **Business value**: less manual judgement time, standardised handling procedures, and further gains when combined with a knowledge base.

---

# Part IV: Self-Hosted Deployment

## 13. Three Self-Hosting Options

### 13.1 Comparison

| Option | Characteristics | Best suited for |
| :--- | :--- | :--- |
| **Ollama** | Minimal command-line deployment, automatic model management, built-in REST API on port 11434 | Individual development, single-user testing, small-scale use |
| **vLLM** | PagedAttention + continuous batching, high throughput | Production, multi-user / multi-GPU parallelism |
| **Local Python (Transformers)** | Load weights directly, full control | Custom generation logic, preparation for fine-tuning |

### 13.2 Downloading Model Weights
```python
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
)
```
*   `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` is the **Model ID** — the model's identity.
*   `allow_patterns` fetches only the necessary files, skipping repository images and legacy weight formats.
*   The function **verifies by hash**, so repeat calls skip already-downloaded files and interrupted transfers resume.
*   **Size reference**: 1.5B in full precision is about 3.5GB, 7B about 15GB (roughly 2GB per 1B parameters at FP16).

### 13.3 Ollama Deployment (the simplest)
*   **Two installation routes**: ① via WSL (`wsl --install` → `curl -fsSL https://ollama.com/install.sh | sh` → `ollama serve`); ② download the Windows build directly (simpler; it runs as a service after installation).
*   **Common commands**: `ollama pull / run / rm deepseek-r1:1.5b`.
*   **System requirements**: a dual-core CPU or better and at least 4GB RAM; a GPU is optional.

### 13.4 vLLM Deployment
*   **Origin**: open-sourced by Berkeley's LMSYS group, managing attention key-value memory efficiently through PagedAttention.
*   **Launch command and parameters**:
    ```bash
    vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
    ```
    *   `--tensor-parallel-size 2`: tensor parallelism, distributing the model across 2 GPUs.
    *   `--max-model-len 32768`: maximum context length of 32K tokens.
    *   `--enforce-eager`: disables CUDA Graph optimisation (more stable, slightly slower).
    *   For a locally quantised model, add `--quantization gptq --dtype half`.
*   **A measured pitfall**: even with conservative settings (max-model-len=4096, gpu-memory-utilization=0.8), **32B GPTQ-int4 still fails to run on a 24GB GPU**, raising `CUDA out of memory`. 7B is the better fit for a personal GPU.

### 13.5 Wrapping It as a Service (Ollama REST API + FastAPI)
*   **Ollama's default endpoint**: `http://localhost:11434/api/generate`, started automatically when `ollama serve` runs. Required parameters are `model` and `prompt`; setting `"stream": True` yields token-by-token streaming.
*   **Why wrap it in FastAPI**: to add extra functionality or custom endpoints and expose a standard API other systems can call. The core is `@app.post("/api/chat")` forwarding requests to port 11434, with CORS configured to allow cross-origin access.
*   **Dependencies**: `pip install fastapi uvicorn`. Flask is an alternative.

---

# Part V: Prompt Engineering

## 14. Prompt Fundamentals and Strategy Differences

### 14.1 How Prompts Work (autoregressive generation)
*   GPT converts input text into word vectors and generates words one at a time autoregressively. **Each word is predicted from the input prompt plus every word generated so far**, repeating until the answer is complete.
*   **The core formula**: **result = model + prompt**.
*   **What a good prompt is worth**: better answer quality, shorter interaction time (30-50% efficiency gain), and fewer misunderstandings.

### 14.2 General Models vs Reasoning Models

| | General model | Reasoning model |
| :--- | :--- | :--- |
| How to guide it | Needs **explicit step-by-step guidance** (e.g. CoT prompting), otherwise it may skip critical logic | Prompts can be **more concise** — state the goal and let the model apply its internalised reasoning |
| Design focus | Uses the prompt to compensate for weak spots (step decomposition, few-shot examples) | **No step-by-step instruction needed**; forcing decomposition actually degrades its reasoning |

### 14.3 Three Key Principles
*   **Model selection**: choose by **task type**, not by hype. Creative work suits general models; maths, physics, programming and other analytical reasoning suit reasoning models.
*   **Prompt design**: for general models => **supply whatever is missing**; for reasoning models => **state what you want, directly**.
*   **Pitfalls to avoid**: do not over-trust general models (complex reasoning needs decomposition); do not ask reasoning models trivially simple questions (they will over-elaborate); and reasoning models can still be wrong on hard problems, so multiple rounds may be needed.
*   **The golden rule**: model choice should depend entirely on the nature of the task, never on market buzz.

---

## 15. Prompt-Writing Principles and Framework

### 15.1 Five Writing Principles
*   **Define the goal**: state the task clearly.
*   **Give specific guidance**: e.g. "summarise this news article" → "summarise the following news in 3-4 sentences, covering the main event, people, time and place". The same applies to image generation — vague descriptions let the training-data distribution dominate (e.g. defaulting to Western model aesthetics).
*   **Be concise**: when a long explanation and a short instruction perform equally, the long one dilutes the key information.
*   **Guide appropriately**: **a good reference template is excellent guidance for the model** — you do not need to craft one specially; past work records serve directly, effectively giving the AI a "working manual".
*   **Iterate**: adjust based on output. For instance, first ask for an `a+b` function, then add "handle the case where a and b are not numbers".

> **Thinking: "specific guidance" and "be concise" pull in opposite directions — how do they fit together?**
> They are not opposed. The point is **to strip away modifiers with no substantive effect while preserving the critical constraints**. Follow the minimum-necessary principle: if one sentence says it clearly, do not use several.

### 15.2 The Six-Element Framework and Priority Order
*   **Elements**: ① Task (start with a verb) ② Context ③ Exemplars ④ Persona ⑤ Format ⑥ Tone.
*   **Priority order**: **Task > Context > Exemplars > Persona > Format > Tone**.
*   **How to use it**: when the answer disappoints, **check your prompt against these elements in order** and adjust. Writing prompts is a process of continuous trial and adjustment.

### 15.3 Some Effective Habits
*   **Emphasis**: repeating a command or operation is acceptable.
*   **Give the model an exit**: if it may not be able to answer, tell it to say "I don't know".
*   **Be as specific as possible**: in highly specialised scenarios, leave little room for interpretation.

---

## 16. Five Prompt-Writing Techniques

### 16.1 Constrain the Output Format
*   Emitting JSON structures the data for easy downstream parsing. For example, "extract the company name, ticker, revenue… from this financial statement and return JSON" yields `{"company": "Yili Group", "revenue": "32.5B CNY", ...}` directly.
*   Without JSON the output is a long paragraph requiring manual re-extraction. This is especially useful for pulling information out of unstructured documents such as PDFs.

### 16.2 Use Delimiters to Separate Input Sections
*   Use triple quotes or similar symbols to separate instructions, examples and caveats.
*   **Flexibility**: delimiters can be anything you like (three equals signs, four hashes, five asterisks) — as long as a human can see the boundary, the AI will understand the intent.

### 16.3 Provide Examples (few-shot)
*   **The management-trainee analogy**: treat the model as a management trainee — give it a complete worked example and it will imitate the pattern on new tasks.
*   **Typical scenarios**: handling sarcasm ("I'm so satisfied with opening this account — it's been a week and support still hasn't replied" is actually negative), explaining industry jargon, enforcing complex output formats.

### 16.4 Chain-of-Thought (CoT)
*   Decompose a complex task into several simple steps so the model thinks systematically.
*   **Typical uses**: arithmetic ("add 1000, subtract 500, then multiply by 1.2", with a fully worked example); customer-service workflow ("record the complaint type → set the priority → assign the department → generate a tracking number").
*   **What it achieves**: systematic problem solving, greater transparency and fewer errors, and better reasoning on similar tasks.
*   **Business value**: in specialised fields such as finance, a chain of thought lets the model learn the business side's standard operating procedure.

### 16.5 Explain to a Specific Audience
*   Example: `explain superconductors to me as if I were five years old.` → the model uses a slide analogy: current = a ball, an ordinary conductor = a rough slide (with friction), a superconductor = an ultra-smooth slide (frictionless).
*   **Applicable scenarios**: education, customer service, product documentation — anywhere the same content needs different registers.

---

## 17. Prompt Engineering in Practice (four cases)
All cases use the wrapper `get_completion(prompt, model="deepseek-v3")` with `temperature=0` for stable output.

### 17.1 Case: Completing a Task with a Prompt
*   **Scenario**: have the AI act as a telecom agent and identify the user's requirements for a mobile data plan (three attributes: name / monthly fee / monthly data).
*   **Template structure**: `# Objective {instruction}` + `# User input {input_text}`.
*   **Result**: the input "Help me subscribe to a 100GB plan with a budget under $30/month." yields monthly data = 100GB and a price ceiling of $30, with the plan name unspecified.

### 17.2 Case: Returning JSON
*   Add `# Output format {output_format}` to the template, set to "output as JSON".
*   **Result**: the output collapses to a JSON object with keys `name`, `price_limit`, `data_gb`, containing only the fields the user actually specified.

### 17.3 Case: Step-by-Step Reasoning with CoT
*   **Scenario**: judge whether a support reply meets the standard (must be polite, use an official register, accurately mention all three plan attributes, and not be a conversation-ender).
*   **The key**: adding `cot = "please analyse the dialogue step by step"` markedly improves reasoning. The dimensions checked are politeness → official register → completeness of information.
*   **Example verdict**: a reply opening with "Hun, we're currently promoting the Unlimited Plan…" is judged **non-compliant** — all three attributes are present, but the over-familiar address is not an official register.

### 17.4 Case: Using a Prompt to Tune a Prompt
*   Cast the AI as a "professional prompt author" and have it output three parts: the improved prompt, critical improvement suggestions, and at most 3 clarifying questions.
*   **Effect**: the improved support prompt includes a warm introduction, a clear scope of service and proactive needs discovery, and generates follow-up questions such as "roughly how much data do you use per month?" and "what is your budget range?"
*   **The core trick**: let the AI role-play an expert and iterate on the prompt automatically.

---

# Part VI: Integrated Case Study

## 18. Case: Building a Bouncing-Ball Simulation with DeepSeek + Cursor

### 18.1 Environment Setup
*   Add the `deepseek-r1` and `deepseek-v3` models under `File -> Preferences -> Cursor Settings`, supplying a custom key and base URL through the OpenAI protocol.

### 18.2 Five Rounds of Iterative Development
1.  **Round one**: `a red ball moves inside a triangular region and bounces off the boundaries — write an HTML page` → R1 thinks at length and produces HTML, but it does not behave correctly.
2.  **Polishing the styling**: `building on your earlier reasoning, improve the HTML` → the styling improves (gradient background, shadowed ball), but the ball still escapes the triangle.
3.  **The key debugging round**: `the ball flies out after bouncing — please check the code` → the model self-diagnoses a **normal-vector calculation error**: the old code treated the edge direction as the normal. The correct approach is to take the perpendicular of the edge and normalise it: `normal.x = edge.y / edgeLength`, `normal.y = -edge.x / edgeLength`. After the fix the ball behaves correctly.
4.  **Physics enrichment**: successively add gravity / normal force / elasticity, randomised launch direction, stronger bounce, and horizontal rolling velocity along the ground.
5.  **Engineering polish**: add a refresh button, fix garbled Chinese (add `<meta charset>`), make the page responsive.
*   **The finished artefact**: 215 lines of complete HTML with Canvas rendering, gravity/elasticity/friction simulation, triangular boundary collision detection, and mobile responsiveness.

### 18.3 Choosing Between V3 and R1 for Coding
> **Thinking: when writing code, when should you use V3-0324 and when R1?**
*   **V3-0324**: everyday programming, rapid development, front-end code generation, routine scripting.
*   **R1**: computation-heavy maths, complex algorithms, deep optimisation of code logic, tasks that need visible reasoning.
*   **Practical tips**: avoid R1 on simple tasks to prevent over-thinking from introducing hallucinations; when hallucinations appear, start a new conversation rather than continuing to patch.

---

# Appendix

## 19. Engineering, Environment and Troubleshooting

### 19.1 Three Tiers of Development Environment

| Tier | Setup | Notes |
| :--- | :--- | :--- |
| Beginner | `learn.bananaresearch.cn` online environment | Jupyter pre-installed with example code uploaded; edit and run `.ipynb` directly. Account convention: prefix the phone number with `u` |
| Intermediate | VS Code + Python 3.10+ | Local development (**Python >= 3.10 recommended**) |
| Efficient | Cursor AI coding tool | AI-assisted programming |

### 19.2 Common Troubleshooting
*   **`ModuleNotFoundError`**: a missing dependency → `pip install <package>`; project dependencies are recorded centrally in the root `requirements.txt`.
*   **File formats**: `.py` is a plain Python script; `.ipynb` is the Jupyter notebook format supporting cell-by-cell execution and visualisation. The two are interconvertible.
*   **API changes from model upgrades**: model APIs may change their parameter format over time, so check the response dictionary structure and boundary conditions such as `hasattr(message, 'function_call')` to stay version-compatible.

### 19.3 Enterprise Self-Hosting Practice
*   Prefer self-hosted models, accessed through a gateway to internal model services, keeping the API calling convention uniform.
*   **Deployment tiers**: the full model needs a GPU cluster; a mid-tier configuration runs on an ordinary server; a lightweight version runs on a personal computer. Small models need only CPU and RAM, large models need a GPU; Docker can simplify the deployment process.
*   **Limits of self-hosting**: constrained model size, potentially constrained internet access, and high system-integration complexity.

### 19.4 The Prompt Engineer's Responsibilities
*   **In essence**: a prompt is the instruction for talking to an AI, comprising a pre-set system role and the user's concrete question.
*   **Two core duties**: ① write the `system`-role prompt (e.g. "act as an assistant that names companies"); ② when packaging an application, design the `user` prompt template so end users have a lower barrier to entry.

### 19.5 Industry Frontiers and Security
*   **Frontier application (chip design)**: use LLMs for paper comprehension (arXiv), CAD drawing analysis and requirements-document collaboration. **KV Cache techniques** can be applied to on-device smart-speaker deployment to improve response speed.
*   **The white-text injection attack**: attackers embed text invisible to the human eye (white on white) in a document, carrying hidden malicious instructions that induce the AI to give a faulty review or a fake endorsement.
*   **Enterprise defence**: establish a hybrid defence of "**IT security review + private knowledge base**" to guard against prompt injection.

---

## 20. Frequently Asked Questions

*   **Q: What is the first thing to do before writing any code?**
    *   **A**: Apply for an LLM API key and configure the environment variables. DeepSeek (`platform.deepseek.com`) is recommended for text reasoning; Google AI Studio's Gemini is recommended for multimodal vision (it has a free tier). Put the key in the project root's `.env`, read it with `os.getenv()`, and **never hard-code it**.

*   **Q: Which models are recommended for getting started?**
    *   **A**: Three, by task type — **`deepseek-chat`** (general dialogue, the default in this module's scripts), **`deepseek-reasoner` / `deepseek-r1`** (reasoning model, emits a chain of thought), and **`gemini-2.5-flash`** (multimodal vision, for images and documents).

*   **Q: Can you explain the four message roles in one line each?**
    *   **A**: `system` is the system prompt; `user` is the user prompt; `assistant` is the model's returned content, i.e. the LLM's inference result; **`tool` is the tool's return value — literally what the function returns**. That last plain-language phrasing is what makes the `tool` role in function calling click.

*   **Q: I can't follow or write the code — what is the right way to learn?**
    *   **A**: Follow the two-step approach: **1. get it running** (make the example work in your own environment first); **2. talk to the AI and modify it into your own**.
    *   **The accompanying point**: **AI coding tools can help you write code** — actively use tools like Cursor to get from "it runs" to "it's mine".
    *   **The learning path**: first make sure the example runs → consult reference material to understand it deeply → use AI assistance to work through problems.

*   **Q: How should a text-to-image prompt be written?**
    *   **A**: A typical prompt reads "a futuristic streamlined running shoe made from recycled ocean plastic", which illustrates the standard structure of an image prompt: **subject + style + material detail**.

*   **Q: How do I read the parameters in Qwen3-Coder-480B-A35B?**
    *   **A**: **480B is the model's full parameter count**, and **A35B means 35B parameters are activated** — precisely the MoE trait of "large in total, small in activation".

*   **Q: Where do I find models, what hardware does each size need, and how do I download them?**
    *   **A**: Search on the HuggingFace Hub. Hardware mapping: **7B => needs a GPU**, **1.5B => CPU RAM** is enough. Download with `snapshot_download`; the repository path is the **Model ID**.

*   **Q: What are the ways to run a downloaded model?**
    *   **A**: Three — ① deploy with vLLM; ② call the deployed model from Python; ③ Ollama (the simplest).

*   **Q: Are `DeepSeek-R1-Distill-Qwen-1.5B` and Ollama's `deepseek-r1:1.5b` the same thing?**
    *   **A**: **They are the same parameters.**

*   **Q: How do I wrap it as an API for others to use? Is FastAPI installed with `ollama pull`?**
    *   **A**: Option 1 is FastAPI, option 2 is Flask, installed with `pip install fastapi uvicorn` (`ollama pull` only pulls models). The prerequisite is deploying locally with Ollama, which exposes **port 11434**, and then wrapping that port with FastAPI.

*   **Q: In enterprise development, are self-hosted models mainly deployed on servers or locally? Does API deployment need a GPU?**
    *   **A**: Mainly on **servers** (a typical configuration is 4×4090). API deployment **still needs a GPU — the API is only an interface wrapper**.

*   **Q: When a prompt doesn't give the result I want, how do I debug it? Is prompting the same as training?**
    *   **A**: **Throw the problem back at the model**, using two lines to trigger reflection: *"Your answer isn't quite what I was looking for."* / *"What else would you need to know about this question?"* Prompting is not training — it does not change the model's parameters.

*   **Q: After self-hosting a small model, can I still train it on domain knowledge?**
    *   **A**: **The two do not conflict.** You can fine-tune that small model => produce new parameters => redeploy and use it.

*   **Q: Among the user, the application developer and the model, who writes the prompt? Why does the prompt-engineer role exist?**
    *   **A**: **A prompt is written by a person**, and the user's question can be backed by a persona built into the LLM. Anyone can instruct an LLM, but doing it **programmatically** is what the prompt-engineer role is for. The typical division: `system_prompt: act as an assistant that names companies` + `user_prompt: colourful flowers`.

*   **A learner's takeaway**: *"Self-hosting is a process of getting familiar with the model, and it pays off later when you get to real fine-tuning."*

---

## 21. Knowledge Summary

| Dimension | Core content | Technical detail / case | Key takeaway |
| :--- | :--- | :--- | :--- |
| **History of AI** | From expert systems to generative AI | 1956 the concept → 1970s expert systems → machine learning → deep learning → large models | **Parameter explosion**: GPT-3 at 175 billion vs 250 trillion human neurons |
| **LLM training** | The three-stage methodology | ① supervised learning (0 to 60 points) ② reinforcement learning (human feedback) ③ generalisation | **RLHF**: Kenyan annotators ranking answers at $2/hour |
| **DeepSeek architecture** | MLA / MoE / FP8 / MTP | MLA stores 98% of the information in 5% of the space; MoE's triage-desk mechanism | **The difference between MLA and MoE**; RL vs supervised learning |
| **Core concept** | Tokenisation | Chinese/English tokenisation differences, tokenizer tooling | **Cost**: both input and output are billed; Chinese is less token-efficient |
| **Parameter control** | Temperature vs Top-p | Higher temperature means more creativity; Top-p filters out low-probability options | **Best practice**: high temperature (1.0+) for creative copy, low (0.3-) for rigorous analysis |
| **Model compression** | Distillation + quantisation | 32-bit → 8-bit saves 75% of the space; dynamic mixed precision | **The difference between GPTQ quantisation and QLoRA fine-tuning** |
| **Model sizing** | 1.5B / 7B / 32B / 70B / 671B | 32B reaches roughly 90% of 671B's performance | **70B deployment cost estimate (RMB 520k / 8×L40)** |
| **Self-hosting** | Ollama / vLLM / Transformers | Weight download, environment variables, multi-GPU parallelism | **WSL compatibility**, intranet API isolation |
| **API capabilities** | The three superpowers | ① web search ② file reading (RAG) ③ memory | **Vector database**: chunk size around 500 |
| **API practice** | Four typical scenarios | ① sentiment analysis ② weather lookup ③ table recognition ④ ops alerting | **Role definitions**: system sets the persona, user asks, assistant answers |
| **Prompt engineering** | Principles + framework + techniques | Five principles, six ranked elements, five techniques | **Prompting differences between reasoning and general models** |
| **Dev environment** | Toolchain options | Beginner: online environment; intermediate: VS Code + Python 3.10+; efficient: Cursor | **Dependency management**: one `requirements.txt`, lazy-import heavy packages |
| **Security** | Defending against attacks | Hidden-instruction attacks in documents (white text carrying malicious prompts) | **Enterprise defence**: IT security review + private knowledge base |

---

## 22. Scripts Produced in This Module

The 8 scripts under `ai-playground/01-llm-foundation/`, all verified by actually running them:

| # | Script | Knowledge covered |
| :---: | :--- | :--- |
| 01 | `01_chat_sentiment_analysis.py` | The chat protocol (system/user/assistant) + zero-shot sentiment classification |
| 02 | `02_weather_function_calling.py` | The five-step function-calling loop, including parallel tool calls |
| 03 | `03_table_multimodal_extraction.py` | Multimodal vision: table image → structured JSON |
| 04 | `04_ops_incident_handler.py` | Agent tool loop — multi-step diagnosis of an operations alert |
| 05 | `05_prompt_engineering.py` | Four paradigms: structured templates, JSON mode, CoT, meta-prompting |
| 06 | `06_web_search_agent.py` | A web-search agent with an iteration cap and a `tool_choice` circuit breaker |
| 07 | `07_ollama_local_chat.py` | Local deployment: automatic model pull, streaming, `<think>` splitting, FastAPI gateway |
| 08 | `08_transformers_inference.py` | Raw-weight inference: download, GPU loading, chat template, throughput measurement |

**Measured performance of 07 vs 08** (same RTX 5070 Ti, same model, 128 tokens, warmed up, mean of three runs):

| Path | Precision | Speed | VRAM |
| :--- | :--- | :--- | :--- |
| Ollama (07) | GGUF Q4_K_M | **309 tok/s** | ~1.1 GB |
| Transformers (08) | bfloat16 | **29 tok/s** | 3.55 GB |

Quantisation paired with a purpose-built inference engine delivers roughly a **10×** throughput advantage,
at the cost of some precision. The conclusion: serve with Ollama/vLLM in production, and reserve the
Transformers path for cases where you need to observe the internals, customise generation logic, or
prepare for fine-tuning.

> ⚠️ **Security note**: teaching examples often hard-code keys as `api_key = "sk-XX"`. That is for
> demonstration only. In real use, always read keys from environment variables or a `.env` file, and
> **never hard-code them or commit them to git**.
