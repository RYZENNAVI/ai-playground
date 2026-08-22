# Module 02: RAG — Knowledge Summary

This document is the complete knowledge summary for the "RAG" topic, covering the whole path from
text vectorisation to retrieval-augmented generation: the evolution of text representation,
embedding models and how to choose one, vector databases, the baseline RAG pipeline, chunking
strategies, advanced recall (rewriting, reranking, hybrid indexes, GraphRAG) and the operational
side of keeping a knowledge base healthy. It corresponds to the 13 scripts in
`ai-playground/02-rag/`.

Eleven sections: the first seven organise the knowledge by subject, the last four are practice
tasks, questions and answers, a summary table, and the script listing.

---

## 1. The Evolution of Text Representation: From Counting to Meaning

A computer cannot compare two pieces of text, only two numbers. This section covers the four steps
of turning text into numbers — **word counts → N-grams → TF-IDF → word vectors** — where each step
fixes a hole left by the one before it.

### 1.1 Segmentation and count vectors

*   **Segmentation**: split a sentence into its smallest processable units. Chinese needs a
    segmenter such as jieba; English splits on whitespace and punctuation.
*   **Building features**: collect every unique word across all sentences as a feature dimension,
    then count how often each sentence uses each one.
*   **Example** (two sentences differing by only a few words, over a 10-word vocabulary
    `and, code, is, messy, more, not, program, standard, that, this`):
    *   Sentence A: `this program code is messy and that code is standard` → `[1,2,2,1,0,0,1,1,1,1]`
    *   Sentence B: `this program code is not standard and that code is more standard` → `[1,2,2,0,1,1,1,2,1,1]`

### 1.2 Cosine similarity

*   **Formula**: `cosθ = (A·B) / (||A|| × ||B||)`, ranging over `[-1, 1]`, where 1 means the two
    vectors point the same way and -1 means they point opposite ways.
*   **Why the angle and not the distance**: vector length tracks how long the text is, while the
    angle only reflects **direction** — that is, the proportions of the words used.
*   The two sentences above score **0.8819**, close to 1, and are judged highly similar.

### 1.3 The fundamental flaw: word order is thrown away

Rearrange sentence B into `this program code is standard and that code is more not standard` and
**not one number in the count vector changes** — it is still `[1,2,2,0,1,1,1,2,1,1]` — **yet the
meaning has reversed**. The cause is that a unigram model counts occurrences only and
**ignores the effect of word order on meaning entirely**.

### 1.4 N-grams: putting order back into the features

*   **Assumption**: the n-th word depends only on the n-1 words before it and on nothing else.
*   **Naming**: N=1 unigram, N=2 bigram, N=3 trigram. The bigrams of `A B C D E` are
    `A B, B C, C D, D E`.
*   **Effect**: **combinations of adjacent words** become features in their own right, so
    "not standard" and "standard not" are two different features and the ordering difference is
    finally captured.

### 1.5 TF-IDF: putting weight into the features

*   **TF (term frequency)** = occurrences / total words in the document. The more a word appears
    here, the more it matters here.
*   **IDF (inverse document frequency)** = `log(total documents / (documents containing the word + 1))`.
    A word appearing in **fewer** documents discriminates better and gets a higher IDF. Words like
    "the" and "is", which appear everywhere, are pushed towards 0.
*   **Final feature value** = TF × IDF; words that never appear score 0.

### 1.6 Practice one: content-based recommendation with TF-IDF

**Goal**: given one hotel, recommend the 10 most similar ones in the dataset.
**Data**: **152** Seattle hotels with the fields `name / address / desc`.
**Method**: turn each description into a TF-IDF vector, compute the cosine similarity between the
target hotel and every other one, and take the top k.

**Seven steps** (see `01_tfidf_hotel_recommender.py`):

1.  Load the dataset and confirm its row count and columns.
2.  Inspect one description to see what the raw text looks like.
3.  Rank the most frequent n-grams after removing stop words.
4.  Clean: lowercase everything and drop stop words, so casing and filler words stop creating
    differences that are not real.
5.  Build the TF-IDF matrix over the cleaned descriptions (1-to-3-gram combinations, giving a
    measured vocabulary of **3348** dimensions and a matrix of shape `152 × 3348`).
6.  Compute cosine similarity into a symmetric `152 × 152` matrix whose diagonal is 1 (each hotel
    against itself).
7.  Sort by similarity and return the top 10 — **excluding the hotel itself**, via `iloc[1:11]`.

**Key components**: `CountVectorizer` (n-gram extraction), `TfidfVectorizer` (TF-IDF transform),
`linear_kernel` (fast cosine similarity), `pandas.Series` (sorting and indexing).

**Engineering detail**: nearly all of those 3348 features are zero, so **the matrix is extremely
sparse**; this is also why methods of this kind are memory-hungry.

**Measured**: querying `Hilton Seattle Airport & Conference Center` returns a top 10 made entirely
of airport-area hotels (first place `Embassy Suites ... Seattle Tacoma International Airport`,
0.2266); querying `The Bacon Mansion Bed and Breakfast` returns five B&Bs and inns at the top
(first place 0.2801).

> **Uniformly low scores (0.1–0.3) are normal in a sparse high-dimensional space** and do not mean
> the recommendations are poor — two hotels only have to word their descriptions differently for
> the cosine to drop. **What matters here is the ordering, not the absolute value.**

### 1.7 Word embeddings: from counting to meaning

*   **Definition**: a form of **dimensionality reduction** that turns different features into dense
    vectors of **identical dimensionality**.
*   **What it solves**: one-hot encoding of discrete variables is enormously wide (as wide as the
    vocabulary), and embedding compresses that to a fixed size, solving the
    **dimensionality explosion**.
*   **Generality**: anything can become a vector — words, sentences, images, products, users.
*   **Computability**: vectors can be compared directly, and a recommender simply takes the most
    similar ones.

**The "magic" of vector arithmetic**:

*   The classic identity `king - man + woman ≈ queen`. Measured,
    `most_similar(positive=["king","woman"], negative=["man"])` returns **queen (0.7226)**,
    princess (0.6473), throne (0.6337), prince (0.6322), elizabeth (0.6204). On the same model
    `paris + italy - france` returns venice (0.7671), florence (0.7268), vienna (0.7259) — **one
    set of vectors has learned both a gender relation and a country-to-city relation**, and nobody
    ever told it either concept existed.
*   **The control**: `king vs queen = 0.7264`, while `king vs banana = 0.0455`. The second number
    being near zero is the important one — it shows a high score is not simply what every pair gets.
*   **Mathematical basis**: element-wise addition and subtraction preserve dimensionality (50-dim
    in, 50-dim out).
*   **The point**: the computer does not "understand" meaning, yet **the result of the arithmetic
    agrees with human intuition** — this is the foundation for everything that follows.
*   **Dimensions are not interpretable**: no single dimension stands for a particular meaning; it
    is just a coordinate in a compressed mathematical space, and the size of that space is a choice.

### 1.8 Word2Vec: the classic implementation

*   **Core idea**: map words from their original space into a new one where semantically close
    words also sit close together.
*   **Input and output**: a one-hot vector in (10000 wide with a single 1), a probability
    distribution of size `[vocab_size]` out.
*   **Network**: the hidden layer has as many neurons as the embedding size (say 300); the weight
    matrix W is `[vocab_size, hidden_size]`.
*   **Why it is really a lookup table**: multiplying a one-hot vector by the weight matrix is
    **mathematically identical to selecting one row of that matrix**, and that row is the word's
    vector. Hidden units = embedding dimensions, and the hidden layer's output *is* the embedding.
*   **Compression ratio**: 10000 → 300. Powers of two are common (128/256/512/1024); a small corpus
    can drop to 50–100.
*   **Training objective**: self-supervised — mask the centre word and have the model predict its
    neighbours, pushing the predicted distribution towards the real one.

**Key Gensim parameters**:

```python
model = word2vec.Word2Vec(sentences)
# window     max distance between the current and the predicted word (commonly 3-5)
# min_count  minimum word frequency, default 5, filters rare words
# size       vector dimensionality, default 100
# workers    training threads
model.save(fname) / model.load(fname)
```

**Practice two: training word vectors on a Chinese novel** (see `02_word2vec_similarity.py`):

1.  Segment the raw Chinese text into words with jieba, writing to a segmented corpus.
2.  Train a baseline Word2Vec model on it.
3.  Measure similarity between character names.
4.  Solve analogies with vector arithmetic.
5.  Retrain with tuned hyper-parameters and save the model to disk.
6.  Reload it and confirm the answers are unchanged (verifying nothing was lost in persistence).
7.  Reproduce the king/queen analogy on a large English corpus.

**Measured**: after tuning, `vector_size=128 / window=5 / min_count=5` gives a vocabulary of
**7735** and trains in 1.2 seconds; after saving and reloading, vocabulary and similarities are
**identical** (`孙悟空 vs 猪八戒` is 0.9369 both times), so persistence lost nothing.

> **⚠️ Those scores are not evidence the model is good.** Every pair of characters in this corpus
> scores **above 0.9** (`孙悟空 vs 妖怪` reaches 0.9602). The reason is not model quality but that
> **a single novel is a small, stylistically repetitive corpus** — character names appear in nearly
> the same dialogue tags and narrative patterns, which pushes them into the same small region of
> the vector space.
> **That is exactly why step 7 repeats the recipe on a larger, more varied English corpus**: there
> `king vs banana` can fall to 0.0455, and the scores regain the ability to discriminate.
> **A set of uniformly high similarities usually says something about the corpus, not about the model.**

> **The Chinese corpus is kept here on purpose**: this section is about how Chinese is split into
> semantic units, and translating it away would leave the section with nothing to demonstrate.
> Step 7 supplies an English corpus for contrast.

**The application pattern**: the real value of Word2Vec is that it
**translates a problem into "words and documents"** — in a recommender, a product is a word and a
user's behaviour sequence is a document; in a follow graph, an account is a word and the order in
which someone followed accounts is a document. Any problem that can be shaped into such a sequence
can use the same method.

---

## 2. Embedding Models and How to Choose One

The word vectors above give **one vector per word**, whereas RAG needs **one vector per passage**.
This section covers how to pick that model and where the differences between models actually lie.

### 2.1 Embedding models vs large language models

| | Embedding model | Large language model |
| :--- | :--- | :--- |
| Nature | **Feature extractor / compressor** | **Generative model** |
| Output | A dense vector of fixed size | Text |
| Job | Vectorising meaning, similarity, filtering | Generating and understanding |
| Cost | Far lower | High (compare against reasoning models of hundreds of billions of parameters) |
| Limits | Classification, retrieval, ranking — feature extraction. **Cannot generate content** | Can generate, but far too expensive to use as a similarity function |

**The division of labour in RAG**: the embedding model **filters** (pulling a handful of
possibly-relevant items out of a large store) and the LLM **answers** (organising a reply from what
survived). **Vectorisation is not there to answer; it is there to filter.**

### 2.2 Where to start: the MTEB leaderboard

**https://huggingface.co/spaces/mteb/leaderboard**

Filter and sort by task type, language and model size, and compare candidates score by score. The
board moves quickly, so any hard-coded snapshot goes stale — **to see numbers, go to the address
above**.

**The task types MTEB covers** (when choosing, read **the column matching your scenario**, not the
overall average):

*   **Retrieval**: find the documents most relevant to a query — **the column RAG should read**.
*   **STS (semantic textual similarity)**: score sentence pairs on a continuous scale.
*   **Reranking**: reorder an initial retrieval result.
*   **Clustering**: group texts with no labels available.
*   **Pair classification**: decide whether two texts stand in some relation (duplicate questions,
    paraphrases).
*   **Bitext mining**: find mutual translations among sentences in two languages.
*   **Summarisation**: score machine summaries against human reference summaries by meaning.

> **Conclusion**: models specialise — some are stronger at retrieval, others at classification.
> **There is no overall winner, only a winner for your scenario.**

### 2.3 Four common families of embedding model

| Family | Representative | Key characteristics |
| :--- | :--- | :--- |
| **General text** | BGE-M3 | 100+ languages, **8192-token** input, fuses dense/sparse/multi-vector retrieval; weights around 2.3 GB |
| | text-embedding-3-large | **3072** dimensions, strong on long English text |
| | Jina-embeddings-v2-small | Only **35M** parameters, **RT < 50 ms**, lightweight and real-time |
| **Single-language** | bge-small-en, M3E-Base and similar | Precise on native phrasing, first choice for a single-language service |
| **Instruction-driven** | gte-Qwen2-instruct, E5-mistral | The query side needs an instruction prefix; supports cross-modal code and text retrieval |
| **Enterprise** | BGE-M3, E5-mistral | Hybrid retrieval, instruction fine-tuning |

> **The usage detail for instruction-driven models**: the query must be assembled as
> `Instruct: {task}\nQuery: {query}`, and **the document carries no prefix**. Omit that prefix and
> the model still produces vectors — they are simply wrong. A textbook case of "it runs, and it is
> incorrect".

### 2.4 Single-language vs multilingual

*   **Single-language models**: precise on native phrasing (fixed expressions such as
    "no-questions-asked return within seven days"), the first choice in a single-language setting.
*   **Multilingual models**: map several languages into **one shared semantic space**, so that
    `"clean room"`, `"部屋が綺麗"` and `"干净的房间"` land near each other — which is what makes
    cross-language retrieval and clustering possible at all.
*   **Typical scenario**: global review analysis for an international hotel chain — headquarters
    searches "Loud music at night" in English and has to surface the Japanese and Chinese reviews
    that say the same thing.
*   **The cost**: a multilingual model is usually slightly less precise in any single language than
    that language's dedicated model.

### 2.5 Matryoshka: one model, several dimensionalities

Taking Jina Embedding V4 as the example:

| Property | Value |
| :--- | :--- |
| Base model | Qwen2.5-VL-3B-Instruct |
| Max sequence length | 32768 |
| Single-vector dimensionality | 2048 |
| **Matryoshka dimensions** | **128, 256, 512, 1024, 2048** |
| Pooling | Mean pooling |

*   **The capability**: 2048 by default, but **truncatable down to 128 with modest loss**, selected
    through a `dimensions` parameter.
*   **Scenario 1 · short text at high volume** (sentiment on social comments): text is short,
    latency matters, resources are limited ⇒ use **128**. Note that **the index and the query must
    use the same dimensionality**, or the two are not comparable at all.
*   **Scenario 2 · high-value long documents** (financial reports, prospectuses): dense
    terminology, decisive details, expensive mistakes ⇒ use **2048**.
*   **The decision factors**: the value of the task (higher value leans towards more dimensions)
    and the latency requirement (tighter latency leans towards fewer).

**Measured**: embedding one sentence at 3072 / 1536 / 768 dimensions returns **identical leading
values** (all three start `-0.014099, -0.0218, -0.000503`) — truncation cuts the tail rather than
recomputing. Comparing retrieval over real documents at all three sizes, the **top-3 ordering is
identical**, with only small movements in the distances (doc3 scores 0.3139 / 0.3354 / 0.3142).

> **⚠️ A real trap in truncation**: some models **only emit unit vectors at full width**. After
> truncation the norm is no longer 1, and ranking by L2 distance then **lets vector length leak
> into the ranking**. Two scripts show this: `12_kb_version_management.py` prints a
> **mean raw norm of 0.620** after truncating to 1024 dimensions, and
> `07_disney_multimodal_rag.py` hit the same problem.
> **The fix: renormalise after truncating and before indexing.**
> (The norm of 1.0000 seen in `03_embedding_faiss_metadata.py` is exactly because it already does.)

### 2.6 Pooling is a property of the model, not a free choice

A model emits **one vector per token**, and turning that into one vector per passage requires
pooling. Three common choices:

*   **CLS pooling**: take the vector of the leading `[CLS]` token.
*   **Mean pooling**: average all valid token vectors (using the attention mask to exclude padding).
*   **Last-token pooling**: take the last valid token, common in instruction-driven models, with
    left and right padding handled separately.

**Each model fixes its pooling strategy during training. Getting it wrong raises no error; it just
quietly degrades the result.**

**Practice three: comparing two local models** (see `04_embedding_models_compare.py`):

1.  Score query-document pairs with a **CLS pooling** model.
2.  Score the same pairs with a **mean pooling** model.
3.  **Rebuild one of those scores by hand** from the model's raw outputs — take the hidden states,
    pool them, normalise them yourself.
4.  Check the hand-built vectors against the wrapper's.
5.  Deliberately use the wrong pooling strategy and see what happens.

**Measured**: the hand-written implementation deviates from the official wrapper by
**0.000000**; the wrong pooling deviates by **0.069623**.

> **The criterion was changed once here, and it is worth recording.** The first version used the
> margin (the gap between matching and non-matching pairs) to argue that wrong pooling degrades
> results — and **the conclusion flipped on a different dataset**: the wrong pooling produced the
> larger margin. Margin depends on sample count and is not comparable across models; it is simply
> not a proxy for correctness.
> Switching to **deviation from the official wrapper** made the criterion deterministic: identical
> is 0, different is not 0.
> **A conclusion has to rest on a mechanism, not on one dataset where it happened to hold.**

### 2.7 Engineering notes on selection

*   **768 dimensions** is a common price-performance balance point.
*   A CPU build of FAISS is enough for a lightweight deployment; no GPU required.
*   **Build your own test set**: run real questions from your own domain. The leaderboard is only
    for drawing up a shortlist.
*   If a general model underperforms on specialist data, train an embedding model on your own
    corpus (Word2Vec works, so do BERT-family models), then wrap it behind a standard interface.
*   You do not have to train one: **get something pre-trained working first, then decide.**

---

## 3. Vector Databases and Indexing

### 3.1 Comparing common vector databases

| Database | Characteristics | Strengths | Limits / fit |
| :--- | :--- | :--- | :--- |
| **FAISS** | Focused on high-performance similarity search, CPU/GPU, many ANN algorithms | Fast, many index types | Aimed at **static data**; updates and deletes are awkward |
| **Elasticsearch** | Distributed search engine with vector search as one of many features | Best-in-class **hybrid search**, keyword and semantic together | Mature in commercial settings |
| **Milvus** | Cloud-native, distributed, dynamic updates | Strong scaling, flexible data management | Large-scale enterprise use |
| **Pinecone** | Managed service, simple API | No operations burden, low latency | Fast validation |
| **Weaviate** | Built-in vectorisation modules | Simplifies development | Building a full pipeline quickly |
| **Qdrant** | Rust, memory-safe | Strong complex filtering | Scenarios with extreme performance demands |

Installation: `pip install faiss-cpu` (the GPU build installs separately).

### 3.2 The three steps of loading data

**Step 1 · Cleaning and preparation**
Ensure the raw data is sound. PDF, Word and presentation formats **must be converted to text first**, then
split, then loaded.

**Step 2 · Vectorisation**
Turn the raw data into vectors with a pre-trained embedding model. Text uses a text embedding
model; images use CLIP, ResNet, DINOv2 and similar.
**The choice of model directly determines vector quality and retrieval quality.**

**Step 3 · Load vectors together with metadata**

*   **Vector**: the array the embedding produced.
*   **Unique id**: identifies each data point, making later updates and deletions possible.
*   **Metadata**: everything describing the vector — source filename, section, URL, category, date,
    author. **Metadata is what makes advanced retrieval and citation possible.**

### 3.3 Practice four: embeddings + FAISS + metadata

See `03_embedding_faiss_metadata.py`, in eight steps:

1.  Embed one sentence and inspect the dimensionality and the shape of the values (signed floats
    reflecting many semantic features).
2.  Compare **Matryoshka truncation** at different output dimensions on the same input.
3.  Compare the **retrieval ranking** those dimensions produce on real documents.
4.  Embed the whole document set together with its metadata.
5.  Build a FAISS index mapping each vector to a custom id.
6.  Search the index with an embedded query.
7.  Resolve the returned ids back to documents and metadata.
8.  Persist the index to disk and reload it.

**Key points**:

*   **Index type**: `IndexFlatL2` is an **exact index** (brute-force comparison), wrapped in
    `IndexIDMap` so it can carry custom ids.
*   **What comes back is a distance, not a similarity**: `IndexFlatL2` returns L2 distance, where
    **smaller is closer**. To convert to "higher is better", use `similarity = 1 / (1 + distance)`.
*   **A query must use the same model and dimensionality as the index**, or the vectors are not in
    the same space at all.
*   **Invalid results**: FAISS can return the id `-1`, meaning no valid result, and loops must
    check for it.
*   **Persistence**: the index can be written and reloaded quickly; at run time it operates in
    memory.
*   **Scale**: FAISS handles billion-scale search, with typical responses in the hundreds of
    milliseconds.

### 3.4 Storing metadata robustly

*   **The simplest approach**: a Python list, with the list index used as the vector id. It works
    and it is fragile — the process exits and it is gone.
*   **The professional approach**:
    *   **Key-value store (Redis)**: lookup by id, extremely fast.
    *   **Relational store (PostgreSQL)**: suits structurally complex metadata.
    *   **Document store (MongoDB)**: stores JSON naturally.
*   **The architectural gain**: **separation of concerns** — FAISS handles vector search, and
    metadata goes to something built for it.

### 3.5 Where the vector store sits in RAG

```
query → vectorise → compare distances in the store → take top-k passages
      → resolve metadata back to source text → assemble into the LLM context → generate
```

*   **The core problem it solves**: the model's **context window limit** — a whole knowledge base
    cannot be pasted into a prompt.
*   **Vectors versus metadata**: vectors are for **computing similarity and selecting passages**;
    metadata is for **restoring the source text the model reads and labelling where it came from**.
*   **Multiple indexes**: several indexes can coexist (a text index and an image index, for
    example), with images handled by a cross-modal model such as CLIP.
    ⚠️ **Distances from different indexes are not directly comparable** — see 4.6.

---

## 4. The Baseline RAG Pipeline

### 4.1 Three ways to build on an LLM, and when to use which

The chain runs: pre-training on vast data → LLM → an AI with broad capability → a user asks →
**a wrong answer**. When the answer is wrong, treat the cause rather than the symptom:

| Cause | Symptom | Treatment |
| :--- | :--- | :--- |
| 1. The question was not clear | Vague wording, an answer to something else | **Prompt engineering** |
| 2. Missing background knowledge | It has never seen your data | **RAG** |
| 3. Missing capability | It cannot do this kind of task at all | **Fine-tuning** |

> **The normal order: RAG first, fine-tuning only if that fails.** RAG is cheap, quick to show
> results, and its knowledge can be updated at any time; fine-tuning needs data, compute, and a
> retraining run every time the knowledge changes.
> There are exceptions: if a client explicitly objects to long reasoning chains and wants short
> direct answers, fine-tuning fits better — **a special case, not the default path**.

### 4.2 What RAG is and what it solves

**RAG (Retrieval-Augmented Generation)** combines information retrieval with text generation by
**retrieving** relevant documents at request time and feeding them in as **context**.

Data flow: `Question → Retriever ⇄ Context → LLM → Response`

**Three advantages**:
① **Freshness** — training data is frozen, a retrieval store can be updated at any time;
② **Fewer hallucinations** — grounded answers are less likely to be invented;
③ **Depth in a specialist domain** — plug in a vertical knowledge base.

**If a model could handle unlimited context, would RAG still matter?**

*   **Efficiency and cost**: long contexts are expensive to compute and slow to answer; retrieved
    passages cut the input dramatically.
*   **Knowledge updates**: the model's knowledge ends at its training cutoff; retrieval can reach
    external sources.
*   **Explainability**: retrieval is transparent and the user can check the source; pure generation
    cannot be traced.
*   **Customisation and privacy**: retrieval can be tailored to a domain, and it can be restricted
    to local or private data.

> ⇒ **RAG is not there to make the model smarter. It is there to give it something to stand on.**

### 4.3 The core principle and its three steps

**Step 1 · Preprocessing**: multi-source data → **chunking** (balancing semantic completeness
against retrieval efficiency) → vectorisation → into the store.

**Step 2 · Retrieval**: vectorise the question → find the closest passages by similarity →
**rerank**.

**Step 3 · Generation**: retrieved passages + the question → an augmented context → an answer.

> **What "recall" means**: quickly finding the candidates that might be relevant out of a large
> store. Ten million items → **recall** narrows to a thousand → **reranking** picks the top 20.
> **Recall is fast and wide, reranking is precise and narrow** — two stages, two kinds of model.

> **Latency reference**: a first token in **under 10 seconds** is acceptable, **5 seconds or less**
> is the target.

### 4.4 Three steps, three hard problems

1.  **Indexing** ⇒ how to store knowledge well.
2.  **Retrieval** ⇒ how to find the small useful part out of a large store.
3.  **Generation** ⇒ how to turn the question plus the retrieved knowledge into a useful answer.

**The second step is where it breaks.** If retrieval goes astray, no amount of generation quality
rescues it. The cause is that **the two sides are phrased differently** — users ask in
conversational, context-dependent, vague, emotionally loaded language, while the store holds
declarative, neutral passages. This is exactly why section 6 exists.

### 4.5 Practice five: PDF question answering with page-level citations

See `06_chatpdf_langchain_faiss.py`. Stack: LangChain + FAISS over an OpenAI-compatible endpoint
(`gemini-embedding-001` for vectors, `gemini-3.1-flash-lite` for generation).

**Eight steps**:

1.  Extract text from the PDF **while remembering which page every character came from**.
2.  Split the text into overlapping chunks.
3.  Map each chunk back to the page it mostly came from.
4.  Embed the chunks and build the FAISS store.
5.  Persist it and reload it.
6.  Retrieve the chunks nearest a question.
7.  Answer through a QA chain and cite the source pages.
8.  Retrieve again under several phrasings and see what the first pass missed.

**Splitting parameters**: `RecursiveCharacterTextSplitter`, `chunk_size=1000, chunk_overlap=200`,
separator priority `paragraph → sentence → space → character` (`["\n\n", "\n", ".", " ", ""]`).

**Page citation is the point of this case, and it has a trap that cannot be avoided**:

*   **The wrong way**: record page numbers per line, then read `page_numbers[i]` for chunk i. The
    splitter cuts on semantic boundaries and merges up to `chunk_size`, so **there is no
    correspondence between line index and chunk index**, and the mapping is guaranteed to be wrong.
*   **The right way**: **record a page number for every character**, then assign each chunk the
    **modal page** of the characters it covers. This depends on no string matching, and a chunk
    spanning pages still resolves to the page that contributed most of it.

**Measured**: 11958 characters across 9 pages → **15 chunks**, `TOP_K=4`, with the two questions
citing pages `[5, 6, 3]` and `[6, 4, 7, 8]`.

> **⚠️ `TOP_K` has to be rechecked whenever the store changes size.** This script once used
> `TOP_K=10`: with a store of only 5 chunks, k=10 returns everything, so **the "source pages"
> looked precise while nothing had ever been filtered**. Growing the store to 15 chunks exposed it
> immediately — one answer cited 8 pages. Dropping k to 4 restored the meaning.
> **A parameter tuned for one version of the data must be retuned when the data changes.**

**Measured (step 8)**: adding three paraphrases of each question widens what retrieval can reach —
the first question goes from 3 pages to 4 (page 8 becomes newly reachable), the second from 4 pages
to 6 (pages 3 and 2 become newly reachable).

### 4.6 Practice six: a multimodal RAG assistant built by hand

See `07_disney_multimodal_rag.py`. The same job again without LangChain, in order to see
**what a framework actually does for you**.

**Goal**: a round-the-clock assistant for a theme park — answering the common questions about
tickets, entry rules and membership; every answer coming from the official knowledge base; and
**handling questions about images**.

**The challenges**: knowledge arrives in many formats (Word, PDF, web pages, event files with
charts); **unstructured processing** (extracting and understanding tables and images, which decides
whether RAG works at all); **organising the knowledge** (how to chunk and index a mass of scattered
facts); **answer validity** (staying strictly inside the retrieved content).

**Seven steps**:

1.  Parse `.docx`, **keeping headings as context** and converting tables to Markdown.
2.  Read images, optionally lifting their text with OCR.
3.  Embed text with a text model and images with CLIP.
4.  Index the two modalities into **two separate FAISS indexes**.
5.  Retrieve text always, and images **only when the question asks for one**.
6.  Describe an image with a vision model as a third, text-only route.
7.  Assemble a grounded prompt and generate the answer.

**Format handling: three file types, three approaches**

| Format | Approach |
| :--- | :--- |
| `.docx` | Walk the document body: paragraphs take the text branch; tables are **converted to Markdown** (read the header, add the separator row, read each row of cells) |
| `.pdf` | Read text page by page with its page number; extract embedded images, save them as `{name}_p{page}_{index}.{ext}` and record the paths |
| Images | OCR the text on the image, returning an empty string rather than crashing on failure |

> OCR uses `rapidocr-onnxruntime`: it installs through pip alone and needs no system-level binary.

**Three image routes, of which the first two are two halves of one chain**:

1.  **CLIP image vectors** (512-dim): index the images so that "finding a picture" is possible at all.
2.  **CLIP text vectors**: encode the question into **the same space**, making "find a picture with
    words" work. ① and ② together are text-to-image search — **they are not alternatives**.
3.  **A vision model describing the image**: image → text → text embedding, **falling back to
    ordinary text RAG**.

> **The distinction that matters**: the vectors from the text embedding model (1024-dim) and the
> vectors from CLIP's text encoder (512-dim) are **both text vectors, and they are not in the same
> space**. CLIP's image and text encoders were trained into one shared space by OpenAI in 2021 on
> 400 million image-text pairs — cross-modal retrieval works because somebody did that first.

**The hybrid strategy: all the text, and exactly one image**

Text results are added to the context in distance order, all of them; if image retrieval fired, the
**single** closest image is added. This is **a deliberately biased rule**, for the reason below.

> **⚠️ Distances from the two indexes are not comparable**: measured on the same question, the text
> distance is **0.9827** and the image distance is **123.6144**. The scales are unrelated, so
> **no single threshold can govern both**, and results from the two cannot be merged into one
> ranking. That blunt-looking rule exists because of this mathematical fact.

**Measured**: a knowledge base of 4 `.docx` files and 2 images produces **26 text blocks and 2 images**.

> **⚠️ Heading-only blocks poison retrieval**: when headings were indexed **on their own**, a
> five-word block such as `Ticket Rules` ranked near the top for any ticket question,
> **pushing the block that actually contained the refund clause out of the top k**, and the model
> could only reply that it had no information. Changing this so **headings are not indexed alone
> but prefixed as context onto the body blocks** dropped the text blocks from 30 to 26 and the
> refund question started answering correctly.

**How hallucination is held down**: the system prompt states plainly that the model must
**use only the information in the background knowledge and not invent anything**; the background is
listed item by item as `Background N (source: filename)` — **source labelling is the first line of
defence**.

### 4.7 QA chains and four ways to combine documents

A framework wraps "how several documents get handed to the model" into a `chain_type`. Four
strategies:

| chain_type | How it works | Character and fit |
| :--- | :--- | :--- |
| **stuff** | Concatenate every document into one prompt | Fewest model calls; **if stuff works, use stuff** |
| **map_reduce** | One call per chunk, then merge | Parallelisable, but **chunks have no context from each other** |
| **refine** | Answer from the first chunk, then merge in the rest one at a time | Keeps some context, token use stays controlled |
| **map_rerank** | Answer and score each chunk, take the best | **Most calls**, chunks entirely independent |

**The generation step is a piece of string concatenation, nothing more**:

```
Here is the user's question: {question}
Here is the relevant knowledge retrieved from the store: {chunk_list}
Answer the user's question using that knowledge. The answer is:
```

There is nothing more mysterious to it.

> **⚠️ Framework APIs move**: LangChain 1.3.15 removed `langchain.chains` entirely, and
> `load_qa_chain` no longer exists. `06_chatpdf_langchain_faiss.py` writes the stuff chain by hand
> with LCEL, which behaves identically.
> Separately, `OpenAIEmbeddings` sends token arrays by default and some compatible endpoints reject
> them, so `check_embedding_ctx_length=False` is needed to make it send raw strings.

---

## 5. Chunking and Building the Knowledge Base

### 5.1 Five chunking strategies

Chunking decides retrieval quality directly: too fine and meaning is severed, too coarse and noise
drowns the point.

| Strategy | How it works | Trade-offs and fit |
| :--- | :--- | :--- |
| **1 Improved fixed length** | Fixed size, but **backing off to a sentence boundary**, with overlap for continuity | Simple, fast, uniform. Fits: bulk processing |
| **2 Semantic** | Split on sentences and paragraphs, **no overlap** | Meaning intact, but **lengths can be wildly uneven** |
| **3 LLM-driven** | The model picks its own break points, balancing meaning and length | Intelligent breaks, **slow and expensive** |
| **4 Hierarchical** | Follow the document structure (headings, sections, paragraphs) | Preserves structure, **depends on the format** |
| **5 Sliding window** | A fixed window advances in fixed steps, producing overlap | Keeps context, improves recall, **redundant** |

The LLM prompt asks for three things: ① preserve semantic completeness ② break at natural points
③ return the chunks as JSON.

**Side by side**:

| Strategy | Meaning | Length control | Complexity | Speed | Fits |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Improved fixed length | Medium | Excellent | Simple | Fast | Technical docs, specifications |
| Semantic | Excellent | Medium | Medium | Medium | Natural prose |
| LLM-driven | Excellent | Excellent | Complex | Slow | High quality requirements |
| Hierarchical | Excellent | Poor | Medium | Medium | Structured documents |
| Sliding window | Medium | Excellent | Simple | Fast | Long documents |

### 5.2 Practice seven: five strategies over one text

See `05_chunking_strategies.py`. Six steps: five implement one strategy each, and the sixth scores
them side by side on the same text.

**Parameters**: `CHUNK_SIZE=800`, `OVERLAP=150`.
**Measured**: on 1299 characters of text, the spread of chunk lengths (longest minus shortest) is
**81 / 70 / 39 / 452 / 401** respectively.

**Three things to read out of those numbers**:

1.  **LLM splitting has the smallest spread at 39** — it is genuinely the most even, because it is
    the only strategy **looking at meaning and length at the same time**.
2.  **Hierarchical has the largest at 452, and that is not a defect** — it is faithful to the
    document's structure, where headings are short and bodies are long. A large spread is
    **what the structure looks like**, not a bad split.
3.  **The two test texts have to be comparable**: they share the same paragraphs and differ only by
    67 characters of headings. Otherwise the columns could not be read across.

> **⚠️ A silent failure worth remembering**: strategy 3 (LLM splitting) **never actually ran** for a
> while — after an authentication failure the code fell back silently to semantic splitting, so
> rows 2 and 3 of the comparison held identical numbers and looked perfectly reasonable. Only after
> switching to a working provider did it produce results of its own.
> **A silent fallback is the hardest kind of bug: it raises nothing and simply shows you a false
> conclusion.**

### 5.3 Chunking and vectorisation are two separate steps

**Chunk first (by rule or by model), then embed the chunks.** Do not conflate them — the embedding
model does not split anything, and a splitting strategy does not change because the embedding model
did.

**Granularity**: 300–1000 characters is usual (around 800 by default), with 300–500 giving finer
grain.
**A practical trick**: build **several indexes** with different chunk sizes for different kinds of
question.

### 5.4 Small-to-Big: index the small, answer from the big

**The idea**: index **small content** — summaries, key sentences — and **link** it to the full body;
once the small content is hit, follow the link to pull the large content in as context.

*   **Small (the index)**: a summary or key sentence per document.
*   **Large (the link)**: the full text, associated by **document id, URL or pointer**.
*   **Three steps**: ① match the query against summaries and key sentences ② follow the link to the
    full document ③ feed the large content in as context.

**The gain**: fast location (small content has a high signal-to-noise ratio) plus complete answers
(large content carries the context), which suits **long or numerous documents** and
**lowers the cost of processing long documents**.

> This is the same direction as merging knowledge points in 7.2: **chunks that are too small answer
> only part of a question even when they are retrieved.**

### 5.5 Index expansion: discrete, continuous and hybrid

A vector index alone misses cases that need **literal matching**. Index expansion means
**keeping more than one index**.

**1) Discrete expansion** — build indexes from keyword extraction and entity recognition:

*   **Keyword extraction** (TF-IDF, TextRank): a passage on training optimisation →
    `["deep learning","model training","optimisation","AdamW","mixed precision","distributed training"]`
*   **Entity recognition (NER)**: "the 2023 Nobel Prize in Physics went to three scientists …
    quantum entanglement" → `["2023","Nobel Prize in Physics","quantum entanglement"]`

> **Why entities in particular need a discrete index**: "2023" sits almost on top of "2022" and
> "2024" in vector space and semantic retrieval cannot separate them, whereas keyword matching is
> exact and a single character matters. **This is precisely the blind spot of vector retrieval.**

**BM25 — the workhorse of discrete indexing**: an improvement on TF-IDF that raises ranking quality
through finer **term-frequency saturation** and **document-length normalisation**:

$$BM25(Q,D)=\sum_{i=1}^{n}DF(q_i)\cdot TF_{BM25}(q_i,D),\quad
TF_{BM25}=\frac{(k_1+1)\cdot tf}{k_1\cdot(1-b+b\cdot\frac{dl}{avgdl})+tf},\quad
DF=\log\left(\frac{N-n+0.5}{n+0.5}+1\right)$$

*   **Term-frequency saturation** — the `+tf` in the denominator makes the score level off as tf
    grows. A term appearing 100 times is not ten times more important than one appearing 10 times;
    TF-IDF scales linearly, BM25 holds it down.
*   **Document-length normalisation** — `b·dl/avgdl` (`dl` this document's length, `avgdl` the
    average). Long documents match any term more easily, and this term removes that advantage.

**2) Continuous expansion** — recall through several vector models at once. Models have different
biases, and merging results covers what any single one misses.

**3) Hybrid recall** — combine a discrete index such as BM25 with the vector index (an ensemble
retriever). For precise retrieval, **match keywords first, then compute vector similarity**.

> **Choosing: do not adopt all of them. Pick one, or combine a few, according to actual need.**

---

## 6. Advanced Recall: Making Retrieval Land on Target

### 6.1 The premise: why this section exists

RAG lives or dies on "retrieve, then generate", and **if retrieval goes astray, generation quality
necessarily follows it down**. Users ask in conversational, context-dependent, vague, emotionally
loaded language; the store holds declarative, neutral passages.
**When the two are phrased differently, no amount of vector precision helps.**

> **Every technique in this section is optional; none of them has to be adopted.** Retrieval
> augmentation is systems work, and each addition costs a model call, some latency, or money.

### 6.2 An overview of recall strategies

**The crudest lever is a larger k**: `similarity_search(query, k=10)`.

**How large**: 4–5 for ordinary cases; up to 10 or even 30 when casting wider; **5–10 is the usual
range**.
⚠️ **A large k (say 30) requires reranking**, or the noise entering the context drowns the few
passages that mattered.

**Five families of recall strategy** (the agenda for the rest of this section):

① **Better retrieval algorithms** — knowledge graphs (→ 6.9)
② **Reranking** — rerank models (→ 6.3); hybrid retrieval (→ 5.5)
③ **Query expansion** — multi-query recall (→ 6.4)
④ **Index expansion** — discrete / continuous / hybrid (→ 5.5)
⑤ **Small-to-Big** (→ 5.4)

> **"Coarse filter, fine sort" is a general engineering pattern**:
> ```
> 10M chunks → recall (cheap; keywords or vectors) 1000 → rerank (expensive; a large model) top 20
> ```
> Recommender systems have the same shape. **Cut the volume down with something cheap, then polish
> what is left with something expensive.**

### 6.3 Reranking

**What it is**: reordering an initial retrieval result to raise the relevance of what finally goes
to the model.

**The key distinction: bi-encoder vs cross-encoder**

*   The embedding model used for retrieval is a **bi-encoder** — the query and the document are
    encoded **independently** into vectors, which are then compared. The advantage is that document
    vectors can be **computed in advance**, so a search encodes only the query.
*   A rerank model is a **cross-encoder** — the query and the document go into the model
    **together**, and it **emits a relevance score directly**. Far more accurate, but
    **every pair costs a forward pass** and nothing can be precomputed.
*   ⇒ That is why the two are stages of one pipeline: the bi-encoder is cheap enough to screen, and
    the cross-encoder is accurate enough to be worth running on the few survivors.

**Open-source option**: a local cross-encoder is enough. The script uses
`cross-encoder/ms-marco-MiniLM-L-6-v2` (tens of megabytes, runs on CPU; larger rerank models are
typically 2–3 GB). The model is a supervised sequence classifier trained on human-labelled
query-document relevance.

**⚠️ Reading the scores is where this is most often misunderstood**: the output is
**unnormalised logits with no fixed bounds**. Measured on one question:

```
 -6.11  The date can be changed once, free of charge, up to 48 hours …   ← the correct answer
-11.36  The Eiffel Tower is a wrought-iron lattice tower in Paris.       ← entirely unrelated
```

**Note that the correct answer also scores negative.** Therefore:

*   **The sign is not a relevance threshold** — no rule of the form "above zero means relevant".
*   **Only the relative ordering within one query on one corpus carries meaning.**
*   **Change the model or the chunk size and the scores stop being comparable.** Do not treat them
    as absolute values that travel between settings.

**Commercial API option**: strong multilingual support, **normalised 0–1 scores that are easier to
interpret**, ready-made framework integrations, and a particularly good fit for
**reordering the output of hybrid retrieval (BM25 + vectors)**, measured by hit rate and MRR.

| Property | Open-source rerank | Commercial rerank API |
| :--- | :--- | :--- |
| Deployment | Local | Cloud |
| Cost | Free, but needs memory | Metered, no operations |
| Scores | Unnormalised logits | Normalised 0–1 |
| Fits | Sensitive data, vertical domains | Fast integration, many languages |

**Practice eight: two-stage retrieval** (see `09_rerank_and_multiquery.py`):
the knowledge base holds 26 paragraph blocks (108 sentence units) from 4 documents.
Stage one uses BM25 to keep 8 of the 26 paragraphs; stage two uses the cross-encoder to rank
**the sentences inside** those 8. Step one already **binds headings to their body text**, avoiding
the heading-block contamination described in 4.6.

> **The unit fed to the reranker is a setting, not a detail.** Same question, same correct answer —
> only the amount of surrounding text changes, and the score moves a long way:
> ```
>  -8.60  heading + whole paragraph   (506 chars)
>  -6.43  whole paragraph             (469 chars)
>  -5.82  the answering sentence only (113 chars)
> ```
> **The answer never moved; only the unrelated text around it did.** Feed a whole paragraph in and
> the one relevant clause is diluted by everything beside it — which is exactly how a correct
> passage ends up ranked below a wrong one.

> **⚠️ Ranking first is not the same as being confident**: on the second question the correct
> sentence wins by **under a tenth of a point**. That is a coin-flip margin, not a verdict.
> **"The reranker put the right answer first" and "the reranker is sure about it" are two
> different claims.**

### 6.4 Multi-query recall (query expansion)

**How it works**: have the model rewrite one query into **several semantically similar ones**,
retrieve with each, then **deduplicate and merge**.

This amounts to **measuring one thing with several rulers**, lowering the chance that a single
phrasing misses.

**Measured** (also in `09_rerank_and_multiquery.py`): after expanding each question into four
phrasings, what stage one can see changes as follows:

| Question | Paragraphs recalled | Newly reachable | Did the final answer change? |
| :--- | :---: | :---: | :--- |
| Can I move my visit to a different day after buying? | 8 → 14 | +6 | No (it was already correct) |
| **My father is 68 - does he pay less?** | 8 → 22 | **+14** | **Yes: from an irrelevant answer to the correct clause** |
| How do I skip the queue on the busiest rides? | 8 → 16 | +8 | No |

**Only the second question was rescued, and it had failed in stage one**: the asker says `father`,
`68` and `pay less`, while the policy says `aged 65 or over` and `senior rate` —
**not one word in common**, so BM25 never handed the right paragraph to the reranker.
**Expansion exists for exactly this case.**

> **The point**: query expansion **fixes recall, not ranking**. The other two questions had already
> recalled the right paragraph, and expansion only added candidates while the best answer did not
> move a fraction — yet every extra phrasing is another model call.
> **Whether it pays depends on whether your users speak a different vocabulary from your documents.**

> **Where it does not apply**: this kind of wrapper suits conventional vector RAG; graph retrieval
> needs more customisation and does not fit inside it.

### 6.5 Query rewriting: five types

**Why it is needed**: something has to act as a **translator**, turning a user's spoken-style query
into a written, precise retrieval phrase.

All five share **one prompt skeleton**, varying only the `instruction`:

```python
prompt = f"""### Instruction ###\n{instruction}\n### Conversation history ###\n{conversation_history}
### Current question ###\n{current_query}\n### Rewritten question ###\n"""
```

The five as measured by the script:

| Type | Trigger | Original → rewritten |
| :--- | :--- | :--- |
| **Context-dependent** | Depends on three prior turns | `Are there any other rides?`<br>→ `Are there any other rides in the Wildwood area at Riverbend Park besides the ranger station, training camp, and ice cream parlour?` |
| **Comparative** | Neither side was ever named | `Which one takes longer and is more fun?`<br>→ `Which takes longer and is more fun: Wildwood or Skyline at Riverbend Park?` |
| **Ambiguous reference** | `both of them` points backwards | `When do both of them start?`<br>→ `When do the fireworks shows at Riverbend Park and Harbour Park start?` |
| **Multi-intent** | Three questions in one turn | `How much is a ticket? Do I need to book ahead? What does parking cost?`<br>→ split into three independent questions (**this returns a list, not a string**) |
| **Rhetorical** | Emotion carries the sentence | `Don't tell me I have to book a month ahead as well?`<br>→ `What is the typical advance booking window for tickets?` |

**Three points worth stopping on**:

*   **Why the context-dependent type must be rewritten**: `Are there any other rides?` sent to
    vector retrieval **matches any chunk containing `rides`**; naming the area and the three rides
    already mentioned **narrows the target immediately**.
*   **Why the rhetorical type must be rewritten**: vectorising the original spends
    **most of the sentence's length on the complaint**, leaving only a few words of retrievable
    fact. Rewriting **strips the emotion and keeps the request**.
*   **The multi-intent type is not like the other four**: its output is
    **a list, not a single query**, and everything downstream has to retrieve each part and merge
    the answers. **It changes the shape of the pipeline**, and must be handled separately.

> **Not every query needs rewriting.** Rewriting costs a model call, and short literal questions
> retrieve fine without it. This is a cost-benefit judgement, not something to switch on
> everywhere.

### 6.6 Intent detection: classification, rewriting and confidence in one prompt

**Do five types mean five prompts and five calls?** No — one multi-task prompt is enough.

The instruction spells out the definitions and trigger words of all five and
**hard-codes a priority rule** (for example, multi-intent outranks ambiguous reference when both
match), with a fixed JSON output:

```json
{"query_type": "...", "rewritten_query": "...", "confidence": "0-1"}
```

**Compared with a rule engine**: one call handles the whole judgement, avoiding the accumulated
latency of several; an "other" catch-all keeps it extensible.

**Measured, five samples**:

| # | Query | Type | Conf. | Rewritten |
| :-: | :--- | :--- | :-: | :--- |
| 1 | `Are there any other rides?` | context_dependent | 1.00 | `Are there any other rides in the Wildwood area at Riverbend Park?` |
| 2 | `Which Riverbend Park area is more fun?` | comparative | 1.00 | `Which Riverbend Park area, Wildwood or Skyline, is more fun?` |
| 3 | `Are they all suitable for small children?` | ambiguous_pronoun | 0.95 | `Are the fireworks shows at Riverbend Park and Harbour Park suitable for small children?` |
| 4 | `Which restaurants are there? What do they cost?` | multi_intent | 1.00 | `Which restaurants are there? What do they cost?` ← **returned unchanged** |
| 5 | `Don't tell me this is another two-hour queue?` | rhetorical | 0.90 | `Is this another two-hour queue?` |

> **⚠️ Row 4 is the failure, and the cause is not a misclassification**: all five types are
> identified correctly. The problem is that this schema declares `rewritten_query` as
> **a single string** — the multi-intent type owes a **list**, the structure has nowhere to put
> one, and the two questions come back flattened into a single line.
>
> **The classification is right and the rewrite is still wrong**, because
> **one output shape cannot serve five query types.** That is the cost of folding classification
> and rewriting into one call, and the reason type 4 in 6.5 is handled on its own.

> **Read the confidence column carefully**: it is the model's impression of its own answer. Row 4
> is wrong and scores a full 1.00; row 5 rewrites cleanly and scores only 0.90.
> **It ranks; it does not measure.**

**What confidence is and is not**: **it is not a cosine between vectors, it is a score the model
produces from its own reading**, not a computation. Two ways to improve on it:
① **cross-check against vector similarity** — patch an impression with an actual measurement;
② route low-confidence results to **human review**.

**Three ways to keep improving**: feed bad cases (like row 4) back into the prompt's example bank;
build an **A/B test** for rewrite quality; train a dedicated classifier for a specialist domain.
**Prefer a mature open-source path for standard flows, and only customise where precision pays.**

> **What the rewrite actually is**: `query' → used for the embedding that retrieves the top-k chunks`
> — the product of rewriting **is not an answer shown to the user; it is an intermediate fed to the
> embedding model.**

**Practice nine: query rewriting** (see `08_query_rewriting.py`) covers rewriting
context-dependent questions, rewriting ambiguous references, classifying and rewriting in a single
call, and **comparing retrieval before and after the rewrite**.

### 6.7 Two-way rewriting: Query2Doc and Doc2Query

**The underlying problem**: **short text vectorises poorly**. A five-word question and a
two-hundred-word document are inherently mismatched in vector space, and two-way rewriting
**brings the two sides to a comparable length and shape**. There are two directions, usable alone
or together.

**Query2Doc (lengthen the query side)**: expand a short query into a hypothetical document.
Example: "how do I speed up deep-learning training?" → five points (optimisers, mixed precision,
distributed training, data preprocessing and augmentation, learning-rate scheduling).

> **⚠️ The point most easily misread**:
> **Q: Query2Doc already looks like the answer. Is the rest of the pipeline still needed?**
> **A: Yes.** Augmentation knowledge must come from **the private knowledge base**; what the model
> expanded only **points retrieval in a direction** so the right chunks can be matched.
> **The knowledge that finally reaches the model has to come from the store.**
> It looks like an answer, but its identity is "an expanded query" — **an intermediate used for
> retrieval, not a final answer.**

**Doc2Query (turn documents into questions)**: generate the queries a document fragment might
answer. Example: a passage on training optimisation → five questions (how do I choose an optimiser?
what does mixed precision buy? …).
**Benefits**: ① the document becomes reachable more often ② a deeper semantic link between question
and document.
**Typical use**: pre-annotating a knowledge base (its concrete form appears in 7.1).

**The workflow**: define the instruction template → supply examples → run the rewrite → retrieve
with the result.
**Three points on instruction design**: state the role clearly, state the purpose (retrieval), and
supply a standard example format.
**On cost**: rewriting adds work, so gate it on the value of the query; it pays best where
freshness and accuracy matter most.

### 6.8 Web search: the half a private store cannot answer

A private knowledge base is a **static snapshot**, and anything that changes it will answer badly.

**Eight scenarios that need live data**:

| Type | Trigger words | Example | Reason |
| :--- | :--- | :--- | :--- |
| Freshness | latest, today, now, current | Is it open today? | Needs information as of now |
| Prices | how much, price, fee, fare | What is a ticket next Saturday? | Prices move |
| Opening status | opening/closing hours, is it open | Is it open right now? | Status can change without notice |
| Events | event, show, performance, festival | Any special events on? | Time-bound and dynamic |
| Weather | weather, rain, temperature | What is tomorrow like? | Must be live |
| Transport | how do I get there, metro, bus | How do I get there from the airport? | Changes with works and events |
| Booking | booking, reservation, tickets | How far ahead must I book? | Policies change |
| Live status | queue, crowded, footfall | How busy is it now? | Only meaningful live |

**The root cause: the model has no sense of the current time**, and the date has to be injected
from outside.

**Three core functions** (each writes the table above into the prompt and returns JSON):

| Function | Input | Output |
| :--- | :--- | :--- |
| Decide whether live data is needed | query + history | `need_web_search` / `search_reason` / `confidence` |
| Rewrite for a search engine | query + search type | `rewritten_query` / `search_keywords` / `search_intent` / `suggested_sources` |
| Build a search plan | as above, with the current date injected | `primary_keywords` / `extended_keywords` / `search_platforms` / `search_tips` / `verification_methods` |

**Six rewriting techniques**: add a specific place, add a time range, use keyword combinations, add
search intent, remove conversational phrasing, add related terms.

**Measured (two questions through all three functions)**:

```
Q: Is Riverbend Park open today, and how busy is it right now?
  ① decide   search: True   confidence: 1.00 (gate at 0.7)
             reason: opening status and live crowd levels are both time-sensitive
  ② rewrite  query   : Riverbend Park open today current crowd level
             keywords: ['Riverbend Park','open today','hours','current crowd','busy now','live status']
             sources : ['official park website','Google Maps','park social media','local news']
  ③ plan     primary : ['Riverbend Park','open today','busy right now']
             extended: ['Riverbend Park hours today','Riverbend Park live crowd status', …]
             window  : today, current time

Q: How much is a Riverbend Park ticket next Saturday, and how far ahead must I book?
  ① decide   search: True   confidence: 0.95
  ② rewrite  query   : Riverbend Park ticket price next Saturday booking advance requirement
  ③ plan     window  : current week
```

> **The two functions produce visibly different shapes, and that is the point**: ② produces
> **keywords and sites for a crawler**, while the rewriting in 6.5 produces
> **a complete sentence for vector retrieval**. The same word "rewrite", two entirely different
> deliverables — **sharing one prompt between them is guaranteed to go wrong.**

> **The time-window row deserves a second look**: the two questions return `today, current time`
> and `current week`, and that judgement **can only come from the current date injected into the
> prompt**. The model has no notion of what today is — without the injection, that row is a guess.

**Common search API parameters** (Tavily as the example): `query` (required), `search_depth`
(basic/advanced), `time_range` (day/week/month/year/all), `max_results`, `topic` (general/news),
`domains` / `exclude_domains` (allow and deny lists), plus three booleans off by default
(`include_images` / `include_answer` / `include_raw_html`).
**The convention: JSON output, required and optional fields separated.** Search tools of this kind
can also be exposed through the MCP protocol.

**How it fires**: a **confidence threshold (0.7 and above)** decides whether to search.
⚠️ Note that this is the **only place in this topic where confidence genuinely acts as a
threshold** — in 6.6 it is a reference value, here it is a switch.

### 6.9 GraphRAG: writing the relationships into the data structure

**What it is**: a **structured, hierarchical** approach to retrieval augmentation rather than
**semantic search over plain text fragments**. It extracts a **knowledge graph** from the source
text, builds a **community hierarchy**, **summarises** those communities, and uses that structure at
query time. The whole workflow is a **DAG**.

**Where baseline RAG falls short**: RAG driven only by vector similarity is weak in two situations:

① **Connecting the dots** — when an answer requires **traversing several fragments through shared
attributes**;
② **Understanding a large body as a whole** — when the question is about semantics **across
documents** or across one large document.

**The same question, three ways.** Query: *how did nineteenth-century art movements influence the
development of twentieth-century modern art?*

| Approach | What was retrieved | Answer | |
| :--- | :--- | :--- | :-: |
| LLM alone | Nothing | "…by encouraging experimentation with colour, form and subject…" | ✗ Vague, no people, no causality |
| Baseline RAG | **Four mutually independent fragments** (Monet introduced new techniques / Impressionism influenced later movements / Picasso founded Cubism / Cubism appeared in the early twentieth century) | "…Picasso founded **Cubist relativity** in the early twentieth century." | ✗ The fragments do not join, so it invents a concept |
| GraphRAG | **Five triples**: (Monet)-[introduced]→(new techniques); (new techniques)-[transformed]→(depiction of light and colour); (Impressionist technique)-[influenced]→(later movements); (Picasso)-[founded]→(Cubism); (Cubism)-[appeared in]→(early twentieth century) | "The new techniques Monet introduced transformed the depiction of light and colour. His Impressionist technique influenced later movements, including Picasso's Cubism in the early twentieth century…" | ✓ The causal chain is complete |

**The whole difference sits in the "what was retrieved" column**: baseline RAG receives four
isolated sentences and the model has to **guess the relationships — and invents when it guesses
wrong**; GraphRAG receives five labelled edges where **the causality is already in the data
structure**, and the model only has to turn it into prose.

**Knowledge base vs knowledge graph**: **a knowledge base is the source documents**;
**a knowledge graph is a `Graph<node, edge>` built on top of them** — nodes are entities, edges are
relationships — essentially **an organised set of notes over the original knowledge**. It connects
fragments **through entities and semantic relations** instead of leaving chunks scattered.

**Four basic steps**:

① Split the corpus into **TextUnits**, the unit of analysis and the basis for **fine-grained
citation**;
② Use an LLM to extract every **entity, relationship and claim**;
③ Cluster the graph hierarchically with the **Leiden algorithm** (each circle is an entity, size
shows degree, colour shows community level);
④ Generate summaries for each community level, **bottom-up**.

**Six indexing stages**. The entity types stored are `Document`, `TextUnit`, `Entity`,
`Relationship`, `Covariate`, `Community Report` and `Node`.

| Stage | What happens |
| :--- | :--- |
| 1 Compose TextUnits | Documents → TextUnits, with **chunk size and grouping configurable** |
| 2 Graph extraction | Entities and relationships via `entity_extract`, claims via `claim_extract` |
| 3 Graph augmentation | **Hierarchical Leiden** for community detection, **Node2Vec** for graph embeddings |
| 4 Community summarisation | LLM-generated community reports at every level of granularity |
| 5 Document processing | Build the "documents" table for the knowledge model |
| 6 Network visualisation | **UMAP** for 2D projection |

Stage two has four sub-steps: **extract** (merging entities that share **name and type**, and
relationships that share **source and target**) → **describe** (ask the LLM for a short description
of each entity and relationship) → **entity resolution (off by default)** →
**claim extraction** (positive factual statements, emitted as `Covariates`).

> **"Entity resolution is off by default" is an easy trap**: the same real thing under two names
> stays **two entities** — the merge in the previous step only deduplicates on identical name plus
> identical type, and **different names for one thing are out of scope.**

**Two query modes**:

| | **Global query** | **Local query** |
| :--- | :--- | :--- |
| Suits | **Corpus-wide questions** ("what is this material about") | **Questions about a named thing** ("what are the properties of X") |
| Mechanism | Reads **community summaries** through **map-reduce**: map turns report chunks into rated intermediate responses, reduce aggregates the most important | Identifies **semantically related entities** in the graph and runs five parallel branches (TextUnit / community report / entity / relationship / covariate), each ranked and filtered into a fixed-size context window |
| Character | **Resource-intensive**; broader questions need a broader view | **Combines the graph's structure with the original unstructured text** |
| Citations | `[Data: Reports (181, 123, +more)]` | `[Data: Entities (291); Relationships (723, +more)]` |

**How they differ in practice**: asked "who is connected to this person", global **groups by role**
(opponents / allies / others) while local **lists names flat**; asked "who did this person defeat",
local goes down to a specific battle while global rises to influence and reputation.

> **⚠️ Both modes produce hard errors.** In testing, local listed people who were not opponents at
> all, and global asserted things the source does not support.
> **GraphRAG does not remove hallucination; it improves how well knowledge is matched** — its gain
> is "using existing knowledge more broadly and completely", **not "being correct".**

**Tuning: getting more entities and relationships to match**
Raise `TOP_K_ENTITIES` (related entities retrieved from the entity-description embedding store,
default 10) and `TOP_K_RELATIONSHIPS` (relationships pulled into the context window, default 10),
**and widen `MAX_TOKENS` at the same time** —
**raising top_k without widening the window simply gets the extra content truncated.**
Other common settings: `TEXT_UNIT_PROP` 0.5, `COMMUNITY_PROP` 0.1,
`CONVERSATION_HISTORY_MAX_TURNS` 5, `SEARCH_MAX_TOKENS` 12000 (5000 suggested for 8k-context
models), `MAP_MAX_TOKENS` 500, `CONCURRENCY` 32, `MAX_RETRIES` 20.

**Cost and ROI, which is what decides whether to adopt it**:

| Metric | Order of magnitude |
| :--- | :--- |
| TextUnit size | About **1200 tokens** per unit |
| Entity vector dimensionality | **768** |
| Indexing time / cost | Around **30 minutes / roughly \$100** for a million-token corpus (tens of thousands of entities, on a low-cost model) |
| Cost per query | **\$1–2** including follow-up sub-questions |

> **Where it earns its keep**: **high-value text analysis** — prospectuses, annual reports, due
> diligence material. The typical case is relationship-network analysis, which is especially
> valuable in finance because that domain deals in entities and relationships already.
> **The ROI threshold: consider it when a document is worth more than \$500.**
> **Alternatives**: other graph-augmented approaches; a mature conventional RAG platform at much
> lower cost; or a **full-context agent approach** — chunk the data and **answer from all of it**,
> sidestepping the incompleteness of answering from only the top-k passages.

**Practice ten: asking one multi-hop question twice** (see `13_graphrag_vs_vector.py`):

1.  Read the corpus and display the **chain of facts** the question depends on.
2.  Retrieve the top 3 with a vector index and answer (the baseline).
3.  Retrieve through the knowledge graph and answer.
4.  Compare the two, and account for the graph route's extra cost.

The corpus is a purpose-written English archive, `northgate_archive.txt` (1220 words), and the
question is `How did Mira Delaunay's way of working end up affecting Port Halbrook?` —
**the answer sits in no single passage**, and requires joining five facts that are never stated
together: `Delaunay trained Ek → Ek founded Northgate → Northgate produced Latch Encoding →
Latch Encoding made the Orrery system possible → Orrery was deployed at Port Halbrook`.

**Three measured results**:

| Route | What was retrieved | Result |
| :--- | :--- | :--- |
| **Vector baseline** | The nearest 3 of 7 chunks (cos 0.686 / 0.655 / 0.652) | **All five links present**, answer correct |
| **Global (community summaries)** | Community reports, cited as `Reports` | Covers all 7 relevant entities, plus 1 outside the chain |
| **Local (from the entities)** | Entities and relationships, cited as `Entities` / `Relationships` | Covers all 7 relevant entities, plus 3 outside the chain |

Index size: **28 entities, 36 relationships, 4 communities, 4 community reports**, extracted from 5
text units.

> **⚠️ The most honest result here: the baseline did not lose.** It retrieved all five links and its
> answer holds up. The reason is **the corpus is small** — 1220 words split seven ways makes the
> nearest three chunks nearly half the archive, and a chain packed that tightly survives being
> retrieved by resemblance.
> **The two approaches separate on a corpus whose five links sit hundreds of pages apart, and this
> one is too small to show it.**
> ⇒ The graph is doing real work here, **but this run does not prove it was needed**. Saying that
> plainly is worth more than a conclusion that the graph won.

> **Unresolved entities, demonstrated in the output**: `ASHFIELD` and `ASHFIELD POLYTECHNIC`,
> `NORTHGATE` and `NORTHGATE LAB` are treated as **four entities**. Extraction merges only entities
> sharing a name and a type; **resolving different names for one real thing is a separate step that
> is off by default** — this is what that costs.

> **Cost is where this comparison lands**: the baseline was **two API calls in total**. The graph
> route needed a full indexing pass before a single question could be asked — about **a minute and
> a half** on this corpus — and **that pass scales with the corpus, not with the number of
> questions**. Over a book it runs for half an hour and bills accordingly.
> **It earns that back only where the connections matter more than the passages do.**

---

## 7. Operating a Knowledge Base and Keeping It Healthy

Everything so far has been about **using** a knowledge base. This section is about **maintaining**
one. A knowledge base is not a static asset that is finished once built; it has a full life cycle:
**question generation → knowledge extraction → health checks → version management**, closing the
loop.

### 7.1 Scenario one: generated questions and retrieval optimisation

**The insight**: **users ask questions, the store holds statements, and the two do not match.**
So generate a set of questions for every chunk in advance and **match question against question**.
(This is 6.7's Doc2Query made concrete.)

**Three pieces**: ① generate varied questions automatically ② build **a dual retrieval index**
(source prose and generated questions, both under BM25) ③ evaluate retrieval.

**Basic version: five questions per chunk**, with the fields `question` / `question_type` /
`difficulty`, spanning direct, indirect, comparative and conditional types, graded easy / medium /
hard.

**Wider version: eight questions, with variety broken into four dimensions** — question type
(adding **hypothetical** and **inferential**), phrasing, difficulty and **perspective**. The output
grows from three fields to six, and the three additions are the important ones: `perspective`,
**`is_answerable`** (can the given knowledge answer this question at all) and **`answer`** (the
answer derived from that knowledge).

> **`is_answerable` plus `answer` are the quality gate, not decoration**: having the model
> **generate the answer alongside the question** verifies in reverse whether this piece of
> knowledge can support the question. **A generated question the source cannot answer is a
> hallucinated question and has to be dropped** — otherwise it poisons the question index and
> retrieval happily matches a chunk that then fails to answer.

**Dual-index evaluation**: 6 knowledge chunks → a prose index of 6 entries versus a question index
of **30** (five questions per chunk), scored on 3 test queries each with a known correct chunk.

```
prose retrieval accuracy: 33.3% (1/3)      question retrieval accuracy: 66.7% (2/3)
```

| # | Query | Prose score | Question score | Prose | Question |
| :-: | :--- | :-: | :-: | :-: | :-: |
| 1 | `Am I allowed to take a picnic in?` | **0.000** | 3.734 | ✗ | ✓ |
| 2 | `What time should I show up to avoid the crowds?` | **0.000** | 9.106 | ✗ | ✓ |
| 3 | `How much does it cost to park a car?` | 1.262 | 3.051 | ✓ | **✗** |

**Four things to read out of this table**:

1.  **The two 0.000 scores are not "a weak match", they are "no match at all"** — those queries
    share **not one content word** with any chunk, so BM25 scores every chunk zero and the winner
    is whichever document argmax reaches first. Rows 1 and 2 both landed on `kb_001` (the general
    overview) for exactly that reason.
2.  **The mechanism behind the turnaround is question-to-chunk backtracking**: the question index
    does not retrieve prose at all — it retrieves **a generated question** and then follows it back
    to the chunk that produced it. A hit depends on **how close the asker came to one of the
    generated phrasings**, not on the wording of the source.
3.  **⚠️ Row 3 went from right to wrong — the most valuable line here.** Generated questions add
    **a second vocabulary** to the store, and that extra surface can pull a query towards the
    correct chunk **just as easily as away from it**: this parking question was captured by
    `kb_002`'s `How much does a weekday adult ticket cost?`.
4.  **So read the net, not the wins.** Two gained, one lost, net +1 — which is what 33.3% → 66.7%
    means. **"This technique improves accuracy" and "this technique never misfires" are different
    claims.**

**Conclusion**: **the wider the gap between how users ask and how documents state things, the more
question retrieval pays**; but **it is not free** — the benefit depends on the quality and variety
of the generated questions, and the extra vocabulary works in both directions.

**Where the questions live**: **in the same chunk as the source text** — `chunk = {prose, questions[]}`.
No extra storage system is required.

See `10_kb_question_generation.py`.

### 7.2 Scenario two: distilling knowledge out of conversations

**A product in production generates conversations every day. How is anything valuable extracted
from them?**

**Three core actions**: extract from one conversation → extract in batch →
**merge similar points with an LLM**.

**Step 1 · Extract** — the instruction casts the model as a knowledge extraction specialist,
pulling five kinds of content: **facts** (places, times, prices, rules) / **user needs and
preferences** / **common questions and answers** / **procedures** / **cautions and reminders**.

Output JSON: `extracted_knowledge[]` (`knowledge_type` / `content` / `confidence` / `source` /
`keywords` / `category`) plus `conversation_summary` and `user_intent`.

**Step 2 · Filter** — the `need` and `question` types have to go. Measured, 6 of the 15 points
extracted from one conversation fell into those two types, for example:

```
drop [need]     The visitor wanted to know the ticket cost for Riverbend Park.
drop [question] What does a ticket cost?
```

> **Why filtering is mandatory**: these record **what somebody wanted to know, not what is true**.
> Left in the index, the next question about ticket prices retrieves "the visitor wanted to know
> the ticket price", and the model then answers "what does a ticket cost" with that —
> **pure noise, and it takes a slot something useful would have filled.**

**Step 3 · Merge** — **group by knowledge type first, then make one model call per group**
(groups of one are kept as they are). The merge prompt asks for five things: preserve every
important detail, remove duplication and consolidate similar statements, improve accuracy and
completeness, keep the logic clear, and **take the highest confidence in the group**. Two extra
fields come out: `sources` and `frequency` (how many points went in).

**Measured: 27 → 21 → 3**

```
27 points extracted from three conversations → 6 dropped as need / question → 21 left → 3 after merging
```

| # | Type | Merged from | Conf. | Content |
| :-: | :--- | :-: | :-: | :--- |
| 1 | `fact` | **14** | 0.95 | Ticket prices (399 weekday / 499 weekend and public holiday) + child tickets + opening hours + parking fee + what may be brought in |
| 2 | `process` | 3 | 0.95 | Metro route from the airport with the interchange + taxi option + where to buy |
| 3 | `caution` | 4 | 0.95 | Book ahead, especially at weekends and on public holidays |

> **What merging buys**: fragments of a dozen words become a single description of a hundred-plus
> words. This lines up directly with Small-to-Big in 5.4 — **chunks that are too small answer only
> part of a question even when retrieved; one entry holding the whole topic answers it in a single
> retrieval.**

**Three intake routes for a knowledge base**: `① conversations ② thumbs-up → a "good answers" book
③ thumbs-down → a "corrections" book`. The script implements the first; the other two turn human
feedback into an intake route of its own.

### 7.3 Scenario three: health checks

**How do you give a whole knowledge base a checkup and find what is missing, stale or contradictory?**

Three checks, each its own prompt, **with different inputs and different scoring**:

| Check | Extra input | Criteria | Output |
| :--- | :--- | :--- | :--- |
| **Coverage** (missing) | **A test query set** | ① can each query be answered ② is the knowledge complete and correct ③ are the main needs covered ④ are there blank areas | `missing_knowledge[]` + `coverage_score` |
| **Freshness** (stale) | **The current date** | Whether times, prices, policies, events, contact details and technical information have aged | `outdated_knowledge[]` + `freshness_score` |
| **Consistency** (conflicts) | None (an internal question) | Different statements on one topic, price discrepancies, inconsistent times, conflicting rules, divergent procedures | `conflicting_knowledge[]` + `consistency_score` |

**Three design notes**:

*   Coverage **requires a test query set** — **what is missing is defined relative to a need**, and
    a knowledge base cannot see its own gaps.
*   Freshness **lives or dies on injecting the current date** — the model has no sense of time (as
    in 6.8). Only with today's date supplied can it decide that last quarter's policy has expired.
*   Consistency works by **casting the model as a conflict detector**, spotting version differences
    within a topic.

**The test method: plant defects, then see whether they are found.** An auditor that finds nothing
has demonstrated nothing, so the audited base carries three planted defects — no pet policy
(coverage), an event that ended in late 2023 (freshness), and `kb_002` and `kb_005` quoting a
parking fee of 100 and 150 respectively (consistency).

**Measured report**:

```
4. Coverage      0.67   1 gap      —— "Can I bring my dog?" no pet policy            [medium]
5. Freshness     0.20   5 stale    —— kb_004 event ended 2024-01-05 [high], kb_002 parking fee [high] …
6. Consistency   0.80   1 conflict —— kb_002 / kb_005 parking fee discrepancy        [high]
   Overall       0.56
   Planted defects detected: 3/3 (coverage, freshness, consistency)
```

> **✅ All three planted defects were found**, and the freshness check correctly judged that a
> December 2023 event had ended — **entirely because the current date was injected into the
> prompt**. The model does not know what today is.

> **⚠️ But the freshness score of 0.20 cannot be used as it stands**: it flagged **5 of 6** entries.
> Unpacking them, two kinds of thing are in there that should not count:
> ① entries that **merely could change one day** (opening hours "may have been adjusted");
> ② **the parking-fee conflict, counted a second time here** when it belongs to the consistency
> check.
> The three audits take different inputs and **their conclusions still overlap**, so a low score
> may not be measuring what it claims to.
>
> ⇒ **Trust the findings list, not the number beside it.** Findings are checkable — each can be
> verified true or false — while the score is an impression. **This is the same defect as the
> confidence value in 6.6**: precise-looking, and neither comparable nor reproducible.
> **Fine as a sort key, not as a KPI.**

**The principle for conflicts**: **the model finds them, a person decides.** Judgements that cannot
be enumerated need the model's generality, so have it emit conflicts as JSON and
**leave the final call to a human**.

See `11_kb_curation.py` (distillation and auditing in one script).

### 7.4 Scenario four: version management and performance comparison

**How do you version a knowledge base so you can run regression tests, sign off a release, and
compare versions?** This is the only one of the four scenarios where **embeddings, not an LLM, do
the work**.

**Core functions**: version creation with description and statistics, **a hash as the version
identity**, statistics (entry count, content length, category distribution), and version comparison.

**Seven modules**:

| Module | Implementation |
| :--- | :--- |
| 1 Vectorisation | Call the embedding endpoint at a fixed dimensionality |
| 2 Index building | Walk the base producing vectors and metadata → `IndexFlatL2` → `add_with_ids` |
| 3 **Version diff** | **Set operations**: `added = set(v2)-set(v1)`, `removed = set(v1)-set(v2)`, `common = intersection`; modifications by `!=`, **exact text comparison, no LLM** |
| 4 Search | `search(query_vector, k=3)`; `similarity = 1/(1+distance)` (turning "smaller is better" into "larger is better") |
| 5 Evaluation | Record response time; judge correctness by **string containment**; `accuracy = correct / total` |
| 6 Comparison | **A/B testing**: the same test set against two versions, reporting changes in accuracy and time |
| 7 Regression | Re-run historical cases against the new version, `pass rate = passed / total` |

> **Module 5's evaluation has a ceiling**: string containment can verify that the answer's text is
> present, not that the answer is right. The stronger option is to have a model judge, at the cost
> of reintroducing something unreproducible.

**Measured** (v1 holds three basic entries; v2 expands all three and adds two more):

| Function | Result |
| :--- | :--- |
| 1 Fingerprints | v1.0 **3 entries / mean 62 chars / hash `1d0aec844c6e`**; v2.0 **5 entries / mean 124 chars / `61d5d51301c1`** |
| 2 Diff | **2 added, 0 removed, 3 modified** (`kb_001 +50` / `kb_002 +103` / `kb_003 +56` chars) |
| 3 Indexing | 3 and 5 vectors; **mean raw norm 0.620 / 0.622 after truncation to 1024 dimensions, normalised before indexing** |
| 4 Evaluation | 5 cases: v1 accuracy **60%**, v2 **100%**; mean search time **0.017 ms → 0.010 ms** |
| 5 Regression | v1 passed 3 of 5, and **all 3 still pass on v2 — no regressions**; the other 2 were fixed by v2 |

**Three things to read out of these numbers**:

1.  **60% → 100% comes entirely from content v1 never had**: the two failures were "how do I get
    there by public transport" and "which rides should I not miss", and v1's three entries hold no
    answer to either; v2 adds `kb_004` and `kb_005` and they pass.
    ⇒ **A version comparison usually measures knowledge coverage, not retrieval quality. v2 does
    not search better; it has more to find.** That is worth saying out loud before anyone reads the
    number as a search improvement.
2.  **The timing "change" is measurement noise, not a trend**: an exact search over 3 vectors and
    over 5 costs effectively the same, which is why v2 can come out 0.007 ms *faster*.
    **A difference of this size should be reported as "no measurable change", not as a number.**
3.  **Regression testing asks a different question from performance comparison**: the comparison
    asks "is the new version better", regression asks **"did anything that used to work stop
    working"**. A release is signed off on the second.
    But do not overstate it — **five cases cannot certify a release; they can only catch the
    breakages those five cases cover.**

See `12_kb_version_management.py`.

### 7.5 Who does the work: LLM, embeddings and traditional methods

| Scenario | What happens | Main worker |
| :--- | :--- | :--- |
| ① Question generation and retrieval optimisation | Generate varied questions, retrieve with **BM25** | **LLM**: generating questions, assessing retrieval quality |
| ② Conversation distillation | Extract points from conversations, merge and classify | **LLM**: extracting, merging, structuring |
| ③ Health checks | Coverage, freshness, consistency | **LLM**: finding gaps, spotting staleness, identifying conflicts, writing the report |
| ④ Version management and comparison | Version creation, diffing, evaluation, regression | **Embeddings**: indexing and semantic search<br>⚠️ **the diff uses exact text matching, with no LLM involved** |

**The three-way division**: **embeddings** handle semantic retrieval; **the LLM** handles generation
and understanding; **traditional methods** (BM25 keyword retrieval, set operations for diffing,
string matching for hits) handle exact matching.

> **This is the most practical engineering conclusion in the topic: not every step should use an
> LLM.** BM25 and a `!=` string comparison are, in their own places,
> **faster, cheaper and more reproducible.**

### 7.6 Common problems at each stage, and what to do

#### Data preparation

**Three problems**: **poor data quality** (ungoverned unstructured enterprise data may hold
sensitive, stale, contradictory or simply wrong information); **multimodal content** (headings,
colours, images and labels are hard to extract and interpret); **difficult PDF extraction**
(**PDF is designed for human reading, and machine parsing is genuinely hard**).

**Five steps**:

1.  **Assessment and classification** — audit the data (identify sensitive, stale, contradictory or
    inaccurate content) and classify it (by type, source, sensitivity, importance).
    *Sensitive*: names, identity numbers, phone numbers, account numbers, transaction records,
    payment card details ⇒ a leak risk if stored unencrypted.
    *Stale*: contact details never updated, closed business still marked active ⇒ failed
    communication and wrong decisions.
2.  **Cleaning** — deduplicate, correct, update, and run a **consistency check to resolve
    contradictions**.
3.  **Sensitive data handling** — identify personal information with tooling or regular
    expressions, then redact or encrypt.
4.  **Labelling and annotation** — metadata (source, creation time) plus content annotation.
5.  **A governance framework** — policies, ownership, monitoring and audit.

**The intelligent document pipeline**: many input formats (PDF/Word/Excel/images/HTML/MD/slides) →
**parsing** (format parsers plus OCR → a unified representation: text, layout, tables, images,
outline, formulas) → **understanding** (multimodal understanding plus domain pre-training) →
**a document tree** → **analysis** (layout analysis, information extraction, classification,
question answering) → downstream applications (contract extraction, review and comparison;
knowledge extraction, search, document QA, table understanding).

> **PDFs full of tables**: use a dedicated parser to **convert tables and images to Markdown**
> before loading them. This is the same idea as converting Word tables to Markdown in 4.6:
> **turn unstructured content into model-friendly structured text before you talk about retrieval.**

#### Retrieval

**Two problems**: **missing content** (retrieval misses the key material, so the answer is
incomplete); **relevant documents ranked too low** (the right document *was* retrieved but sits far
down). The root of the second is that in theory everything is ranked, **but in practice only the
top k is fetched, and k is set by experience**.

**Path one: clarify intent through query transformation**
*Scenario*: "how do I apply for a credit card?" *Problem*: is the question about the steps, the
documents required, or eligibility?
*Steps*: **intent detection** → **query expansion** → expand into "the steps to apply", "the
documents required to apply" and "the eligibility criteria" → retrieve with the expansions.

**Path two: hybrid retrieval plus reranking**
*Scenario*: "what is the annual fee?" *Problem*: retrieval returns a great deal, and the relevant
item ranks low.
*Steps*: **hybrid retrieval** (keyword plus semantic) → **rerank** → generate from the reranked set.

> **The division between the two**: query transformation **clarifies intent and raises retrieval
> accuracy**; hybrid retrieval and reranking **ensure the most relevant document is handled first**.

#### Generation

**Four problems**: **not extracted** (the answer is in the context but the model did not pull it
out, usually because **the context is noisy or self-contradictory**); **incomplete**; **wrong
format** (the formatting instruction was misread); **hallucination**.

**Path one: better prompt templates** — replace "answer the question from the context below" with an
explicit statement of what to extract:

| Question | Improved instruction |
| :--- | :--- |
| How do I apply for a credit card? | From the context below, **extract the specific steps and the documents required to apply** |
| What is the annual fee? | From the context below, **list the annual fee for each card type and state whether any waiver applies** |
| What is a fixed-instalment savings account? | From the context below, **explain the definition, characteristics and target customers accurately and verifiably** |

Optimising the prompt itself can also be delegated to a reasoning model: ① **extract** the key
information from the original prompt → ② **analyse** what the user actually wants →
③ **rewrite** the prompt.

**Path two: dynamic guardrails**
Monitor and adjust the output **during generation**, intervening through rules, constraints and
feedback — which maps onto the four problems above:

| | Against "not extracted" | Against "incomplete" | Against hallucination |
| :--- | :--- | :--- | :--- |
| **Rule** | Check the answer contains both steps and documents; regenerate if not | Check every card type's fee is listed; ask for the rest if not | Check the answer agrees with the context; regenerate if not |
| **Output it catches** | "Applying requires some documents." | "Card A's annual fee is 100." | "A fixed-instalment savings account is a loan product." |

**How to write factual-verification rules**: where the business logic is clear and the rules are
fixed, define them by hand — **rule 1**, the answer must contain the **key entities** from the
retrieved passages; **rule 2**, the answer must follow **the specified format** (a step list, a
table). **Implementation: regular expressions or keyword matching**, no model required.

### 7.7 Industry practice at each stage

*   **Data preparation · multi-granularity extraction** — where documents carry several heading
    levels with relationships between them, split by heading level and train
    **a model dedicated to knowledge extraction**, extracting and combining chunks at each
    granularity and deduplicating so nothing is lost or repeated,
    **finally turning the document into a set of factual dialogues that retrieve better**.
*   **Retrieval · multi-route recall** — vector recall through two families (large-model vectors and
    conventional deep-model vectors) and search recall through several routes (keywords, n-grams).
    **Multi-route recall reaches a high recall rate.**
*   **Generation · two-phase generation** — to address weak factuality and missing logic,
    **produce an outline first, then expand the final answer from it**.

---

## 8. Practice Tasks

**Practice one · Document QA with citations**
Assemble your own knowledge base → extract the text while **recording a page number per character**
→ chunk it and build the vector store → retrieve by similarity → generate through a QA chain →
**show the source page for every chunk used**. The last step is the point: page mapping is the
easiest thing to get wrong and the clearest signal of engineering quality.

**Practice two · A multimodal assistant**
**Data layer**: parse `.docx` for paragraphs and tables (converted to Markdown); run OCR and visual
feature extraction over images.
**Vector layer**: a text embedding model plus CLIP, in **two FAISS indexes**.
**Retrieval layer**: hybrid retrieval (semantic similarity for text, CLIP's text encoder for
images) with keyword triggering.
**Generation layer**: assemble the retrieved context into a structured prompt with sources labelled.

**Practice three · Query rewriting**
Rewrite your own queries under all five types (context-dependent, comparative, ambiguous reference,
multi-intent, rhetorical), then have a single intent-detection prompt classify and rewrite
automatically, and compare the two approaches.

**Practice four · Web-search augmentation**
① Define the scenarios that need live data ② **the decision logic** (does this need a search)
③ **the rewriting logic** (rewrite for a search engine) ④ **the search plan** (sites and keywords,
optional).

**Practice five · Knowledge base operations (choose one)**
Pick one of question generation and retrieval optimisation / conversation distillation / health
checks / version management and performance comparison, and implement it against your own domain.

---

## 9. Questions and Answers

### Embeddings and vector basics

*   **Q: Why do texts of different lengths end up with the same number of dimensions?**
    A: Because they have to, in order to be comparable and computable. That is the point of an
    embedding.
*   **Q: After embedding, will many vectors be identical? Does identical mean similar in meaning?**
    A: Similarity of meaning is decided by **a similarity computation**, not by equality.
*   **Q: Is the dimensionality the result of segmentation?**
    A: No. The flow is "raw sentence → segmented into `[a b c d]` → **compressed into a space**",
    and the dimensionality is a property of the compression, unrelated to the token count.
*   **Q: Do I have to train my own embedding model?**
    A: No, a pre-trained one is fine. For specialist needs you can train on your own corpus
    (Word2Vec or a BERT-family model both work).
*   **Q: The prompt already carries so much background knowledge that it could answer directly. Why
    vectorise at all?**
    A: **That background was selected out of the store by embedding similarity in the first place.
    The purpose of the embedding computation is to filter.**
    (The single most important question here — **vectorisation is not for answering, it is for
    selecting**.)
*   **Q: How do a vector database and a vector matrix relate?**
    A: **A vector database is management software** (offering vector computation among other
    things); **a vector matrix is a raw data format**.
*   **Q: Which model vectorises images?** A: CLIP, DINOv2. Text uses BERT- and GPT-family models.
*   **Q: An LLM can also understand meaning. What is the difference from an embedding model?**
    A: **Cost.** An embedding model is orders of magnitude smaller than a reasoning model.
*   **Q: Do recommender systems still need an LLM?**
    A: Recommenders have their own neural architectures (DeepFM, NFM, Wide & Deep) trained on your
    data. The LLM handles higher-level understanding and generation, and the two **work together**.

### Environment and deployment

*   **Q: What is the minimum hardware to run an embedding model of a couple of gigabytes?**
    A: CPU is enough, or 4 GB of VRAM and up.
*   **Q: How do I deploy this on Linux?** A: Much as on Windows — `pip install` the dependencies and
    run the script.
*   **Q: I have downloaded a pile of models. How do I know how to use each one?**
    A: **Read the examples on the model's own page, get one running, then understand it.**
*   **Q: Does the file format matter for embedding? Is there a difference between txt and markdown?**
    A: Convert PDFs and slides to **Markdown**, because **less is lost** — it supports code blocks,
    tables and embedded images, so layout survives better.
    **What matters most is how well the content is parsed, not the format itself.**
*   **Q: Do I need to be able to type this code out from memory?**
    A: No — **what matters is the logic.** Reasoning about design beats memorising syntax, though
    reviewing the code and checking the logic still needs a person.

### RAG engineering and selection

*   **Q: When RAG and when fine-tuning?** A: **RAG first, fine-tuning only if that fails.**
*   **Q: Can audio go into a knowledge base?** A: Transcribe it to text first, then embed.
*   **Q: How is the knowledge base updated?** A: ① rebuild the index; ② add or remove individual
    entries through the vector database.
*   **Q: Can I build several stores with different chunk sizes for different kinds of question?**
    A: **Yes, several stores is a common arrangement.**
*   **Q: In one chat box, how do I know which questions need RAG and which the model can answer
    directly?**
    A: Answer directly when it can (**and attach a confidence**); otherwise call external retrieval
    (a web search) and then run RAG over what comes back. Going further,
    **register RAG as a tool and let an agent decide.**
*   **Q: Technical PDFs full of tables need exact numbers, and RAG does badly. What now?**
    A: Use a dedicated parser to **convert tables and images to Markdown** before loading.
*   **Q: The more complex a multimodal RAG gets, the harder it is to debug — fixing one thing breaks
    another.**
    A: Separate the pipeline into stages and verify each. The standard chain is
    `QAChain(LLM, chain_type) → similarity_search(query, k) → {input_documents, question} → invoke`.
*   **Q: Can RAG be built with a single prompt?**
    A: No. Every stage carries engineering quality, and whoever operates it needs to be able to read
    the code. Build up step by step.

### Recall, reranking and chunking

*   **Q: Is "recall" a term of art here?** A: Yes. **Recall means finding the relevant subset out of
    a large store**, after which reranking narrows it.
*   **Q: What exactly does reranking do?** A: A rerank model computes the ordering.
*   **Q: Is a reward model the same as a rerank strategy?** A: No, it is a **rerank model**. The
    names are similar; the things are not.
*   **Q: How should I chunk?** A: Step 1, chunk (by rule or by model); step 2, embed the chunks.
    **Chunking and vectorisation are two separate steps.**
*   **Q: How do I improve query accuracy?** A: Multi-query recall (have a tool ask several similar
    phrasings), or have the model rewrite the query.
*   **Q: Is one intent-detection prompt enough for query rewriting?** A: Yes.
*   **Q: Does every query need rewriting?** A: **No.** Short literal questions retrieve fine as they
    are.
*   **Q: Is confidence the cosine of the angle between vectors?**
    A: **No — it is a score the model produces, with an element of impression to it.**
*   **Q: How do I get the model to rewrite a query for a web search?**
    A: Define the input and output — input `query, current_time`; output **JSON**.

### Two-way rewriting and knowledge base operations

*   **Q: Query2Doc seems to have written the answer already. Is the rest of the pipeline needed?**
    A: **Yes.** The knowledge used for augmentation must come from the private store; the rewrite
    only **points retrieval in a direction**.
*   **Q: There are so many query optimisation strategies. How do I choose?**
    A: Query2Doc and Doc2Query share one purpose —
    **building more links between questions and documents** — and combine according to the setting.
*   **Q: How do these work together?**
    A: Split them into tools for an agent — `tool1: query rewriting`, `tool2: web search`,
    `tool3: question interpretation` — and **let the agent choose**.
*   **Q: Where are generated questions stored? What if my tooling has no extra storage?**
    A: **In the same chunk as the source text**: `chunk = {source text, generated questions}`.
*   **Q: How do question retrieval and vector retrieval work together?**
    A: **Vector retrieval selects the chunks; the question plus those chunks then goes to the model
    to answer.**
*   **Q: What is the difference between a knowledge base and RAG?**
    A: **RAG is the process of using a knowledge base** — retrieval, augmentation, generation.
*   **Q: How does a knowledge base relate to a vector database?**
    A: **The vector database selects knowledge**, because it can compute over vectors
    mathematically (cosine similarity).
*   **Q: What is the difference between a knowledge base and a knowledge graph?**
    A: **A knowledge base is the source documents; a knowledge graph is a `Graph<node, edge>` built
    on top of them — an organised set of notes over the original knowledge.**
*   **Q: What do I do about contradictory data?** A: **The model finds the conflicts, a person makes
    the call.** Judgements that cannot be enumerated need the model's generality; have it emit the
    conflicts as JSON.
*   **Q: What tooling do people actually use to manage a knowledge base?**
    A: A low-code agent platform, or something built in-house.
*   **Q: How is the accuracy figure produced?**
    A: Run a test set. Hits can be judged by keyword containment or by having a model score them.

### Multimodal

*   **Q: How are an image store and a text store linked? Say a user asks about an event and wants
    both text and a picture.**
    A: By rule — ① match the text embedding first, giving **text recall**;
    ② if a keyword fires or the model judges that a picture is wanted, match the image embedding,
    giving **image recall**. The prompt is then assembled as:
    ```
    Here is the text knowledge: {chunk_list}
    Here is the image content found, from file XXX: {image_list}
    Answer the user's question using the knowledge above: {query}
    ```
*   **Q: What are the routes for image embedding?**
    A: ① CLIP (512-dim) ② a multimodal embedding model ③ a vision model that turns the image into
    text, which is then embedded.

### Other

*   **Q: Which framework should an operations agent use?**
    A: Any of the mainstream agent frameworks. **What matters is the design of the toolbox, not the
    framework** — work out which tools the agent needs first.
*   **Q: Can a knowledge base built from internal standards be attached to office software to guide
    document writing?** A: Yes, through an agent.
*   **Q: Is embedding a local model in a phone app the future?**
    A: Development is going both ways — **one pole is large (flagship models), the other is small
    (embedded, on-device)**.

---

## 10. Summary

| Subject | Core content | Key points |
| :--- | :--- | :--- |
| **Text representation** | Counts → N-grams → TF-IDF → word vectors | Counts **ignore order**, so opposite meanings share a vector; N-grams add order, TF-IDF adds weight |
| **TF-IDF** | TF = term frequency; IDF = log(total docs / docs containing it + 1) | The fewer documents a term appears in, **the better it discriminates** |
| **Word2Vec** | Really a lookup table | one-hot × weight matrix = selecting a row; hidden units = embedding dimensions |
| **Embeddings vs LLMs** | Feature extractor vs generator | **Vectorisation is not for answering, it is for selecting** |
| **Model selection** | MTEB plus four model families | Read **the column matching your scenario**, not the average; build your own test set |
| **Matryoshka** | One model, several output sizes | **Renormalise after truncating**, or vector length leaks into the ranking |
| **Pooling** | CLS / mean / last-token | **A property of the model, not a free choice**; getting it wrong raises no error, it just degrades |
| **Vector databases** | Six options | FAISS suits static data; what comes back is **a distance, where smaller is closer** |
| **Loading data** | Clean → vectorise → load with metadata | **Metadata is what makes advanced retrieval and citation possible** |
| **Three approaches** | Prompting / RAG / fine-tuning | Unclear question / missing knowledge / missing capability; **RAG first** |
| **RAG's three steps** | Indexing / Retrieval / Generation | How to store, how to find, how to answer; **the second is where it breaks** |
| **Page citation** | Per-character page mapping, modal page per chunk | Per-line mapping is guaranteed to drift; `TOP_K` must be rechecked as the store grows |
| **Multimodal RAG** | Two indexes + keyword trigger + text first | **Distances from two indexes are not comparable** (0.98 vs 123.6); never index headings alone |
| **QA chains** | stuff / map_reduce / refine / map_rerank | **If stuff works, use stuff** |
| **Chunking** | Five strategies compared | No silver bullet, choose by document type; **chunking and vectorisation are separate steps** |
| **Small-to-Big** | Index the small, answer from the big | Fast to locate, complete to answer, **lowers long-document cost** |
| **Index expansion** | Discrete (BM25) / continuous / hybrid | Entities and years are **the blind spot of vector retrieval** and need exact matching |
| **Reranking** | Bi-encoder screens, cross-encoder ranks | Scores are **unnormalised logits**, **the correct answer can also be negative**, the sign is not a threshold; the granularity fed in changes the scores |
| **Query rewriting** | Five types plus a single-prompt classifier | **Not every query needs it**; the product is an intermediate for the embedding model |
| **Confidence** | An impression score from the model | **Not a cosine**; fine as a sort key, **not as a KPI** |
| **Web search** | Eight scenarios, three functions | The model **has no sense of time**, the date must be injected; here confidence really is a threshold |
| **Two-way rewriting** | Query2Doc / Doc2Query | Query2Doc's output **is a query, not an answer** |
| **GraphRAG** | Entities plus relationships as a semantic network | Causality **lives in the data structure**; it does not remove hallucination; **the baseline does not lose on a small corpus**, and indexing scales with the corpus rather than the question count |
| **Knowledge base life cycle** | Questions → distillation → health checks → versioning | Distillation **must filter the need and question types**; audits should **plant defects first**, and the findings list is trustworthy where the score is not |
| **Question retrieval** | Matching question against question | 33.3% → 66.7%, **but one query went from right to wrong** — the extra vocabulary cuts both ways, so read the net |
| **Division of labour** | LLM / embeddings / traditional methods | **Not every step should use an LLM**; BM25 and string comparison are faster and more reproducible |

---

## 11. Scripts Produced in This Module

| Script | Knowledge covered |
| :--- | :--- |
| `01_tfidf_hotel_recommender.py` | TF-IDF and n-gram features, cosine similarity, content-based recommendation (1.5 / 1.6) |
| `02_word2vec_similarity.py` | Segmentation, Word2Vec training and persistence, vector arithmetic (1.8) |
| `03_embedding_faiss_metadata.py` | Embedding calls, Matryoshka dimensions, FAISS indexing with metadata, persistence (2.5 / 3.2 / 3.3) |
| `04_embedding_models_compare.py` | Hand-written pooling vs a wrapper, CLS against mean pooling (2.6) |
| `05_chunking_strategies.py` | Five chunking strategies compared side by side (5.1 / 5.2) |
| `06_chatpdf_langchain_faiss.py` | End-to-end QA with LangChain and FAISS, per-character page citation, rechecking TOP_K (4.5) |
| `07_disney_multimodal_rag.py` | Multimodal RAG without a framework, two indexes, CLIP cross-modal search, OCR, vision description (4.6) |
| `08_query_rewriting.py` | Five rewrite types, single-prompt intent detection and confidence (6.5 / 6.6) |
| `09_rerank_and_multiquery.py` | Two-stage retrieval (wide recall then reranking), multi-query expansion (6.3 / 6.4) |
| `10_kb_question_generation.py` | Doc2Query in practice, dual BM25 indexes, retrieval evaluation (6.7 / 7.1) |
| `11_kb_curation.py` | Conversation distillation (extract / filter / merge) and health auditing (7.2 / 7.3) |
| `12_kb_version_management.py` | Version hashes, set-operation diffing, A/B and regression testing (7.4) |
| `13_graphrag_vs_vector.py` | One multi-hop question asked of a vector index and of a knowledge graph, with cost accounting (6.9) |
