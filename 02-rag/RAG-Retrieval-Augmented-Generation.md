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
*   **What sklearn computes**: `TfidfVectorizer` uses the smoothed form
    `idf = ln((1 + n) / (1 + df)) + 1`, where n is the number of documents and df the number that
    contain the term, multiplies it by the term count, and scales every vector to length 1. That is
    why the dot product (`linear_kernel`) of two vectors is their cosine similarity.

### 1.6 Practice one: content-based recommendation with TF-IDF

**Goal**: given one hotel, recommend the 10 most similar ones in the dataset.
**Data**: **152** Seattle hotels with the fields `name / address / desc`.
**Method**: turn each description into a TF-IDF vector, compute the cosine similarity between the
target hotel and every other one, and take the top k.

**Six parts** (see `01_tfidf_hotel_recommender.py`):

1.  Dataset. The number of hotels and the three columns.
2.  One description. The raw description of W Seattle, before any processing.
3.  Frequent phrases. The 20 most common three-word phrases across all 152 descriptions, with stop
    words removed. "pike place market" comes first, whether or not it tells hotels apart.
4.  Cleaning. Every description is lowercased and stripped of punctuation and stop words.
5.  TF-IDF and cosine similarity. TF-IDF vectors over terms of one to three words (a measured
    vocabulary of **3348**, a matrix of shape `152 × 3348`), then the cosine similarity of every
    pair of hotels in a symmetric `152 × 152` matrix whose diagonal is 1.
6.  Recommendations. For two hotels, their rows are sorted and the ten most similar other hotels
    are printed, **excluding the hotel itself** via `iloc[1:11]`.

**Key components**: `CountVectorizer` (n-gram extraction), `TfidfVectorizer` (TF-IDF transform),
`linear_kernel` (fast cosine similarity), `pandas.Series` (sorting and indexing).

**Engineering detail**: nearly all of those 3348 features are zero, so **the matrix is extremely
sparse**; this is also why methods of this kind are memory-hungry.

**Measured**: querying `Hilton Seattle Airport & Conference Center` returns mostly airport-area
hotels (first place `Embassy Suites ... Seattle Tacoma International Airport`, 0.2266), though not
only: fourth is `Four Points by Sheraton Downtown Seattle Center` and ninth is `Hotel Hotel`.
Querying `The Bacon Mansion Bed and Breakfast` returns five B&Bs and inns at the top (first place
0.2801).

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

1.  Segmentation. The novel, stored as GB18030 text, is cut into words with jieba.
2.  Baseline model. 100-dimensional vectors, a window of 3, every word kept (vocabulary **46005**).
3.  Name similarity. The baseline model's cosine similarity between Sun Wukong and Zhu Bajie,
    Pilgrim Sun (his other name) and "monster".
4.  Analogy. Sun Wukong is to Pilgrim Sun as Tang Seng is to what? The baseline model's top answer
    is **长老** (elder, 0.9805), the way other characters address Tang Seng.
5.  Second model. 128-dimensional vectors, a window of 5, and words seen fewer than five times
    dropped, which shrinks the vocabulary to **7735**. It is saved to disk.
6.  Reload. The reloaded model scores exactly as before saving, so persistence lost nothing.
7.  English corpus. The same recipe on text8 for contrast.

**Measured, with a caveat**: the baseline trains on one thread and gives the same numbers every
run. The second model trains on several threads, so its numbers move between runs: `孙悟空 vs
猪八戒` has ranged from 0.9265 to 0.9369, `唐僧 vs 沙僧` from 0.8277 to 0.8603, and its analogy has
put 长老 or 菩萨 first.

> **⚠️ Those scores are not evidence the model is good.** Every pair of characters in this corpus
> scores **above 0.8** (`孙悟空 vs 妖怪` reaches 0.9602). The reason is not model quality but that
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
recomputing. Comparing retrieval over the four FAQ entries at all three sizes, the **top-3 ordering is
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
*   **Last-token pooling**: take the last valid token. Decoder models such as Qwen3-Embedding use
    it, with left and right padding handled separately.

**Each model fixes its pooling strategy during training. Getting it wrong raises no error; it just
quietly degrades the result.**

**Practice three: comparing three local models** (see `04_embedding_models_compare.py`). Two
questions, one about a ticket refund and one about annual pass perks, are scored against two
passages that answer them:

| Model | Structure | Pooling |
| :--- | :--- | :--- |
| BAAI/bge-small-en-v1.5 (BAAI General Embedding) | encoder, BERT-style | CLS |
| GTE small (General Text Embeddings, Alibaba) | encoder, BERT-style | mean |
| Qwen3-Embedding-0.6B | decoder | last token |

1.  CLS pooling. bge through the SentenceTransformer wrapper: each question scores highest against
    its own passage (0.87 and 0.72).
2.  Mean pooling. GTE through the wrapper, with the same result (0.92 and 0.85).
3.  Last-token pooling. Qwen3 through the wrapper, with the same result (0.72 and 0.72).
4.  By hand. Tokenising, pooling and normalising written out for each model, and the largest
    difference from the wrapper's scores. Qwen3 runs with right and with left padding, because the
    last real token sits in a different place.
5.  Wrong pooling. bge with mean pooling.

**Measured, step 4**:

| Hand-built | Largest difference from the wrapper |
| :--- | ---: |
| bge, cls, right padding | 0.000000 |
| GTE, mean, right padding | 0.000505 |
| Qwen3, last, right padding | 0.002961 |
| Qwen3, last, left padding | 0.002961 |

**Measured, step 5**:

| | Deviation from the wrapper | Margin (right passage over wrong, mean of both questions) |
| :--- | ---: | ---: |
| cls (right) | 0.000000 | +0.2198 |
| mean (wrong) | 0.069623 | +0.2548 |

The wrong pooling raises nothing and still ranks each question's own passage first, and its margin
is even larger. Only the deviation shows the mistake.

**Why last-token pooling works for a decoder**: after many attention layers a token's output
carries what that token could see. An encoder attends both ways, so every position has seen the
whole text and CLS or mean both make sense. A decoder attends only to earlier tokens, so only the
last one has seen the whole text.

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

See `03_embedding_faiss_metadata.py`. Four Disney FAQ entries (two about ticket refunds, one
about the annual pass, one about a ride closed for maintenance) and the question "I want to
understand the refund process for Disney tickets" are embedded with `gemini-embedding-001` at 768
dimensions. Seven parts:

1.  One embedding. The refund question as a 768-value vector of length 1.
2.  Matryoshka dimensions. The same question at 3072, 1536 and 768 dimensions; the first values
    match.
3.  Ranking per dimension. The whole search at each size, and whether the top three stay the same.
4.  Document embeddings. The four entries at 768 dimensions.
5.  FAISS index. A flat L2 index wrapped in `IndexIDMap`, so each vector carries its entry's own id
    (doc3 as 3) instead of its position.
6.  Search. The three nearest entries with squared L2 distance, and their text and metadata looked
    up by id. The two refund entries come first.
7.  Save and reload. The reloaded index returns the same ranking. The metadata lives outside the
    index and has to be saved with it.

**Key points**:

*   **Index type**: `IndexFlatL2` is an **exact index** (brute-force comparison), wrapped in
    `IndexIDMap` so it can carry custom ids.
*   **What comes back is a distance, not a similarity**: `IndexFlatL2` returns the **squared** L2
    distance, where **smaller is closer**. To convert to "higher is better", use
    `similarity = 1 / (1 + distance)`. For unit vectors, squared L2 = 2 - 2 x cosine, so both rank
    the same way: doc3 at 0.3142 is a cosine of 0.84, doc2 at 0.8528 a cosine of 0.57.
*   **Why `IndexIDMap`**: FAISS alone returns positions. With the wrapper each vector keeps an id of
    your choosing, so deleting or reordering entries cannot shift which document a hit points to.
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

`06_chatpdf_langchain_faiss.py` answers two questions about a nine-page PDF, a bank's rules for
assessing retail account managers, and cites the pages each answer came from. One question asks
how many points a customer complaint costs, the other when the yearly appointment review opens.
LangChain splits the text, embeds the chunks (`gemini-embedding-001`) and builds a FAISS store;
`gemini-3.1-flash-lite` writes the answers. The run prints five parts: reading and chunking,
vector store, reload, question answering, and several phrasings.

- The text comes out of pypdf page by page and is split with `RecursiveCharacterTextSplitter`,
  `chunk_size=1000` and `chunk_overlap=200`, trying `["\n\n", "\n", ".", " ", ""]` in order.
- The page citations are the script's own work. The splitter cuts the text wherever it likes, so
  a page number recorded per line cannot be matched to chunk i. The script records a page number
  for every character, finds each chunk in the text with `str.find`, and gives the chunk the
  page most of its characters came from.
- Pages are joined with a single newline. Each page ends on its footer (`- 5 -`), and without the
  newline the next page's first line was glued to it. The extracted text has no `\n\n`, so joining
  with `\n\n` would have made the page breaks the splitter's first choice and stopped chunks from
  crossing pages. With `\n`, 10 of the 15 chunks still span two pages.
- The prompt holds the 4 nearest chunks (the "stuff" approach: every retrieved chunk in one
  prompt, one model call).

| Measure | Result |
| :--- | :--- |
| Text and chunks | 11968 characters from 9 pages, 15 chunks, all 15 with a page number |
| Complaint question | `2 points are deducted for each customer complaint.`, pages [5, 6, 3] |
| Review question | `January of each year is the application window…`, pages [6, 4, 8, 7] |

- `TOP_K` has to fit the size of the store. The script once used `TOP_K=10` on a store of 5
  chunks, which returned everything, so the cited pages looked precise while nothing had been
  filtered. With 15 chunks one answer cited 8 pages, and k=4 made the citations mean something.

Part 5 rewrites each question three ways, retrieves under all four phrasings and merges the chunks.
Each question carries the page that holds its answer (5 and 6, checked against the PDF), so the run
shows whether the extra phrasings found anything that mattered.

```
    single phrasing : 3 pages [5, 6, 3], answer page 5 at position 1
    4 phrasings     : 3 pages [5, 6, 3], answer page 5 at position 1
    new pages       : none

    single phrasing : 4 pages [6, 4, 8, 7], answer page 6 at position 1
    4 phrasings     : 6 pages [6, 4, 8, 7, 3, 2], answer page 6 at position 1
    new pages       : [3, 2], not needed, the answer page was already found
```

- Both answer pages are first with the question alone, so the rephrasings bring in nothing the
  answer needs. The two extra pages for the second question only make the context longer.
- Rephrasing helps when the asker's wording misses the vocabulary of the document. These two
  questions already use the document's words.

### 4.6 Practice six: a multimodal RAG assistant built by hand

`07_disney_multimodal_rag.py` builds a small assistant for a Disney park without LangChain. The
knowledge base is four Word files (ticket rules, senior tickets, a visit guide, hotel and
membership services) and two event posters. The run prints three parts: indexing, retrieval and
answers, and the vision model.

- Each Word file is walked element by element. A paragraph gets the heading above it as a prefix,
  and a table becomes Markdown (header row, separator row, one row per table row). The four files
  give 26 text blocks.
- Headings are never indexed alone. When they were, a five-word block such as `Ticket Rules`
  ranked near the top for any ticket question and pushed the refund clause out of the top k.
  Prefixing them dropped the blocks from 30 to 26, and the refund question was answered.
- Text is embedded with `gemini-embedding-001` at 1024 dimensions and normalised, because a
  vector truncated to 1024 dimensions comes back with a length of about 0.6.
- The posters are embedded with CLIP (`openai/clip-vit-base-patch32`, 512 dimensions). CLIP puts
  pictures and text in one space, so a question encoded by CLIP's text encoder can find a picture.
  The 1024-dimension text vectors and the 512-dimension CLIP text vectors are both text vectors,
  but they are in different spaces.
- CLIP vectors are normalised too. CLIP is trained on cosine similarity, so only the direction
  means anything. Before normalising, the two posters had lengths of 8.56 and 9.59, and a query
  describing the Halloween artwork (`a purple night sky with a yellow moon and bats`) picked the
  Lunar New Year poster on L2 (108.12 against 115.06) although its cosine favoured Halloween
  (0.222 against 0.155). Normalised, it picks Halloween (1.5566 against 1.6899).

An image can reach a text-only prompt in three ways, and the script shows all three.

| Route | What it gives | Where it goes |
| :--- | :--- | :--- |
| CLIP | A vector, so a text question can find the picture | The image index |
| OCR (`rapidocr-onnxruntime`, pip only) | The words printed on the picture | The image's record, and the prompt when the image is found |
| Vision model | A description of the picture itself | Printed for comparison, not used in the answers |

Every question searches the text index for the 3 nearest blocks. The image index is searched only
when the question contains a word such as `poster`, `picture` or `look like`, and then it returns
the single nearest image. The two sets of hits are not merged by distance. Both are squared L2
between unit vectors (2 minus 2 times the cosine), so the numbers look alike, but the models spread
their scores differently: a strong CLIP match sits near cosine 0.3.

| Question | Text hits (squared L2) | Image hit |
| :--- | :--- | :--- |
| Refund process | ticket rules blocks 3, 9, 11 (0.5385 to 0.6558) | not searched |
| What the Halloween poster looks like | blocks 11, 20, 25 (0.9827 to 1.0563): rain, senior visits, concierge service | `02_halloween.jpeg`, 1.3663 (cosine 0.317) |
| Annual pass discounts | blocks 4, 13, 24 (0.6377 to 0.7712) | not searched |

- The poster question finds no relevant text, because the text files say nothing about Halloween.
  Its answer (dates, night event, separate ticket, parade) comes from the poster's OCR text, and it
  cannot say what the poster looks like.
- The prompt labels each passage `[Source N: file]` and tells the model to answer only from them
  and to say so when the answer is not there.

Part 3 shows what the third route adds. Asked to describe the picture and its colours, the vision
model answers with what OCR cannot read:

```
This poster features a dark, gradient purple night sky illuminated by a bright yellow full moon
and accented with simple, dark geometric bats. ...
```

An earlier prompt, `What kind of poster is this? Answer briefly.`, returned only `an event poster
promoting a Halloween-themed festival`, which the OCR text already says.

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

`05_chunking_strategies.py` splits one theme park ticket guide five ways. The guide has three
paragraphs (ticket types, buying a ticket, discounts) and is 1299 characters long, or 1366 in the
copy with headings. The target chunk size is 800. The script measures chunk sizes only. It does
not test which strategy retrieves better.

| Strategy | What the code does | What the run shows |
| :--- | :--- | :--- |
| 1 Fixed length | Cuts every 800 characters, moves back to the last sentence end within 200 characters, and starts the next chunk 150 characters earlier | The second chunk starts at `resellers also sell them`, in the middle of a sentence |
| 2 Sentence packing | Packs whole sentences into chunks of up to 800 characters | No sentence is cut, but the first chunk runs from paragraph 1 into the middle of paragraph 2 |
| 3 LLM | A chat model picks the break points and replies with JSON | DeepSeek returned the three paragraphs unchanged: 414, 428 and 453 characters |
| 4 Hierarchical | Starts a new chunk at every heading (on the copy with headings) | The title `# Ticket Guide` becomes a chunk of 14 characters |
| 5 Sliding window | Moves an 800-character window forward 450 characters at a time | Neighbouring chunks share 350 characters, chunks start mid-word (`ficial channels`), and 1999 characters are indexed for 1299 |

- Strategy 2 is called sentence packing because it only counts characters. No embedding is
  involved, so it is not semantic chunking.
- Strategy 4 only splits at headings. A chunk does not carry its parent heading, so the
  `## Discounts` chunk does not say it belongs to the ticket guide.
- The LLM prompt asks for chunks of at most 800 characters that are semantically complete and
  break at natural boundaries, returned as `{"chunks": [...]}`. The run counts how many chunks
  appear word for word in the guide, because a model can reword text while splitting it. Here it
  was 3 of 3.

### 5.2 The side-by-side table

```
  strategy                chunks     avg     min     max   spread
  1. Fixed length              2     724     684     765       81
  2. Sentence packing          2     648     613     683       70
  3. LLM                       3     432     414     453       39
  4. Hierarchical              4     339      14     466      452
  5. Sliding window            3     666     399     800      401
```

- Spread is the longest chunk minus the shortest.
- The LLM has the smallest spread, but only because it split at the blank lines. Splitting on
  blank lines gives the same three chunks without an API call.
- Hierarchical has the largest spread because of the 14-character title chunk. Without it the
  spread would be 36 (466 minus 430).
- Without an API key, or when the call fails, part 2 runs in place of part 3 and the row reads
  `3. LLM (fell back)`. An earlier version fell back silently, so rows 2 and 3 held the same
  numbers and nothing showed that the model had never run.

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

*   The embedding model used for retrieval is a **bi-encoder**: the query and the document are
    encoded **independently** into vectors, which are then compared. The advantage is that document
    vectors can be **computed in advance**, so a search encodes only the query.
*   A rerank model is a **cross-encoder**: the query and the document go into the model
    **together**, and it **emits a relevance score directly**. Far more accurate, but
    **every pair costs a forward pass** and nothing can be precomputed.
*   ⇒ That is why the two are stages of one pipeline: the bi-encoder is cheap enough to screen, and
    the cross-encoder is accurate enough to be worth running on the few survivors.

**Open-source option**: a local cross-encoder is enough. The script uses
`cross-encoder/ms-marco-MiniLM-L-6-v2` (tens of megabytes, runs on CPU; larger rerank models are
typically 2 to 3 GB). The model is a supervised sequence classifier trained on human-labelled
query-document relevance.

**⚠️ Reading the scores is where this is most often misunderstood**: the output is
**unnormalised logits with no fixed bounds**. Measured on one question:

```
 -6.11  The date can be changed once, free of charge, up to 48 hours …   ← the correct answer
-11.36  The Eiffel Tower is a wrought-iron lattice tower in Paris.       ← entirely unrelated
```

**Note that the correct answer also scores negative.** Therefore:

*   **The sign is not a relevance threshold**: there is no rule of the form "above zero means relevant".
*   **Only the relative ordering within one query on one corpus carries meaning.**
*   **Change the model or the chunk size and the scores stop being comparable.** Do not treat them
    as absolute values that travel between settings.

**Commercial API option**: strong multilingual support, **normalised 0 to 1 scores that are easier to
interpret**, ready-made framework integrations, and a particularly good fit for
**reordering the output of hybrid retrieval (BM25 + vectors)**, measured by hit rate and MRR.

| Property | Open-source rerank | Commercial rerank API |
| :--- | :--- | :--- |
| Deployment | Local | Cloud |
| Cost | Free, but needs memory | Metered, no operations |
| Scores | Unnormalised logits | Normalised 0 to 1 |
| Fits | Sensitive data, vertical domains | Fast integration, many languages |

**Practice eight: two-stage retrieval** (see `09_rerank_and_multiquery.py`):
the knowledge base holds 26 paragraph chunks (108 sentence units) from 4 documents.
Stage one uses BM25 to keep 8 of the 26 paragraphs. Stage two uses the cross-encoder to rank
**the sentences inside** those 8 and keeps 3. Each paragraph carries its file's heading for BM25,
so no heading is a chunk of its own (the contamination described in 4.6).

> **The unit fed to the reranker is a setting, not a detail.** Same question, same correct answer.
> Only the amount of surrounding text changes, and the score moves a long way:
> ```
>  -8.60  heading + whole paragraph   (506 chars)
>  -6.43  whole paragraph             (469 chars)
>  -5.82  the answering sentence      (113 chars)
> ```
> If the reranker scored the 8 recalled paragraphs whole instead of by sentence, a wrong one would come first:
> ```
>  -5.55  There are broadly three paid VIP products. ...
>  -6.43  The refund and change policy is strict. The date ca...  <- holds the answer
>  -7.46  Shanghai Disney Resort sells three ticket types: ...
> ```
> **The answer never moved; only the unrelated text around it did.** In a whole paragraph the one
> relevant clause is diluted by everything beside it.

> **Sentences have a cost too.** On `How do I skip the queue on the busiest rides?` the top
> sentence is `That entrance is far quieter and saves a long walk; ...`. Cut from its paragraph,
> it no longer says which entrance (the Disneytown gate for Concierge guests), and it does not
> answer the question. The sentence that does, `Premier Access has its own lane, normally right
> beside the standard queue.`, ranks seventh at -6.30. Expansion does not change this.

> **Ranking first is not the same as being confident.** Step 5 prints how far each best sentence
> leads the next one: 0.94, **0.12** and 0.40. On question 2 the correct sentence scores -10.61
> and the next one, about two-day tickets, is only 0.12 behind. Every candidate for that question
> sits near -11, so the model finds none of them clearly relevant.
> **"The reranker put the right answer first" and "the reranker is sure about it" are two
> different claims.**

### 6.4 Multi-query recall (query expansion)

**How it works**: have the model rewrite one query into **several semantically similar ones**,
retrieve with each, then **deduplicate and merge**.

This amounts to **measuring one thing with several rulers**, lowering the chance that a single
phrasing misses.

**Measured** (also in `09_rerank_and_multiquery.py`): after expanding each question into four
phrasings, what stage one can see changes as follows. The expanded set is not cut back to 8, so
the reranker reads all of it.

| Question | Paragraphs recalled | Newly reachable | Best answer after expansion |
| :--- | :---: | :---: | :--- |
| Can I move my visit to a different day after buying? | 8 → 17 or 18 | +9 or +10 | Unchanged, and already correct |
| **My father is 68. Does he pay less?** | 8 → 22 | **+14** | **Changed: from a two-day ticket sentence to the senior age rule** |
| How do I skip the queue on the busiest rides? | 8 → 16 | +8 | Unchanged, and still wrong (see 6.3) |

The phrasings come from the model, so the counts move a little from run to run.

**Only the second question was rescued, and it had failed in stage one**: the asker says `father`,
`68` and `pay less`, while the policy says `aged 65 or over` and `senior rate`. **No word that
matters is shared** (only `is`), so BM25 never handed the right paragraph to the reranker.
**Expansion exists for this case.**

> **The point**: query expansion **fixes recall, not ranking**. The first question had already
> recalled the right paragraph, and the third fails at ranking, so expansion only added
> candidates. Every extra phrasing is another model call.
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

`08_query_rewriting.py` runs these rewrites with DeepSeek on a made-up park, Riverbend Park, and
prints nine parts. It only rewrites. It does not retrieve anything, so what a rewrite does for
retrieval is reasoning here, not a measurement. Parts 1 to 5 are the five types:

| Type | Trigger | Original and rewrite |
| :--- | :--- | :--- |
| Context-dependent | Depends on the conversation before it | `Are there any other rides?`<br>`Are there any other rides in the Wildwood area at Riverbend Park besides the ranger station, the training camp, and the ice cream parlour?` |
| Comparative | Neither side was named | `Which one takes longer and is more fun?`<br>`Which one takes longer and is more fun, Wildwood or Skyline?` |
| Ambiguous reference | `both of them` points backwards | `When do both of them start?`<br>`When do both the fireworks show at Riverbend Park and the fireworks show at Harbour Park start?` |
| Multi-intent | Three questions in one turn | `How much is a ticket? Do I need to book ahead? What does parking cost?`<br>split into the three questions, returned as a JSON list |
| Rhetorical | A complaint carries the sentence | `Don't tell me I have to book a month ahead as well?`<br>`How far in advance must tickets be booked?` |

- The context-dependent question does not say which area it means. The rewrite names the area and
  the rides already listed.
- The rhetorical question is a complaint. The rewrite keeps the question inside it.
- The multi-intent type returns a list, not one query, so everything downstream has to retrieve
  each part and merge the answers. It changes the shape of the pipeline.
- The ambiguous reference only resolves when the conversation is clear. When the assistant's reply
  read `Both Riverbend Park and Harbour Park run a fireworks show.`, the nearest plural was the two
  parks, and three runs in four came back as `When do both Riverbend Park and Harbour Park start?`.
  The reply now ends with `both shows are popular`, and five runs in five named the shows.
- Each rewrite is one model call, so rewriting has a cost per question.

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

Part 6 of the script runs this prompt on five samples, one of each type:

| # | Query | Type | Conf. | Rewritten |
| :-: | :--- | :--- | :-: | :--- |
| 1 | `Are there any other rides?` | context_dependent | 0.95 | `Are there any other rides at the Wildwood area of Riverbend Park besides the ranger station, training camp, and ice cream parlour?` |
| 2 | `Which Riverbend Park area is more fun?` | comparative | 0.95 | `Which Riverbend Park area, Wildwood or Skyline, is more fun?` |
| 3 | `Are they all suitable for small children?` | ambiguous_pronoun | 0.95 | `Are the fireworks shows at Riverbend Park and Harbour Park suitable for small children?` |
| 4 | `Which restaurants are there? What do they cost?` | multi_intent | 0.95 | `Which restaurants are there, and what do they cost?` |
| 5 | `Don't tell me this is another two-hour queue?` | rhetorical | 0.90 | `Is this going to be another two-hour queue?` |

- All five types are identified correctly. Row 4 still shows the limit of one call: the schema
  declares `rewritten_query` as one string, so the two questions come back as one sentence. The
  list that part 5 of 6.5 produced has nowhere to go, which is why the multi-intent type is
  handled on its own.
- The confidence runs from 0.90 to 0.95, and the script prints that range. The model reports the
  score itself. It is not a cosine between vectors or any other measurement. Two ways to back it
  up are a cross-check against vector similarity and human review of low scores.
- An earlier run gave row 4 back unchanged at a confidence of 1.00, so a flawed rewrite can carry
  the top score.
- The rewrite is not shown to the user. It is the query that gets embedded to retrieve the top k
  chunks.

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

Parts 7 to 9 of the script handle this, for a real park, Shanghai Disneyland, so that part 8 can
run a real search. Each prompt lists the cases in the table above and asks for JSON:

| Part | Input | Output |
| :--- | :--- | :--- |
| 7. Does it need live data | the question | `need_web_search` / `search_reason` / `confidence` |
| 8. Rewrite for a search engine | the question | `rewritten_query` / `search_keywords` / `search_intent` / `suggested_sources`, then a Tavily search |
| 9. Search plan | the question | `primary_keywords` / `extended_keywords` / `search_platforms` / `time_range` |

- Part 7 asks about three questions. Two need live data and one, which themed lands the park has,
  can be answered from the knowledge base, so the run shows both answers.
- A question goes on to parts 8 and 9 only when the model says it needs a search and its
  confidence is at least 0.7.
- Part 8 sends the original question and the rewrite to Tavily and prints the top three results
  of each. Without `TAVILY_API_KEY` the searches are skipped and the rest runs as before.
- Parts 8 and 9 both start from the original question. Part 8 does not feed part 9, and the
  search plan is not searched.
- Part 9 flags a weak plan: primary keywords that are just the original sentence, or no extended
  keywords at all.

```
  Is Shanghai Disneyland open today, and how busy is it right now?
    search: True   confidence: 0.98 (gate at 0.7)
    query   : Shanghai Disneyland opening status today and current crowd level

  How much is a Shanghai Disneyland ticket next Saturday, and how far ahead must I book?
    search: True   confidence: 0.95 (gate at 0.7)
    query   : Shanghai Disneyland ticket price next Saturday advance booking requirement
    Tavily, original:
      Shanghai Disneyland Planning Guide  (travellingwithnikki.com/2018/06/...)
      Shanghai Disneyland Discount Tickets & Visiting Guide  (us.trip.com)
      One-Day General Admission Ticket Advanced Reservation ...  (shanghaidisneyresort.com)
    Tavily, rewrite:
      Shanghai Disneyland Ticket Prices 2026: ¥475 to ¥799  (disneyparknerds.com)
      Shanghai Disneyland Ticket Prices  (mickeyvisit.com)
      Book Shanghai Disney Resort Tickets: Latest Prices & ...  (us.trip.com)

  Which themed lands does Shanghai Disneyland have?
    search: False   confidence: 0.95 (gate at 0.7)
```

- For the ticket question, the original wording brought back a planning guide from 2018 first,
  and the rewrite brought back three price pages. The rewrite lost the official reservation page,
  which the original had in third place.
- A trial the same day with five results per search showed the risk more plainly. The original
  opening-hours question listed an official notice from 2020 that the park was temporarily
  closed, and the original ticket question listed an official page that no longer exists. Neither
  came back for the rewrites.
- Search results change from day to day, so these lists hold for 2026-09-27 only.
- Part 8 produces keywords for a search engine. The rewrites in 6.5 produce a full sentence for
  vector retrieval. The same word, two different outputs, so they need different prompts.
- The time window only repeats the wording of the question (`today`, `next Saturday`). The script
  does not give the model the current date, so it cannot turn these into dates.
- In this run the 0.7 gate decided nothing. The two questions that need a search score 0.98 and
  0.95, and the third is a no from the model itself.

The script calls Tavily with `POST https://api.tavily.com/search`, the key in an
`Authorization: Bearer` header, and a body of `query` and `max_results` only.

**Common search API parameters** (Tavily as the example): `query` (required), `search_depth`
(basic/advanced), `time_range` (day/week/month/year/all), `max_results`, `topic` (general/news),
`domains` / `exclude_domains` (allow and deny lists), plus three booleans off by default
(`include_images` / `include_answer` / `include_raw_html`).
**The convention: JSON output, required and optional fields separated.** Search tools of this kind
can also be exposed through the MCP protocol.

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

**Practice ten: one multi-hop question, asked of a vector index and of GraphRAG** (see
`13_graphrag_vs_vector.py`).

The corpus is a purpose-written English archive, `northgate_archive.txt` (1220 words), and the
question is `How did Mira Delaunay's way of working end up affecting Port Halbrook?`. The answer
sits in no single passage. It needs five facts that are never stated together: `Delaunay trained
Ek, Ek founded Northgate, Northgate developed Latch Encoding, Latch Encoding made the Orrery
system possible, Orrery was deployed at Port Halbrook`.

- The baseline splits the archive into 7 chunks of about 150 words, embeds them with
  gemini-embedding-001 and answers from the nearest 3 (cos 0.686 / 0.655 / 0.652) with
  gemini-3.1-flash-lite. All five chain terms appear in those chunks. A term can also sit in a
  passage that denies the link, so that count is an upper bound: the third chunk is the Broch
  collection, which the archive itself warns researchers against, and it scored well only by
  sharing names and vocabulary.
- graphrag 2.7.2 runs in its own virtual environment, because it pins numpy 1.x. Its index
  holds 28 entities, 36 relationships, 4 communities and 4 community reports, extracted from 5
  text units. `ASHFIELD` and `ASHFIELD POLYTECHNIC`, `NORTHGATE` and `NORTHGATE LAB` stay four
  entities: extraction merges only entities with the same name and type.
- Global search answers from community summaries and cites them as `Reports`. Local search starts
  from the entities in the question and also cites `Entities`, `Relationships` and `Sources` (the
  text units). The script checks that every cited id exists in its table; all of them do. That
  shows the rows exist, not that they support the sentence they are attached to.

Step 8 counts, in each of the three answers, how many of the question's 7 entities it names:

```
  baseline  names 4 of the 7 entities the question is about, plus 1 beyond it
    not named: Tomas Ek, Coastal Institute, Northgate Lab
    beyond the chain: LATCH-V
  global    names 7 of the 7 entities the question is about, plus 1 beyond it
    beyond the chain: PRIYA RAMAN
  local     names 7 of the 7 entities the question is about, plus 3 beyond it
    beyond the chain: JULIAN ADEYEMI, LATCH-V, VELLUM INSTRUMENTS
```

The retrieved text held all five chain terms, yet the baseline answer names only 4 of the 7
entities: it reaches Port Halbrook through the later Latch-V system and leaves out Ek, the
Coastal Institute and Northgate Lab. Retrieving the passages is not the same as joining them.
Both graph answers name all seven, because the joins were computed when the index was built.

The comparison has limits. It is one question. Naming an entity is not stating the link between
two of them. And the corpus is small: the top 3 of 7 chunks is 43% of it, so the vector route
still retrieves every term. The two approaches separate more clearly when the links sit far
apart, which this corpus is too small to show.

The cost is on the graph side. The baseline made three API calls: embeddings for the chunks, an
embedding for the question, and the answer. The global search took 27 to 87 seconds across runs
and the local search about 19. Before any question, the graph needs a full indexing pass that
grows with the corpus, not with the number of questions. It pays off only where the connections
matter more than the passages.

---

## 7. Operating a Knowledge Base and Keeping It Healthy

Everything so far has been about **using** a knowledge base. This section is about **maintaining**
one. A knowledge base is not a static asset that is finished once built; it has a full life cycle:
**question generation → knowledge extraction → health checks → version management**, closing the
loop.

### 7.1 Scenario one: generated questions and retrieval optimisation

Users ask questions, and the store holds statements, so the two often share few words. The fix
is to generate questions for every chunk in advance and match question against question. This
is 6.7's Doc2Query made concrete.

The knowledge base is six chunks about an invented theme park: basics, prices, opening hours,
transport, rides and park rules. Three visitor questions each map to one chunk. Two of them
share no content word with any chunk. The third uses the chunk's own words, as a control.

- The basic set is five questions for one chunk, with the fields `question`, `question_type`
  and `difficulty`.
- The wider set is eight questions, varied by type (adding hypothetical and inferential),
  wording, difficulty and perspective. It adds `perspective`, `is_answerable` and `answer`.
- `is_answerable` and `answer` are the quality gate. The model answers its own question from
  the chunk, and a question the chunk cannot answer is dropped. Indexed, such a question would
  send a visitor to a chunk without the answer. On the first chunk, 3 of the 8 questions were
  dropped.
- For the whole knowledge base the script generates the wider set for every chunk and keeps
  only the answerable questions: 39 of 48 in the run below. It then builds two BM25 indexes,
  one on the six chunks and one on the kept questions.

```
  prose retrieval accuracy   :  33.3%  (1/3)
  question retrieval accuracy:  33.3%  (1/3)
```

| # | Query | Prose score | Question score | Prose | Question | Matched generated question |
| :-: | :--- | :-: | :-: | :-: | :-: | :--- |
| 1 | `Am I allowed to take a picnic in?` | 0.000 | 3.624 | ✗ | ✗ | `How long does it take to reach the park by taxi?` (kb_004) |
| 2 | `What time should I show up to avoid the crowds?` | 0.000 | 8.979 | ✗ | ✓ | `If you wanted to avoid crowds, when would be the best time to go?` |
| 3 | `How much does it cost to park a car?` | 1.262 | 2.499 | ✓ | ✗ | `How much does a weekday adult ticket cost?` (kb_002) |

- A 0.000 in the prose column means nothing matched. Queries 1 and 2 share no content word with
  any chunk, so BM25 scores every chunk at zero and `max()` returns the first one, `kb_001`.
- The question index never retrieves prose. It finds the closest generated question and returns
  the chunk that produced it, so a hit depends on how close the visitor came to one of those
  phrasings.
- The generated questions bring words of their own, and those work in both directions. Query 2
  was won by `avoid crowds`. Query 1 was lost to a taxi question through the word `take`, and
  the control query was lost to a ticket question through `cost`.
- The scores of the two indexes are not compared. They come from two different BM25 corpora
  (6 long chunks against 39 short questions), so the numbers are on different scales.

The result changes from run to run, because the generated questions change. Four runs with the
same prompts gave the question index 3/3, 2/3, 1/3 and 1/3; the prose index is 1/3 every time.
With three test queries, one query flipping moves the accuracy by 33 points. What holds across
runs is the mechanism: the technique helps where the visitor's words and the document's differ,
and the extra vocabulary can pull a query to the wrong chunk just as easily. Count the net
change, not the wins alone.

The questions can live in the same chunk as the source text, `chunk = {prose, questions[]}`, so
no extra storage system is needed.

See `10_kb_question_generation.py`.

### 7.2 Scenario two: distilling knowledge out of conversations

A support product produces conversations every day. `11_kb_curation.py` uses DeepSeek to turn
three visitor conversations about an invented theme park into knowledge base entries, in three
steps: extract, filter, merge.

- Extract. The model pulls typed points out of each conversation: `fact`, `need`, `question`,
  `process` and `caution`. Each point also carries `confidence`, `source`, `keywords` and
  `category`. In three runs every point came back with confidence 1.00, so the script prints
  that the column does not tell the points apart.
- Filter. `need` and `question` points record what a visitor wanted, not what is true. Left in
  the index, a later question about ticket prices could retrieve "the visitor wanted to know the
  ticket price", which answers nothing. They are dropped: 16 of 39 points in the run below.
- Merge. The remaining points are grouped by type, and each group becomes one entry through one
  model call. 23 points became 3 entries.

```
  fact: 13 points -> 1 (confidence 1.00)
    categories: crowd levels, opening hours, pricing, recommended rides, rules, services
    Riverbend Park opens daily from 08:00 to 20:00. Adult tickets cost 399 on weekdays ...
```

Grouping by type puts different topics into one entry: the fact entry holds opening hours,
prices, rides and park rules together. A real base would group by topic. The `category` field
the model returns is too loose for that here, with 11 to 14 different values for 23 points
across three runs. Nothing later in the script uses the merged entries.

### 7.3 Scenario three: health checks

The audit half checks a knowledge base for three problems, each with its own prompt and its own
extra input:

| Check | Extra input | Finds |
| :--- | :--- | :--- |
| Coverage | the test questions | questions no entry answers |
| Freshness | today's date | entries that have gone out of date |
| Consistency | none | entries that contradict each other |

- Coverage needs the questions because a gap only exists relative to something someone asks.
  Its prompt also says that a contradicted or dated answer still counts as present, so the
  other two checks own those.
- Freshness needs today's date because the model has no clock.

The audit does not run on the merged entries. It runs on a separate six-entry base with three
planted defects, which serve as the answer key: no entry about pets, a winter festival that
ended in January 2024, and `kb_002` and `kb_005` giving the parking charge as 100 and 150. The
report checks each check against its planted defect and counts everything else it flagged.

```
--- 5. Coverage ---
  score 0.67, 1 gap
  - Can I bring my dog?                              [medium] Pet/dog admission policy
--- 6. Freshness ---
  score 0.67, 2 stale entries
  - kb_002   [high] parking fee superseded by kb_005 (100 -> 150 per day)
  - kb_004   [high] event period ended 5 January 2024 (past event)
--- 7. Consistency ---
  score 0.83, 1 conflict
  - kb_002, kb_005     [medium] parking_fee
--- 8. Report ---
  planted defects:
    coverage     found   the dog question, plus 0 other findings
    freshness    found   kb_004, plus 1 other finding
    consistency  found   kb_002 and kb_005, plus 0 other findings
  Freshness flagged kb_002, the conflicting parking pair.
```

All three planted defects were found in each of three runs. The freshness check can tell that
the festival has ended only because today's date is in the prompt.

The freshness prompt used to list "prices likely to have moved" among the signs of staleness.
With it, the check flagged 4 of 6 entries, including kb_003, a ticket price with no date at
all. With that line removed it flagged 2 in each of three runs: kb_004, and kb_002 as
superseded by kb_005. That second finding is the parking conflict again, which belongs to the
consistency check. The three checks take different inputs and their findings still overlap.

The scores are the model's own numbers and nothing here measures them. The findings can be
checked against the planted defects. Read the findings and treat the scores as a rough
impression.

The model finds conflicts, and a person decides which side is right.

See `11_kb_curation.py`.

### 7.4 Scenario four: version management and performance comparison

`12_kb_version_management.py` versions a knowledge base, benchmarks retrieval on each version and
runs a regression check before release. Embeddings do the work here, not a chat model. Version 1
of an invented park's base has 3 entries; version 2 adds 2 entries and extends the other 3.

- Fingerprints. Each version gets an MD5 hash of its sorted ids and texts, so one edited
  character changes it. v1.0 has 3 entries and hash `1d0aec844c6e`, v2.0 has 5 entries and
  `61d5d51301c1`.
- Diff. Added and removed ids come from set operations, modified ones from exact text
  comparison, with no model: 2 added, 0 removed, 3 modified.
- Indexing. Both versions are embedded with gemini-embedding-001 at 1024 dimensions into
  `IndexFlatIP`. The model's full width is 3072 dimensions, where its vectors have length 1.
  Asked for 1024, it returns the first 1024 of those values (2.5 shows the leading values
  match), so the length drops: to about 0.62 on average here, and by a different amount for
  each text. Rescaling divides each vector by its own length. It keeps all 1024 dimensions and
  makes the inner product a cosine, so no entry ranks higher only because its vector is longer.
- Scoring. Each of five test questions names a string the answer must contain. A question
  counts as answered when that string appears in the top 3 entries, and the table shows the
  rank of the first entry holding it.

```
  query                                            v1.0     v2.0
  --------------------------------------------------------------
  Where is the park?                             rank 2   rank 3
  How much is an adult ticket on a Tuesday?      rank 1   rank 1
  What time does it close?                       rank 1   rank 1
  How do I get there by public transport?          MISS   rank 1
  Which rides should I not miss?                   MISS   rank 1
  --------------------------------------------------------------
  accuracy                                          60%     100%
```

- Version 1 has only 3 entries, so the top 3 is its whole base for every question. A MISS there
  means the answer is missing, never that retrieval failed.
- The accuracy hides a weak retrieval. "Where is the park?" counts as answered in both versions,
  but the location entry comes second in version 1 and third in version 2. With the top 1 it
  would fail in both, and with the top 2 in version 2.
- The gap from 60% to 100% comes entirely from content version 1 lacked. The script checks each
  gained question: `line 11` and `launch coaster` are in no entry of version 1. Version 2 does not
  search better; it has more to find.
- The mean search time was 0.007 ms and 0.006 ms. An exact search over 3 or 5 vectors costs the
  same, so the script reports no measurable change instead of a difference.
- The regression check asks whether every question version 1 answered still passes: 3 of 3 do.
  The two fixed questions are the gains above. Five cases can only catch the breakages they
  cover, so the claim is "no regressions on this test set".
- A substring test shows that the answer's text was retrieved, not that a reply built from it
  would be right.

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
| `01_tfidf_hotel_recommender.py` | TF-IDF and n-gram features, cosine similarity, recommendations for two hotels (1.5 / 1.6) |
| `02_word2vec_similarity.py` | Segmentation, two Word2Vec models and persistence, the 长老 analogy, text8 for contrast (1.8) |
| `03_embedding_faiss_metadata.py` | Embedding calls, Matryoshka dimensions, FAISS with document ids and metadata, squared L2, persistence (2.5 / 3.2 / 3.3) |
| `04_embedding_models_compare.py` | Hand-written pooling vs a wrapper for CLS, mean and last-token pooling, right and left padding (2.6) |
| `05_chunking_strategies.py` | Five chunking strategies compared side by side (5.1 / 5.2) |
| `06_chatpdf_langchain_faiss.py` | End-to-end QA with LangChain and FAISS, per-character page citation, rechecking TOP_K (4.5) |
| `07_disney_multimodal_rag.py` | Multimodal RAG without a framework, two indexes, CLIP cross-modal search, OCR, vision description (4.6) |
| `08_query_rewriting.py` | Five rewrite types, single-prompt intent detection and confidence (6.5 / 6.6) |
| `09_rerank_and_multiquery.py` | Two-stage retrieval (wide recall then reranking), multi-query expansion (6.3 / 6.4) |
| `10_kb_question_generation.py` | Doc2Query in practice, dual BM25 indexes, retrieval evaluation (6.7 / 7.1) |
| `11_kb_curation.py` | Conversation distillation (extract / filter / merge) and health auditing (7.2 / 7.3) |
| `12_kb_version_management.py` | Version hashes, set-operation diffing, A/B and regression testing (7.4) |
| `13_graphrag_vs_vector.py` | One multi-hop question asked of a vector index and of a knowledge graph, with cost accounting (6.9) |
