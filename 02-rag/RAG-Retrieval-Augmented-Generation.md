# Retrieval-augmented generation

Scripts 01 to 04 turn text into vectors: word counts, word vectors, sentence embeddings
and a FAISS index. Scripts 05 to 09 build a retrieval pipeline and improve it with
chunking, page citations, images, query rewriting and reranking. Scripts 10 to 13 keep
a knowledge base healthy and compare vector retrieval with GraphRAG. This document
explains what each script does and the ideas it relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_tfidf_hotel_recommender.py` | TF-IDF vectors and cosine similarity, used to recommend similar hotels |
| 02 | `02_word2vec_similarity.py` | Word2Vec trained on a Chinese novel, with an English corpus for contrast |
| 03 | `03_embedding_faiss_metadata.py` | Embeddings, Matryoshka dimensions, and a FAISS index with ids and metadata |
| 04 | `04_embedding_models_compare.py` | CLS, mean and last-token pooling, rebuilt by hand and checked against a wrapper |
| 05 | `05_chunking_strategies.py` | Five chunking strategies compared by chunk size |
| 06 | `06_chatpdf_langchain_faiss.py` | Question answering over a PDF with LangChain and FAISS, with page citations |
| 07 | `07_disney_multimodal_rag.py` | Multimodal RAG without a framework: text index, CLIP image index, OCR, vision model |
| 08 | `08_query_rewriting.py` | Query rewriting: five types, one-call classification, rewriting for web search |
| 09 | `09_rerank_and_multiquery.py` | Two-stage retrieval (BM25 recall, cross-encoder reranking) and query expansion |
| 10 | `10_kb_question_generation.py` | Doc2Query: visitor questions matched against generated questions |
| 11 | `11_kb_curation.py` | Knowledge extraction from conversations, and an audit of a knowledge base |
| 12 | `12_kb_version_management.py` | Knowledge base versions, a retrieval benchmark and a regression check |
| 13 | `13_graphrag_vs_vector.py` | GraphRAG against a vector baseline on one multi-hop question |

## Shared setup (scripts 03 and 05 to 13)

*   Every key lives in `.env` at the repository root, which `.gitignore` excludes. The
    scripts call every provider through the `openai` package and its
    OpenAI-compatible protocol, as in module 01.
*   Chat models: DeepSeek (`deepseek-chat`) in 05 and 08 to 11, and
    `gemini-3.1-flash-lite` in 06, 07 and 13. Embeddings: `gemini-embedding-001` in
    03, 06, 07, 12 and 13. Scripts 06 and 07 take the first key set, Gemini before
    OpenAI.
*   gemini-embedding-001 returns 3072 values of unit length. Asked for fewer
    dimensions, it returns the leading values of the same vector, which is shorter
    than 1 and shorter by a different amount for each text. Scripts 03, 07 and 12
    divide every vector by its own length before indexing. The dimensions stay the
    same, and the distance or inner product then depends on direction alone.

## Script 01: TF-IDF and cosine similarity

*   The data is 152 Seattle hotels with `name`, `address` and `desc`. Every
    description becomes a TF-IDF vector, and hotels whose vectors point in similar
    directions count as similar. No neural model is involved.
*   Cosine similarity is `(A·B) / (||A|| × ||B||)`. It compares directions, so a long
    description does not score higher just for holding more words.
*   Part 3 counts the 20 most common three-word phrases. `pike place market` comes
    first, whether or not it tells hotels apart. Raw counts cannot say which words
    matter, which is what TF-IDF adds.
*   `TfidfVectorizer` weights each term count by `idf = ln((1 + n) / (1 + df)) + 1`,
    where n is the number of descriptions and df the number containing the term. A
    term most hotels use counts less. The script uses terms of one to three words:
    3348 terms, a `152 × 3348` matrix, nearly all zeros.
*   Every vector is scaled to length 1, so the dot product (`linear_kernel`) is the
    cosine. The result is a `152 × 152` matrix with 1 on the diagonal.
*   Recommendations sort one hotel's row and skip the first entry, the hotel itself
    (`iloc[1:11]`).

| Query hotel | Top 10 |
| :--- | :--- |
| Hilton Seattle Airport & Conference Center | Mostly airport hotels, first `Embassy Suites ... Seattle Tacoma International Airport` at 0.2266. Fourth is `Four Points by Sheraton Downtown Seattle Center` and ninth `Hotel Hotel`, neither near the airport. |
| The Bacon Mansion Bed and Breakfast | Five B&Bs and inns at the top, first at 0.2801 |

*   Scores of 0.1 to 0.3 are normal for sparse vectors: two hotels only need to word
    their descriptions differently for the cosine to drop. The order carries the
    result, not the value.

## Script 02: Word2Vec

*   Chinese has no spaces between words, so jieba first cuts the novel (Journey to the
    West, stored as GB18030) into words.
*   Word2Vec learns one vector per word by predicting which words appear near each
    other, so words used in similar contexts get similar vectors. Multiplying a
    one-hot input by the weight matrix selects one row, and that row is the word's
    vector.

| Model | Vector size | Window | Min count | Vocabulary | Threads |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline | 100 | 3 | 1 | 46005 | 1 |
| Second | 128 | 5 | 5 | 7735 | all cores |

*   The analogy asks: Sun Wukong is to Pilgrim Sun (his other name) as Tang Seng is to
    what? The baseline answers 长老 (elder, 0.9805), the way other characters address
    Tang Seng.
*   The second model is saved and reloaded, and the reloaded model gives the same
    similarity. It trains on several threads, so its numbers move between runs:
    `孙悟空 vs 猪八戒` from 0.9265 to 0.9369, and its analogy has put 长老 or 菩萨
    first. The baseline runs on one thread and repeats exactly.
*   Every name pair scores above 0.8 in both models, even Sun Wukong and "monster"
    (0.9602). One novel is small and repetitive, so the names share the same contexts
    and their vectors crowd together. High scores here describe the corpus, not the
    model.
*   Part 7 trains the same recipe on text8, 17 million words of cleaned Wikipedia.
    `king - man + woman` gives queen (0.7226), `paris + italy - france` gives venice
    (0.7671), and `king vs banana` scores 0.0455 against 0.7264 for `king vs queen`.
    A larger, more varied corpus separates words that one novel cannot.

## Script 03: Embeddings and a FAISS index with metadata

Four Disney FAQ entries (two about ticket refunds, one about the annual pass, one about
a ride closed for maintenance) and the question "I want to understand the refund
process for Disney tickets" are embedded with `gemini-embedding-001` at 768 dimensions.

*   Matryoshka representation learning trains the model so that the leading values of
    its vector also work as a shorter embedding. The same question at 3072, 1536 and
    768 dimensions starts with the same values (`-0.014099, -0.0218, -0.000503`).
    Parts 1 to 3 only look at the embeddings. The search from part 4 on embeds
    everything again.
*   Part 3 runs the whole search at all three sizes. The top three are the same each
    time, and the distances move a little (doc3: 0.3139, 0.3354, 0.3142).
*   `IndexFlatL2` compares the query with every vector, so the search is exact. It
    returns squared L2 distance, where smaller is closer. For unit vectors, squared L2
    equals 2 minus 2 times the cosine, so both rank the same way: doc3 at 0.3142 is a
    cosine of 0.84, doc2 at 0.8528 a cosine of 0.57. The two refund entries come
    first.
*   FAISS alone returns positions. `IndexIDMap` stores an id with each vector (doc3 as
    3), so deleting or reordering entries cannot change which document a hit points
    to. FAISS stores only vectors and ids. The text and metadata live in a dict under
    the same ids.
*   The index is written to disk and read back, and the search gives the same
    ranking. The metadata dict is not part of the index and has to be saved with it.

## Script 04: Pooling (CLS, mean and last token)

A transformer returns one vector per token, and pooling turns them into one vector per
text. Each model is trained with one pooling, and the wrong one raises no error. The
script scores two questions (a ticket refund, annual pass perks) against two passages
that answer them, with three models:

| Model | Structure | Pooling |
| :--- | :--- | :--- |
| BAAI/bge-small-en-v1.5 | encoder (BERT-style) | CLS |
| GTE small | encoder (BERT-style) | mean |
| Qwen3-Embedding-0.6B | decoder | last token |

*   CLS pooling takes the first token, mean pooling averages the real tokens (the
    attention mask leaves out padding), and last-token pooling takes the final real
    token.
*   An encoder attends in both directions, so every token has seen the whole text. A
    decoder attends only to earlier tokens, so only the last one has seen it all.
*   Through the SentenceTransformer wrapper, each question scores highest against its
    own passage in all three models.
*   Part 4 writes tokenising, pooling and normalising out by hand and compares the
    scores with the wrapper's. Qwen3 runs with right and with left padding, because
    the last real token sits in a different place.

| Hand-built | Largest difference from the wrapper |
| :--- | ---: |
| bge, cls, right padding | 0.000000 |
| GTE, mean, right padding | 0.000505 |
| Qwen3, last, right padding | 0.002961 |
| Qwen3, last, left padding | 0.002961 |

Part 5 pools bge with mean instead of CLS:

| | Deviation from the wrapper | Margin (right passage over wrong, mean of both questions) |
| :--- | ---: | ---: |
| cls (right) | 0.000000 | +0.2198 |
| mean (wrong) | 0.069623 | +0.2548 |

*   The wrong pooling still ranks each passage first, and its margin is even larger.
    Only the deviation from the wrapper shows the mistake. The margin depends on the
    samples and on each pooling's score scale, so it cannot judge correctness. Judging
    retrieval quality takes hundreds of labelled pairs and metrics such as Recall@K,
    MRR or NDCG.

## Script 05: Chunking strategies

The document is a theme park ticket guide in three paragraphs (ticket types, buying a
ticket, discounts), 1299 characters long, or 1366 in the copy with headings. The target
chunk size is 800. The script measures chunk sizes only. It does not test retrieval.

| Strategy | What the code does | What the run shows |
| :--- | :--- | :--- |
| 1 Fixed length | Cuts every 800 characters, moves back to the last sentence end within 200 characters, and starts the next chunk 150 characters earlier | The second chunk starts at `resellers also sell them`, in the middle of a sentence |
| 2 Sentence packing | Packs whole sentences into chunks of up to 800 characters | No sentence is cut, but the first chunk runs from paragraph 1 into the middle of paragraph 2 |
| 3 LLM | A chat model picks the break points and replies with JSON | DeepSeek returned the three paragraphs unchanged: 414, 428 and 453 characters |
| 4 Hierarchical | Starts a new chunk at every heading (on the copy with headings) | The title `# Ticket Guide` becomes a chunk of 14 characters |
| 5 Sliding window | Moves an 800-character window forward 450 characters at a time | Neighbouring chunks share 350 characters, chunks start mid-word (`ficial channels`), and 1999 characters are indexed for 1299 |

```
  strategy                chunks     avg     min     max   spread
  1. Fixed length              2     724     684     765       81
  2. Sentence packing          2     648     613     683       70
  3. LLM                       3     432     414     453       39
  4. Hierarchical              4     339      14     466      452
  5. Sliding window            3     666     399     800      401
```

*   Spread is the longest chunk minus the shortest.
*   Strategy 2 only counts characters. No embedding is involved, so it is not semantic
    chunking, which cuts where the similarity of neighbouring sentences drops.
*   Strategy 4 only splits at headings. A chunk does not carry its parent heading, so
    the `## Discounts` chunk does not say it belongs to the ticket guide.
*   The LLM prompt asks for chunks of at most 800 characters that are semantically
    complete, returned as `{"chunks": [...]}`. A model can reword text while splitting
    it, so the run counts how many chunks appear word for word in the guide: 3 of 3.
*   The LLM has the smallest spread only because it split at the blank lines, which
    gives the same three chunks without an API call. Hierarchical has the largest
    because of the title chunk; without it the spread would be 36.
*   Without an API key, or when the call fails, part 2 runs in place of part 3 and the
    row reads `3. LLM (fell back)`, so the table cannot pass off part 2's numbers as
    the model's.

## Script 06: Question answering over a PDF with page citations

The PDF is nine pages of a bank's rules for assessing retail account managers. One
question asks how many points a customer complaint costs, the other when the yearly
appointment review opens.

*   pypdf extracts the text page by page. `RecursiveCharacterTextSplitter` splits it
    with `chunk_size=1000` and `chunk_overlap=200`, trying `["\n\n", "\n", ".", " ",
    ""]` in order. The chunks are embedded and stored in FAISS, saved to disk with
    their page map, and loaded back.
*   The page citations are the script's own work. The splitter cuts the text wherever
    it likes, so a page number recorded per line cannot be matched to chunk i. The
    script records a page number for every character, finds each chunk in the text
    with `str.find`, and gives the chunk the page most of its characters came from.
*   Pages are joined with a single newline, which keeps each footer (`- 5 -`) off the
    next page's first line. The extracted text has no `\n\n`, so joining with `\n\n`
    would have made the page breaks the splitter's first choice and stopped chunks from
    crossing pages. With `\n`, 10 of the 15 chunks span two pages.
*   Each question gets the 4 nearest chunks in one prompt and one model call (the
    "stuff" approach). LangChain 1.x no longer has `langchain.chains`, so the chain is
    written with LCEL. `OpenAIEmbeddings` posts token arrays by default, which Gemini
    rejects with a 501, so `check_embedding_ctx_length=False` makes it send plain
    strings.

| Measure | Result |
| :--- | :--- |
| Text and chunks | 11968 characters from 9 pages, 15 chunks, all 15 with a page number |
| Complaint question | `2 points are deducted for each customer complaint.`, pages [5, 6, 3] |
| Review question | `January of each year is the application window…`, pages [6, 4, 8, 7] |

*   k has to be small against the size of the store. With k=10 and a store of 5 chunks,
    every chunk comes back and the cited pages filter nothing.

Part 5 has the model rewrite each question three ways, retrieves under all four
phrasings and merges the chunks, removing duplicates. Each question carries the page
that holds its answer (5 and 6, checked against the PDF):

```
    single phrasing : 3 pages [5, 6, 3], answer page 5 at position 1
    4 phrasings     : 3 pages [5, 6, 3], answer page 5 at position 1
    new pages       : none

    single phrasing : 4 pages [6, 4, 8, 7], answer page 6 at position 1
    4 phrasings     : 6 pages [6, 4, 8, 7, 3, 2], answer page 6 at position 1
    new pages       : [3, 2], not needed, the answer page was already found
```

*   Both answer pages are first with the question alone, so the rephrasings add
    nothing the answer needs. Rephrasing helps when the asker's words miss the
    document's vocabulary, and these two questions already use it.

## Script 07: Multimodal RAG without a framework

The knowledge base is four Word files (ticket rules, senior tickets, a visit guide,
hotel and membership services) and two event posters. The script uses no LangChain.

*   Each Word file is walked element by element. A paragraph gets the heading above it
    as a prefix, and a table becomes Markdown (header row, separator row, one row per
    table row). The four files give 26 text blocks. Indexed alone, a five-word heading
    such as `Ticket Rules` ranked near the top for any ticket question and pushed the
    refund clause out of the top k.
*   Text is embedded with `gemini-embedding-001` at 1024 dimensions and normalised.
*   The posters are embedded with CLIP (`openai/clip-vit-base-patch32`, 512
    dimensions). CLIP trains an image encoder and a text encoder together so that a
    picture and its caption get similar directions, so a question encoded by CLIP's
    text encoder can find a picture. The 1024-dimension text vectors and the
    512-dimension CLIP text vectors are both text vectors, but they are in different
    spaces.
*   CLIP is trained on cosine similarity, so only the direction means anything, and
    its vectors are normalised too. Unnormalised, the posters had lengths of 8.56 and
    9.59, and a query describing the Halloween artwork (`a purple night sky with a
    yellow moon and bats`) picked the Lunar New Year poster on L2 (108.12 against
    115.06) although its cosine favoured Halloween (0.222 against 0.155). Normalised,
    it picks Halloween (1.5566 against 1.6899).

An image can reach a text-only prompt in three ways, and the script shows all three:

| Route | What it gives | Where it goes |
| :--- | :--- | :--- |
| CLIP | A vector, so a text question can find the picture | The image index |
| OCR (`rapidocr-onnxruntime`, pip only) | The words printed on the picture | The image's record, and the prompt when the image is found |
| Vision model | A description of the picture itself | Printed for comparison, not used in the answers |

*   Every question searches the text index for the 3 nearest blocks. The image index
    is searched only when the question contains a word such as `poster`, `picture` or
    `look like`, and then it returns the single nearest image.
*   The two sets of hits are not merged by distance. Both are squared L2 between unit
    vectors, so the numbers look alike, but the models spread their scores
    differently: a strong CLIP match sits near cosine 0.3.
*   The prompt labels each passage `[Source N: file]` and tells the model to answer
    only from them and to say so when the answer is not there.

| Question | Text hits (squared L2) | Image hit |
| :--- | :--- | :--- |
| Refund process | ticket rules blocks 3, 9, 11 (0.5385 to 0.6558) | not searched |
| What the Halloween poster looks like | blocks 11, 20, 25 (0.9827 to 1.0563): rain, senior visits, concierge service | `02_halloween.jpeg`, 1.3663 (cosine 0.317) |
| Annual pass discounts | blocks 4, 13, 24 (0.6377 to 0.7712) | not searched |

*   The poster question finds no relevant text, because the text files say nothing
    about Halloween. Its answer (dates, night event, separate ticket, parade) comes
    from the poster's OCR text, and it cannot say what the poster looks like.
*   Part 3 asks the vision model to describe the picture and its colours. It answers
    with what OCR cannot read:

    ```
    This poster features a dark, gradient purple night sky illuminated by a bright yellow full moon
    and accented with simple, dark geometric bats. ...
    ```

## Script 08: Query rewriting

A visitor asks in conversational, context-dependent language, and the knowledge base
holds plain statements. Query rewriting turns the question into one a retriever can
use. DeepSeek does every rewrite, and all prompts share one frame: instruction,
conversation history, current question. Parts 1 to 6 use a made-up park, Riverbend
Park. They only rewrite and retrieve nothing, so what a rewrite does for retrieval is
not measured here.

| Type | Trigger | Original and rewrite |
| :--- | :--- | :--- |
| Context-dependent | Depends on the conversation before it | `Are there any other rides?`<br>`Are there any other rides in the Wildwood area at Riverbend Park besides the ranger station, the training camp, and the ice cream parlour?` |
| Comparative | Neither side was named | `Which one takes longer and is more fun?`<br>`Which one takes longer and is more fun, Wildwood or Skyline?` |
| Ambiguous reference | `both of them` points backwards | `When do both of them start?`<br>`When do both the fireworks show at Riverbend Park and the fireworks show at Harbour Park start?` |
| Multi-intent | Three questions in one turn | `How much is a ticket? Do I need to book ahead? What does parking cost?`<br>split into the three questions, returned as a JSON list |
| Rhetorical | A complaint carries the question | `Don't tell me I have to book a month ahead as well?`<br>`How far in advance must tickets be booked?` |

*   The multi-intent type returns a list, not one query, so everything downstream has
    to retrieve each part and merge the answers.
*   The ambiguous reference resolves only when the conversation is clear. When the
    assistant's reply read `Both Riverbend Park and Harbour Park run a fireworks
    show.`, the nearest plural was the two parks, and three runs in four named the
    parks. The reply now ends with `both shows are popular`, and five runs in five
    named the shows.

Part 6 classifies and rewrites in one call. The prompt defines the five types, sets a
priority when two match, and asks for `{"query_type", "rewritten_query", "confidence"}`:

| # | Query | Type | Conf. | Rewritten |
| :-: | :--- | :--- | :-: | :--- |
| 1 | `Are there any other rides?` | context_dependent | 0.95 | `Are there any other rides at the Wildwood area of Riverbend Park besides the ranger station, training camp, and ice cream parlour?` |
| 2 | `Which Riverbend Park area is more fun?` | comparative | 0.95 | `Which Riverbend Park area, Wildwood or Skyline, is more fun?` |
| 3 | `Are they all suitable for small children?` | ambiguous_pronoun | 0.95 | `Are the fireworks shows at Riverbend Park and Harbour Park suitable for small children?` |
| 4 | `Which restaurants are there? What do they cost?` | multi_intent | 0.95 | `Which restaurants are there, and what do they cost?` |
| 5 | `Don't tell me this is another two-hour queue?` | rhetorical | 0.90 | `Is this going to be another two-hour queue?` |

*   All five types are identified correctly. Row 4 shows the limit of one call: the
    schema declares `rewritten_query` as one string, so the two questions come back
    as one sentence.
*   The confidence is a score the model reports about itself, not a cosine or any
    other measurement. An earlier run returned row 4 unchanged at 1.00.

Parts 7 to 9 ask about a real park, Shanghai Disneyland, so that part 8 can run a real
search. The model has no clock and no live data, so questions about today, prices,
events, weather, transport, bookings or queues need a web search. Each prompt lists
these cases and asks for JSON:

| Part | Output |
| :--- | :--- |
| 7. Does it need live data | `need_web_search`, `search_reason`, `confidence` |
| 8. Rewrite for a search engine | `rewritten_query`, `search_keywords`, `search_intent`, `suggested_sources`, then a Tavily search |
| 9. Search plan | `primary_keywords`, `extended_keywords`, `search_platforms`, `time_range` |

*   A question goes on to parts 8 and 9 only when the model says it needs a search and
    its confidence is at least 0.7. Parts 8 and 9 both start from the original
    question, and the search plan is not searched.
*   Part 8 sends the original question and the rewrite to Tavily (`POST
    https://api.tavily.com/search`, `query` and `max_results`) and prints the top three
    results of each. Without `TAVILY_API_KEY` the searches are skipped.

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

*   For the ticket question, the original wording brought back a planning guide from
    2018 first, and the rewrite brought back three price pages. The rewrite lost the
    official reservation page, which the original had third. Search results change
    daily, so these lists hold for 2026-09-27 only.
*   Part 8 produces keywords for a search engine, and parts 1 to 6 produce a full
    sentence for vector retrieval, so the two need different prompts.
*   The time window only repeats the question's wording (`today`, `next Saturday`).
    The script does not give the model the current date, so it cannot turn these into
    dates.
*   In this run the 0.7 gate decided nothing: the two questions that need a search
    score 0.98 and 0.95, and the third is a no from the model itself.

## Script 09: Reranking and query expansion

The knowledge base is 26 paragraph chunks (108 sentences) from 4 Word files of Disney
ticket rules. Each paragraph carries its file's heading, so no heading is a chunk of its
own.

*   Stage one, BM25, scores every paragraph by the words it shares with the question
    and keeps 8. BM25 is TF-IDF with two fixes: repeats of a word soon stop adding
    score, and long paragraphs lose the edge of simply holding more words.
*   Stage two, a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), reads the
    question together with each sentence of those 8 paragraphs and keeps the best 3.
    An embedding model is a bi-encoder: it encodes question and document separately,
    so document vectors can be computed ahead. A cross-encoder reads the pair together
    and judges meaning more closely, but costs one model pass per pair, so it only
    runs on the few candidates recall leaves.
*   The cross-encoder returns raw logits with no fixed range. For question 1, the
    correct sentence scores -6.11 and an unrelated sentence about the Eiffel Tower
    -11.36. Zero is not a threshold. Only the order within one question means
    anything.

Part 3 scores question 1's answer as different units:

```
 -8.60  heading + whole paragraph   (506 chars)
 -6.43  whole paragraph             (469 chars)
 -5.82  the answering sentence      (113 chars)
```

Scored whole, the 8 recalled paragraphs put a wrong one first:

```
 -5.55  There are broadly three paid VIP products. ...
 -6.43  The refund and change policy is strict. The date ca...  <- holds the answer
 -7.46  Shanghai Disney Resort sells three ticket types: ...
```

*   The answer never changes; only the unrelated text around it does. In a whole
    paragraph the one relevant clause is diluted.
*   Sentences have a cost too. On `How do I skip the queue on the busiest rides?` the
    top sentence is `That entrance is far quieter and saves a long walk; ...`. Cut from
    its paragraph, it no longer says which entrance, and it does not answer the
    question. The sentence that does, `Premier Access has its own lane, normally right
    beside the standard queue.`, ranks seventh at -6.30.
*   Part 5 prints how far each best sentence leads the next: 0.94, 0.12 and 0.40. On
    question 2 every candidate sits near -11, so ranking first is not the same as the
    model finding it relevant.

Part 5 also expands each question into four phrasings with DeepSeek, recalls for all
of them, and reranks the union, which is not cut back to 8:

| Question | Paragraphs recalled | Newly reachable | Best answer after expansion |
| :--- | :---: | :---: | :--- |
| Can I move my visit to a different day after buying? | 8 to 17 or 18 | +9 or +10 | Unchanged, and already correct |
| My father is 68. Does he pay less? | 8 to 22 | +14 | Changed: from a two-day ticket sentence to the senior age rule |
| How do I skip the queue on the busiest rides? | 8 to 16 | +8 | Unchanged, and still wrong |

*   The phrasings come from the model, so the counts move a little between runs.
*   Only question 2 was rescued, and it had failed in stage one. The asker says
    `father`, `68` and `pay less`, and the policy says `aged 65 or over` and `senior
    rate`, so BM25 never handed the right paragraph to the reranker. Expansion fixes
    recall, not ranking, and every phrasing costs a model call.

## Script 10: Doc2Query

Doc2Query writes the questions each chunk can answer. Retrieval then matches a
visitor's question against those generated questions, not against the chunk text. Both
sides are searched with BM25, so the comparison is about wording alone.

The knowledge base is six chunks about an invented theme park: basics, prices, opening
hours, transport, rides and park rules. Three visitor questions each map to one chunk.
Two of them share no content word with any chunk. The third uses the chunk's own words,
as a control.

*   The basic set is five questions for one chunk, with `question`, `question_type` and
    `difficulty`. It is shown for comparison only.
*   The wider set is eight questions, varied by type, wording, difficulty and
    perspective. It adds `perspective`, `is_answerable` and `answer`: the model
    answers its own question from the chunk, and a question the chunk cannot answer is
    dropped. Indexed, such a question would send a visitor to a chunk without the
    answer. On the first chunk, 3 of 8 were dropped.
*   Part 4 generates the wider set for every chunk and keeps the answerable questions:
    39 of 48 in the run below. It then builds two BM25 indexes, one on the six chunks
    and one on the kept questions. The questions can be stored in the same record as
    the chunk.

```
  prose retrieval accuracy   :  33.3%  (1/3)
  question retrieval accuracy:  33.3%  (1/3)
```

| # | Query | Prose score | Question score | Prose | Question | Matched generated question |
| :-: | :--- | :-: | :-: | :-: | :-: | :--- |
| 1 | `Am I allowed to take a picnic in?` | 0.000 | 3.624 | ✗ | ✗ | `How long does it take to reach the park by taxi?` (kb_004) |
| 2 | `What time should I show up to avoid the crowds?` | 0.000 | 8.979 | ✗ | ✓ | `If you wanted to avoid crowds, when would be the best time to go?` |
| 3 | `How much does it cost to park a car?` | 1.262 | 2.499 | ✓ | ✗ | `How much does a weekday adult ticket cost?` (kb_002) |

*   A 0.000 in the prose column means nothing matched. BM25 scores every chunk at zero
    and `max()` returns the first one, `kb_001`.
*   The question index returns the chunk that produced the closest generated question,
    so a hit depends on how close the visitor came to one of those phrasings.
*   The generated questions bring words of their own, and those work both ways. Query
    2 was won by `avoid crowds`. Query 1 was lost to a taxi question through `take`,
    and the control query to a ticket question through `cost`.
*   The two indexes' scores are not compared. They come from two BM25 corpora (6 long
    chunks, 39 short questions), so they are on different scales.
*   The generated questions change from run to run. Four runs gave the question index
    3/3, 2/3, 1/3 and 1/3; the prose index is 1/3 every time. With three queries, one
    flip moves the accuracy by 33 points. What holds across runs is the mechanism, so
    count the net change, not the wins alone.

## Script 11: Knowledge extraction and a knowledge base audit

DeepSeek does two jobs here. Extraction turns three visitor conversations about an
invented theme park into knowledge base entries in three steps.

*   Extract. The model pulls typed points out of each conversation: `fact`, `need`,
    `question`, `process` and `caution`, each with `confidence`, `source`, `keywords`
    and `category`. In three runs every point came back at confidence 1.00, so the
    script prints that the column does not tell the points apart.
*   Filter. `need` and `question` points record what a visitor wanted, not what is
    true. Left in the index, a question about ticket prices could retrieve "the
    visitor wanted to know the ticket price". They are dropped: 16 of 39 points in the
    run below.
*   Merge. The remaining points are grouped by type, and each group becomes one entry
    through one model call. 23 points became 3 entries.

```
  fact: 13 points -> 1 (confidence 1.00)
    categories: crowd levels, opening hours, pricing, recommended rides, rules, services
    Riverbend Park opens daily from 08:00 to 20:00. Adult tickets cost 399 on weekdays ...
```

*   Grouping by type puts different topics into one entry. A real base would group by
    topic, but the model's `category` is too loose for that, with 11 to 14 different
    values for 23 points across three runs. Nothing later uses the merged entries.

The audit checks a separate six-entry base for three problems, each with its own prompt
and its own extra input:

| Check | Extra input | Finds |
| :--- | :--- | :--- |
| Coverage | the test questions | questions no entry answers |
| Freshness | today's date | entries that have gone out of date |
| Consistency | none | entries that contradict each other |

*   Coverage needs the questions because a gap only exists relative to something
    someone asks. Freshness needs the date because the model has no clock.
*   The base holds three planted defects, which serve as the answer key: no entry
    about pets, a winter festival that ended in January 2024, and `kb_002` and
    `kb_005` giving the parking charge as 100 and 150. The report checks each check
    against its planted defect by id and counts everything else it flagged.

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

*   All three planted defects were found in each of three runs.
*   The freshness prompt names expired dates and periods, superseded rules and
    finished events, all of which the model can judge from an entry and the date.
    With "prices likely to have moved" on that list, it flagged 4 of 6 entries,
    including a ticket price with no date at all. Without it, it flagged 2 in each of
    three runs.
*   The second freshness finding, kb_002, is the parking conflict again. The three
    checks take different inputs and their findings still overlap.
*   The scores are the model's own numbers and nothing measures them. The findings can
    be checked against the planted defects. The model finds conflicts, and a person
    decides which side is right.

## Script 12: Knowledge base versions and a regression check

Version 1 of an invented park's base has 3 entries. Version 2 adds 2 entries and
extends the other 3. Embeddings do the retrieval here, not a chat model.

*   Fingerprints. Each version gets an MD5 hash of its sorted ids and texts, so one
    edited character changes it: v1.0 has hash `1d0aec844c6e`, v2.0 `61d5d51301c1`.
*   Diff. Added and removed ids come from set operations, modified ones from exact
    text comparison, with no model: 2 added, 0 removed, 3 modified.
*   Indexing. Both versions are embedded with gemini-embedding-001 at 1024 dimensions
    into `IndexFlatIP`. At 1024 dimensions the vectors come back with a mean length of
    0.62, so each is divided by its own length first. The inner product is then the
    cosine, and no entry ranks higher only because its vector is longer.
*   Scoring. Each of five test questions names a string the answer must contain. A
    question counts as answered when that string appears in the top 3 entries, and the
    table shows the rank of the first entry holding it.

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

*   Version 1 has only 3 entries, so the top 3 is its whole base. A MISS there means
    the answer is missing, never that retrieval failed.
*   The accuracy hides a weak retrieval. The location entry comes second in version 1
    and third in version 2. With the top 1 both versions would fail, and with the top 2
    version 2 would.
*   The gain from 60% to 100% comes entirely from content version 1 lacked: `line 11`
    and `launch coaster` are in no entry of version 1. Version 2 does not search
    better; it has more to find.
*   The mean search time was 0.007 ms and 0.006 ms. An exact search over 3 or 5
    vectors costs the same, so the script reports no measurable change.
*   The regression check asks whether every question version 1 answered still passes:
    3 of 3 do. Five cases catch only the breakages they cover, so the claim is "no
    regressions on this test set".
*   A substring test shows that the answer's text was retrieved, not that a reply
    built from it would be right.

## Script 13: GraphRAG against a vector baseline

The corpus is an invented English archive, `northgate_archive.txt` (1220 words), and the
question is `How did Mira Delaunay's way of working end up affecting Port Halbrook?`.
The answer needs five facts that no passage states together: `Delaunay trained Ek, Ek
founded Northgate, Northgate developed Latch Encoding, Latch Encoding made the Orrery
system possible, Orrery was deployed at Port Halbrook`.

*   The baseline splits the archive into 7 chunks of about 150 words, embeds them with
    gemini-embedding-001 and answers from the nearest 3 (cos 0.686, 0.655, 0.652) with
    gemini-3.1-flash-lite. All five chain terms appear in those chunks. A term can
    also sit in a passage that denies the link: the third chunk is the Broch
    collection, which the archive itself warns researchers against.
*   GraphRAG (Microsoft's graphrag 2.7.2) has a model extract entities and
    relationships from each text unit when it builds the index. The Leiden algorithm
    groups closely linked entities into communities, and the model summarises each
    community. graphrag pins numpy 1.x, so it runs in its own virtual environment and
    the script drives it from the command line.
*   The index holds 28 entities, 36 relationships, 4 communities and 4 community
    reports, from 5 text units. `ASHFIELD` and `ASHFIELD POLYTECHNIC`, `NORTHGATE` and
    `NORTHGATE LAB` stay four entities: extraction merges only entities with the same
    name and type, and entity resolution, which would join different names for one
    thing, is off by default.
*   Global search answers from the community summaries by map-reduce and cites them as
    `Reports`. Local search starts from the entities in the question and also cites
    `Entities`, `Relationships` and `Sources` (the text units). The script checks that
    every cited id exists in its table; all of them do. That shows the rows exist, not
    that they support the sentence they are attached to.

Part 8 counts, in each of the three answers, how many of the question's 7 entities it
names:

```
  baseline  names 4 of the 7 entities the question is about, plus 1 beyond it
    not named: Tomas Ek, Coastal Institute, Northgate Lab
    beyond the chain: LATCH-V
  global    names 7 of the 7 entities the question is about, plus 1 beyond it
    beyond the chain: PRIYA RAMAN
  local     names 7 of the 7 entities the question is about, plus 3 beyond it
    beyond the chain: JULIAN ADEYEMI, LATCH-V, VELLUM INSTRUMENTS
```

*   The retrieved text held all five chain terms, yet the baseline answer names only 4
    of the 7 entities. It reaches Port Halbrook through the later Latch-V system and
    leaves out Ek, the Coastal Institute and Northgate Lab: incomplete rather than
    wrong. Retrieving the passages is not the same as joining them. Both graph answers
    name all seven, because the joins were computed when the index was built.
*   The comparison has limits. It is one question. Naming an entity is not stating the
    link between two of them. The top 3 of 7 chunks is 43% of the corpus, so the
    vector route still retrieves every term.
*   The baseline made three API calls: embeddings for the chunks, an embedding for the
    question, and the answer. The global search took 27 to 87 seconds across runs and
    the local search about 19. Before any question, the graph needs a full indexing
    pass that grows with the corpus, not with the number of questions.
