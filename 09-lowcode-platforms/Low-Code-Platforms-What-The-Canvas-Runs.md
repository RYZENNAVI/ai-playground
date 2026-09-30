# Low-code platforms: what the canvas runs

On a low-code platform an AI application is built by dragging nodes onto a canvas and wiring
them together. These five scripts rebuild the mechanisms behind that editor locally and in code,
so each claim can be checked against printed output. Script 01 runs a workflow from its exported
JSON definition. Script 02 puts a real chat model behind one node and scores its replies against
the strings the next node compares. Script 03 writes a plugin held to a declared schema. Script
04 indexes a table two ways and asks a question only one of them can answer. Script 05 serves
the platform's HTTP API from a local server and calls it. Most failures shown here raise no
error, so each script prints the number that exposes them. This document explains what each
script does and the ideas it relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_workflow_engine_from_spec.py` | A workflow run from its JSON definition: reference validation, topological order, batch bodies against carried state, a selector branch, and sub-workflow calls with a depth guard |
| 02 | `02_llm_node_output_contract.py` | A real model behind a workflow node, prompted three ways and scored against the vocabulary the next node compares; a deterministic edit given to a model and to code |
| 03 | `03_plugin_io_contract.py` | A plugin held to a declared input and output schema, permissive against strict field mapping, a skip policy that reports, and what paging costs the caller |
| 04 | `04_table_knowledge_base_retrieval.py` | A table indexed two ways, semantic retrieval against a column filter on a three-condition question, and what prose recall costs |
| 05 | `05_platform_api_protocol.py` | A local server with the platform's three endpoints and its event stream, blocking against streaming, two unrelated failures behind one status code, and a probe that rewrites the cause of a failure |

## Shared setup

*   Scripts 01, 03 and 05 make no network call and need no key. Script 02 reads
    `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` or `OPENAI_API_KEY` from `.env`, in that order (the
    OpenAI key also reads `OPENAI_BASE_URL` and `OPENAI_MODEL`), and backs off on a rate limit.
    Script 04 embeds locally with `BAAI/bge-small-en-v1.5`. It looks under any sibling module's
    `weights/` first and downloads the encoder once through `modelscope` only when no module has
    it.
*   The dependencies are openai, python-dotenv, sentence-transformers, numpy, fastapi and
    uvicorn, plus modelscope if script 04 has to fetch its encoder.
*   `data/` holds the inputs: three workflow definitions in `data/workflows/`,
    `commission_plans.csv` (nine rows, four columns), `user_behavior_event.csv` (twenty event
    rows across three user ids and four days) and `service_notes.txt`. Script 03 writes
    `data/review_feed/page{1,2,3}.atom` and script 05 writes `data/mock_platform_server.py` on
    every run. Those two are not tracked.

## Script 01: A workflow engine from a JSON definition

A visual workflow is a typed directed graph plus a registry of node handlers, and the editor is
one way of writing that graph. In code, a name that does not exist fails at import time. On a
canvas, a reference to a field no upstream node emits is drawn like any other line, and the
failure comes later, if at all. The `model` handler here answers each node title with a fixed
local function, so every run prints the same thing. Script 02 puts a real model behind the same
node type.

| Node type | What it does in the engine |
| :--- | :--- |
| `start`, `end` | Hold the values the run was called with, and read the ports the caller receives |
| `plugin`, `code` | Call a registered function by name (`PLUGINS`, `CODE_FNS`) |
| `model` | Call the local stand-in for a model |
| `text` | Fill a template string from its inputs |
| `selector` | Compare one port against a literal and emit a branch label |
| `batch` | Run a body once per element of an array |
| `subworkflow` | Run another definition and return its end ports |

*   Part 1 loads the three definitions and counts their node types. `MarketSentiment` (11 nodes,
    8 edges) calls the other two. `DailyReport` (6 nodes, 7 edges) has more edges than nodes
    because its start node fans out to three model nodes that later rejoin. Those three do not
    depend on each other, so any order among them is valid, and the same holds for two nodes in
    `MarketSentiment`. Wherever a graph allows more than one order, anything that depends on the
    order is a latent bug.
*   Part 2 checks every reference before anything runs. An input is a string in one of three
    forms: `literal:ABC-Trade` is a constant, `item.published` is the current element inside a
    batch body, and `136482.digest` is port `digest` of node `136482`. `validate()` walks every
    node, batch body, `collect` mapping and edge, and the shipped definitions pass. Editing
    `136482.digest` to `136482.summary` gives
    `123474.values wants 136482.summary, but 136482 emits ['digest', 'branch']`, and the canvas
    draws both versions the same. A batch's `over` and a selector's `cases[].when` sit outside
    `inputs`, so they are checked on their own. The same function checks each sub-workflow call
    against the target's declared inputs. `run_workflow` never calls `validate`: it is the
    editor's gate, so a definition handed straight to the engine fails at the node that reads
    the missing port.
*   Part 3 orders the top-level nodes with Kahn's algorithm: Start, FetchNews, PerArticle,
    KeepMarked, Hotwords, ReviewAnalysis, DailyReport, End. One added edge from End back to
    FetchNews leaves 1 of 8 nodes able to run and 7 never ready. The function returns the
    stalled set instead of looping, and `run_workflow` refuses a definition with a cycle.
*   Part 4 runs the main workflow and prints what each node produced, with the sub-workflow
    nodes indented one level.
*   Part 5 lists the three code nodes. Each applies a fixed comparison that reshapes data for
    the next node: a timestamp becomes 1 when it falls on the reference date, two parallel
    arrays become one filtered array, and one array of digests becomes a positive and a negative
    list (`neutral` lands in neither). Then it splits three scenes two ways.
    `Scene\d+:([^Scene]+)` reads like "anything but the word Scene", but the class excludes the
    letters S, c, e and n, so each capture stops at the first of them. Both versions return 3
    parts, and the class keeps 7 of 174 characters (`'A r'`, `'Th'`, `'Ev'`). Counting parts
    cannot tell them apart. Splitting on the heading with `re.split` cannot be cut short by the
    body text. No workflow here uses this split.
*   Part 6 runs `code_same_calendar_day` over all 24 orderings of the four articles. `run_batch`
    gives each iteration its own copy of the context, so an isolated mark reads only its own
    article: `[1, 1, 0, 1]`, one value per article in every ordering. Carried as a running
    total, the value depends on position. The article marked 0 receives 0, 1, 2 or 3, and 6 of
    the 24 orderings print `[1, 2, 2, 3]`. A running total cannot live inside a batch body, and
    neither can a shared style or a de-duplication set. Anything that must hold across elements
    goes into the data before the batch splits it.
*   Part 7 reads the branch the selector `KeepTodayOnly` wrote. The selector removes nothing. It
    labels 1 of 4 articles `drop`, and that element keeps its position in the collected arrays.
    `KeepMarked` is the node that drops it, leaving 3. The two nodes are halves of one chain, a
    dependency that lives in the data rather than in the edges.
*   Part 8 follows both sub-workflow calls. Each inner definition runs to its own end node
    before the outer one continues. `MAX_CALL_DEPTH = 3` guards a definition that calls itself:
    depths 0 to 3 run, and the call into depth 4 raises `call depth 4 exceeded at 'self_call'`.
    Without the guard the engine would not hang. Python's recursion limit (1000 here) would
    raise instead, far deeper, after every level had run the nodes before its call, with a
    message that names no workflow. The guard makes the error arrive early and name the
    definition.

## Script 02: A model node against the next node's strings

The code node after the model compares strings. A row whose verdict is not exactly `positive`,
`neutral` or `negative` matches no branch and drops out without an error. The script asks
`deepseek-chat` at `temperature=0` to label six app reviews with three prompts and scores every
reply against those three words. It has no hand-labelled answers, so every number is about the
output contract, not about whether a judgement was right. The table is one of three runs.

| Variant | Prompt | Parsed as JSON | Verdict in the vocabulary | Routed |
| :--- | :--- | ---: | ---: | ---: |
| `plain` | A role and a task | 0/6 | 0/6 | 0/6 |
| `example` | Plus an output example naming `verdict` and `digest` | 0/6 | 6/6 | 6/6 |
| `json mode` | Plus a written constraint and `response_format={"type": "json_object"}` | 6/6 | 6/6 | 6/6 |

*   Parts 1 to 3 run the node once per variant and print how each reply was read. The plain
    replies are prose such as `**Sentiment:** Positive`. The parser's regular expression reads
    all six, but the value is `Positive`, not `positive`.
*   Part 4 scores the replies. Plain prints the values outside the vocabulary: `Frustrated`,
    `Mixed`, `Negative`, `Neutral`, `Positive`. Temperature 0 did not make plain repeatable. In
    two runs four of the six differed only in case. In the third it was five, because
    `Login keeps failing` came back `Negative` instead of `Frustrated`. Every other line matched
    across the three runs. The example buys the vocabulary and nothing else. It is itself a
    fragment with no braces, and the model copies that shape, so parsing stays at 0/6. The third
    variant changes two things at once, so the run credits the pair, not either one.
    `json_object` guarantees a JSON object and carries no enum, and nothing checks that the
    object holds exactly the two keys.
*   Part 5 hands the rows to the code node. Plain routes 0 of 6 and prints the titles it
    dropped. The other two route all six (positive 2, neutral 1, negative 3).
*   Part 6 normalises each label first (strip, strip `*#."'`, lower-case) and matches the whole
    cleaned string. A substring test would fold `'not positive'` onto `positive`. Plain gets 4
    rows back. `frustrated` and `mixed` stay out, because they are words the model chose, not
    spellings of the three. Part 5's report is what shows those two. Once folded, plain agrees
    with json mode on 4 of 6 rows (5 of 6 in the third run). That is agreement between variants,
    not accuracy.
*   Part 7 asks the model and a regular expression to remove seven listed words (`broker`,
    `brokerage`, `application`, `app`, `user`, `users`, `not`) and keep the rest in order. A
    word count cannot see order (`users trust brokers` and `brokers trust users` count the
    same), so the two results are compared token by token. The model matched the code exactly in
    four runs of four. Knowing that took the code version anyway, so a rule that can be written
    down belongs in a code node.

## Script 03: A plugin held to its schema

A plugin is how a closed workflow reaches something outside it. The platform reads its declared
input and output schema, so the editor can check a wire into it before anything runs. Here
`fetch_page` reads one of three Atom files. A hosted plugin would put an HTTP call there and
change nothing else, because the contract says nothing about where the bytes come from.

*   Part 1 writes the three feed pages into `data/review_feed/`, the same on every run. The
    first entry on page 3 has no rating element.
*   Part 2 prints `PLUGIN_SCHEMA`. The inputs are `app_id` (string) and `page` (integer), both
    required. The outputs are `items` (rows of title, rating, author, updated and content),
    `page`, and `skipped` (title and reason).
*   Part 3 makes a call that satisfies the input schema.
*   Part 4 makes three that do not, and the handler refuses each before any page is fetched:
    `page is required and was not passed`, `page should be integer, got str` for `'1'`, and
    `sort is not a declared input`. An undeclared argument is an error, not ignored, because a
    caller that passes `sort` believes the plugin sorts, and it does not.
*   Part 5 checks the rows of pages 1 and 2 against the output schema. `skipped` is a declared
    output, so the editor has a port to wire it to. Against a schema that forgets it,
    `validate_output` reports `skipped is returned but not a declared output`, the same way
    `validate_args` refuses an undeclared input. Both checks share one type test,
    `matches_type`, because `bool` is a subclass of `int` in Python and the contract should not
    be strict in one direction and loose in the other.
*   Part 6 maps page 3 twice. `map_permissively` reads every field with a default and converts
    nothing. It returns 2 rows and raises nothing, and the output schema then finds 2
    violations: `rating=None` on the incomplete row and the string `'5'` on the good one.
    `map_strictly` stops and names the field: `entry is missing ['rating']`.
*   Part 7 reads through `read_pages`. A plugin that walks pages should not stop on one bad
    entry, so here the handler skips it and returns it in `skipped` with the reason
    (`'Alerts arrive late': entry is missing ['rating']`). Dropping keeps the page readable only
    when the count of what was dropped comes back with it. Then it prices paging:

    | `page_limit` | Pages read | Requests | Rows | Skipped |
    | ---: | ---: | ---: | ---: | ---: |
    | 1 | 1 | 1 | 3 | 0 |
    | 3 | 3 | 3 | 7 | 1 |
    | 20 | 3 | 4 | 7 | 1 |

*   With a limit of 3 the loop stops on its own count. With 20 it asks for page 4, and that
    request only finds the end. Locally it costs a file check. Over HTTP it is a round trip that
    returns nothing. `page_limit` is an argument to `read_pages`, the loop around the one-page
    handler, not an input in `PLUGIN_SCHEMA`. A canvas drawn from the schema shows `app_id` and
    `page`, and nothing about how far the loop walks.

## Script 04: A table knowledge base

Prose is cut by length. A table already has a boundary: a question is asked about a row, so each
row is one chunk. The column names travel with the values, because `19.00` means nothing alone.
The bge-small encoder (384 dimensions) embeds every chunk for retrieval by cosine similarity.

*   Part 1 loads the two tables and the prose file.
*   Part 2 prints chunks such as
    `family: Retail; plan: Active; monthly_fee: 19.00; per_trade_fee: 1.95`: 9 plan chunks and
    20 event chunks, none split mid-row.
*   Part 3 asks which column can single a row out:

    | Column | Distinct values | Rows uniquely identified | Worst case |
    | :--- | ---: | ---: | ---: |
    | `family` | 3 | 0/9 | 3 rows share a value |
    | `plan` | 9 | 9/9 | 1 row |

*   Part 4 asks `"What does the Momentum plan cost per trade?"`. Against whole rows the right
    row comes first (0.793). Against an index on `family`, the top three are identical
    `family: Retail` strings at 0.458. A good index column is one a user is likely to say and
    one that leaves a single row standing.
*   Part 5 asks `"Did user U-100241 sign in on 2026-05-04?"`, which holds three conditions. None
    of the top 4 by similarity satisfies all three. They score 0.795 to 0.791, a spread of
    0.004, and the correct row sits at rank 7 of 20 (0.743). Every sign-in row resembles a
    question about signing in, including a `Contact support` row whose detail says "sign-in
    would not complete". An id has no neighbourhood in a semantic space. A `TOP_K` of 7 would
    reach the row with six wrong ones, and nothing in the scores says which is right.
*   Part 6 turns the question into conditions on named columns. Matched literally, the event
    type finds nothing, because the question writes `sign in` and the column stores `Sign-in`.
    That condition is left out without a word, and 2 rows come back. Folding case and
    punctuation on both sides gives 1 row, which matches a direct scan of the table. A vector
    database with metadata filtering does this step alongside similarity. The contrast here is
    with similarity alone. Folding fixes this spelling, not the silence. `parse_conditions`
    recognises a user id or an event type only as a value the table holds, and still leaves out
    one it cannot find, while a date is taken by pattern and stays in. Both event-type checks
    are substring tests with no word boundary, so a question about opening the research digest
    is read as event type `Search`. The direct scan was written for this question, so it checks
    this answer, not the parser.
*   Part 7 asks `"Why does face unlock stop working after changing the password?"`. No column
    holds that answer, so no filter can be written for it, and retrieval returns the right two
    chunks at 0.773 and 0.754. Every recalled character is paid on every question:

    | `top_k` | Characters | Roughly tokens |
    | ---: | ---: | ---: |
    | 2 | 747 | 186 |
    | 4 | 1416 | 354 |
    | 8 (all chunks) | 2485 | 621 |

*   Eight chunks come to more than the 2069-character file because chunks overlap. Doubling the
    recall costs 1.90x and then 1.76x, not exactly 2x, because the chunks differ in length.

## Script 05: The platform's HTTP API

The script writes a FastAPI server to `data/mock_platform_server.py`, starts it with uvicorn on
a free loopback port, and calls it over real HTTP. The server is a workflow deployment that
declares one input variable, `question`. The chat and completion endpoints exist and refuse it,
with `not_chat_app` and `not_completion_app`.

| Endpoint | Application type | Where the user's words go |
| :--- | :--- | :--- |
| `/v1/chat-messages` | Conversational | Top-level `query` |
| `/v1/completion-messages` | Single-shot completion | Inside `inputs`, under the variable the deployment declares |
| `/v1/workflows/run` | Workflow | Inside `inputs`, under the variable the deployment declares |

The key inside `inputs` is not part of the protocol. It is whatever the builder named the start
variable, so a caller that guesses `text` sends a well-formed request the deployment cannot
read.

*   Part 1 starts the server and prints its port.
*   Part 2 makes a blocking call: HTTP 200 and one body with the answer, with nothing observable
    in between.
*   Part 3 streams the same run: `workflow_started`, one `node_finished` each for Start,
    Retrieve and Answer, and `workflow_finished`. The 5 events take 0.16 to 0.18s across runs,
    most of it the three 0.05s node sleeps. For a multi-node workflow the stream is the only
    view of which node ran. The reader keeps only lines that start with `data: `, so blank and
    keep-alive lines are skipped.
*   Part 4 prints the client's headers as written and redacted (`Bearer ***`). On a server whose
    stdout is collected, the first form puts the credential into the log store.
*   Part 5 sends two requests that both come back HTTP 400. The first goes to
    `/v1/workflows/run` with an empty `inputs` object and gets `app_unavailable`. The endpoint
    and the credential are right, and the one thing that does not travel is the user's question.
    The second is a correct chat body sent to the workflow deployment and gets `not_chat_app`. A
    payload can repair the first. No payload can repair the second, because the endpoint is the
    mistake. Only the error code says which.
*   Part 6 probes for the key, trying no key, then `text`, `query`, `question` and `prompt`. The
    first three get `app_unavailable` and `question` gets HTTP 200, so it takes 4 requests to
    find a name written in the deployment's own settings. The first refusal already said
    `input variable 'question' is required`, but the probe reads only the status.
*   Part 7 stops the server and runs the same probe. All 5 attempts end `timed out`, and the
    caller is told `every input format failed; check the application configuration and API key`.
    Every failure was read as the wrong shape, the only cause the loop can hold, so the real
    cause is replaced by a plausible wrong one.
