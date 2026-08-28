# Low-Code Platforms: What the Canvas Actually Runs

Five scripts on one question: **when an AI application is assembled by dragging boxes onto
a canvas instead of writing code, what is actually executing — and which mistakes does that
way of building make harder to see?**

Everything here runs locally. There is no platform account, no hosted deployment, no
canvas. Each script rebuilds one mechanism a low-code platform performs behind its editor,
in a form where the claims can be checked: a workflow engine that reads a declarative graph
and runs it, a model node scored against the vocabulary the next node compares against, a
plugin held to its declared schema, a table indexed two ways and asked a question only one
of them can answer, and a local server speaking the protocol such a platform exposes.

The shapes are the ones mainstream platforms use — a typed node graph with port
references, batch bodies, selector branches, sub-workflow calls, and an HTTP API with
three endpoints and a server-sent event stream.

**The module needs no download and no platform credential.** Script 02 calls a chat model
(DeepSeek, Gemini or OpenAI — whichever key is present). Script 04 embeds locally with
`BAAI/bge-small-en-v1.5`, reusing a copy this repository already has on disk. Scripts 01,
03 and 05 make no network call at all; 05 starts a `uvicorn` subprocess on a loopback port
and stops it before exiting.

**Requires**: `openai`, `python-dotenv`, `sentence-transformers`, `numpy`, `fastapi`,
`uvicorn`.

---

## 1. What a canvas is, underneath

### 1.1 The proposition

**A visual workflow is a typed directed graph plus a registry of node handlers.**
The editor is a way of writing that graph. Everything the editor does for you —
validating a connection, deciding execution order, splitting an array across a batch body,
following a call into another graph — is something the runtime has to do anyway, and each
of those is a place where a mistake can hide.

The reason this matters is not that graphs are hard. It is that **the canvas removes the
compiler.** In code, a name that does not exist is a failure at import time. On a canvas,
a reference to a field no upstream node emits is a line the editor draws exactly like any
other line, and the failure arrives later, if at all.

### 1.2 Where the decision lives

The three ways of building on such a platform differ in one thing only: **who decides what
happens next.**

| Form | Who decides the next step | What you get |
| :--- | :--- | :--- |
| **Agent** | The model, from tool descriptions | Order is not fixed; the same question can take different paths |
| **Branch on a classification** | The model picks a lane; the lanes are drawn by a person | One decision is delegated, the rest is fixed |
| **Workflow** | The graph. Every run walks the same edges | Same input, same path, every time |

Moving toward the workflow end buys repeatability and spends flexibility. Script 01 is
about that end of the range, because it is the end where the mechanics are explicit enough
to reproduce.

### 1.3 The node types the engine implements

`01_workflow_engine_from_spec.py` registers nine, which is enough to run all three
definitions in `data/workflows/`:

| Type | What it does in the engine |
| :--- | :--- |
| `start` | Holds the values the run was called with |
| `end` | Reads the ports the caller receives |
| `plugin` | Calls a registered function by name (`PLUGINS`) |
| `code` | Calls a registered function by name (`CODE_FNS`) |
| `model` | Calls a handler that stands in for a model (see 1.5) |
| `text` | Fills a template string from its inputs |
| `selector` | Compares one port against a literal and emits a branch label |
| `batch` | Runs a body once per element of an array |
| `subworkflow` | Runs another definition and returns its end ports |

### 1.4 The three definitions

`data/workflows/` holds three JSON files. `market_sentiment` calls the other two:

```
MarketSentiment  11 nodes   8 edges  batchx1, codex2, endx1, modelx2, pluginx1,
                                     selectorx1, startx1, subworkflowx2
ReviewAnalysis    6 nodes   4 edges  batchx1, codex1, endx1, modelx1, pluginx1, startx1
DailyReport       6 nodes   7 edges  endx1, modelx3, startx1, textx1
```

`DailyReport` has more edges than nodes because its start node fans out to three model
nodes that later rejoin. That fan-out is worth noticing: those three nodes have no
dependency on each other, so any order among them is valid, and a topological sort is free
to pick one. The same is true of two nodes in `MarketSentiment`. **Wherever the graph
permits more than one order, anything that depends on the order is a latent bug** — which
is the same property section 4 measures inside a batch body.

### 1.5 One honest limitation

The `model` handler in script 01 does not call a model. It answers each node title with a
deterministic local function, because script 01 is about the engine around the node and has
to produce identical output on every run. Script 02 puts a real model behind the same node
type and measures what comes back.

---

## 2. References: the part the editor draws and does not check

### 2.1 Three forms of reference

Every input on every node is a string. `resolve()` reads three forms:

```python
if ref.startswith("literal:"):
    return ref[len("literal:"):]
source, _, port = ref.partition(".")
if source == "item":
    if item is None:
        raise KeyError("'item' referenced outside a batch body")
    return item[port]
return ctx[source][port]
```

- `literal:ABC-Trade` — a constant typed into the editor
- `item.published` — the current element inside a batch body
- `136482.digest` — port `digest` of node `136482`

### 2.2 Validating them before anything runs

`validate()` walks every node, every batch body, every `collect` mapping and every edge,
and returns the list of references that cannot be satisfied. Run against the three shipped
definitions, it returns nothing. Then the script edits one reference — `136482.digest`
becomes `136482.summary` — and runs the same check:

```
edit one reference from 136482.digest to 136482.summary:
  123474.values wants 136482.summary, but 136482 emits ['digest', 'branch']
the canvas draws that edge exactly the same either way
```

That last line is the point of the exercise. **The two graphs are visually identical.**
One of them cannot run. A platform that does not perform this check hands the difference to
the runtime, where it surfaces as a `KeyError` in the middle of a long job, or — if the
mapper downstream is forgiving — as a missing value nobody notices.

The same function also checks sub-workflow calls against the target's declared inputs, so
`ReviewAnalysis` cannot be called without `app_id`.

### 2.3 Execution order, and a graph that has none

`topological_order()` is Kahn's algorithm over the top-level nodes. On the main definition:

```
Start -> FetchNews -> PerArticle -> KeepMarked -> Hotwords -> ReviewAnalysis -> DailyReport -> End
```

Add one edge from `End` back to `FetchNews` and re-run it:

```
add one edge End -> FetchNews: 1 nodes can start, 7 wait forever
['107368', '123474', '136482', '140895', '170693', '174746', '900001']
```

**One node can start; seven wait forever.** The function returns the stalled set rather
than looping, and `run_workflow` refuses to execute a definition whose cycle set is
non-empty. On a canvas, that back-edge is one drag of the mouse.

---

## 3. Code nodes: shape adaptation, and a regular expression that is wrong

### 3.1 What code nodes are for

The engine registers three, and none of them holds business logic:

| Function | What it does |
| :--- | :--- |
| `code_same_calendar_day` | Return 1 when a published timestamp falls on the reference date |
| `code_keep_marked` | Keep the values whose parallel mark is the string `'keep'` |
| `code_split_by_verdict` | Split digests into two lists according to the verdict beside each one |

Every one of them reshapes data so the next node can read it: a date to a flag, two
parallel arrays to one filtered array, one array to two. **That is the job.** A model node
produces prose; a plugin wants a scalar; a batch body wants an array. Something has to
convert, and the converter is a code node.

### 3.2 A character class that excludes four letters

Splitting a numbered list is the standard first use of a code node. Here are two ways,
both in the script:

```python
def split_scenes_excluding(text):
    return [m.group(1).strip() for m in re.finditer(r"Scene\d+:([^Scene]+)", text)]

def split_scenes_by_separator(text):
    parts = re.split(r"Scene\d+:", text)
    return [part.strip() for part in parts if part.strip()]
```

`[^Scene]` reads as "anything but the word Scene". It is not. It is **anything but the
letters S, c, e and n**, so the capture stops at the first of those letters in the body
text. On three scenes of ordinary English:

```
splitting 3 scenes with a [^Scene] character class: 3 part(s)
  'A r'
  'Th'
  'Ev'
splitting the same text on the heading itself: 3 part(s)
  'A red kite rises over an empty green field at dawn.'
  'The same field at noon, seen from beneath a bending '
  'Evening arrives and the kite is a dark speck against'
```

Both versions return **three parts**. A check that counts the parts passes for both. The
failure is entirely in the content, and English makes it violent — nearly every word
contains an `e`. **Splitting on the separator rather than capturing between separators
removes the whole class of problem**, because the body text can no longer terminate a
match.

---

## 4. Batch bodies: what isolation buys and what it forbids

### 4.1 The rule

`run_batch` gives every iteration its own context:

```python
for element in resolve(node["over"], ctx):
    local = dict(ctx)
    for inner in node["body"]:
        local[inner["id"]] = run_node(inner, local, item=element, trace=trace)
    for name, ref in node["collect"].items():
        collected[name].append(resolve(ref, local))
```

Nothing an iteration writes is visible to the next one. That is what makes the elements
safe to process in any order — and it is also a hard constraint on what can go inside.

### 4.2 The same computation, both ways

The script runs one function over the same four articles twice: once isolated, once
carrying a running total.

```
isolated iterations: [1, 1, 0, 1]   (any order gives this)
carried across them: [1, 2, 2, 3]   (only this order gives this)
```

The first result is a property of each element. The second is a property of the sequence.
**A running total cannot live inside a batch body**, and neither can anything else that has
to hold across elements — a shared style, a de-duplication set, a rolling threshold.

The consequence is practical: **anything that must be consistent across the elements has to
be written into the data before the batch splits it up.** If four generated images have to
share a style, the style goes into each element's text in the code node that builds the
array, not into the node that consumes it.

### 4.3 The selector, and the hole it leaves

The batch body ends with a selector:

```json
{"id": "159567", "type": "selector", "title": "KeepTodayOnly",
 "cases": [{"when": "130992.same_day", "equals": 1, "then": "keep"}],
 "otherwise": "drop", "ports": ["branch"]}
```

It does not remove anything. It writes a label, and the elements it labelled `drop` still
occupy their position in the collected arrays:

```
keep  2026-05-04 18:20  Index closes higher on rate relief
keep  2026-05-04 09:05  Broker cuts commission on index funds
drop  2026-05-03 21:40  Quarterly filings land next week
keep  2026-05-04 11:55  Settlement window shortens in June
3 of 4 digests survive the branch
```

The next node, `code_keep_marked`, is what actually drops them. **The selector and that
code node are two halves of one chain**: one marks, the other cleans up after the marking.
Looking at either alone, neither seems necessary. This is the ordinary form of implicit
coupling on a canvas — a dependency that lives in the data rather than in the edges.

---

## 5. Sub-workflows: a call, and a guard

`run_workflow` recurses, and counts:

```python
def run_workflow(spec_id, library, inputs, depth=0, trace=None):
    if depth > MAX_CALL_DEPTH:
        raise RecursionError(f"call depth {depth} exceeded at {spec_id!r}")
```

The traced run shows the two calls nested one level down, with the inner definitions
running to their own end nodes before the outer one continues:

```
FetchNews            items=4 item(s)
PerArticle           digest=4 item(s), branch=4 item(s)
KeepMarked           kept=3 item(s)
Hotwords             text='index, broker, settlement, window, bas'
  FetchReviews         items=5 item(s)
  PerReview            verdict=5 item(s), digest=5 item(s)
  SplitByVerdict       positive=2 item(s), negative=2 item(s)
  End                  positive=2 item(s), negative=2 item(s)
ReviewAnalysis       positive=2 item(s), negative=2 item(s)
  SummarisePraise      text='2 item(s); first is: Fast and stable'
  ...
End                  report='MARKET\n3 item(s); first is: Index clos', hotwords='index, broker, ...'
```

`MAX_CALL_DEPTH = 3` exists because a definition may call another that calls back into it.
**On the canvas both are one tidy box.** Without the guard the failure is a hang, not an
error.

---

## 6. Model nodes: the contract between a probability and a comparison

`02_llm_node_output_contract.py` puts a real model behind the node type script 01 stubs
out, and scores it against the thing that consumes it.

### 6.1 What the next node does

The downstream code node compares strings:

```python
EXPECTED = ("positive", "neutral", "negative")

def split_by_verdict(rows):
    positive = [r for r in rows if r["verdict"] == "positive"]
    neutral  = [r for r in rows if r["verdict"] == "neutral"]
    negative = [r for r in rows if r["verdict"] == "negative"]
    return positive, neutral, negative
```

**A row matching none of the three branches raises nothing. It is simply gone.**

### 6.2 Three ways of asking, on the same six reviews

| Variant | Prompt |
| :--- | :--- |
| `plain` | A role and a task: "say how the reviewer feels, and give a short digest" |
| `example` | The same, plus an output example naming `"verdict"` and `"digest"` |
| `json mode` | The same, plus a constraint and `response_format={"type": "json_object"}` |

Measured with `deepseek-chat` at `temperature=0`:

```
--- 4. Scored against the vocabulary the next node compares against ---
  plain      0/6 replies parsed as JSON, 0/6 verdicts match the vocabulary exactly
             values outside it: ['Frustrated', 'Negative', 'Neutral', 'Positive']
  example    6/6 replies parsed as JSON, 6/6 verdicts match the vocabulary exactly
  json mode  6/6 replies parsed as JSON, 6/6 verdicts match the vocabulary exactly

--- 5. What the downstream code node does with those rows ---
  plain      routed 0/6  (positive 0, neutral 0, negative 0)
             dropped without an error: ['Fast and stable', 'Login keeps failing', ...]
  example    routed 6/6  (positive 2, neutral 1, negative 3)
  json mode  routed 6/6  (positive 2, neutral 1, negative 3)
```

**The plain prompt loses all six rows.** Not one of them is wrong about sentiment — the
model answered `Sentiment: Positive` in prose, and one review came back with no label at
all, only the word `Frustrated` in a sentence. The classification is fine. The contract
is not.

### 6.3 Two findings worth separating

**The output example does the work; the format switch does not add to it.** For this model,
pinning an example to the prompt already produces 6/6 parseable JSON with 6/6 in-vocabulary
verdicts. `response_format` guarantees the *parser* — the reply will be a JSON object — but
it guarantees nothing about the *vocabulary*. Those are two different contracts, and only
one of them is what the code node depends on.

**Folding the label recovers most but not all of it.** Adding a three-line normaliser
before the comparison:

```python
def normalise(verdict):
    cleaned = verdict.strip().strip("*#.\"' ").lower()
    for word in EXPECTED:
        if word in cleaned:
            return word
    return cleaned
```

```
--- 6. The same rows, with each label normalised first ---
  plain      routed 5/6  (positive 2, neutral 1, negative 2)
             still outside the vocabulary: ['frustrated']
```

Five of six come back. The sixth stays out, because `frustrated` is not a spelling of a
value in the enum — it is a different word. **Normalising moves the failure from silent to
visible, which is the whole of what it can do.** The enum belongs on the boundary, and
whatever cannot be folded onto it has to be reported rather than dropped.

### 6.4 A rule is not a prompt

The last step hands the same deterministic edit to both kinds of node: remove seven listed
words from a paragraph, leave everything else unchanged.

```python
STOPWORDS = ["broker", "brokerage", "application", "app", "user", "users", "not"]
```

```
model node : 'The app is slow, but report the research tab is reachable when reconnects. ...'
             1 listed word(s) survive: ['app']
code node  : 'The is slow, but report the research tab is reachable when the reconnects. ...'
             0 listed word(s) survive: []
```

The prompt even contains the line "Make sure the word *not* is removed" — a sentence that
exists because somebody watched the model leave it in. **The edit is defined by a rule, so
the node that can apply a rule owns it.** Sending it to a model costs a request, costs
latency, and still leaves a word behind.

---

## 7. Plugins: the schema is the interface

A plugin is how a closed system reaches anything outside itself — an internal database, a
company API, a page on the web. `03_plugin_io_contract.py` writes one the way a platform
expects one: a handler whose input and output shapes are declared, so the editor can
validate a connection into it before anything runs.

### 7.1 What gets declared

```python
PLUGIN_SCHEMA = {
    "name": "review_feed",
    "input": {
        "app_id": {"type": "string", "required": True},
        "page": {"type": "integer", "required": True},
    },
    "output": {
        "items": {"type": "array", "of": {
            "title": "string", "rating": "integer", "author": "string",
            "updated": "string", "content": "string",
        }},
        "page": {"type": "integer"},
    },
}
```

The transport is deliberately boring: `fetch_page` reads one of three Atom files the script
writes into `data/review_feed/` on every run. A hosted plugin would put an HTTP call there
and change nothing else. **The contract is what the platform reads, and the contract says
nothing about where the bytes come from.**

### 7.2 Three calls that are refused before any work happens

```
  {'app_id': 'ABC-Trade'}                          ['page is required and was not passed']
  {'app_id': 'ABC-Trade', 'page': '1'}             ['page should be integer, got str']
  {'app_id': 'ABC-Trade', 'page': 1, 'sort': ...}  ['sort is not a declared input']
```

The third is the interesting one. An undeclared argument is not ignored — it is an error,
because a caller that passes `sort` believes the plugin sorts, and it does not.

### 7.3 Two mappers, one missing field

The third feed page carries an entry with no rating element, which is what a real feed does
when someone leaves a comment without a score. The script maps that page twice.

`map_permissively` fetches every field with a default, so a feed that stops sending one
still produces a row that looks complete:

```
permissive mapper returned 2 rows and raised nothing
  rating=None   Alerts arrive late
  rating='5'    Solid since the rewrite
the output schema finds 2: ['items[0].rating should be integer, got NoneType',
                            'items[1].rating should be integer, got str']
```

Note that the permissive mapper is wrong about **both** rows, not just the incomplete one:
it never converts, so even the good row carries the string `'5'` where the schema declares
an integer. Nothing raised. The next node would receive `None` for one rating and a string
for the other, and would find out when it tried to compare them.

`map_strictly` checks each field against the declared row shape and names what is absent:

```
strict mapper stops instead, and names the field: entry is missing ['rating']
one of these hands the next node a rating of None; the other hands it nothing
```

### 7.4 Skipping is a policy, and it has to be reported

Raising on one bad entry is the wrong behaviour for a plugin that walks pages, so the
handler takes a policy:

```python
def handler(args, mapper=map_strictly, skip_invalid=False):
    ...
    for entry in root.findall("atom:entry", NS):
        try:
            items.append(mapper(entry))
        except ValueError as error:
            if not skip_invalid:
                raise
            title = entry.find("atom:title", NS)
            skipped.append((title.text if title is not None else "?", str(error)))
    return {"items": items, "page": args["page"], "skipped": skipped}
```

**Dropping an entry keeps the page readable, but only if the count of what was dropped
comes back with it.** A plugin that skips silently is the permissive mapper one level up.

### 7.5 The page limit belongs to the caller

```
page_limit=1   fetched 1 page(s), 3 rows, 0 entry(s) skipped
page_limit=3   fetched 3 page(s), 7 rows, 1 entry(s) skipped
page_limit=20  fetched 3 page(s), 7 rows, 1 entry(s) skipped
  skipped 'Alerts arrive late': entry is missing ['rating']
```

The source holds three pages, so a limit of 20 costs three fetches here — and twenty
against a source that keeps answering. **A plugin that hardcodes how far it will walk hides
its own cost from whoever wires it up.** Twenty pages of three rows each is sixty requests
behind one box on a canvas.

---

## 8. Table knowledge bases: a row is not a paragraph

`04_table_knowledge_base_retrieval.py` is about the one place where the usual retrieval
advice inverts.

### 8.1 One row, one chunk

Prose is cut by length because sentences have no natural boundary at any particular size.
A table already has the boundary: **a row is the unit a question is asked about**, so a row
is a chunk. The headers travel with the values, because `19.00` means nothing alone:

```python
def row_to_chunk(row):
    return "; ".join(f"{key}: {value}" for key, value in row.items())
```

```
family: Retail; plan: Starter; monthly_fee: 0.00; per_trade_fee: 4.95
family: Retail; plan: Active; monthly_fee: 19.00; per_trade_fee: 1.95
9 plan chunks and 20 event chunks, none of them split mid-row
encoded with BAAI/bge-small-en-v1.5, 384 dimensions
```

### 8.2 Choosing the index column

A platform that indexes tables asks which column to index. The answer decides everything
downstream, and it is measurable:

```
family  3 distinct value(s), 0/9 rows uniquely identified, worst case 3 rows share a value
plan    9 distinct value(s), 9/9 rows uniquely identified, worst case 1 rows share a value
```

Asking `"What does the Momentum plan cost per trade?"` against the full rows puts the right
row first (0.793). Against an index built only on the family column, every candidate is one
of three identical strings:

```
0.458  family: Retail
0.458  family: Retail
0.458  family: Retail
```

**Three rows carry the value `Retail`, so that index cannot separate them.** The criterion
is a single question: *what is the customer most likely to say out loud, and is it
selective enough to leave one row standing?* Both halves matter — a column nobody mentions
is useless, and so is a column everybody's question matches.

### 8.3 What similarity cannot do

The question `"Did user U-100241 sign in on 2026-05-04?"` carries two conditions. Semantic
retrieval over twenty row-chunks returns:

```
0.795  event_time: 2026-05-04 09:30:00; event_type: Sign-in; ... user_id: U-100237
0.794  event_time: 2026-05-06 08:55:00; event_type: Sign-in; ... user_id: U-100258
0.792  event_time: 2026-05-07 09:05:00; event_type: Sign-in; ... user_id: U-100241
0.791  event_time: 2026-05-04 10:11:00; event_type: Contact support; ... user_id: U-100241
0 of the 4 rows returned satisfy both conditions
```

**None of the four.** The correct row — `U-100241`, `2026-05-04`, `Sign-in` — is not in the
top four at all, and the spread across the four returned is 0.004. That is not a bug in the
encoder. Every sign-in row resembles a question about signing in; an identifier has no
neighbourhood in a semantic space, and a date is four tokens that look like every other
date.

The scores also show why raising the recall count is not a fix. At a spread of 0.004,
`TOP_K` would have to grow until it covered the whole table before the ranking mattered.

### 8.4 What a filter does instead

```python
def parse_conditions(question, rows):
    conditions = {}
    for user_id in {row["user_id"] for row in rows}:
        if user_id.lower() in question.lower():
            conditions["user_id"] = user_id
    date = re.search(r"\d{4}-\d{2}-\d{2}", question)
    if date:
        conditions["date"] = date.group(0)
    ...
```

Turning the question into conditions over named columns gives an exact answer, and the
count is checkable against a direct scan of the table:

```
1 row(s) satisfy the filter; scanning the table directly finds 1, so the filter and the
table agree
```

### 8.5 A condition that matched nothing, and said nothing

The same step surfaced a second failure of the same family. The question writes `sign in`;
the column stores `Sign-in`. Matching literally, the event-type condition finds nothing —
**and a condition that matches nothing is simply left out of the filter**:

```
conditions matched literally: {'user_id': 'U-100241', 'date': '2026-05-04'}
  2026-05-04 10:05:00  Sign-in         Face unlock rejected three times then gave up
  2026-05-04 10:11:00  Contact support Reported that sign-in would not complete
2 row(s) satisfy those: the question writes 'sign in' and the column stores 'Sign-in',
so the third condition matched nothing and was dropped from the filter without a word
```

Two rows instead of one, with no error anywhere. Folding case and punctuation on both sides
restores it:

```python
def loosen(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())
```

**This is the same shape as the index-column mistake and the enum drift in section 6:
something did not match, and not matching produced silence rather than a signal.**

### 8.6 Where prose still wins, and what it costs

The reverse case is in the same script. `"Why does face unlock stop working after changing
the password?"` is answered by a paragraph in `data/service_notes.txt`, and **no column
holds that answer, so no filter can be written for it.** Retrieval returns the right two
chunks at 0.773 and 0.754.

The cost is visible:

```
4 chunks recalled carry 1416 characters, roughly 354 tokens, and every question pays that
before the model answers
```

**The recall count converts directly into money.** Four chunks of prose is a few hundred
tokens on every question; ten is a few thousand. A structured filter that returns one row
costs a fraction of that, which is a second reason to use it wherever the question can be
expressed as conditions.

---

## 9. The API a platform exposes, and a client that stops knowing what it is talking to

`05_platform_api_protocol.py` starts a local server that answers the three endpoints such a
platform exposes, then calls it. Nothing is mocked at the client — the client speaks real
HTTP to a real server on a loopback port, and the server declares exactly one input
variable, `question`.

### 9.1 The three endpoints

| Endpoint | Application type | Where the user's words go |
| :--- | :--- | :--- |
| `/v1/chat-messages` | Conversational | Top-level `query` |
| `/v1/completion-messages` | Single-shot completion | Inside `inputs`, under the variable the deployment declares |
| `/v1/workflows/run` | Workflow | Inside `inputs`, under the variable the deployment declares |

The last two matter more than they look. **The key inside `inputs` is not part of the
protocol — it is whatever the person who built the deployment named their start variable.**
A caller who guesses `text` when the deployment declares `question` gets a well-formed
request that the deployment cannot read.

### 9.2 Blocking and streaming

```
--- 2. The blocking call ---
  HTTP 200  run run-0001  status succeeded
  answer: Alerts read one-second quote buckets, so they trail the print.
  one request, one response, and nothing observable in between

--- 3. The same run, streamed ---
  workflow_started   run-0001
  node_finished      Start
  node_finished      Retrieve
  node_finished      Answer
  workflow_finished
  5 events in 0.18s; the node events are the only view of what ran
```

**For a multi-node workflow, streaming is not a typing effect — it is the only
observability there is.** Blocking tells you it succeeded. Streaming tells you which node
it was on when it did not.

The client's stream reader skips any line it cannot parse:

```python
for raw in response:
    line = raw.decode("utf-8").strip()
    if not line.startswith("data: "):
        continue
    try:
        events.append(json.loads(line[6:]))
    except json.JSONDecodeError:
        continue
```

That is not defensive clutter. An event stream carries blank lines and keep-alives between
events; raising on them would take down the whole run at a heartbeat.

### 9.3 A credential printed once per call

The client holds its key in the headers it sends, and a debugging print of those headers is
the single most common way that key escapes:

```
as written: {'Authorization': 'Bearer local-development-key', 'Content-Type': ..., 'Accept': ...}
redacted  : {'Authorization': 'Bearer ***', 'Content-Type': ..., 'Accept': ...}
```

Locally this looks harmless. Once the same code runs on a server whose stdout is collected,
**every request writes one more copy of the credential into the log store, and into every
replica and backup of it.** Redacting is one dictionary literal:

```python
def redacted(headers):
    return {**headers, "Authorization": "Bearer ***"}
```

The advice everyone repeats is *do not put your key in front-end code*. The key does not
usually leave through the front end. It leaves through the logs.

### 9.4 A request that carries no question

```python
def completion_dropping_input(self, question):
    return self.post("/v1/completion-messages",
                     {"inputs": {}, "response_mode": "blocking", "user": "demo"})
```

`question` is a parameter of that method and appears nowhere in the body it sends.

```
HTTP 400  app_unavailable
the endpoint is right and the key is right; the payload has an empty inputs object,
so the deployment refuses for its own missing input variable rather than for anything
the caller can see
```

The endpoint is correct, the credential is accepted, and the one thing that does not travel
is the user's words. Against a deployment whose start variable has a default, this does not
even fail — it returns a fluent answer to a question nobody asked.

### 9.5 Probing for the shape

The alternative to knowing the key is guessing it. The client tries five payload shapes
until one stops failing:

```python
shapes = [{}, {"text": question}, {"query": question},
          {"question": question}, {"prompt": question}]
```

```
inputs=[]             HTTP 400  app_unavailable
inputs=['text']       HTTP 400  app_unavailable
inputs=['query']      HTTP 400  app_unavailable
inputs=['question']   HTTP 200  accepted
4 request(s) to arrive at a key the deployment names in its own configuration
```

Four requests to discover something written down in the deployment's own settings. That is
merely wasteful. The failure mode is the next step.

### 9.6 The same probe with the server gone

The script stops the server and runs the identical loop:

```
inputs=[]             HTTP None  timed out
inputs=['text']       HTTP None  timed out
inputs=['query']      HTTP None  timed out
inputs=['question']   HTTP None  timed out
inputs=['prompt']     HTTP None  timed out
reported to the caller: 'every input format failed; check the application configuration
and API key'
```

**Five transport failures in a row, and the message that comes back names the application
configuration and the API key.** Every failure was read as "wrong shape", because that is
the only hypothesis the loop can hold. The cause is not merely lost — it has been replaced
with a plausible, wrong one, and whoever receives that message will go and check their key.

**A loop that treats every error as the same error will always report the error it was
built to expect.**

---

## 10. What all of these have in common

Every failure in this module has the same shape. **Nothing raised.**

| Where | What went wrong | What it looked like |
| :--- | :--- | :--- |
| A port reference edited to a field nobody emits | The graph cannot run | An identical-looking edge on the canvas |
| An edge added from the end back to the start | Seven nodes wait forever | One more line between two boxes |
| `[^Scene]` instead of splitting on the heading | Every scene truncated to two or three characters | The right **number** of parts |
| A plain prompt in front of a string comparison | Six of six rows discarded | Six correct classifications |
| A permissive field mapper | Wrong types handed downstream | A complete-looking row |
| A plugin that skips bad entries quietly | Rows missing from the result | A shorter list |
| An index column that is not selective | Retrieval cannot separate rows | Confident scores, all identical |
| A condition that matched no column value | The filter silently loses a condition | Two rows instead of one |
| `inputs: {}` on a completion call | The user's question never travels | A fluent answer |
| A probe that reads every error as "wrong shape" | The real cause is overwritten | A specific, wrong diagnosis |

Ten failures, ten silences. The pattern is not a property of low-code tools specifically —
it is a property of **systems assembled from independently-correct parts across interfaces
nobody checks.** A canvas produces more of those interfaces than code does, and checks
fewer of them.

The five scripts are five ways of putting a check back:

1. **Validate references before running**, and say which port was wanted and what the node
   actually emits.
2. **Refuse a graph with a cycle**, and name the nodes that are stuck.
3. **Print the intermediate quantity**, not just the final answer — how many parts the
   split produced, what the extraction node wrote, how many characters came back.
4. **Score against the consumer's vocabulary**, not against your own reading of the output.
5. **Declare the schema and enforce it in both directions**, so a missing field is named
   rather than defaulted.

### 10.1 The one rule that generalises

**Whenever a value has to be written in two places, the version that is wrong will not
raise — it will diverge.** The enum in a prompt and the values in the data; the key inside
`inputs` and the variable named in the deployment; the column name in a condition and the
spelling in the table; the port name in a reference and the port a node emits. Every one of
those pairs appears in this module, and every one of them fails quietly.

The countermeasure is the same in all four cases: **derive one side from the other, or
check them against each other at a point where a mismatch is an error.** `validate()` does
it for ports. `validate_output` does it for rows. `normalise` does it for labels. Nothing
does it for the key inside `inputs`, which is exactly why section 9.5 costs four requests.

---

## 11. The five scripts

| Script | What it demonstrates |
| :--- | :--- |
| `01_workflow_engine_from_spec.py` | A declarative graph → reference validation → topological order → execution; batch bodies against carried state, selector branches, sub-workflow calls with a depth guard, and what code nodes are for |
| `02_llm_node_output_contract.py` | A real model behind a workflow node, scored against the vocabulary the next node compares against; three prompt variants; a deterministic edit given to a model and to code |
| `03_plugin_io_contract.py` | A plugin held to a declared input/output schema; permissive against strict field mapping; a per-entry error policy; what paging costs the caller |
| `04_table_knowledge_base_retrieval.py` | A table indexed two ways; semantic retrieval against an exact filter on a two-condition question; where prose still wins and what its recall costs |
| `05_platform_api_protocol.py` | A local server speaking the three endpoints and the event stream; blocking against streaming; a credential in the logs; a request that drops the user's words; a probe that rewrites the cause of a failure |

### 11.1 Measured results

| # | Result |
| :--- | :--- |
| **01** | 23 nodes across three definitions execute end to end. One edited reference is caught statically: `123474.values wants 136482.summary, but 136482 emits ['digest', 'branch']`. One added back-edge leaves **1 node able to start and 7 waiting forever**. `[^Scene]` truncates three scenes to `'A r'`, `'Th'`, `'Ev'` while still returning three parts. Isolated iterations give `[1, 1, 0, 1]`; the same computation carrying state gives `[1, 2, 2, 3]`. The selector marks 1 of 4 elements `drop`; the cleanup node removes it |
| **02** | `deepseek-chat`, `temperature=0`, six reviews per variant. Plain prompt: **0/6 parse as JSON, 0/6 match the vocabulary, 0/6 routed** — all six discarded without an error. With an output example: 6/6 and 6/6. With `response_format` as well: 6/6 and 6/6 — **the example is what fixed it, not the format switch**. Normalising the label recovers 5/6 of the plain run; `frustrated` remains outside the enum. Model-applied stopword removal leaves 1 of 7 words in place; the code path leaves 0 |
| **03** | Three malformed calls refused before a page is fetched. The page with a missing rating: the permissive mapper returns 2 rows and raises nothing, and the output schema then finds **2 type violations across both rows**; the strict mapper stops and names the field. `page_limit=20` against a 3-page source costs 3 fetches, yields 7 rows and reports 1 skipped entry |
| **04** | Index on `family`: **0/9 rows uniquely identified**, worst case 3 rows share a value. Index on `plan`: 9/9. A two-condition question returns 4 rows by similarity of which **0 satisfy both conditions**, spread across 0.795–0.791; the correct row is not in the top four. The parsed filter returns 1 row, matching a direct scan. Matching the event type literally silently drops that condition and returns 2 rows; folding case and punctuation returns 1. Four prose chunks carry 1416 characters ≈ 354 tokens per question |
| **05** | Server starts on a free loopback port. Blocking returns one body; streaming returns **5 events in 0.18s**. An empty `inputs` object gets HTTP 400 `app_unavailable` with a correct endpoint and a valid key. The probing client needs **4 requests** to find the declared key name. With the server stopped, the same loop makes 5 failed attempts and reports *"check the application configuration and API key"* |

### 11.2 Data

Everything under `data/` is written for this module, in English. The first four are
inputs the scripts read; the last two are written by a script on every run and are
not tracked, so they appear the first time you run 03 and 05:

| Path | What it is |
| :--- | :--- |
| `data/workflows/*.json` | Three workflow definitions — nodes, edges, typed ports, a batch body, a selector, two sub-workflow calls |
| `data/commission_plans.csv` | Nine rows, four columns — the table whose index column decides section 8.2 |
| `data/user_behavior_event.csv` | Twenty event rows across three identifiers and four days |
| `data/service_notes.txt` | The prose document no filter can answer |
| `data/review_feed/page{1,2,3}.atom` | The paged feed script 03 reads, written by the script itself and rewritten identically on every run, with one entry deliberately missing a field |
| `data/mock_platform_server.py` | Written by script 05 at run time, started as a subprocess, stopped before exit |

### 11.3 Running them

```bash
cd 09-lowcode-platforms
python 01_workflow_engine_from_spec.py          # offline, no key
python 02_llm_node_output_contract.py           # needs a chat model key
python 03_plugin_io_contract.py                 # offline, no key
python 04_table_knowledge_base_retrieval.py     # offline, local encoder
python 05_platform_api_protocol.py              # offline, starts a local server
```

Script 02 reads `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` or `OPENAI_API_KEY` from `.env`,
in that order, and backs off on a rate limit. Script 04 looks for
`BAAI/bge-small-en-v1.5` under any sibling module's `weights/` before downloading it, so a
repository that already has it pays nothing to run this one.
