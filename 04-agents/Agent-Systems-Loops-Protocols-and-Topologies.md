# Agent Systems: Loops, Protocols and Topologies

Seven scripts that take one idea apart: **an agent is a loop in which a model decides which
tool to call, how many times, and when to stop.**

Everything here is written against the plain OpenAI request format plus a `base_url`, so the
provider is a two-line change. Every number in this document came out of a real run; the
commands that produce them are at the end.

| Layer | Question it answers | Scripts |
| :--- | :--- | :--- |
| **Input side** | How does text get into the model, and how does a conversation persist? | `01`, `02` |
| **The loop** | What actually makes a model "use a tool"? | `03`, `04` |
| **Protocols** | How do tools and other agents get discovered at runtime? | `05`, `06` |
| **Topology** | Why does the same set of steps behave differently? | `07` |

The through line is that **none of this lives in the framework**. The loop is a prompt format,
tool selection is text matching, and an agent's behaviour is its graph. Each script is built to
make that visible by removing one piece and running again.

---

## 1. What an agent is, and when to build one

### 1.1 The definition, and where its weight sits

> An agent is a system that, given a goal, works out the rest itself.

The weight of that sentence is on **the rest**. A program that calls a model five times in a
fixed order is not an agent no matter how good the prompts are, because the order was decided
by whoever wrote it. What makes something an agent is that the intermediate steps are produced
at runtime, by the model, from what it observed.

The same idea in mechanical terms: **a model in a loop, taking actions against an environment
and reading the feedback**, until it decides it is done.

```
caller -> model -> action -> environment
             ^                   |
             +---- observation --+
                     |
                    stop
```

Everything in this module is a variation on that diagram. `03` builds it by hand, `04` hands it
to a framework, `05` and `06` change where the actions come from, and `07` changes the shape of
the path between them.

### 1.2 The four capabilities, and which ones appear here

| Capability | What it means | Where it shows up |
| :--- | :--- | :--- |
| **Planning** | Break a goal into steps that were not enumerated in advance | `03` chooses which tool to try next; `07`'s five nodes are an explicit decomposition |
| **Memory** | Carry context between steps and between calls | `01` (transcript replay); the ReAct scratchpad in `03` |
| **Tool use** | Reach outside the model for facts or effects | `03`, `04`, `05`, `06` |
| **Autonomy** | Decide the sequence rather than follow one | the whole distinction in §2 |

"Planning" is not one technique. In practice it decomposes into four prompt-level habits that
stack rather than compete: splitting a goal into sub-goals, thinking step by step before
answering, checking your own output, and revising after seeing results. `07` uses the first
(five named stages) and the fourth (a triage verdict that changes the route). `03` uses the
second and third inside a single `Thought` line.

### 1.3 When an agent is the wrong shape

Four questions worth answering before building one:

| Question | If the answer is bad news |
| :--- | :--- |
| **How complex is the task really?** | If the decision tree can be drawn, draw it. A workflow is cheaper and reproducible |
| **What is a run worth?** | Agents burn tokens by design. `03` spends two model passes per tool call before any answer exists |
| **Can it do the core sub-task at all?** | If the model cannot do the hard step reliably, wrapping it in a loop repeats the failure with a budget |
| **What does a wrong answer cost?** | Cheap to detect and undo is fine. Expensive and silent needs a human in the path |

The strongest case for an agent is a task where **the output is easy to verify**: code that has
to compile and pass tests, a query that either returns rows or errors, a diagnosis that can be
checked against a known outcome. Verifiability is what lets you run the loop unattended. Without
it, every run needs a person to read it, and the agent has not saved anything.

`04`'s step-limit demo is that principle as code: the question has no answer in the available
data, and what stops the run is not insight but a cap.

### 1.4 Three design rules the scripts follow

**1. Keep it small.** Agent code accumulates in a specific way: a wrong output leads to an extra
branch, the branch leads to another prompt, and neither is ever removed. Errors compound in the
same direction — a hallucinated intermediate result becomes the input to the next step. The two
levers that keep it small are the system prompt (say what the agent is and what it may call) and
the tool boundary (each tool does one thing and returns a string a model can read).

**2. Make the reasoning visible.** Every script here prints its trace, not just its answer:
`03` prints each step and the count of model passes, `04` prints every tool call and the first
line it returned, `07` prints the state after each node and the tokens spent. That is not
decoration; §7.4 and §14.3 are only arguable *because* the traces exist. A correct answer that
was guessed and a correct answer that was investigated look identical from the outside.

**3. Treat tools as the product.** The tool set, its descriptions, and its error strings are
where an agent's behaviour actually lives. §8 is entirely about that, and the two failure demos
in `03` and `04` are both tool-text failures rather than model failures.

A fourth, more uncomfortable rule follows from the first: **simplicity has to be defended by a
person**. Asking a model to fix agent code reliably produces more code — an extra guard, an
extra fallback, an extra branch — because adding is a safer-looking edit than removing.

---
## 2. Workflow or agent: the one question that decides the design

| Situation | Shape of the work | Build |
| :--- | :--- | :--- |
| The steps are known and their order never changes | Fixed sequence | **Workflow** |
| The next step depends on what the last one returned | Branching, looping, re-deciding | **Agent** |

The distinction is not "does it have tools". A workflow can call five tools and an agent can
call one. The distinction is **who fixes the order**:

```
workflow:  tool A -> tool B -> tool C          (the order is in the code)
agent:     model picks -> observe -> model picks again   (the order is generated)
```

Three signals that the work is agent-shaped:

1. The tool call order is not fixed in advance.
2. Intermediate results have to be judged before the next step is chosen.
3. There are conditional branches that depend on content, not on configuration.

Both ends of this are implemented here, deliberately with the **same five nodes**:
`07_langgraph_topologies.py` runs them once as a straight pipeline and once behind a triage
node that decides which path to take. The steps are identical; only the edges differ, and the
measured cost differs by an order of magnitude (see §14).

The cost of choosing "agent" is that behaviour stops being reproducible. `03` shows the same
loop taking one tool call for a question it can answer and five for a question it cannot, and
`04` shows the same four tools reaching a correct answer in six calls and a wrong answer in
fourteen once their descriptions get vague. That variance is the feature, and it is also the
bill.

---

## 3. Templates and memory: what actually goes into the request

`01_prompt_templates_and_memory.py`

### 3.1 A template is string formatting with a declared input list

```python
template = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
template.format(product="colorful socks")
# What is a good name for a company that makes colorful socks?
```

The value of declaring `input_variables` is not the formatting. It is that a missing variable
fails while the prompt is still a local object, rather than arriving at the model as the literal
text `{product}` and coming back as a confidently wrong answer.

### 3.2 Splitting the instruction from the payload

```python
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You translate {source_language} into {target_language}. Reply with the translation only."),
    ("human", "{text}"),
])
```

renders to two messages, not one string:

```
[system] You translate English into French. Reply with the translation only.
[human] I love programming.
```

Chat models take a list of messages. Flattening an instruction and a payload into one block
makes the model work out which half is the instruction; keeping them in separate roles means
the standing instruction stays fixed while only the human message changes.

### 3.3 The parser is not optional decoration

```python
chain = ChatPromptTemplate.from_messages([...]) | model | StrOutputParser()
```

Measured, on the same input:

```
with parser:    "J'adore la programmation."
without parser: AIMessage carrying "J'adore la programmation."
```

Drop the parser and the next step in the chain receives a message wrapper where it expected a
string. This is the most common surprise when composing chains, and it is invisible until
something downstream tries to use the value.

### 3.4 Memory is a caller habit, not a model feature

The model is stateless: each request is judged only on the messages it carries. So "memory"
means keeping the transcript and prepending it next time. Here a checkpointer does the keeping,
`MessagesPlaceholder` marks the slot it is poured into, and a thread id decides which transcript
a call belongs to.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant. Answer in one short sentence."),
    MessagesPlaceholder(variable_name="messages"),
])
builder = StateGraph(MessagesState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
graph = builder.compile(checkpointer=InMemorySaver())
```

The test is a follow-up that names nothing:

```
user: I am building a small tool that renames photo files by date.
bot:  Use EXIF metadata to extract the capture date and format it into the
      filename (e.g., YYYY-MM-DD_HHMMSS.jpg).
user: What should I call it?
bot:  Call it "PhotoDateRename" or "DateRenamer".
```

The checkpointer then holds four messages, in order: human, ai, human, ai. Send the same
follow-up with no transcript in front of it and the same model answers:

```
user: What should I call it?
bot:  Call it whatever feels right to you.
```

**Same model, same question, one difference: the first turn is gone.** Memory lives in the
request payload and nowhere else.

That also sets the ceiling. Every stored turn is re-sent on every later call, so a transcript
that grows without bound grows the bill and eventually exceeds the context window. The usual
answers are to keep the last K turns, to summarise older ones, or to store everything in a
vector store and retrieve only what is relevant. None of those change the mechanism; they only
change what gets replayed.

---

## 4. Composition: what the pipe operator actually buys

`02_lcel_composition.py`

The pipe operator joins objects whose outputs and inputs line up, so a composed chain takes one
dict in and returns one string out. Five things it gives you that hand-written glue does not:

### 4.1 Sequential composition and retries

Three model calls chained so each consumes the previous answer. Then the same flaky step
retried twice, once by hand and once by the framework:

```
  retrying a flaky step by hand:
    attempt 1 failed: transient failure on attempt 1
    attempt 2 failed: transient failure on attempt 2
    manual loop: 'step succeeded' after 3 attempts
  retrying the same step with .with_retry():
    .with_retry(): 'step succeeded', no loop or except clause written
```

Same outcome, and the second one is a method call rather than a loop plus an `except` clause
that has to be written again at every step that can fail transiently.

### 4.2 Plain functions become chain steps

```python
LOCAL_STEPS = {
    "analyse": RunnableLambda(analyse_text),
    "convert": RunnableLambda(convert_data),
    "process": RunnableLambda(process_text),
}
```

`RunnableLambda` lifts an ordinary Python function into something composable. The three
functions here cost nothing and always return, so the composition stays visible instead of
hiding behind another model call:

```
analyse: words=26 characters=140 sentiment=negative (hits 1+/2-)
convert: [ ... (3 rows)
process: 3 lines
```

### 4.3 Parallel branches, measured

Five independent analyses of one review, run as a `RunnableParallel` and then serially:

```
parallel 1.39s vs serial 4.29s (5 branches)
```

A 3.1x speedup for a one-line structural change. The branches do not depend on each other, so
running them in sequence was only ever an artefact of how the code was written.

### 4.4 Routing that stays composable

```python
router = RunnableBranch(
    (lambda payload: payload["content"].count("\n") >= 2, LOCAL_STEPS["process"]),
    (lambda payload: "," in payload["content"], RunnableLambda(...)),
    RunnableLambda(...),   # default
)
```

Measured on three inputs:

```
First line of the note         -> 3 lines
name,age                       -> looks like tabular data, use the converter
hello                          -> short text, 5 characters, left as is
```

`RunnableBranch` is an if/elif written as data. The reason that matters is that the branch is
still a runnable, so it can be piped into something further:

```
piping the router into a further step:
  The routing verdict makes sense because the input contains exactly three
  distinct lines, matching the specified "3 lines" criterion.
```

The script then walks up to the edge of the coercion rule and shows both sides of it:

```
a plain function piped into that same Runnable still works (LangChain coerces it):
  RunnableSequence, no TypeError
but two plain functions piped together have no Runnable to coerce through:
  TypeError: unsupported operand type(s) for |: 'function' and 'function'
```

A bare function works on one side of `|` because the other side is already a runnable and
triggers coercion. Two bare functions have nothing to coerce through. This is why the branch
above is built from `RunnableLambda` objects rather than plain callables.

### 4.5 Streaming versus blocking, measured

```
streamed: first chunk after 0.29s, 103 chunks total
blocking: nothing visible until 2.04s, when the full answer arrives at once
```

Identical total work, 7x difference in time to first visible output. For anything a person
watches, that difference is the entire perceived latency.

---

## 5. ReAct: reasoning and acting in one loop

Before the implementation, the idea. `03` and `04` are both this pattern; knowing what it
replaces is what makes their failure demos readable.

### 5.1 The claim

> *ReAct: Synergizing Reasoning and Acting in Language Models* (2022),
> <https://arxiv.org/abs/2210.03629>

The claim is that interleaving reasoning with actions beats doing either alone: reasoning keeps
the actions goal-directed, and actions keep the reasoning tied to facts the model did not
generate. The observation behind it is mundane — people doing a multi-step task think between
steps — and the contribution is that the pattern is written into the prompt format rather than
into a model or an algorithm.

### 5.2 Four ways to answer the same question

The paper's example asks which device, besides the Apple Remote, can control the program the
Apple Remote was first designed to interact with.

| Approach | What it does | Answer | |
| :--- | :--- | :--- | :---: |
| **Standard** | Answers from memory | iPod | ✗ |
| **Chain of thought** | Reasons step by step, no lookups | iPhone, iPad, iPod Touch | ✗ |
| **Act-only** | Searches repeatedly, no reasoning | yes | ✗ |
| **ReAct** | Alternates thought and search | keyboard function keys | ✓ |

**The three failures are more instructive than the success:**

- **Standard** has no process at all to inspect.
- **Chain of thought** fails on a *fact*, not on logic: it decides the remote was designed for
  Apple TV, which is invented, and every later step is sound reasoning from a false premise.
  A clean chain of reasoning from a wrong start is the hardest kind of wrong answer to spot.
- **Act-only** retrieves the right page and then cannot use it. It has observations and no
  mechanism for connecting them to the goal, so it terminates on the first thing that looks
  like an answer.

ReAct's winning step is a `Thought` that does the connecting: the observation lists two ways to
control the program, and the reasoning removes the one the question already named.

This maps directly onto the runs in §7. The clear-description run investigates and then
*excludes* evidence that does not fit; the vague-description run has observations and no thread
tying them to the question, and stops at the first anomaly it sees. Same model, and the
difference is whether the reasoning had anything to reason over.

### 5.3 What it is not

**Not few-shot prompting.**

| | Few-shot examples | ReAct |
| :--- | :--- | :--- |
| What it is | Sample answers placed in the context | A control loop |
| What it fixes | The shape or content of one answer | How a problem gets solved over several steps |
| Typical use | Patching a category of wrong answers | Open-ended problems needing lookups |

Adding a worked example is a patch on a symptom. It does not give the model a way to act.

**Not chain of thought.** Chain of thought is a fixed sequence decided by the author: write the
title, then the body, then the conclusion. ReAct decides the sequence while running. The
tempting objection — "an action is just another link in a chain" — is structurally true and
misses the point: fixing the chain in advance is exactly what removes the ability to adapt, and
that ability is the only thing you are paying the extra model calls for.

### 5.4 Where it fits, and where it does not

**Fits:** questions needing several lookups whose number is unknown in advance; answers that
need a verification step; work that combines several tools.

The canonical shape is a search that fails and has to be retried differently:

```
task: find the current state of X
  search("X") -> nothing useful
  search("X, different phrasing") -> the answer
```

No fixed workflow expresses that, because the number of attempts and the second phrasing are
both produced at runtime. `03`'s out-of-scope question is exactly this, ending in an honest
refusal after five attempts.

**Does not fit, or needs care:**

- **It has no natural stop.** A step cap is mandatory, not defensive; `04` demonstrates a run
  that ends only because the cap exists.
- **It multiplies model calls.** Every step is a full request carrying the whole transcript, so
  cost grows with the square of the conversation, not linearly.
- **It is only as good as its tools.** A tool that returns nothing useful gives the loop nothing
  to reason over.
- **Reasoning models fight the format.** A model that already produces long internal deliberation
  tends to write past the structure ReAct asks for; a plain instruction-following model holds
  the format better.

### 5.5 The safety question, answered structurally

Once a model can call tools, "what if it is asked to do something harmful" stops being about
refusals and becomes about the tool surface. The structural answer is to make review a step in
the chain rather than a hope about the model: a check before the call, a check on the result,
and human confirmation for anything destructive.

That is the same shape as the schema and auth checks in `06`: the provider does not trust that
the caller read the card, it validates and rejects. **Anything a model is only asked to respect
is not a control.**

---
## 6. The ReAct loop, written by hand

`03_react_loop_from_scratch.py` uses no agent framework at all. Only a chat completion call, a
regex, and a `while` loop. It exists to show that the four things a framework does for you are
each one line of text handling.

### 6.1 The four things

1. **Render the tool names and descriptions into the prompt as plain text.**
2. **Stop generation at `Observation:`** so the model cannot invent tool output.
3. **Parse the reply** into either an action to run or a finished answer.
4. **Append the real tool result** and call the model again with the longer transcript.

### 6.2 The prompt is the whole mechanism

```
You answer questions about a fund compliance rule book.

You can use these tools:
{tools}

Use exactly this format:

Question: the question you must answer
Thought: what you need to do next
Action: one of [{tool_names}]
Action Input: the input for that tool
Observation: the result the tool returned
... (Thought/Action/Action Input/Observation may repeat)
Thought: I now know the final answer
Final Answer: the answer for the user

If the rule book does not cover the question, say so plainly in the Final Answer
instead of inventing a rule.

Question: {question}
```

`{tools}` and `{tool_names}` are filled from an ordinary dict:

```python
TOOLS = {
    "search_rules":  (search_rules,  "Search the rule book by keywords. Input: two or three keywords."),
    "list_category": (list_category, "List the rules in one category. Input: one of eligibility, supervision."),
    "read_rule":     (read_rule,     "Read the full text of one rule. Input: a rule id such as R-001."),
}
```

The knowledge base is four rules (`R-001` to `R-004`) in two categories, `eligibility` and
`supervision`. Two categories are enough to force a real choice: one tool searches text, another
lists a whole category, a third reads one rule by id, and the model has to decide which fits.

### 6.3 The stop sequence is what makes it a loop

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0,
    stop=["Observation:"],
)
```

Without it the model happily continues past its own `Action` and writes an `Observation` too,
hallucinating the tool result, and no real code ever runs. The stop sequence turns one long
piece of generated text into a loop the program controls.

### 6.4 The transcript is turns, not one growing block

The model's `Thought`/`Action` goes back as an assistant message and the tool result follows as
a user message. This is not cosmetic. Handed a half-finished block of text, the model treats it
as a document to continue, restarts it, re-types the original question and its own first
`Thought`, and that stale `Thought` then sits after the newest `Observation` where the model
reads it as the latest line and reissues the action it already ran. Splitting the same content
into turns removes the ambiguity about which line is newest.

### 6.5 Three fallbacks sit in front of the regex

Each one exists because a model really does reply that way:

| Reply shape | Handling |
| :--- | :--- |
| Contains `Final Answer:` | Normal completion |
| Says the rule book does not cover it, with no `Action:` | Treated as final; the model dropped the format mid-way |
| Long prose with neither `Action:` nor `Final Answer:` | Treated as final; prose is not a malformed step |
| Matches `Action:` / `Action Input:` | Run the tool, take **only the first line** of the input |
| None of the above | The only genuine parse failure |

The "first line only" detail matters because the input regex is greedy across newlines: if the
model keeps writing after the input line and before the stop sequence cuts it off, that trailing
text would otherwise ride along as part of the tool input.

**The format is requested by a prompt, not guaranteed by a protocol.** The parser has to be
ready to catch replies that went off-format but are still useful.

### 6.6 What the loop does, measured

**A question that needs one lookup:**

```
question: What is the minimum a securities fund must raise before it closes?
  step 1: search_rules('minimum raise securities fund') -> [R-001] Who qualifies as an eligible investor?...
  transcript: 3 messages
  passes: 2, tool calls: 1
answer: A securities fund may not close below 10,000,000 in committed capital. Venture and
        growth funds are not subject to this fixed floor and are instead governed by the
        fund agreement.
```

Two model passes for one tool call: one to choose the action, one to turn the observation into
an answer. That ratio is the loop's fixed overhead.

**A question the rule book does not cover:**

```
question: What tax rate applies to carried interest for this fund?
  step 1: search_rules('carried interest tax') -> No rule matches 'carried interest tax'.
  step 2: search_rules('tax rate')             -> No rule matches 'tax rate'.
  step 3: list_category('eligibility')         -> [R-001] Who qualifies as an eligible investor?
  step 4: list_category('supervision')         -> [R-003] What risk reserve must a manager hold?
  step 5: search_rules('carried interest')     -> No rule matches 'carried interest'.
  passes: 6, tool calls: 5
answer: The rule book does not cover tax rates on carried interest. It only contains rules
        about investor eligibility (R-001, R-002) and supervision (R-003, R-004), with no
        tax-related provisions.
```

Five tool calls to establish an absence, versus one to establish a fact. Note what made the
correct refusal possible: the tools return **plain-language misses** (`No rule matches ...`)
rather than empty strings or exceptions, so the model can read the miss, try a different tool,
and eventually enumerate what the rule book *does* contain before answering.

### 6.7 Delete the tool list and watch it degrade

Step 5 swaps in the same prompt with the tool block removed and `Action: one of [...]` replaced
by `Action: the tool to use`. The functions are still registered. Nothing else changes:

```
  step 1: Search rule book for "minimum raise" or "securities fund closing"(...) -> No tool named ...
  step 2: Search rule book('minimum raise securities fund closing')              -> No tool named 'Search rule book'.
  passes: 3, tool calls: 2
answer: The rule book does not cover this question, or I do not have access to the rule book
        content to provide an answer. I cannot invent a minimum raise requirement.
```

**The model invented tool names in English prose and called them.** The registry was complete
and the code looked finished; the only missing piece was the text. A model knows a tool exists
if and only if its name and description appear in what it was given.

---

## 7. Handing the same loop to a framework

`04_tool_agent_diagnosis.py`

### 7.1 The declaration replaces the prompt block

```python
@tool
def ping_host(hostname: str) -> str:
    """Check whether a host answers on the network, and report the round trip time."""
```

The decorator reads the signature and the docstring and turns them into a tool schema. **That
schema, not the Python function, is what the model sees**, which is why an argument with no type
hint or a docstring that omits what the input should look like degrades tool selection while the
code still runs fine. Printed back out:

```
resolve_host(hostname):  Resolve a hostname to an address. Use this before assuming a host exists.
ping_host(hostname):     Check whether a host answers on the network, and report the round trip time.
check_interface(name):   Report the state of one local network interface, such as eth0 or eth1.
search_logs(keyword):    Search the recent service log for a keyword and return the matching lines.
```

The simulated environment is built so that no single lookup solves anything: the resolver knows
names the ping table cannot reach (`billing.internal` resolves to `10.0.4.37` but never answers),
one interface is up and one is down, and the log holds five lines of which three are noise. A
fixed seed keeps the latencies identical between runs, because a diagnosis demo whose numbers
move cannot be discussed after the fact.

### 7.2 What the framework gives back

`create_agent` compiles to a graph, so the return value is the whole message list rather than a
string. Reading the `ToolMessage` entries out of it is how the trace below is printed, and it is
also the only way to see **which probe ran, in what order** rather than just the conclusion.

```
question: Checkout keeps failing with connection errors since 14:00. What is broken?
  call 1: search_logs(['checkout'])        -> No log line contains 'checkout'.
  call 2: search_logs(['connection'])      -> 14:02:11 ERROR pool: connection to billing.internal:5432 refused
  call 3: resolve_host(['billing.internal'])-> billing.internal resolves to 10.0.4.37.
  call 4: ping_host(['billing.internal'])  -> billing.internal (10.0.4.37) does not answer: request timed out.
  call 5: search_logs(['billing'])
  call 6: check_interface(['eth0'])        -> eth0 is up, address 10.0.4.9, gateway 10.0.4.1.
  tool calls: 6
answer: Failing component: the billing database server at billing.internal (10.0.4.37) is
        down or unreachable. Evidence: logs show connection refused from 14:02; the name
        resolves correctly, so DNS is fine; eth0 is up with a valid gateway, so the local
        network is fine; ping to 10.0.4.37 times out.
```

Six calls, and the sequence is a real investigation: search first, follow the hostname the log
named, separate "does not resolve" from "does not answer", then rule out the local interface.
Nothing in the code specifies that order.

### 7.3 A budget, and what hitting it looks like

```python
run_agent(question, TOOLS, recursion_limit=4)
```

Given a question the tools cannot answer (`Which host consumed the most bandwidth this week,
and by how many gigabytes?`), the agent spends its whole budget searching for words that are
not in the log:

```
  call 1: check_interface(['eth0'])
  call 2: search_logs(['bandwidth'])  -> No log line contains 'bandwidth'.
  call 3: search_logs(['traffic'])    -> No log line contains 'traffic'.
  call 4: search_logs(['bytes'])      -> No log line contains 'bytes'.
  call 5: search_logs(['usage'])      -> No log line contains 'usage'.
  call 6: search_logs(['GB'])         -> No log line contains 'GB'.
  tool calls: 6
answer: none, the agent hit its step limit first
```

An agent without a step cap does not stop on its own when the answer does not exist. The cap is
not a safety net for bugs; it is the ordinary termination condition for a loop whose exit is
decided by a model.

### 7.4 Change only the descriptions

Step 5 redefines the same four functions with the same names and the same behaviour, and
replaces their docstrings with `"""Do a lookup."""`, `"""Do a check."""`, `"""Do a check."""`
and `"""Do a search."""`. The question is identical to §7.2:

```
  call 1: search_logs(['checkout'])       -> nothing found
  call 2: search_logs(['connection error'])-> nothing found
  call 3: ping_host(['checkout'])          -> Cannot ping checkout.
  call 4: resolve_host(['checkout'])       -> checkout does not resolve.
  call 5: check_interface(['checkout'])    -> no interface checkout
  call 6: search_logs(['14:00'])           -> nothing found
  call 7: resolve_host(['gateway'])        -> gateway does not resolve.
  call 8: resolve_host(['db'])             -> db does not resolve.
  call 9: resolve_host(['api'])            -> api does not resolve.
  call 10: check_interface(['eth0'])       -> eth0 is up
  call 11: check_interface(['eth1'])       -> eth1 is down
  call 12: search_logs(['eth1'])           -> nothing found
  call 13: search_logs(['down'])           -> nothing found
  call 14: ping_host(['eth1'])             -> Cannot ping eth1.
  tool calls: 14
answer: Failing component: the eth1 network interface is down.
```

**14 calls instead of 6, and the conclusion is wrong.** `eth1` being down has nothing to do with
the incident; the run reached it by guessing hostnames (`checkout`, `gateway`, `db`, `api`) that
do not exist, and then latched onto the first thing that looked abnormal.

Two things changed together, and both come from the same edit:

- **the descriptions**, which are the model's only basis for choosing a tool, and
- **the return strings**, which lost the detail that let the clear run separate "does not
  resolve" from "resolves but does not answer".

That second one matters more. `"answers"` / `"request timed out"` is a shorter string
than `billing.internal (10.0.4.37) does not answer: request timed out.`, and the shortening
removed exactly the evidence the diagnosis depended on.

⇒ **Tool text is not documentation. It is the runtime input to every decision the agent makes,
and it is billed per wrong guess.**

---

## 8. Designing tools

Two of the seven scripts fail on purpose, and both failures are tool-text failures. This chapter
collects what those runs imply.

### 8.1 A tool is three things, and only one of them is code

| Part | Who reads it | What it decides |
| :--- | :--- | :--- |
| **Name** | The model | What it writes when it wants this tool |
| **Description / schema** | The model | **Whether this tool gets chosen at all** |
| **Function** | The runtime | What actually happens |

In `03` the three parts are explicit, because the prompt is assembled by hand:

```python
TOOLS = {
    "search_rules": (search_rules, "Search the rule book by keywords. Input: two or three keywords."),
}
```

In `04` they are derived from the function itself:

```python
@tool
def search_logs(keyword: str) -> str:
    """Search the recent service log for a keyword and return the matching lines."""
```

Different mechanics, same three parts. The convenience of the decorator hides a trap worth
naming: **the docstring is now production input**. Editing it to be terser is editing runtime
behaviour, and nothing in a type checker or a test suite will notice.

### 8.2 The description is the whole selection mechanism

There is no matching algorithm, no embedding of tool names, no registry lookup by capability.
The model reads a list of names and sentences, and writes one of the names. So:

- A description that does not say **what the tool is for** produces guessing. `04`'s vague run
  spends five of fourteen calls inventing hostnames.
- A description that does not say **what the input looks like** produces malformed arguments.
  `03` spells out `Input: a rule id such as R-001` for exactly this reason.
- A description that overlaps another tool's produces coin-flips between them, and the choice is
  not stable between runs.

The corollary is that **improving an agent usually means editing text, not code**. Before
changing a model or adding a framework, read the tool list back as the model receives it — `04`
prints exactly that as its first step — and ask whether a stranger could pick correctly from it.

### 8.3 Return values are control flow

When the caller is a model, what a tool returns on failure decides whether the run recovers.
The two variants in this module are a clean controlled comparison:

| | Return on a miss | What the run did next |
| :--- | :--- | :--- |
| `03` | `No rule matches 'tax rate'.` | Switched tools, listed both categories, refused honestly |
| `04` clear | `billing.internal (10.0.4.37) does not answer: request timed out.` | Separated "unresolvable" from "unreachable", found the real cause in 6 calls |
| `04` vague | `nothing found` / `request timed out` | Guessed four hostnames, latched onto an unrelated interface, 14 calls, wrong answer |

Three rules follow:

1. **Never raise into the loop.** An exception ends the run; a sentence lets the model adapt.
2. **Never return empty.** `""` and `None` are indistinguishable from "the tool broke".
3. **Say what was missing, not just that something was.** `No rule matches 'tax rate'` tells the
   model the query was the problem; `nothing found` tells it nothing.

### 8.4 Two kinds of tools

| Kind | Purpose | Example here |
| :--- | :--- | :--- |
| **Lookup** | Fetch information the model cannot have | `search_logs`, `read_note`, `search_rules` |
| **Process** | Carry out a defined procedure, possibly with effects | `resolve_note_path`-guarded reads, the availability task in `06` |

The distinction matters when writing descriptions: a lookup tool should say what corpus it
searches; a process tool should say what it changes and under what conditions.

Both differ from putting facts directly in the prompt. A tool can branch, validate, refuse, and
be called several times with different inputs. Pasted context can only sit there.

### 8.5 How many tools is too many

Tool descriptions are prompt text, and prompt text is finite. A rough accounting: a rich tool
description with a schema runs to a few thousand tokens once expanded; a context window in the
low hundreds of thousands therefore holds a few dozen before the tool list alone crowds out the
conversation. Reports of agents wired to dozens of servers behaving erratically are consistent
with that arithmetic: nothing broke, the model simply cannot hold a catalogue that size in
view while also holding the task.

The scripts here stay well inside that limit on purpose: three tools in `03`, four in `04`,
three in `05`. When a tool list genuinely has to grow, the two standard moves are to route
between smaller sets rather than merging them (`07`'s triage, applied to tools instead of
depth), or to split the work across agents that each keep a small list and call one another
(`06`'s shape).

---
## 9. MCP: what the protocol standardises

### 9.1 The problem it removes

Before a protocol, every pairing of a client and a data source needed its own adapter: a
bespoke integration per source, rewritten for the next client. The Model Context Protocol
(introduced by Anthropic in late 2024) standardises the shape of that connection so a tool
provider writes one server and any compliant client can use it. The usual analogy is a universal
port replacing a drawer of proprietary cables.

The practical definition is narrower and more useful:

> **A tool exposed over a protocol, whose implementation you never see and never import.**

That is the difference from a function in your own codebase. `05` demonstrates it literally: the
client half of the file cannot call the server half directly, because the server runs in a
separate process and the only channel between them is the protocol.

### 9.2 Three roles

| Role | What it is | In `05` |
| :--- | :--- | :--- |
| **Host** | The application the model runs inside | The script's `main()` |
| **Client** | The component that speaks the protocol and holds a session | `ClientSession` over `stdio_client` |
| **Server** | The process that publishes capabilities | The same file, re-launched with `--serve` |

The separation is what makes substitution cheap: swapping the model does not touch the server,
and swapping the server does not touch the host.

### 9.3 Three capabilities, of which one is used

| Capability | Purpose |
| :--- | :--- |
| **Tools** | Operations the model can invoke |
| **Resources** | Structured data offered as context |
| **Prompts** | Predefined instruction templates |

In practice **Tools is the one that matters**. The three published in `05` are all tools, and
servers in the wild commonly implement tools and nothing else, answering "method not found" to a
resources listing. Worth knowing before designing around the other two.

### 9.4 Against plain function calling

| | Function calling | MCP |
| :--- | :--- | :--- |
| Nature | A model capability | A protocol between processes |
| Scope | The functions in this codebase | Any server that speaks it |
| Reuse | Rewritten per application | Written once, used by many clients |
| Cost | A function call | A process or network boundary per call |
| Discovery | Compile time | Runtime |

The choice is not about sophistication. The usable rule:

> **Are you exposing this to callers you did not write?** If not, a local function is faster to
> write and cheaper to run. If yes, the protocol is what stops you writing an adapter per caller.

`04` and `05` are the same loop under both answers — local `@tool` functions in one, advertised
tools in the other — which is why the client code in `05` is so short. The model does not know
the difference; only the transport does.

### 9.5 Two transports

| | **stdio** | **HTTP / SSE** |
| :--- | :--- | :--- |
| Shape | Launch a process, speak over its pipes | Connect to a URL over the network |
| Lifetime | Bound to the child process | A long-lived connection |
| Suits | Local capability: files, local databases, machine state | Hosted services used by many clients |
| Constraint | **stdout belongs to the protocol** | The URL usually carries the credential |

`05` uses stdio, which is why the server half contains no `print` at all. On the hosted side the
warning is different: for many hosted servers, the connection URL *is* the credential, so it
belongs in an environment variable and never in a committed config file or a screenshot.

### 9.6 The security consequence of an open ecosystem

Anyone can publish a server. That is the source of the ecosystem's value and of its main risk:
a server that looks official and is not, sitting in front of anything sensitive. The mitigations
are ordinary supply-chain hygiene — verify the publisher, prefer servers you or your organisation
run, and treat a third-party server as an external dependency with the access it is granted,
not as a library.

The same logic runs in the other direction for a server you write. `05`'s path check is there
because **the filename in a tool call is attacker-influenced input** the moment the model is
exposed to untrusted text:

```python
path = (NOTES_DIR / filename).resolve()
if not path.is_relative_to(NOTES_DIR.resolve()) or path.suffix != ".txt":
    return None
```

A server that skips this check will happily read whatever a well-phrased request asks it to.

---
## 10. MCP: both halves of the protocol in one file

`05_mcp_client_and_server.py`

The Model Context Protocol is usually shown from one side: either you write a server, or you
point a client at somebody else's. This script is both, in one file, so the boundary between
them is visible rather than assumed. Run it and it starts **itself** as a subprocess with
`--serve` and then talks to that subprocess over stdio.

### 10.1 The server publishes three functions

```python
server = MCPServer("notes")

@server.tool()
def list_notes() -> str:
    """List the note files available, one filename per line."""

@server.tool()
def read_note(filename: str) -> str:
    """Read one note file in full. Input: a filename from list_notes."""

@server.tool()
def count_words(filename: str) -> int:
    """Count the words in one note file. Input: a filename from list_notes."""
```

Only these three are exposed. Everything else in the file stays invisible to the client, which
is the point of the boundary: the caller gets an interface, not a library.

Two details in this half are easy to get wrong:

**Nothing prints to stdout.** A stdio server speaks the protocol on stdout, so one stray `print`
corrupts the stream and the client fails at the handshake with a parse error that says nothing
about the print. Servers log to stderr or to a file.

**The filename is model-supplied input, so it is checked, not trusted:**

```python
path = (NOTES_DIR / filename).resolve()
if not path.is_relative_to(NOTES_DIR.resolve()) or path.suffix != ".txt":
    return None
```

`NOTES_DIR / filename` alone confines nothing: `..` segments walk back out of the directory,
and an absolute path replaces `NOTES_DIR` entirely rather than being appended to it. That is how
the `/` operator is defined, not a bug. Resolving the path and checking it is still inside the
directory is what actually limits reads.

### 10.2 Handshake and discovery, measured

```
--- 2. Handshake ---
  server: notes, protocol 2025-11-25

--- 3. What the server advertises ---
  list_notes(): List the note files available, one filename per line.
  read_note(filename): Read one note file in full. Input: a filename from list_notes.
  count_words(filename): Count the words in one note file. Input: a filename from list_notes.
```

The client did not know those three names before it asked. It learned them at runtime, from the
server, along with the JSON Schema for each argument.

### 10.3 Translating the schemas is a rename, not a translation

```python
{
    "type": "function",
    "function": {
        "name": item.name,
        "description": item.description or "",
        "parameters": item.input_schema,
    },
}
```

Printed for `read_note`, this is what reaches the chat API:

```json
{
  "type": "function",
  "function": {
    "name": "read_note",
    "description": "Read one note file in full. Input: a filename from list_notes.",
    "parameters": {
      "properties": { "filename": { "title": "Filename", "type": "string" } },
      "required": ["filename"],
      "type": "object",
      "title": "read_noteArguments"
    }
  }
}
```

Both sides already speak JSON Schema. The protocol calls the field `input_schema`, the chat API
calls it `parameters`, and that is the whole adapter. **Seeing how little happens here is the
useful part**: a server written for one client works with any model that accepts tools.

### 10.4 A tool-calling loop where every call crosses a process boundary

```
question: Which note explains why the checkout outage took so long to diagnose,
          and what was the fix?
  list_notes({}) -> incident-2024-03-12.txt
  read_note({'filename': 'incident-2024-03-12.txt'}) -> Incident summary: checkout failures
  read_note({'filename': 'onboarding.txt'})          -> Onboarding notes for new engineers
  read_note({'filename': 'release-checklist.txt'})   -> Release checklist for the billing service
rounds: 3
```

The three `read_note` calls arrived in **one** model turn, as three `tool_calls` in a single
reply. They are dispatched with `asyncio.gather`, so the process boundary is paid for once
concurrently instead of three times in sequence.

One call, printed exactly as it crossed:

```
request  : {"name": "list_notes", "arguments": {}}
response : content='incident-2024-03-12.txt'
           structured={'result': 'incident-2024-03-12.txt\nonboarding.txt\nrelease-checklist.txt'}
```

That is the entire mechanism. The model emitted a name and an arguments object; the client sent
them across; the server returned content. **A tool call is a JSON message, and MCP is the
agreement about what that message looks like.**

### 10.5 What the protocol buys, and what it does not

The loop in `04` and the loop here are the same loop. The difference is where the tools come
from: in `04` they are functions in the same file, here they are advertised by a process the
client did not have to be written against. That is the whole trade:

| | Functions in your own code | Tools over MCP |
| :--- | :--- | :--- |
| Discovery | Import them | Ask the server at runtime |
| Coupling | Caller and tool share a codebase | Caller has a transport and a schema |
| Cost | A function call | A process boundary per call |
| When it pays | The tool is yours and stays yours | The tool is somebody else's, or is reused by several clients |

The protocol solves distribution, not capability. Nothing here made the model better at reading
notes; it made the note reader usable by any client that speaks the protocol.

---

## 11. A2A: what agent-to-agent adds

### 11.1 A different question from MCP

MCP answers *how does an agent reach a tool*. A2A (published by Google in April 2025) answers
*how does an agent reach another agent* — one with its own model, its own private data and its
own judgement, running somewhere you do not control.

The compact version of the pair:

> **MCP is the manual for a tool. A2A is the phone book for colleagues.**

Or, in terms of what crosses the boundary: MCP moves a **call and a result**; A2A moves a
**task and an artifact**, and the party on the far side is allowed to think about it.

### 11.2 The principles that shape it

| Principle | Consequence in the design |
| :--- | :--- |
| Agents collaborate as agents | Messages are content, not remote-procedure signatures |
| Reuse existing standards | HTTP, JSON, SSE — nothing exotic to deploy |
| Secure by default | Authentication is declared and expected, not bolted on |
| Long-running work is normal | A task has states, and can pause for more input |
| Multi-modal | Parts can be text, files or structured data |

`06` implements the first three directly: a JSON document over HTTP, a task submitted to an
endpoint, and a bearer token the provider actually enforces.

### 11.3 The vocabulary

| Term | Meaning |
| :--- | :--- |
| **Agent Card** | The public description at `/.well-known/agent.json`: name, endpoints, input schema, auth |
| **Task** | The unit of work, with an id and a lifecycle |
| **Message** | What passes back and forth while a task runs |
| **Part** | A piece of a message: text, file, or data |
| **Artifact** | The task's output |

The nesting is worth holding on to: a **task** is the job, **messages** are the traffic during
it, **parts** are what messages are made of, and the **artifact** is what you keep. In `06` the
line `reply["artifact"]["rooms"]` is exactly this term, and the card, the task id and the
completed status are all visible in the same response.

### 11.4 The five-step flow

| Step | What happens |
| :--- | :--- |
| **1. Discover** | Fetch the card, learn the endpoints and the schema |
| **2. Submit** | Send a task, either for an immediate result or as a subscription |
| **3. Process** | The provider works, optionally streaming updates |
| **4. Clarify** *(optional)* | The task enters `input-required`; the caller sends more under the **same task id** |
| **5. Finish** | The task reaches `completed`, `failed` or `canceled` |

**Step 4 is the part an ordinary HTTP call has no room for.** A normal request succeeds or
fails; it cannot say "I need one more thing from you before I can continue" and stay the same
piece of work. `06` implements steps 1, 2, 3 and 5 — its provider answers immediately — but the
card's declared `sse_subscribe` endpoint is the seam where the streaming variant would attach.

### 11.5 The division of labour, as code

`06` keeps the split honest in a way that is easy to state and easy to violate:

- The provider **reports facts**: which rooms are free on a date for a given size.
- The caller **makes the decision**: whether the workshop happens.

The provider never learns what the rooms were wanted for, and the caller never sees the room
table. Either side can change its rules — new rooms, a different cancellation policy — without
telling the other. Push the decision into the provider and you no longer have two agents; you
have one agent and a remote function that guessed at your policy.

### 11.6 When one agent is enough

The honest ordering, given the state of the ecosystem:

```
one agent with a few tools  ->  more tools, still one agent  ->  several agents talking
```

Move right only when the reason is concrete. The usual concrete reason is the tool-count ceiling
from §8.5: past a few dozen tools a single agent's selection degrades, and splitting the
catalogue across agents restores a small, legible tool list to each of them. The other is
ownership — when the far side is written and operated by someone else, a protocol boundary is
the honest description of the relationship.

Against that, A2A is young, and its ecosystem is thinner than MCP's. **For production work today,
a well-designed tool surface goes further than a multi-agent topology**, which is why `06` is one
provider, one caller, and no framework.

---
## 12. A2A: discovering another agent and delegating to it

`06_a2a_agent_protocol.py`

The protocol above connects an agent to tools. This one connects an agent to **another agent**
that has its own private data and its own rules, with no shared codebase between them.

### 12.1 The capability card

```python
AGENT_CARD = {
    "name": "RoomAvailabilityAgent",
    "version": "1.0",
    "description": "Reports which rooms are free on a given date and for how many people.",
    "endpoints": {"task_submit": "/api/tasks/availability"},
    "input_schema": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "format": "date"},
            "attendees": {"type": "integer", "minimum": 1, "maximum": 60},
        },
        "required": ["date", "attendees"],
    },
    "authentication": {"methods": ["bearer"]},
}
```

Served at `/.well-known/agent.json`. A caller that has never seen this code should be able to
read this one document and learn the endpoint, the accepted inputs, and the authentication
scheme. That is the entire premise of runtime discovery.

### 12.2 The caller reads routes instead of hardcoding them

```python
card = discover(PROVIDER_URL)                       # GET /.well-known/agent.json
response = urllib.request.Request(
    f"{base_url}{card['endpoints']['task_submit']}",  # route comes from the card
    ...
    headers={"Authorization": f"Bearer {token}"},
)
```

Measured:

```
discovered RoomAvailabilityAgent v1.0
learned endpoint: /api/tasks/availability
learned auth:     ['bearer']
```

If the provider renames its endpoint to `/api/v2/availability` and updates its card, this caller
keeps working without a line changed. Hardcode the path and the two agents are coupled again,
at which point the protocol has bought nothing.

### 12.3 The provider reports facts; the caller owns the decision

The provider's room table is private. What crosses the wire is an artifact derived from it:

```json
{"task_id": "...", "status": "completed",
 "artifact": {"date": "2026-04-14", "attendees": 30,
              "rooms": [{"room": "Aspen", "seats": 40}]}}
```

The caller turns that into a decision that is its own:

```
2026-04-14 for 30: confirmed in Aspen (40 seats)
2026-04-15 for 30: cancelled, no room fits
2026-04-16 for 8:  cancelled, no room fits
```

Three different situations behind those lines: a room that fits, a date whose only room seats 12,
and a date with no rooms at all. **Whether a workshop goes ahead is not the provider's call**,
and keeping that split is what lets either side change its rules without renegotiating with the
other. Note also that there is no model in this script at all — delegation between agents is a
protocol question, not an intelligence question.

### 12.4 A declaration that is not enforced is decoration

Two negative cases, both driven by what the card promised:

```
--- 6. A task the card's schema forbids ---
  status 400: attendees must be an integer between 1 and 60
  the card said attendees max is 60

--- 7. A task without the bearer token the card declared ---
  status 401: missing or invalid bearer token
  the card declared authentication methods: ['bearer']
```

The card advertises a constraint and an auth scheme; the handler rejects violations of both.
A card that declared `bearer` while the endpoint waved everyone through would be worse than no
card at all, because callers would trust it.

---

## 13. Three architectures, and which script is which

The scripts are not three unrelated demos. They are the three classical agent architectures,
and knowing which is which is what turns §14's token table into a design argument.

| Architecture | Behaviour | Cost | Here |
| :--- | :--- | :--- | :--- |
| **Reactive** | Observe, decide, act, repeat. No plan beyond the next step | Cheapest, fastest | `03`, `04` |
| **Deliberative** | Model the situation, generate options, choose, then act | Slowest, deepest | `07`'s pipeline |
| **Hybrid** | A coordinator picks between the two per request | Adds one decision | `07`'s router |

### 13.1 Reactive

```
   observation -> decide next action -> act -> observation -> ...
```

There is no world model and no plan. Each step is chosen from what is on screen right now, which
is why it is fast and why it can walk in circles: nothing in the architecture remembers that the
last three attempts were variations of one idea.

`03` is a textbook reactive loop, and both of its behaviours follow from that. On the in-scope
question it answers in one tool call. On the out-of-scope question it makes five attempts —
two searches, two category listings, one more search — before concluding. A deliberative design
would have decided up front that "carried interest tax" is outside a rule book covering
eligibility and supervision. The reactive one has to discover that by bumping into it.

Reactive is the right default when the environment answers quickly and the cost of an extra probe
is low. `04`'s diagnosis is exactly that shape: probes are cheap, and the log tells you where to
look next.

### 13.2 Deliberative

```
   observation -> update model of the situation -> generate options -> choose -> act
```

Two boxes are inserted between input and action: a representation of the situation, and a
planner over it. That is the entire structural difference, and it is what buys multi-step
coherence at the cost of latency.

`07`'s pipeline is this architecture with each box named and made inspectable:

| Stage | Deliberative role |
| :--- | :--- |
| `gather` | Collect the raw considerations |
| `frame` | Build the model of the situation — what is the actual trade-off |
| `propose` | Generate candidate courses of action |
| `choose` | Evaluate and commit, including the condition that would flip it |
| `report` | Act, i.e. produce the deliverable |

The measured cost is 760 tokens and five model calls for one question, and the measured
liability is §14.3: run a question that needs none of this and the machinery still executes,
inventing a trade-off for a question whose answer is a port number.

**A deliberative agent's weakness is not that it is slow. It is that it cannot tell it is
over-thinking.**

### 13.3 Hybrid

```
              observation
                   |
              coordinator
             /            \
        reactive      deliberative
             \            /
                 output
```

A coordinator classifies the request and dispatches. Everything else is unchanged — the same
nodes, the same prompts.

`07`'s router is this, with a single conditional edge as the coordinator:

```python
builder.add_conditional_edges(
    "triage",
    lambda state: "gather" if state["depth"] == "deep" else "answer_directly",
    {"gather": "gather", "answer_directly": "answer_directly"},
)
```

Two things about it are worth copying:

**The coordinator's verdict is state, not a caller flag.** `depth` is produced by a node and
merged into the state, so downstream edges read data the graph made. That is what distinguishes
a hybrid agent from a caller who passes `fast=True`.

**The classifier's uncertainty resolves toward quality.** A verdict naming neither word falls
back to `deep`, and the raw reply is retained. Falling back to the cheap path would make an
unclear classification silently degrade the answer, and would hide the degradation.

The measured result is §14.5: 79 tokens where the pipeline spent 570, at a cost of 75 extra
tokens when the deep path is taken anyway, breaking even at 13% shallow traffic.

### 13.4 Choosing between them

| Signal | Points to |
| :--- | :--- |
| Probes are cheap and the environment answers | Reactive |
| The answer needs several dependent steps and coherence between them | Deliberative |
| The incoming traffic is genuinely mixed in difficulty | Hybrid |
| Traffic is uniformly hard | Deliberative — a coordinator would only add cost |
| Traffic is uniformly easy | Reactive, or no agent at all |

The last row deserves emphasis: **if every request is easy, the correct architecture is a single
model call**. `07`'s shallow path is two nodes, and one of them exists only to decide that the
other one is enough.

---
## 14. Topology: the same five nodes, two behaviours

`07_langgraph_topologies.py`

This is the script that makes §2 concrete. Five nodes are defined once and wired twice.

### 14.1 One state object, and nodes that never call each other

```python
class ReviewState(TypedDict):
    question: str
    depth: Optional[Literal["shallow", "deep"]]
    facts: Optional[str]
    framing: Optional[str]
    options: Optional[str]
    choice: Optional[str]
    answer: Optional[str]
    visited: Annotated[list[str], operator.add]
    tokens: Annotated[int, operator.add]
```

Each node returns a partial dict that is merged into the state, so adding a node means adding a
field rather than rewiring a call chain. `visited` and `tokens` carry an `operator.add` reducer
because several nodes append to them; without the annotation, the last writer would overwrite
every earlier entry.

The five analysis nodes each add exactly one field:

| Node | Instruction it sends | Field it adds |
| :--- | :--- | :--- |
| `gather` | List three concrete factors that bear on this question | `facts` |
| `frame` | In one sentence, name the central trade-off these factors describe | `framing` |
| `propose` | Propose two opposing courses of action | `options` |
| `choose` | Pick one and state the condition under which it stops being right | `choice` |
| `report` | Write a three-sentence answer using this reasoning | `answer` |

Reading a predecessor's field goes through a helper that names what is missing:

```python
raise ValueError(f"node {node!r} needs '{field}', but no earlier node produced it - check the edges")
```

A bare `state[field]` would fail with a `KeyError` that says nothing about why, and the whole
point of this script is inviting the reader to rewire the edges.

### 14.2 Topology one: a straight pipeline

```python
builder.add_edge(START, "gather")
builder.add_edge("gather", "frame")
builder.add_edge("frame", "propose")
builder.add_edge("propose", "choose")
builder.add_edge("choose", "report")
builder.add_edge("report", END)
```

Running a question that genuinely needs the analysis, printing the state after each node:

```
question: Should a small team run its own database server instead of paying for a managed one?
  after gather   state holds: facts
  after frame    state holds: facts, framing
  after propose  state holds: facts, framing, options
  after choose   state holds: facts, framing, options, choice
  after report   state holds: facts, framing, options, choice, answer
  nodes run: 5, tokens: 760
```

Every node earned its place: the answer weighs self-hosting against managed hosting and names
the condition that flips the decision.

### 14.3 The same pipeline, given a question that needed none of it

```
question: What port does PostgreSQL listen on by default?
  nodes run: 5, tokens: 570
  the pipeline's 'framing' step invented a trade-off anyway:
    "The central trade-off is between compile-time configurability and runtime
     flexibility for the PostgreSQL server port..."
```

The correct answer is `5432`. The pipeline produced it wrapped in three sentences about
`PGPORT` overrides and deployment environments, because a node whose instruction is *name the
central trade-off* will name one whether or not a trade-off exists.

**This is the real cost of a fixed pipeline, and it is not only tokens.** Forced analysis
produces analysis-shaped output regardless of the input.

### 14.4 Topology two: the same nodes behind a triage node

```python
builder.add_edge(START, "triage")
builder.add_conditional_edges(
    "triage",
    lambda state: "gather" if state["depth"] == "deep" else "answer_directly",
    {"gather": "gather", "answer_directly": "answer_directly"},
)
```

A conditional edge is a function from state to the name of the next node, so the decision is
**data the graph produced**, not a flag the caller had to set in advance.

The triage node asks for exactly one word, `shallow` or `deep`, and keeps the raw reply:

```python
if "deep" in verdict:      depth = "deep"
elif "shallow" in verdict: depth = "shallow"
else:                      depth = FALLBACK_DEPTH      # "deep"
```

A verdict naming neither word means the model did not follow the format, not that it chose
`shallow`. Falling back to the expensive path keeps quality the default when the classifier
itself is unclear, rather than quietly saving money; keeping `verdict_raw` means a wrong call
can be diagnosed against what the model actually said.

### 14.5 Three questions through the router, measured

```
What port does PostgreSQL listen on by default?
  triage said shallow, nodes run: 2, tokens: 79
  -> triage -> answer_directly
  answer: PostgreSQL listens on port 5432 by default.

Should a small team run its own database server instead of paying for a managed one?
  triage said deep, nodes run: 6, tokens: 835
  -> triage -> gather -> frame -> propose -> choose -> report

Is PostgreSQL better than MySQL?
  triage said deep, nodes run: 6, tokens: 864
  -> triage -> gather -> frame -> propose -> choose -> report
```

Side by side with the pipeline:

| Question | Fixed pipeline | Router | Difference |
| :--- | ---: | ---: | :--- |
| Shallow (`default port`) | 570 tokens, 5 nodes | **79 tokens, 2 nodes** | saves 491 |
| Deep (`self-host or managed`) | 760 tokens, 5 nodes | 835 tokens, 6 nodes | costs 75 extra for triage |

```
break-even share of shallow traffic: 13%
```

Routing pays for itself once more than **13%** of questions are shallow. That number is the
honest way to argue for a topology change: the triage call is not free, and the saving only
exists if the traffic actually contains easy questions.

### 14.6 Print the topologies instead of describing them

```
fixed pipeline                    conditional router
  __start__ -> gather               __start__ -> triage
  gather -> frame                   triage -> answer_directly (conditional)
  frame -> propose                  triage -> gather (conditional)
  propose -> choose                 gather -> frame
  choose -> report                  frame -> propose
  report -> __end__                 propose -> choose
                                    choose -> report
                                    report -> __end__
                                    answer_directly -> __end__
```

Same node functions, same prompts, same model. One conditional edge is the entire difference
between the two columns, and between 570 tokens and 79 on the same question.

---

## 15. Reliability: autonomy is paid for in determinism

Everything that makes an agent useful also makes it variable. This chapter collects what the
runs showed about containing that.

### 15.1 The same input does not give the same run

Not a bug, a definition. If the sequence were fixed it would be a workflow. Observed in this
module:

- `04` answers the same question in **6 calls** with clear tool text and **14** with vague text,
  ending on a different component.
- `03` needs **1** tool call for an in-scope question and **5** for an out-of-scope one, and the
  particular five it picks depend on wording.
- `07`'s triage classifies each question independently, so a rephrasing can move a question
  between the cheap and expensive path.

The consequence for testing is in §16: a single successful run tells you almost nothing.

### 15.2 Failure mode one: the tool is available but unused

The sharpest failure in this module is not a crash. In `03`'s fifth demo the loop runs, the model
produces well-formed actions, and **not one real tool is called**, because the tool names were
never rendered into the prompt. The registry was complete. The code looked finished.

The general form: *a tool the model cannot see does not exist, no matter how correctly it is
registered.* The diagnosis is always the same — print the prompt as sent, or print the tool
list as the model receives it. `04` does the second as its opening step for exactly this reason.

Two habits follow:

1. **Render the tool list from the registry**, never maintain it as a separate string. `03`
   builds `{tools}` and `{tool_names}` from the same dict that holds the functions, so a tool
   cannot be added to one and forgotten in the other.
2. **When behaviour is inexplicable, check visibility before intelligence.** "The model ignored
   the tool" is far more often "the model never had it".

### 15.3 Failure mode two: no natural stop

An agent's exit condition is a model's opinion that it is finished. When the answer does not
exist, that opinion never arrives — `04`'s bandwidth question searches for `bandwidth`,
`traffic`, `bytes`, `usage`, `GB` and would keep going. The cap is what ends it:

```python
config={"recursion_limit": recursion_limit}
```

Two details make the cap usable rather than merely present:

- The run is **streamed**, so the messages produced before the cap are kept and can be inspected.
  A stopped run that vanishes teaches nothing.
- Hitting the cap is reported as itself — `answer: none, the agent hit its step limit first` —
  rather than being smoothed into a plausible-sounding final answer, which is the failure mode
  that costs the most trust.

### 15.4 Failure mode three: structured output that is not

Every step that parses model output is a place a run can end. `03` handles this at the prompt
level with three fallbacks in front of its regex; anything asking for JSON needs the equivalent:
validate at the boundary, keep the raw text for diagnosis, and decide in advance whether a
parse failure means retry, default, or stop.

The structural version of the same fix is to give the model fewer chances to be creative. A
chain of five parsed hand-offs has five failure points; folding a stable sub-sequence into one
tool has one. That is a real trade — you lose the intermediate visibility §1.4 argued for — so
it is worth making deliberately rather than by accident.

### 15.5 What actually buys determinism back

| Lever | What it does | Where it appears |
| :--- | :--- | :--- |
| **Put the fixed part in the graph** | A step wired as an edge cannot be skipped or reordered | `07`'s five edges |
| **Name the tools and the conditions in the prompt** | Selection improves, but stays a decision | `03`'s `{tools}` block |
| **Cap the steps** | Bounds a failure instead of a bill | `04`'s recursion limit |
| **Enforce, do not request** | Schema and auth checked by the receiver | `06`'s 400 and 401 |
| **Seed the randomness** | Makes runs comparable after the fact | `04`'s `random.seed(7)` |
| **Keep the trace** | Distinguishes an investigated answer from a guessed one | every script |

The first row is the strongest and the most often skipped: **if a step must always happen, it
should be an edge, not a sentence in a prompt.** Prompt instructions are a request; graph edges
are a guarantee. Everything the model still chooses is where the variance lives, and that is
where the testing effort belongs.

### 15.6 State containers help more than they look like they should

Running nodes against one merged state object, rather than passing values between functions,
buys three things that show up when something goes wrong: every intermediate value is in one
inspectable place (`07` prints the state after each node), a node failing does not destroy what
earlier nodes produced, and a field that was never filled fails with a message naming the field
and the node instead of a bare `KeyError`:

```python
raise ValueError(f"node {node!r} needs '{field}', but no earlier node produced it - check the edges")
```

---

## 16. Evaluating an agent

### 15.1 Why one successful run proves nothing

§15.1 lists the observed variance. The practical consequence is that "I ran it and it worked" is
not evidence about an agent in the way it is about a function. Two runs of `04` differ by 8 tool
calls and reach different conclusions; a reviewer who saw only the good run would have
concluded the tools were fine.

So the unit of verification is a **set** of cases, run more than once.

### 15.2 What a case needs

| Column | Purpose |
| :--- | :--- |
| **Question** | The input |
| **Reference answer** | What a correct response contains |
| **Checkpoints** | The specific facts or refusals that must appear |

The third column is what makes open-ended answers gradeable. Exact-match scoring fails
immediately on prose; "does the answer contain the fact that the billing host is unreachable" is
checkable and does not care about phrasing.

Cases worth including, drawn directly from what these scripts exposed:

- **A question one tool answers.** Baseline.
- **A question no tool can answer.** `03`'s carried-interest question. Grades the refusal, which
  is the behaviour most likely to regress silently.
- **A question needing several dependent steps.** `04`'s incident. Grades the ordering.
- **A question that is easy but sounds hard.** `07`'s port question. Grades the routing.

### 15.3 Grading with a model, carefully

Judging prose against a reference is itself a language task, so a second model is the usual
grader. Two constraints keep it honest:

- **The grader must not be allowed to improve the answer.** It returns a verdict and, on failure,
  the specific conflict. If it may rewrite, you can no longer tell whether the agent was right or
  the grader repaired it.
- **The grader must not have authored the reference.** Generating questions with a model is fine
  and saves real time; letting the same model write the reference answers means grading its own
  work.

### 15.4 What to record besides pass or fail

The traces these scripts print are what a test harness should keep:

| Signal | Why it matters |
| :--- | :--- |
| **Tool calls per case** | The 6-vs-14 gap in `04` appeared here before it appeared in the answer |
| **Which tools ran, in order** | Distinguishes an investigated answer from a lucky one |
| **Tokens per case** | The only way §14.5's break-even is arguable |
| **Cap hits** | A rising rate means the tools stopped covering the questions |

A regression in an agent frequently shows up in these numbers **while the answers are still
correct** — the agent starts taking twelve calls to reach what it used to reach in six. That is
the early warning, and it is invisible to a suite that only checks final answers.

---
## 17. The tooling around these scripts

A short map of the landscape, and why these seven files are written the way they are.

### 17.1 What is used here, and for what

| Component | Used for | Where |
| :--- | :--- | :--- |
| **OpenAI-format chat API** | Every model call in the module | All except `06` |
| **LangChain core** | Templates, output parsers, composition, the `@tool` decorator | `01`–`04`, `07` |
| **LangChain agents** | `create_agent`, the framework version of the loop | `04` |
| **LangGraph** | State graphs, checkpointing, conditional edges | `01`, `07` |
| **MCP SDK** | Server and client halves of the protocol | `05` |
| **FastAPI + urllib** | The A2A provider and its caller | `06` |
| **Nothing at all** | The hand-written loop | `03` |

`03` is deliberately dependency-free. A reader who only sees framework code cannot tell which
behaviours are the framework's and which are the model's; a version with no framework settles
that question and makes `04` legible as "the same thing, minus the typing".

### 17.2 Chains versus graphs

| | Chain-shaped | Graph-shaped |
| :--- | :--- | :--- |
| Structure | `A -> B -> C` | Nodes and edges, including conditional ones |
| Loops and branches | Awkward | Native |
| State | Passed along | A shared object every node reads and writes |
| Relationship | — | **A superset**: anything linear is expressible as a graph |

The superset relation is worth stating plainly: choosing the chain form is not a capability
judgement, it is a statement that the work is linear. `07` uses the graph form for both
topologies precisely so the comparison is between edges, not between libraries.

### 17.3 Why the provider is reached through a base URL

```python
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
```

Every script in the module reaches its model through the OpenAI request format plus a
`base_url`. The consequence is that changing provider is a two-line edit and nothing else in the
file moves — no different SDK, no different message shape, no different tool-call format.

This also isolates the failure that matters most in a portfolio: a provider-specific SDK ties
the demonstration to an account. A reader with a different key can run all seven scripts by
editing two constants.

### 17.4 Higher-level agent frameworks, and why none are used

There are frameworks that wrap all of the above: an assistant object, a list of tools, a web UI,
and three lines to run it. They are a reasonable choice for building a product and the wrong
choice for a module whose subject is the mechanism, because they hide precisely the parts under
discussion — how the tool list reaches the model, where generation stops, what the transcript
looks like on the second pass.

The same reasoning applies to visual builders. A canvas of nodes and edges is the same data flow
as `07`'s graph, with the wiring done by mouse instead of by `add_edge`, and with the same
property: the path is fixed by whoever drew it. Their real cost is not expressiveness but
portability — the diagram is not the code, so the move from prototype to production is a
rewrite rather than an export.

---
## 18. What the runs showed

Every row below came out of an actual run of the script named in it.

| # | What was measured | Result |
| :--- | :--- | :--- |
| 01 | Follow-up question with the transcript replayed vs. without | `"Call it PhotoDateRename or DateRenamer"` vs. `"Call it whatever feels right to you"` |
| 01 | What the checkpointer holds after two turns | 4 messages: human, ai, human, ai |
| 02 | Five independent branches, parallel vs. serial | **1.39s vs 4.29s** (3.1x) |
| 02 | Streaming vs. blocking, same chain | first chunk at **0.29s**, 103 chunks / nothing until **2.04s** |
| 02 | Retry by hand vs. `.with_retry()` | 3 attempts either way; one is a loop plus `except`, one is a method call |
| 03 | One-lookup question | 2 model passes, 1 tool call |
| 03 | Question outside the rule book | 6 passes, 5 tool calls, then a correct refusal |
| 03 | **Same tools, tool list removed from the prompt** | model invented `Search rule book` and called it; 0 real tools reached |
| 04 | Live incident, clear tool descriptions | **6 tool calls, correct root cause** (`billing.internal` unreachable) |
| 04 | **Same tools, vague descriptions** | **14 tool calls, wrong root cause** (`eth1 is down`) |
| 04 | Unanswerable question, `recursion_limit=4` | 6 calls, no answer, stopped by the cap |
| 05 | Handshake with the subprocess server | `notes`, protocol `2025-11-25` |
| 05 | Tools learned at runtime | 3, with JSON Schema per argument |
| 05 | Loop over the process boundary | 3 rounds; three `read_note` calls dispatched concurrently in one turn |
| 06 | Runtime discovery | endpoint and auth scheme both read from the card |
| 06 | Schema violation (`attendees: 500`, card max 60) | **400**, `attendees must be an integer between 1 and 60` |
| 06 | Missing bearer token | **401**, `missing or invalid bearer token` |
| 07 | Deep question: pipeline vs. router | 760 vs **835** tokens (triage overhead: 75) |
| 07 | Shallow question: pipeline vs. router | 570 vs **79** tokens (saves 491) |
| 07 | Break-even share of shallow traffic | **13%** |

### Three findings worth keeping

**1. Tool text is runtime input, not documentation.** Two scripts remove it in different ways and
both degrade immediately: `03` drops the tool listing from the prompt and the model invents tool
names in prose; `04` keeps the tools but blurs their descriptions and returns, and the same
question costs 14 calls instead of 6 and ends on the wrong component. The functions were correct
in both runs. Only the text changed.

**2. A miss has to be readable.** In `03` the tools answer `No rule matches 'tax rate'.` and the
model recovers: it switches tool, enumerates the categories, and refuses honestly. In `04`'s
vague variant the same class of miss comes back as `nothing found`, and the run drifts into
guessing hostnames. **Error strings are part of the control flow when the caller is a model.**

**3. Topology is a measurable decision, not a style preference.** The router costs 75 extra
tokens on a deep question and saves 491 on a shallow one, so it wins above 13% shallow traffic.
That is arguable with numbers. "Graphs are more flexible" is not.

---

## 19. Where the boundaries are

A short map of what these seven scripts do and do not cover, so the scope is explicit.

**Covered here:** prompt templates and the request payload; conversation state; composition,
parallelism, routing and streaming; the ReAct loop by hand and inside a framework; tool schemas
and what they decide; MCP client and server; agent-to-agent discovery and delegation; pipeline
versus conditional-router topologies with token accounting.

**Deliberately not covered:**

- **Retrieval.** Chunking, embeddings, vector stores and rerank belong to `02-rag/`, and the
  agents here treat retrieval as one more tool rather than reimplementing it.
- **SQL as a tool surface.** Schema prompting, query safety and evaluation are `03-text2sql/`.
- **Hosted third-party servers.** `05` runs both halves locally on purpose: a remote server
  needs an account, a key and a network, none of which a reader can reproduce from this
  repository. The client half is unchanged either way, since transport is what differs.
- **Long-term memory.** `01` shows the mechanism (replay the transcript) and states the ceiling;
  windowing, summarising and vector-backed recall are strategies over the same mechanism.

---

## 20. Running them

```bash
cd 04-agents
python 01_prompt_templates_and_memory.py
python 02_lcel_composition.py
python 03_react_loop_from_scratch.py
python 04_tool_agent_diagnosis.py
python 05_mcp_client_and_server.py
python 06_a2a_agent_protocol.py
python 07_langgraph_topologies.py
```

**Keys.** Put `DEEPSEEK_API_KEY` in a `.env` file at the repository root. Every script that needs
it checks first and prints what it is skipping instead of failing with a stack trace:

| Script | Without a key |
| :--- | :--- |
| `01` | Steps 1 and 2 still run (template rendering is local) |
| `02` | Step 2 still runs (the local functions need no model) |
| `03` | Prints the rendered tool listing, then stops |
| `04` | Prints the tool schemas, then stops |
| `05` | Prints the published tools, then stops |
| `06` | **Runs completely.** There is no model in it |
| `07` | Stops; every node is a model call |

**Ports and processes.** `05` starts itself as a subprocess (`--serve`) and talks to it over
stdio, so nothing listens on a port. `06` starts a local provider on `127.0.0.1:8931` in a
background thread and waits for it to answer before the first request, rather than sleeping a
fixed interval.

**Data.** `data/notes/` holds the three notes `05` serves: an incident write-up, an onboarding
note, and a release checklist. Everything else is inline in the scripts, and the one script with
randomness (`04`) seeds it so its latencies are identical on every run.

---

## 21. Three lines to take away

**The loop is text.** Tool names reach the model as a paragraph in the prompt, the format is a
request rather than a guarantee, and the stop sequence is what makes generation into a loop.
Delete any of the three and the loop keeps running while quietly doing nothing.

**Tool text is the agent's interface to itself.** Descriptions decide what gets called, return
strings decide whether a wrong turn is recoverable, and both are billed per attempt.

**Protocols solve distribution, not capability.** MCP made a note reader usable by any client;
A2A let one agent find another's endpoint at runtime. Neither made a model better at anything.
What they removed was the need to write a new adapter for every pair.
