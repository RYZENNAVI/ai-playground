# Agent systems

Scripts 01 and 02 cover what goes into a request: templates, a conversation kept between
calls, and LCEL composition. Scripts 03 and 04 run the same ReAct loop, once by hand and once
with LangChain's `create_agent`, and each removes one piece of tool text to see what breaks.
Scripts 05 and 06 put a protocol between the caller and the tools: MCP over stdio, and a
simplified A2A-style agent card. Script 07 wires five LangGraph nodes as a fixed pipeline and
behind a triage node. This document explains what each script does and the ideas it relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_prompt_templates_and_memory.py` | Prompt templates, and memory as a transcript a LangGraph checkpointer sends again |
| 02 | `02_lcel_composition.py` | LCEL: a sequential chain, retry, local functions, parallel branches, routing and streaming |
| 03 | `03_react_loop_from_scratch.py` | A ReAct loop with no framework, and the run where the prompt never lists the tools |
| 04 | `04_tool_agent_diagnosis.py` | The same loop with `create_agent` and native function calling, a step limit, and vague tool descriptions |
| 05 | `05_mcp_client_and_server.py` | Both halves of MCP in one file: a stdio server, the handshake, schema translation and a tool-calling loop |
| 06 | `06_a2a_agent_protocol.py` | A2A-style delegation (simplified): an agent card, a task, and the provider's schema and auth checks |
| 07 | `07_langgraph_topologies.py` | Five LangGraph nodes as a fixed pipeline and behind a triage node, compared in tokens |

## Shared setup (scripts 01 to 05 and 07)

*   Every script that calls a model uses DeepSeek's `deepseek-chat` when `DEEPSEEK_API_KEY`
    is set in `.env` at the repository root, which `.gitignore` excludes, and OpenAI's
    `gpt-4o-mini` otherwise. `OPENAI_BASE_URL` and `OPENAI_MODEL` point the OpenAI key at
    another compatible vendor. Temperature is 0 everywhere.
*   The numbers below come from DeepSeek runs on 2026-09-28. Where a script was run several
    times, the range is given.
*   Scripts 03 and 05 also ran through Gemini's OpenAI-compatible endpoint. Script 04 fails on
    its second turn, because Gemini 3 requires a thought signature on each function call and
    LangChain's `ChatOpenAI` does not send it back. Script 07 makes 19 model calls in one run,
    more than the free tier's 15 requests a minute.
*   Without a key, each script runs the parts that need no model and says what it skipped:

    | Script | Without a key |
    | :--- | :--- |
    | `01` | Parts 1 and 2 (template rendering is local), then stops |
    | `02` | Parts 2, 3 and 5 (retry, local functions and routing call no model) |
    | `03` | Part 1, the tool listing rendered into the prompt, then stops |
    | `04` | Part 1, the tool schemas, then stops |
    | `05` | Parts 1 to 4 (handshake, tool listing and schema translation), then stops |
    | `06` | Runs completely. It calls no model |
    | `07` | Part 4, the two topologies, then stops |

## Script 01: Prompt templates and memory

*   A `PromptTemplate` is text with named slots, and LangChain reads the variable names from
    the text. Declaring `input_variables=[]` by hand changes nothing: the template still
    reports `['product']`. A missing value raises `KeyError: 'product'`, the same as Python's
    own `str.format`.
*   A `ChatPromptTemplate` renders to a list of messages, not one string. The instruction goes
    in the system message and stays fixed while only the human message changes. Part 2 only
    renders the messages and sends nothing.
*   Part 3 pipes the template into the model and `StrOutputParser`. With the parser the result
    is the string `"J'adore la programmation."`; without it, an `AIMessage` carrying the same
    text.
*   The model is stateless: each request is judged only on the messages it carries. Memory
    here is a LangGraph checkpointer that stores each thread's messages, and a
    `MessagesPlaceholder` that marks where the graph pours them back in.
*   The test is a follow-up that names nothing. The first turn says "I am building a small tool
    that renames photo files by date", and the second asks "What should I call it?". In six
    runs the answer always named the photo tool, most often PhotoDater.
*   The checkpointer then holds four messages, in order: human, ai, human, ai. The system
    message is not stored; the graph adds it on every call.
*   Sent alone, the same follow-up gets `Could you clarify what "it" refers to?`. Every stored
    turn is sent again on every later call, so a transcript that keeps growing also grows the
    cost of each request.

## Script 02: LCEL composition

LCEL, the LangChain Expression Language, treats every step as a runnable: a template, a
model, a parser, or a plain function wrapped in `RunnableLambda`. The pipe operator `|` joins
runnables into one, and the result has the same `invoke`, `stream` and `batch` methods as each
step. The numbers below come from six runs.

*   Part 1 chains three model calls: translate a review into French, state its main complaint
    in French, and translate that back. Each template expects a named variable, so a dict
    renames the previous output:

    ```python
    chain = to_french | {"french": lambda text: text} | review_it | {"summary": lambda text: text} | back_to_english
    ```

    If a name does not match, the next template raises `KeyError: Input to ChatPromptTemplate
    is missing variables {'french'}` before any request is sent.
*   Part 2 retries a step that fails on its first two calls, once with a hand-written loop and
    once with `.with_retry()`. Both succeed on the third attempt, but they are not identical.
    `.with_retry()` retries on any exception by default and waits with exponential backoff and
    jitter, 3.6 to 4.4 seconds in all. The loop catches only `RuntimeError` and does not wait.
*   Part 3 wraps three plain functions in `RunnableLambda` and invokes them like a model. They
    count words and guess sentiment from two word lists, turn CSV into JSON, and count lines.
*   Part 5 routes three inputs with `RunnableBranch`, which takes the first branch whose
    condition holds: the CSV sample goes to the converter, a three-line note to the line
    counter, and `hello` to the default. The CSV check comes first because the CSV sample also
    has more than two line breaks.
*   Part 5 also shows when `|` accepts a plain function. A plain function piped into the router
    works, because LangChain wraps it. Two plain functions piped together raise
    `TypeError: unsupported operand type(s) for |: 'function' and 'function'`.
*   Parts 4 and 6 measure time:

    | Part | Compared | Measured |
    | :--- | :--- | :--- |
    | 4 | Five prompts about one review, `RunnableParallel` against one after another | 0.85 to 1.10 s against 3.34 to 4.34 s |
    | 6 | The same chain with `stream` against `invoke` | first chunk after 0.28 to 0.57 s, full reply after 0.84 to 1.33 s |

*   `RunnableParallel` gives each branch its own thread, and the branches do not depend on each
    other. Streaming works because every step in the chain passes chunks along. A plain
    function in `RunnableLambda` would wait for the whole input and turn the stream back into
    one block.

## Script 03: A ReAct loop by hand

ReAct (reason and act) has the model write a Thought, then an Action and its input. The
program runs that tool, adds the result as an Observation, and calls the model again, until
the model writes a Final Answer. Script 03 does this with a chat completion call, a regex and
a `for` loop, with no agent framework.

*   The tools answer questions about a rule book of four rules (`R-001` to `R-004`) in two
    categories. One tool searches the text, one lists a category, and one reads a rule by id.
    They are an ordinary dict, and the prompt's `{tools}` and `{tool_names}` are filled from
    it, so a tool cannot be registered and left out of the prompt.
*   The prompt fixes the format the regex parses:

    ```
    Question: the question you must answer
    Thought: what you need to do next
    Action: one of [{tool_names}]
    Action Input: the input for that tool
    Observation: the result the tool returned
    ... (Thought/Action/Action Input/Observation may repeat)
    Final Answer: the answer for the user
    ```

*   Generation stops at `Observation:`, so the program supplies the tool result and the model
    cannot invent it. Two calls without the stop both wrote the Observation themselves: `No
    rules found matching those keywords.` The real search finds R-002, so the invented result
    was also wrong.
*   A reply containing `Final Answer:` ends the loop. A reply matching `Action:` and
    `Action Input:` runs the tool. Anything else is returned as `[unparsable reply]`, so a
    format break is not hidden.
*   The four questions:

    | Part | Question | Tool calls | Passes | Outcome |
    | :--- | :--- | ---: | ---: | :--- |
    | 2 | The minimum a securities fund must raise | 1 | 2 | 10,000,000, from R-002 |
    | 3 | Same question, tool list removed from the prompt | 3 | 4 | "no working tool" |
    | 4 | Tax rate on carried interest (not in the rule book) | 5 | 6 | "the rule book does not cover this" |
    | 5 | Read rule R-004 in full | 1 | 2 | both reporting deadlines |

*   In part 3 the functions are still registered, but the model only sees `Action: the tool to
    use`. It guessed names such as `search_rule_book`, `lookup_rule` and `search`, and after
    two or three misses answered that it had no working tool. A model only knows a tool exists
    if its name is in the prompt.
*   In part 4 every run searched three times, listed both categories, and ended with a normal
    Final Answer. The tools say what they missed (`No rule matches ...`), so the model can read
    a miss and try another tool.
*   Each pass sends the whole transcript again. On the part 4 question the six passes read 207,
    265, 308, 349, 415 and 478 input tokens, 2022 in all, against 1242 if every pass were as
    short as the first.

## Script 04: The same loop with create_agent

`create_agent` runs the loop from script 03 over the model's native function calling: the
model returns tool calls, not text to parse.

*   The `@tool` decorator builds a schema from each function's name, argument types and
    docstring, and part 1 prints what the model receives. The docstring is now runtime input.
    Editing it changes what the agent does, and neither a type checker nor a test notices.
*   The simulated network has three hosts. `billing.internal` resolves to `10.0.4.37` but does
    not answer. One interface is up and one is down, and the log holds five lines. The question
    ("Checkout keeps failing with connection errors since 14:00. What is broken?") names a
    symptom, not a host, so the agent has to read the log before it knows what to test.
*   The trace is read from the tool calls and `ToolMessage` entries in the message list that
    `create_agent` returns.

    | Part | Setting | Runs | Tool calls | Result |
    | :--- | :--- | ---: | :--- | :--- |
    | 2 | Clear docstrings | 6 | 6 or 7 | all named `billing.internal` |
    | 3 | `recursion_limit=4` | 6 | 6 or 7 | all stopped with the evidence in hand, no answer |
    | 4 | Vague docstrings (`Do a lookup.`, `Do a check.`, `Do a search.`) | 8 | 12 to 18 | every answer read named `billing.internal` |

*   In part 2 the first four calls go out together (two log searches, both interfaces). The
    model then follows the host the log named and separates "does not resolve" from "does not
    answer". Nothing in the code specifies that order.
*   The limit in part 3 counts graph steps, and each model turn and each tool round is one. The
    run is read with `agent.stream`, so the messages before the stop are kept, and the stop is
    printed as a stop rather than as an answer.
*   In part 4 each vague tool calls the clear one, so only the description changes. The vague
    runs start by guessing hostnames (`checkout`, `checkout.internal`, `checkout.svc`) and
    pinging the gateway, and reach the log line later. The description changed the cost of
    the run, not its conclusion.

## Script 05: MCP client and server

MCP, the Model Context Protocol, is a standard way to publish tools that any client can
discover and call. The client asks the server what tools it has at runtime instead of
importing them.

*   The file holds both halves. Run normally, it starts a second copy of itself with `--serve`
    as a subprocess. That copy is the server, and it speaks JSON-RPC over stdin and stdout. The
    parent is the client: a `ClientSession` over `stdio_client`. `main()` is the host, the
    application the model runs in. Nothing listens on a port.
*   The server publishes three tools over a folder of notes in `data/notes/`: `list_notes`,
    `read_note` and `count_words`. Parts 2 and 3 are the handshake (protocol `2025-11-25`) and
    the tool listing. The client did not know the three names before it asked.
*   A stdio server sends its protocol messages on stdout, so logging belongs on stderr. With
    mcp 2.0.0 a stray line is not fatal: adding `print("server starting")` made the client log
    `Failed to parse JSONRPC message from server`, skip the line, and finish normally.
*   The filename comes from the model, so it is checked:

    ```python
    path = (NOTES_DIR / filename).resolve()
    if not path.is_relative_to(NOTES_DIR.resolve()) or path.suffix != ".txt":
        return None
    ```

    `NOTES_DIR / filename` alone confines nothing: `..` segments walk back out, and an
    absolute path replaces `NOTES_DIR` entirely. A rejected filename raises `ValueError`, and
    the SDK returns `isError: true` with `No note named '../../.env'.` An earlier version
    returned `-1`, which the protocol reported as a success.
*   Part 4 turns each MCP tool into the chat API's tool format. Both sides already use JSON
    Schema, so the only change is the field name, `input_schema` to `parameters`. The same
    schemas worked unchanged with DeepSeek and with Gemini.
*   Part 5 is a tool-calling loop in which every call goes to the server. Twelve runs (ten with
    DeepSeek, two with Gemini) all took three rounds: `list_notes`, then `read_note` on the
    incident note only, then the answer. The first call's full response shows that the SDK also
    wraps a plain return value as `structuredContent`.
*   The client sends the calls of one round with `asyncio.gather`, so several `tool_calls` in
    one reply would go to the server together. None of these runs returned more than one.
*   The question asks for the follow-up the note recommends. An earlier version asked for "the
    fix", which the note does not contain: DeepSeek added a caveat, and Gemini labelled the root
    cause as the fix.

## Script 06: A2A-style delegation

A2A (Agent2Agent) is a protocol for one agent to find and use another. The provider publishes
an agent card at a well-known URL, and the caller reads the card to learn where to send a
task, what inputs it takes and how to authenticate. Script 06 follows that idea in a
simplified form. Its card fields and its task route are its own: A2A 1.0 cards list skills and
security schemes, and tasks go through the SendMessage operation. No model is called.

*   The provider is a FastAPI app that knows which rooms are free. It runs on
    `127.0.0.1:8931` in a background thread, and the script polls the card URL until it
    answers. The card at `/.well-known/agent-card.json` holds a name, a task endpoint, an input
    schema (`date`, and `attendees` from 1 to 60) and the auth method, `bearer`.
*   The caller reads the task route and the auth scheme from the card instead of hardcoding
    them. If the provider moved its endpoint and updated its card, the caller would keep
    working unchanged.
*   The provider's room table is private. What comes back is an artifact derived from it, and
    the caller makes the decision itself, the smallest room that fits or a cancellation:

    | Task | Provider returned | Caller's decision |
    | :--- | :--- | :--- |
    | 2026-04-14, 10 people | Cedar (12), Aspen (40) | confirmed in Cedar |
    | 2026-04-15, 30 people | none (only a 12-seat room) | cancelled |
    | 2026-04-16, 8 people | none (no room free) | cancelled |
    | 2026-04-14, 500 people | status 400, attendees must be between 1 and 60 | |
    | 2026-04-14, 5 people, no bearer token | status 401 | |

*   The provider reads the attendee limits from its own card when it checks a task, so the
    card and the check cannot drift apart.
*   Of A2A's five steps (discover, submit, process, clarify, finish), the script covers
    discover, submit, process and finish. The provider answers at once, so there is no
    streaming and no `input-required` state, where a task pauses to ask the caller for more
    under the same task id.
*   The output is the same on every run: there is no model and no randomness.

## Script 07: Two topologies in LangGraph

LangGraph builds an agent as a `StateGraph`. Each node is a function that reads one shared
state and returns the fields it adds, and edges decide which node runs next. A conditional
edge picks the next node from the state. Script 07 defines five nodes once and wires them
twice.

*   The state is a `TypedDict`. `visited` and `tokens` carry an `operator.add` reducer because
    several nodes write to them; without it, each node's value would replace the last one.
*   Each of the five analysis nodes makes one model call and adds one field:

    | Node | Instruction it sends | Field it adds |
    | :--- | :--- | :--- |
    | `gather` | List three concrete factors that bear on this question | `facts` |
    | `frame` | In one sentence, name the central trade-off these factors describe | `framing` |
    | `propose` | Propose two opposing courses of action | `options` |
    | `choose` | Pick one and state the condition under which it stops being right | `choice` |
    | `report` | Write a three-sentence answer using this reasoning | `answer` |

*   `frame`, `propose` and `choose` also receive the question. An earlier version gave each of
    them only the previous field, and one of three runs drifted from "self-host or managed" to
    an answer about single-region monoliths.
*   A node reads an earlier field through a helper that names the missing field and the node,
    so a wrongly wired edge does not surface as a bare `KeyError`.
*   Part 1 wires the nodes as a straight pipeline and asks whether a small team should run its
    own database server. Three runs spent 601 to 663 tokens, and the answer names the condition
    that flips the decision.
*   Part 2 sends "What port does PostgreSQL listen on by default?" through the same pipeline.
    The answer, `5432`, came with two sentences about changing the port, because a node told to
    name the central trade-off names one whether or not there is one. All three runs framed
    the same convention against flexibility trade-off.
*   The router puts a triage node in front, which asks for one word, `shallow` or `deep`. A
    conditional edge reads the verdict from the state:

    ```python
    builder.add_conditional_edges(
        "triage",
        lambda state: "gather" if state["depth"] == "deep" else "answer_directly",
        {"gather": "gather", "answer_directly": "answer_directly"},
    )
    ```

*   A reply naming neither word goes to the full pipeline, so an unclear verdict costs tokens
    instead of answer quality. Every reply so far was a single clean word.
*   Part 3 sends three questions through the router. The shallow one takes two nodes; the deep
    one and the ambiguous "Is PostgreSQL better than MySQL?" take six in every run. Over three
    runs:

    | | Run 1 | Run 2 | Run 3 |
    | :--- | ---: | ---: | ---: |
    | Shallow question, pipeline | 606 | 567 | 617 |
    | Shallow question, router | 78 | 78 | 78 |
    | Triage call on the deep question | 55 | 55 | 55 |
    | Break-even share of shallow traffic | 9% | 10% | 9% |

*   Routing pays once more than about 10% of questions are shallow. The deep path's extra cost
    is read from the triage node, which returns its own token count. An earlier version
    subtracted a pipeline run of the deep question from a router run of it, and the length
    difference of two separate generations swamped the 55 tokens: three runs gave break-even
    points of 2%, 37% and 15%.
*   Part 4 prints the edges of both compiled graphs, and runs without a key, since building a
    graph calls no model. The node functions, prompts and model are the same in both. The one
    conditional edge is the difference between about 600 tokens and 78 on the shallow
    question.
