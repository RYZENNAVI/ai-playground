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
    its second turn, because Gemini 3 requires a thought signature on each function call and LangChain's
    `ChatOpenAI` does not send it back. Script 07 makes 19 model calls in one run, more than
    the free tier's 15 requests a minute.
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

*   A `PromptTemplate` is text with named slots. LangChain reads the variable names from the
    text:

    ```python
    template = PromptTemplate.from_template(
        "What is a good name for a company that makes {product}?"
    )
    template.format(product="colorful socks")
    # What is a good name for a company that makes colorful socks?
    template.input_variables
    # ['product']
    ```

    Passing `input_variables` by hand changes nothing: declared as `[]`, the template still
    reports `['product']`. A missing value raises `KeyError: 'product'`, the same as Python's
    own `str.format`.
*   A `ChatPromptTemplate` renders to a list of messages, not one string. The instruction goes
    in the system message and stays fixed while only the human message changes:

    ```python
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You translate {source_language} into {target_language}. Reply with the translation only."),
        ("human", "{text}"),
    ])
    ```

    ```
    [system] You translate English into French. Reply with the translation only.
    [human] I love programming.
    ```

    Part 2 only renders the messages and sends nothing.
*   Part 3 pipes the template into the model and `StrOutputParser`, then runs template and
    model again without the parser. With the parser the result is the string
    `"J'adore la programmation."`; without it, an `AIMessage` carrying the same text.
*   The model is stateless: each request is judged only on the messages it carries. Memory
    here is a LangGraph checkpointer that stores each thread's messages, and a
    `MessagesPlaceholder` that marks where the graph pours them back in:

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

*   The test is a follow-up that names nothing:

    ```
    user: I am building a small tool that renames photo files by date.
    bot:  Use a script that reads each file's EXIF or filesystem date and renames it
          to a format like `YYYY-MM-DD_HHMMSS.jpg`.
    user: What should I call it?
    bot:  Call it **PhotoDater**.
    ```

    In six runs the second answer always named the photo tool, most often PhotoDater. The
    checkpointer then holds four messages, in order: human, ai, human, ai. The system message
    is not stored; the graph adds it on every call.
*   Sent alone, the same follow-up gets `Could you clarify what "it" refers to?`. The model and
    the question are the same, and only the first turn is missing. Every stored turn is sent
    again on every later call, so a transcript that keeps growing also grows the cost of each
    request.

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
    once with `.with_retry()`:

    ```
      retrying a flaky step by hand:
        attempt 1 failed: transient failure on attempt 1
        attempt 2 failed: transient failure on attempt 2
        manual loop: 'step succeeded' after 3 attempts
      retrying the same step with .with_retry():
        .with_retry(): 'step succeeded' after 3.64s, no loop or except clause written
    ```

    The two are not identical. `.with_retry()` retries on any exception by default and waits
    between attempts with exponential backoff and jitter, which is where the 3.6 to 4.4 seconds
    go. The hand-written loop catches only `RuntimeError` and does not wait.
*   Part 3 wraps three plain functions in `RunnableLambda` and invokes them like a model. They
    count words and guess sentiment from two word lists, turn CSV into JSON, and count lines:

    ```
    analyse: words=26 characters=140 sentiment=negative (hits 1+/2-)
    convert: {'name': 'Alice', 'age': '25', 'comment': 'Works exactly as described'} (3 rows)
    process: 3 lines
    ```

*   Part 4 runs five prompts about one review (summary, product area, sentiment, urgency,
    language) as a `RunnableParallel` and then one after another. The parallel call took 0.85
    to 1.10 seconds and the serial loop 3.34 to 4.34 seconds. `RunnableParallel` gives each
    branch its own thread, and the branches do not depend on each other.
*   Part 5 routes three inputs with `RunnableBranch`, which takes the first branch whose
    condition holds:

    ```python
    router = RunnableBranch(
        (lambda payload: "," in payload["content"], RunnableLambda(...)),   # convert CSV to JSON
        (lambda payload: payload["content"].count("\n") >= 2, LOCAL_STEPS["process"]),
        RunnableLambda(...),   # default
    )
    ```

    ```
    name,age,comment               -> [ { "name": "Alice", "age": "25", "comment": "Works exactly as describ...
    First line of the note         -> 3 lines
    hello                          -> short text, 5 characters, left as is
    ```

    The CSV check comes first because the CSV sample also has more than two line breaks.
*   Part 5 then shows when `|` accepts a plain function. A plain function piped into the router
    works, because LangChain wraps it. Two plain functions piped together raise
    `TypeError: unsupported operand type(s) for |: 'function' and 'function'`, because
    Python's `|` knows nothing about functions.
*   Part 6 runs the same chain with `stream` and with `invoke`. The first streamed chunk arrived
    after 0.28 to 0.57 seconds, and the blocking call returned after 0.84 to 1.33 seconds.
    Every step in this chain passes chunks along. A plain function in `RunnableLambda` would
    wait for the whole input and turn the stream back into one block.

## Script 03: A ReAct loop by hand

ReAct (reason and act) has the model write a Thought, then an Action and its input. The
program runs that tool, adds the result as an Observation, and calls the model again, until
the model writes a Final Answer. Script 03 does this with a chat completion call, a regex and
a `for` loop, with no agent framework.

*   The tools answer questions about a rule book of four rules (`R-001` to `R-004`) in two
    categories, `eligibility` and `supervision`. One tool searches the text, one lists a
    category, and one reads a rule by id. They are an ordinary dict:

    ```python
    TOOLS = {
        "search_rules":  (search_rules,  "Search the rule book by keywords. Input: two or three keywords."),
        "list_category": (list_category, "List the rules in one category. Input: one of eligibility, supervision."),
        "read_rule":     (read_rule,     "Read the full text of one rule. Input: a rule id such as R-001."),
    }
    ```

*   The prompt carries the tools as text. `{tools}` and `{tool_names}` are filled from the dict
    above, so a tool cannot be registered and left out of the prompt:

    ```
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
    ```

*   Generation stops at `Observation:`, so the program supplies the tool result and the model
    cannot invent it:

    ```python
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        stop=["Observation:"],
    )
    ```

    Two calls without the stop both went past the first `Action` and wrote the `Observation`
    themselves: `No rules found matching those keywords.` The real search finds R-002, so the
    invented result was also wrong.
*   Each reply goes back as an assistant message and each tool result as the following user
    message. Four test runs that appended everything to one growing user message also worked,
    so the turns are a choice here, not a fix for an observed failure.
*   A reply containing `Final Answer:` ends the loop. A reply matching `Action:` and
    `Action Input:` runs the tool, with only the first line of the input. Anything else is
    returned as `[unparsable reply]`. An earlier version also accepted some prose as a final
    answer. That never happened in 11 test loops, and it would have hidden a format break, so
    it was removed.
*   Part 2 asks a question that needs one lookup:

    ```
    question: What is the minimum a securities fund must raise before it closes?
      step 1: search_rules('minimum raise closing') -> [R-002] What is the minimum size a fund must raise? ...
      transcript: 3 messages
      passes: 2, tool calls: 1
    answer: A securities fund may not close below 10,000,000 in committed capital.
    ```

    One pass chooses the action and one turns the observation into an answer. When a result
    holds several rules, the step line lists their ids instead, for example
    `3 rules: R-001, R-002, R-003`.
*   Part 3 sends the same prompt with the tool block removed and `Action: one of [...]`
    replaced by `Action: the tool to use`. The functions are still registered:

    ```
      step 1: search_rule_book('minimum raise before closing securities fund') -> No tool named 'search_rule_book'.
      step 2: search('minimum raise before closing securities fund')           -> No tool named 'search'.
      step 3: lookup('minimum raise before closing securities fund')           -> No tool named 'lookup'.
      passes: 4, tool calls: 3
    answer: I cannot answer this question because I do not have access to the fund compliance
            rule book (no working search or lookup tool is available). ...
    ```

    The model guessed names such as `search_rule_book`, `lookup_rule` and `search`, and after
    two or three misses answered that it had no working tool. It never ran out of steps. A
    model only knows a tool exists if its name is in the prompt.
*   Part 4 asks a question the rule book does not cover:

    ```
    question: What tax rate applies to carried interest for this fund?
      step 1: search_rules('carried interest tax') -> No rule matches 'carried interest tax'.
      step 2: search_rules('carried interest')     -> No rule matches 'carried interest'.
      step 3: search_rules('tax rate')             -> No rule matches 'tax rate'.
      step 4: list_category('eligibility')         -> 2 rules: R-001, R-002
      step 5: list_category('supervision')         -> 2 rules: R-003, R-004
      passes: 6, tool calls: 5
    answer: The rule book does not cover this question. ...
    ```

    Every run took these five calls and ended with a normal `Final Answer`. The tools say what
    they missed (`No rule matches ...`), so the model can read a miss, try another tool, and
    list what the rule book does contain before it answers.
*   Each pass sends the whole transcript again, so the input grows with every step. On the
    question above, the six passes read 207, 265, 308, 349, 415 and 478 input tokens, 2022 in
    all, against 1242 if every pass were as short as the first.
*   Part 5 names a rule id, the only question that calls `read_rule`:

    ```
    question: Please read rule R-004 in full and tell me the exact reporting deadlines.
      step 1: read_rule('R-004') -> A quarterly report is due within 15 business days of quarter end, and
      passes: 2, tool calls: 1
    answer: Rule R-004 states that a quarterly report is due within 15 business days of quarter end,
            and an audited annual report is due within four months of year end.
    ```

## Script 04: The same loop with create_agent

`create_agent` runs the loop from script 03 over the model's native function calling: the
model returns tool calls, not text to parse.

*   The `@tool` decorator builds a schema from each function's name, argument types and
    docstring, and part 1 prints what the model receives:

    ```
    resolve_host(hostname):  Resolve a hostname to an address. Use this before assuming a host exists.
    ping_host(hostname):     Check whether a host answers on the network, and report the round trip time.
    check_interface(name):   Report the state of one local network interface, such as eth0 or eth1.
    search_logs(keyword):    Search the recent service log for a keyword and return the matching lines.
    ```

    The docstring is now runtime input. Editing it changes what the agent does, and neither a
    type checker nor a test notices.
*   The simulated network has three hosts. `billing.internal` resolves to `10.0.4.37` but does
    not answer. One interface is up and one is down, and the log holds five lines. The question
    names a symptom, not a host, so the agent has to read the log before it knows what to test.
*   Part 2 reads the trace from the tool calls and `ToolMessage` entries in the message list
    that `create_agent` returns:

    ```
    question: Checkout keeps failing with connection errors since 14:00. What is broken?
      call 1: search_logs(['checkout'])
      call 2: search_logs(['connection'])
      call 3: check_interface(['eth0'])
      call 4: check_interface(['eth1'])
        -> No log line contains 'checkout'.
        -> 14:02:11 ERROR pool: connection to billing.internal:5432 refused
        -> eth0 is up, address 10.0.4.9, gateway 10.0.4.1.
        -> eth1 is administratively down and has no address.
      call 5: resolve_host(['billing.internal'])
      call 6: ping_host(['billing.internal'])
        -> billing.internal resolves to 10.0.4.37.
        -> billing.internal (10.0.4.37) does not answer: request timed out.
      tool calls: 6
    answer: Failing component: the billing service host billing.internal (10.0.4.37), specifically
            its PostgreSQL listener on port 5432. ...
    ```

    Six runs took 6 or 7 calls and all named `billing.internal`. The first four calls go out
    together; the model then follows the host the log named and separates "does not resolve"
    from "does not answer". Nothing in the code specifies that order.
*   Part 3 asks the same question with `recursion_limit=4`. Six runs were all stopped after 6 or
    7 tool calls, with the evidence in hand but no answer written:

    ```
      call 5: resolve_host(['billing.internal'])
      call 6: ping_host(['billing.internal'])
        -> billing.internal resolves to 10.0.4.37.
        -> billing.internal (10.0.4.37) does not answer: request timed out.
      tool calls: 6
    answer: none, the agent hit its step limit first
    ```

    The limit counts graph steps, and each model turn and each tool round is one. The run is
    read with `agent.stream`, so the messages before the stop are kept, and the stop is printed
    as a stop rather than as an answer.
*   Part 4 gives the same four functions the docstrings `"""Do a lookup."""`,
    `"""Do a check."""`, `"""Do a check."""` and `"""Do a search."""`. Each vague tool calls the
    clear one, so the names, arguments and return strings stay the same and only the
    description changes:

    | Descriptions | Tool calls |
    | :--- | :--- |
    | Clear (part 2) | 6 to 7 over six runs |
    | Vague (part 4) | 12 to 18 over eight runs |

    The vague runs start by guessing hostnames (`checkout`, `checkout-service`,
    `checkout.internal`, `checkout.svc`) and pinging the gateway, and reach the log line later.
    Every answer read still named `billing.internal`. The description changed the cost of the
    run, not its conclusion.

## Script 05: MCP client and server

MCP, the Model Context Protocol, is a standard way to publish tools that any client can
discover and call. The client asks the server what tools it has at runtime instead of
importing them.

*   The file holds both halves. Run normally, it starts a second copy of itself with `--serve`
    as a subprocess. That copy is the server, and it speaks JSON-RPC over stdin and stdout. The
    parent is the client: a `ClientSession` over `stdio_client`. `main()` is the host, the
    application the model runs in. Nothing listens on a port.
*   The server publishes three tools over a folder of notes in `data/notes/` (an incident
    write-up, an onboarding note and a release checklist):

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

*   A stdio server sends its protocol messages on stdout, so logging belongs on stderr. With
    mcp 2.0.0 a stray line is not fatal: adding `print("server starting")` to `serve()` made the
    client log `Failed to parse JSONRPC message from server`, skip the line, and finish the run
    normally.
*   The filename comes from the model, so it is checked:

    ```python
    path = (NOTES_DIR / filename).resolve()
    if not path.is_relative_to(NOTES_DIR.resolve()) or path.suffix != ".txt":
        return None
    ```

    `NOTES_DIR / filename` alone confines nothing: `..` segments walk back out of the
    directory, and an absolute path replaces `NOTES_DIR` entirely. Resolving the path and
    checking it is still inside the directory is what limits reads. A rejected filename raises
    `ValueError`, and the SDK returns a result with `isError: true` and the text
    `Error executing tool count_words: No note named '../../.env'.` An earlier version returned
    `-1`, which the protocol reported as a success.
*   Parts 2 and 3 are the handshake and the tool listing. The client did not know the three
    names before it asked:

    ```
    --- 2. Handshake ---
      server: notes, protocol 2025-11-25

    --- 3. What the server advertises ---
      list_notes(): List the note files available, one filename per line.
      read_note(filename): Read one note file in full. Input: a filename from list_notes.
      count_words(filename): Count the words in one note file. Input: a filename from list_notes.
    ```

*   Part 4 turns each MCP tool into the chat API's tool format. Both sides already use JSON
    Schema, so the only change is the field name, `input_schema` to `parameters`:

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

    The same schemas worked unchanged with DeepSeek and with Gemini's OpenAI-compatible
    endpoint.
*   Part 5 is a tool-calling loop in which every call goes to the server:

    ```
    question: Which note explains why the checkout outage took so long to diagnose,
              and what follow-up did it recommend?
      list_notes({}) -> incident-2024-03-12.txt
        first call as it crossed the boundary:
        request  : tools/call {"name": "list_notes", "arguments": {}}
        response : {"content": [{"type": "text", "text": "incident-2024-03-12.txt\nonboarding.txt\nrelease-checklist.txt"}],
                    "structuredContent": {"result": "incident-2024-03-12.txt\nonboarding.txt\nrelease-checklist.txt"},
                    "isError": false, "resultType": "complete"}
      read_note({'filename': 'incident-2024-03-12.txt'}) -> Incident summary: checkout failures
    rounds: 3
    ```

    Twelve runs (ten with DeepSeek, two with Gemini) all took this path: one call per round, and only the
    incident note read. The trace line shows the first line of each result. The full result of
    the first call shows that the text holds all three filenames, and that the SDK also wraps a
    plain return value as `structuredContent`.
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
    `127.0.0.1:8931` in a background thread, and the script polls the card URL until it answers
    instead of sleeping a fixed time. The card is served at `/.well-known/agent-card.json`:

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

*   The caller reads the task route and the auth scheme from the card instead of hardcoding
    them:

    ```
    discovered RoomAvailabilityAgent v1.0 at /.well-known/agent-card.json
    learned endpoint: /api/tasks/availability
    learned auth:     ['bearer']
    ```

    If the provider moved its endpoint and updated its card, the caller would keep working
    unchanged.
*   The provider's room table is private. What comes back is an artifact derived from it:

    ```json
    {"task_id": "...", "status": "completed",
     "artifact": {"date": "2026-04-14", "attendees": 10,
                  "rooms": [{"room": "Cedar", "seats": 12}, {"room": "Aspen", "seats": 40}]}}
    ```

*   The caller makes the decision itself, the smallest room that fits or a cancellation:

    ```
    2026-04-14 for 10: provider returned Cedar (12), Aspen (40) -> confirmed in Cedar
    2026-04-15 for 30: provider returned none -> cancelled, no room fits
    2026-04-16 for 8: provider returned none -> cancelled, no room fits
    ```

    The first case has two rooms, so the smallest-room rule has a choice to make. The other
    two return nothing for different reasons: the 15th has only a 12-seat room, and the 16th
    has none. The provider never learns what the room is for, and the caller never sees the
    room table.
*   Parts 5 and 6 send tasks the card rules out:

    ```
    --- 5. A task the card's schema forbids ---
      status 400: attendees must be an integer between 1 and 60
      the card said attendees max is 60

    --- 6. A task without a bearer token ---
      status 401: missing or invalid bearer token
      the card declared authentication methods: ['bearer']
    ```

    The provider reads the attendee limits from its own card when it checks a task, so the
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

*   The state:

    ```python
    class ReviewState(TypedDict):
        question: str
        depth: Optional[Literal["shallow", "deep"]]
        verdict_raw: Optional[str]
        triage_tokens: Optional[int]
        facts: Optional[str]
        framing: Optional[str]
        options: Optional[str]
        choice: Optional[str]
        answer: Optional[str]
        visited: Annotated[list[str], operator.add]
        tokens: Annotated[int, operator.add]
    ```

    `visited` and `tokens` carry an `operator.add` reducer because several nodes write to them;
    without it, each node's value would replace the last one.
*   Each of the five analysis nodes makes one model call and adds one field:

    | Node | Instruction it sends | Field it adds |
    | :--- | :--- | :--- |
    | `gather` | List three concrete factors that bear on this question | `facts` |
    | `frame` | In one sentence, name the central trade-off these factors describe | `framing` |
    | `propose` | Propose two opposing courses of action | `options` |
    | `choose` | Pick one and state the condition under which it stops being right | `choice` |
    | `report` | Write a three-sentence answer using this reasoning | `answer` |

    `frame`, `propose` and `choose` also receive the question. An earlier version gave each of
    them only the previous field, and one of three runs drifted from "self-host or managed" to
    an answer about single-region monoliths. With the question added, three runs stayed on it.
*   A node reads a field an earlier node filled through a helper that names what is missing:

    ```python
    raise ValueError(f"node {node!r} needs '{field}', but no earlier node produced it. Check the edges.")
    ```

    A bare `state[field]` would raise a `KeyError` that does not say which edge is wrong.
*   Part 1 wires the nodes as a straight pipeline and runs a question that needs the analysis:

    ```python
    builder.add_edge(START, "gather")
    builder.add_edge("gather", "frame")
    builder.add_edge("frame", "propose")
    builder.add_edge("propose", "choose")
    builder.add_edge("choose", "report")
    builder.add_edge("report", END)
    ```

    ```
    question: Should a small team run its own database server instead of paying for a managed one?
      after gather   state holds: facts
      after frame    state holds: facts, framing
      after propose  state holds: facts, framing, options
      after choose   state holds: facts, framing, options, choice
      after report   state holds: facts, framing, options, choice, answer
      nodes run: 5, tokens: 663
    ```

    Three runs spent 601 to 663 tokens. The answer weighs self-hosting against managed hosting
    and names the condition that flips the decision.
*   Part 2 sends a question with a one-word answer through the same pipeline:

    ```
    question: What port does PostgreSQL listen on by default?
      nodes run: 5, tokens: 606
      the pipeline's 'framing' step invented a trade-off anyway:
        "PostgreSQL listens on port 5432 by default, and the central trade-off is between
         convention (a predictable, universally assumed default ...) and flexibility (the
         ability to override it via config, flags, or environment variables ...)."
    ```

    The answer is `5432`. The pipeline gave it, followed by two sentences about changing the
    port in `postgresql.conf` or with `-p`, because a node told to name the central trade-off
    names one whether or not there is one. All three runs framed the same convention against
    flexibility trade-off.
*   The router puts a triage node in front. A conditional edge reads the verdict from the
    state, so the path is data the graph produced, not a flag the caller passed in:

    ```python
    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        lambda state: "gather" if state["depth"] == "deep" else "answer_directly",
        {"gather": "gather", "answer_directly": "answer_directly"},
    )
    ```

*   The triage node asks for one word, `shallow` or `deep`, and keeps the raw reply in
    `verdict_raw`:

    ```python
    if "deep" in verdict.lower():      depth = "deep"
    elif "shallow" in verdict.lower(): depth = "shallow"
    else:                              depth = FALLBACK_DEPTH      # "deep"
    ```

    A reply naming neither word means the model broke the format. It goes to the full pipeline,
    so an unclear verdict costs tokens instead of answer quality. In the runs so far every
    reply was a single clean word, and the fallback never fired.
*   Part 3 sends three questions through the router:

    ```
    What port does PostgreSQL listen on by default?
      triage said shallow, nodes run: 2, tokens: 78
      -> triage -> answer_directly
      answer: PostgreSQL listens on port 5432 by default.

    Should a small team run its own database server instead of paying for a managed one?
      triage said deep, nodes run: 6, tokens: 654
      -> triage -> gather -> frame -> propose -> choose -> report

    Is PostgreSQL better than MySQL?
      triage said deep, nodes run: 6, tokens: 639
      -> triage -> gather -> frame -> propose -> choose -> report

    what routing saves or adds, in measured tokens:
      shallow question: pipeline 606, router 78 (saves 528)
      deep question:    the triage call adds 55
      break-even share of shallow traffic: 9%
    ```

    Over three runs:

    | | Run 1 | Run 2 | Run 3 |
    | :--- | ---: | ---: | ---: |
    | Shallow question, pipeline | 606 | 567 | 617 |
    | Shallow question, router | 78 | 78 | 78 |
    | Triage call on the deep question | 55 | 55 | 55 |
    | Break-even share of shallow traffic | 9% | 10% | 9% |

    Routing pays once more than about 10% of questions are shallow. The ambiguous question went
    deep in every run.
*   The deep path's extra cost is read from the triage node, which returns its own token count.
    An earlier version subtracted a pipeline run of the deep question from a router run of it.
    Those are two separate generations of five nodes each, and their length difference swamped
    the 55 tokens of triage: three runs gave 9, 236 and 83 tokens of overhead and break-even
    points of 2%, 37% and 15%.
*   Part 4 prints the edges of both compiled graphs, and runs without a key, since building a
    graph calls no model:

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

    The node functions, prompts and model are the same in both. The one conditional edge is
    the difference between about 600 tokens and 78 on the shallow question.
