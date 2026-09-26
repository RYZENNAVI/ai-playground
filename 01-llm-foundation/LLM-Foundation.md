# LLM foundation

Scripts 01 to 06 call hosted chat models through an API. Scripts 07 and 08 run a model
on your own machine. This document explains what each script does and the ideas it
relies on.

| # | Script | What it shows |
| :---: | :--- | :--- |
| 01 | `01_chat_sentiment_analysis.py` | Chat roles, and a system message that turns the model into a classifier |
| 02 | `02_weather_function_calling.py` | Function calling (tool calling), with one call per city in a single reply |
| 03 | `03_table_multimodal_extraction.py` | An image and an instruction in one message, table returned as JSON |
| 04 | `04_ops_incident_handler.py` | A tool loop that diagnoses a database alert |
| 05 | `05_prompt_engineering.py` | Structured template, JSON mode, chain of thought, meta-prompting |
| 06 | `06_web_search_agent.py` | A search agent with a round cap. Wikipedia stands in for web search. |
| 07 | `07_ollama_local_chat.py` | A local model through Ollama: download, streaming, reasoning field, FastAPI gateway |
| 08 | `08_transformers_inference.py` | The same model loaded with Transformers: chat template, generation, throughput |

## Shared setup (scripts 01 to 06)

*   Every key lives in `.env` at the repository root, which `.gitignore` excludes. The
    scripts load it with `python-dotenv` and read it with `os.getenv()`.
*   DeepSeek, Gemini and OpenAI all offer an endpoint that speaks the OpenAI protocol.
    The scripts use the `openai` package for all three. Switching provider changes only
    the key, `base_url` and the model name:

    ```python
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(model="deepseek-chat", messages=messages)
    reply = response.choices[0].message.content
    ```

    Gemini's endpoint is `https://generativelanguage.googleapis.com/v1beta/openai/`.
*   `choices` holds the candidate replies, and the scripts read the text of the first one.

## Script 01: Chat roles and sentiment labels

*   A request is a list of messages, each with a role. `system` sets the model's job and
    rules. `user` carries the input. Later scripts add `assistant` (the model's own
    earlier message) and `tool` (the result of a function).
*   Part 1 asks a plain question with a generic system message. Part 2 keeps the same
    call and changes only the system message: "Classify the sentiment of the product
    review. Reply with only one word: Positive, Negative, or Neutral." The model becomes
    a classifier with a fixed set of answers.
*   Before the model picks the next token, temperature reshapes the probabilities. A low
    value sharpens them toward the likeliest token, and a high value flattens them. Part
    2 uses 0.1, so a review gets the same label almost every time.

## Script 02: Function calling (tool calling)

The model has no live weather data. It is given the JSON schema of a local function,
`get_current_weather`, and asks the script to run it.

*   The question goes out with `tools=TOOLS_SCHEMA`, which describes the function name,
    what it does and its `location` parameter.
*   `response.choices[0].message.tool_calls` holds nothing when the model answers
    directly. Otherwise each call has a `name` and JSON `arguments`, parsed with
    `json.loads`. The question "How is the weather in Shanghai and Shenzhen today?"
    usually returns two calls, one per city.
*   The script runs `get_current_weather(**fn_args)`. The function reads a fixed table of
    temperatures, so the flow can be tested without a real weather service.
*   The script appends the assistant message that asked for the calls, then one
    `{"role": "tool", "tool_call_id": ..., "content": ...}` per call. Every call needs its
    own `tool` message with the matching id, or the next request fails.
*   The script calls the model again with the whole history and prints the final answer.

## Script 03: An image in a chat message

*   The script draws an order table with Pillow and saves it to `outputs/order_table.png`.
*   The picture goes into an ordinary user message. The message content is a list of two
    parts: a `text` part with the instruction and an `image_url` part holding the PNG as
    a base64 data URL (`data:image/png;base64,...`).
*   A vision-language model reads the table and returns it as JSON. It is
    `gemini-3.1-flash-lite` with a Gemini key and `gpt-4o-mini` otherwise. The script
    prints the reply without parsing it, so it may come wrapped in a Markdown code block.

## Script 04: A tool loop

A database alert goes to the model. The system message tells it to check the server
first with `get_current_status`, then give a root-cause analysis and an action plan.
A simplified version of the loop:

```python
while True:
    response = client.chat.completions.create(model=model, messages=messages, tools=TOOLS_SCHEMA)
    message = response.choices[0].message
    messages.append(message)
    if not message.tool_calls:
        print(message.content)
        break
    for tool_call in message.tool_calls:
        messages.append({"tool_call_id": tool_call.id, "role": "tool", "content": get_current_status()})
```

*   The model decides when it has enough data. The loop ends when a reply comes without a
    tool call, and nothing else stops it. Script 06 adds a round limit.
*   The function draws connections, CPU and memory at random on every call. The connection
    count always stays above the alert threshold of 80.
*   The model only diagnoses. Nothing in the script executes its plan.

## Script 05: Prompt techniques

Every part goes through `get_completion`, which sends one system message and one user
message. `compose_prompt` builds the user message from sections under Markdown headings:
`# Objective`, an optional `# Thinking Requirement`, an optional `# Output Format` and
`# User Input`.

| Part | Technique | Temperature | What comes back |
| :--- | :--- | :---: | :--- |
| 1 | Structured template | 0 | From "Help me subscribe to a 100GB plan with a budget under $30/month.", the model reports 100GB of data, a price ceiling of $30 and no plan name. |
| 2 | JSON mode | 0 | The same prompt plus an output format and `response_format={"type": "json_object"}`. The reply is a JSON object with `name`, `price_limit` and `data_gb` that parses without cleanup. |
| 3 | Chain of thought | 0.1 | `cot=True` adds "Please analyze the conversation details step by step." The model checks a support reply against three rules one at a time. The reply offers the Unlimited Plan at $40/mo, but it costs $50/mo, so the verdict is Non-Compliant. |
| 4 | Meta-prompting | 0.7 | The model, cast as a senior prompt engineer, rewrites a weak system prompt for a support agent. The rewrite comes back in sections (role, plans, responsibilities, rules, tone) and limits the agent to the two listed plans. |

*   The headings keep instruction, rules and input apart, so the model can tell which
    text to follow and which to judge.
*   In part 3, asking for the steps before the verdict makes the model compare each rule
    with the reply, which is where the price mismatch shows up.

## Script 06: A search agent with a round cap

*   The question is "What have DeepSeek and OpenAI announced recently?" The model's
    knowledge stops at its training cutoff, so it gets a search tool.
*   The search is simulated on purpose. The tool queries the Wikipedia search API
    (MediaWiki `list=search`), which needs no key and returns titles and snippets as
    JSON. The model is told it searches the web. Each search returns up to three results.
    A failed request gives the model an error message instead of raising.
*   Script 04's loop only ends when the model stops calling tools. Here the loop runs at
    most three rounds. One round can hold several searches.
*   If the model is still searching after the third round, the script adds a message
    telling it to stop, and calls it once more with `tool_choice="none"`. That setting
    rules out another tool call, so the reply is text.

## Script 07: A local model through Ollama

*   Ollama serves models over a REST API on port 11434. The script uses three endpoints:
    `/api/tags` lists the pulled models, `/api/pull` downloads one and `/api/generate`
    produces a reply. A model name without a tag means `:latest`, as in `ollama run`.
*   `deepseek-r1:1.5b` is DeepSeek-R1-Distill-Qwen-1.5B, a small Qwen model fine-tuned on
    reasoning samples written by DeepSeek-R1. It is not the full R1. It writes its
    reasoning before the answer.
*   With `"stream": true`, Ollama sends one JSON object per line as the model writes, and
    the script prints each piece as it arrives.
*   Ollama 0.34.2 returns the reasoning in its own `thinking` field and the answer in
    `response`. Older versions put both in `response`, with the reasoning wrapped in
    `<think>` tags. The script reads the two fields and prints the length of each.
*   `build_api_app` defines a FastAPI gateway with `POST /api/chat`, which forwards the
    prompt to Ollama and returns the answer. CORS is open, so a browser frontend on
    another origin can call it. Start it from the repository root:

    ```
    uvicorn --app-dir 01-llm-foundation 07_ollama_local_chat:build_api_app --factory --port 8000
    ```

    `--factory` tells uvicorn to call `build_api_app` to get the app. FastAPI also serves
    an interactive page at `http://localhost:8000/docs` for trying the endpoint.

## Script 08: The same model with Transformers

*   `snapshot_download` fetches `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` from the
    Hugging Face Hub. `allow_patterns` keeps only the weights, configs and tokenizer
    files. A cached checkpoint is only checked against the Hub. Set
    `HF_ENDPOINT=https://hf-mirror.com` to use a mirror.
*   `torch_dtype="auto"` keeps the dtype saved in the checkpoint, bfloat16, two bytes per
    parameter. The model takes about 3.5 GB of VRAM.
*   Each model family marks roles with its own special tokens. `apply_chat_template`
    turns the message list into the format the model was trained on. For this model the
    string looks like
    `<｜begin▁of▁sentence｜>system text<｜User｜>question<｜Assistant｜><think>`. It ends
    with an opening `<think>`, so the reply starts inside the reasoning and shows only
    the closing `</think>`.
*   `model.generate` returns the prompt tokens followed by the new ones. The script
    slices off the prompt before decoding, so only the reply is printed.

The table compares 07 and 08 on an RTX 5070 Ti Laptop (128 tokens, warmed up, mean of
three runs):

| Path | Weights | Speed | VRAM |
| :--- | :--- | :--- | :--- |
| Ollama (07) | GGUF Q4_K_M | 309 tok/s | about 1.1 GB |
| Transformers (08) | bfloat16 safetensors | 29 tok/s | 3.55 GB |

When the GPU clocked down, Ollama made 91 tok/s and Transformers 10, so the number to
trust is the ratio, about 10x. Two things explain it:

*   Generating a token reads every weight once, so speed is limited by memory bandwidth.
    4-bit weights are about a third the size of bfloat16 ones, which accounts for
    roughly 3x.
*   The engine accounts for the rest. Ollama builds on the GGML library from llama.cpp,
    with fused GPU kernels written for generation. Transformers runs a Python loop per
    token and launches many small kernels.

Serve a model with Ollama or vLLM. Use the Transformers path to see the internals,
customise generation or prepare for fine-tuning.
