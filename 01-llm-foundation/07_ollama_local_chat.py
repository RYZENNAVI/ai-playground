"""This script runs a reasoning model on your own machine through Ollama, with no API
key. It shows what a hosted API usually does for you: getting the model, streaming the
reply, and serving it over HTTP.

Set OLLAMA_HOST or OLLAMA_MODEL in .env to use another host or model. The run prints
five parts:
    1. Model download. The script checks whether the model is already pulled and
       downloads it only if not.
    2. One reply. One prompt goes to Ollama's REST API on port 11434, and the whole
       reply comes back at once.
    3. Streaming. A longer reply is printed piece by piece as the model writes it.
    4. Reasoning and answer. Ollama returns the model's reasoning in its own thinking
       field, apart from the answer. The script prints the length of each. Older
       Ollama versions put the reasoning inside the answer, wrapped in <think> tags.
       The script follows the newer behaviour of Ollama 0.34.2, the version it was
       tested with.
    5. FastAPI gateway. The script defines a FastAPI app around the model and prints
       the command that serves it.
"""

import json
import os
import sys

import requests

# Read the keys from the .env file at the repository root, if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Ensure UTF-8 output on the Windows terminal (model replies may contain emoji)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Set OLLAMA_HOST or OLLAMA_MODEL in .env to use another host or model.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
GENERATE_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
TAGS_ENDPOINT = f"{OLLAMA_HOST}/api/tags"
PULL_ENDPOINT = f"{OLLAMA_HOST}/api/pull"


# 1. Model download

def list_local_models() -> list:
    """Return the model tags currently available on the Ollama host."""
    response = requests.get(TAGS_ENDPOINT, timeout=10)
    response.raise_for_status()
    return [m.get("name", "") for m in response.json().get("models", [])]


def is_model_available(model: str = OLLAMA_MODEL) -> bool:
    """Check whether a model is pulled. A name without a tag means :latest, as in
    ollama run."""
    if ":" not in model:
        model += ":latest"
    return model in list_local_models()


def pull_model(model: str = OLLAMA_MODEL) -> None:
    """Download a model and print the progress of each layer as a percentage."""
    # A terminal can redraw one line with \r. Piped output cannot, so there the
    # script prints one line per 20% step instead.
    interactive = sys.stdout.isatty()
    last_len = 0
    last_report = {}  # layer digest -> last percentage reported

    with requests.post(PULL_ENDPOINT, json={"model": model}, stream=True, timeout=3600) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in event:
                raise RuntimeError(f"Pull failed: {event['error']}")

            status = event.get("status", "")
            total = event.get("total")
            # `completed` is absent on the first event of each layer.
            completed = event.get("completed") or 0

            if not total:
                print(f"  {status}")
                continue

            pct = completed / total * 100
            text = f"  {status}: {pct:5.1f}%  ({completed / 1e9:.2f}/{total / 1e9:.2f} GB)"

            if interactive:
                print(text.ljust(last_len), end="\r", flush=True)
                last_len = len(text)
            else:
                # Report each layer only every 20% to keep logs readable.
                milestone = int(pct // 20)
                if last_report.get(status) != milestone:
                    last_report[status] = milestone
                    print(text)

    if interactive:
        print()


def ensure_model(model: str = OLLAMA_MODEL) -> bool:
    """Download the model if it is missing. Return False when Ollama is unreachable."""
    try:
        if is_model_available(model):
            print(f"[Model] {model} is already here. Skipping the download.")
            return True

        print(f"[Model] {model} not found locally. Downloading (first run only)...")
        pull_model(model)
        print(f"[Model] {model} ready.")
        return True

    except requests.RequestException as exc:
        print(f"[Error] Cannot reach Ollama at {OLLAMA_HOST}: {exc}")
        print("Install Ollama from https://ollama.com/download and start the service.")
        return False


# 2. One reply

def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> tuple:
    """Return (reasoning, answer). Ollama 0.34.2 sends the reasoning in its own
    thinking field."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(GENERATE_ENDPOINT, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("thinking", "").strip(), data["response"].strip()


# 3. Streaming

def query_ollama_stream(prompt: str, model: str = OLLAMA_MODEL) -> tuple:
    """Print the reply as it streams, reasoning first, and return (reasoning, answer)."""
    payload = {"model": model, "prompt": prompt, "stream": True}
    thinking, chunks = [], []

    with requests.post(GENERATE_ENDPOINT, json=payload, stream=True, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[Skipped] Unreadable line: {exc}")
                continue

            thought = obj.get("thinking", "")
            if thought:
                if not thinking:
                    print("[Reasoning]")
                thinking.append(thought)
                print(thought, end="", flush=True)

            piece = obj.get("response", "")
            if piece:
                if not chunks:
                    print("\n\n[Answer]")
                chunks.append(piece)
                print(piece, end="", flush=True)

    print()
    return "".join(thinking).strip(), "".join(chunks).strip()


# 5. FastAPI gateway

def build_api_app():
    """Build a FastAPI app for POST /api/chat. The imports sit inside so the rest
    runs without FastAPI."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="Local LLM Gateway", description="Ollama-backed chat API")

    # Allow browser frontends on other origins to call this service directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ChatRequest(BaseModel):
        prompt: str
        model: str = OLLAMA_MODEL

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        try:
            _, answer = query_ollama(request.prompt, model=request.model)
            return {"response": answer}
        except requests.RequestException as exc:
            return {"error": f"Failed to reach Ollama: {exc}"}

    return app


if __name__ == "__main__":
    print("=== Local model via Ollama ===")
    print(f"Host  : {OLLAMA_HOST}")
    print(f"Model : {OLLAMA_MODEL}\n")

    # 1. Model download
    print("--- 1. Model download ---")
    if not ensure_model():
        sys.exit(1)
    print()

    try:
        # 2. One reply
        print("--- 2. One reply ---")
        reasoning, answer = query_ollama("Explain what a local LLM is in two sentences.")
        if reasoning:
            print(f"[Reasoning: {len(reasoning)} chars, not shown to users]")
        print(f"Answer: {answer}\n")

        # 3. Streaming
        print("--- 3. Streaming ---")
        reasoning, answer = query_ollama_stream("Write a binary search function in Python.")

        # 4. Reasoning and answer
        print("\n--- 4. Reasoning and answer ---")
        print(f"Reasoning length: {len(reasoning)} chars")
        print(f"Answer length   : {len(answer)} chars")

        # 5. FastAPI gateway
        print("\n--- 5. FastAPI gateway ---")
        print("Start the gateway from the repository root with:")
        print("  uvicorn --app-dir 01-llm-foundation 07_ollama_local_chat:build_api_app --factory --port 8000")
        print("Then POST to http://localhost:8000/api/chat")

    except requests.RequestException as exc:
        print(f"[Error] Lost connection to Ollama at {OLLAMA_HOST}: {exc}")
        print("Check that the Ollama service is still running.")
