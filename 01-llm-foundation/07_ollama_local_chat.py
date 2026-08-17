"""Local model deployment and inference via the Ollama REST API.

Demonstrates the full local-deployment stack for private LLM hosting:
    1. Weight acquisition: check whether the model is already pulled and
       download it only when missing (idempotent, safe to re-run).
    2. Ollama single-shot generation via the built-in REST API (port 11434).
    3. Ollama streaming responses (typewriter effect, low first-token latency).
    4. Reasoning-model output parsing: split `<think>` chain-of-thought from
       the final answer (DeepSeek-R1 distilled models emit both).
    5. FastAPI + CORS microservice wrapping a local model as an HTTP backend.

Module 01: LLM Foundation — Local Model Deployment.
"""

import json
import os
import re
import sys

import requests

# Automatically load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
except ImportError:
    pass

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Configuration — override via .env if your Ollama host or model differs
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
GENERATE_ENDPOINT = f"{OLLAMA_HOST}/api/generate"
TAGS_ENDPOINT = f"{OLLAMA_HOST}/api/tags"
PULL_ENDPOINT = f"{OLLAMA_HOST}/api/pull"


# ---------------------------------------------------------------------------
# 1. Weight acquisition — pull only what is missing
# ---------------------------------------------------------------------------
def list_local_models() -> list:
    """Return the model tags currently available on the Ollama host."""
    response = requests.get(TAGS_ENDPOINT, timeout=10)
    response.raise_for_status()
    return [m.get("name", "") for m in response.json().get("models", [])]


def is_model_available(model: str = OLLAMA_MODEL) -> bool:
    """Check whether a model has already been pulled.

    Ollama reports tags as "name:tag". An untagged request such as
    "deepseek-r1" is satisfied by any tag of that name, which mirrors how
    the `ollama run` command resolves models.
    """
    local = list_local_models()
    if model in local:
        return True
    return ":" not in model and any(tag.split(":")[0] == model for tag in local)


def pull_model(model: str = OLLAMA_MODEL) -> None:
    """Download a model, printing progress as it streams.

    /api/pull emits one JSON object per line; layer downloads carry
    `completed` and `total` byte counts that we render as a percentage.
    """
    # A TTY can be redrawn with \r; piped output cannot, so there we print a
    # sparse line-per-milestone log instead of thousands of redraw frames.
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
    """Make sure `model` is available locally, downloading it only if missing.

    Idempotent: re-running is a cheap no-op once the model is present.
    Returns True when the model is ready, False when Ollama is unreachable.
    """
    try:
        if is_model_available(model):
            print(f"[Model] {model} already present — skipping download.")
            return True

        print(f"[Model] {model} not found locally. Downloading (first run only)...")
        pull_model(model)
        print(f"[Model] {model} ready.")
        return True

    except requests.RequestException as exc:
        print(f"[Error] Cannot reach Ollama at {OLLAMA_HOST}: {exc}")
        print("Install Ollama from https://ollama.com/download and start the service.")
        return False


# ---------------------------------------------------------------------------
# 2. Single-shot generation
# ---------------------------------------------------------------------------
def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send one prompt to a local Ollama model and return the full response."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(GENERATE_ENDPOINT, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


# ---------------------------------------------------------------------------
# 3. Streaming generation
# ---------------------------------------------------------------------------
def query_ollama_stream(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Stream a response token by token, printing as it arrives.

    Ollama emits one JSON object per line while generating, so the client can
    render output immediately instead of waiting for the whole completion.
    """
    payload = {"model": model, "prompt": prompt, "stream": True}
    chunks = []

    with requests.post(GENERATE_ENDPOINT, json=payload, stream=True, timeout=120) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[Stream Notice] Failed to parse chunk: {exc}")
                continue
            piece = obj.get("response", "")
            chunks.append(piece)
            print(piece, end="", flush=True)

    print()  # newline after the stream completes
    return "".join(chunks)


# ---------------------------------------------------------------------------
# 4. Reasoning-model output parsing
# ---------------------------------------------------------------------------
def split_reasoning(raw_output: str):
    """Separate a reasoning model's `<think>` block from its final answer.

    DeepSeek-R1 distilled models wrap their chain-of-thought in <think> tags.
    Production systems usually log the reasoning but show users only the answer.

    Returns:
        (reasoning, answer) — reasoning is an empty string when no tag exists.
    """
    match = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL)
    if not match:
        return "", raw_output.strip()

    reasoning = match.group(1).strip()
    answer = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
    return reasoning, answer


# ---------------------------------------------------------------------------
# 5. FastAPI microservice wrapper
# ---------------------------------------------------------------------------
def build_api_app():
    """Build a FastAPI app exposing the local model at POST /api/chat.

    Imports live inside the function so the demos above run without FastAPI
    installed. Serve with:
        uvicorn 07_ollama_local_chat:build_api_app --factory --port 8000
    """
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
            answer = query_ollama(request.prompt, model=request.model)
            return {"response": answer}
        except requests.RequestException as exc:
            return {"error": f"Failed to reach Ollama: {exc}"}

    return app


def call_local_api(prompt: str, endpoint: str = "http://localhost:8000/api/chat") -> dict:
    """Client helper for the FastAPI service above (run the server first)."""
    response = requests.post(endpoint, json={"prompt": prompt}, timeout=120)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    print("=== Local Model Deployment via Ollama ===")
    print(f"Host  : {OLLAMA_HOST}")
    print(f"Model : {OLLAMA_MODEL}\n")

    # Make sure the weights are there before anything else runs.
    print("--- 1. Weight acquisition ---")
    if not ensure_model():
        sys.exit(1)
    print()

    try:
        print("--- 2. Single-shot generation ---")
        reply = query_ollama("Introduce yourself in one short paragraph.")
        reasoning, answer = split_reasoning(reply)
        if reasoning:
            print(f"[Reasoning trace, {len(reasoning)} chars — hidden from end users]")
        print(f"Answer: {answer}\n")

        print("--- 3. Streaming generation ---")
        raw = query_ollama_stream("Write a binary search function in Python.")

        print("\n--- 4. Reasoning / answer split ---")
        reasoning, answer = split_reasoning(raw)
        print(f"Reasoning length: {len(reasoning)} chars")
        print(f"Answer length   : {len(answer)} chars")

        print("\n--- 5. FastAPI service ---")
        print("Start the gateway with:")
        print("  uvicorn 07_ollama_local_chat:build_api_app --factory --port 8000")
        print("Then POST to http://localhost:8000/api/chat")

    except requests.RequestException as exc:
        print(f"[Error] Lost connection to Ollama at {OLLAMA_HOST}: {exc}")
        print("Check that the Ollama service is still running.")
