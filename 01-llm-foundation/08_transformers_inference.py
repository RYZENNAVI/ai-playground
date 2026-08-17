"""Direct weight loading and inference with Transformers (no Ollama runtime).

Demonstrates full-control local inference on raw model weights:
    1. Weight acquisition from HuggingFace Hub, skipped when already cached.
    2. Loading a checkpoint with automatic device placement (CUDA or CPU).
    3. Applying the model's chat template to build a prompt.
    4. Tokenising, generating, and decoding only the newly produced tokens.
    5. Reporting VRAM usage and throughput to size real deployments.

Module 01: LLM Foundation — Transformers Inference.
"""

import os
import sys
import time

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
# Configuration — override via .env
# ---------------------------------------------------------------------------
# Same model Ollama serves as `deepseek-r1:1.5b`, but in full-precision
# safetensors (~3.5 GB) rather than quantised GGUF (~1.1 GB).
#
# Benchmarked on one RTX 5070 Ti Laptop, 128 tokens, warmed up, mean of 3 runs:
#     Transformers bfloat16 ....   29 tok/s, 3.55 GB VRAM
#     Ollama GGUF Q4_K_M ......   309 tok/s, ~1.1 GB VRAM
# Clock state shifts both numbers together — throttled the pair read 10 and 91 —
# so treat the ~10x ratio, not the absolute values, as the finding.
# Quantisation plus llama.cpp's fused kernels win decisively on throughput, so
# serve with Ollama/vLLM in production. Use this path when you need the raw
# graph, custom generation logic, or a starting point for fine-tuning.
MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
CACHE_DIR = os.getenv("HF_CACHE_DIR", os.path.join(os.path.dirname(__file__), "weights"))


# ---------------------------------------------------------------------------
# 1. Weight acquisition
# ---------------------------------------------------------------------------
def ensure_weights(model_id: str = MODEL_ID, cache_dir: str = CACHE_DIR) -> str:
    """Download the checkpoint unless it is already cached, and return its path.

    `snapshot_download` verifies existing files by hash, so a second run costs
    almost nothing. Set HF_ENDPOINT=https://hf-mirror.com to use a mirror.
    """
    from huggingface_hub import snapshot_download

    cached = os.path.isdir(cache_dir) and any(
        f.endswith(".safetensors") for _, _, files in os.walk(cache_dir) for f in files
    )
    print(f"[Weights] {model_id}")
    print(f"[Weights] {'cached — verifying' if cached else 'downloading (first run only)'}")

    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    print(f"[Weights] ready at {path}")
    return path


# ---------------------------------------------------------------------------
# 2. Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    """Load weights and tokenizer, placing the model on GPU when one exists."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    print(f"[Device] {torch.cuda.get_device_name(0) if has_cuda else 'CPU'}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",   # bfloat16/float16 on GPU, float32 on CPU
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {params / 1e9:.2f}B parameters, dtype={model.dtype}")
    if has_cuda:
        print(f"[Model] VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. Chat template
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Turn plain messages into the exact string this model was trained on.

    Every model family uses different role markers; the tokenizer ships the
    correct template, so never hand-concatenate role tags yourself.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# 4. Generation
# ---------------------------------------------------------------------------
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> tuple:
    """Run one generation pass and return (text, tokens_generated, seconds)."""
    text = build_prompt(tokenizer, prompt)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    started = time.perf_counter()
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - started

    # Drop the prompt tokens so only the completion is decoded.
    new_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    answer = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
    return answer, len(new_ids[0]), elapsed


# ---------------------------------------------------------------------------
# 5. Throughput reporting
# ---------------------------------------------------------------------------
def report(tokens: int, seconds: float) -> None:
    """Print generation speed — the number that decides deployment sizing."""
    print(f"\n[Perf] {tokens} tokens in {seconds:.1f}s = {tokens / seconds:.1f} tok/s")


if __name__ == "__main__":
    print("=== Transformers Direct Inference ===\n")

    print("--- 1. Weight acquisition ---")
    try:
        path = ensure_weights()
    except ImportError:
        print("[Error] Run: pip install huggingface_hub transformers torch")
        sys.exit(1)
    print()

    print("--- 2. Model loading ---")
    model, tokenizer = load_model(path)
    print()

    print("--- 3. Chat template ---")
    preview = build_prompt(tokenizer, "Hello!")
    print(f"Rendered prompt:\n{preview}")
    print()

    print("--- 4. Generation ---")
    question = "What is 17 * 23? Think step by step."
    print(f"Question: {question}\n")
    answer, tokens, elapsed = generate(model, tokenizer, question)
    print(answer)

    print("\n--- 5. Throughput ---")
    report(tokens, elapsed)
