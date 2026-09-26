"""This script loads the model weights with Transformers and generates text without a
serving layer. It shows the steps Ollama hides.

Set HF_MODEL_ID or HF_CACHE_DIR in .env to use another model or cache folder. The run
prints five parts:
    1. Model download. The weights come from the Hugging Face Hub, or from the local
       cache after the first run.
    2. Loading. The model goes onto the GPU when there is one, and the script prints
       the VRAM it takes.
    3. Chat template. The script prints the prompt string the model actually receives.
       The template for this model ends with an opening <think> tag, so the reply
       starts inside the reasoning and shows only the closing </think>.
    4. Generation. Only the new tokens are decoded, so the prompt is not repeated.
    5. Throughput in tokens per second.
"""

import os
import sys
import time

# Read the keys from the .env file at the repository root, if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# The same model Ollama serves as deepseek-r1:1.5b, here as unquantised bfloat16
# safetensors (about 3.5 GB) instead of a quantised GGUF (about 1.1 GB).
# On an RTX 5070 Ti Laptop (128 tokens, warmed up, mean of 3 runs), this path made
# 29 tok/s in 3.55 GB of VRAM, and Ollama made 309 tok/s in about 1.1 GB. When the
# GPU clocked down, the pair read 10 and 91, so the finding is the ratio of about 10x.
# Serve a model with Ollama or vLLM. Use this path for custom generation logic or as
# a starting point for fine-tuning. Set HF_ENDPOINT=https://hf-mirror.com for a mirror.
MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
CACHE_DIR = os.getenv("HF_CACHE_DIR", os.path.join(os.path.dirname(__file__), "weights"))


# 1. Model download

def ensure_weights(model_id: str = MODEL_ID, cache_dir: str = CACHE_DIR) -> str:
    """Download the checkpoint unless it is cached, and return its path. A cached
    checkpoint is only checked against the Hub."""
    from huggingface_hub import snapshot_download

    cached = os.path.isdir(cache_dir) and any(
        f.endswith(".safetensors") for _, _, files in os.walk(cache_dir) for f in files
    )
    print(f"[Weights] {model_id}")
    print(f"[Weights] {'cached, checking with the Hub' if cached else 'downloading (first run only)'}")

    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
    )
    print(f"[Weights] ready at {path}")
    return path


# 2. Loading

def load_model(model_path: str):
    """Load weights and tokenizer, placing the model on GPU when one exists."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    print(f"[Device] {torch.cuda.get_device_name(0) if has_cuda else 'CPU'}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",   # Use the dtype saved in the checkpoint (bfloat16 here)
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {params / 1e9:.2f}B parameters, dtype={model.dtype}")
    if has_cuda:
        print(f"[Model] VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return model, tokenizer


# 3. Chat template

def build_prompt(tokenizer, user_prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Render the messages with the tokenizer's chat template, since each model family
    uses its own role markers."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# 4. Generation

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


# 5. Throughput

def report(tokens: int, seconds: float) -> None:
    """Print tokens per second."""
    print(f"\n[Perf] {tokens} tokens in {seconds:.1f}s = {tokens / seconds:.1f} tok/s")


if __name__ == "__main__":
    print("=== Direct inference with Transformers ===\n")

    try:
        print("--- 1. Model download ---")
        path = ensure_weights()
        print()

        print("--- 2. Loading ---")
        model, tokenizer = load_model(path)
        print()
    except ImportError:
        print("[Error] Install the packages: pip install huggingface_hub transformers accelerate torch")
        sys.exit(1)

    print("--- 3. Chat template ---")
    preview = build_prompt(tokenizer, "Hello!")
    print(f"Rendered prompt:\n{preview}")
    print()

    print("--- 4. Generation ---")
    question = "What is 17 * 23?"
    print(f"Question: {question}\n")
    answer, tokens, elapsed = generate(model, tokenizer, question)
    print(answer)

    print("\n--- 5. Throughput ---")
    report(tokens, elapsed)
