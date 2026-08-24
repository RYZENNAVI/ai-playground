"""Adapt a small vision-language model to read a rendered instrument panel in a fixed format.

Demonstrates parameter-efficient fine-tuning when an image is part of the input:
    1. Render instrument panels locally, so every label is known exactly.
    2. Load a small vision-language checkpoint and inspect its two towers.
    3. Price the choice of attaching adapters to the language tower or to both.
    4. Build image and text batches with the prompt masked out of the loss.
    5. Score the untouched model field by field on held-out panels.
    6. Train the adapter and score again on the same panels.
    7. Separate what the adapter learned from what it did not.

Module 05: Fine-Tuning - Vision Adapter on Rendered Panels.
"""

import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = os.getenv("VLM_MODEL_ID", "HuggingFaceTB/SmolVLM-256M-Instruct")
CACHE_DIR = os.getenv("VLM_CACHE_DIR", str(Path(__file__).parent / "weights"))
PANEL_DIR = Path(__file__).parent / "outputs" / "panels"
ADAPTER_DIR = Path(__file__).parent / "outputs" / "vision_adapter"

RANK = 16
ALPHA = 32
LANGUAGE_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
TRAIN_PANELS = 96
EVAL_PANELS = 16
STEPS = 150
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
MAX_NEW_TOKENS = 32
SEED = 3407

IMAGE_SIZE = 384
GEARS = ("P", "R", "N", "D")
ZONES = ("low", "mid", "high")

QUESTION = (
    "Read the instrument panel. Answer with one line in exactly this form: "
    "GEAR: <P|R|N|D> | LAMP: <on|off> | NEEDLE: <low|mid|high> | ODO: <digits>"
)


def load_font(size):
    """Return a truetype face when the system has one, else the bundled default."""
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_panel(record, size):
    """Step 1. Draw one panel from its labels, so image and label cannot disagree.

    Generating the picture from the answer is what makes this measurable. A folder
    of photographs would need someone to write down what each one shows, and any
    mistake in that transcription becomes a permanent error in both the training
    signal and the score. Here the label is the input to the drawing, so the two
    are consistent by construction.
    """
    image = Image.new("RGB", (size, size), (18, 20, 24))
    draw = ImageDraw.Draw(image)
    centre = (size // 2, int(size * 0.42))
    radius = int(size * 0.30)

    draw.ellipse([centre[0] - radius, centre[1] - radius,
                  centre[0] + radius, centre[1] + radius],
                 outline=(210, 214, 222), width=4)
    for tick in range(11):
        angle = 3.14159 * (1.0 - tick / 10.0)
        inner = radius * 0.82
        start = (centre[0] + inner * torch.cos(torch.tensor(angle)).item(),
                 centre[1] - inner * torch.sin(torch.tensor(angle)).item())
        end = (centre[0] + radius * torch.cos(torch.tensor(angle)).item(),
               centre[1] - radius * torch.sin(torch.tensor(angle)).item())
        draw.line([start, end], fill=(170, 176, 186), width=3)

    # The needle angle is what the zone label means, so the two are derived from
    # the same number rather than chosen independently.
    fraction = record["needle_fraction"]
    angle = 3.14159 * (1.0 - fraction)
    tip = (centre[0] + radius * 0.92 * torch.cos(torch.tensor(angle)).item(),
           centre[1] - radius * 0.92 * torch.sin(torch.tensor(angle)).item())
    draw.line([centre, tip], fill=(235, 96, 74), width=7)
    draw.ellipse([centre[0] - 9, centre[1] - 9, centre[0] + 9, centre[1] + 9],
                 fill=(235, 96, 74))

    gear_font = load_font(int(size * 0.13))
    draw.text((int(size * 0.08), int(size * 0.80)), record["gear"],
              fill=(245, 247, 250), font=gear_font)

    lamp_colour = (255, 176, 32) if record["lamp"] == "on" else (52, 56, 64)
    lamp_box = [int(size * 0.44), int(size * 0.82), int(size * 0.54), int(size * 0.92)]
    draw.ellipse(lamp_box, fill=lamp_colour)

    odo_font = load_font(int(size * 0.075))
    draw.text((int(size * 0.62), int(size * 0.84)), record["odo"],
              fill=(228, 232, 238), font=odo_font)
    return image


def build_panels(count, seed, size, directory):
    """Create the labelled panels and keep a few on disk for inspection."""
    rng = random.Random(seed)
    records = []
    for index in range(count):
        fraction = rng.uniform(0.03, 0.97)
        zone = ZONES[0] if fraction < 0.34 else ZONES[1] if fraction < 0.67 else ZONES[2]
        record = {
            "gear": rng.choice(GEARS),
            "lamp": rng.choice(("on", "off")),
            "needle_fraction": fraction,
            "needle": zone,
            "odo": f"{rng.randrange(1000, 999999):06d}",
        }
        record["target"] = (f"GEAR: {record['gear']} | LAMP: {record['lamp']} | "
                            f"NEEDLE: {record['needle']} | ODO: {record['odo']}")
        record["image"] = render_panel(record, size)
        records.append(record)

    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records[:4]):
        record["image"].save(directory / f"panel_{index:02d}.png")
    print(f"Rendered {len(records)} panels at {size}x{size}")
    print(f"Wrote four samples to {directory}")
    print(f"Example label: {records[0]['target']}")
    return records


def load_vlm(model_id, cache_dir):
    """Load the processor and the vision-language weights."""
    from transformers import AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    hf_logging.set_verbosity_error()

    try:
        from transformers import AutoModelForImageTextToText as VisionModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as VisionModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = VisionModel.from_pretrained(model_id, cache_dir=cache_dir,
                                        dtype=dtype).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] {model_id}, {total / 1e6:.0f}M parameters, dtype={model.dtype}")
    return model, processor, device


def survey_towers(model, rank):
    """Step 2-3. Split the linear layers into the two towers and price each choice.

    A vision-language model is two networks joined by a projection. The image
    encoder turns pixels into embeddings, the language model consumes them
    alongside the text, and the connector maps between the two widths. The choice
    of which of the three to adapt is a real decision: if the answer depends on
    reading something the encoder already represents, adapting the language side
    is enough, and touching the encoder mostly costs memory.
    """
    towers = {"vision": [], "language": [], "connector": []}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lowered = name.lower()
        if "vision" in lowered or "image_encoder" in lowered:
            towers["vision"].append((name, module))
        elif "connector" in lowered or "modality_projection" in lowered:
            towers["connector"].append((name, module))
        else:
            towers["language"].append((name, module))

    total = sum(p.numel() for p in model.parameters())
    print(f"\n{'tower':>12} {'linear layers':>14} {'dense params':>15} {'share':>8}")
    for tower, entries in towers.items():
        dense = sum(module.out_features * module.in_features for _, module in entries)
        print(f"{tower:>12} {len(entries):>14} {dense:>15,} {dense / total:>7.1%}")

    print(f"\nAdapter cost at rank {rank}:")
    for label, chosen in (("language tower only", ("language",)),
                          ("language and vision", ("language", "vision"))):
        entries = [entry for tower in chosen for entry in towers[tower]]
        matched = [module for name, module in entries
                   if name.split(".")[-1] in LANGUAGE_TARGETS]
        params = sum(rank * (m.out_features + m.in_features) for m in matched)
        print(f"  {label:>22}: {len(matched):>4} modules, {params:>10,} parameters "
              f"({params / total:.3%} of the model)")
    return towers


def attach_adapter(model, rank, alpha, module_names):
    """Wrap exactly the named projections and report the trainable share.

    Full module paths are passed rather than name suffixes. A suffix such as
    `q_proj` matches in both towers, because an attention projection in the image
    encoder is named the same way as one in the language model, so asking for
    suffixes silently adapts the encoder as well. Naming the modules is what makes
    the tower choice in step 3 real instead of nominal.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=list(module_names),
        bias="none",
    )
    adapted = get_peft_model(model, config)
    print(f"Modules named for adaptation: {len(module_names)}")
    adapted.print_trainable_parameters()
    attached = sum(1 for name, _ in adapted.named_modules() if name.endswith("lora_A"))
    vision_attached = sum(1 for name, _ in adapted.named_modules()
                          if name.endswith("lora_A") and "vision" in name.lower())
    print(f"Adapters actually attached: {attached}, of which in the image encoder: "
          f"{vision_attached}")
    return adapted


def render_prompt(processor, question):
    """Render the chat template for a single image question."""
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": question}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def encode_example(processor, record, question, device):
    """Step 4. Build one training example and mask the prompt tokens.

    The prompt length is measured by encoding the prompt on its own with the same
    image, because the processor replaces the image placeholder with a block of
    image tokens whose count depends on the picture. Guessing that number would
    silently shift the mask and supervise the wrong positions, so the code
    verifies that the full sequence really does begin with the prompt sequence.
    """
    prompt = render_prompt(processor, question)
    prompt_batch = processor(text=prompt, images=[record["image"]],
                             return_tensors="pt")
    full_batch = processor(text=prompt + record["target"] + processor.tokenizer.eos_token,
                           images=[record["image"]], return_tensors="pt")

    prompt_length = prompt_batch["input_ids"].shape[1]
    full_ids = full_batch["input_ids"]
    if not torch.equal(full_ids[0, :prompt_length], prompt_batch["input_ids"][0]):
        raise RuntimeError("the full sequence does not start with the prompt sequence")

    labels = full_ids.clone()
    labels[0, :prompt_length] = -100
    labels[full_batch["attention_mask"] == 0] = -100
    batch = {key: value.to(device) for key, value in full_batch.items()}
    batch["labels"] = labels.to(device)
    return batch, prompt_length


def train(model, processor, records, device, steps, batch_size, learning_rate, seed):
    """Step 6. Update the adapter on rendered panels and report loss and cost.

    Examples are accumulated one at a time rather than padded into a real batch.
    Image token counts differ between pictures, so a naive stack would need
    padding rules for both the text and the pixel tensors; accumulating gradients
    over single examples reaches the same update with none of that machinery.
    """
    rng = random.Random(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()

    started = time.time()
    losses = []
    for step in range(1, steps + 1):
        optimiser.zero_grad(set_to_none=True)
        step_loss = 0.0
        for record in rng.sample(records, batch_size):
            batch, _ = encode_example(processor, record, QUESTION, device)
            loss = model(**batch).loss / batch_size
            loss.backward()
            step_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimiser.step()
        losses.append(step_loss)
        if step % 25 == 0 or step == 1:
            window = losses[-25:]
            print(f"  step {step:>3d}  loss {step_loss:.4f}  "
                  f"mean of last {len(window)} {sum(window) / len(window):.4f}")

    elapsed = time.time() - started
    print(f"Loss: {losses[0]:.4f} -> {sum(losses[-10:]) / 10:.4f} "
          f"(mean of the last ten steps)")
    print(f"Wall clock: {elapsed:.1f} s for {steps} steps "
          f"({elapsed / steps * 1000:.0f} ms per step)")
    if torch.cuda.is_available():
        print(f"Peak VRAM reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


FIELD_PATTERN = re.compile(
    r"GEAR:\s*([PRND])\s*\|\s*LAMP:\s*(on|off)\s*\|\s*NEEDLE:\s*(low|mid|high)"
    r"\s*\|\s*ODO:\s*(\d+)", re.IGNORECASE)


def parse_fields(text):
    """Pull the four fields out of an answer, or return None for each."""
    match = FIELD_PATTERN.search(text.replace("\n", " "))
    if not match:
        return None
    return {
        "gear": match.group(1).upper(),
        "lamp": match.group(2).lower(),
        "needle": match.group(3).lower(),
        "odo": match.group(4),
    }


def evaluate(model, processor, records, device, max_new_tokens, label, show=3):
    """Steps 5 and 7. Score each field separately instead of one accuracy number.

    Four fields differ in what they demand: three are choices among a handful of
    options that the encoder can plausibly separate, and the fourth asks the model
    to read six digits off a small picture. Reporting one combined figure would
    let the easy fields carry the hard one, and the interesting result here is
    precisely which of the four moved.
    """
    model.eval()
    fields = ("gear", "lamp", "needle", "odo")
    hits = {field: 0 for field in fields}
    shaped = 0
    samples = []
    for record in records:
        prompt = render_prompt(processor, QUESTION)
        batch = processor(text=prompt, images=[record["image"]], return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            output = model.generate(**batch, max_new_tokens=max_new_tokens,
                                    do_sample=False)
        completion = processor.batch_decode(
            output[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        parsed = parse_fields(completion)
        shaped += int(parsed is not None)
        if parsed:
            for field in fields:
                hits[field] += int(parsed[field] == record[field])
        samples.append((record, completion.strip(), parsed))

    count = len(records)
    print(f"\n{label}")
    print(f"  answers in the required shape: {shaped}/{count} ({shaped / count:.1%})")
    for field in fields:
        print(f"  {field:>7} correct: {hits[field]:>3}/{count} "
              f"({hits[field] / count:.1%})")
    for record, completion, parsed in samples[:show]:
        print(f"    expected {record['target']}")
        print(f"    produced {' '.join(completion.split())[:110]!r}")
    return shaped / count, {field: hits[field] / count for field in fields}


def main():
    torch.manual_seed(SEED)

    print("--- 1. Render instrument panels from their labels ---")
    panels = build_panels(TRAIN_PANELS + EVAL_PANELS, SEED, IMAGE_SIZE, PANEL_DIR)
    evaluation = panels[:EVAL_PANELS]
    training = panels[EVAL_PANELS:]
    print(f"Panels: {len(training)} for training, {len(evaluation)} held out")

    model, processor, device = load_vlm(MODEL_ID, CACHE_DIR)
    print(f"[Device] {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    print("\n--- 2-3. Split the towers and price the adapter choices ---")
    towers = survey_towers(model, RANK)
    language_modules = [name for name, _ in towers["language"]
                        if name.split(".")[-1] in LANGUAGE_TARGETS]

    print("\n--- 5. Score the untouched model on the held-out panels ---")
    base_shape, base_fields = evaluate(model, processor, evaluation, device,
                                       MAX_NEW_TOKENS, "Before training:")

    print("\n--- 4. Attach the adapter to the language tower ---")
    model = attach_adapter(model, RANK, ALPHA, language_modules)

    print("\n--- 6. Train on the rendered panels ---")
    train(model, processor, training, device, STEPS, BATCH_SIZE, LEARNING_RATE, SEED)

    print("\n--- 7. Score again and separate what moved from what did not ---")
    tuned_shape, tuned_fields = evaluate(model, processor, evaluation, device,
                                         MAX_NEW_TOKENS, "After training:")

    print(f"\n{'field':>8} {'before':>8} {'after':>8}")
    print(f"{'shape':>8} {base_shape:>7.1%} {tuned_shape:>7.1%}")
    for field in ("gear", "lamp", "needle", "odo"):
        print(f"{field:>8} {base_fields[field]:>7.1%} {tuned_fields[field]:>7.1%}")

    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    model.save_pretrained(ADAPTER_DIR)
    size = sum(f.stat().st_size for f in ADAPTER_DIR.rglob("*") if f.is_file())
    print(f"\nSaved the adapter to {ADAPTER_DIR} ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
