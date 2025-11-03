#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from PIL import Image
from huggingface_hub import login
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

def parse_args():
    p = argparse.ArgumentParser(
        description="Run MedGemma locally (image VQA or text-only)."
    )
    p.add_argument("--model", default="google/medgemma-4b-it",
                   help="Model id on Hugging Face.")
    p.add_argument("--image", type=str, default=None,
                   help="Path to an image. If omitted, runs text-only chat.")
    p.add_argument("--prompt", type=str, default="What are the key findings in this picture?",
                   help="User prompt / question.")
    p.add_argument("--system", type=str, default="You are a helpful medical assistant.",
                   help="System prompt (text-only mode will use this).")
    p.add_argument("--no-4bit", action="store_true",
                   help="Disable 4-bit quantization (needs more VRAM).")
    p.add_argument("--think", action="store_true",
                   help="Enable thinking mode (for 27B variants).")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Computation dtype.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device map. 'auto' lets HF decide.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    return p.parse_args()

def get_torch_dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]

def main():
    args = parse_args()

    # --- Login via HF_TOKEN if present ---
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[Info] Found HF_TOKEN in env, logging in…")
        login(token=hf_token)
    else:
        print("[Warn] HF_TOKEN not set. If the repo requires auth, set it via `export HF_TOKEN=...`")

    # --- Device & dtype ---
    dtype = get_torch_dtype(args.dtype)
    device_map = "auto" if args.device == "auto" else args.device

    # --- 4bit quantization config (optional) ---
    quantization_config = None
    if not args.no_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        print("[Info] Using 4-bit quantization (bitsandbytes).")

    print(f"[Info] Loading processor: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)

    print(f"[Info] Loading model: {args.model}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )

    # --- Build chat messages ---
    if args.image:
        # Image VQA
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found: {args.image}")

        image = Image.open(args.image).convert("RGB")
        content = [{"type": "image"}, {"type": "text", "text": args.prompt}]
        if args.think:
            content.append({"type": "text", "text": "<think>"})

        messages = [{"role": "user", "content": content}]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=[image], text=prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        print("[Info] Generating (image + text)...")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        
        gen = out[0][input_len:]
        decoded = processor.decode(gen, skip_special_tokens=True)

        print("\n=== Model Output ===")
        # Handle thinking mode output
        if "27b" in args.model and args.think and "<end_thought>" in decoded:
            try:
                thought, final_response = decoded.split("<end_thought>")
                thought = thought.replace("thought\n", "")
                print("--- MedGemma thinking ---")
                print(thought)
                decoded = final_response
            except ValueError:
                pass
        print(decoded)

    else:
        # Text-only chat
        system_content = [{"type": "text", "text": args.system}]
        user_content = [{"type": "text", "text": args.prompt}]
        if args.think:
            user_content.append({"type": "text", "text": "<think>"})

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        print("[Info] Generating (text-only)...")
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        
        gen = out[0][input_len:]
        decoded = processor.decode(gen, skip_special_tokens=True)

        print("\n=== Model Output ===")
        # Handle thinking mode output
        if "27b" in args.model and args.think and "<end_thought>" in decoded:
            try:
                thought, final_response = decoded.split("<end_thought>")
                thought = thought.replace("thought\n", "")
                print("--- MedGemma thinking ---")
                print(thought)
                decoded = final_response
            except ValueError:
                pass
        print(decoded)

if __name__ == "__main__":
    main()
