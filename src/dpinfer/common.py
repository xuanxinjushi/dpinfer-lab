"""Shared setup for the HF transformers baseline scripts."""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = Path("~/dpinfer-lab/models/Qwen3-4B").expanduser()
DEFAULT_PROMPT = "Write a C++17 implementation of binary search."


def load_model(path=MODEL_PATH, device="cuda", dtype=torch.bfloat16):
    path = str(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        device_map=device,
        local_files_only=True,
    ).eval()
    return tokenizer, model


def build_inputs(tokenizer, prompt=DEFAULT_PROMPT, device="cuda"):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(device)


def warmup(model, inputs, max_new_tokens=16):
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
