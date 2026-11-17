"""Smoke test: load the model, generate 256 tokens greedily, print the output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dpinfer.common import build_inputs, load_model

tokenizer, model = load_model()
inputs = build_inputs(tokenizer)
print("Input shape:", inputs["input_ids"].shape)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

print(tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
