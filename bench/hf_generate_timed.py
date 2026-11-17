"""Single timed greedy generation (256 tokens) after warmup; prints throughput and output."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import time

import torch

from dpinfer.common import build_inputs, load_model, warmup

tokenizer, model = load_model()
inputs = build_inputs(tokenizer)
n_in = inputs["input_ids"].shape[-1]

warmup(model, inputs)
torch.cuda.synchronize()

start = time.perf_counter()
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

num_generated = output.shape[-1] - n_in

print(f"Input tokens: {n_in}")
print(f"Generated tokens: {num_generated}")
print(f"Elapsed: {elapsed:.3f} seconds")
print(f"Overall generation throughput: {num_generated / elapsed:.2f} tokens/s")
print(tokenizer.decode(output[0, n_in:], skip_special_tokens=True))
