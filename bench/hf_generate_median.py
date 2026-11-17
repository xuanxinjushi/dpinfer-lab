"""Median end-to-end time over 5 greedy runs of exactly 256 new tokens (prefill + decode, not isolated)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import statistics
import time

import torch

from dpinfer.common import build_inputs, load_model, warmup

N_RUNS = 5
NEW_TOKENS = 256

tokenizer, model = load_model()
inputs = build_inputs(tokenizer)

warmup(model, inputs)

gen_times = []
with torch.inference_mode():
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.generate(
            **inputs,
            max_new_tokens=NEW_TOKENS,
            min_new_tokens=NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        torch.cuda.synchronize()
        gen_times.append(time.perf_counter() - start)

median = statistics.median(gen_times)
print("Median generation time:", median)
print("Median overall throughput:", NEW_TOKENS / median)
