# HF baseline: prefill vs decode — Qwen3-4B, 23-token prompt

Date: 2026-11-17

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (24 GiB) |
| Model | Qwen3-4B, BF16, `device_map="cuda"` |
| Stack | torch 2.14.1+cu130, transformers 5.23.0, HF eager, `DynamicCache` |
| Prompt | "Write a C++17 implementation of binary search." (chat template, thinking off) → 23 input tokens |
| Decoding | greedy, batch 1 |
| Script | `bench/hf_prefill_decode.py --decode-steps 256 --runs 5` |
| Env | `/home/wukong/miniconda3/envs/dpinfer/bin/python` |

Method: one forward over the full prompt (prefill, produces token 1), then 256 single-token
forwards reusing the KV cache (decode, tokens 2–257). Each forward is timed with CUDA events,
with no host sync between steps; argmax between steps is excluded. Warmup: one 16-token
`generate()` plus one 16-step manual loop. Output verified token-identical to greedy `generate()`.

## Results

| Metric | Value |
|---|---|
| Prefill (23 tok) | 14.73 ms median (min 14.50, max 14.83) |
| Decode per token | 13.25 ms median; p10 13.05, p90 13.38, p99 13.60 |
| Decode throughput | 75.55 tok/s median (min 75.52, max 75.60) |
| Peak VRAM allocated | 7.54 GiB |
| Matches `generate()` | True |

Per run:

```
run 0: prefill   14.78 ms | decode median  13.25 ms/tok p90  13.39 | total decode   3388.3 ms ->  75.55 tok/s
run 1: prefill   14.54 ms | decode median  13.25 ms/tok p90  13.38 | total decode   3387.7 ms ->  75.57 tok/s
run 2: prefill   14.73 ms | decode median  13.26 ms/tok p90  13.39 | total decode   3389.7 ms ->  75.52 tok/s
run 3: prefill   14.83 ms | decode median  13.26 ms/tok p90  13.39 | total decode   3388.6 ms ->  75.55 tok/s
run 4: prefill   14.50 ms | decode median  13.24 ms/tok p90  13.37 | total decode   3386.1 ms ->  75.60 tok/s
```

Earlier end-to-end `generate()` runs (`bench/hf_generate_timed.py`, 256 new tokens):
3.370 s / 75.96 tok/s and 3.361 s / 76.17 tok/s.

## Observations

- At 23 tokens, end-to-end `generate()` throughput is essentially decode throughput; prefill is
  one ~15 ms step out of ~3.4 s, and `generate()` adds negligible overhead over the manual loop.
- Prefill ≈ one decode step: at this length both are bound by reading the weights, not compute.
- Decode reaches roughly 60% of the memory-bandwidth limit (~8 GB of weights per step at
  ~1 TB/s → ~8 ms floor vs 13.25 ms measured).
- Run-to-run spread is < 0.1%.

## Open questions / next

- Is the ~5 ms/token gap to the bandwidth floor kernel-launch overhead? Test: static cache +
  CUDA graphs (or `torch.compile`) on the same loop. *Hypothesis, not yet tested.*
- Context-length sweep (1K, 4K, 8K, 16K, 32K) with RepoQA code context, 256 decode steps,
  ≥ 5 runs each, peak VRAM per length.
