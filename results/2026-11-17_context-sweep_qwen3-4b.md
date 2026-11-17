# Context-length sweep — Qwen3-4B, HF eager, RTX 4090

Date: 2026-11-17 · Raw data: `results/context_sweep/2026-11-17/` (one JSON per config, `summary.csv`,
`analysis.json`, `decode_profile.json`, `context_sweep.png`)

## Question

How do prefill and per-token decode cost grow with context length on real code, and how much of
decode time at long context is attention / KV-cache work (the part a CPU/GPU KV hierarchy would touch)?

## Setup

| | |
|---|---|
| GPU | RTX 4090 24 GiB, driver 580.126.09 |
| Stack | torch 2.14.1+cu130, transformers 5.23.0, `attn_implementation=sdpa`, BF16, batch 1 |
| Model | Qwen3-4B (36 layers, 32 Q / 8 KV heads, head_dim 128); KV cache = 144 KiB/token |
| Prompts | RepoQA 2024-06-23 "search needle function", needle 0 of one repo per language: `psf/black`, `skypjack/uvw`, `google/gson`, `cheeriojs/cheerio`, `serde-rs/serde`, `fatedier/frp` |
| Context build | repo files in dependency order with path headers; code window cut at whole lines with the needle at 50% depth; full chat-templated prompt within ~1% below target |
| Lengths | 1K, 4K, 8K, 16K, 32K tokens |
| Timing | manual greedy loop, `DynamicCache`; prefill = 1 forward (logits for last position only), decode = 256 single-token forwards; CUDA events per forward; 1 warmup + 5 timed runs per config |
| Check | manual loop is token-identical to greedy `generate()` up to the model's own end of answer, 30/30 configs |
| Commands | `bench/hf_context_sweep.py`, `bench/analyze_context_sweep.py`, `bench/hf_decode_profile.py --repo psf/black --lengths 1024 8192 32768` |

Background GPU use at start: 1.9 GiB (Xorg sessions, Chrome, an idle CosyVoice worker). The CosyVoice
worker (idle, 0.6 GiB) was stopped mid-sweep; it was not computing, so timings are unaffected.

## Results

### Prefill and decode vs context (median over 6 repos)

| Target | Prompt tok | KV cache | Prefill | Prefill tok/s | Decode ms/tok | vs 1K | Decode tok/s | Bandwidth-model ms | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,010 | 0.16 GiB | 63.6 ms | 15,886 | 13.62 | 1.00 | 73.5 | 13.62 | 7.74 GiB |
| 4K | 4,082 | 0.58 GiB | 287 ms | 14,224 | 14.77 | 1.08 | 67.7 | 14.37 | 8.37 GiB |
| 8K | 8,178 | 1.14 GiB | 676 ms | 12,103 | 16.25 | 1.19 | 61.5 | 15.38 | 9.23 GiB |
| 16K | 16,377 | 2.27 GiB | 1.62 s | 10,104 | 20.03 | 1.47 | 49.8 | 17.38 | 10.96 GiB |
| 32K | 32,758 | 4.52 GiB | 4.25 s | 7,717 | 28.95 | 2.13 | 34.6 | 21.39 | 14.42 GiB |

- Timings are content-independent: decode medians differ by ≤ 0.03 ms/token across the six repos at
  every length; run-to-run spread is < 0.5%.
- "Bandwidth model" = weights + KV bytes per step at the 1K configuration's effective bandwidth
  (603 GB/s). Measured decode exceeds it increasingly with context: +35% at 32K.
- Linear fit: decode ≈ 12.6 ms + 0.49 ms per 1K context tokens. The context-dependent term implies
  only ~303 GB/s per KV byte, about half the ~639 GB/s of the fixed (weight) term. The kernel profile
  below explains why.
- Prefill throughput halves from 1K to 32K (attention is quadratic in prompt length).

### Where decode time goes (one repo, `psf/black`; GPU kernel time per decode step)

| Context | Cache | Wall ms/tok | Kernel ms | Weight GEMV | Attention | KV cat/copy | Other |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1K | Dynamic | 13.6 | 12.2 | 9.1 (75%) | 0.4 (3%) | 0.5 (4%) | 2.2 (18%) |
| 8K | Dynamic | 16.3 | 15.0 | 9.3 (62%) | 1.1 (8%) | 2.3 (15%) | 2.2 (15%) |
| 32K | Dynamic | 29.0 | 27.6 | 9.4 (34%) | 6.5 (24%) | 9.4 (34%) | 2.3 (8%) |
| 1K | Static | 15.9 | 14.1 | 9.2 (65%) | 2.0 (14%) | 0.3 (2%) | 2.6 (19%) |
| 8K | Static | 28.0 | 26.5 | 9.3 (35%) | 14.2 (54%) | 0.3 (1%) | 2.7 (10%) |
| 32K | Static | 70.0 | 68.2 | 9.1 (13%) | 56.0 (82%) | 0.3 (1%) | 2.7 (4%) |

Kernels are bucketed by name: GEMV/GEMM, flash/fmha attention, `CatArrayBatchedCopy`, other
(norms, RoPE, elementwise ops).

- **`DynamicCache` copies the whole KV cache on every decode step.** It appends with `torch.cat`, so
  each step reads and re-writes all K/V. At 32K that is ~9.7 GB of traffic in 9.3 ms (≈ peak DRAM
  bandwidth), 34% of decode time. This is an artifact of the HF cache implementation, not of
  attention.
- **Attention itself is efficient with `DynamicCache`.** SDPA picks the FlashAttention split-KV
  decode kernel: 6.3 ms to read 4.83 GB of KV at 32K ≈ 765 GB/s.
- **`StaticCache` is worse in this eager setup.** The preallocated cache needs an attention mask, so
  SDPA falls back to the memory-efficient (cutlass) kernel, which parallelises poorly for a single
  query token: 56 ms at 32K, 2.4× slower overall than `DynamicCache`. (Not tested with
  `torch.compile`, which is how HF intends `StaticCache` to be used.)
- **Estimate, not measured:** without the concatenation copy, 32K decode would be about 18 ms of
  kernels (≈ 20 ms/token wall), close to the bandwidth model's 21.4 ms. At that point attention
  is ~35% of decode time at 32K and ~9% at 8K.

### Needle retrieval (does the answer contain the needle function's name?)

| Context | 1K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|
| Found | 6/6 | 6/6 | 5/6 | 5/6 | 1/6 |

At 32K the model mostly hallucinates plausible functions (`gson`, `uvw`) or returns a different real
function (`black`, `serde`). The `cheerio` misses from 8K return the neighbouring `css` function
instead of `setCss`. Only `frp` is retrieved at every length. This is n=6 (one needle per repo, one
depth) with a loose name-match metric, so treat it as a coarse quality signal only.

## Conclusions

1. The HF eager baseline overstates long-context decode cost. About a third of the 32K decode step is
   `DynamicCache`'s full-cache copy, which a preallocated / paged cache with in-place writes avoids.
   Comparisons against this baseline would credit any method with that artifact's removal, so it
   must not be the reference for the hybrid-attention experiments.
2. The real attention cost is modest up to 8K (~1 ms/token, < 10%) and becomes significant only near
   32K (~6.5 ms, ~35% of an artifact-free step). That bounds what any KV placement scheme can save on
   a 4B dense model at these lengths.
3. GPU attention already runs near memory bandwidth (~765 GB/s). For Gate B, a CPU path would have to
   beat this on the KV it serves. Back-of-envelope (**not yet measured**): desktop DRAM at ~50–80 GB/s
   would need roughly 60–100 ms to read 4.5 GiB of KV, and copying it over PCIe 4.0 (~25 GB/s)
   ~190 ms, versus 6.5 ms on the GPU. Dense full offload is therefore a capacity tool, not a speed
   tool, and a speed benefit would depend on serving only a small, hot fraction of KV (Gate C).
4. Retrieval quality on real code drops sharply at 32K even though it fits in the model's context,
   so long-context accuracy is itself a target, separate from speed.

## Caveats

- One model, one GPU, HF eager + SDPA, batch 1, greedy. Not representative of vLLM / FlashInfer.
- Profile from one repo; timings are content-independent, so this should transfer to the others.
- Retrieval: one needle per repo at 50% depth; name-in-answer is a loose proxy for RepoQA's
  similarity-based scoring.
- The CPU/PCIe figures in conclusion 3 are rough estimates; host memory bandwidth has not been
  measured on this machine.

## Next

1. **Artifact-free GPU baseline:** a preallocated KV cache written in place, plus the FlashAttention
   decode kernel with per-sequence length (e.g. `flash_attn_with_kvcache`, FlashInfer, or vLLM). Then
   re-run this sweep.
2. **Measure host memory bandwidth and a CPU attention kernel** on matched KV (Gate B), replacing the
   estimate in conclusion 3.
3. **E1 attention-locality traces** at 8K–32K on these prompts: how concentrated attention mass is
   per KV page, and how much it shifts between decode steps (Gate C).
4. **Retrieval at scale:** all 10 needles × several depths per repo, scored with RepoQA's own metric.
