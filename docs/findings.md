# Findings log

Newest first. Each entry says what was measured, where the data is, and what it changes.

## 2026-11-17 — Gate A passed end-to-end; first KV-budget curve (RTX 4090 desktop, measured)

`bench/hybrid_budget.py`, `psf/black` needle 0, 64 greedy decode steps, 2 runs, 8 CPU threads,
host KV stored as fp32. Data: `results/hybrid_budget/2026-11-17_check/`.

| Context | GPU budget | full | hybrid (CPU attn) | offload (PCIe) | drop (lossy) |
|---:|---:|---:|---:|---:|---:|
| 8K | 50% | 14.7 | 41.2 | 67.2 | 14.1 |
| 8K | 25% | | 52.3 | 93.1 | 13.8 |
| 8K | 10% | | 59.8 | 108.2 | 13.5 |
| 32K | 50% | 18.8 | 115.9 | 231.5 | 16.1 |
| 32K | 25% | | 163.9 | 337.5 | 16.6 |
| 32K | 10% | | 192.6 | 400.8 | 21.8 |

Decode ms/token (median).

- **Exactness:** `hybrid` and `offload` generate tokens identical to `full` in all 12
  configurations (Gate A on the real model). `drop` diverges from `full` within 3–8 tokens, and at
  8K with a 25% or 10% budget it loses the needle that `full`/`hybrid` find.
- **hybrid ≈ 2× faster than offload** at every budget. Both are bandwidth-bound: the CPU path
  reads cold KV at **~50 GB/s** (the desktop's dual-channel DRAM limit, measured from CPU time),
  offload moves it at ~22 GB/s over PCIe.
- **hybrid is 3–10× slower than full** on this desktop. Almost all of the gap is CPU attention
  time (e.g. 32K/10%: 173 ms CPU of 193 ms); the per-layer GPU↔CPU sync costs only ~1–2 ms/token.
- GPU memory saved: KV 4.51 → 0.46 GiB at 32K/10%; peak VRAM 14.4 → 10.4 GiB (peak still
  includes one layer's transient full prefill KV).
- Caveat (fairness): host KV is fp32 for both exact modes, so both move 2× the bytes of bf16.
  Next run: bf16 host KV for `offload` (halves PCIe bytes), and bf16/fp16 for `hybrid` if the
  CPU kernel stays bandwidth-bound (`bench/cpu_attention.py`).

Consequence: the CPU/GPU crossover (Gate B) holds on this machine: computing on the host beats
shipping KV. Matching `full` needs either much more host bandwidth (L40 server, see
`l40-outlook.md`) or reading far less cold KV per step (sparse selection, Gate C).

## 2026-11-17 — Copy-free KV cache removes the HF long-context artifact (RTX 4090, measured)

`DynamicCache` appends with `torch.cat`, re-copying the whole KV cache every decode step. We added
`src/dpinfer/kvcache.py` (`InPlaceCache`): KV preallocated once, new tokens written in place, views
of the filled prefix returned to attention. SDPA still selects the FlashAttention split-KV kernel.
Generated tokens are identical to `DynamicCache` and to `generate()` on all 30 configurations.

Same sweep as before (6 RepoQA repos, needle 0 at 50% depth, 256 decode steps, 5 runs):

| Context | Decode ms/tok, DynamicCache | Decode ms/tok, in-place | Speed-up | Peak VRAM (in-place) |
|---:|---:|---:|---:|---:|
| 1K | 13.62 | 13.53 | 1.01× | 7.76 GiB |
| 4K | 14.77 | 14.11 | 1.05× | 8.40 GiB |
| 8K | 16.25 | 14.73 | 1.10× | 9.27 GiB |
| 16K | 20.03 | 16.09 | 1.24× | 11.00 GiB |
| 32K | 28.95 | 18.77 | 1.54× | 14.46 GiB |

- Linear fit (in-place): decode ≈ **13.4 ms + 0.164 ms per 1K context tokens**, max residual
  0.05 ms. The context term implies ~**900 GB/s** per KV byte (near the 4090's ~1 TB/s peak), versus
  ~303 GB/s with `DynamicCache`.
- At 32K the context-dependent part (mostly attention) is ~29% of a decode step; at 8K ~9%.
- Prefill and retrieval accuracy are unchanged (same kernels for prefill; same tokens).

Data: `results/context_sweep/2026-11-17_inplace/`. Consequence: **all future baselines use the
in-place cache**, never `DynamicCache`.

## 2026-11-17 — Context sweep and kernel profile with HF `DynamicCache` (RTX 4090, measured)

Write-up: `results/2026-11-17_context-sweep_qwen3-4b.md`.

- Decode 13.6 → 29.0 ms/token from 1K to 32K; ~34% of the 32K step was the `torch.cat` copy.
- `StaticCache` in eager mode is worse (70 ms/token at 32K): the mask forces SDPA's
  memory-efficient kernel.
- GPU flash attention reads KV at ~765 GB/s in the profile.
- Needle retrieval on real code: 6/6 at 1K–4K, 5/6 at 8K–16K, 1/6 at 32K (coarse metric, n=6).

## Open estimates (to be replaced by measurements)

- Desktop host DRAM ~50–80 GB/s → dense CPU attention over 32K KV ≈ 60–100 ms/token
  (`bench/cpu_attention.py` will measure this).
- PCIe 4.0 H2D ~25 GB/s → copying 4.5 GiB of KV per step ≈ 190 ms.
