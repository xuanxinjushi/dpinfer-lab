# Findings log

Newest first. Each entry says what was measured, where the data is, and what it changes.

## 2026-11-17 — E1: Gate C fails on all pre-registered criteria (Qwen3-4B, RepoQA code, measured)

`bench/e1_locality.py`, `bench/analyze_e1.py`; criteria fixed beforehand in
`docs/gate-c-criteria.md` (not changed). 6 languages × 2 repos × 3 needles, depths 0.25/0.5/0.75,
8K/16K/32K (105 prompts; `expressjs/express` too small for 32K), greedy decode to EOS (≤ 256
steps), every decode step, all 36 layers × 8 KV heads. Always-resident set R = first 64 + last
1024 tokens; page size 128 unless noted. Data: `results/e1/2026-11-17/` (per-prompt rows,
`summary.json`, `gate_c.png`); raw per-step traces in `traces/e1/2026-11-17/` (not in git).

Correctly answered prompts (the stratum the criteria use):

| Context | n | C1: share of (t,l,h) with top-10% cold pages ≥ 90% cold mass (need ≥ 0.90) | mean top-10% / top-20% share of cold mass | C2: adjacent-step top-10% overlap (need ≥ 0.80) | cold mass $M_C$ | C3: oracle tokens identical (need ≥ 0.90) | C3: same retrieval (need 1.00) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 25 | 0.21 | 0.65 / 0.79 | 0.73 | 0.27 | 0.84 | 0.96 |
| 16K | 23 | 0.28 | 0.71 / 0.83 | 0.73 | 0.32 | 0.83 | 1.00 |
| 32K | 16 | 0.32 | 0.76 / 0.87 | 0.73 | 0.40 | 0.50 | 0.94 |

**Gate C: fail (C1, C2 and C3 all fail at every length).** Incorrectly answered prompts look the
same on C1/C2 (0.18–0.33, 0.72–0.74) and worse on C3.

- **Attention over cold code is diffuse.** The top 10% of cold pages hold only 65–76% of the
  cold mass; even the top 20% hold 79–87%. Smaller pages help a little (P=64: 0.26–0.39 share),
  larger pages hurt (P=256: 0.14–0.27).
- **The cold region matters.** Outside the sink + recent-1024 window sits 27% (8K) to 40% (32K)
  of all attention mass, growing with context.
- **The hot set churns:** ~27% of the top-10% pages change between adjacent decode steps, at every
  length. A GPU page cache (Quest/ShadowKV-style) would miss on a large fraction of pages per step.
- **Oracle sparsity (R + exact top-10% pages) mostly preserves the retrieval outcome (94–100%)
  but not the text**: at 32K only half of correctly answered prompts give identical tokens.
- **Strong layer dependence** (per-layer C1, correct prompts): layers 0–19 are diffuse
  (C1 ≈ 0–0.3); layers 23–34 are much more concentrated (0.5–0.93, rising with context); layer 30
  is the only one passing C1 (0.93 at 32K). Per-layer C2 stays 0.6–0.88 with no layer
  consistently ≥ 0.8 together with a high C1. Layer 35 has almost no cold mass (0.05).

Consequences, per the plan fixed in `gate-c-criteria.md`:

1. **No sparsity story for this model and task.** Hybrid attention stays **exact**: a capacity
   tool whose speed depends on host bandwidth (Gate B), not on skipping KV.
2. CPU-side compute over the full cold region remains the relevant path; a GPU hot-page cache
   is not supported by the data (C2), which answers review point 2 for this setting.
3. Open, not claimed: a **layer-heterogeneous** policy (exact/dense for layers 0–22, sparse
   only for late layers) — a new hypothesis that would need its own pre-registered test.
4. Retrieval at 32K is better than the first sweep suggested once depths vary
   (16/33 correct vs 1/6 at depth 0.5).

## 2026-11-17 — Fair budget curve: bf16 CPU kernel makes hybrid beat prefetching offload (RTX 4090 desktop, measured)

`psf/black` needle 0, 64 decode steps, 5 runs. Offload now ships bf16 with layer-wise async
prefetch (`offload`); `offload_sync` is the same without prefetch. Hybrid uses the new AVX2 bf16
kernel (`src/dpinfer/csrc/cpu_attn.cpp`, 24 threads) on bf16 host KV; the earlier fp32 PyTorch
CPU path is shown for reference. Data: `results/hybrid_budget/2026-11-17_fair/`,
`results/hybrid_budget/2026-11-17_fair_bf16kernel/`. Decode ms/token, median:

| Context | Budget | full | **hybrid, bf16 kernel** | offload (prefetch) | offload_sync | hybrid, fp32 PyTorch | drop (lossy) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 50% | 14.7 | 30.1 | **26.3** | 41.4 | 42.3 | 14.1 |
| 8K | 25% | | **35.7** | 38.7 | 53.7 | 54.1 | 13.8 |
| 8K | 10% | | **39.6** | 46.0 | 61.2 | 63.6 | 13.5 |
| 16K | 50% | 16.1 | **42.2** | 51.0 | 68.2 | 67.7 | 14.7 |
| 16K | 25% | | **57.4** | 75.9 | 93.1 | 95.1 | 14.1 |
| 16K | 10% | | **65.3** | 91.1 | 107.9 | 108.4 | 13.6 |
| 32K | 50% | 18.8 | **70.8** | 101.5 | 122.1 | 117.3 | 16.1 |
| 32K | 25% | | **99.0** | 152.4 | 173.0 | 168.2 | 14.7 |
| 32K | 10% | | **114.5** | 182.7 | 203.2 | 196.2 | 14.0 |

- All exact modes (hybrid, offload, offload_sync) are token-identical to `full` in every
  configuration. `drop` diverges within 3–8 tokens and loses the needle at 8K ≤ 25% and at all
  16K budgets, where `full` finds it.
- **Gate B passes on this desktop with the bf16 kernel:** hybrid beats prefetching offload in
  8 of 9 configurations, by up to **1.6×** (32K/10%: 114.5 vs 182.7 ms). The exception is
  8K/50%, where cold KV is small and the fixed hybrid overhead dominates.
- The kernel reads bf16 KV at ~50 GB/s (≈70% of the 71 GB/s host read bandwidth); offload is
  transfer-bound at ~24 GB/s. That ratio is the whole advantage.
- Hybrid ≈ `full` + CPU time + 3–9 ms: the CPU part is **not overlapped** with the rest of the
  GPU step, because each layer waits for its CPU result. Overlap would need cross-layer
  pipelining, which the data dependency between layers does not allow for exact attention.
- Prefetch is worth 15–20 ms per token for offload; without it offload ≈ the old fp32 hybrid.
- Decode-phase peak VRAM (prefill transients excluded): 32K `full` 12.01 GiB vs 7.96 GiB at a 10%
  budget (weights 7.49 + KV 0.46).
- The earlier `drop` anomaly (32K/10% slower than `full`) does not reproduce with 5 runs
  (13.97 ms): it was noise.
- Still far from `full` (3–6× slower at 32K): exact hybrid is a capacity tool on this machine.
  Beating it needs fewer cold bytes per step (Gate C) or much higher host bandwidth (L40 server).

Kernel to-dos: reuse a preallocated scratch buffer instead of 5 tensor allocations per call
(≤ ~3% headroom: in-model 49.6 GB/s vs 51 GB/s standalone); AVX-512 path for the L40 server.

## 2026-11-17 — Gate B microbenchmarks (RTX 4090 desktop, i9-13900KF, measured)

`bench/cpu_attention.py`; data `results/gate_b/2026-11-17/cpu_attention.json`,
log `logs/cpu_attention_2026-11-17.log`. Qwen3-4B shapes, one decode query, all 36 layers.

| | Bandwidth |
|---|---:|
| Host DRAM read (torch `sum`, 8 threads) | 70.9 GB/s (4 thr 56, 16 thr 67, 24 thr 68) |
| PCIe H2D / D2H, pinned | 24.0 / 26.3 GB/s |
| GPU flash decode attention at 32K | 857 GB/s (5.6 ms / 36 layers) |
| CPU attention, fp32 KV, 32K, 24 thr | 57 GB/s (169 ms / 36 layers) |
| CPU attention, bf16 / fp16 KV (PyTorch) | 0.5–2 GB/s (8.6–9.4 s at 32K) |

- fp32 CPU attention reaches ~80–85% of host read bandwidth, but per *bf16-equivalent* KV byte
  that is only ~29 GB/s — about the same as PCIe (24 GB/s). So, with PyTorch kernels, computing on
  the CPU and shipping KV over PCIe are roughly tied on this desktop.
- PyTorch's bf16/fp16 CPU matmul for these shapes is pathological (25–80× slower than fp32, and
  getting worse with length). This is an implementation gap, not a hardware limit: bf16→fp32 is a
  shift and fp16→fp32 has F16C on AVX2. A bandwidth-bound bf16 kernel would read ~60–70 GB/s of
  bf16 KV, ~2.5–3× PCIe.
- Dense CPU attention over a full 32K cache is 30× slower than the GPU (169 vs 5.6 ms); the CPU
  path only makes sense for the part of KV that does not fit on the GPU.
- Numerics vs GPU fp32 reference: fp32 max abs error ~1e-7; bf16 ~1e-3.

Gate B status on this machine: **open**; it depends on a bandwidth-bound low-precision CPU kernel.

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
- ~~hybrid ≈ 2× faster than offload~~ **Retracted (review, `docs/review-gemini.md`):** offload in
  this run shipped fp32 KV, twice the bytes a real system moves. With bf16 transfer at the measured
  24 GB/s H2D, offload at 32K/50% would be ~100 ms of copy + ~19 ms compute ≈ hybrid's 116 ms, i.e.
  roughly a tie. The CPU path reads fp32 cold KV at ~50 GB/s, which is only ~25 GB/s of
  bf16-equivalent KV. PyTorch CPU attention in bf16/fp16 is far slower (0.6–2 GB/s,
  `logs/cpu_attention_2026-11-17.log`), so a bandwidth-bound low-precision CPU kernel is needed
  before hybrid can beat offload here. Re-run pending with bf16 offload + layer prefetch.
- **hybrid is 3–10× slower than full** on this desktop. Almost all of the gap is CPU attention
  time (e.g. 32K/10%: 173 ms CPU of 193 ms); the per-layer GPU↔CPU sync costs only ~1–2 ms/token.
- GPU memory saved: KV 4.51 → 0.46 GiB at 32K/10%; peak VRAM 14.4 → 10.4 GiB (peak still
  includes one layer's transient full prefill KV).
- Caveat (fairness): host KV is fp32 for both exact modes, so both move 2× the bytes of bf16.
  Next run: bf16 host KV for `offload` (halves PCIe bytes), and bf16/fp16 for `hybrid` if the
  CPU kernel stays bandwidth-bound (`bench/cpu_attention.py`).

Consequence (revised): Gate B is **not yet established** on this machine. Raw host DRAM read
(~70 GB/s) is ~3× PCIe (24 GB/s), so a bandwidth-bound bf16 CPU kernel could win, but the current
PyTorch path does not reach it. Matching `full` needs either much more host bandwidth (L40 server, see
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
