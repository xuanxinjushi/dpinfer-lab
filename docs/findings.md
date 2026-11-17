# Findings log

Newest first. Each entry says what was measured, where the data is, and what it changes.

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
