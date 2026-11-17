# Which model to run where

Status 2026-11-17. The L40 server already has a Qwen 27B FP8 checkpoint; this note records which
model to use for which experiment and what must be checked before switching.

## Recommendation

| Order | Model | Machine | Why |
|---|---|---|---|
| 1 | **Qwen3-4B BF16** (current control) | L40 | Same model as the 4090 results, so differences are attributable to the hardware (host bandwidth, NUMA, PCIe). All code works unchanged. Follow [l40-runbook.md](l40-runbook.md). |
| 2a | **Qwen 27B FP8** (on the server) | L40 | Real memory pressure: ~27–28 GB of weights leave ~16 GB for KV and activations, so the GPU KV budget is genuinely binding instead of simulated. Needs the checks below first. |
| 2b | Qwen3-14B BF16 (alternative) | L40 | Same architecture as Qwen3-4B (dense GQA), so `HybridCache`, the CPU kernel and the tracer work without changes. ~28 GB weights; KV ≈ 160 KiB/token (**estimate** from the published config: 40 layers × 8 KV heads × 128 × K,V × 2 B), so ~4 × 32K requests no longer fit. Needs a ~28 GB download. |
| — | Smaller models | — | Not useful: less KV and memory pressure moves away from the regime this project studies. |

## Before using the 27B FP8 model: two checks

Run on the server and add the output to this file (or to `results/l40/<date>/model_27b.txt`):

```bash
python scratch/inspect_config.py /path/to/the/27B/model
```

**1. Is it a hybrid linear-attention model?** README §3, phase 6 refers to a Qwen 27B that mixes
DeltaNet-style linear attention with full attention. If `layer_types` shows mostly
`linear_attention` with a few `full_attention` layers:

- only the full-attention layers have KV that grows with context; linear-attention layers keep a
  fixed-size state. KV per token is then much smaller than a dense 27B model would suggest, and
  the KV-vs-weights balance of decode changes;
- `HybridCache` currently assumes every layer is ordinary attention. It must be changed to take
  over only the full-attention layers (and leave the linear-attention cache layers to HF);
- the E1 tracer and Gate C criteria apply only to those full-attention layers;
- the CPU kernel assumes GQA with head_dim ≤ 256 and ≤ 8 query heads per KV head; check
  `num_attention_heads / num_key_value_heads` and `head_dim`.

If it is a dense GQA model, the current code should work with small checks (layer count, head
counts, head_dim all come from the config).

**2. How does FP8 run in HF transformers?** The L40 (sm_89) supports FP8 in hardware, but loading
FP8 weights in HF depends on the checkpoint's `quantization_config` (e.g. fine-grained / block FP8,
compressed-tensors) and its kernels. Verify on the server that

- the model loads with the pinned transformers version,
- a smoke generation gives sensible text,
- decode speed is plausible for the weight bytes (≈ 27 GB / ~600 GB/s ≈ 45 ms/token **estimate**
  at short context; much slower means a fallback dequantisation path, which would distort every
  KV-related comparison).

KV stays BF16 regardless of FP8 weights unless the cache is explicitly quantised; the hybrid code
and CPU kernel are written for BF16 KV.

## What changes in the experiments with a large model

- The budget is real: run with `--fracs` chosen from what actually fits after weights, or better,
  with genuine concurrency (batch > 1), which still needs batched-cache support in `HybridCache`.
- Weight reads dominate each decode step more (FP8 27 GB vs BF16 7.5 GB), so attention/KV is a
  smaller share of the step at the same context. KV placement then matters mainly for capacity
  (how many requests / how long a context fit), less for per-token latency.
- Results go under paths tagged with both machine and model, e.g.
  `results/hybrid_budget/<date>_l40_27bfp8/`.
