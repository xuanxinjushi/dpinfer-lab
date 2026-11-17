# Experiment: decode under an artificial GPU KV budget

## Why "pretend the GPU is too small"

On a 24 GB 4090 or a 48 GB L40, a single 32K request for Qwen3-4B needs only 4.5 GiB of KV, so
nothing forces KV off the GPU. Real memory pressure appears with **many concurrent long requests**,
**bigger models**, or **longer contexts**. Rather than building those workloads first, we cap the
GPU-resident KV at a fixed budget and treat everything beyond it as host-resident. This is the
standard methodology in KV-offload / sparse-attention papers.

The cap must be honest:

- Only `budget` prompt tokens per layer (plus generated tokens) are kept in GPU buffers; the rest
  is moved to host memory right after that layer's prefill. Reported per run: GPU KV bytes, host
  KV bytes, peak VRAM.
- Later: enforce a hard ceiling with `torch.cuda.set_per_process_memory_fraction` so the budget is
  provably not exceeded.
- Each budget level is tied to a real scenario it stands in for (table below), so the "pretend"
  maps to an actual deployment.

| Budget (of a 32K prompt) | Stands in for (estimate) |
|---|---|
| 50% | 4B model, ~16 concurrent 32K requests on an L40 |
| 25% | 14B BF16 model (~28 GB weights) with several 32K requests on an L40 |
| 10% | 128K-context requests, or a model that barely fits |

## Methods compared (same budget, same prompts, same precision)

| Mode | GPU KV | Remaining KV | Exact? | Role |
|---|---|---|---|---|
| `full` | all | — | yes | upper bound / quality reference (not a competitor) |
| `offload` | budget | copied host→GPU every step, GPU attention, LSE fusion | yes | standard offload baseline |
| `hybrid` (ours) | budget (hot) | CPU computes attention on host KV in parallel; only $(o, \mathrm{lse})$ per head returns | yes | the idea under test |
| `drop` | budget | discarded (sink + recent window, StreamingLLM-style) | no | lossy baseline: what you lose by not keeping KV |
| sparse hybrid (later) | budget | CPU/GPU attend only to selected cold pages | no | needs Gate C locality evidence |

Hot set for now: the first `sink` tokens (attention sinks) + the most recent prompt tokens +
all generated tokens. For the exact modes the choice of hot set affects speed, not results.

## Claims we are trying to support

1. **Exactness (Gate A, end-to-end):** `hybrid` and `offload` produce token-identical greedy output
   to `full` on real RepoQA prompts.
2. **Speed under budget (Gate B / D):** at equal budget, `hybrid` decode latency is well below
   `offload`, and the gap to `full` is quantified as a function of budget, context length and CPU
   threads.
3. **Quality under budget:** `drop` loses retrieval accuracy that `hybrid` keeps (it is exact).
4. **Not claimed:** that `hybrid` beats `full` when everything fits. When the budget is not
   binding, `full` is the answer.

## Metrics per (repo, context, budget, mode)

- Decode ms/token (median, p90), prefill ms.
- CPU time per step inside attention; H2D bytes per step (offload).
- GPU KV bytes, host KV bytes, peak VRAM.
- Tokens equal to `full`; first divergence step; RepoQA needle found.

Sweep axes: context {8K, 16K, 32K}, budget {50%, 25%, 10%}, CPU threads {4, 8, 16, 24},
host KV dtype {fp32, bf16}; later page size and batch size.

## Cost model and the main risk

Per layer, per decode step:

$$
T_{\mathrm{layer}} \approx T_{q\to\mathrm{CPU}} + \max\!\big(T_{\mathrm{GPU\,hot}},\; T_{\mathrm{CPU\,cold}}\big) + T_{\mathrm{result}\to\mathrm{GPU}} + T_{\mathrm{fuse}}
$$

$T_{\mathrm{CPU\,cold}} \approx$ cold KV bytes / host memory bandwidth. With 36 layers the CPU sits on
the critical path 36 times per token, and each $q \to$ CPU copy is a GPU→host synchronisation that
drains the kernel queue. That synchronisation and launch overhead is the biggest engineering risk
and must be measured, not modelled.

Rough expectation at 32K, 25% budget (cold ≈ 3.4 GB of bf16 KV), **estimates**:

| Machine | offload (PCIe ~24 GB/s) | hybrid (CPU) | full |
|---|---:|---:|---:|
| 4090 desktop, ~60 GB/s DRAM | ~140 ms | ~55 ms | ~19 ms (measured) |
| L40 server, ~150–300 GB/s DRAM | ~140 ms | ~12–25 ms | ~21–23 ms |

So exact hybrid should beat offload by a wide margin on both machines, and could approach `full`
only on a server with high host bandwidth. Getting **below** `full` would require reading less KV,
i.e. sparse selection backed by Gate C.

## Commands

```bash
# correctness + quick latency (Gate A)
python bench/hybrid_budget.py --repos psf/black --lengths 8192 32768 \
    --fracs 0.5 0.25 0.1 --runs 2 --decode-steps 64 --threads 8 \
    --out results/hybrid_budget/<date>_check

# host bandwidth, PCIe, CPU vs GPU attention microbenchmarks (Gate B)
python bench/cpu_attention.py --out results/gate_b/<date>/cpu_attention.json
```

## Next steps after the first budget curve

1. Thread / dtype sweep for the CPU path; pin threads to P-cores on the desktop, to one NUMA node
   on the L40 server.
2. Replace the per-layer synchronous `q` copy with an overlapped pipeline (CUDA stream callbacks or
   a persistent CPU worker polling pinned memory).
3. E1 locality traces (Gate C), then sparse hybrid with explicit accuracy measurement.
4. Real concurrency: batch > 1 so the budget is genuinely binding, not simulated.
