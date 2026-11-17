# What this means for the L40 server

Machine (from README §2; verify before relying on it): 1× NVIDIA L40 (~44.4 GiB usable,
~864 GB/s GDDR6, Ada sm_89), 2× Xeon Platinum 8362 (Ice Lake, 64 cores, AVX-512, 8-channel
DDR4-3200 per socket), ~1 TiB RAM, pinned H2D ≈ 22.6 GiB/s, D2H ≈ 24.5 GiB/s.

## What transfers directly from the 4090 results

- **The `DynamicCache` artifact and its fix.** Same GPU architecture (sm_89), same SDPA kernels, so
  the in-place cache should give a similar ~1.5× decode speed-up at 32K in HF eager. Use the
  in-place cache as the baseline there too.
- **The bandwidth model of decode**: per token, read the weights once plus every KV byte once.

## Where the idea is *not* useful on the L40

Single request, 4B model, ≤ 32K context: KV is ≤ 4.5 GiB and fits easily; GPU flash attention
already runs near memory bandwidth. Moving KV to the host cannot make this faster.
Estimated 32K decode on the L40 (**estimate**): ~21–23 ms/token, slightly slower than the 4090
because of lower memory bandwidth.

## Where it can be useful

1. **Many concurrent long requests.** Weights are read once per step but KV is read per request.
   8 × 32K requests ≈ 38 GB of KV + 7.5 GB weights ≈ the whole card. Batch size becomes capacity
   bound, and each step is dominated by KV reads (**estimate**: ~70 ms KV vs ~15 ms weights). Keeping
   only hot KV on the GPU raises the number of requests that fit — a throughput win, which is what
   a server GPU is for.
2. **Larger models.** 14B BF16 (~28 GB) leaves ~16 GB for KV; 32B BF16 does not fit at all. KV
   placement then decides the achievable context and batch.
3. **High host bandwidth makes Gate B plausible.** The desktop's dual-channel DRAM (~50–80 GB/s)
   makes CPU attention slow. Two 8-channel DDR4-3200 sockets are ~400 GB/s theoretical,
   perhaps ~150 GB/s per socket in practice (**estimate**). CPU attention over 3.4 GB of cold KV
   would then take ~12–25 ms versus ~140 ms to ship it over PCIe — so `hybrid` should clearly beat
   `offload` there. Must be measured: NUMA placement (KV and threads on the same socket) will
   matter a lot.

## CPU kernel notes for the Xeon 8362 (Ice Lake)

- `src/dpinfer/cpu_kernel.py` builds with `-mavx2 -mfma -mf16c`. The intrinsics are AVX2, so
  `-mavx512f` / `-march=native` alone will not widen them; an AVX-512 variant of
  `csrc/cpu_attn.cpp` has to be written (bf16→fp32 is still a shift; no AVX512_BF16 on Ice Lake).
- The kernel is memory-bound. On a two-socket machine NUMA placement likely matters more than
  vector width: keep each layer's cold KV in one socket's memory and bind the threads that read it
  to that socket (`numactl`, or first-touch from pinned threads). Measure per-socket and
  cross-socket bandwidth first.

## What to run on the L40 (in order)

1. Copy the repo (git), `models/Qwen3-4B`, and `data/repoqa_release/` to the server.
2. `bench/cpu_attention.py` with `--threads 8 16 32 64`, once per NUMA node
   (`numactl --cpunodebind=N --membind=N`) and once across both — this answers Gate B for that machine.
3. `bench/hf_context_sweep.py --cache inplace` — L40 baseline.
4. `bench/hybrid_budget.py` budget curve (50/25/10%, 8K–32K), threads pinned to one socket.
5. Later: real batch > 1 so that the budget is genuinely binding rather than simulated.

Record L40 results under `results/` with the machine name in the file name; never mix them into
the 4090 tables.
