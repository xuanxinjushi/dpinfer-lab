# Research Review — Critical Flaws & Risks (2026-11-17)

Date: 2026-11-17 · Scope: `README.md`, `docs/`, `results/`, `src/dpinfer/`, `bench/`

Overall, the engineering foundation and experimental discipline are strong: catching and fixing HF `DynamicCache`'s `torch.cat` artifact (`InPlaceCache`), passing end-to-end Gate A token-identical verification on a real model, separating `measured` vs `estimate`, and keeping RTX 4090 and L40 numbers strictly separate.

From a systems-research / reviewer perspective, however, there are **4 critical flaws / risks** that need to be addressed before building further on the current assumptions.

---

## 1. The "`hybrid` is 2× faster than `offload`" claim does not currently hold

1. **The measured `offload` baseline was an `fp32` strawman.**
   In `results/hybrid_budget/2026-11-17_check/hybrid_budget.json` (and `docs/findings.md`), `offload` transferred **4.83 GB/step** at 32K / 50% budget (`fp32` volume), resulting in **231.5 ms** vs `hybrid`'s **115.9 ms**.
   `src/dpinfer/hybrid.py` has since been updated so `offload` uses `self.dtype` (`bf16`), but the benchmark has not been re-run. At the measured **24.0 GB/s** PCIe H2D bandwidth (`logs/cpu_attention_2026-11-17.log`), transferring `bf16` cold KV (2.415 GB at 32K/50%) takes **~100.6 ms** even without stream overlap — already **faster** than `hybrid` running `fp32` CPU attention (**115.9 ms**).
   - *Root cause:* On the 4090 desktop, `hybrid` using `fp32` consumes ~50 GB/s DRAM bandwidth, which is only **~25 GB/s** in `bf16`-equivalent throughput — essentially tied with PCIe 4.0 (**24.0 GB/s**).

2. **Lowering CPU KV precision to `bf16` collapses in PyTorch on x86.**
   `logs/cpu_attention_2026-11-17.log` shows `cpu bfloat16` achieving only **0.6–2.0 GB/s** effective bandwidth on this AVX2 host (**4,080 ms** at 16K) due to missing native BF16 SIMD instructions.
   Furthermore, the L40 server's **Xeon Platinum 8362 (Ice Lake)** (`docs/l40-outlook.md`) supports AVX-512F/VNNI, **not `AVX512_BF16` or `AMX`** (which arrived in Cooper Lake / Sapphire Rapids). Without a custom C++/AVX-512 kernel for `fp16` or `int8` quantized KV, PyTorch `bf16` on the L40 will likely hit the same cliff; staying on `fp32` halves effective host bandwidth and doubles host RAM usage.

3. **`offload` lacks basic layer-wise async prefetching.**
   In `src/dpinfer/hybrid.py`, `offload` runs `to(device)` and `_flash_lse` on the default CUDA stream and allocates `kc`/`vc` every layer per step. A realistic offload baseline uses a dedicated copy stream + double-buffered GPU staging to prefetch layer $l+1$ while computing layer $l$'s MLP and layer $l+1$'s QKV projections.

---

## 2. Potential contradiction between Exact `hybrid` and Sparse `hybrid` (Gate C)

Per `docs/kv-budget-experiment.md`, exact `hybrid` is 6–10× slower than `full` on the 4090 desktop and at best approaches `full` on the L40 server. Achieving an actual speedup over `full` (or practical speed on a desktop) requires **Gate C (sparse page selection)**.

However, this raises a fundamental systems question:

> If Gate C holds — meaning each decode step only needs 5%–10% of cold KV pages (e.g., 0.45 GiB `bf16` at 32K) and adjacent steps exhibit **temporal page reuse** — why compute attention on the CPU instead of fetching those hot pages over PCIe into a GPU LRU page cache and running FlashAttention (like Quest / ArkVale / ShadowKV)?

- Moving 0.45 GiB over PCIe 4.0 takes ~18 ms (even less with async prefetch and cross-step GPU page cache hits).
- The project needs to clearly define the regime where **CPU-side compute** is genuinely superior to **GPU-side sparse page fetching** (e.g., moderate sparsity like 30%–50% where PCIe saturates, large-batch concurrency where PCIe is shared across requests while CPU cores/DRAM channels are idle, or CPU-side `int8`/`int4` quantized KV).

---

## 3. Experimental & measurement hygiene issues

1. **Unexplained anomaly in 32K `drop` latency (`--runs 2` noise):**
   In `logs/hybrid_check_2026-11-17.log` and `docs/findings.md`, 32K `drop` at 10% budget (**21.82 ms**) is slower than 50% budget (**16.08 ms**) and even slower than `full` (**18.76 ms**). Attending to fewer tokens should not be slower; this is likely measurement noise from `--runs 2` combined with background desktop/Chrome GPU activity.
2. **Prefill activation peak masks decode VRAM savings:**
   `bench/hybrid_budget.py` calls `torch.cuda.reset_peak_memory_stats()` before warmup + prefill. At 32K / 10% budget, KV is 0.46 GiB, yet `peak_vram_gib` is **10.4 GiB** due to full-prompt prefill activations/logits (~2.4 GiB). On a genuinely memory-constrained GPU, unchunked prefill would OOM before decode begins.
3. **36 synchronous GPU→CPU round-trips per token:**
   `src/dpinfer/hybrid.py` performs `qp[0].float().cpu()` (draining the CUDA launch queue) and a non-pinned `o_c.to(device)` per layer. At 116–193 ms/token this ~1–2 ms overhead is hidden, but if CPU attention is accelerated to ~5–15 ms (on L40 or via sparsity), 36 sync points per token will become a primary bottleneck.

---

## 4. 32K retrieval collapse (1/6) can contaminate E1 locality analysis

In `results/2026-11-17_context-sweep_qwen3-4b.md`, Qwen3-4B needle retrieval drops from 5–6/6 (1K–16K) to **1/6 at 32K** (using the coarse `needle_name in answer` check rather than RepoQA's similarity metric).

- When running **E1 (Gate C attention locality traces)**, failed 32K prompts where the model hallucinates or retrieves the wrong function may exhibit diffuse or misdirected attention.
- **Actionable requirement:** Always stratify E1 locality and page-reuse metrics by **correct vs. incorrect** retrieval, rather than aggregating across all 32K prompts.

---

## Recommended next steps

1. **Re-run `bench/hybrid_budget.py` fairly:** Use the updated `bf16` `offload` path with `--runs 5` and idle desktop GPU; also check whether `cpu float16` avoids the `bfloat16` slowdown on AVX2.
2. **Run E1 (Gate C) before committing to a CPU-kernel investment:** Measure real attention concentration and step-to-step page reuse on *correctly answered* 8K–32K RepoQA prompts first, then decide between CPU-side quantized attention vs. GPU-side sparse page caching.
