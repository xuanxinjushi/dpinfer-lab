# Hybrid CPU/GPU attention inside HF transformers

Code: `src/dpinfer/hybrid.py` (cache + attention), `src/dpinfer/kvcache.py` (copy-free baseline
cache), `bench/hybrid_budget.py` (driver). Works with transformers 5.23 without patching the library.

## Pieces

**`HybridCache` / `HybridLayer`** (subclass of `DynamicLayer`, one per decoder layer)

- *Prefill* (first `update`, `n` prompt tokens): receives the full post-RoPE K/V of this layer on the
  GPU. Keeps `sink` first tokens + the last `budget − sink` tokens in a preallocated GPU buffer
  (with `reserve` slots for generated tokens). Copies the middle ("cold") span to host memory
  (fp32 by default; pinned only in `offload` mode). Returns the full K/V so prefill attention is
  ordinary SDPA; the full K/V of one layer is the only transient over-budget allocation.
- *Decode*: writes the new token's K/V into the GPU buffer in place (no `torch.cat`) and returns
  views of the hot part.
- `get_seq_length` / `get_mask_sizes` report the logical length (hot + cold) so positions and
  masks stay correct.

**`hybrid_attention`**, registered with `AttentionInterface.register("hybrid", …)` and
`AttentionMaskInterface.register("hybrid", sdpa_mask)`; enabled with
`model.set_attn_implementation("hybrid")`.

- Prefill, `full` mode, or no cold KV → falls through to HF's `sdpa_attention_forward`.
- Decode with cold KV → per layer:
  1. Pack GQA: query `[1, 32, 1, 128]` → `[1, 8, 4, 128]` (Q head `j·4+i` uses KV head `j`, the
     same mapping as `repeat_kv`). The 4 query heads of a group become 4 "query rows" of a
     non-causal attention, so no K/V repetition is needed.
  2. GPU part: `torch.ops.aten._scaled_dot_product_flash_attention` over the hot K/V. Unlike the
     public SDPA it also returns the log-sum-exp per row, which fusion needs.
  3. Cold part, by mode:
     - `hybrid`: copy the 8×4×128 query to the CPU (a sync point), launch the GPU part
       asynchronously, compute `cpu_partial` (scores → lse → softmax · V) on the host KV, send the
       `[8, 4, 128]` output and `[8, 4]` lse back.
     - `offload`: copy the cold K/V host→GPU (`non_blocking`, pinned) and run the same flash op on it.
     - `drop`: skip it.
  4. Fuse: $\mathrm{lse} = \mathrm{logaddexp}(\mathrm{lse}_G, \mathrm{lse}_C)$,
     $o = e^{\mathrm{lse}_G-\mathrm{lse}}\,o_G + e^{\mathrm{lse}_C-\mathrm{lse}}\,o_C$.

## Why it is exact

Softmax attention over a disjoint union of key sets $G \cup C$ decomposes exactly:

$$
\mathrm{softmax}(s)\,V = \sum_{P\in\{G,C\}} \frac{e^{\mathrm{lse}_P}}{e^{\mathrm{lse}_G}+e^{\mathrm{lse}_C}}\; \mathrm{softmax}(s_P)\,V_P .
$$

RoPE is applied before K enters the cache, so moving keys between devices or reordering them does
not change any score. Decode has one query and no causal mask inside the prompt, so no masking
is needed on either side. Differences against `full` come only from floating-point rounding
(fp32 accumulation on CPU, bf16 output); the end-to-end check is that greedy tokens are identical.

## Known limitations (current version)

- One synchronous GPU→CPU copy per layer per token (36 per token). This drains the CUDA launch
  queue and is likely the dominant overhead at small cold sizes.
- CPU kernel is plain `torch.matmul` on fp32/bf16; no AVX-512/AMX-specific code, no NUMA pinning.
- Batch size 1 only; hot set is static (sink + recent), no page-level selection.
- Prefill is not chunked: one layer's full K/V is transiently on the GPU.
- `offload` copies the entire cold span every step (worst-case baseline; a real system would
  prefetch layer *l+1* while computing layer *l*, which we should add for fairness).
