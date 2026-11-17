# Gate C′ — layer-split sparsity: pre-registered hypothesis and criteria (E2)

Status: **fixed 2026-11-17, after E1 and before any E2 data.** Changing anything below after seeing
E2 data must be recorded here with the date and reason.

**Outcome (2026-11-17): H′ not supported — arm A fails** (C3′b at 32K: 16/17 correct prompts keep
the retrieval outcome; one loses it). Arms pass only at 16K; 8K fails C3′a for both. Criteria
unchanged. Details: `docs/findings.md`, `results/e2/2026-11-17/`.

## Where the hypothesis comes from

Gate C failed ([gate-c-criteria.md](gate-c-criteria.md), [findings.md](findings.md)). E1's
per-layer breakdown (`results/e1/2026-11-17/gate_c.png`) suggests a weaker, layer-restricted claim:
layers 0–19 are diffuse, layers 23–34 are much more concentrated, and concentration grows with
context length.

That pattern was found by looking at E1 data, so **E1 cannot confirm it** (it would be fitting and
testing on the same prompts). E2 tests it on held-out repositories with every parameter fixed in
advance from E1.

Exploration numbers from E1 (correct stratum, page 128, layers 23–34 pooled) used to fix the
parameters — share of (t, l, h) where the top-k% cold pages hold ≥ 90% of cold mass:

| context | top 5% | top 10% | top 20% |
|---|---|---|---|
| 8K | 0.30 | 0.48 | 0.72 |
| 16K | 0.46 | 0.62 | 0.80 |
| 32K | 0.56 | 0.72 | 0.89 |

Adjacent-step overlap in these layers is 0.75–0.82 (C2 ≈ borderline); cold mass is the *highest*
here (0.3–0.66 at 32K), so these are also the layers where dropping cold KV is riskiest.

## Hypothesis

**H′:** with exact dense attention in layers 0–22 and 35, and oracle sparse attention
(R + top-k% cold pages, chosen per KV head from exact scores each step) only in layers 23–34,
greedy decoding is unchanged on code-retrieval prompts.

## Arms (fixed)

| Arm | Sparse layers | Cold pages kept | Page | R |
|---|---|---|---|---|
| **A** (primary) | 23–34 (12 of 36) | top 20% | 128 | first 64 + last 1024 tokens |
| B (secondary) | 23–34 | top 10% | 128 | same |

Only these two arms are run. No layer range or k is tuned on E2 data.

## Data (held out)

- Same model, prompt builder (`src/dpinfer/repoqa.py`), needles (0, 1, 2), depth rule and
  lengths (8K, 16K, 32K) as E1.
- **Different repositories:** per language the 3rd and 4th repo in `load_repos()` order
  (E1 used the 1st and 2nd). Repos too small for a length are skipped, as in E1.
- Decision uses the stratum where `full` retrieves the needle correctly; incorrect prompts are
  reported separately.

## Criteria

Primary (arm A must pass both, at each of 16K and 32K; 8K is reported):

| ID | Metric | Threshold |
|---|---|---|
| C3′a | greedy tokens identical to `full` | ≥ 90% of correct prompts |
| C3′b | same needle-retrieval outcome as `full` | 100% of correct prompts |

Secondary (reported; not part of the pass decision): C1 and C2 from the trace run restricted to
layers 23–34, at the arm's k. **Predicted from E1: C1 fails for both arms** (0.72–0.89 at top 20%),
C2 ≈ 0.75–0.80. A pass of H′ therefore rests on end-to-end behaviour, not on the 90%-mass rule.

Arm B is reported with the same criteria; a pass of B without A is not expected and would be
treated as noise unless reproduced.

## What a pass would be worth (bound, stated in advance)

Even if H′ passes, the benefit is capped:

- Sparse layers are 12 of 36. At top 20%, cold bytes read fall by 12/36 × 0.8 ≈ **27%**
  (top 10%: 30%). The CPU-side cold attention can get at most **≈ 1.37×** (1.43×) faster.
  Desktop 32K / 10% budget: hybrid 114.5 ms → ≥ ~89 ms/token, still far from full (18.8 ms).
- The oracle reads **all** keys to pick pages. A practical selector must be much cheaper
  (per-page key summaries, Quest-style min/max bounds), and it can only match or lose to the oracle.
- C2 ≈ 0.77 means the selection changes every step; a GPU page cache would re-fetch ~23% of its
  pages per step, so selection is better done where the KV lives (CPU side).

## How the outcome steers the project

| Outcome | Direction |
|---|---|
| A passes at 16K and 32K | Build a practical page selector for layers 23–34 only, evaluate it against the oracle (same criteria), then measure the real speed-up in `hybrid` (Gate D). Keep expectations at ≤ 1.4× on the CPU part. |
| A passes at 32K only | Same, restricted to long contexts; state the restriction. |
| A fails | Close the sparsity line for Qwen3-4B on code; exact hybrid stays a capacity tool. KV quantisation on the CPU side (bytes, not selection) becomes the next lever. |

## Not decided here

Speed (needs a real selector), other models (the 27B on the L40 may differ; only its
full-attention layers would be in scope), other tasks than needle retrieval.
