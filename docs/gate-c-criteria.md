# Gate C — locality: pre-registered pass criteria

Status: **criteria fixed 2026-11-17, before any E1 data was collected.** Changing a threshold after
seeing data must be recorded in this file with the date and reason.

**Outcome (2026-11-17): Gate C failed** — C1, C2 and C3 all fail at 8K, 16K and 32K on correctly
answered prompts; thresholds unchanged. Details: `docs/findings.md`, `results/e1/2026-11-17/`.

Gate C asks whether attention over historical code KV is concentrated and stable enough that keeping
or computing only a small subset of KV pages per step is justified. The method (E1) is in README §8;
this file fixes what counts as a pass.

## Data

- Model: Qwen3-4B (same checkpoint as the baselines), greedy decode.
- Prompts: RepoQA "search needle function" prompts built by `src/dpinfer/repoqa.py`, 6 languages,
  contexts 8K / 16K / 32K, several needles and depths per repo (more than the n=6 of the sweep).
- Decode steps: the full answer (until EOS, capped at 256), every step.
- **Stratify by outcome:** every metric is reported separately for prompts where `full` retrieves the
  needle correctly and where it does not. The pass decision uses the **correct** stratum only;
  incorrect prompts may have diffuse or misdirected attention (see `docs/review-gemini.md` §4).
  At 32K this stratum is small (1/6 in the first sweep), so 32K results need more prompts.

## Definitions

Logical KV pages of $P$ tokens (default $P = 128$; also report 64 and 256). For decode step $t$,
layer $l$, KV head $h$, the attention probabilities of the $G = 4$ query heads sharing $h$ are summed
(GQA: one KV page serves all four, so placement decisions are per KV head):

$$
A_{t,l,h,b} = \sum_{g=1}^{G} \sum_{i \in b} \mathrm{softmax}\!\left(\frac{q_{t,l,(h,g)} K_{l,h}^{\top}}{\sqrt{d}}\right)_i .
$$

- **Always-resident set** $R$: the first 64 tokens (sinks) and the most recent 1024 tokens; these are
  on the GPU in every scheme and excluded from the selection metrics below.
- **Cold pages** $C$: all other pages. Cold mass $M_C = \sum_{b \in C} A_{t,l,h,b}$.
- **Top-k set** $S_k(t,l,h)$: the $k$% of cold pages with the largest $A$.

## Criteria (all must pass)

| ID | Metric | Threshold |
|---|---|---|
| C1 concentration | fraction of $(t,l,h)$ with $\sum_{b \in S_{10}} A / M_C \ge 0.90$ | ≥ 90% |
| C2 temporal reuse | mean over $(t,l,h)$ of $\lvert S_{10}(t) \cap S_{10}(t-1)\rvert / \lvert S_{10}(t)\rvert$ | ≥ 0.80 |
| C3 end-to-end | oracle sparse attention (R + $S_{10}$, chosen with exact scores each step): greedy tokens identical to `full` | ≥ 90% of prompts |
| C3 end-to-end | same needle-retrieval outcome as `full` | 100% of prompts |

Also reported (no threshold): the same metrics for top 5% / 20%; the distribution per layer and head
(early layers may be diffuse — a pass may be layer-restricted, which must be stated); the mean
first-divergence step for C3 failures; cold mass $M_C$ itself (if it is tiny, R alone suffices and
the cold KV hardly matters).

C3 uses an *oracle* (exact scores) on purpose: it bounds what any practical selector can achieve.
A practical selector (page summaries, Quest-style min/max keys, previous-step reuse) is evaluated
afterwards against the oracle.

## How the outcome steers the project

| Outcome | Meaning | Direction |
|---|---|---|
| C1, C2, C3 pass | Few, stable hot pages | Sparse placement. Then compare **GPU page cache + PCIe fetch** (Quest/ShadowKV-style) against **CPU computes the cold tail** — the latter wins only where host DRAM bandwidth ≫ PCIe, PCIe is shared by many requests, or CPU-side KV is quantized (review §2). |
| C1 passes, C2 fails | Concentrated but shifting | Per-step selection is needed; fetch latency is exposed every step, which favours CPU-side compute over a GPU cache. |
| C1 fails | Attention is diffuse | No sparsity story. Exact hybrid as a capacity tool only, or KV compression / quantization. |
| Pass only on correct prompts | Locality tied to success | Report both; sparsity must not be claimed for prompts the model already fails. |

## Not decided by Gate C

Speed (Gate B/D) and coding-task quality beyond needle retrieval (Gate E).
