# dpinfer docs

Design notes and research reasoning. Measured results live in `results/`; this folder explains
what we are testing, why, and how the code does it.

| Doc | What it covers |
|---|---|
| [findings.md](findings.md) | Running log of what has been established so far, with pointers to raw data |
| [kv-budget-experiment.md](kv-budget-experiment.md) | The main experiment: decode under an artificial GPU KV budget, methods compared, metrics, what counts as a win |
| [hybrid-attention-implementation.md](hybrid-attention-implementation.md) | How `HybridCache` / hybrid attention plug into HF transformers, exactness argument, limitations |
| [gate-c-criteria.md](gate-c-criteria.md) | Pre-registered pass/fail thresholds for Gate C (attention locality), fixed before E1 |
| [review-gemini.md](review-gemini.md) | External review of the first results (fp32 offload strawman, CPU vs GPU page cache, hygiene) |
| [l40-outlook.md](l40-outlook.md) | What these results mean for the L40 server and what to run there |
| [l40-runbook.md](l40-runbook.md) | Step-by-step commands for running the experiments on the L40 server |
| [model-choice.md](model-choice.md) | Which model to run where (Qwen3-4B first; 27B FP8 / Qwen3-14B next) and what to check first |

Conventions:

- Numbers marked **measured** come from a file in `results/`; numbers marked **estimate** are
  back-of-envelope and must be replaced by measurements before being cited.
- RTX 4090 and L40 numbers are never mixed in one table without labelling the machine.
