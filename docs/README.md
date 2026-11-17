# dpinfer docs

Design notes and research reasoning. Measured results live in `results/`; this folder explains
what we are testing, why, and how the code does it.

| Doc | What it covers |
|---|---|
| [findings.md](findings.md) | Running log of what has been established so far, with pointers to raw data |
| [kv-budget-experiment.md](kv-budget-experiment.md) | The main experiment: decode under an artificial GPU KV budget, methods compared, metrics, what counts as a win |
| [hybrid-attention-implementation.md](hybrid-attention-implementation.md) | How `HybridCache` / hybrid attention plug into HF transformers, exactness argument, limitations |
| [l40-outlook.md](l40-outlook.md) | What these results mean for the L40 server and what to run there |

Conventions:

- Numbers marked **measured** come from a file in `results/`; numbers marked **estimate** are
  back-of-envelope and must be replaced by measurements before being cited.
- RTX 4090 and L40 numbers are never mixed in one table without labelling the machine.
