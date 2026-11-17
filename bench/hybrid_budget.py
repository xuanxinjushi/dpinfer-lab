"""Decode under an artificial GPU KV budget: full vs hybrid (CPU attention) vs offload vs drop.

For each (repo, context length, budget fraction, mode): prefill a RepoQA prompt, run a greedy
decode of `--decode-steps` tokens with `HybridCache`, and record per-step decode latency, GPU KV
bytes, host KV bytes, and whether the generated tokens equal the full-GPU reference (exact modes
must match; `drop` is the lossy baseline).

Example:
  bench/hybrid_budget.py --repos psf/black --lengths 8192 32768 --fracs 0.5 0.25 0.1 \
      --threads 8 --out results/hybrid_budget/2026-11-17
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import datetime as dt
import json
import statistics

import torch

from dpinfer import hybrid
from dpinfer.common import load_model
from dpinfer.repoqa import build, load_repos
from dpinfer.timing import pct, run_once


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=["psf/black"])
    ap.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768])
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.5, 0.25, 0.1],
                    help="GPU KV budget as a fraction of the prompt length")
    ap.add_argument("--modes", nargs="+", default=["hybrid", "offload", "drop"])
    ap.add_argument("--sink", type=int, default=64)
    ap.add_argument("--cpu-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/hybrid_budget" / dt.date.today().isoformat())
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    hybrid.register()
    tokenizer, model = load_model()
    model.set_attn_implementation("hybrid")
    repos = load_repos()
    cpu_dtype = getattr(torch, args.cpu_dtype)
    env = {"torch": torch.__version__, "threads": args.threads, "cpu_dtype": args.cpu_dtype,
           "gpu": torch.cuda.get_device_name()}

    rows = []
    for name in args.repos:
        for n in args.lengths:
            msgs, info = build(tokenizer, repos[name], 0, n)
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False, return_tensors="pt",
                                                return_dict=True)["input_ids"].to("cuda")
            L = ids.shape[-1]
            ref_toks = None
            configs = [("full", 1.0)] + [(m, f) for f in args.fracs for m in args.modes]
            for mode, frac in configs:
                budget = max(args.sink + 1, int(L * frac))
                holder = {}

                def factory(n_in, steps, mode=mode, budget=budget):
                    holder["c"] = hybrid.HybridCache(model.config, mode=mode, budget=budget, sink=args.sink,
                                                     reserve=steps + 8, cpu_dtype=cpu_dtype)
                    return holder["c"]

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                run_once(model, ids, 4, factory)  # warmup
                runs, toks = [], None
                for r in range(args.runs):
                    hybrid.STATS.update(cpu_ms=0.0, h2d_bytes=0)
                    p, d, t = run_once(model, ids, args.decode_steps, factory)
                    runs.append({"prefill_ms": p, "decode_ms": [round(x, 4) for x in d],
                                 "cpu_ms_total": hybrid.STATS["cpu_ms"],
                                 "h2d_bytes": hybrid.STATS["h2d_bytes"]})
                    toks = t if toks is None else toks
                c = holder["c"]
                if mode == "full":
                    ref_toks = toks
                same = torch.equal(toks, ref_toks)
                first_diff = None if same else int((toks != ref_toks).nonzero()[0, 1])
                all_d = [x for r in runs for x in r["decode_ms"]]
                answer = tokenizer.decode(toks[0], skip_special_tokens=True)
                row = {
                    "repo": name, "target": n, "prompt_tokens": L, "mode": mode, "budget_frac": frac,
                    "budget_tokens": budget if mode != "full" else L, "cold_tokens": c.cold_tokens(),
                    "gpu_kv_gib": c.gpu_kv_bytes() / 2**30,
                    "host_kv_gib": sum((l.k_cold.nbytes + l.v_cold.nbytes) if l.k_cold is not None else 0
                                       for l in c.layers) / 2**30,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "decode_ms_median": statistics.median(all_d), "decode_ms_p90": pct(all_d, 90),
                    "cpu_ms_per_step": statistics.median(r["cpu_ms_total"] for r in runs) / args.decode_steps,
                    "h2d_gb_per_step": statistics.median(r["h2d_bytes"] for r in runs) / args.decode_steps / 1e9,
                    "prefill_ms_median": statistics.median(r["prefill_ms"] for r in runs),
                    "tokens_equal_full": same, "first_diff_step": first_diff,
                    "needle_name_in_answer": info["needle_name"] in answer,
                }
                rows.append(row)
                print(f"{name:14s} L={L:6d} {mode:7s} budget={frac:4.2f} cold={row['cold_tokens']:6d} | "
                      f"decode {row['decode_ms_median']:7.2f} ms (cpu {row['cpu_ms_per_step']:6.2f}, "
                      f"h2d {row['h2d_gb_per_step']:5.2f} GB) | gpuKV {row['gpu_kv_gib']:5.2f} GiB "
                      f"| equal_full={same} diff@{first_diff}", flush=True)
                del c, holder
                hybrid.release_host_memory()
                (args.out / "hybrid_budget.json").write_text(
                    json.dumps({"env": env, "config": vars(args) | {"out": str(args.out)}, "rows": rows},
                               indent=1, default=str))


if __name__ == "__main__":
    main()
