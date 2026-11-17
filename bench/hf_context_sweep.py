"""Prefill/decode timing vs context length on real RepoQA code (HF eager baseline).

For each (repo, context length): build a needle-function retrieval prompt of ~N tokens,
warm up, check the manual loop against greedy generate(), then time `--runs` runs of
prefill + `--decode-steps` decode forwards. Writes one JSON per config plus summary.csv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import csv
import datetime as dt
import json
import statistics
import subprocess

import torch
import transformers

from dpinfer.common import load_model
from dpinfer.repoqa import build, load_repos
from dpinfer.timing import pct, run_once

DEFAULT_REPOS = ["psf/black", "skypjack/uvw", "google/gson",
                 "cheeriojs/cheerio", "serde-rs/serde", "fatedier/frp"]


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        name, drv, used, total = [s.strip() for s in out.split(",")]
        return {"gpu": name, "driver": drv, "mem_used_mib": int(used), "mem_total_mib": int(total)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=DEFAULT_REPOS)
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 8192, 16384, 32768])
    ap.add_argument("--needle", type=int, default=0)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--decode-steps", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cache", choices=["dynamic", "inplace", "static"], default="dynamic",
                    help="KV cache: HF DynamicCache (torch.cat), preallocated in-place, or StaticCache")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute answer/match/found for existing JSONs in --out; keep timings")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/context_sweep" / dt.date.today().isoformat())
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    background = gpu_snapshot()  # before we allocate anything
    tokenizer, model = load_model()
    repos = load_repos()
    env = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "attn_implementation": model.config._attn_implementation,
        "dtype": str(model.dtype), "background_gpu": background, "cache": args.cache,
        "weights_gib": torch.cuda.memory_allocated() / 2**30,
    }
    print(json.dumps(env), flush=True)

    summary_rows = []
    for name in args.repos:
        for n in args.lengths:
            msgs, info = build(tokenizer, repos[name], args.needle, n, args.depth)
            input_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt", return_dict=True)["input_ids"].to("cuda")

            fn = args.out / f"{name.replace('/', '__')}_{n}.json"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if args.rescore:  # recompute answer/match only; keep stored timings
                rec = json.loads(fn.read_text())
                _, _, toks = run_once(model, input_ids, args.decode_steps, args.cache)
                match, answer = check_answer(model, tokenizer, input_ids, toks, args.decode_steps)
                rec["answer"] = answer
                rec["summary"]["matches_generate"] = match
                rec["summary"]["needle_name_in_answer"] = info["needle_name"] in answer
                rec["summary"]["answer_tokens"] = len(tokenizer(answer).input_ids)
                fn.write_text(json.dumps(rec, indent=1, default=str))
                s = rec["summary"]
                summary_rows.append({"repo": name, "lang": info["lang"], "target": n,
                                     "prompt_tokens": info["prompt_tokens"]} | {
                    k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
                print(f"{name:18s} {info['prompt_tokens']:6d} tok | match={match} "
                      f"found={s['needle_name_in_answer']} ({s['answer_tokens']} answer tok)", flush=True)
                write_summary(args.out, summary_rows)
                continue

            run_once(model, input_ids, 16, args.cache)  # warmup at this shape

            runs = []
            for r in range(args.runs):
                p, d, toks = run_once(model, input_ids, args.decode_steps, args.cache)
                runs.append({"prefill_ms": p, "decode_ms": [round(x, 4) for x in d]})
                if r == 0:
                    first_toks = toks
            peak = torch.cuda.max_memory_allocated() / 2**30
            match, answer = check_answer(model, tokenizer, input_ids, first_toks, args.decode_steps)

            all_d = [x for r in runs for x in r["decode_ms"]]
            prefills = [r["prefill_ms"] for r in runs]
            tps = [len(r["decode_ms"]) / sum(r["decode_ms"]) * 1e3 for r in runs]
            s = {
                "prefill_ms_median": statistics.median(prefills),
                "prefill_ms_min": min(prefills), "prefill_ms_max": max(prefills),
                "prefill_tok_per_s": info["prompt_tokens"] / statistics.median(prefills) * 1e3,
                "decode_ms_median": statistics.median(all_d),
                "decode_ms_p10": pct(all_d, 10), "decode_ms_p90": pct(all_d, 90),
                "decode_ms_p99": pct(all_d, 99),
                "decode_tok_per_s_median": statistics.median(tps),
                "decode_tok_per_s_min": min(tps), "decode_tok_per_s_max": max(tps),
                "peak_vram_gib": peak, "matches_generate": match,
                "needle_name_in_answer": info["needle_name"] in answer,
                "answer_tokens": len(tokenizer(answer).input_ids),
            }
            rec = {"config": vars(args) | {"out": str(args.out)}, "env": env,
                   "prompt": info, "summary": s, "answer": answer, "runs": runs}
            fn.write_text(json.dumps(rec, indent=1, default=str))
            row = {"repo": name, "lang": info["lang"], "target": n,
                   "prompt_tokens": info["prompt_tokens"]} | {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
            summary_rows.append(row)
            print(f"{name:18s} {info['prompt_tokens']:6d} tok | prefill {s['prefill_ms_median']:8.1f} ms "
                  f"| decode {s['decode_ms_median']:6.2f} ms/tok (p90 {s['decode_ms_p90']:6.2f}) "
                  f"| {s['decode_tok_per_s_median']:6.2f} tok/s | peak {peak:5.2f} GiB "
                  f"| match={match} found={s['needle_name_in_answer']}", flush=True)

            write_summary(args.out, summary_rows)


def write_summary(out, rows):
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@torch.inference_mode()
def check_answer(model, tokenizer, input_ids, manual_toks, decode_steps):
    """Greedy generate() with natural stopping (EOS allowed). The manual loop never stops,
    so compare it only up to generate()'s length; the answer is generate()'s output."""
    ref = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                         max_new_tokens=decode_steps + 1, do_sample=False)[:, input_ids.shape[-1]:]
    m = ref.shape[-1]
    match = torch.equal(manual_toks[:, :m], ref)
    return match, tokenizer.decode(ref[0], skip_special_tokens=True)


if __name__ == "__main__":
    main()
