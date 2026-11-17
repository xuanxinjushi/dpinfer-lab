"""Where does decode time go at long context? Kernel breakdown + DynamicCache vs StaticCache.

For each context length: prefill a RepoQA prompt, then
  1) time `--steps` decode forwards with CUDA events for DynamicCache and StaticCache, and
  2) profile the same decode steps (DynamicCache and StaticCache) with torch.profiler and
     bucket GPU kernel time into matmul / attention / KV-cache copy / other.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import re
import statistics

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import DynamicCache, StaticCache

from dpinfer.common import load_model
from dpinfer.repoqa import build, load_repos

BUCKETS = [
    ("attention", re.compile(r"flash|fmha|attention|efficient|sdpa|softmax", re.I)),
    ("kv_cat/copy", re.compile(r"CatArray|cat_|index_copy|index_put|scatter", re.I)),
    ("matmul", re.compile(r"gemm|gemv|cutlass|xmma|sm\d\d_|ampere|ada|cublas|splitK|matmul", re.I)),
]


def bucket(name):
    for b, rx in BUCKETS:
        if rx.search(name):
            return b
    return "other"


def make_cache(kind, model, max_len):
    if kind == "dynamic":
        return DynamicCache(config=model.config)
    return StaticCache(config=model.config, max_cache_len=max_len)


@torch.inference_mode()
def prefill(model, ids, cache):
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                cache_position=torch.arange(ids.shape[-1], device=ids.device), logits_to_keep=1)
    return out.logits[:, -1].argmax(-1, keepdim=True)


@torch.inference_mode()
def decode(model, tok, cache, start, steps, timed=True):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * steps)] if timed else None
    for i in range(steps):
        if timed:
            ev[2 * i].record()
        out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                    cache_position=torch.tensor([start + i], device=tok.device))
        if timed:
            ev[2 * i + 1].record()
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    return [ev[2 * i].elapsed_time(ev[2 * i + 1]) for i in range(steps)] if timed else None, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="psf/black")
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024, 8192, 32768])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--profile-steps", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model()
    repo = load_repos()[args.repo]
    results = []
    for n in args.lengths:
        msgs, info = build(tokenizer, repo, 0, n)
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_tensors="pt",
                                            return_dict=True)["input_ids"].to("cuda")
        L = ids.shape[-1]
        row = {"target": n, "prompt_tokens": L}
        for kind in ["dynamic", "static"]:
            torch.cuda.empty_cache()
            max_len = L + args.steps + args.profile_steps + 8
            cache = make_cache(kind, model, max_len)
            tok = prefill(model, ids, cache)
            decode(model, tok, cache, L, 4, timed=False)  # warm
            ms, tok = decode(model, tok, cache, L + 4, args.steps)
            row[f"{kind}_decode_ms_median"] = statistics.median(ms)

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                decode(model, tok, cache, L + 4 + args.steps, args.profile_steps, timed=False)
            agg, top = {}, []
            for e in prof.key_averages():
                t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                if t <= 0 or e.key.startswith("ProfilerStep"):
                    continue
                agg[bucket(e.key)] = agg.get(bucket(e.key), 0) + t
                top.append((t, e.key))
            total = sum(agg.values())
            row[f"{kind}_kernel_ms_per_step"] = total / 1e3 / args.profile_steps
            row[f"{kind}_share"] = {k: round(v / total, 3) for k, v in sorted(agg.items())}
            row[f"{kind}_top_kernels"] = [
                {"ms_per_step": round(t / 1e3 / args.profile_steps, 3), "bucket": bucket(k), "name": k[:120]}
                for t, k in sorted(top, reverse=True)[:12]]
            del cache
        results.append(row)
        print(f"\n=== {args.repo} {L} tokens ===")
        for kind in ["dynamic", "static"]:
            print(f"{kind:8s} decode {row[f'{kind}_decode_ms_median']:6.2f} ms/tok | kernels "
                  f"{row[f'{kind}_kernel_ms_per_step']:6.2f} ms/step | share {row[f'{kind}_share']}")
            for k in row[f"{kind}_top_kernels"][:6]:
                print(f"    {k['ms_per_step']:7.3f} ms  [{k['bucket']:11s}] {k['name']}")
        args.out.write_text(json.dumps({"repo": args.repo, "results": results}, indent=1))


if __name__ == "__main__":
    main()
