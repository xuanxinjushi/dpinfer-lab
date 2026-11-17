"""E1 / Gate C: attention locality on RepoQA code prompts, plus oracle sparse decode (C3).

Per prompt (repo, needle, depth, target length):
  1. greedy decode with exact attention, tracing per-page attention metrics (stops at EOS, ≤ max steps)
  2. greedy decode with oracle sparse attention (R + top-10% cold pages, P=128)
Raw per-step metrics -> traces/e1/<date>/<prompt>.pt (large, not in git);
per-prompt summaries -> results/e1/<date>/prompts.json; Gate C evaluation -> summary.json.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import datetime as dt
import json
import time

import torch

from dpinfer import locality
from dpinfer.common import load_model
from dpinfer.kvcache import make_cache
from dpinfer.repoqa import build, load_repos

LANGS = ["python", "cpp", "java", "typescript", "rust", "go"]
DEPTHS = [0.25, 0.5, 0.75]


@torch.inference_mode()
def greedy(model, ids, max_steps, eos, st):
    """Manual greedy loop with an in-place cache; st.t is advanced per decode step."""
    n = ids.shape[-1]
    cache = make_cache("inplace", model, n + max_steps + 1)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                cache_position=torch.arange(n, device=ids.device), logits_to_keep=1)
    tok = out.logits[:, -1].argmax(-1, keepdim=True)
    toks = [tok.item()]
    st.t = 0
    for i in range(max_steps):
        if toks[-1] in eos:
            break
        out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                    cache_position=torch.tensor([n + i], device=ids.device))
        st.t += 1
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        toks.append(tok.item())
    return toks


def summarise(st, steps):
    """Per-prompt reductions over (t, l, h) for each page size / top-k."""
    T = max(1, steps)
    frac, ov, mc = st.frac[:T], st.overlap[:T], st.mc[:T]
    res = {}
    for pi, P in enumerate(locality.PAGES):
        for ki, k in enumerate(locality.TOPS):
            f = frac[:, pi, ki]
            ok = ~torch.isnan(f)
            res[f"P{P}_top{k}_mean_frac"] = f[ok].mean().item()
            res[f"P{P}_top{k}_share_ge90"] = (f[ok] >= 0.9).float().mean().item()
        o = ov[:, pi]
        ok = ~torch.isnan(o)
        res[f"P{P}_top10_overlap_mean"] = o[ok].mean().item() if ok.any() else float("nan")
    m = mc[~torch.isnan(mc)]
    res["cold_mass_mean"] = m.mean().item()
    res["cold_mass_median"] = m.median().item()
    # per-layer C1 (P=128, top10) for layer-restricted analysis
    f = frac[:, 1, 1]
    res["P128_top10_share_ge90_per_layer"] = [
        ((f[:, l] >= 0.9).float().nansum() / (~torch.isnan(f[:, l])).sum().clamp_min(1)).item()
        for l in range(f.shape[1])]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    ap.add_argument("--repos-per-lang", type=int, default=2)
    ap.add_argument("--needles", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-steps", type=int, default=256)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    out_dir = ROOT / "results/e1" / args.date
    trace_dir = ROOT / "traces/e1" / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    locality.register()
    tokenizer, model = load_model()
    model.set_attn_implementation("trace")
    cfg = model.config
    st = locality.TraceState(cfg.num_hidden_layers, cfg.num_key_value_heads, max_steps=args.max_steps)
    locality.STATE = st
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])

    repos = load_repos()
    by_lang = {}
    for name, r in repos.items():
        by_lang.setdefault(r["lang"], []).append(name)
    chosen = [n for lang in LANGS for n in by_lang[lang][:args.repos_per_lang]]

    pfile = out_dir / "prompts.json"
    rows = json.loads(pfile.read_text()) if pfile.exists() else []
    done = {(r["repo"], r["needle"], r["target"]) for r in rows}
    for n_target in args.lengths:
        for name in chosen:
            for ni, needle in enumerate(args.needles):
                if (name, needle, n_target) in done:
                    continue
                depth = DEPTHS[(ni + chosen.index(name)) % len(DEPTHS)]
                msgs, info = build(tokenizer, repos[name], needle, n_target, depth)
                ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False, return_tensors="pt",
                                                    return_dict=True)["input_ids"].to("cuda")
                L = ids.shape[-1]
                if L < 0.9 * n_target:
                    print(f"skip {name} needle {needle}: only {L} tokens", flush=True)
                    continue
                t0 = time.time()
                st.reset(); st.mode = "trace"
                full = greedy(model, ids, args.max_steps, eos, st)
                steps = st.t
                summ = summarise(st, steps)
                torch.save({"frac": st.frac[:steps].half().cpu(), "overlap": st.overlap[:steps].half().cpu(),
                            "mc": st.mc[:steps].half().cpu(), "pages": locality.PAGES, "tops": locality.TOPS},
                           trace_dir / f"{name.replace('/', '__')}_n{needle}_{n_target}.pt")
                st.mode = "oracle"
                orc = greedy(model, ids, args.max_steps, eos, st)
                st.mode = None
                a_full = tokenizer.decode(full, skip_special_tokens=True)
                a_orc = tokenizer.decode(orc, skip_special_tokens=True)
                m = min(len(full), len(orc))
                diff = next((i for i in range(m) if full[i] != orc[i]), None)
                row = {"repo": name, "lang": repos[name]["lang"], "needle": needle, "needle_name": info["needle_name"],
                       "depth": depth, "target": n_target, "prompt_tokens": L, "decode_steps": steps,
                       "full_found": info["needle_name"] in a_full,
                       "oracle_found": info["needle_name"] in a_orc,
                       "oracle_identical": full == orc, "oracle_first_diff": diff,
                       "answer_full": a_full, "answer_oracle": a_orc,
                       "seconds": round(time.time() - t0, 1)} | summ
                rows.append(row)
                pfile.write_text(json.dumps(rows, indent=1))
                print(f"{name:28s} n{needle} d{depth} {L:6d} tok {steps:3d} steps | found full={row['full_found']!s:5} "
                      f"oracle={row['oracle_found']!s:5} identical={row['oracle_identical']!s:5} | "
                      f"C1 {summ['P128_top10_share_ge90']:.3f} frac {summ['P128_top10_mean_frac']:.3f} "
                      f"C2 {summ['P128_top10_overlap_mean']:.3f} Mc {summ['cold_mass_mean']:.3f} | {row['seconds']}s",
                      flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
