"""E2 / Gate C': layer-split oracle sparsity on held-out repos (docs/gate-c-layer-split.md).

Per prompt: one exact greedy run with tracing (secondary C1/C2), then one greedy run per arm with
oracle sparse attention only in the arm's layers. Repos: per language the 3rd and 4th in
load_repos() order (E1 used the 1st and 2nd). Arms and layers are fixed by the pre-registration.
Raw traces -> traces/e2/<date>/; per-prompt rows -> results/e2/<date>/prompts.json (resumable).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

import argparse
import datetime as dt
import json
import time

import torch

from dpinfer import locality
from dpinfer.common import load_model
from dpinfer.repoqa import build, load_repos
from e1_locality import DEPTHS, LANGS, greedy, summarise

SPARSE = set(range(23, 35))
ARMS = {"A": 0.20, "B": 0.10}   # oracle_frac per arm; sparse layers 23-34, page 128 (pre-registered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", nargs="+", type=int, default=[8192, 16384, 32768])
    ap.add_argument("--repo-slice", nargs=2, type=int, default=[2, 4])
    ap.add_argument("--needles", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max-steps", type=int, default=256)
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()
    out_dir, trace_dir = ROOT / "results/e2" / args.date, ROOT / "traces/e2" / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    locality.register()
    tokenizer, model = load_model()
    model.set_attn_implementation("trace")
    cfg = model.config
    st = locality.TraceState(cfg.num_hidden_layers, cfg.num_key_value_heads, max_steps=args.max_steps)
    locality.STATE = st
    e = model.generation_config.eos_token_id
    eos = set(e if isinstance(e, list) else [e])

    repos = load_repos()
    by_lang = {}
    for name, r in repos.items():
        by_lang.setdefault(r["lang"], []).append(name)
    a, b = args.repo_slice
    chosen = [n for lang in LANGS for n in by_lang[lang][a:b]]
    print("repos:", chosen, flush=True)

    pfile = out_dir / "prompts.json"
    rows = json.loads(pfile.read_text()) if pfile.exists() else []
    done = {(r["repo"], r["needle"], r["target"]) for r in rows}
    for n_target in args.lengths:
        for name in chosen:
            for ni, needle in enumerate(args.needles):
                if (name, needle, n_target) in done:
                    continue
                depth = DEPTHS[(ni + chosen.index(name)) % len(DEPTHS)]
                try:
                    msgs, info = build(tokenizer, repos[name], needle, n_target, depth)
                except ValueError as err:
                    print(f"skip {name} needle {needle} @ {n_target}: {err}", flush=True)
                    continue
                ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False, return_tensors="pt",
                                                    return_dict=True)["input_ids"].to("cuda")
                L = ids.shape[-1]
                if L < 0.9 * n_target:
                    print(f"skip {name} needle {needle}: only {L} tokens", flush=True)
                    continue
                t0 = time.time()
                st.reset(); st.mode = "trace"; st.sparse_layers = None
                full = greedy(model, ids, args.max_steps, eos, st)
                steps = st.t
                summ = summarise(st, steps)
                torch.save({"frac": st.frac[:steps].half().cpu(), "overlap": st.overlap[:steps].half().cpu(),
                            "mc": st.mc[:steps].half().cpu(), "pages": locality.PAGES, "tops": locality.TOPS},
                           trace_dir / f"{name.replace('/', '__')}_n{needle}_{n_target}.pt")
                a_full = tokenizer.decode(full, skip_special_tokens=True)
                row = {"repo": name, "lang": repos[name]["lang"], "needle": needle,
                       "needle_name": info["needle_name"], "depth": depth, "target": n_target,
                       "prompt_tokens": L, "decode_steps": steps,
                       "full_found": info["needle_name"] in a_full, "answer_full": a_full}
                for arm, frac in ARMS.items():
                    st.mode, st.sparse_layers, st.oracle_frac = "oracle", SPARSE, frac
                    orc = greedy(model, ids, args.max_steps, eos, st)
                    a_orc = tokenizer.decode(orc, skip_special_tokens=True)
                    m = min(len(full), len(orc))
                    row |= {f"{arm}_found": info["needle_name"] in a_orc, f"{arm}_identical": full == orc,
                            f"{arm}_first_diff": next((i for i in range(m) if full[i] != orc[i]), None),
                            f"{arm}_answer": a_orc}
                st.mode, st.sparse_layers = None, None
                row["seconds"] = round(time.time() - t0, 1)
                row |= {k: v for k, v in summ.items() if k != "P128_top10_share_ge90_per_layer"}
                rows.append(row)
                pfile.write_text(json.dumps(rows, indent=1))
                print(f"{name:28s} n{needle} d{depth} {L:6d} tok {steps:3d} steps | full={row['full_found']!s:5} "
                      + " ".join(f"{k}: found={row[k + '_found']!s:5} ident={row[k + '_identical']!s:5}"
                                 for k in ARMS) + f" | {row['seconds']}s", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
