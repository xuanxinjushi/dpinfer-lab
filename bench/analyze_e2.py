"""Evaluate Gate C' (docs/gate-c-layer-split.md) from an E2 run -> results/e2/<date>/summary.json.

Primary: per arm, share of correct prompts with identical greedy tokens (>= 0.90) and with the same
retrieval outcome as full (== 1.0), at each length. Secondary: C1 / C2 restricted to layers 23-34
(C1 at the arm's k; C2 is the top-10% overlap, the only one traced).
"""
import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LAYERS = slice(23, 35)
ARMS = {"A": 2, "B": 1}          # index into TOPS = (5, 10, 20) for the arm's k
IDENT, P128 = 0.90, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    args = ap.parse_args()
    res_dir, tr_dir = ROOT / "results/e2" / args.date, ROOT / "traces/e2" / args.date
    rows = json.loads((res_dir / "prompts.json").read_text())
    out = {}
    for target in sorted({r["target"] for r in rows}):
        for stratum in ("correct", "incorrect", "all"):
            rs = [r for r in rows if r["target"] == target and
                  (stratum == "all" or r["full_found"] == (stratum == "correct"))]
            if not rs:
                continue
            trs = [torch.load(tr_dir / f"{r['repo'].replace('/', '__')}_n{r['needle']}_{target}.pt") for r in rs]
            frac = torch.cat([t["frac"].float() for t in trs])[:, P128, :, LAYERS]    # [T, K, 12, H]
            ov = torch.cat([t["overlap"].float() for t in trs])[:, P128, LAYERS]
            e = {"prompts": len(rs)}
            o = ov[~torch.isnan(ov)]
            e["C2_L23_34_top10_overlap"] = o.mean().item()
            for arm, ki in ARMS.items():
                f = frac[:, ki]; f = f[~torch.isnan(f)]
                e[f"{arm}_C1_L23_34_share_ge90"] = (f >= 0.9).float().mean().item()
                e[f"{arm}_identical_share"] = sum(r[f"{arm}_identical"] for r in rs) / len(rs)
                e[f"{arm}_same_found_share"] = sum(r[f"{arm}_found"] == r["full_found"] for r in rs) / len(rs)
                diffs = [r[f"{arm}_first_diff"] for r in rs if r[f"{arm}_first_diff"] is not None]
                e[f"{arm}_first_diff_mean"] = sum(diffs) / len(diffs) if diffs else None
                e[f"{arm}_pass"] = (e[f"{arm}_identical_share"] >= IDENT and e[f"{arm}_same_found_share"] == 1.0)
            out[f"{target}_{stratum}"] = e
            print(f"{target:6d} {stratum:9s} n={len(rs):3d} | C2 {e['C2_L23_34_top10_overlap']:.2f} | "
                  + " | ".join(f"{a}: C1 {e[a + '_C1_L23_34_share_ge90']:.2f} ident {e[a + '_identical_share']:.2f} "
                               f"same-found {e[a + '_same_found_share']:.2f} pass={e[a + '_pass']}" for a in ARMS))
    (res_dir / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
