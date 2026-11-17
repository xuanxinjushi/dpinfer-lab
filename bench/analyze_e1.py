"""Evaluate Gate C (docs/gate-c-criteria.md) from an E1 run.

Pools all (step, layer, KV head) cases from traces/e1/<date>/*.pt, stratified by whether the
full model retrieved the needle, per context length. Writes results/e1/<date>/summary.json and
gate_c.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
PAGES, TOPS = (64, 128, 256), (5, 10, 20)
C1_FRAC, C1_SHARE, C2_MIN, C3_IDENT = 0.90, 0.90, 0.80, 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    args = ap.parse_args()
    res_dir, tr_dir = ROOT / "results/e1" / args.date, ROOT / "traces/e1" / args.date
    rows = json.loads((res_dir / "prompts.json").read_text())

    groups = {}
    for r in rows:
        tr = torch.load(tr_dir / f"{r['repo'].replace('/', '__')}_n{r['needle']}_{r['target']}.pt")
        for key in [(r["target"], "correct" if r["full_found"] else "incorrect"), (r["target"], "all")]:
            g = groups.setdefault(key, {"frac": [], "overlap": [], "mc": [], "rows": []})
            g["frac"].append(tr["frac"].float())      # [T, P, K, L, H]
            g["overlap"].append(tr["overlap"].float())  # [T, P, L, H]
            g["mc"].append(tr["mc"].float())          # [T, L, H]
            g["rows"].append(r)

    out = {}
    for (target, stratum), g in sorted(groups.items()):
        frac = torch.cat(g["frac"]); ov = torch.cat(g["overlap"]); mc = torch.cat(g["mc"])
        rs = g["rows"]
        e = {"prompts": len(rs), "decode_steps": int(frac.shape[0])}
        for pi, P in enumerate(PAGES):
            for ki, k in enumerate(TOPS):
                f = frac[:, pi, ki]; ok = ~torch.isnan(f)
                e[f"P{P}_top{k}_share_ge90"] = (f[ok] >= C1_FRAC).float().mean().item()
                e[f"P{P}_top{k}_mean_frac"] = f[ok].mean().item()
            o = ov[:, pi]; ok = ~torch.isnan(o)
            e[f"P{P}_top10_overlap_mean"] = o[ok].mean().item()
        f = frac[:, 1, 1]
        e["P128_top10_share_ge90_per_layer"] = [((f[:, l] >= C1_FRAC).float().sum() /
                                                 (~torch.isnan(f[:, l])).sum().clamp_min(1)).item()
                                                for l in range(f.shape[1])]
        o = ov[:, 1]
        e["P128_top10_overlap_per_layer"] = [o[:, l][~torch.isnan(o[:, l])].mean().item()
                                             for l in range(o.shape[1])]
        m = mc[~torch.isnan(mc)]
        e["cold_mass_mean"] = m.mean().item()
        e["cold_mass_per_layer"] = [mc[:, l][~torch.isnan(mc[:, l])].mean().item() for l in range(mc.shape[1])]
        e["oracle_identical_share"] = sum(r["oracle_identical"] for r in rs) / len(rs)
        e["oracle_same_found_share"] = sum(r["oracle_found"] == r["full_found"] for r in rs) / len(rs)
        e["C1_pass"] = e["P128_top10_share_ge90"] >= C1_SHARE
        e["C2_pass"] = e["P128_top10_overlap_mean"] >= C2_MIN
        e["C3_pass"] = e["oracle_identical_share"] >= C3_IDENT and e["oracle_same_found_share"] == 1.0
        out[f"{target}_{stratum}"] = e
        print(f"{target:6d} {stratum:9s} n={len(rs):3d} steps={e['decode_steps']:6d} | "
              f"C1 share≥.9 {e['P128_top10_share_ge90']:.3f} (mean frac {e['P128_top10_mean_frac']:.3f}; "
              f"top20 {e['P128_top20_mean_frac']:.3f}) | C2 {e['P128_top10_overlap_mean']:.3f} | "
              f"Mc {e['cold_mass_mean']:.3f} | C3 ident {e['oracle_identical_share']:.2f} "
              f"same-found {e['oracle_same_found_share']:.2f} | pass C1={e['C1_pass']} C2={e['C2_pass']} "
              f"C3={e['C3_pass']}")
    (res_dir / "summary.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for key, e in out.items():
        if not key.endswith("_correct"):
            continue
        lab = key.replace("_correct", " (correct)")
        ax[0].plot(e["P128_top10_share_ge90_per_layer"], label=lab)
        ax[1].plot(e["P128_top10_overlap_per_layer"], label=lab)
        ax[2].plot(e["cold_mass_per_layer"], label=lab)
    ax[0].axhline(C1_SHARE, ls="--", c="k", lw=.8)
    ax[1].axhline(C2_MIN, ls="--", c="k", lw=.8)
    ax[0].set(title="C1: share of (t,h) with top-10% pages ≥ 90% cold mass", xlabel="layer")
    ax[1].set(title="C2: top-10% page overlap, adjacent steps", xlabel="layer")
    ax[2].set(title="cold attention mass M_C", xlabel="layer")
    for a in ax:
        a.grid(alpha=.3); a.set_ylim(0, 1.02); a.legend(fontsize=7)
    fig.suptitle("Gate C — Qwen3-4B, RepoQA code, page 128, correctly answered prompts", fontsize=10)
    fig.tight_layout()
    fig.savefig(res_dir / "gate_c.png", dpi=130)


if __name__ == "__main__":
    main()
