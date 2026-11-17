"""Summarise a context sweep directory: per-length table, linear decode-cost fit, plots."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Qwen3-4B: 36 layers x 8 KV heads x 128 dim x (K,V) x 2 bytes (bf16)
KV_BYTES_PER_TOKEN = 36 * 8 * 128 * 2 * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", type=Path)
    args = ap.parse_args()
    d = args.sweep_dir
    df = pd.read_csv(d / "summary.csv")
    env = json.loads(next(d.glob("*.json")).read_text())["env"]
    w_bytes = env["weights_gib"] * 2**30
    steps = json.loads(next(d.glob("*.json")).read_text())["config"]["decode_steps"]

    # Mean context seen during decode = prompt + half the generated tokens.
    df["ctx_mean"] = df["prompt_tokens"] + steps / 2
    df["kv_gib"] = df["ctx_mean"] * KV_BYTES_PER_TOKEN / 2**30
    df["eff_bw_gbs"] = (w_bytes + df["ctx_mean"] * KV_BYTES_PER_TOKEN) / (df["decode_ms_median"] * 1e-3) / 1e9

    # decode_ms = a + b * ctx  (a: per-step fixed cost; b: cost per context token)
    b, a = np.polyfit(df["ctx_mean"], df["decode_ms_median"], 1)
    resid = df["decode_ms_median"] - (a + b * df["ctx_mean"])
    kv_bw = KV_BYTES_PER_TOKEN / (b * 1e-3) / 1e9  # GB/s implied for the context-dependent part
    w_bw = w_bytes / (a * 1e-3) / 1e9

    g = df.groupby("target")
    tab = pd.DataFrame({
        "prompt_tokens": g["prompt_tokens"].median().astype(int),
        "kv_cache_gib": g["kv_gib"].median().round(2),
        "prefill_ms": g["prefill_ms_median"].median().round(1),
        "prefill_tok_s": g["prefill_tok_per_s"].median().round(0).astype(int),
        "decode_ms": g["decode_ms_median"].median().round(2),
        "decode_ms_min_repo": g["decode_ms_median"].min().round(2),
        "decode_ms_max_repo": g["decode_ms_median"].max().round(2),
        "decode_tok_s": g["decode_tok_per_s_median"].median().round(2),
        "eff_bw_gbs": g["eff_bw_gbs"].median().round(0).astype(int),
        "peak_vram_gib": g["peak_vram_gib"].max().round(2),
        "found": g["needle_name_in_answer"].sum().astype(str) + "/" + g.size().astype(str),
        "match_generate": g["matches_generate"].all(),
    })
    base = tab["decode_ms"].iloc[0]
    tab["decode_vs_first"] = (tab["decode_ms"] / base).round(2)
    # Pure bandwidth model: same effective BW as the shortest context, extra bytes from KV.
    bw0 = df[df["target"] == df["target"].min()]["eff_bw_gbs"].median() * 1e9
    tab["bw_model_ms"] = ((w_bytes + (tab["prompt_tokens"] + steps / 2) * KV_BYTES_PER_TOKEN)
                          / bw0 * 1e3).round(2)

    fit = {"fixed_ms": a, "ms_per_1k_ctx": b * 1000, "max_abs_resid_ms": float(resid.abs().max()),
           "implied_weight_bw_gbs": w_bw, "implied_kv_bw_gbs": kv_bw,
           "attn_share_at_32k": float(b * 32768 / (a + b * 32768))}
    (d / "analysis.json").write_text(json.dumps({"table": tab.reset_index().to_dict("records"),
                                                 "fit": fit, "env": env}, indent=1, default=str))
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab.to_string())
    print(json.dumps(fit, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for repo, sub in df.groupby("repo"):
        ax[0].plot(sub["prompt_tokens"], sub["decode_ms_median"], "o-", ms=4, lw=1, alpha=.7, label=repo)
        ax[1].plot(sub["prompt_tokens"], sub["prefill_ms_median"] / 1e3, "o-", ms=4, lw=1, alpha=.7, label=repo)
    xs = np.linspace(0, df["ctx_mean"].max(), 100)
    ax[0].plot(xs, a + b * xs, "k--", lw=1, label=f"fit: {a:.1f} + {b*1000:.2f}·ctx/1K ms")
    ax[0].plot(xs, (w_bytes + xs * KV_BYTES_PER_TOKEN) / bw0 * 1e3, "k:", lw=1,
               label=f"bandwidth model @ {bw0/1e9:.0f} GB/s")
    ax[0].set(xlabel="prompt tokens", ylabel="decode ms / token", title="Decode latency vs context",
              ylim=(0, None))
    ax[1].set(xlabel="prompt tokens", ylabel="prefill s", title="Prefill time vs context", ylim=(0, None))
    for a_ in ax:
        a_.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Qwen3-4B BF16, HF {env['transformers']} ({env['attn_implementation']}), "
                 f"{env['background_gpu'].get('gpu', '')}", fontsize=10)
    fig.tight_layout()
    fig.savefig(d / "context_sweep.png", dpi=130)
    print("wrote", d / "context_sweep.png")


if __name__ == "__main__":
    main()
