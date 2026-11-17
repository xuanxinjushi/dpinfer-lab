"""Print the architecture facts that decide whether a model works with dpinfer's HybridCache.

Usage: python scratch/inspect_config.py /path/to/model_dir_or_config.json
"""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
c = json.loads((p / "config.json" if p.is_dir() else p).read_text())
t = c.get("text_config", c)
keys = ["architectures", "model_type", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "head_dim", "hidden_size", "intermediate_size", "max_position_embeddings", "rope_scaling",
        "sliding_window", "layer_types", "full_attention_interval", "linear_num_key_heads",
        "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
        "num_experts", "num_experts_per_tok", "torch_dtype", "quantization_config"]
for k in keys:
    v = c.get(k, t.get(k))
    if v is None:
        continue
    if k == "layer_types":
        kinds = {x: v.count(x) for x in sorted(set(v))}
        print(f"layer_types = {len(v)} layers {kinds}; first 8: {v[:8]}")
    else:
        print(f"{k} = {v}")

L = t.get("num_hidden_layers")
kv = t.get("num_key_value_heads")
hd = t.get("head_dim") or (t["hidden_size"] // t["num_attention_heads"] if t.get("hidden_size") else None)
lt = t.get("layer_types")
n_full = sum(x == "full_attention" for x in lt) if lt else L
if n_full and kv and hd:
    per_tok = n_full * kv * hd * 2 * 2  # K+V, bf16
    print(f"\nKV cache (bf16) for {n_full} growing-KV layers: {per_tok / 1024:.0f} KiB/token, "
          f"{per_tok * 32768 / 2**30:.2f} GiB per 32K request")
