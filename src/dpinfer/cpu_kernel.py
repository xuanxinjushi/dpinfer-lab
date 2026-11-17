"""JIT-built AVX2 CPU decode-attention kernel for bf16/fp16 KV (see csrc/cpu_attn.cpp).

Build artefacts go to <repo>/build/ (gitignored), not ~/.cache, to keep the root disk free.
"""
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load
    build = ROOT / "build" / "cpu_attn"
    build.mkdir(parents=True, exist_ok=True)
    return load(name="dpinfer_cpu_attn", sources=[str(Path(__file__).parent / "csrc" / "cpu_attn.cpp")],
                extra_cflags=["-O3", "-mavx2", "-mfma", "-mf16c", "-fopenmp"],
                extra_ldflags=["-fopenmp"], build_directory=str(build), verbose=False)


def gqa_decode(q, k, v, scale, threads=None):
    """q [Hkv, G, D] fp32; k, v [Hkv, L, D] bf16/fp16 contiguous -> (out [Hkv,G,D] fp32, lse [Hkv,G])."""
    return _ext().gqa_decode(q.contiguous(), k, v, float(scale), int(threads or torch.get_num_threads()))
