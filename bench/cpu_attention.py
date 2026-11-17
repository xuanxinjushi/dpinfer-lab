"""Gate B microbenchmarks: can the host serve decode attention for long-context KV?

Measures, on this machine:
  1) host DRAM read / copy bandwidth vs CPU threads (torch, large buffers >> LLC);
  2) PCIe host->GPU bandwidth from pinned memory (one layer of KV and a full-cache equivalent);
  3) CPU decode attention for Qwen3-4B shapes (36 layers, 32 Q / 8 KV heads, d=128, GQA),
     one query token over L cached tokens, PyTorch ops (fp32 by default; bf16/fp16 are very
     slow in PyTorch on x86), checked against GPU SDPA;
  4) the AVX2 bf16 kernel (src/dpinfer/csrc/cpu_attn.cpp) on the same shapes — the CPU path
     hybrid attention actually uses;
  5) the same attention on GPU (flash SDPA) for reference.

Usage: bench/cpu_attention.py --out results/gate_b/<date>/cpu_attention.json
"""
import argparse
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LAYERS, HQ, HKV, D = 36, 32, 8, 128
G = HQ // HKV


def timeit(fn, reps, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def host_bandwidth(threads, gib=2.0):
    n = int(gib * 2**30 / 4)
    a = torch.empty(n, dtype=torch.float32).fill_(1.0)
    b = torch.empty_like(a)
    out = []
    for t in threads:
        torch.set_num_threads(t)
        rd = timeit(lambda: a.sum(), 5)
        cp = timeit(lambda: b.copy_(a), 5)
        out.append({"threads": t, "read_gbs": a.nbytes / rd / 1e9,
                    "copy_gbs": 2 * a.nbytes / cp / 1e9})  # copy = read + write
        print(f"  threads {t:2d}: read {out[-1]['read_gbs']:6.1f} GB/s  copy(r+w) {out[-1]['copy_gbs']:6.1f} GB/s",
              flush=True)
    del a, b
    return out


def pcie_bandwidth():
    res = {}
    for name, nbytes in [("one_layer_32k_kv", 32768 * HKV * D * 2 * 2), ("1_gib", 2**30)]:
        h = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
        g = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

        def h2d():
            g.copy_(h, non_blocking=True)
            torch.cuda.synchronize()

        def d2h():
            h.copy_(g, non_blocking=True)
            torch.cuda.synchronize()
        res[name] = {"bytes": nbytes, "h2d_gbs": nbytes / timeit(h2d, 10) / 1e9,
                     "d2h_gbs": nbytes / timeit(d2h, 10) / 1e9}
        print(f"  {name}: H2D {res[name]['h2d_gbs']:5.1f} GB/s  D2H {res[name]['d2h_gbs']:5.1f} GB/s", flush=True)
        del h, g
    return res


def cpu_attn_layer(q, k, v):
    """q [HKV, G, D]; k, v [HKV, L, D] -> [HKV, G, D]. Explicit GQA, no K/V repeat."""
    s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(D))
    p = torch.softmax(s.float(), dim=-1).to(v.dtype)
    return torch.matmul(p, v)


def attention_bench(lengths, dtypes, threads, reps):
    rows = []
    g = torch.Generator().manual_seed(0)
    for dt_name in dtypes:
        dt = getattr(torch, dt_name)
        for L in lengths:
            # One KV tensor per layer, like a real cache (total >> LLC).
            ks = [torch.randn(HKV, L, D, generator=g).to(dt) for _ in range(LAYERS)]
            vs = [torch.randn(HKV, L, D, generator=g).to(dt) for _ in range(LAYERS)]
            qs = [torch.randn(HKV, G, D, generator=g).to(dt) for _ in range(LAYERS)]
            kv_bytes = 2 * LAYERS * HKV * L * D * ks[0].element_size()

            # correctness vs GPU SDPA on layer 0
            torch.set_num_threads(max(threads))
            ref = F.scaled_dot_product_attention(
                qs[0].reshape(1, HQ, 1, D).cuda().float(),
                ks[0][None].cuda().float(), vs[0][None].cuda().float(),
                enable_gqa=True).reshape(HKV, G, D).cpu()
            err = (cpu_attn_layer(qs[0], ks[0], vs[0]).float() - ref).abs().max().item()

            for t in threads:
                torch.set_num_threads(t)
                sec = timeit(lambda: [cpu_attn_layer(qs[i], ks[i], vs[i]) for i in range(LAYERS)], reps)
                rows.append({"dtype": dt_name, "ctx": L, "threads": t, "kv_bytes": kv_bytes,
                             "ms_all_layers": sec * 1e3, "eff_gbs": kv_bytes / sec / 1e9,
                             "max_abs_err_vs_gpu": err})
                print(f"  cpu {dt_name:8s} L={L:6d} thr={t:2d}: {sec*1e3:8.2f} ms / 36 layers  "
                      f"({kv_bytes/sec/1e9:6.1f} GB/s eff, err {err:.1e})", flush=True)
            del ks, vs, qs
    return rows


def kernel_bench(lengths, threads, reps):
    """AVX2 bf16 kernel (what hybrid uses), all 36 layers, checked against fp32 PyTorch."""
    from dpinfer.cpu_kernel import gqa_decode
    from dpinfer.hybrid import cpu_partial
    rows = []
    g = torch.Generator().manual_seed(0)
    scale = 1 / math.sqrt(D)
    for L in lengths:
        ks = [torch.randn(HKV, L, D, generator=g).bfloat16() for _ in range(LAYERS)]
        vs = [torch.randn(HKV, L, D, generator=g).bfloat16() for _ in range(LAYERS)]
        q = torch.randn(HKV, G, D, generator=g)
        ref, _ = cpu_partial(q, ks[0].float(), vs[0].float(), scale)
        err = (gqa_decode(q, ks[0], vs[0], scale, max(threads))[0] - ref).abs().max().item()
        kv_bytes = 2 * LAYERS * HKV * L * D * 2
        for t in threads:
            sec = timeit(lambda: [gqa_decode(q, ks[i], vs[i], scale, t) for i in range(LAYERS)], reps)
            rows.append({"dtype": "bfloat16", "ctx": L, "threads": t, "kv_bytes": kv_bytes,
                         "ms_all_layers": sec * 1e3, "eff_gbs": kv_bytes / sec / 1e9, "max_abs_err": err})
            print(f"  kernel bf16 L={L:6d} thr={t:2d}: {sec*1e3:8.2f} ms / 36 layers  "
                  f"({kv_bytes/sec/1e9:6.1f} GB/s eff, err {err:.1e})", flush=True)
        del ks, vs
    return rows


@torch.inference_mode()
def gpu_attention(lengths):
    rows = []
    for L in lengths:
        ks = [torch.randn(1, HKV, L, D, device="cuda", dtype=torch.bfloat16) for _ in range(LAYERS)]
        vs = [torch.randn_like(k) for k in ks]
        q = torch.randn(1, HQ, 1, D, device="cuda", dtype=torch.bfloat16)

        def run():
            for i in range(LAYERS):
                F.scaled_dot_product_attention(q, ks[i], vs[i], enable_gqa=True)
        for _ in range(3):
            run()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ms = []
        for _ in range(10):
            s.record(); run(); e.record(); torch.cuda.synchronize()
            ms.append(s.elapsed_time(e))
        m = statistics.median(ms)
        kv_bytes = 2 * LAYERS * HKV * L * D * 2
        rows.append({"ctx": L, "ms_all_layers": m, "eff_gbs": kv_bytes / m / 1e6})
        print(f"  gpu bf16 L={L:6d}: {m:7.3f} ms / 36 layers ({kv_bytes/m/1e6:6.1f} GB/s eff)", flush=True)
        del ks, vs
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 8192, 16384, 32768])
    ap.add_argument("--dtypes", nargs="+", default=["float32"],
                    help="PyTorch CPU attention dtypes (bfloat16/float16 take seconds per call at 32K)")
    ap.add_argument("--no-kernel", action="store_true", help="skip the AVX2 bf16 kernel benchmark")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4, 8, 16, 24])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    env = {"cpu": platform.processor() or open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
           "logical_cpus": os.cpu_count(), "torch": torch.__version__,
           "cpu_capability": torch.backends.cpu.get_cpu_capability(),
           "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS")}
    print(json.dumps(env), flush=True)
    res = {"env": env}
    print("host bandwidth:", flush=True)
    res["host_bandwidth"] = host_bandwidth(args.threads)
    print("pcie:", flush=True)
    res["pcie"] = pcie_bandwidth()
    print("gpu attention:", flush=True)
    res["gpu_attention"] = gpu_attention(args.lengths)
    print("cpu attention:", flush=True)
    res["cpu_attention"] = []
    for dt in args.dtypes:
        res["cpu_attention"] += attention_bench(args.lengths, [dt], args.threads, args.reps)
        args.out.write_text(json.dumps(res, indent=1))
    if not args.no_kernel:
        print("cpu kernel (AVX2 bf16):", flush=True)
        res["cpu_kernel"] = kernel_bench(args.lengths, args.threads, args.reps)
    res["env"]["argv"] = sys.argv
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
