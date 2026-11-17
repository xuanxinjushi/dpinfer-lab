"""Separate prefill and decode timing with a manual greedy forward loop over a KV cache.

Prefill = one forward over the whole prompt (produces the 1st new token).
Decode  = N single-token forwards reusing the cache (produce tokens 2..N+1).
Each forward is bracketed by CUDA events, so there is no host sync between steps.
Argmax/cat between forwards is included in neither prefill nor decode time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import statistics

import torch
from transformers import DynamicCache

from dpinfer.common import build_inputs, load_model, warmup


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


@torch.inference_mode()
def run_once(model, input_ids, decode_steps):
    n_in = input_ids.shape[-1]
    cache = DynamicCache(config=model.config)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * (decode_steps + 1))]

    ev[0].record()
    out = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(n_in, device=input_ids.device),
        logits_to_keep=1,
    )
    ev[1].record()
    tok = out.logits[:, -1].argmax(-1, keepdim=True)
    tokens = [tok]

    for i in range(decode_steps):
        pos = n_in + i
        ev[2 * i + 2].record()
        out = model(
            input_ids=tok,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([pos], device=input_ids.device),
        )
        ev[2 * i + 3].record()
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(tok)

    torch.cuda.synchronize()
    prefill_ms = ev[0].elapsed_time(ev[1])
    decode_ms = [ev[2 * i + 2].elapsed_time(ev[2 * i + 3]) for i in range(decode_steps)]
    return prefill_ms, decode_ms, torch.cat(tokens, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-steps", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer)
    input_ids = inputs["input_ids"]
    n_in = input_ids.shape[-1]

    warmup(model, inputs)
    run_once(model, input_ids, 16)  # warm the manual path too

    # Sanity check: manual loop must reproduce greedy generate().
    with torch.inference_mode():
        ref = model.generate(
            **inputs,
            max_new_tokens=args.decode_steps + 1,
            min_new_tokens=args.decode_steps + 1,
            do_sample=False,
        )[:, n_in:]

    torch.cuda.reset_peak_memory_stats()
    results = []
    for r in range(args.runs):
        prefill_ms, decode_ms, toks = run_once(model, input_ids, args.decode_steps)
        results.append((prefill_ms, decode_ms))
        if r == 0:
            match = torch.equal(toks, ref)
        med = statistics.median(decode_ms)
        print(
            f"run {r}: prefill {prefill_ms:7.2f} ms | decode median {med:6.2f} ms/tok "
            f"p90 {pct(decode_ms, 90):6.2f} | total decode {sum(decode_ms):8.1f} ms "
            f"-> {len(decode_ms) / sum(decode_ms) * 1e3:6.2f} tok/s"
        )

    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    prefills = [p for p, _ in results]
    all_decode = [d for _, ds in results for d in ds]
    decode_tps = [len(ds) / sum(ds) * 1e3 for _, ds in results]

    print()
    print(f"Input tokens:          {n_in}")
    print(f"Decode steps per run:  {args.decode_steps} (+1 token from prefill)")
    print(f"Runs:                  {args.runs}")
    print(f"Matches generate():    {match}")
    print(f"Prefill median:        {statistics.median(prefills):.2f} ms "
          f"(min {min(prefills):.2f}, max {max(prefills):.2f})")
    print(f"Decode per-token:      median {statistics.median(all_decode):.2f} ms, "
          f"p10 {pct(all_decode, 10):.2f}, p90 {pct(all_decode, 90):.2f}, "
          f"p99 {pct(all_decode, 99):.2f}")
    print(f"Decode throughput:     median {statistics.median(decode_tps):.2f} tok/s "
          f"(min {min(decode_tps):.2f}, max {max(decode_tps):.2f})")
    print(f"Peak VRAM allocated:   {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
