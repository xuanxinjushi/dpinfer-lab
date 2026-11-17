"""Manual greedy prefill/decode loop with per-forward CUDA-event timing."""
import torch
from transformers import DynamicCache


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


@torch.inference_mode()
def run_once(model, input_ids, decode_steps):
    """Prefill = one forward over the prompt (yields token 1).
    Decode = `decode_steps` single-token forwards over the KV cache (tokens 2..N+1).
    Each forward is bracketed by CUDA events; no host sync between steps.
    Returns (prefill_ms, [decode_ms...], generated_token_ids[1, decode_steps+1])."""
    n_in = input_ids.shape[-1]
    dev = input_ids.device
    cache = DynamicCache(config=model.config)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * (decode_steps + 1))]

    ev[0].record()
    out = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(n_in, device=dev),
        logits_to_keep=1,
    )
    ev[1].record()
    tok = out.logits[:, -1].argmax(-1, keepdim=True)
    tokens = [tok]

    for i in range(decode_steps):
        ev[2 * i + 2].record()
        out = model(
            input_ids=tok,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([n_in + i], device=dev),
        )
        ev[2 * i + 3].record()
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(tok)

    torch.cuda.synchronize()
    prefill_ms = ev[0].elapsed_time(ev[1])
    decode_ms = [ev[2 * i + 2].elapsed_time(ev[2 * i + 3]) for i in range(decode_steps)]
    return prefill_ms, decode_ms, torch.cat(tokens, dim=-1)
