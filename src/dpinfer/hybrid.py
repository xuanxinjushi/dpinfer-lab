"""Hybrid CPU/GPU KV cache + attention for HF Qwen3 under an explicit GPU KV budget.

The GPU holds at most `budget` prompt tokens of KV per layer ("hot": the first `sink` tokens plus
the most recent prompt tokens) plus every generated token. The rest of the prompt ("cold") lives in
host memory. Decode attention per layer, depending on `mode`:

  full     all KV on GPU (reference; budget ignored)
  hybrid   GPU flash attention over hot KV  ||  CPU attention over cold KV, exact LSE fusion
  offload  cold KV copied host->GPU every step, GPU flash over it, exact LSE fusion
  drop     hot KV only (StreamingLLM-style eviction; lossy baseline)

Prefill runs normal SDPA over the full prompt one layer at a time, then moves that layer's cold KV
to the host, so only one layer's full KV is transiently on the GPU.

Fusion: for partitions P with log-sum-exp lse_P and normalised output o_P,
    lse = logaddexp(lse_G, lse_C);  o = exp(lse_G - lse) o_G + exp(lse_C - lse) o_C
"""
import math
import time

import torch
from transformers import AttentionInterface
from transformers.cache_utils import Cache, DynamicLayer
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

MODES = ("full", "hybrid", "offload", "drop")
_ACTIVE = {"cache": None}
STATS = {"cpu_ms": 0.0, "h2d_bytes": 0, "steps": 0}


def _to_host(x, dtype, pin):
    h = x.to(dtype).cpu().contiguous()
    return h.pin_memory() if pin else h


def release_host_memory():
    """Return cached pinned blocks to the OS (the host caching allocator keeps them otherwise)."""
    torch._C._host_emptyCache()


class HybridLayer(DynamicLayer):
    def __init__(self, mode, budget, sink, reserve, cpu_dtype):
        super().__init__()
        self.mode, self.budget, self.sink, self.reserve, self.cpu_dtype = mode, budget, sink, reserve, cpu_dtype
        self.length = 0      # tokens seen (hot + cold)
        self.n_hot = 0       # tokens in the GPU buffer
        self.k_cold = self.v_cold = None

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        n = key_states.shape[-2]
        if self.length == 0:                       # prefill: split into hot / cold
            b, h, _, d = key_states.shape
            keep_all = self.mode == "full" or n <= self.budget
            hot_idx = (torch.arange(n, device=self.device) if keep_all else torch.cat([
                torch.arange(self.sink, device=self.device),
                torch.arange(n - (self.budget - self.sink), n, device=self.device)]))
            cap = hot_idx.numel() + self.reserve
            self.k_buf = torch.empty(b, h, cap, d, dtype=self.dtype, device=self.device)
            self.v_buf = torch.empty_like(self.k_buf)
            self.k_buf[:, :, :hot_idx.numel()] = key_states[:, :, hot_idx]
            self.v_buf[:, :, :hot_idx.numel()] = value_states[:, :, hot_idx]
            self.n_hot = hot_idx.numel()
            if not keep_all:
                lo, hi = self.sink, n - (self.budget - self.sink)
                pin = self.mode == "offload"           # pinned only where async H2D needs it
                # offload ships KV in the model dtype (what a real system would transfer);
                # cpu_dtype only applies to the CPU-attention path.
                dt = self.dtype if self.mode == "offload" else self.cpu_dtype
                self.k_cold = _to_host(key_states[0, :, lo:hi], dt, pin)
                self.v_cold = _to_host(value_states[0, :, lo:hi], dt, pin)
            self.length = n
            self.keys, self.values = self.k_buf[:, :, :self.n_hot], self.v_buf[:, :, :self.n_hot]
            return key_states, value_states            # prefill attends over the full prompt
        end = self.n_hot + n
        if end > self.k_buf.shape[2]:
            raise RuntimeError("HybridLayer: decode reserve exhausted")
        self.k_buf[:, :, self.n_hot:end].copy_(key_states)
        self.v_buf[:, :, self.n_hot:end].copy_(value_states)
        self.n_hot, self.length = end, self.length + n
        self.keys, self.values = self.k_buf[:, :, :end], self.v_buf[:, :, :end]
        return self.keys, self.values

    @property
    def n_cold(self):
        return 0 if self.k_cold is None else self.k_cold.shape[1]

    def get_seq_length(self):
        return self.length

    def get_mask_sizes(self, query_length):
        return self.length + query_length, 0

    def get_max_length(self):
        return -1

    def gpu_kv_bytes(self):
        return 0 if self.length == 0 else 2 * self.k_buf.numel() * self.k_buf.element_size()


class Prefetcher:
    """Layer-wise async H2D prefetch for `offload`: a dedicated copy stream and two GPU staging
    slots. While layer l computes on slot l%2, layer l+1's cold KV is copied into the other slot;
    after the last layer it prefetches layer 0 for the next token."""

    def __init__(self, layers):
        self.layers = layers
        self.stream = torch.cuda.Stream()
        self.slots = None
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.free = [torch.cuda.Event() for _ in range(2)]
        self.issued = None  # layer index whose copy is in flight / done in its slot

    def _alloc(self):
        l0 = self.layers[0]
        shape = (1,) + tuple(l0.k_cold.shape)
        dev = l0.device
        self.slots = [(torch.empty(shape, dtype=l0.k_cold.dtype, device=dev),
                       torch.empty(shape, dtype=l0.v_cold.dtype, device=dev)) for _ in range(2)]
        self.issued = set()

    def _issue(self, idx):
        s = idx % 2
        layer = self.layers[idx]
        self.stream.wait_event(self.free[s])           # slot's previous user has finished
        with torch.cuda.stream(self.stream):
            self.slots[s][0][0].copy_(layer.k_cold, non_blocking=True)
            self.slots[s][1][0].copy_(layer.v_cold, non_blocking=True)
            self.ready[s].record(self.stream)
        STATS["h2d_bytes"] += layer.k_cold.nbytes + layer.v_cold.nbytes
        self.issued.add(idx)

    def get(self, idx):
        if self.slots is None:
            self._alloc()
        if idx not in self.issued:
            self._issue(idx)
        torch.cuda.current_stream().wait_event(self.ready[idx % 2])
        self.issued.discard(idx)
        return idx % 2

    def release_and_prefetch_next(self, idx):
        s = idx % 2
        self.free[s].record(torch.cuda.current_stream())
        nxt = (idx + 1) % len(self.layers)
        if nxt not in self.issued:
            self._issue(nxt)

    def bytes(self):
        return 0 if self.slots is None else sum(t.nbytes for sl in self.slots for t in sl)


class HybridCache(Cache):
    def __init__(self, config, mode="hybrid", budget=4096, sink=64, reserve=512, cpu_dtype=torch.float32,
                 prefetch=True):
        assert mode in MODES
        n = config.get_text_config(decoder=True).num_hidden_layers
        super().__init__(layers=[HybridLayer(mode, budget, sink, reserve, cpu_dtype) for _ in range(n)])
        self.mode = mode
        self.prefetcher = Prefetcher(self.layers) if (mode == "offload" and prefetch) else None
        _ACTIVE["cache"] = self

    def gpu_kv_bytes(self):
        """Resident KV plus offload staging buffers."""
        return sum(l.gpu_kv_bytes() for l in self.layers) + (
            self.prefetcher.bytes() if self.prefetcher else 0)

    def cold_tokens(self):
        return self.layers[0].n_cold


def _flash_lse(q, k, v, scale):
    """q [1, Hkv, G, D] (GQA packed as G query rows), k/v [1, Hkv, L, D] -> out [1,Hkv,G,D], lse [1,Hkv,G]."""
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False, scale=scale)[:2]
    return out, lse


def cpu_partial(q, k, v, scale):
    """q [Hkv, G, D] fp32 (cpu); k/v [Hkv, L, D] (cpu) -> out [Hkv, G, D] fp32, lse [Hkv, G] fp32."""
    s = torch.matmul(q.to(k.dtype), k.transpose(-1, -2)).float() * scale
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None]).to(v.dtype)
    return torch.matmul(p, v).float(), lse


def _fuse(o_g, lse_g, o_c, lse_c):
    lse = torch.logaddexp(lse_g, lse_c)
    return (o_g.float() * torch.exp(lse_g - lse)[..., None]
            + o_c.float() * torch.exp(lse_c - lse)[..., None])


def hybrid_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    cache = _ACTIVE["cache"]
    layer = cache.layers[module.layer_idx] if cache is not None else None
    if layer is None or query.shape[2] != 1 or cache.mode == "full" or layer.n_cold == 0:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, **kwargs)
    b, hq, _, d = query.shape
    hkv = key.shape[1]
    g = hq // hkv
    scale = scaling if scaling is not None else 1 / math.sqrt(d)
    qp = query.reshape(b, hkv, g, d)               # Q head j*g+i -> KV head j (matches repeat_kv)

    if cache.mode == "hybrid":
        q_cpu = qp[0].float().cpu()                 # sync point: needs this layer's query
        o_g, lse_g = _flash_lse(qp, key, value, scale)   # async on GPU while the CPU works
        t0 = time.perf_counter()
        o_c, lse_c = cpu_partial(q_cpu, layer.k_cold, layer.v_cold, scale)
        STATS["cpu_ms"] += (time.perf_counter() - t0) * 1e3
        o_c = o_c.to(query.device, non_blocking=True)[None]
        lse_c = lse_c.to(query.device, non_blocking=True)[None]
        out = _fuse(o_g, lse_g, o_c, lse_c)
    elif cache.mode == "offload" and cache.prefetcher is not None:
        pf, idx = cache.prefetcher, module.layer_idx
        o_g, lse_g = _flash_lse(qp, key, value, scale)
        s = pf.get(idx)
        kc, vc = pf.slots[s]
        o_c, lse_c = _flash_lse(qp, kc, vc, scale)
        pf.release_and_prefetch_next(idx)
        out = _fuse(o_g, lse_g, o_c, lse_c)
    elif cache.mode == "offload":                   # synchronous copy, no prefetch
        kc = layer.k_cold.to(query.device, non_blocking=True)[None]
        vc = layer.v_cold.to(query.device, non_blocking=True)[None]
        STATS["h2d_bytes"] += layer.k_cold.nbytes + layer.v_cold.nbytes
        o_g, lse_g = _flash_lse(qp, key, value, scale)
        o_c, lse_c = _flash_lse(qp, kc, vc, scale)
        out = _fuse(o_g, lse_g, o_c, lse_c)
    else:  # drop
        out, _ = _flash_lse(qp, key, value, scale)
    out = out.to(query.dtype).reshape(b, hq, 1, d).transpose(1, 2).contiguous()
    return out, None


def register():
    AttentionInterface.register("hybrid", hybrid_attention)
    AttentionMaskInterface.register("hybrid", sdpa_mask)
