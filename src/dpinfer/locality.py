"""E1 / Gate C: attention-locality tracing and oracle sparse attention (see docs/gate-c-criteria.md).

Registered as attention implementation "trace". Prefill and non-decode calls go to HF SDPA.
During decode (one query token) with STATE.mode:

  "trace"   exact SDPA output; additionally compute exact attention probabilities per KV head
            (summed over its G query heads, normalised to 1) and record per-page metrics.
  "oracle"  attend only to the always-resident set R (first `sink` + last `recent` positions)
            plus the top `oracle_frac` cold pages (page size `oracle_page`), chosen per
            (layer, KV head) from exact scores at this step. Lossy upper bound for any selector.
  None      plain SDPA.

Metrics recorded per decode step t, layer l, KV head h, page size P and top-k fraction k:
  frac[t, P, k, l, h]  = cold mass in the top-k% cold pages / total cold mass      (C1)
  overlap[t, P, l, h]  = |S_10(t) ∩ S_10(t-1)| / |S_10(t)|  (NaN at the first step) (C2)
  mc[t, l, h]          = total cold mass M_C
"""
import math

import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

PAGES = (64, 128, 256)
TOPS = (5, 10, 20)


class TraceState:
    def __init__(self, n_layers, n_kv, sink=64, recent=1024, max_steps=256,
                 oracle_page=128, oracle_frac=0.10):
        self.n_layers, self.n_kv = n_layers, n_kv
        self.sink, self.recent, self.max_steps = sink, recent, max_steps
        self.oracle_page, self.oracle_frac = oracle_page, oracle_frac
        self.mode = None
        self.reset()

    def reset(self):
        dev = "cuda"
        L, H, T = self.n_layers, self.n_kv, self.max_steps
        self.t = 0
        self.frac = torch.full((T, len(PAGES), len(TOPS), L, H), float("nan"), device=dev)
        self.overlap = torch.full((T, len(PAGES), L, H), float("nan"), device=dev)
        self.mc = torch.full((T, L, H), float("nan"), device=dev)
        self.prev_sel = {}  # (P, layer) -> bool [H, n_pages] for top-10% at the previous step

    def cold_range(self, n):
        lo, hi = self.sink, n - self.recent
        return (lo, hi) if hi > lo else None

    def record(self, layer, p):
        """p: [H, n] attention probabilities per KV head (rows sum to 1)."""
        t = self.t
        if t >= self.max_steps:
            return
        n = p.shape[1]
        cr = self.cold_range(n)
        if cr is None:
            return
        lo, hi = cr
        cold = torch.zeros(n, device=p.device, dtype=torch.bool)
        cold[lo:hi] = True
        pc = p * cold
        self.mc[t, layer] = pc.sum(1)
        for pi, P in enumerate(PAGES):
            npg = math.ceil(n / P)
            pm = F.pad(pc, (0, npg * P - n)).view(self.n_kv, npg, P).sum(-1)    # [H, npg]
            pl, ph = lo // P, math.ceil(hi / P)
            sub = pm[:, pl:ph]
            ncold = ph - pl
            mc = sub.sum(1).clamp_min(1e-30)
            srt, idx = sub.sort(dim=1, descending=True)
            csum = srt.cumsum(1)
            for ki, k in enumerate(TOPS):
                kk = max(1, math.ceil(k / 100 * ncold))
                self.frac[t, pi, ki, layer] = csum[:, kk - 1] / mc
            kk = max(1, math.ceil(0.10 * ncold))
            sel = torch.zeros(self.n_kv, npg, dtype=torch.bool, device=p.device)
            sel.scatter_(1, idx[:, :kk] + pl, True)
            prev = self.prev_sel.get((P, layer))
            if prev is not None:
                m = min(prev.shape[1], npg)
                inter = (sel[:, :m] & prev[:, :m]).sum(1).float()
                self.overlap[t, pi, layer] = inter / kk
            self.prev_sel[(P, layer)] = sel

    def oracle_mask(self, p):
        """Bool [H, n]: R plus the top oracle_frac cold pages by exact cold mass."""
        H, n = p.shape
        keep = torch.zeros(H, n, dtype=torch.bool, device=p.device)
        keep[:, :self.sink] = True
        keep[:, max(0, n - self.recent):] = True
        cr = self.cold_range(n)
        if cr is None:
            return torch.ones_like(keep)
        lo, hi = cr
        P = self.oracle_page
        npg = math.ceil(n / P)
        cold = torch.zeros(n, device=p.device, dtype=torch.bool)
        cold[lo:hi] = True
        pm = F.pad(p * cold, (0, npg * P - n)).view(H, npg, P).sum(-1)
        pl, ph = lo // P, math.ceil(hi / P)
        kk = max(1, math.ceil(self.oracle_frac * (ph - pl)))
        top = pm[:, pl:ph].topk(kk, dim=1).indices + pl                          # [H, kk]
        pos = (top[:, :, None] * P + torch.arange(P, device=p.device)).view(H, -1)
        pos = pos.clamp_max(n - 1)
        keep.scatter_(1, pos, True)
        return keep


STATE: TraceState | None = None


def trace_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    st = STATE
    if st is None or st.mode is None or query.shape[2] != 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, **kwargs)
    _, hq, _, d = query.shape
    hkv = key.shape[1]
    g = hq // hkv
    scale = scaling if scaling is not None else 1 / math.sqrt(d)
    q = query[0, :, 0].reshape(hkv, g, d).float()
    s = torch.einsum("hgd,hld->hgl", q, key[0].float()) * scale
    p = torch.softmax(s, dim=-1).sum(1) / g                                     # [H, n]
    if st.mode == "trace":
        st.record(module.layer_idx, p)
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, **kwargs)
    keep = st.oracle_mask(p).repeat_interleave(g, 0)[None, :, None, :]          # [1, Hq, 1, n]
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=keep, scale=scale, enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


def register():
    AttentionInterface.register("trace", trace_attention)
    AttentionMaskInterface.register("trace", sdpa_mask)
