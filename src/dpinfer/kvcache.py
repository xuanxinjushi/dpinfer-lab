"""Copy-free KV cache for HF transformers.

`DynamicCache` appends with `torch.cat`, which re-copies the whole cache every decode step.
`InPlaceLayer` preallocates `[B, H_kv, max_len, D]` once and writes new tokens into it in place,
returning views of the filled prefix. The returned K/V are contiguous-in-seq views, no mask is
needed for single-token decode, so SDPA keeps using the FlashAttention split-KV kernel.
"""
import torch
from transformers.cache_utils import Cache, DynamicLayer


class InPlaceLayer(DynamicLayer):
    def __init__(self, max_len: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.length = 0

    def lazy_initialization(self, key_states, value_states):
        b, h, _, d = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device
        self.k_buf = torch.empty(b, h, self.max_len, d, dtype=self.dtype, device=self.device)
        self.v_buf = torch.empty(b, h, self.max_len, value_states.shape[-1], dtype=self.dtype,
                                 device=self.device)
        self.keys, self.values = self.k_buf[:, :, :0], self.v_buf[:, :, :0]
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        n = key_states.shape[-2]
        end = self.length + n
        if end > self.max_len:
            raise RuntimeError(f"InPlaceLayer overflow: {end} > {self.max_len}")
        self.k_buf[:, :, self.length:end].copy_(key_states)
        self.v_buf[:, :, self.length:end].copy_(value_states)
        self.length = end
        self.keys, self.values = self.k_buf[:, :, :end], self.v_buf[:, :, :end]
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.length

    def get_max_length(self) -> int:
        return self.max_len

    def reset(self):
        self.length = 0
        if self.is_initialized:
            self.keys, self.values = self.k_buf[:, :, :0], self.v_buf[:, :, :0]

    def crop(self, tokens_to_remove: int):
        raise NotImplementedError


class InPlaceCache(Cache):
    def __init__(self, config, max_len: int):
        n = config.get_text_config(decoder=True).num_hidden_layers
        super().__init__(layers=[InPlaceLayer(max_len) for _ in range(n)])


def make_cache(kind: str, model, max_len: int):
    from transformers import DynamicCache, StaticCache
    if kind == "dynamic":
        return DynamicCache(config=model.config)
    if kind == "inplace":
        return InPlaceCache(model.config, max_len)
    if kind == "static":
        return StaticCache(config=model.config, max_cache_len=max_len)
    raise ValueError(kind)
