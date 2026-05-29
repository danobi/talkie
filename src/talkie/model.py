"""Talkie 13B transformer architecture.

A 40-layer, 40-head decoder-only GPT with RoPE, SwiGLU, RMS normalisation,
embedding skip connections, and per-head / per-layer gain parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from talkie.sampling import sample_from_logits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 65536
    n_layer: int = 40
    n_head: int = 40
    n_embd: int = 5120
    head_dim: int = 128


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class HeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones([n_head]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class WeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.w_g.type_as(w)


class ActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a_g.type_as(x)


# ---------------------------------------------------------------------------
# Attention & MLP
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        n_state = config.n_embd

        self.attn_query = nn.Linear(n_state, n_state, bias=False)
        self.attn_key = nn.Linear(n_state, n_state, bias=False)
        self.attn_value = nn.Linear(n_state, n_state, bias=False)
        self.attn_resid = nn.Linear(n_state, n_state, bias=False)
        self.head_gain = HeadGain(config.n_head)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple,
        start_pos: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.attn_query(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.attn_key(x).view(bsz, seq_len, self.n_head, self.head_dim)
        v = self.attn_value(x).view(bsz, seq_len, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)

        if k_cache is not None:
            # Chunked prefill (seq_len > 1 with start_pos > 0) would need an
            # explicit mask; current callers only hit (start_pos=0, any seq_len)
            # or (start_pos>0, seq_len=1).
            assert start_pos == 0 or seq_len == 1
            end_pos = start_pos + seq_len
            k_cache[:bsz, start_pos:end_pos] = k
            v_cache[:bsz, start_pos:end_pos] = v
            k = k_cache[:bsz, :end_pos]
            v = v_cache[:bsz, :end_pos]
            is_causal = start_pos == 0
        else:
            is_causal = True

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=is_causal,
            )
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.attn_resid(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        n_state = config.n_embd
        n_mlp = int(round(((8 / 3) * n_state) / 128) * 128)

        self.mlp_gate = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_linear = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_resid = nn.Linear(n_mlp, n_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.mlp_gate(x)) * self.mlp_linear(x)
        return self.mlp_resid(x)


# ---------------------------------------------------------------------------
# Transformer block & full model
# ---------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.attn_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.mlp = MLP(config)
        self.mlp_gain = ActGain((2 * config.n_layer) ** -0.5)
        self.embed_skip = ActGain(0.0)

    def forward(
        self,
        e_x: torch.Tensor,
        x: torch.Tensor,
        cos_sin: tuple,
        start_pos: int = 0,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn_gain(
            self.attn(F.rms_norm(x, (x.shape[-1],)), cos_sin, start_pos=start_pos, k_cache=k_cache, v_cache=v_cache)
        )
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        x = x + self.embed_skip(e_x)
        return x


class TalkieModel(nn.Module):
    """Talkie 13B decoder-only transformer."""

    def __init__(
        self, config: GPTConfig, device: torch.device, max_seq_len: int = 4096
    ):
        super().__init__()
        self.config = config
        self.device = device

        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Parameter(torch.zeros(config.vocab_size, config.n_embd))
        self.lm_head_gain = WeightGain()

        cos, sin = self._precompute_rotary_embeddings(max_seq_len, config.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.suppress_token_ids: set[int] | None = None

    def _precompute_rotary_embeddings(
        self, seq_len: int, head_dim: int, base: int = 1_000_000
    ) -> tuple:
        device = self.embed.weight.device if hasattr(self, "embed") else "cpu"
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run a forward pass and return ``[B, V]`` logits for the last position.

        When the KV cache is provided, ``start_pos`` is the absolute position
        of ``input_ids[:, 0]`` in the full sequence; the new K/V are written
        into the cache at that offset and attention reads back over the full
        cached prefix. With no cache provided, ``start_pos`` should be 0 and
        the call is equivalent to a stateless forward.
        """
        _, seq_len = input_ids.shape
        end_pos = start_pos + seq_len
        cos_sin = self.cos[:, start_pos:end_pos], self.sin[:, start_pos:end_pos]

        x = self.embed(input_ids)
        x = F.rms_norm(x, (x.shape[-1],))
        e_x = x
        for i, block in enumerate(self.blocks):
            if kv_cache is not None:
                k_cache = kv_cache[f"blocks.{i}.k"]
                v_cache = kv_cache[f"blocks.{i}.v"]
            else:
                k_cache = v_cache = None
            x = block(e_x, x, cos_sin, start_pos=start_pos, k_cache=k_cache, v_cache=v_cache)
        x = F.rms_norm(x, (x.shape[-1],))

        return F.linear(x[:, -1, :], self.lm_head_gain(self.lm_head)).float()

    def new_kv_cache(self, batch_size: int, max_seq_len: int) -> dict[str, torch.Tensor]:
        """Create a fresh per-layer KV cache for one generation."""
        if max_seq_len > self.cos.shape[1]:
            cos, sin = self._precompute_rotary_embeddings(
                max_seq_len, self.config.head_dim
            )
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        shape = (batch_size, max_seq_len, self.config.n_head, self.config.head_dim)
        dtype = self.embed.weight.dtype
        device = self.embed.weight.device
        cache: dict[str, torch.Tensor] = {}
        for i in range(self.config.n_layer):
            cache[f"blocks.{i}.k"] = torch.zeros(shape, device=device, dtype=dtype)
            cache[f"blocks.{i}.v"] = torch.zeros(shape, device=device, dtype=dtype)
        return cache

    def sample_batch(
        self,
        x: torch.Tensor,
        t: float | torch.Tensor = 0.7,
        top_p: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample one token per sequence in the batch.

        *t* may be a scalar (one temperature for the whole batch) or a tensor of
        shape ``[B]`` / ``[B, 1]`` (per-sequence temperatures).
        """
        logits = self.forward(x, start_pos=start_pos, kv_cache=kv_cache)
        return sample_from_logits(logits, temperature=t, top_p=top_p, top_k=top_k)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def resize_model_embeddings(
    model: TalkieModel, new_vocab_size: int, device: torch.device | str
) -> TalkieModel:
    """Grow embedding and lm_head to *new_vocab_size*, keeping old weights."""
    device = torch.device(device)
    old_vocab_size, n_embd = model.embed.weight.shape

    if old_vocab_size >= new_vocab_size:
        return model

    new_embed = nn.Embedding(new_vocab_size, n_embd, device=device)
    new_embed.weight.data[:old_vocab_size] = model.embed.weight.data
    new_embed.weight.data[old_vocab_size:] = (
        torch.randn(new_vocab_size - old_vocab_size, n_embd, device=device) * 0.02
    )
    model.embed = new_embed

    old_lm_head = model.lm_head.data
    new_lm_head = torch.zeros(new_vocab_size, n_embd, device=device)
    new_lm_head[:old_vocab_size] = old_lm_head
    new_lm_head[old_vocab_size:] = (
        torch.randn(new_vocab_size - old_vocab_size, n_embd, device=device) * 0.02
    )
    model.lm_head = nn.Parameter(new_lm_head)

    model.config.vocab_size = new_vocab_size
    return model


def _load_from_bf16_cache(cache_path: Path, device: torch.device) -> TalkieModel:
    state_dict = torch.load(cache_path, map_location=device)
    ckpt_vocab_size = state_dict["embed.weight"].shape[0]
    config = GPTConfig(vocab_size=ckpt_vocab_size)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            model = TalkieModel(config, device)
    finally:
        torch.set_default_dtype(prev_dtype)

    model.load_state_dict(state_dict, strict=True)
    model.device = device
    model.eval()

    return model


def _load_from_source(
    checkpoint_path: str, target_vocab_size: int | None
) -> TalkieModel:
    """Load fp32 checkpoint, resize if needed, convert to bf16 on CPU."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    ckpt_vocab_size = state_dict["embed.weight"].shape[0]
    config = GPTConfig(vocab_size=ckpt_vocab_size)

    # Build on CPU, load weights, convert to bfloat16, THEN move to GPU.
    cpu = torch.device("cpu")
    model = TalkieModel(config, cpu)
    model.load_state_dict(state_dict, strict=True)
    del ckpt, state_dict

    if target_vocab_size is not None and ckpt_vocab_size < target_vocab_size:
        model = resize_model_embeddings(model, target_vocab_size, cpu)

    return model.to(dtype=torch.bfloat16)


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    target_vocab_size: int | None = None,
) -> TalkieModel:
    """Load a Talkie model from a PyTorch checkpoint file.

    Handles ``torch.compile`` key prefixes and optional vocab resizing.

    The first load converts fp32 -> bf16 on CPU (to avoid a transient 2x GPU
    memory spike from fp32 init) and writes a ``<checkpoint>.bf16.pt`` sidecar.
    Subsequent loads read the sidecar directly onto *device*, skipping the CPU
    staging step. Delete the sidecar to force a re-conversion.
    """
    cache_path = Path(checkpoint_path + ".bf16.pt")
    if cache_path.exists():
        return _load_from_bf16_cache(cache_path, device)

    model = _load_from_source(checkpoint_path, target_vocab_size)
    torch.save(model.state_dict(), cache_path)

    model = model.to(device)
    model.device = device
    model.eval()
    return model
