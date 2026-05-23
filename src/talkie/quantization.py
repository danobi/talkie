"""Quantization helpers for Talkie models."""

import torch
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
    Int4PackingFormat,
)


def quantize_int4(model: torch.nn.Module, group_size: int = 128) -> torch.nn.Module:
    """In-place int4 weight-only quantization of every ``nn.Linear`` in *model*.

    The Talkie ``embed`` (``nn.Embedding``) and ``lm_head`` (raw ``nn.Parameter``
    used via ``F.linear``) are not ``nn.Linear`` modules, so they remain in
    bfloat16. Custom gain wrappers (``HeadGain``, ``WeightGain``, ``ActGain``)
    are also untouched.
    """
    # TILE_PACKED_TO_4D is the legacy tinygemm CUDA path. The default PLAIN
    # packing in torchao>=0.17 requires the non-public "mslk" package.
    quantize_(
        model,
        Int4WeightOnlyConfig(
            group_size=group_size,
            int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        ),
    )
    return model
