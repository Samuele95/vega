"""
A custom block: a learnable per-unit gain, shape-preserving (D6 §5.4).

    y = x * g        g has the block's input shape (batch axis excluded), initialised to ones

The blocks lane (L05) cites it as `code_ref: "custom/custom_gain.py"` with a `user:` source.
lanekit.fixture copies it under the campaign root; the emitter copies it into the package as
`<model>/custom/custom_gain.py` (a `kind: support` file), and the builder calls `build` with
the per-sample input shape and the block's params. At initialisation the block is the
identity, so a lane that ships it proves the custom path end to end without changing what the
network computes until it has learned to.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

__all__ = ["build"]


class _Gain(nn.Module):
    """x * g, with g one learnable scalar per unit of the per-sample shape."""

    def __init__(self, shape: Sequence[int]) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


def build(in_shape: Sequence[int], params: Mapping[str, object]) -> nn.Module:
    """The module for an input of per-sample shape `in_shape`; `params` must declare an
    `out_shape` equal to it (the block is shape-preserving)."""
    shape = tuple(int(d) for d in in_shape)
    if not shape or any(d < 1 for d in shape):
        raise ValueError(f"custom_gain needs a non-empty positive input shape, got {in_shape!r}")
    declared = params.get("out_shape")
    if declared is not None and tuple(int(d) for d in declared) != shape:
        raise ValueError(f"custom_gain is shape-preserving: out_shape {list(declared)} differs "
                         f"from the input shape {list(shape)}")
    return _Gain(shape)
