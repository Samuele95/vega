"""
inspection.derived — the numbers the readers derive from a module's parameters, and the names
they give them (inspect-contract §5.3; build plan R-F8).

- `arr`, `norms`, `fro`: a tensor read as float64 numpy, its L2 norms along an axis, its
  Frobenius norm — reduced with `np.add.reduce`, never a BLAS call, so one network always gives
  the same bytes whatever the machine;
- `act_name`: the catalog name of an activation module (or of the function an encoder layer
  holds);
- `tensor_name`, `pca2`, `table_entry`: one embedding table of a payload's `tables` — the
  table registered as `<node>.table.<column>` (`table.@<k>` when the column name is not a clean
  id segment), its row norms, the rows projected on the table's two principal axes (each axis
  signed so its largest loading is positive; none beyond `PCA_MAX` rows) and the reserved
  unseen-category row.

`readers.py` re-exports every name, so the reader modules read them from one place.
"""
from __future__ import annotations

import math
import re
from types import MappingProxyType

import numpy as np
from torch import nn

from .codec import host, num, nums

__all__ = ["ACT_NAMES", "PCA_MAX", "SAFE_NAME", "act_name", "arr", "fro", "norms", "pca2",
           "table_entry", "tensor_name"]

PCA_MAX = 4096              # an embedding table's 2-D projection up to this many rows
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
ACT_NAMES = MappingProxyType({"Identity": "none", "ReLU": "relu", "GELU": "gelu",
                              "SiLU": "silu", "Tanh": "tanh", "Sigmoid": "sigmoid",
                              "LeakyReLU": "leaky_relu"})


def arr(t) -> np.ndarray:
    """A tensor's values as float64 numpy (read, never written)."""
    return np.asarray(host(t), dtype=np.float64)


def norms(t, axis) -> np.ndarray:
    """L2 norms of `t` along `axis`, reduced without BLAS."""
    a = arr(t)
    return np.sqrt(np.add.reduce(a * a, axis=axis))


def fro(t) -> float:
    """The Frobenius norm of a tensor (0 for none)."""
    if t is None:
        return 0.0
    a = arr(t).reshape(-1)
    return num(math.sqrt(float(np.add.reduce(a * a))))


def act_name(act) -> str:
    """The catalog name of an activation module (or of a function, the encoder layer's)."""
    if act is None:
        return "none"
    if isinstance(act, nn.Module):
        name = type(act).__name__
        return ACT_NAMES.get(name, name.lower())
    return str(getattr(act, "__name__", act))


def tensor_name(column, k: int) -> str:
    """`table.<column>` when the column name is a clean id segment, else `table.@<k>`."""
    return f"table.{column}" if SAFE_NAME.match(str(column)) else f"table.@{k}"


def pca2(w: np.ndarray):
    """The rows of a table projected on its two principal axes — the SVD of the centred table,
    through its dim × dim Gram matrix — each axis signed so its largest loading is positive;
    None beyond `PCA_MAX` rows."""
    if w.shape[0] > PCA_MAX or w.shape[0] == 0:
        return None
    x = w - np.add.reduce(w, axis=0) / w.shape[0]
    gram = np.stack([np.add.reduce(x * x[:, j:j + 1], axis=0) for j in range(x.shape[1])])
    _, vecs = np.linalg.eigh(gram)
    axes = np.zeros((2, w.shape[1]))
    top = vecs[:, ::-1].T[:2]
    axes[:len(top)] = top
    for a in axes:
        if a.size and a[int(np.argmax(np.abs(a)))] < 0:
            a *= -1.0
    return nums(np.add.reduce(x[:, None, :] * axes[None, :, :], axis=2))


def table_entry(site, column, weight, k: int) -> dict:
    """One embedding table of a payload's `tables` (`site` is the node's `readers.Site`)."""
    w = arr(weight)
    card, dim = int(w.shape[0]), int(w.shape[1])
    return {"column": str(column), "cardinality": card, "dim": dim,
            "table": site.tensor(weight, tensor_name(column, k), "table", ("row", "dim")),
            "row_norms": site.vector(norms(w, 1)),
            "pca2": pca2(w), "unseen_row": card - 1}
