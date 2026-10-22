"""
port_fixtures.py — a network seen through its Model port and nothing else (code-contract §2:
the serving and inspection code read the network through the port's `head` and `anatomy()`).

Not a test file. `PortOnly(network)` forwards the Model port's methods — the emitter's role
table, which is what a blueprint's `signature[]` declares and the rendered `ports.py` Protocol
holds — and has no other attribute: no `__call__`, no `training`, no `parameters()`, no
`__getattr__`. The Predictor and inspector fixtures hand one over in place of the network, so
every serving and inspection test is also the check that their code reaches the network only
through the port: a reach past it fails, at the line that tried, with an AttributeError or a
TypeError.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import render_templates as rt  # noqa: E402

__all__ = ["PortOnly", "model_port", "port_only"]


@functools.lru_cache(maxsize=None)
def model_port() -> tuple:
    """The Model port's methods, in the role table's order."""
    emit = rt.load_emit()
    return tuple(emit._declared(line).name for line in emit.ROLE_SIGNATURES["Model"])


class PortOnly:
    """A network behind its Model port: one forwarding method per port method, nothing else."""

    __slots__ = ("_network",)

    def __init__(self, network):
        object.__setattr__(self, "_network", network)

    def __setattr__(self, name, value):
        raise AttributeError(f"the Model port has no attribute to set ({name})")

    def __repr__(self):
        return f"PortOnly({type(object.__getattribute__(self, '_network')).__name__})"


def _forwarding(name):
    def method(self, *args, **kwargs):
        return getattr(object.__getattribute__(self, "_network"), name)(*args, **kwargs)
    method.__name__ = method.__qualname__ = name
    return method


for _name in model_port():
    setattr(PortOnly, _name, _forwarding(_name))


def port_only(model):
    """`model` behind `PortOnly` (a PortOnly as it is)."""
    return model if isinstance(model, PortOnly) else PortOnly(model)
