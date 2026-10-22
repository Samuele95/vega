"""
frozen_json — JSON made hashable and immutable: `freeze()` turns an object into a `FrozenJson`
and a list into a tuple, and `thaw()` is its exact inverse (EJ-17). `vega_show_core` re-exports
all three; this module imports nothing of the stage.

A `FrozenJson` is the pairs of a JSON object, in order. Iterating it yields the `(key, value)`
pairs, so `dict(frozen)` and `for k, v in frozen` work as they do on a tuple of pairs. Its read
side is a mapping's, keyed by strings: `frozen["k"]`, `"k" in frozen`, `get`, `keys`, `values`,
`items` and `len`. It equals and hashes as a mapping does (vega-forge's `_FrozenMap`), so a list
of pairs is never mistaken for an object.
"""
from __future__ import annotations

from typing import Mapping

__all__ = ["FrozenJson", "freeze", "thaw"]


class FrozenJson(tuple):
    """A frozen JSON object: its `(key, value)` pairs in order, read as a mapping by key."""

    __slots__ = ()

    def __eq__(self, other):
        if isinstance(other, FrozenJson):
            return dict(self) == dict(other)
        return NotImplemented if not isinstance(other, tuple) else False

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    def __hash__(self):
        return hash(("FrozenJson", frozenset(self)))

    def __getitem__(self, key):  # a string is a key; an integer or a slice indexes the pairs
        if not isinstance(key, str):
            return tuple.__getitem__(self, key)
        for k, v in self:
            if k == key:
                return v
        raise KeyError(key)

    def __contains__(self, key) -> bool:  # a string asks for a key; anything else for a pair
        if isinstance(key, str):
            return any(k == key for k, _ in self)
        return tuple.__contains__(self, key)

    def get(self, key: str, default=None):
        return next((v for k, v in self if k == key), default)

    def keys(self) -> tuple:
        return tuple(k for k, _ in self)

    def values(self) -> tuple:
        return tuple(v for _, v in self)

    def items(self) -> tuple:
        return tuple(self)


def freeze(value):
    """A JSON value made hashable and immutable (idempotent); `thaw` is its exact inverse."""
    if isinstance(value, FrozenJson):
        return value
    if isinstance(value, Mapping):
        return FrozenJson((str(k), freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    if isinstance(value, FrozenJson):
        return {k: thaw(v) for k, v in value}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value
