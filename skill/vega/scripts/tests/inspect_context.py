"""
inspect_context.py — what `inspect_check.py` reads beside the live service: the batch the
service assembled for a graph explanation, the weights the bundle ships, and its reference
graph.

    batches = ServedBatches(bundle_dir)           the bundle's own `predict.load(...).prepare`
    batches.context(rows, history, sample)        -> inspect_recompute.GraphContext of the
                                                     explained sample (graph bundles)
    StoredWeights.load(bundle_dir)                weights.pt's state_dict as float32 arrays
    stored.match(published) -> WeightsMatch       each published tensor paired, byte for
                                                  byte, with a state_dict entry (NI4, NI5)
    ReferenceGraph.load(bundle_dir)               graph/graph.json and its two parquet files
                                                  (EI11)

Only the Recipe and the assembler of the bundle run here (`Predictor.prepare`, R7: the one
raw → batch path the explanation itself took) — never its network: every number a check
compares with the service's comes from `GET /v1/network` or `POST /v1/explain`. The rows go
through the bundle's own reader (`read_rows`, the CLI's typed CSV reader), so the frame is the
one the service built from the same JSON rows.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from check_kit import CheckError, read_parquet
from inspect_recompute import GraphContext
from lanekit import contracts as C

__all__ = ["ReferenceGraph", "ServedBatches", "StoredWeights", "WeightsMatch"]


def _bundle_module(root: Path):
    """The bundle's `predict.py`, imported under a name of its own (it puts the bundle root
    on sys.path and imports the package it serves)."""
    spec = importlib.util.spec_from_file_location(f"vega_check_bundle_{abs(hash(str(root)))}",
                                                  root / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ServedBatches:
    """The batches the service assembles, rebuilt by the bundle's own Predictor.prepare."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._module = _bundle_module(self.root)
        self._predictor = self._module.load(self.root)

    def frame(self, rows: Sequence[Mapping] | None):
        """JSON rows as the raw frame: through a CSV and the bundle's typed reader."""
        import pandas as pd
        if not rows:
            return None
        text = pd.DataFrame(list(rows)).to_csv(index=False)
        return self._module.read_rows(io.StringIO(text), self.root)

    def batch(self, rows, history=None) -> dict:
        prepared = self._predictor.prepare(self.frame(rows), history=self.frame(history))
        return {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
                for k, v in prepared.batch.items()}

    def context(self, rows, history, sample: Mapping) -> GraphContext:
        """The explained sample's GraphContext: its target node (node level) or its graph's
        nodes (graph level), as `explanation.sample` names it."""
        b = self.batch(rows, history)
        graph = sample.get("graph") or {}
        level, k = graph.get("level"), int(sample.get("index") or 0)
        weight = b.get("edge_weight")
        common = dict(x_num=b.get("x_num"), x_cat=b.get("x_cat"),
                      edge_index=np.asarray(b["edge_index"], dtype=np.int64),
                      edge_weight=None if weight is None else np.asarray(weight, np.float64))
        if level == "node":
            targets = b.get("target_index")
            target = int(targets[k]) if targets is not None else k
            return GraphContext(level="node", target=target, **common)
        nodes = tuple(int(i) for i in np.flatnonzero(np.asarray(b["graph_index"]) == k))
        return GraphContext(level="graph", graph_nodes=nodes, **common)


# --------------------------------------------------------------------------- weights.pt
@dataclass(frozen=True)
class WeightsMatch:
    """Published tensors paired with state_dict entries by their float32 bytes."""
    pairs: Mapping            # tensor id -> state_dict key
    unmatched: tuple          # published tensor ids with no entry of the same bytes
    uncovered: tuple          # float state_dict keys no published tensor carries
    published_digest: str     # sha256 of the published tensors' bytes, in state_dict order
    stored_digest: str        # sha256 of the same entries' bytes, read from weights.pt
    params: int               # Σ numel of the stored entries a weight, bias or table pairs with


class StoredWeights:
    """weights.pt's float state_dict entries, as little-endian float32 bytes, in order."""

    def __init__(self, entries: Mapping[str, np.ndarray]):
        self.entries = dict(entries)

    @staticmethod
    def load(root: Path) -> "StoredWeights":
        import torch
        doc = torch.load(Path(root) / C.BUNDLE_WEIGHTS, map_location="cpu", weights_only=True)
        state = doc.get(C.WEIGHTS_STATE, doc) if isinstance(doc, Mapping) else doc
        if not isinstance(state, Mapping):
            raise CheckError(f"{C.BUNDLE_WEIGHTS} holds no state_dict")
        return StoredWeights({k: v.detach().cpu().numpy().astype("<f4") for k, v in state.items()
                              if hasattr(v, "is_floating_point") and v.is_floating_point()})

    def match(self, published: Mapping[str, np.ndarray], roles: Mapping[str, str]) -> WeightsMatch:
        order = {k: i for i, k in enumerate(self.entries)}
        free = dict(self.entries)
        pairs, unmatched = {}, []
        for tid, arr in published.items():
            key = self._find(free, np.asarray(arr, dtype="<f4"))
            if key is None:
                unmatched.append(tid)
                continue
            pairs[tid] = key
            free.pop(key)
        uncovered = tuple(k for k in free if self.entries[k].size)
        ranked = sorted(pairs.items(), key=lambda kv: order[kv[1]])
        pub = hashlib.sha256(b"".join(np.ascontiguousarray(published[t], dtype="<f4").tobytes()
                                      for t, _ in ranked)).hexdigest()
        stored = hashlib.sha256(b"".join(self._prefix(k, published[t]).tobytes()
                                         for t, k in ranked)).hexdigest()
        params = sum(int(self.entries[k].size) for t, k in ranked
                     if roles.get(t) in ("weight", "bias", "table"))
        return WeightsMatch(pairs, tuple(unmatched), uncovered, "sha256:" + pub,
                            "sha256:" + stored, params)

    def _find(self, free: Mapping[str, np.ndarray], arr: np.ndarray) -> str | None:
        """The first free entry with these bytes: the whole entry, or (a positional table
        published for the steps it serves) its leading rows."""
        for key, entry in free.items():
            if entry.size == arr.size and entry.reshape(-1).tobytes() == arr.reshape(-1).tobytes():
                return key
        for key, entry in free.items():
            flat = entry.reshape(-1)
            if entry.ndim >= 2 and entry.shape[-1] == arr.shape[-1] and arr.size < flat.size and \
                    flat[:arr.size].tobytes() == arr.reshape(-1).tobytes():
                return key
        return None

    def _prefix(self, key: str, published: np.ndarray) -> np.ndarray:
        size = int(np.asarray(published).size)
        return np.ascontiguousarray(self.entries[key].reshape(-1)[:size], dtype="<f4")


# --------------------------------------------------------------------------- the reference graph
@dataclass(frozen=True)
class ReferenceGraph:
    """graph/graph.json, reference-nodes.parquet and reference-edges.parquet of a graph
    bundle (`vega/graph-reference@1`)."""
    meta: Mapping
    nodes: Any                # pandas frame, one row per ref id
    edges: frozenset          # {(src_ref, dst_ref)}

    @staticmethod
    def load(root: Path) -> "ReferenceGraph":
        files = {k: Path(root) / v for k, v in C.GRAPH_REFERENCE_FILES.items()}
        meta = json.loads(files["meta"].read_text(encoding="utf-8"))
        nodes = read_parquet(files["nodes"])
        edges = read_parquet(files["edges"])
        src, dst = C.GRAPH_REFERENCE_EDGE_COLUMNS[:2]
        pairs = frozenset(zip(edges[src].astype(int), edges[dst].astype(int)))
        return ReferenceGraph(meta, nodes, pairs)

    @property
    def size(self) -> int:
        return int(self.meta.get(C.GRAPH_REFERENCE_META_KEYS["nodes"]) or len(self.nodes))

    def has_edge(self, src: int, dst: int) -> bool:
        if (src, dst) in self.edges:
            return True
        return not self.meta.get("directed", False) and (dst, src) in self.edges

    def key_columns(self) -> tuple | None:
        """(key, linked key) of a key_pair graph, else None."""
        params = self.meta.get(C.GRAPH_REFERENCE_META_KEYS["params"]) or {}
        cols = params.get("columns") or ()
        return tuple(cols[:2]) if self.meta.get("method") == "key_pair" and len(cols) >= 2 \
            else None

    def raw(self, ref: int, column: str):
        name = C.RAW_PREFIX + column
        if name not in self.nodes.columns:
            return None
        value = self.nodes[name].iloc[int(ref)]
        return None if value is None or (isinstance(value, float) and np.isnan(value)) \
            else str(value)


def keys_pair(a: Mapping, b: Mapping, columns: tuple) -> bool:
    """Two raw rows a key_pair graph links: one's linked key names the other's key."""
    key, link = columns

    def text(row, col):
        value = row.get(col)
        return None if value is None else str(value)
    return (text(a, link) is not None and text(a, link) == text(b, key)) or \
        (text(b, link) is not None and text(b, link) == text(a, key))

