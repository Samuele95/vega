"""
{{PACKAGE}}.{{MODULE_SERVING}}.graphs — the reference graph a graph bundle answers through
(support; used by the Predictor, through `samples.assembler_for`).

Rendered by /vega-pack from `backends/pytorch/templates/serving/package/graphs.py`: fix the
template and re-pack, never this copy.

`GraphContext.load(root, recipe_module, features)` reads the bundle's `vega/graph-reference@1`
files once (build plan R-A3, R-A4; bundle-contract §2.3) and holds them, never mutated:

  graph/graph.json               the graph's meta: method, level, directed, self_loops, the
                                 reference parts, the feature and `edges_for` columns, notes;
  graph/reference-nodes.parquet  one row per reference node (train ∪ valid, as shaped) by a
                                 dense `ref` id — no raw position ever ships;
  graph/reference-edges.parquet  `src_ref, dst_ref, weight`: every edge among them.

Before anything is read, each file is checked against its digest in the bundle's
`MANIFEST.json` (`files`), when the root carries one: a reference graph altered after pack
answers for another model, and is refused by name. At level node the context is the
reference graph (`GraphReference`, the domain value the assembler attaches queries to) and
the frame the Recipe's `edges_for` reads — `pos` (the `ref` id, opaque to the recipe) and
`edges_for_columns`, in `ref` order; `query_edges(df_raw)` is `edges_for(df_raw, that frame)`,
the query → reference links only. At level graph a query is a whole graph with no reference
to join, and `query_edges(df_raw)` is the Recipe's `edges_within(df_raw)`: the edges inside
each `graph_id` of the request.

Effective Java: EJ-17 (a frozen value, built once and shared by concurrent requests); EJ-1
(`load` is the one factory, and says what it checks); EJ-49 (a refusal names the file, the
column or the function).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import pandas as pd

from ..domain import FeatureContract, GraphLevel, GraphReference, NetSpec

__all__ = ["GraphContext", "REFERENCE_FILES", "REFERENCE_SCHEMA"]

REFERENCE_SCHEMA = "vega/graph-reference@1"
# the three files by what they hold (bundle-contract §2.3), relative to the bundle root
REFERENCE_FILES = MappingProxyType({"nodes": "graph/reference-nodes.parquet",
                                    "edges": "graph/reference-edges.parquet",
                                    "meta": "graph/graph.json"})
# the Recipe's entry point per level (the other level's raises, naming this one)
_ENTRY = MappingProxyType({GraphLevel.NODE: "edges_for", GraphLevel.GRAPH: "edges_within"})


def _parquet(path: Path) -> pd.DataFrame:
    """A reference file read by Arrow's own filesystem. pandas' default hands Arrow a Python
    file object, and an Arrow thread still touching Python while the interpreter finalizes
    aborts the process at exit ('terminate called without an active exception')."""
    from pyarrow import fs
    return pd.read_parquet(path, filesystem=fs.LocalFileSystem())


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _verified(root: Path) -> None:
    """Each reference file present, and equal to its digest in MANIFEST.files when the root
    is a bundle (a unit test's directory carries no manifest, and is read as it is)."""
    for rel in REFERENCE_FILES.values():
        if not (root / rel).is_file():
            raise FileNotFoundError(f"{rel} is not in {root}: a graph bundle answers through its "
                                    f"reference graph ({REFERENCE_SCHEMA}); re-pack the bundle")
    manifest = root / "MANIFEST.json"
    if not manifest.is_file():
        return
    pinned = json.loads(manifest.read_text(encoding="utf-8")).get("files") or {}
    for rel in REFERENCE_FILES.values():
        if rel not in pinned:
            raise ValueError(f"{rel} is not pinned in MANIFEST.files; a reference graph the "
                             "manifest does not pin is not this bundle's")
        got = _digest(root / rel)
        if got != pinned[rel]:
            raise ValueError(f"{rel} does not match its digest in MANIFEST.files (recorded "
                             f"{pinned[rel]}, on disk {got}): the reference graph was altered "
                             "after pack, and would answer for another model — re-pack")


def _contract(features) -> FeatureContract:
    if isinstance(features, NetSpec):
        return features.feature_contract
    if isinstance(features, FeatureContract):
        return features
    raise TypeError(f"features is a {type(features).__name__}; pass the spec's FeatureContract "
                    "(spec.feature_contract), the columns the reference nodes carry")


def _entry(recipe_module, level: GraphLevel) -> Callable:
    name = _ENTRY[level]
    fn = getattr(recipe_module, name, None)
    if not callable(fn):
        raise ValueError(f"the Recipe module has no {name}(): a graph bundle's transform is "
                         "emitted by dataset-shaper with its graph step (--transform-only), and "
                         f"at level {level.value} it serves the graph through {name} — re-emit it")
    return fn


def _query_frame(nodes: pd.DataFrame, meta: Mapping, recipe_module) -> pd.DataFrame:
    """What `edges_for` reads of the reference: `pos` (the dense `ref` id) and the columns
    graph.json names, in `ref` order — the train nodes first, as the fitted index was built."""
    columns = list(meta.get("edges_for_columns") or ())
    declared = getattr(recipe_module, "EDGES_FOR_COLUMNS", None)
    if declared is not None and list(declared) != columns:
        raise ValueError(f"graph/graph.json names the edges_for columns {columns}, and the "
                         f"Recipe module reads {list(declared)}: the reference graph was packed "
                         "for another transform — re-pack the bundle")
    absent = [c for c in ["ref", *columns] if c not in nodes.columns]
    if absent:
        raise ValueError(f"graph/reference-nodes.parquet lacks the column(s) {absent} that "
                         "graph/graph.json names")
    frame = nodes.sort_values("ref", kind="stable").reset_index(drop=True)
    return frame[["ref", *columns]].rename(columns={"ref": "pos"})


@dataclass(frozen=True, eq=False)
class GraphContext:
    """The reference graph of one graph bundle, loaded once (`predict.load_context`).

    `meta` is graph.json; `level` the graph's level; at level node `reference` is the domain's
    `GraphReference` (the shaped features of the train ∪ valid nodes in `ref` order and the
    edges among them) and `nodes` the frame `edges_for` reads, both None at level graph;
    `entry` is the Recipe's `edges_for` (node) or `edges_within` (graph). It holds arrays, so
    it compares by identity; nothing in it is written after `load`.
    """

    meta: Mapping
    level: GraphLevel
    reference: GraphReference | None
    nodes: pd.DataFrame | None
    entry: Callable

    @staticmethod
    def load(root, recipe_module, features) -> "GraphContext":
        """The bundle's reference graph under `root`, verified (digests), for the spec whose
        FeatureContract is `features`, served through `recipe_module`'s graph entry point."""
        root = Path(root)
        _verified(root)
        meta = json.loads((root / REFERENCE_FILES["meta"]).read_text(encoding="utf-8"))
        if meta.get("schema") != REFERENCE_SCHEMA:
            raise ValueError(f"graph/graph.json is {meta.get('schema')!r}, not "
                             f"{REFERENCE_SCHEMA}; re-pack the bundle with vega-pack v2.3")
        level = GraphLevel(meta.get("level") or GraphLevel.NODE.value)
        entry, contract = _entry(recipe_module, level), _contract(features)
        frozen = MappingProxyType(dict(meta))
        if level is GraphLevel.GRAPH:
            return GraphContext(meta=frozen, level=level, reference=None, nodes=None,
                                entry=entry)
        nodes = _parquet(root / REFERENCE_FILES["nodes"])
        edges = _parquet(root / REFERENCE_FILES["edges"])
        reference = GraphReference.from_frames(nodes, edges, contract)
        return GraphContext(meta=frozen, level=level, reference=reference,
                            nodes=_query_frame(nodes, meta, recipe_module), entry=entry)

    @property
    def n_reference(self) -> int:
        """The reference nodes a query attaches to (0 at level graph)."""
        return 0 if self.reference is None else self.reference.n_nodes

    @property
    def node_key(self) -> str | None:
        """The raw column that names a node: a two-column `key_pair`'s first column (a row's
        own key; the second names the node it links to). knn, spatial and a one-column
        key_pair (a value the rows it links share) name no node: None."""
        params = self.meta.get("params") or {}
        columns = list(params.get("columns") or ())
        method = params.get("method") or self.meta.get("method")
        return columns[0] if method == "key_pair" and len(columns) == 2 else None

    def reference_names(self, ids) -> dict:
        """{ref id: its `node_key` value} for the reference nodes among `ids`, read from the
        `raw:<key>` column the reference carries for `edges_for`; {} when no column names a
        node. A node's name is the data's own key, never its place in the assembled graph."""
        key = self.node_key
        column = None if key is None or self.nodes is None else next(
            (c for c in (f"raw:{key}", key) if c in self.nodes.columns), None)
        if column is None:
            return {}
        values = self.nodes[column].to_numpy()
        return {i: str(values[i]) for i in sorted({int(j) for j in ids})
                if 0 <= i < len(values) and not pd.isna(values[i])}

    def query_edges(self, df_raw) -> Any:
        """The request's edges, from its raw rows: at level node `edges_for`'s query →
        reference links (`query_pos, ref_pos, weight`), at level graph `edges_within`'s edges
        inside each graph (`src_pos, dst_pos, weight`). Raw rows in, as the recipe reads them."""
        if self.level is GraphLevel.GRAPH:
            return self.entry(df_raw)
        return self.entry(df_raw, self.nodes)
