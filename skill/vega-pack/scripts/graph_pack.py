"""
graph_pack — the reference graph a graph bundle answers through (`vega/graph-reference@1`).

    graph_pack.write(ctx) -> {"sample.reference": {"nodes", "edges", "files"}} or {}
    graph_pack.write_reference(root, sources, order=None) -> graph.json as written
    graph_pack.refuse_transductive(sample)                 PackExit(2)
    graph_pack.refuse_unserved(spec)                       PackExit(2), before a file is written
    graph_pack.ReferenceSources.of(campaign)

It runs before the predictor is loaded, because a graph Predictor reads the reference graph at
construction and the round trip runs through it; a row or window bundle has no graph and gets
no record. What it writes (bundle-contract §2.3, build plan R-A3, R-A4):

  graph/reference-nodes.parquet  one row per reference node — the train rows first, then the
      valid rows, each in `__split_raw__` order — by a dense `ref` id 0…R−1, with every
      FeatureContract column as the network reads it (numerics float32, nominal ids int64 with
      the embedding's unseen id where training puts it) and every column the Recipe's
      `edges_for` reads (graph.json `edges_for_columns`): an input column under
      `raw:<column>`, as the input held it (the shaper compares keys as text and measures raw
      coordinates), a fitted knn column as shaped, float32 (the space its index digest is of);
  graph/reference-edges.parquet  `src_ref, dst_ref, weight`: the edges part's rows with both
      ends in train ∪ valid, in stored order, mapped to `ref` ids (the reference graph the
      sealed test view reads); the weight a float, or null for an unweighted method;
  graph/graph.json  the meta (`graph_reference.meta_keys` of the catalog).

**No raw position ships**: train ∪ valid positions would reveal the test part's membership by
complement (the reason `fitted.json` ships without `__split_raw__`). Ids are dense, the files
carry no index, and no column holds a position.

Refused: a transductive protocol (`ERROR: transductive graph protocol: …`, exit 2: the answers
would depend on the fixed deployment graph); a spec whose graph disagrees with the shaped
manifest's (level, direction, self loops; exit 2); a shaped part or edges part that is not the
file the shaped manifest recorded (exit 2); an emitted transform that does not serve this
graph — no `edges_for` at level node, no `edges_within` at level graph, other `edges_for`
columns than the reference carries (exit 3, bundle-contract §4). The columns are compared as a
set: `edges_for` reads them by name, so an order the shaped manifest and the emitted module
spell differently is no defect; graph.json then lists them in the module's order, which the
Predictor checks at load. Above
`warn_reference_nodes_above` (1,000,000) reference nodes it prints `WARN:` with the count and
records it in graph.json `notes`; there is no cap.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from bundle_slice import transformer_file
from pack_context import PackExit, out_ok, out_warn
from vega_pack_core import CATALOG, digest_file, read_parquet

__all__ = ["FILES", "PARTS", "ReferenceSources", "refuse_transductive", "refuse_unserved",
           "write", "write_reference"]

TABLE = CATALOG["sample_shapes"]["graph"]
REFERENCE = CATALOG["graph_reference"]
FILES = dict(zip(("nodes", "edges", "meta"), TABLE["files"]))
PARTS = tuple(TABLE["reference_parts"])
WARN_ABOVE = int(TABLE["warn_reference_nodes_above"])
ENTRY = {"node": "edges_for", "graph": "edges_within"}


class ReferenceSources(NamedTuple):
    """What the reference is cut from: the shaped directory and its manifest, the split's raw
    positions (`fitted.json __split_raw__`), the raw input rows and the resolved spec."""

    shaped_dir: Path
    manifest: dict
    split: dict
    raw: object
    spec: dict

    @staticmethod
    def of(campaign) -> "ReferenceSources":
        return ReferenceSources(Path(campaign.shaped_dir), campaign.shaped_manifest or {},
                                (campaign.fitted or {}).get("__split_raw__") or {},
                                campaign.raw, campaign.spec or {})


# --------------------------------------------------------------------------- the phase

def write(ctx) -> dict:
    """The reference graph of a graph bundle, and the counts `MANIFEST.sample.reference` says."""
    sample = _sample(ctx.spec)
    if (sample.get("shape") or "row") != "graph":
        return {}
    refuse_transductive(sample)
    sources = ReferenceSources.of(ctx.campaign)
    graph = _graph_block(sources)
    order = _check_transform(transformer_file(ctx.paths.pkg), graph)
    meta = write_reference(ctx.paths.root, sources, order)
    _report(meta, sources, ctx.markers)
    return {"sample.reference": {"nodes": meta["n_nodes"], "edges": meta["n_edges"],
                                 "files": dict(FILES)}}


def refuse_unserved(spec) -> None:
    """The protocol refusal, for `pack.py` to make before it writes anything under
    --bundle-dir: a transductive campaign leaves no partial bundle behind. `write` refuses
    again on its own; the earlier refusal is the one a pack run meets."""
    sample = _sample(spec)
    if (sample.get("shape") or "row") == "graph":
        refuse_transductive(sample)


def refuse_transductive(sample) -> None:
    """Only the inductive protocol is served (R-A10): the catalog's refusal, exit 2."""
    protocol = (sample or {}).get("protocol") or "inductive"
    if protocol in TABLE["protocols_served"]:
        return
    stated = TABLE["protocols_refused"].get(protocol) or ""
    match = re.fullmatch(r"ERROR: (.*) \(exit 2\)", stated)
    raise PackExit(2, match.group(1) if match else
                   f"graph protocol {protocol!r}: /vega-pack v2.3 serves inductive graphs only")


def write_reference(root, sources: ReferenceSources, order=None) -> dict:
    """graph/reference-nodes.parquet, graph/reference-edges.parquet and graph/graph.json under
    `root`; returns the meta written. `order` is the emitted transform's `EDGES_FOR_COLUMNS`
    (the same columns as the shaped manifest's, `_check_transform` proved it): the reference
    carries them, and graph.json lists them, in the order the transform reads them."""
    graph = _in_order(_graph_block(sources), order)
    parts = [_part(sources, name) for name in PARTS]
    nodes = _nodes(sources, graph, parts)
    edges = _edges(sources, np.concatenate([pos for _, pos in parts]))
    meta = _meta(sources, graph, len(nodes), len(edges))
    target = Path(root) / "graph"
    target.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(Path(root) / FILES["nodes"], index=False)
    edges.to_parquet(Path(root) / FILES["edges"], index=False)
    (Path(root) / FILES["meta"]).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- the inputs

def _sample(spec) -> dict:
    return ((spec or {}).get("data_contract") or {}).get("sample") or {}


def _graph_block(sources) -> dict:
    """The shaped manifest's `graph` block, agreeing with the spec's sample."""
    graph = sources.manifest.get("graph")
    if not isinstance(graph, dict):
        raise PackExit(2, "the spec's sample is a graph, and the shaped manifest has no graph "
                          "block: the shaper's graph_edges step writes it and the edges part — "
                          "re-run /dataset-shaper with that step")
    sample = _sample(sources.spec)
    stated = {"level": sample.get("level") or "node", "directed": sample.get("directed"),
              "self_loops": sample.get("self_loops")}
    for key, value in stated.items():
        if value is not None and value != graph.get(key):
            raise PackExit(2, f"the spec's sample says {key} {value!r} and the shaped "
                              f"manifest's graph says {graph.get(key)!r}: the network was not "
                              "built for the graph the shaper wrote")
    return graph


def _read(sources, name: str) -> pd.DataFrame:
    """A shaped output named in the manifest (a part, the edges), as the manifest recorded it."""
    entry = ((sources.manifest.get("outputs") or {}).get("files") or {}).get(name) or {}
    recorded = Path(entry.get("path") or f"{name}.parquet")
    path = recorded if recorded.is_file() else sources.shaped_dir / recorded.name
    if not path.is_file():
        raise PackExit(2, f"the shaped {name} file is not there ({recorded}, nor {path})")
    if entry.get("digest") and digest_file(path) != entry["digest"]:
        raise PackExit(2, f"the shaped {name} file {path.name} is not the file the shaped "
                          "manifest recorded (digest); the reference would carry rows the "
                          "network was not trained on")
    return read_parquet(path) if path.suffix in (".parquet", ".pq") else \
        pd.read_csv(path, low_memory=False)


def _part(sources, name: str) -> tuple:
    """(the shaped part, its rows' raw positions), row for row."""
    frame = _read(sources, name).reset_index(drop=True)
    positions = np.asarray(sources.split.get(name) or [], dtype=np.int64)
    if len(positions) != len(frame):
        raise PackExit(2, f"fitted.json __split_raw__ lists {len(positions)} {name} row(s) and "
                          f"the shaped {name} part holds {len(frame)}: a node's identity is its "
                          "raw position, and these two disagree")
    return frame, positions


# --------------------------------------------------------------------------- the files

def _features(spec) -> list:
    return [dict(f) for f in ((spec.get("data_contract") or {}).get("features") or [])]


def _feature_values(values, feature):
    """A FeatureContract column as the network reads it (the ingestion's tensor rule)."""
    role = feature.get("role")
    if role == "key":
        return values.to_numpy()
    if role != "nominal":
        return values.to_numpy(dtype=np.float32, na_value=np.nan)
    ids = values.astype("float64").fillna(-1).to_numpy(copy=True)
    card, unseen = int(feature.get("cardinality") or 0), feature.get("unseen_category_id")
    ids[(ids < 0) | (ids >= card)] = max(card - 1, 0) if unseen is None else int(unseen)
    return ids.astype(np.int64)


def _storable(values):
    """An input column as parquet can hold it: as the input held it, or — when it mixes a
    number with a text, which parquet cannot store — as text, which is how the graph step
    compares keys anyway (str of each value; a missing value stays missing)."""
    arr = np.asarray(values)
    if arr.dtype.kind != "O":
        return arr
    kinds = {type(v) for v in arr if not pd.isna(v)}
    if len(kinds) <= 1:
        return arr
    return np.array([None if pd.isna(v) else str(v) for v in arr], dtype=object)


def _edges_for(graph) -> tuple:
    block = graph.get("edges_for") or {}
    return list(block.get("columns") or []), dict(block.get("raw") or {})


def _in_order(graph, order) -> dict:
    """The graph block with its edges_for columns in `order` — the same columns, never others."""
    wanted, _ = _edges_for(graph)
    if order is None or list(order) == wanted:
        return graph
    if sorted(order) != sorted(wanted):
        raise PackExit(3, f"the edges_for columns {list(order)} are not the shaped manifest's "
                          f"{wanted}")
    block = dict(graph.get("edges_for") or {}, columns=list(order))
    return dict(graph, edges_for=block)


def _nodes(sources, graph, parts) -> pd.DataFrame:
    shaped = pd.concat([frame for frame, _ in parts], ignore_index=True)
    positions = np.concatenate([pos for _, pos in parts])
    columns = {"ref": np.arange(len(shaped), dtype=np.int64)}
    for feature in _features(sources.spec):
        name = feature.get("name")
        if name not in shaped.columns:
            raise PackExit(2, f"the shaped parts lack the feature column {name!r} the spec's "
                              "FeatureContract names")
        columns[name] = _feature_values(shaped[name], feature)
    wanted, raw = _edges_for(graph)
    for name in (n for n in wanted if n not in columns):
        if name in raw:
            if raw[name] not in sources.raw.columns:
                raise PackExit(2, f"the input lacks the column {raw[name]!r} the graph step "
                                  f"reads ({name})")
            columns[name] = _storable(sources.raw[raw[name]].to_numpy()[positions])
        elif name in shaped.columns:
            columns[name] = shaped[name].to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            raise PackExit(2, f"edges_for reads {name!r}, which the shaped parts do not carry")
    return pd.DataFrame(columns)


def _edges(sources, positions) -> pd.DataFrame:
    """The edges among reference nodes, in the edges part's order, by `ref` id."""
    frame = _read(sources, "edges")
    absent = [c for c in ("src_pos", "dst_pos", "src_part", "dst_part") if c not in frame]
    if absent:
        raise PackExit(2, f"the edges part lacks {absent} (shape-contract §2)")
    inside = frame["src_part"].astype(str).isin(PARTS) & frame["dst_part"].astype(str).isin(PARTS)
    kept = frame[inside.to_numpy()].reset_index(drop=True)
    ref_of = pd.Series(np.arange(len(positions), dtype=np.int64), index=positions)
    ends = {}
    for end in ("src", "dst"):
        pos = kept[f"{end}_pos"].to_numpy(dtype=np.int64)
        known = np.isin(pos, positions)
        if not known.all():
            k = int(np.argmin(known))
            raise PackExit(2, f"edge {k}: {end}_pos names a {kept[f'{end}_part'].iloc[k]} row "
                              "that fitted.json __split_raw__ does not list; the edges part and "
                              "the split disagree")
        ends[end] = ref_of.loc[pos].to_numpy(dtype=np.int64)
    weight = (kept["weight"].to_numpy(dtype="float64", na_value=np.nan) if "weight" in kept
              else np.full(len(kept), np.nan))
    return pd.DataFrame({"src_ref": ends["src"], "dst_ref": ends["dst"],
                         "weight": pd.array(weight, dtype="Float64")})


def _meta(sources, graph, n_nodes: int, n_edges: int) -> dict:
    wanted, raw = _edges_for(graph)
    level = graph.get("level") or "node"
    notes = [f"the bundle carries the shaped features of {n_nodes} training/validation rows"
             + (" and the input columns its graph step reads" if raw else "")
             + (" — the inductive protocol answers each query row through them" if
                level == "node" else " — at level graph a query is a whole graph, answered "
                "from its own rows and edges") + "; no raw position ships"]
    if n_nodes > WARN_ABOVE:
        notes.append(f"{n_nodes} reference nodes, above {WARN_ABOVE}: every request is "
                     "answered over all of them, held in memory (there is no cap)")
    values = {"schema": REFERENCE["schema"], "method": graph.get("method"),
              "params": graph.get("params") or {}, "level": level,
              "directed": bool(graph.get("directed")), "self_loops": bool(graph.get("self_loops")),
              "symmetric": bool(graph.get("symmetric", not graph.get("directed"))),
              "protocol": _sample(sources.spec).get("protocol") or "inductive",
              "parts": list(PARTS), "n_nodes": int(n_nodes),
              "n_edges": int(n_edges),
              "feature_columns": [f.get("name") for f in _features(sources.spec)],
              "edges_for_columns": wanted, "notes": notes}
    return {key: values[key] for key in REFERENCE["meta_keys"]}


# --------------------------------------------------------------------------- the checks

def _literal(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                return None
    return None


def _check_transform(path: Path, graph: dict):
    """The emitted transform serves this graph: its GRAPH is of this level, it defines the
    level's entry point, and at level node it reads the columns the reference carries — the
    same set of names, each once (`edges_for` reads them by name, not by position). Returns
    the transform's own order of them (None at level graph), which the reference keeps."""
    level = graph.get("level") or "node"
    entry = ENTRY[level]
    shown = "/".join(Path(path).parts[-2:])       # <features module>/preprocess.py
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise PackExit(3, f"the bundle's {shown} cannot be read ({exc})") from exc
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    served = _literal(tree, "GRAPH")
    if entry not in defined or not isinstance(served, dict) or served.get("level") != level:
        raise PackExit(3, f"the bundle's {shown} serves no level-{level} graph "
                          f"through {entry}(): a graph bundle attaches each request to its graph "
                          "through it — re-emit the transform with dataset-shaper's "
                          "--transform-only from the recipe with its graph_edges step")
    if level != "node":
        return None
    wanted, _ = _edges_for(graph)
    declared = _literal(tree, "EDGES_FOR_COLUMNS")
    names = list(declared) if isinstance(declared, (list, tuple)) else None
    if (names is None or not all(isinstance(n, str) for n in names)
            or len(set(names)) != len(names) or set(names) != set(wanted)
            or len(set(wanted)) != len(wanted)):
        raise PackExit(3, f"the bundle's {shown} reads the edges_for columns "
                          f"{declared} and the shaped manifest names {wanted}: the reference "
                          "would not carry what the transform reads")
    return names


def _report(meta, sources, markers) -> None:
    sizes = ", ".join(f"{name} {len(sources.split.get(name) or [])}" for name in PARTS)
    out_ok(f"graph/ — {meta['schema']}: {meta['n_nodes']} reference node(s) ({sizes}) and "
           f"{meta['n_edges']} edge(s) among them, by dense ref id; no raw position ships"
           + (f"; edges_for reads {', '.join(meta['edges_for_columns'])}"
              if meta["edges_for_columns"] else ""), markers)
    if meta["n_nodes"] > WARN_ABOVE:
        out_warn(f"the reference graph holds {meta['n_nodes']} nodes, above {WARN_ABOVE}: every "
                 "request is answered over all of them, in memory — there is no cap; the count "
                 "is recorded in graph/graph.json notes", markers)
