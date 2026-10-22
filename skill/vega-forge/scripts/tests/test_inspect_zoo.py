#!/usr/bin/env python3
"""
test_inspect_zoo.py — the inspection zoo: every admitted (family, sample shape, graph level,
task) cell of the catalog, built by the rendered package's own `build_model` (CR-58), explained
through `NetworkInspector` over a harness standing in for the Predictor, and held to assertions
(a)–(g) of D3 §16.1 (build plan I10, R-B5, R-H5; amendment A31).

    python3 test_inspect_zoo.py [--cases GLOB[,GLOB…]] [--dump DIR] [pytest args]
    (or: python3 -m pytest test_inspect_zoo.py -q; env VEGA_ZOO_CASES, VEGA_ZOO_DUMP)

**The matrix is derived, never typed.** Its cells are `catalog.json`'s admission table (the 127
cells of family-catalog §7, its three named variants among them) as B19's recipe table makes
them concrete (`matrix/make_matrix_fixtures.cells`, which refuses a table that drifted from the
catalog): each family's composition at the matrix's small sizes, the task's head, the family's
or the task's loss. The inputs are this file's own, seeded: six numeric features and one
nominal of cardinality 5 on every case; a window of L = 12 over two series (a forecast reads
H = 3 steps and the autoregressive channel); a node-level reference graph of 40 nodes and 120
random weighted edges, the query nodes linked to it by a key; graph-level sets of four graphs of
10 nodes and 30 random weighted edges each. The **extras** are what the catalog's text names
beside the recipes' compositions: the quantile reading of a regression and of a forecast
(`tasks.regression.losses_allowed`), the `attention(pool)` reducer of `gru` and `lstm`, the
`layernorm` inner of `mlp-residual` and a normalised `sage_conv` (A31's exceptions).

One package, `zoo_pkg`, is rendered once, as /vega-pack renders a bundle's: the Model side
(`domain.py`, `network.py`, `readout.py`, `assemble.py`, the spliced builder, blocks and losses,
over every catalog block type but `custom`) and the serving package with its `inspection/`
modules in the Predictor's module. Per case: the network built by the package's `build_model`
in eval mode; a regression-family head's target affine fitted from a synthetic train target
(A37); a svdd centre persisted by the package's own loss (R-B6); a **twin** deep-copied before
any hook (every independent reference below is computed on it, or on copies of it);
`MANIFEST.output` / `MANIFEST.sample` written by vega-pack's `output_record` / `sample_record`;
the **`_Harness`** — the inspector's `_Answers`: `prepare` through the package's
`SampleAssembler` over the shaped rows (the rows ARE shaped: no Recipe), `predict` through the
reading's `values.reader_for`, `predictions` its warnings apart, `info` the manifest — and the
inspector opened on example rows. Every request is parsed by `QueryRules.parse`.

  (a) network    `network("full")` validates network-view-2; every spec block id is a node; no
                 `opaque` node but `custom`; `params.total == model.param_count()[0]`; the
                 inline tensors decode to their shapes and to the live parameters (`tensor(id)`),
                 a dense or head node's weight and bias to the twin's module's.
  (b) chain      non-null exactly for `mlp-plain`, `linear` and `pca-floor` over rows whose
                 head is a W·x + b layer (the rank-consistent ordinal head is not one); where it
                 is, the explanation's chain recomputes W·post[l−1] + b = pre[l] from the
                 published weights.
  (c) explain    `open` is ready; every target kind the task allows explains (and every
                 selector value (e) reads), validates explanation-2, keeps head fidelity
                 ≤ 1e-6·max(1, |head|) on the CPU — the bundle contract's round-trip rule: a
                 regression-family head answers in raw units (A37), where the transformer
                 encoder's no-grad fast path and the captured path differ by a float32 step or
                 two — and the served field within the manifest's tolerance (pack's
                 `Tolerance.for_scores`).
  (d) recompute  every node check ≤ 1e-4·max(1, |out|), the report's worst the worst; recurrent
                 nodes: the unroll within 1e-5 of the module, the float64 oracle within 1e-5 of
                 the output; attention rows (attention, encoder layers, attention pool; GAT per
                 destination, its published α·w with the edge weight divided out, R-B3) sum to
                 1 within 1e-5; graph messages plus bias (the root, the MLP) reproduce the
                 twin's conv output at every target within 1e-4.
  (e) targets    every requested and every picked target is the test's own reading of the
                 twin's head: binary sign·z flips with the class (the pick: the class decided
                 at the threshold); logit, label and threshold targets are the head's columns
                 (the argmax class, the label of largest σ, the first threshold not passed);
                 ordinal `expected_rank = Σσ(z)`; `value` the head in raw units; quantiles
                 their columns; forecast step h is column h − 1 (horizon-major under the
                 quantile loss); the reconstruction error is the loss's per-sample value and
                 `feature_error` its terms (the pick: the largest); `svdd_distance =
                 ‖f(x) − c‖²` with the persisted centre.
  (f) IG         completeness gap ≤ 1e-2·max(1, |s(x) − s(x₀)|) at 64 steps, s(x) − s(x₀) as
                 computed apart; on A31's known exceptions (a LayerNorm fed the zero baseline:
                 the tokenizer, the transformer encoder, a residual LayerNorm; a normalised
                 sage conv) the gap is reported as it is, never hidden.
  (g) hygiene    after everything no parameter holds `.grad`, the hooks are the ones `open`
                 attached (none per request), the weights and buffers are the twin's, eval mode.

Beside the matrix: the case list is the admission table; every family is a case and every block
type a family composition names is explained; each case's blocks are its family's; the
exceptions are A31's and their gap is reported; C3/C4/C5/C7 hold on the rendered package; the
dump is one valid pair per case.

**Dump** (for the page's renderers, W10–W15): with `--dump DIR` every case writes
`DIR/<slug>/network.json` (the canonical network body, full weights), `explain.json` (the
default target's explanation, `full` detail, gradient × input, canonical JSON) and `case.json`
(what the case is), and `DIR/index.json` (`vega/inspect-zoo@1`) lists every case written.

Exit 0 when every test passes (pytest's codes otherwise); 2 when `--cases` selects nothing.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import functools
import importlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
PACKAGE_TEMPLATES = TEMPLATES / "serving" / "package"
PACK = SCRIPTS.parents[1] / "vega-pack" / "scripts"
for _p in (str(SCRIPTS), str(HERE), str(HERE / "matrix"), str(BACKEND), str(PACK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import block_oracles as O  # noqa: E402  (the one float64 oracle, R-H4)
import conformance  # noqa: E402
import make_matrix_fixtures as mm  # noqa: E402  (B19: the admission cells, their compositions)
import output_record as orec  # noqa: E402  (vega-pack: MANIFEST.output and MANIFEST.sample)
import render_templates as rt  # noqa: E402
from port_fixtures import port_only  # noqa: E402  (the Model port alone)
from vega_pack_core import Tolerance  # noqa: E402  (vega-pack: MANIFEST.roundtrip.tolerance)

CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
SCHEMAS = SCRIPTS.parent / "references" / "schemas"
NETWORK_SCHEMA = json.loads((SCHEMAS / "network-view-2.schema.json").read_text(encoding="utf-8"))
EXPLANATION_SCHEMA = json.loads((SCHEMAS / "explanation-2.schema.json")
                                .read_text(encoding="utf-8"))
NETWORK_VALIDATOR = Draft202012Validator(NETWORK_SCHEMA)
EXPLANATION_VALIDATOR = Draft202012Validator(EXPLANATION_SCHEMA)

PACKAGE = "zoo_pkg"
MODEL_VERSION = "sha256:" + "5a" * 32
DUMP_SCHEMA = "vega/inspect-zoo@1"
SOURCE = "user:test_inspect_zoo"
SEED = 7
HEAD_FIDELITY = 1e-6      # (c): the captured head against the plain forward, on the CPU, at
                          # the round-trip rule's scale: 1e-6·max(1, |head|) (see check_explain)
RECOMPUTE = 1e-4          # (d): a node check, and the graph messages, at 1e-4·max(1, |out|)
UNROLL = 1e-5             # (d): the recurrent unroll, the oracle, the attention rows
IG_STEPS = 64             # (f)
IG_GAP = 1e-2             # (f): the completeness bound, 1e-2·max(1, |s(x) − s(x₀)|)
TARGET_TOL = 1e-5         # (e): a target's value against the test's own reading of the head
RECIPES = mm.load_table()
LENGTH, HORIZON = int(RECIPES["window"]["length"]), int(RECIPES["window"]["horizon"])
QUANTILES = [0.1, 0.5, 0.9]

NUMERIC = tuple(f"x{i}" for i in range(6))
NOMINAL, CARDINALITY = "zone", 5
FEATURES = tuple([{"name": c, "dtype": "float32", "role": "numeric"} for c in NUMERIC] +
                 [{"name": NOMINAL, "dtype": "int64", "role": "nominal",
                   "cardinality": CARDINALITY, "unseen_category_id": CARDINALITY - 1}])
GRAPH_ID = "graph_id"
N_ROWS = 10
SERIES, N_STEPS = ("A", "B"), 16
N_REF, N_EDGES, KEYS, QUERY_KEYS = 40, 120, 8, (0, 2, 3, 5, 6, 7)
N_GRAPHS, GRAPH_NODES, GRAPH_EDGES = 4, 10, 30

RECURRENT = ("rnn", "gru", "lstm")
ATTENTIVE = ("attention", "encoder_layer", "attention_pool")
CONVS = ("gcn_conv", "sage_conv", "gat_conv", "gin_conv")
CHAIN_FAMILIES = ("mlp-plain", "linear", "pca-floor")


# ================================================================ the cases

@dataclass(frozen=True)
class ZooCase:
    """One case: an admission cell (B19's `Cell`), or an extra — a change of one cell's spec
    (`edit`) the catalog's text names, with the reason (`why`)."""

    id: str
    cell: object
    why: str | None = None
    edit: Callable | None = None

    @property
    def slug(self) -> str:
        return self.id.replace("/", "__").replace(":", "-")

    @property
    def shape_key(self) -> str:
        return self.cell.shape_tag


def _quantile(spec: dict) -> None:
    """The quantile loss: one head column per quantile (per step and quantile, horizon-major,
    for a forecast)."""
    spec["loss"] = {"name": "quantile", "params": {"quantiles": list(QUANTILES)},
                    "source": SOURCE, "rationale": "the quantile reading of the task",
                    "alternatives": []}
    spec["blocks"][-1]["params"]["out"] *= len(QUANTILES)


def _layernorm_inner(spec: dict) -> None:
    """`mlp-residual`'s inner `batchnorm|layernorm`: the layernorm side."""
    for block in spec["blocks"]:
        if block["type"] == "residual":
            block["params"]["inner"] = [
                {"type": "layernorm", "params": {}} if i["type"] == "batchnorm" else i
                for i in block["params"]["inner"]]


def _normalised_sage(spec: dict) -> None:
    for block in spec["blocks"]:
        if block["type"] == "sage_conv":
            block["params"]["normalize"] = True


def _attention_reducer(spec: dict) -> None:
    """`gru` / `lstm`: `attention(pool) or pool(mean)` — the attention side."""
    blocks = spec["blocks"]
    at = next(i for i, b in enumerate(blocks) if b["type"] == "pool")
    blocks[at] = {**blocks[at], "type": "attention", "params": {"heads": 2, "pool": True}}


# (extra id, the admission cell it changes, the change, why)
EXTRAS = (
    ("extra/mlp-plain/row/regression:quantile", "mlp-plain/row/regression/nominals", _quantile,
     "the quantile reading (tasks.regression.losses_allowed): a column per quantile"),
    ("extra/lstm-attention/window/forecast:quantile", "lstm-attention/window/forecast/stride-1",
     _quantile, "a forecast under the quantile loss: steps x quantiles, horizon-major"),
    ("extra/gru/window/forecast:attention-pool", "gru/window/forecast/stride-1",
     _attention_reducer, "the gru composition's attention(pool) reducer"),
    ("extra/lstm/window/sequence-regression:regression:attention-pool",
     "lstm/window/sequence-regression:regression/stride-1", _attention_reducer,
     "the lstm composition's attention(pool) reducer"),
    ("extra/mlp-residual/row/regression:layernorm", "mlp-residual/row/regression/nominals",
     _layernorm_inner, "the mlp-residual composition's layernorm inner (A31: the gap is "
                       "reported)"),
    ("extra/graphsage/graph-node/regression:normalize",
     "graphsage/graph-node/regression/inductive-full", _normalised_sage,
     "a normalised sage conv (A31: the gap is reported)"),
)


def all_cases() -> dict:
    """{id: ZooCase}: the admission cells in the recipe table's order, then the extras."""
    cells = mm.cells(RECIPES, CATALOG)
    out = {c.id: ZooCase(c.id, c) for c in cells}
    for extra, base, edit, why in EXTRAS:
        out[extra] = ZooCase(extra, out[base].cell, why, edit)
    return out


CASES = all_cases()


def selected(env=None) -> list:
    """The case ids `VEGA_ZOO_CASES` selects (a comma list of globs; default every case); a
    glob that matches no case id is refused."""
    how = (os.environ if env is None else env).get("VEGA_ZOO_CASES", "").strip()
    if not how:
        return list(CASES)
    picked = []
    for pattern in [p.strip() for p in how.split(",") if p.strip()]:
        hits = [i for i in CASES if fnmatch.fnmatchcase(i, pattern)]
        if not hits:
            raise ValueError(f"VEGA_ZOO_CASES {pattern!r} matches no zoo case (e.g. "
                             f"{next(iter(CASES))})")
        picked += [i for i in hits if i not in picked]
    return picked


CASE_IDS = selected()


# ================================================================ the specs

def _sample(cell) -> dict:
    if cell.shape == "row":
        return {"shape": "row"}
    over = cell.sample_overrides()
    if cell.shape == "window":
        sample = {"shape": "window", "length": LENGTH, "stride": 1, "series_by": "series",
                  "order_by": "ts", "context": "previous-part",
                  "autoregressive": cell.kind == "forecast"}
        if cell.kind == "forecast":
            sample["horizon"] = HORIZON
        sample.update(over)
        return sample
    level = cell.level
    sample = {"shape": "graph", "edges": "shaped/edges.parquet", "level": level,
              "graph_id": GRAPH_ID if level == "graph" else None, "directed": True,
              "self_loops": False, "protocol": "inductive",
              "batching": "graphs" if level == "graph" else "full", "neighbours": None}
    if over.get("protocol") == "transductive":
        sample.update(protocol="transductive",
                      exposure={"kind": "features", "rows": "test", "source": SOURCE})
    if over.get("batching") == "sampled":
        sample.update(batching="sampled", neighbours=list(over["neighbours"]))
    return sample


def spec_of(case: ZooCase) -> dict:
    """The case's netspec: B19's composition, task, loss and metrics of the cell over the
    zoo's features and sample, then the extra's change."""
    cell = case.cell
    task = mm.compose_task(cell)
    frame = mm.Frame(cell, tuple(copy.deepcopy(list(FEATURES))), {}, RECIPES, 1)
    spec = {"schema": "vega/netspec@1", "seed": SEED, "task": task,
            "input": {"manifest_digest": "sha256:" + "ab" * 32},
            "data_contract": {"input_dim": len(FEATURES),
                              "features": copy.deepcopy(list(FEATURES)), "sample": _sample(cell)},
            "family": {"name": cell.family, "source": SOURCE},
            "blocks": mm.compose_blocks(frame, task), "loss": mm.compose_loss(cell),
            "metrics": mm.compose_metrics(cell),
            "optimizer": {"name": "adamw", "params": {"lr": 0.001}},
            "training": {"max_epochs": 1, "batch_size": 8, "device": "cpu"},
            "init": {"scheme": "kaiming_uniform", "bias": "zeros"}}
    if case.edit is not None:
        case.edit(spec)
    return spec


def walk(blocks):
    """Every block of a spec, the nested `inner` ones included."""
    for block in blocks or ():
        yield block
        yield from walk((block.get("params") or {}).get("inner"))


def ig_exception(spec: dict) -> str | None:
    """A31's known exceptions, read off the spec: why 64 midpoints cannot resolve the
    integrated-gradient path from the zero baseline (None when they can)."""
    for block in walk(spec["blocks"]):
        kind, params = block["type"], block.get("params") or {}
        if kind in ("feature_tokenizer", "transformer_encoder"):
            return "an untrained transformer's LayerNorm is scale invariant at the zero baseline"
        if kind == "layernorm":
            return "a LayerNorm fed a zero-bias map is scale invariant at the zero baseline"
        if kind == "sage_conv" and params.get("normalize"):
            return "an L2-normalised sage conv is scale invariant at the zero baseline"
    return None


def chain_expected(spec: dict, case: ZooCase) -> bool:
    """(b): a dense chain exactly for mlp-plain, linear and pca-floor over rows, the
    rank-consistent ordinal head apart (it is no W·x + b layer)."""
    return (case.cell.family in CHAIN_FAMILIES and case.cell.shape == "row"
            and case.cell.effective != "ordinal")


# ================================================================ the package

_WORK = tempfile.TemporaryDirectory(prefix="inspect-zoo-")
WORK = Path(_WORK.name)


def wide_spec() -> dict:
    """A spec whose blocks splice every segment a zoo network is built from: every catalog
    block type, `custom` included. No family composes `custom` and a partial render copies no
    code_ref, but the builder dispatches only to the block types its package carries (T5r3), so
    the package a case with a custom block is built by (the page fixture matrix's case 34, which
    writes its own module under model/custom) must carry the segment."""
    spec = rt.reference_spec("graph")
    spec["blocks"] = [{"id": f"W{i}", "type": t, "params": {}}
                      for i, t in enumerate(CATALOG["blocks"])]
    return spec


class Package:
    """`zoo_pkg`, rendered once as a bundle carries it, imported by the names the zoo reads."""

    def __init__(self):
        self.pipeline = rt.reference_pipeline()
        self.pipeline["system"]["package"] = PACKAGE
        out, spec = WORK / "package", wide_spec()
        rt.render(TEMPLATES, out, self.pipeline, spec,
                  only=["domain.py", "network.py", "readout.py", "assemble.py"], splice=True)
        rt.render(PACKAGE_TEMPLATES, out, self.pipeline, spec, into="Predictor", recursive=True)
        self.root = out / PACKAGE
        sys.path.insert(0, str(out))
        mod = lambda m: importlib.import_module(f"{PACKAGE}.{m}")  # noqa: E731
        self.domain, self.network = mod("domain"), mod("model.network")
        self.assemble, self.losses = mod("model.assemble"), mod("training.losses")
        self.records, self.values = mod("serving.records"), mod("serving.values")
        self.samples = mod("serving.samples")
        self.inspection = mod("serving.inspection")
        self.query, self.targets = mod("serving.inspection.query"), mod(
            "serving.inspection.targets")
        self.codec = mod("serving.inspection.codec")

    def build(self, spec: dict):
        """The case's network through the package's own constructor (`build_model`, which
        seeds from the spec), in eval mode."""
        return self.network.build_model(self.domain.NetSpec.from_dict(spec)).eval()


@functools.lru_cache(maxsize=None)
def package() -> Package:
    return Package()


# ================================================================ the synthetic inputs

def _features(rng, n: int) -> dict:
    cols = {c: np.round(rng.standard_normal(n), 4) for c in NUMERIC}
    cols[NOMINAL] = rng.integers(0, CARDINALITY, n)
    return cols


def _random_edges(rng, n: int, m: int) -> np.ndarray:
    """(2, m): m distinct directed edges among n nodes, no self loop, in (src, dst) order."""
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    pick = sorted(rng.choice(len(pairs), size=m, replace=False).tolist())
    return np.asarray([pairs[p] for p in pick], dtype=np.int64).T


class RowWorld:
    """Ten shaped rows; a request explains one of them."""

    meta = None

    def __init__(self, spec):
        self.rows = pd.DataFrame(_features(np.random.default_rng(SEED), N_ROWS))

    def reference(self, pkg, spec):
        return None

    def edges_for(self, shaped):
        return None

    def example(self) -> tuple:
        return self.rows, None

    def request(self) -> dict:
        return {"row": self.rows.iloc[[3]].to_dict("records")[0]}


class WindowWorld:
    """Two series of 16 steps (the forecast's target beside the features); a request sends
    series A's last 4 rows with its first 12 as history and explains its third row."""

    meta = None

    def __init__(self, spec):
        rng = np.random.default_rng(SEED)
        n = N_STEPS * len(SERIES)
        frame = {"series": np.repeat(SERIES, N_STEPS),
                 "ts": np.tile(np.arange(N_STEPS), len(SERIES)), **_features(rng, n)}
        label = spec["task"].get("label")
        if spec["data_contract"]["sample"].get("autoregressive") and label:
            frame[label] = np.round(3.0 + 2.0 * rng.standard_normal(n), 4)
        self.rows = pd.DataFrame(frame)

    def reference(self, pkg, spec):
        return None

    def edges_for(self, shaped):
        return None

    def example(self) -> tuple:
        return self.rows, None

    def request(self) -> dict:
        a = self.rows[self.rows["series"] == "A"]
        return {"rows": a.iloc[12:], "history": a.iloc[:12], "at": 2}


class NodeWorld:
    """The reference graph (40 nodes, 120 random weighted edges, a key per node) and six query
    rows, each linked to the reference nodes of its key (edges_for, key-pair style)."""

    meta = {"n_nodes": N_REF, "n_edges": N_EDGES, "method": "key_pair", "directed": True,
            "self_loops": False}

    def __init__(self, spec):
        rng = np.random.default_rng(SEED)
        self.nodes = pd.DataFrame({"ref": np.arange(N_REF), "key": np.arange(N_REF) % KEYS,
                                   **_features(rng, N_REF)})
        ends = _random_edges(rng, N_REF, N_EDGES)
        self.edges = pd.DataFrame({"src_ref": ends[0], "dst_ref": ends[1],
                                   "weight": np.round(rng.uniform(0.5, 1.5, N_EDGES), 4)})
        self.rows = pd.DataFrame({"key": list(QUERY_KEYS), **_features(rng, len(QUERY_KEYS))})

    def reference(self, pkg, spec):
        return pkg.domain.GraphReference.from_frames(self.nodes, self.edges,
                                                     spec.feature_contract)

    def edges_for(self, shaped):
        keys = self.nodes["key"].to_numpy()
        links = [(i, int(ref), 0.75 + 0.25 * (int(ref) % 3))
                 for i, key in enumerate(shaped["key"].to_numpy())
                 for ref in np.flatnonzero(keys == int(key))]
        return pd.DataFrame(links, columns=["query_pos", "ref_pos", "weight"])

    def example(self) -> tuple:
        return self.rows, None

    def request(self) -> dict:
        return {"rows": self.rows, "at": 1}


class GraphWorld:
    """Four graphs of 10 nodes and 30 random weighted edges each; a request sends all four
    and explains graph g2."""

    meta = {"n_nodes": N_GRAPHS * GRAPH_NODES, "n_edges": N_GRAPHS * GRAPH_EDGES,
            "method": "key_pair", "directed": True, "self_loops": False}

    def __init__(self, spec):
        rng = np.random.default_rng(SEED)
        ids = [f"g{i}" for i in range(N_GRAPHS)]
        self.rows = pd.DataFrame({GRAPH_ID: np.repeat(ids, GRAPH_NODES),
                                  "node": np.tile(np.arange(GRAPH_NODES), N_GRAPHS),
                                  **_features(rng, N_GRAPHS * GRAPH_NODES)})
        self.within = {g: (_random_edges(rng, GRAPH_NODES, GRAPH_EDGES),
                           np.round(rng.uniform(0.5, 1.5, GRAPH_EDGES), 4)) for g in ids}

    def reference(self, pkg, spec):
        return None

    def edges_for(self, shaped):
        where = {(g, int(n)): i for i, (g, n) in enumerate(zip(shaped[GRAPH_ID], shaped["node"]))}
        edges = []
        for g in dict.fromkeys(shaped[GRAPH_ID]):
            ends, weights = self.within[g]
            edges += [(where[(g, int(a))], where[(g, int(b))], float(w))
                      for (a, b), w in zip(ends.T, weights)
                      if (g, int(a)) in where and (g, int(b)) in where]
        return pd.DataFrame(edges, columns=["src_pos", "dst_pos", "weight"])

    def example(self) -> tuple:
        return self.rows, None

    def request(self) -> dict:
        return {"rows": self.rows, "at": 2 * GRAPH_NODES + 5}


WORLDS = {"row": RowWorld, "window": WindowWorld, "graph-node": NodeWorld,
          "graph-graph": GraphWorld}


# ================================================================ the harness

@dataclass(frozen=True)
class _Answer:
    """One request row's answer as `Predictor.predictions` gives it: its warnings apart."""

    warnings: tuple


class _Harness:
    """The inspector's `_Answers` over the zoo's shaped rows (D3 §16.1 step 3): a Predictor
    without a Recipe. `prepare` is the package's Model-side `SampleAssembler` (rows; windows
    over their history; a node-level request attached to the reference graph through the
    world's links; whole graphs with their edges); `predict` runs the network once without
    gradients and reads its head through the reading's reader (`values.reader_for`), one
    answer row per request row in `MANIFEST.output.fields` order (an unanswered window row
    empty, `answered` false, its note in `warnings`); `predictions` each row's warnings apart;
    `info` the manifest."""

    NO_WINDOW = (f"insufficient_context: fewer than {LENGTH - 1} rows of its series precede "
                 "this row")

    def __init__(self, pkg, model, spec, manifest, assembler, world):
        records = pkg.records.ServingRecords.of(spec, manifest, {}, MODEL_VERSION)
        self._pkg, self._model, self._manifest = pkg, model, manifest
        self._assembler, self._world = assembler, world
        self._reader = pkg.values.reader_for(records)
        self._kinds = {f: records.output.field_kinds[f]["kind"] for f in records.output.fields}

    def info(self) -> dict:
        return copy.deepcopy(self._manifest)

    def prepare(self, df_raw, history=None):
        shaped = df_raw.reset_index(drop=True)
        got = self._assembler.assemble(shaped, history, self._world.edges_for(shaped))
        at = np.asarray(got.sample_of_row, dtype=np.int64)
        notes = tuple((self.NO_WINDOW,) if s < 0 else () for s in at)
        return self._pkg.samples.Prepared(got.batch, shaped, at, got.index, notes,
                                          len(got.index))

    def _answer(self, df_raw, history) -> tuple:
        prepared = self.prepare(df_raw, history)
        at = prepared.sample_of_row
        with torch.no_grad():
            out = self._model(prepared.batch)
            head = self._reader.head(out, prepared.batch, self._model)
        answers = self._reader.frame(head).reindex(at).reset_index(drop=True)
        own = self._reader.warnings(head)
        notes = [tuple(prepared.warnings[i]) + (tuple(own[s]) if s >= 0 else ())
                 for i, s in enumerate(at)]
        graphs = prepared.index.graphs or ()
        shape = {"group": [str(graphs[s]) if 0 <= s < len(graphs) else None for s in at],
                 "answered": at >= 0, "warnings": ["; ".join(n) for n in notes]}
        frame = pd.DataFrame({f: shape[k] if k in shape else answers[f].to_numpy()
                              for f, k in self._kinds.items()}, index=df_raw.index)
        return frame, notes

    def predict(self, df_raw, history=None) -> pd.DataFrame:
        return self._answer(df_raw, history)[0]

    def predictions(self, df_raw, history=None) -> tuple:
        return tuple(_Answer(n) for n in self._answer(df_raw, history)[1])


# ================================================================ one case, run

def hooks_on(model) -> int:
    """Every forward and forward-pre hook registered on the network's modules."""
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def _fit_affine(pkg, model, spec: dict):
    """A37: the regression family's head ends on a train-fitted affine — fitted here from a
    synthetic train target (mean 3, spread 2), as the Trainer fits it once."""
    task = pkg.domain.Task.from_spec(spec)
    if not pkg.domain.TargetAffine.applies(task):
        return None
    rng = np.random.default_rng(SEED + 1)
    width = HORIZON if task.effective.value == "forecast" else 1
    y = 3.0 + 2.0 * rng.standard_normal((64, width))
    affine = pkg.domain.TargetAffine.of(spec, y if width > 1 else y[:, 0],
                                        spec["blocks"][-1]["id"])
    assert model.fit_target_affine(affine), "the head carries no pending target affine"
    return affine


def _persist_centre(pkg, model, spec, batch) -> None:
    """R-B6: the package's svdd loss sets the centre once from a batch's embeddings (the
    forward in eval mode; the loss writes only while the network reads as training)."""
    loss = pkg.losses.build_loss(spec)
    with torch.no_grad():
        out = model(batch)
        model.train()
        try:
            loss(out, batch, model)
        finally:
            model.eval()
    assert bool(model.svdd_center_set), "the svdd loss did not persist the centre"


def _threshold(reading: str, labelled: bool, classes, scores) -> dict | None:
    """threshold.json as /vega-eval writes it, per reading."""
    if reading == "binary":
        return {"applies": True, "value": 0.5, "scale": "probability", "source": "valid-optimum"}
    if reading == "multilabel":
        names = [str(c) for c in classes]
        return {"applies": True, "value": 0.5, "scale": "probability", "source": "valid-optimum",
                "per_label": {names[0]: 0.3, names[1]: 0.6}}
    if reading in ("anomaly-reconstruction", "anomaly-svdd"):
        if not labelled:
            return {"applies": False}
        return {"applies": True, "value": float(np.median(scores)), "scale": "score",
                "source": "valid-optimum"}
    return None


def _inductive(spec: dict) -> dict:
    """The spec the harness assembles by: a transductive network is explained over the batch
    the inductive assembly gives (pack refuses to serve a transductive bundle, and the network
    is the same)."""
    sample = spec["data_contract"]["sample"]
    if sample.get("protocol") != "transductive":
        return spec
    out = copy.deepcopy(spec)
    out["data_contract"]["sample"].update(protocol="inductive", exposure=None)
    return out


def canonical(body) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=True).encode("utf-8")


class Run:
    """One case run end to end (see the module docstring): every explanation the assertions
    read is made here, in order, and an explanation that raises keeps its exception for the
    assertion that reads it."""

    def __init__(self, case: ZooCase):
        pkg = package()
        self.case, self.spec = case, spec_of(case)
        self.world = WORLDS[case.shape_key](self.spec)
        self.net_spec = pkg.domain.NetSpec.from_dict(self.spec)
        self.model = pkg.build(self.spec)
        self.affine = _fit_affine(pkg, self.model, self.spec)
        assembly = pkg.domain.NetSpec.from_dict(_inductive(self.spec))
        self.assembler = pkg.assemble.SampleAssembler(
            assembly, self.world.reference(pkg, assembly))
        rows, history = self.world.example()
        example = self.assembler.assemble(rows, history, self.world.edges_for(rows)).batch
        self.example = example
        self.reading = orec.resolve_reading(self.spec)
        if self.reading == "anomaly-svdd":
            _persist_centre(pkg, self.model, self.net_spec, example)
        self.twin = copy.deepcopy(self.model)            # before any hook: the references
        self.manifest = self._manifest(example)
        self.harness = _Harness(pkg, self.model, self.net_spec, self.manifest, self.assembler,
                                self.world)
        self.hooks_before = hooks_on(self.model)
        self.inspector = pkg.inspection.NetworkInspector(      # the network as its port
            port_only(self.model), self.net_spec, self.harness, self.manifest["output"],
            self.tolerance)
        self.inspector.open(pkg.query.ExplainQuery(
            rows.reset_index(drop=True), history, len(rows) - 1, detail="summary"))
        self.status = self.inspector.status()
        self.hooks_open = hooks_on(self.model)
        self.rules = pkg.query.QueryRules.of(self.net_spec)
        self.targets = pkg.targets.TaskTargets.of(self.net_spec, self.manifest["output"],
                                                  self.twin)
        self.bodies, self.errors = {}, {}
        if self.status["ready"]:
            self._explain_all()
        self.grads = [n for n, p in self.model.named_parameters() if p.grad is not None]
        self.hooks_after = hooks_on(self.model)
        if self.status["ready"] and os.environ.get("VEGA_ZOO_DUMP") and \
                not {"default", "network"} & set(self.errors):
            dump(self, Path(os.environ["VEGA_ZOO_DUMP"]))     # a failed case fails its tests

    # -- the records -------------------------------------------------------------------------
    def _manifest(self, example) -> dict:
        """MANIFEST.output, .sample and .roundtrip as pack writes them: the round-trip
        tolerance is pack's own rule over the example's scores (`Tolerance.for_scores`:
        1e-6·max(1, max|score|))."""
        task = self.spec["task"]
        with torch.no_grad():
            out = self.model(example)
            rows = self.model.readout.row_score(out, example, self.model)
        scores = out.double().numpy().ravel().tolist()
        if rows is not None:
            scores += rows.double().numpy().ravel().tolist()
        threshold = _threshold(self.reading, bool(task.get("label")), task.get("classes") or (),
                               None if rows is None else rows.numpy())
        tolerance = Tolerance.for_scores(scores).as_json()
        self.tolerance = float(tolerance["value"])
        return {"output": orec.output_record(self.spec, threshold, task.get("label")),
                "sample": orec.sample_record(self.spec, {"steps": []}, self.world.meta),
                "model_version": MODEL_VERSION, "roundtrip": {"tolerance": tolerance}}

    # -- the explanations --------------------------------------------------------------------
    def query(self, **fields):
        body = {**self.world.request(), **fields}
        return self.rules.parse(body)

    def _keep(self, name: str, **fields) -> None:
        try:
            self.bodies[name] = self.inspector.explain(self.query(**fields))
        except Exception as exc:  # kept for the assertion that reads it  # noqa: BLE001
            self.errors[name] = exc

    def _explain_all(self) -> None:
        self._keep("default", detail="full")
        for kind in self.targets.kinds():
            self._keep(f"kind:{kind}", target={"kind": kind}, detail="full")
        self._keep("exact", detail="exact")
        self._keep("ig", method="integrated_gradients", steps=IG_STEPS, detail="summary")
        for name, target in self.selections():
            self._keep(name, target=target, detail="summary")
        try:
            self.network_bytes = self.inspector.network_bytes("full")
        except Exception as exc:  # noqa: BLE001
            self.errors["network"] = exc

    def selections(self) -> list:
        """(name, target) of every selector value (e) reads: each class, label, threshold,
        quantile, step and feature the task's kinds take."""
        out, record = [], self.manifest["output"]
        classes = [str(c) for c in record.get("classes") or ()]
        for kind in self.targets.kinds():
            selector = self.targets_kinds()[kind][0]
            if selector == "class" or selector == "label":
                out += [(f"{kind}:{c}", {selector: c}) for c in classes]
            elif selector == "threshold":
                out += [(f"{kind}:{j}", {"threshold": j}) for j in range(len(classes) - 1)]
            elif selector == "quantile":
                out += [(f"{kind}:{i}", {"quantile": i}) for i in range(len(QUANTILES))]
            elif selector == "step":
                out += [(f"{kind}:{h}", {"step": h}) for h in range(1, HORIZON + 1)]
            elif selector == "feature":
                out += [(f"{kind}:{f['name']}", {"feature": f["name"]}) for f in FEATURES]
        return out

    @staticmethod
    def targets_kinds():
        return importlib.import_module(f"{PACKAGE}.serving.inspection.target_kinds").KINDS

    def body(self, name: str) -> dict:
        """The explanation `name`, or its exception raised here."""
        assert self.status["ready"], f"the inspector did not open: {self.status['why']}"
        if name in self.errors:
            raise self.errors[name]
        return self.bodies[name]

    @property
    def network(self) -> dict:
        assert self.status["ready"], f"the inspector did not open: {self.status['why']}"
        if "network" in self.errors:
            raise self.errors["network"]
        return json.loads(self.network_bytes)

    # -- the independent references ----------------------------------------------------------
    def plain(self, query):
        """(the twin's head over the query's prepared batch, that batch, the explained k)."""
        prepared = self.harness.prepare(query.rows, query.history)
        k = query.sample(prepared.index)
        with torch.no_grad():
            head = self.twin(prepared.batch)
        return head, prepared.batch, k

    @functools.cached_property
    def head_scale(self) -> float:
        """max(1, |head|) of the explained sample (the twin's plain forward)."""
        head, _, k = self.plain(self.query())
        return max(1.0, float(head[k].abs().max()))

    def stage(self, node: dict):
        """The twin's module of a network node (its `path` from the network)."""
        return self.twin.get_submodule(node["path"])


@functools.lru_cache(maxsize=2)
def run(case_id: str) -> Run:
    return Run(CASES[case_id])


# ================================================================ the dump

def dump(r: Run, directory: Path) -> Path:
    """`<directory>/<slug>/{network.json, explain.json, case.json}` and the index (see the
    module docstring)."""
    directory = Path(directory)
    where = directory / r.case.slug
    where.mkdir(parents=True, exist_ok=True)
    (where / "network.json").write_bytes(r.network_bytes)
    (where / "explain.json").write_bytes(canonical(r.body("default")))
    cell = r.case.cell
    record = {"id": r.case.id, "slug": r.case.slug, "family": cell.family, "shape": cell.shape,
              "level": cell.level, "task": cell.task, "effective": cell.effective,
              "variant": cell.variant, "extra": r.case.why, "reading": r.reading,
              "target": r.body("default")["target"]["kind"],
              "files": {"network": f"{r.case.slug}/network.json",
                        "explain": f"{r.case.slug}/explain.json"}}
    (where / "case.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    entries = sorted((json.loads(p.read_text(encoding="utf-8"))
                      for p in directory.glob("*/case.json")), key=lambda e: e["id"])
    (directory / "index.json").write_text(json.dumps(
        {"schema": DUMP_SCHEMA, "source": "vega-forge/scripts/tests/test_inspect_zoo.py",
         "cases": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return where


# ================================================================ helpers

def _valid(validator, value, where: str) -> None:
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
    assert not errors, f"{where}: " + "; ".join(f"{list(e.path)}: {e.message[:200]}"
                                                  for e in errors[:4])


def _decode(payload):
    return None if payload is None else package().codec.decode(payload).astype(np.float64)


def _published(r: Run, tensor_id: str):
    """A published tensor, fetched from the live parameters and decoded (float64)."""
    return _decode(r.inspector.tensor(tensor_id))


def _scale(value) -> float:
    a = np.abs(np.asarray(value, dtype=np.float64))
    return max(1.0, float(a.max()) if a.size else 0.0)


def _nodes(r: Run) -> dict:
    return {n["id"]: n for n in r.network["nodes"]}


def _capture_io(module, model, batch) -> tuple:
    """(args, output) of one module on one forward of `model` over `batch` (the test's own
    hook, removed at once; the twin only)."""
    seen = []
    handle = module.register_forward_hook(lambda _m, args, out: seen.append((args, out)))
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()
    return seen[-1]


def _targets_of(r: Run, body: dict, batch) -> list:
    """The nodes a conv is read at: the explained node, or the explained graph's nodes."""
    graph = body["sample"]["graph"]
    if graph["level"] == "node":
        return [int(graph["target_node"])]
    k = int(body["sample"]["index"])
    return torch.nonzero(batch["graph_index"] == k).flatten().tolist()


# ================================================================ (a)–(g), per case

def check_network(r: Run) -> None:
    """(a)"""
    body = r.network
    _valid(NETWORK_VALIDATOR, body, f"{r.case.id} network")
    ids = {n["id"] for n in body["nodes"]}
    missing = [b["id"] for b in r.spec["blocks"] if b["id"] not in ids]
    assert not missing, f"spec blocks {missing} are no node"
    opaque = [n["id"] for n in body["nodes"] if n["view"] == "opaque" and n["type"] != "custom"]
    assert not opaque, f"opaque nodes {opaque}"
    assert body["params"]["total"] == int(r.model.param_count()[0])
    assert body["model_version"] == MODEL_VERSION and body["weights"] == "full"
    for tid, payload in body["tensors"].items():
        if payload.get("encoding") not in ("json", "base64-le"):
            continue
        got = _decode(payload)
        assert list(got.shape) == list(payload["shape"]) and np.all(np.isfinite(got)), tid
        assert np.array_equal(got, _published(r, tid)), f"{tid} is not the live tensor"
    for node in body["nodes"]:
        if node["type"] not in ("dense", "head"):
            continue
        module = r.stage(node)
        linear = getattr(module, "linear", module)
        weight = linear.weight.detach().double().numpy()
        if node["params"].get("thresholds") is None:
            assert np.array_equal(_published(r, node["params"]["weight"]), weight), node["id"]
        if node["params"].get("bias") and getattr(linear, "bias", None) is not None:
            assert np.array_equal(_published(r, node["params"]["bias"]),
                                  linear.bias.detach().double().numpy()), node["id"]
    status = r.status
    assert status["ready"] and status["coverage"]["opaque"] == [], status


def check_chain(r: Run) -> None:
    """(b)"""
    chain, want = r.network["chain"], chain_expected(r.spec, r.case)
    assert (chain is not None) == want, f"chain {'missing' if want else 'present'}"
    body = r.body("default")
    assert (body["chain"] is not None) == want
    if not want:
        return
    prev = np.asarray(body["chain"]["inputs"], dtype=np.float64)
    for k, layer in enumerate(chain["layers"]):
        w = _published(r, layer["weight"])
        b = _published(r, layer["bias"]) if layer["bias"] else 0.0
        pre = np.asarray(body["chain"]["pre"][k], dtype=np.float64)
        assert np.max(np.abs(prev @ w.T + b - pre)) <= 1e-5 * _scale(pre), layer["node"]
        prev = np.asarray(body["chain"]["post"][k], dtype=np.float64)


def _fidelity(r: Run, body: dict, where: str) -> None:
    """The explanation reproduces the served answer: the captured head within 1e-6 of the
    plain forward at the head's own scale — the bundle contract's round-trip rule,
    1e-6·max(1, |head|): a regression-family head answers in raw units (A37), where the
    transformer encoder's no-grad fast path (the served forward) and the captured gradient
    path differ by a float32 step or two of the value — and the served field within the
    manifest's tolerance."""
    fid = body["fidelity"]
    assert fid["ok"] is True, where
    assert fid["head_max_abs_diff"] <= HEAD_FIDELITY * r.head_scale, (where, fid, r.head_scale)
    assert fid["tolerance"] == pytest.approx(r.tolerance, rel=1e-8), (where, fid)
    assert fid["served_max_abs_diff"] is None or fid["served_max_abs_diff"] <= r.tolerance, \
        (where, fid)


def check_explain(r: Run) -> None:
    """(c)"""
    status = r.status
    assert status["ready"], status["why"]
    assert status["fidelity"]["samples"] >= 1 and status["recompute"]["ok"] is True, status
    assert status["fidelity"]["max_abs_diff"] <= r.tolerance, status
    kinds = r.targets.kinds()
    assert kinds and r.targets.default_kind() in kinds
    for kind in kinds:
        body = r.body(f"kind:{kind}")
        _valid(EXPLANATION_VALIDATOR, body, f"{r.case.id} explain {kind}")
        assert body["target"]["kind"] == kind
        _fidelity(r, body, kind)
        json.dumps(body, allow_nan=False)
    for name in ("default", "exact", "ig"):
        _valid(EXPLANATION_VALIDATOR, r.body(name), f"{r.case.id} {name}")
        _fidelity(r, r.body(name), name)
    for name, _ in r.selections():
        _valid(EXPLANATION_VALIDATOR, r.body(name), f"{r.case.id} {name}")
        _fidelity(r, r.body(name), name)


def check_recompute(r: Run) -> None:
    """(d)"""
    body = r.body("exact")
    nodes = _nodes(r)
    values = {}
    for node_id, payload in body["nodes"].items():
        check = payload["recompute"]
        if check is None:
            continue
        out = _decode((payload.get("exact") or {}).get("out"))
        bound = RECOMPUTE * (_scale(out) if out is not None else 1.0)
        assert check <= bound, f"{node_id} ({nodes[node_id]['type']}): {check} > {bound}"
        values[node_id] = check
    report = body["recompute"]
    assert report["nodes"] == len(values)
    if values:
        assert report["worst"] == pytest.approx(max(values.values()))
        assert values[report["worst_node"]] == pytest.approx(report["worst"])
    query = r.query(detail="exact")
    _, batch, _ = r.plain(query)
    for node_id, payload in body["nodes"].items():
        node = nodes[node_id]
        if node["type"] in RECURRENT:
            _recurrent(r, node, payload)
        if node["type"] in ATTENTIVE:
            _attention_rows(node, payload)
        if node["type"] == "gat_conv":
            _gat_alpha(node, payload, batch)
        if node["type"] in CONVS and node["params"].get("aggregator") != "max":
            _messages(r, node, r.body("default")["nodes"][node_id], batch)


def _recurrent(r: Run, node: dict, payload: dict) -> None:
    unroll = payload["extra"]["unroll"]
    assert unroll["max_abs_diff"] <= UNROLL, (node["id"], unroll["max_abs_diff"])
    weights, params = O.weights_of(r.stage(node), node["type"])
    x = _decode(payload["exact"]["in"])[None]
    oracle = O.ORACLES[node["type"]](weights, x, params)[0]
    out = _decode(payload["exact"]["out"])
    assert np.max(np.abs(oracle - out)) <= UNROLL * _scale(out), node["id"]


def _attention_rows(node: dict, payload: dict) -> None:
    alpha = _decode(payload["exact"]["alpha"])
    assert np.all(alpha >= 0.0), node["id"]
    assert np.max(np.abs(alpha.sum(axis=-1) - 1.0)) <= UNROLL, node["id"]


def _gat_alpha(node: dict, payload: dict, batch) -> None:
    """GAT's published `alpha` is α·w (R-B3: the edge weight multiplies α after the softmax,
    unrenormalised; a self loop the block adds weighs 1): with w divided out, the softmax sums
    to 1 over each destination's in-edges, per head."""
    alpha, ends = _decode(payload["exact"]["alpha"]), _decode(payload["exact"]["edge_index"])
    assert alpha.shape == (ends.shape[1], node["params"]["heads"]), node["id"]
    index, weight = batch["edge_index"], batch.get("edge_weight")
    weights = {} if weight is None else {
        (int(a), int(b)): float(w) for (a, b), w in zip(index.t().tolist(), weight.tolist())}
    w = np.asarray([weights.get((int(a), int(b)), 1.0) for a, b in ends.T], dtype=np.float64)
    softmax = alpha / w[:, None]
    for i in np.unique(ends[1]):
        assert np.max(np.abs(softmax[ends[1] == i].sum(axis=0) - 1.0)) <= UNROLL, (node["id"], i)


def _messages(r: Run, node: dict, payload: dict, batch) -> None:
    """Graph messages plus bias (the root, the MLP) reproduce the twin's conv output at every
    target, from the published weights and the twin's own conv input."""
    args, out = _capture_io(r.stage(node), r.twin, batch)
    h, out = args[0].double().numpy(), out.double().numpy()
    msgs, params = payload["extra"]["messages"], node["params"]
    kind = node["type"]
    bias = _published(r, params["bias"]) if params.get("bias") else 0.0
    for t in _targets_of(r, r.body("default"), batch):
        mine = [m for m in msgs if m["dst"] == t]
        if kind == "gcn_conv":
            w = _published(r, params["weight"])
            got = sum((m["coef"] * (w @ h[m["src"]]) for m in mine), np.zeros(w.shape[0])) + bias
        elif kind == "sage_conv":
            wn = _published(r, params["neigh"])
            wr = _published(r, params["root"]) if params.get("root") else None
            got = sum(((wr @ h[m["src"]]) if m["root"] else m["coef"] * (wn @ h[m["src"]])
                       for m in mine), np.zeros(wn.shape[0])) + bias
            if params.get("normalize"):
                got = got / max(np.linalg.norm(got), 1e-12)
        elif kind == "gat_conv":
            w, heads = _published(r, params["weight"]), int(params["heads"])
            acc = np.zeros((heads, w.shape[0] // heads))
            for m in mine:
                acc += np.asarray(m["coef"])[:, None] * (w @ h[m["src"]]).reshape(heads, -1)
            got = (acc.reshape(-1) if params["concat"] else acc.mean(axis=0)) + bias
        else:
            agg = sum((m["coef"] * h[m["src"]] for m in mine), np.zeros(h.shape[1]))
            with torch.no_grad():
                got = r.stage(node).mlp(torch.from_numpy(agg).float()).double().numpy()
        assert mine or kind in ("gcn_conv", "gat_conv"), (node["id"], t)
        assert np.max(np.abs(got - out[t])) <= RECOMPUTE * _scale(out[t]), (node["id"], t)


@dataclass(frozen=True)
class Want:
    """What the test expects of one explained target, from its own reading of the head: the
    value, and — where the reading fixes them — the head column (`index`), the name and the
    sign."""

    value: float
    index: int | None = None
    name: str | None = None
    sign: int | None = None


def _slice(batch, k: int) -> dict:
    return {key: v[k:k + 1] if torch.is_tensor(v) else v for key, v in batch.items()}


def _feature_errors(out, batch) -> dict:
    """{feature: its term of the reconstruction error} for one sample: (x̂_f − x_f)² per
    numeric feature, the cross-entropy of the nominal's logits against its id."""
    errors = {name: float((out[0, j].double() - batch["x_num"][0, j].double()) ** 2)
              for j, name in enumerate(NUMERIC)}
    logits = out[0, len(NUMERIC):len(NUMERIC) + CARDINALITY].double()
    errors[NOMINAL] = float(torch.nn.functional.cross_entropy(
        logits[None], batch["x_cat"][0, :1].long()))
    return errors


def want_of(r: Run, kind: str, selection, head, batch, k: int) -> Want:
    """(e): the target the request selects — or the one the task picks when it selects none —
    read off the twin's head for sample k, in raw units (the affine applied). A pick: the
    class decided at the threshold (binary), the argmax class, the label of largest σ, the
    first threshold not passed (else the last), the quantile nearest 0.5, step 1, the feature
    of largest error."""
    sel = {key: v for key, v in (selection or {}).items() if key != "kind"}
    z = head[k].double()
    record = r.manifest["output"]
    classes = [str(c) for c in record.get("classes") or ()]
    q = len(QUANTILES) if r.spec["loss"]["name"] == "quantile" else 0
    median = QUANTILES.index(0.5)
    if kind == "binary_logit":
        decided = float(torch.sigmoid(z[0])) >= float(record["threshold"])
        name = sel.get("class") or classes[1 if decided else 0]
        sign = 1 if name == classes[1] else -1
        return Want(sign * float(z[0]), name=name, sign=sign)
    if kind in ("logit", "label_logit"):
        name = sel.get("class") or sel.get("label") or classes[int(z.argmax())]
        return Want(float(z[classes.index(name)]), classes.index(name), name)
    if kind == "threshold_logit":
        passed = [j for j in range(len(z)) if float(z[j]) < 0.0]
        j = sel.get("threshold", passed[0] if passed else len(z) - 1)
        return Want(float(z[j]), j)
    if kind == "expected_rank":
        return Want(float(torch.sigmoid(z).sum()))
    if kind in ("value", "quantile"):
        col = sel.get("quantile", median) if q else 0
        return Want(float(z[col]), col)
    if kind == "forecast_step":
        h = sel.get("step", 1)
        col = (h - 1) * q + median if q else h - 1
        return Want(float(z[col]), col)
    if kind == "svdd_distance":
        return Want(float(((z - r.twin.svdd_center.double()) ** 2).sum()))
    one = _slice(batch, k)
    if kind == "reconstruction_error":
        loss = package().losses.build_loss(r.net_spec)
        with torch.no_grad():
            return Want(float(getattr(loss, "rec", loss)(head[k:k + 1], one, r.twin)))
    errors = _feature_errors(head[k:k + 1], one)
    name = sel.get("feature") or max(errors, key=errors.get)
    return Want(errors[name], name=name)


def check_targets(r: Run) -> None:
    """(e)"""
    query = r.query()
    head, batch, k = r.plain(query)
    record = r.manifest["output"]
    asked = [("default", None)] + [(f"kind:{kind}", {"kind": kind})
                                   for kind in r.targets.kinds()] + r.selections()
    for name, selection in asked:
        target = r.body(name)["target"]
        if selection:
            assert target["kind"] == selection.get("kind", target["kind"]), (name, target)
        want = want_of(r, target["kind"], selection, head, batch, k)
        assert target["value"] == pytest.approx(want.value, rel=TARGET_TOL, abs=TARGET_TOL), \
            (name, target, want)
        for field in ("index", "name", "sign"):
            if getattr(want, field) is not None:
                assert target[field] == getattr(want, field), (name, field, target, want)
        if selection and set(selection) != {"kind"}:
            assert target["why"] == "requested", (name, target)
    z = head[k].double()
    if r.case.cell.effective == "binary":
        a, b = (r.body(f"binary_logit:{c}")["target"] for c in record["classes"])
        assert (a["sign"], b["sign"]) == (-1, 1) and a["value"] == pytest.approx(-b["value"])
        assert b["value"] == pytest.approx(float(z[0]), rel=TARGET_TOL, abs=TARGET_TOL)
        assert b["name"] == str(record["positive"])
    if r.case.cell.kind == "forecast":
        steps = [r.body(f"forecast_step:{h}")["target"] for h in range(1, HORIZON + 1)]
        q = len(QUANTILES) if r.spec["loss"]["name"] == "quantile" else 0
        cols = [(h - 1) * q + QUANTILES.index(0.5) if q else h - 1 for h in range(1, HORIZON + 1)]
        assert [t["index"] for t in steps] == cols
        assert [t["value"] for t in steps] == pytest.approx([float(z[c]) for c in cols],
                                                            rel=TARGET_TOL, abs=TARGET_TOL)
    if r.case.cell.effective == "ordinal":
        rank = r.body("kind:expected_rank")["target"]["value"]
        assert rank == pytest.approx(float(torch.sigmoid(z).sum()), rel=TARGET_TOL)


def _embedders(model):
    return [m for kind, m in zip(model.kinds, model.stages)
            if kind.value in ("emb", "tokens") and len(getattr(m, "embeddings", None) or ())]


def _s(r: Run, model, target: dict, batch, k: int) -> float:
    t = package().targets.ExplainTarget(target["kind"], target["index"], target["name"],
                                        target["sign"], target["why"])
    with torch.no_grad():
        return float(r.targets.scalar(t, model(batch), batch)[k])


def check_ig(r: Run) -> None:
    """(f)"""
    body = r.body("ig")
    comp = body["attribution"]["completeness"]
    assert comp is not None and set(comp) == {"sum", "target_minus_baseline", "gap"}
    query = r.query()
    _, batch, k = r.plain(query)
    target = body["target"]
    baseline = copy.deepcopy(r.twin)
    with torch.no_grad():
        for module in _embedders(baseline):
            for table in module.embeddings:
                table.weight.zero_()
    zero = dict(batch, x_num=torch.zeros_like(batch["x_num"]))
    delta = _s(r, r.twin, target, batch, k) - _s(r, baseline, target, zero, k)
    tmb = comp["target_minus_baseline"]
    assert abs(tmb - delta) <= 1e-4 * max(1.0, abs(delta)), (tmb, delta)
    reported = abs(comp["sum"] - tmb)
    assert abs(comp["gap"] - reported) <= 5e-9 * max(1.0, reported, abs(comp["sum"])), comp
    assert math.isfinite(comp["gap"]), comp
    if ig_exception(r.spec) is None:
        assert comp["gap"] <= IG_GAP * max(1.0, abs(delta)), (comp, delta)


def check_hygiene(r: Run) -> None:
    """(g): the hooks attached are exactly `HookSet.sites` of the network's stage graph (built
    here over a copy of the twin), once, at open; none per request."""
    assert r.status["ready"], r.status["why"]
    assert not r.grads, f"parameters with .grad: {r.grads[:5]}"
    copy_ = copy.deepcopy(r.twin)
    graph = importlib.import_module(f"{PACKAGE}.serving.inspection.graph").StageGraph.build(
        copy_, r.net_spec, (r.example, None), r.manifest["output"])
    sites = importlib.import_module(f"{PACKAGE}.serving.inspection.capture").HookSet.sites(
        copy_, graph)
    assert r.hooks_before == 0 and r.hooks_open == len(sites) > 0, \
        (r.hooks_before, r.hooks_open, len(sites))
    assert r.hooks_after == r.hooks_open == hooks_on(r.model), \
        (r.hooks_open, r.hooks_after, hooks_on(r.model))
    assert not [n for n, p in r.model.named_parameters() if p.grad is not None]
    assert hooks_on(r.twin) == 0
    assert not r.model.training
    twin = r.twin.state_dict()
    for name, value in r.model.state_dict().items():
        assert torch.equal(value, twin[name]), f"{name} was written"


ASSERTIONS = {"a": check_network, "b": check_chain, "c": check_explain, "d": check_recompute,
              "e": check_targets, "f": check_ig, "g": check_hygiene}


@pytest.mark.parametrize("case_id,letter", [(c, a) for c in CASE_IDS for a in ASSERTIONS],
                         ids=[f"{c}-({a})" for c in CASE_IDS for a in ASSERTIONS])
def test_the_zoo(case_id, letter):
    ASSERTIONS[letter](run(case_id))


# ================================================================ beside the matrix

def _admission_from_catalog() -> set:
    """{(family, shape, level, task, variant or None)} read off catalog.json alone."""
    out = set()
    for row in CATALOG["admission"]["rows"]:
        out |= {(f, row["shape"], row.get("level"), t, None)
                for f in row["families"] for t in row["tasks"]}
        out |= {(v["family"], row["shape"], row.get("level"), v["task"], v["variant"])
                for v in row.get("variants") or ()}
    return out


def test_the_cases_are_the_admission_table():
    admitted = {v["variant"]: v["admission"] for v in RECIPES["variants"]}
    cells = {(c.cell.family, c.cell.shape, c.cell.level, c.cell.task,
              admitted.get(c.cell.variant)) for c in CASES.values() if c.why is None}
    assert cells == _admission_from_catalog()
    assert len(cells) == CATALOG["admission"]["cells"] == \
        sum(1 for c in CASES.values() if c.why is None)
    assert {c.cell.family for c in CASES.values()} == set(CATALOG["families"])
    for level in ("node", "graph"):
        assert any(c.cell.level == level for c in CASES.values())
    assert any(c.cell.family == "rnn" for c in CASES.values())
    assert all(c.id.startswith("extra/") for c in CASES.values() if c.why is not None)


def test_each_case_holds_its_familys_blocks():
    for case in CASES.values():
        spec = spec_of(case)
        allowed = set(CATALOG["families"][case.cell.family]["block_types"])
        used = {b["type"] for b in walk(spec["blocks"])}
        assert used <= allowed, (case.id, sorted(used - allowed))
        assert spec["blocks"][-1]["type"] == "head", case.id


def test_every_block_type_a_family_names_is_explained():
    named = set().union(*(set(f["block_types"]) for f in CATALOG["families"].values()))
    used = set().union(*({b["type"] for b in walk(spec_of(c)["blocks"])}
                         for c in CASES.values()))
    assert named <= used, sorted(named - used)
    for kind in ("attention_pool", "rnn", "gcn_conv", "sage_conv", "gat_conv", "gin_conv",
                 "graph_pool", "edge_dropout"):
        assert kind in used


def test_the_exceptions_are_a31s():
    """The IG exceptions are exactly A31's: the untrained ft-transformer and transformer-seq,
    mlp-residual with a layernorm inner, a normalised sage conv — and no other case."""
    found = {c.id for c in CASES.values() if ig_exception(spec_of(c))}
    want = {c.id for c in CASES.values()
            if c.cell.family in ("ft-transformer", "transformer-seq")
            or c.edit in (_layernorm_inner, _normalised_sage)}
    assert found == want
    assert any("layernorm" in i for i in found) and any("normalize" in i for i in found)


EXCEPTIONS = [i for i in CASE_IDS if ig_exception(spec_of(CASES[i]))]


@pytest.mark.parametrize("case_id", EXCEPTIONS)
def test_the_gap_is_reported_as_it_is(case_id):
    """A31: where 64 midpoints cannot resolve the path, the explanation reports the gap it
    has — |Σa − (s(x) − s(x₀))| with both ends the network's own — never a hidden one."""
    r = run(case_id)
    comp = r.body("ig")["attribution"]["completeness"]
    assert comp is not None and comp["gap"] >= 0.0
    assert comp["sum"] == pytest.approx(r.body("ig")["attribution"]["total"])
    check_ig(r)


def test_the_rendered_package_holds_the_gate():
    """D3 §16.1 Conformance: C3, C4, C5 and C7 on zoo_pkg with its pipeline and map."""
    pkg = package()
    bmap = json.loads((pkg.root / "blueprint-map.json").read_text(encoding="utf-8"))
    for entry in bmap["entries"]:
        if entry.get("role") == "Predictor":
            entry["status"] = "implemented"
    mapped = {e["file"] for e in bmap["entries"]}
    inspection = sorted(p.name for p in (PACKAGE_TEMPLATES / "inspection").glob("*.py"))
    assert {f"serving/inspection/{n}" for n in inspection} <= mapped
    markers, _ = conformance.check_report(pkg.root, bmap, pkg.pipeline)
    wrong = [m for m in markers if m.rule in ("C3", "C4", "C5", "C7") and
             m.level.value == "ERROR"]
    assert not wrong, wrong


DUMPED = [i for i in ("mlp-plain/row/binary/nominals", "gru/window/forecast/stride-1",
                      "gcn/graph-node/multiclass/inductive-full",
                      "gin/graph-graph/regression/graphs") if i in CASE_IDS]


@pytest.mark.skipif(not DUMPED, reason="the cases this test dumps are not selected")
def test_the_dump_is_one_valid_pair_per_case(tmp_path):
    for case_id in DUMPED:
        where = dump(run(case_id), tmp_path)
        network = json.loads((where / "network.json").read_text(encoding="utf-8"))
        explain = json.loads((where / "explain.json").read_text(encoding="utf-8"))
        _valid(NETWORK_VALIDATOR, network, f"{case_id} dumped network")
        _valid(EXPLANATION_VALIDATOR, explain, f"{case_id} dumped explanation")
        assert (where / "network.json").read_bytes() == run(case_id).network_bytes
        record = json.loads((where / "case.json").read_text(encoding="utf-8"))
        assert record["id"] == case_id and record["target"] == explain["target"]["kind"]
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["schema"] == DUMP_SCHEMA
    assert [e["id"] for e in index["cases"]] == sorted(DUMPED)
    for entry in index["cases"]:
        assert (tmp_path / entry["files"]["network"]).is_file()
        assert (tmp_path / entry["files"]["explain"]).is_file()


# ================================================================ the command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the inspection zoo (build plan I10, D3 §16.1)")
    ap.add_argument("--cases", help="a comma list of globs over the case ids (default: all)")
    ap.add_argument("--dump", help="write one network.json / explain.json pair per case under "
                                   "this directory, and index.json")
    args, rest = ap.parse_known_args(argv)
    if args.cases:
        os.environ["VEGA_ZOO_CASES"] = args.cases
        try:
            selected()
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    if args.dump:
        os.environ["VEGA_ZOO_DUMP"] = str(Path(args.dump).resolve())
    if "OMP_NUM_THREADS" not in os.environ:
        torch.set_num_threads(2)      # the zoo's networks are tiny: more threads only contend
    return pytest.main([__file__, "-q", "-p", "no:cacheprovider", *rest])


if __name__ == "__main__":
    sys.exit(main())
