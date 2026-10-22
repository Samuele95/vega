#!/usr/bin/env python3
"""
make_matrix_fixtures.py — the backend coverage matrix's fixtures (BUILD-PLAN B19, D1 §17.2): four
raw tables (seed 7), shaped by the **real** `shape.py` into one shaped variant per question the
cells ask, and one netspec per cell that the compiler's core admits.

    python3 make_matrix_fixtures.py <work> [--cells all|smoke|<glob>[,<glob>…]] [--sanity]
                                    [--jobs N] [--json]

The cells are the catalog's admission table (`catalog.json` `admission`, family-catalog §7):
127 of them, (family, sample shape, graph level, effective task) plus the three named variants.
`recipes.json` beside this file is the table of D1 §2's recipes that makes each concrete — its
groups must produce exactly the admission table's cells (`cells()` refuses a table that drifted
from the catalog, naming the difference), its `families` are the compositions, its `smoke` list
is one cell per family and its `sanity` list the learning checks.

The raw tables (D1 §17.2), each deterministic:

  rows.csv    600 rows: six numeric columns, two nominals (5 and 12 categories), an identity
              column; labels `y_bin`, `y_multi` (4 classes), `y_tags` (delimited, 3 labels),
              `y_reg`, `y_ord` (5 levels) and `y_anom` (8 % of the rows pushed far from the
              bulk).
  seq.csv     6 series x 160 hourly steps of an AR(2) process on a season of 12 with an
              exogenous cycle and a nominal regime (`zone`); keys `series`, `ts`; the sequence
              labels `y_bin` (rising), `y_multi` (the third of the season), `y_tags`
              (delimited: hi, up, wet, calm), `y_reg`, `y_ord` (4 levels) and the forecast
              `target`; a time split.
  nodes.csv   600 nodes in cliques of 3-8 on a shared `customer_id`, each clique of one of four
              communities (one node in ten moved to another clique), a position per clique; the
              node's own features say its community weakly and its clique mates say it — the
              node label kinds; `key_pair` edges on the customer id, `knn` on the position.
  graphs.csv  120 graphs of 8-14 nodes, a chain, a ring or a branched tree, whose node
              features do not depend on the motif; `y_bin` (a ring), `y_multi` (the motif),
              `y_reg` (the leaves); `key_pair` bonds within `graph_id`; a group split.

Under `<work>/`:

  raw/<table>.csv                the raw rows
  shaped/<variant>/              `shape.py`'s outputs for one (table, label, edges), its
                                 `recipe.in.json`, a `stamp.json` (a variant is shaped again
                                 only when its recipe, its raw table or `shape.py` changed), and
                                 `pipeline.json` — the forge fixture's world model
                                 (`render_templates.reference_pipeline()`) drawn against this
                                 shaped manifest (the compiler's D1b holds per variant)
  cells/<slug>/netspec.json      the cell's spec (netspec-contract §1): the shaped output schema
                                 less the label and the keys as features, the sample, the
                                 family's composition, every choice with a resolving source (no
                                 model HTML: `shape:`, `blueprint:`, `vega:default` with its
                                 catalog row, and the matrix's own `user:matrix/…` stances — the
                                 sizes and budget of a coverage run, not a campaign's)
  cells/<slug>/fixture.json      what `run_matrix.py`'s worker reads (`vega/backend-matrix-cell@1`)
  sanity/<slug>/…                the same for the learning checks' fits (`--sanity`), at the
                                 check's epochs

Every spec is checked by `compile_spec.compile_spec` before it is written; a refusal is exit 2
naming the cell and the rule. Exit codes: 0 made; 2 the table, a shaping or a spec failed.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
SCRIPTS = TESTS.parent
SKILLS = SCRIPTS.parent.parent
SHAPER = SKILLS / "dataset-shaper" / "scripts" / "shape.py"
RECIPES = HERE / "recipes.json"
for _p in (str(TESTS), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compile_spec as cs  # noqa: E402
import render_templates as rt  # noqa: E402

CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
CELL_SCHEMA = "vega/backend-matrix-cell@1"
PACKAGE = "matrix_pipeline"
LABEL_SOURCE = "shape:manifest/label"
GOAL_SOURCE = "user:matrix/goal"
UNDERFITS = "if the run underfits"
_WIDTH_KEYS = ("out", "units", "hidden", "channels", "d", "z", "ff")


class MatrixError(RuntimeError):
    """The table, a fixture or a spec could not be made; the message names which and why."""


# ================================================================ the table and its cells

def load_table(path=RECIPES) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Cell:
    """One cell of the matrix: (family, shape, level, task) under a variant, and the group and
    raw table it comes from. `task` is the admission table's spelling (`anomaly/labelled`,
    `sequence-classification/binary`); `id` is `family/shape-tag/task-tag/variant` with the
    task's `/` written `:` (`gru/window/sequence-classification:binary/stride-2`)."""

    group: str
    family: str
    shape: str
    level: str | None
    task: str
    variant: str
    data: str
    sample: tuple = ()          # the variant's sample overrides, as sorted (key, value) pairs
    batch_size: object = None   # the variant's batch size, when it sets one

    @property
    def shape_tag(self) -> str:
        return f"graph-{self.level}" if self.shape == "graph" else self.shape

    @property
    def id(self) -> str:
        return "/".join((self.family, self.shape_tag, self.task.replace("/", ":"), self.variant))

    @property
    def slug(self) -> str:
        return self.id.replace("/", "__").replace(":", "-")

    @property
    def kind(self) -> str:
        return self.task.split("/")[0]

    @property
    def base(self) -> str | None:
        head, _, tail = self.task.partition("/")
        return tail if head.startswith("sequence-") else None

    @property
    def effective(self) -> str:
        return self.base or self.kind

    @property
    def labelled(self) -> bool:
        return self.task not in ("unsupervised", "anomaly/unlabelled")

    def sample_overrides(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.sample}


def _freeze(mapping) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                        for k, v in (mapping or {}).items()))


def admission_cells(catalog=CATALOG) -> set:
    """{(family, shape, level, task, admission variant or None)} — the catalog's closed list."""
    out = set()
    for row in catalog["admission"]["rows"]:
        for family in row["families"]:
            for task in row["tasks"]:
                out.add((family, row["shape"], row.get("level"), task, None))
        for v in row.get("variants") or ():
            out.add((v["family"], row["shape"], row.get("level"), v["task"], v["variant"]))
    return out


def cells(table=None, catalog=CATALOG) -> list:
    """Every cell of the table, in table order; refused unless the set is the admission
    table's exactly (the matrix's cell list IS the catalog's, D1 §2)."""
    table = table or load_table()
    out, seen = [], set()
    for g in table["groups"]:
        for family in g["families"]:
            for task in g["tasks"]:
                out.append(Cell(g["id"], family, g["shape"], g.get("level"), task, g["variant"],
                                g["data"]))
                seen.add((family, g["shape"], g.get("level"), task, None))
    groups = {(c.family, c.shape, c.level, c.task): c for c in out}
    for v in table.get("variants") or ():
        base = groups.get((v["family"], v["shape"], v.get("level"), v["task"]))
        if base is None:
            raise MatrixError(f"recipes.json variant {v['variant']!r} names ({v['family']}, "
                              f"{v['shape']}, {v.get('level')}, {v['task']}), which no group "
                              "holds")
        out.append(Cell(base.group, base.family, base.shape, base.level, base.task,
                        v["variant"], base.data, _freeze(v.get("sample")), v.get("batch_size")))
        seen.add((base.family, base.shape, base.level, base.task, v["admission"]))
    want = admission_cells(catalog)
    if seen != want:
        missing = sorted(map(str, want - seen))
        extra = sorted(map(str, seen - want))
        raise MatrixError("recipes.json's cells are not catalog.json's admission table: "
                          f"missing {missing or 'none'}, not admitted {extra or 'none'}")
    ids = [c.id for c in out]
    if len(set(ids)) != len(ids):
        raise MatrixError("recipes.json gives two cells one id: "
                          f"{sorted({i for i in ids if ids.count(i) > 1})}")
    if len(out) != int(catalog["admission"]["cells"]):
        raise MatrixError(f"{len(out)} cells, the admission table counts "
                          f"{catalog['admission']['cells']}")
    return out


def select(all_cells, how: str, table=None) -> list:
    """`all`, `smoke` (the table's one cell per family) or a comma list of globs over cell
    ids; a glob that matches nothing is refused."""
    table = table or load_table()
    by_id = {c.id: c for c in all_cells}
    if how == "all":
        return list(all_cells)
    if how == "smoke":
        unknown = [i for i in table["smoke"] if i not in by_id]
        if unknown:
            raise MatrixError(f"recipes.json smoke names unknown cell(s) {unknown}")
        families = [by_id[i].family for i in table["smoke"]]
        if sorted(families) != sorted(CATALOG["families"]):
            raise MatrixError("recipes.json smoke is not one cell per catalog family: "
                              f"{sorted(set(CATALOG['families']) ^ set(families))}")
        return [by_id[i] for i in table["smoke"]]
    picked = []
    for pattern in [p.strip() for p in how.split(",") if p.strip()]:
        hits = [c for c in all_cells if fnmatch.fnmatchcase(c.id, pattern)]
        if not hits:
            raise MatrixError(f"--cells {pattern!r} matches no cell id (e.g. "
                              f"{all_cells[0].id})")
        picked += [c for c in hits if c not in picked]
    return picked


# ================================================================ the raw tables

def _r(values, digits=4):
    return np.round(np.asarray(values, dtype=np.float64), digits)


def rows_table(seed: int) -> pd.DataFrame:
    """rows.csv: a non-linear score `s` of the numerics and the nominals behind every label."""
    rng = np.random.default_rng(seed)
    n = 600
    x = rng.normal(size=(n, 6))
    x[:, 3] = 0.6 * x[:, 2] + 0.8 * x[:, 3]
    c5 = rng.choice(list("abcde"), n)
    c12 = rng.choice([f"k{i:02d}" for i in range(12)], n)
    eff5 = dict(zip("abcde", (-1.0, -0.5, 0.0, 0.5, 1.0)))
    eff12 = {f"k{i:02d}": float(v) for i, v in enumerate(rng.normal(0, 0.5, 12))}
    anomalous = rng.uniform(size=n) < 0.08
    x[anomalous] = 2.5 * x[anomalous] + rng.choice([-3.0, 3.0], size=(int(anomalous.sum()), 6))
    e5 = np.array([eff5[v] for v in c5])
    e12 = np.array([eff12[v] for v in c12])
    s = 1.2 * x[:, 0] - 0.8 * x[:, 1] + 0.6 * x[:, 2] * x[:, 3] + e5 + 0.5 * e12
    multi = np.stack([s, x[:, 4] + 0.5 * e5, -s + x[:, 5], 0.8 * x[:, 1] - x[:, 2]], axis=1)
    margins = np.stack([s - 0.3, x[:, 4] + x[:, 5], x[:, 3] - 1.0 + 2.0 * np.isin(c5, ["a", "b"])],
                       axis=1)
    noisy = s + rng.normal(0, 0.4, n)
    frame = pd.DataFrame({f"x{i + 1}": _r(x[:, i]) for i in range(6)})
    frame.insert(0, "rid", [f"R{i:04d}" for i in range(n)])
    frame["c5"], frame["c12"] = c5, c12
    frame["y_bin"] = (s + rng.normal(0, 0.5, n) > 0).astype(np.int64)
    frame["y_multi"] = np.argmax(multi + rng.normal(0, 0.3, multi.shape), axis=1).astype(np.int64)
    frame["y_tags"] = _tags(margins, ("a", "b", "c"))
    frame["y_reg"] = _r(s + 0.5 * np.sin(x[:, 4]) + rng.normal(0, 0.3, n))
    frame["y_ord"] = np.digitize(noisy, np.quantile(noisy, [0.2, 0.4, 0.6, 0.8])).astype(np.int64)
    frame["y_anom"] = anomalous.astype(np.int64)
    return frame


def _tags(margins, names) -> list:
    """The delimited label: every name whose margin is positive, in class order; the largest
    margin's name when none is (the matrix keeps the empty set out of its fixtures)."""
    out = []
    for row in np.asarray(margins):
        on = [names[j] for j in range(len(names)) if row[j] > 0] or [names[int(np.argmax(row))]]
        out.append("|".join(on))
    return out


def seq_table(seed: int) -> pd.DataFrame:
    """seq.csv: 6 series x 160 hourly steps; target = level + season(12) + regime + exogenous
    + AR(2) noise."""
    rng = np.random.default_rng(seed)
    zones, effect = ("north", "south", "east"), {"north": -1.0, "south": 1.0, "east": 0.0}
    rows = []
    for s in range(6):
        level, phase = -2.5 + s, float(rng.uniform(0, 2 * math.pi))
        e1 = e2 = 0.0
        zone, prev = zones[int(rng.integers(0, 3))], None
        history = []
        for t in range(160):
            if t % 8 == 0:
                zone = zones[int(rng.integers(0, 3))]
            e1, e2 = 0.5 * e1 + 0.2 * e2 + float(rng.normal(0, 0.5)), e1
            x1 = 3.0 * math.sin(2 * math.pi * t / 24 + phase) + float(rng.normal(0, 0.2))
            target = (level + 4.0 * math.sin(2 * math.pi * t / 12 + phase) + effect[zone]
                      + 0.3 * x1 + e1)
            prev = target if prev is None else prev
            history.append(target)
            back3 = history[-4] if len(history) >= 4 else history[0]
            rows.append({"series": f"s{s}", "t": t, "phase": phase, "level": level,
                         "zone": zone, "x1": x1, "x2": float(rng.normal()), "target": target,
                         "prev": prev, "back3": back3})
            prev = target
    frame = pd.DataFrame(rows)
    rising = frame["target"] > frame["prev"]
    steps = np.floor(frame["phase"] / (2 * math.pi) * 12)
    out = pd.DataFrame({
        "series": frame["series"],
        "ts": (pd.Timestamp("2025-01-01") + pd.to_timedelta(frame["t"], unit="h"))
        .dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "zone": frame["zone"], "x1": _r(frame["x1"]), "x2": _r(frame["x2"]),
        "target": _r(frame["target"])})
    out["y_bin"] = rising.astype(np.int64)
    out["y_multi"] = (((frame["t"] + steps) % 12) // 4).astype(np.int64)
    hi, wet = frame["target"] > frame["level"], frame["zone"] == "south"
    tags = []
    for h, u, w in zip(hi, rising, wet):
        on = [n for n, flag in (("hi", h), ("up", u), ("wet", w)) if flag]
        tags.append("|".join(on) if on else "calm")
    out["y_tags"] = tags
    out["y_reg"] = _r(0.5 * frame["target"] + 0.3 * frame["prev"] + frame["x1"]
                      + rng.normal(0, 0.2, len(frame)))
    delta = frame["target"] - frame["back3"]
    out["y_ord"] = np.digitize(delta, np.quantile(delta, [0.25, 0.5, 0.75])).astype(np.int64)
    order = np.lexsort((out["series"].to_numpy(), out["ts"].to_numpy()))
    return out.iloc[order].reset_index(drop=True)


def nodes_table(seed: int) -> pd.DataFrame:
    """nodes.csv: cliques of 3-8 nodes on a customer id, one community each (four); a node's
    own features f1-f4 carry 0.6 of a standard deviation of its community, its clique mates
    the rest."""
    rng = np.random.default_rng(seed)
    sizes = []
    while sum(sizes) < 600:
        sizes.append(int(rng.integers(3, 9)))
    sizes[-1] -= sum(sizes) - 600
    if sizes[-1] < 2:
        sizes[-2] += sizes.pop()
    clique = np.repeat(np.arange(len(sizes)), sizes)
    community = rng.integers(0, 4, len(sizes))[clique]
    centre = rng.uniform(0, 200, size=(len(sizes), 2))
    xy = centre[clique] + rng.normal(0, 1.5, size=(600, 2))
    customer = clique.copy()
    moved = rng.uniform(size=600) < 0.1
    customer[moved] = rng.integers(0, len(sizes), int(moved.sum()))
    frame = pd.DataFrame({"node_key": [f"N{i:04d}" for i in range(600)],
                          "customer_id": [f"C{c:03d}" for c in customer],
                          "px": _r(xy[:, 0], 3), "py": _r(xy[:, 1], 3)})
    for k in range(4):
        frame[f"f{k + 1}"] = _r(0.6 * (community == k) + rng.normal(0, 1.0, 600))
    frame["region"] = rng.choice([f"r{i}" for i in range(6)], 600)
    frame["y_bin"] = (community >= 2).astype(np.int64)
    frame["y_multi"] = community.astype(np.int64)
    frame["y_tags"] = [("a", "a|b", "b", "c")[c] for c in community]
    frame["y_reg"] = _r(community + 0.3 * frame["f1"].to_numpy() + rng.normal(0, 0.3, 600))
    frame["y_ord"] = community.astype(np.int64)
    return frame.iloc[rng.permutation(600)].reset_index(drop=True)


def graphs_table(seed: int) -> pd.DataFrame:
    """graphs.csv: 120 graphs of 8-14 nodes, 40 per motif; node features drawn alike for every
    motif, so only the bonds tell a ring from a chain."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, motif in enumerate(rng.permutation(np.repeat(np.arange(3), 40))):
        n = int(rng.integers(8, 15))
        parent = [None] + list(range(n - 1)) if motif < 2 else [None] + [(k - 1) // 2
                                                                         for k in range(1, n)]
        if motif == 1:
            parent[0] = n - 1
        degree = np.zeros(n, dtype=int)
        for k, p in enumerate(parent):
            if p is not None:
                degree[k] += 1
                degree[p] += 1
        leaves = int((degree == 1).sum())
        y_reg = round(leaves + float(rng.normal(0, 0.3)), 4)
        for k in range(n):
            rows.append({"atom_key": f"G{g:03d}A{k:02d}", "graph_id": f"G{g:03d}",
                         "bonded_to": f"G{g:03d}A{parent[k]:02d}" if parent[k] is not None else "",
                         "element": str(rng.choice(["C", "N", "O"], p=[0.6, 0.25, 0.15])),
                         "charge": round(float(rng.normal(0, 0.3)), 4),
                         "mass": round(float(rng.normal(12, 1)), 4),
                         "y_bin": int(motif == 1), "y_multi": int(motif), "y_reg": y_reg})
    return pd.DataFrame(rows)


# ================================================================ the datasets

@dataclass(frozen=True)
class Dataset:
    """A raw table and how the matrix shapes it: its identity columns, nominals, numerics,
    time and key columns, its labels (name -> classes, None for a value), and its edges."""

    name: str
    make: object
    identity: tuple
    nominals: tuple
    numeric: tuple
    labels: dict
    time: str | None = None
    keys: tuple = ()
    graph_id: str | None = None
    edges: dict = field(default_factory=dict)
    scale: str = "standard"


DATASETS = {
    "rows": Dataset("rows", rows_table, ("rid",), ("c5", "c12"),
                    tuple(f"x{i}" for i in range(1, 7)),
                    {"y_bin": [0, 1], "y_multi": [0, 1, 2, 3], "y_tags": ["a", "b", "c"],
                     "y_reg": None, "y_ord": [0, 1, 2, 3, 4], "y_anom": [0, 1]}),
    "seq": Dataset("seq", seq_table, (), ("zone",), ("x1", "x2", "target"),
                   {"y_bin": [0, 1], "y_multi": [0, 1, 2], "y_tags": ["hi", "up", "wet", "calm"],
                    "y_reg": None, "y_ord": [0, 1, 2, 3], "target": None},
                   time="ts", keys=("series", "ts")),
    "nodes": Dataset("nodes", nodes_table, ("node_key", "customer_id"), ("region",),
                     ("px", "py", "f1", "f2", "f3", "f4"),
                     {"y_bin": [0, 1], "y_multi": [0, 1, 2, 3], "y_tags": ["a", "b", "c"],
                      "y_reg": None, "y_ord": [0, 1, 2, 3]},
                     edges={"key_pair": {"method": "key_pair", "columns": ["customer_id"]},
                            "knn": {"method": "knn", "k": 6, "features": ["px", "py"],
                                    "metric": "euclidean", "weight": "distance"}}),
    "graphs": Dataset("graphs", graphs_table, ("atom_key", "bonded_to"), ("element",),
                      ("charge", "mass"), {"y_bin": [0, 1], "y_multi": [0, 1, 2], "y_reg": None},
                      keys=("graph_id",), graph_id="graph_id",
                      edges={"key_pair": {"method": "key_pair",
                                          "columns": ["atom_key", "bonded_to"],
                                          "graph_id": "graph_id"}}, scale="minmax"),
}

#: the label each effective task reads (a forecast reads the target itself)
TASK_LABEL = {"binary": "y_bin", "multiclass": "y_multi", "multilabel": "y_tags",
              "regression": "y_reg", "ordinal": "y_ord", "forecast": "target",
              "anomaly/labelled": "y_anom", "unsupervised": None, "anomaly/unlabelled": None}
_CLASS_SPLIT = ("binary", "multiclass", "ordinal", "anomaly/labelled")
_PROSE = {v: k for k, v in CATALOG["task_map"].items() if k}


@dataclass(frozen=True)
class Variant:
    """One shaped variant: a raw table, the label it selects (None: no label), how its split
    reads the label (`class`: stratified on it; `plain`: not), and its edges method."""

    data: str
    label: str | None
    split: str
    edges: str | None = None

    @property
    def key(self) -> str:
        return "-".join(p for p in (self.data, self.label or "nolabel", self.edges) if p)


def variant_of(cell: Cell, table: dict) -> Variant:
    task = cell.task if cell.task in TASK_LABEL else cell.effective
    label = TASK_LABEL[task]
    edges = None
    if cell.shape == "graph":
        edges = table["edges"]["by_family"].get(cell.family, table["edges"]["default"])
        if not DATASETS[cell.data].edges.get(edges):
            edges = table["edges"]["default"]
    split = "class" if task in _CLASS_SPLIT else "plain"
    return Variant(cell.data, label, split, edges)


# ================================================================ the recipes

def _step(sid, op, columns, params, source, rationale, alternative=None) -> dict:
    return {"id": sid, "op": op, "columns": list(columns), "params": params, "source": source,
            "rationale": rationale, "alternatives": [alternative] if alternative else [],
            "consequences": {"rows": "none", "columns": "none", "downstream": {}},
            "reversible": op not in ("drop_identity", "dedupe")}


def _default(op, when, params) -> dict:
    return {"op": op, "params": params, "when": when}


def recipe_steps(variant: Variant, split: dict) -> list:
    """The variant's steps, phase by phase (step-catalog.md): retype and identity (1), the
    partition and the other labels as leakage (2), one split (3), encode and scale (5), the
    graph step (7)."""
    ds, steps = DATASETS[variant.data], []

    def add(*args, **kw):
        steps.append(_step(f"S{len(steps) + 1}", *args, **kw))

    add("retype", ds.nominals, {"to": "nominal"}, "user:matrix/typing",
        "these columns name categories; none is a quantity")
    if ds.identity:
        add("drop_identity", ds.identity, {}, "user:matrix/typing",
            "identifiers, not features (a graph step still reads them from the input)")
    if ds.time:
        add("parse_datetime", [ds.time], {"errors": "coerce"}, "user:matrix/time",
            "the timestamp orders each series' rows")
    others = [c for c in ds.labels if c != variant.label and c not in ds.numeric]
    if variant.label:
        add("select_partition", [variant.label],
            {"label": variant.label, "task": _prose(variant)}, "user:matrix/question",
            f"{variant.label} is the question this variant asks")
        add("drop_leakage", others, {}, "user:matrix/labels",
            "the table's other labels are other questions; each would leak its own")
    else:
        keep = [c for c in (*ds.numeric, *ds.nominals, *ds.keys)]
        add("keep_columns", keep, {"columns": keep}, GOAL_SOURCE,
            "no label: the network reconstructs the features, and every label column goes")
    add(*_split(variant, split))
    add("encode", ds.nominals, {"strategy": "ordinal"}, "shaper:default",
        "the network embeds the nominals: an ordinal code with a reserved unseen id",
        _default("encode", "if no block embedded them", {"strategy": "one-hot"}))
    scaled = [c for c in ds.numeric if c != variant.label]
    if ds.scale == "standard":
        add("scale", scaled, {"kind": "standard", "columns": scaled}, "shaper:default",
            "the numeric columns sit on different scales",
            _default("scale", "if the tails were judged genuine", {"kind": "robust"}))
    else:
        add("scale", scaled, {"kind": ds.scale, "columns": scaled}, "user:matrix/graphs",
            "min-max keeps every node feature non-negative, so a sum of messages grows with "
            "a node's bonds as a molecule's does", _default("scale", "if the features were "
                                                            "read alone", {"kind": "standard"}))
    if variant.edges:
        params = {"symmetric": variant.edges == "key_pair", "self_loops": False, "weight": None,
                  "graph_id": None, **ds.edges[variant.edges]}
        columns = params.get("columns") or params.get("features")
        add("graph_edges", columns, params, "user:matrix/relations",
            f"the rows are related ({variant.edges}); the edges part records it")
    return steps


def _prose(variant: Variant) -> str:
    """The partition's task words, as the geometry would write them (catalog task_map)."""
    if variant.label == "y_anom":
        return "anomaly"
    kind = {"y_tags": "multilabel", "y_ord": "ordinal"}.get(variant.label)
    if kind is None:
        classes = DATASETS[variant.data].labels.get(variant.label)
        kind = ("regression" if classes is None else
                "binary" if len(classes) == 2 else "multiclass")
    return _PROSE.get(kind, kind)


def _split(variant: Variant, split: dict) -> tuple:
    ds = DATASETS[variant.data]
    if ds.time:
        return ("split", [], {"kind": "time", "time_column": ds.time, **split}, "user:matrix/time",
                "a time series is held out in time: the future is never trained on")
    if ds.graph_id:
        return ("split", [], {"kind": "group", "by": ds.graph_id, **split}, "user:matrix/graphs",
                "one graph is one sample: its nodes never straddle two parts")
    if variant.split == "class" and variant.label:
        return ("split", [], {"kind": "stratified", "by": variant.label, **split},
                "shaper:default", "independent rows with a class label: every class in every part",
                _default("split", "if the classes were balanced by design", {"kind": "random"}))
    return ("split", [], {"kind": "random", **split}, "shaper:default",
            "independent rows without a class to stratify on",
            _default("split", "if a class label existed", {"kind": "stratified"}))


# ================================================================ shaping

def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256(path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def write_raw(work: Path, name: str, seed: int) -> Path:
    """raw/<name>.csv, rewritten only when its bytes change (the shaped stamps read it)."""
    path = work / "raw" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = DATASETS[name].make(seed).to_csv(index=False).encode("utf-8")
    if not path.is_file() or path.read_bytes() != data:
        path.write_bytes(data)
    return path


def _recipe(csv: Path, variant: Variant, shaped: Path, table: dict) -> dict:
    frame = pd.read_csv(csv, nrows=5)
    rows = sum(1 for _ in csv.open("rb")) - 1
    return {"schema": "dataset-shaper/recipe@1",
            "input": {"path": str(csv), "digest": _sha256(csv), "rows": rows,
                      "columns": int(frame.shape[1]), "layers_read": []},
            "seed": int(table["seed"]), "goal": "custom", "target": variant.label,
            "outputs": {"dir": str(shaped), "format": "parquet", "files": [],
                        "recipe": str(shaped / "recipe.json"),
                        "manifest": str(shaped / "manifest.json"),
                        "lineage": str(shaped / "lineage.json"),
                        "reproduce": str(shaped / f"reproduce_{variant.key}.py")},
            "steps": recipe_steps(variant, table["split"])}


def _run(args, stage: str) -> str:
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          cwd=str(SKILLS), timeout=900)
    if proc.returncode != 0:
        tail = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
        raise MatrixError(f"{stage} exited {proc.returncode}: {tail}")
    return proc.stdout


def shape_variant(work: Path, variant: Variant, table: dict) -> Path:
    """shaped/<variant>/ by shape.py, unless its stamp (the recipe's, the raw table's and
    shape.py's digests) says nothing changed."""
    csv = work / "raw" / f"{variant.data}.csv"
    shaped = work / "shaped" / variant.key
    recipe = _recipe(csv, variant, shaped, table)
    text = json.dumps(recipe, indent=1)
    stamp = {"recipe": _sha256_bytes(text.encode()), "raw": _sha256(csv),
             "shaper": _sha256(SHAPER)}
    stamp_path = shaped / "stamp.json"
    if stamp_path.is_file() and json.loads(stamp_path.read_text()) == stamp \
            and (shaped / "manifest.json").is_file():
        return shaped
    if shaped.exists():
        shutil.rmtree(shaped)
    shaped.mkdir(parents=True, exist_ok=True)
    recipe_path = shaped / "recipe.in.json"
    recipe_path.write_text(text, encoding="utf-8")
    _run([SHAPER, "--recipe", recipe_path, "--out-dir", shaped, "--format", "parquet"],
         f"shape.py ({variant.key})")
    stamp_path.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    return shaped


def pipeline_for(shaped: Path, manifest: dict, spec_like: dict) -> Path:
    """The forge fixture's world model (render_templates.REFERENCE_PIPELINE), the matrix's
    package, its feature contract this variant's, drawn against this shaped manifest."""
    pipeline = rt.reference_pipeline()
    pipeline["system"].update({"name": PACKAGE.replace("_", "-"), "package": PACKAGE})
    pipeline["input"] = {**pipeline.get("input", {}),
                         "shape_manifest_digest": _sha256(shaped / "manifest.json")}
    pipeline["feature_contract"] = dict(spec_like)
    path = shaped / "pipeline.json"
    path.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
    return path


# ================================================================ the spec

def choice(source, rationale, alternatives, catalog=None) -> dict:
    """{source, [catalog], rationale, alternatives} (netspec-contract §1)."""
    node = {"source": source}
    if catalog is not None and source == "vega:default":
        node["catalog"] = catalog
    node["rationale"] = rationale
    node["alternatives"] = [dict(a) for a in alternatives]
    return node


def features(manifest: dict, keys) -> list:
    """The output schema less the label and the keys: an ordinal-encoded column is a nominal
    (int64, `cardinality = unseen_category_id + 1`), every other one numeric (float32)."""
    encoded = {}
    for step in manifest.get("steps") or []:
        if step.get("op") == "encode":
            for col, entry in ((step.get("parameters_fitted") or {}).get("fitted") or {}).items():
                if isinstance(entry, dict) and entry.get("unseen_category_id") is not None:
                    encoded[col] = (int(entry["unseen_category_id"]), step["id"])
    out = []
    for name, meta in (manifest.get("output_schema") or {}).items():
        if name == manifest.get("label") or name in keys:
            continue
        origin = meta.get("origin") or "input"
        if name in encoded:
            unseen, sid = encoded[name]
            out.append({"name": name, "dtype": "int64", "role": "nominal",
                        "cardinality": unseen + 1, "unseen_category_id": unseen,
                        "origin": origin, "role_source": [f"shape:{sid}"]})
            continue
        dtype = str(meta.get("dtype") or "").lower()
        if not dtype.startswith(("int", "uint", "float", "bool")):
            raise MatrixError(f"output column {name!r} ({dtype}) is neither numeric, an encoded "
                              "nominal, the label nor a key")
        out.append({"name": name, "dtype": "float32", "role": "numeric", "origin": origin})
    return out


def embedding_dim(cardinality: int) -> int:
    """defaults/embedding_dim: min(50, round(1.6 * cardinality ** 0.56)) rounded to an even
    number (at least 2)."""
    raw = min(50.0, 1.6 * cardinality ** 0.56)
    return max(2, int(2 * round(raw / 2)))


@dataclass(frozen=True)
class Frame:
    """What a composition reads off one cell's shaped variant."""

    cell: Cell
    feats: tuple
    manifest: dict
    table: dict
    epochs: int

    @property
    def nominals(self) -> dict:
        return {f["name"]: int(f["cardinality"]) for f in self.feats if f["role"] == "nominal"}

    @property
    def n_numeric(self) -> int:
        return sum(1 for f in self.feats if f["role"] == "numeric")


def head_out(frame: Frame, z=None) -> int:
    """The head's `out` the task row forces (T1b)."""
    cell = frame.cell
    kind, classes = cell.effective, _classes(cell)
    if kind == "binary":
        return 1
    if kind in ("multiclass", "multilabel"):
        return len(classes)
    if kind == "ordinal":
        return len(classes) - 1
    if kind == "regression":
        return 1
    if kind == "forecast":
        return int(frame.table["window"]["horizon"])
    if z is not None:
        return int(z)
    return frame.n_numeric + sum(frame.nominals.values())


def _classes(cell: Cell):
    label = TASK_LABEL.get(cell.task if cell.task in TASK_LABEL else cell.effective)
    return DATASETS[cell.data].labels.get(label) if label else None


def _alternative(block: dict) -> dict:
    """The alternative a block's choice weighs (never empty: no block but concat and the head
    is structural)."""
    kind, params = block["type"], block.get("params") or {}
    if kind == "embedding":
        return {"params": {"dims": {c: max(2, int(d) // 2) for c, d in params["dims"].items()}},
                "when": "if the embedding overfits"}
    if kind in ("dropout", "edge_dropout"):
        return {"params": {"p": round(float(params.get("p", 0.1)) + 0.1, 2)},
                "when": "if train is far above valid"}
    if kind == "activation":
        other = "gelu" if params.get("name") != "gelu" else "relu"
        return {"params": {"name": other}, "when": "as an ablation of the non-linearity"}
    for key in _WIDTH_KEYS:
        if isinstance(params.get(key), int) and not isinstance(params.get(key), bool):
            return {"params": {key: int(params[key]) * 2}, "when": UNDERFITS}
    if kind in ("pool", "graph_pool", "pool1d"):
        return {"params": {}, "when": "another reduction, as an ablation of the readout"}
    if kind in ("residual", "sequential", "decoder"):
        return {"params": {}, "when": "one layer wider, " + UNDERFITS}
    return {"params": {}, "when": "the block removed, as an ablation"}


def _expand(item, frame: Frame) -> list:
    """A composition item: a block, or a placeholder (recipes.json's note)."""
    cell, table = frame.cell, frame.table
    if isinstance(item, dict) and item.get("$") == "head":
        return [_head(frame, item.get("z"), item.get("bias", True))]
    if isinstance(item, dict):
        return [copy.deepcopy(item)]
    if item == "embed":
        noms = frame.nominals
        if not noms:
            return []
        dims = {c: embedding_dim(card) for c, card in noms.items()}
        return [{"type": "embedding", "params": {"columns": list(noms), "dims": dims}},
                {"type": "concat", "params": {}}]
    if item == "flatten@window":
        return [{"type": "flatten", "params": {}}] if cell.shape == "window" else []
    if item == "pool@graph":
        readout = table["pool"]["by_family"].get(cell.family, table["pool"]["default"])
        return ([{"type": "graph_pool", "params": {"readout": readout}}]
                if cell.shape == "graph" and cell.level == "graph" else [])
    if item == "head":
        return [_head(frame)]
    raise MatrixError(f"recipes.json {cell.family}: unknown composition item {item!r}")


def _head(frame: Frame, z=None, bias=True) -> dict:
    params = {"out": head_out(frame, z), "activation": "none"}
    if not bias:
        params["bias"] = False
    return {"type": "head", "params": params}


def _traced(block: dict, frame: Frame, task: dict) -> dict:
    """A top-level block with its choice: the head cites the task's source (structural: the
    task row forces it), concat is structure, the embedding the catalog's dimension rule,
    every other block the family's catalog composition at the matrix's sizes."""
    kind, family = block["type"], frame.cell.family
    if kind == "head":
        node = choice(task["source"], f"out {block['params']['out']} for the "
                      f"{frame.cell.effective} task; the loss applies the link, never the head",
                      [])
    elif kind == "concat":
        node = choice("vega:default", "structural: the numeric features are joined to the "
                      "embeddings before the first block that reads them", [], "blocks/concat")
    elif kind == "embedding":
        node = choice("vega:default", "every nominal embedded at the catalog's dimension rule",
                      [_alternative(block)], "defaults/embedding_dim")
    else:
        node = choice("vega:default", f"the {family} composition "
                      f"({CATALOG['families'][family]['composition']}) at the matrix's small "
                      "sizes", [_alternative(block)], f"families/{family}")
    return {**block, **node}


def compose_blocks(frame: Frame, task: dict) -> list:
    items = frame.table["families"][frame.cell.family]
    bare = [b for item in items for b in _expand(item, frame)]
    return [_traced({"id": f"B{i}", **b}, frame, task) for i, b in enumerate(bare, 1)]


def compose_task(cell: Cell) -> dict:
    label = TASK_LABEL.get(cell.task if cell.task in TASK_LABEL else cell.effective)
    task = {"kind": cell.kind, "label": label,
            "source": LABEL_SOURCE if label else GOAL_SOURCE}
    if cell.base:
        task["base_kind"] = cell.base
    classes = _classes(cell)
    if classes is not None and cell.effective != "regression":
        task["classes"] = list(classes)
    if cell.effective == "multilabel":
        task["label_format"] = {"kind": "delimited", "sep": "|"}
    if cell.effective in ("binary", "anomaly") and classes is not None:
        task["positive"] = classes[1]
    return task


def compose_sample(frame: Frame, steps: list) -> dict:
    cell, table = frame.cell, frame.table
    split_id = next(s["id"] for s in steps if s["op"] == "split")
    if cell.shape == "row":
        return {"shape": "row"}
    if cell.shape == "window":
        w = table["window"]
        sample = {"shape": "window", "length": w["length"], "stride": 1,
                  "series_by": "series", "order_by": "ts", "context": "previous-part",
                  "autoregressive": cell.kind == "forecast", "season": w["season"],
                  "source": f"shape:{split_id}", "context_source": "vega:default",
                  "context_catalog": "defaults/window_context",
                  "autoregressive_source": "vega:default",
                  "autoregressive_catalog": "defaults/autoregressive",
                  "series_source": "user:matrix/series"}
        if cell.kind == "forecast":
            sample["horizon"] = w["horizon"]
        sample.update(cell.sample_overrides())
        return sample
    return _graph_sample(frame)


def _graph_sample(frame: Frame) -> dict:
    cell, graph = frame.cell, frame.manifest.get("graph") or {}
    level = cell.level
    sample = {"shape": "graph", "edges": frame.manifest["outputs"]["files"]["edges"]["path"],
              "level": level, "graph_id": "graph_id" if level == "graph" else None,
              "directed": bool(graph.get("directed")), "self_loops": bool(graph.get("self_loops")),
              "protocol": "inductive", "protocol_source": "vega:default",
              "protocol_catalog": "defaults/graph_protocol",
              "batching": "graphs" if level == "graph" else "full",
              "batching_source": "vega:default", "batching_catalog": "defaults/graph_batching",
              "neighbours": None, "exposure": None, "source": "shape:manifest/graph"}
    over = cell.sample_overrides()
    if over.get("protocol") == "transductive":
        sample.update(protocol="transductive", protocol_source="user:matrix/transductive",
                      exposure={"kind": "features", "rows": "test",
                                "source": "user:matrix/transductive"})
        sample.pop("protocol_catalog")
    if over.get("batching") == "sampled":
        sample.update(batching="sampled", batching_source="user:matrix/sampled",
                      neighbours=list(over["neighbours"]))
        sample.pop("batching_catalog")
    return sample


def compose_loss(cell: Cell) -> dict:
    row = CATALOG["tasks"][cell.effective]
    family_loss = CATALOG["families"][cell.family].get("loss")
    name = family_loss or row["default_loss"]
    catalog = f"families/{cell.family}" if family_loss else f"tasks/{cell.effective}"
    others = [n for n in row["losses_allowed"] if n != name]
    alternative = ({"name": others[0], "when": "as an ablation of the objective"} if others else
                   {"name": name, "params": {}, "when": "the task row allows no other loss: "
                                                         "the same objective, as its comparison"})
    return {"name": name, "params": {}, **choice("vega:default", f"the {cell.effective} "
                                                 "task's loss", [alternative], catalog)}


def compose_metrics(cell: Cell) -> dict:
    row = CATALOG["tasks"][cell.effective]
    primary, secondary = row["primary"], list(row["secondary"])
    if cell.task == "anomaly/unlabelled":
        primary, secondary = row["primary_alt"]["metric"], []
    alt = (row.get("primary_alt") or {}).get("metric") if cell.labelled else None
    alt = alt if alt and alt != primary else (secondary[0] if secondary else None)
    alternatives = ([{"primary": alt, "when": "if the task's balance or reading changes"}] if alt
                    else [{"primary": primary, "secondary": [],
                           "when": "the only reading without a label"}])
    return {"primary": primary, "secondary": secondary,
            "selection": {"monitor": f"valid.{primary}",
                          "mode": CATALOG["metric_direction"][primary]},
            **choice("vega:default", f"the {cell.effective} task row's primary and secondary "
                     "metrics; selection reads the primary on valid", alternatives,
                     f"tasks/{cell.effective}")}


def compose_regularization(blocks: list, weight_decay: float) -> list:
    """The declaration (netspec-contract §3, R2): weight decay, one dropout entry per rate the
    top-level dropout blocks carry (so every block trains at its own rate), the clip. The
    copies it covers are taken out of the blocks and the optimizer by `declared_once`."""
    out = [{"kind": "weight_decay", "value": weight_decay, **choice(
        "vega:default", "mild shrinkage; a coverage run asks for no more",
        [{"value": weight_decay * 10, "when": "if train is far above valid"}],
        "defaults/weight_decay")}]
    rates: dict = {}
    for b in blocks:
        if b["type"] == "dropout":
            rates.setdefault(float(b["params"]["p"]), []).append(b["id"])
    for rate, where in rates.items():
        out.append({"kind": "dropout", "value": rate, "where": where, **choice(
            "vega:default", "the dropout the blocks carry, declared once",
            [{"value": round(rate + 0.2, 2),
              "when": "if the gap between train and valid stays wide"}], "defaults/dropout")})
    out.append({"kind": "grad_clip", "value": float(CATALOG["defaults"]["grad_clip"]["value"]),
                **choice("vega:default", "a cheap guard against the non-finite epoch",
                         [{"value": None, "when": "as an ablation"}], "defaults/grad_clip")})
    return out


def declared_once(blocks: list, optimizer: dict, regularization: list) -> tuple:
    """(blocks, optimizer) without the copies the declaration covers: the emitted code reads
    `regularization[]`, and a copy beside it is refused when a rung moves the declaration."""
    blocks, optimizer = copy.deepcopy(blocks), copy.deepcopy(optimizer)
    if any(r["kind"] == "weight_decay" for r in regularization):
        optimizer["params"].pop("weight_decay", None)
    covered = {bid for r in regularization if r["kind"] == "dropout" for bid in r["where"]}
    for b in blocks:
        if b["type"] == "dropout" and b["id"] in covered:
            b["params"].pop("p", None)
    return blocks, optimizer


def compose_training(frame: Frame, sample: dict) -> dict:
    cell = frame.cell
    batch = cell.batch_size or frame.table["batch_size"][cell.shape_tag]
    other = 256 if batch == "full" else int(batch) * 2
    return {"batch_size": batch, "max_epochs": frame.epochs, "device": "cpu",
            "precision": "fp32", "num_workers": 0, "budget": {"wall_clock_min": 10}, **choice(
                "user:matrix/budget", f"the matrix's budget: {frame.epochs} epochs on the CPU, "
                "where the backend's determinism claim is exact; the batch is the sample's "
                f"({sample.get('batching') or cell.shape})",
                [{"batch_size": other, "when": "on a campaign, by defaults/batch_size"}])}


def compose(cell: Cell, shaped: Path, pipeline_path: Path, table: dict, epochs: int) -> dict:
    """The cell's netspec over its shaped variant (netspec-contract §1, every choice traced)."""
    manifest_path = shaped / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = json.loads((shaped / "recipe.in.json").read_text(encoding="utf-8"))
    ds = DATASETS[cell.data]
    frame = Frame(cell, tuple(features(manifest, ds.keys)), manifest, table, epochs)
    task = compose_task(cell)
    blocks = compose_blocks(frame, task)
    sample = compose_sample(frame, recipe["steps"])
    metrics = compose_metrics(cell)
    opt = copy.deepcopy(table["optimizer"])
    rate = (table.get("lr_by_shape") or {}).get(cell.shape_tag)
    if rate is not None:
        opt["params"]["lr"] = rate
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    files = manifest["outputs"]["files"]
    regularization = compose_regularization(blocks, float(opt["params"]["weight_decay"]))
    blocks, opt = declared_once(blocks, opt, regularization)
    return {
        "schema": "vega/netspec@1",
        "input": {"model": None, "layers_read": ["shape"], "shaped_dir": str(shaped),
                  "manifest": str(manifest_path), "manifest_digest": _sha256(manifest_path),
                  "pipeline": str(pipeline_path),
                  # the canonical digest (netspec-contract §1, D1e): sorted keys, no whitespace
                  "pipeline_digest": _sha256_bytes(json.dumps(pipeline, sort_keys=True,
                                                              separators=(",", ":")).encode("utf-8"))},
        "seed": int(table["seed"]),
        "task": task,
        "data_contract": {"source": "shape:manifest/output_schema",
                          "parts": {"train": files["train"]["path"],
                                    "valid": files["valid"]["path"], "test": "sealed"},
                          "features": list(frame.feats), "input_dim": len(frame.feats),
                          "keys": list(ds.keys), "sample": sample},
        "family": {"name": cell.family, **choice(
            "user:matrix/family", f"the coverage matrix runs the {cell.family} family on the "
            f"{cell.id} cell (group {cell.group}, the catalog's admission table)",
            [{"name": "linear", "when": "the floor every family is read against"}])},
        "blocks": blocks,
        "loss": compose_loss(cell),
        "metrics": metrics,
        "optimizer": {**opt, **choice("user:matrix/optimizer", "adamw above the catalog's "
                                      "step, so a few epochs move the weights (a graph view "
                                      "takes a few steps an epoch at most: its rate is higher)",
                                      [{"name": "adamw", "params": {"lr": 0.001},
                                        "when": "the catalog's rate (defaults/optimizer)"}])},
        "schedule": {"name": "none", "params": {}, **choice(
            "user:matrix/schedule", "a constant rate: a coverage run compares no schedules",
            [{"name": "plateau", "when": "the catalog's default below 30 epochs"}])},
        "regularization": regularization,
        "early_stopping": {"monitor": metrics["selection"]["monitor"],
                           "mode": metrics["selection"]["mode"], "patience": 1000,
                           "min_delta": 0.001, **choice(
                               "user:matrix/budget", "never within the matrix's budget: every "
                               "fit runs all its epochs",
                               [{"params": {"patience": 10}, "when": "the catalog's, on a "
                                                                    "campaign"}])},
        "training": compose_training(frame, sample),
        "init": {"scheme": "kaiming_uniform", "bias": "zeros", **choice(
            "vega:default", "kaiming_uniform keeps the activations' variance at depth",
            [{"scheme": "xavier_uniform", "when": "if the non-linearity changes family"}],
            "defaults/init")},
        "backend": {"name": "pytorch", "min_version": "2.2", "hints": {}},
        "decision_families": copy.deepcopy(CATALOG["decision_families"]),
        "outputs": {"run_dir": f"vega-runs/matrix/{cell.slug}/", "package": PACKAGE,
                    "code": [f"{PACKAGE}/model/network.py", f"{PACKAGE}/ingestion/data.py",
                             f"{PACKAGE}/training/loop.py", f"{PACKAGE}/model/metrics.py"],
                    "notebook": "engineer.ipynb"},
    }


def compile_check(cell: Cell, spec: dict, manifest: dict, pipeline: dict) -> list:
    """The compiler's core over the spec and its chain; any ERROR raises, naming the cell and
    the rule. Returns the warnings, as lines."""
    markers = cs.compile_spec(spec, model_html=None, pipeline=pipeline, manifest=manifest)
    if markers.errors:
        said = "; ".join(f"{m.rule}{f' {m.where}' if m.where else ''}: {m.message}"
                         for m in markers.errors)
        raise MatrixError(f"{cell.id}: compile_spec refuses the spec: {said}")
    return [f"{m.rule}: {m.message}" for m in markers.warns]


# ================================================================ making the fixtures

@dataclass(frozen=True)
class Made:
    """One cell's fixture as made (`fixture.json`, `vega/backend-matrix-cell@1`)."""

    id: str
    slug: str
    group: str
    family: str
    shape: str
    level: str | None
    task: str
    variant: str
    effective: str
    labelled: bool
    mode: str
    epochs: int
    dir: str
    raw: str
    shaped: str
    manifest: str
    fitted: str
    pipeline: str
    spec: str
    package: str
    keys: tuple
    label: str | None
    graph_id: str | None
    baseline: str | None = None
    compile_warnings: tuple = ()

    def as_dict(self) -> dict:
        out = asdict(self)
        out["keys"], out["compile_warnings"] = list(self.keys), list(self.compile_warnings)
        return {"schema": CELL_SCHEMA, **out}


def _write_cell(root: Path, cell: Cell, shaped: Path, table: dict, mode: str, epochs: int,
                baseline=None) -> Made:
    root.mkdir(parents=True, exist_ok=True)
    # the variant's world model as the blueprint of this cell's package draws it: a shaped
    # variant serves cells of several sample shapes, and a port carries its shape's methods
    pipeline_path = root / "pipeline.json"
    pipeline_path.write_text(json.dumps(rt.shape_pipeline(
        json.loads((shaped / "pipeline.json").read_text(encoding="utf-8")), cell.shape,
        cell.effective), indent=2) + "\n", encoding="utf-8")
    spec = compose(cell, shaped, pipeline_path, table, epochs)
    manifest = json.loads((shaped / "manifest.json").read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    warnings = compile_check(cell, spec, manifest, pipeline)
    spec_path = root / "netspec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    ds = DATASETS[cell.data]
    made = Made(id=cell.id, slug=cell.slug, group=cell.group, family=cell.family,
                shape=cell.shape, level=cell.level, task=cell.task, variant=cell.variant,
                effective=cell.effective, labelled=cell.labelled, mode=mode, epochs=epochs,
                dir=str(root), raw=str(Path(shaped).parent.parent / "raw" / f"{cell.data}.csv"),
                shaped=str(shaped), manifest=str(shaped / "manifest.json"),
                fitted=str(shaped / "fitted.json"), pipeline=str(pipeline_path),
                spec=str(spec_path), package=PACKAGE, keys=tuple(ds.keys),
                label=spec["task"]["label"], graph_id=ds.graph_id, baseline=baseline,
                compile_warnings=tuple(warnings))
    (root / "fixture.json").write_text(json.dumps(made.as_dict(), indent=2) + "\n",
                                       encoding="utf-8")
    return made


def sanity_runs(chosen, all_cells, table) -> list:
    """(cell, epochs, baseline) of every fit the table's sanity checks need whose checked cell
    is among `chosen` (a floor cell is fitted with it, chosen or not)."""
    by_id = {c.id: c for c in all_cells}
    picked = {c.id for c in chosen}
    out = []
    for check in table["sanity"]:
        if check["cell"] not in picked:
            continue
        floor = check["floor"]
        out.append((by_id[check["cell"]], int(check["epochs"]), floor.get("baseline")))
        if floor.get("cell"):
            out.append((by_id[floor["cell"]], int(check["epochs"]), None))
    return out


def make(work, chosen, table=None, sanity: bool = False, jobs: int = 4) -> dict:
    """Make the fixtures of `chosen` cells (and, with `sanity`, of the learning checks' fits)
    under `work`; return {"cells": {id: Made}, "sanity": {id: Made}}."""
    table = table or load_table()
    work = Path(work).resolve()
    all_cells = cells(table)
    runs = sanity_runs(chosen, all_cells, table) if sanity else []
    needed = {variant_of(c, table) for c in [*chosen, *(r[0] for r in runs)]}
    for name in sorted({v.data for v in needed}):
        write_raw(work, name, int(table["seed"]))
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        shaped = dict(zip(needed, pool.map(lambda v: shape_variant(work, v, table), needed)))
    for variant, path in shaped.items():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        feats = features(manifest, DATASETS[variant.data].keys)
        pipeline_for(path, manifest, {"input_dim": len(feats), "label": variant.label})
    made = {"cells": {}, "sanity": {}}
    for cell in chosen:
        made["cells"][cell.id] = _write_cell(work / "cells" / cell.slug, cell,
                                             shaped[variant_of(cell, table)], table, "cell",
                                             int(table["epochs"]))
    for cell, epochs, baseline in runs:
        made["sanity"][cell.id] = _write_cell(work / "sanity" / cell.slug, cell,
                                              shaped[variant_of(cell, table)], table, "sanity",
                                              epochs, baseline)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--cells", default="all", help="all (default), smoke, or a comma list of "
                                                   "globs over cell ids")
    ap.add_argument("--sanity", action="store_true", help="also the learning checks' fits")
    ap.add_argument("--jobs", type=int, default=4, help="shape.py runs at once (default 4)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        table = load_table()
        chosen = select(cells(table), a.cells, table)
        made = make(a.work, chosen, table, a.sanity, a.jobs)
    except MatrixError as exc:
        print(f"ERROR: {exc}")
        return 2
    if a.json:
        print(json.dumps({k: {i: m.as_dict() for i, m in v.items()} for k, v in made.items()},
                         indent=2))
        return 0
    for kind, group in made.items():
        for m in group.values():
            print(f"OK: {kind[:-1] if kind == 'cells' else kind}: {m.id} "
                  f"({m.effective}, {m.epochs} epochs) under {m.dir}")
            for w in m.compile_warnings:
                print(f"WARN: {m.id}: compile_spec {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
