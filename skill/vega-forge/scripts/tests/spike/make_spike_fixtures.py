#!/usr/bin/env python3
"""
make_spike_fixtures.py — the vertical-slice spike's fixtures (BUILD-PLAN B15, milestone M1): raw
rows, shaped by the real `shape.py`, under a netspec the compiler admits, emitted by the v2.3
emitter.

    python3 make_spike_fixtures.py <work-dir> [--fixture NAME[,NAME…]|all] [--emit full]
                                   [--json]

The spike proves at template level, before any pack or page work builds on it, that what the
evaluation scores and what the serving assembler answers are one computation (R-A3, R-A5,
R-A6). These are the three shapes it proves it on; each is a campaign directory a later package
can reuse (B14's conformance, B17's predictions layout, B18's sample panel, S05/S10's round
trips):

  (a) `window`        3 stations × 300 hourly steps: a nominal (`weather`, ordinal-encoded and
                      embedded), two numerics, the target `demand`; time split 60/20/20; a `lag`
                      of the target per station (lags 1, 2 — lag depth 2), imputed with the
                      train median and scaled; a `gru` forecast, L = 12, H = 3, autoregressive,
                      `previous-part` context (the catalog defaults, marked).
      `window-nolag`  the same rows and network without the lag step (and its impute): R-A6's
                      boundary is then empty.
  (b) `node-keypair`  600 nodes of a two-community graph, balanced: two noisy own features and a
                      noise feature, a position (`px`, `py`) in 40 clusters whose community is
                      drawn at random (so the position alone does not say it); `key_pair` edges on
                      a shared team id (cliques of 5 inside a cluster, one node in ten moved to a
                      random team); stratified split 60/20/20; a two-layer `gcn`, inductive,
                      full batch.
      `node-knn`      the same rows and network; `knn` edges (k 6, directed, distance-weighted)
                      on the position, fitted on the train nodes. (A recipe carries at most one
                      `graph_edges` step, so the two methods are two fixtures.)
  (c) `graph-level`   60 molecule-like graphs of 6–12 atoms (an element nominal, a charge, the
                      bond count), chain bonds with a ring closure on every third molecule,
                      `key_pair` within `molecule_id`; group split by molecule (36/12/12); a
                      `gin` with `graph_pool(sum)`, graphs collated 8 at a time.

Per fixture, under `<work-dir>/<name>/`:

  raw.csv               the input rows (deterministic: every generator is seeded)
  recipe.in.json        the dataset-shaper recipe, every step with its source
  shaped/               shape.py's outputs (parts, edges part, manifest.json, fitted.json, …)
  shaped/preprocess.py  `shape.py --transform-only`: `transform`, and `edges_for` /
                        `edges_within` on a graph recipe
  pipeline.json         the world model (the forge fixture's, its package named for the fixture,
                        drawn against this shaped manifest)
  netspec.json          the spec composed over the shaped manifest (netspec-contract §1): the
                        output schema less the label and the keys as features (an
                        ordinal-encoded nominal with `cardinality = unseen_category_id + 1`), the
                        sample, the network — every choice with its source, rationale and
                        alternatives. The spike has no model HTML, so it cites only what resolves
                        without one: the shaped manifest (`shape:`), the world model
                        (`blueprint:`), the catalog (`vega:default` with its row) and its own
                        stances (`user:spike/…`, the sizes and budget of a proof, not a campaign)
  package/<package>/    `emit(spec, pipeline, package/, generation="v2.3")`
  fixture.json          what a reader needs back (`Fixture.as_dict()`)

Every spec passes the compiler's core (`compile_spec.compile_spec` over the spec, the pipeline
and the manifest). The one allowance is A7(a)'s, until gate A: rule B2's refusal of a block
`coverage.json` still lists as planned (gcn_conv, gin_conv, graph_pool) is recorded in
`Fixture.compile_notes` instead of refusing — B14 empties `planned` at gate A, and from then on
the allowance cannot apply. The package is always the emitter's own: the file-by-file rendering
that stood in while the v2.3 template set landed was retired after gate A (A29, B18), so a
template the table needs and the set lacks fails the fixture instead of shrinking it.

`make(name, work)` returns the `Fixture` (as does `load(work, name)` for one already made);
`make_all(work)` makes every fixture. A fixture already made is made again (the directory is
replaced). Exit codes: 0 made; 2 a fixture could not be made (the message names the stage).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
SCRIPTS = TESTS.parent
SKILLS = SCRIPTS.parent.parent
SHAPER = SKILLS / "dataset-shaper" / "scripts" / "shape.py"
BACKEND = SCRIPTS / "backends" / "pytorch"
TEMPLATES = BACKEND / "templates"
COVERAGE = BACKEND / "coverage.json"
for _p in (str(TESTS), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compile_spec as cs  # noqa: E402
import render_templates as rt  # noqa: E402

CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
SCHEMA = "vega/spike-fixture@1"
SEED = 7
EPOCHS = 3                      # the spike's fits (the slow learning check sets its own)
SPLIT = {"test": 0.2, "valid": 0.2}
LABEL_SOURCE = "shape:manifest/label"
UNDERFITS = "if the run underfits"


class SpikeError(RuntimeError):
    """A fixture could not be made; the message names the stage and what it printed."""


@dataclass(frozen=True)
class Fixture:
    """One spike fixture as made: where everything is, and the facts the spike checks against
    (`sample` is the netspec's; `lag_depth` the largest lag of the recipe's lag steps, 0
    without one; `keys` the spec's key columns; `compile_notes` the compiler refusals the A7(a)
    allowance recorded instead of refusing — empty from gate A on)."""

    name: str
    root: str
    raw: str
    recipe: str
    shaped: str
    manifest: str
    fitted: str
    transform: str
    pipeline: str
    spec: str
    package_dir: str
    package: str
    label: str
    keys: tuple
    sample: dict
    lag_depth: int
    emission: str = "full"      # always "full" since gate A: the emitter (the partial
    missing: tuple = ()         # rendering and the templates it lacked are retired, A29)
    compile_notes: tuple = ()   # B2 "planned" refusals, recorded under A7(a) until gate A

    def as_dict(self) -> dict:
        out = asdict(self)
        for key in ("keys", "missing", "compile_notes"):
            out[key] = list(getattr(self, key))
        return {"schema": SCHEMA, **out}

    @staticmethod
    def from_dict(d: dict) -> "Fixture":
        fields = {k: v for k, v in d.items() if k != "schema"}
        for key in ("keys", "missing", "compile_notes"):
            fields[key] = tuple(fields.get(key) or ())
        return Fixture(**fields)


# ================================================================ the raw rows

def window_rows(seed: int = SEED) -> pd.DataFrame:
    """3 stations × 300 hourly steps. `demand` has a daily cycle, a weather effect and an AR(1)
    part per station; `temp` and `wind` are exogenous; `weather` changes every few hours."""
    rng = np.random.default_rng(seed)
    kinds = ("sun", "rain", "fog", "snow")
    effect = {"sun": 4.0, "rain": -3.0, "fog": -1.0, "snow": -6.0}
    rows = []
    for s in range(3):
        level, last, weather = 40.0 + 8.0 * s, 0.0, kinds[0]
        for t in range(300):
            if t % 5 == 0:
                weather = kinds[int(rng.integers(0, 4))]
            temp = 12.0 + 6.0 * math.sin(2 * math.pi * (t - 6) / 24) + rng.normal(0, 0.8)
            last = 0.6 * last + rng.normal(0, 2.0)
            demand = (level + 15.0 * math.sin(2 * math.pi * t / 24 + s) + effect[weather]
                      + 0.4 * temp + last)
            rows.append({"station": f"st{s}", "ts": pd.Timestamp("2025-03-01")
                         + pd.Timedelta(hours=t), "weather": weather,
                         "temp": round(temp, 3), "wind": round(float(rng.gamma(2.0, 2.0)), 3),
                         "demand": round(demand, 3)})
    frame = pd.DataFrame(rows).sort_values(["ts", "station"], kind="stable")
    frame["ts"] = frame["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return frame.reset_index(drop=True)


def node_rows(seed: int = SEED) -> pd.DataFrame:
    """600 nodes, two balanced communities. A node's own features say its community weakly;
    its team mates and its spatial neighbours share it, so the relation carries the signal."""
    rng = np.random.default_rng(seed)
    clusters, per = 40, 15
    centre = rng.uniform(0, 100, size=(clusters, 2))
    community = rng.permutation(np.repeat([0, 1], clusters // 2))
    cluster = np.repeat(np.arange(clusters), per)
    team = cluster * 3 + np.tile(np.repeat(np.arange(3), 5), clusters)
    moved = rng.uniform(size=len(team)) < 0.1
    team[moved] = rng.integers(0, clusters * 3, size=int(moved.sum()))
    c = community[cluster]
    xy = centre[cluster] + rng.normal(0, 2.0, size=(len(cluster), 2))
    frame = pd.DataFrame({
        "node_key": [f"N{i:04d}" for i in range(len(cluster))],
        "team_id": [f"T{t:03d}" for t in team],
        "px": np.round(xy[:, 0], 3), "py": np.round(xy[:, 1], 3),
        "f1": np.round(0.8 * c + rng.normal(0, 1.0, len(c)), 4),
        "f2": np.round(0.8 * c + rng.normal(0, 1.0, len(c)), 4),
        "f3": np.round(rng.normal(0, 1.0, len(c)), 4),
        "churned": c.astype(np.int64)})
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def graph_rows(seed: int = SEED) -> pd.DataFrame:
    """60 molecule-like graphs of 6–12 atoms: a chain of bonds (each atom bonded to the one
    before it), closed into a ring on every third molecule; `soluble` depends on the share of
    polar atoms and the mean charge, so the label is one per molecule."""
    rng = np.random.default_rng(seed)
    elements, rows = ("C", "N", "O", "S"), []
    for m in range(60):
        n = int(rng.integers(6, 13))
        atoms = [elements[int(k)] for k in rng.choice(4, size=n, p=[0.55, 0.2, 0.15, 0.1])]
        charge = np.round(rng.normal(0, 0.3, size=n), 4)
        ring = m % 3 == 0
        polar = sum(a in ("N", "O") for a in atoms) / n
        soluble = int(polar + 0.5 * float(charge.mean()) + rng.normal(0, 0.08) > 0.33)
        for k in range(n):
            bonded = (k - 1 if k else (n - 1 if ring else None))
            degree = 2 if (0 < k < n - 1 or ring) else 1
            rows.append({"atom_key": f"A{m:02d}{k:02d}", "molecule_id": f"M{m:02d}",
                         "bonded_to": f"A{m:02d}{bonded:02d}" if bonded is not None else "",
                         "element": atoms[k], "charge": float(charge[k]), "degree": degree,
                         "soluble": soluble})
    return pd.DataFrame(rows)


# ================================================================ the recipes

def _step(sid, op, columns, params, source, rationale, alternative=None):
    return {"id": sid, "op": op, "columns": list(columns), "params": params, "source": source,
            "rationale": rationale,
            "alternatives": [alternative] if alternative else [],
            "consequences": {"rows": "none", "columns": "none", "downstream": {}},
            "reversible": op not in ("drop_identity", "dedupe")}


def _default(op, why, alt_params):
    """A `shaper:default` step's alternative (the shaper warns when one carries none)."""
    return {"op": op, "params": alt_params, "when": why}


def window_steps(lag: bool) -> list:
    steps = [
        _step("S1", "retype", ["weather"], {"to": "nominal"}, "geometry:typing/weather",
              "the weather names a condition; it has no order"),
        _step("S2", "parse_datetime", ["ts"], {"errors": "coerce"}, "analysis:context.time",
              "the timestamp orders each station's rows"),
        _step("S3", "select_partition", ["demand"], {"label": "demand", "task": "regression"},
              "geometry:partitions/demand", "the demand of the next hours is the question"),
        _step("S4", "split", [], {"kind": "time", "time_column": "ts", **SPLIT},
              "analysis:context.time", "a time series is held out in time")]
    scaled = ["temp", "wind"]
    if lag:
        steps += [
            _step("S5", "lag", ["demand"], {"lags": [1, 2], "group_by": "station",
                                             "time_column": "ts"}, "analysis:F2/T1",
                  "demand depends on its own recent past, per station"),
            _step("S6", "impute", ["demand_lag1", "demand_lag2"], {"strategy": "median"},
                  "analysis:F2/T2", "a lag is empty on the first rows of each part")]
        scaled += ["demand_lag1", "demand_lag2"]
    steps += [
        _step("S7", "encode", ["weather"], {"strategy": "ordinal"}, "shaper:default",
              "the network embeds the weather: an ordinal code with a reserved unseen id",
              _default("encode", "if no block embedded it", {"strategy": "one-hot"})),
        _step("S8", "scale", [], {"kind": "standard", "columns": scaled}, "shaper:default",
              "the numeric columns sit on different scales",
              _default("scale", "if the tails were judged genuine", {"kind": "robust"}))]
    return steps


def node_steps(method: str) -> list:
    edges = ({"method": "key_pair", "columns": ["team_id"], "symmetric": True,
              "self_loops": False, "weight": None} if method == "key_pair" else
             {"method": "knn", "k": 6, "features": ["px", "py"], "metric": "euclidean",
              "symmetric": False, "self_loops": False, "weight": "distance"})
    return [
        _step("S1", "drop_identity", ["node_key", "team_id"], {}, "geometry:typing/node_key",
              "identifiers, not features; the team id still relates rows in S5"),
        _step("S2", "select_partition", ["churned"], {"label": "churned", "task": "binary"},
              "geometry:partitions/churned", "the community a node belongs to is the question"),
        _step("S3", "split", [], {"kind": "stratified", "by": "churned", **SPLIT},
              "shaper:default", "independent nodes with a class label",
              _default("split", "if the classes were balanced by design", {"kind": "random"})),
        _step("S4", "scale", [], {"kind": "standard", "columns": ["px", "py", "f1", "f2", "f3"]},
              "shaper:default", "the features sit on different scales",
              _default("scale", "if the tails were judged genuine", {"kind": "robust"})),
        _step("S5", "graph_edges", edges.get("columns") or edges["features"], edges,
              "analysis:F3/T3", f"the nodes are related ({method})")]


def graph_steps() -> list:
    return [
        _step("S1", "retype", ["element"], {"to": "nominal"}, "geometry:typing/element",
              "the element names a kind of atom"),
        _step("S2", "drop_identity", ["atom_key", "bonded_to"], {}, "geometry:typing/atom_key",
              "identifiers; the graph step reads them from the input in S7"),
        _step("S3", "select_partition", ["soluble"], {"label": "soluble", "task": "binary"},
              "geometry:partitions/soluble", "whether a molecule dissolves is the question"),
        _step("S4", "split", [], {"kind": "group", "by": "molecule_id", **SPLIT},
              "user:turn2", "a molecule is one sample: its atoms never straddle parts"),
        _step("S5", "encode", ["element"], {"strategy": "ordinal"}, "shaper:default",
              "the network embeds the element",
              _default("encode", "if no block embedded it", {"strategy": "one-hot"})),
        _step("S6", "scale", [], {"kind": "standard", "columns": ["charge", "degree"]},
              "shaper:default", "the features sit on different scales",
              _default("scale", "if the tails were judged genuine", {"kind": "robust"})),
        _step("S7", "graph_edges", ["atom_key", "bonded_to"],
              {"method": "key_pair", "columns": ["atom_key", "bonded_to"],
               "graph_id": "molecule_id", "symmetric": True, "self_loops": False,
               "weight": None}, "analysis:F3/T3", "atoms are bonded within a molecule")]


# ================================================================ the networks

def _bare(bid, kind, **params) -> dict:
    return {"id": bid, "type": kind, "params": params}


def window_blocks(horizon: int) -> list:
    return [_bare("B1", "embedding", columns=["weather"], dims={"weather": 3}),
            _bare("B2", "concat"),
            _bare("B3", "gru", hidden=16, num_layers=1, **{"return": "last"}),
            _bare("B4", "head", out=horizon, activation="none")]


def gcn_blocks() -> list:
    return [_bare("B1", "gcn_conv", out=16), _bare("B2", "activation", name="relu"),
            _bare("B3", "dropout", p=0.1), _bare("B4", "gcn_conv", out=16),
            _bare("B5", "activation", name="relu"), _bare("B6", "head", out=1, activation="none")]


def gin_blocks() -> list:
    return [_bare("B1", "embedding", columns=["element"], dims={"element": 3}),
            _bare("B2", "concat"), _bare("B3", "gin_conv", out=16),
            _bare("B4", "activation", name="relu"), _bare("B5", "gin_conv", out=16),
            _bare("B6", "activation", name="relu"), _bare("B7", "graph_pool", readout="sum"),
            _bare("B8", "head", out=1, activation="none")]


def floor_blocks(blocks) -> list:
    """R-A12's floor as the ladder builds it (vega-train `plan_ladder.rung_floor`): the spec's
    embedding and concat blocks, then its head, unchanged — at node level structure-blind
    (each node's own features, no edge read)."""
    kept = [copy.deepcopy(b) for b in blocks if b.get("type") in ("embedding", "concat")]
    head = next(b for b in reversed(blocks) if b.get("type") == "head")
    return kept + [copy.deepcopy(head)]


def floor_family(network: str) -> dict:
    """The floor's family node, as the ladder writes it."""
    return {"name": "linear", **choice(
        "vega:default", "the catalog's linear row is the floor every supervised campaign "
        "includes as an ablation (R-A12: structure-blind at node level)",
        [{"name": network, "when": "the network this floor is read against"}],
        "families/linear")}


# ================================================================ provenance

def choice(source, rationale, alternatives, catalog=None) -> dict:
    """{source, [catalog], rationale, alternatives} (netspec-contract §1): the catalog row
    only on a `vega:default`."""
    node = {"source": source}
    if catalog is not None and source == "vega:default":
        node["catalog"] = catalog
    node["rationale"] = rationale
    node["alternatives"] = [dict(a) for a in alternatives]
    return node


_ALTERNATIVE = {
    "embedding": lambda p: {"params": {"dims": {c: max(2, int(d) // 2) for c, d in
                                                (p.get("dims") or {}).items()}},
                            "when": "if the embedding overfits"},
    "gru": lambda p: {"params": {"hidden": int(p.get("hidden", 16)) * 2}, "when": UNDERFITS},
    "gcn_conv": lambda p: {"params": {"out": int(p.get("out", 16)) * 2}, "when": UNDERFITS},
    "gin_conv": lambda p: {"params": {"out": int(p.get("out", 16)) * 2}, "when": UNDERFITS},
    "activation": lambda p: {"params": {"name": "gelu"},
                             "when": "as an ablation of the non-linearity"},
    "dropout": lambda p: {"params": {"p": round(float(p.get("p", 0.1)) + 0.1, 2)},
                          "when": "if train is far above valid"},
    "graph_pool": lambda p: {"params": {"readout": "mean"},
                             "when": "as an ablation of the reduction"},
}


def _traced_block(block: dict, family: str, task: dict) -> dict:
    """A block with its choice: the head cites the task's source (structural: its `out` is
    the task's), concat is structure, every other block the family's catalog composition at
    the spike's small sizes (the spike proves an equality, not a score)."""
    kind, params = block["type"], block["params"]
    if kind == "head":
        row = CATALOG["tasks"][task["kind"]]["head"]
        node = choice(task["source"], f"out {params['out']} for the {task['kind']} task "
                      f"(tasks/{task['kind']}: out {row['out']}); the loss applies the link, "
                      "never the head", [])
    elif kind == "concat":
        node = choice("vega:default", "structural: the numeric features are joined to the "
                      "embeddings before the first block that reads them", [], "blocks/concat")
    else:
        rows = [f"families/{family}"] + (["defaults/recurrent_return"] if kind == "gru" else [])
        node = choice("vega:default", f"the {family} composition "
                      f"({CATALOG['families'][family]['composition']}) at the spike's small "
                      "sizes", [_ALTERNATIVE[kind](params)], rows if len(rows) > 1 else rows[0])
    return {**block, **node}


def _family(name: str) -> dict:
    alternatives = CATALOG["families"][name].get("alternatives") or []
    other = str(alternatives[0]).split(" ")[0] if alternatives else "linear"
    return {"name": name, **choice(
        f"user:spike/{name}", f"the spike's slice runs the {name} family: "
        f"{CATALOG['families'][name]['when']}",
        [{"name": other, "when": "the catalog's alternative, when the slice is proved on it"}])}


def _loss(kind: str) -> dict:
    row = CATALOG["tasks"][kind]
    name = row["default_loss"]
    other = next(n for n in row["losses_allowed"] if n != name)
    return {"name": name, "params": {}, **choice(
        "vega:default", f"the {kind} task's loss", [{"name": other,
                                                      "when": "as an ablation of the objective"}],
        f"tasks/{kind}")}


def _metrics(kind: str) -> dict:
    row = CATALOG["tasks"][kind]
    primary, secondary = row["primary"], list(row["secondary"])
    alt = (row.get("primary_alt") or {}).get("metric") or secondary[0]
    return {"primary": primary, "secondary": secondary,
            "selection": {"monitor": f"valid.{primary}",
                          "mode": CATALOG["metric_direction"][primary]},
            **choice("vega:default", f"the {kind} task row's primary and secondary metrics; "
                     "selection reads the primary on valid",
                     [{"primary": alt, "when": "if the task's balance or reading changes"}],
                     f"tasks/{kind}")}


def _regularization(blocks: list, weight_decay: float) -> list:
    out = [{"kind": "weight_decay", "value": weight_decay, **choice(
        "vega:default", "mild shrinkage; the spike asks for no more",
        [{"value": weight_decay * 10, "when": "if train is far above valid"}],
        "defaults/weight_decay")}]
    dropped = [(b["id"], float(b["params"]["p"])) for b in blocks if b["type"] == "dropout"]
    if dropped:
        out.append({"kind": "dropout", "value": dropped[0][1],
                    "where": [bid for bid, _ in dropped], **choice(
                        "vega:default", "the dropout the blocks carry, declared once",
                        [{"value": round(dropped[0][1] + 0.2, 2),
                          "when": "if the gap between train and valid stays wide"}],
                        "defaults/dropout")})
    out.append({"kind": "grad_clip", "value": float(CATALOG["defaults"]["grad_clip"]["value"]),
                **choice("vega:default", "a cheap guard against the non-finite epoch",
                         [{"value": None, "when": "as an ablation"}], "defaults/grad_clip")})
    return out


def _training(batch_size) -> dict:
    other = (256 if batch_size == "full" else int(batch_size) * 2)
    return {"batch_size": batch_size, "max_epochs": EPOCHS, "device": "cpu", "precision": "fp32",
            "num_workers": 0, "budget": {"wall_clock_min": 5}, **choice(
                "user:spike/budget", f"the spike's budget: {EPOCHS} epochs on the CPU, where "
                "the backend's determinism claim is exact; the batch is the sample's "
                "(full-batch graph view, graphs collated whole, windows in small batches)",
                [{"batch_size": other, "when": "on a campaign, by defaults/batch_size"}])}


# ================================================================ the fixture table

@dataclass(frozen=True)
class Design:
    """What one fixture is made of: its rows, its recipe, its task and sample, its network."""

    rows: object            # () -> DataFrame
    steps: object           # () -> list of recipe steps
    target: str
    task: dict
    sample: dict
    keys: tuple
    family: str
    blocks: object          # () -> list of bare blocks
    batch_size: object


def _window_sample(lag: bool) -> dict:
    return {"shape": "window", "length": 12, "stride": 1, "horizon": 3, "series_by": "station",
            "order_by": "ts", "context": "previous-part", "autoregressive": True, "season": 24,
            "source": "shape:S4", "context_source": "vega:default",
            "context_catalog": "defaults/window_context", "autoregressive_source": "vega:default",
            "autoregressive_catalog": "defaults/autoregressive",
            "series_source": "shape:S5" if lag else "user:spike/series"}


def _graph_sample(level: str) -> dict:
    graph = level == "graph"
    return {"shape": "graph", "edges": None, "level": level,
            "graph_id": "molecule_id" if graph else None, "directed": None, "self_loops": None,
            "protocol": "inductive", "protocol_source": "vega:default",
            "protocol_catalog": "defaults/graph_protocol",
            "batching": "graphs" if graph else "full", "batching_source": "vega:default",
            "batching_catalog": "defaults/graph_batching", "neighbours": None,
            "exposure": None, "source": "shape:manifest/graph"}


def _binary(label):
    return {"kind": "binary", "label": label, "classes": [0, 1], "positive": 1,
            "source": LABEL_SOURCE}


def _window_design(lag: bool) -> Design:
    return Design(rows=window_rows, steps=lambda: window_steps(lag), target="demand",
                  task={"kind": "forecast", "label": "demand", "source": LABEL_SOURCE},
                  sample=_window_sample(lag), keys=("station", "ts"), family="gru",
                  blocks=lambda: window_blocks(3), batch_size=32)


def _node_design(method: str) -> Design:
    return Design(rows=node_rows, steps=lambda: node_steps(method), target="churned",
                  task=_binary("churned"), sample=_graph_sample("node"), keys=(),
                  family="gcn", blocks=gcn_blocks, batch_size="full")


FIXTURES = {
    "window": _window_design(lag=True),
    "window-nolag": _window_design(lag=False),
    "node-keypair": _node_design("key_pair"),
    "node-knn": _node_design("knn"),
    "graph-level": Design(rows=graph_rows, steps=graph_steps, target="soluble",
                          task=_binary("soluble"), sample=_graph_sample("graph"),
                          keys=("molecule_id",), family="gin", blocks=gin_blocks, batch_size=8),
}


# ================================================================ shaping

def _sha256(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(args, stage: str, cwd=SKILLS) -> str:
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          cwd=str(cwd), timeout=900)
    if proc.returncode != 0:
        tail = " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-8:])
        raise SpikeError(f"{stage} exited {proc.returncode}: {tail}")
    return proc.stdout


def _recipe(csv: Path, frame: pd.DataFrame, steps: list, target: str, shaped: Path) -> dict:
    return {"schema": "dataset-shaper/recipe@1",
            "input": {"path": str(csv), "digest": _sha256(csv), "rows": int(len(frame)),
                      "columns": int(frame.shape[1]), "layers_read": []},
            "seed": SEED, "goal": "custom", "target": target,
            "outputs": {"dir": str(shaped), "format": "parquet", "files": [],
                        "recipe": str(shaped / "recipe.json"),
                        "manifest": str(shaped / "manifest.json"),
                        "lineage": str(shaped / "lineage.json"),
                        "reproduce": str(shaped / f"reproduce_{csv.stem}.py")},
            "steps": steps}


def shape(root: Path, design: Design) -> tuple:
    """raw.csv -> shaped/ through shape.py, then the transform-only module beside fitted.json;
    returns (raw csv, recipe path, shaped dir, transform module)."""
    frame = design.rows()
    csv = root / "raw.csv"
    frame.to_csv(csv, index=False)
    shaped = root / "shaped"
    recipe_path = root / "recipe.in.json"
    recipe = _recipe(csv, frame, design.steps(), design.target, shaped)
    recipe_path.write_text(json.dumps(recipe, indent=1), encoding="utf-8")
    _run([SHAPER, "--recipe", recipe_path, "--out-dir", shaped, "--format", "parquet"],
         "shape.py")
    module = shaped / "preprocess.py"
    _run([SHAPER, "--transform-only", "--recipe", recipe_path, "--fitted",
          shaped / "fitted.json", "--emit", module], "shape.py --transform-only")
    return csv, recipe_path, shaped, module


# ================================================================ the spec

def _encoded(manifest: dict) -> dict:
    """{column: (unseen_category_id, encode step id)} for every ordinal-encoded column."""
    out = {}
    for step in manifest.get("steps") or []:
        if step.get("op") != "encode":
            continue
        for col, entry in ((step.get("parameters_fitted") or {}).get("fitted") or {}).items():
            if isinstance(entry, dict) and entry.get("unseen_category_id") is not None:
                out[col] = (int(entry["unseen_category_id"]), step["id"])
    return out


def _typed_by(steps: list) -> dict:
    """{column: the recipe step that retyped it}."""
    return {c: s["id"] for s in steps if s.get("op") == "retype" for c in s.get("columns") or ()}


def features(manifest: dict, keys, steps=()) -> list:
    """The output schema less the label and the keys: an ordinal-encoded column is a nominal
    (int64, `cardinality = unseen_category_id + 1`, its role traced to the recipe steps that
    typed and encoded it), every other one numeric (float32)."""
    encoded, typed, out = _encoded(manifest), _typed_by(list(steps)), []
    for name, meta in (manifest.get("output_schema") or {}).items():
        if name == manifest.get("label") or name in keys:
            continue
        origin = meta.get("origin") or "input"
        if name in encoded:
            unseen, sid = encoded[name]
            out.append({"name": name, "dtype": "int64", "role": "nominal",
                        "cardinality": unseen + 1, "unseen_category_id": unseen,
                        "origin": origin, "role_source": [f"shape:{s}" for s in
                                                          (typed.get(name), sid) if s]})
            continue
        dtype = str(meta.get("dtype") or "").lower()
        if not dtype.startswith(("int", "uint", "float", "bool")):
            raise SpikeError(f"output column {name!r} ({dtype}) is neither numeric, an encoded "
                             "nominal, the label nor a key")
        out.append({"name": name, "dtype": "float32", "role": "numeric", "origin": origin})
    return out


def _sample(design: Design, manifest: dict) -> dict:
    sample = copy.deepcopy(design.sample)
    if sample["shape"] != "graph":
        return sample
    graph = manifest.get("graph") or {}
    sample.update({"edges": manifest["outputs"]["files"]["edges"]["path"],
                   "directed": bool(graph.get("directed")),
                   "self_loops": bool(graph.get("self_loops"))})
    return sample


def _pipeline_digest(pipeline: dict) -> str:
    """The canonical digest (netspec-contract §1, D1e): sorted keys, no whitespace."""
    return "sha256:" + hashlib.sha256(json.dumps(pipeline, sort_keys=True, separators=(",", ":"))
                                      .encode("utf-8")).hexdigest()


def compose(design: Design, manifest_path: Path, pipeline_path: Path, steps=()) -> dict:
    """The netspec over the shaped manifest (netspec-contract §1, every choice traced)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feats = features(manifest, design.keys, steps)
    files = manifest["outputs"]["files"]
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    task = copy.deepcopy(design.task)
    blocks = [_traced_block(b, design.family, task) for b in design.blocks()]
    metrics = _metrics(task["kind"])
    optimizer = {"name": "adamw", "params": {"lr": 0.005, "weight_decay": 0.0001},
                 **choice("user:spike/optimizer", "adamw at a step five times the catalog's, so "
                          "three epochs move the weights",
                          [{"name": "adamw", "params": {"lr": 0.001, "weight_decay": 0.0001},
                            "when": "the catalog's rate (defaults/optimizer), on a campaign"}])}
    package = pipeline["system"]["package"]
    return {
        "schema": "vega/netspec@1",
        "input": {"model": None, "layers_read": ["shape"],
                  "shaped_dir": str(manifest_path.parent), "manifest": str(manifest_path),
                  "manifest_digest": _sha256(manifest_path), "pipeline": str(pipeline_path),
                  "pipeline_digest": _pipeline_digest(pipeline)},
        "seed": SEED,
        "task": task,
        "data_contract": {"source": "shape:manifest/output_schema",
                          "parts": {"train": files["train"]["path"],
                                    "valid": files["valid"]["path"], "test": "sealed"},
                          "features": feats, "input_dim": len(feats),
                          "keys": list(design.keys), "sample": _sample(design, manifest)},
        "family": _family(design.family),
        "blocks": blocks,
        "loss": _loss(task["kind"]),
        "metrics": metrics,
        "optimizer": optimizer,
        "schedule": {"name": "none", "params": {}, **choice(
            "user:spike/schedule", "a constant rate: the spike compares two halves of one "
            "model, not a schedule", [{"name": "plateau",
                                       "when": "the catalog's default below 30 epochs"}])},
        "regularization": _regularization(blocks, optimizer["params"]["weight_decay"]),
        "early_stopping": {"monitor": metrics["selection"]["monitor"],
                           "mode": metrics["selection"]["mode"], "patience": 50,
                           "min_delta": 0.001, **choice(
                               "user:spike/budget", "never within the spike's budget: every "
                               "fit runs its epochs", [{"params": {"patience": 10},
                                                        "when": "the catalog's, on a campaign"}])},
        "training": _training(design.batch_size),
        "init": {"scheme": "kaiming_uniform", "bias": "zeros", **choice(
            "vega:default", "kaiming_uniform keeps the activations' variance at depth",
            [{"scheme": "xavier_uniform", "when": "if the non-linearity changes family"}],
            "defaults/init")},
        "backend": {"name": "pytorch", "min_version": "2.2", "hints": {}},
        "decision_families": copy.deepcopy(CATALOG["decision_families"]),
        "outputs": {"run_dir": "vega-runs/spike/", "package": package,
                    "code": [f"{package}/model/network.py", f"{package}/ingestion/data.py",
                             f"{package}/training/loop.py", f"{package}/model/metrics.py"],
                    "notebook": "engineer.ipynb"},
    }


def pipeline_for(package: str, spec_like: dict, manifest_digest: str) -> dict:
    """The forge fixture's world model (render_templates.REFERENCE_PIPELINE), its package and
    feature contract this fixture's, drawn against this shaped manifest."""
    pipeline = rt.reference_pipeline()
    pipeline["system"].update({"name": package.replace("_", "-"), "package": package})
    pipeline["input"] = {**pipeline.get("input", {}), "shape_manifest_digest": manifest_digest}
    pipeline["feature_contract"] = {"input_dim": spec_like["input_dim"],
                                    "label": spec_like["label"]}
    return pipeline


# ================================================================ the compiler

_PLANNED = re.compile(r"block '([A-Za-z0-9_]+)' is planned in backend")


def planned_blocks() -> frozenset:
    """The blocks coverage.json still lists as planned (empty from gate A on)."""
    cov = json.loads(COVERAGE.read_text(encoding="utf-8"))
    return frozenset((cov.get("planned") or {}).get("blocks") or ())


def compile_check(spec: dict, manifest: dict, pipeline: dict) -> tuple:
    """The compiler's core over the spec and its chain (no model HTML: every source the spike
    cites resolves without one). Returns the refusals recorded under A7(a) — rule B2's for a
    block coverage.json still lists as planned; any other ERROR raises, naming it."""
    markers = cs.compile_spec(spec, model_html=None, pipeline=pipeline, manifest=manifest)
    planned, notes, refused = planned_blocks(), [], []
    for m in markers.errors:
        hit = _PLANNED.search(m.message) if m.rule == "B2" else None
        (notes if hit and hit.group(1) in planned else refused).append(m)
    if refused:
        said = "; ".join(f"{m.rule}{f' {m.where}' if m.where else ''}: {m.message}"
                         for m in refused)
        raise SpikeError(f"compile_spec refuses the spec: {said}")
    return tuple(sorted({f"{m.rule}: {m.message}" for m in notes}))


# ================================================================ the fixture

def lag_depth(recipe: dict) -> int:
    """The largest lag of the recipe's lag steps (a step without `lags` lags by 1); 0 when the
    recipe does not lag."""
    depth = 0
    for step in recipe.get("steps") or []:
        if step.get("op") == "lag":
            depth = max([depth, *(int(k) for k in ((step.get("params") or {}).get("lags")
                                                 or [1]))])
    return depth


def package_name(name: str) -> str:
    return "spike_" + name.replace("-", "_")


def missing_templates(spec: dict, pipeline: dict, templates=None) -> list:
    """The templates D1 §4.1's table renders for this spec that `templates` (default: the
    live v2.3 set) does not hold — the set lands package by package until gate A."""
    emit, templates = rt.load_emit(), Path(templates or TEMPLATES)
    layout = emit.v23_layout(emit.plan(spec, pipeline))
    return sorted(name for name in layout if not (templates / name).is_file())


def emit_package(out: Path, spec: dict, pipeline: dict, how: str = "full",
                 templates=None) -> dict:
    """The package, by the v2.3 emitter: `emit(spec, pipeline, out, generation="v2.3")`,
    refused — naming the files — when the template set lacks one D1 §4.1's table renders for
    this spec. The file-by-file fallback (`how="auto"` before gate A) is retired (A29, B18):
    `coverage.planned` is empty and the set is complete, so a template that goes missing must
    fail the spike, not shrink it. `how` stays for the callers that pass it: `full`, or `auto`,
    which now means the same. Returns `{"emission": "full", "missing": []}`."""
    if how not in ("full", "auto"):
        raise SpikeError(f"emit: how is 'full' ('auto' means the same since gate A), "
                         f"not {how!r}")
    templates = Path(templates or TEMPLATES)
    missing = missing_templates(spec, pipeline, templates)
    if missing:
        raise SpikeError(f"emit (generation v2.3): the template set lacks "
                         f"{', '.join(missing)} (D1 §4.1)")
    try:
        rt.render(templates, out, pipeline, spec, full=True, campaign_root=str(out.parent))
    except rt.RenderError as exc:
        raise SpikeError(f"emit (generation v2.3): {exc}") from exc
    return {"emission": "full", "missing": []}


def make(name: str, work, emit: str = "full") -> Fixture:
    """Make fixture `name` under `work/<name>/` (replacing what was there); return it. `emit`
    is `emit_package`'s `how`."""
    if name not in FIXTURES:
        raise SpikeError(f"no spike fixture {name!r}; known: {', '.join(FIXTURES)}")
    design, root = FIXTURES[name], Path(work).resolve() / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    csv, recipe_path, shaped, module = shape(root, design)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    manifest_path = shaped / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package = package_name(name)
    pipeline_path = root / "pipeline.json"
    first = features(manifest, design.keys, recipe["steps"])
    task = design.task.get("base_kind") or design.task.get("kind")     # the effective kind
    pipeline = rt.shape_pipeline(pipeline_for(package, {"input_dim": len(first),
                                                       "label": design.target},
                                              _sha256(manifest_path)),
                                 str(design.sample.get("shape") or "row"), task)
    pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
    spec = compose(design, manifest_path, pipeline_path, recipe["steps"])
    notes = compile_check(spec, manifest, pipeline)
    spec_path = root / "netspec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    emitted = emit_package(root / "package", spec, pipeline, emit)
    fixture = Fixture(name=name, root=str(root), raw=str(csv), recipe=str(recipe_path),
                      shaped=str(shaped), manifest=str(manifest_path),
                      fitted=str(shaped / "fitted.json"), transform=str(module),
                      pipeline=str(pipeline_path), spec=str(spec_path),
                      package_dir=str(root / "package" / package), package=package,
                      label=design.target, keys=tuple(design.keys),
                      sample=spec["data_contract"]["sample"], lag_depth=lag_depth(recipe),
                      emission=emitted["emission"], missing=tuple(emitted["missing"]),
                      compile_notes=notes)
    (root / "fixture.json").write_text(json.dumps(fixture.as_dict(), indent=2) + "\n",
                                       encoding="utf-8")
    return fixture


def load(work, name: str) -> Fixture:
    """A fixture `make` wrote earlier under `work`."""
    path = Path(work) / name / "fixture.json"
    if not path.is_file():
        raise SpikeError(f"{path} does not exist; make the fixture first")
    return Fixture.from_dict(json.loads(path.read_text(encoding="utf-8")))


def make_all(work, names=None, emit: str = "full") -> dict:
    return {n: make(n, work, emit) for n in (names or FIXTURES)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work")
    ap.add_argument("--fixture", default="all",
                    help=f"a comma list of {', '.join(FIXTURES)}, or all (the default)")
    ap.add_argument("--emit", choices=("full", "auto"), default="full",
                    help="the v2.3 emitter, or nothing when the template set lacks a D1 §4.1 "
                         "file; auto is accepted and means the same (its file-by-file "
                         "fallback was retired after gate A)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    names = list(FIXTURES) if a.fixture == "all" else [n.strip() for n in a.fixture.split(",")]
    try:
        made = make_all(a.work, names, a.emit)
    except SpikeError as exc:
        print(f"ERROR: {exc}")
        return 2
    if a.json:
        print(json.dumps({n: f.as_dict() for n, f in made.items()}, indent=2))
        return 0
    for n, f in made.items():
        print(f"OK: {n}: {f.package} emitted ({f.emission}) under {f.package_dir} "
              f"({f.sample['shape']}, lag depth {f.lag_depth})")
        for note in f.compile_notes:
            print(f"WARN: {n}: compile_spec {note} (recorded under A7(a) until gate A)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
