"""
What a netspec is composed over (lanekit.spec): the forge catalogs, the task a head answers to,
the data contract a shaped manifest implies, and the campaign budget a lane runs under.

    CATALOG, COVERAGE, TASK_ROWS                  # vega-forge's catalog.json and coverage.json
    target = Target.of(lane)                      # the task kind, base, classes, horizon, quantiles
    contract = Contract.from_manifest(manifest, lane)   # features, keys, label, parts, targets
    budget = Budget.read(lanes_file)              # lanes.json defaults: max_epochs, patience

The catalogs are read at import from the skills tree this lanekit sits in
(`contracts.SKILLS_ROOT`), never from a tree a caller names: a spec is composed against the
catalog of the tree that composes it, and `compile_spec.py` of the same tree checks it. The
budget is the one the caller passes (`Budget.read(path)`), the tree's own `lanes.json` when it
passes none.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from . import contracts
from .recipes import effective_task

__all__ = ["SCHEMA", "CATALOG", "COVERAGE", "TaskRow", "TASK_ROWS", "SpecError", "Target", "Contract",
           "Budget", "head_out", "batch_rule", "lane_defaults", "OPTIMIZER_PARAMS", "INIT_DEFAULTS",
           "CONV_TYPES", "WALL_CLOCK_MIN", "raw_of"]

SCHEMA = "vega/netspec@1"
WALL_CLOCK_MIN = 2                              # D6 §5.3: a lane's run is bounded in minutes
OPTIMIZER_PARAMS = {"adamw": {"lr": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999]},
                    "adam": {"lr": 0.001}, "sgd": {"lr": 0.01, "momentum": 0.9},
                    "radam": {"lr": 0.001}, "rmsprop": {"lr": 0.001}}
INIT_DEFAULTS = ("kaiming_uniform", "xavier_uniform")      # defaults/init (relu / tanh family)
CONV_TYPES = ("gcn_conv", "sage_conv", "gat_conv", "gin_conv")


class SpecError(ValueError):
    """The lane, its manifest and the catalog do not make a spec; the message says what."""


def raw_of(lane: Any) -> Mapping[str, Any]:
    """A lane's declaration: a registry `Lane`'s raw mapping, or the mapping itself."""
    return lane.raw if hasattr(lane, "raw") else lane


# --------------------------------------------------------------------------- the catalogs
def _read(rel: str) -> dict:
    return json.loads((contracts.SKILLS_ROOT / rel).read_text(encoding="utf-8"))


CATALOG: Mapping[str, Any] = MappingProxyType(_read(contracts.FORGE_CATALOG))
COVERAGE: Mapping[str, Any] = MappingProxyType(_read(contracts.BACKEND_COVERAGE))


@dataclass(frozen=True)
class TaskRow:
    """One row of the catalog's task table (family-catalog.md §1)."""
    kind: str
    head_out: str
    head_activation: str
    losses_allowed: tuple[str, ...]
    default_loss: str
    primary: str
    secondary: tuple[str, ...]
    primary_alt: Mapping[str, Any] | None

    @classmethod
    def of(cls, kind: str, row: Mapping[str, Any]) -> "TaskRow":
        return cls(kind, str(row["head"]["out"]), row["head"]["activation"],
                   tuple(row.get("losses_allowed") or ()), row.get("default_loss"),
                   row.get("primary"), tuple(row.get("secondary") or ()), row.get("primary_alt"))


TASK_ROWS: Mapping[str, TaskRow] = MappingProxyType(
    {kind: TaskRow.of(kind, row) for kind, row in CATALOG["tasks"].items()})


# --------------------------------------------------------------------------- the task
@dataclass(frozen=True)
class Target:
    """What the head and the loss answer to: the task kind, its base, the classes, the horizon
    and the quantiles (from the loss), and the anomaly objective (from the loss's name)."""
    kind: str
    base: str | None = None
    n_classes: int | None = None
    horizon: int | None = None
    quantiles: int = 0
    loss: str | None = None

    @property
    def effective(self) -> str:
        return self.base if self.kind.startswith("sequence-") and self.base else self.kind

    @classmethod
    def of(cls, lane: Any) -> "Target":
        raw = raw_of(lane)
        target = raw.get("target") or {}
        loss = raw.get("loss") or {}
        quantiles = (loss.get("params") or {}).get("quantiles") or target.get("quantiles") or ()
        return cls(raw["task"], target.get("base"), len(target.get("classes") or ()) or None,
                   (raw.get("window") or {}).get("horizon"),
                   len(quantiles) if loss.get("name") == "quantile" else 0, loss.get("name"))


def head_out(target: Target, contract: "Contract", z: int | None = None) -> int:
    """The head's `out` the task row forces (T1b as BUILD-PLAN D1 §14.1 states it)."""
    kind, q = target.effective, max(target.quantiles, 1)
    if kind == "binary":
        return 1
    if kind in ("multiclass", "multilabel"):
        return int(target.n_classes)
    if kind == "ordinal":
        return int(target.n_classes) - 1
    if kind == "regression":
        return q
    if kind == "forecast":
        return int(target.horizon) * q
    if kind in ("unsupervised", "anomaly"):
        if target.loss == "svdd":
            return int(z or 8)
        return contract.recon_dim
    raise SpecError(f"no head rule for task {kind!r}")


# --------------------------------------------------------------------------- the data contract
@dataclass(frozen=True)
class Contract:
    """The data contract a spec is composed over: the features (numeric float32, embedded
    nominals int64 with their cardinality), the keys, the label and its dtype, the parts, the
    training targets, and the manifest's graph block and edges part."""
    shape: str
    features: tuple[Mapping[str, Any], ...]
    keys: tuple[str, ...] = ()
    label: str | None = None
    label_is_int: bool = True
    parts: Mapping[str, str] = field(default_factory=dict)
    rows: Mapping[str, int] = field(default_factory=dict)
    graph: Mapping[str, Any] | None = None
    edges: str | None = None
    targets: int = 0
    balance: Mapping[str, float] | None = None

    @property
    def input_dim(self) -> int:
        return len(self.features)

    @property
    def numeric(self) -> tuple[str, ...]:
        return tuple(f["name"] for f in self.features if f["role"] == "numeric")

    @property
    def nominals(self) -> dict[str, int]:
        return {f["name"]: int(f["cardinality"]) for f in self.features if f["role"] == "nominal"}

    @property
    def recon_dim(self) -> int:
        """n_numeric + the sum of the embedded nominals' cardinalities (D1 §3.4)."""
        return len(self.numeric) + sum(self.nominals.values())

    @classmethod
    def synthetic(cls, shape: str = "row", n_numeric: int = 4, nominals: Mapping[str, int] | None = None,
                  level: str | None = None, targets: int = 256) -> "Contract":
        """A contract with no chain behind it, for the unit tier's specs."""
        feats = [{"name": f"x{i}", "dtype": "float32", "role": "numeric", "origin": "synthetic"}
                 for i in range(1, n_numeric + 1)]
        for name, card in (nominals or {}).items():
            feats.append({"name": name, "dtype": "int64", "role": "nominal", "cardinality": int(card),
                          "unseen_category_id": int(card) - 1, "origin": "synthetic"})
        graph = None
        keys: tuple[str, ...] = ()
        if shape == "graph":
            graph = {"level": level or "node", "graph_id": "g" if level == "graph" else None,
                     "directed": False, "self_loops": False}
            keys = ("g",) if level == "graph" else ()
        elif shape == "window":
            keys = ("t",)
        return cls(shape, tuple(feats), keys, "y", True, {"train": "train.parquet", "valid": "valid.parquet"},
                   {"train": targets, "valid": targets // 4}, graph,
                   "edges.parquet" if shape == "graph" else None, targets)

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], lane: Any) -> "Contract":
        """The contract a lane's shaped manifest implies (lanekit.spec's docstring)."""
        raw = raw_of(lane)
        schema = manifest.get("output_schema") or {}
        if not isinstance(schema, Mapping) or not schema:
            raise SpecError("the shaped manifest carries no output_schema")
        label = manifest.get("label")
        keys = _keys(raw)
        missing = [k for k in keys if k not in schema]
        if missing:
            raise SpecError(f"key column(s) {missing} are not in the shaped output schema")
        encoded = _encoded(manifest)
        features = [_feature(name, meta, encoded) for name, meta in schema.items()
                    if name != label and name not in keys]
        files = (manifest.get("outputs") or {}).get("files") or {}
        parts = {p: files[p]["path"] for p in ("train", "valid") if p in files}
        rows = {p: int(files[p].get("rows") or 0) for p in ("train", "valid", "test") if p in files}
        label_dtype = str((schema.get(label) or {}).get("dtype", "")) if label else ""
        contract = cls(raw["shape"], tuple(features), keys, label, _is_int(label_dtype), parts, rows,
                       manifest.get("graph"), (files.get("edges") or {}).get("path"))
        stats = _PartStats.read(contract, raw)
        return replace(contract, targets=stats.targets, balance=stats.balance)


def _keys(raw: Mapping[str, Any]) -> tuple[str, ...]:
    if raw["shape"] == "window":
        w = raw.get("window") or {}
        return tuple(k for k in (w.get("series_by"), w.get("order_by")) if k)
    if raw["shape"] == "graph":
        gid = (raw.get("graph") or {}).get("graph_id")
        return (gid,) if gid else ()
    return ()


def _encoded(manifest: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    """{column: (unseen_category_id, encode step id)} for every ordinal-encoded column."""
    out: dict[str, tuple[int, str]] = {}
    for step in manifest.get("steps") or []:
        if step.get("op") != "encode":
            continue
        fitted = ((step.get("parameters_fitted") or {}).get("fitted")) or {}
        for col, entry in fitted.items():
            if isinstance(entry, Mapping) and entry.get("unseen_category_id") is not None:
                out[col] = (int(entry["unseen_category_id"]), step.get("id"))
    return out


def _is_int(dtype: str) -> bool:
    return dtype.lower().startswith(("int", "uint", "bool"))


def _is_numeric(dtype: str) -> bool:
    return dtype.lower().startswith(("int", "uint", "float", "bool"))


def _feature(name: str, meta: Mapping[str, Any], encoded: Mapping[str, tuple[int, str]]) -> dict:
    origin = meta.get("origin") or "input"
    if name in encoded:
        unseen, step = encoded[name]
        return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": unseen + 1,
                "unseen_category_id": unseen, "origin": origin,
                "role_source": [f"geometry:typing/{name}", f"shape:{step}"]}
    dtype = str(meta.get("dtype") or "")
    if not _is_numeric(dtype):
        raise SpecError(f"output column {name!r} ({dtype}) is neither numeric, an encoded nominal, "
                        "the label nor a key — the recipe left it unshaped")
    return {"name": name, "dtype": "float32", "role": "numeric", "origin": origin}


@dataclass(frozen=True)
class _PartStats:
    """What the composer reads off the train part: the training targets (rows, windows, seed
    nodes or graphs) and the class balance."""
    targets: int
    balance: Mapping[str, float] | None

    @classmethod
    def read(cls, contract: Contract, raw: Mapping[str, Any]) -> "_PartStats":
        import pandas as pd
        from pyarrow import fs      # Arrow's own reader: no Python in its threads at exit (A40)

        rows = int(contract.rows.get("train") or 0)
        path = contract.parts.get("train")
        frame = (pd.read_parquet(path, filesystem=fs.LocalFileSystem())
                 if path and Path(path).is_file() else None)
        targets = rows
        if raw["shape"] == "window" and frame is not None:
            w = raw["window"]
            series = w.get("series_by")
            n_series = int(frame[series].nunique()) if series and series in frame else 1
            targets = rows - n_series * ((int(w["length"]) - 1) + int(w.get("horizon") or 0))
        elif raw["shape"] == "graph" and contract.graph and contract.graph.get("graph_id") and frame is not None:
            targets = int(frame[contract.graph["graph_id"]].nunique())
        balance = None
        if frame is not None and contract.label in (frame.columns if frame is not None else ()) \
                and effective_task(raw) in ("binary", "multiclass", "ordinal", "anomaly"):
            shares = frame[contract.label].astype(str).value_counts(normalize=True).sort_index()
            balance = {str(k): round(float(v), 4) for k, v in shares.items()}
        return cls(max(targets, 0), balance)


# --------------------------------------------------------------------------- the budget
def lane_defaults(path: Path | None = None) -> Mapping[str, Any]:
    """A lanes file's `defaults` (the tree's own `lanes.json` when no path is given)."""
    from .registry import LANES_FILE, load_document
    return load_document(Path(path) if path else LANES_FILE)["defaults"]


@dataclass(frozen=True)
class Budget:
    """The campaign a lane's spec is sized for (`lanes.json` defaults, D6 §5.3), or the lane's own
    `budget` override {max_epochs, patience} with the `reason` it must carry (A38); the time
    limits are the defaults' whichever applies."""
    max_epochs: int = 8
    patience: int = 3
    wall_clock_min: int = WALL_CLOCK_MIN
    reason: str | None = None

    def for_lane(self, raw: Mapping[str, Any]) -> "Budget":
        """This budget, or the lane's own override when `raw` carries one (A38)."""
        own = raw.get("budget")
        if not own:
            return self
        if not str(own.get("reason") or "").strip():
            raise SpecError(f"lane {raw.get('id')}: a budget override carries its reason (A38)")
        return Budget(int(own["max_epochs"]), int(own["patience"]), self.wall_clock_min,
                      str(own["reason"]).strip())

    def phrase(self) -> str:
        """How a rationale names the budget: the lane budget, or the lane's own and why."""
        return "the lane budget" if not self.reason else f"the lane's own budget ({self.reason})"

    @classmethod
    def of(cls, defaults: Mapping[str, Any]) -> "Budget":
        return cls(int(defaults.get("max_epochs", 8)), int(defaults.get("patience", 3)))

    @classmethod
    def read(cls, lanes_file: Path | None = None) -> "Budget":
        """The budget of the lanes file a lane was read from (the tree's own when None)."""
        return cls.of(lane_defaults(lanes_file))


def batch_rule(targets: int) -> int:
    """defaults/batch_size: clip(targets / 8, 16, 512) rounded to a power of two, never above
    the targets."""
    x = min(max(targets / 8.0, 16.0), 512.0)
    b = int(2 ** round(math.log2(x)))
    return max(1, min(b, targets)) if targets else b
