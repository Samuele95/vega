"""
page_derive — what `page.json` derives from the chain's own records, and the catalogs it reads
them through: the base module of `page_config` (W06) and of its gate `page_check` (BD2).

- **`Catalogs`** holds every table the page config reads and restates nowhere: the vega-forge
  catalog (tasks, sequence bases, graph level, `metric_direction`, `inspection.links`, block
  defaults), the backend's `coverage.json` (the metric names, the language) and
  `explain-catalog.json` (with the path and digest `page.json.explain` records), vega-show's own
  catalog, and the installed page template's identity (`TEMPLATE.json`, None until W15 writes it).
- **`METRIC_BOUNDS`** is D4 §14.1's metric table: the range the page draws each metric of
  `coverage.json.metrics` in, `(lo, hi)` with `hi` None when the metric has no upper bound. `r2`
  has no lower bound; `-1` is its drawing floor (a model twice as far off as the mean), because
  `vega/page@1` requires a number there. A metric the catalog gives no direction (`none`) has no
  bounds: it is never a primary.
- **The lane** (D4 §5, R-C1): the view key from vega-show's `view_keys`, the reading from the
  effective kind (a `sequence-*` kind reads as its `task.base_kind`) and the loss (`svdd` reads
  a distance, the other anomaly losses a reconstruction), the link from the one vocabulary of the
  vega-forge catalog's `inspection.links`, the goal from its `goal_map`.
- **The recipe** (D4 §14.1): the shaper steps whose outputs reach the network's inputs, in the
  order the executor ran them, each named by the page's closed op set with the number of network
  inputs it touched.

Pure functions over dicts; a refusal is `ShowError` (the input is fixed upstream).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

from vega_show_core import CATALOG, HERE, ShowError, digest_file

__all__ = ["Catalogs", "METRIC_BOUNDS", "LANE_SOURCE", "walk_blocks", "block_types",
           "activation_names", "lane_record", "lane_fields", "modality_of", "recipe_summary"]

SKILLS = HERE.parent.parent
LANE_SOURCE = "netspec:task, netspec:data_contract.sample, netspec:family, netspec:loss"
RECIPE_SOURCE = "shape:manifest.steps"

METRIC_BOUNDS: Mapping[str, tuple | None] = MappingProxyType({
    "roc_auc": (0, 1), "pr_auc": (0, 1), "pr_auc_macro": (0, 1), "log_loss": (0, None),
    "brier": (0, 2), "ece": (0, 1), "accuracy": (0, 1), "balanced_accuracy": (0, 1),
    "precision": (0, 1), "recall": (0, 1), "f1": (0, 1), "f1_macro": (0, 1), "f1_micro": (0, 1),
    "f1_weighted": (0, 1), "f1_per_class": (0, 1), "mcc": (-1, 1), "specificity": (0, 1),
    "top_k": (0, 1), "hamming": (0, 1), "subset_accuracy": (0, 1), "rmse": (0, None),
    "mae": (0, None), "median_ae": (0, None), "r2": (-1, 1), "mape": (0, None),
    "spearman": (-1, 1), "residual_quantiles": None, "mae_ordinal": (0, None), "qwk": (-1, 1),
    "reconstruction_error": (0, None), "per_feature_error": (0, None),
    "anomaly_score_mean": (0, None), "precision_at_k": (0, 1), "recall_at_k": (0, 1),
    "smape": (0, 2), "mase": (0, None), "per_horizon_error": (0, None),
})

# R1–R12 of the serving readings, read for the page: the effective kind → the reading whose link
# the vega-forge catalog names (a quantile regression shares regression's `identity` link)
_READING_OF = MappingProxyType({
    "binary": "binary", "multiclass": "multiclass", "multilabel": "multilabel",
    "ordinal": "ordinal", "regression": "regression", "forecast": "forecast",
    "unsupervised": "reconstruction"})
_ANOMALY_READING = MappingProxyType({"reconstruction": "anomaly-reconstruction",
                                     "distance": "anomaly-svdd"})
_DISTANCE_LOSSES = frozenset({"svdd"})


def _load(path: Path, what: str) -> Mapping:
    try:
        return MappingProxyType(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShowError(f"{what} could not be read ({path}): {exc}") from exc


@dataclass(frozen=True)
class Catalogs:
    """The read-only tables `page.json` is assembled from (D4 §14.2's value object)."""

    forge: Mapping
    coverage: Mapping
    explain: Mapping
    explain_ref: str
    explain_digest: str
    template: Mapping | None = None
    show: Mapping = CATALOG

    @classmethod
    def load(cls, skills: Path | None = None, backend: str = "pytorch") -> "Catalogs":
        """The catalogs installed beside vega-show (`<skills>/vega-forge/…`), for `backend`."""
        root = Path(skills) if skills is not None else SKILLS
        forge = root / "vega-forge" / "scripts"
        explain = forge / "backends" / backend / "explain-catalog.json"
        return cls(_load(forge / "catalog.json", "the vega-forge catalog"),
                   _load(forge / "backends" / backend / "coverage.json", "the backend's coverage.json"),
                   _load(explain, "the backend's explain-catalog.json"),
                   explain.relative_to(root).as_posix(), digest_file(explain),
                   _template_identity(root / "vega-show"))

    def direction(self, metric: str) -> str | None:
        return (self.forge.get("metric_direction") or {}).get(metric)


def _template_identity(skill: Path) -> Mapping | None:
    """`{version, manifest_digest}` of the installed `TEMPLATE.json`, or None before it exists."""
    spec = CATALOG["template"]
    path = skill / spec["dir"] / spec["manifest"]
    if not path.is_file():
        return None
    doc = _load(path, "the page template's TEMPLATE.json")
    return MappingProxyType({"version": doc.get("version"), "manifest_digest": digest_file(path)})


# --------------------------------------------------------------------------- blocks

def walk_blocks(blocks) -> Iterator[Mapping]:
    """Every block in forward order, a composite before its `params.inner` blocks."""
    for block in blocks or ():
        if isinstance(block, Mapping):
            yield block
            inner = (block.get("params") or {}).get("inner")
            if isinstance(inner, (list, tuple)):
                yield from walk_blocks(inner)


def block_types(blocks) -> list[str]:
    """The distinct block types, in the forward order of their first block."""
    seen: dict[str, None] = {}
    for block in walk_blocks(blocks):
        seen.setdefault(str(block.get("type")), None)
    return list(seen)


def activation_names(blocks, forge: Mapping) -> list[str]:
    """The activation names the blocks apply: an `activation` block's `name`, a layer's
    `activation` or `nonlinearity` parameter, else the catalog default of that parameter."""
    names: dict[str, None] = {}
    for block in walk_blocks(blocks):
        params, kind = block.get("params") or {}, block.get("type")
        defaults = ((forge.get("blocks") or {}).get(kind) or {}).get("defaults") or {}
        keys = ("name",) if kind == "activation" else ("activation", "nonlinearity")
        for key in keys:
            value = params.get(key, defaults.get(key))
            if isinstance(value, str):
                names.setdefault(value, None)
    return list(names)


# --------------------------------------------------------------------------- lane

def lane_fields(kind: str, base: str | None, score_source: str | None, catalogs: Catalogs) -> dict:
    """`view_key`, `link` and `goal` of a task kind, its base and (anomaly) its score source."""
    view = catalogs.show["view_keys"].get(kind)
    view_key = view.get(base) if isinstance(view, Mapping) else view
    effective = base if kind.startswith("sequence-") else kind
    reading = _ANOMALY_READING.get(score_source) if kind == "anomaly" else _READING_OF.get(effective)
    links = [name for name, spec in (catalogs.forge["inspection"]["links"]).items()
             if reading in (spec.get("readings") or {})]
    if view_key is None or reading is None or len(links) != 1:
        raise ShowError(f"task {kind!r} (base {base!r}, score source {score_source!r}) has no "
                        "view key and no single link in the catalogs (R-C1)")
    goals = {v: k for k, v in catalogs.forge["goal_map"].items() if v}
    return {"view_key": view_key, "link": links[0], "goal": goals.get(kind, "predict")}


def lane_record(netspec: Mapping, catalogs: Catalogs) -> dict:
    """`page.json.lane` from the netspec: task, base, view key, shape, level, link, family."""
    task, forge = netspec.get("task") or {}, catalogs.forge
    kind = task.get("kind")
    if kind not in forge["tasks"]:
        raise ShowError(f"netspec task.kind {kind!r} is not a vega-forge catalog task")
    base = task.get("base_kind") if str(kind).startswith("sequence-") else None
    if kind.startswith("sequence-") and forge["sequence_map"].get(base) != kind:
        raise ShowError(f"netspec task.base_kind {base!r} is not a base of {kind} (sequence_map)")
    loss = (netspec.get("loss") or {}).get("name")
    source = ("distance" if loss in _DISTANCE_LOSSES else "reconstruction") if kind == "anomaly" else None
    shape, level = _sample(netspec, base or kind, forge)
    family = (netspec.get("family") or {}).get("name")
    if not isinstance(family, str) or not family:
        raise ShowError("netspec family.name is missing")
    fields = lane_fields(kind, base, source, catalogs)
    return {"task": kind, "base": base, "view_key": fields["view_key"], "shape": shape,
            "level": level, "link": fields["link"], "family": family, "goal": fields["goal"],
            "score_source": source, "source": LANE_SOURCE}


def _sample(netspec: Mapping, effective: str, forge: Mapping) -> tuple[str, str | None]:
    sample = (netspec.get("data_contract") or {}).get("sample") or {"shape": "row"}
    shape = sample.get("shape")
    if shape not in forge["sample_shapes"]:
        raise ShowError(f"netspec data_contract.sample.shape {shape!r} is not a sample shape")
    if shape != "graph":
        return shape, None
    level = sample.get("level") or "node"
    if level == "graph" and effective not in forge["graph_level"]["tasks"]:
        raise ShowError(f"a {effective} task has no graph-level reading (graph_level.tasks)")
    return shape, level


def modality_of(shape: str, analysis: Mapping | None) -> str:
    """R-E4: `spatial-row` for a row sample whose analysis names a spatial context."""
    spatial = ((analysis or {}).get("context") or {}).get("spatial")
    return "spatial-row" if shape == "row" and spatial is not None else shape


# --------------------------------------------------------------------------- recipe

_SKIPPED = frozenset({"retype", "drop_identity", "drop_constant", "dedupe", "parse_datetime",
                      "parse_geometry", "orient_cycle", "drop_derived", "select_partition",
                      "drop_leakage", "keep_columns", "split", "select_features",
                      "spatial_reproject"})
# steps that rewrite their own columns (and may add some) → the page's op
_IN_PLACE = MappingProxyType({"encode": "encode", "clip": "clip", "winsorize": "clip"})
# steps whose network inputs are only the columns they add (the sources stay as they were)
_ADDING = MappingProxyType({"lag": "window", "project": "project", "datetime_expand": "other",
                            "spatial_distance": "other", "spatial_join": "other",
                            "spatial_grid": "other", "spatial_features": "other"})
_LOG_KINDS = frozenset({"log1p", "signed-log1p", "log"})


def recipe_summary(shape_record: Mapping, netspec: Mapping) -> dict:
    """`page.json.recipe`: one op per shaper step (two for an imputation with indicators)
    whose outputs reach the network's inputs, with how many inputs it touched."""
    manifest, recipe = shape_record.get("manifest") or {}, shape_record.get("recipe") or {}
    planned = {s.get("id"): s for s in recipe.get("steps") or () if isinstance(s, Mapping)}
    ran = [s for s in manifest.get("steps") or () if isinstance(s, Mapping)] or list(planned.values())
    features = (netspec.get("data_contract") or {}).get("features") or ()
    inputs = {f.get("name"): f.get("role") for f in features if isinstance(f, Mapping)}
    graph = ((netspec.get("data_contract") or {}).get("sample") or {}).get("shape") == "graph"
    ops = []
    for step in ran:
        planned_step = planned.get(step.get("id")) or step
        for op, columns in _page_ops(planned_step, step, inputs):
            touched = len(set(columns) & set(inputs))
            if touched or (op == "graph_edges" and graph):
                ops.append({"op": op, "columns": touched, "step": str(step.get("id"))})
    known = set(CATALOG["page"]["recipe_ops"])
    if any(o["op"] not in known for o in ops):
        raise ShowError(f"recipe ops outside the page's closed set: {[o['op'] for o in ops]}")
    return {"ops": ops, "source": RECIPE_SOURCE}


def _page_ops(planned: Mapping, ran: Mapping, inputs: Mapping) -> list[tuple[str, list]]:
    """(page op, the columns it touched) of one shaper step; [] when no value reaches the net."""
    op, params = planned.get("op") or ran.get("op"), planned.get("params") or {}
    columns = [c for c in planned.get("columns") or () if isinstance(c, str)]
    added = _added(ran)
    if op in _SKIPPED:
        return []
    if op == "impute":
        return [("impute", columns), ("indicator", added)]
    if op == "transform":
        return [("log1p" if params.get("kind") in _LOG_KINDS else "other", columns)]
    if op == "scale":
        kind = params.get("kind")
        return [] if kind in (None, "none") else [(f"{kind}_scale", _scaled(params, ran, columns, inputs))]
    if op == "graph_edges":
        return [("graph_edges", [])]
    if op == "bin":
        return [("bin", added + (columns if params.get("replace") else []))]
    if op in _ADDING:
        return [(_ADDING[op], added)]
    return [(_IN_PLACE.get(op, "other"), columns + added)]


def _added(ran: Mapping) -> list[str]:
    fitted = ran.get("parameters_fitted") or {}
    added = ran.get("columns_added") or (fitted.get("columns_added") if isinstance(fitted, Mapping) else None)
    return [c for c in added or () if isinstance(c, str)]


def _scaled(params: Mapping, ran: Mapping, columns: list, inputs: Mapping) -> list:
    """The columns a scale step scaled: its list, else the fitted statistics' columns, else
    every numeric network input (`columns: "numeric"`)."""
    if isinstance(params.get("columns"), list):
        return list(params["columns"])
    fitted = ((ran.get("parameters_fitted") or {}).get("fitted"))
    if isinstance(fitted, Mapping) and fitted:
        return list(fitted)
    return columns or [name for name, role in inputs.items() if role == "numeric"]


def key_pattern(text: str) -> bool:
    """A strings key the page can look up (`views.off.no_route`)."""
    return re.fullmatch(r"[a-z][a-z0-9_.]*", text) is not None
