"""
The netspec composer: a lane's declared network, over the data its fixture shaped, as a
`vega/netspec@1` whose every choice carries a resolving source (D6 §5.3, BUILD-PLAN R-A15).

    spec = compose(lane, manifest, fitted, model_digest, pipeline_path)      # netspec.json
    spec = Composer(names, Budget.read(lanes_file)).compose(lane, manifest, fitted, digest, path)
    spec = family_spec("gru", "forecast", Contract.synthetic("window"))       # a family's default
    spec = minimal_block_spec("gat_conv")                                     # one block's smallest

`compose` fills the netspec the way `make_fixture.py` fills the orders one — `input` with its
digests, `seed`, `task`, the `data_contract` read from the shaped manifest's output schema
(numeric columns float32; an ordinal-encoded nominal int64 with `cardinality =
unseen_category_id + 1` from its encode step), `input_dim`, `keys`, `sample`, `class_balance` —
then the lane's family, blocks, loss, optimizer, schedule and init, the task row's metrics with
`selection.monitor = valid.<primary>`, the regularisation, early stopping and training budget,
`backend`, `decision_families` and `outputs`.

R-A15, the P1 fields per shape:

- window: `length`, `stride`, `horizon` (forecast only), `series_by`, `order_by`, `context`,
  `autoregressive` (`false` on any task but forecast), `season`, each sourced;
- graph: `edges` (the manifest's edges part), `level`, `graph_id`, `directed` and `self_loops`
  copied from `shape:manifest/graph`, `protocol`, `batching`, `neighbours` iff `sampled`,
  `exposure` with a `user:` source iff transductive;
- `training.batch_size` is `"full"` iff `batching == "full"`, else the catalog's rule over the
  training targets (rows, windows, seed nodes or graphs);
- `data_contract.keys` are the non-null `series_by`, `order_by` and `graph_id`; they are not
  features, and `input_dim` counts the output schema minus the label minus the keys.

What a composition reads besides its arguments: the budget (`max_epochs`, `patience`; a lane's
own `budget` override with its `reason` when it carries one, A38) is the
`Composer`'s — `Budget.read(path)` of the lanes file the lane came from; the module-level
`compose` takes the tree's own `lanes.json`. The catalogs (`catalog.json`, `coverage.json`) are
always this lanekit's own tree's (lanekit.spec_data), whatever skills root a caller runs the
chain's tools from.

The pieces: lanekit.spec_data (the catalogs, `Target`, `Contract`, `Budget`), lanekit.spec_choices
(each choice's source policy) and lanekit.spec_families (`FAMILY_BLOCKS`, `REQUIRED_TYPES`, the
unit tier's specs); this module re-exports what callers use.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from . import spec_choices as choices
from .recipes import effective_task
from .spec_data import (CATALOG, CONV_TYPES, COVERAGE, SCHEMA, TASK_ROWS, Budget, Contract, SpecError,
                        Target, TaskRow, batch_rule, head_out, lane_defaults, raw_of)
from .spec_families import (FAMILY_BLOCKS, REQUIRED_TYPES, family_spec, minimal_block_spec,
                            required_types, synthetic_target)

__all__ = ["TaskRow", "TASK_ROWS", "Target", "Contract", "Budget", "SourceNames", "Composer",
           "FAMILY_BLOCKS", "REQUIRED_TYPES", "FINDING_IDS", "RUN_TS", "CATALOG", "COVERAGE", "CONV_TYPES",
           "compose", "family_spec", "minimal_block_spec", "required_types", "synthetic_target",
           "model_name", "head_out", "batch_rule", "lane_defaults", "SpecError"]

RUN_TS = "2026-09-05T11-03Z"                    # the campaign root every fixture forges into
LAYERS_READ = ("geometry", "analysis", "shape")
# the finding a role cites, as lanekit.recipes.findings_for numbers them
FINDING_IDS: Mapping[str, str] = MappingProxyType(
    {"balance": "F1", "time": "F2", "relations": "F3", "drift": "F4", "importance": "F5",
     "leakage": "F6"})


@dataclass(frozen=True)
class SourceNames:
    """How the chain's layers name what the spec cites: the geometry's partition (the task and
    the head) and the analysis's findings by role. Every lane fixture writes the standard
    names; the orders fixture (`make_fixture.py`) cites its partition by label
    (`geometry:partitions/late`, its one candidate) and numbers its balance finding F2."""
    partition: str = "geometry:partitions/chosen"
    finding_ids: Mapping[str, str] = field(default_factory=lambda: dict(FINDING_IDS))

    def finding(self, role: str) -> str:
        return f"analysis:{self.finding_ids[role]}"


def model_name(lane: Any) -> str:
    """The layered model a lane's fixture writes: `<generator>.domain.analysis.shaped.html`."""
    return f"{raw_of(lane)['generator']}.domain.analysis.shaped.html"


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- task and sample
def _typed(value: Any, as_int: bool) -> Any:
    if value is None:
        return None
    if as_int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return str(value)


def _task(raw: Mapping[str, Any], contract: Contract, source: str) -> dict:
    target = raw.get("target") or {}
    task: dict[str, Any] = {"kind": raw["task"]}
    effective = effective_task(raw)
    if raw["task"].startswith("sequence-"):
        task["base_kind"] = target.get("base")
    task["label"] = target.get("column")
    labelled = target.get("column") is not None
    as_int = contract.label_is_int and effective != "multilabel"
    if labelled and target.get("classes"):
        task["classes"] = [_typed(c, as_int) for c in target["classes"]]
    if effective == "multilabel":
        task["label_format"] = {"kind": "delimited", "sep": target.get("sep") or "|"}
    if labelled and effective in ("binary", "anomaly") and target.get("positive") is not None:
        task["positive"] = _typed(target["positive"], as_int)
    # No field outside netspec-contract §1 (A8): a lane's expected anomaly rate is the user's
    # goal stance, which `task.source` cites and the memory line carries (lanekit.fixture).
    task["source"] = source
    return task


def _sub_source(key: str, is_default: bool, row: str, user: str) -> dict:
    if is_default:
        return {f"{key}_source": "vega:default", f"{key}_catalog": row}
    return {f"{key}_source": user}


def _window_sample(raw: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict:
    w = raw["window"]
    forecast = raw["task"] == "forecast"
    catalog_ctx = CATALOG["defaults"]["window_context"]["value"]
    ctx = w.get("context") or catalog_ctx
    auto_default = CATALOG["defaults"]["autoregressive"]["value"]["forecast" if forecast else "otherwise"]
    declared = w.get("autoregressive")
    if declared and not forecast:
        raise SpecError(f"lane {raw.get('id')}: autoregressive is a forecast channel; on a "
                        f"{raw['task']} lane it would carry the anchor's own label (W1)")
    auto = auto_default if declared is None else bool(declared)
    sample: dict[str, Any] = {"shape": "window", "length": int(w["length"]), "stride": int(w.get("stride") or 1)}
    if forecast:
        sample["horizon"] = int(w["horizon"])
    sample.update({"series_by": w.get("series_by"), "order_by": w.get("order_by"), "context": ctx,
                   "autoregressive": auto, "season": w.get("season"), "source": "analysis:context.time"})
    sample.update(_sub_source("context", ctx == catalog_ctx, "defaults/window_context", "user:stance/window-context"))
    sample.update(_sub_source("autoregressive", auto == auto_default, "defaults/autoregressive",
                              w.get("autoregressive_source") or "user:stance/autoregressive"))
    sample["series_source"] = _series_source(w, manifest)
    return sample


def _recipe_params(manifest: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """{step id: (op, params)} from the recipe the shaper copied beside the manifest
    (`<shaped>/recipe.json`): the manifest's step records carry what was fitted, the recipe what
    was asked (a lag's `group_by`, a split's `by`)."""
    shaped = (manifest.get("outputs") or {}).get("dir")
    path = Path(shaped) / "recipe.json" if shaped else None
    if path is None or not path.is_file():
        return {}
    steps = json.loads(path.read_text(encoding="utf-8")).get("steps") or []
    return {str(s.get("id")): (str(s.get("op")), s.get("params") or {}) for s in steps if s.get("id")}


def _series_source(w: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """defaults/window_series: the lag step's group_by, else the split's by — the shaping step
    that names the series (`shape:S<n>`); else the lane's stance (one series, or a series key
    nothing in the shaping record names)."""
    series = w.get("series_by")
    if series:
        steps = _recipe_params(manifest)
        for op, key in (("lag", "group_by"), ("split", "by")):
            hit = next((sid for sid, (o, params) in steps.items() if o == op and params.get(key) == series), None)
            if hit:
                return f"shape:{hit}"
    return "user:stance/series" if series else "user:stance/one-series"


def _graph_sample(raw: Mapping[str, Any], contract: Contract) -> dict:
    g, block = raw["graph"], contract.graph or {}
    if not block:
        raise SpecError("the lane is a graph lane but the shaped manifest carries no graph block")
    level = block.get("level") or g.get("level")
    default_batching = "graphs" if level == "graph" else "full"
    batching = g.get("batching") or default_batching
    protocol = g.get("protocol") or "inductive"
    sample: dict[str, Any] = {"shape": "graph", "edges": contract.edges, "level": level,
                              "graph_id": block.get("graph_id"), "directed": bool(block.get("directed")),
                              "self_loops": bool(block.get("self_loops")), "protocol": protocol}
    if protocol == "inductive":
        sample.update({"protocol_source": "vega:default", "protocol_catalog": "defaults/graph_protocol"})
    else:
        sample["protocol_source"] = g.get("protocol_source") or "user:stance/transductive"
    sample["batching"] = batching
    sample.update(_sub_source("batching", batching == default_batching, "defaults/graph_batching",
                              f"user:stance/batching-{batching}"))
    sample["neighbours"] = list(g.get("neighbours") or ()) if batching == "sampled" else None
    sample["exposure"] = ({"kind": "features", "rows": "test", "source": sample["protocol_source"]}
                          if protocol == "transductive" else None)
    sample["source"] = "shape:manifest/graph"
    return sample


def _data_contract(raw: Mapping[str, Any], contract: Contract, manifest: Mapping[str, Any],
                   names: SourceNames) -> dict:
    if raw["shape"] == "window":
        sample = _window_sample(raw, manifest)
    elif raw["shape"] == "graph":
        sample = _graph_sample(raw, contract)
    else:
        sample = {"shape": "row"}
    dc: dict[str, Any] = {"source": "shape:manifest/output_schema",
                          "parts": {"train": contract.parts.get("train"),
                                    "valid": contract.parts.get("valid"), "test": "sealed"},
                          "features": [dict(f) for f in contract.features],
                          "input_dim": contract.input_dim, "keys": list(contract.keys), "sample": sample}
    if contract.balance:
        dc["class_balance"] = {"train": dict(contract.balance), "source": names.finding("balance")}
    return dc


# --------------------------------------------------------------------------- compose
@dataclass(frozen=True)
class Composer:
    """What a composition reads besides the lane and its shaped data: how the model's layers name
    the partition and the findings (`SourceNames`), and the campaign budget (`Budget`; None: the
    tree's own `lanes.json` defaults, read when a spec is composed)."""
    names: SourceNames = field(default_factory=SourceNames)
    budget: Budget | None = None

    @classmethod
    def for_lanes_file(cls, lanes_file: Path | None, names: SourceNames | None = None) -> "Composer":
        """The composer for lanes read from `lanes_file` (its defaults are their budget)."""
        return cls(names or SourceNames(), Budget.read(lanes_file))

    def compose(self, lane: Any, manifest: Mapping[str, Any], fitted: Mapping[str, Any], model_digest: str,
                pipeline: Path) -> dict:
        """The lane's netspec over its shaped data (module docstring). `fitted` is the shaper's
        fitted.json: an encode step's unseen id is read from it when the manifest lacks it."""
        raw = raw_of(lane)
        manifest = _with_fitted(manifest, fitted)
        contract = Contract.from_manifest(manifest, raw)
        target = Target.of(raw)
        labelled = contract.label is not None
        task_source = self.names.partition if labelled else "user:stance/goal"
        budget = (self.budget or Budget.read()).for_lane(raw)       # a lane's own override (A38)
        data_contract = _data_contract(raw, contract, manifest, self.names)
        blocks = choices.blocks(raw, contract, target, task_source)
        optimizer = choices.optimizer(raw)
        metrics = choices.metrics(target, labelled)
        balance_source = self.names.finding("balance") if contract.balance else None
        pipe = json.loads(Path(pipeline).read_text(encoding="utf-8"))
        regularization = choices.regularization(blocks, optimizer)
        blocks, optimizer = choices.declared_once(blocks, optimizer, regularization)
        return {
            "schema": SCHEMA, "input": _inputs(raw, manifest, model_digest, Path(pipeline), pipe),
            "seed": int(raw["seed"]), "task": _task(raw, contract, task_source),
            "data_contract": data_contract, "family": choices.family_choice(raw), "blocks": blocks,
            "loss": choices.loss(raw, target, balance_source), "metrics": metrics, "optimizer": optimizer,
            "schedule": choices.schedule(raw, budget, float(optimizer["params"].get("lr", 0.001))),
            "regularization": regularization,
            "early_stopping": choices.early_stopping(metrics, budget), "init": choices.init(raw),
            "training": choices.training(contract, data_contract["sample"], budget),
            "backend": {"name": "pytorch", "min_version": "2.2", "hints": {}},
            "decision_families": copy.deepcopy(dict(CATALOG["decision_families"])),
            "outputs": _outputs(pipe["system"]["package"]),
        }


def compose(lane: Any, manifest: Mapping[str, Any], fitted: Mapping[str, Any], model_digest: str,
            pipeline: Path) -> dict:
    """`Composer().compose(...)`: the standard layer names and the tree's own lanes.json budget."""
    return Composer().compose(lane, manifest, fitted, model_digest, pipeline)


def _inputs(raw: Mapping[str, Any], manifest: Mapping[str, Any], model_digest: str, pipeline: Path,
            pipe: Mapping[str, Any]) -> dict:
    """The spec's `input`: the model, the shaped manifest and the pipeline, each with its digest."""
    shaped = Path((manifest.get("outputs") or {}).get("dir") or ".")
    return {"model": model_name(raw), "model_digest": model_digest, "layers_read": list(LAYERS_READ),
            "shaped_dir": str(shaped), "manifest": str(shaped / "manifest.json"),
            "manifest_digest": _sha((shaped / "manifest.json").read_bytes()),
            "pipeline": str(pipeline),
            # the canonical digest (netspec-contract §1, compiler rule D1e), as the emitter's
            "pipeline_digest": _sha(json.dumps(pipe, sort_keys=True, separators=(",", ":")).encode("utf-8"))}


def _outputs(package: str) -> dict:
    return {"run_dir": f"vega-runs/{RUN_TS}/", "package": package,
            "code": [f"{package}/model/network.py", f"{package}/ingestion/data.py",
                     f"{package}/training/loop.py", f"{package}/model/metrics.py"],
            "notebook": "engineer.ipynb"}


def _with_fitted(manifest: Mapping[str, Any], fitted: Mapping[str, Any]) -> dict:
    """The manifest, with each encode step's per-column fitted entry filled from fitted.json
    when the manifest's own record lacks it."""
    out = copy.deepcopy(dict(manifest))
    for step in out.get("steps") or []:
        if step.get("op") != "encode":
            continue
        record = step.setdefault("parameters_fitted", {})
        if not record.get("fitted") and isinstance(fitted.get(step.get("id")), Mapping):
            record["fitted"] = copy.deepcopy(dict(fitted[step["id"]].get("fitted") or fitted[step["id"]]))
    return out
