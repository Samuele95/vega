"""
eval_package — the emitted package as /vega-eval drives it: opened by blueprint-map role before
the seal, and what its spec states about the sample the evaluation reads.

`open_package` reaches the adapters with `vega_core.PackageAdapters.load` (BUILD-PLAN §2.12(2):
the one resolver every stage shares) and refuses, by returning the reason, a package this stage
cannot drive — one of another generation than v2.3, one whose map draws no `Evaluator` port
(the one scoring path, code-contract §2). Nothing here reads a part.

The rest reads the spec and the parts already opened: the default baseline set a task and a
sample add to (`default_baselines`, the catalog's `default_when`), the `MetricOptions` the run's
metrics read before the threshold (`metric_options`: the season, a quantile loss's levels,
mase's train-side scale — what the package Evaluator fills itself), one row of features per
sample for the slices (`sample_frame`), the expected anomaly rate (`anomaly_rate`) and the
network's bottleneck width for the PCA floor (`bottleneck`).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["FORGE_DIR", "GENERATION", "Package", "anomaly_rate", "bottleneck", "drifted_columns",
           "lens_ranking",
           "default_baselines", "lens_protocol", "metric_options", "open_package",
           "resolved_spec", "sample_frame", "sample_of"]

HERE = Path(__file__).resolve().parent
#: the sibling skill whose `vega_core.PackageAdapters` opens an emitted package (--vega-forge-dir)
FORGE_DIR = HERE.parent.parent / "vega-forge"
#: the one generation whose adapter classes this stage drives (BUILD-PLAN §2.12(2), gate A)
GENERATION = "v2.3"
_BASELINES = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))["baselines"]
DEFAULT_BASELINES = tuple(_BASELINES["default_set"])
# the baselines the default set adds per task and shape (naive for a forecast, graph-linear at
# node level): D1 §11.3, the catalog's `default_when`
DEFAULT_WHEN = dict(_BASELINES.get("default_when") or {})


def _numpy(values):
    """A tensor on any device, or anything array-like, as an ndarray (None stays None)."""
    import numpy as np
    if values is None:
        return None
    detach = getattr(values, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(values)


def resolved_spec(camp_dir: Path, run_id: str, spec: Mapping) -> Mapping:
    """The spec the run trained under (`runs/<id>/netspec.resolved.json`), else the campaign's."""
    path = Path(camp_dir) / "runs" / str(run_id) / "netspec.resolved.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return spec


def sample_of(spec: Mapping) -> Mapping:
    return ((spec.get("data_contract") or {}).get("sample") or {})


def default_baselines(spec: Mapping) -> tuple:
    """The catalog's default set, plus what the task and the shape add (`default_when`): the
    persistence floors for a forecast or a sequence regression (seasonal when the spec names a
    season), the one-propagation contender at node level."""
    kind = (spec.get("task") or {}).get("kind")
    sample = sample_of(spec)
    shape = sample.get("shape") or "row"
    # a graph sample without a level is node level (the domain's GraphSpec default)
    level = (sample.get("level") or "node") if shape == "graph" else None
    out = list(DEFAULT_BASELINES)
    for name, rule in DEFAULT_WHEN.items():
        if rule.get("tasks") and kind not in rule["tasks"]:
            continue
        if rule.get("season") and not sample.get("season"):
            continue
        if rule.get("shape") and shape != rule["shape"]:
            continue
        if rule.get("level") and level != rule["level"]:
            continue
        out.append(name)
    return tuple(out)


@dataclass(frozen=True)
class Package:
    """The emitted package this run is evaluated with, opened by blueprint-map role — before the
    seal, because a package this stage cannot drive must not cost an exposure."""

    adapters: Any
    directory: Path
    resolved: Mapping

    @property
    def domain(self):
        return self.adapters.package.domain


def open_package(camp_dir: Path, spec: Mapping, run_id: str, resolved: Mapping,
                 forge_dir: Path = FORGE_DIR):
    """(Package, None) or (None, the reason it cannot be driven). Nothing here reads a part."""
    package = (spec.get("outputs") or {}).get("package")
    if not package:
        return None, "netspec.json names no outputs.package; there is no emitted code to evaluate"
    package_dir = Path(camp_dir) / "runs" / str(run_id) / package
    if not package_dir.is_dir():
        package_dir = Path(camp_dir) / package
    if not package_dir.is_dir():
        return None, (f"the emitted package {package!r} is neither in runs/{run_id}/ nor beside "
                      "campaign.json; /vega-forge emits it and /vega-train keeps it")
    scripts = str(Path(forge_dir) / "scripts")
    if scripts not in sys.path:
        sys.path.insert(1, scripts)
    try:
        from vega_core import AdapterLoadError, PackageAdapters
    except ImportError as exc:
        return None, (f"vega-forge's vega_core cannot be imported from {scripts} ({exc}); pass "
                      "--vega-forge-dir")
    try:
        adapters = PackageAdapters.load(package_dir, package)
    except AdapterLoadError as exc:
        return None, f"the package cannot be driven: {exc}"
    except Exception as exc:  # noqa: BLE001 — an emitted module that fails as it is imported
        return None, (f"the package {package!r} fails as it is imported "
                      f"({type(exc).__name__}: {exc}); re-emit it with /vega-forge")
    if adapters.generation != GENERATION:
        return None, (f"{package_dir / 'blueprint-map.json'} records generation "
                      f"{adapters.generation!r}; this stage drives the {GENERATION} adapter "
                      "classes (ShapedParts, Evaluator, ModelRegistry) — re-emit the package with "
                      "/vega-forge and retrain")
    if adapters.evaluator is None:
        return None, (f"{package_dir / 'blueprint-map.json'} maps no file to the Evaluator role; "
                      "the evaluation scores the network and every baseline through the "
                      "package's Evaluator adapter, the one scoring path (code-contract §2)")
    return Package(adapters=adapters, directory=package_dir, resolved=resolved), None


def task_classes(task) -> tuple:
    """The spec's classes, in its order, as the package's `Task` holds them: its `LabelSet`
    (`task.labels.classes`), or the bare tuple a package emitted before carries."""
    labels = getattr(task, "labels", None)
    return tuple(labels.classes) if labels is not None else tuple(task.classes or ())


def metric_options(dom, data, spec_obj, task, names):
    """The run's `MetricOptions` before the threshold: the season the spec states
    (`sample.season`), a quantile loss's levels, and mase's train-side seasonal scale — what the
    package Evaluator fills itself (`_options_for`), stated here once so the bootstrap and the
    port read the same options."""
    sample = dom.SampleSpec.from_spec(spec_obj)
    season = int(sample.window.season) if sample.window is not None and sample.window.season \
        else 1
    quantiles = spec_obj.loss.quantiles
    scale = None
    # a forecast is a window sample, whose DataSource carries `target_scale` (SOLID-I)
    if "mase" in names and task.kind is dom.TaskKind.FORECAST and sample.shape.value == "window":
        scale = data.target_scale(spec_obj, season)
    return dom.MetricOptions(season=season, mase_scale=scale, quantiles=quantiles)


def _carried(dom, name):
    """The package domain's class `name`, or None when the package does not carry that sample
    shape (its domain holds its own shape's values only)."""
    return getattr(dom, name, None)


def sample_frame(dom, data, spec_obj, part):
    """The test (or train) samples as one row of features each, for the slices: a row sample's
    design, a window's anchor row, a node's own row; a graph-level sample has no one row, so its
    frame has the samples and no feature column."""
    import numpy as np
    import pandas as pd
    n = len(part)
    columns = [str(c) for c in (getattr(part, "columns", ()) or ())]
    # a package carries its own sample shape's part class only (the forge emits one shape)
    graph_part, window_part = _carried(dom, "GraphPart"), _carried(dom, "WindowedPart")
    graph = graph_part is not None and isinstance(part, graph_part)
    if graph and part.level is dom.GraphLevel.GRAPH:
        return pd.DataFrame(index=pd.RangeIndex(n))
    if graph or (window_part is not None and isinstance(part, window_part)):
        idx = _numpy(part.target_index if graph else part.anchors)
        blocks = [_numpy(part.x_num)[idx].reshape(n, -1)]
        if part.x_cat is not None:
            blocks.append(_numpy(part.x_cat)[idx].reshape(n, -1))
        x = np.concatenate([b.astype(np.float64) for b in blocks], axis=1)
    else:
        x = np.asarray(data.design_matrix(spec_obj, part).x, dtype=np.float64)
    width = min(x.shape[1], len(columns))
    return pd.DataFrame(x[:, :width], columns=columns[:width])


def anomaly_rate(train, resolved):
    """(rate, where it came from): the train part's positive share when a label exists, else
    the spec's task.anomaly_rate (eval-contract §3), else (None, None)."""
    import numpy as np
    y = _numpy(getattr(train, "y", None))
    if y is not None and y.size:
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        share = float(values.mean()) if values.size else 0.0
        if 0.0 < share < 1.0:
            return share, "the train part's positive share"
    declared = (resolved.get("task") or {}).get("anomaly_rate")
    if declared:
        return float(declared), "task.anomaly_rate"
    return None, None


def bottleneck(resolved, effective):
    """The network's bottleneck width z — the size the PCA floor reconstructs at — for a
    reconstruction or anomaly task: a `reparameterize` block's z, else the last encoder width
    before the decoder, else (deep-svdd) the head's embedding width. None otherwise."""
    if effective not in ("unsupervised", "anomaly"):
        return None
    flat = []

    def walk(blocks):
        for block in blocks or ():
            if not isinstance(block, Mapping):
                continue
            flat.append(block)
            walk((block.get("params") or {}).get("inner"))

    walk(resolved.get("blocks"))
    for block in flat:
        if block.get("type") == "reparameterize":
            z = (block.get("params") or {}).get("z")
            if isinstance(z, int) and z > 0:
                return z
    widths = []
    for block in flat:
        if block.get("type") == "decoder":
            break
        units = (block.get("params") or {}).get("units")
        if block.get("type") == "dense" and isinstance(units, int) and units > 0:
            widths.append(units)
    if widths and any(b.get("type") == "decoder" for b in flat):
        return widths[-1]
    head = next((b for b in reversed(flat) if b.get("type") == "head"), None)
    out = (head or {}).get("params", {}).get("out")
    return out if isinstance(out, int) and out > 0 else None


def _hyper(record: Mapping) -> dict:
    """The hyper-parameters a protocol record carries, under either spelling."""
    found = record.get("hyper_parameters", record.get("params"))
    return dict(found) if isinstance(found, Mapping) else {}


def lens_ranking(importance) -> list:
    """The features /data-lens's importance module ranked, most important first (eval-contract
    §3, robustness): the analysis contract's `evidence.permutation_importance[]` (`{feature,
    mean, model}`) of the model whose protocol `lens_protocol` reads — the first model the list
    names — ordered by `mean`, each feature once; an older record's module-level `ranking` or
    `top` (a list of names) is read as it stands. Empty when the module recorded neither (F39:
    the robustness check read only the older spelling, so on every real analysis the
    importance perturbation was skipped as if no ranking existed)."""
    if not isinstance(importance, Mapping):
        return []
    legacy = importance.get("ranking") or importance.get("top")
    if isinstance(legacy, (list, tuple)) and all(isinstance(c, str) for c in legacy):
        return list(legacy)
    evidence = importance.get("evidence") if isinstance(importance.get("evidence"),
                                                        Mapping) else {}
    rows = [p for p in (evidence.get("permutation_importance") or ())
            if isinstance(p, Mapping) and isinstance(p.get("feature"), str)]
    model = next((p.get("model") for p in rows if p.get("model")), None)
    if model is not None:
        rows = [p for p in rows if p.get("model") in (model, None)]
    rows.sort(key=lambda p: -float(p.get("mean") if p.get("mean") is not None else float("-inf")))
    seen, out = set(), []
    for p in rows:
        if p["feature"] not in seen:
            seen.add(p["feature"])
            out.append(p["feature"])
    return out


def drifted_columns(drift) -> list:
    """The columns /data-lens's drift module found drifting (eval-contract §3, slices): the
    analysis contract's `evidence.columns[]` entries whose `drifted` is true, in its order; an
    older record's module-level `columns` (a list of names) is read as it stands (F39)."""
    if not isinstance(drift, Mapping):
        return []
    legacy = drift.get("columns")
    if isinstance(legacy, (list, tuple)) and all(isinstance(c, str) for c in legacy):
        return list(legacy)
    evidence = drift.get("evidence") if isinstance(drift.get("evidence"), Mapping) else {}
    return [c["column"] for c in (evidence.get("columns") or ())
            if isinstance(c, Mapping) and isinstance(c.get("column"), str) and c.get("drifted") is True]


def lens_protocol(importance) -> dict | None:
    """The protocol `lens` re-fits (eval-contract §4): the model family and the hyper-parameters
    /data-lens's importance module recorded, or None when it recorded none.

    Two spellings are read. The module's own record (`{family, hyper_parameters | params}`), and
    the analysis contract's evidence (`evidence.models[]`, each `{name, cv_score, …}` with its
    `hyper_parameters` or `params` when recorded): there the protocol is the model whose
    permutation importance the analysis reported (`permutation_importance[].model`), else the
    best cross-validated one. A recorded hyper-parameter is carried as recorded — never filled
    in from a default — and `source` says which record it came from."""
    if not isinstance(importance, Mapping):
        return None
    if importance.get("family"):
        return {"family": str(importance["family"]), "hyper_parameters": _hyper(importance),
                "source": "analysis:modules/importance"}
    evidence = importance.get("evidence") if isinstance(importance.get("evidence"),
                                                        Mapping) else {}
    models = [m for m in (evidence.get("models") or ()) if isinstance(m, Mapping) and m.get("name")]
    if not models:
        return None
    ranked = [p.get("model") for p in (evidence.get("permutation_importance") or ())
              if isinstance(p, Mapping) and p.get("model")]
    chosen = next((m for m in models if ranked and m["name"] == ranked[0]), None)
    if chosen is None:
        chosen = max(models, key=lambda m: float(m.get("cv_score") or float("-inf")))
    return {"family": str(chosen["name"]), "hyper_parameters": _hyper(chosen),
            "source": f"analysis:modules/importance/evidence/models[{chosen['name']}]"}
