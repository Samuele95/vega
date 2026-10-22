#!/usr/bin/env python3
"""
attribution_build.py — the demonstration sample through the bundle's own network, at show time
(contract §4.8; build plan R-D9): `data/attribution.json` (`vega/attribution@2`), the demo
sample's row scores (`page.json.reference` on the reconstruction and anomaly lanes) and the
network view's lane (gate BD2's cross-check).

    python3 attribution_build.py --bundle B --out RUN.json [--method grad_x_input] [--steps N]

`build_showcase.py` runs it in a child process with the bundle as its working directory, a
clean environment and no bytecode written: the bundle's `predict.py` is imported from the
bundle, `predict.load_inspector(bundle)` opened on the shipped example rows, and — only when
its `status()` says it is ready — `attribute(rows, targets, method, steps, history)` asked over
the demo strata of `examples/demo.json` (the training part, never the test part). The rows
travel as the service decodes a JSON body (`serve.contract.ApiContract` → `tables.frame`), so
the attribution is the one the page would get for the same demo row.

**The axis** follows the bundle's reading (`MANIFEST.output.reading`, D4 §6.4's weights axis):

| reading | axis | a column | the target of its rows |
|---|---|---|---|
| binary | `class` | negative, positive | `{class: <positive>}`: the positive logit |
| multiclass | `class` | each class | `{class: <the column's class>}`: the true class |
| ordinal | `class` | each class | the task's default, E[y] (`expected_rank`) |
| multilabel | `label` | each label | `{label: <the column's label>}` |
| regression, quantile | `bin` | each quintile bin of the target | the value (quantile nearest 0.5) |
| forecast | `horizon` | each step ahead `h` | `{step: h}` over every demo item |
| reconstruction | `group` | one column, `all` | the reconstruction error |
| anomaly (labelled) | `group` | each class | the error or the distance |
| anomaly (unlabelled) | `group` | one column, `all` | the error or the distance |

A column's id is the demo stratum's id (`class-0`, `bin-2`, `h1`, `all`), its name the data's own
(a class or label name, a bin's range, the step); `correct` counts the rows whose answer's class
is the truth (class columns) or whose label is flagged (label columns), null otherwise. Each
row is the per-sample attribution over the network inputs (`inputs`, network order): a row's
own list, a window's per-feature sums over its steps (`reduce: sum_over_steps`), a graph node's
own part (`reduce: self_only`). Rows and graph nodes travel in one request per column (an answer
never depends on the other rows sent); a window or a whole graph is one request per item.

**The result** (`--out`, JSON): `{attribution | null, reason | null, reference_scores | null,
reference_reason | null, network | null, inspection | null}`. No demonstration sample, a bundle
packed `--no-inspect`, an inspector that is not ready or an attribution that fails is
`attribution: null` with the reason (the page then hides that reading; the builder says so with
`WARN:`); row scores that cannot be read are `reference_scores: null` with theirs. The skill's
own directory leaves the import path before the bundle is imported. Exit 0 with the result
written, 2 when the bundle cannot be read or loaded at all.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

__all__ = ["AXES", "SCHEMA", "Column", "AttributionPlan", "AttributionBuilder",
           "reference_scores", "bundle_run", "main"]

SCHEMA = "vega/attribution@2"
DEMO = "examples/demo.json"
SOURCE = {"file": DEMO, "part": "train"}
AXES = {"binary": "class", "multiclass": "class", "ordinal": "class", "multilabel": "label",
        "regression": "bin", "quantile": "bin", "forecast": "horizon",
        "reconstruction": "group", "anomaly-reconstruction": "group", "anomaly-svdd": "group"}
REDUCE = {"row": None, "window": "sum_over_steps", "graph": "self_only"}
SCORE_KINDS = ("row_score", "score")          # the served field a reference score is read from
SCORED_READINGS = ("reconstruction", "anomaly-reconstruction", "anomaly-svdd")
ALL = "all"
REFUSALS = (ValueError, LookupError, RuntimeError, TypeError, ArithmeticError)


@dataclass(frozen=True)
class Column:
    """One column of the axis: its id, name and range, the demo items it reads, the selector
    of its rows' target (None: the task's default) and how a row is judged right."""

    id: str
    name: str
    range: tuple | None
    items: tuple
    selector: Mapping | None
    judged: str | None = None           # "class", "label" or None

    def header(self) -> dict:
        return {"id": self.id, "name": self.name,
                "range": list(self.range) if self.range is not None else None}


def _strata(demo: Mapping, kind: str) -> list[tuple[dict, tuple]]:
    """(stratum, its items) for every stratum of `kind`, in the demo's order."""
    items = demo.get("items") or ()
    return [(s, tuple(i for i in items if i.get("stratum") == s.get("id")))
            for s in demo.get("strata") or () if s.get("kind") == kind]


def _class_columns(demo: Mapping, output: Mapping) -> list[Column]:
    reading, classes = output.get("reading"), [str(c) for c in output.get("classes") or ()]
    found = sorted(_strata(demo, "class"), key=lambda p: classes.index(str(p[0]["name"]))
                   if str(p[0]["name"]) in classes else len(classes))
    positive = output.get("positive") or (classes[1] if len(classes) == 2 else None)
    if reading == "binary" and positive is None:
        raise ValueError("a binary output names no positive class (MANIFEST.output.positive)")

    def selector(name: str):
        if reading == "binary":
            return {"class": str(positive)}
        return {"class": name} if reading == "multiclass" else None
    return [Column(str(s["id"]), str(s["name"]), None, items, selector(str(s["name"])), "class")
            for s, items in found]


def _label_columns(demo: Mapping, output: Mapping) -> list[Column]:
    return [Column(str(s["id"]), str(s["name"]), None, items, {"label": str(s["name"])}, "label")
            for s, items in _strata(demo, "label") if s.get("name")]


def _bin_columns(demo: Mapping, output: Mapping) -> list[Column]:
    out = []
    for s, items in _strata(demo, "bin"):
        span = tuple(s["range"]) if isinstance(s.get("range"), list) else None
        out.append(Column(str(s["id"]), _range_name(span) or str(s["id"]), span, items, None))
    return out


def _horizon_columns(demo: Mapping, output: Mapping) -> list[Column]:
    steps = [int(h) for h in output.get("horizons") or ()]
    if not steps:
        raise ValueError("a forecast output names no horizons (MANIFEST.output.horizons)")
    items = tuple(demo.get("items") or ())
    return [Column(f"h{h}", f"h{h}", None, items, {"step": h}) for h in steps]


def _group_columns(demo: Mapping, output: Mapping) -> list[Column]:
    labelled = _strata(demo, "class")
    if labelled:
        return [Column(str(s["id"]), str(s["name"]), None, items, None, "class")
                for s, items in labelled]
    return [Column(ALL, ALL, None, tuple(demo.get("items") or ()), None)]


_COLUMNS = {"class": _class_columns, "label": _label_columns, "bin": _bin_columns,
            "horizon": _horizon_columns, "group": _group_columns}


def _range_name(span) -> str | None:
    if span is None or len(span) != 2:
        return None
    show = [("…" if v is None else f"{float(v):.4g}") for v in span]
    return f"{show[0]} – {show[1]}"


@dataclass(frozen=True)
class AttributionPlan:
    """The axis of one bundle's demonstration sample: its kind, its columns, the sample shape
    and whether requests may carry many items at once."""

    axis: str
    columns: tuple
    shape: str
    batched: bool

    @staticmethod
    def of(demo: Mapping, output: Mapping, sample: Mapping) -> "AttributionPlan":
        reading, shape = output.get("reading"), (sample or {}).get("shape") or "row"
        if reading not in AXES:
            raise ValueError(f"the reading {reading!r} has no attribution axis ({', '.join(AXES)})")
        if shape not in REDUCE:
            raise ValueError(f"the sample shape {shape!r} is not one of {', '.join(REDUCE)}")
        columns = tuple(_COLUMNS[AXES[reading]](demo, output))
        if not columns or not any(c.items for c in columns):
            raise ValueError("the demonstration sample holds no item for any column of the "
                             f"{AXES[reading]} axis")
        batched = shape == "row" or (shape == "graph" and (sample.get("level") or "node") == "node")
        return AttributionPlan(AXES[reading], columns, shape, batched)


def _requests(items, batched: bool, frame: Callable):
    """(rows frame, history frame or None, the item of each request row) per request."""
    groups = [list(items)] if batched else [[i] for i in items]
    for group in groups:
        rows = [r for item in group for r in item.get("rows") or ()]
        if not rows:
            continue
        owners = [item for item in group for _ in item.get("rows") or ()]
        history = group[0].get("history") if not batched else None
        yield frame(rows), (frame(history) if history else None), owners


class AttributionBuilder:
    """`vega/attribution@2` of a plan, through an inspector's `attribute`."""

    def __init__(self, plan: AttributionPlan, attribute: Callable, frame: Callable,
                 output: Mapping, method: tuple = ("grad_x_input", None)) -> None:
        self._plan, self._attribute, self._frame = plan, attribute, frame
        self._kinds = {k: dict(v) for k, v in (output.get("field_kinds") or {}).items()}
        self._method, self._steps = method
        self._inputs, self._targets = None, set()

    def build(self) -> dict:
        cells = {c.id: self._cell(c) for c in self._plan.columns}
        if len(self._targets) != 1:
            raise ValueError(f"the columns explain {sorted(self._targets) or 'no'} target kinds; "
                             "one attribution explains one kind")
        return {"schema": SCHEMA, "source": dict(SOURCE),
                "method": {"name": self._method, "steps": self._steps
                           if self._method == "integrated_gradients" else None},
                "target": next(iter(self._targets)), "inputs": list(self._inputs),
                "reduce": REDUCE[self._plan.shape],
                "axis": {"kind": self._plan.axis, "columns": [c.header() for c in self._plan.columns]},
                "cells": cells}

    def _cell(self, column: Column) -> dict:
        rows, right = [], 0
        for frame, history, owners in _requests(column.items, self._plan.batched, self._frame):
            targets = None if column.selector is None else [dict(column.selector)] * len(owners)
            table = self._attribute(frame, targets, self._method, self._steps, history)
            self._take(table)
            for sample in table["samples"]:
                rows.append(self._row(sample["attribution"]))
                right += self._judge(column, sample.get("prediction") or {}, owners[sample["row"]])
        return {"n": len(rows), "correct": right if column.judged else None, "rows": rows}

    def _take(self, table: Mapping) -> None:
        inputs = tuple(table["inputs"])
        if self._inputs is not None and inputs != self._inputs:
            raise ValueError("two requests attribute over different network inputs")
        self._inputs = inputs
        self._targets.update(t["kind"] for t in table.get("targets") or ())

    def _row(self, entry) -> list:
        part = {"row": entry, "window": (entry or {}).get("per_feature") if isinstance(entry, Mapping)
                else None, "graph": (entry or {}).get("self") if isinstance(entry, Mapping) else None}
        values = part[self._plan.shape]
        if not isinstance(values, list) or len(values) != len(self._inputs):
            raise ValueError(f"a {self._plan.shape} attribution entry is not one number per "
                             f"network input ({len(self._inputs)})")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("an attribution is not a finite number")
        return [float(v) for v in values]

    def _judge(self, column: Column, answer: Mapping, item: Mapping) -> int:
        if column.judged == "class":
            field = self._field("class", None)
            truth = item.get("truth")
            return int(field is not None and truth is not None and
                       str(answer.get(field)) == str(truth))
        if column.judged == "label":
            field = self._field("flag", column.name)
            return int(field is not None and _flagged(answer.get(field)))
        return 0

    def _field(self, kind: str, of) -> str | None:
        return next((f for f, k in self._kinds.items() if k.get("kind") == kind
                     and (of is None or str(k.get("of")) == str(of))), None)


def _flagged(value) -> bool:
    try:
        return bool(value) and float(value) != 0.0
    except (TypeError, ValueError):
        return str(value).lower() == "true"


def reference_scores(demo: Mapping, output: Mapping, sample: Mapping, predict: Callable,
                     frame: Callable) -> dict | None:
    """The served row score of every demo item the bundle answers (reconstruction and anomaly
    readings; None on the others): what `page.json.reference` takes its quantiles from."""
    if output.get("reading") not in SCORED_READINGS:
        return None
    kinds = output.get("field_kinds") or {}
    field = next((f for want in SCORE_KINDS for f, k in kinds.items() if k.get("kind") == want), None)
    if field is None:
        raise ValueError(f"the {output.get('reading')} output serves no row score field")
    batched = AttributionPlan.of(demo, output, sample).batched
    scores = []
    for rows, history, _ in _requests(demo.get("items") or (), batched, frame):
        answer = predict(rows, history)
        done = answer["answered"] if "answered" in answer else [True] * len(answer)
        scores += [float(v) for v, ok in zip(answer[field], done) if ok and math.isfinite(float(v))]
    if not scores:
        raise ValueError("the bundle answered no demo item with a finite row score")
    return {"scores": scores, "source": f"{DEMO} (train)"}


# --------------------------------------------------------------------------- the bundle run

def _bundle_predict(bundle: Path):
    """The bundle's own `predict.py`, imported from the bundle (it puts the bundle first on
    the path and imports its adapters from there)."""
    spec = importlib.util.spec_from_file_location("predict", bundle / "predict.py")
    if spec is None or spec.loader is None:
        raise ValueError(f"{bundle / 'predict.py'} cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules["predict"] = module
    spec.loader.exec_module(module)
    return module


def _demo(bundle: Path, manifest: Mapping):
    if DEMO not in (manifest.get("files") or {}):
        return None
    return json.loads((bundle / DEMO).read_text(encoding="utf-8"))


def _network_lane(inspector) -> dict:
    view = inspector.network("summary")
    return {k: view.get(k) for k in ("schema", "task", "sample", "family")}


def _attribution(run: dict, parts: tuple, options: tuple) -> None:
    """Fill `run` from the opened inspector: its status, the network's lane, the attribution."""
    predict, bundle, demo, manifest, frame = parts
    try:
        inspector = predict.load_inspector(bundle)
    except LookupError as exc:
        run["reason"] = f"the bundle ships no inspection ({exc})"
        return
    status = inspector.status()
    run["inspection"] = {k: status.get(k) for k in ("enabled", "ready", "why", "weights")}
    if not status.get("ready"):
        run["reason"] = f"the bundle's inspector is not ready: {status.get('why')}"
        return
    run["network"] = _network_lane(inspector)
    if demo is None:
        run["reason"] = f"the bundle ships no demonstration sample ({DEMO}): pack with --demo-sample"
        return
    try:
        plan = AttributionPlan.of(demo, manifest.get("output") or {}, manifest.get("sample") or {})
        run["attribution"] = AttributionBuilder(plan, inspector.attribute, frame,
                                                manifest.get("output") or {}, options).build()
    except REFUSALS as exc:
        run["reason"] = f"the attribution could not be computed: {type(exc).__name__}: {exc}"


def bundle_run(bundle: Path, options: tuple = ("grad_x_input", None)) -> dict:
    """The result document of one bundle (module docstring). The skill's own directory leaves
    the import path first: the bundle runs on its own files, as it would with no skill
    installed."""
    bundle, here = Path(bundle).resolve(), Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != here]
    predict = _bundle_predict(bundle)
    predict.verify(bundle, roles=("data", "example"))
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    from serve.contract import ApiContract                              # the bundle's own
    frame = ApiContract.from_bundle(bundle).tables.frame
    demo = _demo(bundle, manifest)
    run = {"attribution": None, "reason": None, "reference_scores": None,
           "reference_reason": None, "network": None, "inspection": None}
    if demo is not None:
        try:
            run["reference_scores"] = reference_scores(
                demo, manifest.get("output") or {}, manifest.get("sample") or {},
                predict.load(bundle).predict, frame)
        except REFUSALS as exc:
            run["reference_reason"] = f"{type(exc).__name__}: {exc}"
    _attribution(run, (predict, bundle, demo, manifest, frame), options)
    return run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--bundle", required=True, help="the bundle directory")
    ap.add_argument("--out", required=True, help="where the result document goes (JSON)")
    ap.add_argument("--method", default="grad_x_input",
                    choices=("grad_x_input", "integrated_gradients"))
    ap.add_argument("--steps", type=int, default=None, help="integrated gradients' steps")
    args = ap.parse_args(argv)
    try:
        run = bundle_run(Path(args.bundle), (args.method, args.steps))
        Path(args.out).write_text(json.dumps(run, ensure_ascii=False, allow_nan=False),
                                  encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — the child reports any failure as its one line
        print(f"ERROR: attribution_build: the bundle could not be run: {type(exc).__name__}: {exc}")
        return 2
    print(f"OK: attribution_build: {'attribution written' if run['attribution'] else run['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
