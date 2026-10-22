"""
The recipe every lane's data is shaped by, built from a generator's SCHEMA and the lane
(D6 §4: the orders pattern, generalised), and the analysis findings the recipe and the netspec
cite.

    recipe = standard_recipe(SCHEMA, raw_path, shaped_dir, lane)     # dataset-shaper/recipe@1
    findings = findings_for(SCHEMA, lane)                             # F1…F6, what cites them
    plan = ColumnPlan.of(SCHEMA, lane)                                # lanekit.columns, re-exported

A generator's `RECIPE` usually is `standard_recipe` over its own SCHEMA; one whose data needs a
step the standard recipe does not take writes its own RECIPE and may still cite `findings_for`
(it does not change this module: the standard recipe is every generator's, the W5 generators
write in parallel).

The recipe, phase by phase (`step-catalog.md`), over the column plan (lanekit.columns):

1. `retype` the nominal and series columns to nominal; `drop_identity` the keys (and, on a lane
   that is not a window, the timestamps and series ids); `parse_datetime` a window lane's
   timestamps;
2. `select_partition` the lane's label; `drop_leakage` every other label candidate and every
   `drop` column;
3. one `split`: `time` on a window lane (its `order_by`), `group` by the graph id at graph level,
   `stratified` on a class label, `random` otherwise — 70 / 15 / 15;
4. the lane's `recipe_options.lag` (per series, in time order), then `impute` of the lag columns
   the part starts leave empty;
5. `encode` the nominals (ordinal or one-hot, as the plan says) and `scale` (standard) the
   numeric columns and the lags;
7. `graph_edges` from the lane's `graph` (key_pair on the key columns, knn on named features).

Every step carries a source from the shaper's closed set, a rationale, its consequences, and
one alternative when the source is `shaper:default` (the shaper warns otherwise).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .chain import sha_file
from .columns import CLASS_TASKS, ColumnPlan, effective_task, embedded_columns, task_prose

__all__ = ["ColumnPlan", "Finding", "Candidate", "Step", "standard_recipe", "findings_for",
           "effective_task", "embedded_columns", "task_prose", "SPLIT_FRACTIONS", "CLASS_TASKS",
           "time_split_cuts"]

SPLIT_FRACTIONS = {"test": 0.15, "valid": 0.15}


def time_split_cuts(rows: int) -> tuple[int, int, int, int]:
    """(0, end of train, end of valid, rows): where the time split the standard recipe asks for
    cuts `rows` time-ordered rows, computed as shape.py's split computes it (n_test =
    round(test * n), n_valid = round(valid * n), train the rest). A generator that plants
    events in every part (lanekit.datasets.vibration, .tank) reads the parts from here."""
    rows = int(rows)
    n_test = int(round(SPLIT_FRACTIONS["test"] * rows))
    n_valid = int(round(SPLIT_FRACTIONS["valid"] * rows))
    return (0, rows - n_test - n_valid, rows - n_test, rows)


def _raw(lane: Any) -> Mapping[str, Any]:
    return lane.raw if hasattr(lane, "raw") else lane


# --------------------------------------------------------------------------- findings
@dataclass(frozen=True)
class Candidate:
    """A transformation candidate a finding proposes (ids unique across findings)."""
    id: str
    op: str
    columns: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class Finding:
    """One admitted analysis finding, as the analysis layer carries it (analysis-contract §1):
    `so_what` says what the finding changes downstream, keyed by where it lands
    (`preprocessing`, `modeling`, `collection`, `interpretation`), as pairs."""
    id: str
    module: str
    severity: str
    title: str
    columns: tuple[str, ...]
    reading: str
    so_what: tuple[tuple[str, str], ...]
    candidates: tuple[Candidate, ...] = ()

    def so_what_record(self) -> dict[str, str]:
        return dict(self.so_what)

    def source(self, candidate: str | None = None) -> str:
        return f"analysis:{self.id}" + (f"/{candidate}" if candidate else "")


def findings_for(schema: Sequence[Any], lane: Any) -> dict[str, Finding]:
    """The findings a lane's analysis layer admits, by id, with fixed ids so a recipe step and
    the netspec can cite them: F1 the label's balance (a class label), F2 the time context (a
    window lane), F3 the relation between rows (a graph lane), F4 the drift the Monitor
    watches, F5 non-linear importance, F6 the leakage set (other labels, dropped columns)."""
    plan = ColumnPlan.of(schema, lane)
    raw = _raw(lane)
    numeric = plan.numeric or plan.nominal
    out: dict[str, Finding] = {}
    if plan.label and effective_task(lane) in CLASS_TASKS:
        out["F1"] = Finding("F1", "distributions", "medium", f"the classes of {plan.label} are not equally frequent",
                            (plan.label,), f"{plan.label} is not balanced across its classes on the train part",
                            (("modeling", "weight the rare class in the loss or rank it in the metrics"),
                             ("interpretation", "accuracy alone would reward the frequent class")))
    if raw["shape"] == "window":
        cands = ()
        if plan.lag_column:
            cands = (Candidate("T1", "lag", (plan.lag_column,), f"{plan.lag_column} depends on its own recent past"),
                     Candidate("T2", "impute", plan.lag_outputs, "a lag is empty at the start of each part"))
        out["F2"] = Finding("F2", "time_series", "high", "the rows are a time series",
                            tuple(c for c in (plan.order_by, plan.series_by) if c),
                            "the rows are ordered in time and the label depends on the recent past",
                            (("preprocessing", "split in time: the future is never trained on"),
                             ("modeling", "let the network read a window of past rows")), cands)
    if raw["shape"] == "graph":
        g = raw.get("graph") or {}
        out["F3"] = Finding("F3", "relations", "high", "the rows are related to one another",
                            tuple(g.get("columns") or ()), f"rows are linked by {g.get('method')}, and a row's "
                            "label depends on its neighbours",
                            (("preprocessing", "record the relation as an edges part"),
                             ("modeling", "the graph families can read the relation")),
                            (Candidate("T3", "graph_edges", tuple(g.get("columns") or ()),
                                       "the relation between rows is part of the signal"),))
    out["F4"] = Finding("F4", "drift", "low", f"{numeric[0]} moves between the parts", (numeric[0],),
                        f"the distribution of {numeric[0]} is not the same in every part",
                        (("interpretation", f"a shift in {numeric[0]} explains part of any gap between parts"),
                         ("collection", "a deployment should watch it")))
    out["F5"] = Finding("F5", "importance", "medium", "the signal is not linear in the features",
                        tuple(numeric[:3]), "a non-linear model ranks above a linear one on the train part",
                        (("modeling", "a network has something to learn that a linear floor does not"),))
    if plan.leak:
        out["F6"] = Finding("F6", "relations", "high", "columns that are known only after the fact",
                            plan.leak, "these columns determine the label or are other labels",
                            (("preprocessing", "they leave before the split"),),
                            (Candidate("T4", "drop_leakage", plan.leak,
                                       "a leaked column would score perfectly and predict nothing"),))
    return out


# --------------------------------------------------------------------------- the recipe
@dataclass(frozen=True)
class Step:
    """One recipe step before it is numbered: the op, its columns and params, the source it
    cites, why, what it changes (rows, columns, downstream), and the alternative weighed — which
    a `shaper:default` step must carry (the shaper warns otherwise)."""
    op: str
    columns: Iterable[str]
    params: Mapping[str, Any]
    source: str
    rationale: str
    consequences: tuple[str, str, str]
    alternative: Mapping[str, Any] | None = None

    def record(self, sid: str) -> dict[str, Any]:
        rows, cols, down = self.consequences
        step = {"id": sid, "op": self.op, "columns": list(self.columns), "params": dict(self.params),
                "source": self.source, "rationale": self.rationale,
                "consequences": {"rows": rows, "columns": cols, "downstream": down}}
        if self.alternative is not None:
            step["alternatives"] = [dict(self.alternative)]
        elif self.source == "shaper:default":
            raise ValueError(f"{sid} {self.op}: a shaper:default step carries one alternative")
        return step


class _Steps:
    """The recipe's steps, numbered S01…, each with its source, rationale and consequences."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def add(self, step: Step) -> str:
        sid = f"S{len(self.steps) + 1:02d}"
        self.steps.append(step.record(sid))
        return sid


def _phase1(s: _Steps, plan: ColumnPlan) -> None:
    if plan.retyped:
        s.add(Step("retype", plan.retyped, {"to": "nominal"}, f"geometry:typing/{plan.retyped[0]}",
                   "these columns name categories; none of them is a quantity",
                   ("unchanged", f"unchanged ({len(plan.retyped)} retyped)", "they become encodable")))
    if plan.keys:
        s.add(Step("drop_identity", plan.keys, {}, f"geometry:typing/{plan.keys[0]}",
                   "these columns identify a row, a series or an instant; they do not describe it",
                   ("unchanged", f"-{len(plan.keys)}", "no identifier reaches the model")))
    if plan.times:
        s.add(Step("parse_datetime", plan.times, {"errors": "coerce"}, "analysis:context.time",
                   "the timestamp orders the rows; it is parsed so the split and the windows can read it",
                   ("unchanged", "unchanged (parsed)", "the time split and the window order")))


def _phase2(s: _Steps, plan: ColumnPlan, lane: Any, findings: Mapping[str, Finding]) -> None:
    if plan.label:
        s.add(Step("select_partition", [plan.label], {"label": plan.label, "task": task_prose(lane)},
                   f"geometry:partitions/{plan.label}", f"{plan.label} is the question this lane asks",
                   ("unchanged", "unchanged", f"{plan.label} is named as the label")))
    if plan.leak:
        source = findings["F6"].source("T4")
        s.add(Step("drop_leakage", plan.leak, {}, source,
                   "columns that are other labels or are known only after the fact leave before the split",
                   ("unchanged", f"-{len(plan.leak)}", "no leaked column reaches the model")))


def _split(s: _Steps, plan: ColumnPlan, lane: Any) -> None:
    raw, fractions = _raw(lane), dict(SPLIT_FRACTIONS)
    rows = "split 70/15/15"
    if raw["shape"] == "window":
        s.add(Step("split", [], {"kind": "time", "time_column": plan.order_by, **fractions},
                   "analysis:context.time", "a time series is held out in time: the future is never "
                   "trained on", (rows, "unchanged", "every fitted step below is fitted on train alone")))
    elif plan.graph_id:
        s.add(Step("split", [], {"kind": "group", "by": plan.graph_id, **fractions}, "shaper:default",
                   f"one {plan.graph_id} is one sample: its rows never straddle two parts",
                   (rows, "unchanged", "every graph lies in one part"),
                   {"op": "split", "params": {"kind": "random"}, "when": "never at graph level"}))
    elif plan.label and effective_task(lane) in CLASS_TASKS:
        s.add(Step("split", [], {"kind": "stratified", "by": plan.label, **fractions}, "shaper:default",
                   "independent rows with a class label: a seeded stratified split keeps every class in "
                   "every part", (rows, "unchanged", "every fitted step below is fitted on train alone"),
                   {"op": "split", "params": {"kind": "random"}, "when": "if the classes were balanced"}))
    else:
        s.add(Step("split", [], {"kind": "random", **fractions}, "shaper:default",
                   "independent rows without a class to stratify on: a seeded random split",
                   (rows, "unchanged", "every fitted step below is fitted on train alone"),
                   {"op": "split", "params": {"kind": "stratified"}, "when": "if a class label existed"}))


def _phase4(s: _Steps, plan: ColumnPlan, findings: Mapping[str, Finding]) -> None:
    if not plan.lag_column:
        return
    f2 = findings["F2"]
    s.add(Step("lag", [plan.lag_column], {"lags": list(plan.lags), "group_by": plan.series_by,
                                          "time_column": plan.order_by}, f2.source("T1"),
               f"{plan.lag_column} depends on its own recent past; the lags are read per series, in "
               "time order", ("unchanged", f"+{len(plan.lags)}", "the network reads the recent past")))
    s.add(Step("impute", plan.lag_outputs, {"strategy": "median"}, f2.source("T2"),
               "a lag is empty on the first rows of each part; the train median stands in",
               ("unchanged", "unchanged", "no missing value reaches the model")))


def _phase5(s: _Steps, plan: ColumnPlan) -> None:
    user = set(plan.user_encoded)
    for strategy, cols, other in (("ordinal", plan.ordinal, "one-hot"), ("one-hot", plan.onehot, "ordinal")):
        for group in (tuple(c for c in cols if c in user), tuple(c for c in cols if c not in user)):
            if not group:
                continue
            why = ("the network embeds them: an ordinal code with a reserved id for an unseen "
                   "category is what the embedding reads" if strategy == "ordinal" else
                   "no block embeds them: one indicator per category, read as numbers")
            source = f"user:stance/encode-{group[0]}" if group[0] in user else "shaper:default"
            s.add(Step("encode", group, {"strategy": strategy}, source, why,
                       ("unchanged", "unchanged" if strategy == "ordinal" else "one column per category",
                        "codes with a reserved unseen id" if strategy == "ordinal" else "indicator columns"),
                       {"op": "encode", "params": {"strategy": other},
                        "when": "if the network reads them the other way"}))
    scaled = plan.numeric + plan.lag_outputs
    if scaled:
        s.add(Step("scale", scaled, {"kind": "standard", "columns": list(scaled)}, "shaper:default",
                   "the numeric columns sit on different scales; standardised on train they start even",
                   ("unchanged", "unchanged", "standard on train"),
                   {"op": "scale", "params": {"kind": "robust"}, "when": "if the tails are judged genuine"}))


def _graph(s: _Steps, lane: Any, findings: Mapping[str, Finding]) -> None:
    raw = _raw(lane)
    g = raw.get("graph")
    if raw["shape"] != "graph" or not g:
        return
    params: dict[str, Any] = {"method": g["method"], "symmetric": bool(g.get("symmetric", True)),
                              "self_loops": False, "weight": g.get("weight"),
                              "graph_id": g.get("graph_id")}
    if g["method"] == "key_pair":
        params["columns"] = list(g["columns"])
    else:
        params.update({"k": int(g.get("k") or 5), "features": list(g["columns"]), "metric": "euclidean"})
    s.add(Step("graph_edges", g.get("columns") or [], params, findings["F3"].source("T3"),
               f"the rows are related ({g['method']}); the edges part records the relation and the "
               "part of both its ends", ("unchanged", "unchanged", "enables the graph families; the "
                                         "protocol is the network stage's decision")))


def standard_recipe(schema: Sequence[Any], raw_path: Path, shaped_dir: Path, lane: Any) -> dict:
    """The `dataset-shaper/recipe@1` a lane's raw rows are shaped by (module docstring)."""
    import pandas as pd

    raw_path = Path(raw_path)
    frame = pd.read_csv(raw_path, nrows=None, low_memory=False)
    plan = ColumnPlan.of(schema, lane)
    findings = findings_for(schema, lane)
    s = _Steps()
    _phase1(s, plan)
    _phase2(s, plan, lane, findings)
    _split(s, plan, lane)
    _phase4(s, plan, findings)
    _phase5(s, plan)
    _graph(s, lane, findings)
    return {"schema": "dataset-shaper/recipe@1",
            "input": {"path": str(raw_path), "digest": sha_file(raw_path), "rows": int(len(frame)),
                      "columns": int(frame.shape[1])},
            "seed": int(_raw(lane)["seed"]), "goal": "train-ready", "target": plan.label,
            "outputs": {"dir": str(shaped_dir), "format": "parquet", "files": []},
            "steps": s.steps}
