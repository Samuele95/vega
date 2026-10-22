#!/usr/bin/env python3
"""
test_domain_plans.py — the plans in `templates/domain.py` that training, evaluation and
serving share, proved on the *rendered* module (the template with its placeholders filled, as
the emitter writes it into every package).

    python3 test_domain_plans.py [-v] [-k <substring>]
    python3 -m pytest test_domain_plans.py

What is proved, and why it matters:

  windows  (backend design §9.3, build plan R-A5) — no window crosses `series_by`; context rows
           are inputs only, never anchors; forecast targets never cross the part's end; the
           stride keeps the most recent row; dense scoring gives exactly one anchor per
           scorable row; the autoregressive channel reads only rows at or before the anchor.
  graphs   (build plan R-A3, R-A4) — the inductive train view holds no test raw position; the
           valid view is the train graph plus valid queries attached reference → query only;
           the sealed test view's reference is train ∪ valid; the served reference graph
           (`GraphReference.attach`) rebuilds the sealed test view edge for edge; a graph that
           straddles parts, an edge between graphs and a graph whose rows disagree on the label
           each raise naming the id.
  values   — C6 (every class frozen or an enum) and C7 on the rendered module; the size
           discipline C11–C13 measured with the contract's counting rule (a stand-in for the
           gate's own measure); the C12-shaped factories; the batch keys, stage kinds and the 21
           families against the catalog; no dataset literal.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
TEMPLATE = SCRIPTS / "backends" / "pytorch" / "templates" / "domain.py"
CATALOG = SCRIPTS / "catalog.json"
PACKAGE = "plans_pkg"
VERBOSE = "-v" in sys.argv

sys.path.insert(0, str(SCRIPTS))
import conformance  # noqa: E402
from vega_core import Level  # noqa: E402


# --------------------------------------------------------------------------- the rendered module

def _render(text: str) -> str:
    """Fill every `{{NAME}}` placeholder the way the emitter does (a package name, the module
    names of the reference map) and refuse a leftover, as `emit._render` does."""
    values = {"PACKAGE": PACKAGE, "MODULE_MODEL": "model", "MODULE_INGESTION": "ingestion",
              "MODULE_TRAINING": "training", "MODULE_EVALUATION": "evaluation",
              "MODULE_FEATURES": "features", "MODULE_SERVING": "serving",
              "MODULE_MONITOR": "serving", "MODULE_BASELINES": "model"}
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    leftover = sorted(set(re.findall(r"\{\{([A-Z_]+)\}\}", text)))
    assert not leftover, f"domain.py placeholders left unrendered: {leftover}"
    return text


_WORK = tempfile.TemporaryDirectory(prefix="domain-plans-")
PKG_DIR = Path(_WORK.name) / PACKAGE
PKG_DIR.mkdir()
RENDERED = _render(TEMPLATE.read_text(encoding="utf-8"))
(PKG_DIR / "domain.py").write_text(RENDERED, encoding="utf-8")
(PKG_DIR / "__init__.py").write_text("", encoding="utf-8")
sys.path.insert(0, _WORK.name)
D = importlib.import_module(f"{PACKAGE}.domain")


def _markers_ok(markers) -> list:
    return [m for m in markers if m.level is Level.ERROR]


# --------------------------------------------------------------------------- conformance

def test_rendered_module_is_frozen_c6_and_c7():
    pkg, err = conformance.Package.read(PKG_DIR, {}, {})
    assert err is None, err
    bad = _markers_ok(conformance.c6_domain_is_frozen(pkg))
    assert not bad, [m.line() for m in bad]
    bad = _markers_ok(conformance.c7_no_module_level_mutable_state(pkg))
    assert not bad, [m.line() for m in bad]


def test_c6_still_catches_a_mutable_prediction():
    """The forge negative seeds exactly this text; the template must keep it seedable."""
    anchor = "@dataclass(frozen=True)\nclass Prediction:"
    assert anchor in RENDERED
    with tempfile.TemporaryDirectory() as td:
        pkg_dir = Path(td) / PACKAGE
        pkg_dir.mkdir()
        mutable = RENDERED.replace(anchor, "@dataclass\nclass Prediction:")
        (pkg_dir / "domain.py").write_text(mutable)
        pkg, _ = conformance.Package.read(pkg_dir, {}, {})
        bad = _markers_ok(conformance.c6_domain_is_frozen(pkg))
        assert bad and "Prediction" in bad[0].line(), [m.line() for m in bad]


def _statements(body) -> int:
    """`ast.stmt` counted recursively; a nested def counts once and is measured on its own."""
    n = 0
    for stmt in body:
        n += 1
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for name in ("body", "orelse", "finalbody"):
            n += _statements(getattr(stmt, name, None) or [])
        for handler in getattr(stmt, "handlers", None) or []:
            n += _statements(handler.body)
    return n


def _without_docstring(fn) -> list:
    body = fn.body
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
            and isinstance(first.value.value, str):
        return body[1:]
    return body


def test_size_discipline_c11_c12_c13():
    """The contract's rule (code-contract §5): ≤ 30 statements and ≤ 5 parameters beyond
    self/cls per def, `*args`/`**kwargs` only in `_prefixed` helpers, a class ≤ 200 lines
    (domain.py is exempt from the module-length rule only)."""
    tree = ast.parse(RENDERED)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.end_lineno - node.lineno + 1 > 200:
            found.append(f"C13 {node.name}: {node.end_lineno - node.lineno + 1} lines")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = node.args
        params = [x for x in a.posonlyargs + a.args + a.kwonlyargs if x.arg not in ("self", "cls")]
        star = [x for x in (a.vararg, a.kwarg) if x is not None]
        if len(params) + len(star) > 5:
            found.append(f"C12 {node.name}:{node.lineno} — {len(params) + len(star)} parameters")
        if star and not node.name.startswith("_"):
            found.append(f"C12 {node.name}:{node.lineno} — *args/**kwargs in a public def")
        count = _statements(_without_docstring(node))
        if count > 30:
            found.append(f"C11 {node.name}:{node.lineno} — {count} statements")
    assert not found, found


def test_no_dataset_literal():
    """A shared template names no dataset: the v2.2 class name survives only as the one-release
    alias, and no answer is described as a particular domain's."""
    for pattern in (r"\bIDS\b", r"\bCIDS\b", r"(?i)\bcic-?ids", r"(?i)\bflows?\b", r'"Label"',
                    r"(?i)\btraffic\b", r"(?i)\bintrusion"):
        hits = [line for line in RENDERED.splitlines() if re.search(pattern, line)]
        assert not hits, (pattern, hits)
    alias = [line for line in RENDERED.splitlines()
             if "TrafficClass" in line and not line.startswith("# >>> compat:")]
    assert alias == ["TrafficClass = ClassValue"], alias


# --------------------------------------------------------------------------- enums and factories

def test_batch_keys_stage_kinds_and_families():
    assert [k.value for k in D.BatchKey] == ["x_num", "x_cat", "y", "edge_index", "edge_weight",
                                             "target_index", "graph_index", "num_graphs"]
    assert [k.value for k in D.StageKind] == ["emb", "concat", "tokens", "tensor", "graph",
                                              "gdrop", "gpool"]
    assert {"x_num": 1}[D.BatchKey.X_NUM] == 1          # a str enum looks up as its value
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    named = {f.value for f in D.Family} - {"user-defined"}
    assert named == set(catalog["families"]), sorted(named ^ set(catalog["families"]))
    assert len(named) == 21
    assert D.Family.of("gcn") is D.Family.GCN and D.Family.of("mine") is D.Family.USER_DEFINED


def test_c12_factories_are_reshaped():
    def arity(fn):
        return len([p for p in inspect.signature(fn).parameters if p not in ("self", "cls")])
    assert arity(D.Metrics.from_predictions) == 5
    assert arity(D.Prediction.of) == 5        # the probabilities are the reading's, handed in
    assert arity(D.EpochRecord.of) == 1

    scored = D.ScoredPart(score=np.array([0.2, -1.0]), truth=np.array([1, 0]),
                          row_ids=np.array([10, 11]))
    seen = []

    def metric(name, y_true, y_score, options):
        seen.append((name, options))
        return float(len(y_true))
    opts = D.MetricOptions(threshold=0.4)
    m = D.Metrics.from_predictions("valid", scored, ["n", "m"], metric, opts)
    assert m.as_dict() == {"n": 2.0, "m": 2.0} and m.run is None
    assert seen == [("n", opts), ("m", opts)]
    stamped = m.with_provenance("R3", 1)
    assert (stamped.run, stamped.exposure, stamped.values) == ("R3", 1, m.values)

    rec = D.EpochRecord.of({"epoch": 2, "time_s": 0.5, "lr": 1e-3, "train": {"loss": 0.4},
                            "valid": {"loss": 0.5}, "shuffle": {"kind": "global-randperm"}})
    assert rec.as_dict()["shuffle"] == {"kind": "global-randperm"} and rec.nan is False
    _raises(ValueError, "no field", lambda: D.EpochRecord.of({"epoch": 1, "valid_loss": 1}))


def test_prediction_reads_logits():
    two = D.LabelSet(("no", "yes"))
    assert D.Prediction.of(7, 0.3, two).label == two.of("yes") == two.at(1)  # logit ≥ 0
    assert D.Prediction.of(7, -0.3, two).label.name == "no"
    three = D.LabelSet(("a", "b", "c"))
    assert D.Prediction.of(1, [0.1, 2.0, -1.0], three).label.name == "b"
    ranks = D.LabelSet(("low", "mid", "high"))
    assert D.Prediction.of(1, [1.5, -0.2], ranks).label.name == "mid"      # cumulative count
    p = D.Prediction.of(3, np.array([1.0, 2.0]), None, D.Versions("c1", "m1", ["w"]))
    assert (p.score, p.contract_version, p.model_version, p.warnings) == \
        ((1.0, 2.0), "c1", "m1", ("w",))
    direct = D.Prediction(row_id=1, score=[0.5], label=two.at(1))
    assert direct.score == (0.5,) and hash(direct)
    assert D.TrafficClass is D.ClassValue


def test_the_values_hold_what_their_names_promise():
    """The T5 round 2 architect pass: ClassValue's 'built only through LabelSet' was
    convention, Prediction took Versions apart into three loose fields, and DriftReading
    carried its measure and verdict as bare strings."""
    two = D.LabelSet(("no", "yes"))
    _raises(ValueError, "LabelSet.of(name)", lambda: D.ClassValue("seventh", 99))
    _raises(ValueError, "LabelSet.of(name)", lambda: D.ClassValue("yes", 0, ("no", "yes")))
    assert two.at(1) == two.of("yes") and two.at(1).name == "yes" and hash(two.at(1))
    versions = D.Versions("c1", "m1", ["w"])
    p = D.Prediction(row_id=1, score=[0.5], label=two.at(0), versions=versions)
    assert p.versions is versions and (p.contract_version, p.model_version, p.warnings) == \
        ("c1", "m1", ("w",))
    assert D.Prediction(row_id=2, score=[0.1]).versions == D.Versions()
    _raises(TypeError, "never its parts",
            lambda: D.Prediction(row_id=1, score=[0.5], versions=("c1", "m1")))
    r = D.DriftReading(column="b04", measure="psi", value=0.3, verdict="alert")
    assert r.measure is D.DriftMeasure.PSI and r.verdict is D.DriftVerdict.ALERT
    assert r.verdict in ("watch", "alert") and f"{r.measure}={r.value} ({r.verdict})" == \
        "psi=0.3 (alert)" and json.dumps({"v": r.verdict}) == '{"v": "alert"}'
    _raises(ValueError, "drifted", lambda: D.DriftReading("b04", "psi", 0.3, "drifted"))
    # where a reading comes from is typed (the T5 round 3 bundle audit): a ReadingProvenance,
    # never an association list a consumer re-reads by string keys
    origin = D.ReadingProvenance(check="c1", concern="b04 drift", title="b04's PSI", window_rows=7)
    assert D.DriftReading("b04", "psi", 0.3, "alert", origin).window.window_rows == 7
    _raises(TypeError, "ReadingProvenance",
            lambda: D.DriftReading("b04", "psi", 0.3, "alert", (("rows", 7),)))
    # the improve pass: a batch form, an export format and a selection direction are closed
    # sets too, and print as their words
    assert [f.value for f in D.BatchForm] == ["row", "window", "node", "graph"]
    assert f"{D.ExportFormat.ONNX}" == "onnx" and D.SelectionDirection("max") is \
        D.SelectionDirection.MAX
    _raises(ValueError, "maximise", lambda: D.SelectionDirection("maximise"))
    # the improve pass: a key column (it groups rows) is no numeric feature, for every reader
    fc = D.FeatureContract(input_dim=2, label="y", features=({"name": "series", "role": "key"},
                                                         {"name": "x0", "role": "numeric"},
                                                         {"name": "zone", "role": "nominal", "cardinality": 3}),
                           manifest_digest="sha256:" + "0" * 64)
    assert fc.numeric_columns == ("x0",) and fc.key_columns == ("series",)
    assert fc.nominal_columns == ("zone",)


def test_task_label_format_and_encode_multi():
    spec = {"task": {"kind": "sequence-classification", "base_kind": "multilabel",
                     "classes": ["a", "b", "c"], "label": "tags"},
            "loss": {"name": "bce_with_logits"}}
    task = D.Task.from_spec(spec)
    assert task.effective is D.TaskKind.MULTILABEL and task.is_sequence
    assert task.label_format == D.LabelFormat(D.LabelFormatKind.DELIMITED, "|")
    legacy = D.Task.from_spec({"task": {"kind": "sequence-regression", "classes": [1, 2, 3]}})
    assert legacy.effective is D.TaskKind.ORDINAL                   # v2.2 spec without base_kind
    svdd = D.Task.from_spec({"task": {"kind": "anomaly"}, "loss": {"name": "svdd"}})
    assert svdd.objective is D.Objective.SVDD and svdd.effective is D.TaskKind.ANOMALY
    _raises(ValueError, "base_kind", lambda: D.Task.from_spec(
        {"task": {"kind": "sequence-regression", "base_kind": "binary"}}))
    _raises(ValueError, "two classes", lambda: D.Task(D.TaskKind.BINARY, labels=("a", "b", "c")))
    _raises(ValueError, "delimited", lambda: D.Task(
        D.TaskKind.MULTILABEL, labels=("a", "b"), label_format=D.LabelFormat("names")))
    k_hot = D.LabelSet(("a", "b", "c")).encode_multi(["a|c", "", None, float("nan"), " b "])
    assert k_hot.dtype == np.float32 and k_hot.tolist() == [[1, 0, 1], [0, 0, 0], [0, 0, 0],
                                                            [0, 0, 0], [0, 1, 0]]
    _raises(ValueError, "zz", lambda: D.LabelSet(("a", "b")).encode_multi(["a|zz"]))


def test_task_carries_the_spec_label_set():
    """Final rehearsal round 2 (architect, 3 of 3 audits): `Task` kept the label set as a bare
    tuple and `LabelSet` was emitted and dormant. The task now holds the spec's `LabelSet`,
    built once through `LabelSet.from_spec`, and every reader of the classes reads it."""
    spec = {"task": {"kind": "multiclass", "classes": ["water", "crop", "bare"], "label": "c"},
            "loss": {"name": "cross_entropy"}}
    task = D.Task.from_spec(spec)
    assert isinstance(task.labels, D.LabelSet) and task.labels == D.LabelSet.from_spec(spec)
    assert len(task.labels) == 3 and task.labels.at(1).name == "crop"
    assert not hasattr(task, "classes")
    assert task.encode_labels(np.asarray(["bare", "water"])).tolist() == [2, 0]
    assert D.Task(D.TaskKind.MULTICLASS, labels=["a", "b"]).labels == D.LabelSet(("a", "b"))
    _raises(ValueError, "repeats", lambda: D.Task(D.TaskKind.MULTICLASS, labels=("a", "a")))


def test_sample_spec_keys_and_sample_index():
    row = D.SampleSpec.from_spec({"task": {"kind": "binary"}, "data_contract": {}})
    assert row.shape is D.SampleShape.ROW and row.keys == ()
    win = D.SampleSpec.from_spec({"task": {"kind": "forecast"}, "data_contract": {"sample": {
        "shape": "window", "length": 24, "horizon": 3, "series_by": "station", "order_by": "ts",
        "season": 24}}})
    assert win.keys == ("station", "ts") and win.window.autoregressive          # the default
    assert win.window.context is D.WindowContext.PREVIOUS_PART
    assert win.window.offsets == (1, 2, 3) and win.window.history_rows(2) == 25
    graph = D.SampleSpec.from_spec({"task": {"kind": "binary"}, "data_contract": {"sample": {
        "shape": "graph", "edges": "shaped/x.edges.parquet", "level": "graph",
        "graph_id": "molecule"}}})
    assert graph.is_graph_level and graph.keys == ("molecule",)
    assert graph.graph.batching is D.GraphBatching.GRAPHS
    idx = D.SampleIndex.of_rows(3)
    assert idx.rows == (0, 1, 2) and len(idx) == 3 and idx.sample_of(2) == 2
    windows = D.SampleIndex("window", np.array([5, 6]), spans=np.array([[3, 5], [4, 6]]))
    assert windows.spans == ((3, 5), (4, 6))
    _raises(ValueError, "no sample", lambda: windows.sample_of(9))
    _raises(ValueError, "spans", lambda: D.SampleIndex("window", (1, 2), spans=((0, 1),)))


# --------------------------------------------------------------------------- windows

def _series_part(seed=0, series=("s1", "s2", "s3"), n=(9, 6, 11)):
    """A part of rows from several series, shuffled, ordered by a time column; raw positions
    are arbitrary distinct integers (as the shaper preserves them)."""
    rng = np.random.default_rng(seed)
    keys = np.concatenate([[s] * k for s, k in zip(series, n)])
    order = np.concatenate([np.arange(k) + 1000 for k in n])
    perm = rng.permutation(len(keys))
    raw = rng.choice(10_000, size=len(keys), replace=False)
    return D.SeriesRows(raw_pos=raw[perm], series=keys[perm], order=order[perm])


def _context(part_rows, per_series=4, seed=1):
    """Earlier rows of the same series (order values before the part's)."""
    rng = np.random.default_rng(seed)
    keys = sorted({str(s) for s in part_rows.series})
    series = np.repeat(keys, per_series)
    order = np.tile(np.arange(per_series), len(keys))
    raw = 50_000 + rng.choice(1000, size=len(series), replace=False)
    return D.SeriesRows(raw_pos=raw, series=series, order=order)


def _flat_series(layout, part_rows, context_rows):
    return layout.take(context_rows.series if context_rows else None, part_rows.series)


def test_no_window_crosses_series_by():
    part, ctx = _series_part(), None
    ctx = _context(part)
    spec = D.WindowSpec(length=4, series_by="series", order_by="ts")
    layout = spec.layout(part, ctx)
    plan = spec.plan(layout.blocks, dense=True)
    flat = _flat_series(layout, part, ctx)
    inputs = plan.inputs()
    assert len(plan) > 0
    for w in range(len(plan)):
        assert len({str(v) for v in flat[inputs[w]]}) == 1, (w, flat[inputs[w]])
    for block in layout.blocks:                # each block is one series, contiguous
        span = flat[block.start:block.stop]
        assert set(span.tolist()) == {block.key}
    # the order within a series follows order_by
    order = layout.take(ctx.order, part.order)
    for block in layout.blocks:
        rows = order[block.start + block.n_context:block.stop]
        assert np.all(np.diff(rows) > 0)


def test_context_rows_are_never_anchors():
    part = _series_part()
    ctx = _context(part)
    spec = D.WindowSpec(length=5, series_by="series", order_by="ts")
    layout = spec.layout(part, ctx)
    context_positions = {p for b in layout.blocks for p in range(b.start, b.start + b.n_context)}
    assert context_positions and all(b.n_context == 4 for b in layout.blocks)
    for dense in (True, False):
        plan = spec.plan(layout.blocks, dense=dense)
        assert not context_positions & set(plan.anchors.tolist())
        assert set(plan.row_ids.tolist()) <= set(part.raw_pos.tolist())
    # context rows come from the context frame, part rows from the part frame
    for b in layout.blocks:
        assert all(r < layout.n_context_frame for r in layout.rows[b.start:b.start + b.n_context])
        assert all(r >= layout.n_context_frame
                   for r in layout.rows[b.start + b.n_context:b.stop])


def test_context_is_the_tail_of_the_previous_parts():
    """Test context: the valid tail first, then train when valid is too short (stored order)."""
    spec = D.WindowSpec(length=4)
    train = np.arange(0, 10)
    valid = np.arange(10, 12)
    ctx = D.SeriesRows(raw_pos=np.concatenate([train, valid]))
    part = D.SeriesRows(raw_pos=np.arange(12, 20))
    layout = spec.layout(part, ctx)
    borrowed = layout.take(ctx.raw_pos, part.raw_pos)[:layout.blocks[0].n_context]
    assert borrowed.tolist() == [9, 10, 11]
    plan = spec.plan(layout.blocks, dense=True)
    assert plan.row_ids.tolist() == list(range(12, 20)) and plan.unscored == ()
    none = D.WindowSpec(length=4, context="none")
    alone = none.plan(none.layout(part, ctx).blocks, dense=True)
    assert [u.raw_pos for u in alone.unscored] == [12, 13, 14]
    assert {u.reason for u in alone.unscored} == {D.UnscoredReason.INSUFFICIENT_CONTEXT}
    _raises(ValueError, "same context frame", lambda: layout.take(None, part.raw_pos))
    keyed = D.WindowSpec(length=2, series_by="station", order_by="ts")
    _raises(ValueError, "station column is missing at row 21", lambda: keyed.layout(
        D.SeriesRows(raw_pos=[20, 21], series=["a", None], order=[1, 2])))
    _raises(ValueError, "ts column is missing at row 30", lambda: keyed.layout(
        D.SeriesRows(raw_pos=[30, 31], series=["a", "a"], order=[None, 5])))


def test_forecast_targets_never_cross_the_part_end():
    part = _series_part(seed=3)
    ctx = _context(part)
    spec = D.WindowSpec(length=3, horizon=4, series_by="series", order_by="ts",
                        autoregressive=True)
    layout = spec.layout(part, ctx)
    plan = spec.plan(layout.blocks, dense=True)
    assert plan.targets.shape == (len(plan), 4)
    for w, series in enumerate(plan.series_index):
        block = layout.blocks[series]
        first_part_row = block.start + block.n_context
        assert np.all(plan.targets[w] > plan.anchors[w])
        assert np.all(plan.targets[w] >= first_part_row) and np.all(plan.targets[w] < block.stop)
    beyond = [u for u in plan.unscored if u.reason is D.UnscoredReason.HORIZON_BEYOND_PART]
    assert len(beyond) == 4 * len(layout.blocks)           # the last H rows of every series
    sequence = D.WindowSpec(length=3, series_by="series", order_by="ts")
    own = sequence.plan(layout.blocks, dense=True)
    assert np.array_equal(own.targets[:, 0], own.anchors)  # a sequence task: its own label


def test_stride_keeps_the_most_recent_row():
    part = _series_part(seed=5, n=(17, 10, 23))
    spec = D.WindowSpec(length=4, stride=3, series_by="series", order_by="ts", context="none")
    layout = spec.layout(part)
    dense = spec.plan(layout.blocks, dense=True)
    thin = spec.plan(layout.blocks, dense=False)
    for b, block in enumerate(layout.blocks):
        mine = thin.anchors[thin.series_index == b]
        every = dense.anchors[dense.series_index == b]
        assert mine[-1] == block.stop - 1                    # the series' most recent row
        assert set(mine.tolist()) <= set(every.tolist())
        assert np.all(np.diff(mine) == 3)
        assert len(mine) == -(-len(every) // 3)
    horizon = D.WindowSpec(length=4, stride=2, horizon=2, series_by="series", order_by="ts",
                           context="none")
    thin_h = horizon.plan(layout.blocks, dense=False)
    for b, block in enumerate(layout.blocks):
        assert thin_h.anchors[thin_h.series_index == b][-1] == block.stop - 1 - 2


def test_dense_scoring_gives_one_anchor_per_scorable_row():
    part = _series_part(seed=7)
    ctx = _context(part, per_series=2)
    for spec in (D.WindowSpec(length=5, series_by="series", order_by="ts"),
                 D.WindowSpec(length=5, horizon=2, series_by="series", order_by="ts"),
                 D.WindowSpec(length=3, stride=4, series_by="series", order_by="ts",
                              context="none")):
        layout = spec.layout(part, ctx)
        plan = spec.plan(layout.blocks, dense=True)
        answered = plan.row_ids.tolist()
        skipped = [u.raw_pos for u in plan.unscored]
        assert len(answered) == len(set(answered))
        assert not set(answered) & set(skipped)
        assert sorted(answered + skipped) == sorted(part.raw_pos.tolist())
        # the anchor's raw position is the row id
        flat_raw = layout.take(ctx.raw_pos, part.raw_pos)
        assert np.array_equal(flat_raw[plan.anchors], plan.row_ids)


def test_autoregressive_channel_reads_only_rows_up_to_the_anchor():
    part = _series_part(seed=11)
    ctx = _context(part)
    spec = D.WindowSpec(length=4, horizon=3, series_by="series", order_by="ts",
                        autoregressive=True)
    layout = spec.layout(part, ctx)
    plan = spec.plan(layout.blocks, dense=True)
    label = np.random.default_rng(0).normal(size=len(layout.rows))
    inputs = plan.inputs()
    assert np.all(inputs.max(axis=1) == plan.anchors)
    for w in range(len(plan)):
        future = label.copy()
        future[plan.anchors[w] + 1:] += 100.0                   # perturb every later row
        assert np.array_equal(label[inputs[w]], future[inputs[w]])
        assert not np.array_equal(label[plan.targets[w]], future[plan.targets[w]])
    _raises(ValueError, "forecast-only", lambda: D.WindowSpec(length=4, autoregressive=True))
    _raises(ValueError, "forecast-only", lambda: D.WindowSpec.from_spec(
        {"task": {"kind": "sequence-classification"},
         "data_contract": {"sample": {"shape": "window", "length": 4, "autoregressive": True}}}))
    _raises(ValueError, "horizon", lambda: D.WindowSpec.from_spec(
        {"task": {"kind": "forecast"}, "data_contract": {"sample": {"shape": "window",
                                                                     "length": 4}}}))
    _raises(ValueError, "length", lambda: D.WindowSpec(length=1))


def test_windowed_part_cuts_the_plan():
    torch = _torch()
    part = _series_part(seed=13)
    ctx = _context(part)
    spec = D.WindowSpec(length=4, horizon=2, series_by="series", order_by="ts",
                        autoregressive=True)
    layout = spec.layout(part, ctx)
    plan = spec.plan(layout.blocks, dense=True)
    m = len(layout.rows)
    x = torch.arange(m * 3, dtype=torch.float32).reshape(m, 3)
    cat = torch.arange(m * 2).reshape(m, 2)
    y = torch.arange(m, dtype=torch.float32) * 10
    held = D.WindowedPart(x_num=x, x_cat=cat, y=y, anchors=torch.as_tensor(plan.anchors),
                          targets=torch.as_tensor(plan.targets), row_ids=plan.row_ids,
                          length=plan.length, unscored=plan.unscored)
    idx = torch.tensor([0, 5, 2])
    batch = held.gather(idx)
    want = plan.inputs()[[0, 5, 2]]
    assert batch["x_num"].shape == (3, 4, 3) and batch["x_cat"].shape == (3, 4, 2)
    assert torch.equal(batch["x_num"], x[torch.as_tensor(want)])
    assert torch.equal(batch["y"], y[torch.as_tensor(plan.targets[[0, 5, 2]])])
    chunks = list(held.slices(4))
    assert sum(c["x_num"].shape[0] for c in chunks) == len(plan)
    assert torch.equal(chunks[0]["x_num"], x[torch.as_tensor(plan.inputs()[:4])])
    one = D.WindowedPart(x_num=x, x_cat=None, y=y, anchors=torch.as_tensor(plan.anchors),
                         targets=torch.as_tensor(plan.anchors[:, None]), row_ids=plan.row_ids,
                         length=plan.length)
    assert one.gather(idx)["y"].shape == (3,)
    assert len(held.to("cpu")) == len(plan) and held.to("cpu").steps.tolist() == [-3, -2, -1, 0]


# --------------------------------------------------------------------------- graphs

SIZES = (30, 10, 12)          # train, valid, test


def _graph(seed=0, weighted=True):
    """Symmetric random edges over all three parts (train–valid, *–test, test–test included),
    some duplicates and self-loops, and a few one-way edges whose source is the query end
    (valid → train, test → valid); raw positions interleaved across the parts."""
    rng = np.random.default_rng(seed)
    total = sum(SIZES)
    pairs = rng.integers(0, total, size=(90, 2))
    one_way = [[32, 1], [36, 5], [41, 33], [47, 2]]
    pairs = np.concatenate([pairs, pairs[:, ::-1], [[3, 3], [35, 35], [45, 45]], one_way])
    weight = rng.uniform(0.1, 1.0, size=len(pairs)) if weighted else None
    edges = D.GraphEdges(src=pairs[:, 0], dst=pairs[:, 1], weight=weight)
    raw = rng.permutation(total) * 7 + 1               # a raw position per global id
    return edges, raw


def _parts_of(ids):
    bounds = np.cumsum(SIZES)
    return np.searchsorted(bounds, ids, side="right")


def test_inductive_train_view_holds_no_test_raw_position():
    edges, raw = _graph()
    view = D.GraphSpec().view(SIZES, edges, "train")
    test_raw = set(raw[SIZES[0] + SIZES[1]:].tolist())
    seen_raw = set(raw[view.node_ids].tolist())
    assert not seen_raw & test_raw
    assert set(view.node_ids.tolist()) == set(range(SIZES[0]))
    ends = view.node_ids[view.edge_index]
    assert np.all(_parts_of(ends) == 0)
    touches_test = (_parts_of(edges.src) == 2) | (_parts_of(edges.dst) == 2)
    assert view.test_edges_withheld == int(touches_test.sum()) > 0
    assert view.feature_parts == (D.Part.TRAIN,) and view.label_parts == (D.Part.TRAIN,)
    assert np.array_equal(view.target_index, np.arange(SIZES[0]))
    assert np.allclose(view.edge_weight, edges.weight[view.edge_rows])
    valid = D.GraphSpec().view(SIZES, edges, "valid")
    assert not set(raw[valid.node_ids].tolist()) & test_raw
    assert D.GraphSpec().exposure(SIZES, edges).as_dict() == {
        "protocol": "inductive", "edges_into_test": 0, "features_seen_in_training": False}


def _unordered(src, dst):
    return {(min(a, b), max(a, b)) for a, b in zip(src.tolist(), dst.tolist())}


def test_valid_view_is_the_train_graph_plus_valid_queries():
    edges, _ = _graph(seed=1)
    view = D.GraphSpec().view(SIZES, edges, "valid")
    ntr = SIZES[0]
    assert view.n_reference == ntr
    assert np.all(_parts_of(view.reference_ids) == 0) and np.all(_parts_of(view.query_ids) == 1)
    assert np.array_equal(view.node_ids[view.target_index], view.query_ids)
    src, dst = view.node_ids[view.edge_index[0]], view.node_ids[view.edge_index[1]]
    ps, pd = _parts_of(src), _parts_of(dst)
    assert not np.any(ps == 1), "a query sends no message"
    assert np.all(((ps == 0) & (pd == 0)) | ((ps == 0) & (pd == 1)))
    # reference edges: exactly the stored train–train edges, in stored orientation
    stored = [(a, b) for a, b in zip(edges.src.tolist(), edges.dst.tolist())
              if a < ntr and b < ntr]
    assert [(a, b) for a, b, p, q in zip(src.tolist(), dst.tolist(), ps, pd) if q == 0] == stored
    # attachments: every stored train–valid pair once, reference → query
    ep, eq = _parts_of(edges.src), _parts_of(edges.dst)
    cross = ((ep == 0) & (eq == 1)) | ((ep == 1) & (eq == 0))
    want = _unordered(edges.src[cross], edges.dst[cross])
    att = [(a, b) for a, b, q in zip(src.tolist(), dst.tolist(), pd) if q == 1]
    assert len(att) == len(set(att)) and _unordered(*np.array(att).T) == want


def test_sealed_test_view_reference_is_train_and_valid():
    edges, _ = _graph(seed=2)
    view = D.GraphSpec().view(SIZES, edges, "test")
    n_ref = SIZES[0] + SIZES[1]
    assert view.n_reference == n_ref and view.test_edges_withheld == 0
    assert set(view.reference_ids.tolist()) == set(range(n_ref))
    assert np.all(_parts_of(view.query_ids) == 2)
    assert np.array_equal(view.node_ids[view.target_index], view.query_ids)
    assert view.feature_parts == (D.Part.TRAIN, D.Part.VALID, D.Part.TEST)
    src, dst = view.node_ids[view.edge_index[0]], view.node_ids[view.edge_index[1]]
    ps, pd = _parts_of(src), _parts_of(dst)
    assert not np.any(ps == 2) and not np.any((pd == 2) & (ps == 2))
    inside = [(a, b) for a, b in zip(edges.src.tolist(), edges.dst.tolist())
              if a < n_ref and b < n_ref]
    assert [(a, b) for a, b, q in zip(src.tolist(), dst.tolist(), pd) if q < 2] == inside
    ep, eq = _parts_of(edges.src), _parts_of(edges.dst)
    cross = ((ep < 2) & (eq == 2)) | ((ep == 2) & (eq < 2))
    att = [(a, b) for a, b, q in zip(src.tolist(), dst.tolist(), pd) if q == 2]
    assert _unordered(*np.array(att).T) == _unordered(edges.src[cross], edges.dst[cross])


def test_served_reference_graph_rebuilds_the_sealed_test_view():
    """`GraphReference.attach` on the train ∪ valid nodes (dense ref ids, no raw position) and
    the test rows' links gives exactly the sealed test view's graph: one rule, twice."""
    pd_ = _pandas()
    edges, _ = _graph(seed=4)
    spec = {"task": {"kind": "multiclass", "label": "y", "classes": ["a", "b"]},
            "input": {"manifest_digest": "sha256:" + "0" * 64},
            "data_contract": {"input_dim": 3, "features": [
                {"name": "f1", "role": "numeric"}, {"name": "f2", "role": "numeric"},
                {"name": "zone", "role": "nominal", "cardinality": 4}]}}
    contract = D.FeatureContract.from_spec(spec)
    n_ref = SIZES[0] + SIZES[1]
    rng = np.random.default_rng(0)
    order = rng.permutation(n_ref)
    nodes = pd_.DataFrame({"ref": order, "f1": rng.normal(size=n_ref),
                           "f2": rng.normal(size=n_ref), "zone": rng.integers(0, 4, size=n_ref)})
    inside = (edges.src < n_ref) & (edges.dst < n_ref)
    edge_frame = pd_.DataFrame({"src_ref": edges.src[inside], "dst_ref": edges.dst[inside],
                                "weight": edges.weight[inside]})
    reference = D.GraphReference.from_frames(nodes, edge_frame, contract)
    assert reference.n_nodes == n_ref and reference.x_cat.shape == (n_ref, 1)
    assert np.allclose(reference.x_num[order[0]], nodes.loc[0, ["f1", "f2"]].to_numpy(float))
    ep, eq = _parts_of(edges.src), _parts_of(edges.dst)
    forward, backward = (ep < 2) & (eq == 2), (ep == 2) & (eq < 2)
    rows = np.flatnonzero(forward | backward)
    links = D.QueryEdges(query=np.where(forward, edges.dst, edges.src)[rows] - n_ref,
                         ref=np.where(forward, edges.src, edges.dst)[rows],
                         weight=edges.weight[rows])
    served = reference.attach(links, SIZES[2])
    sealed = D.GraphSpec().view(SIZES, edges, "test")
    assert np.array_equal(served.edge_index, sealed.edge_index)
    assert np.allclose(served.edge_weight, sealed.edge_weight)
    assert np.array_equal(served.target_index, sealed.target_index)
    assert served.n_nodes == sealed.n_nodes
    alone = reference.attach(D.QueryEdges(query=links.query[links.query == 0],
                                          ref=links.ref[links.query == 0],
                                          weight=links.weight[links.query == 0]), 1)
    assert alone.n_nodes == n_ref + 1 and alone.target_index.tolist() == [n_ref]
    gap = nodes.assign(ref=nodes["ref"] + (nodes["ref"] == 0))      # 1 twice, 0 missing
    _raises(ValueError, "dense", lambda: D.GraphReference.from_frames(gap, edge_frame, contract))
    assert not hasattr(reference, "raw_pos")


def test_transductive_views_and_their_exposure():
    edges, _ = _graph(seed=6)
    _raises(D.SealedPartError, "user:", lambda: D.GraphSpec(protocol="transductive"))
    _raises(D.SealedPartError, "user:", lambda: D.GraphSpec(
        protocol="transductive", declared_exposure={"kind": "features", "source": "vega:default"}))
    spec = D.GraphSpec(protocol="transductive",
                       declared_exposure={"kind": "features", "rows": "test", "source": "user:t4"})
    view = spec.view(SIZES, edges, "valid")
    assert view.n_nodes == sum(SIZES) and view.n_edges == len(edges)
    assert np.all(_parts_of(view.node_ids[view.target_index]) == 1)
    exposure = spec.exposure(SIZES, edges)
    into = int((_parts_of(edges.dst) == 2).sum())
    assert exposure.edges_into_test == into > 0 and exposure.features_seen_in_training
    _raises(ValueError, "transductive", lambda: D.GraphSpec(
        declared_exposure={"kind": "features", "source": "user:t4"}))


def test_graph_level_refusals_name_the_id():
    spec = D.GraphSpec(level="graph", graph_id="molecule", batching="graphs")
    # graphs of the train part: g1 = nodes 0-2, g2 = nodes 3-4
    ids = np.array(["g1", "g1", "g1", "g2", "g2"])
    groups = D.GraphSpec.graphs(ids, np.array([[0, 1, 3], [1, 2, 4]]), np.array([1, 1, 1, 0, 0]))
    assert groups.values == ("g1", "g2") and groups.graph_index.tolist() == [0, 0, 0, 1, 1]
    assert groups.labels_of(np.array([1, 1, 1, 0, 0])).tolist() == [1, 0]
    _raises(ValueError, "'g1' and graph 'g2'",
            lambda: D.GraphSpec.graphs(ids, np.array([[0, 2], [1, 3]])))
    _raises(ValueError, "graph 'g2'",
            lambda: D.GraphSpec.graphs(ids, None, np.array([1, 1, 1, 0, 1])))
    _raises(ValueError, "graph 'g7' straddles parts train/test",
            lambda: D.GraphSpec.check_split({"train": ["g1", "g7"], "valid": ["g3"],
                                             "test": ["g7", "g9"]}))
    D.GraphSpec.check_split({"train": ["g1", "g1"], "test": ["g2"]})
    sizes = (5, 3, 4)
    inside = D.GraphEdges(src=[0, 3, 5, 8, 9], dst=[1, 4, 6, 9, 11])
    view = spec.view(sizes, inside, "train")
    assert view.node_ids.tolist() == [0, 1, 2, 3, 4] and view.target_index is None
    assert view.edge_index.tolist() == [[0, 3], [1, 4]] and view.test_edges_withheld == 2
    assert spec.view(sizes, inside, "test").edge_index.tolist() == [[0, 1], [1, 3]]
    across = D.GraphEdges(src=[0, 4], dst=[1, 9])
    _raises(ValueError, "joins parts train/test", lambda: spec.view(sizes, across, "train"))
    _raises(ValueError, "graph_id", lambda: D.GraphSpec(level="graph", batching="graphs"))
    _raises(ValueError, "neighbours", lambda: D.GraphSpec(batching="sampled"))
    assert D.GraphSpec(batching="sampled", neighbours=[25, 10]).neighbours == (25, 10)


def test_graph_part_batches():
    torch = _torch()
    x = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    edge_index = torch.tensor([[0, 1, 2, 3, 5, 6], [1, 0, 3, 4, 6, 5]])
    graph_index = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    part = D.GraphPart(x_num=x, x_cat=None, edge_index=edge_index,
                       edge_weight=torch.ones(6), target_index=None,
                       y=torch.tensor([1.0, 0.0, 1.0]), row_ids=np.array(["a", "b", "c"]),
                       graph_index=graph_index, num_graphs=3)
    assert len(part) == 3 and part.row_id_kind is D.RowIdKind.GRAPH_ID
    batch = part.gather(torch.tensor([2, 0]))
    assert batch["num_graphs"] == 2
    assert batch["x_num"].tolist() == x[[0, 1, 5, 6]].tolist()
    assert batch["graph_index"].tolist() == [1, 1, 0, 0]              # graph 2 first, then 0
    assert batch["edge_index"].tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert batch["y"].tolist() == [1.0, 1.0]
    whole = part.as_batch()
    assert torch.equal(whole["edge_index"], edge_index) and whole["num_graphs"] == 3
    assert [b["num_graphs"] for b in part.slices(2)] == [2, 1]
    node = D.GraphPart(x_num=x, x_cat=None, edge_index=edge_index, edge_weight=None,
                       target_index=torch.tensor([4, 5, 6]), y=torch.tensor([0, 1, 2]),
                       row_ids=np.array([40, 50, 60]))
    got = node.gather(torch.tensor([2, 0]))
    assert got["target_index"].tolist() == [6, 4] and got["y"].tolist() == [2, 0]
    assert torch.equal(got["x_num"], x) and "edge_weight" not in got
    assert [b["target_index"].tolist() for b in node.slices(2)] == [[4, 5], [6]]
    _raises(ValueError, "not both", lambda: D.GraphPart(x, None, edge_index, None,
                                                        torch.tensor([0]), None, None,
                                                        graph_index=graph_index, num_graphs=3))


def test_tensor_part_keeps_the_resident_path():
    torch = _torch()
    held = D.ResidentPart(x_num=torch.ones(5, 2), x_cat=None, y=torch.arange(5), device="cpu",
                          digest="sha256:x")
    assert D.ResidentPart is D.TensorPart and len(held) == 5
    assert held.gather(torch.tensor([4, 1]))["y"].tolist() == [4, 1]
    assert [len(b["y"]) for b in held.slices(2)] == [2, 2, 1]
    moved = held.to("cpu")
    assert moved.device == "cpu" and moved.digest == "sha256:x"
    assert D.TensorPart(torch.zeros(3, 1), None, None)[0:2]["x_num"].shape == (2, 1)


# --------------------------------------------------------------------------- runner

def _raises(kind, needle, fn):
    try:
        fn()
    except kind as exc:
        assert needle in str(exc), f"{kind.__name__} raised without {needle!r}: {exc}"
        return
    raise AssertionError(f"expected {kind.__name__} mentioning {needle!r}")


def _torch():
    import torch
    return torch


def _pandas():
    import pandas
    return pandas


def main(argv) -> int:
    pick = argv[argv.index("-k") + 1] if "-k" in argv else ""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and pick in n]
    failed = []
    for name, fn in tests:
        try:
            fn()
            if VERBOSE:
                print(f"  ok    {name}")
        except Exception as exc:                            # a failing test is reported, not raised
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"\n{len(failed)} of {len(tests)} domain plan test(s) failed")
        return 1
    print(f"\n{len(tests)} domain plan test(s) passed on the rendered domain.py")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
