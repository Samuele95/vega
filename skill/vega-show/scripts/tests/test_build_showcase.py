#!/usr/bin/env python3
"""
test_build_showcase.py — the showcase build and its check (H06: `build_showcase.py`,
`showcase_check.py`, `attribution_build.py`; contract §4.8, §5, §10; build plan R-D9, R-D11,
R-E1, R-E2, R-E11).

    python3 -m pytest -q vega-show/scripts/tests/test_build_showcase.py
    VEGA_SHOW_LANES=<matrix dir> python3 -m pytest -q …/test_build_showcase.py -k lane

- **The attribution** (`attribution_build`): the weights axis of every reading (classes, the
  binary sides with the positive logit, labels, the target's bins, the steps ahead, the groups
  of the reconstruction and anomaly lanes), the target each column's rows explain, `correct`,
  one request per column for rows and per item for windows, the reduction per sample shape,
  `vega/attribution@2` valid; the demo sample's row scores.
- **The build** on a packed bundle of the coverage fixtures (a multiclass row lane, brief links
  and a linked report): `page/` a byte copy (`verify_template_copy` empty), `SHOWCASE.json`
  valid, `served ⊆ files`, the closed world as the slot sees it (the service template's own
  `ShowcaseSlot` opens it verified), the bundle's `MANIFEST.json` untouched; the swap keeping
  the previous showcase; the pseudo-locale build under `.work/qps/` only; fork S1 (`FORK:`
  without an answer, nothing written); the refusals PF5, PF9, `resolve` (a story the plan does
  not resolve to), PG7, BD1, BD2, BD3, BD4, BD8.
- **The check**: pass, broken (CK2–CK5) and stale (CK6–CK9), the template WARN (CK10).
- **D5's seeded defects 25–32** (the logic of their gates): 25 a `.py` in the showcase (the
  build refuses BD3, the check says broken CK3); 26 a served path not in files (CK4, and the
  slot refuses); 27 the manifest digest off by one hex (CK5, the slot refuses); 28 the report's
  PDF changed (stale, CK7); 29 a role map older than its `.aux` (PF6 at extraction); 30 a
  template with a leftover `{{TOKEN}}` (PG7); 31 a template holding a class name (PG3, the
  literal gate); 32 `story.json` edited after the build (CK2).
- **The lanes** (with `VEGA_SHOW_LANES`, a matrix directory whose `lanes/L01/work` and
  `lanes/L02/work` went through `pack`): L01 in brief-link mode, L02 with the stub report —
  extract, draft, author, resolve, build and check on a copy of the lane; `page/` a byte copy,
  `SHOWCASE.json` valid, `served ⊆ files`, the closed world, the bundle `MANIFEST.json` digest
  unchanged, the attribution's columns the classes; the bundle's own service serving it
  verified (`/v1/showcase`, `/`, the story, the report); then the transitions: a report that
  appears (stale), the PDF touched (stale), `story.json` altered (broken, the slot refuses, the
  model stays ready), the manifest digest off by one hex (broken), and a real
  `pack.py --re-emit-serving` (stale, the slot refuses with the re-emit reason). A lane packed
  before the inspection routes are served (I08/I09) is first refused by PF5, as it must be;
  its copy then declares the two routes those packages add, so the rest of the build runs.
"""
from __future__ import annotations

import contextlib
import functools
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent))

import attribution_build as ab  # noqa: E402
import autoauthor  # noqa: E402
import build_showcase  # noqa: E402
import coverage_fixtures as cf  # noqa: E402
import draft_plan  # noqa: E402
import extract_facts  # noqa: E402
import page_template  # noqa: E402
import resolve_story  # noqa: E402
import showcase_check as sc  # noqa: E402
import showcase_manifest  # noqa: E402
from page_derive import Catalogs  # noqa: E402
from vega_show_core import CATALOG, digest_bytes, digest_file  # noqa: E402

SKILLS = TESTS.parents[2]
SLOT_TEMPLATE = (SKILLS / "vega-forge" / "scripts" / "backends" / "pytorch" / "templates" /
                 "serving" / "serve" / "routes_showcase.py")
LINT = SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
ATTRIBUTION_SCHEMA = json.loads((TESTS.parent.parent / "references" / "schemas" /
                                 "attribution-2.schema.json").read_text(encoding="utf-8"))
MULTICLASS = {"family": "mlp-plain", "shape": "row", "level": None, "variant": None,
              "task": "multiclass"}
INSPECTION_ROUTES = ("GET /v1/network", "POST /v1/explain")
MODEL = "demo.domain.analysis.shaped.vega.train.eval.pack.html"


def run(main, argv, cwd: Path) -> tuple[int, str]:
    """A command line in-process from `cwd`: (exit, what it printed)."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.chdir(cwd):
        code = main([str(a) for a in argv])
    return code, printed.getvalue()


# =============================================================================== attribution

class FakeInspector:
    """`attribute` as the inspector answers it, recording each request: every row of a request
    is one sample whose attribution is its position among the inputs, answered by `answer`."""

    def __init__(self, shape: str, kind, answer=None):
        self.shape, self.kind, self.answer = shape, kind, answer or (lambda row, target: {})
        self.calls, self.inputs = [], ["a", "b", "zone"]

    def attribute(self, rows, targets, method, steps, history):
        self.calls.append({"rows": list(rows), "targets": targets, "history": history,
                           "method": method, "steps": steps})
        samples = []
        for i, row in enumerate(rows):
            target = targets[i] if targets else None
            values = [0.1 * (i + 1), -0.2, 0.05]
            entry = {"row": values, "window": {"per_step": [0.3, 0.1], "per_feature": values},
                     "graph": {"self": values, "hops": {"0": 0.2}}}[self.shape]
            samples.append({"row": i, "prediction": self.answer(row, target),
                            "target_value": 1.0, "attribution": entry})
        kinds = [self.kind(t) for t in (targets or [None] * len(rows))]
        return {"schema": "vega/attribution-table@1", "inputs": list(self.inputs),
                "targets": [{"kind": k} for k in kinds], "samples": samples}


def demo(strata: list[tuple[str, str, str | None, list | None]], per: int = 2, window: bool = False):
    """A `vega/demo@2` document: strata (id, kind, name, range), `per` items each, the truth the
    stratum's name (a class or label) or a number."""
    items = []
    for sid, kind, name, span in strata:
        for k in range(per):
            row = {"a": k, "b": 1.0, "zone": "n", "__stratum": sid}
            truth = name if kind == "class" else ([name] if kind == "label" else float(k))
            items.append({"stratum": sid, "rows": [row],
                          "history": [dict(row, a=-1)] if window else None, "truth": truth})
    return {"source": "train", "schema": "vega/demo@2",
            "strata": [{"id": s, "kind": k, "name": n, "n": per, "range": r} for s, k, n, r in strata],
            "items": items}


def build_attribution(doc, output, sample, inspector):
    plan = ab.AttributionPlan.of(doc, output, sample)
    return ab.AttributionBuilder(plan, inspector.attribute, lambda rows: rows, output).build()


def valid(doc) -> list[str]:
    from jsonschema import Draft202012Validator
    return [e.message for e in Draft202012Validator(ATTRIBUTION_SCHEMA).iter_errors(doc)]


def test_binary_axis_is_negative_then_positive_on_the_positive_logit():
    doc = demo([("class-0", "class", "late", None), ("class-1", "class", "on-time", None)])
    output = {"reading": "binary", "classes": ["on-time", "late"], "positive": "late",
              "field_kinds": {"label": {"kind": "class", "of": None}}}
    fake = FakeInspector("row", lambda t: "binary_logit", lambda row, t: {"label": "late"})
    out = build_attribution(doc, output, {"shape": "row"}, fake)
    assert valid(out) == []
    assert [c["name"] for c in out["axis"]["columns"]] == ["on-time", "late"]
    assert out["axis"]["kind"] == "class" and out["target"] == "binary_logit" and out["reduce"] is None
    assert all(t == {"class": "late"} for call in fake.calls for t in call["targets"])
    assert len(fake.calls) == 2, "rows travel in one request per column"
    assert out["cells"]["class-0"]["correct"] == 2 and out["cells"]["class-1"]["correct"] == 0
    assert out["inputs"] == ["a", "b", "zone"] and len(out["cells"]["class-0"]["rows"][0]) == 3


def test_multiclass_columns_are_the_classes_each_on_its_true_class():
    names = ["calm", "busy", "jammed"]
    doc = demo([(f"class-{i}", "class", n, None) for i, n in enumerate(names)], per=3)
    output = {"reading": "multiclass", "classes": names,
              "field_kinds": {"label": {"kind": "class", "of": None}}}
    fake = FakeInspector("row", lambda t: "logit", lambda row, t: {"label": "busy"})
    out = build_attribution(doc, output, {"shape": "row"}, fake)
    assert valid(out) == [] and [c["name"] for c in out["axis"]["columns"]] == names
    assert [call["targets"][0] for call in fake.calls] == [{"class": n} for n in names]
    assert {k: (c["n"], c["correct"]) for k, c in out["cells"].items()} == {
        "class-0": (3, 0), "class-1": (3, 3), "class-2": (3, 0)}


def test_ordinal_and_value_lanes_explain_the_default_target():
    doc = demo([("class-0", "class", "low", None), ("class-1", "class", "high", None)])
    fake = FakeInspector("row", lambda t: "expected_rank")
    out = build_attribution(doc, {"reading": "ordinal", "classes": ["low", "high"],
                                  "field_kinds": {}}, {"shape": "row"}, fake)
    assert valid(out) == [] and all(c["targets"] is None for c in fake.calls)
    assert out["cells"]["class-0"]["correct"] == 0, "no class field, nothing judged right"
    bins = demo([("bin-0", "bin", None, [0.0, 1.5]), ("bin-1", "bin", None, [1.5, None])])
    fake = FakeInspector("row", lambda t: "value")
    out = build_attribution(bins, {"reading": "regression", "field_kinds": {}}, {"shape": "row"}, fake)
    assert valid(out) == [] and out["axis"]["kind"] == "bin"
    assert [(c["id"], c["name"], c["range"]) for c in out["axis"]["columns"]] == [
        ("bin-0", "0 – 1.5", [0.0, 1.5]), ("bin-1", "1.5 – …", [1.5, None])]
    assert all(c["correct"] is None for c in out["cells"].values())


def test_labels_skip_the_none_stratum_and_count_the_flag():
    doc = demo([("label-0", "label", "red", None), ("label-1", "label", "blue", None),
                ("none", "none", None, None)])
    output = {"reading": "multilabel", "labels": ["red", "blue"],
              "field_kinds": {"pred_red": {"kind": "flag", "of": "red"},
                              "pred_blue": {"kind": "flag", "of": "blue"}}}
    fake = FakeInspector("row", lambda t: "label_logit", lambda row, t: {"pred_red": 1, "pred_blue": 0})
    out = build_attribution(doc, output, {"shape": "row"}, fake)
    assert valid(out) == [] and [c["id"] for c in out["axis"]["columns"]] == ["label-0", "label-1"]
    assert [call["targets"][0] for call in fake.calls] == [{"label": "red"}, {"label": "blue"}]
    assert (out["cells"]["label-0"]["correct"], out["cells"]["label-1"]["correct"]) == (2, 0)


def test_a_forecast_window_is_one_request_per_item_per_step_summed_over_steps():
    doc = demo([("bin-0", "bin", None, [0.0, 1.0]), ("bin-1", "bin", None, [1.0, 2.0])], window=True)
    fake = FakeInspector("window", lambda t: "forecast_step")
    out = build_attribution(doc, {"reading": "forecast", "horizons": [1, 2], "field_kinds": {}},
                            {"shape": "window"}, fake)
    assert valid(out) == [] and out["reduce"] == "sum_over_steps" and out["axis"]["kind"] == "horizon"
    assert [c["id"] for c in out["axis"]["columns"]] == ["h1", "h2"]
    assert len(fake.calls) == 8 and all(len(c["rows"]) == 1 and c["history"] for c in fake.calls)
    assert {c["targets"][0]["step"] for c in fake.calls} == {1, 2}
    assert out["cells"]["h1"]["n"] == 4


def test_groups_of_the_reconstruction_and_anomaly_lanes():
    unlabelled = demo([("typical", "typical", None, None), ("flagged", "flagged", None, None)])
    fake = FakeInspector("graph", lambda t: "reconstruction_error")
    out = build_attribution(unlabelled, {"reading": "anomaly-reconstruction", "field_kinds": {}},
                            {"shape": "graph", "level": "node"}, fake)
    assert valid(out) == [] and out["reduce"] == "self_only"
    assert out["axis"] == {"kind": "group", "columns": [{"id": "all", "name": "all", "range": None}]}
    assert out["cells"]["all"]["n"] == 4 and len(fake.calls) == 1, "graph nodes travel together"
    labelled = demo([("class-0", "class", "normal", None), ("class-1", "class", "fraud", None)])
    fake = FakeInspector("graph", lambda t: "svdd_distance", lambda row, t: {"label": "fraud"})
    out = build_attribution(labelled, {"reading": "anomaly-svdd", "field_kinds": {
        "label": {"kind": "class", "of": None}}}, {"shape": "graph", "level": "graph"}, fake)
    assert [c["name"] for c in out["axis"]["columns"]] == ["normal", "fraud"]
    assert len(fake.calls) == 4, "a whole graph is one request"
    assert out["cells"]["class-1"]["correct"] == 2 and valid(out) == []


def test_a_plan_refuses_what_it_cannot_draw():
    with pytest.raises(ValueError, match="no attribution axis"):
        ab.AttributionPlan.of(demo([]), {"reading": "tokens"}, {"shape": "row"})
    with pytest.raises(ValueError, match="no item"):
        ab.AttributionPlan.of(demo([]), {"reading": "multiclass"}, {"shape": "row"})
    fake = FakeInspector("row", lambda t: "logit")
    fake.inputs = ["a"]
    doc = demo([("class-0", "class", "x", None)])
    with pytest.raises(ValueError, match="one number per network input"):
        build_attribution(doc, {"reading": "multiclass", "classes": ["x"], "field_kinds": {}},
                          {"shape": "row"}, fake)


def test_reference_scores_read_the_row_score_of_the_answered_rows():
    import pandas as pd
    doc = demo([("all", "all", None, None), ("flagged", "flagged", None, None)])
    output = {"reading": "reconstruction", "field_kinds": {
        "reconstruction_error": {"kind": "row_score", "of": None}}}

    def predict(rows, history):
        return pd.DataFrame({"reconstruction_error": [0.5 + i for i in range(len(rows))],
                             "answered": [i != 1 for i in range(len(rows))]})
    out = ab.reference_scores(doc, output, {"shape": "row"}, predict, lambda rows: rows)
    assert out == {"scores": [0.5, 2.5, 3.5], "source": "examples/demo.json (train)"}
    assert ab.reference_scores(doc, {"reading": "binary"}, {"shape": "row"}, predict, list) is None


# =============================================================================== the build

@dataclass
class Project:
    """A packed coverage-fixture bundle with its plan, story and facts, ready to build."""

    root: Path
    bundle: Path
    report: Path | None

    @property
    def showcase(self) -> Path:
        return self.bundle / "showcase"

    def args(self, *extra) -> list:
        report = ["--report", self.report, "--label-roles", self.report / "label-roles.json"] \
            if self.report else []
        plan = self.root / "plan"
        return ["--bundle", self.bundle, "--model", self.root / MODEL, "--plan",
                plan / "story-plan.json", "--story", plan / "story.json", "--facts",
                plan / "facts.json", *report, "--out", self.showcase, *extra]

    def build(self, *extra) -> tuple[int, str]:
        return run(build_showcase.main, self.args(*extra), self.root)

    def check(self, *extra) -> tuple[int, dict]:
        report = ["--report", self.report] if self.report else []
        code, printed = run(build_showcase.main, ["--check", self.showcase, *report, "--json",
                                                  *extra], self.root)
        return code, json.loads(printed)

    def doc(self) -> dict:
        return json.loads((self.showcase / "SHOWCASE.json").read_text(encoding="utf-8"))


def _docs(routes=None, pinned_showcase=False) -> dict:
    docs = cf.cell_docs(MULTICLASS)
    serve = docs["MANIFEST.json"]["serve"]
    serve["endpoints"] = sorted(set(page_template.REQUIRED_ENDPOINTS if routes is None else routes)
                                | {"GET /", "GET /v1/showcase", "GET /showcase/{path:path}"})
    if pinned_showcase:
        docs["showcase/story.json"] = {"schema": "vega/story@2"}
    return docs


def project(root: Path, linked: bool = False, docs: dict | None = None) -> Project:
    """Write a bundle, its stage briefs (or a report), and run extract, draft, author, resolve."""
    root.mkdir(parents=True, exist_ok=True)
    model, bundle = cf.write_bundle(root, docs or _docs())
    for path in (bundle / "docs" / "pack-brief.md", root / "vega-runs/demo/docs/train-brief.md"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nstage: {path.stem.split('-')[0]}\n---\n# brief\n", encoding="utf-8")
    report = root / "docs" if linked else None
    if linked:
        cf.write_report(report)
    plan = root / "plan"
    extra = ["--report", report, "--label-roles", report / "label-roles.json"] if linked else []
    steps = [(extract_facts.main, ["--model", model, "--bundle", bundle, *extra, "--out",
                                   plan / "facts.json"]),
             (draft_plan.main, ["--facts", plan / "facts.json", "--out", plan / "draft.json",
                                "--unattended"]),
             (autoauthor.main, ["--plan", plan / "draft.json", "--facts", plan / "facts.json",
                                "--out", plan / "story-plan.json"]),
             (resolve_story.main, ["--plan", plan / "story-plan.json", "--facts",
                                   plan / "facts.json", *(["--report", report] if linked else []),
                                   "--out", plan / "story.json"])]
    for main, argv in steps:
        code, printed = run(main, argv, root)
        assert code == 0, printed
    return Project(root, bundle, report)


@functools.cache
def _base(linked: bool) -> Project:
    root = Path(tempfile.mkdtemp(prefix="build-showcase-"))
    return project(root / "p", linked)


@pytest.fixture
def briefs() -> Project:
    """The brief-link project, freshly built (the previous test may have broken its showcase)."""
    p = _base(False)
    code, printed = p.build("--unattended")
    assert code == 0, printed
    return p


@pytest.fixture
def linked() -> Project:
    p = _base(True)
    code, printed = p.build()
    assert code == 0, printed
    return p


def slot(showcase: Path, bundle: Path) -> dict:
    """The service template's own slot over `showcase`: its status (the slot module run as the
    bundle renders it, beside a routes_meta that only holds the page's CSP)."""
    pkg = Path(tempfile.mkdtemp(prefix="slot-")) / "serve"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "routes_meta.py").write_text("CSP = \"default-src 'self'\"\n", encoding="utf-8")
    shutil.copyfile(SLOT_TEMPLATE, pkg / "routes_showcase.py")
    sys.path.insert(0, str(pkg.parent))
    try:
        for name in [m for m in sys.modules if m == "serve" or m.startswith("serve.")]:
            del sys.modules[name]
        from serve.routes_showcase import ShowcaseSlot
        raw = (bundle / "MANIFEST.json").read_bytes()
        return ShowcaseSlot.open(showcase, raw, json.loads(raw)["model_version"]).status()
    finally:
        sys.path.remove(str(pkg.parent))
        for name in [m for m in sys.modules if m == "serve" or m.startswith("serve.")]:
            del sys.modules[name]


def test_a_brief_link_showcase_is_built_pinned_and_verified(briefs):
    p, doc = briefs, briefs.doc()
    assert sc.showcase_schema_problems(doc) == []
    assert page_template.verify_template_copy(p.showcase / "page") == []
    assert set(doc["served"]) <= set(doc["files"]) and sc.tree_problems(p.showcase, doc["files"]) == []
    assert doc["bundle"]["manifest_digest"] == digest_file(p.bundle / "MANIFEST.json")
    assert doc["report"]["status"] == "brief-links" and doc["report"]["pdf"] is None
    assert sorted(doc["report"]["briefs"]) == ["pack", "train"]
    assert all((p.showcase / b["file"]).is_file() for b in doc["report"]["briefs"].values())
    assert CATALOG["forks"]["S1"]["marker"] in doc["markers"]
    assert doc["attribution"] is None and any("no attribution" in m for m in doc["markers"])
    assert doc["roles"]["record"] == ["plan/facts.json", "plan/story-plan.json"]
    assert "plan/facts.json" not in doc["served"] and "page/index.html" in doc["served"]
    assert (p.showcase / "page.json").is_file() and doc["page"] == {"file": "page.json"}
    assert not (p.bundle / "showcase.building").exists()
    status = slot(p.showcase, p.bundle)
    assert status["verified"] is True and status["files"] == len(doc["files"]), status
    assert p.check()[0] == 0


def test_a_re_check_with_the_same_outcome_leaves_the_pinned_record_alone(briefs):
    """release3 final3r1: the show brief pins `.work/checks.json` by digest, and the orchestrator's
    own `--check` after the brief rewrote it with a new `checked_at` — the brief went stale on a
    reading. The same outcome keeps the record byte for byte; a new outcome is written anew."""
    record = briefs.showcase / ".work" / "checks.json"
    assert briefs.check()[0] == 0
    first = record.read_bytes()
    doc = json.loads(first)
    doc["checked_at"] = "2000-01-01T00:00:00Z"   # a check older than the build is not this build's
    record.write_text(json.dumps(doc), encoding="utf-8")
    assert briefs.check()[0] == 0 and json.loads(record.read_bytes())["checked_at"] != doc["checked_at"]
    first = record.read_bytes()
    time.sleep(1.1)
    assert briefs.check()[0] == 0 and record.read_bytes() == first
    (briefs.showcase / "hook.py").write_text("print('no')\n", encoding="utf-8")
    try:
        assert briefs.check()[0] == 2 and json.loads(record.read_bytes())["status"] == "broken"
    finally:
        (briefs.showcase / "hook.py").unlink()


def test_a_linked_showcase_pins_the_report_and_its_roles(linked):
    doc = linked.doc()
    report = doc["report"]
    assert report["status"] == "linked" and report["briefs"] is None
    assert report["pdf"]["file"] == "report/report.pdf" and "report/report.pdf" in doc["served"]
    assert report["pdf"]["digest"] == doc["files"]["report/report.pdf"]
    assert report["roles"]["origin"] == "document-project"
    assert report["roles"]["bound"] == ["net.forward", "pack.slice"]
    assert set(report["roles"]["missing"]) == set(CATALOG["report_roles"]) - {"net.forward", "pack.slice"}
    assert slot(linked.showcase, linked.bundle)["verified"] is True
    assert linked.check()[0] == 0


def test_a_rebuild_keeps_the_previous_showcase_and_supersedes_it(briefs):
    before = digest_file(briefs.showcase / "SHOWCASE.json")
    code, printed = briefs.build("--unattended")
    assert code == 0, printed
    assert digest_file(briefs.bundle / "showcase.prev" / "SHOWCASE.json") == before
    assert briefs.doc()["supersedes"]["digest"] == before
    markers = json.loads((briefs.showcase / ".work" / "markers.json").read_text(encoding="utf-8"))
    assert any("swapped into place" in m for m in markers)


def test_the_pseudo_locale_build_writes_only_under_work_qps(briefs):
    outside = build_showcase.tree_state(briefs.showcase, briefs.showcase / ".work" / "qps")
    code, printed = briefs.build("--unattended", "--locale", "qps-ploc")
    assert code == 0, printed
    assert build_showcase.tree_state(briefs.showcase, briefs.showcase / ".work" / "qps") == outside
    qps = briefs.showcase / ".work" / "qps"
    assert page_template.verify_template_copy(qps / "page") == []
    assert sc.run_check(qps, None)[0] == 0 and not (qps / ".building").exists()
    assert slot(briefs.showcase, briefs.bundle)["verified"] is True, "scratch never unlists"
    # D6 §10.5: the builder wraps the story's prose ⟪…⟫ in the pseudo-locale build only
    published = json.loads((briefs.showcase / "story.json").read_text(encoding="utf-8"))
    wrapped = json.loads((qps / "story.json").read_text(encoding="utf-8"))
    assert wrapped == showcase_manifest.wrap_prose(published) and wrapped != published
    assert "⟪" not in (briefs.showcase / "story.json").read_text(encoding="utf-8")
    assert wrapped["headline"].startswith("⟪") and wrapped["values"] == published["values"]
    assert wrapped["sections"][0]["id"] == published["sections"][0]["id"]


def test_page_json_names_the_template_the_build_copies():
    installed = page_template.TEMPLATE_DIR
    assert showcase_manifest.template_identity(installed) == dict(Catalogs.load().template)
    assert showcase_manifest.template_identity(installed.parent) is None


def test_fork_s1_stops_before_anything_is_written(briefs):
    before = digest_file(briefs.showcase / "SHOWCASE.json")
    shutil.rmtree(briefs.bundle / "showcase.prev", ignore_errors=True)
    code, printed = briefs.build()
    assert code == 2 and printed.startswith("FORK: S1 report"), printed
    assert digest_file(briefs.showcase / "SHOWCASE.json") == before
    assert not (briefs.bundle / "showcase.building").exists()
    code, printed = briefs.build("--brief-links")
    assert code == 0 and "answered: brief links (--brief-links)" in printed, printed


def refused(p: Project, gate: str, *extra) -> str:
    before = digest_file(p.showcase / "SHOWCASE.json") if (p.showcase / "SHOWCASE.json").is_file() else None
    code, printed = p.build("--unattended", *extra)
    assert code == 2 and printed.startswith(f"ERROR: {gate} "), printed
    if before:
        assert digest_file(p.showcase / "SHOWCASE.json") == before, "a refused build swapped"
    return printed


def test_the_preflight_refusals(tmp_path):
    missing = project(tmp_path / "pf5", docs=_docs(routes=[r for r in page_template.REQUIRED_ENDPOINTS
                                                          if r not in INSPECTION_ROUTES]))
    assert "GET /v1/network" in refused(missing, "PF5")
    pinned = project(tmp_path / "pf9", docs=_docs(pinned_showcase=True))
    assert "two owners" in refused(pinned, "PF9")


def test_a_story_the_plan_does_not_resolve_to_is_never_pinned(briefs):
    story = briefs.root / "plan" / "story.json"
    kept = story.read_bytes()
    try:
        doc = json.loads(kept)
        doc["headline"] = "A hand-written headline."
        story.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        refused(briefs, "resolve")
    finally:
        story.write_bytes(kept)


def _template_copy(tmp: Path) -> Path:
    copy_dir = tmp / "page"
    shutil.copytree(page_template.TEMPLATE_DIR, copy_dir)
    return copy_dir


def test_d5_30_a_leftover_token_in_the_template_is_refused(briefs, tmp_path):
    template = _template_copy(tmp_path)
    index = template / "index.html"
    index.write_text(index.read_text(encoding="utf-8").replace("<title>", "<title>{{TITLE_HTML}}", 1),
                     encoding="utf-8")
    assert "TITLE_HTML" in refused(briefs, "PG7", "--template", template)


def test_d5_31_a_class_name_in_the_template_is_caught_by_the_literal_gate(tmp_path):
    template = _template_copy(tmp_path)
    (template / "js" / "views" / "start.js").write_text(
        (template / "js" / "views" / "start.js").read_text(encoding="utf-8")
        + '\nexport const NEUTRAL = "BENIGN";\n', encoding="utf-8")
    done = subprocess.run([sys.executable, str(LINT), str(template / "js" / "views" / "start.js")],
                          capture_output=True, text=True)
    assert done.returncode == 1 and "BENIGN" in done.stdout, done.stdout + done.stderr


def test_the_staged_copy_and_page_are_held_to_bd1_and_bd2(briefs, monkeypatch):
    real = shutil.copyfile

    def corrupt(src, dst, *a, **k):
        out = real(src, dst, *a, **k)
        if Path(dst).name == "index.html":
            Path(dst).write_bytes(Path(dst).read_bytes() + b"<!-- -->")
        return out
    monkeypatch.setattr(build_showcase.shutil, "copyfile", corrupt)
    assert "index.html" in refused(briefs, "BD1")
    monkeypatch.setattr(build_showcase.shutil, "copyfile", real)
    other = {"schema": "vega/network-view@2", "task": {"kind": "regression", "base": None,
                                                       "link": "identity"},
             "sample": {"shape": "row", "level": None}, "family": {"name": "mlp-plain"}}
    monkeypatch.setattr(build_showcase, "bundle_run", lambda b, s: ({"network": other}, "OK"))
    assert "another network" in refused(briefs, "BD2")


def test_d5_25_code_or_a_symlink_in_the_staged_showcase_is_refused(briefs):
    work = briefs.bundle / "showcase.building" / ".work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "hook.py").write_text("print('no')\n", encoding="utf-8")
    try:
        assert "hook.py" in refused(briefs, "BD3")
    finally:
        (work / "hook.py").unlink()
    (work / "link.json").symlink_to(briefs.bundle / "MANIFEST.json")
    try:
        assert "symlink" in refused(briefs, "BD4")
    finally:
        (work / "link.json").unlink()


def test_the_build_never_writes_in_the_bundle_bd8(briefs, monkeypatch):
    def writes(bundle, scratch):
        (Path(bundle) / "stray.txt").write_text("x", encoding="utf-8")
        return None, "exit 2, no output"
    monkeypatch.setattr(build_showcase, "bundle_run", writes)
    try:
        assert "stray.txt" in refused(briefs, "BD8")
    finally:
        (briefs.bundle / "stray.txt").unlink()


# =============================================================================== the check

def _edit_manifest(p: Project, edit) -> bytes:
    path = p.showcase / "SHOWCASE.json"
    kept = path.read_bytes()
    doc = json.loads(kept)
    edit(doc)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return kept


def _rules(doc: dict) -> set[str]:
    return {p["rule"] for p in doc["problems"]}


def test_d5_25_and_32_code_dropped_in_or_a_story_edited_is_broken(briefs):
    (briefs.showcase / "hook.py").write_text("print('no')\n", encoding="utf-8")
    code, doc = briefs.check()
    assert code == 2 and doc["status"] == "broken" and "CK3" in _rules(doc)
    (briefs.showcase / "hook.py").unlink()
    story = briefs.showcase / "story.json"
    edited = json.loads(story.read_text(encoding="utf-8"))
    edited["values"].pop(next(k for k in edited["values"] if k != "primary"))
    story.write_text(json.dumps(edited), encoding="utf-8")
    code, doc = briefs.check()
    assert code == 2 and _rules(doc) == {"CK2"}
    assert json.loads((briefs.showcase / ".work" / "checks.json").read_text())["status"] == "broken"
    status = slot(briefs.showcase, briefs.bundle)
    assert status["verified"] is False and "story.json" in status["reason"]


def test_d5_26_a_served_path_not_in_files_is_broken_and_refused(briefs):
    _edit_manifest(briefs, lambda d: d["served"].append("extra.json"))
    code, doc = briefs.check()
    assert code == 2 and "CK4" in _rules(doc)
    assert slot(briefs.showcase, briefs.bundle)["verified"] is False


def test_d5_27_a_manifest_digest_off_by_one_hex_is_broken_and_refused(briefs):
    def off(doc):
        digest = doc["bundle"]["manifest_digest"]
        doc["bundle"]["manifest_digest"] = digest[:-1] + ("0" if digest[-1] != "0" else "1")
    _edit_manifest(briefs, off)
    code, doc = briefs.check()
    assert code == 2 and _rules(doc) == {"CK5"}
    status = slot(briefs.showcase, briefs.bundle)
    assert status["verified"] is False and "another bundle" in status["reason"]


def test_a_bundle_edited_in_place_is_broken_a_re_emitted_one_stale(briefs):
    path = briefs.bundle / "MANIFEST.json"
    kept = path.read_bytes()
    try:
        manifest = json.loads(kept)
        path.write_text(json.dumps({**manifest, "note": "edited"}, indent=1), encoding="utf-8")
        code, doc = briefs.check()
        assert code == 2 and _rules(doc) == {"CK5"}
        reemitted = {**manifest, "reemits": [{"at": "2026-09-30T00:00:00Z", "reason": "templates",
                                              "previous_manifest_digest": digest_bytes(kept),
                                              "changed": [], "added": [], "removed": []}]}
        path.write_text(json.dumps(reemitted, indent=1), encoding="utf-8")
        code, doc = briefs.check()
        assert code == 3 and doc["status"] == "stale" and _rules(doc) == {"CK6"}
        status = slot(briefs.showcase, briefs.bundle)
        assert status["verified"] is False and status["reason"] == CATALOG["slot"]["stale_reason"]
    finally:
        path.write_bytes(kept)


def test_a_changed_brief_or_a_report_that_appears_is_stale(briefs):
    brief = briefs.bundle / "docs" / "pack-brief.md"
    kept = brief.read_bytes()
    brief.write_bytes(kept + b"\nedited\n")
    try:
        code, doc = briefs.check()
        assert code == 3 and _rules(doc) == {"CK8"}
    finally:
        brief.write_bytes(kept)
    docs = briefs.root / "docs"
    cf.write_report(docs)
    try:
        code, doc = briefs.check()
        assert code == 3 and _rules(doc) == {"CK9"}
    finally:
        shutil.rmtree(docs)
    assert briefs.check()[0] == 0


def test_d5_28_the_report_pdf_changed_is_stale(linked):
    pdf = linked.report / "main.pdf"
    kept = pdf.read_bytes()
    pdf.write_bytes(kept + b" ")
    try:
        code, doc = linked.check()
        assert code == 3 and _rules(doc) == {"CK7"}
    finally:
        pdf.write_bytes(kept)
    assert linked.check()[0] == 0


def test_d5_29_a_role_map_older_than_its_aux_is_refused_at_extraction():
    docs = cf.cell_docs(MULTICLASS)

    def stale_map(bundle):
        cf.write_report(bundle.parent.parent / "docs", digest="sha256:" + "0" * 64)
    printed = cf.refused(docs, "PF6", after=stale_map)
    assert "recompile /document-project" in printed or "older" in printed, printed


def test_an_updated_template_warns_and_passes(briefs, tmp_path):
    template = _template_copy(tmp_path)
    (template / "page.css").write_text((template / "page.css").read_text(encoding="utf-8") + "\n",
                                       encoding="utf-8")
    if (template / "TEMPLATE.json").is_file():       # an updated template carries its new identity (W15)
        assert page_template.main(["--template", str(template), "--write"]) == 0
    code, doc = briefs.check("--template", template)
    assert code == 0 and doc["status"] == "pass"
    assert [w["rule"] for w in doc["warnings"]] == ["CK10"]
    assert doc["warnings"][0]["message"] == "template updated since build"


# =============================================================================== the lanes

LANES = os.environ.get("VEGA_SHOW_LANES")
L01_CAMPAIGN = "vega-runs/2026-09-05T11-03Z"        # the acceptance lane's campaign (pack acceptance)
PYTHON_ENV = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
              "HOME": os.environ.get("HOME", "/tmp"), "LANG": "C.UTF-8",
              "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
SERVE_PROBE = """
import json, sys, time
sys.path.insert(0, '.')
from serve.app import create_app
from fastapi.testclient import TestClient
out = {}
with TestClient(create_app('.')) as c:
    out['showcase'] = c.get('/v1/showcase').json()
    for _ in range(300):
        health = c.get('/health').json()
        if health.get('ready'):
            break
        time.sleep(0.2)
    out['ready'] = health.get('ready')
    page = c.get('/')
    out['page'] = [page.status_code, page.headers.get('etag'), page.headers.get('content-type')]
    out['story'] = c.get('/v1/story').status_code
    out['page_json'] = c.get('/showcase/page.json').status_code
    out['report'] = c.get('/report.pdf').status_code
print(json.dumps(out))
"""


@dataclass
class LaneWork:
    work: Path
    bundle: Path
    model: Path


def _lane(lane: str, tmp: Path) -> LaneWork:
    source = Path(LANES) / "lanes" / lane / "work"
    if not source.is_dir():
        pytest.skip(f"{source} is not there: run the lane through pack first")
    work = tmp / lane
    shutil.copytree(source, work, symlinks=True)
    bundle = next(p for p in sorted((work / "bundle").iterdir()) if (p / "MANIFEST.json").is_file())
    model = next(iter(sorted(work.glob("*.pack.html"))))
    return LaneWork(work, bundle, model)


def _prepare(lw: LaneWork, report: Path | None) -> None:
    """extract, draft, author, resolve into the copy's `show-plan/`."""
    plan = lw.work / "show-plan"
    extra = ["--report", report, "--label-roles", report / "label-roles.json"] if report else []
    steps = [(extract_facts.main, ["--model", lw.model, "--bundle", lw.bundle, *extra, "--out",
                                   plan / "facts.json"]),
             (draft_plan.main, ["--facts", plan / "facts.json", "--out", plan / "draft.json",
                                "--unattended"]),
             (autoauthor.main, ["--plan", plan / "draft.json", "--facts", plan / "facts.json",
                                "--out", plan / "story-plan.json"]),
             (resolve_story.main, ["--plan", plan / "story-plan.json", "--facts",
                                   plan / "facts.json", *(["--report", report] if report else []),
                                   "--out", plan / "story.json"])]
    for main, argv in steps:
        code, printed = run(main, argv, lw.work)
        assert code == 0, printed


def _build(lw: LaneWork, report: Path | None, *build) -> tuple[int, str]:
    plan = lw.work / "show-plan"
    extra = ["--report", report, "--label-roles", report / "label-roles.json"] if report else []
    return run(build_showcase.main, ["--bundle", lw.bundle, "--model", lw.model, "--plan",
                                     plan / "story-plan.json", "--story", plan / "story.json",
                                     "--facts", plan / "facts.json", *extra, "--out",
                                     lw.bundle / "showcase", *build], lw.work)


def _declare_routes(lw: LaneWork) -> None:
    """The two inspection routes I08/I09 add to MANIFEST.serve.endpoints, declared on this copy
    of the lane (its manifest and the model page's bundle layer, which must stay equal)."""
    path = lw.bundle / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["serve"]["endpoints"] = sorted(set(manifest["serve"]["endpoints"]) | set(INSPECTION_ROUTES))
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    html = lw.model.read_text(encoding="utf-8")
    found = re.search(r'(<script id="layer-bundle-data" type="application/json">)(.*?)(</script>)',
                      html, re.S)
    layer = json.loads(found.group(2))
    layer["manifest"] = manifest
    data = json.dumps(layer, ensure_ascii=False).replace("</", "<\\/")
    lw.model.write_text(html[:found.start(2)] + data + html[found.end(2):], encoding="utf-8")


def _built(lw: LaneWork, report: Path | None, *build) -> dict:
    """The lane's showcase built: refused by PF5 while the bundle serves no inspection route
    (then declared on the copy), then built; the bundle's MANIFEST digest unchanged."""
    served = json.loads((lw.bundle / "MANIFEST.json").read_text())["serve"]["endpoints"]
    if not set(INSPECTION_ROUTES) <= set(served):
        _prepare(lw, report)
        code, printed = _build(lw, report, *build)
        assert code == 2 and printed.startswith("ERROR: PF5 "), printed
        _declare_routes(lw)
    before = digest_file(lw.bundle / "MANIFEST.json")
    _prepare(lw, report)
    code, printed = _build(lw, report, *build)
    assert code == 0, printed
    assert digest_file(lw.bundle / "MANIFEST.json") == before
    doc = json.loads((lw.bundle / "showcase" / "SHOWCASE.json").read_text(encoding="utf-8"))
    assert sc.showcase_schema_problems(doc) == []
    assert page_template.verify_template_copy(lw.bundle / "showcase" / "page") == []
    assert set(doc["served"]) <= set(doc["files"])
    assert sc.tree_problems(lw.bundle / "showcase", doc["files"]) == []
    return doc


def _served(bundle: Path) -> dict:
    done = subprocess.run([sys.executable, "-B", "-c", SERVE_PROBE], cwd=bundle, env=PYTHON_ENV,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _check(lw: LaneWork, report: Path | None = None) -> int:
    argv = ["--check", lw.bundle / "showcase", *(["--report", report] if report else [])]
    return run(build_showcase.main, argv, lw.work)[0]


def _attribution_is_the_classes(lw: LaneWork, doc: dict) -> None:
    attribution = json.loads((lw.bundle / "showcase" / doc["attribution"]["file"]).read_text())
    assert valid(attribution) == []
    classes = json.loads((lw.bundle / "MANIFEST.json").read_text())["output"]["classes"]
    assert sorted(c["name"] for c in attribution["axis"]["columns"]) == sorted(classes)
    assert all(cell["n"] > 0 for cell in attribution["cells"].values())


@pytest.mark.skipif(not LANES, reason="VEGA_SHOW_LANES names no matrix directory with L01/L02")
def test_lane_l01_brief_links_served_verified_then_stale(tmp_path):
    lw = _lane("L01", tmp_path)
    for path in (lw.bundle / "docs" / "pack-brief.md",
                 lw.work / L01_CAMPAIGN / "docs" / "train-brief.md"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\nstage: {path.stem.split('-')[0]}\n---\n# brief\n", encoding="utf-8")
    doc = _built(lw, None, "--unattended")
    assert doc["report"]["status"] == "brief-links" and doc["report"]["briefs"]
    _attribution_is_the_classes(lw, doc)
    served = _served(lw.bundle)
    assert served["showcase"]["verified"] is True and served["ready"] is True, served
    assert served["page"][0] == 200 and served["page"][1] == f'"{doc["files"]["page/index.html"]}"'
    assert served["story"] == 200 and served["page_json"] == 200 and served["report"] == 404
    assert _check(lw) == 0
    # a report that appears makes the showcase stale
    docs = lw.work / "docs"
    cf.write_report(docs)
    assert _check(lw) == 3
    shutil.rmtree(docs)
    # the manifest digest off by one hex: broken, and the slot refuses (the model stays ready)
    path = lw.bundle / "showcase" / "SHOWCASE.json"
    kept = path.read_bytes()
    edited = json.loads(kept)
    digest = edited["bundle"]["manifest_digest"]
    edited["bundle"]["manifest_digest"] = digest[:-1] + ("0" if digest[-1] != "0" else "1")
    path.write_text(json.dumps(edited, indent=1), encoding="utf-8")
    assert _check(lw) == 2
    served = _served(lw.bundle)
    assert served["showcase"]["verified"] is False and served["ready"] is True
    path.write_bytes(kept)
    # a real re-emit of the serving adapters: stale, and the slot refuses with the re-emit reason
    pack = SKILLS / "vega-pack" / "scripts" / "pack.py"
    done = subprocess.run([sys.executable, str(pack), "--campaign", L01_CAMPAIGN, "--shaped", "shaped",
                           "--bundle-dir", "bundle", "--pipeline",
                           "blueprint-runs/2026-09-05T10-12Z/pipeline.json", "--model",
                           "orders.domain.analysis.shaped.vega.train.eval.html", "--ship-anyway",
                           "the calibrated score is what the downstream dispatcher needs", "--serve",
                           "--monitor-plan", "monitor-plan.json",       # the plan L01 packed with
                           "--re-emit-serving", "--reason", "showcase staleness"],
                          cwd=lw.work, capture_output=True, text=True, timeout=1200,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    assert json.loads((lw.bundle / "MANIFEST.json").read_text())["reemits"]
    assert _check(lw) == 3
    served = _served(lw.bundle)
    assert served["showcase"]["reason"] == CATALOG["slot"]["stale_reason"], served


@pytest.mark.skipif(not LANES, reason="VEGA_SHOW_LANES names no matrix directory with L01/L02")
def test_lane_l02_stub_report_served_verified_then_stale_and_broken(tmp_path):
    lw = _lane("L02", tmp_path)
    docs = lw.work / "docs"
    done = subprocess.run([sys.executable, "-m", "lanekit.report_stub", "--docs", str(docs),
                           "--lane", "L02"], cwd=SKILLS / "vega" / "scripts" / "tests",
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    doc = _built(lw, docs)
    assert doc["report"]["status"] == "linked" and doc["report"]["roles"]["missing"] == []
    _attribution_is_the_classes(lw, doc)
    served = _served(lw.bundle)
    assert served["showcase"]["verified"] is True and served["report"] == 200, served
    assert _check(lw, docs) == 0
    # the qps build lands under .work/qps only, and checks
    code, printed = _build(lw, docs, "--locale", "qps-ploc")
    assert code == 0, printed
    assert sc.run_check(lw.bundle / "showcase" / ".work" / "qps", docs)[0] == 0
    # the PDF touched: stale
    pdf = docs / "main.pdf"
    kept = pdf.read_bytes()
    pdf.write_bytes(kept + b" ")
    assert _check(lw, docs) == 3
    pdf.write_bytes(kept)
    # story.json altered: broken, the slot refuses, the model stays ready
    story = lw.bundle / "showcase" / "story.json"
    story.write_bytes(story.read_bytes().replace(b'"headline"', b'"headline" ', 1))
    assert _check(lw, docs) == 2
    served = _served(lw.bundle)
    assert served["showcase"]["verified"] is False and served["ready"] is True
    assert served["page"][0] == 200 and served["page"][1] != f'"{doc["files"]["page/index.html"]}"'


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
