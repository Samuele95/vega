#!/usr/bin/env python3
"""
test_rejected_derivation.py — a derivation the geometry layer rejected derives nothing (F44;
dataset-forge report-contract §2, `shape-contract.md` §3, `step-catalog.md` phase 2).

    python3 -m pytest test_rejected_derivation.py      (or: python3 test_rejected_derivation.py)

The release4 rehearsal's geometry layer listed `slope_deg = slope(elevation field)` and
`aspect_deg = aspect(elevation field)` as derivations with `kept: false` and the semantic channel
`refuted`: a rejected derivation stays in the layer with its triple and the consequence block
that says why, and only a kept rule derives a column. The shaper read every listed derivation
as active: the recipe checker refused the analysis's log on `slope_deg` ("a transform on one
member alone breaks the relation"), `drop_derived` would have dropped two basis members, the
verifier failed "active derivation head(s) survive without a keep", and the session had to add
a `keep` the executor then called "redundant with their parents by construction". What is
proved, through the real scripts, on the rehearsal's own two derivation entries:

  kept        `derivation_kept` is dataset-forge's predicate: semantic `confirmed` and not
              `kept: false`;
  rejected    a transform on a rejected head alone passes the checker; `drop_derived` keeps
              both rejected heads with no `keep` and no WARN; the verifier passes structural and
              records each rejected rule as `empirical: not applicable` with no WARN;
  control     a kept derivation (`r = a / b`, confirmed) still binds: `drop_derived` drops `r`,
              and without the drop a transform on `a` alone is refused naming the relation.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SHAPE, VERIFY = HERE.parent / "shape.py", HERE.parent / "verify_shape.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shape = _load(SHAPE, "shaper_shape_rejected_under_test")

# the release4 round-2 rehearsal's geometry layer, its two derivation entries as published
# (data/landcover.domain.html, layer-geometry-data; consequence prose shortened)
REJECTED = [
    {"column": "slope_deg", "rule_id": "swrl-slope", "layer": "swrl", "kept": False,
     "formula": "slope_deg = slope(elevation field around the parcel)", "body": ["elevation_m"],
     "provenance": {"semantic": {"status": "refuted", "note": "slope is a derivative of the terrain "
                                 "surface: one parcel's own elevation fixes none of it"},
                    "symbolic": {"status": "untested", "note": "the rule was not kept"},
                    "empirical": {"status": "refuted", "evidence": "no derivation and no FD found"}},
     "consequences": {"decision": "slope_deg stays in the basis; no rule derives it from elevation_m"}},
    {"column": "aspect_deg", "rule_id": "swrl-aspect", "layer": "swrl", "kept": False,
     "formula": "aspect_deg = aspect(elevation field around the parcel)", "body": ["elevation_m"],
     "provenance": {"semantic": {"status": "refuted", "note": "aspect is a derivative of the terrain "
                                 "surface"},
                    "symbolic": {"status": "untested", "note": "the rule was not kept"},
                    "empirical": {"status": "refuted", "evidence": "no derivation and no FD found"}},
     "consequences": {"decision": "aspect_deg stays in the basis; no rule derives it from elevation_m"}},
]
KEPT = {"column": "r", "rule_id": "swrl-r", "layer": "swrl", "formula": "r = a / b", "body": ["a", "b"],
        "provenance": {"semantic": {"status": "confirmed"}, "symbolic": {"status": "confirmed"},
                       "empirical": {"status": "confirmed"}}}
BASIS = ["elevation_m", "slope_deg", "aspect_deg", "a", "b", "y"]


def _model(tmp: Path, derivations: list) -> Path:
    geo = {"schema": "dataset-forge/geometry@1", "basis": {"members": BASIS},
           "derivations": derivations, "cycles": [],
           "partitions": {"chosen": "y", "candidates": [{"label": "y", "dropped_for_leakage": []}]}}
    html = tmp / "toy.domain.html"
    html.write_text('<html><body><script id="layer-geometry-data" type="application/json">'
                    + json.dumps(geo) + "</script></body></html>\n", encoding="utf-8")
    return html


def _data(tmp: Path) -> Path:
    rng = np.random.default_rng(11)
    n = 300
    df = pd.DataFrame({"elevation_m": rng.uniform(50, 2000, n).round(1),
                       "slope_deg": rng.uniform(0.5, 40, n).round(2),
                       "aspect_deg": rng.uniform(0, 360, n).round(1),
                       "a": rng.uniform(1, 10, n).round(3), "b": rng.uniform(1, 10, n).round(3)})
    df["r"] = df["a"] / df["b"]
    df["y"] = rng.choice(["p", "q", "s"], n)
    csv = tmp / "toy.csv"
    df.to_csv(csv, index=False)
    return csv


def _step(sid, op, columns, params, source="user:test"):
    return {"id": sid, "op": op, "columns": columns, "params": params, "source": source,
            "rationale": f"{op} for the test",
            "consequences": {"rows": "unchanged", "columns": "unchanged", "downstream": "none"}}


def _recipe(csv: Path, out: Path, steps: list) -> dict:
    return {"schema": "dataset-shaper/recipe@1", "input": {"path": str(csv)}, "seed": 1,
            "goal": "train-ready", "target": "y",
            "outputs": {"dir": str(out), "format": "parquet", "files": []}, "steps": steps}


def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def _write(tmp: Path, steps: list, name="recipe.json"):
    csv, out = _data(tmp), tmp / "shaped"
    rp = tmp / name
    rp.write_text(json.dumps(_recipe(csv, out, steps), indent=1), encoding="utf-8")
    return rp, out


def test_kept_is_dataset_forges_predicate():
    assert shape.derivation_kept(KEPT)
    assert not any(shape.derivation_kept(d) for d in REJECTED)
    set_aside = dict(KEPT, kept=False)
    unconfirmed = copy.deepcopy(KEPT)
    unconfirmed["provenance"]["semantic"]["status"] = "untested"
    assert not shape.derivation_kept(set_aside) and not shape.derivation_kept(unconfirmed)
    assert not shape.derivation_kept(None) and not shape.derivation_kept({"column": "x"})
    geo = {"derivations": [*REJECTED, KEPT]}
    assert [d["column"] for d in shape.kept_derivations(geo)] == ["r"]


def test_a_rejected_head_is_a_basis_member_through_the_shaper_and_the_verifier(tmp_path):
    model = _model(tmp_path, [*REJECTED, KEPT])
    steps = [_step("S1", "select_partition", ["y"], {"label": "y", "task": "multiclass classification"},
                   source="geometry:partitions/y"),
             _step("S2", "drop_derived", [], {}, source="geometry:derivations"),
             _step("S3", "split", [], {"kind": "stratified", "by": "y", "test": 0.2, "valid": 0.2}),
             _step("S4", "transform", ["slope_deg"], {"kind": "log1p"})]
    rp, out = _write(tmp_path, steps)
    rc, log = _run(SHAPE, "--recipe", rp, "--model", model, "--check-only")
    assert rc == 0, log[-2000:]
    assert "but not the whole derivation" not in log, log[-2000:]
    rc, log = _run(SHAPE, "--recipe", rp, "--model", model, "--out-dir", out)
    assert rc == 0, log[-2000:]
    assert "kept by request" not in log, log[-2000:]          # no keep, and no false redundancy WARN
    train = pd.read_parquet(next(out.glob("*train.parquet")))
    assert {"slope_deg", "aspect_deg", "elevation_m", "a", "b"} <= set(train.columns), list(train.columns)
    assert "r" not in train.columns                           # the kept rule's head leaves
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert not man.get("kept_derived"), man.get("kept_derived")
    rc, log = _run(VERIFY, "--recipe", rp, "--out-dir", out, "--model", model,
                   "--json", tmp_path / "verification.json")
    assert rc == 0, log[-3000:]
    assert "survive without a keep" not in log and "could not be evaluated" not in log, log[-3000:]
    report = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    rows = {r["rule_id"]: r for r in report["semantic"]}
    for rid in ("swrl-slope", "swrl-aspect"):
        assert rows[rid]["empirical"] == "not applicable", rows[rid]
        assert "rejected" in rows[rid]["why"], rows[rid]


def test_a_kept_derivation_still_binds_a_transform(tmp_path):
    model = _model(tmp_path, [*REJECTED, KEPT])
    steps = [_step("S1", "select_partition", ["y"], {"label": "y", "task": "multiclass classification"},
                   source="geometry:partitions/y"),
             _step("S3", "split", [], {"kind": "stratified", "by": "y", "test": 0.2, "valid": 0.2}),
             _step("S4", "transform", ["a"], {"kind": "log1p"})]
    rp, _ = _write(tmp_path, steps)
    rc, log = _run(SHAPE, "--recipe", rp, "--model", model, "--check-only")
    assert rc != 0 and "not the whole derivation r = f(a, b)" in log, log[-2000:]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
