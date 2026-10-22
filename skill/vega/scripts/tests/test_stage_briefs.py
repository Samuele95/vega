#!/usr/bin/env python3
"""
test_stage_briefs.py — a lane without a report leaves the stage briefs its show links to
(lanekit.stage_briefs; BUILD-PLAN T09, the carried need 'no lane writes stage briefs': a
no-report lane's show stage was refused with PF6).

    python3 -m pytest -q test_stage_briefs.py

The briefs lie where vega-show's facts look for them (the run directories the bundle's records
name, relative or absolute), hold brief-contract.md (check_brief.py passes on them), and nothing
is written inside the bundle (the pack brief is left out: the bundle is pinned after pack).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_brief  # noqa: E402
from lanekit import stage_briefs  # noqa: E402


def lane(tmp: Path) -> tuple[Path, Path]:
    work = tmp / "work"
    bundle = work / "bundle" / "demo-target-R0"
    runs = {"shaped": ("manifest.json", "recipe.json"), "blueprint-runs/T1": ("pipeline.json",),
            "vega-runs/T2": ("architecture-layer.json", "smoke.json", "training-layer.json",
                             "experiments.json", "evaluation-layer.json", "test_seal.json")}
    for d, names in runs.items():
        (work / d).mkdir(parents=True)
        for n in names:
            (work / d / n).write_text(json.dumps({"file": n}), encoding="utf-8")
    for suffix in (".domain.analysis.html", ".domain.analysis.shaped.html", ".domain.analysis.shaped.vega.html",
                   ".domain.analysis.shaped.vega.train.html", ".domain.analysis.shaped.vega.train.eval.html"):
        (work / f"demo{suffix}").write_text("<html></html>", encoding="utf-8")
    (bundle / "records").mkdir(parents=True)
    records = {"shape.json": {"manifest": {"outputs": {"dir": str(work / "shaped")}}},
               "architecture.json": {"netspec": {"outputs": {"run_dir": "vega-runs/T2/"},
                                                 "input": {"pipeline": str(work / "blueprint-runs/T1/pipeline.json")}}},
               "training.json": {"campaign": {"run_dir": str(work / "vega-runs/T2")}}}
    for name, doc in records.items():
        (bundle / "records" / name).write_text(json.dumps(doc), encoding="utf-8")
    return work, bundle


def test_the_briefs_hold_the_contract_where_the_facts_look(tmp_path):
    work, bundle = lane(tmp_path)
    before = sorted(p.relative_to(bundle) for p in bundle.rglob("*"))
    written = stage_briefs.write(work, bundle, "L99")
    assert [p.relative_to(work).as_posix() for p in written] == [
        "shaped/docs/shape-brief.md", "blueprint-runs/T1/docs/blueprint-brief.md", "vega-runs/T2/docs/net-brief.md",
        "vega-runs/T2/docs/train-brief.md", "vega-runs/T2/docs/eval-brief.md"]
    assert check_brief.main([*map(str, written), "--root", str(work)]) == 0
    assert sorted(p.relative_to(bundle) for p in bundle.rglob("*")) == before, "nothing written inside the bundle"
    net = (work / "vega-runs/T2/docs/net-brief.md").read_text(encoding="utf-8")
    assert "output: demo.domain.analysis.shaped.vega.html" in net and "layer: architecture" in net


def test_a_tampered_record_fails_the_brief_check(tmp_path):
    work, bundle = lane(tmp_path)
    written = stage_briefs.write(work, bundle, "L99")
    (work / "vega-runs/T2/smoke.json").write_text("{}", encoding="utf-8")
    assert check_brief.main([*map(str, written), "--root", str(work)]) == 1


def test_no_run_directory_no_brief(tmp_path):
    work, bundle = lane(tmp_path)
    for name in ("shape.json", "architecture.json", "training.json"):
        (bundle / "records" / name).write_text("{}", encoding="utf-8")
    assert stage_briefs.write(work, bundle, "L99") == []
