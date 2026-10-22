#!/usr/bin/env python3
"""
test_size_budgets.py — the lane budgets come from measured runs (D6 §16.5; BUILD-PLAN T12; A44(a),
A47(6)).

    python3 -m pytest -q test_size_budgets.py

1. A lane's budget is 1.5 × its largest measured wall time, rounded up to 30 s; only passing lanes
   count, and a `--from` run (stamped stages) measures its stages but not the lane's wall time.
2. A stage share grows until every lane's stage limit (stage_timeout_factor × budget × share)
   holds 1.5 × the stage's measured time, per share group (L01's acceptance shares apart); a share
   never shrinks; an unmeasured lane keeps its budget.
3. `--write` rewrites lanes.json in its own form, and the tree's lanes.json holds what its own
   recorded measurement sizes (runs/final/T12, reproduced here as the stage table it was read from).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import size_budgets as sb  # noqa: E402
from lanekit import registry  # noqa: E402


def timing(*rows) -> dict:
    return {"schema": "vega/matrix-timing@1", "lanes": [
        {"lane": lane, "verdict": verdict, "wall_s": wall,
         "stages": [{"id": s, "status": st, "wall_s": secs} for s, st, secs in stages]}
        for lane, verdict, wall, stages in rows]}


DOC = {"defaults": {"stage_timeout_factor": 2, "stage_shares": {"pack": 0.1, "bundle_api": 0.25},
                    "acceptance_shares": {"bundle_api": 0.05}, "golden_shares": {"golden": 0.55}},
       "lanes": [{"id": "L01", "driver": "acceptance", "budget_s": 450}, {"id": "L13", "budget_s": 150},
                 {"id": "L14", "budget_s": 150}, {"id": "G00", "driver": "golden", "budget_s": 720}]}


def test_budgets_are_one_and_a_half_times_the_largest_passing_wall():
    seen = sb.measured([timing(("L13", "pass", 120.0, [("bundle_api", "pass", 75.0)]),
                               ("L14", "fail", 900.0, [("pack", "timeout", 30.0)])),
                        timing(("L13", "pass", 341.6, [("bundle_api", "pass", 196.0)]),
                               ("L14", "pass", 10.0, [("pack", "stamped", 0.0), ("bundle_api", "pass", 9.0)]))])
    budgets, shares, notes = sb.size(DOC, seen)
    assert budgets["L13"] == 540 == math.ceil(1.5 * 341.6 / 30) * 30
    assert budgets["L14"] == 150 and budgets["G00"] == 720, "unmeasured: a failed and a --from run only"
    assert seen["L14"]["stages"] == {"bundle_api": 9.0}
    assert any(n.startswith("L14: not measured") for n in notes)


def test_a_share_holds_the_slowest_stage_and_never_shrinks():
    seen = sb.measured([timing(("L13", "pass", 341.6, [("bundle_api", "pass", 196.0), ("pack", "pass", 1.0)]),
                               ("L01", "pass", 456.4, [("bundle_api", "pass", 47.0)]))])
    budgets, shares, _ = sb.size(DOC, seen)
    need = 1.5 * 196.0 / (2 * budgets["L13"])
    assert shares["stage_shares"]["bundle_api"] == math.ceil(100 * need) / 100 == 0.28
    assert 2 * budgets["L13"] * shares["stage_shares"]["bundle_api"] >= 1.5 * 196.0
    assert shares["stage_shares"]["pack"] == 0.1, "a fast stage keeps its share"
    assert shares["acceptance_shares"]["bundle_api"] == 0.06, "L01 sizes its own group"


def test_write_keeps_the_files_form(tmp_path):
    lanes = tmp_path / "lanes.json"
    lanes.write_text(json.dumps(DOC, indent=2) + "\n", encoding="utf-8")
    t = tmp_path / "timing.json"
    t.write_text(json.dumps(timing(("L13", "pass", 341.6, [("bundle_api", "pass", 196.0)]))), encoding="utf-8")
    assert sb.main(["--timing", str(t), "--lanes-file", str(lanes), "--write"]) == 0
    doc = json.loads(lanes.read_text(encoding="utf-8"))
    assert doc["lanes"][1]["budget_s"] == 540 and doc["defaults"]["stage_shares"]["bundle_api"] == 0.28
    assert lanes.read_text(encoding="utf-8") == json.dumps(doc, indent=2) + "\n"


# the release run that sized the tree's budgets (runs/final/T12/timings/m1-timing.json): lane, wall,
# and the stages whose time decides a share
MEASURED = {"L01": 456.4, "L02": 268.2, "L03": 269.7, "L04": 149.0, "L05": 177.0, "L06": 170.5, "L07": 167.7,
            "L08": 182.1, "L09": 175.7, "L10": 292.9, "L11": 200.6, "L12": 178.2, "L13": 341.6, "L14": 162.3,
            "L15": 147.3, "L16": 164.5, "L17": 147.0, "L18": 231.9, "L19": 172.4, "L20": 208.3, "L21": 167.1,
            "L22": 159.3, "L23": 60.8}
SLOWEST = {"bundle_api": ("L13", 196.0), "show": ("L11", 42.0), "fixture": ("L23", 12.0), "page": ("L05", 27.0)}


def test_the_trees_budgets_hold_the_recorded_measurement():
    doc = registry.load_document()
    lanes = {l["id"]: l for l in doc["lanes"]}
    for lane_id, wall in MEASURED.items():
        assert lanes[lane_id]["budget_s"] >= 1.5 * wall, lane_id
    shares, factor = doc["defaults"]["stage_shares"], doc["defaults"]["stage_timeout_factor"]
    for stage, (lane_id, secs) in SLOWEST.items():
        assert factor * lanes[lane_id]["budget_s"] * shares[stage] >= 1.5 * secs, (stage, lane_id)
