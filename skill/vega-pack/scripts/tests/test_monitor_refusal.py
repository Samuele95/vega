#!/usr/bin/env python3
"""
test_monitor_refusal.py — a Monitor the blueprint says /vega-pack fills is never left unfilled
(bundle-contract §1, §7.5; `monitor_pack.refuse_unfilled`).

    python3 -m pytest -q vega-pack/scripts/tests/test_monitor_refusal.py

The refusal is mechanical and comes before anything is written: a blueprint whose Monitor port
says `"filled_by": "/vega-pack"` and no `--monitor-plan` is exit 2 naming the port and the
concerns the plan must answer; a plan given, a blueprint that does not name this stage on the
port, or no `--pipeline` at all pass through (the last two keep the forge's pending stub, as
the map records).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import monitor_pack  # noqa: E402
from pack_context import PackExit  # noqa: E402


def pipeline(tmp: Path, filled_by="/vega-pack", monitoring=(("b04 drift", "[analysis:F4]"),)) -> Path:
    port = {"name": "DriftWatch", "role": "Monitor", "responsibility": "watch the drift"}
    if filled_by:
        port["filled_by"] = filled_by
    doc = {"ports": [{"name": "Scorer", "role": "Predictor", "filled_by": "/vega-pack"}, port],
           "monitoring": [{"name": n, "source": s} for n, s in monitoring]}
    path = tmp / "pipeline.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def options(pipe, plan=None):
    return SimpleNamespace(pipeline=None if pipe is None else str(pipe), monitor_plan=plan)


def test_an_unfilled_monitor_the_blueprint_gives_this_stage_is_refused(tmp_path):
    with pytest.raises(PackExit) as caught:
        monitor_pack.refuse_unfilled(options(pipeline(tmp_path)))
    refusal = caught.value
    text = " ".join(refusal.lines)
    assert refusal.code == 2
    assert "DriftWatch port (role Monitor) says filled_by /vega-pack" in text
    assert "no --monitor-plan was given" in text and "[analysis:F4]" in text
    assert "refused before the bundle was written" in text


def test_a_blueprint_naming_no_concern_still_asks_for_a_check_on_the_answer(tmp_path):
    with pytest.raises(PackExit) as caught:
        monitor_pack.refuse_unfilled(options(pipeline(tmp_path, monitoring=())))
    assert "at least one check on the answer" in " ".join(caught.value.lines)


@pytest.mark.parametrize("case", ["plan given", "not this stage's", "no pipeline", "unreadable"])
def test_what_passes_through(tmp_path, case):
    if case == "plan given":
        opts = options(pipeline(tmp_path), plan=str(tmp_path / "monitor-plan.json"))
    elif case == "not this stage's":
        opts = options(pipeline(tmp_path, filled_by=None))
    elif case == "no pipeline":
        opts = options(None)
    else:
        bad = tmp_path / "pipeline.json"
        bad.write_text("{not json", encoding="utf-8")
        opts = options(bad)
    assert monitor_pack.refuse_unfilled(opts) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
