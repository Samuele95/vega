#!/usr/bin/env python3
"""
test_smoke_unattended.py — `smoke_analysis.py --strict` passes on the layer an unattended
(automatic-pass-only) run writes, and still holds a dialogue's re-run control to its contract.

`--unattended` implies `--no-dialogue` (SKILL step 3): the layer's transcript is empty. The
smoke's `full` copy used to check the transcript's re-run control unconditionally, so every
unattended /data-lens run failed SKILL step 7 on three checks (`rerun-reveals-code`,
`rerun-code-nonempty`, `rerun-toggles-back`) of a control that cannot exist (the T5 rehearsal,
round 1). With no turn the smoke now checks what the page shows instead — the transcript's
empty state and no re-run control — and with a grounded turn the three re-run checks hold as
before (and fail on a turn whose stored code is empty).

A headless browser is required ($CHROME or the usual names): the smoke runs --strict.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = HERE.parents[1]
SCRIPTS = SKILL / "scripts"
SENSORS = SKILL / "fixtures" / "sensors.csv"
sys.path.insert(0, str(SCRIPTS))

from apply_analysis_layer import MODULES  # noqa: E402


def _run(args, timeout=300):
    proc = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                          timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


def _layer(transcript):
    return {
        "schema": "data-lens/analysis@1",
        "source": {"path": str(SENSORS), "rows": 1440, "columns": 9, "seed": 7,
                   "geometry": "absent"},
        "context": {"typing": [], "basis": [], "partition": {},
                    "reading": "no geometry layer; the automatic pass read the table alone"},
        "modules": {m: {"ran": False, "skipped_because": "not run in this test"} for m in MODULES},
        "findings": [{
            "id": "F1", "module": "quality", "severity": "low", "title": "calibration is sparse",
            "columns": ["calibration"], "evidence": {"missing_rate": 0.2},
            "method": {"name": "missingness count", "assumptions_checked": "n/a"},
            "reading": "calibration is missing more often at one station",
            "so_what": {"preprocessing": "impute within the station"}}],
        "transcript": transcript,
        "handoff": {"shaper_candidates": []},
    }


def _layered(tmp_path, name, transcript):
    base = tmp_path / "sensors.base.html"
    if not base.is_file():
        rc, log = _run([SCRIPTS / "bootstrap_base.py", SENSORS, "--out", base])
        assert rc == 0, log[-2000:]
    data = tmp_path / f"{name}.layer.json"
    data.write_text(json.dumps(_layer(transcript)), encoding="utf-8")
    out = tmp_path / f"{name}.analysis.html"
    rc, log = _run([SCRIPTS / "apply_analysis_layer.py", base, "--data", data, "--out", out])
    assert rc == 0, log[-2000:]
    return out


def _smoke(path):
    return _run([SCRIPTS / "smoke_analysis.py", path, "--strict"])


GROUNDED = {"turn": 1, "question": "Is calibration missing more often at the hill station?",
            "answer": "Yes.", "grounded": True,
            "code": "result = df.groupby('station')['calibration'].apply(lambda s: s.isna().mean())",
            "result": {"hill": 0.62, "lake": 0.06}}


def test_an_automatic_pass_without_dialogue_passes_strict_and_shows_the_empty_transcript(tmp_path):
    rc, log = _smoke(_layered(tmp_path, "unattended", []))
    assert rc == 0, log[-3000:]
    assert "OK   full:transcript-empty-state" in log and "OK   full:no-rerun-controls" in log, log
    assert "rerun-reveals-code" not in log, log


def test_a_grounded_turn_is_still_held_to_its_rerun_control(tmp_path):
    rc, log = _smoke(_layered(tmp_path, "dialogue", [GROUNDED]))
    assert rc == 0, log[-3000:]
    for name in ("rerun-reveals-code", "rerun-code-nonempty", "rerun-toggles-back"):
        assert f"OK   full:{name}" in log, log
    assert "transcript-empty-state" not in log, log


def test_a_turn_whose_stored_code_is_empty_fails_strict(tmp_path):
    turn = dict(GROUNDED, grounded=False, code=None, result=None)
    rc, log = _smoke(_layered(tmp_path, "nocode", [turn]))
    assert rc == 1 and "FAIL full:rerun-code-nonempty" in log, log[-3000:]
