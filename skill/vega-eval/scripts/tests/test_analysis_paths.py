"""
test_analysis_paths.py — /vega-eval reads the analysis layer at the paths /data-lens writes
(F39, rehearsal final3r2).

    python3 -m pytest -q vega-eval/scripts/tests/test_analysis_paths.py

The robustness check read `modules.importance.ranking` (or `.top`) and the slices and the drift
sides read `modules.drift.columns` as a list of names; the analysis contract (data-lens
analysis-contract.md §1) records `modules.importance.evidence.permutation_importance[]` and
`modules.drift.evidence.columns[]` (`{column, measure, value, drifted}`), as eval-contract.md
§3 says to read them. On every real analysis the importance perturbation was therefore skipped
"as if no ranking existed" (two rehearsals in a row) and a drifted column never became a slice.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from eval_package import drifted_columns, lens_ranking                     # noqa: E402
from evaluate import build_robustness, build_slices                       # noqa: E402
from test_scoring_threshold import fixture, scoring_with_threshold        # noqa: E402
from vega_eval_core import Resampler                                      # noqa: E402

#: the two modules as data-lens's analysis.py writes them (analysis-contract §1)
IMPORTANCE = {"ran": True, "reading": "r", "evidence": {
    "models": [{"name": "logistic-l2", "cv_score": 0.84, "ci": [0.8, 0.88]},
               {"name": "gradient-boosting", "cv_score": 0.87, "ci": [0.83, 0.9]}],
    "permutation_importance": [
        {"feature": "region", "mean": 0.05, "ci": [0.0, 0.1], "model": "logistic-l2"},
        {"feature": "dist_km", "mean": 0.19, "ci": [0.1, 0.2], "model": "logistic-l2"},
        {"feature": "absent_col", "mean": 0.01, "ci": [0.0, 0.02], "model": "logistic-l2"},
        {"feature": "region", "mean": 0.50, "ci": [0.4, 0.6], "model": "gradient-boosting"}]}}
DRIFT = {"ran": True, "reading": "r", "evidence": {"parts": ["train", "test"], "columns": [
    {"column": "region", "measure": "cramers_v", "value": 0.01, "drifted": False},
    {"column": "dist_km", "measure": "psi", "value": 0.31, "drifted": True}]}}
SPEC = {"metrics": {"primary": "f1_micro"},
        "data_contract": {"features": [{"name": "region", "role": "nominal"},
                                       {"name": "dist_km", "role": "numeric"}]}}


def test_the_ranking_is_the_permutation_importance_of_the_lens_model_by_mean():
    assert lens_ranking(IMPORTANCE) == ["dist_km", "region", "absent_col"]
    assert lens_ranking({"ranking": ["a", "b"]}) == ["a", "b"]          # an older record
    assert lens_ranking({"ran": False, "skipped_because": "no partition"}) == []
    assert lens_ranking(None) == []


def test_the_drifted_columns_are_the_evidence_rows_marked_drifted():
    assert drifted_columns(DRIFT) == ["dist_km"]
    assert drifted_columns({"columns": ["dist_km"]}) == ["dist_km"]     # an older record
    assert drifted_columns({"ran": False, "evidence": {}}) == []


def test_robustness_perturbs_what_the_lens_ranked_and_reads_the_drift_sides():
    frame, y, score = fixture()
    scoring = scoring_with_threshold(0.5)
    analysis = {"modules": {"importance": IMPORTANCE, "drift": DRIFT}, "context": {}}
    checks = build_robustness(frame, SPEC, analysis, y, lambda f: score, scoring,
                              Resampler(len(y), 20, 7), frame, [])
    by = {c["check"]: c for c in checks}
    assert "importance-perturbation" not in by, by.get("importance-perturbation")
    assert "status" not in by["perturb:dist_km"], by["perturb:dist_km"]
    assert by["perturb:region"]["status"] == "skipped"                  # nominal, said so
    assert "dist_km drifted (> median)" in by and "drift-slices" not in by


def test_a_drifted_column_becomes_a_slice():
    frame, y, score = fixture()
    analysis = {"modules": {"drift": DRIFT}, "context": {}}
    slices = build_slices(frame, SPEC, analysis, y, score, scoring_with_threshold(0.5),
                          Resampler(len(y), 20, 7), [], [])
    names = {s["name"] for s in slices}
    assert {"dist_km<=median", "dist_km>median"} <= names, names
