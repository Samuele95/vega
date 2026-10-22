#!/usr/bin/env python3
"""
test_defaults_paths_twin.py — the forge's `defaults_to_test` check and the trainer's delta-path
resolver give one verdict on every path form (F40).

    python3 -m pytest test_defaults_paths_twin.py [-q]

`vega-forge/scripts/defaults_paths.py` refuses, at the layer's apply, an entry `/vega-train`
could not move; `plan_ladder.py` reads the same entry with `read_path(normalise_path(…))`. If
the two disagreed, the forge would publish an entry the ladder drops (the F40 symptom) or
refuse one the ladder runs. Every path the forge admits resolves through the trainer's
resolver, and every path the trainer cannot resolve the forge refuses. A default-test rung is
then drafted from each admitted regularization entry, and the release4 rehearsal's two
entries draft none.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))

import vega_train_core as core  # noqa: E402
import plan_ladder  # noqa: E402

D = core.import_forge("defaults_paths")
TESTS = core.forge_scripts() / "tests"
sys.path.insert(0, str(TESTS))
import test_defaults_paths as forge_cases  # noqa: E402

SPEC = forge_cases.SPEC
PATHS = forge_cases.GOOD + forge_cases.BAD_GRAMMAR + forge_cases.BAD_READ


def _trainer_resolves(path):
    try:
        core.read_path(SPEC, core.normalise_path(path))
        return True
    except core.PathError:
        return False


@pytest.mark.parametrize("path", PATHS)
def test_one_verdict_on_resolution(path):
    forge_ok, _ = D.resolves(SPEC, path)
    assert forge_ok == _trainer_resolves(path)


@pytest.mark.parametrize("path", PATHS)
def test_an_admitted_path_resolves_for_the_trainer(path):
    if D.problem(SPEC, path) is None:
        assert _trainer_resolves(path)


@pytest.mark.parametrize("kind", ["dropout", "weight_decay"])
def test_an_admitted_regularization_entry_drafts_its_rung(kind):
    entry = {"path": f"regularization[kind={kind}]/value", "observation": "the gap"}
    assert D.problem(SPEC, entry["path"]) is None
    rung, why = plan_ladder.rung_default_test(entry, SPEC, 1)
    assert rung is not None, why
    assert rung["delta"][0]["path"] == entry["path"]


@pytest.mark.parametrize("kind", ["dropout", "weight_decay"])
def test_the_rehearsal_entry_is_refused_where_the_ladder_would_drop_it(kind):
    entry = {"path": f"regularization[{kind}]", "observation": "the gap"}
    rung, _ = plan_ladder.rung_default_test(entry, SPEC, 1)
    assert rung is None
    assert D.problem(SPEC, entry["path"])
