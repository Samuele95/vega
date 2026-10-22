"""
test_apply_columns.py — apply_analysis_layer.py refuses a `columns` that is not a list of column
names (F38, rehearsal final3r2): a transformation candidate written with `"columns": "numeric"`
validated against data-lens/analysis@1 and the rendered analysis tab then threw
`(t.columns || []).join is not a function`, caught only by validate_model's invariant 16.

    python3 -m pytest -q data-lens/scripts/tests/test_apply_columns.py
"""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import apply_analysis_layer as AL  # noqa: E402

DOC = {
    "schema": AL.SCHEMA,
    "source": {"path": "landcover.csv", "geometry": "present", "seed": 7},
    "context": {"typing": [], "basis": ["b01", "b02"], "partition": {"label": "cover", "features": ["b01", "b02"]},
                "reading": "inherits the geometry layer"},
    "modules": {m: {"ran": False, "skipped_because": "not needed here"} for m in AL.MODULES},
    "findings": [{
        "id": "F1", "module": "distributions", "severity": "medium", "title": "the bands are skewed",
        "columns": ["b01", "b02"], "evidence": {}, "method": {"assumptions_checked": "n/a"},
        "reading": "r", "so_what": "s",
        "transformation_candidates": [{"id": "T1", "op": "scale", "columns": ["b01", "b02"],
                                       "params": {"method": "robust"}, "rationale": "F1"}]}],
    "transcript": [],
    "handoff": {"shaper_candidates": ["T1"]},
    "stances": [{"kind": "range", "columns": ["b01"], "assertion": "reflectance is in [0, 1]",
                 "source": "user"}],
}


def _errors(doc):
    return AL.validate(doc, lambda line: None)


def test_the_well_formed_layer_validates():
    assert _errors(copy.deepcopy(DOC)) == []


@pytest.mark.parametrize("where", ["candidate", "finding", "stance"])
@pytest.mark.parametrize("bad", ["numeric", None, ["b01", 3], [""]])
def test_columns_that_are_not_a_list_of_names_are_refused(where, bad):
    doc = copy.deepcopy(DOC)
    target = {"candidate": doc["findings"][0]["transformation_candidates"][0],
              "finding": doc["findings"][0], "stance": doc["stances"][0]}[where]
    target["columns"] = bad
    errs = _errors(doc)
    assert any(".columns must be a list of column names" in e for e in errs), errs
