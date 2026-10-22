#!/usr/bin/env python3
"""
test_lanekit_budget.py — a lane's own campaign budget (A38) and the registry's change records.

    python3 -m pytest test_lanekit_budget.py [-q]

`lanes.json` may carry, per lane, a `budget` override {max_epochs, patience} with a required
`reason`; the composer reads it and the global defaults apply otherwise; the stage time limits
stay the lane's `budget_s`. What is proved:

1. `Budget.for_lane` returns the defaults' budget for a lane without an override, the lane's own
   (its reason kept) for one with it, and refuses an override without a reason.
2. The composed spec of a lane with an override trains for its epochs and stops at its patience,
   and the training and early-stopping rationales carry the reason; a lane without one keeps the
   defaults and their words, unchanged.
3. `lanes.schema.json` refuses an override without a reason (or with a field it does not know)
   and a change record without its amendment or its why; the tree's registry validates, and
   the lanes A38 changed (L10, L16) carry both records.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import registry  # noqa: E402
from lanekit.spec_data import Budget, SpecError  # noqa: E402
from test_lanekit_spec import _compose, _variant  # noqa: E402

LANES = registry.load_lanes()
DEFAULTS = registry.load_document()["defaults"]
SCHEMA = json.loads((HERE / "lanes.schema.json").read_text(encoding="utf-8"))
REASON = "the network still improves at the default's last epoch; a longer run is what shows it"


def _lane_errors(raw: dict) -> list:
    validator = Draft202012Validator({"$ref": "#/$defs/lane", "$defs": SCHEMA["$defs"]})
    return sorted(e.message for e in validator.iter_errors(raw))


# --------------------------------------------------------------------------- 1. the override
def test_a_lane_without_an_override_keeps_the_defaults_budget():
    base = Budget.of(DEFAULTS)
    assert base.for_lane(_variant("L14")) is base
    assert base.phrase() == "the lane budget"


def test_a_lane_with_an_override_takes_its_own_budget_and_reason():
    base = Budget.of(DEFAULTS)
    own = base.for_lane(_variant("L14", budget={"max_epochs": 24, "patience": 6, "reason": REASON}))
    assert (own.max_epochs, own.patience, own.reason) == (24, 6, REASON)
    assert own.wall_clock_min == base.wall_clock_min, "the time limits are the defaults'"
    assert own.phrase() == f"the lane's own budget ({REASON})"


def test_an_override_without_a_reason_is_refused():
    with pytest.raises(SpecError, match="carries its reason"):
        Budget.of(DEFAULTS).for_lane(_variant("L14", budget={"max_epochs": 24, "patience": 6,
                                                             "reason": " "}))


# --------------------------------------------------------------------------- 2. the composed spec
@pytest.mark.parametrize("lane_id", ["L10", "L16"])
def test_the_composed_spec_trains_for_the_lanes_own_budget(tmp_path, lane_id):
    own = LANES[lane_id].raw["budget"]
    doc = _compose(tmp_path, LANES[lane_id])
    assert doc["training"]["max_epochs"] == own["max_epochs"]
    assert doc["early_stopping"]["patience"] == own["patience"]
    for node in (doc["training"], doc["early_stopping"]):
        assert own["reason"] in node["rationale"], node["rationale"]
    assert doc["schedule"]["params"].get("warmup_epochs") == (3 if own["max_epochs"] >= 30 else 1)


def test_a_lane_without_an_override_composes_the_defaults_as_before(tmp_path):
    doc = _compose(tmp_path, LANES["L14"])
    assert doc["training"]["max_epochs"] == DEFAULTS["max_epochs"]
    assert doc["early_stopping"]["patience"] == DEFAULTS["patience"]
    assert doc["early_stopping"]["rationale"].startswith("the lane budget: patience")
    assert doc["training"]["rationale"].endswith("are the lane budget (user:stance/lane-budget)")


# --------------------------------------------------------------------------- 3. the registry
def test_the_schema_holds_the_override_and_the_change_records():
    raw = copy.deepcopy(dict(LANES["L16"].raw))
    assert _lane_errors(raw) == []
    no_reason = copy.deepcopy(raw)
    del no_reason["budget"]["reason"]
    assert any("'reason' is a required property" in m for m in _lane_errors(no_reason))
    extra = copy.deepcopy(raw)
    extra["budget"]["wall_clock_min"] = 9
    assert _lane_errors(extra), "an override names epochs and patience only: time limits stay"
    no_why = copy.deepcopy(raw)
    del no_why["changes"][0]["why"]
    assert any("'why' is a required property" in m for m in _lane_errors(no_why))
    bad = copy.deepcopy(raw)
    bad["changes"][0]["amendment"] = "later"
    assert _lane_errors(bad)


def test_the_lanes_a38_changed_record_their_budget_and_change():
    for lane_id in ("L10", "L16"):
        raw = LANES[lane_id].raw
        assert raw["budget"]["reason"] and raw["budget"]["max_epochs"] > DEFAULTS["max_epochs"]
        assert {c["amendment"] for c in raw["changes"]} == {"A38"}
    assert [b["type"] for b in LANES["L10"].raw["blocks"]] == ["embedding", "concat", "gru", "head"]
    assert LANES["L10"].raw["blocks"][2]["params"]["return"] == "last"
    assert [b["type"] for b in LANES["L16"].raw["blocks"]] == ["lstm", "attention", "head"], \
        "the catalog's attention block keeps its lane (test_lanekit.py)"
    assert sum("budget" in lane.raw for lane in LANES.values()) == 2, "every other lane keeps the defaults"
