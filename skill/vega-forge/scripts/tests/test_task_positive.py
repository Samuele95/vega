#!/usr/bin/env python3
"""
test_task_positive.py — T1e on the positive class of a binary task (PLAN-AMENDMENTS A16).

    python3 -m pytest -q vega-forge/scripts/tests/test_task_positive.py

The logit of a binary head, the decision threshold, the serving readers (a binary record whose
positive is not `classes[1]` is refused there) and the page all read `task.classes[1]` as the
positive class. The compiler's T1e therefore refuses a binary — or sequence-classification over
a binary base — spec whose `task.positive` is `classes[0]`, with "list the positive class
second", before anything is emitted; `positive` stays optional on binary (it defaults to the
second class), and a positive outside `classes` keeps its own refusal. Only T1e is read: the
other rules judge a whole chain, which these specs are not.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))

import compile_spec  # noqa: E402


def _spec(task: dict) -> dict:
    return {"schema": compile_spec.SCHEMA, "input": {}, "seed": 0, "task": task,
            "data_contract": {}, "family": {"name": "mlp-plain"},
            "blocks": [{"type": "dense", "params": {"units": 4}},
                       {"type": "head", "params": {"out": 1}}],
            "loss": {"name": "bce_with_logits"}, "metrics": {"primary": "pr_auc"},
            "optimizer": {"name": "adamw"}, "schedule": {"name": "none"},
            "regularization": [], "early_stopping": {}, "init": {}, "training": {},
            "backend": {"name": "pytorch"}, "outputs": {}}


def _t1e(task: dict) -> list:
    markers = compile_spec.compile_spec(_spec(task), backend="pytorch").as_json()
    return [m for m in markers if m.get("rule") == "T1e"]


def _errors(task: dict) -> list:
    return [m["message"] for m in _t1e(task) if m["level"] == "ERROR"]


def test_the_second_class_is_the_positive_one():
    assert _errors({"kind": "binary", "label": "late", "classes": [0, 1], "positive": 1}) == []
    assert _errors({"kind": "binary", "label": "odd", "classes": ["no", "yes"],
                    "positive": "yes"}) == []
    assert _errors({"kind": "binary", "label": "late", "classes": [0, 1]}) == [], \
        "positive stays optional on binary"


def test_the_first_class_as_positive_is_refused():
    errors = _errors({"kind": "binary", "label": "late", "classes": [0, 1], "positive": 0})
    assert len(errors) == 1, errors
    assert "list the positive class second" in errors[0]
    assert "must be the second class" in errors[0] and "[1, 0]" in errors[0], errors[0]
    errors = _errors({"kind": "binary", "label": "odd", "classes": ["yes", "no"],
                      "positive": "yes"})
    assert errors and "['no', 'yes']" in errors[0], errors


def test_a_sequence_task_over_a_binary_base_is_held_to_it():
    task = {"kind": "sequence-classification", "base_kind": "binary", "label": "fault",
            "classes": ["ok", "fault"]}
    assert _errors(dict(task, positive="fault")) == []
    errors = _errors(dict(task, classes=["fault", "ok"], positive="fault"))
    assert errors and "list the positive class second" in errors[0], errors


def test_a_positive_outside_the_classes_keeps_its_own_refusal():
    errors = _errors({"kind": "binary", "label": "late", "classes": [0, 1], "positive": 7})
    assert len(errors) == 1 and "is not one of task.classes" in errors[0], errors


def test_other_kinds_are_not_ordered_by_it():
    """A multiclass or a labelled anomaly task names its positive (or none) without an order."""
    assert _errors({"kind": "multiclass", "label": "c", "classes": ["a", "b", "c"]}) == []


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
