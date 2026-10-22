"""
test_class_support.py — evaluate.class_support: the sealed test part's count of each class, the
support every per-class reading rests on (gate C: R7's Heartbleed scored 1 and SQL injection 0 on
a handful of rows, and the showcase lost the caveat because no record said how few they were).

    python3 -m pytest -q tests/test_class_support.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import class_support  # noqa: E402


def test_class_indices_are_counted_by_name():
    y = np.array([0, 2, 2, 1, 2, 0])
    assert class_support(y, ["benign", "bot", "ddos"]) == {"benign": 2, "bot": 1, "ddos": 3}


def test_a_class_never_seen_counts_zero():
    assert class_support(np.array([0, 0, 1]), ["a", "b", "c"]) == {"a": 2, "b": 1, "c": 0}


def test_multilabel_counts_each_labels_positives():
    y = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]])
    assert class_support(y, ["x", "y", "z"]) == {"x": 2, "y": 0, "z": 2}


def test_no_classes_no_truth_or_no_indices_is_no_support():
    assert class_support(None, ["a", "b"]) is None
    assert class_support(np.array([0.5, 1.5]), ["a", "b"]) is None, "not class indices"
    assert class_support(np.array([1.0, 2.0]), ()) is None, "a reading without classes"
    assert class_support(np.array([0, 5]), ["a", "b"]) is None, "an index past the classes"
