"""
demo_pipeline — a package in the shape the v2.3 templates emit, small enough to read.

It is the conformance fixture: every port adapter a class alone at its module's top level,
`build_model = Network.build`, every method within the size discipline. It passes C1–C14 at
`--size-severity error`, and test_conformance_sizes.py plants one defect at a time in copies of
it. It is parsed, never imported.
"""
from __future__ import annotations

__all__ = ["domain", "ports"]
