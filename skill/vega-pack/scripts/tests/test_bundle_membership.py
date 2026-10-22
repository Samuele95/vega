#!/usr/bin/env python3
"""
test_bundle_membership.py — no bundle shape ships the sealed test part's membership.

    python3 test_bundle_membership.py [BUNDLE_DIR ...]   # or: python3 -m pytest test_bundle_membership.py
    VEGA_PACK_BUNDLES=<dir>:<dir> python3 -m pytest test_bundle_membership.py

The shaper records WHICH rows went where: `fitted.json`'s `__split__`, `__split_raw__` and
`__removed_rows__`, and in the shape layer the split and dedupe steps' `indices`,
`raw_indices` and `raw_rows_removed`. Shipped in a bundle, any of them tells its holder which
raw rows were the sealed test part (bundle-contract §1, the vega-pack verifier's S3). Both
bundle shapes — the serving bundle and `--include full` — ship `fitted.json` as a projection
and `records/shape.json` without the positions; the counts (`sizes`, `rows_removed`) stay.

Pure by default: the projection and the scrub run on small files written here, and a seeded
negative re-introduces every key so the checker is shown to see it. Given bundle directories
(arguments or `VEGA_PACK_BUNDLES`, e.g. the acceptance's `bundle/<name>` and
`bundle-full/<name>`), the same checker reads the real files.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
sys.path.insert(0, str(SCRIPTS))

from bundle_inputs import (MEMBERSHIP_KEYS, MEMBERSHIP_PARAMS,  # noqa: E402
                           project_fitted, without_membership)

PREPROCESS = "FITTED = {}\n\ndef transform(df):\n    a = FITTED['S07']\n    b = FITTED.get('S08')\n"
FITTED = {"S01": {"retyped": []}, "S07": {"fitted": {"x": [0, 1]}},
          "S08": {"fitted": {"x": [0.0, 1.0]}}, "__code__": "x",
          "__split__": {"train": [0, 1], "valid": [2], "test": [3]},
          "__split_raw__": {"train": [0, 1], "valid": [2], "test": [3]},
          "__removed_rows__": [4]}


def _step(op, **fitted):
    return {"op": op, "parameters_fitted": fitted}


SHAPE = {"schema": {}, **{holder: {"steps": [
    _step("dedupe", rows_removed=1, raw_rows_removed=[4]),
    _step("split", kind="time", sizes={"train": 2, "valid": 1, "test": 1},
          indices={"test": [3]}, raw_indices={"test": [3]}),
    _step("scale", kind="standard", fitted={"x": [0.0, 1.0]})]} for holder in ("recipe", "manifest")}}


def leaks(bundle: Path) -> list[str]:
    """Every membership record the bundle at `bundle` carries, named by file and key."""
    found = []
    fitted = bundle / "fitted.json"
    if fitted.is_file():
        doc = json.loads(fitted.read_text(encoding="utf-8"))
        found += [f"fitted.json:{k}" for k in MEMBERSHIP_KEYS if k in doc]
    shape = bundle / "records" / "shape.json"
    if shape.is_file():
        doc = json.loads(shape.read_text(encoding="utf-8"))
        for holder in ("recipe", "manifest"):
            for step in (doc.get(holder) or {}).get("steps") or ():
                fitted_params = (step or {}).get("parameters_fitted") or {}
                found += [f"records/shape.json:{holder}.{step.get('op')}.{k}"
                          for k in MEMBERSHIP_PARAMS if k in fitted_params]
    return found


def _project(mode: str) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pkg = root / "pipe"
        (pkg / "features").mkdir(parents=True)
        (pkg / "features" / "preprocess.py").write_text(PREPROCESS, encoding="utf-8")
        (root / "fitted.json").write_text(json.dumps(FITTED), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            record = project_fitted(SimpleNamespace(root=root, pkg=pkg), dict(FITTED), mode, [])
        return record, json.loads((root / "fitted.json").read_text(encoding="utf-8"))


def _bundle(root: Path, fitted: dict, shape: dict) -> Path:
    (root / "records").mkdir(parents=True, exist_ok=True)
    (root / "fitted.json").write_text(json.dumps(fitted), encoding="utf-8")
    (root / "records" / "shape.json").write_text(json.dumps(shape), encoding="utf-8")
    return root


def test_both_bundle_shapes_project_fitted_json():
    """The serving bundle and `--include full` alike keep only the keys preprocess.py reads."""
    for mode in ("serving", "full"):
        record, shipped = _project(mode)
        assert sorted(shipped) == ["S07", "S08"], (mode, sorted(shipped))
        assert not set(MEMBERSHIP_KEYS) & set(shipped), (mode, sorted(shipped))
        assert set(MEMBERSHIP_KEYS) <= set(record["dropped"]), (mode, record)
        assert record["digest_source"] != record["digest"], mode


def test_records_shape_json_keeps_counts_not_positions():
    markers = []
    with contextlib.redirect_stdout(io.StringIO()):
        scrubbed = without_membership(SHAPE, markers)
    for holder in ("recipe", "manifest"):
        steps = {s["op"]: s["parameters_fitted"] for s in scrubbed[holder]["steps"]}
        assert steps["dedupe"] == {"rows_removed": 1}, steps["dedupe"]
        assert steps["split"] == {"kind": "time", "sizes": {"train": 2, "valid": 1, "test": 1}}
        assert steps["scale"]["fitted"] == {"x": [0.0, 1.0]}
    assert "raw_indices" in json.dumps(SHAPE), "the scrub must not mutate its input"
    assert any(m.startswith("OK: records/shape.json carries no row membership") for m in markers)


def test_a_scrubbed_bundle_has_no_leak():
    with tempfile.TemporaryDirectory() as td:
        _, shipped = _project("full")
        with contextlib.redirect_stdout(io.StringIO()):
            shape = without_membership(SHAPE, [])
        assert leaks(_bundle(Path(td), shipped, shape)) == []


def test_seeded_negative_every_membership_key_is_seen():
    """The checker is not vacuous: re-introducing the keys a bundle must not carry is caught,
    one finding per key and per holder."""
    with tempfile.TemporaryDirectory() as td:
        found = leaks(_bundle(Path(td), FITTED, SHAPE))
    expected = {f"fitted.json:{k}" for k in MEMBERSHIP_KEYS} | {
        f"records/shape.json:{h}.{op}.{k}" for h in ("recipe", "manifest")
        for op, k in (("dedupe", "raw_rows_removed"), ("split", "indices"),
                      ("split", "raw_indices"))}
    assert set(found) == expected, sorted(set(found) ^ expected)


def _given_bundles(argv=()) -> list[Path]:
    named = [*argv, *filter(None, os.environ.get("VEGA_PACK_BUNDLES", "").split(os.pathsep))]
    return [Path(p) for p in named]


def test_given_bundles_carry_no_membership():
    """Real bundles (the acceptance's `bundle/<name>` and `bundle-full/<name>`), when named."""
    for bundle in _given_bundles():
        assert (bundle / "MANIFEST.json").is_file(), f"{bundle} is not a bundle"
        assert leaks(bundle) == [], (str(bundle), leaks(bundle))


def main(argv) -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    for bundle in (Path(a) for a in argv):
        found = leaks(bundle) if (bundle / "MANIFEST.json").is_file() else ["not a bundle"]
        failed += bool(found)
        print(f"  {'FAIL' if found else 'ok'}    {bundle}{' — ' + ', '.join(found) if found else ''}")
    print(f"\n{len(tests) + len(argv) - failed} of {len(tests) + len(argv)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
