"""
test_lens_segments.py — the analysis's segments become slices of the test rows, or the layer says
why not (eval-contract §3, slices; the open item of rehearsal final3r2).

    python3 -m pytest -q vega-eval/scripts/tests/test_lens_segments.py

`build_slices` read `modules.segments` as `{name: {column, value}}`, a shape no stage writes:
/data-lens records k-means profiles (`evidence.profiles[]`) and, now, the rule that places a row
in a segment (`evidence.assignment`). So a segment slice could never be cut from a real analysis,
and nothing said so. Here:

- a real /data-lens run (`analysis.py` on its sensors fixture, against the fixture's geometry
  layer) names segments, and the test rows — the raw rows the shaper's `__split_raw__.test`
  addresses, read through a manifest as `evaluate.py` reads them — are cut into one slice per
  segment, each row in the segment the analysis's own rule puts it in;
- the segments module of a real published layer that predates the rule (rehearsal final3r2,
  `fixtures/lens-segments-final3r2.json`) cuts nothing, and `not_cut` and a `WARN:` say why;
- a module that did not run, a rule whose raw rows cannot be read, and a segment too small to
  read are each said, never dropped.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import lens_segments                                                      # noqa: E402
from evaluate import build_slices                                         # noqa: E402
from test_scoring_threshold import scoring_with_threshold                 # noqa: E402
from vega_eval_core import Resampler                                      # noqa: E402

LENS = HERE.parents[2] / "data-lens"
SPEC = {"metrics": {"primary": "f1_micro"}, "data_contract": {"features": []}}
FINAL3R2 = json.loads((HERE / "fixtures" / "lens-segments-final3r2.json")
                      .read_text(encoding="utf-8"))["segments"]


def slices_of(analysis, n, raw_test=None, seed=7):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    y, score = rng.integers(0, 2, n), rng.random(n)
    markers, not_cut = [], []
    out = build_slices(frame, SPEC, analysis, y, score, scoring_with_threshold(0.5),
                       Resampler(n, 20, 7), [], markers, raw_test=raw_test, not_cut=not_cut)
    return out, not_cut, markers


@pytest.fixture(scope="module")
def lens_run(tmp_path_factory):
    """analysis.py as /data-lens runs it, on its sensors fixture and geometry layer."""
    tmp = tmp_path_factory.mktemp("lens")
    layer = (LENS / "fixtures" / "sensors.geometry-layer.json").read_text(encoding="utf-8")
    model = tmp / "sensors.model.html"
    model.write_text('<html><script id="domain-model" type="text/turtle">ex:x ex:sourceKind '
                     '"dataset" .</script><script id="layer-geometry-data" type="application/json">'
                     f'{layer}</script></html>', encoding="utf-8")
    out = tmp / "analysis.json"
    run = subprocess.run([sys.executable, str(LENS / "scripts" / "analysis.py"),
                          str(LENS / "fixtures" / "sensors.csv"), "--model", str(model),
                          "--split", "split", "--out", str(out), "--seed", "7"],
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    return tmp, json.loads(out.read_text(encoding="utf-8"))


def shaper_records(tmp, positions):
    """The two shaper records `raw_test_rows` reads: a manifest naming the raw input and
    fitted.json, and fitted.json's raw positions of the test rows (in the part's order)."""
    shaped = tmp / "shaped"
    shaped.mkdir(exist_ok=True)
    (shaped / "fitted.json").write_text(json.dumps(
        {"__split_raw__": {"test": [int(p) for p in positions]}}), encoding="utf-8")
    manifest = shaped / "manifest.json"
    manifest.write_text(json.dumps({
        "input": {"path": str(LENS / "fixtures" / "sensors.csv")},
        "outputs": {"dir": str(shaped), "fitted": "fitted.json"}}), encoding="utf-8")
    return manifest


def test_a_real_lens_run_names_segments_and_the_test_rows_are_cut_into_them(lens_run):
    tmp, doc = lens_run
    segments = doc["modules"]["segments"]
    rule, why = lens_segments.segment_plan(segments)
    assert why is None and rule is not None and rule.k == len(rule.segments) >= 2
    raw = pd.read_csv(LENS / "fixtures" / "sensors.csv")
    positions = np.flatnonzero(raw["split"].astype(str) == "test")[::-1]     # the part's order
    rows, why = lens_segments.raw_test_rows(shaper_records(tmp, positions), len(positions),
                                            rule.features)
    assert why is None and list(rows.columns) == list(rule.features)
    assert rows.reset_index(drop=True).equals(
        raw.iloc[positions][list(rule.features)].reset_index(drop=True))
    out, not_cut, markers = slices_of({"modules": doc["modules"]},
                                      len(positions), raw_test=rows)
    cut = [s for s in out if s["source"] == "analysis:modules/segments"]
    placed = rule.assign(rows)
    want = {f"segment:{s}": int((placed == s).sum()) for s in rule.segments
            if (placed == s).sum() >= 5}
    assert {s["name"]: s["n"] for s in cut} == want and want, (cut, not_cut, markers)
    assert all(s["note"].startswith("k-means segment") for s in cut)
    assert not [g for g in not_cut if "rule" in g["why"]], not_cut


def test_a_published_layer_without_the_rule_cuts_nothing_and_says_why():
    out, not_cut, markers = slices_of({"modules": {"segments": FINAL3R2}}, 60)
    assert not [s for s in out if s["source"] == "analysis:modules/segments"]
    assert len(not_cut) == 1 and not_cut[0]["source"] == "analysis:modules/segments"
    assert "3 k-means segment profiles but not the rule" in not_cut[0]["why"], not_cut
    assert any(m.startswith("WARN: a segment slice was not cut:") for m in markers), markers


def test_a_module_that_did_not_run_and_rows_that_cannot_be_read_are_said():
    skipped = {"ran": False, "skipped_because": "failed: boom", "evidence": {}}
    _, not_cut, _ = slices_of({"modules": {"segments": skipped}}, 40)
    assert "did not run (failed: boom)" in not_cut[0]["why"]
    seg = {"ran": True, "evidence": {"chosen_k": 2, "features": ["a", "b"],
                                     "profiles": [{"segment": 0, "n": 9}, {"segment": 1, "n": 9}],
                                     "assignment": {"features": ["a", "b"], "center": [0, 0],
                                                    "scale": [1, 1],
                                                    "centroids": [[-1, 0], [1, 0]],
                                                    "segments": [0, 1]}}}
    _, not_cut, _ = slices_of({"modules": {"segments": seg}}, 40,
                              raw_test="a window sample reads more than one raw row")
    assert not_cut == [{"source": "analysis:modules/segments",
                        "why": "a window sample reads more than one raw row"}]
    raw = pd.DataFrame({"a": [-1.0] * 37 + [1.0, 1.0, np.nan], "b": [0.0] * 40})
    out, not_cut, markers = slices_of({"modules": {"segments": seg}}, 40, raw_test=raw)
    assert [(s["name"], s["n"]) for s in out] == [("segment:0", 37)]
    assert "segment 1 of 2 holds 2 test row(s)" in not_cut[0]["why"], not_cut
    assert any("1 test row(s) miss one of the segment features" in m for m in markers)


def test_raw_test_rows_refuses_a_count_that_is_not_the_rows_scored(tmp_path):
    manifest = shaper_records(tmp_path, [3, 1, 2])
    rows, why = lens_segments.raw_test_rows(manifest, 4, ["reading_c"])
    assert rows is None and "3 position(s) for 4 scored test row(s)" in why
    rows, why = lens_segments.raw_test_rows(manifest, 3, ["no_such_column"])
    assert rows is None and "does not hold" in why


def test_raw_test_rows_reads_a_parquet_input_in_the_parts_order(tmp_path):
    raw = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [5.0, 6.0, 7.0, 8.0]})
    raw.to_parquet(tmp_path / "raw.parquet", index=False)
    manifest = shaper_records(tmp_path, [3, 0])
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["input"]["path"] = str(tmp_path / "raw.parquet")
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    rows, why = lens_segments.raw_test_rows(manifest, 2, ["b"])
    assert why is None and rows["b"].tolist() == [8.0, 5.0]
