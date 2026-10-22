#!/usr/bin/env python3
"""
test_adaptation_report.py — the adaptation proof's own proof (D6 §11.3; BUILD-PLAN T12).

    python3 -m pytest -q test_adaptation_report.py

Each lane here is synthetic: a page result (`results/page.json`, the readings page_smoke.py
records), a showcase with a strings fragment and a story (its role map), and an observed.json
naming the bundle. A clean L02:L01 / L10:L02 pair passes AD1–AD6; then one defect is seeded at a
time and the assertion that should catch it fails, naming it:

AD1 a style-lock value differs · AD2 a key outside the table, a missing key, a shared key with two
Englishes, a classification key on the no-class L10 · AD3 the same content twice · AD4 a leak ·
AD5 an alert drawn without an alert role, an alert role not drawn · AD6 a motif outside the
modality's, two modalities with one motif · AD7 the golden lane's own parity missing, failed or measured
with a wider tolerance, a frame box moved 9 px across the pair (8 px holds; a headline's height does not count);
and a golden side that did not run skips its pair, while a regular lane that did not run fails it.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import adaptation_report as ar  # noqa: E402
from lanekit import registry  # noqa: E402

LANES = registry.load_lanes()
TABLE = {"view.start": "Start", "try.run": "Run it", "try.run@classes": "Pick a class",
         "many.confusion": "Confusion", "target.one": "target"}
LOCK = {"computed": {"night": {"body": {"color": "rgb(227, 234, 240)"}}, "day": {"body": {"color": "rgb(15, 26, 36)"}}}}
BOX = {"header": [0, 0, 1440, 70], "nav": [0, 845, 1440, 55], "main": [0, 70, 1440, 775],
       "h1": [120, 203, 1008, 94], "canvas": [0, 0, 1440, 900]}
CONTENT = {"L01": (["orders ship from the warehouse"], ["order_value", "region", "orders_pipeline/model/store.py"]),
           "L02": (["parcels read their bands"], ["b01", "cropland", "landcover_pipeline/model/store.py"]),
           "L10": (["hours of bike demand"], ["demand_lag1", "station", "bikeshare_pipeline/model/store.py"]),
           "G00": (["flows of the network traffic"], ["flow duration", "cic_pipeline/model/store.py"])}
MOTIF = {"L01": "points", "L02": "points", "L10": "traces", "G00": "lanes"}
USED = {"L01": ["view.start", "try.run@classes"], "L02": ["view.start", "try.run@classes", "explain.dense.what"],
        "L10": ["view.start", "try.run", "target.one"], "G00": ["view.start", "try.run@classes"]}


def page(lane_id: str) -> dict:
    story, data = CONTENT[lane_id]
    roles = LANES[lane_id].raw.get("roles") or {}
    alert = sum(1 for r in roles.values() if r == "alert")
    readings = {"lock": LOCK, "palette": {"alert_used": alert > 0, "alert_roles": alert, "role_alert_elements": alert},
                "motif": {"id": MOTIF[lane_id], "data_motif": MOTIF[lane_id], "alert_marks": alert > 0},
                "missing_keys": [], "used_keys": list(USED[lane_id]),
                "boxes": {v: copy.deepcopy(BOX) for v in ("start", "context", "network")}, "content": None}
    leak = lambda: {"id": "P-LEAK", "ok": True, "problems": []}  # noqa: E731
    runs = [{"id": "R-a", "checks": [leak()], "readings": readings},
            {"id": "R-b", "checks": [leak()], "readings": {**readings, "boxes": {}}},
            {"id": "R-c", "checks": [], "readings": {**readings, "boxes": {}}},
            {"id": "R-q", "checks": [], "readings": {**readings, "boxes": {}, "content": {"story": story, "data": data}}}]
    return {"schema": "vega/page-smoke@1", "lane": lane_id, "verdict": "pass", "runs": runs}


def write_lane(root: Path, lane_id: str, doc: dict | None = None, table: dict | None = None) -> None:
    lane_dir = root / "lanes" / lane_id
    showcase = lane_dir / "work" / "bundle" / "b" / "showcase"
    (showcase / "page" / "js" / "core" / "strings").mkdir(parents=True, exist_ok=True)
    (showcase / "page" / "js" / "core" / "strings" / "core.en.js").write_text(
        "export const STRINGS = Object.freeze({\n" + "".join(f'  {json.dumps(k)}: {json.dumps(v)},\n'
                                                           for k, v in (table or TABLE).items()) + "});\n",
        encoding="utf-8")
    roles = LANES[lane_id].raw.get("roles")
    (showcase / "story.json").write_text(json.dumps({"class_roles": {"roles": roles} if roles else None}), encoding="utf-8")
    (lane_dir / "results").mkdir(parents=True, exist_ok=True)
    (lane_dir / "results" / "page.json").write_text(json.dumps(doc or page(lane_id)), encoding="utf-8")
    (lane_dir / "observed.json").write_text(json.dumps({"artifacts": {"bundle": "work/bundle/b"}}), encoding="utf-8")


def matrix(tmp_path: Path, *lanes: str) -> Path:
    for lane_id in lanes:
        write_lane(tmp_path, lane_id)
    return tmp_path


def problems(root: Path, pair: str, check: str) -> list[str]:
    result = ar.judge(root, LANES, pair)
    return next(c["problems"] for c in result["checks"] if c["id"] == check)


def seeded(root: Path, lane_id: str, edit) -> None:
    doc = page(lane_id)
    edit(doc)
    write_lane(root, lane_id, doc)


def readings(doc: dict, run_id: str = "R-a") -> dict:
    return next(r for r in doc["runs"] if r["id"] == run_id)["readings"]


# --------------------------------------------------------------------------- the clean pairs
def test_the_designated_pairs_pass_and_the_report_is_written(tmp_path):
    root = matrix(tmp_path, "L01", "L02", "L10")
    assert ar.main(["--matrix", str(root), "--pairs", "L02:L01,L10:L02"]) == 0
    doc = json.loads((root / "adaptation.json").read_text(encoding="utf-8"))
    assert doc["schema"] == "vega/adaptation-report@1" and doc["verdict"] == "PASS"
    assert [c["id"] for c in doc["pairs"][0]["checks"]] == ["AD1", "AD2", "AD3", "AD4", "AD5", "AD6"]
    assert "## L10:L02 — pass" in (root / "adaptation.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- AD1-AD6 seeded
def test_ad1_a_style_value_that_differs(tmp_path):
    root = matrix(tmp_path, "L01")
    seeded(root, "L02", lambda d: readings(d, "R-c").update(lock={"computed": {"day": {"body": {"color": "red"}}}}))
    found = problems(root, "L02:L01", "AD1")
    assert len(found) == 1 and found[0].startswith("day:") and "color" in found[0]


def test_ad2_keys_outside_the_table_missing_or_with_two_englishes(tmp_path):
    root = matrix(tmp_path, "L01")
    seeded(root, "L02", lambda d: readings(d)["used_keys"].append("no.such.key"))
    assert "not in L02's table" in problems(root, "L02:L01", "AD2")[0]
    seeded(root, "L02", lambda d: readings(d)["missing_keys"].append("lost.key"))
    assert any("missed key" in p for p in problems(root, "L02:L01", "AD2"))
    write_lane(root, "L02", table={**TABLE, "view.start": "Begin"})
    assert any("render different English" in p and "view.start" in p for p in problems(root, "L02:L01", "AD2"))


def test_ad2_a_page_without_classes_renders_no_classification_key(tmp_path):
    root = matrix(tmp_path, "L02", "L10")
    assert problems(root, "L10:L02", "AD2") == []
    seeded(root, "L10", lambda d: readings(d)["used_keys"].extend(["try.run@classes", "many.confusion"]))
    found = problems(root, "L10:L02", "AD2")
    assert found and "no classes" in found[0] and "many.confusion" in found[0]
    assert ar.class_only("start.fact.outputs@labels#other") and not ar.class_only("try.run@horizon")


def test_ad2_without_a_used_keys_reading_fails(tmp_path):
    root = matrix(tmp_path, "L01")
    seeded(root, "L02", lambda d: [r["readings"].update(used_keys=None) for r in d["runs"]])
    assert "reports no used strings keys" in problems(root, "L02:L01", "AD2")[0]


def test_ad3_the_same_content_twice(tmp_path):
    root = matrix(tmp_path, "L02")
    seeded(root, "L10", lambda d: readings(d, "R-q").update(content={"story": CONTENT["L02"][0], "data": CONTENT["L02"][1]}))
    result = ar.judge(root, LANES, "L10:L02")
    ad3 = next(c for c in result["checks"] if c["id"] == "AD3")
    assert not ad3["ok"] and ad3["detail"]["jaccard"] == 1.0


def test_ad3_ignores_the_chain_words_and_the_code_layout(tmp_path):
    root = matrix(tmp_path, "L02")
    chain = ["A baseline trained on the same shaped inputs scored better than the network on the sealed test part."]
    seeded(root, "L10", lambda d: readings(d, "R-q")["content"]["story"].extend(chain))
    seeded_l02 = page("L02")
    readings(seeded_l02, "R-q")["content"]["story"].extend(chain)
    write_lane(root, "L02", seeded_l02)
    result = ar.judge(root, LANES, "L10:L02")
    ad3 = next(c for c in result["checks"] if c["id"] == "AD3")
    assert ad3["ok"], ad3
    assert "store" not in ad3["detail"]["shared"], "a code path keeps its package only"


def test_ad3_a_chain_marker_quoted_by_a_limit_is_the_chains_words(tmp_path):
    """A limit quotes the chain's own WARN line verbatim; every lane whose records hold the same
    warning shares its words, so they are the chain's, not the lane's content."""
    root = matrix(tmp_path, "L02")
    warned = ["WARN: blueprint declaration (unattended): served endpoint(s) are not declared "
              "(serving.api: GET /console, GET /health, GET /v1/info, POST /v1/predict; "
              "serving.showcase: GET /report.pdf) divergence extends skill"]
    seeded(root, "L10", lambda d: readings(d, "R-q")["content"]["data"].extend(warned))
    seeded_l02 = page("L02")
    readings(seeded_l02, "R-q")["content"]["data"].extend(warned)
    write_lane(root, "L02", seeded_l02)
    ad3 = next(c for c in ar.judge(root, LANES, "L10:L02")["checks"] if c["id"] == "AD3")
    assert ad3["ok"] and not {"console", "divergence", "extends"} & set(ad3["detail"]["shared"]), ad3
    assert ar.MARKER_LINE.match("WARN: x") and not ar.MARKER_LINE.match("idle: 3, jammed: 12")


def test_ad4_a_leak_fails_the_pair(tmp_path):
    root = matrix(tmp_path, "L01")
    seeded(root, "L02", lambda d: d["runs"][1]["checks"][0].update(ok=False, problems=["'BENIGN' in main > p"]))
    assert problems(root, "L02:L01", "AD4") == ["R-b: 'BENIGN' in main > p"]


def test_ad5_alert_follows_the_role_map(tmp_path):
    root = matrix(tmp_path, "L01")
    seeded(root, "L02", lambda d: readings(d)["palette"].update(alert_used=True, role_alert_elements=3))
    assert "no alert, yet the page drew alert" in problems(root, "L02:L01", "AD5")[0]
    root2 = matrix(tmp_path / "b", "L02")
    seeded(root2, "L01", lambda d: readings(d)["palette"].update(alert_used=False, alert_roles=0))
    found = problems(root2, "L01:L02", "AD5")
    assert any("holds 1" in p for p in found) and any("drew none" in p for p in found)


def test_ad6_motif_of_the_modality_and_distinct_across_modalities(tmp_path):
    root = matrix(tmp_path, "L02")
    seeded(root, "L10", lambda d: readings(d)["motif"].update(id="points", data_motif="points"))
    found = problems(root, "L10:L02", "AD6")
    assert any("not one of its modality's" in p for p in found)
    assert any("same motif" in p for p in found)
    seeded(root, "L10", lambda d: readings(d)["motif"].update(data_motif="grid"))
    assert any("the field drew 'grid'" in p for p in problems(root, "L10:L02", "AD6"))


# --------------------------------------------------------------------------- AD7 and the golden side
def golden_result(root: Path, problems=(), tolerance=8) -> None:
    box = {"header": [0, 0, 1440, 70], "nav": [0, 845.3, 1440, 900], "main": [0, 70, 1440, 845.3]}
    measured = {"hand": {"start": box}, "showcase": {"start": box}, "problems": []}
    layout = {"tolerance_px": tolerance, "problems": list(problems),
              "passes": {name: dict(measured) for name in ar.GOLDEN_PASSES}}
    (root / "lanes" / "G00" / "results" / "golden.json").write_text(json.dumps({"layout": layout}), encoding="utf-8")


def test_ad7_layout_parity_within_eight_pixels(tmp_path):
    root = matrix(tmp_path, "L02", "G00")
    assert "G00 recorded no layout parity" in problems(root, "L02:G00", "AD7")[0]
    golden_result(root)
    assert problems(root, "L02:G00", "AD7") == []
    seeded(root, "G00", lambda d: readings(d)["boxes"]["network"]["main"].__setitem__(1, 70 + 8))
    golden_result(root)
    assert problems(root, "L02:G00", "AD7") == []
    seeded(root, "G00", lambda d: readings(d)["boxes"]["network"]["main"].__setitem__(1, 70 + 9))
    assert problems(root, "L02:G00", "AD7") == ["network: main moved 9 px ([0, 70, 1440, 775] vs [0, 79, 1440, 775])"]
    seeded(root, "G00", lambda d: readings(d)["boxes"]["start"]["h1"].__setitem__(3, 194))
    assert problems(root, "L02:G00", "AD7") == [], "a headline's height across two lanes follows its words"
    seeded(root, "G00", lambda d: readings(d)["boxes"].pop("context"))
    assert any("view sets differ" in p for p in problems(root, "L02:G00", "AD7"))
    write_lane(root, "G00")
    golden_result(root, problems=["try: h1 moved 12 px"])
    assert problems(root, "L02:G00", "AD7") == ["G00 against the hand-built page: try: h1 moved 12 px"]
    golden_result(root, tolerance=12)
    assert "tolerance is 12 px" in problems(root, "L02:G00", "AD7")[0]
    golden_result(root)
    doc = json.loads((root / "lanes" / "G00" / "results" / "golden.json").read_text(encoding="utf-8"))
    doc["layout"]["passes"].pop("night-390")
    (root / "lanes" / "G00" / "results" / "golden.json").write_text(json.dumps(doc), encoding="utf-8")
    assert problems(root, "L02:G00", "AD7") == ["G00's layout parity has no night-390 pass"]


def test_a_golden_side_that_did_not_run_skips_and_a_lane_that_did_not_fails(tmp_path):
    root = matrix(tmp_path, "L02")
    skipped = ar.judge(root, LANES, "L02:G00")
    assert skipped["status"] == "skipped" and "CIDS_ROOT" in skipped["why"]
    failed = ar.judge(root, LANES, "L10:L02")
    assert failed["status"] == "fail" and "L10 has no page result" in failed["why"]
    assert ar.main(["--matrix", str(root), "--pairs", "L02:G00"]) == 0
    assert ar.main(["--matrix", str(root), "--pairs", "L10:L02"]) == 1


# --------------------------------------------------------------------------- what the page stage records
def test_the_page_stage_records_the_readings_the_report_compares():
    import page_smoke
    dom = ("<html><head><style>.x{content:'⟪no⟫'}</style><script>var s = '‹nor›';</script></head><body>"
           "<h1>⟦Start⟧ ⟪Parcels read <span class=num>6</span> bands⟫</h1><p>‹b01› and ‹cropland›</p>"
           "<p>⟪A &amp; B⟫</p></body></html>")
    probe = {"palette": {"alert_used": False}, "motif": {"id": "points"}, "missing_keys": [],
             "extensions": {"usedKeys": ["view.start"]}}
    got = page_smoke.readings({"probe": probe, "dom": dom, "lock": {"k": 1}, "boxes": {"start": {"h1": [0, 0, 1, 1]}}})
    assert got["content"] == {"story": ["A & B", "Parcels read  6  bands"], "data": ["b01", "cropland"]}
    assert (got["used_keys"], got["lock"], got["palette"], got["boxes"]) == (["view.start"], {"k": 1},
                                                                            {"alert_used": False}, {"start": {"h1": [0, 0, 1, 1]}})
    assert page_smoke.readings({"probe": probe, "dom": ""})["content"] is None, "a run that read no DOM records no content"
