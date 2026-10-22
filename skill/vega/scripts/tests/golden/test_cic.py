#!/usr/bin/env python3
"""
test_cic.py — the golden CIC lane's parts (BUILD-PLAN R2, D6 §11.5), held without running it.

    python3 -m pytest -q golden/test_cic.py     (CIDS_ROOT set adds the checks on the real files)

- **R-G2 amendment** (`cic_migrate.amend_blueprint`): every port-owning module becomes `OOP`, a
  module with no port keeps its paradigm, every port gets its role's v2.3 `signature[]`, every
  artifact `EXTERNAL` names is declared external with its owner (one already declared, and one
  the package holds, stay as drawn), nothing else of the base changes, the amendment is
  recorded with the base's digest; a port of a role no v2.3 interface carries is refused.
- **The stored predictions re-encoded** (`reencode_predictions`): `row_id` becomes the shaper's
  raw position of the part's row, the class logits stay the head (read back through
  vega-eval's own reader as `vega/predictions-layout@1`, multiclass, raw positions), v2.2's
  `score` copy of the first logit is dropped, the baselines stay; a file already in a layout,
  a `score` that is not the first logit or a part of another length is refused.
- **Read-only**: a file created, changed or removed under a watched path is named.
- **The report roles** (`cic-report-roles.json`): every role is a vega-show catalog role; with
  CIDS_ROOT, the role map written from the real `.aux` is one vega-show's own reader accepts.
- **The authoring**: the bundle layer carries the v2.2 intended use verbatim and states no
  number; the overlay keeps the lane program's slots under carried ones, item by item; the
  `_unplaced` inputs join the group the proposal gives them; the carry edits apply once and
  are listed.
- **Fact recovery**: a value equal at the rounding it was written at passes, another fails; a
  ref no fact carries is `unread`, never a pass.
- **AD7**: boxes within 8 px pass, 9 px fail, a box drawn on one page only fails, a view absent
  from one page fails.
- With CIDS_ROOT: the story-plan fixture is the hand-written plan, byte for byte, and every
  path G00 records exists.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
SKILLS = TESTS.parents[2]
sys.path[:0] = [str(HERE), str(TESTS), str(SKILLS / "vega-eval" / "scripts"),
                str(SKILLS / "vega-show" / "scripts")]

import cic  # noqa: E402
import cic_authoring as authoring  # noqa: E402
import cic_migrate as migrate  # noqa: E402
import cic_parity as parity  # noqa: E402
import predictions_layout  # noqa: E402
from lanekit.pipeline import port_signatures  # noqa: E402

CIDS = Path(os.environ.get("CIDS_ROOT", "/nonexistent"))
FIXTURE = SKILLS / cic.STORY_PLAN_FIXTURE


def _need_cids():
    if not (CIDS / "vega-runs").is_dir():
        pytest.skip("CIDS_ROOT names no CIC-IDS2017 project")


# --------------------------------------------------------------------------- R-G2
BASE = {"schema": "blueprint/pipeline@1", "bets": [{"id": "bet0"}],
        "ports": [{"name": "Corpus", "role": "DataSource"}, {"name": "Net", "role": "Model"},
                  {"name": "Watch", "role": "Monitor", "signature": ["def old(self): ..."]}],
        "modules": [{"name": "contracts", "paradigm": "FP", "ports_owned": []},
                    {"name": "nets", "paradigm": "FP", "ports_owned": ["Net"]},
                    {"name": "ingestion", "paradigm": "OOP", "ports_owned": ["Corpus"]},
                    {"name": "monitoring", "paradigm": "FP", "ports_owned": ["Watch"]}],
        "artifacts": [{"name": "NetSpec", "kind": "value"},
                      {"name": "Campaign", "kind": "entity", "owner": "vega-train tooling"},
                      {"name": "TestSeal", "kind": "entity",
                       "external": {"owner": "/vega-eval", "where": "as drawn"}}],
        "amends": {"run": "earlier"}}


def test_the_amendment_applies_r_g2_and_nothing_else(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "pipeline.json").write_text(json.dumps(BASE), encoding="utf-8")
    (base / "blueprint.md").write_text("# Blueprint\n", encoding="utf-8")
    out = migrate.amend_blueprint(base, tmp_path / "out")
    doc = json.loads(out.read_text(encoding="utf-8"))
    paradigms = {m["name"]: m["paradigm"] for m in doc["modules"]}
    assert paradigms == {"contracts": "FP", "nets": "OOP", "ingestion": "OOP", "monitoring": "OOP"}
    want = port_signatures(*migrate.GOLDEN_PACKAGE)
    assert all(p["signature"] == want[p["role"]] for p in doc["ports"])
    arts = {a["name"]: a for a in doc["artifacts"]}
    assert arts["Campaign"]["external"] == {"owner": "/vega-train",
                                            "where": migrate.EXTERNAL["Campaign"][1]}
    assert arts["Campaign"]["owner"] == "vega-train tooling"       # the base's own text, kept
    assert arts["NetSpec"] == BASE["artifacts"][0] and arts["TestSeal"] == BASE["artifacts"][2]
    keep = ("ports", "modules", "artifacts", "amends")
    assert {k: v for k, v in doc.items() if k not in keep} == \
        {k: v for k, v in BASE.items() if k not in keep}
    amends = doc["amends"]
    assert amends["run"] == "base" and amends["previous"] == {"run": "earlier"}
    assert amends["base_digest"].startswith("sha256:") and len(amends["changes"]) == 7
    assert "artifact Campaign: external, held by /vega-train" in amends["changes"]
    # the Monitor's adapter is /vega-pack's to fill, and the port says so (T5r3)
    assert "port Watch (Monitor): filled_by /vega-pack" in amends["changes"]
    assert next(p for p in doc["ports"] if p["name"] == "Watch")["filled_by"] == "/vega-pack"
    assert "## Amendment (R-G2)" in (tmp_path / "out" / "blueprint.md").read_text(encoding="utf-8")
    assert json.loads((base / "pipeline.json").read_text(encoding="utf-8")) == BASE


def test_a_port_of_an_unknown_role_is_refused(tmp_path):
    (tmp_path / "pipeline.json").write_text(json.dumps(
        {"ports": [{"name": "X", "role": "Oracle"}], "modules": []}), encoding="utf-8")
    with pytest.raises(migrate.MigrationError, match="Oracle"):
        migrate.amend_blueprint(tmp_path, tmp_path / "out")


# --------------------------------------------------------------------------- predictions
def _v22(path: Path, n=6, k=3, score_first=True):
    rng = np.random.default_rng(1)
    head = rng.normal(size=(n, k)).astype("float32")
    frame = pd.DataFrame({"row_id": np.arange(n)[::-1].copy(), "y": rng.integers(0, k, n),
                          "score": head[:, 0] if score_first else head.max(axis=1) + 1})
    for i in range(k):
        frame[f"score_{i}"] = head[:, i]
    frame["score_trivial"] = rng.normal(size=n)
    frame["score_tree"] = rng.normal(size=n)
    frame.to_parquet(path, index=False)
    return frame


def test_the_stored_predictions_are_re_encoded_not_recomputed(tmp_path):
    src, dst = tmp_path / "v22.parquet", tmp_path / "out" / "predictions.test.parquet"
    old = _v22(src)
    positions = [40, 11, 7, 93, 2, 58]
    record = migrate.reencode_predictions(src, dst, positions, predictions_layout)
    frame, meta = predictions_layout.read(dst)
    assert meta["vega.predictions_layout"] == migrate.LAYOUT
    assert (meta["vega.row_id_kind"], meta["vega.task_effective"], meta["vega.score_shape"]) == \
        ("raw_position", "multiclass", "[3]")
    assert list(frame["row_id"]) == [positions[i] for i in old["row_id"]]
    assert np.array_equal(frame[["score_0", "score_1", "score_2"]].to_numpy(),
                          old[["score_0", "score_1", "score_2"]].to_numpy())
    assert "score" not in frame.columns and list(frame["y"]) == list(old["y"])
    assert np.array_equal(frame["score_tree"].to_numpy(), old["score_tree"].to_numpy())
    assert record["dropped"] == ["score"] and record["baselines"] == ["trivial", "tree"]
    assert record["source_digest"] == migrate.digest(src) and record["rows"] == 6


def test_a_layout_file_another_score_or_another_part_is_refused(tmp_path):
    src = tmp_path / "v22.parquet"
    _v22(src)
    migrate.reencode_predictions(src, tmp_path / "a.parquet", list(range(6)), predictions_layout)
    with pytest.raises(migrate.MigrationError, match="already names a layout"):
        migrate.reencode_predictions(tmp_path / "a.parquet", tmp_path / "b.parquet", list(range(6)),
                                     predictions_layout)
    with pytest.raises(migrate.MigrationError, match="not the same part"):
        migrate.reencode_predictions(src, tmp_path / "c.parquet", list(range(5)), predictions_layout)
    _v22(src, score_first=False)
    with pytest.raises(migrate.MigrationError, match="first class logit"):
        migrate.reencode_predictions(src, tmp_path / "d.parquet", list(range(6)), predictions_layout)


def test_a_change_under_a_watched_path_is_named(tmp_path):
    (tmp_path / "a").write_text("x", encoding="utf-8")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b").write_text("y", encoding="utf-8")
    snap = migrate.readonly_snapshot([tmp_path / "a", tmp_path / "d"])
    assert migrate.readonly_changes(snap, [tmp_path / "a", tmp_path / "d"]) == []
    (tmp_path / "d" / "c").write_text("z", encoding="utf-8")
    (tmp_path / "a").write_text("xx", encoding="utf-8")
    assert migrate.readonly_changes(snap, [tmp_path / "a", tmp_path / "d"]) == \
        sorted([str(tmp_path / "a"), str(tmp_path / "d" / "c")])


# --------------------------------------------------------------------------- the report roles
def test_every_bound_role_is_a_catalog_role():
    from vega_show_core import CATALOG
    spec = json.loads(authoring.REPORT_ROLES_FILE.read_text(encoding="utf-8"))
    assert set(spec["roles"]) <= set(CATALOG["report_roles"])
    assert len(set(spec["roles"].values())) == len(spec["roles"]), "one label per role"
    assert spec["report"] == {"tex": "docs/main.tex", "pdf": "docs/main.pdf", "aux": "docs/main.aux"}


def test_the_real_report_gives_a_role_map_vega_show_accepts(tmp_path):
    _need_cids()
    from role_map import read_role_map
    doc = authoring.label_roles(CIDS, tmp_path / "docs", authoring.REPORT_ROLES_FILE,
                                SKILLS / "vega-show" / "scripts")
    roles, problems = read_role_map(tmp_path / "docs" / "label-roles.json")
    assert problems == [] and roles is not None, problems
    assert doc["roles"]["net.forward"]["label"] == "eq:mlp"
    assert doc["roles"]["chapter.blueprint"]["number"] == "3.3"


# --------------------------------------------------------------------------- the authoring
def test_the_bundle_layer_carries_the_intended_use_and_states_no_number(tmp_path):
    old = tmp_path / "bundle-layer.json"
    old.write_text(json.dumps({"readings": {"intended_use": "  Use it on flows.\n\nNot elsewhere. "}}),
                   encoding="utf-8")
    doc = authoring.pack_layer(tmp_path, old, "bundle/x-R7")
    assert doc["readings"]["intended_use"] == "Use it on flows.\n\nNot elsewhere."
    assert doc["from_bundle"] == "bundle/x-R7" and authoring.digit_problems(doc, "pack") == []
    old.write_text(json.dumps({"readings": {"intended_use": "Use it on 3 flows."}}), encoding="utf-8")
    with pytest.raises(ValueError, match="number"):
        authoring.pack_layer(tmp_path, old, ".")
    old.write_text(json.dumps({"readings": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="intended_use"):
        authoring.pack_layer(tmp_path, old, ".")


def test_the_overlay_keeps_the_lane_programs_slots_under_the_carried_ones():
    base = {"lead": "L", "labels": {"R1": "lane", "H1-5": "trial"}, "steps": {"data": "d"}}
    top = {"lead": "carried", "labels": {"R1": "unweighted"}, "steps": None}
    assert authoring._over(base, top) == {"lead": "carried", "labels": {"R1": "unweighted",
                                                                        "H1-5": "trial"},
                                          "steps": {"data": "d"}}
    assert authoring._over({"a": 1}, ["x"]) == ["x"]


def test_the_unplaced_inputs_join_the_group_the_proposal_gives_them():
    groups = [{"id": "rates", "label": "rates", "text": "t", "members": ["a"]},
              {"id": authoring.UNPLACED, "members": ["b", "z"]}]
    proposed = [{"suggested_id": "rates", "members": ["a", "b"], "evidence": []},
                {"suggested_id": "flags", "members": ["z"], "evidence": [{"kind": "prefix"}]}]
    out = authoring._placed(groups, proposed)
    assert [g["id"] for g in out] == ["rates", "flags"]
    assert out[0]["members"] == ["a", "b"] and out[1]["members"] == ["z"]
    assert authoring._placed(groups[:1], proposed) == groups[:1]


def test_the_carry_edits_apply_once_and_are_listed():
    authored = {"lead": "Trained on CIC-IDS2017, compared.", "context": {"steps": {
        "data": "CIC-IDS2017 records days."}}, "results": {"headline": "nothing to edit"}}
    made = authoring._edit(authored)
    assert len(made) == 2 and "CIC-IDS2017" not in json.dumps(authored)
    assert authoring._edit(authored) == []


# --------------------------------------------------------------------------- fact recovery
def test_a_value_equal_at_its_rounding_is_recovered_and_another_is_not():
    assert cic._same(0.8164, 0.816412) and not cic._same(0.8164, 0.8166)
    assert cic._same({"value": 0.8164, "ci": [0.7964, 0.8256], "n": 504473},
                     {"value": 0.81641, "ci": [0.79641, 0.82559], "n": 504473})
    assert cic._same(24, 24) and cic._same("3.3", "3.3") and not cic._same(24, 25)
    assert cic._same(["a", "b"], ["a", "b"]) and not cic._same(["a"], ["a", "b"])


def test_fact_recovery_compares_what_the_extractor_reads():
    plan = {"values": {"rows": "geometry:source.rows", "f1": "evaluation:metrics.models.network.f1_macro",
                       "gone": "netspec:nowhere.x", "bad": "netspec:blocks[id=B1].params.units"}}
    story = {"values": {"rows": 10, "f1": {"value": 0.5, "ci": [0.4, 0.6], "n": 3}, "bad": 64}}
    facts = {"facts": {"source_rows": {"ref": "records/geometry.json#source.rows", "value": 10},
                       "eval_f1": {"ref": "records/evaluation.json#metrics.models.network.f1_macro",
                                   "value": {"value": 0.50001, "ci": [0.4, 0.6], "n": 3}},
                       "units": {"ref": "netspec.json#blocks[id=B1].params.units", "value": 128}}}
    record = cic.fact_recovery(plan, story, facts, SKILLS / "vega-show" / "scripts")
    assert [e["name"] for e in record["equal"]] == ["rows", "f1"]
    assert [e["name"] for e in record["differ"]] == ["bad"]
    assert [e["name"] for e in record["unread"]] == ["gone"]


# --------------------------------------------------------------------------- AD7
def _reading(dx=0, drop=None, active=True):
    view = {"active": active, "header": [0, 0, 1440, 70], "nav": [0, 845, 1440, 900],
            "main": [0, 70, 1440, 845], "h1": [120, 181 + dx, 1128, 321], "canvas": None}
    if drop:
        view[drop] = None
    return {v: dict(view) for v in parity.VIEWS}


def test_layout_parity_holds_within_eight_pixels_and_no_further():
    assert parity.layout_problems(_reading(), _reading(dx=8)) == []
    problems = parity.layout_problems(_reading(), _reading(dx=9))
    assert len(problems) == len(parity.VIEWS) and "off by 9 px" in problems[0]
    assert any("drawn on the hand-built page only" in p
               for p in parity.layout_problems(_reading(), _reading(drop="h1")))
    hand = _reading()
    hand["many"]["active"] = False
    assert parity.layout_problems(hand, _reading()) == [
        "many: the view is absent on the hand-built page and shown on the showcase"]


def test_layout_parity_reads_each_state_and_its_extra_boxes():
    hand, built = _reading(), _reading()
    for doc in (hand, built):
        doc["network:decision-1"] = {**doc["network"], "extra": {"#choices": [120, 300, 700]}}
    assert parity.layout_problems(hand, built) == []
    built["network:decision-1"]["extra"]["#choices"] = [120, 360, 700]
    assert parity.layout_problems(hand, built) == [], "an act's box is held to its edges, not its place"
    built["network:decision-1"]["extra"]["#choices"] = [140, 300, 700]
    assert parity.layout_problems(hand, built) == [
        "network:decision-1 #choices: [140, 300, 700] against the hand-built [120, 300, 700] (off by 20 px > 8)"]
    built["network:decision-1"]["extra"]["#choices"] = None
    assert parity.layout_problems(hand, built) == ["network:decision-1 #choices: drawn on the hand-built page only"]
    built["network:decision-7"] = dict(built["network:decision-1"])
    del built["network:decision-1"]
    assert parity.layout_problems(hand, built) == [], "a state one page alone reaches is content, not layout"
    assert {p[0] for p in parity.PASSES} == {"night-1440", "day-1440", "night-390"}


def test_the_passes_beyond_night_hold_the_frame_whole_and_the_rest_by_its_edges():
    hand, built = _reading(), _reading(dx=30)
    assert parity.layout_problems(hand, built, frame_only=True) == [], "an h1 lower by the words above it"
    assert len(parity.layout_problems(hand, built)) == len(parity.VIEWS), "night at 1440 holds every box whole"
    wide = _reading()
    for view in wide.values():
        view["h1"] = [120, 181, 1150, 321]
    assert any("h1" in p for p in parity.layout_problems(hand, wide, frame_only=True))
    moved = _reading()
    for view in moved.values():
        view["header"] = [0, 0, 1440, 90]
    assert any("header" in p for p in parity.layout_problems(hand, moved, frame_only=True))


def test_content_parity_names_every_lost_word_and_lets_an_allowed_change_through():
    hand = {"limits": ["in distribution only", "rare classes are anecdotes"], "facts": ["accuracy", "test flows"],
            "choices": ["a little dropout"],
            "code_notes": {"dropout": "§3.4 · equation (2) and figure 19", "root": "§3.3 · figure 15"}}
    built = {"limits": ["in distribution only"], "facts": ["accuracy", "flows in the sealed test"],
             "choices": ["a little dropout"],
             "code_notes": {"dropout": "§3.4 the network · table 4", "root": "§3.3 system · figure 15"}}
    lost, allowed = parity.content_problems(hand, built, {"facts": {"test flows": "said as 'flows in the sealed test'"}})
    assert lost == ["limits: 'rare classes are anecdotes' is no longer said",
                    "code note 'dropout': 'equation (2) and figure 19' is not what the showcase names "
                    "('§3.4 the network · table 4')"], lost
    assert allowed == ["facts: 'test flows' — said as 'flows in the sealed test'"]
    # a hand label naming two report objects, or qualifying one, is met by a showcase ref naming one
    two = {"code_notes": {"dropout": "§3.4 · equation (2) and figure 19", "forward": "§3.4 · equation (2) as a whole",
                          "route": "§3.3 · “from a raw flow to an answer”"}}
    one = {"code_notes": {"dropout": "in the report: §3.4 · figure 19 — dropout", "forward": "§3.4 · equation (2) — x",
                          "route": "in the report: figure 18 — what a request carries"}}
    lost, _ = parity.content_problems(two, one, {})
    assert lost == ["code note 'route': '“from a raw flow to an answer”' is not what the showcase names "
                    "('in the report: figure 18 — what a request carries')"], lost
    assert parity.content_problems(two, one, {"code_notes": {"route": "cited by its figure"}}) == \
        ([], ["code note 'route': '“from a raw flow to an answer”' — cited by its figure"])


# --------------------------------------------------------------------------- the real inputs
def test_the_fixture_is_the_hand_written_plan_and_the_recorded_paths_exist():
    _need_cids()
    lanes = json.loads((TESTS / "lanes.json").read_text(encoding="utf-8"))
    rec = next(l for l in lanes["lanes"] if l["id"] == "G00")["recorded"]
    assert all((CIDS / v).exists() for k, v in rec.items() if k != "run"), rec
    assert FIXTURE.read_bytes() == (CIDS / rec["story_plan"]).read_bytes()
    seal = json.loads((CIDS / rec["campaign"] / "test_seal.json").read_text(encoding="utf-8"))
    assert seal["run"] == rec["run"] and seal["test"]["path"].startswith(rec["shaped"] + "/")
