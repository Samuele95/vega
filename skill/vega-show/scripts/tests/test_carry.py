#!/usr/bin/env python3
"""
test_carry.py — the carry (`carry.PlanCarrier`, `draft_plan.py --carry`).

    python3 test_carry.py            # or: python3 -m pytest test_carry.py

- **The CIDS carry** (skipped without the CIC-IDS2017 bundle): the hand-authored
  `vega/story-plan@1` keeps its headline and lead, its results headline with renamed
  placeholders, `BENIGN` neutral (a structural basis: fork S2 is still asked) and its input
  groups as explicit members partitioning the 65 inputs; its run labels and code notes land on
  the proposed runs and walk steps.
- **A story-plan@1** renames every placeholder to its fact key; a text naming a value with no
  fact, a choice on no decision and the inputs no group holds (`_unplaced`) are `WARN:`s.
- **A story-plan@2** keeps what still exists (the locale, an admissible motif, the class roles
  covering the classes, the proposed runs' labels, the groups' surviving members, the proposed
  walk steps' notes) and warns about the rest; an alert it carried without a `user:` or `show:`
  basis is downgraded by the unattended answer.
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _draft_fixtures as fx  # noqa: E402  (puts the scripts on the path)
import draft_plan  # noqa: E402
from carry import UNPLACED, PlanCarrier, rename_placeholders  # noqa: E402
from memory_stances import MemoryStances  # noqa: E402
from propose_roles import s2_open  # noqa: E402
from vega_show_core import FactSet  # noqa: E402


# --------------------------------------------------------------------------- the CIDS carry

def test_the_cids_carry_of_the_hand_authored_plan():
    fx.need_cids()
    if not fx.CIDS_PLAN_V1.is_file():
        fx.skip(f"{fx.CIDS_PLAN_V1} is absent")
    v1 = json.loads(fx.CIDS_PLAN_V1.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        facts = fx.cids_facts(Path(tmp))
        code, plan, printed = fx.draft_cli(facts, Path(tmp) / "draft.json",
                                           ["--carry", fx.CIDS_PLAN_V1])
        assert code == 0, printed
        authored = plan["authored"]
        assert authored["headline"] == v1["headline"] and authored["lead"] == v1["lead"]
        results = next(s for s in v1["sections"] if s["id"] == "results")
        renamed = results["headline"]
        for old, new in (("f1", "eval_network_f1_macro"), ("diff_tree", "eval_diff_tree"),
                         ("diff_linear", "eval_diff_linear")):
            renamed = renamed.replace("{" + old + "}", "{" + new + "}")
        assert renamed != results["headline"] and authored["results"]["headline"] == renamed
        assert "{classes_n}" in authored["context"]["steps"]["data"]
        roles = authored["class_roles"]
        assert roles["roles"]["BENIGN"] == "neutral" and "alert" not in roles["roles"].values()
        assert roles["source"]["BENIGN"] == "structural:majority"
        shaped = json.loads(facts.read_text())["groups"]["inputs"]["shaped"]
        assert not fx.partition_problems(authored["input_groups"], shaped, 3, 9)
        assert authored["experiments"]["labels"]["R7"] == "unweighted loss"
        assert authored["code_notes"]["link"].startswith("softmax")
        assert plan["carried_from"] == str(fx.CIDS_PLAN_V1)
        assert "forks open: S2, S3" in printed, "a carried structural neutral answers no fork"


# --------------------------------------------------------------------------- story-plan@1

def _v1_plan() -> dict:
    values = {"rows": "geometry:source.rows", "units": "netspec:blocks[id=B3].params.units",
              "r1": "training:experiments.runs[id=R1].primary", "gone": "shape:nowhere.at.all"}
    choices = [{"title": "Loss", "focus": "outputs", "text": "It learns."},
               {"title": "Where", "focus": "nowhere", "text": "Lost."}]
    return {"schema": "vega/story-plan@1", "values": values,
            "headline": "Read {rows} rows through {units} units.", "lead": "A lead.",
            "sections": [{"id": "network", "lead": "The {units} units.", "choices": choices},
                         {"id": "experiments", "lead": "Runs.", "caption": "Scores.",
                          "runs": [{"value": "r1", "label": "the floor"}]},
                         {"id": "results", "headline": "It did {gone}.", "facts": [],
                          "limits": []}],
            "input_groups": [{"label": "letters", "text": "greek letters",
                              "match": ["^(alpha|beta|gamma)$"]}],
            "code_notes": {"act": {"note": "the activation"}, "softmax": {"note": "the link"}}}


def test_carrying_a_story_plan_v1_renames_the_placeholders():
    with tempfile.TemporaryDirectory() as tmp:
        code, plan, printed, _ = fx.draft(Path(tmp), fx.cell("multiclass"))
        assert code == 0, printed
        facts = FactSet.from_json(json.loads((Path(tmp) / "plan" / "facts.json").read_text()))
        authored, problems = PlanCarrier(facts, _v1_plan()).carry(plan)
    assert authored["headline"] == "Read {source_rows} rows through {block_b3_units} units."
    assert authored["network"]["lead"] == "The {block_b3_units} units."
    assert authored["network"]["choices"] == [{"decision": "loss", "title": "Loss",
                                               "text": "It learns."}]
    assert authored["experiments"]["labels"] == {"R1": "the floor"}
    assert authored["results"] is None, "a headline naming a value with no fact is not carried"
    assert authored["code_notes"] == {"activation": "the activation", "link": "the link"}
    groups = {g["id"]: g["members"] for g in authored["input_groups"]}
    assert groups == {"letters": ["alpha", "beta", "gamma"], UNPLACED: ["zone"]}, groups
    text = "\n".join(map(str, problems))
    assert "gone (shape:nowhere.at.all)" in text and "{gone}" in text and "'nowhere'" in text
    assert all(p.level == "WARN" and p.rule == "carry" for p in problems)
    assert rename_placeholders("{a:pct} of {row.many}", {"a": "b"}) == ("{b:pct} of {row.many}", [])
    assert rename_placeholders("{a} of {bogus.one}", {}) == ("{a} of {bogus.one}",
                                                            ["{a}", "{bogus.one}"])


# --------------------------------------------------------------------------- story-plan@2

def _old_plan(plan: dict) -> dict:
    """`plan` authored as an earlier run would have, with references that no longer hold."""
    runs = [r["id"] for r in plan["proposals"]["runs"]]
    old = copy.deepcopy(plan)
    old["authored"].update({
        "locale": "en", "motif": "lanes", "headline": "{source_rows} {row.many}",
        "lead": "{vanished} lead",
        "class_roles": {"roles": {"no": "neutral", "yes": "alert"},
                        "source": {"no": "structural:negative", "yes": "vega:default"}},
        "experiments": {"lead": "L", "caption": "C",
                        "labels": {**{r: r.lower() for r in runs}, "R99": "x"}},
        "input_groups": [{"id": "abc", "label": "abc", "text": "t",
                          "members": ["alpha", "beta", "gamma", "omega"]}],
        "code_notes": {"root": "the root", "nowhere": "lost"}, "lexicon": [1, 2]})
    return old


def test_carrying_a_story_plan_v2_keeps_what_still_exists():
    with tempfile.TemporaryDirectory() as tmp:
        code, fresh, printed, _ = fx.draft(Path(tmp), fx.cell("binary"))
        assert code == 0, printed
        facts = FactSet.from_json(json.loads((Path(tmp) / "plan" / "facts.json").read_text()))
    authored, problems = PlanCarrier(facts, _old_plan(fresh)).carry(fresh)
    assert authored["locale"] == "en" and authored["motif"] == "lanes" and authored["lead"] is None
    assert authored["headline"] == "{source_rows} {row.many}" and authored["lexicon"] is None
    assert set(authored["experiments"]["labels"]) == {r["id"] for r in fresh["proposals"]["runs"]}
    groups = {g["id"]: g["members"] for g in authored["input_groups"]}
    assert groups == {"abc": ["alpha", "beta", "gamma"], UNPLACED: ["zone"]}
    assert authored["code_notes"] == {"root": "the root"}
    where = {p.where for p in problems}
    assert {"authored.lead", "authored.experiments.labels.R99", "authored.code_notes.nowhere",
            "authored.lexicon"} <= where, where
    template = {"version": "2.3.0", "manifest_digest": "sha256:" + "0" * 64}
    drafter = draft_plan.PlanDrafter(facts, draft_plan.DraftInputs(template["manifest_digest"],
                                                                   template), MemoryStances())
    assert s2_open(fresh["proposals"]["class_roles"], authored["class_roles"])
    settled = drafter.unattended({**fresh, "authored": authored})["authored"]["class_roles"]
    assert settled["roles"] == {"no": "neutral", "yes": "plain"}
    assert settled["source"]["yes"] == "vega:default"
    assert any("downgraded" in m and m.endswith("— yes") for m in drafter.markers), drafter.markers


if __name__ == "__main__":
    sys.exit(fx.run_tests(globals()))
