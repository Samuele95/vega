"""
test_plan_claims.py — rule V15 (plan_claims.claim_problems): what an authored plan's words claim
holds against what the facts record (gate C's review panel: a beaten verdict said "could not
separate … does as well"; a per-horizon error captioned "the primary metric"; the loss, the
optimizer and the weight decay each titled "The block"; four input groups labelled alike).

    python3 -m pytest -q tests/test_plan_claims.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plan_claims import claim_problems  # noqa: E402

TASK = SimpleNamespace(primary="mase")
DECISIONS = [{"id": "B1", "path": "blocks[id=B1]"}, {"id": "B2", "path": "blocks[id=B2]"},
             {"id": "loss", "path": "loss"}, {"id": "optimizer", "path": "optimizer"}]
BEATEN = {"earned": False, "beaten_by_baseline": True, "strongest_baseline": "linear", "diff": "eval_diff_linear"}
LEVEL = {**BEATEN, "beaten_by_baseline": False}


def groups(verdict=LEVEL, breakdown=None):
    return {"verdict": verdict, "breakdown": breakdown or {}, "decisions": DECISIONS}


def plan(**parts):
    base = {"results": {"limits": [], "breakdown_caption": None},
            "network": {"choices": [{"decision": "B1", "title": "The dense layer", "text": "It mixes."},
                                    {"decision": "loss", "title": "The loss", "text": "What it gets right."}]},
            "input_groups": [{"id": "a", "label": "b01"}, {"id": "b", "label": "b02"}]}
    base.update(parts)
    return base


def rules(problems):
    return [(p.rule, p.where) for p in problems]


def test_a_plan_whose_words_hold_passes():
    assert claim_problems(plan(), groups(), TASK) == []


def test_a_beaten_verdict_is_never_softened():
    soft = {"id": "verdict_not_earned", "title": "Not earned",
            "text": "The evaluation could not separate the network from it: a cheaper model does as well."}
    results = {"limits": [soft], "breakdown_caption": None}
    assert rules(claim_problems(plan(results=results), groups(BEATEN), TASK)) == \
        [("V15", "authored.results.limits[0]")]
    assert claim_problems(plan(results=results), groups(LEVEL), TASK) == [], \
        "an interval that holds zero may be said to not separate them"
    said = {"id": "verdict_beaten", "title": "A baseline did better",
            "text": "A baseline scored better: the network minus it is {eval_diff_linear}."}
    assert claim_problems(plan(results={"limits": [said]}), groups(BEATEN), TASK) == []


def test_a_breakdown_of_another_metric_is_not_the_primary():
    results = {"limits": [], "breakdown_caption": "The primary metric unit by unit on the same test part."}
    got = claim_problems(plan(results=results), groups(breakdown={"metric": "per_horizon_error"}), TASK)
    assert rules(got) == [("V15", "authored.results.breakdown_caption")]
    assert claim_problems(plan(results=results), groups(breakdown={"metric": "mase"}), TASK) == []


def test_choices_are_told_apart_and_only_a_block_is_a_block():
    twice = {"choices": [{"decision": "B1", "title": "The dense layer", "text": "x"},
                         {"decision": "B2", "title": "The dense layer", "text": "y"}]}
    assert rules(claim_problems(plan(network=twice), groups(), TASK)) == \
        [("V15", "authored.network.choices[1].title")]
    loss = {"choices": [{"decision": "loss", "title": "The block", "text": "It is a block of the network."},
                        {"decision": "B1", "title": "The dense layer", "text": "It is a block of the network."}]}
    assert rules(claim_problems(plan(network=loss), groups(), TASK)) == [("V15", "authored.network.choices[0]")]


def test_input_groups_are_labelled_apart():
    alike = [{"id": "a", "label": "the remaining inputs"}, {"id": "b", "label": "the remaining inputs"}]
    assert rules(claim_problems(plan(input_groups=alike), groups(), TASK)) == \
        [("V15", "authored.input_groups[1].label")]


def test_class_display_names_name_the_tasks_classes_once_and_in_words():
    task = SimpleNamespace(primary="pr_auc", classes=("0", "1"))
    assert claim_problems(plan(class_names={"0": "on time", "1": "late"}), groups(), task) == []
    bad = claim_problems(plan(class_names={"0": "on time", "2": "late", "1": "on time"}), groups(), task)
    assert [p.where for p in bad] == ["authored.class_names.2", "authored.class_names.1"], bad
    assert claim_problems(plan(class_names={"1": "late by 2 days"}), groups(), task)[0].message.endswith("by words")
