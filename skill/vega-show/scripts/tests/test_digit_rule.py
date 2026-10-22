#!/usr/bin/env python3
"""
test_digit_rule.py — the digit rule and the placeholder grammar of `digit_rule.py`.

    python3 test_digit_rule.py            # or: python3 -m pytest test_digit_rule.py

The cases of D5 §12.3 and D6 §14.4, as build plan R-E10 settles them, and the logic that
catches D5's seeded defects 1–3 (gate V2):

- names from `facts.names` pass (`IDS2017`, `R7`, `F1`); bare numbers are refused (`256`,
  `0.82`, `1e-3`), and so are the Arabic-Indic `٣` and the full-width `３`;
- `β2` / `α2` are refused unless `facts.names` carries them — a digit after a Greek letter is
  not a name by position, only by the records;
- a placeholder beside text (`{x}%`) passes; lexicon paths (`{row.one}`, `{source.the}`) and
  format suffixes (`{key:pct}`) are placeholders of the grammar; unknown keys, unknown lexicon
  keys or paths, a suffix on a lexicon path and any other brace group without a space that the
  grammar rejects (`{Units}`, `{x.y.z}`, `{a:int:pct}`, `{éx}`) are V3;
- every V2 refusal names the digit rule (`digit rule: a bare number "12"`), as D6's VS01 greps;
- number words warn (never refuse), naming the facts whose value spells the word.
"""
from __future__ import annotations

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))

from digit_rule import (PLACEHOLDER_RE, DigitRule, check_placeholders, check_prose,  # noqa: E402
                        placeholders, strip_placeholders)
from vega_show_core import CATALOG, Fact, FactKind, FactSet, TaskShape  # noqa: E402

NAMES = ["IDS2017", "R7", "B1", "F1", "CIC-IDS2017", "DoS Hulk", "Destination Port", "f1_macro"]


def _rule(extra=()):
    return DigitRule([*NAMES, *extra])


def _refused(text, rule=None):
    return [str(p) for p in (rule or _rule()).refusals(text, "t")]


def _facts():
    task = TaskShape(kind="multiclass", goal="predict", sample_shape="row", family="mlp-plain",
                     view_key="classes", classes=("BENIGN", "Bot"), primary="f1_macro",
                     direction="max")
    facts = (Fact("classes_n", 15, "netspec.json#task.classes", FactKind.COUNT_OF, "count"),
             Fact("runs_n", 12, "records/training.json#experiments.runs", FactKind.COUNT_OF, "count"),
             Fact("train_share_bot", 0.25, "netspec.json#data_contract.class_balance.train.Bot",
                  FactKind.METRIC, "metric"),
             Fact("block_b1_units", 256, "netspec.json#blocks[id=B1].params.units", FactKind.PARAM, "param"))
    return FactSet(facts, {"sizes": {}}, task, frozenset([*NAMES, "BENIGN", "Bot"]))


def test_names_the_records_carry_pass():
    for text in ("IDS2017 flows", "run R7 wins", "the F1 of B1", "CIC-IDS2017", "ids2017", "the r7 run"):
        assert _refused(text) == [], text


def test_bare_numbers_are_refused():
    for text in ("256", "0.82", "1e-3", "a width of 256 units", "the top 5"):
        assert _refused(text), text
    assert len(_refused("0.82")) == 2, "each digit run of 0.82 is a bare number"


def test_unicode_digits_are_digits():
    for text in ("٣ classes", "３ classes", "۴", "class ٣"):
        refused = _refused(text)
        assert refused and "bare number" in refused[0], (text, refused)


def test_a_digit_after_a_greek_letter_is_refused_unless_a_name():
    for token in ("β2", "α2", "Ж3", "é5"):
        assert _refused(f"the {token} term"), token
        assert _refused(f"the {token} term", _rule([token])) == [], token
    assert _refused("the Α2 term", _rule(["α2"])) == [], "names compare case-insensitively"


def test_subscripts_and_superscripts_are_notation():
    assert _refused("z = w₁·{v}₁ + … + b and σ² over the batch") == []


def test_placeholders_are_deleted_first():
    assert _refused("{x}%") == []
    assert _refused("a width of {block_b1_units} units, {eval_network_f1_macro:pct} macro") == []
    assert _refused("{row.one} of {source.the}") == []
    assert strip_placeholders("B{n}7").split() == ["B", "7"], "a placeholder never fuses words"
    assert _refused("B{n}7"), "a digit after a deleted placeholder is still a bare number"


def test_seeded_defects_one_to_three():
    # D5-1: a digit in the headline
    assert _refused("The network reads 12 bands")
    # D5-2: "top7", not a name, in a choice; the catalog's own token top5 is one
    assert _refused("the top7 classes")
    assert DigitRule.from_catalog().refusals("the top5 classes", "t") == []
    # D5-3: a digit after a Greek letter
    assert _refused("the β2 decay of the optimiser")


def test_joined_tokens():
    assert _refused("an IDS2017-style split") == [], "every digit-bearing part is a name"
    assert _refused("the top-5 classes"), "a bare-number part refuses the token"
    assert _refused("a top7-style split"), "a part that is not a name refuses the token"
    assert _refused("R7-R9 moved", _rule(["R9"])) == []


def test_digits_inside_urls_and_underscores():
    assert _refused("see https://example.org/v2/page3 for more")
    assert _refused("the col_1 column"), "an underscore splits tokens: 1 is a bare number"


def test_problem_format():
    problems = _rule().refusals("the network reads 12 bands", "authored.headline")
    assert len(problems) == 1
    text = str(problems[0])
    assert text.startswith('ERROR: V2 authored.headline: digit rule: a bare number "12"'), text
    assert "digit" in text and "headline" in text, "D6 VS01 greps the resolver's line for both"
    assert problems[0].to_json()["rule"] == "V2"
    for token in ("top7", "٣", "β2"):
        refused = _rule().refusals(f"the {token} term", "t")
        assert refused and all(p.message.startswith("digit rule: ") for p in refused), refused


def test_number_words_warn_and_suggest_facts():
    rule = DigitRule.for_facts(_facts())
    assert rule.refusals("fifteen classes and twelve runs", "t") == []
    warns = [str(w) for w in rule.warnings("Fifteen classes, twelve runs, half the rows", "lead")]
    assert len(warns) == 3, warns
    assert all(w.startswith("WARN: V2 lead: number word") and '" in lead — prefer a count fact' in w
               for w in warns), warns
    assert "(classes_n)" in warns[0] and "(runs_n)" in warns[1] and "(" not in warns[2], warns
    compound = [str(w) for w in rule.warnings("two hundred and fifty-six units", "t")]
    assert any('"two hundred"' in w for w in compound) and any("fifty-six" in w for w in compound), compound
    assert DigitRule.for_facts(_facts()).check("one of the fifteen", "t")[0].level == "WARN"
    assert rule.warnings("someone often weighs the tone", "t") == [], "whole words only"


def test_for_facts_uses_the_fact_names():
    rule = DigitRule.for_facts(_facts())
    assert rule.is_name("dos") and rule.is_name("ids2017") and not rule.is_name("top7")


def test_placeholder_grammar():
    assert PLACEHOLDER_RE.pattern == CATALOG["placeholder"]["pattern"]
    found = placeholders("{a} {b:pct} {row.one} {target.many} {c_1:count}")
    assert [(p.name, p.path, p.suffix) for p in found] == [
        ("a", None, None), ("b", None, "pct"), ("row", "one", None), ("target", "many", None),
        ("c_1", None, "count")]
    assert [p.is_lexicon_path for p in found] == [False, False, True, True, False]
    for suffix in CATALOG["placeholder"]["suffixes"]:
        assert placeholders(f"{{k:{suffix}}}")[0].suffix == suffix


def test_placeholders_checked_against_facts_and_the_lexicon():
    known = {"classes_n", "block_b1_units", "p_1"}
    good = "{classes_n} classes, {block_b1_units:int} units, {p_1}, {row.a} {source.the} {target.one}"
    assert check_placeholders(good, "t", known) == []
    cases = {"{runs_n}": "not a known fact key", "{colour.one}": "not a lexicon key",
             "{row.first}": "not a lexicon path", "{row.one:int}": "takes no format suffix",
             "{Units}": "not a placeholder of the grammar", "{classes_n:float}": "not a placeholder",
             "{x.y.z}": "not a placeholder of the grammar", "{a:int:pct}": "not a placeholder",
             "{éx}": "not a placeholder", "{p_1:}": "not a placeholder", "{.one}": "not a placeholder"}
    for text, why in cases.items():
        problems = check_placeholders(text, "t", known)
        assert len(problems) == 1 and why in problems[0].message, (text, problems)
        assert problems[0].rule == "V3"
    assert check_placeholders("{anything}", "t") == [], "known=None accepts any bare name"
    assert check_placeholders("{rows, at} and {}", "t", known) == [], "not placeholder-shaped"
    probe = check_placeholders("{row.one:int} {Row} {row.One} {x.y.z} {a:int:pct}", "w", {"x"})
    groups = [p.message.split()[0].rstrip(":") for p in probe]
    assert groups == ["{row.one:int}", "{Row}", "{row.One}", "{x.y.z}", "{a:int:pct}"], probe
    assert len(check_placeholders("{{classes_n}}", "t", known)) == 0, "the inner group is read"


def test_check_prose_for_keyed_strings():
    assert check_prose("The {units} units of {row.one}", known={"units"}) == []
    errors = check_prose("Takes 3 inputs, {kernel:float} wide", known={"kernel"}, where="explain.dense")
    assert len(errors) == 2 and all(e.startswith("ERROR: V") for e in errors), errors
    assert check_prose("an F1 and a ReLU6") == [], "the catalog's name tokens are names"
    assert check_prose("an R7 run") != [] and check_prose("an R7 run", names=["R7"]) == []


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
