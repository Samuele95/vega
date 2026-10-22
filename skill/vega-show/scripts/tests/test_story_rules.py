#!/usr/bin/env python3
"""
test_story_rules.py — the story plan's validation (`plan_rules.py`, V1–V15), its resolution
(`resolve_story.py`, `vega/story@2`) and the mechanical author (`autoauthor.py`).

    python3 test_story_rules.py            # or: python3 -m pytest test_story_rules.py

- **D5's seeded defects 1–21**, each refused through the command line with its rule id and no
  other (`ERROR: V<n> …`, exit 2, nothing written); D6's VS01–VS14 and VS23 and D4's PN5, PN6 and
  PN12, each with its rule and the words its case names.
- **Every admitted cell** (the vega-forge catalog's admission rows) runs extract → draft →
  autoauthor → resolve into a story that validates `story-2.schema.json`, whose `values` are
  exactly what its fields reference (with their formats and refs), `values.primary` and
  `ci_level` bound to the evaluation, and `--check` byte-equal.
- **The blocks**: input groups (coverage, the 3–9 bound from nine inputs and 1–n under it,
  `_unplaced` refused and completed by the author); `focus@1` (block, family, feature, and a
  decision with none refused); the class-role rule; the lexicon's required keys per shape.
- **Computed and bound values**: `weak` in both directions, and the bar clause; the reserved
  `primary` and `ci_level`, and a bundle without `ci_level` (`WARN:`); `resolve --check`.
- **The report**: roles proposed from a hand-authored report and confirmed, bound against its
  `.aux` (number and page made values with `report:` refs); an emitted role map's facts; brief
  links without a report; V1 against `page_template.digests()`.
- **The CIDS carry** (skipped without the CIC-IDS2017 bundle): story-plan@1 carried, completed by
  the author and resolved: the carried headline, `BENIGN` neutral, `Destination Port` in focus.
"""
from __future__ import annotations

import atexit
import contextlib
import copy
import functools
import io
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _draft_fixtures as fx  # noqa: E402  (puts the scripts on the path)
import autoauthor  # noqa: E402
import coverage_fixtures as cf  # noqa: E402
import page_template  # noqa: E402
import resolve_story  # noqa: E402
from digit_rule import placeholders  # noqa: E402
from fact_sink import Catalogs  # noqa: E402
from page_config import metric_record  # noqa: E402
from page_derive import lane_record  # noqa: E402
from plan_rules import (check_class_roles, check_input_groups, check_lexicon,  # noqa: E402
                        focus_of, group_bounds)
from plan_texts import STORY_SCHEMA  # noqa: E402
from story_sections import estimate, weak_flags  # noqa: E402
from vega_show_core import CATALOG, FactSet, TaskShape, digest_file  # noqa: E402

EXTRA_INPUTS = ("delta", "epsilon", "eta", "theta", "iota", "kappa", "lambda", "sigma")
MULTICLASS = {**fx.ROW, "task": "multiclass"}
_ERRORS = re.compile(r"^ERROR: (V\d+) ", re.M)


# --------------------------------------------------------------------------- fixtures

@dataclass
class Lane:
    """One lane taken through extract → draft → autoauthor: its files and the authored plan."""

    root: Path
    facts: Path
    plan_path: Path
    plan: dict

    @property
    def facts_doc(self) -> dict:
        return json.loads(self.facts.read_text(encoding="utf-8"))


def run(main, argv) -> tuple[int, str]:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        code = main([str(a) for a in argv])
    return code, printed.getvalue()


def widen(docs: dict, names=EXTRA_INPUTS) -> dict:
    """`docs` with more numeric inputs, so the 3–9 group bound applies (nine inputs or more)."""
    spec = docs["netspec.json"]["data_contract"]
    spec["features"] += [{"name": n, "role": "numeric", "origin": "S3"} for n in names]
    spec["input_dim"] = len(spec["features"])
    docs["schema.json"]["input"] += [{"name": n} for n in names]
    docs["records/geometry.json"]["typing"] += [{"column": n, "final_type": "numeric"}
                                               for n in names]
    return docs


def per_unit(docs: dict, metric: str, network: list, trivial: list | None) -> dict:
    """`docs` with the network's and the trivial baseline's per-unit values of `metric`."""
    models = docs["records/evaluation.json"]["metrics"]["models"]
    models["network"][metric]["values"] = network
    if trivial is None:
        models["trivial"].pop(metric, None)
    else:
        models["trivial"][metric]["values"] = trivial
    return docs


def base_docs() -> dict:
    """The base lane: multiclass, twelve inputs, a masking rung that tests `floor`, per-class
    values that make `busy` weak against the trivial baseline."""
    docs = widen(cf.rename_input(cf.cell_docs(MULTICLASS), "gamma", "floor"))
    docs["records/training.json"]["experiments"]["runs"].append(copy.deepcopy(cf.MASK_RUNG))
    return per_unit(docs, "f1_per_class", [0.6, 0.2, 0.9], [0.3, 0.3, 0.3])


def compiled_report(docs_dir: Path) -> None:
    """A compiled report without a label-role map (a hand-authored one): roles are proposed."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "main.aux").write_text(cf.AUX, encoding="utf-8")
    (docs_dir / "main.pdf").write_bytes(b"%PDF-1.4\n%fixture\n")


def lane(tmp, docs: dict, report: str | None = None, briefs: bool = False) -> Lane:
    """Write `docs` as a bundle under `tmp` (with a `proposed` or `emitted` report, stage
    briefs), extract, draft and author its plan."""
    tmp = Path(tmp)
    if report == "proposed":
        compiled_report(tmp / "docs")
    elif report == "emitted":
        cf.write_report(tmp / "docs")
    model, bundle = cf.write_bundle(tmp, docs)
    if briefs:
        for path in (bundle / "docs" / "pack-brief.md", tmp / "vega-runs/demo/docs/train-brief.md"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# brief\n", encoding="utf-8")
    facts = tmp / "plan" / "facts.json"
    code, _, printed = cf.extract(model, bundle, facts)
    assert code == 0, printed
    code, _, printed = fx.draft_cli(facts, tmp / "plan" / "draft.json")
    assert code == 0, printed
    plan_path = tmp / "plan" / "story-plan.json"
    code, printed = run(autoauthor.main, ["--plan", tmp / "plan" / "draft.json", "--facts", facts,
                                          "--out", plan_path])
    assert code == 0, printed
    return Lane(tmp, facts, plan_path, json.loads(plan_path.read_text(encoding="utf-8")))


@functools.cache
def base() -> Lane:
    """The base lane, built once: a proposed report (fork S1 closed), the masking rung."""
    root = Path(tempfile.mkdtemp(prefix="story-rules-"))
    atexit.register(shutil.rmtree, root, True)
    return lane(root, base_docs(), report="proposed")


def resolve(ln: Lane, plan: dict | None = None, extra=(), facts: Path | None = None,
            out: Path | None = None) -> tuple[int, str, dict | None]:
    """`resolve_story.py` on the lane (on `plan` when given): (exit, printed, story or None)."""
    path = ln.plan_path
    if plan is not None:
        path = ln.root / "plan" / "seeded.json"
        path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    out = out or ln.root / "showcase" / "story.json"
    code, printed = run(resolve_story.main, ["--plan", path, "--facts", facts or ln.facts,
                                             "--out", out, *extra])
    story = json.loads(out.read_text(encoding="utf-8")) if code == 0 and out.is_file() else None
    return code, printed, story


def refused_by(ln: Lane, plan: dict, rule: str, words=(), facts: Path | None = None) -> str:
    """Resolve a seeded plan and assert that exactly `rule` refuses it, naming `words`."""
    out = ln.root / "seeded" / "story.json"
    code, printed, _ = resolve(ln, plan, facts=facts, out=out)
    assert code == 2 and not out.exists(), f"{rule}: exit {code}, printed {printed}"
    assert set(_ERRORS.findall(printed)) == {rule}, f"{rule} expected alone: {printed}"
    lines = [line for line in printed.splitlines() if line.startswith(f"ERROR: {rule} ")]
    missing = [w for w in words if not any(w in line for line in lines)]
    assert not missing, f"{rule}: {missing} not named in {lines}"
    return printed


def seeded(mutate) -> dict:
    plan = copy.deepcopy(base().plan)
    mutate(plan)
    return plan


def _groups_of(n_groups: int, shaped: list) -> list:
    size, extra = divmod(len(shaped), n_groups)
    out, at = [], 0
    for i in range(n_groups):
        take = size + (1 if i < extra else 0)
        out.append({"id": f"group_{chr(97 + i)}", "label": f"group {chr(97 + i)}", "text": "",
                    "members": shaped[at:at + take]})
        at += take
    return out


def _shaped() -> list:
    return base().facts_doc["groups"]["inputs"]["shaped"]


def _duplicate_member(p):
    groups = p["authored"]["input_groups"]
    groups[0]["members"].append(groups[1]["members"][0])


def _drop_member(p) -> str:
    group = next(g for g in p["authored"]["input_groups"] if len(g["members"]) > 1)
    return group["members"].pop()


def _drop(items: list, key: str, value) -> None:
    items[:] = [i for i in items if i.get(key) != value]


def _without_class(p, name):
    for part in ("roles", "source"):
        p["authored"]["class_roles"][part].pop(name)


def _roles(p, **pairs):
    for name, (role, basis) in pairs.items():
        p["authored"]["class_roles"]["roles"][name] = role
        p["authored"]["class_roles"]["source"][name] = basis


# the seeded defects of D5 §12.2 (1–21) on the base lane, each (case, rule, mutation)
D5_CASES = (
    (1, "V2", lambda p: p["authored"].update(headline="The network reads 12 bands")),
    (2, "V2", lambda p: p["authored"]["network"]["choices"][0].update(
        text="It ranks the top7 first.")),
    (3, "V2", lambda p: p["authored"]["network"]["choices"][0].update(title="The β2 setting")),
    (4, "V3", lambda p: p["authored"].update(headline="What the {nonexistent} says")),
    (5, "V4", lambda p: p["authored"]["context"]["steps"].update(
        cleaning="See records/shape.json#manifest.steps for how.")),
    (6, "V5", lambda p: p["authored"].update(motif="traces")),
    (7, "V6", lambda p: _without_class(p, "jammed")),
    (8, "V6", lambda p: _roles(p, calm=("neutral", "user:turn-1"),
                               busy=("neutral", "user:turn-1"))),
    (9, "V6", lambda p: _roles(p, jammed=("alert", "structural:majority"))),
    (10, "V7", _duplicate_member),
    (11, "V7", _drop_member),
    (12, "V7", lambda p: p["authored"].update(input_groups=_groups_of(10, _shaped()))),
    (13, "V8", lambda p: p["authored"]["network"]["choices"].append(
        {"decision": "B99", "title": "Another block", "text": "It does something."})),
    (14, "V8", lambda p: _drop(p["authored"]["network"]["choices"], "decision", "loss")),
    (15, "V9", lambda p: _drop(p["authored"]["results"]["limits"], "id", "exposures")),
    (16, "V10", lambda p: p["authored"]["experiments"]["labels"].update(R99="a run")),
    (17, "V11", lambda p: p["authored"]["code_notes"].update(nowhere="A note.")),
    (18, "V12", lambda p: p["authored"].update(headline="a" * 400)),
    (19, "V13", lambda p: p.update(status="draft")),
    (20, "V14", lambda p: p["authored"]["report_roles"].update({"net.forward": "vega:pack.slice"})),
)


# --------------------------------------------------------------------------- the rules

def test_the_base_plan_passes_every_rule_and_resolves():
    code, printed, story = resolve(base())
    assert code == 0 and story is not None, printed
    assert not _ERRORS.findall(printed) and "OK: story" in printed, printed


def test_d5_seeded_defects_each_refused_by_its_rule():
    ln = base()
    for case, rule, mutate in D5_CASES:
        try:
            refused_by(ln, seeded(mutate), rule)
        except AssertionError as exc:
            raise AssertionError(f"D5-{case}: {exc}") from exc
    facts = json.loads(ln.facts.read_text(encoding="utf-8"))
    facts["facts"]["source_rows"]["value"] += 1  # D5-21: the fact set edited after the draft
    edited = ln.root / "plan" / "facts.edited.json"
    edited.write_text(json.dumps(facts), encoding="utf-8")
    refused_by(ln, copy.deepcopy(ln.plan), "V1", ["facts.digest"], facts=edited)


def test_d6_and_d4_cases_name_their_rule_and_words():
    ln, dropped = base(), []

    def drop_input(p):
        dropped.append(_drop_member(p))
    cases = (
        ("VS01", "V2", ["digit", "headline"],
         lambda p: p["authored"].update(headline="the network reads 12 bands")),
        ("VS02", "V3", ["placeholder", "nonexistent"],
         lambda p: p["authored"].update(lead="It reads {nonexistent} things.")),
        ("VS03", "V4", ["fact"], lambda p: p["authored"]["context"]["steps"].update(
            data="The rows are evaluation:metrics.models.network, as recorded.")),
        ("VS04", "V6", ["Bogus"], lambda p: _roles(p, Bogus=("plain", "vega:default"))),
        ("VS05", "V6", ["every class"], lambda p: _without_class(p, "busy")),
        ("VS07", "V5", ["motif", "grid, points, traces, lanes, links, contours"],
         lambda p: p["authored"].update(motif="packets")),
        ("VS08", "V5", ["modality"], lambda p: p["authored"].update(motif="traces")),
        ("VS09", "V5", ["row"], lambda p: p["authored"]["lexicon"].pop("row")),
        ("VS10", "V5", ["noun"], lambda p: p["authored"]["lexicon"].update(
            noun={"one": "flow", "many": "flows", "article": "a"})),
        ("VS11", "V7", ["exactly one"], _duplicate_member),
        ("VS13", "V7", ["groups"], lambda p: p["authored"].update(
            input_groups=_groups_of(2, _shaped()))),
        ("VS14", "V14", ["role"], lambda p: p["authored"]["report_roles"].update(
            {"net.bogus": "vega:net.forward"})),
        ("VS23", "V12", ["length"], lambda p: p["authored"]["network"]["choices"][0].update(
            title="a" * 61)),
        ("PN5", "V6", ["alert"], lambda p: _roles(p, jammed=("alert", "vega:default"))),
        ("PN6", "V5", ["flow2"], lambda p: p["authored"]["lexicon"]["row"].update(one="flow2")))
    for name, rule, words, mutate in cases:
        try:
            refused_by(ln, seeded(mutate), rule, words)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc
    for name in ("VS12", "PN12"):  # one input in no group: the refusal names it
        plan = seeded(drop_input)
        refused_by(ln, plan, "V7", [repr(dropped[-1])])


def test_vs06_an_alert_on_a_task_without_classes():
    with tempfile.TemporaryDirectory() as tmp:
        ln = lane(tmp, fx.cell("regression"))
        assert ln.plan["authored"]["class_roles"] is None
        plan = copy.deepcopy(ln.plan)
        plan["authored"]["class_roles"] = {"roles": {"high": "alert"},
                                           "source": {"high": "user:turn-1"}}
        refused_by(ln, plan, "V6", ["alert", "classes"])


def test_a_malformed_plan_is_refused_never_a_traceback():
    ln = base()
    for rule, where, mutate in (
            ("V13", "authored.network.choices[0].title",
             lambda p: p["authored"]["network"]["choices"][0].pop("title")),
            ("V13", "authored.results.limits[0].text",
             lambda p: p["authored"]["results"]["limits"][0].pop("text")),
            ("V13", "authored.results.notes",
             lambda p: p["authored"]["results"].update(notes=["x"])),
            ("V13", "authored.input_groups[0].label",
             lambda p: p["authored"]["input_groups"][0].pop("label"))):
        refused_by(ln, seeded(mutate), rule, [where])
    code, printed, _ = resolve(ln, seeded(lambda p: p["proposals"]["runs"][0].pop("primary")),
                               out=ln.root / "malformed" / "story.json")
    assert code == 2 and "does not read as vega/story-plan@2" in printed, printed
    for slot in ln.plan["authored"]:
        for wrong in (42, "text", [], {}, [{"x": 1}], {"x": [1]}):
            plan = seeded(lambda p: p["authored"].update({slot: wrong}))
            code, printed, _ = resolve(ln, plan, out=ln.root / "malformed" / "story.json")
            assert code in (0, 2), (slot, wrong, printed)  # an exception would have escaped
            if wrong in (42, [], [{"x": 1}]) and slot not in ("monitor_note", "report_roles"):
                assert code == 2 and _ERRORS.findall(printed), (slot, wrong, printed)


def test_v1_pins_the_fact_set_and_the_installed_template():
    ln = base()
    assert ln.plan["template"]["manifest_digest"] == page_template.digests()["manifest_digest"]
    assert ln.plan["facts"]["digest"] == digest_file(ln.facts)
    plan = copy.deepcopy(ln.plan)
    plan["template"]["manifest_digest"] = "sha256:" + "0" * 64
    refused_by(ln, plan, "V1", ["template.manifest_digest", "page template"])


# --------------------------------------------------------------------------- every admitted cell

def referenced(story: dict) -> set:
    """Every values key a field of the story references (independently of the resolver)."""
    texts = [story["headline"], story["lead"]]
    keys = set()
    for section in story["sections"]:
        sid = section["id"]
        if sid == "context":
            texts += [s["text"] for s in section["steps"]] + [f["label"] for f in section["facts"]]
            keys |= {f["value"] for f in section["facts"]}
        elif sid == "network":
            texts += [section["lead"]] + [t for c in section["choices"]
                                          for t in (c["title"], c["text"])]
        elif sid == "experiments":
            texts += [section["lead"], section["caption"]] + [r["label"] for r in section["runs"]]
            keys |= {r["value"] for r in section["runs"]}
        elif sid == "results":
            texts += [section["headline"]] + [f.get("note", "") for f in section["facts"]]
            texts += [t for lim in section["limits"] for t in (lim["title"], lim["text"])]
            keys |= {m["value"] for m in section["models"]} | {f["value"] for f in section["facts"]}
            b = section["breakdown"] or {}
            texts.append(b.get("caption") or "")
            keys |= {b.get(k) for k in ("names", "values", "weak")} | {section["verdict"]["diff"]}
            keys |= {section["verdict"].get("reason")}
        else:
            texts.append(section["monitor_note"] or "")
    texts += [t for g in story["input_groups"] for t in (g["label"], g["text"])]
    for note in story["code_notes"].values():
        texts.append(note["note"])
        keys |= {(note["ref"] or {}).get(k) for k in ("number", "page")}
        keys |= {((note["ref"] or {}).get("chapter") or {}).get("number")}
    keys |= {p.name for t in texts for p in placeholders(t) if not p.is_lexicon_path}
    return keys - {None}


def story_problems(ln: Lane, story: dict) -> list[str]:
    """What is wrong with a resolved story against its schema and its fact set."""
    from jsonschema import Draft202012Validator
    bad = [f"schema: {e.message}" for e in Draft202012Validator(STORY_SCHEMA).iter_errors(story)]
    facts = ln.facts_doc
    reserved = {"primary", *(["ci_level"] if "eval_ci_level" in facts["facts"] else [])}
    values, want = set(story["values"]), referenced(story) | reserved
    bad += [] if values == want else [f"values {sorted(values ^ want)} differ from the fields"]
    bad += [] if values == set(story["formats"]) == set(story["refs"]) else ["formats/refs keys"]
    primary = facts["facts"][f"eval_network_{facts['task']['primary']}"]
    if story["values"]["primary"] != estimate(primary["value"]) or \
            story["refs"]["primary"] != primary["ref"]:
        bad.append(f"values.primary {story['values']['primary']} is not {primary}")
    level = facts["facts"].get("eval_ci_level", {}).get("value")
    if "ci_level" in reserved and story["values"]["ci_level"] != level:
        bad.append("values.ci_level is not the evaluation's")
    required = CATALOG["lexicon"]["required"][_shape_key(facts["task"])]
    bad += [f"lexicon lacks {k}" for k in required if k not in story["lexicon"]]
    breakdown = story["sections"][3]["breakdown"]
    if (breakdown is None) != (facts["groups"].get("breakdown") is None):
        bad.append(f"breakdown {breakdown} against the facts' {facts['groups'].get('breakdown')}")
    return bad + lane_problems(ln, story)


def lane_problems(ln: Lane, story: dict) -> list[str]:
    """The story's task against the page's lane and metric, as `build_showcase` compares them
    (`page_config._check_story`): `page_derive.lane_record`, `page_config.metric_record`."""
    netspec = next(ln.root.glob("bundle/*/netspec.json"), None)
    if netspec is None:
        return []
    spec, catalogs = json.loads(netspec.read_text(encoding="utf-8")), Catalogs.load()
    lane, metric = lane_record(spec, catalogs), metric_record(spec, catalogs)
    want = {"kind": lane["task"], "base": lane["base"], "view_key": lane["view_key"],
            "sample_shape": lane["shape"], "graph_level": lane["level"], "family": lane["family"],
            "goal": lane["goal"], "primary": metric["primary"], "direction": metric["direction"]}
    return [f"task.{k} {story['task'][k]!r} ≠ the lane's {v!r}" for k, v in want.items()
            if story["task"][k] != v]


def _shape_key(task: dict) -> str:
    shape = task["sample_shape"]
    return f"graph:{task['graph_level']}" if shape == "graph" else shape


def test_every_admitted_cell_resolves_a_story_its_schema_accepts():
    failed = []
    for cell in cf.cells():
        with tempfile.TemporaryDirectory() as tmp:
            ln = lane(tmp, cf.cell_docs(cell))
            code, printed, story = resolve(ln)
            bad = [f"exit {code}: {printed[-400:]}"] if code else story_problems(ln, story)
            if not bad:
                again, printed, _ = resolve(ln, extra=["--check"])
                bad = [] if again == 0 else [f"--check exit {again}: {printed}"]
            if bad:
                failed.append(f"{cell}: {bad[:3]}")
    assert len(cf.cells()) >= 120 and not failed, failed[:3]


# --------------------------------------------------------------------------- the blocks

def test_input_groups_partition_within_the_bounds():
    shaped = [f"in_{c}" for c in "abcdefghijkl"]
    assert group_bounds(12) == (3, 9) and group_bounds(9) == (3, 9) and group_bounds(4) == (1, 4)
    assert not check_input_groups(_groups_of(3, shaped), shaped)
    assert not check_input_groups(_groups_of(9, shaped), shaped)
    for n in (2, 10):
        found = check_input_groups(_groups_of(n, shaped), shaped)
        assert [p.rule for p in found] == ["V7"] and "3 to 9 groups" in found[0].message, found
    assert not check_input_groups(_groups_of(1, shaped[:4]), shaped[:4])
    groups = _groups_of(3, shaped)
    groups[0]["members"].append("nowhere")
    groups[1]["id"] = groups[2]["id"]
    messages = " | ".join(p.message for p in check_input_groups(groups, shaped))
    assert "'nowhere' is not a network input" in messages and "used twice" in messages, messages
    carried = _groups_of(3, shaped[:9]) + [{"id": "_unplaced", "label": "unplaced", "text": "",
                                            "members": shaped[9:]}]
    found = check_input_groups(carried, shaped)
    assert any("'_unplaced'" in p.message and "place each" in p.message for p in found), found


def test_the_author_places_the_unplaced_inputs_of_a_carry():
    ln = base()
    proposed = ln.plan["proposals"]["input_groups"]
    assert len(proposed) >= 3, proposed
    carried = [{"id": g["suggested_id"], "label": f"carried {g['suggested_id']}", "text": "",
                "members": list(g["members"][1:] or g["members"])} for g in proposed[:3]]
    held = {m for g in carried for m in g["members"]}
    unplaced = [n for n in _shaped() if n not in held]
    plan = copy.deepcopy(ln.plan)
    plan["authored"]["input_groups"] = carried + [{"id": "_unplaced", "label": "unplaced",
                                                   "text": "", "members": unplaced}]
    refused_by(ln, plan, "V7", ["_unplaced"])
    facts = FactSet.from_json(ln.facts_doc)
    authored = autoauthor.AutoAuthor(facts, resolve_story.report_labels_for(None, facts, ln.facts))
    done = authored.author(plan)["authored"]["input_groups"]
    assert [g["id"] for g in done] == [g["id"] for g in carried], done
    assert not fx.partition_problems(done, _shaped(), 3, 9), done
    for group, proposal in zip(done, proposed[:3]):
        assert set(proposal["members"]) <= set(group["members"]), (group, proposal)


def test_the_focus_of_each_choice():
    ln = base()
    plan = copy.deepcopy(ln.plan)
    plan["authored"]["network"]["choices"].append({"decision": "B3", "title": "A dense layer",
                                                   "text": "It mixes the inputs."})
    code, printed, story = resolve(ln, plan)
    assert code == 0, printed
    authored = plan["authored"]["network"]["choices"]
    focus = {c["decision"]: s["focus"] for c, s in zip(authored, story["sections"][1]["choices"])}
    assert focus["family"] == {"kind": "feature", "ref": "floor"}, focus
    assert focus["loss"] == {"kind": "family", "ref": "objective"}
    assert focus["optimizer"] == {"kind": "family", "ref": "optimization"}
    assert focus["regularization[dropout]"] == {"kind": "family", "ref": "regularization"}
    assert focus["B3"] == {"kind": "block", "ref": "B3"}
    family = next(d for d in ln.facts_doc["groups"]["decisions"] if d["id"] == "family")
    assert family["features_tested"] == ["floor"]
    assert focus_of(family, "A new family.") == {"kind": "family", "ref": "architecture"}
    assert focus_of({"id": "weird", "path": "weird"}, "text") is None
    plan["proposals"]["choices"].append({"decision": "weird", "family": "architecture",
                                         "mandatory": False})
    plan["authored"]["network"]["choices"].append({"decision": "weird", "title": "Odd",
                                                   "text": "No focus fits it."})
    refused_by(ln, plan, "V8", ["maps to no focus kind"])


def _task(**fields) -> TaskShape:
    base = {"kind": "multiclass", "goal": "predict", "sample_shape": "row", "family": "mlp-plain",
            "view_key": "classes", "classes": ["calm", "busy", "jammed"], "primary": "f1_macro",
            "direction": "max"}
    return TaskShape(**{**base, **fields})


def _block(**pairs) -> dict:
    return {"roles": {c: r for c, (r, _) in pairs.items()},
            "source": {c: b for c, (_, b) in pairs.items()}}


def test_the_class_role_rule():
    task = _task()
    proposal = {"neutral": {"class": "calm", "rule": "structural:majority"}}
    ok = _block(calm=("neutral", "structural:majority"), busy=("alert", "user:turn-2"),
                jammed=("plain", "vega:default"))
    assert not check_class_roles(ok, task, proposal)
    shown = _block(calm=("neutral", "structural:majority"), busy=("alert", "show:2026-09-30"),
                   jammed=("plain", "vega:default"))
    assert not check_class_roles(shown, task, proposal)
    for block, words in (
            (_block(calm=("plain", "vega:default"), busy=("alert", "vega:default"),
                    jammed=("plain", "vega:default")), "alert requires"),
            (_block(calm=("plain", "vega:default"), busy=("neutral", "structural:majority"),
                    jammed=("plain", "vega:default")), "structural rule gives only to 'calm'"),
            (_block(calm=("neutral", "user:turn-1"), busy=("neutral", "user:turn-1"),
                    jammed=("plain", "vega:default")), "at most one"),
            (_block(calm=("plain", "domain:traffic"), busy=("plain", "vega:default"),
                    jammed=("plain", "vega:default")), "basis"),
            (_block(calm=("plain", "vega:default"), busy=("plain", "vega:default")),
             "every class")):
        found = check_class_roles(block, task, proposal)
        assert found and {p.rule for p in found} == {"V6"} and \
            any(words in p.message for p in found), (words, found)
    labels = _task(kind="multilabel", view_key="labels", primary="f1_macro")
    assert not check_class_roles(_block(calm=("neutral", "user:t"), busy=("neutral", "user:t"),
                                        jammed=("plain", "vega:default")), labels, {})


def test_a_registry_basis_is_the_harness_s_only(tmp_path, monkeypatch):
    """`registry:<lane>` holds only when VEGA_LANE_REGISTRY (the matrix's) names a lane registry
    declaring that lane and that role for that class; a production run never sets it."""
    task, proposal = _task(), {"neutral": {"class": "calm", "rule": "structural:majority"}}
    block = _block(calm=("neutral", "structural:majority"), busy=("alert", "registry:L07"),
                   jammed=("plain", "vega:default"))
    monkeypatch.delenv("VEGA_LANE_REGISTRY", raising=False)
    found = check_class_roles(block, task, proposal)
    assert [p.rule for p in found] == ["V6"] and "no lane registry is given" in found[0].message
    lanes = tmp_path / "lanes.json"
    declared = {"lanes": [{"id": "L07", "roles": {"calm": "neutral", "busy": "alert"}}]}
    lanes.write_text(json.dumps(declared))
    monkeypatch.setenv("VEGA_LANE_REGISTRY", str(lanes))
    found = check_class_roles(block, task, proposal)
    assert found and "is no lane registry" in found[0].message, "any JSON is no lane registry"
    lanes.write_text(json.dumps({"schema": "vega/test-lanes@1", **declared}))
    assert not check_class_roles(block, task, proposal)
    other = _block(calm=("neutral", "structural:majority"), busy=("plain", "registry:L07"),
                   jammed=("alert", "registry:L08"))
    said = [p.message for p in check_class_roles(other, task, proposal)]
    assert any("gives 'busy' the role 'alert'" in m for m in said), said
    assert any("has no lane 'L08'" in m for m in said), said


def test_the_anomaly_sides_and_a_task_without_classes():
    task = _task(kind="anomaly", goal="anomaly", family="autoencoder", view_key="anomaly",
                 classes=None, outcomes=["normal", "anomalous"], primary="anomaly_score_mean",
                 direction="min")
    proposal = {"neutral": {"class": "normal", "rule": "task:anomaly"}}
    ok = _block(normal=("neutral", "task:anomaly"), anomalous=("alert", "task:anomaly"))
    assert not check_class_roles(ok, task, proposal)
    down = _block(normal=("neutral", "task:anomaly"), anomalous=("plain", "user:turn-1"))
    assert any("never downgraded" in p.message for p in check_class_roles(down, task, proposal))
    swapped = _block(normal=("alert", "task:anomaly"), anomalous=("alert", "task:anomaly"))
    assert any("normal side neutral" in p.message
               for p in check_class_roles(swapped, task, proposal))
    value = _task(kind="regression", view_key="value", classes=None, primary="rmse",
                  direction="min")
    assert not check_class_roles(None, value, None)
    found = check_class_roles(_block(high=("alert", "user:turn-1")), value, None)
    assert found and "alert" in found[0].message and "classes" in found[0].message, found


def test_the_lexicon_keys_the_shape_requires():
    entry = {"one": "reading", "many": "readings", "article": "a"}
    row = {"row": entry, "feature": entry, "source": {**entry, "article": "the"}}
    assert not check_lexicon(row, "row")
    assert not check_lexicon({**row, "target": entry}, "row")
    missing = {p.where.rsplit(".", 1)[1] for p in check_lexicon(row, "window")}
    assert missing == {"step", "window"}, missing
    assert {p.where.rsplit(".", 1)[1] for p in check_lexicon(
        {**row, "node": entry, "edge": entry}, "graph:graph")} == {"graph"}
    assert not check_lexicon({**row, "node": entry, "edge": entry}, "graph:node")
    for lexicon, words in (({**row, "row": {**entry, "one": "flow2"}}, "flow2"),
                           ({**row, "row": {**entry, "article": "the"}}, "article"),
                           ({**row, "noun": entry}, "'noun' is not a lexicon key"),
                           ({**row, "feature": {"one": "reading"}}, "exactly")):
        found = check_lexicon(lexicon, "row")
        assert found and any(words in p.message for p in found), (words, found)


# --------------------------------------------------------------------------- computed, bound

def test_weak_is_computed_against_the_trivial_side_and_the_bar():
    code, printed, story = resolve(base())
    assert code == 0, printed
    breakdown = story["sections"][3]["breakdown"]
    assert breakdown["kind"] == "per_class" and breakdown["direction"] == "max"
    assert story["values"][breakdown["weak"]] == [False, True, False]
    assert story["refs"][breakdown["weak"]].endswith("models.trivial.f1_per_class.values")
    assert story["values"][breakdown["names"]] == ["calm", "busy", "jammed"]
    with tempfile.TemporaryDirectory() as tmp:
        cell = {"family": "cnn1d", "shape": "window", "level": None, "task": "forecast",
                "variant": None}
        docs = per_unit(cf.cell_docs(cell), "per_horizon_error", [1.0, 2.0, 3.0], [1.5, 1.5, 1.5])
        code, printed, story = resolve(lane(tmp, docs))
        assert code == 0, printed
        breakdown = story["sections"][3]["breakdown"]
        assert breakdown["direction"] == "min" and breakdown["kind"] == "per_horizon"
        assert story["values"][breakdown["weak"]] == [False, True, True]
    with tempfile.TemporaryDirectory() as tmp:
        docs = per_unit(cf.cell_docs(MULTICLASS), "f1_per_class", [0.6, 0.2, 0.9], None)
        code, printed, story = resolve(lane(tmp, docs))
        assert code == 0 and story["sections"][3]["breakdown"]["weak"] is None, printed
        assert "no trivial side and no bar" in printed, printed
    assert weak_flags([0.5, 0.5], None, "max", [[0.4, 0.6], [0.3, 0.45]], 0.5) == [False, True]
    assert weak_flags([2.0, 1.0], [3.0, 3.0], "min", [[1.9, 2.1], [0.5, 1.6]], 1.5) == \
        [True, False]
    assert weak_flags([2.0, 1.0], [1.0, 1.0], "min") == [True, False]


def test_the_reserved_primary_and_ci_level_are_bound():
    code, printed, story = resolve(base())
    assert code == 0, printed
    assert story["values"]["primary"] == {"value": 0.7, "ci": [0.6, 0.8], "n": 180}
    assert story["refs"]["primary"] == "records/evaluation.json#metrics.models.network.f1_macro"
    assert story["formats"]["primary"] == "ci"
    assert story["values"]["ci_level"] == 0.95
    assert story["refs"]["ci_level"] == "records/evaluation.json#metrics.ci_level"
    with tempfile.TemporaryDirectory() as tmp:
        docs = cf.cell_docs(MULTICLASS)
        docs["records/evaluation.json"]["metrics"].pop("ci_level")
        ln = lane(tmp, docs)
        code, printed, story = resolve(ln)
        assert code == 0 and "ci_level" not in story["values"], printed
        assert "WARN: the evaluation records no ci_level" in printed, printed
        assert not story_problems(ln, story)


def _verdict(docs: dict, verdict: dict) -> dict:
    """`docs` whose evaluation records `verdict` (where the record and its metrics hold it)."""
    record = docs["records/evaluation.json"]
    record["verdict"], record["metrics"]["verdict"] = verdict, copy.deepcopy(verdict)
    return docs


def test_an_unearned_verdict_is_said_in_the_markers():
    """Contract §1: an unearned verdict is shown as recorded, with `WARN:` — the resolver prints it
    (into the showcase's markers, which step 8 carries into the layer); an earned one has none."""
    with tempfile.TemporaryDirectory() as tmp:
        docs = _verdict(cf.cell_docs(MULTICLASS), {"earned": False, "strongest_baseline": "trivial",
                                                   "diff": -0.1, "ci": [-0.2, -0.05], "beaten_by_baseline": True})
        code, printed, story = resolve(lane(tmp, docs))
        assert code == 0, printed
        assert "WARN: verdict not earned — beaten by trivial" in printed, printed
    code, printed, _ = resolve(base())
    assert code == 0 and "WARN: verdict not earned" not in printed, printed


def test_a_limit_quotes_the_records_own_words():
    """The ship stance, a warning's line and the classes with few sealed test rows reach the story's
    limits as `recorded` — the record's words, quoted by the page beside the authored text."""
    with tempfile.TemporaryDirectory() as tmp:
        docs = cf.cell_docs(MULTICLASS)
        docs["records/evaluation.json"]["metrics"]["class_support"] = {"busy": 400, "idle": 3, "jammed": 12}
        warned = ("WARN: not declared in /home/someone/runs/blueprint-runs/r-a/pipeline.json "
                  "(serving.inspection: GET /v1/network/tensor/{tensor_id})")
        docs["records/evaluation.json"].setdefault("markers", []).append(warned)
        ln = lane(tmp, docs)
        facts = json.loads(ln.facts.read_text(encoding="utf-8"))
        small = [lim for lim in facts["groups"]["limits"] if lim["id"] == "small_support"]
        assert small and small[0]["recorded"] == "idle: 3, jammed: 12", facts["groups"]["limits"]
        # verbatim, but a path of the machine's file system is cut to its file name; a route stays
        quoted = [lim["recorded"] for lim in facts["groups"]["limits"] if lim["id"].startswith("warn_eval_")]
        assert ("WARN: not declared in …/pipeline.json (serving.inspection: GET "
                "/v1/network/tensor/{tensor_id})") in quoted, quoted
        plan = copy.deepcopy(ln.plan)
        if not any(lim["id"] == "small_support" for lim in plan["authored"]["results"]["limits"]):
            plan["authored"]["results"]["limits"].append({"id": "small_support", "title": "Rare classes",
                                                          "text": "A few classes have a handful of rows."})
        code, printed, story = resolve(ln, plan)
        assert code == 0, printed
        limits = {lim["id"]: lim for lim in story["sections"][3]["limits"]}
        assert limits["small_support"]["recorded"] == "idle: 3, jammed: 12"
        assert "recorded" not in limits["split"]


def test_the_verdict_is_carried_as_recorded():
    with tempfile.TemporaryDirectory() as tmp:  # no baseline fitted: no beaten_by_baseline
        docs = _verdict(cf.cell_docs(MULTICLASS), {"earned": False, "strongest_baseline": None,
                                                   "diff": None, "ci": []})
        code, printed, story = resolve(lane(tmp, docs))
        assert code == 0, printed
        assert story["sections"][3]["verdict"] == {"earned": False, "beaten_by_baseline": False,
                                                   "strongest_baseline": None, "diff": None}
    with tempfile.TemporaryDirectory() as tmp:
        docs = _verdict(cf.cell_docs(MULTICLASS), {"strongest_baseline": "trivial"})
        ln = lane(tmp, docs)
        code, printed, _ = resolve(ln)
        assert code == 2 and "records no verdict" in printed, printed
        assert not (ln.root / "showcase" / "story.json").exists()


def test_a_verdict_that_is_not_assessable_is_carried_with_its_reason():
    """A30: an unlabelled anomaly evaluation records `earned` null with its reason. The story
    carries it as recorded — earned and beaten_by_baseline null, never false, no baseline, no
    difference — with the reason as a values key (verbatim, its ref the record's), and the limit
    `verdict_not_assessable` is mandatory. A null `earned` without a reason, or beside a named
    baseline, is refused: the page never invents a verdict."""
    cell = {"family": "vae", "shape": "row", "level": None, "task": "anomaly/unlabelled",
            "variant": None}
    assert cf.task_of(cell)[2] is False, "an unlabelled anomaly cell"
    with tempfile.TemporaryDirectory() as tmp:
        ln = lane(tmp, cf.cell_docs(cell))
        code, printed, story = resolve(ln)
        assert code == 0, printed
        results = story["sections"][3]
        assert results["verdict"] == {"earned": None, "beaten_by_baseline": None,
                                      "strongest_baseline": None, "diff": None,
                                      "reason": "eval_verdict_reason"}
        assert story["values"]["eval_verdict_reason"] == cf.NOT_ASSESSABLE["reason"]
        assert story["formats"]["eval_verdict_reason"] == "verbatim"
        assert story["refs"]["eval_verdict_reason"] == "records/evaluation.json#verdict.reason"
        assert "verdict_not_assessable" in {lim["id"] for lim in results["limits"]}
        assert not {"verdict_not_earned", "verdict_beaten"} & {lim["id"] for lim in results["limits"]}
        assert not story_problems(ln, story)
        limit = next(lim for lim in ln.facts_doc["groups"]["limits"]
                     if lim["id"] == "verdict_not_assessable")
        assert limit["mandatory"] and limit["prose_ref"] == "records/evaluation.json#readings.verdict"
        plan = copy.deepcopy(ln.plan)
        plan["authored"]["results"]["limits"] = [
            lim for lim in plan["authored"]["results"]["limits"] if lim["id"] != "verdict_not_assessable"]
        refused_by(ln, plan, "V9", ("verdict_not_assessable",))
    for broken, words in (({"reason": None}, "records no verdict"),
                          ({"strongest_baseline": "trivial"}, "names strongest_baseline"),
                          ({"beaten_by_baseline": False}, "names beaten_by_baseline")):
        with tempfile.TemporaryDirectory() as tmp:
            docs = _verdict(cf.cell_docs(cell), {**cf.NOT_ASSESSABLE, **broken})
            ln = lane(tmp, docs)
            code, printed, _ = resolve(ln)
            assert code == 2 and words in printed, (broken, printed)
            assert not (ln.root / "showcase" / "story.json").exists()


def test_resolve_check_is_byte_equality():
    ln = base()
    out = ln.root / "check" / "story.json"
    code, printed, _ = resolve(ln, out=out, extra=["--check"])
    assert code == 2 and "no story to check" in printed, printed
    code, printed, story = resolve(ln, out=out)
    assert code == 0, printed
    first = out.read_bytes()
    code, printed, _ = resolve(ln, out=out)
    assert code == 0 and out.read_bytes() == first, "two resolutions differ"
    code, printed, _ = resolve(ln, out=out, extra=["--check"])
    assert code == 0 and "byte for byte" in printed, printed
    out.write_bytes(first.replace(b'"locale": "en"', b'"locale":  "en"'))
    code, printed, _ = resolve(ln, out=out, extra=["--check"])
    assert code == 3 and "differs" in printed, printed
    out.write_bytes(first)
    plan = copy.deepcopy(ln.plan)
    plan["authored"]["lead"] = "Another lead, the same facts."
    code, printed, _ = resolve(ln, plan, out=out, extra=["--check"])
    assert code == 3, printed


# --------------------------------------------------------------------------- the report

def test_proposed_roles_are_bound_against_the_aux():
    ln = base()
    assert ln.plan["proposals"]["report_roles"]["status"] == "proposed"
    assert ln.plan["authored"]["report_roles"] == {"net.forward": "vega:net.forward",
                                                   "pack.slice": "vega:pack.slice"}
    code, printed, story = resolve(ln)
    assert code == 0, printed
    dense = story["code_notes"]["dense"]
    assert dense["link"] == {"href": "showcase/report/report.pdf#page=2", "kind": "report"}
    assert dense["ref"] == {"kind": "equation", "number": "report_net_forward_number",
                            "page": "report_net_forward_page", "title": "The forward pass"}
    assert story["values"]["report_net_forward_number"] == "3.1"
    assert story["values"]["report_net_forward_page"] == 2
    assert story["refs"]["report_net_forward_number"] == "report:net.forward.number"
    assert story["code_notes"]["store"]["ref"]["number"] == "report_pack_slice_number"
    assert story["code_notes"]["root"] == {"note": ln.plan["authored"]["code_notes"]["root"],
                                           "link": None, "ref": None}
    report = story["report"]
    assert report["status"] == "linked" and report["roles_source"] == "proposed"
    assert report["file"] == "report/report.pdf" and report["roles_digest"] is None
    assert report["aux_digest"] == digest_file(ln.root / "docs" / "main.aux")


def test_an_emitted_role_map_and_brief_links():
    with tempfile.TemporaryDirectory() as tmp:
        ln = lane(tmp, cf.cell_docs(MULTICLASS), report="emitted")
        assert ln.plan["authored"]["report_roles"] is None
        code, printed, story = resolve(ln)
        assert code == 0, printed
        store = story["code_notes"]["store"]
        assert store["link"] == {"href": "showcase/report/report.pdf#page=4", "kind": "report"}
        assert store["ref"]["number"] == "report_pack_slice_number"
        assert story["values"]["report_pack_slice_number"] == "3.2"
        assert story["report"]["roles_source"] == "document-project"
        assert story["report"]["roles_digest"] == digest_file(Path(tmp) / "docs/label-roles.json")
        plan = copy.deepcopy(ln.plan)
        plan["authored"]["report_roles"] = {"net.forward": "vega:net.forward"}
        refused_by(ln, plan, "V14", ["not proposed"])
    with tempfile.TemporaryDirectory() as tmp:
        ln = lane(tmp, cf.cell_docs(MULTICLASS), briefs=True)
        code, printed, story = resolve(ln)
        assert code == 0, printed
        assert story["report"]["status"] == "brief-links"
        assert story["report"]["briefs"] == {"train": "briefs/train-brief.md",
                                             "pack": "briefs/pack-brief.md"}
        assert story["code_notes"]["store"] == {
            "note": ln.plan["authored"]["code_notes"]["store"], "ref": None,
            "link": {"href": "showcase/briefs/pack-brief.md", "kind": "file"}}
        assert story["code_notes"]["dense"]["link"] is None


# --------------------------------------------------------------------------- the CIDS lane

def test_the_cids_carry_completed_and_resolved():
    fx.need_cids()
    if not fx.CIDS_PLAN_V1.is_file():
        fx.skip(f"{fx.CIDS_PLAN_V1} is absent")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        facts = fx.cids_facts(root)
        code, draft, printed = fx.draft_cli(facts, root / "draft.json",
                                            ["--carry", fx.CIDS_PLAN_V1])
        assert code == 0, printed
        ln = Lane(root, facts, root / "plan" / "story-plan.json", {})
        code, printed = run(autoauthor.main, ["--plan", root / "draft.json", "--facts", facts,
                                              "--out", ln.plan_path])
        assert code == 0, printed
        ln.plan = json.loads(ln.plan_path.read_text(encoding="utf-8"))
        code, printed, story = resolve(ln)
        assert code == 0, printed
        assert story["headline"] == draft["authored"]["headline"]
        assert story["class_roles"]["roles"]["BENIGN"] == "neutral"
        focus = [c["focus"] for c in story["sections"][1]["choices"]]
        assert {"kind": "feature", "ref": "Destination Port"} in focus, focus
        assert story["report"]["roles_source"] == "proposed"
        bound = [n["ref"] for n in story["code_notes"].values() if n["ref"]]
        assert bound and all(story["refs"][r["number"]].startswith("report:") for r in bound)
        assert not story_problems(ln, story), story_problems(ln, story)


if __name__ == "__main__":
    sys.exit(fx.run_tests(globals()))
