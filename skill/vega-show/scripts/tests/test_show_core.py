#!/usr/bin/env python3
"""
test_show_core.py — the value objects of `vega_show_core.py` and the bundle adapter
`records.BundleRecords`.

    python3 test_show_core.py             # or: python3 -m pytest test_show_core.py

- The closed sets are the catalog's: every enum's members equal the catalog list it is built
  from.
- `FactRef` is the contract's ref grammar (parse ∘ str is the identity; a ref outside the
  grammar is refused); `fact_key` is its key grammar (`H1-5` → `h1_5`).
- `FactSet.from_json` ∘ `to_json` is the identity on the schema's own example, which still
  validates `show-facts-1.schema.json` after the round trip; frozen values hash and compare
  by value, and a frozen object reads as a mapping (`"k" in f`, `keys`, `values`, `items`).
- `Fact` and `TaskShape` refuse what the schema refuses: a task field outside its enum or
  pattern, a None where the schema wants a value, a missing or unknown field.
- `write_json` gives a file the mode a plain `open()` gives (0666 less the umask), so the
  serving image's unprivileged user can read what the stage wrote.
- `BundleRecords` reads only verified, pinned fact sources: refs resolve through keys,
  indices and `[key=value]` selectors; an unresolvable ref, an unpinned file and a `report:`
  ref are `FactMissing`; a record altered after pack, or a pinned record gone, is
  `BundleAltered` at open (D5's seeded defect 22, gate PF2). On the CIDS bundle, when present,
  the golden values of D5 §12.3 read back.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))

from records import BundleAltered, BundleRecords  # noqa: E402
from show_ports import RecordStore  # noqa: E402
from vega_show_core import (CATALOG, CheckStatus, ClassRole, Fact, FactKind,  # noqa: E402
                            FactMissing, FactRef, FactSet, FrozenJson, LabelKind, Motif, Problem,
                            ShowPaths, TaskShape, digest_bytes, digest_file, fact_key, freeze,
                            thaw, write_json)

SCHEMAS = TESTS.parent.parent / "references" / "schemas"
CIDS_BUNDLE = (Path(os.environ.get("CIDS_ROOT", Path.home() / "Documenti" / "cids"))
               / "bundle" / "cicids2017-label-R7")


def _bundle(tmp: Path) -> Path:
    """A bundle with a netspec, two records and one unpinned record, pinned by MANIFEST.files."""
    docs = {"netspec.json": {"task": {"classes": ["BENIGN", "Bot"]},
                             "blocks": [{"id": "B1", "type": "dense", "params": {"units": 256}},
                                        {"id": "B2", "type": "dense", "params": {"units": 64}}]},
            "records/shape.json": {"manifest": {"steps": [
                {"id": "S4", "rows_after": {"all": 2522362}},
                {"id": "S5", "fit": True, "rows_after": {"all": 11}}]}},
            "records/training.json": {"experiments": {"runs": [{"id": "R7", "primary": 0.81775308593454}]}}}
    files = {}
    for rel, doc in docs.items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text(json.dumps(doc), encoding="utf-8")
        files[rel] = digest_file(tmp / rel)
    (tmp / "records" / "analysis.json").write_text("{}", encoding="utf-8")
    manifest = {"schema": "vega/bundle@2", "files": files, "serve": {"identity": {"noun": "flow"}}}
    (tmp / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp


def test_enums_are_the_catalogs_closed_sets():
    assert [str(m) for m in FactKind] == CATALOG["fact_kinds"]
    assert [str(m) for m in ClassRole] == CATALOG["roles"]["set"]
    assert [str(m) for m in Motif] == CATALOG["motifs"]["set"]
    assert [str(m) for m in LabelKind] == [*CATALOG["label_kinds"]["kinds"], "other"]
    assert [str(m) for m in CheckStatus] == list(CATALOG["check_status"])
    assert FactKind("ci_metric") is FactKind.CI_METRIC and Motif.LINKS == "links"


def test_fact_ref_grammar():
    for text in ("records/shape.json#manifest.steps[id=S4].rows_after.all", "netspec.json#task.classes.1",
                 "MANIFEST.json#serve.identity.noun", "report:net.forward.number",
                 "records/evaluation.json#metrics.models[name=f1.macro].value"):
        assert str(FactRef.parse(text)) == text, text
    ref = FactRef.parse("records/shape.json#manifest.steps[id=S4].rows_after.all")
    assert ref.segments() == (("manifest", None, None), ("steps", "id", "S4"), ("rows_after", None, None),
                              ("all", None, None))
    dotted = FactRef.parse("records/evaluation.json#m[name=f1.macro].v")
    assert dotted.segments()[0] == ("m", "name", "f1.macro")
    assert FactRef.parse("report:net.forward.page").report_field() == ("net.forward", "page")
    for bad in ("records/other.json#a", "netspec.json", "netspec.json#a b", "report:net.forward.label",
                "netspec.json#a..b", "netspec.json#a[id=S4", "weights.pt#x"):
        try:
            FactRef.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was accepted")


def test_fact_key_grammar():
    assert fact_key(("step", "S4", "rows_after")) == "step_s4_rows_after"
    assert fact_key("H1-5") == "h1_5" and fact_key(["train_share", "DoS Hulk"]) == "train_share_dos_hulk"
    assert fact_key(("report", "net.forward", "number")) == "report_net_forward_number"
    assert fact_key(("block", "Ж-B1")) == "block_b1"
    assert fact_key(("run", "R7", "delta", 0)) == "run_r7_delta_0"
    for bad in ("5xx", "", "a" * 65, ("-", "")):
        try:
            fact_key(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} gave a key")


def test_facts_are_validated_and_frozen():
    fact = Fact("eval_network_f1_macro", {"value": 0.84, "ci": [0.83, 0.85], "n": 504473},
                "records/evaluation.json#metrics.models.network.f1_macro", "ci_metric", "ci", "max")
    assert fact.kind is FactKind.CI_METRIC and isinstance(fact.ref, FactRef)
    assert isinstance(fact.value, FrozenJson) and hash(fact) == hash(copy.deepcopy(fact))
    assert fact.json_value == {"value": 0.84, "ci": [0.83, 0.85], "n": 504473}
    assert Fact.from_json(fact.key, fact.to_json()) == fact
    for args in (("Bad-key", 1, "netspec.json#a", "count", "count"),
                 ("k", 1, "netspec.json#a", "tally", "count"), ("k", 1, "netspec.json#a", "count", "bars"),
                 ("k", 1, "netspec.json#a", "metric", "metric", "up")):
        try:
            Fact(*args)
        except ValueError:
            continue
        raise AssertionError(f"{args!r} made a fact")


def test_frozen_json_round_trip():
    doc = {"b": [1, {"c": None}], "a": {"x": [True, "y"]}, "e": {}, "l": []}
    frozen = freeze(doc)
    assert thaw(frozen) == doc and list(thaw(frozen)) == ["b", "a", "e", "l"], "order is kept"
    assert frozen == freeze({"a": {"x": [True, "y"]}, "l": [], "e": {}, "b": [1, {"c": None}]})
    assert hash(frozen) == hash(freeze(dict(reversed(list(doc.items())))))
    assert frozen["a"]["x"] == (True, "y") and frozen.get("zz") is None and dict(frozen)["e"] == FrozenJson()
    assert freeze([["a", 1]]) != freeze({"a": 1}), "a list of pairs is not an object"
    assert freeze(frozen) is frozen and thaw(freeze([frozen, {"k": frozen}])) == [doc, {"k": doc}]


def test_frozen_json_reads_as_a_mapping():
    frozen = freeze({"sizes": {"a": 1}, "b": 2, "breakdown": None})
    assert "sizes" in frozen and "breakdown" in frozen and "zz" not in frozen, "keys, not pairs"
    assert ("b", 2) in frozen and ("b", 3) not in frozen, "a non-string still asks for a pair"
    assert frozen.keys() == ("sizes", "b", "breakdown") and len(frozen) == 3
    assert frozen.values() == (freeze({"a": 1}), 2, None)
    assert frozen.items() == (("sizes", freeze({"a": 1})), ("b", 2), ("breakdown", None))
    assert dict(frozen.items()) == dict(frozen) and frozen["sizes"]["a"] == 1
    assert [k for k, _ in frozen] == list(frozen.keys()), "iteration still yields the pairs"
    assert "a" in freeze({"g": {"a": 1}})["g"] and freeze({}).items() == ()


def test_the_schema_example_round_trips():
    schema = json.loads((SCHEMAS / "show-facts-1.schema.json").read_text(encoding="utf-8"))
    example = schema["examples"][0]
    facts = FactSet.from_json(example)
    again = facts.to_json()
    def unnamed(doc):
        return {k: v for k, v in doc.items() if k != "names"}

    assert unnamed(again) == unnamed(example)
    assert set(again["names"]) == set(example["names"]) and again["names"] == sorted(again["names"])
    assert FactSet.from_json(again) == facts and hash(facts) == hash(FactSet.from_json(again))
    assert dataclasses.replace(facts, names=facts.names | {"R9"}).to_json()["groups"] == example["groups"]
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        raise AssertionError("jsonschema is needed to validate vega/show-facts@1")
    assert not list(Draft202012Validator(schema).iter_errors(again))
    assert facts.get("source_rows").value == 2830743 and "source_rows" in facts.keys()
    try:
        facts.get("no_such_fact")
    except FactMissing:
        return
    raise AssertionError("an unknown key was found")


def test_problems_and_paths():
    problem = Problem("V2", "authored.headline", "a bare number")
    assert str(problem) == "ERROR: V2 authored.headline: a bare number"
    assert str(Problem("V2", "t", "number word", "WARN")).startswith("WARN: V2 t:")
    paths = ShowPaths.of(Path("/b"))
    assert (paths.showcase, paths.building, paths.previous) == (
        Path("/b/showcase"), Path("/b/showcase.building"), Path("/b/showcase.prev"))
    assert paths.manifest == Path("/b/showcase/SHOWCASE.json")
    assert paths.facts == Path("/b/showcase/plan/facts.json")
    assert paths.plan.name == "story-plan.json" and paths.story.name == "story.json"
    assert paths.markers.parent == paths.work and paths.brief == Path("/b/showcase/docs/show-brief.md")


def test_write_json_is_atomic_and_digested():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "plan" / "facts.json"
        digest = write_json(target, {"a": "é", "n": 1})
        assert digest == digest_file(target) == digest_bytes(target.read_bytes())
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": "é", "n": 1}
        assert sorted(p.name for p in target.parent.iterdir()) == ["facts.json"]


def test_write_json_gives_the_mode_of_a_plain_open():
    """The serving image reads the root-owned bundle as an unprivileged user: a file the stage
    writes is as readable as one `open()` makes (0666 less the umask), never mkstemp's 0600."""
    def mode(path):
        return stat.S_IMODE(os.stat(path).st_mode)

    with tempfile.TemporaryDirectory() as tmp:
        for umask in (0o022, 0o002, 0o077):
            old = os.umask(umask)
            try:
                target, plain = Path(tmp) / f"SHOWCASE.{umask:o}.json", Path(tmp) / f"plain.{umask:o}"
                write_json(target, {"schema": "x"})
                write_json(target, {"schema": "y"})  # a rewrite keeps the mode too
                with plain.open("w", encoding="utf-8") as fh:
                    fh.write("{}")
            finally:
                os.umask(old)
            assert mode(target) == mode(plain) == 0o666 & ~umask, (oct(umask), oct(mode(target)))
        target = Path(tmp) / "SHOWCASE.22.json"
        assert mode(target) & (stat.S_IRGRP | stat.S_IROTH) == stat.S_IRGRP | stat.S_IROTH


def _task(**change):
    doc = {"kind": "multiclass", "base": None, "goal": "predict", "sample_shape": "row",
           "graph_level": None, "modality": "row", "family": "mlp-plain", "view_key": "classes",
           "label": "Label", "classes": ["BENIGN", "Bot"], "outcomes": None, "primary": "f1_macro",
           "direction": "max"}
    doc.update(change)
    return doc


def test_task_shape_is_validated():
    task = TaskShape.from_json(_task())
    assert task.to_json() == _task() and task.classes == ("BENIGN", "Bot")
    assert TaskShape(**{**_task(), "modality": None}).modality == "row", "modality defaults"
    spatial = TaskShape.from_json(_task(modality="spatial-row", kind="forecast", view_key="horizon"))
    assert spatial.modality == "spatial-row"
    bad = [{}, {k: v for k, v in _task().items() if k != "direction"}, {**_task(), "extra": 1},
           _task(kind="clustering"), _task(goal=None), _task(sample_shape="image"),
           _task(graph_level="edge"), _task(modality="text"), _task(family=""),
           _task(view_key=None), _task(primary=None), _task(primary="F1 macro"),
           _task(direction="up"), _task(direction=None), _task(base="forecast"), _task(label=3),
           _task(classes="BENIGN"), _task(outcomes={"a": 1})]
    for doc in bad:
        try:
            TaskShape.from_json(doc)
        except ValueError:
            continue
        raise AssertionError(f"{doc!r} made a task")
    try:
        TaskShape(kind="binary", goal="predict", sample_shape="row", family="mlp",
                  view_key="binary", primary="auc", direction="sideways")
    except ValueError as exc:
        assert "direction" in str(exc), exc
    else:
        raise AssertionError("a direction outside max/min made a task")


def test_bundle_records_read_pinned_refs():
    with tempfile.TemporaryDirectory() as tmp:
        root = _bundle(Path(tmp))
        store = BundleRecords.open(root)
        assert isinstance(store, RecordStore) and RecordStore in type(store).__mro__, "declared"
        assert store.read(FactRef.parse("records/shape.json#manifest.steps[id=S4].rows_after.all")) == 2522362
        assert store.read("netspec.json#blocks[id=B1].params.units") == 256
        assert store.read("netspec.json#task.classes.1") == "Bot"
        assert store.read("records/shape.json#manifest.steps[fit=true].id") == "S5"
        assert store.read("MANIFEST.json#serve.identity.noun") == "flow"
        assert store.manifest_digest() == digest_file(root / "MANIFEST.json")
        assert store.pinned() == ("netspec.json", "records/shape.json", "records/training.json")
        store.read("netspec.json#task")["classes"].append("mutated")
        assert store.read("netspec.json#task.classes") == ["BENIGN", "Bot"], "reads are copies"


def test_bundle_records_refuse_unresolvable_refs():
    with tempfile.TemporaryDirectory() as tmp:
        store = BundleRecords.open(_bundle(Path(tmp)))
        for ref in ("netspec.json#blocks[id=B9].params", "netspec.json#task.classes.7", "netspec.json#task.kind",
                    "netspec.json#blocks.first", "netspec.json#task.classes.0.name",
                    "records/analysis.json#drift", "schema.json#label", "report:net.forward.number"):
            try:
                store.read(ref)
            except FactMissing as exc:
                assert isinstance(exc, LookupError) and exc.ref == ref
                continue
            raise AssertionError(f"{ref} resolved")


def test_seeded_defect_22_an_altered_record():
    with tempfile.TemporaryDirectory() as tmp:
        root = _bundle(Path(tmp))
        with (root / "records" / "training.json").open("ab") as fh:
            fh.write(b" ")
        try:
            BundleRecords.open(root)
        except BundleAltered as exc:
            assert [m[0] for m in exc.mismatches] == ["records/training.json"], exc.mismatches
            assert "records/training.json" in str(exc) and "recorded" in str(exc)
        else:
            raise AssertionError("an altered record was read")
        (root / "records" / "training.json").unlink()
        try:
            BundleRecords.open(root)
        except BundleAltered as exc:
            assert exc.mismatches == (("records/training.json", exc.mismatches[0][1], None),)
        else:
            raise AssertionError("a bundle missing a pinned record was opened")


def test_the_cids_bundle_when_present():
    if not (CIDS_BUNDLE / "MANIFEST.json").is_file():
        print(f"    (skipped: {CIDS_BUNDLE} is absent)")
        return
    store = BundleRecords.open(CIDS_BUNDLE)
    assert store.read("records/geometry.json#source.rows") == 2830743
    assert store.read("records/shape.json#manifest.steps[id=S4].rows_after.all") == 2522362
    assert store.read("netspec.json#blocks[id=B1].params.units") == 256
    runs = store.read("records/training.json#experiments.runs")
    r7 = next(r for r in runs if r["id"] == "R7")
    assert store.read("records/training.json#experiments.runs[id=R7].primary") == r7["primary"]


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
