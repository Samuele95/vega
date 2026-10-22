#!/usr/bin/env python3
"""
test_artifact_roster.py — the blueprint's artifacts, accounted for (code-contract §3, C6).

    python3 -m pytest -q test_artifact_roster.py

The map's `domain.py` roster is derived from the blueprint, not from a list of the emitter's:

1. **The derivation** (`vega_roster`): every `pipeline.json.artifacts[]` entry is held (a name
   `domain.py` defines — a class or an alias) or declared external with its owner and where it
   lives; an artifact with no home, an external one `domain.py` defines, an owner that is not a
   stage of the chain (or is the forge itself), a declaration without `where`, a duplicate and
   a nameless entry are each refused by name.
2. **The emitter** writes the roster into `blueprint-map.json` — the domain row lists every held
   artifact in the blueprint's order, `backend_artifacts` only the backend's own the blueprint
   did not list, `external_artifacts` each external one with its owner — and refuses a
   blueprint whose artifacts are not accounted for before it writes anything.
3. **The R3 rehearsal's blueprint** (17 artifacts, the campaign among them) against the real
   `domain.py` template: 16 held, the campaign external once declared so, and refused (naming
   it) while it is not — the divergence /architect raised in R3 is now the emitter's refusal.
4. **The gate** (C6) reads the roster back: the emitted package passes; a map that drops a held
   artifact, lists one the blueprint does not, forgets an external one, records an external one
   the blueprint does not declare, or a blueprint artifact with no home is refused.

Stub templates (test_emit_layout's), no torch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))

import conformance  # noqa: E402
import render_templates as rt  # noqa: E402
import test_emit_layout as tel  # noqa: E402
import vega_roster as vr  # noqa: E402

emit = tel.emit
PACKAGE = tel.PACKAGE
TEMPLATE_DOMAIN = SCRIPTS / "backends" / "pytorch" / "templates" / "domain.py"

CAMPAIGN = {"owner": "/vega-train", "where": "the campaign directory (campaign.json, runs/<id>/)"}

# The 17 artifacts of the T5 rehearsal's blueprint (runs/final/R3/t5, landcover), as it drew
# them: every one but the campaign is a name the v2.3 domain.py defines.
R3_ARTIFACTS = (
    ("NetSpec", "value"), ("FeatureContract", "value"), ("Part", "value"),
    ("SamplePart", "value"), ("BatchPlan", "value"), ("RunContext", "value"),
    ("DesignView", "value"), ("DesignMatrix", "value"), ("GraphExposure", "value"),
    ("FitResult", "value"), ("Checkpoint", "value"), ("ModelVersion", "value"),
    ("Metrics", "value"), ("ScoredPart", "value"), ("Prediction", "value"),
    ("DriftReading", "value"), ("Campaign", "entity"))


def _pipeline(*artifacts):
    p = rt.reference_pipeline()
    p["artifacts"] = [dict(a) for a in artifacts]
    return p


def _art(name, kind="value", external=None):
    entry = {"name": name, "kind": kind, "identity": "digest"}
    if external is not None:
        entry["external"] = external
    return entry


def _problems(pipeline, defined):
    return vr.roster(pipeline, frozenset(defined))[1]


# ================================================================ 1. the derivation

def test_defined_names_reads_classes_aliases_and_annotations():
    names = vr.defined_names(
        "from dataclasses import dataclass\nimport json\n\n"
        "@dataclass(frozen=True)\nclass Checkpoint:\n    digest: str\n\n"
        "ModelVersion = Checkpoint\nLimit: int = 3\n_Private = 1\n"
        "def helper():\n    return 1\n")
    assert {"Checkpoint", "ModelVersion", "Limit", "helper"} <= names
    assert "_Private" not in names and "dataclass" not in names and "json" not in names


def test_roster_keeps_the_blueprints_order_and_splits_held_from_external():
    found, problems = vr.roster(
        _pipeline(_art("Metrics"), _art("Campaign", "entity", CAMPAIGN), _art("NetSpec")),
        frozenset({"NetSpec", "Metrics", "Other"}))
    assert problems == []
    assert found.held == ("Metrics", "NetSpec")
    assert found.external_rows() == [{"name": "Campaign", "kind": "entity",
                                      "owner": "/vega-train", "where": CAMPAIGN["where"],
                                      "source": "blueprint:artifacts"}]


def test_an_artifact_with_no_home_is_refused_by_name():
    (msg,) = _problems(_pipeline(_art("Verdict")), {"NetSpec"})
    assert "'Verdict'" in msg and "not defined in domain.py" in msg
    assert "external" in msg and "--concern" in msg


def test_an_external_artifact_domain_defines_is_refused():
    (msg,) = _problems(_pipeline(_art("Checkpoint", external=CAMPAIGN)), {"Checkpoint"})
    assert "'Checkpoint'" in msg and "domain.py defines it" in msg


def test_the_external_declaration_is_checked():
    cases = {
        "an owner outside the chain": ({"owner": "/somewhere", "where": "x"}, "/somewhere"),
        "the forge as owner": ({"owner": "/vega-forge", "where": "x"}, "holds nothing outside"),
        "no where": ({"owner": "/vega-train"}, "without `where`"),
        "a blank where": ({"owner": "/vega-train", "where": "  "}, "without `where`"),
        "not an object": ("/vega-train", "is an object"),
    }
    for label, (decl, needle) in cases.items():
        (msg,) = _problems(_pipeline(_art("Campaign", "entity", decl)), set())
        assert needle in msg, f"{label}: {msg}"


def test_duplicate_and_nameless_entries_are_refused():
    problems = _problems(_pipeline(_art("NetSpec"), _art("NetSpec"), {"kind": "value"}),
                         {"NetSpec"})
    assert any("twice" in p for p in problems)
    assert any("no name" in p for p in problems)


def test_every_owner_is_a_chain_stage_and_the_forge_is_not_one():
    assert "/vega-forge" not in vr.EXTERNAL_OWNERS
    assert {"/vega-train", "/vega-eval", "/vega-pack", "/dataset-shaper"} <= set(
        vr.EXTERNAL_OWNERS)


# ================================================================ 2. the emitter

def _domain_row(bmap):
    return next(e for e in bmap["entries"] if e["file"] == "domain.py")


def test_the_map_carries_the_blueprints_roster(tmp_path):
    out, rec = tel.emit_stub(tmp_path, rt.reference_spec("row"))
    bmap = json.loads((out / PACKAGE / "blueprint-map.json").read_text(encoding="utf-8"))
    assert bmap == rec["blueprint_map"]
    row = _domain_row(bmap)
    held = [a["name"] for a in rt.reference_pipeline()["artifacts"] if "external" not in a]
    assert row["artifacts"] == held
    domain = (out / PACKAGE / "domain.py").read_text(encoding="utf-8")
    assert row["backend_artifacts"] == [h.name for h in vr.held_vocabulary(domain)
                                        if h.name not in held]
    assert row["backend_artifacts"] and not set(row["backend_artifacts"]) & set(held)
    ports = next(e for e in bmap["entries"] if e["file"] == "ports.py")
    assert ports["backend_protocols"] == ["ModelBuilder", "TrainedNetwork", "TrainingData",
                                          "Callbacks", "ScoredModel", "ArtifactExporter"]
    assert [e["name"] for e in bmap["external_artifacts"]] == ["Campaign"]
    assert bmap["external_artifacts"][0]["owner"] == "/vega-train"


def test_a_blueprint_artifact_the_backend_also_names_is_listed_once(tmp_path):
    src = tel.write_stub_backend(tmp_path / "stub")
    dom = src.templates / "domain.py"
    dom.write_text(dom.read_text(encoding="utf-8")
                   + "\n\n@dataclass(frozen=True)\nclass FitResult:\n    epochs: int\n",
                   encoding="utf-8")
    pipeline = rt.reference_pipeline()
    pipeline["artifacts"].append(_art("FitResult"))
    out, rec = tel.emit_stub(tmp_path, rt.reference_spec("row"), pipeline, sources=src)
    row = _domain_row(rec["blueprint_map"])
    assert row["artifacts"][-1] == "FitResult"
    assert "FitResult" not in row["backend_artifacts"] and row["backend_artifacts"]


def test_the_emitter_refuses_an_artifact_with_no_home_before_writing(tmp_path):
    pipeline = rt.reference_pipeline()
    for a in pipeline["artifacts"]:
        a.pop("external", None)
    tel.raises(lambda: tel.emit_stub(tmp_path, rt.reference_spec("row"), pipeline),
               "'Campaign'", "not declared external", "code-contract §3")
    assert not (tmp_path / "run" / PACKAGE).exists()


def test_the_emitter_refuses_a_false_external_declaration(tmp_path):
    pipeline = rt.reference_pipeline()
    pipeline["artifacts"][0]["external"] = CAMPAIGN          # NetSpec, which domain.py defines
    tel.raises(lambda: tel.emit_stub(tmp_path, rt.reference_spec("row"), pipeline),
               "'NetSpec'", "domain.py defines it")


# ================================================================ 3. the R3 blueprint

def test_the_r3_rehearsal_blueprint_against_the_real_domain_template():
    defined = vr.defined_names(TEMPLATE_DOMAIN.read_text(encoding="utf-8"))
    names = [n for n, _ in R3_ARTIFACTS]
    as_drawn = _pipeline(*(_art(n, k) for n, k in R3_ARTIFACTS))
    (msg,) = vr.roster(as_drawn, defined)[1]
    assert "'Campaign'" in msg
    declared = _pipeline(*(_art(n, k, CAMPAIGN if n == "Campaign" else None)
                           for n, k in R3_ARTIFACTS))
    found, problems = vr.roster(declared, defined)
    assert problems == []
    assert list(found.held) == names[:-1]          # all 16, ModelVersion and SamplePart included
    assert [e.name for e in found.external] == ["Campaign"]


# ================================================================ 4. the gate (C6)

def _c6_errors(root, pipeline, bmap=None):
    bmap = bmap or json.loads((root / "blueprint-map.json").read_text(encoding="utf-8"))
    pkg, err = conformance.Package.read(root, bmap, pipeline)
    assert err is None, err
    return [m.message for m in conformance.c6_domain_is_frozen(pkg)
            if str(m.level).lower().endswith("error")]


def test_c6_passes_the_emitted_roster_and_refuses_every_tampering(tmp_path):
    out, rec = tel.emit_stub(tmp_path, rt.reference_spec("row"))
    root, pipeline = out / PACKAGE, rt.reference_pipeline()
    assert _c6_errors(root, pipeline) == []

    def tampered(change):
        bmap = json.loads(json.dumps(rec["blueprint_map"]))
        change(bmap)
        return _c6_errors(root, pipeline, bmap)

    msgs = tampered(lambda m: _domain_row(m)["artifacts"].remove("Metrics"))
    assert any("does not list" in m and "Metrics" in m for m in msgs), msgs
    assert any("declares neither" in m and "Metrics" in m for m in msgs), msgs
    # every value object domain.py holds is declared once: the backend's own under
    # backend_artifacts (T5 round 2: nine value objects the map named nowhere)
    own = _domain_row(rec["blueprint_map"])["backend_artifacts"]
    (msg,) = tampered(lambda m: _domain_row(m)["backend_artifacts"].remove(own[0]))
    assert own[0] in msg and "declares neither" in msg
    (msg,) = tampered(lambda m: _domain_row(m)["backend_artifacts"].append("Seal"))
    assert "'Seal'" not in msg and "Seal" in msg and "does not hold" in msg
    (msg,) = tampered(lambda m: _domain_row(m)["backend_artifacts"].append("Metrics"))
    assert "Metrics" in msg and "one source per name" in msg
    (msg,) = tampered(lambda m: _domain_row(m)["artifacts"].append("Campaign"))
    assert "Campaign" in msg and "the blueprint does not" in msg
    (msg,) = tampered(lambda m: m.__setitem__("external_artifacts", []))
    assert "'Campaign'" in msg and "external_artifacts" in msg
    (msg,) = tampered(lambda m: m["external_artifacts"].append(
        {"name": "Seal", "owner": "/vega-eval", "where": "test_seal.json"}))
    assert "'Seal'" in msg and "does not declare it external" in msg
    (msg,) = tampered(lambda m: m["external_artifacts"][0].__setitem__("owner", "/vega-eval"))
    assert "'Campaign'" in msg


def test_c6_refuses_a_blueprint_artifact_with_no_home(tmp_path):
    out, _rec = tel.emit_stub(tmp_path, rt.reference_spec("row"))
    pipeline = rt.reference_pipeline()
    pipeline["artifacts"].append(_art("Verdict"))
    errors = _c6_errors(out / PACKAGE, pipeline)
    assert any("'Verdict'" in e and "not declared external" in e for e in errors), errors


# ================================================================ 5. the vocabulary the blueprint reads

def test_every_held_name_the_blueprint_is_offered_is_one_the_roster_holds():
    """held_vocabulary (held_artifacts.py) is what the blueprint patch tells a pipeline
    blueprint to name its held artifacts by: each name it offers is held by the roster against
    the real template, and the exceptions and constants domain.py also defines are not offered."""
    text = TEMPLATE_DOMAIN.read_text(encoding="utf-8")
    held = vr.held_vocabulary(text)
    names = [h.name for h in held]
    assert len(names) == len(set(names)) and names
    roster, problems = vr.roster(_pipeline(*(_art(n) for n in names)), vr.defined_names(text))
    assert problems == [] and list(roster.held) == names
    assert {"SealedPartError", "UnsupportedShapeError", "CARRIED_SHAPES"}.isdisjoint(names)
    kinds = {h.name: h.kind for h in held}
    assert kinds["ModelVersion"] == "alias of Checkpoint" and kinds["TaskKind"] == "enum"
    assert kinds["SamplePart"] == "union"
    assert kinds["NetSpec"] == "value object"
    assert {n for n, _ in R3_ARTIFACTS if n != "Campaign"} <= set(names)


def test_held_artifacts_cli_lists_the_vocabulary():
    import subprocess
    run = subprocess.run([sys.executable, str(SCRIPTS / "held_artifacts.py"), "--json"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    rows = json.loads(run.stdout)
    expected = [h.name for h in vr.held_vocabulary(TEMPLATE_DOMAIN.read_text(encoding="utf-8"))]
    by_shape = {}
    for shape in ("row", "window", "graph"):
        run = subprocess.run([sys.executable, str(SCRIPTS / "held_artifacts.py"), "--shape", shape,
                              "--json"], capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
        by_shape[shape] = {r["name"] for r in json.loads(run.stdout)}
    # a package carries one shape's values: a row blueprint is never offered a window's
    # (the round 1 blueprint held UnscoredRow, which a row package no longer defines)
    assert "UnscoredRow" in by_shape["window"] and "UnscoredRow" not in by_shape["row"]
    assert "GraphSpec" in by_shape["graph"] and "GraphSpec" not in by_shape["window"]
    common = set.intersection(*by_shape.values())
    assert [r["name"] for r in rows] == [n for n in expected if n in common]
    run = subprocess.run([sys.executable, str(SCRIPTS / "held_artifacts.py"), "--backend", "nope"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 2 and "ERROR: no domain.py template" in run.stdout



# ================================================================ 6. the v2.2 names, kept only where used

def test_a_compat_name_is_emitted_only_for_a_blueprint_that_uses_it():
    """`TrafficClass = ClassValue` and `ResidentPart = TensorPart` are v2.2 names kept for one
    release (BUILD-PLAN §1): a package whose blueprint names neither carries neither (the T5
    round 1 audit read `TrafficClass` in a land-cover package as another project's
    vocabulary), and a blueprint that still names one finds it held."""
    text = TEMPLATE_DOMAIN.read_text(encoding="utf-8")
    plain = rt.reference_pipeline()
    words = emit.blueprint_words(plain)
    assert "TrafficClass" not in words and "ResidentPart" not in words
    rendered = emit.select_compat(text, words)
    names = vr.defined_names(rendered)
    assert "TrafficClass" not in names and "ResidentPart" not in names
    assert "# >>> compat" not in rendered and "# <<< compat" not in rendered
    assert {"ClassValue", "TensorPart", "Checkpoint"} <= names and "ModelVersion" not in names
    old = _pipeline(_art("TrafficClass"), _art("NetSpec"))
    old["ports"][0]["signature"] = ["def load(self, spec: NetSpec) -> ResidentPart: ..."]
    kept = vr.defined_names(emit.select_compat(text, emit.blueprint_words(old)))
    assert {"TrafficClass", "ResidentPart"} <= kept
    roster, problems = vr.roster(old, kept)
    assert problems == [] and roster.held == ("TrafficClass", "NetSpec")
    assert vr.roster(old, names)[1], "without the compat name the blueprint's artifact has no home"
    assert emit.select_compat(text, None).count("TrafficClass = ClassValue") == 1


def test_compat_sections_are_refused_when_malformed():
    import pytest
    with pytest.raises(emit.EmitError, match="never closed"):
        emit.select_compat("# >>> compat: A\nA = B\n", frozenset())
    with pytest.raises(emit.EmitError, match="closes no section"):
        emit.select_compat("A = B\n# <<< compat\n", frozenset())
