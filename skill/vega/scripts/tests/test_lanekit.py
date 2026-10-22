"""
Unit tests of the lane program itself (T01), part 1: the registry and its validation, the
consumed names and their drift, the stage plan and the run_matrix.py CLI. The lane runs are
test_lanekit_runs.py; the runner, observed.json, the service, the skills tree, routing and the
stage assertions are test_lanekit_parts.py.

    python -m pytest -q vega/scripts/tests/test_lanekit.py vega/scripts/tests/test_lanekit_runs.py \
        vega/scripts/tests/test_lanekit_parts.py

Nothing here trains a network: the lanes these tests drive are synthetic stage functions over
temporary directories, so the suites run in seconds. The real lanes run under matrix.sh.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lanekit_testkit import HERE, PY, REPO, catalogs, doc, errors_for, lane_of  # noqa: E402,F401 (fixtures)
from lanekit import contracts, drift, observe, registry, stages, validation  # noqa: E402
from lanekit.registry import Lane  # noqa: E402


def test_the_schemas_are_draft_2020_12():
    import jsonschema
    for path in (registry.LANES_SCHEMA, observe.SCHEMA_FILE):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        jsonschema.Draft202012Validator.check_schema(schema)


def test_lanes_json_validates(doc, catalogs):
    flakes = json.loads(registry.FLAKES_FILE.read_text(encoding="utf-8"))
    report = registry.validate(doc, catalogs, flakes)
    assert report.errors == []
    ids = [e["id"] for e in doc["lanes"]]
    assert ids == [f"L{i:02d}" for i in range(1, 24)] + ["G00"]


def test_registry_declares_the_plan_tiers(doc):
    pr = sorted(e["id"] for e in doc["lanes"] if "pr" in e["tiers"])
    assert pr == ["L01", "L02", "L05", "L10", "L17", "L19", "L20", "L22"]       # BUILD-PLAN §4.8
    assert [e["id"] for e in doc["lanes"] if "nightly" not in e["tiers"]] == []
    assert lane_of(doc, "G00")["tiers"] == ["nightly"]


def test_every_family_task_and_block_is_declared(doc, catalogs):
    lanes = [Lane(e["id"], e) for e in doc["lanes"] if e["driver"] != "golden"]
    assert {l.family for l in lanes} == set(catalogs.forge["families"])
    assert {l.task for l in lanes} == set(catalogs.forge["tasks"])
    assert {t for l in lanes for t in l.flat_block_types()} == set(catalogs.forge["blocks"])


WINDOW = {"length": 8, "stride": 1, "horizon": None, "series_by": None, "order_by": "ts",
          "context": "previous-part", "autoregressive": None, "season": None}


def with_floor_shapes(catalogs: registry.Catalogs, family: str, floor: list[str] | None) -> registry.Catalogs:
    """The catalogs with `families.<family>.floor_shapes` set to `floor`, or removed when None:
    the shape tests fix the floor themselves instead of reading it from the live catalog, whose
    floor families R-A13 (K1) sets."""
    forge = copy.deepcopy(dict(catalogs.forge))
    entry = dict(forge["families"][family])
    entry.pop("floor_shapes", None)
    if floor is not None:
        entry["floor_shapes"] = list(floor)
    forge["families"][family] = entry
    return registry.Catalogs(forge, catalogs.coverage, catalogs.show, catalogs.generators)


def test_shape_must_be_the_family_sample_shape(doc, catalogs):
    lane = lane_of(doc, "L02")                       # mlp-plain, row; no floor_shapes in any catalog
    lane["shape"], lane["window"] = "window", dict(WINDOW)
    needle = "L02: shape 'window' is not family mlp-plain's sample_shape"
    assert any(needle in e for e in errors_for(doc, catalogs)), errors_for(doc, catalogs)
    assert any(needle in e for e in errors_for(doc, with_floor_shapes(catalogs, "mlp-plain", None)))


def test_floor_shapes_admit_a_floor_family(doc, catalogs):
    lane = lane_of(doc, "L04")                       # linear, row
    lane["shape"], lane["window"] = "window", dict(WINDOW)
    floor = with_floor_shapes(catalogs, "linear", ["row", "window", "graph"])
    no_floor = with_floor_shapes(catalogs, "linear", None)
    assert not [e for e in errors_for(doc, floor) if "sample_shape" in e]
    assert any("L04: shape 'window' is not family linear's sample_shape" in e
               for e in errors_for(doc, no_floor)), errors_for(doc, no_floor)
    assert any("L04: shape 'window'" in e for e in errors_for(doc, with_floor_shapes(catalogs, "linear", ["row", "graph"])))


@pytest.mark.parametrize("edit, needle", [
    (lambda l: l.update(family="no-such-family"), "family 'no-such-family' is not in catalog.json"),
    (lambda l: l["loss"].update(name="mse"), "loss 'mse' is not in tasks.binary.losses_allowed"),
    (lambda l: l.update(optimizer="lion"), "optimizer 'lion' is not in coverage.json"),
    (lambda l: l["roles"].pop("1"), "roles cover ['0'], the classes are ['0', '1']"),
    (lambda l: l.update(stances=[]), "an alert role needs a user:/show: stance"),
    (lambda l: l["lexicon"].pop("source"), "lexicon keys"),
    (lambda l: l["campaign"].update(rungs=["R1", "R0"]), "campaign.rungs starts with R0"),
    (lambda l: l["blocks"].append({"id": "B9", "type": "warp", "params": {}}), "block type 'warp'"),
    (lambda l: l["blocks"][0]["params"].pop("columns"), "embedding needs param 'columns'"),
])
def test_cross_checks_refuse(doc, catalogs, edit, needle):
    edit(lane_of(doc, "L01"))
    assert any(needle in e for e in errors_for(doc, catalogs)), errors_for(doc, catalogs)


def test_graph_rules(doc, catalogs):
    lane = lane_of(doc, "L23")
    lane["expect"].pop("pack_refusal")
    lane["graph"]["protocol_source"] = "vega:default"
    errs = errors_for(doc, catalogs)
    assert any("transductive needs a user: source" in e for e in errs)
    assert any("stops at eval and asserts pack's refusal" in e for e in errs)
    lane = lane_of(doc, "L22")
    lane["blocks"] = [b for b in lane["blocks"] if b["type"] != "graph_pool"]
    assert any("level graph needs a graph_pool block" in e for e in errors_for(doc, catalogs))


def test_motif_departure_needs_a_stance(doc, catalogs):
    lane_of(doc, "L04")["stances"] = []
    assert any("motif 'grid' departs from the base 'points'" in e for e in errors_for(doc, catalogs))


def test_schema_refuses_an_unknown_field(doc, catalogs):
    lane_of(doc, "L02")["colour"] = "teal"
    assert any(e.startswith("schema: lanes/1") and "colour" in e for e in errors_for(doc, catalogs))


def test_flake_entry_without_upstream_fails():
    flakes = json.loads(registry.FLAKES_FILE.read_text(encoding="utf-8"))
    assert validation.flake_problems(flakes) == []
    flakes["flakes"][0]["upstream"] = ""
    assert any("names no upstream fix" in p for p in validation.flake_problems(flakes))


def test_lexicon_required_follows_the_show_catalog():
    show = {"lexicon": {"required": {"row": ["row", "feature"], "graph:node": ["row", "node"]},
                        "articles": {"default": ["a", "an"], "source": ["a", "an", "the"]}}}
    assert registry.required_lexicon(show, "row") == ("row", "feature")
    assert registry.required_lexicon(show, "graph-node") == ("row", "node")        # K3's key form
    assert registry.required_lexicon(None, "graph-graph") == contracts.LEXICON_REQUIRED["graph-graph"]
    assert registry.lexicon_articles(show, "source") == ("a", "an", "the")
    assert registry.lexicon_articles(None, "row") == ("a", "an")


@pytest.mark.parametrize("edit, needle", [
    (lambda lex: lex["source"].pop("many"), "lexicon.source lacks many"),
    (lambda lex: lex["row"].update(article="the"), "lexicon.row: article 'the' is not one of ['a', 'an']"),
    (lambda lex: lex["feature"].update(many="Inputs"), "lexicon.feature words must match"),
])
def test_lexicon_entries_are_complete(doc, catalogs, edit, needle):
    edit(lane_of(doc, "L02")["lexicon"])
    assert any(needle in e for e in errors_for(doc, catalogs)), errors_for(doc, catalogs)


# --------------------------------------------------------------------------- consumed names
def test_output_fields_read_field_kinds():
    manifest = {"output": {"fields": ["label", "p_0", "p_1", "score", "prediction", "warnings"],
                           "field_kinds": {"label": {"kind": "class"}, "p_0": {"kind": "probability", "of": "0"},
                                           "p_1": {"kind": "probability", "of": "1"},
                                           "score": {"kind": "raw", "of": "1"},
                                           "prediction": {"kind": "decision"},
                                           "warnings": {"kind": "warnings"}}}}
    assert contracts.OUTPUT_FIELDS(manifest) == ("label", "probabilities", "warnings")
    assert contracts.OUTPUT_FIELDS(manifest, include=("logits",)) == ("label", "probabilities", "logits", "warnings")
    assert contracts.output_record_problems(manifest) == []
    manifest["output"]["compared"] = [["logit_x", "score_0"]]
    assert contracts.output_record_problems(manifest) == ["compared/decided field 'logit_x' is not in fields"]
    assert contracts.OUTPUT_FIELDS({"output": {"fields": ["score", "prediction"]}}) == ("score", "prediction", "warnings")


def test_contracts_name_what_the_plan_fixes():
    assert contracts.SHOWCASE_MANIFEST == "SHOWCASE.json"
    assert contracts.VIEWS == ("start", "context", "network", "try", "many", "results", "code")
    assert contracts.expected_views(2)[3:5] == ("tried", "weights") and "tried" not in contracts.expected_views(1)
    assert set(contracts.READING_LINK.values()) <= set(contracts.LINKS)
    assert contracts.SHAPER_GRAPH_GATES[-1] == "SG7" and "SS9" in contracts.SLOT_GATES and "IN7" in contracts.SLOT_GATES
    assert contracts.BATCH_KEY_ALIASES["graph_id"] == "graph_index"


def test_no_drift_between_contracts_and_the_catalogs():
    """The copies contracts.py keeps equal the catalogs that own them, wherever those catalogs
    already hold the key (K1's links, K2's field kinds, K3's motifs, roles, lexicon, views,
    slot and probe)."""
    assert drift.drift(REPO) == []


def test_drift_is_found(tmp_path):
    (tmp_path / "vega-pack" / "scripts").mkdir(parents=True)
    (tmp_path / "vega-show" / "scripts").mkdir(parents=True)
    kinds = {k: {"key": v, "shown": f"include={contracts.FIELD_KIND_INCLUDE[k]}"
                 if k in contracts.FIELD_KIND_INCLUDE else "always"}
             for k, v in contracts.FIELD_KIND_KEYS.items()}
    kinds["class"]["key"] = "klass"
    (tmp_path / contracts.PACK_CATALOG).write_text(json.dumps({"field_kinds": kinds}))
    (tmp_path / contracts.SHOW_CATALOG).write_text(json.dumps(
        {"views": {"ids": ["start", "network"], "sub_views": {}, "aliases": {}},
         "lexicon": {"keys": list(contracts.LEXICON_KEYS),
                     "required": {"row": ["row", "feature", "source"], "window": ["row"],
                                  "graph:node": list(contracts.LEXICON_REQUIRED["graph-node"]),
                                  "graph:graph": list(contracts.LEXICON_REQUIRED["graph-graph"])}}}))
    found = drift.drift(tmp_path)
    assert any(f.startswith("field_kinds kind -> key") and "klass" in f for f in found)
    assert any(f.startswith("views.ids") for f in found)
    assert any(f.startswith("lexicon.required") for f in found)
    assert not any(f.startswith("lexicon.keys") for f in found)


def test_the_stage_plan_per_driver(doc):
    plans = {e["id"]: stages.stage_plan(Lane(e["id"], e)) for e in doc["lanes"]}
    assert plans["L01"][:4] == ("forge", "train", "eval", "pack") and plans["L01"][-1] == "page"
    assert plans["L23"] == ("fixture", "forge", "train", "eval")
    assert plans["G00"][0] == "golden"


def test_l01_runs_the_acceptance_scripts_in_lane_mode(monkeypatch, tmp_path):
    """L01's forge…pack stages run the acceptance scripts with VEGA_ACCEPTANCE_SUITES=lane, which
    leaves the scripts' own suites (no work dir read) to the standalone run and the tiers; every
    script that holds suites honours the variable."""
    seen = []
    monkeypatch.setattr(stages, "flake_record_path", lambda ctx: tmp_path / "flakes.jsonl")
    for name in ("count_inner_calls", "record_inner_flakes", "record_flakes_seen"):
        monkeypatch.setattr(stages, name, lambda *a: 0)
    monkeypatch.setattr(stages, "assert_packed_blocks", lambda *a: "blocks")
    monkeypatch.setattr(stages, "record_pin", lambda *a: None)
    run = lambda argv, retry, env_extra: seen.append(env_extra) or type(
        "R", (), {"output": "acceptance passed", "markers": type("M", (), {"warn": 0})})()
    for stage in ("forge", "pack"):
        ctx = type("C", (), {"stage": stage, "work": tmp_path, "path": lambda self, p: REPO / p,
                             "run": lambda self, *a, **k: run(*a, **k)})()
        stages.run_acceptance(None, None, ctx)
    name, value = contracts.ACCEPTANCE_SUITES
    assert [e.get(name) for e in seen] == [value, value] and seen[1][contracts.PACK_EXTRA] == "--serve"
    for script in (contracts.ACCEPTANCE["forge"], contracts.ACCEPTANCE["pack"]):
        text = (REPO / script).read_text(encoding="utf-8")
        assert f'SUITES="${{{name}:-run}}"' in text and f'[ "$SUITES" = {value} ]' in text


def test_run_matrix_cli(tmp_path):
    out = subprocess.run([PY, str(HERE / "run_matrix.py"), "validate"], capture_output=True, text=True)
    assert out.returncode == 0 and "lanes.json validates" in out.stdout
    out = subprocess.run([PY, str(HERE / "run_matrix.py"), "lanes", "--lanes", "L99",
                          "--matrix", str(tmp_path / "m")], capture_output=True, text=True)
    assert out.returncode == 2 and "unknown lane(s): L99" in out.stdout
    inside = subprocess.run([PY, str(HERE / "run_matrix.py"), "lanes", "--lanes", "L01",
                             "--matrix", str(HERE / "m")], capture_output=True, text=True)
    assert inside.returncode != 0 and "inside the repository" in (inside.stdout + inside.stderr)
