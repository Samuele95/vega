"""
Unit tests of lanekit.authoring (T03; D6 §2.2 "authoring writers (no digits)", BUILD-PLAN R-E5):
the judgement the lane stages publish in place of a model, and the `show:` stance lines.

    python -m pytest -q vega/scripts/tests/test_lanekit_authoring.py

Closing tests of T03 held here:
- authoring files carry no digit outside placeholders: the forge, train and pack writers on
  records full of digits, and the story writer on the drafts of seven lanes (every shape, a
  proposed and an emitted report role map, an anomaly task, a monitor), each document
  accepted by the validator of the apply script that reads it (or by the story-plan@2
  schema and the plan rules the contract states, for the story);
- `show:` stance lines written for the lanes declaring alert — every lane of lanes.json.

`VEGA_AUTHORING_WORK=<a finished pack acceptance work dir>` adds the end-to-end check: the
three layer writers on the real records, through the real apply scripts, validators and
smoke tests (headless Chrome, `$CHROME`).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import authoring, registry  # noqa: E402
from lanekit.registry import Lane  # noqa: E402

REPO = HERE.parents[2]
PY = sys.executable
SHOW = REPO / "vega-show"
SHOW_CATALOG = json.loads((SHOW / "scripts" / "catalog.json").read_text(encoding="utf-8"))
FORGE_CATALOG = json.loads((REPO / "vega-forge" / "scripts" / "catalog.json").read_text(encoding="utf-8"))
LANES = registry.load_lanes()


def load_script(rel: str, name: str):
    """A skill script as a module (the apply scripts' validators are what the layers must pass)."""
    path = REPO / rel
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module                  # dataclasses resolve their module by name
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


# =========================================================================== fixtures: records with digits
def spec_for(lane: Lane) -> dict:
    """A netspec in the forge's shape for a lane, its sources and values full of digits."""
    raw = lane.raw
    blocks = []
    for block in raw["blocks"]:
        source = "vega:default" if block["type"] not in ("head", "residual") else "analysis:F5"
        blocks.append(dict(copy.deepcopy(block), source=source, catalog="defaults/sizing",
                           rationale="width 64 for 500 rows: clip(2*9, 32, 256)"))
    sample = {"shape": raw["shape"]}
    if raw["shape"] == "window":
        sample.update(window_sample(raw))
    if raw["shape"] == "graph":
        sample.update(graph_sample(raw))
    return {
        "schema": "vega/netspec@1", "seed": 7,
        "task": {"kind": raw["task"], "classes": (raw.get("target") or {}).get("classes")},
        "data_contract": {"sample": sample, "input_dim": 9},
        "family": {"name": raw["family"], "source": "user:stance/family-2026-09-05", "rationale": "F5 at 0.83"},
        "blocks": blocks,
        "loss": {"name": raw["loss"]["name"], "params": raw["loss"]["params"], "source": "analysis:F2"},
        "metrics": {"primary": "pr_auc", "secondary": ["roc_auc", "f1", "top5"], "source": "vega:default",
                    "catalog": "tasks/binary"},
        "optimizer": {"name": "adamw", "params": {"lr": 0.001}, "source": "vega:default", "catalog": "defaults/optimizer"},
        "schedule": {"name": "cosine", "params": {"warmup_epochs": 3}, "source": "vega:default",
                     "catalog": "defaults/schedule"},
        "regularization": [{"kind": "weight_decay", "value": 1e-4, "source": "vega:default",
                            "catalog": "defaults/weight_decay"},
                           {"kind": "dropout", "value": 0.2, "source": "analysis:F4"},
                           {"kind": "grad_clip", "value": 1.0, "source": "vega:default", "catalog": "defaults/grad_clip"}],
        "early_stopping": {"patience": 10, "source": "vega:default", "catalog": "defaults/early_stopping"},
        "init": {"scheme": "kaiming_uniform", "source": "vega:default", "catalog": "defaults/init"},
        "training": {"batch_size": 64, "max_epochs": 200, "source": "vega:default",
                     "catalog": ["defaults/batch_size", "defaults/max_epochs", "defaults/device"]},
        "backend": {"name": "pytorch"},
        "outputs": {"package": "orders_pipeline", "run_dir": "vega-runs/2026-09-05T11-03Z/"}}


def window_sample(raw: dict) -> dict:
    """The sample keys netspec-contract §1 gives a window, with the provenance keys it names
    (`context_source`, `autoregressive_source`, `series_source`): the series key the lane
    names comes from the shaping record, the other points from the catalog's defaults."""
    w = raw["window"]
    sample = {k: w[k] for k in ("length", "stride", "series_by", "order_by", "context", "season")}
    sample.update(source="analysis:context.time", context_source="vega:default",
                  context_catalog="defaults/window_context", series_source="shape:S8" if w["series_by"] else None)
    if raw["task"] == "forecast":
        sample.update(horizon=w["horizon"], autoregressive=w["autoregressive"],
                      autoregressive_source=w.get("autoregressive_source") or "vega:default")
    return sample


def graph_sample(raw: dict) -> dict:
    """A graph's sample keys, `protocol_source` and `batching_source` as the composer sets them."""
    g = raw["graph"]
    default = "graphs" if g["level"] == "graph" else "full"
    return {"level": g["level"], "graph_id": g["graph_id"], "protocol": g["protocol"],
            "protocol_source": g.get("protocol_source") or "vega:default", "batching": g["batching"],
            "batching_source": "vega:default" if g["batching"] == default else f"user:stance/batching-{g['batching']}",
            "neighbours": g["neighbours"]}


def run_dir(tmp: Path, lane: Lane, architect: bool = True, memory: str | None = None,
            spec: dict | None = None) -> Path:
    work = tmp / "work"
    run = work / "vega-runs" / "2026-09-05T11-03Z"
    run.mkdir(parents=True)
    (run / "netspec.json").write_text(json.dumps(spec or spec_for(lane)), encoding="utf-8")
    claude = work / ".claude"
    claude.mkdir()
    if memory is not None:
        (claude / "domain-forge-memory.md").write_text(memory, encoding="utf-8")
    if architect:
        arch = claude / "architect-runs" / "2026-09-05T11-20Z"
        arch.mkdir(parents=True)
        (arch / "findings.json").write_text(json.dumps({"findings": [
            {"id": "A-1", "depth": "surface", "file": "training/loop.py", "title": "fit() is 90 lines"},
            {"id": "A-2", "depth": "architectural", "file": "domain.py", "title": "EpochRecord.of() takes **kw"}]}),
            encoding="utf-8")
    return run


def experiments() -> dict:
    def run(rid, kind, status, family=None, fired=(), eligible=True, stopped="early_stopping"):
        return {"id": rid, "kind": kind, "family": family, "status": status, "stopped_by": stopped,
                "hypothesis": "a train/valid gap under 0.05 at epoch 47 would mean 1e-4 is inert",
                "delta_text": "regularization[kind=weight_decay]/value: 0.0001 → 0.0", "epochs": 57,
                "best_epoch": 47, "primary": 0.7308, "eligible": eligible,
                "ineligible_because": [] if eligible else ["a pruned trial (epoch 3 of 57)"],
                "diagnostics": [c for c, _ in fired],
                "diagnostics_detail": [{"code": c, "name": n, "fired": True, "evidence": "rose 0.09694 over 11 epochs"}
                                       for c, n in fired] + [{"code": "D4", "name": "underfitting", "fired": False}]}
    return {"schema": "vega/experiments@1",
            "runs": [run("R0", "baseline", "done", fired=[("D3", "overfitting after best epoch")]),
                     run("R1", "ablation", "done", family="architecture"),
                     run("R6", "ablation", "done", family="regularization", stopped="budget"),
                     run("H1-3", "hpo", "done", fired=[("D12", "calibration drift over 10 epochs")]),
                     run("H1-4", "hpo", "pruned", eligible=False, stopped="pruned", fired=[("D5", "thrashing")]),
                     run("R9", "ablation", "failed", family="objective", stopped=None)],
            "selection": {"run": "H1-3", "rule": "best valid primary in the direction of metrics.selection.mode",
                          "value": 0.785277138239992, "margin_over_baseline": 0.054434, "override": None},
            "hpo": [{"study": "H1", "trials": 5, "best": "H1-3", "importance": {"optimizer/params/lr": 0.3729}}]}


def bundle_dir(tmp: Path, stances: list[str], noun: str | None = "order") -> Path:
    bundle = tmp / "work" / "bundle" / "orders-late-H1-3"
    bundle.mkdir(parents=True)
    identity = {"cmd": "orders", "title": "orders late pipeline v2", "description": "Answers each row (9 columns)."}
    if noun:
        identity.update(noun=noun, nouns=noun + "s")
    (bundle / "MANIFEST.json").write_text(json.dumps({"schema": "vega/bundle@1", "slug": "orders-late", "run": "H1-3",
                                                      "stances": stances, "serve": {"identity": identity},
                                                      "roundtrip": {"status": "pass", "rows": 119}}), encoding="utf-8")
    return bundle


# =========================================================================== forge
def recorded_judgement() -> dict:
    """The architect judgement the apply script splices from the fixture's recorded run
    (vega-forge's audit_quorum: one first audit with no structural or divergence finding)."""
    quorum = load_script("vega-forge/scripts/audit_quorum.py", "audit_quorum_t03")
    judgement = quorum.judge_paths([REPO / "vega-forge/scripts/tests/fixtures/dot-claude/architect-runs/"
                                    "2026-09-05T11-20Z"], "architect")
    assert judgement.passes and judgement.record["recorded"] == 3
    return judgement.record


def test_forge_authoring_carries_no_digit_and_passes_the_forge_validator(tmp_path):
    lane = LANES["L01"]
    memory = "# m\n\n## Network stances\n\n- 2026-09-05 | orders | family: mlp-residual (user)\n"
    run = run_dir(tmp_path, lane, memory=memory)
    doc = authoring.forge_authoring(run, lane)
    assert authoring.digit_problems(doc, "forge") == []
    assert doc["from_run"] == "." and doc["schema"] == "vega/architecture@1"
    assert [f["id"] for f in doc["code_findings"]] == ["A-1", "A-2"]
    assert "architect" not in doc["conformance"], "the architect run is spliced by the apply script"
    apply = load_script("vega-forge/scripts/apply_architecture_layer.py", "apply_architecture_layer_t03")
    spliced = dict(copy.deepcopy(doc), netspec=spec_for(lane), smoke={"forward_ok": True, "params_total": 10705},
                   code={}, blueprint_map={"entries": [], "unmapped": []},
                   notebook={"path": "engineer.ipynb", "executed": True})
    spliced["conformance"]["architect"] = recorded_judgement()
    said: list[str] = []
    assert apply.validate(spliced, said.append) == []
    assert not [s for s in said if s.startswith("WARN")], f"every testable default has its row: {said}"
    for fork in doc["forks"]:
        assert bool(fork.get("answer")) != bool(fork.get("or")), fork
        assert fork.get("or") in (None, authoring.UNATTENDED)
    answered = {f["family"] for f in doc["forks"] if f.get("answer")}
    assert "architecture" in answered, "the family's user: source and the memory stance settle the architecture"
    assert {"optimization"} <= {f["family"] for f in doc["forks"] if f.get("or")}


def test_forge_authoring_says_the_architect_is_absent(tmp_path):
    lane = LANES["L10"]
    doc = authoring.forge_authoring(run_dir(tmp_path, lane, architect=False), lane)
    assert doc["conformance"]["architect"] == "not installed"
    assert doc["conformance"]["improve"] == "not installed"
    assert any(m.startswith("WARN: improve not installed") for m in doc["markers"])
    assert doc["code_findings"] == []
    assert any(m.startswith("WARN:") and "architect" in m for m in doc["markers"])
    assert authoring.digit_problems(doc, "forge") == []
    points = point_forks(doc)
    assert set(points) == {"context", "autoregressive"}, \
        "a window forecast asks the context and the channel; the shaping record names its series key"
    assert all(f["or"] == authoring.UNATTENDED for f in points.values())
    rows = {r["path"]: r["observation"] for r in doc["defaults_to_test"]}
    assert rows["data_contract/sample/context"] == authoring.POINT_OBSERVATIONS["context"]
    assert rows["data_contract/sample/autoregressive"] == authoring.POINT_OBSERVATIONS["autoregressive"]
    assert "data_contract/sample/series_by" not in rows and authoring.SERIES_WARN not in doc["markers"]
    assert_forge_validates(doc, spec_for(lane))


# =========================================================================== forge: the sample's fork points
FORK_ASKED = {text: point for point, (_family, text) in authoring.FORK_POINTS.items()}


def point_forks(doc: dict) -> dict[str, dict]:
    """The fork-point forks of a forge authoring, by point."""
    return {FORK_ASKED[f["asked"]]: f for f in doc["forks"] if f["asked"] in FORK_ASKED}


def assert_forge_validates(doc: dict, spec: dict) -> None:
    """The real apply_architecture_layer.validate accepts the document, with no WARN (the
    recorded architect run spliced in as the apply script would, when the work dir holds one)."""
    apply = load_script("vega-forge/scripts/apply_architecture_layer.py", "apply_architecture_layer_t03c")
    spliced = dict(copy.deepcopy(doc), netspec=spec, smoke={"forward_ok": True, "params_total": 1},
                   code={}, blueprint_map={"entries": [], "unmapped": []},
                   notebook={"path": "engineer.ipynb", "executed": True})
    spliced["conformance"].setdefault("architect", recorded_judgement())
    said: list[str] = []
    assert apply.validate(spliced, said.append) == []
    assert not [s for s in said if s.startswith("WARN")], said


def assert_ladder_reads(doc: dict, spec: dict) -> None:
    """Every fork-point row's path is one plan_ladder reads, and it addresses the sample's value."""
    core = load_script("vega-train/scripts/vega_train_core.py", "vega_train_core_t03")
    sample = spec["data_contract"]["sample"]
    for row in doc["defaults_to_test"]:
        if row["path"].startswith("data_contract/sample/"):
            point = row["path"].rsplit("/", 1)[1]
            assert core.read_path(spec, core.normalise_path(row["path"])) == sample[point], row["path"]


def forge_on(tmp: Path, lane: Lane, **sample: Any) -> tuple[dict, dict]:
    """forge_authoring on the lane's spec with the sample's keys edited (a value of ... deletes the key)."""
    spec = spec_for(lane)
    for key, value in sample.items():
        spec["data_contract"]["sample"].pop(key, None)
        if value is not ...:
            spec["data_contract"]["sample"][key] = value
    return authoring.forge_authoring(run_dir(tmp, lane, spec=spec), lane), spec


# (series_by, series_source, how the fork stands, a defaults_to_test row) — netspec-contract §3:
# asked only when the shaping record names no series key; unattended null with the WARN
SERIES_CASES = (("station", "shape:S8", None, False),
                ("station", "user:stance/series", "answer", False),
                (None, "user:stance/one-series", "answer", False),
                (None, None, "or", True),
                (None, "vega:default", "or", True),
                (..., ..., "or", True),
                ("station", "vega:default", None, True),
                ("station", ..., None, False))


@pytest.mark.parametrize("series_by,series_source,state,row", SERIES_CASES)
def test_the_series_fork_follows_the_contract(tmp_path, series_by, series_source, state, row):
    lane = LANES["L10"]
    doc, spec = forge_on(tmp_path, lane, series_by=series_by, series_source=series_source)
    fork = point_forks(doc).get("series_by")
    assert (fork is None) == (state is None), fork
    if state == "answer":
        assert fork == {"family": "architecture", "asked": authoring.FORK_POINTS["series_by"][1],
                        "answer": authoring.POINT_ANSWERS["user"]}
    if state == "or":
        assert fork["or"] == authoring.SERIES_UNATTENDED and "null" in fork["or"]
        assert authoring.UNATTENDED in fork["or"], "smoke_architecture looks for the phrase on an unattended fork"
    assert (authoring.SERIES_WARN in doc["markers"]) == (state == "or")
    rows = {r["path"]: r["observation"] for r in doc["defaults_to_test"]}
    assert ("data_contract/sample/series_by" in rows) == row
    if row:
        key = "series_by_key" if series_by not in (None, ...) else "series_by"
        assert rows["data_contract/sample/series_by"] == authoring.POINT_OBSERVATIONS[key]
    assert authoring.digit_problems(doc, "forge") == []
    assert_forge_validates(doc, spec)
    if series_by is not ...:
        assert_ladder_reads(doc, spec)


def test_the_series_source_is_read_where_the_contract_writes_it(tmp_path):
    """The review's probe: a series key a user stance set is the user's answer, never the default."""
    doc, _spec = forge_on(tmp_path, LANES["L10"], series_by="station", series_source="user:stance/series",
                          series_by_source=...)
    assert point_forks(doc)["series_by"].get("answer") == authoring.POINT_ANSWERS["user"]


def test_window_points_settled_or_defaulted(tmp_path):
    """A context a user turn chose is answered and not tested; the channel is a fork point on a
    forecast only; a sequence task with one series (the user said so) asks only the context."""
    doc, spec = forge_on(tmp_path / "a", LANES["L10"], context="none", context_source="user:stance/window-context",
                         context_catalog=...)
    forks = point_forks(doc)
    assert forks["context"]["answer"] == authoring.POINT_ANSWERS["user"]
    assert forks["autoregressive"]["or"] == authoring.UNATTENDED
    paths = [r["path"] for r in doc["defaults_to_test"]]
    assert "data_contract/sample/context" not in paths and "data_contract/sample/autoregressive" in paths
    assert_forge_validates(doc, spec)
    doc, spec = forge_on(tmp_path / "b", LANES["L12"], series_source="user:stance/one-series",
                         autoregressive=False, autoregressive_source="vega:default")
    assert set(point_forks(doc)) == {"context", "series_by"}
    assert point_forks(doc)["series_by"]["answer"] == authoring.POINT_ANSWERS["user"]
    paths = [r["path"] for r in doc["defaults_to_test"]]
    assert "data_contract/sample/autoregressive" not in paths, "a non-forecast window has no channel to test"
    assert authoring.SERIES_WARN not in doc["markers"]
    assert_ladder_reads(doc, spec)
    doc, _spec = forge_on(tmp_path / "c", LANES["L10"], context_source="shape:S9")
    assert point_forks(doc)["context"]["answer"] == authoring.POINT_ANSWERS["shape"]


# (lane, forks by point: answer | or, the batching row's observation key or None)
GRAPH_CASES = (("L19", {"protocol": "or", "batching": "or"}, "batching"),
               ("L21", {"protocol": "or", "batching": "answer"}, None),
               ("L22", {"protocol": "or", "batching": "or"}, "batching_graphs"),
               ("L23", {"protocol": "answer", "batching": "or"}, "batching"))


@pytest.mark.parametrize("lane_id,forks,batching", GRAPH_CASES)
def test_graph_points_and_their_rows(tmp_path, lane_id, forks, batching):
    """protocol is asked but never tested (always inductive unattended); batching's default
    lands in defaults_to_test at the path plan_ladder reads."""
    lane = LANES[lane_id]
    spec = spec_for(lane)
    doc = authoring.forge_authoring(run_dir(tmp_path, lane, spec=spec), lane)
    got = point_forks(doc)
    assert {p: ("answer" if f.get("answer") else "or") for p, f in got.items()} == forks
    assert all(f.get("or") in (None, authoring.UNATTENDED) for p, f in got.items() if p != "protocol")
    if got["protocol"].get("or"):
        assert got["protocol"]["or"] == authoring.PROTOCOL_UNATTENDED and "inductive" in got["protocol"]["or"]
        assert authoring.UNATTENDED in got["protocol"]["or"], "smoke_architecture looks for the phrase"
    rows = {r["path"]: r["observation"] for r in doc["defaults_to_test"]}
    assert "data_contract/sample/protocol" not in rows
    assert rows.get("data_contract/sample/batching") == (authoring.POINT_OBSERVATIONS[batching] if batching else None)
    assert authoring.digit_problems(doc, "forge") == []
    assert_forge_validates(doc, spec)
    assert_ladder_reads(doc, spec)


def test_the_fork_points_are_the_catalogs():
    """FORK_POINTS, their families, the untested point and the series WARN are the forge
    catalog's (fork_order points, defaults_to_test flags, defaults.window_series)."""
    points = {p["id"]: (entry["family"], p) for entry in FORGE_CATALOG["fork_order"] for p in entry.get("points") or []}
    assert list(points) == list(authoring.FORK_POINTS)
    assert {pid: fam for pid, (fam, _p) in points.items()} == {k: v[0] for k, v in authoring.FORK_POINTS.items()}
    assert {pid for pid, (_f, p) in points.items() if not p["defaults_to_test"]} == set(authoring.UNTESTED_POINTS)
    assert authoring.SERIES_WARN == "WARN: " + FORGE_CATALOG["defaults"]["window_series"]["unattended_warn"]
    assert not [o for o in authoring.POINT_OBSERVATIONS.values() if re.search(r"\d", o)]
    for pid, (_f, p) in points.items():     # the unattended answer each point's catalog row names
        said = authoring.POINT_UNATTENDED.get(pid, authoring.UNATTENDED)
        head, word = {"series_by": ("null", "null"), "protocol": ("always inductive", "inductive")}.get(
            pid, ("the default, applied and marked", "default applied"))
        assert p["unattended"].startswith(head) and word in said, (pid, said)
        assert not re.search(r"\d", said)


def test_forge_authoring_on_every_lane_composition(tmp_path):
    """Every lane's declared blocks (graph, recurrent, custom, generative) read as words."""
    for i, lane in enumerate(l for l in LANES.values() if l.driver == "lanekit"):
        doc = authoring.forge_authoring(run_dir(tmp_path / str(i), lane), lane)
        assert authoring.digit_problems(doc, "forge") == [], lane.id
        text = doc["readings"]["per_family"]["architecture"]
        for btype in lane.flat_block_types():
            assert authoring.block_words(btype) in text, (lane.id, btype)


# =========================================================================== train
def test_train_authoring_carries_no_digit_and_passes_the_training_validator(tmp_path):
    camp = tmp_path / "camp"
    camp.mkdir()
    exp = experiments()
    (camp / "experiments.json").write_text(json.dumps(exp), encoding="utf-8")
    (camp / "netspec.json").write_text(json.dumps(spec_for(LANES["L01"])), encoding="utf-8")
    doc = authoring.train_authoring(camp, LANES["L01"])
    assert authoring.digit_problems(doc, "train") == []
    assert doc["from_campaign"] == "."
    assert list(doc["readings"]["per_run"]) == [r["id"] for r in exp["runs"]]
    assert "picked it" in doc["readings"]["per_run"]["H1-3"]
    assert "pruned" in doc["readings"]["per_run"]["H1-4"] and "excludes it" in doc["readings"]["per_run"]["H1-4"]
    assert "did not finish" in doc["readings"]["per_run"]["R9"]
    assert "budget" in doc["readings"]["per_run"]["R6"]
    assert len(doc["readings"]["abstract"]) > 200, "the training smoke wants an abstract over 200 characters"
    assert any(m.startswith("WARN:") for m in doc["markers"]), "a failed run is said out loud"
    apply = load_script("vega-train/scripts/apply_training_layer.py", "apply_training_layer_t03")
    spliced = dict(copy.deepcopy(doc), campaign={}, ladder={}, experiments=exp,
                   histories={r["id"]: {"path": f"runs/{r['id']}/history.jsonl"} for r in exp["runs"]},
                   mechanics={"one_step": "recorded"}, notebook={"path": "train.ipynb", "executed": True})
    said: list[str] = []
    assert apply.validate(spliced, said.append) == []
    assert not [s for s in said if "no per-run reading" in s]
    exp["selection"]["override"] = {"run": "R0", "reason": "user: interpretability"}
    (camp / "experiments.json").write_text(json.dumps(exp), encoding="utf-8")
    doc = authoring.train_authoring(camp, LANES["L01"])
    assert doc["forks"][1]["answer"].startswith("an override")
    assert "override" in doc["readings"]["selection"]


# =========================================================================== pack
def test_pack_authoring_carries_no_digit_and_passes_the_card_rule(tmp_path):
    card = load_script("vega-pack/scripts/model_card.py", "model_card_t03")
    for stances, noun in (([], "order"), (["user: ship although not earned vs linear — 2 runs"], None),
                          ([], "row2")):
        bundle = bundle_dir(tmp_path / f"{len(stances)}{noun}", stances, noun)
        doc = authoring.pack_authoring(bundle, LANES["L01"])
        assert authoring.digit_problems(doc, "pack") == []
        readings = doc["readings"]
        assert all(len(readings[k]) >= 40 for k in ("abstract", "intended_use"))
        assert not re.search(r"\d", readings["intended_use"]), "apply_bundle_layer refuses a digit there"
        assert card.check_intended_use(readings["intended_use"]) == []
        for tool in ("architect", "improve"):
            assert any(re.search(rf"\b{tool}\b", m, re.I) for m in doc["markers"]), \
                f"no {tool} run: a marker must say so (apply_bundle_layer)"
        assert doc["from_bundle"] == "bundle/orders-late-H1-3"
        assert ("recorded stance" in readings["intended_use"]) == bool(stances)
    assert "each order given" in authoring.pack_authoring(bundle_dir(tmp_path / "n", [], "order"),
                                                          LANES["L01"])["readings"]["intended_use"]


# =========================================================================== the digit discipline itself
def test_digit_problems_find_digits_in_prose_only():
    doc = {"schema": "vega/architecture@1", "from_run": "run-2026", "code_findings": [{"id": "A-1", "next": "fix it"}],
           "defaults_to_test": [{"path": "blocks[B1]", "observation": "width 64 is too wide"}],
           "forks": [{"family": "architecture", "asked": "which", "or": authoring.UNATTENDED}],
           "readings": {"per_run": {"R0": "fine"}, "abstract": "٣ Arabic-Indic digits count too"}}
    problems = authoring.digit_problems(doc, "forge")
    assert len(problems) == 2
    assert any("defaults_to_test[0].observation" in p for p in problems)
    assert any("readings.abstract" in p for p in problems)
    plan = {"authored": {"headline": "{eval_network_pr_auc:pct} of {row.many}", "lead": "a b2c lead",
                         "context": {"facts": [{"label": "rows", "fact": "step_s4_rows_after"}]},
                         "input_groups": [{"id": "g2", "label": "ok", "text": "ok", "members": ["col_7"]}],
                         "class_roles": {"roles": {"1": "alert"}, "source": {"1": "show:2026-09-29"}}}}
    problems = authoring.digit_problems(plan, "show")
    assert problems == ["authored.lead: digit '2' outside a placeholder in 'a b2c lead'"]
    assert authoring.PLACEHOLDER == SHOW_CATALOG["placeholder"]["pattern"], "one placeholder grammar (R-E10)"
    with pytest.raises(ValueError):
        authoring.prose(doc, "story")


def test_words_never_carry_a_digit():
    for name in list(FORGE_CATALOG["families"]) + list(FORGE_CATALOG["blocks"]) + ["top7", "f1_macro", "x2y", "l2"]:
        assert not re.search(r"\d", authoring.words(name)), name
    assert authoring.words("conv1d") == "one-dimensional convolution"
    assert authoring.words("top7") == "top-seven"
    assert set(authoring.BLOCK_WORDS) == set(FORGE_CATALOG["blocks"]), "a phrase for every catalog block"
    assert not [w for w in authoring.BLOCK_WORDS.values() if re.search(r"\d", w)]


# =========================================================================== show: drafts of seven lanes
VIEW_KEYS = SHOW_CATALOG["view_keys"]


def view_key(lane: Lane) -> str:
    entry = VIEW_KEYS[lane.task]
    return entry[lane.target["base"]] if isinstance(entry, dict) else entry


def walk_ids(lane: Lane) -> list[str]:
    head, tail = SHOW_CATALOG["walk"]["head"], SHOW_CATALOG["walk"]["tail"]
    return head + list(dict.fromkeys(lane.flat_block_types())) + tail


def task_metrics(lane: Lane) -> tuple[str, str]:
    """The task row's primary and first secondary metric (a sequence task reads its base row)."""
    row = FORGE_CATALOG["tasks"][lane.task]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", str(row["primary"])):          # "as the base task"
        row = FORGE_CATALOG["tasks"][lane.target["base"]]
    return row["primary"], (row.get("secondary") or ["loss"])[0]


def fact(value: Any, ref: str, kind: str = "count", fmt: str = "count") -> dict:
    return {"value": value, "ref": ref, "kind": kind, "format": fmt}


SHAPED = [f"f{i}_value" for i in range(1, 11)]
TRAINING, EVALUATION = "records/training.json#experiments.runs", "records/evaluation.json#metrics"


def facts_for(lane: Lane, tmp: Path, monitor: bool) -> dict:
    """A show-facts@1 document in the shape the extractor (H03) writes for this lane, with
    digits wherever the records carry them; its bundle's MANIFEST carries a monitor or not."""
    raw = lane.raw
    primary, secondary = task_metrics(lane)
    bundle = tmp / "bundle" / f"{raw['generator']}-R7"
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {"monitor": {"kind": "psi"}} if monitor else {}
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    shape_rows = "records/shape.json#manifest.outputs.files"
    facts = {"source_rows": fact(2400, "records/geometry.json#source.rows"),
             "train_rows": fact(1680, f"{shape_rows}.train.rows"),
             "valid_rows": fact(240, f"{shape_rows}.valid.rows"),
             "test_rows": fact(480, f"{shape_rows}.test.rows"),
             "shaped_inputs": fact(10, "netspec.json#data_contract.input_dim"),
             "block_b1_units": fact(64, "netspec.json#blocks[id=B1].params.units", "param", "param"),
             "optimizer_lr": fact(0.001, "netspec.json#optimizer.params.lr", "param", "param"),
             "run_r0_primary": fact(0.73, f"{TRAINING}[id=R0].primary", "metric", "metric"),
             "run_h1_3_primary": fact(0.78, f"{TRAINING}[id=H1-3].primary", "metric", "metric"),
             f"eval_network_{primary}": fact({"value": 0.86, "ci": [0.7, 0.97], "n": 119},
                                             f"{EVALUATION}.models.network.{primary}", "ci_metric", "ci"),
             f"eval_network_{secondary}": fact({"value": 0.9, "ci": [0.8, 0.95], "n": 119},
                                               f"{EVALUATION}.models.network.{secondary}", "ci_metric", "ci"),
             "eval_ci_level": fact(0.95, f"{EVALUATION}.ci_level", "param", "param")}
    goal = raw["goal"] if raw["goal"] in ("predict", "reconstruct", "anomaly") else "predict"
    breakdown = {"kind": "per_class", "names": "classes_n", "values": f"eval_network_{secondary}"}
    return {
        "schema": "vega/show-facts@1", "extracted_at": "2026-09-29T12:00:00Z",
        "bundle": {"dir": str(bundle), "manifest_digest": sha("m"), "slug": raw["generator"], "run": "R7",
                   "model_version": sha("v")},
        "model": {"path": f"{raw['generator']}.domain.analysis.shaped.vega.train.eval.pack.html",
                  "turtle_digest": sha("t"),
                  "layers": ["geometry", "shape", "architecture", "training", "evaluation", "bundle"]},
        "task": {"kind": lane.task, "base": lane.target.get("base"), "goal": goal, "sample_shape": raw["shape"],
                 "graph_level": lane.graph_level, "modality": lane.modality, "family": raw["family"],
                 "view_key": view_key(lane), "label": lane.target.get("column"),
                 "classes": list(lane.classes) if lane.classes else None, "outcomes": None,
                 "primary": primary, "direction": "max"},
        "facts": facts, "names": ["R0", "H1-3", "B1", "F1"] + SHAPED,
        "groups": {"sizes": {k: k for k in ("source_rows", "train_rows", "valid_rows", "test_rows", "shaped_inputs")},
                   "inputs": {"shaped": SHAPED}, "network": {"family": raw["family"]}, "decisions": [],
                   "runs": [], "selection": {"run": "H1-3"}, "models": [], "metrics": {"primary": primary},
                   "breakdown": breakdown if lane.classes else None,
                   "verdict": {"earned": False}, "limits": [], "identity": {"noun": "row"}}}


def choices_for(lane: Lane) -> list[dict]:
    """The draft's choices: every block (the first mandatory, with a fact-keyed parameter), the
    objective, the optimizer (a parameter whose name carries a digit), dropout, the schedule."""
    choices = [{"decision": block["id"], "family": "architecture", "type": block["type"],
                "params": {"units": "block_b1_units"} if i == 0 else {}, "rationale_ref": None,
                "tested_by": ["R1"] if i == 0 else [], "mandatory": i == 0}
               for i, block in enumerate(lane.raw["blocks"])]

    def choice(decision, family, mandatory, params=None, tested=()):
        return {"decision": decision, "family": family, "type": None, "params": params or {},
                "mandatory": mandatory, "tested_by": list(tested)}
    return choices + [choice("loss", "objective", True),
                      choice("optimizer", "optimization", True, {"lr": "optimizer_lr", "beta2": "optimizer_lr"}),
                      choice("regularization[kind=dropout]", "regularization", True, tested=["R6"]),
                      choice("schedule", "optimization", False)]


RUNS = (("R0", "the spec as engineered", False, True, "run_r0_primary"),
        ("R1", "family mlp-residual → linear", False, False, "run_r0_primary"),
        ("R6", "weight_decay 0.0001 → 0.0", False, False, "run_r0_primary"),
        ("H1-3", "lr 0.001 → 0.0003", True, False, "run_h1_3_primary"),
        ("H1-4", "trial 4", False, False, "run_h1_3_primary"))


def draft_for(lane: Lane, facts: dict, report: str) -> dict:
    """A story-plan@2 draft in the shape the draft (H04) writes for this lane: its proposals,
    every authored slot null. `report` is the role map's status: emitted, proposed or absent."""
    raw, modality = lane.raw, lane.modality
    primary, secondary = task_metrics(lane)
    motifs = SHOW_CATALOG["motifs"]["by_modality"][modality]
    roles = raw.get("roles")
    neutral = next((c for c, r in (roles or {}).items() if r == "neutral"), None)
    rule = ("task:anomaly" if lane.task == "anomaly"
            else "structural:negative" if lane.target.get("positive") else "structural:majority")
    stance = next((s for s in raw.get("stances") or [] if s.startswith("show: roles")), None)
    class_roles = None if not roles else {
        "neutral": {"class": neutral, "rule": rule, "share": "train_share_x"} if neutral else None,
        "alert_candidates": [c for c, r in roles.items() if r == "alert"],
        "basis_found": "show:2026-09-29" if stance else None,
        "memory": f"2026-09-29 | {raw['generator']} | {stance}" if stance else None,
        "evidence": [], "fork": None if stance else "S2"}
    bound = report != "absent"
    bindings = {r: {"label": f"vega:{r}", "source": "document-project" if report == "emitted" else "proposed",
                    "evidence": None if report == "emitted" else "anchor equation.3.14"}
                for r in ("net.forward", "pack.slice", "chapter.net")} if bound else {}
    groups = [("g1_rates", SHAPED[:3], [{"kind": "correlation", "pairs": 2}]),
              ("codes", SHAPED[3:5], [{"kind": "encoded"}]),
              ("made", SHAPED[5:8], [{"kind": "created", "step": "S4"}]), ("rest", SHAPED[8:], [])]
    limits = [("split", True), ("exposures", True), ("one_seed", True), ("verdict_not_earned", True),
              ("ship_stance", True), ("roundtrip_boundary", raw["shape"] == "window"), ("calibration", False),
              ("warn_eval_1", False)]
    proposals = {
        "modality": modality, "view_key": view_key(lane),
        "lexicon": {k: dict(v, source="vega:default") for k, v in raw["lexicon"].items()},
        "motif": {"modality": modality, "base": motifs["base"], "admissible": motifs["admissible"],
                  "stance": None, "fork": "S3" if len(motifs["admissible"]) > 1 else None},
        "class_roles": class_roles,
        "input_groups": [{"suggested_id": g, "members": m, "evidence": e} for g, m, e in groups],
        "choices": choices_for(lane),
        "runs": [{"id": i, "label": label, "selected": sel, "reference": ref, "primary": key}
                 for i, label, sel, ref, key in RUNS],
        "results_facts": [f"eval_network_{primary}", f"eval_network_{secondary}"],
        "limits": [{"id": i, "mandatory": m, "facts": ["train_rows"] if i == "split" else [], "prose_ref": None}
                   for i, m in limits],
        "code_steps": [{"step": s, "role": "net.forward" if bound else None, "status": "bound" if bound else "brief"}
                       for s in walk_ids(lane)],
        "report_roles": {"status": report, "bindings": bindings, "fork": None if bound else "S1"},
        "context_steps": SHOW_CATALOG["context_steps"]["by_shape"][raw["shape"]],
        "prose_sources": [{"what": "the verdict reading", "ref": "records/evaluation.json#readings.verdict"}]}
    slots = ("locale", "lexicon", "motif", "class_roles", "report_roles", "headline", "lead", "context", "network",
             "experiments", "results", "input_groups", "code_notes", "monitor_note")
    return {"schema": "vega/story-plan@2", "status": "draft",
            "facts": {"file": "plan/facts.json", "digest": sha(json.dumps(facts))},
            "template": {"version": "2.3.0", "manifest_digest": sha("TEMPLATE.json")}, "carried_from": None,
            "proposals": proposals, "authored": {k: None for k in slots}}


def facts_and_draft(lane: Lane, tmp: Path, report: str, monitor: bool = False) -> tuple[dict, dict]:
    facts = facts_for(lane, tmp, monitor)
    return facts, draft_for(lane, facts, report)


STORY_LANES = (("L01", "emitted", False), ("L05", "proposed", True), ("L06", "absent", False),
               ("L10", "emitted", False), ("L12", "absent", False), ("L19", "absent", True),
               ("L22", "emitted", False), ("L04", "absent", False), ("L08", "absent", False))


def plan_rule_problems(plan: dict, facts: dict, lane: Lane) -> list[str]:
    """The plan rules the contract states (show-contract §3.8) that the authoring decides:
    V3 placeholders, V4 record paths, V5 closed sets, V6 class roles, V7 input groups, V8
    choices, V9 limits, V10 run labels, V11 code notes, V12 lengths, V13 status and slots."""
    a, p, out = plan["authored"], plan["proposals"], []
    lengths = SHOW_CATALOG["authored_lengths"]
    lexicon_paths = set(SHOW_CATALOG["lexicon"]["paths"])
    placeholder = re.compile(SHOW_CATALOG["placeholder"]["pattern"])
    for where, text in authoring.prose(plan, "show"):
        for m in placeholder.finditer(text):
            key, path = m.group(1), m.group(2)
            if path is None and key not in facts["facts"]:
                out.append(f"V3 {where}: {{{key}}} is not a fact key")
            if path is not None and (key not in a["lexicon"] or path not in lexicon_paths):
                out.append(f"V3 {where}: {{{key}.{path}}} is not a lexicon path of this lane")
        for needle in ("records/", "MANIFEST.json", "netspec.json", "sha256:", "report:", "#"):
            if needle in text:
                out.append(f"V4 {where}: record path {needle!r}")
    if a["locale"] not in SHOW_CATALOG["locales"]:
        out.append("V5 locale")
    if a["motif"] not in p["motif"]["admissible"]:
        out.append("V5 motif")
    shape = lane.raw["shape"]
    required = SHOW_CATALOG["lexicon"]["required"][f"graph:{lane.graph_level}" if shape == "graph" else shape]
    if set(a["lexicon"]) - {"target"} != set(required):
        out.append(f"V5 lexicon keys {sorted(a['lexicon'])}")
    roles = a["class_roles"]
    if (roles is None) != (lane.raw.get("roles") is None):
        out.append("V6 class roles present iff the task has classes")
    if roles:
        basis = re.compile(SHOW_CATALOG["roles"]["basis_pattern"])
        table = SHOW_CATALOG["roles"]["basis"]
        for cls, role in roles["roles"].items():
            src = roles["source"][cls]
            if not basis.match(src):
                out.append(f"V6 basis {src!r}")
                continue
            key = src if src in table else "user:<turn>" if src.startswith("user:") else "show:<stance date>"
            if role not in table[key]["grants"]:
                out.append(f"V6 {cls}: {src} does not grant {role}")
            if role == "alert" and not src.startswith(tuple(SHOW_CATALOG["roles"]["alert_requires"])):
                out.append(f"V6 {cls}: alert on {src}")
        if set(roles["roles"]) != set(roles["source"]):
            out.append("V6 every class has a role and a basis")
        neutrals = sum(1 for r in roles["roles"].values() if r == "neutral")
        if neutrals > 1 and p["view_key"] in ("classes", "binary", "anomaly"):
            out.append("V6 more than one neutral")
    members = [m for g in a["input_groups"] for m in g["members"]]
    ids = [g["id"] for g in a["input_groups"]]
    if sorted(members) != sorted(facts["groups"]["inputs"]["shaped"]) or len(set(ids)) != len(ids):
        out.append("V7 input groups do not partition the shaped inputs")
    decisions = [c["decision"] for c in a["network"]["choices"]]
    mandatory = [c["decision"] for c in p["choices"] if c["mandatory"]]
    if not set(mandatory) <= set(decisions):
        out.append(f"V8 mandatory choices missing: {set(mandatory) - set(decisions)}")
    if not (min(SHOW_CATALOG["choices"]["min"], len(p["choices"])) <= len(decisions) <= SHOW_CATALOG["choices"]["max"]):
        out.append(f"V8 {len(decisions)} choices")
    if not {l["id"] for l in p["limits"] if l["mandatory"]} <= {l["id"] for l in a["results"]["limits"]}:
        out.append("V9 a mandatory limit is not covered")
    if set(a["experiments"]["labels"]) != {r["id"] for r in p["runs"]}:
        out.append("V10 run labels")
    if not set(a["code_notes"]) <= {s["step"] for s in p["code_steps"]}:
        out.append("V11 code notes")
    if not set(a["results"]["notes"]) <= set(p["results_facts"]):
        out.append("results notes outside results_facts")
    if not 2 <= len(a["context"]["facts"]) <= 6 or any(f["fact"] not in facts["facts"] for f in a["context"]["facts"]):
        out.append("context facts: two to six fact keys")
    if list(a["context"]["steps"]) != p["context_steps"]:
        out.append("context steps are exactly the proposed ones")
    checks = [("headline", a["headline"]), ("lead", a["lead"]), ("caption", a["experiments"]["caption"])]
    checks += [("context_step", t) for t in a["context"]["steps"].values()]
    checks += [("fact_label", f["label"]) for f in a["context"]["facts"]]
    checks += [(k, c[f]) for c in a["network"]["choices"] for k, f in (("choice_title", "title"), ("choice_text", "text"))]
    checks += [("run_label", t) for t in a["experiments"]["labels"].values()]
    checks += [("note", t) for t in a["results"]["notes"].values()]
    checks += [(k, l[f]) for l in a["results"]["limits"] for k, f in (("limit_title", "title"), ("limit_text", "text"))]
    checks += [(k, g[f]) for g in a["input_groups"] for k, f in (("group_label", "label"), ("group_text", "text"))]
    checks += [("code_note", t) for t in a["code_notes"].values()]
    checks += [("monitor_note", a["monitor_note"])] if a["monitor_note"] else []
    out += [f"V12 {k} is {len(t)} > {lengths[k]}: {t[:50]!r}" for k, t in checks if len(t) > lengths[k]]
    if plan["status"] != "authored" or any(a[k] is None for k in ("locale", "lexicon", "motif", "headline", "lead",
                                                                    "context", "network", "experiments", "results",
                                                                    "input_groups", "code_notes")):
        out.append("V13 status and slots")
    return out


@pytest.mark.parametrize("lane_id,report,monitor", STORY_LANES)
def test_show_authoring_fills_the_plan_without_a_digit(tmp_path, lane_id, report, monitor):
    import jsonschema
    lane = LANES[lane_id]
    facts, draft = facts_and_draft(lane, tmp_path, report, monitor)
    schemas = SHOW / "references" / "schemas"
    jsonschema.Draft202012Validator(json.loads((schemas / "show-facts-1.schema.json").read_text())).validate(facts)
    plan_schema = jsonschema.Draft202012Validator(json.loads((schemas / "story-plan-2.schema.json").read_text()))
    plan_schema.validate(draft)
    frozen = copy.deepcopy(draft)
    plan = authoring.show_authoring(facts, lane, draft)
    assert draft == frozen, "the draft is not modified"
    plan_schema.validate(plan)
    assert plan["proposals"] == draft["proposals"] and plan["facts"] == draft["facts"]
    assert authoring.digit_problems(plan, "show") == []
    assert authoring.number_words(plan, "show", SHOW_CATALOG["number_words"]) == []
    assert plan_rule_problems(plan, facts, lane) == []
    a = plan["authored"]
    assert a["lexicon"] == {k: {p: v[p] for p in ("one", "many", "article")} for k, v in lane.raw["lexicon"].items()}
    assert a["motif"] == (lane.expect.get("motif") or draft["proposals"]["motif"]["base"])
    assert (a["class_roles"] or {}).get("roles") == lane.raw.get("roles")
    assert (a["report_roles"] is not None) == (report == "proposed")
    assert (a["monitor_note"] is not None) == monitor
    assert (a["results"]["breakdown_caption"] is not None) == bool(lane.classes)
    labels = a["experiments"]["labels"]
    assert labels["R0"] == "the spec as engineered" and labels["R1"] == "family mlp-residual → linear"
    assert labels["H1-3"] == "the run that shipped" and labels["H1-4"] == "a trial of the search"


def test_show_authoring_class_role_bases():
    """R-E5: the structural neutral the draft names, task:anomaly on the anomaly side, the
    show: stance the draft found, else the lane registry as the user; plain is vega:default."""
    cases = {"L01": {"0": "structural:negative", "1": "show:2026-09-29"},
             "L06": {"ok": "task:anomaly", "fault": "task:anomaly"},
             "L12": {"normal": "structural:majority", "imbalance": "show:2026-09-29",
                     "misalignment": "show:2026-09-29", "bearing": "show:2026-09-29"},
             "L02": {c: "vega:default" for c in LANES["L02"].raw["roles"]}}
    for lane_id, want in cases.items():
        lane = LANES[lane_id]
        props = {"class_roles": {"neutral": ({"class": next(c for c, r in lane.raw["roles"].items() if r == "neutral"),
                                              "rule": {"L01": "structural:negative", "L06": "task:anomaly",
                                                       "L12": "structural:majority"}[lane_id]}
                                             if lane_id != "L02" else None),
                                 "basis_found": "show:2026-09-29" if lane_id in ("L01", "L12") else None}}
        assert authoring._class_roles(props, lane.raw)["source"] == want, lane_id
    alone = authoring._class_roles({"class_roles": {"neutral": None, "basis_found": None}}, LANES["L19"].raw)
    assert alone["source"] == {"ok": "vega:default", "overloaded": "registry:L19"}, \
        "the registry is said as the registry, never as a user's turn (gate C)"
    assert authoring._class_roles({"class_roles": None}, LANES["L05"].raw) is None


def test_the_fallback_alert_basis_is_said_out_loud(tmp_path):
    """R-E5: an alert role the draft found no show: stance for rests on the lane registry
    (registry:<lane>, never a user's turn); the plan has no marker slot, so show_authoring says it as a WARN — to the
    stage log when T01's author helper runs it, where the runner counts it."""
    lane = LANES["L19"]
    facts, draft = facts_and_draft(lane, tmp_path, "absent")
    draft["proposals"]["class_roles"].update(basis_found=None, memory=None)
    said: list[str] = []
    plan = authoring.show_authoring(facts, lane, draft, out=said.append)
    assert plan["authored"]["class_roles"]["source"]["overloaded"] == "registry:L19"
    assert len(said) == 1 and said[0].startswith("WARN: no show: roles stance found for lane L19"), said
    assert "overloaded" in said[0] and "lanekit.authoring stances" in said[0]
    found = copy.deepcopy(draft)
    found["proposals"]["class_roles"]["basis_found"] = "show:2026-09-29"
    said.clear()
    plan = authoring.show_authoring(facts, lane, found, out=said.append)
    assert said == [] and plan["authored"]["class_roles"]["source"]["overloaded"] == "show:2026-09-29"
    for lane_id in ("L06", "L02"):      # task:anomaly on both sides; plain roles only
        f, d = facts_and_draft(LANES[lane_id], tmp_path / lane_id, "absent")
        d["proposals"]["class_roles"].update(basis_found=None)
        authoring.show_authoring(f, LANES[lane_id], d, out=said.append)
    assert said == [], "only an alert role on the registry fallback is said"
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    (plan_dir / "draft.json").write_text(json.dumps(draft), encoding="utf-8")
    run = subprocess.run([PY, "-m", "lanekit.checks", "author", "--lanes-file", str(HERE / "lanes.json"),
                          "--lane", "L19", "--lane-dir", str(tmp_path / "lane"), "--skills", str(REPO),
                          "--kind", "show", "--target", str(plan_dir / "story-plan.json"),
                          "--facts", str(plan_dir / "facts.json"), "--draft", str(plan_dir / "draft.json")],
                         cwd=HERE, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert re.search(r"^WARN: no show: roles stance found for lane L19", run.stdout, re.M), run.stdout


# =========================================================================== stances (R-E5)
ROLES_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) \| (\S+) \| show: roles (.+) \(user\)$")
BASIS_PATTERN = SHOW_CATALOG["roles"]["basis_pattern"]
BASIS_TABLE = SHOW_CATALOG["roles"]["basis"]
L01_ROLES_LINE = ("2026-09-29 | orders | show: roles 0 neutral (structural:negative), "
                  "1 alert (registry:L01) (user)")


def basis_key(basis: str) -> str:
    """The vega-show catalog's basis row a basis falls under."""
    return basis if basis in BASIS_TABLE else "user:<turn>" if basis.startswith("user:") else "show:<stance date>"


def test_a_roles_stance_for_every_lane_declaring_alert():
    alerting = [l for l in LANES.values() if "alert" in (l.raw.get("roles") or {}).values()]
    assert {l.id for l in alerting} >= {"L01", "L06", "L08", "L09", "L12", "L14", "L15", "L19"}
    for lane in LANES.values():
        lines = authoring.show_stances(lane, date="2026-09-29")
        roles_lines = [ROLES_LINE.match(line) for line in lines if " | show: roles " in line]
        if lane in alerting:
            assert len(roles_lines) == 1 and roles_lines[0], (lane.id, lines)
            m = roles_lines[0]
            assert m.group(1) == "2026-09-29" and m.group(2) == lane.raw["generator"]
            items = [re.match(r"(.+) (neutral|alert|plain) \((.+)\)$", p).group(1, 2, 3) for p in m.group(3).split(", ")]
            assert {c: r for c, r, _b in items} == {str(k): v for k, v in lane.raw["roles"].items()}, lane.id
            for cls, role, basis in items:
                assert re.match(BASIS_PATTERN, basis), (lane.id, cls, basis)
                assert role in BASIS_TABLE[basis_key(basis)]["grants"], (lane.id, cls, basis)
        else:
            assert roles_lines == [], (lane.id, lines)
    assert authoring.show_stances(LANES["L01"], date="2026-09-29")[0] == L01_ROLES_LINE
    assert "fault alert (task:anomaly)" in authoring.show_stances(LANES["L06"], date="2026-09-29")[0]


def test_a_motif_stance_for_the_lanes_that_state_one():
    assert authoring.show_stances(LANES["L04"], date="2026-09-29") == ["2026-09-29 | wine | show: motif grid (user)"]
    golden = authoring.show_stances(LANES["G00"], date="2026-09-29", dataset="cicids2017")
    assert "2026-09-29 | cicids2017 | show: motif lanes (user)" in golden
    assert authoring.show_stances(LANES["L02"], date="2026-09-29") == []
    fmt = SHOW_CATALOG["stances"]["lines"]
    assert fmt["roles"].startswith("YYYY-MM-DD | <dataset> | show: roles ") and fmt["roles"].endswith("(user)")
    assert fmt["motif"] == "YYYY-MM-DD | <dataset> | show: motif <id> (user)"


def test_write_show_stances_into_the_memory_file(tmp_path):
    """Appended under § Network stances (before the next section), once; the forge compiler's
    reader sees them and its M2 rule is unaffected."""
    memory = tmp_path / ".claude" / "domain-forge-memory.md"
    memory.parent.mkdir()
    memory.write_text("# domain-forge memory\n\n## Network stances\n\n"
                      "- 2026-09-05 | orders | family: mlp-residual (user); dropout B3: 0.2\n"
                      "- 2026-09-05 | orders | loss: bce_with_logits (declared)\n\n"
                      "## Analysis stances\n\n- 2026-09-05 | orders | outliers genuine\n", encoding="utf-8")
    written = authoring.write_show_stances(memory, LANES["L01"], date="2026-09-29")
    assert written == [L01_ROLES_LINE]
    text = memory.read_text(encoding="utf-8")
    assert text.index("show: roles") < text.index("## Analysis stances")
    assert text.count("- 2026-09-05 | orders | loss") == 1
    assert authoring.write_show_stances(memory, LANES["L01"], date="2026-09-30") == [], "written once"
    compile_spec = load_script("vega-forge/scripts/compile_spec.py", "compile_spec_t03")
    stances = compile_spec.read_stances(memory)
    assert written[0] in stances and len(stances) == 3
    spec = spec_for(LANES["L01"])
    spec["loss"]["name"] = "bce_with_logits"

    class Ctx:
        inputs = type("I", (), {"stances": stances})
        family_name = spec["family"]["name"]
    Ctx.spec = spec
    m2 = next(rule.check for rule in compile_spec.RULES if rule.id == "M2")   # the stances rule
    assert m2(Ctx) == [], "a show: line trips no forge memory rule"
    fresh = tmp_path / "new" / ".claude" / "domain-forge-memory.md"
    motif = "2026-09-29 | wine | show: motif grid (user)"
    assert authoring.write_show_stances(fresh, LANES["L04"], date="2026-09-29") == [motif]
    assert f"## Network stances\n\n- {motif}\n" in fresh.read_text(encoding="utf-8")
    none = tmp_path / "none.md"
    assert authoring.write_show_stances(none, LANES["L02"]) == [] and not none.exists()


def test_the_stances_and_check_cli(tmp_path):
    memory = tmp_path / "work" / ".claude" / "domain-forge-memory.md"
    run = subprocess.run([PY, "-m", "lanekit.authoring", "stances", "--lane", "L19", "--memory", str(memory)],
                         cwd=HERE, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith("OK: stance written: ")
    assert "overloaded alert (registry:L19)" in run.stdout
    good = tmp_path / "bundle-layer.json"
    good.write_text(json.dumps(authoring.pack_authoring(bundle_dir(tmp_path, []), LANES["L01"])), encoding="utf-8")
    ok = subprocess.run([PY, "-m", "lanekit.authoring", "check", "--kind", "pack", str(good)], cwd=HERE,
                        capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.startswith("OK:")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"readings": {"abstract": "it scored 0.86"}}), encoding="utf-8")
    refused = subprocess.run([PY, "-m", "lanekit.authoring", "check", "--kind", "pack", str(bad)], cwd=HERE,
                             capture_output=True, text=True)
    assert refused.returncode == 1 and refused.stdout.startswith("ERROR: pack.readings.abstract")


# =========================================================================== through the lane helper
def test_the_author_helper_writes_every_kind(tmp_path):
    """T01's `python -m lanekit.checks author --kind …` (the forge, train, pack and show
    stages) drives these writers: exit 0, the authoring file written, no digit in it."""
    lane = LANES["L01"]
    run = run_dir(tmp_path, lane)
    (run / "experiments.json").write_text(json.dumps(experiments()), encoding="utf-8")
    bundle = bundle_dir(tmp_path, ["user: ship"])
    facts, draft = facts_and_draft(lane, tmp_path, "emitted")
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    (plan / "story-plan.draft.json").write_text(json.dumps(draft), encoding="utf-8")
    targets = {"forge": (run / "architecture-layer.json", []), "train": (run / "training-layer.json", []),
               "pack": (bundle / "bundle-layer.json", []),
               "show": (plan / "story-plan.json", ["--facts", str(plan / "facts.json"),
                                                   "--draft", str(plan / "story-plan.draft.json")])}
    for kind, (target, extra) in targets.items():
        result = subprocess.run([PY, "-m", "lanekit.checks", "author", "--lanes-file", str(HERE / "lanes.json"),
                                 "--lane", "L01", "--lane-dir", str(tmp_path / "lane"), "--skills", str(REPO),
                                 "--kind", kind, "--target", str(target), *extra], cwd=HERE,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert authoring.digit_problems(json.loads(target.read_text(encoding="utf-8")), kind) == [], kind


# =========================================================================== end to end (opt-in)
def _run(argv: list[Any], cwd: Path) -> str:
    result = subprocess.run([str(a) for a in argv], cwd=cwd, capture_output=True, text=True)
    out = result.stdout + result.stderr
    assert result.returncode == 0 and "ERROR" not in result.stdout, out[-3000:]
    return out


@pytest.mark.skipif(not os.environ.get("VEGA_AUTHORING_WORK"),
                    reason="VEGA_AUTHORING_WORK names no finished pack acceptance work dir")
def test_the_writers_through_the_real_apply_scripts(tmp_path):
    """The forge, train and pack writers on a real campaign and bundle (a copy of a finished
    pack acceptance work dir), then the apply scripts, domain-forge's validator and the three
    layer smoke tests — what the lanekit forge/train/pack stages run."""
    src = Path(os.environ["VEGA_AUTHORING_WORK"])
    work = tmp_path / "work"
    shutil.copytree(src, work, symlinks=True)
    lane = LANES["L01"]
    camp = next(p.parent for p in sorted(work.glob("vega-runs/*/experiments.json")))
    bundle = next(p.parent for p in sorted(work.glob("bundle/*/MANIFEST.json")))
    arch = sorted((work / ".claude" / "architect-runs").glob("*/"))[-1]
    base = next(work.glob("*.domain.analysis.shaped.html"))
    stem = base.name[: -len(".html")]
    for kind, target in (("forge", camp / "architecture-layer.json"), ("train", camp / "training-layer.json"),
                         ("pack", bundle / "bundle-layer.json")):
        doc = getattr(authoring, f"{kind}_authoring")(target.parent, lane)
        assert authoring.digit_problems(doc, kind) == []
        target.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    validate = REPO / "domain-forge" / "scripts" / "validate_model.py"
    steps = [
        ("vega-forge/scripts/apply_architecture_layer.py", [base, "--data", camp / "architecture-layer.json",
         "--notebook", camp / "engineer.ipynb", "--architect-run", arch, "--out", f"{stem}.vega.html"],
         "vega-forge/scripts/smoke_architecture.py", f"{stem}.vega.html"),
        ("vega-train/scripts/apply_training_layer.py", [f"{stem}.vega.html", "--data", camp / "training-layer.json",
         "--notebook", camp / "train.ipynb", "--out", f"{stem}.vega.train.html"],
         "vega-train/scripts/smoke_training.py", f"{stem}.vega.train.html"),
        ("vega-eval/scripts/apply_evaluation_layer.py", [f"{stem}.vega.train.html", "--data",
         camp / "evaluation-layer.json", "--notebook", camp / "evaluate.ipynb", "--out", f"{stem}.vega.train.eval.html"],
         None, f"{stem}.vega.train.eval.html"),
        ("vega-pack/scripts/model_card.py", ["--bundle", bundle, "--model", f"{stem}.vega.train.eval.html",
         "--intended-use", bundle / "bundle-layer.json", "--out", bundle / "model-card.html"], None, None),
        ("vega-pack/scripts/apply_bundle_layer.py", [f"{stem}.vega.train.eval.html", "--data", bundle / "bundle-layer.json",
         "--notebook", bundle / "predict.ipynb", "--out", f"{stem}.vega.train.eval.pack.html"],
         "vega-pack/scripts/smoke_bundle.py", f"{stem}.vega.train.eval.pack.html")]
    for script, args, smoke, out in steps:
        _run([PY, REPO / script, *args], work)
        if out:
            _run([PY, validate, out], work)
        if smoke:
            _run([PY, REPO / smoke, out, "--strict"], work)
