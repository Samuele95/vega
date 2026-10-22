"""
Unit tests of the lane program itself (T01), part 3: its parts — the command runner and the
flake policy, observed.json, the service harness, the skills tree, `--changed` routing, and
the stage assertions (lanekit.assertions).

Nothing here trains a network: the lanes these tests drive are synthetic stage functions over
temporary directories, so the suites run in seconds. The real lanes run under matrix.sh.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lanekit_testkit import PY, REPO, SIGNATURE, fake_validator  # noqa: E402
from lanekit import (assertions, contracts, observe, registry, routing, runner, service,  # noqa: E402
                     skilltree)


# --------------------------------------------------------------------------- the runner and the flake policy
def make_runner(tmp: Path, seconds: float | None = None) -> tuple[runner.CommandRunner, runner.FlakeRecorder]:
    recorder = runner.FlakeRecorder(stage="eval")
    deadline = runner.Deadline(time.monotonic() + seconds if seconds else None)
    run = runner.CommandRunner(os.environ, tmp / "log" / "eval.log", deadline,
                               runner.Flakes.load(), recorder)
    return run, recorder


def test_seeded_errno39_is_retried_once_and_recorded(tmp_path):
    script = fake_validator(tmp_path, fail_times=1)
    run, recorder = make_runner(tmp_path)
    result = run.run([PY, script, tmp_path / "count"])
    assert result.ok and result.flake == {"id": "df-profile-race", "second_attempt": "pass"}
    assert (tmp_path / "count").read_text() == "2"                      # exactly one retry
    assert recorder.retries == [{"id": "df-profile-race", "stage": "eval", "command": result.command,
                                 "retried": True, "second_attempt": "pass", "via": "runner"}]
    log = (tmp_path / "log" / "eval.log").read_text()
    assert log.count("$ ") == 2 and "WARN: flake df-profile-race fired; retrying once" in log
    assert recorder.calls == {"df-profile-race": 1}


def test_a_second_failure_fails_and_is_recorded(tmp_path):
    script = fake_validator(tmp_path, fail_times=2)
    run, recorder = make_runner(tmp_path)
    result = run.run([PY, script, tmp_path / "count"])
    assert not result.ok and (tmp_path / "count").read_text() == "2"
    assert recorder.retries[0]["second_attempt"] == "fail"


def test_other_failures_are_never_retried(tmp_path):
    script = fake_validator(tmp_path, fail_times=1, output="OSError: [Errno 2] No such file\n")
    run, recorder = make_runner(tmp_path)
    assert not run.run([PY, script, tmp_path / "count"]).ok
    assert (tmp_path / "count").read_text() == "1" and recorder.retries == []


def test_negatives_are_never_retried(tmp_path):
    script = tmp_path / "run_negatives.py"
    script.write_text(f"import sys\nsys.stdout.write({SIGNATURE!r} + 'validate_model.py')\nsys.exit(1)\n")
    run, recorder = make_runner(tmp_path)
    assert not run.run([PY, script, "validate_model.py"]).ok
    assert recorder.retries == []


def test_retry_is_off_when_asked(tmp_path):
    script = fake_validator(tmp_path, fail_times=1)
    run, recorder = make_runner(tmp_path)
    assert not run.run([PY, script, tmp_path / "count"], retry=False).ok and recorder.retries == []


def test_inner_retry_is_seen_in_a_passing_output():
    flakes = runner.Flakes.load()
    assert flakes.fired_in("step 7\n" + SIGNATURE + "OK: 15 passed") == ["df-profile-race"]
    assert flakes.fired_in("WARN: flake df-profile-race fired; retrying once: python3 v.py") == ["df-profile-race"]
    x1 = ("WARN: known flake df-profile-race (owner domain-forge; upstream patches/domain-forge/"
          "README.md): validate_model.py … failed with its signature (exit 1); retrying once")
    assert flakes.fired_in(x1) == ["df-profile-race"]
    assert flakes.fired_in("OK: all green") == []


def test_a_command_over_its_deadline_is_killed(tmp_path):
    run, _ = make_runner(tmp_path, seconds=1.0)
    started = time.monotonic()
    result = run.run([PY, "-c", "import time; time.sleep(30)"])
    assert result.timed_out and not result.ok and time.monotonic() - started < 10
    assert "ERROR: killed after" in (tmp_path / "log" / "eval.log").read_text()


def test_markers_are_counted_with_prefixes():
    m = runner.Markers.of("OK: a\n  shaper| WARN: b\nERROR: c\nFORK: d\nNOTE: e\nnot OK: f\n")
    assert m.to_json() == {"ok": 1, "warn": 1, "error": 1, "fork": 1}
    # a stage prefix with a space (gate C: L02's fixture stage recorded warn 0 for one present)
    spaced = ("  geometry layer| WARN: partitions.candidates[0] (cover): the label is neither the head "
              "of a kept rule nor determined by a functional dependency\n  geometry layer| OK: basis\n")
    assert runner.Markers.of(spaced).to_json() == {"ok": 1, "warn": 1, "error": 0, "fork": 0}
    assert runner.warn_lines(spaced) == [spaced.splitlines()[0].strip()]
    assert runner.error_lines("  data lens| ERROR: x\n") == ["data lens| ERROR: x"]
    assert runner.Markers.of("not OK: f\nthe OK: count\n").to_json()["ok"] == 0


def test_a_per_element_metric_counts_as_measured():
    """gate C: f1_per_class, per_horizon_error, per_feature_error and residual_quantiles are
    vectors (value null, values a list) and were reported as recorded by no lane."""
    from lanekit import observe
    assert observe._measured({"value": 0.7, "ci": [0.6, 0.8]})
    assert observe._measured({"value": None, "values": [0.9, None, 0.2]}), "an undefined element is null"
    assert not observe._measured({"value": None, "values": []})
    assert not observe._measured({"value": None, "values": [None, None]})
    assert not observe._measured({"value": None, "values": [0.1, float("nan")]})
    assert not observe._measured({"value": float("inf")})
    assert not observe._measured({"value": True})


# --------------------------------------------------------------------------- observed.json
def test_observed_reads_the_packed_spec(tmp_path):
    lane_dir = tmp_path / "L77"
    bundle = lane_dir / "work" / "bundle" / "demo-R0"
    camp = lane_dir / "work" / "vega-runs" / "T1"
    for path in (bundle, camp / "eval" / "R0"):
        path.mkdir(parents=True)
    spec = {"task": {"kind": "binary"}, "family": {"name": "mlp-residual"},
            "data_contract": {"sample": {"shape": "row"}},
            "blocks": [{"id": "B1", "type": "residual", "params": {"inner": [
                {"type": "dense", "params": {"units": 4, "activation": "none"}},
                {"type": "activation", "params": {"name": "gelu"}}]}},
                {"id": "B2", "type": "head", "params": {"out": 1, "activation": "none"}}],
            "loss": {"name": "bce_with_logits"}, "optimizer": {"name": "adamw"},
            "schedule": {"name": "cosine"}, "init": {"scheme": "kaiming_uniform"}}
    (bundle / "netspec.json").write_text(json.dumps(spec))
    (bundle / "MANIFEST.json").write_text(json.dumps({"serve": {"endpoints": ["POST /v1/predict"]}}))
    (camp / "test_seal.json").write_text(json.dumps({"run": "R0"}))
    (camp / "experiments.json").write_text(json.dumps({"runs": [{"id": "R0", "status": "done"},
                                                                  {"id": "R1", "status": "dropped"}]}))
    (camp / "eval" / "R0" / "metrics.json").write_text(json.dumps({"models": {
        "network": {"pr_auc": {"value": 0.8}, "ece": {"value": None, "reason": "n/a"}},
        "trivial": {}, "linear": {}}, "verdict": {"earned": True}}))
    run = {"git": {"commit": "abc", "dirty": False}, "started_at": "2026-09-29T12:00:00Z",
           "finished_at": "2026-09-29T12:01:00Z", "verdict": "pass", "stages": [], "flakes": []}
    doc = observe.build("L77", lane_dir, lane_dir / "work", run)
    assert observe.validate(doc) == []
    got = doc["observed"]
    assert got["spec_source"] == "bundle" and got["blocks"] == ["residual", "dense", "activation", "head"]
    assert got["activations"] == ["gelu", "none"] and got["init"] == "kaiming_uniform"
    assert got["metrics_finite"] == ["pr_auc"] and got["baselines"] == ["linear", "trivial"]
    assert got["runs"] == 1 and got["selected_run"] == "R0" and got["verdict"] == "earned"
    assert got["endpoints"] == ["POST /v1/predict"] and got["goal"] == "predict"
    doc["observed"]["runs"] = "one"
    assert observe.validate(doc)


def test_observed_reads_the_golden_lanes_campaign_and_slot(tmp_path):
    """R3: G00 migrates its campaign to <work>/campaign and builds its own showcase (no show
    stage), so the nightly coverage found neither its verdict nor its motif."""
    lane_dir = tmp_path / "G77"
    bundle, camp = lane_dir / "work" / "bundle" / "demo-R7", lane_dir / "work" / "campaign"
    for path in (bundle / "showcase", camp / "eval" / "R7"):
        path.mkdir(parents=True)
    (bundle / "MANIFEST.json").write_text("{}")
    (camp / "test_seal.json").write_text(json.dumps({"run": "R7"}))
    (camp / "eval" / "R7" / "metrics.json").write_text(json.dumps({
        "models": {"network": {"f1_macro": {"value": 0.8}}}, "verdict": {"earned": True}}))
    slot = bundle / "showcase"
    (slot / "SHOWCASE.json").write_text(json.dumps({"story": {"file": "story.json"},
                                                    "page": {"file": "page.json"}, "report": {"status": "linked"}}))
    (slot / "story.json").write_text(json.dumps({"class_roles": {"roles": {"BENIGN": "plain", "Bot": "alert"}}}))
    (slot / "page.json").write_text(json.dumps({"motif": {"id": "lanes", "base": "points"}}))
    run = {"git": {"commit": "abc", "dirty": False}, "started_at": "2026-10-09T12:00:00Z",
           "finished_at": "2026-10-09T12:01:00Z", "verdict": "pass", "stages": [], "flakes": []}
    doc = observe.build("G77", lane_dir, lane_dir / "work", run)
    assert observe.validate(doc) == []
    got = doc["observed"]
    assert got["verdict"] == "earned" and got["selected_run"] == "R7"
    assert got["showcase"] == {"motif": "lanes", "class_roles": [("BENIGN", "plain"), ("Bot", "alert")],
                               "report": "linked", "locale": "en"}
    (lane_dir / "results").mkdir()
    (lane_dir / "results" / "show.json").write_text(json.dumps({"motif": "points", "class_roles": [],
                                                                 "report": None, "locale": "en"}))
    assert observe.build("G77", lane_dir, lane_dir / "work", run)["observed"]["showcase"]["motif"] == "points", \
        "a show stage's checked result wins over the slot"


# --------------------------------------------------------------------------- the service harness
FAKE_APP = textwrap.dedent("""\
    import argparse, json, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    STARTED = time.monotonic()
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            ready = time.monotonic() - STARTED > 0.6
            body = json.dumps({"ready": ready, "showcase": {"verified": False}}).encode()
            self.send_response(200 if ready else 503)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int)
    ap.add_argument("--torch-threads", type=int)
    a = ap.parse_args()
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()
    """)


def test_service_waits_for_ready_and_stops(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "serve").mkdir(parents=True)
    (bundle / "serve" / "__init__.py").write_text("")
    (bundle / "serve" / "app.py").write_text(FAKE_APP)
    svc = service.start(bundle, threads=2, launch=service.Launch(PY, tmp_path / "service.log"))
    try:
        status, health = svc.get_json(contracts.HEALTH_ROUTE)
        assert status == 200 and health["ready"] is True
    finally:
        service.stop(svc)
    assert not svc.alive()


def test_a_service_that_dies_is_an_error(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "serve").mkdir(parents=True)
    (bundle / "serve" / "__init__.py").write_text("")
    (bundle / "serve" / "app.py").write_text("raise SystemExit('ERROR: no weights')\n")
    with pytest.raises(service.ServiceError, match="exited"):
        service.start(bundle, launch=service.Launch(PY, tmp_path / "service.log"))


# --------------------------------------------------------------------------- the skills tree
def test_skill_tree_copies_tracked_files_and_pins_domain_forge(tmp_path):
    repo = tmp_path / "repo"
    (repo / "vega-forge" / "scripts" / "__pycache__").mkdir(parents=True)
    (repo / "vega-forge" / "scripts" / "a.py").write_text("print('a')\n")
    (repo / "vega-forge" / "scripts" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    (repo / "notes.tmp").write_text("ignored\n")
    (repo / ".gitignore").write_text("*.tmp\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "vega-forge/scripts/a.py", ".gitignore"], check=True)
    (repo / "vega-forge" / "scripts" / "b.py").write_text("print('b, untracked but not ignored')\n")
    df = tmp_path / "domain-forge"
    for rel in contracts.DOMAIN_FORGE_PINNED:
        (df / rel).parent.mkdir(parents=True, exist_ok=True)
        (df / rel).write_text(f"# {rel}\n")
    matrix = tmp_path / "matrix"
    tree = skilltree.build(matrix, repo, {"DOMAIN_FORGE_DIR": str(df)})
    copied = sorted(str(p.relative_to(tree.root)) for p in tree.root.rglob("*") if p.is_file())
    assert "vega-forge/scripts/a.py" in copied and "vega-forge/scripts/b.py" in copied
    assert not [c for c in copied if "__pycache__" in c or c.endswith(".tmp")]
    assert not any(p.is_symlink() for p in tree.root.rglob("*"))
    assert set(tree.pins) == set(contracts.DOMAIN_FORGE_PINNED)
    assert skilltree.check_pins(matrix, tree) == []
    skilltree.record_green(matrix)
    (df / "scripts" / "validate_model.py").write_text("# changed upstream\n")
    tree = skilltree.build(matrix, repo, {"DOMAIN_FORGE_DIR": str(df)})
    warns = skilltree.check_pins(matrix, tree)
    assert len(warns) == 1 and warns[0].startswith("WARN: domain-forge changed since the last green run: scripts/validate_model.py")


def test_tree_digest_excludes_the_showcase(tmp_path):
    (tmp_path / "showcase").mkdir()
    (tmp_path / "MANIFEST.json").write_text("{}")
    before = skilltree.tree_digest(tmp_path, exclude_top=("showcase",))
    (tmp_path / "showcase" / "SHOWCASE.json").write_text("{}")
    assert skilltree.tree_digest(tmp_path, exclude_top=("showcase",)) == before
    (tmp_path / "MANIFEST.json").write_text('{"x": 1}')
    assert skilltree.tree_digest(tmp_path, exclude_top=("showcase",)) != before


# --------------------------------------------------------------------------- --changed routing
def test_changed_paths_route_lanes():
    lanes = registry.load_lanes()
    route = lambda *paths: sorted(registry.affected_lanes(lanes, paths, REPO, None))
    assert route("vega-forge/references/family-catalog.md") == []
    assert route(contracts.SERVING_TEMPLATES + "/serve/app.py") == sorted(lanes)
    assert route("vega-show/assets/page/index.html") == sorted(lanes)
    assert route(contracts.LANEKIT_DIR + "/datasets/tank.py") == ["L14", "L15", "L16", "L17"]
    assert route("vega-train/scripts/trainer.py") == sorted(lanes)


def test_block_segments_route_the_lanes_that_use_them(tmp_path, monkeypatch):
    lanes = registry.load_lanes()
    blocks = tmp_path / "vega-forge" / "blocks_seq.py"
    blocks.parent.mkdir(parents=True)
    blocks.write_text("# --8<-- gru\nclass GRU: pass\n# --8<-- attention_pool\nclass AttentionPool: pass\n")
    monkeypatch.setattr(routing, "_changed_lines", lambda repo, path, rev: [4])
    got = sorted(registry.affected_lanes(lanes, ["vega-forge/blocks_seq.py"], tmp_path, None))
    assert got == ["L17", "L18"]




# --------------------------------------------------------------------------- the stage assertions
WANT = {"kind": "features", "rows": "test"}


@pytest.mark.parametrize("layer, ok", [
    ({"graph": {"exposure": {"kind": "features", "rows": "test", "source": "user:stance/papers"}}}, True),
    ({"sample": {"exposure": dict(WANT)}, "records": [{"exposure": dict(WANT)}]}, True),
    ({"notes": "the exposure of test features", "kind": "features"}, False),     # words are not a record
    ({"graph": {"exposure": None}}, False),
    ({"graph": {"exposure": {"kind": "features", "rows": "valid"}}}, False),
    ({"sample": {"exposure": dict(WANT)}, "graph": {"exposure": {"kind": "labels", "rows": "test"}}}, False),
])
def test_the_exposure_record_is_compared(layer, ok):
    """D6 §6.4: the evaluation layer records the lane's `expect.exposure`, compared key by key."""
    problem = assertions.exposure_problem(layer, WANT)
    assert (problem is None) == ok, problem


def test_forge_says_the_shape_oracle_did_not_run(tmp_path, capsys):
    """Without oracles.py (T08) the per-block shape check is a counted `WARN: check not run`,
    never an OK, and the rest of the forge assertions still run."""
    raw = next(e for e in registry.load_document()["lanes"] if e["id"] == "L02")
    lane = registry.Lane("L02", raw)
    run_dir, lane_dir = tmp_path / "run", tmp_path / "lane"
    run_dir.mkdir()
    lane_dir.mkdir()
    (run_dir / "smoke.json").write_text(json.dumps({"forward_ok": True}))
    (run_dir / "netspec.json").write_text(json.dumps({"blocks": raw["blocks"]}))
    (lane_dir / "fixture.json").write_text(json.dumps({"run_dir": str(run_dir)}))
    check = assertions.Checker()
    assertions.check_forge(assertions.Ctx(lane, lane_dir, tmp_path / "skills", None), check)
    out = capsys.readouterr().out
    assert check.failed == 0 and "OK: the forged blocks are the lane's" in out
    assert runner.not_run_lines(out) == ["the per-block shape oracle (D6 §6.2) — oracles.py arrives with T08"]
    assert runner.Markers.of(out).warn == 1 and "OK: per-block" not in out
