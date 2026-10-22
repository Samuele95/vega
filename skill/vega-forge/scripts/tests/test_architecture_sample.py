#!/usr/bin/env python3
"""
test_architecture_sample.py — the `architecture` layer's sample panel (BUILD-PLAN B18; D1 §16,
E20; netspec-contract §4) on layered files of every sample shape.

    python3 -m pytest -q vega-forge/scripts/tests/test_architecture_sample.py
    python3 vega-forge/scripts/tests/test_architecture_sample.py [--work DIR]

Each layered file is built the way the forge stage builds one — `emit_backend.py --smoke`,
`conformance.py`, the forge authoring, `nb_emit.py --stage forge`, `apply_architecture_layer.py`
onto a domain-forge model (the compiler re-run inside it) — from:

  row           the orders fixture (`make_fixture.py`, its recorded architect run spliced)
  window        B15's `window` spike fixture: a gru forecast, L 12, H 3, three series, a lag
                step, autoregressive, previous-part context
  node          B15's `node-keypair`: a gcn at node level, inductive, full batch
  transductive  the same rows and network with `protocol: transductive` and its `user:`
                exposure: the one shape whose panel carries the exposure chip
  sampled       the same rows and network under neighbour sampling (fan-outs 6 and 4)
  graph-level   B15's `graph-level`: a gin with graph_pool, whole graphs batched

and `smoke_architecture.py --strict` must pass on every one: the panel mounts beside the block
diagram with the spec's shape and shows exactly the smoke pass's `sample` record (and, on the
copy that lost its measurements, the declaration and a statement of what is missing). The
render stays within the contract's 60 KB. Seeded render defects (the panel left out, a count
read from the wrong key, an unscored caption claiming a reason the record does not carry,
counts drawn without a sample block, a missing loss formula left unsaid) must make the smoke
fail — a check that cannot fail proves nothing — and `apply_architecture_layer.py` must refuse
a smoke record measured on another spec.

Needs torch, pandas, pyarrow, scikit-learn, nbformat and a headless Chromium ($CHROME, else
found on PATH; without one every test but the render budget's is skipped, said by name). The
layered files are built and smoked a few at a time (`Layered.prebuild`). As a script it keeps
the files under `--work` (default: a temporary directory) and exits 0 when every test passes.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILL = SCRIPTS.parent
SKILLS = SKILL.parent
TESTS = SKILLS / "vega" / "scripts" / "tests"
for _p in (str(HERE), str(HERE / "spike"), str(SCRIPTS), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import architecture_sample as samples  # noqa: E402
import make_spike_fixtures as msf  # noqa: E402
import smoke_architecture as sa  # noqa: E402
from lanekit import authoring  # noqa: E402
from lanekit.model_source import ModelEntity, compose_base_model, write_model_source  # noqa: E402

PY = sys.executable
APPLY = SCRIPTS / "apply_architecture_layer.py"
SMOKE = SCRIPTS / "smoke_architecture.py"
RENDER = SKILL / "assets" / "architecture-render.js"
VALIDATE = SKILLS / "domain-forge" / "scripts" / "validate_model.py"
KNOWN_FLAKE = TESTS / "known_flake.py"

# the forge acceptance fixture's names (vega-forge/scripts/tests/acceptance.sh)
ORDERS_RUN = "vega-runs/2026-09-05T11-03Z"
ORDERS_MODEL = "orders.domain.analysis.shaped.html"
ORDERS_PIPE = "blueprint-runs/2026-09-05T10-12Z/pipeline.json"
ORDERS_ARCH = ".claude/architect-runs/2026-09-05T11-20Z"

SPIKES = {"window": "window", "node": "node-keypair", "graph-level": "graph-level"}
SHAPES = ("row", "window", "node", "transductive", "sampled", "graph-level")
EXPOSURE = {"kind": "features", "rows": "test", "source": "user:b18/transductive"}


class BuildError(RuntimeError):
    """A layered file could not be built; the message names the step and what it printed."""


def _run(argv, what, cwd=None, ok=(0,)):
    proc = subprocess.run([str(a) for a in argv], capture_output=True, text=True, cwd=cwd,
                          timeout=600, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if proc.returncode not in ok:
        raise BuildError(f"{what} exited {proc.returncode}:\n{proc.stdout[-2500:]}\n"
                         f"{proc.stderr[-2500:]}")
    return proc


# ================================================================ the layered files

def _host_model(root: Path, name: str, raw: Path, label: str) -> Path:
    """A domain-forge base model for a spike fixture (the spike has none): one entity holding
    the raw columns, composed by domain-forge's own composer, so the layer lands in a model
    validate_model.py holds to every invariant."""
    frame = pd.read_csv(raw, nrows=50)
    props = [(re.sub(r"[^A-Za-z0-9]", "_", c), "xsd:float" if pd.api.types.is_numeric_dtype(t)
              else "xsd:string", "The value the network learns to predict." if c == label
              else None) for c, t in frame.dtypes.items()]
    overview = (f"# Spike {name}\n\n"
                f"A stand-in model for the vertical-slice fixture `{name}`.\n"
                "It holds one entity whose data properties are the raw columns.\n"
                "The shaper cut these rows into train, valid and test parts.\n"
                "The forge stage composed a network over the shaped parts.\n"
                "The architecture layer records that network and its sample.\n"
                f"The label is `{label}`.\n")
    entity = ModelEntity(base_iri=f"https://example.org/spike/{name}#", name="Record",
                         component="spike", properties=props, overview=overview,
                         comment=f"One raw row of the {name} fixture.")
    html = compose_base_model(write_model_source(root / f"{name}.model-source", entity), SKILLS)
    model = root / f"{name}.domain.html"
    model.write_text(html, encoding="utf-8")
    return model


def _forge(root: Path, name: str, spec: dict, pipeline: Path, model: Path) -> Path:
    """The forge stage on `spec` from the smoke on, in `root/run-<name>`: emit + smoke, the
    mechanical gate, the authoring, the notebook, the layer. Returns the layered file."""
    run = root / f"run-{name}"
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    spec = copy.deepcopy(spec)
    spec["input"]["model"] = str(model)          # the model the layer is written onto
    spec_path = run / "netspec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    package = spec["outputs"]["package"]
    _run([PY, SCRIPTS / "emit_backend.py", "--spec", spec_path, "--pipeline", pipeline,
          "--out", run, "--smoke"], f"{name}: emit_backend.py --smoke")
    _run([PY, SCRIPTS / "conformance.py", "--package", run / package, "--pipeline", pipeline],
         f"{name}: conformance.py")
    doc = authoring.forge_authoring(run, {"task": spec["task"]["kind"]})
    (run / "architecture-layer.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    _run([PY, SCRIPTS / "nb_emit.py", "--stage", "forge", "--from", run, "--out",
          run / "engineer.ipynb"], f"{name}: nb_emit.py --stage forge")
    out = root / f"{name}.domain.vega.html"
    _run([PY, APPLY, model, "--data", run / "architecture-layer.json", "--notebook",
          run / "engineer.ipynb", "--out", out], f"{name}: apply_architecture_layer.py")
    return out


def build_spike(layered: "Layered", name: str, edit=None, base: str | None = None) -> Path:
    """B15's fixture (`base`, default the one `name` stands for) forged as `name`; `edit(spec)`
    makes a variant of its spec. Variants share the shaped fixture and forge in their own run
    directory beside it."""
    fixture = layered.fixture(base or SPIKES[name])
    root = Path(fixture.root)
    spec = json.loads(Path(fixture.spec).read_text(encoding="utf-8"))
    if edit:
        edit(spec)
    model = _host_model(root, name, Path(fixture.raw), fixture.label)
    return _forge(root, name, spec, Path(fixture.pipeline), model)


def _transductive(spec: dict) -> None:
    """The test nodes' features in the training graph, disclosed by a `user:` exposure
    (netspec-contract §1, rule W2)."""
    sample = spec["data_contract"]["sample"]
    sample.update(protocol="transductive", protocol_source=EXPOSURE["source"],
                  exposure=dict(EXPOSURE))
    sample.pop("protocol_catalog", None)


def _sampled(spec: dict) -> None:
    """Neighbour sampling: one fan-out per conv block, seed nodes counted by the batch (R1)."""
    sample = spec["data_contract"]["sample"]
    sample.update(batching="sampled", batching_source="user:b18/sampled", neighbours=[6, 4])
    sample.pop("batching_catalog", None)
    spec["training"]["batch_size"] = 64


def build_orders(layered: "Layered") -> Path:
    """The forge acceptance fixture, steps 0b-5 (the notebook is not executed here)."""
    root = layered.work / "orders"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    shutil.copytree(HERE / "fixtures" / "dot-claude", root / ".claude")
    _run([PY, HERE / "make_fixture.py", root], "orders: make_fixture.py", cwd=root)
    run = root / ORDERS_RUN
    _run([PY, SCRIPTS / "emit_backend.py", "--spec", "netspec.json", "--pipeline", ORDERS_PIPE,
          "--out", run, "--smoke"], "orders: emit_backend.py --smoke", cwd=root)
    _run([PY, SCRIPTS / "conformance.py", "--package", run / "orders_pipeline", "--pipeline",
          ORDERS_PIPE], "orders: conformance.py", cwd=root)
    shutil.copy2(root / "netspec.json", run / "netspec.json")
    shutil.copy2(HERE / "architecture-layer.authoring.json", run / "architecture-layer.json")
    _run([PY, SCRIPTS / "nb_emit.py", "--stage", "forge", "--from", run, "--out",
          run / "engineer.ipynb"], "orders: nb_emit.py --stage forge", cwd=root)
    out = root / "orders.domain.analysis.shaped.vega.html"
    _run([PY, APPLY, ORDERS_MODEL, "--data", run / "architecture-layer.json", "--notebook",
          run / "engineer.ipynb", "--architect-run", ORDERS_ARCH, "--out", out],
         "orders: apply_architecture_layer.py", cwd=root)
    return out


BUILDERS = {"row": build_orders,
            "transductive": lambda lay: build_spike(lay, "transductive", _transductive,
                                                    "node-keypair"),
            "sampled": lambda lay: build_spike(lay, "sampled", _sampled, "node-keypair"),
            **{n: (lambda lay, _n=n: build_spike(lay, _n)) for n in SPIKES}}


class Layered:
    """The layered files, built once per session under one work directory."""

    def __init__(self, work: Path):
        self.work = Path(work)
        self.files: dict[str, Path] = {}
        self.smoked: dict[str, tuple] = {}
        self.fixtures: dict[str, msf.Fixture] = {}
        # the forge authoring reads stances and architect runs from the nearest `.claude/`: an
        # empty one here keeps a `.claude/` above the work dir out of the spike fixtures' layers
        (self.work / ".claude").mkdir(parents=True, exist_ok=True)

    def __call__(self, shape: str) -> Path:
        if shape not in self.files:
            self.files[shape] = BUILDERS[shape](self)
        return self.files[shape]

    def fixture(self, base: str) -> "msf.Fixture":
        """B15's spike fixture `base`, made once under the work directory."""
        if base not in self.fixtures:
            self.fixtures[base] = msf.make(base, self.work / "spike")
        return self.fixtures[base]

    def data(self, shape: str) -> dict:
        doc, _ = sa.layer_data(self(shape).read_text(encoding="utf-8"))
        return doc

    def smoke(self, shape: str) -> tuple:
        """`smoke_architecture.py --strict` on the shape's layered file (run once)."""
        if shape not in self.smoked:
            self.smoked[shape] = smoke(self(shape))
        return self.smoked[shape]

    def prebuild(self, workers: int = 3) -> None:
        """Every layered file and its smoke, a few at a time: the spike fixtures are made one
        after the other first (each runs the shaper), then each shape forges in its own run
        directory and each smoke in its own browser, so they do not share a file."""
        for base in sorted(set(SPIKES.values())):
            self.fixture(base)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            built = dict(zip(SHAPES, pool.map(lambda s: BUILDERS[s](self), SHAPES)))
            self.files.update(built)
            self.smoked.update(zip(SHAPES, pool.map(smoke, (built[s] for s in SHAPES))))


def _chrome():
    return sa.find_chrome()


@pytest.fixture(scope="module")
def layered():
    if not _chrome():
        pytest.skip("no headless browser: smoke_architecture.py needs one ($CHROME or PATH)")
    keep = os.environ.get("VEGA_ARCH_SAMPLE_WORK")
    if keep:
        lay = Layered(Path(keep))
        lay.prebuild()
        yield lay
        return
    with tempfile.TemporaryDirectory(prefix="b18-sample-") as td:
        lay = Layered(Path(td))
        lay.prebuild()
        yield lay


def smoke(path: Path):
    proc = subprocess.run([PY, SMOKE, path, "--strict"], capture_output=True, text=True,
                          timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def _ok(out: str, name: str) -> bool:
    return re.search(rf"^\s+OK\s+{re.escape(name)}$", out, re.M) is not None


# ================================================================ the render's budget

def test_render_within_the_contract_budget():
    size = len(RENDER.read_text(encoding="utf-8").rstrip().encode("utf-8"))
    assert size <= sa.MAX_RENDER, f"architecture-render.js is {size} bytes (> 60 KB)"


# ================================================================ the panel, per shape

PANEL_KEYS = {
    "row": ("shape", "rows_train"),
    "window": ("shape", "length", "stride", "horizon", "series", "windows_train",
               "windows_valid", "unscored_valid", "context", "autoregressive"),
    "node": ("shape", "level", "protocol", "batching", "train_nodes", "train_edges",
             "train_targets", "valid_nodes", "valid_edges", "valid_targets",
             "test_edges_withheld"),
    "graph-level": ("shape", "level", "protocol", "batching", "train_nodes", "train_edges",
                    "train_targets", "valid_nodes", "valid_edges", "valid_targets",
                    "test_edges_withheld"),
}
PANEL_KEYS["transductive"] = PANEL_KEYS["node"] + ("edges_into_test",)
PANEL_KEYS["sampled"] = PANEL_KEYS["node"] + ("neighbours",)
WANT_SHAPE = {"row": "row", "window": "window", "node": "graph", "transductive": "graph",
              "sampled": "graph", "graph-level": "graph"}


@pytest.mark.parametrize("shape", SHAPES)
def test_the_sample_panel_mounts_and_shows_the_smoke_record(layered, shape):
    doc = layered.data(shape)
    sample = doc["smoke"]["sample"]
    assert sample["shape"] == WANT_SHAPE[shape], sample
    code, out = layered.smoke(shape)
    assert code == 0, out[-4000:]
    assert _ok(out, "render-within-60-KB"), out[-2000:]
    for label in ("full", "no-measurements"):
        for name in ("sample-panel-mounted", "sample-panel-beside-the-diagram",
                     "sample-panel-shape", "sample-missing-said-iff-missing",
                     "sample-exposure-chip", "hook-sample", "view-sample"):
            assert _ok(out, f"{label}:{name}"), f"{label}:{name} not checked\n{out[-3000:]}"
    for key in PANEL_KEYS[shape]:
        assert _ok(out, f"full:sample-{key}"), f"full:sample-{key} not checked\n{out[-3000:]}"
    assert _ok(out, "full:madds-unit-named"), out[-2000:]
    if WANT_SHAPE[shape] != "row":
        # the copy without a sample block shows the declaration and no count
        assert _ok(out, "no-measurements:sample-no-counts-without-a-record"), out[-3000:]


def test_window_panel_values(layered):
    s = layered.data("window")["smoke"]["sample"]
    assert (s["length"], s["stride"], s["horizon"], s["series"]) == (12, 1, 3, 3)
    assert s["context"] == "previous-part" and s["autoregressive"] is True
    assert s["windows"]["train"] > 0 and s["windows"]["valid"] > 0
    assert layered.data("window")["smoke"]["madds_unit"] == "anchor"


def test_the_unscored_caption_follows_the_declaration(layered, tmp_path):
    """The record counts the valid rows left unscored, not why (D1 §9.3): on the window file
    (previous-part context, H 3) the caption names the horizon past the part's end first, and
    the smoke holds it to the declaration — the old fixed caption ('too early in their
    series') claimed a reason the record cannot give, and must fail."""
    code, out = layered.smoke("window")
    assert code == 0 and _ok(out, "full:sample-unscored_valid-caption"), out[-3000:]
    assert samples.unscored_note("previous-part", 3).startswith("a horizon past")
    assert "horizon" not in samples.unscored_note("none", 0)
    assert "horizon" in samples.unscored_note("none", 6)
    bad = _mutated(layered("window"), "unscoredNote(ctx, H))",
                   "'too early in their series for a window')", tmp_path / "caption.html")
    code, out = smoke(bad)
    assert code == 1 and "FAIL full:sample-unscored_valid-caption" in out, out[-3000:]


def test_graph_panels_say_the_protocol(layered):
    node, level = layered.data("node")["smoke"], layered.data("graph-level")["smoke"]
    trans = layered.data("transductive")["smoke"]
    assert node["sample"]["exposure"] is None and node["sample"]["test_edges_withheld"] > 0
    assert node["madds_unit"] == "target_node" and level["madds_unit"] == "graph"
    assert level["sample"]["level"] == "graph" and level["sample"]["batching"] == "graphs"
    exposure = trans["sample"]["exposure"]
    assert exposure["features_seen_in_training"] is True and exposure["edges_into_test"] > 0
    assert trans["sample"]["test_edges_withheld"] == 0
    # the exposure chip is drawn for the transductive file only (smoke: 1 chip vs 0)
    code, out = layered.smoke("transductive")
    assert code == 0 and _ok(out, "full:sample-exposure-names-the-edges-into-test"), out[-3000:]


def test_a_sampled_graph_names_its_fan_outs(layered):
    sample = layered.data("sampled")["smoke"]["sample"]
    assert sample["batching"] == "sampled" and sample["exposure"] is None
    code, out = layered.smoke("sampled")
    assert code == 0 and _ok(out, "full:sample-neighbours"), out[-3000:]


def test_a_sequence_window_shows_no_horizon(layered, tmp_path):
    """A sequence task's window has no horizon (its target is the window's last row): on a copy
    of the window file recast as a sequence regression, the panel shows none, and the smoke
    says so."""
    html = layered("window").read_text(encoding="utf-8")
    doc, m = sa.layer_data(html)
    spec, measured = doc["netspec"], doc["smoke"]["sample"]
    spec["task"]["kind"] = "sequence-regression"
    spec["data_contract"]["sample"].pop("horizon", None)
    spec["data_contract"]["sample"]["autoregressive"] = False
    measured.update(horizon=0, autoregressive=False)
    copy_path = tmp_path / "sequence.vega.html"
    copy_path.write_text(html[:m.start(2)] + json.dumps(doc) + html[m.end(2):], encoding="utf-8")
    code, out = smoke(copy_path)
    assert code == 0, out[-3000:]
    assert _ok(out, "full:sample-no-horizon-off-forecast"), out[-3000:]
    assert "full:sample-horizon" not in out and _ok(out, "full:sample-autoregressive"), out[-3000:]


def test_noscript_carries_the_sample(layered):
    text = layered("window").read_text(encoding="utf-8")
    noscript = text[text.index("layer-architecture-noscript"):]
    assert "The sample: a window of 12 rows, stride 1, horizon 3" in noscript
    assert "MAdds per anchor" in noscript
    text = layered("transductive").read_text(encoding="utf-8")
    assert "exposure disclosed" in text[text.index("layer-architecture-noscript"):]


def test_the_layered_files_validate(layered):
    """domain-forge's validator (invariants 13-16 are the layer's) on a window and a graph
    file, through the known-flake wrapper the acceptance scripts use."""
    for shape in ("window", "graph-level"):
        proc = subprocess.run([PY, KNOWN_FLAKE, "run", "--stage", "forge", "--", PY, VALIDATE,
                               layered(shape)], capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]


# ================================================================ the smoke can fail

def _mutated(path: Path, old: str, new: str, dest: Path) -> Path:
    html = path.read_text(encoding="utf-8")
    render = sa.RENDER_RE.search(html)
    assert render and old in render.group(1), f"the render no longer carries {old!r}"
    body = render.group(1).replace(old, new, 1)
    dest.write_text(html[:render.start(1)] + body + html[render.end(1):], encoding="utf-8")
    return dest


def test_the_smoke_fails_without_the_panel(layered, tmp_path):
    bad = _mutated(layered("window"), "split.appendChild(samplePanel(SPEC, SM));", "",
                   tmp_path / "no-panel.html")
    code, out = smoke(bad)
    assert code == 1, out[-3000:]
    assert "FAIL full:sample-panel-mounted" in out and "FAIL full:view-sample" in out, out[-3000:]


def test_the_smoke_fails_on_a_count_from_the_wrong_key(layered, tmp_path):
    bad = _mutated(layered("node"), "num(t[r[0]])", "num(v[r[0]])", tmp_path / "swapped.html")
    code, out = smoke(bad)
    assert code == 1 and "FAIL full:sample-train_edges" in out, out[-3000:]


def test_the_smoke_fails_on_counts_without_a_record(layered, tmp_path):
    """A render that draws the window counts when the smoke record has no sample block (here
    as middle dots) is caught on the degraded copy, whatever it says about the missing block."""
    bad = _mutated(layered("window"), "if (!missing) {\n      var w = M.windows",
                   "if (true) {\n      var w = M.windows", tmp_path / "stale.html")
    code, out = smoke(bad)
    assert code == 1, out[-3000:]
    assert "FAIL no-measurements:sample-no-counts-without-a-record" in out, out[-3000:]
    assert _ok(out, "no-measurements:sample-missing-said-iff-missing"), out[-3000:]


def test_a_missing_loss_formula_is_said(layered, tmp_path):
    """Neither the smoke record nor the spec carries the loss formula: the mechanics panel
    writes no formula for it and says so, and a render that stays silent fails the smoke."""
    html = layered("window").read_text(encoding="utf-8")
    doc, m = sa.layer_data(html)
    doc["smoke"].pop("loss_formula", None)
    (doc["netspec"].get("loss") or {}).pop("formula", None)
    bare = tmp_path / "no-formula.vega.html"
    bare.write_text(html[:m.start(2)] + json.dumps(doc) + html[m.end(2):], encoding="utf-8")
    code, out = smoke(bare)
    assert code == 0 and _ok(out, "full:missing-loss-formula-said"), out[-3000:]
    assert _ok(out, "full:loss-formula-and-update-rule"), out[-3000:]
    silent = _mutated(bare, "'Neither the smoke record nor the spec carries the loss formula.'",
                      "''", tmp_path / "silent.html")
    code, out = smoke(silent)
    assert code == 1 and "FAIL full:missing-loss-formula-said" in out, out[-3000:]


# ================================================================ the apply script's refusals

def _apply(layered, shape, tmp_path, edit):
    """apply_architecture_layer.py on the shape's run with its smoke record edited."""
    root = layered(shape).parent
    run = root / f"run-{shape}"
    doc = json.loads((run / "architecture-layer.json").read_text(encoding="utf-8"))
    smoke_rec = json.loads((run / "smoke.json").read_text(encoding="utf-8"))
    edit(smoke_rec)
    doc = dict(doc, smoke=smoke_rec, from_run=str(run))
    data = tmp_path / "layer.json"
    data.write_text(json.dumps(doc), encoding="utf-8")
    model = root / f"{shape}.domain.html"
    proc = subprocess.run([PY, APPLY, model, "--data", data, "--notebook", run / "engineer.ipynb",
                           "--skip-compile", "--out", tmp_path / "out.html"],
                          capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def test_apply_refuses_a_smoke_record_of_another_spec(layered, tmp_path):
    graph_sample = layered.data("node")["smoke"]["sample"]
    code, out = _apply(layered, "window", tmp_path, lambda s: s.update(sample=graph_sample))
    assert code == 2 and "smoke.sample.shape is 'graph'" in out, out[-2000:]
    code, out = _apply(layered, "window", tmp_path,
                       lambda s: s["sample"].update(length=24))
    assert code == 2 and "smoke.sample.length is 24" in out, out[-2000:]
    code, out = _apply(layered, "graph-level", tmp_path,
                       lambda s: s.update(madds_unit="target_node"))
    assert code == 2 and "counts per 'graph'" in out, out[-2000:]
    code, out = _apply(layered, "node", tmp_path, lambda s: s["sample"].pop("train_view"))
    assert code == 2 and "smoke.sample lacks train_view" in out, out[-2000:]


def test_apply_warns_on_a_smoke_record_without_a_sample(layered, tmp_path):
    code, out = _apply(layered, "window", tmp_path, lambda s: s.pop("sample"))
    assert code == 0, out[-2000:]
    assert "WARN: smoke.json carries no `sample` block" in out, out[-2000:]


# ================================================================ as a script

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", default=None, help="keep the layered files here")
    a = ap.parse_args(argv)
    if a.work:
        os.environ["VEGA_ARCH_SAMPLE_WORK"] = str(Path(a.work).resolve())
        Path(a.work).mkdir(parents=True, exist_ok=True)
    return pytest.main(["-q", "-p", "no:cacheprovider", __file__])


if __name__ == "__main__":
    sys.exit(main())
