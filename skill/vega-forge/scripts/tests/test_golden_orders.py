#!/usr/bin/env python3
"""
test_golden_orders.py — the orders fixture's R0 history through the v2.3 `Trainer`, bit for bit
against the golden B01 captured on the pre-refactor tree (BUILD-PLAN B12; D1 §17.3).

    python3 -m pytest test_golden_orders.py [-q]        (or: python3 test_golden_orders.py)

The fixture is built by `make_fixture.py`; its train and valid parts must be the bytes the
capture trained on (their sha256 are in `golden/orders-valid-logit-range.json`), or nothing
below could be equal and the test says so first. The spec is the fixture's `netspec.json` with
the capture's overrides (`training.max_epochs = 3`, `training.device = cpu`). The v2.3 templates
(`domain`, `network`, `readout`, `metrics`, `frames`, `data`, `loop`, `optim`, `record`,
`engines`, and the spliced `builder`, `blocks_*`, `losses`) are rendered with
`render_templates.py` into a package, the network is built by `Network.build`, and
`Trainer().fit(model, spec, ShapedParts(), callbacks)` runs with every other argument at its
default — the capture's `fit(model, spec, data, callbacks)`.

What must be identical, per epoch (the golden history less the wall clock `time_s`):

  * `epoch`, `lr`, `train` (the loss, the label metrics a streamed epoch computes from its
    `z > 0` reduction, and which metrics it reports unavailable and why), `grad_norm` (global,
    clipped, per block), `weight_norm`, `update_ratio`, `nan`, `early_stopping` and `shuffle`
    (the streamed source is v2.2's path: a buffered shuffle seeded `seed + epoch`);
  * `valid.loss` and the rank metrics `valid.roc_auc` / `valid.pr_auc`;
  * the other valid metrics (log_loss, brier, the threshold metrics) on every epoch the
    capture found a valid logit outside [0, 1]. On an epoch whose logits all lay in [0, 1] the
    old metrics read them as probabilities (E18), so there they are *expected to differ (E18
    fixed)* and are listed, not compared — the golden's `expected_to_differ_e18`;
  * the fit's result (epochs run, best epoch and value, what stopped it), and the network's
    `param_count()` and `block_shapes()` against `golden/orders-smoke.json`.

The golden was captured on torch 2.14 (CPU, 20 threads, deterministic algorithms). The fit
runs on the capture's intra-op thread count whatever the process was started with (a CPU float
reduction is split by the thread count, and the lane program starts this file with
OMP_NUM_THREADS=2); a different torch build can still move the last bits of a reduction, and a
failure then names the environment beside every field that differs.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
GOLDEN = HERE / "golden"
for _p in (str(HERE), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_templates as rt  # noqa: E402

PACKAGE = "b12_golden_orders"
HISTORY = GOLDEN / "orders-R0.history.jsonl"
LOGITS = GOLDEN / "orders-valid-logit-range.json"
SMOKE = GOLDEN / "orders-smoke.json"
#: the templates the orders fit reads (the spliced builder, blocks and losses come with --splice)
TRAINER_SET = ("domain.py", "network.py", "readout.py", "metrics.py", "frames.py", "data.py",
               "loop.py", "optim.py", "record.py", "engines.py")
#: the valid metrics E18 cannot move: the loss and the rank metrics (σ is monotone)
ALWAYS_COMPARED = ("loss", "roc_auc", "pr_auc")
#: the record's fields compared whole
WHOLE_FIELDS = ("epoch", "lr", "train", "grad_norm", "weight_norm", "update_ratio", "nan",
                "early_stopping", "shuffle")
WALL_CLOCK = ("time_s",)


def _purge(prefix):
    for name in [m for m in sys.modules if m == prefix or m.startswith(prefix + ".")]:
        del sys.modules[name]


def _sha256(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _golden():
    history = [json.loads(line) for line in HISTORY.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return history, json.loads(LOGITS.read_text(encoding="utf-8")), \
        json.loads(SMOKE.read_text(encoding="utf-8"))


def _environment() -> str:
    """This run's environment, as the failure message names it (the fit's thread count is the
    capture's, see `fitted`)."""
    import numpy
    _, logits, _ = _golden()
    threads = logits["capture"]["environment"]["torch_threads"]
    return f"torch {torch.__version__}, fitted on {threads} threads, numpy {numpy.__version__}"


class _Collect:
    """The callbacks the capture used: the records, as /vega-train writes them."""

    def __init__(self):
        self.records, self.first = [], []

    def on_epoch(self, record):
        self.records.append(record.as_dict())

    def on_first_step(self, record):
        self.first.append(dict(record))


@pytest.fixture(scope="module")
def orders(tmp_path_factory):
    """The fixture, its parts checked against the capture's, and the spec with its overrides."""
    work = tmp_path_factory.mktemp("b12-orders")
    shutil.copytree(HERE / "fixtures" / "dot-claude", work / ".claude")
    r = subprocess.run([sys.executable, str(HERE / "make_fixture.py"), str(work)],
                       capture_output=True, text=True, cwd=str(work))
    assert r.returncode == 0, f"make_fixture.py failed:\n{r.stdout}\n{r.stderr}"
    spec = json.loads((work / "netspec.json").read_text(encoding="utf-8"))
    _, logits, _ = _golden()
    capture = logits["capture"]
    for part, digest in capture["parts"].items():
        path = Path(spec["data_contract"]["parts"][part])
        path = path if path.is_absolute() else work / path
        assert _sha256(path) == digest, (
            f"the fixture's {part} part is not the one the golden was captured on ({digest}); "
            "make_fixture.py or the shaper changed what it writes, so no history can be "
            "bit-identical — recapture with capture_golden.py or restore the fixture")
    for key, value in capture["spec_overrides"].items():
        section, field = key.split(".", 1)
        spec[section][field] = value
    pipeline = json.loads(Path(spec["input"]["pipeline"]).read_text(encoding="utf-8"))
    return spec, pipeline, work


@pytest.fixture(scope="module")
def package(orders, tmp_path_factory):
    """The v2.3 trainer set rendered for the orders spec, imported."""
    spec, pipeline, _ = orders
    pipeline = json.loads(json.dumps(pipeline))
    pipeline["system"]["package"] = PACKAGE
    out = tmp_path_factory.mktemp("b12-render")
    rec = rt.render(TEMPLATES, out, pipeline, spec, only=list(TRAINER_SET), splice=True,
                    shapes=("row",))
    assert {"training/loop.py", "training/losses.py", "training/engines.py",
            "model/builder.py"} <= set(rec["files"]), rec["files"]
    sys.path.insert(0, str(out))
    importlib.invalidate_caches()
    try:
        yield {name: importlib.import_module(f"{PACKAGE}.{mod}") for name, mod in
               (("domain", "domain"), ("network", "model.network"), ("data", "ingestion.data"),
                ("loop", "training.loop"))}
    finally:
        sys.path.remove(str(out))
        _purge(PACKAGE)


@pytest.fixture(scope="module")
def fitted(orders, package):
    """The fit, on the capture's intra-op thread count: a CPU float reduction is split by the
    thread count, so the same kernels on another count round differently in the last bits (the
    lane program runs this file with OMP_NUM_THREADS=2; the capture ran on 20). The count is
    the process's, restored afterwards, and the spec's `backend.hints.cpu_threads` (a fit
    of a network this small otherwise runs on one thread, final rehearsal round 2)."""
    spec, _, work = orders
    _, logits, _ = _golden()
    threads = int(logits["capture"]["environment"]["torch_threads"])
    spec = json.loads(json.dumps(spec))
    spec.setdefault("backend", {}).setdefault("hints", {})["cpu_threads"] = threads
    ns = package["domain"].NetSpec.from_dict(spec)
    before = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        model = package["network"].Network.build(ns, run_dir=str(work))
        collect = _Collect()
        result = package["loop"].Trainer().fit(model, ns, package["data"].ShapedParts(),
                                               callbacks=collect)
    finally:
        torch.set_num_threads(before)
    return collect, result, model, ns


def _differences(got, want, where):
    """Every leaf of `want` that `got` does not hold bit for bit, as 'path: got != want'."""
    if isinstance(want, dict) and isinstance(got, dict):
        out = [f"{where}: keys {sorted(got)} != {sorted(want)}"] if set(got) != set(want) else []
        for key in sorted(set(got) & set(want)):
            out += _differences(got[key], want[key], f"{where}.{key}")
        return out
    numbers = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (got, want))
    if type(got) is not type(want) and not numbers:
        return [f"{where}: {got!r} ({type(got).__name__}) != {want!r} ({type(want).__name__})"]
    return [] if got == want else [f"{where}: {got!r} != {want!r}"]


def test_the_golden_names_its_e18_epochs():
    """`expected_to_differ_e18` lists exactly the epochs whose valid logits all lay in [0, 1]."""
    _, logits, _ = _golden()
    inside = [e["epoch"] for e in logits["epochs"] if not e["outside_unit_interval"]]
    assert logits["expected_to_differ_e18"] == inside
    for e in logits["epochs"]:
        assert e["outside_unit_interval"] == (e["min"] < 0.0 or e["max"] > 1.0), e


def test_the_history_is_bit_identical(fitted):
    collect, _, _, _ = fitted
    history, logits, _ = _golden()
    e18 = set(logits["expected_to_differ_e18"])
    assert len(collect.records) == len(history), (len(collect.records), len(history))
    diffs = []
    for got, want in zip(collect.records, history):
        got = {k: v for k, v in got.items() if k not in WALL_CLOCK}
        epoch = want["epoch"]
        assert set(got) == set(want), (sorted(got), sorted(want))
        for field in WHOLE_FIELDS:
            diffs += _differences(got[field], want[field], f"epoch {epoch} {field}")
        compared = (ALWAYS_COMPARED if epoch in e18 else tuple(want["valid"]))
        valid = {k: got["valid"].get(k) for k in compared}
        diffs += _differences(valid, {k: want["valid"].get(k) for k in compared},
                              f"epoch {epoch} valid")
    assert not diffs, (f"{len(diffs)} field(s) differ from {HISTORY.name} ({_environment()}; "
                       f"captured on {logits['capture']['environment']}):\n" + "\n".join(diffs))


def test_the_e18_epochs_are_listed_not_compared(fitted):
    """An epoch named in `expected_to_differ_e18` still carries every valid metric (it is the
    comparison that is waived, not the number)."""
    collect, _, _, _ = fitted
    history, logits, _ = _golden()
    for epoch in logits["expected_to_differ_e18"]:
        got, want = collect.records[epoch - 1]["valid"], history[epoch - 1]["valid"]
        assert set(got) == set(want), (epoch, sorted(got), sorted(want))


def test_the_fit_result_is_the_captured_one(fitted):
    _, result, _, _ = fitted
    _, logits, _ = _golden()
    got = {"epochs_run": int(result.epochs_run), "best_epoch": result.best_epoch,
           "best_value": result.best_value, "stopped_by": str(result.stopped_by.value)}
    assert got == logits["fit"]
    assert result.checkpoint is None           # no RunContext.store: nothing is written
    assert result.best_valid is not None and \
        result.best_valid.as_dict()["pr_auc"] == logits["fit"]["best_value"]


def test_the_first_step_record_is_written_once(fitted):
    collect, _, model, _ = fitted
    assert len(collect.first) == 1
    first = collect.first[0]
    assert first["block"] == model.ids[-1]
    assert set(first) == {"block", "grad_norm_head", "delta_norm_head", "ratio_head", "loss",
                          "lr"}
    assert first["grad_norm_head"] > 0 and first["delta_norm_head"] > 0 and first["ratio_head"] > 0


def test_params_and_blocks_equal_the_smoke_golden(fitted):
    _, _, model, ns = fitted
    _, _, smoke = _golden()
    assert model.param_count() == (smoke["params_total"], smoke["params_trainable"])
    feats = ns.to_dict()["data_contract"]["features"]
    n_num = sum(1 for f in feats if f.get("role") != "nominal")
    cards = [int(f["cardinality"]) for f in feats if f.get("role") == "nominal"]
    g = torch.Generator().manual_seed(5)
    batch = {"x_num": torch.randn(smoke["batch_size"], n_num, generator=g)}
    if cards:
        batch["x_cat"] = torch.stack([torch.randint(0, c, (smoke["batch_size"],), generator=g)
                                      for c in cards], -1)
    assert model.cpu().block_shapes(batch) == smoke["per_block"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
