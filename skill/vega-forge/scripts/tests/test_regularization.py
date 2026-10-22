#!/usr/bin/env python3
"""
test_regularization.py — rule R2: one regularization declaration, live paths derived from it
(netspec-contract §3; PLAN-AMENDMENTS A46 track F).

    python3 -m pytest test_regularization.py [-q]

What is proved:

  derivation  `vega_regularization`: the dropout sites (a `dropout` block's p, a transformer's
              and a recurrent block's of two or more layers `dropout`, composites' `inner`
              included, a one-layer recurrent block reading none), what `where` covers (a
              composite covers its inner; no `where` covers every block), `read_paths` (each
              declaration read, every live path no declaration covers), `shadowed_paths`,
              `resolve` and the `spec_snapshot.json` record the emitter writes;
  rule R2     a live copy that disagrees with its declaration, two declarations of one kind
              that disagree, two dropout declarations over one block that disagree, an unknown
              kind and a value out of range are ERROR; the same copies with equal values, a
              declaration that reaches nothing and an uncovered `p`-less dropout block WARN;
              a single declaration is clean;
  the twin    the emitted `domain.Regularization` writes the same rates into the same blocks
              as `vega_regularization.resolve`, on every spec here;
  the effect  on the orders fixture, the emitted code trains with the declared values — the
              optimiser's weight decay, every dropout module's rate — and a rung that moves
              `regularization[kind=weight_decay]/value` or `regularization[kind=dropout]/value`
              trains different numbers from R0, while R0 twice trains the same numbers (the
              symptom R1 found: R6 trained R0's numbers to the bit);
  the golden  the fixture's R0 resolves to the values the golden was captured with
              (weight decay 1e-4, B3's dropout 0.2): its form changed — the live copies were
              dropped, A46 — and what it trains did not, so the golden stays bit-identical and
              is not re-captured.
"""
from __future__ import annotations

import copy
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
for _p in (str(HERE), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_templates as rt  # noqa: E402
import rules_regularization as RR  # noqa: E402
import vega_regularization as R  # noqa: E402
from vega_core import Level  # noqa: E402

PACKAGE = "r2_regularization"
TRAINER_SET = ("domain.py", "network.py", "readout.py", "metrics.py", "frames.py", "data.py",
               "loop.py", "optim.py", "record.py", "engines.py")


def _b(bid, kind, **params):
    block = {"type": kind, "params": params}
    if bid is not None:
        block["id"] = bid
    return block


def _reg(kind, value, where=None):
    row = {"kind": kind, "value": value}
    if where is not None:
        row["where"] = where
    return row


def _spec(blocks, regularization, optimizer_params=None, loss=None, training=None):
    return {"blocks": blocks, "regularization": regularization,
            "optimizer": {"name": "adamw", "params": dict(optimizer_params or {"lr": 1e-3})},
            "loss": loss or {"name": "bce_with_logits", "params": {}},
            "training": training or {"batch_size": 8}}


NESTED = [_b("B1", "dense", units=8),
          _b("B2", "residual", repeat=2, inner=[_b(None, "dense", units=8),
                                                 _b(None, "dropout")]),
          _b("B3", "dropout"),
          _b("B4", "gru", hidden=8, num_layers=2),
          _b("B5", "gru", hidden=8),
          _b("B6", "transformer_encoder", d=8, heads=2),
          _b("B7", "head", out=1)]


# --------------------------------------------------------------------------- the derivation

def test_the_dropout_sites_are_every_block_that_reads_a_rate():
    paths = [s.path for s in R.dropout_sites(_spec(NESTED, []))]
    assert paths == ["blocks[id=B2]/params/inner/1/params/p", "blocks[id=B3]/params/p",
                     "blocks[id=B4]/params/dropout", "blocks[id=B6]/params/dropout"]


def test_where_covers_a_composites_inner_and_no_where_covers_everything():
    spec = _spec(NESTED, [_reg("dropout", 0.3, ["B2"])])
    assert R.live_values(spec) == {"blocks[id=B2]/params/inner/1/params/p": 0.3}
    spec = _spec(NESTED, [_reg("dropout", 0.3)])
    assert set(R.live_values(spec)) == {s.path for s in R.dropout_sites(spec)}


def test_read_paths_name_what_the_code_reads_and_never_a_shadowed_copy():
    spec = _spec(NESTED, [_reg("weight_decay", 1e-4), _reg("dropout", 0.3, ["B2", "B4"]),
                          _reg("grad_clip", 1.0)])
    read, shadowed = R.read_paths(spec), R.shadowed_paths(spec)
    assert read == sorted(["regularization[kind=weight_decay]/value", "regularization/0/value",
                           "regularization[kind=dropout]/value", "regularization/1/value",
                           "regularization[kind=grad_clip]/value", "regularization/2/value",
                           "blocks[id=B3]/params/p", "blocks[id=B6]/params/dropout"])
    assert shadowed == sorted(["optimizer/params/weight_decay", "training/grad_clip",
                               "blocks[id=B2]/params/inner/1/params/p",
                               "blocks[id=B4]/params/dropout"])
    assert not set(read) & set(shadowed)
    # nothing declared: the live paths are what the code reads (label smoothing only under a
    # loss that takes it)
    bare = _spec([_b("B1", "dropout", p=0.1)], [])
    assert R.read_paths(bare) == ["blocks[id=B1]/params/p", "optimizer/params/weight_decay",
                                  "training/grad_clip"]
    ce = _spec([], [], loss={"name": "cross_entropy", "params": {}})
    assert "loss/params/label_smoothing" in R.read_paths(ce)


def test_resolve_writes_the_declared_values_into_the_live_paths():
    spec = _spec(NESTED, [_reg("weight_decay", 0.01), _reg("dropout", 0.25, ["B2"]),
                          _reg("label_smoothing", 0.1)],
                 loss={"name": "cross_entropy", "params": {}})
    out = R.resolve(spec)
    assert out["optimizer"]["params"]["weight_decay"] == 0.01
    assert out["loss"]["params"]["label_smoothing"] == 0.1
    assert out["blocks"][1]["params"]["inner"][1]["params"]["p"] == 0.25
    assert "p" not in out["blocks"][2]["params"]                 # B3: not covered
    assert spec["optimizer"]["params"] == {"lr": 1e-3}           # the input is not touched


# --------------------------------------------------------------------------- rule R2

def _markers(spec):
    return RR.regularization(SimpleNamespace(spec=spec))


def _levels(spec):
    out = {Level.ERROR: [], Level.WARN: [], Level.OK: []}
    for m in _markers(spec):
        out[m.level].append(m.message)
    return out


def test_a_single_declaration_is_clean():
    spec = _spec(NESTED, [_reg("weight_decay", 1e-4), _reg("dropout", 0.2, ["B2", "B3"])])
    got = _levels(spec)
    assert not got[Level.ERROR] and not got[Level.WARN], got
    assert any("the emitted code reads" in m for m in got[Level.OK])


@pytest.mark.parametrize("spec, needle", [
    (_spec([], [_reg("weight_decay", 1e-3)], optimizer_params={"lr": 1e-3, "weight_decay": 1e-4}),
     "optimizer/params/weight_decay is 0.0001 but regularization[0] declares weight_decay 0.001"),
    (_spec([_b("B1", "dropout", p=0.2)], [_reg("dropout", 0.4, ["B1"])]),
     "blocks[id=B1]/params/p is 0.2 but regularization[0] declares dropout 0.4"),
    (_spec([], [_reg("grad_clip", 1.0)], training={"batch_size": 8, "grad_clip": 5.0}),
     "training/grad_clip is 5.0"),
    (_spec([], [_reg("label_smoothing", 0.1)],
           loss={"name": "cross_entropy", "params": {"label_smoothing": 0.05}}),
     "loss/params/label_smoothing is 0.05"),
    (_spec([], [_reg("weight_decay", 1e-4), _reg("weight_decay", 1e-3)]),
     "conflicting declarations: weight_decay"),
    (_spec([_b("B1", "dropout")], [_reg("dropout", 0.1, ["B1"]), _reg("dropout", 0.3)]),
     "conflicting declarations: blocks[id=B1]/params/p"),
    (_spec([], [_reg("mixup", 0.2)]), "no emitted code reads this kind"),
    (_spec([_b("B1", "dropout")], [_reg("dropout", 1.0, ["B1"])]), "out of range"),
    (_spec([], [_reg("weight_decay", None)]), "out of range"),
])
def test_r2_refuses(spec, needle):
    errors = _levels(spec)[Level.ERROR]
    assert any(needle in m for m in errors), errors


@pytest.mark.parametrize("spec, needle", [
    (_spec([], [_reg("weight_decay", 1e-4)], optimizer_params={"weight_decay": 1e-4}),
     "repeats the declared weight_decay"),
    (_spec([_b("B1", "dropout", p=0.2)], [_reg("dropout", 0.2, ["B1"])]),
     "repeats the declared dropout"),
    (_spec([_b("B1", "dense", units=4)], [_reg("dropout", 0.2, ["B9"])]),
     "where names 'B9', which is no block"),
    (_spec([_b("B1", "gru", hidden=4)], [_reg("dropout", 0.2, ["B1"])]),
     "covers no block that reads a rate"),
    (_spec([], [_reg("label_smoothing", 0.1)]), "takes none"),
    (_spec([_b("B1", "dropout")], []), "a dropout block with no p"),
    (_spec([], [_reg("grad_clip", 1.0), _reg("grad_clip", 1.0)]), "declared twice"),
])
def test_r2_warns(spec, needle):
    got = _levels(spec)
    assert not got[Level.ERROR], got
    assert any(needle in m for m in got[Level.WARN]), got


def test_r2_is_registered_in_the_compiler():
    import compile_spec as C
    assert [r.id for r in C.RULES].count("R2") == 1


# --------------------------------------------------------------------------- the fixture

@pytest.fixture(scope="module")
def orders(tmp_path_factory):
    work = tmp_path_factory.mktemp("r2-orders")
    shutil.copytree(HERE / "fixtures" / "dot-claude", work / ".claude")
    r = subprocess.run([sys.executable, str(HERE / "make_fixture.py"), str(work)],
                       capture_output=True, text=True, cwd=str(work))
    assert r.returncode == 0, f"make_fixture.py failed:\n{r.stdout}\n{r.stderr}"
    spec = json.loads((work / "netspec.json").read_text(encoding="utf-8"))
    spec["training"]["max_epochs"], spec["training"]["device"] = 3, "cpu"
    pipeline = json.loads(Path(spec["input"]["pipeline"]).read_text(encoding="utf-8"))
    return spec, pipeline, work


@pytest.fixture(scope="module")
def package(orders, tmp_path_factory):
    spec, pipeline, _ = orders
    pipeline = json.loads(json.dumps(pipeline))
    pipeline["system"]["package"] = PACKAGE
    out = tmp_path_factory.mktemp("r2-render")
    rt.render(TEMPLATES, out, pipeline, spec, only=list(TRAINER_SET), splice=True,
              shapes=("row",))
    sys.path.insert(0, str(out))
    importlib.invalidate_caches()
    try:
        yield {name: importlib.import_module(f"{PACKAGE}.{mod}") for name, mod in
               (("domain", "domain"), ("network", "model.network"), ("data", "ingestion.data"),
                ("loop", "training.loop"), ("optim", "training.optim"))}
    finally:
        sys.path.remove(str(out))
        for name in [m for m in sys.modules if m == PACKAGE or m.startswith(PACKAGE + ".")]:
            del sys.modules[name]


def test_the_fixture_declares_once_and_resolves_to_the_golden_values(orders):
    """R0's form changed (A46: the live copies dropped), what it trains did not: the values
    the golden B01 was captured with are the declaration's, so the golden is not re-captured."""
    spec, _, _ = orders
    assert "weight_decay" not in spec["optimizer"]["params"]
    resolved = R.resolve(spec)
    assert resolved["optimizer"]["params"]["weight_decay"] == 0.0001
    b3 = next(b for b in resolved["blocks"] if b["id"] == "B3")
    assert b3["params"]["inner"][3] == {"type": "dropout", "params": {"p": 0.2}}
    got = _levels(spec)
    assert not got[Level.ERROR] and not got[Level.WARN], got


def _edit(spec, kind, value):
    out = copy.deepcopy(spec)
    next(r for r in out["regularization"] if r["kind"] == kind)["value"] = value
    return out


TWIN_SPECS = [
    [_reg("dropout", 0.3, ["B2"])],
    [_reg("dropout", 0.3)],
    [_reg("dropout", 0.15, ["B3", "B4"]), _reg("dropout", 0.35, ["B6"])],
    [_reg("weight_decay", 1e-3)],
]


@pytest.mark.parametrize("regularization", TWIN_SPECS)
def test_the_emitted_twin_writes_the_same_rates(orders, package, regularization):
    base, _, _ = orders
    spec = dict(copy.deepcopy(base), blocks=copy.deepcopy(NESTED), regularization=regularization)
    ns = package["domain"].NetSpec.from_dict(spec)
    emitted = package["domain"].Regularization.of(ns).blocks(spec["blocks"])
    assert emitted == R.resolve(spec)["blocks"]


def _dropout_rates(model):
    from torch import nn
    return sorted({round(m.p, 6) for m in model.modules() if isinstance(m, nn.Dropout)})


def test_the_emitted_code_builds_with_the_declared_values(orders, package):
    spec, _, work = orders
    ns = package["domain"].NetSpec.from_dict
    for wd, p in ((0.0001, 0.2), (0.0, 0.2), (0.0001, 0.4)):
        moved = _edit(_edit(spec, "weight_decay", wd), "dropout", p)
        model = package["network"].Network.build(ns(moved), run_dir=str(work))
        assert _dropout_rates(model) == [p]
        opt = package["optim"].build_optimizer(model, ns(moved))
        assert opt.param_groups[0]["weight_decay"] == wd
    assert package["optim"].grad_clip(ns(_edit(spec, "grad_clip", None))) is None
    assert package["optim"].grad_clip(ns(spec)) == 1.0


def test_the_emitted_code_reads_the_declared_label_smoothing(orders, package):
    spec, _, _ = orders
    losses = importlib.import_module(f"{PACKAGE}.training.losses")
    ce = dict(copy.deepcopy(spec), loss={"name": "cross_entropy", "params": {}})
    ce["regularization"] = ce["regularization"] + [_reg("label_smoothing", 0.1)]
    ns = package["domain"].NetSpec.from_dict
    assert losses.build_loss(ns(ce)).label_smoothing == 0.1
    assert losses.build_loss(ns(_edit(ce, "label_smoothing", 0.0))).label_smoothing == 0.0


@pytest.mark.parametrize("where, edit", [
    ("optimizer.params.weight_decay", lambda s: s["optimizer"]["params"].update(weight_decay=0.5)),
    ("params.p", lambda s: s["blocks"][3]["params"]["inner"][3]["params"].update(p=0.5)),
    ("loss.params.label_smoothing", lambda s: s.update(
        loss={"name": "cross_entropy", "params": {"label_smoothing": 0.5}},
        regularization=s["regularization"] + [_reg("label_smoothing", 0.1)])),
])
def test_the_emitted_code_refuses_a_copy_that_disagrees(orders, package, where, edit):
    """The compiler refuses it first (R2); a spec that skipped the compiler is refused when the
    network, the optimiser or the loss is built, never trained silently on either value."""
    spec, _, work = orders
    bad = copy.deepcopy(spec)
    edit(bad)
    ns = package["domain"].NetSpec.from_dict(bad)
    losses = importlib.import_module(f"{PACKAGE}.training.losses")
    with pytest.raises(ValueError, match=where.replace(".", r"\.")):
        model = package["network"].Network.build(ns, run_dir=str(work))
        package["optim"].build_optimizer(model, ns)
        losses.build_loss(ns)
    assert any(where.split(".")[-1] in m and "declares" in m
               for m in _levels(bad)[Level.ERROR]), _levels(bad)


class _Collect:
    def __init__(self):
        self.records = []

    def on_epoch(self, record):
        self.records.append(record.as_dict())

    def on_first_step(self, record):
        pass


def _fit(package, spec, work):
    import torch
    ns = package["domain"].NetSpec.from_dict(spec)
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = package["network"].Network.build(ns, run_dir=str(work))
        collect = _Collect()
        package["loop"].Trainer().fit(model, ns, package["data"].ShapedParts(), callbacks=collect)
    finally:
        torch.set_num_threads(before)
    return [{k: v for k, v in r.items() if k != "time_s"} for r in collect.records]


@pytest.fixture(scope="module")
def r0(orders, package):
    spec, _, work = orders
    return _fit(package, spec, work)


def test_r0_twice_trains_the_same_numbers(orders, package, r0):
    spec, _, work = orders
    assert _fit(package, spec, work) == r0


@pytest.mark.parametrize("kind, value", [("weight_decay", 0.0), ("weight_decay", 0.1),
                                         ("dropout", 0.4), ("dropout", 0.0)])
def test_a_rung_on_the_declaration_trains_different_numbers(orders, package, r0, kind, value):
    spec, _, work = orders
    rung = _fit(package, _edit(spec, kind, value), work)
    assert len(rung) == len(r0)
    assert [e["weight_norm"] for e in rung] != [e["weight_norm"] for e in r0]
    assert [e["train"]["loss"] for e in rung] != [e["train"]["loss"] for e in r0]


# --------------------------------------------------------------------------- the snapshot

def test_the_emitter_writes_read_paths_into_spec_snapshot(orders, tmp_path):
    spec, pipeline, work = orders
    sys.path.insert(0, str(SCRIPTS))
    import backends
    backend = backends.load("pytorch")
    out = tmp_path / "run"
    backend.emit(spec, pipeline, out, run_dir=work)
    snap = json.loads((out / "spec_snapshot.json").read_text(encoding="utf-8"))
    assert snap["read_paths"] == R.read_paths(spec)
    assert "regularization[kind=weight_decay]/value" in snap["read_paths"]
    assert "regularization[kind=dropout]/value" in snap["read_paths"]
    assert "optimizer/params/weight_decay" not in snap["read_paths"]
    reg = snap["_resolved"]["regularization"]
    assert reg["shadowed_paths"] == ["blocks[id=B3]/params/inner/3/params/p",
                                     "optimizer/params/weight_decay", "training/grad_clip"]
    assert reg["live_values"]["optimizer/params/weight_decay"] == 0.0001
    assert reg["live_values"]["blocks[id=B3]/params/inner/3/params/p"] == 0.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
