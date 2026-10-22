#!/usr/bin/env python3
"""
test_losses.py — the loss library (`backends/pytorch/losses.py`) as the emitter splices it into
the Trainer's support file `<training>/losses.py` (BUILD-PLAN B07; D1 §3.3–§3.4, §7.4, §8;
R-B4, R-B6).

    python3 test_losses.py            (or: python3 -m pytest test_losses.py -q)

Every loss is exercised through the emitted file: the `prelude` and `losses` segments spliced by
`emit.spliced_files` into a package whose `domain.py` is the rendered template, so the domain
names the segment reads are bound the way a real emission binds them. What it proves:

- the splice: `training/losses.py` opens with the prelude's imports, imports exactly
  `NetSpec, ClassWeights, Task, TaskKind` from `..domain` and nothing else of the package, is a
  `kind: support` entry used by the Trainer, and has no C11–C13 finding (nor has the source);
- per task, the shapes of D1 §3.4 against a hand computation: binary, multiclass, multilabel
  with a K-vector `pos_weight`, focal (sigmoid per label; softmax per class, which build_loss
  resolves from the effective task), regression, forecast, quantile `(B,Q)` and `(B,H,Q)`
  horizon-major (column h·Q + q), ordinal, the `recon_dim` layout `[n_num | card_1 | …]` whose
  loss is the mean of D1's per-row reconstruction score, and every width mismatch refused;
- vae: a training pass reads its own forward's KL through `model.kl()`; an eval pass reads
  `Reparameterize.divergence` of this batch, not the stale attribute, and leaves no hook and no
  written attribute behind;
- svdd: the centre is taken from a stub model's buffers (`objective_center()`,
  `svdd_center_set`) — set from the first training batch (mean, |c| < 0.1 pushed to ±0.1 by
  its sign), never moved after, never held by the loss, never written by an eval pass, and the
  loss is identical after a `state_dict` round trip;
- no loss reads a tensor's value on the host during a training call (the step stays
  capturable in a CUDA graph);
- `cross_entropy` is byte-identical to c849f8b on the weighted + smoothed CIC R7 case (value
  and gradient, float32 and float64, through `build_loss`), and so are the unweighted smoothed
  case and the orders `bce_with_logits` with `pos_weight` 4.9 — and their `formula()` strings
  equal the B01 golden captures and R7's recorded mechanics;
- `formula()` for every loss (quantile names H and Q; svdd says the centre is stored with the
  weights); `from_params` refuses a parameter a loss does not take, drops a `None`, and every
  `__init__` names at most five parameters; `build_loss` takes a NetSpec only and resolves the
  reconstruction layout (the FeatureContract's numeric columns, the one definition the builder
  and the data adapter read), a forecast's horizon and the focal link from the contract, and
  refuses a spec `multiclass` or `horizon` the contract contradicts;
- the ladder: a key vega-train's spec-alternative rung carries from the loss the block was
  weighed against (the orders fixture's bce `pos_weight` under its focal alternative) is dropped
  with a warning, as c849f8b ignored it; any other key a loss does not take is still refused.
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import types
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
REPO = SCRIPTS.parents[1]
LOSSES_PY = BACKEND / "losses.py"
GOLDEN = HERE / "golden"
BASELINE_COMMIT = "c849f8b"
PACKAGE = "losses_pkg"
for _p in (str(SCRIPTS), str(HERE), str(BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conformance  # noqa: E402
import render_templates as R  # noqa: E402

# CIC-IDS2017 R7's measured loss (records/architecture.json netspec.loss.params, R7 bundle): the
# inverse-frequency class weights clipped at 10, and label smoothing 0.05
CIC_WEIGHTS = [0.080209, 0.972861, 1.313558, 1.851579] + [10.0] * 11
CIC_SMOOTHING = 0.05
CIC_MECHANICS = ("L = Σ_i [ (1-0.05)·w[y_i]·(-log p_i,y_i) + (0.05/K)·Σ_c w[c]·(-log p_i,c) ]"
                 " / Σ_i w[y_i],  K = 15, class-weighted")
CIC_BUNDLE = "bundle/cicids2017-label-R7"


# ------------------------------------------------------------------ the emitted module

def _emit_package(root: Path):
    """domain.py rendered from the template (render_templates, partial) and the Trainer's
    spliced `losses.py`, exactly as `emit.spliced_files` writes it."""
    emit = R.load_emit()
    pipeline = R.reference_pipeline()
    pipeline["system"]["package"] = PACKAGE
    spec = R.reference_spec("row")
    R.render(BACKEND / "templates", root, pipeline=pipeline, spec=spec, only=["domain.py"])
    plan = emit.plan(spec, pipeline)
    src = emit.sources_for("v2.3")
    ctx = emit._Ctx(spec, plan, emit.sample_shape(spec), emit.block_plan(spec, src.blocks))
    sf = next(s for s in emit.SUPPORT_FILES["Trainer"] if s.name == "losses.py")
    module = plan["module_names"]["Trainer"]
    rel = f"{module}/losses.py"
    pkg = root / PACKAGE
    domain_text = (pkg / "domain.py").read_text(encoding="utf-8")
    text, segments = emit.spliced_files(ctx, src, [(rel, sf)], domain_text, "sha256:test")[rel]
    (pkg / module).mkdir(exist_ok=True)
    (pkg / module / "__init__.py").write_text("", encoding="utf-8")
    (pkg / rel).write_text(text, encoding="utf-8")
    return {"rel": rel, "text": text, "segments": segments,
            "row": emit._file_row(sf, rel, module, plan)}


_WORK = tempfile.TemporaryDirectory(prefix="losses-")
EMITTED = _emit_package(Path(_WORK.name))
sys.path.insert(0, _WORK.name)
D = importlib.import_module(f"{PACKAGE}.domain")
L = importlib.import_module(f"{PACKAGE}.{EMITTED['rel'][:-3].replace('/', '.')}")


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _randn(*shape, seed=0, dtype=torch.float32):
    return torch.randn(*shape, generator=_gen(seed), dtype=dtype)


def _spec(shape="row", task=None, loss=None, features=None, sample=None):
    """A reference netspec (render_templates) with the fields a case changes, as a NetSpec."""
    d = R.reference_spec(shape)
    if task is not None:
        d["task"] = task
    if loss is not None:
        d["loss"] = loss
    if features is not None:
        d["data_contract"]["features"] = features
        d["data_contract"]["input_dim"] = len([f for f in features if f.get("role") != "key"])
    if sample is not None:
        d["data_contract"]["sample"].update(sample)
    return D.NetSpec.from_dict(d)


# ------------------------------------------------------------------ the splice

def test_the_splice_binds_the_domain_names_and_nothing_else_of_the_package():
    tree = ast.parse(EMITTED["text"])
    relative = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.level]
    assert [(n.module, n.level) for n in relative] == [("domain", 2)]
    assert sorted(a.name for a in relative[0].names) == ["ClassWeights", "NetSpec",
                                                        "Regularization", "SampleSpec", "Task",
                                                        "TaskKind"]
    absolute = {a.name for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                and not getattr(n, "level", 0) for a in n.names}
    assert {"torch", "functional", "Mapping", "warnings"} <= absolute
    assert EMITTED["segments"] == ("prelude", "losses")
    assert EMITTED["row"]["kind"] == "support" and EMITTED["row"]["used_by"] == ["Trainer"]


def test_no_size_finding_on_the_emitted_file_or_the_source():
    for rel, text in ((EMITTED["rel"], EMITTED["text"]),
                      ("losses.py", LOSSES_PY.read_text(encoding="utf-8"))):
        findings = conformance.size_findings(ast.parse(text), rel)
        assert findings == [], [f.message() for f in findings]


def test_the_source_stays_a_runnable_module():
    """The splice source imports alone (the domain names are only read at call time)."""
    spec = importlib.util.spec_from_file_location("losses_source", str(LOSSES_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = _randn(4, 3, seed=1)
    assert torch.equal(mod.MSE()(out, {"y": out.clone()}), torch.zeros(()))
    assert set(mod.LOSSES) == set(L.LOSSES) == set(json.loads(
        (BACKEND / "coverage.json").read_text(encoding="utf-8"))["losses"])


# ------------------------------------------------------------------ per task: classification

def test_binary_bce_shapes_and_value():
    out, y = _randn(6, 1, seed=2), torch.tensor([0, 1, 1, 0, 1, 0])
    loss = L.BCEWithLogits(pos_weight=3.0)(out, {"y": y})
    z, t = out[:, 0], y.float()
    want = -(3.0 * t * F.logsigmoid(z) + (1 - t) * F.logsigmoid(-z)).mean()
    assert loss.shape == () and torch.allclose(loss, want, atol=1e-6)


def test_multilabel_bce_takes_a_k_vector_pos_weight():
    out = _randn(5, 3, seed=3)
    y = torch.tensor([[1, 0, 1], [0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32)
    w = [0.5, 2.0, 4.0]
    loss = L.BCEWithLogits(pos_weight=w)(out, {"y": y})
    wt = torch.tensor(w)
    want = -(wt * y * F.logsigmoid(out) + (1 - y) * F.logsigmoid(-out)).mean()
    assert torch.allclose(loss, want, atol=1e-6)
    assert isinstance(L.BCEWithLogits(pos_weight=w).pos_weight, D.ClassWeights)
    same = L.BCEWithLogits(pos_weight=D.ClassWeights.of(w))(out, {"y": y})
    assert torch.equal(loss, same)
    with pytest.raises(ValueError, match="3 pos_weight values for a head of 4 outputs"):
        L.BCEWithLogits(pos_weight=w)(_randn(5, 4, seed=4), {"y": torch.zeros(5, 4)})
    with pytest.raises(ValueError, match="not finite and positive"):
        L.BCEWithLogits(pos_weight=[1.0, -1.0])


def test_multiclass_cross_entropy_and_its_width_check():
    out, y = _randn(7, 4, seed=5), torch.tensor([0, 1, 2, 3, 3, 1, 0])
    assert torch.allclose(L.CrossEntropy()(out, {"y": y}), F.cross_entropy(out, y))
    with pytest.raises(ValueError, match="3 weight values for a head of 4 outputs"):
        L.CrossEntropy(weight=[1.0, 2.0, 3.0])(out, {"y": y})


def test_focal_sigmoid_per_label_and_softmax_per_class():
    out = _randn(6, 3, seed=6)
    y = (torch.rand(6, 3, generator=_gen(7)) > 0.5).float()
    p = torch.sigmoid(out)
    pt = p * y + (1 - p) * (1 - y)
    ce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
    alpha = torch.tensor([0.25, 0.5, 0.75])
    want = (ce * (1 - pt) ** 2 * (alpha * y + (1 - alpha) * (1 - y))).mean()
    got = L.Focal(gamma=2, alpha=[0.25, 0.5, 0.75])(out, {"y": y})
    assert torch.allclose(got, want, atol=1e-6)
    scalar = L.Focal(gamma=2, alpha=0.25)(out, {"y": y})
    assert torch.allclose(scalar, (ce * (1 - pt) ** 2 * (0.25 * y + 0.75 * (1 - y))).mean(),
                          atol=1e-6)
    # softmax: gamma 0 is the cross-entropy; alpha weighs each row by its class
    ids = torch.tensor([0, 2, 1, 2, 0, 1])
    assert torch.allclose(L.Focal(gamma=0, multiclass=True)(out, {"y": ids}),
                          F.cross_entropy(out, ids), atol=1e-6)
    logp = F.log_softmax(out, -1).gather(1, ids[:, None])[:, 0]
    w = torch.tensor([1.0, 2.0, 0.5])
    want = (-(1 - logp.exp()) ** 1.5 * logp * w[ids]).mean()
    got = L.Focal(gamma=1.5, alpha=[1.0, 2.0, 0.5], multiclass=True)(out, {"y": ids})
    assert torch.allclose(got, want, atol=1e-6)
    for bad in ({"alpha": 0.25, "multiclass": True}, {"alpha": 1.5}, {"alpha": [0.5, 2.0]}):
        with pytest.raises(ValueError, match="focal alpha"):
            L.Focal(**bad)


def test_ordinal_cumulative_link_reads_k_minus_1_thresholds():
    out, y = _randn(5, 3, seed=8), torch.tensor([0, 1, 2, 3, 2])
    target = torch.stack([(y > j).float() for j in range(3)], 1)
    want = F.binary_cross_entropy_with_logits(out, target)
    assert torch.equal(L.CumulativeLink()(out, {"y": y}), want)


# ------------------------------------------------------------------ per task: values

def test_regression_and_forecast_pointwise_losses():
    out, y = _randn(6, 1, seed=9), _randn(6, seed=10)
    assert torch.allclose(L.MSE()(out, {"y": y}), ((out[:, 0] - y) ** 2).mean())
    assert torch.allclose(L.MAE()(out, {"y": y}), (out[:, 0] - y).abs().mean())
    assert torch.allclose(L.Huber(delta=0.5)(out, {"y": y}),
                          F.huber_loss(out[:, 0], y, delta=0.5))
    h_out, h_y = _randn(6, 4, seed=11), _randn(6, 4, seed=12)       # forecast (B,H) vs (B,H)
    assert torch.allclose(L.MSE()(h_out, {"y": h_y}), ((h_out - h_y) ** 2).mean())
    assert torch.allclose(L.MAE()(h_out, {"y": h_y}), (h_out - h_y).abs().mean())


def _pinball(y, pred, q):
    d = y - pred
    return max(q * d, (q - 1) * d)


def test_quantile_regression_is_b_by_q():
    qs = [0.1, 0.5, 0.9]
    out, y = _randn(5, 3, seed=13, dtype=torch.float64), _randn(5, seed=14, dtype=torch.float64)
    want = sum(_pinball(float(y[i]), float(out[i, k]), q)
               for i in range(5) for k, q in enumerate(qs)) / 15
    assert abs(float(L.Quantile(quantiles=qs)(out, {"y": y})) - want) < 1e-12


def test_quantile_forecast_is_horizon_major():
    """(B, H·Q) is read as (B, H, Q): level q of horizon h is column h·Q + q (D2 Q6)."""
    qs, h = [0.1, 0.5, 0.9], 4
    out = _randn(3, h * len(qs), seed=15, dtype=torch.float64)
    y = _randn(3, h, seed=16, dtype=torch.float64)
    want = sum(_pinball(float(y[i, t]), float(out[i, t * 3 + k]), q)
               for i in range(3) for t in range(h) for k, q in enumerate(qs)) / (3 * h * 3)
    got = L.Quantile(quantiles=qs, horizon=h)(out, {"y": y})
    assert abs(float(got) - want) < 1e-12
    quantile_major = sum(_pinball(float(y[i, t]), float(out[i, k * h + t]), q)
                         for i in range(3) for t in range(h) for k, q in enumerate(qs)) / 36
    assert abs(quantile_major - want) > 1e-6, "the layout must matter for this case"
    with pytest.raises(ValueError, match=r"H·Q = 4·3 head outputs; the head has 4"):
        L.Quantile(quantiles=qs, horizon=h)(_randn(3, 4, seed=17), {"y": y.float()})
    with pytest.raises(ValueError, match=r"H·Q = 1·3 head outputs; the head has 12"):
        L.Quantile(quantiles=qs)(out, {"y": y})
    for bad in ({"quantiles": [0.5, 1.0]}, {"quantiles": []}, {"horizon": 0}):
        with pytest.raises(ValueError, match="quantile levels lie in"):
            L.Quantile(**bad)


def test_reconstruction_recon_dim_layout_and_its_row_score():
    """[n_num | card_1 | card_2]; the loss is the mean over rows of D1 §3.4's per-row score
    (numeric MSE + the mean nominal cross-entropy)."""
    n, cards, b = 3, [4, 2], 6
    x_num, out = _randn(b, n, seed=18), _randn(b, n + sum(cards), seed=19)
    x_cat = torch.stack([torch.randint(0, c, (b,), generator=_gen(20 + i))
                         for i, c in enumerate(cards)], 1)
    batch = {"x_num": x_num, "x_cat": x_cat}
    loss = L.Reconstruction(n_numeric=n, cat_cardinalities=cards)(out, batch)
    num = ((out[:, :n] - x_num) ** 2).mean(1)
    ce = torch.stack([F.cross_entropy(out[:, 3:7], x_cat[:, 0], reduction="none"),
                      F.cross_entropy(out[:, 7:9], x_cat[:, 1], reduction="none")], 1).mean(1)
    assert torch.allclose(loss, (num + ce).mean(), atol=1e-6)
    numeric_only = L.Reconstruction()(out[:, :n], {"x_num": x_num})
    assert torch.allclose(numeric_only, num.mean())
    # E13: a head `input_dim` wide (one column per nominal) cannot carry the logits
    with pytest.raises(ValueError, match=r"has 5 outputs; recon_dim = 3 numeric \+ 6"):
        L.Reconstruction(n_numeric=n, cat_cardinalities=cards)(out[:, :5], batch)
    with pytest.raises(ValueError, match="no cardinality"):
        L.Reconstruction(n_numeric=n, cat_cardinalities=[4, None])


# ------------------------------------------------------------------ vae: the KL term

import blocks as B  # noqa: E402


class _VaeNet(nn.Module):
    """The vae recipe in miniature: encoder -> reparameterize(z) -> decoder -> head, with the
    Network's `kl()` (the term of the last training forward)."""

    def __init__(self, n=4, z=2):
        super().__init__()
        torch.manual_seed(3)
        self.enc = nn.Sequential(nn.Linear(n, 8), nn.Dropout(0.3), nn.Linear(8, 2 * z))
        self.rep = B.Reparameterize(z)
        self.dec = nn.Linear(z, n)

    def forward(self, batch):
        return self.dec(self.rep(self.enc(batch["x_num"])))

    def kl(self):
        return self.rep.kl


def _attrs(m):
    return {k: v for k, v in vars(m).items() if not k.startswith("_")}


def test_vae_training_reads_its_forward_kl_and_eval_reads_the_divergence():
    net, loss_fn = _VaeNet(), L.VAE(beta=0.5, n_numeric=4)
    batch = {"x_num": _randn(16, 4, seed=21)}
    net.train()
    torch.manual_seed(5)
    out = net(batch)
    train = loss_fn(out, batch, net)
    assert torch.allclose(train, F.mse_loss(out, batch["x_num"]) + 0.5 * net.rep.kl)
    stale = net.rep.kl
    net.eval()
    other = {"x_num": _randn(16, 4, seed=22)}
    state = {k: v.clone() for k, v in net.state_dict().items()}
    attrs = {id(m): _attrs(m) for m in net.modules()}
    with torch.no_grad():
        out = net(other)
        got = loss_fn(out, other, net)
        want = F.mse_loss(out, other["x_num"]) + 0.5 * net.rep.divergence(net.enc(other["x_num"]))
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got, F.mse_loss(out, other["x_num"]) + 0.5 * stale)
    assert net.rep.kl is stale, "an eval pass must not write the block's attribute (R-B4)"
    assert not net.rep._forward_pre_hooks, "the capture hook must be removed"
    assert all(torch.equal(state[k], v) for k, v in net.state_dict().items())
    assert all(attrs[id(m)] == _attrs(m) for m in net.modules())
    with pytest.raises(ValueError, match="needs the KL term of a reparameterize block"):
        L.VAE()(out, other, nn.Linear(4, 4).eval())
    with pytest.raises(ValueError, match="needs the KL term"):
        L.VAE()(out, other, None)


# ------------------------------------------------------------------ svdd: the centre buffer

class _SvddNet(nn.Module):
    """The two buffers R-B6 names and `objective_center()`, as the Network registers them."""

    def __init__(self, z=4):
        super().__init__()
        self.register_buffer("svdd_center", torch.zeros(z))
        self.register_buffer("svdd_center_set", torch.zeros((), dtype=torch.bool))

    def objective_center(self):
        return self.svdd_center


def test_svdd_centre_is_the_models_buffer_set_once_from_the_first_training_batch():
    net, loss_fn = _SvddNet(), L.SVDD()
    first = torch.tensor([[0.02, -0.05, 1.0, 0.0], [0.04, -0.01, 3.0, 0.0]])
    loss = loss_fn(first, {}, net.train())
    c = torch.tensor([0.1, -0.1, 2.0, 0.1])       # |c| < 0.1 pushed to ±0.1 by its sign
    assert torch.equal(net.svdd_center, c) and bool(net.svdd_center_set)
    assert torch.allclose(loss, ((first - c) ** 2).sum(1).mean())
    later = _randn(8, 4, seed=23) * 5
    loss_fn(later, {}, net)
    assert torch.equal(net.svdd_center, c), "the centre is set once"
    assert not any(torch.is_tensor(v) for v in vars(loss_fn).values()) and not loss_fn._consts
    fresh = L.SVDD()(later, {}, net.eval())
    assert torch.allclose(fresh, ((later - c) ** 2).sum(1).mean()), "read from the buffer"
    reloaded = _SvddNet()
    reloaded.load_state_dict(net.state_dict())
    assert torch.equal(L.SVDD()(later, {}, reloaded.eval()), fresh)


def test_svdd_eval_never_writes_and_a_model_without_the_buffers_is_refused():
    net = _SvddNet().eval()
    out = _randn(6, 4, seed=24)
    loss = L.SVDD()(out, {}, net)
    assert not bool(net.svdd_center_set) and torch.equal(net.svdd_center, torch.zeros(4))
    c = out.mean(0)
    c = torch.where(c.abs() < 0.1, torch.full_like(c, 0.1).copysign(c), c)
    assert torch.allclose(loss, ((out - c) ** 2).sum(1).mean())
    for model in (None, nn.Linear(4, 4), _SvddNet(z=3)):
        with pytest.raises(ValueError, match="svdd centre is the network's buffers"):
            L.SVDD()(out, {}, model)
    with pytest.raises(ValueError, match=r"'center'; it takes none; the centre is the network's"):
        L.SVDD.from_params({"center": [0.0] * 4})


# ------------------------------------------------------------------ capturable steps

@pytest.fixture
def no_host_reads(monkeypatch):
    """A tensor's value read on the host (bool, item, float, int, tolist) raises: what a CUDA
    graph capture forbids, checked on CPU."""
    def refuse(*_a, **_k):
        raise AssertionError("a loss read a tensor's value on the host during the step")
    for name in ("__bool__", "item", "__float__", "__int__", "tolist"):
        monkeypatch.setattr(torch.Tensor, name, refuse)


def test_no_loss_reads_a_tensor_on_the_host_in_a_training_step(no_host_reads):
    cases = [
        (L.BCEWithLogits(pos_weight=4.9), _randn(5, 1, seed=30), {"y": torch.ones(5)}),
        (L.BCEWithLogits(pos_weight=[1.0, 2.0]), _randn(5, 2, seed=31), {"y": torch.ones(5, 2)}),
        (L.CrossEntropy(weight=CIC_WEIGHTS[:3], label_smoothing=0.05), _randn(5, 3, seed=32),
         {"y": torch.tensor([0, 1, 2, 1, 0])}),
        (L.Focal(alpha=[1.0, 2.0, 1.0], multiclass=True), _randn(5, 3, seed=33),
         {"y": torch.tensor([0, 1, 2, 1, 0])}),
        (L.Focal(alpha=0.25), _randn(5, 1, seed=34), {"y": torch.ones(5)}),
        (L.Quantile(quantiles=[0.1, 0.9], horizon=2), _randn(5, 4, seed=35),
         {"y": torch.zeros(5, 2)}),
        (L.CumulativeLink(), _randn(5, 2, seed=36), {"y": torch.tensor([0, 1, 2, 1, 0])}),
        (L.Reconstruction(n_numeric=2, cat_cardinalities=[3]), _randn(5, 5, seed=37),
         {"x_num": torch.zeros(5, 2), "x_cat": torch.zeros(5, 1, dtype=torch.long)}),
        (L.MSE(), _randn(5, 2, seed=40), {"y": torch.zeros(5, 2)}),
        (L.MAE(), _randn(5, 1, seed=41), {"y": torch.zeros(5)}),
        (L.Huber(delta=0.5), _randn(5, 1, seed=42), {"y": torch.zeros(5)}),
    ]
    for loss_fn, out, batch in cases:
        loss_fn(out, batch, None)
    net = _SvddNet(z=3).train()
    for seed in (38, 39):                          # the setting step and a later one
        L.SVDD()(_randn(5, 3, seed=seed), {}, net)
    vae, batch = _VaeNet().train(), {"x_num": _randn(8, 4, seed=43)}
    L.VAE(n_numeric=4)(vae(batch), batch, vae)     # the training pass reads model.kl()


# ------------------------------------------------------------------ c849f8b, byte for byte

def _baseline_text():
    """c849f8b's losses.py, read from git (`render_templates.baseline_source`: this tree, else
    the repository `$VEGA_SOURCE_REPO` names — the lane program runs the tests in a copy)."""
    return R.baseline_source("losses.py", BASELINE_COMMIT)


def _baseline_module():
    mod = types.ModuleType("losses_c849f8b")
    mod.__dict__.update(ClassWeights=D.ClassWeights, NetSpec=D.NetSpec)
    exec(compile(_baseline_text(), f"{BASELINE_COMMIT}:losses.py", "exec"), mod.__dict__)
    return mod


OLD = _baseline_module()


def _value_and_grad(loss_fn, logits, y):
    z = logits.clone().requires_grad_(True)
    value = loss_fn(z, {"y": y})
    value.backward()
    return value.detach(), z.grad


def test_the_cic_weights_and_mechanics_are_r7s():
    """The constants above are R7's record: its loss params, and the formula its mechanics
    panel printed — which the v2.3 loss built from those params still prints."""
    root = os.environ.get("CIDS_ROOT")
    record = Path(root or "/nonexistent") / CIC_BUNDLE / "records" / "architecture.json"
    if not record.is_file():
        pytest.skip("CIDS_ROOT names no R7 bundle; the weights are the recorded constants")
    arch = json.loads(record.read_text(encoding="utf-8"))
    params = arch["netspec"]["loss"]["params"]
    assert params == {"weight": CIC_WEIGHTS, "label_smoothing": CIC_SMOOTHING}
    assert arch["smoke"]["loss_formula"] == CIC_MECHANICS
    assert L.CrossEntropy.from_params(params).formula() == arch["smoke"]["loss_formula"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [1, 7, 64, 4096])
def test_cross_entropy_is_byte_identical_to_c849f8b_on_the_cic_case(dtype, rows):
    logits = _randn(rows, 15, seed=40 + rows, dtype=dtype) * 3
    y = torch.randint(0, 15, (rows,), generator=_gen(50 + rows))
    spec = _spec("graph", task={"kind": "multiclass", "label": "Label",
                                "classes": [f"c{i}" for i in range(15)]},
                 loss={"name": "cross_entropy",
                       "params": {"weight": CIC_WEIGHTS, "label_smoothing": CIC_SMOOTHING}})
    pairs = [(OLD.CrossEntropy(weight=CIC_WEIGHTS, label_smoothing=CIC_SMOOTHING),
              L.build_loss(spec)),
             (OLD.CrossEntropy(weight=None, label_smoothing=CIC_SMOOTHING),
              L.CrossEntropy.from_params({"weight": None, "label_smoothing": CIC_SMOOTHING})),
             (OLD.CrossEntropy(weight=CIC_WEIGHTS), L.CrossEntropy(weight=CIC_WEIGHTS))]
    for old, new in pairs:
        (v_old, g_old), (v_new, g_new) = (_value_and_grad(f, logits, y) for f in (old, new))
        assert torch.equal(v_old, v_new) and torch.equal(g_old, g_new), repr(new)
        assert old.formula() == new.formula() and repr(old) == repr(new)
    assert pairs[0][1].formula() == CIC_MECHANICS


def test_orders_bce_is_byte_identical_to_c849f8b():
    logits, y = _randn(512, 1, seed=60) * 4, torch.randint(0, 2, (512,), generator=_gen(61))
    old, new = OLD.BCEWithLogits(pos_weight=4.9), L.BCEWithLogits.from_params({"pos_weight": 4.9})
    (v_old, g_old), (v_new, g_new) = (_value_and_grad(f, logits, y) for f in (old, new))
    assert torch.equal(v_old, v_new) and torch.equal(g_old, g_new)
    assert repr(old) == repr(new)


def test_the_golden_formulas_are_unchanged():
    orders = json.loads((GOLDEN / "orders-smoke.json").read_text(encoding="utf-8"))
    cic = json.loads((GOLDEN / "cic-r7-smoke.json").read_text(encoding="utf-8"))
    assert L.BCEWithLogits(pos_weight=4.9).formula() == orders["loss_formula"]
    assert L.CrossEntropy(label_smoothing=0.05).formula() == cic["loss_formula"]
    for name, params in (("mse", {}), ("mae", {}), ("huber", {"delta": 2.0}),
                         ("focal", {"gamma": 2}), ("bce_with_logits", {}),
                         ("reconstruction", {"n_numeric": 3, "cat_cardinalities": [4]}),
                         ("vae", {"beta": 0.5}), ("cross_entropy", {"weight": [1.0, 2.0]})):
        old = OLD.LOSSES[name](**params)
        assert L.LOSSES[name].from_params(params).formula() == old.formula(), (name, params)


# ------------------------------------------------------------------ formulas, factories

def test_formula_names_what_each_loss_applies():
    for name, cls in L.LOSSES.items():
        assert cls().formula().startswith("L = "), name
    fq = L.Quantile(quantiles=[0.1, 0.5, 0.9], horizon=6).formula()
    assert "(Q = 3)" in fq and "(H = 6)" in fq and "h·Q + q" in fq
    assert "(Q = 2)" in L.Quantile(quantiles=[0.1, 0.9]).formula()
    assert "c fixed from the first training batch, stored with the weights" in L.SVDD().formula()
    # D1 §5.1: the ordinal loss is c849f8b's; its mechanics string names the thresholds' order
    ordinal = L.CumulativeLink().formula()
    assert ordinal.startswith(OLD.CumulativeLink().formula()[:-1])
    assert "rank-consistent thresholds, z_0 ≥ z_1 ≥ …" in ordinal
    assert "K = 3" in L.BCEWithLogits(pos_weight=[1, 2, 3]).formula()
    assert "softmax" in L.Focal(multiclass=True).formula()
    assert "α per label, K = 2" in L.Focal(alpha=[0.2, 0.4]).formula()
    # c849f8b applied alpha without writing it out; the formula now shows the loss it applies
    assert "α_t = α·y + (1-α)·(1-y), α = 0.25" in L.Focal(alpha=0.25).formula()
    assert "α" not in OLD.Focal(alpha=0.25).formula()


def test_from_params_refuses_what_a_loss_does_not_take():
    with pytest.raises(ValueError, match=r"focal loss takes no parameter 'pos_weight'; it "
                                         r"takes gamma, alpha, multiclass"):
        L.Focal.from_params({"gamma": 2, "pos_weight": 4.9})
    with pytest.raises(ValueError, match=r"mse loss takes no parameter 'delta'; it takes none"):
        L.MSE.from_params({"delta": 1.0})
    assert L.CrossEntropy.from_params({"weight": None}).weight is None
    assert repr(L.Huber.from_params({"delta": 2.0})) == "huber({'delta': 2.0})"
    for name, cls in L.LOSSES.items():
        if "__init__" not in vars(cls):     # the base's recorder: a loss that takes nothing
            assert cls.PARAMS == () and cls.__init__ is L._Loss.__init__, name
            continue
        params = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
        assert len(params) <= 5 and set(params) == set(cls.PARAMS), name


def test_build_loss_takes_a_netspec_and_resolves_the_contract():
    spec = _spec(loss={"name": "bce_with_logits", "params": {"pos_weight": 2.0}})
    assert L.build_loss(spec).formula() == L.BCEWithLogits(pos_weight=2.0).formula()
    with pytest.raises(TypeError, match="where the ports take a NetSpec"):
        L.build_loss(spec.to_dict())
    with pytest.raises(ValueError, match="unknown loss 'hinge'"):
        L.build_loss(_spec(loss={"name": "hinge", "params": {}}))
    feats = [R._num("a"), R._nominal("zone", 5), R._num("b"), R._nominal("kind", 3)]
    rec = L.build_loss(_spec(task={"kind": "unsupervised"}, features=feats,
                             loss={"name": "vae", "params": {"beta": 2.0}}))
    assert (rec.rec.n_numeric, rec.rec.cat_cardinalities, rec.beta) == (2, [5, 3], 2.0)
    pinned = L.build_loss(_spec(task={"kind": "unsupervised"}, features=feats,
                                loss={"name": "reconstruction", "params": {"n_numeric": 1}}))
    assert pinned.n_numeric == 1, "a value the spec names wins over the resolved one"
    q = L.build_loss(_spec("window", sample={"horizon": 3},
                           loss={"name": "quantile", "params": {"quantiles": [0.1, 0.5, 0.9]}}))
    assert q.horizon == 3
    seq = {"kind": "sequence-regression", "base_kind": "regression", "label": "target"}
    assert L.build_loss(_spec("window", task=seq, sample={"horizon": None},
                              loss={"name": "quantile", "params": {}})).horizon is None


@pytest.mark.parametrize("task, multiclass", [
    ({"kind": "multiclass", "label": "c", "classes": ["a", "b", "c"]}, True),
    ({"kind": "sequence-classification", "base_kind": "multiclass", "label": "c",
      "classes": ["a", "b", "c"]}, True),
    ({"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1}, False),
    ({"kind": "multilabel", "label": "tags", "classes": ["a", "b"],
      "label_format": {"kind": "delimited", "sep": "|"}}, False),
])
def test_build_loss_resolves_the_focal_link_from_the_effective_task(task, multiclass):
    focal = L.build_loss(_spec(task=task, loss={"name": "focal", "params": {"gamma": 2}}))
    assert focal.multiclass is multiclass


def test_n_numeric_is_the_feature_contracts_numeric_columns():
    """One definition of the numeric columns: every feature that is neither nominal nor a
    key, as the FeatureContract, the builder's n_num and the data adapter's x_num count them
    (T5 round 2: the contract counted a `key` column the data adapter never puts in x_num, so
    the reconstruction loss and the builder read a width the batch does not have)."""
    feats = [R._num("a"), dict(R._num("id"), role="key"), R._num("b"), R._nominal("zone", 5)]
    spec = _spec(task={"kind": "unsupervised"}, features=feats,
                 loss={"name": "reconstruction", "params": {}})
    rec = L.build_loss(spec)
    assert spec.feature_contract.numeric_columns == ("a", "b")
    assert spec.feature_contract.key_columns == ("id",)
    assert (rec.n_numeric, rec.cat_cardinalities) == (2, [5])
    out, batch = _randn(4, 2 + 5, seed=70), {"x_num": _randn(4, 2, seed=71),
                                             "x_cat": torch.zeros(4, 1, dtype=torch.long)}
    assert rec(out, batch).shape == ()


def test_a_spec_value_the_contract_fixes_must_agree_with_it():
    multi = {"kind": "multiclass", "label": "c", "classes": ["a", "b", "c"]}
    binary = {"kind": "binary", "label": "y", "classes": [0, 1], "positive": 1}
    for task, given, fixed in ((multi, False, True), (binary, True, False)):
        with pytest.raises(ValueError, match=rf"loss.params.multiclass is {given}, but the "
                                             rf"contract fixes it at {fixed} \(the effective"):
            L.build_loss(_spec(task=task, loss={"name": "focal",
                                                "params": {"gamma": 2, "multiclass": given}}))
    agreed = L.build_loss(_spec(task=multi, loss={"name": "focal",
                                                  "params": {"gamma": 2, "multiclass": True}}))
    assert agreed.multiclass is True
    quantile = {"name": "quantile", "params": {"quantiles": [0.1, 0.9], "horizon": 2}}
    with pytest.raises(ValueError, match=r"loss.params.horizon is 2, but the contract fixes it "
                                         r"at 3 \(a forecast's sample.horizon"):
        L.build_loss(_spec("window", sample={"horizon": 3}, loss=quantile))
    with pytest.raises(ValueError, match=r"horizon is 2, but the contract fixes it at None"):
        L.build_loss(_spec(task={"kind": "regression", "label": "y"}, loss=quantile))
    same = dict(quantile, params={"quantiles": [0.1, 0.9], "horizon": 3})
    assert L.build_loss(_spec("window", sample={"horizon": 3}, loss=same)).horizon == 3


PLAN_LADDER = REPO / "vega-train" / "scripts" / "plan_ladder.py"
BINARY = {"kind": "binary", "label": "late", "classes": [0, 1], "positive": 1}


def _fixture_loss():
    """The orders fixture's `loss` block, read from make_fixture.py's netspec literal."""
    for node in ast.walk(ast.parse((HERE / "make_fixture.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Dict):
            keys = [k.value if isinstance(k, ast.Constant) else None for k in node.keys]
            if "loss" in keys and "schema" in keys and \
                    ast.literal_eval(node.values[keys.index("schema")]) == "vega/netspec@1":
                return ast.literal_eval(node.values[keys.index("loss")])
    raise AssertionError("make_fixture.py writes no netspec with a loss block")


def _plan_ladder():
    if not PLAN_LADDER.is_file():
        pytest.skip("no vega-train beside this skill: the ladder's rung is not here to run")
    saved = list(sys.path)                         # plan_ladder puts its own directory first
    try:
        spec = importlib.util.spec_from_file_location("plan_ladder_b07", str(PLAN_LADDER))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


def test_the_orders_ladder_rung_builds_its_focal_alternative():
    """vega-train's spec-alternative rung on the fixture's loss (bce_with_logits pos_weight 4.9,
    alternative focal gamma 2) gives focal {pos_weight, gamma}: the carried pos_weight is dropped
    with a warning, and the loss is c849f8b's focal, which ignored it."""
    node = _fixture_loss()
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # the spec's own choice builds silently
        assert repr(L.build_loss(_spec(task=BINARY, loss=node))) == \
            "bce_with_logits({'pos_weight': 4.9})"
    rung, why = _plan_ladder().rung_spec_alternative("loss", {"loss": node}, 1)
    assert rung is not None, why
    merged = rung["delta"][0]["to"]
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        focal = L.build_loss(_spec(task=BINARY, loss=merged))
    # plan_ladder merges the old params in (a warning) until B16 replaces them (none)
    assert len(seen) == ("pos_weight" in merged["params"])
    assert all("pos_weight" in str(w.message) for w in seen)
    assert repr(focal) == "focal({'gamma': 2})" and not focal.multiclass
    logits, y = _randn(64, 1, seed=72) * 3, torch.randint(0, 2, (64,), generator=_gen(73))
    old = OLD.LOSSES["focal"](**merged["params"])
    assert torch.equal(focal(logits, {"y": y}), old(logits, {"y": y}))


def test_only_a_key_carried_from_another_weighed_loss_is_dropped():
    bce = {"name": "bce_with_logits", "params": {"pos_weight": 4.9}, "when": "the spec's own"}
    merged = {"name": "focal", "params": {"pos_weight": 4.9, "gamma": 2}, "alternatives": [bce]}
    with pytest.warns(UserWarning, match=r"loss.params pos_weight: not focal parameters but the "
                                         r"weighed alternative's"):
        assert repr(L.build_loss(_spec(task=BINARY, loss=merged))) == "focal({'gamma': 2})"
    multi = {"kind": "multiclass", "label": "c", "classes": ["a", "b", "c"]}
    ce = {"name": "cross_entropy", "params": {"weight": [1.0, 2.0, 3.0], "label_smoothing": 0.05}}
    from_ce = {"name": "focal", "params": dict(ce["params"], gamma=2), "alternatives": [ce]}
    with pytest.warns(UserWarning, match=r"loss.params label_smoothing, weight: not focal"):
        assert L.build_loss(_spec(task=multi, loss=from_ce)).multiclass is True
    refused = [  # a value the alternative does not hold; the same loss; a loss that does not
        # take the key either; no alternative at all
        dict(merged, params={"pos_weight": 5.0, "gamma": 2}),
        dict(merged, alternatives=[dict(bce, name="focal")]),
        {"name": "focal", "params": {"delta": 1.0}, "alternatives": [
            {"name": "mse", "params": {"delta": 1.0}}]},
        {"name": "focal", "params": {"pos_weight": 4.9}, "alternatives": []},
    ]
    for loss in refused:
        with pytest.raises(ValueError, match=r"the focal loss takes no parameter"):
            L.build_loss(_spec(task=BINARY, loss=loss))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
