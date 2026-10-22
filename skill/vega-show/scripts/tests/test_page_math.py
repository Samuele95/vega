#!/usr/bin/env python3
"""
test_page_math.py — the page's own arithmetic against torch (BUILD-PLAN W09, D4 §8.1, §8.3).

    python3 -m pytest -q test_page_math.py          # needs node (the page's modules) and torch

The page recomputes what the network did from the parameters it was sent, so its arithmetic must
be the network's. Node runs the page's modules (`js/how/**`); torch is the reference.

- **the seven activations** (`how/act.js` ACT, keyed by the catalog's names: none, relu, gelu —
  the erf form, `approximate="none"` —, silu, tanh, sigmoid, leaky_relu at the catalog's slope)
  within 1e-5 of `torch.nn.functional` over [−8, 8], in float64 and against the float32 values the
  network itself computes; `erf` within 1.5e-7 of `torch.erf`;
- **the eight links' readings** (`LINK_READ`, the ordinal class probabilities and expected rank,
  the reconstruction error — numeric mean squared error plus the nominal columns' mean
  cross-entropy —, the one-class distance) against torch;
- **the stepper's recompute, step by step, on every network-view fixture** (the 13 of
  `assets/page/tests/fixtures/network-view`, plus a multilabel variant of `mlp-plain` for the
  `sigmoid_each` link, plus five window networks built here with the fixtures' own builder
  (`make_fixtures.py`) for the steps no fixture has: `pool` mean, max and cls, `pool1d` max and
  avg — one with an odd length, its last position dropped —, and a standalone `activation` after
  a pooling): each network is run in torch from its own tensors (a small interpreter of the stage
  graph: dense and head with their bends, the rank-consistent ordinal head, batch and layer
  normalisation, activation, dropout, residual repeats with their projections, parallel branches
  merged, sequential and decoder composites, embeddings joined to the numeric inputs, flatten, and
  `pool` / `pool1d` run by the forge's own `blocks.Pool` / `blocks.Pool1d` in their (B, L, C)
  layout), the explained target differentiated, and the answer written as
  `vega/explanation@2` at detail `exact` (R-F5: `in`, `out`, `grad_out` per node, `pre`, `post`,
  `grad_pre` for dense and head, the dense chain when the fixture has one). The page's `Stepper`
  derives the steps and every explainer's `recompute` must agree with torch within
  1e-4·max(1, |value|) — the weighted sums, the bends, the normalisations, the residual sums and
  projections, the merges, the embedding rows, the pooled values and each link's reading. Every
  step kind the core explainers recompute is met at least once (a reshape is never a step, only a
  note: its recompute is held to hand-computed values in `tests/stepper.test.js`).
"""
from __future__ import annotations

import base64
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the page's math is checked against torch")
F = torch.nn.functional

TESTS = Path(__file__).resolve().parent
PAGE = TESTS.parents[1] / "assets" / "page"
FIXTURES = PAGE / "tests" / "fixtures" / "network-view"
FORGE = TESTS.parents[2] / "vega-forge" / "scripts"
EXPLAIN = FORGE / "backends" / "pytorch" / "explain-catalog.json"
CATALOG = FORGE / "catalog.json"
NAMES = ("mlp-plain", "linear", "mlp-residual", "embedding", "autoencoder", "parallel", "lstm-window", "gcn-node",
         "ordinal", "svdd", "forecast", "recon-nominal", "residual-deep")
TOLERANCE = 1e-4


def _module(name: str, path: Path):
    """A module loaded from its file under a name of its own (so no other test's import clashes)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# the forge's block implementations (the pooling the network runs) and the fixtures' builder
BLOCKS = _module("page_math_forge_blocks", FORGE / "backends" / "pytorch" / "blocks.py")
MAKE = _module("page_math_network_view_fixtures", FIXTURES / "make_fixtures.py")


def node_json(script: str, data=None):
    """Run an ES module script with node inside the page template (so `./js/…` resolves); `data`
    is written to a JSON file whose path the script reads as process.argv[1]."""
    node = shutil.which("node")
    assert node, "node is required: the page's arithmetic is its ES modules"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        run = subprocess.run([node, "--input-type=module", "-e", script, str(path)], cwd=PAGE,
                             capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


# --------------------------------------------------------------------------- activations, erf, links
TORCH_ACT = {
    "none": lambda z: z,
    "relu": F.relu,
    "gelu": lambda z: F.gelu(z, approximate="none"),
    "silu": F.silu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "leaky_relu": lambda z: F.leaky_relu(z, 0.01),
}


def test_the_seven_activations_match_torch_within_1e_5_over_minus_8_to_8():
    catalog = json.loads(EXPLAIN.read_text(encoding="utf-8"))
    assert sorted(TORCH_ACT) == sorted(catalog["activations"])
    assert catalog["activations"]["leaky_relu"]["params"]["slope"] == 0.01
    z = torch.linspace(-8, 8, 3201, dtype=torch.float64)
    page = node_json("""
      import { readFileSync } from "node:fs";
      import { ACT, erf } from "./js/how/act.js";
      const z = JSON.parse(readFileSync(process.argv[1], "utf8"));
      const out = Object.fromEntries(Object.entries(ACT).map(([k, f]) => [k, z.map((v) => f(v))]));
      out.erf = z.map(erf);
      console.log(JSON.stringify(out));
    """, z.tolist())
    assert sorted(k for k in page if k != "erf") == sorted(TORCH_ACT)
    for name, f in TORCH_ACT.items():
        got = torch.tensor(page[name], dtype=torch.float64)
        want64, want32 = f(z), f(z.to(torch.float32)).to(torch.float64)
        assert float((got - want64).abs().max()) <= 1e-5, f"{name}: page vs torch float64"
        assert float((got - want32).abs().max()) <= 1e-5, f"{name}: page vs the float32 network"
    assert float((torch.tensor(page["erf"], dtype=torch.float64) - torch.erf(z)).abs().max()) <= 1.5e-7


def test_the_eight_links_read_a_head_as_torch_does():
    g = torch.Generator().manual_seed(7)
    z = (torch.randn(9, generator=g, dtype=torch.float64) * 3).tolist()
    ordinal = sorted((torch.randn(4, generator=g, dtype=torch.float64) * 2).tolist(), reverse=True)
    xhat = torch.randn(4 + 3 + 2, generator=g, dtype=torch.float64)
    x, ids, cards = torch.randn(4, generator=g, dtype=torch.float64), [2, 0], [3, 2]
    centre = torch.randn(9, generator=g, dtype=torch.float64)
    page = node_json("""
      import { readFileSync } from "node:fs";
      import { LINK_READ, LINKS, ordinalProbs, ordinalRank, expectedRank, reconstructionError, svddDistance } from "./js/how/act.js";
      const d = JSON.parse(readFileSync(process.argv[1], "utf8"));
      const read = Object.fromEntries(LINKS.map((l) => [l, LINK_READ[l](d.z)]));
      console.log(JSON.stringify({ links: LINKS, read, probs: ordinalProbs(d.ordinal), rank: ordinalRank(d.ordinal),
        expected: expectedRank(d.ordinal), recon: reconstructionError(d.xhat, [...d.x, ...d.ids], d.cards),
        numeric: reconstructionError(d.xhat.slice(0, 4), d.x), distance: svddDistance(d.z, d.centre) }));
    """, {"z": z, "ordinal": ordinal, "xhat": xhat.tolist(), "x": x.tolist(), "ids": ids, "cards": cards,
          "centre": centre.tolist()})
    assert sorted(page["links"]) == ["cumulative", "distance", "horizon", "identity", "reconstruction", "sigmoid",
                                     "sigmoid_each", "softmax"]
    zt = torch.tensor(z, dtype=torch.float64)
    close = lambda got, want: float((torch.tensor(got, dtype=torch.float64) - want).abs().max()) <= 1e-12  # noqa: E731
    assert close(page["read"]["softmax"], torch.softmax(zt, 0))
    assert close(page["read"]["sigmoid"], torch.sigmoid(zt)) and close(page["read"]["sigmoid_each"], torch.sigmoid(zt))
    assert close(page["read"]["cumulative"], torch.sigmoid(zt))
    for same in ("identity", "reconstruction", "horizon"):
        assert close(page["read"][same], zt)
    assert page["read"]["distance"] is None
    s = torch.sigmoid(torch.tensor(ordinal, dtype=torch.float64))
    probs = torch.cat([1 - s[:1], s[:-1] - s[1:], s[-1:]])
    assert close(page["probs"], probs) and abs(float(probs.sum()) - 1) < 1e-12
    assert page["rank"] == int((s >= 0.5).sum()) and abs(page["expected"] - float(s.sum())) < 1e-12
    ce = torch.stack([F.cross_entropy(xhat[4:7][None], torch.tensor([2])), F.cross_entropy(xhat[7:9][None], torch.tensor([0]))])
    want = F.mse_loss(xhat[:4], x) + ce.mean()
    assert abs(page["recon"]["error"] - float(want)) < 1e-12
    assert abs(page["numeric"]["error"] - float(F.mse_loss(xhat[:4], x))) < 1e-12
    assert abs(page["distance"] - float(((zt - centre) ** 2).sum())) < 1e-12


# --------------------------------------------------------------------------- torch runs of the fixtures
def _values(t) -> torch.Tensor:
    """A network-view tensor payload (json or base64-le float32) as a float32 tensor."""
    shape = t["shape"]
    if t["encoding"] == "json":
        return torch.tensor(t["data"], dtype=torch.float32).reshape(shape)
    raw = base64.b64decode(t["data"])
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).clone()


def _exact(t: torch.Tensor, dtype: str = "float32") -> dict:
    a = t.detach().contiguous()
    data = a.to(torch.int64 if dtype == "int64" else torch.float32).numpy().tobytes()
    return {"shape": list(a.shape), "axes": [f"a{i}" for i in range(a.dim())], "dtype": dtype, "encoding": "base64-le",
            "data": base64.b64encode(data).decode("ascii")}


class Run:
    """One forward pass of a fixture's stage graph in torch, every node's input and output kept."""

    def __init__(self, view: dict):
        self.view, self.nodes = view, {n["id"]: n for n in view["nodes"]}
        # every learned parameter differentiable (buffers are not), so each node's output carries its
        # gradient of the target
        self.tensors = {k: _values(t).requires_grad_(t.get("role") != "buffer") for k, t in view["tensors"].items()}
        self.seen = {}

    def vec(self, ref):
        if ref is None:
            return None
        return self.tensors[ref] if isinstance(ref, str) else torch.tensor(ref, dtype=torch.float32)

    def keep(self, nid, x, y, **extra):
        if isinstance(y, torch.Tensor) and y.requires_grad:
            y.retain_grad()
        self.seen[nid] = {"in": x, "out": y, **extra}
        return y

    def act(self, name, z):
        return TORCH_ACT[name or "none"](z)

    def dense(self, n, x):
        p, W = n["params"], self.tensors[n["params"]["weight"]]
        th = p.get("thresholds")
        z = F.linear(x, W, self.vec(p.get("bias")))
        if th:                                   # the rank-consistent ordinal head: w·x + θ_j
            z = z + torch.tensor(th, dtype=torch.float32)
        if z.requires_grad:
            z.retain_grad()
        post = self.act(p.get("activation"), z)
        return self.keep(n["id"], x, post, pre=z, post=post)

    def norm(self, n, x):
        p = n["params"]
        if n["type"] == "batchnorm":
            y = F.batch_norm(x[None], self.vec(p["running_mean"]), self.vec(p["running_var"]), self.vec(p.get("gamma")),
                             self.vec(p.get("beta")), False, 0.0, float(p["eps"]))[0]
        else:
            y = F.layer_norm(x, (x.shape[-1],), self.vec(p.get("gamma")), self.vec(p.get("beta")), float(p["eps"]))
        return self.keep(n["id"], x, y)

    def sequence(self, ids, x):
        for nid in ids:
            x = self.stage(nid, x)
        return x

    def residual(self, n, x):
        x0 = x
        for r in n["params"]["repeats"]:
            f = self.sequence(r["inner"], x)
            skip = self.stage(r["projection"], x) if r["projection"] else x
            add = next(c for c in n["children"] if c.startswith(r["inner"][0].rsplit(".", 1)[0]) and c.endswith(".add"))
            x = self.keep(add, x, skip + f)
        return self.keep(n["id"], x0, x)

    def parallel(self, n, x):
        outs = [self.stage(b, x) for b in n["params"]["branches"]]
        y = torch.cat(outs, -1) if n["params"]["merge"] == "concat" else torch.stack(outs).sum(0)
        merge = next(c for c in n["children"] if self.nodes[c]["type"] == "parallel_merge")
        self.keep(merge, torch.cat(outs, -1), y)
        return self.keep(n["id"], x, y)

    def stage(self, nid, x):
        n = self.nodes[nid]
        kind = n["type"]
        if kind in ("dense", "head"):
            return self.dense(n, x)
        if kind in ("batchnorm", "layernorm"):
            return self.norm(n, x)
        if kind == "activation":
            return self.keep(nid, x, self.act(n["params"]["name"], x))
        if kind in ("dropout", "custom"):                  # identity while answering (custom: shapes only)
            return self.keep(nid, x, x * 1.0)
        if kind == "flatten":
            return self.keep(nid, x, x.reshape(-1))
        if kind == "pool":                                   # (L, C) -> (C), as the network runs it on (B, L, C)
            return self.keep(nid, x, BLOCKS.Pool(n["params"]["kind"])(x[None])[0])
        if kind == "pool1d":                                 # (L, C) -> (L // size, C)
            p = n["params"]
            return self.keep(nid, x, BLOCKS.Pool1d(p["kind"], p["size"])(x[None])[0])
        if kind in ("sequential", "decoder"):
            return self.keep(nid, x, self.sequence(n["children"], x))
        if kind == "residual":
            return self.residual(n, x)
        if kind == "parallel":
            return self.parallel(n, x)
        raise NotImplementedError(kind)

    def forward(self, x_num, x_cat):
        """The top-level stages in path order; an embedding reads the nominal ids and leaves the
        running tensor untouched until the concat joins them (inspect-contract §3.2)."""
        top = sorted((n for n in self.view["nodes"] if n["parent"] is None and n["path"].startswith("stages.")),
                     key=lambda n: int(n["path"].split(".")[1]))
        x, emb = x_num, None
        for n in top:
            if n["type"] == "embedding":
                rows = [self.tensors[t["table"]][int(x_cat[k])] for k, t in enumerate(n["params"]["tables"])]
                emb = self.keep(n["id"], x_cat, torch.cat(rows))
            elif n["type"] == "concat":
                x = self.keep(n["id"], x, torch.cat([x, emb]))
            else:
                x = self.stage(n["id"], x)
        return x


def _target(view, raw: torch.Tensor, x_num, x_cat, run: Run):
    """The default target of the fixture's task (inspect-contract §4.1): (kind, index, scalar)."""
    task, link = view["task"], view["task"]["link"]
    if link == "softmax":
        k = int(raw.argmax())
        return "logit", k, raw[k]
    if link in ("sigmoid", "sigmoid_each"):
        k = int(raw.argmax())
        return ("binary_logit", 0, raw[0]) if link == "sigmoid" else ("label_logit", k, raw[k])
    if link == "cumulative":
        return "expected_rank", None, torch.sigmoid(raw).sum()
    if link == "distance":
        c = run.tensors[f"{_head(view)['id']}.centre"]
        return "svdd_distance", None, ((raw - c) ** 2).sum()
    if link == "reconstruction":
        return "reconstruction_error", None, _recon(view, raw, x_num, x_cat)
    return ("value" if link == "identity" else "forecast_step"), 0, raw[0]


def _head(view):
    return [n for n in view["nodes"] if n["type"] == "head"][-1]


def _recon(view, raw, x_num, x_cat):
    """The reconstruction error as the readout scores it: numeric MSE + mean nominal CE."""
    n = len(view["inputs"]["numeric"])
    err = F.mse_loss(raw[:n], x_num.reshape(-1)[:n]) if n else raw.new_zeros(())
    at, terms = n, []
    for k, col in enumerate(view["inputs"]["nominal"]):
        terms.append(F.cross_entropy(raw[at:at + col["cardinality"]][None], x_cat[k:k + 1].long()))
        at += col["cardinality"]
    return err + torch.stack(terms).mean() if terms and raw.numel() == at else err


def _answer(view, raw, x_num, x_cat, run):
    """explanation@2 outputs.read and outputs.answer, as the service states them."""
    link, raw, x_num = view["task"]["link"], raw.detach(), x_num.detach()
    read = {"softmax": lambda: torch.softmax(raw, 0), "sigmoid": lambda: torch.sigmoid(raw),
            "sigmoid_each": lambda: torch.sigmoid(raw), "cumulative": lambda: torch.sigmoid(raw)}.get(link, lambda: raw)()
    names = view["task"]["outputs"]["names"]
    answer = {"softmax": lambda: {"class": names[int(raw.argmax())], "p": float(read.max())},
              "cumulative": lambda: {"rank": int((read >= 0.5).sum()), "expected_rank": float(read.sum())},
              "reconstruction": lambda: {"error": float(_recon(view, raw, x_num, x_cat))},
              "distance": lambda: {"distance": float(((raw - run.tensors[f"{_head(view)['id']}.centre"]) ** 2).sum())}}
    return (None if link == "distance" else [float(v) for v in read.reshape(-1)]), answer.get(link, lambda: {})()


def explanation(view: dict, seed: int = 11) -> dict:
    """The fixture's network run on one sample drawn from the seed, written as explanation@2 at
    detail "exact" (in, out, grad_out per node; pre, post, grad_pre for dense and head)."""
    g = torch.Generator().manual_seed(seed)
    shape = [n for n in view["nodes"] if n["id"] == "input.num"][0]["out_shape"]
    x_num = torch.randn(*shape, generator=g).requires_grad_(True)
    x_cat = torch.tensor([int(torch.randint(0, c["cardinality"], (1,), generator=g)) for c in view["inputs"]["nominal"]])
    run = Run(view)
    raw = run.forward(x_num, x_cat)
    kind, index, s = _target(view, raw, x_num, x_cat, run)
    s.backward()
    nodes = {}
    for nid, c in run.seen.items():
        exact = {"in": _exact(c["in"], "int64" if nid in {n["id"] for n in view["nodes"] if n["type"] == "embedding"} else "float32"),
                 "out": _exact(c["out"]), "grad_out": _exact(c["out"].grad) if c["out"].grad is not None else None}
        if "pre" in c:
            exact.update(pre=_exact(c["pre"]), post=_exact(c["post"]),
                         grad_pre=_exact(c["pre"].grad) if c["pre"].grad is not None else None)
        nodes[nid] = {"level": "exact", "why": None, "in": None, "out": None, "grad_out": None, "relevance": None,
                      "extra": {}, "recompute": None, "exact": exact}
    read, answer = _answer(view, raw, x_num, x_cat, run)
    chain = None
    if view["chain"]:
        layers = [run.seen[l["node"]] for l in view["chain"]["layers"]]
        post = [run.seen[l["node"]]["post"] for l in view["chain"]["layers"]]
        # a chain layer's post is what the next layer reads (its activation block's output, dropout the identity)
        post = [layers[i + 1]["in"] if i + 1 < len(layers) else post[i] for i in range(len(layers))]
        chain = {"inputs": layers[0]["in"].tolist(), "pre": [c["pre"].tolist() for c in layers],
                 "post": [p.tolist() for p in post], "grad_pre": [c["pre"].grad.tolist() for c in layers]}
    order = view["inputs"]["order"]
    values = x_num.reshape(-1).tolist() + x_cat.tolist() if len(shape) == 1 else None
    return {
        "schema": "vega/explanation@2", "detail": "exact", "degraded": False,
        "sample": {"shape": view["sample"]["shape"], "index": 0, "row": 0, "span": None, "graph": None},
        "prediction": {}, "target": {"kind": kind, "index": index, "name": None, "sign": 1, "value": float(s.detach()), "why": "default",
                                     "unit": "logit"},
        "outputs": {"kind": view["task"]["outputs"]["kind"], "names": view["task"]["outputs"]["names"], "link": view["task"]["link"],
                    "raw": [float(v) for v in raw.detach()], "read": read, "answer": answer},
        "inputs": {"order": order, "shaped": x_num.tolist(), "raw": None},
        "attribution": {"method": "grad_x_input", "unit": "logit", "shape": view["sample"]["shape"],
                        "features": {"names": order, "values": values, "attribution": [0.0] * len(order), "grad": [0.0] * len(order)},
                        "total": 0.0, "completeness": None},
        "nodes": nodes, "chain": chain, "fidelity": {}, "recompute": {}, "budget": {}, "warnings": [],
    }


def multilabel(view: dict) -> dict:
    """mlp-plain read as a multilabel network: the same head, every label on its own (σ each)."""
    v = copy.deepcopy(view)
    v["task"].update(kind="multilabel", link="sigmoid_each", outputs={"kind": "labels", "names": ["l0", "l1", "l2"]},
                     threshold=0.5, decision_rule="per_label_threshold", targets=["label_logit"], default_target="label_logit")
    for n in v["nodes"]:
        if n["type"] == "output":
            n["params"].update(link="sigmoid_each", outputs=v["task"]["outputs"])
    v["chain"] = None
    return v


# --------------------------------------------------------------------------- window networks built here
def _window(name: str, seed: int, length: int, width: int):
    """A window network under construction: its builder, input names and step labels."""
    net, xs = MAKE.Net(name, seed, json.loads(CATALOG.read_text(encoding="utf-8"))), MAKE.names("x", width)
    steps = [f"t−{length - 1 - i}" if i < length - 1 else "t" for i in range(length)]
    MAKE.io_in(net, "input.num", "numeric", xs, [length, width], axes=("step", "feature"))
    return net, xs, steps


def _pool(net, nid: str, path: str, kind: str, length: int, width: int):
    return net.node(nid, "pool", path=path, in_shape=[length, width], out_shape=[width], config={"kind": kind}, params={"kind": kind})


def _pool1d(net, nid: str, path: str, kind: str, size: int, length: int, width: int):
    return net.node(nid, "pool1d", path=path, in_shape=[length, width], out_shape=[length // size, width], axes=("step", "unit"),
                    config={"kind": kind, "size": size}, params={"kind": kind, "size": size})


def _classes(net, xs, steps, ks, length):
    return net.document(family="synthetic", task=MAKE.task("sequence-classification", "softmax", "classes", ks, "argmax", ["logit"], base="multiclass"),
                        sample=MAKE.sample("window", length=length, stride=1), inputs=MAKE.inputs(xs, step=steps))


def _value(net, xs, steps, length):
    return net.document(family="synthetic", task=MAKE.task("sequence-regression", "identity", "value", ["y"], "none", ["value"], base="regression"),
                        sample=MAKE.sample("window", length=length, stride=1), inputs=MAKE.inputs(xs, step=steps))


def pool_mean():
    """window (6, 4), multiclass: pool(mean), a standalone activation (tanh), dense, head."""
    net, xs, steps = _window("pool-mean", 31, 6, 4)
    ks = MAKE.names("k", 3)
    _pool(net, "B1", "stages.0", "mean", 6, 4)
    net.pointwise("B2", "activation", 4, name="tanh", path="stages.1")
    net.dense("B3", 4, 5, act="relu", path="stages.2")
    net.dense("B4", 5, 3, type_="head", path="stages.3", axes=("output",))
    MAKE.io_out(net, "softmax", "classes", ks)
    net.chain("input.num", "B1", "B2", "B3", "B4")
    net.edge("B4", "output", "readout")
    return _classes(net, xs, steps, ks, 6)


def pool_max():
    """window (5, 3), regression: pool(max), head."""
    net, xs, steps = _window("pool-max", 32, 5, 3)
    _pool(net, "B1", "stages.0", "max", 5, 3)
    net.dense("B2", 3, 1, type_="head", path="stages.1", axes=("output",))
    MAKE.io_out(net, "identity", "value", ["y"])
    net.chain("input.num", "B1", "B2")
    net.edge("B2", "output", "readout")
    return _value(net, xs, steps, 5)


def pool_cls():
    """window (4, 3), binary: pool(cls), dense (gelu), head."""
    net, xs, steps = _window("pool-cls", 33, 4, 3)
    _pool(net, "B1", "stages.0", "cls", 4, 3)
    net.dense("B2", 3, 4, act="gelu", path="stages.1")
    net.dense("B3", 4, 1, type_="head", path="stages.2", axes=("output",))
    MAKE.io_out(net, "sigmoid", "score", ["yes"])
    net.chain("input.num", "B1", "B2", "B3")
    net.edge("B3", "output", "readout")
    return net.document(family="synthetic", task=MAKE.task("sequence-classification", "sigmoid", "score", ["yes"], "threshold", ["binary_logit"],
                                                            base="binary", positive="yes", threshold=0.5, scale="probability"),
                        sample=MAKE.sample("window", length=4, stride=1), inputs=MAKE.inputs(xs, step=steps))


def pool1d_max():
    """window (6, 3), multiclass: pool1d(max, 2), a standalone activation (relu) on the (3, 3) groups,
    flatten, dense (silu), head — the cnn1d family's tail (BUILD-PLAN L12)."""
    net, xs, steps = _window("pool1d-max", 34, 6, 3)
    ks = MAKE.names("k", 3)
    _pool1d(net, "B1", "stages.0", "max", 2, 6, 3)
    net.pointwise("B2", "activation", [3, 3], name="relu", path="stages.1", axes=("step", "unit"))
    net.node("B3", "flatten", path="stages.2", in_shape=[3, 3], out_shape=[9], config={})
    net.dense("B4", 9, 4, act="silu", path="stages.3")
    net.dense("B5", 4, 3, type_="head", path="stages.4", axes=("output",))
    MAKE.io_out(net, "softmax", "classes", ks)
    net.chain("input.num", "B1", "B2", "B3", "B4", "B5")
    net.edge("B5", "output", "readout")
    return _classes(net, xs, steps, ks, 6)


def pool1d_avg():
    """window (7, 2), regression: pool1d(avg, 2) — the 7th position dropped, as AvgPool1d does —,
    flatten, head."""
    net, xs, steps = _window("pool1d-avg", 35, 7, 2)
    _pool1d(net, "B1", "stages.0", "avg", 2, 7, 2)
    net.node("B2", "flatten", path="stages.1", in_shape=[3, 2], out_shape=[6], config={})
    net.dense("B3", 6, 1, type_="head", path="stages.2", axes=("output",))
    MAKE.io_out(net, "identity", "value", ["y"])
    net.chain("input.num", "B1", "B2", "B3")
    net.edge("B3", "output", "readout")
    return _value(net, xs, steps, 7)


SYNTHETIC = {"pool-mean": pool_mean, "pool-max": pool_max, "pool-cls": pool_cls, "pool1d-max": pool1d_max, "pool1d-avg": pool1d_avg}


STEPPER = """
  import { readFileSync } from "node:fs";
  import { install } from "./tests/dom-shim.js";
  const doc = install();
  const { StageGraph } = await import("./js/net/graph.js");
  const { Stepper } = await import("./js/how/stepper.js");
  const { load } = await import("./js/how/explainers/index.js");
  const { configureText, missingKeys, unresolved } = await import("./js/core/text.js");
  const { Lexicon } = await import("./js/core/lexicon.js");
  const cases = JSON.parse(readFileSync(process.argv[1], "utf8")), out = {};
  for (const c of cases) {
    const graph = StageGraph.from(c.view);
    configureText({ lex: new Lexicon(null, null), lane: { viewKey: graph.viewKey, shape: graph.sample.shape }, explain: c.page.explain });
    const host = doc.createElement("div");
    doc.body.append(host);
    const st = await new Stepper({ graph, ex: c.ex, page: c.page, onCode: () => {} }).mount(host);
    const bodies = st.steps().map((_, i) => { st.show(i); return doc.getElementById("how-body").textContent; });
    out[c.name] = { values: st.values(), gaps: st.explainers.gaps.map((g) => g.kind), bodies, failed: st.reading().failed };
    host.replaceChildren();
    host.remove ? host.remove() : doc.body.replaceChildren();
  }
  console.log(JSON.stringify({ cases: out, missing: missingKeys(), unresolved: unresolved() }));
"""


def page_of(view: dict) -> dict:
    """The part of page.json the stepper reads: the explain catalog's entries for the graph's block
    types, every activation and the task's link (page_config._explain's subset), and a recipe."""
    catalog = json.loads(EXPLAIN.read_text(encoding="utf-8"))
    types = sorted({n["type"] for n in view["nodes"] if n["type"] in catalog["blocks"]})
    return {"explain": {"blocks": {t: catalog["blocks"][t] for t in types}, "activations": catalog["activations"],
                        "links": {view["task"]["link"]: catalog["links"][view["task"]["link"]]}},
            "recipe": {"ops": [{"op": "impute", "columns": 2, "step": "S1"}, {"op": "log1p", "columns": 3, "step": "S2"},
                               {"op": "robust_scale", "columns": 5, "step": "S3"}], "source": "shape:manifest.steps"}}


# The step kinds the core explainers recompute (the others are W14's, or none: input, custom).
RECOMPUTED = {"dense", "activation", "norm", "residual", "merge", "embedding", "pool", "pool1d", "reshape", "link"}
# A reshape (flatten, reshape) does nothing while answering: the stepper makes it a note, never a
# step (net/kinds.js QUIET_KINDS), so no network brings its recompute here.
NEVER_A_STEP = {"reshape"}


def _cases():
    views = {n: json.loads((FIXTURES / f"{n}.json").read_text(encoding="utf-8")) for n in NAMES}
    views["multilabel"] = multilabel(views["mlp-plain"])
    views.update({n: make() for n, make in SYNTHETIC.items()})
    runnable = [n for n in views if n not in ("lstm-window", "gcn-node")]   # recurrent and graph blocks: W14's
    return [{"name": n, "view": views[n], "ex": explanation(views[n]), "page": page_of(views[n])} for n in runnable]


def test_the_window_networks_built_here_are_network_view_documents():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((FORGE.parent / "references" / "schemas" / "network-view-2.schema.json").read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for name, make in SYNTHETIC.items():
        errors = [f"{'/'.join(map(str, e.path))}: {e.message[:160]}" for e in validator.iter_errors(make())]
        assert errors == [], f"{name}: {errors[:3]}"


def test_the_stepper_recomputes_every_step_of_every_fixture_as_torch_ran_it():
    cases = _cases()
    got = node_json(STEPPER, cases)["cases"]
    links = set()
    for c in cases:
        steps = got[c["name"]]["values"]
        assert steps and steps[0]["kind"] == "input" and steps[-1]["kind"] == "link", c["name"]
        assert got[c["name"]]["gaps"] == [], f"{c['name']}: a core step fell back to the generic explainer"
        for s in steps:
            if s["kind"] not in RECOMPUTED:
                continue
            chk = s["check"]
            assert chk is not None, f"{c['name']} {s['name']}: no recompute"
            for part in [chk, *chk.get("ops", [])]:
                tol = TOLERANCE * max(1.0, abs(part["network"]))
                assert abs(part["gap"]) <= tol, f"{c['name']} {s['name']} {part.get('what')}: {part['page']} vs {part['network']}"
            assert chk["within"], f"{c['name']} {s['name']}"
        link = steps[-1]["check"]
        what = {part.get("what") for part in [link, *link.get("ops", [])]}
        assert "z" in what, f"{c['name']}: the head's sum"
        if c["view"]["task"]["link"] not in ("identity", "horizon"):
            assert what & {"read", "error", "distance"}, f"{c['name']}: the link's reading"
        links.add(c["view"]["task"]["link"])
    assert links == {"softmax", "sigmoid", "sigmoid_each", "identity", "cumulative", "reconstruction", "distance", "horizon"}
    met = {s["kind"] for c in cases for s in got[c["name"]]["values"] if s["kind"] in RECOMPUTED and s["check"] is not None}
    assert met == RECOMPUTED - NEVER_A_STEP, f"recompute never checked for {sorted(RECOMPUTED - NEVER_A_STEP - met)}"
    pooled = {(c["name"], s["kind"]): s["check"] for c in cases for s in got[c["name"]]["values"] if s["kind"] in ("pool", "pool1d")}
    assert set(pooled) == {("pool-mean", "pool"), ("pool-max", "pool"), ("pool-cls", "pool"), ("pool1d-max", "pool1d"), ("pool1d-avg", "pool1d")}


def test_the_steps_of_each_fixture_are_the_stage_graphs():
    cases = _cases()
    got = node_json(STEPPER, cases)["cases"]
    kinds = {c["name"]: [s["kind"] for s in got[c["name"]]["values"]] for c in cases}
    assert kinds["mlp-plain"] == ["input", "dense", "dense", "link"]
    assert kinds["mlp-residual"] == ["input", "dense", "dense", "residual", "link"]
    assert kinds["embedding"] == ["input", "embedding", "dense", "link"]
    assert kinds["parallel"] == ["input", "dense", "dense", "dense", "merge", "custom", "link"]
    assert kinds["residual-deep"] == ["input", "dense", "dense", "residual", "dense", "residual", "dense", "residual", "norm", "link"]
    assert kinds["svdd"] == ["input", "dense", "link"]
    assert kinds["forecast"] == ["input", "dense", "link"]
    fused = [s for s in got["svdd"]["values"] if s["kind"] == "dense"][0]["check"]
    assert [o["what"] for o in fused["ops"]] == ["norm", "act"], "a batchnorm and a bend fused into the dense step, in order"
    assert kinds["pool-mean"] == ["input", "pool", "activation", "dense", "link"], "a bend after a pooling stands alone"
    assert kinds["pool-max"] == ["input", "pool", "link"]
    assert kinds["pool-cls"] == ["input", "pool", "dense", "link"]
    assert kinds["pool1d-max"] == ["input", "pool1d", "activation", "dense", "link"]
    assert kinds["pool1d-avg"] == ["input", "pool1d", "link"]



def test_every_step_of_every_fixture_renders_with_the_catalogs_words_and_no_key_missing():
    """Each step's body, built in a small DOM (tests/dom-shim.js) with page.json's explain subset
    and a recipe: no explainer failed, no key of the strings table is missing, no placeholder was
    left unfilled (the catalog's {units}, {inputs}, {eps}, … and the lexicon's paths), and every
    body says something."""
    doc = node_json(STEPPER, _cases())
    assert doc["missing"] == [], doc["missing"]
    assert doc["unresolved"] == [], doc["unresolved"]
    for name, got in doc["cases"].items():
        assert got["failed"] == [], f"{name}: {got['failed']}"
        for i, body in enumerate(got["bodies"]):
            assert len(body.strip()) > 20, f"{name} step {i}: {body!r}"
        for v, body in zip(got["values"], got["bodies"]):
            if v["kind"] in ("dense", "norm", "residual", "merge"):
                assert v["name"] and v["name"] in body, f"{name}: the {v['kind']} step names itself ({v['name']!r})"
    first = doc["cases"]["mlp-plain"]["bodies"][0]
    assert "heavy tails compressed" in first and "missing values filled in" in first, "the recipe's clauses (D4 §14.1)"
    assert "Softmax" in doc["cases"]["mlp-plain"]["bodies"][-1] and "boundaries passed" in doc["cases"]["ordinal"]["bodies"][-1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
