#!/usr/bin/env python3
"""
test_inspection_surface.py — every block class exposes the inspection surface that
`backends/pytorch/coverage.json` `inspection.surface` names for its type (BUILD-PLAN R-B3,
backend-contract §2.1, D3 §14.3).

    python3 test_inspection_surface.py      (or: python3 -m pytest test_inspection_surface.py -q)

The inspector (`/v1/network`, `/v1/explain`) reads a block through these attributes and
methods and nothing else, so a block class that drops one breaks every page that draws it.
What is checked:
- every catalog block type has a surface row, and every block class (`CLASS_FOR`, and the
  variants of `CLASS_VARIANTS`, e.g. the ordinal `CumulativeHead`) exposes its type's row on
  every configuration `test_blocks.py` builds (a SAGE block without a root weight still has a
  `lin_root`, which is None);
- the `surface_variants` and `surface_optional` refinements of coverage.json, when present;
- the methods are callable with the signatures R-B2 / R-B3 fix: `coefficients(edge_index, n,
  edge_weight=None)` for GCN, SAGE and GIN, `coefficients(h, edge_index, edge_weight=None)`
  for GAT, `AttentionPool.weights(x)`, and the positional graph forwards;
- the values mean what the contract says (`ret` is a readout name, `rnn` a torch recurrent
  module, `normalize` / `aggregator` / `readout` are catalog values, ...).

The surface table is read from coverage.json. Until the contracts package (K1) has written
its `inspection` section, the table of D3 §14.3 — which R-B3 makes normative verbatim — is
used instead, and the test says so.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
from torch import nn

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
BACKEND = SCRIPTS / "backends" / "pytorch"
for _p in (str(BACKEND), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import blocks as B  # noqa: E402
from test_blocks import CASES  # noqa: E402

COVERAGE = BACKEND / "coverage.json"
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))

# D3 §14.3, verbatim (R-B3): the table coverage.json carries once K1 has written it
D3_SURFACE = {
    "dense": ["linear", "act"], "head": ["linear", "act"],
    "embedding": ["columns", "embeddings", "out_features"],
    "batchnorm": ["bn"], "layernorm": ["ln"], "dropout": ["drop"], "activation": ["act"],
    "reshape": ["shape"], "pool": ["kind"], "sequential": ["inner"],
    "residual": ["repeats", "projections"], "parallel": ["branches", "merge"],
    "conv1d": ["conv"], "pool1d": ["pool"], "rnn": ["rnn", "ret"], "gru": ["rnn", "ret"],
    "lstm": ["rnn", "ret"],
    "attention": ["attn", "pool"], "attention_pool": ["score", "query", "weights"],
    "transformer_encoder": ["enc"], "feature_tokenizer": ["weight", "bias", "cls"],
    "positional": ["kind", "pe"], "decoder": ["net"], "reparameterize": ["z"],
    "gcn_conv": ["lin", "bias", "normalize", "self_loops", "coefficients"],
    "sage_conv": ["lin_root", "lin_neigh", "aggregator", "normalize", "coefficients"],
    "gat_conv": ["lin", "att_src", "att_dst", "heads", "concat", "negative_slope",
                 "coefficients"],
    "gin_conv": ["mlp", "eps", "coefficients"], "graph_pool": ["readout"], "edge_dropout": ["p"],
    "flatten": [], "concat": [], "custom": [],
}

METHODS = {"coefficients", "weights"}


def inspection_section():
    """(the `inspection` section to check against, where it came from)."""
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    section = coverage.get("inspection") or {}
    if section.get("surface"):
        return section, "coverage.json inspection.surface"
    return {"surface": D3_SURFACE}, ("D3 §14.3 verbatim (coverage.json has no inspection "
                                     "section yet; K1 writes it)")


SECTION, SOURCE = inspection_section()
SURFACE = SECTION["surface"]


def _instances(type_, cls_name=None):
    """One built block per case of this type (every configuration test_blocks builds)."""
    out = []
    for case in CASES:
        if case.type != type_:
            continue
        module = case.build()
        if cls_name is None or type(module).__name__ == cls_name:
            out.append((case.id, module))
    return out


def test_the_surface_table_covers_every_catalog_block():
    print(f"surface table: {SOURCE}")
    assert set(SURFACE) == set(CATALOG["blocks"]), sorted(set(SURFACE) ^ set(CATALOG["blocks"]))
    assert set(B.CLASS_FOR) | {"custom"} <= set(SURFACE)


@pytest.mark.parametrize("type_", sorted(t for t in SURFACE if t != "custom"))
def test_every_block_class_exposes_its_surface(type_):
    instances = _instances(type_)
    assert instances, f"no block of type {type_} is built by test_blocks.CASES"
    classes = {type(m).__name__ for _, m in instances}
    wanted = {B.CLASS_FOR[type_]} | set(B.CLASS_VARIANTS.get(type_, {}).values())
    assert wanted <= classes, f"{type_}: no case builds {sorted(wanted - classes)}"
    for cid, module in instances:
        missing = [a for a in SURFACE[type_] if not hasattr(module, a)]
        assert not missing, f"{cid} ({type(module).__name__}) lacks {missing}"
        for name in METHODS & set(SURFACE[type_]):
            assert callable(getattr(module, name)), f"{cid}.{name} is not callable"


def test_the_surface_variants():
    variants = SECTION.get("surface_variants") or {}
    for type_, by_name in variants.items():
        for variant, spec in by_name.items():
            cls_name = (B.CLASS_VARIANTS.get(type_) or {}).get(variant)
            assert cls_name, f"coverage.json names the {type_} variant {variant!r}, which " \
                             "blocks.CLASS_VARIANTS does not implement"
            instances = _instances(type_, cls_name)
            assert instances, f"no case builds {cls_name}"
            for cid, module in instances:
                missing = [a for a in spec["attributes"] if not hasattr(module, a)]
                assert not missing, f"{cid} ({cls_name}) lacks {missing}"
    # whatever the table says, the ordinal head reads as a head and names its thresholds
    for cid, module in _instances("head", "CumulativeHead"):
        assert all(hasattr(module, a) for a in ("linear", "act", "score", "first", "gaps")), cid


def test_the_optional_surface():
    optional = dict(SECTION.get("surface_optional") or {})
    optional.setdefault("feature_tokenizer", {"attributes": ["embeddings"]})
    for type_, spec in optional.items():
        for cid, module in _instances(type_):
            missing = [a for a in spec["attributes"] if not hasattr(module, a)]
            assert not missing, f"{cid} lacks {missing}"


def _params(fn):
    return [(p.name, p.default) for p in inspect.signature(fn).parameters.values()]


EMPTY = inspect.Parameter.empty


def test_the_method_signatures():
    """R-B3: `coefficients` per conv, `AttentionPool.weights`; R-B2: positional forwards."""
    for cls in (B.GCNConv, B.SAGEConv, B.GINConv):
        assert _params(cls.coefficients) == [("self", EMPTY), ("edge_index", EMPTY),
                                             ("n", EMPTY), ("edge_weight", None)], cls.__name__
    assert _params(B.GATConv.coefficients) == [("self", EMPTY), ("h", EMPTY),
                                               ("edge_index", EMPTY), ("edge_weight", None)]
    assert _params(B.AttentionPool.weights) == [("self", EMPTY), ("x", EMPTY)]
    for cls in (B.GCNConv, B.SAGEConv, B.GATConv, B.GINConv):
        assert _params(cls.forward) == [("self", EMPTY), ("x", EMPTY), ("edge_index", EMPTY),
                                        ("edge_weight", None)], cls.__name__
    assert _params(B.EdgeDropout.forward) == [("self", EMPTY), ("edge_index", EMPTY),
                                              ("edge_weight", None)]
    assert _params(B.GraphPool.forward) == [("self", EMPTY), ("x", EMPTY),
                                            ("graph_index", EMPTY), ("num_graphs", EMPTY)]


def _check_values(type_, module):
    """The meaning behind the names (backend-contract §2.1)."""
    checks = {
        "rnn": lambda m: isinstance(m.rnn, nn.RNNBase) and m.ret in ("last", "all", "mean", "max"),
        "attention_pool": lambda m: isinstance(m.score, nn.Linear) and isinstance(
            m.query, nn.Linear) and m.query.bias is None and m.query.out_features == 1,
        "gcn_conv": lambda m: isinstance(m.lin, nn.Linear) and m.normalize in ("sym", "rw")
        and isinstance(m.self_loops, bool),
        "sage_conv": lambda m: isinstance(m.lin_neigh, nn.Linear) and m.aggregator in (
            "mean", "max", "sum") and isinstance(m.normalize, bool)
        and (m.lin_root is None or isinstance(m.lin_root, nn.Linear)),
        "gat_conv": lambda m: isinstance(m.lin, nn.Linear) and tuple(m.att_src.shape) == (
            m.heads, m.lin.out_features // m.heads) and isinstance(m.concat, bool)
        and isinstance(m.negative_slope, float),
        "gin_conv": lambda m: isinstance(m.mlp, nn.Sequential) and m.eps.numel() == 1,
        "graph_pool": lambda m: m.readout in ("mean", "max", "sum"),
        "edge_dropout": lambda m: isinstance(m.p, float) and 0.0 <= m.p < 1.0,
        "head": lambda m: isinstance(m.linear, nn.Linear),
        "reparameterize": lambda m: isinstance(m.z, int),
    }
    key = "rnn" if type_ in ("gru", "lstm") else type_
    return checks[key](module) if key in checks else True


@pytest.mark.parametrize("type_", sorted(t for t in SURFACE if t != "custom"))
def test_the_surface_values_mean_what_the_contract_says(type_):
    for cid, module in _instances(type_):
        assert _check_values(type_, module), cid


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s", "-p", "no:cacheprovider"]))
