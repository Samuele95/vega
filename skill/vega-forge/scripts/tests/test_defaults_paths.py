#!/usr/bin/env python3
"""
test_defaults_paths.py — a `defaults_to_test` path is a rung's delta: the trainer's grammar,
on a path the emitted code reads (netspec-contract §4; F40).

    python3 -m pytest test_defaults_paths.py [-q]

The release4 rehearsal's forge wrote `regularization[dropout]` and `regularization[weight_decay]`;
apply_architecture_layer.py published them, and plan_ladder.py, which ranks dropout and weight
decay first among the four unattended default-test rungs, dropped both with a WARN: the two
hypotheses the forge handed on were never trained. What is proved:

  grammar   keys, indices, `key[field=value]` selectors and the block-id shorthands
            (`blocks/B1/…`, `blocks[B1]/…`) resolve; a bare bracket on any other list
            (`regularization[dropout]`) does not, and the refusal names the selector form;
  read      under `regularization` only a path the emitted code reads is admitted
            (`regularization[kind=dropout]/value`, `regularization/0/value`); the declaration's
            entry without `/value`, and a live path a declaration shadows
            (`blocks[id=B3]/params/p` under a dropout declaration), are refused;
  the apply apply_architecture_layer.validate refuses a layer carrying such an entry, by path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for _p in (str(HERE), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import defaults_paths as D  # noqa: E402

SPEC = {
    "family": {"name": "mlp-plain", "source": "blueprint:posture"},
    "blocks": [{"id": "B1", "type": "dense", "params": {"units": 32}},
               {"id": "B2", "type": "batchnorm", "params": {}},
               {"id": "B3", "type": "dropout", "params": {}},
               {"id": "B4", "type": "head", "params": {"out": 6}}],
    "regularization": [{"kind": "weight_decay", "value": 1e-4},
                       {"kind": "dropout", "value": 0.1},
                       {"kind": "grad_clip", "value": 1.0}],
    "loss": {"name": "cross_entropy", "params": {}},
    "optimizer": {"name": "adamw", "params": {"lr": 1e-3}},
    "schedule": {"name": "cosine", "params": {"warmup_epochs": 3}},
    "training": {"batch_size": 128, "max_epochs": 200},
}

GOOD = ["family", "loss", "optimizer", "schedule", "optimizer/params/lr", "training/batch_size",
        "training/max_epochs", "blocks[id=B1]/params/units", "blocks/B1/params/units",
        "blocks[B1]/params/units", "blocks[B2]/type", "blocks/3/params/out",
        "regularization[kind=dropout]/value", "regularization[kind=weight_decay]/value",
        "regularization[kind=grad_clip]/value", "regularization/0/value"]

BAD_GRAMMAR = ["regularization[dropout]", "regularization[weight_decay]", "regularization[grad_clip]",
               "blocks[B9]/params/units", "training/patience", "optimizer/params/momentum",
               "blocks/7/params", "", "loss//params"]

BAD_READ = ["regularization[kind=dropout]", "regularization", "regularization/1"]


@pytest.mark.parametrize("path", GOOD)
def test_a_path_the_trainer_moves_is_admitted(path):
    assert D.problem(SPEC, path) is None


@pytest.mark.parametrize("path", BAD_GRAMMAR)
def test_a_path_that_addresses_nothing_is_refused(path):
    why = D.problem(SPEC, path)
    assert why and "does not resolve" in why


def test_the_refusal_names_the_selector_form():
    assert "regularization[kind=dropout]/value" in D.problem(SPEC, "regularization[dropout]")


@pytest.mark.parametrize("path", BAD_READ)
def test_a_regularization_path_the_code_does_not_read_is_refused(path):
    why = D.problem(SPEC, path)
    assert why and ("shadows" in why or "not a regularization path the emitted code reads" in why)


def test_a_live_path_a_declaration_shadows_is_refused():
    import copy
    spec = copy.deepcopy(SPEC)
    spec["blocks"][2]["params"]["p"] = 0.1          # the live copy the dropout declaration covers
    spec["optimizer"]["params"]["weight_decay"] = 1e-4
    for path in ("blocks[id=B3]/params/p", "blocks/B3/params/p", "optimizer/params/weight_decay"):
        why = D.problem(spec, path)
        assert why and "shadows" in why, (path, why)


def test_a_defaulted_sample_point_stands_while_the_spec_has_a_sample():
    import copy
    spec = copy.deepcopy(SPEC)
    spec["data_contract"] = {"sample": {"shape": "window", "length": 8}}
    assert D.SAMPLE_POINTS >= {"context", "autoregressive", "series_by", "batching"}
    assert "protocol" not in D.SAMPLE_POINTS                    # never tested (netspec-contract §3)
    for path in ("data_contract/sample/series_by", "sample/context"):
        assert D.problem(spec, path) is None, path
    assert D.problem(spec, "data_contract/sample/protocol")      # not a tested point, and absent
    assert D.problem(SPEC, "data_contract/sample/series_by")     # no sample at all


def test_the_apply_refuses_the_layer_by_path():
    import apply_architecture_layer as A
    doc = {"defaults_to_test": [{"path": "regularization[dropout]", "observation": "x"},
                                {"path": "blocks[B1]/params/units", "observation": "y"}],
           "netspec": SPEC}
    errs = []
    for entry in doc["defaults_to_test"]:
        why = D.problem(doc["netspec"], entry["path"])
        if why:
            errs.append(why)
    assert len(errs) == 1 and "regularization[dropout]" in errs[0]
    # the same refusal is in validate(): it is wired, not only importable
    assert "defaults_paths.problem" in Path(A.__file__).read_text(encoding="utf-8")
