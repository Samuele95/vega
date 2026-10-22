#!/usr/bin/env python3
"""
test_shape_layer_digest.py — the `shape` layer carries `manifest_digest`, the sha256 of the
manifest.json FILE as the executor wrote it (shape-contract §4), because it is the one digest
/blueprint (`pipeline.json.input.shape_manifest_digest`), /vega-forge's compiler and the emitted
FeatureContract compare.

    python3 -m pytest test_shape_layer_digest.py [-q]

What is proved here is what the apply script writes and accepts: with `from_run` it computes the
digest from the file (never from the spliced copy, which it re-serialises and may elide), and a
digest the author typed is kept only when it is that file's. `run_negatives_layer.py` seeds the
documents it must refuse.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _layer_fixture as fx  # noqa: E402
import apply_shape_layer as asl  # noqa: E402


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("shape-layer"))


def test_from_run_writes_the_manifest_files_digest(chain, tmp_path):
    model, run_dir = chain
    data = fx.write_doc(tmp_path / "layer.json", fx.judgement(from_run=str(run_dir)))
    rc, said = fx.apply(model, data, tmp_path / "out.html")
    assert rc == 0, said
    layer = fx.layer_data(tmp_path / "out.html")
    written = fx.file_digest(run_dir / "manifest.json")
    assert layer["manifest_digest"] == written
    # the file's digest, not the spliced copy's: the layer holds the manifest re-serialised
    copy_ = "sha256:" + hashlib.sha256(json.dumps(layer["manifest"]).encode("utf-8")).hexdigest()
    assert copy_ != written


def test_a_typed_digest_is_kept_when_it_is_the_files(chain, tmp_path):
    model, run_dir = chain
    written = fx.file_digest(run_dir / "manifest.json")
    data = fx.write_doc(tmp_path / "layer.json",
                        fx.judgement(from_run=str(run_dir), manifest_digest=written))
    rc, said = fx.apply(model, data, tmp_path / "out.html")
    assert rc == 0, said
    assert fx.layer_data(tmp_path / "out.html")["manifest_digest"] == written


def test_an_inline_document_publishes_the_digest_it_carries(chain, tmp_path):
    model, run_dir = chain
    written = fx.file_digest(run_dir / "manifest.json")
    data = fx.write_doc(tmp_path / "layer.json", fx.inline(run_dir, manifest_digest=written))
    rc, said = fx.apply(model, data, tmp_path / "out.html")
    assert rc == 0, said
    assert fx.layer_data(tmp_path / "out.html")["manifest_digest"] == written


def test_validate_names_manifest_digest_as_required(chain):
    _, run_dir = chain
    errs = asl.validate(fx.inline(run_dir), lambda _line: None)
    assert any("manifest_digest" in e for e in errs), errs
    errs = asl.validate(fx.inline(run_dir, manifest_digest="sha256:abc"), lambda _line: None)
    assert any("sha256:<64 hex>" in e for e in errs), errs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
