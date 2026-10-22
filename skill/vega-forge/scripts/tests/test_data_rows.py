#!/usr/bin/env python3
"""
test_data_rows.py — the row half of the DataSource (`templates/data.py` + `templates/frames.py`),
proved on the rendered package (build plan B08; backend design §9.1, §9.2, §9.5).

    python3 test_data_rows.py [-v] [-k <substring>]
    python3 -m pytest test_data_rows.py

A row spec renders without `windows.py` or `graphs.py` (they are emitted for their shapes only),
so this package holds `domain.py`, `frames.py` and `data.py` — plus c849f8b's generation-v2.2
`data.py`, rendered beside it as `data_v22.py`, the reference the streamed path is held to.
What is proved:

  * the streamed row path (`iter_batches`, the golden path the Trainer's `StreamedRows` reads,
    backend design §10.2) gives, batch for batch, the tensors v2.2's `iter_batches` gave —
    plain, and under the buffered seeded shuffle across several buffer flushes; `make_loader`'s
    seeded order is v2.2's too;
  * a row's id is its raw-file position (`fitted.json.__split_raw__`), in the part's order;
    the part's digest is checked and carried;
  * the label decoders of §9.2 per effective kind (indices, names, delimited multilabel k-hot,
    ordinal, regression, anomaly by `positive`, none for unsupervised), refusing an unknown
    class and a bad index by name; a nominal id outside its cardinality is the unseen id;
  * the contract checks: a changed part, a manifest the contract was not built from, a missing
    feature column, an `input_dim` the columns do not add up to, raw positions that do not
    match the part;
  * the seal: the test part is refused before anything is opened, the sealed reader opens only
    the manifest it was unsealed for, `load_sealed_part` reads the test rows by raw position;
  * the design matrix `[x_num | x_cat]`, and C8, C14 (the adapter alone at its module's top
    level) and C11–C13 on the rendered files.

Exit 0 when every test passes.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import data_fixtures as fx

NS = fx.ingestion("rows_pkg", "row", only=("frames.py", "data.py"))
D = NS.domain


def _v22_module():
    """c849f8b's v2.2 data.py (read from git: `render_templates.baseline_source`), rendered
    into the same package (it imports only names the v2.3 domain still defines: TensorPart,
    ResidentPart, LabelSet, FeatureContract, ...)."""
    import importlib
    text = fx.rt.baseline_source("templates/data.py")
    for key, value in {"PACKAGE": NS.package, "MODULE_INGESTION": "ingestion",
                       "PORT_DATASOURCE": "DataSource"}.items():
        text = text.replace("{{" + key + "}}", value)
    text = re.sub(r"\{\{[A-Z_]+\}\}", "(reference)", text)
    # v2.2 read the spec's sections by key; the v2.3 NetSpec reads them as values, and its
    # canonical mapping (`to_dict`) is what the reference reads them out of
    text = text.replace(".section(", ".to_dict().get(")
    (NS.root / "ingestion" / "data_v22.py").write_text(text, encoding="utf-8")
    importlib.invalidate_caches()
    return importlib.import_module(f"{NS.package}.ingestion.data_v22")


V22 = _v22_module()
N = 260


def raw_frame(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    zip_ids = pd.array(rng.integers(0, 10, size=n), dtype="Int64")
    zip_ids[::17] = pd.NA                                    # a missing nominal reads as unseen
    zip_ids[5::23] = 42                                      # outside the cardinality: unseen
    names = np.array(["red", "green", "blue"])
    return pd.DataFrame({
        "x0": rng.normal(size=n), "x1": rng.uniform(size=n), "x2": np.arange(n, dtype=float),
        "zip": zip_ids, "late": rng.integers(0, 2, size=n),
        "colour": names[rng.integers(0, 3, size=n)],
        "tags": ["|".join(t for t, on in zip("abc", rng.uniform(size=3) > 0.5) if on)
                 for _ in range(n)],
        "grade": np.array(["lo", "mid", "hi"])[rng.integers(0, 3, size=n)],
        "amount": rng.normal(size=n) * 3, "fraud": np.where(rng.uniform(size=n) > 0.9,
                                                            "yes", "no")})


def parts_of(n: int = N) -> dict:
    perm = np.random.default_rng(12).permutation(n)
    return {"train": perm[:180], "valid": perm[180:220], "test": perm[220:]}


def fixture(task=None, frame=None, parts=None):
    frame = raw_frame() if frame is None else frame
    parts = parts_of() if parts is None else parts
    root = Path(tempfile.mkdtemp(prefix="rows-fx-"))
    man = fx.write_shaped(root, {k: frame.iloc[v] for k, v in parts.items()}, parts)
    spec = fx.rt.reference_spec("row")
    spec["task"] = task or {"kind": "binary", "label": "late", "classes": [0, 1], "positive": 1}
    spec["data_contract"]["features"] = [fx.numeric("x0"), fx.numeric("x1"), fx.numeric("x2"),
                                         fx.nominal("zip", 10)]
    spec["data_contract"]["input_dim"] = 4
    spec = D.NetSpec.from_dict(fx.spec_on(spec, man))
    return NS.data.ShapedParts(), spec, root, (frame, parts)


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k])
                                        for k in a)


# --------------------------------------------------------------------------- v2.2 parity

def test_stream_matches_v22_batch_for_batch():
    sp, spec, _, _ = fixture()
    plain = list(sp.iter_batches(spec, "valid", 16))
    ref = list(V22.iter_batches(spec, "valid", batch_size=16))
    assert len(plain) == len(ref) == 3 and all(same(a, b) for a, b in zip(plain, ref))
    for buffer in (None, 40):
        got = list(sp.iter_batches(spec, "train", 16,
                                   plan=D.BatchPlan(shuffle=True, seed=5, shuffle_buffer=buffer)))
        want = list(V22.iter_batches(spec, "train", batch_size=16, shuffle=True, seed=5,
                                     shuffle_buffer=buffer))
        assert len(got) == len(want) and all(same(a, b) for a, b in zip(got, want)), buffer
    got = list(sp.iter_batches(spec, "train", plan=D.BatchPlan(shuffle=True)))
    want = list(V22.iter_batches(spec, "train", shuffle=True))
    assert all(same(a, b) for a, b in zip(got, want)), "the default batch size or seed moved"


def test_held_part_and_loader_match_v22():
    sp, spec, _, _ = fixture()
    mine, ref = sp.load_part(spec, "train"), V22.load_part(spec, "train")
    for name in ("x_num", "x_cat", "y"):
        assert torch.equal(getattr(mine, name), getattr(ref, name)), name
    got = list(sp.make_loader(mine, spec, shuffle=True, seed=3, batch_size=50))
    want = list(V22.make_loader(ref, spec, shuffle=True, seed=3, batch_size=50))
    assert all(same(a, b) for a, b in zip(got, want))
    resident = sp.load_resident(spec, "train", "cpu")
    assert resident.device == "cpu" and torch.equal(resident.x_num, mine.x_num)
    assert resident.digest == mine.digest


# --------------------------------------------------------------------------- identity

def test_row_ids_are_raw_positions_and_the_digest_is_carried():
    sp, spec, root, (frame, parts) = fixture()
    valid = sp.load_part(spec, "valid")
    assert np.array_equal(valid.row_ids, parts["valid"]) and valid.row_id_kind.value == \
        "raw_position"
    assert np.array_equal(valid.x_num[:, 2].numpy(), frame.x2.to_numpy()[parts["valid"]])
    files = json.loads((root / "manifest.json").read_text())["outputs"]["files"]
    assert valid.digest == files["valid"]["digest"]
    assert valid.columns == ("x0", "x1", "x2", "zip")


def test_nominal_ids_outside_the_cardinality_are_the_unseen_id():
    sp, spec, _, (frame, parts) = fixture()
    train = sp.load_part(spec, "train")
    raw = frame.zip.iloc[parts["train"]]
    unseen = (raw.isna() | (raw >= 10)).to_numpy()
    assert unseen.any() and (train.x_cat[torch.from_numpy(unseen), 0] == 9).all()
    assert torch.equal(train.x_cat[torch.from_numpy(~unseen), 0],
                       torch.as_tensor(raw[~unseen].to_numpy(dtype=np.int64)))


def test_label_decoders_per_effective_kind():
    frame, parts = raw_frame(), parts_of()
    rows = parts["valid"]

    def y_of(task):
        sp, spec, _, _ = fixture(task, frame, parts)
        return sp.load_part(spec, "valid").y

    y = y_of({"kind": "multiclass", "label": "colour", "classes": ["red", "green", "blue"]})
    assert y.dtype == torch.long and y.tolist() == [["red", "green", "blue"].index(c)
                                                    for c in frame.colour.iloc[rows]]
    y = y_of({"kind": "multilabel", "label": "tags", "classes": ["a", "b", "c"],
              "label_format": {"kind": "delimited", "sep": "|"}})
    want = [[float(t in cell.split("|")) for t in "abc"] for cell in frame.tags.iloc[rows]]
    assert y.dtype == torch.float32 and y.tolist() == want
    y = y_of({"kind": "ordinal", "label": "grade", "classes": ["lo", "mid", "hi"]})
    assert y.tolist() == [["lo", "mid", "hi"].index(g) for g in frame.grade.iloc[rows]]
    y = y_of({"kind": "regression", "label": "amount"})
    assert y.dtype == torch.float32 and np.allclose(y.numpy(), frame.amount.iloc[rows])
    y = y_of({"kind": "anomaly", "label": "fraud", "classes": ["no", "yes"], "positive": "yes"})
    assert y.dtype == torch.float32 and y.tolist() == [float(v == "yes")
                                                       for v in frame.fraud.iloc[rows]]
    y = y_of({"kind": "anomaly", "label": "late", "positive": 1})
    assert y.tolist() == frame.late.iloc[rows].astype(float).tolist()
    assert y_of({"kind": "anomaly", "label": None}) is None
    assert y_of({"kind": "unsupervised", "label": "late"}) is None
    y = y_of({"kind": "binary", "label": "late", "classes": [0, 1], "positive": 1})
    assert y.dtype == torch.long and y.tolist() == frame.late.iloc[rows].tolist()


def test_label_refusals_name_the_offender():
    frame, parts = raw_frame(), parts_of()
    sp, spec, _, _ = fixture({"kind": "multiclass", "label": "colour",
                              "classes": ["red", "green"]}, frame, parts)
    fx.raises(lambda: sp.load_part(spec, "train"), "ValueError", "blue")
    bad = frame.copy()
    bad["grade_i"] = 1
    bad.loc[parts["train"][4], "grade_i"] = 7
    sp, spec, _, _ = fixture({"kind": "ordinal", "label": "grade_i", "classes": [0, 1, 2]},
                             bad, parts)
    fx.raises(lambda: sp.load_part(spec, "train"), "ValueError", "row 4", "class index")


# --------------------------------------------------------------------------- the checks

def test_contract_and_digest_checks():
    sp, spec, root, (frame, parts) = fixture()
    changed = D.NetSpec.from_dict({**spec.to_dict(),
                                   "input": {**spec.to_dict()["input"],
                                             "manifest_digest": "sha256:" + "0" * 64}})
    fx.raises(lambda: sp.load_part(changed, "train"), "ValueError", "FeatureContract")
    narrow = spec.to_dict()
    narrow["data_contract"]["features"].append(fx.numeric("absent"))
    narrow["data_contract"]["input_dim"] = 5
    fx.raises(lambda: sp.load_part(D.NetSpec.from_dict(narrow), "valid"), "ValueError",
              "absent", "valid")
    wrong = spec.to_dict()
    wrong["data_contract"]["input_dim"] = 7
    fx.raises(lambda: sp.load_part(D.NetSpec.from_dict(wrong), "valid"), "ValueError",
              "input_dim 7")
    fitted = json.loads((root / "fitted.json").read_text())
    fitted["__split_raw__"]["valid"] = fitted["__split_raw__"]["valid"][:-1]
    (root / "fitted.json").write_text(json.dumps(fitted))
    fx.raises(lambda: sp.load_part(spec, "valid"), "ValueError", "not the same part")
    del fitted["__split_raw__"]["valid"]
    (root / "fitted.json").write_text(json.dumps(fitted))
    fx.raises(lambda: sp.load_part(spec, "valid"), "ValueError", "__split_raw__.valid")
    part = Path(spec.inputs.part("train"))
    frame.iloc[parts["train"][:-1]].to_parquet(part, index=False)
    fx.raises(lambda: sp.load_part(spec, "train"), "ValueError", "changed since the shaper")
    fx.raises(lambda: list(sp.iter_batches(spec, "train")), "ValueError", "changed since")


def test_batch_count_is_the_streams_without_a_pass():
    """The T5 round 2 improve pass: the Trainer learned a stream's batch count by running a
    whole extra epoch into tensors. `batch_count` gives the count `iter_batches` yields, read
    from a one-column pass over the chunks, for every plan; the test part is refused."""
    sp, spec, root, (frame, parts) = fixture()
    for bs in (7, 32, 300):
        for plan in (D.BatchPlan(), D.BatchPlan(shuffle=True, seed=3),
                     D.BatchPlan(shuffle=True, seed=3, shuffle_buffer=50)):
            want = sum(1 for _ in sp.iter_batches(spec, "train", batch_size=bs, plan=plan))
            assert sp.batch_count(spec, "train", batch_size=bs, plan=plan) == want, (bs, plan)
    fx.raises(lambda: sp.batch_count(spec, "test"), "SealedPartError", "batch_count")


def test_seal_for_row_samples():
    sp, spec, root, (frame, parts) = fixture()
    dead = spec.to_dict()
    dead["input"]["manifest"] = "/nonexistent/manifest.json"
    dead = D.NetSpec.from_dict(dead)
    for call in (lambda: sp.load_part(dead, "test"), lambda: sp.iter_batches(dead, "test"),
                 lambda: sp.load_resident(dead, "test", "cpu")):
        fx.raises(call, "SealedPartError", "test part")
    fx.raises(lambda: NS.frames.PartFrames().open(spec, "test"), "SealedPartError",
              "load_sealed_part")
    fx.raises(lambda: list(NS.frames.PartFrames().chunks(spec, "test", ["x0"], 8)),
              "SealedPartError", "never streamed")
    seal = fx.seal(root, "tok")
    seal.write_text(json.dumps({"token": "tok", "run": "R3", "exposures": 1}), encoding="utf-8")
    test = sp.load_sealed_part(spec, "tok", str(seal))
    assert np.array_equal(test.row_ids, parts["test"]) and len(test) == N - 220
    # the seal is read once, by the door that checked the token, and travels with the part:
    # the evaluation stamps what admitted the read (T5 round 2: it re-read the file)
    record = json.loads(Path(seal).read_text(encoding="utf-8"))
    assert test.seal == D.SealReading.of(record) == D.SealReading("R3", 1)
    assert sp.load_part(spec, "valid").seal is None
    _, other, _, _ = fixture()
    unsealed = NS.frames.unseal(spec, "tok", str(seal))
    assert len(unsealed.open(spec, "test").frame) == N - 220
    fx.raises(lambda: unsealed.open(other, "test"), "SealedPartError", "load_sealed_part")
    fx.raises(lambda: NS.frames.unseal(spec, "nope", str(seal)), "SealedPartError", "token")
    # final rehearsal round 2 (architect, 3 of 3 audits): the seal was an overridable hook on
    # the sealed reader, and the unsealed reader its subclass. They are siblings now: the sealed
    # reader has no path to the test part, no seal and no door, so a subclass gains nothing
    assert not issubclass(NS.frames.UnsealedFrames, NS.frames.PartFrames)
    assert not {"unseal", "seal", "_opens_test"} & set(dir(NS.frames.PartFrames))

    class Rogue(NS.frames.PartFrames):
        def _opens_test(self, spec):
            return True
    fx.raises(lambda: Rogue().open(spec, "test"), "SealedPartError", "load_sealed_part")
    # T5 round 2's net check: the constructor was a second door (PartFrames(unsealed_for=...)
    # opened the test part with no token); it takes no unsealed state now
    fx.raises(lambda: NS.frames.PartFrames(unsealed_for=str(root)), "TypeError", "unsealed_for")


def test_design_matrix_rows():
    sp, spec, _, _ = fixture()
    valid = sp.load_part(spec, "valid")
    design = sp.design_matrix(spec, valid)
    want = torch.cat([valid.x_num, valid.x_cat.float()], dim=1).numpy()
    assert design.x.dtype == np.float32 and np.array_equal(design.x, want)
    assert np.array_equal(design.y, valid.y.numpy()) and np.array_equal(design.row_ids,
                                                                        valid.row_ids)
    fx.raises(lambda: sp.design_matrix(spec, valid, "graph-pooled"), "ValueError", "graph part")


def test_the_data_source_carries_its_shape_and_kind_methods_only():
    """The T5 round 3 audit (finding 3): a row package's DataSource carries no graph exposure,
    no forecast scale and — off the regression family — no train targets; each is the port's
    method in the packages that use it (SOLID-I), and the port lists it there only."""
    emit = fx.rt.load_emit()
    scoped = {"exposure", "target_scale", "train_targets"}
    assert not scoped & set(emit.adapter_surface("DataSource", "row", task="multiclass"))
    assert scoped & set(emit.adapter_surface("DataSource", "row", task="regression")) == \
        {"train_targets"}
    assert {"exposure"} <= set(emit.adapter_surface("DataSource", "graph", task="multiclass"))
    assert {"target_scale", "train_targets"} <= set(emit.adapter_surface("DataSource", "window",
                                                                          task="forecast"))
    port = [emit._declared(m).name for m in emit._role_lines("DataSource", "row", "binary")]
    assert not scoped & set(port), port
    assert all("SamplePart" not in m for m in emit._role_lines("DataSource", "row", "binary"))


def test_template_discipline():
    """C8 (the seal on the adapter class), C14 (ShapedParts alone at data.py's top level, with
    helpers, imports and constants) and C11–C13 on the rendered row files."""
    import ast

    import conformance
    files = [NS.root / "ingestion" / n for n in ("data.py", "frames.py")]
    _, findings = conformance.size_only([str(f) for f in files])
    assert not findings, [f.message() for f in findings]
    pkg, err = conformance.Package.read(NS.root, NS.blueprint_map, NS.pipeline)
    assert err is None, err
    assert all(m.level.name == "OK" for m in conformance.c8_datasource_seals_the_test(pkg))
    tree = ast.parse(files[0].read_text(encoding="utf-8"))
    adapter = conformance.adapter_class(tree, "DataSource", "DataSource")
    assert adapter is not None and adapter.name == "ShapedParts"
    stray = [f for node in tree.body
             for f in conformance._top_level_findings("ingestion/data.py", node, adapter,
                                                      "DataSource")]
    assert not stray, [f.message() for f in stray]
    text = files[0].read_text(encoding="utf-8")
    assert 'raise SealedPartError(_SEALED.format(door="load_part"))' in text, \
        "the C8 negative's anchor line moved"


def test_no_dataset_literal_in_the_ingestion_templates():
    """A shared template names no dataset (the patterns test_domain_plans.py holds domain.py
    to), over all four ingestion templates."""
    for name in fx.INGESTION:
        lines = (fx.TEMPLATES / name).read_text(encoding="utf-8").splitlines()
        for pattern in (r"\bIDS\b", r"\bCIDS\b", r"(?i)\bcic-?ids", r"(?i)\bflows?\b",
                        r'"Label"', r"(?i)\btraffic\b", r"(?i)\bintrusion"):
            hits = [line for line in lines if re.search(pattern, line)]
            assert not hits, (name, pattern, hits)


def main(argv) -> int:
    return fx.main_for(globals(), argv, "row DataSource")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
