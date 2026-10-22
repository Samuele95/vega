#!/usr/bin/env python3
"""
test_plan_ladder.py — the ladder's floors are the floors of the sample's shape (R-A12, D1 §12.3),
and its other rules plan what they say.

    python3 -m pytest test_plan_ladder.py [-q]

The specs are real: B15's spike fixtures (vega-forge/scripts/tests/spike), shaped by the real
`shape.py` and composed so the compiler admits them — a `gru` forecast over windows, a `gcn`
at node level and a `gin` at graph level — plus variants of the window spec for the other
recurrent families. What it proves:

1. **The floor per shape (E17).** On a window the floor is `linear` with `flatten` inserted
   before the head; at graph level `linear` with `graph_pool(mean)` before the head; at node
   level it is structure-blind — the task's head over each node's own features, no block that
   reads an edge. Every floor keeps the spec's embedding and concat and R0's head, cites the
   linear composition, compiles through the forge's compiler core, and — built from a freshly
   emitted package — its forward pass answers one row per sample: `(B, H)` for the windows,
   `(G, 1)` for the graphs, `(targets, 1)` for the nodes. The reconstruction and anomaly
   floor is `pca-floor` through `dense(z, none)` before the head (D1 §6): z the spec's own
   bottleneck (the narrowest dense, or a VAE's reparameterize z), else the geometry layer's
   dims_95, and no floor without either; it compiles on a row autoencoder.
2. **R1b for the recurrent families only.** `gru`, `lstm`, `bi-lstm`, `bi-gru`,
   `lstm-attention`, `gru-attention` get the recurrent floor: the gated block becomes `rnn` with
   the same `hidden` and `num_layers`, `return: last`, `bidirectional: false`, the
   `attention_pool` (or a pool) is removed, the head is kept, and the result compiles; `rnn`,
   `cnn1d`, `transformer-seq`, the graph and the row families get none, and no WARN.
3. **Ids are the rules'.** R0 the baseline, R1 the floor, R1b the recurrent floor, R2… the
   rest; a floor that is dropped (a spec whose family is already `linear`) leaves R1 unused
   rather than lending it to another rung.
4. **The search spaces of E21** resolve their placeholders on these specs (`<rnn id>`,
   `<attention_pool id>`, `<first conv id>`), and an optional dimension whose block the spec
   lacks is `not_applicable`, not a WARN. A block param the spec leaves unstated (a
   forge-composed `lstm` without `num_layers`) is searched from the value the block is built
   with — the forge catalog's default, or the older `layers` — and a trial that moves it
   compiles.
5. **Alternatives**: a different choice replaces the params of the one it replaces (`focal`
   does not inherit `pos_weight`), the same choice merges them, and a choice whose values sit
   at its top level (early_stopping) takes an alternative's params as those values; a
   `data_contract/sample/<point>` default runs the forge catalog's alternative for the point.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
SPIKE = SKILLS / "vega-forge" / "scripts" / "tests" / "spike"
FORGE = SKILLS / "vega-forge" / "scripts"
for _p in (str(SCRIPTS), str(SPIKE), str(FORGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plan_ladder as PL  # noqa: E402
from vega_train_core import Delta, load_catalog  # noqa: E402

CATALOG = load_catalog()[0]
FORGE_CATALOG = PL.forge_catalog()
RECURRENT = ("gru", "lstm", "bi-lstm", "bi-gru", "lstm-attention", "gru-attention")


# --------------------------------------------------------------------------- the fixtures

@pytest.fixture(scope="module")
def spike(tmp_path_factory):
    """The three spike fixtures, made once for the module."""
    import make_spike_fixtures as msf
    work = tmp_path_factory.mktemp("spike")
    return {name: msf.make(name, work, "full")
            for name in ("window", "node-keypair", "graph-level")}


def _spec(fixture) -> dict:
    return json.loads(Path(fixture.spec).read_text(encoding="utf-8"))


def _chain(fixture) -> tuple:
    return (json.loads(Path(fixture.manifest).read_text(encoding="utf-8")),
            json.loads(Path(fixture.pipeline).read_text(encoding="utf-8")))


def _compiles(spec, fixture) -> None:
    """The forge's compiler core over the rung's spec: raises, naming the rule, on a refusal."""
    import make_spike_fixtures as msf
    manifest, pipeline = _chain(fixture)
    assert msf.compile_check(spec, manifest, pipeline) == ()


def _forward(spec, fixture, out_dir):
    """Emit `spec` into `out_dir`, open it by role, build the network and run one forward
    pass over the first valid batch; returns (output shape, batch)."""
    import make_spike_fixtures as msf
    import torch
    from vega_adapters import PackageAdapters
    _, pipeline = _chain(fixture)
    msf.emit_package(out_dir, spec, pipeline, "full")
    adapters = PackageAdapters.load(out_dir / fixture.package, fixture.package)
    ns = adapters.package.domain.NetSpec.from_dict(spec)
    model = adapters.build(ns)
    part = adapters.data.load_part(ns, "valid")
    batch = next(iter(adapters.data.make_loader(part, ns, False)))
    model.eval()
    with torch.no_grad():
        out = model(batch)
    return tuple(out.shape), batch


def _types(spec) -> list:
    return [b["type"] for b in spec["blocks"]]


def _applied(rung, spec) -> dict:
    delta = Delta.of(rung["delta"])
    assert delta.problems(spec, rung["family"]) == []
    return delta.apply(spec)


def _window_variant(spec, family, blocks) -> dict:
    """The window spec re-cast as another recurrent family (for the R1b construction)."""
    out = copy.deepcopy(spec)
    out["family"] = dict(out["family"], name=family)
    out["blocks"] = blocks
    return out


def _block(bid, block_type, **params):
    return {"id": bid, "type": block_type, "params": params, "source": "vega:default",
            "catalog": "defaults/sizing", "rationale": "a test block",
            "alternatives": [{"params": {}, "when": "a test alternative"}]}


# --------------------------------------------------------------------------- 1. the floors

def test_the_window_floor_is_linear_with_flatten_before_the_head(spike, tmp_path):
    fx = spike["window"]
    spec = _spec(fx)
    rung, why = PL.rung_floor(spec, CATALOG)
    assert why is None and rung["id"] == "R1" and rung["source"] == "vega:ladder/floor"
    floor = _applied(rung, spec)
    assert floor["family"]["name"] == "linear"
    assert floor["family"]["catalog"] == "families/linear"
    assert _types(floor) == ["embedding", "concat", "flatten", "head"]
    flatten = floor["blocks"][2]
    assert flatten["source"] == "vega:default" and flatten["catalog"] == "families/linear"
    assert flatten["alternatives"] and flatten["id"] not in {b["id"] for b in spec["blocks"]}
    assert floor["blocks"][-1]["params"] == spec["blocks"][-1]["params"]     # R0's head
    _compiles(floor, fx)
    shape, batch = _forward(floor, fx, tmp_path / "floor")
    horizon = spec["data_contract"]["sample"]["horizon"]
    assert batch["x_num"].dim() == 3                       # (B, L, F'): a window
    assert shape == (batch["x_num"].shape[0], horizon)     # the head read the flattened window


def test_the_graph_level_floor_pools_by_mean_before_the_head(spike, tmp_path):
    fx = spike["graph-level"]
    spec = _spec(fx)
    rung, why = PL.rung_floor(spec, CATALOG)
    assert why is None and rung["id"] == "R1"
    floor = _applied(rung, spec)
    assert floor["family"]["name"] == "linear"
    assert _types(floor) == ["embedding", "concat", "graph_pool", "head"]
    pool = floor["blocks"][2]
    assert pool["params"] == {"readout": "mean"}           # not the gin's own sum
    assert pool["catalog"] == "families/linear" and pool["alternatives"]
    assert not any(t.endswith("_conv") for t in _types(floor))
    _compiles(floor, fx)
    shape, batch = _forward(floor, fx, tmp_path / "floor")
    assert shape == (int(batch["num_graphs"]), 1)          # one row per graph


def test_the_node_level_floor_is_structure_blind(spike, tmp_path):
    fx = spike["node-keypair"]
    spec = _spec(fx)
    rung, why = PL.rung_floor(spec, CATALOG)
    assert why is None and rung["id"] == "R1"
    floor = _applied(rung, spec)
    assert floor["family"]["name"] == "linear"
    assert _types(floor) == ["head"]                       # no conv, no pool, no edge read
    assert "structure-blind" in floor["family"]["rationale"]
    assert "blind to the edges" in rung["hypothesis"]
    _compiles(floor, fx)
    shape, batch = _forward(floor, fx, tmp_path / "floor")
    assert shape == (len(batch["target_index"]), 1)        # one row per target node


def test_no_floor_under_neighbour_sampling(spike):
    """A sampled node-level sample draws a fan-out per convolution; the structure-blind floor
    has none, so the compiler would refuse it — the ladder says why instead."""
    spec = _spec(spike["node-keypair"])
    spec["data_contract"]["sample"].update(batching="sampled", neighbours=[5, 5])
    rung, why = PL.rung_floor(spec, CATALOG)
    assert rung is None and "batching sampled" in why and "not planned" in why


def test_the_floor_inserts_come_from_the_catalog():
    inserts = CATALOG["ladder_rules"]["floor"]["inserts"]
    assert [b["type"] for b in inserts["window"]] == ["flatten"]
    assert inserts["graph/graph"] == [{"type": "graph_pool", "params": {"readout": "mean"}}]
    assert inserts["row"] == [] and inserts["graph/node"] == []


def test_a_spec_that_already_is_the_floor_gets_no_floor(spike):
    spec = _spec(spike["window"])
    spec["family"] = dict(spec["family"], name="linear")
    rung, why = PL.rung_floor(spec, CATALOG)
    assert rung is None and "already linear" in why


def _choice(row) -> dict:
    return {"source": "vega:default", "catalog": row, "rationale": f"the {row} row",
            "alternatives": [{"when": "an ablation says so"}]}


def _row_block(bid, block_type, **params):
    structural = block_type in ("head", "concat")
    prov = ({"source": "vega:default", "catalog": f"blocks/{block_type}",
             "rationale": "structural", "alternatives": []} if structural
            else _choice(f"blocks/{block_type}"))
    return {"id": bid, "type": block_type, "params": params, **prov}


def _row_autoencoder(blocks, family="autoencoder") -> tuple:
    """(spec, manifest, pipeline): an anomaly autoencoder on rows — three numeric columns and
    a nominal `zip` of 5 categories, so the reconstruction is 8 wide — as plain values the
    forge's compiler core reads (no model HTML: every source resolves without one)."""
    feats = [{"name": n, "dtype": "float32", "role": "numeric"} for n in ("a", "b", "c")]
    feats.append({"name": "zip", "dtype": "int64", "role": "nominal", "cardinality": 5,
                  "unseen_category_id": 4})
    primary = "anomaly_score_mean"
    spec = {
        "schema": "vega/netspec@1", "seed": 7,
        "input": {"model": "demo.shaped.html", "manifest": "shaped/manifest.json"},
        "task": {"kind": "anomaly", "label": None, "source": "user:turn-1"},
        "data_contract": {"source": "shape:manifest/output_schema",
                          "parts": {"train": "shaped/demo.train.parquet",
                                    "valid": "shaped/demo.valid.parquet", "test": "sealed"},
                          "features": feats, "input_dim": 4, "sample": {"shape": "row"}},
        "family": {"name": family, **_choice(f"families/{family}")},
        "blocks": blocks,
        "loss": {"name": "reconstruction", "params": {}, **_choice("tasks/anomaly")},
        "metrics": {"primary": primary, "secondary": [],
                    "selection": {"monitor": f"valid.{primary}"}, **_choice("tasks/anomaly")},
        "optimizer": {"name": "adamw", "params": {"lr": 0.001}, **_choice("defaults/optimizer")},
        "schedule": {"name": "cosine", "params": {}, **_choice("defaults/schedule")},
        "regularization": [],
        "early_stopping": {"monitor": f"valid.{primary}", "patience": 5,
                           **_choice("defaults/early_stopping")},
        "init": {"scheme": "kaiming_uniform", **_choice("defaults/init")},
        "training": {"batch_size": 32, "max_epochs": 20, "budget": {"wall_clock_min": 5},
                     **_choice(["defaults/batch_size", "defaults/max_epochs"])},
        "backend": {"name": "pytorch", "hints": {}},
        "outputs": {"run_dir": "vega-runs/demo/", "package": "demo_pipeline"}}
    manifest = {"output_schema": {f: {"dtype": "float64", "role": "feature"}
                                  for f in ("a", "b", "c", "zip")},
                "outputs": {"files": {p: {"path": f"shaped/demo.{p}.parquet", "rows": n}
                                      for p, n in (("train", 400), ("valid", 100),
                                                   ("test", 100))}},
                "steps": [{"id": "S1", "op": "split",
                           "parameters_fitted": {"kind": "random", "by": None}},
                          {"id": "S2", "op": "encode", "parameters_fitted": {
                              "fitted": {"zip": {"unseen_category_id": 4}}}}]}
    pipeline = {"system": {"package": "demo_pipeline"}, "posture": {"family": "neural"},
                "serving": {"mode": "batch"},
                "feature_contract": {"input_dim": 4, "label": None},
                "ports": [{"name": r, "role": r} for r in
                          ("DataSource", "Model", "Trainer", "ModelStore", "Evaluator")]}
    return spec, manifest, pipeline


def _row_compiles(spec, manifest, pipeline) -> None:
    import compile_spec
    from vega_core import canonical_digest
    # the spec names the blueprint it is compiled against (netspec-contract §1, D1e)
    spec = {**spec, "input": {**(spec.get("input") or {}),
                              "pipeline_digest": canonical_digest(pipeline)}}
    markers = compile_spec.compile_spec(spec, model_html=None, pipeline=pipeline,
                                        manifest=manifest)
    assert markers.ok, [m.line() for m in markers.errors]


_ENCODER = [_row_block("B1", "embedding", columns=["zip"], dims={"zip": 4}),
            _row_block("B2", "concat"), _row_block("B3", "dense", units=16, activation="relu"),
            _row_block("B4", "dense", units=3, activation="none"),
            _row_block("B5", "decoder", widths=[16])]


def test_the_pca_floor_reads_the_features_through_dense_z():
    spec, manifest, pipeline = _row_autoencoder(
        _ENCODER + [_row_block("B6", "head", out=8, activation="none")])
    _row_compiles(spec, manifest, pipeline)
    rung, why = PL.rung_floor(spec, CATALOG, dims_95=5)
    assert why is None and rung["id"] == "R1"
    floor = _applied(rung, spec)
    assert floor["family"]["name"] == "pca-floor"
    assert floor["family"]["catalog"] == "families/pca-floor"
    assert _types(floor) == ["embedding", "concat", "dense", "head"]
    z = floor["blocks"][2]
    # the spec's own bottleneck (B4, 3 wide), not dims_95: R0 and R1 compress to one width
    assert z["params"] == {"units": 3, "activation": "none"}
    assert z["catalog"] == "families/pca-floor" and z["source"] == "vega:default"
    assert "dense B4" in z["rationale"]
    assert z["alternatives"] == [{"params": {"units": 5}, "when": z["alternatives"][0]["when"]}]
    assert z["id"] not in {b["id"] for b in spec["blocks"]}
    assert floor["blocks"][-1]["params"] == {"out": 8, "activation": "none"}    # R0's head
    assert z["params"]["units"] < floor["blocks"][-1]["params"]["out"]        # not full rank
    _row_compiles(floor, manifest, pipeline)


def test_the_pca_floor_z_falls_back_to_the_vae_latent_then_dims_95():
    vae = [_row_block("B1", "concat"), _row_block("B2", "dense", units=6),
           _row_block("B3", "reparameterize", z=3), _row_block("B4", "decoder", widths=[8]),
           _row_block("B5", "head", out=8)]
    spec, _, _ = _row_autoencoder(vae, family="vae")
    latent, _ = PL.Latent.of(spec, 5)
    assert (latent.z, latent.where) == (3, "the spec's bottleneck (reparameterize B3)")
    svdd = [_row_block("B1", "concat"), _row_block("B2", "sequential", inner=[
        {"type": "dense", "params": {"units": 8, "activation": "tanh"}}]),
        _row_block("B3", "head", out=4, bias=False)]
    spec, _, _ = _row_autoencoder(svdd, family="deep-svdd")
    latent, _ = PL.Latent.of(spec, 5)
    assert (latent.z, latent.where) == (5, "the geometry layer's dims_95")
    rung, why = PL.rung_floor(spec, CATALOG)                 # no dims_95, no bottleneck
    assert rung is None and "neither the spec's bottleneck" in why and "not planned" in why


def test_the_geometry_layer_gives_dims_95(tmp_path):
    model = tmp_path / "m.html"
    model.write_text('<script id="layer-geometry-data" type="application/json">'
                     '{"space": {"dims_95": 6}}</script>', encoding="utf-8")
    assert PL.geometry_dims(model) == 6
    assert PL.geometry_dims(tmp_path / "absent.html") is None


# --------------------------------------------------------------------------- 2. R1b

def test_r1b_swaps_the_gru_to_rnn_and_compiles(spike, tmp_path):
    fx = spike["window"]
    spec = _spec(fx)
    rung, why = PL.rung_recurrent_floor(spec, FORGE_CATALOG)
    assert why is None and rung["id"] == "R1b"
    assert rung["source"] == "vega:ladder/recurrent-floor" and rung["family"] == "architecture"
    floor = _applied(rung, spec)
    assert floor["family"]["name"] == "rnn" and floor["family"]["catalog"] == "families/rnn"
    gru = next(b for b in spec["blocks"] if b["type"] == "gru")
    rnn = next(b for b in floor["blocks"] if b["type"] == "rnn")
    assert rnn["id"] == gru["id"]
    assert rnn["params"]["hidden"] == gru["params"]["hidden"]
    assert rnn["params"]["num_layers"] == gru["params"].get("num_layers", 1)
    assert rnn["params"]["return"] == "last" and rnn["params"]["bidirectional"] is False
    assert floor["blocks"][-1] == spec["blocks"][-1]       # the head, as R0's
    _compiles(floor, fx)
    shape, batch = _forward(floor, fx, tmp_path / "r1b")
    assert shape == (batch["x_num"].shape[0], spec["data_contract"]["sample"]["horizon"])


@pytest.mark.parametrize("family,blocks,removed", [
    ("lstm-attention", [_block("B3", "lstm", hidden=16, num_layers=2, **{"return": "all"}),
                        _block("B4", "attention_pool", hidden=16), _block("B5", "dropout", p=0.1)],
     {"attention_pool"}),
    ("gru-attention", [_block("B3", "gru", hidden=16, num_layers=1, **{"return": "all"}),
                       _block("B4", "attention_pool", hidden=8), _block("B5", "dropout", p=0.1)],
     {"attention_pool"}),
    ("bi-lstm", [_block("B3", "lstm", hidden=16, num_layers=2, bidirectional=True,
                        **{"return": "last"}), _block("B5", "dropout", p=0.1)], set()),
    ("bi-gru", [_block("B3", "gru", hidden=16, layers=2, bidirectional=True,
                       **{"return": "last"}), _block("B5", "dropout", p=0.1)], set()),
    ("lstm", [_block("B3", "lstm", hidden=16, num_layers=1, **{"return": "all"}),
              _block("B4", "pool", kind="mean")], {"pool"}),
])
def test_r1b_for_every_recurrent_family(spike, family, blocks, removed):
    fx = spike["window"]
    base = _spec(fx)
    head = base["blocks"][-1]
    spec = _window_variant(base, family, base["blocks"][:2] + blocks + [head])
    rung, why = PL.rung_recurrent_floor(spec, FORGE_CATALOG)
    assert why is None and rung["id"] == "R1b"
    floor = _applied(rung, spec)
    types = _types(floor)
    assert types.count("rnn") == 1 and not {"gru", "lstm"} & set(types)
    assert not removed & set(types) and "attention" not in types
    rnn = next(b for b in floor["blocks"] if b["type"] == "rnn")
    gated = blocks[0]["params"]
    assert rnn["params"]["hidden"] == gated["hidden"]
    assert rnn["params"]["num_layers"] == gated.get("num_layers", gated.get("layers"))
    assert rnn["params"]["return"] == "last" and rnn["params"]["bidirectional"] is False
    assert floor["blocks"][-1] == head
    _compiles(floor, fx)                     # rnn forecasts; a bi-* base could not (W3)


@pytest.mark.parametrize("family", ["rnn", "cnn1d", "transformer-seq", "gcn", "gin",
                                    "mlp-residual", "linear", "autoencoder"])
def test_no_r1b_outside_the_recurrent_families(spike, family):
    spec = _spec(spike["window"])
    spec["family"] = dict(spec["family"], name=family)
    assert PL.rung_recurrent_floor(spec, FORGE_CATALOG) == (None, None)


def test_r1b_families_are_the_forge_catalogs():
    assert set(FORGE_CATALOG["families"]["rnn"]["ablation_floor_for"]) == set(RECURRENT)


# --------------------------------------------------------------------------- 3. the ids

def _stub_campaign(fixture, root: Path) -> Path:
    """A campaign directory over a spike fixture: its spec, its package, a model file with no
    layer (the spike has none; every source it cites resolves without one)."""
    import shutil
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(fixture.spec, root / "netspec.json")
    shutil.copytree(fixture.package_dir, root / fixture.package)
    (root / "model.html").write_text("<!doctype html><html><head><title>stub</title></head>"
                                     "<body></body></html>", encoding="utf-8")
    return root


def test_the_unattended_window_ladder_is_r0_r1_r1b_then_the_rest(spike, tmp_path):
    camp = _stub_campaign(spike["window"], tmp_path / "camp")
    code = PL.main(["--campaign", str(camp), "--model", str(camp / "model.html"),
                    "--budget", "3", "--no-hpo", "--unattended"])
    assert code == 0
    ladder = json.loads((camp / "ladder.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in ladder["runs"]]
    assert ids[:3] == ["R0", "R1", "R1b"]
    assert ids[3:] == [f"R{n}" for n in range(2, 2 + len(ids) - 3)]
    by_id = {r["id"]: r for r in ladder["runs"]}
    assert by_id["R1"]["source"] == "vega:ladder/floor"
    assert by_id["R1b"]["source"] == "vega:ladder/recurrent-floor"
    assert "recurrent-floor" in ladder["rules_applied"]


def test_a_dropped_floor_leaves_r1_unused():
    accepted = [{"id": "R0", "kind": "baseline", "source": "netspec"},
                {"id": "D1", "kind": "ablation", "source": "vega:ladder/default-test"},
                {"id": "A1", "kind": "ablation", "source": "netspec:optimizer/alternatives[0]"},
                {"id": "H1", "kind": "hpo", "source": "vega:ladder/hpo"}]
    PL.number_rungs(accepted)
    assert [r["id"] for r in accepted] == ["R0", "R2", "R3", "H1"]
    with_floor = [{"id": "x", "kind": "ablation", "source": "vega:ladder/floor"},
                  {"id": "y", "kind": "ablation", "source": "vega:ladder/recurrent-floor"},
                  {"id": "z", "kind": "ablation", "source": "vega:ladder/capacity"}]
    PL.number_rungs(with_floor)
    assert [r["id"] for r in with_floor] == ["R1", "R1b", "R2"]


# --------------------------------------------------------------------------- 4. the spaces

def test_the_recurrent_space_resolves_on_the_window_spec(spike):
    spec = _spec(spike["window"])
    space, dropped = PL.build_space(spec, CATALOG)
    paths = {d["path"] for d in space["dimensions"]}
    assert not space["generic"]
    # the spike's gru declares no dropout, so that dimension is the one the spec cannot move
    assert dropped == ["regularization[kind=dropout]/value (the spec has nothing there)"]
    assert {"optimizer/params/lr", "blocks/B3/params/hidden",
            "blocks/B3/params/num_layers"} <= paths
    attention = _window_variant(spec, "gru-attention", spec["blocks"][:2] + [
        _block("B3", "gru", hidden=16, num_layers=1, **{"return": "all"}),
        _block("B4", "attention_pool", hidden=16)] + spec["blocks"][-1:])
    space, _ = PL.build_space(attention, CATALOG)
    assert "blocks/B4/params/hidden" in {d["path"] for d in space["dimensions"]}


def test_an_unstated_num_layers_is_searched_from_the_catalog_default(spike):
    """E21 (D1 §12.4): a forge-composed lstm-attention leaves `num_layers` at its catalog
    default; the depth dimension is still searched, from that default, and a trial at 2
    compiles."""
    fx = spike["window"]
    base = _spec(fx)
    lstm = _block("B3", "lstm", hidden=16, **{"return": "all"})
    spec = _window_variant(base, "lstm-attention", base["blocks"][:2] + [
        lstm, _block("B4", "attention_pool", hidden=16)] + base["blocks"][-1:])
    assert "num_layers" not in lstm["params"]
    space, dropped = PL.build_space(spec, CATALOG)
    assert not any("num_layers" in d for d in dropped), dropped
    dim = next(d for d in space["dimensions"] if d["path"] == "blocks/B3/params/num_layers")
    assert dim["current"] == FORGE_CATALOG["blocks"]["lstm"]["defaults"]["num_layers"] == 1
    assert dim["current_from"] == "vega-forge catalog blocks/lstm.defaults"
    assert dim["range"] == [1, 2]
    trial = Delta.of([{"path": dim["path"], "to": 2}]).apply(spec)
    assert next(b for b in trial["blocks"] if b["id"] == "B3")["params"]["num_layers"] == 2
    _compiles(trial, fx)
    # the older name the builder still reads is the current value, not the catalog's
    older = copy.deepcopy(spec)
    older["blocks"][2]["params"]["layers"] = 2
    space, _ = PL.build_space(older, CATALOG)
    dim = next(d for d in space["dimensions"] if d["path"] == "blocks/B3/params/num_layers")
    assert dim["current"] == 2 and "older name" in dim["current_from"]
    # a param no catalog default covers is still dropped, with its reason
    assert PL.unstated_param(spec, "blocks/B4/params/heads", FORGE_CATALOG) == (None, None)
    assert PL.unstated_param(spec, "optimizer/params/lr", FORGE_CATALOG) == (None, None)


def test_the_autoencoder_range_resolves_on_dims_95():
    spec, _, _ = _row_autoencoder(_ENCODER + [_row_block("B6", "head", out=8)])
    space, _ = PL.build_space(spec, CATALOG, FORGE_CATALOG, dims_95=6)
    units = next(d for d in space["dimensions"] if d["path"] == "blocks/B4/params/units")
    assert units["current"] == 3 and units["range"] == [3, 12]
    space, _ = PL.build_space(spec, CATALOG, FORGE_CATALOG)          # no geometry layer
    units = next(d for d in space["dimensions"] if d["path"] == "blocks/B4/params/units")
    assert units["range"] == [4, 16]


def test_the_graph_space_resolves_and_edge_dropout_is_optional(spike):
    spec = _spec(spike["node-keypair"])
    space, dropped = PL.build_space(spec, CATALOG)
    paths = {d["path"] for d in space["dimensions"]}
    assert not space["generic"]
    assert "blocks/B1/params/out" in paths                  # the first gcn_conv
    assert any("<edge_dropout id>" in n for n in space["not_applicable"])
    assert not any("edge_dropout" in d for d in dropped)    # no WARN for an absent option
    lr = next(d for d in space["dimensions"] if d["path"] == "optimizer/params/lr")
    assert lr["range"] == [0.001, 0.05]
    spec["blocks"].insert(0, _block("B0", "edge_dropout", p=0.1))
    space, _ = PL.build_space(spec, CATALOG)
    assert "blocks/B0/params/p" in {d["path"] for d in space["dimensions"]}
    gat = copy.deepcopy(spec)
    gat["family"]["name"] = "gat"
    gat["blocks"][1] = _block("B1", "gat_conv", out=8, heads=4, concat=True)
    space, _ = PL.build_space(gat, CATALOG)
    assert "blocks/B1/params/heads" in {d["path"] for d in space["dimensions"]}


def test_the_mlp_space_on_a_plain_mlp_has_no_residual_and_no_warn():
    """The release4 rehearsal's mlp-plain spec planned with `WARN: search dimension dropped —
    blocks/<residual id>/params/repeat`: the `mlp-*` row's depth dimension is mlp-residual's,
    so on mlp-plain it is not applicable, like an absent edge_dropout — not a WARN."""
    spec = {"family": {"name": "mlp-plain"},
            "blocks": [_row_block("B1", "dense", units=32, activation="none"),
                       _row_block("B2", "head", out=6, activation="none")],
            "optimizer": {"name": "adamw", "params": {"lr": 1e-3}}}
    space, dropped = PL.build_space(spec, CATALOG, FORGE_CATALOG)
    assert any("<residual id>" in n for n in space["not_applicable"])
    assert not any("residual" in d for d in dropped), dropped
    residual = copy.deepcopy(spec)
    residual["family"]["name"] = "mlp-residual"
    residual["blocks"].insert(1, _row_block("B3", "residual", repeat=2, inner=[]))
    space, _ = PL.build_space(residual, CATALOG, FORGE_CATALOG)
    assert "blocks/B3/params/repeat" in {d["path"] for d in space["dimensions"]}


# --------------------------------------------------------------------------- 5. alternatives

def test_a_different_choice_replaces_the_params():
    loss = {"name": "bce_with_logits", "params": {"pos_weight": 4.9}, "source": "analysis:F2",
            "rationale": "r", "alternatives": [{"name": "focal", "params": {"gamma": 2},
                                                "when": "w"}]}
    merged = PL._merge_alternative(loss, loss["alternatives"][0], "why")
    assert merged["name"] == "focal" and merged["params"] == {"gamma": 2}
    assert merged["alternatives"][0]["params"] == {"pos_weight": 4.9}
    sched = {"name": "cosine", "params": {"warmup_epochs": 3}, "source": "vega:default",
             "rationale": "r", "alternatives": [{"name": "plateau", "when": "w"}]}
    assert PL._merge_alternative(sched, sched["alternatives"][0], "why")["params"] == {}
    same = {"name": "adamw", "params": {"lr": 0.001, "weight_decay": 0.0001},
            "source": "vega:default", "rationale": "r",
            "alternatives": [{"params": {"lr": 0.01}, "when": "w"}]}
    assert PL._merge_alternative(same, same["alternatives"][0], "why")["params"] == \
        {"lr": 0.01, "weight_decay": 0.0001}


def test_a_flat_choice_takes_the_alternatives_params_as_its_values():
    stop = {"monitor": "valid.pr_auc", "mode": "max", "patience": 10, "min_delta": 0.001,
            "source": "vega:default", "rationale": "r",
            "alternatives": [{"params": {"patience": 20}, "when": "noisy"}]}
    merged = PL._merge_alternative(stop, stop["alternatives"][0], "why")
    assert merged["patience"] == 20 and "params" not in merged
    assert merged["alternatives"][0]["patience"] == 10


def test_a_sample_fork_point_runs_the_catalogs_alternative(spike):
    spec = _spec(spike["window"])
    for point, want in (("context", "none"), ("autoregressive", False)):
        entry = {"path": f"data_contract/sample/{point}", "observation": "o"}
        rung, why = PL.rung_default_test(entry, spec, 1, FORGE_CATALOG)
        assert why is None, why
        assert rung["delta"] == [{"path": f"data_contract/sample/{point}",
                                  "from": spec["data_contract"]["sample"][point], "to": want}]
        assert "the catalog's alternative" in rung["hypothesis"]
    entry = {"path": "data_contract/sample/series_by", "observation": "o"}
    rung, _ = PL.rung_default_test(entry, spec, 1, FORGE_CATALOG)
    assert rung["delta"][0]["to"] is None
    graph = _spec(spike["graph-level"])
    rung, why = PL.rung_default_test({"path": "data_contract/sample/batching"}, graph, 1,
                                     FORGE_CATALOG)
    assert rung is None and "graphs is the only batching" in why
    node = _spec(spike["node-keypair"])
    rung, why = PL.rung_default_test({"path": "data_contract/sample/batching"}, node, 1,
                                     FORGE_CATALOG)
    assert why is None and [(e["path"], e["to"]) for e in rung["delta"]] == [
        ("data_contract/sample/batching", "sampled"), ("data_contract/sample/neighbours",
                                                       [25, 10])]


def test_a_block_default_by_id_reads_the_blocks_own_alternative(spike):
    spec = _spec(spike["window"])
    rung, why = PL.rung_default_test({"path": "blocks/B1"}, spec, 1, FORGE_CATALOG)
    assert why is None
    assert rung["delta"][0]["path"] == "blocks[id=B1]"
    assert rung["delta"][0]["to"]["params"]["dims"] != spec["blocks"][0]["params"]["dims"]


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))


# --------------------------------------------------------------------------- F37

def test_a_plan_without_a_model_is_refused_and_writes_nothing(spike, tmp_path, capsys):
    """F37 (rehearsal final3r2): plan_ladder.py ran without --model wrote a campaign.json whose
    input.model was null, and every rung the trainer then ran was refused as uncompilable.
    --model is required, and a model that is not a file is refused before anything is written."""
    camp = tmp_path / "camp"
    camp.mkdir()
    shutil.copy(spike["window"].spec, camp / "netspec.json")
    with pytest.raises(SystemExit) as exc:
        PL.main(["--campaign", str(camp), "--unattended"])
    assert exc.value.code == 2
    assert "--model" in capsys.readouterr().err
    assert PL.main(["--campaign", str(camp), "--model", str(tmp_path / "absent.vega.html"),
                    "--unattended"]) == 2
    said = capsys.readouterr().out
    assert "--model" in said and "not found" in said
    assert not (camp / "campaign.json").exists() and not (camp / "ladder.json").exists()


def test_the_documented_plan_call_carries_the_model():
    """F37: SKILL.md step 1 gave the call without --model, and a session that followed it
    planned a campaign no rung could compile in."""
    import re
    skill = (SCRIPTS.parent / "SKILL.md").read_text(encoding="utf-8")
    calls = re.findall(r"`python3 scripts/plan_ladder\.py ([^`]*)`", skill)
    assert calls and all("--model" in c for c in calls), calls
