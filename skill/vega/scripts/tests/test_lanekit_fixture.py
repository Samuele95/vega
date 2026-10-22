"""
The fixture builder's tests (BUILD-PLAN T02): `make_fixture.py` writes what G1's version wrote,
byte for byte, now that its model source and world model are lanekit's; a lane fixture is the
file set the later stages read, with a base model and layers domain-forge accepts; the pieces
the builder is made of (the model source, the dataset generators' protocol, the custom block).

    python3 -m pytest -q vega/scripts/tests/test_lanekit_fixture.py

The G1 reference is read from git history: the tree's own repository, else the one the lane
program copied the tree from (`VEGA_SOURCE_REPO`, run_matrix.py). With neither reachable the
identity test fails rather than skips — it is a closing test.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import checks, contracts, datasets, fixture, model_source, registry  # noqa: E402
from lanekit.assertions import Checker, Ctx  # noqa: E402
from lanekit.datasets import ColumnSpec  # noqa: E402
from lanekit.recipes import standard_recipe  # noqa: E402

REPO = HERE.parents[2]
PY = sys.executable
MAKE_FIXTURE = "vega-forge/scripts/tests/make_fixture.py"
G1_COMMIT = "7d4192a"                  # Wave W0: G1 — the version T02 must reproduce
G1_BLOB = "6a72f89ec27e0ff2e98a3b2fc2cd755d95cee32e"
# the platform writer stamps each layer with the wall clock; nothing else in the output moves
PRODUCED_AT = re.compile(rb"^(\s*produced-at: )\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", re.M)


# --------------------------------------------------------------------------- make_fixture.py
def _g1_source() -> bytes:
    failures = []
    for repo in [str(REPO)] + [r for r in [os.environ.get("VEGA_SOURCE_REPO")] if r]:
        try:
            blob = subprocess.run(["git", "-C", repo, "rev-parse", f"{G1_COMMIT}:{MAKE_FIXTURE}"],
                                  capture_output=True, text=True, check=True).stdout.strip()
            assert blob == G1_BLOB, f"{G1_COMMIT}:{MAKE_FIXTURE} is {blob}, not G1's {G1_BLOB}"
            return subprocess.run(["git", "-C", repo, "cat-file", "blob", blob],
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            failures.append(f"{repo}: {exc}")
    pytest.fail(f"G1's make_fixture.py is not reachable in git history ({'; '.join(failures)})")


def _tree(root: Path) -> dict[str, bytes]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = PRODUCED_AT.sub(rb"\1<stamp>", path.read_bytes())
    return out


#: the one file whose bytes moved on purpose since G1: W8's track F (PLAN-AMENDMENTS A46) made
#: `regularization[]` the one declaration (rule R2) and dropped the live copies the G1 spec
#: carried beside it (`optimizer.params.weight_decay`, B3's inner dropout `p`); resolved — the
#: declared values written into the live paths, as the emitted code builds them — it is G1's
#: spec resolved, exactly
RESOLVED_SINCE_G1 = ("netspec.json",)
#: the layered model's bytes moved on purpose too (release track FX-small): its geometry layer
#: writes the report contract's partitions (a chosen label and candidates by label), which the
#: compiler resolves `geometry:partitions/late` against, and its shape layer records the
#: manifest file's digest as `apply_shape_layer.py` now writes it. Everything else in the file —
#: the base model, every other layer key, every header — is G1's, byte for byte.
MODEL = "orders.domain.analysis.shaped.html"
LAYER_DATA = re.compile(rb'(<script id="layer-([a-z]+)-data" type="application/json">)([\s\S]*?)(</script>)')


def _model_since_g1(new: bytes, old: bytes, manifest: bytes) -> None:
    """The model differs from G1's in the geometry's partitions and the shape layer's
    manifest_digest only, and each holds what FX-small says it holds."""
    def split(raw):
        return (LAYER_DATA.sub(rb"\1<data>\4", raw),
                {m.group(2).decode(): json.loads(m.group(3)) for m in LAYER_DATA.finditer(raw)})
    (new_rest, new_layers), (old_rest, old_layers) = split(new), split(old)
    assert new_rest == old_rest, "the model moved outside its layers' data"
    assert sorted(new_layers) == sorted(old_layers)
    geometry, g1_geometry = new_layers.pop("geometry"), old_layers.pop("geometry")
    (label, g1_partition), = g1_geometry.pop("partitions").items()
    assert geometry.pop("partitions") == {"chosen": label, "provenance": "user-chosen",
                                          "candidates": [{"label": label, **{
                                              k: v for k, v in g1_partition.items() if k != "label"}}]}
    assert geometry == g1_geometry
    assert new_layers["shape"].pop("manifest_digest") == \
        "sha256:" + hashlib.sha256(manifest).hexdigest()
    assert new_layers == old_layers

#: the blueprint hand-off, whose bytes moved on purpose since G1: the release's FX-roster track
#: (code-contract §3) lets an artifact the package does not hold be declared `external` with its
#: owner, and the fixture's campaign — /vega-train's directory — is so declared; without those
#: declarations it is G1's pipeline.json, exactly
DECLARED_SINCE_G1 = {"blueprint-runs/2026-09-05T10-12Z/pipeline.json": ("Campaign",)}
#: the port signatures that moved on purpose: the ModelStore's `save` takes the spec the model
#: was trained against, required (release track T5r1); `save` and `load` take their required
#: collaborators positionally and their options by keyword (T5r3, a port's arity is its truth)
SIGNED_SINCE_G1 = {
    "def save(self, model: TrainedNetwork, path, spec: NetSpec, *, epoch: int | None = None) -> "
    "Checkpoint: ...":
    "def save(self, model, path, spec: NetSpec | None = None, epoch: int | None = None) -> "
    "Checkpoint: ...",
    "def load(self, ref: Checkpoint | str, spec: NetSpec, build: ModelBuilder, *, "
    "expected_digest: str | None = None, run_dir=None) -> TrainedNetwork: ...":
    "def load(self, ref: Checkpoint | str, expected_digest: str | None = None, "
    "spec: NetSpec | None = None, build=None, run_dir=None) -> Any: ...",
    # the final release round (rel2/final1): the Trainer's and the Evaluator's collaborators are
    # typed with the backend's interfaces, and `predict` returns the scored part of a held part
    "def fit(self, model: TrainedNetwork, spec: NetSpec, data: TrainingData, "
    "callbacks: Callbacks | None = None, run: RunContext | None = None) -> FitResult: ...":
    "def fit(self, model, spec: NetSpec, data, callbacks=None, run: RunContext | "
    "None = None) -> FitResult: ...",
    "def evaluate(self, model: TrainedNetwork, spec: NetSpec, loader, loss_fn=None) "
    "-> dict: ...":
    "def evaluate(self, model, spec: NetSpec, loader, loss_fn=None) -> dict: ...",
    "def predict(self, model: TrainedNetwork, spec: NetSpec, part: SamplePart) "
    "-> ScoredPart: ...":
    "def predict(self, model, spec: NetSpec, loader) -> tuple: ...",
    "def scored(self, spec: NetSpec, dataset: SamplePart, model: ScoredModel) "
    "-> ScoredPart: ...":
    "def scored(self, spec: NetSpec, dataset: SamplePart, model) -> ScoredPart: ...",
    "def score_part(self, spec: NetSpec, part: Part, checkpoint=None, "
    "model: ScoredModel | None = None) -> Metrics: ...":
    "def score_part(self, spec: NetSpec, part: Part, checkpoint=None, model=None) "
    "-> Metrics: ...",
    "def score_sealed_test(self, spec: NetSpec, token: str, checkpoint=None, "
    "model: ScoredModel | None = None, seal_path: str = \"test_seal.json\") -> Metrics: ...":
    "def score_sealed_test(self, spec: NetSpec, token: str, checkpoint=None, "
    "model=None, seal_path: str = \"test_seal.json\") -> Metrics: ..."}
#: the port methods the world model gained on purpose (release track T5r2): a port declares
#: every port method of its role — the ones the chain's stages call through it — with its
#: adapter's parameters (code-contract §1, `port_surface.py`); G1 drew a subset. The Model
#: port's `head` and `anatomy` are the serving seam (release 2 track seam): what the Predictor
#: and the inspection read of the network, and all they read
ADDED_SINCE_G1 = {
    "Model": ("def head(self, batch: dict) -> Any: ...",
              "def anatomy(self) -> NetworkAnatomy | None: ..."),
    "DataSource": ("def batch_count(self, spec: NetSpec, part: Part, batch_size: int | None = "
                   "None, plan: BatchPlan | None = None) -> int: ...",),
    "Evaluator": ("def measure(self, spec: NetSpec, part: Part, scored: ScoredPart) -> Metrics: ...",
                  "def heads(self, spec: NetSpec, dataset: SamplePart, model: ScoredModel) "
                  "-> Any: ...",
                  "def metric_names(self, spec: NetSpec) -> list: ...",
                  "def with_options(self, options: MetricOptions) -> Any: ..."),
    "Registry": ("def version(self, path) -> Checkpoint: ...",)}
#: the module arrows that moved on purpose (release track T5r3): §4 draws the arrows the
#: package's code takes and no other (C4 now refuses an arrow no import takes) — the Trainer,
#: the Evaluator and the Predictor read the Model module; nothing reads `ingestion` or
#: `features` across a module boundary (the DataSource and the Recipe arrive as ports). G1
#: drew these, as `depends_on` and in blueprint.md's §4 sentence
ARROWS_SINCE_G1 = {"training": ["domain", "ingestion", "model"],
                   "evaluation": ["domain", "ingestion", "model"],
                   "serving": ["domain", "features", "model"]}
MODULE_MAP_SINCE_G1 = (b"training, evaluation and serving <- domain, model.",
                       b"training and evaluation <- domain, ingestion, model; serving <- domain, "
                       b"features, model.")
#: FILLED_SINCE_G1 (release track T5r3): a port whose adapter /vega-pack fills says so,
#: `"filled_by": "/vega-pack"` — G1's ports carried no such key
#: the port methods a row package of a binary task no longer carries (release track T5r3: a
#: graph's exposure and a forecast window's seasonal scale are those packages' methods only,
#: SOLID-I); G1 drew them last on its DataSource
REMOVED_SINCE_G1 = {
    "DataSource": ("def exposure(self, spec: NetSpec) -> GraphExposure | None: ...",
                   "def target_scale(self, spec: NetSpec, season: int = 1) -> float | None: ...")}


def _pinned(spec: dict, tree: dict[str, bytes], canonical: bool) -> dict:
    """The spec with its `input.pipeline_digest` checked against the pipeline.json of its own
    tree and set aside: the hand-off moved by DECLARED_SINCE_G1, so its digest did; everything
    else must still be G1's. `canonical`: the digest netspec-contract §1 defines (sorted keys, no
    whitespace — FX-small, rule D1e); G1's make_fixture used json's default separators."""
    inputs = spec.get("input") or {}
    pipeline = next(name for name in DECLARED_SINCE_G1 if name.endswith("pipeline.json"))
    separators = (",", ":") if canonical else None
    want = "sha256:" + hashlib.sha256(json.dumps(json.loads(tree[pipeline]), sort_keys=True,
                                                 separators=separators).encode()).hexdigest()
    assert inputs.get("pipeline_digest") == want, "the spec does not pin its tree's pipeline.json"
    return {**spec, "input": {k: v for k, v in inputs.items() if k != "pipeline_digest"}}


def _undeclared(raw: bytes) -> tuple[dict, list[str]]:
    """The pipeline with every `external` declaration removed, and the artifacts that had one;
    a signature SIGNED_SINCE_G1 names is read back as G1 drew it."""
    doc = json.loads(raw)
    declared = [a["name"] for a in doc.get("artifacts") or [] if a.pop("external", None)]
    for module in doc.get("modules") or []:
        if module.get("name") in ARROWS_SINCE_G1:
            module["depends_on"] = list(ARROWS_SINCE_G1[module["name"]])
    for port in doc.get("ports") or []:
        if isinstance(port, dict):
            port.pop("filled_by", None)             # FILLED_SINCE_G1
        if isinstance(port, dict) and port.get("signature"):
            added = ADDED_SINCE_G1.get(port.get("name"), ())
            port["signature"] = [SIGNED_SINCE_G1.get(line, line) for line in port["signature"]
                                 if line not in added] + list(REMOVED_SINCE_G1.get(port.get("name"), ()))
    return doc, declared


def test_the_signature_moved_since_g1_is_the_one_the_ports_draw():
    """SIGNED_SINCE_G1 tolerates exactly the lines the world model now draws (a stale entry
    would hide a moved signature)."""
    from lanekit.pipeline import PORTS, port_signatures
    drawn = {line for port in PORTS for line in port[4]}
    assert set(SIGNED_SINCE_G1) <= drawn and not set(SIGNED_SINCE_G1.values()) & drawn
    by_port = {port[0]: set(port[4]) for port in PORTS}
    for name, lines in ADDED_SINCE_G1.items():
        assert set(lines) <= by_port[name], (name, set(lines) - by_port[name])
    orders = port_signatures("row", "binary")              # the orders fixture's package
    for name, lines in REMOVED_SINCE_G1.items():
        assert not set(lines) & set(orders[name]), (name, set(lines) & set(orders[name]))


def _resolved(raw: bytes) -> dict:
    forge = str(REPO / "vega-forge" / "scripts")
    if forge not in sys.path:
        sys.path.insert(0, forge)
    import vega_regularization
    return vega_regularization.resolve(json.loads(raw))


def test_make_fixture_output_is_byte_identical_to_g1(tmp_path):
    """G1's make_fixture.py and this one, run into the same directory one after the other, write
    the same files with the same bytes (the layers' produced-at stamps aside), and say the same
    thing while doing it — but for the spec, which since A46 states each regularization value
    once and resolves to exactly what G1's spec resolves to (`RESOLVED_SINCE_G1`), and the
    blueprint hand-off, which declares the campaign external and is otherwise G1's
    (`DECLARED_SINCE_G1`; the spec pins its new digest)."""
    g1 = tmp_path / "g1" / MAKE_FIXTURE
    g1.parent.mkdir(parents=True)
    g1.write_bytes(_g1_source())
    for sibling in ("dataset-shaper", "domain-forge"):          # what G1's script resolves
        (tmp_path / "g1" / sibling).symlink_to((REPO / sibling).resolve())
    out = tmp_path / "out"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    runs = {}
    for name, script in (("g1", g1), ("t02", REPO / MAKE_FIXTURE)):
        proc = subprocess.run([PY, str(script), str(out)], capture_output=True, env=env, timeout=600)
        assert proc.returncode == 0, f"{name}: {proc.stdout[-1500:]!r} {proc.stderr[-1500:]!r}"
        runs[name] = (_tree(out), PRODUCED_AT.sub(rb"\1<stamp>", proc.stdout))
        shutil.move(str(out), str(tmp_path / f"out.{name}"))
    (g1_files, g1_log), (files, log) = runs["g1"], runs["t02"]
    assert sorted(files) == sorted(g1_files), "the two versions write different file sets"
    for name in [n for n in files if n.endswith("blueprint.md")]:
        new, old = MODULE_MAP_SINCE_G1
        assert new in files[name], f"{name} no longer draws the arrows the code takes"
        files[name] = files[name].replace(new, old)
    changed = [name for name in g1_files if files[name] != g1_files[name]
               and name not in RESOLVED_SINCE_G1 + (MODEL,) and name not in DECLARED_SINCE_G1]
    assert not changed, f"files that differ from G1's: {changed}"
    _model_since_g1(files[MODEL], g1_files[MODEL], files["shaped/manifest.json"])
    for name, externals in DECLARED_SINCE_G1.items():
        doc, declared = _undeclared(files[name])
        assert tuple(declared) == externals, f"{name} declares {declared} external, not {externals}"
        assert doc == json.loads(g1_files[name]), \
            f"{name} without its external declarations is not G1's: more than the declarations moved"
        assert _undeclared(g1_files[name])[1] == [], f"G1's {name} already declared an external"
    for name in RESOLVED_SINCE_G1:
        assert files[name] != g1_files[name], f"{name} no longer differs: drop it from the list"
        # FX-small: this tree's spec pins the canonical digest (netspec-contract §1, D1e); G1's
        # pinned its own pipeline's sorted-keys digest with json's default separators
        assert _pinned(_resolved(files[name]), files, canonical=True) == \
            _pinned(_resolved(g1_files[name]), g1_files, canonical=False), \
            f"{name} resolved is not G1's resolved: more than the live copies moved"
    assert log == g1_log
    assert len(g1_files) >= 12


def test_make_fixture_draws_its_model_source_and_world_model_from_lanekit():
    text = (REPO / MAKE_FIXTURE).read_text(encoding="utf-8")
    assert "from lanekit.model_source import" in text and "from lanekit.pipeline import WorldModel" in text
    assert "def write_model_source" not in text and '"ports": [' not in text


# --------------------------------------------------------------------------- the model source
def test_local_name_is_camel_case_and_never_starts_with_a_digit():
    assert model_source.local_name("unit_price") == "unitPrice"
    assert model_source.local_name("kwh_00_02") == "kwh0002"
    assert model_source.local_name("b01") == "b01"
    assert model_source.local_name("2nd_axis") == "c2ndAxis"
    with pytest.raises(model_source.ModelSourceError):
        model_source.local_name("__")


def test_properties_leave_out_dropped_columns_and_say_the_unit():
    schema = (ColumnSpec("area_m2", "float", "numeric", "m2", "The living area."),
              ColumnSpec("rooms", "int", "numeric", None, None),
              ColumnSpec("known_after", "float", "drop", None, "Known later."),
              ColumnSpec("district", "str", "nominal", None, None))
    assert model_source.properties_for(schema) == [
        ("areaM2", "xsd:float", "The living area. (m2)"), ("rooms", "xsd:integer", None),
        ("district", "xsd:string", None)]


# --------------------------------------------------------------------------- the generators
def test_generator_discovery_refuses_what_is_not_there_by_name():
    with pytest.raises(datasets.GeneratorError, match="no generator 'no_such_generator'"):
        datasets.load("no_such_generator")
    assert "no_such_generator" not in datasets.GENERATORS
    for name, wp in datasets.pending().items():
        with pytest.raises(datasets.GeneratorError, match=f"planned: {wp} provides it"):
            datasets.load(name)
        break


def test_generator_names_are_the_package_files():
    files = {p.stem for p in (HERE / "lanekit" / "datasets").glob("*.py") if not p.stem.startswith("_")}
    assert set(datasets.names()) == files == set(datasets.GENERATORS)
    lane_generators = {lane.raw["generator"] for lane in registry.load_lanes().values()
                       if lane.driver == "lanekit"}
    assert lane_generators <= set(datasets.PLANNED) | files, "a lane names a generator nobody provides"


def test_column_specs_and_schemas_are_checked():
    with pytest.raises(ValueError, match="role"):
        ColumnSpec("x", "float", "feature")
    with pytest.raises(ValueError, match="dtype"):
        ColumnSpec("x", "double", "numeric")
    dup = (ColumnSpec("x", "float", "numeric"), ColumnSpec("x", "float", "numeric"))
    assert datasets.check_schema(dup) == ["column 'x' is declared twice"]
    assert datasets.check_schema((ColumnSpec("k", "str", "key"),)) == [
        "SCHEMA declares no feature column (numeric or nominal)"]


# --------------------------------------------------------------------------- the custom block
def test_custom_gain_is_the_identity_until_it_learns():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(HERE / "lanekit" / "assets"))
    try:
        import custom_gain
    finally:
        sys.path.remove(str(HERE / "lanekit" / "assets"))
    block = custom_gain.build((16,), {"out_shape": [16]})
    x = torch.randn(5, 16)
    assert torch.equal(block(x), x)
    assert sum(p.numel() for p in block.parameters()) == 16
    with pytest.raises(ValueError, match="shape-preserving"):
        custom_gain.build((16,), {"out_shape": [8]})


# --------------------------------------------------------------------------- a lane fixture
def _housing_like() -> types.SimpleNamespace:
    """A generator keeping the protocol (lanekit.datasets), for L05's declaration: a key, four
    measured quantities, the coordinate pair L05's spatial data needs (`expect.spatial`), the
    district L05 embeds, the label, and a column known only after the sale (the recipe drops it
    as leakage)."""
    import numpy as np
    import pandas as pd

    schema = (ColumnSpec("sale_id", "str", "key", comment="The sale's identifier."),
              ColumnSpec("area_m2", "float", "numeric", "m2", "The living area."),
              ColumnSpec("rooms", "int", "numeric", None, "Rooms."),
              ColumnSpec("age_years", "float", "numeric", "years", "The building's age."),
              ColumnSpec("dist_center_km", "float", "numeric", "km", "Distance to the centre."),
              ColumnSpec("lat", "float", "numeric", "deg", "Latitude."),
              ColumnSpec("lon", "float", "numeric", "deg", "Longitude."),
              ColumnSpec("district", "str", "nominal", None, "The district."),
              ColumnSpec("price_keur", "float", "label", "kEUR", "The sale price."),
              ColumnSpec("notary_fee", "float", "drop", "kEUR", "Known after the sale."))

    def make(seed: int, rows: int):
        rng = np.random.default_rng(seed)
        area = rng.gamma(4.0, 20.0, rows) + 20
        district = rng.choice([f"d{i}" for i in range(8)], rows)
        price = 1.8 * area + rng.normal(0, 15, rows) + np.array([int(d[1:]) * 9 for d in district])
        return pd.DataFrame({"sale_id": [f"S{i:05d}" for i in range(rows)], "area_m2": area.round(1),
                             "rooms": rng.integers(1, 7, rows), "age_years": rng.gamma(2, 12, rows).round(1),
                             "dist_center_km": rng.gamma(2, 3, rows).round(2),
                             "lat": (46.0 + rng.normal(0, 0.03, rows)).round(6),
                             "lon": (11.1 + rng.normal(0, 0.04, rows)).round(6), "district": district,
                             "price_keur": price.round(1), "notary_fee": (price * 0.07).round(2)})

    return types.SimpleNamespace(
        make=make, SCHEMA=schema, VOCAB=frozenset({"sale", "district", "price"}),
        RECIPE=lambda raw_path, shaped_dir, lane: standard_recipe(schema, raw_path, shaped_dir, lane))


def _validate(model: Path) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(HERE / "known_flake.py"), "run", "--stage", "fixture", "--",
                           PY, str(REPO / "domain-forge/scripts/validate_model.py"), str(model)],
                          capture_output=True, text=True, timeout=600)


def _check_shaped(fx) -> None:
    """The shaped parts: every raw row in one part, the key and the leak gone, the twin and the
    determinism checks passed."""
    manifest_path = fx.shaped_dir / "manifest.json"
    assert fx.manifest_digest == "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts = manifest["outputs"]["files"]
    assert sum(parts[p]["rows"] for p in ("train", "valid", "test")) == 400
    assert "notary_fee" not in manifest["output_schema"] and "sale_id" not in manifest["output_schema"]
    verification = json.loads((fx.shaped_dir / "verification.json").read_text(encoding="utf-8"))
    assert (verification["twin"], verification["determinism"]) == ("pass", "pass")


def _check_world(fx) -> None:
    """The netspec and the world model agree on the manifest, the pipeline and the input_dim;
    every port-owning module is OOP and every port carries its signature."""
    spec = json.loads(fx.netspec.read_text(encoding="utf-8"))
    pipeline = json.loads(fx.pipeline.read_text(encoding="utf-8"))
    assert spec["input"]["manifest_digest"] == fx.manifest_digest == pipeline["input"]["shape_manifest_digest"]
    assert spec["input"]["pipeline"] == str(fx.pipeline)
    assert pipeline["serving"]["mode"] == "online"
    assert pipeline["feature_contract"]["input_dim"] == spec["data_contract"]["input_dim"]
    owning = [m for m in pipeline["modules"] if m["ports_owned"]]
    assert owning and all(m["paradigm"] == "OOP" for m in owning)
    assert all(p.get("signature") for p in pipeline["ports"])


def test_a_lane_fixture_is_the_file_set_the_stages_read(tmp_path):
    lane = copy.deepcopy(dict(registry.load_lanes()["L05"].raw))
    lane["rows"] = 400
    fx = fixture.make_lane(lane, tmp_path / "work", REPO, generator=_housing_like())
    work = fx.work
    assert fx.model_html == work / "housing.domain.analysis.shaped.html" and fx.model_html.is_file()
    assert fx.package == "housing_pipeline"
    for model in (work / "housing.model-source" / "housing.domain.html", fx.model_html):
        proc = _validate(model)
        assert proc.returncode == 0, f"{model.name}: {proc.stdout[-2000:]} {proc.stderr[-1000:]}"
    _check_shaped(fx)
    _check_world(fx)
    assert (fx.run_dir / "custom" / "custom_gain.py").is_file(), "the custom block sits under the campaign root"
    memory = (work / ".claude" / "domain-forge-memory.md").read_text(encoding="utf-8")
    assert "family: mlp-plain (user)" in memory and "network: custom block custom_gain (user)" in memory
    assert list((work / ".claude" / "architect-runs").glob("*/findings.json"))
    compiled = subprocess.run([PY, str(REPO / "vega-forge/scripts/compile_spec.py"), "--spec", str(fx.netspec),
                               "--model", str(fx.model_html), "--pipeline", str(fx.pipeline),
                               "--memory", str(work / ".claude/domain-forge-memory.md"), "--check-only"],
                              capture_output=True, text=True, cwd=str(work), timeout=300)
    errors = [ln for ln in compiled.stdout.splitlines() if ln.startswith("ERROR:")]
    assert compiled.returncode == 0 and not errors, compiled.stdout[-3000:]


def test_a_fixture_refuses_a_generator_that_drops_its_label(tmp_path):
    gen = _housing_like()
    gen.SCHEMA = tuple(c for c in gen.SCHEMA if c.name != "price_keur")
    lane = copy.deepcopy(dict(registry.load_lanes()["L05"].raw))
    with pytest.raises(fixture.FixtureError, match="^ERROR: lane L05: target column 'price_keur' is not a label"):
        fixture.make_lane(lane, tmp_path / "work", REPO, generator=gen)
    assert not (tmp_path / "work").exists(), "nothing is written before the lane and its generator agree"


# --------------------------------------------------------------------------- the fixture stage
def test_the_fixture_stage_digests_the_whole_lanekit_tree():
    """A stamp of the fixture stage goes stale when any module a fixture runs changes: the
    builder, the composer and its choices, the recipes, the layers, the stand-ins, the datasets."""
    code = contracts.STAGE_CODE["fixture"]
    assert contracts.LANEKIT_DIR in code
    lanekit = HERE / "lanekit"
    for module in sorted(lanekit.rglob("*.py")):
        rel = f"{contracts.LANEKIT_DIR}/{module.relative_to(lanekit).as_posix()}"
        assert any(rel == c or rel.startswith(c + "/") for c in code), rel


def test_the_fixture_helper_composes_with_the_budget_of_its_lanes_file(tmp_path, monkeypatch):
    doc = registry.load_document()
    doc["defaults"] = {**doc["defaults"], "max_epochs": 5, "patience": 2}
    lanes_file = tmp_path / "lanes.json"
    lanes_file.write_text(json.dumps(doc), encoding="utf-8")
    seen = {}

    class Built(Exception):
        pass

    def make_lane(lane, work, skills, generator=None, composer=None):
        seen["composer"] = composer
        raise Built

    monkeypatch.setattr(fixture, "make_lane", make_lane)
    args = types.SimpleNamespace(lanes_file=str(lanes_file), out=str(tmp_path / "fixture.json"))
    ctx = Ctx(registry.load_lanes(lanes_file)["L05"], tmp_path / "lane", REPO, args)
    with pytest.raises(Built):
        checks.do_fixture(ctx, Checker())
    budget = seen["composer"].budget
    assert (budget.max_epochs, budget.patience) == (5, 2)
