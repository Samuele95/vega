"""
A lane's fixture: raw rows -> the real shaper -> a composed domain model with its three data
layers -> the blueprint's world model -> the memory file -> the netspec (D6 §5.1).

    fx = make_lane(lane, work, skills)            # -> LaneFixture (the fixture stage's record)
    fx = make_lane(lane, work, skills, composer=Composer.for_lanes_file(path))

It writes the file set `make_fixture.py` writes for the orders fixture, under `work`:

    <g>.csv  <g>.recipe.json               the generator's raw rows and the recipe they are
                                           shaped by (lanekit.datasets.<g>: make, RECIPE)
    <g>.model-source/  <g>.domain.html     the base model, composed by domain-forge
                                           (lanekit.model_source)
    <g>.domain.analysis.shaped.html        the base model with its geometry, analysis and shape
                                           layers (lanekit.layers)
    shaped/  (parts, manifest.json, fitted.json, recipe.json, lineage.json,
              verification.json, reproduce_<g>.py, the edges part of a graph lane)
                                           the real shape.py's output, verified by
                                           verify_shape.py, its twin check and shape.py --check
                                           (lanekit.chain.Workspace.shape)
    blueprint-runs/<ts>/{pipeline.json, blueprint.md}   lanekit.pipeline.WorldModel
    netspec.json                           lanekit.spec.Composer.compose
    .claude/domain-forge-memory.md         the lane's Network stances
    .claude/architect-runs/<ts>/findings.json   the recorded /architect run the forge layer splices
    vega-runs/<ts>/custom/<stem>.py        each custom block's module, under the campaign root

The layers come before the shaper, so `shape.py --model` resolves every recipe source against
the layers the recipe cites. How each layer is written — composed from the generator's SCHEMA,
or through the real data sub-chain on a lane whose `expect.real_subchain` is set (L02) — is
lanekit.layers'. The chain's tools are run from `skills`; the netspec is composed against the
catalogs of the tree this lanekit sits in (lanekit.spec_data), with the budget of the
`composer` (the tree's own `lanes.json` when none is given).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import contracts, datasets
from .chain import FixtureError, Workspace, sha_file
from .layers import LayerWriter, RawTable, analysis_payload, geometry_payload, shape_payload
from .model_source import ModelEntity, compose_base_model, properties_for, write_model_source
from .pipeline import WorldModel
from .columns import effective_task
from .recipes import ColumnPlan, findings_for
from .spec import RUN_TS, Composer, Contract, model_name
from .spec_data import raw_of

__all__ = ["LaneFixture", "FixtureError", "make_lane", "pipeline_for", "memory_for",
           "geometry_payload", "analysis_payload", "BLUEPRINT_TS", "ARCHITECT_RUN"]

BLUEPRINT_TS = "2026-09-05T10-12Z"
ARCHITECT_RUN = "vega-forge/scripts/tests/fixtures/dot-claude/architect-runs/2026-09-05T11-20Z"
STANCE_DATE = "2026-09-05"
SERVING_MODES = {"online": "online", "batch+serve": "batch", "batch": "batch", "stream": "stream"}


@dataclass(frozen=True)
class LaneFixture:
    """What the fixture stage records for the later stages (`lanes/<id>/fixture.json`)."""
    work: Path
    model_html: Path
    netspec: Path
    pipeline: Path
    shaped_dir: Path
    manifest_digest: str
    run_dir: Path
    package: str


@dataclass(frozen=True)
class _Shaped:
    """What the shaping phase leaves: the manifest, fitted.json, the manifest file's digest
    (X-2), and the layered model with its domain digest."""
    manifest: Mapping[str, Any]
    fitted: Mapping[str, Any]
    digest: str
    model: Path
    model_digest: str


# --------------------------------------------------------------------------- the base model
def _lexicon_one(raw: Mapping[str, Any], key: str, fallback: str) -> str:
    entry = (raw.get("lexicon") or {}).get(key) or {}
    return entry.get("one") or fallback


def _entity(gen: Any, raw: Mapping[str, Any]) -> ModelEntity:
    """The base model's one entity: the lane's, with the generator's words where it has them."""
    name, entity = raw["generator"], raw["entity"]
    row = _lexicon_one(raw, "row", entity.lower())
    label = (raw.get("target") or {}).get("column")
    overview = getattr(gen, "OVERVIEW", None) or (
        f"# {name}\n\n{raw.get('title', name)}.\n\nThe domain has one class, {entity}: every "
        f"property describes one {row} as it was recorded, and "
        + (f"`{label}` is the attribute the chain partitions on.\n" if label else
           "no property is singled out as a label: the chain reads the rows as they are.\n"))
    rationale = {"choice": f"One class, {entity}: one {row} is the unit of everything downstream.",
                 "alternatives": [f"{entity} and a second class for what groups the rows"],
                 "tradeoffs": "nothing the chain decides needs a second entity before the data is shaped",
                 "anchors": []}
    return ModelEntity(base_iri=f"http://example.org/{name}#", name=entity,
                       component=getattr(gen, "COMPONENT", None) or name, properties=properties_for(gen.SCHEMA),
                       overview=overview, rationale=rationale,
                       comment=getattr(gen, "ENTITY_COMMENT", None) or f"One {row} as it was recorded.")


# --------------------------------------------------------------------------- world model, memory
def pipeline_for(lane: Any, contract: Contract, manifest_digest: str, watched: str) -> WorldModel:
    """The orders world model with the lane's names: `<g>_pipeline`, the lane's serving mode
    (`online` for every lanekit lane), the feature contract read from the shaped manifest (keys
    excluded), and the column the drift finding names (F4) as what the Monitor watches."""
    raw = raw_of(lane)
    g = raw["generator"]
    label = contract.label
    mode = SERVING_MODES.get(str(raw.get("serving_mode") or "online"))
    if mode is None:
        raise FixtureError(f"ERROR: lane {raw.get('id')}: serving_mode {raw.get('serving_mode')!r} "
                           f"is not one of {sorted(SERVING_MODES)}")
    return WorldModel(name=f"{g}-{label or 'rows'}-pipeline", package=f"{g}_pipeline",
                      manifest_digest=manifest_digest, serving_mode=mode, input_dim=contract.input_dim,
                      label=label, partition_source="[geometry:partitions/chosen]" if label else "[user:stance/goal]",
                      monitoring=((f"{watched} drift", "[analysis:F4]"),),
                      shape=str(raw.get("shape") or "row"), task=effective_task(lane))


def _stance_value(value: Any) -> str:
    """A stance's value as the memory file writes it (JSON spelling for booleans and null)."""
    return json.dumps(value) if isinstance(value, bool) or value is None else str(value)


def _sample_stances(sample: Mapping[str, Any]) -> list[str]:
    """The stances a composed sample cites as the user's (context, target channel, batching,
    series, graph protocol)."""
    out = [f"sample {key}: {_stance_value(sample.get(key))} (user)"
           for key in ("context", "autoregressive", "batching")
           if str(sample.get(f"{key}_source", "")).startswith("user:")]
    if sample.get("series_source") == "user:stance/one-series":
        out.append("series: none — the rows are one single series (user)")
    elif sample.get("series_source") == "user:stance/series":
        out.append(f"series: {sample.get('series_by')} (user)")
    if str(sample.get("protocol_source", "")).startswith("user:"):
        out.append(f"graph protocol: {sample.get('protocol')} (user)")
    return out


def memory_for(lane: Any, spec: Mapping[str, Any]) -> str:
    """The memory file's Network stances: one line per choice the netspec cites as the user's
    (`user:stance/...`), then the lane's own stance lines. A loss stance is written only when
    the spec's loss is the user's, so the compiler's M2 rule reads exactly the stance it holds."""
    raw = raw_of(lane)
    stances = [f"family: {spec['family']['name']} (user)"]
    if any(str(b.get("source", "")) == "user:stance/blocks" for b in spec["blocks"]):
        stances.append("blocks: the lane's own composition, beyond the family's canonical blocks (user)")
    for key, value in (("loss", "name"), ("optimizer", "name"), ("schedule", "name"), ("init", "scheme")):
        if str(spec[key].get("source")) == f"user:stance/{key}":
            stances.append(f"{key}: {spec[key][value]} (user)")
    t = spec["training"]
    stances.append(f"training budget: {t['max_epochs']} epochs, patience {spec['early_stopping']['patience']}, "
                   f"{t['budget']['wall_clock_min']} minutes a run (user)")
    stances += _sample_stances(spec["data_contract"]["sample"])
    if spec["task"].get("source") == "user:stance/goal":
        rate = (raw.get("target") or {}).get("anomaly_rate")
        stances.append(f"goal: {raw.get('goal')} without a label"
                       + (f", expected anomaly rate {rate}" if rate is not None else "") + " (user)")
    stances += list(raw.get("stances") or [])
    lines = "".join(f"- {STANCE_DATE} | {raw['id']} | {text}\n" for text in stances)
    return "# domain-forge memory\n\n## Network stances\n\n" + lines


def _custom_blocks(blocks: Sequence[Mapping[str, Any]]) -> list[str]:
    out = []
    for block in blocks:
        params = block.get("params") or {}
        if block.get("type") == "custom":
            out.append(params["code_ref"])
        out.extend(_custom_blocks(params.get("inner") or []))
    return out


def _copy_custom(blocks: Sequence[Mapping[str, Any]], run_dir: Path) -> None:
    """Each custom block's module (lanekit/assets/<stem>.py) under the campaign root, where the
    netspec's `code_ref` resolves (netspec-contract §2)."""
    assets = Path(__file__).resolve().parent / "assets"
    for ref in _custom_blocks(blocks):
        source = assets / Path(ref).name
        if not source.is_file():
            raise FixtureError(f"ERROR: custom block {ref!r}: lanekit/assets has no {Path(ref).name}")
        target = run_dir / ref
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


# --------------------------------------------------------------------------- the phases
def _checked(raw: Mapping[str, Any], generator: Any) -> Any:
    """The lane's generator, refused unless the lane and its SCHEMA agree — before any file is
    written."""
    gen = generator if generator is not None else _generator(raw)
    try:
        ColumnPlan.of(gen.SCHEMA, raw)
    except ValueError as exc:
        raise FixtureError(f"ERROR: {exc}") from exc
    return gen


def _raw_rows(gen: Any, raw: Mapping[str, Any], ws: Workspace) -> RawTable:
    """The generator's rows, written to `<g>.csv` and read back as the shaper reads them."""
    import pandas as pd

    path = ws.work / f"{ws.name}.csv"
    gen.make(int(raw["seed"]), int(raw["rows"])).to_csv(path, index=False)
    frame = pd.read_csv(path, low_memory=False)
    print(f"  fixture| raw {ws.name}: {len(frame)} rows x {frame.shape[1]} columns", flush=True)
    return RawTable(path, frame)


def _shape_lane(gen: Any, lane: Any, layers: LayerWriter, table: RawTable) -> _Shaped:
    """The base model, its geometry and analysis layers, the recipe run through the shaper and
    its checks, then the shape layer: the layered model the netspec cites."""
    raw, ws, real = raw_of(lane), layers.ws, layers.real
    base = compose_base_model(write_model_source(ws.work / f"{ws.name}.model-source", _entity(gen, raw)), ws.skills)
    evidence = ws.geometry_evidence(table.path, raw["seed"]) if real else {}
    geo = layers.geometry(base, geometry_payload(gen.SCHEMA, raw, table, evidence))
    series = (raw.get("window") or {}).get("series_by")
    engine = ws.analysis_evidence(table.path, geo, series) if real else None
    ana = layers.analysis(geo, analysis_payload(gen.SCHEMA, raw, table, engine))
    recipe_path = ws.work / f"{ws.name}.recipe.json"
    recipe_path.write_text(json.dumps(gen.RECIPE(table.path, ws.shaped, lane), indent=2), encoding="utf-8")
    verification = ws.shape(recipe_path, ana, int(raw["seed"]))
    manifest = json.loads((ws.shaped / "manifest.json").read_text(encoding="utf-8"))
    fitted = json.loads((ws.shaped / "fitted.json").read_text(encoding="utf-8"))
    digest = sha_file(ws.shaped / "manifest.json")
    model = layers.shape(ana, shape_payload(manifest, digest, verification, ws.shaped, real))
    if model.name != model_name(raw):
        raise FixtureError(f"ERROR: the layered model is {model.name}, the composer names {model_name(raw)}")
    return _Shaped(manifest, fitted, digest, model, layers.writer.domain_digest(model.read_text(encoding="utf-8")))


def _world_model(gen: Any, raw: Mapping[str, Any], ws: Workspace, shaped: _Shaped) -> tuple[Path, WorldModel]:
    """`blueprint-runs/<ts>/{pipeline.json, blueprint.md}` for the lane (pipeline_for)."""
    contract = Contract.from_manifest(shaped.manifest, raw)
    world = pipeline_for(raw, contract, shaped.digest, findings_for(gen.SCHEMA, raw)["F4"].columns[0])
    runs = ws.work / "blueprint-runs" / BLUEPRINT_TS
    runs.mkdir(parents=True, exist_ok=True)
    pipeline = runs / "pipeline.json"
    pipeline.write_text(json.dumps(world.pipeline(), indent=2), encoding="utf-8")
    (runs / "blueprint.md").write_text(world.blueprint_md(), encoding="utf-8")
    return pipeline, world


def _campaign(raw: Mapping[str, Any], ws: Workspace, spec: Mapping[str, Any]) -> tuple[Path, Path]:
    """(netspec.json, the campaign root): the netspec, the memory file, the recorded /architect
    run, and each custom block's module under the campaign root."""
    netspec = ws.work / "netspec.json"
    netspec.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    memory = ws.work / ".claude" / "domain-forge-memory.md"
    memory.parent.mkdir(parents=True, exist_ok=True)
    memory.write_text(memory_for(raw, spec), encoding="utf-8")
    target = ws.work / ".claude" / "architect-runs" / Path(ARCHITECT_RUN).name / "findings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ws.script(ARCHITECT_RUN + "/findings.json"), target)
    run_dir = ws.work / "vega-runs" / RUN_TS
    run_dir.mkdir(parents=True, exist_ok=True)
    _copy_custom(spec["blocks"], run_dir)
    return netspec, run_dir


# --------------------------------------------------------------------------- make_lane
def make_lane(lane: Any, work: Path, skills: Path, generator: Any = None,
              composer: Composer | None = None) -> LaneFixture:
    """Build the lane's fixture under `work` (module docstring). `generator` stands in for
    `lanekit.datasets.GENERATORS[lane.generator]` when given (a module or object keeping the
    generator protocol); `composer` carries the budget of the lanes file the lane came from."""
    raw = raw_of(lane)
    gen = _checked(raw, generator)
    ws = Workspace(Path(skills).resolve(), Path(work).resolve(), raw["generator"]).prepare()
    layers = LayerWriter(ws, bool((raw.get("expect") or {}).get("real_subchain")))
    shaped = _shape_lane(gen, lane, layers, _raw_rows(gen, raw, ws))
    pipeline, world = _world_model(gen, raw, ws, shaped)
    spec = (composer or Composer()).compose(raw, shaped.manifest, shaped.fitted, shaped.model_digest, pipeline)
    netspec, run_dir = _campaign(raw, ws, spec)
    parts = ", ".join(f"{k} {v['rows']}" for k, v in shaped.manifest["outputs"]["files"].items())
    print(f"OK: fixture for {raw['id']} written to {ws.work} ({shaped.model.name}; parts {parts})", flush=True)
    return LaneFixture(ws.work, shaped.model, netspec, pipeline, ws.shaped, shaped.digest, run_dir, world.package)


def _generator(raw: Mapping[str, Any]) -> Any:
    name = raw["generator"]
    if name in contracts.EXTERNAL_GENERATORS:
        raise FixtureError(f"ERROR: generator {name!r} is built by {contracts.EXTERNAL_GENERATORS[name]}, "
                           "not by lanekit.fixture (its lane runs another driver)")
    try:
        return datasets.load(name)
    except datasets.GeneratorError as exc:
        raise FixtureError(f"ERROR: {exc}") from exc


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    """`python -m lanekit.fixture --lane L05 --work DIR [--skills ROOT] [--lanes-file F]`: build
    one lane's fixture. The lane and its budget come from `--lanes-file` (the tree's own
    lanes.json by default); the chain's tools run from `--skills`; the catalogs the netspec is
    composed against are always this lanekit's own tree's."""
    from .registry import LANES_FILE, load_lanes

    ap = argparse.ArgumentParser(prog="python -m lanekit.fixture", description=main.__doc__)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--skills", default=str(contracts.SKILLS_ROOT))
    ap.add_argument("--lanes-file", default=str(LANES_FILE))
    a = ap.parse_args(argv)
    lanes_file = Path(a.lanes_file)
    try:
        fx = make_lane(load_lanes(lanes_file)[a.lane], Path(a.work), Path(a.skills),
                       composer=Composer.for_lanes_file(lanes_file))
    except FixtureError as exc:
        print(str(exc).splitlines()[0])
        return 2
    print(json.dumps({k: str(v) for k, v in dataclasses.asdict(fx).items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
