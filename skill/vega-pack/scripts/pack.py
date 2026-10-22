#!/usr/bin/env python3
"""
pack.py — build the bundle: what /vega-eval measured, packaged to run without any skill.

    python3 pack.py --campaign <dir> --run <id> --shaped <dir> --pipeline <pipeline.json>
                    --bundle-dir <dir> [--eval <dir>] [--slug NAME] [--examples 50|synthetic]
                    [--include serving|full] [--serve|--no-serve] [--export onnx,torchscript]
                    [--ship-anyway "<reason>"] [--unattended] [--dataset-shaper-dir PATH]
                    [--re-emit-serving [--reason "<text>"]] [--no-inspect]
                    [--inspect-weights full|summary] [--monitor-plan <json>] [--demo-sample N]
                    [--service-title T] [--service-port N] [--row-noun W] [--service-description D]

(`--run-dir` is `--campaign` and `--out-dir` is `--bundle-dir`, under the names the skill uses.)

The order is the contract's, and the reason it is fixed is that each step is only meaningful
once the one before it has passed:

  1. **Prove it is the same model.** The chain recorded a digest for the resolved spec, the
     checkpoint and the recipe; the bundle carries whatever is on disk. A disagreement stops
     the pack, because a bundle built from a different file would wear numbers it did not
     earn. `fitted.json` has no upstream digest — the shaper's twin check is its test instead.
  2. **Take the transformer from the shaper.** `features/preprocess.py` is emitted by
     `dataset-shaper/scripts/shape.py --transform-only`, never written here: a hand-rolled
     preprocessing is a second implementation of the recipe, and it will drift from the one
     the network was trained on. The shaper's own twin check runs before the round trip.
  3. **Round-trip against the sealed predictions.** The bundle's own predict path is run on
     the raw rows behind *every* test row and compared with `predictions.test.parquet`. This
     is the definition of "the same model": raw rows -> preprocess -> network here equals
     shaped rows -> network there. It exposes nothing — the outputs are compared and
     discarded, no metric is computed, the seal is not touched.

This file is the composition of the run and nothing else. After the assembly (the records,
the verdict, the provenance, the slice, the transformer, the schema) every step is a phase
module with a fixed seam — it takes the frozen `PackContext` (and, after the round trip, the
loaded predictor) and returns its manifest record(s):

    output_record.records(ctx)             -> {"output", "sample"}
    twin check
    graph_pack.write(ctx)                  -> {"sample.reference"}  (graph/*; the predictor loads through it)
    serving_pack.render_adapters(ctx)      -> {"entrypoints"}    (Predictor, predict.py, tests)
    roundtrip_plan.run(ctx, predictor)     -> {"roundtrip"}
    examples_pack.write(ctx, predictor)    -> {"examples"}
    demo_pack.write(ctx, predictor)        -> {"examples.demo", "serve.demo_sample"}
    monitor_pack.render(ctx, predictor)    -> {"monitor"}
    serving_pack.render(ctx)               -> {"serve", "entrypoints"}  (CLIs + served files)
    examples_pack.write_request(ctx)       -> {}                 (examples/request.json)
    inspection_pack.check(ctx, predictor)  -> {"serve.inspection"}
    export_pack.write(ctx, predictor)      -> {"exports"}        (with --export)
    bundle_slice.finish(ctx); blueprint_record(ctx) -> {"blueprint"}   (the map, the gate)
    build_manifest(ctx, records)           # each record merged under its key
    reemit.finish(ctx, manifest)           # MANIFEST.json, or the re-emit proof and the swap

The story and the written report are not the bundle's: `/vega-show` builds them beside it, and
`--story-plan`, `--report-aux` and `--report-pdf` answer where they moved.

An unearned verdict needs `--ship-anyway "<reason>"`, which is recorded in the manifest and on
the card. The verdict travels with the bundle whatever it says.

Exit codes: 0 the bundle is built and the round trip passed; 2 an input or provenance refusal
(no bundle written); 3 the round trip or a check failed after the bundle was assembled.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bundle_inputs  # noqa: E402
import bundle_manifest  # noqa: E402
import bundle_slice  # noqa: E402
import demo_pack  # noqa: E402
import examples_pack  # noqa: E402
import export_pack  # noqa: E402
import graph_pack  # noqa: E402
import inspection_pack  # noqa: E402
import monitor_pack  # noqa: E402
import output_record  # noqa: E402
import pack_options  # noqa: E402
import reemit  # noqa: E402
import roundtrip_plan  # noqa: E402
import serving_pack  # noqa: E402
from pack_context import Campaign, PackContext, PackExit, out_ok, out_warn  # noqa: E402
from vega_pack_core import CATALOG, BundlePaths, slugify  # noqa: E402


def main(argv=None) -> int:
    try:
        options = pack_options.parse(argv, description=__doc__)
    except PackExit as refusal:
        return _refused(refusal, None, [])
    return pack(options)


def pack(options) -> int:
    """The bundle, built and proven; the exit code."""
    markers = []
    try:
        ctx, records = _assemble(options, markers)
    except PackExit as refusal:
        return _refused(refusal, None, markers)
    try:
        return _finish(ctx, _phases(ctx, records))
    except PackExit as refusal:
        return _refused(refusal, ctx.paths, markers)


def _refused(refusal, paths, markers) -> int:
    for line in refusal.lines:
        print(f"{refusal.level}: {line}")
    if refusal.keep_markers and paths is not None and paths.root.is_dir():
        bundle_manifest.write_markers(paths.root, markers)
    return refusal.code


# --------------------------------------------------------------------------- the assembly

def _assemble(options, markers):
    """Everything the phases read: the chain's records checked, the slice copied and proven,
    the transformer emitted, the raw rows read, the schema and the served identity derived."""
    campaign = Campaign.read(options)
    stances = _verdict_gate(options, campaign, markers)
    _check_sources(campaign)
    _check_provenance(campaign, markers)
    graph_pack.refuse_unserved(_read_spec(campaign.sources["netspec"]))
    monitor_pack.refuse_unfilled(options)
    paths = reemit.prepare(options, _bundle_paths(options, campaign), markers)
    copied = bundle_slice.copy_package_slice(campaign.src_pkg, paths.pkg, markers,
                                             options.include)
    bundle_slice.verify_code(campaign, paths, copied, markers)
    bundle_inputs.copy_inputs(campaign, paths)
    bundle_inputs.emit_transformer(options, paths, markers)
    campaign = _with_inputs(campaign, paths, markers)
    fitted = bundle_inputs.project_fitted(paths, campaign.fitted, options.include, markers)
    schema = output_record.write_schema(campaign, paths.root, markers)
    ident = serving_pack.identity(options, options.pipeline, schema)
    bundle_manifest.copy_records(options.model, paths, markers, campaign)
    ctx = PackContext(options=options, paths=paths, campaign=campaign,
                      package=campaign.package, ident=ident, schema=schema,
                      copied=tuple(copied), markers=markers, stances=stances)
    return ctx, {"chain.shape.fitted": fitted}


def _verdict_gate(options, campaign, markers) -> list:
    """The stances the verdict needs: none when it was earned, the user's reason otherwise."""
    verdict = campaign.verdict
    if verdict.get("earned"):
        out_ok("the evaluation earned its verdict; the bundle carries it either way", markers)
        return []
    if not options.ship_anyway:
        why = (f" — {verdict.get('strongest_baseline')} beats the network on the test part"
               if verdict.get("beaten_by_baseline") else
               f" — the interval against {verdict.get('strongest_baseline')} covers zero")
        if options.unattended:
            raise PackExit(2, "--unattended: the evaluation's verdict is NOT earned" + why
                           + "; no bundle is written. Shipping it needs --ship-anyway "
                             "\"<reason>\", a stance only a user gives.", level="WARN")
        raise PackExit(2, "the evaluation's verdict is NOT earned" + why
                       + ". Shipping it needs --ship-anyway \"<reason>\", which goes in the "
                         "manifest and on the model card; or take the baseline, or return to "
                         "/vega-train.")
    out_warn(f"shipping an unearned network by stance: {options.ship_anyway}", markers)
    return [f"user: ship although not earned vs {verdict.get('strongest_baseline')} — "
            f"{options.ship_anyway}"]


def _check_sources(campaign) -> None:
    sources, src_pkg = campaign.sources, campaign.src_pkg
    missing = [k for k, v in sources.items() if not Path(v).is_file()]
    if missing or not src_pkg.is_dir():
        raise PackExit(2, f"the bundle cannot be built: missing "
                          f"{', '.join(missing) or ''}"
                          f"{' and the package' if not src_pkg.is_dir() else ''} "
                          f"({', '.join(str(sources[k]) for k in missing)})")


def _check_provenance(campaign, markers) -> None:
    prov = bundle_manifest.check_provenance(campaign)
    problems = prov.disagreements()
    if problems:
        raise PackExit(2, (*problems, "refused before the bundle was written — what ships is "
                                      "what was evaluated"))
    out_ok("provenance: the spec, the checkpoint and the recipe are the files the chain "
           "recorded" + (f" (no upstream digest for {', '.join(prov.unverifiable())}; the "
                         "shaper's twin check is its test)" if prov.unverifiable() else ""),
           markers)


def _read_spec(path) -> dict:
    """The run's resolved spec, read before anything is written: a sample protocol the bundle
    cannot serve (a transductive graph) is refused while --bundle-dir is still untouched."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackExit(2, f"the run's resolved spec {path} cannot be read ({exc})") from exc


def _bundle_paths(options, campaign) -> BundlePaths:
    recipe = json.loads(campaign.sources["recipe"].read_text(encoding="utf-8"))
    dataset = Path(recipe.get("input", {}).get("path", "data")).stem
    slug = options.slug or slugify(dataset, campaign.label)
    return BundlePaths(root=Path(options.bundle_dir).resolve() / f"{slug}-{campaign.run_id}",
                       package=campaign.package)


def _with_inputs(campaign, paths, markers) -> Campaign:
    """The campaign with the bundle's recipe, spec, raw rows and whole fitted file."""
    recipe = json.loads((paths.root / "recipe.json").read_text(encoding="utf-8"))
    spec = json.loads((paths.root / "netspec.json").read_text(encoding="utf-8"))
    campaign = campaign.with_inputs(recipe, spec, None, None)
    bundle_inputs.check_feature_contract(campaign, markers)
    raw = bundle_inputs.read_raw(campaign, recipe)
    fitted = json.loads((paths.root / "fitted.json").read_text(encoding="utf-8"))
    return campaign.with_inputs(recipe, spec, raw, fitted)


# --------------------------------------------------------------------------- the phases

def _phases(ctx, records) -> dict:
    """Every phase in its fixed order, each record kept under its manifest key."""
    records.update(output_record.records(ctx))
    bundle_inputs.twin_check(ctx.options, ctx.campaign, ctx.paths, ctx.markers)
    records.update(graph_pack.write(ctx))
    records.update(serving_pack.render_adapters(ctx))
    predictor = _load_predictor(ctx, records)
    records.update(roundtrip_plan.run(ctx, predictor))
    records.update(examples_pack.write(ctx, predictor))
    records.update(demo_pack.write(ctx, predictor))
    records.update(monitor_pack.render(ctx, predictor))
    records.update(serving_pack.render(ctx))
    records.update(examples_pack.write_request(ctx))
    records.update(inspection_pack.check(ctx, predictor))
    records.update(export_pack.write(ctx, predictor))
    return records


def _load_predictor(ctx, records):
    """Import the bundle exactly as a user would, from its own directory and nothing else,
    against the manifest stub that pins the data files already written."""
    bundle_manifest.write_stub(ctx, records)
    saved = list(sys.path)
    sys.path.insert(0, str(ctx.paths.root))
    try:
        import importlib
        for name in list(sys.modules):
            if name == ctx.package or name.startswith(ctx.package + "."):
                del sys.modules[name]
        sys.modules.pop("predict", None)
        return importlib.import_module("predict").load(ctx.paths.root)
    except Exception as exc:                                          # noqa: BLE001
        raise PackExit(3, f"the bundle's own predictor does not load: "
                          f"{type(exc).__name__}: {exc}") from exc
    finally:
        sys.path[:] = saved


# --------------------------------------------------------------------------- the end

def _finish(ctx, records) -> int:
    """The map and the gate, the manifest, the markers; the bundle checked complete."""
    bundle_slice.finish(ctx)
    records.update(bundle_manifest.blueprint_record(ctx))
    manifest = bundle_manifest.build_manifest(ctx, records)
    paths = reemit.finish(ctx, manifest)
    bundle_manifest.write_markers(paths.root, ctx.markers)
    _sweep(paths, ctx.markers)
    missing_files = paths.missing(CATALOG["required_files"])
    if missing_files:
        raise PackExit(3, f"the bundle is missing {', '.join(missing_files)}")
    out_ok(f"bundle {paths.root} — every required file present, {len(manifest['files'])} "
           "digest(s) recorded", ctx.markers)
    return 0


def _sweep(paths, markers) -> None:
    """The round trip imported the package, which left this machine's bytecode in it. A bundle
    carrying __pycache__ ships compiled files for one interpreter version and one path, and a
    stale .pyc beside an edited .py is a debugging afternoon nobody asked for."""
    swept = 0
    for cache in sorted(paths.root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache, ignore_errors=True)
        swept += 1
    if swept:
        out_ok(f"{swept} __pycache__ director{'y' if swept == 1 else 'ies'} swept — the bundle "
               "ships source, not this machine's bytecode", markers)


if __name__ == "__main__":
    sys.exit(main())
