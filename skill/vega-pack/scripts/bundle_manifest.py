"""
bundle_manifest — what the bundle says about itself: `MANIFEST.json`, its file roles, the
chain record, the records it copies, the provenance it checked and the conformance gate.

    check_provenance(campaign) -> Provenance            before anything is written
    copy_records(model, paths, markers, campaign=None) -> {path: digest}
    write_stub(ctx, records)                            the manifest the round trip loads against
    refresh_stub(ctx)                                   its pins re-read, for a file written later
    blueprint_record(ctx) -> {"blueprint": …}           the mechanical conformance gate
    build_manifest(ctx, records) -> manifest            every phase's record merged in
    check_entrypoints(manifest)                         every entry point a file it pins

Every file the bundle carries is pinned by digest except three: `MANIFEST.json` (a digest of
itself would change the file it is the digest of), `README.md` (prose about the bundle, which a
user may correct) and `markers.json` (the run's account, written after the manifest). Each
pinned file has a role, and every reader verifies by role.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from bundle_inputs import without_membership
from pack_context import SKILLS, PackExit, merge_records, out_ok, out_warn
from vega_pack_core import Provenance, digest_file, now

__all__ = ["DATA_FILES", "UNPINNED", "blueprint_record", "build_manifest", "chain_record",
           "check_entrypoints", "check_provenance", "conformance_gate", "copy_records",
           "file_roles", "pinned_files", "refresh_stub", "weights_version", "write_markers",
           "write_stub"]

SCHEMA = "vega/bundle@1"

# The data files the bundle's code reads. Everything the pack ships gets a role in
# MANIFEST.roles, and every reader verifies by role — the list lives here, once, where the files
# are written, not as literals in the templates that read them. `graph/` (the reference graph
# of a graph bundle) and `exports/` are data by prefix.
DATA_FILES = ("netspec.json", "fitted.json", "schema.json", "monitor-reference.json")
DATA_PREFIXES = ("exports/", "graph/")
UNPINNED = ("MANIFEST.json", "README.md", "markers.json")
LAYERS = ("geometry", "analysis", "shape", "architecture", "training", "evaluation")


# --------------------------------------------------------------------------- provenance

def check_provenance(campaign) -> Provenance:
    """What the chain recorded against what is on disk, file by file."""
    row = campaign.evaluation["run"]          # the ledger's row for the evaluated run
    sources = campaign.sources
    prov = Provenance()
    prov = prov.with_check("the resolved netspec", row.get("spec_digest"),
                           digest_file(sources["netspec"]))
    prov = prov.with_check("the checkpoint", row.get("checkpoint_digest"),
                           digest_file(sources["weights"]))
    prov = prov.with_check("fitted.json", None, digest_file(sources["fitted"]))
    return prov


# --------------------------------------------------------------------------- roles and pins

def file_roles(files, package) -> dict:
    """Each pinned file's role: `code` (imported), `data` (read to answer), `example` (the
    shipped rows and the self-test evidence), `record` (provenance, read by nobody at run time)
    or `build` (how to install and test it)."""
    roles = {"code": [], "data": [], "example": [], "record": [], "build": []}
    for rel in sorted(files):
        if rel in ("predict.py", "train.py", "evaluate.py") or \
                rel.startswith(("serve/", f"{package}/")):
            role = "code"
        elif rel in DATA_FILES or re.fullmatch(r"weights\.[A-Za-z0-9]+", rel) or \
                rel.startswith(DATA_PREFIXES):
            role = "data"
        elif rel.startswith("examples/"):
            role = "example"
        elif rel.startswith("records/") or rel == "recipe.json":
            role = "record"
        else:
            role = "build"
        roles[role].append(rel)
    return roles


def pinned_files(root) -> dict:
    """Every file under the bundle root but the three that are not pinned, by digest."""
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        if rel not in UNPINNED:
            files[rel] = digest_file(path)
    return dict(sorted(files.items()))


def weights_version(path) -> str | None:
    """The weights digest the checkpoint carries — the model's identity, which the store
    recomputes and compares on every load."""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob.get("weights_digest")


def write_stub(ctx, records) -> None:
    """The manifest the predictor is loaded against before the round trip: it pins the data
    files already written, so the predictor verifies them as it will in the finished bundle,
    and it carries the output and sample records the Predictor reads at construction. The
    manifest proper replaces it after the round trip."""
    root = ctx.paths.root
    stub_files = _data_pins(root)
    stub = {"package": ctx.package, "files": stub_files,
            "roles": file_roles(stub_files, ctx.package),
            "model_version": weights_version(root / "weights.pt"),
            "input_schema": "schema.json", "label": ctx.campaign.label}
    merge_records(stub, {k: v for k, v in records.items()
                         if k.split(".")[0] in ("output", "sample", "graph")})
    ctx.paths.manifest.write_text(json.dumps(stub, indent=2), encoding="utf-8")


def refresh_stub(ctx) -> None:
    """The stub's pins re-read from disk: a data file a phase wrote after the predictor loaded
    (the monitor reference) is verified by the bundle code that reads it next — the pack-time
    monitor smoke — as it will be in the finished bundle."""
    stub = json.loads(ctx.paths.manifest.read_text(encoding="utf-8"))
    stub["files"] = _data_pins(ctx.paths.root)
    stub["roles"] = file_roles(stub["files"], ctx.package)
    ctx.paths.manifest.write_text(json.dumps(stub, indent=2), encoding="utf-8")


def _data_pins(root) -> dict:
    """The data files written so far and the weights, by digest."""
    pins = {name: digest_file(root / name)
            for name in DATA_FILES + ("weights.pt",) if (root / name).is_file()}
    for prefix in DATA_PREFIXES:
        for path in sorted((root / prefix).rglob("*")) if (root / prefix).is_dir() else ():
            if path.is_file():
                pins[path.relative_to(root).as_posix()] = digest_file(path)
    return pins


# --------------------------------------------------------------------------- the records

def copy_records(model_path, paths, markers, campaign=None) -> dict:
    """The chain's layer data, copied into the bundle as `records/<layer>.json`.

    The bundle could already reproduce the network's predictions; it could not say where the
    network came from without the model page beside it. These records are what make it
    self-describing: the model card splices its figures from them, and `predict.ipynb` — the
    card's executable twin — reads the same files rather than a copy of the card's conclusions.

    They are data, not code: nothing in the bundle imports them, and a bundle whose records
    were deleted still predicts. What it loses is the ability to explain itself.

    One record is about the packed network rather than the chain's page: the architecture
    layer's `smoke` (per_block, params_total) is the forged spec's. When `campaign`'s selected
    run is a ladder rung or a trial that re-emitted its package, the bundle ships that run's
    own `runs/<run>/smoke.json` there instead (`smoke_of` says whose it is, with its digest
    and the forged counts it replaced), so the record describes the network `netspec.json` and
    `weights.pt` hold — the trainer's own rule for a run's parameter count.
    """
    if not model_path or not Path(model_path).is_file():
        out_warn("no --model given, so the bundle carries no records/ and its notebook and "
                 "card will have nothing to explain it with", markers)
        return {}
    page = Path(model_path).read_text(encoding="utf-8")
    target = paths.root / "records"
    target.mkdir(parents=True, exist_ok=True)
    written = {}
    for layer in LAYERS:
        doc = _layer_data(page, layer, markers)
        if doc is None:
            continue
        if layer == "architecture" and campaign is not None:
            doc = _with_run_smoke(doc, campaign, markers)
        if layer == "shape":
            doc = without_membership(doc, markers)
        path = target / f"{layer}.json"
        path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written[f"records/{layer}.json"] = digest_file(path)
    out_ok(f"records/ — {len(written)} layer(s) copied "
           f"({', '.join(sorted(k.split('/')[1][:-5] for k in written))}); "
           "the bundle can now say where it came from, not only what it predicts", markers)
    return written


def _with_run_smoke(doc, campaign, markers):
    """The architecture record with the selected run's own smoke when that run re-emitted its
    package (a rung, a trial); unchanged when it trained the campaign's package."""
    run = campaign.run_id
    own = campaign.run_dir / "smoke.json"
    if not own.is_file():
        if (campaign.run_dir / campaign.package).is_dir():
            out_warn(f"run {run} re-emitted its package but carries no smoke.json; "
                     "records/architecture.json keeps the forged spec's smoke, which may not "
                     "count this network", markers)
        return doc
    try:
        smoke = json.loads(own.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackExit(2, f"run {run}'s smoke.json cannot be read ({exc}); the bundle would "
                          "describe a network it does not ship") from exc
    if not isinstance(smoke, dict) or smoke.get("params_total") is None \
            or not isinstance(smoke.get("per_block"), list):
        raise PackExit(2, f"run {run}'s smoke.json records no params_total and per_block; the "
                          "bundle would describe a network it does not ship")
    forged = doc.get("smoke") or {}
    out = dict(doc, smoke=smoke)
    out["smoke_of"] = {
        "run": run, "kind": campaign.ledger_row.get("kind"),
        "source": f"runs/{run}/smoke.json", "digest": digest_file(own),
        "forged": {"params_total": forged.get("params_total"),
                   "per_block": forged.get("per_block")},
        "note": "the selected run re-emitted its package: smoke is that network's (the one "
                "netspec.json and weights.pt hold); netspec stays the forged spec the run is a "
                "delta of"}
    out_ok(f"records/architecture.json — smoke of run {run} ({smoke.get('params_total')} "
           f"parameter(s); the forged spec has {forged.get('params_total')}): the record "
           "counts the network the bundle ships", markers)
    return out


def _layer_data(page, layer, markers):
    match = re.search(
        r'<script id="layer-' + layer + r'-data" type="application/json">([\s\S]*?)</script>',
        page)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        out_warn(f"the {layer} layer's data did not parse; it is not copied", markers)
        return None


def chain_record(ctx) -> dict:
    """Where every number in this bundle came from, in one block.

    Copied from the layers rather than recomputed: the point of the record is that a reader
    can follow it back, and a number this stage derived again would be this stage's number.
    """
    campaign, model_path = ctx.campaign, ctx.options.model
    metrics = campaign.metrics or {}
    primary = metrics.get("primary")
    net = ((metrics.get("models") or {}).get("network") or {}).get(primary) or {}
    threshold = campaign.threshold or {}
    return {
        "model": Path(model_path).name if model_path else None,
        "model_digest": digest_file(model_path) if model_path
        and Path(model_path).is_file() else None,
        # the FeatureContract's identity is the shaped manifest file's digest, and the Recipe's
        # is the digest the shaper recorded in that manifest — both read, neither derived
        "shape": {"manifest_digest": digest_file(campaign.shaped_dir / "manifest.json"),
                  "recipe_digest": campaign.shaped_manifest.get("recipe_digest")},
        "architecture": {"netspec_digest": campaign.ledger_row.get("spec_digest")},
        "training": {"selection_rule": ((campaign.ledger.get("selection") or {}).get("rule"))},
        "evaluation": {
            "seal_exposures": campaign.seal.get("exposures"),
            "task": metrics.get("task"),
            "primary": {"name": primary, "value": net.get("value"), "ci": net.get("ci")},
            "verdict": campaign.verdict,
            "threshold": threshold.get("value"),
            "threshold_source": threshold.get("source")},
        "label": campaign.label,
    }


def _usage(ctx) -> dict:
    """The lines the README, the card and the notebook all repeat, written once."""
    name = ctx.paths.root.name
    usage = {"cli": f"python predict.py --bundle {name} --input rows.csv --output preds.csv",
             "module": f"from predict import load; load('{name}').predict(df)"}
    if ctx.options.include == "full":
        usage["train"] = "python train.py --shaped path/to/shaped --out runs/mine"
        usage["evaluate"] = "python evaluate.py --part valid --shaped path/to/shaped"
    return usage


# --------------------------------------------------------------------------- the gate

def conformance_gate(ctx) -> dict:
    """The mechanical half of the bundle's conformance gate.

    The round trip proves the numbers; this proves the architecture. A bundle that passes only
    the first is a working network *beside* the blueprint rather than in it — and since the
    bundle becomes the user's codebase, an import that runs outside the module map is a
    boundary they inherit without being told.

    The judgement half is `/architect --report-only --baseline`, invoked through the Skill
    tool by the skill rather than from here; its absence is recorded, never assumed to pass.
    """
    pipeline, mode, markers = ctx.options.pipeline, ctx.options.include, ctx.markers
    if not pipeline:
        out_warn("no --pipeline given, so the bundle's conformance gate did not run; the "
                 "layer records that rather than a pass", markers)
        return {"mechanical": "not run", "why": "no pipeline.json was given"}
    checker = SKILLS / "vega-forge" / "scripts" / "conformance.py"
    if not checker.is_file():
        out_warn(f"the conformance checker is not at {checker}; the gate did not run", markers)
        return {"mechanical": "not run", "why": "vega-forge's conformance.py is not there"}
    # `--bundle` tells the checker that some ports are legitimately `out of scope`. A full
    # bundle has none, so it is checked as the complete package it is — the stricter of the
    # two, and the right one for code a user is about to train with.
    # `--entrypoints` holds what the bundle writes beside the package — `predict.py`, `serve/`,
    # `tests/` — to the same size discipline (C11–C13), at the gate's default severity, error.
    cmd = [sys.executable, str(checker), "--package", str(ctx.paths.pkg),
           "--pipeline", str(pipeline), "--map", str(ctx.paths.pkg / "blueprint-map.json"),
           "--entrypoints", str(ctx.paths.root)]
    if mode != "full":
        cmd.append("--bundle")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in (proc.stdout or "").splitlines()
             if ln.startswith(("OK:", "WARN:", "ERROR:"))]
    if proc.returncode == 0:
        out_ok(f"the bundle passes the mechanical conformance gate ({len(lines)} check(s)) — "
               "its imports run along the blueprint's module arrows and every port it keeps "
               "has exactly one adapter", markers)
        return {"mechanical": "pass", "checks": len(lines), "include": mode}
    for ln in lines:
        if ln.startswith("ERROR:"):
            markers.append("WARN: conformance: " + ln[len("ERROR:"):].strip())
    return {"mechanical": "fail", "checks": len(lines), "include": mode,
            "why": "; ".join(ln for ln in lines if ln.startswith("ERROR:"))[:500]}


def blueprint_record(ctx) -> dict:
    conformance = conformance_gate(ctx)
    pipeline = ctx.options.pipeline
    return {"blueprint": {"run": str(pipeline) if pipeline else None,
                          "map": "blueprint-map.json",
                          "conformance": dict(conformance, architect=None)}}


# --------------------------------------------------------------------------- the manifest

def build_manifest(ctx, records) -> dict:
    """The manifest: the bundle's identity and pins, then every phase's record under its key."""
    campaign = ctx.campaign
    manifest = {
        "schema": SCHEMA, "slug": ctx.paths.root.name[: -(len(campaign.run_id) + 1)],
        "run": campaign.run_id, "campaign": campaign.dir.name,
        # the emitted package's name, said once here so nothing downstream has to guess it
        # from the shape of the file list
        "package": ctx.package,
        "chain": chain_record(ctx),
        "files": pinned_files(ctx.paths.root),
        "backend": campaign.backend,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "created_at": now(),
        "input_schema": "schema.json",
        "label": campaign.label,
        "model_version": weights_version(ctx.paths.root / "weights.pt"),
        "exports": [],
        "include": ctx.options.include,
        "entrypoints": ["predict.py"],
        "usage": _usage(ctx),
        "stances": list(ctx.stances),
    }
    merge_records(manifest, records)
    check_entrypoints(manifest)
    manifest["records"] = sorted(k for k in manifest["files"] if k.startswith("records/"))
    manifest["roles"] = file_roles(manifest["files"], ctx.package)
    return manifest


def check_entrypoints(manifest) -> None:
    """An entry point is a file the bundle carries and pins: `serving_pack` lists the served
    ones in c849f8b's order (amendment A1), and a name nothing wrote is a claim the bundle
    breaks."""
    absent = [rel for rel in manifest["entrypoints"] if rel not in manifest["files"]]
    if absent:
        raise PackExit(3, f"MANIFEST.entrypoints names {', '.join(absent)}, which the bundle "
                          "does not carry")


def write_markers(root, markers) -> None:
    (Path(root) / "markers.json").write_text(
        json.dumps({"markers": markers}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
