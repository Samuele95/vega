"""
inspection_pack — the bundle's inspection surface, proven at pack time, and the endpoints the
bundle serves declared against the blueprint (build plan I09, R-D14, R-F5; D3 §13;
inspect-contract §2, §6.2, §7; bundle-contract §2.4, §7.7).

    inspection_pack.check(ctx, predictor) -> {"serve.inspection": …,
                                              "blueprint.conformance.api_declared": …,
                                              "blueprint.conformance.declaration": …}
    inspection_check(root, predictor, tolerance, backend="pytorch") -> Opened
                                                  the self-test; PackExit(3) when not ready
    block_types(spec), inspection_files(pkg, module)          what the record lists

`serving_templates.render_package` rendered the backend's `package/inspection/*.py` into the
Predictor's module (none with `--no-inspect`) and mapped each file `kind: support`,
`used_by: ["Predictor"]`. This phase holds the map to that (C3 reads it: a rendered file without
its entry is added, and said so), then opens a `NetworkInspector` over the round-tripped
Predictor as the service's start-up opens one — on `examples/rows.csv` through the bundle's own
`_example_query`, at the round-trip tolerance `MANIFEST.roundtrip` will carry — and requires it
ready. Not ready is `ERROR: inspection: <why>` (exit 3): a defect of the backend's inspection
templates, fixed there and never in the bundle — except a one-class network whose checkpoint
lacks its centre, which names the upstream fix (the trainer). Ready, it pins the network digest
of both weights modes in `MANIFEST.serve.inspection` (the catalog's keys, in their order), which
the inspector checks whenever it opens again — at the service's start-up and in
`load_inspector` ("the network the service reads is not the one packed"). The hooks are removed
before the phase returns: the export traces the same network next.

`--no-inspect` records `enabled: false` over a bundle that holds no `inspection/` at all and
whose Predictor refuses `inspector()` (both are checked). `--no-inspect` and
`--inspect-weights summary` are the user's stances in `MANIFEST.stances`; the default (`full`)
is the record's `weights` and the OK line's — `MANIFEST.stances` is read downstream as the
conditions a bundle ships under, and a default is not one of them. The record is written in
every serving mode that carries the Predictor's module: `load_inspector` is the library path
`/vega-show` uses for a bundle without a service, and it re-verifies the same pin.

The declaration fork — every served endpoint looked up in the blueprint, R-G3 — is
`api_declaration.declare(ctx)`, run here, after the route modules are rendered and before the
conformance gate.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import api_declaration
from bundle_map import add_map_entries
from bundle_slice import bundle_modules
from pack_context import PackExit, out_ok, out_warn
from roundtrip_compare import tolerance_for
from roundtrip_inputs import Stored
from vega_pack_core import CATALOG, read_json

__all__ = ["CONTRACT", "Opened", "RECORD", "block_types", "check", "inspection_check",
           "inspection_files"]

RECORD = CATALOG["serve_record"]["inspection"]
MODES = tuple(RECORD["weights"])                      # ("full", "summary")
CONTRACT = "vega-forge/references/inspect-contract.md"
SOURCE = "the vega-pack inspection check"
TEMPLATES = "vega-forge/scripts/backends/{backend}/templates/serving/package/inspection/"
SVDD = "one-class centre"
SVDD_FIX = ("the centre is a buffer the one-class loss sets once and the ModelStore saves with "
            "the weights (svdd_center, svdd_center_set; R-B6): this checkpoint was trained "
            "without it — re-run /vega-train so the checkpoint carries it; the bundle never "
            "invents a centre")


# =========================================================================== the phase

def check(ctx, predictor) -> dict:
    """`MANIFEST.serve.inspection`, then the declaration fork's record."""
    records = {"serve.inspection": _surface(ctx, predictor)}
    records.update(api_declaration.declare(ctx))
    return records


def _surface(ctx, predictor) -> dict:
    module = _predictor_module(ctx)
    files = inspection_files(ctx.paths.pkg, module)
    if not ctx.options.inspect:
        return _without(ctx, predictor, files, module)
    if not files:
        out_warn(f"the {ctx.campaign.backend_name} backend ships no inspection templates "
                 "(templates/serving/package/inspection/): the bundle has no inspection surface, "
                 "and MANIFEST.serve.inspection says enabled: false", ctx.markers)
        return _record(False, ctx, (), None)
    _mapped(ctx, files)
    opened = inspection_check(ctx.paths.root, predictor, _tolerance(ctx),
                              ctx.campaign.backend_name)
    _say_opened(ctx, opened)
    if ctx.options.inspect_weights != "full":
        ctx.stances.append(f"user: --inspect-weights {ctx.options.inspect_weights} — "
                           "/v1/network publishes weight summaries only; raw tensors are "
                           "withheld and /v1/network/tensor answers 404")
    return _record(True, ctx, files, opened)


def _predictor_module(ctx) -> str:
    """The Predictor's module, as `render_package` placed it (the port's file, else the map's
    serving module)."""
    port = ctx.port("Predictor")
    adapter = port.get("file") or f"{bundle_modules(ctx.paths.pkg)['serving']}/predict.py"
    return port.get("module") or adapter.split("/")[0]


def inspection_files(pkg, module) -> list:
    """The rendered inspection modules, package-relative (`<module>/inspection/<name>.py`)."""
    where = Path(pkg) / module / "inspection"
    return sorted(p.relative_to(pkg).as_posix() for p in where.glob("*.py")) \
        if where.is_dir() else []


def _without(ctx, predictor, files, module) -> dict:
    """--no-inspect: no inspection/ in the bundle, a Predictor that refuses inspector()."""
    if files:
        raise PackExit(3, f"--no-inspect, yet the bundle carries {module}/inspection/ "
                          f"({len(files)} file(s)); a bundle packed without inspection holds "
                          "none of its package", keep_markers=True)
    try:
        predictor.inspector()
    except LookupError:
        pass
    else:
        raise PackExit(3, "--no-inspect, yet the bundle's Predictor built an inspector; it must "
                          "refuse (LookupError) when its module holds no inspection/",
                          keep_markers=True)
    if ctx.options.inspect_weights != "full":
        out_warn(f"--inspect-weights {ctx.options.inspect_weights} with --no-inspect: there is "
                 "no inspection surface for it to restrict, so it was not applied", ctx.markers)
    ctx.stances.append("user: --no-inspect — the bundle ships without inspection: /v1/network "
                       "and /v1/explain are not served and load_inspector() refuses")
    out_ok(f"inspection: --no-inspect — no {module}/inspection/ in the bundle, its Predictor "
           "refuses inspector(), MANIFEST.serve.inspection.enabled is false", ctx.markers)
    return _record(False, ctx, (), None)


def _record(enabled, ctx, files, opened) -> dict:
    """The record in the catalog's key order (`serve_record.inspection.keys`)."""
    status = opened.status if opened is not None else None
    values = {"schema": RECORD["schema"], "enabled": bool(enabled),
              "weights": ctx.options.inspect_weights,
              "network_schema": RECORD["network_schema"],
              "explanation_schema": RECORD["explanation_schema"], "files": list(files),
              "block_types": block_types(ctx.spec),
              "opaque": list(((status or {}).get("coverage") or {}).get("opaque") or []),
              "chain": opened.chain if opened is not None else None,
              "network_digest": dict(opened.digests) if opened is not None else None,
              "self_test": status, "source": CONTRACT}
    unknown = [key for key in RECORD["keys"] if key not in values]
    if unknown:
        raise PackExit(2, f"vega-pack catalog serve_record.inspection.keys names "
                          f"{', '.join(unknown)}, which this phase does not write")
    return {key: values[key] for key in RECORD["keys"]}


def block_types(spec) -> list:
    """Every block type of the spec, nested ones (a composite's `inner`, `branches`) included."""
    found, stack = set(), list((spec or {}).get("blocks") or ())
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            if item.get("type"):
                found.add(str(item["type"]))
            stack.extend(v for v in (item.get("params") or {}).values()
                         if isinstance(v, (list, dict)))
    return sorted(found)


def _mapped(ctx, files) -> None:
    """Each inspection file a `kind: support` entry of the copied map (C3, inspect-contract §2)."""
    doc, err = read_json(ctx.paths.pkg / "blueprint-map.json", "blueprint-map.json")
    if err:
        raise PackExit(3, f"{err}; the inspection modules cannot be mapped", keep_markers=True)
    entries = {e.get("file"): e for e in (doc.get("entries") or []) if isinstance(e, dict)}
    wrong = [f for f in files if f in entries and entries[f].get("kind") != "support"]
    if wrong:
        raise PackExit(3, f"blueprint-map.json maps {', '.join(wrong)} as something other than "
                          "kind: support; the inspection modules implement no port",
                          keep_markers=True)
    missing = [f for f in files if f not in entries]
    if missing:
        add_map_entries(ctx, SOURCE, files=missing)
        out_warn(f"blueprint-map.json lacked {len(missing)} inspection file(s) "
                 f"({', '.join(missing)}); added as kind: support, used_by Predictor",
                 ctx.markers)


def _tolerance(ctx) -> float:
    """The round trip's tolerance (MANIFEST.roundtrip.tolerance.value, computed as it is), the
    one the inspector's fidelity is held to wherever the bundle opens it."""
    stub = json.loads(ctx.paths.manifest.read_text(encoding="utf-8"))
    stored = Stored.read(ctx.campaign.eval_dir, ctx.campaign.metrics)
    return float(tolerance_for(stored.frame, stub.get("output") or {}).value)


def _say_opened(ctx, opened) -> None:
    status = opened.status
    coverage, fid, rec = (status.get(k) or {} for k in ("coverage", "fidelity", "recompute"))
    weights = ctx.options.inspect_weights
    out_ok(f"inspection: the inspector opened on examples/rows.csv as the service will — "
           f"{coverage.get('nodes')} node(s) of {len(coverage.get('types') or [])} type(s), "
           f"{len(coverage.get('opaque') or [])} opaque; fidelity worst "
           f"{fid.get('max_abs_diff')} within {fid.get('tolerance')} over {fid.get('samples')} "
           f"sample(s); network digest pinned for both weights modes (full "
           f"{opened.digests['full'][:19]}…, summary {opened.digests['summary'][:19]}…); "
           f"weights published: {weights}"
           + (" (the default; --inspect-weights summary withholds the raw tensors)"
              if weights == "full" else ""), ctx.markers)
    if not rec.get("ok"):
        out_warn(f"inspection: the node checks did not all hold (recompute worst "
                 f"{rec.get('worst')}); the explanations are still the network's own numbers, "
                 "and only the page's 'recomputed' tick is at stake", ctx.markers)


# =========================================================================== the self-test

@dataclass(frozen=True)
class Opened:
    """A ready inspector's account: its status (inspect-contract §6.2), the network digest per
    weights mode, and whether the network has a dense chain."""

    status: dict
    digests: dict
    chain: bool


def inspection_check(root, predictor, tolerance, backend="pytorch") -> Opened:
    """The inspector over `predictor`, opened on the bundle's example rows at `tolerance` and
    closed again; `PackExit(3)` naming why it is not ready and where that is fixed."""
    inspector = _inspector(predictor, tolerance)
    try:
        status = _opened(inspector, _example(root), backend)
        bodies = {mode: json.loads(inspector.network_bytes(mode)) for mode in MODES}
    finally:
        inspector.close()
    digests = {mode: str(body["network_digest"]) for mode, body in bodies.items()}
    if status.get("network_digest") != digests["full"]:
        raise PackExit(3, f"inspection: the status names the network digest "
                          f"{status.get('network_digest')}, the published body "
                          f"{digests['full']}", keep_markers=True)
    return Opened(status, digests, bodies["summary"].get("chain") is not None)


def _inspector(predictor, tolerance):
    """`NetworkInspector(model, spec, answers, output, tolerance)` from the Predictor's own
    module, with the arguments `Predictor.inspector()` passes but the tolerance, which is the
    one MANIFEST.roundtrip will carry (the stub the predictor loaded against has none yet)."""
    package = type(predictor).__module__.rpartition(".")[0]
    try:
        inspection = importlib.import_module(f"{package}.inspection")
        output = dict((predictor.info() or {}).get("output") or {})
        return inspection.NetworkInspector(predictor.model, predictor.records.spec, predictor,
                                           output, float(tolerance))
    except Exception as exc:                                          # noqa: BLE001
        raise PackExit(3, f"inspection: the inspector of {package} cannot be built: "
                          f"{type(exc).__name__}: {exc}", keep_markers=True) from exc


def _example(root):
    """The query the service opens on: the bundle's own `_example_query` over
    `examples/rows.csv` (the bundle CLI's, which `load_inspector` calls)."""
    try:
        return _bundle_cli(root)._example_query(Path(root).resolve())
    except Exception as exc:                                          # noqa: BLE001
        raise PackExit(3, f"inspection: the bundle's example rows make no query to open the "
                          f"inspector on: {type(exc).__name__}: {exc}", keep_markers=True) from exc


def _opened(inspector, example, backend) -> dict:
    try:
        status = inspector.open(example).to_dict()
    except Exception as exc:                                          # noqa: BLE001
        raise PackExit(3, f"inspection: the inspector could not open: {type(exc).__name__}: "
                          f"{exc}", keep_markers=True) from exc
    if not status.get("ready"):
        raise PackExit(3, (f"inspection: {status.get('why')}", _fix(status.get("why"), backend)),
                       keep_markers=True)
    return status


def _fix(why, backend) -> str:
    if SVDD in str(why or ""):
        return f"inspection: {SVDD_FIX}"
    return ("inspection: a defect of the backend's inspection templates — fixed in "
            + TEMPLATES.format(backend=backend) + ", never in the bundle")


def _bundle_cli(root):
    """The bundle's own `predict.py` module (the one pack loaded, else loaded from its file)."""
    root = Path(root).resolve()
    loaded = sys.modules.get("predict")
    if loaded is not None and Path(getattr(loaded, "__file__", "") or ".").resolve() == \
            root / "predict.py":
        return loaded
    spec = importlib.util.spec_from_file_location("_vega_pack_bundle_cli", root / "predict.py")
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module
