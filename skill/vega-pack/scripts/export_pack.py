"""
export_pack — the network exported to the formats `--export` names, each checked against the
predictor it was exported from.

    export_pack.write(ctx, predictor) -> {"exports": […]} or {}

The export and the parity check are the bundle's own `ArtifactExporter`'s — the Model
module's `export.py`, `TorchExporter` (`export`, `parity`; the positional input order is its
`BatchSignature.for_sample`, per sample shape) — never the ModelStore's, which stores by digest
and exports nothing: this module only chooses the rows and records what happened. The rows are
the shipped examples, prepared by the bundle's own Predictor, so the batch the exported model
is traced with and compared on is one the service would build. A format that cannot be
exported (its library missing, a block its tracer refuses, ONNX below opset 18) and an export
whose parity misses `export_parity_atol` are both `WARN:` and recorded in `MANIFEST.exports`
with the reason — the export ships marked, never silently, and never blocks the bundle the
round trip proved. What the exporter warned while exporting (`torch.jit.trace` is deprecated
in torch >= 2.9) is recorded in the export's `warnings`.
"""
from __future__ import annotations

import importlib
import json

from bundle_slice import bundle_modules
from pack_context import out_ok, out_warn
from roundtrip_plan import test_rows
from vega_pack_core import CATALOG, now

__all__ = ["FILES", "write"]

FILES = {"torchscript": "exports/model.pt", "onnx": "exports/model.onnx"}


def write(ctx, predictor) -> dict:
    formats = tuple(ctx.options.export or ())
    if not formats:
        return {}
    import pandas as pd
    raw_rows, _stored = test_rows(ctx)
    rows = raw_rows.head(int(CATALOG["examples"]["default_rows"])) \
        .drop(columns=[c for c in (ctx.schema.get("ignored") or []) if c in raw_rows.columns])
    batch = _batch(predictor, pd.DataFrame(rows))
    exporter, spec = _exporter_and_spec(ctx)
    atol = float(CATALOG["export_parity_atol"])
    (ctx.paths.root / "exports").mkdir(exist_ok=True)
    records = [_export_one(fmt, (exporter, spec, _model(predictor)), batch, ctx, atol)
               for fmt in formats]
    parity = {"schema": "vega/export-parity@1", "created_at": now(), "rows": int(len(rows)),
              "atol": atol, "formats": {r["format"]: r for r in records}}
    (ctx.paths.root / "exports" / "parity.json").write_text(
        json.dumps(parity, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    return {"exports": records}


def _export_one(fmt, model_parts, batch, ctx, atol) -> dict:
    """One format: exported by the bundle's `TorchExporter`, checked against the network."""
    exporter, spec, model = model_parts
    rel = FILES[fmt]
    try:
        written = exporter.export(model, spec, fmt, ctx.paths.root / rel, batch=batch)
        result = exporter.parity(model, ctx.paths.root / rel, batch, atol=atol, fmt=fmt)
    except Exception as exc:                                          # noqa: BLE001
        (ctx.paths.root / rel).unlink(missing_ok=True)
        why = f"{type(exc).__name__}: {str(exc)[:300]}"
        out_warn(f"export {fmt} failed and does not ship — {why}", ctx.markers)
        return {"format": fmt, "file": None, "status": "failed", "why": why}
    status = "pass" if result.get("pass") else "parity-fail"
    record = {"format": fmt, "file": rel, "status": status,
              "max_abs_diff": result.get("max_abs_diff"), "atol": result.get("atol"),
              "inputs": list(written.get("inputs") or [])}
    if written.get("opset") is not None:
        record["opset"] = written["opset"]
    if written.get("warnings"):
        record["warnings"] = list(written["warnings"])
    if status == "pass":
        out_ok(f"export {fmt}: {rel}, parity with the predictor's network within {atol:g} "
               f"(worst |Δ| {result.get('max_abs_diff'):.3g})", ctx.markers)
    else:
        diff = result.get("max_abs_diff")
        how = (f"by {diff:.3g} > {atol:g}" if diff is not None
               else f"({result.get('why') or 'not comparable'})")
        if result.get("why"):
            record["why"] = result["why"]
        out_warn(f"export {fmt}: {rel} ships MARKED — its outputs differ from the network's "
                 f"{how}; use predict.py, not the export, until that is understood",
                 ctx.markers)
    return record


def _exporter_and_spec(ctx):
    """The bundle's own `TorchExporter` (the Model module's `export.py`, D1 §4.1), through the
    module's published API (its `__init__.py`; a package emitted before the module published
    it: the file), and its NetSpec, imported from the loaded package."""
    model_module = bundle_modules(ctx.paths.pkg)["model"]
    module = importlib.import_module(f"{ctx.package}.{model_module}")
    exporter = (module.TorchExporter if "TorchExporter" in getattr(module, "__all__", ())
                else importlib.import_module(f"{ctx.package}.{model_module}.export").TorchExporter)
    domain = importlib.import_module(f"{ctx.package}.domain")
    return exporter(), domain.NetSpec.from_dict(ctx.spec)


def _model(predictor):
    """The network behind the Predictor: its `model` (a property or an accessor) when it
    exposes one, else the attribute the v2.2 Predictor keeps it in."""
    import torch
    model = getattr(predictor, "model", None)
    if callable(model) and not isinstance(model, torch.nn.Module):
        model = model()
    return model if model is not None else predictor._model


def _batch(predictor, rows) -> dict:
    """The network's input batch for these rows, as the Predictor prepares it."""
    prepare = getattr(predictor, "prepare", None)
    if callable(prepare):
        return prepare(rows).batch
    import numpy as np
    import torch
    shaped, _unseen = predictor._transform(rows, with_unseen=True)
    numeric, nominal = list(predictor._numeric), list(predictor._nominal)
    x_num = (torch.tensor(np.asarray(shaped[numeric].astype("float32").to_numpy(),
                                     dtype=np.float32)) if numeric
             else torch.zeros((len(shaped), 0)))
    batch = {"x_num": x_num, "y": None, "x_cat": None}
    if nominal:
        batch["x_cat"] = torch.tensor(np.asarray(shaped[nominal].astype("int64").to_numpy(),
                                                 dtype=np.int64), dtype=torch.long)
    return batch
