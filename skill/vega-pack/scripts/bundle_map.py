"""
bundle_map — the blueprint map the bundle carries (`<package>/blueprint-map.json`): what the
phases add to it, and the rewrite that makes it true of the bundle at the end.

    add_map_entries(ctx, source, files, points)        a phase's rendered files and entry points
    rewrite_map(map_path, copied, markers, mode) -> doc    every port's status, at the end
    mark_entrypoints(map_path, markers, pipeline)          the entry points, and which are here

The forge wrote the map for the whole package; the bundle carries a slice of it, plus files the
pack renders. A phase that renders package files (the serving templates, the inspection
modules) or entry points beside the CLI (a route module, a consumer) maps them with
`add_map_entries`, from its own module: every phase runs before `bundle_slice.finish`, and the
rewrite keeps what they added — a support entry carries no role for it to judge, and an entry
point the blueprint does not declare is kept and marked present. No phase edits this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pack_context import out_ok, out_warn, read_pipeline
from vega_pack_core import CATALOG, read_json

__all__ = ["add_map_entries", "mark_entrypoints", "pipeline_digest", "rewrite_map"]


# --------------------------------------------------------------------------- what phases add

def add_map_entries(ctx, source, files=(), points=()) -> None:
    """What a phase rendered, in the copied map: a `kind: support` entry per package file (the
    serving templates, D2 §8.5; the inspection modules, D3 §13.2) and an entry point per
    `{"file", "calls"}`; a file listed already stays as it is. `finish` keeps both — a support
    entry has no role for the rewrite to judge, and an undeclared entry point is kept."""
    map_path = ctx.paths.pkg / "blueprint-map.json"
    doc, err = read_json(map_path, "blueprint-map.json")
    if err:
        out_warn(f"{err}; what {source} rendered is not in the map", ctx.markers)
        return
    listed = {e.get("file") for e in (doc.get("entries") or []) + (doc.get("entrypoints") or [])
              if isinstance(e, dict)}
    doc.setdefault("entries", []).extend(
        {"file": f, "module": f.split("/")[0], "kind": "support", "used_by": ["Predictor"],
         "source": source, "status": "implemented"} for f in files if f not in listed)
    doc.setdefault("entrypoints", []).extend(
        dict({"stage": "/vega-pack"}, **p) for p in points if p.get("file") not in listed)
    _write_map(map_path, doc)
    out_ok(f"blueprint-map.json maps what {source} rendered", ctx.markers)


def _write_map(map_path, doc) -> None:
    map_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- the rewrite, at the end

def rewrite_map(map_path: Path, copied, markers, mode="serving") -> dict:
    """Every port's status, now that the bundle is what it is.

    A map that still says `pending: /vega-pack` after /vega-pack ran is a map that does not
    describe the code beside it, and the conformance gate reads the map.
    """
    doc, err = read_json(map_path, "blueprint-map.json")
    if err:
        out_warn(f"{err}; the bundle carries no port map", markers)
        return {}
    full = mode == "full"
    changed = [_entry_status(map_path, entry, copied, full) for entry in doc.get("entries") or []]
    doc["package_slice"] = (
        "full: the whole emitted package is in this bundle; every port has its adapter"
        if full else
        "serving: ingestion, training and evaluation are not in this bundle")
    _write_map(map_path, doc)
    out_ok(f"blueprint-map.json rewritten — {'; '.join(sorted({c for c in changed if c}))}",
           markers)
    return doc


def _entry_status(map_path, entry, copied, full):
    """Set one entry's status; the change it made, said in a few words, or None."""
    reasons = {} if full else CATALOG["package_slice"]["out_of_scope"]
    dropped = set() if full else set(CATALOG["package_slice"]["drop_packages"])
    role, module = entry.get("role"), entry.get("module")
    rel = entry.get("file") or entry.get("path")
    if module in dropped or role in reasons:
        entry["status"] = f"out of scope: {reasons.get(role, f'{module} is not part of serving')}"
        return f"{role or rel or module} out of scope"
    if rel and str(rel) in copied:
        if _is_stub(map_path.parent / rel):
            # a file copied as the forge emitted it, still raising its pending stub, is not an
            # adapter; calling it implemented would be a map that lies to the gate
            entry["status"] = "pending: /vega-pack — the adapter was not rendered"
            return f"{role} still pending"
        was_pending = str(entry.get("status", "")).startswith("pending")
        entry["status"] = "implemented"
        return f"{role} implemented" if was_pending else None
    if rel and not role and not full and entry.get("kind") == "support" \
            and str(rel).endswith(".py") and not (map_path.parent / rel).is_file():
        # a support file no role the bundle serves reads (the Trainer's and the Evaluator's
        # metrics, the exporter of a bundle that exports nothing) is left behind with the
        # roles that read it, and the map says so rather than naming a file the bundle lacks
        entry["status"] = f"out of scope: {rel} is read by no role this bundle serves"
        return f"{rel} out of scope"
    if rel and role and not full:
        # A role-bearing file the slice did not carry — the evaluation's baselines are the
        # usual case: /vega-eval adds them to the map as Model implementations, and a map that
        # still calls them `implemented` names a file the bundle does not contain.
        why = ("the evaluation's comparison models, fitted and scored upstream under the "
               "seal; the bundle serves the evaluated network only"
               if role == "baseline" else f"{rel} is not in the serving slice")
        entry["status"] = f"out of scope: {why}"
        return f"{role} out of scope"
    return None


def _is_stub(path: Path) -> bool:
    try:
        return "_PENDING = (" in Path(path).read_text(encoding="utf-8")
    except OSError:
        return False


def pipeline_digest(doc) -> str:
    """The canonical digest the conformance gate recomputes (C0): sorted keys, no spaces."""
    return "sha256:" + hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def mark_entrypoints(map_path: Path, markers, pipeline=None) -> None:
    """The entrypoints the blueprint declares, and which of them are actually here.

    Runs after every entrypoint is written — the CLI, the tests and, with --serve, the service —
    so `present` describes the bundle as shipped. The declared list is the blueprint's
    (`pipeline.json.entrypoints`, the serving concern's when the bundle is packed against it),
    not the forge's: a map still listing `serve/app.py` as absent and calling one port, after
    the pack wrote it calling two, is a map that lies about its own package. When the blueprint
    the pack was given is not the one the forge emitted against, the map pins the new one by
    digest and keeps the old as `emitted_against`, so both claims stay attributable.
    """
    doc, err = read_json(map_path, "blueprint-map.json")
    if err:
        return
    plan = read_pipeline(pipeline) if pipeline else {}
    if plan.get("entrypoints"):
        doc["entrypoints"] = _declared_entrypoints(plan["entrypoints"], doc)
    if plan:
        digest = pipeline_digest(plan)
        if doc.get("pipeline_digest") and doc.get("pipeline_digest") != digest:
            doc.setdefault("emitted_against", {"blueprint": doc.get("blueprint"),
                                               "pipeline_digest": doc.get("pipeline_digest")})
        doc["blueprint"] = str(pipeline)
        doc["pipeline_digest"] = digest
    root = map_path.parent.parent
    points = [p for p in (doc.get("entrypoints") or []) if isinstance(p, dict)]
    for point in points:
        point["present"] = (root / str(point.get("file", ""))).is_file()
    _write_map(map_path, doc)
    present = [p.get("file") for p in points if p.get("present")]
    absent = [p.get("file") for p in points if not p.get("present")]
    out_ok(f"entrypoints present: {', '.join(present) or 'none'}"
           + (f"; declared but not written by this build: {', '.join(absent)}" if absent
              else "") + (f"; the map pins {pipeline}" if plan else ""), markers)


def _declared_entrypoints(declared, doc) -> list:
    """The blueprint's entry points, each keeping the stage the forge's map gave it, followed
    by the map's own entry points the blueprint does not name."""
    known = {p.get("file"): p for p in doc.get("entrypoints") or [] if isinstance(p, dict)}
    rows = []
    for point in declared:
        if not isinstance(point, dict) or not point.get("file"):
            continue
        calls = point.get("calls")
        row = {"file": point["file"],
               "calls": calls if isinstance(calls, list) else [calls] if calls else [],
               "stage": (known.get(point["file"]) or {}).get("stage", "/vega-pack")}
        for key in ("role", "via"):
            if point.get(key):
                row[key] = point[key]
        rows.append(row)
    declared_files = {r["file"] for r in rows}
    return rows + [p for f, p in known.items() if f not in declared_files]
