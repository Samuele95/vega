#!/usr/bin/env python3
"""
test_template_discipline.py — the code discipline C0–C14 holds, at `--size-severity error`, on
everything the backend's templates emit: the package of every cell of the backend matrix, and
any /vega-pack bundle named (D1 WP-B17, BUILD-PLAN G2).

    python3 test_template_discipline.py [<work>] [--cells all|smoke|<glob>[,<glob>…]]
                                        [--jobs N] [--bundle DIR[=PIPELINE] ...]
                                        [--no-matrix] [--json OUT] [--keep]

The matrix. `make_matrix_fixtures.py` makes the B19 fixtures under `<work>` (the raw tables
through the real `shape.py`, one netspec per cell of the catalog's admission table: 127 for
`--cells all`, the default). Each cell's package is emitted by the v2.3 emitter
(`emit_backend.py`, without `--smoke`: the discipline is the code's, not the forward pass's),
then `conformance.check_report` reads it with the gate's default options — `--size-severity
error` since G2 — and the cell is green when no marker is `ERROR:`: C0–C10 hold and there is no
C11–C14 finding. Each finding is recorded with the template (or the spliced segments) its file
was rendered from, so it goes to that template's owner.

The bundles. Each `--bundle DIR` is a bundle /vega-pack wrote: its package (`MANIFEST.package`)
is read with `--bundle` (a `serving` bundle's out-of-scope ports are legal; a `full` bundle is
the complete package) and `--entrypoints DIR` (C11–C13 over `predict.py`, `serve/`, `tests/`
and the CLIs the manifest lists), at `error`. The blueprint run is `=PIPELINE` when given, else
`MANIFEST.blueprint.run` resolved against the bundle directory and its parents.

The gate is live. Before any verdict, a copy of the first cell's package with a seeded
31-statement method is read at the default options and must be refused with that C11 finding —
a clean matrix under a gate that refuses nothing would prove nothing.

`<work>` defaults to a temporary directory (removed at exit unless `--keep`). The report
(`vega/template-discipline@1`) is printed as marker lines and, with `--json OUT`, written:
`{schema, severity, cells: [{id, package, emitted, errors, size}], bundles: [{dir, include,
errors, size}], live, summary: {cells, bundles, findings, failed}}`.

Exit codes: 0 every package and bundle conforms; 1 a finding, an error or a dead gate; 2 the
fixtures, a bundle or the arguments could not be used.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
MATRIX = HERE / "matrix"
for _p in (str(SCRIPTS), str(HERE), str(MATRIX)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conformance  # noqa: E402
import make_matrix_fixtures as mmf  # noqa: E402

SCHEMA = "vega/template-discipline@1"
SEEDED = "_seeded_discipline_probe"


class UsageError(Exception):
    """An input this test cannot use (exit 2); the message names it."""


def say(line: str) -> None:
    print(line, flush=True)


# --------------------------------------------------------------------------- one cell

def _templates(spec: dict, pipeline: dict) -> dict:
    """{package-relative file: the template it was rendered from} (the v2.3 emit table)."""
    import render_templates as rt
    emit = rt.load_emit()
    layout = emit.v23_layout(emit.plan(spec, pipeline))
    return {rel: name for name, (rel, _) in layout.items()}


def _size_record(findings, templates: dict, bmap: dict) -> dict:
    segments = {e["file"]: e.get("segments") for e in bmap.get("entries") or () if e.get("file")}
    return {"counts": _counts(findings),
            "findings": [{"code": f.code, "file": f.file, "line": f.line, "symbol": f.symbol,
                          "template": templates.get(f.file), "segments": segments.get(f.file),
                          "message": f.message()} for f in findings]}


def _counts(findings) -> dict:
    return {c: sum(1 for f in findings if f.code == c) for c in ("C11", "C12", "C13", "C14")}


def _emit(fx: dict, out: Path) -> tuple:
    """(exit code, the ERROR lines) of `emit_backend.py` on the cell's spec, in this process."""
    import emit_backend
    if out.exists():
        shutil.rmtree(out)
    kept = io.StringIO()
    with contextlib.redirect_stdout(kept):
        try:
            code = emit_backend.main(["--spec", fx["spec"], "--pipeline", fx["pipeline"],
                                      "--out", str(out)])
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
    return code, [ln for ln in kept.getvalue().splitlines() if ln.startswith("ERROR")]


def _gate(package_dir: Path, pipeline: dict, options=None) -> tuple:
    """(the ERROR lines, the C11–C14 findings, the map) of the gate on one package. At the
    default severity each finding is also an ERROR line; the lines kept are the others (C0–C10,
    IO), and the findings are kept as values, so a package fails on either."""
    bmap = json.loads((package_dir / "blueprint-map.json").read_text(encoding="utf-8"))
    markers, findings = conformance.check_report(package_dir, bmap, pipeline,
                                                 options or conformance.GateOptions())
    sized = {f.message() for f in findings}
    errors = [m.line() for m in markers.errors if m.message not in sized]
    if findings and not any(m.message in sized for m in markers.errors):
        errors.append("the gate reported its C11–C14 findings below ERROR at its default "
                      "severity")
    return errors, list(findings), bmap


def check_cell(cell_dir: str) -> dict:
    """One cell: emitted, then gated at the default options (a worker's whole job)."""
    fx = json.loads((Path(cell_dir) / "fixture.json").read_text(encoding="utf-8"))
    record = {"id": fx["id"], "package": None, "emitted": False, "errors": [], "size": None}
    out = Path(cell_dir) / "discipline"
    code, errors = _emit(fx, out)
    if code != 0:
        record["errors"] = errors or [f"emit_backend exited {code}"]
        return record
    package = out / fx["package"]
    spec = json.loads(Path(fx["spec"]).read_text(encoding="utf-8"))
    pipeline = json.loads(Path(fx["pipeline"]).read_text(encoding="utf-8"))
    record.update(emitted=True, package=str(package))
    record["errors"], findings, bmap = _gate(package, pipeline)
    record["size"] = _size_record(findings, _templates(spec, pipeline), bmap)
    return record


# --------------------------------------------------------------------------- the matrix

def matrix_records(work: Path, how: str, jobs: int) -> list:
    table = mmf.load_table()
    chosen = mmf.select(mmf.cells(table), how, table)
    made = mmf.make(work, chosen, table, sanity=False, jobs=max(jobs, 4))
    dirs = [str(m.dir) for m in made["cells"].values()]
    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    os.environ.update(env)
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        return list(pool.map(check_cell, dirs))


def live_check(record: dict, work: Path) -> dict:
    """A seeded 31-statement method in a copy of an emitted package must be refused at the
    default options, naming it: the gate this test trusts is the one that runs."""
    fx = json.loads((Path(record["package"]).parents[1] / "fixture.json")
                    .read_text(encoding="utf-8"))
    copy = work / "live" / fx["package"]
    if copy.parent.exists():
        shutil.rmtree(copy.parent)
    shutil.copytree(record["package"], copy)
    target = copy / "domain.py"
    body = "".join(f"    x{i} = {i}\n" for i in range(31))
    target.write_text(target.read_text(encoding="utf-8") + f"\n\ndef {SEEDED}():\n{body}",
                      encoding="utf-8")
    pipeline = json.loads(Path(fx["pipeline"]).read_text(encoding="utf-8"))
    bmap = json.loads((copy / "blueprint-map.json").read_text(encoding="utf-8"))
    markers, _ = conformance.check_report(copy, bmap, pipeline)
    errors = [m.line() for m in markers.errors]
    refused = any(e.startswith("ERROR: C11") and SEEDED in e and "31 statements > 30" in e
                  for e in errors)
    return {"package": str(copy), "refused": refused, "errors": errors[:5]}


# --------------------------------------------------------------------------- the bundles

def _bundle_arg(text: str) -> tuple:
    directory, _, pipeline = text.partition("=")
    return Path(directory).resolve(), (Path(pipeline).resolve() if pipeline else None)


def _pipeline_of(bundle: Path, manifest: dict, given) -> Path:
    if given is not None:
        return given
    run = str(((manifest.get("blueprint") or {}).get("run")) or "")
    if not run:
        raise UsageError(f"{bundle}: MANIFEST.blueprint.run names no pipeline; pass "
                         "--bundle DIR=PIPELINE")
    for base in (bundle, *bundle.parents):
        if (base / run).is_file():
            return base / run
    raise UsageError(f"{bundle}: the blueprint run {run!r} is not found beside the bundle or "
                     "above it; pass --bundle DIR=PIPELINE")


def check_bundle(text: str) -> dict:
    bundle, given = _bundle_arg(text)
    try:
        manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UsageError(f"{bundle} is not a readable bundle (MANIFEST.json: {exc})") from exc
    pipeline_path = _pipeline_of(bundle, manifest, given)
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    include = manifest.get("include") or "serving"
    options = conformance.GateOptions(bundle=include != "full", entrypoints=bundle)
    errors, findings, _ = _gate(bundle / manifest["package"], pipeline, options)
    return {"dir": str(bundle), "include": include, "pipeline": str(pipeline_path),
            "errors": errors, "size": {"counts": _counts(findings),
                                       "findings": [f.message() for f in findings]}}


# --------------------------------------------------------------------------- the verdict

def _say_cells(records: list) -> list:
    failed = []
    for rec in records:
        if rec["errors"] or not rec["emitted"] or (rec["size"] or {}).get("findings"):
            failed.append(rec["id"])
            for line in rec["errors"][:8]:
                say(f"ERROR: {rec['id']}: {line}")
    distinct = {f["message"]: f for rec in records for f in (rec["size"] or {}).get("findings", ())}
    for message, f in sorted(distinct.items()):
        cells = sum(1 for rec in records if any(g["message"] == message
                                                 for g in (rec["size"] or {}).get("findings", ())))
        say(f"ERROR: {message} (template {f['template'] or f['segments']}, {cells} cell(s))")
    say(f"{'OK' if not failed else 'ERROR'}: matrix: {len(records) - len(failed)}/{len(records)} "
        "emitted package(s) pass C0–C14 at --size-severity error")
    return failed


def _say_bundles(records: list) -> list:
    failed = []
    for rec in records:
        ok = not rec["errors"] and not rec["size"]["findings"]
        for line in [*rec["errors"], *rec["size"]["findings"]]:
            say(f"ERROR: bundle {rec['dir']}: {line}")
        say(f"{'OK' if ok else 'ERROR'}: bundle {rec['dir']} ({rec['include']}): C0–C14 with "
            f"its entrypoints at --size-severity error, findings {rec['size']['counts']}")
        if not ok:
            failed.append(rec["dir"])
    return failed


def verdict(cells: list, bundles: list, live) -> dict:
    failed = _say_cells(cells) if cells else []
    failed += _say_bundles(bundles)
    if live is not None:
        say(f"{'OK' if live['refused'] else 'ERROR'}: the gate is live: a seeded 31-statement "
            f"method is {'refused' if live['refused'] else 'NOT refused'} at the default options")
        if not live["refused"]:
            failed.append("live")
    findings = sum(sum((r["size"] or {}).get("counts", {}).values()) for r in [*cells, *bundles])
    return {"cells": len(cells), "bundles": len(bundles), "findings": findings,
            "failed": failed}


def _work(given, keep: bool) -> Path:
    if given:
        path = Path(given).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = Path(tempfile.mkdtemp(prefix="vega-discipline-"))
    if not keep:
        atexit.register(shutil.rmtree, path, True)
    return path


def _arguments(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", nargs="?", default=None)
    ap.add_argument("--cells", default="all",
                    help="all (default), smoke, or a comma list of globs over cell ids")
    ap.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--bundle", action="append", default=[], metavar="DIR[=PIPELINE]")
    ap.add_argument("--no-matrix", action="store_true", help="the bundles alone")
    ap.add_argument("--json", default=None, metavar="OUT")
    ap.add_argument("--keep", action="store_true", help="keep a temporary work directory")
    a = ap.parse_args(argv)
    if a.no_matrix and not a.bundle:
        ap.error("--no-matrix leaves nothing to check without --bundle")
    return a


def main(argv=None) -> int:
    a = _arguments(argv)
    work = _work(a.work, a.keep)
    try:
        cells = [] if a.no_matrix else matrix_records(work, a.cells, a.jobs)
        bundles = [check_bundle(b) for b in a.bundle]
    except (mmf.MatrixError, UsageError) as exc:
        say(f"ERROR: {exc}")
        return 2
    first = next((r for r in cells if r["emitted"]), None)
    live = live_check(first, work) if first is not None else None
    summary = verdict(cells, bundles, live)
    if a.json:
        Path(a.json).write_text(json.dumps({"schema": SCHEMA, "severity": "error",
                                            "cells": cells, "bundles": bundles, "live": live,
                                            "summary": summary}, indent=2) + "\n",
                                encoding="utf-8")
    say(f"{'OK' if not summary['failed'] else 'ERROR'}: template discipline: "
        f"{summary['cells']} cell package(s), {summary['bundles']} bundle(s), "
        f"{summary['findings']} C11–C14 finding(s); failed {summary['failed'] or '[]'}")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
