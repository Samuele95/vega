#!/usr/bin/env python3
"""
apply_architecture_layer.py — validate a `layer-architecture-data` document and inject the
/vega-forge `architecture` layer (data + the shipped render script + style + <noscript>) into
a domain-forge model.

    python3 apply_architecture_layer.py MODEL.html --data LAYER.json --out OUT.html
                                        [--notebook engineer.ipynb]
                                        [--architect-run .claude/architect-runs/<ts> ...]
                                        [--improve-run .claude/improve-runs/<ts> ...]
                                        [--unattended] [--domain-forge-dir PATH]
                                        [--skip-compile]

The layer document may carry `"from_run": "<run-dir>"` instead of retyping the emitter's
output: `netspec.json`, `smoke.json`, `blueprint-map.json` and the emitted files' digests are
then read from that directory and spliced in verbatim. The author writes the **judgement** —
the abstract, the per-family readings, the mechanics, the forks and what answered them, the
`defaults_to_test` observations, and each finding's `next` line — and the numbers stay the
emitter's. The notebook record is written by this script from `--notebook`, and the
`code_findings` are spliced from the architect audits named by `--architect-run` and the
improve audits named by `--improve-run` (each flag once per audit, the first audit first),
each finding stamped with its `tool` and its `audit` (the author supplies only each finding's
`next`, by id — or by tool, audit and id): no stage types those records. The audits of each
tool are judged by `audit_quorum.py` (code-contract §5): one audit with no structural or
divergence finding passes; when the first reports one, two more fresh audits are passed with
it, and a structural or divergence finding at least two of the three audits report (the same
file and the same principle anchor or category) is **confirmed** — every audit and the matching
are the tool's `conformance` record, and an unconfirmed finding is recorded there and in
`code_findings`. A confirmed finding is **advisory** (*Advisory*, code-contract §5): with
`--unattended` it is published as a `WARN:` in `markers` and an open item handed to /architect
and /improve (`conformance.<tool>.advisory.open_items`, and the finding's `open_item` in
`code_findings`); attended, the user's answer to its `FORK:` must be recorded in `forks`
(`{"gate": "<tool>", "findings": ["<audit>:<id>", …], "asked": …, "answer": "accept and record
— …"}`): *accept and record* publishes with the WARN and the open item, *fix now* and *stop*
publish nothing. A tool not installed is a WARN.

What this script refuses to publish (netspec-contract §4):

  * a layer whose `smoke.forward_ok` is not true — a spec that does not build is fixed, never
    published;
  * a layer whose `smoke.sample` (the counts the sample panel shows, D1 §16) was not measured on
    this spec: its shape, a parameter the spec declares (a window's length, stride, horizon,
    context and autoregressive flag; a graph's level, protocol and batching) or the unit
    `madds_unit` counts per disagrees with the spec — a smoke record without a `sample` block is
    a WARN (the panel then shows the spec's declaration only);
  * a layer whose `netspec` fails `compile_spec.py` — the layer cannot be more trusted than
    the spec it carries;
  * a layer whose conformance is not mechanically `pass` (C0-C14 refuse on any finding);
  * a layer whose architect or improve judgement is not the record spliced from the audits,
    whose first audit reported a `structural` or `divergence` finding with the two fresh
    audits not given, whose audits are not one or three distinct runs (run ids included) of
    the same code, or whose advisory block does not name every confirmed finding as an open
    item (a malformed record); and, attended, a layer with a confirmed finding whose `FORK:`
    has no recorded answer, or the answer *fix now* or *stop* — a confirmed finding never
    refuses an unattended layer;
  * a layer whose `defaults_to_test` names a path a `/vega-train` rung cannot move — one that
    does not resolve on the spec in the delta-path grammar, or a regularization path the
    emitted code does not read (`defaults_paths.py`, F40);
  * a layer with no `--notebook`, or whose `blueprint_map` has a non-empty `unmapped` or a
    port that is neither implemented nor pending.

The block is written ONLY by the platform writer `domain-forge/scripts/apply_layer.py`. The
input model is never modified.

Exit codes: 0 wrote OUT; 2 validation or input error; 3 platform writer missing or failed.
"""
from __future__ import annotations

import argparse
import hashlib
import html as _html
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# the sample record (D1 §16): one module states it for this script and the smoke
from architecture_sample import declared_sample, sample_errors, sample_line  # noqa: E402,F401
# the judgement half's rule (code-contract §5): one module states it for the forge and the pack
import audit_quorum as quorum  # noqa: E402
# a defaults_to_test path is a rung's delta: the trainer's grammar, on a path the code reads (F40)
import defaults_paths  # noqa: E402

SKILL_DIR = HERE.parent
ASSETS = SKILL_DIR / "assets"
RENDER_JS = ASSETS / "architecture-render.js"
STYLE_CSS = ASSETS / "architecture-layer.css"

LAYER = "architecture"
SCHEMA = "vega/architecture@1"
MAX_RENDER = 60 * 1024

_BLOCK_RE = re.compile(r"[ \t]*<!--\s*@LAYER:start\s+" + LAYER +
                       r"\b[\s\S]*?@LAYER:end\s+" + LAYER + r"\s*-->[ \t]*\n?")

REQUIRED = ("schema", "netspec", "smoke", "code", "blueprint_map", "conformance",
            "code_findings", "notebook", "defaults_to_test", "forks", "markers", "readings")

# the runtime rows no ladder rung tests, so they get no defaults_to_test entry
RUNTIME_ROWS = ("defaults/device", "defaults/precision", "defaults/seed")


def _digest_text(text):
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------- splicing

def splice(doc, out, base_dir):
    """Fill the mechanical half from the emitter's run directory."""
    ref = doc.get("from_run")
    if not ref:
        return doc, []
    # a relative from_run is resolved against the layer document's own directory first: the
    # authoring file lives in the run directory, so `"from_run": "."` means *this* run, not
    # whatever directory the script happened to be invoked from
    run = Path(ref)
    if not run.is_absolute():
        beside = Path(base_dir) / ref
        run = beside if beside.is_dir() else run
    if not run.is_dir():
        return doc, [f"from_run {ref!r} is not a directory (looked beside "
                     f"{base_dir} and in the working directory)"]

    for key, name in (("netspec", "netspec.json"), ("smoke", "smoke.json")):
        if doc.get(key):
            continue
        path = run / name
        if not path.is_file():
            return doc, [f"from_run: {path} not found (the emitter writes it; run "
                         f"emit_backend.py --smoke first)"]
        try:
            doc[key] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return doc, [f"from_run: {path} is not valid JSON: {exc}"]

    package = ((doc.get("netspec") or {}).get("outputs") or {}).get("package")
    pkg_dir = run / package if package else None
    if "blueprint_map" not in doc and pkg_dir and (pkg_dir / "blueprint-map.json").is_file():
        doc["blueprint_map"] = json.loads(
            (pkg_dir / "blueprint-map.json").read_text(encoding="utf-8"))

    if "code" not in doc and pkg_dir and pkg_dir.is_dir():
        files = sorted(p.relative_to(run).as_posix() for p in pkg_dir.rglob("*.py"))
        doc["code"] = {"run_dir": str(run), "package": package, "files": files,
                       "digests": {f: _digest_file(run / f) for f in files}}
        out(f"OK: from_run: {len(files)} emitted file(s) digested from {pkg_dir}")
    return doc, []


_GATE_RUNS = {"architect": "`/architect <package> --report-only --baseline <blueprint.md>`",
              "improve": "`/improve <package> --report-only`"}
_ROW_KEYS = ("anchor_justification", "title", "axis")


def _finding_row(raw, tool, audit, quorum):
    """One spliced finding: what the layer shows of it, the audit it came from and, for a
    structural or divergence finding, the quorum's word on it."""
    row = {"id": raw.get("id"), "tool": tool, "audit": audit, "depth": raw.get("depth"),
           "file": raw.get("file") or raw.get("location"), "anchors": raw.get("anchors") or [],
           "tags": raw.get("tags") or [], **{k: raw.get(k) for k in _ROW_KEYS}}
    if quorum is not None:
        row["quorum"] = {"reported_by": quorum["reported_by"], "confirmed": quorum["confirmed"]}
    return row


def _authored_next(doc):
    """The author's `next` lines: by (tool, audit, id) when the entry names its audit, else by
    (tool, id) for the first audit's finding of that id."""
    authored = {}
    for f in doc.get("code_findings") or []:
        if isinstance(f, dict) and f.get("next") is not None:
            tool = f.get("tool") or "architect"
            authored[(tool, int(f.get("audit") or 1), str(f.get("id")))] = f.get("next")
    return authored


def splice_findings(doc, runs, out, tool="architect"):
    """`code_findings` come from the architect and the improve audits, each finding stamped with
    the tool and the audit that raised it; the author writes only each `next` (by id, or by
    tool, audit and id). The audits are judged by `audit_quorum.judge` (code-contract §5): the
    first audit alone when it reports no structural or divergence finding, else the first and
    two fresh ones, and the judgement — every audit and the matching — is the tool's
    `conformance` record. Splicing one tool keeps the other tool's findings already spliced."""
    if not runs:
        return doc, []
    runs = [runs] if isinstance(runs, (str, Path)) else list(runs)
    try:
        audits = [quorum.load_run(r, n) for n, r in enumerate(runs, 1)]
    except quorum.QuorumError as exc:
        return doc, [f"--{tool}-run: {exc}; the judgement half of the gate is "
                     f"{_GATE_RUNS[tool]} and its findings.json is what this splices"]
    judgement = quorum.judge(audits, tool)
    authored = _authored_next(doc)
    verdicts = {row["finding"]: row for row in judgement.record["matching"]}
    kept = [f for f in doc.get("code_findings") or []
            if isinstance(f, dict) and f.get("tool") and f.get("tool") != tool
            and "depth" in f]
    spliced = []
    for audit in audits:
        for finding in audit.findings:
            row = _finding_row(finding.raw, tool, audit.index, verdicts.get(finding.key))
            plan = authored.get((tool, audit.index, finding.id))
            if plan is not None:
                row["next"] = plan
            spliced.append(row)
    doc["code_findings"] = kept + spliced
    conformance = doc.setdefault("conformance", {})
    conformance[tool] = judgement.record
    _say_judgement(judgement, tool, out)
    return doc, []


FORGE_REMEDY = ("fix the backend template through /architect or /improve (apply) on "
                "backends/<name>/templates/, re-emit, and run the gate again")


def advise_layer(doc, mode, out, remedy=FORGE_REMEDY):
    """The advisory half (code-contract §5, *Advisory*), per judged tool: every confirmed
    finding becomes a WARN in `markers` and an open item — `conformance.<tool>.advisory` and the
    finding's `open_item` in `code_findings` — unattended; attended, only when the user's
    recorded answer in `forks` is *accept and record*. Returns the refusals (attended: no answer,
    *fix now*, *stop*)."""
    errs, conf = [], doc.get("conformance") or {}
    for tool in ("architect", "improve"):
        record = conf.get(tool)
        if not isinstance(record, dict) or record.get("pending") or record.get("invalid") \
                or not record.get("audits"):
            continue
        advice = quorum.advise(record, mode, doc.get("forks") or (), remedy)
        record["advisory"] = advice.record
        items = {item["finding"]: item["text"] for item in advice.open_items}
        for row in doc.get("code_findings") or []:
            key = f"{row.get('audit')}:{row.get('id')}"
            if isinstance(row, dict) and row.get("tool") == tool and key in items:
                row["open_item"] = items[key]
        markers = doc.setdefault("markers", [])
        markers += [w for w in advice.warns if w not in markers]
        for warn in advice.warns:
            out(warn)
        errs += list(advice.errors)
    return errs


def _say_judgement(judgement, tool, out):
    rec = judgement.record
    for audit in rec["audits"]:
        out(f"OK: {tool} audit {audit['audit']} ({audit['run']}): {audit['findings']} finding(s), "
            f"{audit['structural']} structural, {audit['divergence']} divergence")
    for row in rec["matching"]:
        out(f"{quorum.row_level(row)}: {tool} {row['finding']} reported by audit(s) "
            f"{row['reported_by']} of {len(rec['audits'])} — {quorum.row_state(row)}")
    out(f"OK: spliced {tool}: {len(rec['audits'])} audit(s), confirmed structural "
        f"{rec['structural']}, divergence {rec['divergence']}; unconfirmed "
        f"{rec['unconfirmed']['structural']} structural, {rec['unconfirmed']['divergence']} "
        f"divergence; recorded {rec['recorded']}")


def notebook_record(path, out):
    """`executed` is true when every code cell carries outputs and none is an error output.
    No stage types this record — it is read off the file (notebook-contract §3)."""
    p = Path(path)
    if not p.is_file():
        return None, f"--notebook {p} not found; the stage order is judgement → notebook → apply"
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{p} is not a readable notebook: {exc}"
    code = [c for c in nb.get("cells") or [] if c.get("cell_type") == "code"]
    outputs = [c.get("outputs") or [] for c in code]
    errored = any(o.get("output_type") == "error" for cell in outputs for o in cell)
    executed = bool(code) and all(cell for cell in outputs) and not errored
    if not executed:
        out("WARN: notebook not executed" if not errored else
            "WARN: notebook not executed — a cell ended in an error output")
    return {"path": str(p), "executed": executed, "digest": _digest_file(p),
            "code_cells": len(code)}, None


# --------------------------------------------------------------------------- validation

def smoke_errors(doc):
    """The smoke record: a forward pass that ran, its parameter count, and a `sample` block (if
    any) measured on this spec."""
    smoke, errs = doc.get("smoke") or {}, []
    if not smoke.get("forward_ok"):
        errs.append("smoke.forward_ok is not true: a spec that does not build is fixed and "
                    "re-emitted, never published — re-run emit_backend.py --smoke")
    if smoke.get("params_total") is None:
        errs.append("smoke.params_total is required")
    return errs + sample_errors(doc.get("netspec") or {}, smoke)


def _gate_problems(tool, record, markers) -> list:
    """The judgement half of the gate, per tool (netspec-contract §4, code-contract §5): the
    record `audit_quorum.judge` made of the audits — judged to its end (one clean first audit,
    or the first and two fresh ones; nothing pending, nothing invalid), its confirmed findings
    carried by its advisory block as open items — or `"not installed"` said out loud in a
    WARN. A confirmed finding is advisory and refuses nothing here; a malformed record does."""
    if record == "not installed":
        if any(str(m).startswith("WARN:") and tool in str(m) for m in markers):
            return []
        return [f"conformance.{tool} is \"not installed\" but markers carry no `WARN: {tool} "
                "not installed — conformance judged mechanically only`; a missing judgement "
                "half is stated, never silent"]
    if not isinstance(record, dict) or not record.get("audits"):
        return [f"conformance.{tool} must be the record spliced from the audits (--{tool}-run, "
                "once per audit) or the string \"not installed\"; the judgement is read from "
                "the runs, never typed into the layer"]
    if record.get("pending"):
        return [f"conformance.{tool}: {record['pending']} — run {_GATE_RUNS[tool]} twice more, "
                "each in a fresh context that has not read the earlier audits, and pass all three "
                f"--{tool}-run in order (code-contract §5); a single audit's structural or "
                "divergence finding is reproduced or recorded, never published unjudged"]
    problems = [f"conformance.{tool}: {p}" for p in record.get("invalid") or []]
    return problems + quorum.advisory_problems(record)


def validate(doc, out, spec_check=None):
    errs = []
    if not isinstance(doc, dict):
        return ["layer document is not a JSON object"]
    for key in REQUIRED:
        if key not in doc:
            errs.append(f"missing required top-level key {key!r}")
    if errs:
        return errs
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")

    errs += smoke_errors(doc)

    if spec_check is not None and spec_check is False:
        errs.append("the layer's netspec does not pass compile_spec.py --check-only; the "
                    "layer cannot be more trusted than the spec it carries")

    bmap = doc.get("blueprint_map") or {}
    if bmap.get("unmapped"):
        errs.append(f"blueprint_map.unmapped is not empty ({bmap['unmapped']}); a file nobody "
                    "mapped is a designer's choice the emitter is not allowed to make")
    for entry in bmap.get("entries") or []:
        if not entry.get("role") or entry.get("role") == "baseline":
            continue
        status = str(entry.get("status") or "")
        if status != "implemented" and not status.startswith(("pending", "out of scope")):
            errs.append(f"blueprint_map: {entry.get('file')} has status {status!r}; every "
                        "port is implemented, `pending: <stage>` or `out of scope: <reason>`")

    conf = doc.get("conformance") or {}
    if conf.get("mechanical") != "pass":
        errs.append(f"conformance.mechanical is {conf.get('mechanical')!r}; the layer is "
                    "written only after the mechanical gate passes (conformance.py)")
    for tool in ("architect", "improve"):
        errs += _gate_problems(tool, conf.get(tool), doc.get("markers") or [])

    nb = doc.get("notebook") or {}
    if not nb.get("path"):
        errs.append("notebook.path is required — pass --notebook; the stage order is "
                    "judgement → notebook → apply")
    elif not nb.get("executed") and not any("notebook not executed" in str(m)
                                            for m in doc.get("markers") or []):
        errs.append("notebook.executed is false but markers carry no `WARN: notebook not "
                    "executed`; an unexecuted notebook is a WARN, never silently a report")

    spec = doc.get("netspec") or {}
    tested = {str(d.get("path")) for d in doc.get("defaults_to_test") or []
              if isinstance(d, dict)}
    for entry in doc.get("defaults_to_test") or []:
        if not isinstance(entry, dict) or not entry.get("observation"):
            errs.append("every defaults_to_test entry names the observation in the first run "
                        "that would overturn it")
            break
    for entry in doc.get("defaults_to_test") or []:
        why = defaults_paths.problem(spec, (entry or {}).get("path")) if isinstance(entry, dict) \
            else None
        if why:
            errs.append(why)
    missing = _untested_defaults(spec, tested)
    for path in missing:
        out(f"WARN: the spec's vega:default at {path} has no defaults_to_test entry; the "
            "ablation ladder starts from those")

    for fork in doc.get("forks") or []:
        if not isinstance(fork, dict):
            errs.append("every fork is an object")
            break
        if bool(fork.get("answer")) == bool(fork.get("or")):
            errs.append(f"fork {fork.get('asked')!r} carries "
                        + ("both `answer` and `or`" if fork.get("answer") else "neither "
                           "`answer` nor `or`") + "; exactly one records what settled it")

    readings = doc.get("readings") or {}
    for key in ("abstract", "per_family", "mechanics"):
        if not readings.get(key):
            errs.append(f"readings.{key} is required — it is what a reader sees instead of "
                        "the JSON")
    return errs


def _untested_defaults(spec, tested):
    """Every non-structural, non-runtime `vega:default` should have a defaults_to_test row."""
    missing = []

    def note(label, node):
        if not isinstance(node, dict) or node.get("source") != "vega:default":
            return
        rows = node.get("catalog")
        rows = [rows] if isinstance(rows, str) else (rows or [])
        if rows and all(r in RUNTIME_ROWS for r in rows):
            return
        if not any(label.split("[")[0] in t or t.startswith(label) for t in tested):
            missing.append(label)

    for key in ("family", "loss", "metrics", "optimizer", "schedule", "early_stopping",
                "init", "training"):
        note(key, spec.get(key))
    for reg in spec.get("regularization") or []:
        note(f"regularization[kind={(reg or {}).get('kind')}]/value", reg)
    for block in spec.get("blocks") or []:
        if isinstance(block, dict):
            note(f"blocks[{block.get('id')}]", block)
    return missing


# --------------------------------------------------------------------------- noscript

def _esc(value):
    return _html.escape("" if value is None else str(value), quote=True)


def _judgement_line(record):
    """The judgement half of one tool in one line: the audits, what they confirmed, and what
    they reported that no second audit reproduced."""
    if not isinstance(record, dict):
        return str(record or "not run")
    left = record.get("unconfirmed") or {}
    advisory = record.get("advisory") or {}
    items = advisory.get("open_items") or []
    said = (f"; advisory ({advisory.get('mode')}, {advisory.get('outcome')}): {len(items)} open "
            "item(s) for /architect and /improve — "
            + "; ".join(str(i.get("text")) for i in items) if items else "")
    return (f"{len(record.get('audits') or [])} audit(s) from {record.get('run')}; confirmed "
            f"{record.get('structural')} structural, {record.get('divergence')} divergence; "
            f"unconfirmed (recorded, not blocking) {left.get('structural', 0)} structural, "
            f"{left.get('divergence', 0)} divergence; recorded {record.get('recorded')}{said}")


def noscript_html(doc):
    spec = doc.get("netspec") or {}
    smoke = doc.get("smoke") or {}
    conf = doc.get("conformance") or {}
    readings = doc.get("readings") or {}
    parts = ['<section class="layer-architecture-noscript">', "<h2>Network</h2>",
             f"<p>{_esc(readings.get('abstract'))}</p>",
             "<h3>The network</h3>",
             f"<p>{_esc((spec.get('family') or {}).get('name'))} — "
             f"{_esc(smoke.get('params_total'))} parameters "
             f"({_esc(smoke.get('params_trainable'))} trainable), "
             f"{_esc(smoke.get('madds_per_sample'))} MAdds per "
             f"{_esc(str(smoke.get('madds_unit') or 'sample').replace('_', ' '))}, forward pass "
             f"{'ok' if smoke.get('forward_ok') else 'FAILED'} on "
             f"{_esc(smoke.get('device'))}.</p>",
             f"<p>The sample: {_esc(sample_line(spec, smoke))}.</p>",
             *[f"<p>Initialisation: {_esc(n)}.</p>"
               for n in (smoke.get("init_notes") or {}).get("notes") or []],
             "<h3>The decisions</h3>",
             "<table><thead><tr><th>choice</th><th>value</th><th>source</th>"
             "<th>rationale</th></tr></thead><tbody>"]
    for key in ("family", "loss", "metrics", "optimizer", "schedule", "early_stopping",
                "init", "training"):
        node = spec.get(key)
        if not isinstance(node, dict):
            continue
        value = node.get("name") or node.get("primary") or node.get("scheme") or ""
        parts.append("<tr>" + "".join(f"<td>{_esc(x)}</td>" for x in (
            key, value, node.get("source"), node.get("rationale"))) + "</tr>")
    for block in spec.get("blocks") or []:
        if isinstance(block, dict):
            parts.append("<tr>" + "".join(f"<td>{_esc(x)}</td>" for x in (
                block.get("id"), block.get("type"), block.get("source"),
                block.get("rationale"))) + "</tr>")
    parts.append("</tbody></table>")
    parts += ["<h3>Conformance</h3><ul>",
              f"<li>mechanical: {_esc(conf.get('mechanical'))}</li>"]
    for tool in ("architect", "improve"):
        parts.append(f"<li>{tool}: {_esc(_judgement_line(conf.get(tool)))}</li>")
    parts.append("</ul>")
    parts += ["<h3>Mechanics</h3>", f"<p>{_esc(readings.get('mechanics'))}</p>",
              "<h3>What the first run must show</h3><ul>"]
    for entry in doc.get("defaults_to_test") or []:
        parts.append(f"<li>{_esc((entry or {}).get('path'))}: "
                     f"{_esc((entry or {}).get('observation'))}</li>")
    parts.append("</ul>")
    parts.append("<p>Enable JavaScript for the block diagram, the blueprint map, the "
                 "conformance panel and the schedule curve.</p></section>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- the writer

def platform_apply(writer, html, data_text, render_js, style_css, produced_by, noscript,
                   tmpdir, model_path):
    """The layer block is written only by domain-forge's apply_layer.py — imported when that
    works, and run as a CLI when it does not. A private writer here would be a second
    implementation of the platform's own format."""
    notes = []
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("df_apply_layer", str(writer))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.apply_layer(html, LAYER, data_text, render_js, style_css=style_css,
                                 data_type="application/json", produced_by=produced_by,
                                 noscript_html=noscript)
        if isinstance(result, str) and f"@LAYER:start {LAYER}" in result:
            return result, "platform writer (import)"
        notes.append("import path returned no layer block")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"import path failed: {exc}")
    try:
        tmp = Path(tmpdir)
        data_file = tmp / "layer.json"
        data_file.write_text(data_text, encoding="utf-8")
        render_file = tmp / "architecture-render.js"
        render_file.write_text(render_js, encoding="utf-8")
        style_file = tmp / "architecture-layer.css"
        style_file.write_text(style_css, encoding="utf-8")
        noscript_file = tmp / "noscript.html"
        noscript_file.write_text(noscript, encoding="utf-8")
        out_file = tmp / "out.html"
        proc = subprocess.run(
            [sys.executable, str(writer), str(model_path), "--layer", LAYER,
             "--data", str(data_file), "--render", str(render_file), "--style", str(style_file),
             "--data-type", "application/json", "--produced-by", produced_by,
             "--noscript", str(noscript_file), "--out", str(out_file)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and out_file.is_file():
            return out_file.read_text(encoding="utf-8"), "platform writer (CLI)"
        notes.append(f"CLI path failed (exit {proc.returncode}): "
                     f"{(proc.stderr or proc.stdout).strip()[:300]}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"CLI path failed: {exc}")
    return None, "; ".join(notes)


def compile_ok(doc, model_path, out):
    """Re-run the compiler on the spec the layer carries. A layer may not be more trusted
    than its spec, so this is not optional; `--skip-compile` exists only for a caller that
    just ran it and passes the result in."""
    spec = doc.get("netspec") or {}
    try:
        sys.path.insert(0, str(HERE))
        import compile_spec as C  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        out(f"WARN: compile_spec.py could not be imported ({exc}); the layer's spec was not "
            "re-checked")
        return None
    try:
        html = Path(model_path).read_text(encoding="utf-8")
        manifest_path = (spec.get("input") or {}).get("manifest")
        pipeline_path = (spec.get("input") or {}).get("pipeline")
        manifest = (json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                    if manifest_path and Path(manifest_path).is_file() else None)
        pipeline = (json.loads(Path(pipeline_path).read_text(encoding="utf-8"))
                    if pipeline_path and Path(pipeline_path).is_file() else None)
        markers = C.compile_spec(spec, model_html=html, pipeline=pipeline, manifest=manifest)
    except Exception as exc:  # noqa: BLE001
        out(f"WARN: the layer's spec could not be re-checked ({exc})")
        return None
    if markers.ok:
        out(f"OK: the layer's netspec still compiles ({len(markers.warns)} warning(s))")
        return True
    for marker in markers.errors:
        out(f"ERROR: netspec: {marker.message}")
    return False


def print_accepted(doc, data):
    """What was accepted, in one line, after a WARN when the smoke record has no `sample`."""
    smoke, conf = doc["smoke"], doc["conformance"]
    shape = declared_sample(doc["netspec"]).get("shape") or "row"
    if smoke.get("sample") is None:
        print(f"WARN: smoke.json carries no `sample` block, so the layer's sample panel shows the "
              f"spec's declared {shape} sample without its counts — re-run emit_backend.py "
              "--smoke")
    print(f"OK: {data} validates against {SCHEMA} "
          f"({len(doc['netspec'].get('blocks') or [])} top-level block(s), "
          f"{smoke.get('params_total'):,} parameters, {shape} sample"
          f"{'' if isinstance(smoke.get('sample'), dict) else ' (no smoke sample block)'}, "
          f"mechanical {conf.get('mechanical')}, "
          f"{len(doc.get('code_findings') or [])} code finding(s), "
          f"notebook executed {doc['notebook'].get('executed')})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="input MODEL.html (never modified)")
    ap.add_argument("--data", required=True, help="layer-architecture-data JSON document")
    ap.add_argument("--out", required=True, help="output HTML path (must differ from MODEL)")
    ap.add_argument("--notebook", default=None, help="engineer.ipynb (executed)")
    ap.add_argument("--architect-run", action="append", default=None,
                    help="an /architect --report-only --baseline run whose findings.json is "
                         "spliced; once per audit, the first audit first")
    ap.add_argument("--improve-run", action="append", default=None,
                    help="an /improve --report-only run whose findings.json is spliced; once "
                         "per audit, the first audit first")
    ap.add_argument("--unattended", action="store_true",
                    help="the stage runs unattended: a confirmed judgement finding is a WARN and "
                         "an open item; attended (the default) it needs the user's recorded "
                         "FORK answer")
    ap.add_argument("--produced-by", default="/vega-forge")
    ap.add_argument("--domain-forge-dir", default=None)
    ap.add_argument("--skip-compile", action="store_true",
                    help="the caller has just run compile_spec.py on this spec")
    ap.add_argument("--render", default=str(RENDER_JS), help=argparse.SUPPRESS)
    ap.add_argument("--style", default=str(STYLE_CSS), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    model_path, out_path = Path(a.model), Path(a.out)
    if model_path.resolve() == out_path.resolve():
        print("ERROR: --out must differ from the input model (the input is never modified)")
        return 2
    try:
        html = model_path.read_text(encoding="utf-8")
        doc = json.loads(Path(a.data).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read the inputs: {exc}")
        return 2
    for asset in (Path(a.render), Path(a.style)):
        if not asset.is_file():
            print(f"ERROR: asset missing: {asset}")
            return 2
    render_js = Path(a.render).read_text(encoding="utf-8").rstrip()
    style_css = Path(a.style).read_text(encoding="utf-8").rstrip()
    if len(render_js.encode("utf-8")) > MAX_RENDER:
        print(f"ERROR: render script is {len(render_js.encode('utf-8'))} bytes; "
              f"contract §4 caps it at {MAX_RENDER // 1024} KB")
        return 2

    doc, errs = splice(doc, print, Path(a.data).resolve().parent)
    if not errs:
        doc, errs = splice_findings(doc, a.architect_run, print)
    if not errs:
        doc, errs = splice_findings(doc, a.improve_run, print, tool="improve")
    if not errs:
        errs = advise_layer(doc, "unattended" if a.unattended else "attended", print)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2
    doc.pop("from_run", None)

    if a.notebook:
        record, nb_err = notebook_record(a.notebook, print)
        if nb_err:
            print(f"ERROR: {nb_err}")
            return 2
        doc["notebook"] = record
        if not record["executed"]:
            markers = doc.setdefault("markers", [])
            if not any("notebook not executed" in str(m) for m in markers):
                markers.append("WARN: notebook not executed")

    spec_check = None if a.skip_compile else compile_ok(doc, model_path, print)
    errs = validate(doc, print, spec_check=spec_check)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2

    print_accepted(doc, a.data)

    data_text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    noscript = noscript_html(doc)
    df_dir = (Path(a.domain_forge_dir) if a.domain_forge_dir
              else SKILL_DIR.parent / "domain-forge")
    writer = df_dir / "scripts" / "apply_layer.py"
    if not writer.is_file():
        print(f"ERROR: platform writer not found at {writer}; /vega-forge writes layers only "
              "through domain-forge's apply_layer.py (pass --domain-forge-dir if the skill "
              "lives elsewhere)")
        return 3
    with tempfile.TemporaryDirectory() as td:
        result, how = platform_apply(writer, html, data_text, render_js, style_css,
                                     a.produced_by, noscript, td, model_path)
    if result is None:
        print(f"ERROR: platform writer {writer} failed ({how})")
        return 3
    if _BLOCK_RE.search(html):
        print("WARN: input already carried an architecture layer; it was replaced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"OK: wrote {out_path} via {how}")
    print(f"OK: render script {len(render_js.encode('utf-8'))} bytes, "
          f"data {len(data_text.encode('utf-8'))} bytes, "
          f"output {len(result.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
