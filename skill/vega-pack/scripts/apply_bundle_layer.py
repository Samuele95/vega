#!/usr/bin/env python3
"""
apply_bundle_layer.py — validate a `layer-bundle-data` document and inject the /vega-pack
`bundle` layer into the model.

    python3 apply_bundle_layer.py MODEL.html --data <bundle dir>/bundle-layer.json
                                  --out OUT.html [--notebook predict.ipynb]
                                  [--architect-run PATH ...] [--improve-run PATH ...]
                                  [--unattended] [--domain-forge-dir PATH]

The authoring file carries only the **judgement** — `readings.abstract` and
`readings.intended_use`, plus a `next` line for each finding the audits recorded (by id, or
by tool, audit and id) — and `"from_bundle": "<dir>"`. Everything else is spliced from the
bundle itself: `MANIFEST.json`, `schema.json`, the round-trip and parity results, the serve
record, the model-card and notebook records, and the markers `pack.py` wrote; and from the
audits: `code_findings` and, per tool, the `conformance` record vega-forge's
`audit_quorum.py` makes of the `/architect --baseline` audits named by `--architect-run` and
the `/improve` audits named by `--improve-run` (each flag once per audit, the first first):
every audit, and the matching of every structural or divergence finding across them. A
finding two of three audits confirm is **advisory** (code-contract §5, *Advisory*): with
`--unattended` it is published as a `WARN:` and an open item handed to /architect and /improve
(`conformance.<tool>.advisory.open_items`, the finding's `open_item`); attended, the user's
answer to its `FORK:` must be recorded in the authoring file's `forks` (`{"gate": "<tool>",
"findings": ["<audit>:<id>", …], "asked": …, "answer": "accept and record — …"}`) — *accept
and record* publishes, *fix now* and *stop* publish nothing.

What this script refuses to publish (bundle-contract §8):

  * **a round trip that did not pass.** The layer's whole claim is that what ships reproduces
    what was evaluated; publishing the claim without the proof is the one thing this stage
    cannot do. A bundle whose round trip failed does not get a layer, and the fix is the
    shaper's emitter or the copied file — never the tolerance;
  * a mechanical conformance result that is not `pass` (C0-C14 refuse on any finding); and,
    per tool — architect, improve — a judgement that did not run without its *absence* said
    out loud in `markers`, a first audit reporting a structural or divergence finding without
    the two fresh audits beside it, audits that are not one or three distinct runs (run ids
    included) of the same code, or a record whose advisory block does not name every confirmed
    finding as an open item (code-contract §5: a confirmed finding is advisory — unattended a
    WARN and an open item, never a refusal; attended, refused only without the user's recorded
    answer or with the answer *fix now* or *stop*; an unconfirmed one is recorded). A bundle
    that passed only the round trip is a working network beside the blueprint rather than in
    it;
  * a missing or unexecuted-and-unmarked notebook, a missing model card, and an
    `intended_use` reading that states a number — the card refuses that too, and for the same
    reason: every figure a reader sees must trace to a record that measured it;
  * an output the layer cannot read by its READING (`validate_reading`): a `MANIFEST.output`
    that is not vega/output@2 with a known reading, a `schema.json` output that does not
    describe exactly the field patterns the reading's fields expand from, a round trip whose
    compared columns and decisions are not the record's `compared` and `decided` pairs over the
    rows it compared, a lagged window bundle whose round trip records no lag boundary (R-A6),
    and a window bundle whose examples do not say how many context rows they ship first.

The block is written ONLY by `domain-forge/scripts/apply_layer.py`. The input is never modified.

Exit codes: 0 wrote OUT; 2 validation or input error; 3 platform writer missing or failed.
"""
from __future__ import annotations

import argparse
import functools
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

from output_record import schema_output  # noqa: E402
from vega_pack_core import digest_file, read_json  # noqa: E402

SKILL_DIR = HERE.parent
CATALOG = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
ASSETS = SKILL_DIR / "assets"
RENDER_JS = ASSETS / "bundle-render.js"
STYLE_CSS = ASSETS / "bundle-layer.css"

LAYER = "bundle"
SCHEMA = "vega/bundle-layer@1"
MAX_RENDER = 40 * 1024
MAX_DATA = 400 * 1024

READINGS = ("abstract", "intended_use")
REQUIRED = ("schema", "manifest", "input_schema", "roundtrip", "code_findings",
            "notebook", "usage", "readings")

_BLOCK_RE = re.compile(r"[ \t]*<!--\s*@LAYER:start\s+" + LAYER +
                       r"\b[\s\S]*?@LAYER:end\s+" + LAYER + r"\s*-->[ \t]*\n?")
_DIGIT = re.compile(r"\d")

_SPLICE = (("manifest", "MANIFEST.json"), ("input_schema", "schema.json"))


# --------------------------------------------------------------------------- splicing

def splice(doc, out, base_dir):
    """Replace `from_bundle` with what the bundle records. The judgement stays the author's."""
    src = doc.get("from_bundle")
    if not src:
        return doc, ["the authoring file carries no \"from_bundle\": the manifest, the schema "
                     "and the round trip are spliced from the bundle, never retyped into the "
                     "layer"]
    bundle = Path(src)
    if not bundle.is_absolute():
        bundle = next((c for c in (base_dir / src, Path.cwd() / src) if c.is_dir()),
                      base_dir / src)
    if not bundle.is_dir():
        return doc, [f"from_bundle {src!r} is not a directory (looked in {bundle})"]

    errs = []
    for key, name in _SPLICE:
        loaded, err = read_json(bundle / name, name)
        if err:
            errs.append(f"{err}; run pack.py before applying the layer — the layer publishes "
                        "what the bundle records")
        else:
            doc[key] = loaded
    if errs:
        return doc, errs

    manifest = doc["manifest"]
    doc["roundtrip"] = manifest.get("roundtrip") or {}
    doc["exports"] = manifest.get("exports") or []
    doc["serve"] = manifest.get("serve") or {}
    doc["usage"] = manifest.get("usage") or {}
    doc.setdefault("schema", SCHEMA)

    markers, _ = read_json(bundle / "markers.json", "markers.json")
    spliced = (markers or {}).get("markers") or []
    doc["markers"] = list(spliced) + [m for m in (doc.get("markers") or [])
                                      if m not in spliced]

    card = bundle / "model-card.html"
    if card.is_file():
        doc["model_card"] = {"path": str(Path(src) / "model-card.html"),
                             "digest": digest_file(card), "bytes": card.stat().st_size}
    out(f"OK: from_bundle: the manifest, the schema, the round trip and the markers spliced "
        f"from {bundle}")
    return doc, []


TOOLS = ("architect", "improve")
# the word, not the substring: the packer's markers name the `architecture` layer, and a test
# that accepted that as a declaration of the architect's absence would let the judgement half be
# skipped by coincidence
_TOOL_WORD = {tool: re.compile(rf"\b{tool}\b", re.IGNORECASE) for tool in TOOLS}
_GATE_RUNS = {"architect": "`/architect <bundle dir> --report-only --baseline <blueprint.md>`",
              "improve": "`/improve <bundle dir> --report-only`"}
_ROW_KEYS = ("anchor_justification", "title", "axis", "tags")


@functools.lru_cache(maxsize=1)
def _quorum():
    """vega-forge's `audit_quorum` — the one statement of the judgement half's rule
    (code-contract §5), which the forge's apply script reads too."""
    path = SKILL_DIR.parent / "vega-forge" / "scripts" / "audit_quorum.py"
    spec = importlib.util.spec_from_file_location("vega_forge_audit_quorum", str(path))
    if spec is None or not path.is_file():
        raise FileNotFoundError(f"vega-forge's audit_quorum.py is not at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _authored_next(doc):
    """The author's `next` lines, by (tool, audit, id); an entry naming no tool is the
    architect's and one naming no audit is the first audit's — merging by id rather than by
    position is what keeps a reordered findings.json from attaching the wrong plan to the wrong
    finding."""
    authored = {}
    for f in doc.get("code_findings") or []:
        if isinstance(f, dict) and f.get("id") is not None and f.get("next"):
            key = (f.get("tool") or "architect", int(f.get("audit") or 1), str(f.get("id")))
            authored[key] = f["next"]
    return authored


def _finding_row(raw, tool, audit, quorum):
    row = {"id": raw.get("id"), "tool": tool, "audit": audit,
           "depth": raw.get("depth") or raw.get("severity"),
           "where": raw.get("where") or raw.get("file") or raw.get("location"),
           "anchors": raw.get("anchors") or [], **{k: raw.get(k) for k in _ROW_KEYS}}
    if quorum is not None:
        row["quorum"] = {"reported_by": quorum["reported_by"], "confirmed": quorum["confirmed"]}
    return row


def audit_findings(runs_by_tool, doc, out):
    """The findings the architect and improve audits recorded, with the author's `next` lines
    merged by tool, audit and id, and each tool's judgement — every audit and the matching —
    as the layer's `conformance.<tool>` (code-contract §5: one clean first audit, or the first
    and two fresh ones; a structural or divergence finding refuses only when two of the three
    report it). The findings are the auditors' and are spliced; what to do about each is the
    skill's judgement and is authored. Returns the errors that stop the splice."""
    authored = _authored_next(doc)
    doc["code_findings"], doc["conformance"] = [], {}
    given = {tool: list(runs or []) for tool, runs in runs_by_tool.items() if runs}
    if not given:
        return []
    try:
        quorum = _quorum()
    except (FileNotFoundError, ImportError) as exc:
        return [f"{exc}; the judgement half's rule is read from vega-forge, never restated here"]
    for tool, runs in given.items():
        try:
            audits = [quorum.load_run(r, n) for n, r in enumerate(runs, 1)]
        except quorum.QuorumError as exc:
            return [f"--{tool}-run: {exc}; the judgement half of the gate is {_GATE_RUNS[tool]} "
                    "and its findings.json is what this splices"]
        judgement = quorum.judge(audits, tool)
        verdicts = {row["finding"]: row for row in judgement.record["matching"]}
        for audit in audits:
            for finding in audit.findings:
                row = _finding_row(finding.raw, tool, audit.index, verdicts.get(finding.key))
                plan = authored.get((tool, audit.index, finding.id))
                if plan:
                    row["next"] = plan
                doc["code_findings"].append(row)
        doc["conformance"][tool] = judgement.record
        _say_judgement(quorum, judgement.record, out)
    return []


PACK_REMEDY = ("fix the serving template through /architect or /improve (apply) on "
               "backends/<name>/templates/serving/ and pack.py --re-emit-serving — or, in a file "
               "the evaluation pinned, at the stage that wrote it — and run the gate again")


def advise_layer(doc, mode, out, remedy=PACK_REMEDY) -> list:
    """The advisory half (code-contract §5, *Advisory*), per judged tool: each confirmed finding
    becomes a WARN in `markers` and an open item (`conformance.<tool>.advisory`, the finding's
    `open_item`) — unattended always; attended only under the user's recorded *accept and
    record*. Returns the refusals (attended: no answer, *fix now*, *stop*)."""
    errs = []
    for tool in TOOLS:
        record = (doc.get("conformance") or {}).get(tool)
        if not isinstance(record, dict) or record.get("pending") or record.get("invalid"):
            continue
        advice = _quorum().advise(record, mode, doc.get("forks") or (), remedy)
        record["advisory"] = advice.record
        items = {item["finding"]: item["text"] for item in advice.open_items}
        for row in doc.get("code_findings") or []:
            if row.get("tool") == tool and f"{row.get('audit')}:{row.get('id')}" in items:
                row["open_item"] = items[f"{row.get('audit')}:{row.get('id')}"]
        markers = doc.setdefault("markers", [])
        markers += [w for w in advice.warns if w not in markers]
        for warn in advice.warns:
            out(warn)
        errs += list(advice.errors)
    return errs


def _say_judgement(quorum, rec, out):
    tool = rec["tool"]
    for row in rec["matching"]:
        out(f"{quorum.row_level(row)}: {tool} {row['finding']} reported by audit(s) "
            f"{row['reported_by']} of {len(rec['audits'])} — {quorum.row_state(row)}")
    out(f"OK: spliced {tool}: {len(rec['audits'])} audit(s), "
        f"{sum(a['findings'] for a in rec['audits'])} finding(s); confirmed structural "
        f"{rec['structural']}, divergence {rec['divergence']}; unconfirmed "
        f"{rec['unconfirmed']['structural']} structural, {rec['unconfirmed']['divergence']} "
        f"divergence; recorded {rec['recorded']}")


def judgement_problems(doc, manifest_record, markers) -> list:
    """The judgement half of the bundle's gate, per tool: the record the audits made, or the
    tool's absence said out loud in `markers` — never silently skipped. A count `pack.py`
    never writes (`blueprint.conformance.architect` is null there) still refuses if a manifest
    carries one."""
    errs = []
    if isinstance(manifest_record, dict):
        errs += [f"MANIFEST.blueprint.conformance.architect carries {manifest_record[d]} {d} "
                 "finding(s), a count pack.py never writes: the judgement half is recorded from "
                 "the audits (--architect-run), never typed into the manifest — a malformed record"
                 for d in ("structural", "divergence") if int(manifest_record.get(d) or 0) > 0]
    for tool in TOOLS:
        record = (doc.get("conformance") or {}).get(tool)
        if record is None:
            if not any(_TOOL_WORD[tool].search(m) for m in markers):
                errs.append(f"no {tool} run is recorded and no marker says so; the judgement "
                            "half of the conformance gate either ran or is declared absent — it "
                            "is never silently skipped")
            continue
        errs += _record_problems(tool, record)
    return errs


def _record_problems(tool, record) -> list:
    if record.get("pending"):
        return [f"conformance.{tool}: {record['pending']} — run {_GATE_RUNS[tool]} twice more, "
                "each in a fresh context that has not read the earlier audits, and pass all three "
                f"--{tool}-run in order (code-contract §5)"]
    errs = [f"conformance.{tool}: {p}" for p in record.get("invalid") or []]
    if not record.get("audits"):
        errs.append(f"conformance.{tool} carries no audits; the judgement is spliced from the "
                    f"runs (--{tool}-run, once per audit), never typed into the layer")
    # a confirmed finding is advisory: what refuses is an advisory block that does not carry it
    return errs + _quorum().advisory_problems(record)


# --------------------------------------------------------------------------- validation

def validate(doc, out):
    errs = []
    missing = [k for k in REQUIRED
               if k not in doc or (doc.get(k) in (None, {}, "") and k != "code_findings")]
    if missing:
        errs.append(f"the layer document is missing {', '.join(missing)}; bundle-contract §8 "
                    "names them as required")
        return errs

    trip = doc["roundtrip"] or {}
    if trip.get("status") != "pass":
        errs.append(
            f"the round trip's status is {trip.get('status')!r}, not 'pass'. The layer's claim "
            "is that what ships reproduces what was evaluated; a bundle that does not "
            "reproduce it does not get a layer saying it does. Fix the shaper's emitter or "
            "the copied file — the tolerance is the contract's formula and not a knob"
            + (f" (first disagreement: {trip.get('first_disagreement')})"
               if trip.get("first_disagreement") else ""))
    if trip.get("shaper_transform_check") not in ("pass", None):
        errs.append(f"the shaper's twin check is {trip.get('shaper_transform_check')!r}: the "
                    "bundle's transform does not reproduce the shaped columns it was built "
                    "from, so the round trip below it means nothing")

    manifest = doc["manifest"]
    errs += validate_reading(manifest, doc["input_schema"], trip)
    conformance = ((manifest.get("blueprint") or {}).get("conformance") or {})
    markers = [str(m) for m in (doc.get("markers") or [])]
    if conformance.get("mechanical") != "pass":
        errs.append(
            f"the bundle's mechanical conformance is {conformance.get('mechanical')!r}. The "
            "round trip proves the numbers and the gate proves the architecture; a bundle "
            "that passes only the first is a working network beside the blueprint rather "
            "than in it"
            + (f" — {conformance.get('why')}" if conformance.get("why") else ""))
    errs += judgement_problems(doc, conformance.get("architect"), markers)

    # the shape the manifest claims and the entry points it lists have to be the same claim:
    # a card that says `full` beside a bundle with only predict.py sends a reader looking for
    # a training loop that is not there
    shape = manifest.get("include") or "serving"
    entrypoints = set(manifest.get("entrypoints") or [])
    wanted = set(CATALOG["include"]["modes"].get(shape, {}).get("entrypoints") or [])
    if wanted - entrypoints:
        errs.append(f"the manifest says include {shape!r}, which writes "
                    f"{', '.join(sorted(wanted))}, but its entrypoints are "
                    f"{', '.join(sorted(entrypoints)) or 'none'}; the shape a reader is told "
                    "about and the files that are there must be the same bundle")

    card = doc.get("model_card") or {}
    if not card.get("path") or not card.get("digest"):
        errs.append("the layer carries no model card record; the card is the document a "
                    "reader opens without the chain, and a bundle layer without one publishes "
                    "numbers with nowhere to check them")

    readings = doc.get("readings") or {}
    for name in READINGS:
        text = readings.get(name)
        if not isinstance(text, str) or len(text.strip()) < 40:
            errs.append(f"readings.{name} is missing or too short to be a reading")
    if errs:
        return errs

    # the same rule the model card enforces, for the same reason
    digits = _DIGIT.findall(readings["intended_use"])
    if digits:
        errs.append(
            "readings.intended_use states a number. Every other figure in this layer and on "
            "the model card is spliced from a record that measured it; a digit typed into the "
            "hand-written passage would be the one number tracing to nobody, and it is the "
            "one a reader would most want to check. Say it in words, or point at the ranges "
            "the schema table already carries")

    notebook = doc.get("notebook") or {}
    if not notebook.get("executed"):
        if not any("notebook not executed" in m for m in markers):
            errs.append("the notebook was not executed and `markers` does not say so; §8 "
                        "requires the marker beside an unexecuted notebook")
        else:
            out("WARN: the prediction notebook was not executed — the layer says so in its "
                "markers, and the usage below was not run")
    return errs


def validate_reading(manifest, schema, trip) -> list:
    """The output as its reading says it is (bundle-contract §2.1, §5): the record, the
    patterns `schema.json` describes, and the round trip's pairs, rows and boundary."""
    record = manifest.get("output") or {}
    readings = CATALOG["output_readings"]
    if record.get("schema") != readings["schema"] or \
            record.get("reading") not in readings["readings"]:
        return [f"MANIFEST.output is {record.get('schema')!r} with reading "
                f"{record.get('reading')!r}; the layer reads a {readings['schema']} record of "
                f"one of the readings {', '.join(readings['readings'])} — re-pack with "
                "vega-pack v2.3"]
    errs = []
    try:
        want = sorted(schema_output(record, manifest.get("sample"))["output"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"MANIFEST.output / MANIFEST.sample cannot be read by the {record['reading']} "
                f"reading's patterns ({type(exc).__name__}: {exc}) — re-pack with vega-pack v2.3"]
    have = sorted((schema.get("output") or {}).keys())
    if want != have:
        errs.append(f"schema.json describes the output patterns {have}, and the "
                    f"{record['reading']} reading's fields expand from {want}; a caller reads "
                    "the fields through those descriptions")
    return errs + _roundtrip_reading(record, manifest, trip)


def _roundtrip_reading(record, manifest, trip) -> list:
    errs, pairs = [], (len(record.get("compared") or ()), len(record.get("decided") or ()))
    boundary = trip.get("boundary") or {}
    compared = int(trip.get("rows") or 0) - int(boundary.get("rows") or 0)
    decisions = compared if pairs[1] else 0
    if trip.get("columns_compared") != pairs[0] or trip.get("decisions_compared") != decisions:
        errs.append(f"the round trip compared {trip.get('columns_compared')} column(s) and "
                    f"{trip.get('decisions_compared')} decision(s); the record's "
                    f"{pairs[0]} compared and {pairs[1]} decided pair(s) over {compared} "
                    f"compared row(s) give {pairs[0]} and {decisions}")
    sample = manifest.get("sample") or {}
    window = sample.get("shape") == "window"
    lagged = window and int(sample.get("lag_depth") or 0) > 0 \
        and (sample.get("context") or "previous-part") == "previous-part"
    if lagged and not (isinstance(boundary.get("rows"), int) and boundary.get("reason")):
        errs.append("a lagged window bundle's round trip records no boundary {rows, reason}: "
                    "the rows R-A6 excludes must be said, never silently dropped")
    if window and (manifest.get("examples") or {}).get("context_rows") != \
            int(sample.get("length") or 1) - 1:
        errs.append("a window bundle's MANIFEST.examples.context_rows is not length − 1: the "
                    "shipped rows must say how many context rows come first")
    return errs


def _reading_line(manifest, trip) -> str:
    record, boundary = manifest.get("output") or {}, trip.get("boundary") or {}
    return (f"<p>Reading <strong>{_esc(record.get('reading'))}</strong>: "
            f"{_esc(len(record.get('fields') or []))} field(s); the round trip compared "
            f"{_esc(trip.get('columns_compared'))} field(s) and "
            f"{_esc(trip.get('decisions_compared'))} decision(s)"
            + (f", {_esc(boundary.get('rows'))} row(s) at the lag boundary not compared "
               f"({_esc(boundary.get('reason'))})" if boundary else "") + ".</p>")


# --------------------------------------------------------------------------- noscript

def _esc(value):
    return _html.escape("" if value is None else str(value), quote=True)


def _open_items_html(doc) -> str:
    """The confirmed judgement findings the layer carries as open items (code-contract §5,
    *Advisory*), so a reader without JavaScript sees them too."""
    conformance = doc.get("conformance") or {}
    records = [conformance.get(tool) for tool in TOOLS]
    items = [item for rec in records if isinstance(rec, dict)
             for item in (rec.get("advisory") or {}).get("open_items") or []]
    if not items:
        return ""
    return ("<h4>Open items for /architect and /improve</h4><ul>"
            + "".join(f"<li>{_esc(i.get('text'))}</li>" for i in items) + "</ul>")


def noscript_html(doc):
    manifest = doc["manifest"]
    chain = (manifest.get("chain") or {}).get("evaluation") or {}
    verdict = chain.get("verdict") or {}
    trip = doc["roundtrip"]
    usage = doc.get("usage") or {}
    rows = "".join(
        f"<tr><th scope=\"row\">{_esc(c.get('name'))}</th><td>{_esc(c.get('dtype'))}</td>"
        f"<td>{_esc(c.get('allowed') if c.get('allowed') is not None else c.get('range_seen'))}"
        "</td></tr>"
        for c in (doc["input_schema"].get("input") or []))
    stances = "".join(f"<li>{_esc(s)}</li>" for s in (manifest.get("stances") or []))
    return (
        "<section class=\"layer-bundle-noscript\">"
        f"<h3>Bundle {_esc(manifest.get('slug'))} — run {_esc(manifest.get('run'))}</h3>"
        f"<p><strong>{'Earned' if verdict.get('earned') else 'NOT earned'}</strong>"
        + (f" — {_esc(verdict.get('strongest_baseline'))} beats the network on the held-out "
           "part, and this bundle ships it anyway."
           if verdict.get("beaten_by_baseline") else
           f" against {_esc(verdict.get('strongest_baseline'))}.")
        + f" The test part was opened {_esc(chain.get('seal_exposures'))} time(s).</p>"
        + (f"<p>Shipped under: <ul>{stances}</ul></p>" if stances else "")
        + f"<p>Round trip: <strong>{_esc(trip.get('status'))}</strong> on "
          f"{_esc(trip.get('rows'))} row(s), worst |&#916;| {_esc(trip.get('max_abs_diff'))} "
          f"against {_esc((trip.get('tolerance') or {}).get('rule'))}.</p>"
        + _reading_line(manifest, trip)
        + f"<table><caption>Input schema</caption><tbody>{rows}</tbody></table>"
        + f"<h4>How to use</h4><pre>{_esc(usage.get('cli'))}\n{_esc(usage.get('module'))}</pre>"
        + _open_items_html(doc)
        + f"<h4>Intended use and limits</h4><p>{_esc((doc.get('readings') or {}).get('intended_use'))}</p>"
        + f"<p>The full record is in {_esc((doc.get('model_card') or {}).get('path'))}.</p>"
        "</section>")
def platform_apply(writer, html, data_text, render_js, style_css, produced_by, noscript,
                   tmpdir, model_path):
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
        files = {}
        for name, text in (("layer.json", data_text), ("bundle-render.js", render_js),
                           ("bundle-layer.css", style_css), ("noscript.html", noscript)):
            files[name] = tmp / name
            files[name].write_text(text, encoding="utf-8")
        out_file = tmp / "out.html"
        proc = subprocess.run(
            [sys.executable, str(writer), str(model_path), "--layer", LAYER,
             "--data", str(files["layer.json"]),
             "--render", str(files["bundle-render.js"]),
             "--style", str(files["bundle-layer.css"]),
             "--data-type", "application/json", "--produced-by", produced_by,
             "--noscript", str(files["noscript.html"]), "--out", str(out_file)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and out_file.is_file():
            return out_file.read_text(encoding="utf-8"), "platform writer (CLI)"
        notes.append(f"CLI path failed (exit {proc.returncode}): "
                     f"{(proc.stderr or proc.stdout).strip()[:300]}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"CLI path failed: {exc}")
    return None, "; ".join(notes)


def notebook_record(path, out):
    p = Path(path)
    if not p.is_file():
        return None, f"--notebook {p} not found; the stage order is pack → card → notebook → apply"
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{p} is not a readable notebook: {exc}"
    code = [c for c in nb.get("cells") or [] if c.get("cell_type") == "code"]
    outputs = [c.get("outputs") or [] for c in code]
    errored = any(o.get("output_type") == "error" for cell in outputs for o in cell)
    executed = bool(code) and all(cell for cell in outputs) and not errored
    # the notebook's last cell prints `VEGA-PACK: reproduced` when the bundle's own predict
    # path matched the stored expectations from inside the bundle, `mismatch` when it did not
    text = "\n".join("".join(o.get("text") or []) for cell in outputs for o in cell)
    marker = re.search(r"VEGA-PACK:\s*(reproduced|mismatch)\b", text)
    reproduced = (marker.group(1) == "reproduced") if marker else None
    if marker is None and executed:
        out("WARN: the prediction notebook carries no VEGA-PACK marker; it was emitted by an "
            "older nb_emit.py, and whether it reproduced the bundle's own examples is unknown "
            "rather than proven")
    if not executed:
        out("WARN: notebook not executed" + (" — a cell ended in an error output" if errored
                                             else ""))
    return {"path": str(p), "executed": executed, "digest": digest_file(p),
            "reproduced": reproduced, "code_cells": len(code)}, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="input MODEL.html (never modified)")
    ap.add_argument("--data", required=True, help="the bundle's bundle-layer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--notebook", default=None, help="predict.ipynb (executed)")
    ap.add_argument("--architect-run", action="append", default=None,
                    help="an .claude/architect-runs/<ts> whose findings.json is spliced; once "
                         "per audit, the first audit first")
    ap.add_argument("--improve-run", action="append", default=None,
                    help="an .claude/improve-runs/<ts> whose findings.json is spliced; once per "
                         "audit, the first audit first")
    ap.add_argument("--unattended", action="store_true",
                    help="the stage runs unattended: a confirmed judgement finding is a WARN and "
                         "an open item; attended (the default) it needs the user's recorded "
                         "FORK answer")
    ap.add_argument("--produced-by", default="/vega-pack")
    ap.add_argument("--domain-forge-dir", default=None)
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
    if not isinstance(doc, dict):
        print(f"ERROR: {a.data} parses to a {type(doc).__name__}, not an object; the authoring "
              "file is a JSON object carrying `from_bundle` and `readings`")
        return 2
    for asset in (Path(a.render), Path(a.style)):
        if not asset.is_file():
            print(f"ERROR: asset missing: {asset}")
            return 2
    render_js = Path(a.render).read_text(encoding="utf-8").rstrip()
    style_css = Path(a.style).read_text(encoding="utf-8").rstrip()
    if len(render_js.encode("utf-8")) > MAX_RENDER:
        print(f"ERROR: render script is {len(render_js.encode('utf-8'))} bytes; contract §8 "
              f"caps it at {MAX_RENDER // 1024} KB")
        return 2

    doc, errs = splice(doc, print, Path(a.data).resolve().parent)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2
    errs = audit_findings({"architect": a.architect_run, "improve": a.improve_run}, doc, print)
    if not errs:
        errs = advise_layer(doc, "unattended" if a.unattended else "attended", print)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2
    doc.pop("from_bundle", None)

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

    errs = validate(doc, print)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2

    data_text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    if len(data_text.encode("utf-8")) > MAX_DATA:
        print(f"ERROR: the layer's data is {len(data_text.encode('utf-8')) // 1024} KB; "
              f"contract §8 caps it at {MAX_DATA // 1024} KB — the model card and the weights "
              "stay on disk, so a document this large means something else was inlined")
        return 2

    manifest, trip = doc["manifest"], doc["roundtrip"]
    verdict = ((manifest.get("chain") or {}).get("evaluation") or {}).get("verdict") or {}
    print(f"OK: {a.data} validates against {SCHEMA} (bundle {manifest.get('slug')}-"
          f"{manifest.get('run')}, round trip {trip.get('status')} on {trip.get('rows')} row(s),"
          f" conformance {((manifest.get('blueprint') or {}).get('conformance') or {}).get('mechanical')},"
          f" {len(doc.get('code_findings') or [])} code finding(s), "
          f"{'earned' if verdict.get('earned') else 'NOT earned'}, notebook executed "
          f"{(doc.get('notebook') or {}).get('executed')})")
    if manifest.get("stances"):
        for stance in manifest["stances"]:
            print(f"WARN: this bundle ships under a recorded stance: {stance}")

    noscript = noscript_html(doc)
    df_dir = (Path(a.domain_forge_dir) if a.domain_forge_dir
              else SKILL_DIR.parent / "domain-forge")
    writer = df_dir / "scripts" / "apply_layer.py"
    if not writer.is_file():
        print(f"ERROR: the platform writer is not at {writer}; the layer block is written "
              "only by domain-forge/scripts/apply_layer.py, never by this script")
        return 3
    with tempfile.TemporaryDirectory() as tmpdir:
        result, how = platform_apply(writer, html, data_text, render_js, style_css,
                                     a.produced_by, noscript, tmpdir, model_path)
    if result is None:
        print(f"ERROR: the platform writer did not produce a layer block: {how}")
        return 3
    if _BLOCK_RE.search(html):
        print("WARN: the input already carried a bundle layer; it was replaced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"OK: wrote {out_path} via {how}")
    print(f"OK: render script {len(render_js.encode('utf-8'))} bytes, data "
          f"{len(data_text.encode('utf-8')) // 1024} KB of the {MAX_DATA // 1024} KB the "
          f"contract allows, output {len(result.encode('utf-8')) // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
