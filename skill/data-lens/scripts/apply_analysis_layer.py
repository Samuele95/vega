#!/usr/bin/env python3
"""
apply_analysis_layer.py — validate a `layer-analysis-data` document and inject the /data-lens
`analysis` layer (data + fixed render script + style + <noscript>) into a domain-forge model.

    python3 apply_analysis_layer.py MODEL.html --data LAYER.json --out OUT.html
                                    [--produced-by /data-lens] [--domain-forge-dir PATH]

Validation follows `references/analysis-contract.md` §1: `ERROR:` and exit 2 on a schema
violation, `WARN:` for an optional section. The rules that are not merely structural, and the
reason each exists:

  * every module key is present, and a module that did not run says why — an absent module
    would be silence where the contract promises a statement;
  * a finding whose method reports a p-value carries an effect size AND the correction
    applied across its family — a p-value alone is the failure mode this skill exists against;
  * transformation-candidate ids are unique across findings and turns together, because
    `handoff.shaper_candidates` addresses them by id alone;
  * a turn that is not grounded carries no code and no result, and names the gap in its answer;
  * `source.seed` is present, because the transcript's reproducibility rests on it.

The analyst authors the JUDGEMENT — the readings, the findings, the so-whats, the turns'
answers — and never retypes the numbers. A layer document may therefore carry

    "from_analysis": "<run-dir>/analysis.json"

and this script splices in, from that file: `source` (path, rows, columns, seed, geometry),
`context` (typing, basis, derivations, partition, time, spatial — the author's `reading` is
kept), every module's `ran` / `skipped_because` / `evidence` (the author's `reading` is kept),
and the SVG of every figure `analysis.json` recorded, read from its `path`. That is what
"evidence verbatim" means mechanically: the numbers in the layer are the numbers the engine
wrote, because the same file wrote both. A figure may also be given inline as `svg`, or as a
`path` to an .svg file, and is inlined here.

The layer block is written ONLY by the platform writer domain-forge/scripts/apply_layer.py:
this script carries no copy of the block format or the digest algorithm. The input is never
modified.

Exit codes: 0 wrote OUT; 2 validation or input error; 3 platform writer missing or failed.
"""
from __future__ import annotations

import argparse
import html as _html
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL_DIR = HERE.parent
ASSETS = SKILL_DIR / "assets"
RENDER_JS = ASSETS / "analysis-render.js"
STYLE_CSS = ASSETS / "analysis-layer.css"
LAYER = "analysis"
SCHEMA = "data-lens/analysis@1"
MODULES = ("quality", "distributions", "relations", "inference", "segments", "importance",
           "time_series", "spatial", "drift")
SEVERITIES = {"high", "medium", "low"}
CHECKED = {"passed", "violated", "n/a"}
_BLOCK_RE = re.compile(r"[ \t]*<!--\s*@LAYER:start\s+" + LAYER +
                       r"\b[\s\S]*?@LAYER:end\s+" + LAYER + r"\s*-->[ \t]*\n?")
_P_KEYS = ("p", "p_value", "p_adj", "pvalue")


def _has_p(obj):
    if not isinstance(obj, dict):
        return False
    for k in _P_KEYS:
        if k in obj and obj[k] is not None:
            return True
    return False



def splice(doc, out, base_dir):
    """Fill the mechanical half of the layer from analysis.json (contract §1): the author
    writes judgement, the engine's numbers are spliced in verbatim. Returns the merged doc."""
    ref = doc.get("from_analysis")
    figs_by_id = {}
    if ref:
        ap = Path(ref)
        if not ap.is_absolute():
            ap = (Path(base_dir) / ref).resolve() if not ap.is_file() else ap
        if not ap.is_file():
            out(f"ERROR: from_analysis points at {ref} which does not exist")
            return doc, [f"from_analysis {ref!r} not found"]
        try:
            eng = json.loads(ap.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return doc, [f"from_analysis {ref!r} is not valid JSON: {e}"]
        if eng.get("schema") != SCHEMA:
            out(f"WARN: {ref} declares schema {eng.get('schema')!r}, not {SCHEMA!r}")
        esrc = eng.get("source") or {}
        src = dict(doc.get("source") or {})
        for k in ("path", "rows", "columns", "seed", "geometry"):
            if k in esrc and k not in src:
                src[k] = esrc[k]
        src.setdefault("analysis_json", str(ref))
        if isinstance(src.get("columns"), list):
            src["columns"] = len(src["columns"])
        doc["source"] = src
        ectx = eng.get("context") or {}
        ctx = dict(ectx)
        ctx.update({k: v for k, v in (doc.get("context") or {}).items() if v not in (None, [], {})})
        doc["context"] = ctx
        mods = dict(doc.get("modules") or {})
        for name, rec in (eng.get("modules") or {}).items():
            author = dict(mods.get(name) or {})
            merged = {"ran": rec.get("ran", False), "evidence": rec.get("evidence") or {}}
            if not merged["ran"]:
                merged["skipped_because"] = rec.get("skipped_because") or \
                    author.get("skipped_because") or "not run"
            if author.get("reading"):
                merged["reading"] = author["reading"]
            mods[name] = merged
        doc["modules"] = mods
        for f in eng.get("figures") or []:
            if isinstance(f, dict) and f.get("id"):
                figs_by_id[f["id"]] = dict(f)
        doc.setdefault("markers", eng.get("markers") or [])
    # figures: keep the author's list when there is one, else every figure the engine drew;
    # inline the SVG of anything given as a path.
    figures = doc.get("figures")
    if figures is None and figs_by_id:
        figures = [dict(v) for v in figs_by_id.values()]
    inlined, missing = 0, []
    for f in figures or []:
        if not isinstance(f, dict):
            continue
        if not f.get("svg"):
            src_f = f.get("path") or (figs_by_id.get(f.get("id")) or {}).get("path")
            if src_f:
                fp = Path(src_f)
                if not fp.is_absolute() and not fp.is_file():
                    fp = (Path(base_dir) / src_f)
                if fp.is_file():
                    f["svg"] = fp.read_text(encoding="utf-8")
                    inlined += 1
                else:
                    missing.append(str(src_f))
            for k in ("kind", "title", "alt"):
                if k not in f and k in (figs_by_id.get(f.get("id")) or {}):
                    f[k] = figs_by_id[f["id"]][k]
        f.pop("path", None)
    if figures is not None:
        doc["figures"] = [f for f in figures if isinstance(f, dict) and f.get("svg")]
    if inlined:
        out(f"OK: inlined {inlined} figure(s) from disk")
    for m in missing:
        out(f"WARN: figure file {m} not found; the figure is dropped from the layer")
    return doc, []

def validate(doc, out, step_catalog=None):
    errs = []
    if not isinstance(doc, dict):
        return ["layer document is not a JSON object"]
    for k in ("schema", "source", "context", "modules", "findings", "transcript", "handoff"):
        if k not in doc:
            errs.append(f"missing required top-level key {k!r}")
    if errs:
        return errs
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")

    src = doc["source"]
    if not isinstance(src, dict):
        errs.append("source must be an object")
    else:
        if not src.get("path"):
            errs.append("source.path is required")
        if src.get("geometry") not in ("present", "absent"):
            errs.append("source.geometry must be 'present' or 'absent'")
        if src.get("seed") is None:
            errs.append("source.seed is required — the transcript's determinism rests on it")

    ctx = doc["context"]
    if not isinstance(ctx, dict):
        errs.append("context must be an object")
    else:
        for k in ("typing", "basis", "partition"):
            if k not in ctx:
                errs.append(f"context.{k} is required")
        P = ctx.get("partition") or {}
        if isinstance(P, dict) and P.get("label"):
            feats = P.get("features") or []
            if P["label"] in feats:
                errs.append(f"context.partition: the label {P['label']!r} is among its own features")
            leaked = [c for c in (P.get("leakage") or []) if c in feats]
            if leaked:
                errs.append(f"context.partition: leakage column(s) {leaked} are still among the features")
        if not ctx.get("reading"):
            out("WARN: context.reading missing — the tab opens without saying what it inherits")

    mods = doc["modules"]
    if not isinstance(mods, dict):
        errs.append("modules must be an object")
    else:
        for m in MODULES:
            if m not in mods:
                errs.append(f"modules.{m} is missing (every module key must be present)")
                continue
            rec = mods[m]
            if not isinstance(rec, dict) or "ran" not in rec:
                errs.append(f"modules.{m} must be an object with 'ran'")
                continue
            if not rec.get("ran") and not rec.get("skipped_because"):
                errs.append(f"modules.{m}: ran is false but skipped_because is missing")
            if rec.get("ran") and not rec.get("reading"):
                out(f"WARN: modules.{m} ran but carries no reading")

    cand_ids, cand_ops = {}, []
    findings = doc["findings"]
    if not isinstance(findings, list):
        errs.append("findings must be a list")
        findings = []
    fids = set()
    for i, f in enumerate(findings):
        w = f"findings[{i}]"
        if not isinstance(f, dict):
            errs.append(f"{w} must be an object")
            continue
        for k in ("id", "module", "severity", "title", "columns", "evidence", "method",
                  "reading", "so_what"):
            if k not in f:
                errs.append(f"{w} missing {k!r}")
        _check_columns(f, w, errs)
        if f.get("id") in fids:
            errs.append(f"{w}: duplicate finding id {f.get('id')!r}")
        fids.add(f.get("id"))
        if f.get("severity") not in SEVERITIES:
            errs.append(f"{w}.severity must be one of {sorted(SEVERITIES)}, got {f.get('severity')!r}")
        if f.get("module") not in MODULES:
            errs.append(f"{w}.module {f.get('module')!r} is not one of {', '.join(MODULES)}")
        meth = f.get("method")
        if not isinstance(meth, dict):
            errs.append(f"{w}.method must be an object")
        else:
            if meth.get("assumptions_checked") not in CHECKED:
                errs.append(f"{w}.method.assumptions_checked must be one of {sorted(CHECKED)}, "
                            f"got {meth.get('assumptions_checked')!r}")
            if _has_p(f.get("evidence")) or _has_p(meth):
                ev = f.get("evidence") or {}
                if not (ev.get("effect") or ev.get("effect_size") or meth.get("effect")):
                    errs.append(f"{w}: reports a p-value but carries no effect size "
                                "(contract §2: a p-value never travels alone)")
                if not meth.get("correction"):
                    errs.append(f"{w}: reports a p-value but names no multiple-comparison "
                                "correction (use 'none — single pre-registered test' when there "
                                "was no family)")
        for j, t in enumerate(f.get("transformation_candidates") or []):
            _check_candidate(t, f"{w}.transformation_candidates[{j}]", errs, cand_ids, cand_ops)

    transcript = doc["transcript"]
    if not isinstance(transcript, list):
        errs.append("transcript must be a list")
        transcript = []
    for i, t in enumerate(transcript):
        w = f"transcript[{i}]"
        if not isinstance(t, dict):
            errs.append(f"{w} must be an object")
            continue
        for k in ("question", "answer", "grounded"):
            if k not in t:
                errs.append(f"{w} missing {k!r}")
        if t.get("grounded") is True:
            if not t.get("code"):
                errs.append(f"{w}: grounded turns carry the exact code that ran")
            if "result" not in t or t.get("result") is None:
                errs.append(f"{w}: grounded turns carry the result the answer was composed from")
        elif t.get("grounded") is False:
            if t.get("code") or t.get("result"):
                errs.append(f"{w}: an ungrounded turn must carry code: null and result: null")
        for j, c in enumerate(t.get("transformation_candidates") or []):
            _check_candidate(c, f"{w}.transformation_candidates[{j}]", errs, cand_ids, cand_ops)

    ho = doc["handoff"]
    if not isinstance(ho, dict) or "shaper_candidates" not in ho:
        errs.append("handoff must be an object with 'shaper_candidates' (an array, possibly empty, "
                    "in application order)")
    elif not isinstance(ho["shaper_candidates"], list):
        errs.append("handoff.shaper_candidates must be a list")
    else:
        missing = [c for c in ho["shaper_candidates"] if c not in cand_ids]
        if missing:
            errs.append(f"handoff.shaper_candidates names {missing}, which no finding or turn defines")

    for i, s in enumerate(doc.get("stances") or []):
        if isinstance(s, dict):
            _check_columns(s, f"stances[{i}]", errs)
        if not isinstance(s, dict) or not s.get("assertion"):
            errs.append("every stance must be an object with at least 'assertion' "
                        "(contract §1: stances are structured so /dataset-shaper can check a step "
                        "against them)")
            break
    figs = {f.get("id") for f in (doc.get("figures") or []) if isinstance(f, dict)}
    for i, f in enumerate(findings):
        for fid in (f or {}).get("figures") or []:
            if fid not in figs:
                out(f"WARN: findings[{i}] references figure {fid!r} which the layer does not carry")
    total = 0
    for f in doc.get("figures") or []:
        svg = (f or {}).get("svg") or ""
        n = len(svg.encode("utf-8"))
        total += n
        if n > 120 * 1024:
            errs.append(f"figure {f.get('id')!r} is {n} bytes; contract §1 caps a figure at 120 KB")
    if total > 1536 * 1024:
        errs.append(f"figures total {total} bytes; contract §1 caps them at 1.5 MB")
    if step_catalog and cand_ops:
        unknown = sorted({o for o in cand_ops if o not in step_catalog})
        if unknown:
            out(f"WARN: transformation candidate op(s) {unknown} are not in dataset-shaper's "
                f"step catalog; /dataset-shaper will treat them as custom steps")
    return errs


def _check_candidate(t, where, errs, cand_ids, cand_ops):
    if not isinstance(t, dict):
        errs.append(f"{where} must be an object")
        return
    if not t.get("op"):
        errs.append(f"{where} missing 'op'")
    else:
        cand_ops.append(t["op"])
    tid = t.get("id")
    if not tid:
        errs.append(f"{where} missing 'id' (ids are unique across findings and turns together, "
                    "because handoff.shaper_candidates addresses them by id alone)")
    elif tid in cand_ids:
        errs.append(f"{where}: duplicate transformation-candidate id {tid!r} "
                    f"(already defined at {cand_ids[tid]})")
    else:
        cand_ids[tid] = where
    if not t.get("rationale"):
        errs.append(f"{where} missing 'rationale' — the finding it comes from is its provenance")
    _check_columns(t, where, errs)


def _check_columns(obj, where, errs):
    """`columns`, where a finding, a candidate or a stance carries it, is a list of column names
    (contract §3: named columns): the page joins it, and a selector such as "numeric" written in
    its place passed this check and broke the rendered tab (F38, rehearsal final3r2)."""
    if "columns" not in obj:
        return
    cols = obj["columns"]
    if not isinstance(cols, list) or not all(isinstance(c, str) and c for c in cols):
        errs.append(f"{where}.columns must be a list of column names, got {cols!r} — name the "
                    "columns; a selector such as 'numeric' is not executable by /dataset-shaper")


def load_step_catalog():
    """The op vocabulary of /dataset-shaper, when that skill is a sibling. Advisory only."""
    p = SKILL_DIR.parent / "dataset-shaper" / "references" / "step-catalog.md"
    if not p.is_file():
        return None
    ops = set(re.findall(r"^\|\s*`([a-z_0-9]+)`\s*\|", p.read_text(encoding="utf-8"), re.M))
    return ops or None


def _esc(s):
    return _html.escape("" if s is None else str(s), quote=True)


def noscript_html(doc):
    P = (doc.get("context") or {}).get("partition") or {}
    parts = ['<section class="layer-analysis-noscript">', "<h2>Analysis</h2>",
             f"<p><em>{_esc((doc.get('source') or {}).get('path', 'dataset'))}</em></p>"]
    if (doc.get("context") or {}).get("reading"):
        parts.append(f"<p>{_esc(doc['context']['reading'])}</p>")
    parts.append("<h3>What this pass inherits</h3>")
    parts.append(f"<p>Basis: {_esc(', '.join((doc.get('context') or {}).get('basis') or []) or '—')}. "
                 f"Label: {_esc(P.get('label') or 'none')}"
                 + (f" ({_esc(P.get('task'))})" if P.get("task") else "")
                 + (f"; leakage dropped: {_esc(', '.join(P.get('leakage') or []))}"
                    if P.get("leakage") else "") + ".</p>")
    parts.append("<h3>Findings</h3>")
    if not doc.get("findings"):
        parts.append("<p>No finding was admitted.</p>")
    else:
        parts.append("<table><thead><tr><th>id</th><th>severity</th><th>module</th><th>title</th>"
                     "<th>columns</th><th>assumptions</th></tr></thead><tbody>")
        for f in doc["findings"]:
            m = f.get("method") or {}
            parts.append("<tr>" + "".join(f"<td>{_esc(x)}</td>" for x in (
                f.get("id"), f.get("severity"), f.get("module"), f.get("title"),
                ", ".join(f.get("columns") or []), m.get("assumptions_checked"))) + "</tr>")
        parts.append("</tbody></table>")
        for f in doc["findings"]:
            parts.append(f"<h4>{_esc(f.get('id'))} — {_esc(f.get('title'))}</h4>")
            parts.append(f"<p>{_esc(f.get('reading'))}</p>")
            sw = f.get("so_what") or {}
            if sw:
                parts.append("<dl>" + "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>"
                                              for k, v in sw.items()) + "</dl>")
    parts.append("<h3>Modules</h3><ul>")
    for name, m in (doc.get("modules") or {}).items():
        parts.append(f"<li><strong>{_esc(name)}</strong>: " +
                     (_esc(m.get("reading")) if m.get("ran")
                      else "skipped — " + _esc(m.get("skipped_because"))) + "</li>")
    parts.append("</ul>")
    if doc.get("transcript"):
        parts.append("<h3>Dialogue</h3>")
        for t in doc["transcript"]:
            parts.append(f"<h4>{_esc(t.get('turn'))}. {_esc(t.get('question'))}</h4>")
            parts.append(f"<p>{_esc(t.get('answer'))}</p>")
            if t.get("code"):
                parts.append(f"<pre>{_esc(t['code'])}</pre>")
    ho = doc.get("handoff") or {}
    parts.append("<h3>Hand-off</h3>")
    parts.append(f"<p>{_esc(', '.join(ho.get('shaper_candidates') or []) or 'no candidate')}. "
                 f"{_esc(ho.get('note') or '')}</p>")
    parts.append("<p>Enable JavaScript for the findings board, the module panels and the "
                 "dialogue transcript.</p></section>")
    return "\n".join(parts)


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
    except Exception as e:  # noqa: BLE001
        notes.append(f"import path failed: {e}")
    try:
        tmp = Path(tmpdir)
        d = tmp / "layer.json"; d.write_text(data_text, encoding="utf-8")
        r = tmp / "analysis-render.js"; r.write_text(render_js, encoding="utf-8")
        s = tmp / "analysis-layer.css"; s.write_text(style_css, encoding="utf-8")
        n = tmp / "noscript.html"; n.write_text(noscript, encoding="utf-8")
        o = tmp / "out.html"
        cmd = [sys.executable, str(writer), str(model_path), "--layer", LAYER, "--data", str(d),
               "--render", str(r), "--style", str(s), "--data-type", "application/json",
               "--produced-by", produced_by, "--noscript", str(n), "--out", str(o)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and o.is_file():
            return o.read_text(encoding="utf-8"), "platform writer (CLI)"
        notes.append(f"CLI path failed (exit {proc.returncode}): "
                     f"{(proc.stderr or proc.stdout).strip()[:300]}")
    except Exception as e:  # noqa: BLE001
        notes.append(f"CLI path failed: {e}")
    return None, "; ".join(notes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="input MODEL.html (never modified)")
    ap.add_argument("--data", required=True, help="layer-analysis-data JSON document")
    ap.add_argument("--out", required=True, help="output HTML path (must differ from MODEL)")
    ap.add_argument("--produced-by", default="/data-lens")
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
    except OSError as e:
        print(f"ERROR: cannot read model {model_path}: {e}")
        return 2
    try:
        data_text = Path(a.data).read_text(encoding="utf-8")
        doc = json.loads(data_text)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read layer data {a.data}: {e}")
        return 2
    for p in (Path(a.render), Path(a.style)):
        if not p.is_file():
            print(f"ERROR: asset missing: {p}")
            return 2
    render_js = Path(a.render).read_text(encoding="utf-8").rstrip()
    style_css = Path(a.style).read_text(encoding="utf-8").rstrip()
    if len(render_js.encode("utf-8")) > 60 * 1024:
        print(f"ERROR: render script is {len(render_js.encode('utf-8'))} bytes; "
              "contract §7 caps it at 60 KB")
        return 2

    doc, serrs = splice(doc, print, Path(a.data).resolve().parent)
    if serrs:
        for e in serrs:
            print(f"ERROR: {e}")
        return 2
    doc.pop("from_analysis", None)
    errs = validate(doc, print, load_step_catalog())
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2
    ran = [m for m, v in (doc.get("modules") or {}).items() if v.get("ran")]
    print(f"OK: {a.data} validates against {SCHEMA} ({len(doc.get('findings') or [])} findings, "
          f"{len(doc.get('transcript') or [])} turns, {len(ran)}/{len(MODULES)} modules ran)")

    data_text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    noscript = noscript_html(doc)
    df_dir = Path(a.domain_forge_dir) if a.domain_forge_dir else SKILL_DIR.parent / "domain-forge"
    writer = df_dir / "scripts" / "apply_layer.py"
    if not writer.is_file():
        print(f"ERROR: platform writer not found at {writer}; /data-lens writes layers only "
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
        print("WARN: input already carried an analysis layer; it was replaced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"OK: wrote {out_path} via {how}")
    print(f"OK: render script {len(render_js.encode('utf-8'))} bytes, "
          f"data {len(data_text.encode('utf-8'))} bytes, output {len(result.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
