#!/usr/bin/env python3
"""
apply_shape_layer.py — validate a `layer-shape-data` document and inject the /dataset-shaper
`shape` layer (data + fixed render script + style + <noscript>) into a domain-forge model.

    python3 apply_shape_layer.py MODEL.html --data LAYER.json --out OUT.html
                                 [--produced-by /dataset-shaper] [--domain-forge-dir PATH]

The layer document may carry `"from_run": "<out-dir>"` instead of retyping the executor's
output: the recipe, the manifest, the lineage and the verification report are then read from
that directory (`recipe.json`, `manifest.json`, `lineage.json`, `verification.json`) and
spliced in verbatim, and `manifest_digest` is computed from the `manifest.json` FILE there (the
sha256 of its bytes as the executor wrote them; a digest the document already carries must
equal it). The author writes the judgement — the abstract, the per-phase readings,
the forks and what answered them — and the numbers stay the executor's.

Validation follows `references/shape-contract.md` §4, and the rules that are not merely
structural are the ones the skill exists to keep:

  * every recipe step carries a source from the closed set, and no step is untraced;
  * `verification.structural` and `verification.determinism` are present — a layer that does
    not say whether the output was checked is worse than no layer;
  * a structural, split, determinism, spatial or graph failure may not be published as if it
    had passed;
  * `before_after` and `readings.abstract` are present, because a reader must be able to see
    what the dataset became without opening the manifest;
  * `manifest_digest` is present as `sha256:<64 hex>`: the one digest /blueprint
    (`pipeline.json.input.shape_manifest_digest`), compile_spec.py and the emitted
    FeatureContract compare. It is the manifest FILE's digest, never recomputed from the
    spliced copy, so a document without `from_run` must carry it.

The block is written ONLY by the platform writer domain-forge/scripts/apply_layer.py. The
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
SKILL_DIR = HERE.parent
ASSETS = SKILL_DIR / "assets"
RENDER_JS = ASSETS / "shape-render.js"
STYLE_CSS = ASSETS / "shape-layer.css"
LAYER = "shape"
SCHEMA = "dataset-shaper/shape@1"
SOURCE_RE = re.compile(r"^(geometry:[A-Za-z0-9_./\-]+|analysis:(F\d+|turn\d+)/T\w+|"
                       r"analysis:[A-Za-z0-9_./\-]+|user:[^\s].*|shaper:default)$")
_BLOCK_RE = re.compile(r"[ \t]*<!--\s*@LAYER:start\s+" + LAYER +
                       r"\b[\s\S]*?@LAYER:end\s+" + LAYER + r"\s*-->[ \t]*\n?")
MAX_FITTED = 20 * 1024
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def file_digest(path):
    """`sha256:<hex>` of a file's bytes as written: the shape layer's `manifest_digest`."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def splice(doc, out, base_dir):
    """Fill the mechanical half from the executor's output directory."""
    ref = doc.get("from_run")
    if not ref:
        return doc, []
    rd = Path(ref)
    if not rd.is_absolute() and not rd.is_dir():
        rd = (Path(base_dir) / ref)
    if not rd.is_dir():
        return doc, [f"from_run {ref!r} is not a directory"]
    for key, fname in (("recipe", "recipe.json"), ("manifest", "manifest.json"),
                       ("lineage", "lineage.json"), ("verification", "verification.json")):
        p = rd / fname
        if key in doc and doc[key]:
            continue
        if not p.is_file():
            if key in ("recipe", "manifest"):
                return doc, [f"from_run: {p} not found (the executor writes it)"]
            out(f"WARN: from_run: {fname} not found; the layer carries no {key} section")
            continue
        try:
            doc[key] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return doc, [f"from_run: {p} is not valid JSON: {e}"]
    if (rd / "manifest.json").is_file():
        # the digest of the FILE the executor wrote, never of the spliced (and possibly elided)
        # copy: it is the one digest the blueprint, compile_spec.py and the FeatureContract share
        written = file_digest(rd / "manifest.json")
        given = doc.get("manifest_digest")
        if given and given != written:
            return doc, [f"manifest_digest {given} is not the digest of {rd / 'manifest.json'} "
                         f"({written}); leave it out and the apply script writes it from the file"]
        doc["manifest_digest"] = written
    man = doc.get("manifest") or {}
    # keep the layer readable: large fitted blobs stay on disk and are referenced
    for st in man.get("steps") or []:
        blob = json.dumps(st.get("parameters_fitted") or {})
        if len(blob.encode("utf-8")) > MAX_FITTED:
            st["parameters_fitted"] = {
                "__elided__": True, "bytes": len(blob.encode("utf-8")),
                "see": str(Path(man.get("outputs", {}).get("dir", str(rd))) /
                           man.get("outputs", {}).get("fitted", "fitted.json")),
                "keys": sorted(list((st.get("parameters_fitted") or {}).keys()))[:20]}
            out(f"OK: step {st.get('id')}: fitted parameters elided from the layer "
                f"({len(blob)} bytes) and referenced on disk")
    if "before_after" not in doc:
        inp = man.get("input") or {}
        files = (man.get("outputs") or {}).get("files") or {}
        first = next(iter(files.values()), {})
        added = []
        for st in man.get("steps") or []:
            added += st.get("columns_added") or []
        # the edges part (graph_edges, shape-contract §3) holds edges, not dataset rows
        doc["before_after"] = {
            "rows": [inp.get("rows"), sum(f.get("rows", 0) for name, f in files.items()
                                          if name != "edges")],
            "columns": [inp.get("columns"), first.get("columns")],
            "added": sorted(set(added))}
    if "verification" not in doc:
        doc["verification"] = {"structural": "not run", "determinism": "not run",
                               "markers": ["WARN: verify_shape.py was not run"]}
    return doc, []


def validate(doc, out):
    errs = []
    if not isinstance(doc, dict):
        return ["layer document is not a JSON object"]
    for k in ("schema", "recipe", "manifest", "manifest_digest", "verification", "before_after",
              "readings"):
        if k not in doc:
            errs.append(f"missing required top-level key {k!r}")
    if errs:
        return errs
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")
    rec = doc["recipe"]
    if not isinstance(rec, dict) or not isinstance(rec.get("steps"), list) or not rec["steps"]:
        errs.append("recipe.steps must be a non-empty list")
    else:
        for i, s in enumerate(rec["steps"]):
            if not isinstance(s, dict):
                errs.append(f"recipe.steps[{i}] must be an object")
                continue
            if not s.get("source"):
                errs.append(f"recipe.steps[{i}] ({s.get('id')}) carries no source; an untraced "
                            "step is the one thing this layer may never publish")
            elif not SOURCE_RE.match(str(s["source"])):
                errs.append(f"recipe.steps[{i}]: source {s['source']!r} is not one of "
                            "geometry:… / analysis:… / user:… / shaper:default")
            if not s.get("rationale"):
                out(f"WARN: recipe.steps[{i}] ({s.get('id')}) carries no rationale")
    man = doc["manifest"]
    if not isinstance(man, dict) or "outputs" not in man:
        errs.append("manifest must be the executor's manifest.json (with 'outputs')")
    if not DIGEST_RE.match(str(doc["manifest_digest"])):
        errs.append(f"manifest_digest {doc['manifest_digest']!r} is not 'sha256:<64 hex>' — it is "
                    "the sha256 of the manifest.json file as written (set from_run and the apply "
                    "script computes it)")
    V = doc["verification"]
    if not isinstance(V, dict):
        errs.append("verification must be an object")
    else:
        for k in ("structural", "determinism"):
            if k not in V:
                errs.append(f"verification.{k} is required — a layer that does not say whether "
                            "the output was checked is worse than no layer")
        for k in ("structural", "split", "determinism", "spatial", "graph"):
            if str(V.get(k, "")).lower() == "fail":
                errs.append(f"verification.{k} is 'fail': fix the recipe and re-run rather than "
                            "publishing a layer that records a failure as a result")
        for r in V.get("semantic") or []:
            if isinstance(r, dict) and r.get("empirical") == "refuted":
                errs.append(f"verification.semantic: rule {r.get('rule_id')!r} is refuted on the "
                            "output; a broken definitional relationship is not shippable")
            if isinstance(r, dict) and r.get("symbolic") not in (None, "confirmed", "refuted",
                                                                 "untested"):
                errs.append(f"verification.semantic: symbolic status {r.get('symbolic')!r} is not "
                            "one of confirmed / refuted / untested")
    ba = doc["before_after"]
    if not isinstance(ba, dict) or "rows" not in ba or "columns" not in ba:
        errs.append("before_after needs at least 'rows' and 'columns'")
    rd = doc["readings"]
    if not isinstance(rd, dict) or not rd.get("abstract"):
        errs.append("readings.abstract is required — it is what a reader sees first")
    for f in doc.get("forks") or []:
        if not isinstance(f, dict) or not (f.get("answer") or f.get("or")):
            errs.append("every fork records what answered it ('answer', or 'or' for a default "
                        "applied unattended)")
            break
    return errs


def _esc(s):
    return _html.escape("" if s is None else str(s), quote=True)


def noscript_html(doc):
    rec, man = doc.get("recipe") or {}, doc.get("manifest") or {}
    V, BA = doc.get("verification") or {}, doc.get("before_after") or {}
    p = ['<section class="layer-shape-noscript">', "<h2>Shape</h2>",
         f"<p>{_esc((doc.get('readings') or {}).get('abstract', ''))}</p>",
         "<h3>Before and after</h3>",
         f"<p>rows {_esc((BA.get('rows') or [None, None])[0])} → "
         f"{_esc((BA.get('rows') or [None, None])[1])}; columns "
         f"{_esc((BA.get('columns') or [None, None])[0])} → "
         f"{_esc((BA.get('columns') or [None, None])[1])}.</p>",
         "<h3>The recipe</h3>",
         "<table><thead><tr><th>id</th><th>op</th><th>columns</th><th>source</th>"
         "<th>rationale</th></tr></thead><tbody>"]
    for s in rec.get("steps") or []:
        p.append("<tr>" + "".join(f"<td>{_esc(x)}</td>" for x in (
            s.get("id"), s.get("op"), ", ".join(s.get("columns") or []), s.get("source"),
            s.get("rationale"))) + "</tr>")
    p.append("</tbody></table>")
    p.append("<h3>Outputs</h3><ul>")
    for name, f in ((man.get("outputs") or {}).get("files") or {}).items():
        p.append(f"<li>{_esc(name)}: {_esc(f.get('path'))} — {_esc(f.get('rows'))} rows × "
                 f"{_esc(f.get('columns'))} columns, {_esc(f.get('digest'))}</li>")
    p.append("</ul>")
    p.append("<h3>Verification</h3><ul>")
    for k in ("structural", "split", "determinism", "spatial"):
        if k in V:
            p.append(f"<li>{_esc(k)}: {_esc(V[k])}</li>")
    for r in V.get("semantic") or []:
        p.append(f"<li>{_esc(r.get('rule_id'))}: empirical {_esc(r.get('empirical'))}, "
                 f"symbolic {_esc(r.get('symbolic'))}</li>")
    p.append("</ul>")
    p.append("<p>Enable JavaScript for the recipe, the lineage graph and the verification "
             "panel.</p></section>")
    return "\n".join(p)


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
        r = tmp / "shape-render.js"; r.write_text(render_js, encoding="utf-8")
        st = tmp / "shape-layer.css"; st.write_text(style_css, encoding="utf-8")
        n = tmp / "noscript.html"; n.write_text(noscript, encoding="utf-8")
        o = tmp / "out.html"
        proc = subprocess.run([sys.executable, str(writer), str(model_path), "--layer", LAYER,
                               "--data", str(d), "--render", str(r), "--style", str(st),
                               "--data-type", "application/json", "--produced-by", produced_by,
                               "--noscript", str(n), "--out", str(o)],
                              capture_output=True, text=True, timeout=120)
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
    ap.add_argument("--data", required=True, help="layer-shape-data JSON document")
    ap.add_argument("--out", required=True, help="output HTML path (must differ from MODEL)")
    ap.add_argument("--produced-by", default="/dataset-shaper")
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
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read the inputs: {e}")
        return 2
    for p in (Path(a.render), Path(a.style)):
        if not p.is_file():
            print(f"ERROR: asset missing: {p}")
            return 2
    render_js = Path(a.render).read_text(encoding="utf-8").rstrip()
    style_css = Path(a.style).read_text(encoding="utf-8").rstrip()
    if len(render_js.encode("utf-8")) > 60 * 1024:
        print(f"ERROR: render script is {len(render_js.encode('utf-8'))} bytes; "
              "contract §4 caps it at 60 KB")
        return 2

    doc, serrs = splice(doc, print, Path(a.data).resolve().parent)
    if serrs:
        for e in serrs:
            print(f"ERROR: {e}")
        return 2
    doc.pop("from_run", None)
    errs = validate(doc, print)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2
    n_steps = len(((doc.get("recipe") or {}).get("steps")) or [])
    files = ((doc.get("manifest") or {}).get("outputs") or {}).get("files") or {}
    print(f"OK: {a.data} validates against {SCHEMA} ({n_steps} steps, {len(files)} output file(s), "
          f"structural {doc['verification'].get('structural')}, "
          f"determinism {doc['verification'].get('determinism')})")

    data_text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    noscript = noscript_html(doc)
    df_dir = Path(a.domain_forge_dir) if a.domain_forge_dir else SKILL_DIR.parent / "domain-forge"
    writer = df_dir / "scripts" / "apply_layer.py"
    if not writer.is_file():
        print(f"ERROR: platform writer not found at {writer}; /dataset-shaper writes layers only "
              "through domain-forge's apply_layer.py (pass --domain-forge-dir if the skill lives "
              "elsewhere)")
        return 3
    with tempfile.TemporaryDirectory() as td:
        result, how = platform_apply(writer, html, data_text, render_js, style_css,
                                     a.produced_by, noscript, td, model_path)
    if result is None:
        print(f"ERROR: platform writer {writer} failed ({how})")
        return 3
    if _BLOCK_RE.search(html):
        print("WARN: input already carried a shape layer; it was replaced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"OK: wrote {out_path} via {how}")
    print(f"OK: render script {len(render_js.encode('utf-8'))} bytes, "
          f"data {len(data_text.encode('utf-8'))} bytes, output {len(result.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
