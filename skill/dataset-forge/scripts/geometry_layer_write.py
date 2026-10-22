"""
geometry_layer_write.py — what goes into the `geometry` block and how it gets there: the
`<noscript>` fallback (the prose and the tables, contract §6), the JSON text of the data script,
and the hand-off to domain-forge's platform writer `apply_layer.py`, which alone knows the block
format and the digest (imported first, its CLI when the import fails).

Imported by `apply_geometry_layer.py`; not a command.
"""
from __future__ import annotations

import html as _html
import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from geometry_layer_model import LAYER, is_num, status


# --------------------------------------------------------------------------- noscript
def _esc(s):
    return _html.escape("" if s is None else str(s), quote=True)


def _table(headers, rows):
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _space_and_basis(doc):
    src, sp, B = doc.get("source") or {}, doc.get("space") or {}, doc.get("basis") or {}
    ncols = src.get("columns") if is_num(src.get("columns")) else len(src.get("column_names") or [])
    return [f"<p><em>{_esc(src.get('path'))}</em> — {_esc(src.get('rows'))} rows, "
            f"{_esc(ncols)} columns.</p>",
            "<h3>The space</h3>", f"<p>{_esc(sp.get('reading'))}</p>",
            f"<p>ambient dimension {_esc(sp.get('ambient_dim'))}, exact rank "
            f"{_esc(sp.get('exact_rank'))}, 95% variance in {_esc(sp.get('dims_95'))} directions, "
            f"TwoNN intrinsic dimension {_esc(sp.get('intrinsic_dim_twonn'))}.</p>",
            "<h3>The basis</h3>",
            f"<p>{_esc(', '.join(B.get('members') or []))} ({_esc(B.get('size'))} members).</p>",
            f"<p>{_esc(B.get('reading'))}</p>"]


def _derivations(doc):
    rows = []
    for d in doc.get("derivations") or []:
        sym = (d.get("provenance") or {}).get("symbolic") or {}
        counted = f" ({sym.get('verified_rows')}/{sym.get('of_rows')})" \
            if sym.get("status") != "untested" else ""
        rows.append((d.get("column"), d.get("formula"), d.get("rule_id"), status(d, "semantic"),
                     f"{status(d, 'symbolic')}{counted}", status(d, "empirical")))
    p = ["<h3>Derivations</h3>",
         _table(("column", "formula", "rule", "semantic", "symbolic", "empirical"), rows)
         if rows else "<p>No column is derived.</p>"]
    if doc.get("disagreements"):
        p.append("<h3>Disagreements between the channels</h3>")
        p += [f"<p><strong>{_esc(x.get('column'))}</strong> ({_esc(x.get('kind'))}): "
              f"{_esc(x.get('reading'))}</p>" for x in doc["disagreements"]]
    O = doc.get("orthogonality") or {}
    if O.get("reading"):
        p += ["<h3>Orthogonality</h3>", f"<p>{_esc(O['reading'])}</p>"]
    return p


def _partition_and_stats(doc):
    P = doc.get("partitions") or {}
    p = ["<h3>Partition</h3>",
         f"<p>chosen: {_esc(P.get('chosen') or 'none')} ({_esc(P.get('provenance'))}).</p>",
         _table(("label", "task", "features", "dropped for leakage", "input_dim"),
                [(c.get("label"), c.get("task"), ", ".join(c.get("features") or []),
                  ", ".join(c.get("dropped_for_leakage") or []), c.get("input_dim"))
                 for c in P.get("candidates") or []])]
    st = doc.get("stats") or {}
    if st:
        keys = []
        for v in st.values():
            keys += [k for k in v or {} if k not in keys]
        p += ["<h3>Statistics</h3>",
              _table(["column"] + keys, [[c] + [(v or {}).get(k, "") for k in keys]
                                         for c, v in st.items()])]
    H = doc.get("handoff") or {}
    p.append(f"<h3>Hand-off</h3><p>{_esc(H.get('next'))} {_esc(H.get('note'))}</p>")
    return p


def noscript_html(doc):
    """The prose and the tables, for a reader without JavaScript (contract §6)."""
    p = ['<section class="layer-geometry-noscript">', "<h2>Geometry</h2>"]
    p += _space_and_basis(doc) + _derivations(doc) + _partition_and_stats(doc)
    p.append("<p>Enable JavaScript for the geometry explorer: the space, the derivation graph, "
             "the orthogonality heatmap and the partition cards.</p></section>")
    return "\n".join(p)


# --------------------------------------------------------------------------- the data script
def finite(obj, found=None):
    """The document with every NaN / ±inf replaced by null: geometry.json may carry them (a
    one-value column's std, say) and neither JSON.parse nor the explorer can read them."""
    found = found if found is not None else []
    if isinstance(obj, float) and not math.isfinite(obj):
        found.append(obj)
        return None
    if isinstance(obj, dict):
        return {k: finite(v, found) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v, found) for v in obj]
    return obj


def serialize(doc):
    """Compact JSON a <script> cannot be closed by and that forges no layer marker."""
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("<", "\\u003c").replace("@LAYER:", "\\u0040LAYER:")


# --------------------------------------------------------------------------- the platform writer
@dataclass(frozen=True)
class Block:
    """The four parts of the geometry block, as text."""
    data: str
    render_js: str
    style_css: str
    noscript: str


@dataclass(frozen=True)
class Stamp:
    """The model the block goes into and the provenance the writer stamps on it."""
    model_path: Path
    html: str
    produced_by: str
    produced_at: str | None = None


def _by_import(writer, block, stamp):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("df_apply_layer", str(writer))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply_layer(stamp.html, LAYER, block.data, block.render_js,
                           style_css=block.style_css, data_type="application/json",
                           produced_by=stamp.produced_by, noscript_html=block.noscript,
                           produced_at=stamp.produced_at)


def _by_cli(writer, block, stamp, tmp):
    parts = {"layer.json": block.data, "geometry-render.js": block.render_js,
             "geometry-layer.css": block.style_css, "noscript.html": block.noscript}
    for name, text in parts.items():
        (tmp / name).write_text(text, encoding="utf-8")
    out = tmp / "out.html"
    cmd = [sys.executable, str(writer), str(stamp.model_path), "--layer", LAYER,
           "--data", str(tmp / "layer.json"), "--render", str(tmp / "geometry-render.js"),
           "--style", str(tmp / "geometry-layer.css"), "--data-type", "application/json",
           "--produced-by", stamp.produced_by, "--noscript", str(tmp / "noscript.html"),
           "--out", str(out)]
    if stamp.produced_at:
        cmd += ["--produced-at", stamp.produced_at]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode == 0 and out.is_file():
        return out.read_text(encoding="utf-8"), None
    return None, f"exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:300]}"


def platform_apply(writer, block, stamp, tmpdir):
    """(the layered HTML, how it was written) — or (None, why both paths failed)."""
    notes = []
    try:
        result = _by_import(writer, block, stamp)
        if isinstance(result, str) and f"@LAYER:start {LAYER}" in result:
            return result, "platform writer (import)"
        notes.append("import path returned no layer block")
    except Exception as e:  # noqa: BLE001 — fall through to the CLI path, then report both
        notes.append(f"import path failed: {e}")
    try:
        result, why = _by_cli(writer, block, stamp, Path(tmpdir))
        if result is not None:
            return result, "platform writer (CLI)"
        notes.append(f"CLI path failed ({why})")
    except Exception as e:  # noqa: BLE001
        notes.append(f"CLI path failed: {e}")
    return None, "; ".join(notes)
