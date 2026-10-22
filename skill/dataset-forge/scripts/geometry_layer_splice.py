"""
geometry_layer_splice.py — fill the script's half of a `layer-geometry-data` document from the
`geometry.json` its `from_run` names (`references/report-contract.md` §1: numbers come verbatim).

The author writes the judgement; this module copies in `source` rows / column names / seed /
parameters, the script's typing, every `space` number (plus the contract's short names), the
TwoNN estimate as `basis.intrinsic_dim`, the verbatim sections, the measures of every authored
orthogonality pair and the script's markers. A number the author did write must be the script's
number or a rounding of it; anything else is an error.

Imported by `apply_geometry_layer.py`; not a command.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from geometry_layer_model import is_num
from geometry_layer_shape import check_shape

SPACE_ALIASES = {"near_rank": "near_rank_1e-6", "dims_95": "dims_for_95pct_variance",
                 "dims_99": "dims_for_99pct_variance"}
MEASURES = ("nmi", "pearson", "spearman", "eta2", "cramers_v")
ENGINE_VERBATIM = ("dependencies", "functional_dependencies", "linear_derivability", "stats",
                   "explore")


def _decimals(x):
    if isinstance(x, int):
        return 0
    s = repr(float(x))
    if "e" in s or "E" in s:
        return 12
    return len(s.split(".")[1]) if "." in s else 0


def agrees(author, engine):
    """True when the author's number is the engine's number or a rounding of it."""
    if isinstance(author, bool) or isinstance(engine, bool):
        return author == engine
    if author is None or engine is None:
        return author is None and engine is None
    if is_num(author) and is_num(engine):
        if math.isclose(author, engine, rel_tol=1e-9, abs_tol=1e-12):
            return True
        d = _decimals(author)
        return math.isclose(round(float(engine), d), float(author), rel_tol=1e-9,
                            abs_tol=10.0 ** -(d + 3))
    return author == engine


def _load_engine(ref, base_dir):
    p = Path(ref)
    if not p.is_absolute() and (Path(base_dir) / ref).exists():
        p = Path(base_dir) / ref              # the document's own directory first, then cwd
    if p.is_dir():
        p = p / "geometry.json"               # the run directory names its geometry.json
    if not p.is_file():
        return None, f"from_run points at {ref!r}, which does not exist"
    try:
        eng = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return None, f"from_run {ref!r} is not valid JSON: {e}"
    bad = check_shape(eng)
    if bad:
        return None, (f"from_run {ref!r} is not a geometry.json the script wrote: {bad[0]}"
                      + (f" (and {len(bad) - 1} more)" if len(bad) > 1 else ""))
    return eng, None


def _splice_source(doc, eng, ref):
    src = dict(doc.get("source") or {})
    esrc = eng.get("source") or {}
    names = esrc.get("columns")
    src.setdefault("path", esrc.get("path"))
    src.setdefault("rows", esrc.get("rows"))
    if isinstance(names, list):
        src["column_names"] = [str(c) for c in names]
        if not is_num(src.get("columns")):
            src["columns"] = len(names)
    src.setdefault("geometry_json", str(ref))
    params = eng.get("parameters") or {}
    if params:
        src["parameters"] = params
        src.setdefault("seed", params.get("seed"))
    abox = eng.get("abox_sample") or {}
    if is_num(abox.get("n")):
        src["abox_rows"] = abox["n"]
    doc["source"] = src


def _splice_space(doc, eng, errs):
    espace = dict(eng.get("space") or {})
    for short, long_ in SPACE_ALIASES.items():
        if long_ in espace:
            espace[short] = espace[long_]
    author = doc.get("space") or {}
    for k, v in author.items():
        if k == "reading" or not (is_num(v) or v is None):
            continue
        if k not in espace:
            errs.append(f"space.{k} = {v!r} is not a number geometry.json holds; the layer "
                        "carries the script's numbers, never its own")
        elif not agrees(v, espace[k]):
            errs.append(f"space.{k} = {v!r} but geometry.json says {espace[k]!r} (numbers come "
                        "verbatim from the script)")
    doc["space"] = {**espace, **{k: v for k, v in author.items() if k not in espace}}
    _splice_basis(doc, errs)


def _splice_basis(doc, errs):
    basis = doc.get("basis")
    if not isinstance(basis, dict):
        return
    twonn = doc["space"].get("intrinsic_dim_twonn")
    if "intrinsic_dim" in basis and not agrees(basis["intrinsic_dim"], twonn):
        errs.append(f"basis.intrinsic_dim = {basis['intrinsic_dim']!r} but the script's "
                    f"TwoNN estimate is {twonn!r}")
    basis["intrinsic_dim"] = twonn
    if isinstance(basis.get("members"), list):
        basis.setdefault("size", len(basis["members"]))


def _splice_typing(doc, eng, errs):
    cols = eng.get("columns") or {}
    doc["columns"] = cols
    for i, t in enumerate(doc.get("typing") or []):
        if not isinstance(t, dict) or t.get("column") not in cols:
            continue
        stype = (cols[t["column"]] or {}).get("type")
        if "script_type" in t and t["script_type"] != stype:
            errs.append(f"typing[{i}] ({t['column']}): script_type {t['script_type']!r} but "
                        f"geometry.py typed it {stype!r} — the script's type is recorded, the "
                        "retyping goes in final_type")
        t["script_type"] = stype


def _pair_index(pairs):
    idx = {}
    for p in pairs or []:
        if isinstance(p, dict):
            idx[(p.get("a"), p.get("b"))] = p
            idx[(p.get("b"), p.get("a"))] = p
    return idx


def _splice_orthogonality(doc, eng, errs, out):
    deps = eng.get("dependencies") or {}
    orth = doc.get("orthogonality")
    if not isinstance(orth, dict):
        return
    orth.setdefault("thresholds", deps.get("thresholds"))
    idx = _pair_index(deps.get("pairs"))
    for i, p in enumerate(orth.get("pairs") or []):
        e = idx.get((p.get("a"), p.get("b")))
        if e is None:
            out(f"WARN: orthogonality.pairs[{i}] ({p.get('a')}, {p.get('b')}) is not a pair "
                "geometry.py measured; its numbers cannot be checked")
            continue
        for m in MEASURES:
            if m in p and not agrees(p[m], e.get(m)):
                errs.append(f"orthogonality.pairs[{i}] ({p.get('a')}, {p.get('b')}): {m} = "
                            f"{p[m]!r} but geometry.json says {e.get(m)!r}")
            elif m in e:
                p[m] = e[m]


def _splice_verbatim(doc, eng, out):
    for k in ENGINE_VERBATIM:
        if k in eng:
            if k in doc and doc[k] != eng[k]:
                out(f"WARN: the document's own {k!r} was replaced by geometry.json's")
            doc[k] = eng[k]
    marks = list(doc.get("markers") or [])
    for m in eng.get("markers") or []:
        if m not in marks:
            marks.append(m)
    doc["markers"] = marks


def splice(doc, out, base_dir):
    """Fill the script's half of the layer from geometry.json (contract §1). Returns
    (merged doc, errors). The document has passed the structural pass."""
    ref = doc.get("from_run")
    if not ref:
        out("WARN: no from_run: the layer's numbers could not be checked against geometry.json")
        return doc, []
    eng, err = _load_engine(ref, base_dir)
    if err:
        return doc, [err]
    errs = []
    _splice_source(doc, eng, ref)
    _splice_space(doc, eng, errs)
    _splice_typing(doc, eng, errs)
    _splice_verbatim(doc, eng, out)
    _splice_orthogonality(doc, eng, errs, out)
    doc.pop("from_run", None)
    return doc, errs
