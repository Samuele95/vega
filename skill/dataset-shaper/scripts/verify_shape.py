#!/usr/bin/env python3
"""
verify_shape.py — check what the executor produced, before the `shape` layer is written.

    python3 verify_shape.py --recipe recipe.json --out-dir shaped/ [--model MODEL.html]
                            [--psi-threshold 0.25] [--json report.json]
    python3 verify_shape.py --transform-only --recipe recipe.json --out-dir shaped/
                            --module preprocess.py [--part test]

Six families of check, from `references/shape-contract.md` §3:

  structural      the label is not among the feature columns; the leakage set is gone; every
                  active derivation head is gone unless a user: step kept it; no basis member
                  is missing unless a user: step dropped it.
  semantic        for every derivation whose body survives, the relation is recomputed on the
                  output — algebraically when the formula parses, and as a functional
                  dependency otherwise. A transform applied to one member of a derivation and
                  not the others breaks a definitional truth silently; this is the check that
                  catches it. With a headless browser and a model that carries rules, the same
                  claim is put to the model's own SWRL/Horn engine and recorded as
                  confirmed / refuted / untested — never asserted from arithmetic here.
  distributional  PSI between input and output for every column no step named. A column
                  nobody transformed should not have moved; when it has, the report names the
                  step it suspects.
  split           no row in two parts; a group split keeps a group whole and its key is no
                  feature (gone from every part, or the graph key with role key); a time split
                  keeps the order; the label rate per part is within tolerance of the whole; every
                  part keeps the row order `__split_raw__` records, read on the columns the
                  recipe carried through unchanged (the twin, the semantic check and the
                  loaders' row ids all align a part with its raw rows by that order). And the
                  leakage a phase order cannot see: no impute or lag grouped by the label, no
                  importance selection after a split, every fitted step's fit_on the train
                  part, every pre-split custom step probed row-local, a neighbour index no
                  larger than the train part.
  spatial         when spatial steps ran: coordinates inside the declared CRS bounds, a CRS
                  recorded on the output, no invalid geometry.
  graph           when a graph_edges step ran: the edges part is the file the manifest
                  pinned; every end resolves through __split_raw__ to the part it names; the
                  reverses of a symmetric part, the self-loop and degree rules, one graph per
                  edge at level graph, no two non-train ends for a fitted index and its
                  matrix_digest recomputed, and the manifest's counts against a recount —
                  then the relation itself, recomputed without the executor's code: key
                  equality (and, uncapped, the exact edge set) for key_pair, the k nearest /
                  the radius in the fitted space for knn and spatial, with the distance
                  weights.

Determinism is NOT checked here — it is `shape.py --check`, which re-runs the generated
reproduction script and compares digests.

`--transform-only` is a seventh, separate check: the **twin check**. It fetches the raw rows a
shaped part came from, runs them through the emitted `preprocess.py`, and asserts the result
equals that part on the feature columns. It is the only test of the transformer that does not
go through a network — everything downstream of it compares scores, where a wrongly shaped
row does not announce itself — so when a column diverges it names the step that last touched
it, from `lineage.json`, and the first row on which it did. On a graph recipe it then runs
`edges_for` (level node) against a reference built as /vega-pack ships it — the parts before
the checked one, dense ids, every feature as shaped in float32, each input column edges_for
reads as `raw:<column>`, no raw position — or `edges_within` (level graph), and asserts the
edges served for those rows equal the part's edges between them and the reference, in either
direction (or inside their graphs), byte for byte.

Exit codes: 0 every check passed (warnings allowed); 3 a check failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SKILL_DIR = HERE.parent
FAIL, WARN, OK = [], [], []


def bad(m):
    FAIL.append(m)
    print(f"ERROR: {m}")


def warn(m):
    WARN.append(m)
    print(f"WARN: {m}")


def ok(m):
    OK.append(m)
    print(f"OK: {m}")


def read_parquet(p):
    """A parquet file through Arrow's own filesystem (PLAN-AMENDMENTS A40): pandas' default
    hands Arrow a Python file object, and a reader thread still inside Python while the
    interpreter finalizes aborts the process at exit (SIGABRT) after every check printed."""
    import pyarrow.fs
    return pd.read_parquet(p, filesystem=pyarrow.fs.LocalFileSystem())


def read_any(p):
    p = Path(p)
    if p.suffix == ".parquet":
        return read_parquet(p)
    return pd.read_csv(p, low_memory=False)


def read_shaped(p):
    """An output the executor wrote, read back bit for bit. `to_csv` writes each float as its
    shortest round-trip text, and pandas' default csv parser is not correctly rounded (it can
    land one ulp away), so a csv output is parsed with `float_precision="round_trip"`: the
    graph checks compare a fitted matrix by digest, which one ulp breaks."""
    p = Path(p)
    if p.suffix == ".parquet":
        return read_parquet(p)
    return pd.read_csv(p, low_memory=False, float_precision="round_trip")


def read_layers(model_path):
    out = {}
    if not model_path:
        return out
    html = Path(model_path).read_text(encoding="utf-8")
    for name in ("geometry", "analysis"):
        m = re.search(r'<script\s+id="layer-' + name + r'-data"[^>]*>([\s\S]*?)</script>',
                      html, re.IGNORECASE)
        if m:
            try:
                out[name] = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
    return out


def psi(expected, actual, bins=10):
    e = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().values
    a = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().values
    if e.size < 20 or a.size < 20:
        return None
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if qs.size < 3:
        return None
    qs[0], qs[-1] = -np.inf, np.inf
    ec = np.clip(np.histogram(e, bins=qs)[0] / e.size, 1e-6, None)
    ac = np.clip(np.histogram(a, bins=qs)[0] / a.size, 1e-6, None)
    return float(np.sum((ac - ec) * np.log(ac / ec)))


def psi_cat(expected, actual):
    e = pd.Series(expected).astype(str).value_counts(normalize=True)
    a = pd.Series(actual).astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    ev = np.clip(e.reindex(idx).fillna(0).values, 1e-6, None)
    av = np.clip(a.reindex(idx).fillna(0).values, 1e-6, None)
    return float(np.sum((av - ev) * np.log(av / ev)))


_FORMULA_OPS = {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "⁄": "/"}


def parse_formula(formula):
    """'total = unit_price × qty' -> ('total', 'unit_price * qty'); None when it is prose."""
    if not formula or "=" not in formula:
        return None
    head, expr = formula.split("=", 1)
    head, expr = head.strip(), expr.strip()
    for a, b in _FORMULA_OPS.items():
        expr = expr.replace(a, b)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", head):
        return None
    if not re.fullmatch(r"[A-Za-z_0-9_+\-*/(). ]+", expr):
        return None          # a threshold rule, a lookup, prose: not arithmetic
    return head, expr


def determination(df, lhs, rhs):
    """Share of rows whose lhs-group has a single rhs value: an FD holds at 1.0."""
    sub = df[list(lhs) + [rhs]].dropna()
    if not len(sub):
        return None
    g = sub.groupby([sub[c].astype(str) for c in lhs])[rhs].nunique()
    sizes = sub.groupby([sub[c].astype(str) for c in lhs])[rhs].size()
    good = sizes[g <= 1].sum()
    return float(good / len(sub))


# --------------------------------------------------------------------------- the twin check
def load_module(path):
    """Import the emitted module by path, without putting its directory on sys.path.

    The bundle's copy sits inside a package it must not import from, and the skill's scripts
    directory must not be importable either: the module is standalone or it is wrong."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("shaper_preprocess", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_column(want, got, tol):
    """('exact' | 'tolerance' | 'differ', detail) for one column.

    Exact is the bar. A float column is allowed to miss it only by an amount no arithmetic
    difference explains — the same expression evaluated over a different number of rows, where
    a reduction (a matrix product in `project`, a vectorised ufunc over a shorter array) may
    accumulate in a different order. Anything larger is a different computation, not a
    different summation order, and this returns 'differ'."""
    a, b = want.to_numpy(), got.to_numpy()
    if len(a) != len(b):
        return "differ", f"{len(a)} rows in the part, {len(b)} back from transform"
    if pd.api.types.is_float_dtype(want) and pd.api.types.is_float_dtype(got):
        a, b = a.astype(float), b.astype(float)
        if np.array_equal(a, b, equal_nan=True):
            return "exact", ""
        both = np.isfinite(a) & np.isfinite(b)
        if (np.isnan(a) != np.isnan(b)).any():
            i = int(np.flatnonzero(np.isnan(a) != np.isnan(b))[0])
            return "differ", f"row {i}: {a[i]!r} in the part, {b[i]!r} from transform (one is NaN)"
        diff = np.abs(a[both] - b[both])
        scale = np.maximum(1.0, np.abs(a[both]))
        if diff.size and (diff <= tol * scale).all():
            return "tolerance", f"max |Δ| {float(diff.max()):.3e} (≤ {tol:g} × max(1, |value|))"
        i = int(np.flatnonzero(diff > tol * scale)[0])
        idx = int(np.flatnonzero(both)[i])
        return "differ", (f"row {idx}: {a[idx]!r} in the part, {b[idx]!r} from transform "
                          f"(|Δ| {float(diff[i]):.6g})")
    # Not floating point: there is no tolerance to grant. A category id, a bin number, an
    # indicator or a string either matches or the recipe was applied differently.
    same = pd.Series(a).astype(object).fillna("\x00__na__").to_numpy() == \
        pd.Series(b).astype(object).fillna("\x00__na__").to_numpy()
    if same.all():
        return "exact", ""
    i = int(np.flatnonzero(~same)[0])
    return "differ", f"row {i}: {a[i]!r} in the part, {b[i]!r} from transform"


def twin_check(a):
    """`transform(raw rows of a part)` must equal the shaped part on the feature columns."""
    out_dir = Path(a.out_dir)
    recipe = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
    man = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    fitted = json.loads((out_dir / man["outputs"].get("fitted", "fitted.json"))
                        .read_text(encoding="utf-8"))
    lineage = {}
    lin_p = out_dir / man["outputs"].get("lineage", "lineage.json")
    if lin_p.is_file():
        lineage = (json.loads(lin_p.read_text(encoding="utf-8")) or {}).get("columns") or {}
    part = a.part
    files = man["outputs"]["files"]
    if part not in files or part == "edges":
        bad(f"twin: the manifest has no {part!r} part (it has {sorted(files)})")
        return 3
    want = read_any(out_dir / Path(files[part]["path"]).name)
    raw = read_any(recipe["input"]["path"])

    # The raw rows this part came from. `__split_raw__` addresses the raw file; `__split__`
    # addresses the frame the split saw, which is the same thing unless a dedupe ran first.
    sets = fitted.get("__split_raw__") or fitted.get("__split__") or {}
    if part in sets:
        pos = [int(i) for i in sets[part]]
    elif len([k for k in files if k != "edges"]) == 1:
        removed = set(int(i) for i in fitted.get("__removed_rows__") or [])
        pos = [i for i in range(len(raw)) if i not in removed]
    else:
        bad(f"twin: {part!r} has no recorded raw row positions; the shaper that wrote "
            f"{out_dir/'fitted.json'} is older than transform-only mode")
        return 3
    rows = raw.iloc[pos]

    mod = load_module(a.module)
    lags = getattr(mod, "NEEDS_HISTORY", False)
    history = None
    if lags:
        # The window is deliberately EMPTY. The shaped part was produced from these rows and
        # nothing else — `lag` is a phase-4 step, so the executor took it inside each part,
        # after the split — and a twin check handed a wider window would be comparing two
        # different questions. What the real preceding window does instead is measured below,
        # where it is a reading about the recipe rather than a verdict on the module.
        history = raw.iloc[:0]
        ok("twin: the recipe lags; the twin runs with an empty history window, which is the "
           "window the executor had when it wrote this part")
    try:
        got = mod.transform(rows, history=history)
    except Exception as e:
        bad(f"twin: transform() raised {type(e).__name__}: {e}")
        return 3
    got = got.reset_index(drop=True)
    want = want.reset_index(drop=True)

    label = man.get("label")
    feats = [c for c in want.columns if c != label]
    extra = [c for c in got.columns if c not in feats]
    absent = [c for c in feats if c not in got.columns]
    if absent:
        for c in absent:
            bad(f"twin: {c!r} is in the shaped {part} part and not in transform's output "
                f"(last touched by {(lineage.get(c) or {}).get('touched_by') or 'the input'})")
    if extra:
        bad(f"twin: transform returned column(s) the shaped part does not have: {extra}")
    if label and label in got.columns:
        bad(f"twin: transform returned the label {label!r}; at inference it is an output, and "
            "a row carrying it would make the network's input one column too wide")
    if not absent and not extra:
        order = [c for c in got.columns]
        if order != feats:
            bad(f"twin: the feature columns come back in a different order — the network reads "
                f"them positionally, so this is a wrong answer, not a cosmetic one: expected "
                f"{feats}, got {order}")
        else:
            ok(f"twin: {len(feats)} feature column(s), same names in the same order")

    exact, toler = [], []
    for c in feats:
        if c not in got.columns:
            continue
        verdict, detail = compare_column(want[c], got[c], a.tolerance)
        if verdict == "exact":
            exact.append(c)
        elif verdict == "tolerance":
            toler.append(c)
            warn(f"twin: {c} agrees within tolerance but not bit for bit — {detail}; the "
                 f"step(s) behind it: {(lineage.get(c) or {}).get('touched_by') or 'the input'}")
        else:
            bad(f"twin: {c} diverges — {detail}; the step(s) behind it: "
                f"{(lineage.get(c) or {}).get('touched_by') or 'the input'}")
    if exact:
        ok(f"twin: {len(exact)} column(s) match byte for byte on the {part} part "
           f"({len(want)} rows)")
    if toler:
        warn(f"twin: {len(toler)} float column(s) matched only within {a.tolerance:g} × "
             "max(1, |value|) — a reduction over a different number of rows sums in a "
             "different order, and no exact comparison is available for it")
    # Second reading, on a lag recipe only: what the *real* preceding window changes. It is
    # not a defect in the module — it is what the executor's per-part lag costs. The rows a
    # split boundary left without a predecessor were trained on as missing; served, they have
    # one. Whoever reads this decides whether that gap matters; the check only refuses to
    # leave it unsaid.
    if lags and min(pos) > 0 and not FAIL:
        served = mod.transform(rows, history=raw.iloc[:min(pos)]).reset_index(drop=True)
        moved = {c: int((~pd.Series(served[c]).eq(pd.Series(got[c])) &
                         ~(pd.isna(served[c]) & pd.isna(got[c]))).sum())
                 for c in served.columns if c in got.columns}
        moved = {c: n for c, n in moved.items() if n}
        if moved:
            warn(f"twin: given the real preceding window (raw rows 0..{min(pos) - 1}), "
                 f"{', '.join(f'{c} moves on {n} row(s)' for c, n in sorted(moved.items()))} — "
                 "the executor takes lags inside each part, so the first rows of this part "
                 "were shaped without a predecessor and trained on as missing, while a served "
                 "row has one. Serving is not wrong here; the training frame was")
        else:
            ok("twin: the real preceding window changes nothing — no row of this part sat "
               "against a split boundary")

    unseen = getattr(mod, "last_unseen", lambda: [])()
    flagged = sum(1 for u in unseen if u)
    if flagged:
        warn(f"twin: transform flagged {flagged} of {len(unseen)} row(s) as carrying a "
             "category the fit never saw; predict.py carries these into its warnings column")
    twin_graph(mod, {"raw": raw, "out_dir": out_dir, "man": man, "fitted": fitted,
                     "part": part, "rows": rows, "history": history, "tol": a.tolerance})
    print(f"\n{len(OK)} passed, {len(WARN)} warned, {len(FAIL)} failed")
    return 3 if FAIL else 0


# --------------------------------------------------------------------------- row order
# Ops that read a column and leave its values as they were: a column only they touched is
# carried through, and must equal the input at the part's raw positions, row for row.
READ_ONLY_OPS = {"lag"}
# The input tokens the executor reads as missing (shape-contract §2), whatever their case.
NON_FINITE_TOKENS = {"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan"}


def _carried(values):
    """A column as comparable values: floats with the non-finite read as missing, or text
    with the non-finite tokens and every missing value as one sentinel."""
    num = pd.to_numeric(values, errors="coerce")
    if values.notna().sum() == num.notna().sum():
        x = num.to_numpy(dtype=float, copy=True)
        x[~np.isfinite(x)] = np.nan
        return x
    text = values.astype(object).where(values.notna(), None)
    return np.array(["\x00__na__" if v is None or str(v).strip().lower() in NON_FINITE_TOKENS
                     else str(v) for v in text], dtype=object)


def _first_difference(want, got):
    """The first row two carried columns disagree on, or None (floats within 1e-12 relative,
    the slack a csv part's parse can need)."""
    if want.dtype == float and got.dtype == float:
        same = np.isclose(want, got, rtol=1e-12, atol=0.0, equal_nan=True)
    else:
        same = want.astype(str) == got.astype(str)
    return None if same.all() else int(np.flatnonzero(~same)[0])


def check_row_order(recipe, out_dir, src, parts):
    """Every part keeps the row order `fitted.json.__split_raw__` records (shape-contract §2: a
    raw position resolves to a part's row through it, which the shaped parts preserve in
    order). Read on the columns the recipe carried through unchanged — untouched, or touched
    only by a step that reads them — against the input at those positions. Returns 'pass',
    'fail', or None when there is nothing to read it on (no split, no carried column, or a
    custom step, whose columns lineage.json does not record)."""
    fit_p, lin_p = out_dir / "fitted.json", out_dir / "lineage.json"
    if any(s.get("op") == "custom" for s in recipe["steps"]) or not fit_p.is_file() \
            or not lin_p.is_file():
        return None
    sets = (json.loads(fit_p.read_text(encoding="utf-8")) or {}).get("__split_raw__") or {}
    lineage = (json.loads(lin_p.read_text(encoding="utf-8")) or {}).get("columns") or {}
    ops = {s.get("id"): s.get("op") for s in recipe["steps"]}
    carried = sorted(c for c, v in lineage.items() if c in src.columns
                     and all(ops.get(t) in READ_ONLY_OPS for t in v.get("touched_by") or []))
    checked, wrong = [], []
    for name, d in parts.items():
        cols = [c for c in carried if c in d.columns]
        if name not in sets or not cols:
            continue
        pos = [int(i) for i in sets[name]]
        if len(pos) != len(d):
            wrong.append(f"{name}: {len(d)} rows, {len(pos)} raw positions")
            continue
        for c in cols:
            want = _carried(src[c].iloc[pos].reset_index(drop=True))
            got = _carried(d[c].reset_index(drop=True))
            i = _first_difference(want, got)
            if i is not None:
                wrong.append(f"{name}.{c} row {i}: {got[i]!r}, the input's raw row {pos[i]} "
                             f"holds {want[i]!r}")
        checked.append(name)
    if wrong:
        bad("split: a part's rows are not in the order __split_raw__ records — "
            + "; ".join(wrong[:4]) + " — every reader that aligns a part with its raw rows "
            "by position reads the wrong rows")
        return "fail"
    if checked:
        ok(f"split: {', '.join(checked)} keep the row order __split_raw__ records (read on "
           f"{', '.join(carried)}, carried through unchanged)")
        return "pass"
    return None


# --------------------------------------------------------------------------- leakage
_SHAPER = []


def _shaper():
    """shape.py's rules, read from the executor itself: the refusals below are the rules the
    recipe checker holds, applied here to the recipe and manifest that were published."""
    if not _SHAPER:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dataset_shaper_shape",
                                                      str(HERE / "shape.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SHAPER.append(mod)
    return _SHAPER[0]


def _fitted_here(st, recipe_step):
    """True when a manifest step fitted something on a part: an always-fitted op, a box-cox /
    yeo-johnson transform, a variance selection, a spatial_features neighbour index."""
    p = (recipe_step or {}).get("params") or {}
    op = st.get("op")
    if op == "transform":
        return p.get("kind") in ("box-cox", "yeo-johnson")
    if op == "select_features":
        return p.get("kind") == "variance"
    if op == "spatial_features":
        return bool((st.get("parameters_fitted") or {}).get("index"))
    return op in _shaper().FITTED_OPS


def _index_error(st, want_part):
    """What is wrong with a spatial_features neighbour index, or None: it is fitted on the
    fit part, its counts add up to its rows, and it holds no more rows than that part."""
    index = (st.get("parameters_fitted") or {}).get("index")
    if not index:
        return None
    n_part = (st.get("rows_after") or {}).get(want_part)
    n_fit = index.get("n_fit")
    if index.get("fit_on") == want_part and isinstance(n_fit, int) and n_part is not None \
            and int(sum(index.get("counts") or [])) == n_fit and n_fit <= n_part:
        return None
    return (f"the neighbour index holds {n_fit} row(s), records fit_on "
            f"{index.get('fit_on')!r}, and the {want_part} part has {n_part} row(s) — it is "
            "fitted on that part alone")


def check_leakage(recipe, man):
    """The leakage rules a phase order cannot express, read on what was published (shape-contract
    §3, split hygiene): no `impute` or `lag` grouped by the label; no `select_features
    kind=importance` in a recipe that splits; every fitted step records the part it was fitted
    on, the train part when a split ran; every `custom` step before the split was probed
    row-local by the executor; a `spatial_features` neighbour index holds no more rows than the
    train part. Returns 'pass' or 'fail'."""
    sh = _shaper()
    steps = [s for s in recipe.get("steps") or [] if isinstance(s, dict)]
    by_id = {s.get("id"): s for s in steps}
    split_at = next((i for i, s in enumerate(steps) if s.get("op") == "split"), None)
    want_part = "train" if split_at is not None else "all"
    label = man.get("label") or sh.planned_label(steps)
    before = len(FAIL)
    for s in steps:
        why = sh.label_grouping_error(s, label)
        if why:
            bad(f"leakage: {s.get('id')}: {why}")
        if split_at is not None and s.get("op") == "select_features" and \
                (s.get("params") or {}).get("kind", "importance") == "importance":
            bad(f"leakage: {s.get('id')}: {sh.IMPORTANCE_AFTER_SPLIT}")
    fitted_n, at_split = 0, False
    for st in man.get("steps") or []:
        at_split = at_split or st.get("op") == "split"
        probe = (st.get("parameters_fitted") or {}).get("row_local_probe") or {}
        if st.get("op") == "custom" and split_at is not None and not at_split \
                and not probe.get("passed"):
            bad(f"leakage: {st.get('id')} is a custom step before the split that the executor "
                "did not probe row-local — it may compute over rows the test part shares")
        # A step that must fit (its op, its kind) or says it did (a spatial projection's
        # centroid, a graph index) is held to the part it was fitted on.
        if st.get("fit_on") is None and not _fitted_here(st, by_id.get(st.get("id"))):
            continue
        fitted_n += 1
        if st.get("fit_on") != want_part:
            bad(f"leakage: {st.get('id')} ({st.get('op')}) is fitted and records fit_on "
                f"{st.get('fit_on')!r}, not {want_part!r} — "
                + ("after a split every fitted step is fitted on the train part"
                   if split_at is not None else "without a split it is fitted on all rows"))
        why = _index_error(st, want_part)
        if why:
            bad(f"leakage: {st.get('id')}: {why}")
    if len(FAIL) > before:
        return "fail"
    ok(f"leakage: no impute or lag grouped by the label, no importance selection after a split, "
       f"{fitted_n} fitted step(s) recording fit_on {want_part!r}")
    return "pass"


def check_split_key(recipe, man, parts):
    """A group split's key never reaches the shaped parts as a feature (shape-contract §1 and
    §3, F33): no step after the split reads it, and it is in no part — or, at level graph, it is
    the graph key and `output_schema` gives it `role: key`. Returns 'pass', 'fail' or 'n/a'."""
    sh = _shaper()
    steps = [s for s in recipe.get("steps") or [] if isinstance(s, dict)]
    key, kept = sh.split_key(steps)
    if key is None:
        return "n/a"
    before = len(FAIL)
    split_at = next((i for i, s in enumerate(steps) if s.get("op") == "split"), None)
    for e in sh.split_key_errors(steps, split_at):
        bad(f"split: {e}")
    role = ((man.get("output_schema") or {}).get(key) or {}).get("role")
    holding = sorted(k for k, d in parts.items() if key in d.columns)
    if kept:
        if role != "key":
            bad(f"split: the group key {key!r} is the graph key, but output_schema gives it role "
                f"{role!r}, not 'key' — it would reach the network as a feature")
    elif holding or role is not None:
        bad(f"split: the group split's key {key!r} reaches the shaped parts "
            f"({', '.join(holding) or 'output_schema'}, role {role!r}) — {sh.SPLIT_KEY_WHY}")
    if len(FAIL) > before:
        return "fail"
    ok(f"split: the group key {key!r} is no feature ("
       + ("the graph key, role key" if kept else "gone from every part") + ")")
    return "pass"


# --------------------------------------------------------------------------- the graph
EDGE_COLUMNS =["src_pos", "dst_pos", "src_part", "dst_part", "weight"]
EARTH_R = 6371008.8


def key_text(values):
    """A key as text, whole floats as integers, missing as None — recomputed here rather than
    imported, so the semantic check is a second reading of the relation, not the same code."""
    s = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_float_dtype(s) and bool((s.dropna() % 1 == 0).all()):
        s = s.astype("Int64")
    out = s.astype("string")
    return np.array([None if pd.isna(v) else str(v) for v in out], dtype=object)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def projected(frame, coords):
    """spatial: raw latitude / longitude in the metric CRS spatial_reproject recorded."""
    la = pd.to_numeric(frame[coords["lat"]], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(frame[coords["lon"]], errors="coerce").to_numpy(dtype=float)
    pr = coords["projection"]
    if pr.get("kind") == "pyproj":
        from pyproj import Transformer
        tr = Transformer.from_crs(pr.get("from") or "EPSG:4326", pr["to"], always_xy=True)
        x, y = tr.transform(lo, la)
        return np.column_stack([x, y])
    x = np.radians(lo - pr["lon0"]) * EARTH_R * np.cos(np.radians(pr["lat0"]))
    y = np.radians(la - pr["lat0"]) * EARTH_R
    return np.column_stack([x, y])


def graph_report(recipe, man, out_dir, src, parts):
    """`verification.graph`: n/a without a graph step, else pass or fail by check_graph."""
    if not (any(s["op"] == "graph_edges" for s in recipe["steps"]) or man.get("graph")):
        return "n/a"
    before = len(FAIL)
    check_graph(recipe, man, out_dir, src, parts)
    return "fail" if len(FAIL) > before else "pass"


def check_graph(recipe, man, out_dir, src, parts):
    """`shape-contract.md` §3, graph: the edges part against the manifest, the split and the
    relation it claims.

    Structural — the file is the one the manifest pinned; every end resolves to a row of the
    part it names; a symmetric part has every reverse; no self loop unless declared; the
    out-degree cap on a directed part; no edge across two graphs; for a fitted index (knn /
    spatial at level node) no edge between two non-train nodes, and the fitted entry's
    matrix_digest is the train matrix's; the manifest's counts equal a recount. Semantic — the
    relation is recomputed from scratch, independently of the executor's code: key equality
    from the input for key_pair (and, uncapped, the exact edge set), the k nearest (or the
    radius) in the fitted space for knn / spatial, with the distance weights."""
    c = _graph_load(recipe, man, out_dir)
    if c is None:
        return
    c.src, c.parts = src, parts
    before = len(FAIL)
    if not _graph_ends(c):
        return
    _graph_shape(c)
    _graph_scope(c)
    _graph_counts(c)
    keep = c.src_pos != c.dst_pos                    # a self loop relates a node to itself
    c.s_n = np.array([c.node[int(i)] for i in c.src_pos[keep]], dtype="int64")
    c.d_n = np.array([c.node[int(i)] for i in c.dst_pos[keep]], dtype="int64")
    c.w = c.e["weight"].to_numpy(dtype=float)[keep]
    if c.g.get("method") == "key_pair":
        _graph_keys(c)
    else:
        _graph_neighbours(c)
    if len(FAIL) == before:
        ok(f"graph: {c.g['method']} level {c.g['level']} — {len(c.e)} edge rows, "
           f"{'directed' if not c.g.get('symmetric', True) else 'symmetric'}, checks pass")


def _graph_file(man, out_dir):
    """The edges part the manifest pinned, read back, or None (reported)."""
    meta = man["outputs"]["files"]["edges"]
    path = out_dir / Path(meta["path"]).name
    if not path.is_file():
        if meta.get("digest") is None:
            warn("graph: no edges file (a --dry-run manifest); the graph checks are skipped")
        else:
            bad(f"graph: the edges part {path.name} named by the manifest is missing")
        return None
    if meta.get("digest") and sha256_file(path) != meta["digest"]:
        bad(f"graph: {path.name} does not digest to the manifest's {meta['digest']} — the edges "
            "part was changed after the executor wrote it")
    e = read_shaped(path)
    if list(e.columns) != EDGE_COLUMNS:
        bad(f"graph: the edges part's columns are {list(e.columns)}, not {EDGE_COLUMNS}")
        return None
    return e


def _graph_load(recipe, man, out_dir):
    """The edges part, the manifest's block and the step's resolved spec, or None (reported)."""
    from types import SimpleNamespace
    step = next((s for s in recipe["steps"] if s["op"] == "graph_edges"), None)
    if step is None:
        bad("graph: the manifest carries a graph block but the recipe has no graph_edges step")
        return None
    if not man.get("graph") or "edges" not in (man["outputs"].get("files") or {}):
        bad("graph: graph_edges ran but the manifest has no graph block or no "
            "outputs.files.edges — the edges part is what the graph families read")
        return None
    e = _graph_file(man, out_dir)
    fitted = json.loads((out_dir / man["outputs"].get("fitted", "fitted.json"))
                        .read_text(encoding="utf-8"))
    g = (fitted.get(step["id"]) or {}).get("spec") or {}
    if e is None or not g:
        if e is not None:
            bad(f"graph: fitted.json carries no resolved spec for {step['id']}")
        return None
    sets = fitted.get("__split_raw__") or {}
    names = [k for k in ("train", "valid", "test") if k in sets]
    ids = np.concatenate([np.asarray(sets[k], dtype="int64") for k in names])
    return SimpleNamespace(
        e=e, g=g, block=man["graph"], meta=man["outputs"]["files"]["edges"], sets=sets,
        names=names, ids=ids, fit=fitted.get(step["id"]) or {},
        part_of=np.concatenate([np.array([k] * len(sets[k]), dtype=object) for k in names]),
        node={int(r): i for i, r in enumerate(ids)},
        src_pos=e["src_pos"].to_numpy(dtype="int64"), dst_pos=e["dst_pos"].to_numpy(dtype="int64"),
        sp=e["src_part"].astype(str).to_numpy(), dp=e["dst_part"].astype(str).to_numpy())


def _graph_ends(c):
    """Every end is a raw position of the part it names (through `__split_raw__`)."""
    ends = np.r_[c.src_pos, c.dst_pos]
    stray = sorted({int(i) for i in ends if int(i) not in c.node})
    if stray:
        bad(f"graph: {len(stray)} edge end(s) name a raw position no part holds "
            f"(the first: {stray[0]})")
        return False
    named = np.r_[c.sp, c.dp]
    held = c.part_of[[c.node[int(i)] for i in ends]]
    wrong = np.flatnonzero(held != named)
    if len(wrong):
        bad(f"graph: {len(wrong)} edge end(s) do not resolve through __split_raw__ to the part "
            f"they name (raw position {int(ends[wrong[0]])} is not in {named[wrong[0]]!r})")
        return False
    ok(f"graph: every edge end resolves through __split_raw__ to the part it names "
       f"({len(c.e)} edge rows)")
    return True


def _graph_shape(c):
    """Duplicates, self loops, reverses, the out-degree cap."""
    g, src_pos, dst_pos = c.g, c.src_pos, c.dst_pos
    dup = int(c.e.duplicated(["src_pos", "dst_pos"]).sum())
    if dup:
        bad(f"graph: {dup} directed edge(s) appear twice")
    loops = int((src_pos == dst_pos).sum())
    if loops and not g.get("self_loops"):
        bad(f"graph: {loops} self loop(s) although self_loops is false")
    pairs = set(zip(src_pos.tolist(), dst_pos.tolist()))
    caps = [x for x in (g.get("max_degree"),
                        g.get("k") if g.get("method") != "key_pair" else None) if x]
    if g.get("symmetric", True):
        missing = [(s, d) for s, d in sorted(pairs) if (d, s) not in pairs]
        if missing:
            bad(f"graph: symmetric is declared but {len(missing)} edge(s) have no reverse row "
                f"(the first: {missing[0][0]} -> {missing[0][1]}; its reverse is missing)")
        else:
            ok("graph: symmetric — every edge has its reverse row")
    elif caps:
        deg = pd.Series(src_pos[src_pos != dst_pos]).value_counts()
        over = deg[deg > min(caps)]
        if len(over):
            bad(f"graph: {len(over)} node(s) exceed the out-degree cap {min(caps)} "
                f"(max_degree / k; raw position {int(over.index[0])} has {int(over.iloc[0])})")
        else:
            ok(f"graph: every node's out-degree is within {min(caps)}")


def _graph_scope(c):
    """One graph per edge at level graph; a train end on every edge of a fitted index."""
    g, src_pos, dst_pos = c.g, c.src_pos, c.dst_pos
    c.gk = key_text(c.src[g["graph_id"]].to_numpy()[c.ids]) if g.get("graph_id") else None
    if c.gk is not None:
        cross = int((c.gk[[c.node[int(i)] for i in src_pos]] !=
                     c.gk[[c.node[int(i)] for i in dst_pos]]).sum())
        if cross:
            bad(f"graph: {cross} edge(s) join rows of two different {g['graph_id']!r} values — "
                "at level graph every edge stays inside one graph_id")
        else:
            ok(f"graph: every edge stays inside one {g['graph_id']!r} graph")
        return
    if g.get("method") == "key_pair":
        return
    between = src_pos != dst_pos
    off = int(((c.sp != "train") & (c.dp != "train") & between).sum())
    if off:
        bad(f"graph: {off} edge(s) join two non-train nodes — the {g['method']} index was "
            "fitted on train, so a valid or test node reaches only into train "
            "(cross_part_edges would count what the protocol must never see)")
    elif not g.get("symmetric", True) and int((c.dp[between] != "train").sum()):
        bad("graph: a directed fitted edge points to a non-train node")
    else:
        ok("graph: every edge has a train end — the index was fitted on train alone")


def _graph_counts(c):
    """The manifest's counts against a recount of the part."""
    sp, dp, block = c.sp, c.dp, c.block
    recount = {f"{a}-{b}": int((((sp == a) & (dp == b)) | ((sp == b) & (dp == a))).sum())
               for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    within = {k: int(((sp == k) & (dp == k)).sum()) for k in c.names}
    n_nodes = {k: int(len(c.sets[k])) for k in c.names}
    diffs = []
    if recount != block.get("cross_part_edges"):
        diffs.append(f"cross_part_edges {block.get('cross_part_edges')} != recount {recount}")
    if block.get("within_part_edges") is not None and within != block.get("within_part_edges"):
        diffs.append(f"within_part_edges {block.get('within_part_edges')} != recount {within}")
    if block.get("n_edges") != len(c.e) or c.meta.get("rows") != len(c.e):
        diffs.append(f"n_edges {block.get('n_edges')} (file rows {c.meta.get('rows')}) != "
                     f"{len(c.e)}")
    if block.get("n_nodes") != n_nodes:
        diffs.append(f"n_nodes {block.get('n_nodes')} != the parts' rows {n_nodes}")
    if block.get("directed") != (not c.g.get("symmetric", True)):
        diffs.append("directed does not equal not symmetric")
    if diffs:
        bad("graph: the manifest's graph block disagrees with the edges part — " + "; ".join(diffs))
    else:
        ok(f"graph: cross_part_edges recount equals the manifest's ({recount})")


def _graph_keys(c):
    """key_pair: the key equality on every edge, and — uncapped — the exact relation."""
    g, gk, ids = c.g, c.gk, c.ids
    cols = g["columns"]
    a_col, b_col = (cols[1], cols[0]) if len(cols) == 2 else (cols[0], cols[0])
    ka = key_text(c.src[a_col].to_numpy()[ids])
    kb = key_text(c.src[b_col].to_numpy()[ids])

    def related(i, j):
        return (ka[i] is not None and ka[i] == kb[j]) and (gk is None or gk[i] == gk[j])
    wrong = [(int(ids[i]), int(ids[j])) for i, j in zip(c.s_n, c.d_n)
             if not (related(i, j) or (g.get("symmetric", True) and related(j, i)))]
    if wrong:
        bad(f"graph: {len(wrong)} key_pair edge(s) join rows whose keys differ "
            f"(the first: {wrong[0][0]} -> {wrong[0][1]}; {a_col} of the source must equal "
            f"{b_col} of the destination) — the key equality this edge claims does not hold")
    else:
        ok(f"graph: key equality holds on every edge ({a_col} -> {b_col}), recomputed from "
           "the input at the raw positions")
    if g.get("max_degree") or g.get("max_clique"):
        return                                          # capped: the part is a subset by design
    left = pd.DataFrame({"i": np.arange(len(ids)), "v": ka})
    right = pd.DataFrame({"j": np.arange(len(ids)), "v": kb})
    if gk is not None:
        left["g"], right["g"] = gk, gk
    on = ["v"] + (["g"] if gk is not None else [])
    m = left[left["v"].notna()].merge(right[right["v"].notna()], on=on)
    m = m[m["i"] != m["j"]]
    want = set(zip(m["i"].tolist(), m["j"].tolist()))
    if g.get("symmetric", True):
        want |= {(j, i) for i, j in want}
    have = set(zip(c.s_n.tolist(), c.d_n.tolist()))
    if want != have:
        bad(f"graph: the key relation has {len(want)} directed pair(s) and the edges part "
            f"{len(have)} ({len(want - have)} missing, {len(have - want)} extra)")
    else:
        ok(f"graph: the edges part is exactly the key relation ({len(want)} directed pairs)")


def _graph_matrix(c):
    """The fitted space, one row per node in part order: the shaped features for knn, rounded
    to float32 as the network (and the bundle's reference nodes) read them; the raw coordinates
    projected as spatial_reproject recorded for spatial."""
    if c.g.get("method") == "knn":
        frames = [c.parts[k] for k in c.names if k in c.parts]
        if len(frames) != len(c.names):
            return None
        return pd.concat([f[c.g["features"]] for f in frames], ignore_index=True) \
            .apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32").astype(float)
    return projected(c.src.iloc[c.ids].reset_index(drop=True), c.g["coords"])


def _graph_digest(c, x):
    """A fitted index's matrix_digest against the train matrix recomputed here: float32 for
    knn, float64 for spatial, C order, the train nodes in part order."""
    want = c.fit.get("matrix_digest")
    if c.gk is not None or not want:
        return
    m = np.ascontiguousarray(x[c.part_of == "train"],
                             dtype="float32" if c.g["method"] == "knn" else "float64")
    got = "sha256:" + hashlib.sha256(m.tobytes()).hexdigest()
    if got != want:
        bad(f"graph: the fitted entry's matrix_digest {want} is not the train matrix's ({got}) — "
            "edges_for would refuse every reference built from the shaped train part")
    else:
        ok("graph: the fitted entry's matrix_digest is the train matrix's, recomputed")


def _graph_reach(c, x):
    """Per node: how many neighbours it must have (`need`) and the distance of the farthest
    of them (`kth`), from scipy's distances — not the executor's."""
    from scipy.spatial.distance import cdist
    g, n = c.g, len(c.ids)
    metric = "cosine" if g.get("metric") == "cosine" else "euclidean"
    k, radius, cap = g.get("k"), g.get("radius_m"), g.get("max_degree")
    keep_n = min(v for v in (k, cap) if v) if (k or cap) else None
    if c.gk is None:                                  # a fitted index: train candidates
        groups = [(np.arange(n), np.flatnonzero(c.part_of == "train"))]
    else:                                             # within each graph, unfitted
        codes = pd.factorize(pd.Series(c.gk, dtype=object))[0]
        groups = [(m, m) for m in (np.flatnonzero(codes == i) for i in range(codes.max() + 1))]
    kth, need = np.full(n, -np.inf), np.zeros(n, dtype=int)
    for srcs, cand in groups:
        if not len(cand):
            continue
        step_ = max(1, 2000000 // len(cand))
        for c0 in range(0, len(srcs), step_):
            chunk = srcs[c0:c0 + step_]
            D = cdist(x[chunk], x[cand], metric=metric)
            D[np.isnan(D)] = 1.0                      # a zero vector: cosine distance 1
            D[chunk[:, None] == cand[None, :]] = np.inf
            if radius is not None:
                D[D > radius] = np.inf
            cnt = np.isfinite(D).sum(axis=1)
            want = cnt if keep_n is None else np.minimum(keep_n, cnt)
            srt = np.sort(D, axis=1)
            need[chunk] = want
            kth[chunk] = np.where(want > 0, srt[np.arange(len(chunk)), np.maximum(want, 1) - 1],
                                  -np.inf)
    return kth, need


def _graph_distances(c, x):
    """Each edge's distance in the fitted space, recomputed (numpy, not the executor's)."""
    a, b = x[c.s_n], x[c.d_n]
    if c.g.get("metric") != "cosine":
        return np.sqrt(((a - b) ** 2).sum(axis=1))
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((na == 0) | (nb == 0), 1.0, 1.0 - (a * b).sum(axis=1) / (na * nb))


def _graph_neighbours(c):
    """knn / spatial: the fitted matrix's digest; each edge is among its source's nearest (or
    the reverse of one), each node has all of its own, and a distance weight is the distance."""
    g = c.g
    x = _graph_matrix(c)
    if x is None:
        warn("graph: a part is missing; the neighbour recount is skipped")
        return
    _graph_digest(c, x)
    kth, need = _graph_reach(c, x)
    dvec = _graph_distances(c, x)
    if c.gk is None:
        cand_sd, cand_ds = c.part_of[c.d_n] == "train", c.part_of[c.s_n] == "train"
    else:
        cand_sd = cand_ds = c.gk[c.s_n] == c.gk[c.d_n]
    fwd = cand_sd & (dvec <= kth[c.s_n] * (1 + 1e-9) + 1e-9)
    rev = cand_ds & (dvec <= kth[c.d_n] * (1 + 1e-9) + 1e-9) if g.get("symmetric", True) \
        else np.zeros(len(dvec), dtype=bool)
    unjust = int((~(fwd | rev)).sum())
    short = int((np.bincount(c.s_n[fwd], minlength=len(c.ids)) < need).sum())
    wbad = int((~(np.abs(c.w - dvec) <= 1e-9 * np.maximum(1.0, np.abs(dvec)))).sum()) \
        if g.get("weight") == "distance" else 0
    k, radius, cap = g.get("k"), g.get("radius_m"), g.get("max_degree")
    what = (f"{k} nearest" if k else f"within {radius} m") + (f", at most {cap}" if cap else "")
    if unjust:
        bad(f"graph: {unjust} {g['method']} edge(s) are not among their source's {what} "
            "candidates in the fitted space (recomputed independently)")
    if short:
        bad(f"graph: {short} node(s) lack some of their {what} neighbours")
    if wbad:
        bad(f"graph: {wbad} edge weight(s) differ from the recomputed distance")
    if not (unjust or short or wbad):
        ok(f"graph: every node's {what} {'train ' if c.gk is None else ''}neighbours are its "
           f"edges, recomputed independently in the fitted space"
           + (", with the distance weights" if g.get("weight") == "distance" else ""))


# --------------------------------------------------------------------------- the graph's twin
def twin_graph(mod, t):
    """The graph half of the twin check: `edges_for` (level node) or `edges_within` (level
    graph) run on the part's raw rows must give exactly the part's edges the served graph
    carries — a query's attachments to the reference, the part's edges between the query rows
    and the reference nodes in either direction (train and valid for the test part, train for
    the valid part), or the edges inside each whole graph. `t` carries what the frame half
    already read: raw, out_dir, man, fitted, part, rows, history, tol. A module without a
    graph has no graph half."""
    files = t["man"]["outputs"]["files"]
    if getattr(mod, "GRAPH", None) is None:
        return
    if "edges" not in files:
        bad("twin: the module serves a graph but the manifest has no edges part to compare "
            "it with")
        return
    t["edges"] = read_shaped(t["out_dir"] / Path(files["edges"]["path"]).name)
    t["q_pos"] = np.asarray(t["fitted"]["__split_raw__"][t["part"]], dtype="int64")
    served = (_twin_within if mod.GRAPH["level"] == "graph" else _twin_node)(mod, t)
    if served is not None:
        _twin_compare(served, t["tol"])


def _twin_within(mod, t):
    """edges_within on the part's rows, and the part's edges inside that part."""
    part, e = t["part"], t["edges"]
    try:
        got = mod.edges_within(t["rows"], history=t["history"])
    except Exception as ex:
        bad(f"twin: edges_within() raised {type(ex).__name__}: {ex}")
        return None
    got = pd.DataFrame({"src_pos": t["q_pos"][got["src_pos"].to_numpy()],
                        "dst_pos": t["q_pos"][got["dst_pos"].to_numpy()],
                        "weight": got["weight"].to_numpy(dtype=float)})
    want = e[(e["src_part"] == part) & (e["dst_part"] == part)][["src_pos", "dst_pos", "weight"]]
    return got, want, "edges_within", f"the edges inside the {part} part's graphs"


def twin_reference(mod, t, ref_parts):
    """The reference nodes as /vega-pack ships them (`vega/graph-reference@1`, D2 §3.4): dense
    ids, the train rows then the valid rows in `__split_raw__` order, every output feature
    column as shaped — float32 numerics, int64 integer ids — beside each `raw:<column>` the
    module names, read from the input; no raw position, no part column. Passed as
    `nodes[["pos", *EDGES_FOR_COLUMNS]]`, as the GraphAssembler passes it. Returns (reference,
    the raw position behind each `pos`), or None (reported)."""
    raw_names = mod.GRAPH["reference"]["raw"]
    sets, files = t["fitted"]["__split_raw__"], t["man"]["outputs"]["files"]
    frames = []
    for k in ref_parts:
        f = read_shaped(t["out_dir"] / Path(files[k]["path"]).name)
        f = f.drop(columns=[c for c in (t["man"].get("label"),) if c in f.columns])
        f = f.astype({c: ("float32" if f[c].dtype.kind == "f" else "int64") for c in f.columns
                      if isinstance(f[c].dtype, np.dtype) and f[c].dtype.kind in "fiu"})
        clash = [n for n in raw_names if n in f.columns]
        if clash:
            bad(f"twin: the shaped {k} part has a column {clash[0]!r}, the name the reference "
                "gives an input column edges_for reads")
            return None
        for n, col in raw_names.items():
            f[n] = t["raw"][col].to_numpy()[np.asarray(sets[k], dtype="int64")]
        frames.append(f)
    nodes = pd.concat(frames, ignore_index=True)
    nodes.insert(0, "ref", np.arange(len(nodes), dtype="int64"))
    ref_raw = np.concatenate([np.asarray(sets[k], dtype="int64") for k in ref_parts])
    return nodes.rename(columns={"ref": "pos"})[["pos", *mod.EDGES_FOR_COLUMNS]], ref_raw


def _twin_node(mod, t):
    """edges_for on the part's rows against the reference before it, and the part's
    attachments of those rows: every (query, reference) pair it joins in either direction,
    once, weighed by the query-sourced row when both exist (R-A3)."""
    part, e = t["part"], t["edges"]
    order = ["train", "valid", "test"]
    ref_parts = [k for k in order[:order.index(part)] if k in t["fitted"]["__split_raw__"]] \
        if part in order else []
    built = twin_reference(mod, t, ref_parts) if ref_parts else None
    if built is None:
        if not ref_parts:
            ok(f"twin: the {part} part has no reference before it; edges_for is not exercised")
        return None
    try:
        got = mod.edges_for(t["rows"], built[0], history=t["history"])
    except Exception as ex:
        bad(f"twin: edges_for() raised {type(ex).__name__}: {ex}")
        return None
    if list(got.columns) != ["query_pos", "ref_pos", "weight"]:
        bad(f"twin: edges_for returned columns {list(got.columns)}, not "
            "['query_pos', 'ref_pos', 'weight']")
        return None
    got = pd.DataFrame({"src_pos": t["q_pos"][got["query_pos"].to_numpy()],
                        "dst_pos": built[1][got["ref_pos"].to_numpy()],
                        "weight": got["weight"].to_numpy(dtype=float)})
    cols = ["src_pos", "dst_pos", "weight"]
    own = e[(e["src_part"] == part) & e["dst_part"].isin(ref_parts)][cols]
    into = e[e["src_part"].isin(ref_parts) & (e["dst_part"] == part)] \
        .rename(columns={"src_pos": "dst_pos", "dst_pos": "src_pos"})[cols]
    want = pd.concat([own, into], ignore_index=True).drop_duplicates(cols[:2], keep="first")
    # The part's own query -> query rows are the edges a served query never carries, by
    # construction: a reading about the method, not a defect of the module.
    qq = int(((e["src_part"] == part) & (e["dst_part"] == part)).sum())
    ok(f"twin: the served graph carries no {part}->{part} edge ({qq} such row(s) in the part "
       "are not attachments: an answer never depends on the rows sent beside it)")
    return got, want, "edges_for", (f"the {part} rows' attachments to {'+'.join(ref_parts)} "
                                    "(either direction), served from a vega/graph-reference@1 "
                                    "frame")


def _twin_compare(served, tol):
    """The served edges against the part's, as sets, then weight by weight."""
    got, want, name, what = served
    got = got.sort_values(["src_pos", "dst_pos"], kind="stable").reset_index(drop=True)
    want = want.sort_values(["src_pos", "dst_pos"], kind="stable").reset_index(drop=True)
    a_set = set(zip(got["src_pos"].tolist(), got["dst_pos"].tolist()))
    b_set = set(zip(want["src_pos"].tolist(), want["dst_pos"].tolist()))
    if a_set != b_set:
        extra, miss = sorted(a_set - b_set), sorted(b_set - a_set)
        bad(f"twin: {name} diverges from {what}: {len(miss)} edge(s) missing"
            + (f" (the first: {miss[0][0]} -> {miss[0][1]})" if miss else "")
            + f", {len(extra)} extra" + (f" (the first: {extra[0][0]} -> {extra[0][1]})"
                                        if extra else ""))
        return
    wa, wb = got["weight"].to_numpy(dtype=float), want["weight"].to_numpy(dtype=float)
    same_nan = np.isnan(wa) == np.isnan(wb)
    fin = ~np.isnan(wa) & ~np.isnan(wb)
    close = np.abs(wa[fin] - wb[fin]) <= tol * np.maximum(1.0, np.abs(wb[fin]))
    if not same_nan.all() or not close.all():
        i = int(np.flatnonzero(~same_nan)[0]) if not same_nan.all() else \
            int(np.flatnonzero(fin)[np.flatnonzero(~close)[0]])
        bad(f"twin: {name} weight diverges on {int(got['src_pos'][i])} -> "
            f"{int(got['dst_pos'][i])}: {wb[i]!r} in the edges part, {wa[i]!r} served")
        return
    exact = bool(np.array_equal(wa, wb, equal_nan=True))
    ok(f"twin: {name} reproduces {what} — {len(b_set)} edge(s), "
       + ("weights byte for byte" if exact else f"weights within {tol:g} × max(1, |w|)"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--psi-threshold", type=float, default=0.25)
    ap.add_argument("--label-tolerance", type=float, default=0.15,
                    help="allowed absolute difference in label rate between a part and the whole")
    ap.add_argument("--json", default=None, help="write the machine-readable report here")
    ap.add_argument("--transform-only", action="store_true",
                    help="run the twin check on the emitted preprocess module instead")
    ap.add_argument("--module", default=None,
                    help="the emitted preprocess.py (with --transform-only)")
    ap.add_argument("--part", default="test",
                    help="which shaped part the twin check reproduces (default test)")
    ap.add_argument("--tolerance", type=float, default=1e-12,
                    help="relative slack a float column may need when a reduction sums in a "
                         "different order (with --transform-only)")
    a = ap.parse_args(argv)

    if a.transform_only:
        if not a.module:
            print("ERROR: --transform-only needs --module <preprocess.py>")
            return 2
        return twin_check(a)

    out_dir = Path(a.out_dir)
    recipe = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
    man = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    layers = read_layers(a.model)
    geo = layers.get("geometry") or {}
    src = read_any(recipe["input"]["path"])
    parts = {}
    # the graph's edges part is checked below, not read as a frame part
    for name, meta in ((k, v) for k, v in man["outputs"]["files"].items() if k != "edges"):
        p = out_dir / Path(meta["path"]).name
        if p.is_file():
            parts[name] = read_any(p)
        elif meta.get("digest") is None:
            warn(f"{name}: no data file (a --dry-run manifest); the data checks are skipped")
    if not parts:
        warn("no output data to verify")
    label = man.get("label")
    cand = None
    for c in (geo.get("partitions") or {}).get("candidates") or []:
        if c.get("label") == label:
            cand = c
    report = {"structural": "pass", "semantic": [], "distributional": [], "split": "pass",
              "spatial": "n/a", "graph": "n/a", "markers": []}

    # ---------------------------------------------------------------- structural
    cols = set().union(*[set(d.columns) for d in parts.values()]) if parts else set()
    user_dropped = set()
    kept_derived = set()
    for s in recipe["steps"]:
        if str(s.get("source", "")).startswith("user:"):
            user_dropped |= set(s.get("columns") or [])
        if s["op"] == "drop_derived":
            kept_derived |= set((s.get("params") or {}).get("keep") or [])
    if label:
        if label not in cols:
            bad(f"structural: the label {label!r} is not in the output at all")
        schema = man.get("output_schema") or {}
        feats = [c for c, v in schema.items() if v.get("role") == "feature"]
        if label in feats:
            bad(f"structural: the label {label!r} is listed among the feature columns")
        else:
            ok(f"structural: the label {label!r} is present and is not a feature")
    if cand:
        leak = [c for c in (cand.get("dropped_for_leakage") or []) if c in cols]
        if leak:
            bad(f"structural: leakage column(s) {leak} survive in the output")
        else:
            ok(f"structural: the leakage set {cand.get('dropped_for_leakage') or []} is absent")
    heads = [d.get("column") for d in _shaper().kept_derivations(geo)]
    surviving = [h for h in heads if h in cols and h != label and h not in kept_derived]
    if surviving:
        bad(f"structural: active derivation head(s) {surviving} survive without a keep")
    elif heads:
        ok(f"structural: every active derivation head is gone or explicitly kept "
           f"({sorted(kept_derived) or 'none kept'})")
    basis = [b for b in (geo.get("basis") or {}).get("members") or []]
    # A basis member that is also the chosen partition's leakage set is *supposed* to be gone:
    # drop_leakage removed it, and that is the whole point of the partition.
    leakage_set = set((cand or {}).get("dropped_for_leakage") or [])
    # So is a group split's key (F33): the split drops it, every valid and test group unseen.
    split_key = _shaper().split_key(recipe["steps"])
    missing = [b for b in basis if b not in cols and b not in user_dropped and b not in leakage_set
               and not (b == split_key[0] and not split_key[1])]
    encoded = set()
    for s in recipe["steps"]:
        if s["op"] in ("encode", "project", "datetime_expand", "bin"):
            encoded |= set(s.get("columns") or [])
    missing = [b for b in missing if b not in encoded]
    if missing:
        bad(f"structural: basis member(s) {missing} are missing and no user: step dropped them")
    elif basis:
        ok("structural: every basis member survives or was transformed by a named step")
    if FAIL:
        report["structural"] = "fail"

    # ---------------------------------------------------------------- semantic
    # The concatenated output is not in the input's row order (a time split reorders it), so
    # every semantic check that compares an output row with its input row aligns through the
    # split's own index sets. Without them — and with the row count changed by a dedupe — the
    # alignment is not knowable and the check says so rather than guessing.
    split_idx = {}
    for st in man.get("steps") or []:
        if st["op"] == "split":
            split_idx = (st.get("parameters_fitted") or {}).get("indices") or {}
    # PATCH (CIDS 2026-09-15): the manifest's split indices are `__split__` -- positions in
    # the frame the SPLIT saw. The dropped-head check below indexes `src`, the RAW input
    # frame, with them. The two address spaces differ exactly when a dedupe ran before the
    # split, which this contract's own SS 2 states and which the analysis here forces (the
    # dedupe must precede the split or duplicates straddle the boundary). With 308,381 rows
    # removed first, every row after the first removal is read from a different flow, and a
    # pure identity such as `SYN Flag Count = Fwd PSH Flags` scores a determination ratio of
    # 0.0000 -- reported as "a value-losing step broke the relation" when nothing broke.
    # `fitted.json.__split_raw__` carries the raw-file positions; prefer it when present.
    _fit_p = Path(a.out_dir) / "fitted.json"
    if _fit_p.exists():
        try:
            _raw = (json.loads(_fit_p.read_text()) or {}).get("__split_raw__") or {}
        except Exception:
            _raw = {}
        if _raw and set(_raw) == set(split_idx or _raw):
            split_idx = _raw
    frames, origins = [], []
    for name, d in parts.items():
        frames.append(d)
        if name in split_idx:
            origins.extend(int(i) for i in split_idx[name])
        elif len(parts) == 1 and len(d) == len(src):
            origins.extend(range(len(d)))
        else:
            origins = None
            break
    all_out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if origins is not None and len(origins) == len(all_out):
        all_out.index = pd.Index(origins, name="__row__")
    else:
        origins = None
        if parts:
            warn("semantic: the output rows cannot be aligned with the input rows (no split "
                 "index and the row count changed); the dropped-head checks are skipped")
    for d in geo.get("derivations") or []:
        head, body = d.get("column"), list(d.get("body") or [])
        rule = {"rule_id": d.get("rule_id"), "column": head, "formula": d.get("formula")}
        if not _shaper().derivation_kept(d):
            # a rejected derivation derives nothing (F44): there is no relation to recompute
            rule.update({"empirical": "not applicable", "symbolic": "untested",
                         "why": "the geometry layer rejected the derivation (semantic "
                                f"{((d.get('provenance') or {}).get('semantic') or {}).get('status') or 'unstated'}"
                                f"{', kept: false' if d.get('kept') is False else ''}); only a kept rule derives a column"})
            report["semantic"].append(rule)
            continue
        body_present = [c for c in body if c in all_out.columns]
        if not body_present:
            rule.update({"empirical": "not applicable", "why": "no body column survives",
                         "symbolic": "untested"})
            report["semantic"].append(rule)
            continue
        parsed = parse_formula(d.get("formula"))
        if head in all_out.columns and parsed and set(body) <= set(all_out.columns):
            _, expr = parsed
            try:
                got = all_out.eval(expr)
                want = pd.to_numeric(all_out[head], errors="coerce")
                m = np.isfinite(got) & np.isfinite(want)
                agree = int((np.abs(got[m] - want[m]) <= 1e-6 * np.maximum(1, np.abs(want[m]))).sum())
                total = int(m.sum())
                verdict = "confirmed" if total and agree == total else "refuted"
                rule.update({"empirical": verdict, "rows": f"{agree}/{total}"})
                (ok if verdict == "confirmed" else bad)(
                    f"semantic: {d.get('rule_id')} {verdict} on the output ({agree}/{total} rows)")
            except Exception as e:
                rule.update({"empirical": "untested", "why": f"formula did not evaluate: {e}"})
                warn(f"semantic: {d.get('rule_id')} could not be evaluated on the output ({e})")
        else:
            # the head is gone: the body must still determine the ORIGINAL head values on the
            # rows that survived, or a value-losing transform has quietly broken the relation.
            if head in src.columns and len(body_present) == len(body) and origins is not None:
                try:
                    joined = all_out[body_present].copy()
                    joined["__head__"] = src[head].to_numpy()[np.asarray(all_out.index)]
                    det = determination(joined, body_present, "__head__")
                except Exception as e:
                    det = None
                    rule["why"] = f"alignment failed: {e}"
                # PATCH (CIDS 2026-09-15), part 2: the verdict needs the INPUT's own ratio as
                # its baseline. The message this branch prints -- "a value-losing STEP broke
                # the relation" -- is a claim about the shaping, and an absolute 0.999 floor
                # cannot support it on a file whose stored head is already inconsistent with
                # its stored body. Measured here: R1 (Average Packet Size) determines at
                # 0.963188 on the retained RAW rows and at 0.963200 on the shaped output, so
                # the recipe degraded it by nothing at all, and the accusation was the
                # threshold's rather than the data's. Cause: the published CSVs truncate a
                # subset of cells to 3 significant digits before serialisation, which breaks
                # the arithmetic between columns that a rule relates.
                det_in = None
                if det is not None:
                    try:
                        _pos = np.asarray(all_out.index)
                        _ji = pd.DataFrame({c: src[c].to_numpy()[_pos] for c in body_present})
                        _ji["__head__"] = src[head].to_numpy()[_pos]
                        det_in = determination(_ji, body_present, "__head__")
                    except Exception:
                        det_in = None
                if det is None:
                    rule.update({"empirical": "untested", "why": rule.get("why", "could not align the rows")})
                elif det < 0.999 and det_in is not None and det >= det_in - 1e-6:
                    rule.update({"empirical": "confirmed",
                                 "why": "the head is dropped, and the INPUT itself does not "
                                        f"determine it at group level (ratio {det_in:.6f}, the "
                                        "published CSV's 3-significant-digit truncation); the "
                                        "shaping preserved the relation exactly as the file holds it",
                                 "determination_ratio": round(det, 6),
                                 "determination_ratio_input": round(det_in, 6)})
                    ok(f"semantic: {d.get('rule_id')} — body determines the dropped head as well "
                       f"as the input does (output {det:.4f} vs input {det_in:.4f}); the file's own "
                       f"truncation is the shortfall, not a step")
                elif det >= 0.999:
                    rule.update({"empirical": "confirmed",
                                 "why": "the head is dropped; its body still determines the "
                                        "original values on the retained rows",
                                 "determination_ratio": round(det, 6),
                                 "determination_ratio_input": (round(det_in, 6) if det_in is not None else None)})
                    ok(f"semantic: {d.get('rule_id')} — body still determines the dropped head "
                       f"(ratio {det:.4f})")
                else:
                    rule.update({"empirical": "refuted", "determination_ratio": round(det, 6)})
                    bad(f"semantic: {d.get('rule_id')} — the retained body no longer determines "
                        f"{head!r} (ratio {det:.4f}): a value-losing step broke the relation")
            elif origins is None:
                rule.update({"empirical": "untested", "why": "the output rows could not be "
                                                             "aligned with the input rows"})
            else:
                rule.update({"empirical": "not applicable",
                             "why": "the head is not in the input either"})
        rule.setdefault("symbolic", "untested")
        report["semantic"].append(rule)
    # symbolic channel: the model's own engines, when it has rules and a browser is present
    chrome = os.environ.get("CHROME") or shutil.which("chromium") or shutil.which("google-chrome")
    swrl = []
    if a.model:
        html = Path(a.model).read_text(encoding="utf-8")
        m = re.search(r'<script id="model-swrl"[^>]*>([\s\S]*?)</script>', html)
        if m:
            try:
                swrl = json.loads(m.group(1)).get("rules") or []
            except json.JSONDecodeError:
                swrl = []
    runner = SKILL_DIR.parent / "domain-forge" / "scripts" / "run_query.py"
    if swrl and chrome and runner.is_file():
        try:
            proc = subprocess.run(
                [sys.executable, str(runner), a.model, "--engine", "swrl", "--query",
                 "SELECT (COUNT(*) AS ?n)\nWHERE { ?s ?p ?o }"],
                capture_output=True, text=True, timeout=180)
            if proc.returncode == 0:
                for r in report["semantic"]:
                    r["symbolic"] = "untested"
                warn("semantic: the model's SWRL rules ran, but this check compares the SHAPED "
                     "rows, which are not in the model's A-box; the symbolic channel stays "
                     "untested (re-forge the model on the shaped data to close it)")
            else:
                warn(f"semantic: the engine runner exited {proc.returncode}; symbolic untested")
        except Exception as e:
            warn(f"semantic: the engine runner failed ({e}); symbolic untested")
    elif swrl and not chrome:
        warn("semantic: no headless browser; the symbolic channel is untested for every rule")
    else:
        ok("semantic: the model carries no rules to run; the symbolic channel is untested "
           "by construction")

    # ---------------------------------------------------------------- distributional
    named = set()
    for s in recipe["steps"]:
        named |= set(s.get("columns") or [])
        named |= set((s.get("params") or {}).get("columns") or [])
    untouched = [c for c in src.columns if c in all_out.columns and c not in named]
    for c in untouched:
        try:
            if pd.api.types.is_numeric_dtype(src[c]):
                v = psi(src[c], all_out[c])
            else:
                v = psi_cat(src[c], all_out[c])
        except Exception:
            v = None
        if v is None:
            continue
        rec = {"column": c, "psi": round(v, 6), "expected": True}
        if v > a.psi_threshold:
            rec["expected"] = False
            suspects = [s["id"] for s in recipe["steps"]
                        if s["op"] in ("dedupe", "clip", "winsorize", "split", "custom")]
            rec["suspected_steps"] = suspects
            warn(f"distributional: {c} moved (PSI {v:.3f}) although no step names it; "
                 f"suspect {suspects or 'the split'}")
        report["distributional"].append(rec)
    if report["distributional"] and all(r["expected"] for r in report["distributional"]):
        ok(f"distributional: {len(report['distributional'])} untouched column(s) did not move "
           f"(PSI below {a.psi_threshold})")

    # ---------------------------------------------------------------- split hygiene
    split_step = next((s for s in recipe["steps"] if s["op"] == "split"), None)
    if split_step and len(parts) > 1:
        idx = {}
        for st in man["steps"]:
            if st["op"] == "split":
                idx = st["parameters_fitted"].get("indices") or {}
        sets = {k: set(v) for k, v in idx.items()}
        overlap = []
        keys = sorted(sets)
        for i, x in enumerate(keys):
            for y in keys[i + 1:]:
                if sets[x] & sets[y]:
                    overlap.append((x, y, len(sets[x] & sets[y])))
        if overlap:
            bad(f"split: rows appear in more than one part: {overlap}")
            report["split"] = "fail"
        else:
            ok(f"split: the {len(sets)} parts are disjoint "
               f"({', '.join(f'{k} {len(v)}' for k, v in sorted(sets.items()))})")
        p = split_step.get("params") or {}
        if p.get("kind") == "group" and p.get("by") in src.columns:
            g = src[p["by"]].astype(str)
            straddle = [k for k in sets for v in [set(g.iloc[sorted(sets[k])])]
                        if any(v & set(g.iloc[sorted(sets[o])]) for o in sets if o != k)]
            if straddle:
                bad(f"split: kind=group but group values straddle parts {sorted(set(straddle))}")
                report["split"] = "fail"
            else:
                ok("split: every group stays inside one part")
        if p.get("kind") == "time" and p.get("time_column") in src.columns:
            t = pd.to_datetime(src[p["time_column"]], errors="coerce")
            maxes = {k: t.iloc[sorted(v)].max() for k, v in sets.items() if v}
            mins = {k: t.iloc[sorted(v)].min() for k, v in sets.items() if v}
            order = [k for k in ("train", "valid", "test") if k in sets and sets[k]]
            bad_order = [(x, y) for x, y in zip(order, order[1:]) if maxes[x] > mins[y]]
            if bad_order:
                bad(f"split: kind=time but the parts overlap in time: {bad_order}")
                report["split"] = "fail"
            else:
                ok("split: the parts are ordered in time and do not overlap")
        if label and label in all_out.columns:
            whole = pd.to_numeric(src[label], errors="coerce")
            if whole.notna().any() and whole.nunique() <= 20:
                base = float(whole.mean())
                rates = {}
                for k, d in parts.items():
                    v = pd.to_numeric(d[label], errors="coerce")
                    rates[k] = float(v.mean()) if v.notna().any() else None
                off = {k: r for k, r in rates.items()
                       if r is not None and abs(r - base) > a.label_tolerance}
                report["label_rate"] = {"whole": round(base, 6),
                                        **{k: (round(v, 6) if v is not None else None)
                                           for k, v in rates.items()}}
                if off:
                    warn(f"split: label rate drifts from the whole ({base:.3f}) in "
                         f"{ {k: round(v, 3) for k, v in off.items()} } — expected for a time "
                         f"split, a defect for a stratified one")
                else:
                    ok(f"split: the label rate holds across parts (whole {base:.3f})")
        if check_row_order(recipe, out_dir, src, parts) == "fail":
            report["split"] = "fail"
    if check_leakage(recipe, man) == "fail":
        report["split"] = "fail"
    if check_split_key(recipe, man, parts) == "fail":
        report["split"] = "fail"

    # ---------------------------------------------------------------- spatial
    if any(s["op"].startswith("spatial_") for s in recipe["steps"]):
        crs = man.get("crs")
        report["spatial"] = "pass"
        proj = None
        for st in man.get("steps") or []:
            pr = (st.get("parameters_fitted") or {}).get("projection")
            if pr:
                proj = pr
        if not crs:
            bad("spatial: spatial steps ran but the manifest records no CRS")
            report["spatial"] = "fail"
        else:
            ok(f"spatial: the coordinates are declared in {crs}")
        if proj:
            report["projection"] = proj
            if proj.get("exact"):
                ok(f"spatial: metric columns were produced by {proj.get('kind')} "
                   f"({proj.get('from')} -> {proj.get('to')})")
            else:
                warn(f"spatial: metric columns were produced by the fallback projection "
                     f"({proj.get('kind')}, units {proj.get('units')}), not by the requested CRS: "
                     "install pyproj for an exact transform, and read the distances as local")
        geom_step = next((s for s in recipe["steps"] if s["op"] == "parse_geometry"), None)
        if geom_step:
            lat = (geom_step.get("params") or {}).get("lat")
            lon = (geom_step.get("params") or {}).get("lon")
            for c, lo, hi in ((lat, -90, 90), (lon, -180, 180)):
                if c and c in all_out.columns:
                    v = pd.to_numeric(all_out[c], errors="coerce")
                    outb = int(((v < lo) | (v > hi)).sum())
                    if outb:
                        bad(f"spatial: {outb} value(s) of {c} fall outside [{lo}, {hi}]")
                        report["spatial"] = "fail"
            if report["spatial"] == "pass":
                ok("spatial: every coordinate lies inside the declared bounds")

    # ---------------------------------------------------------------- graph
    report["graph"] = graph_report(recipe, man, out_dir, src, parts)

    report["markers"] = [f"OK: {m}" for m in OK] + [f"WARN: {m}" for m in WARN] + \
                        [f"ERROR: {m}" for m in FAIL]
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n{len(OK)} passed, {len(WARN)} warned, {len(FAIL)} failed")
    return 3 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
