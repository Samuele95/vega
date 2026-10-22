#!/usr/bin/env python3
"""
analysis.py — the empirical engine of /data-lens: runs every module of
`references/analysis-contract.md` §3 over a tabular dataset, on top of the geometry
the /dataset-forge model already settled, and writes ONE JSON document.

    python3 analysis.py DATASET --model MODEL.html --out analysis.json
                        [--split COL | --split-file PATH] [--modules quality,inference,...]
                        [--seed 7] [--figures DIR] [--max-rows N]

It computes; it does not judge. Every test it runs carries the assumption check that
licenses it (`assumptions_checked`: passed | violated | n/a) and, when an assumption is
violated, the robust alternative named in `references/method-catalog.md` is the test that
actually ran (`switched_to`). Every p-value it reports travels with an effect size and the
multiple-comparison correction applied across its family (Benjamini-Hochberg per family,
Holm within a pre-declared set of contrasts). Nothing here is a finding: the analyst reads
this JSON and admits findings against the decision test.

Determinism: every sample, split, resample and model is seeded from --seed; the JSON is
written with sorted keys and no timestamps, so two runs of the same inputs are byte-identical.

Exit codes: 0 wrote the JSON; 2 input error.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCHEMA = "data-lens/analysis@1"
MODULES = ("quality", "distributions", "relations", "inference", "segments", "importance",
           "time_series", "spatial", "drift")
EXCLUDED_ROLES = {"identity", "key", "degenerate", "constant"}
ALPHA = 0.05
_LAYER_RE = re.compile(r'<script\s+id="layer-geometry-data"[^>]*>([\s\S]*?)</script>', re.IGNORECASE)
_TURTLE_RE = re.compile(r'<script\s+id="domain-model"[^>]*>([\s\S]*?)</script>', re.IGNORECASE)

MARKERS: list[str] = []


def mark(s):
    MARKERS.append(s)
    print(s)


# --------------------------------------------------------------------------- json safety
def jsonable(o):
    """Numpy/pandas -> plain JSON, NaN/inf -> None. Applied once to the whole document."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 10)
    if isinstance(o, (np.ndarray,)):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is None or isinstance(o, str):
        return o
    if isinstance(o, (np.str_,)):
        return str(o)
    return str(o)


# --------------------------------------------------------------------------- statistics helpers
def bh(pvals):
    """Benjamini-Hochberg adjusted p-values, in input order. Pure numpy (no statsmodels dep)."""
    p = np.asarray([1.0 if v is None or not np.isfinite(v) else float(v) for v in pvals], dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return [float(v) for v in out]


def holm(pvals):
    p = np.asarray([1.0 if v is None or not np.isfinite(v) else float(v) for v in pvals], dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    adj = np.maximum.accumulate(p[order] * (n - np.arange(n)))
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return [float(v) for v in out]


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return None, None
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return 0.0, None
    d = (a.mean() - b.mean()) / sp
    g = d * (1 - 3 / (4 * (na + nb) - 9))          # Hedges correction
    se = math.sqrt((na + nb) / (na * nb) + d * d / (2 * (na + nb)))
    return float(g), [float(g - 1.96 * se), float(g + 1.96 * se)]


def rank_biserial(a, b, u):
    na, nb = len(a), len(b)
    return float(2 * u / (na * nb) - 1) if na and nb else None


def cramers_v(table):
    from scipy import stats as st
    table = np.asarray(table, float)
    if table.size == 0 or table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, None, None
    chi2, p, dof, exp = st.chi2_contingency(table, correction=False)
    n = table.sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))          # bias-corrected (Bergsma)
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    v = math.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1)))
    return float(chi2), float(p), float(min(1.0, v)), float(exp.min())


def mc_chi2_p(table, seed, permutations=None):
    """Monte-Carlo chi-square: the share of tables with the observed margins (one variable's
    labels permuted against the other's) whose statistic is at least the observed one. 999
    permutations, 199 above 50 000 rows; seeded, so the p is the same on a re-run."""
    t = np.asarray(table, dtype=np.int64)
    r, k = t.shape
    n = int(t.sum())
    B = permutations or (999 if n <= 50000 else 199)
    exp = np.outer(t.sum(1), t.sum(0)) / n
    obs = float(((t - exp) ** 2 / exp).sum())
    rows = np.repeat(np.arange(r), t.sum(1))
    cols = np.concatenate([np.repeat(np.arange(k), t[i]) for i in range(r)])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(B):
        sim = np.bincount(rows * k + rng.permutation(cols), minlength=r * k).reshape(r, k)
        hits += float(((sim - exp) ** 2 / exp).sum()) >= obs - 1e-9
    return float((1 + hits) / (1 + B)), B


def robust_chi2(table, seed):
    """The catalog's robust alternative to a chi-square whose expected counts fall under 5:
    Fisher's exact test on a 2x2 table, the Monte-Carlo chi-square on a larger one. Returns the
    keys a test record takes (`switched_to`, `p`, and `permutations` for the Monte-Carlo)."""
    from scipy import stats as st
    t = np.asarray(table)
    try:
        if t.shape == (2, 2):
            _, p2 = st.fisher_exact(t)
            return {"switched_to": "fisher-exact", "p": float(p2)}
        p2, B = mc_chi2_p(t, seed)
        return {"switched_to": "monte-carlo chi-square", "p": p2, "permutations": B}
    except Exception:
        return {"switched_to": None}


def _wilson(x, n):
    """Wilson score interval of a proportion (95 %)."""
    if n == 0:
        return 0.0, 1.0
    z, ph = 1.959964, x / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def eta_squared(groups):
    """omega^2 for a one-way layout (bias-corrected eta^2); returns (omega2, eta2)."""
    groups = [np.asarray(g, float) for g in groups if len(g) > 0]
    if len(groups) < 2:
        return None, None
    allv = np.concatenate(groups)
    n, k = allv.size, len(groups)
    gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    sst = ((allv - gm) ** 2).sum()
    ssw = sst - ssb
    if sst <= 0 or n - k <= 0:
        return None, None
    msw = ssw / (n - k)
    omega2 = (ssb - (k - 1) * msw) / (sst + msw)
    return float(max(0.0, omega2)), float(ssb / sst)


def epsilon_squared(h, n, k):
    if n <= k:
        return None
    return float(max(0.0, (h - k + 1) / (n - k)))


def boot_ci(fn, arrays, seed, reps=2000):
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        res = [a[rng.integers(0, len(a), len(a))] for a in arrays]
        try:
            v = fn(*res)
        except Exception:
            v = np.nan
        if v is not None and np.isfinite(v):
            stats.append(v)
    if len(stats) < reps // 4:
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def psi(expected, actual, bins=10):
    """Population stability index over quantile bins of `expected`."""
    e = np.asarray(pd.to_numeric(expected, errors="coerce"), float)
    a = np.asarray(pd.to_numeric(actual, errors="coerce"), float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if e.size < 10 or a.size < 10:
        return None
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if qs.size < 3:
        return None
    qs[0], qs[-1] = -np.inf, np.inf
    ec = np.histogram(e, bins=qs)[0] / e.size
    ac = np.histogram(a, bins=qs)[0] / a.size
    ec, ac = np.clip(ec, 1e-6, None), np.clip(ac, 1e-6, None)
    return float(np.sum((ac - ec) * np.log(ac / ec)))


def psi_categorical(expected, actual):
    e = pd.Series(expected).astype(str).value_counts(normalize=True)
    a = pd.Series(actual).astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    ev = np.clip(e.reindex(idx).fillna(0).values, 1e-6, None)
    av = np.clip(a.reindex(idx).fillna(0).values, 1e-6, None)
    return float(np.sum((av - ev) * np.log(av / ev)))


def normality(x, seed):
    """Shapiro on <= 5000 rows (subsampled, seeded), D'Agostino K^2 above. Returns (test, p)."""
    from scipy import stats as st
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 8 or np.allclose(x, x[0]):
        return "none", None
    if x.size <= 5000:
        try:
            return "shapiro", float(st.shapiro(x).pvalue)
        except Exception:
            return "none", None
    rng = np.random.default_rng(seed)
    try:
        return "dagostino", float(st.normaltest(x).pvalue)
    except Exception:
        return "none", None


def theil_sen(x, y):
    from scipy import stats as st
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return None
    r = st.theilslopes(y[m], x[m], 0.95)
    return {"slope": float(r[0]), "intercept": float(r[1]), "ci": [float(r[2]), float(r[3])]}


def _acf(v, nlags):
    """Sample autocorrelation of `v` at lags 0..nlags (FFT, biased estimator, as statsmodels)."""
    v = np.asarray(v, float) - np.mean(v)
    n = v.size
    f = np.fft.rfft(v, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[:nlags + 1]
    return r / r[0] if r[0] > 0 else np.zeros(nlags + 1)


def mann_kendall(y, resid=None):
    """Mann-Kendall's trend test with its independence assumption checked (method-catalog.md,
    trend row): the residual after the Theil-Sen line (`resid`, else `y`) is serially
    independent when its lag-1 autocorrelation lies inside +-1.96/sqrt(n). When it does not,
    the test's variance is too small and its p too optimistic, so the Hamed-Rao correction —
    the variance scaled by the significant rank-autocorrelation lags of the residual — is the
    robust alternative, and `p_reported` is its p."""
    from scipy import stats as st
    y = np.asarray(y, float)
    keep = np.isfinite(y)
    y = y[keep]
    n = y.size
    if n < 10 or n > 20000:
        return None
    s = 0
    # vectorised sign sum
    for i in range(n - 1):
        s += np.sign(y[i + 1:] - y[i]).sum()
    _, counts = np.unique(y, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var <= 0:
        return None
    z = (s - np.sign(s)) / math.sqrt(var)
    p = float(2 * (1 - st.norm.cdf(abs(z))))
    r = np.asarray(resid, float)[keep] if resid is not None and len(resid) == keep.size else y
    ac = _acf(st.rankdata(r), n - 1)
    bound = 1.96 / math.sqrt(n)
    serial = bool(abs(ac[1]) > bound)
    k = np.arange(1, n)
    sig = np.abs(ac[1:]) > bound
    ratio = 1.0 + 2.0 / (n * (n - 1) * (n - 2)) * float(
        np.sum(((n - k) * (n - k - 1) * (n - k - 2) * ac[1:])[sig]))
    robust = {"method": "Hamed-Rao variance correction (significant rank-autocorrelation lags of "
                        "the residual after the Theil-Sen line)", "variance_ratio": ratio}
    if ratio > 0:
        zc = (s - np.sign(s)) / math.sqrt(var * ratio)
        robust.update({"z": float(zc), "p": float(2 * (1 - st.norm.cdf(abs(zc))))})
    return {"z": float(z), "p": p, "assumptions": ["serially independent residuals"],
            "check": {"lag1_autocorrelation": float(ac[1]), "bound": bound},
            "assumptions_checked": "violated" if serial else "passed", "robust": robust,
            "p_reported": robust.get("p") if serial else p}


def _period_stability(v, period):
    """The seasonality row's check (method-catalog.md): the period found on the whole series is
    found again on each half. A half too short to hold two periods cannot answer (`n/a`); halves
    that disagree make the season not one stable period (`violated`), and the robust reading is
    the period per half, with no single seasonal strength claimed for the whole."""
    half = len(v) // 2
    if half < 2 * period + 1:
        return {"halves": None, "assumptions": ["a period stable across the series"],
                "assumptions_checked": "n/a",
                "check": f"each half holds fewer than two periods of {period}"}
    p1, _ = _acf_peak(v[:half], half)
    p2, _ = _acf_peak(v[half:], len(v) - half)
    tol = max(1, round(0.1 * period))
    stable = p1 is not None and p2 is not None and abs(p1 - period) <= tol and abs(p2 - period) <= tol
    out = {"halves": [p1, p2], "assumptions": ["a period stable across the series"],
           "assumptions_checked": "passed" if stable else "violated",
           "check": f"the ACF peak of each half within {tol} lag(s) of {period}"}
    if not stable:
        out["robust"] = {"method": "the period read per half; no single seasonal strength claimed",
                         "period_by_half": [p1, p2]}
    return out


def _acf_peak(v, n):
    """The seasonal period: the strongest LOCAL autocorrelation peak above 0.15 from lag 3 on
    (a decaying ACF — a trend — has its maximum at lag 2 and no season at all), or None."""
    from statsmodels.tsa.stattools import acf as sm_acf
    ac = sm_acf(v, nlags=min(60, n // 2), fft=True)
    peak, best = None, 0.0
    for L in range(3, len(ac) - 1):
        if ac[L] > ac[L - 1] and ac[L] >= ac[L + 1] and ac[L] > 0.15 and ac[L] > best:
            peak, best = L, ac[L]
    return (peak if peak and 2 <= peak <= n // 2 else None), ac


# --------------------------------------------------------------------------- input
def read_dataset(path, sep=None):
    p = Path(path)
    s = p.suffix.lower()
    if s in (".csv", ".txt"):
        return pd.read_csv(p, sep=sep or ",", low_memory=False)
    if s == ".tsv":
        return pd.read_csv(p, sep=sep or "\t", low_memory=False)
    if s == ".parquet":    # Arrow's own filesystem: no Python in Arrow's threads at exit
        from pyarrow import fs
        return pd.read_parquet(p, filesystem=fs.LocalFileSystem())
    if s in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if s == ".jsonl":
        return pd.read_json(p, lines=True)
    if s == ".json":
        return pd.read_json(p)
    raise ValueError(f"unsupported dataset extension {s!r}")


def read_geometry(model_path):
    """Return (geometry_layer_dict | None, source_kind)."""
    if not model_path:
        return None, None
    html = Path(model_path).read_text(encoding="utf-8")
    kind = "dataset" if 'ex:sourceKind "dataset"' in html else "software-domain"
    m = _LAYER_RE.search(html)
    if not m:
        return None, kind
    try:
        return json.loads(m.group(1).strip()), kind
    except json.JSONDecodeError as e:
        mark(f"WARN: layer-geometry-data is not valid JSON ({e}); proceeding without geometry")
        return None, kind


def derivation_kept(d):
    """A derivation the geometry layer keeps (dataset-forge report-contract §2): the semantic
    channel confirms it and it is not set aside with `kept: false`. A rejected one stays in the
    layer with its triple, but only kept rules derive a column."""
    semantic = ((d.get("provenance") or {}).get("semantic") or {}).get("status")
    return semantic == "confirmed" and d.get("kept", True) is not False


def build_context(df, geo, split_col):
    """The statement of what this pass inherits. Roles and basis come from the geometry
    layer when present; without one, every all-distinct column is identity and the rest
    are dimensions (guessed, and the layer says so)."""
    ctx = {"typing": [], "basis": [], "derivations": [], "partition":
           {"label": None, "task": None, "features": [], "leakage": [], "provenance": "none"},
           "time": None, "spatial": None, "split_column": split_col}
    cols = list(df.columns)
    if geo:
        roles = {}
        for t in geo.get("typing") or []:
            roles[t.get("column")] = (t.get("final_type") or t.get("type"), t.get("role", "dimension"))
        for c in cols:
            ty, role = roles.get(c, (None, "dimension"))
            ctx["typing"].append({"column": c, "type": ty or infer_type(df[c]), "role": role})
        ctx["basis"] = [c for c in ((geo.get("basis") or {}).get("members") or []) if c in cols]
        for d in geo.get("derivations") or []:
            ctx["derivations"].append({"column": d.get("column"), "body": d.get("body") or [],
                                       "rule_id": d.get("rule_id"), "formula": d.get("formula"),
                                       "kept": derivation_kept(d)})
        P = geo.get("partitions") or {}
        chosen = P.get("chosen")
        cand = None
        for c in P.get("candidates") or []:
            if c.get("label") == chosen:
                cand = c
                break
        if cand is None and len(P.get("candidates") or []) == 1:
            cand, chosen = P["candidates"][0], P["candidates"][0].get("label")
        if cand:
            ctx["partition"] = {"label": chosen, "task": cand.get("task"),
                                "features": [c for c in (cand.get("features") or []) if c in cols],
                                "leakage": cand.get("dropped_for_leakage") or [],
                                "provenance": P.get("provenance") or "single-candidate"}
    else:
        for c in cols:
            distinct = df[c].nunique(dropna=True)
            role = "identity" if distinct == len(df) and len(df) > 1 else \
                   ("constant" if distinct <= 1 else "dimension")
            ctx["typing"].append({"column": c, "type": infer_type(df[c]), "role": role})
        ctx["basis"] = [t["column"] for t in ctx["typing"] if t["role"] == "dimension"]
    # datetime and coordinates are detected here, never roles (contract §1)
    ctx["time"] = detect_time(df, ctx)
    ctx["spatial"] = detect_spatial(df, ctx)
    return ctx


def is_texty(s):
    """True for object/str-backed columns (pandas 2 object dtype and pandas 3 str dtype)."""
    return (s.dtype == object or pd.api.types.is_string_dtype(s)) and \
        not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s)


def infer_type(s):
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    nun = s.nunique(dropna=True)
    if nun <= max(20, 0.02 * max(1, len(s))):
        return "nominal"
    return "text"


def detect_time(df, ctx):
    best = None
    for t in ctx["typing"]:
        c = t["column"]
        s = df[c]
        parsed = None
        if pd.api.types.is_datetime64_any_dtype(s):
            parsed = s
        elif is_texty(s):
            sample = s.dropna().astype(str).head(200)
            if sample.empty:
                continue
            if sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.9 or \
               sample.str.match(r"^\d{2}/\d{2}/\d{4}").mean() > 0.9:
                parsed = pd.to_datetime(s, errors="coerce")
        if parsed is None or parsed.notna().sum() < max(10, 0.5 * len(df)):
            continue
        span = [str(parsed.min()), str(parsed.max())]
        d = parsed.dropna().drop_duplicates().sort_values().diff().dropna()
        modal = d.mode()
        resolution = str(modal.iloc[0]) if len(modal) else None
        regular = bool(len(d) and (d == modal.iloc[0]).mean() > 0.9) if len(modal) else False
        cand = {"column": c, "resolution": resolution, "span": span, "regular": regular,
                "n_parsed": int(parsed.notna().sum())}
        if best is None or cand["n_parsed"] > best["n_parsed"]:
            best = cand
    return best


def detect_spatial(df, ctx):
    lat = lon = None
    for t in ctx["typing"]:
        c, n = t["column"], t["column"].lower()
        if t["type"] != "numeric":
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if lat is None and (n in ("lat", "latitude", "y_lat") or "latitude" in n or n.endswith("_lat")):
            if v.between(-90, 90).mean() > 0.95:
                lat = c
        if lon is None and (n in ("lon", "lng", "long", "longitude") or "longitude" in n or n.endswith("_lon")):
            if v.between(-180, 180).mean() > 0.95:
                lon = c
    if lat and lon:
        return {"columns": [{"column": lat, "pair": lon, "kind": "latlon"}], "crs": "unknown"}
    for t in ctx["typing"]:
        c = t["column"]
        if is_texty(df[c]):
            s = df[c].dropna().astype(str).head(50)
            if len(s) and s.str.match(r"^\s*(POINT|POLYGON|LINESTRING|MULTI)").mean() > 0.9:
                return {"columns": [{"column": c, "pair": None, "kind": "wkt"}], "crs": "unknown"}
    return None


# --------------------------------------------------------------------------- module: quality
def mod_quality(df, ctx, seed):
    from scipy import stats as st
    ev = {"n_rows": int(len(df)), "n_columns": int(df.shape[1]), "columns": [], "duplicates": {},
          "missingness_dependence": [], "type_coherence": [], "range_anomalies": [],
          "category_hygiene": []}
    roles = {t["column"]: t["role"] for t in ctx["typing"]}
    types = {t["column"]: t["type"] for t in ctx["typing"]}
    for c in df.columns:
        s = df[c]
        miss = int(s.isna().sum())
        ev["columns"].append({
            "column": c, "type": types.get(c), "role": roles.get(c),
            "missing": miss, "missing_rate": float(miss / max(1, len(df))),
            "distinct": int(s.nunique(dropna=True)),
            "cardinality_ratio": float(s.nunique(dropna=True) / max(1, s.notna().sum())),
            "dominant_share": float(s.value_counts(normalize=True, dropna=True).iloc[0])
            if s.notna().any() else None})
    dup_all = int(df.duplicated().sum())
    basis = [c for c in ctx["basis"] if c in df.columns]
    dup_basis = int(df.duplicated(subset=basis).sum()) if basis else None
    ev["duplicates"] = {"exact_rows": dup_all, "exact_rate": float(dup_all / max(1, len(df))),
                        "on_basis": dup_basis, "basis": basis}

    # missingness mechanism: does the missing indicator depend on another column?
    tests, pv = [], []
    for c in df.columns:
        miss = df[c].isna()
        if not (0 < miss.mean() < 1) or miss.sum() < 10:
            continue
        for o in df.columns:
            if o == c or roles.get(o) in EXCLUDED_ROLES:
                continue
            if types.get(o) == "numeric":
                a = pd.to_numeric(df.loc[miss, o], errors="coerce").dropna().values
                b = pd.to_numeric(df.loc[~miss, o], errors="coerce").dropna().values
                if len(a) < 8 or len(b) < 8:
                    continue
                u, p = st.mannwhitneyu(a, b, alternative="two-sided")
                tests.append({"missing_in": c, "against": o, "test": "mann-whitney",
                              "assumptions": ["independent samples"], "assumptions_checked": "n/a",
                              "statistic": float(u), "p": float(p),
                              "effect": {"name": "rank-biserial", "value": rank_biserial(a, b, u)},
                              "n": [int(len(a)), int(len(b))]})
                pv.append(p)
            elif types.get(o) in ("nominal", "boolean") and df[o].nunique(dropna=True) <= 30:
                tab = pd.crosstab(miss, df[o].astype(str))
                if tab.shape[0] < 2 or tab.shape[1] < 2:
                    continue
                chi2, p, v, minexp = cramers_v(tab.values)
                if chi2 is None:
                    continue
                ok = minexp is not None and minexp >= 5
                rec = {"missing_in": c, "against": o, "test": "chi-square",
                       "assumptions": ["expected counts >= 5"],
                       "assumptions_checked": "passed" if ok else "violated",
                       "min_expected": minexp, "statistic": chi2, "p": p,
                       "effect": {"name": "cramers_v", "value": v}, "n": int(tab.values.sum())}
                if not ok:
                    rec.update(robust_chi2(tab.values, seed))
                tests.append(rec)
                pv.append(rec["p"])
    adj = bh(pv)
    for t, a in zip(tests, adj):
        t["p_adj"] = a
        t["correction"] = "BH across the missingness family"
    ev["missingness_dependence"] = sorted([t for t in tests if t["p_adj"] < ALPHA],
                                          key=lambda t: t["p_adj"])[:40]
    ev["missingness_family_size"] = len(tests)

    # type coherence and ranges
    for c in df.columns:
        s = df[c]
        if is_texty(s):
            nonnull = s.dropna().astype(str)
            if not len(nonnull):
                continue
            num = pd.to_numeric(nonnull, errors="coerce")
            if num.notna().mean() > 0.9 and types.get(c) != "numeric":
                ev["type_coherence"].append({"column": c, "issue": "numeric stored as text",
                                             "parseable_share": float(num.notna().mean())})
            dt = pd.to_datetime(nonnull, errors="coerce", format="mixed")
            # a date falls in a plausible year and names more than one calendar day: a code such
            # as `T09` parses as 9 January of year 1, a time of day as today
            good = dt.dropna()
            if dt.notna().mean() > 0.9 and good.dt.year.between(1000, 2999).mean() > 0.9 \
                    and good.dt.normalize().nunique() > 1 and types.get(c) not in ("datetime",):
                ev["type_coherence"].append({"column": c, "issue": "date stored as text",
                                             "parseable_share": float(dt.notna().mean())})
            stripped = nonnull.str.strip().str.lower()
            collapsed = int(nonnull.nunique() - stripped.nunique())
            if collapsed > 0:
                ev["category_hygiene"].append({"column": c, "issue": "case/whitespace variants",
                                               "levels_collapsed": collapsed,
                                               "levels": int(nonnull.nunique())})
        elif pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce")
            name = c.lower()
            if (name.endswith("_pct") or "percent" in name) and (v > 100).any():
                ev["range_anomalies"].append({"column": c, "issue": "percentage above 100",
                                              "n": int((v > 100).sum())})
            if any(k in name for k in ("qty", "count", "quantity", "n_", "days", "age", "price",
                                       "amount", "weight")) and (v < 0).any():
                ev["range_anomalies"].append({"column": c, "issue": "negative value in a non-negative quantity",
                                              "n": int((v < 0).sum()), "min": float(v.min())})
    return ev


# --------------------------------------------------------------------------- module: distributions
def mod_distributions(df, ctx, seed):
    from scipy import stats as st
    from sklearn.ensemble import IsolationForest
    ev = {"numeric": [], "nominal": [], "transform_suggestions": []}
    types = {t["column"]: t["type"] for t in ctx["typing"]}
    roles = {t["column"]: t["role"] for t in ctx["typing"]}
    for c in df.columns:
        if roles.get(c) in EXCLUDED_ROLES:
            continue
        s = df[c]
        if types.get(c) == "numeric" and pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna().values
            if v.size < 5:
                continue
            q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
            iqr = q[4] - q[2]
            lo, hi = q[2] - 1.5 * iqr, q[4] + 1.5 * iqr
            iqr_out = np.where((v < lo) | (v > hi))[0]
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            mad_z = np.abs(v - med) / (1.4826 * mad) if mad > 0 else np.zeros_like(v)
            mad_out = np.where(mad_z > 3.5)[0]
            try:
                iso = IsolationForest(random_state=seed, contamination="auto").fit(v.reshape(-1, 1))
                iso_out = np.where(iso.predict(v.reshape(-1, 1)) == -1)[0]
            except Exception:
                iso_out = np.array([], dtype=int)
            agree = len(set(iqr_out) & set(mad_out) & set(iso_out))
            test, p = normality(v, seed)
            modes = kde_modes(v, seed)
            rec = {"column": c, "n": int(v.size), "mean": float(v.mean()), "std": float(v.std(ddof=1))
                   if v.size > 1 else 0.0, "min": float(v.min()), "max": float(v.max()),
                   "quantiles": {"p1": q[0], "p5": q[1], "p25": q[2], "p50": q[3], "p75": q[4],
                                 "p95": q[5], "p99": q[6]},
                   "skew": float(st.skew(v)), "excess_kurtosis": float(st.kurtosis(v)),
                   "zero_share": float((v == 0).mean()),
                   "normality": {"test": test, "p": p, "assumptions": ["i.i.d. sample"],
                                 "assumptions_checked": "n/a"},
                   "modality": {"modes": modes["n"], "peaks": modes["peaks"]},
                   "outliers": {"iqr": int(iqr_out.size), "mad": int(mad_out.size),
                                "isolation_forest": int(iso_out.size), "agreement_all_three": int(agree),
                                "fences": [float(lo), float(hi)],
                                "max_abs_z_mad": float(np.max(mad_z)) if mad > 0 else None}}
            ev["numeric"].append(rec)
            if abs(rec["skew"]) > 1:
                sug = {"column": c, "reason": f"skew {rec['skew']:.2f}",
                       "kind": "log1p" if v.min() >= 0 else "yeo-johnson"}
                if v.min() > 0:
                    try:
                        _, lam = st.boxcox(v)
                        sug["boxcox_lambda"] = float(lam)
                        sug["kind"] = "box-cox"
                    except Exception:
                        pass
                    tv = np.log1p(v - v.min())
                    sug["skew_after"] = float(st.skew(tv))
                ev["transform_suggestions"].append(sug)
        elif types.get(c) in ("nominal", "boolean") or is_texty(s):
            vc = s.dropna().astype(str).value_counts()
            if vc.empty:
                continue
            pvec = (vc / vc.sum()).values
            ent = float(-(pvec * np.log2(pvec)).sum())
            ev["nominal"].append({"column": c, "levels": int(vc.size),
                                  "entropy_bits": ent,
                                  "max_entropy_bits": float(math.log2(vc.size)) if vc.size > 1 else 0.0,
                                  "top": [[str(k), int(v)] for k, v in vc.head(8).items()],
                                  "long_tail_share": float(vc[vc < max(2, 0.01 * vc.sum())].sum() / vc.sum())})
    return ev


def kde_modes(v, seed, grid=256):
    from scipy import stats as st
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 30 or np.allclose(v, v[0]):
        return {"n": 1, "peaks": []}
    if v.size > 5000:
        rng = np.random.default_rng(seed)
        v = v[rng.choice(v.size, 5000, replace=False)]
    try:
        kde = st.gaussian_kde(v)
        xs = np.linspace(v.min(), v.max(), grid)
        ys = kde(xs)
    except Exception:
        return {"n": 1, "peaks": []}
    peaks = []
    for i in range(1, grid - 1):
        if ys[i] > ys[i - 1] and ys[i] >= ys[i + 1] and ys[i] > 0.05 * ys.max():
            peaks.append(float(xs[i]))
    return {"n": max(1, len(peaks)), "peaks": peaks[:6]}


# --------------------------------------------------------------------------- module: relations
def label_rule_columns(ctx, lab, df):
    """The columns of the label's own rule (the body of the kept derivation whose head is the
    label — a refuted or set-aside one derives nothing, so it excludes nothing):
    no fit on the label sees them — importance's leakage probe refits with any the features
    still list, and the interaction screen leaves them out."""
    bodies = [d.get("body") or [] for d in ctx["derivations"]
              if d.get("column") == lab and d.get("kept", True) is not False]
    return [c for c in (bodies[-1] if bodies else []) if c in df.columns]


def mod_relations(df, ctx, seed):
    from scipy import stats as st
    ev = {"pairs": [], "partial": [], "vif": [], "interactions": [], "basis_numeric": []}
    types = {t["column"]: t["type"] for t in ctx["typing"]}
    basis = [c for c in ctx["basis"] if c in df.columns and types.get(c) == "numeric"
             and pd.api.types.is_numeric_dtype(df[c])]
    ev["basis_numeric"] = basis
    if len(basis) < 2:
        return ev
    sub = df[basis].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 10:
        return ev
    small = sub if len(sub) <= 2000 else sub.sample(2000, random_state=seed)
    pv = []
    for i, a in enumerate(basis):
        for b in basis[i + 1:]:
            rho, p = st.spearmanr(sub[a], sub[b])
            tau, tp = st.kendalltau(small[a], small[b])
            n = len(sub)
            z = 0.5 * math.log((1 + rho) / (1 - rho)) if abs(rho) < 1 else None
            ci = [float(math.tanh(z - 1.96 / math.sqrt(n - 3))),
                  float(math.tanh(z + 1.96 / math.sqrt(n - 3)))] if z is not None and n > 4 else None
            ev["pairs"].append({"a": a, "b": b, "spearman": float(rho), "ci": ci, "p": float(p),
                                "kendall": float(tau), "kendall_p": float(tp), "n": int(n),
                                "assumptions": ["paired observations"], "assumptions_checked": "n/a"})
            pv.append(p)
    for rec, a in zip(ev["pairs"], bh(pv)):
        rec["p_adj"] = a
        rec["correction"] = "BH across the basis pair family"
    # partial spearman: rank-transform, invert the correlation matrix
    try:
        R = sub.rank().corr().values
        P = np.linalg.pinv(R)
        d = np.sqrt(np.diag(P))
        Pc = -P / np.outer(d, d)
        for i, a in enumerate(basis):
            for j, b in enumerate(basis):
                if j <= i:
                    continue
                ev["partial"].append({"a": a, "b": b, "partial_spearman": float(Pc[i, j]),
                                      "controlling_for": [c for c in basis if c not in (a, b)]})
    except Exception as e:
        mark(f"WARN: partial correlations skipped ({e})")
    # VIF
    try:
        X = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
        X = X.dropna(axis=1, how="all").fillna(0.0)
        C = np.linalg.pinv(X.corr().values)
        for c, v in zip(X.columns, np.diag(C)):
            ev["vif"].append({"column": c, "vif": float(v)})
    except Exception as e:
        mark(f"WARN: VIF skipped ({e})")
    # interaction screen on the label
    lab = ctx["partition"]["label"]
    if lab and lab in df.columns and len(basis) >= 2:
        y = pd.to_numeric(df[lab], errors="coerce")
        if y.notna().sum() > 30 and y.nunique() > 1:
            # never the leakage set, nor the label's own rule columns (a product of the
            # columns the label is computed from screens the definition, not an interaction)
            ruled = set(ctx["partition"]["leakage"] or []) | set(label_rule_columns(ctx, lab, df))
            feats = [c for c in basis if c not in ruled and c != lab][:8]
            rows = df[feats + [lab]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(rows) > 30:
                yy = rows[lab].values
                base_ss = float(((yy - yy.mean()) ** 2).sum())
                for i, a in enumerate(feats):
                    for b in feats[i + 1:]:
                        X0 = np.column_stack([np.ones(len(rows)), rows[a].values, rows[b].values])
                        X1 = np.column_stack([X0, rows[a].values * rows[b].values])
                        r0 = float(((yy - X0 @ np.linalg.lstsq(X0, yy, rcond=None)[0]) ** 2).sum())
                        r1 = float(((yy - X1 @ np.linalg.lstsq(X1, yy, rcond=None)[0]) ** 2).sum())
                        if base_ss <= 0:
                            continue
                        dr2 = (r0 - r1) / base_ss
                        f = ((r0 - r1) / 1) / (r1 / max(1, len(rows) - 4)) if r1 > 0 else None
                        p = float(1 - st.f.cdf(f, 1, len(rows) - 4)) if f and f > 0 else None
                        ev["interactions"].append({"a": a, "b": b, "delta_r2": float(dr2),
                                                   "f": f, "p": p,
                                                   "assumptions": ["linear model fits the main effects"],
                                                   "assumptions_checked": "n/a"})
                pvi = [x["p"] for x in ev["interactions"]]
                for rec, a in zip(ev["interactions"], bh(pvi)):
                    rec["p_adj"] = a
                    rec["correction"] = "BH across the interaction screen"
                ev["interactions"] = sorted(ev["interactions"],
                                            key=lambda r: -(r["delta_r2"] or 0))[:20]
    return ev


# --------------------------------------------------------------------------- module: inference
def games_howell(groups, names):
    """Pairwise Games-Howell (unequal variances, unequal n). Returns rows with p and Hedges g."""
    from scipy import stats as st
    out = []
    k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = np.asarray(groups[i], float), np.asarray(groups[j], float)
            na, nb = a.size, b.size
            if na < 2 or nb < 2:
                continue
            va, vb = a.var(ddof=1), b.var(ddof=1)
            se = math.sqrt(va / na + vb / nb)
            if se == 0:
                continue
            t = abs(a.mean() - b.mean()) / se
            dfree = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
            p = float(st.studentized_range.sf(t * math.sqrt(2), k, dfree))
            g, ci = cohen_d(a, b)
            out.append({"a": names[i], "b": names[j], "p": min(1.0, p), "hedges_g": g, "ci": ci,
                        "n": [int(na), int(nb)], "test": "games-howell"})
    return out


def dunn(groups, names):
    """Pairwise Dunn (rank sums, normal approximation) with tie correction."""
    from scipy import stats as st
    allv = np.concatenate([np.asarray(g, float) for g in groups])
    n = allv.size
    ranks = st.rankdata(allv)
    idx, out, means, sizes = 0, [], [], []
    for g in groups:
        m = len(g)
        means.append(ranks[idx:idx + m].mean())
        sizes.append(m)
        idx += m
    _, counts = np.unique(allv, return_counts=True)
    ties = np.sum(counts ** 3 - counts)
    sigma2 = (n * (n + 1) / 12.0) - ties / (12.0 * (n - 1)) if n > 1 else 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            se = math.sqrt(sigma2 * (1 / sizes[i] + 1 / sizes[j])) if sigma2 > 0 else 0
            if se == 0:
                continue
            z = (means[i] - means[j]) / se
            p = float(2 * (1 - st.norm.cdf(abs(z))))
            a, b = np.asarray(groups[i], float), np.asarray(groups[j], float)
            u = st.mannwhitneyu(a, b, alternative="two-sided").statistic
            out.append({"a": names[i], "b": names[j], "z": float(z), "p": p,
                        "rank_biserial": rank_biserial(a, b, u),
                        "n": [int(sizes[i]), int(sizes[j])], "test": "dunn"})
    return out


def mod_inference(df, ctx, seed):
    from scipy import stats as st
    ev = {"numeric_by_nominal": [], "nominal_by_nominal": [], "label": None, "family_sizes": {}}
    types = {t["column"]: t["type"] for t in ctx["typing"]}
    roles = {t["column"]: t["role"] for t in ctx["typing"]}
    usable = [c for c in df.columns if roles.get(c) not in EXCLUDED_ROLES]
    noms = [c for c in usable if types.get(c) in ("nominal", "boolean")
            and 2 <= df[c].nunique(dropna=True) <= 12]
    nums = [c for c in usable if types.get(c) == "numeric" and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique(dropna=True) > 5]
    pv = []
    for g in noms:
        for y in nums:
            sub = df[[g, y]].dropna()
            if len(sub) < 20:
                continue
            groups, names = [], []
            for lvl, part in sub.groupby(sub[g].astype(str)):
                v = pd.to_numeric(part[y], errors="coerce").dropna().values
                if v.size >= 5:
                    groups.append(v)
                    names.append(str(lvl))
            if len(groups) < 2:
                continue
            norm_ok = all(normality(v, seed)[1] is None or normality(v, seed)[1] > ALPHA or v.size >= 30
                          for v in groups)
            try:
                lev_p = float(st.levene(*groups, center="median").pvalue)
            except Exception:
                lev_p = None
            small_n = any(len(v) < 30 for v in groups)
            checked = "passed" if norm_ok else "violated"
            rec = {"group": g, "value": y, "k": len(groups), "levels": names,
                   "n": int(sum(len(v) for v in groups)),
                   "assumptions": ["approximate normality within groups or n >= 30 per group",
                                   "unequal variances allowed (Welch)"],
                   "assumptions_checked": checked, "levene_p": lev_p,
                   "small_groups": bool(small_n)}
            if norm_ok:
                if len(groups) == 2:
                    t, p = st.ttest_ind(groups[0], groups[1], equal_var=False)
                    g_, ci = cohen_d(groups[0], groups[1])
                    rec.update({"test": "welch-t", "statistic": float(t), "p": float(p),
                                "effect": {"name": "hedges_g", "value": g_, "ci": ci}})
                else:
                    f, p = st.f_oneway(*groups)     # classic F as the statistic
                    w = welch_anova(groups)
                    om, et = eta_squared(groups)
                    rec.update({"test": "welch-anova", "statistic": w["F"], "p": w["p"],
                                "classic_f": float(f), "classic_p": float(p),
                                "effect": {"name": "omega_squared", "value": om, "eta_squared": et},
                                "contrasts": games_howell(groups, names),
                                "contrast_correction": "studentized range (Games-Howell)"})
            else:
                if len(groups) == 2:
                    u, p = st.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                    rec.update({"test": "welch-t", "switched_to": "mann-whitney",
                                "statistic": float(u), "p": float(p),
                                "effect": {"name": "rank_biserial",
                                           "value": rank_biserial(groups[0], groups[1], u)}})
                else:
                    h, p = st.kruskal(*groups)
                    rec.update({"test": "welch-anova", "switched_to": "kruskal-wallis",
                                "statistic": float(h), "p": float(p),
                                "effect": {"name": "epsilon_squared",
                                           "value": epsilon_squared(h, sum(len(v) for v in groups), len(groups))},
                                "contrasts": dunn(groups, names),
                                "contrast_correction": "Holm"})
            cs = rec.get("contrasts") or []
            if cs:
                for c_, adj in zip(cs, holm([c["p"] for c in cs])):
                    c_["p_adj"] = adj
                rec["contrasts"] = sorted(cs, key=lambda c: c["p_adj"])[:12]
            ev["numeric_by_nominal"].append(rec)
            pv.append(rec["p"])
    for rec, a in zip(ev["numeric_by_nominal"], bh(pv)):
        rec["p_adj"] = a
        rec["correction"] = "BH across the numeric-by-nominal family"
    ev["family_sizes"]["numeric_by_nominal"] = len(pv)

    pv2 = []
    for i, a in enumerate(noms):
        for b in noms[i + 1:]:
            tab = pd.crosstab(df[a].astype(str), df[b].astype(str))
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            chi2, p, v, minexp = cramers_v(tab.values)
            if chi2 is None:
                continue
            ok = minexp >= 5
            rec = {"a": a, "b": b, "test": "chi-square", "statistic": chi2, "p": p,
                   "effect": {"name": "cramers_v", "value": v}, "min_expected": minexp,
                   "assumptions": ["expected counts >= 5"],
                   "assumptions_checked": "passed" if ok else "violated",
                   "n": int(tab.values.sum()), "shape": list(tab.shape)}
            if not ok:
                rec.update(robust_chi2(tab.values, seed))
            ev["nominal_by_nominal"].append(rec)
            pv2.append(rec["p"])
    for rec, a in zip(ev["nominal_by_nominal"], bh(pv2)):
        rec["p_adj"] = a
        rec["correction"] = "BH across the nominal-by-nominal family"
    ev["family_sizes"]["nominal_by_nominal"] = len(pv2)

    lab = ctx["partition"]["label"]
    if lab and lab in df.columns:
        s = df[lab].dropna()
        vc = s.value_counts(normalize=True)
        ev["label"] = {"column": lab, "distinct": int(s.nunique()),
                       "shares": {str(k): float(v) for k, v in vc.head(12).items()},
                       "minority_share": float(vc.min()) if len(vc) > 1 else None,
                       "baseline_accuracy": float(vc.max()) if len(vc) else None}
    ev["numeric_by_nominal"] = sorted(ev["numeric_by_nominal"], key=lambda r: r.get("p_adj", 1))[:60]
    ev["nominal_by_nominal"] = sorted(ev["nominal_by_nominal"], key=lambda r: r.get("p_adj", 1))[:60]
    return ev


def welch_anova(groups):
    from scipy import stats as st
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([np.mean(g) for g in groups], float)
    v = np.array([np.var(g, ddof=1) for g in groups], float)
    w = np.where(v > 0, n / np.where(v > 0, v, 1), 0.0)
    if w.sum() <= 0:
        return {"F": None, "p": None}
    mw = (w * m).sum() / w.sum()
    num = ((w * (m - mw) ** 2).sum()) / (k - 1)
    lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    den = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = num / den
    df2 = (k ** 2 - 1) / (3 * lam) if lam > 0 else np.inf
    p = float(st.f.sf(F, k - 1, df2))
    return {"F": float(F), "p": p, "df2": float(df2)}


# --------------------------------------------------------------------------- module: segments
def mod_segments(df, ctx, seed):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.preprocessing import StandardScaler
    ev = {"scans": [], "chosen_k": None, "profiles": [], "features": []}
    types = {t["column"]: t["type"] for t in ctx["typing"]}
    feats = [c for c in ctx["basis"] if c in df.columns and types.get(c) == "numeric"
             and pd.api.types.is_numeric_dtype(df[c])]
    ev["features"] = feats
    if len(feats) < 2:
        return ev
    X = df[feats].apply(pd.to_numeric, errors="coerce").dropna()
    if len(X) < 30:
        return ev
    idx = X.index
    scaler = StandardScaler().fit(X.values)
    Xs = scaler.transform(X.values)
    if len(Xs) > 5000:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(Xs), 5000, replace=False)
        Xs_fit, idx = Xs[sel], idx[sel]
    else:
        Xs_fit = Xs
    best = None
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs_fit)
        try:
            sil = float(silhouette_score(Xs_fit, km.labels_, random_state=seed,
                                         sample_size=min(2000, len(Xs_fit))))
        except Exception:
            sil = None
        aris = []
        rng = np.random.default_rng(seed + k)
        for r in range(10):
            sub = rng.choice(len(Xs_fit), int(0.8 * len(Xs_fit)), replace=False)
            km2 = KMeans(n_clusters=k, n_init=5, random_state=seed + r).fit(Xs_fit[sub])
            aris.append(float(adjusted_rand_score(km.labels_[sub], km2.labels_)))
        try:
            ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xs_fit)
            cross = float(adjusted_rand_score(km.labels_, ag.labels_))
        except Exception:
            cross = None
        rec = {"k": k, "silhouette": sil, "stability_ari_mean": float(np.mean(aris)),
               "stability_ari_sd": float(np.std(aris)), "agglomerative_agreement_ari": cross,
               "inertia": float(km.inertia_)}
        ev["scans"].append(rec)
        score = (sil or 0) * (0.5 + 0.5 * float(np.mean(aris)))
        if best is None or score > best[0]:
            best = (score, k, km)
    if best:
        _, k, km = best
        ev["chosen_k"] = k
        # The rule that places a row in a segment (analysis-contract §1): the k-means
        # prediction, the nearest centroid in the basis standardized by the whole data's mean
        # and scale. The profiles describe a segment; this assigns one, so /vega-eval can cut a
        # segment slice of its test rows from the raw columns (a profile alone cannot).
        ev["assignment"] = {
            "rule": "nearest centroid in the standardized basis (k-means)",
            "features": list(feats),
            "center": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
            "centroids": [[float(v) for v in row] for row in km.cluster_centers_],
            "segments": list(range(k))}
        lab = pd.Series(km.labels_, index=idx, name="_segment")
        joined = df.loc[idx].copy()
        joined["_segment"] = lab
        for s, part in joined.groupby("_segment"):
            prof = {"segment": int(s), "n": int(len(part)), "share": float(len(part) / len(joined)),
                    "medians": {c: float(pd.to_numeric(part[c], errors="coerce").median()) for c in feats},
                    "standardized_difference": {}}
            for c in feats:
                whole = pd.to_numeric(joined[c], errors="coerce")
                sd = whole.std(ddof=0)
                prof["standardized_difference"][c] = float(
                    (pd.to_numeric(part[c], errors="coerce").mean() - whole.mean()) / sd) if sd else None
            nomcols = [c for c in df.columns if types.get(c) in ("nominal", "boolean")
                       and df[c].nunique(dropna=True) <= 20][:5]
            prof["dominant"] = {c: str(part[c].astype(str).value_counts().index[0])
                                for c in nomcols if part[c].notna().any()}
            lab_col = ctx["partition"]["label"]
            if lab_col and lab_col in part.columns:
                v = pd.to_numeric(part[lab_col], errors="coerce")
                prof["label_rate"] = float(v.mean()) if v.notna().any() else None
            ev["profiles"].append(prof)
    return ev


# --------------------------------------------------------------------------- module: importance
def cv_interval(scores, n_rows):
    """The interval of a k-fold CV mean: the corrected resampled t (Nadeau & Bengio 2003) — the
    folds share their training rows, so the naive sd/√k is too narrow; the variance is
    s²·(1/k + n_test/n_train) with s the sample sd (ddof 1), and the quantile t with k−1 degrees
    of freedom, never the normal 1.96 (five folds are not a large sample)."""
    from scipy import stats as st
    k = len(scores)
    if k < 2:
        return [float(scores[0]), float(scores[0])] if k else [None, None]
    m, s = float(np.mean(scores)), float(np.std(scores, ddof=1))
    n_test = n_rows / k
    half = float(st.t.ppf(0.975, k - 1)) * s * math.sqrt(1 / k + n_test / max(1.0, n_rows - n_test))
    return [m - half, m + half]


def repeat_interval(mean, sd, reps):
    """A t interval (reps−1 df, sample sd) over the repeats of one permutation importance: the
    repeats' own spread, which leaves out the split's."""
    from scipy import stats as st
    if reps < 2:
        return [float(mean), float(mean)]
    half = float(st.t.ppf(0.975, reps - 1)) * float(sd) * math.sqrt(reps / (reps - 1)) / math.sqrt(reps)
    return [float(mean - half), float(mean + half)]


def mod_importance(df, ctx, seed):
    from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    lab = ctx["partition"]["label"]
    types = {t["column"]: t["type"] for t in ctx["typing"]}
    leak = list(ctx["partition"]["leakage"] or [])
    label_derivs = label_rule_columns(ctx, lab, df)
    feats = [c for c in (ctx["partition"]["features"] or []) if c in df.columns]
    if not feats:
        feats = [c for c in ctx["basis"] if c in df.columns and c != lab]
    feats = [c for c in feats if c not in leak and c != lab and types.get(c) != "text"]
    # The label's own rule columns never reach the fit (SKILL.md: importance does not see a
    # derivation of the label). Those the leakage set missed are the probe's subject below.
    slipped = [c for c in label_derivs if c in feats]
    feats = [c for c in feats if c not in slipped]
    ev = {"task": ctx["partition"]["task"], "label": lab, "features": feats,
          "excluded": {"leakage": leak, "label_derivations": label_derivs},
          "baseline": None, "metric": None, "models": [], "permutation_importance": [],
          "leakage_probe": {}, "learning_curve": []}
    if not lab or lab not in df.columns or not feats:
        return ev
    data = df[feats + slipped + [lab]].dropna(subset=[lab])
    # method-catalog.md: time-based folds when a datetime orders the rows — a shuffled fold, or a
    # contiguous block that is not the last, would score rows with their own future in the
    # training part; forward chaining (TimeSeriesSplit) trains on the past of each fold only
    tcol = (ctx.get("time") or {}).get("column")
    order = pd.to_datetime(df.loc[data.index, tcol], errors="coerce") if tcol in df.columns else None
    timed = order is not None and order.notna().mean() > 0.9
    if timed:
        data = data.loc[order.sort_values(kind="mergesort").index]
    if len(data) < 50:
        return ev
    y_raw = data[lab]
    classification = (y_raw.nunique() <= 20 and (types.get(lab) in ("nominal", "boolean")
                      or y_raw.nunique() <= 10))
    y = y_raw.astype(str) if classification else pd.to_numeric(y_raw, errors="coerce")
    keep = y.notna()
    data, y = data[keep], y[keep]
    def frame(cols):
        Xc = data[cols].copy()
        nm = [c for c in cols if pd.api.types.is_numeric_dtype(Xc[c]) and types.get(c) == "numeric"]
        for c in cols:
            if c not in nm:
                Xc[c] = Xc[c].astype(str)
        return Xc, nm, [c for c in cols if c not in nm]

    def make_models(num, cat):
        pre_lin = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=30,
                                                   sparse_output=False))]), cat)])
        pre_tree = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=30,
                                                   sparse_output=False))]), cat)])
        if classification:
            return [("logistic-l2", Pipeline([("pre", pre_lin),
                                              ("m", LogisticRegression(max_iter=2000, random_state=seed))])),
                    ("gradient-boosting", Pipeline([("pre", pre_tree),
                                                    ("m", HistGradientBoostingClassifier(random_state=seed))]))]
        return [("ridge", Pipeline([("pre", pre_lin), ("m", Ridge(random_state=seed))])),
                ("gradient-boosting", Pipeline([("pre", pre_tree),
                                                ("m", HistGradientBoostingRegressor(random_state=seed))]))]

    X, num, cat = frame(feats)
    if classification:
        ev["metric"] = "roc_auc" if y.nunique() == 2 else "accuracy"
        ev["baseline"] = {"kind": "majority",
                          "score": float(y.value_counts(normalize=True).max())
                          if ev["metric"] == "accuracy" else 0.5}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    else:
        ev["metric"] = "r2"
        ev["baseline"] = {"kind": "mean", "score": 0.0}
        cv = KFold(5, shuffle=True, random_state=seed)
    if timed:
        cv = TimeSeriesSplit(5)
    ev["folds"] = (f"forward chaining in time (by {tcol}): each fold trains on the rows before it" if timed else
                   "stratified, shuffled" if classification else "shuffled")
    ev["ci_method"] = ("corrected resampled t over the fold scores (Nadeau-Bengio, k-1 df); "
                       "permutation importances: a t interval over the repeats on one held-out split")
    models = make_models(num, cat)
    scoring = ev["metric"]
    best = None
    for name, pipe in models:
        try:
            sc = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        except Exception as e:
            mark(f"WARN: importance model {name} failed ({e})")
            continue
        rec = {"name": name, "cv_score": float(sc.mean()), "ci": cv_interval(sc, len(X)),
               "folds": [float(v) for v in sc]}
        ev["models"].append(rec)
        if best is None or rec["cv_score"] > best[0]:
            best = (rec["cv_score"], name, pipe)
    if best is None:
        return ev
    _, bname, bpipe = best
    strat = y if classification and not timed else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=None if timed else seed,
                                          stratify=strat, shuffle=not timed)
    bpipe.fit(Xtr, ytr)
    try:
        pi = permutation_importance(bpipe, Xte, yte, n_repeats=10, random_state=seed,
                                    scoring=scoring, n_jobs=1)
        for c, m_, s_ in zip(feats, pi.importances_mean, pi.importances_std):
            ev["permutation_importance"].append({
                "feature": c, "mean": float(m_),
                "ci": repeat_interval(m_, s_, 10),
                "model": bname})
        ev["permutation_importance"].sort(key=lambda r: -r["mean"])
    except Exception as e:
        mark(f"WARN: permutation importance skipped ({e})")
    # leakage probe: the label's rule columns the partition still listed as features (the
    # leakage set missed them) were kept out of every fit above; the probe refits the best
    # model WITH them, and the jump is what the definition of the label would have carried.
    probe = {"label_derivations_in_features": slipped, "suspect": [],
             "note": "no column of the label's rule was among the features once the leakage set "
                     "was removed"}
    if slipped:
        try:
            Xp, nump, catp = frame(feats + slipped)
            pipe_p = dict(make_models(nump, catp))[bname]
            sc2 = cross_val_score(pipe_p, Xp, y, cv=cv, scoring=scoring, n_jobs=1)
            probe.update({"score_without": float(best[0]), "score_with": float(sc2.mean()),
                          "delta": float(sc2.mean() - best[0]), "suspect": slipped,
                          "note": "the partition listed the label's own rule columns as features; "
                                  "they were kept out of the fit, and the delta is what they "
                                  "would have carried"})
        except Exception as e:
            probe["note"] = f"probe failed: {e}"
    near_perfect = best[0] > 0.99 if scoring in ("roc_auc", "accuracy", "r2") else False
    if near_perfect:
        probe["warning"] = "near-perfect cross-validated score: suspect leakage before celebrating"
    ev["leakage_probe"] = probe
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        n = max(30, int(frac * len(Xtr)))
        if n > len(Xtr):
            n = len(Xtr)
        try:
            sub = Xtr.iloc[:n]
            suby = ytr.iloc[:n]
            if classification and suby.nunique() < 2:
                continue
            p2 = bpipe.fit(sub, suby)
            from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
            if scoring == "roc_auc":
                s = float(roc_auc_score((yte == sorted(y.unique())[-1]).astype(int),
                                        p2.predict_proba(Xte)[:, -1]))
            elif scoring == "accuracy":
                s = float(accuracy_score(yte, p2.predict(Xte)))
            else:
                s = float(r2_score(yte, p2.predict(Xte)))
            ev["learning_curve"].append({"fraction": frac, "n": int(n), "score": s})
        except Exception:
            continue
    return ev


# --------------------------------------------------------------------------- module: time_series
def mod_time_series(df, ctx, seed, force_column=None):
    ev = {"column": None, "regularity": {}, "trend": {}, "seasonality": {}, "acf": [], "pacf": [],
          "stationarity": {}, "change_points": [], "series": []}
    t = ctx.get("time")
    if not t:
        return ev
    col = t["column"]
    ev["column"] = col
    s = pd.to_datetime(df[col], errors="coerce")
    ok = s.notna()
    ev["regularity"] = {"resolution": t.get("resolution"), "span": t.get("span"),
                        "regular": t.get("regular"), "parsed": int(ok.sum()),
                        "unparsed": int((~ok).sum())}
    d = s[ok].drop_duplicates().sort_values().diff().dropna()
    if len(d):
        modal = d.mode().iloc[0]
        gaps = d[d > modal * 1.5]
        ev["regularity"].update({"modal_interval": str(modal), "gaps": int(len(gaps)),
                                 "largest_gap": str(d.max()),
                                 "gap_share": float(len(gaps) / len(d))})
    # value series: the label if numeric, else the first basis numeric
    types = {x["column"]: x["type"] for x in ctx["typing"]}
    lab = ctx["partition"]["label"]
    def idish(c):
        n = c.lower()
        return n == "id" or n.endswith("_id") or n.startswith("id_")
    cands = ([lab] if lab and types.get(lab) == "numeric" else []) + \
            [c for c in ctx["basis"] if types.get(c) == "numeric" and c in df.columns and not idish(c)]
    cands = list(dict.fromkeys(cands))
    if not cands:
        return ev
    # Which column is the series? The one whose values remember the previous instant —
    # lag-1 autocorrelation over the time order is exactly that question, and picking by
    # column order instead would hand the module whatever happens to come first.
    scored = []
    for c in cands:
        v = (pd.DataFrame({"t": s, "v": pd.to_numeric(df[c], errors="coerce")})
             .dropna().sort_values("t")["v"].values)
        if v.size < 20 or np.allclose(v, v[0]):
            continue
        r1 = float(np.corrcoef(v[:-1], v[1:])[0, 1]) if v.size > 2 else 0.0
        scored.append({"column": c, "lag1_autocorrelation": r1 if np.isfinite(r1) else 0.0})
    if not scored:
        return ev
    scored.sort(key=lambda r: -abs(r["lag1_autocorrelation"]))
    ev["series_candidates"] = scored[:8]
    value = force_column if force_column in cands else scored[0]["column"]
    ev["value_column"] = value
    ev["value_chosen_because"] = ("named with --series-column" if force_column == value else
                                  "strongest lag-1 autocorrelation among the numeric basis")
    ser = (pd.DataFrame({"t": s, "v": pd.to_numeric(df[value], errors="coerce")})
           .dropna().sort_values("t").set_index("t")["v"])
    if len(ser) < 20:
        return ev
    # Resample at the data's own modal interval, never at a coarser one: aggregating hourly
    # readings to days would average the daily cycle out of existence before STL sees it.
    step = ev["regularity"].get("modal_interval")
    try:
        freq = pd.Timedelta(step) if step else None
    except Exception:
        freq = None
    if ser.index.duplicated().any() or not ev["regularity"].get("regular"):
        if freq is not None and freq > pd.Timedelta(0):
            ser = ser.resample(freq).mean().interpolate("linear")
        else:
            ser = ser.groupby(level=0).mean()
    daily = ser
    ev["resampled_to"] = str(freq) if freq is not None else "none (already one value per instant)"
    x = np.arange(len(daily), dtype=float)
    _tsf = theil_sen(x, daily.values)
    # The period is looked for in the residual: on a trending series the autocorrelation
    # decays monotonically and any "peak" in it is the trend, not a season.
    detrended = (daily.values - (_tsf["slope"] * x + _tsf["intercept"])) if _tsf else daily.values
    _mk = mann_kendall(daily.values, detrended)
    ev["trend"] = {"theil_sen": _tsf, "mann_kendall": _mk,
                   "assumptions": ["monotone trend", "serially independent residuals"],
                   "assumptions_checked": _mk["assumptions_checked"] if _mk else "n/a"}
    try:
        from statsmodels.tsa.seasonal import STL
        from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf, adfuller, kpss
        n = len(daily)
        peak, ac = _acf_peak(detrended, n)
        if peak:
            period = peak
        else:
            ev["seasonality"] = {"period": None, "method": "ACF peak search",
                                 "note": "no local autocorrelation peak above 0.15: no periodic "
                                         "component this file can support"}
            period = None
        if period and n >= 2 * period + 1:
            stl = STL(daily, period=period, robust=True).fit()
            var_r = float(np.var(stl.resid))
            var_sr = float(np.var(stl.seasonal + stl.resid))
            var_tr = float(np.var(stl.trend + stl.resid))
            ev["seasonality"] = {"period": int(period),
                                 "seasonal_strength": float(max(0, 1 - var_r / var_sr)) if var_sr > 0 else None,
                                 "trend_strength": float(max(0, 1 - var_r / var_tr)) if var_tr > 0 else None,
                                 "method": "STL (robust)"}
            ev["seasonality"].update(_period_stability(detrended, int(period)))
        ev["acf"] = [float(v) for v in ac[:25]]
        ev["acf_basis"] = "residual after the Theil-Sen trend"
        ev["pacf"] = [float(v) for v in sm_pacf(detrended, nlags=min(20, n // 3))]
        adf = adfuller(daily.values, autolag="AIC")
        kp = kpss(daily.values, regression="c", nlags="auto")
        ev["stationarity"] = {
            "adf": {"statistic": float(adf[0]), "p": float(adf[1])},
            "kpss": {"statistic": float(kp[0]), "p": float(kp[1])},
            "reading": four_way(adf[1], kp[1]),
            "assumptions": ["the two tests answer opposite nulls; read them together"],
            "assumptions_checked": "n/a"}
    except Exception as e:
        mark(f"WARN: time-series decomposition partially skipped ({e})")
    # Level shifts are read on the residual after the monotone trend is removed: a straight
    # ramp has no change point, and segmenting it would invent four.
    y = daily.values.astype(float)
    ts_fit = ev["trend"].get("theil_sen") if isinstance(ev["trend"], dict) else None
    if ts_fit:
        xr = np.arange(len(y), dtype=float)
        resid = y - (ts_fit["slope"] * xr + ts_fit["intercept"])
        ev["change_points"] = binary_segmentation(resid, max_cp=4)
        ev["change_point_basis"] = "residual after the Theil-Sen trend"
    else:
        ev["change_points"] = binary_segmentation(y, max_cp=4)
        ev["change_point_basis"] = "raw series (no trend fitted)"
    for cp in ev["change_points"]:
        i = min(len(daily.index) - 1, max(0, cp["index"]))
        cp["at"] = str(daily.index[i])
    step = max(1, len(daily) // 400)
    ev["series"] = [[str(i.date()) if hasattr(i, "date") else str(i), float(v)]
                    for i, v in list(zip(daily.index, daily.values))[::step]]
    return ev


def four_way(adf_p, kpss_p):
    a = adf_p < 0.05          # ADF rejects unit root -> stationary
    k = kpss_p < 0.05         # KPSS rejects stationarity
    if a and not k:
        return "stationary (both tests agree)"
    if not a and k:
        return "non-stationary (both tests agree): difference before modelling"
    if a and k:
        return "conflicting: likely difference-stationary around a trend"
    return "conflicting: neither test is decisive; the series may be short or weakly dependent"


def binary_segmentation(y, max_cp=4, min_size=10):
    """Mean-shift change points by binary segmentation with a BIC stop. No extra dependency."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 4 * min_size:
        return []
    def cost(a, b):
        seg = y[a:b]
        return float(seg.size * np.var(seg)) if seg.size else 0.0
    segs = [(0, n)]
    cps = []
    total = cost(0, n)
    for _ in range(max_cp):
        best = None
        for (a, b) in segs:
            if b - a < 2 * min_size:
                continue
            base = cost(a, b)
            for t in range(a + min_size, b - min_size):
                g = base - (cost(a, t) + cost(t, b))
                if best is None or g > best[0]:
                    best = (g, a, t, b)
        if best is None:
            break
        gain, a, t, b = best
        pen = math.log(n) * np.var(y) if np.var(y) > 0 else 0
        if gain <= pen:
            break
        segs = [s for s in segs if s != (a, b)] + [(a, t), (t, b)]
        left, right = y[a:t], y[t:b]
        cps.append({"index": int(t), "mean_before": float(left.mean()), "mean_after": float(right.mean()),
                    "shift": float(right.mean() - left.mean()), "gain": float(gain)})
    return sorted(cps, key=lambda c: c["index"])


# --------------------------------------------------------------------------- module: spatial
def mod_spatial(df, ctx, seed):
    ev = {"crs": None, "columns": [], "sanity": {}, "morans_i": [], "clusters": {},
          "nearest_neighbour": {}, "points": []}
    sp = ctx.get("spatial")
    if not sp or not sp.get("columns"):
        return ev
    c0 = sp["columns"][0]
    ev["crs"] = sp.get("crs")
    ev["columns"] = sp["columns"]
    if c0.get("kind") != "latlon":
        ev["sanity"] = {"note": f"geometry kind {c0.get('kind')} is not parsed here; "
                                "only latlon pairs are measured without geopandas"}
        return ev
    lat = pd.to_numeric(df[c0["column"]], errors="coerce")
    lon = pd.to_numeric(df[c0["pair"]], errors="coerce")
    ok = lat.notna() & lon.notna()
    ev["sanity"] = {"n": int(ok.sum()),
                    "lat_out_of_range": int((~lat.between(-90, 90) & lat.notna()).sum()),
                    "lon_out_of_range": int((~lon.between(-180, 180) & lon.notna()).sum()),
                    "axis_order_suspect": bool(lat.abs().max() > 90 >= lon.abs().max()),
                    "duplicate_locations": int(pd.DataFrame({"a": lat, "b": lon})[ok].duplicated().sum())}
    if ok.sum() < 20:
        return ev
    # local metric projection (equirectangular around the centroid) — metres, no pyproj needed
    la, lo = lat[ok].values, lon[ok].values
    lat0 = float(np.mean(la))
    xm = np.radians(lo - float(np.mean(lo))) * 6371000.0 * math.cos(math.radians(lat0))
    ym = np.radians(la - lat0) * 6371000.0
    P = np.column_stack([xm, ym])
    ev["projection"] = {"kind": "equirectangular about the centroid", "units": "metres",
                        "lat0": lat0, "note": "local approximation; declare a CRS for anything exact"}
    from sklearn.neighbors import NearestNeighbors
    k = min(8, len(P) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(P)
    dist, idx = nn.kneighbors(P)
    ev["nearest_neighbour"] = {"k": k, "median_distance_m": float(np.median(dist[:, 1])),
                               "p95_distance_m": float(np.percentile(dist[:, 1], 95))}
    types = {t["column"]: t["type"] for t in ctx["typing"]}
    nums = [c for c in ctx["basis"] if types.get(c) == "numeric" and c in df.columns][:8]
    rng = np.random.default_rng(seed)
    for c in nums:
        v = pd.to_numeric(df[c], errors="coerce")[ok].values
        m = np.isfinite(v)
        if m.sum() < 20:
            continue
        z = v - np.nanmean(v)
        z[~m] = 0.0
        neigh = idx[:, 1:]
        num = float(np.sum(z[:, None] * z[neigh]))
        den = float(np.sum(z ** 2))
        n = len(z)
        W = n * k
        I = (n / W) * (num / den) if den > 0 else None
        if I is None:
            continue
        def geary(zz):
            return (n - 1) / (2 * W) * float(np.sum((zz[:, None] - zz[neigh]) ** 2)) / den
        C = geary(z)
        perms, gperms = [], []
        for _ in range(999):
            zp = rng.permutation(z)
            perms.append((n / W) * float(np.sum(zp[:, None] * zp[neigh])) / den)
            gperms.append(geary(zp))
        p = float((1 + np.sum(np.abs(np.array(perms)) >= abs(I))) / (1 + len(perms)))
        gp = float((1 + np.sum(np.abs(np.array(gperms) - 1) >= abs(C - 1))) / (1 + len(gperms)))
        ev["morans_i"].append({"column": c, "I": float(I), "expected_I": float(-1 / (n - 1)),
                               "pseudo_p": p, "permutations": len(perms), "k": k,
                               "gearys_c": {"C": float(C), "expected_C": 1.0, "pseudo_p": gp},
                               "assumptions": ["projected coordinates (metres)"],
                               "check": "the latitude/longitude pair projected to local metres "
                                        "(`projection`); another geometry kind is not measured",
                               "assumptions_checked": "passed"})
    from sklearn.cluster import DBSCAN
    kd = np.sort(dist[:, -1])
    eps = float(np.percentile(kd, 90))
    db = DBSCAN(eps=eps, min_samples=max(4, k // 2)).fit(P)
    labels = db.labels_
    ev["clusters"] = {"algorithm": "DBSCAN", "eps_m": eps, "min_samples": int(max(4, k // 2)),
                      "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
                      "noise_share": float(np.mean(labels == -1))}
    step = max(1, len(P) // 1500)
    ev["points"] = [[float(a), float(b), int(c_)] for a, b, c_ in
                    zip(lo[::step], la[::step], labels[::step])]
    return ev


# --------------------------------------------------------------------------- module: drift
def mod_drift(df, ctx, seed, split_series):
    from scipy import stats as st
    ev = {"parts": [], "columns": [], "label_rate": {}, "split_column": ctx.get("split_column")}
    if split_series is None:
        return ev
    parts = [str(p) for p in pd.Series(split_series).dropna().unique()]
    if len(parts) < 2:
        return ev
    ev["parts"] = parts
    a_mask = pd.Series(split_series).astype(str) == parts[0]
    types = {t["column"]: t["type"] for t in ctx["typing"]}
    roles = {t["column"]: t["role"] for t in ctx["typing"]}
    pv = []
    for c in df.columns:
        if roles.get(c) in EXCLUDED_ROLES or c == ctx.get("split_column"):
            continue
        A, B = df.loc[a_mask.values, c], df.loc[~a_mask.values, c]
        if types.get(c) == "numeric" and pd.api.types.is_numeric_dtype(df[c]):
            a = pd.to_numeric(A, errors="coerce").dropna().values
            b = pd.to_numeric(B, errors="coerce").dropna().values
            if a.size < 10 or b.size < 10:
                continue
            ks = st.ks_2samp(a, b)
            rec = {"column": c, "measure": "ks+psi", "ks_d": float(ks.statistic), "p": float(ks.pvalue),
                   "psi": psi(a, b), "wasserstein": float(st.wasserstein_distance(a, b)),
                   "n": [int(a.size), int(b.size)],
                   "assumptions": [], "assumptions_checked": "n/a"}
        else:
            tab = pd.crosstab(a_mask.values, df[c].astype(str))
            if tab.shape[1] < 2:
                continue
            chi2, p, v, minexp = cramers_v(tab.values)
            if chi2 is None:
                continue
            rec = {"column": c, "measure": "chi2+psi", "statistic": chi2, "p": p,
                   "cramers_v": v, "psi": psi_categorical(A.astype(str), B.astype(str)),
                   "min_expected": minexp,
                   "assumptions": ["expected counts >= 5"],
                   "assumptions_checked": "passed" if (minexp or 0) >= 5 else "violated"}
            if rec["assumptions_checked"] == "violated":
                rec.update(robust_chi2(tab.values, seed))
        rec["drifted"] = bool((rec.get("psi") or 0) > 0.25)
        ev["columns"].append(rec)
        pv.append(rec["p"])
    for rec, adj in zip(ev["columns"], bh(pv)):
        rec["p_adj"] = adj
        rec["correction"] = "BH across the drift family"
    lab = ctx["partition"]["label"]
    if lab and lab in df.columns:
        ya = pd.to_numeric(df.loc[a_mask.values, lab], errors="coerce").dropna()
        yb = pd.to_numeric(df.loc[~a_mask.values, lab], errors="coerce").dropna()
        if len(ya) > 10 and len(yb) > 10 and set(pd.unique(df[lab].dropna())) <= {0, 1, True, False}:
            na, nb = len(ya), len(yb)
            xa, xb = int(ya.astype(float).sum()), int(yb.astype(float).sum())
            pa, pb = xa / na, xb / nb
            ok = min(xa, na - xa, xb, nb - xb) >= 5
            se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
            pool = (xa + xb) / (na + nb)
            se0 = math.sqrt(pool * (1 - pool) * (1 / na + 1 / nb))
            z = (pa - pb) / se0 if se0 > 0 else 0.0
            rec = {parts[0]: pa, parts[1]: pb, "risk_difference": pa - pb,
                   "ci": [pa - pb - 1.96 * se, pa - pb + 1.96 * se], "test": "two-proportion z",
                   "z": float(z), "p": float(2 * (1 - st.norm.cdf(abs(z)))),
                   "counts": {parts[0]: [xa, na], parts[1]: [xb, nb]},
                   "assumptions": ["n·p and n·(1−p) >= 5 in each part"],
                   "assumptions_checked": "passed" if ok else "violated"}
            if not ok:
                (la, ua), (lb, ub) = _wilson(xa, na), _wilson(xb, nb)
                d = pa - pb
                _, pf = st.fisher_exact([[xa, na - xa], [xb, nb - xb]])
                rec.update({"switched_to": "fisher-exact with the Newcombe interval", "p": float(pf),
                            "ci": [d - math.sqrt((pa - la) ** 2 + (ub - pb) ** 2),
                                   d + math.sqrt((ua - pa) ** 2 + (pb - lb) ** 2)]})
            ev["label_rate"] = rec
    return ev


# --------------------------------------------------------------------------- figures
PALETTE = ["#1b3a73", "#c8552f", "#2e8b57", "#8a6420", "#7a3c77", "#2c6570"]


def _fig_setup():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "svg.hashsalt": "data-lens",              # deterministic element ids
        "font.family": "sans-serif", "font.size": 9, "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 96, "savefig.transparent": True})
    import matplotlib.pyplot as plt
    return plt


def _save(plt, fig, out_dir, fid, figures, title, alt, kind):
    path = Path(out_dir) / f"{fid}.svg"
    fig.tight_layout()
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    figures.append({"id": fid, "kind": kind, "title": title, "alt": alt, "path": str(path)})


def make_figures(df, ctx, mods, out_dir, seed):
    figures = []
    if out_dir is None:
        return figures
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        plt = _fig_setup()
    except Exception as e:
        mark(f"WARN: figures skipped ({e})")
        return figures
    q = mods.get("quality", {}).get("evidence", {})
    cols = [c for c in q.get("columns", []) if (c.get("missing_rate") or 0) > 0]
    if cols:
        cols = sorted(cols, key=lambda c: -c["missing_rate"])[:12]
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(cols) + 1.1))
        ax.barh([c["column"] for c in cols][::-1], [c["missing_rate"] for c in cols][::-1],
                color=PALETTE[0])
        ax.set_xlabel("missing rate")
        ax.set_title("Missing values by column")
        _save(plt, fig, out_dir, "fig-quality-missing", figures, "Missing values by column",
              "Horizontal bars of the missing rate for every column that has one.", "bar")
    d = mods.get("distributions", {}).get("evidence", {})
    nums = (d.get("numeric") or [])[:6]
    if nums:
        n = len(nums)
        fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.3), squeeze=False)
        for ax, rec in zip(axes[0], nums):
            v = pd.to_numeric(df[rec["column"]], errors="coerce").dropna().values
            ax.hist(v, bins=30, color=PALETTE[0], alpha=0.85)
            ax.axvline(rec["quantiles"]["p50"], color=PALETTE[1], lw=1)
            ax.set_title(rec["column"], fontsize=8)
            ax.set_yticks([])
        fig.suptitle("Distributions of the numeric basis (median in orange)", fontsize=9)
        _save(plt, fig, out_dir, "fig-distributions", figures, "Numeric distributions",
              "Histograms of the leading numeric columns with the median marked.", "histogram")
    r = mods.get("relations", {}).get("evidence", {})
    basis = r.get("basis_numeric") or []
    if len(basis) >= 2 and r.get("pairs"):
        M = pd.DataFrame(np.eye(len(basis)), index=basis, columns=basis)
        for p in r["pairs"]:
            M.loc[p["a"], p["b"]] = M.loc[p["b"], p["a"]] = p["spearman"]
        fig, ax = plt.subplots(figsize=(0.5 * len(basis) + 2.2, 0.5 * len(basis) + 1.8))
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(basis)), basis, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(basis)), basis, fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
        ax.set_title("Residual monotone association among basis members", fontsize=9)
        _save(plt, fig, out_dir, "fig-relations-heatmap", figures, "Basis correlation heatmap",
              "Heatmap of Spearman correlation between basis members.", "heatmap")
    imp = mods.get("importance", {}).get("evidence", {})
    pi = (imp.get("permutation_importance") or [])[:12]
    if pi:
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(pi) + 1.2))
        names = [p["feature"] for p in pi][::-1]
        vals = [p["mean"] for p in pi][::-1]
        err = [[max(0, p["mean"] - p["ci"][0]) for p in pi][::-1],
               [max(0, p["ci"][1] - p["mean"]) for p in pi][::-1]]
        ax.barh(names, vals, xerr=err, color=PALETTE[2], error_kw={"lw": 0.8})
        ax.set_xlabel(f"permutation importance ({imp.get('metric')})")
        ax.set_title("What carries the task")
        _save(plt, fig, out_dir, "fig-importance", figures, "Permutation importance",
              "Bars of permutation importance per feature with confidence intervals.", "bar")
    lc = imp.get("learning_curve") or []
    if len(lc) >= 3:
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ax.plot([p["n"] for p in lc], [p["score"] for p in lc], "o-", color=PALETTE[0])
        ax.set_xlabel("training rows")
        ax.set_ylabel(imp.get("metric"))
        ax.set_title("Learning curve")
        _save(plt, fig, out_dir, "fig-learning-curve", figures, "Learning curve",
              "Held-out score against training-set size.", "line")
    ts = mods.get("time_series", {}).get("evidence", {})
    if ts.get("series"):
        xs = [p[0] for p in ts["series"]]
        ys = [p[1] for p in ts["series"]]
        fig, ax = plt.subplots(figsize=(6.4, 2.4))
        ax.plot(range(len(ys)), ys, lw=0.9, color=PALETTE[0])
        for cp in ts.get("change_points") or []:
            ax.axvline(cp["index"] * len(ys) / max(1, len(ys)), color=PALETTE[1], lw=1, ls="--")
        step = max(1, len(xs) // 8)
        ax.set_xticks(range(0, len(xs), step), [xs[i] for i in range(0, len(xs), step)],
                      rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{ts.get('value_column')} over {ts.get('column')} (change points dashed)",
                     fontsize=9)
        _save(plt, fig, out_dir, "fig-time-series", figures, "Time series",
              "The value series over time with detected change points.", "line")
    dr = mods.get("drift", {}).get("evidence", {})
    if dr.get("columns"):
        rows = sorted([c for c in dr["columns"] if c.get("psi") is not None],
                      key=lambda c: -(c["psi"] or 0))[:12]
        if rows:
            fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rows) + 1.2))
            ax.barh([c["column"] for c in rows][::-1], [c["psi"] for c in rows][::-1],
                    color=[PALETTE[1] if c["psi"] > 0.25 else PALETTE[0] for c in rows][::-1])
            ax.axvline(0.1, color="#888", lw=0.8, ls=":")
            ax.axvline(0.25, color="#888", lw=0.8, ls="--")
            ax.set_xlabel("PSI between parts (0.1 / 0.25 thresholds dotted)")
            ax.set_title("Distribution drift between splits")
            _save(plt, fig, out_dir, "fig-drift", figures, "Drift by column",
                  "PSI per column between the two split parts.", "bar")
    sg = mods.get("segments", {}).get("evidence", {})
    if sg.get("scans"):
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        ks = [s["k"] for s in sg["scans"]]
        ax.plot(ks, [s["silhouette"] or 0 for s in sg["scans"]], "o-", color=PALETTE[0],
                label="silhouette")
        ax.plot(ks, [s["stability_ari_mean"] for s in sg["scans"]], "s--", color=PALETTE[2],
                label="stability (ARI)")
        ax.set_xlabel("k")
        ax.legend(fontsize=7)
        ax.set_title("Segment scan")
        _save(plt, fig, out_dir, "fig-segments", figures, "Segment scan",
              "Silhouette and bootstrap stability against the number of clusters.", "line")
    sp = mods.get("spatial", {}).get("evidence", {})
    if sp.get("points"):
        fig, ax = plt.subplots(figsize=(4.4, 3.4))
        pts = sp["points"]
        lab = np.array([p[2] for p in pts])
        for i, l in enumerate(sorted(set(lab.tolist()))):
            m = lab == l
            ax.scatter([p[0] for p, k in zip(pts, m) if k], [p[1] for p, k in zip(pts, m) if k],
                       s=6, color="#999" if l == -1 else PALETTE[i % len(PALETTE)],
                       label=("noise" if l == -1 else f"cluster {l}"))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.legend(fontsize=6, markerscale=1.5)
        ax.set_title("Spatial clusters (DBSCAN, metres)", fontsize=9)
        _save(plt, fig, out_dir, "fig-spatial", figures, "Spatial clusters",
              "Scatter of the sampled locations coloured by DBSCAN cluster.", "scatter")
    return figures


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset file (.csv .tsv .parquet .xlsx .json .jsonl)")
    ap.add_argument("--model", default=None,
                    help="the /dataset-forge model HTML whose geometry layer is the baseline")
    ap.add_argument("--out", default="analysis.json", help="output JSON path")
    ap.add_argument("--split", default=None, help="column that names the split part (enables drift)")
    ap.add_argument("--split-file", default=None,
                    help="one part label per row, aligned with the dataset (enables drift)")
    ap.add_argument("--modules", default=None,
                    help="comma-separated subset of " + ",".join(MODULES))
    ap.add_argument("--seed", type=int, default=7, help="seed for every sample, resample and model")
    ap.add_argument("--figures", default=None, help="directory for the SVG figures")
    ap.add_argument("--max-rows", type=int, default=200000,
                    help="rows above which the dataset is subsampled (seeded) for the scans")
    ap.add_argument("--series-column", default=None,
                    help="force the time-series module onto this numeric column")
    ap.add_argument("--sep", default=None, help="CSV separator override")
    a = ap.parse_args(argv)

    try:
        df = read_dataset(a.dataset, a.sep)
    except Exception as e:
        print(f"ERROR: cannot read {a.dataset}: {e}")
        return 2
    mark(f"OK: parsed {len(df)} rows × {df.shape[1]} columns from {a.dataset}")
    if len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=a.seed).sort_index()
        mark(f"WARN: subsampled to {a.max_rows} rows (seed {a.seed}) for the scans")

    geo, kind = (None, None)
    if a.model:
        try:
            geo, kind = read_geometry(a.model)
        except OSError as e:
            print(f"ERROR: cannot read model {a.model}: {e}")
            return 2
        if geo is None:
            mark("WARN: the model carries no geometry layer; roles and basis are guessed here "
                 "and the layer records source.geometry: absent")
        else:
            mark(f"OK: geometry layer read from {a.model} "
                 f"(basis of {len((geo.get('basis') or {}).get('members') or [])}, "
                 f"{len(geo.get('derivations') or [])} derivations)")
    else:
        mark("WARN: no --model; roles and basis are guessed from the data alone")

    split_series = None
    if a.split_file:
        split_series = pd.Series([l.strip() for l in Path(a.split_file).read_text().splitlines()
                                  if l.strip()])
        if len(split_series) != len(df):
            mark(f"WARN: --split-file has {len(split_series)} labels for {len(df)} rows; drift skipped")
            split_series = None
    ctx = build_context(df, geo, a.split)
    if a.split and a.split in df.columns:
        split_series = df[a.split]

    want = [m.strip() for m in a.modules.split(",")] if a.modules else list(MODULES)
    unknown = [m for m in want if m not in MODULES]
    if unknown:
        print(f"ERROR: unknown module(s) {unknown}; known: {', '.join(MODULES)}")
        return 2

    mods = {}
    def run(name, fn, precondition=None):
        if name not in want:
            mods[name] = {"ran": False, "skipped_because": "not requested (--modules)", "evidence": {}}
            return
        if precondition:
            mods[name] = {"ran": False, "skipped_because": precondition, "evidence": {}}
            mark(f"OK: module {name} skipped — {precondition}")
            return
        try:
            ev = fn()
            mods[name] = {"ran": True, "evidence": ev}
            mark(f"OK: module {name} ran")
        except Exception as e:
            import traceback
            mods[name] = {"ran": False, "skipped_because": f"failed: {e}", "evidence": {}}
            mark(f"WARN: module {name} failed ({e})")
            if __debug__ and "--trace" in sys.argv:
                traceback.print_exc()

    run("quality", lambda: mod_quality(df, ctx, a.seed))
    run("distributions", lambda: mod_distributions(df, ctx, a.seed))
    run("relations", lambda: mod_relations(df, ctx, a.seed))
    run("inference", lambda: mod_inference(df, ctx, a.seed))
    run("segments", lambda: mod_segments(df, ctx, a.seed))
    run("importance", lambda: mod_importance(df, ctx, a.seed),
        None if ctx["partition"]["label"] else "no partition chosen")
    run("time_series", lambda: mod_time_series(df, ctx, a.seed, a.series_column),
        None if ctx.get("time") else "no datetime column")
    run("spatial", lambda: mod_spatial(df, ctx, a.seed),
        None if ctx.get("spatial") else "no coordinate pair")
    run("drift", lambda: mod_drift(df, ctx, a.seed, split_series),
        None if split_series is not None else "no split column and no --split")

    figures = make_figures(df, ctx, mods, a.figures, a.seed)
    if figures:
        mark(f"OK: {len(figures)} figures written to {a.figures}")

    doc = {"schema": SCHEMA,
           "source": {"path": str(a.dataset), "rows": int(len(df)), "columns": list(df.columns),
                      "seed": int(a.seed), "model": str(a.model) if a.model else None,
                      "geometry": "present" if geo else "absent", "source_kind": kind},
           "parameters": {"modules": want, "split": a.split, "split_file": a.split_file,
                          "max_rows": a.max_rows, "alpha": ALPHA},
           "context": ctx, "modules": mods, "figures": figures, "markers": MARKERS}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(doc), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    ran = [m for m, v in mods.items() if v.get("ran")]
    print(f"OK: wrote {out} ({len(ran)}/{len(MODULES)} modules ran: {', '.join(ran)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
