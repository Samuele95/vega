#!/usr/bin/env python3
"""geometry.py — the empirical channel of /dataset-forge.

Reads a tabular dataset and emits ONE JSON document describing its geometry:
column types, rank and effective dimension of the numeric space, intrinsic
dimension, linear derivability of each numeric column from the others, exact
and near functional dependencies, mixed-type dependency measures, per-column
statistics, and a deterministic row sample for the A-box.

It computes; it does not judge. Every proposal here (a derivation, an FD, a
column type) is a *hypothesis with evidence* that the orchestrating skill
reconciles against its own semantic reading and against the domain-forge
reasoner. Output starts with greppable markers (OK:/WARN:/ERROR:).

Usage:
  geometry.py <path.(csv|tsv|parquet|xlsx|json)> [--out geometry.json]
              [--sample 200] [--explore-rows 2000] [--seed 7] [--max-lhs 2] [--fd-rows 20000]
              [--r2 0.999] [--sep ,]
Dependencies: numpy, pandas; scipy and scikit-learn optional (better MI).
"""
from __future__ import annotations

import argparse
import re
import json
import math
import sys
from itertools import combinations

import numpy as np
import pandas as pd

MARKERS: list[str] = []


def mark(kind: str, msg: str) -> None:
    MARKERS.append(f"{kind}: {msg}")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(path: str, sep: str | None) -> pd.DataFrame:
    p = path.lower()
    if p.endswith((".csv", ".tsv", ".txt")):
        return pd.read_csv(path, sep=sep or ("\t" if p.endswith(".tsv") else ","))
    if p.endswith(".parquet"):    # Arrow's own filesystem: no Python in Arrow's threads at exit
        from pyarrow import fs
        return pd.read_parquet(path, filesystem=fs.LocalFileSystem())
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if p.endswith((".json", ".jsonl")):
        return pd.read_json(path, lines=p.endswith(".jsonl"))
    raise ValueError(f"unsupported file type: {path}")


# --------------------------------------------------------------------------- #
# Stage 0 — column typing (heuristic; the skill confirms semantically)
# --------------------------------------------------------------------------- #
def date_like(strs: pd.Series) -> bool:
    """Most values parse as datetimes in a plausible year (1000-2999) AND they name more than one
    calendar day. The lenient parser reads a code such as `T09` (a scene tile) as 9 January of
    year 1, and a time of day as today: neither is a date, and a monthly discretisation would
    collapse the column to one period."""
    parsed = pd.to_datetime(strs, errors="coerce", format="mixed")
    good = parsed.dropna()
    return bool(parsed.notna().mean() > 0.9 and good.dt.year.between(1000, 2999).mean() > 0.9
                and good.dt.normalize().nunique() > 1)


def type_columns(df: pd.DataFrame) -> dict[str, dict]:
    n = len(df)
    out: dict[str, dict] = {}
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        nun = int(nn.nunique())
        info = {"dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "missing_pct": round(100 * s.isna().mean(), 2), "nunique": nun}
        if nun <= 1:
            t = "constant"
        elif pd.api.types.is_bool_dtype(s) or (nun == 2 and set(map(str, nn.unique())) <= {"0", "1", "True", "False", "true", "false"}):
            t = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            t = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            unique_ratio = nun / max(len(nn), 1)
            is_int = np.allclose(nn.astype(float), nn.astype(float).round())
            if is_int and unique_ratio > 0.98 and n > 20:
                t = "id"  # integer key-like column
            elif is_int and nun <= 20:
                t = "ordinal"  # small integer codes; the skill may retype to nominal
            else:
                t = "numeric"
        else:
            strs = nn.astype(str)
            unique_ratio = nun / max(len(nn), 1)
            if len(strs) and date_like(strs) and nun > 2:
                t = "datetime"
            elif unique_ratio > 0.98 and n > 20 and strs.str.len().mean() < 40:
                t = "id"
            elif strs.str.len().mean() > 30 or (unique_ratio > 0.5 and strs.str.contains(r"\s").mean() > 0.5):
                t = "text"
            else:
                t = "nominal"
        info["type"] = t
        name = str(c).lower()
        if re.search(r"(^|_)(id|key|uuid|code)$", name):
            info["name_hint"] = "identifier-like name (may be a foreign key: repeated values, still not a dimension)"
        elif re.search(r"(zip|postal|cap|plz)", name):
            info["name_hint"] = "postal-code-like name: treat as nominal even if parsed as integer"
        if t in ("nominal", "ordinal", "boolean"):
            vc = nn.astype(str).value_counts()
            info["top_values"] = {str(k): int(v) for k, v in vc.head(8).items()}
            info["cardinality"] = nun
        out[str(c)] = info
    return out


def numeric_frame(df: pd.DataFrame, types: dict) -> pd.DataFrame:
    cols = [c for c, i in types.items() if i["type"] in ("numeric", "ordinal", "boolean")]
    X = df[cols].copy()
    for c in cols:
        if types[c]["type"] == "boolean":
            X[c] = X[c].astype(str).str.lower().map({"true": 1, "1": 1, "false": 0, "0": 0}).astype(float)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# --------------------------------------------------------------------------- #
# Stage 1 — the space: rank, effective dimension, intrinsic dimension
# --------------------------------------------------------------------------- #
def space(X: pd.DataFrame) -> dict:
    if X.shape[1] == 0:
        mark("WARN", "no numeric columns — geometric stage runs on encoded categoricals only")
        return {"numeric_columns": [], "n_rows_used": 0}
    Xc = X.dropna()
    if len(Xc) < len(X):
        mark("WARN", f"rank/dimension computed on {len(Xc)} complete rows of {len(X)} (rows with NaN dropped)")
    A = Xc.to_numpy(dtype=float)
    A = A - A.mean(axis=0)
    std = A.std(axis=0)
    std[std == 0] = 1.0
    Z = A / std
    sv = np.linalg.svd(Z, compute_uv=False) if len(Z) else np.array([])
    p = Z.shape[1]
    res: dict = {"numeric_columns": list(X.columns), "n_rows_used": int(len(Z)), "ambient_dim": p}
    if len(sv) == 0:
        return res
    tol = sv.max() * max(Z.shape) * np.finfo(float).eps
    exact_rank = int((sv > tol).sum())
    near_rank = int((sv > sv.max() * 1e-6).sum())
    var = sv ** 2 / (sv ** 2).sum()
    cum = np.cumsum(var)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    pr = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
    res.update({
        "singular_values": [round(float(v), 6) for v in sv],
        "explained_variance_ratio": [round(float(v), 6) for v in var],
        "exact_rank": exact_rank, "near_rank_1e-6": near_rank,
        "dims_for_95pct_variance": k95, "dims_for_99pct_variance": k99,
        "participation_ratio": round(pr, 3),
        "intrinsic_dim_twonn": twonn(Z),
        "condition_number": round(float(sv.max() / max(sv.min(), 1e-300)), 3),
    })
    if exact_rank < p:
        mark("OK", f"numeric space has exact rank {exact_rank} < {p} columns — at least {p - exact_rank} column(s) are linear combinations of the others")
    elif near_rank < p:
        mark("OK", f"numeric space is near-degenerate: near-rank {near_rank} of {p}")
    else:
        mark("OK", f"numeric space has full rank {p}; effective dimension {k95} (95% variance), TwoNN {res['intrinsic_dim_twonn']}")
    return res


def twonn(Z: np.ndarray, max_points: int = 3000, seed: int = 7) -> float | None:
    """Facco et al. (2017) two-nearest-neighbours intrinsic-dimension estimator."""
    n, p = Z.shape
    if n < 20 or p < 2:
        return None
    rng = np.random.default_rng(seed)
    if n > max_points:
        Z = Z[rng.choice(n, max_points, replace=False)]
    Zu = np.unique(Z, axis=0)
    if len(Zu) < 20:
        return None
    d2 = ((Zu[:, None, :] - Zu[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    part = np.partition(d2, 1, axis=1)[:, :2]
    r1, r2 = np.sqrt(part[:, 0]), np.sqrt(part[:, 1])
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    mu = mu[: int(len(mu) * 0.9)]  # discard the tail as in the paper
    F = np.arange(1, len(mu) + 1) / len(mu)
    x, y = np.log(mu), -np.log(1 - F + 1e-12)
    d = float((x * y).sum() / (x * x).sum())
    return round(d, 2)


# --------------------------------------------------------------------------- #
# Stage 2 (empirical half) — linear derivability of each numeric column
# --------------------------------------------------------------------------- #
def linear_derivability(X: pd.DataFrame, r2_threshold: float) -> list[dict]:
    Xc = X.dropna()
    cols = list(Xc.columns)
    out = []
    if len(cols) < 2 or len(Xc) < 3:
        return out
    for tgt in cols:
        others = [c for c in cols if c != tgt]
        A = np.column_stack([Xc[others].to_numpy(dtype=float), np.ones(len(Xc))])
        y = Xc[tgt].to_numpy(dtype=float)
        if y.std() == 0:
            continue
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        if r2 >= r2_threshold:
            terms = {c: round(float(k), 6) for c, k in zip(others, coef[:-1]) if abs(k) > 1e-8 * max(1, abs(y).max())}
            out.append({"column": tgt, "r2": round(float(r2), 6), "intercept": round(float(coef[-1]), 6),
                        "formula_terms": terms, "kind": "linear"})
            mark("OK", f"'{tgt}' is linearly derivable (R²={r2:.6f}) from {list(terms)}")
    # degree-2: pairwise products of the other columns (catches a*b, a*(1-b/100), ...)
    if 3 <= len(cols) <= 14:
        for tgt in cols:
            if any(d["column"] == tgt for d in out):
                continue
            others = [c for c in cols if c != tgt]
            O = Xc[others].to_numpy(dtype=float)
            prods = [(f"{a}*{b}", O[:, i] * O[:, j]) for (i, a), (j, b) in combinations(list(enumerate(others)), 2)]
            names = others + [n for n, _ in prods]
            A = np.column_stack([O] + [v for _, v in prods] + [np.ones(len(Xc))])
            y = Xc[tgt].to_numpy(dtype=float)
            if y.std() == 0 or A.shape[1] >= len(Xc):
                continue
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if r2 >= r2_threshold:
                scale = max(1.0, float(abs(y).max()))
                terms = {n: round(float(k), 6) for n, k in zip(names, coef[:-1]) if abs(k) > 1e-7 * scale}
                rec = {"column": tgt, "r2": round(float(r2), 6), "intercept": round(float(coef[-1]), 6),
                       "formula_terms": terms, "kind": "polynomial-2"}
                if np.linalg.matrix_rank(A) < A.shape[1]:
                    rec["formula_ambiguous"] = True
                    rec["note"] = "regressors are mutually dependent: derivability is established, the coefficient split is not canonical — the semantic channel must state the formula, the reasoner verifies it"
                out.append(rec)
                mark("OK", f"'{tgt}' is derivable with degree-2 terms (R²={r2:.6f}) from {list(terms)}")
    # products/ratios: cheap log-space check for positive columns
    pos = [c for c in cols if (Xc[c] > 0).all()]
    if len(pos) >= 3:
        L = np.log(Xc[pos].to_numpy(dtype=float))
        for i, tgt in enumerate(pos):
            others = [c for j, c in enumerate(pos) if j != i]
            A = np.column_stack([np.delete(L, i, axis=1), np.ones(len(L))])
            y = L[:, i]
            if y.std() == 0:
                continue
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            if r2 >= r2_threshold and not any(d["column"] == tgt for d in out):
                terms = {c: round(float(k), 4) for c, k in zip(others, coef[:-1]) if abs(k) > 1e-6}
                out.append({"column": tgt, "r2": round(float(r2), 6), "formula_terms": terms,
                            "intercept_log": round(float(coef[-1]), 6), "kind": "multiplicative"})
                mark("OK", f"'{tgt}' is multiplicatively derivable (log-R²={r2:.6f}) from {list(terms)}")
    return out


# --------------------------------------------------------------------------- #
# Stage 2/3 — functional dependencies (type-agnostic) on discrete columns
# --------------------------------------------------------------------------- #
def functional_dependencies(df: pd.DataFrame, types: dict, max_lhs: int, max_rows: int) -> list[dict]:
    disc = [c for c, i in types.items()
            if i["type"] in ("nominal", "ordinal", "boolean", "datetime") or
            (i["type"] == "numeric" and i["nunique"] <= 50)]
    if len(disc) < 2:
        return []
    D = df[disc].head(max_rows).astype(str)
    if len(df) > max_rows:
        mark("WARN", f"functional dependencies mined on the first {max_rows} rows of {len(df)}")
    # a near-key LHS determines everything trivially — exclude high-cardinality columns from LHS
    lhs_pool = [c for c in disc if D[c].nunique() <= 0.3 * len(D)]
    out = []
    for k in range(1, max_lhs + 1):
        for lhs in combinations(lhs_pool, k):
            g = D.groupby(list(lhs), sort=False)
            n_groups = g.ngroups
            if n_groups > 0.3 * len(D):  # combination is a near-key on this slice
                continue
            for rhs in disc:
                if rhs in lhs:
                    continue
                # skip if a subset of lhs already determines rhs exactly
                if k > 1 and any(d["exact"] and d["rhs"] == rhs and set(d["lhs"]) < set(lhs) for d in out):
                    continue
                nun = g[rhs].nunique()
                exact = bool((nun == 1).all())
                ratio = float((nun == 1).mean())
                if exact or ratio >= 0.95:
                    out.append({"lhs": list(lhs), "rhs": rhs, "exact": exact,
                                "determination_ratio": round(ratio, 4), "groups": int(n_groups)})
                    if exact:
                        mark("OK", f"functional dependency {list(lhs)} -> '{rhs}' holds exactly ({n_groups} groups)")
    return out


# --------------------------------------------------------------------------- #
# Stage 3 — dependency measures between columns, by type pair
# --------------------------------------------------------------------------- #
def cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    obs = ct.to_numpy(dtype=float)
    exp = obs.sum(1, keepdims=True) @ obs.sum(0, keepdims=True) / obs.sum()
    chi2 = ((obs - exp) ** 2 / np.where(exp == 0, 1, exp)).sum()
    n = obs.sum()
    return float(math.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    d = pd.DataFrame({"c": cat.astype(str), "y": pd.to_numeric(num, errors="coerce")}).dropna()
    if d["c"].nunique() < 2 or len(d) < 3:
        return 0.0
    grand = d["y"].mean()
    ss_between = (d.groupby("c")["y"].agg(["mean", "count"]).eval("count * (mean - @grand) ** 2")).sum()
    ss_total = ((d["y"] - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def discretize(s: pd.Series, t: str, bins: int = 10) -> pd.Series:
    if t in ("numeric",):
        x = pd.to_numeric(s, errors="coerce")
        try:
            return pd.qcut(x, q=min(bins, max(x.nunique(), 1)), duplicates="drop").astype(str)
        except ValueError:
            return x.astype(str)
    if t == "datetime":
        return pd.to_datetime(s, errors="coerce", format="mixed").dt.to_period("M").astype(str)
    return s.astype(str)


def normalized_mi(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 3:
        return 0.0
    ct = pd.crosstab(d["a"], d["b"]).to_numpy(dtype=float)
    pxy = ct / ct.sum()
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    mi = (pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum()
    ha = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hb = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    # a one-bin marginal carries no information: its entropy is 0, but its probabilities can sum
    # to 1 + ulp and read as -2e-16 (math domain error) or +2e-16 (a spurious ratio)
    if ha < 1e-12 or hb < 1e-12:
        return 0.0
    return float(mi / math.sqrt(ha * hb))


def dependencies(df: pd.DataFrame, types: dict, max_rows: int) -> dict:
    cols = [c for c, i in types.items() if i["type"] not in ("id", "constant", "text")]
    D = df[cols].head(max_rows)
    pairs = []
    disc_cache = {c: discretize(D[c], types[c]["type"]) for c in cols}
    for a, b in combinations(cols, 2):
        ta, tb = types[a]["type"], types[b]["type"]
        rec: dict = {"a": a, "b": b, "nmi": round(normalized_mi(disc_cache[a], disc_cache[b]), 4)}
        num = {"numeric", "ordinal", "boolean"}
        if ta in num and tb in num:
            x, y = pd.to_numeric(D[a], errors="coerce"), pd.to_numeric(D[b], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0:
                rec["pearson"] = round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 4)
                rec["spearman"] = round(float(x[ok].rank().corr(y[ok].rank())), 4)
        elif ta in num and tb in ("nominal", "datetime"):
            rec["eta2"] = round(correlation_ratio(disc_cache[b], D[a]), 4)
        elif tb in num and ta in ("nominal", "datetime"):
            rec["eta2"] = round(correlation_ratio(disc_cache[a], D[b]), 4)
        else:
            rec["cramers_v"] = round(cramers_v(disc_cache[a], disc_cache[b]), 4)
        pairs.append(rec)
    strong = [p for p in pairs if p["nmi"] >= 0.5 or abs(p.get("pearson", 0)) >= 0.9
              or p.get("eta2", 0) >= 0.8 or p.get("cramers_v", 0) >= 0.8]
    for p in strong:
        mark("OK", f"strong dependency '{p['a']}' ~ '{p['b']}' ({', '.join(f'{k}={v}' for k, v in p.items() if k not in ('a', 'b'))})")
    return {"columns": cols, "pairs": pairs, "strong": strong,
            "thresholds": {"nmi": 0.5, "pearson": 0.9, "eta2": 0.8, "cramers_v": 0.8}}


# --------------------------------------------------------------------------- #
# Stage 5 — classic statistics, compact
# --------------------------------------------------------------------------- #
def stats(df: pd.DataFrame, types: dict) -> dict:
    out = {}
    for c, i in types.items():
        t = i["type"]
        s = df[c]
        if t in ("numeric", "ordinal"):
            x = pd.to_numeric(s, errors="coerce").dropna()
            if len(x) == 0:
                continue
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            out[c] = {"mean": round(float(x.mean()), 6), "std": round(float(x.std()), 6),
                      "min": float(x.min()), "q25": float(q1), "median": float(x.median()),
                      "q75": float(q3), "max": float(x.max()), "skew": round(float(x.skew()), 4),
                      "kurtosis": round(float(x.kurt()), 4),
                      "iqr_outliers": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
                      "zeros": int((x == 0).sum())}
        elif t == "datetime":
            d = pd.to_datetime(s, errors="coerce", format="mixed").dropna()
            if len(d):
                out[c] = {"min": str(d.min()), "max": str(d.max()),
                          "span_days": int((d.max() - d.min()).days),
                          "monotonic": bool(d.is_monotonic_increasing)}
        elif t == "text":
            strs = s.dropna().astype(str)
            out[c] = {"avg_len": round(float(strs.str.len().mean()), 1),
                      "max_len": int(strs.str.len().max()) if len(strs) else 0,
                      "avg_tokens": round(float(strs.str.split().str.len().mean()), 1) if len(strs) else 0}
        elif t in ("nominal", "boolean"):
            vc = s.dropna().astype(str).value_counts(normalize=True)
            out[c] = {"cardinality": int(len(vc)),
                      "top_share": round(float(vc.iloc[0]), 4) if len(vc) else None,
                      "entropy_bits": round(float(-(vc * np.log2(vc)).sum()), 4) if len(vc) else 0.0}
    return out


# --------------------------------------------------------------------------- #
# A-box sample — deterministic rows the reasoner will see
# --------------------------------------------------------------------------- #
def abox_sample(df: pd.DataFrame, n: int, seed: int) -> dict:
    s = df.sample(n=min(n, len(df)), random_state=seed) if len(df) > n else df
    rows = json.loads(s.to_json(orient="records", date_format="iso"))
    return {"n": len(rows), "seed": seed, "index": [int(i) for i in s.index], "rows": rows,
            "note": "deterministic sample for the ontology A-box; the reasoner verifies rules on THESE rows only"}


# --------------------------------------------------------------------------- #
# Explorer payload — a larger columnar sample plus a precomputed PCA projection
# --------------------------------------------------------------------------- #
def explore_payload(df: pd.DataFrame, X: pd.DataFrame, types: dict, n: int, seed: int) -> dict:
    s = df.sample(n=min(n, len(df)), random_state=seed) if len(df) > n else df
    cols = [c for c in df.columns if types[str(c)]["type"] != "text"]
    columnar = {}
    for c in cols:
        t = types[str(c)]["type"]
        v = s[c]
        if t == "datetime":
            v = pd.to_datetime(v, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
        columnar[str(c)] = json.loads(v.to_json(orient="values", date_format="iso"))
    out = {"n": int(len(s)), "seed": seed, "index": [int(i) for i in s.index], "columns": columnar,
           "note": "columnar sample for the interactive explorer; text columns omitted"}
    if X.shape[1] >= 2:
        Xs = X.loc[s.index].dropna()
        A = Xs.to_numpy(dtype=float)
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        Z = (A - mu) / sd
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        k = min(3, Vt.shape[0])
        scores = U[:, :k] * S[:k]
        out["pca"] = {"index": [int(i) for i in Xs.index], "columns": list(X.columns),
                      "scores": [[round(float(v), 4) for v in row] for row in scores],
                      "loadings": [[round(float(v), 4) for v in Vt[j]] for j in range(k)],
                      "explained": [round(float(v), 4) for v in (S[:k] ** 2 / (S ** 2).sum())],
                      "mean": [round(float(v), 6) for v in mu], "std": [round(float(v), 6) for v in sd]}
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="geometry.json")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--explore-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-lhs", type=int, default=2)
    ap.add_argument("--fd-rows", type=int, default=20000)
    ap.add_argument("--r2", type=float, default=0.999)
    ap.add_argument("--sep", default=None)
    a = ap.parse_args()

    try:
        df = load(a.path, a.sep)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: cannot load {a.path}: {e}")
        return 2
    mark("OK", f"parsed {len(df)} rows × {df.shape[1]} columns from {a.path}")
    if df.shape[1] > 60:
        mark("WARN", f"{df.shape[1]} columns: pairwise dependency mining is O(p²); consider a column subset")

    types = type_columns(df)
    ids = [c for c, i in types.items() if i["type"] == "id"]
    consts = [c for c, i in types.items() if i["type"] == "constant"]
    if ids:
        mark("OK", f"identifier columns excluded from dependency analysis: {ids}")
    if consts:
        mark("WARN", f"constant columns carry no dimension: {consts}")
    X = numeric_frame(df, types)
    for c in df.columns:
        if df[c].isna().mean() > 0.3:
            mark("WARN", f"'{c}' is {types[str(c)]['missing_pct']}% missing")

    result = {
        "markers": None,  # filled last, placed first
        "source": {"path": a.path, "rows": int(len(df)), "columns": [str(c) for c in df.columns]},
        "columns": types,
        "space": space(X),
        "linear_derivability": linear_derivability(X, a.r2),
        "functional_dependencies": functional_dependencies(df, types, a.max_lhs, a.fd_rows),
        "dependencies": dependencies(df, types, a.fd_rows),
        "stats": stats(df, types),
        "abox_sample": abox_sample(df, a.sample, a.seed),
        "explore": explore_payload(df, X, types, a.explore_rows, a.seed),
        "parameters": {"sample": a.sample, "seed": a.seed, "max_lhs": a.max_lhs, "fd_rows": a.fd_rows, "r2": a.r2, "explore_rows": a.explore_rows},
    }
    result["markers"] = MARKERS
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=str)
    print("\n".join(MARKERS))
    print(f"OK: wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
