#!/usr/bin/env python3
"""
shape.py — check and execute a /dataset-shaper recipe: the decisions an analysis already took,
applied to the data, deterministically, with every step traced to what justified it.

    python3 shape.py --recipe recipe.json --model MODEL.html --check-only
    python3 shape.py --recipe recipe.json --out-dir shaped/ [--format parquet|csv] [--dry-run]
    python3 shape.py --check --out-dir shaped/
    python3 shape.py --transform-only --recipe recipe.json --fitted fitted.json --emit preprocess.py

Four jobs, one script:

  --check-only  validates the recipe against `references/shape-contract.md` §1 and, when
                --model is given, resolves every step's `source` against the model's layers.
                A step with no source, a fitted step before the split, a leakage column that
                survives, a basis member dropped without a `user:` step, a step that
                contradicts a recorded analysis stance, an impute or a lag grouped by the
                label, a select_features kind=importance in a recipe that splits, a pre-split custom
                step declaring fitted: true: each is an ERROR here, before any data moves.
  (default)     executes the recipe phase by phase and writes the dataset file(s),
                `manifest.json`, `lineage.json`, `recipe.json` and `reproduce_<slug>.py` —
                a standalone pandas program that regenerates the outputs without this skill.
  --check       re-runs that reproduction script and compares the digests it produces with
                the ones in the manifest. Determinism is a test, not a claim.
  --transform-only
                reads the recipe and `fitted.json` — never the dataset — and emits
                `preprocess.py`, the inference twin: `transform(df_raw, history=None)`, the
                recipe applied to new rows with the parameters already fitted. It is a
                *filtered re-emission* of the very lines the reproduction script carries
                (`fitted.json.__code__` records them per step), run over a single part, so
                the training program and the serving program are one implementation and not
                two that agree today. `/vega-pack` calls it for the bundle's
                `FeatureTransformer` adapter; `verify_shape.py --transform-only` is its twin
                check. On a recipe with a `graph_edges` step the module also carries
                `edges_for(df_raw, reference)` (level node: a query row's edges into the
                reference nodes, never into another query row) and `edges_within(df_raw)`
                (level graph: the edges inside each whole graph of the query), computed by the
                same emitted functions that wrote the `edges` part.

A `graph_edges` step (phase 7, after the split) changes neither rows nor columns: it writes the
`edges` part beside the split parts — one row per directed edge, both ends as raw-file
positions with their part — the manifest's `graph` block, and, for `knn` / `spatial` at level
node, a fitted entry naming the train matrix its index was built on. The executor, the
reproduction script and the transform-only module run one emitted implementation of it
(`GRAPH_LIB`), so the three cannot drift.

Fitted steps (impute with a statistic, encode, scale, project, quantile bin, winsorize, a
box-cox / yeo-johnson transform, a variance selection, a spatial step's fallback projection or
neighbour index) are fitted on the train part alone when a split exists — the manifest records
`fit_on` per step — and applied to every part. That is not a nicety: it is the difference
between an honest estimate and one that has already seen its own test set. Before the split a
`custom` step runs on the whole frame, the future test rows in it, so the executor probes it for
row-locality (`row_local_probe`) and refuses one that computes over rows.

Exit codes: 0 done; 2 recipe or input error; 3 execution or verification failure.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import copy
import hashlib
import json
import math
import re
import subprocess
import symtable
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SKILL_DIR = HERE.parent
SCHEMA = "dataset-shaper/recipe@1"

PHASES = {
    1: ("retype", "drop_identity", "drop_constant", "dedupe", "parse_datetime", "parse_geometry"),
    2: ("orient_cycle", "drop_derived", "select_partition", "drop_leakage", "keep_columns"),
    3: ("split",),
    4: ("impute", "clip", "winsorize", "transform", "bin", "datetime_expand", "lag",
        "spatial_reproject", "spatial_distance", "spatial_join", "spatial_grid",
        "spatial_features"),
    5: ("encode", "scale", "project"),
    6: ("select_features",),
    7: ("graph_edges",),
}
OP_PHASE = {op: ph for ph, ops in PHASES.items() for op in ops}
FITTED_OPS = {"impute", "encode", "scale", "project", "bin", "winsorize"}
SPATIAL_OPS = {op for op in PHASES[4] if op.startswith("spatial_")}

# --- what an op means at inference, when the frame is a query and not a training corpus.
# Two different questions, and confusing them is silent: does the step shape the *corpus*
# (how many rows there are, which rows they are) or does it shape the *row* (which columns
# the network is handed, and with what values)? A corpus step applied to a query returns
# fewer predictions than rows given; a row step skipped returns a differently-shaped vector
# the network will score anyway.
TRANSFORM_SKIP = {
    "split": "there is nothing to split: a query is one part",
    "dedupe": "deduplicating a query would return fewer predictions than rows given; "
              "duplicate rows are duplicate questions, each of which has an answer",
    # graph_edges shapes neither the corpus nor the row but the relation of a row to the
    # corpus: the module serves it through edges_for / edges_within, never through transform.
    "graph_edges": "it shapes the relation of a row to the corpus, not the row: served by "
                   "edges_for (level node) and edges_within (level graph) below",
}
# `select_partition` shapes neither: it names the label. At inference the label is not an
# input, so the module drops that column when a caller leaves it in, records the partition
# so `predict.py` can say which question these rows are assumed to be asking, and filters
# nothing — the caller supplies the rows, and dropping any of them here would answer fewer
# questions than were asked without saying so.
TRANSFORM_RECORD = {"select_partition"}
# `lag` needs the rows before the query. Everything else in phase 4-6 is row-local under the
# fitted parameters, and `drop_*` / `keep_columns` / `select_features` reduce to selection by
# name — the emitted line is already `drop(columns=[c for c in <names> if c in ...])`, which
# re-decides nothing.
TRANSFORM_HISTORY_OPS = {"lag"}
# Row-local is not the same as batch-independent, and every applied catalog op is both: even
# `spatial_features` reads its neighbours from an index fitted on the train part (fitted.json
# carries it), never from the frame it is given, so a one-row query and a 10 000-row batch get
# the same value, and the twin check holds on every part.

SOURCE_RE = re.compile(r"^(geometry:[A-Za-z0-9_./\-]+|analysis:(F\d+|turn\d+)/T\w+|"
                       r"analysis:[A-Za-z0-9_./\-]+|user:[^\s].*|shaper:default)$")
MARKERS: list[str] = []


def mark(s):
    MARKERS.append(s)
    print(s)


def record_determinism(out_dir, verdict: str) -> None:
    """The determinism verdict, recorded where the layer reads it (F45): `verification.json` in
    the output directory — the report `verify_shape.py --json` wrote — gains `determinism`, so the
    apply script splices the check's own result and nobody types it. A directory with no report
    yet gets one holding only this verdict (the apply script still refuses it: structural is
    missing until the verifier runs)."""
    path = Path(out_dir) / "verification.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    doc["determinism"] = verdict
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OK: determinism {verdict!r} recorded in {path}")


def digest_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def digest_text(s):
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is None or isinstance(o, str):
        return o
    return str(o)


# --------------------------------------------------------------------------- model layers
def read_layers(model_path):
    """Return {"geometry": {...}, "analysis": {...}} for whichever layers the model carries."""
    out = {}
    if not model_path:
        return out
    html = Path(model_path).read_text(encoding="utf-8")
    for name in ("geometry", "analysis", "shape"):
        m = re.search(r'<script\s+id="layer-' + name + r'-data"[^>]*>([\s\S]*?)</script>',
                      html, re.IGNORECASE)
        if m:
            try:
                out[name] = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                mark(f"WARN: layer-{name}-data did not parse; its sources cannot be resolved")
    return out


def resolve_source(src, layers):
    """True when the source names something the layers actually contain."""
    if src == "shaper:default" or src.startswith("user:"):
        return True
    if src.startswith("geometry:"):
        geo = layers.get("geometry")
        if geo is None:
            return None                       # unresolvable, not false
        path = src.split(":", 1)[1].split("/")
        head = path[0]
        if head == "typing":
            cols = {t.get("column") for t in geo.get("typing") or []}
            return len(path) < 2 or path[1] in cols
        if head in ("derivations",):
            cols = {d.get("column") for d in geo.get("derivations") or []}
            return len(path) < 2 or path[1] in cols
        if head == "cycles":
            ids = {c.get("id") for c in geo.get("cycles") or []}
            return len(path) < 2 or path[1] in ids
        if head == "partitions":
            labs = {c.get("label") for c in (geo.get("partitions") or {}).get("candidates") or []}
            return len(path) < 2 or path[1].split(".")[0] in labs
        if head in ("space", "basis", "orthogonality", "functional_dependencies", "disagreements"):
            return head in geo
        return False
    if src.startswith("analysis:"):
        ana = layers.get("analysis")
        if ana is None:
            return None
        tail = src.split(":", 1)[1]
        if "/" in tail:
            owner, cand = tail.split("/", 1)
            ids = set()
            for f in ana.get("findings") or []:
                if f.get("id") == owner:
                    ids |= {t.get("id") for t in f.get("transformation_candidates") or []}
            for t in ana.get("transcript") or []:
                if f"turn{t.get('turn')}" == owner:
                    ids |= {c.get("id") for c in t.get("transformation_candidates") or []}
            return cand in ids
        if tail.startswith("context") or tail.startswith("modules"):
            return True
        return any(f.get("id") == tail for f in ana.get("findings") or [])
    return False


def stance_conflicts(step, layers):
    """A step that contradicts a recorded analysis stance. The stance wins unless a
    `user:` source overrides it and says so."""
    ana = layers.get("analysis") or {}
    out = []
    cols = set(step.get("columns") or [])
    for s in ana.get("stances") or []:
        if not isinstance(s, dict):
            continue
        scols = set(s.get("columns") or [])
        kind = s.get("kind")
        hit = cols & scols
        if not hit:
            continue
        if kind == "genuine-outliers" and step["op"] in ("winsorize", "clip"):
            out.append(f"{step['op']} on {sorted(hit)} contradicts stance {s.get('id')}: "
                       f"{s.get('assertion')}")
        if kind == "out-of-scope":
            out.append(f"{step['op']} touches {sorted(hit)}, which stance {s.get('id')} "
                       f"declares out of scope")
        if kind == "missingness-mechanism" and step["op"] == "impute":
            strat = (step.get("params") or {}).get("strategy")
            if strat in ("median", "mean", "mode") and "group" in str(s.get("assertion", "")):
                out.append(f"impute {strat} on {sorted(hit)} ignores stance {s.get('id')}: "
                           f"{s.get('assertion')}")
    return out


# --------------------------------------------------------------------------- recipe checking
def check_recipe(rec, layers, out):
    errs = []
    if not isinstance(rec, dict):
        return ["recipe is not a JSON object"]
    if rec.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}, got {rec.get('schema')!r}")
    inp = rec.get("input") or {}
    if not inp.get("path"):
        errs.append("input.path is required")
    steps = rec.get("steps")
    if not isinstance(steps, list) or not steps:
        errs.append("steps must be a non-empty list")
        return errs
    ids, last_phase, split_at = set(), 0, None
    for i, s in enumerate(steps):
        w = f"steps[{i}]"
        if not isinstance(s, dict):
            errs.append(f"{w} must be an object")
            continue
        for k in ("id", "op", "source", "rationale"):
            if not s.get(k):
                errs.append(f"{w} missing {k!r}" + (
                    " — every step cites the geometry element, the analysis finding, the user's "
                    "word or the preset default that justifies it" if k == "source" else ""))
        if s.get("id") in ids:
            errs.append(f"{w}: duplicate step id {s.get('id')!r}")
        ids.add(s.get("id"))
        op = s.get("op")
        if op == "custom":
            ph = (s.get("params") or {}).get("phase")
            if ph not in (1, 2, 3, 4, 5, 6):
                errs.append(f"{w}: a custom step must declare params.phase (1-6) so the executor "
                            "can place it")
            if not str(s.get("source", "")).startswith("user:"):
                errs.append(f"{w}: a custom step's source must be user: — nothing else can "
                            "justify code the catalog does not define")
            if not (s.get("params") or {}).get("code"):
                errs.append(f"{w}: a custom step needs params.code")
            phase = ph if ph in (1, 2, 3, 4, 5, 6) else 4
        elif op not in OP_PHASE:
            errs.append(f"{w}: unknown op {op!r}; see references/step-catalog.md")
            continue
        else:
            phase = OP_PHASE[op]
        if phase < last_phase:
            errs.append(f"{w}: op {op!r} belongs to phase {phase} but follows a phase-"
                        f"{last_phase} step; the phase order is what keeps a fitted step from "
                        "seeing the test part")
        last_phase = max(last_phase, phase)
        if op == "split":
            if split_at is not None:
                errs.append(f"{w}: a recipe carries at most one split")
            split_at = i
        src = str(s.get("source") or "")
        if src and not SOURCE_RE.match(src):
            errs.append(f"{w}: source {src!r} is not one of geometry:… / analysis:F<n>/T<m> / "
                        "analysis:turn<N>/T<m> / user:… / shaper:default")
        elif src and layers:
            r = resolve_source(src, layers)
            if r is False:
                errs.append(f"{w}: source {src!r} names nothing in the model's layers")
            elif r is None:
                out(f"WARN: {w}: source {src!r} cannot be resolved (the model carries no "
                    f"{src.split(':')[0]} layer)")
        if s.get("source") == "shaper:default" and not s.get("alternatives"):
            out(f"WARN: {w}: a shaper:default step should carry at least one alternative — "
                "it is a choice the recipe made on its own")
        cons = s.get("consequences") or {}
        for k in ("rows", "columns", "downstream"):
            if k not in cons:
                out(f"WARN: {w}: consequences.{k} missing")
        for c in stance_conflicts(s, layers):
            if str(s.get("source", "")).startswith("user:"):
                out(f"WARN: {w}: {c} — allowed because a user: step overrides it")
            else:
                errs.append(f"{w}: {c}")
    # fitted steps before the split
    if split_at is not None:
        for i, s in enumerate(steps[:split_at]):
            if isinstance(s, dict) and s.get("op") in FITTED_OPS:
                errs.append(f"steps[{i}]: {s.get('op')!r} is a fitted step placed before the "
                            "split; it would be fitted on rows that end up in the test part")
    else:
        for s in steps:
            if isinstance(s, dict) and s.get("op") == "encode" and \
                    (s.get("params") or {}).get("strategy") == "target":
                errs.append("target encoding without a split: it needs out-of-fold fitting, "
                            "which the executor performs only when a split precedes it")
    errs.extend(leakage_errors(steps, split_at))
    # structural rules against the geometry layer
    geo = layers.get("geometry")
    if geo:
        label = None
        for s in steps:
            if isinstance(s, dict) and s.get("op") == "select_partition":
                label = (s.get("params") or {}).get("label")
        cand = None
        for c in (geo.get("partitions") or {}).get("candidates") or []:
            if c.get("label") == label:
                cand = c
        if label and cand is None:
            errs.append(f"select_partition names {label!r}, which the geometry layer does not "
                        "offer as a candidate")
        dropped = set()
        kept = set()
        for s in steps:
            if not isinstance(s, dict):
                continue
            if s.get("op") in ("drop_derived",):
                kept |= set((s.get("params") or {}).get("keep") or [])
            if s.get("op") == "keep_columns":
                kept |= set((s.get("params") or {}).get("columns") or [])
            if s.get("op") in ("drop_identity", "drop_constant", "drop_leakage", "select_features"):
                dropped |= set(s.get("columns") or [])
        if cand:
            leak = set(cand.get("dropped_for_leakage") or [])
            has_drop_leak = any(isinstance(s, dict) and s.get("op") == "drop_leakage"
                                for s in steps)
            if leak and not has_drop_leak:
                errs.append(f"the chosen partition holds back {sorted(leak)} as its leakage set, "
                            "but the recipe has no drop_leakage step")
            if kept & leak:
                errs.append(f"keep_columns retains leakage column(s) {sorted(kept & leak)}")
        basis = set((geo.get("basis") or {}).get("members") or [])
        for s in steps:
            if not isinstance(s, dict) or s.get("op") not in ("drop_identity", "drop_constant",
                                                              "select_features"):
                continue
            hit = basis & set(s.get("columns") or [])
            if hit and not str(s.get("source", "")).startswith("user:"):
                errs.append(f"step {s.get('id')} drops basis member(s) {sorted(hit)} without a "
                            "user: step saying so")
        # a transform on one member of a kept derivation cycle
        heads = {d.get("column"): set(d.get("body") or []) for d in kept_derivations(geo)}
        drops_derived = any(isinstance(s, dict) and s.get("op") == "drop_derived" for s in steps)
        for s in steps:
            if not isinstance(s, dict) or s.get("op") != "transform":
                continue
            touched = set(s.get("columns") or [])
            for head, body in heads.items():
                if drops_derived and head not in kept:
                    continue                      # the head leaves; the relation leaves with it
                fam = body | {head}
                if touched & fam and not fam <= touched:
                    errs.append(f"step {s.get('id')}: transform touches {sorted(touched & fam)} "
                                f"but not the whole derivation {head} = f({', '.join(sorted(body))}); "
                                "a transform on one member alone breaks the relation")
    return errs + split_key_errors(steps, split_at) + graph_step_errors(steps, split_at)


def planned_label(steps):
    """The label the recipe's `select_partition` names (the last one, as the executor reads it)."""
    label = None
    for s in steps:
        if isinstance(s, dict) and s.get("op") == "select_partition":
            label = (s.get("params") or {}).get("label")
    return label


def _names(v):
    """A grouping parameter as a list of column names (a lone name is one column)."""
    v = v or []
    return [v] if isinstance(v, str) else [b for b in v if isinstance(b, str)]


def impute_by(step):
    """An `impute` step's grouping columns, as a list (a lone column name is one column)."""
    return _names((step.get("params") or {}).get("by"))


def lag_by(step):
    """A `lag` step's grouping columns (`params.group_by`), as a list."""
    return _names((step.get("params") or {}).get("group_by"))


def label_grouping_error(step, label):
    """The refusal of an `impute` or a `lag` grouped by the label, or None (A47(5)). Grouped by
    its own label, a test row's filled value is the statistic of the rows that share that
    label, and its lagged value is read from the previous row that shares it: either way the
    label is carried into the row's features, the evaluation scores an answer it already
    holds, and a served query, which has no label, cannot be given the same value. The
    message names the step and the label."""
    if label is None:
        return None
    op, sid = step.get("op"), step.get("id")
    if op == "impute" and label in impute_by(step):
        return (f"impute grouped by the label {label!r} — a row's filled value would be the "
                "statistic of the rows sharing its label, so the label reaches its features (an "
                "evaluation the served answer cannot reproduce: a query has no label); group by "
                "a feature, or impute without a group. Refused whatever the source, a user: step "
                "included")
    if op == "lag" and label in lag_by(step):
        return (f"lag {sid} grouped by the label {label!r} — a row's lagged value would be read "
                "from the previous row sharing its label, so the label reaches its features (an "
                "evaluation the served answer cannot reproduce: a query has no label to group "
                "its history by); group by the series key, or lag without a group. Refused "
                "whatever the source, a user: step included")
    return None


IMPORTANCE_AFTER_SPLIT = (
    "select_features kind=importance after a split — the analysis layer's permutation "
    "importance was measured before the split, over rows that are now the test part, so a "
    "selection by it has seen the test labels; select by variance (measured on the train "
    "part) or name the columns (kind=explicit)")


def leakage_errors(steps, split_at):
    """The leakage rules the phase order alone does not catch (`shape-contract.md` §1): an
    `impute` or a `lag` grouped by the label, a `select_features kind=importance` in a recipe
    that splits, and a pre-split `custom` step that declares it fits (`params.fitted: true`) — a fitted
    pre-split step is fitted on the future test rows. The executor holds the same rules where
    it runs the steps (and probes every pre-split `custom` step for row-locality, which no
    static reading can decide)."""
    errs = []
    label = planned_label(steps)
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        p = s.get("params") or {}
        why = label_grouping_error(s, label)
        if why:
            errs.append(f"steps[{i}] {s.get('id')}: {why}")
        if split_at is None:
            continue
        if s.get("op") == "select_features" and p.get("kind", "importance") == "importance":
            errs.append(f"steps[{i}]: {IMPORTANCE_AFTER_SPLIT}")
        if s.get("op") == "custom" and i < split_at and p.get("fitted") is True:
            errs.append(f"steps[{i}]: a custom step before the split declares fitted: true — "
                        "fitted there, it would be fitted on the rows that become the test "
                        "part; give it a phase after the split (it then runs per part)")
    return errs


def split_key(steps):
    """(the group split's key, kept as the graph key) — `split kind=group`'s `by` and whether a
    `graph_edges` step at level graph names it as its `graph_id` (then it reaches the shaped
    parts with `role: key`); (None, False) when the recipe has no group split (F33)."""
    split = next((s for s in steps if isinstance(s, dict) and s.get("op") == "split"), None)
    p = (split or {}).get("params") or {}
    if p.get("kind") != "group" or not isinstance(p.get("by"), str) or not p.get("by"):
        return None, False
    gids = {(s.get("params") or {}).get("graph_id") for s in steps
            if isinstance(s, dict) and s.get("op") == "graph_edges"}
    return p["by"], p["by"] in gids


# Why a group split's key never reaches the shaped parts as a feature (F33). The split puts
# each group whole into one part, so every valid and test row carries a group the fit never
# saw: encoded, the column is information in train alone and the reserved unseen value
# everywhere else, which the network learns from and the evaluation cannot score.
SPLIT_KEY_WHY = ("every valid and test group is one the train part never saw, so as a feature "
                 "the key would carry information in train alone and a reserved unseen value "
                 "everywhere else")
# The line the executor emits after a group split, before it drops the key: the reproduction
# script runs it with the split, the transform-only module (which skips the split) runs the
# lines from it onward, so a served row loses the key exactly as the shaped parts did.
SPLIT_KEY_MARK = "# the group split's key leaves the features here"


def split_key_errors(steps, split_at):
    """A step after a group split that reads the split's key (`shape-contract.md` §1, F33):
    the executor drops the key at the split, so a value or representation step that names it
    (an `encode`, a `scale`, a `bin`, a `lag` or an `impute` grouped by it, a `keep` in
    `select_features`) would make it a feature again or read a column that is gone. At level
    graph the key stays as the graph key (`role: key`), and only `graph_edges` may name it."""
    key, kept = split_key(steps)
    if key is None or split_at is None:
        return []
    errs = []
    for i, s in enumerate(steps[split_at + 1:], start=split_at + 1):
        if not isinstance(s, dict) or s.get("op") == "graph_edges" or key not in step_reads(s):
            continue
        errs.append(f"steps[{i}] {s.get('id')}: {s.get('op')} reads {key!r}, the group split's "
                    f"key — {SPLIT_KEY_WHY}. "
                    + ("It stays only as the graph key (role key) of the graph_edges step"
                       if kept else "The executor drops it at the split")
                    + f"; take {key!r} out of this step (it is never a feature, whatever the "
                      "source)")
    return errs


GRAPH_METHODS = ("key_pair", "knn", "spatial")


def _positive_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def graph_step_errors(steps, split_at):
    """The structural rules of `graph_edges` (`shape-contract.md` §1, `step-catalog.md` phase 7).

    Edges are a claim, not a default: the step needs an `analysis:` or `user:` source, a split
    before it (its edges record the part of both ends, and a kNN index is a train-only estimate
    like any fitted parameter), and, at level graph, a group split on the graph key — a graph
    that straddles parts would put one molecule in train and in test. The parameters are
    checked here, before any data moves; the columns they name are checked by the executor,
    which has the frame."""
    errs = []
    found = [(i, s) for i, s in enumerate(steps)
             if isinstance(s, dict) and s.get("op") == "graph_edges"]
    if len(found) > 1:
        errs.append(f"steps[{found[1][0]}]: a recipe carries at most one graph_edges step — the "
                    "edges part is one relation, recorded once")
    first = {}
    for i, s in enumerate(steps):
        if isinstance(s, dict):
            first.setdefault(s.get("op"), i)
    split = steps[split_at] if split_at is not None else None
    for i, s in found:
        w = f"steps[{i}]"
        src = str(s.get("source") or "")
        if src == "shaper:default":
            errs.append(f"{w}: graph_edges cannot be a shaper:default step — that rows are related "
                        "is a claim only an analysis: finding or a user: turn can make")
        elif not src.startswith(("analysis:", "user:")):
            errs.append(f"{w}: graph_edges needs an analysis: or user: source, not {src!r} — "
                        "that rows are related is a claim only a finding or a user turn makes")
        if split_at is None:
            errs.append(f"{w}: graph_edges requires a split — every edge records the part of both "
                        "its ends, and a recipe without parts has nothing to record")
        elif i < split_at:
            errs.append(f"{w}: graph_edges must come after the split (steps[{split_at}]): its edges "
                        "record which part each end is in, and a knn index built before the split "
                        "would be fitted on the test part")
        errs.extend(f"{w}: {e}" for e in graph_param_errors(s, i, first))
        gid = (s.get("params") or {}).get("graph_id")
        sp = (split.get("params") or {}) if isinstance(split, dict) else {}
        if isinstance(gid, str) and gid and split is not None and \
                not (sp.get("kind") == "group" and sp.get("by") == gid):
            errs.append(f"{w}: a graph_id recipe needs a group split by {gid!r} (split kind=group, "
                        f"by={gid}) — a graph whose rows straddle parts would be trained on and "
                        "tested on at once")
    return errs


def graph_param_errors(step, at, first):
    """A graph_edges step's parameters, per method; `first` maps an op to its first index."""
    p = step.get("params") or {}
    method, errs = p.get("method"), []
    if method not in GRAPH_METHODS:
        return [f"graph_edges method must be one of {list(GRAPH_METHODS)}, got {method!r}"]
    for key in ("symmetric", "self_loops"):
        if key in p and not isinstance(p[key], bool):
            errs.append(f"graph_edges params.{key} must be true or false")
    if p.get("max_degree") is not None and not _positive_int(p.get("max_degree")):
        errs.append("graph_edges params.max_degree must be a positive integer")
    weight, gid = p.get("weight"), p.get("graph_id")
    if weight is not None and not isinstance(weight, str):
        errs.append("graph_edges params.weight is a column name, \"distance\" or null")
    if weight == "distance" and method == "key_pair":
        errs.append("weight \"distance\" needs a metric — knn or spatial; a key_pair edge weighs a "
                    "column of its source row, or nothing")
    if gid is not None and not (isinstance(gid, str) and gid):
        errs.append("graph_id is a column name or null")
    if method == "key_pair":
        errs.extend(keypair_param_errors(p, p.get("columns") or step.get("columns")))
    elif method == "knn":
        if not _positive_int(p.get("k", 5)):
            errs.append("knn params.k must be a positive integer")
        if p.get("metric", "euclidean") not in ("euclidean", "cosine"):
            errs.append("knn params.metric must be euclidean or cosine")
        feats = p.get("features", "basis")
        if feats != "basis" and not (isinstance(feats, list) and feats and
                                     all(isinstance(c, str) for c in feats)):
            errs.append("knn params.features is \"basis\" or a list of column names")
    else:
        errs.extend(spatial_param_errors(p, at, first))
    return errs


def keypair_param_errors(p, cols):
    """key_pair's columns and caps. A cap ranks a node's candidates by raw position, which a
    served row does not carry: at level node edges_for could not cap a query's attachments the
    way the edges part did, so the evaluation and the bundle would see different graphs."""
    max_clique = p.get("max_clique")
    if not isinstance(cols, list) or len(cols) not in (1, 2) or \
            not all(isinstance(c, str) and c for c in cols):
        return ["key_pair needs params.columns: [src_col, dst_col] (a foreign key) or [col] "
                "(rows sharing a value form a clique)"]
    if not p.get("graph_id") and (p.get("max_degree") is not None or max_clique is not None):
        return ["a key_pair cap (max_degree / max_clique) needs a graph_id: at level node it "
                "ranks candidates by raw position, which a served row does not carry, so "
                "edges_for could not reproduce the capped edges the evaluation sees — drop the "
                "cap, or relate the rows within graphs"]
    if max_clique is not None and len(cols) != 1:
        return ["max_clique caps a one-column clique; a foreign key pair is capped by max_degree"]
    if max_clique is not None and not (_positive_int(max_clique) and max_clique >= 2):
        return ["max_clique must be an integer >= 2"]
    return []


def spatial_param_errors(p, at, first):
    errs = []
    has_k, has_r = p.get("k") is not None, p.get("radius_m") is not None
    if has_k == has_r:
        errs.append("spatial takes exactly one of k and radius_m")
    elif has_k and not _positive_int(p["k"]):
        errs.append("spatial params.k must be a positive integer")
    elif has_r and not (isinstance(p["radius_m"], (int, float)) and
                        not isinstance(p["radius_m"], bool) and p["radius_m"] > 0):
        errs.append("spatial params.radius_m must be a positive number of metres")
    if first.get("parse_geometry", at) >= at or first.get("spatial_reproject", at) >= at:
        errs.append("spatial graph_edges needs parse_geometry and spatial_reproject before it — a "
                    "neighbour radius in degrees is not a distance, and the metric CRS is the one "
                    "spatial_reproject recorded")
    return errs


# --------------------------------------------------------------------------- execution state
class Ctx:
    def __init__(self, df, recipe, layers, seed):
        self.parts = {"all": df}
        # The input as read, untouched: every op copies before it writes, so this is still the
        # raw file when phase 7 reads a key, a weight or a coordinate at a row's raw position.
        # Held only by a recipe with a graph step: past the split nothing else reads the whole
        # input, and keeping it alive would double the peak memory of every large recipe.
        self.raw = df if any(isinstance(s, dict) and s.get("op") == "graph_edges"
                             for s in recipe.get("steps") or []) else None
        self.edges = None
        self.graph = None
        self.order = ["all"]
        self.recipe = recipe
        self.layers = layers
        self.seed = seed
        self.label = None
        # The label is known before the first step runs: the executor pre-scans the recipe for
        # select_partition. Nothing drops the target on its way past — a rule head that happens
        # to be the label (late = delivered_days > 7) is the thing being predicted, not a
        # redundant column. Only an explicit user: step may remove it.
        self.label_planned = None
        self.active_orientation = {}
        self.crs = None
        self.lineage = {c: [] for c in df.columns}
        self.removed = {}
        self.manifest = []
        self.code = []
        # The same lines, sliced per step. `--transform-only` re-emits a filtered subset of
        # them over a single part; without the slice it would have to parse the reproduction
        # script back out, or own a second implementation of every op.
        self.code_by_step = {}
        self.removed_rows = []
        self.fitted = {}

    @property
    def fit_part(self):
        return "train" if "train" in self.parts else "all"

    def each(self):
        return [(k, self.parts[k]) for k in self.order]

    def set_all(self, fn):
        for k in self.order:
            self.parts[k] = fn(self.parts[k])

    def cols(self):
        return list(self.parts[self.order[0]].columns)

    def touch(self, step, columns):
        for c in columns:
            self.lineage.setdefault(c, []).append(step["id"])

    def emit(self, *lines):
        self.code.extend(lines)


def derivation_kept(d) -> bool:
    """A derivation the geometry layer keeps (dataset-forge report-contract §2, F44): the semantic
    channel confirms it and it is not set aside with `kept: false`. A rejected derivation stays in
    the layer with its triple and the consequence block that says why, but only a kept rule
    derives a column: its head is a basis member like any other — never an active head to drop,
    never a relation a transform on one member would break, never a rule to recompute."""
    if not isinstance(d, dict):
        return False
    semantic = ((d.get("provenance") or {}).get("semantic") or {}).get("status")
    return semantic == "confirmed" and d.get("kept", True) is not False


def kept_derivations(geo) -> list:
    """The geometry layer's kept derivations, in the layer's order (`derivation_kept`)."""
    return [d for d in (geo or {}).get("derivations") or [] if derivation_kept(d)]


def geo_of(ctx):
    return ctx.layers.get("geometry") or {}


def py(v):
    """A Python literal for the generated reproduction script. json.dumps would write null,
    true and false, which are not Python — repr is the correct serializer here."""
    return repr(v)


# --------------------------------------------------------------------------- phase 1
def _nominal_text(series):
    """A nominal column as text; whole floats as the integers they are (see op_retype)."""
    if pd.api.types.is_float_dtype(series) and bool((series.dropna() % 1 == 0).all()):
        return series.astype("Int64").astype("string")
    return series.astype("string")


def op_retype(ctx, step):
    to = (step.get("params") or {}).get("to")
    order = (step.get("params") or {}).get("order")
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    for c in cols:
        def conv(d, c=c):
            d = d.copy()
            if to == "numeric":
                d[c] = pd.to_numeric(d[c], errors="coerce")
            elif to in ("nominal", "text"):
                d[c] = _nominal_text(d[c])
            elif to == "ordinal":
                d[c] = pd.Categorical(d[c].astype("string"),
                                      categories=[str(x) for x in (order or [])], ordered=True)
            elif to == "boolean":
                d[c] = d[c].astype("boolean")
            elif to == "datetime":
                d[c] = pd.to_datetime(d[c], errors="coerce")
            return d
        ctx.set_all(conv)
        if to == "numeric":
            ctx.emit(f"for k in parts: parts[k][{py(c)}] = pd.to_numeric(parts[k][{py(c)}], errors='coerce')")
        elif to in ("nominal", "text"):
            # a code read as float (a CSV with a gap in it turns 80 into 80.0) is the same
            # category as the integer it was fitted on: whole floats become integer text,
            # everything else is left as the text it is
            ctx.emit(f"for k in parts: parts[k][{py(c)}] = (parts[k][{py(c)}].astype('Int64')"
                     f".astype('string') if pd.api.types.is_float_dtype(parts[k][{py(c)}]) and "
                     f"bool((parts[k][{py(c)}].dropna() % 1 == 0).all()) else "
                     f"parts[k][{py(c)}].astype('string'))")
        elif to == "ordinal":
            ctx.emit(f"for k in parts: parts[k][{py(c)}] = pd.Categorical("
                     f"parts[k][{py(c)}].astype('string'), categories={py([str(x) for x in (order or [])])}, ordered=True)")
        elif to == "boolean":
            ctx.emit(f"for k in parts: parts[k][{py(c)}] = parts[k][{py(c)}].astype('boolean')")
        elif to == "datetime":
            ctx.emit(f"for k in parts: parts[k][{py(c)}] = pd.to_datetime(parts[k][{py(c)}], errors='coerce')")
    ctx.touch(step, cols)
    return {"retyped": {c: to for c in cols}}


def _drop(ctx, step, cols, why):
    cols = [c for c in cols if c in ctx.cols()]
    target = ctx.label or ctx.label_planned
    # A user: step may drop the target — but only by naming it. "Explicit" means written down:
    # a user step about something else must not carry the label away with it.
    named_explicitly = target in (step.get("columns") or [])
    if target and target in cols and not (str(step.get("source", "")).startswith("user:")
                                          and named_explicitly):
        cols = [c for c in cols if c != target]
        mark(f"OK: {step['id']} keeps {target!r}: it is the label, not a droppable column")
    if not cols:
        return {"dropped": []}
    ctx.set_all(lambda d: d.drop(columns=[c for c in cols if c in d.columns]))
    for c in cols:
        ctx.removed[c] = {"step": step["id"], "op": step["op"], "why": why}
    ctx.emit(f"for k in parts: parts[k] = parts[k].drop(columns=[c for c in {py(cols)} if c in parts[k].columns])")
    ctx.touch(step, cols)
    return {"dropped": cols}


def op_drop_identity(ctx, step):
    cols = step.get("columns") or [t["column"] for t in geo_of(ctx).get("typing") or []
                                   if t.get("role") in ("identity", "key")]
    return _drop(ctx, step, cols, "identity / key")


def op_drop_constant(ctx, step):
    tol = float((step.get("params") or {}).get("tolerance", 0.0))
    cols = step.get("columns") or []
    if not cols:
        d = ctx.parts[ctx.fit_part]
        for c in d.columns:
            vc = d[c].value_counts(normalize=True, dropna=True)
            if len(vc) <= 1 or (len(vc) and vc.iloc[0] >= 1.0 - tol):
                cols.append(c)
    return _drop(ctx, step, cols, f"constant or quasi-constant (tolerance {tol})")


def op_dedupe(ctx, step):
    p = step.get("params") or {}
    on = p.get("on", "all")
    keep = p.get("keep", "first")
    subset = None if on == "all" else [c for c in (on if isinstance(on, list) else [on])
                                       if c in ctx.cols()]
    before = {k: len(v) for k, v in ctx.each()}
    # The index still carries the raw file's row positions here (dedupe is phase 1, and nothing
    # before it renumbers). Recording what left is what lets a later reader — the twin check
    # above all — map a shaped row back to the raw row it came from.
    was = {k: set(v.index) for k, v in ctx.each()}
    ctx.set_all(lambda d: d.drop_duplicates(subset=subset, keep=keep))
    after = {k: len(v) for k, v in ctx.each()}
    gone = sorted(int(i) for k, v in ctx.each() for i in (was[k] - set(v.index)))
    ctx.removed_rows.extend(gone)
    ctx.emit(f"for k in parts: parts[k] = parts[k].drop_duplicates(subset={py(subset)}, keep={py(keep)})")
    return {"rows_removed": {k: before[k] - after[k] for k in before}, "raw_rows_removed": gone}


def op_parse_datetime(ctx, step):
    p = step.get("params") or {}
    fmt, errors = p.get("format"), p.get("errors", "coerce")
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    coerced = {}
    for c in cols:
        before = ctx.parts[ctx.fit_part][c].notna().sum()
        def conv(d, c=c):
            d = d.copy()
            d[c] = pd.to_datetime(d[c], format=fmt, errors=errors)
            return d
        ctx.set_all(conv)
        coerced[c] = int(before - ctx.parts[ctx.fit_part][c].notna().sum())
        ctx.emit(f"for k in parts: parts[k][{py(c)}] = pd.to_datetime(parts[k][{py(c)}], "
                 f"format={py(fmt)}, errors={py(errors)})")
    ctx.touch(step, cols)
    return {"coerced_to_missing": coerced}


def op_parse_geometry(ctx, step):
    p = step.get("params") or {}
    crs = p.get("crs")
    if not crs:
        raise ValueError("parse_geometry needs a declared crs — without one no distance or "
                         "reprojection below it means anything")
    ctx.crs = crs
    kind, lat, lon = p.get("kind", "latlon"), p.get("lat"), p.get("lon")
    if kind == "latlon":
        for c in (lat, lon):
            if c not in ctx.cols():
                raise ValueError(f"parse_geometry: column {c!r} is not in the frame")
        d = ctx.parts[ctx.fit_part]
        la, lo = pd.to_numeric(d[lat], errors="coerce"), pd.to_numeric(d[lon], errors="coerce")
        bad = int((~la.between(-90, 90)).sum() + (~lo.between(-180, 180)).sum())
        if bad:
            mark(f"WARN: parse_geometry: {bad} coordinate value(s) fall outside the declared CRS "
                 f"bounds")
    ctx.emit(f"# geometry declared: kind={kind}, crs={crs} (no column change)")
    ctx.touch(step, [c for c in (lat, lon) if c])
    return {"crs": crs, "kind": kind, "lat": lat, "lon": lon}


# --------------------------------------------------------------------------- phase 2
def op_orient_cycle(ctx, step):
    p = step.get("params") or {}
    cid, oid = p.get("cycle_id"), p.get("orientation_id")
    if not oid:
        for c in geo_of(ctx).get("cycles") or []:
            if c.get("id") == cid:
                for o in c.get("orientations") or []:
                    if o.get("default"):
                        oid = o.get("id")
    ctx.active_orientation[cid] = oid
    ctx.emit(f"# cycle {cid}: orientation {oid} (decides which member is derived below)")
    return {"cycle": cid, "orientation": oid}


def active_rule_heads(ctx):
    """The columns that are the head of an active rule, under the chosen orientations."""
    geo = geo_of(ctx)
    heads = {d.get("column") for d in kept_derivations(geo)}
    for c in geo.get("cycles") or []:
        chosen = ctx.active_orientation.get(c.get("id"))
        for o in c.get("orientations") or []:
            if chosen and o.get("id") != chosen:
                continue
            if not chosen and not o.get("default"):
                continue
            basis = set(o.get("basis") or [])
            for m in c.get("members") or []:
                if m in basis:
                    heads.discard(m)
            for r in o.get("rules") or []:
                if isinstance(r, dict) and r.get("column"):
                    heads.add(r["column"])
    return heads


def op_drop_derived(ctx, step):
    keep = set((step.get("params") or {}).get("keep") or [])
    heads = sorted(h for h in active_rule_heads(ctx) if h and h not in keep)
    res = _drop(ctx, step, heads, "active rule head (the basis keeps its parents)")
    res["kept_derived"] = sorted(keep)
    if keep:
        mark(f"WARN: derived column(s) {sorted(keep)} kept by request: they are redundant with "
             "their parents by construction")
    return res


def op_select_partition(ctx, step):
    p = step.get("params") or {}
    ctx.label = p.get("label")
    if ctx.label not in ctx.cols():
        raise ValueError(f"select_partition: the label {ctx.label!r} is not in the frame")
    ctx.emit(f"LABEL = {py(ctx.label)}")
    return {"label": ctx.label, "task": p.get("task")}


def op_drop_leakage(ctx, step):
    cols = step.get("columns") or []
    if not cols:
        for c in (geo_of(ctx).get("partitions") or {}).get("candidates") or []:
            if c.get("label") == ctx.label:
                cols = list(c.get("dropped_for_leakage") or [])
    return _drop(ctx, step, cols, f"derives the label {ctx.label!r}")


def op_keep_columns(ctx, step):
    cols = [c for c in ((step.get("params") or {}).get("columns") or step.get("columns") or [])]
    keep = [c for c in ctx.cols() if c in cols or c == ctx.label]
    drop = [c for c in ctx.cols() if c not in keep]
    ctx.set_all(lambda d: d[[c for c in keep if c in d.columns]])
    for c in drop:
        ctx.removed[c] = {"step": step["id"], "op": step["op"], "why": "not in keep_columns"}
    ctx.emit(f"for k in parts: parts[k] = parts[k][[c for c in {py(keep)} if c in parts[k].columns]]")
    return {"kept": keep, "dropped": drop}


# --------------------------------------------------------------------------- phase 3
def _sparse_strata(groups, test: float, valid: float) -> str | None:
    """Why `groups` (stratum -> rows) cannot be stratified, or None: more than half the strata
    hold fewer rows than it takes for each to place one row in the smaller held-out part."""
    small = [f for f in (test, valid) if f > 0]
    need = int(np.ceil(1 / min(small))) if small else 1
    few = sum(1 for rows in groups.values() if len(rows) < need)
    if few * 2 > len(groups):
        return f"{few} of its {len(groups)} strata hold fewer than {need} rows"
    return None


def op_split(ctx, step):
    p = step.get("params") or {}
    kind = p.get("kind", "random")
    test = float(p.get("test", 0.2))
    valid = float(p.get("valid", 0.0))
    df = ctx.parts["all"]
    rng = np.random.default_rng(ctx.seed)
    idx = np.arange(len(df))
    if kind == "time":
        tcol = p.get("time_column")
        if tcol not in df.columns:
            raise ValueError(f"split kind=time needs an existing time_column; {tcol!r} is not here")
        order = pd.to_datetime(df[tcol], errors="coerce").sort_values(kind="stable").index
        pos = df.index.get_indexer(order)
        n = len(pos)
        n_test = int(round(test * n))
        n_valid = int(round(valid * n))
        parts = {"train": pos[: n - n_test - n_valid], "valid": pos[n - n_test - n_valid: n - n_test],
                 "test": pos[n - n_test:]}
        ctx.emit(f"_order = pd.to_datetime(parts['all'][{py(tcol)}], errors='coerce').sort_values(kind='stable').index",
                 "_pos = parts['all'].index.get_indexer(_order); _n = len(_pos)",
                 f"_nt = int(round({test} * _n)); _nv = int(round({valid} * _n))",
                 "parts = {'train': parts['all'].iloc[_pos[:_n-_nt-_nv]], "
                 "'valid': parts['all'].iloc[_pos[_n-_nt-_nv:_n-_nt]], "
                 "'test': parts['all'].iloc[_pos[_n-_nt:]]}")
    else:
        if kind == "stratified":
            by = p.get("by") or ctx.label
            if by not in df.columns:
                raise ValueError(f"split kind=stratified needs an existing 'by' column ({by!r})")
            groups = df.groupby(df[by].astype(str), sort=True).indices
            sparse = _sparse_strata(groups, test, valid)
            if sparse:
                raise ValueError(f"split kind=stratified by {by!r}: {sparse} — a stratum that small "
                                 "rounds out of test and valid, so the parts would not hold the "
                                 "strata (a continuous column has one row per value). Split a "
                                 "regression target `random` (or `time`/`group`), or stratify on "
                                 "a class column")
            tr, va, te = [], [], []
            for g in sorted(groups):
                gi = np.sort(groups[g])
                gi = rng.permutation(gi)
                n = len(gi)
                n_test, n_valid = int(round(test * n)), int(round(valid * n))
                te.extend(gi[:n_test]); va.extend(gi[n_test:n_test + n_valid])
                tr.extend(gi[n_test + n_valid:])
            parts = {"train": np.sort(tr), "valid": np.sort(va), "test": np.sort(te)}
        elif kind == "group":
            by = p.get("by")
            if by not in df.columns:
                raise ValueError(f"split kind=group needs an existing 'by' column ({by!r})")
            keys = pd.unique(df[by].astype(str))
            keys = np.sort(keys)
            perm = rng.permutation(len(keys))
            keys = keys[perm]
            n = len(keys)
            n_test, n_valid = int(round(test * n)), int(round(valid * n))
            kt, kv = set(keys[:n_test]), set(keys[n_test:n_test + n_valid])
            col = df[by].astype(str).values
            parts = {"train": np.where(~np.isin(col, list(kt | kv)))[0],
                     "valid": np.where(np.isin(col, list(kv)))[0],
                     "test": np.where(np.isin(col, list(kt)))[0]}
        else:
            perm = rng.permutation(idx)
            n = len(perm)
            n_test, n_valid = int(round(test * n)), int(round(valid * n))
            parts = {"train": np.sort(perm[n_test + n_valid:]), "valid": np.sort(perm[n_test:n_test + n_valid]),
                     "test": np.sort(perm[:n_test])}
        ctx.emit(f"# split kind={kind} seed={ctx.seed}: the index sets are stored in the manifest",
                 "_sets = MANIFEST_SPLIT",
                 "parts = {k: parts['all'].iloc[v].copy() for k, v in _sets.items() if len(v)}")
    out = {}
    for k in ("train", "valid", "test"):
        v = np.asarray(parts.get(k, []), dtype=int)
        if len(v):
            out[k] = v
    ctx.parts = {k: df.iloc[v].copy() for k, v in out.items()}
    ctx.order = [k for k in ("train", "valid", "test") if k in ctx.parts]
    key = drop_split_key(ctx, step) if kind == "group" else None
    # `indices` are positions in the frame the split saw, which is what the reproduction
    # script's `.iloc` needs. `raw_indices` are positions in the *raw file* — the same numbers
    # unless a dedupe ran, and the only ones with which the twin check can fetch the raw rows
    # a shaped part came from.
    raw = {k: [int(x) for x in df.index[v]] for k, v in out.items()}
    return {"kind": kind, "sizes": {k: int(len(v)) for k, v in out.items()},
            "indices": {k: [int(x) for x in v] for k, v in out.items()},
            "raw_indices": raw,
            "by": p.get("by"), "time_column": p.get("time_column"),
            **({"key": key} if key else {})}


def drop_split_key(ctx, step):
    """After a group split the key leaves the frame (F33, `SPLIT_KEY_WHY`), unless a
    `graph_edges` step at level graph names it as its `graph_id`: then it stays, as the graph
    key (`role: key` in `output_schema`), never as a feature. Returns the split's `key` record:
    `{column, fate: dropped | graph key, why}`."""
    by, kept = split_key(ctx.recipe.get("steps") or [])
    if by is None:
        return None
    if kept:
        mark(f"OK: {step['id']} split: {by!r} stays as the graph key (role key), not a feature")
        return {"column": by, "fate": "graph key", "why": "the graph_edges step's graph_id"}
    why = f"the group split's key: {SPLIT_KEY_WHY}"
    ctx.set_all(lambda d: d.drop(columns=[c for c in (by,) if c in d.columns]))
    ctx.removed[by] = {"step": step["id"], "op": "split", "why": why}
    ctx.emit(f"{SPLIT_KEY_MARK}: {by!r}",
             f"for k in parts: parts[k] = parts[k].drop(columns=[c for c in {py([by])} "
             "if c in parts[k].columns])")
    ctx.touch(step, [by])
    mark(f"OK: {step['id']} split: the group key {by!r} leaves the features ({SPLIT_KEY_WHY})")
    return {"column": by, "fate": "dropped", "why": why}


# --------------------------------------------------------------------------- phase 4 (values)
def op_impute(ctx, step):
    p = step.get("params") or {}
    strat = p.get("strategy", "median")
    by = impute_by(step)
    ind = bool(p.get("indicator"))
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    target = ctx.label or ctx.label_planned
    if target and target in cols:
        raise ValueError(f"impute: the label {target!r} is never imputed")
    why = label_grouping_error(step, target)
    if why:
        raise ValueError(f"impute {step.get('id')}: {why}")
    fit = ctx.parts[ctx.fit_part]
    fitted = {}
    for c in cols:
        if strat in ("median", "mean", "mode", "constant"):
            if strat == "median":
                v = float(pd.to_numeric(fit[c], errors="coerce").median())
            elif strat == "mean":
                v = float(pd.to_numeric(fit[c], errors="coerce").mean())
            elif strat == "mode":
                m = fit[c].mode(dropna=True)
                v = (m.iloc[0] if len(m) else None)
            else:
                v = p.get("value")
            fitted[c] = {"fill": jsonable(v)}
        elif strat in ("group-median", "group-mode"):
            g = fit.groupby([fit[b].astype(str) for b in by], dropna=False)[c]
            table = (g.median() if strat == "group-median" else g.agg(
                lambda s: s.mode().iloc[0] if len(s.mode()) else None))
            glob = (float(pd.to_numeric(fit[c], errors="coerce").median())
                    if strat == "group-median"
                    else (fit[c].mode().iloc[0] if len(fit[c].mode()) else None))
            fitted[c] = {"by": by, "table": {str(k): jsonable(v) for k, v in table.items()},
                         "fallback": jsonable(glob)}
        else:
            raise ValueError(f"impute strategy {strat!r} is not implemented by this executor")
    def apply(d):
        d = d.copy()
        for c in cols:
            if ind:
                d[c + "_missing"] = d[c].isna().astype("int8")
            if strat in ("median", "mean", "mode", "constant"):
                d[c] = d[c].fillna(fitted[c]["fill"])
            else:
                keys = d[by[0]].astype(str) if len(by) == 1 else \
                    d[by].astype(str).agg(tuple, axis=1).astype(str)
                filled = keys.map(fitted[c]["table"]).astype(object)
                d[c] = d[c].fillna(pd.Series(filled.values, index=d.index)).fillna(
                    fitted[c]["fallback"])
        return d
    ctx.set_all(apply)
    if ind:
        for c in cols:
            ctx.lineage.setdefault(c + "_missing", []).append(step["id"])
    ctx.emit(f"_f = FITTED[{py(step['id'])}]['fitted']")
    for c in cols:
        if ind:
            ctx.emit(f"for k in parts: parts[k][{py(c + '_missing')}] = parts[k][{py(c)}].isna().astype('int8')")
        if strat in ("median", "mean", "mode", "constant"):
            ctx.emit(f"for k in parts: parts[k][{py(c)}] = parts[k][{py(c)}].fillna(_f[{py(c)}]['fill'])")
        else:
            ctx.emit("for k in parts:",
                     f"    _keys = parts[k][{py(by[0])}].astype(str) if {len(by)} == 1 else "
                     f"parts[k][{py(by)}].astype(str).agg(tuple, axis=1).astype(str)",
                     f"    _m = _keys.map(_f[{py(c)}]['table']).astype(object)",
                     f"    parts[k][{py(c)}] = parts[k][{py(c)}].fillna(pd.Series(_m.values, index=parts[k].index)).fillna(_f[{py(c)}]['fallback'])")
    ctx.touch(step, cols)
    return {"fitted": fitted, "indicator": ind, "columns_added":
            [c + "_missing" for c in cols] if ind else []}


def op_clip(ctx, step):
    p = step.get("params") or {}
    lo, hi = p.get("lower"), p.get("upper")
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    def apply(d):
        d = d.copy()
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce").clip(lower=lo, upper=hi)
        return d
    ctx.set_all(apply)
    for c in cols:
        ctx.emit(f"for k in parts: parts[k][{py(c)}] = pd.to_numeric(parts[k][{py(c)}], errors='coerce').clip(lower={py(lo)}, upper={py(hi)})")
    ctx.touch(step, cols)
    return {"lower": lo, "upper": hi}


def op_winsorize(ctx, step):
    p = step.get("params") or {}
    ql, qh = float(p.get("lower_q", 0.01)), float(p.get("upper_q", 0.99))
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    fit = ctx.parts[ctx.fit_part]
    fitted = {c: {"lower": float(pd.to_numeric(fit[c], errors="coerce").quantile(ql)),
                  "upper": float(pd.to_numeric(fit[c], errors="coerce").quantile(qh))}
              for c in cols}
    def apply(d):
        d = d.copy()
        for c in cols:
            d[c] = pd.to_numeric(d[c], errors="coerce").clip(fitted[c]["lower"], fitted[c]["upper"])
        return d
    ctx.set_all(apply)
    ctx.emit(f"_f = FITTED[{py(step['id'])}]['fitted']")
    for c in cols:
        ctx.emit(f"for k in parts: parts[k][{py(c)}] = pd.to_numeric(parts[k][{py(c)}], errors='coerce').clip(_f[{py(c)}]['lower'], _f[{py(c)}]['upper'])")
    ctx.touch(step, cols)
    return {"fitted": fitted, "quantiles": [ql, qh]}


def op_transform(ctx, step):
    p = step.get("params") or {}
    kind = p.get("kind", "log1p")
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    fit = ctx.parts[ctx.fit_part]
    fitted = {}
    if kind in ("box-cox", "yeo-johnson"):
        from scipy import stats as st
        for c in cols:
            v = pd.to_numeric(fit[c], errors="coerce").dropna().values
            if kind == "box-cox":
                if (v <= 0).any():
                    raise ValueError(f"box-cox needs strictly positive values; {c!r} has some <= 0 "
                                     "(use yeo-johnson)")
                _, lam = st.boxcox(v)
            else:
                _, lam = st.yeojohnson(v)
            fitted[c] = {"lambda": float(lam)}
    def apply(d):
        d = d.copy()
        for c in cols:
            v = pd.to_numeric(d[c], errors="coerce")
            if kind == "log1p":
                d[c] = np.log1p(v)
            elif kind == "signed-log1p":
                # sign(x)·log1p(|x|): the log's tail compression on a column that also holds
                # negatives, where log1p is undefined below -1. Monotone, odd, 0 -> 0, and
                # unfitted, so it needs no train statistic and cannot leak across parts.
                d[c] = np.sign(v) * np.log1p(np.abs(v))
            elif kind == "log":
                d[c] = np.log(v)
            elif kind == "sqrt":
                d[c] = np.sqrt(v)
            elif kind == "reciprocal":
                d[c] = 1.0 / v
            elif kind == "box-cox":
                from scipy import stats as st
                d[c] = st.boxcox(v.values, lmbda=fitted[c]["lambda"])
            elif kind == "yeo-johnson":
                from scipy import stats as st
                d[c] = st.yeojohnson(v.values, lmbda=fitted[c]["lambda"])
        return d
    ctx.set_all(apply)
    expr = {"log1p": "np.log1p(_v)", "signed-log1p": "np.sign(_v)*np.log1p(np.abs(_v))",
            "log": "np.log(_v)", "sqrt": "np.sqrt(_v)",
            "reciprocal": "1.0/_v"}.get(kind)
    if expr:
        for c in cols:
            ctx.emit("for k in parts:",
                     f"    _v = pd.to_numeric(parts[k][{py(c)}], errors='coerce'); parts[k][{py(c)}] = {expr}")
    else:
        ctx.emit("from scipy import stats as _st", f"_f = FITTED[{py(step['id'])}]['fitted']")
        fn = "boxcox" if kind == "box-cox" else "yeojohnson"
        for c in cols:
            ctx.emit("for k in parts:",
                     f"    _v = pd.to_numeric(parts[k][{py(c)}], errors='coerce').values",
                     f"    parts[k][{py(c)}] = _st.{fn}(_v, lmbda=_f[{py(c)}]['lambda'])")
    ctx.touch(step, cols)
    # box-cox / yeo-johnson estimate λ per column on the fit part: a fitted step, recorded as
    # one (the manifest's fit_on); the other kinds are closed-form and fit nothing.
    return {"kind": kind, "fitted": fitted,
            "fit_on": ctx.fit_part if kind in ("box-cox", "yeo-johnson") else None}


def op_bin(ctx, step):
    p = step.get("params") or {}
    kind, k = p.get("kind", "quantile"), int(p.get("k", 4))
    replace = bool(p.get("replace"))
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    fit = ctx.parts[ctx.fit_part]
    fitted = {}
    for c in cols:
        v = pd.to_numeric(fit[c], errors="coerce")
        if kind == "quantile":
            edges = list(np.unique(np.quantile(v.dropna(), np.linspace(0, 1, k + 1))))
        elif kind == "width":
            edges = list(np.linspace(float(v.min()), float(v.max()), k + 1))
        else:
            edges = [float(x) for x in (p.get("edges") or [])]
        if len(edges) < 3:
            raise ValueError(f"bin: {c!r} produced fewer than two bins")
        edges[0], edges[-1] = -np.inf, np.inf
        fitted[c] = {"edges": [float(x) for x in edges]}
    def apply(d):
        d = d.copy()
        for c in cols:
            v = pd.to_numeric(d[c], errors="coerce")
            b = pd.cut(v, bins=fitted[c]["edges"], labels=False, include_lowest=True)
            name = c if replace else c + "_bin"
            d[name] = b.astype("Int64")
        return d
    ctx.set_all(apply)
    ctx.emit(f"_f = FITTED[{py(step['id'])}]['fitted']")
    for c in cols:
        name = c if replace else c + "_bin"
        # The outer edges are ±inf by construction — a value beyond the training range still
        # falls in the first or last bin rather than becoming null. JSON has no infinity, so
        # the fitted file carries them as null and the two ends are restored here; without
        # this the edges come back non-monotonic and pd.cut refuses them.
        ctx.emit(f"_e = list(_f[{py(c)}]['edges']); _e[0], _e[-1] = -np.inf, np.inf",
                 f"for k in parts: parts[k][{py(name)}] = pd.cut(pd.to_numeric(parts[k][{py(c)}], errors='coerce'), "
                 f"bins=_e, labels=False, include_lowest=True).astype('Int64')")
        if not replace:
            ctx.lineage.setdefault(name, []).append(step["id"])
    ctx.touch(step, cols)
    return {"fitted": fitted, "replace": replace,
            "columns_added": [] if replace else [c + "_bin" for c in cols]}


def op_datetime_expand(ctx, step):
    p = step.get("params") or {}
    want = p.get("parts") or ["year", "month", "dow", "hour"]
    cyclic = bool(p.get("cyclic"))
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    added = []
    def apply(d):
        d = d.copy()
        for c in cols:
            t = pd.to_datetime(d[c], errors="coerce")
            for w in want:
                name = f"{c}_{w}"
                if w == "year":
                    d[name] = t.dt.year.astype("Int64")
                elif w == "month":
                    d[name] = t.dt.month.astype("Int64")
                elif w == "day":
                    d[name] = t.dt.day.astype("Int64")
                elif w == "dow":
                    d[name] = t.dt.dayofweek.astype("Int64")
                elif w == "hour":
                    d[name] = t.dt.hour.astype("Int64")
                elif w == "is_weekend":
                    d[name] = (t.dt.dayofweek >= 5).astype("Int64")
                if name in d.columns and name not in added:
                    added.append(name)
            if cyclic:
                for w, period in (("month", 12), ("dow", 7), ("hour", 24)):
                    if w in want:
                        base = f"{c}_{w}"
                        val = d[base].astype(float)
                        d[base + "_sin"] = np.sin(2 * np.pi * val / period)
                        d[base + "_cos"] = np.cos(2 * np.pi * val / period)
                        for s in (base + "_sin", base + "_cos"):
                            if s not in added:
                                added.append(s)
        return d
    ctx.set_all(apply)
    ctx.emit("for k in parts:")
    for c in cols:
        ctx.emit(f"    _t = pd.to_datetime(parts[k][{py(c)}], errors='coerce')")
        for w in want:
            expr = {"year": "_t.dt.year", "month": "_t.dt.month", "day": "_t.dt.day",
                    "dow": "_t.dt.dayofweek", "hour": "_t.dt.hour",
                    "is_weekend": "(_t.dt.dayofweek >= 5)"}.get(w)
            if expr:
                ctx.emit(f"    parts[k][{py(c + '_' + w)}] = {expr}.astype('Int64')")
        if cyclic:
            for w, period in (("month", 12), ("dow", 7), ("hour", 24)):
                if w in want:
                    b = f"{c}_{w}"
                    ctx.emit(f"    _v = parts[k][{py(b)}].astype(float)",
                             f"    parts[k][{py(b + '_sin')}] = np.sin(2*np.pi*_v/{period})",
                             f"    parts[k][{py(b + '_cos')}] = np.cos(2*np.pi*_v/{period})")
    for a in added:
        ctx.lineage.setdefault(a, []).append(step["id"])
    ctx.touch(step, cols)
    return {"columns_added": added, "cyclic": cyclic}


def op_lag(ctx, step):
    p = step.get("params") or {}
    lags = [int(x) for x in (p.get("lags") or [1])]
    gb, tcol = p.get("group_by"), p.get("time_column")
    # The checker refuses a lag grouped by the label before any data moves; the executor holds
    # the same rule where the step runs (a recipe reaching execute() past the checker).
    why = label_grouping_error(step, ctx.label or ctx.label_planned)
    if why:
        raise ValueError(f"lag: {why}")
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols()]
    added = []
    # The lags are read over the part in time order and written back by row label, so the part
    # keeps the row order the split gave it — the order `__split_raw__` records and every reader
    # that aligns a part with its raw rows by position relies on (the twin check, the semantic
    # check, the loaders' row ids). Sorting the part back by its index instead would put a time
    # split's part of several interleaved series back in raw-file order, one series after the
    # other, while `__split_raw__` still lists it in time order.
    def apply(d):
        d = d.copy()
        w = d.sort_values(tcol, kind="stable") if tcol and tcol in d.columns else d
        for c in cols:
            for L in lags:
                name = f"{c}_lag{L}"
                d[name] = (w.groupby(w[gb].astype(str))[c].shift(L) if gb and gb in w.columns
                           else w[c].shift(L))
                if name not in added:
                    added.append(name)
        return d
    ctx.set_all(apply)
    ctx.emit("for k in parts:")
    ctx.emit(f"    _w = parts[k].sort_values({py(tcol)}, kind='stable')" if tcol else
             "    _w = parts[k]")
    for c in cols:
        for L in lags:
            src = (f"_w.groupby(_w[{py(gb)}].astype(str))[{py(c)}].shift({L})"
                   if gb else f"_w[{py(c)}].shift({L})")
            ctx.emit(f"    parts[k][{py(f'{c}_lag{L}')}] = {src}")
    for a in added:
        ctx.lineage.setdefault(a, []).append(step["id"])
    ctx.touch(step, cols)
    return {"columns_added": added, "lags": lags}


# --------------------------------------------------------------------------- phase 5-6
def op_encode(ctx, step):
    p = step.get("params") or {}
    strat = p.get("strategy", "one-hot")
    minf = p.get("min_frequency")
    target = ctx.label or ctx.label_planned
    cols = [c for c in (step.get("columns") or []) if c in ctx.cols() and c != target]
    fit = ctx.parts[ctx.fit_part]
    fitted, added, dropped = {}, [], []
    for c in cols:
        vc = fit[c].astype(str).value_counts()
        if minf:
            keep = [str(k) for k, v in vc.items() if v >= (minf if minf >= 1 else minf * len(fit))]
        else:
            keep = [str(k) for k in vc.index]
        keep = sorted(keep)
        # Every encoding reserves what an unseen value becomes, and records it as a fitted
        # parameter rather than leaving it to a literal in the code. Training data cannot
        # contain every category; the reserved landing place is the difference between a
        # valid index and a null the network cannot embed. `/vega-forge` sizes the embedding
        # as `unseen_category_id + 1` from this record, so the two must be one number.
        if strat == "one-hot":
            fitted[c] = {"categories": keep, "unseen_category_id": None,
                         "unseen": "all-zero indicators"}
        elif strat == "ordinal":
            order = p.get("order") or keep
            mapping = {str(k): i for i, k in enumerate(order)}
            fitted[c] = {"mapping": mapping, "n_categories": len(mapping),
                         "unseen_category_id": len(mapping)}
        elif strat == "frequency":
            fitted[c] = {"freq": {str(k): float(v / len(fit)) for k, v in vc.items()},
                         "unseen_value": 0.0}
        elif strat == "target":
            if "train" not in ctx.parts:
                raise ValueError("target encoding needs a split: it is fitted out-of-fold on the "
                                 "train part alone")
            from sklearn.model_selection import KFold
            y = pd.to_numeric(fit[ctx.label], errors="coerce")
            prior = float(y.mean())
            means = y.groupby(fit[c].astype(str)).mean()
            fitted[c] = {"means": {str(k): float(v) for k, v in means.items()}, "prior": prior,
                         "unseen_value": prior}
        else:
            raise ValueError(f"encode strategy {strat!r} is not implemented by this executor")
    seen_unseen = {}
    def apply(d):
        d = d.copy()
        for c in cols:
            s = d[c].astype(str)
            if strat == "one-hot":
                miss = ~s.isin(fitted[c]["categories"])
                for cat in fitted[c]["categories"]:
                    name = f"{c}={cat}"
                    d[name] = (s == cat).astype("int8")
                    if name not in added:
                        added.append(name)
                d = d.drop(columns=[c])
            elif strat == "ordinal":
                m = s.map(fitted[c]["mapping"])
                miss = m.isna()
                d[c] = m.fillna(fitted[c]["unseen_category_id"]).astype("Int64")
            elif strat == "frequency":
                m = s.map(fitted[c]["freq"]).astype(float)
                miss = m.isna()
                d[c] = m.fillna(fitted[c]["unseen_value"])
            elif strat == "target":
                m = s.map(fitted[c]["means"]).astype(float)
                miss = m.isna()
                d[c] = m.fillna(fitted[c]["unseen_value"])
            seen_unseen[c] = seen_unseen.get(c, 0) + int(miss.sum())
        return d
    ctx.set_all(apply)
    for c, n in sorted(seen_unseen.items()):
        if n:
            mark(f"WARN: {step['id']}: {n} value(s) of {c!r} are categories the fit never saw; "
                 + ("they encode as all-zero indicators" if strat == "one-hot"
                    else f"they encode as the reserved "
                         f"{'id ' + str(fitted[c].get('unseen_category_id')) if strat == 'ordinal' else 'value ' + str(fitted[c].get('unseen_value'))}"))
    if strat == "one-hot":
        dropped = cols
        for c in cols:
            ctx.removed[c] = {"step": step["id"], "op": "encode", "why": "one-hot expanded"}
    ctx.emit(f"_f = FITTED[{py(step['id'])}]['fitted']")
    for c in cols:
        if strat == "one-hot":
            ctx.emit("for k in parts:",
                     f"    _s = parts[k][{py(c)}].astype(str)",
                     f"    _unseen(k, {py(c)}, ~_s.isin(_f[{py(c)}]['categories']))",
                     f"    for _cat in _f[{py(c)}]['categories']: parts[k][{py(c)} + '=' + _cat] = (_s == _cat).astype('int8')",
                     f"    parts[k] = parts[k].drop(columns=[{py(c)}])")
        elif strat == "ordinal":
            ctx.emit("for k in parts:",
                     f"    _m = parts[k][{py(c)}].astype(str).map(_f[{py(c)}]['mapping'])",
                     f"    _unseen(k, {py(c)}, _m.isna())",
                     f"    parts[k][{py(c)}] = _m.fillna(_f[{py(c)}]['unseen_category_id']).astype('Int64')")
        elif strat in ("frequency", "target"):
            table = "freq" if strat == "frequency" else "means"
            ctx.emit("for k in parts:",
                     f"    _m = parts[k][{py(c)}].astype(str).map(_f[{py(c)}][{py(table)}]).astype(float)",
                     f"    _unseen(k, {py(c)}, _m.isna())",
                     f"    parts[k][{py(c)}] = _m.fillna(_f[{py(c)}]['unseen_value'])")
    for a in added:
        ctx.lineage.setdefault(a, []).append(step["id"])
    ctx.touch(step, cols)
    return {"strategy": strat, "fitted": fitted, "columns_added": added,
            "columns_removed": dropped}


def op_scale(ctx, step):
    p = step.get("params") or {}
    kind = p.get("kind", "standard")
    want = p.get("columns", "numeric")
    fit = ctx.parts[ctx.fit_part]
    target = ctx.label or ctx.label_planned
    if want == "numeric":
        cols = [c for c in ctx.cols() if pd.api.types.is_numeric_dtype(fit[c]) and c != target]
    else:
        cols = [c for c in (want or step.get("columns") or []) if c in ctx.cols()]
    if kind == "none":
        return {"kind": "none", "columns": cols}
    fitted = {}
    for c in cols:
        v = pd.to_numeric(fit[c], errors="coerce")
        if kind == "standard":
            fitted[c] = {"center": float(v.mean()), "scale": float(v.std(ddof=0)) or 1.0}
        elif kind == "robust":
            iqr = float(v.quantile(0.75) - v.quantile(0.25)) or 1.0
            fitted[c] = {"center": float(v.median()), "scale": iqr}
        elif kind == "minmax":
            lo, hi = float(v.min()), float(v.max())
            fitted[c] = {"center": lo, "scale": (hi - lo) or 1.0}
    def apply(d):
        d = d.copy()
        for c in cols:
            v = pd.to_numeric(d[c], errors="coerce")
            d[c] = (v - fitted[c]["center"]) / fitted[c]["scale"]
        return d
    ctx.set_all(apply)
    ctx.emit(f"_f = FITTED[{py(step['id'])}]['fitted']")
    for c in cols:
        ctx.emit(f"for k in parts: parts[k][{py(c)}] = (pd.to_numeric(parts[k][{py(c)}], errors='coerce') "
                 f"- _f[{py(c)}]['center']) / _f[{py(c)}]['scale']")
    ctx.touch(step, cols)
    return {"kind": kind, "fitted": fitted, "columns": cols}


def op_project(ctx, step):
    p = step.get("params") or {}
    keep_named = bool(p.get("keep_named", True))
    want = p.get("columns")
    fit = ctx.parts[ctx.fit_part]
    cols = [c for c in (want or [c for c in ctx.cols()
                                 if pd.api.types.is_numeric_dtype(fit[c]) and c != ctx.label])
            if c in ctx.cols()]
    X = fit[cols].apply(pd.to_numeric, errors="coerce").dropna()
    from sklearn.decomposition import PCA
    n = p.get("k") or None
    var = p.get("variance")
    pca = PCA(n_components=(n if n else (var if var else 0.95)), random_state=ctx.seed).fit(X.values)
    fitted = {"columns": cols, "mean": [float(x) for x in pca.mean_],
              "components": [[float(x) for x in row] for row in pca.components_],
              "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]}
    names = [f"pc{i+1}" for i in range(len(pca.components_))]
    def apply(d):
        d = d.copy()
        V = d[cols].apply(pd.to_numeric, errors="coerce").fillna(pd.Series(fitted["mean"], index=cols))
        Z = (V.values - np.asarray(fitted["mean"])) @ np.asarray(fitted["components"]).T
        for i, nm in enumerate(names):
            d[nm] = Z[:, i]
        if not keep_named:
            d = d.drop(columns=[c for c in cols if c in d.columns])
        return d
    ctx.set_all(apply)
    ctx.emit(f"_f = FITTED[{py(step['id'])}]['fitted']",
             "for k in parts:",
             "    _V = parts[k][_f['columns']].apply(pd.to_numeric, errors='coerce').fillna(pd.Series(_f['mean'], index=_f['columns']))",
             "    _Z = (_V.values - np.asarray(_f['mean'])) @ np.asarray(_f['components']).T",
             f"    for _i, _nm in enumerate({py(names)}): parts[k][_nm] = _Z[:, _i]")
    if not keep_named:
        ctx.emit("    parts[k] = parts[k].drop(columns=[c for c in _f['columns'] if c in parts[k].columns])")
        for c in cols:
            ctx.removed[c] = {"step": step["id"], "op": "project", "why": "replaced by components"}
    for nm in names:
        ctx.lineage.setdefault(nm, []).append(step["id"])
    ctx.touch(step, cols)
    return {"fitted": fitted, "columns_added": names, "keep_named": keep_named}


def op_select_features(ctx, step):
    p = step.get("params") or {}
    kind = p.get("kind", "importance")
    thr = float(p.get("threshold", 0.0))
    src = p.get("from", "analysis:importance")
    keep_basis = set((geo_of(ctx).get("basis") or {}).get("members") or [])
    drop = []
    if kind == "importance" and ctx.fit_part == "train":
        raise ValueError(IMPORTANCE_AFTER_SPLIT)
    if kind == "importance":
        ana = ctx.layers.get("analysis") or {}
        pi = (((ana.get("modules") or {}).get("importance") or {}).get("evidence") or {}) \
            .get("permutation_importance") or []
        if not pi:
            raise ValueError("select_features kind=importance needs the analysis layer's "
                             "modules.importance.evidence.permutation_importance; this model "
                             "carries none")
        low = [r["feature"] for r in pi if (r.get("mean") or 0) <= thr]
        drop = [c for c in low if c in ctx.cols() and c not in keep_basis and c != ctx.label]
    elif kind == "variance":
        fit = ctx.parts[ctx.fit_part]
        for c in ctx.cols():
            if c == ctx.label or c in keep_basis:
                continue
            if pd.api.types.is_numeric_dtype(fit[c]) and float(pd.to_numeric(
                    fit[c], errors="coerce").var(ddof=0) or 0) <= thr:
                drop.append(c)
    elif kind == "explicit":
        # The named columns, and only after the split — which is the whole reason phase 6
        # exists. A column produced by a phase-5 step (a projection's components, an
        # encoder's expansion) does not exist before the split, so no earlier phase can
        # name it, and without this kind there is no way to say "computed, recorded, but
        # not an input". A basis member still cannot go without a `user:` step, which the
        # recipe checker enforces.
        named = list(p.get("columns") or step.get("columns") or [])
        if not named:
            raise ValueError("select_features kind=explicit needs the columns to drop, in "
                             "params.columns or the step's columns")
        missing = [c for c in named if c not in ctx.cols()]
        if missing:
            raise ValueError(f"select_features kind=explicit names column(s) that are not in "
                             f"the frame at phase 6: {', '.join(sorted(missing))}")
        if ctx.label in named:
            raise ValueError("select_features may not drop the label")
        drop = named
        res = _drop(ctx, step, drop, "named explicitly by the recipe")
        res["source_of_scores"] = src
        return res
    else:
        raise ValueError(f"select_features kind={kind!r} is not implemented by this executor")
    res = _drop(ctx, step, drop, f"{kind} at or below {thr} (from {src})")
    res["source_of_scores"] = src
    if kind == "variance":
        res["fit_on"] = ctx.fit_part          # the variances are the fit part's
    return res


def op_custom(ctx, step):
    p = step.get("params") or {}
    code = p.get("code") or ""
    ns = {"pd": pd, "np": np}
    exec(compile(code, "<custom-step>", "exec"), ns)
    fn = ns.get("step")
    if not callable(fn):
        raise ValueError("a custom step's code must define step(df, ctx) -> df")
    before = ctx.cols()
    meta = {"label": ctx.label, "crs": ctx.crs, "seed": ctx.seed}
    probe = None
    if pre_split(ctx):
        ctx.parts["all"], probe = row_local_probe(fn, ctx.parts["all"], dict(meta, part="all"),
                                                  step["id"])
    else:
        for k in list(ctx.parts):
            ctx.parts[k] = fn(ctx.parts[k].copy(), dict(meta, part=k))
    after = ctx.cols()
    # LABEL is a module-level constant in both generated programs, so it is referenced
    # directly: the old `'LABEL' in dir()` guard reads the *local* scope and would silently
    # hand the step None from inside a function body, which is where the transform-only
    # module runs it.
    ctx.emit("# custom step (user-supplied; the layer marks it unverified)",
             "_ns = {}", f"exec(compile({py(code)}, '<custom-step>', 'exec'), " "{'pd': pd, 'np': np}, _ns)",
             "for k in parts: parts[k] = _ns['step'](parts[k].copy(), "
             f"{{'label': LABEL, 'crs': {py(ctx.crs)}, 'seed': {ctx.seed}, 'part': k}})")
    return {"columns_added": [c for c in after if c not in before],
            "columns_removed": [c for c in before if c not in after], "verified": False,
            **({"row_local_probe": probe} if probe else {})}


def pre_split(ctx):
    """True while the frame is still whole and the recipe splits it later: what a step does
    here, every part inherits."""
    return set(ctx.parts) == {"all"} and any(
        isinstance(s, dict) and s.get("op") == "split" for s in ctx.recipe.get("steps") or [])


def _probe_slices(n):
    pos = np.arange(n)
    return (("the even rows", pos[0::2]), ("the odd rows", pos[1::2]),
            ("the first half", pos[: n // 2]), ("the second half", pos[n // 2:]))


def _frame_difference(got, want):
    """None when `got` holds the rows (by index, in order), the columns and the values of
    `want`; otherwise what differs, in words. Values compare as objects with missing equal to
    missing, so a slice whose column comes back int64 where the whole's is float64 (a missing
    value on the other side of the frame) still compares by value."""
    if not isinstance(got, pd.DataFrame):
        return f"it returned {type(got).__name__}, not a DataFrame"
    if list(got.columns) != list(want.columns):
        return f"its columns are {list(got.columns)}, not {list(want.columns)}"
    if not got.index.equals(want.index):
        miss, extra = want.index.difference(got.index), got.index.difference(want.index)
        if not len(miss) and not len(extra):
            return "it returns the rows in another order"
        return (f"it keeps {len(extra)} row(s) the whole frame drops and drops {len(miss)} the "
                "whole frame keeps")
    for j, c in enumerate(got.columns):
        a, b = got.iloc[:, j].astype(object), want.iloc[:, j].astype(object)
        differ = ~(a.eq(b) | (a.isna() & b.isna()))
        if differ.any():
            r = differ.idxmax()
            return (f"column {c!r} on row {r} is {a[r]!r} there and {b[r]!r} on the whole "
                    "frame")
    return None


def row_local_probe(fn, frame, meta, sid):
    """Run a pre-split `custom` step on the whole frame, then on slices of it, and refuse it
    unless every slice gives exactly the whole's rows for that slice.

    Before the split there is one frame, the future test rows in it. A step that computes
    over rows — a mean, a rank, a frequency, a dedupe across rows — gives a row a value that
    depends on the other rows, the test rows among them, and every part inherits it. A
    row-local step (a retyping, a parse, a filter on the row's own values) gives a row the
    same result whatever rows are beside it, so a slice's result is the whole's restricted to
    the slice. That is decidable by running it, where reading the code is not. The step must
    also keep each row's index: it is the raw-file position the split records."""
    whole = fn(frame.copy(), dict(meta))
    if not isinstance(whole, pd.DataFrame):
        raise ValueError(f"{sid}: a custom step must return a DataFrame, not "
                         f"{type(whole).__name__}")
    if whole.index.has_duplicates or not whole.index.isin(frame.index).all():
        raise ValueError(f"{sid}: a custom step before the split must keep each row's index "
                         "(the raw-file position the split records) — this one renumbers, "
                         "adds or repeats rows")
    slices = [what for what, pos in _probe_slices(len(frame)) if len(pos)]
    for what, pos in _probe_slices(len(frame)):
        if not len(pos):
            continue
        part = frame.iloc[pos]
        diff = _frame_difference(fn(part.copy(), dict(meta)),
                                 whole[whole.index.isin(part.index)])
        if diff:
            raise ValueError(
                f"{sid}: a custom step before the split must be row-local, and this one "
                f"computes over rows — run on {what} alone, {diff}. Every part would inherit "
                "a value the future test rows helped compute: give it a phase after the "
                "split, where it runs on each part alone, or use the catalog op")
    return whole, {"probe": "row-local", "passed": True, "rows": int(len(frame)),
                   "slices": slices}


# --------------------------------------------------------------------------- phase 7 (graph)
# One implementation of the graph step, emitted as source: the executor runs it (exec'd below),
# the reproduction script carries it as the step's lines, and the transform-only module carries
# the part of it its entry point reaches. A kNN whose tie order or distance summation differed
# between the program that wrote the edges part and the one that serves a query would join a
# served row to a different neighbour than the evaluation did, and nothing downstream would
# say so — so there is exactly one copy, and it avoids every computation whose result depends
# on the rows computed beside it (a BLAS product blocks by matrix shape; a feature-by-feature
# accumulation does not). Each method's own functions are bound to role names (GRAPH_ROLES)
# the shared code calls, so a program carries nothing another method would need.
GRAPH_LIB = r'''
_G_EARTH_R = 6371008.8


def _g_text(values):
    """Key values as text: a key read as 80.0 and one read as 80 are the same key; a missing
    key stays missing and relates nothing."""
    s = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_float_dtype(s) and bool((s.dropna() % 1 == 0).all()):
        s = s.astype("Int64")
    return s.astype("string").astype(object).where(s.notna(), None).to_numpy(dtype=object)


def _g_numbers(frame, columns, who):
    """`columns` of `frame` as a float64 matrix. A missing or non-finite value is refused:
    a distance to a hole is not a distance."""
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise KeyError(who + ": column(s) " + ", ".join(map(repr, absent))
                       + " are not in the frame")
    m = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan)
    fin = np.isfinite(m)
    if not fin.all():
        rows = np.flatnonzero(~fin.all(axis=1))
        cols = [c for c, f in zip(columns, fin.all(axis=0)) if not f]
        err = ValueError(f"{who}: {len(rows)} row(s) lack a finite value in {cols} (the first at "
                         f"position {int(rows[0])}); a distance needs every coordinate — impute them before graph_edges")
        err.problem_errors = lambda: [{"column": c, "error": "missing or non-finite", "rows": [int(r) for r in np.flatnonzero(~fin[:, j])][:5]} for j, c in enumerate(columns) if not fin[:, j].all()]  # a service: 422
        raise err
    return np.ascontiguousarray(m)


def _g_space(frame, columns, who):
    """knn's fitted space: the columns as the network reads them, float32 — the dtype the
    bundle's reference nodes store — held in float64 for the sums. Every program rounds here,
    so a float32 reference and a float64 part give the same matrix, bit for bit."""
    return _g_numbers(frame, columns, who).astype("float32").astype("float64")


def _g_coords(frame, coords, who):
    """Raw latitude / longitude as the metric coordinates spatial_reproject recorded."""
    la = _g_numbers(frame, [coords["lat"]], who)[:, 0]
    lo = _g_numbers(frame, [coords["lon"]], who)[:, 0]
    pr = coords["projection"]
    if pr.get("kind") == "pyproj":
        from pyproj import Transformer
        tr = Transformer.from_crs(pr.get("from") or "EPSG:4326", pr["to"], always_xy=True)
        x, y = tr.transform(lo, la)
        return np.column_stack([np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")])
    x = np.radians(lo - pr["lon0"]) * _G_EARTH_R * np.cos(np.radians(pr["lat0"]))
    y = np.radians(la - pr["lat0"]) * _G_EARTH_R
    return np.column_stack([x, y])


def _g_digest(m, g):
    """A fitted matrix's digest, C order: float32 for knn (its space), float64 for spatial."""
    import hashlib
    m = np.ascontiguousarray(m, dtype="float32" if g["method"] == "knn" else "float64")
    return "sha256:" + hashlib.sha256(m.tobytes()).hexdigest()


def _g_euclidean(a, b):
    """(len(a), len(b)) euclidean distances (role _g_metric), accumulated one feature at a
    time: a pair's value depends on those two rows only, so a one-row query and a batch get
    the same number."""
    d = np.zeros((len(a), len(b)))
    for f in range(a.shape[1]):
        t = a[:, f, None] - b[None, :, f]
        d += t * t
    return np.sqrt(d)


def _g_cosine(a, b):
    """(len(a), len(b)) cosine distances (role _g_metric), accumulated one feature at a time,
    each row scaled to unit length first (a zero row stays zero: distance 1 to everything)."""
    na, nb = np.zeros(len(a)), np.zeros(len(b))
    for f in range(a.shape[1]):
        na, nb = na + a[:, f] * a[:, f], nb + b[:, f] * b[:, f]
    na, nb = np.sqrt(na), np.sqrt(nb)
    a, b = a / np.where(na == 0, 1.0, na)[:, None], b / np.where(nb == 0, 1.0, nb)[:, None]
    s = np.zeros((len(a), len(b)))
    for f in range(a.shape[1]):
        s += a[:, f, None] * b[None, :, f]
    return 1.0 - s


def _g_nearest(a, b, g, same=None):
    """For each row of `a`, its neighbours among the rows of `b`: the `k` nearest, or every one
    within `radius_m`, then at most `max_degree`; a tie goes to the row of `b` that comes
    first. `same = (a_ids, b_ids)` names the node behind each row, and a node is never its own
    neighbour. Returns (ai, bi, distance), ordered by ai, then distance, then bi."""
    k, radius, cap = g.get("k"), g.get("radius_m"), g.get("max_degree")
    keep = min(x for x in (k, cap) if x) if (k or cap) else None
    ai, bi, di = [np.zeros(0, "int64")], [np.zeros(0, "int64")], [np.zeros(0)]
    step = max(1, 4000000 // max(1, len(b)))
    for s0 in range(0, len(a) if len(b) else 0, step):
        d = _g_metric(a[s0:s0 + step], b)
        if same is not None:
            d[same[0][s0:s0 + step, None] == same[1][None, :]] = np.inf
        if radius is not None:
            d[d > radius] = np.inf
        for r, row in enumerate(d):
            cand = np.flatnonzero(np.isfinite(row))
            if keep is not None and len(cand) > keep:
                cand = cand[row[cand] <= np.partition(row[cand], keep - 1)[keep - 1]]
            cand = cand[np.lexsort((cand, row[cand]))][:keep].astype("int64")
            ai.append(np.full(len(cand), s0 + r, dtype="int64"))
            bi.append(cand)
            di.append(row[cand])
    return np.concatenate(ai), np.concatenate(bi), np.concatenate(di).astype("float64")


def _g_pairs(src, dst, g):
    """The key relation from the rows of `src` to the rows of `dst`: row i -> row j when i's
    second key column equals j's first (a foreign key: dst_col of i names src_col of j), or,
    with one key column, when the two rows share it. Returns (i, j), ordered."""
    cols = list(g["columns"])
    a, b = (cols[1], cols[0]) if len(cols) == 2 else (cols[0], cols[0])
    left = pd.DataFrame({"i": np.arange(len(src), dtype="int64"), "v": _g_text(src[a])})
    right = pd.DataFrame({"j": np.arange(len(dst), dtype="int64"), "v": _g_text(dst[b])})
    m = left[left["v"].notna()].merge(right[right["v"].notna()], on="v", how="inner")
    m = m.sort_values(["i", "j"], kind="stable")
    return m["i"].to_numpy(dtype="int64"), m["j"].to_numpy(dtype="int64")


def _g_cap(s, d, pos, cap):
    """At most `cap` edges per source: the smallest position gaps, a tie to the earlier row."""
    if len(s) == 0:
        return np.zeros(0, dtype=bool)
    gap = np.abs(pos[s] - pos[d])
    order = np.lexsort((pos[d], gap, s))
    ss = s[order]
    start = np.r_[True, ss[1:] != ss[:-1]]
    first = np.maximum.accumulate(np.where(start, np.arange(len(ss)), 0))
    keep = np.zeros(len(s), dtype=bool)
    keep[order[(np.arange(len(ss)) - first) < cap]] = True
    return keep


def _g_weights(g, table, rows, dist):
    """What an edge weighs: the metric distance, the source row's `weight` column read from
    the input, or nothing (NaN throughout)."""
    w = g.get("weight")
    if w == "distance":
        return np.asarray(dist, dtype="float64")
    if w:
        vals = pd.to_numeric(table[w], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
        return vals[np.asarray(rows, dtype="int64")]
    return np.full(len(rows), np.nan)


def _g_loop_weights(g, table):
    """A self loop's weight: distance 0, the row's own `weight` column, or nothing."""
    w = g.get("weight")
    if w == "distance":
        return np.zeros(len(table))
    if w:
        return pd.to_numeric(table[w], errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
    return np.full(len(table), np.nan)


def _g_keys(g, table, x, pos, cand=None):
    """key_pair's own directed edges inside one node set (row numbers of `table`), a cap ranked
    by `pos` (role _g_method; `x` and `cand` are the neighbour methods' and unused here)."""
    s, d = _g_pairs(table, table, g)
    s, d = s[s != d], d[s != d]
    caps = [c for c in (g.get("max_degree"),
                        (g.get("max_clique") or 0) - 1 if g.get("max_clique") else None) if c]
    if caps:
        keep = _g_cap(s, d, pos, min(caps))
        s, d = s[keep], d[keep]
    return s, d, _g_weights(g, table, s, None)


def _g_near(g, table, x, pos, cand=None):
    """knn / spatial's own directed edges: each row's nearest among the `cand` rows (every row
    when None) in `x` (role _g_method; `pos` is key_pair's and unused here)."""
    cand = np.arange(len(table), dtype="int64") if cand is None else np.asarray(cand, "int64")
    s, b, dist = _g_nearest(x, x[cand], g, same=(np.arange(len(table), dtype="int64"), cand))
    return s, cand[b], _g_weights(g, table, s, dist)


def _g_over(g, table, x, pos, cand=None):
    """The method's directed edges over one node table, before _g_finish.

    `pos` orders the rows: the raw position at level node; at level graph the row's place in
    the input, ranked within its graph here, so a whole-graph query ranks its rows exactly as
    the shaper did. `cand` (level node, knn / spatial) is the fitted part's rows in that part's
    order: the only rows a neighbour may be."""
    pos = np.asarray(pos, dtype="int64")
    if not g.get("graph_id"):
        return _g_method(g, table, x, pos, cand)
    codes = pd.factorize(pd.Series(table["__graph__"], dtype=object))[0]
    order = np.lexsort((pos, codes))
    bounds = np.flatnonzero(np.r_[True, codes[order][1:] != codes[order][:-1], True])
    out_s, out_d, out_w = [np.zeros(0, "int64")], [np.zeros(0, "int64")], [np.zeros(0)]
    for a0, a1 in zip(bounds[:-1], bounds[1:]):
        idx = order[a0:a1]
        s, d, w = _g_method(g, table.iloc[idx].reset_index(drop=True),
                            None if x is None else x[idx], np.arange(len(idx), dtype="int64"))
        out_s.append(idx[s]); out_d.append(idx[d]); out_w.append(w)
    return np.concatenate(out_s), np.concatenate(out_d), np.concatenate(out_w).astype("float64")


def _g_finish(src, dst, w, g, loops):
    """The edges as the part stores them: the method's own edges first, their reverses when
    `symmetric`, one self loop per node of `loops = (nodes, weights)` when `self_loops`; each
    directed pair once (the method's edge wins over a reverse), ordered by (src, dst)."""
    f = pd.DataFrame({"src": np.asarray(src, "int64"), "dst": np.asarray(dst, "int64"),
                      "weight": np.asarray(w, "float64")})
    f = f[f["src"] != f["dst"]]
    frames = [f]
    if g.get("symmetric", True):
        frames.append(pd.DataFrame({"src": f["dst"].to_numpy(), "dst": f["src"].to_numpy(),
                                    "weight": f["weight"].to_numpy()}))
    if g.get("self_loops"):
        frames.append(pd.DataFrame({"src": np.asarray(loops[0], "int64"),
                                    "dst": np.asarray(loops[0], "int64"),
                                    "weight": np.asarray(loops[1], "float64")}))
    e = pd.concat(frames, ignore_index=True).drop_duplicates(["src", "dst"], keep="first")
    return e.sort_values(["src", "dst"], kind="stable").reset_index(drop=True)


def _g_once(qi, ri, w):
    """(query row, reference row, weight), each pair once — its first row wins — ordered."""
    f = pd.DataFrame({"q": np.asarray(qi, "int64"), "r": np.asarray(ri, "int64"),
                      "w": np.asarray(w, "float64")})
    f = f.drop_duplicates(["q", "r"], keep="first").sort_values(["q", "r"], kind="stable")
    return f["q"].to_numpy(), f["r"].to_numpy(), f["w"].to_numpy()


def _g_attach_keys(g, query, ref, xq=None, xr=None):
    """key_pair (role _g_attach): a query row's attachments to the reference — its own edges
    into it, weighed by its own column, and the reference rows' edges into it (their own on a
    directed step, the reverses of the query's on a symmetric one), by theirs; each pair once,
    the query's own edge winning. The consumer orients every pair reference -> query, and no
    query -> query pair exists, so no row's answer depends on the rows sent beside it."""
    qi, ri = _g_pairs(query, ref, g)
    rj, qj = _g_pairs(ref, query, g)
    return _g_once(np.r_[qi, qj], np.r_[ri, rj],
                   np.r_[_g_weights(g, query, qi, None), _g_weights(g, ref, rj, None)])


def _g_attach_near(g, query, ref, xq=None, xr=None):
    """knn / spatial at level node (role _g_attach): a query row's attachments are its own
    edges — a neighbour is a train node, and so is every reference node's, so none points at
    the query."""
    qi, ri, dist = _g_nearest(xq, xr, g)
    return _g_once(qi, ri, _g_weights(g, query, qi, dist))


def _graph_part(parts, raw, g):
    """The `edges` part: every row of every part is a node, named by its raw position."""
    names = [k for k in ("train", "valid", "test") if k in parts]
    ids = np.concatenate([np.asarray(parts[k].index, dtype="int64") for k in names])
    part_of = np.concatenate([np.array([k] * len(parts[k]), dtype=object) for k in names])
    table = raw.iloc[ids][list(g["raw_columns"])].reset_index(drop=True)
    if g.get("graph_id"):
        table["__graph__"] = _g_text(raw.iloc[ids][g["graph_id"]])
    x, cand = None, None
    if g["method"] == "knn":
        x = np.vstack([_g_space(parts[k], g["features"], "graph_edges (" + k + " part)")
                       for k in names])
    elif g["method"] == "spatial":
        x = _g_coords(table, g["coords"], "graph_edges")
    if g["method"] != "key_pair" and not g.get("graph_id"):
        cand = np.flatnonzero(part_of == g["fit_part"])
    s, d, w = _g_over(g, table, x, ids, cand)
    e = _g_finish(ids[s], ids[d], w, g, (ids, _g_loop_weights(g, table)))
    where = pd.Series(part_of, index=ids)
    return pd.DataFrame({"src_pos": e["src"].to_numpy(dtype="int64"),
                         "dst_pos": e["dst"].to_numpy(dtype="int64"),
                         "src_part": where.loc[e["src"].to_numpy()].to_numpy(dtype=object),
                         "dst_part": where.loc[e["dst"].to_numpy()].to_numpy(dtype=object),
                         "weight": e["weight"].to_numpy(dtype="float64")})
'''

# The role a function plays: the spec key that picks who plays it, and who, per value.
# `_g_method` (a node set's own directed edges), `_g_metric` (a distance) and `_g_attach` (a
# query row's attachments) are GRAPH_LIB's roles; `_candidates` (edges_for's candidate nodes
# and their space) and `_graph_x` (edges_within's space) are SERVE_LIB's. A program binds
# them once, for its step.
GRAPH_ROLES = {
    "_g_method": ("method", {"key_pair": "_g_keys", "knn": "_g_near", "spatial": "_g_near"}),
    "_g_metric": ("metric", {None: "_g_euclidean", "euclidean": "_g_euclidean",
                             "cosine": "_g_cosine"}),
    "_g_attach": ("method", {"key_pair": "_g_attach_keys", "knn": "_g_attach_near",
                             "spatial": "_g_attach_near"}),
    "_candidates": ("method", {"key_pair": "_key_candidates", "knn": "_knn_candidates",
                               "spatial": "_spatial_candidates"}),
    "_graph_x": ("method", {"key_pair": "_key_graph_x", "knn": "_knn_graph_x",
                            "spatial": "_spatial_graph_x"}),
}
LIB_ROLES = ("_g_method", "_g_metric", "_g_attach")
# edges_for reads a reference node's input columns under this prefix: the reference frame
# carries the shaped features too, and a raw key, weight or coordinate may share its name with
# one (a standardised `lat` beside the raw `lat`, a scaled weight beside the raw one).
RAW_PREFIX = "raw:"


def graph_role(g, role):
    """The function that plays `role` for the resolved step `g`."""
    key, players = GRAPH_ROLES[role]
    return players[g.get(key)]


def graph_role_lines(g, roles):
    """The lines binding `roles` to the step's functions."""
    return [f"{r} = {graph_role(g, r)}" for r in roles]


def graph_lib(g):
    """GRAPH_LIB as callables, its roles bound for the step `g`: what the executor runs."""
    ns = {"pd": pd, "np": np}
    src = GRAPH_LIB + "\n".join(graph_role_lines(g, LIB_ROLES)) + "\n"
    exec(compile(src, "<dataset-shaper GRAPH_LIB>", "exec"), ns)
    return ns


def reader_expr(input_path):
    """The pandas call that reads the input, as the generated programs spell it."""
    return {"csv": f"pd.read_csv({str(input_path)!r}, low_memory=False)",
            "tsv": f"pd.read_csv({str(input_path)!r}, sep='\\t', low_memory=False)",
            # Arrow's own filesystem, not a Python file object (A40: an abort at exit).
            "parquet": f"pd.read_parquet({str(input_path)!r}, "
                       "filesystem=__import__('pyarrow.fs').fs.LocalFileSystem())",
            "xlsx": f"pd.read_excel({str(input_path)!r})",
            "json": f"pd.read_json({str(input_path)!r})",
            "jsonl": f"pd.read_json({str(input_path)!r}, lines=True)"}.get(
        Path(input_path).suffix.lstrip(".").lower(), f"pd.read_csv({str(input_path)!r})")


def resolve_basis(ctx, step):
    """knn `features: "basis"` — the basis members as they stand at phase 7 (imputed, scaled,
    encoded, projected), which is the fitted space the network sees."""
    target = ctx.label or ctx.label_planned
    fit = ctx.parts[ctx.fit_part]
    gid = (step.get("params") or {}).get("graph_id")
    numeric = [c for c in ctx.cols() if c not in (target, gid)
               and pd.api.types.is_numeric_dtype(fit[c])]
    members = list((geo_of(ctx).get("basis") or {}).get("members") or [])
    if not members:
        mark(f"WARN: {step['id']} graph_edges: features \"basis\" resolved to every numeric "
             f"column at phase 7 ({len(numeric)}) — no geometry layer names the basis")
        return numeric
    out = [c for c in numeric if c in members or any(c.startswith(m + "=") for m in members)]
    if any(m not in ctx.cols() for m in members):
        # a projection that replaced the basis leaves its components in its place
        out += [c for c in numeric if re.fullmatch(r"pc\d+", c) and c not in out]
    return out


def knn_spec(ctx, step, gid):
    """knn: the fitted columns, resolved and checked against the frame at phase 7."""
    p, sid = step.get("params") or {}, step["id"]
    feats = p.get("features", "basis")
    feats = resolve_basis(ctx, step) if feats == "basis" else list(feats)
    absent = [c for c in feats if c not in ctx.cols()]
    if absent:
        raise ValueError(f"{sid} graph_edges: knn feature(s) {absent} are not in the frame at "
                         "phase 7 (a step before it dropped or renamed them)")
    if not feats:
        raise ValueError(f"{sid} graph_edges: knn has no feature to measure a distance on")
    fit = ctx.parts[ctx.fit_part]
    for c in feats:
        if c in (ctx.label, gid):
            raise ValueError(f"{sid} graph_edges: {c!r} cannot be a knn feature — it is the "
                             f"{'label' if c == ctx.label else 'graph key'}")
        if not pd.api.types.is_numeric_dtype(fit[c]):
            raise ValueError(f"{sid} graph_edges: knn feature {c!r} is not numeric at phase 7; "
                             "encode it first")
    return {"features": feats, "k": int(p.get("k", 5)), "metric": p.get("metric", "euclidean")}


def spatial_spec(ctx, step, gid=None):
    """spatial: the raw coordinates parse_geometry names and the projection spatial_reproject
    recorded — the geometry as those two steps left it."""
    p = step.get("params") or {}
    geom = next((s for s in ctx.recipe["steps"] if s.get("op") == "parse_geometry"), None)
    repro = next((s for s in ctx.recipe["steps"] if s.get("op") == "spatial_reproject"), None)
    proj = (ctx.fitted.get(repro["id"]) or {}).get("projection") if repro else None
    if not geom or not proj:
        raise ValueError(f"{step['id']} graph_edges: spatial needs parse_geometry and "
                         "spatial_reproject to have run before it")
    gp = geom.get("params") or {}
    return {"coords": {"lat": gp.get("lat"), "lon": gp.get("lon"), "projection": proj},
            "k": p.get("k"), "radius_m": p.get("radius_m"), "metric": "euclidean"}


def keypair_spec(ctx, step, gid=None):
    """key_pair: the key columns (`[src_col, dst_col]`, a foreign key, or `[col]`, a clique)
    and the clique cap."""
    p = step.get("params") or {}
    return {"columns": list(p.get("columns") or step.get("columns")),
            "max_clique": p.get("max_clique")}


GRAPH_SPECS = {"key_pair": keypair_spec, "knn": knn_spec, "spatial": spatial_spec}


def graph_raw_columns(ctx, g):
    """The input columns the step reads at the rows' raw positions — keys, coordinates, a
    weight column — each checked to exist."""
    method, sid = g["method"], g["step"]
    named = ([(c, "key column") for c in g["columns"]] if method == "key_pair" else
             [(g["coords"][k], "coordinate column") for k in ("lat", "lon")]
             if method == "spatial" else [])
    if g["weight"] not in (None, "distance"):
        named.append((g["weight"], "weight column"))
    for c, what in named:
        if c not in ctx.raw.columns:
            raise ValueError(f"{sid} graph_edges: {what} {c!r} is not a column of the input")
    gid = g["graph_id"]
    if gid and gid not in ctx.raw.columns:
        raise ValueError(f"{sid} graph_edges: graph_id {gid!r} is not a column of the input")
    if gid and gid not in ctx.cols():
        raise ValueError(f"{sid} graph_edges: graph_id {gid!r} left the frame before phase 7; "
                         "a graph-level network groups its nodes by it, so it must reach the "
                         "shaped parts (keep it out of drop_identity)")
    return list(dict.fromkeys(c for c, _ in named))


def graph_reference(ctx, g):
    """What edges_for needs of a reference node (level node; none at level graph): the fitted
    columns as shaped, under their own names, and each input column it reads — keys and weight
    for key_pair, coordinates for spatial — under `raw:<column>`, the name a reference frame
    that also carries the shaped features gives it."""
    if g["graph_id"]:
        return {}
    raw = {"key_pair": g.get("columns", []) + ([g["weight"]] if g["weight"] not in
                                               (None, "distance") else []),
           "spatial": [g.get("coords", {}).get("lat"), g.get("coords", {}).get("lon")],
           "knn": []}[g["method"]]
    names = {RAW_PREFIX + c: c for c in dict.fromkeys(raw)}
    clash = [n for n in names if n in ctx.cols()]
    if clash:
        raise ValueError(f"{g['step']} graph_edges: the shaped frame has a column named "
                         f"{clash[0]!r}, the name a reference gives the input column "
                         f"{names[clash[0]]!r}; rename it before graph_edges")
    return {"raw": names, "shaped": list(g.get("features") or []), "parts": ["train", "valid"]}


def graph_spec(ctx, step):
    """Resolve a graph_edges step against the frame it runs on, into the one literal the three
    programs share: the method, the level, the columns read from the input at the rows' raw
    positions, the fitted columns, and what an edges_for call needs of the reference."""
    p = step.get("params") or {}
    gid = p.get("graph_id") or None
    g = {"step": step["id"], "method": p.get("method"), "level": "graph" if gid else "node",
         "graph_id": gid, "symmetric": bool(p.get("symmetric", True)),
         "self_loops": bool(p.get("self_loops", False)), "weight": p.get("weight"),
         "max_degree": p.get("max_degree"), "fit_part": "train"}
    g.update(GRAPH_SPECS[g["method"]](ctx, step, gid))
    g["raw_columns"] = graph_raw_columns(ctx, g)
    g["query_columns"] = g["raw_columns"] + ([gid] if gid else [])
    g["reference"] = graph_reference(ctx, g)
    g["n_fit"] = int(len(ctx.parts["train"])) if g["method"] != "key_pair" and not gid else None
    return g


def graph_block(ctx, step, g, edges):
    """The manifest's `graph` block: what /vega-forge reads as `shape:manifest/graph`."""
    names = [k for k in ("train", "valid", "test") if k in ctx.parts]
    pair = edges["src_part"].astype(str) + "|" + edges["dst_part"].astype(str)
    cross = {f"{a}-{b}": int(((pair == f"{a}|{b}") | (pair == f"{b}|{a}")).sum())
             for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    touched = set(edges["src_pos"].tolist()) | set(edges["dst_pos"].tolist())
    params = dict(step.get("params") or {})
    if g["method"] == "knn":
        params["features"] = g["features"]
    return {"step": step["id"], "method": g["method"], "params": jsonable(params),
            "n_nodes": {k: int(len(ctx.parts[k])) for k in names}, "n_edges": int(len(edges)),
            "directed": not g["symmetric"], "symmetric": g["symmetric"],
            "self_loops": g["self_loops"], "weight": g["weight"], "cross_part_edges": cross,
            "within_part_edges": {k: int((pair == f"{k}|{k}").sum()) for k in names},
            "isolated_nodes": {k: int(sum(1 for i in ctx.parts[k].index if int(i) not in touched))
                               for k in names},
            "level": g["level"], "graph_id": g["graph_id"],
            "fitted": g["level"] == "node" and g["method"] != "key_pair",
            "edges_for": ({"columns": edges_for_columns(g), "raw": g["reference"]["raw"]}
                          if g["level"] == "node" else None),
            "edges": "outputs.files.edges"}


def graph_fitted_reference(ctx, g, lib):
    """The fitted entry of a knn / spatial index at level node: which matrix it was built on
    (by part and columns, never copied) and that matrix's digest."""
    if g["method"] == "knn":
        m = lib["_g_space"](ctx.parts["train"], g["features"], "graph_edges (train part)")
        cols, what = g["features"], "the train part's fitted columns in part order, as float32"
    else:
        tr = ctx.raw.iloc[np.asarray(ctx.parts["train"].index, dtype="int64")]
        m = lib["_g_coords"](tr.reset_index(drop=True), g["coords"], "graph_edges")
        cols = [g["coords"]["lat"], g["coords"]["lon"]]
        what = ("the train rows' raw coordinates projected as spatial_reproject recorded, "
                "(x, y), as float64")
    return {"fit_on": "train", "reference": "__split_raw__.train", "part": "train",
            "columns": cols, "n_reference": int(len(m)), "matrix": what + ", C order",
            "matrix_digest": lib["_g_digest"](m, g),
            "k": g.get("k"), "radius_m": g.get("radius_m"), "metric": g.get("metric")}


def graph_capped_groups(ctx, step, g, lib):
    """How many cliques a max_clique cap capped (a key_pair cap is refused at level node, so
    this is a within-graph reading)."""
    if g["method"] != "key_pair" or len(g["columns"]) != 1 or not g.get("max_clique"):
        return
    ids = np.concatenate([np.asarray(v.index, dtype="int64") for _, v in ctx.each()])
    groups = {"key": lib["_g_text"](ctx.raw.iloc[ids][g["columns"][0]])}
    if g["graph_id"]:
        groups["graph"] = lib["_g_text"](ctx.raw.iloc[ids][g["graph_id"]])
    groups = pd.DataFrame(groups).dropna()
    sizes = groups.groupby(list(groups.columns)).size()
    ctx.graph["capped_groups"] = int((sizes > int(g["max_clique"])).sum())
    mark(f"OK: {step['id']} graph_edges: {ctx.graph['capped_groups']} clique(s) larger than "
         f"max_clique {g['max_clique']} were capped")


def graph_parts_check(ctx, step):
    """The parts as the split left them, or a refusal: every end of an edge is a raw position
    read off a part's index, so a step between the split and here that renumbered or filtered
    a part would name the wrong rows."""
    if "train" not in ctx.parts:
        raise ValueError(f"{step['id']} graph_edges: no split ran, so there are no parts for the "
                         "edges to record")
    split = next((s for s in ctx.recipe["steps"] if s.get("op") == "split"), None)
    raw_sets = (ctx.fitted.get(split["id"]) or {}).get("raw_indices") if split else None
    for k, frame in ctx.each():
        if raw_sets is not None and [int(i) for i in frame.index] != list(raw_sets.get(k, [])):
            raise ValueError(f"{step['id']} graph_edges: the {k} part's rows are no longer the "
                             "raw positions the split recorded (a step after the split "
                             "renumbered, reordered or filtered them), so an edge could not "
                             "name its ends; graph_edges needs the parts as the split left them")


def op_graph_edges(ctx, step):
    """Phase 7: the `edges` part, the manifest's `graph` block, and — for knn / spatial at level
    node — the fitted reference. Rows and columns are untouched."""
    graph_parts_check(ctx, step)
    g = graph_spec(ctx, step)
    lib = graph_lib(g)
    if g["graph_id"]:
        ids = np.concatenate([np.asarray(v.index, dtype="int64") for _, v in ctx.each()])
        n = sum(1 for v in lib["_g_text"](ctx.raw.iloc[ids][g["graph_id"]]) if v is None)
        if n:
            raise ValueError(f"{step['id']} graph_edges: {n} row(s) carry no {g['graph_id']!r}; "
                             "at level graph every row belongs to one graph")
    ctx.edges = lib["_graph_part"](ctx.parts, ctx.raw, g)
    ctx.graph = graph_block(ctx, step, g, ctx.edges)
    graph_capped_groups(ctx, step, g, lib)
    res = {"method": g["method"], "level": g["level"], "fitted": ctx.graph["fitted"],
           "spec": jsonable(g), "n_edges": int(len(ctx.edges)), "columns_added": []}
    if ctx.graph["fitted"]:
        res.update(graph_fitted_reference(ctx, g, lib))
    ctx.emit("# graph_edges writes the edges part beside the split parts (see the manifest's "
             "graph block); the functions below are the ones the executor ran",
             *GRAPH_LIB.strip("\n").splitlines(), *graph_role_lines(g, LIB_ROLES),
             f"_g = {py(jsonable(g))}",
             f"_g_raw = {reader_expr(ctx.recipe['input']['path'])}",
             "EDGES = _graph_part(parts, _g_raw, _g)")
    ctx.touch(step, [])
    b = ctx.graph
    mark(f"OK: {step['id']} graph_edges {g['method']} (level {g['level']}): {b['n_edges']} "
         f"directed edge row(s) over {sum(b['n_nodes'].values())} node(s); cross-part " +
         ", ".join(f"{k} {v}" for k, v in b["cross_part_edges"].items()) +
         (f"; index fitted on train ({b['n_nodes']['train']} nodes)" if b["fitted"] else ""))
    lonely = sum(b["isolated_nodes"].values())
    if lonely:
        mark(f"WARN: {step['id']} graph_edges: {lonely} node(s) have no edge "
             f"({', '.join(f'{k} {v}' for k, v in b['isolated_nodes'].items() if v)}); a graph "
             "network answers them from their own features alone")
    return res


# --------------------------------------------------------------------------- driver
OPS = {"retype": op_retype, "drop_identity": op_drop_identity, "drop_constant": op_drop_constant,
       "dedupe": op_dedupe, "parse_datetime": op_parse_datetime, "parse_geometry": op_parse_geometry,
       "orient_cycle": op_orient_cycle, "drop_derived": op_drop_derived,
       "select_partition": op_select_partition, "drop_leakage": op_drop_leakage,
       "keep_columns": op_keep_columns, "split": op_split, "impute": op_impute, "clip": op_clip,
       "winsorize": op_winsorize, "transform": op_transform, "bin": op_bin,
       "datetime_expand": op_datetime_expand, "lag": op_lag, "encode": op_encode,
       "scale": op_scale, "project": op_project, "select_features": op_select_features,
       "graph_edges": op_graph_edges, "custom": op_custom}


def read_parquet(path):
    """A parquet file through Arrow's own filesystem (PLAN-AMENDMENTS A40). pandas' default
    hands Arrow a Python file object; Arrow's reader threads then take the GIL, and one still
    inside Python while the interpreter finalizes aborts the process at exit (`terminate
    called without an active exception`, SIGABRT) after the work is done."""
    import pyarrow.fs
    return pd.read_parquet(path, filesystem=pyarrow.fs.LocalFileSystem())


def read_dataset(path, sep=None):
    p = Path(path)
    s = p.suffix.lower()
    if s in (".csv", ".txt"):
        return pd.read_csv(p, sep=sep or ",", low_memory=False)
    if s == ".tsv":
        return pd.read_csv(p, sep=sep or "\t", low_memory=False)
    if s == ".parquet":
        return read_parquet(p)
    if s in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if s == ".jsonl":
        return pd.read_json(p, lines=True)
    if s == ".json":
        return pd.read_json(p)
    raise ValueError(f"unsupported dataset extension {s!r}")


def execute(recipe, layers, df, seed):
    steps = recipe.get("steps") or []
    key_errs = split_key_errors(steps, next((i for i, s in enumerate(steps)
                                             if isinstance(s, dict) and s.get("op") == "split"),
                                            None))
    if key_errs:
        # the checker refuses the same recipe; held here too, because a step that names a
        # column the split dropped would otherwise skip it without a word (encode, scale)
        raise ValueError(key_errs[0])
    ctx = Ctx(df, recipe, layers, seed)
    for s in recipe["steps"]:
        if s.get("op") == "select_partition":
            ctx.label_planned = (s.get("params") or {}).get("label")
    geom = None
    for step in recipe["steps"]:
        op = step["op"]
        before_cols = ctx.cols()
        before_rows = {k: len(v) for k, v in ctx.each()}
        ctx.emit("", f"# --- {step['id']}  {op}  [{step.get('source')}]  {step.get('rationale','')}")
        code_from = len(ctx.code)
        if op in SPATIAL_OPS:
            import importlib.util
            spec = importlib.util.spec_from_file_location("shaper_spatial", HERE / "spatial_ops.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if geom is None:
                for s2 in recipe["steps"]:
                    if s2["op"] == "parse_geometry":
                        geom = {"lat": (s2.get("params") or {}).get("lat"),
                                "lon": (s2.get("params") or {}).get("lon")}
            if not geom or not geom.get("lat"):
                raise ValueError(f"{step['id']}: a spatial step needs a parse_geometry step before "
                                 "it (which is also where the CRS is declared)")
            fitted = mod.OPS[op](ctx, step, geom)
        else:
            fn = OPS.get(op)
            if fn is None:
                raise ValueError(f"{step['id']}: no executor for op {op!r}")
            fitted = fn(ctx, step)
        after_cols = ctx.cols()
        after_rows = {k: len(v) for k, v in ctx.each()}
        ctx.code_by_step[step["id"]] = list(ctx.code[code_from:])
        ctx.fitted[step["id"]] = fitted
        ctx.manifest.append({
            "id": step["id"], "op": op, "source": step.get("source"),
            "rows_before": before_rows, "rows_after": after_rows,
            "columns_added": [c for c in after_cols if c not in before_cols],
            "columns_removed": [c for c in before_cols if c not in after_cols],
            # Always fitted ops say so by their op; an op fitted only in some configurations
            # (transform box-cox / yeo-johnson, select_features variance, a spatial step's
            # projection or neighbour index, graph_edges' train index) says so in its result.
            "fit_on": (ctx.fit_part if op in FITTED_OPS else
                       (fitted or {}).get("fit_on") if isinstance(fitted, dict) else None),
            "parameters_fitted": fitted})
        fit_on = ctx.manifest[-1]["fit_on"]
        mark(f"OK: {step['id']} {op} — rows {before_rows} -> {after_rows}, "
             f"columns {len(before_cols)} -> {len(after_cols)}"
             + (f", fitted on {fit_on}" if fit_on else ""))
    return ctx


REPRO_HEAD = '''#!/usr/bin/env python3
"""Reproduce {out_dir} from {input_path} — generated by /dataset-shaper, and standalone.

This script imports nothing from the skill. It is the recipe written out as the pandas program
it amounts to, one block per step, in the order the executor ran them, using the parameters that
were fitted on the training part (they are inlined below as FITTED, exactly as the manifest
records them). Running it must reproduce the output files byte for byte; `shape.py --check` is
that comparison, and it is how the determinism claim is tested rather than asserted.

    python3 {script_name}            # writes the same files next to this script
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT = {input_path!r}
FITTED = json.loads(HERE.joinpath({fitted_name!r}).read_text())
MANIFEST_SPLIT = {{k: np.asarray(v, dtype=int) for k, v in FITTED.get("__split__", {{}}).items()}}
EXPECTED = {expected!r}
UNSEEN = {{}}


def _unseen(part, column, mask):
    """Count the values an encoder had never seen when it was fitted.

    The encode blocks below call this before they map. It exists so the two generated
    programs — this one and `preprocess.py`, which records the same events per row for the
    predictor's warnings column — call the *same* emitted line, and the reserved id an
    unseen value lands on is never a fallback one of them invented on its own.
    """
    n = int(mask.sum())
    if n:
        UNSEEN.setdefault(part, {{}})[column] = UNSEEN.setdefault(part, {{}}).get(column, 0) + n


df = {reader}
parts = {{"all": df}}
'''

REPRO_TAIL = '''

for _part, _cols in sorted(UNSEEN.items()):
    print(f"WARN: {_part}: categories the fit never saw — " +
          ", ".join(f"{c} ({n})" for c, n in sorted(_cols.items())) +
          "; they carry the encoder's reserved unseen value")

# --- write the outputs
out = {}
for name, frame in parts.items():
    path = HERE / (FILES[name] if name in FILES else (SLUG + "." + name + "." + FORMAT))
    if FORMAT == "parquet":
        frame.reset_index(drop=True).to_parquet(path, index=False)
    else:
        frame.reset_index(drop=True).to_csv(path, index=False)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    out[name] = {"path": str(path), "digest": "sha256:" + h, "rows": len(frame),
                 "columns": len(frame.columns)}
    print(f"OK: wrote {path} ({len(frame)} rows x {len(frame.columns)} columns)")

bad = [n for n, v in out.items() if EXPECTED.get(n) and EXPECTED[n] != v["digest"]]
if bad:
    print("ERROR: digests differ from the manifest for: " + ", ".join(bad))
    raise SystemExit(3)
print("OK: every output digest matches the manifest")
'''


# The edges part is not a split part: the step leaves it in EDGES, and it is written and
# digest-checked with the parts, so `--check` covers the graph as it covers the frame.
REPRO_EDGES = '''
_p = HERE / FILES["edges"]
if FORMAT == "parquet":
    EDGES.to_parquet(_p, index=False)
else:
    EDGES.to_csv(_p, index=False)
out["edges"] = {"path": str(_p), "digest": "sha256:" + hashlib.sha256(_p.read_bytes()).hexdigest(),
                "rows": len(EDGES), "columns": len(EDGES.columns)}
print(f"OK: wrote {_p} ({len(EDGES)} edge rows)")
'''


def write_edges(ctx, path, fmt, dry_run):
    """The edges part, written beside the split parts unless --dry-run: `{"edges": entry}`
    for the manifest's outputs.files, or {} when no graph step ran."""
    if ctx.edges is None:
        return {}
    if not dry_run:
        if fmt == "parquet":
            ctx.edges.to_parquet(path, index=False)
        else:
            ctx.edges.to_csv(path, index=False)
        mark(f"OK: wrote {path} ({len(ctx.edges)} edge rows, the graph's edges part)")
    return {"edges": {"path": str(path), "digest": None if dry_run else digest_file(path),
                      "rows": int(len(ctx.edges)), "columns": int(ctx.edges.shape[1])}}


def write_reproduction(ctx, out_dir, slug, fmt, files, expected, input_path, fitted_name):
    reader = reader_expr(input_path)
    name = f"reproduce_{slug}.py"
    head = REPRO_HEAD.format(out_dir=str(out_dir), input_path=str(input_path),
                             script_name=name, expected=expected, reader=reader,
                             fitted_name=fitted_name)
    body = "\n".join(ctx.code)
    consts = (f"\nSLUG = {slug!r}\nFORMAT = {fmt!r}\nFILES = {files!r}\n"
              f"LABEL = {ctx.label!r}\n")
    tail = REPRO_TAIL
    if ctx.edges is not None:
        tail = tail.replace("\nbad = [", REPRO_EDGES + "\nbad = [", 1)
    (Path(out_dir) / name).write_text(head + consts + body + tail, encoding="utf-8")
    return name


# --------------------------------------------------------------------------- transform-only
PREPROCESS_DOC = '''"""preprocess — the recipe of {recipe_name}, applied to new rows.

`transform(df_raw, history=None) -> DataFrame` takes rows in the *input* schema the recipe
consumed and returns them in the shaped schema the network was trained on. It satisfies the
`FeatureTransformer` Protocol structurally: a module with a `transform` of that signature is
all the port asks for, and this module imports nothing from /dataset-shaper, /vega-forge or
any other skill — pandas, numpy and the standard library{extra_imports}.

Generated by `dataset-shaper/scripts/shape.py --transform-only`. It is the *same emitted
lines* as `reproduce_{slug}.py`, filtered to the steps that shape a row and run over a single
part, so serving and training are one implementation of the recipe rather than two that
happen to agree. Do not edit it: edit the recipe and emit it again.

What it deliberately does not do:

  * it does not fit anything. Every statistic, category map, bin edge, scaler and loading
    comes from {fitted_name}, which was fitted on the training part alone. A transformer that
    re-fitted on the query would give a different answer for the same row depending on what
    arrived beside it.
  * it does not split, and it does not deduplicate. Those shape a training corpus, not a
    row: the caller asked for one prediction per row given, and returning fewer would be a
    silent answer to a different question.
  * it does not filter by partition. `select_partition` names the label; here the label is
    an output, so the column is dropped when a caller leaves it in and nothing else happens.
  * it does not decide anything the recipe left open. Column drops are applied by name, from
    the list the executor resolved when it ran; nothing is re-derived from these rows.

Unseen categories. Training data cannot contain every category. Each encoder reserves what an
unseen value becomes — an ordinal one the id `unseen_category_id`, the one the manifest
records and the network's embedding has a row for; a frequency or target one the recorded
`unseen_value`; a one-hot one all-zero indicators, which is a defined and honest state but
still a row the model has no evidence about. All three are *recorded per row*: after a call,
`transform(df_raw, with_unseen=True)` returns them with the frame — one tuple of column names
per returned row, in the returned row order — and `predict.py` puts them in its `warnings`
column. Nothing about a call outlives it except a per-thread copy that `last_unseen()` reads
for older callers, so concurrent calls on different threads never see each other's rows.
{history_doc}{warnings_doc}"""
'''

PREPROCESS_IMPORTS = '''from __future__ import annotations

import json
import os
import threading
from pathlib import Path

'''

PREPROCESS_PRELUDE = '''
__all__ = ["transform", "last_unseen", "LABEL", "PARTITION", "REQUIRED_COLUMNS",
           "NEEDS_HISTORY", "HISTORY_COLUMNS"]

HERE = Path(__file__).resolve().parent


def _load_fitted():
    """Find the fitted parameters, or say exactly where it looked.

    The bundle ships this module inside the emitted package and `fitted.json` at the bundle
    root, so `beside itself` is two directories up as often as it is here. A transformer that
    silently ran without its parameters would return unscaled, unencoded columns and the
    network would score them.
    """
    named = os.environ.get("VEGA_FITTED")
    tried = []
    for cand in ([Path(named)] if named else []) + [HERE / n for n in _FITTED_NAMES] + \\
            [HERE.parent / n for n in _FITTED_NAMES] + \\
            [HERE.parent.parent / n for n in _FITTED_NAMES]:
        tried.append(str(cand))
        if cand.is_file():
            return json.loads(cand.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "the fitted parameters this recipe was emitted against are not beside this module; "
        "looked in " + ", ".join(tried) + ". Copy the shaper's fitted file next to it, or "
        "point VEGA_FITTED at it.")


# One slot per thread, written at the start of each call: a module-level dict here would let
# two concurrent calls report each other's unseen categories (measured: 27 of 200 calls on two
# threads), which is a warning attached to the wrong flow.
_STATE = threading.local()


def _unseen(part, column, mask):
    """Record, per row, that an encoder met a category it was never fitted on.

    The encode blocks below call this before they map — the same call the reproduction
    script's blocks make, where it only counts. Here it keeps the rows, because the
    predictor's contract returns `warnings` per row and a caller cannot tell from a scored
    number that the row it came from was outside the training vocabulary.
    """
    flags = mask.fillna(False).to_numpy(dtype=bool)
    rows = _STATE.unseen
    for label in mask.index[flags]:
        rows.setdefault(label, set()).add(column)


def last_unseen():
    """Deprecated shim: the unseen columns of this thread's last `transform` call.

    Prefer `transform(df_raw, with_unseen=True)`, which returns them with the frame.
    """
    return [list(r) for r in getattr(_STATE, "last", ())]
'''

PREPROCESS_TRANSFORM_HEAD = '''

def _query_frame(df_raw, history):
    """(the frame the steps run over, the labels of its query rows). The query rows are checked
    before a history is prepended, which could stand in for a column they miss."""
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("transform takes a pandas DataFrame of raw rows, not "
                        + type(df_raw).__name__)
    absent = [c for c in REQUIRED_COLUMNS if c not in df_raw.columns]
    if absent:
        raise KeyError("the recipe reads column(s) these rows do not carry: "
                       + ", ".join(absent))
    frame, start = df_raw.reset_index(drop=True), 0
    if NEEDS_HISTORY:
        if history is None:
            raise ValueError(_NO_HISTORY)
        thin = [c for c in HISTORY_COLUMNS if c not in history.columns]
        if thin:
            raise KeyError("the history rows do not carry " + ", ".join(thin)
                           + "; the lags are read from them, so they need the columns the "
                             "query needs and the label when a lag reads it")
        start = len(history)
        frame = pd.concat([history.reset_index(drop=True), frame], ignore_index=True, sort=False)
    if LABEL and LABEL not in frame.columns:
        # A query row has no target: a step that repairs the label in place still runs, on
        # this placeholder, and the label leaves where the partition step says. A lagged label
        # comes from the history, where a query row that does not carry it reads as missing.
        frame[LABEL] = pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame, list(frame.index[start:])

'''

PREPROCESS_TRANSFORM_TAIL = '''

def transform(df_raw, history=None, with_unseen=False):
    """Raw rows in, shaped rows out, with the parameters already fitted.

    The returned frame carries `df_raw`'s index, so a caller can join predictions back to the
    rows it sent. `history` is read only when the recipe lags and is ignored otherwise. With
    `with_unseen=True` the result is `(frame, unseen)`, `unseen` holding one tuple of column
    names per returned row — the categories an encoder had never been fitted on. Raises
    `KeyError` naming the column when a required input column is absent, and `ValueError` when
    a lag recipe is called without its history.
    """
    frame, keep = _query_frame(df_raw, history)
    _STATE.unseen = {}
    parts = {"rows": frame.copy()}
    for step in _STEPS:
        step(parts)
    # The query rows are recovered by label rather than by position: a lag step reads the
    # frame in time order and writes back by row label, and `.loc` is right either way.
    out = parts["rows"].loc[keep].copy()
    unseen = tuple(tuple(sorted(_STATE.unseen.get(i, ()))) for i in keep)
    _STATE.last = unseen
    out.index = df_raw.index
    return (out, unseen) if with_unseen else out
'''

PREPROCESS_GRAPH_DOC = '''
The graph. `{sid}` ({method}, level {level}) relates rows and shapes none, so `transform`
leaves it alone. The functions of dataset-shaper's GRAPH_LIB that wrote the `edges` part
serve it through `{entry}`; the other entry point raises, naming this one.
'''

# The module's entry points and their helpers. The method-specific ones (`_key_*`, `_knn_*`,
# `_spatial_*`) play SERVE_LIB's roles (GRAPH_ROLES); a module carries its method's alone.
SERVE_LIB = r'''
def edges_for(df_raw, reference, history=None):
    """Raw query rows -> DataFrame[query_pos, ref_pos, weight], their attachments (level node).

    `query_pos` indexes `df_raw`; `ref_pos` is `reference["pos"]`, an opaque node id. The
    reference carries `pos` and EDGES_FOR_COLUMNS (a fitted column as shaped, float32 or
    float64; a `raw:<column>` as the input held it), the train nodes first in the train
    part's order (vega/graph-reference@1's). A row is a (query, reference) pair the edges part
    joins either way, once, weighed by the query's own edge if any: orient it reference ->
    query. Never query -> query, so no answer depends on the rows sent beside it.
    """
    query = _rows(df_raw, "edges_for")
    if not isinstance(reference, pd.DataFrame) or "pos" not in reference.columns:
        raise KeyError("the reference frame needs a 'pos' column: the node id edges_for "
                       "returns as ref_pos")
    absent = [c for c in EDGES_FOR_COLUMNS if c not in reference.columns]
    if absent:
        raise KeyError("the reference nodes do not carry " + ", ".join(absent)
                       + "; edges_for reads them to join a query row")
    ref, xq, xr = _candidates(query, reference.reset_index(drop=True), history)
    qi, ri, w = _g_attach(GRAPH, query, _raw_of(ref), xq, xr)
    return pd.DataFrame({"query_pos": np.asarray(qi, dtype="int64"),
                         "ref_pos": ref["pos"].to_numpy()[np.asarray(ri, dtype="int64")],
                         "weight": np.asarray(w, dtype="float64")})


def edges_within(df_raw, history=None):
    """Whole graphs of raw rows -> the edges inside each graph (level graph).

    Returns DataFrame[src_pos, dst_pos, weight], positions local to `df_raw`, computed as the
    shaper computed the part: an edge never joins two graphs, and a cap or a tie is ranked by
    a row's place within its graph — send each graph's rows in the order the input held them.
    Raises KeyError naming a column the rows lack.
    """
    query = _rows(df_raw, "edges_within")
    table = query[list(GRAPH["raw_columns"])].copy()
    table["__graph__"] = _g_text(query[GRAPH["graph_id"]])
    if table["__graph__"].isna().any():
        raise ValueError(f"{int(table['__graph__'].isna().sum())} row(s) carry no "
                         f"{GRAPH['graph_id']!r}: every row of a graph-level query belongs "
                         "to a graph")
    pos = np.arange(len(query), dtype="int64")
    s, d, w = _g_over(GRAPH, table, _graph_x(query, table, history), pos)
    e = _g_finish(pos[s], pos[d], w, GRAPH, (pos, _g_loop_weights(GRAPH, table)))
    return pd.DataFrame({"src_pos": e["src"].to_numpy(dtype="int64"),
                         "dst_pos": e["dst"].to_numpy(dtype="int64"),
                         "weight": e["weight"].to_numpy(dtype="float64")})


def _rows(df_raw, who):
    """The query rows at positions 0..n-1; refused without a column the graph reads."""
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(who + " takes a pandas DataFrame of raw rows, not "
                        + type(df_raw).__name__)
    thin = [c for c in GRAPH["query_columns"] if c not in df_raw.columns]
    if thin:
        raise KeyError(who + " reads column(s) these rows do not carry: " + ", ".join(thin))
    return df_raw.reset_index(drop=True)


def _raw_of(ref):
    """The reference's `raw:<column>` columns, back under their input names."""
    return pd.DataFrame({c: ref[n].to_numpy() for n, c in GRAPH["reference"]["raw"].items()},
                        index=ref.index)


def _fitted_nodes(ref):
    """The train nodes a knn / spatial index was fitted on: the reference's first n_fit rows."""
    n = GRAPH["n_fit"]
    if len(ref) < n:
        raise ValueError(f"{GRAPH['method']} joins a query row to the {n} train nodes its index "
                         f"was fitted on; the reference holds {len(ref)} node(s) — list the "
                         "train nodes first, then the valid ones, or pass them alone")
    return ref.iloc[:n]


def _check_index(xr):
    """`xr`, once it digests to the fitted matrix: a reference in another order, scaling or
    subset would join a served row to other neighbours than the evaluation's graph gave it."""
    got = _g_digest(xr, GRAPH)
    if got != GRAPH["matrix_digest"]:
        raise ValueError("the reference's train nodes are not the matrix the " + GRAPH["method"]
                         + " index was fitted on (digest " + got + ", fitted "
                         + GRAPH["matrix_digest"] + "): list the train nodes first, in the "
                         "train part's order, their fitted columns as shaped and their "
                         "raw:<column> values as the input held them")
    return xr


def _key_candidates(query, ref, history):
    """key_pair (role _candidates): every reference node, and no space to measure in."""
    return ref, None, None


def _knn_candidates(query, ref, history):
    """knn (role _candidates): the train nodes, and both sides in the fitted space — the
    query rows shaped by transform first, the reference's fitted columns as they are."""
    ref = _fitted_nodes(ref)
    xr = _check_index(_g_space(ref, GRAPH["features"], "edges_for (reference nodes)"))
    xq = _g_space(transform(query, history=history), GRAPH["features"], "edges_for (query rows)")
    return ref, xq, xr


def _spatial_candidates(query, ref, history):
    """spatial (role _candidates): the train nodes, and both sides' raw coordinates, projected."""
    ref = _fitted_nodes(ref)
    xr = _check_index(_g_coords(_raw_of(ref), GRAPH["coords"], "edges_for (reference nodes)"))
    return ref, _g_coords(query, GRAPH["coords"], "edges_for (query rows)"), xr


def _key_graph_x(query, table, history):
    """key_pair within each graph (role _graph_x): no space to measure in."""
    return None


def _knn_graph_x(query, table, history):
    """knn within each graph (role _graph_x): the query rows shaped, in the fitted space."""
    return _g_space(transform(query, history=history), GRAPH["features"], "edges_within")


def _spatial_graph_x(query, table, history):
    """spatial within each graph (role _graph_x): the raw coordinates, projected."""
    return _g_coords(table, GRAPH["coords"], "edges_within")
'''

# The entry point of the other level, which a module carries as a refusal naming its own.
EDGES_FOR_REFUSAL = '''def edges_for(df_raw, reference, history=None):
    """Level node only: this recipe's graph is level graph — call edges_within(df_raw)."""
    raise ValueError("this recipe's graph is level graph (graph_id " + repr(GRAPH["graph_id"])
                     + "): a query is a whole graph with no reference to join — call "
                       "edges_within(df_raw)")'''
EDGES_WITHIN_REFUSAL = '''def edges_within(df_raw, history=None):
    """Level graph only: this recipe's graph is level node — call edges_for."""
    raise ValueError("this recipe's graph is level node: a query row joins the reference "
                     "nodes — call edges_for(df_raw, reference)")'''

GRAPH_SECTION_HEAD = '''

# ------------------------------------------------------------------------------- the graph
# dataset-shaper's GRAPH_LIB — the functions that wrote the `edges` part, those this module's
# entry point reaches, each private one without its docstring — then the entry points.'''


def graph_step_of(recipe):
    return next((s for s in recipe["steps"] if s.get("op") == "graph_edges"), None)


def edges_for_columns(g):
    """The reference columns edges_for reads, in order: each `raw:<column>` (an input column,
    as read), then the fitted columns as shaped for knn. What /vega-pack ships per reference
    node beside `ref`, and what its graph.json lists as `edges_for_columns`."""
    ref = g.get("reference") or {}
    return list(ref.get("raw") or {}) + list(ref.get("shaped") or [])


def graph_bare(node, text):
    """A private function's lines without its docstring and its full-line comments — the same
    code, which the parse below proves before a line of it is emitted."""
    doc = node.body[0] if ast.get_docstring(node) is not None else None
    skip = set(range(doc.lineno, doc.end_lineno + 1)) if doc else set()
    out = [ln for i, ln in enumerate(text, node.lineno)
           if i not in skip and not ln.lstrip().startswith("#")]
    same = copy.copy(node)
    same.body = node.body[1:] if doc else node.body
    if ast.dump(ast.parse("\n".join(out)).body[0]) != ast.dump(same):
        raise RuntimeError(f"dataset-shaper: {node.name} lost code with its docstring")
    return out


def graph_defs(source):
    """Each top-level definition of a GRAPH_LIB-style source: name -> {text, refs}, a private
    function's text bare (graph_bare)."""
    lines = source.strip("\n").splitlines()
    out = {}
    for node in ast.parse("\n".join(lines)).body:
        fn = isinstance(node, ast.FunctionDef)
        name = node.name if fn else node.targets[0].id
        text = lines[node.lineno - 1:node.end_lineno]
        out[name] = {"text": "\n".join(graph_bare(node, text) if fn and name[0] == "_" else text),
                     "refs": {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}}
    return out


def graph_closure(defs, root, g):
    """The definitions `root` reaches, a role resolved to the step's player: (names, roles)."""
    keep, roles, todo = set(), set(), [root]
    while todo:
        name = todo.pop()
        if name in GRAPH_ROLES:
            roles.add(name)
            name = graph_role(g, name)
        if name in defs and name not in keep:
            keep.add(name)
            todo.extend(defs[name]["refs"])
    return keep, roles


def graph_module_source(graph):
    """The module's graph section: GRAPH and EDGES_FOR_COLUMNS, the GRAPH_LIB and SERVE_LIB
    definitions the level's entry point reaches for this method, the other level's refusal,
    and the method's role bindings."""
    root, refusal = (("edges_for", EDGES_WITHIN_REFUSAL) if graph["level"] == "node" else
                     ("edges_within", EDGES_FOR_REFUSAL))
    defs = {**graph_defs(GRAPH_LIB), **graph_defs(SERVE_LIB)}
    keep, roles = graph_closure(defs, root, graph)
    blocks = [d["text"] for n, d in defs.items() if n in keep] + [refusal]
    head = [GRAPH_SECTION_HEAD, f"GRAPH = {py(graph)}",
            f"EDGES_FOR_COLUMNS = {py(edges_for_columns(graph))}"]
    return ("\n".join(head) + "\n\n\n" + "\n\n\n".join(blocks) + "\n\n\n"
            + "\n".join(graph_role_lines(graph, [r for r in GRAPH_ROLES if r in roles])) + "\n")


def graph_module(recipe, fitted, fitted_name, out):
    """The graph half of the transform-only module — {doc, prelude, source} — or None; `out`
    hears which entry point serves the graph."""
    step = graph_step_of(recipe)
    if step is None:
        return None
    entry = fitted.get(step["id"]) or {}
    graph = entry.get("spec")
    if not isinstance(graph, dict) or not isinstance((graph.get("reference") or {}).get(
            "raw", {}), dict):
        raise ValueError(f"{fitted_name} carries no resolved graph spec for step {step['id']} "
                         "in this executor's form: it was written by an older one. Re-run "
                         "shape.py on the dataset to regenerate it.")
    if entry.get("matrix_digest"):
        # the module carries the fitted index's digest itself, so edges_for can refuse
        # another candidate matrix wherever its fitted file is projected or trimmed
        graph = dict(graph, matrix_digest=entry["matrix_digest"])
    node = graph["level"] == "node"
    served = ("edges_for(df_raw, reference)" if node else "edges_within(df_raw)")
    out(f"OK: {step['id']} graph_edges ({graph['method']}, level {graph['level']}) served by "
        f"{served}: " + (f"query -> reference attachments only, reference columns "
                         f"{edges_for_columns(graph)}" if node else
                         f"the edges inside each {graph['graph_id']!r} graph"))
    return {"doc": PREPROCESS_GRAPH_DOC.format(sid=step["id"], method=graph["method"],
                                               level=graph["level"], entry=served),
            "prelude": PREPROCESS_PRELUDE.replace(
                '"HISTORY_COLUMNS"]', '"HISTORY_COLUMNS", "GRAPH", "EDGES_FOR_COLUMNS",\n           '
                '"edges_for", "edges_within"]', 1),
            "source": graph_module_source(graph)}


def transform_plan(recipe):
    """Which steps shape a row, which shape only the corpus, and why.

    Returns (applied, recorded, skipped, warnings) — the classification the emitter writes
    into the module and prints. It is a pure reading of the recipe: no data, no fitting.
    """
    applied, recorded, skipped, warnings = [], [], [], []
    for s in recipe["steps"]:
        op = s.get("op")
        if op in TRANSFORM_SKIP:
            skipped.append((s, TRANSFORM_SKIP[op]))
        elif op in TRANSFORM_RECORD:
            recorded.append(s)
        else:
            applied.append(s)
            if op == "custom":
                warnings.append(f"{s['id']} is a custom step: its code is carried into the "
                                "module verbatim and runs at inference unverified, exactly as "
                                "the layer marks it")
    return applied, recorded, skipped, warnings


# Steps whose emitted line selects by name among the columns present (`if c in ...columns`):
# they read nothing a query must carry.
SELECTION_OPS = {"drop_identity", "drop_constant", "drop_derived", "drop_leakage",
                 "keep_columns", "select_features"}


def step_reads(s):
    """The columns a step reads: its `columns`, the columns its params name (`lat`, `lon`,
    `by`, `group_by`, `time_column`, `columns`)."""
    p = s.get("params") or {}
    consumed = set(s.get("columns") or [])
    for key in ("lat", "lon", "by", "group_by", "time_column"):
        v = p.get(key)
        if isinstance(v, str):
            consumed.add(v)
        elif isinstance(v, list):
            consumed |= {x for x in v if isinstance(x, str)}
    if isinstance(p.get("columns"), list):
        consumed |= {x for x in p["columns"] if isinstance(x, str)}
    return consumed


def label_readers(recipe, applied, label):
    """The applied steps after `select_partition` that read the label (a `lag` of the target
    …; an `impute` or a `lag` grouped by it never gets here, the checker refuses it), in
    recipe order. The executor keeps the label in every part to the end, so each of them read
    it there; the transform-only module keeps it until the last of them has run and drops it
    then — dropped where the partition is named, the first of them would raise on a column
    the executor had."""
    if not label:
        return []
    ids = [s.get("id") for s in recipe["steps"]]
    at = next((i for i, s in enumerate(recipe["steps"]) if s.get("op") == "select_partition"), 0)
    return [s for s in applied if ids.index(s["id"]) > at and s.get("op") not in SELECTION_OPS
            and label in step_reads(s)]


def required_columns(recipe, fitted, applied):
    """The input columns the applied steps read and no earlier step creates.

    A missing one is the failure this list exists to name: without it the first step would
    raise a bare KeyError from inside pandas, several steps away from the recipe that asked
    for the column."""
    produced, out = set(), []
    for s in applied:
        consumed = step_reads(s)
        if s.get("op") not in SELECTION_OPS:
            for c in sorted(consumed - produced):
                if c not in out:
                    out.append(c)
        produced |= set((fitted.get(s["id"]) or {}).get("columns_added") or [])
    return out


# A package module is at most 400 lines (vega-forge code-contract §5, C13); a recipe is not
# bounded. Past it, the step functions go to `<stem>_steps.py`, `<stem>_steps_2.py` … beside the
# module, each within the limit, loaded by path (a package member in the bundle, a lone file in
# the twin check) with the module-level names its steps read.
MODULE_LINES = 400
STEPS_SHARED = ("FITTED", "_unseen", "LABEL", "PARTITION", "HERE")
STEPS_LOADER = """

def _steps_of(name, shared=%r):
    \"\"\"The step functions of a sibling module, run with this module's shared names.\"\"\"
    spec = importlib.util.spec_from_file_location(f"{__name__}_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update({k: globals()[k] for k in shared})
    spec.loader.exec_module(module)
    return module._STEPS


_STEPS = tuple(step for name in %r for step in _steps_of(name))
"""


def _split_steps(parts: list, stem: str, path: Path) -> str:
    """Write the step functions of `parts` (`write_transform_module`'s list) to sibling modules
    of at most MODULE_LINES lines; return the module that loads them."""
    body = parts[13][:parts[13].rindex("\n_STEPS = (")]
    names, chunks = [], [""]
    for piece in re.split(r"\n(?=\n(?:#[^\n]*\n)*def _step_)", body):
        if chunks[-1] and (chunks[-1] + piece).count("\n") > MODULE_LINES - 40:
            chunks.append("")
        chunks[-1] += piece + "\n"
    for i, chunk in enumerate(chunks):
        name = f"{stem}_steps" + (f"_{i + 1}" if i else "")
        listed = re.findall(r"^def (_step_\w+)\(", chunk, re.M)
        text = (f'"""{name} — step functions of {stem}.py, split from it by the code discipline '
                f'(C13).\nGenerated with it by `dataset-shaper/scripts/shape.py --transform-only`; '
                f'{stem}.py runs them\nwith its own {", ".join(STEPS_SHARED)}. Do not edit it.\n"""\n'
                + parts[1] + chunk + f"\n\n_STEPS = ({', '.join(listed)}{',' if len(listed) == 1 else ''})\n")
        if unbound_step_names(text + " = ".join(STEPS_SHARED) + " = None\n"):
            raise ValueError(f"{name}: a step reads a name the transform module does not share")
        path.with_name(name + ".py").write_text(text, encoding="utf-8")
        names.append(name)
    loader = STEPS_LOADER % (STEPS_SHARED, tuple(names))
    imports = parts[1].replace("import json\n", "import importlib.util\nimport json\n", 1)
    return "".join([parts[0], imports, *parts[2:13], loader, parts[14]])


LABEL_DROP = ('    parts["rows"] = parts["rows"].drop(columns=[c for c in (LABEL,) '
              'if c in parts["rows"].columns])\n')


# --------------------------------------------------------------------------- the step functions
# The transform-only module runs each applied step's lines inside a short function of its own
# (`_step_<id>`), listed in `_STEPS` in recipe order: the emitted module is held to the code
# discipline of the package it ships in (vega-forge code-contract §5.1: at most 30 statements
# per function), and a recipe's length is not bounded. The lines are the executor's, unchanged:
# this only decides where one function ends and the next begins. Two rules keep that a pure
# re-arrangement. A value a step reads from a module constant (`_f = FITTED['S07']['fitted']`,
# an import) is read again at the top of every function that uses it; any other name a line
# reads from an earlier line keeps the two in one function.
STEP_BUDGET = 30
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_PURE_NODES = (ast.Name, ast.Load, ast.Subscript, ast.Attribute, ast.Constant, ast.Tuple,
               ast.Slice)


def _stmt_cost(node) -> int:
    """`ast.stmt`s, recursively; a nested def or class is one (code-contract §5.1's count)."""
    total = 1
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return total
    for field in ("body", "orelse", "finalbody"):
        total += sum(_stmt_cost(n) for n in getattr(node, field, None) or ())
    for part in [*(getattr(node, "handlers", None) or ()), *(getattr(node, "cases", None) or ())]:
        total += sum(_stmt_cost(n) for n in part.body)
    return total


def _name_events(node, seen: dict, inner: frozenset | None = None) -> None:
    """The first use of each name in evaluation order, `load` or `store`. `inner` is the set
    of names a comprehension or a lambda binds for itself while inside one (None outside):
    there only the loads of the other names count, since nothing it binds outlives it."""
    if isinstance(node, ast.Name):
        if inner is None:
            seen.setdefault(node.id, "load" if isinstance(node.ctx, ast.Load) else "store")
        elif isinstance(node.ctx, ast.Load) and node.id not in inner:
            seen.setdefault(node.id, "load")
        return
    if isinstance(node, _COMPREHENSIONS + (ast.Lambda,)):
        bound = ({a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg)}
                 if isinstance(node, ast.Lambda) else
                 {n.id for g in node.generators for n in ast.walk(g.target)
                  if isinstance(n, ast.Name)})
        for child in ast.iter_child_nodes(node):
            _name_events(child, seen, (inner or frozenset()) | frozenset(bound))
        return
    if inner is None and isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            seen.setdefault((alias.asname or alias.name).split(".")[0], "store")
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for child in ast.walk(node):            # what its body reads, conservatively
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                seen.setdefault(child.id, "load")
        if inner is None:
            seen.setdefault(node.name, "store")
        return
    if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and inner is None:
        seen.setdefault(node.target.id, "load")
    for child in _in_evaluation_order(node):
        _name_events(child, seen, inner)


def _in_evaluation_order(node) -> list:
    """The children of a node, the value of an assignment before its targets and the iterable
    of a loop before its target (the order Python evaluates them in)."""
    fields = {ast.Assign: ("value", "targets"), ast.AnnAssign: ("value", "target"),
              ast.AugAssign: ("value", "target"),
              ast.For: ("iter", "target", "body", "orelse")}.get(type(node))
    if fields is None:
        return list(ast.iter_child_nodes(node))
    out = []
    for name in fields:
        value = getattr(node, name)
        out += value if isinstance(value, list) else [value] if value is not None else []
    return out


def _replayable(nodes) -> bool:
    """One assignment of a name from module constants (subscripts and attributes of names,
    no call), or an import: reading it again where it is used gives the same object."""
    if len(nodes) != 1:
        return False
    node = nodes[0]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return True
    return (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and all(isinstance(n, _PURE_NODES) for n in ast.walk(node.value)))


def _step_items(sid: str, lines: list) -> tuple:
    """(the statements of one step's lines as items, the comment lines after the last one).
    An item is the lines of one or more statements (statements sharing a line stay one),
    with the comment lines before it."""
    text = "\n".join(lines)
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ValueError(f"step {sid}'s emitted lines do not parse ({exc.msg}, line "
                         f"{exc.lineno}): the executor's emit for that op is wrong") from exc
    items = []
    for node in tree.body:
        if items and node.lineno <= items[-1]["end"]:
            items[-1]["nodes"].append(node)
            items[-1]["end"] = max(items[-1]["end"], node.end_lineno)
            continue
        items.append({"step": sid, "nodes": [node], "start": node.lineno,
                      "end": node.end_lineno})
    cursor = 0
    for item in items:
        item["lead"] = lines[cursor:item["start"] - 1]
        item["lines"] = lines[item["start"] - 1:item["end"]]
        cursor = item["end"]
        item["cost"] = sum(_stmt_cost(n) for n in item["nodes"])
        events = {}
        for n in item["nodes"]:
            _name_events(n, events)
        item["free"] = {k for k, v in events.items() if v == "load"}
        item["binds"] = {k for k, v in events.items() if v == "store"}
        item["replay"] = _replayable(item["nodes"])
    tail = lines[items[-1]["end"]:] if items else list(lines)
    return items, tail


def _units(items: list) -> list:
    """Items grouped so that no name crosses a function boundary: an item that reads a name an
    earlier item bound joins that item's unit, unless the binder is replayable — then it only
    records that it needs the binder read again (`needs`: name -> binder index)."""
    units, unit_of, replay_of = [], {}, {}
    for i, item in enumerate(items):
        if item["replay"] and item["free"] & (set(unit_of) | set(replay_of)):
            item["replay"] = False              # it reads a line's value, not a constant's
        item["needs"] = {n: replay_of[n] for n in item["free"] if n in replay_of}
        joined = sorted({unit_of[n] for n in item["free"] if n in unit_of})
        if item["replay"]:
            for n in item["binds"]:
                replay_of[n] = i
                unit_of.pop(n, None)
            continue
        if joined:
            first = joined[0]
            merged = [x for u in units[first:] for x in u] + [i]
            units = units[:first] + [merged]
            unit_of = {n: (first if u >= first else u) for n, u in unit_of.items()}
        else:
            units.append([i])
        for n in item["binds"]:
            unit_of[n] = len(units) - 1
            replay_of.pop(n, None)
    needed = {b for item in items for b in item.get("needs", {}).values()}
    units += [[i] for i, item in enumerate(items) if item["replay"] and i not in needed]
    return sorted(units, key=min)


def _chunk_lines(items: list, unit_ids: list) -> tuple:
    """(the body lines of one function over these items, its statement count, the steps whose
    lines it holds): every binder an item needs is read again before it, once per function
    while it is the one in force."""
    body, cost, active, steps = [], 0, {}, []
    for i in sorted(x for u in unit_ids for x in u):
        item = items[i]
        for name, binder in sorted(item["needs"].items(), key=lambda kv: kv[1]):
            if active.get(name) != binder:
                body += items[binder]["lines"]
                cost += items[binder]["cost"]
                active.update({n: binder for n in items[binder]["binds"]})
                steps.append(items[binder]["step"])
        body += item["lead"] + item["lines"]
        cost += item["cost"]
        active.update({n: i for n in item["binds"]})
        steps.append(item["step"])
    return body, cost, steps


def _chunks(items: list) -> list:
    """Functions over the units: a new one at each step boundary, and wherever the next unit
    would take the current one past STEP_BUDGET statements."""
    chunks = []
    for unit in _units(items):
        step = items[min(unit)]["step"]
        if chunks and chunks[-1]["step"] == step:
            trial = _chunk_lines(items, chunks[-1]["units"] + [unit])[1]
            if trial <= STEP_BUDGET:
                chunks[-1]["units"].append(unit)
                continue
        chunks.append({"step": step, "units": [unit]})
    return chunks


def _function_name(sid: str, taken: set) -> str:
    base = "_step_" + (re.sub(r"\W+", "_", str(sid)).strip("_").lower() or "x")
    name, n = base, 1
    while name in taken:
        n += 1
        name = f"{base}_{n}"
    taken.add(name)
    return name


def _comment(line: str) -> str:
    return "# " + line.strip().lstrip("#").strip() if line.strip() else ""


def transform_steps_source(steps: list) -> str:
    """The step functions and `_STEPS`. `steps` is the recipe in order, each entry
    `{"id", "head": [comment lines], "lines": [code lines]}`; a step with no statement (a
    skipped step, a recorded one, one that only comments) contributes its comments alone, at
    module level before the next function."""
    items, heads = [], {}
    for step in steps:
        found, tail = _step_items(step["id"], step["lines"]) if step["lines"] else ([], [])
        heads[step["id"]] = list(step["head"])
        if found:
            found[-1]["lines"] = found[-1]["lines"] + tail
        else:
            heads[step["id"]] += [_comment(x) for x in step["lines"] if x.strip()]
        items += found
    order = [step["id"] for step in steps]
    out, said, taken, names = [], set(), set(), []
    for chunk in _chunks(items):
        body, _, touched = _chunk_lines(items, chunk["units"])
        upto = max(order.index(t) for t in touched)
        new = [sid for sid in order[:upto + 1] if sid not in said]
        said.update(new)
        name = _function_name(chunk["step"], taken)
        names.append(name)
        out += [""] + [line for sid in new for line in heads[sid]] + [f"def {name}(parts):"]
        out += [("    " + x).rstrip() for x in body] + [""]
    out += [line for sid in order if sid not in said for line in heads[sid]]
    listed = ", ".join(names) + ("," if len(names) == 1 else "")
    wrapped = textwrap.wrap(f"_STEPS = ({listed})", 96, subsequent_indent="          ",
                            break_on_hyphens=False, break_long_words=False)
    return "\n".join(out + [""] + wrapped) + "\n"


def unbound_step_names(source: str) -> list:
    """`<function>: <name>` for every name a `_step_*` function of this module source reads
    that it does not bind, the module does not define and Python does not provide. Splitting
    the lines into functions must never leave a line reading a name an earlier line bound in
    another function: at inference that is a NameError on the first request, so the emitter
    refuses to write such a module (a guard over `_units`, which keeps binder and reader
    together)."""
    top = symtable.symtable(source, "<preprocess>", "exec")
    module = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}

    def scopes(table):                          # a function and its lambdas, generators …
        yield table
        for child in table.get_children():
            yield from scopes(child)
    return [f"{fn.get_name()}: {s.get_name()}" for fn in top.get_children()
            if fn.get_name().startswith("_step_") for table in scopes(fn)
            for s in table.get_symbols()
            if s.is_global() and s.is_referenced() and s.get_name() not in module
            and not hasattr(builtins, s.get_name())]


def _recorded_step(step: dict, head: str, readers: list, label_drop: str) -> dict:
    """`select_partition` at inference: the label is named, not filtered. It leaves here, or —
    when a later step reads it — after the last of those steps."""
    if readers:
        who = ", ".join(r["id"] for r in readers)
        return {"id": step["id"], "lines": [], "head": [
            head + "  — the label is named here, not filtered: at inference it is",
            f"# an output, and every row stays. {who} {'reads' if len(readers) == 1 else 'read'} "
            "it, as in the executor, so the column stays",
            f"# until {readers[-1]['id']} has run and leaves after it."]}
    return {"id": step["id"], "lines": [label_drop], "head": [
        head + "  — the label is named here, not filtered: at inference it is",
        "# an output, so the column leaves if the caller sent it and every row stays."]}


def split_key_lines(lines: list) -> list:
    """The lines a group split emitted to drop its key (from `SPLIT_KEY_MARK` on), or []."""
    at = next((i for i, ln in enumerate(lines) if ln.startswith(SPLIT_KEY_MARK)), None)
    return list(lines[at:]) if at is not None else []


def transform_steps(recipe: dict, code: dict, plan: tuple, readers: list) -> list:
    """The recipe's steps as `transform_steps_source` takes them: each step's head comment and
    the lines it runs at inference. `plan` is `transform_plan`'s (applied, recorded, skipped)."""
    _, recorded, skipped = plan
    why_of = {x["id"]: w for x, w in skipped}
    recorded_ids = {s["id"] for s in recorded}
    label_drop = LABEL_DROP.strip("\n")[4:]
    steps = []
    for s in recipe["steps"]:
        # A rationale is free prose the user wrote; folded onto one line it stays a comment
        # instead of turning the rest of the step's block into syntax.
        src = str(s.get("source") or "").replace("\n", " ")
        head = f"# --- {s['id']}  {s.get('op')}  [{src}]"
        if s["id"] in why_of:
            # a group split is skipped, but the key it dropped leaves a served row too (F33)
            key = split_key_lines(code.get(s["id"]) or []) if s.get("op") == "split" else []
            steps.append({"id": s["id"], "head": [f"{head}  — skipped: {why_of[s['id']]}"
                                                  + ("; its group key leaves here, as in the "
                                                     "executor" if key else "")],
                          "lines": key})
        elif s["id"] in recorded_ids:
            steps.append(_recorded_step(s, head, readers, label_drop))
        else:
            lines = list(code[s["id"]])
            if readers and s["id"] == readers[-1]["id"]:
                lines += ["# the label has been read by every step that reads it; it leaves "
                          f"here (named by {recorded[0]['id']})", label_drop]
            rationale = " ".join(str(s.get("rationale") or "").split())
            steps.append({"id": s["id"], "head": [f"{head}  {rationale}"], "lines": lines})
    return steps


def write_transform_module(recipe, fitted, code, path, fitted_name, out):
    """Emit the inference twin: the applied steps' own lines, over one part.

    `code` is `fitted.json.__code__` — what the executor emitted per step when it ran. This
    function selects and indents; it does not translate. An op whose emitted line cannot serve
    a query unchanged is a defect in that op's `ctx.emit`, not something to paper over here,
    because the moment the two diverge the reproduction script stops being a test of what
    serving actually does."""
    applied, recorded, skipped, warnings = transform_plan(recipe)
    label = next((((s.get("params") or {}).get("label")) for s in recipe["steps"]
                  if s.get("op") == "select_partition"), None)
    partition = next((dict((s.get("params") or {})) for s in recipe["steps"]
                      if s.get("op") == "select_partition"), None)
    lag_steps = [s for s in applied if s.get("op") in TRANSFORM_HISTORY_OPS]
    readers = label_readers(recipe, applied, label)
    lagged_label = [s for s in lag_steps if label in step_reads(s)]
    required = [c for c in required_columns(recipe, fitted, applied) if c != label]
    history_columns = required + ([label] if lagged_label else [])
    missing_code = [s["id"] for s in applied if s["id"] not in code]
    if missing_code:
        raise ValueError(f"{fitted_name} carries no emitted code for step(s) "
                         f"{missing_code}: it was written by an older executor. Re-run "
                         "shape.py on the dataset to regenerate it.")
    needs_scipy = any((s.get("params") or {}).get("kind") in ("box-cox", "yeo-johnson")
                      for s in applied if s.get("op") == "transform")
    needs_sklearn = any(s.get("op") in ("project", "spatial_features") for s in applied)
    extras = [n for n, want in (("scipy", needs_scipy), ("scikit-learn", needs_sklearn))
              if want]
    history_doc = ""
    if lag_steps:
        history_doc = (
            "\n`history` is required, not optional: " +
            ", ".join(f"{s['id']} lags {(s.get('params') or {}).get('lags') or [1]} of "
                      f"{s.get('columns')}" for s in lag_steps) +
            ". The rows preceding the query are prepended, the lags are taken over the whole\n"
            "window, and only the query rows come back.\n")
        if lagged_label:
            history_doc += (
                f"The label {label!r} is lagged: the history carries it (a query row need not;\n"
                "one that does lends it to the rows after it), and the column leaves after\n"
                f"{readers[-1]['id']}, the last step that reads it.\n")
    warnings_doc = ("\n" + "".join(f"WARN: {w}\n" for w in warnings)) if warnings else ""
    graph = graph_module(recipe, fitted, fitted_name, out)
    doc = PREPROCESS_DOC.format(
        recipe_name=Path(recipe["input"]["path"]).name,
        slug=Path(recipe["input"]["path"]).stem,
        fitted_name=fitted_name,
        extra_imports=(", plus " + " and ".join(extras) + " for a fitted step" if extras
                       else ", nothing else"),
        history_doc=history_doc + (graph["doc"] if graph else ""), warnings_doc=warnings_doc)
    # Step ids and column names reach the docstring; a triple quote among them would end it
    # early and the rest of the module would be parsed as code.
    doc = doc[:3] + doc[3:-4].replace('"""', "'''") + doc[-4:]

    no_history = ""
    if lag_steps:
        no_history = (
            "this recipe carries a lag step (" +
            "; ".join(f"{s['id']}: {s.get('columns')} at lags "
                      f"{(s.get('params') or {}).get('lags') or [1]}" for s in lag_steps) +
            "), and a lagged feature is the value from the rows before this one. Pass them: "
            "predict.py --history <file>, or transform(df_raw, history=<DataFrame>). Filling "
            "the lags with NaN instead would hand the network a row it scores without "
            "complaint.")

    # numpy only when a step's own line reaches for it — an import nothing uses is a lint
    # failure in a file this skill hands to someone else's package.
    body = [ln for s in applied for ln in code[s["id"]]]
    # numpy when a line names it — `np.` in most ops, the bare name in a custom step's
    # namespace ({'pd': pd, 'np': np}), which a substring test for "np." misses.
    imports = PREPROCESS_IMPORTS + ("import numpy as np\n" if graph is not None or
                                    any(re.search(r"\bnp\b", ln) for ln in body)
                                    else "") + "import pandas as pd\n"
    parts = [doc, imports, graph["prelude"] if graph else PREPROCESS_PRELUDE, "\n",
             f"_FITTED_NAMES = {py(list(dict.fromkeys([fitted_name, 'fitted.json'])))}\n",
             "FITTED = _load_fitted()\n",
             f"LABEL = {py(label)}\n",
             f"PARTITION = {py(partition)}\n",
             # The target is what the model predicts; a query never carries it, so it is never
             # required — a step that reads it in place runs on the placeholder transform() adds.
             f"REQUIRED_COLUMNS = {py(required)}\n",
             f"NEEDS_HISTORY = {py(bool(lag_steps))}\n",
             # The columns the history must carry: the query's, and the label when a lag reads
             # it — without it every lag of the target would be NaN, imputed and scored.
             f"HISTORY_COLUMNS = {py(history_columns)}\n",
             f"_NO_HISTORY = {py(no_history)}\n",
             PREPROCESS_TRANSFORM_HEAD]
    parts.append(transform_steps_source(transform_steps(recipe, code,
                                                        (applied, recorded, skipped), readers)))
    parts.append(PREPROCESS_TRANSFORM_TAIL + (graph["source"] if graph else ""))
    lost = unbound_step_names("".join(parts))
    if lost:
        raise ValueError("the transform-only module would read name(s) its step functions "
                         "never bind: " + "; ".join(lost) + ". The split of the steps' lines "
                         "into functions separated a binder from its reader; nothing written")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text, stem = "".join(parts), Path(path).stem
    for stale in Path(path).parent.glob(f"{stem}_steps*.py"):
        stale.unlink()
    if graph is None and text.count("\n") > MODULE_LINES:
        text = _split_steps(parts, stem, Path(path))
        out(f"OK: the step functions are in {stem}_steps*.py beside {Path(path).name}: one module "
            f"would pass the {MODULE_LINES} lines of the code discipline (C13)")
    Path(path).write_text(text, encoding="utf-8")

    out(f"OK: {path} emits transform(df_raw, history=None) from {len(applied)} applied step(s): "
        + ", ".join(f"{s['id']} {s['op']}" for s in applied))
    for s in recorded:
        out(f"OK: {s['id']} select_partition recorded, not applied — the label {label!r} is "
            "dropped when present" + (f", after {readers[-1]['id']} (the last step that reads "
                                      "it)" if readers else "") + ", and no row is filtered out")
    for s, why in skipped:
        out(f"OK: {s['id']} {s['op']} skipped — {why}")
    for w in warnings:
        out(f"WARN: {w}")
    if lag_steps:
        out("WARN: this recipe lags: transform() raises unless the caller passes the preceding "
            "rows (predict.py --history <file>)")
    return applied, recorded, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", help="recipe.json (required unless --check)")
    ap.add_argument("--model", default=None, help="the model HTML whose layers the sources cite")
    ap.add_argument("--out-dir", default="shaped", help="where the outputs go")
    ap.add_argument("--format", default="parquet", choices=("parquet", "csv"))
    ap.add_argument("--check-only", action="store_true", help="validate the recipe and stop")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the recipe, the manifest skeleton and the reproduction script; "
                         "materialize no data")
    ap.add_argument("--check", action="store_true",
                    help="re-run the reproduction script in --out-dir and compare digests")
    ap.add_argument("--transform-only", action="store_true",
                    help="emit the inference twin from the recipe and --fitted; read no data "
                         "and write no dataset")
    ap.add_argument("--fitted", default=None, help="fitted.json (with --transform-only)")
    ap.add_argument("--emit", default=None,
                    help="where the transform-only module goes (with --transform-only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sep", default=None)
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    if a.check:
        man_p = out_dir / "manifest.json"
        if not man_p.is_file():
            print(f"ERROR: {man_p} not found; run the executor first")
            return 2
        man = json.loads(man_p.read_text(encoding="utf-8"))
        script = out_dir / man["outputs"]["reproduce"]
        if not script.is_file():
            print(f"ERROR: {script} not found")
            return 2
        before = {n: v.get("digest") for n, v in man["outputs"]["files"].items()}
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                              timeout=1800)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            print(f"ERROR: the reproduction script exited {proc.returncode}")
            sys.stderr.write(proc.stderr[-2000:])
            return 3
        after = {}
        for n, v in man["outputs"]["files"].items():
            p = out_dir / Path(v["path"]).name
            after[n] = digest_file(p) if p.is_file() else None
        bad = {n: (before[n], after[n]) for n in before if before[n] != after[n]}
        if bad:
            print(f"ERROR: determinism check FAILED for {sorted(bad)}")
            for n, (x, y) in bad.items():
                print(f"  {n}: manifest {x} != rerun {y}")
            record_determinism(out_dir, "fail")
            return 3
        print(f"OK: determinism check passed — {len(after)} output file(s) reproduce byte for byte")
        record_determinism(out_dir, "pass")
        return 0

    if not a.recipe:
        print("ERROR: --recipe is required (or --check)")
        return 2
    try:
        recipe = json.loads(Path(a.recipe).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read recipe {a.recipe}: {e}")
        return 2
    layers = read_layers(a.model) if a.model else {}
    if a.model and not layers:
        mark(f"WARN: {a.model} carries no readable layer; sources cannot be resolved")
    errs = check_recipe(recipe, layers, mark)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2
    n_steps = len(recipe["steps"])
    sources = {}
    for s in recipe["steps"]:
        sources[str(s.get("source", "")).split(":")[0]] = \
            sources.get(str(s.get("source", "")).split(":")[0], 0) + 1
    print(f"OK: {a.recipe} validates against {SCHEMA} ({n_steps} steps; sources " +
          ", ".join(f"{k} {v}" for k, v in sorted(sources.items())) + ")")
    if a.check_only:
        return 0

    if a.transform_only:
        # The dataset is deliberately not read here: this mode exists to hand /vega-pack a
        # module for rows that do not exist yet, and a step that needed the training data to
        # emit would be a step that re-fits at inference.
        if not a.fitted or not a.emit:
            print("ERROR: --transform-only needs --fitted <fitted.json> and --emit <path>")
            return 2
        try:
            fitted_blob = json.loads(Path(a.fitted).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"ERROR: cannot read the fitted parameters {a.fitted}: {e}")
            return 2
        code = fitted_blob.get("__code__")
        if not code:
            print(f"ERROR: {a.fitted} carries no __code__ block; it was written by an executor "
                  "older than transform-only mode. Re-run shape.py on the dataset to "
                  "regenerate it.")
            return 2
        try:
            write_transform_module(recipe, fitted_blob, code, Path(a.emit),
                                   Path(a.fitted).name, mark)
        except (ValueError, OSError) as e:
            print(f"ERROR: could not emit the transform-only module: {e}")
            return 3
        return 0

    input_path = Path(recipe["input"]["path"])
    try:
        df = read_dataset(input_path, a.sep)
    except Exception as e:
        print(f"ERROR: cannot read the dataset {input_path}: {e}")
        return 2
    dig = digest_file(input_path)
    if recipe["input"].get("digest") and recipe["input"]["digest"] != dig:
        mark(f"WARN: input digest differs from the recipe's ({dig} != "
             f"{recipe['input']['digest']}): the data is not the one the analysis saw")
    seed = int(recipe.get("seed", a.seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        ctx = execute(recipe, layers, df, seed)
    except Exception as e:
        print(f"ERROR: execution stopped at a step: {e}")
        return 3

    slug = input_path.stem
    fmt = a.format
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            fmt = "csv"
            mark("WARN: pyarrow is not installed; writing csv instead of parquet")
    files, outputs = {}, {}
    for name in ctx.order + (["edges"] if ctx.edges is not None else []):
        fname = f"{slug}.{name}.{fmt}" if name != "all" else f"{slug}.{fmt}"
        files[name] = fname
    fitted_blob = dict(ctx.fitted)
    split_rec = next((m for m in ctx.manifest if m["op"] == "split"), None)
    fitted_blob["__split__"] = (split_rec["parameters_fitted"]["indices"] if split_rec else {})
    # `__split__` addresses the frame the split saw; `__split_raw__` addresses the raw file.
    # They differ exactly when a dedupe removed rows before the split, and the twin check
    # needs the second: it fetches a shaped part's raw rows out of the input file.
    fitted_blob["__split_raw__"] = (split_rec["parameters_fitted"].get("raw_indices")
                                    if split_rec else {})
    fitted_blob["__removed_rows__"] = ctx.removed_rows
    # The lines the executor emitted, per step. `--transform-only` re-emits a filtered subset
    # of exactly these over one part, which is what makes the transformer and the
    # reproduction script one implementation rather than two.
    fitted_blob["__code__"] = ctx.code_by_step
    fitted_name = "fitted.json"
    (out_dir / fitted_name).write_text(json.dumps(jsonable(fitted_blob), indent=1,
                                                  sort_keys=True) + "\n", encoding="utf-8")
    if not a.dry_run:
        for name in ctx.order:
            p = out_dir / files[name]
            frame = ctx.parts[name].reset_index(drop=True)
            if fmt == "parquet":
                frame.to_parquet(p, index=False)
            else:
                frame.to_csv(p, index=False)
            outputs[name] = {"path": str(p), "digest": digest_file(p), "rows": int(len(frame)),
                             "columns": int(frame.shape[1])}
            mark(f"OK: wrote {p} ({len(frame)} rows × {frame.shape[1]} columns)")
    else:
        for name in ctx.order:
            outputs[name] = {"path": str(out_dir / files[name]), "digest": None,
                             "rows": int(len(ctx.parts[name])),
                             "columns": int(ctx.parts[name].shape[1])}
        mark("WARN: --dry-run: no data file was written; the recipe, the manifest skeleton and "
             "the reproduction script are on disk")

    outputs.update(write_edges(ctx, out_dir / files.get("edges", "-"), fmt, a.dry_run))
    repro = write_reproduction(ctx, out_dir, slug, fmt, files,
                               {k: v.get("digest") for k, v in outputs.items()},
                               input_path, fitted_name)
    schema_out = {}
    ref = ctx.parts[ctx.order[0]]
    for c in ref.columns:
        origin = ctx.lineage.get(c) or []
        schema_out[c] = {"dtype": str(ref[c].dtype),
                         "role": ("label" if c == ctx.label else
                                  "key" if c == (ctx.graph or {}).get("graph_id") else "feature"),
                         "origin": origin[-1] if origin else "input"}
    manifest = {
        "schema": "dataset-shaper/manifest@1",
        "input": {"path": str(input_path), "digest": dig, "rows": int(len(df)),
                  "columns": int(df.shape[1])},
        "recipe_digest": digest_text(json.dumps(recipe, sort_keys=True)),
        "seed": seed, "label": ctx.label, "crs": ctx.crs,
        "library_versions": {"python": sys.version.split()[0], "pandas": pd.__version__,
                             "numpy": np.__version__},
        "steps": jsonable(ctx.manifest),
        "output_schema": schema_out,
        "outputs": {"dir": str(out_dir), "format": fmt, "files": jsonable(outputs),
                    "manifest": "manifest.json", "lineage": "lineage.json",
                    "recipe": "recipe.json", "fitted": fitted_name, "reproduce": repro},
        "markers": MARKERS,
        # beside steps[]: what /vega-forge reads as `shape:manifest/graph`
        **({"graph": ctx.graph} if ctx.graph is not None else {})}
    (out_dir / "manifest.json").write_text(json.dumps(jsonable(manifest), indent=1) + "\n",
                                           encoding="utf-8")
    lineage = {"columns": {c: {"touched_by": ctx.lineage.get(c) or [],
                               "present_in_output": c in ref.columns}
                           for c in sorted(set(list(ctx.lineage) + list(df.columns)))},
               "removed": ctx.removed,
               "removed_rows": ctx.removed_rows,
               "steps": [{"id": m["id"], "op": m["op"], "source": m["source"],
                          "columns_added": m["columns_added"],
                          "columns_removed": m["columns_removed"]} for m in ctx.manifest]}
    (out_dir / "lineage.json").write_text(json.dumps(jsonable(lineage), indent=1) + "\n",
                                          encoding="utf-8")
    (out_dir / "recipe.json").write_text(json.dumps(recipe, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(f"OK: manifest.json, lineage.json, recipe.json, {fitted_name} and {repro} written to "
          f"{out_dir}")
    print(f"OK: output schema — {len(schema_out)} columns"
          + (f", label {ctx.label!r}" if ctx.label else ", no label"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
