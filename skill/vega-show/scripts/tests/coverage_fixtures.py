#!/usr/bin/env python3
"""
coverage_fixtures.py — a packed bundle for every admitted (family, task, shape, goal) cell, and
the fact-level coverage check over all of them.

    python3 coverage_fixtures.py            # extract the facts of every cell and check them
    from coverage_fixtures import cells, cell_docs, write_bundle

`cells()` reads the vega-forge catalog's `admission` rows (every family × sample shape × graph
level × effective task, `anomaly` with and without a label, plus the named variants): the closed
definition of complete coverage. `cell_docs(cell)` builds the records a bundle of that cell pins,
in the shapes the chain writes them (the netspec with the family's block types, the shaper's
manifest, the ladder with a floor rung and a one-change rung, the evaluation with the task's
metric set, its per-unit breakdown, `ci_level` and its verdict — on an unlabelled anomaly lane
vega-eval's not-assessable one, A30), and `write_bundle(root, docs)` writes them
with `MANIFEST.files` digests, a `predict.py` whose `--check` passes, and a model page that
carries every required layer, its `bundle` layer's manifest equal to `MANIFEST.json` unless
`layer_manifest` edits it (D5's seeded defect 23).

The fact tests share `run()`, `refused()`, `write_report()`, `rename_input()` and two CIDS-shaped
rungs: `FLOOR_RUNG` (R1, prose saying "floor") and `MASK_RUNG` (R6, an added block masking one).

`check_cell()` runs `extract_facts.py` on a cell, as the CLI (every preflight gate included),
and asserts what the fact set must say for that cell: the task as the catalog maps it (kind,
base, goal, shape, level, view key, direction), the reserved values, the results facts of the
task (none of the label-only ones on an unlabelled anomaly lane), the breakdown (its kind, and
its `names`, `values` and `trivial` fact keys, lists of one length: `breakdown_problems()`), the
sample facts of windows and graphs, the two sides of a two-sided task, the decisions, the
`sizes` group as D5 §6.2 shapes it (`sizes_problems()`) and the `runs` group: D5 §6.2's fields,
the fixture ladder's mechanical labels, each passing the digit rule with placeholders that are
fact keys (`runs_problems()`).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))

import extract_facts  # noqa: E402
from digit_rule import DigitRule, check_placeholders  # noqa: E402
from fact_sink import Catalogs  # noqa: E402
from vega_show_core import FactSet, digest_file  # noqa: E402

CATALOGS = Catalogs.load()
FORGE, SHOW = CATALOGS.forge, CATALOGS.show
FEATURES = (("alpha", "numeric"), ("beta", "numeric"), ("gamma", "numeric"), ("zone", "nominal"))
CLASSES = {"binary": ["no", "yes"], "multiclass": ["calm", "busy", "jammed"],
           "multilabel": ["red", "green", "blue"], "ordinal": ["low", "mid", "high"]}
PARAMS = {"dense": {"units": 8}, "embedding": {"columns": ["zone"], "dim": 3}, "dropout": {"p": 0.1},
          "activation": {"name": "gelu"}, "head": {"out": 1}, "graph_pool": {"readout": "mean"},
          "residual": {"repeat": 1, "inner": [{"type": "dense", "params": {"units": 8}}]},
          "sequential": {"inner": [{"type": "dense", "params": {"units": 8}}]},
          "gru": {"hidden": 8}, "lstm": {"hidden": 8}, "rnn": {"hidden": 8}, "conv1d": {"filters": 4}}
LAYER = ('<!-- @LAYER:start {name} v1\n     produced-by: /coverage\n     produced-at: '
         '2026-09-30T00:00:00Z\n     input-digest: sha256:{digest}\n     reverts-by: open the file at '
         'input-digest (the predecessor)\n -->\n<script id="layer-{name}-data" type="application/json">'
         '{data}</script>\n<script id="layer-{name}-render" type="text/javascript"></script>\n'
         '<!-- @LAYER:end {name} -->\n')


def cells() -> list[dict]:
    """Every admitted cell: {family, shape, level, task, variant}."""
    out = []
    for row in FORGE["admission"]["rows"]:
        for family in row["families"]:
            out += [{"family": family, "shape": row["shape"], "level": row["level"], "task": task,
                     "variant": None} for task in row["tasks"]]
        out += [{"family": v["family"], "shape": row["shape"], "level": row["level"],
                 "task": v["task"], "variant": v["variant"]} for v in row["variants"]]
    return out


def task_of(cell: dict) -> tuple[str, str | None, bool]:
    """(kind, base, labelled) of a cell's task: `sequence-regression/ordinal`, `anomaly/labelled`."""
    kind, _, rest = cell["task"].partition("/")
    base = rest if kind.startswith("sequence-") else ("regression" if kind == "forecast" else None)
    return kind, base, rest != "unlabelled"


def primary_of(kind: str, base: str | None, labelled: bool) -> str:
    if kind == "anomaly":
        return "pr_auc" if labelled else FORGE["tasks"]["anomaly"]["primary_alt"]["metric"]
    return FORGE["tasks"][base if kind.startswith("sequence-") else kind]["primary"]


def breakdown_of(kind: str, base: str | None) -> dict | None:
    spec = SHOW["breakdown"]["by_task"][kind]
    return SHOW["breakdown"]["by_task"][base] if spec == "base" else spec


def cell_docs(cell: dict) -> dict:
    """{relative path: JSON document} of a cell's bundle (MANIFEST.json without its files)."""
    kind, base, labelled = task_of(cell)
    spec = _netspec(cell, kind, base, labelled)
    return {"netspec.json": spec, "schema.json": {"input": [{"name": n} for n, _ in FEATURES]},
            "records/geometry.json": {"source": {"path": "data/demo-set.csv", "rows": 900, "columns": 6},
                                      "typing": [{"column": n, "final_type": r} for n, r in FEATURES]},
            "records/analysis.json": {"schema": "data-lens/analysis@1", "context": {"spatial": None},
                                      "modules": {"relations": {"evidence": {"pairs": [
                                          {"a": "alpha", "b": "beta", "spearman": 0.91}]}},
                                          "drift": {"columns": []}}},
            "records/shape.json": _shape(cell), "records/training.json": _training(),
            "records/architecture.json": {"netspec": spec,
                                          "smoke": {"params_total": 321, "madds_per_sample": 300}},
            "records/evaluation.json": _evaluation(kind, base, labelled, spec),
            "MANIFEST.json": _manifest(cell, kind)}


def _netspec(cell: dict, kind: str, base: str | None, labelled: bool) -> dict:
    classes = CLASSES.get(base if kind.startswith("sequence-") else kind)
    task = {"kind": kind, "label": "target" if labelled and kind != "unsupervised" else None}
    if kind == "anomaly" and labelled:
        classes, task["positive"] = ["normal", "fraud"], "fraud"
    if classes:
        task["classes"] = list(classes)  # a copy: a test that edits a bundle's classes leaves CLASSES whole
    if kind.startswith("sequence-"):
        task["base_kind"] = base
    types = FORGE["families"][cell["family"]]["block_types"]
    blocks = [{"id": f"B{i}", "type": t, "params": copy.deepcopy(PARAMS.get(t, {})),
               "rationale": "fixture", "alternatives": []} for i, t in enumerate(types, 1)]
    decided = {"source": "vega:default", "rationale": "fixture", "alternatives": [{"when": "never"}]}
    return {"task": task, "family": {"name": cell["family"], **decided}, "blocks": blocks,
            "data_contract": {"features": [{"name": n, "role": r, "origin": "S3"} for n, r in FEATURES],
                              "input_dim": len(FEATURES), "sample": _sample(cell, kind),
                              "class_balance": {"train": {c: round(1 / len(classes), 4) for c in classes}}
                              if classes else None},
            "loss": {"name": "mse", "params": {"reduction": "mean"}, **decided},
            "metrics": {"primary": primary_of(kind, base, labelled), **decided},
            "optimizer": {"name": "adamw", "params": {"lr": 0.001}, **decided},
            "schedule": {"name": "cosine", "params": {"warmup_epochs": 2}, **decided},
            "training": {"batch_size": 32, "max_epochs": 5, **decided},
            "init": {"scheme": "kaiming_uniform", **decided},
            "early_stopping": {"patience": 3, "min_delta": 0.001, **decided},
            "regularization": [{"kind": "dropout", "value": 0.1, **decided}]}


def _sample(cell: dict, kind: str) -> dict:
    if cell["shape"] == "window":
        out = {"shape": "window", "length": 8, "stride": 2 if cell["variant"] == "stride 2" else 1,
               "series_by": None, "order_by": "t", "context": "previous-part",
               "autoregressive": kind == "forecast"}
        return {**out, "horizon": 3} if kind == "forecast" else out
    if cell["shape"] == "graph":
        return {"shape": "graph", "edges": "shaped/demo.edges.parquet", "level": cell["level"],
                "graph_id": "g" if cell["level"] == "graph" else None,
                "protocol": "transductive" if cell["variant"] == "transductive" else "inductive",
                "batching": "graphs" if cell["level"] == "graph" else
                ("sampled" if cell["variant"] == "sampled" else "full")}
    return {"shape": "row"}


def _shape(cell: dict) -> dict:
    parts = {"train": 540, "valid": 180, "test": 180}
    steps = [{"id": "S1", "op": "dedupe", "rows_before": {"all": 900}, "rows_after": {"all": 900},
              "columns_added": [], "columns_removed": ["note"]},
             {"id": "S2", "op": "split", "rows_before": {"all": 900}, "rows_after": parts,
              "columns_added": [], "columns_removed": [], "parameters_fitted": {"kind": "time"}},
             {"id": "S3", "op": "scale", "rows_before": parts, "rows_after": parts,
              "columns_added": [], "columns_removed": [], "fit_on": "train"}]
    manifest = {"input": {"path": "data/demo-set.csv", "rows": 900, "columns": 6}, "steps": steps,
                "outputs": {"files": {p: {"rows": n} for p, n in parts.items()}}}
    if cell["shape"] == "graph":
        manifest["graph"] = {"method": "knn", "params": {"k": 5}, "n_edges": 4500, "level": cell["level"]}
    return {"schema": "dataset-shaper/shape@1", "manifest": manifest,
            "recipe": {"steps": [{"id": s["id"], "op": s["op"], "rationale": "fixture"} for s in steps]},
            "readings": {"abstract": "The data was shaped."}}


def _training() -> dict:
    runs = [{"id": "R0", "kind": "baseline", "delta": [], "status": "done", "epochs": 5,
             "best_epoch": 4, "primary_name": "m", "primary": 0.5, "eligible": True},
            {"id": "R1", "kind": "ablation", "delta": [{"path": "family", "from": {"name": "x"},
                                                        "to": {"name": "linear"}}],
             "status": "done", "epochs": 5, "best_epoch": 3, "primary": 0.4, "eligible": True},
            {"id": "R2", "kind": "ablation", "delta": [{"path": "optimizer/params/lr", "from": 0.001,
                                                        "to": 0.0003}],
             "status": "done", "epochs": 5, "best_epoch": 5, "primary": 0.45, "eligible": True}]
    return {"schema": "vega/training@1", "campaign": {"run_dir": "vega-runs/demo"},
            "experiments": {"runs": runs, "hpo": [],
                            "selection": {"run": "R0", "rule": "best valid primary", "value": 0.5,
                                          "margin_over_baseline": 0.0}},
            "readings": {"abstract": "One campaign."}}


# vega-eval's not-assessable verdict (eval-contract §5, A30), as an unlabelled anomaly lane records it
NOT_ASSESSABLE = {"earned": None, "strongest_baseline": None, "diff": None, "ci": [],
                  "beaten_by_baseline": None,
                  "reason": "the primary reads no label: each model's anomaly score lives on a "
                            "scale of its own, so no paired difference is computed"}


def _evaluation(kind: str, base: str | None, labelled: bool, spec: dict) -> dict:
    primary = spec["metrics"]["primary"]
    est = {"value": 0.7, "ci": [0.6, 0.8], "n": 180}
    shown = SHOW["results_facts"][kind] if labelled else []
    network = {m: dict(est) for m in {primary, *shown}}
    trivial = {primary: {"value": 0.3, "ci": [0.2, 0.4], "n": 180}}
    per_unit = breakdown_of(kind, base)
    if per_unit:
        size = {"class": 3, "label": 3, "feature": len(FEATURES), "horizon": 3}[per_unit["kind"][4:]]
        for model, value in ((network, 0.6), (trivial, 0.1)):
            model[per_unit["metric"]] = {"value": None, "values": [value] * size, "n": 180,
                                         "note": "a per-element vector"}
    verdict = {"earned": True, "strongest_baseline": "trivial", "diff": 0.4, "beaten_by_baseline": False}
    differences, reading = {"trivial": {"value": 0.4, "ci": [0.3, 0.5], "n": 180}}, "Earned."
    if kind == "anomaly" and not labelled:   # A30: scores on each model's own scale, no paired verdict
        verdict, differences = dict(NOT_ASSESSABLE), {}
        reading = "Not assessable: no model is ranked against another."
    return {"schema": "vega/evaluation@1",
            "metrics": {"primary": primary, "n_test": 180, "ci_level": 0.95, "verdict": verdict,
                        "models": {"network": network, "trivial": trivial},
                        "differences": differences},
            "seal": {"exposures": 1}, "threshold": {"applies": kind in ("binary", "anomaly"), "value": 0.5},
            "robustness": {"checks": [{"check": "time-quartiles", "status": "skipped"}]},
            "slices": {"slices": []}, "verdict": verdict,
            "readings": {"verdict": reading, "limits": "One split."},
            "markers": ["OK: sealed", "WARN: one seed"]}


def _manifest(cell: dict, kind: str) -> dict:
    return {"schema": "vega/bundle@1", "slug": f"demo-{cell['family']}", "run": "R0",
            "model_version": "sha256:" + "0" * 64, "stances": [],
            "roundtrip": {"status": "pass"}, "examples": {"kind": "test"},
            "output": {"horizons": [1, 2, 3] if kind == "forecast" else None},
            "serve": {"present": True, "identity": {"title": "demo set scorer", "noun": "row"},
                      "showcase_slot": {"dir": "showcase", "schema": "vega/showcase@1"}}}


def write_bundle(root: Path, docs: dict, layer_manifest=None) -> tuple[Path, Path]:
    """Write a cell's bundle and model page under `root`; (model page, bundle dir)."""
    bundle = Path(root) / "bundle" / "demo-R0"
    manifest = dict(docs["MANIFEST.json"], files={})
    for rel, doc in docs.items():
        if rel != "MANIFEST.json":
            data = json.dumps(doc, indent=1).encode("utf-8")
            (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
            (bundle / rel).write_bytes(data)
            manifest["files"][rel] = "sha256:" + hashlib.sha256(data).hexdigest()
    (bundle / "predict.py").write_text("import sys\nsys.exit(0 if '--check' in sys.argv else 2)\n")
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shown = layer_manifest(json.loads(json.dumps(manifest))) if layer_manifest else manifest
    turtle = "@prefix ex: <urn:x#> .\nex:Demo a ex:Thing .\n"
    digest = hashlib.sha256(turtle.encode()).hexdigest()
    layers = "".join(LAYER.format(name=n, digest=digest, data=json.dumps(
        {"manifest": shown, "markers": ["WARN: shipped by stance"]} if n == "bundle" else {}))
        for n in SHOW["stage"]["required_layers"])
    model = Path(root) / "demo.domain.analysis.shaped.vega.train.eval.pack.html"
    model.write_text(f'<html><body>\n<script id="domain-model" type="text/turtle">{turtle}</script>\n'
                     f"{layers}</body></html>\n", encoding="utf-8")
    return model, bundle


def extract(model: Path, bundle: Path, out: Path, extra=()) -> tuple[int, dict | None, str]:
    """Run the CLI in-process from the model's directory, so no report of the caller's working
    directory is found: (exit code, the fact set written or None, what it printed)."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed), contextlib.chdir(model.parent):
        code = extract_facts.main(["--model", str(model), "--bundle", str(bundle), "--out",
                                   str(out), *extra])
    doc = json.loads(out.read_text(encoding="utf-8")) if code == 0 else None
    return code, doc, printed.getvalue()


def check_cell(cell: dict, tmp: Path) -> list[str]:
    """What is wrong with the fact set of one cell (empty when nothing is)."""
    docs = cell_docs(cell)
    model, bundle = write_bundle(tmp, docs)
    code, doc, printed = extract(model, bundle, tmp / "facts.json")
    if code != 0:
        return [f"extract_facts exited {code}: {printed.strip()}"]
    kind, base, labelled = task_of(cell)
    task, facts, groups = doc["task"], doc["facts"], doc["groups"]
    primary = primary_of(kind, base, labelled)
    want = {"kind": kind, "base": base, "sample_shape": cell["shape"], "graph_level": cell["level"],
            "family": cell["family"], "primary": primary, "direction": CATALOGS.direction(primary),
            "goal": {v: k for k, v in FORGE["goal_map"].items() if v}.get(kind, "predict")}
    bad = [f"task.{k} {task[k]!r} ≠ {v!r}" for k, v in want.items() if task[k] != v]
    assessable = kind != "anomaly" or labelled
    keys = {f"eval_network_{primary}", "eval_ci_level", "train_rows", "source_rows", "runs_n",
            "selected_primary", "block_b1_alternatives_n",
            "eval_diff_trivial" if assessable else "eval_verdict_reason"}
    spec = breakdown_of(kind, base)
    vector = spec["metric"] if spec else None  # a per-element vector is no point estimate
    keys |= {f"eval_network_{m}" for m in SHOW["results_facts"][kind] if labelled and m != vector}
    keys |= {"window_length", "window_stride"} if cell["shape"] == "window" else set()
    keys |= {"horizon"} if kind == "forecast" else set()
    keys |= {"edges_n", "graph_edges_k"} if cell["shape"] == "graph" else set()
    bad += [f"no fact {k}" for k in sorted(keys - set(facts))]
    bad += [f"label-only fact {k} on an unlabelled lane" for k in facts
            if not labelled and k in {f"eval_network_{m}" for m in ("roc_auc", "precision_at_k", "recall_at_k")}]
    bad += breakdown_problems(doc, spec)
    sides = 2 if kind in ("binary", "anomaly") else None
    bad += [f"outcomes {task['outcomes']!r}"] if (len(task["outcomes"] or []) or None) != sides else []
    ids = {d["id"] for d in groups["decisions"]}
    bad += [f"no decision {i}" for i in ("family", "B1", "loss", "regularization[dropout]") if i not in ids]
    bad += [f"runs {[r['id'] for r in groups['runs']]}"] if len(groups["runs"]) != 3 else []
    bad += sizes_problems(doc, classes="classes" in docs["netspec.json"]["task"])
    bad += runs_problems(doc, FIXTURE_LABELS)
    limits = {lim["id"] for lim in groups["limits"]}
    bad += [f"no limit {i}" for i in ("split", "exposures", "one_seed", "warn_eval_1", "warn_pack_1")
            if i not in limits]
    return bad + verdict_problems(doc, assessable)


def verdict_problems(doc: dict, assessable: bool) -> list[str]:
    """The `verdict` group against the evaluation's: an assessed verdict has a boolean `earned`
    and no reason; one that is not assessable (A30, an unlabelled anomaly lane) has `earned` and
    `beaten_by_baseline` null, no baseline or difference, its reason as the fact
    `eval_verdict_reason` and the limit `verdict_not_assessable` — and no paired difference."""
    verdict, facts = doc["groups"]["verdict"], doc["facts"]
    limits = {lim["id"] for lim in doc["groups"]["limits"]}
    if assessable:
        return ([f"verdict {verdict}"] if not isinstance(verdict["earned"], bool)
                or verdict.get("reason") is not None else []) + \
            (["verdict_not_assessable on an assessed verdict"] if "verdict_not_assessable" in limits
             else [])
    want = {"earned": None, "beaten_by_baseline": None, "strongest_baseline": None, "diff": None,
            "reason": "eval_verdict_reason"}
    bad = [f"verdict.{k} {verdict.get(k)!r} ≠ {v!r}" for k, v in want.items() if verdict.get(k) != v]
    reason = facts.get("eval_verdict_reason") or {}
    if reason.get("value") != NOT_ASSESSABLE["reason"] or reason.get("format") != "verbatim":
        bad.append(f"eval_verdict_reason {reason}")
    bad += [f"paired difference {k} on a verdict that is not assessable" for k in facts
            if k.startswith("eval_diff_")]
    return bad + ([] if "verdict_not_assessable" in limits else ["no limit verdict_not_assessable"])


RUN_FIELDS = ("id", "kind", "family", "primary", "epochs", "best_epoch", "delta", "features",
              "label", "selected", "reference", "eligible", "status")
FIXTURE_LABELS = {"R0": None, "R1": "family {run_r1_delta_0_from} → {run_r1_delta_0_to}",
                  "R2": "optimizer lr {run_r2_delta_0_from} → {run_r2_delta_0_to}"}


def sizes_problems(doc: dict, classes: bool) -> list[str]:
    """What is wrong with the `sizes` group against D5 §6.2: `{source_rows, source_columns,
    shaped_inputs, parts: {train, valid, test}}`, every value a fact key, and `classes_n` exactly
    when the task has classes."""
    sizes, facts = doc["groups"]["sizes"], doc["facts"]
    want = ["source_rows", "source_columns", "shaped_inputs", "parts"] + (["classes_n"] if classes else [])
    bad = [f"sizes fields {list(sizes)} ≠ {want}"] if list(sizes) != want else []
    parts = sizes.get("parts") or {}
    bad += [f"sizes.parts {parts}"] if parts != {p: f"{p}_rows" for p in ("train", "valid", "test")} else []
    keys = [v for k, v in sizes.items() if k != "parts"] + list(parts.values())
    return bad + [f"sizes value {k!r} is no fact" for k in keys if k not in facts]


def runs_problems(doc: dict, labels: dict | None = None) -> list[str]:
    """What is wrong with the `runs` group against D5 §6.2: the fields of every run, in order;
    `features` the union of its changes' inputs; a label wherever the run changed something
    (equal to `labels[id]` when given), passing the digit rule over the fact set's names, its
    placeholders fact keys."""
    facts = FactSet.from_json(doc)
    rule, bad = DigitRule.for_facts(facts), []
    for run in doc["groups"]["runs"]:
        rid, label = run["id"], run["label"]
        bad += [f"run {rid} fields {list(run)}"] if tuple(run) != RUN_FIELDS else []
        named = [n for d in run["delta"] for n in d.get("features", [])]
        bad += [f"run {rid} features {run['features']} ≠ {named}"] if sorted(set(named)) != sorted(run["features"]) else []
        bad += [f"run {rid} label {label!r} ≠ {labels[rid]!r}"] if labels and labels.get(rid, label) != label else []
        if (label is None) != (not run["delta"]):
            bad.append(f"run {rid}: label {label!r} for a delta of {len(run['delta'])} change(s)")
        if label is not None:
            where = f"groups.runs[id={rid}].label"
            bad += [str(p) for p in rule.refusals(label, where) + check_placeholders(label, where, facts.keys())]
    return bad


def breakdown_problems(doc: dict, spec: dict | None) -> list[str]:
    """What is wrong with the fact set's `breakdown` for a task whose breakdown is `spec`: its
    kind, and the facts the resolver computes `weak` from (H05, R-E3) — `names`, `values` and
    `trivial` are fact keys whose lists are of one length."""
    facts, got = doc["facts"], doc["groups"].get("breakdown") or {}
    want = spec and spec["kind"]
    if got.get("kind") != want:
        return [f"breakdown {got.get('kind')!r} ≠ {want!r}"]
    if not spec:
        return []
    sides = ("names", "values", "trivial")
    bad = [f"breakdown.{side} {got.get(side)!r} is no fact" for side in sides
           if got.get(side) is None or got[side] not in facts]
    if bad:
        return bad
    lengths = {side: len(v) if isinstance(v := facts[got[side]]["value"], list) else None
               for side in sides}
    return [] if None not in lengths.values() and len(set(lengths.values())) == 1 else \
        [f"breakdown lists of different lengths or no lists: {lengths}"]


# --------------------------------------------------------------------------- shared by the fact tests

_RUNG = {"kind": "ablation", "status": "done", "epochs": 5, "eligible": True}
# the floor rung as the CIDS ladder writes it (R1): the linear family's own prose says "floor"
FLOOR_RUNG = {**_RUNG, "id": "R1", "best_epoch": 3, "primary": 0.4, "delta": [
    {"path": "family", "from": {"name": "mlp-plain", "rationale": "the blueprint's bet"},
     "to": {"name": "linear", "source": "vega:default", "catalog": "families/linear",
            "rationale": "the catalog's linear row is the floor every supervised campaign includes",
            "alternatives": [{"name": "mlp-plain", "when": "the baseline this floor is read against"}]}},
    {"path": "blocks", "from": [{"id": "B1", "type": "dense", "params": {"units": 8}}],
     "to": [{"id": "B1", "type": "linear", "params": {"out": 1}, "rationale": "the floor's map"}]}]}
# a masking rung as the CIDS ladder writes it (R6): an added block names the input it zeroes
MASK_RUNG = {**_RUNG, "id": "R3", "best_epoch": 4, "primary": 0.42, "delta": [
    {"path": "blocks", "from": [{"id": "B1", "type": "dense", "params": {"units": 8}}],
     "to": [{"id": "B0", "type": "custom", "params": {"zero": [2]},
             "rationale": "zeroes floor (column 2) after scaling"},
            {"id": "B1", "type": "dense", "params": {"units": 8}}]}]}


def run(tmp: Path, docs: dict, extra=(), layer_manifest=None) -> tuple[int, dict | None, str]:
    """Write a bundle of `docs` under `tmp` and extract its facts into `tmp/plan/facts.json`."""
    model, bundle = write_bundle(tmp, docs, layer_manifest)
    return extract(model, bundle, tmp / "plan" / "facts.json", extra)


def refused(docs: dict, gate: str, *, layer_manifest=None, after=None, extra=()) -> str:
    """Extract a bundle of `docs` (`after` edits it once written) and assert that `gate` refuses
    it: `ERROR: <gate> …`, exit 2, no fact set written. What it printed."""
    with tempfile.TemporaryDirectory() as tmp:
        model, bundle = write_bundle(Path(tmp), docs, layer_manifest)
        if after:
            after(bundle)
        code, doc, printed = extract(model, bundle, Path(tmp) / "facts.json", extra)
        assert code == 2 and doc is None, f"{gate}: exit {code}, printed {printed!r}"
        assert printed.startswith(f"ERROR: {gate} "), f"{gate} expected, printed {printed!r}"
        assert not (Path(tmp) / "facts.json").exists(), "a refused extraction wrote a fact set"
        return printed


AUX = ("\\relax\n\\newlabel{vega:net.forward}{{3.1}{2}{The forward pass}{equation.3.1}{}}\n"
       "\\newlabel{vega:net.forward@cref}{{[equation][1][3]3.1}{[1][2][]2}{}{}{}}\n"
       "\\newlabel{vega:pack.slice}{{3.2}{4}{The serving slice}{figure.caption.7}{}}\n")


def write_report(docs_dir: Path, digest: str | None = None) -> Path:
    """A compiled report (`main.aux`, `main.pdf`) and its label-role map under `docs_dir`, the
    map's `aux_digest` the `.aux`'s own unless `digest` makes it stale; the map's path."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "main.aux").write_text(AUX, encoding="utf-8")
    (docs_dir / "main.pdf").write_bytes(b"%PDF-1.4\n%fixture\n")
    roles = {"net.forward": {"label": "vega:net.forward", "kind": "equation", "stage": "net"},
             "pack.slice": {"label": "vega:pack.slice", "kind": "figure", "stage": "pack",
                            "number": "3.2", "page": 4}}
    report = {"tex": "docs/main.tex", "pdf": "docs/main.pdf", "aux": "docs/main.aux",
              "aux_digest": digest or digest_file(docs_dir / "main.aux")}
    path = docs_dir / "label-roles.json"
    path.write_text(json.dumps({"schema": "vega/label-roles@1", "lane": "L01", "report": report,
                                "generated_at": "2026-09-30T00:00:00Z", "adapter": "latex-aux",
                                "roles": roles}), encoding="utf-8")
    return path


def rename_input(docs: dict, old: str, new: str) -> dict:
    """`docs` with the input `old` called `new` wherever a record names it."""
    for rows, key in ((docs["netspec.json"]["data_contract"]["features"], "name"),
                      (docs["schema.json"]["input"], "name"),
                      (docs["records/geometry.json"]["typing"], "column")):
        for row in rows:
            row[key] = new if row[key] == old else row[key]
    return docs


def main() -> int:
    failed = 0
    for cell in cells():
        with tempfile.TemporaryDirectory() as tmp:
            bad = check_cell(cell, Path(tmp))
        if bad:
            failed += 1
            print(f"  FAIL  {cell} — {'; '.join(bad)}", file=sys.stderr)
    print(f"{len(cells()) - failed} of {len(cells())} admitted cells give a complete fact set")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
