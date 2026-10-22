"""
The artifact assertions of the lane stages (D6 §6.1–§6.8), run by `python -m lanekit.checks
check-<stage>` in the lane's own process, where pandas and pyarrow may be imported:

- fixture: part rows add up; a full window per series; the graph's edges, level, cross-part
  counts, and (knn) no edge between held-out nodes;
- forge: the smoke pass, the lane's blocks, each out_shape against the shape oracle (T08);
- train: every declared rung ran or was dropped as expected, the floor rung is the shape's floor
  with R0's head (R-A12), finite epochs, a falling R0 loss, a selection, the replay cell;
- eval: one exposure, the metric cells, the trivial baseline, the threshold, the exposure record;
- pack: the round trip over every stored prediction, the output record, the endpoints, no
  ingestion/training/evaluation, graph reference files without raw positions, window history.

The show stage's assertions (D6 §6.8) are lanekit.show_assertions, on the same Checker and Ctx.
A check that cannot run yet (its provider has not landed) says so with Checker.not_run: a WARN
the runner counts and lists in observed.json, never an OK.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import contracts
from .registry import Lane, flatten_types

FLOOR_FAMILY = {"linear": "linear", "linear+flatten": "linear", "linear+graph_pool": "linear",
                "pca-floor": "pca-floor"}
THRESHOLD_TASKS = {"binary", "multilabel", "anomaly"}


class Checker:
    """Collects OK / WARN / ERROR lines for one helper run."""

    def __init__(self) -> None:
        self.failed = 0

    def ok(self, text: str) -> None:
        print(f"OK: {text}", flush=True)

    def not_run(self, what: str, why: str) -> None:
        """A check this run could not make: a WARN the runner counts and lists in the stage's
        `not_run` (observed.json), never an OK."""
        print(f"WARN: check not run: {what} — {why}", flush=True)

    def error(self, text: str) -> None:
        self.failed += 1
        print(f"ERROR: {text}", flush=True)

    def that(self, condition: bool, ok: str, error: str) -> bool:
        (self.ok if condition else self.error)(ok if condition else error)
        return condition


@dataclasses.dataclass(frozen=True)
class Ctx:
    lane: Lane
    lane_dir: Path
    skills: Path
    args: argparse.Namespace

    @property
    def work(self) -> Path:
        return self.lane_dir / "work"

    def fixture(self) -> dict:
        return json.loads((self.lane_dir / "fixture.json").read_text(encoding="utf-8"))

    def campaign(self) -> Path:
        return Path(self.fixture()["run_dir"])


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _import(module: str) -> Any:
    return importlib.import_module(module)


def _parquet(path, columns=None):
    """A parquet file read by Arrow's own filesystem. pandas hands Arrow a Python file object
    for a local path; an Arrow worker still touching Python when the interpreter finalizes
    aborts the helper at exit, its work done ('terminate called without an active
    exception', PLAN-AMENDMENTS A40; vega/scripts/tests/exit_abort_probe.py)."""
    import pandas as pd
    from pyarrow import fs
    return pd.read_parquet(path, columns=columns, filesystem=fs.LocalFileSystem())


# --------------------------------------------------------------------------- fixture (D6 §6.1)
def check_fixture(ctx: Ctx, check: Checker) -> None:
    fx = ctx.fixture()
    manifest = _json(Path(fx["shaped_dir"]) / "manifest.json")
    files = (manifest.get("outputs") or {}).get("files") or {}
    parts = {p: files[p] for p in ("train", "valid", "test") if p in files}
    raw = (manifest.get("input") or {}).get("rows")
    removed = sum(_removed(step) for step in manifest.get("steps") or [])
    total = sum(int(v.get("rows") or 0) for v in parts.values())
    check.that(len(parts) == 3, "three parts written", f"parts written: {sorted(parts)}")
    check.that(total == raw - removed, f"part rows {total} = raw {raw} - removed {removed}",
               f"part rows {total} != raw {raw} - removed {removed}")
    if ctx.lane.raw["shape"] == "window":
        _window_parts(ctx, check, parts)
    if ctx.lane.raw["shape"] == "graph":
        _graph_parts(ctx, check, manifest, parts)


def _removed(step: Mapping[str, Any]) -> int:
    before, after = step.get("rows_before") or {}, step.get("rows_after") or {}
    if set(before) != set(after) or not before:
        return 0
    return sum(int(before[k]) - int(after[k]) for k in before)


def _window_parts(ctx: Ctx, check: Checker, parts: Mapping[str, Mapping]) -> None:
    import pandas as pd
    window = ctx.lane.raw["window"]
    need = window["length"] + (window.get("horizon") or 0)
    series = window.get("series_by")
    for name, meta in parts.items():
        frame = _parquet(meta["path"])
        sizes = frame.groupby(series).size() if series and series in frame else pd.Series([len(frame)])
        check.that(int(sizes.min()) >= need, f"{name}: every series holds >= {need} rows",
                   f"{name}: a series holds {int(sizes.min())} rows, fewer than length + horizon {need}")


def _graph_parts(ctx: Ctx, check: Checker, manifest: Mapping, parts: Mapping) -> None:
    """The edges part (`src_pos, dst_pos, src_part, dst_part, weight`, D1 §15): it exists, the
    manifest's level is the lane's, `cross_part_edges` equals a recount, and a knn graph joins
    no two held-out nodes (knn is fitted on train only)."""
    import pandas as pd
    graph = ctx.lane.raw["graph"]
    edges_meta = (manifest.get("outputs") or {}).get("files", {}).get("edges")
    if not check.that(bool(edges_meta), "the edges part exists", "manifest.outputs.files.edges missing"):
        return
    meta = manifest.get("graph") or {}
    check.that(meta.get("level") == graph["level"], f"graph level {graph['level']}",
               f"manifest graph level {meta.get('level')!r}, the lane declares {graph['level']!r}")
    edges = _parquet(edges_meta["path"], columns=["src_part", "dst_part"])
    recount: dict[str, int] = {}
    for a, b in zip(edges["src_part"], edges["dst_part"]):
        key = _pair(a, b)
        recount[key] = recount.get(key, 0) + 1
    declared = {_pair(*_split(k)): int(v) for k, v in (meta.get("cross_part_edges") or {}).items() if int(v)}
    keys = set(declared) | {k for k in recount if len(set(k.split("-"))) == 2}
    counted = {k: recount[k] for k in keys if recount.get(k)}
    check.that(declared == counted, f"cross_part_edges equals a recount {counted}",
               f"cross_part_edges {declared} != recount {counted}")
    if graph["method"] == "knn":
        bad = {k: v for k, v in recount.items() if k in ("test-valid", "test-test")}
        check.that(not bad, "knn: no valid-test or test-test edge", f"knn edges between held-out nodes: {bad}")


def _pair(a: str, b: str) -> str:
    return "-".join(sorted((str(a), str(b))))


def _split(key: str) -> tuple[str, str]:
    import re
    names = re.findall(r"[a-z]+", key)
    return (names[0], names[-1]) if names else (key, key)


# --------------------------------------------------------------------------- forge (D6 §6.2)
def check_forge(ctx: Ctx, check: Checker) -> None:
    run_dir = ctx.campaign()
    smoke = _json(run_dir / "smoke.json")
    check.that(bool(smoke.get("forward_ok")), "smoke forward pass ok", "smoke.json forward_ok is false")
    spec = _json(run_dir / "netspec.json")
    declared, emitted = sorted(ctx.lane.flat_block_types()), sorted(flatten_types(spec.get("blocks") or []))
    check.that(declared == emitted, "the forged blocks are the lane's",
               f"forged blocks {emitted} != declared {declared}")
    oracles = ctx.skills / contracts.TESTS_DIR / "oracles.py"
    if not oracles.is_file():
        check.not_run("the per-block shape oracle (D6 §6.2)", "oracles.py arrives with T08")
        return
    shape_rule = getattr(_import(contracts.ORACLES_API[0]), contracts.ORACLES_API[1])
    shapes = {b["id"]: b.get("out_shape") for b in smoke.get("per_block") or []}
    running, nominal, context = _shape_inputs(spec, shape_rule)
    for block in spec.get("blocks") or []:
        kind = block["type"]
        # an embedding reads x_cat and leaves the running tensor as it is (its width reaches
        # the concat through the context); every other block reads the running tensor
        primary = nominal if kind == "embedding" else running
        want = tuple(shape_rule(kind, block.get("params") or {}, primary, context))
        got = tuple(shapes.get(block["id"]) or ())
        check.that(got == want, f"{block['id']} {kind} out_shape {got}",
                   f"{block['id']} {kind} out_shape {got}, the shape rule says {want}")
        if kind != "embedding":
            running = got


def _shape_inputs(spec: Mapping[str, Any], shape_rule) -> tuple:
    """(x_num per sample, x_cat per sample, the shape rule's context) of a netspec: the
    numeric columns (and a window's autoregressive past-target channel) over its length, the
    nominal columns, the embedding width a concat adds, the nominal count a tokenizer reads,
    one graph per sample, and the width a decoder reconstructs (recon_dim)."""
    contract = spec.get("data_contract") or {}
    sample = contract.get("sample") or {"shape": "row"}
    features = contract.get("features") or []
    nominal = [f for f in features if f.get("role") == "nominal"]
    n_num = len(features) - len(nominal)
    steps: tuple = ()
    if sample.get("shape") == "window":
        n_num += 1 if sample.get("autoregressive") else 0
        steps = (int(sample["length"]),)
    emb = sum(shape_rule("embedding", b.get("params") or {}, (len(nominal),), {})[-1]
              for b in spec.get("blocks") or [] if b["type"] == "embedding")
    recon = len(features) - len(nominal) + sum(int(f.get("cardinality") or 0) for f in nominal)
    return (steps + (n_num,), steps + (len(nominal),),
            {"emb_width": emb, "n_cat": len(nominal), "num_graphs": 1, "recon_dim": recon})


# --------------------------------------------------------------------------- train (D6 §6.3)
def check_train(ctx: Ctx, check: Checker) -> None:
    camp = ctx.campaign()
    campaign = ctx.lane.raw["campaign"]
    ladder = {r["id"]: r for r in _json(camp / "ladder.json").get("runs", [])}
    exps = _json(camp / "experiments.json")
    ran = {r["id"]: r for r in exps.get("runs", [])}
    for rung in campaign["rungs"]:
        if rung == "R1" and campaign.get("expect_floor") == "dropped":
            check.that(rung not in ran, "R1 (the floor) was dropped, as the lane expects",
                       "R1 ran; the lane expects the floor to be dropped with a WARN")
            continue
        check.that(rung in ran and rung in ladder, f"{rung} ran", f"{rung} did not run")
    _floor(ctx, check, camp, campaign.get("expect_floor"))
    _epochs(check, camp, ran)
    selected = (exps.get("selection") or {}).get("run")
    check.that(bool(selected), f"the selection names {selected}", "the selection names no run")
    if ctx.lane.get("profile") == "full" and selected:
        _notebook_says(check, camp / "train.ipynb", f"VEGA-REPLAY: ok {selected}")


def _floor(ctx: Ctx, check: Checker, camp: Path, expect: str | None) -> None:
    """R1's composition is the shape's floor (R-A12), and its head is R0's."""
    if expect in (None, "dropped"):
        return
    r0, r1 = (_resolved(camp, r) for r in ("R0", "R1"))
    if not (r0 and r1):
        check.error("R0 or R1 has no resolved netspec")
        return
    family = ((r1.get("family") or {}).get("name"))
    types = flatten_types(r1.get("blocks") or [])
    check.that(family == FLOOR_FAMILY[expect], f"R1 is the {family} floor",
               f"R1's family is {family!r}, the lane expects {FLOOR_FAMILY[expect]!r}")
    extra = expect.split("+")[1] if "+" in expect else None
    if extra:
        check.that(extra in types, f"R1 carries {extra}", f"R1's blocks {types} lack {extra}")
    heads = [((s.get("blocks") or [{}])[-1].get("params") or {}).get("out") for s in (r0, r1)]
    check.that(heads[0] == heads[1], f"R1's head out {heads[1]} is R0's",
               f"R1's head out {heads[1]} differs from R0's {heads[0]}")


def _resolved(camp: Path, run: str) -> dict | None:
    path = camp / "runs" / run / "netspec.resolved.json"
    return _json(path) if path.is_file() else None


def _epochs(check: Checker, camp: Path, ran: Mapping[str, Any]) -> None:
    for run in ran:
        history = camp / "runs" / run / "history.jsonl"
        if not history.is_file():
            continue
        losses = [((json.loads(line).get("train") or {}).get("loss")) for line in history.read_text().splitlines() if line.strip()]
        finite = all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses)
        check.that(finite, f"{run}: {len(losses)} finite epoch record(s)", f"{run}: a non-finite train loss")
        if run == "R0" and len(losses) > 1 and finite:
            check.that(losses[-1] < losses[0], f"R0 train loss fell {losses[0]:.4g} -> {losses[-1]:.4g}",
                       f"R0 train loss did not fall ({losses[0]:.4g} -> {losses[-1]:.4g})")


def _notebook_says(check: Checker, notebook: Path, needle: str) -> None:
    nb = _json(notebook)
    text = "\n".join("".join(o.get("text") or []) for c in nb.get("cells", []) for o in c.get("outputs") or [])
    check.that(needle in text, f"{notebook.name} says {needle!r}", f"{notebook.name} does not say {needle!r}")


# --------------------------------------------------------------------------- eval (D6 §6.4)
def check_eval(ctx: Ctx, check: Checker) -> None:
    camp = ctx.campaign()
    seal = _json(camp / "test_seal.json")
    check.that(seal.get("exposures") == 1, "the seal records one exposure", f"exposures {seal.get('exposures')}")
    check.that(bool((seal.get("test") or {}).get("digest_matches_manifest")), "the test digest matches",
               "the sealed test part's digest does not match the manifest")
    run_dir = camp / "eval" / seal["run"]
    metrics = _json(run_dir / "metrics.json")
    _metric_cells(check, metrics)
    models = metrics.get("models") or {}
    check.that("trivial" in models, "the trivial baseline was measured", "no trivial baseline")
    task = ctx.lane.raw["task"]
    if task in THRESHOLD_TASKS or ctx.lane.target.get("base") in THRESHOLD_TASKS:
        check.that((run_dir / "threshold.json").is_file(), "threshold.json written", "threshold.json missing")
    want = ctx.lane.expect.get("exposure")
    if want:
        # the applied layer's data, not the authoring file beside the campaign: the authoring
        # carries only the judgement, and apply splices the measured records into the layer
        layer = evaluation_layer(ctx)
        problem = (exposure_problem(layer, want) if layer is not None else
                   "the evaluation layer's data (layer-evaluation-data) is not in the eval model")
        check.that(problem is None, f"the evaluation records the exposure {want}", str(problem))
    if ctx.lane.get("profile") == "full":
        _notebook_says(check, camp / "evaluate.ipynb", "VEGA-EVAL: sealed")


def evaluation_layer(ctx: Ctx) -> Any:
    """The `layer-evaluation-data` the eval stage applied to `<model>.vega.train.eval.html`, or
    None when that model or its data script is missing."""
    import re
    base = Path(ctx.fixture()["model_html"])
    model = base.with_name(base.name[: -len(".html")] + ".vega.train.eval.html")
    if not model.is_file():
        return None
    match = re.search(r'<script[^>]*id="layer-evaluation-data"[^>]*>([\s\S]*?)</script>',
                      model.read_text(encoding="utf-8"))
    return json.loads(match.group(1)) if match else None


def exposure_records(doc: Any) -> list[Mapping[str, Any]]:
    """Every non-null `exposure` object in a record, wherever it sits (the layer carries the
    spec's `sample.exposure` and the eval's own record, D1 §3 / D6 §6.4)."""
    if isinstance(doc, list):
        return [r for item in doc for r in exposure_records(item)]
    if not isinstance(doc, Mapping):
        return []
    here = [doc["exposure"]] if isinstance(doc.get("exposure"), Mapping) else []
    return here + [r for value in doc.values() for r in exposure_records(value)]


def exposure_problem(layer: Any, want: Mapping[str, Any]) -> str | None:
    """None when the layer records the lane's declared exposure: at least one exposure record,
    and every one of them agreeing with `expect.exposure` on each key the lane declares (kind,
    rows); a record may carry more (its `user:` source)."""
    records = exposure_records(layer)
    if not records:
        return f"the evaluation layer records no exposure; the lane declares {dict(want)}"
    wrong = [r for r in records if any(r.get(k) != v for k, v in want.items())]
    if wrong:
        return f"the evaluation layer records the exposure {wrong[0]}; the lane declares {dict(want)}"
    return None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric_cells(check: Checker, metrics: Mapping[str, Any]) -> None:
    """Each network metric is finite, null with a recorded reason (vega-eval's Estimate writes it
    as `note`; `reason` is read too), or a per-element vector (a multilabel pr_auc per label, a
    reconstruction's per-feature error: `value` null, every one of `values` finite)."""
    network = (metrics.get("models") or {}).get("network") or {}
    primary = metrics.get("primary")
    check.that(primary in network, f"primary {primary} measured", f"primary {primary} missing")
    for name, cell in network.items():
        value = (cell or {}).get("value")
        values = (cell or {}).get("values")
        per_element = value is None and isinstance(values, list) and bool(values) and all(map(_finite, values))
        fine = _finite(value) or (value is None and (cell.get("reason") or cell.get("note"))) or per_element
        shown = f"{name} per element {values}" if per_element else f"{name} {value}"
        check.that(bool(fine), shown, f"{name} is neither finite, nor null with a reason, nor a vector "
                                      "of finite elements")


# --------------------------------------------------------------------------- pack (D6 §6.5)
def check_pack(ctx: Ctx, check: Checker) -> None:
    bundle = sorted(ctx.work.glob("bundle/*/MANIFEST.json"))[0].parent
    manifest = _json(bundle / "MANIFEST.json")
    trip = manifest.get(contracts.ROUNDTRIP_KEY) or {}
    check.that(trip.get("status") == "pass", "the round trip passed", f"round trip {trip.get('status')}")
    check.that(trip.get("shaper_transform_check") == "pass", "the shaper's twin check passed",
               "the shaper's twin check did not pass")
    stored = _stored_rows(ctx, check)
    check.that(stored is None or trip.get("rows") == stored, f"the round trip covered {trip.get('rows')} rows",
               f"the round trip covered {trip.get('rows')} of {stored} stored predictions")
    for problem in contracts.output_record_problems(manifest):
        check.error(f"MANIFEST.output: {problem}")
    endpoints = set(((manifest.get("serve") or {}).get("endpoints")) or [])
    for route in contracts.REQUIRED_ENDPOINTS:
        check.that(route in endpoints, f"{route} served", f"MANIFEST.serve.endpoints lacks {route}")
    package = bundle / manifest["package"]
    for forbidden in contracts.BUNDLE_FORBIDDEN_DIRS:
        check.that(not (package / forbidden).exists(), f"{forbidden}/ absent", f"{forbidden}/ shipped")
    _shape_records(ctx, check, bundle, manifest)
    if ctx.lane.get("profile") == "full":
        _notebook_says(check, bundle / "predict.ipynb", "VEGA-PACK: reproduced")


def _stored_rows(ctx: Ctx, check: Checker) -> int | None:
    """The stored test predictions — the scored rows only, len(test) - |unscored| (R-A5): what
    the round trip owes."""
    import pandas as pd
    camp = ctx.campaign()
    run = _json(camp / "test_seal.json")["run"]
    stored = camp / "eval" / run / "predictions.test.parquet"
    if not stored.is_file():
        check.error(f"no stored predictions at {stored}")
        return None
    return len(_parquet(stored))


def _shape_records(ctx: Ctx, check: Checker, bundle: Path, manifest: Mapping[str, Any]) -> None:
    shape = ctx.lane.raw["shape"]
    if shape == "graph":
        nodes = contracts.GRAPH_REFERENCE_FILES["nodes"]
        check.that(nodes in (manifest.get("files") or {}), f"{nodes} pinned", f"{nodes} not in MANIFEST.files")
        import pandas as pd
        path = bundle / nodes
        cols = set(_parquet(path).columns) if path.is_file() else set()
        leak = cols & set(contracts.GRAPH_REFERENCE_FORBIDDEN_COLUMNS)
        check.that(not leak, "the reference nodes carry no raw position", f"raw positions shipped: {sorted(leak)}")
    if shape == "window":
        examples = [p for p in (bundle / "examples").glob("*.json")]
        with_history = [p.name for p in examples if _json(p).get("history") or (_json(p).get("test") or {}).get("history")]
        check.that(bool(with_history), f"examples carry history ({', '.join(with_history)})",
                   "no example carries the history its rows need")
