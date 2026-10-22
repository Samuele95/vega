#!/usr/bin/env python3
"""
test_predictions_layout.py — `vega/predictions-layout@1` (BUILD-PLAN R-C2) as `evaluate.py`
writes it, on the three shapes of B15's vertical-slice spike: a window forecast, a node-level
graph, a graph-level set.

    python3 -m pytest test_predictions_layout.py [-q]
    python3 test_predictions_layout.py [-k NAME] [-v]            (without pytest)

Each fixture of `vega-forge/scripts/tests/spike/make_spike_fixtures.py` (raw rows through the
real `shape.py`, a netspec the compiler admits, the package the v2.3 emitter wrote) becomes a
campaign as /vega-train leaves one — the package beside `campaign.json`, run `R0` fitted for the
spec's epochs by the package's own `Trainer` with its `ModelRegistry` writing `best.pt`, a ledger
that selects it — and the real `evaluate.py` looks at it once, under the seal. Then the stored
predictions are checked against an independent reading of the same sealed part through the
package's `Evaluator` (`scored`, `heads`):

  * the columns: `row_id`, `graph_id` at graph level, `y` or `y_0 …` keeping the truth's
    dimension, the raw head as `score` or `score_0 …` (every `head_out` column, equal to
    `Evaluator.heads`), `prediction` exactly where a threshold decides (σ(logit) ≥ threshold),
    each fitted baseline's `score_<name>[_k]`, no NaN anywhere;
  * the metadata: `vega.row_id_kind`, `vega.truth_shape`, `vega.score_shape`,
    `vega.task_effective`, and `metrics.json.predictions_layout`, `ci_level == 0.95`;
  * the row ids per shape: a window's are the anchors' raw positions, and together with
    `unscored.json`'s rows they are exactly the test part's raw positions (dense scoring: every
    test row is answered or said unscored, never both); a node's are the test nodes' raw
    positions; a graph's are the test graphs' ids, with `graph_id` repeating them;
  * what the shape adds: the forecast's `naive` baseline and its season (mase's train-side
    scale in `metrics.json.options`), the per-horizon error with an interval per step, no
    threshold for a value reading; the graph exposure
    record in `metrics.json.graph` and in the seal (under the transductive protocol, the
    edges into the test part, the features seen and the spec's declared disclosure);
    `graph-linear` at node level over the
    neighbour mean; the baselines' pooled view at graph level;
  * what cannot be read is refused before the seal (exit 2, no seal): an unknown baseline, a
    threshold the reading cannot use, a package of another generation or without an Evaluator.

`VEGA_LAYOUT_WORK=<dir>` keeps the fixtures and campaigns there (default: a temporary directory,
removed at exit). Needs torch, pandas, pyarrow, scikit-learn. Exit 0 when every check passes.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
FORGE_SCRIPTS = SKILLS / "vega-forge" / "scripts"
SPIKE = FORGE_SCRIPTS / "tests" / "spike"
for _p in (str(SCRIPTS), str(SPIKE), str(FORGE_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_spike_fixtures as msf  # noqa: E402
import predictions_layout as layout  # noqa: E402
from vega_core import PackageAdapters  # noqa: E402

VERBOSE = "-v" in sys.argv
N_BOOT = 30
_WORK: list = []
_CAMPAIGNS: dict = {}


def work_dir() -> Path:
    if not _WORK:
        named = os.environ.get("VEGA_LAYOUT_WORK")
        if named:
            Path(named).mkdir(parents=True, exist_ok=True)
            _WORK.append(Path(named))
        else:
            made = Path(tempfile.mkdtemp(prefix="vega-layout-"))
            atexit.register(shutil.rmtree, made, True)
            _WORK.append(made)
    return _WORK[0]


def _sha256(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, doc):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _numpy(values):
    if values is None:
        return None
    detach = getattr(values, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(values)


@dataclass
class Campaign:
    """One fixture as a trained campaign, evaluated once by evaluate.py."""

    fixture: msf.Fixture
    camp: Path
    adapters: object
    spec: object
    spec_dict: dict
    proc: subprocess.CompletedProcess
    split: dict

    @property
    def eval_dir(self) -> Path:
        return self.camp / "eval" / "R0"

    def record(self, name):
        return json.loads((self.eval_dir / name).read_text(encoding="utf-8"))

    def stored(self):
        return layout.read(self.eval_dir / "predictions.test.parquet")

    def independent(self):
        """The sealed part scored again here, through the package Evaluator: the truth, the
        row ids, the raw head the stored file must hold."""
        seal = json.loads((self.camp / "test_seal.json").read_text(encoding="utf-8"))
        data, evaluator = self.adapters.data, self.adapters.evaluator
        part = data.load_sealed_part(self.spec, seal["token"], str(self.camp / "test_seal.json"))
        model = self.adapters.store.load(self.camp / "runs" / "R0" / "checkpoints" / "best.pt",
                                         spec=self.spec, build=self.adapters.build)
        return part, evaluator.scored(self.spec, part, model), \
            _numpy(evaluator.heads(self.spec, part, model))


def _transductive(spec_dict: dict) -> dict:
    """The node-level spec under the transductive protocol, with the disclosure W2 requires
    (`sample.exposure` from a `user:` source)."""
    sample = spec_dict["data_contract"]["sample"]
    sample["protocol"] = "transductive"
    sample["protocol_source"] = "user:layout-test/transductive"
    sample.pop("protocol_catalog", None)
    sample["exposure"] = {"kind": "features", "rows": "test",
                          "source": "user:layout-test/transductive"}
    return spec_dict


#: campaign name -> (spike fixture, what the campaign changes in its spec)
_VARIANTS = {"node-transductive": ("node-keypair", _transductive)}


def campaign_of(name: str) -> Campaign:
    """Fixture `name` made (or a variant of one), trained as run R0 and evaluated (once per
    process)."""
    if name in _CAMPAIGNS:
        return _CAMPAIGNS[name]
    base, mutate = _VARIANTS.get(name, (name, None))
    fixture = (_CAMPAIGNS[base].fixture if base in _CAMPAIGNS
               else msf.make(base, work_dir() / "fixtures", "full"))
    camp = Path(fixture.root) / ("campaign" if mutate is None else f"campaign-{name}")
    if camp.exists():
        shutil.rmtree(camp)
    camp.mkdir(parents=True)
    shutil.copytree(fixture.package_dir, camp / fixture.package)
    spec_dict = json.loads(Path(fixture.spec).read_text(encoding="utf-8"))
    if mutate is not None:
        spec_dict = mutate(spec_dict)
    _write(camp / "netspec.json", spec_dict)
    _write(camp / "campaign.json", {"schema": "vega/campaign@1", "package": fixture.package,
                                    "note": "test_predictions_layout: a campaign of one run"})
    adapters = PackageAdapters.load(camp / fixture.package, fixture.package)
    dom = adapters.package.domain
    spec = dom.NetSpec.from_dict(spec_dict)
    run_dir = camp / "runs" / "R0"
    model = adapters.build(spec)
    fit = adapters.trainer.fit(model, spec, adapters.data, None, run=dom.RunContext(
        checkpoint_dir=str(run_dir / "checkpoints")))
    best = run_dir / "checkpoints" / "best.pt"
    assert best.is_file(), f"{name}: the Trainer wrote no best.pt ({fit})"
    _write(run_dir / "netspec.resolved.json", spec_dict)
    _write(camp / "experiments.json", {
        "schema": "vega/experiments@1",
        "runs": [{"id": "R0", "kind": "baseline", "checkpoint_digest": _sha256(best),
                  "primary": fit.best_value}],
        "selection": {"run": "R0", "rule": "the only run"}})
    proc = subprocess.run([sys.executable, str(SCRIPTS / "evaluate.py"), "--campaign",
                           str(camp), "--n-boot", str(N_BOOT), "--unattended",
                           "--vega-forge-dir", str(FORGE_SCRIPTS.parent)],
                          capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    split = json.loads(Path(fixture.fitted).read_text(encoding="utf-8"))["__split_raw__"]
    _CAMPAIGNS[name] = Campaign(fixture, camp, adapters, spec, spec_dict, proc,
                                {k: np.asarray(v, dtype=np.int64) for k, v in split.items()})
    return _CAMPAIGNS[name]


# ================================================================ checks every shape shares

def _columns(frame, prefix):
    found = [c for c in frame.columns
             if c.startswith(prefix + "_") and c[len(prefix) + 1:].isdigit()]
    return sorted(found, key=lambda c: int(c.rsplit("_", 1)[1]))


def _block(frame, prefix):
    cols = _columns(frame, prefix)
    return frame[cols].to_numpy() if cols else (frame[prefix].to_numpy()
                                                 if prefix in frame.columns else None)


def _shape_of(values) -> str:
    if values is None:
        return "null"
    values = np.asarray(values)
    return "[]" if values.ndim == 1 else json.dumps(list(values.shape[1:]))


def shared_checks(c: Campaign, effective: str, row_id_kind: str) -> tuple:
    assert c.proc.returncode == 0, (c.proc.returncode, c.proc.stdout[-3000:],
                                    c.proc.stderr[-3000:])
    frame, meta = c.stored()
    part, scored, head = c.independent()
    metrics = c.record("metrics.json")

    assert metrics["predictions_layout"] == layout.LAYOUT, metrics.get("predictions_layout")
    assert metrics["ci_level"] == 0.95, metrics.get("ci_level")
    assert metrics["task_effective"] == effective
    assert metrics["n_test"] == len(frame) == len(scored), (metrics["n_test"], len(frame))
    assert set(layout.METADATA_KEYS) <= set(meta), meta
    assert meta["vega.row_id_kind"] == row_id_kind, meta
    assert meta["vega.task_effective"] == effective, meta
    assert meta["vega.predictions_layout"] == layout.LAYOUT, meta

    # row ids: the Evaluator's, in its order, never re-mapped
    assert np.array_equal(frame["row_id"].to_numpy(), np.asarray(scored.row_ids)), "row ids"
    # the truth keeps its dimension
    truth = _numpy(scored.truth)
    stored_truth = _block(frame, "y")
    assert meta["vega.truth_shape"] == _shape_of(truth), (meta["vega.truth_shape"], truth.shape)
    assert np.array_equal(np.asarray(stored_truth, dtype=np.float64).reshape(truth.shape),
                          truth.astype(np.float64)), "the stored truth is not the Evaluator's"
    # the raw head, every head_out column
    head = head.reshape(len(head), -1)
    stored_head = _block(frame, "score").reshape(len(frame), -1)
    assert stored_head.shape == head.shape, (stored_head.shape, head.shape)
    assert meta["vega.score_shape"] == ("[]" if head.shape[1] == 1 else f"[{head.shape[1]}]")
    tol = 1e-6 * max(1.0, float(np.abs(head).max()))
    assert np.abs(stored_head - head).max() <= tol, "the stored head is not Evaluator.heads"
    # no fabricated prediction: nothing missing in a stored column
    assert not frame.drop(columns=["row_id", "graph_id"], errors="ignore").isna().any().any()
    # the unscored samples, by reason, as the part holds them
    unscored = c.record("unscored.json")
    want = layout.unscored_record(part.unscored)
    assert unscored == json.loads(json.dumps(want)), (unscored["count"], want["count"])
    assert metrics["n_unscored"] == unscored["count"]
    # every fitted baseline has its score column(s), and so does the campaign's R0 when it
    # was predicted through the seal (a compared row, not a fitted one); any other has none
    baselines = c.record("baselines.json")["baselines"]
    for b in baselines:
        cols = [k for k in frame.columns
                if k == f"score_{b['name']}" or k.startswith(f"score_{b['name']}_")]
        stored = bool(b.get("fitted") or b.get("predicted"))
        assert bool(cols) == stored, (b["name"], b.get("fitted"), b.get("predicted"), cols)
    cb = _by_name(baselines).get("campaign-baseline")
    if cb is not None:
        # R0 is predicted through the seal: a test number with its CI and a paired difference
        assert cb["predicted"] and cb["test"]["ci"], cb
        assert "campaign-baseline" in metrics["differences"], sorted(metrics["differences"])
    return frame, meta, metrics, part, scored, baselines


def _by_name(baselines) -> dict:
    return {b["name"]: b for b in baselines}


# ================================================================ per shape

def test_window_forecast_layout():
    c = campaign_of("window")
    frame, meta, metrics, part, scored, baselines = shared_checks(c, "forecast", "raw_position")
    horizon = int(c.fixture.sample["horizon"])
    assert meta["vega.truth_shape"] == f"[{horizon}]", meta
    assert _columns(frame, "y") == [f"y_{h}" for h in range(horizon)], list(frame.columns)
    assert "prediction" not in frame.columns and "row_score" not in frame.columns
    # dense scoring: the anchors and the unscored rows are exactly the test part, disjoint
    anchors = set(frame["row_id"].astype(int))
    unscored = {int(r["row_id"]) for r in c.record("unscored.json")["rows"]}
    test_rows = set(int(r) for r in c.split["test"])
    assert anchors | unscored == test_rows, (len(anchors), len(unscored), len(test_rows))
    assert not anchors & unscored
    assert c.record("unscored.json")["reasons"]["horizon_beyond_part"] > 0
    # mase reads the train target's seasonal scale, at the spec's season
    season = int(c.fixture.sample["season"])
    assert metrics["options"]["season"] == season, metrics["options"]
    scale = c.adapters.data.target_scale(c.spec, season)
    assert scale and abs(metrics["options"]["mase_scale"] - scale) < 1e-12, metrics["options"]
    assert metrics["models"]["network"]["mase"]["value"] is not None
    # a per-unit reading (one error per horizon step) carries an interval per unit, drawn on
    # the shared resamples — what a breakdown reads to call a unit weak
    per_step = metrics["models"]["network"]["per_horizon_error"]
    assert len(per_step["values"]) == len(per_step["cis"]) == horizon, per_step
    for value, (lo, hi) in zip(per_step["values"], per_step["cis"]):
        assert lo <= round(value, 4) <= hi, per_step
    # a value reading decides nothing
    threshold = c.record("threshold.json")
    assert threshold["applies"] is False and threshold["value"] is None, threshold
    # the persistence floor is fitted and stored per horizon step; the seasonal one says why not
    named = _by_name(baselines)
    assert named["naive"]["fitted"], named.get("naive")
    assert _columns(frame, "score_naive") == [f"score_naive_{h}" for h in range(horizon)]
    assert not named["seasonal-naive"]["fitted"] and "season" in named["seasonal-naive"]["why"]
    assert named["linear"]["view"] == "flat"
    robust = c.record("robustness.json")["checks"]
    assert any(r.get("check") == "importance-perturbation" and r.get("status") == "skipped"
               and "windowed" in r.get("why", "") for r in robust), robust


def test_node_level_layout():
    c = campaign_of("node-keypair")
    frame, meta, metrics, part, scored, baselines = shared_checks(c, "binary", "raw_position")
    # every test node is a target, by its raw position
    assert set(frame["row_id"].astype(int)) == set(int(r) for r in c.split["test"])
    assert "graph_id" not in frame.columns
    assert meta["vega.truth_shape"] == "[]" and meta["vega.score_shape"] == "[]"
    # the decision: σ(logit) ≥ the probability-scale threshold chosen on valid
    threshold = c.record("threshold.json")
    assert threshold["applies"] and threshold["scale"] == "probability", threshold
    sig = 1.0 / (1.0 + np.exp(-frame["score"].to_numpy(dtype=np.float64)))
    assert np.array_equal(frame["prediction"].to_numpy(),
                          (sig >= threshold["value"]).astype(np.int64))
    # the exposure record, in metrics.json and in the seal
    record = {"protocol": "inductive", "edges_into_test": 0, "features_seen_in_training": False}
    assert metrics["graph"] == record, metrics.get("graph")
    seal = json.loads((c.camp / "test_seal.json").read_text(encoding="utf-8"))
    assert seal["graph"] == record, seal.get("graph")
    # the one-propagation contender reads the neighbour mean; linear is structure-blind
    named = _by_name(baselines)
    # each fitted baseline is read at its own threshold, chosen on its valid scores by the
    # network's rule (eval-contract §2) — its f1 is the f1 of its own decision, not the network's
    truth = frame["y"].to_numpy(dtype=np.int64)
    for b in (b for b in baselines if b.get("fitted")):
        own = b.get("decision_threshold") or {}
        assert own.get("scale") == "probability" and own.get("source") == threshold["source"], b
        s_b = frame[f"score_{b['name']}"].to_numpy(dtype=np.float64)
        decided = (1.0 / (1.0 + np.exp(-s_b)) >= own["at"]).astype(np.int64)
        tp = int(((decided == 1) & (truth == 1)).sum())
        f1 = 2 * tp / max(1, int(decided.sum()) + int(truth.sum()))
        assert abs(metrics["models"][b["name"]]["f1"]["value"] - f1) <= 5e-5, (b["name"], own, f1)  # at 4 dp
    assert named["graph-linear"]["fitted"] and named["graph-linear"]["view"] == "neighbour-mean"
    assert "score_graph-linear" in frame.columns
    assert "structure-blind" in named["linear"]["definition"], named["linear"]
    robust = c.record("robustness.json")["checks"]
    assert any(r.get("status") == "skipped" and "graph" in r.get("why", "") for r in robust)


def test_transductive_exposure_is_recorded():
    """Under the transductive protocol the training graph held the test nodes' features: the
    eval record says so — edges into the test part counted, features seen — and carries the
    spec's declared disclosure, in metrics.json and (the measured half) in the seal."""
    c = campaign_of("node-transductive")
    frame, meta, metrics, part, scored, baselines = shared_checks(c, "binary", "raw_position")
    graph = metrics["graph"]
    assert graph["protocol"] == "transductive", graph
    assert graph["features_seen_in_training"] is True, graph
    assert graph["edges_into_test"] > 0, graph
    assert graph["exposure"] == {"kind": "features", "rows": "test",
                                 "source": "user:layout-test/transductive"}, graph
    seal = json.loads((c.camp / "test_seal.json").read_text(encoding="utf-8"))
    assert seal["graph"] == {k: graph[k] for k in ("protocol", "edges_into_test",
                                                   "features_seen_in_training")}, seal["graph"]
    assert set(frame["row_id"].astype(int)) == set(int(r) for r in c.split["test"])


def test_graph_level_layout():
    c = campaign_of("graph-level")
    frame, meta, metrics, part, scored, baselines = shared_checks(c, "binary", "graph_id")
    assert "graph_id" in frame.columns
    assert np.array_equal(frame["graph_id"].to_numpy(), frame["row_id"].to_numpy())
    raw = pd.read_csv(c.fixture.raw, low_memory=False)
    graph_col = c.fixture.sample["graph_id"]
    test_graphs = {str(v) for v in raw.iloc[c.split["test"]][graph_col].unique()}
    assert {str(v) for v in frame["row_id"]} == test_graphs, (len(frame), len(test_graphs))
    assert frame["row_id"].is_unique
    assert metrics["graph"]["protocol"] == "inductive", metrics.get("graph")
    named = _by_name(baselines)
    for name in ("trivial", "linear", "tree"):
        assert named[name]["view"] == "graph-pooled", named[name]
    assert "graph-linear" not in named, "graph-linear is not a default at graph level"
    assert "pooled" in named["linear"]["definition"], named["linear"]


def _unsealed_copy(c: Campaign, where: str) -> Path:
    """A copy of an evaluated campaign as it was before its look: no seal, no eval records."""
    dst = Path(c.fixture.root) / where
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(c.camp, dst)
    (dst / "test_seal.json").unlink()
    shutil.rmtree(dst / "eval")
    doc = json.loads((dst / "campaign.json").read_text(encoding="utf-8"))
    doc.pop("test_seal", None)
    _write(dst / "campaign.json", doc)
    return dst


def _evaluate(camp: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / "evaluate.py"), "--campaign", str(camp),
                           "--n-boot", str(N_BOOT), "--unattended", *extra],
                          capture_output=True, text=True, timeout=900,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_what_cannot_be_read_is_refused_before_the_seal():
    """An unknown baseline, a threshold the reading cannot use, a package of another generation
    or without an Evaluator: each refused with exit 2 and no seal written — none of them may
    cost a look at the test part."""
    c = campaign_of("node-keypair")
    cases = []
    camp = _unsealed_copy(c, "refused-baseline")
    cases.append(("an unknown baseline", camp, ("--baselines", "trivial,linear,oracle"),
                  "oracle"))
    camp = _unsealed_copy(c, "refused-threshold")
    cases.append(("a probability threshold above 1", camp, ("--threshold", "1.5"), "[0, 1]"))
    camp = _unsealed_copy(c, "refused-generation")
    bmap = camp / c.fixture.package / "blueprint-map.json"
    doc = json.loads(bmap.read_text(encoding="utf-8"))
    doc["generation"] = "v2.2"
    _write(bmap, doc)
    cases.append(("a v2.2 package", camp, (), "v2.3"))
    camp = _unsealed_copy(c, "refused-evaluator")
    bmap = camp / c.fixture.package / "blueprint-map.json"
    doc = json.loads(bmap.read_text(encoding="utf-8"))
    doc["entries"] = [e for e in doc["entries"] if e.get("role") != "Evaluator"]
    _write(bmap, doc)
    cases.append(("a package without an Evaluator port", camp, (), "Evaluator"))
    for what, camp, extra, needle in cases:
        proc = _evaluate(camp, *extra)
        assert proc.returncode == 2, (what, proc.returncode, proc.stdout[-2000:])
        assert needle in proc.stdout and "before the seal" in proc.stdout, (what, proc.stdout)
        assert not (camp / "test_seal.json").exists(), f"{what}: a seal was written"
    # and the control: the untouched copy evaluates
    camp = _unsealed_copy(c, "refused-control")
    proc = _evaluate(camp)
    assert proc.returncode == 0 and (camp / "test_seal.json").is_file(), proc.stdout[-2000:]


def test_the_layout_writer_refuses_misaligned_columns():
    """The writer's own guards, without a campaign."""
    ok = layout.PredictionsLayout(row_ids=np.arange(3), row_id_kind="raw_position",
                                  truth=np.zeros((3, 2)), head=np.ones((3, 1)),
                                  task_effective="multilabel",
                                  baselines=(("trivial", np.zeros((3, 2))),))
    cols = list(ok.columns())
    assert cols == ["row_id", "y_0", "y_1", "score", "score_trivial_0", "score_trivial_1"], cols
    assert ok.metadata()["vega.truth_shape"] == "[2]" and ok.metadata()["vega.score_shape"] == "[]"
    for bad in (dict(truth=np.zeros(2)), dict(row_id_kind="position"),
                dict(baselines=(("0", np.zeros(3)),)), dict(head=None)):
        args = dict(row_ids=np.arange(3), row_id_kind="raw_position", truth=None,
                    head=np.ones((3, 1)), task_effective="binary")
        args.update(bad)
        try:
            layout.PredictionsLayout(**args)
        except ValueError:
            continue
        raise AssertionError(f"the writer accepted {bad}")
    # a one-step forecast: the Evaluator's (N, 1) head is the one `score` column, the (N,)
    # truth `y`, and a baseline's (N, 1) score keeps its column index
    h1 = layout.PredictionsLayout(row_ids=np.arange(4), row_id_kind="raw_position",
                                  truth=np.zeros(4), head=np.ones((4, 1)),
                                  task_effective="forecast",
                                  baselines=(("naive", np.zeros((4, 1))),))
    assert list(h1.columns()) == ["row_id", "y", "score", "score_naive_0"], list(h1.columns())
    assert h1.metadata()["vega.truth_shape"] == "[]" and h1.metadata()["vega.score_shape"] == "[]"
    graph = layout.PredictionsLayout(row_ids=np.array(["M1", "M2"]), row_id_kind="graph_id",
                                     truth=np.array([0, 1]), head=np.ones((2, 1)),
                                     task_effective="binary")
    assert list(graph.columns())[:2] == ["row_id", "graph_id"]
    record = layout.unscored_record([{"raw_pos": 4, "reason": "insufficient_context"}])
    assert record["count"] == 1 and record["reasons"] == {"insufficient_context": 1,
                                                          "horizon_beyond_part": 0}


def main(argv) -> int:
    pick = argv[argv.index("-k") + 1] if "-k" in argv else ""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f) and pick in n]
    failed = []
    for name, fn in tests:
        start = time.monotonic()
        try:
            fn()
            print(f"  ok    {name}  ({time.monotonic() - start:.1f} s)")
        except Exception as exc:                        # a failing test is reported, not raised
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:1500]}")
    if VERBOSE:
        for name, c in _CAMPAIGNS.items():
            print(f"\n--- evaluate.py on {name}\n" + "\n".join(
                line for line in c.proc.stdout.splitlines()
                if line.startswith(("OK:", "WARN:", "ERROR:"))))
    print(f"\n{len(tests) - len(failed)} of {len(tests)} layout test(s) passed "
          f"(fixtures under {work_dir()})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
