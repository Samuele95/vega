#!/usr/bin/env python3
"""
test_trainer_runs.py — `trainer.py` drives an emitted v2.3 package through its adapter classes,
on every sample shape, and stores the valid part in the test part's layout.

    python3 -m pytest test_trainer_runs.py [-q]

The campaigns are B15's spike fixtures (a `gru` forecast over windows, a `gcn` at node level, a
`gin` at graph level) laid out as `/vega-forge` leaves a run directory — the spec, the emitted
package, a model file — with `campaign.json` from `plan_ladder.make_campaign`. Each run goes
through `trainer.run_one`, the path `trainer.py --run` and `hpo.py` take. What it proves:

1. **The adapters** (`vega_core.PackageAdapters`): the run trains through
   `Trainer(store).fit(model, spec, data, callbacks, run=RunContext(...))`, writes one
   `EpochRecord.as_dict()` line per epoch, keeps `netspec.resolved.json`, saves `best.pt` and
   `last.pt` through the `ModelRegistry`, and records plain JSON values (`stopped_by` is the
   string, not the domain's enum).
2. **The valid predictions are `vega/predictions-layout@1`** (R-C2), from `best.pt` through
   the Evaluator's `ScoredPart` (E16): on windows the truth keeps its horizon (`y_0…y_2`), the
   raw head every column (`score_0…score_2`), the row id is the anchor's raw position — the
   part's own row ids, untruncated — and the rows a forecast cannot score are counted by
   reason; at node level the row ids are the valid nodes' raw positions; at graph level they
   are the graph ids, with a `graph_id` column; the parquet metadata names the row-id kind,
   the truth and score shapes and the effective task; the stored scores are best.pt's heads.
3. **An architecture rung re-emits** into `runs/<id>/<package>/` (against the campaign root)
   and trains that package: R1b, the recurrent floor, on the window campaign.
4. **The replay** re-runs the resolved spec unchanged and stops itself: it matches the
   records (exit 0) and writes no checkpoint; a record that does not match is exit 4.
5. **The floor's ledger row is found by its source**, so D4 is evaluated once the floor ran;
   a package whose map cannot be driven is refused before a run starts (exit 2).
6. **`--resume` never writes into another run.** Resuming a finished R1 in a campaign that
   holds R1b (the recurrent floor) opens R1c with `supersedes: R1`, warm-started from R1's
   own last.pt, and leaves every file of runs/R1b as it was; the superseding id skips every
   ladder id, ledger row, run directory and rule-owned id.
7. **A trial moves a param the spec leaves at its default**: `num_layers` absent from the gru
   is read at the forge catalog's default, so the trial's delta passes the checks, re-emits,
   trains, and its resolved spec states `num_layers: 2`.
8. **The device** is the Trainer module's `optim.device_for(spec)`, as D1 §12.2 records it.
9. **No run without a rung, no dead delta, no unchecked expect** (A46 T): a `--delta` run
   whose id is no ladder rung (nor a trial of a planned study) is refused, as is a `--delta`
   that is not the rung's own, a study id run as a run and a trial without its point; a delta
   on a declaration no emitted code reads is refused as a dead delta; a rung whose expect
   cannot be checked is refused; all before a file is written. `hpo.py` refuses a study the
   ladder does not plan.
10. **The ledger row records expect_held**: R0's and R1b's rows carry their rule's check, its
   text, the verdict and the numbers compared, and the apply script's re-judging agrees.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
SPIKE = SKILLS / "vega-forge" / "scripts" / "tests" / "spike"
FORGE = SKILLS / "vega-forge" / "scripts"
for _p in (str(SCRIPTS), str(SPIKE), str(FORGE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plan_ladder as PL  # noqa: E402
import trainer as T  # noqa: E402
from vega_train_core import Delta, Rung, load_catalog  # noqa: E402

CATALOG = load_catalog()[0]

EPOCH_KEYS = {"epoch", "time_s", "lr", "train", "valid", "grad_norm", "weight_norm",
              "update_ratio", "nan", "early_stopping", "shuffle"}
STUB_MODEL = "<!doctype html><html><head><title>stub</title></head><body></body></html>"


# --------------------------------------------------------------------------- the campaigns

def _campaign(fixture, root: Path, rungs) -> T.Campaign:
    """A run directory as /vega-forge leaves it (the spec, the emitted package, a model file
    with no layer: every source the spike cites resolves without one), with campaign.json and
    the given rungs as its ladder."""
    root.mkdir(parents=True)
    spec = json.loads(Path(fixture.spec).read_text(encoding="utf-8"))
    (root / "netspec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    shutil.copytree(fixture.package_dir, root / fixture.package)
    (root / "model.html").write_text(STUB_MODEL, encoding="utf-8")
    campaign = PL.make_campaign(root, spec, root / "model.html", 3.0, "cpu", True)
    (root / "campaign.json").write_text(json.dumps(campaign, indent=2), encoding="utf-8")
    ladder = {"schema": "vega/ladder@1", "runs": [PL.rung_baseline(spec, None), *rungs(spec)]}
    (root / "ladder.json").write_text(json.dumps(ladder, indent=2), encoding="utf-8")
    camp, err = T.Campaign.read(root)
    assert err is None, err
    return camp


@pytest.fixture(scope="module")
def camps(tmp_path_factory):
    import make_spike_fixtures as msf
    work = tmp_path_factory.mktemp("trainer")
    made = {name: msf.make(name, work / "spike", "full")
            for name in ("window", "node-keypair", "graph-level")}
    forge = PL.forge_catalog()
    return {
        "window": (_campaign(made["window"], work / "window",
                             lambda s: [PL.rung_recurrent_floor(s, forge)[0]]), made["window"]),
        "node": (_campaign(made["node-keypair"], work / "node", lambda s: []),
                 made["node-keypair"]),
        "graph": (_campaign(made["graph-level"], work / "graph", lambda s: []),
                  made["graph-level"]),
    }


@pytest.fixture(scope="module")
def ran(camps):
    """R0 on every campaign, R1b on the window one — each through `run_one` once."""
    out = {}
    for key, (camp, _) in camps.items():
        for run_id in (("R0", "R1b") if key == "window" else ("R0",)):
            said = []
            record, row, code = T.run_one(camp, run_id, out=said.append)
            assert code == 0, "\n".join(said)
            out[(key, run_id)] = (record, row, said)
    return out


def _stored(camp, run_id):
    import pyarrow.parquet as pq
    table = pq.read_table(camp.run_dir(run_id) / "predictions.valid.parquet")
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()
            if k.startswith(b"vega.")}
    return table.to_pandas(), meta


def _valid_part(camp, run_id="R0"):
    adapters = T.open_package(camp.package_dir_for(run_id), camp.package)
    spec = json.loads((camp.run_dir(run_id) / "netspec.resolved.json").read_text())
    ns = adapters.package.domain.NetSpec.from_dict(spec)
    return adapters, ns, adapters.data.load_part(ns, "valid")


# --------------------------------------------------------------------------- 1. the adapters

def test_a_run_trains_through_the_adapter_classes(camps, ran):
    camp, _ = camps["window"]
    record, row, _ = ran[("window", "R0")]
    run_dir = camp.run_dir("R0")
    assert record["status"] == "done" and record["stopped_by"] == "max_epochs"
    assert type(record["stopped_by"]) is str and record["epochs"] == 3
    on_disk = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert on_disk["stopped_by"] == "max_epochs"
    adapters, ns, _ = _valid_part(camp)
    assert on_disk["device"] == str(adapters.device_for(ns)) == "cpu"
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["epoch"] for h in history] == [1, 2, 3]
    assert all(set(h) == EPOCH_KEYS for h in history)
    assert json.loads((run_dir / "netspec.resolved.json").read_text()) == camp.spec
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert on_disk["checkpoint_digest"] == T.digest_file(run_dir / "checkpoints" / "best.pt")
    assert (run_dir / "first_step.json").is_file() and (run_dir / "grad_stats.jsonl").is_file()
    assert row["primary_name"] == "mase" and row["primary"] is not None


# --------------------------------------------------------------------------- 2. the layout

def test_window_valid_predictions_keep_the_horizon_and_the_anchor(camps, ran):
    camp, _ = camps["window"]
    frame, meta = _stored(camp, "R0")
    horizon = camp.spec["data_contract"]["sample"]["horizon"]
    assert list(frame.columns) == (["row_id"] + [f"y_{k}" for k in range(horizon)]
                                   + [f"score_{k}" for k in range(horizon)])
    assert meta == {"vega.layout": "vega/predictions-layout@1", "vega.part": "valid",
                    "vega.row_id_kind": "raw_position", "vega.truth_shape": f"[{horizon}]",
                    "vega.score_shape": f"[{horizon}]", "vega.task_effective": "forecast",
                    "vega.score_source": "head"}
    assert all(frame[f"score_{k}"].dtype == np.float32 for k in range(horizon))
    adapters, ns, part = _valid_part(camp)
    assert frame["row_id"].tolist() == [int(r) for r in part.row_ids]     # no truncation
    unscored = json.loads((camp.run_dir("R0") / "run.json").read_text())["predictions_unscored"]
    assert unscored["count"] == len(part.unscored) > 0
    assert set(unscored["reasons"]) <= {"insufficient_context", "horizon_beyond_part"}
    raw = __import__("pandas").read_parquet(Path(ns.to_dict()["data_contract"]["parts"]["valid"]))
    assert len(frame) + unscored["count"] == len(raw)      # every valid row is accounted for
    # the stored head is best.pt's, row for row
    model = adapters.store.load(camp.run_dir("R0") / "checkpoints" / "best.pt", spec=ns,
                                build=adapters.build)
    heads = adapters.evaluator.heads(ns, part, model)
    np.testing.assert_array_equal(frame[[f"score_{k}" for k in range(horizon)]].to_numpy(),
                                  heads.astype(np.float32))


def test_node_level_valid_rows_are_the_nodes_raw_positions(camps, ran):
    camp, _ = camps["node"]
    frame, meta = _stored(camp, "R0")
    assert list(frame.columns) == ["row_id", "y", "score"]
    assert meta["vega.row_id_kind"] == "raw_position"
    assert meta["vega.truth_shape"] == "[]" and meta["vega.score_shape"] == "[]"
    assert meta["vega.task_effective"] == "binary"
    _, _, part = _valid_part(camp)
    assert frame["row_id"].tolist() == [int(r) for r in part.row_ids]
    assert set(frame["y"].unique()) <= {0, 1}


def test_graph_level_valid_rows_are_graph_ids(camps, ran):
    camp, _ = camps["graph"]
    frame, meta = _stored(camp, "R0")
    assert list(frame.columns) == ["row_id", "graph_id", "y", "score"]
    assert meta["vega.row_id_kind"] == "graph_id"
    assert frame["row_id"].tolist() == frame["graph_id"].tolist()
    _, _, part = _valid_part(camp)
    assert frame["graph_id"].tolist() == list(part.row_ids)


@pytest.mark.parametrize("key", ["window", "node", "graph"])
def test_without_an_evaluator_port_trainer_predict_stores_the_same_rows(camps, ran, key,
                                                                        tmp_path):
    """A blueprint that drew no Evaluator: `Trainer.predict` scores the part. Where the
    readout keeps every head column (a forecast's values, a binary logit) the stored rows are
    the Evaluator's, value for value."""
    import dataclasses
    import pandas as pd
    camp, _ = camps[key]
    adapters, ns, _ = _valid_part(camp)
    alone = dataclasses.replace(adapters, evaluator=None)
    done = T.write_valid_predictions(tmp_path, alone, ns,
                                     camp.run_dir("R0") / "checkpoints" / "best.pt", camp.root)
    frame, meta = _stored(camp, "R0")
    import pyarrow.parquet as pq
    fallback = pq.read_table(done["path"])
    fmeta = {k.decode(): v.decode() for k, v in fallback.schema.metadata.items()
             if k.startswith(b"vega.")}
    assert fmeta == meta
    pd.testing.assert_frame_equal(fallback.to_pandas(), frame)
    assert done["unscored"] == json.loads(
        (camp.run_dir("R0") / "run.json").read_text())["predictions_unscored"]


# --------------------------------------------------------------------------- 3. re-emission

def test_an_architecture_rung_re_emits_and_trains_its_own_package(camps, ran):
    camp, fixture = camps["window"]
    record, _, said = ran[("window", "R1b")]
    own = camp.run_dir("R1b") / fixture.package
    assert own.is_dir() and (own / "blueprint-map.json").is_file()
    assert Path(record["package_dir"]) == own and camp.package_dir_for("R1b") == own
    assert any("re-emitted" in line and "conformance" in line for line in said)
    assert (camp.run_dir("R1b") / "smoke.json").is_file()
    resolved = json.loads((camp.run_dir("R1b") / "netspec.resolved.json").read_text())
    assert resolved["family"]["name"] == "rnn"
    assert any(k.startswith(f"{fixture.package}/") for k in record["code_digests"])
    frame, meta = _stored(camp, "R1b")
    assert meta["vega.task_effective"] == "forecast" and len(frame) > 0


# --------------------------------------------------------------------------- 4. the replay

def test_the_replay_matches_and_writes_nothing(camps, ran):
    camp, _ = camps["window"]
    checkpoints = camp.run_dir("R0") / "checkpoints"
    before = {p.name: p.stat().st_mtime_ns for p in checkpoints.iterdir()}
    said = []
    assert T.replay(camp, "R0", 2, out=said.append) == 0, "\n".join(said)
    assert "2 epoch(s) match" in said[-1]
    assert {p.name: p.stat().st_mtime_ns for p in checkpoints.iterdir()} == before


def test_a_replay_that_does_not_match_is_exit_4(camps, ran, tmp_path):
    camp, _ = camps["graph"]
    copy = tmp_path / "graph"
    shutil.copytree(camp.root, copy)
    history = copy / "runs" / "R0" / "history.jsonl"
    lines = history.read_text().splitlines()
    first = json.loads(lines[0])
    first["valid"]["loss"] = float(first["valid"]["loss"]) + 0.5
    history.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    tampered, _ = T.Campaign.read(copy)
    said = []
    assert T.replay(tampered, "R0", 2, out=said.append) == 4
    assert any("valid.loss" in line for line in said)


# --------------------------------------------------------------------------- 5. the ledger

def test_the_floor_row_is_found_by_its_source(tmp_path):
    (tmp_path / "experiments.json").write_text(json.dumps({"runs": [
        {"id": "R0", "kind": "baseline", "source": "netspec", "primary": 0.8},
        {"id": "R1", "kind": "ablation", "source": "vega:ladder/floor", "primary": 0.4}]}))
    camp = T.Campaign(tmp_path, {}, {}, {})
    assert T._floor_primary(camp) == 0.4


def test_a_package_that_cannot_be_driven_is_refused_before_it_runs(camps, tmp_path):
    camp, fixture = camps["node"]
    copy = tmp_path / "node"
    shutil.copytree(camp.root, copy, ignore=shutil.ignore_patterns("runs"))
    bmap = copy / fixture.package / "blueprint-map.json"
    doc = json.loads(bmap.read_text())
    doc["entries"] = [e for e in doc["entries"] if e.get("role") != "Trainer"]
    bmap.write_text(json.dumps(doc))
    broken, _ = T.Campaign.read(copy)
    said = []
    record, row, code = T.run_one(broken, "R0", out=said.append)
    assert code == 2 and record is None
    assert any("cannot be driven" in line and "Trainer" in line for line in said)
    assert not (copy / "runs" / "R0" / "run.json").exists()


# --------------------------------------------------------------------------- 6. --resume

def _tree(root: Path) -> dict:
    """Every file under `root`, by relative path, with its digest."""
    return {p.relative_to(root).as_posix(): T.digest_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_superseding_id_skips_every_id_the_campaign_holds(tmp_path):
    (tmp_path / "runs" / "R2b").mkdir(parents=True)
    (tmp_path / "experiments.json").write_text(json.dumps({"runs": [{"id": "R0b"}]}))
    ladder = {"runs": [{"id": "R0"}, {"id": "R1"}, {"id": "R2"}]}
    camp = T.Campaign(tmp_path, {}, {}, ladder)
    assert camp.superseding_id("R1") == "R1c"          # R1b is the recurrent floor's
    assert camp.superseding_id("R2") == "R2c"          # runs/R2b exists
    assert camp.superseding_id("R0") == "R0c"          # the ledger holds R0b
    assert camp.superseding_id("R1b") == "R1bb"


def test_resuming_a_finished_r1_never_touches_r1b(camps, ran, tmp_path):
    camp, _ = camps["window"]
    root = tmp_path / "window"
    shutil.copytree(camp.root, root)
    ladder = json.loads((root / "ladder.json").read_text(encoding="utf-8"))
    ladder["runs"].append(PL.rung_floor(camp.spec, CATALOG)[0])
    (root / "ladder.json").write_text(json.dumps(ladder), encoding="utf-8")
    work, _ = T.Campaign.read(root)
    said = []
    _, _, code = T.run_one(work, "R1", out=said.append)
    assert code == 0, "\n".join(said)
    r1b, r1 = _tree(root / "runs" / "R1b"), _tree(root / "runs" / "R1")
    said = []
    record, row, code = T.run_one(work, "R1", resume=True, out=said.append)
    assert code == 0, "\n".join(said)
    assert record["id"] == "R1c" and record["supersedes"] == "R1"
    assert _tree(root / "runs" / "R1b") == r1b         # the recurrent floor, untouched
    assert _tree(root / "runs" / "R1") == r1           # and the run it supersedes
    log = (root / "runs" / "R1c" / "log.txt").read_text(encoding="utf-8")
    assert "warm-started from runs/R1/checkpoints/last.pt" in log
    resolved = json.loads((root / "runs" / "R1c" / "netspec.resolved.json").read_text())
    assert resolved["family"]["name"] == "linear"
    ids = [r["id"] for r in work.experiments()["runs"]]
    assert ids[-2:] == ["R1", "R1c"] and ids.count("R1b") == 1


# --------------------------------------------------------------------------- 7. unstated params

def test_a_trial_moves_a_param_the_spec_leaves_at_its_default(camps, ran, tmp_path):
    import hpo
    camp, _ = camps["window"]
    root = tmp_path / "unstated"
    shutil.copytree(camp.root, root, ignore=shutil.ignore_patterns("runs", "experiments.json"))
    spec = json.loads((root / "netspec.json").read_text(encoding="utf-8"))
    gru = next(b for b in spec["blocks"] if b["type"] == "gru")
    gru["params"].pop("num_layers", None)
    gru["params"].pop("layers", None)
    (root / "netspec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    work, _ = T.Campaign.read(root)
    space, _ = PL.build_space(spec, CATALOG)
    dim = next(d for d in space["dimensions"] if d["path"].endswith("/num_layers"))
    assert dim["current"] == 1 and "catalog" in dim["current_from"]
    delta = hpo.point_to_delta({dim["path"]: 2}, {"dimensions": [dim]}, spec)
    assert delta == [{"path": dim["path"], "from": None, "to": 2}]    # the spec states none
    base, stated = T.stated_defaults(spec, Delta.of(delta))
    assert stated.problems(base, None, multi_family_ok=True) == []
    rung = Rung.from_json({"id": "H1-1", "kind": "hpo", "study": "H1", "delta": delta,
                           "source": "train:hpo/H1", "hypothesis": "h", "expect": "e"})
    ladder = json.loads((root / "ladder.json").read_text(encoding="utf-8"))
    ladder["runs"].append({"id": "H1", "kind": "hpo", "study": "H1", "delta": [],
                           "space": "hpo/H1/space.json", "source": "vega:ladder/hpo",
                           "hypothesis": "h", "expect": "e"})
    (root / "ladder.json").write_text(json.dumps(ladder), encoding="utf-8")
    work, _ = T.Campaign.read(root)
    said = []
    record, _, code = T.run_one(work, "H1-1", delta_override=delta, rung=rung,
                                out=said.append)
    assert code == 0, "\n".join(said)
    assert record["delta"] == [{"path": dim["path"], "from": 1, "to": 2}]
    resolved = json.loads((root / "runs" / "H1-1" / "netspec.resolved.json").read_text())
    trained = next(b for b in resolved["blocks"] if b["id"] == gru["id"])
    assert trained["params"]["num_layers"] == 2
    assert any("re-emitted" in line for line in said)



# --------------------------------------------------------------------------- 9. the refusals

def _copy(camps, tmp_path, key="window", rungs=(), snapshot=None):
    camp, _ = camps[key]
    root = tmp_path / key
    shutil.copytree(camp.root, root, ignore=shutil.ignore_patterns("runs", "experiments.json"))
    ladder = json.loads((root / "ladder.json").read_text(encoding="utf-8"))
    ladder["runs"] += list(rungs)
    (root / "ladder.json").write_text(json.dumps(ladder), encoding="utf-8")
    if snapshot is not None:
        (root / "spec_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    work, err = T.Campaign.read(root)
    assert err is None, err
    return work


def _refused(work, run_id, needle, **kw):
    said = []
    record, row, code = T.run_one(work, run_id, out=said.append, **kw)
    text = "\n".join(said)
    assert code == 2 and record is None and row is None, text
    assert needle in text, text
    assert not (work.root / "runs").exists() and not (work.root / "experiments.json").exists()
    return text


LR = [{"path": "optimizer/params/lr", "to": 0.01}]
STUDY = {"id": "H1", "kind": "hpo", "study": "H1", "delta": [], "space": "hpo/H1/space.json",
         "source": "vega:ladder/hpo", "hypothesis": "h", "expect": "e"}


def test_a_delta_run_with_no_rung_is_refused(camps, tmp_path):
    work = _copy(camps, tmp_path)
    _refused(work, "RX", "has no rung in ladder.json", delta_override=LR)
    _refused(work, "H7-1", "has no rung in ladder.json", delta_override=LR)   # no study H7
    _refused(work, "RX", "has no rung in ladder.json")


def test_a_delta_that_is_not_the_rungs_own_is_refused(camps, tmp_path):
    work = _copy(camps, tmp_path)
    _refused(work, "R0", "--delta is not the rung's planned delta", delta_override=LR)


def test_a_study_is_not_a_run_and_a_trial_carries_its_point(camps, tmp_path):
    work = _copy(camps, tmp_path, rungs=[STUDY])
    _refused(work, "H1", "is a study, not a run")
    _refused(work, "H1-1", "carries the point its study sampled")


def test_a_dead_delta_is_refused_before_it_runs(camps, tmp_path):
    rung = {"id": "RX", "kind": "ablation", "family": "regularization",
            "source": "vega:ladder/default-test", "hypothesis": "h", "expect": "e",
            "delta": [{"path": "regularization[kind=weight_decay]/value", "to": 0.0}]}
    work = _copy(camps, tmp_path, rungs=[rung],
                 snapshot={"read_paths": ["regularization[kind=grad_clip]/value"]})
    _refused(work, "RX", "dead delta — no emitted code reads "
                         "regularization[kind=weight_decay]/value")
    shadow = dict(rung, id="RY", family="optimization",
                  delta=[{"path": "optimizer/params/weight_decay", "to": 0.0}])
    work = _copy(camps, tmp_path / "shadow", rungs=[shadow], snapshot={
        "read_paths": ["regularization[kind=weight_decay]/value"],
        "_resolved": {"regularization": {"shadowed_paths": ["optimizer/params/weight_decay"]}}})
    _refused(work, "RY", "a regularization[] declaration covers it")


def test_a_rung_whose_expect_cannot_be_checked_is_refused(camps, tmp_path):
    user = {"id": "RU", "kind": "ablation", "family": "optimization", "source": "user:turn4",
            "hypothesis": "h", "expect": "faster", "delta": LR}
    bad = dict(user, id="RB", source="vega:ladder/default-test",
               expect_check={"relation": "faster", "than": "R0"})
    work = _copy(camps, tmp_path, rungs=[user, bad])
    _refused(work, "RU", "states no expect_check")
    _refused(work, "RB", "relation 'faster' is not one of")


def test_hpo_refuses_a_study_the_ladder_does_not_plan(camps, tmp_path, capsys):
    import hpo
    work = _copy(camps, tmp_path)
    (work.root / "hpo" / "H9").mkdir(parents=True)
    (work.root / "hpo" / "H9" / "space.json").write_text(json.dumps(
        {"dimensions": [{"path": "optimizer/params/lr", "kind": "float",
                         "range": [1e-4, 1e-2]}]}), encoding="utf-8")
    code = hpo.main(["--campaign", str(work.root), "--study", "H9", "--space",
                     "hpo/H9/space.json", "--budget", "trials=1"])
    assert code == 2 and "is not an hpo rung of ladder.json" in capsys.readouterr().out
    assert not (work.root / "experiments.json").exists()


# --------------------------------------------------------------------------- 10. expect_held

def test_the_ledger_rows_record_their_expect_verdict(camps, ran):
    import apply_training_layer as ATL
    camp, _ = camps["window"]
    ledger = camp.experiments()
    rows = {r["id"]: r for r in ledger["runs"]}
    assert rows["R0"]["expect_check"]["relation"] == "learns"
    assert rows["R1b"]["expect_check"]["relation"] == "worse"
    for run_id in ("R0", "R1b"):
        row = rows[run_id]
        assert row["expect_check"]["text"], run_id
        assert isinstance(row["expect_held"], bool), (run_id, row.get("expect_evidence"))
        assert row["expect_evidence"], run_id
    assert "against R0" in rows["R1b"]["expect_evidence"]
    mode = (camp.spec["metrics"].get("selection") or {}).get("mode", "max")
    assert ATL.expect_problems(dict(ledger, ranking={"mode": mode})) == []
    said = ran[("window", "R1b")][2]
    assert any(line.startswith("OK: R1b: expect ") for line in said), said


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))


# --------------------------------------------------------------------------- 11. F37

def _refused_to_compile(work, run_id, needle):
    """A rung the compiler refuses: exit 2, no record, no ledger row, the reason said (the
    resolved spec it tried is kept in the run directory, as for any uncompilable rung)."""
    said = []
    record, row, code = T.run_one(work, run_id, out=said.append)
    text = "\n".join(said)
    assert code == 2 and record is None and row is None, text
    assert needle in text, text
    assert not (work.root / "experiments.json").exists()
    return text

def test_a_campaign_planned_without_a_model_says_so_before_any_rung(camps, tmp_path):
    """F37 (rehearsal final3r2): a campaign.json with no input.model made every rung, R0
    included, read as "the delta produces a spec that does not compile" — the compiler's
    usage error (--model required) was swallowed. The refusal names the cause and the fix."""
    work = _copy(camps, tmp_path)
    path = work.root / "campaign.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["input"]["model"] = None
    path.write_text(json.dumps(doc), encoding="utf-8")
    work, err = T.Campaign.read(work.root)
    assert err is None, err
    text = _refused_to_compile(work, "R0", "campaign.json records no input.model")
    assert "plan_ladder.py --campaign <dir> --model" in text


def test_a_compiler_failure_without_an_error_line_still_says_why(camps, tmp_path, monkeypatch):
    """F37: a compiler that fails with a usage error or a traceback has no `ERROR:` line; its
    last lines are the reason, and the refusal prints them."""
    work = _copy(camps, tmp_path)
    usage = ("usage: compile_spec.py [-h] --spec SPEC --model MODEL\n"
             "compile_spec.py: error: the following arguments are required: --model\n")
    monkeypatch.setattr(T, "_run_cli", lambda script, args, cwd=None, forge=None: (2, usage))
    text = _refused_to_compile(work, "R0", "does not compile")
    assert "the following arguments are required: --model" in text
