#!/usr/bin/env python3
"""
hpo.py — run one study, whose trials are runs in the same ledger.

    python3 hpo.py --campaign <dir> --study H1 [--space hpo/H1/space.json]
                   [--budget trials=20,minutes=8] [--seed 7] [--vega-forge-dir PATH]

A trial is not a special kind of thing: `hpo.py` samples a point of the study's space into a
**delta**, and calls the same `trainer.run_one` every ladder rung goes through. So every trial
recompiles, records its digests, writes its history and gets its diagnostics evaluated, and its
ledger row sits beside `R0`'s. The one difference the contract allows is that a trial's delta
may span decision families — it samples a declared space, and its source says so.

Optuna's TPE sampler and median pruner are used when optuna is installed; otherwise the study
is a random search with the same interface and a `WARN:` saying so. Pruning is a real stop:
the pruner's verdict is raised out of the epoch callback, so a pruned trial has spent exactly
the epochs its partial history shows and the rest of the budget is still there.

A study is a rung of `ladder.json` (kind `hpo`): a study the ladder does not hold is refused,
because its trials would be runs without a hypothesis. Its `expect` is judged once, on the
study's record in `experiments.json` `hpo[]` (`expect_check`, its own or the catalog's `hpo`
row: the best trial at least the baseline), and the trials' rows say they inherit it.

`importance.json` is fANOVA under optuna and, without it, the spread of the objective across
each dimension's quantiles — the layer shows which of the two produced the numbers, because
they are not the same claim.

Exit codes: 0 the study ran; 2 an input error; 3 every trial failed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_train_core import Rung, load_catalog, read_path, normalise_path  # noqa: E402
from expectations import check_for, rejudge  # noqa: E402
import trainer as T  # noqa: E402


# --------------------------------------------------------------------------- the space

def parse_budget(text):
    """`trials=20,minutes=8` — both optional, and whichever ends first ends the study."""
    out = {"trials": None, "minutes": None}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        key = key.strip().lower()
        if key in ("trials", "n", "n_trials"):
            out["trials"] = int(value)
        elif key in ("minutes", "min", "wall_clock_min"):
            out["minutes"] = float(value)
        else:
            raise ValueError(f"unknown budget key {key!r}; use trials=… and minutes=…")
    return out


def _suggest_random(dim, rng):
    kind = dim.get("kind", "float")
    if kind == "categorical":
        return rng.choice(list(dim["choices"]))
    lo, hi = dim["range"]
    if dim.get("log"):
        value = math.exp(rng.uniform(math.log(float(lo)), math.log(float(hi))))
    else:
        value = rng.uniform(float(lo), float(hi))
    return int(round(value)) if kind == "int" else float(value)


def _suggest_optuna(trial, dim):
    kind = dim.get("kind", "float")
    name = dim["path"]
    if kind == "categorical":
        return trial.suggest_categorical(name, list(dim["choices"]))
    lo, hi = dim["range"]
    if kind == "int":
        return trial.suggest_int(name, int(lo), int(hi), log=bool(dim.get("log")))
    return trial.suggest_float(name, float(lo), float(hi), log=bool(dim.get("log")))


def point_to_delta(point, space, spec):
    """A sampled point as a delta the trainer can apply, with every `from` read off the spec.

    The linked widths of the catalog's note are honoured here: when the space moves the first
    dense block's width, the residual's inner dense blocks move with it, or `x + f(x)` stops
    adding up and the run would fail at the emitter rather than telling the study anything.
    """
    edits = []
    for dim in space["dimensions"]:
        path = dim["path"]
        value = point[path]
        try:
            current = read_path(spec, normalise_path(path))
        except Exception:  # noqa: BLE001
            current = None
        edits.append({"path": path, "from": current, "to": value})
        if path.endswith("/params/units"):
            edits.extend(_linked_widths(spec, current, value))
    return edits


def _linked_widths(spec, was, now):
    """A residual's inner dense widths follow the block that feeds it."""
    out = []
    for block in spec.get("blocks") or []:
        if not isinstance(block, Mapping) or block.get("type") != "residual":
            continue
        for i, inner in enumerate((block.get("params") or {}).get("inner") or []):
            if isinstance(inner, Mapping) and inner.get("type") == "dense" \
                    and (inner.get("params") or {}).get("units") == was:
                out.append({"path": f"blocks/{block.get('id')}/params/inner/{i}/params/units",
                            "from": was, "to": now})
    return out


# --------------------------------------------------------------------------- importance

def importance_from_trials(trials, space):
    """Without optuna: how much the objective moves across each dimension's own quantiles.

    This is a weaker claim than fANOVA and the record says so — it is a marginal spread, not a
    variance decomposition, and a dimension that only matters in interaction will look inert.
    """
    scored = [t for t in trials if t.get("value") is not None]
    if len(scored) < 4:
        return {}, "too few completed trials for a marginal spread"
    out = {}
    for dim in space["dimensions"]:
        path = dim["path"]
        pairs = [(t["params"].get(path), t["value"]) for t in scored
                 if t["params"].get(path) is not None]
        if len(pairs) < 4:
            continue
        try:
            pairs.sort(key=lambda p: p[0])
        except TypeError:                       # categorical: group by value instead
            groups = {}
            for value, score in pairs:
                groups.setdefault(value, []).append(score)
            means = [statistics.fmean(v) for v in groups.values() if v]
            out[path] = (max(means) - min(means)) if len(means) > 1 else 0.0
            continue
        half = len(pairs) // 2
        low = statistics.fmean(v for _, v in pairs[:half])
        high = statistics.fmean(v for _, v in pairs[-half:])
        out[path] = abs(high - low)
    total = sum(out.values())
    if total > 0:
        out = {k: round(v / total, 4) for k, v in out.items()}
    return out, "marginal spread across each dimension's quantiles (optuna absent)"


# --------------------------------------------------------------------------- the study

def run_study(camp: T.Campaign, study_id, space, budget, *, seed=7, forge=None, out=print):
    mode = (camp.campaign.get("metrics") or {})
    metrics = camp.spec.get("metrics") or {}
    primary = metrics.get("primary")
    direction = (metrics.get("selection") or {}).get("mode", "max")
    study_dir = camp.root / "hpo" / study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    trials_path = study_dir / "trials.jsonl"

    rung = camp.rung(study_id)
    refused = unplanned(rung, study_id)
    if refused:
        out(f"ERROR: {refused}")
        return 2

    n_trials = budget.get("trials") or 12
    minutes = budget.get("minutes")
    per_trial = (minutes / n_trials) if minutes else None
    started = time.time()

    optuna = None
    try:
        import optuna as _optuna
        optuna = _optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        out("WARN: optuna absent; random search with the same interface, and the median "
            "pruner is replaced by the running median of completed trials")

    if optuna is not None:
        sampler = optuna.samplers.TPESampler(seed=seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
        study = optuna.create_study(direction="maximize" if direction == "max" else "minimize",
                                    sampler=sampler, pruner=pruner)
    rng = random.Random(seed)
    records, completed, pruned, failed = [], [], 0, 0

    for i in range(1, n_trials + 1):
        if minutes and (time.time() - started) / 60.0 >= minutes:
            out(f"OK: {study_id}: the study's {minutes} min are spent after {i - 1} trial(s)")
            break
        run_id = f"{study_id}-{i}"

        if optuna is not None:
            trial = study.ask()
            point = {d["path"]: _suggest_optuna(trial, d) for d in space["dimensions"]}
        else:
            trial, point = None, {d["path"]: _suggest_random(d, rng)
                                  for d in space["dimensions"]}

        delta = point_to_delta(point, space, camp.spec)
        seen_values = []

        def on_epoch(row, _trial=trial, _values=seen_values):
            value = (row.get("valid") or {}).get(primary)
            if value is None:
                return
            _values.append(value)
            epoch = int(row.get("epoch") or len(_values))
            if _trial is not None:
                _trial.report(float(value), epoch)
                if _trial.should_prune():
                    raise T.Pruned(epoch, value)
            elif len(completed) >= 3 and epoch >= 4:
                median = statistics.median(completed)
                worse = value < median if direction == "max" else value > median
                if worse and epoch >= 6:
                    raise T.Pruned(epoch, value)

        record, row, code = T.run_one(
            camp, run_id, delta_override=delta, budget_min=per_trial, forge=forge, out=out,
            on_epoch=on_epoch, rung=T.trial_rung(rung, run_id, delta))
        if code != 0 or record is None:
            failed += 1
            if optuna is not None and trial is not None:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            _append_trial(trials_path, records, {"id": run_id, "params": point, "value": None,
                                                 "status": "failed"})
            continue

        value = (row or {}).get("primary")
        status = record.get("status")
        if status == "pruned":
            pruned += 1
            if optuna is not None and trial is not None:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            if value is not None:
                completed.append(float(value))
            if optuna is not None and trial is not None:
                study.tell(trial, float(value) if value is not None else
                           (-math.inf if direction == "max" else math.inf))
        _append_trial(trials_path, records, {"id": run_id, "params": point, "value": value,
                                             "status": status,
                                             "epochs": record.get("epochs")})

    # --- importance
    if optuna is not None and len([r for r in records if r["value"] is not None]) >= 4:
        try:
            raw = optuna.importance.get_param_importances(study)
            importance = {k: round(float(v), 4) for k, v in raw.items()}
            how = "fANOVA (optuna)"
        except Exception as exc:  # noqa: BLE001
            importance, how = importance_from_trials(records, space)
            how += f" (optuna's fANOVA failed: {str(exc)[:80]})"
    else:
        importance, how = importance_from_trials(records, space)

    best = None
    scored = [r for r in records if r["value"] is not None and r["status"] != "pruned"]
    if scored:
        pick = max if direction == "max" else min
        best = pick(scored, key=lambda r: r["value"])

    (study_dir / "importance.json").write_text(
        json.dumps({"schema": "vega/importance@1", "study": study_id, "how": how,
                    "importance": importance,
                    "dimensions": [d["path"] for d in space["dimensions"]]},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {"study": study_id, "trials": len(records), "completed": len(scored),
               "pruned": pruned, "failed": failed,
               "best": best["id"] if best else None,
               "best_value": best["value"] if best else None,
               "importance": importance, "importance_how": how,
               "sampler": "TPE (optuna)" if optuna is not None else "random",
               "pruner": "median (optuna)" if optuna is not None else
                         "running median of completed trials",
               "space": [d["path"] for d in space["dimensions"]],
               "generic_space": bool(space.get("generic"))}
    summary.update(_study_expect(rung, camp.spec))
    summary = _record_study(camp, summary)

    out(f"OK: {study_id}: {len(records)} trial(s) — {len(scored)} completed, {pruned} pruned, "
        f"{failed} failed; best {summary['best']} at {_fmt(summary['best_value'])} "
        f"({summary['sampler']})")
    verdict = {True: "held", False: "did not hold"}.get(summary.get("expect_held"), "pending")
    out(f"OK: {study_id}: expect {verdict} — {summary.get('expect_evidence')}")
    if importance:
        out(f"OK: {study_id}: importance by {how} — " +
            ", ".join(f"{k} {v}" for k, v in sorted(importance.items(),
                                                    key=lambda kv: -kv[1])))
    else:
        out(f"WARN: {study_id}: no importance could be computed ({how})")
    return 0 if scored else 3


def _fmt(value):
    return "·" if value is None else f"{float(value):.5g}"


def _append_trial(path, records, record):
    records.append(record)
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _study_expect(rung, spec) -> dict:
    """The study's expect and its check (the rung's own, else the catalog's `hpo` row),
    pending until `_record_study` judges it against the ledger."""
    catalog, _ = load_catalog()
    check, why = check_for(rung.expect_check, rung.source or "vega:ladder/hpo", catalog,
                           rung.id)
    primary = ((spec or {}).get("metrics") or {}).get("primary")
    return {"expect": rung.expect,
            "expect_check": check.as_json(primary) if check is not None else None,
            "expect_held": None,
            "expect_evidence": "" if check is not None else why}


def _record_study(camp: T.Campaign, summary):
    """The study's record in the ledger, its expect judged (and every other pending one)."""
    ledger = camp.experiments()
    studies = [s for s in ledger.get("hpo") or [] if s.get("study") != summary["study"]]
    studies.append(summary)
    ledger["hpo"] = studies
    rejudge(ledger, camp.spec)
    camp.write_experiments(ledger)
    return studies[-1]


# --------------------------------------------------------------------------- the shell

def unplanned(rung, study_id) -> str | None:
    """Why `study_id` cannot run, or None: a study is an `hpo` rung of `ladder.json`."""
    if rung is not None and rung.kind == "hpo":
        return None
    return (f"study {study_id!r} is not an hpo rung of ladder.json — a study is planned like "
            "every rung, with its hypothesis, source and expect, or its trials would be runs "
            "nobody stated a hypothesis for (run-contract §3). Plan it in ladder.json first.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--space", default=None,
                    help="default: the study rung's `space` in ladder.json")
    ap.add_argument("--budget", default=None, metavar="trials=20,minutes=8")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--vega-forge-dir", default=None)
    a = ap.parse_args(argv)

    camp, err = T.Campaign.read(a.campaign)
    if err:
        print(f"ERROR: {err}")
        return 2
    if camp.sealed():
        print("ERROR: this campaign's test part has been opened; a study never reopens the "
              "campaign that saw it")
        return 2

    rung = camp.rung(a.study)
    refused = unplanned(rung, a.study)
    if refused:
        print(f"ERROR: {refused}")
        return 2
    space_path = a.space or rung.space
    if not space_path:
        print(f"ERROR: no space for study {a.study!r}: pass --space, or let plan_ladder.py "
              "write the study rung with its `space`")
        return 2
    path = Path(space_path)
    if not path.is_absolute():
        path = camp.root / space_path
    try:
        space = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: the space could not be read ({path}): {exc}")
        return 2
    if not space.get("dimensions"):
        print(f"ERROR: {path} declares no dimensions; there is nothing to search")
        return 2

    try:
        budget = parse_budget(a.budget) if a.budget else {}
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    if not budget.get("trials"):
        budget["trials"] = (rung.budget or {}).get("trials") or 12
    if not budget.get("minutes"):
        budget["minutes"] = (rung.budget or {}).get("wall_clock_min")

    existing = [r for r in (camp.experiments().get("runs") or [])
                if str(r.get("id", "")).startswith(f"{a.study}-")]
    if existing:
        print(f"ERROR: study {a.study!r} already has {len(existing)} trial(s) in the ledger "
              f"({', '.join(r['id'] for r in existing[:5])}…). A campaign is append-only: "
              "re-running a study under the same id would leave two sets of trials claiming "
              "the same names. Use a new study id (H2, H3, …) — the ledger then shows both "
              "studies and what each one searched.")
        return 2

    print(f"OK: study {a.study} — {len(space['dimensions'])} dimension(s) "
          f"({', '.join(d['path'] for d in space['dimensions'])}), "
          f"{budget['trials']} trial(s), {budget.get('minutes')} min")
    if space.get("generic"):
        print("WARN: the family has no §4 search space; this space is generic (lr plus every "
              "regularization value the spec carries)")

    return run_study(camp, a.study, space, budget,
                     seed=a.seed if a.seed is not None else camp.spec.get("seed", 7),
                     forge=a.vega_forge_dir)


if __name__ == "__main__":
    sys.exit(main())
