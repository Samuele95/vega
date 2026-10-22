#!/usr/bin/env python3
"""
roundtrip_fixtures.py — B15's spike fixtures as campaigns the real `pack.py` packs (build plan
S10's closing test: "the round trip passes on B15's three fixtures packed with the real
pack.py"). Not a test file: `test_roundtrip_plan.py` drives it, and it runs on its own.

    python3 roundtrip_fixtures.py <work-dir> [--fixture window,node-keypair,graph-level|all]
                                  [--json]

Per fixture, under `<work-dir>/<name>/`:

  (the spike's own files)  raw.csv, shaped/, netspec.json, pipeline.json, package/<package>/ —
                           `make_spike_fixtures.make` (the real shape.py and the v2.3 emitter)
  campaign/                a campaign as /vega-train leaves it: campaign.json, netspec.json,
                           the emitted package, runs/R0/ (netspec.resolved.json, the fit's
                           checkpoints/best.pt through the package's ModelRegistry, run.json)
                           and experiments.json (the ledger row with its spec, checkpoint and
                           code digests, the selection); the fit is the package's own
                           `Trainer().fit` over `ShapedParts`, in a child process
  campaign/eval/R0/        the REAL `vega-eval/scripts/evaluate.py` under its seal: the stored
                           predictions in `vega/predictions-layout@1`, unscored.json,
                           threshold.json, metrics.json
  bundle/<slug>-R0/        the REAL `vega-pack/scripts/pack.py --serve --unattended
                           --ship-anyway …` over that campaign

Nothing here stands in for a stage the chain runs: the harness only writes the two records a
campaign needs and /vega-train's ladder would have written (the ledger row and the selection),
because the spike carries no model page to plan a ladder from. `packed(name, work, extra)`
returns the `Packed` record (the exit codes and the output of both CLIs, the bundle directory);
`pack(campaign, fixture, bundle_dir, extra)` packs a campaign again (a seeded copy, other
flags). The CLI makes the closing three by default; `--json` prints the records as one JSON
document and the account on stderr. Exit codes: 0 every fixture packed and its round trip
passed; 1 one did not; 2 a fixture could not be made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
SKILLS = PACK.parent.parent
FORGE_SCRIPTS = SKILLS / "vega-forge" / "scripts"
SPIKE = FORGE_SCRIPTS / "tests" / "spike"
EVALUATE = SKILLS / "vega-eval" / "scripts" / "evaluate.py"
PACK_PY = PACK / "pack.py"
for _p in (str(SPIKE), str(FORGE_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLOSING = ("window", "node-keypair", "graph-level")
RUN = "R0"
N_BOOT = 50
STANCE = "the round-trip harness packs a 3-epoch spike network; the verdict is not its claim"
TIMEOUT = 1200


@dataclass(frozen=True)
class Packed:
    """One fixture through evaluate.py and pack.py: where things are and what each CLI said."""

    name: str
    campaign: str
    shaped: str
    pipeline: str
    bundle: str | None
    eval_exit: int
    eval_output: str
    pack_exit: int
    pack_output: str

    @property
    def manifest(self) -> dict:
        return json.loads((Path(self.bundle) / "MANIFEST.json").read_text(encoding="utf-8"))

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(doc) -> "Packed":
        return Packed(**doc)


def _digest_file(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _env() -> dict:
    """The child processes' environment: CPU only, a bounded thread count, no bytecode."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def _run(argv, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, argv)], capture_output=True, text=True,
                          timeout=TIMEOUT, env=_env(), cwd=cwd)


# --------------------------------------------------------------------------- the campaign

def campaign(name: str, work: Path) -> tuple:
    """(the spike Fixture, the campaign directory) with R0 fitted and selected."""
    import make_spike_fixtures as msf
    fixture = msf.make(name, Path(work))
    camp = Path(fixture.root) / "campaign"
    shutil.rmtree(camp, ignore_errors=True)
    (camp / "runs" / RUN).mkdir(parents=True)
    shutil.copytree(fixture.package_dir, camp / fixture.package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    spec_text = Path(fixture.spec).read_text(encoding="utf-8")
    (camp / "netspec.json").write_text(spec_text, encoding="utf-8")
    (camp / "runs" / RUN / "netspec.resolved.json").write_text(spec_text, encoding="utf-8")
    (camp / "campaign.json").write_text(json.dumps(
        {"schema": "vega/campaign@1", "input": {"package": fixture.package,
                                               "spec": "netspec.json"},
         "backend": {"name": "pytorch"}}, indent=2) + "\n", encoding="utf-8")
    fit = _run([__file__, "--train", camp, "--package", fixture.package])
    if fit.returncode != 0:
        raise RuntimeError(f"{name}: the fit failed\n{fit.stdout[-2000:]}{fit.stderr[-2000:]}")
    _ledger(camp, fixture.package, json.loads(fit.stdout.strip().splitlines()[-1]))
    return fixture, camp


def _ledger(camp: Path, package: str, fit: dict) -> None:
    """experiments.json as /vega-train writes it for one run and its selection."""
    run_dir = camp / "runs" / RUN
    best = run_dir / "checkpoints" / "best.pt"
    code = {f"{package}/{p.relative_to(camp / package).as_posix()}": _digest_file(p)
            for p in sorted((camp / package).rglob("*.py"))}
    row = {"id": RUN, "kind": "baseline", "status": "done", "epochs": fit["epochs"],
           "best_epoch": fit["best_epoch"], "primary": fit["best_value"],
           "spec_digest": _digest_file(run_dir / "netspec.resolved.json"),
           "checkpoint_digest": _digest_file(best), "checkpoint": str(best),
           "code_digests": code}
    (run_dir / "run.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    ledger = {"schema": "vega/experiments@1", "runs": [row],
              "selection": {"run": RUN, "rule": "the harness's one run"}}
    (camp / "experiments.json").write_text(json.dumps(ledger, indent=2) + "\n",
                                           encoding="utf-8")


def _train(camp: Path, package: str) -> int:
    """The child process: the package's own Trainer over ShapedParts, best.pt through its
    ModelRegistry (the Trainer saves at each improvement), last.pt beside it."""
    from vega_adapters import PackageAdapters
    adapters = PackageAdapters.load(camp / package, package)
    run_dir = camp / "runs" / RUN
    spec_dict = json.loads((run_dir / "netspec.resolved.json").read_text(encoding="utf-8"))
    domain = adapters.package.domain
    spec = domain.NetSpec.from_dict(spec_dict)
    model = adapters.build(spec)
    checkpoints = run_dir / "checkpoints"
    result = adapters.trainer.fit(model, spec, adapters.data, callbacks=None,
                                  run=domain.RunContext(checkpoint_dir=str(checkpoints)))
    adapters.store.save(model, checkpoints / "last.pt", spec=spec, epoch=result.epochs_run)
    if not (checkpoints / "best.pt").is_file():
        print("ERROR: the fit wrote no best.pt")
        return 1
    print(json.dumps({"epochs": int(result.epochs_run), "best_epoch": _plain(result.best_epoch),
                      "best_value": _plain(result.best_value)}))
    return 0


def _plain(value):
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, (int, float, str, type(None))) else str(value)


# --------------------------------------------------------------------------- the stages

def _baselines(spec: dict) -> str:
    """The floor, a contender, and what the task and shape add (evaluate.py's default_when)."""
    sample = (spec.get("data_contract") or {}).get("sample") or {}
    out = ["trivial", "linear"]
    if (spec.get("task") or {}).get("kind") == "forecast":
        out.append("naive")
    if sample.get("shape") == "graph" and (sample.get("level") or "node") == "node":
        out.append("graph-linear")
    return ",".join(out)


def evaluate(camp: Path) -> subprocess.CompletedProcess:
    spec = json.loads((camp / "netspec.json").read_text(encoding="utf-8"))
    return _run([EVALUATE, "--campaign", camp, "--run", RUN, "--unattended", "--n-boot",
                 N_BOOT, "--baselines", _baselines(spec)])


def monitor_plan(camp: Path, fixture) -> Path:
    """The harness's monitor plan (`monitor_plans.py`): the spike blueprints' Monitor port says
    filled_by /vega-pack, and pack refuses to leave it unfilled."""
    plan = Path(camp) / "monitor-plan.json"
    done = _run([HERE / "monitor_plans.py", "--campaign", camp, "--pipeline", fixture.pipeline,
                 "--out", plan])
    if done.returncode != 0:
        raise RuntimeError(f"monitor_plans.py exited {done.returncode}: {done.stdout[-2000:]}")
    return plan


def pack(camp: Path, fixture, bundle_dir: Path, extra=()) -> subprocess.CompletedProcess:
    shutil.rmtree(bundle_dir, ignore_errors=True)
    return _run([PACK_PY, "--campaign", camp, "--shaped", fixture.shaped, "--bundle-dir",
                 bundle_dir, "--pipeline", fixture.pipeline, "--ship-anyway", STANCE,
                 "--unattended", "--serve", "--demo-sample", "3",
                 "--monitor-plan", monitor_plan(camp, fixture), *extra])


def packed(name: str, work: Path, extra=()) -> Packed:
    """One fixture made, fitted, evaluated by the real evaluate.py and packed by the real
    pack.py."""
    fixture, camp = campaign(name, Path(work))
    ev = evaluate(camp)
    if ev.returncode != 0:
        return Packed(name, str(camp), fixture.shaped, fixture.pipeline, None, ev.returncode,
                      ev.stdout + ev.stderr, -1, "")
    bundle_dir = Path(fixture.root) / "bundle"
    pk = pack(camp, fixture, bundle_dir, extra)
    bundles = sorted(p for p in bundle_dir.glob("*") if (p / "MANIFEST.json").is_file()) \
        if bundle_dir.is_dir() else []
    return Packed(name, str(camp), fixture.shaped, fixture.pipeline,
                  str(bundles[0]) if bundles else None, ev.returncode, ev.stdout + ev.stderr,
                  pk.returncode, pk.stdout + pk.stderr)


def passed(result: Packed) -> bool:
    if result.pack_exit != 0 or not result.bundle:
        return False
    return (result.manifest.get("roundtrip") or {}).get("status") == "pass"


# --------------------------------------------------------------------------- the CLI

def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", nargs="?")
    ap.add_argument("--fixture", default=",".join(CLOSING))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--train", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--package", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)
    if a.train:
        return _train(Path(a.train), a.package)
    if not a.work:
        print("ERROR: a work directory is required")
        return 2
    import make_spike_fixtures as msf
    names = list(msf.FIXTURES) if a.fixture == "all" else a.fixture.split(",")
    # with --json the one JSON document is stdout; the account goes to stderr
    say = (lambda text: print(text, file=sys.stderr)) if a.json else print
    results, failed = [], 0
    for name in names:
        try:
            result = packed(name, Path(a.work))
        except Exception as exc:                                     # noqa: BLE001
            say(f"ERROR: {name}: {type(exc).__name__}: {exc}")
            return 2
        results.append(result)
        failed += not passed(result)
        _report(result, say)
    if a.json:
        print(json.dumps([r.as_dict() for r in results], indent=1))
    return 1 if failed else 0


def _report(result: Packed, say) -> None:
    ok = passed(result)
    trip = (result.manifest.get("roundtrip") or {}) if result.bundle else {}
    say(f"{'OK' if ok else 'FAIL'}: {result.name}: evaluate exit {result.eval_exit}, pack exit "
        f"{result.pack_exit}, round trip {trip.get('status')} on {trip.get('rows')} row(s), "
        f"worst |delta| {trip.get('max_abs_diff')}"
        + (f", boundary {trip['boundary']['rows']} row(s)" if trip.get("boundary") else ""))
    if not ok:
        tail = (result.pack_output or result.eval_output).strip().splitlines()[-12:]
        say("\n".join("    " + line for line in tail))


if __name__ == "__main__":
    sys.exit(main())
