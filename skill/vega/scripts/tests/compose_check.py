#!/usr/bin/env python3
"""
compose_check.py — every lane's composed netspec against the compiler (BUILD-PLAN T02, the W1
integration check: "spec.compose of every lane passes compile_spec.py --check-only").

    python3 compose_check.py [--lanes all|L01,L05,...] [--work DIR] [-j N] [--json OUT]

For each lane of `lanes.json`:

- the acceptance lane (L01) composes over the orders fixture `make_fixture.py` builds, with the
  names that fixture's layers use (its partition keyed by label, its balance finding F2);
- a lanekit lane builds its fixture with `lanekit.fixture.make_lane` — the real shaper, the
  composed model and its layers, the world model, the memory — from its generator, or, while
  the generator has not landed (T05–T07), from a **stand-in** (lanekit.standin) derived from
  the lane's own declaration: the target and its classes, the columns its blocks embed, its
  window's series and time columns, its graph's keys, a few numeric columns. The stand-in
  proves the composer and the fixture on the lane's shape, task and blocks; the lane's real
  data arrives with its generator, and the check runs on it unchanged;
- the golden lane (G00) is not composed: its spec is the recorded CIC campaign's.

Then `compile_spec.py --check-only --json` on the lane's netspec, model, pipeline and memory.
A lane passes when the compiler exits 0 with no more `WARN:` than the lane's forge budget
(`expect.markers.forge.warn_max`, default 0 — the forge stage's own rule). One refusal is
reported and not failed: rule B2 naming a block, family or metric the backend's `coverage.json`
still lists under `planned` — the v2.3 blocks move into `blocks` at B14 (W3, BUILD-PLAN §2.12
rule 1), and the refusal disappears with the `planned` entry, by construction. The deferral
reads `planned` at every run: once B14 empties it nothing is deferred, and a B2 refusal fails
its lane like any other.

Both allowances are W1's (the orchestrator ratifies them): the stand-in generators until T05–T07
land (the W5 integration check re-runs this command on the real ones, which it picks up by
file), and the B2 deferral until gate A.

Exit codes: 0 every lane passed; 1 a lane failed; 2 a usage error.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import contracts  # noqa: E402
from lanekit.registry import Lane, load_lanes  # noqa: E402
from lanekit.standin import StandIn  # noqa: E402

SKILLS = HERE.parents[2]
COMPILER = SKILLS / "vega-forge" / "scripts" / "compile_spec.py"
MAKE_FIXTURE = SKILLS / "vega-forge" / "scripts" / "tests" / "make_fixture.py"
DEFAULT_WORK = Path(os.environ.get("TMPDIR", "/tmp")) / "vega-compose-check"
# the orders fixture's names: partitions keyed by label, its balance finding numbered F2
ORDERS_PARTITION = "geometry:partitions/late"
ORDERS_FINDINGS = {"balance": "F2", "drift": "F3", "importance": "F5"}


# --------------------------------------------------------------------------- one lane
@dataclass
class Outcome:
    lane: str
    status: str                 # pass | fail | skip
    detail: str
    generator: str = ""
    deferred: tuple[str, ...] = ()
    warns: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    seconds: float = 0.0

    def to_json(self) -> dict:
        return {"lane": self.lane, "status": self.status, "detail": self.detail, "generator": self.generator,
                "deferred": list(self.deferred), "warns": list(self.warns), "errors": list(self.errors),
                "seconds": round(self.seconds, 1)}


def planned(skills: Path) -> set[str]:
    coverage = json.loads((skills / contracts.BACKEND_COVERAGE).read_text(encoding="utf-8"))
    p = coverage.get("planned") or {}
    return set(p.get("blocks") or []) | set(p.get("families") or []) | set(p.get("metrics") or [])


def classify(markers: Sequence[Mapping[str, Any]], lane: Lane, still_planned: set[str]) -> tuple[list, list, list]:
    """(errors, deferred, warns): a B2 refusal of a name the backend still plans is deferred;
    with `still_planned` empty (after B14) every refusal is an error."""
    errors, deferred, warns = [], [], []
    for m in markers:
        text = f"{m.get('rule')}: {(m.get('where') + ': ') if m.get('where') else ''}{m.get('message')}"
        if m.get("level") == "ERROR":
            message = str(m.get("message"))
            names = set(re.findall(r"'([A-Za-z0-9_\-]+)'", message))
            if m.get("rule") == "B2" and "planned" in message and names & still_planned:
                deferred.append(text)
            else:
                errors.append(text)
        elif m.get("level") == "WARN":
            warns.append(text)
    return errors, deferred, warns


def compile_markers(netspec: Path, model: Path, pipeline: Path, memory: Path, cwd: Path) -> tuple[int, list, str]:
    proc = subprocess.run([sys.executable, str(COMPILER), "--spec", str(netspec), "--model", str(model),
                           "--pipeline", str(pipeline), "--memory", str(memory), "--check-only", "--json"],
                          capture_output=True, text=True, cwd=str(cwd), timeout=300)
    try:
        doc, _ = json.JSONDecoder().raw_decode(proc.stdout.lstrip())
        return proc.returncode, list(doc.get("markers") or []), proc.stdout
    except (json.JSONDecodeError, ValueError):
        return proc.returncode, [], (proc.stdout + proc.stderr)[-3000:]


def build_l01(lane: Lane, work: Path) -> tuple[Path, Path, Path, Path]:
    """The orders fixture, and the lane's spec composed over it."""
    from lanekit import spec
    from lanekit.chain import platform_writer

    proc = subprocess.run([sys.executable, str(MAKE_FIXTURE), str(work)], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"make_fixture.py exited {proc.returncode}: {(proc.stdout + proc.stderr)[-1500:]}")
    module = platform_writer(SKILLS)
    model = work / "orders.domain.analysis.shaped.html"
    pipeline = next((work / "blueprint-runs").glob("*/pipeline.json"))
    composer = spec.Composer(spec.SourceNames(ORDERS_PARTITION, {**spec.FINDING_IDS, **ORDERS_FINDINGS}))
    composed = composer.compose(lane, json.loads((work / "shaped/manifest.json").read_text(encoding="utf-8")),
                                json.loads((work / "shaped/fitted.json").read_text(encoding="utf-8")),
                                module.domain_digest(model.read_text(encoding="utf-8")), pipeline)
    netspec = work / "netspec.composed.json"
    netspec.write_text(json.dumps(composed, indent=2), encoding="utf-8")
    return netspec, model, pipeline, work / ".claude" / "domain-forge-memory.md"


def check_lane(lane: Lane, work: Path) -> Outcome:
    started = time.monotonic()
    if lane.driver == "golden":
        return Outcome(lane.id, "skip", "the golden lane's spec is the recorded campaign's; nothing is composed")
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        if lane.driver == "acceptance":
            generator = "orders (make_fixture.py)"
            netspec, model, pipeline, memory = build_l01(lane, work)
        else:
            from lanekit import datasets, fixture
            real = lane.raw["generator"] in datasets.GENERATORS
            gen = None if real else StandIn(lane)
            generator = lane.raw["generator"] if real else f"stand-in for {lane.raw['generator']}"
            fx = fixture.make_lane(lane, work, SKILLS, generator=gen)
            netspec, model, pipeline, memory = fx.netspec, fx.model_html, fx.pipeline, work / ".claude/domain-forge-memory.md"
    except Exception as exc:                      # a fixture that cannot be built fails the lane
        lines = str(exc).splitlines() or [repr(exc)]
        refusals = tuple(ln.strip() for ln in lines[1:] if "ERROR:" in ln)
        return Outcome(lane.id, "fail", f"fixture: {type(exc).__name__}: {lines[0][:400]}",
                       errors=refusals, seconds=time.monotonic() - started)
    code, markers, output = compile_markers(netspec, model, pipeline, memory, work)
    if code not in (0, 2) or (not markers and code != 0):
        return Outcome(lane.id, "fail", f"compile_spec.py exited {code} without markers: {output[-600:]}",
                       generator, seconds=time.monotonic() - started)
    errors, deferred, warns = classify(markers, lane, planned(SKILLS))
    budget = ((lane.expect.get("markers") or {}).get("forge") or {}).get("warn_max", 0)
    problems = list(errors)
    if budget is not None and len(warns) > budget:
        problems.append(f"{len(warns)} WARN marker(s) over the lane's forge budget {budget}")
    status = "fail" if problems else "pass"
    detail = "; ".join(problems[:3]) if problems else (
        "compiles" + (f" (B2 deferred to B14: {len(deferred)} planned name(s))" if deferred else ""))
    return Outcome(lane.id, status, detail, generator, tuple(deferred), tuple(warns), tuple(errors),
                   time.monotonic() - started)


# --------------------------------------------------------------------------- the CLI
def _one(argv: Sequence[str]) -> int:
    """--one <lane> --work <dir> --out <json>: one lane in this process (the workers)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    outcome = check_lane(load_lanes()[a.one], Path(a.work))
    Path(a.out).write_text(json.dumps(outcome.to_json(), indent=1) + "\n", encoding="utf-8")
    return 0


def run(ids: Sequence[str], work: Path, jobs: int) -> list[dict]:
    work.mkdir(parents=True, exist_ok=True)

    def worker(lane_id: str) -> dict:
        out, log = work / f"{lane_id}.result.json", work / f"{lane_id}.log"
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.run([sys.executable, __file__, "--one", lane_id, "--work", str(work / lane_id),
                            "--out", str(out)], stdout=fh, stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if not out.is_file():
            return {"lane": lane_id, "status": "fail", "detail": f"the worker crashed; see {log}"}
        return json.loads(out.read_text(encoding="utf-8"))

    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = list(pool.map(worker, ids))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--one" in argv:
        return _one(argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", default="all")
    ap.add_argument("--work", default=str(DEFAULT_WORK))
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    lanes = load_lanes()
    ids = list(lanes) if a.lanes == "all" else [i.strip() for i in a.lanes.split(",") if i.strip()]
    unknown = [i for i in ids if i not in lanes]
    if unknown:
        print(f"ERROR: unknown lane(s): {', '.join(unknown)}")
        return 2
    results = run(ids, Path(a.work).resolve(), a.jobs)
    for r in results:
        mark = {"pass": "OK", "skip": "OK", "fail": "ERROR"}[r["status"]]
        print(f"{mark}: {r['lane']} {r['status']} — {r['detail']}" + (f" [{r.get('generator')}]" if r.get("generator") else ""))
        for d in r.get("deferred") or []:
            print(f"    deferred to B14: {d[:200]}")
        for w in (r.get("warns") or []) if r["status"] == "fail" else []:
            print(f"    WARN {w[:200]}")
        for e in r.get("errors") or []:
            print(f"    {e[:300] if e.startswith('ERROR') else 'ERROR ' + e[:300]}")
    failed = [r["lane"] for r in results if r["status"] == "fail"]
    if a.json:
        Path(a.json).write_text(json.dumps({"schema": "vega/compose-check@1", "lanes": results}, indent=1) + "\n",
                                encoding="utf-8")
    print(("ERROR: " if failed else "OK: ") + f"{len(results) - len(failed)} of {len(results)} lane(s) "
          f"compose and compile" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
