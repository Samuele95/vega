#!/usr/bin/env python3
"""
make_graph_fixtures.py — the payloads of the page's graph tests (tests/graph-shape.test.js): real
network-view@2 / explanation@2 pairs of the inspection zoo's graph cases (BUILD-PLAN W11: "node
tests on I10's graph payloads"), copied here so the node tests need neither torch nor the zoo.

    python3 make_graph_fixtures.py                 # run the zoo on the cases below, write them here
    python3 make_graph_fixtures.py --from DIR      # copy them from a dump already written
                                                   # (test_inspect_zoo.py --dump DIR)
    python3 make_graph_fixtures.py --check         # every pair here validates its schema

Each case is `<slug>/{network.json, explain.json, case.json}` exactly as the zoo dumps it (the
canonical network body with full weights; the default target's explanation, `full` detail,
gradient × input); `index.json` lists them. The zoo's inputs are its own seeded, neutral ones
(x0 … x5, a nominal `zone`, a linking `key`, graphs `g0` …): a fixture belongs to no lane, and the
template lint scans this directory too.

The cases: every graph convolution at node level (gcn with its normalised self-loops, sage with its
root weight, gat with four heads then one), a graph family with no convolution (linear), a regression
reading, and graph level with each readout the zoo builds (gin with sum, gcn with mean).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
ZOO = REPO / "vega-forge" / "scripts" / "tests" / "test_inspect_zoo.py"
SCHEMAS = REPO / "vega-forge" / "references" / "schemas"

# the zoo's slugs (its case ids with "/" written "__")
CASES = (
    "gcn__graph-node__multiclass__inductive-full",
    "graphsage__graph-node__binary__inductive-full",
    "gat__graph-node__binary__inductive-full",
    "gcn__graph-node__regression__inductive-full",
    "linear__graph-node__binary__inductive-full",
    "gin__graph-graph__multiclass__graphs",
    "gcn__graph-graph__regression__graphs",
)
FILES = ("network.json", "explain.json", "case.json")


def dump(into: Path) -> None:
    """Runs the zoo on CASES with --dump into `into` (CPU, two threads)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": env_threads()}
    cmd = [sys.executable, str(ZOO), "--cases", ",".join(c.replace("__", "/") for c in CASES), "--dump", str(into), "-q"]
    done = subprocess.run(cmd, env=env, cwd=str(REPO))
    if done.returncode != 0:
        raise SystemExit(f"ERROR: the zoo exited {done.returncode} on the graph cases")


def env_threads() -> str:
    return os.environ.get("OMP_NUM_THREADS", "2")


def copy(source: Path) -> list[dict]:
    entries = []
    for slug in CASES:
        where = source / slug
        missing = [f for f in FILES if not (where / f).is_file()]
        if missing:
            raise SystemExit(f"ERROR: {where} lacks {', '.join(missing)}: dump the zoo's graph cases first")
        target = HERE / slug
        target.mkdir(exist_ok=True)
        for f in FILES:
            shutil.copyfile(where / f, target / f)
        entries.append(json.loads((where / "case.json").read_text(encoding="utf-8")))
    (HERE / "index.json").write_text(json.dumps({"schema": "vega/inspect-zoo@1", "source": "tests/fixtures/graph-shape/make_graph_fixtures.py",
                                                 "cases": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entries


def check() -> int:
    import jsonschema
    validators = {name: jsonschema.Draft202012Validator(json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8")))
                  for name in ("network-view-2", "explanation-2")}
    bad = 0
    for slug in CASES:
        for name, file in (("network-view-2", "network.json"), ("explanation-2", "explain.json")):
            doc = json.loads((HERE / slug / file).read_text(encoding="utf-8"))
            errors = list(validators[name].iter_errors(doc))
            if errors:
                bad += 1
                print(f"FAIL: {slug}/{file}: {errors[0].message}")
    print(("OK" if not bad else "FAIL") + f": {len(CASES)} graph case(s) checked")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="source", help="a zoo dump directory to copy from")
    ap.add_argument("--check", action="store_true", help="validate the fixtures here")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.source:
        copy(Path(args.source))
    else:
        with tempfile.TemporaryDirectory(prefix="graph-fixtures-") as tmp:
            dump(Path(tmp))
            copy(Path(tmp))
    print(f"OK: {len(CASES)} graph case(s) written to {HERE}")
    return check()


if __name__ == "__main__":
    sys.exit(main())
