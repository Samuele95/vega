#!/usr/bin/env python3
"""
exit_abort_probe.py — the reproducer of the exit-time abort (PLAN-AMENDMENTS A40).

    python3 exit_abort_probe.py PARQUET [--runs 400] [--parallel 24] [--json OUT]
    python3 exit_abort_probe.py --sites [--root DIR] [--json OUT]

A stage that printed all its OK lines sometimes died with `terminate called without an
active exception` (exit −6, SIGABRT) under load. The cause, reproduced here: pandas (3.x) reads
a local parquet path by opening a *Python* file object and handing it to Arrow; Arrow's worker
threads read through that object, so they take the GIL; when one of them still touches Python
while the interpreter finalizes, CPython ends the thread with a forced unwind, the unwind meets
a `noexcept` frame of Arrow's, and the C++ runtime calls `std::terminate` — after the work is
done. Arrow's own filesystem (`filesystem=pyarrow.fs.LocalFileSystem()`, or
`pyarrow.parquet.read_table(<path>)`) keeps Python out of Arrow's threads, and the abort does
not occur.

The probe starts `--runs` short child processes, `--parallel` at a time, for each way of
reading — `pandas` (pandas' default: a Python file object), `handle` (Arrow given a Python file
object), `native` (pandas with Arrow's LocalFileSystem) and `arrow` (read_table on the path) —
and counts the children that abort. The abort needs load (it was measured at load 50–75 on 20
cores: 5 % of `pandas` children, 13 % of `handle`, none of `native` or `arrow`). Exit 0 when the
native ways never abort; the counts are the evidence, the verdict does not depend on whether
the load was high enough to make the default way abort this time.

`--sites` is the other half of the acceptance: it reads the package's own sources (the skill
directories under --root, default the repository this file sits in; an external skill, linked
or copied, is not read) and lists every pandas `read_parquet(` call — in code, or in the code
a script emits (a notebook cell, a reproduction script) — whose arguments do not pass
`filesystem=`: the pandas default way, the one that aborts. Pytest modules (`test_*.py`, `conftest.py`) and their
`*_fixtures.py` helpers are not production and are left out; the shipped bundle tests (the
serving templates' `tests/`) are read. Exit 0 when no such call is left.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

READERS = {
    "pandas": "import pandas as pd; pd.read_parquet(sys.argv[1])",
    "handle": ("import pyarrow.parquet as pq\nwith open(sys.argv[1], 'rb') as fh:\n"
               "    pq.read_table(fh)"),
    "native": ("import pandas as pd, pyarrow.fs as fs; "
               "pd.read_parquet(sys.argv[1], filesystem=fs.LocalFileSystem())"),
    "arrow": "import pyarrow.parquet as pq; pq.read_table(sys.argv[1]).to_pandas()",
}
NATIVE = ("native", "arrow")
SIGNATURE = "terminate called without an active exception"
HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
# a pandas read_parquet call: pd.read_parquet(, pandas.read_parquet(, __import__("pandas").read_parquet(
PANDAS_READ = re.compile(r"(?:\bpd|\bpandas|\(\s*[\"']pandas[\"']\s*\))\.read_parquet\(")
# external skills: linked into a working tree, copied into a matrix's skills tree (D6 §2.3)
EXTERNAL = frozenset({"domain-forge"})
NOT_PRODUCTION = re.compile(r"^(?:test_.*|conftest|.*_fixtures)\.py$")


def _arguments(text: str, start: int) -> str:
    """The text between the parenthesis that opens at `start` and the one that closes it."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1:i]
    return text[start + 1:]


def call_sites(text: str) -> list[tuple[int, bool]]:
    """(line, passes filesystem=) of every pandas read_parquet call in a source text."""
    out = []
    for m in PANDAS_READ.finditer(text):
        args = _arguments(text, m.end() - 1)
        out.append((text.count("\n", 0, m.start()) + 1, re.search(r"\bfilesystem\s*=", args) is not None))
    return out


def production_files(root: Path) -> list[Path]:
    """The package's own Python sources under `root`'s skill directories (a linked directory is an
    external skill, not read), less pytest modules, their fixtures and this probe; every file
    under a `templates` directory is read, a shipped bundle's tests among them."""
    files = []
    for skill in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                        and not p.name.startswith(".") and p.name not in EXTERNAL):
        for path in sorted(skill.rglob("*.py")):
            shipped = "templates" in path.parts      # a template is shipped, its tests too
            if (NOT_PRODUCTION.match(path.name) and not shipped) or "__pycache__" in path.parts:
                continue
            if path.resolve() != HERE:
                files.append(path)
    return files


def default_reads(root: Path) -> list[str]:
    """`<file>:<line>` of every production pandas read_parquet without `filesystem=`."""
    found = []
    for path in production_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        found += [f"{path.relative_to(root)}:{line}" for line, native in call_sites(text) if not native]
    return found


def _child(reader: str, parquet: str) -> tuple[int, bool]:
    code = "import sys\n" + READERS[reader] + "\nprint('OK')"
    proc = subprocess.run([sys.executable, "-c", code, parquet], capture_output=True, text=True,
                          timeout=300)
    return proc.returncode, SIGNATURE in proc.stderr


def probe(parquet: str, runs: int, parallel: int) -> dict:
    out = {}
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for reader in READERS:
            results = list(pool.map(lambda _: _child(reader, parquet), range(runs)))
            out[reader] = {"runs": runs,
                           "aborted": sum(1 for code, _ in results if code in (-6, 134)),
                           "signature": sum(1 for _, seen in results if seen),
                           "other_failures": sum(1 for code, _ in results
                                                 if code not in (0, -6, 134))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("parquet", type=Path, nargs="?")
    ap.add_argument("--sites", action="store_true",
                    help="list the production read_parquet calls that do not pass filesystem=")
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--runs", type=int, default=400)
    ap.add_argument("--parallel", type=int, default=24)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.sites:
        return _sites(args.root, args.json)
    if args.parquet is None:
        ap.error("PARQUET is required without --sites")
    counts = probe(str(args.parquet), args.runs, args.parallel)
    for reader, c in counts.items():
        print(f"{reader:7s} {c['aborted']:4d} of {c['runs']} aborted at exit "
              f"({c['signature']} with the signature, {c['other_failures']} other failures)")
    if args.json:
        Path(args.json).write_text(json.dumps({"schema": "vega/exit-abort-probe@1",
                                               "readers": counts}, indent=2) + "\n")
    clean = all(counts[r]["aborted"] == 0 and counts[r]["other_failures"] == 0 for r in NATIVE)
    print(("OK: " if clean else "ERROR: ") + "the native readers never abort at exit")
    return 0 if clean else 1


def _sites(root: Path, out) -> int:
    found = default_reads(root)
    for site in found:
        print(f"{site}: pandas read_parquet without filesystem= (a Python file object, A40)")
    if out:
        Path(out).write_text(json.dumps({"schema": "vega/exit-abort-sites@1", "root": str(root),
                                         "files": len(production_files(root)),
                                         "default_reads": found}, indent=2) + "\n")
    print((f"ERROR: {len(found)} production read_parquet call(s) without filesystem="
           if found else "OK: every production read_parquet passes Arrow's own filesystem"))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
