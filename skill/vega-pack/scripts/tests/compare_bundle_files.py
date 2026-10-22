#!/usr/bin/env python3
"""
compare_bundle_files.py — the bundle this tree packs, against the bundle a reference commit's
`pack.py` packs from the same inputs.

    python3 compare_bundle_files.py <work-dir> [--ref c849f8b] [--variants serving,serve,full,full-serve]
                                    [--allow story.json --allow report/] [--keep DIR] [--strict]

The work dir is a pack acceptance directory (the orders campaign, its shaped parts, the
blueprint run and the evaluated model page, as `acceptance.sh` leaves them). For each variant —
the default serving slice, `--serve`, `--include full`, and both — the reference commit's
`pack.py` (with its `vega_pack_core.py` and `catalog.json`, read out of git, and the rest of the
skills tree as it is) and this tree's `pack.py` build a bundle each, and the two are compared:

  * **the file set** must be equal, except the paths `--allow` names (by default the two things
    this version deliberately stopped shipping: `story.json` and `report/`) — a difference here
    fails the run;
  * **`MANIFEST.entrypoints`** must be the reference's list, content and order, less the
    `--allow`ed paths (amendment A1: the CLIs, then every served file) — a difference fails too;
  * **the bytes** of every file present in both are compared too and each difference is listed;
    `MANIFEST.json` (its creation time, and the records this version adds) and `markers.json`
    (the run's account, in this version's order) always differ and are not reported. `--strict`
    makes a byte difference fail as well.

Exit 0 when every variant's file set agrees.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
SKILL = SCRIPTS.parent
SKILLS = SKILL.parent
CAMP = "vega-runs/2026-09-05T11-03Z"
PIPELINE = "blueprint-runs/2026-09-05T10-12Z/pipeline.json"
MODEL = "orders.domain.analysis.shaped.vega.train.eval.html"
STANCE = "the calibrated score is what the downstream dispatcher needs"
VARIANTS = {"serving": [], "serve": ["--serve"], "full": ["--include", "full"],
            "full-serve": ["--include", "full", "--serve"]}
ALWAYS_DIFFER = ("MANIFEST.json", "markers.json")
REFERENCE_FILES = ("pack.py", "vega_pack_core.py", "catalog.json")


def reference_tree(ref, into) -> Path:
    """A skills tree whose vega-pack scripts are the reference commit's and whose other skills
    are this tree's: the reference pack.py resolves them through its own parent directories."""
    root = Path(into) / "skills"
    scripts = root / "vega-pack" / "scripts"
    scripts.mkdir(parents=True)
    for name in REFERENCE_FILES:
        blob = subprocess.run(["git", "-C", str(SKILLS), "show", f"{ref}:vega-pack/scripts/{name}"],
                              capture_output=True, check=True).stdout
        (scripts / name).write_bytes(blob)
    for entry in SKILLS.iterdir():
        if entry.name not in ("vega-pack", ".git"):
            (root / entry.name).symlink_to(entry.resolve())
    return scripts / "pack.py"


def pack(pack_py, work, out_dir, extra) -> Path:
    cmd = [sys.executable, str(pack_py), "--campaign", CAMP, "--shaped", "shaped",
           "--bundle-dir", str(out_dir), "--pipeline", PIPELINE, "--model", MODEL,
           "--ship-anyway", STANCE, *extra]
    proc = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f"ERROR: {pack_py} exited {proc.returncode}:\n"
                         + (proc.stdout + proc.stderr)[-2000:])
    return next(d for d in sorted(Path(out_dir).iterdir()) if (d / "MANIFEST.json").is_file())


def files_of(root) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def allowed(rel, allow) -> bool:
    return any(rel == a or (a.endswith("/") and rel.startswith(a)) for a in allow)


def entrypoints_of(root) -> list:
    return json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))["entrypoints"]


def compare(reference, bundle, allow) -> tuple:
    """(file-set and entry-point differences, byte differences) between two bundle
    directories."""
    ref, new = files_of(reference), files_of(bundle)
    only_ref = sorted(r for r in set(ref) - set(new) if not allowed(r, allow))
    only_new = sorted(r for r in set(new) - set(ref) if not allowed(r, allow))
    set_diffs = [f"only in the reference: {r}" for r in only_ref] + \
        [f"only in this tree's bundle: {r}" for r in only_new]
    ref_points = [r for r in entrypoints_of(reference) if not allowed(r, allow)]
    new_points = entrypoints_of(bundle)
    if new_points != ref_points:
        set_diffs.append(f"MANIFEST.entrypoints {new_points}, the reference lists {ref_points}")
    byte_diffs = sorted(r for r in set(ref) & set(new)
                        if r not in ALWAYS_DIFFER and ref[r] != new[r])
    return set_diffs, byte_diffs


def report(variant, against, n, diffs) -> None:
    ref, allow = against
    set_diffs, byte_diffs = diffs
    print(f"{'ok  ' if not set_diffs else 'FAIL'}  {variant}: {n} file(s); the file set and "
          f"the entry points {'agree' if not set_diffs else 'differ'} with {ref}'s"
          + (f" (allowed: {', '.join(allow)})" if allow else ""))
    for line in set_diffs:
        print(f"        {line}")
    for rel in byte_diffs:
        print(f"        bytes differ: {rel}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", help="the pack acceptance work directory")
    ap.add_argument("--ref", default="c849f8b", help="the reference commit (default c849f8b)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--allow", action="append", default=None,
                    help="a path (or a directory ending in /) the two bundles may disagree on "
                         "(default: story.json and report/)")
    ap.add_argument("--strict", action="store_true", help="a byte difference fails as well")
    ap.add_argument("--keep", default=None, help="build the bundles here and keep them")
    a = ap.parse_args(argv)
    allow = a.allow if a.allow is not None else ["story.json", "report/"]
    work = Path(a.work).resolve()
    failed = 0
    with tempfile.TemporaryDirectory() as td:
        base = Path(a.keep).resolve() if a.keep else Path(td)
        ref_pack = reference_tree(a.ref, Path(td))
        for variant in [v.strip() for v in a.variants.split(",") if v.strip()]:
            reference = pack(ref_pack, work, base / f"ref-{variant}", VARIANTS[variant])
            bundle = pack(SCRIPTS / "pack.py", work, base / f"new-{variant}", VARIANTS[variant])
            set_diffs, byte_diffs = compare(reference, bundle, allow)
            failed += bool(set_diffs or (a.strict and byte_diffs))
            report(variant, (a.ref, allow), len(files_of(bundle)), (set_diffs, byte_diffs))
    print()
    if failed:
        print(f"{failed} variant(s) disagree with {a.ref}")
        return 1
    print(f"OK: every variant's bundle file set and MANIFEST.entrypoints equal {a.ref}'s, "
          f"except {', '.join(allow)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
