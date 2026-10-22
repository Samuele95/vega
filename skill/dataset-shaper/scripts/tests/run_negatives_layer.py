#!/usr/bin/env python3
"""
run_negatives_layer.py — the `shape` layer's apply script must refuse a layer whose
`manifest_digest` is missing, malformed or not the manifest file's (shape-contract §4).

    python3 run_negatives_layer.py [<work-dir>] [-v] [--json <path>]

`test_shape_layer_digest.py` proves what the apply script writes. This script seeds one defect
at a time into the layer document and asserts that `apply_shape_layer.py` refuses it (exit 2)
with a message naming the digest, and writes no output file.

| id    | seed                                                                  | expect                         |
|-------|-----------------------------------------------------------------------|--------------------------------|
| SM1   | the executor's half inline, no `from_run`, no `manifest_digest`       | exit 2: manifest_digest        |
| SM1b  | inline, `manifest_digest` not `sha256:<64 hex>`                       | exit 2: sha256:<64 hex>        |
| SM2   | `from_run`, with the digest of another file typed as `manifest_digest`| exit 2: is not the digest of   |
| SM2b  | `from_run`, the manifest file edited after its digest was taken       | exit 2: is not the digest of   |

Exit 0 when every seeded defect was refused.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _layer_fixture as fx  # noqa: E402

sys.path.insert(0, str(fx.SKILLS / "vega" / "scripts" / "tests" / "negatives"))
import negkit  # noqa: E402


def refused(cases, name, work, doc, contains):
    data = fx.write_doc(work / f"{negkit.case_id(name)}.json", doc)
    out = work / f"{negkit.case_id(name)}.html"
    rc, said = fx.apply(work / "model.html", data, out)
    if cases.expect(name, rc, said, code=2, contains=contains) and out.exists():
        cases.check(name + " (no output written)", False, f"{out} was written", said)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("work", nargs="?", default=None)
    ap.add_argument("-v", dest="verbose", action="store_true")
    ap.add_argument("--json", default=None, metavar="PATH")
    a = ap.parse_args()
    work = (Path(a.work).resolve() / "neg-layer") if a.work else \
        Path(tempfile.mkdtemp(prefix="shape-layer-neg-"))
    shutil.rmtree(work, ignore_errors=True)
    model, run_dir = fx.build(work)
    print(f"run_negatives_layer: work in {work}")
    cases = negkit.Cases(__file__, work, verbose=a.verbose)
    written = fx.file_digest(run_dir / "manifest.json")

    cases.section("apply_shape_layer.py — the layer carries the manifest file's digest")
    refused(cases, "SM1  an inline layer with no manifest_digest is refused", work,
            fx.inline(run_dir), ("manifest_digest",))
    refused(cases, "SM1b  a manifest_digest that is not sha256:<64 hex> is refused", work,
            fx.inline(run_dir, manifest_digest=written[:-1]), ("manifest_digest", "sha256:<64 hex>"))
    refused(cases, "SM2  a from_run layer typing another file's digest is refused", work,
            fx.judgement(from_run=str(run_dir),
                         manifest_digest=fx.file_digest(run_dir / "recipe.json")),
            ("manifest_digest", "is not the digest of", "manifest.json"))

    edited = work / "edited"
    shutil.copytree(run_dir, edited)
    manifest = json.loads((edited / "manifest.json").read_text(encoding="utf-8"))
    manifest["seed"] = (manifest.get("seed") or 0) + 1
    (edited / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    refused(cases, "SM2b  a manifest edited after its digest was taken is refused", work,
            fx.judgement(from_run=str(edited), manifest_digest=written),
            ("manifest_digest", "is not the digest of"))
    assert model.is_file()
    return cases.finish(a.json)


if __name__ == "__main__":
    sys.exit(main())
