#!/usr/bin/env python3
"""
data_fixtures.py — what the DataSource tests (`test_windows.py`, `test_graphs.py`,
`test_data_rows.py`) share: the ingestion templates rendered into an importable package, and
shaped datasets written the way the shaper writes them.

    ingestion(package, shape, only)   render domain.py + the named ingestion templates of the
                                      v2.3 set (`render_templates.render`, partial mode) into a
                                      temporary package and import it; returns a namespace with
                                      the modules (`domain`, `data`, `frames`, `windows`,
                                      `graphs`) and the package root
    write_shaped(dir, parts, raw, ...) the files the DataSource reads: one parquet per part,
                                      `fitted.json` with `__split_raw__` (each part's raw-file
                                      positions, in part order), the edges part when given, and a
                                      `manifest.json` recording each file's path and sha256 —
                                      the subset of the shaper's outputs `frames.py` reads
    spec_on(spec, manifest)           a netspec pointed at a shaped manifest: input.manifest and
                                      its digest, data_contract.parts, sample.edges
    seal(dir, token)                  a `test_seal.json` as /vega-eval writes it
    run_shaper(dir, csv, recipe)      the real `dataset-shaper/scripts/shape.py` over a raw csv and
                                      a recipe (steps only; the envelope is filled here)

Not a test module (no `test_` functions); imported by the three test modules.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
TEMPLATES = SCRIPTS / "backends" / "pytorch" / "templates"
SKILLS = SCRIPTS.parent.parent
SHAPER = SKILLS / "dataset-shaper" / "scripts" / "shape.py"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))
import render_templates as rt  # noqa: E402

INGESTION = ("frames.py", "windows.py", "graphs.py", "data.py")
_KEEP = []          # the temporary directories of the rendered packages live as long as the run


def sha256(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


# the support file a sample shape's ingestion code lives in (row's is frames.py, always here)
SHAPE_FILES = {"window": "windows.py", "graph": "graphs.py"}


def ingestion(package: str, shape: str = "row", only=INGESTION):
    """The ingestion templates `only` (and domain.py) rendered as package `package`, imported.
    The package carries row and the shapes whose support file `only` names — as the emitter,
    which keeps a shape's code where its files are emitted (`emit.select_shapes`)."""
    work = tempfile.TemporaryDirectory(prefix=f"{package}-")
    _KEEP.append(work)
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = package
    shapes = ("row",) + tuple(s for s, f in SHAPE_FILES.items() if f in only)
    rec = rt.render(TEMPLATES, Path(work.name), pipeline, rt.reference_spec(shape),
                    only=["domain.py", *only], shapes=shapes)
    sys.path.insert(0, work.name)
    importlib.invalidate_caches()
    ns = types.SimpleNamespace(root=Path(rec["root"]), out=Path(work.name), package=package,
                               pipeline=pipeline, blueprint_map=rec["blueprint_map"])
    ns.domain = importlib.import_module(f"{package}.domain")
    for name in only:
        stem = name.removesuffix(".py")
        setattr(ns, stem, importlib.import_module(f"{package}.ingestion.{stem}"))
    return ns


def write_shaped(root, parts: dict, raw: dict, edges=None, slug: str = "fx") -> Path:
    """Write `parts` ({"train": frame, ...}), `fitted.json.__split_raw__` from `raw` and the
    optional edges frame into `root`; return the manifest path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    named = dict(parts)
    if edges is not None:
        named["edges"] = edges
    for name, frame in named.items():
        path = root / f"{slug}.{name}.parquet"
        frame.reset_index(drop=True).to_parquet(path, index=False)
        files[name] = {"path": str(path), "digest": sha256(path), "rows": int(len(frame)),
                       "columns": int(frame.shape[1])}
    fitted = {"__split_raw__": {k: [int(v) for v in vals] for k, vals in raw.items()},
              "__split__": {k: list(range(len(vals))) for k, vals in raw.items()}}
    (root / "fitted.json").write_text(json.dumps(fitted), encoding="utf-8")
    manifest = {"schema": "dataset-shaper/manifest@1",
                "outputs": {"dir": str(root), "format": "parquet", "files": files,
                            "manifest": "manifest.json", "fitted": "fitted.json"}}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def spec_on(spec: dict, manifest) -> dict:
    """`spec` reading the shaped dataset whose manifest is `manifest`."""
    spec = copy.deepcopy(spec)
    manifest = Path(manifest)
    files = json.loads(manifest.read_text(encoding="utf-8"))["outputs"]["files"]
    spec["input"]["manifest"] = str(manifest)
    spec["input"]["manifest_digest"] = sha256(manifest)
    dc = spec["data_contract"]
    dc["parts"] = {"train": files["train"]["path"], "valid": files["valid"]["path"],
                   "test": "sealed"}
    if "edges" in files:
        dc["sample"]["edges"] = files["edges"]["path"]
    return spec


def seal(directory, token: str = "sealed-token") -> Path:
    path = Path(directory) / "test_seal.json"
    path.write_text(json.dumps({"token": token, "opened": False}), encoding="utf-8")
    return path


def numeric(name: str) -> dict:
    return {"name": name, "dtype": "float32", "role": "numeric", "origin": "S1"}


def nominal(name: str, cardinality: int) -> dict:
    return {"name": name, "dtype": "int64", "role": "nominal", "cardinality": cardinality,
            "unseen_category_id": cardinality - 1, "origin": "S2"}


def recipe(csv: Path, steps: list, target: str) -> dict:
    """A recipe envelope around `steps` for the raw file `csv`."""
    import pandas as pd
    frame = pd.read_csv(csv)
    return {"schema": "dataset-shaper/recipe@1",
            "input": {"path": str(csv), "digest": sha256(csv), "rows": int(len(frame)),
                      "columns": int(frame.shape[1]), "layers_read": []},
            "seed": 7, "goal": "custom", "target": target,
            "outputs": {"dir": "shaped/", "format": "parquet", "files": [],
                        "recipe": "shaped/recipe.json", "manifest": "shaped/manifest.json",
                        "lineage": "shaped/lineage.json", "reproduce": "shaped/reproduce.py"},
            "steps": steps}


def step(sid: str, op: str, columns, params: dict, source: str) -> dict:
    return {"id": sid, "op": op, "columns": list(columns), "params": params, "source": source,
            "rationale": f"{op} for the DataSource tests", "alternatives": [],
            "consequences": {"rows": "none", "columns": "none", "downstream": {}},
            "reversible": op not in ("drop_identity", "dedupe")}


def run_shaper(work, csv: Path, recipe_doc: dict) -> Path:
    """shape.py over the recipe; the shaped directory's manifest path."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    recipe_path = work / "recipe.in.json"
    recipe_path.write_text(json.dumps(recipe_doc, indent=1), encoding="utf-8")
    out = work / "shaped"
    proc = subprocess.run([sys.executable, str(SHAPER), "--recipe", str(recipe_path),
                           "--out-dir", str(out), "--format", "parquet"],
                          capture_output=True, text=True, cwd=str(SKILLS), timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"shape.py exited {proc.returncode}: "
                           + " | ".join((proc.stdout + proc.stderr).strip().splitlines()[-6:]))
    return out / "manifest.json"


def raises(fn, kind, *needles) -> str:
    """`fn()` raises `kind` (by class name, so a rendered module's exception matches) with every
    needle in its message; returns the message."""
    try:
        fn()
    except Exception as exc:                             # noqa: BLE001 — checked by name below
        if type(exc).__name__ != kind:
            raise AssertionError(f"expected {kind}, got {type(exc).__name__}: {exc}") from exc
        msg = str(exc)
        missing = [n for n in needles if n not in msg]
        assert not missing, f"{kind} raised without {missing}: {msg}"
        return msg
    raise AssertionError(f"expected {kind}; nothing was raised")


def main_for(namespace: dict, argv, label: str) -> int:
    """The runner the three test modules share: every `test_*` in `namespace`, `-k` filter."""
    pick = argv[argv.index("-k") + 1] if "-k" in argv else ""
    verbose = "-v" in argv
    tests = [(n, f) for n, f in sorted(namespace.items())
             if n.startswith("test_") and callable(f) and pick in n]
    failed = []
    for name, fn in tests:
        try:
            fn()
            if verbose:
                print(f"  ok    {name}")
        except Exception as exc:                         # a failing test is reported, not raised
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"\n{len(failed)} of {len(tests)} {label} test(s) failed")
        return 1
    print(f"\n{len(tests)} {label} test(s) passed")
    return 0
