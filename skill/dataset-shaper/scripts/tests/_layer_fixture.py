"""
_layer_fixture — a real shaped run and a minimal model to write the `shape` layer onto, shared
by `test_shape_layer_digest.py` and `run_negatives_layer.py`.

The run is the executor's own output on the `graph-keypair` fixture recipe (`shape.py`, then
`verify_shape.py --json` and `shape.py --check`, the determinism verdict recorded beside the
verifier's), so `manifest.json` is the file as the executor wrote it — the bytes the shape
layer's `manifest_digest` is the sha256 of (shape-contract §4). The model is the smallest
document domain-forge's platform writer accepts: a page carrying a canonical Turtle block.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
RECIPE = SKILLS / "dataset-shaper" / "fixtures" / "graph-keypair.recipe.json"
APPLY = SCRIPTS / "apply_shape_layer.py"
BASE_MODEL = ('<!doctype html>\n<html><head><meta charset="utf-8"><title>fixture</title></head>'
              '<body>\n<script id="domain-model" type="text/turtle">\n'
              '@prefix ex: <http://example.org/fixture#> .\nex:Node a ex:Entity .\n'
              '</script>\n</body></html>\n')
_LAYER_DATA = re.compile(r'<script\s+id="layer-shape-data"[^>]*>([\s\S]*?)</script>')


def run(cmd, cwd=None) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True,
                          cwd=str(cwd or SKILLS), timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def file_digest(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(work) -> tuple[Path, Path]:
    """(model.html, the run directory): a shaped, verified, reproduced run and a base model."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / "shaped"
    for cmd in (("shape.py", "--recipe", RECIPE, "--out-dir", out, "--format", "parquet",
                 "--seed", "7"),
                ("verify_shape.py", "--recipe", RECIPE, "--out-dir", out,
                 "--json", out / "verification.json"),
                ("shape.py", "--check", "--out-dir", out)):
        rc, said = run([SCRIPTS / cmd[0], *cmd[1:]])
        if rc != 0:
            raise RuntimeError(f"{cmd[0]} exit {rc}:\n{said[-2000:]}")
    report = json.loads((out / "verification.json").read_text(encoding="utf-8"))
    if report.get("determinism") != "pass":   # `shape.py --check` records its own verdict (F45)
        raise RuntimeError(f"shape.py --check recorded determinism {report.get('determinism')!r}")
    model = work / "model.html"
    model.write_text(BASE_MODEL, encoding="utf-8")
    return model, out


def write_doc(path, doc) -> Path:
    path = Path(path)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def judgement(**extra) -> dict:
    """The author's half of the layer document: the schema and the abstract."""
    return {"schema": "dataset-shaper/shape@1",
            "readings": {"abstract": "the graph-keypair fixture shaped for the digest tests"},
            **extra}


def inline(run_dir, **extra) -> dict:
    """A layer document carrying the executor's half inline (no `from_run`)."""
    run_dir = Path(run_dir)
    read = lambda name: json.loads((run_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    return judgement(recipe=read("recipe.json"), manifest=read("manifest.json"),
                     verification=read("verification.json"),
                     before_after={"rows": [None, None], "columns": [None, None]}, **extra)


def apply(model, data, out) -> tuple[int, str]:
    return run([APPLY, model, "--data", data, "--out", out])


def layer_data(html_path) -> dict:
    m = _LAYER_DATA.search(Path(html_path).read_text(encoding="utf-8"))
    if not m:
        raise AssertionError(f"{html_path} carries no layer-shape-data")
    return json.loads(m.group(1))
