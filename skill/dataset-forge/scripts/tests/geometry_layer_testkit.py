"""
geometry_layer_testkit.py — what the X2 tests share: the paths of the tree they run in, the
check register, and the calls into the scripts (each in a subprocess, the way the skill runs
them). Imported by `test_geometry_layer.py` and `geometry_layer_consumers.py`; not a test.
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
SKILL = SCRIPTS.parent
ROOT = SKILL.parent
FIXTURE = SKILL / "fixtures" / "orders.geometry-layer.json"
ORDERS = SKILL / "fixtures" / "orders.csv"
GEOMETRY = SCRIPTS / "geometry.py"
APPLY = SCRIPTS / "apply_geometry_layer.py"
SMOKE = SCRIPTS / "smoke_geometry.py"
RENDER = SKILL / "assets" / "geometry-render.js"
STYLE = SKILL / "assets" / "geometry-layer.css"
BOOTSTRAP = ROOT / "data-lens" / "scripts" / "bootstrap_base.py"
VALIDATE = ROOT / "domain-forge" / "scripts" / "validate_model.py"
ENV_CHECK = ROOT / "vega" / "scripts" / "env_check.py"
KNOWN_FLAKE = ROOT / "vega" / "scripts" / "tests" / "known_flake.py"
SENSORS = ROOT / "data-lens" / "fixtures" / "sensors.geometry-layer.json"
SHAPE = ROOT / "dataset-shaper" / "scripts" / "shape.py"
RECIPE = ROOT / "dataset-shaper" / "fixtures" / "orders.recipe.json"
STAMP = "2026-01-01T00:00:00Z"
VERBOSE = "-v" in sys.argv

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and
                                                        (VERBOSE or not cond) else ""))
    return bool(cond)


def run(argv, timeout=600):
    """(exit code, stdout + stderr) of a Python script run from the skills root."""
    proc = subprocess.run([sys.executable, *map(str, argv)], capture_output=True, text=True,
                          cwd=str(ROOT), timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr


def apply_layer(model, data, out, extra=()):
    """apply_geometry_layer.py with the fixed --produced-at stamp."""
    return run([APPLY, model, "--data", data, "--out", out, "--produced-at", STAMP, *extra])


def validate_model(model):
    """validate_model.py, through the known-flake wrapper when the tree carries it (the
    domain-forge profile race is the one signature it may retry, once, and it says so)."""
    if KNOWN_FLAKE.is_file():
        return run([KNOWN_FLAKE, "run", "--stage", "X2-geometry", "--", sys.executable, VALIDATE, model])
    return run([VALIDATE, model])


def tail(out, n=4):
    return " | ".join(out.strip().splitlines()[-n:])


def errors(log):
    return " | ".join(ln for ln in log.splitlines() if ln.startswith("ERROR"))[:400]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def layer_doc(html):
    m = re.search(r'<script id="layer-geometry-data"[^>]*>([\s\S]*?)</script>', html)
    return json.loads(m.group(1)) if m else None
