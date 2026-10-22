#!/usr/bin/env python3
"""
test_code_room.py — the code room's walk on the code the backend renders (BUILD-PLAN W13; D4 §6.3,
R-D7, R-E8).

    python3 -m pytest -q test_code_room.py          # needs node (the page's modules)

The page resolves page.json's walk against the files `GET v1/code` sends (views/code.js
resolveWalk). Here the files are the ones a bundle of each sample shape carries, rendered from the
live templates: the forge's package in full (`render_templates.py --full` on the reference row,
window, node-level graph and graph-level networks), the serving package layered into its Predictor
module (`--into Predictor --recursive`), the bundle's composition root (`serving/bundle/predict.py`
as `predict.py`) and the service's route modules (`serving/serve/*.py` as `serve/*`), each with its
role and kind as vega/code-view@2 names them; the walk is `page_config.build_walk` on the same
network with the backend's explain catalog. For every shape:

- the page (node, the room's own modules) and an independent resolution written in Python
  (tests/fixtures/code-view/make_code_views.py `resolve`) give the same steps — the same file, the
  same first and last line — and the same steps left out, for the same reason;
- no step is lost to its anchor: a step whose file is served is found in it (a step whose file is
  not rendered yet — the inspector, the HTTP routes, until their packages land — drops out, and the
  lead says so);
- the steps R-E8 anchors in the forge's own files resolve: root, store, build, one per block type,
  forward, link, shape, run.

It also holds the committed fixtures of the page's unit tests (tests/code.test.js) together: each
fixture's expected walk is the Python resolution of its own walk and files, its files' roles and
kinds are the vega-pack catalog's, its notes are story-2's.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
REPO = TESTS.parents[2]
PAGE = SCRIPTS.parent / "assets" / "page"
FIXTURES = PAGE / "tests" / "fixtures" / "code-view"
FORGE_TESTS = REPO / "vega-forge" / "scripts" / "tests"
TEMPLATES = REPO / "vega-forge" / "scripts" / "backends" / "pytorch" / "templates"
EXPLAIN = REPO / "vega-forge" / "scripts" / "backends" / "pytorch" / "explain-catalog.json"
PACK_CATALOG = REPO / "vega-pack" / "scripts" / "catalog.json"
SHAPES = ("row", "window", "graph", "graph-level")
ANCHORED = ("root", "store", "build", "forward", "link", "shape", "run")


def _module(name: str, path: Path):
    """A module loaded from its file under a name of its own, leaving no bytecode beside it (the
    fixtures directory is part of the page template's tree)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    kept, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = kept
    return mod


MAKE = _module("code_room_fixtures", FIXTURES / "make_code_views.py")
for p in (str(SCRIPTS), str(FORGE_TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
import render_templates as rt  # noqa: E402
from page_config import build_walk  # noqa: E402

RESOLVE_JS = """
  import { readFileSync } from "node:fs";
  const { codeView, resolveWalk } = await import("./js/views/code.js");
  const python = (await import("./js/views/highlight/python.js")).default;
  const cases = JSON.parse(readFileSync(process.argv[1], "utf8")), out = {};
  for (const [name, c] of Object.entries(cases)) {
    const view = codeView(c.code), walk = resolveWalk(c.walk, view.files, python.extent);
    out[name] = { steps: walk.steps.map((s) => ({ id: s.id, path: s.path, from: s.range[0] + 1, to: s.range[1] + 1 })),
                  missing: walk.missing.map((m) => ({ id: m.id, why: m.miss })) };
  }
  console.log(JSON.stringify(out));
"""


def node_json(script: str, data) -> dict:
    node = shutil.which("node")
    assert node, "node is required: the walk is resolved by the page's own modules"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        run = subprocess.run([node, "--input-type=module", "-e", script, str(path)], cwd=PAGE,
                             capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


# ---- a bundle's code, rendered ----------------------------------------------------------------------
def _role(rel: str, entries: dict, roles: list) -> tuple[str, str]:
    """(role, kind) of a file as vega/code-view@2 names them."""
    if rel == "predict.py":
        return "composition_root", "entrypoint"
    if rel.startswith("serve/"):
        return "http_entry", "entrypoint"
    name = rel.split("/", 1)[1]
    if name in ("ports.py", "domain.py"):
        return name[:-3], "support"
    role = (entries.get(name) or {}).get("role")
    return (role, "adapter") if role in roles else ("support", "support")


def rendered_code(shape: str, work: Path) -> tuple[dict, list]:
    """(the vega/code-view@2 answer of a bundle of that shape, the page.json walk of its network)."""
    spec, pipeline = rt.reference_spec(shape), rt.reference_pipeline()
    emit = rt.load_emit()
    plan = emit.plan(spec, pipeline)
    mapping = emit.common_mapping(spec, pipeline, plan)
    fill = lambda text: re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: str(mapping.get(m.group(1), m.group(1).lower())), text)
    serving = TEMPLATES / "serving" / "package"
    used = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", "".join(p.read_text(encoding="utf-8") for p in serving.rglob("*.py"))))
    rt.render(TEMPLATES, work, pipeline, spec, full=True)
    rt.render_partial(serving, work, pipeline, spec, rt.PartialOptions(
        into="Predictor", recursive=True, defines={k: k.lower() for k in used if k not in mapping}))
    package = work / plan["package"]
    entries = {e["file"]: e for e in json.loads((package / "blueprint-map.json").read_text(encoding="utf-8"))["entries"]
               if e.get("file")}
    texts = {f"{plan['package']}/{p.relative_to(package).as_posix()}": p.read_text(encoding="utf-8")
             for p in sorted(package.rglob("*.py")) if p.name != "__init__.py" and "__pycache__" not in p.parts}
    texts["predict.py"] = fill((TEMPLATES / "serving" / "bundle" / "predict.py").read_text(encoding="utf-8"))
    for p in sorted((TEMPLATES / "serving" / "serve").glob("*.py")):
        texts[f"serve/{p.name}"] = fill(p.read_text(encoding="utf-8"))
    roles = json.loads(PACK_CATALOG.read_text(encoding="utf-8"))["code_view"]["roles"]
    files = [MAKE.code_file(rel, *_role(rel, entries, roles), text) for rel, text in texts.items()]
    walk = build_walk(spec, json.loads(EXPLAIN.read_text(encoding="utf-8")))
    return MAKE.code_view(files), walk


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    cases = {}
    for shape in SHAPES:
        code, walk = rendered_code(shape, tmp_path_factory.mktemp(shape))
        cases[shape] = {"code": code, "walk": walk}
    return cases, node_json(RESOLVE_JS, cases)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_page_resolves_the_walk_as_the_python_rule_does_on_the_rendered_code(rendered, shape):
    cases, page = rendered
    case = cases[shape]
    assert page[shape] == MAKE.resolve(case["walk"], case["code"]["files"], "python")


@pytest.mark.parametrize("shape", SHAPES)
def test_no_step_is_lost_to_its_anchor_and_the_forge_anchors_resolve(rendered, shape):
    cases, page = rendered
    walk, got = cases[shape]["walk"], page[shape]
    lost = [m for m in got["missing"] if m["why"] != "file"]
    assert lost == [], f"{shape}: a served file does not hold its step's anchor: {lost}"
    found = {s["id"] for s in got["steps"]}
    blocks = [s["id"] for s in walk if s["id"] not in (*ANCHORED, "explain", "route")]
    assert blocks, shape
    assert [i for i in (*ANCHORED, *blocks) if i not in found] == [], shape
    paths = {s["id"]: s["path"] for s in got["steps"]}
    assert paths["root"] == "predict.py" and paths["run"].endswith("/predict.py") and paths["run"] != "predict.py"
    assert all(re.search(r"/model/blocks_(core|seq|graph|gen)\.py$", paths[b]) for b in blocks), paths


def test_the_committed_fixtures_hold_together():
    for name in MAKE.FIXTURES:
        doc = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        assert doc["schema"] == MAKE.SCHEMA and doc["name"] == name
        assert MAKE.problems(doc) == [], name
