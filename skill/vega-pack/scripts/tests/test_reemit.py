#!/usr/bin/env python3
"""
test_reemit.py — `pack.py --re-emit-serving` (build plan S12; bundle-contract §7.8; R-D2, R-D11).

    python3 -m pytest test_reemit.py [-q] [-k "not packed"]
    python3 test_reemit.py                             (without pytest)

Two halves.

**The logic, on manifests and directories built here** (fast): which bundle files the serving
templates write (the catalog's patterns at their destinations; a `never` pattern always wins);
`compare` names every pinned file and manifest key that moved and nothing that may move (the
catalog's free keys; `files`, `entrypoints` and the package map only in the outputs); `altered`
names what a bundle no longer holds as its manifest pins it; `prepare` refuses a missing or an
altered bundle and stages beside a verified one; `swap` puts the staged bundle in place and the
old one back when it cannot; what other stages wrote beside the bundle is carried over; a plain
pack records `reemits: []`.

**The real thing** (`packed`): B15's graph-level spike fixture trained, evaluated by the REAL
`evaluate.py` and packed by the REAL `pack.py` (`roundtrip_fixtures.py`), then re-emitted by
the REAL `pack.py --re-emit-serving` from the same inputs:

  * unchanged templates change only `created_at`, `blueprint.conformance` and `reemits` — every
    file byte-identical; `reemits[]` records the previous manifest's digest, the templates'
    digest, the reason and no changed file; a second re-emit appends to it; the gate line is
    printed and recorded; the card and the showcase beside the bundle are carried over, and the
    bundle's own slot now refuses that showcase as stale (R-D11); `predict.py --check` passes;
  * a changed template changes exactly its outputs — one template per destination (`package`,
    `package/inspection`, `serve`, `bundle`, `tests`) changed in a copy of the backend's serving
    templates: the files that differ are exactly theirs and `serve.templates` differs exactly at
    them; a `package` template added (a support module a structural fix split out) adds its
    output and its map entry and moves nothing else;
  * changed weights (the campaign's checkpoint saved again, the ledger updated as a re-trained
    run would be) are refused with exit 3 naming `weights.pt`, and the bundle is untouched;
  * a bundle that no longer verifies against its manifest is refused with exit 2 before
    anything is built.

`VEGA_REEMIT_WORK=<dir>` keeps the packed fixture there (default: a temporary directory).
"""
from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
for _p in (str(PACK), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reemit  # noqa: E402
import roundtrip_fixtures as rf  # noqa: E402
from bundle_slice import bundle_modules  # noqa: E402
from pack_context import PackExit  # noqa: E402
from vega_pack_core import CATALOG, BundlePaths, matching_templates  # noqa: E402

FIXTURE = "graph-level"
TIMEOUT = 1200
PROBE = "\n# re-emit probe: this line changes the template and nothing but what it renders\n"
PROBE_MODULE = ('"""{{PACKAGE}}.{{MODULE_SERVING}}.reemit_probe — a support module a template '
                'fix added (test_reemit)."""\n')
ADDED = "package/reemit_probe.py"
STALE = "the bundle was re-emitted after this showcase; re-run /vega-show"
# the only manifest keys an unchanged re-emit may move (the closing test's words)
UNCHANGED_MAY_MOVE = {"created_at", "blueprint.conformance", "reemits"}
ENTRY_KEYS = {"at", "reason", "previous_manifest_digest", "templates_digest", "changed"}
# the template override: the real pack.py, with the backend's serving templates read from a copy
OVERRIDE = """
import sys
from pathlib import Path
sys.path.insert(0, {scripts!r})
import pack_context
templates = Path(sys.argv[1])
pack_context.PackContext.templates = property(lambda self: templates)
import pack
sys.exit(pack.main(sys.argv[2:]))
"""

OUT = reemit.ServingOutputs(
    package="pkg", module="serving",
    adapters=("pkg/serving/predict.py", "pkg/monitoring/monitor.py"),
    root_names=(".dockerignore", "Dockerfile", "README.serve.md", "predict.py",
                "pyproject.toml"),
    exported=("exports",))


def _manifest(**changes) -> dict:
    """A bundle manifest in the v2.3 shape, small enough to reason about."""
    doc = {"schema": "vega/bundle@1", "slug": "s", "run": "R0", "package": "pkg",
           "chain": {"model": "m.html", "model_digest": "sha256:m"},
           "files": {"weights.pt": "sha256:w", "netspec.json": "sha256:n",
                     "examples/rows.csv": "sha256:e", "requirements.txt": "sha256:r",
                     "pkg/model/network.py": "sha256:net", "pkg/blueprint-map.json": "sha256:map",
                     "pkg/serving/values.py": "sha256:v", "serve/app.py": "sha256:a",
                     "Dockerfile": "sha256:d", "tests/test_roundtrip.py": "sha256:t"},
           "created_at": "2026-09-30T00:00:00Z", "model_version": "sha256:mv",
           "entrypoints": ["predict.py", "serve/app.py", "examples/request.json", "Dockerfile"],
           "roles": {"code": ["pkg/model/network.py"], "data": ["weights.pt", "netspec.json"]},
           "output": {"reading": "binary", "threshold": 0.5},
           "roundtrip": {"status": "pass", "max_abs_diff": 0.0}, "exports": [],
           "serve": {"present": True, "templates": {"digest": "sha256:t0"}},
           "blueprint": {"run": "pipeline.json", "conformance": {"mechanical": "pass"}},
           "reemits": []}
    doc.update(copy.deepcopy(changes))
    return doc


# ============================================================ what may move

def test_the_outputs_are_the_catalogs_patterns_at_their_destinations():
    for rel in ("serve/app.py", "serve/routes_new.py", "serve/static/index.html",
                "tests/test_api_contract.py", "pkg/serving/values.py",
                "pkg/serving/inspection/codec.py", "pkg/serving/predict.py",
                "pkg/monitoring/monitor.py", "predict.py", "Dockerfile", ".dockerignore",
                "pyproject.toml", "exports/model.pt", "exports/parity.json"):
        assert OUT.holds(rel), rel
    for rel in ("serve/static/app.js", "serve/sub/x.py", "tests/sub/test_x.py",
                "pkg/serving/inspection/deep/x.py", "pkg/serving/data.json",
                "pkg/model/network.py", "pkg/monitoring/other.py", "pkg/blueprint-map.json",
                "pkg/ports.py", "pkg/__init__.py", "requirements.txt", "requirements-serve.txt",
                "train.py", "README.md", "MANIFEST.json", "exportsx/model.pt"):
        assert not OUT.holds(rel), rel
    # every catalog pattern but the root's is placed at its destination, and none is invented
    expected = {("pkg/serving/" if dest == "package" else f"{dest}/") + p.split("/", 1)[1]
                for dest, patterns in CATALOG["serving_templates"].items() if dest != "bundle"
                for p in patterns}
    assert set(OUT.patterns()) == expected
    assert sorted(CATALOG["reemit"]["may_change"]["templates"]) == \
        sorted(p for ps in CATALOG["serving_templates"].values() for p in ps)


def test_a_never_pattern_is_never_an_output():
    greedy = reemit.ServingOutputs("pkg", "serving", ("weights.pt",),
                                   ("weights.pt", "netspec.json", "schema.json"),
                                   ("examples", "graph", "records"))
    for pattern in reemit.NEVER:
        rel = pattern.replace("*", "pt")
        assert not greedy.holds(rel), (pattern, rel)
    assert not greedy.holds("examples/rows.csv") and not greedy.holds("graph/graph.json")
    assert "weights.*" in reemit.NEVER and "examples/*" in reemit.NEVER
    assert all("<" not in p for p in reemit.NEVER)


def test_unchanged_manifests_move_nothing():
    old = _manifest()
    assert reemit.compare(old, copy.deepcopy(old), OUT) == []
    assert reemit.changed_files(old, copy.deepcopy(old)) == []


def test_the_keys_and_outputs_that_may_change_move_freely():
    old = _manifest()
    new = _manifest(created_at="2026-10-01T00:00:00Z", reemits=[{"at": "x"}],
                    roles={"code": ["pkg/model/network.py", "serve/new.py"]},
                    roundtrip={"status": "pass", "max_abs_diff": 1e-7},
                    exports=[{"format": "onnx", "file": "exports/model.onnx"}],
                    serve={"present": True, "templates": {"digest": "sha256:t1"},
                           "endpoints": ["GET /health"]},
                    blueprint={"run": "pipeline.json", "conformance": {"mechanical": "fail"}},
                    entrypoints=["predict.py", "serve/app.py", "serve/new.py",
                                 "examples/request.json", "Dockerfile"])
    new["files"].update({"serve/app.py": "sha256:a2", "pkg/serving/values.py": "sha256:v2",
                         "Dockerfile": "sha256:d2", "serve/new.py": "sha256:new",
                         "exports/model.onnx": "sha256:x"})
    del new["files"]["tests/test_roundtrip.py"]
    assert reemit.compare(old, new, OUT) == []
    assert reemit.changed_files(old, new) == sorted(
        ["Dockerfile", "exports/model.onnx", "pkg/serving/values.py", "serve/app.py",
         "serve/new.py", "tests/test_roundtrip.py"])
    free = {e for e in CATALOG["reemit"]["may_change"]["manifest"] if "(" not in e}
    assert reemit.MAY_CHANGE == free and "blueprint.conformance" in free


def test_a_pinned_file_or_key_that_moved_is_named():
    old = _manifest()
    new = _manifest(model_version="sha256:other", output={"reading": "binary", "threshold": 0.4},
                    chain={"model": "m.html", "model_digest": "sha256:m2"},
                    blueprint={"run": "/abs/pipeline.json", "conformance": {"mechanical": "pass"}})
    new["files"].update({"weights.pt": "sha256:w2", "examples/rows.csv": "sha256:e2",
                         "pkg/model/network.py": "sha256:net2", "requirements.txt": "sha256:r2",
                         "examples/extra.csv": "sha256:x"})
    del new["files"]["netspec.json"]
    assert reemit.compare(old, new, OUT) == [
        "examples/extra.csv", "examples/rows.csv", "netspec.json", "pkg/model/network.py",
        "requirements.txt", "weights.pt", "MANIFEST.blueprint.run",
        "MANIFEST.chain.model_digest", "MANIFEST.model_version", "MANIFEST.output.threshold"]
    added, dropped = _manifest(extra=1), _manifest()
    del dropped["chain"]
    assert reemit.compare(old, added, OUT) == ["MANIFEST.extra"]
    assert reemit.compare(old, dropped, OUT) == ["MANIFEST.chain"]


def test_entrypoints_move_only_in_their_outputs():
    old = _manifest()
    moved = _manifest(entrypoints=["predict.py", "serve/app.py", "examples/request.json",
                                   "examples/demo.json", "Dockerfile"])
    assert reemit.compare(old, moved, OUT) == ["MANIFEST.entrypoints"]
    grown = _manifest(entrypoints=["predict.py", "serve/app.py", "examples/request.json",
                                   "Dockerfile", "requirements-serve.txt"])
    assert reemit.compare(old, grown, OUT) == ["MANIFEST.entrypoints"]
    served_only = _manifest(entrypoints=["predict.py", "examples/request.json"])
    assert reemit.compare(old, served_only, OUT) == []


def test_the_package_map_may_change_only_in_the_outputs_entries():
    old = _manifest()
    new = _manifest()
    new["files"]["pkg/blueprint-map.json"] = "sha256:map2"
    before = {"schema": "vega/blueprint-map@1", "blueprint": "pipeline.json",
              "entries": [{"role": "Model", "file": "model/network.py", "status": "implemented"},
                          {"role": "Predictor", "file": "serving/predict.py",
                           "status": "implemented"}],
              "entrypoints": [{"file": "predict.py", "calls": ["Predictor"]}]}
    grown = copy.deepcopy(before)
    grown["entries"].append({"file": "serving/new.py", "kind": "support"})
    grown["entries"][1]["status"] = "implemented (re-emitted)"
    grown["entrypoints"].append({"file": "serve/new.py", "calls": []})
    assert reemit.compare(old, new, OUT, (before, grown)) == []
    model_moved = copy.deepcopy(grown)
    model_moved["entries"][0]["status"] = "pending"
    assert reemit.compare(old, new, OUT, (before, model_moved)) == ["pkg/blueprint-map.json"]
    repointed = dict(copy.deepcopy(grown), blueprint="/elsewhere/pipeline.json")
    assert reemit.compare(old, new, OUT, (before, repointed)) == ["pkg/blueprint-map.json"]
    assert reemit.compare(old, new, OUT) == ["pkg/blueprint-map.json"]


# ============================================================ the bundle in place

def _bundle(root: Path) -> Path:
    """A tiny bundle: two pinned files with roles, the manifest, an unpinned README."""
    (root / "serve").mkdir(parents=True)
    (root / "netspec.json").write_text("{}\n", encoding="utf-8")
    (root / "serve" / "app.py").write_text("app = None\n", encoding="utf-8")
    (root / "README.md").write_text("prose\n", encoding="utf-8")
    files = {rel: _digest(root / rel) for rel in ("netspec.json", "serve/app.py")}
    (root / "MANIFEST.json").write_text(json.dumps(
        {"files": files, "roles": {"data": ["netspec.json"], "code": ["serve/app.py"]}}),
        encoding="utf-8")
    return root


def _digest(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_altered_names_what_the_manifest_no_longer_pins(tmp_path):
    root = _bundle(tmp_path / "b" / "s-R0")
    assert reemit.altered(root) == []
    (root / "README.md").write_text("a user's correction\n", encoding="utf-8")
    assert reemit.altered(root) == []
    (root / "netspec.json").write_text('{"seed": 8}\n', encoding="utf-8")
    assert reemit.altered(root) == ["netspec.json"]
    (root / "serve" / "app.py").unlink()
    assert reemit.altered(root) == ["netspec.json", "serve/app.py"]
    doc = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    doc["roles"]["data"].append("schema.json")
    (root / "MANIFEST.json").write_text(json.dumps(doc), encoding="utf-8")
    assert "schema.json" in reemit.altered(root)
    (root / "MANIFEST.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    assert reemit.altered(root) == ["MANIFEST.json (it records no file roles)"]
    (root / "MANIFEST.json").write_text("{", encoding="utf-8")
    assert reemit.altered(root) == ["MANIFEST.json (unreadable)"]


def test_prepare_refuses_no_bundle_and_an_altered_one_and_stages_beside_a_verified_one(
        tmp_path):
    paths = BundlePaths(root=tmp_path / "b" / "s-R0", package="pkg")
    options = SimpleNamespace(re_emit_serving=True)
    with pytest.raises(PackExit) as none:
        reemit.prepare(options, paths, [])
    assert none.value.code == 2 and "no bundle to re-emit" in none.value.lines[0]
    _bundle(paths.root)
    markers = []
    staged = reemit.prepare(options, paths, markers)
    assert staged.root == tmp_path / "b" / ".reemit" / "s-R0" == reemit.stage_dir(paths.root)
    assert staged.root.is_dir() and not any(staged.root.iterdir())
    assert (paths.root / "netspec.json").is_file() and "verifies" in markers[0]
    (paths.root / "netspec.json").write_text("[]\n", encoding="utf-8")
    shutil.rmtree(staged.root.parent)
    with pytest.raises(PackExit) as refused:
        reemit.prepare(options, paths, [])
    assert refused.value.code == 2 and refused.value.lines[0] == reemit.NOT_PINNED
    assert "netspec.json" in refused.value.lines[1]
    assert not (tmp_path / "b" / ".reemit").exists()
    plain = reemit.prepare(SimpleNamespace(re_emit_serving=False), paths, [])
    assert plain == paths and paths.root.is_dir() and not any(paths.root.iterdir())


def test_swap_puts_the_staged_bundle_in_place_and_the_old_one_back_on_failure(tmp_path):
    root = tmp_path / "b" / "s-R0"
    staging = reemit.stage_dir(root)
    for directory, text in ((root, "old"), (staging, "new")):
        directory.mkdir(parents=True)
        (directory / "f.txt").write_text(text, encoding="utf-8")
    reemit.swap(root, staging)
    assert (root / "f.txt").read_text(encoding="utf-8") == "new"
    assert not (tmp_path / "b" / ".reemit").exists()
    reemit.stage_dir(root).parent.mkdir()
    with pytest.raises(OSError):
        reemit.swap(root, reemit.stage_dir(root))          # nothing staged: the rename fails
    assert (root / "f.txt").read_text(encoding="utf-8") == "new"
    assert not (reemit.stage_dir(root).parent / "s-R0.prev").exists()


def test_what_other_stages_wrote_beside_the_bundle_is_carried_over(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    for rel, text in (("weights.pt", "w"), ("model-card.html", "card"),
                      ("showcase/SHOWCASE.json", "{}"), ("showcase/page/index.html", "<p>"),
                      ("markers.json", "{}"), ("README.md", "old prose"),
                      ("__pycache__/x.pyc", "b"), ("serve/gone.py", "x = 1")):
        (old / rel).parent.mkdir(parents=True, exist_ok=True)
        (old / rel).write_text(text, encoding="utf-8")
    os.symlink("page", old / "showcase" / "link")
    new.mkdir()
    (new / "weights.pt").write_text("w", encoding="utf-8")
    (new / "README.md").write_text("new prose", encoding="utf-8")
    reemit._carry_over(old, new, {"weights.pt": "sha256:w", "serve/gone.py": "sha256:g"})
    assert (new / "model-card.html").read_text(encoding="utf-8") == "card"
    assert (new / "showcase" / "page" / "index.html").is_file()
    assert (new / "showcase" / "link").is_symlink()
    assert (new / "README.md").read_text(encoding="utf-8") == "new prose"
    assert not (new / "markers.json").exists() and not (new / "__pycache__").exists()
    assert not (new / "serve" / "gone.py").exists()


def test_a_plain_pack_records_an_empty_reemits_list(tmp_path):
    ctx = SimpleNamespace(options=SimpleNamespace(re_emit_serving=False),
                          paths=BundlePaths(root=tmp_path, package="pkg"))
    assert reemit.finish(ctx, {"files": {}}) == ctx.paths
    assert json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))["reemits"] == []


# ============================================================ the real thing

_WORK: list = []
_DONE: dict = {}


def _work() -> Path:
    if not _WORK:
        named = os.environ.get("VEGA_REEMIT_WORK")
        if named:
            Path(named).mkdir(parents=True, exist_ok=True)
            _WORK.append(Path(named))
        else:
            made = Path(tempfile.mkdtemp(prefix="vega-reemit-"))
            atexit.register(shutil.rmtree, made, True)
            _WORK.append(made)
    return _WORK[0]


def _env() -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def _run(argv, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, argv)], capture_output=True, text=True,
                          timeout=TIMEOUT, env=_env(), cwd=cwd)


def packed() -> rf.Packed:
    """The fixture made, trained, evaluated and packed, in a child process (the spike's
    emitter runs in the process that makes a fixture)."""
    if "packed" not in _DONE:
        proc = _run([rf.__file__, _work(), "--fixture", FIXTURE, "--json"])
        try:
            result = rf.Packed.from_dict(json.loads(proc.stdout)[0])
        except (json.JSONDecodeError, IndexError) as exc:
            raise AssertionError(f"the harness failed (exit {proc.returncode})\n"
                                 f"{proc.stderr[-3000:]}") from exc
        assert rf.passed(result), result.pack_output[-3000:]
        _DONE["packed"] = result
    return _DONE["packed"]


def _pack_args(result, bundle_dir, reason) -> list:
    """`roundtrip_fixtures.pack`'s own inputs, with the re-emit flag and a reason."""
    return ["--campaign", result.campaign, "--shaped", result.shaped, "--bundle-dir",
            bundle_dir, "--pipeline", result.pipeline, "--ship-anyway", rf.STANCE,
            "--unattended", "--serve", "--demo-sample", "3", "--re-emit-serving",
            "--monitor-plan", Path(result.campaign) / "monitor-plan.json", "--reason", reason]


def _copy(result, name) -> Path:
    """A copy of the packed bundle under its own name, in a bundle directory of its own."""
    target = _work() / name / "bundle" / Path(result.bundle).name
    shutil.rmtree(target.parent, ignore_errors=True)
    shutil.copytree(result.bundle, target)
    return target


def _reemit(result, bundle, reason, templates=None) -> dict:
    """One re-emit of `bundle` by the real pack.py; the manifests either side and the output."""
    before = (bundle / "MANIFEST.json").read_bytes()
    args = _pack_args(result, bundle.parent, reason)
    argv = ([rf.PACK_PY, *args] if templates is None else
            ["-c", OVERRIDE.format(scripts=str(PACK)), templates, *args])
    proc = _run(argv)
    return {"exit": proc.returncode, "out": proc.stdout + proc.stderr,
            "before": json.loads(before), "before_digest": "sha256:" +
            hashlib.sha256(before).hexdigest(),
            "after": json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))}


def _snapshot(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.is_symlink(), _digest(p) if p.is_file() else None)
            for p in sorted(root.rglob("*")) if not p.is_dir() or p.is_symlink()}


def _moved_keys(old, new, path="") -> set:
    """The dotted manifest keys that differ (a key's children only where both are objects;
    `files` as one key)."""
    out = set()
    for key in set(old) | set(new):
        a, b, where = old.get(key), new.get(key), f"{path}{key}"
        if isinstance(a, dict) and isinstance(b, dict) and where != "files":
            out |= _moved_keys(a, b, where + ".")
        elif a != b:
            out.add(where)
    return out


def _outside(keys, allowed) -> set:
    """The keys that are neither one of `allowed` nor below one."""
    return {k for k in keys if not any(k == a or k.startswith(a + ".") for a in allowed)}


def _showcase(bundle: Path) -> None:
    """A minimal verified showcase and a card beside the bundle, as /vega-show and the card
    stage leave them: pinned by nothing the pack wrote."""
    manifest = bundle / "MANIFEST.json"
    (bundle / "model-card.html").write_text("<html><body>card</body></html>\n",
                                            encoding="utf-8")
    show = bundle / "showcase"
    (show / "page").mkdir(parents=True)
    (show / "data").mkdir()
    (show / "page" / "index.html").write_text("<!doctype html><p>showcase</p>\n",
                                              encoding="utf-8")
    (show / "data" / "story.json").write_text("{}\n", encoding="utf-8")
    files = {rel: _digest(show / rel) for rel in ("page/index.html", "data/story.json")}
    doc = {"schema": "vega/showcase@1",
           "bundle": {"manifest_digest": _digest(manifest),
                      "model_version": json.loads(manifest.read_text())["model_version"]},
           "files": files, "served": sorted(files), "entry": "page/index.html",
           "story": {"file": "data/story.json"}}
    (show / "SHOWCASE.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def _slot(bundle: Path) -> dict:
    """The bundle's own showcase slot, opened as its service opens it."""
    code = ("import json, sys\nfrom pathlib import Path\n"
            "from serve.routes_showcase import ShowcaseSlot\n"
            "root = Path('.')\nmanifest = (root / 'MANIFEST.json').read_bytes()\n"
            "slot = ShowcaseSlot.open(root / 'showcase', manifest, "
            "json.loads(manifest)['model_version'])\nprint(json.dumps(slot.status()))\n")
    proc = _run(["-c", code], cwd=bundle)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _unchanged(result) -> dict:
    """Two re-emits in a row with the templates as they are, a card and a showcase beside."""
    bundle = _copy(result, "unchanged")
    _showcase(bundle)
    slot_before = _slot(bundle)
    first = _reemit(result, bundle, "unchanged templates, first")
    second = _reemit(result, bundle, "unchanged templates, second")
    check = _run(["predict.py", "--check"], cwd=bundle)
    return {"bundle": bundle, "first": first, "second": second, "slot_before": slot_before,
            "slot_after": _slot(bundle), "check": check}


def _templates_copy(result, target: Path) -> tuple:
    """A copy of the backend's serving templates; the package's serving module in the bundle."""
    source = PACK.parent.parent / "vega-forge" / "scripts" / "backends" / "pytorch" / \
        "templates" / "serving"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    package = result.manifest["package"]
    return target, f"{package}/{bundle_modules(Path(result.bundle) / package)['serving']}"


def _probed_templates(result, target: Path) -> dict:
    """The templates' copy with one template per destination changed: {template: the bundle
    file it renders}."""
    target, base = _templates_copy(result, target)
    pinned = result.manifest["files"]
    found = matching_templates(target)
    groups = {"serve": [(r, r) for r in found["serve"]],
              "tests": [(r, r) for r in found["tests"]],
              "bundle": [(r, r.split("/", 1)[1]) for r in found["bundle"]],
              "package": [(r, f"{base}/{r.split('/', 1)[1]}") for r in found["package"]
                          if r.count("/") == 1 and r not in ("package/predict.py",
                                                             "package/monitor.py")],
              "inspection": [(r, f"{base}/{r.split('/', 1)[1]}") for r in found["package"]
                             if r.startswith("package/inspection/")]}
    chosen = {}
    for group, pairs in groups.items():
        pick = next(((r, o) for r, o in pairs if o in pinned), None)
        assert pick or group == "inspection", f"no {group} template renders a pinned file"
        if pick:
            chosen[pick[0]] = pick[1]
            (target / pick[0]).write_text((target / pick[0]).read_text(encoding="utf-8")
                                          + PROBE, encoding="utf-8")
    return chosen


def _changed(result) -> dict:
    templates = _work() / "changed" / "templates" / "serving"
    bundle = _copy(result, "changed")
    chosen = _probed_templates(result, templates)
    return {"bundle": bundle, "chosen": chosen,
            "run": _reemit(result, bundle, "a template fix", templates)}


def _added(result) -> dict:
    """The templates' copy with one `package` template more — a support module a structural
    fix split out — re-emitted."""
    templates, base = _templates_copy(result, _work() / "added" / "templates" / "serving")
    (templates / ADDED).write_text(PROBE_MODULE, encoding="utf-8")
    bundle = _copy(result, "added")
    return {"bundle": bundle, "output": f"{base}/{ADDED.split('/', 1)[1]}",
            "run": _reemit(result, bundle, "a support module split out", templates)}


def _altered(result) -> dict:
    bundle = _copy(result, "altered")
    spec = bundle / "netspec.json"
    spec.write_text(spec.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    snapshot = _snapshot(bundle)
    return {"bundle": bundle, "run": _reemit(result, bundle, "an altered bundle"),
            "snapshot": snapshot}


def reemitted() -> dict:
    """The four re-emits that only read the campaign, side by side."""
    if "runs" not in _DONE:
        result = packed()
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {name: pool.submit(fn, result) for name, fn in
                    (("unchanged", _unchanged), ("changed", _changed), ("added", _added),
                     ("altered", _altered))}
            _DONE["runs"] = {name: job.result() for name, job in jobs.items()}
    return _DONE["runs"]


def _resave_checkpoint(camp: Path) -> dict:
    """The campaign's checkpoint saved again — the same tensors, other bytes — and the ledger
    pointing at it, as a re-trained run under the same id would leave it. The originals."""
    import torch
    run_dir = camp / "runs" / rf.RUN
    best = run_dir / "checkpoints" / "best.pt"
    kept = {p: p.read_bytes() for p in (best, camp / "experiments.json", run_dir / "run.json")}
    blob = torch.load(best, map_location="cpu", weights_only=True)
    blob["saved_again_by"] = "test_reemit"
    torch.save(blob, best)
    digest = _digest(best)
    ledger = json.loads((camp / "experiments.json").read_text(encoding="utf-8"))
    for row in ledger["runs"]:
        if row["id"] == rf.RUN:
            row["checkpoint_digest"] = digest
    (camp / "experiments.json").write_text(json.dumps(ledger, indent=2) + "\n",
                                           encoding="utf-8")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run["checkpoint_digest"] = digest
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return kept


def weights_changed() -> dict:
    """A re-emit after the campaign's weights changed (run after the others: it edits the
    campaign, and puts it back)."""
    if "weights" not in _DONE:
        result = packed()
        reemitted()
        bundle = _copy(result, "weights")
        snapshot = _snapshot(bundle)
        kept = _resave_checkpoint(Path(result.campaign))
        try:
            run = _reemit(result, bundle, "the weights changed")
        finally:
            for path, data in kept.items():
                path.write_bytes(data)
        _DONE["weights"] = {"bundle": bundle, "run": run, "snapshot": snapshot}
    return _DONE["weights"]


def test_packed_unchanged_templates_change_only_time_conformance_and_reemits():
    runs = reemitted()["unchanged"]
    first = runs["first"]
    assert first["exit"] == 0, first["out"][-3000:]
    old, new = first["before"], first["after"]
    assert new["files"] == old["files"], reemit.changed_files(old, new)
    assert not _outside(_moved_keys(old, new), UNCHANGED_MAY_MOVE), _moved_keys(old, new)
    assert old["reemits"] == [] and len(new["reemits"]) == 1
    entry = new["reemits"][0]
    assert set(entry) == ENTRY_KEYS
    assert entry["previous_manifest_digest"] == first["before_digest"]
    assert entry["templates_digest"] == new["serve"]["templates"]["digest"]
    assert entry["changed"] == [] and entry["reason"] == "unchanged templates, first"
    assert "GATE: replacing" in first["out"] and "(0 file(s))" in first["out"]
    markers = json.loads((runs["bundle"] / "markers.json").read_text(encoding="utf-8"))
    assert any(m.startswith("GATE: replacing") for m in markers["markers"])
    assert not (runs["bundle"].parent / reemit.STAGING).exists()


def test_packed_a_second_re_emit_appends_to_the_first():
    runs = reemitted()["unchanged"]
    first, second = runs["first"]["after"], runs["second"]
    assert second["exit"] == 0, second["out"][-3000:]
    assert second["before"] == first
    entries = second["after"]["reemits"]
    assert len(entries) == 2 and entries[0] == first["reemits"][0]
    assert entries[1]["previous_manifest_digest"] == second["before_digest"]
    assert second["after"]["files"] == first["files"]


def test_packed_what_other_stages_wrote_is_carried_and_the_showcase_is_now_stale():
    runs = reemitted()["unchanged"]
    bundle = runs["bundle"]
    card = (bundle / "model-card.html").read_text(encoding="utf-8")
    assert card == "<html><body>card</body></html>\n"
    assert (bundle / "showcase" / "SHOWCASE.json").is_file()
    assert runs["slot_before"]["verified"] is True, runs["slot_before"]
    assert runs["slot_after"]["verified"] is False and runs["slot_after"]["reason"] == STALE
    showcase = json.loads((bundle / "showcase" / "SHOWCASE.json").read_text(encoding="utf-8"))
    assert showcase["bundle"]["manifest_digest"] == runs["first"]["before_digest"]
    assert runs["check"].returncode == 0, runs["check"].stdout[-2000:]


def test_packed_a_changed_template_changes_exactly_its_outputs():
    runs = reemitted()["changed"]
    run, chosen = runs["run"], runs["chosen"]
    assert run["exit"] == 0, run["out"][-3000:]
    old, new = run["before"], run["after"]
    assert set(reemit.changed_files(old, new)) == set(chosen.values()), \
        (reemit.changed_files(old, new), chosen)
    assert set(new["reemits"][-1]["changed"]) == set(chosen.values())
    assert {"serve", "tests", "bundle"} <= {rel.split("/")[0] for rel in chosen}
    assert sum(rel.startswith("package/") for rel in chosen) >= 1
    listed = (old["serve"]["templates"]["files"], new["serve"]["templates"]["files"])
    assert {t for t in set(listed[0]) | set(listed[1])
            if listed[0].get(t) != listed[1].get(t)} == set(chosen)
    assert new["reemits"][-1]["templates_digest"] == new["serve"]["templates"]["digest"] != \
        old["serve"]["templates"]["digest"]
    allowed = UNCHANGED_MAY_MOVE | {"files", "serve.templates"}
    assert not _outside(_moved_keys(old, new), allowed), _moved_keys(old, new)
    for rel, out in chosen.items():
        text = (runs["bundle"] / out).read_text(encoding="utf-8")
        assert text.rstrip().endswith(PROBE.strip()), (rel, out)


def test_packed_an_added_template_adds_its_output_and_its_map_entry():
    runs = reemitted()["added"]
    run, output = runs["run"], runs["output"]
    assert run["exit"] == 0, run["out"][-3000:]
    old, new = run["before"], run["after"]
    package = new["package"]
    the_map = f"{package}/blueprint-map.json"
    assert reemit.changed_files(old, new) == sorted([output, the_map])
    text = (runs["bundle"] / output).read_text(encoding="utf-8")
    assert "reemit_probe" in text and "{{" not in text, text
    assert new["entrypoints"] == old["entrypoints"]
    assert output in new["roles"]["code"] and output not in old["roles"]["code"]
    allowed = UNCHANGED_MAY_MOVE | {"files", "serve.templates", "roles"}
    assert not _outside(_moved_keys(old, new), allowed), _moved_keys(old, new)
    before = json.loads((Path(packed().bundle) / the_map).read_text(encoding="utf-8"))
    after = json.loads((runs["bundle"] / the_map).read_text(encoding="utf-8"))
    added = [e for e in after["entries"] if e.get("file") == output.split("/", 1)[1]]
    assert len(added) == 1 and added[0].get("kind") == "support", added
    after["entries"] = [e for e in after["entries"] if e not in added]
    assert after == before


def test_packed_changed_weights_are_refused_and_the_bundle_untouched():
    runs = weights_changed()
    run = runs["run"]
    assert run["exit"] == 3, run["out"][-3000:]
    line = next(ln for ln in run["out"].splitlines() if "re-emit would change" in ln)
    named = line.split("re-emit would change ", 1)[1].split("; what the evaluation", 1)[0]
    assert named.split(", ") == ["weights.pt"], line
    assert "re-pack deliberately" in line and "left as it was" in run["out"]
    assert _snapshot(runs["bundle"]) == runs["snapshot"]
    assert run["after"] == run["before"]
    assert not (runs["bundle"].parent / reemit.STAGING).exists()


def test_packed_an_altered_bundle_is_refused_before_anything_is_built():
    runs = reemitted()["altered"]
    run = runs["run"]
    assert run["exit"] == 2, run["out"][-3000:]
    assert reemit.NOT_PINNED in run["out"] and "netspec.json" in run["out"]
    assert _snapshot(runs["bundle"]) == runs["snapshot"]
    assert not (runs["bundle"].parent / reemit.STAGING).exists()


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                made = tempfile.mkdtemp(prefix="reemit-")
                atexit.register(shutil.rmtree, made, True)
                fn(Path(made))
            else:
                fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
