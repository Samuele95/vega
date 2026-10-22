#!/usr/bin/env python3
"""
test_pack_seams.py — `pack.py` is a composition over fixed seams, and its command line is one
frozen `PackOptions`.

    python3 test_pack_seams.py            # or: python3 -m pytest test_pack_seams.py

Pure: nothing is packed. What it holds fixed is what later work builds on without editing
`pack.py` — every phase module's entry point and its parameters, the flags every one of them
reads, the refusal of the three flags that moved to /vega-show, the way records nest into the
manifest — and the size that says `pack.py` stayed a composition.
"""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib
import inspect
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
sys.path.insert(0, str(SCRIPTS))

import bundle_manifest  # noqa: E402
import bundle_map  # noqa: E402
import pack_options  # noqa: E402
import serving_pack  # noqa: E402
from pack_context import SKILLS, PackExit, merge_records  # noqa: E402

REQUIRED = ["--campaign", "c", "--shaped", "s", "--bundle-dir", "b"]

# module -> {function: parameters}: the seams pack.py calls, in its order
SEAMS = {
    "output_record": {"records": ["ctx"], "write_schema": ["campaign", "root", "markers"]},
    "graph_pack": {"write": ["ctx"]},
    "serving_pack": {"render_adapters": ["ctx"], "render": ["ctx"],
                     "identity": ["options", "pipeline", "schema"]},
    "roundtrip_plan": {"run": ["ctx", "predictor"]},
    "examples_pack": {"write": ["ctx", "predictor"], "write_request": ["ctx"]},
    "demo_pack": {"write": ["ctx", "predictor"]},
    "monitor_pack": {"refuse_unfilled": ["options"], "render": ["ctx", "predictor"]},
    "inspection_pack": {"check": ["ctx", "predictor"]},
    "export_pack": {"write": ["ctx", "predictor"]},
    "reemit": {"prepare": ["options", "paths", "markers"], "finish": ["ctx", "manifest"]},
    # what a phase calls to change what is not its own (pack_context's docstring lists them)
    "bundle_map": {"add_map_entries": ["ctx", "source", "files", "points"]},
    "bundle_manifest": {"refresh_stub": ["ctx"]},
}

# the modules S00 wrote: the size discipline (C11-C13, R-G1) holds on every one of them
PHASE_MODULES = ("pack", "pack_options", "pack_context", "output_record", "roundtrip_plan",
                 "examples_pack", "demo_pack", "monitor_pack", "serving_pack", "graph_pack",
                 "inspection_pack", "reemit", "export_pack", "bundle_inputs", "bundle_slice",
                 "bundle_map", "bundle_manifest")


def _refusal(argv):
    try:
        with contextlib.redirect_stderr(io.StringIO()):      # argparse's usage text
            pack_options.parse(argv)
    except PackExit as refusal:
        return refusal
    except SystemExit as exc:
        return exc
    return None


def test_every_seam_has_its_signature():
    for module, functions in SEAMS.items():
        mod = importlib.import_module(module)
        for name, params in functions.items():
            fn = getattr(mod, name, None)
            assert callable(fn), f"{module}.{name} is missing"
            got = list(inspect.signature(fn).parameters)
            assert got == params, f"{module}.{name}{tuple(got)}, the seam is {tuple(params)}"


def test_every_documented_flag_is_declared():
    o = pack_options.parse(REQUIRED + [
        "--eval", "e", "--export", "torchscript,onnx", "--re-emit-serving", "--reason", "r",
        "--no-inspect", "--inspect-weights", "summary", "--unattended", "--serve"])
    assert (o.eval, o.export, o.re_emit_serving, o.reason) == \
        ("e", ("torchscript", "onnx"), True, "r")
    assert (o.inspect, o.inspect_weights, o.unattended, o.serve) == (False, "summary", True, True)


def test_the_defaults_are_todays_behaviour():
    o = pack_options.parse(REQUIRED)
    assert (o.export, o.re_emit_serving, o.inspect, o.inspect_weights) == ((), False, True, "full")
    assert (o.unattended, o.serve, o.include, o.demo_sample) == (False, None, "serving", 20)


def test_run_dir_and_out_dir_are_the_campaign_and_the_bundle_dir():
    o = pack_options.parse(["--run-dir", "c", "--shaped", "s", "--out-dir", "b"])
    assert (o.campaign, o.bundle_dir) == ("c", "b")


def test_the_options_are_frozen():
    o = pack_options.parse(REQUIRED)
    try:
        o.include = "full"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("PackOptions accepted an assignment")


def test_the_story_flags_moved_to_vega_show():
    for flag in pack_options.REMOVED_FLAGS:
        for argv in (REQUIRED + [flag, "x"], REQUIRED + [f"{flag}=x"]):
            refusal = _refusal(argv)
            assert isinstance(refusal, PackExit) and refusal.code == 2, argv
            assert refusal.lines[0].startswith(f"{flag} moved to /vega-show"), refusal.lines
            assert "carries no story" in refusal.lines[0]


def test_a_reason_needs_a_re_emit_and_an_export_needs_a_known_format():
    for argv in (REQUIRED + ["--reason", "r"], REQUIRED + ["--export", "pdf"]):
        refusal = _refusal(argv)
        assert isinstance(refusal, SystemExit) and refusal.code == 2, argv


def test_records_nest_under_their_keys_top_level_first():
    manifest = {"chain": {"shape": {"manifest_digest": "d"}}, "exports": []}
    merge_records(manifest, {"serve.inspection": {"enabled": True},
                             "chain.shape.fitted": {"projection": "serving"},
                             "serve": {"present": True}, "exports": [{"format": "onnx"}],
                             "graph": None})
    assert manifest["serve"] == {"present": True, "inspection": {"enabled": True}}
    assert manifest["chain"]["shape"] == {"manifest_digest": "d",
                                          "fitted": {"projection": "serving"}}
    assert manifest["exports"] == [{"format": "onnx"}] and "graph" not in manifest


def test_a_refusal_takes_its_lines_as_one_parameter():
    one, several = PackExit(2, "why"), PackExit(3, ("first", "second"), keep_markers=True)
    assert (one.code, one.lines, one.level) == (2, ("why",), "ERROR")
    assert (several.lines, several.keep_markers, str(several)) == \
        (("first", "second"), True, "first; second")


def test_what_a_phase_maps_survives_the_rewrite():
    """`add_map_entries` before `finish`: the support entry keeps `implemented`, the entry
    point is kept and marked present, a file already listed is not listed twice."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pkg = root / "pipe"
        (pkg / "serving" / "inspection").mkdir(parents=True)
        (pkg / "serving" / "inspection" / "codec.py").write_text("", encoding="utf-8")
        (root / "serve").mkdir()
        (root / "serve" / "routes_inspect.py").write_text("", encoding="utf-8")
        entries = [{"file": "serving/predict.py", "module": "serving", "role": "Predictor",
                    "status": "pending: /vega-pack"}]
        points = [{"file": "predict.py", "calls": "Predictor", "stage": "/vega-pack"}]
        (pkg / "blueprint-map.json").write_text(json.dumps(
            {"entries": entries, "entrypoints": points}), encoding="utf-8")
        ctx = SimpleNamespace(paths=SimpleNamespace(pkg=pkg), markers=[])
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                bundle_map.add_map_entries(ctx, "the inspection templates",
                                           files=["serving/inspection/codec.py"],
                                           points=[{"file": "serve/routes_inspect.py",
                                                    "calls": ["Predictor"]}])
            bundle_map.rewrite_map(pkg / "blueprint-map.json", ["serving/predict.py"], [])
            bundle_map.mark_entrypoints(pkg / "blueprint-map.json", [])
        doc = json.loads((pkg / "blueprint-map.json").read_text(encoding="utf-8"))
        support = [e for e in doc["entries"] if e.get("kind") == "support"]
        assert support == [{"file": "serving/inspection/codec.py", "module": "serving",
                            "kind": "support", "used_by": ["Predictor"],
                            "source": "the inspection templates", "status": "implemented"}]
        added = [p for p in doc["entrypoints"] if p["file"] == "serve/routes_inspect.py"]
        assert added == [{"stage": "/vega-pack", "file": "serve/routes_inspect.py",
                          "calls": ["Predictor"], "present": True}], doc["entrypoints"]


def test_a_served_bundle_lists_every_served_file_as_an_entry_point():
    """Amendment A1: MANIFEST.entrypoints is c849f8b's list — the CLIs, then every file the
    service is made of and run from, in c849f8b's order: serve/ (its `__init__.py` and entry
    first, the other modules by name, the page last), the demonstration files only when
    demo_pack wrote them, the example request, the requirements, the distribution, the image
    and the API tests. From S07 the served files are the ones `serve/` holds (the route
    modules and the row contract of v2.3), read from the bundle."""
    tail = ["examples/request.json", "requirements-serve.txt", "pyproject.toml", "Dockerfile",
            ".dockerignore"]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "examples").mkdir()
        (root / "serve" / "static").mkdir(parents=True)
        (root / "tests").mkdir()
        for rel in ("serve/__init__.py", "serve/app.py", "serve/routes_meta.py",
                    "serve/contract.py", "serve/static/index.html", "tests/test_api.py",
                    "tests/test_roundtrip.py"):
            (root / rel).write_text("", encoding="utf-8")
        served = ["serve/__init__.py", "serve/app.py", "serve/contract.py",
                  "serve/routes_meta.py", "serve/static/index.html"]

        def ctx(include):
            return SimpleNamespace(options=SimpleNamespace(include=include),
                                   paths=SimpleNamespace(root=root))
        assert serving_pack.served_entrypoints(ctx("serving")) == \
            ["predict.py", *served, *tail, "tests/test_api.py"]
        for name in ("demo.json", "presets.json"):
            (root / "examples" / name).write_text("", encoding="utf-8")
        demo = ["examples/demo.json", "examples/presets.json"]
        assert serving_pack.served_entrypoints(ctx("serving")) == \
            ["predict.py", *served, *demo, *tail, "tests/test_api.py"]
        assert serving_pack.served_entrypoints(ctx("full")) == \
            ["predict.py", "train.py", "evaluate.py", *served, *demo, *tail,
             "tests/test_api.py"]


def test_an_entry_point_the_bundle_does_not_carry_is_refused():
    bundle_manifest.check_entrypoints({"entrypoints": ["predict.py"],
                                       "files": {"predict.py": "d", "schema.json": "d"}})
    try:
        bundle_manifest.check_entrypoints({"entrypoints": ["predict.py", "serve/app.py"],
                                           "files": {"predict.py": "d"}})
    except PackExit as refusal:
        assert refusal.code == 3 and "serve/app.py" in refusal.lines[0], refusal.lines
        return
    raise AssertionError("an entry point the bundle does not carry was accepted")


def test_a_data_file_written_after_the_load_is_pinned_for_its_next_reader():
    """The monitor reference is written after the predictor loaded; `refresh_stub` pins it (role
    `data`) before the pack-time smoke reads it, keeping what the stub already carried."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "netspec.json").write_text("{}", encoding="utf-8")
        stub = {"package": "pipe", "files": {}, "roles": {}, "output": {"reading": "binary"}}
        (root / "MANIFEST.json").write_text(json.dumps(stub), encoding="utf-8")
        (root / "monitor-reference.json").write_text("{}", encoding="utf-8")
        ctx = SimpleNamespace(paths=SimpleNamespace(root=root, manifest=root / "MANIFEST.json"),
                              package="pipe")
        bundle_manifest.refresh_stub(ctx)
        doc = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        assert sorted(doc["files"]) == ["monitor-reference.json", "netspec.json"]
        assert doc["roles"]["data"] == ["monitor-reference.json", "netspec.json"]
        assert doc["output"] == {"reading": "binary"} and doc["package"] == "pipe"


def test_the_phase_modules_hold_the_size_discipline():
    """G1's gate (`conformance.py --size-only`) when this tree has it; until then its C12 rule
    on variadics, measured here: `*args` / `**kwargs` only in a `_prefixed` helper."""
    files = [str(SCRIPTS / f"{name}.py") for name in PHASE_MODULES]
    gate = SKILLS / "vega-forge" / "scripts" / "conformance.py"
    if gate.is_file() and "--size-only" in gate.read_text(encoding="utf-8"):
        proc = subprocess.run([sys.executable, str(gate), "--size-only", *files, "--json"],
                              capture_output=True, text=True, timeout=300)
        findings = json.loads(proc.stdout)["size"]["findings"]
        assert not findings, [f["message"] for f in findings]
        return
    for path in files:
        for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            private = node.name.startswith("_") and not node.name.endswith("__")
            variadic = node.args.vararg or node.args.kwarg
            assert private or not variadic, f"{Path(path).name}:{node.lineno} {node.name}"


def test_pack_py_is_a_composition():
    lines = len((SCRIPTS / "pack.py").read_text(encoding="utf-8").splitlines())
    assert lines < 400, f"pack.py is {lines} lines; a phase belongs in its module"
    # the story and the report moved to /vega-show, and the templates to the backend's files
    for script in sorted(SCRIPTS.glob("*.py")):
        text = script.read_text(encoding="utf-8")
        for gone in ("def write_story", "def _story_lookup", "def report_labels",
                     "STORY_SOURCES", "STORY_TEXT_KEYS", "def _brace_group",
                     "def _class_attribution", "API_TEST =", "CLI =", "DOCKERFILE ="):
            assert gone not in text, f"{script.name} still holds {gone!r}"


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
