#!/usr/bin/env python3
"""
test_conformance_sizes.py — the size discipline C11–C14 on AST fixtures, and the class-aware
C8/C9 (code-contract.md §5.1; D6 §14.3).

    python3 test_conformance_sizes.py [-v]        (or: pytest test_conformance_sizes.py)

1. The measures — `conformance.measure_module`, D6's `CONFORMANCE_MEASURES` — on the modules
   of `conformance_fixtures/` whose counts `expected.py` knows: exact statements, parameters,
   lines.
2. `conformance.size_findings` names exactly the over-limit ones, in the contract's format.
3. The clean package `conformance_fixtures/package/demo_pipeline` (the v2.3 shape: class
   adapters, `build_model = Network.build`) passes C0–C14 at `--size-severity error`. Every
   seed of `seeds.SEEDS` planted in a copy of it exits as it must: the boundaries C11b, C12b,
   C12d, the classmethod C12e, C13c, C13d and C14d exit 0; every other seed exits 2 naming
   what was seeded. So do `scenarios.RULE_SEEDS` (C14f–C14i: a lower-case module-level name,
   the import guards, a helper class). The default severity is `error` (G2): a seed is refused
   without the flag, and is a `WARN:`, exit 0, only under `--size-severity warn`.
4. `--entrypoints` measures `predict.py`, `serve/`, `tests/` and the map's `entrypoints[]`
   beside the package (C11–C13, not C14); `--bundle` reads the paradigms of the serving slice
   only; `--size-only` measures loose files, and a directory sweep parses the skills' test
   harnesses without measuring them (PLAN-AMENDMENTS A13) while a directory inside a harness
   tree, named as the thing to measure, is measured; `--json` is one document with the
   findings as values, an unreadable input included (exit 3).
5. C8 finds `load_part` on the adapter class, C9 reads `build_model = Network.build`; a
   `load_part` that does not raise and a `Network(...)` outside `Network.build` are refused.
6. C14's class rule reads implemented adapters only (a function-shaped pending stub passes
   until it is marked implemented), and never the FeatureTransformer's module (the shaper's
   functions, R-A3), which C10 and C11–C13 still read; an implemented Predictor stays held.

Exit 0 when every test passes.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "conformance_fixtures"
PACKAGE = FIXTURES / "package" / "demo_pipeline"
PIPELINE = FIXTURES / "package" / "pipeline.json"
sys.path.insert(0, str(FIXTURES))
sys.path.insert(0, str(HERE.parent))

import conformance  # noqa: E402
import expected  # noqa: E402
import scenarios  # noqa: E402
import seeds  # noqa: E402
from scenarios import c14_lines, edit, implement, json_document  # noqa: E402

VERBOSE = "-v" in sys.argv


def _measures(name: str):
    path = FIXTURES / name
    return conformance.measure_module(ast.parse(path.read_text(encoding="utf-8")), name)


def _run(*args, cwd=None) -> tuple:
    proc = subprocess.run([sys.executable, str(seeds.CONFORMANCE), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def _clean(*extra) -> tuple:
    return _run("--package", PACKAGE, "--pipeline", PIPELINE, *extra)


def _copy(td) -> Path:
    return scenarios.copy_package(PACKAGE, td)


# --------------------------------------------------------------------------- 1. measures

def test_statement_parameter_and_variadic_counts_are_exact():
    for (name, symbol), (stmts, params, variadic, helper) in expected.FUNCTIONS.items():
        f = _measures(name).function(symbol)
        assert f is not None, f"{name}: no measure for {symbol}"
        got = (f.statements, f.parameters, f.variadic, f.helper)
        assert got == (stmts, params, variadic, helper), f"{name}:{symbol} measured {got}"


def test_class_and_module_lines_are_exact():
    for (name, symbol), lines in expected.CLASSES.items():
        assert _measures(name).klass(symbol).lines == lines, f"{name}:{symbol}"
    for name, lines in expected.MODULE_LINES.items():
        assert _measures(name).lines == lines, name


def test_a_docstring_and_comments_are_not_statements():
    text = (FIXTURES / "stmt_30.py").read_text(encoding="utf-8")
    assert '"""The docstring is not a statement."""' in text and "# a comment" in text
    assert _measures("stmt_30.py").function("thirty").statements == 30


# --------------------------------------------------------------------------- 2. findings

def test_size_findings_name_exactly_the_over_limit_defs():
    for name, wanted in expected.FINDINGS.items():
        tree = ast.parse((FIXTURES / name).read_text(encoding="utf-8"))
        got = [(f.code, f.line, f.symbol, f.measured)
               for f in conformance.size_findings(tree, name)]
        assert got == wanted, f"{name}: {got}"


def test_the_message_is_the_contracts_format():
    tree = ast.parse((FIXTURES / "stmt_31.py").read_text(encoding="utf-8"))
    (finding,) = conformance.size_findings(tree, "stmt_31.py")
    assert finding.message() == "C11 stmt_31.py:3 thirty_one — 31 statements > 30 — EJ-51"
    tree = ast.parse((FIXTURES / "varargs_public.py").read_text(encoding="utf-8"))
    spread = conformance.size_findings(tree, "varargs_public.py")[0].message()
    assert spread.startswith("C12 varargs_public.py:3 spread — *args, **kwargs in a public "
                             "signature") and "_prefixed" in spread and spread.endswith("EJ-51")


def test_the_module_exemption_is_by_name_and_covers_the_module_rule_only():
    source = (FIXTURES / "module_401.py").read_text(encoding="utf-8")
    for exempt in ("domain.py", "model/metrics.py"):
        assert conformance.size_findings(ast.parse(source), exempt) == []
    long_method = source + "\n\ndef long():\n" + "".join("    x = 1\n" for _ in range(31))
    codes = [f.code for f in conformance.size_findings(ast.parse(long_method), "domain.py")]
    assert codes == ["C11"], codes


# --------------------------------------------------------------------------- 3. the package

def test_the_clean_package_passes_c0_to_c14_at_error():
    rc, out = _clean("--size-severity", "error")
    assert rc == 0, out
    for code in ("C11", "C12", "C13", "C14"):
        assert f"OK: {code} " in out, f"no OK line for {code}:\n{out}"
    assert "WARN:" not in out, out


def test_every_seed_exits_as_it_must():
    failures = []
    for seed in (*seeds.SEEDS, *scenarios.RULE_SEEDS):
        with tempfile.TemporaryDirectory() as td:
            try:
                copy, planted = seeds.plant(seed, PACKAGE, PIPELINE, Path(td))
            except seeds.SeedError as exc:
                failures.append(f"{seed.name}: could not be planted: {exc}")
                continue
            rc, out = _run(*seeds.command(copy, PIPELINE, planted)[2:])
        wrong = seeds.problems(seed, planted, rc, out, boundary_exit=0)
        if VERBOSE or wrong:
            print(f"    {'ok  ' if not wrong else 'FAIL'}  {seed.name}"
                  + (f"  — {'; '.join(wrong)}" if wrong else ""))
        if wrong:
            failures.append(f"{seed.name}: {'; '.join(wrong)}")
    assert not failures, "\n".join(failures)


def test_the_boundaries_named_by_the_plan_exit_zero():
    boundaries = {s.id for s in seeds.SEEDS if not s.refused}
    assert {"C11b", "C12b", "C12d", "C13c", "C13d", "C14d"} <= boundaries, boundaries
    refused = {s.id for s in seeds.SEEDS if s.refused}
    expected = {"C0a", "C11a", "C11c", "C11d", "C11e", "C12a", "C12c", "C12e", "C13a", "C13b",
                "C14a", "C14b", "C14c", "C14e"}
    assert expected <= refused, refused


def test_under_the_default_severity_a_seed_is_refused():
    seed = next(s for s in seeds.SEEDS if s.id == "C11a")
    with tempfile.TemporaryDirectory() as td:
        copy, _ = seeds.plant(seed, PACKAGE, PIPELINE, Path(td))
        rc, out = _run("--package", copy, "--pipeline", PIPELINE)
        assert rc == 2, out
        assert "ERROR: C11 model/store.py:" in out and "_seeded_long — 31 statements > 30" in out
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--size-severity", "warn")
    assert rc == 0, out
    assert "WARN: C11 model/store.py:" in out and "_seeded_long — 31 statements > 30" in out
    assert "reported as WARN under --size-severity warn" in out, out


def test_a_sweep_does_not_measure_the_test_harnesses():
    """A13: a sweep of `<skill>/scripts` parses `scripts/tests/**` and measures none of it; the
    same file is measured where it is shipped code, when it is named, and when the directory
    named is inside the harness tree."""
    body = (FIXTURES / "stmt_31.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        scripts = Path(td) / "skill" / "scripts"
        (scripts / "tests").mkdir(parents=True)
        for rel in ("shipped.py", "tests/harness.py"):
            (scripts / rel).write_text(body, encoding="utf-8")
        rc, out = _run("--size-only", scripts)
        assert rc == 2 and "shipped.py:3 thirty_one" in out and "harness.py" not in out, out
        assert "1 file(s) of the skills' test harnesses" in out, out
        rc, out = _run("--size-only", scripts / "tests")
        assert rc == 2 and "harness.py:3 thirty_one" in out, out
        rc, out = _run("--size-only", scripts / "tests" / "harness.py")
        assert rc == 2 and "harness.py:3 thirty_one" in out, out
        (scripts / "tests" / "harness.py").write_text("def broken(:\n", encoding="utf-8")
        rc, out = _run("--size-only", scripts)
        assert rc == 2 and "C0" in out and "harness.py" in out, out       # still parsed


def test_json_is_one_document_with_the_findings_as_values():
    seed = next(s for s in seeds.SEEDS if s.id == "C11a")
    with tempfile.TemporaryDirectory() as td:
        copy, _ = seeds.plant(seed, PACKAGE, PIPELINE, Path(td))
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--size-severity", "error",
                       "--json")
    doc = json_document(out)
    assert rc == 2 and doc["mechanical"] == "fail"
    assert doc["size"]["severity"] == "error"
    assert doc["size"]["counts"] == {"C11": 1, "C12": 0, "C13": 0, "C14": 0}
    (finding,) = doc["size"]["findings"]
    assert (finding["symbol"], finding["measured"], finding["limit"]) == ("_seeded_long", 31, 30)
    assert any(m["rule"] == "C11" and m["level"] == "ERROR" for m in doc["markers"])


def test_json_stays_one_document_when_an_input_is_unreadable():
    for args in (("--package", PACKAGE, "--pipeline", FIXTURES / "no_such_pipeline.json"),
                 ("--package", PACKAGE, "--pipeline", PIPELINE, "--map",
                  FIXTURES / "no_such_map.json"),
                 ("--package", FIXTURES / "no_such_package", "--pipeline", PIPELINE),
                 ("--size-only", FIXTURES / "no_such_file.py")):
        rc, out = _run(*args, "--json")
        doc = json_document(out)                        # the whole output, and nothing else
        assert rc == 3 and doc["mechanical"] == "fail", out
        assert any(m["rule"] == "IO" and m["level"] == "ERROR" for m in doc["markers"]), out
        assert doc["size"]["findings"] == [], out


# --------------------------------------------------------------------------- 4. scopes

def test_entrypoints_are_measured_by_c11_to_c13_only():
    with tempfile.TemporaryDirectory() as td:
        root = scenarios.make_bundle(PACKAGE, td)
        package = root / PACKAGE.name
        rc, out = _run("--package", package, "--pipeline", PIPELINE, "--entrypoints",
                       "--size-severity", "error")
        assert rc == 0, out
        assert "3 entrypoint file(s)" in out, out
        app = root / "serve" / "app.py"
        app.write_text(app.read_text() + "\n\ndef handler():\n"
                       + "".join("    x = 1\n" for _ in range(31)), encoding="utf-8")
        rc, out = _run("--package", package, "--pipeline", PIPELINE, "--entrypoints",
                       "--size-severity", "error")
        assert rc == 2 and "C11 serve/app.py:5 handler — 31 statements > 30" in out, out
        rc, out = _run("--package", package, "--pipeline", PIPELINE, "--size-severity", "error")
        assert rc == 0, out                  # without the flag, entrypoints are not read
        rc, out = _run("--package", package, "--pipeline", PIPELINE, "--size-severity",
                       "error", "--entrypoints", root)
        assert rc == 2 and "serve/app.py" in out, out   # an explicit ROOT


def test_an_entrypoint_the_map_lists_is_measured_even_off_the_globs():
    long_fn = "def run_batch():\n" + "".join("    x = 1\n" for _ in range(31))
    with tempfile.TemporaryDirectory() as td:
        root = scenarios.make_bundle(PACKAGE, td)
        package = root / PACKAGE.name
        (root / "cli").mkdir()
        (root / "cli" / "run_batch.py").write_text(long_fn, encoding="utf-8")
        rc, out = _run("--package", package, "--pipeline", PIPELINE, "--entrypoints",
                       "--size-severity", "error")
        assert rc == 0 and "cli/run_batch.py" not in out, out   # no glob, no map entry
        scenarios.add_entrypoint(package, "cli/run_batch.py")
        rc, out = _run("--package", package, "--pipeline", PIPELINE, "--entrypoints",
                       "--size-severity", "error")
    assert rc == 2 and "C11 cli/run_batch.py:1 run_batch — 31 statements > 30" in out, out
    assert "4 entrypoint file(s)" in out, out


def test_a_bundle_reads_the_paradigms_of_its_serving_slice_only():
    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        scenarios.out_of_slice(copy, "training")
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--bundle",
                       "--size-severity", "error")
        assert rc == 0, out                                   # the slice itself is clean
        training_fp = scenarios.with_paradigm(PIPELINE, td, "training", "FP")
        rc, out = _run("--package", copy, "--pipeline", training_fp, "--bundle",
                       "--size-severity", "error")
        assert rc == 0 and "modules[training]" not in out, out   # out of the slice
        model_fp = scenarios.with_paradigm(PIPELINE, td, "model", "FP")
        rc, out = _run("--package", copy, "--pipeline", model_fp, "--bundle",
                       "--size-severity", "error")
    assert rc == 2 and "C14 pipeline.json modules[model]" in out and "OOP" in out, out


def test_size_only_measures_loose_files_and_directories():
    rc, out = _run("--size-only", FIXTURES / "stmt_30.py", FIXTURES / "params_5.py",
                   "--size-severity", "error")
    assert rc == 0, out
    rc, out = _run("--size-only", FIXTURES / "stmt_31.py", "--size-severity", "error")
    assert rc == 2 and "thirty_one — 31 statements > 30" in out, out
    rc, out = _run("--size-only", FIXTURES / "stmt_31.py")
    assert rc == 2 and "ERROR: C11" in out, out
    rc, out = _run("--size-only", FIXTURES / "stmt_31.py", "--size-severity", "warn")
    assert rc == 0 and "WARN: C11" in out, out
    rc, out = _run("--size-only", PACKAGE, "--size-severity", "error")
    assert rc == 0 and "on all 19 file(s)" in out, out
    rc, out = _run("--size-only", FIXTURES / "no_such_file.py")
    assert rc == 3, out
    rc, out = _run("--size-only", FIXTURES / "stmt_30.py", "--package", PACKAGE)
    assert rc == 2 and "takes no --package" in out, out


def test_a_file_that_does_not_parse_is_c0_in_every_scope():
    with tempfile.TemporaryDirectory() as td:
        broken = Path(td) / "broken.py"
        broken.write_text("def broken(:\n    pass\n", encoding="utf-8")
        rc, out = _run("--size-only", broken)
        assert rc == 2 and "C0 " in out and "does not parse" in out, out
        root = scenarios.make_bundle(PACKAGE, td)
        (root / "predict.py").write_text("def main(:\n", encoding="utf-8")
        rc, out = _run("--package", root / PACKAGE.name, "--pipeline", PIPELINE,
                       "--entrypoints")
        assert rc == 2 and "C0 predict.py:1 does not parse" in out, out


# --------------------------------------------------------------------------- 5. C8 / C9

def test_c8_reads_load_part_on_the_adapter_class():
    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        edit(copy / "ingestion" / "data.py", 'raise SealedPartError("load_part refuses',
              'print("load_part refuses')
        rc, out = _run("--package", copy, "--pipeline", PIPELINE)
    assert rc == 2 and "load_part does not raise SealedPartError" in out, out
    assert "ingestion/data.py:" in out, out


def test_c9_reads_the_build_alias_and_refuses_a_second_constructor():
    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        evaluate = copy / "evaluation" / "evaluate.py"
        evaluate.write_text(evaluate.read_text() + "\n\ndef _rogue():\n"
                            "    return Network((), \"rogue\")\n", encoding="utf-8")
        rc, out = _run("--package", copy, "--pipeline", PIPELINE)
    assert rc == 2 and "evaluation/evaluate.py:" in out and "constructs Network(...)" in out, out

    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        edit(copy / "model" / "network.py", "    def param_count(self) -> tuple:",
              "    def clone(self) -> \"Network\":\n        return Network(self._weights, "
              "self._spec_name)\n\n    def param_count(self) -> tuple:")
        rc, out = _run("--package", copy, "--pipeline", PIPELINE)
    assert rc == 2 and "model/network.py:" in out and "constructs Network(...)" in out, out

    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        edit(copy / "model" / "network.py", "build_model = Network.build\n", "")
        rc, out = _run("--package", copy, "--pipeline", PIPELINE)
    assert rc == 2 and "exposes no build_model" in out, out


def test_c14_names_the_adapter_that_is_not_a_class():
    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        store = copy / "model" / "store.py"
        edit(store, "class ModelRegistry:", "class _Registry:")
        edit(store, '__all__ = ["ModelRegistry"]', '__all__ = []')
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--size-severity", "error")
    assert rc == 2 and "C14 model/store.py:1 Registry — the adapter of port 'Registry'" in out, out


def test_the_feature_transformer_module_is_c10s_and_not_c14s():
    # the pending stub in the clean package: nothing
    rc, out = _clean("--size-severity", "error")
    assert rc == 0 and not c14_lines(out, "features/preprocess.py"), out
    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        rel = implement(copy, "FeatureTransformer")
        (copy / rel).write_text(scenarios.SHAPER_MODULE, encoding="utf-8")
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--size-severity", "error")
        assert rc == 0 and not c14_lines(out, rel), out      # public functions: the port
        assert f"OK: {rel} satisfies transform(df_raw, history=None) structurally" in out, out
        # C11–C13 still read it
        edit(copy / rel, "def last_unseen():\n",
              "def last_unseen():\n" + "".join("    x = 1\n" for _ in range(30)))
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--size-severity", "error")
        assert rc == 2 and f"C11 {rel}:19 last_unseen — 31 statements > 30" in out, out
        assert not c14_lines(out, rel), out
        # C10 still reads its shape
        edit(copy / rel, "def transform(df_raw,", "def transform_rows(df_raw,")
        rc, out = _run("--package", copy, "--pipeline", PIPELINE)
    assert rc == 2 and f"{rel} exposes no transform(df_raw, history=None)" in out, out


def test_an_implemented_predictor_is_still_held_to_c14():
    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        rel = "serving/predict.py"
        (copy / rel).write_text(scenarios.FUNCTION_PREDICTOR, encoding="utf-8")
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--size-severity", "error")
        assert rc == 0 and not c14_lines(out, rel), out       # pending: not read yet
        assert implement(copy, "Predictor") == rel
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--size-severity", "error")
    assert rc == 2, out
    assert f"C14 {rel}:1 Predictor — the adapter of port 'Predictor' (role Predictor) is not " \
           "a class" in out, out
    assert f"C14 {rel}:7 predict — a public function" in out, out
    assert "edges_for — a public function" in out, out
    with tempfile.TemporaryDirectory() as td:
        copy = _copy(td)
        implement(copy, "Predictor")
        rc, out = _run("--package", copy, "--pipeline", PIPELINE, "--size-severity", "error")
    assert rc == 0, out                                       # the class-shaped one passes


# --------------------------------------------------------------------------- the runner

def main() -> int:
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}\n        " + "\n        ".join(str(exc).splitlines()[:12]))
    print(f"\n{len(tests) - len(failed)} of {len(tests)} conformance size test(s) passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
