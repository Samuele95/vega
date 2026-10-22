#!/usr/bin/env python3
"""
test_serving_templates.py — the serving templates hold the code discipline before any bundle
is packed (D2 §10.2, BUILD-PLAN G2, R-D2, R-F10).

    python3 -m pytest test_serving_templates.py [-q]
    python3 test_serving_templates.py

`T/serving/` is `vega-forge/scripts/backends/pytorch/templates/serving/`: `package/` renders
into the Predictor's module, `serve/` into the bundle's `serve/`, `bundle/` into the bundle
root, `tests/` into the bundle's `tests/` (vega-pack `catalog.json` `serving_templates`).

What is proved:

  render    every `T/serving/**/*.py` renders with a placeholder value per `{{NAME}}` — the
            names vega-pack's `serving_templates.fills` gives, with neutral values — and its
            `# >>> shape:` sections for every shape and for each one alone (the pack keeps the
            bundle's, through the emitter's `select_shapes`), leaves no placeholder, and parses; a placeholder this test does not know is a failure that
            names it (a template grew a fill the pack must supply, and so must this test);
  C11–C13   `conformance.size_findings` finds nothing in any rendered template, at the
            contract's limits (≤ 30 statements, ≤ 5 parameters, ≤ 200 lines per class, ≤ 400
            per module), for every shape and for each one alone — the same measure `conformance.py --entrypoints` applies to a bundle's
            `predict.py`, `serve/` and `tests/`, and the package check to the Predictor's
            module;
  C14       `package/predict.py` and `package/monitor.py` are the Predictor's and the Monitor's
            adapters: `conformance.adapter_findings` finds nothing — the top level holds the
            adapter class, `_prefixed` helpers, imports and UPPER_CASE constants, and nothing
            else;
  literals  `lint_templates.py` over `T/serving/` exits 0: no dataset literal of any lane, the
            one deny-list gate (R-F10);
  patterns  every `serving_templates` pattern names at least one template file, under
            vega-pack's own matching rule (`vega_pack_core.template_matches`: a `*` never
            crosses a `/`); every Python template is named by a pattern (a template no pattern
            names would never render, nor be measured in a bundle); `reemit.may_change
            .templates` names the same patterns (R-D2).

Exit codes (as a script): 0 every check holds; 1 a check failed.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
SERVING = SCRIPTS / "backends" / "pytorch" / "templates" / "serving"
LINT = SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
PACK_SCRIPTS = SKILLS / "vega-pack" / "scripts"
for _p in (str(SCRIPTS), str(PACK_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conformance  # noqa: E402
from vega_pack_core import CATALOG, template_matches  # noqa: E402

_EMIT_SPEC = importlib.util.spec_from_file_location(
    "vega_serving_test_emit", str(SCRIPTS / "backends" / "pytorch" / "emit.py"))
EMIT = importlib.util.module_from_spec(_EMIT_SPEC)
_EMIT_SPEC.loader.exec_module(EMIT)
# every shape (the template as it stands), then each sample shape alone (a bundle's)
SHAPE_SETS = (None,) + tuple((s,) for s in EMIT.SAMPLE_SHAPES)

PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")
# The bundle's CLIs (`bundle/*.py`) take the pack's single-brace fills instead
# (serving_pack._write_entrypoints): `{package}`, `{env}` and `{mod_<key>}`, a module per key.
CLI_FILL = re.compile(r"\{(package|env|mod_[a-z_]+)\}")

# A neutral value per placeholder the pack fills (serving_templates.fills and _identity_fills).
# The values are syntactically what the pack writes — an identifier where a module name goes,
# a Python literal where `repr()` goes — and name no dataset.
FILLS = {
    "PACKAGE": "demo_pipeline", "SERVING_MODE": "online", "MONITOR_MODULE": "serving",
    "TITLE_PY": repr("Demo service"), "DESCRIPTION_PY": repr("Answers for demo records."),
    "TITLE_HTML": "Demo service", "DESCRIPTION_HTML": "Answers for demo records.",
    "NOUNS": "records", "NOUN": "record", "ROW_MODEL": "RecordRow", "CMD": "demo",
    "ENV": "DEMO", "PORT": "8000",
}
# The per-role families the pack fills from the bundle's map and the blueprint's ports.
FAMILIES = (("MODULE_", lambda role: role.lower()),
            ("PORT_", lambda role: role.capitalize()),
            ("RESPONSIBILITY_", lambda role: f"the {role.lower()} port's responsibility"),
            ("SOURCE_", lambda role: f"blueprint:ports/{role.capitalize()}"))

# The two adapters among the package templates (C14): template -> (role, port).
ADAPTERS = {"package/predict.py": ("Predictor", "Predictor"),
            "package/monitor.py": ("Monitor", "Monitor")}


def fill(name: str) -> str:
    """The neutral value of one placeholder, or KeyError naming it."""
    if name in FILLS:
        return FILLS[name]
    for prefix, value in FAMILIES:
        if name.startswith(prefix) and len(name) > len(prefix):
            return value(name[len(prefix):])
    raise KeyError(f"{{{{{name}}}}} is a placeholder this test does not know — the pack must "
                   "fill it (serving_templates.fills), and FILLS here must name it too")


def _cli_fill(name: str) -> str:
    return {"package": FILLS["PACKAGE"], "env": FILLS["ENV"]}.get(name) or name[len("mod_"):]


def render(text: str, rel: str = "", shapes=None) -> str:
    """`text` with the shape sections of `shapes` (default: every shape) and every placeholder
    filled as the pack fills it for the template `rel`."""
    text = EMIT.select_shapes(text, shapes, rel)
    if rel.startswith("bundle/"):
        text = CLI_FILL.sub(lambda m: _cli_fill(m.group(1)), text)
    return PLACEHOLDER.sub(lambda m: fill(m.group(1)), text)


def python_templates() -> list:
    """Every `T/serving/**/*.py`, as template-relative paths."""
    return sorted(p.relative_to(SERVING).as_posix() for p in SERVING.rglob("*.py")
                  if "__pycache__" not in p.parts)


def rendered_tree(rel: str, shapes=None) -> ast.Module:
    return ast.parse(render((SERVING / rel).read_text(encoding="utf-8"), rel, shapes),
                     filename=rel)


def all_template_files() -> list:
    return sorted(p.relative_to(SERVING).as_posix() for p in SERVING.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts)


# --------------------------------------------------------------------------- the checks

def test_every_python_template_renders_and_parses():
    assert python_templates(), f"no serving template under {SERVING}"
    problems = []
    for rel in python_templates():
        for shapes in SHAPE_SETS:
            try:
                text = render((SERVING / rel).read_text(encoding="utf-8"), rel, shapes)
                ast.parse(text, filename=rel)
            except (KeyError, SyntaxError, EMIT.EmitError) as exc:
                problems.append(f"{rel} ({shapes or 'every shape'}): {exc}")
                continue
            if PLACEHOLDER.search(text):
                problems.append(f"{rel}: a placeholder is left after rendering")
    assert not problems, "\n".join(problems)


def test_no_template_breaks_c11_to_c13():
    findings = [f"{f.message()} ({shapes or 'every shape'})"
                for rel in python_templates() for shapes in SHAPE_SETS
                for f in conformance.size_findings(rendered_tree(rel, shapes), rel)]
    assert not findings, "C11–C13 over the rendered serving templates:\n" + "\n".join(findings)


def test_the_two_adapters_hold_c14():
    found = []
    for rel, (role, port) in ADAPTERS.items():
        assert (SERVING / rel).is_file(), f"{rel} is not a serving template"
        found += [f.message() for f in conformance.adapter_findings(rendered_tree(rel), rel,
                                                                    role, port)]
    assert not found, "C14 on the serving adapters:\n" + "\n".join(found)


def test_the_adapter_check_refuses_a_second_public_class():
    """The C14 reading this test relies on is live: a public class beside the Predictor (the
    hand-built bundle's `NetworkView`) is a finding, a `_prefixed` one is not."""
    rel = "package/predict.py"
    text = render((SERVING / rel).read_text(encoding="utf-8"), rel)
    seeded = ast.parse(text + "\n\nclass NetworkView:\n    pass\n")
    helper = ast.parse(text + "\n\nclass _NetworkView:\n    pass\n")
    codes = [(f.code, f.symbol) for f in conformance.adapter_findings(
        seeded, rel, "Predictor", "Predictor")]
    assert codes == [("C14", "NetworkView")], codes
    assert conformance.adapter_findings(helper, rel, "Predictor", "Predictor") == []


def test_no_foreign_literal_in_any_serving_template():
    proc = subprocess.run([sys.executable, str(LINT), str(SERVING)], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_every_serving_templates_pattern_names_a_template():
    files = all_template_files()
    empty = [f"{dest}: {pattern}" for dest, patterns in CATALOG["serving_templates"].items()
             for pattern in patterns if not any(template_matches(pattern, rel) for rel in files)]
    assert not empty, "serving_templates patterns naming no template file: " + ", ".join(empty)


def test_every_python_template_is_named_by_a_pattern():
    patterns = [p for group in CATALOG["serving_templates"].values() for p in group]
    unnamed = [rel for rel in python_templates()
               if not any(template_matches(p, rel) for p in patterns)]
    assert not unnamed, ("Python templates no serving_templates pattern names (never rendered, "
                         "never measured in a bundle): " + ", ".join(unnamed))


def _package_template(name: str) -> ast.Module:
    """A package template (`templates/<name>`), its placeholders filled with plain names."""
    text = (SERVING.parent / name).read_text(encoding="utf-8")
    return ast.parse(PLACEHOLDER.sub(lambda m: m.group(1).lower(), text), filename=name)


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def test_the_bundle_calls_the_package_with_its_own_signatures():
    """A serving template constructs the package's `RunContext` with the fields `domain.py`
    declares and its `Trainer` the way `loop.py`'s constructor takes it — a bundle's
    `train.py` is run by no unit test of the package, only by a lane's pack stage, so a
    moved field (the ModelStore left RunContext for `Trainer(store)`) is caught here."""
    fields = {n.target.id for n in _class(_package_template("domain.py"), "RunContext").body
              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    init = next(n for n in _class(_package_template("loop.py"), "Trainer").body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    takes = [a.arg for a in init.args.args[1:]]
    assert fields and takes == ["store"], (fields, takes)
    problems = []
    for rel in python_templates():
        for call in ast.walk(rendered_tree(rel)):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            given = {k.arg for k in call.keywords}
            if call.func.id == "RunContext" and (call.args or given - fields):
                problems.append(f"{rel}:{call.lineno} RunContext takes {sorted(fields)} by "
                                f"keyword, given {sorted(given)} and {len(call.args)} positional")
            if call.func.id == "Trainer" and (len(call.args) > len(takes)
                                              or given - set(takes)):
                problems.append(f"{rel}:{call.lineno} Trainer takes {takes}")
            if call.func.id == "Trainer" and "checkpoint_dir" in ast.unparse(rendered_tree(rel)) \
                    and not (call.args or given):
                problems.append(f"{rel}:{call.lineno} a Trainer that writes checkpoints is "
                                "given the ModelStore they are written through")
    assert not problems, "\n".join(problems)


def test_the_reemit_allow_list_names_the_same_patterns():
    patterns = sorted(p for group in CATALOG["serving_templates"].values() for p in group)
    may_change = sorted(CATALOG["reemit"]["may_change"]["templates"])
    assert may_change == patterns, json.dumps({"serving_templates": patterns,
                                               "reemit.may_change.templates": may_change})


# --------------------------------------------------------------------------- as a script

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"OK: {name}")
        except AssertionError as exc:
            failed += 1
            print(f"ERROR: {name}: {exc}")
    print(f"{'OK' if not failed else 'ERROR'}: {len(tests) - failed}/{len(tests)} serving "
          "template check(s) hold")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
