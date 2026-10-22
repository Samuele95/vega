#!/usr/bin/env python3
"""
conformance.py — the mechanical half of the gate every emission passes.

    python3 conformance.py --package <run-dir>/<package>
                           --pipeline blueprint-runs/<ts>/pipeline.json
                           [--map <run-dir>/<package>/blueprint-map.json]
                           [--bundle] [--entrypoints [ROOT]]
                           [--size-severity error|warn] [--json] [--explain]

    python3 conformance.py --size-only PATH [PATH ...] [--size-severity error|warn] [--json]

This script answers the questions a machine can answer about whether the emitted code is
*the blueprint's* code (`code-contract.md` §5.1). The other half — whether it is good code —
is `/architect <package> --report-only --baseline <blueprint.md>`, which the stage invokes
through the Skill tool and whose `structural` / `divergence` findings are also `ERROR:`.

The checks, each named and each citing its clause:

  C0  every file of the package (and, with --entrypoints, every entrypoint file) parses;
  C1  every blueprint port has exactly one primary adapter, or a `pending: <stage>` entry, or
      — with `--bundle` — an `out of scope: <reason>` entry;
  C2  extra implementations of a port exist only where the map marks `role: baseline`;
  C3  every file in the package has an adapter or a support entry, and `unmapped` is empty;
  C4  the import graph respects `modules[].depends_on` plus the two implicit arrows to
      `domain.py` and `ports.py`, and every package-relative import — a lazy one inside a
      function included — names a module the package holds (a dispatch to a module that was
      never emitted fails only when its branch runs);
  C5  adapters import ports and `domain`, never another adapter (the `role: baseline`
      entries of one port may import each other: one family of alternatives);
  C6  every class in `domain.py` is a frozen dataclass or an enum, and every blueprint
      artifact is accounted for (`vega_roster`): a name `domain.py` defines, listed in the
      map's `domain.py` entry, or declared external with its owner and recorded in the map's
      `external_artifacts` — never both, never neither; and every name `domain.py` holds an
      artifact under is declared in that entry once, a blueprint artifact under `artifacts` or
      the backend's own under `backend_artifacts`;
  C7  no module-level mutable state anywhere in the package (an UPPER_CASE container is a
      constant lookup table by convention and passes; a lower-case one is state);
  C8  the DataSource-role adapter raises on the test part — `load_part` is found on the
      adapter class or at module level;
  C9  the Model-role adapter's `build_model` is the only constructor of the network — a
      module-level `def build_model` or the alias `build_model = Network.build`, and then
      `Network(...)` is called nowhere but inside `Network.build`;
  C10 the FeatureTransformer-role adapter satisfies the Protocol structurally
      (`transform(df_raw, history=None)`, a function or a method), which the shaper's emitted
      module does without knowing `ports.py`;
  C11 no function or method longer than 30 statements;
  C12 no function or method with more than 5 parameters beyond `self` / `cls`, and `*args` /
      `**kwargs` only in a `_prefixed` helper;
  C13 no class longer than 200 lines, no module longer than 400 (`domain.py` and `metrics.py`
      are exempt from the module rule, never from C11–C12);
  C14 every port-owning module is declared `OOP` in `pipeline.json`, and every implemented
      port adapter is a class whose module's top level holds only that class, `_prefixed`
      helpers, imports and constants (`build_model = Network.build` included); `kind: support`
      files, `role: baseline` modules, the `FeatureTransformer`-role module and entrypoints
      are exempt from it, not from C11–C13. The FeatureTransformer's module is the shaper's
      transform-only module, whose public functions `transform`, `edges_for` and
      `edges_within` ARE the port (R-A3): C10 reads it structurally instead.

The size discipline C11–C14 is measured on the AST, never on the text (the counting rule is
`conformance_sizes.py`'s docstring). Each finding is one line in the contract's format,
`C11 training/loop.py:88 fit — 131 statements > 30 — EJ-51`. C11–C14 findings are `ERROR:`
under `--size-severity error` — the default since the templates meet the limits (BUILD-PLAN
G2) — and `WARN:` under `--size-severity warn`, which a caller asks for to survey code that is
not held to the discipline yet; C0–C10 are always `ERROR:`. The thresholds are the contract's
and are not flags.

`--entrypoints [ROOT]` applies C11–C13 to the files a stage writes outside the package —
`predict.py`, `serve/**/*.py`, `tests/**/*.py`, every `entrypoints[].file` of the map and every
`*.py` a bundle's `MANIFEST.json` lists in `entrypoints` (a `full` bundle's `train.py` and
`evaluate.py`) — under ROOT (default: the package's parent, the bundle root).
`--size-only PATH...` applies C11–C13 alone to the Python files named (a directory stands for
every `*.py` under it), for code that is not an emitted package: a skill's own scripts. A
directory sweep parses the
skills' test harnesses (`*/scripts/tests/**`, `vega-show/assets/page/tests/**`) and does not
measure them — they are neither emitted nor shipped (PLAN-AMENDMENTS A13); a file named on its
own, or a directory inside a harness tree named as the thing to measure, is measured.

The API other scripts consume: `measure_module(tree, rel) -> ModuleMeasures` (the measures),
`size_findings(tree, rel) -> list[Finding]` (C11–C13 of one module), `adapter_findings(tree,
rel, role, port)` (C14 on one adapter module), `check(...)` / `check_report(...)` (a package)
and `size_only(paths, ...)`.

Like the compiler, this is a functional core (`check(...) -> Markers`) behind an imperative
shell: the rules read an AST, never argv or stdout. The core lives in its support modules —
`conformance_sizes.py` (the measures and C11–C13 of one module, the A13 scope),
`conformance_package.py` (the parsed package and the gate's options), `conformance_map.py`
(C1–C5), `conformance_code.py` (C6–C10) and `conformance_discipline.py` (C11–C14 over a
package) — and this module re-exports their surface, so a caller imports `conformance` alone.

Exit codes: 0 conformant; 2 violations (an `ERROR:` for the stage); 3 an input was unreadable.

`--json` prints one JSON document and nothing else, `{mechanical, markers, size: {severity,
counts, findings}}` — an unreadable input included, whose `IO` markers the document carries
before the exit 3 (a usage error is argparse's, on stderr, exit 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from conformance_code import (MUTABLE_NODES, c6_domain_is_frozen,  # noqa: E402
                              c7_no_module_level_mutable_state, c8_datasource_seals_the_test,
                              c9_one_constructor, c10_transformer_protocol)
from conformance_discipline import (C14_EXEMPT_ROLES, ROLE_CLASS, adapter_class,  # noqa: E402
                                    adapter_findings, c11_findings, c11_short_methods,
                                    c12_findings, c12_few_parameters, c13_findings,
                                    c13_small_classes_and_modules, c14_adapters_are_classes,
                                    c14_findings, top_level_findings)
from conformance_map import (c1_every_port_has_one_adapter, c2_extras_are_baselines,  # noqa: E402
                             c3_every_file_is_mapped, c4_imports_follow_the_arrows,
                             c5_adapters_depend_on_ports)
from conformance_package import (ENTRYPOINT_GLOBS, SUPPORT_FILES, GateOptions,  # noqa: E402
                                 Module, Package, parse_file)
from conformance_sizes import (HARNESS_TREES, LIMITS, MODULE_LENGTH_EXEMPT,  # noqa: E402
                               SIZE_CLAUSE, ClassMeasure, Finding, FunctionMeasure,
                               ModuleMeasures, SizeLimits, harness_root, measure_module,
                               size_findings, size_level, swept_exempt)
from vega_core import Level, Markers, error, ok, read_json, warn  # noqa: E402

__all__ = ["C14_EXEMPT_ROLES", "CHECKS", "DEFAULT_SIZE_SEVERITY", "ENTRYPOINT_GLOBS",
           "HARNESS_TREES", "LIMITS", "MODULE_LENGTH_EXEMPT", "MUTABLE_NODES", "ROLE_CLASS",
           "SIZE_CLAUSE", "SUPPORT_FILES", "ClassMeasure", "Finding", "FunctionMeasure",
           "GateOptions", "Module", "ModuleMeasures", "Package", "SizeLimits", "adapter_class",
           "adapter_findings", "check", "check_report", "harness_root", "main",
           "measure_module", "size_findings", "size_only", "swept_exempt",
           *(c.__name__ for c in (c1_every_port_has_one_adapter, c2_extras_are_baselines,
                                  c3_every_file_is_mapped, c4_imports_follow_the_arrows,
                                  c5_adapters_depend_on_ports, c6_domain_is_frozen,
                                  c7_no_module_level_mutable_state,
                                  c8_datasource_seals_the_test, c9_one_constructor,
                                  c10_transformer_protocol, c11_short_methods,
                                  c12_few_parameters, c13_small_classes_and_modules,
                                  c14_adapters_are_classes))]

# The severity C11–C14 findings are reported at when a caller names none (BUILD-PLAN G2).
DEFAULT_SIZE_SEVERITY = Level.ERROR

# The names the tests of the size discipline read (kept from before the split).
_top_level_findings = top_level_findings
_size_level = size_level

CHECKS = (c1_every_port_has_one_adapter, c2_extras_are_baselines, c3_every_file_is_mapped,
          c4_imports_follow_the_arrows, c5_adapters_depend_on_ports, c6_domain_is_frozen,
          c7_no_module_level_mutable_state, c8_datasource_seals_the_test, c9_one_constructor,
          c10_transformer_protocol, c11_short_methods, c12_few_parameters,
          c13_small_classes_and_modules, c14_adapters_are_classes)


def _severity_note(level: Level, findings) -> list:
    if not findings or level is not Level.WARN:
        return []
    return [warn(f"{len(findings)} size finding(s) (C11–C14) are reported as WARN under "
                 "--size-severity warn, which this run asked for; the default, --size-severity "
                 "error, refuses each of them", rule="C11-C14", clause=SIZE_CLAUSE)]


def check_report(package_dir, bmap, pipeline, options: GateOptions | None = None) -> tuple:
    """(Markers, the C11–C14 findings): the functional core, with the size findings kept as
    values for a caller that records them (`--json`, a matrix)."""
    pkg, err = Package.read(package_dir, bmap, pipeline, options)
    if err is not None:
        return Markers.gather([err]), ()
    findings = (*c11_findings(pkg), *c12_findings(pkg), *c13_findings(pkg), *c14_findings(pkg))
    markers = Markers.gather([c(pkg) for c in CHECKS] + [_severity_note(pkg.size_level,
                                                                         findings)])
    return markers, findings


def check(package_dir, bmap, pipeline, bundle=False,
          options: GateOptions | None = None) -> Markers:
    """The functional core: a package and the world model that authorises it, markers out.
    `options`, when given, carries `bundle` as well."""
    return check_report(package_dir, bmap, pipeline, options or GateOptions(bundle=bundle))[0]


# --------------------------------------------------------------------------- --size-only

def _python_files(paths) -> tuple:
    """(rel, path, harness) for every `*.py` a path names; a directory stands for every file
    under it, and `harness` says a file the sweep reached is a test harness's (A13)."""
    out = []
    for given in map(Path, paths):
        if given.is_dir():
            out += [(p.as_posix(), p, swept_exempt(p, given)) for p in sorted(given.rglob("*.py"))
                    if "__pycache__" not in p.parts]
        else:
            out.append((given.as_posix(), given, False))
    return tuple(out)


def _harness_note(exempt: int) -> list:
    if not exempt:
        return []
    return [ok(f"{exempt} file(s) of the skills' test harnesses (*/scripts/tests/**, "
               "vega-show/assets/page/tests/**) parse and are not measured: they are neither "
               "emitted nor shipped (PLAN-AMENDMENTS A13)", rule="C11-C13", clause=SIZE_CLAUSE)]


def size_only(paths, size_severity=DEFAULT_SIZE_SEVERITY) -> tuple:
    """(Markers, findings) of C11–C13 over files that are not an emitted package."""
    level, markers, findings, exempt = size_level(size_severity), [], [], 0
    files = _python_files(paths)
    for rel, path, harness in files:
        if not path.is_file():
            markers.append(error(f"no such file or directory: {rel}", rule="IO"))
            continue
        module, problem = parse_file(path, rel)
        if problem is not None:
            markers.append(problem)
        elif harness:
            exempt += 1
        else:
            findings += size_findings(module.tree, rel)
    markers += [f.marker(level) for f in findings] + _severity_note(level, findings)
    markers += _harness_note(exempt)
    if not findings:
        markers.append(ok(f"C11–C13 hold on all {len(files) - exempt} file(s) measured",
                          rule="C11-C13", clause=SIZE_CLAUSE))
    return Markers.gather([markers]), tuple(findings)


# --------------------------------------------------------------------------- the shell

def _counts(findings) -> dict:
    return {code: sum(1 for f in findings if f.code == code)
            for code in ("C11", "C12", "C13", "C14")}


def _print_json(level: Level, markers, findings) -> None:
    print(json.dumps({"mechanical": "pass" if markers.ok else "fail",
                      "markers": markers.as_json(),
                      "size": {"severity": str(level).lower(), "counts": _counts(findings),
                               "findings": [f.as_json() for f in findings]}}, indent=2))


def _print_result(a, markers, findings, what, tail="") -> int:
    """Print (the marker lines, or one JSON document and nothing else) and choose the exit."""
    if a.json:
        _print_json(size_level(a.size_severity), markers, findings)
    else:
        for line in markers.lines(verbose=a.explain):
            print(line)
    if any(m.rule == "IO" for m in markers.errors):
        return 3
    if not markers.ok:
        if not a.json:
            print(f"ERROR: conformance failed ({len(markers.errors)} violation(s)) — the fix "
                  "goes to the backend's templates through /improve or /architect and the "
                  "package is re-emitted, never to this run's code")
        return 2
    if not a.json:
        print(f"OK: {what} passed ({len(markers.items)} check result(s), "
              f"{len(markers.warns)} warning(s)){tail}")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", help="the emitted package directory")
    ap.add_argument("--pipeline", help="the blueprint run's pipeline.json")
    ap.add_argument("--map", default=None, help="blueprint-map.json (default: in the package)")
    ap.add_argument("--bundle", action="store_true",
                    help="a /vega-pack bundle: `out of scope: <reason>` becomes legal")
    ap.add_argument("--entrypoints", nargs="?", const="", default=None, metavar="ROOT",
                    help="also apply C11–C13 to predict.py, serve/**/*.py, tests/**/*.py and "
                         "the map's entrypoints under ROOT (default: the package's parent)")
    ap.add_argument("--size-only", nargs="+", default=None, metavar="PATH",
                    help="apply C11–C13 alone to these Python files or directories (a "
                         "directory sweep does not measure the skills' test harnesses)")
    ap.add_argument("--size-severity", choices=("error", "warn"),
                    default=str(DEFAULT_SIZE_SEVERITY).lower(),
                    help="how C11–C14 findings are reported (default: error)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--explain", action="store_true")
    return ap


def _arguments(argv):
    ap = _parser()
    a = ap.parse_args(argv)
    if a.size_only is not None and (a.package or a.pipeline or a.entrypoints is not None):
        ap.error("--size-only measures files outside a package; it takes no --package, "
                 "--pipeline or --entrypoints")
    if a.size_only is None and not (a.package and a.pipeline):
        ap.error("--package and --pipeline are required (or --size-only PATH...)")
    return a


def main(argv=None):
    a = _arguments(argv)
    if a.size_only is not None:
        markers, findings = size_only(a.size_only, a.size_severity)
        return _print_result(a, markers, findings, "the size discipline (C11–C13)")

    map_path = a.map or str(Path(a.package) / "blueprint-map.json")
    bmap, map_err = read_json(map_path, "blueprint-map.json")
    pipeline, pipe_err = read_json(a.pipeline, "pipeline.json")
    if map_err is not None or pipe_err is not None:     # IO markers: printed, then exit 3
        return _print_result(a, Markers.of(map_err, pipe_err), (), "mechanical conformance")
    root = None
    if a.entrypoints is not None:
        root = a.entrypoints or str(Path(a.package).resolve().parent)
    options = GateOptions(bundle=a.bundle, size_severity=a.size_severity, entrypoints=root)
    markers, findings = check_report(a.package, bmap, pipeline, options)
    return _print_result(a, markers, findings, "mechanical conformance",
                         " — now run `/architect <package> --report-only --baseline "
                         "<blueprint.md>` for the judgement half")


if __name__ == "__main__":
    sys.exit(main())
