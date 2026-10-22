"""
The non-lane tiers of the program (D6 §2.1): T0 static, T1 unit, T3 negatives, T4 report, each
a list of `Component`s — one command each — run against the matrix's skills tree.

- static: every `test_catalog_sync.py`, the registry validation, and the static checks later
  packages add (the declared-coverage check T12, the literal lint, the style lock,
  `test_no_planned.py`, `check_unattended.py`, `verifier_gate.py` — T04);
- unit: every `test_*.py` under a skill's `scripts/tests/` (pytest when it defines test
  functions, else run as a script), the backend smoke cells (B19);
- negatives: every `run_negatives*.py` — the four existing suites on L01's work dir, the new
  ones on the matrix dir — with `--json` where the suite accepts it (T10 adds it);
- report: the coverage and adaptation reports (T12).

A component whose file is not in the tree yet is `absent`, named with its provider; the caller
decides whether that fails the tier (strict) or is reported.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from . import contracts

EXISTING_NEGATIVES = ("vega-forge", "vega-train", "vega-eval", "vega-pack")


@dataclass(frozen=True)
class Component:
    """One command of a static / unit / negatives / report tier."""
    id: str
    argv: tuple[str, ...]           # {python} {skills} {tests} {matrix} {work} {tier} placeholders
    provider: str
    needs: tuple[str, ...] = ()     # skills-relative paths that must exist

    def resolve(self, fields: Mapping[str, str]) -> list[str]:
        return [a.format(**fields) for a in self.argv]


def discover(skills: Path, pattern: str, exclude: Iterable[str] = ()) -> list[Path]:
    skip = set(exclude)
    found = sorted(p for p in skills.glob(pattern) if p.is_file() and p.name not in skip)
    return [p for p in found if "domain-forge" not in p.parts]


def static_components(skills: Path) -> list[Component]:
    tests = contracts.TESTS_DIR
    comps = [Component(f"catalog-sync:{p.relative_to(skills).parts[0]}", ("{python}", str(p)),
                       p.relative_to(skills).parts[0])
             for p in discover(skills, "*/scripts/tests/test_catalog_sync.py")]
    comps += [Component("registry", ("{python}", "{tests}/run_matrix.py", "validate", "--skills", "{skills}"), "T01"),
              Component("coverage-declared", ("{python}", "{tests}/coverage_report.py", "--matrix", "{matrix}",
                                              "--declared-only"), "T12", (f"{tests}/coverage_report.py",)),
              Component("lint-templates", ("{python}", "{tests}/lint_templates.py"), "T04", (f"{tests}/lint_templates.py",)),
              Component("style-lock", ("{python}", "{tests}/style_lock.py", "check"), "T04", (f"{tests}/style_lock.py",)),
              Component("no-planned", ("{python}", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                       "{skills}/vega-forge/scripts/tests/test_no_planned.py"), "T04",
                        ("vega-forge/scripts/tests/test_no_planned.py",)),
              Component("check-unattended", ("{python}", "{tests}/check_unattended.py"), "T04", (f"{tests}/check_unattended.py",)),
              Component("verifier-gate", ("{python}", "{tests}/verifier_gate.py"), "T04", (f"{tests}/verifier_gate.py",))]
    return comps


def unit_components(skills: Path) -> list[Component]:
    """Every unit test file the package's skills hold (pytest when it defines test functions,
    else run as a script), the backend smoke cells, and this program's own tests."""
    comps = []
    for path in discover(skills, "*/scripts/tests/**/test_*.py", exclude=("test_catalog_sync.py", "test_no_planned.py")):
        rel = str(path.relative_to(skills))
        pytest_style = re.search(r"^def test_", path.read_text(encoding="utf-8", errors="replace"), re.M)
        argv = (("{python}", "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)) if pytest_style
                else ("{python}", str(path)))
        comps.append(Component(f"unit:{rel}", argv, rel.split("/")[0]))
    comps.append(Component("backend-smoke-cells", ("{python}", "{skills}/vega-forge/scripts/tests/matrix/run_matrix.py",
                                                   "--cells", "smoke"), "B19",
                           ("vega-forge/scripts/tests/matrix/run_matrix.py",)))
    return comps


def negative_components(skills: Path, matrix: Path) -> list[Component]:
    """The four existing suites run on L01's work dir; the new ones on the matrix dir. `--json`
    is passed where the suite accepts it (T10 adds it)."""
    comps = []
    for path in discover(skills, "*/scripts/tests/run_negatives*.py"):
        rel = path.relative_to(skills)
        existing = rel.parts[0] in EXISTING_NEGATIVES and path.name == "run_negatives.py"
        target = "{work}" if existing else "{matrix}"
        argv = ["{python}", str(path), target]
        if '"--json"' in path.read_text(encoding="utf-8", errors="replace"):
            argv += ["--json", f"{{matrix}}/negatives/{rel.parts[0]}-{path.stem}.json"]
        comps.append(Component(f"negatives:{rel.parts[0]}/{path.name}", tuple(argv), rel.parts[0]))
    return comps


def report_components(tier: str | None, pairs: Sequence[str]) -> list[Component]:
    tests = contracts.TESTS_DIR
    return [Component("coverage", ("{python}", "{tests}/coverage_report.py", "--matrix", "{matrix}", "--tier",
                                   "{tier}", "--json", "{matrix}/coverage.json", "--md", "{matrix}/coverage.md"),
                      "T12", (f"{tests}/coverage_report.py",)),
            Component("adaptation", ("{python}", "{tests}/adaptation_report.py", "--matrix", "{matrix}",
                                     "--pairs", ",".join(pairs)), "T12", (f"{tests}/adaptation_report.py",))]


@dataclass
class ComponentOutcome:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass(frozen=True)
class TierRun:
    """Where a tier's components run: the skills tree, the environment, the placeholder values
    ({python} {skills} {tests} {matrix} {work} {tier}) and the directory for their logs."""
    skills: Path
    env: Mapping[str, str]
    fields: Mapping[str, str]
    log_dir: Path


def run_components(run: TierRun, name: str, comps: Sequence[Component],
                   say: Callable[[str], None] = print) -> ComponentOutcome:
    out, started = ComponentOutcome(), time.monotonic()
    run.log_dir.mkdir(parents=True, exist_ok=True)
    for comp in comps:
        missing = [n for n in comp.needs if not (run.skills / n).exists()]
        if missing:
            out.absent.append(f"{comp.id} (needs {missing[0]}, {comp.provider})")
            say(f"WARN: {name}: {comp.id} absent — needs {missing[0]} ({comp.provider})")
            continue
        log = run.log_dir / (re.sub(r"[^\w.-]+", "_", comp.id) + ".log")
        with open(log, "w", encoding="utf-8") as fh:
            code = subprocess.run(comp.resolve(run.fields), cwd=run.skills, env=dict(run.env),
                                  stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT).returncode
        (out.passed if code == 0 else out.failed).append(comp.id)
        say(f"{'OK' if code == 0 else 'ERROR'}: {name}: {comp.id}" + ("" if code == 0 else f" exited {code} [{log}]"))
    out.seconds = time.monotonic() - started
    return out
