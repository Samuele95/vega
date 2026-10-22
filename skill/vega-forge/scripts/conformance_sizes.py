"""
conformance_sizes — the size discipline C11–C13 measured on one module's AST (code-contract.md
§5.1), and the scope rule that says which files it reads.

A support module of `conformance.py`, which re-exports everything named in `__all__`; import
it from there. The measures are D6's `CONFORMANCE_MEASURES` (`measure_module`), the findings
are the contract's lines (`size_findings`), and the thresholds are the contract's, not flags.

Counting (R-G1):

  * a statement is an `ast.stmt`, counted recursively, without the docstring; `elif` is the
    `If` the AST makes of it; a nested `def` or `class` counts once in its parent and is
    measured on its own;
  * parameters are counted without `self` / `cls`; `*args` and `**kwargs` count one each;
  * a class's lines run from its first decorator to its last statement, a module's from line
    1 to its last statement;
  * a `_prefixed` def or class — and whatever is nested inside one — is a helper; a dunder is
    not `_prefixed`, it is a public protocol method.

Scope (PLAN-AMENDMENTS A13): the skills' test harnesses — every file under
`<skill>/scripts/tests/**` and `vega-show/assets/page/tests/**` — are neither emitted nor
shipped, so a sweep of a skill's scripts (`--size-only <dir>`) parses them (C0) and does not
measure them. A file named on its own, or a directory inside a harness tree named as the thing
to measure (the conformance fixtures), is measured: the exemption is a sweep's, never a
caller's explicit request.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from vega_core import Level, Marker

__all__ = ["HARNESS_TREES", "LIMITS", "MODULE_LENGTH_EXEMPT", "SIZE_CLAUSE", "ClassMeasure",
           "Finding", "FunctionMeasure", "ModuleMeasures", "SizeLimits", "harness_root",
           "is_helper_name", "is_docstring", "measure_module", "size_findings", "size_level",
           "c11_of", "c12_of", "c13_of", "swept_exempt"]

SIZE_CLAUSE = "code-contract.md §5.1"


@dataclass(frozen=True)
class SizeLimits:
    """The contract's thresholds (code-contract.md §5.1). They are not flags: a template that
    needs 31 statements needs a second method."""

    statements: int = 30
    parameters: int = 5
    class_lines: int = 200
    module_lines: int = 400


LIMITS = SizeLimits()

# A library of small pure functions and a set of small value objects are long by count, not by
# cohesion: the module-length rule exempts them by file name (never C11–C12).
MODULE_LENGTH_EXEMPT = frozenset({"domain.py", "metrics.py"})

# The skills' test harnesses (A13): path segments that open a harness tree. `scripts/tests`
# must sit under a skill directory (`*/scripts/tests/**`).
HARNESS_TREES = (("scripts", "tests"), ("vega-show", "assets", "page", "tests"))

_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


# --------------------------------------------------------------------------- the measures

@dataclass(frozen=True)
class FunctionMeasure:
    """One def, measured: `symbol` is `Class.method` for a method and the bare name for a
    module-level or a nested function (a nested def is measured on its own, not as part of
    the function that holds it)."""

    symbol: str
    line: int
    statements: int
    parameters: int
    variadic: tuple
    helper: bool


@dataclass(frozen=True)
class ClassMeasure:
    """One class, measured from its first decorator to its last statement."""

    symbol: str
    line: int
    lines: int
    helper: bool


@dataclass(frozen=True)
class ModuleMeasures:
    """Everything C11–C13 read in one module (D6's `CONFORMANCE_MEASURES`)."""

    rel: str
    lines: int
    functions: tuple
    classes: tuple

    def function(self, symbol: str):
        return next((f for f in self.functions if f.symbol == symbol), None)

    def klass(self, symbol: str):
        return next((c for c in self.classes if c.symbol == symbol), None)


@dataclass(frozen=True)
class Finding:
    """One size-discipline violation, in the contract's words: `C11 <file>:<line> <symbol> —
    <n> statements > 30 — EJ-51`. A measured finding carries `measured`, `limit` and `unit`;
    a rule finding (a public `*args`, a second public class) carries `detail` instead."""

    code: str
    file: str
    line: int
    symbol: str
    anchor: str
    measured: int | None = None
    limit: int | None = None
    unit: str = ""
    detail: str = ""

    def __post_init__(self):
        if self.code not in ("C11", "C12", "C13", "C14"):
            raise ValueError(f"a size finding is C11–C14, not {self.code!r}")
        if (self.measured is None) == (not self.detail):
            raise ValueError("a finding is either measured (measured/limit/unit) or a rule "
                             "(detail), exactly one of the two")

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line}" if self.line else self.file

    def message(self) -> str:
        body = (f"{self.measured} {self.unit} > {self.limit}" if self.measured is not None
                else self.detail)
        return f"{self.code} {self.where} {self.symbol} — {body} — {self.anchor}"

    def marker(self, level: Level) -> Marker:
        return Marker(level, self.message(), rule=self.code, clause=SIZE_CLAUSE)

    def as_json(self) -> dict:
        return {"code": self.code, "file": self.file, "line": self.line,
                "symbol": self.symbol, "measured": self.measured, "limit": self.limit,
                "unit": self.unit or None, "detail": self.detail or None,
                "anchor": self.anchor, "message": self.message()}


@dataclass(frozen=True)
class _Scope:
    """Where a def sits: the enclosing class path (empty at module level and inside a
    function, which starts a fresh naming scope) and whether a `_prefixed` scope encloses it."""

    qual: str = ""
    helper: bool = False


def is_helper_name(name: str) -> bool:
    """`_x` and `__x` are private; `__x__` is a public protocol method, not a helper."""
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def is_docstring(node) -> bool:
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))


def _count_statements(body) -> int:
    """`ast.stmt`s, recursively; a nested def or class is one statement of its parent."""
    total = 0
    for node in body:
        total += 1
        if isinstance(node, _SCOPE_NODES):
            continue
        for field in ("body", "orelse", "finalbody"):
            total += _count_statements(getattr(node, field, None) or ())
        parts = [*(getattr(node, "handlers", None) or ()), *(getattr(node, "cases", None) or ())]
        for part in parts:
            total += _count_statements(part.body)
    return total


def _statements(fn) -> int:
    body = fn.body[1:] if fn.body and is_docstring(fn.body[0]) else fn.body
    return _count_statements(body)


def _decorator_names(node) -> set:
    names = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        names.add(target.id if isinstance(target, ast.Name) else getattr(target, "attr", ""))
    return names


def _parameters(fn) -> tuple:
    """(count, variadic): parameters beyond `self` / `cls`, `*args` and `**kwargs` one each."""
    a = fn.args
    positional = [*a.posonlyargs, *a.args]
    bound = bool(positional) and positional[0].arg in ("self", "cls") \
        and "staticmethod" not in _decorator_names(fn)
    variadic = tuple(f"{star}{p.arg}" for star, p in (("*", a.vararg), ("**", a.kwarg))
                     if p is not None)
    return len(positional) - int(bound) + len(a.kwonlyargs) + len(variadic), variadic


def _qualified(scope: _Scope, name: str) -> tuple:
    """(symbol, helper) of a def or class named `name` inside `scope`."""
    symbol = f"{scope.qual}.{name}" if scope.qual else name
    return symbol, scope.helper or is_helper_name(name)


def _collect(node, scope: _Scope, functions: list, classes: list) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _FUNCTION_NODES):
            symbol, helper = _qualified(scope, child.name)
            count, variadic = _parameters(child)
            functions.append(FunctionMeasure(symbol, child.lineno, _statements(child), count,
                                             variadic, helper))
            _collect(child, _Scope("", helper), functions, classes)
        elif isinstance(child, ast.ClassDef):
            symbol, helper = _qualified(scope, child.name)
            first = min([d.lineno for d in child.decorator_list] + [child.lineno])
            classes.append(ClassMeasure(symbol, child.lineno, child.end_lineno - first + 1,
                                        helper))
            _collect(child, _Scope(symbol, helper), functions, classes)
        else:
            _collect(child, scope, functions, classes)


def measure_module(tree: ast.Module, rel: str) -> ModuleMeasures:
    """Every def and class of one parsed module, measured by the contract's counting rule."""
    functions, classes = [], []
    _collect(tree, _Scope(), functions, classes)
    lines = max((getattr(n, "end_lineno", None) or n.lineno for n in tree.body), default=0)
    return ModuleMeasures(rel, lines, tuple(functions), tuple(classes))


# --------------------------------------------------------------------------- the findings

def c11_of(m: ModuleMeasures) -> list:
    return [Finding("C11", m.rel, f.line, f.symbol, "EJ-51", f.statements, LIMITS.statements,
                    "statements") for f in m.functions if f.statements > LIMITS.statements]


def c12_of(m: ModuleMeasures) -> list:
    out = []
    for f in m.functions:
        if f.parameters > LIMITS.parameters:
            out.append(Finding("C12", m.rel, f.line, f.symbol, "EJ-51", f.parameters,
                               LIMITS.parameters, "parameters"))
        if f.variadic and not f.helper:
            out.append(Finding(
                "C12", m.rel, f.line, f.symbol, "EJ-51",
                detail=f"{', '.join(f.variadic)} in a public signature; variadic parameters "
                       "are allowed only in a _prefixed helper — name them, or pass a value "
                       "object"))
    return out


def c13_of(m: ModuleMeasures) -> list:
    out = [Finding("C13", m.rel, c.line, f"class {c.symbol}", "EJ-25", c.lines,
                   LIMITS.class_lines, "lines")
           for c in m.classes if c.lines > LIMITS.class_lines]
    if m.lines > LIMITS.module_lines and PurePosixPath(m.rel).name not in MODULE_LENGTH_EXEMPT:
        out.append(Finding("C13", m.rel, 1, "module", "EJ-25", m.lines, LIMITS.module_lines,
                           "lines"))
    return out


def size_findings(tree: ast.Module, rel: str) -> list:
    """C11–C13 of one module (C14 needs the map and the pipeline, so it is not here)."""
    measures = measure_module(tree, rel)
    return [*c11_of(measures), *c12_of(measures), *c13_of(measures)]


def size_level(value) -> Level:
    """`warn` / `error` (or the Level) -> the Level C11–C14 findings are reported at."""
    if isinstance(value, Level) and value in (Level.WARN, Level.ERROR):
        return value
    named = {"warn": Level.WARN, "error": Level.ERROR}.get(str(value).lower())
    if named is None:
        raise ValueError(f"--size-severity is warn or error, not {value!r}")
    return named


# --------------------------------------------------------------------------- the scope (A13)

def harness_root(path) -> Path | None:
    """The harness tree a file sits in (`<skill>/scripts/tests`, `vega-show/assets/page/tests`),
    or None when it is emitted or shipped code."""
    parts = Path(path).resolve().parts
    for tail in HARNESS_TREES:
        n = len(tail)
        for i in range(1, len(parts) - n + 1):
            if parts[i:i + n] == tail:
                return Path(*parts[:i + n])
    return None


def swept_exempt(path, swept_from) -> bool:
    """Whether a file a directory sweep reached is a harness file the sweep does not measure:
    it sits in a harness tree, and the swept directory is not itself inside that tree."""
    root = harness_root(path)
    if root is None:
        return False
    start = Path(swept_from).resolve()
    return not (start == root or root in start.parents)
