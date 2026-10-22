"""
conformance_package — the emitted package as the gate reads it: every file parsed once (C0),
the map and the blueprint that authorise it, the entrypoints C11–C13 also measure, and the
options of one run of the gate.

A support module of `conformance.py`, which re-exports everything named in `__all__`; import
it from there. Nothing here decides a check: it answers the questions the checks ask.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from conformance_sizes import measure_module, size_level
from vega_core import Level, Markers, error

__all__ = ["ENTRYPOINT_GLOBS", "SUPPORT_FILES", "GateOptions", "Module", "Package",
           "defs_named", "entrypoint_paths", "functions_of", "imports_of", "parse_file",
           "parse_all", "raises_names", "target_module"]

SUPPORT_FILES = ("domain.py", "ports.py", "__init__.py")

# What a stage writes outside the package, relative to the bundle root (C11–C13 only).
ENTRYPOINT_GLOBS = ("predict.py", "serve/**/*.py", "tests/**/*.py")

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(frozen=True)
class GateOptions:
    """How one run of the gate reads a package: as a bundle or not, the severity C11–C14 are
    reported at (`error` by default: the templates meet the limits since W7), and the root
    whose entrypoints C11–C13 also measure (None: not measured)."""

    bundle: bool = False
    size_severity: Level = Level.ERROR
    entrypoints: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "size_severity", size_level(self.size_severity))
        if self.entrypoints is not None:
            object.__setattr__(self, "entrypoints", Path(self.entrypoints))


@dataclass(frozen=True)
class Module:
    """One emitted file, parsed once."""

    rel: str                 # package-relative, as the map addresses it
    path: Path
    tree: ast.Module
    source: str

    @property
    def package_module(self) -> str:
        return self.rel[:-3].replace("/", ".")

    @property
    def module_name(self) -> str:
        return self.rel.split("/")[0] if "/" in self.rel else "domain"


def parse_file(path: Path, rel: str) -> tuple:
    """(Module, None) or (None, the C0 marker naming the file and the line)."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, error(f"{rel} could not be read: {exc}", rule="IO")
    try:
        return Module(rel, path, ast.parse(source, filename=str(path)), source), None
    except SyntaxError as exc:
        return None, error(f"C0 {rel}:{exc.lineno} does not parse: {exc.msg} — the emitter "
                           "wrote a file Python cannot read", rule="C0",
                           clause="code-contract.md §2")


def parse_all(root: Path, paths) -> tuple:
    modules, problems = [], []
    for path in paths:
        module, problem = parse_file(path, path.relative_to(root).as_posix())
        if module is not None:
            modules.append(module)
        if problem is not None:
            problems.append(problem)
    return tuple(modules), problems


def _manifest_entrypoints(root: Path) -> list:
    """A bundle's `MANIFEST.json` `entrypoints` (the CLIs a `full` bundle adds — `train.py`,
    `evaluate.py` — are listed there and not in the map); nothing outside a bundle."""
    try:
        doc = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    listed = doc.get("entrypoints") if isinstance(doc, dict) else None
    return [str(e) for e in listed or () if isinstance(e, str)]


def entrypoint_paths(root: Path, package: Path, bmap) -> list:
    """The files a stage writes outside the package: the globs, the map's entrypoints and a
    bundle manifest's."""
    found = {p for pattern in ENTRYPOINT_GLOBS for p in root.glob(pattern) if p.is_file()}
    listed = [str((entry or {}).get("file") or "")
              for entry in (bmap or {}).get("entrypoints") or []]
    for rel in listed + _manifest_entrypoints(root):
        if rel.endswith(".py") and (root / rel).is_file():
            found.add(root / rel)
    inside = package.resolve()
    return sorted(p for p in found if inside not in p.resolve().parents)


def _read_entrypoints(ep_root, package: Path, bmap) -> tuple:
    """(entrypoint modules, problems) under `ep_root`; nothing when it is None."""
    if ep_root is None:
        return (), []
    if not ep_root.is_dir():
        return (), [error(f"entrypoint root not found: {ep_root}", rule="IO")]
    return parse_all(ep_root, entrypoint_paths(ep_root, package, bmap))


@dataclass(frozen=True)
class Package:
    """The emitted package plus the map and the blueprint that authorise it."""

    root: Path
    modules: tuple
    bmap: Mapping[str, Any]
    pipeline: Mapping[str, Any]
    bundle: bool = False
    size_level: Level = Level.ERROR
    entrypoints: tuple = ()
    entrypoint_root: Path | None = None

    @staticmethod
    def read(root, bmap, pipeline, options: GateOptions | None = None) -> tuple:
        options = options or GateOptions()
        root = Path(root)
        if not root.is_dir():
            return None, error(f"package directory not found: {root}", rule="IO")
        modules, problems = parse_all(root, sorted(root.rglob("*.py")))
        entrypoints, more = _read_entrypoints(options.entrypoints, root, bmap)
        problems += more
        if problems:
            return None, Markers.gather([problems])
        return Package(root, modules, bmap, pipeline, options.bundle, options.size_severity,
                       entrypoints, options.entrypoints), None

    # -- the map, asked questions

    def entries(self) -> tuple:
        return tuple(self.bmap.get("entries") or ())

    def adapter_entries(self) -> tuple:
        return tuple(e for e in self.entries() if e.get("role"))

    def support_entries(self) -> tuple:
        return tuple(e for e in self.entries() if e.get("kind") == "support")

    def entry_for_role(self, role):
        return tuple(e for e in self.entries() if e.get("role") == role)

    def file_of_role(self, role):
        found = self.entry_for_role(role)
        return found[0].get("file") if found else None

    def entry_of_file(self, rel) -> Mapping:
        return next((e for e in self.entries() if e.get("file") == rel), {})

    def module_by_rel(self, rel):
        return next((m for m in self.modules if m.rel == rel), None)

    def blueprint_ports(self) -> tuple:
        return tuple(p for p in (self.pipeline.get("ports") or []) if isinstance(p, Mapping))

    def depends_on(self) -> dict:
        out = {}
        for mod in self.pipeline.get("modules") or []:
            if isinstance(mod, Mapping) and mod.get("name"):
                out[mod["name"]] = set(mod.get("depends_on") or [])
        return out

    # -- the size discipline, asked questions

    def measures(self) -> tuple:
        """The measures of every file C11–C13 read: the package's, then the entrypoints'."""
        return tuple(measure_module(m.tree, m.rel) for m in (*self.modules, *self.entrypoints))

    def scope_note(self) -> str:
        if not self.entrypoints:
            return f"{len(self.modules)} file(s) of the package"
        return (f"{len(self.modules)} file(s) of the package and {len(self.entrypoints)} "
                f"entrypoint file(s) under {self.entrypoint_root}")


# --------------------------------------------------------------------------- AST helpers

def imports_of(module: Module) -> list:
    """(module_expr, level, names, lineno) for every import in the file."""
    out = []
    for node in ast.walk(module.tree):
        if isinstance(node, ast.ImportFrom):
            out.append((node.module or "", node.level,
                        [a.name for a in node.names], node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, 0, [alias.name], node.lineno))
    return out


def target_module(module: Module, expr: str, level: int):
    """Which package module an import points at: 'domain', 'model', 'model.metrics', or None
    when it leaves the package."""
    if level == 0:
        return None                       # absolute: third-party or stdlib
    if level == 1 and "/" in module.rel:
        return f"{module.module_name}.{expr}" if expr else module.module_name
    return expr or None


def raises_names(fn: ast.FunctionDef) -> set:
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Raise) and node.exc is not None:
            call = node.exc
            if isinstance(call, ast.Call):
                call = call.func
            if isinstance(call, ast.Name):
                names.add(call.id)
            elif isinstance(call, ast.Attribute):
                names.add(call.attr)
    return names


def functions_of(module: Module) -> dict:
    return {n.name: n for n in module.tree.body if isinstance(n, ast.FunctionDef)}


def defs_named(module: Module, name: str) -> tuple:
    """Every def called `name` at module level or as a method of a top-level class — where a
    port's operation lives in a function-shaped adapter and in a class-shaped one."""
    found = [n for n in module.tree.body if isinstance(n, _FUNCTION_NODES) and n.name == name]
    for cls in (n for n in module.tree.body if isinstance(n, ast.ClassDef)):
        found += [n for n in cls.body if isinstance(n, _FUNCTION_NODES) and n.name == name]
    return tuple(found)
