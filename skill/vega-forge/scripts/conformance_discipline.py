"""
conformance_discipline — C11–C14 over a package: the size findings of every file the gate
reads (C11–C13, measured by `conformance_sizes`), and C14, the class discipline of the port
adapters and the paradigm of the modules that own ports.

A support module of `conformance.py`, which re-exports everything named in `__all__`; import
it from there. `adapter_findings(tree, rel, role, port)` is C14's rule on one module, for a
caller that holds a rendered template rather than a package (the serving templates' test).
"""
from __future__ import annotations

import ast
from types import MappingProxyType
from typing import Mapping

from conformance_package import Package
from conformance_sizes import (LIMITS, SIZE_CLAUSE, Finding, c11_of, c12_of, c13_of,
                               is_docstring, is_helper_name)
from vega_core import ok

__all__ = ["C14_EXEMPT_ROLES", "ROLE_CLASS", "adapter_class", "adapter_findings",
           "c11_findings", "c12_findings", "c13_findings", "c14_findings",
           "c11_short_methods", "c12_few_parameters", "c13_small_classes_and_modules",
           "c14_adapters_are_classes", "top_level_findings"]

# The adapter class each role's template defines (the blueprint names the Protocol, the
# backend names the class that satisfies it structurally). The FeatureTransformer has none: its
# adapter is the shaper's module of functions, which C10 reads and C14 exempts.
ROLE_CLASS = MappingProxyType({
    "DataSource": "ShapedParts", "Model": "Network", "ModelStore": "ModelRegistry",
    "Trainer": "Trainer", "Evaluator": "Evaluator", "Predictor": "Predictor",
    "Monitor": "Monitor"})

# The map roles C14's class rule and top-level rule do not read (C11–C13 still do):
#   baseline            — the extra implementations of a port, several classes and a public
#                         build() by design (C2; D1 §11.3);
#   FeatureTransformer  — the shaper's transform-only module, whose public module functions
#                         `transform`, `edges_for`, `edges_within` are the port (R-A3, D2
#                         §4.4); C10 checks that shape structurally.
C14_EXEMPT_ROLES = frozenset({"baseline", "FeatureTransformer"})

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_BLOCK_NODES = tuple(getattr(ast, n) for n in ("If", "Try", "TryStar") if hasattr(ast, n))


# --------------------------------------------------------------------------- C11–C13

def _size_markers(pkg: Package, findings, code: str, clean: str) -> list:
    if not findings:
        return [ok(clean, rule=code, clause=SIZE_CLAUSE)]
    return [f.marker(pkg.size_level) for f in findings]


def c11_findings(pkg: Package) -> list:
    return [f for m in pkg.measures() for f in c11_of(m)]


def c12_findings(pkg: Package) -> list:
    return [f for m in pkg.measures() for f in c12_of(m)]


def c13_findings(pkg: Package) -> list:
    return [f for m in pkg.measures() for f in c13_of(m)]


def c11_short_methods(pkg: Package):
    n = sum(len(m.functions) for m in pkg.measures())
    return _size_markers(pkg, c11_findings(pkg), "C11",
                         f"C11 all {n} function(s) and method(s) are within "
                         f"{LIMITS.statements} statements ({pkg.scope_note()})")


def c12_few_parameters(pkg: Package):
    n = sum(len(m.functions) for m in pkg.measures())
    return _size_markers(pkg, c12_findings(pkg), "C12",
                         f"C12 all {n} function(s) and method(s) take at most "
                         f"{LIMITS.parameters} parameters beyond self/cls, and *args/**kwargs "
                         f"appear only in _prefixed helpers ({pkg.scope_note()})")


def c13_small_classes_and_modules(pkg: Package):
    return _size_markers(pkg, c13_findings(pkg), "C13",
                         f"C13 every class is within {LIMITS.class_lines} lines and every "
                         f"module within {LIMITS.module_lines} (domain.py and metrics.py "
                         f"exempt from the module rule; {pkg.scope_note()})")


# --------------------------------------------------------------------------- C14, one module

def _dunder_all(tree: ast.Module) -> set:
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__"
                                                 for t in node.targets)
                and isinstance(node.value, (ast.List, ast.Tuple))):
            return {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    return set()


def adapter_class(tree: ast.Module, role: str, port: str | None = None):
    """The top-level class that is a port's adapter, or None.

    In order: a class with a base named after the port or the role (`class
    ShapedParts(DataSource)`, `class Trainer(ports.Trainer)`); a class named as the role's
    template names it (ROLE_CLASS), or after the port or the role; the one public class that
    `__all__` exports."""
    classes = [n for n in tree.body
               if isinstance(n, ast.ClassDef) and not is_helper_name(n.name)]
    names = {x for x in (port, role) if x}
    for node in classes:
        bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases}
        if names & bases:
            return node
    wanted = names | {ROLE_CLASS.get(role)}
    for node in classes:
        if node.name in wanted:
            return node
    exported = [n for n in classes if n.name in _dunder_all(tree)]
    return exported[0] if len(exported) == 1 else None


def _is_constant_name(name: str) -> bool:
    """A dunder (`__all__`), a `_prefixed` helper value, or an UPPER_CASE constant."""
    return ((name.startswith("__") and name.endswith("__")) or is_helper_name(name)
            or name.lstrip("_").isupper())


def _is_build_alias(node, adapter) -> bool:
    """`build_model = Network.build`: the name C9 reads, bound to the adapter's own factory."""
    return (adapter is not None and isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "build_model"
            and isinstance(node.value, ast.Attribute) and node.value.attr == "build"
            and isinstance(node.value.value, ast.Name) and node.value.value.id == adapter.name)


def _assigned_names(node) -> list | None:
    """The plain names an assignment binds, or None when it assigns into something else."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = []
    for target in targets:
        parts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        if not all(isinstance(p, ast.Name) for p in parts):
            return None
        names += [p.id for p in parts]
    return names


def _stray(rel, node, symbol, what) -> Finding:
    return Finding("C14", rel, node.lineno, symbol, "EJ-25",
                   detail=f"{what}; an adapter module's top level holds only its adapter "
                          "class, _prefixed helpers, imports and constants")


def _function_findings(rel: str, node, role: str) -> list:
    if is_helper_name(node.name):
        return []
    hint = (" (`build_model = Network.build` is the module-level form C9 and C14 read)"
            if node.name == "build_model" else "")
    return [Finding("C14", rel, node.lineno, node.name, "EJ-15",
                    detail=f"a public function at the top level of the {role} adapter "
                           f"module; module-level functions are _prefixed helpers{hint}")]


def _assignment_findings(rel: str, node, adapter) -> list:
    if _is_build_alias(node, adapter):
        return []
    names = _assigned_names(node)
    if names is None:
        return [_stray(rel, node, "assignment", "a top-level assignment into another object")]
    return [_stray(rel, node, n, "a public module-level name that is neither an UPPER_CASE "
                                 "constant nor a _prefixed helper")
            for n in names if not _is_constant_name(n)]


def top_level_findings(rel: str, node, adapter, role: str) -> list:
    """What one top-level statement of an adapter module does to C14."""
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)) or is_docstring(node):
        return []
    if isinstance(node, _FUNCTION_NODES):
        return _function_findings(rel, node, role)
    if isinstance(node, ast.ClassDef):
        if node is adapter or is_helper_name(node.name):
            return []
        return [_stray(rel, node, node.name,
                       f"a public class that is not the {role} adapter class")]
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return _assignment_findings(rel, node, adapter)
    if isinstance(node, _BLOCK_NODES):          # import guards: TYPE_CHECKING, optional deps
        inner = [*node.body, *getattr(node, "orelse", ()), *getattr(node, "finalbody", ()),
                 *(s for h in getattr(node, "handlers", ()) for s in h.body)]
        return [f for s in inner for f in top_level_findings(rel, s, adapter, role)]
    return [_stray(rel, node, type(node).__name__.lower(),
                   "an executable statement at module level")]


def adapter_findings(tree: ast.Module, rel: str, role: str, port: str | None = None) -> list:
    """C14 on one implemented primary adapter's module: a class, alone at its module's top
    level with `_prefixed` helpers, imports and constants."""
    adapter = adapter_class(tree, role, port)
    out = []
    if adapter is None:
        out.append(Finding("C14", rel, 1, str(port or role), "EJ-20",
                           detail=f"the adapter of port {port!r} (role {role}) is not a class; a "
                                  "port is a type, and what satisfies it is an object that "
                                  "guards a state"))
    for node in tree.body:
        out += top_level_findings(rel, node, adapter, role)
    return out


# --------------------------------------------------------------------------- C14, the package

def _module_in_slice(pkg: Package, name: str) -> bool:
    return any(e.get("module") == name
               and not str(e.get("status", "")).startswith("out of scope")
               for e in pkg.entries())


def _paradigm_findings(pkg: Package) -> list:
    out = []
    for mod in pkg.pipeline.get("modules") or []:
        if not (isinstance(mod, Mapping) and mod.get("ports_owned")):
            continue
        name, paradigm = mod.get("name"), mod.get("paradigm")
        if pkg.bundle and not _module_in_slice(pkg, name):
            continue                               # not in this bundle's serving slice
        if str(paradigm or "").upper() != "OOP":
            owned = ", ".join(map(str, mod["ports_owned"]))
            out.append(Finding("C14", "pipeline.json", 0, f"modules[{name}]", "EJ-20",
                               detail=f"paradigm {paradigm!r} while it owns port(s) {owned}; a "
                                      "module that owns a port is OOP (the blueprint's to "
                                      "change: /blueprint with the patched SKILL.md)"))
    return out


def _held_to_c14(entry) -> bool:
    """An implemented primary adapter whose role C14's class rule reads. A `pending: <stage>`
    stub or an `out of scope` entry is not an implementation yet; the roles of
    C14_EXEMPT_ROLES are not class-shaped by design."""
    return entry.get("role") not in C14_EXEMPT_ROLES and entry.get("status") == "implemented"


def c14_findings(pkg: Package) -> list:
    """Paradigms first, then every implemented primary adapter C14 holds to the class rule."""
    out = _paradigm_findings(pkg)
    for entry in pkg.adapter_entries():
        module = pkg.module_by_rel(entry.get("file")) if _held_to_c14(entry) else None
        if module is not None:                     # a missing file is C3's to name
            out += adapter_findings(module.tree, module.rel, entry.get("role"),
                                    entry.get("port"))
    return out


def c14_adapters_are_classes(pkg: Package):
    n = sum(1 for e in pkg.adapter_entries() if _held_to_c14(e))
    return _size_markers(pkg, c14_findings(pkg), "C14",
                         f"C14 every port-owning module is OOP and each of the {n} implemented "
                         "adapter(s) is a class alone at its module's top level with _prefixed "
                         "helpers, imports and constants (the FeatureTransformer's function "
                         "module is C10's)")
