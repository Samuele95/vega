"""
conformance_code — C6–C10, the checks that read the code of particular roles: frozen value
objects in `domain.py` and the blueprint's artifact roster they hold, no module-level mutable
state, the DataSource's seal, the Model's one constructor, the FeatureTransformer's structural
protocol.

A support module of `conformance.py`, which re-exports every check; import it from there.
Each check takes the parsed `Package` and returns markers — it reads no argv and prints
nothing.
"""
from __future__ import annotations

import ast
from typing import Mapping

from conformance_package import Package, defs_named, functions_of, raises_names
from vega_core import Level, error, ok, warn
from vega_roster import CLAUSE as _ROSTER, defined_names, held_vocabulary, roster

__all__ = ["MUTABLE_NODES", "c6_domain_is_frozen", "c7_no_module_level_mutable_state",
           "c8_datasource_seals_the_test", "c9_one_constructor", "c10_transformer_protocol"]

MUTABLE_NODES = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)

_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
_IMMUTABLE_BUILTINS = {"tuple", "frozenset", "str", "int", "float", "bytes"}
_VALUE = "code-contract.md §4 (EJ-17)"
_GATE = "code-contract.md §5.1"
_BUILD = "code-contract.md §4 (EJ-2)"


# --------------------------------------------------------------------------- C6

def _is_immutable_subtype(node: ast.ClassDef, bases: set) -> bool:
    """A class that subclasses an immutable builtin AND declares `__slots__ = ()` has no
    instance dict and no mutable base: it is a value by construction (the `_FrozenMap` of the
    domain template — a tagged tuple of pairs). Anything else that is not a frozen dataclass or
    an enum is refused by C6."""
    if not (bases & _IMMUTABLE_BUILTINS):
        return False
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__slots__" for t in stmt.targets):
            v = stmt.value
            return isinstance(v, (ast.Tuple, ast.List)) and not v.elts
    return False


def _is_frozen_dataclass(node: ast.ClassDef) -> bool:
    frozen = False
    for dec in node.decorator_list:
        call = dec if isinstance(dec, ast.Call) else None
        name = (call.func if call else dec)
        label = name.id if isinstance(name, ast.Name) else getattr(name, "attr", "")
        if label != "dataclass":
            continue
        frozen = any(kw.arg == "frozen" and getattr(kw.value, "value", False)
                     for kw in (call.keywords if call else []))
    return frozen


def _is_value_class(node: ast.ClassDef) -> bool:
    """An enum, an exception, an immutable builtin subtype, or a frozen dataclass."""
    bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases}
    if bases & {"Enum", "IntEnum", "StrEnum", "Flag"}:
        return True
    if any(isinstance(b, ast.Name) and b.id in ("Exception", "RuntimeError", "ValueError")
           for b in node.bases):
        return True
    return _is_immutable_subtype(node, bases) or _is_frozen_dataclass(node)


def _held_markers(pkg: Package, held) -> list:
    """The map's domain.py roster against the artifacts domain.py holds for the blueprint."""
    listed = list(pkg.entry_of_file("domain.py").get("artifacts") or [])
    missing = [a for a in held if a not in listed]
    extra = [a for a in listed if a not in held]
    out = []
    if missing:
        out.append(error(f"blueprint-map.json's domain.py entry does not list the blueprint "
                         f"artifact(s) {', '.join(missing)}, which domain.py holds; the roster "
                         "is the blueprint's, every artifact accounted for", rule="C6",
                         clause=_ROSTER))
    if extra:
        out.append(error(f"blueprint-map.json's domain.py entry lists {', '.join(extra)} as "
                         "blueprint artifacts domain.py holds, but the blueprint does not "
                         "(or declares it external)", rule="C6", clause=_ROSTER))
    return out


def _declared_markers(pkg: Package, module) -> list:
    """Every name domain.py holds an artifact under (value objects, enums, aliases —
    `held_vocabulary`) is declared in the map once: a blueprint artifact under `artifacts`, or
    the backend's own under `backend_artifacts` (code-contract §3). A value object of the
    package the map does not name is an artifact nobody declared; a `backend_artifacts` name
    domain.py does not hold is a declaration of nothing."""
    entry = pkg.entry_of_file("domain.py")
    if not entry:
        return []                   # a domain.py the map does not list is C3's to report
    listed = list(entry.get("artifacts") or [])
    own = list(entry.get("backend_artifacts") or [])
    held = [h.name for h in held_vocabulary(module.tree)]
    out = []
    undeclared = [n for n in held if n not in listed and n not in own]
    if undeclared:
        out.append(error(f"domain.py holds {', '.join(undeclared)}, which blueprint-map.json's "
                         "domain.py entry declares neither as a blueprint artifact nor under "
                         "`backend_artifacts`: every value object of the package is declared "
                         "once", rule="C6", clause=_ROSTER))
    twice = [n for n in own if n in listed]
    stray = [n for n in own if n not in held]
    if twice:
        out.append(error(f"blueprint-map.json's domain.py entry lists {', '.join(twice)} both as "
                         "blueprint artifacts and under `backend_artifacts` (one source per "
                         "name)", rule="C6", clause=_ROSTER))
    if stray:
        out.append(error(f"blueprint-map.json's domain.py entry lists {', '.join(stray)} under "
                         "`backend_artifacts`, which domain.py does not hold", rule="C6",
                         clause=_ROSTER))
    return out


def _external_markers(pkg: Package, external) -> list:
    """The map's `external_artifacts` against the blueprint's external declarations."""
    recorded = {r.get("name"): r for r in pkg.bmap.get("external_artifacts") or []
                if isinstance(r, Mapping)}
    out = []
    for ext in external:
        row = recorded.get(ext.name) or {}
        if (row.get("owner"), row.get("where")) != (ext.owner, ext.where):
            out.append(error(f"the blueprint declares artifact {ext.name!r} external (owner "
                             f"{ext.owner}, {ext.where}), but blueprint-map.json's "
                             "`external_artifacts` does not record it so", rule="C6",
                             clause=_ROSTER))
    for name in sorted(set(recorded) - {e.name for e in external}, key=str):
        out.append(error(f"blueprint-map.json records {name!r} in `external_artifacts`, but the "
                         "blueprint does not declare it external", rule="C6", clause=_ROSTER))
    return out


def _roster_markers(pkg: Package, module) -> tuple:
    """(markers, the roster's counts): every blueprint artifact held by domain.py or declared
    external, and the map saying the same (code-contract §3)."""
    found, problems = roster(pkg.pipeline, defined_names(module.tree))
    out = [error(p, rule="C6", clause=_ROSTER) for p in problems]
    out += _held_markers(pkg, found.held) + _external_markers(pkg, found.external)
    out += _declared_markers(pkg, module)
    return out, (len(found.held), len(found.external))


def c6_domain_is_frozen(pkg: Package):
    module = pkg.module_by_rel("domain.py")
    if module is None:
        return [error("the package has no domain.py; the blueprint's artifacts are its value "
                      "objects and they have to live somewhere", rule="C6",
                      clause="code-contract.md §2")]
    out = [error(f"class {node.name} is neither a frozen dataclass nor an enum; an artifact "
                 "that can change after construction is not a value", rule="C6",
                 clause=_VALUE, where=f"domain.py:{node.lineno}")
           for node in module.tree.body
           if isinstance(node, ast.ClassDef) and not _is_value_class(node)]
    if not out:
        out.append(ok("every class in domain.py is a frozen dataclass or an enum", rule="C6",
                      clause=_VALUE))
    roster_out, (held, external) = _roster_markers(pkg, module)
    out += roster_out or [ok(f"every one of the blueprint's {held + external} artifact(s) is "
                             f"accounted for: {held} held by domain.py, {external} declared "
                             "external with its owner, as blueprint-map.json records",
                             rule="C6", clause=_ROSTER)]
    return out


# --------------------------------------------------------------------------- C7

def _mutable_state(module) -> list:
    out = []
    for node in module.tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if not names or all(n.startswith("__") for n in names):
            continue
        if not isinstance(node.value, MUTABLE_NODES):
            continue
        if all(n.lstrip("_").isupper() for n in names):
            # An UPPER_CASE container is a constant lookup table by convention — the block
            # registry, the optimiser map, the metric map. The rule is about *state*, not about
            # tables: what makes two runs of the same code differ is a container something
            # writes to, and a table nothing writes to is not that. Naming is the only signal
            # available statically, so the convention is the check, and it is documented here
            # rather than left implicit.
            continue
        out.append(error(f"{', '.join(names)} is module-level mutable state; state that "
                         "outlives a call and belongs to no object is what makes two runs of "
                         "the same code differ", rule="C7", clause=_GATE,
                         where=f"{module.rel}:{node.lineno}"))
    return out


def c7_no_module_level_mutable_state(pkg: Package):
    out = [m for module in pkg.modules for m in _mutable_state(module)]
    if not any(m.level is Level.ERROR for m in out):
        out.append(ok("no module-level mutable state (UPPER_CASE containers are read as "
                      "constant tables by convention)", rule="C7", clause=_GATE))
    return out


# --------------------------------------------------------------------------- C8

def _out_of_scope(pkg: Package, rel) -> bool:
    return pkg.bundle and str(pkg.entry_of_file(rel).get("status", "")).startswith("out of scope")


def c8_datasource_seals_the_test(pkg: Package):
    rel = pkg.file_of_role("DataSource")
    if rel is None:
        return [error("no DataSource-role adapter is mapped; this stage cannot run without "
                      "one", rule="C8", clause="code-contract.md §1")]
    if _out_of_scope(pkg, rel):
        # a bundle has no ingestion; the sealed-part gate is upstream's to keep, and the file
        # that holds it is deliberately not here
        return []
    module = pkg.module_by_rel(rel)
    if module is None:
        return [error(f"the map names {rel} as the DataSource adapter but the package does "
                      "not contain it", rule="C8", clause="code-contract.md §3")]
    found = defs_named(module, "load_part")
    if not found:
        return [error(f"{rel} exposes no load_part, neither on its adapter class nor at module "
                      "level; the DataSource port is `load_part(spec, part)` and "
                      "`make_loader(...)`", rule="C8", clause="backend-contract.md §1")]
    open_doors = [fn for fn in found if "SealedPartError" not in raises_names(fn)]
    if open_doors:
        return [error(f"{rel}'s load_part does not raise SealedPartError; the seal is a "
                      "property of this code, not of anyone's discipline — the test part must "
                      "be refused before a file is opened", rule="C8",
                      clause="netspec-contract §1", where=f"{rel}:{open_doors[0].lineno}")]
    out = [ok(f"{rel} raises SealedPartError on the test part", rule="C8",
              clause="netspec-contract §1")]
    if not defs_named(module, "load_sealed_part"):
        out.append(warn(f"{rel} exposes no load_sealed_part; /vega-eval will have no door to "
                        "the test part at all", rule="C8", clause="backend-contract.md §1"))
    return out


# --------------------------------------------------------------------------- C9

def _alias_of_build(module):
    """The class `build_model = <Class>.build` names, or None."""
    for node in module.tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "build_model"):
            continue
        value = node.value
        if (isinstance(value, ast.Attribute) and value.attr == "build"
                and isinstance(value.value, ast.Name)):
            return value.value.id
    return None


def _the_constructor(module) -> tuple:
    """(network class name or None, the def that is the only constructor) or (None, None).

    Two shapes are the one constructor: a module-level `def build_model(spec, ...)` that makes
    the `Network`, or the class-shaped adapter's `build` exposed as `build_model =
    Network.build` (R-G1)."""
    classes = {n.name: n for n in module.tree.body if isinstance(n, ast.ClassDef)}
    builder = functions_of(module).get("build_model")
    if builder is not None:
        return ("Network" if "Network" in classes else None), builder
    owner = _alias_of_build(module)
    if owner in classes:
        build = next((n for n in classes[owner].body
                      if isinstance(n, _FUNCTION_NODES) and n.name == "build"), None)
        if build is not None:
            return owner, build
    return None, None


def _direct_constructions(pkg: Package, rel: str, network: str, builder) -> list:
    inside = {id(n) for n in ast.walk(builder)}
    out = []
    for other in pkg.modules:
        for node in ast.walk(other.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == network):
                continue
            if other.rel == rel and id(node) in inside:
                continue
            out.append(error(f"constructs {network}(...) directly; `build_model` is the only "
                             "constructor of the network, so a change to the network is a "
                             "change to the spec", rule="C9", clause=_BUILD,
                             where=f"{other.rel}:{node.lineno}"))
    return out


def c9_one_constructor(pkg: Package):
    rel = pkg.file_of_role("Model")
    if rel is None:
        return [error("no Model-role adapter is mapped", rule="C9",
                      clause="code-contract.md §1")]
    module = pkg.module_by_rel(rel)
    network, builder = _the_constructor(module) if module is not None else (None, None)
    if builder is None:
        return [error(f"{rel} exposes no build_model(spec) — neither a module-level def nor "
                      "the alias `build_model = Network.build`; it is the only constructor of "
                      "the network", rule="C9", clause=_BUILD)]
    out = _direct_constructions(pkg, rel, network, builder) if network else []
    if not out:
        out.append(ok(f"{rel}'s build_model is the only constructor of the network", rule="C9",
                      clause=_BUILD))
    return out


# --------------------------------------------------------------------------- C10

def _transform_shape(fn) -> list:
    """The parameter names of a `transform` as the Protocol sees them (without self)."""
    args = [a.arg for a in fn.args.args]
    return args[1:] if args[:1] == ["self"] else args


def c10_transformer_protocol(pkg: Package):
    rel = pkg.file_of_role("FeatureTransformer")
    if rel is None:
        return []
    module = pkg.module_by_rel(rel)
    if module is None:
        if _out_of_scope(pkg, rel):
            return []
        return [error(f"the map names {rel} as the FeatureTransformer adapter but the package "
                      "does not contain it", rule="C10", clause="code-contract.md §3")]
    found = defs_named(module, "transform")
    if not found:
        return [error(f"{rel} exposes no transform(df_raw, history=None); the shaper's emitted "
                      "module satisfies that Protocol structurally, and this adapter must "
                      "present the same shape", rule="C10", clause=_GATE)]
    if not any(_transform_shape(fn)[:1] == ["df_raw"] and "history" in _transform_shape(fn)
               for fn in found):
        args = tuple(_transform_shape(found[0]))
        return [error(f"{rel}'s transform{args} does not match "
                      "transform(df_raw, history=None)", rule="C10", clause=_GATE)]
    return [ok(f"{rel} satisfies transform(df_raw, history=None) structurally", rule="C10",
               clause=_GATE)]
