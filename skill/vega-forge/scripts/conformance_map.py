"""
conformance_map — C1–C5, the checks that read the blueprint map and the module arrows: every
port has one adapter, extras are baselines, every file is mapped, imports follow the arrows
and name files the package holds, adapters depend on ports.

A support module of `conformance.py`, which re-exports every check; import it from there.
Each check takes the parsed `Package` and returns markers — it reads no argv and prints
nothing.
"""
from __future__ import annotations

import ast

from conformance_package import SUPPORT_FILES, Package, imports_of, target_module
from vega_core import error, ok, warn

__all__ = ["c1_every_port_has_one_adapter", "c2_extras_are_baselines",
           "c3_every_file_is_mapped", "c4_imports_follow_the_arrows",
           "c5_adapters_depend_on_ports"]

_MAP = "code-contract.md §3"
_GATE = "code-contract.md §5.1"


def _status(entry) -> str:
    return str(entry.get("status", ""))


def _port_markers(pkg: Package, port) -> list:
    """C1 on one blueprint port: an entry, exactly one primary, a usable status."""
    role, name = port.get("role"), port.get("name")
    found = [e for e in pkg.entries() if e.get("port") == name and e.get("role") != "baseline"]
    if not found:
        return [error(f"the blueprint's port {name!r} (role {role}) has no entry in "
                      "blueprint-map.json — every port the blueprint drew is either "
                      "implemented, `pending: <stage>`, or (in a bundle) "
                      "`out of scope: <reason>`", rule="C1", clause=_MAP)]
    primary = [e for e in found if _status(e) == "implemented"]
    pending = [e for e in found if _status(e).startswith("pending")]
    scoped = [e for e in found if _status(e).startswith("out of scope")]
    out = []
    if scoped and not pkg.bundle:
        out.append(error(f"port {name!r} is marked `out of scope` outside a bundle; that "
                         "status is only legal with --bundle", rule="C1", clause=_GATE))
    if len(primary) > 1:
        out.append(error(f"port {name!r} has {len(primary)} primary adapters "
                         f"({', '.join(e['file'] for e in primary)}); exactly one is "
                         "allowed, and extras only where the map marks role: baseline",
                         rule="C1", clause=_MAP))
    if not (primary or pending or scoped):
        out.append(error(f"port {name!r} has an entry with no usable status "
                         f"({found[0].get('status')!r})", rule="C1", clause=_MAP))
    return out


def c1_every_port_has_one_adapter(pkg: Package):
    out = [m for port in pkg.blueprint_ports() for m in _port_markers(pkg, port)]
    if not out:
        out.append(ok(f"every one of the blueprint's {len(pkg.blueprint_ports())} port(s) has "
                      "exactly one adapter, or a stage that will fill it", rule="C1",
                      clause=_MAP))
    return out


def c2_extras_are_baselines(pkg: Package):
    seen, out = {}, []
    for entry in pkg.adapter_entries():
        port = entry.get("port")
        if entry.get("role") == "baseline":
            continue
        if port in seen:
            out.append(error(f"port {port!r} is implemented twice ({seen[port]} and "
                             f"{entry.get('file')}) and neither is marked role: baseline",
                             rule="C2", clause=_MAP))
        seen[port] = entry.get("file")
    return out


def _unmapped_files(pkg: Package) -> list:
    mapped = {e.get("file") for e in pkg.entries()}
    entrypoints = {e.get("file") for e in (pkg.bmap.get("entrypoints") or [])}
    out = [error(f"{module.rel} has neither an adapter, a support nor an entrypoint entry — a "
                 "file nobody mapped is a designer's choice the emitter is not allowed to make",
                 rule="C3", clause=_MAP)
           for module in pkg.modules if module.rel not in mapped | entrypoints]
    declared = pkg.bmap.get("unmapped")
    if declared:
        out.append(error(f"blueprint-map.json declares {len(declared)} unmapped file(s) "
                         f"({', '.join(map(str, declared))}); `unmapped` must be empty",
                         rule="C3", clause=_MAP))
    return out


def _mapped_files(pkg: Package) -> tuple:
    """(problems, how many entries are out of scope for this bundle) over the map's files."""
    out, scoped_out = [], 0
    for entry in pkg.entries():
        rel = entry.get("file")
        if not rel or not rel.endswith(".py"):
            continue
        present = pkg.module_by_rel(rel) is not None
        if _status(entry).startswith("out of scope") and pkg.bundle:
            # A bundle keeps the serving slice and drops the rest, so the map's `out of scope`
            # entries name files that are *supposed* to be absent. Their presence is the
            # defect: a bundle still carrying ingestion/data.py would carry the one door to
            # the sealed test part, which is the reason the slice exists.
            scoped_out += 1
            if present:
                out.append(error(
                    f"blueprint-map.json marks {rel} out of scope, but the bundle contains "
                    "it; a file the map says is not part of serving is a file the bundle "
                    "must not ship — above all under ingestion/, which reaches the sealed "
                    "test part", rule="C3", clause=_MAP))
        elif not present:
            out.append(error(f"blueprint-map.json maps {rel}, which the package does not "
                             "contain", rule="C3", clause=_MAP))
    return out, scoped_out


def c3_every_file_is_mapped(pkg: Package):
    missing, scoped_out = _mapped_files(pkg)
    out = _unmapped_files(pkg) + missing
    if not out:
        out.append(ok(f"all {len(pkg.modules)} emitted file(s) are mapped and `unmapped` is "
                      "empty"
                      + (f"; {scoped_out} entry/ies are out of scope for this bundle and are "
                         "correctly absent" if scoped_out else ""),
                      rule="C3", clause=_MAP))
    return out


def _against_the_arrows(module, arrows: dict) -> list:
    here = module.module_name
    allowed = set(arrows.get(here, set())) | {"domain", here}
    out = []
    for expr, level, _, lineno in imports_of(module):
        target = target_module(module, expr, level)
        head = target.split(".")[0] if target else None
        if head is None or head in ("domain", "ports") or head not in arrows:
            continue              # outside the package, an implicit arrow, a relative helper
        if head not in allowed:
            out.append(error(f"imports {head!r}, but the blueprint's module map gives "
                             f"{here!r} the arrows {sorted(arrows.get(here, set())) or '[]'}"
                             " — an import against the arrow is the divergence the map "
                             "exists to catch", rule="C4", clause=_GATE,
                             where=f"{module.rel}:{lineno}"))
    return out


def _top_level_names(module) -> set:
    """The names a module binds at its top level (what `from <package> import name` reads)."""
    names = set()
    for node in module.tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(n.id for tgt in targets for n in ast.walk(tgt) if isinstance(n, ast.Name))
    return names


def _holds(pkg: Package, files: set, parts: list, name: str | None = None) -> bool:
    """Whether the package holds the module `parts` (a file or a sub-package) — or, with
    `name`, the module `parts/name` or a name the package `parts` binds in its `__init__.py`."""
    if name is not None:
        if _holds(pkg, files, parts + [name]):
            return True
        init = pkg.module_by_rel("/".join(parts + ["__init__.py"]))
        return init is not None and name in _top_level_names(init)
    stem = "/".join(parts)
    return f"{stem}.py" in files or f"{stem}/__init__.py" in files


def _dangling(pkg: Package, module, files: set) -> list:
    """The package-relative imports of `module` that name a module the package does not
    hold — lazy ones inside a function included: such an import fails only when its branch
    runs, so a dispatch to a module that was never emitted passes every other check."""
    base = module.rel.split("/")[:-1]
    out = []
    for node in sorted(ast.walk(module.tree), key=lambda n: getattr(n, "lineno", 0)):
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        if node.level - 1 > len(base):
            continue                       # above the package root: not the package's
        here = base[:len(base) - (node.level - 1)]
        parts = here + (node.module.split(".") if node.module else [])
        missing = ([".".join(parts)] if node.module and not _holds(pkg, files, parts) else
                   [".".join(parts + [a.name]) for a in node.names
                    if not node.module and not _holds(pkg, files, parts, a.name)])
        for target in missing:
            out.append(error(f"imports {target!r}, which the package does not hold — a "
                             "package carries the code of the sample shape its spec uses, and "
                             "an import of a module that was not emitted fails only when its "
                             "branch runs: emit the module, or keep the branch out of this "
                             "package (the template's `# >>> shape:` sections)", rule="C4",
                             clause=_GATE, where=f"{module.rel}:{node.lineno}"))
    return out


def c4_imports_follow_the_arrows(pkg: Package):
    files = {m.rel for m in pkg.modules}
    held = [m for module in pkg.modules for m in _dangling(pkg, module, files)]
    arrows = pkg.depends_on()
    if not arrows:
        return held + [warn("pipeline.json declares no modules[], so the import graph could "
                            "not be checked against the blueprint's arrows", rule="C4",
                            clause=_GATE)]
    out = held + [m for module in pkg.modules for m in _against_the_arrows(module, arrows)]
    if not out:
        out.append(ok("every import runs along the blueprint's module arrows and names a "
                      "file the package holds", rule="C4", clause=_GATE))
    return out


def _same_port_baselines(entry, other) -> bool:
    """Two `role: baseline` entries of one port: the extra implementations of that port, split
    across files (`model/baselines.py` <-> `model/baselines_extra.py`). They are one family of
    alternatives, exempt from C14's one-class rule for the same reason (C14_EXEMPT_ROLES), and
    one may import the other (PLAN-AMENDMENTS A14); a baseline importing a primary adapter, or
    another port's baseline, is still refused."""
    return (entry.get("role") == "baseline" and other.get("role") == "baseline"
            and entry.get("port") == other.get("port"))


def _adapter_imports(pkg: Package, rel: str, adapters: dict, support: set) -> list:
    module, entry, out = pkg.module_by_rel(rel), adapters[rel], []
    for expr, level, names, lineno in (imports_of(module) if module is not None else ()):
        target = target_module(module, expr, level)
        if target is None:
            continue
        base = target.replace(".", "/")
        candidates = {f"{base}.py", *(f"{base}/{name}.py" for name in names)}
        hit = next((c for c in candidates if c in adapters and c != rel
                    and not _same_port_baselines(entry, adapters[c])), None)
        if hit and hit not in support:
            out.append(error(f"imports the adapter {hit} (role {adapters[hit].get('role')}); an "
                             "adapter depends on the ports and on `domain`, never on another "
                             "adapter — pass it in as a port instead", rule="C5",
                             clause=_GATE, where=f"{rel}:{lineno}"))
    return out


def c5_adapters_depend_on_ports(pkg: Package):
    adapters = {e.get("file"): e for e in pkg.adapter_entries()}
    support = {e.get("file") for e in pkg.support_entries()} | set(SUPPORT_FILES)
    out = [m for rel in adapters for m in _adapter_imports(pkg, rel, adapters, support)]
    if not out:
        out.append(ok("no adapter imports another adapter", rule="C5", clause=_GATE))
    return out

