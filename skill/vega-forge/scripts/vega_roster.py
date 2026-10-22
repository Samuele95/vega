"""
vega_roster — the blueprint's artifacts, accounted for (code-contract.md §3): the one derivation
the emitter writes into `blueprint-map.json` and the conformance gate (C6) reads back.

Every entry of `pipeline.json.artifacts[]` has exactly one home:

  held      — a name `domain.py` defines (a class, or an alias such as `SamplePart`
              of the carried part), listed in the map's `domain.py` entry under `artifacts`;
  external  — declared by the blueprint as living outside the package,
              `"external": {"owner": "/vega-train", "where": "the campaign directory"}`, never
              emitted into `domain.py`, and listed in the map's `external_artifacts` with its
              owner and where it lives.

An artifact that is neither — the blueprint lists it, `domain.py` does not define it, and the
blueprint does not say who holds it — is a world-model element with no home, and so is an
artifact declared external that `domain.py` defines after all (the declaration is false: the
code holds it). Both are refused by name; the remedy is the blueprint's (`/blueprint …
--concern`), never an emitter that drops an artifact from the roster or invents a class for it.

`owner` is a stage of the chain other than `/vega-forge`: what the forge holds is in the
package. The module is pure — it reads a parsed `domain.py` (or its text) and a pipeline dict,
and returns values; it prints nothing and imports nothing of the backend.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["EXTERNAL_OWNERS", "External", "Held", "Roster", "defined_names", "held_vocabulary",
           "roster"]

# The stages that may hold a blueprint artifact outside the emitted package: the chain's stage
# skills, the forge excepted (its product IS the package, so what it holds is in domain.py).
EXTERNAL_OWNERS = ("/dataset-forge", "/data-lens", "/dataset-shaper", "/blueprint",
                   "/vega-train", "/vega-eval", "/vega-pack", "/vega-show")

CLAUSE = "code-contract.md §3"


def defined_names(domain) -> frozenset:
    """The public top-level names a `domain.py` defines — classes, functions and assigned
    aliases — from its text or its parsed `ast.Module`. Nothing is imported or executed."""
    tree = domain if isinstance(domain, ast.Module) else ast.parse(domain)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(n for n in names if not n.startswith("_"))


@dataclass(frozen=True)
class Held:
    """A name a blueprint artifact can take to be held by the package: what `domain.py` defines
    as a value object, an enum or an alias, with the first line of what it says it is."""

    name: str
    kind: str          # "value object" | "view" | "enum" | "alias of <Name>"
    summary: str


def _summary(node) -> str:
    doc = ast.get_docstring(node) if isinstance(node, ast.ClassDef) else None
    return " ".join((doc or "").split("\n\n")[0].split())


def held_vocabulary(domain) -> tuple:
    """The names a blueprint artifact can be held under (`Held`, in `domain.py`'s order): its
    public classes — value objects and enums — and its aliases of them (`SamplePart`, the
    carried part). Exceptions and constants are names `domain.py` defines but are no artifact of
    the world model, so they are not offered. `/blueprint --mode pipeline` reads this list (the
    CLI `held_artifacts.py`) to name §2's package-held artifacts as the package defines them."""
    tree = domain if isinstance(domain, ast.Module) else ast.parse(domain)
    classes, out = {}, []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases}
            if bases & {"Exception", "ValueError", "RuntimeError", "TypeError", "KeyError"}:
                continue
            kind = "enum" if bases & {"Enum", "IntEnum", "StrEnum"} else (
                "view" if "read-only view" in (ast.get_docstring(node) or "").split("\n\n")[0]
                else "value object")
            classes[node.name] = node
            out.append(Held(node.name, kind, _summary(node)))
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)
                     and not t.id.startswith("_") and not t.id.isupper()]
            members = _union_members(node.value)
            if isinstance(node.value, ast.Name) and node.value.id in classes:
                held = [Held(n, f"alias of {node.value.id}", _summary(classes[node.value.id]))
                        for n in names]
            elif members and all(m in classes for m in members):
                held = [Held(n, "union of " + " | ".join(members),
                             "any one of " + ", ".join(members)) for n in names]
            elif _is_union(node.value):
                held = [Held(n, "union", "the part of the sample shape the package carries")
                        for n in names]
            else:
                held = []
            out.extend(held)
    return tuple(out)


def _is_union(value) -> bool:
    return isinstance(value, ast.Subscript) and getattr(value.value, "id", None) == "Union"


def _union_members(value) -> tuple:
    """The class names of `Union[A, B, C]` (or `A | B | C`), else ()."""
    if isinstance(value, ast.Subscript) and getattr(value.value, "id", None) == "Union":
        elts = value.slice.elts if isinstance(value.slice, ast.Tuple) else [value.slice]
        return tuple(e.id for e in elts if isinstance(e, ast.Name))
    if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
        left, right = _union_members(value.left) or (getattr(value.left, "id", None),), \
            _union_members(value.right) or (getattr(value.right, "id", None),)
        names = left + right
        return names if all(names) else ()
    return ()


@dataclass(frozen=True)
class External:
    """A blueprint artifact held outside the package, as the map records it."""

    name: str
    kind: str
    owner: str
    where: str

    def as_map_row(self) -> dict:
        return {"name": self.name, "kind": self.kind, "owner": self.owner,
                "where": self.where, "source": "blueprint:artifacts"}


@dataclass(frozen=True)
class Roster:
    """Every blueprint artifact, in the blueprint's order: those `domain.py` holds, and those
    the blueprint declared external."""

    held: tuple
    external: tuple

    def external_rows(self) -> list:
        return [e.as_map_row() for e in self.external]


def _external(entry: Mapping[str, Any], name: str) -> tuple:
    """(External | None, problem | None) for an entry that carries `external`."""
    decl = entry.get("external")
    if not isinstance(decl, Mapping):
        return None, (f"artifact {name!r} declares `external` as {decl!r}; the declaration is "
                      "an object {\"owner\": \"<stage>\", \"where\": \"<where it lives>\"}")
    owner, where = decl.get("owner"), decl.get("where")
    if owner not in EXTERNAL_OWNERS:
        return None, (f"artifact {name!r} is declared external with owner {owner!r}; the owner "
                      f"is the stage that holds it, one of {', '.join(EXTERNAL_OWNERS)} — "
                      "/vega-forge holds nothing outside its package")
    if not isinstance(where, str) or not where.strip():
        return None, (f"artifact {name!r} is declared external (owner {owner}) without `where`;"
                      " a declaration that does not say where the artifact lives is not an "
                      "account of it")
    return External(name, str(entry.get("kind") or ""), owner, where.strip()), None


def _place(entry, name, defined, held, external) -> str | None:
    """Put one artifact in its home; the problem when it has none (or two)."""
    if "external" not in entry:
        if name in defined:
            held.append(name)
            return None
        return (f"the blueprint's artifact {name!r} is not defined in domain.py and is not "
                "declared external — an artifact with no home in the package or outside it; "
                "if a later stage holds it (the campaign directory is /vega-train's, the recipe "
                "/dataset-shaper's, the seal /vega-eval's), declare it in pipeline.json as "
                "`\"external\": {\"owner\": \"<stage>\", \"where\": \"<where it lives>\"}` "
                "through `/blueprint … --concern`; else the backend's domain.py must define it")
    ext, problem = _external(entry, name)
    if problem:
        return problem
    if name in defined:
        return (f"the blueprint declares artifact {name!r} external (owner {ext.owner}), but "
                "domain.py defines it — the package holds it, so the declaration is false; "
                "drop `external` from the blueprint's entry")
    external.append(ext)
    return None


def roster(pipeline: Mapping[str, Any], defined) -> tuple:
    """(Roster, [problem, …]) for the blueprint's `artifacts[]` against the names `defined`
    in `domain.py` (`defined_names`). The roster is complete only when the problems are none."""
    held, external, problems, seen = [], [], [], set()
    for entry in pipeline.get("artifacts") or []:
        name = entry.get("name") if isinstance(entry, Mapping) else None
        if not isinstance(name, str) or not name:
            problems.append(f"pipeline.json.artifacts[] holds an entry with no name ({entry!r})")
            continue
        if name in seen:
            problems.append(f"the blueprint lists artifact {name!r} twice")
            continue
        seen.add(name)
        problem = _place(entry, name, defined, held, external)
        if problem:
            problems.append(problem)
    return Roster(tuple(held), tuple(external)), problems
