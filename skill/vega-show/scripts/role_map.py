"""
role_map — `read_role_map()`, the reader of `vega/label-roles@1` (`docs/label-roles.json`).

`/document-project` in Vega mode (and lanekit's `report_stub`) writes the map at every compile:
which LaTeX label is the network's forward equation, which figure is the serving slice. The
page links into the report through these roles, never through typed numbers. This module
reads the map and checks it against the `.aux` it names, so a role never binds a label the
report no longer carries (contract §6.3, build plan R-E7):

- the map validates `references/schemas/label-roles-1.schema.json`;
- the `.aux` it names exists, and its digest equals `report.aux_digest` — otherwise the map is
  older than the report, `ERROR:` exit 2 "recompile /document-project" (gate PF6), not a fork;
- every role is a catalog `report_roles` entry, with the catalog's kind and stage;
- every label exists in the `.aux`, with the role's kind (from its anchor or name prefix);
- the optional `number`, `page` and `title` of a role equal what the `.aux` records.

Every refusal is a `Problem` of gate PF6: a map that disagrees with its `.aux`, in whatever
field, is a map that was not written from this compile. The reader returns the map or the
problems, never a map with problems. The bindings carry the `.aux`'s number, page and title,
so a consumer reads the report's numbering from one place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from report_labels import AuxLabelSource, ReportUnreadable
from show_ports import LabelRef, LabelSource
from vega_show_core import CATALOG, HERE, Problem, read_json

__all__ = ["RoleBinding", "RoleMap", "read_role_map", "resolve_aux"]

SCHEMA = HERE.parent / CATALOG["schemas"]["vega/label-roles@1"]["file"]
_RULE = "PF6"
_ORIGIN = "document-project"
_RECOMPILE = "recompile /document-project"


@dataclass(frozen=True)
class RoleBinding:
    """One report role bound to a label, with the label's number, page and title."""

    role: str
    label: str
    kind: str
    stage: str
    number: str | None = None
    page: int | str | None = None
    title: str | None = None


@dataclass(frozen=True)
class RoleMap:
    """The report's role → label binding: emitted (`vega/label-roles@1`, origin
    `document-project`) or proposed from the `.aux` titles (origin `proposed`, WP H04)."""

    bindings: tuple[RoleBinding, ...]
    source: str
    origin: str
    aux_digest: str | None = None
    aux: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "bindings", tuple(sorted(self.bindings, key=lambda b: b.role)))
        if self.origin not in CATALOG["showcase"]["role_origins"]:
            raise ValueError(f"role-map origin {self.origin!r} is not one of "
                             f"{CATALOG['showcase']['role_origins']}")

    @property
    def roles(self) -> tuple[tuple[str, str], ...]:
        return tuple((b.role, b.label) for b in self.bindings)

    def binding(self, role: str) -> RoleBinding | None:
        return next((b for b in self.bindings if b.role == role), None)


def read_role_map(path: Path, roles_catalog: Mapping[str, dict] | None = None,
                  labels: LabelSource | None = None) -> tuple[RoleMap | None, list[Problem]]:
    """The role map at `path` checked against its `.aux`, or None and the problems.

    `roles_catalog` defaults to the catalog's `report_roles`. `labels` is the report's label
    source; by default the `.aux` the map names is opened (`resolve_aux`)."""
    path, roles = Path(path), CATALOG["report_roles"] if roles_catalog is None else roles_catalog
    doc, err = read_json(path, "the label-role map")
    if err:
        return None, [Problem(_RULE, str(path), err)]
    problems = _schema_problems(doc, path)
    if problems:
        return None, problems
    labels, problems = _label_source(doc, path, labels)
    if labels is None:
        return None, problems
    problems += _role_problems(doc["roles"], roles, labels.labels(), path)
    if problems:
        return None, problems
    aux = labels.labels()
    bindings = tuple(_binding(role, spec, aux[spec["label"]]) for role, spec in doc["roles"].items())
    return RoleMap(bindings, source=str(path), origin=_ORIGIN, aux_digest=labels.digest(),
                   aux=doc["report"]["aux"]), []


def resolve_aux(path: Path, aux: str) -> Path:
    """Where the `.aux` a map names is: the map's paths are relative to the directory that
    holds `docs/` (`docs/label-roles.json` names `docs/main.aux`); a map moved elsewhere
    finds a same-named `.aux` beside itself."""
    given = Path(aux)
    if given.is_absolute():
        return given
    candidates = [path.parent.parent / given, path.parent / given.name]
    return next((c for c in candidates if c.is_file()), candidates[0])


def _schema_problems(doc: object, path: Path) -> list[Problem]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return [Problem(_RULE, str(path), "jsonschema (draft 2020-12) is needed to validate "
                                          "vega/label-roles@1; install jsonschema>=4")]
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.path))
    return [Problem(_RULE, f"{path}#{'/'.join(map(str, e.path)) or '(root)'}",
                    f"not vega/label-roles@1: {e.message}") for e in errors]


def _label_source(doc: dict, path: Path, labels: LabelSource | None) -> tuple:
    where = f"{path}#report.aux"
    if labels is None:
        aux = resolve_aux(path, doc["report"]["aux"])
        if not aux.is_file():
            why = f"the report's .aux {aux} does not exist — {_RECOMPILE}"
            return None, [Problem(_RULE, where, why)]
        try:
            labels = AuxLabelSource(aux)
        except ReportUnreadable as exc:
            return None, [Problem(_RULE, where, f"{exc} — {_RECOMPILE}")]
    if labels.digest() != doc["report"]["aux_digest"]:
        return None, [Problem(_RULE, where, "the role map is older than the report's .aux "
                                            f"({doc['report']['aux_digest']} recorded, "
                                            f"{labels.digest()} now) — {_RECOMPILE}")]
    return labels, []


def _role_problems(bound: Mapping[str, dict], roles: Mapping[str, dict],
                   aux: Mapping[str, LabelRef], path: Path) -> list[Problem]:
    out = []
    for role, spec in bound.items():
        where = f"{path}#roles.{role}"
        want = roles.get(role)
        if want is None:
            out.append(Problem(_RULE, where, f"{role!r} is not a catalog report role"))
            continue
        for key in ("kind", "stage"):
            if spec[key] != want[key]:
                out.append(Problem(_RULE, where, f"{key} {spec[key]!r}, the catalog's role is "
                                                 f"{want[key]!r}"))
        found = aux.get(spec["label"])
        if found is None:
            out.append(Problem(_RULE, where, f"the label {spec['label']!r} is not in the .aux — "
                                             f"{_RECOMPILE}"))
            continue
        out += _label_problems(spec, found, where)
    return out


def _label_problems(spec: Mapping, found: LabelRef, where: str) -> list[Problem]:
    out = []
    if str(found.kind) != spec["kind"]:
        out.append(Problem(_RULE, where, f"the label {found.label!r} is a {found.kind} in the .aux, "
                                         f"the role needs a {spec['kind']}"))
    for key in ("number", "page", "title"):
        if key in spec and str(spec[key]) != str(getattr(found, key)):
            out.append(Problem(_RULE, where, f"{key} {spec[key]!r} differs from the .aux's "
                                             f"{getattr(found, key)!r} — {_RECOMPILE}"))
    return out


def _binding(role: str, spec: Mapping, found: LabelRef) -> RoleBinding:
    return RoleBinding(role=role, label=spec["label"], kind=spec["kind"], stage=spec["stage"],
                       number=found.number, page=found.page, title=found.title)
