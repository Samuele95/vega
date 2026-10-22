"""
The base domain model a lane's chain starts from: a domain-forge model source folder with one
entity, composed into a model HTML by domain-forge's own `compose_model.py` (D6 §5.2).

    entity = ModelEntity(base_iri="http://example.org/landcover#", name="Parcel", component="survey",
                         properties=properties_for(SCHEMA), overview=text, rationale={...},
                         comment="One parcel as surveyed.")
    src = write_model_source(work / "landcover.model-source", entity)
    html = compose_base_model(src, skills)

These are `make_fixture.py`'s two functions at `c849f8b`, lifted with the entity, its component
and its data properties as parameters — gathered in one value, `ModelEntity`, where D6 §5.2
lists them as keywords (seven parameters would break C12's five). `make_fixture.py` calls them
with the orders values and its output does not change by a byte; every lane fixture
(lanekit.fixture) calls them with its generator's schema.

The base model is composed, never written by hand: compose_model.py injects the entity into
domain-forge's own template, so the model carries the canonical Turtle, its JSON-LD mirror, the
projectors and the runtime, and passes every invariant `validate_model.py` holds a real model
to. It is T-box only on purpose: an A-box would make the validator require the @KG_RUNTIME
block that belongs to domain-forge's own template.

This module imports nothing beyond the standard library: `make_fixture.py` imports it before
anything heavy, and the lane runner process never imports pandas.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["ModelSourceError", "ModelEntity", "SOURCE_SUFFIX", "prefixes_for", "properties_for",
           "local_name", "write_model_source", "compose_base_model", "composer_path", "base_model_path"]

SOURCE_SUFFIX = ".model-source"
COMPOSER = ("domain-forge", "scripts", "compose_model.py")
STANDARD_PREFIXES = (("owl", "http://www.w3.org/2002/07/owl#"),
                     ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
                     ("xsd", "http://www.w3.org/2001/XMLSchema#"))

# the xsd range a generator column's dtype maps to (lanekit.datasets.ColumnSpec.dtype)
XSD_RANGE = {"float": "xsd:float", "int": "xsd:integer", "str": "xsd:string",
             "bool": "xsd:boolean", "datetime": "xsd:dateTime"}
# columns of these roles are not properties of the entity: the generator marks them dropped
# before the chain reads them
NOT_A_PROPERTY = frozenset({"drop"})


class ModelSourceError(RuntimeError):
    """The base model could not be written or composed; the message says why and what to do."""


# --------------------------------------------------------------------------- the source folder
def prefixes_for(base_iri: str) -> dict[str, str]:
    """The model's prefixes: `ex` for its own IRI, then owl, rdfs and xsd."""
    return {"ex": base_iri, **dict(STANDARD_PREFIXES)}


def local_name(column: str) -> str:
    """A column name as an RDF local name: snake_case -> camelCase (`unit_price` -> `unitPrice`),
    any character a local name cannot hold dropped."""
    parts = [p for p in re.split(r"[^0-9A-Za-z]+", column) if p]
    if not parts:
        raise ModelSourceError(f"column {column!r} has no character a property name can keep")
    head = parts[0][0].lower() + parts[0][1:]
    name = head + "".join(p[0].upper() + p[1:] for p in parts[1:])
    return name if not name[0].isdigit() else "c" + name


def properties_for(schema: Iterable[Any]) -> list[tuple[str, str, str | None]]:
    """The entity's data properties from a generator's SCHEMA (lanekit.datasets.ColumnSpec):
    (local name, xsd range, comment), in the schema's order; the dropped columns are left out,
    and a column's unit, when it has one, is said in its comment."""
    out: list[tuple[str, str, str | None]] = []
    for col in schema:
        if col.role in NOT_A_PROPERTY:
            continue
        rng = XSD_RANGE.get(col.dtype)
        if rng is None:
            raise ModelSourceError(f"column {col.name!r} has dtype {col.dtype!r}; the model "
                                   f"source knows {sorted(XSD_RANGE)}")
        note = col.comment
        if col.unit:
            note = f"{note} ({col.unit})" if note else f"In {col.unit}."
        out.append((local_name(col.name), rng, note))
    return out


@dataclass(frozen=True)
class ModelEntity:
    """The one entity a base model holds: its model's base IRI, its name and component, its
    data properties (local name, xsd range, comment), the model's overview prose, the design
    rationale `model.meta.json` records, and the entity's own comment."""
    base_iri: str
    name: str
    component: str
    properties: Sequence[tuple[str, str, str | None]]
    overview: str
    rationale: Mapping[str, Any] = field(default_factory=dict)
    comment: str = ""

    def graph(self) -> tuple[list[dict], list[str]]:
        """(JSON-LD nodes, Turtle statements): the class, then each data property."""
        cls = f"ex:{self.name}"
        nodes = [{"@id": cls, "@type": "owl:Class", "component": self.component,
                  "label": self.name, "comment": self.comment}]
        ttl = [f'{cls} a owl:Class ;\n  ex:component "{self.component}" ;\n  rdfs:label "{self.name}" ;\n'
               f'  rdfs:comment {json.dumps(self.comment)} .']
        for name, rng, note in self.properties:
            node = {"@id": f"ex:{name}", "@type": "owl:DatatypeProperty", "domain": cls,
                    "range": rng}
            line = f"ex:{name} a owl:DatatypeProperty ; rdfs:domain {cls} ; rdfs:range {rng}"
            if note:
                node["comment"] = note
                line += f" ; rdfs:comment {json.dumps(note)}"
            nodes.append(node)
            ttl.append(line + " .")
        return nodes, ttl


def write_model_source(src: Path, entity: ModelEntity) -> Path:
    """Write the domain-forge model source folder the base model is composed from.

    The layout and the fragment format are compose_model.py's: `model.meta.json`,
    `overview.md`, and one self-contained entity fragment whose
    `application/x-df-entity-data` script holds the class node with its data properties as
    sibling nodes, and whose `application/x-df-entity-ttl` script holds the same statements
    in Turtle — the shape explode_model.py writes.
    """
    src = Path(src)
    (src / "entities").mkdir(parents=True, exist_ok=True)
    meta = {"base": entity.base_iri, "prefixes": prefixes_for(entity.base_iri), "codegenTargets": {},
            "components": {}, "designRationale": dict(entity.rationale)}
    (src / "model.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (src / "overview.md").write_text(entity.overview, encoding="utf-8")
    nodes, ttl = entity.graph()
    name, cls = entity.name, f"ex:{entity.name}"
    (src / "entities" / f"{name}.html").write_text(
        f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><title>{name}</title>'
        f"</head>\n<body>\n<h1>{name}</h1>\n<code>{cls}</code> · component "
        f"<code>{entity.component}</code>\n<p>" + entity.comment + "</p>\n\n"
        '<script type="application/x-df-entity-data" id="entity-data">\n'
        + json.dumps(nodes, indent=2, ensure_ascii=False) + "\n</script>\n"
        '<script type="application/x-df-entity-ttl" id="entity-ttl">\n'
        + "\n".join(ttl) + "\n</script>\n</body></html>\n", encoding="utf-8")
    return src


# --------------------------------------------------------------------------- composing
def composer_path(skills: Path) -> Path:
    return Path(skills).joinpath(*COMPOSER)


def base_model_path(src: Path) -> Path:
    """`<stem>.model-source/<stem>.domain.html`: where the composed base model lands."""
    src = Path(src)
    stem = src.name[: -len(SOURCE_SUFFIX)] if src.name.endswith(SOURCE_SUFFIX) else src.name
    return src / f"{stem}.domain.html"


def compose_base_model(src: Path, skills: Path) -> str:
    """Compose the base model HTML from the source folder with domain-forge's composer; returns
    the composed HTML (also written to `base_model_path(src)`)."""
    composer = composer_path(skills)
    if not composer.is_file():
        raise ModelSourceError(f"ERROR: the fixture's base model is composed by {composer}, "
                               "which is not there; /domain-forge is a sibling this chain "
                               "depends on")
    out = base_model_path(src)
    proc = subprocess.run([sys.executable, str(composer), str(src), "--out", str(out)],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr[-3000:])
        raise ModelSourceError(f"ERROR: compose_model.py exited {proc.returncode}; the fixture "
                               "cannot be built on a base model domain-forge will not compose")
    print("  compose| " + proc.stdout.splitlines()[0] if proc.stdout else "  compose| done")
    return out.read_text(encoding="utf-8")
