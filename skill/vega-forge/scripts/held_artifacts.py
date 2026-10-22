#!/usr/bin/env python3
"""
held_artifacts.py — the names a pipeline blueprint's artifacts can be held under (code-contract
§3), listed from the backend's `domain.py` template.

    python3 held_artifacts.py [--shape row|window|graph] [--backend pytorch] [--json]

Every `pipeline.json.artifacts[]` entry is held by the emitted package — a name `domain.py`
defines — or declared external with the stage that holds it; `/vega-forge` refuses, by name,
an artifact that is neither. `/blueprint --mode pipeline` runs this before it draws §2, so a
package-held artifact takes the name the package gives it (a domain word for it is prose beside
that name, never a second artifact) and everything else is declared external. One line per
name: the name, what it is (value object, enum, alias of …) and the first line of its
docstring. The package carries one sample shape's values only (the emitter keeps the
template's sections of the spec's shape), so `--shape` names the shape the shaped data has —
`row` for a table of independent rows, `window` for series cut into windows, `graph` for nodes
and edges; without it the list is the names every shape's package holds. The list is
`vega_roster.held_vocabulary` on the template with that shape's sections kept; nothing is
imported or executed. A compat alias (a former name the template keeps one release for a
blueprint that still draws it) is not listed: a blueprint draws each artifact by its one name.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import vega_roster  # noqa: E402


SHAPES = ("row", "window", "graph")


def domain_template(backend: str) -> Path:
    return HERE / "backends" / backend / "templates" / "domain.py"


def _emitter(backend: str):
    """The backend's emitter module (for its `select_shapes`), loaded by path."""
    path = HERE / "backends" / backend / "emit.py"
    spec = importlib.util.spec_from_file_location(f"held_artifacts_emit_{backend}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vocabulary(path: Path, backend: str, shape: str | None) -> tuple:
    """The held names of the template with `shape`'s sections kept (the emitter's own cut), or
    — without a shape — the ones every shape's package holds, in the template's order."""
    emit = _emitter(backend)
    # a compat name (a former name kept one release for a blueprint that still uses it) is
    # never offered: a new blueprint takes the one name the package gives each artifact
    text, select = emit.select_compat(path.read_text(encoding="utf-8"), frozenset()), emit.select_shapes
    if shape:
        return vega_roster.held_vocabulary(select(text, (shape,), str(path)))
    per = [{h.name for h in vega_roster.held_vocabulary(select(text, (s,), str(path)))}
           for s in SHAPES]
    common = set.intersection(*per)
    return tuple(h for h in vega_roster.held_vocabulary(select(text, SHAPES, str(path)))
                 if h.name in common)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="pytorch")
    ap.add_argument("--shape", choices=SHAPES, default=None,
                    help="the sample shape of the shaped data (default: the names every shape holds)")
    ap.add_argument("--json", action="store_true", help="print the list as JSON")
    a = ap.parse_args(argv)
    path = domain_template(a.backend)
    if not path.is_file():
        print(f"ERROR: no domain.py template for backend {a.backend!r} at {path}")
        return 2
    held = vocabulary(path, a.backend, a.shape)
    if a.json:
        print(json.dumps([asdict(h) for h in held], indent=1))
        return 0
    width = max(len(h.name) for h in held)
    for h in held:
        print(f"{h.name:<{width}}  {h.kind:<22}  {h.summary[:110]}")
    print(f"OK: {len(held)} names domain.py ({a.backend}) can hold an artifact under; any other "
          "blueprint artifact is declared external with its owner and where it lives")
    return 0


if __name__ == "__main__":
    sys.exit(main())
