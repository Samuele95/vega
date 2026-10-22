"""
reemit_proof — what a `--re-emit-serving` run may move, and the comparison that proves it moved
nothing else (bundle-contract §7.8, R-D2). `reemit` runs it; nothing here writes.

    outputs = ServingOutputs.of(ctx, previous, current)  the bundle files a re-emit writes again
    outputs.holds(rel) -> bool
    moved = compare(old, new, outputs, maps)             what moved and may not, named
    changed = changed_files(old, new)                    every file whose pin differs

**What may move** is what the serving templates write, named by the catalog's patterns
(`reemit.may_change.templates`, the `serving_templates` patterns) with each destination where
the pack renders it: `package/*.py` and `package/inspection/*.py` in the Predictor's module
(and the Predictor's and the Monitor's adapter files, which `package/predict.py` and
`package/monitor.py` render wherever the map puts them), `serve/*.py` and
`serve/static/index.html` in `serve/`, `tests/*.py` in `tests/`, and each `bundle/<name>`
template as `<name>` at the root — plus the export files, exported again from the same weights
and checked again in the same run (`MANIFEST.exports` may change, and no exporter promises the
same bytes twice). A file a `reemit.never` pattern names is never one of them. In the manifest,
the keys `reemit.may_change.manifest` lists move freely, and the listings of bundle files move
only where they name an output: `files` (the catalog says so), `entrypoints` (the served files
are outputs) and the package's `blueprint-map.json`, whose entries for the outputs are the only
ones that may differ — so a template the backend adds is re-emitted like one it changed, with
no catalog edit (R-D2). Everything else must come out byte-identical.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import PurePosixPath

from bundle_slice import bundle_modules
from vega_pack_core import (CATALOG, matching_templates, read_json, serving_templates,
                            template_matches)

__all__ = ["MAP", "MAY_CHANGE", "NEVER", "ServingOutputs", "changed_files", "compare"]

CONTRACT = CATALOG["reemit"]
MAP = "blueprint-map.json"
# the evaluation's inputs and evidence, as glob patterns (the catalog's `<package>/` entry is
# the package rule of `ServingOutputs`: a package file is an output or it is pinned)
NEVER = tuple(p for p in CONTRACT["never"] if "<" not in p)
# manifest keys (dotted) that move freely; a qualified entry ("files (the outputs of the
# templates only)") is a listing of bundle files that moves only where it names an output
MAY_CHANGE = frozenset(e for e in CONTRACT["may_change"]["manifest"] if "(" not in e)
OUTPUT_LISTINGS = tuple(e.split(" (")[0] for e in CONTRACT["may_change"]["manifest"]
                        if "(" in e) + ("entrypoints",)
PINS = "files"
# the adapters `package/predict.py` and `package/monitor.py` render, at the map's files
ADAPTER_ROLES = ("Predictor", "Monitor")
_ABSENT = object()


# --------------------------------------------------------------------------- what may move

@dataclass(frozen=True)
class ServingOutputs:
    """The bundle files a re-emit writes again. `package` is the emitted package, `module`
    the Predictor's module (where the `package` templates render), `adapters` the bundle paths
    of the Predictor's and the Monitor's adapter files, `root_names` the names the `bundle/`
    templates take at the bundle root, `exported` the directories the `exports` records write
    to (the exported models and their parity report, exported again from the same weights and
    checked again in the same run: `MANIFEST.exports` may change, and no exporter promises the
    same bytes twice)."""

    package: str
    module: str
    adapters: tuple = ()
    root_names: tuple = ()
    exported: tuple = ()

    @staticmethod
    def of(ctx, previous: dict, current: dict) -> "ServingOutputs":
        """This bundle's outputs: its modules and adapters from the staged map, the root names
        from the templates rendered now and from the ones the previous manifest lists (so a
        template the backend dropped still names the file it used to write), the export
        directories from both manifests."""
        entries = (read_json(ctx.paths.pkg / MAP, MAP)[0] or {}).get("entries") or []
        adapters = tuple(f"{ctx.package}/{e['file']}" for e in entries
                         if isinstance(e, dict) and e.get("role") in ADAPTER_ROLES
                         and e.get("file"))
        listed = (((previous.get("serve") or {}).get("templates") or {}).get("files") or {})
        rels = set(matching_templates(ctx.templates)["bundle"]) | {
            rel for rel in listed
            if any(template_matches(p, rel) for p in serving_templates()["bundle"])}
        exported = {str(PurePosixPath(r["file"]).parent) for doc in (previous, current)
                    for r in doc.get("exports") or () if isinstance(r, dict) and r.get("file")}
        return ServingOutputs(ctx.package, bundle_modules(ctx.paths.pkg)["serving"], adapters,
                              tuple(sorted(rel.split("/", 1)[1] for rel in rels)),
                              tuple(sorted(exported - {"."})))

    def patterns(self) -> tuple:
        """The catalog's template patterns, each at its destination in the bundle (the
        `bundle` destination is the root, by name: `root_names`)."""
        out = []
        for dest, patterns in serving_templates().items():
            if dest == "bundle":
                continue
            for pattern in patterns:
                head, rest = pattern.split("/", 1)
                base = f"{self.package}/{self.module}" if dest == "package" else head
                out.append(f"{base}/{rest}")
        return tuple(out)

    def holds(self, rel: str) -> bool:
        """Whether a re-emit writes the bundle file `rel` again (a `never` pattern wins)."""
        rel = str(rel)
        if any(fnmatch.fnmatchcase(rel, p) for p in NEVER):
            return False
        return rel in self.root_names or rel in self.adapters or \
            any(rel.startswith(f"{d}/") for d in self.exported) or \
            any(template_matches(p, rel) for p in self.patterns())


def compare(old: dict, new: dict, outputs: ServingOutputs, maps=(None, None)) -> list:
    """What the re-emit would move that no serving template writes: the bundle files first
    (added, removed or altered), then the manifest keys, as `MANIFEST.<dotted.path>`. `maps`
    are the package's old and new `blueprint-map.json` documents, which may differ only in
    the outputs' entries; without them a changed map is a moved file."""
    map_rel = f"{outputs.package}/{MAP}"
    moved = [rel for rel in changed_files(old, new) if not outputs.holds(rel)
             and not (rel == map_rel and not _map_moved(maps, outputs))]
    return moved + _moved_keys(old, new, outputs)


def changed_files(old: dict, new: dict) -> list:
    """Every bundle file whose pin differs between two manifests (added and removed too)."""
    before, after = old.get("files") or {}, new.get("files") or {}
    return sorted(r for r in set(before) | set(after) if before.get(r) != after.get(r))


def _moved_keys(old: dict, new: dict, outputs) -> list:
    moved = []
    for key in sorted(set(old) | set(new)):
        if key == PINS:                     # compared file by file, the map by its entries
            continue
        if key in OUTPUT_LISTINGS:
            if _outside(old.get(key), outputs) != _outside(new.get(key), outputs):
                moved.append(f"MANIFEST.{key}")
            continue
        moved += _diff(old.get(key, _ABSENT), new.get(key, _ABSENT), key)
    return moved


def _outside(listing, outputs):
    """A listing of bundle files (a list, or a mapping by file) without the outputs."""
    if isinstance(listing, dict):
        return {k: v for k, v in listing.items() if not outputs.holds(k)}
    return [rel for rel in listing or () if not outputs.holds(rel)]


def _diff(a, b, path: str) -> list:
    if path in MAY_CHANGE:
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        return [moved for key in sorted(set(a) | set(b))
                for moved in _diff(a.get(key, _ABSENT), b.get(key, _ABSENT), f"{path}.{key}")]
    return [] if a is b or a == b else [f"MANIFEST.{path}"]


def _map_moved(maps, outputs) -> bool:
    old, new = maps
    if not isinstance(old, dict) or not isinstance(new, dict):
        return True
    return _map_rest(old, outputs) != _map_rest(new, outputs)


def _map_rest(doc: dict, outputs) -> dict:
    """The map without the outputs' entries (package-relative files) and entry points
    (bundle-relative files)."""
    rest = dict(doc)
    rest["entries"] = [e for e in doc.get("entries") or [] if not (
        isinstance(e, dict) and outputs.holds(f"{outputs.package}/{e.get('file')}"))]
    rest["entrypoints"] = [e for e in doc.get("entrypoints") or [] if not (
        isinstance(e, dict) and outputs.holds(str(e.get("file"))))]
    return rest
