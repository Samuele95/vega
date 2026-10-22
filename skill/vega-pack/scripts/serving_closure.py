"""
serving_closure — the Model side the serving package needs, carried into the bundle.

    copy_model_side(ctx) -> [file]      package-relative files copied from the campaign's package

The catalog's serving slice names the adapter files (`model/network.py`, `model/store.py`, …),
and a v2.3 adapter is not one file: the network is built by `builder.py` from the spliced
`blocks_*.py` and answers through `readout.py`, the Predictor's support files read `readout.py`
and `assemble.py` (the one sample assembler, R-D15), an export goes through `export.py`, a custom
block's code (`<model>/custom/<stem>.py`) is read by file. Two rules decide what travels:

  * **the map**: every file the package's `blueprint-map.json` lists in a module the bundle
    serves — the support files beside the adapters the slice copied — since the conformance
    gate holds the bundle to its map (C3: a mapped file the package lacks is a defect); the
    ports that go out of scope (the data source, the trainer, the evaluator, the evaluation's
    baselines) and the modules `catalog.json` `drop_packages` names are not served;
  * **the imports**: the closure of every relative import of every module the bundle holds,
    after the serving templates are rendered, resolved against the bundle and then against the
    package the evaluated run was trained with, followed until nothing new is reached — so a
    file a template needs travels even when a map omits it.

A file is copied from that package byte for byte and checked against the ledger's code digests
like the rest of the slice (`bundle_slice.verify_code`). The closure never reaches past the
serving slice: in a `serving` bundle, a served module that imports the data source, the trainer
or the evaluator is a defect in a template, and the pack stops (exit 3) rather than ship the
door to the test part.
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path, PurePosixPath

from bundle_slice import verify_code
from pack_context import PackExit, out_ok
from vega_pack_core import CATALOG, read_json

__all__ = ["copy_model_side", "relative_imports"]


def copy_model_side(ctx) -> list:
    """Copy what the bundle's package maps or imports and does not hold; the files copied."""
    src, dst = ctx.campaign.src_pkg, ctx.paths.pkg
    copied = [f for rel in _mapped(ctx) if not (dst / rel).is_file() and (src / rel).is_file()
              for f in _copy(src, dst, rel)]
    copied += _closure(ctx, src, dst)
    if copied:
        verify_code(ctx.campaign, ctx.paths, copied, ctx.markers)
        out_ok(f"the Model side the serving package maps and imports, from the evaluated "
               f"run's package: {', '.join(copied)}", ctx.markers)
    return copied


def _mapped(ctx) -> list:
    """The files the copied map lists in the modules this bundle serves (every one, for a
    `full` bundle): not an out-of-scope port's adapter, not a dropped module's file, and not a
    support file no served role reads — one whose `used_by` roles are all out of scope (the
    metrics the Trainer and the Evaluator read), or the ArtifactExporter's module in a bundle
    that exports nothing (`--export`): what the bundle carries is the code that answers."""
    doc, _err = read_json(ctx.paths.pkg / "blueprint-map.json", "blueprint-map.json")
    full = ctx.options.include == "full"
    skipped = set(CATALOG["package_slice"]["out_of_scope"]) | {"baseline"}
    dropped = set(CATALOG["package_slice"]["drop_packages"])
    return [e["file"] for e in (doc or {}).get("entries") or []
            if isinstance(e, dict) and str(e.get("file") or "").endswith(".py")
            and (full or (e.get("role") not in skipped
                          and e["file"].split("/")[0] not in dropped
                          and _served(e, skipped, bool(ctx.options.export))))]


def _served(entry, skipped, exports: bool) -> bool:
    """Whether a role the bundle serves reads this map entry's file."""
    if entry.get("kind") != "support":
        return True
    if entry.get("implements") == "ArtifactExporter" and not exports:
        return False
    used_by = set(entry.get("used_by") or ())
    return not used_by or bool(used_by - skipped)


def _closure(ctx, src: Path, dst: Path) -> list:
    """What the bundle's modules import, relatively, and the bundle does not hold."""
    todo = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*.py")
                  if "__pycache__" not in p.parts)
    seen, copied = set(todo), []
    while todo:
        rel = todo.pop()
        for module in relative_imports((dst / rel).read_text(encoding="utf-8"), rel):
            found = _module_file(module, dst, src)
            if found is None or found in seen:
                continue
            seen.add(found)
            if not (dst / found).is_file():
                _refuse_dropped(ctx, rel, found)
                copied += _copy(src, dst, found)
            todo.append(found)
    return copied


def relative_imports(text: str, rel: str) -> set:
    """The package-relative module paths (`model/readout`) a module's relative imports may
    name: `from ..model.readout import X` names `model/readout` and, were X a submodule,
    `model/readout/X`; `from . import registry` names `<its package>/registry`."""
    here = PurePosixPath(rel).parent
    depth = 0 if str(here) == "." else len(here.parts)
    out = set()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.ImportFrom) or not node.level or node.level - 1 > depth:
            continue
        base = PurePosixPath(*here.parts[:depth - (node.level - 1)]) if depth - (node.level - 1) \
            else PurePosixPath(".")
        target = base.joinpath(*node.module.split(".")) if node.module else base
        if node.module:
            out.add(str(target))
        out |= {str(target / alias.name) for alias in node.names if alias.name != "*"}
    return out


def _module_file(module: str, dst: Path, src: Path):
    """The package-relative file of a module path, in the bundle or else the source package;
    None for a name that is no module of either (a class, a function)."""
    for name in (f"{module}.py", f"{module}/__init__.py"):
        rel = str(PurePosixPath(name))
        if (dst / rel).is_file() or (src / rel).is_file():
            return rel
    return None


def _refuse_dropped(ctx, importer: str, rel: str) -> None:
    top = rel.split("/")[0]
    if ctx.options.include != "full" and top in CATALOG["package_slice"]["drop_packages"]:
        raise PackExit(3, f"{ctx.package}/{importer} imports {ctx.package}/{rel}, which the "
                          f"serving slice leaves behind ({top}/ is not part of serving): a "
                          "served package must not reach it — fix the template that imports it")


def _copy(src: Path, dst: Path, rel: str) -> list:
    """`rel` copied, and the `__init__.py` of each directory it opens in the bundle."""
    copied = []
    parts = PurePosixPath(rel).parts
    for i in range(1, len(parts)):
        init = str(PurePosixPath(*parts[:i], "__init__.py"))
        if not (dst / init).is_file() and (src / init).is_file():
            copied += _copy_one(src, dst, init)
    return copied + _copy_one(src, dst, rel)


def _copy_one(src: Path, dst: Path, rel: str) -> list:
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / rel, dst / rel)
    return [rel]
