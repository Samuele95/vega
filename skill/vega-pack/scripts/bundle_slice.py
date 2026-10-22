"""
bundle_slice — what of the emitted package the bundle carries, and its prose made true of it.

    copy_package_slice(src_pkg, dst_pkg, markers, mode) -> copied    the start of the pack
    verify_code(campaign, paths, copied, markers)                     PackExit(3) on drift
    bundle_modules(pkg_dir) -> {key: module}                          the blueprint's modules
    finish(ctx)                                                       the end: map, docs, entry points

The slice is resolved through the package's own `blueprint-map.json` (a pipeline blueprint may
put the network in `nets/` and the store in `registry/`), never assumed. At the end of the pack,
after every adapter was rendered and every entry point written, `finish` has `bundle_map`
rewrite the map, rewrites the package's own prose (`ports.py`, the package docstring) from it,
and marks the entry points present — so all three describe the bundle as shipped; the
conformance gate reads them next.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from bundle_map import mark_entrypoints, rewrite_map
from pack_context import SKILLS, PackExit, out_ok
from vega_pack_core import CATALOG, digest_file, read_json

__all__ = ["bundle_modules", "copy_package_slice", "finish", "rewrite_slice_docs", "transformer_file",
           "verify_code"]


# --------------------------------------------------------------------------- the slice

def _slice_paths(src_pkg: Path, keep) -> list:
    """The catalog's serving slice, translated into this package's own layout.

    The catalog names the emitter's default layout (`model/network.py`, `serving/monitor.py`);
    a pipeline blueprint may put the network in `nets/`, the store in `registry/` and the
    monitor in `monitoring/`. The package's blueprint-map.json says which file implements each
    role, so the slice is resolved through it rather than assumed — a file the map places
    elsewhere would otherwise be silently left out of the bundle.
    """
    doc, _err = read_json(Path(src_pkg) / "blueprint-map.json", "blueprint-map.json")
    by_role = {e.get("role"): e.get("file") for e in ((doc or {}).get("entries") or [])
               if e.get("role") and e.get("file")}
    model = by_role.get("Model")
    translate = {"model/network.py": model,
                 "model/store.py": by_role.get("ModelStore"),
                 "model/metrics.py": (str(Path(model).parent / "metrics.py") if model else None),
                 "features/preprocess.py": by_role.get("FeatureTransformer"),
                 "serving/predict.py": by_role.get("Predictor"),
                 "serving/monitor.py": by_role.get("Monitor")}
    out = []
    for rel in keep:
        actual = translate.get(rel) or rel
        if actual not in out:
            out.append(actual)
    return out


def copy_package_slice(src_pkg: Path, dst_pkg: Path, markers, mode="serving") -> list:
    """What of the emitted package the bundle carries, which is what `--include` decides.

    `serving` copies the serving slice and leaves `ingestion/`, `training/` and `evaluation/`
    behind — not for containment, which they do not provide (`fitted.json` names the split and
    has to, or the round trip could not run), but because a served model has no use for them
    and an evaluator sitting beside a model with no seal to answer to is an invitation to
    re-score a part someone already spent their one look on.

    `full` copies the whole package, because a user who will re-fit on their own data needs
    the loop, the data source and the evaluator as code. The seal is not weakened by that:
    `load_part` refuses the test part in the bundle exactly as it does in the chain, and
    `load_sealed_part` still wants a `test_seal.json` and its token. The door travels with
    its lock.
    """
    keep = _slice_paths(src_pkg, CATALOG["package_slice"]["keep"])
    dropped = CATALOG["package_slice"]["drop_packages"]
    dst_pkg.mkdir(parents=True, exist_ok=True)
    if mode == "full":
        copied = _copy_whole(src_pkg, dst_pkg)
        absent = [m for m in keep if m not in copied]
        out_ok(f"the whole emitted package copied — {len(copied)} file(s), "
               f"{', '.join(dropped)} included (--include full): the seal travels with them, "
               "since load_part refuses the test part here as it does in the chain"
               + (f"; the source package holds no {', '.join(absent)}" if absent else ""),
               markers)
        return copied
    copied = _copy_slice(src_pkg, dst_pkg, keep)
    out_ok(f"the serving slice copied — {len(copied)} file(s); "
           f"{', '.join(dropped)} left behind: a served model has no use for them, and an "
           "evaluator with no seal to answer to invites a second look at a part someone "
           "already spent their one look on (--include full ships them, seal included)",
           markers)
    return copied


def _copy_whole(src_pkg, dst_pkg) -> list:
    copied = []
    for src in sorted(src_pkg.rglob("*")):
        if src.is_dir() or "__pycache__" in src.parts or src.suffix not in (".py", ".json"):
            continue
        rel = src.relative_to(src_pkg).as_posix()
        target = dst_pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(rel)
    return copied


def _copy_slice(src_pkg, dst_pkg, keep) -> list:
    copied = []
    for rel in keep:
        src = src_pkg / rel
        if not src.is_file():
            continue
        target = dst_pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(rel)
    for sub in {Path(r).parent for r in copied if "/" in r} | {Path(".")}:
        init = dst_pkg / sub / "__init__.py"
        if not init.exists():
            src_init = src_pkg / sub / "__init__.py"
            if src_init.is_file():
                shutil.copy2(src_init, init)
            else:
                init.write_text("", encoding="utf-8")
    return copied


def verify_code(campaign, paths, copied, markers) -> None:
    """The code the bundle carries must be the code the ledger recorded for this run, byte for
    byte — the same rule as the spec and the checkpoint. An import check would only catch an
    alteration that breaks the syntax; a digest catches every one."""
    package, run_id = paths.package, campaign.run_id
    recorded = (campaign.ledger_row.get("code_digests")
                or (campaign.doc.get("input") or {}).get("code_digests") or {})
    drifted, verified = [], 0
    for rel in copied:
        want = recorded.get(f"{package}/{rel}")
        if not want:
            continue
        have = digest_file(paths.pkg / rel)
        if have != want:
            drifted.append(f"{package}/{rel} (recorded {want[:19]}…, on disk {have[:19]}…)")
        verified += 1
    if drifted:
        raise PackExit(3, "the package on disk is not the code the campaign recorded for "
                          f"{run_id}: {'; '.join(drifted)}. Something changed it after the run "
                          "— find what before shipping; the bundle would carry code nobody "
                          "trained or evaluated.")
    out_ok(f"code provenance: {verified} copied file(s) match the ledger's code digests for "
           f"{run_id}" + (f"; {len(copied) - verified} carry no recorded digest"
                          if len(copied) > verified else ""), markers)


def bundle_modules(pkg_dir) -> dict:
    """The blueprint's module for each role, read off the package's own blueprint-map.json.

    The entry points import the adapters by module, and the module names are the blueprint's
    (a pipeline blueprint may put the network in `nets/` and the store in `registry/`), not a
    layout this script assumes. Falls back to the emitter's default names only for a role the
    map does not list.
    """
    default = {"model": "model", "store": "model", "features": "features",
               "serving": "serving", "ingestion": "ingestion", "training": "training",
               "evaluation": "evaluation", "monitor": "serving"}
    doc, _err = read_json(Path(pkg_dir) / "blueprint-map.json", "blueprint-map.json")
    by_role = {e.get("role"): e for e in ((doc or {}).get("entries") or []) if e.get("role")}
    for role, key in (("Model", "model"), ("ModelStore", "store"),
                      ("FeatureTransformer", "features"), ("Predictor", "serving"),
                      ("DataSource", "ingestion"), ("Trainer", "training"),
                      ("Evaluator", "evaluation"), ("Monitor", "monitor")):
        entry = by_role.get(role) or {}
        if entry.get("module"):
            default[key] = entry["module"]
        elif entry.get("file") and "/" in entry["file"]:
            default[key] = entry["file"].split("/")[0]
    return default



def transformer_file(pkg_dir) -> Path:
    """The FeatureTransformer's file: `preprocess.py` in the module the package's map names for
    the role (`features/` by default) — the file the bundle's entry point imports as
    `<package>.<features module>.preprocess`, and the one the shaper's --transform-only writes."""
    return Path(pkg_dir) / bundle_modules(pkg_dir)["features"] / "preprocess.py"


# --------------------------------------------------------------------------- the end

def finish(ctx) -> None:
    """The map and the package's prose, made true of the bundle as it now is."""
    doc = rewrite_map(ctx.paths.pkg / "blueprint-map.json", ctx.copied, ctx.markers,
                      ctx.options.include)
    rewrite_slice_docs(ctx.paths, doc, ctx.markers, ctx.options.pipeline)
    mark_entrypoints(ctx.paths.pkg / "blueprint-map.json", ctx.markers, ctx.options.pipeline)


def rewrite_slice_docs(paths, doc, markers, pipeline=None) -> None:
    """The copied package's own prose, made true of the slice it now is.

    `ports.py` names each port's adapter with the status the forge wrote (`pending: /vega-pack`)
    and the package docstring lists every module and a `spec_snapshot.json` the bundle does not
    carry. The code is the ledger's (its digest was checked on copy); these lines are about
    the package, and after the slice they describe a different one.
    """
    status = {e.get("file"): e.get("status") for e in (doc.get("entries") or [])
              if e.get("file")}
    ports = paths.pkg / "ports.py"
    if ports.is_file():
        text = ports.read_text(encoding="utf-8")
        for rel, st in status.items():
            short = ("implemented" if st == "implemented" else
                     "out of scope in this bundle" if str(st).startswith("out of scope") else st)
            text = re.sub(r"(Adapter: " + re.escape(rel) + r" \()[^)]*(\))",
                          lambda m: m.group(1) + str(short) + m.group(2), text)
        text = _refresh_signatures(text, markers, _pipeline_doc(pipeline), paths)
        ports.write_text(text, encoding="utf-8")
    init = paths.pkg / "__init__.py"
    if init.is_file():
        present = sorted(d.name for d in paths.pkg.iterdir()
                         if d.is_dir() and (d / "__init__.py").is_file()
                         and d.name != "__pycache__")
        text = init.read_text(encoding="utf-8")
        text = re.sub(r"Modules: [^\n]*\.", "Modules in this bundle: " + ", ".join(present)
                      + ".", text, count=1)
        text = text.replace("the spec is `spec_snapshot.json`",
                            "the spec is `netspec.json` at the bundle root")
        init.write_text(text, encoding="utf-8")
    out_ok("ports.py adapter statuses and the package docstring rewritten for the slice",
           markers)


def _emitter(backend="pytorch"):
    """The backend's emitter module (its scoped role table), loaded by path; it imports no
    framework."""
    import importlib.util
    name = f"vega_pack_emit_{backend}"
    if name not in sys.modules:
        path = SKILLS / "vega-forge" / "scripts" / "backends" / backend / "emit.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _pipeline_doc(pipeline):
    if not pipeline:
        return {}
    doc, _err = read_json(Path(pipeline), "the blueprint's pipeline.json")
    return doc or {}


def _drawn_signatures(pipeline_doc, paths, backend="pytorch") -> dict:
    """{role: signature lines} of the bundle's ports: the blueprint's own `ports[].signature`
    (the baseline the bundle is audited against), else — a port the blueprint drew without one
    — the emitter's role table for this package's sample shape and task kind (a row package's
    DataSource has no graph exposure, a row recipe no edge entry points)."""
    emit = _emitter(backend)
    spec, _err = read_json(paths.root / "netspec.json", "the bundle's netspec.json")
    try:
        shape, task = emit.sample_shape(spec or {}), emit.effective_task(spec or {})
    except Exception:  # noqa: BLE001 — a spec the emitter cannot read: the unscoped table
        shape, task = None, None
    out = {role: list(emit._role_lines(role, shape, task)) for role in emit.ROLE_SIGNATURES}
    for port in pipeline_doc.get("ports") or []:
        drawn = [str(m) for m in (port.get("signature") or []) if str(m).strip()]
        if port.get("role") in out and drawn:
            out[port["role"]] = drawn
    return out


def _refresh_signatures(text, markers, pipeline_doc=None, paths=None, backend="pytorch"):
    """Each Protocol's methods, as the blueprint draws them (or, for a port it drew without a
    signature, the emitter's scoped role table).

    The serving adapters are rendered from today's templates; a ports.py still declaring the
    forge run's fallback signatures would describe a narrower port than the adapters implement,
    and one declaring the emitter's unscoped table a wider one than the package carries (a row
    recipe's edge entry points, a graph's exposure). The domain names the annotations read are
    imported as the bundle's domain.py defines them, and no other.
    """
    sigs = _drawn_signatures(pipeline_doc or {}, paths) if paths is not None else {}
    lines, out, i, refreshed = text.split("\n"), [], 0, []
    while i < len(lines):
        out.append(lines[i])
        m = re.match(r"\s+Role: (\w+)\.", lines[i])
        if m and m.group(1) in sigs:
            role = m.group(1)
            i += 1
            while i < len(lines) and not lines[i].startswith("    def "):
                out.append(lines[i])
                i += 1
            while i < len(lines) and lines[i].startswith("    def "):
                i += 1
            out.extend("    " + sig for sig in sigs[role])
            refreshed.append(role)
            continue
        i += 1
    text = "\n".join(out)
    held = _domain_names(paths)
    words = {w for sig in sum(sigs.values(), []) for w in re.findall(r"\b[A-Z]\w*\b", sig)}
    used = sorted(words & held)
    if used:
        text = re.sub(r"    from \.domain import \([^)]*\)|    from \.domain import [^\n]*",
                      lambda _m: "\n".join(_import_block(used)), text, count=1)
    markers.append("OK: ports.py signatures refreshed from the blueprint's ports for "
                   + ", ".join(refreshed))
    return text


def _domain_names(paths) -> set:
    """The top-level names the bundle's domain.py binds."""
    import ast
    if paths is None or not (paths.pkg / "domain.py").is_file():
        return set()
    tree = ast.parse((paths.pkg / "domain.py").read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _import_block(names) -> list:
    """`    from .domain import (A, B, ...)` wrapped at 99 columns, as ports.py writes it."""
    lines, cur = [], "    from .domain import ("
    indent = " " * len(cur)
    for i, n in enumerate(names):
        piece = n + (", " if i < len(names) - 1 else ")")
        if len(cur) + len(piece.rstrip()) > 99 and not cur.endswith("("):
            lines.append(cur.rstrip())
            cur = indent
        cur += piece
    lines.append(cur)
    return lines
