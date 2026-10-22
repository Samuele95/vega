"""
reemit — where the bundle is built, and how it is written into place (bundle-contract §7.8,
R-D11).

    paths = reemit.prepare(options, paths, markers)      before anything is written
    paths = reemit.finish(ctx, manifest)                 the manifest, and the bundle in place
    moved = reemit.compare(old, new, outputs, maps)      what moved and may not (reemit_proof)
    altered = reemit.altered(root)                       the files a bundle no longer pins
    staging = reemit.stage_dir(root)                     where a re-emit builds
    reemit.swap(root, staging)                           the staged bundle put in place

A plain pack rebuilds the bundle directory from scratch: `prepare` clears it, `finish` writes
`MANIFEST.json` into it (with `reemits: []`, the list every accepted re-emit appends to).

`--re-emit-serving` is the same pack run with a proof attached, not a second pipeline. The
bundle in place must still be the one its manifest pins (every pinned file at its digest, else
exit 2: a re-emit compares against those pins, and would drop a hand edit without a word). The
whole pack then runs again, from the same inputs, into a staging directory that keeps the
bundle's own name (`<bundle-dir>/.reemit/<slug>-<run>/`) — the same provenance, transformer,
twin check, round trip, examples, monitor and gate — and `finish` compares the two manifests.

What may move is what the serving templates write and the exports (`reemit_proof`: the
catalog's patterns at their destinations; the listings of bundle files — `files`,
`entrypoints`, the package map — only where they name one of those; the catalog's free
manifest keys). **Everything else must come out byte-identical**: the weights, the spec, the
recipe and its fitted parameters, the schema, the monitor reference, the examples, the
reference graph, the records, the requirements, and every package file outside the serving
outputs. Anything that moved refuses the re-emit (exit 3): the staging directory is deleted
and the bundle in place is left as it was.

When the proof holds, the stage prints and records the gate line (`GATE: replacing <root> …`;
the flag is the consent — attended, the stage asks before it runs the flag), the files other
stages wrote beside the bundle (the model card, the notebook, the layer's authoring file, a
showcase) are carried over, the staged bundle replaces the old one (`root -> .prev`, `staging
-> root`, `.prev` removed; the old bundle is put back if the second rename fails), and
`MANIFEST.reemits[]` gains `{at, reason, previous_manifest_digest, templates_digest,
changed}`. A showcase built on the previous manifest is then stale: the slot and `/vega-show
--check` find its `bundle.manifest_digest` among the `previous_manifest_digest`s (R-D11). One
re-emit of a bundle runs at a time: the staging directory is the bundle's own.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from bundle_manifest import UNPINNED
from pack_context import PackExit, out_ok
from reemit_proof import MAP, MAY_CHANGE, NEVER, ServingOutputs, changed_files, compare
from serving_templates import templates_digest
from vega_pack_core import CATALOG, BundlePaths, digest_file, now, read_json

__all__ = ["MAY_CHANGE", "NEVER", "ServingOutputs", "altered", "changed_files", "compare",
           "finish", "prepare", "stage_dir", "swap"]

STAGING = ".reemit"
RETIRED = ".prev"
REFUSAL = CATALOG["reemit"]["refusal"].split(" (exit")[0].removeprefix("ERROR: ")
NOT_PINNED = "the bundle is not the one its MANIFEST pins; re-pack instead"
# the manifest keys that echo an input: named in the refusal when they moved
INPUT_ECHOES = {"blueprint.run": "--pipeline", "chain.model": "--model", "include": "--include"}
SHOWN = 10


# --------------------------------------------------------------------------- the bundle in place

def altered(root) -> list:
    """The files the bundle at `root` no longer holds as its MANIFEST.json pins them — every
    file of every role, as the bundle's own `predict.verify` reads them — or why the manifest
    cannot say."""
    manifest, err = read_json(Path(root) / "MANIFEST.json", "MANIFEST.json")
    if err:
        return ["MANIFEST.json (unreadable)"]
    files, roles = manifest.get("files") or {}, manifest.get("roles") or {}
    if not roles:
        return ["MANIFEST.json (it records no file roles)"]
    listed = {rel for rels in roles.values() for rel in rels or ()}
    return [rel for rel in sorted(set(files) | listed)
            if rel not in files or not (Path(root) / rel).is_file()
            or digest_file(Path(root) / rel) != files[rel]]


def stage_dir(root) -> Path:
    """Where a re-emit of the bundle at `root` builds: beside it, under its own name."""
    root = Path(root)
    return root.parent / STAGING / root.name


def _final_root(staging) -> Path:
    staging = Path(staging)
    return staging.parent.parent / staging.name


def prepare(options, paths, markers) -> BundlePaths:
    """The directory this run builds into: the bundle itself, cleared, or — for a re-emit —
    a staging directory beside it, with the bundle in place checked and left untouched."""
    if not options.re_emit_serving:
        if paths.root.exists():
            shutil.rmtree(paths.root)
        paths.root.mkdir(parents=True)
        return paths
    if not paths.manifest.is_file():
        raise PackExit(2, f"no bundle to re-emit: --re-emit-serving re-emits the bundle at "
                          f"{paths.root}, and there is none there (no MANIFEST.json); pack it "
                          "first")
    moved = altered(paths.root)
    if moved:
        raise PackExit(2, (NOT_PINNED, f"{paths.root}: {_listed(moved)} — not what "
                                       "MANIFEST.json pins; a re-emit proves itself against "
                                       "those pins, and would drop the edit without a word"))
    staging = stage_dir(paths.root)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    out_ok(f"re-emit: the bundle in place verifies against its MANIFEST.json; building beside "
           f"it, in {staging}; it is replaced only if nothing the evaluation pinned moves",
           markers)
    return BundlePaths(root=staging, package=paths.package)


# --------------------------------------------------------------------------- the end of a run

def finish(ctx, manifest) -> BundlePaths:
    """`MANIFEST.json` written; for a re-emit, the proof checked and the bundle swapped in.
    Returns the paths of the bundle as it now stands."""
    if not ctx.options.re_emit_serving:
        manifest.setdefault("reemits", [])
        _write(ctx.paths.manifest, manifest)
        return ctx.paths
    final = BundlePaths(root=_final_root(ctx.paths.root), package=ctx.paths.package)
    previous = _previous(final, ctx.paths.root)
    outputs = ServingOutputs.of(ctx, previous, manifest)
    moved = compare(previous, manifest, outputs, _maps(final, ctx.paths))
    if moved:
        _discard(ctx.paths.root)
        raise PackExit(3, (REFUSAL.replace("<files>", _listed(moved)),
                           *_echo_hints(moved, previous, manifest),
                           f"the bundle at {final.root} is left as it was"))
    entry = _entry(ctx, final, previous, manifest)
    manifest["reemits"] = list(previous.get("reemits") or []) + [entry]
    _write(ctx.paths.manifest, manifest)
    _gate(ctx, final, entry)
    _carry_over(final.root, ctx.paths.root, previous.get("files") or {})
    try:
        swap(final.root, ctx.paths.root)
    except OSError as exc:
        _discard(ctx.paths.root)
        kept = final.root if final.root.is_dir() else _retired(final.root, ctx.paths.root)
        raise PackExit(3, (f"the re-emitted bundle could not be put in place ({exc})",
                           f"the previous bundle is at {kept}, as it was")) from exc
    _done(ctx, entry)
    return final


def _previous(final, staging) -> dict:
    """The manifest of the bundle in place, re-read and re-verified now: a bundle edited while
    the re-emit ran is not the one it was proven against."""
    moved = altered(final.root) if final.manifest.is_file() else ["MANIFEST.json"]
    if moved:
        _discard(staging)
        raise PackExit(3, (f"{NOT_PINNED}: {_listed(moved)} changed while the re-emit ran",
                           f"the bundle at {final.root} is left as it is"))
    return json.loads(final.manifest.read_text(encoding="utf-8"))


def _maps(final, staged) -> tuple:
    return (read_json(final.pkg / MAP, MAP)[0], read_json(staged.pkg / MAP, MAP)[0])


def _entry(ctx, final, previous, manifest) -> dict:
    return {"at": now(), "reason": ctx.options.reason,
            "previous_manifest_digest": digest_file(final.manifest),
            "templates_digest": templates_digest(ctx.templates,
                                                 ctx.campaign.backend_name)["digest"],
            "changed": changed_files(previous, manifest)}


def _gate(ctx, final, entry) -> None:
    """The gate line, printed and recorded with the run's markers."""
    changed = entry["changed"]
    consent = ("unattended: --re-emit-serving is the consent" if ctx.options.unattended
               else "--re-emit-serving asked for it")
    line = (f"GATE: replacing {final.root} with the re-emitted serving adapters "
            f"({len(changed)} file(s)" + (f": {_listed(changed)}" if changed else "")
            + f") — {consent}")
    ctx.markers.append(line)
    print(line)


def _done(ctx, entry) -> None:
    n = len(entry["changed"])
    what = (f"{n} file(s) changed, each one a serving template output or an export" if n
            else "no file changed")
    out_ok(f"re-emitted in place: {what} — nothing the evaluation pinned moved; "
           f"MANIFEST.reemits records the previous manifest "
           f"{entry['previous_manifest_digest'][:19]}…, and a showcase built on it is now "
           "stale (re-run /vega-show)", ctx.markers)


def _listed(names) -> str:
    names = list(names)
    more = len(names) - SHOWN
    return ", ".join(names[:SHOWN]) + (f" and {more} more" if more > 0 else "")


def _echo_hints(moved, previous, manifest) -> list:
    """For each moved key that echoes an input, the two values: a re-emit takes the pack's own
    inputs, and a different one is a different bundle."""
    hints = []
    for path, flag in INPUT_ECHOES.items():
        if f"MANIFEST.{path}" in moved:
            was, given = (_at(doc, path) for doc in (previous, manifest))
            hints.append(f"the bundle was packed with {flag} {was!r} and this run was given "
                         f"{given!r}: a re-emit takes the inputs of the pack that built it")
    return hints


def _at(doc, dotted):
    for key in dotted.split("."):
        doc = doc.get(key) if isinstance(doc, dict) else None
    return doc


def _write(path, manifest) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


# --------------------------------------------------------------------------- the swap

def _carry_over(old_root, new_root, packed) -> None:
    """The files other stages wrote beside the bundle — pinned by no manifest and not the
    pack's own unpinned files — copied into the staged bundle as they are (a symlink stays a
    link, for the slot to refuse as it would have)."""
    old_root, new_root = Path(old_root), Path(new_root)
    for dirpath, dirnames, filenames in os.walk(old_root):
        base = Path(dirpath)
        links = [d for d in dirnames if (base / d).is_symlink()]
        dirnames[:] = [d for d in dirnames if d not in links and d != "__pycache__"]
        for name in filenames + links:
            rel = (base / name).relative_to(old_root).as_posix()
            target = new_root / rel
            if rel in packed or rel in UNPINNED or target.exists() or target.is_symlink():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base / name, target, follow_symlinks=False)


def swap(root, staging) -> None:
    """`staging` becomes `root`: the old bundle is renamed aside, the staged one renamed in,
    the old one removed. If the second rename fails, the old bundle is put back."""
    root, staging = Path(root), Path(staging)
    retired = _retired(root, staging)
    if retired.exists():
        shutil.rmtree(retired)
    root.rename(retired)
    try:
        staging.rename(root)
    except OSError:
        retired.rename(root)
        raise
    _discard(retired)


def _retired(root, staging) -> Path:
    return Path(staging).parent / (Path(root).name + RETIRED)


def _discard(directory) -> None:
    """A staging or retired directory removed, and the staging area with it once empty."""
    shutil.rmtree(directory, ignore_errors=True)
    try:
        Path(directory).parent.rmdir()
    except OSError:
        pass
