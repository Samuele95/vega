"""
bundle_inputs — the files the bundle is built from, and the transformer it runs them through.

    copy_inputs(campaign, paths)                      netspec.json, weights.pt, recipe.json, fitted.json
    emit_transformer(options, paths, markers)         the shaper's --transform-only; PackExit(2)
    twin_check(options, campaign, paths, markers)     the shaper's own test; PackExit(3)
    check_feature_contract(campaign, markers)         PackExit(2) when the spec names another shape
    read_raw(campaign, recipe)                        the raw rows the recipe consumed; PackExit(2)
    project_fitted(paths, fitted, mode, markers)      fitted.json as every bundle ships it
    without_membership(shape_layer, markers)          records/shape.json without row positions
    transformer_requirements(pkg_dir)                 [(distribution, version)] the transformer imports

`features/preprocess.py` is emitted by `dataset-shaper/scripts/shape.py --transform-only`,
never written here: a hand-rolled preprocessing is a second implementation of the recipe, and
it will drift from the one the network was trained on.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from bundle_map import add_map_entries
from bundle_slice import transformer_file
from pack_context import SKILLS, PackExit, out_ok, out_warn
from vega_pack_core import digest_file, read_parquet

__all__ = ["MEMBERSHIP_KEYS", "MEMBERSHIP_PARAMS", "check_feature_contract", "copy_inputs", "emit_transformer",
           "project_fitted", "read_raw", "shaper_dir", "transformer_requirements", "twin_check",
           "without_membership"]

# The keys of fitted.json that record WHICH rows went where — the split's membership, the
# removed rows — are the shaper's audit trail, not parameters: no transform reads them, and in
# a serving bundle they would ship the sealed test part's membership index to every caller.
MEMBERSHIP_KEYS = ("__split_raw__", "__split__", "__removed_rows__")


def copy_inputs(campaign, paths) -> None:
    sources = campaign.sources
    shutil.copy2(sources["netspec"], paths.root / "netspec.json")
    shutil.copy2(sources["weights"], paths.root / "weights.pt")
    shutil.copy2(sources["recipe"], paths.root / "recipe.json")
    shutil.copy2(sources["fitted"], paths.root / "fitted.json")


def shaper_dir(options) -> Path:
    return Path(options.dataset_shaper_dir) if options.dataset_shaper_dir \
        else SKILLS / "dataset-shaper"


def emit_transformer(options, paths, markers) -> None:
    """`<package>/<features module>/preprocess.py` (`features/` unless the map names another
    module for the FeatureTransformer), emitted by the shaper from the bundle's recipe and
    fitted parameters, or `PackExit(2)`."""
    shaper = shaper_dir(options) / "scripts" / "shape.py"
    if not shaper.is_file():
        raise PackExit(2, f"dataset-shaper lacks its executor at {shaper}; the transformer "
                          "comes from --transform-only and is never re-implemented here")
    target = transformer_file(paths.pkg)
    proc = subprocess.run(
        [sys.executable, str(shaper), "--transform-only", "--recipe",
         str(paths.root / "recipe.json"), "--fitted", str(paths.root / "fitted.json"),
         "--emit", str(target)],
        capture_output=True, text=True, timeout=600)
    for line in (proc.stdout or "").splitlines():
        if line.startswith("OK:"):
            markers.append(line)
    if proc.returncode != 0:
        raise PackExit(2, f"the shaper refused to emit the transformer (exit "
                          f"{proc.returncode}): {(proc.stderr or proc.stdout).strip()[-400:]}")
    if not target.is_file():
        raise PackExit(2, f"the shaper reported success but wrote no {target.name}")
    steps = sorted(target.parent.glob(target.stem + "_steps*.py"))   # the shaper's C13 split
    if steps:
        add_map_entries(SimpleNamespace(paths=paths, markers=markers), "the shaper's "
                        "--transform-only", [p.relative_to(paths.pkg).as_posix() for p in steps])
    out_ok(f"{target.relative_to(paths.pkg).as_posix()} emitted by the shaper's "
           "--transform-only; it is the "
           "same recipe the training program ran, not a second implementation", markers)


def twin_check(options, campaign, paths, markers) -> None:
    """The shaper's own test of the transformer, and the only one that does not go through
    the network. Run before the round trip so a failure names the transform rather than the
    weights; `PackExit(3)` when it fails."""
    verifier = shaper_dir(options) / "scripts" / "verify_shape.py"
    part = "test"
    proc = subprocess.run(
        [sys.executable, str(verifier), "--transform-only", "--recipe",
         str(paths.root / "recipe.json"), "--out-dir", str(campaign.shaped_dir), "--module",
         str(transformer_file(paths.pkg)), "--part", part],
        capture_output=True, text=True, timeout=900)
    if proc.returncode == 0:
        out_ok(f"the shaper's twin check passed on the {part} part — the bundle's transform "
               "reproduces the shaped columns it was built from", markers)
        return
    for line in (proc.stdout or "").strip().splitlines()[-3:]:
        out_warn(f"twin check: {line}", markers)
    raise PackExit(3, "the shaper's twin check failed; the bundle's transform does not "
                      "reproduce the shaped columns it was built from, so nothing below would "
                      "mean anything")


def check_feature_contract(campaign, markers) -> None:
    """The FeatureContract admits only the manifest it was built from (EJ-49): checked here,
    once, because a bundle whose spec names another shape would transform correctly and feed
    the network columns it was not trained on."""
    spec_contract = ((campaign.spec or {}).get("input") or {}).get("manifest_digest")
    manifest_file_digest = digest_file(campaign.shaped_dir / "manifest.json")
    if spec_contract and spec_contract != manifest_file_digest:
        raise PackExit(2, f"the spec's FeatureContract was built from {spec_contract}, but the "
                          f"shaped manifest digests to {manifest_file_digest}; the bundle would "
                          "pair a network with a shape it never saw")
    out_ok("feature contract: the spec's manifest digest is the shaped manifest's"
           + (" (the spec stamped none)" if not spec_contract else ""), markers)


def read_raw(campaign, recipe):
    """The raw rows the recipe consumed, found where the chain ran, or `PackExit(2)`."""
    import pandas as pd
    raw_path = Path(recipe.get("input", {}).get("path", ""))
    if not raw_path.is_absolute():
        # the recipe names its input against the project root the chain ran in; a campaign
        # sits at <root>/vega-runs/<ts>, so the root is two levels up, not one
        candidates = [campaign.shaped_dir / raw_path, campaign.dir.parent / raw_path,
                      campaign.dir.parent.parent / raw_path, Path.cwd() / raw_path]
        raw_path = next((c for c in candidates if c.is_file()), candidates[-1])
    if not raw_path.is_file():
        raise PackExit(2, f"the recipe's input {raw_path} is not there; the round trip runs on "
                          "the raw rows, which is the whole of what it proves")
    return pd.read_csv(raw_path) if raw_path.suffix == ".csv" else read_parquet(raw_path)


def project_fitted(paths, fitted_all, mode, markers) -> dict:
    """fitted.json as the serving bundle ships it: every parameter the emitted Recipe adapter
    reads, none of the row-membership records. Derived, not listed: the keys kept are the ones
    `preprocess.py` indexes (`FITTED['S9']` …), so a recipe with other steps projects itself.
    The round trip that follows runs against the projection — that is its test. A `full`
    bundle is projected the same way: its train.py and evaluate.py read the fitted file beside
    the shaped parts they are given (`--shaped`), never the bundle's, so nothing it ships needs
    the membership — and shipped, the membership would tell its holder which raw rows were the
    sealed test part. `mode` is the bundle shape asked for; both get the same projection."""
    source = digest_file(paths.root / "fitted.json")
    module = transformer_file(paths.pkg)
    code = "".join(p.read_text(encoding="utf-8") for p in [module, *sorted(module.parent.glob(
        module.stem + "_steps*.py"))])                       # with the shaper's C13 split
    used = set(re.findall(r"FITTED\[['\"]([^'\"]+)['\"]\]", code)) | \
        set(re.findall(r"FITTED\.get\(['\"]([^'\"]+)['\"]", code))
    kept = {k: v for k, v in fitted_all.items() if k in used}
    missing = sorted(used - set(kept))
    if missing:
        raise SystemExit(f"ERROR: preprocess.py reads fitted key(s) {', '.join(missing)} that "
                         "fitted.json does not carry")
    dropped = sorted(set(fitted_all) - set(kept))
    (paths.root / "fitted.json").write_text(json.dumps(kept, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    record = {"digest_source": source, "digest": digest_file(paths.root / "fitted.json"),
              "projection": "serving", "kept": sorted(kept),
              "dropped": dropped,
              "why": "row membership (the split, the removed rows) is the shaper's audit trail, "
                     "read by no transform; shipped, it would disclose the sealed test part's "
                     "membership"}
    out_ok(f"fitted.json projected for serving: {len(kept)} step(s) the transform reads kept, "
           f"{len(dropped)} dropped ({', '.join(k for k in dropped if k in MEMBERSHIP_KEYS)} "
           "among them) — the source file's digest is kept in the manifest", markers)
    return record


# what the shape layer's split and dedupe steps record of WHICH rows went where, the same
# membership as MEMBERSHIP_KEYS: `records/shape.json` ships without it (bundle-contract §1)
MEMBERSHIP_PARAMS = ("indices", "raw_indices", "raw_rows_removed")


def without_membership(doc, markers):
    """The shape layer's data with the row-membership parameters of every step left out (the
    recipe's and the manifest's alike) — the record keeps the counts (`sizes`,
    `rows_removed`), never the positions."""
    doc, dropped = json.loads(json.dumps(doc)), set()
    for holder in (doc.get("recipe"), doc.get("manifest")):
        for step in (holder or {}).get("steps") or ():
            fitted = step.get("parameters_fitted") if isinstance(step, dict) else None
            for key in MEMBERSHIP_PARAMS if isinstance(fitted, dict) else ():
                if fitted.pop(key, None) is not None:
                    dropped.add(f"{step.get('op')}.{key}")
    if dropped:
        out_ok(f"records/shape.json carries no row membership: {', '.join(sorted(dropped))} "
               "left out (the counts stay)", markers)
    return doc


# the distributions every requirements file of a bundle declares already
_DECLARED = frozenset({"torch", "pandas", "numpy", "pyarrow"})


def transformer_requirements(pkg_dir) -> list:
    """[(distribution, version)] of every third-party module the FeatureTransformer imports —
    at module level or inside a step's function — that the bundle does not declare already.

    `preprocess.py` is the shaper's emitted recipe, so its imports depend on the recipe: a
    Box-Cox or Yeo-Johnson step imports scipy inside its function. The bundle's requirements
    were a fixed list, and the release4 rehearsal shipped a Box-Cox bundle whose
    requirements-serve.txt, pyproject and Dockerfile installed no scipy: the start-up digest
    check passed and the first transform died with ImportError (F41). Each module is mapped
    to its distribution by the packing environment's metadata and pinned to the version the
    round trip ran with; one the environment cannot resolve is returned unpinned."""
    import ast
    import importlib.metadata as md
    path = transformer_file(pkg_dir)
    if not path.is_file():
        return []
    tops = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            tops.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            tops.add(node.module.split(".")[0])
    own = {path.parent.parent.name}
    dists = md.packages_distributions()
    out = {}
    for top in sorted(tops - set(sys.stdlib_module_names) - own - {"__future__"}):
        for dist in dists.get(top) or [top]:
            name = dist.lower()
            if name in _DECLARED or name in out:
                continue
            try:
                out[name] = md.version(dist)
            except md.PackageNotFoundError:
                out[name] = None
    return sorted(out.items())
