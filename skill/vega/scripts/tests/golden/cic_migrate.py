"""
cic_migrate — the recorded CIC-IDS2017 campaign as this release packs it (BUILD-PLAN R2, D6 §11.5).

The R7 campaign was trained and evaluated by the v2.2 chain, and its test part was opened once
under the seal (`test_seal.json`, one exposure). This release's `pack.py` reads three things the
v2.2 chain did not write, and none of them may be had by training or evaluating again:

- **the blueprint** — R-G2: every module that owns a port is `OOP`, every port carries its
  own `signature[]`, and every artifact a later stage holds outside the package is declared
  `external` with that stage (the patched blueprint SKILL.md; code-contract §3).
  `amend_blueprint` applies exactly those three rules to the recorded pipeline run the v2.2
  bundle was packed against, and nothing else: the style, the decomposition, the port names,
  the artifacts' names, the bets and the serving surface stay the base run's. The signatures
  are the v2.3 port interfaces of `lanekit.pipeline.port_signatures` by role — the ones the v2.3 emitter
  falls back to when a blueprint draws none; the owners are `EXTERNAL`'s, by artifact name (an
  artifact it does not name keeps its entry, and the emitter refuses it if it has no home).
- **the stored predictions** in `vega/predictions-layout@1` — `reencode_predictions` rewrites
  the v2.2 file's columns without computing anything: `row_id` becomes the raw-file position
  the shaper recorded for that row of the test part (`fitted.json.__split_raw__.test`, the
  mapping v2.2's own round trip used), the class logits `score_<k>` stay the head, `score` — a
  copy of the first class logit v2.2's evaluate.py wrote beside them — is dropped, the
  baselines' scores stay; the layout's metadata is written. Nothing reads the test part: the
  stored numbers are re-encoded, never recomputed.
- **the package** in the v2.3 generation — `migrate_campaign` re-emits R7's resolved spec with
  this release's emitter into `runs/R7/<package>/`, as a rung that re-emits does, with its
  `smoke.json`, and records the new code digests in the ledger row with the v2.2 ones kept
  under `migration`. The weights and the resolved spec are the recorded files, byte for byte
  (the pack's provenance check reads their recorded digests unchanged), and the proof that the
  re-emitted code is the evaluated network is the pack's round trip: every test row, from raw,
  against the stored predictions.

The migrated campaign is written under the lane's work dir; `$CIDS_ROOT` is only read
(`readonly_snapshot` / `readonly_changes` hold the golden lane to it).

    amend_blueprint(base_run_dir, out_dir) -> Path               the v2.3 pipeline.json
    reencode_predictions(src, dst, split_test) -> dict           the migration record
    migrate_campaign(cids, recorded, work, skills, pipeline, python, log) -> Migration
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from lanekit.pipeline import FILLED_BY, port_signatures  # noqa: E402

EMIT = TESTS.parents[2] / "vega-forge" / "scripts" / "backends" / "pytorch" / "emit.py"


def _emitter():
    """The emitter module (for `module_arrows`), loaded by path; it imports no framework."""
    import importlib.util
    name = "cic_migrate_emit"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, EMIT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

__all__ = ["Migration", "MigrationError", "amend_blueprint", "digest", "migrate_campaign",
           "readonly_changes", "readonly_snapshot", "reencode_predictions"]

LAYOUT = "vega/predictions-layout@1"
CLASS_HEAD = "score_"                       # v2.2: score_<k> the class logits, `score` = score_0
BASELINES = ("trivial", "linear", "tree")   # the v2.2 evaluator's stored baseline scores
AMENDED_BY = "R2 golden lane (BUILD-PLAN R-G2): the patched blueprint SKILL.md's three rules"
# the golden lane's package: CIC's rows, a multiclass label (the ports a package of that shape
# and kind carries: lanekit.pipeline.port_signatures)
GOLDEN_PACKAGE = ("row", "multiclass")

# R-G2's third rule: the artifacts the v2.2 blueprint drew that the v2.3 package does not hold
# (its domain.py defines none of them), each declared external with the stage that holds it and
# where it lives (code-contract §3). Keyed by the artifact's name in the recorded blueprint.
EXTERNAL = {
    "Recipe": ("/dataset-shaper", "the shaper's recipe (recipe.json), carried into the bundle "
                                  "by /vega-pack"),
    "FittedParameters": ("/dataset-shaper", "the shaper's fitted parameters (shaped/fitted.json)"),
    "ShapedPart": ("/dataset-shaper", "the shaped parts (shaped/*.parquet), digested in the "
                                      "shaped manifest"),
    "Campaign": ("/vega-train", "the campaign directory (campaign.json, runs/<id>/)"),
    "Run": ("/vega-train", "a run directory of the campaign (runs/<id>/)"),
    "TestSeal": ("/vega-eval", "the campaign's seal file (test_seal.json)"),
    "Verdict": ("/vega-eval", "the evaluation's records and its layer"),
    "ApiContract": ("/vega-pack", "the bundle's service contract (serve/contract.py, "
                                  "schema.json)"),
    "ServiceStatus": ("/vega-pack", "the bundle's service (serve/)"),
    "ProblemDetail": ("/vega-pack", "the bundle's service errors (serve/problems.py)"),
}


class MigrationError(Exception):
    """The recorded campaign cannot be migrated as it stands (the golden lane is blocked)."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------- read-only CIDS_ROOT
def readonly_snapshot(paths) -> dict:
    """{path: (size, mtime_ns)} of every file under `paths` (files or directories)."""
    out = {}
    for root in map(Path, paths):
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
        for f in files:
            st = f.stat()
            out[str(f)] = (st.st_size, st.st_mtime_ns)
    return out


def readonly_changes(before: Mapping, paths) -> list[str]:
    """The files the golden lane created, changed or removed under `paths` since `before`."""
    after = readonly_snapshot(paths)
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


# --------------------------------------------------------------------------- the blueprint (R-G2)
def amend_blueprint(base_run: Path, out_dir: Path) -> Path:
    """`out_dir/pipeline.json` (and `blueprint.md`): the base run with R-G2's rules applied —
    every port-owning module `OOP`, every port its role's v2.3 `signature[]` for the golden
    lane's package (CIC: rows, multiclass), every module's `depends_on` the arrows its code
    takes."""
    base = json.loads((base_run / "pipeline.json").read_text(encoding="utf-8"))
    signatures = port_signatures(*GOLDEN_PACKAGE)
    doc = json.loads(json.dumps(base))
    changed = []
    for module in doc.get("modules") or []:
        if module.get("ports_owned") and module.get("paradigm") != "OOP":
            changed.append(f"module {module['name']}: {module.get('paradigm')} -> OOP (owns "
                           f"{', '.join(module['ports_owned'])})")
            module["paradigm"] = "OOP"
    # the third rule: §4 draws the arrows the package's code takes, no other (C4 refuses an
    # arrow no import takes): the R7 base drew training/evaluation -> ingestion and
    # serving -> features, registry, which the v2.3 code reaches through ports, not imports
    taken = _emitter().module_arrows(doc, GOLDEN_PACKAGE[0])
    for module in doc.get("modules") or []:
        kept = [d for d in module.get("depends_on") or [] if d in taken.get(module.get("name"), ())]
        if kept != list(module.get("depends_on") or []):
            changed.append(f"module {module['name']}: depends_on {module.get('depends_on')} -> {kept} "
                           "(the arrows its code takes)")
            module["depends_on"] = kept
    for port in doc.get("ports") or []:
        role = port.get("role")
        if role not in signatures:
            raise MigrationError(f"port {port.get('name')!r} has role {role!r}, which no v2.3 "
                                 "port interface carries")
        filled = FILLED_BY.get(role)
        if port.get("filled_by") != filled:
            changed.append(f"port {port['name']} ({role}): filled_by {filled}")
            port.pop("filled_by", None)
            if filled:
                port["filled_by"] = filled
        if port.get("signature") != signatures[role]:
            changed.append(f"port {port['name']} ({role}): signature[] of {len(signatures[role])} "
                           "method(s)")
            port["signature"] = signatures[role]
    changed += _declare_external(doc.get("artifacts") or [])
    doc["amends"] = {"run": base_run.name, "by": AMENDED_BY,
                     "base_digest": "sha256:" + hashlib.sha256(json.dumps(
                         base, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                     "changes": changed, "previous": base.get("amends")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pipeline.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    md = base_run / "blueprint.md"
    text = md.read_text(encoding="utf-8") if md.is_file() else f"# Blueprint — {base_run.name}\n"
    (out_dir / "blueprint.md").write_text(
        text.rstrip("\n") + "\n\n## Amendment (R-G2)\n\n" + AMENDED_BY + ".\n\n"
        + "".join(f"- {c}\n" for c in changed), encoding="utf-8")
    return out_dir / "pipeline.json"


def _declare_external(artifacts) -> list[str]:
    """R-G2's third rule on the base's `artifacts[]`, in place; the changes it made."""
    changed = []
    for entry in artifacts:
        name = entry.get("name") if isinstance(entry, dict) else None
        if name in EXTERNAL and "external" not in entry:
            owner, where = EXTERNAL[name]
            entry["external"] = {"owner": owner, "where": where}
            changed.append(f"artifact {name}: external, held by {owner}")
    return changed


# --------------------------------------------------------------------------- the stored predictions
def reencode_predictions(src: Path, dst: Path, split_test, layout_module) -> dict:
    """The v2.2 stored predictions `src` written to `dst` in `vega/predictions-layout@1` through
    vega-eval's own writer (`layout_module.PredictionsLayout`); returns the migration record.
    `split_test` is the shaper's raw position of each row of the test part, in part order."""
    import numpy as np
    import pandas as pd
    from pyarrow import fs
    import pyarrow.parquet as pq
    meta = pq.read_schema(src).metadata or {}
    if b"vega.predictions_layout" in meta:
        raise MigrationError(f"{src} already names a layout; nothing to re-encode")
    old = pd.read_parquet(src, filesystem=fs.LocalFileSystem())
    heads = sorted((c for c in old.columns if c.startswith(CLASS_HEAD) and c[len(CLASS_HEAD):].isdigit()),
                   key=lambda c: int(c[len(CLASS_HEAD):]))
    positions = np.asarray(split_test, dtype=np.int64)
    ids = old["row_id"].to_numpy()
    if len(positions) != len(old) or not len(heads) or ids.min() < 0 or ids.max() >= len(positions) \
            or len(np.unique(ids)) != len(ids):
        raise MigrationError(f"{src}: {len(old)} stored row(s), row ids {ids.min()}..{ids.max()}, "
                             f"{len(positions)} raw position(s) recorded for the test part — not "
                             "the same part")
    head = old[heads].to_numpy()
    if "score" in old.columns and not np.array_equal(old["score"].to_numpy(), head[:, 0]):
        raise MigrationError(f"{src}: the stored `score` is not the first class logit "
                             "(v2.2 evaluate.py wrote `score_net[:, 0]`); this is not the v2.2 "
                             "multiclass layout this migration reads")
    baselines = tuple((b, old[f"score_{b}"].to_numpy()) for b in BASELINES if f"score_{b}" in old.columns)
    layout = layout_module.PredictionsLayout(
        row_ids=positions[ids], row_id_kind="raw_position", truth=old["y"].to_numpy(), head=head,
        task_effective="multiclass", baselines=baselines)
    layout.write(dst)
    return {"schema": "vega/golden-migration@1", "what": "stored test predictions re-encoded",
            "source": str(src), "source_digest": digest(src), "target_digest": digest(dst),
            "rows": int(len(old)), "layout": LAYOUT,
            "row_id": "raw position = fitted.json __split_raw__.test[v2.2 row_id]",
            "head": heads, "dropped": ["score"] if "score" in old.columns else [],
            "baselines": [b for b, _ in baselines]}


# --------------------------------------------------------------------------- the campaign
@dataclass(frozen=True)
class Migration:
    campaign: Path            # the migrated campaign directory (under the work dir)
    run: str
    package: str
    pipeline: Path
    record: dict              # what was migrated, with every digest


def _link_tree(src: Path, dst: Path, keep: Callable[[Path], bool]) -> None:
    """`dst` mirrors `src` with symlinks to every file `keep` accepts (read-only reuse)."""
    for item in sorted(src.iterdir()):
        if not keep(item):
            continue
        target = dst / item.name
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _link_tree(item, target, keep)
        else:
            target.symlink_to(item)


def migrate_campaign(cids: Path, recorded: Mapping[str, Any], work: Path, skills: Path,
                     pipeline: Path, python: str, log: Path) -> Migration:
    """The recorded campaign under `work/campaign`, packable by this release (module doc)."""
    src = cids / recorded["campaign"]
    run, shaped = recorded["run"], cids / recorded["shaped"]
    camp = work / "campaign"
    if camp.exists():
        shutil.rmtree(camp)
    camp.mkdir(parents=True)
    doc = json.loads((src / "campaign.json").read_text(encoding="utf-8"))
    package = (doc.get("input") or {}).get("package") or "pipeline"
    seal = json.loads((src / "test_seal.json").read_text(encoding="utf-8"))
    if seal.get("run") != run:
        raise MigrationError(f"the seal records run {seal.get('run')!r}, the lane {run!r}")
    for name in ("campaign.json", "test_seal.json"):
        shutil.copy2(src / name, camp / name)
    _link_tree(src, camp, lambda p: p.parent != src or p.name not in (
        "campaign.json", "test_seal.json", "experiments.json", "runs", "eval"))
    # the evaluation: every record linked, the predictions re-encoded
    ev_src, ev = src / "eval" / run, camp / "eval" / run
    ev.mkdir(parents=True)
    _link_tree(ev_src, ev, lambda p: p.name != "predictions.test.parquet")
    fitted = json.loads((shaped / "fitted.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(skills / "vega-eval" / "scripts"))
    try:
        import predictions_layout
    finally:
        sys.path.remove(str(skills / "vega-eval" / "scripts"))
    predictions = reencode_predictions(ev_src / "predictions.test.parquet",
                                       ev / "predictions.test.parquet",
                                       (fitted.get("__split_raw__") or {}).get("test") or [],
                                       predictions_layout)
    # the run: the recorded spec, weights and history, the package re-emitted by this release
    run_src, run_dir = src / "runs" / run, camp / "runs" / run
    run_dir.mkdir(parents=True)
    _link_tree(run_src, run_dir, lambda p: True)
    emit_out = work / "emit"
    if emit_out.exists():
        shutil.rmtree(emit_out)
    argv = [python, str(skills / "vega-forge" / "scripts" / "emit_backend.py"),
            "--spec", str(run_src / "netspec.resolved.json"), "--backend", "pytorch",
            "--pipeline", str(pipeline), "--out", str(emit_out), "--campaign-root", str(src),
            "--smoke", "--device", "cpu"]
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(argv) + "\n")
        fh.flush()
        proc = subprocess.run(argv, cwd=cids, stdout=fh, stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=900)
    if proc.returncode != 0:
        raise MigrationError(f"emit_backend.py exited {proc.returncode} on {run}'s resolved spec "
                             f"(log: {log})")
    shutil.copytree(emit_out / package, run_dir / package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(emit_out / "smoke.json", run_dir / "smoke.json")
    code = {f"{package}/{p.relative_to(run_dir / package).as_posix()}": digest(p)
            for p in sorted((run_dir / package).rglob("*")) if p.is_file()}
    ledger = json.loads((src / "experiments.json").read_text(encoding="utf-8"))
    row = next((r for r in ledger.get("runs") or [] if r.get("id") == run), None)
    if row is None:
        raise MigrationError(f"the ledger has no run {run}")
    record = {"schema": "vega/golden-migration@1", "campaign": str(src), "run": run,
              "seal": {"exposures": seal.get("exposures"), "digest": digest(src / "test_seal.json")},
              "spec": {"file": f"runs/{run}/netspec.resolved.json",
                       "digest": digest(run_src / "netspec.resolved.json"),
                       "recorded": row.get("spec_digest")},
              "weights": {"file": f"runs/{run}/checkpoints/best.pt",
                          "digest": digest(run_src / "checkpoints" / "best.pt"),
                          "recorded": row.get("checkpoint_digest")},
              "pipeline": {"file": str(pipeline), "digest": digest(pipeline)},
              "code": {"emitter": "vega-forge/scripts/emit_backend.py (v2.3)",
                       "from": row.get("code_digests"), "to": code},
              "predictions": predictions,
              "proof": "the pack's round trip: every test row, from raw, against the stored "
                       "predictions; the test part is not opened again"}
    row["migration"] = {k: record[k] for k in ("code", "pipeline")} | {"by": AMENDED_BY.split(":")[0]}
    row["code_digests"] = code
    (camp / "experiments.json").write_text(json.dumps(ledger, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    (camp / "migration.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    return Migration(campaign=camp, run=run, package=package, pipeline=pipeline, record=record)
