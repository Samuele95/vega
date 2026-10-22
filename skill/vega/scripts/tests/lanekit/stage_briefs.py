"""
The stage briefs of a lane without a report (vega/references/brief-contract.md), written by the
lane in its stages' place — as lanekit.authoring writes what a stage's model would author — so
that the show stage's brief-link mode has something to link (build_showcase gate PF6: a
brief-link showcase links one brief at least; D6 §6.7 `report: none`, R-E11).

    write(work, bundle, lane_id) -> [Path]      the briefs written (shape, blueprint, net, train, eval)

Each brief lies where vega-show's facts look for it (facts_report.brief_paths: the run directory
the bundle's records name, `<run-dir>/docs/<stage>-brief.md`), carries the contract's frontmatter
(its `records` the run directory's own files, with their digests) and its seven sections. The
prose states no number: the records hold them, and the brief points there. The pack brief is not
written: it would lie inside the bundle after `pack`, and the bundle is pinned from then on.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

# stage: (skill, layer, the bundle record and key naming its run directory, the stage's records
# in that directory (each one there), the model page it read and the one it wrote, by suffix,
# the lane's log of the step that ran it)
STAGES: Mapping[str, tuple] = {
    "shape": ("dataset-shaper", "shape", ("records/shape.json", ("manifest", "outputs", "dir")),
              ("manifest.json", "verification.json", "recipe.json"),
              (".domain.analysis.html", ".domain.analysis.shaped.html"), "fixture"),
    "blueprint": ("blueprint", None, ("records/architecture.json", ("netspec", "input", "pipeline")),
                  ("pipeline.json", "blueprint.md"),
                  (".domain.analysis.shaped.html", "pipeline.json"), "fixture"),
    "net": ("vega-forge", "architecture", ("records/architecture.json", ("netspec", "outputs", "run_dir")),
            ("architecture-layer.json", "smoke.json", "netspec.json"),
            (".domain.analysis.shaped.html", ".domain.analysis.shaped.vega.html"), "forge"),
    "train": ("vega-train", "training", ("records/training.json", ("campaign", "run_dir")),
              ("training-layer.json", "experiments.json", "ladder.json"),
              (".shaped.vega.html", ".shaped.vega.train.html"), "train"),
    "eval": ("vega-eval", "evaluation", ("records/training.json", ("campaign", "run_dir")),
             ("evaluation-layer.json", "test_seal.json"),
             (".shaped.vega.train.html", ".shaped.vega.train.eval.html"), "eval"),
}
NEXT = {"shape": "/blueprint", "blueprint": "/vega-forge", "net": "/vega-train", "train": "/vega-eval",
        "eval": "/vega-pack"}


def _peek(doc: Any, keys: tuple) -> Any:
    for k in keys:
        doc = doc.get(k) if isinstance(doc, Mapping) else None
    return doc


def run_dir(work: Path, bundle: Path, stage: str) -> Path | None:
    """The stage's run directory, as the bundle's records name it (relative: to the work dir)."""
    record, keys = STAGES[stage][2]
    try:
        value = _peek(json.loads((bundle / record).read_text(encoding="utf-8")), keys)
    except (OSError, ValueError):
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    path = path if path.is_absolute() else work / path
    return path.parent if stage == "blueprint" else path


def _relative(path: Path, work: Path) -> str:
    """A path under the work dir (the lane's project root) relative to it, any other absolute."""
    try:
        return PurePosixPath(path.resolve().relative_to(work.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _page(work: Path, suffix: str) -> str:
    found = sorted(p.name for p in work.glob(f"*{suffix}") if p.name.endswith(suffix))
    return found[0] if found else suffix.lstrip(".")


def brief(work: Path, directory: Path, stage: str, lane_id: str) -> str | None:
    """The brief's text, or None when the run directory holds none of the stage's records."""
    skill, layer, _, names, (given, made), log = STAGES[stage]
    records = [directory / n for n in names if (directory / n).is_file()]
    if not records:
        return None
    rel = lambda p: _relative(Path(p), work)  # noqa: E731
    front = [f"stage: {stage}", f"skill: {skill}", f"lane: lanekit:{lane_id}", f"input: {_page(work, given)}",
             f"output: {_page(work, made)}", f"run_dir: {rel(directory)}", f"layer: {layer or 'null'}",
             f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}", "records:"]
    front += [line for p in records for line in (f"  - path: {rel(p)}", f"    sha256: {_sha256(p)}")]
    front.append("supersedes: null")
    body = {
        "What this stage was for": f"The {stage} stage of lane {lane_id}, run by the verification matrix "
                                   f"with {skill}'s scripts: it read {_page(work, given)} and handed on "
                                   f"{_page(work, made)}.",
        "What was decided": "The lane's scripted authoring answered every choice of this stage the way the "
                            "lane registry fixes it, unattended; the layer records each decision with its source.",
        "What ran": f"```\n# lanekit drove this stage; its commands and their marker lines: log/{log}.log\n```",
        "What came out": "The records listed in the frontmatter hold every number of this stage; this brief "
                         "restates none of them.",
        "What the reader should check": "The lane's stage checks ran on these records (the lane's observed.json "
                                        "names their verdicts); re-run the lane to repeat them.",
        "Open": "Nothing open.",
        "Next": f"{NEXT[stage]}, as the lane drives it.",
    }
    return "---\n" + "\n".join(front) + "\n---\n\n" + "\n\n".join(f"## {h}\n\n{t}" for h, t in body.items()) + "\n"


def write(work: Path, bundle: Path, lane_id: str) -> list[Path]:
    """Every stage brief this lane's records place outside the bundle, written; their paths."""
    out = []
    for stage in STAGES:
        directory = run_dir(work, bundle, stage)
        text = brief(work, directory, stage, lane_id) if directory and directory.is_dir() else None
        if text is None:
            continue
        path = directory / "docs" / f"{stage}-brief.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        out.append(path)
    return out
