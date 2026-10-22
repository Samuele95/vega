"""
`--changed <git range>`: the lanes a pull request's changed paths affect (D6 §3.4).

- serving templates and vega-show assets route every lane;
- a block file (`blocks*.py`, `builder.py`) routes the lanes whose blocks' `# --8<-- <type>`
  segments the change touches (every lane when the file has no markers);
- the shaper's graph code routes the graph lanes, any other shaper change every lane;
- a generator module routes the lanes that use it; `lanes.json` the lanes whose entry changed;
- documentation (`*.md`, patches, references, verifiers) routes none;
- anything else a lane runs routes every lane.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

from . import contracts
from .registry import Lane, load_document


def affected_lanes(lanes: Mapping[str, Lane], changed: Iterable[str], repo: Path,
                   rev_range: str | None = None) -> dict[str, list[str]]:
    """{lane id: [the changed paths that route it]} for a PR's changed paths (D6 §3.4)."""
    hits: dict[str, list[str]] = {}
    for path in changed:
        for lane_id in _route(path, lanes, repo, rev_range):
            hits.setdefault(lane_id, []).append(path)
    return {k: hits[k] for k in lanes if k in hits}


def _route(path: str, lanes: Mapping[str, Lane], repo: Path, rev_range: str | None) -> list[str]:
    everyone = list(lanes)
    if path.endswith(".md") or path.startswith(("patches/", "vega/references/", "vega/verifiers/")):
        return []
    if path.startswith(contracts.SERVING_TEMPLATES + "/") or path.startswith("vega-show/"):
        return everyone
    name = Path(path).name
    if re.match(r"blocks(_\w+)?\.py$|builder\.py$", name) and "vega-forge/" in path:
        types = _changed_segments(repo, path, rev_range)
        if types is None:
            return everyone
        return [i for i, lane in lanes.items() if types & set(lane.flat_block_types())]
    if path.startswith("dataset-shaper/scripts/") and name in ("shape.py", "verify_shape.py"):
        return _shaper_route(repo, path, rev_range, lanes)
    if path.startswith(contracts.LANEKIT_DIR + "/datasets/"):
        return [i for i, lane in lanes.items() if lane.raw.get("generator") == Path(path).stem]
    if path == contracts.TESTS_DIR + "/lanes.json":
        return _changed_entries(repo, path, rev_range, lanes)
    return everyone


SEGMENT = re.compile(r"#\s*--8<--\s*(\w+)")


def _changed_lines(repo: Path, path: str, rev_range: str | None) -> list[int] | None:
    """The changed line numbers of the new side of `path` in `rev_range` (working tree when
    None); None when git cannot say."""
    cmd = ["git", "-C", str(repo), "diff", "-U0"] + ([rev_range] if rev_range else []) + ["--", path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    lines: list[int] = []
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", out, re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        lines.extend(range(start, start + max(count, 1)))
    return lines


def _changed_segments(repo: Path, path: str, rev_range: str | None) -> set[str] | None:
    """The block types whose `# --8<-- <type>` segment a change touches; None when the file
    has no markers or git cannot say (then every lane is routed)."""
    lines = _changed_lines(repo, path, rev_range)
    file = repo / path
    if lines is None or not file.is_file():
        return None
    marks = [(i, m.group(1)) for i, text in enumerate(file.read_text(encoding="utf-8").splitlines(), 1)
             if (m := SEGMENT.search(text))]
    if not marks:
        return None
    types = set()
    for line in lines:
        owner = [t for start, t in marks if start <= line]
        if owner:
            types.add(owner[-1])
    return types


def _shaper_route(repo: Path, path: str, rev_range: str | None, lanes: Mapping[str, Lane]) -> list[str]:
    """Graph code in the shaper routes the graph lanes; any other change routes every lane."""
    lines = _changed_lines(repo, path, rev_range)
    file = repo / path
    if not lines or not file.is_file():
        return list(lanes)
    defs = [(i, m.group(1)) for i, text in enumerate(file.read_text(encoding="utf-8").splitlines(), 1)
            if (m := re.match(r"(?:def|class)\s+(\w+)", text))]
    names = {next((n for s, n in reversed(defs) if s <= line), "") for line in lines}
    if all(re.search(r"graph|edge", n) for n in names):
        return [i for i, lane in lanes.items() if lane.raw.get("shape") == "graph"]
    return list(lanes)


def _changed_entries(repo: Path, path: str, rev_range: str | None, lanes: Mapping[str, Lane]) -> list[str]:
    """The lanes whose registry entry (or the shared defaults) changed across `rev_range`."""
    base = (rev_range or "HEAD").split("..")[0] or "HEAD"
    try:
        old = json.loads(subprocess.run(["git", "-C", str(repo), "show", f"{base}:{path}"],
                                        capture_output=True, text=True, check=True).stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return list(lanes)
    new = load_document(repo / path)
    if old.get("defaults") != new.get("defaults"):
        return list(lanes)
    before = {e["id"]: e for e in old.get("lanes", [])}
    return [i for i, lane in lanes.items() if before.get(i) != lane.raw]
