"""
Stamps: what `--from <stage>` trusts (D6 §6).

Every stage that passes writes `lanes/<id>/stamps/<stage>.json` (`vega/lane-stamp@1`) with the
digests of its INPUTS — the lane's registry entry with the shared defaults, the skill code the
stage runs (`contracts.STAGE_CODE`), the domain-forge pins — the digests of the OUTPUTS later
stages consume (`contracts.STAGE_OUTPUTS`), and the digest of the previous stage's stamp
(`upstream`), so the stamps of one lane form a chain.

`resume_problems` re-checks every stamp before the stage `--from` names, and says why it cannot
resume: a stamp missing or not a pass, an input changed, a stamped output changed, or the chain
broken (an earlier stage re-ran on its own after a later one).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from . import contracts


@dataclass(frozen=True)
class StampInputs:
    lane: str                                   # digest of the lane entry and the shared defaults
    code: Mapping[str, str | None]              # skill paths the stage runs -> digest
    external: Mapping[str, str]                 # domain-forge pins

    def to_json(self) -> dict:
        return {"lane": self.lane, "code": dict(self.code), "external": dict(self.external)}


def stamp_path(lane_dir: Path, stage: str) -> Path:
    return lane_dir / "stamps" / f"{stage}.json"


def write_stamp(lane_dir: Path, stage: str, inputs: StampInputs, outputs: Mapping[str, str],
                upstream: str | None) -> str:
    """Record a passed stage; returns the stamp's own digest (the next stamp's `upstream`)."""
    doc = {"schema": contracts.LANE_STAMP, "stage": stage, "status": "pass",
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inputs": inputs.to_json(),
           "outputs": dict(outputs), "upstream": upstream}
    path = stamp_path(lane_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contracts.sha256_file(path)


def clear_stamps(lane_dir: Path, stages: Sequence[str]) -> None:
    for stage in stages:
        stamp_path(lane_dir, stage).unlink(missing_ok=True)


def resume_problems(lane_dir: Path, before: Sequence[str],
                    current: Callable[[str], StampInputs],
                    outputs_now: Callable[[Mapping[str, str]], Mapping[str, str | None]]) -> list[str]:
    """Why `--from` cannot resume after the stages `before`: a stamp missing, not passed, its
    inputs changed (lane entry, code, domain-forge), a stamped output changed, or the chain of
    stamps broken (an earlier stage re-ran after a later one)."""
    problems, upstream = [], None
    for stage in before:
        path = stamp_path(lane_dir, stage)
        if not path.is_file():
            problems.append(f"{stage}: no stamp — it never passed in this matrix dir")
            break
        doc = json.loads(path.read_text(encoding="utf-8"))
        problems += _stamp_changes(stage, doc, current(stage), outputs_now(doc.get("outputs") or {}))
        if doc.get("upstream") != upstream:
            problems.append(f"{stage}: its stamp does not follow the previous stage's — an "
                            "earlier stage re-ran after it")
        upstream = contracts.sha256_file(path)
    return problems


def _stamp_changes(stage: str, doc: Mapping, now: StampInputs, outputs: Mapping[str, str | None]) -> list[str]:
    was = doc.get("inputs") or {}
    problems = []
    if doc.get("status") != "pass":
        problems.append(f"{stage}: its stamp is not a pass")
    if was.get("lane") != now.lane:
        problems.append(f"{stage}: the lane's registry entry changed since it passed")
    for label, old, new in (("code", was.get("code") or {}, now.code),
                            ("domain-forge", was.get("external") or {}, now.external)):
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                problems.append(f"{stage}: {label} {key} changed since it passed "
                                f"({_short(old.get(key))} -> {_short(new.get(key))})")
    for rel, digest in (doc.get("outputs") or {}).items():
        if outputs.get(rel) != digest:
            problems.append(f"{stage}: its output {rel} changed since it passed "
                            f"({_short(digest)} -> {_short(outputs.get(rel))})")
    return problems


def _short(digest: str | None) -> str:
    return "absent" if digest is None else digest.split(":")[-1][:12]
