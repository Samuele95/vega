#!/usr/bin/env python3
"""
_acceptance_lanes.py — acceptance.sh's lane steps (D5 §12.1 steps 2–5; BUILD-PLAN H09): the stage
run on a copy of a matrix lane as a user runs it, and every claim the design makes about it held.

    python3 _acceptance_lanes.py <matrix-dir> --lane L01|L02 --tmp DIR

**L01** (orders: binary, `mlp-residual`, packed `--serve`, no report) — lane A of D5 §12.1:
1. without a report: extract → draft → author → resolve → build → check → `smoke_show --serve
   --strict` → `apply_show_layer` (the authoring of `showcase-layer.authoring.json`, the run's WARN
   lines) → `validate_model.py` (invariants 13–16, through the flake wrapper) → `smoke_show OUT
   --strict`; `report.status` brief-links, `WARN: no report` in the markers, no class `alert`
   (unattended, with no `show:` stance: `plain`, the neutral only by a structural basis — the
   orders lane's `0` by `structural:negative`), the motif `points`, the bundle's MANIFEST.json bytes unchanged, `predict.py --check`
   from a clean shell and the bundle's own `tests/test_roundtrip.py` still passing;
2. with the report fixture (`fixtures/report/`, a stub compiled report and its role map) copied
   into `docs/`: `--check` exits 3 (a report appears: stale); the stage run again links it
   (`report.status` linked, the roles bound the fixture's, none missing) and the served walk
   fetches every link at 200 with its pinned digest;
3. staleness: one byte appended to the report's PDF → `--check` 3; `story.json` altered → `--check`
   2, and the bundle's own service answers `/v1/showcase` with `verified: false` while `/health` is
   ready (the model never waits on a showcase).

**L02** (land cover: multiclass, `mlp-plain`, the stub report) — the second lane, differing in
task, family, lexicon and report mode: the stage linked, the served walk, the layer, invariants
13–16 and the tab's smoke, the MANIFEST unchanged.

Exit 0 when every claim held; 1 at the first that did not (`FAIL:` naming it).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent), str(SKILLS / "vega" / "scripts" / "tests"),
                str(SKILLS / "vega" / "scripts" / "tests" / "negatives")]

import negkit  # noqa: E402
from _negatives_kit import PY, StageError, Staged, Work, read_json, run, sha, show, write_json  # noqa: E402

DF = SKILLS / "domain-forge" / "scripts" / "validate_model.py"
KNOWN_FLAKE = SKILLS / "vega" / "scripts" / "tests" / "known_flake.py"
AUTHORING = HERE / "showcase-layer.authoring.json"
REPORT_FIXTURE = HERE / "fixtures" / "report"
CLEAN_ENV = {"PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
APP_CHILD = r"""
import json, sys
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from serve.app import create_app
with TestClient(create_app('.')) as c:
    print('RESULT ' + json.dumps({'showcase': c.get('/v1/showcase').json(), 'health': c.get('/health').json()}))
"""


class Failed(Exception):
    pass


def ok(claim: str) -> None:
    print(f"OK: {claim}")


def need(cond: bool, claim: str, detail: str = "") -> None:
    if not cond:
        raise Failed(f"{claim}{': ' + detail[-2000:] if detail else ''}")
    ok(claim)


def step(w: Work, what: str, rc_out: tuple, code: int = 0) -> str:
    rc, out = rc_out
    need(rc == code, f"{what} exits {code}", f"exit {rc}\n{out}")
    return out


# =============================================================================== the steps

def walk(w: Work) -> dict:
    step(w, "smoke_show.py --serve --strict (the served walk)", w.walk())
    doc = read_json(w.showcase / ".smoke" / "smoke.json")
    need(doc["status"] == "pass" and doc["verified"] and doc["page_errors"] == 0 and not doc["placeholders_unresolved"],
         "the walk passed, the slot verified, no page error, no unresolved placeholder")
    need(bool(doc["links"]) and all(l["ok"] and l["status"] == 200 for l in doc["links"]),
         f"every link of the story answered 200 at its pinned digest ({len(doc['links'])})")
    return doc


def layer(w: Work) -> None:
    """SKILL step 8 in the model's place, then the layer, its invariants and its tab."""
    recorded = read_json(w.showcase / ".work" / "markers.json")
    warns = [line.split(": ", 1)[1] if line.split(": ", 1)[0].endswith(".py") else line
             for line in recorded if isinstance(line, str) and "WARN:" in line]
    authoring = read_json(AUTHORING)
    write_json(w.layer_data, {"from_showcase": str(w.showcase), "readings": authoring["readings"],
                              "markers": [*authoring["markers"], *warns]})
    step(w, "apply_show_layer.py", w.layer())
    out = step(w, "validate_model.py on the layered page", run([PY, KNOWN_FLAKE, "run", "--stage", "show", "--",
                                                                 PY, DF, w.layered], timeout=900))
    for n in (13, 14, 15, 16):
        need(f"PASS  invariant {n}:" in out, f"validate_model invariant {n} passes", out)
    step(w, "smoke_show.py OUT --strict (the Showcase tab's nine checks)",
         show("smoke_show.py", w.layered, "--strict", cwd=w.work))


def bundle_still_runs(w: Work, before: str) -> None:
    need(sha(w.manifest.read_bytes()) == before, "the bundle's MANIFEST.json is byte for byte as packed")
    step(w, "predict.py --check from a clean shell", run([PY, "predict.py", "--check"], cwd=w.bundle, env=CLEAN_ENV))
    step(w, "the bundle's tests/test_roundtrip.py", run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                                         "tests/test_roundtrip.py"], cwd=w.bundle, env=CLEAN_ENV))


def served_status(w: Work) -> dict:
    rc, out = run([PY, "-c", APP_CHILD], cwd=w.bundle, env=CLEAN_ENV)
    line = next((ln for ln in reversed(out.splitlines()) if ln.startswith("RESULT ")), None)
    need(line is not None, "the bundle's app answers", out)
    return json.loads(line[len("RESULT "):])


# =============================================================================== the lanes

def lane_a(w: Work, before: str) -> None:
    print("\n--- L01 without a report")
    doc = read_json(w.showcase / "SHOWCASE.json")
    need(doc["report"]["status"] == "brief-links", "SHOWCASE.json report.status is brief-links")
    need(any(m.startswith("WARN: no report") for m in doc["markers"]), "WARN: no report is in the markers")
    story = read_json(w.showcase / "story.json")
    roles = story.get("class_roles") or {}
    plain = {c for c, r in roles.get("roles", {}).items() if r == "plain"}
    neutral = {c: roles["source"][c] for c, r in roles.get("roles", {}).items() if r == "neutral"}
    need(bool(plain) and len(plain) + len(neutral) == len(roles.get("roles", {}))
         and all(s.startswith("structural:") for s in neutral.values()),
         f"no class is alert unattended: plain, the neutral only by structure ({roles.get('source')})")
    need(read_json(w.showcase / "page.json")["motif"]["id"] == "points", "the motif is points (a row lane's base)")
    walk(w)
    layer(w)
    bundle_still_runs(w, before)

    print("\n--- L01 with the report fixture")
    shutil.copytree(REPORT_FIXTURE, w.docs)
    step(w, "--check once a compiled report appears (stale)", w.check(), code=3)
    w.stage()
    doc = read_json(w.showcase / "SHOWCASE.json")
    fixture = set(read_json(REPORT_FIXTURE / "label-roles.json")["roles"])
    bound, missing = set(doc["report"]["roles"]["bound"]), doc["report"]["roles"]["missing"]
    need(doc["report"]["status"] == "linked", "report.status is linked")
    need(bound <= fixture and bool(bound) and not missing, f"the roles bound are the fixture's ({len(bound)}), none missing")
    walk(w)

    print("\n--- L01 staleness")
    pdf = w.docs / "main.pdf"
    kept = pdf.read_bytes()
    pdf.write_bytes(kept + b" ")
    step(w, "--check after the report's PDF changed (stale)", w.check(), code=3)
    pdf.write_bytes(kept)
    step(w, "--check with the PDF restored", w.check())
    story = w.showcase / "story.json"
    story.write_bytes(story.read_bytes().replace(b'"headline": "', b'"headline": "Altered: ', 1))
    step(w, "--check after story.json was altered (broken)", w.check(), code=2)
    got = served_status(w)
    need(got["showcase"].get("verified") is False, f"/v1/showcase says verified: false ({got['showcase'].get('reason')})")
    need(got["health"].get("ready") is True, "/health stays ready")
    need(sha(w.manifest.read_bytes()) == before, "the bundle's MANIFEST.json is still as packed")


def lane_b(w: Work, before: str) -> None:
    print("\n--- L02 with its stub report")
    doc = read_json(w.showcase / "SHOWCASE.json")
    need(doc["report"]["status"] == "linked" and not doc["report"]["roles"]["missing"],
         "report.status is linked, no role missing")
    walk(w)
    layer(w)
    need(sha(w.manifest.read_bytes()) == before, "the bundle's MANIFEST.json is byte for byte as packed")


def stage_copy(matrix: Path, lane_id: str, tmp: Path) -> tuple[Work, str]:
    """The lane's work copied and staged, and the digest of its MANIFEST.json as packed (the
    L01 acceptance lane has no fixture.json: its work dir is read as it is)."""
    lane = negkit.lane(matrix, lane_id)
    if lane is None:
        root = matrix / "lanes" / lane_id
        if not (root / "work").is_dir():
            raise Failed(f"lane {lane_id} has no work dir in {matrix} (run matrix.sh lanes --lanes {lane_id} --to report)")
        lane = negkit.Lane(lane_id, root, {})
    packed = sha(Work(lane_id, lane.root).manifest.read_bytes())
    if lane_id != "L01":
        return Work(lane_id, Staged.of(lane, tmp).root), packed
    # lane A starts without a report: the lane's stub report stays behind, and the stage briefs a
    # report-less lane leaves (lanekit.stage_briefs, the stages' stand-ins) are what it links
    from lanekit import stage_briefs
    staged = Staged.of(lane, tmp, docs=False, prepare=lambda w: stage_briefs.write(w.work, w.bundle, lane_id))
    return Work(lane_id, staged.root), packed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("matrix")
    ap.add_argument("--lane", required=True, choices=("L01", "L02"))
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args(argv)
    tmp = Path(a.tmp)
    shutil.rmtree(tmp / a.lane, ignore_errors=True)
    shutil.rmtree(tmp / f"{a.lane}.snap", ignore_errors=True)
    try:
        w, before = stage_copy(Path(a.matrix).resolve(), a.lane, tmp)
        need(sha(w.manifest.read_bytes()) == before, "the build left the bundle's MANIFEST.json as packed")
        (lane_a if a.lane == "L01" else lane_b)(w, before)
    except (Failed, StageError) as exc:
        print(f"FAIL: {a.lane}: {exc}")
        return 1
    print(f"OK: {a.lane}: every claim held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
