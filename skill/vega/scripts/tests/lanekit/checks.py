"""
The lane stages' helpers that run in their own process (`python -m lanekit.checks <name> …`):
the calls into the other lanekit modules — the fixture (T02), the authoring writers and the
report stub (T03) — and the artifact assertions (lanekit.assertions, D6 §6.1–§6.8).

    python -m lanekit.checks <name> --lanes-file F --lane ID --lane-dir D --skills S [options]

Every assertion prints `OK: …` or `ERROR: …`; the helper exits 1 when any ERROR was printed, so
the stage fails with the reason in its log. The runner process never imports pandas or torch.

The calls into other packages are contracts (contracts.FIXTURE_API, AUTHORING_API,
REPORT_STUB_API) that those packages implement:

    fixture.make_lane(lane, work, skills, composer=spec.Composer.for_lanes_file(F)) -> LaneFixture
                                                                        (a dataclass; written as JSON)
    authoring.<forge|train|pack>_authoring(directory, lane) -> dict     (the layer authoring JSON)
    authoring.show_authoring(facts, lane, draft) -> dict                (the story plan, slots filled)
    report_stub.write(docs_dir, roles) -> dict
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Callable, Mapping

from . import contracts
from .assertions import (Checker, Ctx, _import, _json, check_eval, check_fixture, check_forge,
                         check_pack, check_train)
from .show_assertions import check_show
from .registry import load_lanes

# --------------------------------------------------------------------------- calls into lanekit
def do_fixture(ctx: Ctx, check: Checker) -> None:
    module, name = contracts.FIXTURE_API
    # the budget of the lanes file the lane came from, not the tree's own lanes.json
    composer = _import("lanekit.spec").Composer.for_lanes_file(Path(ctx.args.lanes_file))
    fx = getattr(_import(module), name)(ctx.lane, ctx.work, ctx.skills, composer=composer)
    doc = {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(fx).items()}
    Path(ctx.args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    check.ok(f"fixture written for {ctx.lane.id}: {Path(doc['model_html']).name}")


def do_author(ctx: Ctx, check: Checker) -> None:
    module, _ = contracts.AUTHORING_API
    kind, target = ctx.args.kind, Path(ctx.args.target)
    fn = getattr(_import(module), f"{kind}_authoring")
    if kind == "show":
        doc = fn(_json(ctx.args.facts), ctx.lane, _json(ctx.args.draft))
    else:
        doc = fn(target.parent, ctx.lane)
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    check.ok(f"{kind} authoring written to {target.name}")


def do_report(ctx: Ctx, check: Checker) -> None:
    module, name = contracts.REPORT_STUB_API
    docs = Path(ctx.args.docs)
    getattr(_import(module), name)(docs, roles=contracts.REPORT_ROLES(ctx.skills))
    for rel in ("main.aux", "main.pdf", "label-roles.json"):
        check.that((docs / rel).is_file(), f"docs/{rel} written", f"docs/{rel} missing")
    pdf = docs / "main.pdf"
    if pdf.is_file():
        check.that(pdf.read_bytes()[:5] == b"%PDF-", "main.pdf starts %PDF-", "main.pdf is not a PDF")


# --------------------------------------------------------------------------- entry point
HELPERS: Mapping[str, Callable[[Ctx, Checker], None]] = {
    "fixture": do_fixture, "author": do_author, "report": do_report,
    "check-fixture": check_fixture, "check-forge": check_forge, "check-train": check_train,
    "check-eval": check_eval, "check-pack": check_pack, "check-show": check_show}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lanekit.checks", description=__doc__.split("\n\n")[0])
    ap.add_argument("name", choices=sorted(HELPERS))
    for flag in ("--lanes-file", "--lane", "--lane-dir", "--skills"):
        ap.add_argument(flag, required=True)
    for flag in ("--out", "--kind", "--target", "--facts", "--draft", "--docs", "--pin"):
        ap.add_argument(flag)
    args = ap.parse_args(argv)
    lane = load_lanes(Path(args.lanes_file))[args.lane]
    ctx = Ctx(lane, Path(args.lane_dir), Path(args.skills), args)
    check = Checker()
    HELPERS[args.name](ctx, check)
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
