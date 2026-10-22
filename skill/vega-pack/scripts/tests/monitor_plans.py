#!/usr/bin/env python3
"""
monitor_plans.py — the monitor plan a test harness packs with.

    python3 monitor_plans.py --campaign <dir> --pipeline <pipeline.json> --out <plan.json>
                             [--run <id>]

Not a test file (the unit tier collects test_*.py). A blueprint whose Monitor port says
`"filled_by": "/vega-pack"` is refused by `pack.py` without `--monitor-plan`
(`monitor_pack.refuse_unfilled`, bundle-contract §7.5), and every blueprint the forge's emitter
admits says so. The plan is the packer's judgement in the chain; a harness has none to make, so
this writes the one plan that needs no knowledge of the data: for every `pipeline.json`
monitoring concern, the check `pack.py` would add on the answer itself
(`monitor_plan.output_check` — `output_psi` on the score field, else the decided classes' mix,
else the labels' rates), read off the output record of the evaluated run's resolved spec and its
threshold. Pack then adds no second output check of its own.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import output_record  # noqa: E402
from monitor_plan import output_check  # noqa: E402

__all__ = ["draft", "write"]


def _json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def draft(campaign, pipeline, run=None) -> dict:
    """The plan: one answer check per blueprint monitoring concern of `pipeline`."""
    camp = Path(campaign)
    run = run or _json(camp / "test_seal.json")["run"]
    spec = _json(camp / "runs" / run / "netspec.resolved.json")
    threshold_path = camp / "eval" / run / "threshold.json"
    threshold = _json(threshold_path) if threshold_path.is_file() else None
    output = output_record.output_record(spec, threshold, (spec.get("task") or {}).get("label"))
    check = output_check(output)
    if check is None:
        raise ValueError(f"the {output['reading']} reading admits no check on the answer; this "
                         "harness's plan has nothing to watch")
    concerns = [m.get("source") or m.get("name") for m in _json(pipeline).get("monitoring") or []
                if isinstance(m, dict)] or ["[harness:answer]"]
    checks = [dict(check, id=f"answer-{n}", concern=str(concern),
                   title=f"{check['title']} ({concern})") for n, concern in enumerate(concerns, 1)]
    return {"note": f"the test harness's plan for run {run}: the answer watched for every "
                    "blueprint concern", "checks": checks}


def write(campaign, pipeline, out, run=None) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(draft(campaign, pipeline, run), indent=2) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None)
    a = ap.parse_args(argv)
    try:
        path = write(a.campaign, a.pipeline, a.out, a.run)
    except Exception as exc:                                # noqa: BLE001 (a harness's tool)
        print(f"ERROR: no monitor plan: {type(exc).__name__}: {exc}")
        return 2
    print(f"OK: monitor plan written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
