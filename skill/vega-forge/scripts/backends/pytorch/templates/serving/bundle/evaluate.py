#!/usr/bin/env python3
"""
evaluate.py — the metrics this spec names, on a part you name.

    python3 evaluate.py --part valid --shaped path/to/shaped
    python3 evaluate.py --part test  --shaped path/to/shaped --seal test_seal.json --token …

This is **not** /vega-eval. It computes point metrics and prints them. There are no bootstrap
intervals, no baselines, no paired differences, no slices and no verdict — which is to say it
produces none of what the model card's Evaluation section reports, and a number from here
should never be put beside one from there as though they were the same measurement.

The test part is sealed. `load_part` refuses it, here as in the chain, and the one door —
`load_sealed_part` — wants the campaign's `test_seal.json` and its token. That refusal
travelled into this bundle with the code that makes it: shipping the evaluator does not ship
a way around the seal, and every look it does allow is a look someone has to account for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from {package}.domain import SealedPartError                         # noqa: E402
from {package}.{mod_evaluation}.evaluate import Evaluator                 # noqa: E402
from {package}.{mod_ingestion}.data import ShapedParts                    # noqa: E402
from {package}.{mod_model}.network import Network                         # noqa: E402
from {package}.{mod_store}.store import ModelRegistry                     # noqa: E402

sys.path.insert(0, str(ROOT))
from train import spec_over                                         # noqa: E402


def _arguments(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="valid", choices=("train", "valid", "test"))
    ap.add_argument("--shaped", required=True, help="the directory holding the shaped parts")
    ap.add_argument("--weights", default=None, help="default: the bundle's own weights.pt")
    ap.add_argument("--seal", default=None, help="the campaign's test_seal.json (--part test)")
    ap.add_argument("--token", default=None, help="the seal's token (--part test)")
    return ap.parse_args(argv)


def _score_test(scoring, spec, a, weights):
    """The sealed part's metrics, or None after printing why the seal refused."""
    if not (a.seal and a.token):
        print("ERROR: the test part is sealed. It is opened through test_seal.json and "
              "the token that seal records, and every look is counted against it — pass "
              "--seal and --token, or score valid instead. This bundle carries the "
              "evaluator, not a way past the seal.")
        return None
    try:
        metrics = scoring.score_sealed_test(spec, a.token, checkpoint=str(weights),
                                            seal_path=a.seal)
    except SealedPartError as refusal:
        # the seal refusing is the system working, not the program crashing, so it reads
        # as a refusal rather than a traceback
        print(f"ERROR: {refusal}")
        return None
    print("WARN: that was a look at the held-out part. It counts.")
    return metrics


def _print_metrics(part, metrics, weights) -> None:
    values = metrics.as_dict()
    unavailable = values.pop("_unavailable", None)
    print(f"OK: {part} — {len(values)} metric(s) from {weights.name}")
    for name, value in sorted(values.items()):
        print(f"  {name:<20} {value:.6g}" if isinstance(value, (int, float))
              else f"  {name:<20} {value}")
    if unavailable:
        for name, why in unavailable.items():
            print(f"  {name:<20} unavailable — {why}")
    print("NOTE: point values with no interval. The card's Evaluation section reports "
          "bootstrap intervals, baselines and a verdict; these are not those numbers.")


def main(argv=None):
    a = _arguments(argv)
    spec = spec_over(a.shaped)
    weights = Path(a.weights).resolve() if a.weights else (ROOT / "weights.pt")
    if not weights.is_file():
        print(f"ERROR: no checkpoint at {weights}")
        return 2

    scoring = Evaluator.restoring(ShapedParts(), ModelRegistry(), Network.build)
    if a.part == "test":
        metrics = _score_test(scoring, spec, a, weights)
        if metrics is None:
            return 2
    else:
        metrics = scoring.score_part(spec, a.part, checkpoint=str(weights))
    _print_metrics(a.part, metrics, weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
