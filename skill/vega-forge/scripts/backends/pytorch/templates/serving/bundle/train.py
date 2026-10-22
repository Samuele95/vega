#!/usr/bin/env python3
"""
train.py — one run of this bundle's spec on shaped parts you name.

    python3 train.py --shaped path/to/shaped [--out runs/mine] [--epochs 50] [--budget 20]

This is **not** /vega-train. It fits `netspec.json` once and writes a checkpoint. There is no
ladder, no ablation, no hyper-parameter study, no diagnostics catalog and no ledger, so
nothing it produces carries the chain's provenance: the weights that ship with this bundle
were selected by a campaign, and a checkpoint from here is a different network that no model
card describes.

The shaped parts do not travel with the bundle, so `--shaped` is required. It is a directory
holding the parts the recipe produced, under the names `netspec.json` records: `recipe.json`
is what makes new ones, through the shaper that wrote it. `ingestion/data.py` refuses the
test part here exactly as it does in the chain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from {package}.domain import NetSpec, RunContext                    # noqa: E402
from {package}.{mod_ingestion}.data import ShapedParts                    # noqa: E402
from {package}.{mod_model}.network import Network                         # noqa: E402
from {package}.{mod_store}.store import ModelRegistry                     # noqa: E402
from {package}.{mod_training}.loop import Trainer                         # noqa: E402


def spec_over(shaped, epochs=None):
    """The bundle's spec with its part paths pointed at a directory you hold.

    The paths in `netspec.json` are the build machine's. Rewriting them here — rather than
    asking `data.py` to guess — keeps the DataSource adapter reading the spec and only the
    spec, which is the arrangement the round trip and the seal both depend on.
    """
    d = json.loads((ROOT / "netspec.json").read_text(encoding="utf-8"))
    parts = ((d.get("data_contract") or {}).get("parts") or {})
    shaped = Path(shaped).resolve()
    missing = []
    inp = d.get("input") or {}
    if inp.get("manifest"):
        # the shaped manifest names every part by its digest; the DataSource reads it first
        manifest = shaped / Path(inp["manifest"]).name
        if not manifest.is_file():
            missing.append(manifest.name)
        inp["manifest"] = str(manifest)
    for name in ("train", "valid"):
        original = parts.get(name)
        if not original or original == "sealed":
            continue
        candidate = shaped / Path(original).name
        if not candidate.is_file():
            missing.append(candidate.name)
        parts[name] = str(candidate)
    if missing:
        raise SystemExit(
            f"ERROR: {shaped} does not hold {', '.join(missing)}. --shaped wants the "
            "directory the recipe's parts were written to, under the names netspec.json "
            "records; the bundle does not carry them and will not invent them.")
    if epochs:
        d.setdefault("training", {})["max_epochs"] = int(epochs)
    return NetSpec.from_dict(d)


class _Log:
    """The epoch record, printed and appended. `fit` hands these over; writing them is the
    caller's job, which is why a bundle that wanted no history could pass nothing."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.write_text("", encoding="utf-8")
        self.n = 0

    def on_epoch(self, record):
        row = record.as_dict()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        self.n += 1
        train = (row.get("train") or {}).get("loss")
        valid = (row.get("valid") or {}).get("loss")
        print(f"  epoch {row['epoch']:>3}  train {train:.5f}  valid {valid:.5f}"
              if train is not None and valid is not None else f"  epoch {row['epoch']}")

    def on_first_step(self, record):
        pass

    def on_step(self, record):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shaped", required=True, help="the directory holding the shaped parts")
    ap.add_argument("--out", default="runs/local", help="where the checkpoint and history go")
    ap.add_argument("--epochs", type=int, default=None, help="override training.max_epochs")
    ap.add_argument("--budget", type=float, default=None, help="minutes of wall clock")
    a = ap.parse_args(argv)

    spec = spec_over(a.shaped, a.epochs)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print(f"OK: {ROOT.name} — fitting netspec.json on the parts in {a.shaped}")
    print("    this is one run of one spec; the network this bundle ships was selected by a "
          "campaign, and what comes out here is not it")

    model = Network.build(spec)
    log = _Log(out / "history.jsonl")
    run = RunContext(budget_min=a.budget, checkpoint_dir=str(out))
    result = Trainer(ModelRegistry()).fit(model, spec, ShapedParts(), callbacks=log, run=run)
    stopped = getattr(result.stopped_by, "value", result.stopped_by)
    print(f"OK: {result.epochs_run} epoch(s), stopped by {stopped}; best epoch "
          f"{result.best_epoch} at {result.best_value}")
    print(f"OK: checkpoint {out / 'best.pt'}; history {out / 'history.jsonl'} "
          f"({log.n} row(s))")
    print("NOTE: MANIFEST.json, model-card.html and predict.ipynb describe the shipped "
          "weights, not these. Nothing here has been evaluated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
