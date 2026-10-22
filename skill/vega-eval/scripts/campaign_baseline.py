#!/usr/bin/env python3
"""
campaign_baseline — the campaign's own `R0` run, predicted through the seal (eval-contract §4).

`campaign-baseline` is a comparison row, not a fitted baseline: the network as the forge
engineered it, before the ladder and the search, so the evaluation can say on the test part
what the campaign bought. Its valid number is the ledger's (upstream, no new interval); its
**test** number is measured here like any other model's, on the same rows and the same
resamples, so it carries a CI and a paired difference against the selected run.

R0's checkpoint is read with R0's own emitted code: a rung or a trial may carry its own copy of
the package (`runs/<id>/<package>`, emitted with its own block sizes), and two copies of one
package cannot live in one interpreter. So `evaluate.py` — inside its one look, after the seal is
written — runs this file as a child process: it opens R0's package by blueprint-map role, loads
`runs/R0/checkpoints/best.pt` through the ModelStore, opens the test part through
`load_sealed_part` with the token the seal holds, scores it through the package Evaluator's one
scoring path, and hands back the scores and their row ids. The child is never run by hand: it
refuses without an existing seal, and it writes nothing but the score file it is given.

The parent half (`predict`) checks first what can be checked without the test part — an R0 row
in the ledger, its checkpoint, the checkpoint's digest — and afterwards that the child scored
the very rows the network was scored on; anything else leaves the row unpredicted, with why.

Exit codes (child): 0 scored; 2 refused.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_eval_core import digest_file  # noqa: E402

__all__ = ["R0", "predict"]

#: the run the campaign baseline is: the spec as engineered (run-contract §4)
R0 = "R0"


def _numpy(values):
    import numpy as np
    detach = getattr(values, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(values)


def predict(camp_dir, seal_path, ledger, row_ids, forge_dir, timeout=1800):
    """(R0's test score aligned to `row_ids`, None) or (None, why it could not be predicted)."""
    import numpy as np
    camp_dir = Path(camp_dir)
    row = next((r for r in ledger.get("runs") or () if r.get("id") == R0), None)
    if row is None:
        return None, "the ledger holds no R0 run (the spec as engineered) to compare against"
    checkpoint = camp_dir / "runs" / R0 / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        return None, f"R0 has no checkpoint at {checkpoint.relative_to(camp_dir)}"
    if row.get("checkpoint_digest") and digest_file(checkpoint) != row["checkpoint_digest"]:
        return None, ("R0's checkpoint does not hash to the ledger's checkpoint_digest: the "
                      "weights changed since the run, and a comparison against other weights "
                      "is worse than none")
    with tempfile.TemporaryDirectory(prefix="vega-eval-r0-") as td:
        out = Path(td) / "r0.npz"
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--campaign", str(camp_dir),
             "--seal", str(seal_path), "--out", str(out), "--vega-forge-dir", str(forge_dir)],
            capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not out.is_file():
            said = [ln[len("ERROR: "):] for ln in (proc.stdout + proc.stderr).splitlines()
                    if ln.startswith("ERROR: ")]
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["no output"]
            return None, ("R0 could not be scored through the seal: "
                          + ("; ".join(said) if said else tail[0][:300]))
        stored = np.load(out, allow_pickle=False)
        score, theirs = stored["score"], stored["row_ids"]
    if len(theirs) != len(row_ids) or not np.array_equal(np.asarray(theirs).astype(str),
                                                         np.asarray(row_ids).astype(str)):
        return None, (f"R0 scored {len(theirs)} test sample(s) and the selected run "
                      f"{len(row_ids)}, not the same rows in the same order (its sample differs), "
                      "so no paired comparison exists")
    return score, None


# --------------------------------------------------------------------------- the child

def _child(args) -> int:
    import numpy as np
    from eval_package import open_package, resolved_spec
    camp_dir, seal_path = Path(args.campaign), Path(args.seal)
    if not seal_path.is_file():
        print(f"ERROR: no seal at {seal_path}; R0 is predicted only inside evaluate.py's look")
        return 2
    try:
        spec = json.loads((camp_dir / "netspec.json").read_text(encoding="utf-8"))
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        resolved = resolved_spec(camp_dir, R0, spec)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: the campaign's records cannot be read ({exc})")
        return 2
    package, err = open_package(camp_dir, spec, R0, resolved, Path(args.vega_forge_dir))
    if err:
        print(f"ERROR: {err}")
        return 2
    adapters, dom = package.adapters, package.domain
    spec_obj = dom.NetSpec.from_dict(resolved)
    model = adapters.store.load(camp_dir / "runs" / R0 / "checkpoints" / "best.pt",
                                spec=spec_obj, build=adapters.build,
                                run_dir=str(package.directory.parent))
    test = adapters.data.load_sealed_part(spec_obj, seal["token"], str(seal_path))
    scored = adapters.evaluator.scored(spec_obj, test, model)
    # row ids as text: a graph-level part's ids are graph_id values, and the score file is read
    # back without pickling
    np.savez(args.out, score=np.asarray(_numpy(scored.score), dtype=np.float64),
             row_ids=np.asarray(scored.row_ids).astype(str))
    print(f"OK: R0 scored {len(scored)} test sample(s) through the seal")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--seal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vega-forge-dir", required=True)
    return _child(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
