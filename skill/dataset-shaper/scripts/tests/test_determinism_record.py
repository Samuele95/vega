#!/usr/bin/env python3
"""
test_determinism_record.py — `shape.py --check` records its determinism verdict where the layer
reads it (F45; `shape-contract.md` §3, SKILL.md step 4).

    python3 -m pytest test_determinism_record.py      (or: python3 test_determinism_record.py)

The release4 rehearsal ran the documented steps — the verifier with `--json
<dir>/verification.json`, then `shape.py --check` — and `apply_shape_layer.py` refused the layer:
"verification.determinism is required". No script wrote it: `--check` printed its verdict and
recorded nothing, and the session typed `determinism: pass` into the report by hand, as the
package's own lane harness and this suite's layer fixture did. What is proved, through the real
scripts on the graph-keypair fixture:

  pass        after the verifier's report, `--check` adds `determinism: pass` and keeps every
              key the verifier wrote; the apply script publishes the layer with no hand edit;
  fail        a part altered after the run: `--check` exits 3 and records `determinism: fail`,
              and the apply script refuses to publish it;
  order       a report rewritten by the verifier after `--check` has no determinism again, and
              the apply script refuses it (run the check after the verifier).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _layer_fixture as F  # noqa: E402


def _apply(model, out, tmp):
    data = F.write_doc(tmp / "shape-layer.json", F.judgement(from_run=str(out)))
    return F.apply(model, data, tmp / "out.html")


def test_check_records_pass_and_the_layer_publishes(tmp_path):
    model, out = F.build(tmp_path / "w")            # verifier --json, then --check: no hand edit
    report = json.loads((out / "verification.json").read_text(encoding="utf-8"))
    assert report["determinism"] == "pass"
    assert {"structural", "semantic", "distributional", "split"} <= set(report), sorted(report)
    rc, said = _apply(model, out, tmp_path)
    assert rc == 0, said[-2000:]


def test_a_failed_check_records_fail_and_the_layer_is_refused(tmp_path):
    model, out = F.build(tmp_path / "w")
    mp = out / "manifest.json"
    original = mp.read_text(encoding="utf-8")
    man = json.loads(original)                   # a manifest the rerun cannot reproduce
    man["outputs"]["files"]["train"]["digest"] = "sha256:" + "0" * 64
    mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    rc, said = F.run([F.SCRIPTS / "shape.py", "--check", "--out-dir", out])
    assert rc == 3 and "determinism check FAILED" in said, said[-1500:]
    assert json.loads((out / "verification.json").read_text(encoding="utf-8"))["determinism"] == "fail"
    mp.write_text(original, encoding="utf-8")    # the manifest restored; the recorded fail stands
    rc, said = _apply(model, out, tmp_path)
    assert rc != 0 and "determinism" in said, said[-1500:]


def test_a_report_rewritten_after_the_check_is_refused(tmp_path):
    model, out = F.build(tmp_path / "w")
    rc, said = F.run([F.SCRIPTS / "verify_shape.py", "--recipe", F.RECIPE, "--out-dir", out,
                      "--json", out / "verification.json"])
    assert rc == 0, said[-1500:]
    assert "determinism" not in json.loads((out / "verification.json").read_text(encoding="utf-8"))
    rc, said = _apply(model, out, tmp_path)
    assert rc != 0 and "determinism" in said, said[-1500:]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
