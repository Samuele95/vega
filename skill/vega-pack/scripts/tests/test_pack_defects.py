#!/usr/bin/env python3
"""
test_pack_defects.py — the pack defects the W7 lane run found (PLAN-AMENDMENTS A42, track C4),
each held fixed by a test that fails on the code that had it.

    python3 test_pack_defects.py            # or: python3 -m pytest test_pack_defects.py

Pure: nothing is trained or served. (L13's missing-label decoding is in test_output_record.py,
beside the schema it breaks.)

  * L20 — `graph_pack` compares the edges_for columns of the shaped manifest and of the emitted
    transform as a set (edges_for reads them by name), refuses a missing, an extra or a doubled
    column as before, and the reference keeps the transform's order, which the Predictor's
    load-time check compares as a list;
  * N-S3 — a transductive campaign is refused before anything is written under --bundle-dir;
  * B18 — a selected run that re-emitted its package (a rung, a trial) ships its own smoke
    (per_block, params_total) in `records/architecture.json`, with `smoke_of` saying whose;
  * A44(b) — the model card of such a bundle describes the selected run (the bundle's
    `netspec.json` and that smoke), not the forged spec, and says which run it is;
  * L14 / L15 / L17 — the bundle layer states a recorded statistic (the verdict's paired
    difference, its primary metric, the round trip's worst difference) as the record writes
    it — Python's repr — so the smoke finds `str(value)` in the text whatever its magnitude;
  * A40 — every parquet read in vega-pack goes through Arrow's own filesystem.
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
sys.path.insert(0, str(SCRIPTS))

import bundle_manifest  # noqa: E402
import graph_pack  # noqa: E402
import model_card  # noqa: E402
import pack  # noqa: E402
import pack_options  # noqa: E402
from pack_context import PackExit  # noqa: E402

RENDER = SCRIPTS.parent / "assets" / "bundle-render.js"


def _exit(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except PackExit as refusal:
        return refusal.code, "; ".join(refusal.lines)
    return 0, ""


# --------------------------------------------------------------------------- L20

def _transform(td, columns, level="node"):
    path = Path(td) / "preprocess.py"
    entry = "edges_for" if level == "node" else "edges_within"
    path.write_text(f"GRAPH = {{'level': {level!r}}}\nEDGES_FOR_COLUMNS = {columns!r}\n\n"
                    f"def {entry}(rows, reference=None):\n    return None\n", encoding="utf-8")
    return path


GRAPH = {"level": "node", "edges_for": {"columns": ["raw:paper_key", "raw:cites"],
                                        "raw": {"raw:paper_key": "paper_key",
                                                "raw:cites": "cites"}}}


def test_the_edges_for_columns_are_compared_as_a_set_and_keep_the_transforms_order():
    with tempfile.TemporaryDirectory() as td:
        order = graph_pack._check_transform(_transform(td, ["raw:cites", "raw:paper_key"]),
                                            GRAPH)
        assert order == ["raw:cites", "raw:paper_key"]
        assert graph_pack._check_transform(_transform(td, ["raw:paper_key", "raw:cites"]),
                                           GRAPH) == ["raw:paper_key", "raw:cites"]
        for wrong in (["raw:cites"], ["raw:cites", "raw:paper_key", "raw:extra"],
                      ["raw:cites", "raw:cites", "raw:paper_key"], "raw:cites", None):
            code, text = _exit(lambda: graph_pack._check_transform(_transform(td, wrong), GRAPH))
            assert code == 3 and "edges_for columns" in text, (wrong, code, text)
        level = {"level": "graph", "edges_for": {}}
        assert graph_pack._check_transform(_transform(td, None, "graph"), level) is None


def test_the_reference_lists_the_columns_in_the_transforms_order():
    reordered = graph_pack._in_order(GRAPH, ["raw:cites", "raw:paper_key"])
    assert graph_pack._edges_for(reordered) == (["raw:cites", "raw:paper_key"],
                                                GRAPH["edges_for"]["raw"])
    assert graph_pack._edges_for(GRAPH)[0] == ["raw:paper_key", "raw:cites"], "not mutated"
    assert graph_pack._in_order(GRAPH, None) is GRAPH
    code, text = _exit(lambda: graph_pack._in_order(GRAPH, ["raw:cites"]))
    assert code == 3 and "not the shaped manifest's" in text, text


# --------------------------------------------------------------------------- N-S3

def test_a_transductive_campaign_is_refused_before_bundle_dir_is_touched():
    graph_pack.refuse_unserved({"data_contract": {"sample": {"shape": "row"}}})
    graph_pack.refuse_unserved({"data_contract": {"sample": {"shape": "graph",
                                                             "protocol": "inductive"}}})
    spec = {"data_contract": {"sample": {"shape": "graph", "protocol": "transductive"}}}
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "camp" / "runs" / "R0"
        run.mkdir(parents=True)
        (run / "netspec.resolved.json").write_text(json.dumps(spec), encoding="utf-8")
        bundle_dir = Path(td) / "out"
        campaign = SimpleNamespace(sources={"netspec": run / "netspec.resolved.json"},
                                   verdict={"earned": True})
        writes = []
        saved = (pack.Campaign.read, pack._check_sources, pack._check_provenance,
                 pack.reemit.prepare)
        pack.Campaign.read = staticmethod(lambda options: campaign)
        pack._check_sources = lambda c: None
        pack._check_provenance = lambda c, m: None
        pack.reemit.prepare = lambda *a: writes.append(a) or bundle_dir.mkdir()
        try:
            options = pack_options.parse(["--campaign", str(run.parent.parent), "--shaped",
                                          str(td), "--bundle-dir", str(bundle_dir)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = pack.pack(options)
        finally:
            (pack.Campaign.read, pack._check_sources, pack._check_provenance,
             pack.reemit.prepare) = saved
        assert code == 2, out.getvalue()
        assert "transductive" in out.getvalue() and "inductive graphs only" in out.getvalue()
        assert writes == [] and not bundle_dir.exists(), "the refusal came after a write"


def test_the_refusal_precedes_the_first_write_in_the_assembly():
    tree = ast.parse((SCRIPTS / "pack.py").read_text(encoding="utf-8"))
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_assemble")
    calls = [ast.unparse(n.func) for n in ast.walk(body) if isinstance(n, ast.Call)]
    assert calls.index("graph_pack.refuse_unserved") < calls.index("reemit.prepare"), calls


# --------------------------------------------------------------------------- B18

FORGED = {"params_total": 10705, "per_block": [{"id": "B3", "out_shape": [64], "params": 8576}]}
TRIAL = {"params_total": 184937, "per_block": [{"id": "B3", "out_shape": [296],
                                                "params": 177008}], "madds_per_sample": 1}


def _campaign(td, kind="hpo", smoke=None, own_package=True):
    camp = Path(td) / "camp"
    run = camp / "runs" / "H1-3"
    run.mkdir(parents=True)
    if own_package:
        (run / "pipeline").mkdir()
    if smoke is not None:
        (run / "smoke.json").write_text(smoke if isinstance(smoke, str) else json.dumps(smoke),
                                        encoding="utf-8")
    return SimpleNamespace(run_id="H1-3", run_dir=run, package="pipeline",
                           ledger_row={"id": "H1-3", "kind": kind})


def _page(td):
    layer = {"netspec": {"blocks": [{"id": "B3"}]}, "smoke": FORGED, "readings": {}}
    page = Path(td) / "model.html"
    page.write_text('<html><script id="layer-architecture-data" type="application/json">'
                    + json.dumps(layer) + "</script></html>", encoding="utf-8")
    return page


def test_a_trial_ships_its_own_smoke_in_the_architecture_record():
    with tempfile.TemporaryDirectory() as td:
        paths = SimpleNamespace(root=Path(td) / "bundle")
        markers = []
        with contextlib.redirect_stdout(io.StringIO()):
            written = bundle_manifest.copy_records(_page(td), paths, markers,
                                                   _campaign(td, smoke=TRIAL))
        record = json.loads((paths.root / "records" / "architecture.json").read_text("utf-8"))
        assert "records/architecture.json" in written
        assert record["smoke"] == TRIAL
        assert record["netspec"] == {"blocks": [{"id": "B3"}]}, "the forged spec stays"
        of = record["smoke_of"]
        assert (of["run"], of["kind"], of["source"]) == ("H1-3", "hpo", "runs/H1-3/smoke.json")
        assert of["digest"].startswith("sha256:")
        assert of["forged"] == {"params_total": 10705, "per_block": FORGED["per_block"]}
        assert any(m.startswith("OK: records/architecture.json — smoke of run H1-3 (184937")
                   for m in markers), markers


def test_a_run_on_the_campaigns_package_keeps_the_forged_smoke():
    with tempfile.TemporaryDirectory() as td:
        markers = []
        doc = {"smoke": FORGED}
        out = bundle_manifest._with_run_smoke(doc, _campaign(td, kind="ablation",
                                                             own_package=False), markers)
        assert out is doc and markers == []


def test_a_reemitted_run_without_its_smoke_says_so_and_a_broken_one_is_refused():
    with tempfile.TemporaryDirectory() as td:
        markers = []
        with contextlib.redirect_stdout(io.StringIO()):
            out = bundle_manifest._with_run_smoke({"smoke": FORGED}, _campaign(td), markers)
        assert out["smoke"] == FORGED and "smoke_of" not in out
        assert markers and markers[0].startswith("WARN: run H1-3 re-emitted its package")
    for bad in ("{not json", {"per_block": []}, {"params_total": 3}):
        with tempfile.TemporaryDirectory() as td:
            code, text = _exit(lambda: bundle_manifest._with_run_smoke(
                {"smoke": FORGED}, _campaign(td, smoke=bad), []))
            assert code == 2 and "H1-3" in text, (bad, text)


# --------------------------------------------------------------------------- A44(b)

FORGED_SPEC = {"blocks": [{"id": "B3", "type": "dense"}],
               "loss": {"name": "cross_entropy", "params": {"weight": [0.5, 2.0]}}}
RUNG_SPEC = {"blocks": [{"id": "B3", "type": "dense"}],
             "loss": {"name": "cross_entropy", "params": {"weight": None}}}


def _shipped(td, spec, record):
    root = Path(td) / "bundle"
    (root / "records").mkdir(parents=True)
    (root / "netspec.json").write_text(json.dumps(spec), encoding="utf-8")
    (root / "records" / "architecture.json").write_text(json.dumps(record), encoding="utf-8")
    return root


def test_the_card_describes_the_selected_rung_not_the_forged_spec():
    layer = {"netspec": FORGED_SPEC, "smoke": FORGED}
    with tempfile.TemporaryDirectory() as td:
        root = _shipped(td, RUNG_SPEC, {"netspec": FORGED_SPEC, "smoke": TRIAL,
                                        "smoke_of": {"run": "H1-3", "kind": "hpo"}})
        out = model_card.shipped_architecture(root, layer, {"run": "H1-3"})
    assert out["netspec"] == RUNG_SPEC and out["forged_netspec"] == FORGED_SPEC
    assert out["smoke"] == TRIAL and out["smoke_of"]["run"] == "H1-3"
    assert layer == {"netspec": FORGED_SPEC, "smoke": FORGED}, "the layer itself is not changed"


def test_the_card_of_the_forged_run_and_a_smoke_of_another_run_keep_the_layer():
    layer = {"netspec": FORGED_SPEC, "smoke": FORGED}
    with tempfile.TemporaryDirectory() as td:
        root = _shipped(td, FORGED_SPEC, {"netspec": FORGED_SPEC, "smoke": FORGED})
        assert model_card.shipped_architecture(root, layer, {"run": "R0"}) == layer
    with tempfile.TemporaryDirectory() as td:
        root = _shipped(td, FORGED_SPEC, {"smoke": TRIAL, "smoke_of": {"run": "R9"}})
        assert model_card.shipped_architecture(root, layer, {"run": "R7"}) == layer


def test_the_architecture_section_names_the_run_it_draws():
    with tempfile.TemporaryDirectory() as td:
        root = _shipped(td, RUNG_SPEC, {"smoke": TRIAL, "smoke_of": {"run": "H1-3"}})
        arch = model_card.shipped_architecture(root, {"netspec": FORGED_SPEC, "smoke": FORGED},
                                               {"run": "H1-3"})
    training = {"mechanics": {"loss": "L", "update_rule": "U"},
                "experiments": {"runs": [{"id": "H1-3", "kind": "hpo",
                                          "delta_text": "loss/params/weight: [0.5,2.0] → ·"}]}}
    src = model_card.Sources(layers={"architecture": arch, "training": training},
                             manifest={"run": "H1-3"},
                             blueprint_map={"entries": [{"file": "model/network.py"}]})
    check = model_card.Checker()
    text = model_card.architecture_section(src, check)
    assert not check.problems, check.problems
    assert ("This bundle ships run H1-3 (hpo), not the forged spec" in text
            and "its change from the forged spec: loss/params/weight" in text), text[-2000:]
    assert "the network as run H1-3&#x27;s resolved spec declares it" in text
    assert "184,937" in text or "184937" in text, "the counts are the run's own smoke"


# --------------------------------------------------------------------------- L14 / L15 / L17

def _render_rec(values):
    """The renderer's `rec` on `values`, run by node (the function as shipped, not a copy)."""
    node = shutil.which("node")
    if not node:
        return None
    src = RENDER.read_text(encoding="utf-8")
    start, end = src.index("  function rec(v)"), src.index("  function ciText")
    script = ("var isNum = function (v) { return typeof v === 'number' && isFinite(v); };\n"
              "var fmt = String;\n" + src[start:end] + "\nvar vals = JSON.parse("
              "require('fs').readFileSync(0, 'utf8'));\n"
              "process.stdout.write(JSON.stringify(vals.map(rec)));\n")
    proc = subprocess.run([node, "-e", script], input=json.dumps(values), capture_output=True,
                          text=True, timeout=60, check=True)
    return json.loads(proc.stdout)


def test_a_recorded_statistic_is_rendered_as_the_record_writes_it():
    rnd = random.Random(7)
    values = [0.0, -0.0, 1.0, -0.0505, 0.1111, 165.4472, 1e-05, 0.0001, 1e16, 2.5e-07,
              1.1920928955078125e-07, 2.384185791015625e-07, 9999999999999998.0, 0.1 + 0.2]
    for _ in range(2000):
        v = rnd.choice((1, -1)) * 10 ** rnd.uniform(-12, 18)
        values += [v, round(v, 4)]
    rendered = _render_rec(values)
    if rendered is None:
        print("SKIP: no node on PATH; the renderer's number format is checked by the lane smoke")
        return
    assert rendered == [repr(v) for v in values], [
        (repr(v), r) for v, r in zip(values, rendered) if repr(v) != r][:5]


def test_the_verdict_and_the_worst_difference_are_rendered_with_rec():
    src = RENDER.read_text(encoding="utf-8")
    verdict = src[src.index("    var sentence;"):src.index("'bn-verdict-text'")]
    assert "fmt(VERD.diff)" not in verdict and "fmt(PRIM.value)" not in verdict
    assert verdict.count("rec(VERD.diff)") == 3 and verdict.count("rec(PRIM.value)") == 3
    assert "seen was ' + rec(RT.max_abs_diff)" in src


# --------------------------------------------------------------------------- A40

def test_every_parquet_read_goes_through_arrows_filesystem():
    offenders = []
    for path in sorted(SCRIPTS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ast.unparse(node.func) in (
                    "pd.read_parquet", "pandas.read_parquet", "pq.read_table"):
                if not any(k.arg == "filesystem" for k in node.keywords):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], offenders


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


def test_the_documented_pack_call_carries_the_model():
    """F37 (rehearsal final3r2): SKILL.md step 2 gave pack.py's call without --model, and a
    session that followed it packed a bundle with no records/, which its notebook and card read."""
    import re
    skill = (Path(__file__).resolve().parents[2] / "SKILL.md").read_text(encoding="utf-8")
    calls = re.findall(r"`python3 scripts/pack\.py ([^`]*)`", skill)
    assert calls and all("--model" in c for c in calls if "--campaign" in c), calls
