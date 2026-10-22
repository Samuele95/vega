#!/usr/bin/env python3
"""
test_page_math_advanced.py — the advanced explainers' arithmetic against the network (BUILD-PLAN
W14, D4 §8.1): `js/how/explainers/{conv, recurrent, attention, graph, latent}.js` recompute, from the
weights `GET v1/network` publishes and the numbers `POST v1/explain` (detail "exact", R-F5) brings,
what the network computed, and the stepper's reading of each step (the probe's
`state().stepper.values`, R-F1) must agree with the network within the probe's tolerance
max(1e-3, 1e-3·|network|) (`how/sum.js` GAP_WITHIN).

    python3 -m pytest -q test_page_math_advanced.py            # the zoo; the lanes when given
    python3 test_page_math_advanced.py --bundles DIR [pytest args]
    (env VEGA_PAGE_BUNDLES=DIR: a matrix.sh lanes directory, DIR/<lane>/work/bundle/<bundle>/,
     or a directory of bundles, DIR/<lane>/MANIFEST.json)

Two sources of real payloads, never hand-written ones:

- **the inspection zoo** (`vega-forge/scripts/tests/test_inspect_zoo.py`, I10): its 133 cases —
  every admitted family, shape, graph level and task, built by the rendered package's own
  `build_model` and explained through `NetworkInspector` — each explained here at detail "exact"
  and at detail "full", with its network at weights "full" and "summary". At "exact" every step of
  a W14 kind (conv, recurrent, attention, attention_pool, transformer, tokenizer, latent,
  graph_conv, graph_pool) must recompute, each of its checks within the tolerance; every such
  kind, and every block type behind them (conv1d; rnn, gru, lstm, one and two directions, every
  `return`; attention pooled; attention_pool; transformer_encoder; feature_tokenizer;
  reparameterize; gcn, sage — normalised too —, gat and gin convs at node and graph level;
  graph_pool), must be met. At "full" (the reduced forms, no exact tensor) and with the weights
  withheld, those steps must say what stops them — the answer's missing tensor ids, or the
  weights it did not receive — and recompute nothing, never fail. No step falls back to the
  generic explainer, no key is missing and no placeholder is left.
- **the lane bundles** (the closing test of W14: L10 gru, L12 cnn1d, L17 lstm-attention, L19 gcn
  node level, L22 gin graph level, L08 vae, L09 deep-svdd): each bundle's own service, started
  in-process (`serve.app.create_app`), answers `GET /v1/network?weights=full` and `POST
  /v1/explain` with its `examples/request.json` at detail "exact", exactly as the page asks; the
  stepper's every step — W14's and the core explainers' — must then be within the tolerance.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
PAGE = TESTS.parents[1] / "assets" / "page"
FORGE = TESTS.parents[2] / "vega-forge" / "scripts"
ZOO = FORGE / "tests"
EXPLAIN = FORGE / "backends" / "pytorch" / "explain-catalog.json"
TOLERANCE = 1e-3                    # the probe's: max(1e-3, 1e-3·|network|), how/sum.js GAP_WITHIN
ADVANCED = ("conv", "recurrent", "attention", "attention_pool", "transformer", "tokenizer", "latent",
            "graph_conv", "graph_pool")
LANES = ("L10", "L12", "L17", "L19", "L22", "L08", "L09")
BUNDLES_ENV = "VEGA_PAGE_BUNDLES"


# ------------------------------------------------------------------------- the page, run by node
STEPPER = """
  import { readFileSync } from "node:fs";
  import { install } from "./tests/dom-shim.js";
  const doc = install();
  const { StageGraph } = await import("./js/net/graph.js");
  const { Stepper } = await import("./js/how/stepper.js");
  const { configureText, missingKeys, unresolved } = await import("./js/core/text.js");
  const { Lexicon } = await import("./js/core/lexicon.js");
  const cases = JSON.parse(readFileSync(process.argv[1], "utf8")), out = {};
  for (const c of cases) {
    const graph = StageGraph.from(c.view);
    configureText({ lex: new Lexicon(null, null), lane: { viewKey: graph.viewKey, shape: graph.sample.shape }, explain: c.page.explain });
    const host = doc.createElement("div");
    doc.body.append(host);
    const st = await new Stepper({ graph, ex: c.ex, page: c.page, onCode: () => {} }).mount(host);
    const bodies = st.steps().map((_, i) => { st.show(i); return doc.getElementById("how-body").textContent; });
    const r = st.reading();
    out[c.name] = { values: r.values, gaps: r.gaps.map((g) => g.kind), failed: r.failed, bodies };
    doc.body.replaceChildren();
  }
  console.log(JSON.stringify({ cases: out, missing: missingKeys(), unresolved: unresolved() }));
"""


def run_page(cases: list) -> dict:
    """The stepper's reading of every case (`{name, view, ex, page}`), walked in a small DOM."""
    node = shutil.which("node")
    assert node, "node is required: the page's arithmetic is its ES modules"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cases.json"
        path.write_text(json.dumps(cases), encoding="utf-8")
        run = subprocess.run([node, "--input-type=module", "-e", STEPPER, str(path)], cwd=PAGE,
                             capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout)


def page_of(view: dict) -> dict:
    """The part of page.json the stepper reads: the explain catalog's entries for the network's
    block types, every activation and the task's link (page_config._explain's subset)."""
    catalog = json.loads(EXPLAIN.read_text(encoding="utf-8"))
    types = sorted({n["type"] for n in view["nodes"] if n["type"] in catalog["blocks"]})
    return {"explain": {"blocks": {t: catalog["blocks"][t] for t in types}, "activations": catalog["activations"],
                        "links": {view["task"]["link"]: catalog["links"][view["task"]["link"]]}}}


def parts_of(check: dict) -> list:
    return [check, *check.get("ops", [])]


def out_of_tolerance(check: dict) -> list:
    """Each part of a step's recompute whose gap passes max(1e-3, 1e-3·|network|), or that the page
    could not compare because the network's numbers do not fit (`fit: false`), as text."""
    return [f"{p.get('what')}: {p.get('sizes')} does not fit {p.get('shape')}" if p.get("fit") is False
            else f"{p.get('what')}: page {p['page']} vs network {p['network']}" for p in parts_of(check)
            if p.get("fit") is False or abs(p["network"] - p["page"]) > max(TOLERANCE, TOLERANCE * abs(p["network"]))]


def advanced_steps(got: dict):
    return [s for s in got["values"] if s["kind"] in ADVANCED]


# ------------------------------------------------------------------------- the zoo's payloads
@lru_cache(maxsize=1)
def zoo():
    """test_inspect_zoo as a module (it adds its own paths)."""
    pytest.importorskip("torch", reason="the zoo builds the networks with torch")
    sys.path.insert(0, str(ZOO))
    os.environ.pop("VEGA_ZOO_DUMP", None)
    return importlib.import_module("test_inspect_zoo")


def _canonical(body) -> dict:
    return json.loads(json.dumps(body, allow_nan=False))


@lru_cache(maxsize=1)
def zoo_cases() -> tuple:
    """(exact, full, withheld): the zoo's cases with an advanced block, as stepper cases — at
    detail exact over the full weights, at detail full, and at detail exact over the summary
    weights (every tensor withheld)."""
    Z, exact, full, withheld = zoo(), [], [], []
    for cid in sorted(Z.CASES):
        r = Z.Run(Z.CASES[cid])
        assert r.status["ready"], f"{cid}: {r.status['why']}"
        view = json.loads(r.network_bytes)
        if not ({n["type"] for n in view["nodes"]} & ADVANCED_TYPES):
            continue
        ex = _canonical(r.body("exact"))
        exact.append({"name": cid, "view": view, "ex": ex, "page": page_of(view)})
        full.append({"name": cid, "view": view, "ex": _canonical(r.body("default")), "page": page_of(view)})
        summary = json.loads(r.inspector.network_bytes("summary"))
        withheld.append({"name": cid, "view": summary, "ex": ex, "page": page_of(summary)})
    return exact, full, withheld


# The block types behind the W14 step kinds (net/kinds.js KIND_OF).
ADVANCED_TYPES = {"conv1d", "rnn", "gru", "lstm", "attention", "attention_pool", "transformer_encoder",
                  "feature_tokenizer", "reparameterize", "gcn_conv", "sage_conv", "gat_conv", "gin_conv",
                  "graph_pool"}


@lru_cache(maxsize=3)
def zoo_reading(which: int) -> dict:
    return run_page(list(zoo_cases()[which]))


def test_every_advanced_step_of_the_zoo_recomputes_within_the_probe_tolerance():
    got = zoo_reading(0)
    assert got["missing"] == [] and got["unresolved"] == [], (got["missing"], got["unresolved"])
    problems = []
    for name, reading in got["cases"].items():
        assert reading["failed"] == [], f"{name}: {reading['failed']}"
        for s in advanced_steps(reading):
            chk = s["check"]
            if chk is None or "error" in chk:
                problems.append(f"{name} {s['name']} ({s['kind']}): {chk and chk.get('error') or 'no recompute'}")
                continue
            bad = out_of_tolerance(chk)
            if bad or not chk["within"]:
                problems.append(f"{name} {s['name']} ({s['kind']}): {bad or 'not within'}")
    assert problems == [], "\n".join(problems[:20])


def test_every_advanced_kind_and_block_type_is_met():
    got, cases = zoo_reading(0), {c["name"]: c for c in zoo_cases()[0]}
    kinds, types = set(), set()
    variants = set()
    for name, reading in got["cases"].items():
        view = cases[name]["view"]
        byid = {n["id"]: n for n in view["nodes"]}
        for s in advanced_steps(reading):
            if s["check"] is None:
                continue
            kinds.add(s["kind"])
            n = byid.get(s["stage"]) or {}
            types.add(n.get("type"))
            cfg = n.get("config") or {}
            if s["kind"] == "recurrent":
                variants.add(("recurrent", n["type"], int(n["params"]["directions"]), n["params"]["return"]))
            if s["kind"] == "graph_conv":
                variants.add(("graph", n["type"], view["sample"]["level"], bool(cfg.get("normalize") is True)))
            if s["kind"] == "attention":
                variants.add(("attention", bool(cfg.get("pool"))))
    assert kinds == set(ADVANCED), sorted(set(ADVANCED) - kinds)
    assert types >= ADVANCED_TYPES, sorted(ADVANCED_TYPES - types)
    for cell in ("rnn", "gru", "lstm"):
        assert ("recurrent", cell, 1, "last") in variants or ("recurrent", cell, 1, "all") in variants, cell
    assert ("recurrent", "gru", 2, "last") in variants and ("recurrent", "lstm", 2, "last") in variants
    assert ("recurrent", "gru", 1, "all") in variants and ("recurrent", "lstm", 1, "all") in variants
    for conv in ("gcn_conv", "sage_conv", "gat_conv"):
        assert ("graph", conv, "node", False) in variants and ("graph", conv, "graph", False) in variants, conv
    assert ("graph", "gin_conv", "graph", False) in variants
    assert ("graph", "sage_conv", "node", True) in variants, "a normalised sage conv (A31's exception)"
    assert ("attention", True) in variants, "an attention layer pooled over the positions"


def test_no_advanced_step_falls_back_to_the_generic_explainer():
    for which in (0, 1, 2):
        got = zoo_reading(which)
        for name, reading in got["cases"].items():
            assert [k for k in reading["gaps"] if k in ADVANCED] == [], f"{name}: {reading['gaps']}"
        assert got["missing"] == [], got["missing"]


def test_a_reduced_answer_says_which_tensors_it_lacks_and_recomputes_nothing():
    got, cases = zoo_reading(1), {c["name"]: c for c in zoo_cases()[1]}
    said = 0
    for name, reading in got["cases"].items():
        assert reading["failed"] == [], f"{name}: {reading['failed']}"
        for s in reading["values"]:
            if s["kind"] not in ADVANCED:
                continue
            if s["kind"] == "latent" and s["check"] is not None:
                # a row's latent code reads two vectors, which a full answer carries whole
                assert s["check"]["within"] and not out_of_tolerance(s["check"]), f"{name} {s['name']}: {s['check']}"
                continue
            assert s["check"] is None, f"{name} {s['name']}: recomputed from reduced forms {s['check']}"
            assert "did not bring" in reading["bodies"][s["index"]], f"{name} {s['name']}: {reading['bodies'][s['index']][:300]}"
            said += 1
        assert cases[name]["ex"]["detail"] == "full"
    assert said > 100


def test_withheld_weights_are_said_to_be_withheld_and_nothing_is_recomputed():
    got = zoo_reading(2)
    said = 0
    for name, reading in got["cases"].items():
        assert reading["failed"] == [], f"{name}: {reading['failed']}"
        for s in reading["values"]:
            if s["kind"] not in ADVANCED or s["kind"] == "latent" or (s["kind"] == "graph_pool"):
                continue                    # a latent code and a readout carry no weight
            assert s["check"] is None, f"{name} {s['name']}: {s['check']}"
            assert "did not receive the weights" in reading["bodies"][s["index"]], f"{name} {s['name']}"
            said += 1
    assert said > 100


# ------------------------------------------------------------------------- the lane bundles
def bundles_dir():
    raw = os.environ.get(BUNDLES_ENV)
    return Path(raw) if raw else None


def bundle_of(lane: str):
    """The packed bundle of a lane under the bundles directory, or None."""
    base = bundles_dir()
    if base is None:
        return None
    for cand in [base / lane, *sorted((base / lane / "work" / "bundle").glob("*/"))]:
        if (cand / "MANIFEST.json").is_file() and (cand / "serve").is_dir():
            return cand
    return None


SERVED = """
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from fastapi.testclient import TestClient
from serve.app import create_app
request = json.loads((root / "examples" / "request.json").read_text(encoding="utf-8"))
with TestClient(create_app(root)) as client:
    net = client.get("/v1/network", params={"weights": "full"})
    ex = client.post("/v1/explain", json={**request, "detail": "exact"})
print(json.dumps({"network": [net.status_code, net.json()], "explain": [ex.status_code, ex.json()]}))
"""


def served_payloads(bundle: Path) -> tuple:
    """(network-view, explanation) as the bundle's own service answers the page."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    run = subprocess.run([sys.executable, "-c", SERVED, str(bundle)], capture_output=True, text=True,
                         timeout=600, cwd=tempfile.gettempdir(), env=env)
    assert run.returncode == 0, run.stderr[-4000:]
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["network"][0] == 200, out["network"]
    assert out["explain"][0] == 200, str(out["explain"])[:1000]
    return out["network"][1], out["explain"][1]["explanation"]


@pytest.mark.parametrize("lane", LANES)
def test_the_lane_bundle_steps_recompute_within_the_probe_tolerance(lane):
    bundle = bundle_of(lane)
    if bundle is None:
        pytest.skip(f"no {lane} bundle: set {BUNDLES_ENV} to a matrix.sh lanes directory (--to pack)")
    view, ex = served_payloads(bundle)
    assert ex["detail"] == "exact" and not ex["degraded"], f"{lane}: the explanation was summarised"
    got = run_page([{"name": lane, "view": view, "ex": ex, "page": page_of(view)}])
    reading = got["cases"][lane]
    assert got["missing"] == [] and got["unresolved"] == [], (got["missing"], got["unresolved"])
    assert reading["gaps"] == [] and reading["failed"] == [], (reading["gaps"], reading["failed"])
    problems = []
    for s in reading["values"]:
        if s["check"] is None:
            assert s["kind"] in ("input",), f"{lane} {s['name']} ({s['kind']}): no recompute"
            continue
        bad = out_of_tolerance(s["check"]) if "error" not in s["check"] else [s["check"]["error"]]
        if bad or not s["check"].get("within"):
            problems.append(f"{lane} {s['name']} ({s['kind']}, {s['stage']}): {bad or 'not within'}")
    assert problems == [], "\n".join(problems)
    assert any(s["kind"] in ADVANCED for s in reading["values"]) or lane == "L09", f"{lane}: no advanced step"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundles", help="a matrix.sh lanes directory, or a directory of bundles by lane")
    args, rest = ap.parse_known_args(argv)
    if args.bundles:
        os.environ[BUNDLES_ENV] = str(Path(args.bundles).resolve())
    return pytest.main([__file__, "-q", *rest])


if __name__ == "__main__":
    sys.exit(main())
