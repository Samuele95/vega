#!/usr/bin/env python3
"""
test_geometry_layer.py — the /dataset-forge layer tooling, run the way the skill runs it
(build plan X2, D6 G9).

    python3 test_geometry_layer.py [<work-dir>] [-v]

On the orders fixture it goes through the SKILL's procedure with the real scripts: geometry.py
writes `<run-dir>/geometry.json`, the fixture's judgement half (`fixtures/orders.geometry-layer.json`,
`from_run: geometry.json`) sits beside it, a base model comes from data-lens's bootstrap_base.py
(a T-box-only dataset model), and then

  * `apply_geometry_layer.py` writes the layer through domain-forge's platform writer: the input
    is untouched, the output is a byte superset of it with one `geometry` block, the script's
    numbers are spliced verbatim, re-applying replaces the layer, a fixed `--produced-at` makes
    the output reproducible, `--out` equal to the input is refused;
  * `validate_model.py` exits 0 on the output (invariants 13-16 included), through the
    known-flake wrapper when the tree carries it;
  * `smoke_geometry.py --strict` mounts the four views on it and moves them (brush, PCA,
    re-orientation, candidate selection) — a browser is required — and fails on a copy of the
    explorer whose re-orientation does not redraw;
  * the validator refuses the contract's three defects — a derivation without its provenance
    triple, a candidate without its leakage set (or its consequence block), `symbolic:
    confirmed` without `verified_rows` — and the other rules it states (numbers not verbatim,
    a basis member that is a rule head, a column neither in the basis nor derived, an identity
    in the basis, the label's rule body among its features, the determinant of an exact
    functional dependency on the label among its features, a cycle without the report's
    orientation or without its `cycles` entry, a silent disagreement, provenance/chosen
    contradictions) and the structural ones (a rule body, a feature list, a leakage set, a
    cycle's members or an orientation's derived set that is not a list of column names, a
    container of the wrong type), each with exit 2 and a message naming the defect, and writes
    nothing;
  * a role left unstated follows the analyst's final_type, not the script's type, in the
    validator and in the explorer alike: a column the script typed id and the analyst retyped
    numeric is a basis member, and smoke --strict holds the explorer's basis to it (and fails on
    a copy of the explorer that reads the script's type first);
  * `env_check.py --env` reports dataset-forge complete (no specified-but-unbuilt file);
  * the shipped assets respect contract §6 (render script <= 80 KB, no network, CSS scoped);
  * /dataset-shaper reads the layer: its orders recipe's geometry: citations resolve, the
    report's orientation and an alternative one named by id drop the right rule heads, and the
    candidate's features are what remains beside the label;
  * the data-lens fixture's hand-written geometry layer, the other one the lane ships, validates.

The documents it feeds the validator are in `geometry_layer_cases.py`, the consumer stages in
`geometry_layer_consumers.py`, the shared paths and calls in `geometry_layer_testkit.py`.
Exit 0 when every check holds.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True               # the case module below leaves no __pycache__
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from geometry_layer_cases import NEGATIVES, POSITIVES, mutate, retyped, unattended_abstained  # noqa: E402
from geometry_layer_consumers import stage_messy, stage_sensors, stage_shaper  # noqa: E402
from geometry_layer_testkit import (APPLY, BOOTSTRAP, ENV_CHECK, FIXTURE, GEOMETRY, ORDERS,  # noqa: E402
                                    RENDER, ROOT, SCRIPTS, SMOKE, STYLE, VALIDATE, apply_layer,
                                    check, errors, layer_doc, results, run, sha, tail,
                                    validate_model)

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]


# ------------------------------------------------------------------ the run directory
def build(work):
    rd = work / "run"
    rd.mkdir(parents=True, exist_ok=True)
    rc, out = run([GEOMETRY, ORDERS, "--out", rd / "geometry.json"])
    if not check("geometry.py writes geometry.json", rc == 0 and (rd / "geometry.json").is_file(),
                 tail(out)):
        return None
    shutil.copyfile(FIXTURE, rd / "geometry-layer.json")
    if not check("data-lens bootstrap_base.py is in the tree", BOOTSTRAP.is_file(), str(BOOTSTRAP)):
        return None
    rc, out = run([BOOTSTRAP, ORDERS, "--out", work / "orders.base.html"])
    if not check("a base dataset model is written", rc == 0, tail(out)):
        return None
    return rd


def _check_spliced(doc, eng):
    """The script's half of the layer is geometry.json's, verbatim."""
    sp, es = doc["space"], eng["space"]
    check("space numbers are geometry.json's, with the contract's short names",
          sp["dims_95"] == es["dims_for_95pct_variance"] and sp["dims_99"] == es["dims_for_99pct_variance"]
          and sp["near_rank"] == es["near_rank_1e-6"] and sp["singular_values"] == es["singular_values"]
          and sp["intrinsic_dim_twonn"] == es["intrinsic_dim_twonn"] and sp.get("reading"))
    check("basis.intrinsic_dim and size are spliced",
          doc["basis"]["intrinsic_dim"] == es["intrinsic_dim_twonn"] and doc["basis"]["size"] == 8)
    check("explore, stats, dependencies and FDs are verbatim",
          all(doc[k] == eng[k] for k in ("explore", "stats", "dependencies", "functional_dependencies",
                                         "linear_derivability")))
    pair = next(p for p in doc["orthogonality"]["pairs"] if {p["a"], p["b"]} == {"qty", "weight_kg"})
    epair = next(p for p in eng["dependencies"]["pairs"] if {p["a"], p["b"]} == {"qty", "weight_kg"})
    check("authored orthogonality pairs receive the script's measures",
          pair.get("nmi") == epair["nmi"] and pair.get("pearson") == epair["pearson"])
    check("typing records the script's type beside the analyst's",
          next(t for t in doc["typing"] if t["column"] == "zip")["script_type"] == "ordinal")
    check("the script's markers follow the author's", doc["markers"][0].startswith("OK: 15 columns")
          and all(m in doc["markers"] for m in eng["markers"]))
    check("the source names every column and the A-box size",
          doc["source"]["column_names"] == eng["source"]["columns"] and doc["source"]["abox_rows"] == 200)
    check("from_run does not travel into the layer", "from_run" not in doc)


def _check_block(base, out):
    html_in, html_out = base.read_text(encoding="utf-8"), out.read_text(encoding="utf-8")
    idx = html_in.rfind("</body>")
    check("the output is a byte superset of the input",
          html_out.startswith(html_in[:idx]) and html_out.endswith(html_in[idx:]))
    check("exactly one geometry layer block",
          html_out.count("@LAYER:start geometry") == 1 and html_out.count("@LAYER:end geometry") == 1)
    check("the block carries the render script, the style and a noscript fallback",
          'id="layer-geometry-render"' in html_out and 'id="layer-geometry-style"' in html_out
          and "layer-geometry-noscript" in html_out)
    return layer_doc(html_out)


def _check_rewrites(work, rd, base, out):
    """Reproducible, create-or-update, and never in place."""
    again = work / "orders.again.html"
    rc, log = apply_layer(base, rd / "geometry-layer.json", again)
    check("a fixed --produced-at reproduces the output byte for byte",
          rc == 0 and again.read_bytes() == out.read_bytes())
    twice = work / "orders.twice.html"
    rc, log = apply_layer(out, rd / "geometry-layer.json", twice)
    check("re-applying replaces the layer (create-or-update) and says so",
          rc == 0 and "replaced" in log and twice.read_text(encoding="utf-8").count("@LAYER:start geometry") == 1,
          tail(log))
    rc, log = apply_layer(out, rd / "geometry-layer.json", out)
    check("--out equal to the input is refused", rc == 2 and "must differ" in log, tail(log))


def stage_apply(work, rd):
    base = work / "orders.base.html"
    out = work / "orders.domain.html"
    before, cache = sha(base), SCRIPTS / "__pycache__"
    had_cache = cache.exists()
    rc, log = apply_layer(base, rd / "geometry-layer.json", out)
    check("apply_geometry_layer.py exits 0 on the orders layer", rc == 0 and out.is_file(), tail(log, 8))
    if rc != 0:
        return None
    check("the summary line counts the provenance", "5 derivation(s)" in log and
          "2 cycle(s)" in log and "chosen 'late'" in log, tail(log, 3))
    check("the input model is never modified", sha(base) == before)
    check("the split modules leave no bytecode beside the scripts", had_cache or not cache.exists())
    doc = _check_block(base, out)
    check("layer data parses with the contract's schema", doc and doc.get("schema") == "dataset-forge/geometry@1")
    if doc:
        _check_spliced(doc, json.loads((rd / "geometry.json").read_text(encoding="utf-8")))
    _check_rewrites(work, rd, base, out)
    return out


def stage_validate_and_smoke(out):
    if not check("domain-forge validate_model.py is in the tree", VALIDATE.is_file(), str(VALIDATE)):
        return
    rc, log = validate_model(out)
    check("validate_model.py exits 0 on the layered model (13-16 included)", rc == 0,
          " | ".join(ln for ln in log.splitlines() if "FAIL" in ln or "passed" in ln)[:600])
    check("invariants 13-16 ran on the geometry layer",
          all(re.search(rf"PASS\s+invariant {n}\b[^\n]*geometry|PASS\s+invariant {n}:[^\n]*1 ", log)
              for n in (13, 14, 15)) and "invariant 16: 1 layer(s)" in log, tail(log, 6))
    rc, log = run([SMOKE, out, "--strict"])
    check("smoke_geometry.py --strict mounts the four views and moves them", rc == 0, tail(log, 6))


def _explorer_variant(work, name, good, bad):
    """A copy of the shipped explorer with `good` replaced by `bad` (None when `good` is gone)."""
    js = RENDER.read_text(encoding="utf-8")
    if not check(f"the render script carries {good!r}", good in js):
        return None
    path = work / name
    path.write_text(js.replace(good, bad), encoding="utf-8")
    return path


def stage_smoke_has_teeth(work, rd):
    """The smoke test is held to a defect it must see: a copy of the explorer whose "make this the
    basis member" button changes the state without redrawing fails --strict on the re-orientation."""
    good = "state.activeOrientation[cy.id] = target; redraw();"
    broken_js = _explorer_variant(work, "geometry-render.no-redraw.js", good,
                                  "state.activeOrientation[cy.id] = target;")
    if broken_js is None:
        return
    broken = work / "orders.broken-render.html"
    rc, log = apply_layer(work / "orders.base.html", rd / "geometry-layer.json", broken, ["--render", broken_js])
    if not check("a layer with the defective explorer is written", rc == 0, tail(log)):
        return
    rc, log = run([SMOKE, broken, "--strict"])
    check("smoke_geometry.py --strict fails when re-orienting does not update the basis",
          rc == 1 and "FAIL full:reorient-basis" in log and "FAIL no-explore:reorient-basis" in log, tail(log, 4))


def stage_retyped(work, rd):
    """The analyst's typing decides an unstated role in the explorer too: on the retyped variant
    (order_id id→numeric in the basis, customer_id numeric→id outside it) the explorer's basis is
    the layer's, and a copy of the explorer that reads the script's type first is caught."""
    data = rd / "retyped.json"
    data.write_text(json.dumps(mutate(json.loads(FIXTURE.read_text(encoding="utf-8")), retyped)),
                    encoding="utf-8")
    out = work / "orders.retyped.html"
    rc, log = apply_layer(work / "orders.base.html", data, out)
    if not check("the retyped layer is written", rc == 0, errors(log) or tail(log)):
        return
    rc, log = run([SMOKE, out, "--strict"])
    check("smoke_geometry.py --strict: the explorer's basis follows the analyst's retyping", rc == 0,
          " | ".join(ln for ln in log.splitlines() if "FAIL" in ln)[:400] or tail(log, 3))
    old_js = _explorer_variant(work, "geometry-render.script-type-first.js", "var t = ftype(c);",
                               "var t = stype(c) || ftype(c);")
    if old_js is None:
        return
    stale = work / "orders.retyped-script-type-first.html"
    rc, log = apply_layer(work / "orders.base.html", data, stale, ["--render", old_js])
    rc, log = run([SMOKE, stale, "--strict"]) if rc == 0 else (rc, log)
    check("smoke_geometry.py --strict fails on an explorer that types by the script first",
          rc == 1 and "FAIL full:report-basis" in log, tail(log, 3))


# ------------------------------------------------------------------ negatives
def _check_only(path):
    return run([APPLY, "--data", path, "--check-only"])


def stage_negatives(work, rd):
    base_doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for i, (name, needle, fn) in enumerate(NEGATIVES):
        p = rd / f"neg-{i:02d}.json"
        p.write_text(json.dumps(mutate(base_doc, fn)), encoding="utf-8")
        rc, log = _check_only(p)
        check(f"refuses {name}", rc == 2 and needle in log and "Traceback" not in log,
              f"exit {rc}; " + errors(log))
    for i, (name, fn) in enumerate(POSITIVES):
        p = rd / f"pos-{i:02d}.json"
        p.write_text(json.dumps(mutate(base_doc, fn)), encoding="utf-8")
        rc, log = _check_only(p)
        check(name, rc == 0, errors(log))
    # the unattended hypothesis is accepted with the WARN that nobody chose it, never silently
    p = rd / "unattended.json"
    p.write_text(json.dumps(mutate(base_doc, unattended_abstained)), encoding="utf-8")
    rc, log = _check_only(p)
    check("an unattended hypothesis is accepted with a WARN that nobody chose it",
          rc == 0 and "WARN:" in log and "nobody chose it" in log, tail(log))
    # the basis is set against TwoNN by its numeric members only (orders: 5 of its 8 — zip is
    # nominal, order_date a date, note text — against 7.1), never by its whole size
    p = rd / "summary.json"
    p.write_text(json.dumps(base_doc), encoding="utf-8")
    rc, log = _check_only(p)
    check("the summary sets the numeric basis members, not the whole basis, against the intrinsic dimension",
          rc == 0 and "basis 8 (5 numeric against intrinsic dimension" in log, tail(log))
    out = work / "refused.html"
    rc, log = apply_layer(work / "orders.base.html", rd / "neg-00.json", out)
    check("a refused document writes no output", rc == 2 and not out.exists(), tail(log))
    p = rd / "no-run.json"
    p.write_text(json.dumps(mutate(base_doc, lambda d: d.__setitem__("from_run", "missing/geometry.json"))),
                 encoding="utf-8")
    rc, log = _check_only(p)
    check("a from_run that does not exist is refused", rc == 2 and "does not exist" in log, tail(log))


# ------------------------------------------------------------------ environment and assets
def stage_env():
    if not check("vega env_check.py is in the tree", ENV_CHECK.is_file(), str(ENV_CHECK)):
        return
    rc, out = run([ENV_CHECK, "--env", "--json", "--skills-root", ROOT])
    try:
        rep = json.loads(out[out.index("{"):])
        rep = rep.get("environment", rep)
    except ValueError:
        check("env_check.py --env --json parses", False, tail(out))
        return
    check("env_check.py --env reports dataset-forge installed",
          (rep.get("skills_installed") or {}).get("dataset-forge") is True)
    check("env_check.py --env reports dataset-forge complete (nothing specified but unbuilt)",
          "dataset-forge" not in (rep.get("skills_specified_but_unbuilt") or {}),
          json.dumps((rep.get("skills_specified_but_unbuilt") or {}).get("dataset-forge")))
    lane = next((v for k, v in (rep.get("lanes_open") or {}).items() if k.startswith("dataset")), None)
    print(f"  info  the dataset lane is {'open' if lane else 'closed'} "
          "(it also needs data-lens, dataset-shaper and their libraries)")


def stage_assets():
    js = RENDER.read_text(encoding="utf-8")
    check("the render script is at most 80 KB (contract §6)", len(js.encode("utf-8")) <= 80 * 1024,
          f"{len(js.encode('utf-8'))} bytes")
    check("the render script makes no network call and loads nothing",
          not re.search(r"\bfetch\(|XMLHttpRequest|import\(|\.src\s*=|<script|WebSocket", js))
    check("the render script draws SVG, never canvas", "canvas" not in js.lower())
    check("the render script cannot close its own <script> or forge a layer marker",
          "</script" not in js.lower() and "@LAYER:" not in js)
    check("the render script reads only its own data script",
          set(re.findall(r"getElementById\('([^']+)'\)", js)) == {"layer-geometry-data"})
    css = re.sub(r"/\*[\s\S]*?\*/", "", STYLE.read_text(encoding="utf-8"))
    css = re.sub(r"@media[^{]*\{", "", css)
    sels = [s.strip() for block in re.findall(r"([^{}]+)\{[^{}]*\}", css) for s in block.split(",")]
    loose = [s for s in sels if s and not s.startswith(".layer-geometry")]
    check("every CSS rule is scoped under .layer-geometry", not loose, str(loose[:5]))


def stage_orders(work):
    print("the orders run")
    rd = build(work)
    if rd is None:
        return
    out = stage_apply(work, rd)
    if out is not None:
        stage_validate_and_smoke(out)
        stage_smoke_has_teeth(work, rd)
        stage_retyped(work, rd)
        print("the shaper reads the layer")
        stage_shaper(work, out)
    print("the validator's refusals")
    stage_negatives(work, rd)


def run_all(work):
    print("environment")
    stage_env()
    print("assets")
    stage_assets()
    stage_orders(work)
    print("the lane's other geometry layer")
    stage_sensors(work)
    print("a generated file with every column kind")
    stage_messy(work)


def main():
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if ARGS:
        work = Path(ARGS[0]).resolve()
        work.mkdir(parents=True, exist_ok=True)
        run_all(work)
    else:
        with tempfile.TemporaryDirectory(prefix="geometry-layer-") as tmp:
            run_all(Path(tmp))
    failed = [n for n, ok in results if not ok]
    print(f"{'ERROR' if failed else 'OK'}: {len(results) - len(failed)}/{len(results)} checks passed"
          + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
