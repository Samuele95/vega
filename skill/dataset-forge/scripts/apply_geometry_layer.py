#!/usr/bin/env python3
"""
apply_geometry_layer.py — validate a `layer-geometry-data` document and inject the
/dataset-forge `geometry` layer (data + the shipped explorer + style + <noscript>) into a
domain-forge model.

    python3 apply_geometry_layer.py MODEL.html --data LAYER.json --out OUT.html
                                    [--produced-by /dataset-forge] [--produced-at ISO-UTC]
                                    [--domain-forge-dir PATH]
    python3 apply_geometry_layer.py --data LAYER.json --check-only

The analyst authors the JUDGEMENT — the readings, the retypings with their reasons, the
derivations with their provenance triple, the cycles and their orientations, the consequence
blocks, the partition candidates — and never retypes a number. A layer document therefore
carries

    "from_run": "<run-dir>/geometry.json"

(resolved against the document's own directory when relative; a run directory stands for the
geometry.json inside it) and this script splices the script's half in verbatim
(`references/report-contract.md` §1, "numbers come verbatim"):

  * `source` rows, column names, seed and parameters;
  * `columns` (the script's typing per column), and every typing entry's `script_type`;
  * `space` — every number `geometry.py` wrote, plus the contract's short names
    (`near_rank`, `dims_95`, `dims_99`) for the script's long ones; the author's `reading`
    is kept;
  * `basis.intrinsic_dim` and `basis.size`;
  * `dependencies`, `functional_dependencies`, `linear_derivability`, `stats` and the
    explorer's `explore` payload, verbatim;
  * the measures of every authored `orthogonality.pairs` entry, and its `thresholds`;
  * the script's markers after the author's.

A number the author DID write must be the script's number (a rounding of it is accepted):
a layer whose prose and explorer disagree with `geometry.json` is refused.

Validation follows the contract (§1 schema, §2 provenance triple, §3 consequence block,
§5 hand-off, §6 explorer inputs). Exit 2 on any `ERROR:`; `WARN:` does not block. A structural
pass runs first, on the document as written and again after the splice: every container has
its type (`typing`, `derivations`, `cycles`, `disagreements` lists of objects; `partitions`,
`basis`, `space` objects), every column list holds column names (a rule's `body` is a non-empty
list of them, as are a candidate's `features` and `dropped_for_leakage`, a cycle's `members`, an
orientation's `derived`), every identifier is a string. A document it refuses is refused there,
before any other check reads it. The rules that are more than structural, and why each exists:

  * every derivation carries the provenance triple — semantic / symbolic / empirical, each
    confirmed / refuted / untested (semantic is never untested: the model always has a view);
  * `symbolic: confirmed` carries `verified_rows` and `of_rows` from the engine run, equal
    and positive, and never more rows than the A-box sample the script drew; `untested`
    carries no verified count; `refuted` counts fewer verified rows than it ran;
  * a derivation whose channels disagree (a confirmed and a refuted verdict) has a
    `disagreements` entry for its column — a disagreement is never resolved silently;
  * every non-identity column is either a basis member or the head of exactly one kept
    rule (the basis is the set of columns that are the head of no rule), and identities,
    constants and foreign keys stay out of the basis. A column's role is its typing `role`;
    without one, the analyst's `final_type` decides it (`id` is an identity, `constant` is
    degenerate), and the script's type only when the analyst typed nothing — the script's
    types are heuristics, the analyst owns the final typing;
  * every retyping, every derivation, every cycle orientation and every partition candidate
    carries a consequence block (§3) — the explorer shows it when the analyst clicks the
    alternative, so a missing block is a what-if with no content;
  * a derivation that names a `cycle` has a `cycles` entry whose orientations include the
    report's own and at least one alternative, so View B can re-orient it;
  * every partition candidate states its leakage set (`dropped_for_leakage`, possibly
    empty) and never keeps among its features the label, a leakage column, a column of the
    label's own rule, or the determinant of an exact functional dependency on the label
    (every column that derives the label is a leakage, SKILL.md step 4);
  * `partitions.chosen` agrees with `partitions.provenance` (abstained ⇒ no choice,
    single-candidate ⇒ one candidate, user-chosen ⇒ a named candidate).

`cycles` (cited by /dataset-shaper as `geometry:cycles/<id>`, its `orient_cycle` step naming
an orientation by `id`) is the slot for the orientations of a symmetric derivation — a slot
report-contract §1 does not list yet, and the one a derivation's `cycle` needs:

    {"id": "price-qty-total", "members": ["unit_price", "qty", "total"], "reading": "...",
     "orientations": [
       {"id": "total-derived", "derived": ["total"], "consequences": {...}},
       {"id": "unit-price-derived", "derived": ["unit_price"], "consequences": {...},
        "derivations": [{"column": "unit_price", "formula": "unit_price = total / qty",
                         "body": ["total", "qty"]}]}]}

The report's orientation is the one whose `derived` set equals the kept rule heads among
the members; an alternative orientation may state the rules it would use (else each derived
member is drawn from the members that stay in the basis). The author writes `id`, `derived`,
the consequence block and those optional rules; this script completes every orientation with
the keys the shaper's `orient_cycle` / `drop_derived` read — `default` (true on the report's
orientation alone), `basis` (the members that stay) and `rules` (one per derived member: the
orientation's own rule, else the kept derivation on the report's orientation, else the
members that stay) — and refuses an authored `default`, `basis` or `rules` that contradicts
`derived`, so the explorer and the shaper cannot read two different orientations.

The block is written ONLY by the platform writer domain-forge/scripts/apply_layer.py: this
script carries no copy of the block format or the digest. The input model is never modified.
The work lives in the modules beside this script: geometry_layer_shape (the structural pass),
_splice, _checks, _cycles and _partitions (the rules), _write (noscript, data, writer) and
_model (the vocabulary they share).

Exit codes: 0 wrote OUT (or, with --check-only, the document validates); 2 validation or
input error; 3 platform writer missing or failed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True               # the siblings below leave no __pycache__ in the skill
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from geometry_layer_checks import summary, validate  # noqa: E402
from geometry_layer_cycles import complete_cycles  # noqa: E402
from geometry_layer_model import LAYER, SCHEMA  # noqa: E402
from geometry_layer_shape import check_shape  # noqa: E402
from geometry_layer_splice import splice  # noqa: E402
from geometry_layer_write import Block, Stamp, finite, noscript_html, platform_apply, serialize  # noqa: E402

SKILL_DIR = HERE.parent
ASSETS = SKILL_DIR / "assets"
RENDER_JS = ASSETS / "geometry-render.js"
STYLE_CSS = ASSETS / "geometry-layer.css"
RENDER_CAP = 80 * 1024            # report-contract §6
_BLOCK_RE = re.compile(r"[ \t]*<!--\s*@LAYER:start\s+" + LAYER +
                       r"\b[\s\S]*?@LAYER:end\s+" + LAYER + r"\s*-->[ \t]*\n?")


def check_document(doc, out, base_dir):
    """(the spliced document, its ERROR lines): the structural pass on the document as written,
    the splice from geometry.json, then the full validation."""
    errs = check_shape(doc)
    if errs:
        return doc, errs
    doc, errs = splice(doc, out, base_dir)
    return doc, errs + validate(doc, out)


def _writer(a):
    base = a.domain_forge_dir or os.environ.get("DOMAIN_FORGE_DIR")
    df_dir = Path(base) if base else SKILL_DIR.parent / "domain-forge"
    return df_dir / "scripts" / "apply_layer.py"


def _read_model(a):
    """The input model's text (None under --check-only without one), or an exit code."""
    if not a.model:
        return None
    if a.out and Path(a.model).resolve() == Path(a.out).resolve():
        print("ERROR: --out must differ from the input model (the input is never modified)")
        return 2
    try:
        return Path(a.model).read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: cannot read model {a.model}: {e}")
        return 2


def _read_assets(a):
    """(render script, style), or an exit code."""
    for p in (Path(a.render), Path(a.style)):
        if not p.is_file():
            print(f"ERROR: asset missing: {p}")
            return 2
    render_js = Path(a.render).read_text(encoding="utf-8").rstrip()
    style_css = Path(a.style).read_text(encoding="utf-8").rstrip()
    size = len(render_js.encode("utf-8"))
    if size > RENDER_CAP:
        print(f"ERROR: render script is {size} bytes; contract §6 caps it at {RENDER_CAP // 1024} KB")
        return 2
    return render_js, style_css


def _read_inputs(a):
    """(html, doc, render_js, style_css) or an exit code."""
    html = _read_model(a)
    if isinstance(html, int):
        return html
    try:
        doc = json.loads(Path(a.data).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: cannot read layer data {a.data}: {e}")
        return 2
    if a.check_only:
        return html, doc, None, None
    assets = _read_assets(a)
    return assets if isinstance(assets, int) else (html, doc, *assets)


def _write(a, html, doc, assets):
    """Complete, serialize and hand the block to the platform writer; the exit code."""
    render_js, style_css = assets
    complete_cycles(doc)
    nonfinite = []
    doc = finite(doc, nonfinite)
    if nonfinite:
        print(f"WARN: {len(nonfinite)} non-finite number(s) from geometry.json are written as null")
    writer = _writer(a)
    if not writer.is_file():
        print(f"ERROR: platform writer not found at {writer}; /dataset-forge writes its layer only "
              "through domain-forge's apply_layer.py (pass --domain-forge-dir, or set "
              "DOMAIN_FORGE_DIR, when the skill lives elsewhere)")
        return 3
    block = Block(serialize(doc), render_js, style_css, noscript_html(doc))
    stamp = Stamp(Path(a.model), html, a.produced_by, a.produced_at)
    with tempfile.TemporaryDirectory() as td:
        result, how = platform_apply(writer, block, stamp, td)
    if result is None:
        print(f"ERROR: platform writer {writer} failed ({how})")
        return 3
    if _BLOCK_RE.search(html):
        print("WARN: input already carried a geometry layer; it was replaced")
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"OK: wrote {out_path} via {how}")
    print(f"OK: render script {len(render_js.encode('utf-8'))} bytes, data "
          f"{len(block.data.encode('utf-8'))} bytes, output {len(result.encode('utf-8'))} bytes")
    return 0


def _parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="input MODEL.html (never modified)")
    ap.add_argument("--data", required=True, help="layer-geometry-data JSON document")
    ap.add_argument("--out", help="output HTML path (must differ from MODEL)")
    ap.add_argument("--check-only", action="store_true",
                    help="splice and validate the document, write nothing")
    ap.add_argument("--produced-by", default="/dataset-forge")
    ap.add_argument("--produced-at", default=None, help="UTC ISO stamp (default: now)")
    ap.add_argument("--domain-forge-dir", default=None)
    ap.add_argument("--render", default=str(RENDER_JS), help=argparse.SUPPRESS)
    ap.add_argument("--style", default=str(STYLE_CSS), help=argparse.SUPPRESS)
    return ap


def main(argv=None):
    a = _parser().parse_args(argv)
    if not a.check_only and (not a.model or not a.out):
        print("ERROR: MODEL and --out are required unless --check-only")
        return 2
    got = _read_inputs(a)
    if isinstance(got, int):
        return got
    html, doc, render_js, style_css = got
    doc, errs = check_document(doc, print, Path(a.data).resolve().parent)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        print(f"ERROR: {a.data} does not validate against {SCHEMA} ({len(errs)} error(s))")
        return 2
    print(f"OK: {a.data} validates against {SCHEMA}: {summary(doc)}")
    if a.check_only:
        return 0
    return _write(a, html, doc, (render_js, style_css))


if __name__ == "__main__":
    sys.exit(main())
