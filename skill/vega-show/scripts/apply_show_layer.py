#!/usr/bin/env python3
"""
apply_show_layer.py — validate the `showcase` layer's data and write the /vega-show `showcase`
layer into the model (contract §8, gates LY1–LY7; D5 §7.9).

    python3 apply_show_layer.py MODEL.pack.html --data <bundle>/showcase/.work/showcase-layer.json
                                --out OUT.html [--domain-forge-dir D] [--json]

The authoring file carries only the judgement — `readings.abstract` and the run's `markers` —
and `"from_showcase": "<bundle>/showcase"`. Everything else is spliced from the showcase
(`show_layer_data.ShowLayer`): `SHOWCASE.json` verbatim, the story summary (headline and lead
rendered with their values, the section outline, input groups, class roles, motif, polarity),
the bindings (every fact key with its value, record and uses), the report binding, the build
check and the served walk (read from the showcase's `.work/checks.json` and `.smoke/smoke.json`,
never from the markers), the stage's markers and the usage (the serve command, the address, the
container line).

It refuses (exit 2) a layer whose build check did not pass or predates the build (LY1); whose
served walk failed, or did not run without `WARN: served page not checked` (LY2); whose
abstract is missing, short, carries a digit or a placeholder (LY3); whose brief-link showcase
lacks `WARN: no report` (LY4); whose `SHOWCASE.json` does not pin the bundle's current
`MANIFEST.json` (LY5); whose data or render exceeds its cap (LY6); or whose `--out` is the input
(LY7). The block is written ONLY by `domain-forge/scripts/apply_layer.py` (imported, else its
CLI), with a `<noscript>` carrying the headline, the report status and the serve command, as
`produced-by: /vega-show`. The input is never modified.

Exit codes: 0 wrote OUT; 2 refused or an input could not be read; 3 the platform writer is
missing or failed.
"""
from __future__ import annotations

import argparse
import html as _html
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from show_layer_data import ShowLayer  # noqa: E402
from vega_show_core import CATALOG  # noqa: E402

SKILL_DIR = HERE.parent
ASSETS = SKILL_DIR / "assets"
RENDER_JS = ASSETS / "showcase-render.js"
STYLE_CSS = ASSETS / "showcase-layer.css"
LAYER = CATALOG["layer"]["name"]
MAX_RENDER = CATALOG["layer"]["max_render_kb"] * 1024
PRODUCED_BY = "/" + CATALOG["stage"]["skill"]
_BLOCK_RE = re.compile(r"@LAYER:start\s+" + LAYER + r"\b")


def _esc(value) -> str:
    return _html.escape("" if value is None else str(value), quote=True)


def noscript_html(doc: dict) -> str:
    """What a reader without scripts sees: the headline, the report status, the serve line."""
    story, usage, report = doc["story"], doc["usage"], doc["report"]
    checks = doc["checks"]
    return ("<section class=\"layer-showcase-noscript\">"
            f"<h3>Showcase — {_esc(story.get('headline'))}</h3>"
            f"<p>{_esc(story.get('lead'))}</p>"
            f"<p>Report: <strong>{_esc(report.get('status'))}</strong>. Build check: "
            f"{_esc((checks.get('build') or {}).get('status'))}; served walk: "
            f"{_esc((checks.get('served') or {}).get('status'))}.</p>"
            f"<p>Open it from the bundle directory:</p><pre>{_esc(usage.get('serve'))}</pre>"
            f"<p>then {_esc(usage.get('url'))}</p>"
            f"<p>{_esc((doc.get('readings') or {}).get('abstract'))}</p>"
            "</section>")


def _writer_module(writer: Path):
    spec = importlib.util.spec_from_file_location("df_apply_layer", str(writer))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def platform_apply(writer: Path, model: Path, parts: dict) -> tuple[str | None, str]:
    """The layer written by domain-forge's apply_layer: imported, else its CLI."""
    notes = []
    sys.dont_write_bytecode = True
    try:
        result = _writer_module(writer).apply_layer(
            parts["html"], LAYER, parts["data"], parts["render"], style_css=parts["style"],
            data_type="application/json", produced_by=PRODUCED_BY, noscript_html=parts["noscript"])
        if isinstance(result, str) and _BLOCK_RE.search(result):
            return result, "platform writer (import)"
        notes.append("import path returned no layer block")
    except Exception as exc:  # noqa: BLE001 — any failure falls back to the CLI
        notes.append(f"import path failed: {exc}")
    return _platform_cli(writer, model, parts, notes)


def _platform_cli(writer: Path, model: Path, parts: dict, notes: list) -> tuple[str | None, str]:
    with tempfile.TemporaryDirectory() as tmp:
        files = {}
        for name, key in (("layer.json", "data"), ("render.js", "render"), ("style.css", "style"),
                          ("noscript.html", "noscript")):
            files[key] = Path(tmp) / name
            files[key].write_text(parts[key], encoding="utf-8")
        out = Path(tmp) / "out.html"
        proc = subprocess.run(
            [sys.executable, str(writer), str(model), "--layer", LAYER, "--data", str(files["data"]),
             "--render", str(files["render"]), "--style", str(files["style"]),
             "--data-type", "application/json", "--produced-by", PRODUCED_BY,
             "--noscript", str(files["noscript"]), "--out", str(out)],
            capture_output=True, text=True, timeout=120, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if proc.returncode == 0 and out.is_file():
            return out.read_text(encoding="utf-8"), "platform writer (CLI)"
        notes.append(f"CLI path failed (exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()[:300]}")
    return None, "; ".join(notes)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="the model page (<stem>.pack.html; never modified)")
    ap.add_argument("--data", required=True, help="<bundle>/showcase/.work/showcase-layer.json")
    ap.add_argument("--out", required=True, help="<stem>.pack.show.html")
    ap.add_argument("--domain-forge-dir", help="where domain-forge lives (default: beside this skill, "
                                                "or $DOMAIN_FORGE_DIR)")
    ap.add_argument("--json", action="store_true", help="print the layer data instead of the summary")
    ap.add_argument("--render", default=str(RENDER_JS), help=argparse.SUPPRESS)
    ap.add_argument("--style", default=str(STYLE_CSS), help=argparse.SUPPRESS)
    return ap


def _refuse(problems) -> int:
    for p in problems:
        print(f"ERROR: {p.rule} {p.where}: {p.message}")
    return 2


def _assets(a) -> tuple[str, str] | str:
    try:
        render = Path(a.render).read_text(encoding="utf-8").rstrip()
        style = Path(a.style).read_text(encoding="utf-8").rstrip()
    except OSError as exc:
        return f"asset missing: {exc}"
    size = len(render.encode("utf-8"))
    if size > MAX_RENDER:
        return f"LY6 render: the render script is {size} bytes; the contract caps it at {MAX_RENDER // 1024} KB"
    return render, style


def _layer_data(a) -> tuple[dict, str] | int:
    """The validated layer data and its text, or the exit code of a refusal."""
    try:
        layer = ShowLayer.load(a.data)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read the layer's authoring file: {exc}")
        return 2
    problems = layer.problems()
    if problems:
        return _refuse(problems)
    doc = layer.document()
    missing = layer.missing_keys(doc)
    if missing:
        print(f"ERROR: the layer data lacks {', '.join(missing)} (contract §8 layer.required)")
        return 2
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    too_big = layer.size_problems(text)
    return _refuse(too_big) if too_big else (doc, text)


def _writer(a) -> Path:
    root = a.domain_forge_dir or os.environ.get("DOMAIN_FORGE_DIR") or str(SKILL_DIR.parent / "domain-forge")
    return Path(root) / "scripts" / "apply_layer.py"


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    model, out = Path(a.model), Path(a.out)
    if model.resolve() == out.resolve():
        print("ERROR: LY7 --out: --out must differ from the input model (the input is never modified)")
        return 2
    assets = _assets(a)
    if isinstance(assets, str):
        print(f"ERROR: {assets}")
        return 2
    built = _layer_data(a)
    if isinstance(built, int):
        return built
    doc, data_text = built
    if a.json:
        print(json.dumps(doc, ensure_ascii=False, indent=1))
    try:
        html = model.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read the model page: {exc}")
        return 2
    return _write(a, model, out, {"html": html, "data": data_text, "render": assets[0], "style": assets[1],
                                  "noscript": noscript_html(doc)}, doc)


def _write(a, model: Path, out: Path, parts: dict, doc: dict) -> int:
    writer = _writer(a)
    if not writer.is_file():
        print(f"ERROR: the platform writer is not at {writer}; the layer block is written only by "
              "domain-forge/scripts/apply_layer.py")
        return 3
    result, how = platform_apply(writer, model, parts)
    if result is None:
        print(f"ERROR: the platform writer did not produce a layer block: {how}")
        return 3
    if _BLOCK_RE.search(parts["html"]):
        print("WARN: the input already carried a showcase layer; it was replaced")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result, encoding="utf-8")
    sc, served = doc["showcase"], doc["checks"]["served"]
    print(f"OK: showcase layer of {sc['bundle']['slug']}-{sc['bundle']['run']}: report {doc['report']['status']}, "
          f"{len(doc['bindings'])} binding(s), build check {doc['checks']['build']['status']}, "
          f"served walk {served.get('status')}")
    print(f"OK: wrote {out} via {how}; render {len(parts['render'].encode('utf-8'))} bytes, data "
          f"{len(parts['data'].encode('utf-8')) // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
