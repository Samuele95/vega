#!/usr/bin/env python3
"""
env_check.py — the two mechanical halves of /vega's first two steps, as one command.

    python3 env_check.py --env                 # the ledger's Environment line
    python3 env_check.py --scan PATH [--json]   # the project's sources, their digests, and,
                                                # for a model, its layers and source kind

`--env` reports what the lanes need and whether it is here: the Python libraries the lanes
use, a headless browser, a LaTeX toolchain, which of the fourteen skills (and the three the
conformance gate uses) are installed, and — the check a skill list alone cannot make — whether
each installed skill's scripts actually exist. A skill whose SKILL.md names a script the directory lacks is a specified-but-unbuilt
stage, and the chain map closes the lane at it rather than discovering it mid-run. Two of the
probes read another skill than the one they close: `vega-show` is unbuilt while the serving
templates of `vega-forge` lack the showcase slot (`serve/routes_showcase.py`, which serves a
showcase pinned by `SHOWCASE.json`) or `vega-pack`'s catalog lacks the record it writes as
`MANIFEST.serve.showcase_slot` — a page the bundle's own service cannot serve is no page.

A skill can also be **degraded**: built, and usable, but without a capability a later stage
would use. Today one probe: a `document-project` whose SKILL.md has no label-role map
(`docs/label-roles.json`) still writes the report, but `vega-show` can then only propose report
roles from the `.aux` or link the stage briefs. A degraded skill closes no lane and does not
change the exit code; it is reported (`<skill> DEGRADED (<why>)` on the line, a `WARN` below it)
so the ledger records it once and the summary carries it.

`--scan` classifies what is at PATH the way the chain map's *Source kinds* table does, digests
each source so an unchanged one is not re-forged, and for every domain-forge model reads its
`ex:sourceKind` and lists its layers — which is what places a model on its lane. It proposes
nothing: the routing decision, and the reason given for it, stay with the orchestrator.

Exit codes: 0 always for --scan; for --env, 0 when every prerequisite of at least one lane is
present, 1 when none is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS_ROOT = HERE.parent.parent

DATA_EXT = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json", ".jsonl"}
CODE_HINTS = {"package.json", "pyproject.toml", "setup.py", "Cargo.toml", "go.mod", "pom.xml",
              "build.gradle", "Makefile", "CMakeLists.txt", "requirements.txt"}
DB_HINTS = re.compile(r"(schema|migrations?|ddl|models?)\.(sql|py|rb|ts)$|\.sql$", re.I)
NOTEBOOK = {".ipynb"}
PROSE_EXT = {".md", ".txt", ".rst"}
SKILLS = ("spec-analysis", "domain-forge", "dataset-forge", "data-lens", "dataset-shaper",
          "inferred-questions", "model-chat", "blueprint",
          "vega-forge", "vega-train", "vega-eval", "vega-pack", "vega-show", "document-project",
          "architect", "improve", "effective-java")
REQUIRED_SCRIPTS = {
    "domain-forge": ["scripts/apply_layer.py", "scripts/strip_layer.py", "scripts/run_query.py",
                     "scripts/validate_model.py", "assets/template.html"],
    "dataset-forge": ["scripts/geometry.py", "scripts/apply_geometry_layer.py",
                      "scripts/smoke_geometry.py", "assets/geometry-render.js"],
    "data-lens": ["scripts/analysis.py", "scripts/cell.py", "scripts/apply_analysis_layer.py",
                  "scripts/smoke_analysis.py", "scripts/bootstrap_base.py",
                  "assets/analysis-render.js"],
    "dataset-shaper": ["scripts/shape.py", "scripts/verify_shape.py",
                       "scripts/apply_shape_layer.py", "scripts/smoke_shape.py",
                       "assets/shape-render.js"],
    "model-chat": ["scripts/run_query.py"],
    "inferred-questions": ["scripts/apply_layer.py"],
    "vega-forge": ["scripts/compile_spec.py", "scripts/emit_backend.py", "scripts/conformance.py",
                   "scripts/nb_emit.py",
                   "scripts/apply_architecture_layer.py", "scripts/smoke_architecture.py",
                   "scripts/architecture_sample.py", "scripts/audit_quorum.py",
                   "scripts/catalog.json", "scripts/backends/pytorch/emit.py",
                   "scripts/backends/pytorch/templates",
                   "assets/architecture-render.js"],
    "vega-train": ["scripts/plan_ladder.py", "scripts/trainer.py", "scripts/hpo.py",
                   "scripts/apply_training_layer.py", "scripts/smoke_training.py",
                   "assets/training-render.js"],
    "vega-eval": ["scripts/evaluate.py", "scripts/apply_evaluation_layer.py",
                  "scripts/smoke_evaluation.py", "assets/evaluation-render.js"],
    "vega-pack": ["scripts/pack.py", "scripts/model_card.py", "scripts/apply_bundle_layer.py",
                  "scripts/smoke_bundle.py", "assets/bundle-render.js"],
    # R-E2: the page template is D4's, identified by TEMPLATE.json beside its strings keys
    "vega-show": ["scripts/extract_facts.py", "scripts/draft_plan.py", "scripts/resolve_story.py",
                  "scripts/build_showcase.py", "scripts/showcase_check.py",
                  "scripts/apply_show_layer.py", "scripts/smoke_show.py", "scripts/catalog.json",
                  "assets/showcase-render.js", "assets/showcase-layer.css",
                  "assets/page/index.html", "assets/page/TEMPLATE.json",
                  "assets/page/strings.keys.json"],
}
# the showcase slot vega-show needs: the serving template's route module and its manifest name,
# and the vega-pack catalog record pack.py writes as MANIFEST.serve.showcase_slot
SHOWCASE_ROUTES = "vega-forge/scripts/backends/pytorch/templates/serving/serve/routes_showcase.py"
SHOWCASE_MANIFEST = "SHOWCASE.json"
PACK_CATALOG = "vega-pack/scripts/catalog.json"
LABEL_ROLE_MAP = "label-roles.json"
SHOW_UPSTREAM = ("vega-pack", "vega-forge")
LIBS = ("pandas", "numpy", "scipy", "sklearn", "rdflib", "pyarrow", "statsmodels", "matplotlib",
        "geopandas", "pyproj", "h3", "torch", "nbconvert", "optuna", "lightgbm", "fastapi",
        "uvicorn")
TURTLE_RE = re.compile(r'<script\s+id="domain-model"[^>]*>([\s\S]*?)</script>', re.I)
LAYER_RE = re.compile(r"<!--\s*@LAYER:start\s+([A-Za-z0-9_-]+)\s+v\d+([\s\S]*?)-->")


def digest(p: Path):
    h = hashlib.sha256()
    if p.is_file():
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    names = sorted(str(q.relative_to(p)) for q in p.rglob("*") if q.is_file())
    for n in names[:20000]:
        h.update(n.encode("utf-8"))
        try:
            h.update(str((p / n).stat().st_size).encode())
        except OSError:
            pass
    return "sha256:" + h.hexdigest() + " (tree)"


def find_chrome():
    env = os.environ.get("CHROME")
    if env and (Path(env).is_file() or shutil.which(env)):
        return env
    for c in ("chromium", "google-chrome", "chromium-browser", "chrome"):
        w = shutil.which(c)
        if w:
            return w
    return None


def _text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def showcase_slot_missing(skills_root: Path) -> list:
    """What vega-show lacks upstream: the serving template's showcase slot (vega-forge) and the
    record pack writes as MANIFEST.serve.showcase_slot (vega-pack's catalog). Empty when both hold."""
    out = []
    routes = skills_root / SHOWCASE_ROUTES
    if not routes.is_file() or SHOWCASE_MANIFEST not in _text(routes):
        out.append(f"the serving template's showcase slot ({SHOWCASE_ROUTES}, serving "
                   f"{SHOWCASE_MANIFEST}) — the bundle's service cannot serve a showcase")
    try:
        slot = json.loads(_text(skills_root / PACK_CATALOG))["serve_record"]["showcase_slot"]
    except (ValueError, KeyError, TypeError):
        slot = None
    if not slot:
        out.append(f"{PACK_CATALOG} serve_record.showcase_slot (MANIFEST.serve.showcase_slot, "
                   "which /vega-pack --serve writes)")
    return out


def degraded_reasons(skill: str, d: Path) -> list:
    """Capabilities an installed, built skill lacks that a later stage would use."""
    if skill == "document-project" and LABEL_ROLE_MAP not in _text(d / "SKILL.md"):
        return ["no label-role map (docs/label-roles.json): vega-show links the stage briefs "
                "or proposes report roles from the .aux"]
    return []


def libraries() -> dict:
    libs = {}
    for m in LIBS:
        try:
            mod = __import__(m)
            libs[m] = getattr(mod, "__version__", "present")
        except Exception:
            libs[m] = None
    return libs


def skill_dir(skills_root: Path, skill: str) -> Path:
    """The skill's directory: under the skills root, else account-synced (architect,
    effective-java live under ~/.claude/skills/synced/<id>/)."""
    d = skills_root / skill
    if not (d / "SKILL.md").is_file():
        for cand in sorted(Path.home().glob(".claude/skills/synced/*/" + skill)):
            if (cand / "SKILL.md").is_file():
                return cand
    return d


def missing_parts(skill: str, d: Path, skills_root: Path) -> list:
    """What an installed skill's SKILL.md promises and its directory (or, for vega-show, the
    skills it is served through) lacks: a non-empty list makes it specified-but-unbuilt."""
    missing = [f for f in REQUIRED_SCRIPTS.get(skill, []) if not (d / f).exists()]
    prompt = _text(d / "SKILL.md")
    if skill == "architect" and "--baseline" not in prompt:
        missing.append("SKILL.md --baseline (needed by the Vega conformance gate)")
    if skill == "document-project" and "vega <lane>" not in prompt:
        missing.append("SKILL.md Vega mode (`/document-project vega <lane>` — the patched skill)")
    if skill == "blueprint" and "docs/blueprint-brief.md" not in prompt:
        missing.append("SKILL.md stage brief (the patched skill writes docs/blueprint-brief.md)")
    if skill == "vega-show":
        missing += showcase_slot_missing(skills_root)
    if skill == "dataset-shaper" and (d / "scripts/shape.py").exists():
        if "--transform-only" not in _text(d / "scripts/shape.py"):
            missing.append("scripts/shape.py --transform-only (needed by vega-pack)")
    return missing


def _ready(skills, installed, unbuilt) -> bool:
    return all(installed.get(s) for s in skills) and not (set(skills) & set(unbuilt))


def lanes_open(installed, unbuilt, libs, latex) -> dict:
    network = ("vega-forge", "vega-train", "vega-eval", "vega-pack", "dataset-shaper")
    return {
        "software (spec → forge-prose → blueprint)":
            all(installed.get(s) for s in ("spec-analysis", "domain-forge", "blueprint"))
            and "domain-forge" not in unbuilt,
        "dataset (forge-data → lens → shape → blueprint)":
            _ready(("dataset-forge", "data-lens", "dataset-shaper"), installed, unbuilt)
            and all(libs.get(m) for m in ("pandas", "numpy", "scipy", "sklearn")),
        "network (net → train → eval → pack)":
            _ready(network, installed, unbuilt) and bool(libs.get("torch")),
        "document (document-project)":
            _ready(("document-project",), installed, unbuilt) and bool(latex),
        # the post-report presentation stage: it reads a bundle and is served by the bundle's own
        # service; torch for the preflight's predict.py --check (PF3); no LaTeX (it reads the
        # compiled report, never compiles one)
        "showcase (show)":
            _ready(("vega-show",) + SHOW_UPSTREAM, installed, unbuilt) and bool(libs.get("torch")),
    }


def gpu_name(libs):
    if not libs.get("torch"):
        return None
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return None
    return None


def env_report(skills_root):
    libs = libraries()
    installed, unbuilt, degraded = {}, {}, {}
    for s in SKILLS:
        d = skill_dir(skills_root, s)
        installed[s] = (d / "SKILL.md").is_file()
        if not installed[s]:
            continue
        missing = missing_parts(s, d, skills_root)
        if missing:
            unbuilt[s] = missing
        why = degraded_reasons(s, d)
        if why:
            degraded[s] = why
    latex = shutil.which("pdflatex") or shutil.which("lualatex") or shutil.which("xelatex")
    return {"python": sys.version.split()[0], "libraries": libs, "gpu": gpu_name(libs),
            "headless_browser": find_chrome(), "latex": latex,
            "skills_installed": installed, "skills_specified_but_unbuilt": unbuilt,
            "skills_degraded": degraded, "lanes_open": lanes_open(installed, unbuilt, libs, latex),
            "skills_root": str(skills_root)}


def env_line(r):
    libs = " · ".join(f"{k} {v}" for k, v in r["libraries"].items() if v)
    absent = ", ".join(k for k, v in r["libraries"].items() if not v)
    parts = [f"python {r['python']}", libs]
    if absent:
        parts.append(f"absent: {absent}")
    parts.append("chromium " + ("present" if r["headless_browser"] else "absent"))
    parts.append("latex " + ("present" if r["latex"] else "absent"))
    parts.append("gpu " + (r.get("gpu") or "absent"))
    miss = [k for k, v in r["skills_installed"].items() if not v]
    parts.append("skills " + (f"all required skills installed" if not miss else f"missing: {', '.join(miss)}"))
    for s, files in r["skills_specified_but_unbuilt"].items():
        parts.append(f"{s} UNBUILT ({', '.join(files)})")
    for s, why in r.get("skills_degraded", {}).items():
        parts.append(f"{s} DEGRADED ({'; '.join(why)})")
    return " · ".join(p for p in parts if p)


def read_model(p: Path):
    try:
        html = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if 'id="domain-model"' not in html:
        return None
    m = TURTLE_RE.search(html)
    ttl = m.group(1) if m else ""
    kind = "dataset" if 'ex:sourceKind "dataset"' in ttl or 'ex:sourceKind "dataset"' in html \
        else "software-domain"
    layers = []
    for name, head in LAYER_RE.findall(html):
        prod = re.search(r"produced-by:\s*(\S+)", head)
        at = re.search(r"produced-at:\s*(\S+)", head)
        layers.append({"name": name, "produced_by": prod.group(1) if prod else None,
                       "produced_at": at.group(1) if at else None})
    return {"source_kind": kind, "layers": layers}


def classify(root: Path, max_files=20000):
    root = root.resolve()
    files = [p for p in root.rglob("*") if p.is_file()][:max_files]
    hidden = {".git", "__pycache__", "node_modules", ".venv", ".claude"}
    files = [p for p in files if not any(part in hidden for part in p.parts)]
    sources = []
    models = [p for p in files if p.suffix.lower() in (".html", ".htm")]
    for p in models:
        info = read_model(p)
        if info:
            sources.append({"kind": "model", "path": str(p.relative_to(root)),
                            "digest": digest(p), **info})
    data = [p for p in files if p.suffix.lower() in DATA_EXT and p.stat().st_size > 512]
    for p in data:
        sources.append({"kind": "dataset", "path": str(p.relative_to(root)), "digest": digest(p),
                        "bytes": p.stat().st_size})
    notebooks = [p for p in files if p.suffix.lower() in NOTEBOOK]
    sql = [p for p in files if DB_HINTS.search(p.name)]
    codeish = [p for p in files if p.name in CODE_HINTS]
    code_ext = {".py", ".js", ".ts", ".java", ".go", ".rs", ".rb", ".c", ".cpp", ".cs", ".php"}
    code = [p for p in files if p.suffix.lower() in code_ext]
    if codeish or code:
        kind = "codebase"
        if notebooks or (data and code):
            kind = "data-project"
        sources.append({"kind": kind, "path": ".", "digest": digest(root),
                        "files": len(code) + len(notebooks),
                        "signals": sorted({p.name for p in codeish})[:8]})
    if sql and not any(s["kind"] in ("codebase", "data-project") for s in sources):
        sources.append({"kind": "database", "path": ".", "digest": digest(root),
                        "schema_files": [str(p.relative_to(root)) for p in sql[:12]]})
    elif sql:
        for s in sources:
            if s["kind"] in ("codebase", "data-project"):
                s["schema_files"] = [str(p.relative_to(root)) for p in sql[:12]]
    prose = [p for p in files if p.suffix.lower() in PROSE_EXT and p.stat().st_size > 200]
    spec_html = [s for s in sources if s["kind"] == "model" and False]
    for p in models:
        if not read_model(p) and "spec-analysis" in p.name:
            sources.append({"kind": "prose", "path": str(p.relative_to(root)),
                            "digest": digest(p), "note": "a /spec-analysis artifact"})
    for p in prose[:20]:
        sources.append({"kind": "prose", "path": str(p.relative_to(root)), "digest": digest(p),
                        "bytes": p.stat().st_size})
    mem = root / ".claude" / "domain-forge-memory.md"
    ledger = root / ".claude" / "vega-ledger.md"
    if not ledger.is_file() and (root / ".claude" / "noctua-ledger.md").is_file():
        ledger = root / ".claude" / "noctua-ledger.md"  # the former name; /vega renames it
    memory = {}
    if mem.is_file():
        txt = mem.read_text(encoding="utf-8", errors="replace")
        for sec in ("Modeling stances", "Dataset stances", "Analysis stances", "Shaping stances",
                    "Applied", "Declined", "Out-of-scope"):
            body = re.search(r"^##\s+" + re.escape(sec) + r"\s*$([\s\S]*?)(?=^##\s|\Z)",
                             txt, re.M)
            lines = [l.strip() for l in (body.group(1).splitlines() if body else [])
                     if l.strip() and not l.strip().startswith("<!--")
                     and not l.strip().startswith("-->")]
            memory[sec] = lines
    return {"root": str(root), "sources": sources,
            "memory_present": mem.is_file(), "memory": memory,
            "ledger_present": ledger.is_file()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--scan", default=None)
    ap.add_argument("--skills-root", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not a.env and not a.scan:
        ap.error("one of --env or --scan is required")
    out = {}
    code = 0
    if a.env:
        r = env_report(Path(a.skills_root) if a.skills_root else SKILLS_ROOT)
        out["environment"] = r
        if not a.json:
            print(env_line(r))
            for lane, open_ in r["lanes_open"].items():
                print(f"  {'OK  ' if open_ else 'WARN'} lane {lane}: "
                      f"{'open' if open_ else 'closed'}")
            for s, files in r["skills_specified_but_unbuilt"].items():
                print(f"  ERROR {s} is installed but unbuilt: missing {', '.join(files)}")
            for s, why in r["skills_degraded"].items():
                print(f"  WARN {s} is degraded: {'; '.join(why)}")
        code = 0 if any(r["lanes_open"].values()) else 1
    if a.scan:
        s = classify(Path(a.scan))
        out["scan"] = s
        if not a.json:
            print(f"root: {s['root']}")
            for src in s["sources"]:
                extra = ""
                if src["kind"] == "model":
                    extra = (f" — sourceKind {src['source_kind']}, layers "
                             f"{[l['name'] for l in src['layers']] or 'none'}")
                elif src.get("files"):
                    extra = f" — {src['files']} code/notebook file(s) {src.get('signals') or ''}"
                elif src.get("schema_files"):
                    extra = f" — schema {src['schema_files'][:3]}"
                print(f"  {src['kind']:<12} {src['path']}{extra}")
            print(f"  memory: {'present' if s['memory_present'] else 'absent'}"
                  + (f" ({sum(len(v) for v in s['memory'].values())} recorded decisions)"
                     if s["memory_present"] else "")
                  + f" · ledger: {'present' if s['ledger_present'] else 'absent'}")
    if a.json:
        print(json.dumps(out, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
