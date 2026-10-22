#!/usr/bin/env python3
"""
cic_fixture.py — the CIC R7 bundle's showcase documents, for the parity server: R7's story@1 and
records converted into a `vega/story@2` + `vega/page@1` pair (and the demo sample's attribution
as `vega/attribution@2`), until R2 builds the real CIC showcase with /vega-show.

    python3 cic_fixture.py --write DIR [--bundle R7]            the pair (and data/, report/) in DIR
    python3 cic_fixture.py --parity [--strict] [--keep]        the W01 parity probe on R7, through
        [--template DIR] [--upstream URL | --bundle R7]         the template, with the pair as the
        [--python EXE] [--chrome EXE] [--golden PATH]           showcase
    python3 cic_fixture.py --qps [same options]                 W02's pseudo-locale check on R7,
                                                                through the same showcase

The page template (D4 §4.2) boots from `showcase/page.json` and `showcase/story.json`; R7 was
packed before v2.3 and ships a `vega/story@1` (served at /v1/story, written as story.json in the
bundle) whose blocks the page no longer reads. This module converts it, mechanically and with
every choice stated here, so the parity probe can run the template on R7 exactly as a showcase
slot would serve it:

  story@2 (vega-show/references/schemas/story-2.schema.json, validated before it is written)
    task          from page.json#lane and #metric (the page_config builders on R7's netspec)
    lexicon       row: MANIFEST.serve.identity.noun / nouns; feature: "statistic" (the word the
                  hand-built page gave R7's inputs); source: "flow extractor"
    class_roles   the neutral class of story@1 (structural:majority), every other class alert on
                  the CIC stance (D4 §17.6: "roles: alert = every class but BENIGN (user)")
    values        story@1's values that a field references, plus primary (the network's primary
                  metric on the sealed test part, story@1's f1) and the breakdown's weak marks;
                  no ci_level (R7's evaluation records none: the page prints the interval
                  without a level); formats by value type; refs empty (story@1 carries no
                  per-value provenance)
    sections      context steps by id (CONTEXT_STEP), the network's decisions with focus@1
                  (CHOICE_FOCUS), the experiments with their metric, the results' models by
                  name (MODEL_OF) and facts by metric (FACT_METRIC; a fact that is no metric is
                  dropped), breakdown@1 per class with weak = the network's per-class value below
                  the trivial baseline's (records/evaluation.json), the verdict, the limits
    input_groups  story@1's regular expressions applied to the network's inputs (the first group
                  whose pattern matches takes an input, as the hand-built page did) → members
    code_notes    by walk id (WALK_ID), each with its report link and reference
  page.json       page_config.build_page_config (W06) on R7's netspec and records, with an
                  authored plan whose motif stance is the CIC one (lanes, D4 §17.6); gate BD2
                  (check_page_config) must be empty
  attribution     data/attribution.json: the v2.2 examples' per-class attribution (the rows the
                  hand-built page's "on real flows" map read) as vega/attribution@2
  report          report/report.pdf copied from the bundle (the code notes link to it)

--qps runs check_pseudo_locale.py's reading of the walked DOM on the same showcase, the page at
`?probe=all&locale=qps-ploc`, with the story's prose wrapped ⟪…⟫ as the showcase build wraps it
(STORY_KEEP: the story@2 fields that are not prose stay as they are): every visible chrome word
must come from the strings table, every dataset or service word be marked as data.

--parity serves the template (vega-show/assets/page) with that directory as the showcase,
through parity_server.py in front of R7's own running service, drives it with the parity probe
(`?probe=all`) and compares its facts with golden/cic-parity.json: capture_parity_golden.py
--check's run, with the showcase the template now needs. Exit codes as capture_parity_golden.py:
0 parity (or, without --strict, no browser or no bundle: WARN), 1 a fact differs or the run is
not clean, 2 the check could not run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))

import capture_parity_golden as cpg  # noqa: E402
import check_pseudo_locale as qps  # noqa: E402
import parity_browser  # noqa: E402
import parity_server  # noqa: E402
import page_template  # noqa: E402
from page_config import build_page_config, check_page_config  # noqa: E402
from page_derive import Catalogs, lane_record  # noqa: E402
from page_config import metric_record  # noqa: E402

SHOW = SCRIPTS.parent
STORY_SCHEMA = SHOW / "references" / "schemas" / "story-2.schema.json"
ATTRIBUTION_SCHEMA = SHOW / "references" / "schemas" / "attribution-2.schema.json"
STANCE = "2026-09-30 | cicids2017 | show: motif lanes (user)"
ROLES_BASIS = "user:cic-stance"
FEATURE = {"one": "statistic", "many": "statistics", "article": "a"}
SOURCE = {"one": "flow extractor", "many": "flow extractors", "article": "the"}
CONTEXT_STEP = {"The problem": "problem", "The data": "data", "Cleaning": "cleaning", "The split": "split"}
CHOICE_FOCUS = {
    "Two hidden layers of {width}": {"kind": "family", "ref": "architecture"},
    "GELU between the layers": {"kind": "family", "ref": "architecture"},
    "A little dropout": {"kind": "family", "ref": "regularization"},
    "Unweighted loss, lightly smoothed": {"kind": "family", "ref": "objective"},
    "The destination port matters": {"kind": "feature", "ref": "Destination Port"},
    "How it was trained": {"kind": "family", "ref": "optimization"},
}
MODEL_OF = {"f1": "network", "tree": "tree", "linear": "linear", "trivial": "trivial"}
FACT_METRIC = {"accuracy": "accuracy", "balanced": "balanced_accuracy", "ece": "ece"}
LIMIT_ID = {"In distribution only": "in-distribution", "Rare classes are anecdotes": "rare-classes",
            "Not every check ran": "checks-not-run"}
WALK_ID = {"act": "activation", "softmax": "link"}
REF_KIND = (("r_fig_", "figure"), ("r_tab_", "table"), ("r_eq_", "equation"), ("r_", "section"))
# story@2 fields that are not prose: the task, the lexicon (its words are marked as data by the
# page), the roles, ids, value keys and values, input names, focus, references and links
STORY_KEEP = frozenset({"schema", "locale", "task", "lexicon", "class_roles", "polarity", "values", "formats", "refs",
                        "id", "value", "names", "weak", "direction", "metric", "focus", "members", "report", "bundle",
                        "selected", "reference", "link", "ref", "kind", "format", "verdict", "name"})
PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)(?:\.[a-z][a-z0-9_]*)?(?::[a-z]+)?\}")


class FixtureError(ValueError):
    """R7's documents do not convert (an unknown step, choice, model or note)."""


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---- the R7 bundle ------------------------------------------------------------------------------
class Bundle:
    """What the conversion reads from R7: its story@1, netspec, manifest, records, examples."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "MANIFEST.json").is_file():
            raise FixtureError(f"{self.root} is not a bundle (no MANIFEST.json)")
        self.manifest_bytes = (self.root / "MANIFEST.json").read_bytes()
        self.manifest = json.loads(self.manifest_bytes)
        self.story = read_json(self.root / "story.json")
        if self.story.get("schema") != "vega/story@1":
            raise FixtureError(f"{self.root}/story.json is {self.story.get('schema')!r}, not vega/story@1")
        self.netspec = read_json(self.root / "netspec.json")
        self.shape = read_json(self.root / "records" / "shape.json")
        self.analysis = read_json(self.root / "records" / "analysis.json")
        self.evaluation = read_json(self.root / "records" / "evaluation.json")
        self.presets = read_json(self.root / "examples" / "presets.json")
        self.attribution = read_json(self.root / "examples" / "attribution.json")

    @property
    def inputs(self) -> list:
        """The network's inputs in network order (the attribution's order, the page's order)."""
        return list(self.attribution["inputs"])

    def section(self, sid: str) -> dict:
        return next((s for s in self.story["sections"] if s.get("id") == sid), {})


# ---- story@2 ----------------------------------------------------------------------------------------
def _placeholders(text) -> set:
    return set(PLACEHOLDER.findall(str(text or "")))


def _lexicon(b: Bundle) -> dict:
    ident = b.manifest["serve"]["identity"]
    one, many = ident["noun"], ident["nouns"]
    return {"schema": "vega/lexicon@1",
            "row": {"one": one, "many": many, "article": "an" if one[:1] in "aeiou" else "a"},
            "feature": dict(FEATURE), "source": dict(SOURCE),
            "source_of": {"row": "bundle:MANIFEST.serve.identity.noun", "feature": ROLES_BASIS, "source": ROLES_BASIS}}


def _class_roles(b: Bundle, classes: list) -> dict:
    neutral = b.story.get("neutral_class")
    if neutral not in classes:
        raise FixtureError(f"story@1 neutral_class {neutral!r} is not a class of the network")
    alert = [c for c in classes if c != neutral]
    return {"schema": "vega/class-roles@1", "roles": {c: ("neutral" if c == neutral else "alert") for c in classes},
            "source": {c: ("structural:majority" if c == neutral else ROLES_BASIS) for c in classes},
            "proposed": {"neutral": neutral, "rule": "structural:majority", "alert": alert, "evidence": [ROLES_BASIS]}}


def _input_groups(b: Bundle) -> list:
    """story@1's regex groups applied to the inputs: the first group whose pattern matches."""
    groups = b.story["input_groups"]
    taken = [[] for _ in groups]
    for name in b.inputs:
        k = next((k for k, g in enumerate(groups) if any(re.search(p, name) for p in g["match"])), None)
        if k is None:
            raise FixtureError(f"no story@1 input group takes input {name!r}")
        taken[k].append(name)
    out = []
    for g, members in zip(groups, taken):
        if members:
            gid = re.sub(r"[^a-z0-9]+", "_", g["label"].lower()).strip("_")[:32]
            out.append({"id": gid, "label": g["label"], "text": g.get("text", ""), "members": members})
    return out


def _context(b: Bundle) -> dict:
    sec = b.section("context")
    steps = []
    for s in sec["steps"]:
        if s["title"] not in CONTEXT_STEP:
            raise FixtureError(f"context step {s['title']!r} has no story@2 id (CONTEXT_STEP)")
        steps.append({"id": CONTEXT_STEP[s["title"]], "text": s["text"]})
    facts = [{k: f[k] for k in ("label", "value", "format") if k in f} for f in sec["facts"]]
    return {"id": "context", "steps": steps, "facts": facts}


def _network(b: Bundle) -> dict:
    sec, choices = b.section("network"), []
    for c in sec["choices"]:
        if c["title"] not in CHOICE_FOCUS:
            raise FixtureError(f"network decision {c['title']!r} has no focus@1 (CHOICE_FOCUS)")
        choices.append({"title": c["title"], "text": c["text"], "focus": CHOICE_FOCUS[c["title"]]})
    return {"id": "network", "lead": sec["lead"], "choices": choices}


def _experiments(b: Bundle, metric: dict) -> dict:
    sec = b.section("experiments")
    runs = [{"id": r["value"], "value": r["value"], "label": r["label"], "selected": bool(r.get("selected")),
             "reference": bool(r.get("reference"))} for r in sec["runs"]]
    return {"id": "experiments", "lead": sec["lead"], "caption": sec.get("caption", ""),
            "metric": {"name": metric["primary"], "direction": metric["direction"]}, "runs": runs}


def _weak(b: Bundle) -> list:
    """Per class: the network's F-score below the trivial baseline's (R-E3's weak, from the
    evaluation records; R7 records no per-class interval and no bar)."""
    models = b.evaluation["metrics"]["models"]
    net, trivial = models["network"]["f1_per_class"]["values"], models["trivial"]["f1_per_class"]["values"]
    return [bool(n < t) for n, t in zip(net, trivial)]


def _results(b: Bundle) -> dict:
    sec = b.section("results")
    models = []
    for m in sec["models"]:
        if m["value"] not in MODEL_OF:
            raise FixtureError(f"results model {m['label']!r} ({m['value']}) has no story@2 name (MODEL_OF)")
        models.append({"name": MODEL_OF[m["value"]], "value": m["value"], "selected": bool(m.get("selected"))})
    facts = [{"metric": FACT_METRIC[f["value"]], "value": f["value"], **({"note": f["note"]} if f.get("note") else {})}
             for f in sec["facts"] if f["value"] in FACT_METRIC]
    pc = sec["per_class"]
    breakdown = {"kind": "per_class", "metric": "f1", "names": pc["names"], "values": pc["values"], "weak": "per_class_weak",
                 "direction": "max", "caption": pc.get("caption")}
    limits = [{"id": LIMIT_ID[x["title"]], "title": x["title"], "text": x["text"]} for x in sec["limits"]]
    verdict = {"earned": True, "beaten_by_baseline": False, "strongest_baseline": "tree", "diff": "diff_tree"}
    return {"id": "results", "headline": sec["headline"], "models": models, "facts": facts, "breakdown": breakdown,
            "verdict": verdict, "limits": limits}


def _ref_of(page_key: str) -> tuple[str, str]:
    """(kind, number key) of a report reference from its page key (r_fig_arch_p → figure, r_fig_arch)."""
    number = page_key[:-2] if page_key.endswith("_p") else page_key
    return next(kind for prefix, kind in REF_KIND if number.startswith(prefix)), number


def _code_notes(b: Bundle, values: dict, walk_ids: set) -> dict:
    out = {}
    for sid, note in b.story["code_notes"].items():
        wid = WALK_ID.get(sid, sid)
        if wid not in walk_ids:
            raise FixtureError(f"code note {sid!r} has no walk id of this network ({sorted(walk_ids)})")
        page_key = next(iter(_placeholders(note.get("page"))), None)
        kind, number = _ref_of(page_key)
        title = re.sub(r"\s+", " ", PLACEHOLDER.sub("", note["label"]).replace("§", "")).strip(" ·")
        page = values[page_key]
        out[wid] = {"note": note["note"],
                    "link": {"href": f"showcase/report/report.pdf#page={int(page)}", "kind": "report"},
                    "ref": {"kind": kind, "number": number, "page": page_key, "title": title}}
    return out


def _formats(values: dict) -> dict:
    def fmt(v):
        if isinstance(v, dict):
            return "metric"
        if isinstance(v, list):
            return "list"
        if isinstance(v, bool) or isinstance(v, str):
            return "verbatim"
        return "count" if isinstance(v, int) else "param"
    return {k: fmt(v) for k, v in values.items() if k != "primary"}


def _referenced(story: dict) -> set:
    """The value keys some field of the story@2 names: placeholders and structured keys."""
    keys = set()
    texts = [story["headline"], story["lead"]]
    for s in story["sections"]:
        texts += [s.get("lead"), s.get("caption"), s.get("headline")]
        texts += [x.get("text") for x in s.get("steps", [])] + [x.get("title") for x in s.get("choices", [])]
        texts += [x.get("text") for x in s.get("choices", [])] + [x.get("text") for x in s.get("limits", [])]
        keys |= {f["value"] for f in s.get("facts", [])} | {r["value"] for r in s.get("runs", [])}
        keys |= {m["value"] for m in s.get("models", [])}
        if s.get("breakdown"):
            keys |= {s["breakdown"][k] for k in ("names", "values", "weak") if s["breakdown"].get(k)}
            texts.append(s["breakdown"].get("caption"))
        if s.get("verdict", {}).get("diff"):
            keys.add(s["verdict"]["diff"])
    for n in story["code_notes"].values():
        keys |= {n["ref"]["number"], n["ref"]["page"]}
        texts.append(n["note"])
    for t in texts:
        keys |= _placeholders(t)
    return keys


def build_story(b: Bundle, lane: dict, metric: dict, walk_ids: set) -> dict:
    classes = list(b.netspec["task"]["classes"])
    values = dict(b.story["values"])
    values["per_class_weak"] = _weak(b)
    story = {
        "schema": "vega/story@2", "locale": "en",
        "task": {"kind": lane["task"], "base": lane["base"], "goal": lane["goal"], "sample_shape": lane["shape"],
                 "graph_level": lane["level"], "family": lane["family"], "view_key": lane["view_key"],
                 "classes": classes, "outcomes": None, "primary": metric["primary"], "direction": metric["direction"]},
        "lexicon": _lexicon(b), "class_roles": _class_roles(b, classes), "polarity": "neutral-alert",
        "values": {}, "formats": {}, "refs": {},
        "headline": b.story["headline"], "lead": b.story["lead"],
        "sections": [_context(b), _network(b), _experiments(b, metric), _results(b)],
        "input_groups": _input_groups(b), "code_notes": _code_notes(b, values, walk_ids),
        "report": {"status": "linked", "file": "report/report.pdf", "pdf_digest": b.story["report"]["pdf_digest"],
                   "aux_digest": b.story["report"]["aux_digest"], "roles_digest": None, "roles_source": None, "briefs": None},
        "bundle": {"manifest_digest": sha256(b.manifest_bytes), "slug": b.manifest["slug"], "run": b.manifest["run"]},
    }
    used = _referenced(story)
    missing = sorted(k for k in used if k not in values)
    if missing:
        raise FixtureError(f"story@1 values lack {missing}")
    kept = {k: values[k] for k in sorted(used)}
    primary = values["f1"]
    story["values"] = {"primary": {"value": primary["value"], "ci": primary.get("ci"), "n": primary.get("n")}, **kept}
    story["formats"] = _formats(kept)
    return story


# ---- attribution@2 -------------------------------------------------------------------------------------
def build_attribution(b: Bundle) -> dict:
    """The v2.2 examples' per-class attribution (gradient of the true class's score × the shaped
    input, per demo row of the training part) in vega/attribution@2's form."""
    a = b.attribution
    if a.get("schema") != "vega/class-attribution@1":
        raise FixtureError(f"examples/attribution.json is {a.get('schema')!r}, not vega/class-attribution@1")
    columns = [{"id": f"c{k}", "name": c, "range": None} for k, c in enumerate(a["classes"])]
    cells = {col["id"]: {"n": a["by_class"][col["name"]]["n"], "correct": a["by_class"][col["name"]]["correct"],
                         "rows": a["by_class"][col["name"]]["rows"]} for col in columns if col["name"] in a["by_class"]}
    return {"schema": "vega/attribution@2", "source": {"file": "examples/demo.json", "part": "train"},
            "method": {"name": "grad_x_input", "steps": None}, "target": "logit", "inputs": list(a["inputs"]),
            "reduce": None, "axis": {"kind": "class", "columns": columns}, "cells": cells}


# ---- page.json ------------------------------------------------------------------------------------------
class _Paths:
    def __init__(self, plan: Path):
        self.plan = plan


def build_page(b: Bundle, story: dict, catalogs: Catalogs, scratch: Path) -> dict:
    """page_config.build_page_config on R7, with an authored plan that settles the CIC motif."""
    template = page_template.digests()
    plan = {"schema": "vega/story-plan@2", "status": "authored",
            "template": {"version": template["version"], "manifest_digest": template["manifest_digest"]},
            "proposals": {"motif": {"modality": "row", "stance": STANCE}}, "authored": {"motif": "lanes"}}
    (scratch / "story-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    examples = {"presets": b.presets} if isinstance(b.presets, list) else b.presets
    records = {"manifest": b.manifest, "manifest_digest": sha256(b.manifest_bytes),
               "shape": {"manifest": b.shape["manifest"], "recipe": b.shape["recipe"]},
               "analysis": {"context": b.analysis.get("context")}, "examples": examples, "reference_scores": None}
    page = build_page_config(_Paths(scratch / "story-plan.json"), b.netspec, records, story, catalogs)
    problems = check_page_config(page, None, catalogs)
    if problems:
        raise FixtureError("page.json fails BD2: " + "; ".join(str(p) for p in problems))
    return page


def _validate(doc: dict, schema_path: Path, what: str) -> None:
    import jsonschema
    errors = sorted(jsonschema.Draft202012Validator(read_json(schema_path)).iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        raise FixtureError(f"{what} fails {schema_path.name}: " + "; ".join(
            f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message[:160]}" for e in errors[:8]))


def write_pair(bundle: Path, out: Path) -> dict:
    """story.json, page.json, data/attribution.json and report/report.pdf of R7 into `out`."""
    b, catalogs = Bundle(bundle), Catalogs.load()
    lane, metric = lane_record(b.netspec, catalogs), metric_record(b.netspec, catalogs)
    from page_config import build_walk
    walk_ids = {s["id"] for s in build_walk(b.netspec, catalogs.explain)}
    story = build_story(b, lane, metric, walk_ids)
    _validate(story, STORY_SCHEMA, "story.json")
    attribution = build_attribution(b)
    _validate(attribution, ATTRIBUTION_SCHEMA, "data/attribution.json")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        page = build_page(b, story, catalogs, Path(scratch))
    (out / "data").mkdir(exist_ok=True)
    (out / "report").mkdir(exist_ok=True)
    for name, doc in (("story.json", story), ("page.json", page), ("data/attribution.json", attribution)):
        (out / name).write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pdf = b.root / "report" / "report.pdf"
    if pdf.is_file():
        shutil.copyfile(pdf, out / "report" / "report.pdf")
    return {"story": story, "page": page, "attribution": attribution}


# ---- the parity probe through the showcase ----------------------------------------------------------------
def wrap_prose(node, key=None):
    """The story with every authored string wrapped ⟪…⟫, as the showcase build wraps it under the
    pseudo-locale; the fields of STORY_KEEP kept."""
    if key in STORY_KEEP:
        return node
    if isinstance(node, str):
        return f"{qps.STORY_WRAP[0]}{node}{qps.STORY_WRAP[1]}" if node else node
    if isinstance(node, list):
        return [wrap_prose(v) for v in node]
    if isinstance(node, dict):
        return {k: wrap_prose(v, k) for k, v in node.items()}
    return node


def run_qps(a: argparse.Namespace, workdir: Path) -> int:
    chrome = parity_browser.find_chrome(a.chrome)
    if not chrome:
        raise cpg.Unavailable(f"no headless browser (checked $CHROME, {', '.join(parity_browser.CHROME_CANDIDATES)})")
    if a.bundle is None or not (Path(a.bundle) / "MANIFEST.json").is_file():
        raise cpg.Unavailable("no R7 bundle (set CIDS_ROOT or pass --bundle)")
    showcase = workdir / "showcase"
    docs = write_pair(Path(a.bundle), showcase)
    (showcase / "story.json").write_text(json.dumps(wrap_prose(docs["story"]), ensure_ascii=False), encoding="utf-8")
    upstream = cpg.Upstream(a.upstream, Path(a.bundle), a.python, workdir)
    try:
        upstream.wait_ready()
        srv = parity_server.start(a.template, upstream.url, parity_server.ServeOptions(showcase=showcase))
        try:
            probe, dom = qps.drive(chrome, srv.url, workdir)
        finally:
            srv.stop()
    finally:
        upstream.stop()
    (workdir / "probe.json").write_text(json.dumps(probe, indent=1, ensure_ascii=False), encoding="utf-8")
    problems, counts = qps.check(probe, dom, srv.requests, srv.url, story_wrapped=True)
    return qps.report(problems, counts, len(probe.get("views") or {}), workdir)


def run_parity(a: argparse.Namespace, workdir: Path) -> int:
    chrome = parity_browser.find_chrome(a.chrome)
    if not chrome:
        raise cpg.Unavailable(f"no headless browser (checked $CHROME, {', '.join(parity_browser.CHROME_CANDIDATES)})")
    if a.bundle is None or not (Path(a.bundle) / "MANIFEST.json").is_file():
        raise cpg.Unavailable("no R7 bundle (set CIDS_ROOT or pass --bundle)")
    showcase = workdir / "showcase"
    write_pair(Path(a.bundle), showcase)
    golden = read_json(a.golden)
    upstream = cpg.Upstream(a.upstream, Path(a.bundle), a.python, workdir)
    try:
        upstream.wait_ready()
        info, _ = cpg.identity_of(upstream)
        if info.get("model_version") != golden["upstream"]["model_version"]:
            raise SystemExit(f"ERROR: the upstream serves {info.get('model_version')}, the golden was captured on "
                             f"{golden['upstream']['model_version']}")
        srv = parity_server.start(a.template, upstream.url, parity_server.ServeOptions(showcase=showcase))
        try:
            probe = parity_browser.drive(chrome, srv.url, workdir)
        finally:
            srv.stop()
    finally:
        upstream.stop()
    (workdir / "probe-template.json").write_text(json.dumps(probe, indent=1, ensure_ascii=False), encoding="utf-8")
    return report(probe, cpg.clean(probe, srv.requests, srv.url), golden, workdir)


def report(probe: dict, problems: list, golden: dict, workdir: Path) -> int:
    for p in problems:
        cpg.say("FAIL", f"template: {p}")
    facts = cpg.compared(probe.get("parity") or {})
    diffs = [] if problems else cpg.diff({k: golden["facts"][k] for k in golden["compare"]},
                                         {k: facts[k] for k in golden["compare"]})
    for d in diffs:
        cpg.say("FAIL", f"differs from the golden: {d}")
    ext = probe.get("extensions") or {}
    lane = ext.get("lane") or {}
    cpg.say("INFO", f"lane {lane.get('view_key')}/{lane.get('shape')}, registry {ext.get('registry')}, "
                    f"missing keys {ext.get('missingKeys')}, unresolved {ext.get('unresolvedPlaceholders')}")
    if problems or diffs:
        cpg.say("FAIL", f"parity probe on {golden['upstream'].get('run')} through the CIC showcase: {len(problems)} run "
                        f"problem(s), {len(diffs)} difference(s) (probe output kept in {workdir})")
        return 1
    cpg.say("OK", f"parity with {golden['source']['commit']} on {golden['upstream'].get('run')} through the CIC "
                  f"showcase: {cpg.summary(facts)}; {len(probe['views'])} views rendered, no page error, no HTTP error")
    return 0


def _parse(argv) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path, metavar="DIR", help="write the showcase documents into DIR")
    mode.add_argument("--parity", action="store_true", help="the W01 parity probe on R7 through them")
    mode.add_argument("--qps", action="store_true", help="the pseudo-locale check on R7 through them")
    ap.add_argument("--bundle", type=Path, default=cpg.BUNDLE,
                    help="R7's bundle (default: $CIDS_ROOT/bundle/cicids2017-label-R7)")
    ap.add_argument("--template", type=Path, default=cpg.TEMPLATE)
    ap.add_argument("--golden", type=Path, default=cpg.GOLDEN)
    ap.add_argument("--upstream", default=None, help="R7's service, when it runs already")
    ap.add_argument("--python", default=sys.executable, help="the interpreter that runs the bundle's service")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    ap.add_argument("--strict", action="store_true", help="a missing browser or bundle is an error, not a WARN")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _parse(argv)
    if a.write:
        if a.bundle is None:
            print("ERROR: no R7 bundle (set CIDS_ROOT or pass --bundle)", file=sys.stderr)
            return 2
        try:
            docs = write_pair(a.bundle, a.write)
        except FixtureError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"OK: story@2, page@1 and attribution@2 of {a.bundle.name} written to {a.write} "
              f"({len(docs['story']['input_groups'])} input groups, {len(docs['page']['walk'])} walk steps, "
              f"motif {docs['page']['motif']['id']})")
        return 0
    workdir = Path(tempfile.mkdtemp(prefix="vega-cic-parity-"))
    try:
        rc = run_qps(a, workdir) if a.qps else run_parity(a, workdir)
        a.keep = a.keep or rc != 0
        return rc
    except cpg.Unavailable as exc:
        cpg.say("ERROR" if a.strict else "WARN", f"{exc}; parity not checked")
        return 2 if a.strict else 0
    except FixtureError as exc:
        cpg.say("ERROR", str(exc))
        return 2
    finally:
        if a.keep:
            cpg.say("INFO", f"work directory kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
