#!/usr/bin/env python3
"""
make_code_views.py — the fixtures of the code room's unit tests (tests/code.test.js): each one a
`vega/code-view@2` answer (R-D7) with the `page.json` walk and code language, the story's code
notes, and the walk the room must resolve from them.

    python3 make_code_views.py            # write every fixture next to this file
    python3 make_code_views.py --check    # every fixture as a fresh write gives it, and each one
                                          # holds together (below)

Each fixture is `vega/code-room-fixture@1`:

    page      {walk, code}: the walk `page_config.build_walk` computes (R-E8) on a small network of
              the fixture's block types, with the backend's explain catalog (its titles, texts and
              code anchors verbatim) — or, for the plain fixture, a walk of the same form written
              here — and page.json's `code` block
    story     {lexicon, values, code_notes, report}: the notes by walk id in the story-2 form
              ({note, link, ref}; number and page are values keys), the report facts they read
    code      the service's answer: vega/code-view@2, each file with its role and kind (the
              vega-pack catalog's closed sets), its digest, its line count and its text
    expected  {steps: [{id, path, from, to}], missing: [{id, why}]}: the walk resolved against
              the files by `resolve` below — an independent implementation of the room's rule
              (views/code.js resolveWalk): the first file whose path matches `file` and that holds
              an `at` line, from the `after` line on; the definition's extent (the Python
              highlighter's: decorators, header, every deeper-indented line, trailing blank lines
              out, at most 90 lines; the plain reader's: to the next blank line); `from`/`to` are
              1-based. A step that does not resolve is missing, with why: file | anchor | pattern.

The fixtures:
  row-linked   a network of dense, activation, dropout and head blocks; every walk file served;
               notes in linked mode (report references of each kind: figure, section, equation,
               table; a reference without a link; a note with neither)
  graph-briefs a node-level graph network (gcn_conv, activation, dropout, head); the graph blocks'
               file, the inspector and the HTTP routes not served, and the store's file without its
               load method, so four steps drop out (file, file, file, anchor); notes in brief-link
               mode (a stage brief per note, no report reference)
  plain        a language the room has no highlighter for: a walk of two steps over plain text, a
               third whose anchor is missing, one whose file is not served

The sources are small, written here, and name no dataset: a fixture belongs to no lane, and the
template lint scans this directory too. `resolve`, `python_extent` and `plain_extent` are imported
by vega-show/scripts/tests/test_code_room.py, which runs the same comparison on the templates the
backend renders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
SHOW_SCRIPTS = REPO / "vega-show" / "scripts"
EXPLAIN = REPO / "vega-forge" / "scripts" / "backends" / "pytorch" / "explain-catalog.json"
PACK_CATALOG = REPO / "vega-pack" / "scripts" / "catalog.json"
STORY_SCHEMA = REPO / "vega-show" / "references" / "schemas" / "story-2.schema.json"
SCHEMA = "vega/code-room-fixture@1"
CODE_VIEW = "vega/code-view@2"
MAX_LINES = 90


# ---- the room's rule, written again -----------------------------------------------------------------
def python_extent(lines: list[str], i: int) -> list[int]:
    """[first, last] of the definition lines[i] opens (views/highlight/python.js extent)."""
    indent = lambda s: len(s) - len(s.lstrip())
    decorator = lambda s: s.lstrip().startswith("@")
    a = i
    while a > 0 and decorator(lines[a - 1]):
        a -= 1
    d = i
    while d < len(lines) - 1 and decorator(lines[d]):
        d += 1
    base, e = indent(lines[d]), d + 1
    while e < len(lines) and (not lines[e].strip() or indent(lines[e]) > base):
        e += 1
    while e > d + 1 and not lines[e - 1].strip():
        e -= 1
    return [a, min(e, a + MAX_LINES) - 1]


def plain_extent(lines: list[str], i: int) -> list[int]:
    """[first, last]: lines[i] down to the next blank line (views/highlight/plain.js extent)."""
    e = i + 1
    while e < len(lines) and lines[e].strip():
        e += 1
    return [i, min(e, i + MAX_LINES) - 1]


def _compiled(src):
    try:
        return re.compile(src) if isinstance(src, str) else None
    except re.error:
        return None


def resolve_step(step: dict, files: list[dict], extent) -> dict:
    code = step.get("code") or {}
    file, at = _compiled(code.get("file")), [_compiled(a) for a in code.get("at") or []]
    after = _compiled(code["after"]) if code.get("after") else None
    to = _compiled(code["to"]) if code.get("to") else None
    if not file or not at or None in at or (code.get("after") and not after) or (code.get("to") and not to):
        return {"id": step["id"], "why": "pattern"}
    served = [f for f in files if file.search(f["path"])]
    for f in served:
        lines = f["text"].split("\n")
        start = next((k for k, line in enumerate(lines) if after.search(line)), -1) if after else 0
        if start < 0:
            continue
        hits = [next((k for k, line in enumerate(lines) if k >= start and r.search(line)), -1) for r in at]
        i = next((k for k in hits if k >= 0), -1)
        if i < 0:
            continue
        rng = extent(lines, i)
        j = next((k for k, line in enumerate(lines) if k > rng[1] and to.search(line)), -1) if to else -1
        if j > rng[1] and j - rng[1] < 12:
            rng[1] = extent(lines, j)[1]
        return {"id": step["id"], "path": f["path"], "from": rng[0] + 1, "to": rng[1] + 1}
    return {"id": step["id"], "why": "anchor" if served else "file"}


def resolve(walk: list[dict], files: list[dict], language: str | None) -> dict:
    extent = python_extent if language == "python" else plain_extent
    out = [resolve_step(s, files, extent) for s in walk]
    return {"steps": [s for s in out if "why" not in s], "missing": [s for s in out if "why" in s]}


# ---- the sources ----------------------------------------------------------------------------------
ROOT = '''"""The bundle's composition root: load() wires the adapters; the command line calls it."""
from __future__ import annotations

import json
from pathlib import Path

from pkg.model.network import build_model
from pkg.model.store import ModelRegistry
from pkg.serving.predict import Predictor


def bundle_root(bundle_dir=None) -> Path:
    return Path(bundle_dir or Path(__file__).parent)


def load(bundle_dir=None) -> Predictor:
    """Checks the files against the manifest, then wires the predictor."""
    root = bundle_root(bundle_dir)
    manifest = json.loads((root / "MANIFEST.json").read_text())

    spec = json.loads((root / "netspec.json").read_text())
    model = ModelRegistry().load(root / "weights.pt", spec=spec, build=build_model)
    return Predictor(model, manifest)


def main(argv=None) -> int:
    return 0 if load() else 1
'''

NETWORK = '''"""pkg.model.network — the Model adapter: the stages the spec lists, built by the builder."""
from __future__ import annotations

import torch
from torch import nn


class Scale(nn.Module):
    """A fixed rescaling."""

    def __init__(self, factor=1.0):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class Network(nn.Module):
    """The stages in the spec's order, then the readout."""

    def __init__(self, stages, readout):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.readout = readout

    def forward(self, batch):
        x = batch["x_num"]
        for stage in self.stages:
            x = stage(x)
        return x

    @torch.no_grad()
    def predict_proba(self, batch):
        return self.readout.proba(self.forward(batch))

    @classmethod
    def build(cls, spec, run_dir=None):
        from .builder import NetworkBuilder
        return NetworkBuilder(spec).network(cls)


build_model = Network.build
'''

BLOCKS_CORE = """\"\"\"pkg.model.blocks_core — the core blocks, one class each.\"\"\"
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "none": nn.Identity}


class Dense(nn.Module):
    '''z = W x + b, then the block's own activation.

    The weights are the grid the network view draws.'''

    def __init__(self, n_in, units, activation="none"):
        super().__init__()
        self.linear = nn.Linear(n_in, units)
        self.act = ACTIVATIONS[activation]()

    def forward(self, x):
        return self.act(self.linear(x))


class Activation(nn.Module):
    def __init__(self, name="relu"):
        super().__init__()
        self.fn = ACTIVATIONS[name]()

    def forward(self, x):
        return self.fn(x)


class Dropout(nn.Dropout):
    \"\"\"Silent while answering: the predictor puts the model in evaluation mode.\"\"\"


class Head(nn.Module):
    def __init__(self, n_in, out, bias=True):
        super().__init__()
        self.linear = nn.Linear(n_in, out, bias=bias)
        self.scale = 1e-3

    def forward(self, x):
        return self.linear(x)  # the raw scores: the readout reads them
"""

BLOCKS_GRAPH = '''"""pkg.model.blocks_graph — the graph blocks."""
from __future__ import annotations

from torch import nn


class GCNConv(nn.Module):
    def __init__(self, n_in, out):
        super().__init__()
        self.linear = nn.Linear(n_in, out)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)
'''

STORE = '''"""pkg.model.store — the ModelStore adapter."""
from __future__ import annotations

import hashlib

import torch


class ModelRegistry:
    """Saves and loads the weights, each checked against its recorded digest."""

    def save(self, model, path, spec=None, epoch=None):
        torch.save({"state": model.state_dict(), "epoch": epoch}, path)

    def load(self, checkpoint, spec=None, build=None):
        blob = torch.load(checkpoint, weights_only=True)
        digest = hashlib.sha256(repr(sorted(blob["state"])).encode()).hexdigest()
        model = build(spec)
        model.load_state_dict(blob["state"])
        return model.eval()

    def version(self, path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
'''

STORE_WITHOUT_LOAD = STORE.replace("    def load(self, checkpoint", "    def restore(self, checkpoint")

PREPROCESS = '''"""pkg.features.preprocess — the recipe, transform only, with its fitted parameters."""
from __future__ import annotations

FITTED = {"x0": {"median": 0.5, "scale": 2.0}}


def transform(frame, history=None):
    out = frame.copy()
    for column, p in FITTED.items():
        out[column] = (out[column] - p["median"]) / p["scale"]
    return out
'''

SERVING = '''"""pkg.serving.predict — the Predictor adapter."""
from __future__ import annotations

import torch


def _forward(model, batch):
    return model(batch)


class Predictor:
    def __init__(self, model, manifest):
        if model.training:
            raise ValueError("the model is in training mode")
        self.model, self.manifest = model, manifest

    def predict(self, rows):
        return self._forward(rows)

    @torch.no_grad()
    def _forward(self, prepared):
        out = self.model(prepared)
        return out
'''

INSPECTOR = '''"""pkg.serving.inspection.inspector — the read-only view of the network the page draws."""
from __future__ import annotations


class NetworkInspector:
    def __init__(self, model, graph):
        self.model, self.graph = model, graph

    def explain(self, request):
        batch = request.batch()
        return self.graph.capture(self.model, batch)
'''

ROUTES = '''"""serve/routes_predict.py — POST /v1/predict and /v1/predict/csv."""
from __future__ import annotations

ENDPOINTS = (("POST", "/v1/predict"), ("POST", "/v1/predict/csv"))
CAPABILITY = "core"


def register(app, state):
    @app.post("/v1/predict")
    def predict(body: dict):
        return state.answer(body)

    @app.post("/v1/predict/csv")
    def predict_csv(body: bytes):
        return state.answer_table(body)
'''

PORTS = '''"""pkg.ports — the ports: the interfaces the adapters implement."""
from typing import Protocol


class Predictor(Protocol):
    def predict(self, rows): ...
'''

DOMAIN = '''"""pkg.domain — the value objects."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    kind: str
'''

PLAIN_MAIN = """# the entry point of a service written in another language
start <- function(dir) {
  model <- read_model(dir)
  serve(model)
}

answer <- function(model, rows) {
  scores <- model %*% rows
  scores
}
"""

PLAIN_MODEL = """# the model, read once
read_model <- function(dir) {
  weights <- read_weights(dir)
  weights
}
"""


def code_file(path: str, role: str, kind: str, text: str) -> dict:
    return {"path": path, "role": role, "kind": kind,
            "digest": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "lines": text.count("\n") + 1, "text": text}


def code_view(files: list[dict], language: str = "python") -> dict:
    return {"schema": CODE_VIEW, "verified": True, "language": language, "files": files}


# ---- the fixtures ---------------------------------------------------------------------------------
LEXICON = {"schema": "vega/lexicon@1", "row": {"one": "item", "many": "items", "article": "an"},
           "feature": {"one": "field", "many": "fields", "article": "a"},
           "source": {"one": "record system", "many": "record systems", "article": "the"},
           "source_of": {"row": "vega:default", "feature": "vega:default", "source": "vega:default"}}


def _walk(block_types: list[str]) -> list[dict]:
    """page_config.build_walk on a network of those block types, with the explain catalog."""
    sys.path.insert(0, str(SHOW_SCRIPTS))
    from page_config import build_walk
    catalog = json.loads(EXPLAIN.read_text(encoding="utf-8"))
    blocks = [{"id": f"B{i + 1}", "type": t, "params": {}} for i, t in enumerate(block_types)]
    return build_walk({"blocks": blocks}, catalog)


def _ref(kind, key, title):
    return {"kind": kind, "number": f"{key}_number", "page": f"{key}_page", "title": title}


def _report(page):
    return {"href": f"showcase/report/report.pdf#page={page}", "kind": "report"}


def row_linked() -> dict:
    files = [code_file("pkg/model/network.py", "Model", "adapter", NETWORK),
             code_file("pkg/model/blocks_core.py", "Model", "support", BLOCKS_CORE),
             code_file("pkg/features/preprocess.py", "FeatureTransformer", "adapter", PREPROCESS),
             code_file("pkg/serving/predict.py", "Predictor", "adapter", SERVING),
             code_file("pkg/model/store.py", "ModelStore", "adapter", STORE),
             code_file("pkg/serving/inspection/inspector.py", "Inspection", "support", INSPECTOR),
             code_file("predict.py", "composition_root", "entrypoint", ROOT),
             code_file("serve/routes_predict.py", "http_entry", "entrypoint", ROUTES),
             code_file("pkg/ports.py", "ports", "support", PORTS),
             code_file("pkg/domain.py", "domain", "support", DOMAIN)]
    values = {"report_arch_number": "3.3", "report_arch_page": 24, "report_fig_wiring_number": 15,
              "report_fig_wiring_page": 25, "report_eq_dense_number": 2, "report_eq_dense_page": 27,
              "report_tab_blocks_number": 4, "report_tab_blocks_page": 28}
    notes = {
        "root": {"note": "one adapter per role, wired in this one place", "link": _report(25),
                 "ref": _ref("figure", "report_fig_wiring", "Ports and adapters")},
        "store": {"note": "the weights are refused if their digest differs", "link": _report(24),
                  "ref": _ref("section", "report_arch", "System architecture")},
        "dense": {"note": "W x + b, the affine part of every layer", "link": _report(27),
                  "ref": _ref("equation", "report_eq_dense", "The dense layer")},
        "head": {"note": "one raw score per class", "link": _report(28),
                 "ref": _ref("table", "report_tab_blocks", "The blocks")},
        "forward": {"note": "the layers in the order the equation writes them (Equation {report_eq_dense_number})",
                    "link": None, "ref": _ref("equation", "report_eq_dense", "The dense layer")},
        "shape": {"note": "the steps this file replays, one per {row.one}", "link": None, "ref": None},
    }
    return _fixture("row-linked", _walk(["dense", "activation", "dropout", "dense", "head"]), "python",
                    files, {"values": values, "code_notes": notes,
                            "report": {"status": "linked", "file": "report/report.pdf"}})


def graph_briefs() -> dict:
    files = [code_file("pkg/model/network.py", "Model", "adapter", NETWORK),
             code_file("pkg/model/blocks_core.py", "Model", "support", BLOCKS_CORE),
             code_file("pkg/features/preprocess.py", "FeatureTransformer", "adapter", PREPROCESS),
             code_file("pkg/serving/predict.py", "Predictor", "adapter", SERVING),
             code_file("pkg/model/store.py", "ModelStore", "adapter", STORE_WITHOUT_LOAD),
             code_file("predict.py", "composition_root", "entrypoint", ROOT)]
    brief = lambda stage: {"href": f"showcase/briefs/{stage}-brief.md", "kind": "file"}
    notes = {"root": {"note": "where the modules meet", "link": brief("vega-pack"), "ref": None},
             "activation": {"note": "the bend between the convolutions", "link": brief("vega-forge"), "ref": None},
             "gcn_conv": {"note": "a note on a step whose file is not served", "link": brief("vega-forge"), "ref": None},
             "run": {"note": "the batch the network reads, answered", "link": None, "ref": None}}
    return _fixture("graph-briefs", _walk(["gcn_conv", "activation", "dropout", "gcn_conv", "head"]), "python",
                    files, {"values": {}, "code_notes": notes,
                            "report": {"status": "brief-links", "file": None}})


def plain() -> dict:
    anchor = lambda file, at: {"file": file, "at": [at], "after": None, "to": None}
    walk = [{"id": "root", "title": "Start-up", "text": "The service reads its model once.",
             "code": anchor("(^|/)main\\.txt$", "^start <- function")},
            {"id": "store", "title": "The weights", "text": "Read from the bundle.",
             "code": anchor("(^|/)model\\.txt$", "^read_model <- function")},
            {"id": "build", "title": "The network", "text": "Not in these files.",
             "code": anchor("(^|/)model\\.txt$", "^build <- function")},
            {"id": "route", "title": "The route", "text": "Not served.",
             "code": anchor("(^|/)routes\\.txt$", "^route")},
            {"id": "run", "title": "Answering", "text": "The scores, from the rows.",
             "code": anchor("(^|/)main\\.txt$", "^answer <- function")}]
    files = [code_file("app/main.txt", "composition_root", "entrypoint", PLAIN_MAIN),
             code_file("app/model.txt", "Model", "adapter", PLAIN_MODEL)]
    notes = {"run": {"note": "the scores the page reads", "link": None, "ref": None}}
    return _fixture("plain", walk, "text", files, {"values": {}, "code_notes": notes,
                                                    "report": {"status": "brief-links", "file": None}})


def _fixture(name, walk, language, files, story) -> dict:
    code = code_view(files, language)
    return {"schema": SCHEMA, "name": name,
            "page": {"walk": walk, "code": {"language": language, "source": "coverage.json:language"}},
            "story": {"lexicon": LEXICON, **story},
            "code": code,
            "expected": resolve(walk, code["files"], language)}


FIXTURES = {"row-linked": row_linked, "graph-briefs": graph_briefs, "plain": plain}


# ---- the checks -----------------------------------------------------------------------------------
def problems(doc: dict) -> list[str]:
    """What does not hold in one fixture: its expected walk, the answer's closed sets, the notes."""
    out = []
    code, page = doc["code"], doc["page"]
    if resolve(page["walk"], code["files"], page["code"]["language"]) != doc["expected"]:
        out.append("expected is not the walk resolved against the files")
    closed = json.loads(PACK_CATALOG.read_text(encoding="utf-8"))["code_view"]
    for f in code["files"]:
        if f["role"] not in closed["roles"] or f["kind"] not in closed["kinds"]:
            out.append(f"{f['path']}: role {f['role']!r} / kind {f['kind']!r} outside the catalog's code_view")
        if f["lines"] != f["text"].count("\n") + 1:
            out.append(f"{f['path']}: lines {f['lines']}")
    ids = {s["id"] for s in page["walk"]}
    out += [f"code note {k!r} is not a walk id" for k in doc["story"]["code_notes"] if k not in ids]
    try:
        import jsonschema
    except ImportError:
        return out
    story = json.loads(STORY_SCHEMA.read_text(encoding="utf-8"))
    note = jsonschema.Draft202012Validator({**story["$defs"]["code_note"], "$defs": story["$defs"]})
    out += [f"code_notes.{k}: {e.message}" for k, n in doc["story"]["code_notes"].items() for e in note.iter_errors(n)]
    return out


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with a fresh write and check each fixture")
    a = ap.parse_args(argv)
    bad = []
    for name, make in FIXTURES.items():
        doc, path = make(), HERE / f"{name}.json"
        bad += [f"{name}: {p}" for p in problems(doc)]
        if a.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != render(doc):
                bad.append(f"{name}: {path.name} is not a fresh write (run make_code_views.py)")
        else:
            path.write_text(render(doc), encoding="utf-8")
            print(f"OK: wrote {path.name} ({len(doc['expected']['steps'])} steps, {len(doc['expected']['missing'])} left out)")
    for line in bad:
        print(f"FAIL: {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
