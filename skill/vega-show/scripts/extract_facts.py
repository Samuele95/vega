#!/usr/bin/env python3
"""
extract_facts.py — the fact set of one shipped network (`vega/show-facts@1`, contract §2).

    python3 extract_facts.py --model M.pack.html --bundle B [--report R] [--label-roles J]
                             --out plan/facts.json [--markers PATH] [--json]

Every number the page can show is read here, once, from the bundle's pinned records (each
verified against `MANIFEST.files` before it is read) and the report's labels, with the ref it
came from; the draft, the validator and the resolver read the fact set and nothing else. The
model page is read for three things only: the `bundle` layer (to prove it describes this
bundle), the Turtle digest, and the layer names.

**Preflight** (contract §10), each refusal naming its gate: `ERROR: <gate> <where>: <why>`, exit 2.

- `PF2` the bundle's records cannot be the fact source: `MANIFEST.json` or a pinned record does
  not read, a pinned record differs from its `MANIFEST.files` digest (`BundleAltered`), the
  bundle does not pin the spec and a record of every required layer, the evaluation records no
  network estimate of the primary (`facts_results`), `MANIFEST.json` holds a NaN or an
  infinity (named by its path: canonical JSON cannot spell it), or the facts read from the
  records do not validate `show-facts-1.schema.json` (a record carries a value the schema does
  not admit). A non-finite number elsewhere in a record is no fact and no group value, with a
  `WARN:` naming its ref or group path (`fact_sink`);
- `PF1` the model page does not read (nor its layers, without domain-forge's `apply_layer.py`),
  is not a `.pack.html` page, lacks a required layer, or its `bundle` layer's manifest is not
  `MANIFEST.json` (canonical JSON; a non-finite number in it is named): the page describes
  another bundle;
- `PF8` the round trip failed (`MANIFEST.roundtrip.status == fail`);
- `PF4` the bundle has no service, or no `MANIFEST.serve.showcase_slot` (re-pack with `--serve`);
- `PF3` `predict.py --check` fails from a clean shell (no `PYTHONPATH`, no user site, no
  bytecode written into the bundle);
- `PF6` the report's labels cannot be read or are stale: the label-role map is missing or not
  the one written from the report's current `.aux`, `--report` names no compiled report, or the
  `.aux` does not parse;
- `PF7` the task kind, family, a block type (composites' `inner` blocks included) or the sample
  shape is unknown to the vega-forge catalog, the primary has no max/min direction there, a
  `sequence-*` task has no base kind its `sequence_map` knows, the vega-show catalog has no view
  key for the task (`facts_network.task_shape`), or the spec names no network input
  (`facts_data`).

**The report.** `--label-roles`, else `label-roles.json` in `--report` (a directory) or beside
it, else `docs/label-roles.json` beside the model, in the project that holds `bundle/`, or in the
working directory: a role map binds the roles. Without one, a compiled report is `*.pdf` with a
same-stem `.aux` (in `--report`, or in those `docs/` directories): the roles are proposed later,
with `WARN:`. Without either, the code notes link to the stage briefs, with `WARN: no report`.

The fact set is validated against `show-facts-1.schema.json` before it is written (`PF2`). A
record id spelling a fact key another id spells (`S1`, `s1`), or none, is a `WARN:` and no second
fact (`FactSink.take`). Markers are printed (`--json`: one JSON result instead) and, with
`--markers` (by default when `--out` lies in a showcase directory's `plan/`, its
`.work/markers.json`), appended to a JSON list — a file there that is not one is left as it
is, with a `WARN:`. Exit 0 written, 2 refused (or the tool unable to run: its catalogs, `--out`
not writable, no `jsonschema` to validate with).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fact_sink import Catalogs, FactSink, Refusal, nonfinite, peek  # noqa: E402
from facts_campaign import CampaignFacts  # noqa: E402
from facts_data import DataFacts  # noqa: E402
from facts_network import NetworkFacts, task_shape  # noqa: E402
from facts_report import ReportFacts, report_facts  # noqa: E402
from facts_results import ResultFacts  # noqa: E402
from digit_rule import TOKEN_RE  # noqa: E402
from records import BundleRecords  # noqa: E402
from show_ports import RecordStore  # noqa: E402
from vega_show_core import (CATALOG, FactSet, ShowError, canonical_json, now,  # noqa: E402
                            read_json, write_json)

__all__ = ["FactExtractor", "ExtractContext", "ModelPage", "preflight", "report_facts", "main"]

_LAYER_DATA = '<script id="layer-{name}-data" type="application/json">(.*?)</script>'
_READINGS = ("records/geometry.json", "records/analysis.json", "records/shape.json",
             "records/architecture.json", "records/training.json", "records/evaluation.json")
_REQUIRED = ("netspec.json", *(f"records/{layer}.json" for layer
                               in CATALOG["stage"]["required_layers"] if layer != "bundle"))
_PREDICT_TIMEOUT_S = 1800


@dataclass(frozen=True)
class ExtractContext:
    """What the extraction reads beside the records: the fact set's header (`extracted_at`,
    `bundle`, `model`) and the bundle layer's markers (their `WARN:` lines are limits)."""

    header: dict = field(default_factory=dict)
    layer_markers: tuple = ()


class FactExtractor:
    """The composition of the five collectors over one bundle's records."""

    def __init__(self, records: RecordStore, report: ReportFacts, catalog: Catalogs,
                 context: ExtractContext | None = None) -> None:
        self._store, self._report, self._catalog = records, report, catalog
        self._context = context or ExtractContext()
        self._markers: tuple[str, ...] = ()

    @property
    def markers(self) -> tuple[str, ...]:
        """The `WARN:` lines of the last extraction."""
        return self._markers

    def extract(self) -> FactSet:
        pinned = (self._store.manifest().get("files") or {})
        missing = [r for r in _REQUIRED if r not in pinned]
        if missing:
            raise Refusal("PF2", "MANIFEST.json#files", f"the bundle pins no {missing}: the facts "
                          "are read from the records a bundle pins — re-run /vega-pack")
        task = task_shape(self._store, self._catalog)
        sink = FactSink(self._catalog.show["name_tokens"])
        for collector in (DataFacts(self._store, self._catalog),
                          NetworkFacts(self._store, self._catalog),
                          CampaignFacts(self._store, self._catalog),
                          ResultFacts(self._store, self._catalog, self._context.layer_markers)):
            collector.collect(sink)
        self._report.collect(sink)
        identity = peek(self._store, "MANIFEST.json#serve.identity", {}) or {}
        sink.group("identity", identity)
        sink.group("prose_sources", _prose_sources(self._store))
        sink.name([*_tokens(identity.get("title")), *(task.classes or ()),
                   *_metric_names(self._store)])
        if peek(self._store, "records/analysis.json#schema") is None:
            sink.warn("the bundle pins no analysis record: the facts that read it (relations, "
                      "drift, the spatial modality) are omitted")
        self._markers = sink.markers
        return sink.build(task, self._context.header)


class ModelPage:
    """The model page `<stem>.pack.html`, read through domain-forge's layer module."""

    def __init__(self, path: Path, html: str) -> None:
        self.path, self._html, self._layers = Path(path), html, _apply_layer_module()

    @classmethod
    def open(cls, path) -> "ModelPage":
        try:
            return cls(Path(path), Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise Refusal("PF1", path, f"the model page could not be read: {exc}") from exc

    def layers(self) -> list[str]:
        return [layer["name"] for layer in self._layers.list_layers(self._html)]

    def turtle_digest(self) -> str:
        return self._layers.domain_digest(self._html)

    def layer_data(self, name: str):
        match = re.search(_LAYER_DATA.format(name=re.escape(name)), self._html, re.S)
        try:
            return json.loads(match.group(1)) if match else None
        except json.JSONDecodeError:
            return None


def preflight(model: ModelPage, store: BundleRecords, check_predict: bool = True) -> list[str]:
    """Gates PF1, PF8, PF4 and PF3 (PF2 was `BundleRecords.open`); the `OK:` lines."""
    if not model.path.name.endswith(".pack.html"):
        raise Refusal("PF1", model.path, "the model page is not the /vega-pack page "
                                         "<stem>.pack.html of a bundle")
    missing = [n for n in CATALOG["stage"]["required_layers"] if n not in model.layers()]
    if missing:
        raise Refusal("PF1", model.path, f"the model page lacks the layer(s) {missing}")
    layer = model.layer_data("bundle") or {}
    _refuse_nonfinite(store.manifest(), "PF2", "MANIFEST.json")
    _refuse_nonfinite(layer.get("manifest"), "PF1", f"{model.path}#layer-bundle-data.manifest")
    if canonical_json(layer.get("manifest")) != canonical_json(store.manifest()):
        raise Refusal("PF1", f"{model.path}#layer-bundle-data.manifest",
                      f"the bundle layer's manifest is not {store.bundle / 'MANIFEST.json'}: the "
                      "page describes another bundle — pass the matching --model or --bundle")
    manifest = store.manifest()
    if (manifest.get("roundtrip") or {}).get("status") == "fail":
        raise Refusal("PF8", "MANIFEST.json#roundtrip.status", "the bundle's round trip failed — "
                      "a network that does not reproduce its evaluation is not shown")
    serve = manifest.get("serve") or {}
    if serve.get("present") is not True or not serve.get("showcase_slot"):
        raise Refusal("PF4", "MANIFEST.json#serve.showcase_slot", "the bundle has no service with "
                      "a showcase slot — re-run /vega-pack with --serve")
    lines = ["OK: PF1 the model page's bundle layer describes this bundle",
             "OK: PF4 the bundle serves a showcase slot"]
    if check_predict:
        lines.append(_predict_check(store.bundle))
    return lines


def _predict_check(bundle: Path) -> str:
    if not (bundle / "predict.py").is_file():
        raise Refusal("PF3", bundle / "predict.py", "the bundle ships no predict.py")
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
           "HOME": os.environ.get("HOME", str(bundle)), "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        done = subprocess.run([sys.executable, "predict.py", "--check"], cwd=bundle, env=env,
                              capture_output=True, text=True, timeout=_PREDICT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise Refusal("PF3", bundle / "predict.py", f"--check ran past {_PREDICT_TIMEOUT_S} s")
    if done.returncode != 0:
        tail = " | ".join((done.stdout + done.stderr).strip().splitlines()[-4:])
        raise Refusal("PF3", bundle / "predict.py", f"--check failed from a clean shell: {tail}")
    return "OK: PF3 predict.py --check passes from a clean shell"


# --------------------------------------------------------------------------- helpers

def _apply_layer_module():
    """domain-forge's `apply_layer` (the one copy of the layer format and the Turtle digest):
    `$DOMAIN_FORGE_DIR`, else the sibling skill."""
    base = Path(os.environ.get("DOMAIN_FORGE_DIR")
                or HERE.parent.parent / "domain-forge" / "scripts")
    path = base / "apply_layer.py" if base.is_dir() else base
    if not path.is_file():
        raise Refusal("PF1", path, "the model page's layers cannot be read without domain-forge's "
                                   "apply_layer.py (set DOMAIN_FORGE_DIR)")
    spec = importlib.util.spec_from_file_location("vega_show_df_apply_layer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prose_sources(store: RecordStore) -> list[dict]:
    """The readings each record carries, by ref: what the model reads and never copies."""
    out = []
    for record in _READINGS:
        readings = peek(store, f"{record}#readings", {}) or {}
        for key, value in readings.items() if isinstance(readings, dict) else ():
            nested = value.items() if isinstance(value, dict) else [(None, value)]
            for sub, text in nested:
                path = f"readings.{key}" + (f".{sub}" if sub else "")
                if isinstance(text, str) and text.strip() and " " not in path:
                    out.append({"what": f"{record.split('/')[-1][:-5]} {path[9:]}",
                                "ref": f"{record}#{path}"})
    return out


def _tokens(text) -> list[str]:
    return [m.group(0) for m in TOKEN_RE.finditer(text)] if isinstance(text, str) else []


def _metric_names(store: RecordStore) -> list[str]:
    models = peek(store, "records/evaluation.json#metrics.models", {}) or {}
    return sorted({m for scores in models.values() if isinstance(scores, dict) for m in scores})


def _header(args, model: ModelPage, store: BundleRecords) -> dict:
    manifest = store.manifest()
    return {"extracted_at": now(),
            "bundle": {"dir": str(args.bundle), "manifest_digest": store.manifest_digest(),
                       "slug": manifest.get("slug"), "run": manifest.get("run"),
                       "model_version": manifest.get("model_version")},
            "model": {"path": str(args.model), "turtle_digest": model.turtle_digest(),
                      "layers": model.layers()}}


def _refuse_nonfinite(value, gate: str, where: str) -> None:
    """Refuse (`gate`) a JSON value that holds a NaN or an infinity, naming its first path."""
    found = nonfinite(value)
    if found:
        path = f"{where}#{found[0]}" if "#" not in where else f"{where}.{found[0]}"
        raise Refusal(gate, path, f"a non-finite number (NaN or an infinity), which JSON cannot "
                      f"spell{f' ({len(found)} in all)' if len(found) > 1 else ''} — the "
                      "stage that wrote it must write a finite number or null")


def _schema_problems(doc: dict) -> list[str]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # the tool cannot check what it writes: no fact set, no gate
        raise ShowError("jsonschema (draft 2020-12) is needed to validate vega/show-facts@1 "
                        "before it is written; install jsonschema>=4") from exc
    schema = json.loads((HERE.parent / CATALOG["schemas"]["vega/show-facts@1"]["file"])
                        .read_text(encoding="utf-8"))
    return [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
            for e in Draft202012Validator(schema).iter_errors(doc)]


def _markers_path(args) -> Path | None:
    if args.markers:
        return Path(args.markers)
    out = Path(args.out).resolve()
    siblings = (CATALOG["showcase"]["dir"], *CATALOG["showcase"]["siblings"].values())
    if out.parent.name == "plan" and out.parent.parent.name in siblings:
        return out.parent.parent / ".work" / "markers.json"
    return None


def _append_markers(path: Path | None, lines: list[str]) -> list[str]:
    """Append `lines` to the markers file (a JSON list, created when absent); the lines to
    print, with a `WARN:` when the file cannot be written or is not a JSON list — it is then
    left as it is and the markers are only printed, never a traceback, never a lost file."""
    if path is None:
        return lines
    doc, error = read_json(path, "the markers file")
    if not isinstance(doc, list) and (path.exists() or path.is_symlink()):
        why = error or f"it holds a JSON {type(doc).__name__}, not a list"
        return [*lines, f"WARN: the markers file {path} is not a JSON list ({why}); it is left "
                        "as it is, and these markers are only printed"]
    old = doc if isinstance(doc, list) else []
    try:
        write_json(path, old + [f"extract_facts.py: {m}" for m in lines])
    except (OSError, ValueError) as exc:  # ValueError: a NaN the list held; nothing written
        return [*lines, f"WARN: the markers could not be appended to {path} ({exc}); they are "
                        "only printed"]
    return lines


def _run(args) -> tuple[list[str], dict]:
    catalogs = Catalogs.load()
    model = ModelPage.open(args.model)
    try:
        store = BundleRecords.open(Path(args.bundle))
    except ShowError as exc:  # BundleAltered, or a manifest or record that does not read
        raise Refusal("PF2", Path(args.bundle) / "MANIFEST.json", str(exc)) from exc
    _refuse_nonfinite(store.manifest(), "PF2", "MANIFEST.json")
    lines = ["OK: PF2 every pinned record matches its MANIFEST.files digest",
             *preflight(model, store)]
    report = report_facts(args, Path(args.model), store)
    layer = model.layer_data("bundle") or {}
    context = ExtractContext(_header(args, model, store), tuple(layer.get("markers") or ()))
    extractor = FactExtractor(store, report, catalogs, context)
    doc = extractor.extract().to_json()
    _refuse_nonfinite(doc, "PF2", str(args.out))  # a backstop: group() nulls them first
    problems = _schema_problems(doc)
    if problems:
        raise Refusal("PF2", Path(args.bundle) / "MANIFEST.json", "the facts read from the "
                      "bundle's records do not validate vega/show-facts@1 (a record carries a "
                      "value the fact-set schema does not admit): " + "; ".join(problems[:5]))
    write_json(args.out, doc)
    lines += [*extractor.markers, f"OK: PF7 the task, family, blocks and sample shape are the "
              f"vega-forge catalog's; {len(doc['facts'])} facts → {args.out}"]
    return lines, doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the model page <stem>.pack.html")
    ap.add_argument("--bundle", required=True, help="the bundle directory")
    ap.add_argument("--report", help="the report: its docs/ directory or its PDF")
    ap.add_argument("--label-roles", help="the label-role map (vega/label-roles@1)")
    ap.add_argument("--out", required=True, help="where facts.json goes")
    ap.add_argument("--markers", help="the JSON list the markers are appended to")
    ap.add_argument("--json", action="store_true", help="print a machine result")
    args = ap.parse_args(argv)
    try:
        lines, doc = _run(args)
        code = 0
    except (ShowError, OSError) as exc:  # a Refusal names its gate: "ERROR: PF4 …"; any
        # other is the tool failing (its catalogs, or --out not writable), not a refusal
        lines, doc, code = [f"ERROR: {exc}"], None, 2
    lines = _append_markers(_markers_path(args), lines)
    if args.json:
        print(json.dumps({"status": "ok" if code == 0 else "refused", "out": args.out,
                          "facts": len(doc["facts"]) if doc else 0, "markers": lines}))
    else:
        print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
