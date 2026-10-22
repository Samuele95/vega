"""
showcase_inputs — what one `build_showcase.py` run reads, read once and checked before anything
is written (contract §5, §10: gates PF1, PF2, PF4–PF6, PF8, PF9 and V1–V15), and the child run
that puts the demonstration sample through the bundle (`attribution_build.py`).

    BuildInputs.load(request) -> BuildInputs         the bundle, the model page, the fact set,
                                                     the authored plan and the story (as bytes)
    BuildInputs.validate(report, template) -> (placeholders, markers)
    report_sources(inputs) -> ReportSources          what the showcase copies of the report or
                                                     the briefs, each at its recorded digest
    bundle_run(bundle, scratch) -> dict              the child's result (attribution_build)
    snapshot(bundle) -> dict                         every file outside showcase*/ (gate BD8)

**The inputs are this bundle's.** The bundle's pinned records verify against `MANIFEST.files`
(`PF2`, `records.BundleRecords`); the model page's `bundle` layer is this `MANIFEST.json` and the
round trip did not fail, the service has a showcase slot (`PF1`, `PF8`, `PF4`, as
`extract_facts.preflight` holds them, without running `predict.py --check` again: the facts
did); the fact set, and the story, name this bundle's manifest digest, and the fact set this
model page's Turtle digest (`PF1`). The authored plan is validated again (V1–V15, the
resolver's own validator: V1 holds the plan to this fact set's bytes and to the template being
copied) and resolved again: the story handed in must be byte for byte the story the plan
resolves to (rule `resolve`, as `resolve_story.py --check`), so a hand-edited `story.json` is
never pinned.

**The report is the one the facts saw** (`PF6`): in linked mode the PDF, the `.aux` and the role
map still have the digests the fact set recorded (a report without a PDF cannot be linked); in
brief-link mode every stage brief the story links has its recorded digest. A recorded relative
source is looked for in `--report` (the report's files), the working directory, the project
that holds the bundle and beside the model page.
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import extract_facts
from fact_sink import Refusal
from plan_rules import PlanPins, PlanValidator
from records import BundleRecords
from resolve_story import StoryResolver, report_labels_for, story_bytes
from vega_show_core import (CATALOG, FactSet, ShowError, digest_bytes, digest_file, read_json,
                            thaw)

__all__ = ["BuildRequest", "BuildInputs", "ReportSources", "report_sources", "bundle_run",
           "snapshot", "pinned_json", "examples_body", "relative"]

HERE = Path(__file__).resolve().parent
CHILD = HERE / "attribution_build.py"
CHILD_TIMEOUT_S = 1800
SIBLINGS = (CATALOG["showcase"]["dir"], *CATALOG["showcase"]["siblings"].values())


@dataclass(frozen=True)
class BuildRequest:
    """The command line's paths: the bundle, the model page, the plan, the story, the fact set
    and the report (`--report`, `--label-roles`), any of them None where it defaults."""

    bundle: Path
    model: Path | None = None
    plan: Path | None = None
    story: Path | None = None
    facts: Path | None = None
    report: Path | None = None


def pinned_bytes(bundle: Path, files: Mapping, rel: str) -> bytes:
    """A file the bundle pins, read once and held to its `MANIFEST.files` digest (PF2)."""
    if rel not in files:
        raise Refusal("PF2", f"MANIFEST.json#files.{rel}", "the bundle does not pin it")
    raw = (Path(bundle) / rel).read_bytes()
    if digest_bytes(raw) != files[rel]:
        raise Refusal("PF2", Path(bundle) / rel, "its bytes are not the ones MANIFEST.json pins: "
                                                 "the bundle was altered after pack")
    return raw


def pinned_json(bundle: Path, files: Mapping, rel: str):
    return json.loads(pinned_bytes(bundle, files, rel).decode("utf-8"))


def relative(path: Path, root: Path) -> str:
    """`path` relative to `root` when it lies under it, else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True, eq=False)
class BuildInputs:
    """The bundle's records and manifest, the model page, and the plan, story and fact set as
    the bytes that will be pinned."""

    request: BuildRequest
    store: BundleRecords
    model: Path
    turtle_digest: str
    facts: FactSet
    documents: Mapping = field(default_factory=dict)     # "facts", "plan", "story" -> bytes

    @property
    def bundle(self) -> Path:
        return self.store.bundle

    @property
    def manifest(self) -> dict:
        return self.store.manifest()

    @property
    def root(self) -> Path:
        """The project that holds `bundle/<slug>-<run>/`."""
        return self.bundle.resolve().parent.parent

    def json(self, name: str) -> dict:
        return json.loads(self.documents[name].decode("utf-8"))

    @staticmethod
    def load(request: BuildRequest) -> "BuildInputs":
        try:
            store = BundleRecords.open(request.bundle)
        except ShowError as exc:
            raise Refusal("PF2", Path(request.bundle) / "MANIFEST.json", str(exc)) from exc
        paths = _defaults(request)
        docs = {name: _read(path, name) for name, path in paths.items()}
        facts = _facts(docs["facts"], paths["facts"])
        model = Path(request.model or thaw(facts.header).get("model", {}).get("path") or "")
        page = extract_facts.ModelPage.open(model)
        extract_facts.preflight(page, store, check_predict=False)
        inputs = BuildInputs(request, store, model, page.turtle_digest(), facts, docs)
        inputs._same_bundle()
        return inputs

    def _same_bundle(self) -> None:
        """PF1: the fact set and the story were made from this bundle and this model page."""
        header, digest = thaw(self.facts.header), self.store.manifest_digest()
        named = {"the fact set": (header.get("bundle") or {}).get("manifest_digest"),
                 "the story": (self.json("story").get("bundle") or {}).get("manifest_digest")}
        for what, recorded in named.items():
            if recorded != digest:
                raise Refusal("PF1", self.bundle / "MANIFEST.json", f"{what} was made from another "
                              f"bundle state ({recorded}; MANIFEST.json is {digest}) — re-run "
                              "extract_facts.py, draft, resolve")
        if (header.get("model") or {}).get("turtle_digest") != self.turtle_digest:
            raise Refusal("PF1", self.model, "the fact set was extracted from another model page "
                          "— pass the matching --model, or re-run extract_facts.py")

    def validate(self, report: Path | None, template: Path) -> tuple[int, list[str]]:
        """V1–V15 on the plan, then the story it resolves to against the story handed in:
        (the story's placeholders, the resolver's markers)."""
        import draft_io
        facts_path, plan = Path(_defaults(self.request)["facts"]), self.json("plan")
        labels = report_labels_for(report, self.facts, facts_path)
        pins = PlanPins(digest_bytes(self.documents["facts"]), draft_io.template_digests(template))
        problems = PlanValidator(self.facts, pins, labels).validate(plan)
        errors = [p for p in problems if p.level == "ERROR"]
        if errors:
            raise Refusal(errors[0].rule, errors[0].where, "; ".join(
                f"{p.rule} {p.where}: {p.message}" for p in errors[:6]) + " — fix the plan and "
                "resolve again")
        resolver = StoryResolver(self.facts, labels)
        if story_bytes(resolver.resolve(plan)) != self.documents["story"]:
            raise Refusal("resolve", _defaults(self.request)["story"], "the story is not the "
                          "one this plan resolves to — resolve the plan again (story.json is "
                          "never edited)")
        return resolver.placeholders_n, [str(p) for p in problems if p.level != "ERROR"]


def _defaults(request: BuildRequest) -> dict[str, Path]:
    """The plan, story and fact set: as named, else where the stage's procedure writes them
    (`showcase.building/plan/…`, `showcase.building/story.json`)."""
    building = Path(request.bundle) / CATALOG["showcase"]["siblings"]["building"]
    return {"facts": Path(request.facts or building / "plan" / "facts.json"),
            "plan": Path(request.plan or building / "plan" / "story-plan.json"),
            "story": Path(request.story or building / "story.json")}


def _read(path: Path, what: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise Refusal("resolve", path, f"the {what} could not be read ({exc})") from exc


def _facts(raw: bytes, path: Path) -> FactSet:
    try:
        return FactSet.from_json(json.loads(raw.decode("utf-8")))
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise Refusal("resolve", path, f"not a vega/show-facts@1 fact set ({exc}) — run "
                                       "extract_facts.py") from exc


# --------------------------------------------------------------------------- the report

@dataclass(frozen=True)
class ReportSources:
    """What the showcase copies (`{showcase path: source file}`) and `SHOWCASE.json.report`."""

    copies: Mapping
    record: Mapping


def report_sources(inputs: BuildInputs) -> ReportSources:
    """The report's copy and record (linked), or the briefs' (brief links); PF6 when a source
    is not the one the facts recorded."""
    groups = thaw(inputs.facts.groups)
    report = groups.get("report") or {}
    if report.get("status") != "linked":
        return _briefs(inputs, groups.get("briefs") or {})
    pdf, aux = report.get("pdf"), report.get("aux") or {}
    if not pdf:
        raise Refusal("PF6", "report.pdf", "the report's PDF was not found when the facts were "
                      "extracted: a linked showcase serves its PDF — compile the report")
    pdf_path, aux_path = _current(inputs, pdf, "PDF"), _current(inputs, aux, ".aux")
    record = {"status": "linked",
              "pdf": {"source": relative(pdf_path, inputs.root), "file": "report/report.pdf",
                      "digest": pdf["digest"]},
              "aux": {"source": relative(aux_path, inputs.root), "digest": aux["digest"]},
              "roles": _roles(inputs, report), "briefs": None}
    return ReportSources({"report/report.pdf": pdf_path}, record)


def _roles(inputs: BuildInputs, report: Mapping) -> dict:
    catalog = list(CATALOG["report_roles"])
    source = report.get("roles_source")
    if source:
        path = _current(inputs, source, "label-role map")
        bound = sorted(r for r in (report.get("roles") or {}) if r in catalog)
        return {"source": relative(path, inputs.root), "origin": source.get("origin")
                or "document-project", "digest": source["digest"], "bound": bound,
                "missing": [r for r in catalog if r not in bound]}
    authored = (inputs.json("plan").get("authored") or {}).get("report_roles") or {}
    bound = sorted(r for r in authored if r in catalog)
    return {"source": "plan/story-plan.json", "origin": "proposed",
            "digest": digest_bytes(inputs.documents["plan"]), "bound": bound,
            "missing": [r for r in catalog if r not in bound]}


def _briefs(inputs: BuildInputs, found: Mapping) -> ReportSources:
    linked = (inputs.json("story").get("report") or {}).get("briefs") or {}
    copies, record = {}, {}
    for stage, rel in sorted(linked.items()):
        if stage not in found:
            raise Refusal("PF6", f"story.report.briefs.{stage}", "the story links a brief the "
                          "fact set did not find — re-run extract_facts.py")
        path = _current(inputs, {"source": found[stage]["path"], "digest": found[stage]["digest"]},
                        f"{stage} brief")
        copies[rel] = path
        record[stage] = {"source": relative(path, inputs.root), "file": rel,
                         "digest": found[stage]["digest"]}
    if not record:
        raise Refusal("PF6", "report.briefs", "no report and no stage brief to link the code "
                      "notes to — write the stage briefs, or compile the report")
    return ReportSources(copies, {"status": "brief-links", "pdf": None, "aux": None,
                                  "roles": None, "briefs": record})


def _current(inputs: BuildInputs, entry: Mapping, what: str) -> Path:
    """The recorded source on disk, still at its recorded digest (PF6)."""
    rec = Path(str(entry.get("source")))
    bases = [inputs.request.report] if inputs.request.report else []
    bases = [b if b.is_dir() else b.parent for b in bases]
    candidates = [b / rec.name for b in bases] + ([rec] if rec.is_absolute() else [
        base / rec for base in (Path.cwd(), inputs.root, inputs.model.parent)])
    path = next((c for c in candidates if c.is_file()), None)
    if path is None:
        raise Refusal("PF6", rec, f"the report's {what} the facts recorded is gone — re-run "
                                  "extract_facts.py")
    if digest_file(path) != entry.get("digest"):
        raise Refusal("PF6", path, f"the {what} changed since the facts were extracted — re-run "
                                   "extract_facts.py (and resolve)")
    return path


# --------------------------------------------------------------------------- the bundle run

def examples_body(inputs: BuildInputs) -> dict:
    """`/v1/examples` as `page_config.views_available` reads it: the shipped test rows, the
    demonstration sample and the presets, each read from the bundle at its pinned digest."""
    files, bundle = inputs.manifest.get("files") or {}, inputs.bundle
    rows = []
    if "examples/rows.csv" in files:
        text = pinned_bytes(bundle, files, "examples/rows.csv").decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))
    demo = pinned_json(bundle, files, "examples/demo.json") if "examples/demo.json" in files else None
    presets = pinned_json(bundle, files, "examples/presets.json") \
        if "examples/presets.json" in files else {}
    kind = (inputs.manifest.get("examples") or {}).get("kind")
    return {"schema": "vega/examples@2",
            "test": {"source": kind if kind in ("test", "synthetic") else "test", "rows": rows,
                     "history": None},
            "demo": demo, "presets": list(presets.get("presets") or [])}


def bundle_run(bundle: Path, scratch: Path) -> tuple[dict | None, str]:
    """(the child's result document, its last line) — `attribution_build.py` run with the
    bundle as its working directory, a clean environment and no bytecode written; (None, why)
    when it could not run the bundle."""
    bundle, out = Path(bundle).resolve(), Path(scratch).resolve() / "bundle-run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {"PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
           "HOME": os.environ.get("HOME", str(bundle)), "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"):
        if name in os.environ:
            env[name] = os.environ[name]
    try:
        done = subprocess.run([sys.executable, "-B", str(CHILD), "--bundle", str(bundle),
                               "--out", str(out)], cwd=bundle, env=env, capture_output=True,
                              text=True, timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, f"the bundle run passed {CHILD_TIMEOUT_S} s"
    last = ((done.stdout + done.stderr).strip().splitlines() or
            [f"exit {done.returncode}, no output"])[-1]
    if done.returncode != 0 or not out.is_file():
        return None, last
    doc, error = read_json(out, "the bundle run")
    return (doc, last) if isinstance(doc, dict) else (None, error or last)


def snapshot(bundle: Path) -> dict:
    """{relative path: (size, mtime_ns)} of every file of the bundle outside `showcase*/`."""
    out, bundle = {}, Path(bundle)
    for top, dirs, names in os.walk(bundle):
        if Path(top) == bundle:
            dirs[:] = [d for d in dirs if d not in SIBLINGS]
        for name in names:
            path = Path(top) / name
            info = path.lstat()
            out[path.relative_to(bundle).as_posix()] = (info.st_size, info.st_mtime_ns)
    return out
