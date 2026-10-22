"""
facts_report — `ReportFacts`: where the page's code notes can point (contract §6, the groups
`report` and `briefs`), and `brief_paths()`, where each stage's brief is.

A code note links into the report through a **role**, never through a typed number: the role map
(`vega/label-roles@1`, read and checked against its `.aux` by `role_map.read_role_map`, gate PF6)
names the label, and the `.aux` gives its number and page. Each bound role becomes two facts,
`report_<role>_number` and `report_<role>_page` (`report:net.forward.number`: the role's dots and
dashes become `_`), and one entry of the `report` group with the label, its kind and its title.

- **Linked**: a compiled report (its `.aux`) exists. With a role map, the roles above; without one
  (a hand-authored report), no role is bound yet — the draft proposes them from the `.aux`
  titles — with `WARN: label roles proposed from the .aux titles …`.
- **Brief links**: no report. Each code note will link to the brief of the stage that documents
  it, copied into the showcase; `WARN: no report …` (fork S1's unattended answer is these links).

The `briefs` group is always written: `{stage: {path, digest}}` for every stage brief that exists
where the records say its stage ran (`catalog.briefs.stages`); a brief that is not there is named
in one `WARN:`. A relative run directory is resolved against the project roots the caller gives
(the directory that holds `bundle/`, the model's directory), in that order.

`report_facts(args, model, store)` is the adapter the CLI's arguments name (gate PF6): a role
map (`--label-roles`, else `label-roles.json` in or beside `--report`, else in a `docs/` beside
the model, in the project that holds `bundle/`, or in the working directory), else a compiled
report (a PDF with a same-stem `.aux`) without roles, else the brief links.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from fact_sink import FactSink, Refusal, peek
from report_labels import AuxLabelSource, ReportUnreadable
from role_map import RoleMap, read_role_map, resolve_aux
from show_ports import LabelSource, RecordStore
from vega_show_core import CATALOG, digest_file, fact_key, read_json

__all__ = ["ReportFacts", "brief_paths", "report_facts"]

_BRIEF_DIRS = {"shape": "records/shape.json#manifest.outputs.dir",
               "net": "records/architecture.json#netspec.outputs.run_dir",
               "train": "records/training.json#campaign.run_dir",
               "eval": "records/training.json#campaign.run_dir"}
_PIPELINE = "records/architecture.json#netspec.input.pipeline"


def brief_paths(records: RecordStore, bundle: Path, roots: Sequence[Path]) -> dict[str, Path]:
    """{stage: path of its brief} for every stage of `catalog.briefs.stages`, found or not."""
    out: dict[str, Path] = {}
    for stage in CATALOG["briefs"]["stages"]:
        name = f"{stage}-brief.md"
        if stage == "pack":
            out[stage] = Path(bundle) / "docs" / name
            continue
        run_dir = peek(records, _BRIEF_DIRS[stage]) if stage in _BRIEF_DIRS else None
        if stage == "blueprint":
            pipeline = peek(records, _PIPELINE)
            run_dir = str(PurePosixPath(pipeline).parent) if isinstance(pipeline, str) else None
        if isinstance(run_dir, str) and run_dir.strip():
            out[stage] = _resolve(Path(run_dir) / "docs" / name, roots)
    return out


def _resolve(path: Path, roots: Sequence[Path]) -> Path:
    if path.is_absolute():
        return path
    candidates = [Path(r) / path for r in roots]
    return next((c for c in candidates if c.is_file()), candidates[0] if candidates else path)


class ReportFacts:
    """The report's label facts, or the brief links that stand in for them."""

    def __init__(self, labels: LabelSource | None, roles: RoleMap | None,
                 briefs: Mapping[str, Path], pdf: Path | None = None) -> None:
        self._labels, self._roles, self._briefs, self._pdf = labels, roles, dict(briefs), pdf

    def collect(self, sink: FactSink) -> None:
        self._briefs_group(sink)
        if self._labels is None:
            sink.warn(CATALOG["forks"]["S1"]["marker"])
            sink.group("report", {"status": "brief-links", "roles": {}, "pdf": None, "aux": None,
                                  "roles_source": None})
            return
        roles = self._bound(sink) if self._roles is not None else {}
        if self._roles is None:
            sink.warn("label roles proposed from the .aux titles — /document-project in Vega mode "
                      "emits them")
        aux = getattr(self._labels, "path", None)
        sink.group("report", {
            "status": "linked", "roles": roles,
            "pdf": self._file(self._pdf),
            "aux": {"source": str(aux) if aux else None, "digest": self._labels.digest()},
            "roles_source": None if self._roles is None else self._roles_source()})

    def _roles_source(self) -> dict:
        source = Path(self._roles.source)
        return {"source": str(source), "origin": self._roles.origin,
                "digest": digest_file(source) if source.is_file() else None,
                "aux_digest": self._roles.aux_digest}

    def _bound(self, sink: FactSink) -> dict:
        out = {}
        for binding in self._roles.bindings:
            base = fact_key(("report", binding.role))
            keys = {}
            for field, kind in (("number", "report_number"), ("page", "report_page")):
                value = getattr(binding, field)
                ref = f"report:{binding.role}.{field}"
                if value is not None and value != "" and sink.vacant(f"{base}_{field}", ref):
                    keys[field] = sink.add(f"{base}_{field}", ref, value, kind)
            out[binding.role] = {"label": binding.label, "kind": binding.kind,
                                 "number": keys.get("number"), "page": keys.get("page"),
                                 "title": binding.title}
        return out

    def _briefs_group(self, sink: FactSink) -> None:
        found, missing = {}, []
        for stage, path in self._briefs.items():
            if Path(path).is_file():
                found[stage] = {"path": str(path), "digest": digest_file(path)}
            else:
                missing.append(f"{stage} ({path})")
        if missing:
            sink.warn(f"no stage brief for {', '.join(missing)} — a code note of that stage has no "
                      "brief to link to")
        sink.group("briefs", found)

    @staticmethod
    def _file(path: Path | None) -> dict | None:
        if path is None or not Path(path).is_file():
            return None
        return {"source": str(path), "digest": digest_file(path)}


# --------------------------------------------------------------------------- the report

def report_facts(args, model: Path, store: RecordStore) -> ReportFacts:
    """The report adapter the arguments name (gate PF6 on a role map), with the briefs."""
    roots = [Path(args.bundle).resolve().parent.parent, model.resolve().parent, Path.cwd()]
    briefs = brief_paths(store, Path(args.bundle), roots)
    roles_path, aux = _role_map_path(args, model), None
    try:
        if roles_path is not None:
            roles, problems = read_role_map(roles_path)
            if problems:
                raise Refusal("PF6", problems[0].where, "; ".join(p.message for p in problems))
            doc, _ = read_json(roles_path, "the label-role map")
            pdf = resolve_aux(roles_path, doc["report"]["pdf"])
            aux = resolve_aux(roles_path, roles.aux)
            return ReportFacts(AuxLabelSource(aux), roles, briefs, pdf)
        pdf, aux = _compiled_report(args, model)
        if aux is None and args.report:
            raise Refusal("PF6", args.report, "--report names no compiled report (a PDF with a "
                          "same-stem .aux) — compile the report, or leave --report out")
        return ReportFacts(AuxLabelSource(aux) if aux else None, None, briefs, pdf)
    except ReportUnreadable as exc:
        raise Refusal("PF6", aux, str(exc)) from exc


def _role_map_path(args, model: Path) -> Path | None:
    if args.label_roles:
        path = Path(args.label_roles)
        if not path.is_file():
            raise Refusal("PF6", path, "the label-role map named by --label-roles does not exist")
        return path
    report = Path(args.report) if args.report else None
    candidates = ([report / "label-roles.json"] if report and report.is_dir() else
                  [report.parent / "label-roles.json"] if report else [])
    candidates += [d / "label-roles.json" for d in _docs_dirs(args, model)]
    return next((c for c in candidates if c.is_file()), None)


def _docs_dirs(args, model: Path) -> list[Path]:
    """Where a report is looked for without --report: `docs/` beside the model, in the project
    that holds the bundle (`<root>/bundle/<slug>-<run>`), and in the working directory."""
    dirs = [model.parent / "docs", Path(args.bundle).resolve().parent.parent / "docs",
            Path.cwd() / "docs"]
    return [d for i, d in enumerate(dirs) if d.resolve() not in {e.resolve() for e in dirs[:i]}]


def _compiled_report(args, model: Path) -> tuple[Path | None, Path | None]:
    report = Path(args.report) if args.report else None
    if report is not None and report.suffix == ".pdf":
        pdfs = [report]
    else:
        dirs = [report] if report is not None else _docs_dirs(args, model)
        pdfs = [p for d in dirs if d.is_dir() for p in sorted(d.glob("*.pdf"))]
    found = next((p for p in pdfs if p.with_suffix(".aux").is_file()), None)
    return (found, found.with_suffix(".aux")) if found else (None, None)
