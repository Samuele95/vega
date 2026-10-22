"""
draft_io — the IO edge of `draft_plan.py`: what the command line reads beside the fact set, and
the markers file it appends to. `PlanDrafter` itself reads no file.

- `load_facts(path)` — the fact set (`vega/show-facts@1`), refused when it does not read or is
  not one;
- `template_digests(arg)` — the page template's `{name, version, manifest_digest}`
  (`page_template.digests`: the installed `TEMPLATE.json`, or the manifest a fresh `--write`
  gives while none is installed); `arg` is `TEMPLATE.json` or its directory (lanekit passes the
  file as `--page-manifest`, R-E2's replacement of D5's page manifest);
- `report_labels(facts, facts_path)` — the labels of a report the facts found without a role
  map: the `.aux` the fact set pinned (a relative path looked for from the working directory,
  beside the model page and under the fact set's directories), refused (gate PF6) when it is not
  found, no longer reads or its digest moved — the facts are older than the report, so
  `extract_facts.py` runs again;
- `block_types(facts, facts_path, bundle)` — the network's distinct block types in forward
  order, composites' inner blocks included (`page_derive.block_types`, the walk's own reading),
  from the pinned `netspec.json` of the bundle the facts name (`--bundle`, else the header's
  directory as given, from the working directory, or from the facts file's directories upward),
  used only when that bundle's `MANIFEST.json` digest is the one the facts recorded; otherwise
  None and a `WARN:` (the walk then reads the fact set's top-level blocks);
- `load_plan(path)` — an earlier story plan to carry (`vega/story-plan@1` or `@2`);
- `refuse_invalid(plan, schema)` — the draft against `story-plan-2.schema.json` (a draft its own
  schema rejects is a defect of the draft, never written);
- `markers_path`, `append_markers` — the showcase's `.work/markers.json` (a JSON list; anything
  else is left as it is, with a `WARN:`); `open_forks(plan)` — the forks the model still asks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import page_template
from fact_sink import Refusal
from page_derive import block_types as walk_block_types
from propose_roles import s2_open
from records import BundleRecords
from report_labels import AuxLabelSource, ReportUnreadable
from vega_show_core import CATALOG, FactRef, FactSet, ShowError, read_json, thaw, write_json

__all__ = ["load_facts", "template_digests", "report_labels", "block_types", "load_plan",
           "refuse_invalid", "open_forks", "markers_path", "append_markers"]

_RULE = "draft"
_PLANS = ("vega/story-plan@1", "vega/story-plan@2")


def load_facts(path: Path) -> FactSet:
    doc, error = read_json(path, "the fact set")
    if error or not isinstance(doc, dict):
        raise Refusal(_RULE, path, error or "the fact set is not a JSON object")
    try:
        return FactSet.from_json(doc)
    except (ValueError, KeyError, TypeError) as exc:
        raise Refusal(_RULE, path, f"not a vega/show-facts@1 fact set ({exc}) — run "
                                   "extract_facts.py") from exc


def template_digests(arg) -> dict:
    where = page_template.TEMPLATE_DIR if arg is None else Path(arg)
    folder = where if where.is_dir() else where.parent
    if not folder.is_dir() or not (folder / "index.html").is_file():
        raise Refusal(_RULE, where, "no page template there (its index.html and TEMPLATE.json)")
    try:
        return page_template.digests(folder)
    except (page_template.TemplateError, OSError, ValueError) as exc:
        raise Refusal(_RULE, folder, f"the page template cannot be pinned: {exc}") from exc


def report_labels(facts: FactSet, facts_path: Path | None = None) -> Mapping | None:
    report = thaw(facts.groups).get("report") or {}
    if report.get("status") != "linked" or report.get("roles_source"):
        return None
    aux = report.get("aux") or {}
    source, recorded = str(aux.get("source")), aux.get("digest")
    found = [c for c in _candidates(source, thaw(facts.header), facts_path) if c.is_file()]
    if not found:
        raise Refusal("PF6", source, "the report's .aux the facts were extracted from is not "
                                     "found — re-run extract_facts.py")
    try:
        sources = [AuxLabelSource(path) for path in found]
    except ReportUnreadable as exc:
        raise Refusal("PF6", source, f"{exc} — re-run extract_facts.py") from exc
    same = next((s for s in sources if s.digest() == recorded), None)
    if same is None:
        raise Refusal("PF6", source, "the report's .aux changed since the facts were extracted "
                                     "— re-run extract_facts.py")
    return same.labels()


def _candidates(source: str, header: Mapping, facts_path: Path | None) -> list[Path]:
    """Where a recorded relative path may be: as given (the working directory), then beside
    the model page and under the fact set's directories."""
    path = Path(source)
    if path.is_absolute():
        return [path]
    model = str(((header or {}).get("model") or {}).get("path") or "")
    roots = [Path.cwd(), *([Path(model).resolve().parent] if model else []),
             *(Path(facts_path).resolve().parents if facts_path else [])]
    return list(dict.fromkeys(root / path for root in roots))


def block_types(facts: FactSet, facts_path: Path, bundle=None) -> tuple[tuple | None, str | None]:
    header = thaw(facts.header).get("bundle") or {}
    for folder in _bundle_candidates(header.get("dir"), Path(facts_path), bundle):
        if not (folder / "MANIFEST.json").is_file():
            continue
        try:
            store = BundleRecords.open(folder)
            if store.manifest_digest() != header.get("manifest_digest"):
                continue
            return tuple(walk_block_types(store.read(FactRef.parse("netspec.json#blocks")))), None
        except ShowError:
            continue
    return None, ("WARN: the bundle the facts describe was not found (pass --bundle); the code "
                  "steps name the network's top-level block types only")


def _bundle_candidates(recorded, facts_path: Path, override) -> list[Path]:
    if override:
        return [Path(override)]
    if not recorded:
        return []
    path = Path(str(recorded))
    if path.is_absolute():
        return [path]
    return [Path.cwd() / path, *(parent / path for parent in facts_path.resolve().parents)]


def load_plan(path) -> dict:
    doc, error = read_json(path, "the plan to carry")
    if error or not isinstance(doc, dict) or doc.get("schema") not in _PLANS:
        raise Refusal(_RULE, path, error or f"not a story plan ({' or '.join(_PLANS)})")
    return doc


def refuse_invalid(plan: dict, schema: Path) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ShowError("jsonschema (draft 2020-12) is needed to validate the draft before it is "
                        "written; install jsonschema>=4") from exc
    doc = json.loads(Path(schema).read_text(encoding="utf-8"))
    problems = [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}"
                for e in Draft202012Validator(doc).iter_errors(plan)]
    if problems:
        raise Refusal(_RULE, "story-plan", "the draft does not validate vega/story-plan@2 (a "
                      "defect of the draft): " + "; ".join(problems[:5]))


def open_forks(plan: dict | None) -> list[str]:
    """The forks the model still asks: S3 while no motif is authored, S2 while no authored role
    rests on a `user:` turn or a `show:` stance (`propose_roles.s2_open`), S1 (no report:
    continue with the brief links, or document first) whenever the proposal names it."""
    if not plan:
        return []
    props, authored = plan["proposals"], plan["authored"]
    out = []
    if (props.get("report_roles") or {}).get("fork"):
        out.append(props["report_roles"]["fork"])
    if s2_open(props.get("class_roles"), authored.get("class_roles")):
        out.append(props["class_roles"]["fork"])
    if (props.get("motif") or {}).get("fork") and authored.get("motif") is None:
        out.append(props["motif"]["fork"])
    return sorted(out)


def markers_path(markers, out) -> Path | None:
    """`--markers`, else the `.work/markers.json` of the showcase directory `out` lies in (its
    `.work/` or `plan/`)."""
    if markers:
        return Path(markers)
    out = Path(out).resolve()
    siblings = (CATALOG["showcase"]["dir"], *CATALOG["showcase"]["siblings"].values())
    if out.parent.name in (".work", "plan") and out.parent.parent.name in siblings:
        return out.parent.parent / ".work" / "markers.json"
    return None


def append_markers(path: Path | None, lines: list[str]) -> list[str]:
    """Append `lines` to the markers list at `path`; the lines to print, with a `WARN:` when the
    file is not a JSON list or cannot be written (it is then left as it is)."""
    if path is None:
        return lines
    doc, error = read_json(path, "the markers file")
    if not isinstance(doc, list) and (path.exists() or path.is_symlink()):
        why = error or f"it holds a JSON {type(doc).__name__}, not a list"
        return [*lines, f"WARN: the markers file {path} is not a JSON list ({why}); it is left "
                        "as it is, and these markers are only printed"]
    try:
        write_json(path, (doc if isinstance(doc, list) else []) +
                   [f"draft_plan.py: {m}" for m in lines])
    except (OSError, ValueError) as exc:
        return [*lines, f"WARN: the markers could not be appended to {path} ({exc}); they are "
                        "only printed"]
    return lines
