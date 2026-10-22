#!/usr/bin/env python3
"""
check_brief.py — a stage brief is what vega/references/brief-contract.md says (D6 §11.4).

    python3 check_brief.py BRIEF.md ... [--root DIR] [--json OUT]

For each `<run-dir>/docs/<stage>-brief.md`:
  - the frontmatter (between `---` lines) holds stage, skill, lane, input, output, run_dir,
    layer, date, records and supersedes; the stage is a chain-map stage and the file is named
    after it; the non-nullable fields are not empty (layer and supersedes may be null);
  - records is a non-empty list of {path, sha256}; every path resolves (as written, against
    --root, against the project root the brief's run_dir implies, against the run directory)
    and its sha256 equals the file's (`sha256:` prefix optional);
  - the body's `##` headings are the seven sections, in this order, each with content: What this
    stage was for, What was decided, What ran (with a fenced block of the commands), What came
    out, What the reader should check, Open ("Nothing open." when empty), Next;
  - when the stage's output HTML resolves (its `output`, found as a record path is) and its
    `layer` records open items — the confirmed judgement findings of the conformance gate,
    advisory (`conformance.<tool>.advisory.open_items`, vega-forge code-contract §5) — the
    Open section names each one as `<tool> <audit>:<id>` (`architect 1:A-3`): the open item a
    stage hands to /architect and /improve appears in its brief.

Exit 0: every brief holds. 1: a brief does not (the brief and the defect are named). 2: no brief
given, or a brief cannot be read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

SECTIONS = ("What this stage was for", "What was decided", "What ran", "What came out",
            "What the reader should check", "Open", "Next")
FIELDS = ("stage", "skill", "lane", "input", "output", "run_dir", "layer", "date", "records", "supersedes")
NULLABLE = frozenset({"layer", "supersedes"})
STAGES = frozenset({"forge-data", "lens", "shape", "blueprint", "net", "train", "eval", "pack", "show"})
HEX = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")


class Unreadable(Exception):
    """No brief, or a brief that cannot be read (exit 2)."""


# ---- the frontmatter: the YAML subset the contract writes ------------------------------------------------------
def _scalar(text: str):
    text = re.sub(r"\s+#.*$", "", text).strip()
    if text in ("", "null", "~"):
        return None
    return text[1:-1] if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'" else text


def frontmatter(text: str) -> tuple[dict | None, str]:
    """(fields, body): `key: value` lines and one list of mappings (`- path: …` / `  sha256: …`)."""
    m = re.match(r"^---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", text, re.S)
    if not m:
        return None, text
    fields, current = {}, None
    for line in m.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = re.match(r"^\s*-\s+(\w+):\s*(.*)$", line)
        pair = re.match(r"^(\s*)(\w+):\s*(.*)$", line)
        if item and isinstance(current, list):
            current.append({item.group(1): _scalar(item.group(2))})
        elif pair and pair.group(1) and isinstance(current, list) and current:
            current[-1][pair.group(2)] = _scalar(pair.group(3))
        elif pair and not pair.group(1):
            value = _scalar(pair.group(3))
            fields[pair.group(2)] = [] if value is None and pair.group(2) == "records" else value
            current = fields[pair.group(2)] if isinstance(fields[pair.group(2)], list) else None
    return fields, text[m.end():]


# ---- the checks --------------------------------------------------------------------------------------------
def field_findings(fields: dict, brief: Path) -> list[str]:
    out = [f"frontmatter has no {f}" for f in FIELDS if f not in fields]
    out += [f"frontmatter {f} is empty" for f in FIELDS
            if f in fields and f not in NULLABLE and f != "records" and not fields[f]]
    stage = fields.get("stage")
    if stage and stage not in STAGES:
        out.append(f"stage {stage!r} is not a chain-map stage ({', '.join(sorted(STAGES))})")
    if stage and brief.name != f"{stage}-brief.md":
        out.append(f"the brief of stage {stage} is named {brief.name}, not {stage}-brief.md")
    return out


def _candidates(path: str, brief: Path, fields: dict, root: Path | None) -> list[Path]:
    """Where a record path may live: as written, under --root, under the project root the brief's
    run_dir implies (<root>/<run_dir>/docs/<brief>), under the run directory itself."""
    run_dir = brief.resolve().parent.parent
    out = [Path(path)] + ([root / path] if root else [])
    parts = PurePosixPath(str(fields.get("run_dir") or "")).parts
    if parts and run_dir.parts[-len(parts):] == parts:
        out.append(Path(*run_dir.parts[:-len(parts)]) / path)
    return out + [run_dir / path]


def record_findings(fields: dict, brief: Path, root: Path | None) -> list[str]:
    records = fields.get("records")
    if not isinstance(records, list) or not records:
        return ["records is not a non-empty list: every number of a brief resolves to a record"]
    out = []
    for i, rec in enumerate(records):
        path, want = rec.get("path"), HEX.match(str(rec.get("sha256") or "").strip())
        if not path or not want:
            out.append(f"records[{i}] needs a path and a sha256 (64 hex digits), got {rec}")
            continue
        found = next((p for p in _candidates(path, brief, fields, root) if p.is_file()), None)
        if found is None:
            out.append(f"records[{i}] {path}: no such file")
        elif hashlib.sha256(found.read_bytes()).hexdigest() != want.group(1):
            out.append(f"records[{i}] {path}: its sha256 differs from the file's (the record changed after the brief)")
    return out


def _sections(body: str) -> tuple[list[str], dict]:
    parts = re.split(r"^##[ \t]+(.+?)[ \t]*$", body, flags=re.M)
    heads, texts = [h.strip() for h in parts[1::2]], [t.strip() for t in parts[2::2]]
    return heads, dict(zip(heads, texts))


def layer_open_items(html: str, layer: str) -> list[str]:
    """`<tool> <audit>:<id>` of every open item the layer's conformance records (the confirmed,
    advisory judgement findings); [] when the layer is absent or unreadable."""
    m = re.search(r'<script\b[^>]*\bid="layer-' + re.escape(layer) + r'-data"[^>]*>([\s\S]*?)</script>',
                  html)
    try:
        doc = json.loads(m.group(1)) if m else {}
    except json.JSONDecodeError:
        return []
    conformance = doc.get("conformance") if isinstance(doc, dict) else None
    records = conformance.values() if isinstance(conformance, dict) else ()
    return [f"{item.get('tool')} {item.get('finding')}" for rec in records if isinstance(rec, dict)
            for item in (rec.get("advisory") or {}).get("open_items") or [] if isinstance(item, dict)]


def open_item_findings(fields: dict, body: str, brief: Path, root: Path | None) -> list[str]:
    output, layer = fields.get("output"), fields.get("layer")
    if not output or not layer:
        return []
    found = next((p for p in _candidates(str(output), brief, fields, root) if p.is_file()), None)
    if found is None:
        return []
    items = layer_open_items(found.read_text(encoding="utf-8", errors="replace"), str(layer))
    said = _sections(body)[1].get("Open") or ""
    return [f"section \"Open\" does not name the open item {key} the {layer} layer records (a "
            "confirmed judgement finding handed to /architect and /improve)"
            for key in items if key not in said]


def section_findings(body: str) -> list[str]:
    heads, content = _sections(body)
    if heads != list(SECTIONS):
        return [f"the sections are {heads}, not the seven in order: {list(SECTIONS)}"]
    out = [f"section \"{h}\" is empty" for h in SECTIONS if not content[h]]
    if content["What ran"] and "```" not in content["What ran"]:
        out.append("section \"What ran\" has no fenced block of the commands")
    return out


def check(brief: Path, root: Path | None) -> list[str]:
    try:
        text = brief.read_text(encoding="utf-8")
    except OSError as exc:
        raise Unreadable(f"cannot read {brief}: {exc}") from exc
    fields, body = frontmatter(text)
    if fields is None:
        return ["no frontmatter (a `---` block first)"] + section_findings(body)
    return (field_findings(fields, brief) + record_findings(fields, brief, root) + section_findings(body)
            + open_item_findings(fields, body, brief, root))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("briefs", nargs="*", type=Path)
    ap.add_argument("--root", type=Path, default=None, help="the project root record paths are relative to")
    ap.add_argument("--json", default=None, help="write {brief: findings} here ('-': stdout)")
    a = ap.parse_args(argv)
    try:
        if not a.briefs:
            raise Unreadable("no brief given (check_brief.py <run>/docs/*-brief.md)")
        findings = {str(b): check(b, a.root) for b in a.briefs}
    except Unreadable as exc:
        print(f"ERROR: {exc}")
        return 2
    bad = [b for b, p in findings.items() if p]
    if a.json:
        text = json.dumps({"schema": "vega/brief-check@1", "findings": findings}, indent=1) + "\n"
        if a.json == "-":
            sys.stdout.write(text)
            return 1 if bad else 0
        Path(a.json).write_text(text, encoding="utf-8")
    for brief, problems in findings.items():
        for p in problems:
            print(f"FAIL: {brief}: {p}")
    if bad:
        print(f"FAIL: {len(bad)} of {len(findings)} brief(s) break the brief contract")
        return 1
    print(f"OK: {len(findings)} brief(s): frontmatter, records and digests, seven sections in order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
