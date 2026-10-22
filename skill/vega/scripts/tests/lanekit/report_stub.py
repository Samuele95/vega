"""
The stub report a lane with `report: stub` binds its code notes to (BUILD-PLAN R-E7, D6 §6.7).

A lane cannot run LaTeX (no toolchain on the matrix machines), yet the show stage must be proved
in its linked mode: every code-walk note resolving, through a label role, to a number and a page
of a real report. So this module writes what `/document-project` in Vega mode would leave in
`docs/` after a compile, directly and deterministically:

- `main.aux`: one `\\newlabel{vega:<role>}{{3.<i>}{<page>}{<role>}{<kind>.3.<i>}{}}` per role,
  the anchors carrying the `section.`, `figure.`, `table.` and `equation.` prefixes the `.aux`
  adapter (H01's `AuxLabelSource`) reads the kind from, each followed by the `@cref` twin
  cleveref writes (which the adapter skips);
- `main.pdf`: a valid PDF of N pages, a title page and then one Helvetica text object per page
  naming the roles whose objects sit on it (pages 2..N are spread over the roles);
- `main.tex`: the source the map points at — a stub that labels every role, never compiled;
- `label-roles.json` (`vega/label-roles@1`, the vega-show schema): the role → label map with
  each role's kind, stage, number, page and title, and the digest of the `.aux` it was made
  from. It is written last, so the map is never older than its `.aux`.

    report_stub.write(docs_dir, roles, options=None) -> dict   # the label-roles document it wrote
    report_stub.StubOptions(lane=, pages=, skills=, generated_at=)  # every field optional
    python -m lanekit.report_stub --docs <dir> [--lane ID] [--skills S] [--pages N]

`roles` is what `contracts.REPORT_ROLES()` returns (the role names, in the vega-show catalog's
order); each role's kind and stage are read from the same catalog. A mapping `{role: {kind,
stage}}` is taken as it is. Re-running with another `pages` moves the page numbers, which is how
a lane makes a showcase stale on purpose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import contracts

CHAPTER = 3                                  # every role's object lives in chapter 3: "3.<i>"
LABEL_PREFIX = "vega:"
ADAPTER = "latex-aux"
KIND_ANCHOR = {"section": "section", "figure": "figure", "table": "table", "equation": "equation"}
ROLES_PER_PAGE = 3                           # the default spread: N = 1 + ceil(roles / 3)
FILES = {"tex": "main.tex", "pdf": "main.pdf", "aux": "main.aux", "roles": "label-roles.json"}
PAGE_SIZE = (612, 792)                       # US letter, in points


@dataclass(frozen=True)
class RoleLabel:
    """One role's labelled object in the stub report."""
    role: str
    kind: str
    stage: str
    number: str
    page: int

    @property
    def label(self) -> str:
        return LABEL_PREFIX + self.role

    @property
    def anchor(self) -> str:
        return f"{KIND_ANCHOR[self.kind]}.{self.number}"

    @property
    def title(self) -> str:
        return self.role

    def binding(self) -> dict:
        return {"label": self.label, "kind": self.kind, "stage": self.stage, "number": self.number,
                "page": self.page, "title": self.title}


# --------------------------------------------------------------------------- the plan
def catalog_roles(skills: Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    """The vega-show catalog's `report_roles` ({role: {kind, stage, keywords}})."""
    path = (skills or contracts.SKILLS_ROOT) / contracts.SHOW_CATALOG
    return json.loads(path.read_text(encoding="utf-8"))["report_roles"]


def resolve_roles(roles: Sequence[str] | Mapping[str, Mapping[str, Any]],
                  skills: Path | None = None) -> list[tuple[str, str, str]]:
    """(role, kind, stage) for every role, in the given order."""
    if isinstance(roles, Mapping):
        table, names = roles, list(roles)
    else:
        table, names = catalog_roles(skills), list(roles)
    if not names:
        raise ValueError("a stub report needs at least one role")
    out = []
    for name in names:
        entry = table.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"report role {name!r} is not in the vega-show catalog's report_roles")
        kind, stage = entry.get("kind"), entry.get("stage")
        if kind not in KIND_ANCHOR:
            raise ValueError(f"report role {name!r} has kind {kind!r}, not one of {sorted(KIND_ANCHOR)}")
        out.append((name, kind, stage))
    return out


def layout(roles: Sequence[tuple[str, str, str]], pages: int | None = None) -> tuple[list[RoleLabel], int]:
    """Number the roles 3.1, 3.2, … and spread them over pages 2..N (page 1 is the title page)."""
    count = len(roles)
    total = pages if pages is not None else 1 + math.ceil(count / ROLES_PER_PAGE)
    if total < 2:
        raise ValueError(f"a stub report has a title page and at least one content page, not {total}")
    body = total - 1
    labels = [RoleLabel(role, kind, stage, f"{CHAPTER}.{i + 1}", 2 + (i * body) // count)
              for i, (role, kind, stage) in enumerate(roles)]
    return labels, total


# --------------------------------------------------------------------------- the files
def aux_text(labels: Sequence[RoleLabel], pages: int) -> str:
    """The `.aux` a hyperref + cleveref compile leaves: a `\\newlabel` with five groups per
    label, then its `@cref` twin."""
    lines = ["\\relax", "\\providecommand\\hyper@newdestlabel[2]{}"]
    for i, lab in enumerate(labels, start=1):
        lines.append(f"\\newlabel{{{lab.label}}}{{{{{lab.number}}}{{{lab.page}}}{{{lab.title}}}"
                     f"{{{lab.anchor}}}{{}}}}")
        lines.append(f"\\newlabel{{{lab.label}@cref}}{{{{[{lab.kind}][{i}][{CHAPTER}]{lab.number}}}"
                     f"{{[1][{lab.page}][]{lab.page}}}}}")
    lines.append(f"\\gdef \\@abspage@last{{{pages}}}")
    return "\n".join(lines) + "\n"


TEX_ENV = {"section": None, "figure": "figure", "table": "table", "equation": "equation"}


def tex_text(labels: Sequence[RoleLabel], lane: str) -> str:
    """The source the role map points at. It labels every role, and says it is a stub: the
    `.aux` and the PDF beside it are written directly, never compiled from it."""
    out = ["% lanekit report stub (vega/scripts/tests/lanekit/report_stub.py).",
           "% main.aux and main.pdf are written directly by the stub, never compiled from this file.",
           "\\documentclass{report}", "\\usepackage{amsmath}", "\\usepackage{hyperref}",
           "\\usepackage{cleveref}", "\\begin{document}",
           f"\\title{{Stub report of lane {_tex_escape(lane)}}}", "\\maketitle",
           f"\\setcounter{{chapter}}{{{CHAPTER - 1}}}", "\\chapter{Report roles}"]
    for lab in labels:
        env = TEX_ENV[lab.kind]
        if env is None:
            out.append(f"\\section{{{_tex_escape(lab.title)}}}\\label{{{lab.label}}}")
        elif env == "equation":
            out.append(f"\\begin{{equation}}\\label{{{lab.label}}} y = f(x) \\end{{equation}}")
        else:
            out.append(f"\\begin{{{env}}}\\caption{{{_tex_escape(lab.title)}}}\\label{{{lab.label}}}"
                       f"\\end{{{env}}}")
    out.append("\\end{document}")
    return "\n".join(out) + "\n"


def _tex_escape(text: str) -> str:
    return "".join("\\" + c if c in "&%$#_{}" else c for c in str(text))


def page_lines(labels: Sequence[RoleLabel], pages: int, lane: str) -> list[list[str]]:
    """What each PDF page says: the title page, then the roles on each content page."""
    content = [[f"Stub report of lane {lane}",
                "Written by lanekit's report_stub for the vega-show report binding (R-E7).",
                f"{len(labels)} report roles on pages 2 to {pages}."]]
    for page in range(2, pages + 1):
        here = [lab for lab in labels if lab.page == page]
        lines = [f"Page {page}"]
        lines += [f"{KIND_ANCHOR[lab.kind].capitalize()} {lab.number}: {lab.role} ({lab.label})"
                  for lab in here] or ["(no report role on this page)"]
        content.append(lines)
    return content


def pdf_bytes(pages: Sequence[Sequence[str]]) -> bytes:
    """A minimal valid PDF 1.4: a catalog, a page tree, one Helvetica font and one page per
    entry of `pages`, each with a single text object (BT … ET) holding its lines; a correct
    cross-reference table and trailer."""
    if not pages:
        raise ValueError("a PDF needs at least one page")
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: ("<< /Type /Pages /Kids [" + " ".join(f"{p} 0 R" for p in page_ids)
            + f"] /Count {n} >>").encode("ascii"),
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    width, height = PAGE_SIZE
    for page_id, lines in zip(page_ids, pages):
        stream = _text_object(lines, height)
        objects[page_id] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
                            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
                            ).encode("ascii")
        objects[page_id + 1] = (f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream
                                + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for oid in sorted(objects):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode("ascii") + objects[oid] + b"\nendobj\n"
    xref = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode("ascii") + b"0000000000 65535 f \n"
    for oid in range(1, size):
        out += f"{offsets[oid]:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").encode("ascii")
    return bytes(out)


def _text_object(lines: Sequence[str], height: int) -> bytes:
    body = [f"BT /F1 12 Tf 16 TL 72 {height - 72} Td"]
    for i, line in enumerate(lines):
        body.append(("" if i == 0 else "T* ") + f"({_pdf_escape(line)}) Tj")
    body.append("ET")
    return "\n".join(body).encode("cp1252", errors="replace")


def _pdf_escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- write
@dataclass(frozen=True)
class StubOptions:
    """What `write` may be told beyond the roles, each with its default: the lane id (the lane
    directory's name, `lanes/<id>/docs`), the PDF's page count (1 + ceil(roles / 3)), the skills
    tree whose vega-show catalog names the roles' kinds and stages (this tree), and the map's
    `generated_at` (now, UTC)."""
    lane: str | None = None
    pages: int | None = None
    skills: Path | None = None
    generated_at: str | None = None


def write(docs_dir: Path, roles: Sequence[str] | Mapping[str, Mapping[str, Any]],
          options: StubOptions | None = None) -> dict:
    """Write main.aux, main.pdf, main.tex and label-roles.json into `docs_dir` and return the
    label-roles document."""
    opts = options or StubOptions()
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    lane_id = opts.lane or docs.resolve().parent.name
    labels, total = layout(resolve_roles(roles, opts.skills), opts.pages)
    aux = aux_text(labels, total).encode("utf-8")
    (docs / FILES["aux"]).write_bytes(aux)
    (docs / FILES["pdf"]).write_bytes(pdf_bytes(page_lines(labels, total, lane_id)))
    (docs / FILES["tex"]).write_text(tex_text(labels, lane_id), encoding="utf-8")
    doc = {"schema": contracts.LABEL_ROLES, "lane": lane_id,
           "generated_at": opts.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "adapter": ADAPTER,
           "report": {"tex": f"{docs.name}/{FILES['tex']}", "pdf": f"{docs.name}/{FILES['pdf']}",
                      "aux": f"{docs.name}/{FILES['aux']}", "aux_digest": sha256_bytes(aux)},
           "roles": {lab.role: lab.binding() for lab in labels}}
    (docs / FILES["roles"]).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lanekit.report_stub",
                                 description="Write a lane's stub report (main.aux, main.pdf, main.tex, "
                                             "label-roles.json) into a docs directory.")
    ap.add_argument("--docs", required=True, type=Path)
    ap.add_argument("--lane")
    ap.add_argument("--skills", type=Path, help="the skills tree whose vega-show catalog names the roles")
    ap.add_argument("--pages", type=int, help="the PDF's page count (default 1 + ceil(roles / 3))")
    args = ap.parse_args(argv)
    try:
        doc = write(args.docs, contracts.REPORT_ROLES(args.skills),
                    StubOptions(lane=args.lane, pages=args.pages, skills=args.skills))
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    last = max(entry["page"] for entry in doc["roles"].values())
    print(f"OK: stub report for lane {doc['lane']} in {args.docs}: {len(doc['roles'])} roles on pages "
          f"2..{last}, aux {doc['report']['aux_digest'][:19]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
