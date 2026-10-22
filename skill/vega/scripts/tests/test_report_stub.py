"""
Unit tests of lanekit.report_stub (T03; BUILD-PLAN R-E7, D6 §6.7): the stub report a lane with
`report: stub` binds its code notes to.

    python -m pytest -q vega/scripts/tests/test_report_stub.py

Closing tests of T03 held here:
- `main.pdf` starts `%PDF-` with N pages (a structural parse of the file, and poppler's
  `pdfinfo` / `pdftotext` when they are installed);
- `main.aux` parses with H01's `AuxLabelSource` and yields the anchor kinds (the class is used
  when `vega-show/scripts/report_labels.py` is in the tree; the contract's grammar — show-contract
  §6.4, the catalog's `label_kinds` — is always checked by the reference reader below);
- `docs/label-roles.json` validates `vega/label-roles@1`.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import contracts, report_stub  # noqa: E402

REPO = HERE.parents[2]
PY = sys.executable
SHOW = REPO / "vega-show"
CATALOG = json.loads((SHOW / "scripts" / "catalog.json").read_text(encoding="utf-8"))
LABEL_ROLES_SCHEMA = SHOW / "references" / "schemas" / "label-roles-1.schema.json"


@pytest.fixture()
def docs(tmp_path: Path) -> Path:
    return tmp_path / "lanes" / "L02" / "docs"


@pytest.fixture()
def written(docs: Path) -> dict:
    return report_stub.write(docs, roles=contracts.REPORT_ROLES(REPO))


# --------------------------------------------------------------------------- reading a PDF
def pdf_objects(data: bytes) -> dict[int, bytes]:
    """{object number: body} read through the cross-reference table: every offset must point
    at `<n> 0 obj`, which is what a viewer relies on."""
    start = int(re.search(rb"startxref\s+(\d+)\s+%%EOF\s*$", data).group(1))
    assert data[start:start + 4] == b"xref", "startxref does not point at the xref table"
    head = re.match(rb"xref\s+0 (\d+)\s+", data[start:])
    size = int(head.group(1))
    table = data[start + head.end():]
    entries = [table[i * 20:(i + 1) * 20] for i in range(size)]
    assert entries[0].startswith(b"0000000000 65535 f"), entries[0]
    objects = {}
    for oid, entry in enumerate(entries[1:], start=1):
        assert len(entry) == 20 and entry.endswith(b" \n"), f"xref entry {oid} is not 20 bytes: {entry!r}"
        offset = int(entry[:10])
        assert data[offset:].startswith(f"{oid} 0 obj".encode()), f"xref offset of object {oid} is wrong"
        end = data.index(b"endobj", offset)
        objects[oid] = data[offset:end]
    return objects


def page_count(data: bytes) -> int:
    objects = pdf_objects(data)
    pages = [o for o in objects.values() if re.search(rb"/Type /Page\b(?!s)", o)]
    tree = next(o for o in objects.values() if b"/Type /Pages" in o)
    count = int(re.search(rb"/Count (\d+)", tree).group(1))
    assert count == len(pages), f"/Count {count} but {len(pages)} page objects"
    for body in objects.values():                    # every stream's /Length is its byte count
        m = re.search(rb"/Length (\d+) >>\nstream\n", body)
        if m:
            stream = body[m.end():body.rindex(b"\nendstream")]
            assert len(stream) == int(m.group(1)), "a content stream's /Length is wrong"
    return count


# --------------------------------------------------------------------------- reading an .aux
def brace_groups(text: str) -> list[str]:
    """The top-level `{…}` groups of a string, nested braces kept."""
    groups, depth, current = [], 0, []
    for ch in text:
        if ch == "{":
            if depth:
                current.append(ch)
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth:
                current.append(ch)
            else:
                groups.append("".join(current))
                current = []
        elif depth:
            current.append(ch)
    return groups


def reference_aux_labels(aux: Path) -> dict[str, dict]:
    """show-contract §6.4, read from the catalog: `\\newlabel{name}{{number}{page}{title}
    {anchor}{…}}`, `@cref` entries skipped, the kind from the anchor prefix, falling back to
    the label prefix."""
    kinds = CATALOG["label_kinds"]
    out = {}
    for line in aux.read_text(encoding="utf-8").splitlines():
        if not line.startswith("\\newlabel{"):
            continue
        name, body = brace_groups(line[len("\\newlabel"):])[:2]
        if name.endswith("@cref"):
            continue
        number, page, title, anchor = brace_groups(body)[:4]
        kind = next((k for p, k in kinds["anchor_prefixes"].items() if anchor.startswith(p)), None) \
            or next((k for p, k in kinds["label_prefixes"].items() if name.startswith(p)), None)
        out[name] = {"number": number, "page": page, "title": title, "anchor": anchor, "kind": kind}
    return out


def h01_aux_labels(aux: Path) -> dict[str, dict] | None:
    """H01's AuxLabelSource on the file, as plain dicts; None while report_labels.py is not in
    the tree (it lands in the same wave)."""
    scripts = SHOW / "scripts"
    if not (scripts / "report_labels.py").is_file():
        return None
    sys.path.insert(0, str(scripts))
    try:
        module = importlib.import_module("report_labels")
    finally:
        sys.path.remove(str(scripts))
    labels = module.AuxLabelSource(aux).labels()

    def field(ref, name):
        value = ref.get(name) if isinstance(ref, dict) else getattr(ref, name, None)
        return getattr(value, "value", value)              # an enum kind reads as its value

    return {str(name): {k: field(ref, k) for k in ("number", "page", "title", "kind")}
            for name, ref in labels.items()}


# =========================================================================== the PDF
def test_main_pdf_starts_pdf_and_has_n_pages(docs, written):
    data = (docs / "main.pdf").read_bytes()
    assert data[:5] == b"%PDF-"
    roles = contracts.REPORT_ROLES(REPO)
    expected = 1 + math.ceil(len(roles) / report_stub.ROLES_PER_PAGE)
    assert page_count(data) == expected
    assert max(r["page"] for r in written["roles"].values()) == expected
    assert min(r["page"] for r in written["roles"].values()) == 2          # page 1 is the title page
    assert data.rstrip().endswith(b"%%EOF")


def test_main_pdf_page_count_follows_pages(tmp_path):
    for pages in (2, 5, 24, 30):
        docs = tmp_path / f"p{pages}" / "docs"
        doc = report_stub.write(docs, roles=contracts.REPORT_ROLES(REPO), options=report_stub.StubOptions(pages=pages))
        assert page_count((docs / "main.pdf").read_bytes()) == pages
        used = {r["page"] for r in doc["roles"].values()}
        assert used <= set(range(2, pages + 1))
        assert max(used) == pages or pages > len(doc["roles"]) + 1
    with pytest.raises(ValueError):
        report_stub.write(tmp_path / "bad" / "docs", roles=contracts.REPORT_ROLES(REPO),
                          options=report_stub.StubOptions(pages=1))


@pytest.mark.skipif(shutil.which("pdfinfo") is None, reason="poppler's pdfinfo is not installed")
def test_main_pdf_opens_in_poppler(docs, written):
    """A second, independent reader: poppler parses the file without a syntax error, counts
    the same pages, and finds each role's name on the page the role map gives it."""
    pdf = docs / "main.pdf"
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    assert info.returncode == 0 and info.stderr.strip() == "", info.stderr
    pages = int(re.search(r"^Pages:\s+(\d+)$", info.stdout, re.M).group(1))
    assert pages == page_count(pdf.read_bytes())
    if shutil.which("pdftotext") is None:
        return
    for role, binding in written["roles"].items():
        text = subprocess.run(["pdftotext", "-f", str(binding["page"]), "-l", str(binding["page"]), str(pdf), "-"],
                              capture_output=True, text=True, check=True).stdout
        assert f"{role} ({binding['label']})" in text, f"{role} is not on page {binding['page']}"


# =========================================================================== the .aux
def test_main_aux_yields_the_anchor_kinds(docs, written):
    labels = reference_aux_labels(docs / "main.aux")
    roles = CATALOG["report_roles"]
    assert set(labels) == {"vega:" + r for r in roles}, "one label per role, @cref twins skipped"
    for role, entry in roles.items():
        got = labels["vega:" + role]
        assert got["kind"] == entry["kind"], f"{role}: anchor {got['anchor']} reads as {got['kind']}"
        assert got["anchor"].split(".")[0] + "." in {"section.", "figure.", "table.", "equation."}
        assert re.fullmatch(r"3\.\d+", got["number"])
    prefixes = {v["anchor"].split(".")[0] for v in labels.values()}
    assert prefixes == {"section", "figure", "table", "equation"}
    raw = (docs / "main.aux").read_text(encoding="utf-8")
    assert raw.count("@cref}") == len(roles), "every label carries the cleveref twin the adapter skips"


def test_main_aux_parses_with_h01_aux_label_source(docs, written):
    labels = h01_aux_labels(docs / "main.aux")
    if labels is None:
        pytest.skip("vega-show/scripts/report_labels.py (H01, wave W1) is not in this tree yet; "
                    "test_main_aux_yields_the_anchor_kinds reads the same file by the contract's grammar")
    reference = reference_aux_labels(docs / "main.aux")
    assert set(labels) == set(reference)
    for name, ref in reference.items():
        assert str(labels[name]["kind"]) == ref["kind"], name
        assert str(labels[name]["number"]) == ref["number"], name
        assert str(labels[name]["page"]) == ref["page"], name


# =========================================================================== label-roles.json
def test_label_roles_validates_label_roles_1(docs, written):
    import jsonschema
    schema = json.loads(LABEL_ROLES_SCHEMA.read_text(encoding="utf-8"))
    doc = json.loads((docs / "label-roles.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(doc)
    assert doc == written
    assert doc["schema"] == contracts.LABEL_ROLES and doc["lane"] == "L02"
    assert doc["report"] == {"tex": "docs/main.tex", "pdf": "docs/main.pdf", "aux": "docs/main.aux",
                             "aux_digest": "sha256:" + hashlib.sha256((docs / "main.aux").read_bytes()).hexdigest()}
    assert list(doc["roles"]) == list(contracts.REPORT_ROLES(REPO))
    for role, binding in doc["roles"].items():
        assert binding["kind"] == CATALOG["report_roles"][role]["kind"]
        assert binding["stage"] == CATALOG["report_roles"][role]["stage"]


def test_label_roles_agree_with_the_aux(docs, written):
    """number, page and title per role are exactly the .aux's (the resolver verifies each,
    R-E7), and the map is not older than its .aux (PF6)."""
    labels = reference_aux_labels(docs / "main.aux")
    for role, binding in written["roles"].items():
        ref = labels[binding["label"]]
        assert (binding["number"], str(binding["page"]), binding["title"]) == (ref["number"], ref["page"], ref["title"])
    assert (docs / "label-roles.json").stat().st_mtime_ns >= (docs / "main.aux").stat().st_mtime_ns
    tex = (docs / "main.tex").read_text(encoding="utf-8")
    assert all(f"\\label{{{b['label']}}}" in tex for b in written["roles"].values())


def test_a_rewrite_with_other_pages_moves_the_digest(tmp_path):
    """The staleness lever (D6 §6.8): another page layout is another .aux, another digest."""
    docs = tmp_path / "L10" / "docs"
    at_noon = report_stub.StubOptions(generated_at="2026-09-29T12:00:00Z")
    first = report_stub.write(docs, roles=contracts.REPORT_ROLES(REPO), options=at_noon)
    again = report_stub.write(docs, roles=contracts.REPORT_ROLES(REPO), options=at_noon)
    assert first == again, "the stub is deterministic"
    moved = report_stub.write(docs, roles=contracts.REPORT_ROLES(REPO), options=report_stub.StubOptions(pages=4))
    assert moved["report"]["aux_digest"] != first["report"]["aux_digest"]
    assert any(moved["roles"][r]["page"] != first["roles"][r]["page"] for r in first["roles"])


def test_roles_as_a_mapping_and_unknown_roles(tmp_path):
    doc = report_stub.write(tmp_path / "x" / "docs", roles={"net.forward": {"kind": "equation", "stage": "net"}},
                            options=report_stub.StubOptions(lane="L99"))
    assert doc["roles"] == {"net.forward": {"label": "vega:net.forward", "kind": "equation", "stage": "net",
                                            "number": "3.1", "page": 2, "title": "net.forward"}}
    assert doc["lane"] == "L99"
    with pytest.raises(ValueError, match="not in the vega-show catalog"):
        report_stub.write(tmp_path / "y" / "docs", roles=("net.forward", "net.nonexistent"))
    with pytest.raises(ValueError):
        report_stub.write(tmp_path / "z" / "docs", roles=())


# =========================================================================== through the lane helper
def test_the_report_helper_writes_the_stub(tmp_path):
    """T01's `python -m lanekit.checks report` (the report stage) drives this module: exit 0,
    the three files, `%PDF-`."""
    lane_dir = tmp_path / "lanes" / "L02"
    lane_dir.mkdir(parents=True)
    result = subprocess.run([PY, "-m", "lanekit.checks", "report", "--lanes-file", str(HERE / "lanes.json"),
                             "--lane", "L02", "--lane-dir", str(lane_dir), "--skills", str(REPO),
                             "--docs", str(lane_dir / "docs")], cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: main.pdf starts %PDF-" in result.stdout
    assert "ERROR" not in result.stdout


def test_the_cli(tmp_path):
    docs = tmp_path / "lanes" / "L05" / "docs"
    result = subprocess.run([PY, "-m", "lanekit.report_stub", "--docs", str(docs), "--skills", str(REPO)],
                            cwd=HERE, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("OK: stub report for lane L05")
    assert json.loads((docs / "label-roles.json").read_text(encoding="utf-8"))["lane"] == "L05"
    bad = subprocess.run([PY, "-m", "lanekit.report_stub", "--docs", str(docs), "--pages", "1"],
                         cwd=HERE, capture_output=True, text=True)
    assert bad.returncode == 2 and bad.stdout.startswith("ERROR:")
