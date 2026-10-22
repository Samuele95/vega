#!/usr/bin/env python3
"""
test_aux_adapter.py — the report adapter (`AuxLabelSource`) and the role-map reader.

    python3 test_aux_adapter.py           # or: python3 -m pytest test_aux_adapter.py

`AuxLabelSource` parses a LaTeX `.aux`: the fixture below always, and the CIDS report's
`cids/docs/main.aux` when it is present (`CIDS_ROOT`, default `~/Documenti/cids`). The fixture
holds what a real `.aux` holds and what lanekit's `report_stub` writes (`vega:<role>` labels
with `equation.`/`figure.` anchors): `@cref` shadows, nested and escaped braces, a label with
no hyperref groups and a roman page, anchors of every kind prefix, a label that no prefix
names, a label defined twice. `read_role_map` is checked against the same `.aux`: a current
map binds, and a stale digest, a missing `.aux`, a missing label, a kind, number or catalog
mismatch, or a map that is not `vega/label-roles@1` is refused under gate PF6.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))

from report_labels import AuxLabelSource, ReportUnreadable  # noqa: E402
from role_map import RoleBinding, RoleMap, read_role_map, resolve_aux  # noqa: E402
from show_ports import LabelRef, LabelSource  # noqa: E402
from vega_show_core import CATALOG, LabelKind, ShowError, digest_file  # noqa: E402

CIDS_AUX = Path(os.environ.get("CIDS_ROOT", Path.home() / "Documenti" / "cids")) / "docs" / "main.aux"

FIXTURE_AUX = r"""\relax
\providecommand\hyper@newdestlabel[2]{}
\@writefile{toc}{\contentsline {section}{\numberline {1}Introduction}{1}{section.1}\protected@file@percent }
\newlabel{sec:intro}{{1}{1}{Introduction}{section.1}{}}
\newlabel{sec:intro@cref}{{[section][1][]1}{[1][1][]1}{}{}{}}
\newlabel{sec:method:net}{{3.4}{27}{Network architecture}{subsection.3.4}{}}
\newlabel{eq:mlp}{{2}{27}{Network architecture}{equation.2}{}}
\newlabel{eq:mlp@cref}{{[equation][2][]2}{[1][27][]27}{}{}{}}
\newlabel{fig:serving}{{18}{27}{The {\em inference} path, with \num {65} inputs and a \{brace\}}{figure.caption.37}{}}
\newlabel{tab:network}{{4}{27}{Layers and parameters}{table.caption.39}{}}
\newlabel{eq:ams}{{5}{29}{Optimisation}{AMS.5}{}}
\newlabel{vega:net.forward}{{3.1}{2}{net.forward}{equation.3.1}{}}
\newlabel{vega:pack.slice}{{3.2}{3}{pack.slice}{figure.3.2}{}}
\newlabel{fig:noanchor}{{7}{iv}}
\newlabel{plain}{{2}{5}{A title}{Doc-Start}{}}
\newlabel{chap:one}{{1}{1}{One}{chapter.1}{}}
\newlabel{sec:dup}{{1}{1}{First}{section.1}{}}
\newlabel{sec:dup}{{2}{2}{Second}{section.2}{}}
"""

KINDS = {"sec:intro": "section", "sec:method:net": "section", "eq:mlp": "equation",
         "fig:serving": "figure", "tab:network": "table", "eq:ams": "equation",
         "vega:net.forward": "equation", "vega:pack.slice": "figure", "fig:noanchor": "figure",
         "plain": "other", "chap:one": "section", "sec:dup": "section"}


def _project(tmp: Path, aux_text: str = FIXTURE_AUX) -> Path:
    """A project directory holding docs/main.aux; returns the aux path."""
    aux = tmp / "docs" / "main.aux"
    aux.parent.mkdir(parents=True, exist_ok=True)
    aux.write_text(aux_text, encoding="utf-8")
    return aux


def _write_map(aux: Path, roles: dict, digest: str | None = None) -> Path:
    doc = {"schema": "vega/label-roles@1", "lane": "L01", "generated_at": "2026-09-29T11:00:00Z",
           "adapter": "latex-aux",
           "report": {"tex": "docs/main.tex", "pdf": "docs/main.pdf", "aux": "docs/main.aux",
                      "aux_digest": digest or digest_file(aux)},
           "roles": roles}
    path = aux.parent / "label-roles.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


GOOD_ROLES = {"net.forward": {"label": "vega:net.forward", "kind": "equation", "stage": "net",
                              "number": "3.1", "page": 2, "title": "net.forward"},
              "pack.slice": {"label": "vega:pack.slice", "kind": "figure", "stage": "pack"},
              "chapter.net": {"label": "sec:method:net", "kind": "section", "stage": "net"}}


def test_fixture_labels_and_kinds():
    with tempfile.TemporaryDirectory() as tmp:
        source = AuxLabelSource(_project(Path(tmp)))
        labels = source.labels()
        assert isinstance(source, LabelSource) and LabelSource in type(source).__mro__, "declared"
        assert {n: str(r.kind) for n, r in labels.items()} == KINDS
        assert not any(n.endswith("@cref") for n in labels)
        assert labels["eq:mlp"] == LabelRef("eq:mlp", "2", 27, "Network architecture",
                                            LabelKind.EQUATION, "equation.2")
        assert labels["vega:net.forward"].number == "3.1" and labels["vega:net.forward"].page == 2


def test_fixture_braces_pages_and_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        aux = _project(Path(tmp))
        labels = AuxLabelSource(aux).labels()
        title = r"The {\em inference} path, with \num {65} inputs and a \{brace\}"
        assert labels["fig:serving"].title == title
        assert labels["fig:noanchor"].page == "iv" and labels["fig:noanchor"].title == ""
        assert all(isinstance(r.page, int) for n, r in labels.items() if n != "fig:noanchor")
        assert (labels["sec:dup"].number, labels["sec:dup"].title) == ("2", "Second"), "the last definition"
        assert AuxLabelSource(aux).digest() == digest_file(aux)


def test_unreadable_aux_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        broken = FIXTURE_AUX + "\\newlabel{broken}{{1}{2}{an {open title}{x.1}{}}\n"
        try:
            AuxLabelSource(_project(Path(tmp), broken))
        except ReportUnreadable as exc:
            assert "unbalanced" in str(exc) and isinstance(exc, ShowError)
        else:
            raise AssertionError("an unbalanced .aux was read")
        try:
            AuxLabelSource(Path(tmp) / "absent.aux")
        except ReportUnreadable as exc:
            assert "could not be read" in str(exc)
        else:
            raise AssertionError("a missing .aux was read")


def test_the_cids_report_when_present():
    if not CIDS_AUX.is_file():
        print(f"    (skipped: {CIDS_AUX} is absent)")
        return
    text = CIDS_AUX.read_text(encoding="utf-8", errors="replace")
    names = re.findall(r"\\newlabel\{([^{}]*)\}", text)
    expected = {n for n in names if not n.endswith("@cref")}
    labels = AuxLabelSource(CIDS_AUX).labels()
    assert set(labels) == expected and len(labels) == len(names) - sum(n.endswith("@cref") for n in names)
    assert labels["eq:mlp"].kind == LabelKind.EQUATION and isinstance(labels["eq:mlp"].page, int)
    assert labels["fig:serving"].kind == LabelKind.FIGURE
    assert all(str(r.kind) in CATALOG["label_kinds"]["kinds"] for r in labels.values())
    assert all(isinstance(r.page, int) for r in labels.values())
    if "tab:classes" in labels:
        assert r"\num {2830743}" in labels["tab:classes"].title, "a nested group stays in its title"


def test_a_current_role_map_binds():
    with tempfile.TemporaryDirectory() as tmp:
        aux = _project(Path(tmp))
        role_map, problems = read_role_map(_write_map(aux, GOOD_ROLES))
        assert problems == [] and role_map is not None, [str(p) for p in problems]
        assert role_map.origin == "document-project" and role_map.aux_digest == digest_file(aux)
        assert dict(role_map.roles) == {r: s["label"] for r, s in GOOD_ROLES.items()}
        slice_ = role_map.binding("pack.slice")
        assert (slice_.number, slice_.page, slice_.title) == ("3.2", 3, "pack.slice"), "from the .aux"
        assert role_map.binding("net.update") is None


def test_a_proposed_role_map_is_a_value():
    bindings = (RoleBinding("pack.slice", "fig:serving", "figure", "pack", "18", 27, "The path"),
                RoleBinding("net.forward", "eq:mlp", "equation", "net"))
    proposed = RoleMap(bindings, source="docs/main.aux", origin="proposed")
    assert proposed.roles == (("net.forward", "eq:mlp"), ("pack.slice", "fig:serving")), "sorted by role"
    assert proposed == RoleMap(tuple(reversed(bindings)), "docs/main.aux", "proposed")
    assert hash(proposed) == hash(RoleMap(bindings, "docs/main.aux", "proposed"))
    try:
        RoleMap(bindings, source="x", origin="guessed")
    except ValueError:
        return
    raise AssertionError("an origin outside showcase.role_origins was accepted")


def _refusal(aux: Path, roles: dict, digest: str | None = None) -> list[str]:
    role_map, problems = read_role_map(_write_map(aux, roles, digest))
    assert role_map is None and problems, "the map was accepted"
    assert all(p.rule == "PF6" and p.level == "ERROR" for p in problems)
    return [str(p) for p in problems]


def test_a_stale_or_missing_aux_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        aux = _project(Path(tmp))
        path = _write_map(aux, GOOD_ROLES)
        aux.write_text(FIXTURE_AUX + "% recompiled\n", encoding="utf-8")
        _, problems = read_role_map(path)
        assert len(problems) == 1 and "older than the report's .aux" in problems[0].message
        assert "recompile /document-project" in problems[0].message and problems[0].rule == "PF6"
        aux.unlink()
        _, problems = read_role_map(path)
        assert len(problems) == 1 and "does not exist" in problems[0].message


def test_a_map_that_disagrees_with_its_aux_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        aux = _project(Path(tmp))
        missing = {"net.loss": {"label": "eq:nowhere", "kind": "equation", "stage": "net"}}
        assert "is not in the .aux" in _refusal(aux, missing)[0]
        wrong_kind = {"net.forward": {"label": "fig:serving", "kind": "equation", "stage": "net"}}
        assert "is a figure in the .aux" in _refusal(aux, wrong_kind)[0]
        for key, value in (("number", "9"), ("page", 5), ("title", "Another title")):
            spec = {**GOOD_ROLES["net.forward"], key: value}
            assert f"{key} {value!r} differs" in _refusal(aux, {"net.forward": spec})[0]
        catalog = {"pack.slice": {"label": "tab:network", "kind": "table", "stage": "pack"}}
        refused = _refusal(aux, catalog)
        assert any("the catalog's role is 'figure'" in p for p in refused), refused


def test_a_map_that_is_not_label_roles_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        aux = _project(Path(tmp))
        refused = _refusal(aux, {"net.magic": {"label": "eq:mlp", "kind": "equation", "stage": "net"}})
        assert any("not vega/label-roles@1" in p for p in refused), refused
        assert "not vega/label-roles@1" in _refusal(aux, GOOD_ROLES, digest="md5:0")[0]
        (aux.parent / "label-roles.json").write_text("{not json", encoding="utf-8")
        _, problems = read_role_map(aux.parent / "label-roles.json")
        assert problems and "could not be read" in problems[0].message


def test_the_label_source_port_and_the_aux_location():
    class Labels:                                   # a non-LaTeX adapter, as a test builds one
        def labels(self):
            return {"vega:net.forward": LabelRef("vega:net.forward", "3.1", 2, "net.forward",
                                                 LabelKind.EQUATION, "equation.3.1")}

        def digest(self):
            return "sha256:" + "ab" * 32

    with tempfile.TemporaryDirectory() as tmp:
        roles = {"net.forward": GOOD_ROLES["net.forward"]}
        path = _write_map(_project(Path(tmp)), roles, Labels().digest())
        role_map, problems = read_role_map(path, labels=Labels())
        assert problems == [] and role_map.binding("net.forward").label == "vega:net.forward"
        moved = Path(tmp) / "reports" / "elsewhere"
        moved.mkdir(parents=True)
        (moved / "main.aux").write_text(FIXTURE_AUX, encoding="utf-8")
        assert resolve_aux(moved / "label-roles.json", "docs/main.aux") == moved / "main.aux"
        assert resolve_aux(path, "docs/main.aux") == Path(tmp) / "docs" / "main.aux"


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
