#!/usr/bin/env python3
"""
check_unattended.py — every fork has its unattended answer (D6 §15.6, BUILD-PLAN R-G3).

    python3 check_unattended.py [SKILL.md ...] [--root DIR] [--chain-map FILE] [--list]

1. The chain map (vega/references/chain-map.md): every row of its Stages table has a
   non-empty "unattended behaviour" cell; and, when no file is named, every package stage whose
   SKILL.md exists (forge-data, lens, shape, net, train, eval, pack, show) has a row.
2. Each SKILL.md (default: the package's — vega, vega-forge, vega-train, vega-eval, vega-pack,
   vega-show, dataset-forge, data-lens, dataset-shaper — and patches/*/SKILL.md): every
   paragraph, list item or table row where the prompt emits a `FORK:` also says what
   `--unattended` does there (a default applied and marked, a stop with `WARN:`, "not run"...).
   Not a fork: the marker legend ("`FORK:` when you stop to ask") and a reference to forks
   emitted elsewhere ("the `FORK:` items", "copy any `FORK:` the stage left", "a stage's
   `FORK:`", "settles a `FORK:`"). A fork: "emit the `FORK:` naming ...", "the step's `FORK:`
   asks ...", "any `FORK:` names both". Fenced code is skipped.

--list prints every fork found with the sentence that answers it unattended. The dynamic half
is the lanes': every command runs with stdin=DEVNULL, so a fork that waits hits EOF.

Exit 0: every fork is answered. 1: a fork is not (the file, the line and the fork are named),
or a stage row is missing or empty. 2: a named file or the chain map cannot be read.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CHAIN_MAP = "vega/references/chain-map.md"
PACKAGE_SKILLS = ("vega", "vega-forge", "vega-train", "vega-eval", "vega-pack", "vega-show", "dataset-forge",
                  "data-lens", "dataset-shaper")
STAGE_OF = {"dataset-forge": "forge-data", "data-lens": "lens", "dataset-shaper": "shape", "vega-forge": "net",
            "vega-train": "train", "vega-eval": "eval", "vega-pack": "pack", "vega-show": "show"}
MARK = re.compile(r"`?FORK:`?")
LEGEND_AFTER = re.compile(r"`?\s*(?:markers?\s+)?when you stop to ask", re.I)
REFERENCE_BEFORE = re.compile(r"(?:\b(?:copy|record|carry)\s+any|\ba\s+stage's|\bsettles\s+an?)\s*`?\s*$", re.I)
PLURAL_AFTER = re.compile(r"`?\s*(?:items|forks|questions|lines)\b", re.I)
UNATTENDED = re.compile(r"unattended", re.I)
EMPTY_CELLS = frozenset({"", "—", "–", "-", "n/a", "none"})
ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+|^\s*\|")


class Unreadable(Exception):
    """A named file or the chain map cannot be read (exit 2)."""


@dataclass(frozen=True)
class Unit:
    """A paragraph, a list item or a table row of a prompt, with the line it starts on."""
    line: int
    text: str


@dataclass(frozen=True)
class Fork:
    file: str
    line: int
    text: str
    answered: bool

    def excerpt(self) -> str:
        m = MARK.search(self.text)
        start = max(0, (m.start() if m else 0) - 60)
        return " ".join(self.text[start:start + 220].split())

    def answer(self) -> str:
        m = UNATTENDED.search(self.text)
        return " ".join(self.text[max(0, m.start() - 20):m.start() + 140].split()) if m else ""


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Unreadable(f"cannot read {path}: {exc}") from exc


# ---- the prompts ----------------------------------------------------------------------------------------
def units(text: str) -> list[Unit]:
    """Paragraphs split further at list items and table rows; fenced code left out."""
    out, cur, start, fenced = [], [], 0, False
    for no, line in enumerate(text.splitlines() + [""], start=1):
        stripped = line.strip()
        fence = stripped.startswith(("```", "~~~"))
        breaks = fence or fenced or not stripped or bool(ITEM.match(line)) or stripped.startswith("#")
        if breaks and cur:
            out.append(Unit(start, " ".join(cur)))
            cur = []
        fenced = fenced != fence
        if fence or fenced or not stripped:
            continue
        if not cur:
            start = no
        cur.append(stripped)
    return out


def emits_fork(text: str) -> bool:
    """True when some `FORK:` in the text is the prompt emitting one, not its legend or a reference."""
    for m in MARK.finditer(text):
        before = text[:m.start()]
        referred = REFERENCE_BEFORE.search(before) or (re.search(r"\bthe\s*$", before, re.I)
                                                        and PLURAL_AFTER.match(text, m.end()))
        if not (LEGEND_AFTER.match(text, m.end()) or referred):
            return True
    return False


def forks(path: Path, root: Path) -> list[Fork]:
    rel = path.resolve().relative_to(root.resolve()).as_posix() if path.resolve().is_relative_to(root.resolve()) \
        else str(path)
    return [Fork(rel, u.line, u.text, bool(UNATTENDED.search(u.text)))
            for u in units(read(path)) if MARK.search(u.text) and emits_fork(u.text)]


def package_prompts(root: Path) -> list[Path]:
    found = [root / s / "SKILL.md" for s in PACKAGE_SKILLS if (root / s / "SKILL.md").is_file()]
    return found + sorted((root / "patches").glob("*/SKILL.md"))


# ---- the chain map --------------------------------------------------------------------------------------
def _cells(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def stage_rows(chain_map: Path) -> list[dict]:
    """The Stages table of the chain map, one {column: cell} per row."""
    text = read(chain_map)
    section = re.search(r"^## Stages\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    table = [ln for ln in (section.group(1).splitlines() if section else []) if ln.lstrip().startswith("|")]
    if len(table) < 3:
        raise Unreadable(f"{chain_map}: no Stages table")
    header = [h.lower() for h in _cells(table[0])]
    return [dict(zip(header, _cells(row))) for row in table[2:]]


def chain_map_findings(rows: list[dict], root: Path, every_stage: bool) -> list[str]:
    column = next((h for h in (rows[0] if rows else {}) if "unattended" in h), None)
    if column is None:
        return ["chain-map.md: the Stages table has no unattended column"]
    out = [f"chain-map.md: stage {r.get('stage', '?')} has no unattended behaviour"
           for r in rows if r.get(column, "").strip("` ").lower() in EMPTY_CELLS]
    if every_stage:
        named = {m for r in rows for m in re.findall(r"/([a-z][\w-]*)", r.get("skill", ""))}
        out += [f"chain-map.md: no Stages row for {STAGE_OF[s]} ({s}, whose SKILL.md exists)"
                for s in STAGE_OF if (root / s / "SKILL.md").is_file() and s not in named]
    return out


# ---- the command ----------------------------------------------------------------------------------------
def run(files: list[Path], root: Path, chain_map: Path) -> tuple[list[Fork], list[str]]:
    """Every fork found and every finding (an unanswered fork, a missing or empty stage row)."""
    prompts = files or package_prompts(root)
    found = [f for p in prompts for f in forks(p, root)]
    problems = chain_map_findings(stage_rows(chain_map), root, every_stage=not files)
    problems += [f"{f.file}:{f.line}: a FORK: with no --unattended behaviour — \"{f.excerpt()}\""
                 for f in found if not f.answered]
    return found, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("files", nargs="*", type=Path, help="SKILL.md files (default: the package's)")
    ap.add_argument("--root", type=Path, default=REPO, help="the skills tree (default: this repository)")
    ap.add_argument("--chain-map", type=Path, default=None, help=f"default: <root>/{CHAIN_MAP}")
    ap.add_argument("--list", action="store_true", help="print every fork and its unattended answer")
    a = ap.parse_args(argv)
    try:
        found, problems = run(a.files, a.root, a.chain_map or a.root / CHAIN_MAP)
    except Unreadable as exc:
        print(f"ERROR: {exc}")
        return 2
    if a.list:
        for f in found:
            print(f"{'OK' if f.answered else 'FAIL'}: {f.file}:{f.line}: {f.excerpt()}\n    unattended: "
                  f"{f.answer() or '(none)'}")
    for p in problems:
        print(f"FAIL: {p}")
    files = len({f.file for f in found})
    if problems:
        print(f"FAIL: {len(problems)} finding(s); {len(found)} fork(s) in {files} prompt(s)")
        return 1
    print(f"OK: {len(found)} fork(s) in {files} prompt(s), each with its --unattended behaviour; "
          "every stage row of the chain map has one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
