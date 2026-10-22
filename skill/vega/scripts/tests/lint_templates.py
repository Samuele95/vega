#!/usr/bin/env python3
"""
lint_templates.py — the one dataset-literal gate over every shared template (D6 §15.2, R-F10).

    python3 lint_templates.py [PATH ...] [--root DIR] [--literals FILE] [--allow FILE]
                              [--json OUT] [--verbose]

Without PATH it scans the package's shared templates under --root (default: the repository
this file sits in):
  - vega-forge/scripts/backends/pytorch/templates/** (the emitted and served files);
  - vega-show/assets/** (the page template) and every layer renderer under <skill>/assets/**;
  - vega-pack/scripts/*.py: every string literal, docstrings excepted. D6 §15.2 names the
    template constants (API_TEST, ROUNDTRIP_TEST, CLI, TRAIN_CLI, EVAL_CLI, README*, DOCKERFILE,
    SERVE_README, PYPROJECT) and the functions that write lane files (_presets, _demo_sample,
    _class_attribution, service_identity); any other literal can reach a bundle too (the model
    card's post-eval block is written by model_card.py), so none is left out.
With PATH it scans those files (a directory: every text file under it) whole.

The tokens are foreign-literals.json's hard and soft tokens (a shared template belongs to no
lane, so a soft token is a hit there too) and every lane generator's VOCAB
(lanekit/datasets: GENERATORS, each module's VOCAB), less the file's `generic` words. Allowed
exceptions live in lint-allow.json, `{file, token, reason, match?}`; an entry without a reason
is refused. Each hit prints `<file>:<line>: <token>` (with --verbose, the line after it).

Exit 0: no hit. 1: a hit. 2: the literals, the allow list or a PATH could not be read.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LITERALS = HERE / "foreign-literals.json"
ALLOW = HERE / "lint-allow.json"
TEMPLATE_DIRS = ("vega-forge/scripts/backends/pytorch/templates", "vega-show/assets", "vega-forge/assets",
                 "vega-train/assets", "vega-eval/assets", "vega-pack/assets", "dataset-forge/assets",
                 "data-lens/assets", "dataset-shaper/assets")
PACK_SOURCES = "vega-pack/scripts"
LANEKIT = "lanekit"
SKIP_PARTS = frozenset({"__pycache__", "node_modules", ".git"})
MAX_BYTES = 4 * 1024 * 1024


class LintError(Exception):
    """The gate cannot run: the literals, the allow list or a path is unreadable (exit 2)."""


@dataclass(frozen=True)
class Token:
    token: str
    kind: str                 # hard | soft | vocab:<generator>
    regex: re.Pattern


@dataclass(frozen=True)
class Hit:
    file: str
    line: int
    token: str
    kind: str
    text: str

    def line_out(self, verbose: bool) -> str:
        head = f"{self.file}:{self.line}: {self.token}"
        return f"{head}    {self.text.strip()[:160]}" if verbose else head


@dataclass(frozen=True)
class Allow:
    file: str
    token: str
    reason: str
    match: re.Pattern | None

    def covers(self, hit: Hit) -> bool:
        return (hit.token == self.token and fnmatch.fnmatchcase(hit.file, self.file)
                and (self.match is None or bool(self.match.search(hit.text))))


# ---- the tokens ---------------------------------------------------------------------------------
def read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LintError(f"cannot read {what} {path}: {exc}") from exc


def literal_tokens(doc: dict) -> list[Token]:
    """The hard and soft tokens of a vega/foreign-literals@1 document, compiled."""
    if doc.get("schema") != "vega/foreign-literals@1":
        raise LintError(f"the literals file is not vega/foreign-literals@1 (schema {doc.get('schema')!r})")
    out = []
    for kind in ("hard", "soft"):
        for entry in doc.get(kind) or []:
            try:
                out.append(Token(entry["token"], kind, re.compile(entry["regex"])))
            except (KeyError, re.error) as exc:
                raise LintError(f"foreign-literals {kind} entry {entry!r}: {exc}") from exc
    return out


def _lanekit_modules() -> list[str]:
    return [k for k in sys.modules if k == LANEKIT or k.startswith(LANEKIT + ".")]


def generator_vocab(tests_dir: Path) -> tuple[dict[str, frozenset], str]:
    """{generator: VOCAB} of lanekit.datasets.GENERATORS under tests_dir, and a note on where it
    came from. The package is optional until the generators land (T02, T05–T07); a package that
    is present but fails to import, or a generator that fails to load or has no VOCAB, is an
    error (exit 2), never a silently empty vocabulary. The lanekit of tests_dir is read even when
    another tree's lanekit is already imported, and the modules imported before are restored."""
    if not (tests_dir / LANEKIT / "datasets" / "__init__.py").is_file():
        return {}, "no lane generators in this tree (lanekit/datasets absent): lane VOCAB not scanned"
    cached = getattr(sys.modules.get(LANEKIT), "__file__", None)
    fresh = cached is None or Path(cached).resolve().parent != (tests_dir / LANEKIT).resolve()
    saved = {k: sys.modules.pop(k) for k in _lanekit_modules()} if fresh else {}
    sys.path.insert(0, str(tests_dir))
    importlib.invalidate_caches()
    try:
        generators = importlib.import_module(f"{LANEKIT}.datasets").GENERATORS
        vocab = {}
        for name in sorted(generators):
            words = getattr(generators[name], "VOCAB", None)
            if not isinstance(words, (frozenset, set, tuple, list)):
                raise TypeError(f"generator {name!r} has no VOCAB of words")
            vocab[name] = frozenset(words)
    except Exception as exc:  # noqa: BLE001 — any failure to read the generators stops the gate
        raise LintError(f"lanekit.datasets cannot be read: {type(exc).__name__}: {exc}") from exc
    finally:
        sys.path.remove(str(tests_dir))
        if fresh:
            for k in _lanekit_modules():
                del sys.modules[k]
            sys.modules.update(saved)
    return vocab, f"lane VOCAB of {len(vocab)} generator(s): {', '.join(vocab) or 'none'}"


def vocab_tokens(vocab: dict[str, frozenset], doc: dict) -> list[Token]:
    """One token per VOCAB word no hard or soft token already names and no generic word is."""
    taken = {e["token"].lower() for k in ("hard", "soft") for e in doc.get(k) or []}
    generic = {w.lower() for w in doc.get("generic") or []}
    seen, out = set(), []
    for name, words in vocab.items():
        for word in sorted(words):
            if len(word) < 3 or word.lower() in taken | generic | seen:
                continue
            seen.add(word.lower())
            out.append(Token(word, f"vocab:{name}", re.compile(r"\b" + re.escape(word) + r"\b")))
    return out


def load_allow(path: Path) -> list[Allow]:
    if not path.is_file():
        return []
    doc = read_json(path, "the allow list")
    out = []
    for entry in doc.get("allow") or []:
        if not str(entry.get("reason") or "").strip():
            raise LintError(f"lint-allow entry without a reason: {entry!r}")
        match = entry.get("match")
        out.append(Allow(entry["file"], entry["token"], entry["reason"], re.compile(match) if match else None))
    return out


# ---- what is read -------------------------------------------------------------------------------
def text_lines(path: Path) -> list[tuple[int, str]]:
    """The lines of a text file; nothing for a binary or an oversized one."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LintError(f"cannot read {path}: {exc}") from exc
    if len(data) > MAX_BYTES or b"\0" in data[:8192]:
        return []
    return list(enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1))


def _string_lines(node: ast.AST, skip: set[int]) -> list[tuple[int, str]]:
    """Every line of every string constant under node, at its source line (docstrings in skip)."""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and id(sub) not in skip:
            out.extend((sub.lineno + i, text) for i, text in enumerate(sub.value.splitlines()))
    return out


def _docstring_ids(node: ast.AST) -> set[int]:
    out = set()
    for sub in ast.walk(node):
        body = getattr(sub, "body", None)
        if (isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)):
            out.add(id(body[0].value))
    return out


def template_literals(path: Path) -> list[tuple[int, str]]:
    """The lines of a pack source that can become lane files: every string literal (template
    constants, lane-writing functions, rendered model-card text), docstrings excepted."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise LintError(f"cannot parse {path}: {exc}") from exc
    return _string_lines(tree, _docstring_ids(tree))


def files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise LintError(f"no such file or directory: {path}")
    return sorted(p for p in path.rglob("*") if p.is_file() and not SKIP_PARTS & set(p.parts)
                  and p.suffix != ".pyc")


def default_units(root: Path) -> list[tuple[Path, str]]:
    """(file, mode) for every shared template under root: whole text, or pack string literals."""
    units = [(f, "text") for d in TEMPLATE_DIRS if (root / d).is_dir() for f in files_under(root / d)]
    units += [(f, "pack") for f in sorted((root / PACK_SOURCES).glob("*.py"))]
    return units


# ---- the scan -----------------------------------------------------------------------------------
def display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def scan(units: list[tuple[Path, str]], tokens: list[Token], root: Path) -> list[Hit]:
    hits = []
    for path, mode in units:
        lines = template_literals(path) if mode == "pack" else text_lines(path)
        rel = display(path, root)
        for line_no, text in lines:
            hits.extend(Hit(rel, line_no, t.token, t.kind, text) for t in tokens if t.regex.search(text))
    return sorted(set(hits), key=lambda h: (h.file, h.line, h.token))


@dataclass(frozen=True)
class Report:
    hits: list
    allowed: list
    scanned: int
    note: str

    def as_json(self) -> dict:
        row = lambda h: {"file": h.file, "line": h.line, "token": h.token, "kind": h.kind}  # noqa: E731
        return {"schema": "vega/lint-templates@1", "scanned": self.scanned, "note": self.note,
                "hits": [row(h) for h in self.hits], "allowed": [row(h) for h in self.allowed]}


def lint(paths: list[Path] | None = None, root: Path = REPO, literals: Path = LITERALS,
         allow: Path = ALLOW) -> Report:
    """The gate as a function: hits in the given paths (default: every shared template under root)."""
    doc = read_json(literals, "the literals")
    vocab, note = generator_vocab(root / "vega" / "scripts" / "tests")
    tokens = literal_tokens(doc) + vocab_tokens(vocab, doc)
    units = default_units(root) if not paths else [(f, "text") for p in paths for f in files_under(p)]
    if not units:
        raise LintError(f"nothing to scan: no shared template under {root}" if not paths
                        else f"nothing to scan in {', '.join(map(str, paths))}")
    rules = load_allow(allow)
    found = scan(units, tokens, root)
    allowed = [h for h in found if any(a.covers(h) for a in rules)]
    return Report([h for h in found if h not in allowed], allowed, len(units), note)


def page_leaks(text: str, own_vocab: frozenset = frozenset(), other_vocab: frozenset = frozenset(),
               literals: Path = LITERALS) -> list[str]:
    """The P-LEAK reading of one page's visible text (D6 §10.6): hard tokens always; soft tokens
    unless the lane's own VOCAB holds them; another lane's VOCAB word unless it is this lane's
    own or a generic word. The tokens found, sorted."""
    doc = read_json(literals, "the literals")
    own = {w.lower() for w in own_vocab}
    found = {t.token for t in literal_tokens(doc) if t.regex.search(text)
             and not (t.kind == "soft" and t.token.lower() in own)}
    others = {"other": frozenset(w for w in other_vocab if w.lower() not in own)}
    found |= {t.token for t in vocab_tokens(others, doc) if t.regex.search(text)}
    return sorted(found)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="files or directories to scan whole")
    ap.add_argument("--root", type=Path, default=REPO, help="the tree whose shared templates are scanned")
    ap.add_argument("--literals", type=Path, default=LITERALS)
    ap.add_argument("--allow", type=Path, default=ALLOW)
    ap.add_argument("--json", default=None, help="write the vega/lint-templates@1 report here ('-': stdout)")
    ap.add_argument("--verbose", action="store_true", help="print each hit's line after it")
    a = ap.parse_args(argv)
    try:
        report = lint(a.paths or None, a.root, a.literals, a.allow)
    except LintError as exc:
        print(f"ERROR: {exc}")
        return 2
    return _report(report, a)


def _report(report: Report, a: argparse.Namespace) -> int:
    if a.json:
        text = json.dumps(report.as_json(), indent=1) + "\n"
        if a.json == "-":
            sys.stdout.write(text)
            return 1 if report.hits else 0
        Path(a.json).write_text(text, encoding="utf-8")
    for hit in report.hits:
        print(hit.line_out(a.verbose))
    files = len({h.file for h in report.hits})
    tail = f" ({len(report.allowed)} allowed by lint-allow.json)" if report.allowed else ""
    print(f"INFO: {report.note}")
    if report.hits:
        print(f"FAIL: {len(report.hits)} foreign literal(s) in {files} of {report.scanned} file(s){tail}")
        return 1
    print(f"OK: {report.scanned} file(s) scanned, no foreign literal{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
