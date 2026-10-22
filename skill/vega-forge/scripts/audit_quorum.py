#!/usr/bin/env python3
"""
audit_quorum.py — the judgement half of the conformance gate, made reproducible and advisory
(code-contract.md §5, *Reproduced, or recorded* and *Advisory*).

    python3 audit_quorum.py --tool architect RUN [RUN RUN] [--mode attended|unattended] [--json OUT]
    python3 audit_quorum.py --tool improve   RUN [RUN RUN] [--mode attended|unattended] [--json OUT]

RUN is an audit's run directory (`.claude/architect-runs/<ts>` or `.claude/improve-runs/<ts>`)
holding its `findings.json`; the directory's name is the audit's **run id**. The first RUN is
the first audit; the others are the two fresh, independent audits the rule calls for. The rule,
stated once here for the forge's and the pack's apply scripts (which import `judge` and
`advise`) and for the stage that runs the audits (which runs this command):

  * A **blocking-class** finding is one of depth `structural` or tagged `divergence`.
  * One audit with no blocking-class finding passes: no further audit runs.
  * When the first audit reports a blocking-class finding, two more independent fresh audits
    run on the same code, **one after another** (never concurrently: two audits started in the
    same second write one run id), and all three are judged together — never one or two.
  * Two findings of two audits **report the same thing** when they share a file location (a
    file both name, compared by path components from the file name up, line numbers ignored;
    a finding naming no file matches only a location written the same) **and** share a
    principle anchor (`EJ-64`, `SOLID-D`, `HEX`, …) or a category (the finding's `category`,
    else its `axis`). A report at any depth counts: the depth is the auditor's judgement of
    the same defect.
  * A blocking-class finding is **confirmed** when at least two of the three audits report it
    (its own and at least one other); an unconfirmed one is recorded with the audits that did
    and did not report it.
  * The three audits are distinct runs (no directory, no run id and no `findings.json` given
    twice) of the same code (the `scope.paths` every run states are the same).
  * **Advisory** (the user's decision of 2026-10-11): a confirmed finding never stops an
    unattended chain — it is a `WARN:` and an **open item** handed to `/architect` and
    `/improve` (the stage brief's *Open*, the layer's `code_findings`, the ledger's **Open**);
    in an attended run it is a `FORK:` the user answers — *fix now*, *accept and record*, or
    *stop* — and the layer records the answer. What refuses is only a procedure that did not
    run or a record that is malformed: a pending judgement, an invalid one (the wrong number
    of audits, a run or run id given twice, a copied run, audits of other code), and, attended,
    a confirmed finding with no recorded answer or an answer of *fix now* or *stop*. The
    mechanical half (conformance.py C0-C14) is not judged here and blocks on any finding.

Every audit run (path, run id, digest, counts, scope) and every blocking-class finding's
matching (who reported it, which findings matched, confirmed or not, counted or a repeat of an
earlier audit's report) are returned as one record, which the layers carry with the advisory
outcome beside it.

Exit codes: 0 nothing confirmed; 1 the first audit reported a blocking-class finding and the
two fresh audits are not given yet (`NEED:`); 2 a confirmed finding — advisory: unattended a
`WARN:` and an open item, attended the `FORK:` the user answers; 3 a run cannot be read, is
given twice, shares a run id with another, audits other code, or the number of runs is not
one or three.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

__all__ = ["ADVISORY", "AUDITS", "CHOICES", "MODES", "QUORUM", "RULE", "Advice", "AuditFinding",
           "AuditRun", "Judgement", "QuorumError", "advise", "advisory_problems",
           "confirmed_problems", "confirmed_rows", "file_keys", "gate_fork", "judge",
           "judge_paths", "load_run", "open_item", "reports_same", "row_level", "row_state",
           "same_file"]

AUDITS = 3
QUORUM = 2
TOOLS = ("architect", "improve")
RULE = ("a structural or divergence finding of the judgement half is confirmed only when at "
        f"least {QUORUM} of {AUDITS} independent fresh audits report it (the same file and the "
        "same principle anchor or category); when the first audit reports none, that one audit "
        "passes; when it reports one, two more fresh audits run one after another and the three "
        "are judged together; a confirmed finding is advisory — unattended a WARN and an open "
        "item for /architect and /improve, attended a FORK the user answers (fix now, accept "
        "and record, stop); the mechanical half (conformance.py C0-C14) blocks on any finding")
MODES = ("attended", "unattended")
# the three answers of the attended FORK, by the word an answer starts with
CHOICES = {"fix": "fix now", "accept": "accept and record", "stop": "stop"}
ADVISORY = ("advisory: a confirmed judgement finding never stops an unattended chain (a WARN and "
            "an open item handed to /architect and /improve); attended, the user answers its "
            "FORK — fix now, accept and record, or stop — and only accept and record publishes")

_EXT = r"(?:py|pyi|json|md|txt|toml|yaml|yml|cfg|ini|html|js|css|sh|ipynb|parquet|csv)"
_FILE = re.compile(r"(?<![\w.\-/])((?:[\w.\-]+/)*[\w.\-]*\w\." + _EXT + r")(?![\w])")
_SPACE = re.compile(r"\s+")
_BASELINE_DIRS = ("blueprint-runs/", ".claude/")


class QuorumError(ValueError):
    """A run that cannot be judged: unreadable, given twice, of other code (exit 3)."""


@dataclass(frozen=True)
class AuditFinding:
    """One finding of one audit, reduced to what the rule compares."""

    run: int
    id: str
    depth: str
    divergence: bool
    files: frozenset
    anchors: frozenset
    category: str
    raw: Mapping

    @property
    def key(self) -> str:
        return f"{self.run}:{self.id}"

    @property
    def blocking(self) -> bool:
        return self.depth == "structural" or self.divergence


@dataclass(frozen=True)
class AuditRun:
    """One audit: its directory, the digest of its findings.json, what it audited, its findings."""

    index: int
    path: str
    digest: str
    scope: tuple
    timestamp: str | None
    findings: tuple

    def counts(self) -> dict:
        found = self.findings
        return {"findings": len(found),
                "structural": sum(1 for f in found if f.depth == "structural"),
                "divergence": sum(1 for f in found if f.divergence),
                "blocking": sum(1 for f in found if f.blocking)}

    @property
    def run_id(self) -> str:
        """The audit's run id: the name of its run directory (`.claude/<tool>-runs/<id>`), the
        name the skill gave the run — two audits that share one wrote, or shadow, one run."""
        given = PurePosixPath(Path(self.path).as_posix())
        return (given.parent.name if given.name == "findings.json" else given.name) or self.path

    def record(self) -> dict:
        return {"audit": self.index, "run": self.path, "run_id": self.run_id,
                "digest": self.digest, "scope": list(self.scope), "run_timestamp": self.timestamp,
                **self.counts()}


@dataclass(frozen=True)
class Judgement:
    """The gate's judgement half for one tool: the record the layer carries, the procedure's
    refusals (`invalid`: the audits cannot be judged), the confirmed findings (advisory — see
    `advise`), and how many fresh audits are still to run."""

    record: dict
    invalid: tuple
    confirmed: tuple
    need: int

    @property
    def problems(self) -> tuple:
        """Everything that is not a clean pass: the refusals, then the confirmed findings."""
        return self.invalid + self.confirmed

    @property
    def passes(self) -> bool:
        """Nothing confirmed, nothing pending, nothing invalid."""
        return not self.problems and not self.need

    @property
    def judged(self) -> bool:
        """The procedure ran to its end: no fresh audit pending and nothing invalid — what the
        layer may carry, confirmed findings included (they are advisory)."""
        return not self.invalid and not self.need


@dataclass(frozen=True)
class Advice:
    """What a judged record means for the layer in the run's mode: the refusals (`errors`), the
    `WARN:` lines the layer's markers carry, the open items handed to /architect and /improve,
    and the record (`conformance.<tool>.advisory`) that states the outcome."""

    errors: tuple
    warns: tuple
    open_items: tuple
    record: dict


# --------------------------------------------------------------------------- reading a run

def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_text(v) for v in value)
    return "" if value is None else str(value)


def project_of(run_path) -> Path | None:
    """The project an audit run belongs to: the parent of the `.claude` directory its run
    directory sits under (`<project>/.claude/<tool>-runs/<ts>`), or None."""
    for parent in Path(run_path).resolve().parents:
        if parent.name == ".claude":
            return parent.parent
    return None


def _relative(path: str, project: Path | None) -> str:
    """An absolute path inside the project, spelled relative to it (F42): one audit wrote its
    scope `/…/project/bundle/x` and the others `bundle/x`, and the quorum, comparing the
    strings, refused three audits of one directory as audits of different code."""
    if project is None or not path.startswith("/"):
        return path
    try:
        return PurePosixPath(Path(path).resolve().relative_to(project.resolve())).as_posix()
    except ValueError:
        return path


_GLOB = re.compile(r"[*?\[]")


def _literal_root(path: str) -> str:
    """A scope written as a glob names the code under its literal root (F43): one audit wrote
    `pkg/**/*.py`, the others `pkg`, and the quorum, comparing the strings, refused three audits
    of one package as audits of different code. The components from the first wildcard on are
    dropped; a glob with no literal root stays as written."""
    parts = path.split("/")
    cut = next((i for i, c in enumerate(parts) if _GLOB.search(c)), None)
    if cut is None or cut == 0:
        return path
    return "/".join(parts[:cut])


def _scope_paths(doc, project: Path | None = None) -> tuple:
    scope = doc.get("scope") if isinstance(doc, Mapping) else None
    paths = scope.get("paths") if isinstance(scope, Mapping) else None
    if isinstance(paths, str):
        paths = [paths]
    return tuple(sorted({_relative(_literal_root(str(p).strip()).rstrip("/"), project).removeprefix("./")
                         for p in paths or [] if str(p).strip()}))


def _baseline_files(doc) -> frozenset:
    """The files the audit compared the code against (the blueprint run, the audit's own
    copy): named in a location as the other side of a divergence, never as where it is."""
    if not isinstance(doc, Mapping):
        return frozenset()
    scope = doc.get("scope") if isinstance(doc.get("scope"), Mapping) else {}
    said = [scope.get(k) for k in scope if str(k).startswith("baseline")]
    said += [doc.get("baseline"), doc.get("blueprint_path")]
    return frozenset(p.removeprefix("./") for p in _FILE.findall(_text(said)))


def _is_baseline(path: str, baseline: frozenset) -> bool:
    return path in baseline or path.startswith(_BASELINE_DIRS)


def file_keys(location, scope: Sequence[str] = (), baseline: frozenset = frozenset(),
              project: Path | None = None) -> frozenset:
    """The files a finding's location names, as relative POSIX paths: each audited root in
    `scope` stripped from the front, line numbers and prose dropped, and the baseline's files
    (`blueprint-runs/…`, the audit's own `.claude/…` copy) left out — they are what the code was
    compared with, not where a finding is. A location naming no file keeps its whole text
    (collapsed, lower-cased), so it matches only the same words."""
    text = _text(location)
    if project is not None:
        # a location spelled from the filesystem root names a project file by its project path
        for root in {str(project), str(project.resolve())}:
            text = text.replace(root.rstrip("/") + "/", "")
    keys = set()
    for raw in _FILE.findall(text):
        path = raw.removeprefix("./")
        if _is_baseline(path, baseline):
            continue
        for root in sorted(scope, key=len, reverse=True):
            if root and path.startswith(root + "/"):
                path = path[len(root) + 1:]
                break
        keys.add(str(PurePosixPath(path)))
    if keys:
        return frozenset(keys)
    words = _SPACE.sub(" ", text).strip().lower()
    return frozenset({"text:" + words}) if words else frozenset()


def same_file(a: str, b: str) -> bool:
    """Two paths name the same file when the shorter is the longer's tail, component by
    component (`domain.py` is `pkg/domain.py`; `model/store.py` is not `serving/store.py`)."""
    if a.startswith("text:") or b.startswith("text:"):
        return a == b
    pa, pb = PurePosixPath(a).parts, PurePosixPath(b).parts
    n = min(len(pa), len(pb))
    return n > 0 and pa[-n:] == pb[-n:]


def _finding(raw: Mapping, run: int, position: int, scope: Sequence[str],
             baseline: frozenset, project: Path | None = None) -> AuditFinding:
    tags = raw.get("tags") or []
    tags = [tags] if isinstance(tags, str) else tags
    anchors = raw.get("anchors") or []
    anchors = [anchors] if isinstance(anchors, str) else anchors
    fid = raw.get("id")
    return AuditFinding(
        run=run, id=str(fid if fid is not None else f"#{position}"),
        depth=str(raw.get("depth") or raw.get("severity") or "").strip().lower(),
        divergence=any(str(t).strip().lower() == "divergence" for t in tags),
        files=file_keys(raw.get("file") or raw.get("location") or raw.get("where"), scope,
                        baseline, project),
        anchors=frozenset(str(a).strip().upper() for a in anchors if str(a).strip()),
        category=str(raw.get("category") or raw.get("axis") or "").strip().lower(),
        raw=raw)


def load_run(path, index: int) -> AuditRun:
    """An audit run directory (or its findings.json) read into an `AuditRun`."""
    given = Path(path)
    found = given / "findings.json" if given.is_dir() else given
    if not found.is_file():
        raise QuorumError(f"audit {index}: {given} holds no findings.json — the audit's run "
                          "directory is what the gate reads")
    data = found.read_bytes()
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuorumError(f"audit {index}: {found} is not valid JSON: {exc}") from exc
    raw = doc.get("findings") if isinstance(doc, Mapping) else doc
    if not isinstance(raw, list):
        raise QuorumError(f"audit {index}: {found} carries no findings list")
    project = project_of(given)
    scope, baseline = _scope_paths(doc, project), _baseline_files(doc)
    findings = tuple(_finding(f, index, n, scope, baseline, project) for n, f in enumerate(raw, 1)
                     if isinstance(f, Mapping))
    stamp = doc.get("run_timestamp") or doc.get("run") if isinstance(doc, Mapping) else None
    return AuditRun(index=index, path=str(given), digest="sha256:" + hashlib.sha256(data).hexdigest(),
                    scope=scope, timestamp=None if stamp is None else str(stamp),
                    findings=findings)


# --------------------------------------------------------------------------- the rule

def reports_same(a: AuditFinding, b: AuditFinding) -> bool:
    """The decision's matching rule: a shared file, and a shared anchor or the same category."""
    if not any(same_file(x, y) for x in a.files for y in b.files):
        return False
    return bool(a.anchors & b.anchors) or (bool(a.category) and a.category == b.category)


def _distinct(runs: Sequence[AuditRun]) -> list:
    problems = []
    for i, run in enumerate(runs):
        for other in runs[:i]:
            if Path(run.path).resolve() == Path(other.path).resolve():
                problems.append(f"audits {other.index} and {run.index} are the same run "
                                f"({run.path}); each audit is its own fresh run")
            elif run.run_id == other.run_id:
                problems.append(f"audits {other.index} and {run.index} share the run id "
                                f"{run.run_id!r} ({other.path}, {run.path}); concurrent audits "
                                "write, or shadow, one run — run the fresh audits one after "
                                "another, each writing its own run directory")
            elif run.digest == other.digest:
                problems.append(f"audits {other.index} and {run.index} carry the same "
                                f"findings.json ({run.digest[:19]}…); a copied audit is not "
                                "an independent one")
    scopes = {run.scope for run in runs if run.scope}
    if len(scopes) > 1:
        problems.append("the audits did not audit the same code (scope.paths "
                        + " vs ".join(repr(list(s)) for s in sorted(scopes))
                        + "); after a re-emit the procedure starts over with a first audit")
    return problems


def _matching(runs: Sequence[AuditRun]) -> list:
    """One row per blocking-class finding of any audit: who reported it, and with what."""
    rows, counted = [], []
    for run in runs:
        for f in run.findings:
            if not f.blocking:
                continue
            matched = [g for other in runs if other.index != run.index
                       for g in other.findings if reports_same(f, g)]
            reported = sorted({run.index, *(g.run for g in matched)})
            confirmed = len(reported) >= QUORUM
            same_as = next((c.key for c in counted if reports_same(f, c)), None) \
                if confirmed else None
            if confirmed and same_as is None:
                counted.append(f)
            rows.append({"finding": f.key, "audit": run.index, "id": f.id, "depth": f.depth,
                         "divergence": f.divergence, "files": sorted(f.files),
                         "anchors": sorted(f.anchors), "category": f.category,
                         "title": f.raw.get("title"), "reported_by": reported,
                         "matched": [g.key for g in matched], "confirmed": confirmed,
                         "counted": confirmed and same_as is None,
                         **({"same_as": same_as} if same_as else {})})
    return rows


def _counts(rows, confirmed: bool) -> dict:
    pick = [r for r in rows if r["confirmed"] is confirmed and (r["counted"] or not confirmed)]
    return {"structural": sum(1 for r in pick if r["depth"] == "structural"),
            "divergence": sum(1 for r in pick if r["divergence"])}


def _need(runs: Sequence[AuditRun], tool: str) -> tuple:
    """(problems, fresh audits still needed) for the number of runs given."""
    first = runs[0].counts()["blocking"] if runs else 0
    if len(runs) == 1:
        return (), (AUDITS - 1 if first else 0)
    if len(runs) == AUDITS:
        return (), 0
    return ((f"{len(runs)} {tool} audits were given; the gate judges one audit (when it reports "
             f"no structural or divergence finding) or {AUDITS} — the first and two fresh "
             "ones — never another number"),), 0


def judge(runs: Sequence[AuditRun], tool: str) -> Judgement:
    """The judgement half for one tool over its audits, first audit first."""
    if not runs:
        raise QuorumError(f"no {tool} audit given")
    invalid, need = _need(runs, tool)
    invalid = list(invalid) + _distinct(runs)
    rows = _matching(runs) if not need else []
    confirmed, unconfirmed = _counts(rows, True), _counts(rows, False)
    record = {"ran": True, "tool": tool, "run": runs[0].path, "rule": RULE,
              "audits": [r.record() for r in runs], "audits_required": AUDITS if (
                  need or len(runs) > 1) else 1,
              "structural": confirmed["structural"], "divergence": confirmed["divergence"],
              "unconfirmed": unconfirmed, "matching": rows,
              "recorded": sum(1 for r in runs for f in r.findings if not f.blocking)}
    if need:
        record["pending"] = (f"the first audit reported {runs[0].counts()['blocking']} "
                             f"structural or divergence finding(s); {need} more fresh "
                             "audit(s) run, one after another, before the gate judges it")
    if invalid:
        record["invalid"] = invalid
    return Judgement(record=record, invalid=tuple(invalid), confirmed=confirmed_problems(record),
                     need=need)


def confirmed_rows(record: Mapping) -> list:
    """The confirmed findings the record counts (a repeat of a defect counted once is not one)."""
    return [row for row in record.get("matching") or ()
            if row.get("confirmed") and row.get("counted")]


def _finding_line(row, tool: str, audits: int) -> str:
    return (f"{tool} {row['finding']} ({_kind(row)}, {', '.join(row['files']) or 'no file'}: "
            f"{row.get('title') or 'untitled'}) confirmed by audits "
            f"{', '.join(map(str, row['reported_by']))} of {audits}")


def confirmed_problems(record: Mapping) -> tuple:
    """One line per confirmed finding the record counts — advisory, never a refusal by itself."""
    tool, audits = record.get("tool") or "audit", len(record.get("audits") or ())
    return tuple(_finding_line(row, tool, audits) for row in confirmed_rows(record))


def row_level(row) -> str:
    return "WARN" if row["counted"] else "OK"


def row_state(row) -> str:
    if row["counted"]:
        return "confirmed — advisory: an open item for /architect and /improve"
    if row["confirmed"]:
        return f"confirmed — the same defect as {row['same_as']}, counted there"
    return "unconfirmed — recorded, not blocking"


# --------------------------------------------------------------------------- advisory

def open_item(row, tool: str, audits: int, remedy: str) -> dict:
    """The open item a confirmed finding becomes: handed to /architect and /improve, named by
    `<tool> <finding>` in the stage brief's *Open* and the ledger's **Open**."""
    return {"tool": tool, "finding": row["finding"], "files": list(row["files"]),
            "title": row.get("title"), "reported_by": list(row["reported_by"]),
            "for": ["/architect", "/improve"], "remedy": remedy,
            "text": f"{_finding_line(row, tool, audits)} — open item for /architect and "
                    f"/improve: {remedy}"}


def _choice(answer) -> str | None:
    words = str(answer or "").strip().lower().split()
    return CHOICES.get(words[0].strip(".,:;—-")) if words else None


def gate_fork(forks, tool: str, keys: Sequence[str]):
    """The recorded answer to the attended FORK on `tool`'s confirmed findings: the last fork
    with `gate: <tool>` whose `findings` name every confirmed key and whose `answer` starts with
    one of the three choices. Returns (choice, fork), or (None, the closest fork or None)."""
    gates = [f for f in forks or () if isinstance(f, Mapping) and f.get("gate") == tool]
    for fork in reversed(gates):
        named = {str(k) for k in fork.get("findings") or ()}
        if set(keys) <= named and _choice(fork.get("answer")):
            return _choice(fork.get("answer")), fork
    return None, (gates[-1] if gates else None)


def _attended(keys, tool, forks, lines) -> tuple:
    """(errors, answer) of an attended run's confirmed findings: the user's recorded answer."""
    choice, fork = gate_fork(forks, tool, keys)
    ask = (f"FORK: {len(keys)} {tool} finding(s) confirmed ({'; '.join(lines)}) — the user "
           "answers: fix now, accept and record, or stop; the answer is recorded as a `forks[]` "
           f"entry {{\"gate\": \"{tool}\", \"findings\": {json.dumps(list(keys))}, \"asked\": …, "
           "\"answer\": \"accept and record — <the user's words>\"}")
    if choice is None:
        why = ("its fork carries `or` (a default) — attended, the answer is the user's"
               if fork is not None and fork.get("or") else
               "no recorded answer names every confirmed finding" if fork is not None else
               "no answer is recorded")
        return (f"conformance.{tool}: {why}; {ask}",), None
    if choice == "fix now":
        return (f"conformance.{tool}: the user answered fix now — fix the template through "
                f"/architect or /improve, re-emit, and run the gate again; nothing is "
                "published until then",), fork.get("answer")
    if choice == "stop":
        return (f"conformance.{tool}: the user answered stop — the stage ends here and "
                "publishes nothing",), fork.get("answer")
    return (), fork.get("answer")


def advise(record: Mapping, mode: str, forks=(), remedy: str = "") -> Advice:
    """The advisory half of the decision for one judged tool record: unattended, every confirmed
    finding is a WARN and an open item and nothing refuses; attended, the user's recorded answer
    to the FORK decides — *accept and record* publishes with the WARN and the open item, *fix
    now* and *stop* (and no answer) refuse. A pending or invalid record is not advised on: the
    caller refuses it before."""
    if mode not in MODES:
        raise QuorumError(f"mode {mode!r} is not one of {', '.join(MODES)}")
    tool, audits = record.get("tool") or "audit", len(record.get("audits") or ())
    rows = confirmed_rows(record)
    keys = [row["finding"] for row in rows]
    items = tuple(open_item(row, tool, audits, remedy) for row in rows)
    errors, answer = (), None
    if rows and mode == "attended":
        errors, answer = _attended(keys, tool, forks, [_finding_line(r, tool, audits) for r in rows])
    outcome = ("clear" if not rows else "warn" if mode == "unattended"
               else "accepted" if not errors else "refused")
    said = "advisory (unattended)" if mode == "unattended" else "accepted and recorded (the user)"
    warns = () if errors else tuple(f"WARN: {item['text']} — {said}; the chain continues"
                                    for item in items)
    return Advice(errors=errors, warns=warns, open_items=items if not errors else (),
                  record={"mode": mode, "outcome": outcome, "confirmed": keys,
                          "answer": answer, "open_items": list(items) if not errors else [],
                          "rule": ADVISORY})


def advisory_problems(record: Mapping) -> list:
    """A judged record's advisory block, checked against its matching: confirmed findings carry
    an `advisory` whose outcome is `warn` (unattended) or `accepted` (attended, with the
    answer), naming exactly the confirmed findings, each an open item — else the record is
    malformed. Nothing confirmed needs no advisory."""
    tool = record.get("tool") or "audit"
    keys = sorted(row["finding"] for row in confirmed_rows(record))
    advisory = record.get("advisory")
    if not keys:
        return []
    if not isinstance(advisory, Mapping):
        return [f"conformance.{tool} confirms {', '.join(keys)} but carries no advisory outcome; "
                "the record is the apply script's, never typed"]
    problems = []
    if advisory.get("outcome") not in ("warn", "accepted"):
        problems.append(f"conformance.{tool}.advisory.outcome is {advisory.get('outcome')!r}, "
                        "not warn (unattended) or accepted (attended)")
    if advisory.get("outcome") == "accepted" and not advisory.get("answer"):
        problems.append(f"conformance.{tool}.advisory is accepted with no recorded answer")
    if sorted(map(str, advisory.get("confirmed") or ())) != keys:
        problems.append(f"conformance.{tool}.advisory names {advisory.get('confirmed')}, not the "
                        f"confirmed findings {keys}")
    named = sorted(str((i or {}).get("finding")) for i in advisory.get("open_items") or ())
    if named != keys:
        problems.append(f"conformance.{tool}.advisory.open_items name {named}, not {keys}: every "
                        "confirmed finding is an open item")
    return problems


def _kind(row) -> str:
    kinds = [k for k, on in (("structural", row["depth"] == "structural"),
                              ("divergence", row["divergence"])) if on]
    return " and ".join(kinds)


def judge_paths(paths: Iterable, tool: str) -> Judgement:
    """`judge` over run directories given in order, the first audit first."""
    return judge([load_run(p, n) for n, p in enumerate(paths, 1)], tool)


# --------------------------------------------------------------------------- the command

def _say(j: Judgement, tool: str, mode: str) -> int:
    rec = j.record
    for audit in rec["audits"]:
        print(f"OK: {tool} audit {audit['audit']}: {audit['run']} (run id {audit['run_id']}) — "
              f"{audit['findings']} finding(s), {audit['structural']} structural, "
              f"{audit['divergence']} divergence")
    if j.invalid:
        for problem in j.invalid:
            print(f"ERROR: {problem}")
        return 3
    if j.need:
        print(f"NEED: {rec['pending']} — run /{tool} --report-only {j.need} more time(s), one "
              "after another, each in a fresh context that has not read the earlier audits and "
              "writing its own run directory, then pass all the runs in order")
        return 1
    for row in rec["matching"]:
        print(f"{row_level(row)}: {row['finding']} {_kind(row)} — reported by audit(s) "
              f"{row['reported_by']}: {row_state(row)}")
    if j.confirmed:
        if mode == "unattended":
            for line in j.confirmed:
                print(f"WARN: {line} — advisory (unattended): an open item for /architect and "
                      "/improve; the chain continues")
        else:
            print(f"FORK: {len(j.confirmed)} {tool} finding(s) confirmed — "
                  + "; ".join(j.confirmed) + " — fix now (the template, then re-emit), accept "
                  "and record (published with a WARN and an open item), or stop; the user "
                  "answers, and the answer is recorded in the layer's forks")
        return 2
    print(f"OK: the {tool} half passes — {rec['unconfirmed']['structural']} structural and "
          f"{rec['unconfirmed']['divergence']} divergence finding(s) unconfirmed and recorded")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("runs", nargs="+", help="the audits' run directories, the first audit first")
    ap.add_argument("--tool", required=True, choices=TOOLS)
    ap.add_argument("--mode", default="attended", choices=MODES,
                    help="how a confirmed finding is said: unattended a WARN and an open item, "
                         "attended the FORK the user answers (default attended)")
    ap.add_argument("--json", default=None, help="also write the record here ('-': stdout only)")
    a = ap.parse_args(argv)
    try:
        j = judge_paths(a.runs, a.tool)
    except QuorumError as exc:
        print(f"ERROR: {exc}")
        return 3
    if a.json == "-":
        print(json.dumps(j.record, indent=1))
        return 0 if j.passes else (3 if j.invalid else 1 if j.need else 2)
    if a.json:
        Path(a.json).write_text(json.dumps(j.record, indent=1) + "\n", encoding="utf-8")
    return _say(j, a.tool, a.mode)


if __name__ == "__main__":
    sys.exit(main())
