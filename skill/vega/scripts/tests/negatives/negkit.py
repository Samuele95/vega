"""
negkit — what every seeded-negative suite shares (D6 §12, BUILD-PLAN R-H6, T10).

    cases = negkit.Cases(script_path, work, verbose=False)
    cases.section("pack.py — what ships is what was evaluated", gate="vega-pack/pack.py")
    cases.expect("N-S1  a stored layout without row_score is refused", rc, out, code=3,
                 contains=("per-row score",))
    cases.check("SS8  a verified showcase leaves MANIFEST.json as it was", ok, detail)
    cases.not_run("N-S3  the transductive lane is refused", "lane L23 is not in this matrix")
    sys.exit(cases.finish(json_path))                   # 0 when every case that ran passed

    negkit.case_id(name) -> "N-S1" | "SS3" | None       the id a case name starts with
    negkit.problems(doc) -> [problem, ...]              `vega/negatives-result@1`, checked
    negkit.lane(matrix, "L19") -> Lane | None           a lane's work dir in a matrix dir

**The id convention** (D6 §12): a new case's name starts with its gate id — `[A-Z]{1,3}\\d+`
or the pack's `N-S\\d+`, an optional lower-case letter — then two spaces and a sentence. A case
of a gate that predates v2.3 carries no id; it is listed in `grandfathered.json` beside this
module under its gate (`{gate: [case names]}`), and its record carries that gate.

**The result** (`--json <path>`): `vega/negatives-result@1` — `script` (skills-relative),
`work`, `cases[] {id, name, ok, detail, gate?}` in the order they ran, `passed`, `failed`, and
`not_run[] {id, name, why}`: a case that could not run here (a lane the matrix does not hold)
is said, never counted as passed. The coverage report reads these files and requires every
new gate id to be the id of at least one passing case (D6 §12.8).

A case passes when the gate refused with the expected exit code and every needle
(case-insensitive) is in what it said, and none of the `absent` ones is.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

__all__ = ["CASE_ID", "SCHEMA", "Cases", "Lane", "case_id", "grandfathered", "lane", "lane_in",
           "matrix_of", "problems", "result_doc"]

SCHEMA = "vega/negatives-result@1"
CASE_ID = re.compile(r"^((?:N-)?[A-Z]{1,3}\d+[a-z]?)\s{2}\S")
HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[3]
GRANDFATHERED = HERE / "grandfathered.json"
_CASE_KEYS = {"id", "name", "ok", "detail"}
_OPTIONAL_CASE_KEYS = {"gate"}
_TOP_KEYS = {"schema", "script", "work", "cases", "passed", "failed"}
_OPTIONAL_TOP_KEYS = {"not_run"}


def case_id(name: str) -> Optional[str]:
    """The gate id a case name starts with (`N-S1`, `SS3`, `W2a`), or None."""
    m = CASE_ID.match(name or "")
    return m.group(1) if m else None


def grandfathered() -> dict:
    """`{gate: [case names]}`: the pre-v2.3 cases, listed under their gate, with no id."""
    return json.loads(GRANDFATHERED.read_text(encoding="utf-8"))


def _relative(script: Path) -> str:
    try:
        return Path(script).resolve().relative_to(SKILLS).as_posix()
    except ValueError:
        return Path(script).name


class Cases:
    """One suite's cases, printed as they run and kept for `--json`."""

    def __init__(self, script, work, verbose: bool = False, head: bool = False):
        self.script, self.work, self.verbose, self.head = Path(script), str(work), verbose, head
        self.results: list[dict] = []
        self.skipped: list[dict] = []
        self.gate: Optional[str] = None

    # -- sections ------------------------------------------------------------------------
    def section(self, title: str, gate: Optional[str] = None) -> None:
        """A heading; `gate` is what a case without an id is grandfathered under."""
        print(f"\n{title}")
        self.gate = gate

    # -- verdicts ------------------------------------------------------------------------
    def expect(self, name, rc, out, *, code=2, contains=(), absent=()) -> bool:
        """The gate refused with `code` and said every needle (and none of `absent`)."""
        problems_ = [] if rc == code else [f"exit {rc}, expected {code}"]
        low = (out or "").lower()
        problems_ += [f"message does not mention {n!r}" for n in contains if n.lower() not in low]
        problems_ += [f"message should not mention {n!r}" for n in absent if n.lower() in low]
        return self.record(name, problems_, out)

    def check(self, name, ok_: bool, detail: str = "", out: str = "") -> bool:
        """A verdict decided by the caller: `detail` says what is wrong when it is not ok."""
        return self.record(name, [] if ok_ else [detail or "not as expected"], out)

    def record(self, name, problems_, out="") -> bool:
        entry = {"id": case_id(name), "name": name, "ok": not problems_,
                 "detail": "; ".join(problems_)}
        if entry["id"] is None and self.gate:
            entry["gate"] = self.gate
        self.results.append(entry)
        line = f"  {'ok  ' if not problems_ else 'FAIL'}  {name}"
        print(line + (f"  — {entry['detail']}" if problems_ else ""))
        if out and (self.verbose or problems_):
            lines = str(out).strip().splitlines()
            shown = lines[:6] if self.head else lines[-8:]
            print("        " + "\n        ".join(shown))
        return not problems_

    def not_run(self, name, why: str) -> None:
        """A case this run could not seed (what it needs is absent): said, never passed."""
        self.skipped.append({"id": case_id(name), "name": name, "why": why})
        print(f"  WARN  not run: {name} — {why}")

    # -- the end -------------------------------------------------------------------------
    def failed(self) -> list:
        return [r for r in self.results if not r["ok"]]

    def finish(self, json_path=None) -> int:
        """The summary line, the `--json` file, the exit code (1 when a case failed)."""
        failed = self.failed()
        print(f"\n{len(self.results) - len(failed)} of {len(self.results)} seeded defects were "
              "caught by the gate that should catch them"
              + (f"; {len(self.skipped)} case(s) not run here" if self.skipped else ""))
        if json_path:
            doc = result_doc(self.script, self.work, self.results, self.skipped)
            wrong = problems(doc)
            if wrong:
                raise AssertionError(f"the result is not {SCHEMA}: {wrong}")
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            Path(json_path).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
            print(f"wrote {json_path} ({SCHEMA})")
        if failed:
            print("\nnot caught:")
            for r in failed:
                print(f"  - {r['name']}: {r['detail']}")
            return 1
        return 0


def result_doc(script, work, results: Iterable[Mapping], skipped: Iterable[Mapping] = ()) -> dict:
    cases = [dict(r) for r in results]
    passed = sum(1 for c in cases if c["ok"])
    doc = {"schema": SCHEMA, "script": _relative(Path(script)), "work": str(work),
           "cases": cases, "passed": passed, "failed": len(cases) - passed}
    skipped = [dict(s) for s in skipped]
    if skipped:
        doc["not_run"] = skipped
    return doc


def problems(doc: Any) -> list:
    """Every way `doc` is not `vega/negatives-result@1` (empty: it is)."""
    if not isinstance(doc, Mapping):
        return ["the result is not an object"]
    out = []
    missing = sorted(_TOP_KEYS - set(doc))
    extra = sorted(set(doc) - _TOP_KEYS - _OPTIONAL_TOP_KEYS)
    out += [f"missing key {k!r}" for k in missing] + [f"unknown key {k!r}" for k in extra]
    if doc.get("schema") != SCHEMA:
        out.append(f"schema is {doc.get('schema')!r}, not {SCHEMA!r}")
    for key in ("script", "work"):
        if key in doc and (not isinstance(doc[key], str) or not doc[key]):
            out.append(f"{key} is not a non-empty string")
    cases = doc.get("cases")
    if not isinstance(cases, list):
        return out + ["cases is not a list"]
    for n, case in enumerate(cases):
        out += [f"cases[{n}]: {p}" for p in _case_problems(case)]
    ok_n = sum(1 for c in cases if isinstance(c, Mapping) and c.get("ok") is True)
    for key, want in (("passed", ok_n), ("failed", len(cases) - ok_n)):
        value = doc.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != want:
            out.append(f"{key} is {value!r}; the cases say {want}")
    for n, entry in enumerate(doc.get("not_run") or ()):
        if not isinstance(entry, Mapping) or set(entry) != {"id", "name", "why"} or \
                not isinstance(entry.get("name"), str) or not isinstance(entry.get("why"), str):
            out.append(f"not_run[{n}] is not {{id, name, why}}")
    if "not_run" in doc and not isinstance(doc["not_run"], list):
        out.append("not_run is not a list")
    return out


def _case_problems(case) -> list:
    if not isinstance(case, Mapping):
        return ["not an object"]
    out = [f"missing key {k!r}" for k in sorted(_CASE_KEYS - set(case))]
    out += [f"unknown key {k!r}" for k in sorted(set(case) - _CASE_KEYS - _OPTIONAL_CASE_KEYS)]
    if not isinstance(case.get("name"), str) or not case.get("name"):
        out.append("name is not a non-empty string")
    if case.get("id") is not None and not isinstance(case.get("id"), str):
        out.append("id is neither a string nor null")
    if not isinstance(case.get("ok"), bool):
        out.append("ok is not a boolean")
    if not isinstance(case.get("detail"), str):
        out.append("detail is not a string")
    if case.get("ok") is False and not case.get("detail"):
        out.append("a failed case says nothing about why")
    if "gate" in case and not isinstance(case["gate"], str):
        out.append("gate is not a string")
    return out


# --------------------------------------------------------------------------- lanes

@dataclass(frozen=True)
class Lane:
    """A lane's work in a matrix dir (`<matrix>/lanes/<id>/`), read off its `fixture.json`."""
    id: str
    root: Path
    fx: Mapping

    @property
    def work(self) -> Path:
        return Path(self.fx["work"])

    @property
    def campaign(self) -> Path:
        return Path(self.fx["run_dir"])

    @property
    def shaped(self) -> Path:
        return Path(self.fx["shaped_dir"])

    @property
    def pipeline(self) -> Path:
        return Path(self.fx["pipeline"])

    @property
    def netspec(self) -> Path:
        return Path(self.fx["netspec"])

    @property
    def model(self) -> Path:
        """The shaped model the fixture wrote (what the compiler reads)."""
        return Path(self.fx["model_html"])

    def layered(self, suffix: str) -> Path:
        base = self.model
        return base.with_name(base.name[: -len(".html")] + suffix + ".html")

    @property
    def memory(self) -> Path:
        return self.work / ".claude" / "domain-forge-memory.md"

    def evaluated(self) -> bool:
        return (self.campaign / "test_seal.json").is_file() and \
            self.layered(".vega.train.eval").is_file()

    def bundle(self) -> Optional[Path]:
        """The lane's packed bundle (`<work>/bundle/<slug>`), or None."""
        root = self.work / "bundle"
        found = sorted(p.parent for p in root.glob("*/MANIFEST.json")) if root.is_dir() else []
        return found[0] if found else None


def lane(matrix, lane_id: str) -> Optional[Lane]:
    """`lane_id`'s work in the matrix dir, or None when the lane has no fixture there."""
    if matrix is None:
        return None
    return lane_in(Path(matrix) / "lanes" / lane_id, lane_id)


def lane_in(root, lane_id: str) -> Optional[Lane]:
    """The lane whose `fixture.json` is in `root`, or None."""
    root = Path(root)
    fx = root / "fixture.json"
    if not fx.is_file():
        return None
    doc = json.loads(fx.read_text(encoding="utf-8"))
    if not Path(doc.get("netspec", "")).is_file():
        return None
    return Lane(lane_id, root, doc)


def matrix_of(work) -> Optional[Path]:
    """The matrix dir a lane's work dir sits in (`<matrix>/lanes/<id>/work`), or None."""
    work = Path(work).resolve()
    if work.name == "work" and work.parent.parent.name == "lanes":
        return work.parents[2]
    return None
