"""
_negatives_kit — what `run_negatives.py`'s cases share: a lane's work staged once through the
stage (extract → draft → author → resolve → build → check, and the served walk when asked),
restored at the same path before every case, the stage's command lines run as a user runs them,
and the small seeds the cases plant.

    staged = Staged.of(negkit_lane, tmp, walk=True)    # None when the lane cannot be staged
    with staged.fresh() as w:                           # the staged copy, as staged, at one path
        w.edit_json(w.plan / "story-plan.json", lambda p: ...)
        code, out = w.resolve()

The stage records absolute paths (the fact set names its bundle and model page, SHOWCASE.json
its report sources), so a case never works on a copy at another path: the staged directory is
snapshotted once and put back, byte for byte, at the same path before each case.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = HERE.parents[2]
TEMPLATE = SCRIPTS.parent / "assets" / "page"
CATALOG = json.loads((SCRIPTS / "catalog.json").read_text(encoding="utf-8"))
PY = sys.executable
TIMEOUT_S = 900


class StageError(Exception):
    """A lane could not be staged (a step of the stage refused the unseeded lane)."""


def run(argv, cwd=None, env: Optional[dict] = None, timeout: int = TIMEOUT_S) -> tuple[int, str]:
    """A command line: (exit, stdout and stderr together)."""
    full = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env or {}))
    try:
        done = subprocess.run([str(a) for a in argv], cwd=str(cwd) if cwd else None, env=full,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return 124, f"timed out after {timeout} s: {exc}"
    return done.returncode, done.stdout + done.stderr


def show(script: str, *args, cwd=None, env=None, timeout: int = TIMEOUT_S) -> tuple[int, str]:
    """A vega-show script (`scripts/<script>`, or `tests/<script>`) as the stage runs it."""
    return run([PY, SCRIPTS / script, *args], cwd=cwd, env=env, timeout=timeout)


def py(code: str, *args, cwd=None, env=None, timeout: int = TIMEOUT_S) -> tuple[int, str]:
    """A child Python with the vega-show scripts first on its path (a seeded code path)."""
    head = f"import sys; sys.path[:0] = [{str(SCRIPTS)!r}, {str(HERE)!r}]\n"
    return run([PY, "-c", head + code, *args], cwd=cwd, env=env, timeout=timeout)


def sha(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, doc) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def edit_json(path: Path, edit: Callable) -> None:
    doc = read_json(path)
    edit(doc)
    write_json(path, doc)


def edit_text(path: Path, old: str, new: str) -> None:
    text = Path(path).read_text(encoding="utf-8")
    if old not in text:
        raise AssertionError(f"{Path(path).name} no longer holds {old!r}: re-anchor the seed")
    Path(path).write_text(text.replace(old, new, 1), encoding="utf-8")


def template_copy(dest: Path, seed: Optional[Callable[[Path], None]] = None, pin: bool = True) -> Path:
    """A copy of the installed page template, seeded, its TEMPLATE.json rewritten (`pin`) so
    the copy is a template in its own right."""
    page = Path(shutil.copytree(TEMPLATE, Path(dest) / "page"))
    if seed:
        seed(page)
    if pin:
        code, out = show("page_template.py", "--template", page, "--write")
        if code != 0:
            raise StageError(f"page_template.py --write on the seeded copy: {out[-800:]}")
    return page


# --------------------------------------------------------------------------- a staged lane

@dataclass
class Work:
    """A lane's staged work (`<root>/work`, the report in `<root>/docs`, the plan in
    `<root>/plan`) and the stage's commands on it."""
    lane: str
    root: Path

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def plan(self) -> Path:
        return self.root / "plan"

    @property
    def bundle(self) -> Path:
        return next(p.parent for p in sorted((self.work / "bundle").glob("*/MANIFEST.json")))

    @property
    def model(self) -> Path:
        return next(p for p in sorted(self.work.glob("*.pack.html")))

    @property
    def showcase(self) -> Path:
        return self.bundle / "showcase"

    @property
    def manifest(self) -> Path:
        return self.bundle / "MANIFEST.json"

    @property
    def linked(self) -> bool:
        return (self.docs / "label-roles.json").is_file()

    def report(self) -> list:
        """The report arguments: the docs directory when it holds a compiled report."""
        return ["--report", self.docs] if (self.docs / "main.aux").is_file() else []

    # ---- the stage, step by step
    def facts(self, *extra) -> tuple[int, str]:
        return show("extract_facts.py", "--model", self.model, "--bundle", self.bundle, *self.report(),
                    "--out", self.plan / "facts.json", *extra, cwd=self.work)

    def draft(self, template: Optional[Path] = None, unattended: bool = True) -> tuple[int, str]:
        return show("draft_plan.py", "--facts", self.plan / "facts.json",
                    "--page-manifest", Path(template or TEMPLATE) / "TEMPLATE.json",
                    "--memory", self.work / ".claude" / "domain-forge-memory.md",
                    *(["--unattended"] if unattended else []), "--out", self.plan / "draft.json", cwd=self.work)

    def author(self) -> tuple[int, str]:
        return show("tests/autoauthor.py", "--plan", self.plan / "draft.json", "--facts", self.plan / "facts.json",
                    "--out", self.plan / "story-plan.json", *self.report(), cwd=self.work)

    def resolve(self, plan: Optional[Path] = None, template: Optional[Path] = None,
                facts: Optional[Path] = None, out: Optional[Path] = None) -> tuple[int, str]:
        return show("resolve_story.py", "--plan", plan or self.plan / "story-plan.json",
                    "--facts", facts or self.plan / "facts.json",
                    "--page-manifest", Path(template or TEMPLATE) / "TEMPLATE.json", *self.report(),
                    "--out", out or self.plan / "story.json", cwd=self.work)

    def build(self, *extra, unattended: bool = True) -> tuple[int, str]:
        return show("build_showcase.py", "--bundle", self.bundle, "--model", self.model,
                    "--plan", self.plan / "story-plan.json", "--story", self.plan / "story.json",
                    "--facts", self.plan / "facts.json", *self.report(), "--out", self.showcase,
                    *(["--unattended"] if unattended else []), *extra, cwd=self.work)

    def check(self, *extra) -> tuple[int, str]:
        return show("build_showcase.py", "--check", self.showcase, *self.report(), *extra, cwd=self.work)

    def walk(self, timeout: int = TIMEOUT_S) -> tuple[int, str]:
        return show("smoke_show.py", "--serve", self.bundle, "--strict", cwd=self.work, timeout=timeout,
                    env={"CUDA_VISIBLE_DEVICES": ""})

    def layer(self, *extra) -> tuple[int, str]:
        return show("apply_show_layer.py", self.model, "--data", self.layer_data, "--out", self.layered, *extra,
                    cwd=self.work)

    @property
    def layer_data(self) -> Path:
        return self.showcase / ".work" / "showcase-layer.json"

    @property
    def layered(self) -> Path:
        return self.model.with_name(self.model.name[: -len(".html")] + ".show.html")

    def stage(self, template: Optional[Path] = None, steps=("facts", "draft", "author", "resolve", "build",
                                                              "check")) -> None:
        """Run the stage's steps; any refusal is a StageError naming the step."""
        calls = {"facts": self.facts, "draft": lambda: self.draft(template), "author": self.author,
                 "resolve": lambda: self.resolve(template=template),
                 "build": lambda: self.build(*(["--template", template] if template else [])),
                 "check": self.check}
        for step in steps:
            code, out = calls[step]()
            if code != 0:
                raise StageError(f"{self.lane}: {step} exited {code}: {out.strip()[-1500:]}")

    # ---- seeds
    def repin(self, *rels: str) -> None:
        """SHOWCASE.json pins the showcase files `rels` as they now are (a hand edit with its
        digest rebuilt, D5-32's second half): the slot verifies the edited file."""
        def pin(doc):
            for rel in rels:
                doc["files"][rel] = sha((self.showcase / rel).read_bytes())
        edit_json(self.showcase / "SHOWCASE.json", pin)

    def edit_layer(self, edit: Callable[[dict], None]) -> None:
        """Edit the model page's bundle layer data (`layer-bundle-data`) in place."""
        html = self.model.read_text(encoding="utf-8")
        head = '<script id="layer-bundle-data" type="application/json">'
        start = html.index(head) + len(head)
        end = html.index("</script>", start)
        data = json.loads(html[start:end])
        edit(data)
        self.model.write_text(html[:start] + json.dumps(data, ensure_ascii=False) + html[end:], encoding="utf-8")

    def relayer(self, edit: Callable[[dict], None]) -> None:
        """Edit MANIFEST.json and the model page's bundle layer alike (the page then still
        describes this bundle, so the refusal is the one the edit is about, not PF1)."""
        edit_json(self.manifest, edit)
        manifest = read_json(self.manifest)
        self.edit_layer(lambda data: data.update(manifest=manifest))

    def repin_bundle(self, *rels: str) -> None:
        """MANIFEST.files pins the bundle files `rels` as they now are (and the layer says so)."""
        def pin(doc):
            for rel in rels:
                doc["files"][rel] = sha((self.bundle / rel).read_bytes())
        self.relayer(pin)


@dataclass
class Staged:
    """A lane staged once and snapshotted; `fresh()` puts it back at its path for a case."""
    lane: str
    root: Path
    snap: Path
    walked: bool

    @classmethod
    def of(cls, lane, tmp: Path, walk: bool = False, say: Callable[[str], None] = print,
           docs: bool = True, prepare: Optional[Callable[["Work"], None]] = None) -> "Staged":
        """Copy the lane's work (and its report, `docs`), `prepare` the copy, stage it."""
        root, snap = Path(tmp) / lane.id, Path(tmp) / f"{lane.id}.snap"
        root.mkdir(parents=True)
        shutil.copytree(lane.root / "work", root / "work", symlinks=True)
        if docs and (lane.root / "docs").is_dir():
            shutil.copytree(lane.root / "docs", root / "docs", symlinks=True)
        w = Work(lane.id, root)
        for old in [*w.bundle.glob("showcase*"), *w.work.glob("*.pack.show.html")]:
            shutil.rmtree(old) if old.is_dir() else old.unlink()
        if prepare:
            prepare(w)
        w.stage()
        say(f"OK: {lane.id} staged: {w.bundle.name}/showcase built and checked "
            f"({'a linked report' if w.linked else 'brief links'})")
        if walk:
            code, out = w.walk()
            if code != 0:
                raise StageError(f"{lane.id}: the served walk of the unseeded showcase exited {code}: {out[-1500:]}")
            say(f"OK: {lane.id}: the unseeded showcase passes the served walk")
        shutil.copytree(root, snap, symlinks=True)
        return cls(lane.id, root, snap, walk)

    @contextlib.contextmanager
    def fresh(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.copytree(self.snap, self.root, symlinks=True)
        yield Work(self.lane, self.root)


def classes_of(w: Work) -> list:
    return list(read_json(w.plan / "facts.json")["task"].get("classes") or [])


def shaped_inputs(w: Work) -> list:
    return list(read_json(w.plan / "facts.json")["groups"]["inputs"]["shaped"])


def seeded_plan(w: Work, mutate: Callable[[dict], None], name: str = "seeded.json") -> Path:
    """A copy of the authored plan with one defect; its path."""
    plan = read_json(w.plan / "story-plan.json")
    mutate(plan)
    out = w.plan / name
    write_json(out, plan)
    return out


# --------------------------------------------------------------------------- the cases

@dataclass(frozen=True)
class Expect:
    """The gate refused with `code`, saying every needle of `contains` and none of `absent`."""
    rc: int
    out: str
    code: int = 2
    contains: tuple = ()
    absent: tuple = ()


@dataclass(frozen=True)
class Check:
    """A verdict the case decided itself: `detail` says what is wrong when it is not ok."""
    ok: bool
    detail: str = ""
    out: str = ""


@dataclass(frozen=True)
class Case:
    """One seeded defect: the gate that must catch it, the design cases it realises, the lane
    it is seeded on (None: no lane, the skill's own files), and the seed and its verdict."""
    gate: str
    ids: tuple
    lane: Optional[str]
    what: str
    fn: Callable

    @property
    def name(self) -> str:
        """`<gate>  <design ids>: <seed>`; a case caught outside vega-show (no gate of its catalog)
        starts with its design id and is recorded under the suite that catches it."""
        ids = " ".join(self.ids) + ": " if self.ids else ""
        return f"{self.gate}  {ids}{self.what}" if self.gate else f"{ids}{self.what}"


def case(gate: str, ids: str, lane: Optional[str], what: str):
    """Decorator: the function is the case's seed and verdict."""
    def wrap(fn):
        return Case(gate, tuple(ids.split()), lane, what, fn)
    return wrap


def refused(rc_out: tuple, gate: str, *words, code: int = 2, absent=()) -> Expect:
    """The command refused with `ERROR: <gate> …` (exit `code`) and named `words`."""
    rc, out = rc_out
    return Expect(rc, out, code, (f"ERROR: {gate} ", *words), absent)
