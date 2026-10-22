"""
seeds.py — the seeded defects of the size discipline (D6 §12.1: C0a, C11a–C14e), in one place.

`run_negatives.py` plants them in copies of the acceptance fixture's package (today's
templates); `test_conformance_sizes.py` plants them in copies of the clean `package/` beside
this file. A seed never names a file: it asks the package's `blueprint-map.json` which file
plays a role, and which class in it is the adapter, so the same seed works on any package the
emitter writes — function-shaped adapters (v2.2) and class-shaped ones (v2.3) alike.

Every seed is judged under `--size-severity error`. A *refused* seed must make the gate exit 2
and name what was seeded (`present`); a *boundary* seed must leave the gate's exit where the
unseeded package left it and must not be named by any line (`absent`).

`SEEDS` is D6 §12.1's list, which both scripts plant; `scenarios.py` builds its own seeds
beyond that list with `store_module` and `append`.

This module is a library of the two scripts, not a measurement fixture.
"""
from __future__ import annotations

import ast
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import conformance  # noqa: E402

CONFORMANCE = SCRIPTS / "conformance.py"
HELPER_CLASS = "_SeededStore"


class SeedError(RuntimeError):
    """A seed that cannot be planted in this package, with the reason."""


@dataclass(frozen=True)
class Planted:
    """What a planted seed asks of the gate's output."""

    args: tuple = ()        # extra conformance.py arguments (a later --pipeline wins)
    present: tuple = ()     # needles the refusal names
    absent: tuple = ()      # needles no output line may contain


@dataclass(frozen=True)
class Seed:
    """One seeded defect: its id (D6 §12 convention), what it proves, how it is planted."""

    id: str
    sentence: str
    refused: bool
    plant: Callable
    wants_bundle: bool = False

    @property
    def name(self) -> str:
        return f"{self.id}  {self.sentence}"


# --------------------------------------------------------------------------- the package

def _map(root: Path) -> dict:
    return json.loads((root / "blueprint-map.json").read_text(encoding="utf-8"))


def _entry(root: Path, role: str) -> dict:
    entry = next((e for e in _map(root).get("entries") or [] if e.get("role") == role), None)
    if entry is None or not (root / entry["file"]).is_file():
        raise SeedError(f"the package maps no file for role {role}")
    return entry


def _support(root: Path, basename: str) -> str:
    for entry in _map(root).get("entries") or []:
        rel = str(entry.get("file") or "")
        if entry.get("kind") == "support" and Path(rel).name == basename:
            return rel
    raise SeedError(f"the package maps no support file named {basename}")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n") + "\n\n\n" + text,
                    encoding="utf-8")


def _insert_after(path: Path, line_no: int, text: str) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    path.write_text("\n".join(lines[:line_no] + text.rstrip("\n").split("\n")
                              + lines[line_no:]), encoding="utf-8")


def store_module(root: Path) -> tuple:
    """(path, rel, the ModelStore adapter class node or None)."""
    entry = _entry(root, "ModelStore")
    path = root / entry["file"]
    return path, entry["file"], conformance.adapter_class(_tree(path), "ModelStore",
                                                           entry.get("port"))


def _into_store_class(root: Path, methods: str) -> tuple:
    """Put `methods` (indented for a class body) into the ModelStore adapter class — or, in a
    package whose store has no adapter class yet, into a `_prefixed` helper class appended to
    the module, so that no public name is added. Returns (rel, the class's name)."""
    path, rel, cls = store_module(root)
    if cls is None:
        append(path, f"class {HELPER_CLASS}:\n{methods}")
        return rel, HELPER_CLASS
    _insert_after(path, cls.end_lineno, "\n" + methods)
    return rel, cls.name


def _statements(n: int, indent: str) -> str:
    return "".join(f"{indent}x = 1\n" for _ in range(n))


# --------------------------------------------------------------------------- padding

def _pad_store_class(root: Path, total: int) -> tuple:
    path, rel, cls = store_module(root)
    if cls is None:
        body = "".join(f"    def _p{i:03d}(self):  return {i}\n" for i in range(1, total))
        append(path, f"class {HELPER_CLASS}:\n{body}")
        return rel, HELPER_CLASS
    measure = conformance.measure_module(_tree(path), rel).klass(cls.name)
    missing = total - measure.lines
    if missing < 1:
        raise SeedError(f"{rel}'s {cls.name} is already {measure.lines} lines; it cannot be "
                        f"padded to {total}")
    body = "".join(f"    def _p{i:03d}(self):  return {i}\n" for i in range(1, missing + 1))
    _insert_after(path, cls.end_lineno, body)
    return rel, cls.name


def _pad_module(path: Path, total: int) -> None:
    """Append UPPER_CASE constants until the module's last statement is on line `total`."""
    text = path.read_text(encoding="utf-8").rstrip()
    missing = total - len(text.split("\n"))
    if missing < 1:
        raise SeedError(f"{path.name} already runs to line {total} or past it")
    pad = "".join(f"\nSEEDED_PAD_{i:03d} = {i}" for i in range(1, missing + 1))
    path.write_text(text + pad + "\n", encoding="utf-8")


def _pad_module_past(path: Path, total: int) -> None:
    """Pad to `total` lines, or — a module already past it — add one constant (it stays past
    the module limit, which is all an exemption case needs)."""
    text = path.read_text(encoding="utf-8").rstrip()
    if len(text.split("\n")) < total:
        _pad_module(path, total)
    else:
        path.write_text(text + "\nSEEDED_PAD_000 = 0\n", encoding="utf-8")


# --------------------------------------------------------------------------- the seeds

def _c0a(root, pipeline, scratch):
    path, rel, _ = store_module(root)
    append(path, "def broken(:\n    pass\n")
    return Planted(present=(f"C0 {rel}:", "does not parse"))


def _c11a(root, pipeline, scratch):
    path, rel, _ = store_module(root)
    append(path, "def _seeded_long():\n" + _statements(31, "    "))
    return Planted(present=(f"C11 {rel}:", "_seeded_long — 31 statements > 30 — EJ-51"))


def _c11b(root, pipeline, scratch):
    path, _, _ = store_module(root)
    comments = "".join(f"    # comment {i}: not a statement\n" for i in range(1, 6))
    append(path, "def _seeded_long():\n    \"\"\"A docstring the counter does not count.\"\"\"\n"
           + comments + "\n" + _statements(30, "    "))
    return Planted(absent=("_seeded_long",))


def _c11c(root, pipeline, scratch):
    path, _, _ = store_module(root)
    append(path, "def _outer():\n    def _inner():\n" + _statements(31, "        ")
           + "    y = _inner()\n    return y\n")
    return Planted(present=("_inner — 31 statements > 30",), absent=("_outer",))


def _c11d(root, pipeline, scratch):
    rel = _support(root, "domain.py")
    append(root / rel, "import dataclasses as _seeded_dc\n\n\n"
           "@_seeded_dc.dataclass(frozen=True)\nclass SeededValue:\n"
           "    \"\"\"A value object with one method too long.\"\"\"\n\n    x: int = 0\n\n"
           "    def seeded_method(self):\n" + _statements(31, "        "))
    return Planted(present=(f"C11 {rel}:", "SeededValue.seeded_method — 31 statements > 30"))


def _c11e(root, pipeline, scratch):
    path, _, _ = store_module(root)
    arms = "".join(f"    elif v == {k}:\n        return {k}\n" for k in range(1, 15))
    append(path, "def _seeded_chain(v):\n    if v == 0:\n        return 0\n" + arms
           + "    else:\n        return -1\n")
    return Planted(present=("_seeded_chain — 31 statements > 30",))


def _c12a(root, pipeline, scratch):
    rel, cls = _into_store_class(root, "    def _seeded(self, a, b, c, d, e, f):\n"
                                       "        return a\n")
    return Planted(present=(f"C12 {rel}:", f"{cls}._seeded — 6 parameters > 5"))


def _c12b(root, pipeline, scratch):
    _, cls = _into_store_class(root, "    def _seeded(self, a, b, c, d, e):\n"
                                     "        return a\n")
    return Planted(absent=(f"{cls}._seeded —",))


def _c12c(root, pipeline, scratch):
    path, rel, _ = store_module(root)
    append(path, "def seeded(*args):\n    return args\n")
    return Planted(present=(f"C12 {rel}:", "seeded — *args in a public signature", "_prefixed"))


def _c12d(root, pipeline, scratch):
    path, _, _ = store_module(root)
    append(path, "def _seeded(*args, **kwargs):\n    return args, kwargs\n")
    return Planted(absent=("_seeded —",))


def _c12e_static(root, pipeline, scratch):
    rel, cls = _into_store_class(root, "    @staticmethod\n    def s(a, b, c, d, e, f):\n"
                                       "        return a\n")
    return Planted(present=(f"C12 {rel}:", f"{cls}.s — 6 parameters > 5"))


def _c12e_class(root, pipeline, scratch):
    _, cls = _into_store_class(root, "    @classmethod\n    def c(cls, a, b, c, d, e):\n"
                                     "        return cls\n")
    return Planted(absent=(f"{cls}.c —",))


def _c13a(root, pipeline, scratch):
    rel, cls = _pad_store_class(root, 201)
    return Planted(present=(f"C13 {rel}:", f"class {cls} — 201 lines > 200"))


def _c13b(root, pipeline, scratch):
    rel = _entry(root, "DataSource")["file"]
    _pad_module(root / rel, 401)
    return Planted(present=(f"C13 {rel}:1 module — 401 lines > 400",))


def _c13c(root, pipeline, scratch):
    rel = _support(root, "domain.py")
    _pad_module_past(root / rel, 450)
    return Planted(absent=(f"{rel}:1 module",))


def _c13d(root, pipeline, scratch):
    rel = _support(root, "metrics.py")
    _pad_module_past(root / rel, 450)
    return Planted(absent=(f"{rel}:1 module",))


def _c14a(root, pipeline, scratch):
    module = _entry(root, "Model")["module"]
    doc = json.loads(Path(pipeline).read_text(encoding="utf-8"))
    hits = [m for m in doc.get("modules") or [] if m.get("name") == module]
    if not hits:
        raise SeedError(f"pipeline.json declares no module {module!r}")
    hits[0]["paradigm"] = "FP"
    seeded = Path(scratch) / "pipeline.fp.json"
    seeded.write_text(json.dumps(doc), encoding="utf-8")
    return Planted(args=("--pipeline", str(seeded)),
                   present=(f"C14 pipeline.json modules[{module}]", "OOP"))


def _c14b(root, pipeline, scratch):
    path, rel, _ = store_module(root)
    append(path, "class Extra:\n    \"\"\"A second public class.\"\"\"\n")
    return Planted(present=(f"C14 {rel}:", "Extra — a public class"))


def _c14c(root, pipeline, scratch):
    path, rel, _ = store_module(root)
    append(path, "def helper():\n    return 1\n")
    return Planted(present=(f"C14 {rel}:", "helper — a public function", "_prefixed"))


def _c14d(root, pipeline, scratch):
    path, _, _ = store_module(root)
    append(path, "def _helper():\n    return LIMIT\n\n\nLIMIT = 3\n")
    return Planted(absent=("_helper —", "LIMIT —"))


def _c14e(root, pipeline, scratch):
    """The hand-built bundle's shape: a second public class beside the Predictor. The map is
    set to `implemented` as /vega-pack sets it when it fills the adapter (a no-op on a real
    bundle), and the check runs with --bundle."""
    entry = _entry(root, "Predictor")
    bmap = _map(root)
    for e in bmap["entries"]:
        if e.get("role") == "Predictor":
            e["status"] = "implemented"
    (root / "blueprint-map.json").write_text(json.dumps(bmap, indent=2), encoding="utf-8")
    append(root / entry["file"], "class Seeded:\n    \"\"\"A second public class.\"\"\"\n")
    return Planted(args=("--bundle",), present=(f"C14 {entry['file']}:", "Seeded — a public class"))


SEEDS = (
    Seed("C0a", "a file that does not parse is refused", True, _c0a),
    Seed("C11a", "a 31-statement function is refused", True, _c11a),
    Seed("C11b", "30 statements with a docstring and 5 comment lines pass (neither counts)",
         False, _c11b),
    Seed("C11c", "a nested def is measured on its own, and its parent is not charged for it",
         True, _c11c),
    Seed("C11d", "a 31-statement method of a value object in domain.py is refused (the module "
                 "exemption does not cover C11)", True, _c11d),
    Seed("C11e", "an if/elif chain of 16 arms is 31 statements: the AST counts elif", True,
         _c11e),
    Seed("C12a", "a method of 6 parameters beyond self is refused", True, _c12a),
    Seed("C12b", "a method of 5 parameters beyond self passes", False, _c12b),
    Seed("C12c", "a public *args signature is refused", True, _c12c),
    Seed("C12d", "*args and **kwargs in a _prefixed helper pass", False, _c12d),
    Seed("C12e", "a staticmethod of 6 parameters is refused (it has no self to discount)", True,
         _c12e_static),
    Seed("C12e", "a classmethod of cls and 5 parameters passes", False, _c12e_class),
    Seed("C13a", "a class of 201 lines is refused", True, _c13a),
    Seed("C13b", "a module of 401 lines is refused", True, _c13b),
    Seed("C13c", "domain.py past 400 lines passes (module exemption)", False, _c13c),
    Seed("C13d", "metrics.py past 400 lines passes (module exemption)", False, _c13d),
    Seed("C14a", "a port-owning module declared FP is refused", True, _c14a),
    Seed("C14b", "a second public class in an adapter module is refused", True, _c14b),
    Seed("C14c", "a public function in an adapter module is refused", True, _c14c),
    Seed("C14d", "a _prefixed helper and an UPPER_CASE constant in an adapter module pass",
         False, _c14d),
    Seed("C14e", "a second public class beside the bundle's Predictor is refused (--bundle)",
         True, _c14e, wants_bundle=True),
)


# --------------------------------------------------------------------------- running one

def plant(seed: Seed, package: Path, pipeline: Path, scratch: Path) -> tuple:
    """(the seeded copy of `package` under `scratch`, what the gate must say about it)."""
    copy = Path(scratch) / Path(package).name
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy, seed.plant(copy, Path(pipeline), Path(scratch))


def command(copy: Path, pipeline: Path, planted: Planted) -> list:
    return [sys.executable, str(CONFORMANCE), "--package", str(copy), "--pipeline",
            str(pipeline), "--size-severity", "error", *planted.args]


def expected_exit(seed: Seed, boundary_exit: int = 0) -> int:
    """2 for a refused seed; for a boundary, the exit the unseeded package gets (0 on a clean
    package; 2 on today's templates, whose own findings refuse at --size-severity error)."""
    return 2 if seed.refused else boundary_exit


def problems(seed: Seed, planted: Planted, rc: int, out: str, boundary_exit: int = 0) -> list:
    """What is wrong with one run of the gate on a seeded copy: [] when it did its job."""
    found = []
    expected = expected_exit(seed, boundary_exit)
    if rc != expected:
        found.append(f"exit {rc}, expected {expected}")
    low = out.lower()
    found += [f"message does not mention {n!r}" for n in planted.present if n.lower() not in low]
    for needle in planted.absent:
        hit = next((ln for ln in out.splitlines() if needle.lower() in ln.lower()), None)
        if hit is not None:
            found.append(f"{needle!r} was named: {hit.strip()[:160]}")
    return found
