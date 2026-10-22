"""
scenarios.py — what `test_conformance_sizes.py` builds from the clean package beyond D6 §12.1:
the rule seeds C14f–C14i (read the way `seeds.SEEDS` are), and the copies, bundles and module
texts its scope tests run the gate on.

A library of the test, not a measurement fixture.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from seeds import Planted, Seed, append, store_module

# --------------------------------------------------------------------------- rule seeds


def _c14f(root, pipeline, scratch):
    path, rel, _ = store_module(root)
    append(path, 'default_name = "registry"\n')
    return Planted(present=(f"C14 {rel}:", "default_name — a public module-level name",
                            "UPPER_CASE"))


def _c14g(root, pipeline, scratch):
    path, _, _ = store_module(root)
    append(path, "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n"
                 "    from pathlib import PurePath\n\ntry:\n    import json as _json\n"
                 "except ImportError:\n    _json = None\n")
    return Planted(absent=(" if —", " try —", "_json", "PurePath"))


def _c14h(root, pipeline, scratch):
    path, rel, _ = store_module(root)
    append(path, "try:\n    from math import fsum as _fsum\nexcept ImportError:\n"
                 "    def fsum(values):\n        return sum(values)\n")
    return Planted(present=(f"C14 {rel}:", "fsum — a public function", "_prefixed"))


def _c14i(root, pipeline, scratch):
    path, _, _ = store_module(root)
    append(path, "class _Cache:\n    \"\"\"A _prefixed helper class.\"\"\"\n\n"
                 "    def get(self, key):\n        return key\n")
    return Planted(absent=("_Cache",))


RULE_SEEDS = (
    Seed("C14f", "a lower-case public module-level name in an adapter module is refused", True,
         _c14f),
    Seed("C14g", "an `if TYPE_CHECKING:` guard and a try/except ImportError guard of imports and "
                 "_prefixed names pass", False, _c14g),
    Seed("C14h", "a public def hidden in an import guard is refused (a guard is read inside)",
         True, _c14h),
    Seed("C14i", "a _prefixed helper class beside the adapter class passes", False, _c14i),
)

# --------------------------------------------------------------------------- module texts

ENTRYPOINT_PREDICT = '''"""The bundle's CLI: an entrypoint, held to C11-C13 and not to C14."""
from __future__ import annotations


class Loaded:
    """A second public class is fine here: an entrypoint is not an adapter module."""


class Answer:
    """And a third."""


def load(root):
    return Loaded()


def main(argv=None):
    return 0
'''

# The shaper's transform-only module as /vega-pack copies it (R-A3): the port is its public
# functions. `last_unseen` is on line 19, `transform` on line 7.
SHAPER_MODULE = '''"""The shaper's transform-only module, as /vega-pack copies it: functions."""
from __future__ import annotations

FITTED = {"amount": {"mean": 1.0, "std": 2.0}}


def transform(df_raw, history=None, with_unseen=False):
    return df_raw


def edges_for(df_raw, reference):
    return None


def edges_within(df_raw):
    return None


def last_unseen():
    return ()
'''

# The same shape where a class belongs: a Predictor written as module functions (`predict`
# on line 7), which C14 refuses.
FUNCTION_PREDICTOR = (SHAPER_MODULE.replace("transform-only", "predict")
                      .replace("def transform(", "def predict("))

# --------------------------------------------------------------------------- package copies


def copy_package(package: Path, td) -> Path:
    copy = Path(td) / package.name
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy


def make_bundle(package: Path, td) -> Path:
    """A bundle root around a copy of the package: predict.py, serve/, tests/."""
    root = Path(td) / "bundle"
    copy_package(package, root)
    (root / "serve").mkdir()
    (root / "tests").mkdir()
    (root / "predict.py").write_text(ENTRYPOINT_PREDICT, encoding="utf-8")
    (root / "serve" / "app.py").write_text("def create_app(bundle_dir=None):\n    return None\n",
                                           encoding="utf-8")
    (root / "tests" / "test_roundtrip.py").write_text(
        "def test_roundtrip():\n    assert True\n", encoding="utf-8")
    return root


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{path.name} no longer holds {old!r}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _rewrite_map(copy: Path, change) -> None:
    path = copy / "blueprint-map.json"
    bmap = json.loads(path.read_text(encoding="utf-8"))
    change(bmap)
    path.write_text(json.dumps(bmap), encoding="utf-8")


def implement(copy: Path, role: str) -> str:
    """Mark the map's `role` entry implemented, as the stage that fills it does; its file."""
    found = []

    def change(bmap):
        entry = next(e for e in bmap["entries"] if e.get("role") == role)
        entry["status"] = "implemented"
        found.append(entry["file"])

    _rewrite_map(copy, change)
    return found[0]


def add_entrypoint(copy: Path, rel: str) -> None:
    _rewrite_map(copy, lambda bmap: bmap["entrypoints"].append(
        {"file": rel, "calls": "Predictor", "stage": "/vega-pack"}))


def out_of_slice(copy: Path, module: str) -> None:
    """Mark every entry of `module` out of scope and drop its files, as /vega-pack does."""
    def change(bmap):
        for entry in bmap["entries"]:
            if entry.get("module") == module:
                entry["status"] = f"out of scope: {module} is not part of serving"
                (copy / entry["file"]).unlink()

    _rewrite_map(copy, change)


def with_paradigm(pipeline: Path, td, module: str, paradigm: str) -> Path:
    """A copy of `pipeline` whose `module` is declared `paradigm`."""
    doc = json.loads(Path(pipeline).read_text(encoding="utf-8"))
    next(m for m in doc["modules"] if m["name"] == module)["paradigm"] = paradigm
    path = Path(td) / f"pipeline.{module}.{paradigm}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def c14_lines(out: str, rel: str) -> list:
    """The gate's C14 lines that name `rel`."""
    return [ln for ln in out.splitlines() if ln.split(":", 1)[-1].strip().startswith("C14")
            and rel in ln]


def json_document(out: str) -> dict:
    """The gate's whole output as one JSON document; anything else fails the test."""
    try:
        return json.loads(out)
    except ValueError as exc:
        raise AssertionError(f"--json printed more than one JSON document ({exc}):\n{out}")
