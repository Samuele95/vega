"""
test_builder_carried.py — the emitted builder dispatches to the block types its package carries,
and refuses any other by name (the T5 round 3 audit, finding 1).

The builder's tables (`_DISPATCH`, the misplacement rules, the composites) are cut
by `# >>> segment:` sections, and the emitter keeps the sections of the blocks.py segments it
spliced: an emitted builder names no class its package does not hold, and a spec block of a
type the package does not carry fails by name — before, a 31-entry table of lambdas deferred
a `NameError` to the first build of such a block.
"""
from __future__ import annotations

import ast
import builtins
import copy
import importlib
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import render_templates as rt  # noqa: E402

TEMPLATES = HERE.parent / "backends" / "pytorch" / "templates"
_KEEP = []


def _package(tag: str, spec: dict):
    work = tempfile.TemporaryDirectory(prefix=f"carried-{tag}-")
    _KEEP.append(work)
    package = f"carried_{tag}"
    pipeline = rt.reference_pipeline()
    pipeline["system"]["package"] = package
    rec = rt.render(TEMPLATES, Path(work.name), pipeline, spec, full=True)
    sys.path.insert(0, work.name)
    importlib.invalidate_caches()
    return Path(rec["root"]), importlib.import_module(f"{package}.model.network"), \
        importlib.import_module(f"{package}.domain")


def _free_names(path: Path) -> set:
    """The names builder.py loads and neither defines, imports nor gets from builtins."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            bound.update(a.arg for a in node.args.args + node.args.kwonlyargs) \
                if not isinstance(node, ast.ClassDef) else None
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loaded - bound


def test_a_row_builder_carries_its_blocks_and_refuses_another_by_name():
    spec = rt.reference_spec("row")
    root, network, domain = _package("row", spec)
    builder = root / "model" / "builder.py"
    assert not _free_names(builder), sorted(_free_names(builder))
    text = builder.read_text(encoding="utf-8")
    assert "BatchNorm" not in text and "RecurrentBlock" not in text and "GCNConv" not in text
    assert "# >>> segment" not in text and "# <<< segment" not in text
    other = copy.deepcopy(spec)
    blocks = other["blocks"]
    blocks.insert(len(blocks) - 1, {"id": "Bn", "type": "batchnorm", "params": {}})
    with pytest.raises(ValueError, match="carries no 'batchnorm' block") as caught:
        network.Network.build(domain.NetSpec.from_dict(other))
    assert "dense" in str(caught.value) and "re-emitted" in str(caught.value)


def test_the_source_builder_still_dispatches_every_type():
    """Unselected, builder.py is the program for every block type (the runnable source the
    unit tests import), every class bound."""
    sys.path.insert(0, str(HERE.parent / "backends" / "pytorch"))
    import builder  # noqa: PLC0415
    every = {**builder.NetworkBuilder._DISPATCH, **builder._CompositeBlocks._COMPOSITE}
    assert len(every) == 33 and "batchnorm" in every and "custom" in every
    assert all(cls is not None for t, (cls, _m) in every.items() if t != "custom")
