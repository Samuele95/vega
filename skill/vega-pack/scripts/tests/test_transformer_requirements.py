#!/usr/bin/env python3
"""
test_transformer_requirements.py — a bundle declares every distribution its FeatureTransformer
imports (F41).

    python3 -m pytest test_transformer_requirements.py [-q]

The release4 rehearsal's recipe carried a Box-Cox step; the shaper's transform-only module
imports scipy inside that step's function, and the bundle's requirements were a fixed list
(pandas, numpy, pyarrow, torch; the service's web rows): requirements-serve.txt, the pyproject
and the Docker image installed no scipy, the start-up digest check passed and the first
transform died with ImportError. Every product check ran in the packing interpreter, which has
scipy, so none saw it. What is proved, on that rehearsal's real `preprocess.py`
(`fixtures/preprocess-boxcox-release4.py`) and on small variants:

  * `transformer_requirements` finds a function-level import, maps it to its distribution
    (`sklearn` -> `scikit-learn`) and pins it to the packing environment's version; the
    standard library, `__future__`, the package's own modules and the rows every bundle
    declares (pandas, numpy, pyarrow, torch) add nothing;
  * `requirements.txt`, `requirements-serve.txt` and the pyproject's dependencies all carry it,
    once, and a transformer that imports nothing new changes none of them.
"""
from __future__ import annotations

import importlib.metadata as md
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent
sys.path.insert(0, str(SCRIPTS))

import bundle_inputs  # noqa: E402
import serving_pack as sp  # noqa: E402

FIXTURE = TESTS / "fixtures" / "preprocess-boxcox-release4.py"


def _pkg(tmp_path, text=None):
    pkg = tmp_path / "bundle" / "demo_pipeline"
    (pkg / "features").mkdir(parents=True)
    if text is None:
        shutil.copy(FIXTURE, pkg / "features" / "preprocess.py")
    else:
        (pkg / "features" / "preprocess.py").write_text(text, encoding="utf-8")
    return pkg


def _ctx(pkg):
    return SimpleNamespace(paths=SimpleNamespace(root=pkg.parent, pkg=pkg), markers=[])


def _have(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def test_the_rehearsal_transformer_needs_scipy(tmp_path):
    assert "from scipy import stats" in FIXTURE.read_text(encoding="utf-8")
    assert bundle_inputs.transformer_requirements(_pkg(tmp_path)) == [("scipy", _have("scipy"))]


def test_a_function_level_import_maps_to_its_distribution(tmp_path):
    pkg = _pkg(tmp_path, "from __future__ import annotations\nimport json, math\n"
                         "import numpy as np\nimport pandas as pd\nfrom . import helpers\n"
                         "from demo_pipeline.domain import X\n"
                         "def _step(parts):\n    from sklearn.decomposition import PCA\n"
                         "    import scipy.stats\n")
    got = dict(bundle_inputs.transformer_requirements(pkg))
    assert set(got) == {"scikit-learn", "scipy"}
    assert got["scipy"] == _have("scipy") and got["scikit-learn"] == _have("scikit-learn")


def test_a_transformer_of_core_imports_adds_nothing(tmp_path):
    pkg = _pkg(tmp_path, "import json\nimport numpy as np\nimport pandas as pd\n")
    assert bundle_inputs.transformer_requirements(pkg) == []
    ctx = _ctx(pkg)
    sp._write_serve_requirements(ctx)
    text = (pkg.parent / "requirements-serve.txt").read_text(encoding="utf-8")
    assert "scipy" not in text


@pytest.mark.skipif(_have("scipy") is None, reason="scipy is not installed here")
def test_every_requirements_file_carries_it_once(tmp_path):
    pkg = _pkg(tmp_path)
    ctx = _ctx(pkg)
    sp._write_requirements(ctx)
    sp._write_serve_requirements(ctx)
    pins = f"scipy=={_have('scipy')}"
    for name in ("requirements.txt", "requirements-serve.txt"):
        lines = (pkg.parent / name).read_text(encoding="utf-8").splitlines()
        assert lines.count(pins) == 1, (name, lines)
    core, _web = sp._core_and_web(pkg)
    assert ("scipy", _have("scipy")) in core            # the pyproject's `deps` are formatted from it
