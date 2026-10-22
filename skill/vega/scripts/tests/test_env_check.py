#!/usr/bin/env python3
"""test_env_check.py — /vega's environment line registers the presentation stage (work package H10).

    python3 -m pytest -q test_env_check.py

`env_check.py --env` must:

1. list `vega-show` among the skills it checks and report the `showcase (show)` lane, open when
   `vega-show`, `vega-pack` and `vega-forge` are installed and built (and torch imports);
2. report `vega-show` UNBUILT when one of its scripts or page-template files is absent (R-E2:
   `assets/page/{index.html, TEMPLATE.json, strings.keys.json}`, never D5's `chrome.en.json` or
   `page-manifest.json`), when the vega-forge serving template lacks `serve/routes_showcase.py`
   (or it does not serve `SHOWCASE.json`), and when the vega-pack catalog lacks the record pack
   writes as `MANIFEST.serve.showcase_slot` — and close the lane then;
3. report `document-project` DEGRADED, never UNBUILT, when its SKILL.md lacks the label-role map
   (`docs/label-roles.json`): on the line, as a `WARN` below it, under `skills_degraded` in
   `--json`, with the exit code unchanged; the package's own `patches/document-project/SKILL.md`
   carries the map, so the patched skill is not degraded.

Each case builds a skills root in a temporary directory holding exactly what the probes read;
the libraries are stubbed so the probe logic, not this machine, decides the lanes.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENV_CHECK = HERE.parent / "env_check.py"


def _load():
    spec = importlib.util.spec_from_file_location("vega_env_check", ENV_CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ec = _load()
LIBS_PRESENT = {m: "1.0" for m in ec.LIBS}
SLOT_RECORD = {"serve_record": {"showcase_slot": {"dir": "showcase/", "manifest": "SHOWCASE.json",
                                                  "routes": ["GET /v1/showcase"]}}}
PROMPTS = {"architect": "--baseline", "blueprint": "writes docs/blueprint-brief.md",
           "document-project": "/document-project vega <lane> ... docs/label-roles.json"}


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_root(tmp: Path) -> Path:
    """A skills root where every skill env_check knows is installed and built."""
    root = tmp / "skills"
    for skill in ec.SKILLS:
        _touch(root / skill / "SKILL.md", f"# {skill}\n{PROMPTS.get(skill, '')}\n")
        for rel in ec.REQUIRED_SCRIPTS.get(skill, []):
            if "." in Path(rel).name:
                _touch(root / skill / rel)
            else:                                   # a directory (vega-forge's templates)
                (root / skill / rel).mkdir(parents=True, exist_ok=True)
    _touch(root / "dataset-shaper" / "scripts" / "shape.py", "ap.add_argument('--transform-only')\n")
    _touch(root / ec.SHOWCASE_ROUTES, 'MANIFEST = "SHOWCASE.json"\n')
    _touch(root / ec.PACK_CATALOG, json.dumps(SLOT_RECORD))
    return root


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "libraries", lambda: dict(LIBS_PRESENT))
    monkeypatch.setattr(ec, "gpu_name", lambda libs: None)
    return make_root(tmp_path)


def _run_main(root: Path, *extra: str) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = ec.main(["--env", "--skills-root", str(root), *extra])
    return code, out.getvalue()


def test_vega_show_is_a_checked_skill_with_the_page_template_of_r_e2():
    assert "vega-show" in ec.SKILLS
    assert ec.SKILLS.index("vega-pack") < ec.SKILLS.index("vega-show") < ec.SKILLS.index("document-project")
    required = ec.REQUIRED_SCRIPTS["vega-show"]
    for rel in ("assets/page/index.html", "assets/page/TEMPLATE.json", "assets/page/strings.keys.json",
                "scripts/build_showcase.py", "scripts/apply_show_layer.py", "scripts/smoke_show.py"):
        assert rel in required
    assert not [r for r in required if r.endswith(("chrome.en.json", "page-manifest.json"))]


def test_a_complete_root_opens_the_showcase_lane(root):
    r = ec.env_report(root)
    assert r["skills_installed"]["vega-show"] is True
    assert "vega-show" not in r["skills_specified_but_unbuilt"]
    assert r["skills_degraded"] == {}
    assert r["lanes_open"]["showcase (show)"] is True
    code, out = _run_main(root)
    assert code == 0
    assert "OK   lane showcase (show): open" in out
    assert "UNBUILT" not in out and "DEGRADED" not in out


@pytest.mark.parametrize("why, damage", [
    ("routes_showcase.py", lambda root: (root / ec.SHOWCASE_ROUTES).unlink()),
    ("routes_showcase.py", lambda root: _touch(root / ec.SHOWCASE_ROUTES, "MANIFEST = 'other'\n")),
    ("showcase_slot", lambda root: _touch(root / ec.PACK_CATALOG, json.dumps({"serve_record": {}}))),
    ("showcase_slot", lambda root: _touch(root / ec.PACK_CATALOG, "{not json")),
    ("assets/page/TEMPLATE.json", lambda root: (root / "vega-show/assets/page/TEMPLATE.json").unlink()),
    ("scripts/smoke_show.py", lambda root: (root / "vega-show/scripts/smoke_show.py").unlink()),
])
def test_vega_show_is_unbuilt_without_its_slot_or_its_files(root, why, damage):
    damage(root)
    r = ec.env_report(root)
    missing = r["skills_specified_but_unbuilt"].get("vega-show") or []
    assert any(why in m for m in missing), missing
    assert r["lanes_open"]["showcase (show)"] is False
    assert r["lanes_open"]["network (net → train → eval → pack)"] is True   # only show closes
    code, out = _run_main(root)
    assert code == 0
    assert "vega-show UNBUILT (" in out and why in out
    assert "WARN lane showcase (show): closed" in out
    assert "ERROR vega-show is installed but unbuilt" in out


def test_the_showcase_lane_closes_with_an_unbuilt_upstream_or_without_torch(root, monkeypatch):
    (root / "vega-pack" / "scripts" / "pack.py").unlink()
    assert ec.env_report(root)["lanes_open"]["showcase (show)"] is False
    _touch(root / "vega-pack" / "scripts" / "pack.py")
    monkeypatch.setattr(ec, "libraries", lambda: dict(LIBS_PRESENT, torch=None))
    assert ec.env_report(root)["lanes_open"]["showcase (show)"] is False


def test_document_project_without_the_label_role_map_is_degraded_not_unbuilt(root):
    _touch(root / "document-project" / "SKILL.md", "# document-project\n/document-project vega <lane>\n")
    r = ec.env_report(root)
    assert "document-project" not in r["skills_specified_but_unbuilt"]
    assert "label-roles.json" in " ".join(r["skills_degraded"]["document-project"])
    assert r["lanes_open"]["showcase (show)"] is True        # degraded closes no lane
    assert "document-project DEGRADED (" in ec.env_line(r)
    code, out = _run_main(root)
    assert code == 0
    assert "WARN document-project is degraded: no label-role map" in out
    code, out = _run_main(root, "--json")
    assert code == 0
    assert json.loads(out)["environment"]["skills_degraded"]["document-project"]


def test_the_document_lane_closes_on_an_unpatched_document_project(root, monkeypatch):
    monkeypatch.setattr(ec.shutil, "which", lambda name: "/usr/bin/" + name)
    assert ec.env_report(root)["lanes_open"]["document (document-project)"] is True
    _touch(root / "document-project" / "SKILL.md", "# document-project\n")
    r = ec.env_report(root)
    assert "document-project" in r["skills_specified_but_unbuilt"]
    assert r["lanes_open"]["document (document-project)"] is False


def test_the_package_patch_carries_the_label_role_map():
    patch = ROOT / "patches" / "document-project"
    assert ec.degraded_reasons("document-project", patch) == []


def test_the_shipped_slot_satisfies_the_probe():
    assert ec.showcase_slot_missing(ROOT) == []


def test_the_real_tree_lists_vega_show_and_the_showcase_lane():
    code, out = _run_main(ROOT, "--json")
    env = json.loads(out)["environment"]
    assert env["skills_installed"]["vega-show"] is True
    assert "showcase (show)" in env["lanes_open"]
    assert "skills_degraded" in env
    slot = [m for m in env["skills_specified_but_unbuilt"].get("vega-show", []) if "routes_showcase" in m or "showcase_slot" in m]
    assert slot == []
    assert code in (0, 1)


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", __file__]))
