"""
tests/test_api_closed_world.py — the service runs and reads only the files it was packed with.

    python3 -m pytest tests/test_api_closed_world.py

Every file the service answers from — the data (weights, spec, fitted parameters, schema, the
reference graph, the monitor reference), the examples its self-test replays, its own code — is
the pinned one, or the service does not start: /health is 503 and names the file, and every
other route answers 503 too. The module API refuses altered data the same way. An installed
wheel serves the bundle it was built from and refuses an altered copy; a copy with altered code,
or with a route module the manifest does not pin, stops itself before running it; a same-named
package earlier on PYTHONPATH never runs; the examples served are the verified ones for the life
of the process. The service checks code where it runs it from, and one process serves one
bundle (the Recipe reads its fitted parameters at import), so a copy is served by its own code,
in its own interpreter.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient                  # noqa: E402

from _oracle import ORACLE, ROOT                           # noqa: E402
from serve.app import create_app                           # noqa: E402

MANIFEST = ORACLE.manifest
ROLES = MANIFEST.get("roles") or {}
DATA = sorted(ROLES.get("data") or ())
SERVED = sorted(set(DATA) | set(ROLES.get("example") or ()))
IGNORED = shutil.ignore_patterns("records", "__pycache__", "*.ipynb", "showcase", "showcase.*")
TIMEOUT = 600
# a probe's head: the copy's own code first on the path, then its service
SERVE_COPY = ("import sys; sys.path.insert(0, sys.argv[1]); from serve.app import create_app; "
              "from fastapi.testclient import TestClient\n")
EXTRA_ROUTES = ('ENDPOINTS = (("GET", "/v1/extra"),)\nMODEL_ROUTES = ()\n'
                'OPEN_WHILE_UNREADY = ()\nCAPABILITY = "core"\n\n\n'
                'def register(app, state):\n    app.get("/v1/extra")(lambda: {"ran": True})\n')


def _copy(where: Path) -> Path:
    return Path(shutil.copytree(ROOT, where / "bundle", ignore=IGNORED))


def _append(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)


def _probe(code: str, *args, cwd: Path, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True,
                          text=True, cwd=str(cwd), env=env, timeout=TIMEOUT)


def test_the_manifest_gives_the_served_files_their_roles():
    input_schema = MANIFEST.get("input_schema", "schema.json")
    assert input_schema in DATA and any(r.startswith("weights.") for r in DATA), DATA
    assert {"examples/rows.csv", "examples/expected.parquet"} <= set(ROLES.get("example") or ())
    assert "predict.py" in (ROLES.get("code") or ()), ROLES.get("code")


@pytest.mark.parametrize("altered", SERVED)
def test_a_file_that_is_not_the_packed_one_stops_the_service(tmp_path, altered):
    copy = _copy(tmp_path)
    _append(copy / altered, "\n")
    with TestClient(create_app(copy)) as c:
        health = c.get("/health")
        assert health.status_code == 503 and altered in health.json()["error"], health.text
        assert c.get("/v1/examples").status_code == 503
        assert c.post("/v1/predict", json={"rows": []}).status_code == 503


@pytest.mark.parametrize("altered", DATA)
def test_the_module_api_refuses_altered_data(tmp_path, altered):
    from predict import load, load_monitor
    copy = _copy(tmp_path)
    _append(copy / altered, " ")
    for entry in (load, load_monitor):
        with pytest.raises(ValueError, match=re.escape(altered)):
            entry(copy)


def test_an_installed_wheel_serves_the_bundle_and_refuses_an_altered_one(tmp_path):
    try:
        import setuptools  # noqa: F401  (pip builds the wheel with it, without isolation)
    except ImportError:
        pytest.skip("no wheel could be built here: setuptools is not installed")
    src = _copy(tmp_path / "src")
    built = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                            "--no-build-isolation", "-q", "-w", str(tmp_path / "whl"), str(src)],
                           capture_output=True, text=True, timeout=TIMEOUT)
    if built.returncode != 0:
        pytest.skip("no wheel could be built here: " + built.stderr[-200:])
    wheel = next((tmp_path / "whl").glob("*.whl"))
    site = tmp_path / "site"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-q", "--target",
                    str(site), str(wheel)], check=True, timeout=TIMEOUT)
    altered = _copy(tmp_path / "alt")
    _append(altered / "fitted.json", " ")
    probe = ("import sys; from serve.app import create_app; "
             "from fastapi.testclient import TestClient\n"
             "for b in sys.argv[1:]:\n"
             "    with TestClient(create_app(b)) as c:\n"
             "        print(c.get('/health').status_code)\n"
             "import predict; print(predict.__file__)")
    out = _probe(probe, ROOT, altered, cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(site)))
    lines = out.stdout.split()
    assert lines[:2] == ["200", "503"], out.stdout + out.stderr
    assert lines[2].startswith(str(site)), "the installed service ran another copy's code"


def test_a_copy_refuses_unpinned_code_and_keeps_its_verified_examples(tmp_path):
    # one interpreter, one copy, three starts (each probe costs a torch import): an unpinned
    # route module and an altered package each stop the service before any of the copy's code
    # runs (the altered module would leave a mark); restored, the copy serves, and the examples
    # it verified at start are the ones it answers for the life of the process
    copy, marker = _copy(tmp_path), tmp_path / "ran"
    init = copy / MANIFEST["package"] / "__init__.py"
    probe = SERVE_COPY + (
        "from pathlib import Path\n"
        "copy, init, extra = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])\n"
        "extra.write_text(sys.argv[4])\n"
        "with TestClient(create_app(copy)) as c:\n"
        "    h, x = c.get('/health'), c.get('/v1/extra')\n"
        "    print('route', h.status_code, x.status_code, h.json()['detail'])\n"
        "extra.unlink()\n"
        "kept = init.read_bytes()\n"
        "init.write_bytes(kept + sys.argv[5].encode())\n"
        "with TestClient(create_app(copy)) as c:\n"
        "    h = c.get('/health'); print('code', h.status_code, h.json()['error'])\n"
        "init.write_bytes(kept)\n"
        "with TestClient(create_app(copy)) as c:\n"
        "    before = c.get('/v1/examples')\n"
        "    rows = copy / 'examples' / 'rows.csv'\n"
        "    rows.write_text(rows.read_text().splitlines()[0] + '\\n')\n"
        "    after = c.get('/v1/examples')\n"
        "    print('examples', before.status_code, after.status_code,"
        " before.content == after.content)\n")
    altered = f"\nopen({str(marker)!r}, 'w').write('x')\n"
    out = _probe(probe, copy, init, copy / "serve" / "routes_extra.py", EXTRA_ROUTES, altered,
                 cwd=tmp_path)
    said = {line.split()[0]: line.split(maxsplit=1)[1] for line in out.stdout.splitlines()
            if line.split()[:1] and line.split()[0] in ("route", "code", "examples")}
    assert set(said) == {"route", "code", "examples"}, out.stdout + out.stderr
    assert said["route"].startswith("503 503") and "serve/routes_extra.py" in said["route"], said
    assert said["code"].startswith("503") and "__init__.py" in said["code"], said
    assert not marker.exists(), "the altered package's code ran"
    assert said["examples"].split() == ["200", "200", "True"], out.stdout + out.stderr


def test_a_shadowing_package_is_refused_before_it_runs(tmp_path):
    # a same-named package earlier on PYTHONPATH must neither run nor serve
    shadow = tmp_path / "shadow" / MANIFEST["package"]
    shadow.mkdir(parents=True)
    marker = tmp_path / "ran"
    (shadow / "__init__.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n",
                                        encoding="utf-8")
    probe = ("import sys; from serve.app import create_app; "
             "from fastapi.testclient import TestClient\n"
             "with TestClient(create_app(sys.argv[1])) as c:\n"
             "    print(c.get('/health').status_code)")
    env = dict(os.environ, PYTHONPATH=f"{tmp_path / 'shadow'}{os.pathsep}{ROOT}")
    out = _probe(probe, ROOT, cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[:1] == ["200"], out.stdout + out.stderr   # its own code serves
    assert not marker.exists(), "the shadowing package's code ran"
