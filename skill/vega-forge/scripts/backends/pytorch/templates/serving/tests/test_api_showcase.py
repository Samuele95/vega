"""
tests/test_api_showcase.py — the showcase slot (serve/routes_showcase.py), in process.

    python3 -m pytest tests/test_api_showcase.py

Showcases are built here, beside the bundle (`<ENV>_SHOWCASE`) or in a copy (`showcase/`).
Absent, the slot's routes are 404 and `/` is the console; verified, `/` is its entry and its
files are served from memory (nosniff, no-cache, ETag = digest with 304, CSP on HTML), also
while the model is unready; refused, nothing of it is served, with the reason (a changed file,
another bundle, a re-emit, a path out, a symlink, an unpinned servable file outside the scratch
directories, the byte cap, a served file not pinned, a showcase path the bundle pins), and the
model stays ready. No showcase file shadows a service route, /v1/showcase is /health's
`showcase`, and the bundle's MANIFEST.json is never written.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient      # noqa: E402

from serve import routes_showcase as rs         # noqa: E402
from serve.app import create_app               # noqa: E402

MANIFEST_BYTES = (ROOT / "MANIFEST.json").read_bytes()
MANIFEST = json.loads(MANIFEST_BYTES)
SLOT = MANIFEST["serve"]["showcase_slot"]
ENV, VERSION = SLOT["env"], MANIFEST["model_version"]
STATUS_FIELDS = ("present", "verified", "reason", "digest", "entry", "template",
                 "report_status", "created_at", "files", "unlisted")
RECORDS = ("plan/facts.json", "plan/story-plan.json")      # pinned, never served


def _doc(value) -> bytes:
    return (json.dumps(value, indent=1) + "\n").encode("utf-8")


def _digest(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


PAGE = (b'<!doctype html>\n<html><head><link rel="stylesheet" href="showcase/page/page.css">'
        b'<script type="module" src="showcase/page/js/main.js"></script></head>'
        b'<body data-fixture="showcase entry"></body></html>\n')
FILES = {
    "page/index.html": PAGE,
    "page/page.css": b"body { margin: 0 }\n", "page/js/main.js": b"export const ready = true;\n",
    "page.json": _doc({"schema": "vega/page@1"}),
    "story.json": _doc({"schema": "vega/story@2", "sections": {}}),
    "data/attribution.json": _doc({"schema": "vega/attribution@2"}),
    "report/report.pdf": b"%PDF-1.4\n% a fixture report\n%%EOF\n",
    "health.json": _doc({"shadow": "health"}),          # named like service routes (SL11)
    "v1/predict.json": _doc({"shadow": "predict"}),
    "plan/facts.json": _doc({"schema": "vega/show-facts@1"}),
    "plan/story-plan.json": _doc({"schema": "vega/story-plan@2"}),
}
BRIEFS = dict({k: v for k, v in FILES.items() if k != "report/report.pdf"},
              **{"briefs/net-brief.md": b"# The network\n\nA fixture brief.\n"})


def _report(pins: dict) -> dict:
    if "report/report.pdf" not in pins:
        return {"status": "brief-links", "pdf": None, "aux": None, "roles": None, "briefs": {
            "net": {"source": "docs/net-brief.md", "file": "briefs/net-brief.md",
                    "digest": pins["briefs/net-brief.md"]}}}
    return {"status": "linked", "briefs": None, "aux": {"source": "a.aux", "digest": _digest(b"")},
            "pdf": {"source": "docs/main.pdf", "file": "report/report.pdf",
                    "digest": pins["report/report.pdf"]},
            "roles": {"source": "docs/label-roles.json", "origin": "proposed",
                      "digest": _digest(b"roles"), "bound": [], "missing": []}}


def showcase_doc(files, bundle=MANIFEST_BYTES, version=VERSION) -> dict:
    """SHOWCASE.json (vega/showcase@1) pinning `files`, made for the bundle of these bytes."""
    pins = {rel: _digest(body) for rel, body in sorted(files.items())}
    served = [rel for rel in pins if rel not in RECORDS]
    return {"schema": "vega/showcase@1", "created_at": "2026-09-30T12:00:00Z",
            "bundle": {"manifest": "MANIFEST.json", "manifest_digest": _digest(bundle),
                       "slug": str(MANIFEST.get("slug") or "bundle"),
                       "run": str(MANIFEST.get("run") or "R0"), "model_version": version},
            "model": {"path": "fixture.pack.html", "turtle_digest": _digest(b"turtle")},
            "entry": "page/index.html", "page": {"file": "page.json"}, "produced_by": "/vega-show",
            "template": {"name": "vega-show/page", "version": "2.3.0",
                         "manifest_digest": _digest(b"template")},
            "story": {"file": "story.json", "schema": "vega/story@2", "values_n": 0,
                      "placeholders_n": 0,
                      "sections": ["context", "network", "experiments", "results"]},
            "attribution": {"file": "data/attribution.json"},
            "plan": {"file": "plan/story-plan.json", "facts": "plan/facts.json"},
            "report": _report(pins), "files": pins, "served": served,
            "roles": {"page": [r for r in served if r.startswith("page")], "story": ["story.json"],
                      "report": [r for r in served if r[:7] in ("report/", "briefs/")],
                      "data": ["data/attribution.json"], "record": list(RECORDS)},
            "requires": {"endpoints": [], "examples": "vega/examples@2", "network_view":
                         "vega/network-view@2", "explanation": "vega/explanation@2"},
            "supersedes": None, "markers": ["OK: a fixture showcase"]}


def build(where: Path, files=None, bundle: bytes = MANIFEST_BYTES) -> Path:
    """A showcase at `where` for the bundle of these bytes, SHOWCASE.json pinning its files."""
    files = FILES if files is None else files
    for rel, body in files.items():
        (where / rel).parent.mkdir(parents=True, exist_ok=True)
        (where / rel).write_bytes(body)
    (where / "SHOWCASE.json").write_bytes(_doc(showcase_doc(files, bundle)))
    return where


def edit(showcase: Path, change) -> None:          # SHOWCASE.json changed by change(doc)
    doc = json.loads((showcase / "SHOWCASE.json").read_text(encoding="utf-8"))
    change(doc)
    (showcase / "SHOWCASE.json").write_bytes(_doc(doc))


def app_for(showcase, bundle: Path = ROOT):
    """`bundle`'s service, its slot at `showcase` (None: `<bundle>/showcase`), read once."""
    named = {ENV: str(showcase)} if showcase is not None else {}
    with mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != ENV} | named,
                         clear=True):
        return create_app(bundle)


def opened(showcase: Path, bundle: bytes = MANIFEST_BYTES):
    return rs.ShowcaseSlot.open(showcase, bundle, VERSION)


def copy_bundle(tmp_path: Path) -> Path:
    # the whole bundle (predict.py --check verifies the records too), less any showcase
    ignored = shutil.ignore_patterns("__pycache__", "showcase", "showcase.*")
    return Path(shutil.copytree(ROOT, tmp_path / "bundle", ignore=ignored))


def _status(c) -> dict:
    """The slot status: /v1/showcase, the same object /health and /v1/info carry."""
    status = c.get("/v1/showcase").json()
    assert status == c.get("/health").json()["showcase"] == c.get("/v1/info").json()["showcase"]
    assert tuple(status) == STATUS_FIELDS
    return status


@pytest.fixture(scope="module")
def showcase(tmp_path_factory):
    return build(tmp_path_factory.mktemp("slot") / "showcase")


@pytest.fixture(scope="module")
def client(showcase):
    with TestClient(app_for(showcase)) as c:
        yield c


def test_the_module_declares_the_slot_routes(client):
    endpoints = {f"{method} {path}" for method, path in rs.ENDPOINTS}
    assert {e.replace("{path:path}", "{path}") for e in endpoints} == set(SLOT["routes"])
    assert endpoints <= set(MANIFEST["serve"]["endpoints"])
    registered = {f"{m} {r.path}" for r in client.app.routes
                  for m in (getattr(r, "methods", None) or ())}
    assert endpoints <= registered
    assert rs.OPEN_WHILE_UNREADY == ("/", "/console", "/showcase/", "/v1/showcase")
    assert (rs.CAPABILITY, rs.MODEL_ROUTES, rs.SLOT_MAX_BYTES) == ("showcase", (), 64 << 20)
    assert ENV == MANIFEST["serve"]["identity"]["env"] + "_SHOWCASE"
    assert (SLOT["dir"], SLOT["schema"], rs.SCHEMA) == ("showcase",) + ("vega/showcase@1",) * 2
    assert not [rel for rel in MANIFEST["files"] if rel.startswith("showcase/")]


def test_an_absent_showcase_is_404_and_the_root_is_the_console(tmp_path):
    for where in (tmp_path / "none", tmp_path):
        with TestClient(app_for(where)) as c:
            assert c.get("/health").status_code == 200
            for res in map(c.get, ("/showcase/page/index.html", "/v1/story", "/report.pdf")):
                assert res.status_code == 404 and "problem" in res.headers["content-type"]
            assert c.get("/").content == c.get("/console").content
            status = _status(c)
            assert (status["present"], status["verified"], status["reason"]) == (False, False, None)


def test_a_verified_showcase_is_served_with_its_headers(client):
    root = client.get("/")
    assert root.status_code == 200 and root.content == PAGE
    assert root.content != client.get("/console").content
    assert root.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in root.headers["content-security-policy"]
    answers = {path: client.get(path) for path in (
        "/v1/story", "/report.pdf", "/showcase/page/js/main.js", "/showcase/page/page.css")}
    for path, res in dict(answers, **{"/": root}).items():
        assert res.status_code == 200, path
        assert res.headers["cache-control"] == "no-cache", path
        assert res.headers["x-content-type-options"] == "nosniff", path
        assert res.headers["etag"].startswith('"sha256:'), path
    assert answers["/v1/story"].content == FILES["story.json"]
    assert answers["/v1/story"].headers["content-type"].startswith("application/json")
    assert answers["/report.pdf"].content.startswith(b"%PDF-")
    assert answers["/report.pdf"].headers["content-type"] == "application/pdf"
    assert answers["/showcase/page/js/main.js"].headers["content-type"].startswith("text/java")
    assert answers["/showcase/page/page.css"].headers["content-type"].startswith("text/css")


def test_the_status_is_one_object_in_every_place(client, showcase):
    status = _status(client)
    assert (status["present"], status["verified"], status["reason"]) == (True, True, None)
    assert status["digest"] == _digest((showcase / "SHOWCASE.json").read_bytes())
    assert [*map(status.get, ("entry", "files", "unlisted"))] == ["page/index.html", len(FILES), []]
    assert (status["report_status"], status["created_at"]) == ("linked", "2026-09-30T12:00:00Z")
    assert status["template"] == {"version": "2.3.0", "manifest_digest": _digest(b"template")}


def test_the_etag_is_the_digest_and_answers_304(client):
    tag = client.get("/showcase/page.json").headers["etag"]
    assert tag == f'"{_digest(FILES["page.json"])}"'
    other = f'"sha256:{"0" * 64}"'
    for header in (tag, "W/" + tag, f"{other}, {tag}", "*"):
        again = client.get("/showcase/page.json", headers={"If-None-Match": header})
        assert again.status_code == 304 and again.content == b"", header
        assert again.headers["etag"] == tag and again.headers["cache-control"] == "no-cache"
    assert client.get("/showcase/page.json", headers={"If-None-Match": other}).status_code == 200
    story = client.get("/v1/story").headers["etag"]
    assert client.get("/v1/story", headers={"If-None-Match": story}).status_code == 304


def test_the_files_are_the_ones_read_at_start(client, showcase):
    (showcase / "story.json").write_bytes(b'{"altered": true}\n')
    try:
        for path in ("/v1/story", "/showcase/story.json"):
            assert client.get(path).content == FILES["story.json"], path
    finally:
        (showcase / "story.json").write_bytes(FILES["story.json"])


def test_paths_cannot_escape_and_only_served_files_answer(client):
    for path in ("/showcase/../MANIFEST.json", "/showcase/%2e%2e/MANIFEST.json",
                 "/showcase/..%2fMANIFEST.json", "/showcase/SHOWCASE.json",
                 "/showcase/plan/facts.json", "/showcase/page", "/showcase/",
                 "/showcase/page//index.html", "/showcase/%2fetc/hostname"):
        res = client.get(path)
        assert res.status_code == 404, path
        assert b"model_version" not in res.content, path


def test_a_showcase_file_never_shadows_a_service_route(client):
    assert client.get("/showcase/health.json").json() == {"shadow": "health"}
    assert client.get("/showcase/v1/predict.json").json() == {"shadow": "predict"}
    health = client.get("/health").json()
    assert health["ready"] is True and health["showcase"]["verified"] is True
    # the shipped request of this shape (a row; a window's last row with its history; a graph)
    request = json.loads((ROOT / "examples" / "request.json").read_text(encoding="utf-8"))
    answer = client.post("/v1/predict", json=request)
    assert answer.status_code == 200 and answer.json()["model_version"] == VERSION
    assert answer.json()["predictions"][0]["answered"] is True, answer.json()


def test_the_slot_answers_while_the_model_is_not_ready(showcase):
    c = TestClient(app_for(showcase))     # no lifespan: the self-test has not run
    assert c.get("/health").status_code == 503
    assert c.get("/").content == PAGE
    assert c.get("/showcase/page/js/main.js").status_code == 200
    assert c.get("/v1/showcase").json()["verified"] is True
    assert c.get("/console").status_code == 200
    assert c.get("/v1/story").status_code == 503 and c.get("/v1/predict").status_code == 503


def test_a_tampered_showcase_is_refused_and_the_model_stays_ready(tmp_path):
    showcase = build(tmp_path / "showcase")
    (showcase / "page" / "index.html").write_bytes(bytes([PAGE[0] ^ 1]) + PAGE[1:])
    with TestClient(app_for(showcase)) as c:
        assert c.get("/health").status_code == 200 and c.get("/health").json()["ready"] is True
        status = _status(c)
        assert (status["present"], status["verified"]) == (True, False)
        assert "page/index.html" in status["reason"] and "digest" in status["reason"]
        assert c.get("/").content == c.get("/console").content
        for path in ("/v1/story", "/report.pdf", "/showcase/page.json"):
            assert c.get(path).status_code == 404, path


def _symlinked(showcase: Path, rel: str) -> None:
    """`rel` moved out of the showcase, a symlink to it (the same bytes) left in its place."""
    moved = showcase.parent / ("moved-" + rel.replace("/", "-"))
    shutil.move(str(showcase / rel), str(moved))
    (showcase / rel).symlink_to(moved, target_is_directory=moved.is_dir())


X = _digest(b"x")
DOC_REFUSALS = {        # SHOWCASE.json changed; what the reason says
    "schema": (lambda d: d.update(schema="vega/showcase@2"), "schema"),
    "another bundle": (lambda d: d["bundle"].update(manifest_digest=X), rs.OTHER_BUNDLE),
    "another model": (lambda d: d["bundle"].update(model_version=X), rs.OTHER_BUNDLE),
    "outside": (lambda d: d["files"].update({"../MANIFEST.json": X}), "outside"),
    "absolute": (lambda d: d["files"].update({"/etc/hostname": X}), "outside"),
    "unservable": (lambda d: d["files"].update({"health": X}), "extension"),
    "served unpinned": (lambda d: d["served"].append("page/other.css"), "does not pin"),
    "entry unserved": (lambda d: d["served"].remove("page/index.html"), "entry"),
    "story unserved": (lambda d: d["served"].remove("story.json"), "story.file"),
    "report digest": (lambda d: d["report"]["pdf"].update(digest=X), "report.pdf.digest"),
    "entry not a page": (lambda d: d.update(entry="page.json"), "is not an .html page"),
    "lists itself": (lambda d: d["files"].update({"SHOWCASE.json": X}), "lists itself"),
}
FILE_REFUSALS = {       # the directory changed; what the reason says
    "unlisted script": (lambda s: (s / "extra.js").write_bytes(b"1\n"), "extra.js is not pinned"),
    "symlink": (lambda s: _symlinked(s, "story.json"), "story.json is a symlink"),
    "symlinked dir": (lambda s: _symlinked(s, "data"), "data is a symlink"),
    "missing": (lambda s: (s / "page.json").unlink(), "page.json is pinned in SHOWCASE.json but"),
    "tampered": (lambda s: (s / "plan" / "facts.json").write_bytes(b"{}"),
                 "plan/facts.json does not match its digest"),
    "not a regular file": (lambda s: ((s / "page.json").unlink(), (s / "page.json").mkdir()),
                           "page.json is not a regular file"),
}


@pytest.mark.parametrize("case", sorted(DOC_REFUSALS) + sorted(FILE_REFUSALS))
def test_a_showcase_is_refused_with_its_reason(tmp_path, case):
    showcase = build(tmp_path / "showcase")
    if case in DOC_REFUSALS:
        edit(showcase, DOC_REFUSALS[case][0])
    else:
        FILE_REFUSALS[case][0](showcase)
    slot = opened(showcase)
    assert (slot.present, slot.verified) == (True, False)
    assert {**DOC_REFUSALS, **FILE_REFUSALS}[case][1] in slot.reason, slot.reason
    assert slot.page() is None and dict(slot.served) == {}
    assert slot.status()["unlisted"] == (["extra.js"] if case == "unlisted script" else [])


def test_the_scratch_directories_are_ignored(tmp_path):
    showcase = build(tmp_path / "showcase")
    for rel in (".work/draft.json", ".work/qps/page/js/main.js", ".smoke/shot.png",
                "docs/show-brief.md", "notes.aux", "checks.log"):
        (showcase / rel).parent.mkdir(parents=True, exist_ok=True)
        (showcase / rel).write_bytes(b"scratch\n")
    slot = opened(showcase)
    assert slot.verified, slot.reason
    assert set(slot.served) == set(FILES) - set(RECORDS)


def test_a_bundle_that_pins_a_showcase_path_is_refused(tmp_path):
    owned = json.loads(MANIFEST_BYTES)
    owned["files"]["showcase/page.json"] = _digest(FILES["page.json"])
    slot = opened(build(tmp_path / "showcase", bundle=_doc(owned)), _doc(owned))
    assert not slot.verified and "two owners" in slot.reason


def test_the_byte_cap_refuses_a_larger_showcase(tmp_path, monkeypatch):
    showcase = build(tmp_path / "showcase")
    assert opened(showcase).verified
    monkeypatch.setattr(rs, "SLOT_MAX_BYTES", sum(len(b) for b in FILES.values()) - 1)
    assert "cap" in opened(showcase).reason


def test_brief_links_serve_the_briefs_and_no_report(tmp_path):
    with TestClient(app_for(build(tmp_path / "showcase", BRIEFS))) as c:
        status = c.get("/v1/showcase").json()
        assert status["verified"] and status["report_status"] == "brief-links"
        brief = c.get("/showcase/briefs/net-brief.md")
        assert brief.status_code == 200 and brief.content == BRIEFS["briefs/net-brief.md"]
        assert brief.headers["content-type"] == "text/markdown; charset=utf-8"
        missing = c.get("/report.pdf")
        assert missing.status_code == 404 and "no report" in missing.json()["detail"]


def test_a_reemitted_bundle_gives_the_reemit_reason(tmp_path):
    bundle = copy_bundle(tmp_path)
    build(bundle / "showcase")                          # made before the re-emit
    reemitted = dict(MANIFEST, reemits=[{
        "at": "2026-09-30T13:00:00Z", "reason": "a serving template fix",
        "previous_manifest_digest": _digest(MANIFEST_BYTES),
        "templates_digest": _digest(b"templates"), "changed": ["serve/routes_showcase.py"]}])
    (bundle / "MANIFEST.json").write_bytes(_doc(reemitted))
    status = TestClient(app_for(None, bundle)).get("/v1/showcase").json()
    assert (status["verified"], status["reason"]) == (False, rs.STALE)
    assert rs.STALE == "the bundle was re-emitted after this showcase; re-run /vega-show"


PROBE = ("import json, sys; sys.path.insert(0, sys.argv[1]); from serve.app import create_app; "
         "from fastapi.testclient import TestClient\n"
         "with TestClient(create_app(sys.argv[1])) as c:\n"
         "    print(json.dumps([c.get('/health').json()['ready'], "
         "c.get('/v1/showcase').json()['verified'], c.get('/').text]))")


def test_a_showcase_inside_the_bundle_leaves_it_as_packed(tmp_path):
    bundle = copy_bundle(tmp_path)
    build(bundle / "showcase")
    env = {k: v for k, v in os.environ.items() if k != ENV}
    # one bundle per process: the copy is served, and checked, in interpreters of its own
    served = subprocess.run([sys.executable, "-c", PROBE, str(bundle)], cwd=str(tmp_path),
                            env=env, capture_output=True, text=True, timeout=600)
    assert served.returncode == 0, served.stderr[-3000:]
    assert json.loads(served.stdout.splitlines()[-1]) == [True, True, PAGE.decode()]
    check = subprocess.run([sys.executable, "predict.py", "--check"], cwd=str(bundle), env=env,
                           capture_output=True, text=True, timeout=600)
    assert check.returncode == 0, check.stdout + check.stderr[-3000:]
    assert (bundle / "MANIFEST.json").read_bytes() == MANIFEST_BYTES
    assert (ROOT / "MANIFEST.json").read_bytes() == MANIFEST_BYTES
