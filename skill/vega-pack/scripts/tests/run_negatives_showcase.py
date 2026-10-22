#!/usr/bin/env python3
"""
run_negatives_showcase.py — the showcase slot and the inspection routes of a packed bundle must
refuse what they say they refuse (D6 §12.4; D3 §16.3 NS1/NS2 are SS3/SS2; BUILD-PLAN T10).

    python3 run_negatives_showcase.py <matrix-dir> [-v] [--json <path>]

Each case copies a lane's bundle (L02's; L10's and L19's for the window and graph cases) to a
temp dir, seeds one defect, and builds the bundle's own app in a child process
(`serve.app.create_app(root)` under FastAPI's `TestClient`, the way the bundle's API tests do):
the service is the one that ships, started from the copy.

The showcase is the lane's own when the lane's show stage built one
(`<bundle>/showcase/SHOWCASE.json`); otherwise a fixture showcase is written into the copy for
that copy's MANIFEST.json (an entry page, a story, a report, an asset, the plan records), the
form `/vega-show` writes (vega/showcase@1) — the slot's gates are about the slot, not about
what the page says.

| id   | seed                                                         | expect                                    |
|------|--------------------------------------------------------------|-------------------------------------------|
| SS1  | one byte of the showcase's entry page flipped               | model routes 200; /health's showcase refused naming the entry; `GET /` is the console page; the story 404 |
| SS2  | SHOWCASE.json's bundle.manifest_digest from another bundle   | refused; made for another bundle, manifest_digest |
| SS3  | a pinned entry `../MANIFEST.json`                            | refused; outside                           |
| SS4  | an unpinned `extra.js`                                       | refused; extra.js, not pinned              |
| SS5  | showcase files named `health`, `v1/predict` (and pinned `health.json`, `v1/predict.json`) | `GET /health` is the service's JSON; `POST /v1/predict` answers from the model |
| SS6  | the story replaced by a symlink to `../records/…`             | refused; symlink                           |
| SS7  | the bundle's MANIFEST.files also pins a `showcase/` path     | refused; two owners                        |
| SS8  | positive: a verified showcase                                | `predict.py --check` passes in the copy; sha256(MANIFEST.json) unchanged; `GET /` is the entry |
| SS9  | SHOWCASE.json's schema an unknown version                    | refused; schema                            |
| IN1  | inspection off (`<ENV>_INSPECT=off`; IN1b: MANIFEST.serve.inspection.enabled false) | /v1/network, /v1/explain 404 naming inspection; /v1/predict unaffected |
| IN2  | explain with `{"row": {}}`                                   | 422                                        |
| IN3  | explain with a target naming an unknown class                | 422 naming it                              |
| IN4  | L10: explain with `length − 2` history rows                   | 422; history, "window needs <length> rows" |
| IN5  | L19: explain a row whose keys match no reference node        | 200 with the warning "no neighbour in the reference graph" |
| IN6  | explain body above the body bound                            | 413 before the body is read                |
| IN7  | 1 + max_inflight concurrent explains under a blocked model   | the extra one 503 with Retry-After; the others 200 |

Exit 0 when every case behaved as stated; `--json` writes `vega/negatives-result@1`. A lane the
matrix does not hold leaves its cases not run (said, never passed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parents[2]
sys.path.insert(0, str(SKILLS / "vega" / "scripts" / "tests" / "negatives"))

import negkit  # noqa: E402

_ap = argparse.ArgumentParser(description="the showcase slot and the inspection routes, seeded")
_ap.add_argument("matrix", help="the matrix dir (lanes/<id>/fixture.json and the lane bundles)")
_ap.add_argument("-v", dest="verbose", action="store_true")
_ap.add_argument("--json", default=None, metavar="PATH")
ARGS = _ap.parse_args()
MATRIX = Path(ARGS.matrix).resolve()
CASES = negkit.Cases(__file__, MATRIX, ARGS.verbose)
SCHEMA = "vega/showcase@1"
BLOCKED_WAIT = 0.3            # <ENV>_INFLIGHT_WAIT for IN7: a place not found this fast is a 503


# --------------------------------------------------------------------------- the child

CHILD = r"""
import hashlib, json, os, sys, threading, time
from pathlib import Path
root = Path(sys.argv[1]).resolve()
plan = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
env = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))["serve"]["identity"]["env"]
for name, value in (plan.get("env") or {}).items():
    os.environ[f"{env}_{name}"] = value
sys.path.insert(0, str(root))
from fastapi.testclient import TestClient
from serve.app import create_app

def seen(res):
    body = res.content
    try:
        doc = res.json()
    except ValueError:
        doc = None
    return {"status": res.status_code, "headers": {k.lower(): v for k, v in res.headers.items()},
            "digest": "sha256:" + hashlib.sha256(body).hexdigest(), "json": doc,
            "text": body[:3000].decode("utf-8", "replace")}

def send(c, step):
    kw = {}
    if "json" in step:
        kw["json"] = step["json"]
    if "content" in step:
        kw["content"] = step["content"].encode("utf-8")
        kw["headers"] = {"content-type": "application/json"}
    return c.request(step["method"], step["path"], **kw)

def blocked(c, step):
    import serve.routes_inspect as ri
    gate, entered = threading.Event(), threading.Semaphore(0)
    original = ri.InspectionSurface.explain_answer
    def held(self, *a, **k):
        entered.release()
        gate.wait(60)
        return original(self, *a, **k)
    ri.InspectionSurface.explain_answer = held
    places = c.get("/health").json()["max_inflight"]
    answers = [None] * places
    def one(i):
        answers[i] = send(c, step).status_code
    threads = [threading.Thread(target=one, args=(i,)) for i in range(places)]
    for t in threads:
        t.start()
    took = all(entered.acquire(timeout=120) for _ in range(places))
    extra = seen(send(c, step))
    gate.set()
    for t in threads:
        t.join(120)
    ri.InspectionSurface.explain_answer = original
    return {"places": places, "took": took, "extra": extra, "held": answers}

out = {}
with TestClient(create_app(root, **(plan.get("app") or {}))) as c:
    for step in plan["steps"]:
        out[step["name"]] = blocked(c, step) if step.get("blocked") else seen(send(c, step))
print("RESULT " + json.dumps(out))
"""


def serve(bundle: Path, steps, env=None, app=None) -> dict:
    """The bundle's app, started from `bundle` in a child process; each step's answer."""
    with tempfile.TemporaryDirectory(prefix="neg-plan-") as td:
        plan = Path(td) / "plan.json"
        plan.write_text(json.dumps({"steps": steps, "env": env or {}, "app": app or {}}),
                        encoding="utf-8")
        proc = subprocess.run([sys.executable, "-c", CHILD, str(bundle), str(plan)],
                              capture_output=True, text=True, timeout=900, cwd=str(bundle),
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                       PYTHONDONTWRITEBYTECODE="1"))
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("RESULT ")),
                None)
    if line is None:
        raise RuntimeError(f"the service did not start (exit {proc.returncode}): "
                           f"{(proc.stdout + proc.stderr)[-1500:]}")
    return json.loads(line[len("RESULT "):])


def get(path, name=None):
    return {"name": name or f"GET {path}", "method": "GET", "path": path}


def post(path, body, name=None):
    return {"name": name or f"POST {path}", "method": "POST", "path": path, "json": body}


# --------------------------------------------------------------------------- the showcase

def _doc(value) -> bytes:
    return (json.dumps(value, indent=1) + "\n").encode("utf-8")


def _digest(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


PAGE = (b'<!doctype html>\n<html><head><link rel="stylesheet" href="showcase/page/page.css">'
        b'<script type="module" src="showcase/page/js/main.js"></script></head>'
        b'<body data-fixture="negatives showcase entry"></body></html>\n')
FILES = {"page/index.html": PAGE,
         "page/page.css": b"body { margin: 0 }\n",
         "page/js/main.js": b"export const ready = true;\n",
         "page.json": _doc({"schema": "vega/page@1"}),
         "story.json": _doc({"schema": "vega/story@2", "sections": {}}),
         "data/attribution.json": _doc({"schema": "vega/attribution@2"}),
         "report/report.pdf": b"%PDF-1.4\n% a fixture report\n%%EOF\n",
         "plan/facts.json": _doc({"schema": "vega/show-facts@1"}),
         "plan/story-plan.json": _doc({"schema": "vega/story-plan@2"})}
RECORDS = ("plan/facts.json", "plan/story-plan.json")


def fixture_showcase(bundle: Path, files=None) -> None:
    """A showcase in `<bundle>/showcase/` for this bundle's MANIFEST.json (vega/showcase@1)."""
    files = FILES if files is None else files
    where = bundle / "showcase"
    manifest_bytes = (bundle / "MANIFEST.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for rel, body in files.items():
        (where / rel).parent.mkdir(parents=True, exist_ok=True)
        (where / rel).write_bytes(body)
    pins = {rel: _digest(body) for rel, body in sorted(files.items())}
    served = [rel for rel in pins if rel not in RECORDS]
    doc = {"schema": SCHEMA, "created_at": "2026-10-01T00:00:00Z", "produced_by": "/vega-show",
           "bundle": {"manifest": "MANIFEST.json", "manifest_digest": _digest(manifest_bytes),
                      "slug": str(manifest.get("slug") or bundle.name),
                      "run": str(manifest.get("run") or "R0"),
                      "model_version": manifest.get("model_version")},
           "model": {"path": "fixture.pack.html", "turtle_digest": _digest(b"turtle")},
           "entry": "page/index.html", "page": {"file": "page.json"},
           "template": {"name": "vega-show/page", "version": "2.3.0",
                        "manifest_digest": _digest(b"template")},
           "story": {"file": "story.json", "schema": "vega/story@2", "values_n": 0,
                     "placeholders_n": 0, "sections": ["context", "network", "results"]},
           "attribution": {"file": "data/attribution.json"},
           "plan": {"file": "plan/story-plan.json", "facts": "plan/facts.json"},
           "report": {"status": "linked", "briefs": None,
                      "aux": {"source": "main.aux", "digest": _digest(b"")},
                      "pdf": {"source": "docs/main.pdf", "file": "report/report.pdf",
                              "digest": pins["report/report.pdf"]},
                      "roles": {"source": "docs/label-roles.json", "origin": "proposed",
                                "digest": _digest(b"roles"), "bound": [], "missing": []}},
           "files": pins, "served": served,
           "roles": {"page": [r for r in served if r.startswith("page")],
                     "story": ["story.json"], "report": ["report/report.pdf"],
                     "data": ["data/attribution.json"], "record": list(RECORDS)},
           "requires": {"endpoints": [], "examples": "vega/examples@2",
                        "network_view": "vega/network-view@2",
                        "explanation": "vega/explanation@2"},
           "supersedes": None, "markers": ["OK: a fixture showcase (run_negatives_showcase)"]}
    (where / "SHOWCASE.json").write_bytes(_doc(doc))


class Source:
    """The lane bundle the cases copy, and whether its showcase is the lane's own."""

    def __init__(self, lane):
        self.lane, self.bundle = lane, lane.bundle()
        self.own = (self.bundle / "showcase" / "SHOWCASE.json").is_file()

    def copy(self, td) -> Path:
        """The bundle copied to `td`, a verified-able showcase in it."""
        dst = Path(td) / self.bundle.name
        shutil.copytree(self.bundle, dst, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".smoke"))
        if not self.own:
            shutil.rmtree(dst / "showcase", ignore_errors=True)
            fixture_showcase(dst)
        return dst


def showcase_doc(bundle: Path) -> dict:
    return json.loads((bundle / "showcase" / "SHOWCASE.json").read_text(encoding="utf-8"))


def edit_showcase(bundle: Path, change) -> None:
    doc = showcase_doc(bundle)
    change(doc)
    (bundle / "showcase" / "SHOWCASE.json").write_bytes(_doc(doc))


def slot(answers) -> dict:
    return (answers["GET /v1/showcase"]["json"] or {})


def request_of(bundle: Path) -> dict:
    return json.loads((bundle / "examples" / "request.json").read_text(encoding="utf-8"))


SLOT_STEPS = [get("/v1/showcase"), get("/"), get("/console"), get("/health")]


def refused(name, answers, *needles) -> None:
    """The slot refused (present, not verified) with every needle in its reason; the model
    stayed ready."""
    status = slot(answers)
    reason = str(status.get("reason") or "")
    problems = []
    if status.get("present") is not True or status.get("verified") is not False:
        problems.append(f"the slot is {status}")
    problems += [f"the reason does not mention {n!r}" for n in needles
                 if n.lower() not in reason.lower()]
    health = answers["GET /health"]["json"] or {}
    if health.get("ready") is not True:
        problems.append("the model is not ready beside the refused showcase")
    if answers["GET /"]["digest"] != answers["GET /console"]["digest"]:
        problems.append("GET / is not the console page while the showcase is refused")
    CASES.record(name, problems, reason)


# --------------------------------------------------------------------------- SS1–SS9

def slot_cases(src: Source) -> None:
    CASES.section("the showcase slot — a showcase is served only as it was built, for this "
                  f"bundle (L02; {'the lane showcase' if src.own else 'a fixture showcase'})")
    one = request_of(src.bundle)

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        entry = showcase_doc(copy)["entry"]
        story = showcase_doc(copy)["story"]["file"]
        page = copy / "showcase" / entry
        body = bytearray(page.read_bytes())
        body[len(body) // 2] ^= 0x01
        page.write_bytes(bytes(body))
        answers = serve(copy, SLOT_STEPS + [get("/v1/story"), post("/v1/predict", one)])
        refused("SS1  one flipped byte of the entry refuses the showcase, naming the file",
                answers, entry, "digest")
        CASES.check("SS1  ...the model still answers, and the story is 404",
                    answers["POST /v1/predict"]["status"] == 200
                    and answers["GET /v1/story"]["status"] == 404,
                    f"predict {answers['POST /v1/predict']['status']}, story "
                    f"{answers['GET /v1/story']['status']} ({story})")

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        other = _digest(b"another bundle's MANIFEST.json")
        edit_showcase(copy, lambda d: d["bundle"].update(manifest_digest=other))
        refused("SS2  a showcase made for another bundle is refused",
                serve(copy, SLOT_STEPS), "made for another bundle", "manifest_digest")

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        edit_showcase(copy, lambda d: d["files"].update(
            {"../MANIFEST.json": _digest((copy / "MANIFEST.json").read_bytes())}))
        refused("SS3  a pinned path outside the showcase is refused",
                serve(copy, SLOT_STEPS), "../MANIFEST.json", "outside")

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        (copy / "showcase" / "extra.js").write_text("export const x = 1;\n", encoding="utf-8")
        refused("SS4  an unpinned extra.js is refused", serve(copy, SLOT_STEPS),
                "extra.js", "not pinned")

    with tempfile.TemporaryDirectory() as td:
        _shadows(src, td, one)

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        story = copy / "showcase" / showcase_doc(copy)["story"]["file"]
        records = sorted(p for p in (copy / "records").glob("*.json"))
        target = records[0] if records else copy / "schema.json"
        story.unlink()
        story.symlink_to(os.path.relpath(target, story.parent))
        refused("SS6  a story that is a symlink into the bundle is refused",
                serve(copy, SLOT_STEPS), "symlink")

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        path = copy / "MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entry = showcase_doc(copy)["entry"]
        manifest["files"][f"showcase/{entry}"] = showcase_doc(copy)["files"][entry]
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        refused("SS7  a showcase path the bundle's MANIFEST also pins is refused",
                serve(copy, SLOT_STEPS), "two owners")

    with tempfile.TemporaryDirectory() as td:
        _positive(src, td, one)

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        edit_showcase(copy, lambda d: d.update(schema="vega/showcase@99"))
        refused("SS9  a showcase manifest of an unknown schema is refused",
                serve(copy, SLOT_STEPS), "schema", "vega/showcase@99")


def _shadows(src: Source, td, one) -> None:
    """SS5: showcase files named like the service's routes shadow none of them."""
    copy = src.copy(td)
    root = copy / "showcase"
    for rel, text in (("health", "shadow"), ("v1/predict", "shadow")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    added = {"health.json": _doc({"shadow": "health"}),
             "v1/predict.json": _doc({"shadow": "predict"})}
    for rel, body in added.items():
        (root / rel).write_bytes(body)

    def pin(doc):
        doc["files"].update({rel: _digest(body) for rel, body in added.items()})
        doc["served"] = list(doc["served"]) + list(added)
    edit_showcase(copy, pin)
    answers = serve(copy, SLOT_STEPS + [post("/v1/predict", one),
                                        get("/showcase/health.json")])
    health, predicted = answers["GET /health"], answers["POST /v1/predict"]
    status = slot(answers)
    problems = [] if status.get("verified") is True else [f"the showcase is not verified: "
                                                          f"{status.get('reason')}"]
    if health["status"] != 200 or "ready" not in (health["json"] or {}):
        problems.append(f"GET /health is not the service's status: {health['text'][:200]}")
    if predicted["status"] != 200 or "predictions" not in (predicted["json"] or {}):
        problems.append(f"POST /v1/predict did not answer from the model: "
                        f"{predicted['text'][:200]}")
    shown = answers["GET /showcase/health.json"]
    if shown["status"] != 200 or (shown["json"] or {}).get("shadow") != "health":
        problems.append("the showcase's own health.json is not served under /showcase/")
    CASES.record("SS5  showcase files named health and v1/predict shadow no service route",
                 problems)


def _positive(src: Source, td, one) -> None:
    """SS8: a verified showcase changes nothing the bundle verifies."""
    copy = src.copy(td)
    before = _digest((copy / "MANIFEST.json").read_bytes())
    doc = showcase_doc(copy)
    entry = (copy / "showcase" / doc["entry"]).read_bytes()
    answers = serve(copy, SLOT_STEPS + [post("/v1/predict", one), get("/v1/story")])
    status = slot(answers)
    problems = [] if status.get("verified") is True else [f"not verified: {status.get('reason')}"]
    if answers["GET /"]["digest"] != _digest(entry):
        problems.append("GET / is not the showcase's entry page")
    if answers["GET /v1/story"]["status"] != 200:
        problems.append(f"GET /v1/story answered {answers['GET /v1/story']['status']}")
    CASES.record("SS8  a verified showcase is served: GET / is its entry, its story answers",
                 problems)
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(td),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    check = subprocess.run([sys.executable, "predict.py", "--check"], cwd=str(copy),
                           capture_output=True, text=True, timeout=900, env=env)
    CASES.expect("SS8  ...predict.py --check passes in the copy that holds it",
                 check.returncode, check.stdout + check.stderr, code=0,
                 contains=("checks passed",))
    after = _digest((copy / "MANIFEST.json").read_bytes())
    CASES.check("SS8  ...and sha256(MANIFEST.json) is unchanged", after == before,
                f"MANIFEST.json changed: {before} -> {after}")


# --------------------------------------------------------------------------- IN1–IN7

def _problem(answer, status, *needles) -> list:
    got = answer["status"]
    problems = [] if got == status else [f"answered {got}, expected {status}"]
    text = answer["text"].lower()
    problems += [f"the answer does not mention {n!r}" for n in needles if n.lower() not in text]
    return problems


def explain_body(bundle: Path) -> dict:
    """The shipped one-sample request, as an explain body (row: the row; else rows at 0)."""
    one = request_of(bundle)
    shape = (json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8")).get("sample")
             or {}).get("shape", "row")
    if shape == "row":
        return {"row": one["rows"][0]}
    return dict(one, at=len(one["rows"]) - 1 if shape == "window" else 0)


def inspection_cases(src: Source, window, graph) -> None:
    CASES.section("the inspection routes — off means 404, a bad request 422, the edge holds (L02)")
    body = explain_body(src.bundle)
    one = request_of(src.bundle)
    routes = [get("/v1/network"), post("/v1/explain", body), post("/v1/predict", one)]

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        answers = serve(copy, routes, env={"INSPECT": "off"})
        problems = _problem(answers["GET /v1/network"], 404, "inspection")
        problems += _problem(answers["POST /v1/explain"], 404, "inspection")
        problems += _problem(answers["POST /v1/predict"], 200)
        CASES.record("IN1  <ENV>_INSPECT=off: the network and explain routes are 404 naming "
                     "inspection, predict answers", problems)

        path = copy / "MANIFEST.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["serve"]["inspection"]["enabled"] = False
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        answers = serve(copy, routes + [get("/health")])
        problems = _problem(answers["GET /v1/network"], 404, "inspection")
        problems += _problem(answers["POST /v1/explain"], 404, "inspection")
        problems += _problem(answers["POST /v1/predict"], 200)
        enabled = ((answers["GET /health"]["json"] or {}).get("inspection") or {}).get("enabled")
        if enabled is not False:
            problems.append(f"/health.inspection.enabled is {enabled!r}")
        CASES.record("IN1b  a bundle whose MANIFEST turns inspection off answers 404 naming it",
                     problems)

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        target = dict(body, target={"class": "no-such-class"})
        answers = serve(copy, [post("/v1/explain", {"row": {}}, "empty"),
                               post("/v1/explain", target, "target"),
                               post("/v1/explain", body, "control")])
        CASES.record("IN2  explain with an empty row is 422", _problem(answers["empty"], 422))
        CASES.record("IN3  explain with a target naming an unknown class is 422 naming it",
                     _problem(answers["target"], 422, "no-such-class"), answers["target"]["text"])
        CASES.record("IN2  control: the shipped example is explained (200)",
                     _problem(answers["control"], 200))

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        bound = 4096
        per_row = max(1, len(json.dumps(one["rows"][0])))
        wide = {"rows": one["rows"][:1] * (1 + 4 * bound // per_row)}
        answers = serve(copy, [post("/v1/explain", wide, "wide"),
                               post("/v1/explain", body, "within")],
                        env={"MAX_BODY_BYTES": str(bound)})
        problems = _problem(answers["wide"], 413)
        if not answers["wide"]["headers"].get("content-type", "").startswith(
                "application/problem+json"):
            problems.append("the 413 is not a problem body")
        CASES.record(f"IN6  an explain body over the {bound}-byte bound is 413 before it is read",
                     problems)
        CASES.record("IN6  control: an explain body within the bound is answered",
                     _problem(answers["within"], 200))

    with tempfile.TemporaryDirectory() as td:
        copy = src.copy(td)
        answers = serve(copy, [dict(post("/v1/explain", body, "blocked"), blocked=True)],
                        env={"INFLIGHT_WAIT": str(BLOCKED_WAIT)}, app={"max_inflight": 2})
        held = answers["blocked"]
        extra = held["extra"]
        problems = [] if held["took"] else ["the held explains never reached the model"]
        problems += _problem(extra, 503)
        if not extra["headers"].get("retry-after"):
            problems.append("the 503 carries no Retry-After")
        if held["held"] != [200] * held["places"]:
            problems.append(f"the held explains answered {held['held']}")
        CASES.record(f"IN7  1 + max_inflight ({held['places']}) concurrent explains under a "
                     "blocked model: the extra one is 503 with Retry-After", problems)

    _window_case(window)
    _graph_case(graph)


def _window_case(window) -> None:
    name = "IN4  L10: explain with length − 2 history rows is 422 naming history and length"
    if window is None:
        return
    with tempfile.TemporaryDirectory() as td:
        copy = Source(window).copy(td)
        one = request_of(copy)
        sample = json.loads((copy / "MANIFEST.json").read_text(encoding="utf-8"))["sample"]
        length = int(sample["length"])
        rows = one["rows"][-1:]
        history = (list(one.get("history") or []) + one["rows"][:-1])[-(length - 2):]
        short = {"rows": rows, "history": history, "at": 0}
        full = {"rows": rows, "history": (list(one.get("history") or [])
                                          + one["rows"][:-1])[-(length - 1):], "at": 0}
        answers = serve(copy, [post("/v1/explain", short, "short"),
                               post("/v1/explain", full, "full")])
        # the query rules say the length as the window's row count ("a window needs 24 rows
        # ending at `at` ... send 1 more as history")
        CASES.record(name, _problem(answers["short"], 422, "history", "window needs",
                                    f"{length} rows"), answers["short"]["text"])
        CASES.record("IN4  control: the same anchor with length − 1 history rows is explained",
                     _problem(answers["full"], 200))


def _graph_case(graph) -> None:
    if graph is None:
        return
    with tempfile.TemporaryDirectory() as td:
        copy = Source(graph).copy(td)
        one = request_of(copy)
        meta = json.loads((copy / "graph" / "graph.json").read_text(encoding="utf-8"))
        keys = [c.split(":", 1)[-1] for c in meta.get("edges_for_columns") or ()]
        row = dict(one["rows"][0])
        for key in keys:
            if key in row:
                row[key] = f"NEG-{key}-matches-no-node"
        answers = serve(copy, [post("/v1/explain", {"rows": [row], "at": 0}, "lonely"),
                               post("/v1/explain", {"rows": one["rows"][:1], "at": 0},
                                    "control")])
        lonely = answers["lonely"]
        problems = _problem(lonely, 200)
        warned = json.dumps(((lonely["json"] or {}).get("explanation") or {}).get("warnings")
                            or (lonely["json"] or {}).get("warnings") or lonely["text"])
        # the served wording: "no neighbour in the reference graph — answered from the row's
        # own features" (the row attaches by no edge)
        if "no neighbour in the reference graph" not in warned.lower():
            problems.append(f"no warning names the missing edges: {warned[:300]}")
        CASES.record(f"IN5  L19: a row whose keys ({', '.join(keys)}) match no reference node "
                     "is explained with a warning naming no edges", problems)
        CASES.record("IN5  control: the shipped example row is explained",
                     _problem(answers["control"], 200))


# --------------------------------------------------------------------------- main

def lane_or_none(lane_id, names):
    lane = negkit.lane(MATRIX, lane_id)
    why = None
    if lane is None:
        why = f"lane {lane_id} has no fixture in {MATRIX}"
    elif lane.bundle() is None:
        why = f"lane {lane_id} packed no bundle in {MATRIX}"
    elif not (lane.bundle() / "serve" / "app.py").is_file():
        why = f"lane {lane_id}'s bundle has no service"
    if why:
        for name in names:
            CASES.not_run(name, why)
        return None
    return lane


def main() -> int:
    CASES.section(f"the lanes (matrix: {MATRIX})")
    l02 = lane_or_none("L02", [f"SS{i}  the showcase slot on L02" for i in range(1, 10)]
                       + [f"IN{i}  the inspection routes on L02" for i in (1, 2, 3, 6, 7)])
    window = lane_or_none("L10", ["IN4  L10: explain with length − 2 history rows"])
    graph = lane_or_none("L19", ["IN5  L19: a row whose keys match no reference node"])
    if l02 is not None:
        src = Source(l02)
        print(f"  L02: {src.bundle} — "
              + ("its own showcase" if src.own else "no showcase built yet: a fixture showcase "
                                                    "is written into each copy"))
        slot_cases(src)
        inspection_cases(src, window, graph)
    else:
        CASES.section("the inspection routes of the window and graph lanes")
        if window is not None or graph is not None:
            _window_case(window)
            _graph_case(graph)
    return CASES.finish(ARGS.json)


if __name__ == "__main__":
    sys.exit(main())
