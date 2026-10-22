"""
_negatives_served — `run_negatives.py`'s cases on what serves the showcase and what describes it:
the slot (SL1–SL11, the bundle's own `serve/routes_showcase.py`, the module the service runs,
opened on a seeded showcase in a child started from the bundle; SL11 through the bundle's app),
the layer (LY1–LY7, `apply_show_layer.py` on seeded authoring data or a seeded recorded check or
walk) and the served walk (SV1–SV5, PG10: `smoke_show.py --serve --strict` on a showcase seeded
and re-pinned, so the slot verifies it and the walk meets the seed; PG12: a pseudo-locale build
walked by the lanes' page stage).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from _negatives_kit import (CATALOG, SKILLS, Check, Expect, case, edit_json, edit_text, py, read_json, refused,
                            run, sha, show, template_copy, write_json)

BASE = "L02"
ABSTRACT = ("The page shows a reader outside the chain how the network reads each parcel and how it "
            "did on the sealed test part.")
NO_REPORT = "WARN: no report — code notes link to the stage briefs; re-run /vega-show after /document-project"
NOT_CHECKED = "WARN: served page not checked — the served walk is not this case's"

SLOT_CHILD = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from serve.routes_showcase import ShowcaseSlot
raw = Path('MANIFEST.json').read_bytes()
print('RESULT ' + json.dumps(ShowcaseSlot.open(Path('showcase'), raw, json.loads(raw)['model_version']).status()))
"""

APP_CHILD = r"""
import json, sys
sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from serve.app import create_app
out = {}
with TestClient(create_app('.')) as c:
    for path in sys.argv[1:]:
        r = c.get(path)
        out[path] = {'status': r.status_code, 'type': r.headers.get('content-type'), 'body': r.text[:400]}
print('RESULT ' + json.dumps(out))
"""


def _result(rc_out) -> dict:
    rc, out = rc_out
    line = next((ln for ln in reversed(out.splitlines()) if ln.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError(f"the child exited {rc} without a result: {out[-1200:]}")
    return json.loads(line[len("RESULT "):])


def slot_status(w) -> dict:
    """The slot's status, as the bundle's own module opens the seeded showcase."""
    return _result(run([sys.executable, "-c", SLOT_CHILD], cwd=w.bundle, env={"CUDA_VISIBLE_DEVICES": ""}))


def slot_refuses(w, *words) -> Check:
    status = slot_status(w)
    reason = str(status.get("reason") or "")
    missing = [x for x in words if x.lower() not in reason.lower()]
    ok = status.get("present") is True and status.get("verified") is False and not missing
    return Check(ok, f"status {status}" + (f"; the reason does not name {missing}" if missing else ""),
                 json.dumps(status))


def flip(path: Path) -> None:
    raw = path.read_bytes()
    path.write_bytes(raw[:-2] + bytes([raw[-2] ^ 1]) + raw[-1:])


# =============================================================================== slot

@case("SL1", "SS9", BASE, "SHOWCASE.json of an unknown schema version")
def sl1(w):
    edit_json(w.showcase / "SHOWCASE.json", lambda d: d.update(schema="vega/showcase@9"))
    return slot_refuses(w, "schema")


@case("SL2", "SS2 D5-27", BASE, "SHOWCASE.json's bundle.manifest_digest off by one hex")
def sl2(w):
    edit_json(w.showcase / "SHOWCASE.json", lambda d: d["bundle"].update(
        manifest_digest=d["bundle"]["manifest_digest"][:-1] + ("0" if d["bundle"]["manifest_digest"][-1] != "0" else "1")))
    return slot_refuses(w, "another bundle")


@case("SL3", "", BASE, "a showcase of a bundle re-emitted after it was built")
def sl3(w):
    previous = sha(w.manifest.read_bytes())
    edit_json(w.manifest, lambda m: m.setdefault("reemits", []).append(
        {"at": "2026-10-09T00:00:00Z", "reason": "templates", "previous_manifest_digest": previous,
         "changed": [], "added": [], "removed": []}))
    return slot_refuses(w, CATALOG["slot"]["stale_reason"])


@case("SL4", "SS3", BASE, "a pinned entry ../MANIFEST.json")
def sl4(w):
    edit_json(w.showcase / "SHOWCASE.json", lambda d: d["files"].update({"../MANIFEST.json": sha(w.manifest.read_bytes())}))
    return slot_refuses(w, "../MANIFEST.json")


@case("SL5", "SS6", BASE, "the story replaced by a symlink into the bundle's records")
def sl5(w):
    story = w.showcase / "story.json"
    story.unlink()
    story.symlink_to(w.bundle / "records" / "evaluation.json")
    return slot_refuses(w, "symlink")


@case("SL6", "SS1 D5-32", BASE, "one byte of the entry page flipped")
def sl6(w):
    flip(w.showcase / "page" / "index.html")
    return slot_refuses(w, "page/index.html")


@case("SL7", "", BASE, "a pinned file that takes the showcase over its byte cap")
def sl7(w):
    big = w.showcase / "data" / "big.json"
    with open(big, "wb") as fh:
        fh.write(b"[" + b" " * (64 * 1024 * 1024) + b"]")
    edit_json(w.showcase / "SHOWCASE.json", lambda d: d["files"].update({"data/big.json": sha(big.read_bytes())}))
    return slot_refuses(w, "bytes")


@case("SL8", "SS4", BASE, "an unpinned extra.js in the showcase")
def sl8(w):
    (w.showcase / "page" / "extra.js").write_text("export {};\n", encoding="utf-8")
    return slot_refuses(w, "extra.js")


@case("SL9", "D5-26", BASE, "a served path that files does not pin")
def sl9(w):
    edit_json(w.showcase / "SHOWCASE.json", lambda d: d["served"].append("extra.json"))
    return slot_refuses(w, "extra.json")


@case("SL10", "SS7", BASE, "the bundle's MANIFEST.files also pins a showcase/ path")
def sl10(w):
    edit_json(w.manifest, lambda m: m["files"].update({"showcase/story.json": sha(b"")}))
    return slot_refuses(w, "two owners")


@case("SL11", "SS5", BASE, "showcase files named health.json and v1/predict.json, pinned and served")
def sl11(w):
    for rel in ("health.json", "v1/predict.json"):
        (w.showcase / rel).parent.mkdir(parents=True, exist_ok=True)
        (w.showcase / rel).write_text('{"seeded": "a showcase file"}\n', encoding="utf-8")

    def pin(doc):
        for rel in ("health.json", "v1/predict.json"):
            doc["files"][rel] = sha((w.showcase / rel).read_bytes())
            doc["served"].append(rel)
    edit_json(w.showcase / "SHOWCASE.json", pin)
    got = _result(run([sys.executable, "-c", APP_CHILD, "/health", "/health.json", "/v1/predict.json",
                       "/showcase/health.json"], cwd=w.bundle, env={"CUDA_VISIBLE_DEVICES": ""}))
    health = got["/health"]
    problems = []
    if health["status"] != 200 or "seeded" in health["body"] or '"ready"' not in health["body"]:
        problems.append(f"GET /health is not the service's own answer: {health}")
    problems += [f"GET {p} answered {got[p]['status']} with the showcase file" for p in ("/health.json", "/v1/predict.json")
                 if "seeded" in got[p]["body"]]
    if got["/showcase/health.json"]["status"] != 200 or "seeded" not in got["/showcase/health.json"]["body"]:
        problems.append(f"the showcase file is not served under /showcase/: {got['/showcase/health.json']}")
    return Check(not problems, "; ".join(problems), json.dumps(got)[:1500])


# =============================================================================== layer

def author(w, abstract: str = ABSTRACT, markers=()) -> None:
    write_json(w.layer_data, {"from_showcase": str(w.showcase), "readings": {"abstract": abstract},
                              "markers": list(markers)})


def layer_refused(w, gate: str, *extra) -> Expect:
    rc, out = w.layer(*extra)
    if w.layered.exists():
        return Check(False, f"{w.layered.name} was written", out)
    return refused((rc, out), gate)


@case("LY1", "D5-34", BASE, "the recorded check is stale")
def ly1(w):
    author(w)
    edit_json(w.showcase / ".work" / "checks.json", lambda d: d.update(
        status="stale", problems=[{"rule": "CK7", "where": "report.pdf", "message": "seeded"}]))
    return layer_refused(w, "LY1")


@case("LY2", "VS21", BASE, "the served walk recorded as failed")
def ly2(w):
    author(w)
    edit_json(w.showcase / ".smoke" / "smoke.json", lambda d: d.update(status="fail"))
    return layer_refused(w, "LY2")


@case("LY3", "D5-33", BASE, "a layer abstract with a digit")
def ly3(w):
    author(w, abstract="The page shows the network to its 3 readers and how it did on the sealed test part.")
    return layer_refused(w, "LY3")


@case("LY4", "D5-35", "L03", "brief-link mode without the WARN: no report marker")
def ly4(w):
    author(w, markers=[NOT_CHECKED])
    return layer_refused(w, "LY4")


@case("LY5", "", BASE, "a showcase that does not pin the bundle's current MANIFEST.json")
def ly5(w):
    author(w)
    w.manifest.write_bytes(w.manifest.read_bytes() + b"\n")
    return layer_refused(w, "LY5")


@case("LY6", "", BASE, "a render asset over its cap")
def ly6(w):
    author(w)
    big = w.root / "big.js"
    big.write_text("/*" + "x" * (CATALOG["layer"]["max_render_kb"] * 1024) + "*/", encoding="utf-8")
    return layer_refused(w, "LY6", "--render", big)


@case("LY7", "VS22", BASE, "--out equal to the model page it reads")
def ly7(w):
    author(w)
    before = w.model.read_bytes()
    rc, out = show("apply_show_layer.py", w.model, "--data", w.layer_data, "--out", w.model, cwd=w.work)
    if w.model.read_bytes() != before:
        return Check(False, "the model page was rewritten", out)
    return refused((rc, out), "LY7")


# =============================================================================== served walk

def walked(w, gate: str, *words) -> Check:
    """The served walk fails, its smoke.json naming `gate` not ok with `words` in its problems."""
    rc, out = w.walk()
    smoke = w.showcase / ".smoke" / "smoke.json"
    if not smoke.is_file():
        return Check(False, f"exit {rc} and no smoke.json", out)
    doc = read_json(smoke)
    checks = {c["id"]: c for c in doc.get("checks") or []}
    gate_check = checks.get(gate) or {}
    said = " ".join(gate_check.get("problems") or [])
    problems = ([] if rc == 1 else [f"exit {rc}, expected 1"]) + \
        ([] if doc.get("status") == "fail" else [f"status {doc.get('status')}"]) + \
        ([] if gate_check and not gate_check.get("ok") else [f"{gate} did not fail"]) + \
        [f"{gate} does not name {x!r}" for x in words if x.lower() not in said.lower()]
    return Check(not problems, "; ".join(problems), out)


@case("SV1", "", BASE, "story.json edited after the build, its digest not rebuilt")
def sv1(w):
    edit_json(w.showcase / "story.json", lambda s: s.update(headline="A hand-written headline."))
    return walked(w, "SV1", "not verified")


@case("SV2", "", BASE, "a view module that throws on the page, re-pinned")
def sv2(w):
    rel = "page/js/views/results.js"
    with open(w.showcase / rel, "a", encoding="utf-8") as fh:
        fh.write('\nsetTimeout(() => { throw new Error("seeded page error"); }, 0);\n')
    w.repin(rel)
    return walked(w, "SV2", "seeded page error")


@case("SV3", "D5-32", BASE, "story.json with a placeholder's value removed, re-pinned")
def sv3(w):
    story = read_json(w.showcase / "story.json")
    key = next(k for k in story["values"] if "{" + k in json.dumps(story.get("headline", "")) + json.dumps(story.get("lead", ""))
               or "{" + k in json.dumps(story["sections"]))
    story["values"].pop(key)
    write_json(w.showcase / "story.json", story)
    w.repin("story.json")
    return walked(w, "SV3", key)


@case("SV4", "", BASE, "a code note linking a file the showcase does not serve, re-pinned")
def sv4(w):
    def relink(story):
        note = next(n for n in story["code_notes"].values() if (n or {}).get("link"))
        note["link"]["href"] = "/showcase/report/absent.pdf"
    edit_json(w.showcase / "story.json", relink)
    w.repin("story.json")
    return walked(w, "SV4", "absent.pdf")


@case("SV5", "", BASE, "story.json gives a class the alert role the page does not draw, re-pinned")
def sv5(w):
    def alert(story):
        roles = story["class_roles"]["roles"]
        roles[next(iter(roles))] = "alert"
    edit_json(w.showcase / "story.json", alert)
    w.repin("story.json")
    return walked(w, "SV5", "alert")


def walk_fails(w, *words) -> Check:
    """The served walk fails, some failed check naming every one of `words`."""
    rc, out = w.walk()
    doc = read_json(w.showcase / ".smoke" / "smoke.json") if (w.showcase / ".smoke" / "smoke.json").is_file() else {}
    said = " ".join(p for c in doc.get("checks") or [] if not c.get("ok") for p in c.get("problems") or [])
    missing = [x for x in words if x.lower() not in said.lower()]
    ok = rc == 1 and doc.get("status") == "fail" and not missing
    return Check(ok, f"exit {rc}, status {doc.get('status')}" + (f"; no failed check names {missing}" if missing else ""),
                 out)


@case("PG10", "PN8", BASE, "page.json's lane link sigmoid against the softmax network, re-pinned")
def pg10(w):
    edit_json(w.showcase / "page.json", lambda d: d["lane"].update(link="sigmoid"))
    w.repin("page.json")
    return walk_fails(w, "the page never became ready", "lane/link")


QPS_CHILD = r"""
import subprocess, sys
from pathlib import Path
from serve_probe import Service
bundle, page_smoke, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with Service(bundle, sys.executable, None) as svc:
    sys.exit(subprocess.call([sys.executable, page_smoke, "--base-url", svc.base, "--bundle", str(bundle),
                              "--lane", sys.argv[4], "--out", out, "--runs", "R-q"]))
"""


@case("PG12", "VS26", BASE, "a pseudo-locale build whose results view writes words outside t()")
def pg12(w):
    def seed(page):
        anchor = '$("res-ci-note").replaceChildren('
        edit_text(page / "js" / "views" / "results.js", anchor, anchor + '"Seeded words outside the table ", ')
    page = template_copy(w.root / "tpl", seed)
    w.stage(template=page, steps=("draft", "author", "resolve"))
    rc, out = w.build("--template", page, "--locale", "qps-ploc")
    if rc != 0:
        return Check(False, f"the pseudo-locale build exited {rc}; it must build (exit 0)", out)
    page_smoke = SKILLS / "vega" / "scripts" / "tests" / "page_smoke.py"
    rc, out = py(QPS_CHILD, w.bundle, page_smoke, w.root / "page-smoke.json", w.lane, cwd=w.work)
    return Expect(rc, out, 1, ("P-QPS", "res-ci-note", "Seeded"))


CASES = [v for v in list(globals().values()) if type(v).__name__ == "Case"]
