"""
smoke_page_checks — the page driver's checks (smoke_page.py; BUILD-PLAN W15, D4 §17.5, D6 §10.4):
each a function of (result, served) returning its problems, read off the probe's output
(vega/page-probe@1), what the browser heard and what the fixture server could not answer.

    CHECKS                 ((id, check), …) in the order smoke_page reports them
    judge(result, served)  [{id, ok, problems}] — a page that never became ready is judged by
                           P-PROBE and P-ERR alone

`result` is {probe, console: [{kind, text, url}], responses: [(url, status)], misses}; `served`
carries the lane expected (`expect`), the failed loads the page may meet (`allowed`: path
prefixes it may meet answered 404, and only 404 — a favicon, the monitor of a bundle shipping
none), the service's own documents the weights map is held to (`documents()`: /v1/schema and the
showcase's attribution) and `strict` (an unvalidated probe output fails P-PROBE). The checks'
table is smoke_page.py's docstring.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE_SCHEMA_FILE = HERE.parent / "references" / "schemas" / "page-probe-1.schema.json"
SPHERE_TOL = 1e-4
FETCH_PREFIXES = ("health", "v1/", "showcase/")
ALWAYS_JUDGED = ("P-PROBE", "P-ERR")
NO_VALIDATOR = "jsonschema is not installed: the probe output was not validated against page-probe-1"
_warned = []


def c_probe(r, s) -> list[str]:
    p, out = r["probe"], []
    if not p.get("ready"):
        out.append("the page never became ready")
    if p.get("error"):
        out.append(f"the probe failed: {p['error']}")
    out += schema_problems(p, getattr(s, "strict", False))
    return out


def schema_problems(p: dict, strict: bool = False) -> list[str]:
    """The probe output's breaches of page-probe-1. Without jsonschema the output cannot be
    validated: a problem under `strict`, else one `WARN:` line per run."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        if strict:
            return [NO_VALIDATOR]
        if not _warned:
            _warned.append(True)
            print(f"WARN: {NO_VALIDATOR}", file=sys.stderr)
        return []
    schema = json.loads(PROBE_SCHEMA_FILE.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(p), key=lambda e: list(e.path))
    return [f"the probe output breaks page-probe-1 at {'/'.join(map(str, e.path)) or '(root)'}: {e.message[:200]}"
            for e in errors[:5]]


def c_err(r, s) -> list[str]:
    p = r["probe"]
    out = [f"page error: {e}" for e in p.get("page_errors") or []]
    out += [f"page error after the walk: {e}" for e in p.get("late_errors") or []]
    out += [f"browser: {e['text']}" for e in r["console"] if not _allowed(e.get("url"), e["text"], s.allowed)]
    for url, status in r["responses"]:
        path = "/" + url.split("/", 3)[3].split("?")[0] if url.count("/") >= 3 else url
        if status >= 400 and not (status == 404 and any(path.startswith(a) for a in s.allowed)):
            out.append(f"HTTP {status} for {url}")
    out += [f"the fixture holds no answer: {m['method']} {m['path']}: {m['why']}" for m in r["misses"]]
    return out


def _allowed(url: str | None, text: str, allowed: tuple) -> bool:
    """A failed load the page expects, answered 404 (a favicon; the monitor of a bundle shipping
    none): the browser's line names the status."""
    path = "/" + url.split("/", 3)[3].split("?")[0] if url and url.count("/") >= 3 else ""
    said = re.search(r"\bstatus of (\d{3})\b", text)
    return bool(path) and (said.group(1) == "404" if said else "404" in text) and any(path.startswith(a) for a in allowed)


def c_vis(r, s) -> list[str]:
    p, out = r["probe"], []
    for vid, v in (p.get("views") or {}).items():
        if v.get("off"):
            continue
        if not (v.get("reached") and v.get("rendered")):
            out.append(f"view {vid}: reached {v.get('reached')}, rendered {v.get('rendered')}")
    return out


def c_txt(r, s) -> list[str]:
    p = r["probe"]
    out = [f"unresolved placeholder {x}" for x in p.get("placeholders_unresolved") or []]
    out += [f"missing strings key {k}" for k in p.get("missing_keys") or []]
    out += [f"view {vid} shows {v['bad_text']}" for vid, v in (p.get("views") or {}).items() if v.get("bad_text")]
    out += [f"view {vid} cuts off a value at this width: {v['cut']}" for vid, v in (p.get("views") or {}).items() if v.get("cut")]
    return out + verdict_problems(p)


def verdict_problems(p: dict) -> list[str]:
    """The story's verdict (earned, not earned, beaten, not assessable) said on the results view in
    its own state: a story that records one and a view that hides it, or says another, is a
    problem (gate C: a beaten verdict reached the page only inside the collapsed limits)."""
    v, views = p.get("verdict") or {}, p.get("views") or {}
    want = v.get("expected")
    if not want or not (views.get("results") or {}).get("reached"):
        return []
    if not v.get("shown"):
        return [f"the results view does not say the verdict ({want})"]
    return [] if v.get("state") == want else [f"the results view says the verdict {v.get('state')!r}, the story records {want!r}"]


def c_lane(r, s) -> list[str]:
    lane = r["probe"].get("lane") or {}
    if not lane.get("ok"):
        return [f"the lane was refused: {lane.get('mismatch')}"]
    return [f"lane {k} is {lane.get(k)!r}, expected {v!r}" for k, v in s.expect.items()
            if k in ("view_key", "link", "shape", "level") and lane.get(k) != v]


def c_motif(r, s) -> list[str]:
    m = r["probe"].get("motif") or {}
    return [] if m.get("id") and m.get("id") == m.get("data_motif") else [f"motif {m.get('id')!r} drawn as {m.get('data_motif')!r}"]


def c_role(r, s) -> list[str]:
    pal, motif = r["probe"].get("palette") or {}, r["probe"].get("motif") or {}
    if pal.get("alert_roles"):
        return []
    out = [f"{pal.get('role_alert_elements')} element(s) in the alert role with no alert role"] if pal.get("role_alert_elements") else []
    return out + (["the field drew an alert-tinted mark with no alert role"] if motif.get("alert_marks") else [])


def c_net(r, s) -> list[str]:
    net = r["probe"].get("network") or {}
    if not net.get("available"):
        return [f"the network view read no network: {net.get('why')}"]
    out = [f"the drawing warned: {w}" for w in net.get("warnings") or []]
    if net.get("detail_kinds") is None:
        out.append("the probe could not read which stage kinds draw an every-weight level")
    elif net["detail_kinds"] and "detail" not in (net.get("modes") or {}):
        out.append(f"the stage kind(s) {', '.join(net['detail_kinds'])} draw an every-weight level, "
                   "but the page offers no switch to it")
    for mode, d in (net.get("modes") or {}).items():
        if d is None:
            out.append(f"the {mode} drawing was not read")
        elif d["drawn"] == "chain":
            out += [] if d["ink"] > 0 else [f"the chain drawn whole ({mode}) has no ink"]
        else:
            out += [f"segment {i} ({x['kind']} {x['stage']}) has no ink ({mode})" for i, x in enumerate(d["segments"]) if x["pixels"] <= 0]
    return out


def c_axis(r, s) -> list[str]:
    """The weights map read on real rows has the lane's weights axis for columns (D4 §6.4)."""
    w = (r["probe"].get("sub_views") or {}).get("weights") or {}
    if not w.get("rendered"):
        return []                                   # P-VIS judges a disclosure that did not render
    docs = s.documents() if callable(getattr(s, "documents", None)) else {}
    cols, attribution = w.get("columns") or {}, docs.get("attribution")
    real = cols.get("real")
    if attribution is None:
        return [] if real is None else [f"the weights map reads real rows {real}, but the showcase ships no attribution"]
    if real is None:
        return ["the weights map offers no reading on real rows, though the showcase ships its attribution"]
    want, out = expected_axis((r["probe"].get("lane") or {}).get("view_key"), docs.get("schema") or {}, attribution)
    if want is not None and list(real) != want:
        out.append(f"the weights map's columns on real rows are {real}, expected {want}")
    return out


def expected_axis(view_key: str | None, schema: dict, attribution: dict) -> tuple[list | None, list[str]]:
    """(the columns the weights map must show on real rows, problems of the axis itself): the
    binary sides (negative, positive), the output's classes, labels or ordinal levels, a forecast's
    steps ahead, a value's bins of the target's range (ascending, each starting where the one
    before ends) or the attribution's strata of a reconstruction or anomaly reading (each an
    output class when the output names its classes)."""
    out = schema.get("output") or {}
    classes = [str(c) for c in out.get("classes") or ()]
    axis = attribution.get("axis") or {}
    names = [str(c.get("name")) for c in axis.get("columns") or ()]
    if view_key == "binary":
        if len(classes) != 2:
            return None, [f"a binary output names {len(classes)} classes"]
        pos = str(out["positive"]) if out.get("positive") is not None else classes[1]
        return [c for c in classes if c != pos] + [pos], []
    if view_key in ("classes", "labels", "ordinal"):
        return classes, ([] if classes else [f"the {view_key} output names no classes"])
    if view_key == "horizon":
        return [f"h{int(h)}" for h in out.get("horizons") or ()], []
    if view_key == "value":
        return names, _bins(axis)
    if view_key in ("anomaly", "reconstruction"):
        foreign = [n for n in names if classes and n not in classes]
        return names, [f"the attribution's strata {foreign} are not the output's classes {classes}"] if foreign else []
    return None, []


def _bins(axis: dict) -> list[str]:
    cols = axis.get("columns") or []
    if axis.get("kind") != "bin" or len(cols) < 2:
        return [f"a value's weights axis is not two or more bins: {axis.get('kind')} × {len(cols)}"]
    spans = [c.get("range") for c in cols]
    if not all(isinstance(r, list) and len(r) == 2 for r in spans):
        return [f"a bin of the value's axis has no range: {spans}"]
    out = [f"bin {k} is empty or reversed: {r}" for k, r in enumerate(spans)
           if r[0] is not None and r[1] is not None and not r[0] < r[1]]
    for k in range(1, len(spans)):
        a, b = spans[k - 1][1], spans[k][0]
        if a is None or b is None or not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
            out.append(f"bin {k} does not start where bin {k - 1} ends: {spans[k - 1]} then {spans[k]}")
    return out


def c_step(r, s) -> list[str]:
    p = r["probe"]
    st, ex, sub = p.get("stepper") or {}, p.get("explain") or {}, (p.get("sub_views") or {}).get("stepper") or {}
    out = [f"explain failed: {e}" for e in ex.get("errors") or []]
    if not sub.get("presets"):
        out.append("no preset was walked")
    out += [f"stepper: {f}" for f in st.get("failed") or []]
    out += [f"explainer fell back to the generic one: {f}" for f in st.get("fallbacks") or []]
    return out


def c_latent(r, s) -> list[str]:
    lat = r["probe"].get("latent") or {}
    net, tried = lat.get("network"), lat.get("try")
    if not net:
        return []
    fam = net.get("family")
    if fam in ("autoencoder", "pca-floor"):
        return _waist(net)
    if net.get("bottleneck") and net["bottleneck"].get("kind") == "latent":
        return _code(net, tried)
    if net.get("sphere"):
        return _sphere(net, tried)
    return []


def _drawn(reading: dict | None, canvas: str) -> dict:
    return ((reading or {}).get("canvases") or {}).get(canvas) or {}


def _waist(net: dict) -> list[str]:
    b, d = net.get("bottleneck"), _drawn(net, "network")
    if not b or b.get("kind") != "dense":
        return [f"no dense waist: {b}"]
    w = d.get("waist") or {}
    return ([] if w.get("stage") == b["stage"] else [f"the waist drawn is {w.get('stage')}, not {b['stage']}"]) + \
        ([] if w.get("pixels", 0) > 0 else ["the waist column has no ink"])


def _code(net: dict, tried: dict | None) -> list[str]:
    z, out = net["bottleneck"]["units"], []
    drawn = _drawn(net, "network").get("latent") or []
    if len(drawn) != 1 or drawn[0]["pixels"] <= 0 or drawn[0]["z"] != z:
        out.append(f"the latent code drawn: {drawn}")
    a = (_drawn(tried, "try").get("latent") or [None])[0]
    if not a or not a.get("answered"):
        return out + ["no answer's latent code was drawn"]
    if len(a.get("mu") or []) != z or len(a.get("sigma") or []) != z or min(a["sigma"]) <= 0:
        out.append(f"an answer's μ and σ are not {z} numbers with σ > 0")
    return out + ([] if isinstance(a.get("kl"), (int, float)) and a["kl"] >= 0 else [f"an answer's KL is {a.get('kl')}"])


def _sphere(net: dict, tried: dict | None) -> list[str]:
    sp = net["sphere"]
    if not sp.get("centre") or len(sp["centre"]) != sp.get("units"):
        return [f"the head exports no centre as long as its output: {sp.get('units')}"]
    out = []
    for name, reading in (("network", net), ("try", tried)):
        d = _drawn(reading, name).get("sphere")
        if not d or d.get("centre") != sp["units"] or d.get("pixels", 0) <= 0:
            out.append(f"the {name} canvas's sphere: {d}")
    ans = (_drawn(tried, "try").get("sphere") or {}).get("answer")
    if not ans or ans.get("network") is None:
        return out + ["no answered distance to recompute"]
    return out + ([] if ans["gap"] <= SPHERE_TOL * max(1.0, abs(ans["network"])) else [f"distance {ans['page']} vs the service's {ans['network']}"])


def c_many(r, s) -> list[str]:
    m = (r["probe"].get("sub_views") or {}).get("many")
    if m is None or (r["probe"].get("views") or {}).get("many", {}).get("off"):
        return []
    if m.get("error"):
        return [f"many at once: {m.get('error')}"]
    out = [] if m.get("rows") else ["many at once answered no row"]
    read = [f for f in r["probe"].get("fetched") or [] if str(f.get("url", "")).startswith("v1/monitor") and f.get("status") == 200]
    if read and not m.get("monitor"):
        out.append(f"the service answered the drift monitor, but the page shows {m.get('monitor')} reading(s)")
    return out


def c_code(r, s) -> list[str]:
    c, walk = (r["probe"].get("sub_views") or {}).get("code") or {}, (r["probe"].get("extensions") or {}).get("codeWalk") or {}
    out = [] if walk.get("read") else [f"the code room did not read its walk: {walk.get('error')}"]
    out += [f"walk step {m['id']} left out: {m['why']}" for m in walk.get("missing") or []]
    return out + ([f"{c['broken']} walk step(s) shown broken"] if c.get("broken") else []) + ([] if c.get("walk") else ["no walk step"])


def c_fetch(r, s) -> list[str]:
    return [f"fetched outside health, v1/ and showcase/: {f['url']}" for f in r["probe"].get("fetched") or []
            if not str(f.get("url", "")).startswith(FETCH_PREFIXES)]


def c_theme(r, s) -> list[str]:
    t = r["probe"].get("theme_walk") or {}
    ok = t.get("first") and t.get("flipped") and t["first"] != t["flipped"] and t.get("back") == t["first"]
    return [] if ok else [f"the theme control: {t}"]


def c_links(r, s) -> list[str]:
    return [f"report link {l['href']} of {l['step']} answered {l['status']}" for l in r["probe"].get("links") or [] if l.get("status") != 200]


CHECKS = (("P-PROBE", c_probe), ("P-ERR", c_err), ("P-VIS", c_vis), ("P-TXT", c_txt), ("P-LANE", c_lane),
          ("P-MOTIF", c_motif), ("P-ROLE", c_role), ("P-NET", c_net), ("P-AXIS", c_axis), ("P-STEP", c_step),
          ("P-LATENT", c_latent),
          ("P-MANY", c_many), ("P-CODE", c_code), ("P-FETCH", c_fetch), ("P-THEME", c_theme), ("P-LINKS", c_links))


def judge(result: dict, served) -> list[dict]:
    out = []
    for cid, fn in CHECKS:
        problems = fn(result, served) if result["probe"].get("ready") or cid in ALWAYS_JUDGED else []
        out.append({"id": cid, "ok": not problems, "problems": problems})
    return out
