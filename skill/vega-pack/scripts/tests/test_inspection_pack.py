#!/usr/bin/env python3
"""
test_inspection_pack.py — the pack's inspection phase and the blueprint declaration fork (build
plan I09; D3 §13, §16 NI1–NI5; inspect-contract §2, §6.2, §7; bundle-contract §2.4, §7.7; R-G3).

    python3 -m pytest test_inspection_pack.py [-q] [-k "not packed"]
    python3 test_inspection_pack.py                          (without pytest)

Three parts.

**The logic** (fast, nothing packed): the served endpoints read off the route modules
`serve/app.py` composes (`{path:path}` read as `{path}`); the declaration against
`pipeline.json.serving.api.endpoints`, grouped by `--extends` target (`serving.demo` for the
core routes of an adapter forced over a batch blueprint, else `serving.api`; `serving.inspection`,
`serving.showcase`) with the blueprint's port names; the `/blueprint --extends` call in X1's form;
the fork's answer per mode — attended `FORK:`, unattended `(a)` with the skill's `--extends`,
unattended `(c)` without it (NI5: `api_declared: false`, `WARN: … not declared`); the WARN for
bets an unattended /vega-pack made; the record in the catalog's keys; `--no-inspect`'s two
refusals; the map entries; the refusal that names the upstream fix for a missing SVDD centre;
NI1's and NI2's refusals from a stand-in inspector.

**The real thing** (`packed`): B15's window spike fixture — a lagged multi-series forecast —
trained, evaluated by the REAL evaluate.py and packed by the REAL pack.py (`roundtrip_fixtures`,
`--serve --unattended`): `MANIFEST.serve.inspection` carries the catalog's keys with both
network digests, and the bundle's own `load_inspector` — the start-up path — re-verifies the pin
(`pinned: true`); a tampered pin is refused at start-up; the unattended fork recorded as the
installed blueprint skill allows (NI5 on the real bundle); `--no-inspect` ships no
`inspection/` and `enabled: false`; `--inspect-weights summary` withholds the raw tensors and
is the user's stance; a pack against the blueprint amended as `/blueprint --extends` amends it
records `api_declared: true` and says the bets were an unattended default. The negatives on a
copy of the packed bundle: NI1 (a reader removed from the rendered `registry.py`: exit 3, `no
inspection reader for block type`), NI2 (a probe that writes into the captured output: exit 3,
`does not reproduce the served answer`), NI3 (an inspection file without its map entry: the
conformance gate's C3, `has neither an adapter, a support nor an entrypoint`), NI4 (a second
public class in the Predictor's module: C14, `a public class that is not the Predictor adapter
class`, at `--size-severity error`). `VEGA_INSPECTION_WORK=<dir>` keeps the packed fixture there.

**L01** (`VEGA_INSPECTION_L01=<the chained acceptance's work dir>`): the orders campaign packed as
the lane packs it (`--serve`) records `serve.inspection`, and the digest is re-verified at
start-up; without the variable the test says what it needs and passes nothing off as run.
"""
from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
SKILLS = PACK.parent.parent
for _p in (str(PACK), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import api_declaration as ad  # noqa: E402
import apply_bundle_layer as abl  # noqa: E402
import inspection_pack as ip  # noqa: E402
import roundtrip_fixtures as rf  # noqa: E402
from pack_context import PackExit  # noqa: E402
from vega_pack_core import CATALOG  # noqa: E402

CONFORMANCE = SKILLS / "vega-forge" / "scripts" / "conformance.py"
PATCHED_BLUEPRINT = SKILLS / "patches" / "blueprint" / "SKILL.md"
KEYS = CATALOG["serve_record"]["inspection"]["keys"]
UNATTENDED_BY = "/vega-pack (vega:default)"


def _refusal(fn, *args) -> PackExit:
    try:
        fn(*args)
    except PackExit as exc:
        return exc
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not refuse")


def _env() -> dict:
    return dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


class _Skill:
    """`api_declaration.BLUEPRINT_SKILL` pointed at another file for one block."""

    def __init__(self, path):
        self.path, self.saved = Path(path), None

    def __enter__(self):
        self.saved, ad.BLUEPRINT_SKILL = ad.BLUEPRINT_SKILL, self.path
        return self

    def __exit__(self, *exc):
        ad.BLUEPRINT_SKILL = self.saved
        return False


# =========================================================================== the logic

ROUTES = {
    "routes_meta": ("core", (("GET", "/"), ("GET", "/console"), ("GET", "/health"))),
    "routes_predict": ("core", (("POST", "/v1/predict"),)),
    "routes_inspect": ("inspection", (("GET", "/v1/network"),
                                      ("GET", "/v1/network/tensor/{tensor_id}"),
                                      ("POST", "/v1/explain"))),
    "routes_showcase": ("showcase", (("GET", "/"), ("GET", "/console"),
                                     ("GET", "/showcase/{path:path}"), ("GET", "/v1/story"),
                                     ("GET", "/report.pdf"), ("GET", "/v1/showcase"))),
}
SLOT = {"GET /", "GET /console", "GET /showcase/{path}", "GET /v1/story", "GET /report.pdf",
        "GET /v1/showcase"}


def _serve(root: Path, modules=ROUTES, extra=("routes_unlisted",)) -> Path:
    """A serve/ of route modules with their literals; app.py composes all but `extra`."""
    serve = root / "serve"
    serve.mkdir(parents=True, exist_ok=True)
    names = tuple(modules) + ("routes_monitor",)             # listed, not shipped: skipped
    (serve / "app.py").write_text(f"ROUTE_MODULES = {names!r}\n", encoding="utf-8")
    for name in tuple(modules) + tuple(extra):
        capability, endpoints = modules.get(name, ("core", (("GET", "/nowhere"),)))
        (serve / f"{name}.py").write_text(
            f"ENDPOINTS = {endpoints!r}\nMODEL_ROUTES = ()\nOPEN_WHILE_UNREADY = ()\n"
            f"CAPABILITY = {capability!r}\n", encoding="utf-8")
    return serve


def _fork(**kw) -> ad.Fork:
    base = {"pipeline": "runs/p/pipeline.json", "ports": {"Predictor": "OrderScorer",
                                                          "Monitor": "LateWatch"}}
    return ad.Fork(**dict(base, **kw))


def test_the_served_endpoints_are_the_composed_modules_with_the_blueprints_spelling(tmp_path):
    served = ad.served_endpoints(_serve(tmp_path))
    texts = {(e.text, e.capability) for e in served}
    assert ("GET /showcase/{path}", "showcase") in texts
    assert not any("{path:path}" in e.path for e in served)
    assert ("GET /", "core") in texts and ("GET /", "showcase") in texts
    assert not any(e.path == "/nowhere" for e in served), "a module app.py does not compose"
    assert {e.calls for e in served if e.capability == "inspection"} == {"Predictor"}
    assert ad.normal_path("/v1/network/tensor/{tensor_id}") == "/v1/network/tensor/{tensor_id}"
    assert ad.served_endpoints(tmp_path / "absent") == ()


def test_the_slot_record_names_exactly_the_routes_the_showcase_module_serves(tmp_path):
    served = ad.served_endpoints(_serve(tmp_path))
    assert {e.text for e in served if e.capability == "showcase"} == SLOT
    ad.slot_routes(served)                                     # the catalog's six, {path} read
    fewer = tuple(e for e in served if e.text != "GET /v1/story")
    drift = _refusal(ad.slot_routes, fewer)
    assert drift.code == 3 and "showcase_slot.routes" in drift.lines[0]
    ad.slot_routes(tuple(e for e in served if e.capability != "showcase"))   # no slot served


def test_a_blueprint_that_declares_every_route_leaves_nothing_missing(tmp_path):
    served = ad.served_endpoints(_serve(tmp_path))
    pipeline = {"serving": {"api": {"endpoints": [
        {"method": e.method.lower(), "path": e.path.replace("{path}", "{path:path}")}
        for e in served]}}}
    decl = ad.declaration(served, pipeline)
    assert decl.declared and not decl.by_default
    assert _fork(unattended=True).answer(decl) == "declared"


def test_missing_routes_are_grouped_by_target_with_the_blueprints_port_names(tmp_path):
    served = ad.served_endpoints(_serve(tmp_path))
    pipeline = {"serving": {"api": {"endpoints": [{"method": "POST", "path": "/v1/predict"}]}}}
    decl = ad.declaration(served, pipeline)
    demo = ad.extends_entries(decl.missing, _fork(demo=True))
    assert list(demo) == ["serving.demo", "serving.inspection", "serving.showcase"]
    assert {e["path"] for e in demo["serving.demo"]} == {"/", "/console", "/health"}
    assert all(e["calls"] == "OrderScorer" for e in demo["serving.inspection"])
    assert all("calls" not in e for e in demo["serving.showcase"])
    assert {f"GET {e['path']}" for e in demo["serving.showcase"]} == SLOT
    online = ad.extends_entries(decl.missing, _fork(demo=False))
    assert list(online)[0] == "serving.api" and "serving.demo" not in online


def test_the_extends_call_is_x1s_form(tmp_path):
    served = ad.served_endpoints(_serve(tmp_path))
    decl = ad.declaration(served, {})
    targets = ad.extends_entries(decl.missing, _fork(demo=True))
    call = shlex.split(ad.extends_call(targets, _fork(demo=True, weights="summary")))
    assert call[:3] == ["/blueprint", "--extends",
                        "serving.demo,serving.inspection,serving.showcase"]
    assert json.loads(call[call.index("--entries") + 1]) == targets
    assert call[call.index("--weights") + 1] == "summary"
    assert call[call.index("--slot") + 1] == "showcase/"
    assert call[call.index("--by") + 1] == "/vega-pack" and "--report-only" not in call
    assert call[call.index("--pipeline") + 1] == "runs/p/pipeline.json"
    assert "--because" in call and call[call.index("--because") + 1]
    only = {"serving.inspection": targets["serving.inspection"]}
    unattended = shlex.split(ad.extends_call(only, _fork(unattended=True)))
    assert json.loads(unattended[unattended.index("--entries") + 1]) == only["serving.inspection"]
    assert unattended[-1] == "--report-only" and UNATTENDED_BY in unattended
    assert "--slot" not in unattended


def test_the_forks_answer_per_mode_ni5_logic(tmp_path):
    decl = ad.declaration(ad.served_endpoints(_serve(tmp_path)), {})
    assert _fork(unattended=False).answer(decl) == "open"
    assert _fork(unattended=True, extends_available=True).answer(decl) == "extend"
    assert _fork(unattended=True, extends_available=False).answer(decl) == "divergence"
    record = ad.declaration_record(decl, _fork(unattended=True))
    assert record["answer"] == "divergence" and record["source"] == "vega:default"
    assert record["extends"].startswith("/blueprint --extends ") and record["targets"]
    assert set(record["missing"]) == set(record["served"])


def test_bets_an_unattended_pack_made_are_named(tmp_path):
    served = ad.served_endpoints(_serve(tmp_path))
    pipeline = {"serving": {"api": {"endpoints": [
        dict(method=e.method, path=e.path, target="serving.inspection" if
             e.capability == "inspection" else "serving.api", bet="bet7" if
             e.capability == "inspection" else "bet2") for e in served]}},
        "extends": [{"target": "serving.inspection", "bet": "bet7", "by": UNATTENDED_BY},
                    {"target": "serving.api", "bet": "bet2", "by": "user"}]}
    decl = ad.declaration(served, pipeline)
    assert decl.declared
    assert {t for t, _g, _b in decl.by_default} == {"GET /v1/network", "POST /v1/explain",
                                                   "GET /v1/network/tensor/{tensor_id}"}
    assert {b for _t, _g, b in decl.by_default} == {"bet7"}


def test_extends_is_available_only_where_the_skill_documents_it(tmp_path):
    with_it, without = tmp_path / "with.md", tmp_path / "without.md"
    with_it.write_text("... /blueprint --extends serving.api | ...", encoding="utf-8")
    without.write_text("# blueprint\nno amendment here", encoding="utf-8")
    assert ad.extends_available(with_it) and not ad.extends_available(without)
    assert not ad.extends_available(tmp_path / "absent.md")
    assert ad.extends_available(PATCHED_BLUEPRINT), "X1's patch documents --extends"


def _ctx(tmp: Path, blueprint, **options):
    """A pack context as `declare` and the record read it (`blueprint` the pipeline.json)."""
    path = tmp / "pipeline.json"
    path.write_text(json.dumps(blueprint), encoding="utf-8")
    opts = dict(pipeline=str(path), unattended=False, inspect=True, inspect_weights="full")
    opts.update(options)
    ports = {"Predictor": "OrderScorer", "Monitor": "LateWatch"}
    return SimpleNamespace(
        paths=SimpleNamespace(root=tmp / "bundle", pkg=tmp / "bundle" / "pkg",
                              manifest=tmp / "bundle" / "MANIFEST.json"),
        options=SimpleNamespace(**opts), mode=options.get("mode", "batch"), markers=[],
        stances=[], spec={}, port=lambda role: {"name": ports.get(role)},
        pipeline=lambda: json.loads(path.read_text(encoding="utf-8")),
        campaign=SimpleNamespace(backend_name="pytorch"))


def test_ni5_unattended_without_the_skill_records_the_divergence(tmp_path):
    _serve(tmp_path / "bundle")
    ctx = _ctx(tmp_path, {"serving": {"mode": "batch"}}, unattended=True)
    with _Skill(tmp_path / "absent.md"):
        records = ad.declare(ctx)
    assert records["blueprint.conformance.api_declared"] is False
    record = records["blueprint.conformance.declaration"]
    assert record["answer"] == "divergence" and record["source"] == "vega:default"
    warns = [m for m in ctx.markers if m.startswith("WARN:")]
    assert len(warns) == 1 and "not declared" in warns[0], ctx.markers
    assert "api_declared: false" in warns[0]
    assert not [m for m in ctx.markers if m.startswith("FORK:")], "unattended never asks"


def test_unattended_with_the_skill_records_the_call_to_apply(tmp_path):
    _serve(tmp_path / "bundle")
    ctx = _ctx(tmp_path, {}, unattended=True)
    with _Skill(PATCHED_BLUEPRINT):
        records = ad.declare(ctx)
    record = records["blueprint.conformance.declaration"]
    assert records["blueprint.conformance.api_declared"] is False
    assert record["answer"] == "extend" and "--report-only" in record["extends"]
    assert UNATTENDED_BY in shlex.split(record["extends"])
    assert any(m.startswith("WARN:") and "Skill tool" in m for m in ctx.markers), ctx.markers


def test_attended_the_fork_is_asked_and_the_bundle_records_it_open(tmp_path):
    _serve(tmp_path / "bundle")
    ctx = _ctx(tmp_path, {})
    records = ad.declare(ctx)
    assert records["blueprint.conformance.declaration"]["answer"] == "open"
    forks = [m for m in ctx.markers if m.startswith("FORK:")]
    assert len(forks) == 1 and "(a) /blueprint --extends" in forks[0]
    assert "(b)" in forks[0] and "(c)" in forks[0]


def test_no_line_of_the_fork_reads_as_the_judgement_halfs_absence(tmp_path):
    """The bundle layer reads the judgement tool's word in the markers as the declaration of
    its absence; a fork line that named it would silence that gate (pack negative 'an
    unrecorded architect absence is refused')."""
    _serve(tmp_path / "bundle")
    lines = []
    for unattended, skill in ((False, PATCHED_BLUEPRINT), (True, PATCHED_BLUEPRINT),
                              (True, tmp_path / "absent.md")):
        ctx = _ctx(tmp_path, {"serving": {"inspection": {"weights": "summary"}}},
                   unattended=unattended)
        with _Skill(skill):
            ad.declare(ctx)
        lines += ctx.markers
    served = ad.served_endpoints(tmp_path / "bundle" / "serve")
    everything = {"serving": {"api": {"endpoints": [
        {"method": e.method, "path": e.path, "target": "serving.api", "bet": "bet3"}
        for e in served]}}, "extends": [{"target": "serving.api", "bet": "bet3",
                                          "by": UNATTENDED_BY}]}
    ctx = _ctx(tmp_path, everything, unattended=True)
    ad.declare(ctx)
    lines += ctx.markers
    assert len(lines) >= 5 and any(m.startswith("FORK:") for m in lines), lines
    assert not [m for m in lines if any(w.search(m) for w in abl._TOOL_WORD.values())], lines


def test_nothing_served_or_no_blueprint_is_no_record(tmp_path):
    (tmp_path / "bundle").mkdir()
    assert ad.declare(_ctx(tmp_path, {})) == {}
    _serve(tmp_path / "bundle")
    assert ad.declare(_ctx(tmp_path, {}, pipeline=None)) == {}


def test_a_blueprint_that_withholds_the_weights_is_held_to(tmp_path):
    _serve(tmp_path / "bundle")
    ctx = _ctx(tmp_path, {"serving": {"inspection": {"enabled": True, "weights": "summary",
                                                     "bet": "bet4"}}}, unattended=True)
    with _Skill(tmp_path / "absent.md"):
        ad.declare(ctx)
    assert any("weights summary" in m and "bet4" in m for m in ctx.markers), ctx.markers


def test_the_record_carries_the_catalogs_keys_in_order(tmp_path):
    ctx = _ctx(tmp_path, {})
    ctx.spec = {"blocks": [{"id": "B1", "type": "dense"},
                           {"id": "B2", "type": "residual", "params": {"inner": [
                               {"id": "B2a", "type": "layernorm"}, {"id": "B2b", "type": "dense"}]}},
                           {"id": "B3", "type": "parallel", "params": {"branches": [
                               [{"id": "p1", "type": "gru"}], [{"id": "p2", "type": "lstm"}]]}},
                           {"id": "B4", "type": "head"}]}
    opened = ip.Opened({"ready": True, "coverage": {"opaque": ["X"]}},
                       {"full": "sha256:a", "summary": "sha256:b"}, True)
    record = ip._record(True, ctx, ["serving/inspection/codec.py"], opened)
    assert list(record) == KEYS
    assert record["block_types"] == ["dense", "gru", "head", "layernorm", "lstm", "parallel",
                                     "residual"]
    assert record["network_digest"] == {"full": "sha256:a", "summary": "sha256:b"}
    assert record["opaque"] == ["X"] and record["chain"] is True
    off = ip._record(False, ctx, (), None)
    assert list(off) == KEYS and off["enabled"] is False and off["network_digest"] is None


class _Predictor:
    def __init__(self, refuses=True):
        self.refuses = refuses

    def inspector(self):
        if self.refuses:
            raise LookupError("this bundle ships without inspection")
        return object()


def test_no_inspect_is_a_bundle_without_inspection_and_a_stance(tmp_path):
    ctx = _ctx(tmp_path, {}, inspect=False, inspect_weights="summary")
    record = ip._without(ctx, _Predictor(), [], "serving")
    assert record["enabled"] is False and record["files"] == []
    assert any(s.startswith("user: --no-inspect") for s in ctx.stances)
    assert any("not applied" in m for m in ctx.markers), ctx.markers
    left = _refusal(ip._without, ctx, _Predictor(), ["serving/inspection/codec.py"], "serving")
    assert left.code == 3 and "--no-inspect" in left.lines[0]
    built = _refusal(ip._without, ctx, _Predictor(refuses=False), [], "serving")
    assert built.code == 3 and "LookupError" in built.lines[0]


def test_every_inspection_file_is_a_support_entry_of_the_map(tmp_path):
    ctx = _ctx(tmp_path, {})
    ctx.paths.pkg.mkdir(parents=True)
    bmap = ctx.paths.pkg / "blueprint-map.json"
    bmap.write_text(json.dumps({"entries": [{"file": "serving/inspection/codec.py",
                                             "kind": "support"}]}), encoding="utf-8")
    ip._mapped(ctx, ["serving/inspection/codec.py", "serving/inspection/graph.py"])
    doc = json.loads(bmap.read_text(encoding="utf-8"))
    added = [e for e in doc["entries"] if e["file"] == "serving/inspection/graph.py"]
    assert added and added[0]["kind"] == "support" and added[0]["used_by"] == ["Predictor"]
    bmap.write_text(json.dumps({"entries": [{"file": "serving/inspection/codec.py",
                                             "kind": "adapter", "role": "Predictor"}]}),
                    encoding="utf-8")
    wrong = _refusal(ip._mapped, ctx, ["serving/inspection/codec.py"])
    assert wrong.code == 3 and "kind: support" in wrong.lines[0]


class _Inspector:
    """A stand-in NetworkInspector whose opening ends as the templates' would."""

    def __init__(self, why=None):
        self.why, self.closed = why, False

    def open(self, _example):
        return SimpleNamespace(to_dict=lambda: {"ready": self.why is None, "why": self.why})


def test_ni1_ni2_and_the_svdd_centre_name_where_the_fix_goes():
    ni1 = _refusal(ip._opened, _Inspector("no inspection reader for block type 'gru'"), None,
                   "pytorch")
    assert ni1.code == 3 and ni1.lines[0] == "inspection: no inspection reader for block type 'gru'"
    assert "templates/serving/package/inspection/" in ni1.lines[1]
    assert "never in the bundle" in ni1.lines[1]
    ni2 = _refusal(ip._opened, _Inspector("the explanation does not reproduce the served "
                                          "answer: the captured head differs"), None, "pytorch")
    assert "does not reproduce the served answer" in ni2.lines[0]
    svdd = _refusal(ip._opened, _Inspector("the one-class centre was not persisted with the "
                                           "weights"), None, "pytorch")
    assert "/vega-train" in svdd.lines[1] and "svdd_center" in svdd.lines[1]
    assert ip._opened(_Inspector(), None, "pytorch")["ready"] is True


# =========================================================================== the real thing

_WORK: list = []
_PACKED: dict = {}


def _work() -> Path:
    if not _WORK:
        named = os.environ.get("VEGA_INSPECTION_WORK")
        if named:
            Path(named).mkdir(parents=True, exist_ok=True)
            _WORK.append(Path(named))
        else:
            made = Path(tempfile.mkdtemp(prefix="vega-inspection-"))
            atexit.register(shutil.rmtree, made, True)
            _WORK.append(made)
    return _WORK[0]


def _bundle_of(bundle_dir: Path):
    found = sorted(Path(bundle_dir).glob("*/MANIFEST.json"))
    return found[0].parent if found else None


def _repack(result, name, extra, pipeline=None):
    """The fixture's campaign packed again into bundle-<name> with more flags."""
    camp = Path(result.campaign)
    fixture = SimpleNamespace(shaped=result.shaped, pipeline=pipeline or result.pipeline)
    out = rf.pack(camp, fixture, camp.parent / f"bundle-{name}", extra)
    return SimpleNamespace(exit=out.returncode, output=out.stdout + out.stderr,
                           bundle=_bundle_of(camp.parent / f"bundle-{name}"))


def packed():
    """The window fixture made, trained, evaluated and packed (the harness, in a child process:
    the spike's emitter runs in the process that makes a fixture), and its re-packs."""
    if not _PACKED:
        proc = subprocess.run([sys.executable, rf.__file__, str(_work()), "--fixture", "window",
                               "--json"], capture_output=True, text=True, timeout=1800,
                              env=_env())
        try:
            result = rf.Packed.from_dict(json.loads(proc.stdout)[0])
        except (json.JSONDecodeError, IndexError) as exc:
            raise AssertionError(f"the harness failed (exit {proc.returncode})\n"
                                 f"{proc.stderr[-3000:]}") from exc
        assert result.pack_exit == 0 and result.bundle, result.pack_output[-3000:]
        _PACKED["base"] = result
        _PACKED["off"] = _repack(result, "off", ("--no-inspect",))
        _PACKED["summary"] = _repack(result, "summary", ("--inspect-weights", "summary"))
        amended = _amended(result)
        _PACKED["amended"] = _repack(result, "amended", (), pipeline=str(amended))
    return _PACKED


def _amended(result) -> Path:
    """The fixture's blueprint as `/blueprint --extends … --report-only --by "/vega-pack
    (vega:default)"` amends it (X1's patch): the recorded entries in serving.api.endpoints with
    their target and bet, one bet and one extends[] record per target, a new run directory."""
    record = result.manifest["blueprint"]["conformance"]["declaration"]
    base = Path(result.pipeline)
    doc = json.loads(base.read_text(encoding="utf-8"))
    digest = "sha256:" + __import__("hashlib").sha256(json.dumps(
        doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    bets, extends = list(doc.get("bets") or []), list(doc.get("extends") or [])
    endpoints = doc.setdefault("serving", {}).setdefault("api", {}).setdefault("endpoints", [])
    for target, entries in record["targets"].items():
        bet = f"bet{len(bets) + 1}"
        bets.append({"id": bet, "title": target, "resolution": "auto-resolved",
                     "chosen": f"{len(entries)} endpoint(s)"})
        have = {(e["method"], e["path"]) for e in endpoints}
        endpoints += [dict(e, target=target, bet=bet, source="[DESIGNER'S CHOICE — /vega-pack]")
                      for e in entries if (e["method"], e["path"]) not in have]
        extends.append({"target": target, "base": str(base), "base_digest": digest, "bet": bet,
                        "by": UNATTENDED_BY, "at": "2026-09-30T00:00:00Z"})
    doc["bets"], doc["extends"] = bets, extends
    out = base.parent / "blueprint-amended" / "pipeline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return out


def _run_py(code: str, *args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True,
                          text=True, timeout=900, env=_env(), cwd=cwd)


STARTUP = r"""
import json, sys
from pathlib import Path
bundle = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(bundle))
import predict
try:
    inspector = predict.load_inspector(bundle)
except LookupError as exc:
    print(json.dumps({"lookup": str(exc)})); sys.exit(0)
out = {"status": inspector.status()}
for mode in ("full", "summary"):
    try:
        out[mode] = json.loads(inspector.network_bytes(mode))["network_digest"]
    except LookupError as exc:
        out[mode] = "withheld: " + str(exc)
print(json.dumps(out))
"""

CHECK = r"""
import json, sys
from pathlib import Path
scripts, bundle = sys.argv[1], Path(sys.argv[2]).resolve()
sys.path.insert(0, scripts)
sys.path.insert(0, str(bundle))
import predict
import inspection_pack
from pack_context import PackExit
tolerance = json.loads((bundle / "MANIFEST.json").read_text())["roundtrip"]["tolerance"]["value"]
try:
    opened = inspection_pack.inspection_check(bundle, predict.load(bundle), tolerance)
except PackExit as exc:
    print(json.dumps({"code": exc.code, "lines": list(exc.lines)})); sys.exit(0)
print(json.dumps({"code": 0, "digests": opened.digests}))
"""


def _startup(bundle) -> dict:
    proc = _run_py(STARTUP, bundle)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _seeded(bundle, name) -> Path:
    copy = Path(bundle).parent.parent / f"seeded-{name}" / Path(bundle).name
    shutil.rmtree(copy.parent, ignore_errors=True)
    shutil.copytree(bundle, copy)
    return copy


def _inspection_dir(bundle) -> Path:
    manifest = json.loads((Path(bundle) / "MANIFEST.json").read_text(encoding="utf-8"))
    return Path(bundle) / manifest["package"] / \
        Path(manifest["serve"]["inspection"]["files"][0]).parent


def _checked(bundle) -> dict:
    proc = _run_py(CHECK, PACK, bundle)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_packed_serve_inspection_is_recorded_and_the_pin_re_verified_at_start_up():
    base = packed()["base"]
    manifest = base.manifest
    record = manifest["serve"]["inspection"]
    assert list(record) == KEYS and record["enabled"] is True and record["weights"] == "full"
    assert set(record["network_digest"]) == {"full", "summary"}
    assert record["self_test"]["ready"] is True and record["files"]
    pkg = Path(base.bundle) / manifest["package"]
    on_disk = sorted(p.relative_to(pkg).as_posix() for p in _inspection_dir(base.bundle)
                     .glob("*.py"))
    assert record["files"] == on_disk
    bmap = json.loads((pkg / "blueprint-map.json").read_text(encoding="utf-8"))
    kinds = {e["file"]: e.get("kind") for e in bmap["entries"] if e.get("file") in on_disk}
    assert kinds == {f: "support" for f in on_disk}
    assert "gru" in record["block_types"] and record["chain"] is False
    started = _startup(base.bundle)
    status = started["status"]
    assert status["ready"] is True and status["pinned"] is True, status
    assert status["network_digest"] == record["network_digest"]["full"]
    assert started["summary"] == record["network_digest"]["summary"]
    assert "does not render yet" not in base.pack_output
    assert "OK: inspection: the inspector opened" in base.pack_output
    mine = [m for m in base.pack_output.splitlines()
            if m.startswith(("OK: inspection", "WARN: blueprint", "FORK: blueprint"))]
    assert mine and not [m for m in mine if any(w.search(m) for w in abl._TOOL_WORD.values())], mine


def test_packed_a_tampered_pin_is_refused_at_start_up():
    bundle = _seeded(packed()["base"].bundle, "pin")
    path = bundle / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["serve"]["inspection"]["network_digest"]["full"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status = _startup(bundle)["status"]
    assert status["ready"] is False and "not the one packed" in status["why"], status


def test_packed_ni5_the_unattended_fork_as_the_installed_skill_allows():
    base = packed()["base"]
    conformance = base.manifest["blueprint"]["conformance"]
    record = conformance["declaration"]
    assert conformance["api_declared"] is False and conformance["mechanical"] == "pass"
    assert conformance["architect"] is None
    served = {e for e in base.manifest["serve"]["endpoints"]}
    assert set(record["served"]) == {ad.normal_path(e) for e in served}
    want = "extend" if ad.extends_available() else "divergence"
    assert record["answer"] == want and record["source"] == "vega:default"
    markers = json.loads((Path(base.bundle) / "markers.json").read_text())["markers"]
    lines = [m for m in markers if m.startswith("WARN: blueprint declaration")]
    assert len(lines) == 1, markers
    assert ("not declared" in lines[0]) and (want != "divergence" or "absent" in lines[0])


def test_packed_no_inspect_ships_no_inspection_and_says_so():
    off = packed()["off"]
    assert off.exit == 0 and off.bundle, off.output[-3000:]
    manifest = json.loads((off.bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    record = manifest["serve"]["inspection"]
    assert record["enabled"] is False and record["files"] == [] and record["network_digest"] is None
    pkg = off.bundle / manifest["package"]
    assert not list(pkg.rglob("inspection")), "--no-inspect ships no inspection/ at all"
    assert not [e for e in manifest["serve"]["endpoints"] if "/v1/network" in e or "explain" in e]
    assert any(s.startswith("user: --no-inspect") for s in manifest["stances"])
    assert "lookup" in _startup(off.bundle)


def test_packed_inspect_weights_summary_withholds_the_tensors_and_is_a_stance():
    summary = packed()["summary"]
    assert summary.exit == 0 and summary.bundle, summary.output[-3000:]
    manifest = json.loads((summary.bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    record = manifest["serve"]["inspection"]
    assert record["weights"] == "summary" and record["enabled"] is True
    assert any("--inspect-weights summary" in s for s in manifest["stances"])
    started = _startup(summary.bundle)
    assert started["status"]["pinned"] is True and started["status"]["weights"] == "summary"
    assert started["full"].startswith("withheld"), started
    assert started["summary"] == record["network_digest"]["summary"]
    base = packed()["base"].manifest["serve"]["inspection"]["network_digest"]
    assert record["network_digest"] == base, "one network, the same digests in both modes"


def test_packed_against_the_amended_blueprint_every_endpoint_is_declared():
    amended = packed()["amended"]
    assert amended.exit == 0 and amended.bundle, amended.output[-3000:]
    manifest = json.loads((amended.bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    conformance = manifest["blueprint"]["conformance"]
    assert conformance["api_declared"] is True and conformance["mechanical"] == "pass"
    assert conformance["declaration"]["answer"] == "declared"
    assert conformance["declaration"]["by_default"]
    assert "WARN: blueprint extended with" in amended.output


def test_packed_the_check_passes_on_the_bundle_as_shipped():
    """The control of NI1 and NI2: the same check over the unseeded bundle, as its final
    manifest pins it, is ready with the pinned digests."""
    base = packed()["base"]
    out = _checked(base.bundle)
    assert out["code"] == 0, out
    assert out["digests"] == base.manifest["serve"]["inspection"]["network_digest"]


def test_packed_ni1_a_reader_removed_from_the_registry_stops_the_check():
    bundle = _seeded(packed()["base"].bundle, "ni1")
    registry = _inspection_dir(bundle) / "registry.py"
    text = registry.read_text(encoding="utf-8")
    seeded = text.replace("for r in (*CORE, *GROUP, *SEQ, *GRAPH)})",
                          "for r in (*CORE, *GROUP, *SEQ, *GRAPH) if r.TYPE != 'head'})")
    assert seeded != text, "the registry's READERS line moved; update the seed"
    registry.write_text(seeded, encoding="utf-8")
    out = _checked(bundle)
    assert out["code"] == 3, out
    assert "no inspection reader for block type 'head'" in out["lines"][0]


def test_packed_ni2_a_probe_writing_into_the_captured_output_stops_the_check():
    bundle = _seeded(packed()["base"].bundle, "ni2")
    probes = _inspection_dir(bundle) / "probes.py"
    text = probes.read_text(encoding="utf-8")
    # the dense/head probe's check, which every explanation runs (summary detail included)
    anchor = "    def recompute(self, cap, node):\n        x, z, y, _ = self._parts(cap, node)\n"
    assert text.count(anchor) == 1, "the dense/head probe moved; update the seed"
    probes.write_text(text.replace(anchor, anchor + "        y.data.mul_(2.0)\n"),
                      encoding="utf-8")
    out = _checked(bundle)
    assert out["code"] == 3, out
    assert "does not reproduce the served answer" in out["lines"][0]


def _conformance(bundle, pipeline, *extra) -> subprocess.CompletedProcess:
    manifest = json.loads((Path(bundle) / "MANIFEST.json").read_text(encoding="utf-8"))
    pkg = Path(bundle) / manifest["package"]
    return subprocess.run([sys.executable, str(CONFORMANCE), "--package", str(pkg), "--pipeline",
                           str(pipeline), "--map", str(pkg / "blueprint-map.json"), "--bundle",
                           *extra], capture_output=True, text=True, timeout=600, env=_env())


def test_packed_ni3_an_inspection_file_without_its_map_entry_fails_c3():
    base = packed()["base"]
    bundle = _seeded(base.bundle, "ni3")
    assert _conformance(bundle, base.pipeline).returncode == 0
    pkg = bundle / base.manifest["package"]
    path = pkg / "blueprint-map.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    gone = base.manifest["serve"]["inspection"]["files"][1]
    doc["entries"] = [e for e in doc["entries"] if e.get("file") != gone]
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    proc = _conformance(bundle, base.pipeline)
    assert proc.returncode != 0, proc.stdout[-2000:]
    assert f"{gone} has neither an adapter, a support nor an entrypoint" in proc.stdout
    markers = json.loads(_conformance(bundle, base.pipeline, "--json").stdout)["markers"]
    rules = {m.get("rule") for m in markers if str(m.get("level")).upper() == "ERROR"}
    assert rules == {"C3"}, markers


def test_packed_ni4_a_second_public_class_beside_the_predictor_fails_c14():
    base = packed()["base"]
    bundle = _seeded(base.bundle, "ni4")
    predictor = _inspection_dir(bundle).parent / "predict.py"
    with predictor.open("a", encoding="utf-8") as fh:
        fh.write("\n\nclass NetworkView:\n    \"\"\"v2.2's second public class.\"\"\"\n")
    needle = "NetworkView — a public class that is not the Predictor adapter class"
    clean = _conformance(base.bundle, base.pipeline, "--size-severity", "error")
    assert needle not in clean.stdout
    seeded = _conformance(bundle, base.pipeline, "--size-severity", "error")
    assert seeded.returncode != 0 and "ERROR: C14 " in seeded.stdout, seeded.stdout[-2000:]
    assert needle in seeded.stdout, seeded.stdout[-2000:]
    # the pack's own gate reads C11–C14 at `warn` until G2 flips the default: said, not hidden
    assert needle in _conformance(bundle, base.pipeline).stdout


# =========================================================================== L01

L01_CAMPAIGN = "vega-runs/2026-09-05T11-03Z"
L01_PIPELINE = "blueprint-runs/2026-09-05T10-12Z/pipeline.json"
L01_MODEL = "orders.domain.analysis.shaped.vega.train.eval.html"


def test_l01_pack_records_serve_inspection_and_the_pin_holds_at_start_up():
    work = os.environ.get("VEGA_INSPECTION_L01")
    if not work:
        _skip("L01 not packed: VEGA_INSPECTION_L01 names no work dir (the chained acceptance's: "
              "vega-pack/scripts/tests/acceptance.sh <dir>)")
    work = Path(work).resolve()
    out_dir = work / "bundle-inspection"
    shutil.rmtree(out_dir, ignore_errors=True)
    proc = subprocess.run([sys.executable, str(PACK / "pack.py"), "--campaign", L01_CAMPAIGN,
                           "--shaped", "shaped", "--bundle-dir", str(out_dir), "--pipeline",
                           L01_PIPELINE, "--model", L01_MODEL, "--ship-anyway",
                           "the lane packs every verdict", "--serve"], cwd=work,
                          capture_output=True, text=True, timeout=1800, env=_env())
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    bundle = _bundle_of(out_dir)
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    record = manifest["serve"]["inspection"]
    assert list(record) == KEYS and record["enabled"] and record["self_test"]["ready"]
    status = _startup(bundle)["status"]
    assert status["ready"] and status["pinned"], status
    assert status["network_digest"] == record["network_digest"]["full"]
    assert manifest["blueprint"]["conformance"]["api_declared"] is False
    assert "FORK: blueprint declaration" in proc.stdout, "attended: the fork is asked"


class _Skipped(Exception):
    """A test that cannot run here, said so (pytest's own skip under pytest)."""


def _skip(why: str) -> None:
    try:
        import pytest
    except ImportError:
        raise _Skipped(why) from None
    pytest.skip(why)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = skipped = 0
    for name, fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                fn(Path(tempfile.mkdtemp(prefix="insp-")))
            else:
                fn()
            print(f"  ok    {name}")
        except _Skipped as exc:
            skipped += 1
            print(f"  skip  {name} — {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {name} — {exc}")
    print(f"\n{len(tests) - failed - skipped} of {len(tests)} passed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
