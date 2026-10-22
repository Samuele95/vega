"""
api_declaration — the blueprint declaration fork: the endpoints a bundle serves, looked up in
the blueprint it was packed against (build plan I09, R-D1, R-G3; D3 §13.4; bundle-contract §7).

    api_declaration.declare(ctx) -> {"blueprint.conformance.api_declared": bool,
                                     "blueprint.conformance.declaration": {…}} or {}
    served_endpoints(serve_dir) -> (Endpoint, …)     what the route modules serve
    slot_routes(served)                              the slot record names what it serves
    declaration(served, pipeline) -> Declaration     what the blueprint lacks
    extends_entries(missing, fork) / extends_call(targets, fork)   X1's `--extends` form

A pipeline blueprint is gated before any code exists, so the surface a later stage adds — the
inspection routes, the showcase slot, a demonstration service packed with `--serve` over a batch
or stream blueprint — was never a bet, and `/architect --baseline` on the bundle would read it as
a divergence. Every endpoint of the route modules `serve/app.py` composes (their `ENDPOINTS`,
what MANIFEST.serve.endpoints lists) is looked up by method and path in
`pipeline.json.serving.api.endpoints` — `{path:path}` read as `{path}`: a converter is the
framework's, not the API's.

- **All declared:** `api_declared: true`; when the bets that declared some were made by an
  unattended /vega-pack (`extends[].by` carries `vega:default`), `WARN: blueprint extended with
  <n> serving endpoints (…)` — bets nobody gated are said out loud.
- **Some missing:** grouped by the `/blueprint --extends` target each belongs to —
  `serving.inspection`, `serving.showcase`, and for the `core` / `code` routes `serving.demo`
  when the online adapter was forced over a batch or stream blueprint, else `serving.api` — as
  `{method, path, capability, calls: <the blueprint's port name>}` entries, and recorded with
  the call that declares them. Then: attended, `FORK:` (a) that call, (b) pack without them
  (`--no-inspect` drops the inspection routes, `--no-serve` the service), (c) ship with the
  divergence recorded — which this bundle is until the answer comes back; unattended, (a) when
  the installed blueprint skill documents `--extends` (the skill applies the recorded call
  through the Skill tool, `--report-only --by "/vega-pack (vega:default)"`, and packs again
  against the amended run), else (c) with `WARN: … not declared`. `api_declared` is false either
  way until a pack against a blueprint that declares them.

A bundle that serves no HTTP endpoint (batch, or a stream consumer) has nothing to declare, and
a pack without `--pipeline` has no blueprint to declare against (the conformance gate says so):
no record then. `blueprint.conformance.architect` stays null: the judgement half runs through the
Skill tool after pack.py.
"""
from __future__ import annotations

import ast
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import serving_templates as st
from pack_context import SKILLS, PackExit, out_ok, out_warn
from vega_pack_core import CATALOG

__all__ = ["BLUEPRINT_SKILL", "Declaration", "Endpoint", "Fork", "declaration",
           "declaration_record", "declare", "extends_available", "extends_call",
           "extends_entries", "normal_path", "served_endpoints", "slot_routes"]

BLUEPRINT_SKILL = SKILLS / "blueprint" / "SKILL.md"
EXTENDS_FORM = re.compile(r"/blueprint --extends\b")
SLOT = CATALOG["serve_record"]["showcase_slot"]["dir"].rstrip("/") + "/"
# the port each route module's endpoints call, as a role (the blueprint's port name replaces it)
CALLS = {"routes_predict": "Predictor", "routes_monitor": "Monitor",
         "routes_inspect": "Predictor"}
TARGET_OF = {"inspection": "serving.inspection", "showcase": "serving.showcase"}
TARGET_ORDER = ("serving.demo", "serving.api", "serving.inspection", "serving.showcase")
WORDS = {"serving.demo": "demonstration service", "serving.api": "service api",
         "serving.inspection": "inspection", "serving.showcase": "showcase slot"}
UNATTENDED_BY = "/vega-pack (vega:default)"
BECAUSE = "/vega-pack serves them from the bundle's route modules (serve/routes_*.py)"
SOURCES = {"declared": "blueprint:serving.api", "open": "FORK: open",
           "extend": "vega:default", "divergence": "vega:default"}
NOTE = ("blueprint.conformance.architect is null here: /architect --report-only --baseline "
        "is the skill's to run through the Skill tool after pack.py, and the bundle layer "
        "records its result")


@dataclass(frozen=True)
class Endpoint:
    """One served route: `path` spelled as the blueprint spells it (`{path}`, not `{path:path}`),
    the capability of the module that serves it and the role its handler calls, if any."""

    method: str
    path: str
    capability: str
    calls: str | None = None

    @property
    def key(self) -> tuple:
        return (self.method, self.path)

    @property
    def text(self) -> str:
        return f"{self.method} {self.path}"


@dataclass(frozen=True)
class Declaration:
    """The served endpoints, those the blueprint lacks, and the declared ones whose bets an
    unattended /vega-pack made (`(endpoint text, target, bet)`)."""

    served: tuple
    missing: tuple
    by_default: tuple = ()

    @property
    def declared(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class Fork:
    """What the fork's answer and the `--extends` call depend on, read once from the context."""

    pipeline: str
    weights: str = "full"
    demo: bool = False
    unattended: bool = False
    extends_available: bool = False
    because: str = BECAUSE
    ports: dict = field(default_factory=dict)

    @staticmethod
    def of(ctx) -> "Fork":
        """The served adapter is the online one; over a batch or stream blueprint it was forced
        by `--serve` (R-D10), and its core routes are a demonstration service."""
        demo = ctx.mode != "online"
        because = BECAUSE + (f"; pack --serve rendered an online demonstration adapter over a "
                             f"{ctx.mode} blueprint" if demo else "")
        return Fork(str(ctx.options.pipeline), ctx.options.inspect_weights, demo,
                    bool(ctx.options.unattended), extends_available(), because,
                    {role: ctx.port(role).get("name") or role for role in set(CALLS.values())})

    def answer(self, decl: Declaration) -> str:
        """`declared`; attended `open` (the FORK); unattended `extend` or `divergence`."""
        if decl.declared:
            return "declared"
        if not self.unattended:
            return "open"
        return "extend" if self.extends_available else "divergence"


# --------------------------------------------------------------------------- the arithmetic

def normal_path(path) -> str:
    """`/showcase/{path:path}` -> `/showcase/{path}`."""
    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*):[^}]*\}", r"{\1}", str(path))


def served_endpoints(serve_dir) -> tuple:
    """Every endpoint of the route modules `serve/app.py` composes (`ROUTE_MODULES`) that
    `serve/` holds, read with `ast.literal_eval` as `serving_templates` reads them."""
    serve_dir = Path(serve_dir)
    app = serve_dir / "app.py"
    if not app.is_file():
        return ()
    names = next((ast.literal_eval(n.value) for n in ast.parse(app.read_text(encoding="utf-8"))
                  .body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) ==
                                                             "ROUTE_MODULES" for t in n.targets)),
                 ())
    found = set()
    for name in names:
        path = serve_dir / f"{name}.py"
        if path.is_file():
            lit = st.route_literals(path.read_text(encoding="utf-8"), f"serve/{name}.py")
            found |= {Endpoint(str(m).upper(), normal_path(p), lit["CAPABILITY"],
                               CALLS.get(name)) for m, p in lit["ENDPOINTS"]}
    return tuple(sorted(found, key=lambda e: (e.path, e.method, e.capability)))


def declaration(served, pipeline) -> Declaration:
    """The served endpoints against `pipeline.json.serving.api.endpoints`, by method and path."""
    api = (((pipeline or {}).get("serving") or {}).get("api") or {}).get("endpoints") or []
    declared = {(str(e.get("method", "")).upper(), normal_path(e.get("path", ""))): e
                for e in api if isinstance(e, dict)}
    unattended = {x.get("bet") for x in (pipeline or {}).get("extends") or []
                  if isinstance(x, dict) and "vega:default" in str(x.get("by", ""))} - {None}
    by_default = {(e.text, declared[e.key].get("target"), declared[e.key].get("bet"))
                  for e in served if e.key in declared and declared[e.key].get("bet") in unattended}
    missing = tuple(e for e in served if e.key not in declared)
    return Declaration(tuple(served), missing, tuple(sorted(by_default, key=str)))


def extends_entries(missing, fork: Fork) -> dict:
    """{target: [entry]} in the blueprint's gating order; `calls` the blueprint's port name. A
    route served both as a core fallback and by the showcase slot (`GET /`) is passed under both
    capabilities, as the route modules declare it: the blueprint's fallback rule joins them."""
    targets = {}
    for e in missing:
        entry = {"method": e.method, "path": e.path, "capability": e.capability}
        if e.calls:
            entry["calls"] = fork.ports.get(e.calls, e.calls)
        target = TARGET_OF.get(e.capability) or ("serving.demo" if fork.demo else "serving.api")
        targets.setdefault(target, []).append(entry)
    return {t: targets[t] for t in TARGET_ORDER if t in targets}


def extends_call(targets: dict, fork: Fork) -> str:
    """The `/blueprint --extends` call that declares `targets`: the entries a JSON list for one
    target, an object keyed by target for several; unattended, `--report-only` and the by-line
    that marks the bets `vega:default`."""
    entries = next(iter(targets.values())) if len(targets) == 1 else targets
    parts = ["/blueprint", "--extends", ",".join(targets), "--pipeline",
             shlex.quote(fork.pipeline), "--entries",
             shlex.quote(json.dumps(entries, separators=(",", ":"))),
             "--because", shlex.quote(fork.because)]
    if "serving.inspection" in targets:
        parts += ["--weights", fork.weights]
    if "serving.showcase" in targets:
        parts += ["--slot", SLOT]
    parts += ["--by", shlex.quote(UNATTENDED_BY if fork.unattended else "/vega-pack")]
    return " ".join(parts + (["--report-only"] if fork.unattended else []))


def declaration_record(decl: Declaration, fork: Fork) -> dict:
    """`MANIFEST.blueprint.conformance.declaration`."""
    answer = fork.answer(decl)
    record = {"pipeline": fork.pipeline, "served": sorted({e.text for e in decl.served}),
              "missing": sorted({e.text for e in decl.missing}), "answer": answer,
              "source": SOURCES[answer],
              "by_default": [{"endpoint": t, "target": g, "bet": b}
                             for t, g, b in decl.by_default], "note": NOTE}
    if decl.missing:
        record["targets"] = extends_entries(decl.missing, fork)
        record["extends"] = extends_call(record["targets"], fork)
    return record


def extends_available(skill=None) -> bool:
    """Whether the installed blueprint skill documents `--extends` (X1's patch applied): the
    unattended answer (a) needs it, and without it the fork's answer is (c)."""
    path = Path(skill or BLUEPRINT_SKILL)
    try:
        return bool(EXTENDS_FORM.search(path.read_text(encoding="utf-8")))
    except OSError:
        return False


# --------------------------------------------------------------------------- the phase

def declare(ctx) -> dict:
    """The fork over the bundle as rendered: its record, and its line (OK, WARN or FORK)."""
    served = served_endpoints(ctx.paths.root / "serve")
    slot_routes(served)
    if not served or not ctx.options.pipeline:
        return {}
    pipeline = ctx.pipeline()
    decl = declaration(served, pipeline)
    record = declaration_record(decl, Fork.of(ctx))
    _say(ctx, decl, record)
    _weights_bet(ctx, pipeline)
    return {"blueprint.conformance.api_declared": decl.declared,
            "blueprint.conformance.declaration": record}


def slot_routes(served) -> None:
    """MANIFEST.serve.showcase_slot.routes (the catalog's, R-D6) and the routes the showcase
    module serves are one list, read through the same spelling; two lists that disagree are a
    bundle that describes a slot it does not serve (exit 3)."""
    showcase = sorted({e.text for e in served if e.capability == "showcase"})
    recorded = sorted({normal_path(r) for r in CATALOG["serve_record"]["showcase_slot"]["routes"]})
    if showcase and showcase != recorded:
        raise PackExit(3, f"MANIFEST.serve.showcase_slot.routes names {', '.join(recorded)}; the "
                          f"showcase route module serves {', '.join(showcase)}",
                       keep_markers=True)


def _grouped(record) -> str:
    return "; ".join(f"{t}: " + ", ".join(f"{e['method']} {e['path']}" for e in entries)
                     for t, entries in (record.get("targets") or {}).items())


def _say(ctx, decl, record) -> None:
    where, n, answer = record["pipeline"], len(record["missing"]), record["answer"]
    if answer == "declared":
        _say_declared(ctx, decl, record)
    elif answer == "open":
        _fork(f"blueprint declaration — the bundle serves {n} endpoint(s) {where} does not "
              f"declare ({_grouped(record)}): (a) /blueprint --extends "
              f"{','.join(record['targets'])}, the call "
              "MANIFEST.blueprint.conformance.declaration.extends records, then pack again "
              "against the amended run (recommended: they become gated bets the bundle's "
              "baseline review reads); (b) pack without them — --no-inspect drops the inspection "
              "routes, --no-serve the whole service; (c) ship with the divergence recorded, "
              "which is this bundle as built (MANIFEST.blueprint.conformance.api_declared: "
              "false)", ctx.markers)
    elif answer == "extend":
        out_warn(f"blueprint declaration (unattended, vega:default): {n} served endpoint(s) are "
                 f"not declared in {where} ({_grouped(record)}); the answer is (a) — apply "
                 "MANIFEST.blueprint.conformance.declaration.extends through the Skill tool, "
                 "then pack again with --pipeline <the amended run>/pipeline.json; this bundle "
                 "records api_declared: false until then", ctx.markers)
    else:
        out_warn(f"blueprint declaration (unattended): {n} served endpoint(s) are not declared "
                 f"in {where} ({_grouped(record)}); the blueprint skill with --extends is "
                 "absent, so the bundle ships with the divergence recorded "
                 "(MANIFEST.blueprint.conformance.api_declared: false)", ctx.markers)


def _say_declared(ctx, decl, record) -> None:
    if not decl.by_default:
        out_ok(f"blueprint declaration: the {len(record['served'])} served endpoint(s) are "
               f"declared in {record['pipeline']} (serving.api.endpoints)", ctx.markers)
        return
    words = sorted({WORDS.get(g, str(g)) for _t, g, _b in decl.by_default})
    bets = sorted({str(b) for _t, _g, b in decl.by_default})
    out_warn(f"blueprint extended with {len({t for t, _g, _b in decl.by_default})} serving "
             f"endpoints ({', '.join(words)}) — bet(s) {', '.join(bets)} of "
             f"{record['pipeline']} were made by an unattended /vega-pack (vega:default), not "
             "gated by a person", ctx.markers)


def _fork(msg, markers) -> None:
    """A FORK: line is part of the run's account like the OK / WARN lines. No line this module
    writes names the judgement tool by its word: the bundle layer reads that word in the markers
    as the declaration of its absence (apply_bundle_layer's architect check)."""
    markers.append(f"FORK: {msg}")
    print(f"FORK: {msg}")


def _weights_bet(ctx, pipeline) -> None:
    """A blueprint whose inspection bet withholds the weights, and a bundle that publishes them."""
    bet = ((pipeline.get("serving") or {}).get("inspection")) or {}
    if ctx.options.inspect and bet.get("weights") == "summary" and \
            ctx.options.inspect_weights == "full":
        out_warn(f"the blueprint's serving.inspection bet ({bet.get('bet') or 'no bet id'}) says "
                 "weights summary, and this bundle publishes full weights; pack with "
                 "--inspect-weights summary to keep to it", ctx.markers)
