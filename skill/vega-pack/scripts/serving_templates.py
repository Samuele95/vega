"""
serving_templates — the backend's serving templates, rendered into a bundle by pattern.

    ServePlan.of(ctx) -> ServePlan                      what `serve/` holds: online, stream, or
                                                        nothing (and why)
    fills(ctx, ident=None) -> {placeholder: value}      what a serving template's {{…}} become
    render_package(ctx) -> [file]                       `package` templates into the Predictor's
                                                        module (package-relative files)
    render_serve(ctx, plan, ident) -> [file]            `serve` and `tests` templates (bundle-
                                                        relative files)
    route_literals(text, where) -> {literal: value}     a route module's four literals
    serving_endpoints(serve_dir) -> ["METHOD /path"]    MANIFEST.serve.endpoints
    templates_digest(root, backend) -> {...}            MANIFEST.serve.templates
    enabled_capabilities(ctx) -> (capability, ...)

Which template renders where is `catalog.json` `serving_templates` (R-D2): glob patterns per
destination, matched by `vega_pack_core.template_matches`, so a template a later package adds is
picked up with no edit here. `package/*.py` and `package/inspection/*.py` render into the
Predictor's module (`inspection/` below it, left out with `--no-inspect`) — `package/predict.py`
as the Predictor's own adapter at the port's file, the others as support files mapped
`kind: support, used_by: [Predictor]` — except `package/monitor.py`, the Monitor's adapter,
which `monitor_pack` renders into the Monitor port's file from a monitor plan. `serve/*.py` and
`serve/static/index.html` render into `serve/`: for the online adapter all of them but the
stream consumer, and among the route modules only those whose capability this bundle enables;
for the stream adapter the consumer and the row contract it reuses. `tests/*.py` render into
`tests/` verbatim (`tests/test_roundtrip.py` with the CLI, before the round trip; the API tests
with the online service).

A template's code for some sample shapes sits in `# >>> shape: …` / `# <<< shape` sections; the
`package` and `serve` templates are rendered with the bundle's own shape (its spec's: the
package was emitted for it), through the backend emitter's `select_shapes` — so the serving
code carries the shape its package carries and dispatches to no other. A package template of
one shape only (`SHAPE_TEMPLATES`: `graphs.py`, the reference graph) is not rendered into a
bundle of another shape.

Every placeholder a template carries is filled here or the pack stops: an unfilled `{{…}}` is
`ERROR:` (exit 2). The values come from the bundle itself — its package, the modules its map
gives each role, the ports the blueprint drew, the served identity — never from a literal.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import html
import importlib.util
import json
import re
from dataclasses import dataclass
from types import MappingProxyType

from bundle_map import add_map_entries
from bundle_slice import bundle_modules
from pack_context import SKILLS, PackExit, out_ok, out_warn
from vega_pack_core import CATALOG, matching_templates, serving_mode

__all__ = ["ROUTE_LITERALS", "ServePlan", "bundle_shape", "enabled_capabilities", "fills",
           "render_package", "render_serve", "render_text", "route_literals",
           "serving_endpoints", "shape_selector", "templates_digest"]

PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")
ROUTE_LITERALS = tuple(CATALOG["serve_record"]["route_module_literals"])
ALWAYS_ENABLED = ("core", "code", "showcase")
PREDICTOR_TEMPLATE, MONITOR_TEMPLATE = "predict.py", "monitor.py"
# the package templates of one sample shape only: a bundle of another shape does not carry
# them (their domain values are not in its domain.py: the forge emits its spec's shape only)
SHAPE_TEMPLATES = MappingProxyType({"graphs.py": "graph"})
INSPECTION = "inspection/"
CONSUMER = "serve/consumer.py"
# the stream adapter: the consumer and the row contract it reuses (D2 §8.5, §12)
GAPS = "serve/gaps.py"                    # the consumer's own: the windows a refused row reaches
STREAM_SERVE = ("serve/__init__.py", CONSUMER, GAPS, "serve/contract.py", "serve/answers.py",
                "serve/problems.py", "serve/rows.py", "serve/tables.py")
ROUNDTRIP_TEST = "tests/test_roundtrip.py"
SOURCE = "the vega-pack serving templates"
ROLES = ("Predictor", "Monitor", "Model", "ModelStore", "FeatureTransformer", "DataSource",
         "Trainer", "Evaluator")
MODULE_KEYS = {"MODEL": "model", "STORE": "store", "INGESTION": "ingestion",
               "TRAINING": "training", "EVALUATION": "evaluation", "FEATURES": "features",
               "SERVING": "serving", "MONITOR": "monitor"}
INSPECTION_INIT = ('"""{package}.{module}.inspection — the read-only view of the network the '
                   'inspection routes serve (rendered by /vega-pack from the backend\'s '
                   'serving templates)."""\n')


# --------------------------------------------------------------------------- what serve/ holds

@dataclass(frozen=True)
class ServePlan:
    """What the bundle serves: `mode` the adapter rendered (the blueprint's mode when nothing
    is), `blueprint` the blueprint's own mode, `forced_by` the flag that rendered the online
    adapter over a batch or stream blueprint (R-D10), `absent` why nothing is rendered."""

    mode: str
    blueprint: str
    present: bool
    forced_by: str | None = None
    absent: str | None = None

    @staticmethod
    def of(ctx) -> "ServePlan":
        blueprint, asked = ctx.mode, ctx.options.serve
        if asked is False:
            return ServePlan(blueprint, blueprint, False, absent="--no-serve")
        if asked is True:
            return ServePlan("online", blueprint, True,
                             forced_by="--serve" if blueprint != "online" else None)
        if not serving_mode(blueprint).get("serve"):
            return ServePlan(blueprint, blueprint, False, absent=f"a {blueprint} blueprint")
        if blueprint == "stream" and not (ctx.templates / CONSUMER).is_file():
            out_warn(f"the blueprint serves a stream, and the {ctx.campaign.backend_name} "
                     f"backend ships no {CONSUMER} template: no consumer is written (the "
                     "stream adapter is best effort outside the serving matrix's stream cell)",
                     ctx.markers)
            return ServePlan(blueprint, blueprint, False, absent=f"no {CONSUMER} template")
        return ServePlan(blueprint, blueprint, True)


def enabled_capabilities(ctx) -> tuple:
    """The capabilities this bundle serves: core, code and the showcase slot always; the
    inspection surface unless `--no-inspect` (or the backend ships no inspection templates)."""
    ships = any(rel.split("/", 1)[1].startswith(INSPECTION)
                for rel in matching_templates(ctx.templates)["package"])
    return ALWAYS_ENABLED + (("inspection",) if ctx.options.inspect and ships else ())


# --------------------------------------------------------------------------- the placeholders

def fills(ctx, ident=None) -> dict:
    """{placeholder name: value}: the package, the blueprint's serving mode, the module of each
    role (from the bundle's own map), each port's name, responsibility and source, and — for
    the `serve` templates — the served identity."""
    modules = bundle_modules(ctx.paths.pkg)
    mapping = {"PACKAGE": ctx.package, "SERVING_MODE": ctx.mode,
               "MONITOR_MODULE": modules["monitor"]}
    mapping.update({f"MODULE_{key}": modules[name] for key, name in MODULE_KEYS.items()})
    for role in ROLES:
        port = ctx.port(role)
        name = port.get("name") or role
        mapping[f"PORT_{role.upper()}"] = name
        mapping[f"RESPONSIBILITY_{role.upper()}"] = port.get("responsibility") or \
            "(no responsibility line in the blueprint)"
        mapping[f"SOURCE_{role.upper()}"] = f"blueprint:ports/{name}"
    if ident is not None:
        mapping.update(_identity_fills(ident))
    return mapping


def _identity_fills(ident) -> dict:
    return {"TITLE_PY": repr(ident["title"]), "DESCRIPTION_PY": repr(ident["description"]),
            "TITLE_HTML": html.escape(ident["title"]),
            "DESCRIPTION_HTML": html.escape(ident["description"]),
            "NOUNS": ident["nouns"], "NOUN": ident["noun"],
            "ROW_MODEL": "".join(w.capitalize() for w in re.split(r"[ -]+", ident["noun"]))
            + "Row", "CMD": ident["cmd"], "ENV": ident["env"], "PORT": str(ident["port"])}


def bundle_shape(ctx) -> str:
    """The sample shape the bundle carries: its spec's `data_contract.sample.shape` (row when
    the spec names none), the one the forge emitted its package for."""
    sample = (ctx.spec.get("data_contract") or {}).get("sample") or {}
    return str(sample.get("shape") or "row")


@functools.lru_cache(maxsize=None)
def _backend_emitter(emit_path: str):
    """The backend's emitter module, loaded by path (it is not a package module)."""
    spec = importlib.util.spec_from_file_location(f"vega_pack_emit_{abs(hash(emit_path))}",
                                                  emit_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shape_selector(ctx):
    """`(text, where) -> text` keeping the template sections of the bundle's sample shape: the
    backend emitter's `select_shapes`, so a section reads the same in the package and here."""
    backend = ctx.campaign.backend_name
    emit_path = SKILLS / "vega-forge" / "scripts" / "backends" / backend / "emit.py"
    try:
        select = _backend_emitter(str(emit_path)).select_shapes
    except (OSError, AttributeError, ImportError) as exc:
        raise PackExit(2, f"the {backend} backend's emitter ({emit_path}) "
                          f"offers no select_shapes ({exc}); the serving templates' shape "
                          "sections cannot be rendered") from None
    shapes = (bundle_shape(ctx),)

    def selected(text: str, where: str) -> str:
        try:
            return select(text, shapes, where)
        except ValueError as exc:            # the emitter's EmitError
            raise PackExit(2, f"the serving template {where}: {exc}") from None
    return selected


def render_text(text: str, mapping: dict, where: str, select=None) -> str:
    """`text` with the shape sections `select` keeps (`shape_selector(ctx)`; None leaves the
    text as it is) and every `{{NAME}}` of `mapping` filled; a placeholder left is exit 2."""
    if select is not None:
        text = select(text, where)
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", str(value))
    left = sorted(set(PLACEHOLDER.findall(text)))
    if left:
        raise PackExit(2, f"the serving template {where} keeps unfilled placeholder(s) "
                          f"{', '.join(left)}")
    return text


# --------------------------------------------------------------------------- package

def render_package(ctx) -> list:
    """The `package` templates into the Predictor's module; the files written (package-
    relative), the Predictor's adapter first. The support files are mapped."""
    found = matching_templates(ctx.templates)["package"]
    if f"package/{PREDICTOR_TEMPLATE}" not in found:
        out_warn(f"the {ctx.campaign.backend_name} backend ships no package/"
                 f"{PREDICTOR_TEMPLATE} template under {ctx.templates}; the bundle keeps the "
                 "stub and predict.py will refuse", ctx.markers)
        return []
    port = ctx.port("Predictor")
    adapter = port.get("file") or f"{bundle_modules(ctx.paths.pkg)['serving']}/predict.py"
    module = port.get("module") or adapter.split("/")[0]
    mapping, written, select = fills(ctx), [], shape_selector(ctx)
    for rel in sorted(found, key=lambda r: r != f"package/{PREDICTOR_TEMPLATE}"):
        name = rel.split("/", 1)[1]
        if name == MONITOR_TEMPLATE or (name.startswith(INSPECTION) and not ctx.options.inspect):
            continue
        if SHAPE_TEMPLATES.get(name, bundle_shape(ctx)) != bundle_shape(ctx):
            continue
        target = adapter if name == PREDICTOR_TEMPLATE else f"{module}/{name}"
        _write(ctx.paths.pkg / target, render_text(_read(ctx, rel), mapping, rel, select))
        written.append(target)
    written += _inspection_init(ctx, module, written)
    add_map_entries(ctx, SOURCE, files=[f for f in written if f != adapter])
    out_ok(f"{adapter} and {len(written) - 1} support file(s) rendered from the "
           f"{ctx.campaign.backend_name} backend's package templates into {module}/ (serving "
           f"mode {ctx.mode}" + ("" if ctx.options.inspect else "; --no-inspect: no "
                                 "inspection modules") + ")", ctx.markers)
    return written


def _inspection_init(ctx, module, written) -> list:
    """`<module>/inspection/__init__.py`, when inspection modules were rendered and the
    backend's templates carry none of their own (since I06 they carry one, exporting
    `NetworkInspector` and `ExplainQuery`: it is rendered with the rest and kept)."""
    rel = f"{module}/{INSPECTION}__init__.py"
    if rel in written or not any(f.startswith(f"{module}/{INSPECTION}") for f in written):
        return []
    _write(ctx.paths.pkg / rel, INSPECTION_INIT.format(package=ctx.package, module=module))
    return [rel]


# --------------------------------------------------------------------------- serve and tests

def render_serve(ctx, plan: ServePlan, ident) -> list:
    """`serve/` for the plan's adapter and, with the online service, the API tests; the
    bundle-relative files written. The tests are copied verbatim (they read the bundle)."""
    found = matching_templates(ctx.templates)
    mapping, enabled = fills(ctx, ident), enabled_capabilities(ctx)
    written, select = [], shape_selector(ctx)
    for rel in found["serve"]:
        if not _in_plan(rel, plan):
            continue
        text = render_text(_read(ctx, rel), mapping, rel, select)
        if _is_route_module(rel) and route_literals(text, rel)["CAPABILITY"] not in enabled:
            continue
        _write(ctx.paths.root / rel, text)
        written.append(rel)
    if plan.mode == "online":
        for rel in found["tests"]:
            if rel != ROUNDTRIP_TEST:
                _write(ctx.paths.root / rel, _read(ctx, rel))
                written.append(rel)
    return written


def _in_plan(rel: str, plan: ServePlan) -> bool:
    return rel in STREAM_SERVE if plan.mode == "stream" else rel not in (CONSUMER, GAPS)


def _is_route_module(rel: str) -> bool:
    return re.fullmatch(r"serve/routes_[a-z0-9_]+\.py", rel) is not None


def route_literals(text: str, where: str) -> dict:
    """A route module's ENDPOINTS, MODEL_ROUTES, OPEN_WHILE_UNREADY and CAPABILITY, read with
    `ast.literal_eval` — never by importing the module; a missing or computed one is exit 2."""
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name) and node.targets[0].id in ROUTE_LITERALS:
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                raise PackExit(2, f"{where}: {node.targets[0].id} is not a literal; pack reads "
                                  "the route modules without importing them") from None
    missing = [name for name in ROUTE_LITERALS if name not in found]
    if missing:
        raise PackExit(2, f"{where} declares no {', '.join(missing)} (a route module's "
                          "literals, R-D1)")
    return found


def serving_endpoints(serve_dir) -> list:
    """MANIFEST.serve.endpoints: the sorted `"METHOD /path"` of every route module of
    `ROUTE_MODULES` (read from `serve/app.py`) that `serve/` holds — what the app registers."""
    app = serve_dir / "app.py"
    if not app.is_file():
        return []
    names = next((ast.literal_eval(n.value) for n in ast.parse(app.read_text(encoding="utf-8"))
                  .body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) ==
                                                             "ROUTE_MODULES" for t in n.targets)),
                 None)
    if not names:
        raise PackExit(2, "serve/app.py declares no literal ROUTE_MODULES")
    endpoints = set()
    for name in names:
        path = serve_dir / f"{name}.py"
        if path.is_file():
            literals = route_literals(path.read_text(encoding="utf-8"), f"serve/{name}.py")
            endpoints |= {f"{method} {route}" for method, route in literals["ENDPOINTS"]}
    return sorted(endpoints)


def templates_digest(root, backend: str) -> dict:
    """MANIFEST.serve.templates: every template the serving patterns name, by digest, and the
    digest of that listing (canonical JSON), which `--re-emit-serving` compares."""
    files = {rel: "sha256:" + hashlib.sha256((root / rel).read_bytes()).hexdigest()
             for rels in matching_templates(root).values() for rel in rels}
    listing = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"backend": backend, "digest": "sha256:" + hashlib.sha256(listing).hexdigest(),
            "files": dict(sorted(files.items()))}


def _read(ctx, rel: str) -> str:
    return (ctx.templates / rel).read_text(encoding="utf-8")


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
