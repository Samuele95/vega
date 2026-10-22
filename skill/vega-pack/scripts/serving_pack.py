"""
serving_pack — everything the bundle renders from the backend's serving templates.

    serving_pack.identity(options, pipeline, schema) -> ident     (MANIFEST.serve.identity)
    serving_pack.render_adapters(ctx) -> {"entrypoints": […]}      before the round trip
    serving_pack.render(ctx) -> {"serve": …[, "entrypoints": […]]}  after it
    serving_pack.served_entrypoints(ctx) -> […]                    MANIFEST.entrypoints, served

`render_adapters` writes what the round trip runs through: the `package` templates in the
Predictor's module (the one Predictor and its support files, `serving_templates`), the Model
side they import (`serving_closure`), the entry points (`predict.py`, and `train.py` /
`evaluate.py` for `--include full`), `tests/test_roundtrip.py` and the pinned
`requirements.txt`. `render` writes the adapter the serving mode implies — for `online` the HTTP
service (`serve/app.py` over its route modules, the row contract, the console page), the API
tests; for `stream` the consumer and the row contract it reuses; nothing for `batch` — then the
service's requirements, the distribution and the Docker image, and the bundle's README; and it
returns the `serve` record whether or not a service was written. `pack --serve` on a batch or
stream blueprint renders the online adapter anyway and says so (`forced_by`, R-D10). The example
request (`examples/request.json`) is written after `render` by `examples_pack.write_request`,
encoded by the contract module `render` wrote.

The serve record (bundle-contract §2.4) carries, beside v2.2's keys, `endpoints` — the sorted
`"METHOD /path"` of the route modules `serve/` holds, read from their `ENDPOINTS` literals, so it
is exactly what the app registers — `templates` (the serving templates by digest), `forced_by`
when `--serve` overrode the blueprint, and for the online service `showcase_slot`, where
`/vega-show`'s showcase goes and which routes serve it (R-D6).

`MANIFEST.entrypoints` is this module's record: the CLIs `render_adapters` wrote and — for a
served bundle — every file the service is made of and run from, in the order c849f8b listed
them (amendment A1): `serve/` (`__init__.py`, the entry, the other modules, the page), the
demonstration sample and its presets when `demo_pack` wrote them, the example request, the
service's requirements, the distribution, the image, and the API tests. Nothing here writes
code: the templates live in `backends/<name>/templates/serving/` (`readme/` and `docker/` hold
the fragments the README and the Dockerfile are composed of).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

import output_record
import serving_closure
import serving_templates as st
from bundle_map import add_map_entries
from bundle_inputs import transformer_requirements
from bundle_slice import bundle_modules
from pack_context import out_ok, out_warn, read_pipeline
from vega_pack_core import CATALOG, serving_mode

__all__ = ["base_image", "identity", "render", "render_adapters", "served_entrypoints"]

CLI_FULL = ("predict.py", "train.py", "evaluate.py")
DEMO_ENTRYPOINTS = ("examples/demo.json", "examples/demo.csv", "examples/presets.json")
SERVED_TAIL = ("examples/request.json", "requirements-serve.txt", "pyproject.toml",
               "Dockerfile", ".dockerignore")
SERVE_HEAD = ("serve/__init__.py", "serve/app.py", "serve/consumer.py")
PAGE = "serve/static/index.html"
CLASS_READINGS = ("binary", "multiclass", "ordinal")
# the ports each entry point calls, for the map (a module that calls none: an empty list)
CALLS = {"serve/app.py": ["Predictor", "Monitor"], "serve/consumer.py": ["Predictor", "Monitor"],
         "serve/routes_predict.py": ["Predictor"], "serve/routes_monitor.py": ["Monitor",
                                                                              "Predictor"],
         "serve/routes_inspect.py": ["Predictor"]}
ADAPTER = {"online": ("an HTTP service", "docker/online.txt", "bundle/README.serve.md"),
           "stream": ("a stream consumer", "docker/stream.txt", "readme/README.stream.md")}
# the installed `{cmd}-serve` command: the adapter the mode ships (the stream bundle has no app)
SERVE_ENTRY = {"online": "serve.app:main", "stream": "serve.consumer:main"}


def identity(options, pipeline, schema) -> dict:
    """What the served bundle calls itself — never a literal in the shared templates.

    The command stem and the environment prefix come from the blueprint's system name (its
    first token: `cids-flow-classifier` -> `cids-serve`, `CIDS_BUNDLE`); the title, the noun for
    one input row and the one-paragraph description come from --service-title, --row-noun and
    --service-description. Without --service-description the description is derived at render
    time from the output record (`render`), which is written after this. The record goes into
    MANIFEST.serve.identity, so a re-pack can reproduce it."""
    del schema                        # the description reads the output record, at render
    system = (read_pipeline(pipeline).get("system") or {}) if pipeline else {}
    name = str(system.get("name") or (system.get("package") or "bundle").split("_")[0])
    cmd = re.sub(r"[^a-z0-9]", "", name.split("-")[0].lower()) or "bundle"
    noun = (options.row_noun or "row").strip().lower()
    if not re.fullmatch(r"[a-z][a-z -]{0,30}", noun):
        raise SystemExit(f"ERROR: --row-noun {noun!r}: a plain lower-case word is expected")
    port = int(options.service_port or 8765)
    if not 1024 <= port <= 65535:
        raise SystemExit(f"ERROR: --service-port {port}: an unprivileged port is expected")
    return {"cmd": cmd, "env": cmd.upper(), "noun": noun, "port": port,
            "nouns": noun if noun.endswith("s") else noun + "s",
            "title": options.service_title or name.replace("-", " "),
            "description": options.service_description, "source": f"blueprint:system/{name}"}


def _served_identity(ctx) -> dict:
    """The identity with its description: the user's, else one derived from the output record
    (MANIFEST.output as the stub carries it) and the schema."""
    ident = dict(ctx.ident)
    if not ident.get("description"):
        ident["description"] = _description(_output(ctx), ctx.schema, ident["noun"])
    return ident


def _output(ctx) -> dict:
    try:
        return json.loads(ctx.paths.manifest.read_text(encoding="utf-8"))["output"]
    except (OSError, ValueError, KeyError):
        label = ctx.campaign.label or ((ctx.spec.get("task") or {}).get("label"))
        return output_record.output_record(ctx.spec, ctx.campaign.threshold, label)


def _description(output, schema, noun) -> str:
    n_in = len(schema.get("input") or [])
    classes = list(output.get("classes") or [])
    if classes and output.get("reading") in CLASS_READINGS:
        return (f"Classifies each {noun} of the input schema ({n_in} columns) into one of "
                f"{len(classes)} classes. Valid in distribution only.")
    return (f"Answers each {noun} of the input schema ({n_in} columns) with its "
            f"{output.get('reading') or 'model'} reading. Valid in distribution only.")


def _template(ctx, rel) -> str:
    return (ctx.templates / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- before the round trip

def render_adapters(ctx) -> dict:
    """The Predictor and its support files, the Model side they import, the entry points and
    the requirements, before the round trip that runs through them."""
    st.render_package(ctx)
    serving_closure.copy_model_side(ctx)
    written = _write_entrypoints(ctx)
    _write_requirements(ctx)
    return {"entrypoints": written}


def _cli_names(ctx) -> tuple:
    """The CLIs of this shape: `predict.py`, and `train.py` / `evaluate.py` for `full`."""
    return CLI_FULL if ctx.options.include == "full" else CLI_FULL[:1]


def _write_entrypoints(ctx) -> list:
    """The bundle's entry points, one per thing it can be asked to do in this shape."""
    written, select = [], st.shape_selector(ctx)
    for name in _cli_names(ctx):
        # the bundle's sample shape's sections only (a row bundle exports no graph context)
        text = select(_template(ctx, f"bundle/{name}"), f"bundle/{name}") \
            .replace("{package}", ctx.package).replace("{env}", ctx.ident["env"])
        for key, module in bundle_modules(ctx.paths.pkg).items():
            text = text.replace("{mod_" + key + "}", module)
        (ctx.paths.root / name).write_text(text, encoding="utf-8")
        written.append(name)
    ctx.paths.tests.mkdir(parents=True, exist_ok=True)
    (ctx.paths.tests / "test_roundtrip.py").write_text(_template(ctx, st.ROUNDTRIP_TEST),
                                                       encoding="utf-8")
    if ctx.options.include == "full":
        out_ok(f"{', '.join(written)} and tests/test_roundtrip.py written — train.py fits one "
               "spec and evaluate.py scores one part, neither of them the stage it echoes: "
               "no ladder, no intervals, no verdict, and the test part still sealed",
               ctx.markers)
    else:
        out_ok("predict.py (CLI + module API) and tests/test_roundtrip.py written", ctx.markers)
    return written


def _write_requirements(ctx) -> None:
    """Pinned to the versions in use, because a bundle that installs a different numpy is a
    bundle whose round trip proves nothing about what the user will run."""
    lines = ["# pinned to the versions this bundle was built and round-tripped against"]
    for module in ("torch", "pandas", "numpy", "pyarrow"):
        try:
            mod = __import__(module)
        except ImportError:
            continue
        version = str(getattr(mod, "__version__", ""))
        public, _, local = version.partition("+")
        if local:
            # a local build tag (torch's "+cu130") names a wheel index, not a release: PyPI
            # serves no such file, so the pin is the release and the build is said beside it
            lines.append(f"# {module} was {version} at pack time; any build of {public} "
                         "(CPU or CUDA) is expected to reproduce the examples within the "
                         "manifest's tolerance — predict.py --check says whether it does")
        lines.append(f"{module}=={public}".rstrip("="))
    try:
        import sklearn
        lines.append(f"scikit-learn=={sklearn.__version__}")
    except ImportError:
        pass
    listed = {ln.split("==")[0].lower() for ln in lines if not ln.startswith("#")}
    extra = [(d, v) for d, v in transformer_requirements(ctx.paths.pkg) if d not in listed]
    if extra:
        lines.append("# imported by the recipe's FeatureTransformer (features/preprocess.py)")
        lines += [f"{d}=={v}" if v else d for d, v in extra]
    (ctx.paths.root / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_ok(f"requirements.txt — {len(lines) - 1} pinned dependency/ies", ctx.markers)


# --------------------------------------------------------------------------- after the round trip

def render(ctx) -> dict:
    """The adapter the serving mode implies (or the record of its absence), and the README;
    for a served bundle, the entry points with the served files after the CLIs."""
    plan = st.ServePlan.of(ctx)
    if not plan.present:
        out_ok(f"no serving scaffold: {plan.absent}", ctx.markers)
        _write_readme(ctx, None, None)
        return {"serve": _absent(ctx, plan)}
    ident = _served_identity(ctx)
    files = st.render_serve(ctx, plan, ident)
    add_map_entries(ctx, st.SOURCE, points=[{"file": f, "calls": CALLS.get(f, [])}
                                            for f in files])
    record = _write_serving(ctx, plan, ident)
    _write_readme(ctx, plan, record)
    return {"serve": record, "entrypoints": served_entrypoints(ctx)}


def served_entrypoints(ctx) -> list:
    """MANIFEST.entrypoints of a served bundle, in c849f8b's order: the CLIs, `serve/` (its
    `__init__.py` and entry first, the other modules in name order, the page last), the
    demonstration files `demo_pack` wrote, then the example request `examples_pack` writes
    next, the service's requirements, the distribution, the image and the API tests."""
    root = ctx.paths.root
    serve = sorted(p.relative_to(root).as_posix() for p in (root / "serve").rglob("*.py")
                   if "__pycache__" not in p.parts) if (root / "serve").is_dir() else []
    head = [rel for rel in SERVE_HEAD if rel in serve]
    body = head + [rel for rel in serve if rel not in head]
    body += [PAGE] if (root / PAGE).is_file() else []
    demo = [rel for rel in DEMO_ENTRYPOINTS if (root / rel).is_file()]
    tests = sorted(p.relative_to(root).as_posix() for p in (root / "tests").glob("*.py")
                   if p.name != "test_roundtrip.py") if (root / "tests").is_dir() else []
    return [*_cli_names(ctx), *body, *demo, *SERVED_TAIL, *tests]


def _absent(ctx, plan) -> dict:
    serve = serving_mode(plan.blueprint)
    return {"present": False, "mode": plan.blueprint, "framework": serve.get("framework"),
            "endpoint": serve.get("endpoint"), "source": "blueprint:serving",
            "note": (f"the blueprint's serving mode is {plan.blueprint}; no serving scaffold "
                     f"was written ({plan.absent}) — the Predictor port is the same over one "
                     "row (predict_one, predict.py --one) or a frame (predict.py --input)")}


def _write_serving(ctx, plan, ident) -> dict:
    """The service's requirements, the distribution (pyproject) and the Docker image, then
    the serve record."""
    torch_ver = _write_serve_requirements(ctx)
    fmt = _packaging(ctx, torch_ver, plan)
    root = ctx.paths.root
    (root / "pyproject.toml").write_text(_template(ctx, "bundle/pyproject.toml").format(**fmt),
                                         encoding="utf-8")
    (root / "Dockerfile").write_text(_template(ctx, "bundle/Dockerfile").format(**fmt),
                                     encoding="utf-8")
    (root / ".dockerignore").write_text(_template(ctx, "bundle/.dockerignore"),
                                        encoding="utf-8")
    record = _serve_record(ctx, plan, ident, fmt)
    out_ok(f"serving: {plan.mode} — {record['entry']} from the {ctx.campaign.backend_name} "
           f"templates, {len(record.get('endpoints') or [])} endpoint(s), pyproject "
           f"({fmt['dist']}), Dockerfile (CPU torch {torch_ver})"
           + (f"; rendered by {plan.forced_by} over a {plan.blueprint} blueprint"
              if plan.forced_by else ""), ctx.markers)
    return record


def _serve_record(ctx, plan, ident, fmt) -> dict:
    serve = serving_mode(plan.mode)
    record = {"present": True, "mode": plan.mode, "framework": serve.get("framework"),
              "entry": serve.get("entry"), "endpoint": serve.get("endpoint"),
              "endpoints": st.serving_endpoints(ctx.paths.root / "serve"),
              "templates": st.templates_digest(ctx.templates, ctx.campaign.backend_name),
              "run": _run_line(plan, ident, fmt), "identity": ident,
              "image": {"base": fmt["base_image"], "torch": f"{fmt['torch']} (CPU)",
                        "user": "non-root",
                        "healthcheck": "/health" if plan.mode == "online" else None},
              "source": "blueprint:serving"}
    if plan.mode == "online":
        record["showcase_slot"] = _showcase_slot(ident)
    if plan.forced_by:
        record["forced_by"] = plan.forced_by
        record["note"] = (f"the blueprint's serving mode is {plan.blueprint}; {plan.forced_by} "
                          "rendered the online adapter anyway (a demonstration service), and "
                          "the divergence is recorded against the blueprint")
    return record


def _showcase_slot(ident) -> dict:
    """MANIFEST.serve.showcase_slot (R-D6): the catalog's record, its variable named."""
    slot = json.loads(json.dumps(CATALOG["serve_record"]["showcase_slot"]))
    slot["env"] = slot["env"].replace("<ENV>", ident["env"])
    return slot


def _run_line(plan, ident, fmt) -> str:
    if plan.mode == "stream":
        return (f"python -m serve.consumer --bundle . < rows.jsonl  |  docker build -t "
                f"{fmt['dist']} . && docker run -i {fmt['dist']} < rows.jsonl")
    return ("%s-serve --port %d  |  docker build -t %s . && docker run -p %d:%d %s"
            % (ident["cmd"], ident["port"], fmt["dist"], ident["port"], ident["port"],
               fmt["dist"]))


def _version(dist):
    import importlib.metadata as md
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _core_and_web(pkg_dir):
    # the core rows with the FeatureTransformer's own imports (F41), then the web rows
    core = [("pandas", _version("pandas")), ("numpy", _version("numpy")),
            ("pyarrow", _version("pyarrow"))] + transformer_requirements(pkg_dir)
    web = [("fastapi", _version("fastapi")), ("uvicorn", _version("uvicorn")),
           ("python-multipart", _version("python-multipart"))]
    return core, web


def _write_serve_requirements(ctx) -> str:
    torch_ver = (_version("torch") or "2").split("+")[0]
    core, web = _core_and_web(ctx.paths.pkg)
    missing = [d for d, v in web if v is None]
    if missing:
        out_warn(f"{', '.join(missing)} not installed in the packing environment; "
                 "requirements-serve.txt leaves them unpinned", ctx.markers)
    (ctx.paths.root / "requirements-serve.txt").write_text(
        "# the service's dependencies, pinned to the versions it was tested with; torch is\n"
        f"# installed separately (CPU build {torch_ver}) — see the Dockerfile\n"
        + "".join(f"{d}=={v}\n" if v else f"{d}\n" for d, v in core + web),
        encoding="utf-8")
    return torch_ver


def _packaging(ctx, torch_ver, plan) -> dict:
    """The values the pyproject and Dockerfile templates are formatted with."""
    core, web = _core_and_web(ctx.paths.pkg)
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    ident = ctx.ident
    fmt = dict(dist=f"{ctx.paths.root.name.lower()}", version="1.0.0", run=ctx.paths.root.name,
               python=py, package=ctx.package,
               base_image=base_image(f"python:{py}-slim", ctx.markers), cmd=ident["cmd"],
               env=ident["env"], title=ident["title"].replace('"', "'").replace("\n", " "),
               port=ident["port"], torch=torch_ver, what=ADAPTER[plan.mode][0],
               serve_entry=SERVE_ENTRY[plan.mode],
               deps=", ".join(f'"{d}>={v.split(".")[0]}"' for d, v in core if v)
               + f', "torch>={torch_ver.split(".")[0]}.{torch_ver.split(".")[1]}"',
               serve_deps=", ".join(f'"{d}>={v}"' if v else f'"{d}"' for d, v in web))
    fmt["runtime"] = _template(ctx, ADAPTER[plan.mode][1]).format(**fmt).rstrip("\n")
    return fmt


def base_image(tag: str, markers) -> str:
    """The base image pinned by digest, so a rebuild gets the bytes this bundle was tested on:
    the local image's repo digest, else the registry index's, else the tag with a WARN."""
    probes = (["docker", "image", "inspect", tag, "--format", "{{index .RepoDigests 0}}"],
              ["docker", "buildx", "imagetools", "inspect", tag])
    for cmd in probes:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode != 0:
            continue
        found = re.search(r"sha256:[0-9a-f]{64}", proc.stdout)
        if found:
            out_ok(f"base image {tag} pinned by digest {found.group(0)[:19]}…", markers)
            return f"{tag}@{found.group(0)}"
    out_warn(f"the base image {tag} could not be resolved to a digest (no docker, or no "
             "registry); the Dockerfile names the tag, which can move", markers)
    return tag


# --------------------------------------------------------------------------- the README

def _write_readme(ctx, plan, record) -> None:
    """README.md: the bundle's five lines, the verdict, and the served adapter's section."""
    verdict = ctx.campaign.verdict
    verdict_line = ("This network earned its verdict."
                    if verdict.get("earned") else
                    f"This network did NOT beat {verdict.get('strongest_baseline')} on the "
                    f"test part. It ships under a recorded stance; read model-card.html "
                    "before using it.")
    readme = _template(ctx, "readme/README.md")
    include_lines = _template(ctx, "readme/README.full.md") \
        if ctx.options.include == "full" else ""
    slug = ctx.paths.root.name[: -(len(ctx.campaign.run_id) + 1)]     # <slug>-<run>
    for key, value in (("{slug}", slug),
                       ("{run}", ctx.campaign.run_id), ("{campaign}", ctx.campaign.dir.name),
                       ("{include_lines}", include_lines), ("{verdict_line}", verdict_line)):
        readme = readme.replace(key, value)
    if plan is not None:
        ident = record["identity"]
        readme += _template(ctx, ADAPTER[plan.mode][2]).format(
            dist=ctx.paths.root.name.lower(), cmd=ident["cmd"], env=ident["env"],
            port=ident["port"], endpoints=", ".join(f"`{e}`" for e in record["endpoints"])
            or "none")
    (ctx.paths.root / "README.md").write_text(readme, encoding="utf-8")
