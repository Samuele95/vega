"""
serve.startup — the closed world the service of {{PACKAGE}} starts in: which bundle, which code,
each checked before any of it runs.

    bundle_root(bundle_dir=None) -> Path              the bundle directory (the data)
    read_manifest(root) -> dict                       MANIFEST.json ({} when it cannot be read)
    untrusted_serve(root, manifest) -> (rel, path)    the first serve/ file that is not the pinned
                                                      one, or None
    unpinned(root, manifest) -> (rel, path)           the first code or evidence file that is not
                                                      the pinned one, or None
    refusal(found) -> str                             what the service says about it
    verified(root, manifest) -> (bundle, why)         all of it: code, evidence, then the data
    composition_root(package) -> (module, why)        predict.py, imported from where the code is
    route_modules(names) -> (modules, problems)       the route modules this bundle ships

The bundle directory is data; the code runs from where this package was loaded (CODE_HOME) —
the bundle itself, or an installed distribution. The directory is resolved first (--bundle,
{{ENV}}_BUNDLE, the directory this package sits in, the working directory). Then, before
anything is imported beyond this module and `serve/app.py`, every file of `serve/` is checked
against the digests MANIFEST.json pinned (`untrusted_serve`): a service whose own modules are
not the packed ones cannot be trusted to say anything but that. Every other code file, and the
evidence the self-test replays, is checked next (`unpinned`) — a module under the package or
`serve/` that the manifest does not pin is refused too, because an import may run it. The data
half (weights, spec, fitted parameters, schema, examples) goes through the bundle's own
`predict.verify`, the one implementation the CLI's `load()` uses. A refusal is never an
exception out of `create_app`: the service starts unready and says why.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # serve/, where this code is installed
CODE_HOME = HERE.parent                           # the bundle, or site-packages
EVIDENCE = ("examples/rows.csv", "examples/expected.parquet")
# the module-level literals a route module declares (pack reads them with ast.literal_eval)
ROUTE_LITERALS = ("ENDPOINTS", "MODEL_ROUTES", "OPEN_WHILE_UNREADY", "CAPABILITY")
log = logging.getLogger("{{CMD}}.serve")

__all__ = ["CODE_HOME", "bundle_root", "composition_root", "digest", "read_manifest",
           "refusal", "route_modules", "unpinned", "untrusted_serve", "verified"]


def bundle_root(bundle_dir=None) -> Path:
    """The bundle directory: the one named, else {{ENV}}_BUNDLE, else the directory this package
    sits in when it is a bundle, else the working directory. A directory without a
    MANIFEST.json is not a bundle, and the service says so before importing anything."""
    for cand in (bundle_dir, os.environ.get("{{ENV}}_BUNDLE"), CODE_HOME, Path.cwd()):
        if cand and (Path(cand) / "MANIFEST.json").is_file():
            return Path(cand).resolve()
    named = bundle_dir or os.environ.get("{{ENV}}_BUNDLE")
    raise FileNotFoundError(f"no bundle at {named}: MANIFEST.json not found" if named else
                            "no bundle found: pass --bundle <dir> or set {{ENV}}_BUNDLE")


def read_manifest(root: Path) -> dict:
    """MANIFEST.json as a dict; {} when it does not parse, which every check then refuses."""
    try:
        doc = json.loads((Path(root) / "MANIFEST.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.error("MANIFEST.json could not be read")
        return {}
    return doc if isinstance(doc, dict) else {}


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def untrusted_serve(root: Path, manifest: dict):
    """The first file of `serve/` this process would run or serve that is not the pinned one,
    as (what, where), or None: every `serve/` file MANIFEST.json pins must hash to its digest
    where the code is installed, and no `serve/` module may be there that it does not pin."""
    files = manifest.get("files") or {}
    if not files:
        return "MANIFEST.json (no file pins)", Path(root) / "MANIFEST.json"
    for rel in sorted(r for r in files if r.startswith("serve/")):
        path = CODE_HOME / rel
        if not path.is_file() or digest(path) != files[rel]:
            return rel, path
    return _unlisted(files, ("serve",))


def unpinned(root: Path, manifest: dict):
    """The first code file this service would run, or evidence it would replay, that is not the
    file MANIFEST.json pinned, as (what, where), or None. Code is checked where it is imported
    from (CODE_HOME); the roles come from the manifest."""
    files, roles = manifest.get("files") or {}, manifest.get("roles") or {}
    if not roles.get("code"):
        return "MANIFEST.json (no file roles)", Path(root) / "MANIFEST.json"
    for rel in EVIDENCE:
        if rel not in (roles.get("example") or ()):
            return rel + " (not pinned as an example)", Path(root) / rel
    for rel in roles["code"]:
        path = CODE_HOME / rel
        if not path.is_file() or digest(path) != files.get(rel):
            return rel, path
    return _unlisted(files, (manifest.get("package") or "", "serve"))


def _unlisted(files: dict, tops: tuple):
    """A module under one of `tops` that MANIFEST.json does not pin: code nobody can verify."""
    for top in tops:
        base = CODE_HOME / top
        if not top or not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(CODE_HOME).as_posix()
            if "__pycache__" not in path.parts and rel not in files:
                return rel + " (not pinned at all)", path
    return None


def refusal(found) -> str:
    """What the service says about a file that is not the pinned one."""
    what, where = found
    log.error("unpinned file: %s at %s", what, where)
    return (f"{what} is not the file MANIFEST.json pinned; the service runs and reads only the "
            "files it was packed with, so it does not start")


def verified(root: Path, manifest: dict):
    """(the bundle's predict module, None), or (it or None, why not): the code and the evidence
    against their pins, the composition root imported from the code's location, the data
    through the bundle's own `verify`."""
    found = unpinned(root, manifest)
    if found:
        return None, refusal(found)
    bundle, why = composition_root(manifest.get("package") or "")
    if why:
        return None, why + "; the service runs only the code it was installed with"
    try:
        bundle.verify(root, ("data", "example"))
    except (ValueError, OSError) as exc:
        log.error("unpinned data: %s", exc)
        return bundle, str(exc)
    return bundle, None


def composition_root(package: str):
    """predict.py — the bundle's one composition root — imported from the code's own location
    (CODE_HOME), never found through sys.path order: a data directory that also holds a
    predict.py must not decide which code serves it. Where the package would come from is
    resolved first (find_spec runs nothing), so a shadowing package is refused before any of
    its code executes. Returns (module, None), or (None, why)."""
    spec = importlib.util.find_spec(package) if package else None
    origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if origin is None or CODE_HOME not in origin.parents:
        return None, (f"the package {package!r} would be imported from {origin}, not from "
                      "where this service is installed")
    loaded = sys.modules.get(package)
    if loaded is not None and CODE_HOME not in Path(loaded.__file__ or "").resolve().parents:
        return None, f"the package {package!r} is already loaded from another location"
    module, why = _predict_module()
    if why:
        return None, why
    for name, mod in (("predict", module), (package, sys.modules.get(package))):
        where = Path(getattr(mod, "__file__", "") or "").resolve()
        if mod is None or CODE_HOME not in where.parents:
            return None, (f"{name} was imported from another location than where this "
                          "service is installed")
    return module, None


def _predict_module():
    """`predict`, imported by file from CODE_HOME (or the one already imported from there)."""
    module = sys.modules.get("predict")
    if module is not None:
        return module, None
    spec = importlib.util.spec_from_file_location("predict", CODE_HOME / "predict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["predict"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop("predict", None)
        log.exception("predict.py failed to import")
        return None, f"predict.py failed to import ({type(exc).__name__})"
    return module, None


def route_modules(names) -> tuple:
    """(the route modules of `names` this bundle ships, imported in that order; the problems of
    the ones that could not be): a route module is a file `serve/<name>.py` declaring every
    literal of ROUTE_LITERALS and a `register(app, state)`."""
    modules, problems = [], []
    for name in names:
        if not (HERE / f"{name}.py").is_file():
            continue
        try:
            module = importlib.import_module(f"serve.{name}")
        except Exception as exc:
            log.exception("serve/%s.py failed to import", name)
            problems.append(f"serve/{name}.py failed to import ({type(exc).__name__})")
            continue
        missing = [n for n in (*ROUTE_LITERALS, "register") if not hasattr(module, n)]
        if missing:
            problems.append(f"serve/{name}.py declares no {', '.join(missing)}")
            continue
        modules.append(module)
    return tuple(modules), tuple(problems)
