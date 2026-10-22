"""
serve_app_fixtures.py — the bundles test_serve_app.py proves the service on (build plan S07).

Not a test file (the unit tier collects test_*.py). A bundle here is built the way `pack.py`
builds one, through vega-pack's own phase modules, over a package the v2.3 emitter wrote:

  campaign  `render_templates.render(..., full=True)` — exactly the emitter — writes the
            evaluated run's package for one of predictor_fixtures' row cases (the shaper's real
            row recipe, a seeded network), under the forge fixture's world model: the orders
            blueprint, whose serving mode is `batch`.
  slice     `bundle_slice.copy_package_slice` copies the serving slice; the shaper's
            transform-only module is the FeatureTransformer; netspec, fitted, recipe and schema
            are written beside it.
  adapters  `serving_pack.render_adapters(ctx)` renders the package templates into the
            Predictor's module, copies the Model side they import, writes predict.py,
            tests/test_roundtrip.py and requirements.txt — the code under test.
  data      the weights saved through the bundle's own ModelRegistry; the examples (`__row__`,
            the compared and decided fields of the evaluation's stored rows) and a monitor
            reference with its Monitor adapter rendered as monitor_pack renders it; the
            demonstration sample and presets as examples@2 files.
  serve     `serving_pack.render(ctx)` with `--serve` (forced over the batch blueprint): serve/,
            the API tests, the distribution, the image, the README; `bundle_slice.finish`; then
            MANIFEST.json with every file pinned by role and the serve record.

`served(bundle)` imports the bundle's own `serve.app` (one bundle per process: the service
imports the bundle's `predict` module by name), and `probe(bundle_dir, code)` runs a snippet in
a clean interpreter with a bundle directory's own code first on the path.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import predictor_fixtures as pf  # noqa: E402  (paths, the row cases, the rendered renderer)
import bundle_manifest  # noqa: E402
import bundle_slice  # noqa: E402
import pack_options  # noqa: E402
import serving_pack  # noqa: E402
from pack_context import Campaign, PackContext  # noqa: E402
from vega_pack_core import BundlePaths, Tolerance  # noqa: E402

_WORK = tempfile.TemporaryDirectory(prefix="serve-app-")
WORK = Path(_WORK.name)
RUN = "R0"
DEMO_ROWS = 3


@dataclass
class Built:
    """A bundle built through pack's phases: its directory, context, serve record, manifest."""

    root: Path
    ctx: object
    record: dict
    manifest: dict
    case: object

    @property
    def package(self) -> str:
        return self.ctx.package


def _pipeline(package, mode):
    pipeline = pf._pipeline(package)
    pipeline["serving"]["mode"] = mode
    return pipeline


def context(case, package, mode="batch", flags=("--serve",)) -> PackContext:
    """A pack context over a freshly emitted campaign and a copied slice, as `pack._assemble`
    leaves it: the package emitted, the slice copied, the transformer and the records written."""
    work = WORK / f"{package}-{mode}-{'-'.join(f.strip('-') for f in flags) or 'plain'}"
    shutil.rmtree(work, ignore_errors=True)
    (work / "campaign").mkdir(parents=True)
    pipeline = work / "pipeline.json"
    pipeline.write_text(json.dumps(_pipeline(package, mode), indent=1), encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        pf.rt.render(pf.TEMPLATES, work / "campaign", json.loads(pipeline.read_text()),
                     case.spec, full=True)
    options = pack_options.parse(["--campaign", str(work / "campaign"), "--shaped",
                                  str(case.shaped.dir), "--bundle-dir", str(work / "bundle"),
                                  "--pipeline", str(pipeline), "--demo-sample", "0", *flags])
    s = case.shaped
    campaign = Campaign(dir=work / "campaign", doc={"input": {"package": package}}, ledger={},
                        seal={}, run_id=RUN, eval_dir=work, metrics={"verdict": {"earned": True}},
                        threshold=case.threshold, shaped_dir=s.dir, shaped_manifest=s.manifest,
                        recipe=s.recipe, spec=case.spec, raw=s.raw, fitted=s.fitted)
    paths = BundlePaths(root=work / "bundle" / f"rows-{RUN}", package=package)
    markers = []
    with contextlib.redirect_stdout(io.StringIO()):
        copied = bundle_slice.copy_package_slice(campaign.src_pkg, paths.pkg, markers, "serving")
    shutil.copyfile(s.module_path, paths.pkg / "features" / "preprocess.py")
    for name, doc in (("netspec.json", case.spec), ("schema.json", case.schema),
                      ("recipe.json", s.recipe)):
        (paths.root / name).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    shutil.copyfile(s.dir / "fitted.json", paths.root / "fitted.json")
    ident = serving_pack.identity(options, options.pipeline, case.schema)
    return PackContext(options=options, paths=paths, campaign=campaign, package=package,
                       ident=ident, schema=case.schema, copied=tuple(copied), markers=markers)


def build(reading="binary", package="s07_served", mode="batch", flags=("--serve",)) -> Built:
    """A served bundle of one row reading (see the module docstring)."""
    case = pf.row_case(reading)
    ctx = context(case, package, mode, flags)
    with contextlib.redirect_stdout(io.StringIO()):
        serving_pack.render_adapters(ctx)
        version = _weights(ctx, case)
        _stub(ctx, case, version)
        _examples(ctx, case)
        _monitor(ctx, case)
        record = serving_pack.render(ctx)["serve"]
        _request(ctx)
        bundle_slice.finish(ctx)
    manifest = _manifest(ctx, case, version, record)
    return Built(ctx.paths.root, ctx, record, manifest, case)


def _weights(ctx, case) -> str:
    """The case's network saved through the bundle's own store; the weights digest."""
    root, package = ctx.paths.root, ctx.package
    sys.path.insert(0, str(root))
    try:
        store = importlib.import_module(f"{package}.model.store")
        domain = importlib.import_module(f"{package}.domain")
        ref = store.ModelRegistry().save(case.model, root / "weights.pt",
                                         spec=domain.NetSpec.from_dict(case.spec))
    finally:
        sys.path.remove(str(root))
        for name in [m for m in sys.modules if m.split(".")[0] == package]:
            del sys.modules[name]
    return ref.weights_digest


def _stub(ctx, case, version) -> None:
    """The manifest the predictor loads against before the service is rendered, as
    `bundle_manifest.write_stub` writes it: the data pins, the output and sample records."""
    files = bundle_manifest.pinned_files(ctx.paths.root)
    stub = {"package": ctx.package, "files": files,
            "roles": bundle_manifest.file_roles(files, ctx.package), "model_version": version,
            "input_schema": "schema.json", **case.manifest}
    ctx.paths.manifest.write_text(json.dumps(stub, indent=1), encoding="utf-8")


def _examples(ctx, case) -> None:
    """examples/rows.csv and expected.parquet (`__row__`), and examples@2's demo and presets."""
    rows, expected = pf._row_examples(case)
    ctx.paths.examples.mkdir(parents=True, exist_ok=True)
    rows.to_csv(ctx.paths.examples / "rows.csv", index=False)
    expected.to_parquet(ctx.paths.examples / "expected.parquet", index=False)
    train = case.shaped.rows("train").head(DEMO_ROWS)
    inputs = [c["name"] for c in case.schema["input"]]
    items = [{"stratum": "all", "rows": [_json_row(r, inputs)], "history": None, "truth": None}
             for r in train.to_dict(orient="records")]
    demo = {"source": "train", "schema": "vega/demo@2", "items": items,
            "strata": [{"id": "all", "kind": "all", "name": None, "n": len(items), "range": None}]}
    presets = {"schema": "vega/presets@2", "presets": [
        {"id": "edge-range-x0", "kind": "edge_range", "group": "edge", "stratum": "all",
         "expected": None, "facts": {"column": "x0", "factor": 10}, "row": items[0]["rows"][0],
         "rows": None, "history": None}]}
    for name, doc in (("demo.json", demo), ("presets.json", presets)):
        (ctx.paths.examples / name).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def _json_row(row, columns) -> dict:
    out = {}
    for name in columns:
        value = row.get(name)
        out[name] = None if isinstance(value, float) and np.isnan(value) else \
            (value.item() if hasattr(value, "item") else value)
    return out


def _monitor(ctx, case) -> None:
    """A monitor reference (one psi check on x0, measured on the train part) and the Monitor
    adapter rendered into the Monitor port's file, as monitor_pack renders it."""
    port = ctx.port("Monitor")
    text = (pf.PACKAGE_TEMPLATES / "monitor.py").read_text(encoding="utf-8")
    for key, value in (("{{PACKAGE}}", ctx.package), ("{{MONITOR_MODULE}}", port["module"]),
                       ("{{PORT_MONITOR}}", port["name"]),
                       ("{{RESPONSIBILITY_MONITOR}}", port["responsibility"]),
                       ("{{SOURCE_MONITOR}}", f"blueprint:ports/{port['name']}")):
        text = text.replace(key, value)
    (ctx.paths.pkg / port["file"]).write_text(text, encoding="utf-8")
    train = case.shaped.rows("train")["x0"].to_numpy(dtype=float)
    edges = np.quantile(train, np.linspace(0, 1, 6))
    edges[0], edges[-1] = -1e12, 1e12
    counts = np.histogram(train, bins=edges)[0]
    doc = {"schema": "vega/monitor-reference@2", "concerns": ["[analysis:F3]"], "plan": "test",
           "checks": [{"id": "x0", "kind": "psi", "column": "x0", "bins": edges.tolist(),
                       "reference": (counts / counts.sum()).tolist(), "min_rows": 5,
                       "concern": "[analysis:F3]", "title": "x0 against train"}], "note": ""}
    (ctx.paths.root / "monitor-reference.json").write_text(json.dumps(doc, indent=1),
                                                           encoding="utf-8")


def _request(ctx) -> None:
    """examples/request.json: the first shipped row, encoded by the rendered contract's table
    codec — the service's `serve` package imported from the bundle, as a caller would."""
    import pandas as pd
    text = {c["name"]: "string" for c in ctx.schema["input"] if c.get("dtype") == "string"}
    first = pd.read_csv(ctx.paths.examples / "rows.csv", dtype=text, nrows=1)
    saved, sys.path[:0] = list(sys.path), [str(ctx.paths.root)]
    try:
        rows = importlib.import_module("serve.tables").TableCodec.json_rows(first)
    finally:
        sys.path[:] = saved
        for name in [m for m in sys.modules if m == "serve" or m.startswith("serve.")]:
            del sys.modules[name]
    (ctx.paths.examples / "request.json").write_text(json.dumps({"rows": rows}, indent=1),
                                                     encoding="utf-8")


def _manifest(ctx, case, version, record) -> dict:
    """MANIFEST.json as `bundle_manifest.build_manifest` writes it, over the files on disk."""
    files = bundle_manifest.pinned_files(ctx.paths.root)
    scores = [c for c in case.stored.columns if c.startswith("score") or c == "row_score"]
    doc = {"schema": "vega/bundle@1", "slug": "rows", "run": RUN, "campaign": "campaign",
           "package": ctx.package, "files": files, "model_version": version,
           "created_at": "2026-09-30T00:00:00Z", "input_schema": "schema.json",
           "label": case.spec["task"].get("label"), "include": "serving",
           "roundtrip": {"tolerance": Tolerance.for_scores(
               case.stored[scores].to_numpy().ravel().tolist()).as_json()},
           "examples": {"kind": "test", "rows": 12}, "serve": record,
           "entrypoints": serving_pack.served_entrypoints(ctx),
           "monitor": {"reference": "monitor-reference.json"}, **case.manifest}
    bundle_manifest.check_entrypoints(doc)
    doc["roles"] = bundle_manifest.file_roles(files, ctx.package)
    ctx.paths.manifest.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def served(bundle: Built):
    """The bundle's own `serve.app` module, imported from the bundle directory."""
    os.environ["VEGA_FITTED"] = str((bundle.root / "fitted.json").resolve())
    if str(bundle.root) not in sys.path:
        sys.path.insert(0, str(bundle.root))
    return importlib.import_module("serve.app")


def copy(bundle: Built, where: Path) -> Path:
    target = where / bundle.root.name
    shutil.copytree(bundle.root, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from fastapi.testclient import TestClient
from serve.app import create_app
with TestClient(create_app(sys.argv[1])) as c:
    out = {{}}
{body}
print(json.dumps(out))
"""


def probe(root: Path, body: str, env=None) -> dict:
    """`body` (indented statements filling `out` through `c`, a TestClient of the bundle at
    `root` served by `root`'s own code) in a clean interpreter; its `out`."""
    code = PROBE.format(body="\n".join("    " + line for line in body.strip().splitlines()))
    proc = subprocess.run([sys.executable, "-c", code, str(root)], capture_output=True,
                          text=True, timeout=600, cwd=str(root.parent),
                          env=dict(os.environ, **(env or {}), PYTHONDONTWRITEBYTECODE="1"))
    if proc.returncode != 0:
        raise AssertionError(f"the probe failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])
