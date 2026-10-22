#!/usr/bin/env python3
"""
serving_matrix.py — the serving acceptance (BUILD-PLAN S13, D2 §10.2 cells M01–M16 plus M17):
every reading and every sample shape carried from the real shaper to a served bundle whose own
tests pass, with every family of the catalog swept.

    python3 serving_matrix.py [<work>] [--cells all|M01,M13,…] [--families <glob>[,<glob>…]]
                              [--quick] [--jobs N] [--threads N] [--timeout S] [--keep]
                              [--no-inspect] [--json]

`<work>` defaults to `$VEGA_SERVING_MATRIX_WORK`, else a temporary directory removed at exit
(`--keep` keeps it). `--cells` picks cells by id; `--quick` runs one family per cell (the first
the table lists), the release gate runs them all (27 runs over the 21 families); `--families`
narrows the families of the chosen cells by glob. Each run is (cell, family):

  fixture      (in this process, before any worker) the raw table of the backend matrix
               (`matrix/make_matrix_fixtures.py`, seed 7), shaped by the REAL
               `dataset-shaper/scripts/shape.py` — the matrix's recipe (retype, drop the
               identities, select_partition, drop the other labels, split, encode, scale), with a
               `lag` step of an exogenous column (and its impute) in every window cell and the
               `graph_edges` step in every graph cell — and the cell's netspec composed over it
               and checked by `compile_spec`; M05 carries the quantile loss (0.1, 0.5, 0.9). A
               family whose blocks `coverage.json` still lists as planned (or does not list as
               built) fails here, naming them; so does a spec the compiler refuses.
  compile      `compile_spec` again, in the worker, over the written spec
  emit         the REAL emitter (`emit_backend.py --smoke`: `emit.py` writes the v2.3 package)
  smoke        the package's one forward pass (`smoke.json forward_ok`)
  conformance  C0–C10 pass on the emitted package (C11–C14 recorded, never failing a run)
  fit          a seeded model: the package's own `Trainer().fit` for the spec's 3 epochs, the
               checkpoint through its ModelStore port (the trainer-held state — the SVDD centre,
               the regression head's train-fitted target affine — is the one the chain makes)
  campaign     the campaign /vega-train leaves (`roundtrip_fixtures._ledger`: campaign.json,
               the package, runs/R0 with its checkpoint, experiments.json with the selection)
  evaluate     the REAL `vega-eval/scripts/evaluate.py` under its seal: it writes
               `predictions.test.parquet` in `vega/predictions-layout@1` through the emitted
               evaluator (checked here: the layout key, `row_id`, the score columns)
  pack         the REAL `pack.py --serve --unattended --demo-sample 5 --monitor-plan <plan>`
               (the harness's plan: a PSI on a raw input column and the reading's own answer
               check — class recall, label rate, alert rate or residual — built from the output
               record `output_record.py` resolves); exit 0, the round trip `pass`, the reading,
               sample shape and level the cell names, a served bundle with a monitor; every
               `[project.scripts]` target in its pyproject.toml names a shipped module and a
               callable it defines
  bundle       `predict.py --check`, then `pytest <bundle>/tests` in an emptied environment
               (the skills unreachable); every test passes, or skips with a reason the cell
               allows (lanes.json `allowed_skips`, never a `forbidden_skip_patterns` one, and not
               the monitor's: every cell ships one; `--no-inspect` adds its own reason)
  stream       (M13 only) the same campaign packed against the blueprint with serving mode
               `stream` (no `--serve`): `serve/consumer.py` is the adapter, `serve/app.py` is
               absent and so is any installed command pointing at it (a serve command, if
               any, is `serve.consumer:main`); the test part's raw
               rows as JSON lines through `python -m serve.consumer` in batches shorter than a
               window answer row for row as one batch holding the whole stream does (the history
               buffer completes windows across batches), and only a prefix of each series (at
               least `length − 1`, at most `history_rows` rows) is left unanswered

A stage runs only when every earlier one passed. `<work>/serving-matrix.json`
(`vega/serving-matrix@1`):

    {"schema": "vega/serving-matrix@1", "selection": {"cells", "families", "quick"},
     "runs": [{"id": "M13/gru", "cell", "family", "task", "reading", "shape", "level",
               "matrix_cell", "stages": {"fixture": "ok", …}, "bundle", "tests": {"passed",
               "failed", "skipped", "skips": [reason, …]}, "roundtrip", "stream", "seconds",
               "errors": [{"stage", "message"}]}, …],
     "coverage": {"families": [...], "missing": [...]},
     "summary": {"runs", "ok", "failed": [ids], "cells": {"M01": "ok", …}}}

Green is `summary.failed == []`. Exit codes: 0 green; 1 a run failed; 2 the arguments, the table
or the fixtures could not be used.
"""
from __future__ import annotations

import argparse
import ast
import atexit
import copy
import fnmatch
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK_SCRIPTS = HERE.parent
SKILLS = PACK_SCRIPTS.parent.parent
FORGE_SCRIPTS = SKILLS / "vega-forge" / "scripts"
FORGE_TESTS = FORGE_SCRIPTS / "tests"
MATRIX = FORGE_TESTS / "matrix"
PACK_PY = PACK_SCRIPTS / "pack.py"
LANES = SKILLS / "vega" / "scripts" / "tests" / "lanes.json"
COVERAGE = FORGE_SCRIPTS / "backends" / "pytorch" / "coverage.json"
for _p in (str(HERE), str(PACK_SCRIPTS), str(MATRIX), str(FORGE_TESTS), str(FORGE_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPORT_SCHEMA = "vega/serving-matrix@1"
RUN_SCHEMA = "vega/serving-matrix-run@1"
LAYOUT = "vega/predictions-layout@1"
STANCE = "the serving matrix packs a 3-epoch seeded network; the verdict is not its claim"
DEMO_SAMPLE = 5
STAGES = ("compile", "emit", "smoke", "conformance", "fit", "campaign", "evaluate", "pack",
          "bundle")
STREAM_STAGE = "stream"
STREAM_BATCH = 7               # rows per consumer batch: shorter than any matrix window
QUANTILES = (0.1, 0.5, 0.9)
LAG = {"column": "x1", "lags": [1, 2], "group_by": "series", "time_column": "ts"}
NO_INSPECT_SKIP = "packed with --no-inspect"
MONITOR_SKIP = "monitor reference"


def say(line: str) -> None:
    print(line, flush=True)


class MatrixError(RuntimeError):
    """The table, the arguments or a fixture could not be used; the message says which."""


class StageFailed(Exception):
    """A stage's check did not hold; the message says what was seen."""


# ================================================================ the table (D2 §10.2)

@dataclass(frozen=True)
class ServingCell:
    """One serving cell: the task the backend matrix names it by (`make_matrix_fixtures`
    spelling), its sample, the reading pack must resolve, and the families it sweeps (the first
    is `--quick`'s). `loss` overrides the task row's loss (M05); `stream` adds the stream stage
    (M13)."""

    id: str
    task: str
    shape: str
    level: str | None
    reading: str
    families: tuple
    what: str
    loss: str | None = None
    stream: bool = False

    @property
    def shape_tag(self) -> str:
        return f"graph-{self.level}" if self.shape == "graph" else self.shape

    @property
    def variant(self) -> str:
        return {"row": "nominals", "window": "stride-1", "graph-node": "inductive-full",
                "graph-graph": "graphs"}[self.shape_tag]

    def matrix_id(self, family: str) -> str:
        """The backend matrix's cell this run composes its fixture from."""
        return "/".join((family, self.shape_tag, self.task.replace("/", ":"), self.variant))


CELLS = (
    ServingCell("M01", "binary", "row", None, "binary", ("mlp-plain",), "binary"),
    ServingCell("M02", "multiclass", "row", None, "multiclass", ("mlp-residual", "linear"),
                "multiclass (with embedding)"),
    ServingCell("M03", "multilabel", "row", None, "multilabel", ("ft-transformer",),
                "multilabel"),
    ServingCell("M04", "regression", "row", None, "regression", ("mlp-plain",), "regression"),
    ServingCell("M05", "regression", "row", None, "quantile", ("mlp-plain",),
                "regression (quantile loss)", loss="quantile"),
    ServingCell("M06", "ordinal", "row", None, "ordinal", ("mlp-plain",), "ordinal"),
    ServingCell("M07", "unsupervised", "row", None, "reconstruction",
                ("autoencoder", "vae", "pca-floor"), "unsupervised"),
    ServingCell("M08", "anomaly/labelled", "row", None, "anomaly-reconstruction",
                ("autoencoder",), "anomaly (labelled)"),
    ServingCell("M09", "anomaly/unlabelled", "row", None, "anomaly-svdd", ("deep-svdd",),
                "anomaly (unlabelled)"),
    ServingCell("M10", "sequence-classification/multiclass", "window", None, "multiclass",
                ("gru", "cnn1d", "rnn", "bi-gru", "gru-attention"), "sequence-classification"),
    ServingCell("M11", "sequence-classification/binary", "window", None, "binary",
                ("lstm", "bi-lstm"), "sequence-classification (binary base)"),
    ServingCell("M12", "sequence-regression/regression", "window", None, "regression",
                ("transformer-seq", "lstm-attention"), "sequence-regression"),
    ServingCell("M13", "forecast", "window", None, "forecast", ("gru",), "forecast (H=3)",
                stream=True),
    ServingCell("M14", "binary", "graph", "node", "binary", ("gcn", "graphsage"),
                "binary (node)"),
    ServingCell("M15", "regression", "graph", "node", "regression", ("gat",),
                "regression (node)"),
    ServingCell("M16", "multiclass", "graph", "graph", "multiclass", ("gin",),
                "multiclass (graph level)"),
    ServingCell("M17", "sequence-classification/multilabel", "window", None, "multilabel",
                ("bi-lstm",), "sequence-classification (multilabel base)"),
)
BY_ID = {c.id: c for c in CELLS}


@dataclass(frozen=True)
class RunPlan:
    """One (cell, family) run as the parent planned it."""

    cell: str
    family: str

    @property
    def id(self) -> str:
        return f"{self.cell}/{self.family}"

    @property
    def slug(self) -> str:
        return f"{self.cell}__{self.family}"


def select(cells_arg: str, families_arg: str | None, quick: bool) -> list:
    """The runs asked for: the cells by id (`all` or a comma list), each with its families (one
    under `--quick`), narrowed by `--families` globs; nothing selected is refused."""
    if cells_arg == "all":
        chosen = list(CELLS)
    else:
        ids = [c.strip().upper() for c in cells_arg.split(",") if c.strip()]
        unknown = [i for i in ids if i not in BY_ID]
        if unknown:
            raise MatrixError(f"--cells names unknown cell(s) {unknown} (the cells are "
                              f"{', '.join(BY_ID)})")
        chosen = [BY_ID[i] for i in dict.fromkeys(ids)]
    globs = [g.strip() for g in (families_arg or "").split(",") if g.strip()]
    runs = []
    for cell in chosen:
        families = cell.families[:1] if quick else cell.families
        if globs:
            families = tuple(f for f in cell.families
                             if any(fnmatch.fnmatchcase(f, g) for g in globs))
            families = families[:1] if quick else families
        runs += [RunPlan(cell.id, f) for f in families]
    if not runs:
        raise MatrixError(f"--cells {cells_arg} --families {families_arg} selects no run")
    return runs


def table_coverage(catalog_families) -> dict:
    """The families the table sweeps against the catalog's (D2: all 21 covered)."""
    swept = sorted({f for c in CELLS for f in c.families})
    return {"families": swept, "missing": sorted(set(catalog_families) - set(swept)),
            "unknown": sorted(set(swept) - set(catalog_families))}


# ================================================================ the fixtures (the parent)

def _lag_steps(steps: list) -> list:
    """The matrix recipe with a `lag` of an exogenous column (and the impute its first rows
    need) after the split, the lag columns scaled with the numerics; ids renumbered."""
    at = next(i for i, s in enumerate(steps) if s["op"] == "split") + 1
    names = [f"{LAG['column']}_lag{k}" for k in LAG["lags"]]
    lag = {"id": "", "op": "lag", "columns": [LAG["column"]],
           "params": {"lags": list(LAG["lags"]), "group_by": LAG["group_by"],
                      "time_column": LAG["time_column"]},
           "source": "user:matrix/lag",
           "rationale": "the exogenous cycle's recent past, per series (R-A6's lag boundary)",
           "alternatives": [], "consequences": {"rows": "none", "columns": "none",
                                                "downstream": {}}, "reversible": True}
    impute = {**copy.deepcopy(lag), "op": "impute", "columns": list(names),
              "params": {"strategy": "median"},
              "rationale": "a lag is empty on the first rows of each series in each part"}
    out = [*steps[:at], lag, impute, *steps[at:]]
    for step in out:
        if step["op"] == "scale":
            step["columns"] = [*step["columns"], *names]
            step["params"]["columns"] = [*step["params"]["columns"], *names]
    for n, step in enumerate(out, 1):
        step["id"] = f"S{n}"
    return out


def shape_variant(mmf, work: Path, variant, table: dict, lag: bool) -> Path:
    """shaped/<variant>[-lag]/ by the real shape.py (make_matrix_fixtures' stamp: shaped again
    only when the recipe, the raw table or shape.py changed)."""
    if not lag:
        return mmf.shape_variant(work, variant, table)
    csv = work / "raw" / f"{variant.data}.csv"
    shaped = work / "shaped" / f"{variant.key}-lag"
    recipe = mmf._recipe(csv, variant, shaped, table)
    recipe["steps"] = _lag_steps(recipe["steps"])
    text = json.dumps(recipe, indent=1)
    stamp = {"recipe": mmf._sha256_bytes(text.encode()), "raw": mmf._sha256(csv),
             "shaper": mmf._sha256(mmf.SHAPER)}
    stamp_path = shaped / "stamp.json"
    if stamp_path.is_file() and json.loads(stamp_path.read_text()) == stamp \
            and (shaped / "manifest.json").is_file():
        return shaped
    shutil.rmtree(shaped, ignore_errors=True)
    shaped.mkdir(parents=True, exist_ok=True)
    (shaped / "recipe.in.json").write_text(text, encoding="utf-8")
    mmf._run([mmf.SHAPER, "--recipe", shaped / "recipe.in.json", "--out-dir", shaped,
              "--format", "parquet"], f"shape.py ({variant.key}-lag)")
    stamp_path.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    return shaped


def _block_types(blocks) -> set:
    """Every block type a block list holds, nested ones included: any list of blocks under a
    block's params (`inner`, `blocks`, `branches`, ...) is followed, not only `inner`."""
    out = set()
    for block in blocks or ():
        if not isinstance(block, dict):
            continue
        if block.get("type") is not None:
            out.add(block.get("type"))
        for value in (block.get("params") or {}).values():
            out |= _block_types(_nested_blocks(value))
    return out


def _nested_blocks(value) -> list:
    """The blocks a params value holds: a block, a list of blocks, or a dict of either (a
    named-branch map)."""
    if isinstance(value, dict):
        if "type" in value:
            return [value]
        return [b for v in value.values() for b in _nested_blocks(v)]
    if isinstance(value, list):
        return [b for v in value for b in _nested_blocks(v)]
    return []


def planned_blocks(spec: dict) -> list:
    """The spec's blocks the backend does not build yet: the block types (and the family)
    `coverage.json` lists under `planned`, and any block type its `blocks` does not hold (D2:
    such a cell fails naming them, it never skips)."""
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    planned = coverage.get("planned") or {}
    if not isinstance(planned, dict):
        planned = {"blocks": list(planned)}
    names = {_named(p) for p in planned.get("blocks") or ()}
    built = {_named(b) for b in coverage.get("blocks") or ()}
    types = _block_types(spec.get("blocks"))
    out = sorted(t for t in types if t in names or (built and t not in built))
    family = (spec.get("family") or {}).get("name")
    if family in {_named(f) for f in planned.get("families") or ()}:
        out.insert(0, f"family {family}")
    return out


def _named(entry) -> str | None:
    return entry if isinstance(entry, str) else \
        (entry or {}).get("type") or (entry or {}).get("block") or (entry or {}).get("name")


def quantile_spec(mmf, cell, spec: dict, manifest: dict, pipeline: dict) -> dict:
    """M05: the regression spec under the quantile loss (head out = one channel a quantile),
    checked again by the compiler."""
    spec = copy.deepcopy(spec)
    head = next(b for b in reversed(spec["blocks"]) if b["type"] == "head")
    head["params"]["out"] = len(QUANTILES)
    head["rationale"] = (f"out {len(QUANTILES)}: one channel per quantile of the loss "
                         f"({', '.join(map(str, QUANTILES))})")
    loss = spec["loss"]
    spec["loss"] = {"name": "quantile", "params": {"quantiles": list(QUANTILES)},
                    "source": "user:matrix/quantile",
                    "rationale": "the serving matrix's quantile cell (D2 §10.2 M05): the "
                                 "reading answers three quantiles of the label",
                    "alternatives": [{"name": loss["name"], "when": "a point regression"}]}
    mmf.compile_check(cell, spec, manifest, pipeline)
    return spec


def make_fixtures(work: Path, runs: list, jobs: int) -> tuple:
    """({run id: run directory}, {run id: why its fixture could not be made}): raw tables,
    shaped variants, one compiled netspec per run. A run whose blocks the backend still plans,
    or whose spec the compiler refuses, is a failed run naming why — never a skip, and never
    a reason to stop the others."""
    import make_matrix_fixtures as mmf
    table = mmf.load_table()
    matrix = {c.id: c for c in mmf.cells(table)}
    wanted = {}
    for run in runs:
        cell = BY_ID[run.cell]
        mid = cell.matrix_id(run.family)
        if mid not in matrix:
            raise MatrixError(f"{run.id}: the backend matrix has no cell {mid} (the catalog's "
                              "admission table does not admit it)")
        wanted[run.id] = (run, cell, matrix[mid])
    variants = {(mmf.variant_of(m, table), c.shape == "window") for _, c, m in wanted.values()}
    for data in sorted({v.data for v, _ in variants}):
        mmf.write_raw(work, data, int(table["seed"]))
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        shaped = dict(zip(variants, pool.map(
            lambda vl: shape_variant(mmf, work, vl[0], table, vl[1]), variants)))
    for (variant, _), path in shaped.items():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        feats = mmf.features(manifest, mmf.DATASETS[variant.data].keys)
        mmf.pipeline_for(path, manifest, {"input_dim": len(feats), "label": variant.label})
    out, failures = {}, {}
    for rid, (run, cell, mcell) in wanted.items():
        root = work / "runs" / run.slug
        shutil.rmtree(root, ignore_errors=True)
        path = shaped[(mmf.variant_of(mcell, table), cell.shape == "window")]
        epochs = int(table["epochs"])
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        pipeline = json.loads((path / "pipeline.json").read_text(encoding="utf-8"))
        try:
            missing = planned_blocks(mmf.compose(mcell, path, path / "pipeline.json", table,
                                                 epochs))
            if missing:
                raise MatrixError(f"the {run.family} family needs what the backend does not "
                                  f"build (coverage.json): {', '.join(missing)}")
            made = mmf._write_cell(root, mcell, path, table, "cell", epochs)
            spec_path = Path(made.spec)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if cell.loss == "quantile":         # against the cell's own hand-off, the one
                cell_pipeline = json.loads((root / "pipeline.json")      # its spec pins
                                           .read_text(encoding="utf-8"))
                spec = quantile_spec(mmf, mcell, spec, manifest, cell_pipeline)
                spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        except (MatrixError, mmf.MatrixError) as exc:
            failures[rid] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001 — one run's fixture crash fails that run only
            tail = traceback.format_exc().strip().splitlines()[-4:]
            failures[rid] = (f"the {run.family} family's fixture could not be made: "
                             f"{type(exc).__name__}: {exc} [{' / '.join(tail)}]")
            continue
        fx = json.loads((root / "fixture.json").read_text(encoding="utf-8"))
        fx["serving"] = {"schema": RUN_SCHEMA, "run": run.id, "cell": cell.id,
                         "family": run.family, "task": cell.task, "reading": cell.reading,
                         "shape": cell.shape, "level": cell.level, "stream": cell.stream,
                         "matrix_cell": mcell.id, "lag": cell.shape == "window",
                         "psi_column": mmf.DATASETS[mcell.data].numeric[0]}
        if cell.stream:
            stream = copy.deepcopy(pipeline)
            stream.setdefault("serving", {})["mode"] = "stream"
            (root / "pipeline.stream.json").write_text(json.dumps(stream, indent=2) + "\n",
                                                        encoding="utf-8")
            fx["serving"]["stream_pipeline"] = str(root / "pipeline.stream.json")
        (root / "fixture.json").write_text(json.dumps(fx, indent=2) + "\n", encoding="utf-8")
        out[rid] = root
    return out, failures


# ================================================================ the worker: one run

def _allowed_skips(no_inspect: bool) -> tuple:
    """(allowed, forbidden) skip-reason fragments: the lane registry's policy, less the
    monitor's reason (every matrix bundle ships a monitor), plus --no-inspect's own."""
    defaults = json.loads(LANES.read_text(encoding="utf-8"))["defaults"]
    allowed = [a for a in defaults["allowed_skips"] if a not in (MONITOR_SKIP, NO_INSPECT_SKIP)]
    if no_inspect:
        allowed.append(NO_INSPECT_SKIP)
    return tuple(allowed), tuple(defaults["forbidden_skip_patterns"])


def _run(argv, cwd=None, env=None, timeout=1500, stdin=None) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, cwd=cwd,
                          env=env or dict(os.environ), timeout=timeout, input=stdin)


def _tail(text: str, n: int = 12) -> str:
    return " | ".join(text.strip().splitlines()[-n:])


def _errors(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.startswith(("ERROR", "FAIL"))]
    return " | ".join(lines)[:3000]


class ServingRun:
    """One (cell, family) run through its stages, in one process."""

    def __init__(self, run_dir: Path, no_inspect: bool = False):
        import run_matrix
        self.dir = Path(run_dir)
        self.fx = json.loads((self.dir / "fixture.json").read_text(encoding="utf-8"))
        self.sv = self.fx["serving"]
        self.no_inspect = no_inspect
        self.core = run_matrix.CellRun(self.dir)         # compile, emit, smoke, conformance, fit
        self.camp = self.dir / "campaign"
        self.bundle = None
        self.output = None
        self.record = {"schema": RUN_SCHEMA, "id": self.sv["run"], "cell": self.sv["cell"],
                       "family": self.sv["family"], "task": self.sv["task"],
                       "reading": self.sv["reading"], "shape": self.sv["shape"],
                       "level": self.sv["level"], "matrix_cell": self.sv["matrix_cell"],
                       "stages": {"fixture": "ok"}, "errors": [], "stage_seconds": {},
                       "dir": str(self.dir)}

    # -- the backend matrix's stages, reused
    def compile(self):
        self.core.compile()

    def emit(self):
        self.core.emit()

    def smoke(self):
        self.core.smoke()

    def conformance(self):
        self.core.conformance()
        self.record["size_counts"] = (self.core.record.get("size") or {}).get("counts")

    def fit(self):
        self.core.fit_stage()
        self.record["valid_primary"] = self.core.record.get("valid_primary")

    # -- the chain's stages
    def campaign(self):
        """The campaign /vega-train leaves for one selected run (roundtrip_fixtures' ledger)."""
        import roundtrip_fixtures as rf
        package = self.fx["package"]
        shutil.rmtree(self.camp, ignore_errors=True)
        run_dir = self.camp / "runs" / rf.RUN
        (run_dir / "checkpoints").mkdir(parents=True)
        shutil.copytree(self.core.package_dir, self.camp / package,
                        ignore=shutil.ignore_patterns("__pycache__"))
        spec_text = Path(self.fx["spec"]).read_text(encoding="utf-8")
        (self.camp / "netspec.json").write_text(spec_text, encoding="utf-8")
        (run_dir / "netspec.resolved.json").write_text(spec_text, encoding="utf-8")
        (self.camp / "campaign.json").write_text(json.dumps(
            {"schema": "vega/campaign@1", "input": {"package": package, "spec": "netspec.json"},
             "backend": {"name": "pytorch"}}, indent=2) + "\n", encoding="utf-8")
        fit = self.core.fit
        shutil.copy2(getattr(fit.checkpoint, "path", fit.checkpoint),
                     run_dir / "checkpoints" / "best.pt")
        best = fit.best_valid.as_dict().get(self.core.primary) if fit.best_valid else None
        rf._ledger(self.camp, package, {"epochs": int(fit.epochs_run),
                                        "best_epoch": rf._plain(fit.best_epoch),
                                        "best_value": rf._plain(best)})

    def evaluate(self):
        import pyarrow.parquet as pq
        import roundtrip_fixtures as rf
        done = rf.evaluate(self.camp)
        said = done.stdout + done.stderr
        (self.dir / "evaluate.log").write_text(said, encoding="utf-8")
        if done.returncode != 0:
            raise StageFailed(f"evaluate.py exited {done.returncode}: "
                              f"{_errors(said) or _tail(said)}")
        stored = self.camp / "eval" / rf.RUN / "predictions.test.parquet"
        if not stored.is_file():
            raise StageFailed("evaluate.py wrote no predictions.test.parquet")
        table = pq.read_table(stored)
        meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
        if meta.get("vega.predictions_layout") != LAYOUT:
            raise StageFailed(f"the stored predictions are not {LAYOUT} "
                              f"(metadata {sorted(meta)})")
        names = table.schema.names
        if "row_id" not in names or not any(n == "score" or n.startswith("score_")
                                            for n in names):
            raise StageFailed(f"the stored predictions lack row_id or a score column: {names}")
        self.record["stored"] = {"rows": table.num_rows,
                                 "row_id_kind": meta.get("vega.row_id_kind"),
                                 "task_effective": meta.get("vega.task_effective")}

    def monitor_plan(self) -> Path:
        """The harness's monitor plan: a PSI on a raw input column, and the reading's own check
        on the answer — built from the output record pack itself resolves."""
        import output_record
        import roundtrip_fixtures as rf
        spec = json.loads(Path(self.fx["spec"]).read_text(encoding="utf-8"))
        threshold_path = self.camp / "eval" / rf.RUN / "threshold.json"
        threshold = json.loads(threshold_path.read_text(encoding="utf-8")) \
            if threshold_path.is_file() else None
        label = (spec.get("task") or {}).get("label")
        output = output_record.output_record(spec, threshold, label)
        self.output = output
        pipeline = json.loads(Path(self.fx["pipeline"]).read_text(encoding="utf-8"))
        concerns = [m.get("source") for m in pipeline.get("monitoring") or []
                    if isinstance(m.get("source"), str)] or ["[serving-matrix:input-drift]"]
        # every blueprint concern answered by a check (pack refuses a plan that skips one)
        checks = [{"id": f"drift-input-{n}", "kind": "psi", "column": self.sv["psi_column"],
                   "concern": concern,
                   "title": f"{self.sv['psi_column']} distribution against train"}
                  for n, concern in enumerate(concerns, 1)]
        checks += [dict(c, concern="[serving-matrix:answer]") for c in _answer_checks(output)]
        plan = {"note": f"the serving matrix's plan for {self.sv['run']}", "checks": checks}
        path = self.dir / "monitor-plan.json"
        path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        self.record["monitor_checks"] = [c["kind"] for c in checks]
        return path

    def _pack(self, bundle_dir: Path, pipeline: str, serve: bool, plan: Path):
        shutil.rmtree(bundle_dir, ignore_errors=True)
        argv = [sys.executable, PACK_PY, "--campaign", self.camp, "--shaped",
                self.fx["shaped"], "--bundle-dir", bundle_dir, "--pipeline", pipeline,
                "--ship-anyway", STANCE, "--unattended", "--demo-sample", DEMO_SAMPLE,
                "--monitor-plan", plan]
        argv += ["--serve"] if serve else []
        argv += ["--no-inspect"] if self.no_inspect else []
        done = _run(argv)
        text = done.stdout + done.stderr
        if done.returncode != 0:
            raise StageFailed(f"pack.py exited {done.returncode}: {_errors(text) or _tail(text)}")
        bundles = sorted(p for p in bundle_dir.glob("*") if (p / "MANIFEST.json").is_file())
        if len(bundles) != 1:
            raise StageFailed(f"pack.py wrote {len(bundles)} bundle(s) under {bundle_dir}")
        return bundles[0], text

    def pack(self):
        plan = self.monitor_plan()
        self.bundle, text = self._pack(self.dir / "bundle", self.fx["pipeline"], True, plan)
        (self.dir / "pack.log").write_text(text, encoding="utf-8")
        manifest = json.loads((self.bundle / "MANIFEST.json").read_text(encoding="utf-8"))
        self.record["bundle"] = str(self.bundle)
        self.record["roundtrip"] = {k: (manifest.get("roundtrip") or {}).get(k)
                                    for k in ("status", "rows", "max_abs_diff")}
        problems = _manifest_problems(manifest, self.sv, serve_mode="online")
        problems += script_problems(self.bundle, "online")
        problems += _inspection_problems(manifest, self.no_inspect)
        problems += demo_problems(self.bundle, manifest)
        if problems:
            raise StageFailed("; ".join(problems))

    def bundle_tests(self):
        """`predict.py --check` and the bundle's own tests with the skills unreachable, under
        the cell's skip policy."""
        allowed, forbidden = _allowed_skips(self.no_inspect)
        home = Path(tempfile.mkdtemp(prefix="vega-serving-home-"))
        env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "HOME": str(home),
               "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2"),
               "MKL_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "2")}
        junit = self.dir / "bundle-tests.xml"
        try:
            check = _run([sys.executable, "predict.py", "--check"], cwd=self.bundle, env=env)
            if check.returncode != 0:
                raise StageFailed(f"predict.py --check exited {check.returncode}: "
                                  f"{_tail(check.stdout + check.stderr)}")
            done = _run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                         f"--junitxml={junit}", "tests/"], cwd=self.bundle, env=env)
        finally:
            shutil.rmtree(home, ignore_errors=True)
        (self.dir / "bundle-tests.log").write_text(done.stdout + done.stderr, encoding="utf-8")
        tally = junit_tally(junit)
        self.record["tests"] = tally
        if tally["collected"] == 0:
            raise StageFailed(f"pytest collected no test in {self.bundle}/tests: "
                              f"{_tail(done.stdout + done.stderr)}")
        if tally["failed"]:
            raise StageFailed(f"{len(tally['failed'])} bundle test(s) failed: "
                              + "; ".join(tally["failed"][:4]))
        bad = skip_violations(tally["skips"], allowed, forbidden)
        if bad:
            raise StageFailed(f"{len(bad)} skip(s) outside the cell's policy: {bad[:4]}")
        ran = {m: tally["modules"].get(m, 0) for m in required_modules(self.no_inspect)}
        absent = [m for m, n in ran.items() if not n]
        if absent:
            raise StageFailed(f"no test of {', '.join(absent)} passed in the bundle "
                              f"(passed per module: {tally['modules']})")
        if done.returncode != 0:
            raise StageFailed(f"pytest exited {done.returncode}: {_tail(done.stdout)}")

    def stream(self):
        """M13: the stream blueprint's consumer, JSON lines in and out, windows completed
        across batches by the history buffer."""
        plan = self.dir / "monitor-plan.json"
        bundle, text = self._pack(self.dir / "bundle-stream", self.sv["stream_pipeline"],
                                  False, plan)
        (self.dir / "pack-stream.log").write_text(text, encoding="utf-8")
        manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
        problems = _manifest_problems(manifest, self.sv, serve_mode="stream")
        if not (bundle / "serve" / "consumer.py").is_file():
            problems.append("the stream bundle carries no serve/consumer.py")
        if (bundle / "serve" / "app.py").is_file():
            problems.append("the stream bundle carries the online app (serve/app.py)")
        problems += script_problems(bundle, "stream")
        if problems:
            raise StageFailed("; ".join(problems))
        rows, series = stream_rows(self.fx, bundle)
        small = consume(bundle, rows, STREAM_BATCH)
        whole = consume(bundle, rows, len(rows))
        self.record["stream"] = stream_verdict(small, whole, series, manifest["sample"],
                                               len(rows))

    # -- the driver
    def run(self) -> dict:
        steps = {"compile": self.compile, "emit": self.emit, "smoke": self.smoke,
                 "conformance": self.conformance, "fit": self.fit, "campaign": self.campaign,
                 "evaluate": self.evaluate, "pack": self.pack, "bundle": self.bundle_tests,
                 STREAM_STAGE: self.stream}
        names = STAGES + ((STREAM_STAGE,) if self.sv.get("stream") else ())
        started, failed = time.monotonic(), False
        for name in names:
            if failed:
                self.record["stages"][name] = "not_run"
                continue
            began = time.monotonic()
            try:
                steps[name]()
                self.record["stages"][name] = "ok"
            except (StageFailed, _stage_error()) as exc:
                failed = self._failed(name, str(exc))
            except Exception as exc:  # noqa: BLE001 — a run's crash is its stage's failure
                tail = traceback.format_exc().strip().splitlines()[-6:]
                failed = self._failed(name, f"{type(exc).__name__}: {exc} [{' / '.join(tail)}]")
            self.record["stage_seconds"][name] = round(time.monotonic() - began, 2)
        self.record["seconds"] = round(time.monotonic() - started, 2)
        return self.record

    def _failed(self, stage: str, message: str) -> bool:
        self.record["stages"][stage] = "failed"
        self.record["errors"].append({"stage": stage, "message": message[:4000]})
        print(f"ERROR: {self.sv['run']}: {stage}: {message}", flush=True)
        return True


def _stage_error():
    """run_matrix's own stage failure, which its reused stages raise."""
    import run_matrix
    return run_matrix.StageFailed


def _answer_checks(output: dict) -> list:
    """The reading's own check on the answer, as a plan names it (fields left to their
    defaults): class recall on a class truth, label rates on labels, the residual on a number
    truth, the alert rate where a threshold decides the alarm."""
    reading, truth = output["reading"], output.get("truth") or {}
    classes = [str(c) for c in output.get("classes") or ()]
    decided = output.get("threshold") is not None
    out = []
    if truth.get("type") == "class" and classes and \
            (not reading.startswith("anomaly-") or decided):
        out.append({"id": "answer-recall", "kind": "class_recall", "classes": classes,
                    "label_column": truth.get("column"), "title": "recall per class"})
    if reading == "multilabel" and classes:
        out.append({"id": "answer-labels", "kind": "label_rate", "labels": classes,
                    "title": "each label's rate"})
    if truth.get("type") == "number" and reading in ("regression", "quantile", "forecast"):
        out.append({"id": "answer-residual", "kind": "residual",
                    "target_column": truth.get("column"), "title": "absolute error"})
    if reading.startswith("anomaly-") and decided:
        out.append({"id": "answer-alerts", "kind": "alert_rate", "title": "alarm rate"})
    return out


def script_problems(bundle: Path, serve_mode: str) -> list:
    """The bundle's installed commands: every `[project.scripts]` target in its pyproject.toml
    must name a module the bundle ships and a callable that module defines. In stream mode the
    serve command, if any, is the consumer (`serve.consumer:main`), never the online app."""
    path = bundle / "pyproject.toml"
    if not path.is_file():
        return [f"the {serve_mode} bundle carries no pyproject.toml"]
    try:
        scripts = (tomllib.loads(path.read_text(encoding="utf-8")).get("project") or {}) \
            .get("scripts") or {}
    except tomllib.TOMLDecodeError as exc:
        return [f"the {serve_mode} bundle's pyproject.toml does not parse: {exc}"]
    out = []
    for name, target in sorted(scripts.items()):
        module, _, attr = str(target).partition(":")
        where = f"[project.scripts] {name} = {target!r} in the {serve_mode} bundle"
        parts = module.strip().split(".")
        candidates = [bundle.joinpath(*parts).with_suffix(".py"),
                      bundle.joinpath(*parts) / "__init__.py"]
        source = next((c for c in candidates if c.is_file()), None)
        if source is None:
            out.append(f"{where}: the bundle ships no module {module.strip()!r}")
            continue
        attr = attr.strip().split(".")[0]
        if attr and attr not in _defined_names(source):
            out.append(f"{where}: {source.relative_to(bundle)} defines no {attr!r}")
        if serve_mode == "stream" and module.strip().startswith("serve.") \
                and module.strip() != "serve.consumer":
            out.append(f"{where}: a stream bundle's serve command is serve.consumer:main")
    return out


def _defined_names(source: Path) -> set:
    """The top-level names a module defines (functions, classes, assignments, imports)."""
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            out |= {t.id for t in targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return out


def _manifest_problems(manifest: dict, sv: dict, serve_mode: str) -> list:
    """What a packed bundle's MANIFEST must say for this cell."""
    out = []
    trip = manifest.get("roundtrip") or {}
    if trip.get("status") != "pass":
        out.append(f"round trip {trip.get('status')!r} ({trip.get('rows')} rows, worst "
                   f"{trip.get('max_abs_diff')})")
    reading = (manifest.get("output") or {}).get("reading")
    if reading != sv["reading"]:
        out.append(f"the bundle reads {reading!r}; cell {sv['cell']} is the {sv['reading']} "
                   "reading")
    sample = manifest.get("sample") or {}
    if sample.get("shape") != sv["shape"]:
        out.append(f"MANIFEST.sample.shape is {sample.get('shape')!r}, not {sv['shape']}")
    if sv["shape"] == "graph" and sample.get("level") != sv["level"]:
        out.append(f"MANIFEST.sample.level is {sample.get('level')!r}, not {sv['level']}")
    serve = manifest.get("serve") or {}
    if not serve.get("present") or serve.get("mode") != serve_mode:
        out.append(f"MANIFEST.serve is {serve.get('mode')!r} (present "
                   f"{serve.get('present')}), not {serve_mode}")
    if not manifest.get("monitor"):
        out.append("the bundle carries no monitor although a plan was given")
    return out


def _inspection_problems(manifest: dict, no_inspect: bool) -> list:
    """The inspection surface as asked: enabled and self-tested ready with its two routes, or
    switched off with neither route (`--no-inspect`)."""
    serve = manifest.get("serve") or {}
    inspection = serve.get("inspection") or {}
    routes = {"GET /v1/network", "POST /v1/explain"} & set(serve.get("endpoints") or [])
    if no_inspect:
        return [f"--no-inspect, and the bundle serves {sorted(routes)}"] if routes else []
    out = []
    if not inspection.get("enabled"):
        out.append("MANIFEST.serve.inspection is not enabled")
    ready = (inspection.get("self_test") or {}).get("ready")
    if not ready:
        out.append(f"the inspection self-test is not ready: "
                   f"{(inspection.get('self_test') or {}).get('why')}")
    if len(routes) != 2:
        out.append(f"the inspection routes served are {sorted(routes)}")
    return out


def demo_problems(bundle: Path, manifest: dict) -> list:
    """The demonstration sample and its presets were written (`--demo-sample`) and validate
    against `examples-2.schema.json`."""
    import jsonschema
    from vega_pack_core import CATALOG
    demo = (manifest.get("examples") or {}).get("demo") or {}
    files = {"demo": demo.get("file"), "presets_file": (demo.get("presets") or {}).get("file")}
    if not all(files.values()):
        return [f"MANIFEST.examples.demo names no demonstration sample or presets: {files}"]
    schema = json.loads((PACK_SCRIPTS.parent / CATALOG["examples"]["json_schema"])
                        .read_text(encoding="utf-8"))
    out = []
    for name, rel in files.items():
        path = bundle / rel
        if not path.is_file():
            out.append(f"{rel} is not in the bundle")
            continue
        validator = jsonschema.Draft202012Validator(
            {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{name}"})
        errors = [f"{list(e.absolute_path)}: {e.message}"
                  for e in validator.iter_errors(json.loads(path.read_text(encoding="utf-8")))]
        out += [f"{rel} does not validate against examples@2: {errors[:3]}"] if errors else []
    return out


# ================================================================ the bundle's own tests

def junit_tally(path: Path) -> dict:
    """{collected, passed, failed: [test: message], skipped, skips: [reason]} from pytest's
    JUnit XML."""
    tally = {"collected": 0, "passed": 0, "failed": [], "skipped": 0, "skips": [],
             "modules": {}}
    if not Path(path).is_file():
        return tally
    for case in ET.parse(path).getroot().iter("testcase"):
        tally["collected"] += 1
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        bad = case.find("failure") if case.find("failure") is not None else case.find("error")
        skip = case.find("skipped")
        if bad is not None:
            tally["failed"].append(f"{name}: {(bad.get('message') or '')[:300]}")
        elif skip is not None:
            tally["skipped"] += 1
            tally["skips"].append(_skip_reason(skip.get("message") or ""))
        else:
            tally["passed"] += 1
            module = (case.get("classname") or "").rsplit(".", 1)[-1]
            tally["modules"][module] = tally["modules"].get(module, 0) + 1
    return tally


def required_modules(no_inspect: bool) -> tuple:
    """The bundle test modules that must have passed tests in every cell: the round trip, the
    task-neutral contract, the reading's answers, the shapes, the edges, the closed world, the
    showcase slot (absent: no showcase is built here) and, unless --no-inspect, inspection."""
    base = ("test_roundtrip", "test_api_contract", "test_api_answers", "test_api_shapes",
            "test_api_edges", "test_api_closed_world", "test_api_showcase")
    return base if no_inspect else base + ("test_api_inspect",)


def _skip_reason(message: str) -> str:
    """pytest writes a skip as `Skipped: <reason>` (or the reason alone)."""
    return re.sub(r"^Skipped:\s*", "", message.strip())


def skip_violations(reasons, allowed, forbidden) -> list:
    """The skip reasons outside the policy: any forbidden fragment, or no allowed one."""
    bad = [r for r in reasons if any(p in r for p in forbidden)]
    return bad + [r for r in reasons if r not in bad and not any(a in r for a in allowed)]


# ================================================================ the stream stage

def stream_rows(fx: dict, bundle: Path) -> tuple:
    """(the test part's raw rows as JSON objects in raw order, restricted to the bundle's input
    columns; the series key of each)."""
    import pandas as pd
    schema = json.loads((bundle / "schema.json").read_text(encoding="utf-8"))
    columns = [c["name"] for c in schema.get("input") or []]
    raw = pd.read_csv(fx["raw"], dtype=str, keep_default_na=False)
    fitted = json.loads(Path(fx["fitted"]).read_text(encoding="utf-8"))
    test = sorted(int(i) for i in fitted["__split_raw__"]["test"])
    missing = [c for c in columns if c not in raw.columns]
    if missing:
        raise StageFailed(f"the raw table lacks the bundle's input column(s) {missing}")
    frame = raw.iloc[test][columns]
    series_by = (json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
                 .get("sample") or {}).get("series_by")
    kinds = {c["name"]: c for c in schema.get("input") or []}
    rows = [{k: _typed(v, kinds.get(k) or {}) for k, v in rec.items()}
            for rec in frame.to_dict(orient="records")]
    series = frame[series_by].tolist() if series_by else ["" for _ in rows]
    return rows, series


def _typed(value: str, column: dict):
    """A raw CSV cell as JSON would carry it: a number where the schema reads one."""
    if value == "":
        return None
    dtype = str(column.get("dtype") or column.get("type") or "").lower()
    if any(t in dtype for t in ("float", "int", "number", "double")):
        try:
            number = float(value)
            return int(number) if "int" in dtype and number.is_integer() else number
        except ValueError:
            return value
    return value


def consume(bundle: Path, rows: list, batch: int) -> list:
    """The consumer's answers, one per row in stream order (`positions` from 1)."""
    lines = "".join(json.dumps(r) + "\n" for r in rows)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    done = _run([sys.executable, "-m", "serve.consumer", "--bundle", ".", "--batch-rows",
                 batch, "--monitor", "off"], cwd=bundle, env=env, stdin=lines)
    if done.returncode != 0:
        raise StageFailed(f"serve.consumer (batch {batch}) exited {done.returncode}: "
                          f"{_tail(done.stdout + done.stderr)}")
    answers, envelopes = {}, 0
    for line in done.stdout.splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if "predictions" not in doc:
            raise StageFailed(f"serve.consumer (batch {batch}) answered a problem line: "
                              f"{line[:400]}")
        envelopes += 1
        for pos, pred in zip(doc["positions"], doc["predictions"]):
            answers[int(pos)] = pred
    want = math.ceil(len(rows) / batch)
    if envelopes != want or sorted(answers) != list(range(1, len(rows) + 1)):
        raise StageFailed(f"serve.consumer (batch {batch}) wrote {envelopes} envelope(s) for "
                          f"{want} batch(es) and answered {len(answers)} of {len(rows)} rows")
    return [answers[p] for p in range(1, len(rows) + 1)]


def _same(a, b, tol: float = 1e-5) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def stream_verdict(small: list, whole: list, series: list, sample: dict, n: int) -> dict:
    """Batches shorter than a window answer as one batch does, row for row; the unanswered rows
    are a prefix of each series, of one length between `length − 1` and `history_rows`."""
    differ = [i + 1 for i, (a, b) in enumerate(zip(small, whole)) if not _same(a, b)]
    if differ:
        i = differ[0] - 1
        raise StageFailed(f"{len(differ)} row(s) answered differently in batches of "
                          f"{STREAM_BATCH} than in one batch (first at position {differ[0]}: "
                          f"{json.dumps(small[i])[:300]} vs {json.dumps(whole[i])[:300]})")
    answered = [a.get("answered") is not False for a in whole]
    if not any(answered):
        raise StageFailed("the consumer answered no row of the stream")
    length, most = int(sample["length"]), int(sample.get("history_rows") or sample["length"])
    prefixes, seen = {}, {}
    for key, ok in zip(series, answered):
        seen[key] = seen.get(key, 0) + 1
        if not ok:
            if prefixes.get(key, 0) != seen[key] - 1:
                raise StageFailed(f"series {key!r}: row {seen[key]} is unanswered after an "
                                  "answered one (not a cold-start prefix)")
            prefixes[key] = seen[key]
    counts = sorted(set(prefixes.get(k, 0) for k in seen))
    if len(counts) != 1 or not (length - 1 <= counts[0] <= most):
        raise StageFailed(f"unanswered prefixes per series {counts}; a cold start leaves "
                          f"between length-1 = {length - 1} and history_rows = {most}")
    return {"rows": n, "batch_rows": STREAM_BATCH, "answered": int(sum(answered)),
            "series": len(seen), "cold_prefix": counts[0]}


# ================================================================ the parent

def worker(run_dir: Path, no_inspect: bool) -> int:
    record = ServingRun(run_dir, no_inspect).run()
    (Path(run_dir) / "result.json").write_text(json.dumps(record, indent=2, default=str) + "\n",
                                               encoding="utf-8")
    return 0 if run_ok(record) else 1


def run_ok(rec: dict) -> bool:
    return bool(rec.get("stages")) and all(v == "ok" for v in rec["stages"].values()) \
        and not rec.get("errors")


def worker_env(threads: int) -> dict:
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def run_worker(plan: RunPlan, run_dir: Path, env: dict, timeout: float,
               no_inspect: bool) -> dict:
    (run_dir / "result.json").unlink(missing_ok=True)
    log = run_dir / "run.log"
    began = time.monotonic()
    argv = [sys.executable, str(Path(__file__).resolve()), "--worker", str(run_dir)]
    argv += ["--no-inspect"] if no_inspect else []
    with open(log, "w", encoding="utf-8") as fh:
        try:
            code = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = None
    result = run_dir / "result.json"
    if result.is_file():
        return json.loads(result.read_text(encoding="utf-8"))
    tail = _tail(log.read_text(encoding="utf-8", errors="replace"), 8)
    why = f"timed out after {timeout:g} s" if code is None else \
        f"the worker exited {code} without a result"
    cell = BY_ID[plan.cell]
    return {"schema": RUN_SCHEMA, "id": plan.id, "cell": plan.cell, "family": plan.family,
            "task": cell.task, "reading": cell.reading, "shape": cell.shape,
            "level": cell.level, "stages": {s: "not_run" for s in STAGES},
            "errors": [{"stage": "worker", "message": f"{why}: {tail}"}],
            "seconds": round(time.monotonic() - began, 2)}


def fixture_failure(plan: RunPlan, message: str) -> dict:
    """The record of a run whose fixture could not be made."""
    cell = BY_ID[plan.cell]
    return {"schema": RUN_SCHEMA, "id": plan.id, "cell": plan.cell, "family": plan.family,
            "task": cell.task, "reading": cell.reading, "shape": cell.shape,
            "level": cell.level, "stages": {"fixture": "failed",
                                            **{s: "not_run" for s in STAGES}},
            "errors": [{"stage": "fixture", "message": message}], "seconds": 0.0}


def run_all(runs: list, dirs: dict, failures: dict, jobs: int, env: dict, timeout: float,
            no_inspect: bool) -> list:
    def one(plan):
        rec = fixture_failure(plan, failures[plan.id]) if plan.id in failures else \
            run_worker(plan, dirs[plan.id], env, timeout, no_inspect)
        if run_ok(rec):
            tests = rec.get("tests") or {}
            say(f"OK: {rec['id']} ({rec.get('seconds', 0):.0f} s): reading {rec['reading']}, "
                f"round trip {(rec.get('roundtrip') or {}).get('status')} on "
                f"{(rec.get('roundtrip') or {}).get('rows')} row(s), bundle tests "
                f"{tests.get('passed')} passed, {tests.get('skipped')} allowed skip(s)"
                + (f", stream {rec['stream']['answered']}/{rec['stream']['rows']} answered in "
                   f"batches of {rec['stream']['batch_rows']}" if rec.get("stream") else ""))
        else:
            err = (rec.get("errors") or [{"stage": "?", "message": "?"}])[0]
            say(f"ERROR: {rec['id']}: {err['stage']}: {err['message'][:600]}")
        return rec

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return list(pool.map(one, runs))


def build_report(selection: dict, results: list) -> dict:
    import make_matrix_fixtures as mmf
    failed = [r["id"] for r in results if not run_ok(r)]
    cells = {}
    for r in results:
        cells[r["cell"]] = "ok" if run_ok(r) and cells.get(r["cell"], "ok") == "ok" else "failed"
    return {"schema": REPORT_SCHEMA, "selection": selection,
            "runs": [{k: v for k, v in r.items() if k != "schema"} for r in results],
            "coverage": table_coverage(mmf.CATALOG["families"]),
            "summary": {"runs": len(results), "ok": len(results) - len(failed),
                        "failed": failed, "cells": cells}}


def work_dir(given, keep: bool) -> tuple:
    temporary = False
    if given:
        path = Path(given).resolve()
    elif os.environ.get("VEGA_SERVING_MATRIX_WORK"):
        path = Path(os.environ["VEGA_SERVING_MATRIX_WORK"]).resolve()
    else:
        path = Path(tempfile.mkdtemp(prefix="vega-serving-matrix-"))
        temporary = not keep
        if temporary:
            atexit.register(shutil.rmtree, path, True)
    path.mkdir(parents=True, exist_ok=True)
    return path, temporary


def _arguments(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", nargs="?", default=None)
    ap.add_argument("--cells", default="all", help="all (default) or a comma list of cell ids")
    ap.add_argument("--families", default=None, help="a comma list of globs over families")
    ap.add_argument("--quick", action="store_true", help="one family per cell")
    ap.add_argument("--jobs", type=int, default=None,
                    help="runs at once (default: min(8, cores / threads))")
    ap.add_argument("--threads", type=int, default=None,
                    help="intra-op threads per run (default: $OMP_NUM_THREADS, else 2)")
    ap.add_argument("--timeout", type=float, default=1500.0, help="seconds per run (1500)")
    ap.add_argument("--keep", action="store_true", help="keep a temporary work directory")
    ap.add_argument("--no-inspect", action="store_true",
                    help="pack without the inspection routes (their skip reason is allowed)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON at the end")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _arguments(argv)
    if a.worker:
        return worker(Path(a.worker), a.no_inspect)
    started = time.monotonic()
    threads = a.threads or int(os.environ.get("OMP_NUM_THREADS") or 2)
    jobs = a.jobs or max(1, min(8, (os.cpu_count() or 1) // max(1, threads)))
    try:
        import make_matrix_fixtures as mmf
        coverage = table_coverage(mmf.CATALOG["families"])
        if coverage["missing"] or coverage["unknown"]:
            raise MatrixError(f"the table's families are not the catalog's: missing "
                              f"{coverage['missing']}, unknown {coverage['unknown']}")
        runs = select(a.cells, a.families, a.quick)
        work, temporary = work_dir(a.work, a.keep)
        say(f"OK: serving matrix: {len(runs)} run(s) over "
            f"{len({r.cell for r in runs})} cell(s){' (quick)' if a.quick else ''} under "
            f"{work}, {jobs} at a time, {threads} thread(s) each")
        dirs, failures = make_fixtures(work, runs, jobs=max(jobs, 4))
    except (MatrixError, _fixture_error()) as exc:
        say(f"ERROR: {exc}")
        return 2
    say(f"{'OK' if not failures else 'ERROR'}: fixtures made "
        f"({time.monotonic() - started:.0f} s): the real shape.py, one compiled netspec per "
        f"run; {len(failures)} run(s) without a fixture{': ' if failures else ''}"
        + "; ".join(f"{k}: {v}" for k, v in failures.items()))
    results = run_all(runs, dirs, failures, jobs, worker_env(threads), a.timeout,
                      a.no_inspect)
    selection = {"cells": a.cells, "families": a.families, "quick": a.quick,
                 "no_inspect": a.no_inspect}
    report = build_report(selection, results)
    report["seconds"] = round(time.monotonic() - started, 1)
    (work / "serving-matrix.json").write_text(json.dumps(report, indent=2, default=str) + "\n",
                                              encoding="utf-8")
    if a.json:
        print(json.dumps(report, indent=1, default=str))
    s = report["summary"]
    say(f"{'OK' if not s['failed'] else 'ERROR'}: serving matrix: {s['ok']}/{s['runs']} run(s) "
        f"green over {len(s['cells'])} cell(s) in {report['seconds']:.0f} s; failed "
        f"{s['failed'] or '[]'} — report {work / 'serving-matrix.json'}"
        f"{' (a temporary directory, removed at exit; --keep keeps it)' if temporary else ''}")
    return 0 if not s["failed"] else 1


def _fixture_error():
    import make_matrix_fixtures as mmf
    return mmf.MatrixError


if __name__ == "__main__":
    sys.exit(main())
