#!/usr/bin/env python3
"""
run_matrix.py — the backend coverage matrix (BUILD-PLAN B19, D1 §17.2 without its `inspect`
stage, R-B5): every admitted cell of the catalog, emitted, gated, fitted and scored under the
seal, one cell per worker process.

    python3 run_matrix.py [<work>] [--cells all|smoke|<glob>[,<glob>…]] [--sanity]
                          [--jobs N] [--threads N] [--timeout S] [--keep]

`<work>` defaults to `$VEGA_MATRIX_WORK`, else a temporary directory removed at exit (`--keep`
keeps it). `--cells` takes `all` (the 127 cells of `catalog.json`'s admission table), `smoke`
(one cell per family, 21 — the unit tier's `backend-smoke-cells`), or globs over cell ids such as
`gru/*` or `*/graph-node/*` (an id is `family/shape-tag/task-tag/variant`, e.g.
`gcn/graph-node/multiclass/inductive-full`, `gru/window/sequence-classification:binary/stride-2`).

`make_matrix_fixtures.py` makes the fixtures first (raw tables through the real `shape.py`, one
netspec per cell). Each cell then runs in its own process (`--jobs` at once, each with
`--threads` intra-op threads, on the CPU), through the stages in order — a stage runs only when
every earlier one passed:

  compile       `compile_spec.compile_spec` over the spec, its pipeline and its shaped manifest
  emit          `emit_backend.py --smoke` (in the worker): the v2.3 emitter writes the package …
  smoke         … and the Model port's builder runs one forward pass on a real train batch
                (`smoke.json`)
  conformance   `conformance.check_report` C0–C14: C0–C10 must pass; the C11–C14 size findings
                are recorded per cell with the template each emitted file comes from, and never
                fail a cell (G2 flips their severity in W7)
  fit           `PackageAdapters.load` → `Network.build` → `Trainer().fit` for the spec's epochs
                (3), the best checkpoint written through the ModelStore port; every epoch runs,
                none is non-finite, and the best valid primary is a finite number
  sealed_score  under a `test_seal.json` as /vega-eval writes one, `Evaluator.score_sealed_test`
                restores the checkpoint through the store and opens the test part through the
                DataSource's one door; the test primary is a finite number
  predictions   the same door scored row by row (`Evaluator.scored` and `heads`; their metrics
                are `score_sealed_test`'s) and laid out as `vega/predictions-layout@1` (R-C2):
                `row_id` (the anchor's or the node's raw position, the graph id — then also a
                `graph_id` column), `y`/`y_k`, `score`/`score_k` = every column of the raw head,
                `row_score` for the reconstruction and anomaly readings, the parquet metadata
                `vega.row_id_kind`, `vega.truth_shape`, `vega.score_shape`,
                `vega.task_effective`; the score and truth shapes are the score contract's (D1
                §3.4), every value is finite, and the row ids are the test part's: its rows
                (row), its nodes (node level), its graphs (graph level), its anchors with every
                other test row counted unscored with its reason (window; a forecast leaves the
                last `horizon` rows of each series unscored, a sequence task none — scoring is
                dense whatever the stride, R-A5)

`--sanity` adds the learning checks of `recipes.json` (D1 §17.2, `slow`): the named family cell
and its floor — a floor cell, or a baseline of the package's `baselines` module fitted on train
— both at the check's epochs (30), compared on valid in the primary's direction: `gcn` over the
structure-blind `linear` at node level, `gin` over `linear` + `graph_pool`, `gru` forecast over
`naive`.

`<work>/matrix-report.json` (`vega/backend-matrix@1`):

    {"schema": "vega/backend-matrix@1", "backend": "pytorch", "torch": "2.14.0+cu130",
     "selection": "all", "admission": {"cells": 127, "table": "equal"},
     "cells": [{"id", "group", "family", "shape", "level", "task", "variant", "effective",
                "stages": {"compile": "ok", "emit": "ok", "smoke": "ok", "conformance": "ok",
                           "fit": "ok", "sealed_score": "ok", "predictions": "ok"},
                "params_total", "epochs", "primary", "valid_primary", "test_primary",
                "seconds", "package", "size": {"counts", "findings"}, "predictions",
                "errors": [{"stage", "message"}]}, …],
     "size": {"counts", "findings": [{"code", "file", "symbol", "template", "cells", …}]},
     "sanity": [{"id", "cell", "floor", "metric", "direction", "value", "floor_value",
                 "passed"}],
     "summary": {"cells": 127, "ok": 127, "failed": []}}

A stage that did not run is `not_run`. **Green is `summary.failed == []`**: every selected cell
passed every stage, and every learning check asked for held (a failed one is listed as
`sanity:<id>`). Exit codes: 0 green; 1 a cell or a check failed; 2 the table, the fixtures or the
arguments could not be used.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import importlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
SCRIPTS = TESTS.parent
for _p in (str(HERE), str(TESTS), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_matrix_fixtures as mmf  # noqa: E402

REPORT_SCHEMA = "vega/backend-matrix@1"
RESULT_SCHEMA = "vega/backend-matrix-result@1"
LAYOUT_SCHEMA = "vega/predictions-layout@1"
TOKEN = "vega-matrix-seal"
STAGES = ("compile", "emit", "smoke", "conformance", "fit", "sealed_score", "predictions")
SANITY_STAGES = ("compile", "emit", "smoke", "fit", "baseline")


def say(line: str) -> None:
    print(line, flush=True)


# ================================================================ the worker: one cell

class StageFailed(Exception):
    """A stage's check did not hold; the message says which and what was seen."""


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class _Tee(io.TextIOBase):
    """stdout for an in-process CLI: kept (for the stage's error) and passed through (the log)."""

    def __init__(self, through):
        self.through, self.kept = through, io.StringIO()

    def write(self, text):
        self.kept.write(text)
        return self.through.write(text)

    def flush(self):
        self.through.flush()


class CellRun:
    """One cell's stages, in one process, over the fixture `make_matrix_fixtures` wrote."""

    def __init__(self, cell_dir: Path):
        self.dir = Path(cell_dir)
        self.fx = json.loads((self.dir / "fixture.json").read_text(encoding="utf-8"))
        self.spec_dict = json.loads(Path(self.fx["spec"]).read_text(encoding="utf-8"))
        self.run_dir = self.dir / "run"
        self.package_dir = self.run_dir / self.fx["package"]
        self.primary = self.spec_dict["metrics"]["primary"]
        self.ad = self.domain = self.spec = self.fit = self.metrics = None
        self._emit_code, self._emit_text = None, ""
        self.record = {"schema": RESULT_SCHEMA,
                       **{k: self.fx[k] for k in ("id", "group", "family", "shape", "level",
                                                  "task", "variant", "effective", "mode")},
                       "stages": {}, "primary": self.primary, "epochs": self.fx["epochs"],
                       "package": str(self.package_dir), "errors": [], "stage_seconds": {}}

    # -- the stages

    def compile(self) -> None:
        import compile_spec as cs
        pipeline = json.loads(Path(self.fx["pipeline"]).read_text(encoding="utf-8"))
        manifest = json.loads(Path(self.fx["manifest"]).read_text(encoding="utf-8"))
        markers = cs.compile_spec(self.spec_dict, model_html=None, pipeline=pipeline,
                                  manifest=manifest)
        self.record["compile_warnings"] = [f"{m.rule}: {m.message}" for m in markers.warns]
        if markers.errors:
            raise StageFailed("; ".join(f"{m.rule}: {m.message}" for m in markers.errors))

    def emit(self) -> None:
        """`emit_backend.py --smoke` in this process: exit 2 is the emission's failure, exit 3
        the smoke pass's (the package is then on disk)."""
        import emit_backend
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        tee = _Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):
            try:
                code = emit_backend.main(["--spec", self.fx["spec"], "--pipeline",
                                          self.fx["pipeline"], "--out", str(self.run_dir),
                                          "--smoke", "--device", "cpu"])
            except SystemExit as exc:           # an unreadable input exits from inside the CLI
                code = exc.code if isinstance(exc.code, int) else 2
        self._emit_code, self._emit_text = code, tee.kept.getvalue()
        if code == 2:
            raise StageFailed(self._errors(self._emit_text) or "emit_backend exited 2")

    def smoke(self) -> None:
        if self._emit_code != 0:
            raise StageFailed(self._errors(self._emit_text) or
                              f"emit_backend --smoke exited {self._emit_code}")
        smoke = json.loads((self.run_dir / "smoke.json").read_text(encoding="utf-8"))
        if not smoke.get("forward_ok"):
            raise StageFailed(f"smoke.json says forward_ok {smoke.get('forward_ok')!r}")
        self.record["params_total"] = smoke.get("params_total")
        self.record["madds_per_sample"] = smoke.get("madds_per_sample")

    def conformance(self) -> None:
        import conformance
        bmap = json.loads((self.package_dir / "blueprint-map.json").read_text(encoding="utf-8"))
        pipeline = json.loads(Path(self.fx["pipeline"]).read_text(encoding="utf-8"))
        markers, findings = conformance.check_report(self.package_dir, bmap, pipeline,
                                                     conformance.GateOptions())
        self.record["size"] = size_record(findings, self._templates(pipeline), bmap)
        if not markers.ok:
            raise StageFailed("; ".join(f"{m.rule} {m.where or ''} {m.message}".strip()
                                        for m in markers.errors))

    def fit_stage(self) -> None:
        from vega_adapters import PackageAdapters
        self.ad = PackageAdapters.load(self.package_dir, self.fx["package"])
        self.domain = importlib.import_module(f"{self.fx['package']}.domain")
        self.spec = self.domain.NetSpec.from_dict(self.spec_dict)
        model = self.ad.build(self.spec)
        run = self.domain.RunContext(checkpoint_dir=str(self.run_dir / "checkpoints"))
        self.fit = fit = self.ad.trainer.fit(model, self.spec, self.ad.data, run=run)
        best = fit.best_valid.as_dict() if fit.best_valid is not None else {}
        stopped = getattr(fit.stopped_by, "value", str(fit.stopped_by))
        self.record.update(epochs_run=fit.epochs_run, stopped_by=stopped,
                           best_epoch=fit.best_epoch, valid_primary=best.get(self.primary))
        want = int(self.spec_dict["training"]["max_epochs"])
        if stopped != "max_epochs" or fit.epochs_run != want:
            raise StageFailed(f"the fit stopped by {stopped} after {fit.epochs_run} of {want} "
                              "epochs (the matrix's budget runs every epoch)")
        if not _finite(best.get(self.primary)):
            raise StageFailed(f"the best valid {self.primary} is {best.get(self.primary)!r} "
                              f"(unavailable: {best.get('_unavailable')})")
        if fit.checkpoint is None:
            raise StageFailed("the fit wrote no checkpoint through the ModelStore port")

    def sealed_score(self) -> None:
        seal = self.dir / "test_seal.json"
        seal.write_text(json.dumps({"token": TOKEN, "opened": False}), encoding="utf-8")
        self.metrics = self.ad.evaluator.score_sealed_test(self.spec, TOKEN,
                                                           checkpoint=self.fit.checkpoint,
                                                           seal_path=str(seal))
        values = self.metrics.as_dict()
        self.record["test_primary"] = values.get(self.primary)
        if not _finite(values.get(self.primary)):
            raise StageFailed(f"the test {self.primary} is {values.get(self.primary)!r} "
                              f"(unavailable: {values.get('_unavailable')})")

    def predictions(self) -> None:
        restored = self.ad.store.load(self.fit.checkpoint, spec=self.spec, build=self.ad.build)
        part = self.ad.data.load_sealed_part(self.spec, TOKEN, str(self.dir / "test_seal.json"))
        scored = self.ad.evaluator.scored(self.spec, part, restored)
        heads = self.ad.evaluator.heads(self.spec, part, restored)
        again = self.ad.evaluator.measure(self.spec, self.domain.Part.TEST, scored).as_dict()
        if _canon(again) != _canon(self.metrics.as_dict()):
            raise StageFailed("the part scored row by row does not give score_sealed_test's "
                              f"metrics: {_canon(again)} vs {_canon(self.metrics.as_dict())}")
        layout = PredictionsLayout(self.fx, self.spec_dict)
        self.record["predictions"] = layout.check(scored, heads, self.dir / "predictions.parquet")

    def baseline(self) -> None:
        """The sanity floor a baseline gives: fitted on train, its valid primary."""
        name = self.fx["baseline"]
        window = self.spec_dict["data_contract"]["sample"]
        context = self.domain.BaselineContext(season=window.get("season"),
                                              horizon=int(window.get("horizon") or 1))
        recipe = self.ad.baselines.build(self.spec, [name], context)[0]
        train = self.ad.data.load_part(self.spec, self.domain.Part.TRAIN)
        base = recipe.fit(recipe.design_of(self.ad.data, train))
        if not base.fitted:
            raise StageFailed(f"the {name} baseline was not fitted: {base.why}")
        valid = self.ad.data.load_part(self.spec, self.domain.Part.VALID)
        scored = self.ad.evaluator.scored(self.spec, valid, base)
        value = self.ad.evaluator.measure(self.spec, self.domain.Part.VALID, scored).as_dict()
        self.record["baseline"] = {"name": name, "valid_primary": value.get(self.primary)}
        if not _finite(value.get(self.primary)):
            raise StageFailed(f"the {name} baseline's valid {self.primary} is "
                              f"{value.get(self.primary)!r}")

    # -- helpers

    @staticmethod
    def _errors(text: str) -> str:
        return " | ".join(line for line in text.splitlines() if line.startswith("ERROR"))[:2000]

    def _templates(self, pipeline) -> dict:
        """{package-relative file: the template it was rendered from} (the v2.3 emit table)."""
        import render_templates as rt
        emit = rt.load_emit()
        layout = emit.v23_layout(emit.plan(self.spec_dict, pipeline))
        return {rel: name for name, (rel, _) in layout.items()}

    def run(self) -> dict:
        names = SANITY_STAGES if self.fx["mode"] == "sanity" else STAGES
        if self.fx["mode"] == "sanity" and not self.fx.get("baseline"):
            names = names[:-1]
        steps = {"compile": self.compile, "emit": self.emit, "smoke": self.smoke,
                 "conformance": self.conformance, "fit": self.fit_stage,
                 "sealed_score": self.sealed_score, "predictions": self.predictions,
                 "baseline": self.baseline}
        started, failed = time.monotonic(), False
        for name in names:
            if failed:
                self.record["stages"][name] = "not_run"
                continue
            began = time.monotonic()
            try:
                steps[name]()
                self.record["stages"][name] = "ok"
            except StageFailed as exc:
                failed = self._failed(name, str(exc))
            except Exception as exc:  # noqa: BLE001 — a cell's crash is its stage's failure
                tail = traceback.format_exc().strip().splitlines()[-6:]
                failed = self._failed(name, f"{type(exc).__name__}: {exc} [{' / '.join(tail)}]")
            self.record["stage_seconds"][name] = round(time.monotonic() - began, 2)
        self.record["seconds"] = round(time.monotonic() - started, 2)
        return self.record

    def _failed(self, stage: str, message: str) -> bool:
        self.record["stages"][stage] = "failed"
        self.record["errors"].append({"stage": stage, "message": message[:4000]})
        print(f"ERROR: {self.fx['id']}: {stage}: {message}", flush=True)
        return True


def _canon(values: dict) -> str:
    return json.dumps(values, sort_keys=True, default=str)


def size_record(findings, templates: dict, bmap: dict) -> dict:
    """The C11–C14 findings of one package, each with the template (or the spliced segments)
    its file was rendered from."""
    segments = {e["file"]: e.get("segments") for e in bmap.get("entries") or () if e.get("file")}
    out = []
    for f in findings:
        out.append({"code": f.code, "file": f.file, "line": f.line, "symbol": f.symbol,
                    "template": templates.get(f.file), "segments": segments.get(f.file),
                    "message": f.message()})
    counts = {c: sum(1 for f in out if f["code"] == c) for c in ("C11", "C12", "C13", "C14")}
    return {"counts": counts, "findings": out}


class PredictionsLayout:
    """What `vega/predictions-layout@1` (R-C2) holds for one cell, from its spec and its fixture,
    and the check of an Evaluator's scored part against it."""

    def __init__(self, fx: dict, spec: dict):
        import numpy as np
        import pandas as pd
        self.fx, self.spec = fx, spec
        self.kind = fx["effective"]
        task = spec["task"]
        self.k = len(task.get("classes") or ())
        feats = spec["data_contract"]["features"]
        self.n_numeric = sum(1 for f in feats if f["role"] == "numeric")
        self.head_out = next(b["params"]["out"] for b in spec["blocks"] if b["type"] == "head")
        self.horizon = spec["data_contract"]["sample"].get("horizon")
        self.graph_level = fx["shape"] == "graph" and fx["level"] == "graph"
        split = json.loads(Path(fx["fitted"]).read_text(encoding="utf-8"))["__split_raw__"]
        self.test = np.asarray(split["test"], dtype=np.int64)
        if self.graph_level:
            raw = pd.read_csv(fx["raw"], usecols=[fx["graph_id"]], dtype=str)
            self.test_graphs = set(raw[fx["graph_id"]].to_numpy()[self.test].tolist())
        series_by = spec["data_contract"]["sample"].get("series_by")
        self.series = 1
        if fx["shape"] == "window" and series_by:
            raw = pd.read_csv(fx["raw"], usecols=[series_by], dtype=str)
            self.series = int(pd.Series(raw[series_by].to_numpy()[self.test]).nunique())

    def score_shape(self) -> tuple:
        """The score contract's trailing shape (D1 §3.4)."""
        return {"binary": (), "regression": (), "anomaly": (), "multiclass": (self.k,),
                "multilabel": (self.k,), "ordinal": (self.k - 1,), "forecast": (self.horizon,),
                "unsupervised": (self.n_numeric,)}[self.kind]

    def truth_shape(self):
        if self.kind == "anomaly" and not self.fx["labelled"]:
            return None
        return {"multilabel": (self.k,), "forecast": (self.horizon,),
                "unsupervised": (self.n_numeric,)}.get(self.kind, ())

    def check(self, scored, heads, path: Path) -> dict:
        import numpy as np
        n = len(scored.row_ids)
        self._shapes(scored, np.asarray(heads), n)
        self._row_ids(scored)
        frame = self.frame(scored, np.asarray(heads))
        meta = self.metadata(scored, np.asarray(heads))
        self._roundtrip(frame, meta, path)
        return {"rows": n, "unscored": len(scored.unscored), "columns": list(frame.columns),
                "row_id_kind": meta["vega.row_id_kind"], "file": str(path)}

    def _shapes(self, scored, heads, n: int) -> None:
        import numpy as np
        score = np.asarray(scored.score)
        if tuple(score.shape) != (n, *self.score_shape()):
            raise StageFailed(f"score is {tuple(score.shape)}, the {self.kind} score contract "
                              f"reads {(n, *self.score_shape())}")
        if tuple(heads.shape) != (n, self.head_out):
            raise StageFailed(f"the raw head is {tuple(heads.shape)}, the spec's head_out is "
                              f"{self.head_out}")
        want = self.truth_shape()
        truth = None if scored.truth is None else np.asarray(scored.truth)
        if (want is None) != (truth is None) or (truth is not None
                                                 and tuple(truth.shape) != (n, *want)):
            raise StageFailed(f"the truth is {None if truth is None else tuple(truth.shape)}, "
                              f"the {self.kind} reading holds {None if want is None else (n, *want)}")
        rows = scored.row_score
        if (self.kind in ("unsupervised", "anomaly")) != (rows is not None):
            raise StageFailed(f"row_score is {'present' if rows is not None else 'absent'} on a "
                              f"{self.kind} reading (R-C2: the reconstruction and anomaly "
                              "readings carry it, and only they)")
        for name, values in (("score", score), ("the raw head", heads), ("row_score", rows)):
            if values is not None and not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                raise StageFailed(f"{name} holds a non-finite value")

    def _row_ids(self, scored) -> None:
        import numpy as np
        kind = getattr(scored.row_id_kind, "value", str(scored.row_id_kind))
        ids = list(np.asarray(scored.row_ids).tolist())
        if len(set(map(str, ids))) != len(ids):
            raise StageFailed("a row id repeats in the scored part")
        if self.graph_level:
            if kind != "graph_id" or set(map(str, ids)) != self.test_graphs:
                raise StageFailed(f"row ids ({kind}) are not the test part's {len(self.test_graphs)} "
                                  f"graph ids ({len(ids)} given)")
            if scored.graph_ids is None or list(map(str, scored.graph_ids)) != list(map(str, ids)):
                raise StageFailed("graph_ids beside the row ids are missing or differ")
            return
        if kind != "raw_position":
            raise StageFailed(f"row_id_kind is {kind}, the {self.fx['shape']} shape's is "
                              "raw_position")
        test, got = set(self.test.tolist()), set(int(i) for i in ids)
        if self.fx["shape"] != "window":
            if got != test:
                raise StageFailed(f"row ids are not the test part's raw positions ({len(got)} of "
                                  f"{len(test)}; {len(got - test)} outside it)")
            return
        self._window_rows(scored, test, got)

    def _window_rows(self, scored, test: set, got: set) -> None:
        unscored = {int(u.raw_pos): getattr(u.reason, "value", str(u.reason))
                    for u in scored.unscored}
        if not got <= test or got & set(unscored) or got | set(unscored) != test:
            raise StageFailed(f"the anchors ({len(got)}) and the unscored rows ({len(unscored)}) "
                              f"do not partition the test part's {len(test)} rows")
        want = {"horizon_beyond_part"} if self.kind == "forecast" else set()
        if set(unscored.values()) - want:
            raise StageFailed(f"unscored reasons {sorted(set(unscored.values()))}; with the "
                              "previous-part context only a forecast's horizon leaves rows")
        if self.kind == "forecast" and len(unscored) != self.series * self.horizon:
            raise StageFailed(f"{len(unscored)} unscored forecast anchors, the part's end "
                              f"leaves horizon x series = {self.horizon * self.series}")
        if self.kind != "forecast" and unscored:
            raise StageFailed(f"{len(unscored)} rows unscored on a sequence task")

    def frame(self, scored, heads):
        import numpy as np
        import pandas as pd
        cols = {"row_id": list(np.asarray(scored.row_ids).tolist())}
        if self.graph_level:
            cols["graph_id"] = [str(g) for g in scored.graph_ids]
        cols.update(_columns("y", scored.truth))
        cols.update(_columns("score", heads))
        if scored.row_score is not None:
            cols["row_score"] = np.asarray(scored.row_score, dtype=np.float64).reshape(-1)
        return pd.DataFrame(cols)

    def metadata(self, scored, heads) -> dict:
        import numpy as np
        truth = None if scored.truth is None else list(np.asarray(scored.truth).shape[1:])
        return {"vega.row_id_kind": getattr(scored.row_id_kind, "value", str(scored.row_id_kind)),
                "vega.truth_shape": json.dumps(None if truth is None else ["N", *truth]),
                "vega.score_shape": json.dumps(["N", *list(np.asarray(heads).shape[1:])]),
                "vega.task_effective": self.kind, "vega.layout": LAYOUT_SCHEMA}

    def _roundtrip(self, frame, meta: dict, path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(frame, preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                               **{k.encode(): str(v).encode()
                                                  for k, v in meta.items()}})
        pq.write_table(table, path)
        back = pq.read_table(path)
        stored = {k.decode(): v.decode() for k, v in (back.schema.metadata or {}).items()}
        missing = [k for k, v in meta.items() if stored.get(k) != str(v)]
        if missing or back.column_names != list(frame.columns) or back.num_rows != len(frame):
            raise StageFailed(f"the predictions parquet does not read back as written "
                              f"(metadata {missing}, columns {back.column_names})")


def _columns(stem: str, values) -> dict:
    """`stem` for one column, `stem_0…` for several (R-C2); nothing when there are no values."""
    import numpy as np
    if values is None:
        return {}
    arr = np.asarray(values)
    if arr.ndim == 1:
        return {stem: arr}
    if arr.shape[1] == 1:
        return {stem: arr[:, 0]}
    return {f"{stem}_{j}": arr[:, j] for j in range(arr.shape[1])}


def worker(cell_dir: Path) -> int:
    record = CellRun(cell_dir).run()
    (Path(cell_dir) / "result.json").write_text(json.dumps(record, indent=2, default=str) + "\n",
                                                encoding="utf-8")
    return 0 if all(v == "ok" for v in record["stages"].values()) else 1


# ================================================================ the parent

def worker_env(threads: int) -> dict:
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def run_worker(made, env: dict, timeout: float) -> dict:
    """One cell's worker process; its `result.json`, or a failed record naming why none came."""
    cell_dir = Path(made.dir)
    for stale in ("result.json", "predictions.parquet", "test_seal.json"):
        (cell_dir / stale).unlink(missing_ok=True)
    log = cell_dir / "cell.log"
    began = time.monotonic()
    with open(log, "w", encoding="utf-8") as fh:
        try:
            code = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                                   str(cell_dir)], stdout=fh, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = None
    result = cell_dir / "result.json"
    if result.is_file():
        return json.loads(result.read_text(encoding="utf-8"))
    tail = " | ".join(log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:])
    why = (f"timed out after {timeout:g} s" if code is None
           else f"the worker exited {code} without a result")
    return {"schema": RESULT_SCHEMA, "id": made.id, "group": made.group, "family": made.family,
            "shape": made.shape, "level": made.level, "task": made.task,
            "variant": made.variant, "effective": made.effective, "mode": made.mode,
            "stages": {s: "not_run" for s in STAGES}, "epochs": made.epochs,
            "errors": [{"stage": "worker", "message": f"{why}: {tail}"}],
            "seconds": round(time.monotonic() - began, 2)}


def run_cells(made: dict, jobs: int, env: dict, timeout: float, what: str) -> list:
    """Every cell of `made` in its worker, `jobs` at once; one line each as it ends."""
    def one(m):
        rec = run_worker(m, env, timeout)
        ok = rec["stages"] and all(v == "ok" for v in rec["stages"].values()) \
            and not rec.get("errors")
        if ok:
            say(f"OK: {what}: {rec['id']} ({rec.get('seconds', 0):.1f} s, "
                f"{rec.get('params_total')} params, valid {rec.get('primary')} "
                f"{_fmt(rec.get('valid_primary'))})")
        else:
            err = (rec.get("errors") or [{"stage": "?", "message": "?"}])[0]
            say(f"ERROR: {what}: {rec['id']}: {err['stage']}: {err['message'][:400]}")
        return rec

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return list(pool.map(one, list(made.values())))


def _fmt(value) -> str:
    return f"{value:.4g}" if _finite(value) else str(value)


def cell_ok(rec: dict) -> bool:
    return bool(rec.get("stages")) and all(v == "ok" for v in rec["stages"].values()) \
        and not rec.get("errors")


def sanity_checks(table: dict, results: list, chosen_ids: set) -> list:
    """Each learning check asked for: the family's best valid primary against its floor's
    (a floor cell's best valid primary, or a baseline's valid primary), in the primary's
    direction (catalog `metric_direction`)."""
    by_id = {r["id"]: r for r in results}
    out = []
    for check in table["sanity"]:
        if check["cell"] not in chosen_ids:
            continue
        fam = by_id.get(check["cell"]) or {}
        floor = check["floor"]
        if floor.get("cell"):
            floor_rec, floor_name = by_id.get(floor["cell"]) or {}, floor["cell"]
            floor_value = floor_rec.get("valid_primary")
        else:
            floor_rec, floor_name = fam, floor["baseline"]
            floor_value = (fam.get("baseline") or {}).get("valid_primary")
        metric = fam.get("primary")
        direction = mmf.CATALOG["metric_direction"].get(metric, "max")
        value = fam.get("valid_primary")
        ran = cell_ok(fam) and cell_ok(floor_rec)
        passed = ran and _finite(value) and _finite(floor_value) and (
            value > floor_value if direction == "max" else value < floor_value)
        out.append({"id": check["id"], "cell": check["cell"], "floor": floor_name,
                    "epochs": check["epochs"], "metric": metric, "direction": direction,
                    "value": value, "floor_value": floor_value, "passed": bool(passed),
                    "ran": ran, "why": check.get("why")})
    return out


def aggregate_size(results: list) -> dict:
    """The C11–C14 findings over every cell's package, each once, with the cells it shows in
    (risk 5: a finding in a template past its owner's wave goes to that owner)."""
    seen = {}
    for rec in results:
        for f in (rec.get("size") or {}).get("findings") or ():
            key = (f["code"], f["file"], f["symbol"])
            entry = seen.setdefault(key, {**{k: f[k] for k in ("code", "file", "symbol",
                                                               "template", "segments",
                                                               "message")}, "cells": 0})
            entry["cells"] += 1
    findings = sorted(seen.values(), key=lambda e: (e["code"], e["file"], e["symbol"]))
    counts = {c: sum(1 for f in findings if f["code"] == c) for c in ("C11", "C12", "C13", "C14")}
    return {"counts": counts, "findings": findings}


def _torch_version() -> str:
    try:
        import torch
        return str(torch.__version__)
    except ImportError:
        return "absent"


def build_report(selection: str, results: list, sanity: list) -> dict:
    failed = [r["id"] for r in results if not cell_ok(r)]
    failed += [f"sanity:{s['id']}" for s in sanity if not s["passed"]]
    return {"schema": REPORT_SCHEMA, "backend": "pytorch", "torch": _torch_version(),
            "selection": selection,
            "admission": {"cells": int(mmf.CATALOG["admission"]["cells"]), "table": "equal"},
            "cells": [{k: v for k, v in r.items() if k != "schema"} for r in results],
            "size": aggregate_size(results), "sanity": sanity,
            "summary": {"cells": len(results), "ok": sum(1 for r in results if cell_ok(r)),
                        "failed": failed}}


def work_dir(given, keep: bool) -> tuple:
    """(the work directory, whether it is removed at exit): `<work>`, `$VEGA_MATRIX_WORK`,
    else a temporary directory (kept with `--keep`)."""
    temporary = False
    if given:
        path = Path(given).resolve()
    elif os.environ.get("VEGA_MATRIX_WORK"):
        path = Path(os.environ["VEGA_MATRIX_WORK"]).resolve()
    else:
        path = Path(tempfile.mkdtemp(prefix="vega-matrix-"))
        temporary = not keep
        if temporary:
            atexit.register(shutil.rmtree, path, True)
    path.mkdir(parents=True, exist_ok=True)
    return path, temporary


def _arguments(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("work", nargs="?", default=None)
    ap.add_argument("--cells", default="all",
                    help="all (default), smoke, or a comma list of globs over cell ids")
    ap.add_argument("--sanity", action="store_true",
                    help="also the learning checks (slow: fits at the check's epochs)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="cells at once (default: min(4, cores / threads))")
    ap.add_argument("--threads", type=int, default=None,
                    help="intra-op threads per cell (default: $OMP_NUM_THREADS, else 2)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per cell (900)")
    ap.add_argument("--keep", action="store_true", help="keep a temporary work directory")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = _arguments(argv)
    if a.worker:
        return worker(Path(a.worker))
    threads = a.threads or int(os.environ.get("OMP_NUM_THREADS") or 2)
    jobs = a.jobs or max(1, min(4, (os.cpu_count() or 1) // max(1, threads)))
    try:
        table = mmf.load_table()
        chosen = mmf.select(mmf.cells(table), a.cells, table)
        work, temporary = work_dir(a.work, a.keep)
        say(f"OK: backend matrix: {len(chosen)} cell(s) ({a.cells}){' + sanity' if a.sanity else ''}"
            f" under {work}, {jobs} at a time, {threads} thread(s) each")
        made = mmf.make(work, chosen, table, a.sanity, jobs=max(jobs, 4))
    except mmf.MatrixError as exc:
        say(f"ERROR: {exc}")
        return 2
    env = worker_env(threads)
    results = run_cells(made["cells"], jobs, env, a.timeout, "cell")
    sanity = []
    if a.sanity:
        sanity_results = run_cells(made["sanity"], jobs, env, a.timeout, "sanity fit")
        sanity = sanity_checks(table, sanity_results, {c.id for c in chosen})
        for s in sanity:
            say(f"{'OK' if s['passed'] else 'ERROR'}: sanity {s['id']}: {s['cell']} valid "
                f"{s['metric']} {_fmt(s['value'])} vs {s['floor']} {_fmt(s['floor_value'])} "
                f"({s['direction']} is better, {s['epochs']} epochs)")
    report = build_report(a.cells, results, sanity)
    (work / "matrix-report.json").write_text(json.dumps(report, indent=2, default=str) + "\n",
                                             encoding="utf-8")
    s = report["summary"]
    size = report["size"]["counts"]
    say(f"OK: C11–C14 findings recorded (distinct over the packages): {size}")
    say(f"{'OK' if not s['failed'] else 'ERROR'}: backend matrix: {s['ok']}/{s['cells']} cell(s) "
        f"green{', sanity ' + str(sum(x['passed'] for x in sanity)) + '/' + str(len(sanity)) if sanity else ''}"
        f"; failed {s['failed'] or '[]'} — report {work / 'matrix-report.json'}"
        f"{' (a temporary directory, removed at exit; --keep keeps it)' if temporary else ''}")
    return 0 if not s["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
