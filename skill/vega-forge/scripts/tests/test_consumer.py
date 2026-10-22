#!/usr/bin/env python3
"""
test_consumer.py — the stream consumer, `templates/serving/serve/consumer.py` (build plan S09;
D2 §12, R-A5, R-A6, R-D14).

    python3 -m pytest test_consumer.py [-q]        (or: python3 test_consumer.py [-v])

The bundle is a rendered window bundle of the vertical-slice spike's fixture (a)
(`spike/make_spike_fixtures.py` `window`: 3 series × 300 steps through the real `shape.py`, a
lag of the target (lag depth 2), a `gru` forecast with L = 12, H = 3, autoregressive, emitted
by the v2.3 emitter) laid out as vega-pack lays a bundle out: the fixture's package with the
serving package templates rendered into its Predictor's module (`render_templates.py`, as
S05's fixtures do), the shaper's transform-only module and `fitted.json`, the bundle CLI from
`bundle/predict.py`, `netspec.json`, `schema.json` (vega-pack's `input_schema.derive`), a
seeded network saved through the package's own ModelRegistry, the records vega-pack writes
(`output_record`, `sample_record`), a monitor reference, examples/ (context rows first,
`__row__` at the answered rows) and MANIFEST.json with `files` and `roles` as
`bundle_manifest` writes them — and `serve/` rendered from the serve templates with the fills
`serving_pack` gives them (the consumer and the contract modules it reuses).

1. JSON lines in and out: the test part's raw rows, one JSON object per line, through
   `python -m serve.consumer` in the bundle; every output line is the answer envelope (the
   versions, the reading, `n_rows`, one `RowPrediction` per row with exactly the codec's keys)
   with the stream's `batch` and `positions`, plus the monitor's readings; the summary on
   stderr counts what was read. `--history` seeds each series: every row is answered.
2. The history buffer completes windows across batches: with batches smaller than a window,
   the answers are row for row the Predictor's over the whole stream in one request (answered
   flags, values within the round-trip tolerance, warnings), and the only unanswered rows are
   the first `length − 1` of each series; without the buffer no row is answered. A seeded
   history answers every row, as one request with that history does.
3. The buffer keeps the last `history_rows` rows of each series in arrival order (MANIFEST's
   sample record sizes it); the source reads JSON lines in batches and keeps a line that is
   not JSON in its row's place.
4. A row that breaks the contract is refused in its batch's problem line (422, the column
   named, the stream position) and the others answered; refused rows never join the history.
   The monitor's readings follow `--monitor`.
5. What the consumer cannot serve is refused before a row is read: a sample that is not a
   window (exit 2), an altered pinned file (exit 2), examples the bundle does not reproduce
   (exit 3).
6. The template: every placeholder is one serving_pack fills; C11–C13 (`conformance.py
   --size-only`) report nothing; the literal gate (`lint_templates.py`) finds nothing.

Exit 0 when every test passes.
"""
from __future__ import annotations

import atexit
import contextlib
import functools
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
SKILLS = SCRIPTS.parent.parent
SERVING = SCRIPTS / "backends" / "pytorch" / "templates" / "serving"
PACKAGE_TEMPLATES = SERVING / "package"
SERVE_TEMPLATES = SERVING / "serve"
TEMPLATE = SERVE_TEMPLATES / "consumer.py"
PACK = SKILLS / "vega-pack" / "scripts"
LINT = SKILLS / "vega" / "scripts" / "tests" / "lint_templates.py"
for _p in (str(SCRIPTS), str(HERE), str(HERE / "spike"), str(PACK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bundle_manifest  # noqa: E402  (vega-pack: file roles and pins, as pack writes them)
import bundle_slice  # noqa: E402  (vega-pack: the module of each role, for the CLI)
import input_schema  # noqa: E402
import make_spike_fixtures as msf  # noqa: E402
import output_record as orec  # noqa: E402
import render_templates as rt  # noqa: E402
from pack_context import Campaign  # noqa: E402
from vega_pack_core import Tolerance  # noqa: E402

# the serve modules a stream bundle carries: the consumer and the contract it reuses
SERVE_FILES = ("consumer.py", "gaps.py", "contract.py", "rows.py", "tables.py", "answers.py",
               "problems.py")
# (a stream bundle carries windows: `graphs.py`, the reference graph, is a graph bundle's only)
SUPPORT_SET = ("records.py", "guards.py", "heads.py", "values.py", "samples.py")
# the fills serving_pack._serve_fills gives the serve templates (an identity, not a dataset)
FILLS = {"{{PACKAGE}}": "spike_window", "{{PORT_PREDICTOR}}": "Predictor",
         "{{PORT_MONITOR}}": "Monitor", "{{TITLE_PY}}": "'A thing'",
         "{{DESCRIPTION_PY}}": "'things'", "{{TITLE_HTML}}": "A thing",
         "{{DESCRIPTION_HTML}}": "things", "{{NOUNS}}": "things", "{{NOUN}}": "thing",
         "{{ROW_MODEL}}": "ThingRow", "{{CMD}}": "thing", "{{ENV}}": "STREAMTEST",
         "{{PORT}}": "8000"}
ENV = "STREAMTEST"
MODEL_VERSION_SEED = 0
EXAMPLE_CONTEXT, EXAMPLE_ROWS = 11, 8
WINDOW_NOTES = ("window incomplete", "lag features at the start of the window")
VERBOSE = "-v" in sys.argv


# --------------------------------------------------------------------------- the work directory

@functools.lru_cache(maxsize=None)
def work() -> Path:
    made = Path(tempfile.mkdtemp(prefix="vega-consumer-"))
    atexit.register(shutil.rmtree, made, True)
    return made


def _json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _quiet(fn, *args, **kwargs):
    """A pack helper's OK/WARN lines kept off the test output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def digest(path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- the bundle

class Built:
    """The fixture and the bundle made from it: where each file is, the records, the rows."""

    def __init__(self):
        self.fixture = msf.make("window", work() / "spike", emit="full")
        self.spec = json.loads(Path(self.fixture.spec).read_text(encoding="utf-8"))
        self.pipeline = json.loads(Path(self.fixture.pipeline).read_text(encoding="utf-8"))
        self.recipe = json.loads(Path(self.fixture.recipe).read_text(encoding="utf-8"))
        self.fitted = json.loads(Path(self.fixture.fitted).read_text(encoding="utf-8"))
        self.shaped_manifest = json.loads(Path(self.fixture.manifest).read_text(
            encoding="utf-8"))
        self.raw = pd.read_csv(self.fixture.raw, low_memory=False)
        self.split = self.fitted["__split_raw__"]
        self.package = self.fixture.package
        self.root = work() / "bundle"
        self.sample = orec.sample_record(self.spec, self.recipe)
        self.output = orec.output_record(self.spec, None, self.spec["task"]["label"])
        self.schema = _quiet(input_schema.derive, self._campaign(), self.sample, [])
        self._lay_out()

    # -- the raw rows ------------------------------------------------------------------------
    def part(self, name) -> pd.DataFrame:
        """A part's raw rows as a caller holds them: the schema's input columns only."""
        rows = self.raw.iloc[np.asarray(self.split[name], dtype=np.int64)]
        keep = [c["name"] for c in self.schema["input"]]
        return rows[keep].reset_index(drop=True)

    def tail(self, name, n) -> pd.DataFrame:
        """The last `n` rows of each series of a part, in the part's order."""
        rows = self.part(name)
        series = self.sample["series_by"]
        from_end = rows.groupby(series, sort=False).cumcount(ascending=False)
        return rows.loc[from_end.to_numpy() < n].reset_index(drop=True)

    # -- the layout --------------------------------------------------------------------------
    def _campaign(self) -> Campaign:
        return Campaign(dir=work(), doc={}, ledger={}, seal={}, run_id="R0", eval_dir=work(),
                        metrics={}, threshold=None, shaped_dir=Path(self.fixture.shaped),
                        shaped_manifest=self.shaped_manifest, recipe=self.recipe,
                        spec=self.spec, raw=self.raw, fitted=self.fitted)

    def _lay_out(self) -> None:
        root, pkg = self.root, self.root / self.package
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(self.fixture.package_dir, pkg)
        # the serving package set in the Predictor's module, the two adapters at their ports
        rt.render(PACKAGE_TEMPLATES, root, self.pipeline, self.spec, only=list(SUPPORT_SET),
                  into="Predictor", shapes=("window",))
        rt.render(PACKAGE_TEMPLATES, root, self.pipeline, self.spec,
                  only=["predict.py", "monitor.py"], shapes=("window",))
        shutil.copyfile(self.fixture.transform, pkg / "features" / "preprocess.py")
        shutil.copyfile(self.fixture.fitted, root / "fitted.json")
        (root / "predict.py").write_text(self._cli_text(), encoding="utf-8")
        _json(root / "netspec.json", self.spec)
        _json(root / "schema.json", self.schema)
        _json(root / "monitor-reference.json", self._monitor_reference())
        render_serve(root / "serve")
        self.model_version = self._weights()
        self.write_manifest()
        self._examples()
        self.write_manifest()

    def _cli_text(self) -> str:
        """`bundle/predict.py` with the placeholders serving_pack fills."""
        text = (SERVING / "bundle" / "predict.py").read_text(encoding="utf-8")
        text = text.replace("{package}", self.package).replace("{env}", ENV)
        for key, module in bundle_slice.bundle_modules(self.root / self.package).items():
            text = text.replace("{mod_" + key + "}", module)
        return text

    def _weights(self) -> str:
        """A seeded network of the spec, saved through the bundle's own ModelRegistry."""
        sys.path.insert(0, str(self.root))
        try:
            domain = importlib.import_module(f"{self.package}.domain")
            network = importlib.import_module(f"{self.package}.model.network")
            store = importlib.import_module(f"{self.package}.model.store")
            spec = domain.NetSpec.from_dict(self.spec)
            torch.manual_seed(MODEL_VERSION_SEED)
            model = network.Network.build(spec).eval()
            ref = store.ModelRegistry().save(model, self.root / "weights.pt", spec=spec)
        finally:
            sys.path.remove(str(self.root))
            for mod in [m for m in sys.modules if m.split(".")[0] == self.package]:
                del sys.modules[mod]
        return ref.weights_digest

    def _monitor_reference(self) -> dict:
        """One psi check on a numeric input, measured on the train part as pack measures it,
        with a row floor a batch of the stream reaches."""
        name = next(c["name"] for c in self.schema["input"]
                    if c.get("dtype") != "string" and c.get("role") != "autoregressive")
        train = pd.to_numeric(self.part("train")[name], errors="coerce").dropna()
        edges = np.quantile(train, np.linspace(0, 1, 6))
        edges[0], edges[-1] = -1e12, 1e12
        counts = np.histogram(train, bins=edges)[0]
        return {"schema": "vega/monitor-reference@2", "concerns": [], "plan": "test",
                "checks": [{"id": "p1", "kind": "psi", "column": name, "bins": edges.tolist(),
                            "reference": (counts / counts.sum()).tolist(), "min_rows": 5,
                            "concern": "[test]", "title": "input drift"}], "note": ""}

    def _examples(self) -> None:
        """Context rows first (one series' last `length − 1` valid rows), then its first test
        rows; `__row__` names the answered rows, the expected values the bundle's own answer
        (the consumer's self-test holds the bundle to them)."""
        series = self.sample["series_by"]
        valid, test = self.part("valid"), self.part("test")
        key = test[series].iloc[0]
        rows = pd.concat([valid[valid[series] == key].tail(EXAMPLE_CONTEXT),
                          test[test[series] == key].head(EXAMPLE_ROWS)], ignore_index=True)
        ex = self.root / "examples"
        ex.mkdir(parents=True, exist_ok=True)
        rows.to_csv(ex / "rows.csv", index=False)
        cli = load_cli(self)
        got = cli.load(self.root).predict(cli.read_rows(ex / "rows.csv", self.root))
        answered = np.flatnonzero(got["answered"].to_numpy(dtype=bool))
        fields = [f for f, _ in self.output["compared"]]
        expected = got.iloc[answered][fields].reset_index(drop=True).assign(__row__=answered)
        expected.to_parquet(ex / "expected.parquet", index=False)
        self.example_rows = len(answered)

    def write_manifest(self, changes=None) -> dict:
        """MANIFEST.json over the files on disk now, as bundle_manifest pins and roles them."""
        files = bundle_manifest.pinned_files(self.root)
        doc = {"schema": "vega/bundle@1", "package": self.package, "input_schema": "schema.json",
               "model_version": self.model_version, "output": self.output,
               "sample": self.sample,
               "roundtrip": {"tolerance": Tolerance.for_scores([1.0]).as_json()},
               "files": files, "roles": bundle_manifest.file_roles(files, self.package)}
        doc.update(changes or {})
        _json(self.root / "MANIFEST.json", doc)
        return doc


def render_serve(out: Path) -> None:
    """The serve templates a stream bundle carries, with serving_pack's fills, none left."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "__init__.py").write_text('"""serve (rendered for test_consumer)."""\n',
                                     encoding="utf-8")
    for name in SERVE_FILES:
        text = (SERVE_TEMPLATES / name).read_text(encoding="utf-8")
        for key, value in FILLS.items():
            text = text.replace(key, value)
        left = sorted(set(re.findall(r"\{\{[A-Z_]+\}\}", text)))
        assert not left, f"serve/{name} keeps unfilled placeholder(s) {left}"
        (out / name).write_text(text, encoding="utf-8")


def load_cli(built: Built):
    """The bundle's `predict.py` as the consumer imports it: the module `predict`, from the
    bundle directory (one bundle per process: its Recipe reads VEGA_FITTED at import)."""
    module = sys.modules.get("predict")
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location("predict", built.root / "predict.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["predict"] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=None)
def built() -> Built:
    return Built()


@functools.lru_cache(maxsize=None)
def consumer_module():
    """The rendered `serve.consumer`, imported from the bundle (in-process tests)."""
    b = built()
    if str(b.root) not in sys.path:
        sys.path.insert(0, str(b.root))
    load_cli(b)
    return importlib.import_module("serve.consumer")


def consumer(monitor="moved", buffer=None):
    """A Consumer of the bundle, as `main` builds it (a fresh history each time)."""
    mod, b = consumer_module(), built()
    root, bundle = mod.open_bundle(b.root)
    made = mod.build(root, bundle, (), monitor)
    if buffer is not None:
        made = mod.Consumer(made._predictor, made._contract, buffer, made._watch)
    return made


@functools.lru_cache(maxsize=None)
def shared():
    """One consumer whose Predictor and contract the reference answers are read through."""
    return consumer("off")


def json_rows(frame: pd.DataFrame) -> list:
    """Rows as a caller sends them (the contract's own JSON form of a frame)."""
    mod = consumer_module()
    return shared()._contract.tables.json_rows(frame)


def batches(rows: list, n: int) -> list:
    return [rows[i:i + n] for i in range(0, len(rows), n)]


def run_consumer(rows: list, n: int, made=None) -> list:
    """The records the consumer writes for `rows` read `n` at a time."""
    made = made or consumer()
    out = []
    made.run(batches(rows, n), out.append)
    return out


def envelopes(records) -> list:
    return [r for r in records if "predictions" in r]


def reference(frame: pd.DataFrame, history=None) -> pd.DataFrame:
    """The Predictor's answer to the whole stream in one request."""
    return shared()._predictor.predict(frame, history=history)


def answers_of(records) -> list:
    return [p for r in envelopes(records) for p in r["predictions"]]


def assert_same_answers(got: list, frame: pd.DataFrame, expected: pd.DataFrame, tol: float):
    """Row for row: answered, each horizon value within `tol`, the warnings."""
    rows = shared()._contract.answers.rows(expected)
    assert len(got) == len(rows) == len(frame)
    for i, (mine, theirs) in enumerate(zip(got, rows)):
        assert mine["answered"] == theirs["answered"], (i, mine, theirs)
        assert mine["warnings"] == theirs["warnings"], (i, mine["warnings"], theirs["warnings"])
        values, want = mine.get("values") or {}, theirs.get("values") or {}
        assert set(values) == set(want), (i, values, want)
        for h, v in want.items():
            if v is None:
                assert values[h] is None, (i, h)
            else:
                assert abs(values[h] - v) <= tol, (i, h, values[h], v)


def window_notes(answer: dict) -> list:
    """The window's own warnings of an answer (an incomplete window, lags without history);
    the schema guard's range notes aside."""
    return [w for w in answer["warnings"] if w.startswith(WINDOW_NOTES)]


def tolerance(expected: pd.DataFrame) -> float:
    fields = [f for f, _ in built().output["compared"]]
    return Tolerance.for_scores(expected[fields].to_numpy(dtype=np.float64,
                                                          na_value=np.nan)[
        expected["answered"].to_numpy(dtype=bool)].ravel().tolist()).value


def cli(args, stdin: str = "", root=None) -> subprocess.CompletedProcess:
    """`python -m serve.consumer ARGS` run in the bundle directory, as a user runs it."""
    root = root or built().root
    env = {k: v for k, v in os.environ.items() if k not in ("VEGA_FITTED", f"{ENV}_BUNDLE")}
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-m", "serve.consumer", *args], cwd=root,
                          input=stdin, capture_output=True, text=True, env=env, timeout=300)


def jsonl(rows: list) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def copy_bundle(name: str) -> Path:
    """A copy of the bundle (for a test that alters it), its caches left behind."""
    out = work() / name
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(built().root, out, ignore=shutil.ignore_patterns("__pycache__"))
    return out


# =========================================================================== 1. JSON lines

ENVELOPE_KEYS = {"api_version", "model_version", "contract_version", "feature_contract_version",
                 "n_rows", "output_kind", "batch", "positions", "predictions"}


def test_json_lines_in_and_out_on_the_window_bundle():
    b = built()
    test = b.part("test")
    rows = json_rows(test)
    done = cli(["--batch-rows", "7"], stdin=jsonl(rows))
    assert done.returncode == 0, done.stderr
    lines = [json.loads(line) for line in done.stdout.splitlines()]
    assert len(lines) == -(-len(rows) // 7), len(lines)
    keys = shared()._contract.answers.keys()
    positions = []
    for k, line in enumerate(lines, 1):
        assert ENVELOPE_KEYS | {"monitor"} == set(line), sorted(line)
        assert line["api_version"] == "v1" and line["output_kind"] == "forecast"
        assert line["model_version"] == b.model_version
        assert line["contract_version"] == digest(b.root / "schema.json")
        assert line["batch"] == k and line["n_rows"] == len(line["predictions"]) \
            == len(line["positions"])
        for p in line["predictions"]:
            assert tuple(p) == tuple(key for key in keys if key in p), p
            assert set(p) == set(keys), (set(p), keys)
            assert (p["values"] is None) != p["answered"], p
            if p["answered"]:
                assert set(p["values"]) == {"h1", "h2", "h3"}
        monitor = line["monitor"]
        assert set(monitor) == {"moved", "readings"}
        assert monitor["moved"] == len(monitor["readings"])
        positions += line["positions"]
    assert positions == list(range(1, len(rows) + 1))
    answered = sum(p["answered"] for line in lines for p in line["predictions"])
    n_series, need = test[b.sample["series_by"]].nunique(), b.sample["length"] - 1
    assert answered == len(rows) - n_series * need, answered
    summary = done.stderr.strip().splitlines()[-1]
    assert summary == (f"OK: {len(lines)} batch(es): {len(rows)} row(s) read, {answered} "
                       f"answered, {len(rows) - answered} unanswered (window incomplete), "
                       "0 refused"), summary


def test_a_history_file_seeds_every_series():
    b = built()
    past = b.tail("valid", b.sample["history_rows"])
    path = work() / "history.csv"
    past.to_csv(path, index=False)
    rows = json_rows(b.part("test"))
    out = work() / "answers.jsonl"
    done = cli(["--history", str(path), "--batch-rows", "50", "--output", str(out),
                "--monitor", "off"], stdin=jsonl(rows))
    assert done.returncode == 0, done.stderr
    assert done.stdout == ""
    held = len(past)
    assert f"ok  history: {held} row(s) held from {path}" in done.stderr, done.stderr
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all("monitor" not in line for line in lines)
    got = [p for line in lines for p in line["predictions"]]
    assert len(got) == len(rows) and all(p["answered"] for p in got)
    assert not any(window_notes(p) for p in got), [window_notes(p) for p in got]


# =========================================================================== 2. the history

def test_the_history_buffer_completes_windows_across_batches():
    b = built()
    test = b.part("test")
    rows = json_rows(test)
    length, series = b.sample["length"], b.sample["series_by"]
    n = 5
    # no batch holds a window of any one series: only the buffer can complete one
    assert n < length - 1
    records = run_consumer(rows, n)
    assert all("predictions" in r for r in records), records[:1]
    got = answers_of(records)
    frame = shared()._contract.tables.frame(rows)
    whole = reference(frame)
    assert_same_answers(got, frame, whole, tolerance(whole))
    rank = test.groupby(series, sort=False).cumcount().to_numpy()
    answered = np.asarray([p["answered"] for p in got])
    assert (answered == (rank >= length - 1)).all()
    late = [p for p, r in zip(got, rank) if r >= b.sample["history_rows"]]
    assert late and not any(window_notes(p) for p in late)
    assert all(window_notes(p) for p, r in zip(got, rank) if r < b.sample["history_rows"])
    # the control: without the buffer, a batch of five rows answers nothing
    mod = consumer_module()
    none = answers_of(run_consumer(rows, n, consumer(buffer=mod.HistoryBuffer(0, series))))
    assert len(none) == len(rows) and not any(p["answered"] for p in none)


def test_a_seeded_history_answers_every_row_as_one_request_does():
    b = built()
    past = b.tail("valid", b.sample["history_rows"])
    rows = json_rows(b.part("test"))
    made = consumer()
    assert made.seed(past) == len(past)
    got = answers_of(run_consumer(rows, 9, made))
    frame = made._contract.tables.frame(rows)
    history = made._contract.tables.frame(json_rows(past))
    whole = reference(frame, history)
    assert whole["answered"].all()
    assert_same_answers(got, frame, whole, tolerance(whole))
    # a longer past is cut to the last history_rows rows of each series
    again = consumer()
    assert again.seed(b.part("valid")) == len(past)


def test_the_buffer_answers_as_the_whole_stream_whatever_the_batch_size():
    b = built()
    rows = json_rows(b.part("test"))
    frame = shared()._contract.tables.frame(rows)
    whole = reference(frame)
    tol = tolerance(whole)
    for n in (1, 13, 64, len(rows)):
        assert_same_answers(answers_of(run_consumer(rows, n)), frame, whole, tol)


# =========================================================================== 3. buffer, source

def test_the_buffer_keeps_the_last_rows_of_each_series_in_arrival_order():
    mod = consumer_module()
    buf = mod.HistoryBuffer(2, "s")
    buf.extend(pd.DataFrame({"s": ["a", "b", "a"], "t": [1, 1, 2]}))
    buf.extend(pd.DataFrame({"s": ["a", "c", "b", "b"], "t": [3, 1, 2, 3]}))
    held = buf.frame()
    assert held.to_dict("records") == [{"s": "a", "t": 2}, {"s": "a", "t": 3},
                                       {"s": "c", "t": 1}, {"s": "b", "t": 2},
                                       {"s": "b", "t": 3}]
    held.loc[0, "t"] = 99
    assert buf.frame().loc[0, "t"] == 2 and len(buf) == 5
    whole = mod.HistoryBuffer(2)
    whole.extend(pd.DataFrame({"s": ["a", "b", "a"], "t": [1, 1, 2]}))
    assert whole.frame()["t"].tolist() == [1, 2]
    empty = mod.HistoryBuffer(0, "s")
    empty.extend(pd.DataFrame({"s": ["a"], "t": [1]}))
    assert len(empty) == 0 and empty.frame().empty
    for bad in (lambda: mod.HistoryBuffer(-1), lambda: buf.extend([{"s": "a"}]),
                lambda: mod.HistoryBuffer(2, "series").extend(pd.DataFrame({"s": ["a"]}))):
        try:
            bad()
        except (ValueError, TypeError, KeyError):
            continue
        raise AssertionError("refused input accepted")


def test_the_sample_record_sizes_the_buffer():
    mod, b = consumer_module(), built()
    buf = mod.HistoryBuffer.for_sample(b.sample)
    assert (buf.size, buf.series_by) == (b.sample["length"] - 1 + b.fixture.lag_depth,
                                         b.sample["series_by"])
    older = {k: v for k, v in b.sample.items() if k != "history_rows"}
    assert mod.HistoryBuffer.for_sample(older).size == buf.size


def test_json_lines_batches_skip_blanks_and_keep_unreadable_lines_in_place():
    mod = consumer_module()
    text = '{"a": 1}\n\n  \n{"a": 2}\nnot json\n[1, 2]\n{"a": NaN}\n'
    got = list(mod.json_lines(io.StringIO(text), 2))
    assert [len(b) for b in got] == [2, 2, 1]
    assert got[0] == [{"a": 1}, {"a": 2}]
    assert isinstance(got[1][0], mod.Unreadable) and got[1][0].reason.startswith("line 5:")
    assert got[1][1] == [1, 2] and np.isnan(got[2][0]["a"])
    try:
        list(mod.json_lines(io.StringIO(text), 0))
    except ValueError:
        pass
    else:
        raise AssertionError("a batch of no rows accepted")


# =========================================================================== 4. refusals, monitor

def test_a_row_that_breaks_the_contract_is_refused_and_the_rest_answered():
    mod, b = consumer_module(), built()
    rows = json_rows(b.part("test").head(6))
    numeric = next(c["name"] for c in b.schema["input"]
                   if c.get("dtype") != "string" and c.get("role") != "autoregressive")
    bad = dict(rows[2], **{numeric: "warm"})
    extra = dict(rows[3], unexpected=1)
    batch = [rows[0], rows[1], bad, extra, mod.Unreadable("line 5: not JSON (x)"), [1], rows[4]]
    made = consumer()
    records = made.handle(batch)
    assert len(records) == 2
    problem, answer = records
    assert problem["status"] == 422 and problem["batch"] == 1
    assert problem["title"] == mod.REFUSED
    errors = {(e["column"], e["error"]): e for e in problem["errors"]}
    assert errors[(numeric, "expects a number, null, or one of the tokens Infinity, "
                            "-Infinity, NaN")]["rows"] == [3]
    assert errors[("unexpected", "unknown column")]["rows"] == [4]
    assert errors[(None, "line 5: not JSON (x)")]["rows"] == [5]
    assert errors[(None, mod.NOT_AN_OBJECT)]["rows"] == [6]
    assert answer["positions"] == [1, 2, 7] and answer["n_rows"] == 3
    assert len(made._buffer) == 3
    tally = made.tally
    assert (tally.batches, tally.rows, tally.refused) == (1, 7, 4)
    # the next batch goes on counting positions after the refused rows
    more = made.handle([rows[5]])
    assert more[-1]["positions"] == [8] and more[-1]["batch"] == 2
    assert made.handle([]) == [] and made.tally.batches == 2


def test_the_answers_a_refused_row_takes_out_of_their_window_say_so():
    """R1 / bundle-contract §7.9: a refused row never joins the history, so the next
    `history_rows` answers of its series are answered over a window without it — each carries
    the gap warning; earlier rows, and other series, do not."""
    mod, b = consumer_module(), built()
    gaps = importlib.import_module("serve.gaps")
    series, size = b.sample["series_by"], b.sample["history_rows"]
    test = b.part("test")
    one = test[test[series] == test[series].iloc[0]]
    other = test[test[series] != test[series].iloc[0]].head(4)
    rows, others = json_rows(one.head(size + 6)), json_rows(other)
    numeric = next(c["name"] for c in b.schema["input"]
                   if c.get("dtype") != "string" and c.get("role") != "autoregressive")
    rows[3] = dict(rows[3], **{numeric: "warm"})                       # refused, position 4
    made = consumer("off")
    records = made.handle(rows[:5]) + made.handle(rows[5:] + others)
    answers = [p for r in envelopes(records) for p in r["predictions"]]
    flagged = [gaps.GAP_WARNING in (p.get("warnings") or []) for p in answers]
    mine = flagged[:len(rows) - 1]
    assert mine[:3] == [False, False, False], mine             # before the refused row
    assert mine[3:3 + size] == [True] * size, mine             # the next history_rows of it
    assert not any(mine[3 + size:]), mine                       # then the window is whole again
    assert not any(flagged[len(rows) - 1:]), flagged           # another series is untouched
    made = consumer("off")
    made.handle([mod.Unreadable("line 1: not JSON (x)")])      # no series: every series
    later = envelopes(made.handle(others))[0]["predictions"]
    assert all(gaps.GAP_WARNING in p["warnings"] for p in later[:size])


class _Refusing:
    """A Predictor that refuses every request, as the window assembler refuses rows without a
    column the recipe reads (a KeyError) or history of another shape (a ValueError)."""

    def __init__(self, exc):
        self.exc = exc

    def info(self):
        return {"model_version": "sha256:" + "0" * 64}

    def predict(self, frame, history=None):
        raise self.exc


def test_rows_the_predictor_refuses_are_a_problem_line_and_stay_out_of_the_history():
    mod, base = consumer_module(), shared()
    rows = json_rows(built().part("test").head(4))
    for exc, text in ((KeyError("the rows do not carry x"), "the rows do not carry x"),
                      (ValueError("a graph sample takes no history"),
                       "a graph sample takes no history")):
        buf = mod.HistoryBuffer(13, built().sample["series_by"])
        made = mod.Consumer(_Refusing(exc), base._contract, buf)
        records = made.handle(rows)
        assert len(records) == 1, records
        problem = records[0]
        assert problem["status"] == 422 and problem["title"] == mod.UNANSWERED
        assert problem["detail"] == text and problem["batch"] == 1
        assert problem["errors"] == [{"column": None, "error": text, "rows": [1, 2, 3, 4],
                                      "count": 4}]
        assert len(buf) == 0 and made.tally.refused == 4 and made.tally.answered == 0


def test_the_cli_refuses_bad_arguments_before_loading_the_bundle():
    for args, said in ((["--batch-rows", "0"], "a batch holds at least one row"),
                       (["--input", str(work() / "absent.jsonl")], "no such file"),
                       (["--monitor", "sometimes"], "invalid choice"),
                       (["--include", "everything"], "invalid choice")):
        done = cli(args)
        assert done.returncode == 2, (args, done.returncode, done.stderr)
        assert said in done.stderr, (args, done.stderr)
        assert done.stdout == ""


def test_the_monitor_reads_each_batch_as_asked():
    b = built()
    rows = json_rows(b.part("test").head(40))
    every = envelopes(run_consumer(rows, 20, consumer("all")))
    for line in every:
        readings = line["monitor"]["readings"]
        assert [r["check"] for r in readings] == ["p1"], readings
        assert readings[0]["title"] == "input drift" and readings[0]["measure"] == "psi"
        assert readings[0]["window_rows"] == sum(p["answered"] for p in line["predictions"])
        assert line["monitor"]["moved"] == sum(r["verdict"] in ("watch", "alert")
                                               for r in readings)
    moved = envelopes(run_consumer(rows, 20, consumer("moved")))
    for line in moved:
        assert all(r["verdict"] in ("watch", "alert") for r in line["monitor"]["readings"])
    off = envelopes(run_consumer(rows, 20, consumer("off")))
    assert off and all("monitor" not in line for line in off)
    try:
        consumer("sometimes")
    except ValueError as exc:
        assert "sometimes" in str(exc)
    else:
        raise AssertionError("an unknown monitor mode accepted")


# =========================================================================== 5. start refusals

def test_a_sample_that_is_not_a_window_is_refused():
    root = copy_bundle("rows-bundle")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["sample"] = {"shape": "row"}
    _json(root / "MANIFEST.json", manifest)
    done = cli([], root=root)
    assert done.returncode == 2, (done.returncode, done.stderr)
    assert "ERROR: stream mode answers window samples" in done.stderr and "'row'" in done.stderr


def test_an_altered_file_is_refused_before_a_row_is_read():
    root = copy_bundle("altered-bundle")
    schema = root / "schema.json"
    schema.write_text(schema.read_text(encoding="utf-8") + " ", encoding="utf-8")
    done = cli([], stdin=jsonl(json_rows(built().part("test").head(3))), root=root)
    assert done.returncode == 2, (done.returncode, done.stderr)
    assert "schema.json is not the file MANIFEST.json pinned" in done.stderr
    assert done.stdout == ""


def test_examples_the_bundle_does_not_reproduce_stop_it():
    root = copy_bundle("drifted-bundle")
    path = root / "examples" / "expected.parquet"
    expected = pd.read_parquet(path)
    field = [f for f, _ in built().output["compared"]][0]
    expected[field] = expected[field] + 1.0
    expected.to_parquet(path, index=False)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["files"]["examples/expected.parquet"] = digest(path)
    _json(root / "MANIFEST.json", manifest)
    done = cli([], stdin=jsonl(json_rows(built().part("test").head(3))), root=root)
    assert done.returncode == 3, (done.returncode, done.stderr)
    assert "ERROR: the bundle does not reproduce its examples" in done.stderr
    assert done.stdout == ""


# =========================================================================== 6. the template

def test_the_template_fills_only_serving_packs_placeholders():
    text = TEMPLATE.read_text(encoding="utf-8")
    used = set(re.findall(r"\{\{[A-Z_]+\}\}", text))
    assert used and used <= set(FILLS), sorted(used - set(FILLS))
    # the vega-pack module that renders serve/ (serving_templates since S07) fills each of
    # them: its own fills, over a bundle's package, ports and served identity, leave none
    from types import SimpleNamespace
    import serving_templates
    ident = {"title": "A thing", "description": "things", "nouns": "things", "noun": "thing",
             "cmd": "thing", "env": ENV, "port": 8000}
    ctx = SimpleNamespace(package="spike_window", mode="stream",
                          paths=SimpleNamespace(pkg=work() / "no-map"),
                          port=lambda role: {"name": role})
    mapping = serving_templates.fills(ctx, ident)
    filled = {"{{" + key + "}}" for key in mapping}
    assert used <= filled, sorted(used - filled)
    rendered = serving_templates.render_text(text, mapping, "serve/consumer.py")
    assert not re.findall(r"\{\{[A-Z_]+\}\}", rendered)


def test_the_template_is_within_the_size_discipline_and_literal_free():
    done = subprocess.run([sys.executable, str(SCRIPTS / "conformance.py"), "--size-only",
                           str(TEMPLATE), "--size-severity", "error"], capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not re.search(r"\bC1[123]\b.*consumer", done.stdout), done.stdout
    if LINT.is_file():
        lint = subprocess.run([sys.executable, str(LINT), str(TEMPLATE)], capture_output=True,
                              text=True, timeout=120)
        assert lint.returncode == 0, lint.stdout + lint.stderr


# =========================================================================== the script runner

def _main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}" if VERBOSE else f"ok  {name}")
        except Exception as exc:  # noqa: BLE001  (the runner reports every failure)
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
