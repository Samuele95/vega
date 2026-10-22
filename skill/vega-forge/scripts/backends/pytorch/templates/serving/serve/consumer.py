"""
serve.consumer — the stream entry point of {{PACKAGE}}: JSON lines in, JSON lines out, over the
{{PORT_PREDICTOR}} and {{PORT_MONITOR}} ports (the blueprint's serving mode `stream`); rendered
by /vega-pack from `templates/serving/serve/consumer.py` (fix the template, never this copy).
It builds nothing: the Predictor, the Monitor and the file checks come from `predict.py`, the
composition root, and rows are held to the HTTP service's contract (`serve.contract`), so a
streamed row is refused or answered as the same row inside a request's `rows` would be.

    python -m serve.consumer [--bundle DIR] [--input rows.jsonl] [--output answers.jsonl]
        [--batch-rows 256] [--history earlier.csv] [--include logits] [--monitor moved|all|off]

In: one raw row per line (a JSON object in the input schema), `--batch-rows` at a time. Out:
one line per batch, the contract's answer envelope (versions, `output_kind`, `n_rows`,
`predictions`) with `batch` and `positions` (each answer's row in the stream, from 1), and the
Monitor's readings (`monitor`) when the bundle carries a reference. A row that breaks the
contract is not answered: its batch's RFC 9457 problem line (422) comes first, naming every
offending column with the rows' positions.

History. A window row is answered when `length − 1` rows of its series precede it (a lag recipe
reads `lag_depth` more). Per series (`MANIFEST.sample.series_by`) the last `history_rows`
accepted rows go to the Predictor with the next batch: windows complete across batches, each
answer the one a single request holding the whole stream gets; a cold start answers the first
`length − 1` rows of a series `answered: false`, and `--history` seeds the buffer. A refused row
never joins it: the answers its absence reaches carry `serve.gaps`' warning. Before a row
is read every pinned file must hash to its digest and the examples reproduce (`check_examples`);
the bundle is found as `predict.py` finds it (--bundle, {{ENV}}_BUNDLE, its directory, the
working directory). Exit: 0 the source was read to its end; 2 a refusal; 3 examples differ.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional

import pandas as pd
from pydantic import ValidationError

HERE = Path(__file__).resolve().parent
CODE_HOME = HERE.parent          # where this code lives: the bundle, or an installed copy of it
# first on the path, so a same-named package earlier on PYTHONPATH cannot win the import
if str(CODE_HOME) in sys.path:
    sys.path.remove(str(CODE_HOME))
sys.path.insert(0, str(CODE_HOME))

from serve.answers import INCLUDES  # noqa: E402
from serve.contract import ApiContract  # noqa: E402
from serve.gaps import SeriesGaps  # noqa: E402
from serve.problems import ContractError, ProblemDetail, thawed  # noqa: E402

__all__ = ["Consumer", "HistoryBuffer", "SelfTestFailed", "StreamTally", "Unreadable", "build",
           "json_lines", "main", "open_bundle", "stdin_source"]

API_VERSION = "v1"
BATCH_ROWS = 256
FIRST_ROWS = 5
MONITOR_MODES = ("moved", "all", "off")
MOVED = ("watch", "alert")
MONITOR_REFERENCE = "monitor-reference.json"
STREAM_SHAPES = ("window",)
REFUSED = "Rows do not honour the input schema"
REFUSED_DETAIL = "every offending column is in errors, with the rows' stream positions"
UNANSWERED = "Rows could not be answered"
NOT_AN_OBJECT = "not a JSON object: one raw row per line"
SHAPE_REFUSED = ("stream mode answers window samples (a stream is a series read in order), and "
                 "this bundle's sample is {shape!r}: serve it online, or through predict.py")
OTHER_PREDICT = "a module 'predict' is loaded from {path}, not from beside this consumer"


class SelfTestFailed(RuntimeError):
    """The bundle does not reproduce its shipped examples: nothing is answered from it."""


@dataclass(frozen=True)
class Unreadable:
    """A line of the source that is not JSON, in the place of the row it should have been."""

    reason: str


@dataclass
class StreamTally:
    """What a consumer has read and answered so far (the summary line of `main`)."""

    batches: int = 0
    rows: int = 0
    answered: int = 0
    unanswered: int = 0
    refused: int = 0

    def line(self) -> str:
        return (f"{self.batches} batch(es): {self.rows} row(s) read, {self.answered} answered, "
                f"{self.unanswered} unanswered (window incomplete), {self.refused} refused")


class HistoryBuffer:
    """The next batch's history: per series (`series_by`, else one) the last `size` accepted raw
    rows in arrival order — `MANIFEST.sample.history_rows`, what a full answer reads, no more."""

    def __init__(self, size: int, series_by: Optional[str] = None):
        if int(size) < 0:
            raise ValueError(f"a history of {size} rows: the size is a count, 0 or more")
        self.size = int(size)
        self.series_by = series_by or None
        self._rows = pd.DataFrame()

    @staticmethod
    def for_sample(sample: Mapping) -> "HistoryBuffer":
        """`history_rows` (else `length − 1 + lag_depth`) rows per `series_by` series."""
        size = sample.get("history_rows")
        if size is None:
            size = int(sample.get("length") or 1) - 1 + int(sample.get("lag_depth") or 0)
        return HistoryBuffer(int(size), sample.get("series_by"))

    def __len__(self) -> int:
        return len(self._rows)

    def extend(self, frame: pd.DataFrame) -> None:
        """Rows after the ones held (a raw frame), keeping the last `size` of each series."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"the history holds raw rows as a DataFrame, not {type(frame)}")
        if not self.size or not len(frame):
            return
        rows = (frame.reset_index(drop=True) if self._rows.empty else
                pd.concat([self._rows, frame], ignore_index=True, sort=False))
        self._rows = self._tail(rows)

    def frame(self) -> pd.DataFrame:
        """The history, oldest first; a copy, so a reader never changes what the buffer holds."""
        return self._rows.copy()

    def _tail(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self.series_by is None:
            return rows.tail(self.size).reset_index(drop=True)
        if self.series_by not in rows.columns:
            raise KeyError(f"the rows carry no {self.series_by!r}, the column MANIFEST.sample "
                           "names each row's series by")
        last = rows.groupby(self.series_by, sort=False, dropna=False).cumcount(ascending=False)
        return rows.loc[last.to_numpy() < self.size].reset_index(drop=True)


def json_lines(stream, batch_rows: int = BATCH_ROWS) -> Iterator[list]:
    """The rows of a text stream of JSON lines, `batch_rows` at a time. A blank line is not a
    row; a line that is not JSON is an `Unreadable` in its place (later rows keep theirs)."""
    if int(batch_rows) < 1:
        raise ValueError(f"batches of {batch_rows} rows: a batch holds at least one row")
    batch = []
    for number, line in enumerate(stream, 1):
        text = line.strip()
        if not text:
            continue
        try:
            batch.append(json.loads(text))
        except ValueError as exc:
            batch.append(Unreadable(f"line {number}: not JSON ({getattr(exc, 'msg', exc)})"))
        if len(batch) >= int(batch_rows):
            yield batch
            batch = []
    if batch:
        yield batch


def stdin_source(batch_rows: int = BATCH_ROWS) -> Iterator[list]:
    """The default source: JSON lines on standard input. Any iterable of row batches is one."""
    return json_lines(sys.stdin, batch_rows)


class Consumer:
    """Batches of raw rows -> answer envelopes, with the stream's history carried between them.
    `predictor`: the bundle's Predictor; `contract`: its `ApiContract`; `buffer`: the history;
    `watch`: the Monitor's `check` (the readings that moved) or `report` (every one), or None;
    `include`: the gated answer keys shown (`logits`, `reconstruction`, `embedding`)."""

    def __init__(self, predictor, contract: ApiContract, buffer: HistoryBuffer,
                 watch: Optional[Callable] = None, include: Iterable[str] = ()):
        self._predictor, self._contract, self._buffer = predictor, contract, buffer
        self._watch, self._include = watch, tuple(include)
        self._model_version = (predictor.info() or {}).get("model_version")
        self.tally, self._gaps = StreamTally(), SeriesGaps(buffer.size, buffer.series_by)

    def run(self, source: Iterable[list], sink: Callable[[dict], None]) -> int:
        """Every batch of `source` answered into `sink`, in order; returns the batches handled."""
        before = self.tally.batches
        for batch in source:
            for record in self.handle(batch):
                sink(record)
        return self.tally.batches - before

    def handle(self, batch) -> list:
        """One batch's records: its refusal when rows break the contract, then the answer."""
        items = list(batch)
        if not items:
            return []
        first, self.tally.batches = self.tally.rows + 1, self.tally.batches + 1
        self.tally.rows += len(items)
        rows, positions, errors = self._accepted(items, first)
        self.tally.refused += len(items) - len(rows)
        self._gaps.refused_in(items, first, positions)
        out = [self._refusal(REFUSED, REFUSED_DETAIL, errors)] if errors else []
        if rows:
            out.append(self._answer(rows, positions))
        return out

    def seed(self, table: pd.DataFrame) -> int:
        """Rows before the stream (a raw table) into the history, its tail per series held to
        the row contract as an upload is; returns the rows the buffer holds."""
        tail = HistoryBuffer(self._buffer.size, self._buffer.series_by)
        tail.extend(table)
        frame, _truth = self._contract.tables.accept(tail.frame())
        self._buffer.extend(frame)
        return len(self._buffer)

    def _accepted(self, items: list, first: int) -> tuple:
        """(the rows the row model accepts, their stream positions, the refusal's errors)."""
        objects = [i for i, item in enumerate(items) if isinstance(item, Mapping)]
        rows, kept, refused = self._validated([items[i] for i in objects])
        for entry in refused:
            if "rows" in entry:
                entry["rows"] = [first + objects[k] for k in entry["rows"]]
        return rows, [first + objects[k] for k in kept], _unreadable(items, first) + refused

    def _validated(self, objects: list) -> tuple:
        """(the validated rows, their indexes in `objects`, the grouped errors of the others)."""
        adapter = self._contract.models.rows
        try:
            return adapter.validate_python(objects), list(range(len(objects))), []
        except ValidationError as exc:
            errors = exc.errors()
        bad = {e["loc"][0] for e in errors if e.get("loc") and isinstance(e["loc"][0], int)}
        kept = [k for k in range(len(objects)) if k not in bad]
        rows = adapter.validate_python([objects[k] for k in kept]) if kept else []
        return rows, kept, self._contract.problems.group(errors, rows_field=None)

    def _answer(self, rows: list, positions: list) -> dict:
        """The accepted rows answered over the history, which they then join (unless refused)."""
        contract = self._contract
        records = [r.model_dump(by_alias=True) for r in rows]
        frame = contract.tables.frame(records)
        try:
            answers = self._predictor.predict(frame, history=self._buffer.frame())
        except (ValueError, KeyError) as exc:
            self.tally.refused += len(rows)
            self._gaps.refused(records, positions)
            return self._refusal(UNANSWERED, _text(exc), [_entry(_text(exc), positions)])
        body = contract.envelope(self._model_version, len(frame), API_VERSION)
        body.update(batch=self.tally.batches, positions=list(positions),
                    predictions=contract.answers.rows(answers, self._include))
        self._gaps.mark(records, body["predictions"], positions)
        if self._watch is not None:
            body["monitor"] = self._readings(frame, records, answers)
        self._buffer.extend(frame)
        done = sum(1 for p in body["predictions"] if p.get("answered"))
        self.tally.answered += done
        self.tally.unanswered += len(frame) - done
        return body

    def _readings(self, frame: pd.DataFrame, records: list, answers: pd.DataFrame) -> dict:
        """The Monitor over the batch's rows (with any truth they carried) and answers."""
        window, column = frame.copy(), self._contract.models.truth.column
        truth = [r.get(column) for r in records] if column else []
        if column not in window.columns and any(t is not None for t in truth):
            window[column] = truth
        readings = self._contract.readings(self._watch(window, answers))
        return {"moved": sum(1 for r in readings if r["verdict"] in MOVED),
                "readings": readings}

    def _refusal(self, title: str, detail: str, errors: list) -> dict:
        body = ProblemDetail(422, title, detail, tuple(errors)).body()
        body["batch"] = self.tally.batches
        return body


def _unreadable(items: list, first: int) -> list:
    """The refusal entries of what is not a row: each unreadable line, the non-objects grouped."""
    entries = [_entry(item.reason, [first + i]) for i, item in enumerate(items)
               if isinstance(item, Unreadable)]
    others = [first + i for i, item in enumerate(items)
              if not isinstance(item, (Mapping, Unreadable))]
    return entries + ([_entry(NOT_AN_OBJECT, others)] if others else [])


def _entry(error: str, positions: list) -> dict:
    """One refusal entry of rows no column is to blame for: the first rows and the count."""
    return {"column": None, "error": error, "rows": list(positions[:FIRST_ROWS]),
            "count": len(positions)}


def open_bundle(bundle_dir=None) -> tuple:
    """(the bundle directory, its composition root `predict.py`): the module imported from where
    this code lives, never found through sys.path order; it then resolves the directory."""
    path, module = CODE_HOME / "predict.py", sys.modules.get("predict")
    loaded = getattr(module, "__file__", None) if module is not None else path
    if Path(loaded or "").resolve() != path.resolve():
        raise RuntimeError(OTHER_PREDICT.format(path=loaded))
    if module is None:
        if not path.is_file():
            raise FileNotFoundError(f"no predict.py beside this code ({path}): the consumer "
                                    "runs from inside a bundle, or an installed copy of one")
        spec = importlib.util.spec_from_file_location("predict", path)
        module = sys.modules["predict"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.bundle_root(bundle_dir), module


def build(root, bundle, include: Iterable[str] = (), monitor: str = "moved") -> Consumer:
    """The consumer of the bundle at `root` (`bundle`: its `predict.py`): a window sample, every
    pinned file verified, the Predictor held to its examples (else SelfTestFailed), the Monitor
    when the bundle carries a reference (`monitor`: `moved`, `all` or `off`)."""
    if monitor not in MONITOR_MODES:
        raise ValueError(f"--monitor {monitor!r} is not one of {', '.join(MONITOR_MODES)}")
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    sample = manifest.get("sample") or {"shape": "row"}
    if sample.get("shape") not in STREAM_SHAPES:
        raise ValueError(SHAPE_REFUSED.format(shape=sample.get("shape")))
    bundle.verify(root, roles=tuple((manifest.get("roles") or {}).keys()))
    predictor = bundle.load(root)
    result = bundle.check_examples(root, predictor)
    if not result["passed"]:
        raise SelfTestFailed("the bundle does not reproduce its examples: "
                             + "; ".join(result["problems"]))
    watch = None
    if monitor != "off" and (root / MONITOR_REFERENCE).is_file():
        watched = bundle.load_monitor(root)
        watch = watched.report if monitor == "all" else watched.check
    return Consumer(predictor, ApiContract.from_bundle(root), HistoryBuffer.for_sample(sample),
                    watch, include)


def _parse(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="the bundle directory (default: {{ENV}}_BUNDLE, ...)")
    ap.add_argument("--input", help="JSON lines of raw rows (default: stdin)")
    ap.add_argument("--output", help="where the answer lines go (default: stdout)")
    ap.add_argument("--batch-rows", type=int, default=BATCH_ROWS,
                    help="rows answered together, the window the monitor reads")
    ap.add_argument("--history", help="raw rows before the stream (csv or parquet)")
    ap.add_argument("--include", action="append", default=[], choices=sorted(INCLUDES.values()),
                    help="a gated answer key to show (repeatable)")
    ap.add_argument("--monitor", default="moved", choices=MONITOR_MODES,
                    help="the monitor readings each answer line carries (default: moved)")
    a = ap.parse_args(argv)
    if a.batch_rows < 1:
        ap.error(f"--batch-rows {a.batch_rows}: a batch holds at least one row")
    if a.input and not Path(a.input).is_file():
        ap.error(f"--input {a.input}: no such file")
    return a


def _opened(path, mode: str, default):
    return contextlib.nullcontext(default) if path is None else open(path, mode, encoding="utf-8")


def _sink(stream) -> Callable[[dict], None]:
    def write(record: dict) -> None:
        # no NaN or infinity reaches a line: the codecs made them null, and a defect is loud
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
    return write


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _text(exc: Exception) -> str:
    """What a refusal says (a KeyError's message, without the quotes its str() adds)."""
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


def _start(a) -> Consumer:
    root, bundle = open_bundle(a.bundle)
    consumer = build(root, bundle, a.include, a.monitor)
    if a.history:
        kept = consumer.seed(bundle.read_rows(a.history, root))
        _say(f"ok  history: {kept} row(s) held from {a.history}")
    return consumer


def main(argv=None) -> int:
    a = _parse(argv)
    try:
        consumer = _start(a)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        _say("ERROR: " + (f"--history: {exc.detail}: {json.dumps(thawed(exc.errors))}"
                          if isinstance(exc, ContractError) else _text(exc)))
        return 3 if isinstance(exc, SelfTestFailed) else 2
    with _opened(a.input, "r", sys.stdin) as source, _opened(a.output, "w", sys.stdout) as out:
        consumer.run(json_lines(source, a.batch_rows), _sink(out))
    _say(f"OK: {consumer.tally.line()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
