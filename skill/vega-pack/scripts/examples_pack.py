"""
examples_pack — the rows that ship with the bundle, per sample shape, and what the bundle must
still say about them: `examples/rows.csv` and `examples/expected.parquet`, the evidence the
bundle's `check_examples` (`tests/test_roundtrip.py`, the service's start-up self-test, the
notebook) replays (bundle-contract §5; build plan S10, D2 §8.3).

    examples_pack.write(ctx, predictor) -> {"examples": …}
    examples_pack.write_request(ctx) -> {}          after serving_pack.render

`expected.parquet` always carries `__row__` (the row of `rows.csv` whose answer it is) and every
compared and decided field of the output record. `test` examples carry the evaluation's own
answer — the stored compared values and the stored decision read through the bundle's reader —
so the shipped test proves the bundle reproduces the *evaluation*; they are answered here the way
`check_examples` answers them before they ship. `synthetic` examples (a user stance, `--examples
synthetic`) carry the predictor's outputs at pack time, so they prove the bundle reproduces
*itself* — a weaker claim, and the manifest says which one is made.

| shape  | `rows.csv` (test)                                   | `__row__`                          |
|--------|-----------------------------------------------------|------------------------------------|
| row    | the first n test rows                               | 0 … n−1                            |
| window | per series from the test part's start: `length − 1` | the rows with `length − 1`         |
|        | context rows, then the rows compared (series after  | predecessors that the evaluation   |
|        | series until n are compared)                        | scored                             |
| node   | the first n test rows                               | 0 … n−1                            |
| graph  | whole test graphs until at least n rows             | each graph's first row (+graph_id) |

A window example starts at the test part's start, so its lags restart where the evaluation's
restarted and its compared windows lie inside the test part: they reproduce exactly, R-A6's
boundary included. `synthetic`: rows resampled column by column (row, node), one contiguous run of
`(length − 1) + n` train rows of one series (window), whole train graphs (graph level).

`examples/request.json` is one ready request body for the shape, encoded by the service's own
contract module: `{"rows": [first]}` (row, node), `{"rows": [anchor], "history": [its
predecessors]}` (window), the rows of one graph (graph level). It is written after the service
is rendered, and only when there is a service.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import roundtrip_plan as rp
from pack_context import PackExit, out_ok, out_warn
from roundtrip_compare import compare, tolerance_for
from vega_pack_core import CATALOG, digest_file, read_parquet

__all__ = ["Chosen", "write", "write_request"]

ROW = "__row__"
SYNTHETIC_SEED = 7


@dataclass(frozen=True, eq=False)
class Chosen:
    """The rows a bundle ships and the answers expected of them: `rows` as a caller sends
    them, `expected` (one row per compared answer: `__row__`, the compared and decided
    fields, a graph's id), and the window's `context_rows` (else None)."""

    rows: Any
    expected: Any
    context_rows: int | None = None


def write(ctx, predictor) -> dict:
    ev = rp.evaluated(ctx, predictor)
    want = str(ctx.options.examples or CATALOG["examples"]["default_rows"])
    if want == "synthetic":
        chosen = _SYNTHETIC[ev.plan.shape](ev, ctx, predictor,
                                           int(CATALOG["examples"]["default_rows"]))
        kind, source = "synthetic", "predictor"
        out_warn("examples are synthetic by stance: the shipped test proves the bundle "
                 "reproduces itself, not that it reproduces the evaluation", ctx.markers)
    else:
        chosen = _TEST[ev.plan.shape](ev, max(1, int(want)))
        _reproduced(chosen, predictor, ev)
        kind, source = "test", f"eval/{ctx.campaign.run_id}/predictions.test.parquet"
    return {"examples": _written(ctx, chosen, kind, source)}


def _written(ctx, chosen, kind, source) -> dict:
    paths, markers = ctx.paths, ctx.markers
    paths.examples.mkdir(parents=True, exist_ok=True)
    left = [c for c in (ctx.schema.get("ignored") or []) if c in ctx.campaign.raw.columns
            and c not in chosen.rows.columns]
    if left:
        out_ok(f"examples/rows.csv carries the request columns only; {', '.join(left)} left "
               "out (the predictor ignores them, and the label is part of the held-out "
               "answer)", markers)
    chosen.rows.to_csv(paths.examples / "rows.csv", index=False)
    chosen.expected.to_parquet(paths.examples / "expected.parquet", index=False)
    record = {"kind": kind, "rows": int(len(chosen.rows)),
              "compared_rows": int(len(chosen.expected)), "source": source,
              "digest": digest_file(paths.examples / "rows.csv")}
    if chosen.context_rows is not None:
        record["context_rows"] = int(chosen.context_rows)
    out_ok(f"examples: {len(chosen.rows)} {kind} row(s), {len(chosen.expected)} with an "
           "expected answer (examples/expected.parquet at __row__)"
           + (f", {chosen.context_rows} context row(s) first per series"
              if chosen.context_rows is not None else ""), markers)
    return record


def _reproduced(chosen, predictor, ev) -> None:
    """The shipped test examples answered as `check_examples` answers them (rows.csv alone,
    no history), against their expected answers at the round trip's tolerance — before they
    ship, so a bundle whose own test would fail is refused here."""
    try:
        got = predictor.predict(chosen.rows.reset_index(drop=True))
    except (KeyError, ValueError, RuntimeError) as exc:
        raise PackExit(3, f"the shipped examples cannot be answered as the bundle's "
                          f"check_examples answers them (examples/rows.csv, no history): "
                          f"{type(exc).__name__}: {exc}") from exc
    fields = chosen.expected.drop(columns=[ROW, "graph_id"], errors="ignore")
    result = compare(got, chosen.expected[ROW].to_numpy(), fields, ev.output,
                     tolerance_for(ev.stored.frame, ev.output))
    if not result.passed:
        bad = result.first_bad
        raise PackExit(3, f"the shipped examples do not reproduce the evaluation: "
                          f"{bad['column']} ({bad['kind']}) at expected row {bad['row']} — the "
                          f"bundle answers {bad['bundle']}, the evaluation stored "
                          f"{bad['evaluation']}")


# --------------------------------------------------------------------------- test examples

def _expected(ev, stored_index, rows_at, extra=None):
    """The evaluation's answers at stored rows `stored_index`, each for `rows.csv` row
    `rows_at`, `__row__` first."""
    import numpy as np
    frame = ev.expected.iloc[list(stored_index)].reset_index(drop=True)
    frame.insert(0, ROW, np.asarray(rows_at, dtype=np.int64))
    for name, values in (extra or {}).items():
        frame.insert(1, name, values)
    return frame


def _stored_at(plan) -> dict:
    """Raw position -> the stored prediction it answers."""
    return {int(r): i for i, r in enumerate(plan.stored_row_ids)}


def _first_rows(ev, n) -> Chosen:
    """row and node level: the first n test rows, each with its stored answer."""
    plan = ev.plan
    n = min(n, len(plan.query))
    at = _stored_at(plan)
    stored = [at[int(p)] for p in plan.positions[:n]]
    return Chosen(plan.query.head(n).reset_index(drop=True), _expected(ev, stored, range(n)))


def _window_rows(ev, n) -> Chosen:
    """Series by series from the test part's start: the `length − 1` context rows, then rows
    until n compared rows (stored, `length − 1` predecessors in the file) are reached."""
    import numpy as np
    plan, length = ev.plan, int(ev.sample.get("length") or 1)
    at, picked, compared, have = _stored_at(plan), [], [], 0
    for code in dict.fromkeys(plan.series.tolist()):
        if have >= n:
            break
        mine = np.flatnonzero(plan.series == code)
        mine = mine[np.argsort(plan.rank[mine], kind="stable")]
        for j in mine[: (length - 1) + (n - have)]:
            if plan.rank[j] >= length - 1 and int(plan.positions[j]) in at:
                compared.append((len(picked), at[int(plan.positions[j])]))
                have += 1
            picked.append(int(j))
    rows = plan.query.iloc[picked].reset_index(drop=True)
    expected = _expected(ev, [s for _, s in compared], [r for r, _ in compared])
    return Chosen(rows, expected, length - 1)


def _graph_rows(ev, n) -> Chosen:
    """Whole test graphs until at least n rows; each graph's answer at its first row."""
    plan = ev.plan
    ids = [rp.id_text(g) for g in plan.query[plan.group_column].to_numpy()]
    stored = {g: i for i, g in enumerate(plan.graph_ids)}
    chosen, rows = [], []
    for g in dict.fromkeys(ids):
        if len(rows) >= n:
            break
        chosen.append(g)
        rows += [j for j, other in enumerate(ids) if other == g]
    frame = plan.query.iloc[rows].reset_index(drop=True)
    first = [[rp.id_text(v) for v in frame[plan.group_column]].index(g) for g in chosen]
    return Chosen(frame, _expected(ev, [stored[g] for g in chosen], first,
                                   {"graph_id": chosen}))


_TEST = {"row": _first_rows, "node": _first_rows, "window": _window_rows, "graph": _graph_rows}


# --------------------------------------------------------------------------- synthetic examples

def _own(got, at, output, extra=None):
    """The predictor's own answers at rows `at` as the expected frame (`__row__` first)."""
    import numpy as np
    import pandas as pd
    fields = [str(p[0]) for p in (output.get("compared") or ())] + \
        [str(p[0]) for p in (output.get("decided") or ())]
    frame = pd.DataFrame({f: got[f].to_numpy()[np.asarray(at, dtype=np.int64)] for f in fields})
    frame.insert(0, ROW, np.asarray(at, dtype=np.int64))
    for name, values in (extra or {}).items():
        frame.insert(1, name, values)
    return frame


def _resampled(ev, ctx, predictor, n) -> Chosen:
    """row and node level: each column resampled independently from the test rows, so no
    original row survives the shuffle; the queries are independent of each other."""
    import numpy as np
    rng = np.random.default_rng(SYNTHETIC_SEED)
    rows = ev.plan.query
    n = min(n, len(rows))
    frame = rows.apply(lambda col: col.sample(n=n, replace=True,
                                              random_state=int(rng.integers(1 << 30)))
                       .reset_index(drop=True)).reset_index(drop=True)
    got = predictor.predict(frame)
    return Chosen(frame, _own(got, range(n), ev.output))


def _window_run(ev, ctx, predictor, n) -> Chosen:
    """One contiguous run of `(length − 1) + n` train rows of the longest train series."""
    import numpy as np
    split = rp.split_of(ctx.campaign.fitted)
    train = split["train"]
    codes, rank = rp.series_ranks(ev.sample, ev.raw, train)
    longest = int(np.bincount(codes).argmax())
    mine = np.flatnonzero(codes == longest)
    mine = mine[np.argsort(rank[mine], kind="stable")]
    length = int(ev.sample.get("length") or 1)
    run = train[mine[-((length - 1) + n):]]
    rows = ev.raw.rows(run, ctx.schema)
    got = predictor.predict(rows)
    answered = got["answered"].to_numpy(dtype=bool) if "answered" in got.columns \
        else np.ones(len(got), dtype=bool)
    return Chosen(rows, _own(got, np.flatnonzero(answered), ev.output), length - 1)


def _train_graphs(ev, ctx, predictor, n) -> Chosen:
    """Whole train graphs until at least n rows; each graph's own answer at its first row."""
    train = rp.split_of(ctx.campaign.fitted)["train"]
    column = ev.plan.group_column
    ids = [rp.id_text(g) for g in ev.raw.key(column, train)]
    chosen, picked = [], []
    for g in dict.fromkeys(ids):
        if len(picked) >= n:
            break
        chosen.append(g)
        picked += [int(train[j]) for j, other in enumerate(ids) if other == g]
    rows = ev.raw.rows(picked, ctx.schema)
    got = predictor.predict(rows)
    first = [[rp.id_text(v) for v in rows[column]].index(g) for g in chosen]
    return Chosen(rows, _own(got, first, ev.output, {"graph_id": chosen}))


_SYNTHETIC = {"row": _resampled, "node": _resampled, "window": _window_run,
              "graph": _train_graphs}


# --------------------------------------------------------------------------- the request

def write_request(ctx) -> dict:
    """The curl example: one ready request body for the bundle's shape, the example rows read
    with the schema's types and encoded by the service's own table codec (`serve/tables.py`,
    the row contract's) — the one reader and the one encoder, so the example is exactly a
    request the service accepts (a code as its digits, not 49188.0). A bundle without a
    service has no row contract and no request to show."""
    if not (ctx.paths.root / "serve" / "tables.py").is_file():
        return {}
    import pandas as pd
    text_cols = {c["name"]: "string" for c in ctx.schema.get("input") or []
                 if c.get("dtype") == "string"}
    rows = pd.read_csv(ctx.paths.examples / "rows.csv", dtype=text_cols)
    expected = read_parquet(ctx.paths.examples / "expected.parquet")
    sample = json.loads(ctx.paths.manifest.read_text(encoding="utf-8")).get("sample") or {}
    query, history = _request_rows(rows, expected, sample)
    codec = _serve_module(ctx.paths.root, "tables").TableCodec
    body = {"rows": codec.json_rows(query)}
    if history is not None:
        body["history"] = codec.json_rows(history)
    (ctx.paths.examples / "request.json").write_text(json.dumps(body, indent=1) + "\n",
                                                     encoding="utf-8")
    return {}


def _request_rows(rows, expected, sample) -> tuple:
    """(the request's rows, its history or None) for the shape."""
    shape = rp.shape_of(sample)
    first = int(expected[ROW].iloc[0]) if len(expected) else 0
    if shape == "window":
        series = sample.get("series_by")
        before = rows.iloc[:first]
        if series:
            before = before[before[series].astype(str) == str(rows[series].iloc[first])]
        need = int(sample.get("history_rows") or int(sample.get("length") or 1) - 1)
        return rows.iloc[[first]], before.tail(need)
    if shape == "graph":
        column = sample.get("graph_id")
        ids = [rp.id_text(v) for v in rows[column]]
        return rows[[g == ids[first] for g in ids]], None
    return rows.iloc[[0]], None


def _serve_module(root, name):
    """A module of the bundle's `serve` package, imported as the package it is (its modules
    import each other relatively) from the bundle directory, and forgotten after: nothing of
    the bundle stays importable in the packing process."""
    import importlib
    saved = list(sys.path)
    held = {k: sys.modules.pop(k) for k in list(sys.modules)
            if k == "serve" or k.startswith("serve.")}
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(f"serve.{name}")
    finally:
        sys.path[:] = saved
        for key in [k for k in sys.modules if k == "serve" or k.startswith("serve.")]:
            del sys.modules[key]
        sys.modules.update(held)
