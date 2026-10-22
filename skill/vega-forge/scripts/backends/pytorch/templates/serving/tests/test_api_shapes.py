"""
tests/test_api_shapes.py — how the service reads a request of this bundle's sample shape.

    python3 -m pytest tests/test_api_shapes.py

MANIFEST.sample says how raw rows become the network's samples, and the answers hold to it
whatever a request looks like (bundle-contract §2.2): one answer per row sent, in order; a row
is answered from itself alone; a window row is answered when `length − 1` earlier rows of its
series are in the history and the request — else `answered: false` with a warning, never a
refusal and never a guess — so the history completes the first windows, and where a series is
cut between history and rows changes no answer; a graph query receives messages from the
reference graph only, so a node is answered the same alone or among others, the rows of one
graph share one answer, and a graph request takes no history. A window row whose series or
order key is null, or one the recipe cannot read, is refused 422 with the column named — as a
JSON body, as history and as a CSV upload — never answered 500. The tests of the shapes this
bundle does not have are not collected (tests/_oracle.py `applies`).
"""
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from _oracle import ORACLE, TAIL, applies, client, numbers_agree, same_value  # noqa: E402,F401

TOL = ORACLE.tolerance
SAMPLE = ORACLE.sample
NUMERIC_KINDS = ("probability", "raw", "value", "quantile", "horizon", "score", "row_score",
                 "cumulative", "recon", "embedding")
ALONE = 5                            # the rows answered one by one against the whole request


def _post(client, body, include=ORACLE.includes()) -> list:
    res = client.post("/v1/predict", params=[("include", v) for v in include], json=body)
    assert res.status_code == 200, res.text
    return res.json()["predictions"]


def _test_rows(client) -> list:
    return client.get("/v1/examples").json()["test"]["rows"]


def _same(a, b, warnings: bool = True) -> list:
    """Where two answers to the same row differ: a number beyond the round-trip tolerance, a
    decision, whether it was answered, (with `warnings`) its warnings."""
    diffs = [] if a["answered"] == b["answered"] else ["answered"]
    if warnings and a["warnings"] != b["warnings"]:
        diffs.append("warnings")
    for field in ORACLE.shown(ORACLE.includes()):
        kind = ORACLE.kind_of(field)
        if kind in TAIL:
            continue
        mine, theirs = ORACLE.value(a, field), ORACLE.value(b, field)
        agree = numbers_agree(mine, theirs, TOL) if kind in NUMERIC_KINDS else \
            same_value(mine, theirs)
        if not agree:
            diffs.append(f"{field}: {mine!r} != {theirs!r}")
    return diffs


def _ranks(rows) -> list:
    """Each row's place in its series (`series_by`), in `order_by` order: how many earlier rows
    of its series the request holds."""
    frame = pd.DataFrame(rows)
    series, order = SAMPLE.get("series_by"), SAMPLE.get("order_by")
    keys = frame[series].astype(str) if series else pd.Series("", index=frame.index)
    ordered = frame.sort_values(order, kind="stable") if order else frame
    return ordered.groupby(keys.loc[ordered.index]).cumcount().loc[frame.index].tolist()


def test_row_shape_ignores_nothing_it_was_given(client):
    rows = _test_rows(client)
    answers = _post(client, {"rows": rows})
    assert len(answers) == len(rows), (len(answers), len(rows))
    if ORACLE.window or ORACLE.graph_level:
        return                 # a window reads its series' order; a graph its whole rows
    backwards = _post(client, {"rows": rows[::-1]})
    for i, (a, b) in enumerate(zip(answers, backwards[::-1])):
        assert not _same(a, b), (i, _same(a, b))
    for i, row in enumerate(rows[:ALONE]):
        alone = _post(client, {"rows": [row]})[0]
        assert not _same(alone, answers[i]), (i, _same(alone, answers[i]))


@applies(ORACLE.window)
def test_rows_without_a_full_window_are_answered_false(client):
    anchor = ORACLE.one_request()["rows"]             # the request's row, without its history
    answer = _post(client, {"rows": anchor}, ())[0]
    assert answer["answered"] is False and answer["warnings"], answer
    keys = set(ORACLE.keys())
    assert set(answer) == keys, answer
    assert all(answer[k] is None for k in keys - set(TAIL)), answer


def _refused_by_name(res, column) -> None:
    assert res.status_code == 422, (column, res.status_code, res.text[:300])
    named = [e.get("column") for e in res.json().get("errors") or ()]
    assert column in named, (column, res.json())


@applies(ORACLE.window)
def test_a_null_or_unreadable_window_key_is_refused_by_name(client):
    request = ORACLE.one_request()
    for column in (SAMPLE.get("series_by"), SAMPLE.get("order_by")):
        if not column:
            continue
        body = {**request, "rows": [{**r, column: None} for r in request["rows"]]}
        _refused_by_name(client.post("/v1/predict", json=body), column)
        if request.get("history"):
            past = [{**r, column: None} for r in request["history"]]
            _refused_by_name(client.post("/v1/predict", json={**request, "history": past}), column)
        # a value the recipe may or may not read (an order parsed as a time, a new series):
        # answered, or refused by name — never a 500
        odd = {**request, "rows": [{**r, column: "vega-check-unreadable"} for r in request["rows"]]}
        res = client.post("/v1/predict", json=odd)
        if res.status_code != 200:
            _refused_by_name(res, column)
        table = pd.DataFrame(request["rows"]).astype(object)
        table.loc[table.index[0], column] = None
        res = client.post("/v1/predict/csv", files={
            "file": ("t.csv", table.to_csv(index=False).encode("utf-8"), "text/csv")})
        _refused_by_name(res, column)


@applies(ORACLE.window and bool(SAMPLE.get("order_by")))
def test_an_order_the_recipe_cannot_read_is_refused_by_name(client, monkeypatch):
    """An order value the recipe leaves missing (an unparseable time under a coercing parse)
    reaches the window plan as a hole: the assembler refuses it by name (samples.KeyRefused),
    and the service answers 422 — once a 500 (gate C, L10 ts='abc')."""
    predictor = client.app.state.service.predictor
    assembler, column = predictor._assembler, SAMPLE["order_by"]
    real = assembler._transform

    def holed(frame, history=None, with_unseen=False):
        shaped, unseen = real(frame, history=history, with_unseen=True)
        shaped = shaped.copy()
        if column in shaped.columns:
            shaped[column] = shaped[column].astype(object)
            shaped.loc[shaped.index[-1], column] = None
        else:                               # the key column is taken from the raw rows
            frame.loc[frame.index[-1], column] = None
        return (shaped, unseen) if with_unseen else shaped
    monkeypatch.setattr(assembler, "_transform", holed)
    res = client.post("/v1/predict", json=ORACLE.one_request())
    _refused_by_name(res, column)
    assert res.json()["status"] == 422


@applies(ORACLE.window)
def test_history_completes_the_first_windows(client):
    rows, need = _test_rows(client), int(SAMPLE["length"]) - 1
    ranks = _ranks(rows)
    whole = _post(client, {"rows": rows})
    assert [a["answered"] for a in whole] == [r >= need for r in ranks]
    assert all(a["warnings"] for a, r in zip(whole, ranks) if r < need)
    context = [row for row, r in zip(rows, ranks) if r < need]
    scored = [i for i, r in enumerate(ranks) if r >= need]
    completed = _post(client, {"rows": [rows[i] for i in scored], "history": context})
    assert all(a["answered"] for a in completed), completed[:2]
    for a, i in zip(completed, scored):
        assert not _same(a, whole[i]), (i, _same(a, whole[i]))
    request = ORACLE.one_request()                    # the shipped request: complete as sent
    assert request.get("history") and _post(client, request)[0]["answered"] is True


@applies(ORACLE.window)
def test_a_window_answer_does_not_depend_on_how_rows_are_split(client):
    rows = _test_rows(client)
    whole = _post(client, {"rows": rows})
    for cut in (len(rows) // 3, 2 * len(rows) // 3):
        part = _post(client, {"rows": rows[cut:], "history": rows[:cut]})
        assert len(part) == len(rows) - cut
        for k, (a, b) in enumerate(zip(part, whole[cut:])):
            assert not _same(a, b), (cut, cut + k, _same(a, b))


@applies(ORACLE.node_level)
def test_graph_answers_do_not_depend_on_the_batch(client):
    rows = _test_rows(client)[:ALONE + 1]
    together = _post(client, {"rows": rows})
    for i, row in enumerate(rows):
        alone = _post(client, {"rows": [row]})[0]
        assert not _same(alone, together[i]), (i, _same(alone, together[i]))


@applies(ORACLE.graph_level)
def test_rows_of_one_graph_share_one_answer(client):
    rows, group = _test_rows(client), SAMPLE["graph_id"]
    graphs = list(dict.fromkeys(str(r[group]) for r in rows))[:2]
    together = _post(client, {"rows": [r for r in rows if str(r[group]) in graphs]})
    for graph in graphs:
        alone = _post(client, {"rows": [r for r in rows if str(r[group]) == graph]})
        assert {a["graph_id"] for a in alone} == {graph}, alone[:2]
        assert all(not _same(alone[0], a, warnings=False) for a in alone), graph
        among = next(a for a in together if a["graph_id"] == graph)
        assert not _same(alone[0], among, warnings=False), (graph, _same(alone[0], among))
        backwards = _post(client, {"rows": [r for r in rows if str(r[group]) == graph][::-1]})
        assert not _same(alone[0], backwards[0], warnings=False), graph


@applies(ORACLE.shape == "graph")
def test_history_is_refused_for_a_graph(client):
    body = ORACLE.one_request()
    res = client.post("/v1/predict", json=dict(body, history=body["rows"][:1]))
    assert res.status_code == 422, res.text
    assert res.headers["content-type"].startswith("application/problem+json")
    assert any("history" in e["error"] for e in res.json()["errors"]), res.json()
