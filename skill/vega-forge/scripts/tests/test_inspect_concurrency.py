#!/usr/bin/env python3
"""
test_inspect_concurrency.py — explanations and predictions share one served network, and
neither sees the other (build plan I03; inspect-contract §1 R1–R6; R-B4; D3 §1.3's race and
gradient leak, §16.1).

    python3 test_inspect_concurrency.py       (or: python3 -m pytest test_inspect_concurrency.py -q)

On rendered `mlp-residual`, `lstm`, `lstm-attention` and `gat` packages, with the inspector's
permanent hooks attached once (`capture.HookSet`), two threads each run 50 explanations (every
sample of the batch in turn, at `full`, `exact` and `summary` detail) while a third runs 200
plain no-grad forwards — the `/v1/predict` path — over other batches. Every explanation must
equal its sequential reference byte for byte, every plain output the sequential one exactly;
afterwards no parameter holds `.grad`, the hooks are the ones attached once (the plain thread
counts them on every call: none is added per request), the network's state and mode are
unchanged and no capture is left active on any thread.

Exit 0 when every test passes.
"""
from __future__ import annotations

import sys
import threading

import pytest
import torch

import _inspect_fixtures as fx
from _inspect_fixtures import batch_for, case, explain

CASES = ("mlp-residual", "lstm", "lstm-attention", "gat")
EXPLAINERS, EXPLAINS, FORWARDS = 2, 50, 200
DETAILS = ("full", "exact", "summary")
SEEDS = (11, 12, 13, 14)


def _plan(c, which: int) -> list:
    """(sample, detail) of each explanation of one thread: the second walks the batch
    backwards, so the two threads explain different samples at the same moment."""
    n = c.samples()
    order = [(k if which == 0 else n - 1 - k) % n for k in range(EXPLAINS)]
    return [(i, DETAILS[(k + which) % len(DETAILS)]) for k, i in enumerate(order)]


def _references(c) -> dict:
    wanted = {step for which in range(EXPLAINERS) for step in _plan(c, which)}
    return {step: fx.canonical(explain(c, *step)[0]) for step in sorted(wanted)}


def _plain_batches(c) -> list:
    return [batch_for(c.spec, seed) for seed in SEEDS]


def _plain(model, batch):
    with torch.no_grad():
        return model(batch)


def _run_threads(c, batches) -> dict:
    """The explainers and the plain thread, started together; what each saw, and failures."""
    start = threading.Barrier(EXPLAINERS + 1)
    seen = {"explained": {w: [] for w in range(EXPLAINERS)}, "plain": [], "hooks": [],
            "errors": []}

    def explainer(which):
        try:
            start.wait()
            for step in _plan(c, which):
                seen["explained"][which].append((step, fx.canonical(explain(c, *step)[0])))
        except BaseException as exc:          # noqa: BLE001 — reported to the main thread
            seen["errors"].append(f"explainer {which}: {type(exc).__name__}: {exc}")

    def predictor():
        try:
            start.wait()
            for k in range(FORWARDS):
                seen["plain"].append((k % len(batches), _plain(c.model, batches[k % len(batches)])))
                seen["hooks"].append(fx.hooks_on(c.model))
                if c.capture.active() is not None:
                    seen["errors"].append("a capture is active on the predict thread")
        except BaseException as exc:          # noqa: BLE001
            seen["errors"].append(f"predictor: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=explainer, args=(w,)) for w in range(EXPLAINERS)]
    threads.append(threading.Thread(target=predictor))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert not any(t.is_alive() for t in threads), "a thread did not finish"
    return seen


@pytest.mark.parametrize("name", CASES)
def test_concurrent_explanations_and_predictions_equal_the_sequential_ones(name):
    c = case(name)
    before = {k: v.detach().clone() for k, v in c.model.state_dict().items()}
    hooks = fx.hooks_on(c.model)
    assert hooks == c.hooks.count > 0
    references = _references(c)
    batches = _plain_batches(c)
    plain = [_plain(c.model, b) for b in batches]

    seen = _run_threads(c, batches)

    assert not seen["errors"], seen["errors"]
    for which in range(EXPLAINERS):
        got = seen["explained"][which]
        assert [step for step, _ in got] == _plan(c, which)
        for step, body in got:
            assert body == references[step], f"{name}: explanation {step} differs under load"
    assert len(seen["plain"]) == FORWARDS
    for k, out in seen["plain"]:
        assert torch.equal(out, plain[k]), f"{name}: a plain forward differs under load"
    assert set(seen["hooks"]) == {hooks}
    assert all(p.grad is None for p in c.model.parameters())
    assert fx.hooks_on(c.model) == hooks and not c.model.training
    after = c.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert c.capture.active() is None


def test_explanations_are_not_disturbed_by_predictions_of_other_rows():
    """The race of D3 §1.3: a predict forward that runs while an explanation's pass is open
    fires the same hooks; the explanation keeps its own batch's values."""
    c = case("mlp-residual")
    reference = fx.canonical(explain(c, 2)[0])
    other = batch_for(c.spec, 99)
    cap = c.capture.Capture(c.graph)
    done = threading.Event()

    def predict_while_open():
        _plain(c.model, other)
        done.set()

    original = cap.put

    def put(key, args, output):             # a predict thread runs inside the open pass
        if not done.is_set():
            t = threading.Thread(target=predict_while_open)
            t.start()
            t.join()
        original(key, args, output)

    cap.put = put
    head = cap.run(c.model, c.batch, 2)
    assert done.is_set()
    assert fx.canonical(fx.section(c, cap, head, 2)) == reference
    assert all(p.grad is None for p in c.model.parameters())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
