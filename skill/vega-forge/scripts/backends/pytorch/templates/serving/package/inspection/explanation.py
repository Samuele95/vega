"""
inspection.explanation — one `vega/explanation@2` body: what one sample of a prepared request
did as it passed through the served network (inspect-contract §5.4; D3 §6.2, §8.2; build plan
R-D15, R-F5, R-F6).

`Explainer(model, graph, targets, attributor, tolerance)` explains sample `at.k` of the batch
`Predictor.prepare()` returned — the very batch `predict` runs (R7) — beside the Predictor's
answer frame for the same rows (`Served`):

1. **run** — a `capture.Capture` pass over the batch, the leaves requiring a gradient, the
   network's hooks keeping every node's call (the permanent hooks `HookSet` attached once);
2. **target** — `targets.TaskTargets.choose` (the request's selector, else the task's default
   picked from the head) and its scalar `s` for sample `k`, differentiated once
   (`Capture.differentiate`: no parameter's `.grad` is ever written);
3. **nodes** — every node of the stage graph placed in graph order under the request's
   `reduce.Budget` (`parts.NodeParts`), the recompute report and the dense chain;
4. **attribution** — gradient × input from the differentiated pass, or integrated gradients
   run afterwards, outside the capture (`attribution.Attributor`); on a graph the interpolated
   node count is bounded first (`ExplainQuery.interpolants`, a 422);
5. **fidelity** — `head_max_abs_diff`: the captured head against a plain no-grad forward of the
   same batch; `served_max_abs_diff`: the target's value against the served field that answers
   it (`TaskTargets.served`, by `MANIFEST.output.field_kinds`); either above the tolerance
   (`MANIFEST.roundtrip.tolerance.value`, else 1e-5) is `FidelityError` — the route's 500, "the
   explanation does not reproduce the served answer" — never a body with `ok: false`.

A sample the network answers with a number that is not finite (a window whose lag features
lack history) has no explanation to publish: `query.QueryError` on `at` (a 422) naming the
row's warnings; `finite(prepared)` names the samples that have one.

The body's `sample` names the explained sample (`index` k, `row` the request row its answer
belongs to, the window's `span` in request coordinates, the graph's level, size and target
node), `prediction` is the served frame's row for that request row (plain values, its warnings
the list `Predictor.predictions` gives, which `warnings` repeats; the route replaces it with the
HTTP answer), `inputs` the shaped values the network read for the sample over `inputs.order` (a
vector for a row or a node, steps × channels for a window, nodes × channels for a graph — a heat
map beyond `MATRIX_MAX` cells) and the raw request row echoed. `run` also says whether the node
checks held (`recompute.worst ≤ 1e-4·max(1, |out|)` of the worst node's output): a warning when
not, never a refusal.

`Served.of(answers, rows, history)` reads the answer: the frame, and each row's warnings apart
(a warning may hold the frame's `; `); `answered_rows(prepared)` pairs each sample with the
first request row it answers (a window's anchor, a query node, a graph's first row).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .attribution import MATRIX_MAX
from .capture import Capture
from .codec import num, nums
from .parts import NodeParts
from .query import QueryError
from .reduce import Budget, Reducer

__all__ = ["DEFAULT_TOLERANCE", "RECOMPUTE_BOUND", "SCHEMA", "WARNING_SEP", "Explained",
           "Explainer", "FidelityError", "Served", "answered_rows", "on_device", "plain"]

SCHEMA = "vega/explanation@2"
DEFAULT_TOLERANCE = 1e-5          # the round-trip tolerance a manifest without one falls back to
RECOMPUTE_BOUND = 1e-4            # a node check holds at ≤ 1e-4·max(1, |out|)
WARNING_SEP = "; "                # the Predictor joins a row's warnings with this (its frame)
IG = "integrated_gradients"
NOT_FINITE = ("request row {row} is answered with numbers that are not finite ({why}), so no "
              "explanation of it can be published; send the rows its sample reads (more "
              "history), or explain another row")
UNFAITHFUL = ("the explanation does not reproduce the served answer: the captured head differs "
              "from the plain forward by {head}, the target {kind} from the served field by "
              "{served}, above the tolerance {tolerance}")


class FidelityError(RuntimeError):
    """The explanation's numbers are not the served ones (the route answers 500)."""


def plain(value):
    """A frame or row cell as a JSON value: numpy scalars as Python ones; NaN, NA and None as
    None; anything else not JSON-native as its text."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value if isinstance(value, (str, bool, int, float)) else str(value)


def on_device(batch, model) -> dict:
    """The batch's tensors where the model's live (the Predictor's own move): its Model
    port's `anatomy().device`."""
    device = model.anatomy().device
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def answered_rows(prepared) -> list:
    """[(sample k, the first request row sample k answers)] in sample order."""
    first = {}
    for i, k in enumerate(np.asarray(prepared.sample_of_row, dtype=np.int64).tolist()):
        if k >= 0 and k not in first:
            first[k] = i
    return [(k, first[k]) for k in range(int(prepared.n_samples)) if k in first]


@dataclass(frozen=True)
class Explained:
    """The explained sample `k` of a prepared batch and the request row `row` it answers."""

    k: int
    row: int


@dataclass(frozen=True, eq=False)
class Served:
    """The Predictor's answer for a request's rows, read by row: its frame (indexed like them)
    and, when the frame's joined text cannot say them apart, each row's warnings as
    `Predictor.predictions` gives them (`notes`)."""

    frame: pd.DataFrame
    notes: tuple | None = None

    @staticmethod
    def of(answers, rows, history=None) -> "Served":
        """The answer of `answers` (the Predictor, or a harness standing in for one) for `rows`.
        A warning may itself hold `WARNING_SEP` (the schema guard's does), so a frame whose text
        holds it is read with the answers' own `predictions` — one `Prediction` per row, its
        `warnings` a tuple — and only a harness without one has its text split."""
        frame = answers.predict(rows, history)
        texts = frame["warnings"] if "warnings" in frame.columns else ()
        ambiguous = any(isinstance(t, str) and WARNING_SEP in t for t in texts)
        apart = getattr(answers, "predictions", None)
        if not ambiguous or not callable(apart):
            return Served(frame)
        notes = tuple(tuple(str(w) for w in p.warnings) for p in apart(rows, history))
        if len(notes) != len(frame):
            raise ValueError(f"the answers' predictions give {len(notes)} row(s) for the "
                             f"{len(frame)} its frame answers: one per request row")
        return Served(frame, notes)

    def row(self, i: int) -> dict:
        """Request row `i`'s answer as {field: value} (missing values as None)."""
        rec = self.frame.iloc[int(i)]
        return {str(field): plain(rec[field]) for field in self.frame.columns}

    def prediction(self, i: int) -> dict:
        """The row's answer with its warnings as a list: the Predictor's own when `notes`
        holds them, else the frame's one text split at `WARNING_SEP` (unambiguous there)."""
        answer = self.row(i)
        if self.notes is not None:
            answer["warnings"] = list(self.notes[int(i)])
            return answer
        text = answer.get("warnings")
        answer["warnings"] = [w for w in str(text).split(WARNING_SEP) if w] if text else []
        return answer


def _ordered(layout, batch):
    """The network's inputs over `inputs.order` for every row of the batch, float64: the
    numeric channels, the nominal ids, the autoregressive channel."""
    x, ids = batch.get("x_num"), batch.get("x_cat")
    lead = x if x is not None else ids
    x = x.detach().double() if x is not None else torch.zeros(*lead.shape[:-1], 0,
                                                              dtype=torch.float64)
    ids = ids.detach().double() if ids is not None else torch.zeros(*x.shape[:-1], 0,
                                                                    dtype=torch.float64)
    return layout.ordered(x.cpu(), ids.cpu())


class Explainer:
    """The explanations of one served network (see the module docstring)."""

    def __init__(self, model, graph, targets, attributor, tolerance: float):
        self.model, self.graph, self.targets = model, graph, targets
        self.attributor, self.tolerance = attributor, float(tolerance)

    def explain(self, prepared, served: Served, at: Explained, query) -> dict:
        """The `explanation` body of sample `at.k` for `query` (its target, method, steps,
        detail; its rows for the raw echo)."""
        return self.run(prepared, served, at, query)[0]

    def finite(self, prepared) -> frozenset:
        """The samples whose answer is finite: the ones an explanation can be published of
        (a window whose lags lack history is answered with NaN, and says so)."""
        if not int(prepared.n_samples):
            return frozenset()
        rows = self._plain(on_device(prepared.batch, self.model))
        rows = rows.reshape(rows.shape[0], -1) if rows.dim() else rows.reshape(1, 1)
        return frozenset(int(k) for k in torch.nonzero(torch.isfinite(rows).all(dim=1))
                         .flatten().tolist())

    def run(self, prepared, served: Served, at: Explained, query) -> tuple:
        """(the body, whether the node checks held)."""
        batch = on_device(prepared.batch, self.model)
        plain_head = self._plain(batch)
        if not bool(torch.isfinite(plain_head[at.k]).all()):
            why = "; ".join(served.prediction(at.row)["warnings"]) or "its inputs are not finite"
            raise QueryError("at", NOT_FINITE.format(row=at.row, why=why))
        if query.method == IG:
            query.interpolants(self._interpolated(batch, at.k), self.graph.codec.limits)
        cap = Capture(self.graph)
        head = cap.run(self.model, batch, at.k)
        target = self.targets.choose(head, cap.batch, query.target, at.k)
        cap.differentiate(self.targets.scalar(target, head, cap.batch)[at.k])
        parts, budget = NodeParts(cap), Budget(query.detail, self.graph.codec.limits)
        nodes = parts.place(budget)
        value = self.targets.value(target, head, cap.batch, at.k)
        prediction = served.prediction(at.row)
        body = {"schema": SCHEMA, "detail": query.detail, "degraded": budget.degraded,
                "sample": self._sample(prepared, cap.focus, at, batch),
                "prediction": prediction, "target": target.to_dict(value),
                "outputs": self.targets.readout(head, cap.batch, at.k),
                "inputs": self._inputs(batch, cap.focus, query.rows, at.row),
                "attribution": self._attribution(cap, batch, target, query),
                "nodes": nodes, "chain": parts.chain(self.graph.chain()),
                "fidelity": self._fidelity(head, plain_head, at.k,
                                           (target, value, served.row(at.row))),
                "recompute": parts.report(), "budget": budget.report(),
                "warnings": list(prediction["warnings"])}
        return body, self._checks_hold(cap, body["recompute"])

    # -- the parts --------------------------------------------------------------------------
    def _interpolated(self, batch, k: int) -> int:
        """The nodes one interpolant of sample k holds (1 off a graph)."""
        inputs = self.graph.inputs
        if inputs.shape != "graph":
            return 1
        if inputs.graph_level:
            return int((batch["graph_index"] == int(k)).sum())
        return int(self.attributor.layout.rows(batch))

    def _attribution(self, cap, batch, target, query) -> dict:
        if query.method == IG:
            return self.attributor.integrated(self.model, batch, target, cap.focus.index,
                                              query.steps)
        return self.attributor.gradient_x_input(cap, target)

    def _plain(self, batch):
        """The network's own head on the batch (the Model port's `head`): no capture, no
        gradient."""
        return self.model.head(batch)

    def _fidelity(self, head, plain_head, k: int, served) -> dict:
        """The captured head against the plain forward; the target against the served field."""
        target, value, answer = served
        diff = float((head[k].detach().double() - plain_head[k].double()).abs().max())
        answered = self.targets.served(target, answer)
        away = None if answered is None else abs(float(value) - answered)
        if not diff <= self.tolerance or (away is not None and not away <= self.tolerance):
            raise FidelityError(UNFAITHFUL.format(head=f"{diff:.3g}", kind=target.kind,
                                                  served="n/a" if away is None else f"{away:.3g}",
                                                  tolerance=f"{self.tolerance:.3g}"))
        return {"head_max_abs_diff": num(diff),
                "served_max_abs_diff": None if away is None else num(away),
                "tolerance": num(self.tolerance), "ok": True}

    def _sample(self, prepared, focus, at: Explained, batch) -> dict:
        spans = prepared.index.spans
        graph = None
        if focus.shape == "graph":
            nodes, edges = self._extent(batch, focus)
            graph = {"level": focus.level, "nodes": nodes, "edges": edges,
                     "target_node": focus.target}
        return {"shape": focus.shape, "index": int(at.k), "row": int(at.row),
                "span": None if spans is None else [int(v) for v in spans[at.k]],
                "graph": graph}

    @staticmethod
    def _extent(batch, focus) -> tuple:
        """(nodes, edges) of the graph the sample reads: the batch's graph at node level, the
        explained graph's own at graph level."""
        edge_index = batch["edge_index"]
        rows = int(batch["x_num"].shape[0] if batch.get("x_num") is not None
                   else batch["x_cat"].shape[0])
        if focus.level == "node":
            return rows, int(edge_index.shape[1])
        members = torch.zeros(rows, dtype=torch.bool, device=edge_index.device)
        members[list(focus.nodes)] = True
        return len(focus.nodes), int(members[edge_index[0]].sum())

    def _inputs(self, batch, focus, rows, row: int) -> dict:
        layout = self.attributor.layout
        values = focus.take(_ordered(layout, batch), "node" if focus.shape == "graph"
                            else "sample")
        if values.dim() == 2 and values.numel() > MATRIX_MAX:
            shaped = Reducer(self.graph.codec.limits).heat(values, "mean")
        else:
            shaped = nums(values)
        raw = None
        if rows is not None and 0 <= int(row) < len(rows):
            rec = rows.iloc[int(row)]
            raw = {str(column): plain(rec[column]) for column in rows.columns}
        return {"order": list(layout.order), "shaped": shaped, "raw": raw}

    def _checks_hold(self, cap, report) -> bool:
        """Whether the worst node check is within 1e-4·max(1, |out|) of that node's output."""
        if report["worst"] is None:
            return True
        node = cap.graph.node(report["worst_node"])
        out = cap.value(node.id)
        taken = None if out is None else cap.take(out, node)
        scale = 1.0 if taken is None or not taken.numel() else \
            float(taken.detach().abs().max())
        return float(report["worst"]) <= RECOMPUTE_BOUND * max(1.0, scale)
