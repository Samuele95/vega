"""
{{PACKAGE}}.{{MODULE_INGESTION}}.windows — the window half of the {{PORT_DATASOURCE}} adapter
(support; emitted when `data_contract.sample.shape` is `window`).

Emitted by /vega-forge from `backends/pytorch/templates/windows.py`. Fix defects in the template
and re-emit; never in this copy (code-contract §4).

Which rows form a window and which rows it predicts is decided once, in `domain.py`
(`WindowSpec.layout` and `WindowSpec.plan`, rules 1–6 of the backend design §9.3), the plan the
serving assembler cuts its windows by as well. This file only tensorises it:

  * a part's rows are grouped by `series_by` and ordered by `order_by`; with `context:
    previous-part` each series is prefixed by its last L−1 rows of the part(s) before (train for
    valid; train then valid for the sealed test) — inputs only, never anchors;
  * the part is held FLAT (`WindowedPart`: the context and part rows once each, O(rows·F)); a
    batch cuts its `(B, L, F')` windows on the fly by a static-shaped gather;
  * the training part keeps the stride-thinned anchors; every other part is scored densely, one
    window per scorable row, and a row without one is an `UnscoredRow` with its reason;
  * a sequence task predicts the anchor's own label, a forecast the next H rows of the same part
    and series; with `autoregressive` the label is appended as the last numeric channel, and a
    window reads it only at rows up to its anchor;
  * a window's row id is its anchor's raw-file position.

No window crosses a series, no target crosses the part's end, the nominal ids arrive as
`(B, L, C)`: the three v2.2 defects this layout exists to remove (E2–E5).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch

from ..domain import (NetSpec, Part, SeriesRows, Task, TaskKind, WindowContext, WindowedPart,
                      WindowSpec)
from .frames import PartFrames, UnsealedFrames, feature_tensors, label_tensor

# the parts a part's windows borrow context from, in the order they are concatenated
_BEFORE = {Part.TRAIN: (), Part.VALID: (Part.TRAIN,), Part.TEST: (Part.TRAIN, Part.VALID)}


class _Rows(NamedTuple):
    """Rows of one or more frames as flat arrays, with the keys a window plan groups them by."""

    x_num: Any
    x_cat: Any
    y: Any
    keys: SeriesRows

    @staticmethod
    def concat(pieces) -> "_Rows":
        def cat(values):
            return None if values[0] is None else np.concatenate(values)
        keys = [p.keys for p in pieces]
        return _Rows(cat([p.x_num for p in pieces]), cat([p.x_cat for p in pieces]),
                     cat([p.y for p in pieces]),
                     SeriesRows(np.concatenate([k.raw_pos for k in keys]),
                                cat([k.series for k in keys]), cat([k.order for k in keys])))


def _key(frame, column, part: Part):
    if not column:
        return None
    if column not in frame.columns:
        raise ValueError(f"the {part.value} part has no {column!r} column, which the window "
                         "sample names as a key (series_by / order_by)")
    return frame[column].to_numpy()


def _torch(values, dtype=None):
    if values is None:
        return None
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    return tensor if dtype is None else tensor.to(dtype)


class WindowParts:
    """A part of a window sample as a `WindowedPart`, and the train-side seasonal scale."""

    def __init__(self, frames: PartFrames | UnsealedFrames):
        self._frames = frames

    def part(self, spec: NetSpec, part) -> WindowedPart:
        """The part's windows: stride-thinned on train, dense everywhere else."""
        spec, part = NetSpec.require(spec), Part(part)
        wspec, task = WindowSpec.from_spec(spec), Task.from_spec(spec)
        now = self._rows(spec, (part,), wspec, task)
        before = (self._rows(spec, _BEFORE[part], wspec, task)
                  if _BEFORE[part] and wspec.context is WindowContext.PREVIOUS_PART else None)
        layout = wspec.layout(now.keys, before.keys if before else None)

        def take(field):
            values = getattr(now, field)
            return None if values is None else layout.take(
                getattr(before, field) if before else None, values)

        plan = wspec.plan(layout.blocks, dense=part is not Part.TRAIN)
        if not len(plan):
            raise ValueError(
                f"the {part.value} part yields no window: {len(layout.blocks)} series, length "
                f"{wspec.length}, horizon {wspec.horizon}, {len(plan.unscored)} row(s) unscored; "
                "each series needs length-1 earlier rows (context included) and horizon later ones")
        return WindowedPart(x_num=_torch(take("x_num"), torch.float32),
                            x_cat=_torch(take("x_cat"), torch.long), y=_torch(take("y")),
                            anchors=torch.from_numpy(plan.anchors),
                            targets=torch.from_numpy(np.ascontiguousarray(plan.targets)),
                            row_ids=plan.row_ids, length=wspec.length, unscored=plan.unscored,
                            columns=self._columns(spec, wspec, task))

    def target_scale(self, spec: NetSpec, season: int = 1) -> float | None:
        """mase's scale: the mean absolute `season`-lag difference of the TRAIN target within
        each series, pooled (D1 §11.2). None off forecast, and when no series is long enough."""
        spec = NetSpec.require(spec)
        task = Task.from_spec(spec)
        if task.kind is not TaskKind.FORECAST:
            return None
        if int(season) < 1:
            raise ValueError(f"the seasonal lag is at least 1, got {season!r}")
        wspec, lag = WindowSpec.from_spec(spec), int(season)
        keys = [c for c in (wspec.series_by, wspec.order_by) if c]
        opened = self._frames.open(spec, Part.TRAIN, columns=[task.label, *keys])
        frame = opened.frame
        layout = wspec.layout(SeriesRows(opened.raw_pos, _key(frame, wspec.series_by, Part.TRAIN),
                                         _key(frame, wspec.order_by, Part.TRAIN)))
        y = layout.take(None, frame[task.label].to_numpy(dtype=np.float64, na_value=np.nan))
        diffs = [np.abs(y[b.start + lag:b.stop] - y[b.start:b.stop - lag])
                 for b in layout.blocks if b.n_rows > lag]
        pooled = np.concatenate(diffs) if diffs else np.zeros(0)
        pooled = pooled[np.isfinite(pooled)]
        return float(pooled.mean()) if len(pooled) else None

    def _rows(self, spec: NetSpec, parts, wspec: WindowSpec, task: Task) -> _Rows:
        return _Rows.concat([self._arrays(spec, p, wspec, task) for p in parts])

    def _arrays(self, spec: NetSpec, part: Part, wspec: WindowSpec, task: Task) -> _Rows:
        opened = self._frames.open(spec, part)
        frame = opened.frame
        x_num, x_cat = feature_tensors(frame, spec.feature_contract, part.value)
        y = label_tensor(frame, task)
        if y is None:
            raise ValueError(f"the {part.value} part has no label column {task.label!r}; every "
                             "window task predicts a label")
        x_num, y = x_num.numpy(), y.numpy()
        if wspec.autoregressive:
            # the label as the last numeric channel; a window reads it at rows <= its anchor,
            # and a forecast's targets are rows > the anchor (rule 6)
            x_num = np.concatenate([x_num, y.reshape(len(y), 1).astype(np.float32)], axis=1)
        keys = SeriesRows(opened.raw_pos, _key(frame, wspec.series_by, part),
                          _key(frame, wspec.order_by, part))
        return _Rows(x_num, None if x_cat is None else x_cat.numpy(), y, keys)

    @staticmethod
    def _columns(spec: NetSpec, wspec: WindowSpec, task: Task) -> tuple:
        contract = spec.feature_contract
        names = contract.columns
        n_cat = len([c for c in contract.nominal_columns if c])
        numeric, nominal = names[:len(names) - n_cat], names[len(names) - n_cat:]
        return numeric + ((task.label,) if wspec.autoregressive else ()) + nominal
