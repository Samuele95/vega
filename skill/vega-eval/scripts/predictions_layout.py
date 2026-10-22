"""
predictions_layout — `vega/predictions-layout@1` (BUILD-PLAN R-C2): the stored predictions of the
one look at the test part, the file every later number is computed from.

`evaluate.py` writes it once, under the seal; the evaluation notebook recomputes every metric
from it, `/vega-pack` rebuilds its round trip and its examples from it (S10), the serving readers
read it (`from_stored`, S03). One layout for every shape, task and reading:

  row_id            the anchor's raw position (a window), the node's (a node-level graph), the
                    row's (a row sample) — `ScoredPart.row_ids`, never re-mapped — or the
                    `graph_id` value at graph level, where a `graph_id` column repeats it
  y | y_0 …         the truth as the Evaluator read it, keeping its dimension: one column for a
                    (N,) truth, `y_<k>` for (N, K) — the labels of a multilabel row, the horizon
                    of a forecast, the numeric input of a reconstruction. No column for an
                    unlabelled part.
  score | score_0 … the network's RAW HEAD, every `head_out` column, for every reading (a
                    binary logit, the class logits, a quantile head's every level, the
                    reconstruction's every output, the svdd embedding): what `TaskReadout` reads
                    — the score contract's score is derived from it, never stored instead of it
  row_score         `TaskReadout`'s per-row score for the reconstruction and anomaly readings
                    (numeric MSE plus the mean nominal cross-entropy; ‖f(x) − c‖² for svdd)
  prediction        the decision of a threshold reading, 0/1: σ(score) ≥ threshold for a binary
                    head (the probability of task.classes[1]), row_score ≥ threshold for an
                    anomaly (the score's own scale); absent when no threshold was chosen
  score_<baseline>  each fitted baseline's score per the score contract (what its metrics
  | score_<b>_<k>   read), one column, or one per column of a matrix score

Parquet key-value metadata: `vega.row_id_kind` (raw_position | graph_id), `vega.truth_shape` and
`vega.score_shape` (the per-sample shapes of the stored truth and head, JSON: `[]` one column,
`[K]` K columns, `null` no truth), `vega.task_effective` (the kind the scores are read as — a
`sequence-*` task's base) and `vega.predictions_layout`. `metrics.json.predictions_layout` names
the layout too, so a reader refuses a file it would misread before it opens it.

`unscored.json` beside it: the samples that got no prediction — windows without enough context,
forecast anchors whose horizon runs past the part — counted by reason and listed, never given a
fabricated one (backend design §3.5).

Pure: arrays in, a frame and a file out. No emitted package is imported here.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = ["LAYOUT", "METADATA_KEYS", "PredictionsLayout", "columns_of", "read",
           "unscored_record"]

LAYOUT = "vega/predictions-layout@1"
#: the key-value metadata every stored predictions file carries (R-C2)
METADATA_KEYS = ("vega.row_id_kind", "vega.truth_shape", "vega.score_shape",
                 "vega.task_effective", "vega.predictions_layout")
ROW_ID_KINDS = ("raw_position", "graph_id")
#: the reasons a sample goes unscored (the domain's `UnscoredReason`), always counted
UNSCORED_REASONS = ("insufficient_context", "horizon_beyond_part")


def _array(values):
    if values is None:
        return None
    detach = getattr(values, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(values)


def columns_of(prefix: str, values, *, squeeze: bool = False) -> dict:
    """`{prefix: v}` for one column, `{prefix_k: v[:, k]}` for a matrix. `squeeze` stores a
    (N, 1) matrix as its one column (the head: `score` for a single output)."""
    values = _array(values)
    if values is None:
        return {}
    if values.ndim == 1 or (squeeze and values.ndim == 2 and values.shape[1] == 1):
        return {prefix: values.reshape(len(values))}
    if values.ndim != 2:
        raise ValueError(f"{prefix}: a stored column block is (N,) or (N, K), got shape "
                         f"{tuple(values.shape)}")
    return {f"{prefix}_{k}": values[:, k] for k in range(values.shape[1])}


def _shape(values, *, squeeze: bool = False) -> str:
    """The per-sample shape as stored, as JSON: `[]`, `[K]`, or `null` for no values."""
    values = _array(values)
    if values is None:
        return "null"
    if values.ndim == 1 or (squeeze and values.ndim == 2 and values.shape[1] == 1):
        return "[]"
    return json.dumps([int(d) for d in values.shape[1:]])


def _kind(value) -> str:
    return str(getattr(value, "value", value))


@dataclass(frozen=True)
class PredictionsLayout:
    """One look's stored predictions, before they are a file."""

    row_ids: Any
    row_id_kind: str
    truth: Any
    head: Any
    task_effective: str
    row_score: Any = None
    prediction: Any = None
    graph_ids: Any = None
    baselines: tuple = ()          # ((name, score), …) in the order they were scored

    def __post_init__(self):
        kind = _kind(self.row_id_kind)
        if kind not in ROW_ID_KINDS:
            raise ValueError(f"row_id_kind {kind!r} is not one of {ROW_ID_KINDS}")
        object.__setattr__(self, "row_id_kind", kind)
        object.__setattr__(self, "task_effective", _kind(self.task_effective))
        n = len(self.row_ids)
        for name in ("truth", "head", "row_score", "prediction", "graph_ids"):
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(f"{len(values)} {name} value(s) for {n} row id(s): the stored "
                                 "columns of one look are aligned sample for sample")
        if self.head is None:
            raise ValueError("the layout stores the network's raw head; none was given")
        for name, score in self.baselines:
            if not str(name)[:1].isalpha():
                raise ValueError(f"baseline {name!r}: a stored score_<name> column needs a name "
                                 "that starts with a letter (score_<k> is the network's head)")
            if len(score) != n:
                raise ValueError(f"baseline {name!r}: {len(score)} score(s) for {n} row id(s)")
        if self.row_id_kind == "graph_id" and self.graph_ids is None:
            object.__setattr__(self, "graph_ids", self.row_ids)

    @staticmethod
    def of(scored, head, task_effective, prediction=None, baselines=()) -> "PredictionsLayout":
        """From the network's `ScoredPart` and its raw head (`Evaluator.heads`), the threshold
        decision (or None) and `((name, baseline ScoredPart.score), …)`."""
        return PredictionsLayout(
            row_ids=np.asarray(scored.row_ids), row_id_kind=scored.row_id_kind,
            truth=_array(scored.truth), head=_array(head), task_effective=task_effective,
            row_score=_array(getattr(scored, "row_score", None)),
            prediction=_array(prediction), graph_ids=_array(getattr(scored, "graph_ids", None)),
            baselines=tuple((str(n), _array(s)) for n, s in baselines))

    def __len__(self):
        return len(self.row_ids)

    def columns(self) -> dict:
        """The stored columns, in the layout's order."""
        out = {"row_id": np.asarray(self.row_ids)}
        if self.graph_ids is not None:
            out["graph_id"] = np.asarray(self.graph_ids)
        out.update(columns_of("y", self.truth))
        out.update(columns_of("score", self.head, squeeze=True))
        if self.row_score is not None:
            out["row_score"] = np.asarray(self.row_score, dtype=np.float64).reshape(len(self))
        if self.prediction is not None:
            out["prediction"] = np.asarray(self.prediction).astype(np.int64).reshape(len(self))
        for name, score in self.baselines:
            block = columns_of(f"score_{name}", score)
            clash = sorted(set(block) & set(out))
            if clash:
                raise ValueError(f"baseline {name!r} would overwrite the stored column(s) "
                                 f"{clash}")
            out.update(block)
        return out

    def frame(self):
        import pandas as pd
        return pd.DataFrame(self.columns())

    def metadata(self) -> dict:
        return {"vega.row_id_kind": self.row_id_kind,
                "vega.truth_shape": _shape(self.truth),
                "vega.score_shape": _shape(self.head, squeeze=True),
                "vega.task_effective": self.task_effective,
                "vega.predictions_layout": LAYOUT}

    def write(self, path) -> Path:
        """`predictions.test.parquet` with the layout's key-value metadata."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(self.frame(), preserve_index=False)
        meta = dict(table.schema.metadata or {})
        meta.update({k.encode(): v.encode() for k, v in self.metadata().items()})
        pq.write_table(table.replace_schema_metadata(meta), path)
        return path


def read(path) -> tuple:
    """(frame, the `vega.*` metadata) of a stored predictions file."""
    import pandas as pd
    import pyarrow.parquet as pq
    from pyarrow import fs
    meta = pq.read_schema(path).metadata or {}
    # Arrow's own filesystem: a Python file object (pandas' default) lets an Arrow thread
    # touch Python while the interpreter finalizes, and the process aborts at exit (A40)
    return (pd.read_parquet(path, filesystem=fs.LocalFileSystem()),
            {k.decode(): v.decode() for k, v in meta.items() if k.startswith(b"vega.")})


def unscored_record(unscored) -> dict:
    """`unscored.json`: `{count, reasons: {reason: n}, rows: [{row_id, reason}]}` — every
    reason named, zero when none applied, so a reader never guesses what an absent key means."""
    rows = []
    for item in unscored or ():
        if isinstance(item, Mapping):
            raw, reason = item.get("raw_pos"), item.get("reason")
        else:
            raw, reason = getattr(item, "raw_pos"), getattr(item, "reason")
        raw = raw.item() if hasattr(raw, "item") else raw
        rows.append({"row_id": raw, "reason": _kind(reason)})
    counts = Counter(r["reason"] for r in rows)
    reasons = {r: int(counts.get(r, 0)) for r in UNSCORED_REASONS}
    reasons.update({r: int(n) for r, n in counts.items() if r not in reasons})
    return {"schema": "vega/unscored@1", "count": len(rows), "reasons": reasons, "rows": rows}
