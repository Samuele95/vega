"""
stored_predictions.py — the valid part scored from a run's best.pt, stored the way /vega-eval
stores the test part: `predictions.valid.parquet` in `vega/predictions-layout@1` (R-C2).

    write_valid_predictions(run_dir, adapters, spec_obj, checkpoint, campaign_root)
        -> {path, rows, unscored}

The scores are the Evaluator's `ScoredPart` (one scoring path with /vega-eval, D1 §11.1) and
its raw heads, or `Trainer.predict` when the blueprint drew no Evaluator port. The columns:
`row_id` (the anchor's or node's raw position, or the graph id with a `graph_id` column at
graph level), `y` or `y_0…` (the truth in its own dimension), `score` or `score_0…` (the raw
head, float32), `row_score` (float32) for the reconstruction and anomaly readings. The parquet
metadata: `vega.layout`, `vega.part`, `vega.row_id_kind`, `vega.truth_shape`,
`vega.score_shape`, `vega.task_effective`, `vega.score_source`. The rows a forecast cannot
score are counted by reason, never answered. `trainer.py` writes it after every run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: the stored layout of predictions.valid.parquet — the one /vega-eval writes for the test part
PREDICTIONS_LAYOUT = "vega/predictions-layout@1"


def plain(value):
    """A value of the package's domain (a str enum such as StopReason, a numpy or torch
    scalar) as the plain JSON value the records hold."""
    if value is None or isinstance(value, bool):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    if isinstance(value, str):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return value


@dataclass(frozen=True, eq=False)
class StoredScores:
    """The valid part as scored for `vega/predictions-layout@1`: the raw head (every
    `head_out` column; `score_source` says "readout" in the one case it could not be had), the
    truth in its own dimension, the row ids and their kind, the per-row score of the
    reconstruction and anomaly readings, the graph ids at graph level and the unscored rows."""

    head: Any
    truth: Any
    row_ids: Any
    row_id_kind: str
    row_score: Any = None
    graph_ids: Any = None
    unscored: tuple = ()
    score_source: str = "head"

    @staticmethod
    def of_evaluator(evaluator, spec, part, model) -> "StoredScores":
        """The one scoring path (D1 §11.1): the Evaluator's `ScoredPart`, and its `heads` for
        the raw head R-C2 stores (the score contract reads only part of it for a quantile
        head, a reconstruction or a distance to the centre)."""
        scored = evaluator.scored(spec, part, model)
        return StoredScores(head=evaluator.heads(spec, part, model), truth=scored.truth,
                            row_ids=scored.row_ids, row_id_kind=plain(scored.row_id_kind),
                            row_score=scored.row_score, graph_ids=scored.graph_ids,
                            unscored=tuple(scored.unscored or ()))

    @staticmethod
    def of_trainer(trainer, spec, part, model, head_out) -> "StoredScores":
        """No Evaluator port was drawn: `Trainer.predict` over the held part gives the scored
        part — the score contract's reading of the head, the truth and the row ids, made by the
        same scoring pass as the Evaluator's. Where that reading keeps
        every head column it is the raw head; where it does not (a quantile head's median, a
        reconstruction's numeric part, an anomaly's per-row score) the columns are the
        readout's and the layout says so (`vega.score_source: readout`) — an anomaly's score
        is then also its per-row score."""
        import numpy as np
        scored = trainer.predict(model, spec, part)
        score = np.asarray(scored.score)
        width = 1 if score.ndim == 1 else int(score.shape[1])
        kind = plain(scored.row_id_kind)
        per_row = score if score.ndim == 1 and head_out not in (None, 1) else None
        return StoredScores(
            head=score, truth=scored.truth, row_ids=np.asarray(scored.row_ids), row_id_kind=kind,
            row_score=per_row, graph_ids=scored.graph_ids, unscored=tuple(scored.unscored),
            score_source="head" if head_out in (None, width) else "readout")

    def frame(self):
        """The rows of the layout: `row_id` (and `graph_id` at graph level), `y` or `y_0…`,
        `score` or `score_0…` (float32), `row_score` where the reading has one."""
        import numpy as np
        import pandas as pd
        n = len(self.row_ids)
        cols = {"row_id": np.asarray(self.row_ids)}
        if self.graph_ids is not None:
            cols["graph_id"] = np.asarray(self.graph_ids)
        truth = None if self.truth is None else np.asarray(self.truth)
        if truth is not None:
            cols.update(_columns("y", truth.reshape(n, -1) if truth.ndim > 2 else truth))
        head = np.asarray(self.head)
        head = head.reshape(n, -1) if head.ndim != 1 else head.reshape(n, 1)
        cols.update(_columns("score", head.astype(np.float32)
                             if head.shape[1] > 1 else head[:, 0].astype(np.float32)))
        if self.row_score is not None:
            cols["row_score"] = np.asarray(self.row_score, dtype=np.float64).reshape(n) \
                .astype(np.float32)
        return pd.DataFrame(cols)

    def metadata(self, task_effective) -> dict:
        """The parquet key-value metadata R-C2 names, plus the part and the layout."""
        import numpy as np
        truth = None if self.truth is None else np.asarray(self.truth)
        head = np.asarray(self.head)
        width = 1 if head.ndim == 1 else int(np.prod(head.shape[1:]))
        return {"vega.layout": PREDICTIONS_LAYOUT, "vega.part": "valid",
                "vega.row_id_kind": str(self.row_id_kind),
                "vega.truth_shape": json.dumps(None if truth is None
                                               else [int(d) for d in truth.shape[1:]]),
                "vega.score_shape": json.dumps([] if width == 1 else [width]),
                "vega.task_effective": str(task_effective),
                "vega.score_source": self.score_source}

    def unscored_record(self) -> dict:
        """{count, reasons}: the valid rows that got no prediction, by reason — never a
        fabricated answer (D1 §3.5)."""
        reasons: dict = {}
        for row in self.unscored:
            reason = str(plain(getattr(row, "reason", "unknown")))
            reasons[reason] = reasons.get(reason, 0) + 1
        return {"count": len(self.unscored), "reasons": reasons}


def _columns(prefix, values) -> dict:
    """`prefix` for a one-dimensional array, else `prefix_0 … prefix_{k-1}`."""
    import numpy as np
    values = np.asarray(values)
    if values.ndim == 1:
        return {prefix: values}
    return {f"{prefix}_{k}": values[:, k] for k in range(values.shape[1])}


def _head_out(spec) -> int | None:
    """The spec's head width (the last `head` block's `out`), None when it states none."""
    heads = [b for b in (spec.get("blocks") or []) if isinstance(b, Mapping)
             and b.get("type") == "head"]
    out = ((heads[-1].get("params") or {}).get("out")) if heads else None
    return int(out) if isinstance(out, int) and not isinstance(out, bool) else None


def write_valid_predictions(run_dir, adapters, spec_obj, checkpoint, campaign_root) -> dict:
    """The valid part scored from best.pt, as `predictions.valid.parquet` in
    `vega/predictions-layout@1` — the layout /vega-eval stores for the test part and
    /vega-pack's readers read (`reader.from_stored`), so a valid row and a test row are read
    by one rule. Returns {path, rows, unscored}."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    model = adapters.store.load(checkpoint, spec=spec_obj, build=adapters.build,
                                run_dir=str(campaign_root))
    part = adapters.data.load_part(spec_obj, "valid")
    if adapters.evaluator is not None:
        stored = StoredScores.of_evaluator(adapters.evaluator, spec_obj, part, model)
    else:
        stored = StoredScores.of_trainer(adapters.trainer, spec_obj, part, model,
                                         _head_out(spec_obj.to_dict()))
    domain = adapters.package.domain
    effective = plain(domain.Task.from_spec(spec_obj).effective)
    table = pa.Table.from_pandas(stored.frame(), preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({k.encode(): v.encode() for k, v in stored.metadata(effective).items()})
    path = Path(run_dir) / "predictions.valid.parquet"
    pq.write_table(table.replace_schema_metadata(meta), path)
    return {"path": str(path), "rows": len(stored.row_ids), "unscored": stored.unscored_record()}
