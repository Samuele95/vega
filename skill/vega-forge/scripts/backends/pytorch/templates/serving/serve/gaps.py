"""
serve.gaps — what a refused row takes out of the stream's windows (the stream consumer's warning).

A streamed row that breaks the contract is refused and never joins the history, so the rows of
its series that follow are answered over a window without it — the answer a request without the
refused row gets, which is not the one the full stream would have given. `SeriesGaps` remembers,
per series (`MANIFEST.sample.series_by`, else the one stream), how many of the next answers a
refused row's absence can reach — `history_rows`, the window and the lags behind each row — and
puts `GAP_WARNING` on each of them, so no such answer is silently different. A refusal whose
series cannot be read (a line that is not a row, a row without its series key) reaches every
series.

    SeriesGaps(size, series_by)
        .refused(items, positions)         refused raw items, at their stream positions
        .refused_in(items, first, accepted)   a batch's items (from position `first`) not accepted
        .mark(records, predictions, positions)   warn the answers after a gap that it reaches
"""
from __future__ import annotations

from typing import Mapping, Optional

__all__ = ["GAP_WARNING", "SeriesGaps"]

GAP_WARNING = ("a refused row earlier in this series is missing from this answer's window: it is "
               "answered over the accepted rows only")


class SeriesGaps:
    """Per series, the answers a refused row's absence still reaches (`None`: every series)."""

    def __init__(self, size: int, series_by: Optional[str] = None):
        self._size, self._series_by = int(size), series_by or None
        self._left: dict = {}

    def _series_of(self, item):
        if self._series_by is None or not isinstance(item, Mapping):
            return None
        value = item.get(self._series_by)
        return None if value is None else str(value)

    def refused(self, items, positions) -> None:
        """Each refused item opens a gap in its series, at its stream position, for the next
        `size` answers of that series after it."""
        if self._size:
            for item, position in zip(items, positions):
                self._left[self._series_of(item)] = (int(position), self._size)

    def refused_in(self, items, first: int, accepted) -> None:
        """The items of a batch that starts at stream position `first`, but the accepted ones."""
        taken = set(accepted)
        gone = [i for i in range(len(items)) if first + i not in taken]
        self.refused([items[i] for i in gone], [first + i for i in gone])

    def mark(self, records, predictions, positions) -> None:
        """`GAP_WARNING` on every answer after a gap (records, answers and stream positions
        aligned) that the gap still reaches."""
        for record, prediction, position in zip(records, predictions, positions):
            key = self._series_of(record)
            hits = [k for k in {key, None} if k in self._left and self._left[k][0] < position]
            if hits and GAP_WARNING not in prediction.setdefault("warnings", []):
                prediction["warnings"].append(GAP_WARNING)
            for k in hits:
                at, left = self._left[k]
                self._left[k] = (at, left - 1)
            self._left = {k: v for k, v in self._left.items() if v[1] > 0}
