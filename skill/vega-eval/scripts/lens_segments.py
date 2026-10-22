"""
lens_segments — the segments /data-lens found, as slices of the test rows (eval-contract §3).

The analysis's `segments` module records its k-means profiles (`evidence.profiles[]`: segment,
n, share, medians, standardized differences) and the rule that places a row in a segment
(`evidence.assignment`: the nearest of `centroids` to `(x - center) / scale` over `features`,
analysis-contract §1). A profile describes a segment; only the assignment can put a test row in
one, and it reads the **raw** columns the clustering read — so the rows are the test
part's raw rows, fetched after the seal through the shaper's `fitted.json.__split_raw__.test`
(raw-file positions, in the shaped test part's order).

When a slice cannot be cut, the reason is a record, never a silence: `segment_plan` returns
either the rule or the plain statement of why no segment slice was cut (the module did not run,
it found no structure, it recorded profiles without the rule, the sample is not one raw row),
which `slices.json.not_cut` carries into the layer.

Pure apart from `raw_test_rows`, which reads the shaper's records and the raw input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = ["SOURCE", "SegmentRule", "segment_plan", "raw_test_rows"]

SOURCE = "analysis:modules/segments"
#: the tokens the shaper reads as missing, never as real infinities (shape-contract §2)
NA_TOKENS = ["Infinity", "-Infinity", "inf", "-inf", "+inf", "INF", "-INF", "NaN", "nan"]


@dataclass(frozen=True)
class SegmentRule:
    """`evidence.assignment` as a value: the k-means prediction over the raw columns."""

    features: tuple
    center: tuple
    scale: tuple
    centroids: tuple
    segments: tuple
    k: int
    profiles: Mapping                     # segment -> its profile (n, share, …)

    def assign(self, raw) -> Any:
        """Each row's segment (-1 for a row missing one of the features or holding a value
        that is not a number): the nearest centroid in the standardized basis."""
        import numpy as np
        import pandas as pd
        x = raw[list(self.features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        z = (x - np.asarray(self.center, dtype=float)) / np.asarray(self.scale, dtype=float)
        dist = ((z[:, None, :] - np.asarray(self.centroids, dtype=float)[None]) ** 2).sum(-1)
        out = np.asarray(self.segments)[dist.argmin(axis=1)]
        return np.where(np.isfinite(x).all(axis=1), out, -1)

    def describe(self, segment: int) -> str:
        """What the segment is, in the analysis's own terms: its share of the analysed rows and
        the basis columns it differs on most (the standardized difference of its mean)."""
        prof = self.profiles.get(segment) or {}
        diffs = {c: v for c, v in (prof.get("standardized_difference") or {}).items()
                 if isinstance(v, (int, float))}
        top = sorted(diffs, key=lambda c: -abs(diffs[c]))[:2]
        parts = [f"{c} {'above' if diffs[c] > 0 else 'below'} the whole" for c in top]
        share = prof.get("share")
        return (f"k-means segment {segment} of {self.k} from the analysis"
                + (f" ({share:.1%} of the rows it read)" if isinstance(share, float) else "")
                + (": " + ", ".join(parts) if parts else ""))


def _numbers(values, n=None) -> bool:
    ok = isinstance(values, (list, tuple)) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
    return ok and (n is None or len(values) == n)


def segment_plan(segments) -> tuple[SegmentRule | None, str | None]:
    """(the rule, None) when the analysis's segments module can place a test row in a segment;
    (None, why) when it cannot — said plainly, because a slice the chain named and the
    evaluation did not cut is a finding about the evaluation; (None, None) when there is no
    segments module at all (no analysis layer)."""
    if not isinstance(segments, Mapping) or not segments:
        return None, None
    if segments.get("ran") is False:
        why = segments.get("skipped_because") or "no reason recorded"
        return None, f"the analysis's segments module did not run ({why}), so no segment was named"
    ev = segments.get("evidence") if isinstance(segments.get("evidence"), Mapping) else {}
    profiles = [p for p in ev.get("profiles") or () if isinstance(p, Mapping)]
    k = ev.get("chosen_k")
    if not k or not profiles:
        feats = ev.get("features") or []
        return None, ("the analysis's segments module found no segment structure to cut "
                      f"(no k chosen; it read {len(feats)} numeric basis column(s), and a scan "
                      "needs at least two and 30 complete rows)")
    rule = ev.get("assignment")
    if not isinstance(rule, Mapping):
        return None, (f"the analysis recorded {k} k-means segment profiles but not the rule that "
                      "places a row in one (no `evidence.assignment`: centroids, center, scale) — "
                      "a profile describes a segment and cannot assign a test row to it, so no "
                      "segment slice was cut; re-running /data-lens records the rule")
    feats = rule.get("features")
    m = len(feats) if isinstance(feats, list) else -1
    cents = rule.get("centroids")
    segs = rule.get("segments") or list(range(len(cents or [])))
    if m < 1 or not all(isinstance(c, str) for c in feats) or not _numbers(rule.get("center"), m) \
            or not _numbers(rule.get("scale"), m) or not isinstance(cents, list) or not cents \
            or not all(_numbers(c, m) for c in cents) or len(segs) != len(cents) \
            or any(not v for v in rule.get("scale")):
        return None, ("the analysis's segment assignment is malformed (features, center, scale "
                      "and every centroid must be numbers of one length, and a scale never 0), "
                      "so no segment slice was cut")
    return SegmentRule(features=tuple(feats), center=tuple(rule["center"]),
                       scale=tuple(rule["scale"]), centroids=tuple(tuple(c) for c in cents),
                       segments=tuple(int(s) for s in segs), k=int(k),
                       profiles={int(p.get("segment", -1)): p for p in profiles}), None


def _resolve(path, *bases) -> Path | None:
    p = Path(str(path))
    for cand in ([p] if p.is_absolute() else [p] + [b / p for b in bases if b is not None]):
        if cand.is_file():
            return cand
    return None


def raw_test_rows(manifest_path, n_rows: int, features=()) -> tuple[Any, str | None]:
    """(the raw rows of the test part in the shaped test part's order, None) or (None, why).

    Read after the seal, by the evaluator, as part of its one look: the shaper's manifest names
    the input and `fitted.json`, whose `__split_raw__.test` holds the raw-file positions of the
    test rows in the part's order. Only `features` are read when given. A count that does not
    match the scored test rows is refused (the rows would not be the ones scored)."""
    import pandas as pd
    mpath = _resolve(manifest_path) if manifest_path else None
    if mpath is None:
        return None, f"the shaper's manifest ({manifest_path!r}) cannot be read"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs") or {}
    out_dir = Path(str(outputs.get("dir") or mpath.parent))
    fitted = _resolve(Path(out_dir) / str(outputs.get("fitted") or "fitted.json"),
                      mpath.parent) or _resolve(mpath.parent / "fitted.json")
    if fitted is None:
        return None, "the shaper's fitted.json (the raw positions of the test rows) is missing"
    positions = (json.loads(fitted.read_text(encoding="utf-8")).get("__split_raw__") or {}) \
        .get("test")
    if not isinstance(positions, list) or len(positions) != n_rows:
        return None, (f"the shaper's __split_raw__.test holds "
                      f"{len(positions) if isinstance(positions, list) else 'no'} position(s) for "
                      f"{n_rows} scored test row(s), so the raw rows are not the rows scored")
    src = (manifest.get("input") or {}).get("path")
    raw_path = _resolve(src, mpath.parent, mpath.parent.parent) if src else None
    if raw_path is None:
        return None, f"the raw input the shaper read ({src!r}) is not where its manifest says"
    cols = list(dict.fromkeys(features)) or None
    try:
        if raw_path.suffix in (".parquet", ".pq"):
            from pyarrow.fs import LocalFileSystem      # A40: never a Python file object
            frame = pd.read_parquet(raw_path, columns=cols, filesystem=LocalFileSystem())
        else:
            frame = pd.read_csv(raw_path, usecols=cols, na_values=NA_TOKENS,
                                keep_default_na=True, low_memory=False)
    except (ValueError, KeyError) as exc:          # a column the raw input does not hold
        return None, f"the raw input {raw_path.name} does not hold {cols} ({exc})"
    return frame.iloc[positions].reset_index(drop=True), None
