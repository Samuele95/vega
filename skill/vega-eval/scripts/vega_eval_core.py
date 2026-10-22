"""
vega_eval_core — the seal, the shared bootstrap, and the numbers a reading may contain.

`evaluate.py` and `apply_evaluation_layer.py` both need these, and putting them here once is
what keeps the seal's rules and the interval arithmetic from being stated twice and drifting.
Same discipline as the rest of the chain's tooling: pure functions and frozen value objects,
no argv, no printing, no exits — the CLI shells decide what to say and what to return.

Three things are worth stating plainly, because they are the substance rather than the plumbing:

**The bootstrap is paired.** `Resampler` draws its index arrays *once* at `boot_seed` and every
model is scored on the same resamples. That is what makes the *difference* between the network
and a baseline have a confidence interval at all: resampling each model independently would
give two intervals whose overlap says nothing about whether one is better than the other.

**A metric without an interval is not a result.** `Estimate` carries the point and the CI
together, and there is no constructor that produces one without the other. The one exception
the contract allows — an upstream number copied from the ledger — is marked `upstream` and says
so, so a reader can see which numbers this stage measured and which it inherited.

**The seal is a record, not a convention.** `Seal` is written before the first byte of the test
part is read, and every later look increments `exposures` and appends its reason. The
multiple-look penalty is not computed away; it is disclosed.

**Every metric is read one way.** `Scoring` carries the emitted metrics module, the task's
effective kind and one `MetricOptions` value (the package domain's: the threshold the run chose
on valid, the season, mase's train-side scale) and calls `metric(name, y_true, y_score,
options, task)` — the backend's five-parameter entry point, no keyword knobs (C12). The interval
level is `CI_LEVEL` (`1 − ALPHA`), the one `Estimate.of` draws at, and `metrics.json.ci_level`
records it (BUILD-PLAN R-C2).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = ["ALPHA", "CI_LEVEL", "Estimate", "Resampler", "Scoring", "Seal", "SealError", "at",
           "digest_file", "now", "read_json",
           "numbers_in", "reading_numbers", "round_like", "unsupported_numbers"]

#: the two-sided bootstrap interval's tail mass: every interval this stage draws is at
#: CI_LEVEL, and metrics.json records that level (`ci_level`) so no reader assumes it
ALPHA = 0.05
CI_LEVEL = round(1.0 - ALPHA, 6)


class SealError(RuntimeError):
    """A refusal that concerns the test part. Never worked around — fixed upstream."""


# --------------------------------------------------------------------------- small helpers

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def read_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"{what} not found: {path}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{what} could not be read ({path}): {exc}"


def at(values, idx):
    """`values[idx]`, or None when there are no values — an unlabelled part has no truth, and
    a metric that reads none (`anomaly_score_mean`) is still resampled row for row."""
    return None if values is None else values[idx]


# --------------------------------------------------------------------------- estimates

def _round(value, places=4):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), places)


@dataclass(frozen=True)
class Estimate:
    """A number this stage measured, with the interval that says how much to trust it.

    There is deliberately no way to build one without a CI. The contract's rule — "every
    metric travels with a confidence interval" — is enforced by the type rather than by
    remembering to add one at each call site.
    """

    value: float | None
    ci: tuple = field(default_factory=tuple)
    n: int | None = None
    upstream: bool = False
    note: str | None = None
    #: a per-element reading (per-class f1, per-feature error) has a vector where a scalar
    #: metric has a point. It lives here rather than under a `<name>_values` sibling key,
    #: because a key suffix encoding "this entry is a different kind of thing" makes every
    #: consumer parse the string to recover what the type could have told it.
    values: tuple = field(default_factory=tuple)
    #: the interval of each element of `values`, drawn on the shared resamples at CI_LEVEL
    #: (`[lo, hi]`, or None where too few resamples could score that element) — what a reader
    #: of a per-unit breakdown needs to call a unit weak on its interval, not on its point
    cis: tuple = field(default_factory=tuple)

    def __post_init__(self):
        """The class's one rule, checked rather than documented.

        A point may stand without an interval in exactly three cases, each of which says so on
        the record: it came from an earlier stage (`upstream`), or it carries a `note` saying
        why no interval could be drawn (too few scorable resamples), or there is no point at
        all. Anything else — a bare number on a held-out part — is the reading this whole stage
        exists to keep out of a record, so it is refused at construction rather than caught by
        a gate three files away.
        """
        if self.value is None:
            return
        if self.ci or self.upstream or self.note:
            return
        raise ValueError(
            f"Estimate({self.value!r}) has a point, no confidence interval, and no note "
            "saying why: a metric on a held-out part without an interval is a number without "
            "a scale. Build it with Estimate.of(value, samples) to draw one, with "
            "Estimate.upstream_value(value, note) if an earlier stage measured it, or pass a "
            "note saying why no interval exists.")

    @staticmethod
    def from_json(doc) -> "Estimate":
        """Rehydrate an Estimate from its own `as_json` form.

        The construction need that had no factory, and was therefore being served by the raw
        constructor at the one call site that wanted only `better_than_zero`.
        """
        if not isinstance(doc, Mapping):
            raise TypeError(f"not an estimate record: {doc!r}")
        ci = doc.get("ci") or ()
        return Estimate(doc.get("value"), tuple(ci), doc.get("n"),
                        bool(doc.get("_upstream")), doc.get("note"),
                        tuple(doc.get("values") or ()),
                        tuple(tuple(c) if c else None for c in doc.get("cis") or ()))

    @staticmethod
    def of(value, samples: Sequence[float] | None = None, *, alpha=ALPHA, n=None,
           note=None) -> "Estimate":
        """A point estimate and the percentile CI of its bootstrap samples, at `1 − alpha`
        (`CI_LEVEL` unless a caller says otherwise — and then it records its own level)."""
        if value is None:
            return Estimate(None, (), n, False, note or "not computable on this part")
        clean = [float(s) for s in (samples or [])
                 if s is not None and math.isfinite(float(s))]
        if len(clean) < 2:
            return Estimate(_round(value), (), n, False,
                            note or "too few bootstrap resamples for an interval")
        clean.sort()
        lo = clean[max(0, int((alpha / 2) * len(clean)) - 1)]
        hi = clean[min(len(clean) - 1, int((1 - alpha / 2) * len(clean)))]
        return Estimate(_round(value), (_round(lo), _round(hi)), n, False, note)

    @staticmethod
    def per_element(values, *, n=None, note=None, cis=()) -> "Estimate":
        """A vector reading: per-class f1, per-feature error, residual quantiles.

        It carries no single point, and says why: inventing a scalar to stand for the vector
        would be a number the record does not hold. Each element may carry its own interval
        (`cis`, from `Resampler.estimate_elements`), because a per-unit breakdown that calls a
        unit weak reads that unit's interval.
        """
        return Estimate(None, (), n, False,
                        note or "a per-element vector; the elements are in `values`",
                        tuple(values), tuple(cis))

    @staticmethod
    def upstream_value(value, note) -> "Estimate":
        """A number copied from an earlier stage's record: no new interval, and it says so.

        The contract names exactly two — the selected run's valid metrics from the ledger, and
        `campaign-baseline`'s ledger value — and the apply script's CI rule skips rows marked
        this way rather than pretending an interval exists for them.
        """
        return Estimate(_round(value), (), None, True, note)

    def as_json(self) -> dict:
        out = {"value": self.value}
        if self.values:
            out["values"] = list(self.values)
        if self.cis:
            out["cis"] = [list(c) if c else None for c in self.cis]
        if self.ci:
            out["ci"] = list(self.ci)
        if self.n is not None:
            out["n"] = self.n
        if self.upstream:
            out["_upstream"] = True
        if self.note:
            out["note"] = self.note
        return out

    @property
    def has_interval(self) -> bool:
        """Whether a usable two-sided interval is present. The predicate lives here because
        four files were each spelling it out, and a rule spelled out four times is four rules
        that can disagree."""
        return (isinstance(self.ci, (list, tuple)) and len(self.ci) == 2
                and self.ci[0] is not None and self.ci[1] is not None)

    @property
    def publishable(self) -> bool:
        """Whether eval-contract §6 lets this entry into the layer.

        Stricter than construction: the core will build a point whose `note` explains that too
        few resamples could be scored, because that is an honest thing to record — but the
        published layer carries no point without an interval unless the record itself says the
        number came from an earlier stage.
        """
        return self.value is None or self.upstream or self.has_interval

    @property
    def excludes_zero(self) -> bool | None:
        if not self.has_interval:
            return None
        return self.ci[0] > 0 or self.ci[1] < 0

    def better_than_zero(self, mode="max") -> bool | None:
        """Whether the difference is in the mode's favour AND its interval excludes 0."""
        if not self.has_interval:
            return None
        return self.ci[0] > 0 if mode == "max" else self.ci[1] < 0


@dataclass(frozen=True)
class Scoring:
    """How this run reads a metric, carried as one value instead of five arguments.

    The threshold this run chose on the valid part is the reason this type exists. It was
    being passed to the headline metrics as a keyword dict and *not* to the slice and
    robustness builders, whose eleven-parameter signatures gave nothing away — so those
    numbers were read at the metrics module's own `threshold=0.5` default while the headline
    was read at the chosen one, and on a binary fixture (where the ranking metrics take no
    threshold) nothing showed. Routing every metric call through one frozen object makes the
    omission unrepresentable: there is no call that can forget the options, because the options
    are not a parameter.

    `options` is the package domain's `MetricOptions` (any frozen dataclass with a `threshold`
    field: this module imports no emitted package); `task` the effective kind the metrics read
    (`binary`, `multilabel`, `anomaly`, …) — a score is never read by guessing from its values.
    `kind` is the catalog row the metric names came from (a `sequence-*` kind keeps its own
    row while its scores are read as its base).
    """

    metrics: Any                       # the emitted model.metrics module
    kind: str | None                   # the task kind, per the catalog
    primary: str
    mode: str = "max"
    names: tuple = field(default_factory=tuple)
    options: Any = None                # MetricOptions: the threshold, season, mase_scale, …
    task: Any = None                   # the effective kind the metrics read the score as

    def with_options(self, options) -> "Scoring":
        """The same reading, at `options` (the threshold the run chose on valid, …)."""
        return replace(self, options=options)

    def at_threshold(self, threshold) -> "Scoring":
        """The same reading at another threshold — a candidate the valid-part chooser tries,
        one value or one per label."""
        if self.options is None:
            raise ValueError("a Scoring without MetricOptions has no threshold to move; build "
                             "it with the package's MetricOptions")
        return replace(self, options=replace(self.options, threshold=threshold))

    @property
    def threshold(self):
        return None if self.options is None else self.options.threshold

    def point(self, name, y, score):
        """One metric, at this run's options. `None` when it is undefined on this sample or is
        a per-element vector — never a number standing in for one."""
        value = self.vector(name, y, score)
        return None if isinstance(value, (list, dict, tuple)) else value

    def vector(self, name, y, score):
        """The raw result, per-element vectors included; for the callers that want them."""
        try:
            return self.metrics.metric(name, y, score, self.options, self.task)
        except Exception:  # noqa: BLE001
            return None

    def primary_of(self, y, score):
        return self.point(self.primary, y, score)


class Resampler:
    """The bootstrap's index draws, made once and reused for every model.

    The paired difference is the whole point: `network[i] - tree[i]` on the *same* resample i,
    for every i, gives an interval on the difference. Two independently drawn intervals do not,
    and reading their overlap as a comparison is the most common way a benchmark lies.
    """

    #: the level of every interval `estimate` and `difference_of` draw (metrics.json ci_level)
    ci_level = CI_LEVEL

    def __init__(self, n_rows: int, n_boot: int = 1000, seed: int = 7):
        import numpy as np
        self.n_rows, self.n_boot, self.seed = int(n_rows), int(n_boot), int(seed)
        rng = np.random.default_rng(self.seed)
        self.indices = [rng.integers(0, self.n_rows, self.n_rows) for _ in range(self.n_boot)]

    def per_draw(self, fn: Callable[[Any], float]) -> list:
        """`fn(idx)` on every resample, **in draw order**, with `None` where it could not be
        scored (a draw holding one class only, say).

        Draw order is what makes the vector reusable: two models scored this way are aligned
        draw for draw, so their difference can be taken without either being scored twice. A
        model that appears in several comparisons is scored once.
        """
        out = []
        for idx in self.indices:
            try:
                value = fn(idx)
            except Exception:  # noqa: BLE001
                out.append(None)
                continue
            out.append(float(value)
                       if value is not None and math.isfinite(float(value)) else None)
        return out

    def samples(self, fn: Callable[[Any], float]) -> list:
        """The scored resamples, with the unscorable ones dropped. A resample the metric cannot
        score contributes nothing rather than a fabricated number."""
        return [v for v in self.per_draw(fn) if v is not None]

    def estimate(self, point, fn, *, note=None) -> Estimate:
        return Estimate.of(point, self.samples(fn), n=self.n_rows, note=note)

    def estimate_elements(self, values, fn, *, note=None) -> Estimate:
        """A vector reading with an interval per element: `fn(idx)` returns the vector on one
        resample (a draw that cannot score it, or scores a vector of another length, counts
        for no element), and element j's interval is the percentile interval of its draws."""
        # an element the metric leaves undefined (nan: a label that never occurs) is recorded
        # as null — a JSON record the layer's reader can parse holds no NaN
        values = [None if v is None or not math.isfinite(float(v)) else float(v)
                  for v in values]
        draws = []
        for idx in self.indices:
            try:
                got = fn(idx)
            except Exception:  # noqa: BLE001
                continue
            got = list(got.values()) if isinstance(got, Mapping) else got
            if got is not None and len(got) == len(values):
                draws.append(got)
        cis = []
        for j, value in enumerate(values):
            if value is None or not math.isfinite(value):
                cis.append(None)
                continue
            element = Estimate.of(value, [d[j] for d in draws
                                          if d[j] is not None and math.isfinite(float(d[j]))])
            cis.append(tuple(element.ci) if element.has_interval else None)
        return Estimate.per_element(values, n=self.n_rows, note=note, cis=cis)

    def difference_of(self, draws_a, draws_b, point_a, point_b, *, note=None) -> Estimate:
        """The difference a − b from two vectors already scored on the same draws.

        A draw either side could not score contributes nothing to the interval, exactly as it
        would if both had been scored inside this call.
        """
        if point_a is None or point_b is None:
            return Estimate(None, (), self.n_rows, False,
                            note or "one of the two could not be scored")
        diffs = [a - b for a, b in zip(draws_a, draws_b)
                 if a is not None and b is not None]
        return Estimate.of(float(point_a) - float(point_b), diffs, n=self.n_rows, note=note)

    def paired_difference(self, fn_a, fn_b, point_a, point_b, *, note=None) -> Estimate:
        """The difference a − b, resample by resample, scoring both here.

        Convenient for a one-off comparison. A caller comparing one model against several
        should score it once with `per_draw` and use `difference_of`.
        """
        if point_a is None or point_b is None:
            return Estimate(None, (), self.n_rows, False,
                            note or "one of the two could not be scored")
        return self.difference_of(self.per_draw(fn_a), self.per_draw(fn_b),
                                  point_a, point_b, note=note)


# --------------------------------------------------------------------------- the seal

@dataclass(frozen=True)
class Seal:
    """`test_seal.json` — written before the first byte of the test part is read."""

    campaign: str
    run: str
    test: Mapping[str, Any]
    opened_at: str
    exposures: int
    history: tuple
    token: str
    #: a graph sample's exposure record — {protocol, edges_into_test, features_seen_in_training}
    #: (eval-contract §1) — absent for every other shape
    graph: Mapping | None = None
    #: the looks an earlier campaign took at this test part — `campaign.json.after_eval`,
    #: `{previous_campaign, exposures}` (eval-contract §1) — absent in a campaign of its own
    after_eval: Mapping | None = None

    @staticmethod
    def create(campaign_dir, run_id, test_path, rows, reason, gate,
               after_eval: Mapping | None = None) -> "Seal":
        """The first look in this campaign. In a campaign `/vega-train --after-eval` started,
        the test part has already been seen `after_eval.exposures` times: this look is the next
        one, and `exposures` counts them all — the earlier looks are disclosed, never reset."""
        earlier = int((after_eval or {}).get("exposures") or 0)
        return Seal(campaign=str(campaign_dir), run=str(run_id),
                    test={"path": str(test_path), "digest": None, "rows": rows,
                          "digest_matches_manifest": None},
                    opened_at=now(), exposures=earlier + 1,
                    history=({"at": now(), "run": str(run_id), "reason": reason,
                              "gate": gate},),
                    token=secrets.token_hex(16),
                    after_eval=dict(after_eval) if after_eval else None)

    @staticmethod
    def load(path) -> tuple:
        raw, err = read_json(path, "test_seal.json")
        if err:
            return None, err
        try:
            return Seal(campaign=raw["campaign"], run=raw["run"], test=raw["test"],
                        opened_at=raw["opened_at"], exposures=int(raw["exposures"]),
                        history=tuple(raw.get("history") or ()), token=raw["token"],
                        graph=raw.get("graph"), after_eval=raw.get("after_eval")), None
        except (KeyError, TypeError, ValueError) as exc:
            return None, f"test_seal.json is malformed: {exc}"

    def again(self, run_id, reason, gate) -> "Seal":
        """A further look. The count goes up and the reason goes on the record — the penalty
        for looking twice is disclosed, never computed away."""
        if not reason:
            raise SealError("a further exposure of the test part needs a stated reason "
                            "(--again \"<reason>\"); an unexplained second look is the one "
                            "thing a held-out part cannot survive")
        return Seal(self.campaign, str(run_id), dict(self.test), now(), self.exposures + 1,
                    self.history + ({"at": now(), "run": str(run_id), "reason": reason,
                                     "gate": gate},), self.token, self.graph, self.after_eval)

    def with_digest(self, digest, matches) -> "Seal":
        """The digest this run found, and whether it matched the manifest's.

        `matches` is deliberately three-valued and is stored as given: True it matched, False
        it did not, **None the manifest recorded no digest to match against**. Coercing that
        None to a boolean would make the seal assert a comparison nobody performed, in the one
        record a later reader cannot re-derive.
        """
        if matches not in (True, False, None):
            raise ValueError(f"matches must be True, False or None (not checked), not "
                             f"{matches!r}: the seal records what was compared, and a truthy "
                             "stand-in for 'no digest to compare against' is the claim this "
                             "stage exists to prevent")
        test = dict(self.test)
        test["digest"] = digest
        test["digest_matches_manifest"] = matches
        return replace(self, test=test)

    def with_graph(self, record: Mapping | None) -> "Seal":
        """A graph sample's exposure record — what the training graph saw of the test part
        (zero and false under `inductive` by construction; counted and disclosed under
        `transductive`) — or the seal unchanged for any other shape."""
        return self if record is None else replace(self, graph=dict(record))

    def as_json(self, *, with_token=True) -> dict:
        out = {"schema": "vega/test-seal@1", "campaign": self.campaign, "run": self.run,
               "test": dict(self.test), "opened_at": self.opened_at,
               "exposures": self.exposures, "history": [dict(h) for h in self.history]}
        if self.graph is not None:
            out["graph"] = dict(self.graph)
        if self.after_eval is not None:
            out["after_eval"] = dict(self.after_eval)
        if with_token:
            out["token"] = self.token
        return out

    def write(self, path) -> None:
        Path(path).write_text(json.dumps(self.as_json(), indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")


# --------------------------------------------------------------------------- retyped digits

_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")

# numbers a reading may use without them appearing in the records: they are prose, not results
_FREE = {0.0, 1.0, 2.0, 3.0, 100.0}


def numbers_in(value, out=None) -> set:
    """Every number anywhere in a record tree, as floats."""
    out = set() if out is None else out
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            out.add(float(value))
        return out
    if isinstance(value, Mapping):
        for v in value.values():
            numbers_in(v, out)
        return out
    if isinstance(value, str):
        for match in _NUMBER.finditer(value):
            try:
                out.add(float(match.group()))
            except ValueError:
                continue
        return out
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for v in value:
            numbers_in(v, out)
    return out


def reading_numbers(text: str) -> list:
    """The numbers a reading states, in order, as (text, float)."""
    out = []
    for match in _NUMBER.finditer(text or ""):
        try:
            out.append((match.group(), float(match.group())))
        except ValueError:
            continue
    return out


def round_like(value: float, shown: str) -> float:
    """The record's number rounded the way the reading wrote it, so `0.71` matches
    `0.7132` when the reading rounded to two places and not when it invented a digit."""
    if "e" in shown.lower():
        return float(f"{value:.{max(1, len(shown.split('.')[-1]))}g}")
    places = len(shown.split(".")[1]) if "." in shown else 0
    return round(float(value), places)


def unsupported_numbers(text: str, records) -> list:
    """The numbers in a reading that no record supports at the rounding the reading used.

    This is the check the contract asks for in as many words — "a retyped digit is refused".
    It exists because the readings are where a verdict is stated in prose, and a verdict whose
    numbers drifted from the records it claims to summarise is worse than no verdict: it reads
    as evidence while being a memory of evidence.
    """
    available = numbers_in(records)
    bad = []
    for shown, value in reading_numbers(text):
        if value in _FREE or abs(value) in _FREE:
            continue
        if any(abs(round_like(candidate, shown) - value) < 1e-12 for candidate in available):
            continue
        # an integer the records hold as a float, or the other way round
        if float(value).is_integer() and any(
                abs(float(c) - value) < 1e-9 for c in available):
            continue
        bad.append(shown)
    return bad
