#!/usr/bin/env python3
"""
evaluate.py — open the test part once, under a stamped seal, and say what the network is worth.

    python3 evaluate.py --campaign <dir> --run <id> [--again "<reason>"]
                        [--threshold valid-optimum|0.5|<value>] [--slices col=value …]
                        [--baselines trivial,linear,tree,lens,campaign-baseline]
                        [--n-boot 1000] [--boot-seed 7] [--unattended]
                        [--vega-forge-dir <skills>/vega-forge]

The order of operations is the contract's, and it is not arrangeable:

  1. **Refuse before sealing.** A baseline set below the required one, a `--slices` item whose
     column is not a feature or whose literal the column's dtype cannot hold, and — when a seal
     already exists, or the campaign is an after-eval one whose predecessor already looked —
     a `--threshold <value>`, any `--slices`, or a baseline set narrower than the previous
     exposure's, are refused with **no seal written**. Anything chosen after a look is a choice
     made with knowledge of the answer.
  2. **Seal, then read.** `test_seal.json` is written before the first byte of the test part is
     read; the digest is taken right afterwards and compared with the shaper's manifest. A
     changed test file is a new chain, not a new number.
  3. **One look.** The network is loaded from the selected run's `best.pt` through the
     ModelStore and predicted through `load_sealed_part`, whose token only this file holds.
     Everything downstream — the notebook, the layer, /vega-pack — reads the stored predictions.

The emitted package is driven through its adapter classes, reached by blueprint-map role with
`vega_core.PackageAdapters.load` (BUILD-PLAN §2.12(2); a v2.3 package — an older one is refused
before the seal): `ShapedParts` opens the parts (the test part only through its sealed door),
`ModelRegistry` restores the checkpoint, and the package's `Evaluator.scored` is the **one
scoring path** for the network and every baseline alike (`Evaluator.heads` gives the raw head the
predictions store). Every metric is `metrics.metric(name, y, score, MetricOptions, task)` with
the task's effective kind: binary and multilabel scores are logits and σ is applied
unconditionally (R-C3). The stored predictions follow `vega/predictions-layout@1`
(`predictions_layout.py`, R-C2); `threshold.json` states its scale and, for a multilabel task,
its per-label thresholds; `metrics.json` records `ci_level`, the layout and the metric options;
`unscored.json` the samples that got no prediction; a graph sample's exposure record goes into
the seal and `metrics.json.graph`.

Every number this stage measures travels with a bootstrap CI, and every model is scored on the
**same** resamples, so the paired difference between the network and a baseline has an interval
of its own. The machine verdict is written from that interval, not from the point estimates —
except where no such interval means anything: on an unlabelled anomaly task the primary,
`anomaly_score_mean`, lives on each model's own score scale, so the verdict is **not
assessable** (`earned` null, with the reason on the record), every model's metrics are still
reported with their intervals, and no paired difference is computed (eval-contract §5).

Exit codes: 0 the evaluation ran; 2 an input or a pre-seal refusal (no seal written); 3 the
evaluation failed after the seal (the seal stands, and the exposure is recorded).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import campaign_baseline  # noqa: E402
import earlier_looks  # noqa: E402
import lens_segments  # noqa: E402
import user_slices  # noqa: E402
from eval_package import (DEFAULT_WHEN, FORGE_DIR, Package, anomaly_rate,  # noqa: E402
                          bottleneck, default_baselines, drifted_columns, lens_protocol,
                          lens_ranking, metric_options,
                          task_classes,
                          open_package, resolved_spec, sample_frame, sample_of)
from metric_gates import decided as metric_gates  # noqa: E402
from predictions_layout import LAYOUT, PredictionsLayout, unscored_record  # noqa: E402
from threshold_choice import (ThresholdChoice, check_threshold_arg,  # noqa: E402
                              choose_threshold, requested_mode)
from vega_eval_core import (CI_LEVEL, Estimate, Resampler, Scoring, Seal,  # noqa: E402
                            SealError, at, digest_file, read_json)

# The catalog is eval-contract §3–§4 in machine form. It is read, not restated: a metric row
# or a baseline definition that lived in this file too would be a second copy free to drift
# from the prose, and tests/test_catalog_sync.py exists to keep exactly one copy honest.
CATALOG = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
TASKS = CATALOG["tasks"]
BASELINES = CATALOG["baselines"]
REQUIRED_BASELINES = tuple(BASELINES["required"])
REQUIRED_ONE_OF = tuple(BASELINES["required_one_of"])
DEFAULT_BASELINES = tuple(BASELINES["default_set"])
KNOWN_BASELINES = tuple(BASELINES["rows"])
CLASSIFICATION = tuple(CATALOG["classification_kinds"])
# metrics that rank one row against another: undefined when a slice holds a single class
RANKING = tuple(CATALOG["ranking_metrics"])
# the metrics module's own default; the reliability curve is binned the same way so that the
# picture and the number are the same reading of the calibration
ECE_BINS = int(CATALOG["ece_bins"])
THRESHOLD_TASKS = tuple(CATALOG["threshold_tasks"])
SECONDARY = {kind: list(row.get("secondary") or []) for kind, row in TASKS.items()}
# eval-contract §5: the primaries whose value lives on each model's own score scale, by the
# task kind they are read under. `anomaly_score_mean` is the anomaly primary exactly when no
# label exists (§3): the network's reconstruction error, an isolation forest's negated
# score_samples and a PCA floor's error are different quantities, so a paired difference of
# them compares scales, not models — and a verdict read off it would be no finding.
SCALE_BOUND_PRIMARIES = {"anomaly": ("anomaly_score_mean",)}
NOT_ASSESSABLE = (
    "the task is anomaly detection and its primary, anomaly_score_mean (the anomaly primary "
    "when no label exists), reads no label: it is each model's own anomaly score averaged over "
    "the test part (a reconstruction error, a negated isolation-forest score, a distance). "
    "With no label to rank them against, those scores live on different scales, so a paired "
    "difference between two models would compare their scales rather than the models. Every "
    "model's metrics are reported with their intervals, and no paired difference is computed.")


def out_ok(msg, markers):
    markers.append(f"OK: {msg}")
    print(f"OK: {msg}")


def out_warn(msg, markers):
    markers.append(f"WARN: {msg}")
    print(f"WARN: {msg}")


# --------------------------------------------------------------------------- inputs

def load_campaign(camp_dir):
    """The campaign's records. This stage reads them; it never writes to the ledger."""
    out = {}
    for key, name in (("campaign", "campaign.json"), ("spec", "netspec.json"),
                      ("experiments", "experiments.json")):
        doc, err = read_json(Path(camp_dir) / name, name)
        if err:
            return None, err
        out[key] = doc
    return out, None


# --------------------------------------------------------------------------- pre-seal gates

@dataclass(frozen=True)
class PreSeal:
    """Everything a pre-seal refusal may look at, established once and never mutated.

    The rules below are pure functions of this value. Nothing here has touched the test part:
    that is the whole point of the phase — every one of these questions must be answerable, and
    answerable *wrongly*, before the seal exists.
    """

    args: Any
    campaign: Mapping
    ledger: Mapping
    run_id: str
    seal_path: Path
    baselines: tuple
    prior: Any = None                 # the existing Seal, when this campaign has one
    row: Mapping | None = None        # the ledger row for run_id
    earlier: Any = None               # EarlierLooks: an after-eval campaign's predecessor
    earlier_problem: str | None = None
    user_slices: tuple = ()           # the --slices items, declared (coerced) before the seal
    slice_problems: tuple = ()

    @property
    def exposure(self) -> int:
        """Which look this would be — counting an after-eval campaign's earlier looks."""
        if self.prior:
            return self.prior.exposures + 1
        return (self.earlier.exposures + 1) if self.earlier else 1

    @property
    def looked(self) -> bool:
        """Whether the test part has been seen before this look (here, or in a predecessor)."""
        return self.prior is not None or self.earlier is not None

    @property
    def previous_baselines(self) -> tuple:
        """What the last look measured against: this campaign's, else the predecessor's."""
        if self.prior is not None:
            return tuple(_previous_baselines(self.prior, self.seal_path))
        return tuple(self.earlier.baselines) if self.earlier else ()

    @property
    def previous_threshold_mode(self) -> str | None:
        """The `--threshold` mode the last look chose under (this campaign's, else the
        predecessor's): `valid-optimum`, `0.5`, or None when it cannot be repeated or is not
        recorded."""
        if self.prior is not None:
            doc, err = read_json(Path(self.seal_path).parent / "eval" / str(self.prior.run)
                                 / "threshold.json", "the previous exposure's threshold.json")
            return None if err else requested_mode(doc)
        return self.earlier.threshold_mode if self.earlier else None

    @property
    def last_look(self) -> str:
        """The last look, in words, for a refusal to name."""
        if self.prior is not None:
            return f"exposure {self.prior.exposures}"
        return (f"the {self.earlier.exposures} look(s) of the previous campaign "
                f"{self.earlier.previous_campaign}")


@dataclass(frozen=True)
class Refusal:
    """A named pre-seal rule, with the contract clause behind it.

    A registry rather than an if-chain for the reason the sibling stages use one: these are the
    refusals the whole stage exists for, and each should be citable, individually testable, and
    addable as a row rather than as an edit inside a sixty-line function. `stops` marks a rule
    whose failure makes the later rules meaningless — no ledger row means no exposure question
    to ask — and only those short-circuit, so an ordinary refusal can no longer hide the ones
    after it.
    """

    id: str
    clause: str
    check: Callable                   # PreSeal -> str | None
    when: Callable = lambda ctx: True
    stops: bool = False


def _r_baseline_floor(ctx):
    missing = [b for b in REQUIRED_BASELINES if b not in ctx.baselines]
    if missing:
        return (f"the baseline set is missing {', '.join(missing)}; the trivial floor is what "
                "every metric is read against, and a network below it is broken")
    return None


def _nor(names):
    """`neither `a` nor `b`` — the catalog's list, read as English rather than as a list."""
    quoted = [f"`{n}`" for n in names]
    if len(quoted) == 1:
        return f"no {quoted[0]}"
    return "neither " + " nor ".join([", ".join(quoted[:-1]), quoted[-1]]
                                     if len(quoted) > 2 else quoted)


def _r_baseline_contender(ctx):
    if not any(b in ctx.baselines for b in REQUIRED_ONE_OF):
        return (f"the baseline set has {_nor(REQUIRED_ONE_OF)}; without one of them the "
                "evaluation cannot say what the network's non-linearity bought")
    return None


def _r_baselines_known(ctx):
    unknown = [b for b in ctx.baselines if b not in KNOWN_BASELINES]
    if unknown:
        return (f"the baseline set names {', '.join(map(repr, unknown))}, which the catalog does "
                f"not define (it defines {', '.join(KNOWN_BASELINES)})")
    return None


def _r_run_in_ledger(ctx):
    if ctx.row is None:
        return f"run {ctx.run_id!r} is not in the ledger"
    return None


def _r_seal_removed(ctx):
    """A look leaves two tracks: the seal, and the records under `eval/`. The second without
    the first means the seal was removed, and re-sealing would reset an exposure count that has
    already been earned."""
    if ctx.prior is not None:
        return None
    traces = sorted(d.name for d in (ctx.seal_path.parent / "eval").glob("*")
                    if (d / "metrics.json").is_file())
    if not traces and not ctx.campaign.get("test_seal"):
        return None
    return (f"there is no {ctx.seal_path.name} but this campaign already holds an evaluation "
            f"of {', '.join(traces) or 'a run'}"
            + (" and campaign.json names a seal" if ctx.campaign.get("test_seal") else "")
            + ". The test part has been opened and the seal that recorded it is gone; restore "
            "it from the campaign's history — writing a new one would put the exposure count "
            "back to 1 for a part that has already been seen")


def _r_again_required(ctx):
    if not ctx.args.again:
        return (f"this campaign's test part has already been opened {ctx.prior.exposures} "
                f"time(s) (first at {ctx.prior.opened_at}). A further look needs "
                "--again \"<reason>\": the count and the reason go on the record, and the "
                "layer, the notebook and the summary all say how many times the test has "
                "been seen")
    return None


def _after_eval(ctx) -> str:
    """How an after-eval campaign's first look is a further one, for a refusal to say."""
    if ctx.prior is not None or ctx.earlier is None:
        return ""
    return (f" (an after-eval campaign: {ctx.earlier.previous_campaign} already opened this "
            f"test part {ctx.earlier.exposures} time(s), so this campaign's first look is "
            "a further one)")


def _r_threshold_after_a_look(ctx):
    if ctx.args.threshold not in (None, "valid-optimum", "0.5"):
        return (f"--threshold {ctx.args.threshold!r} on exposure {ctx.exposure}"
                f"{_after_eval(ctx)}: a threshold value chosen after a look is chosen knowing "
                "the answer. On a further exposure the threshold is valid-optimum or 0.5")
    previous, mode = ctx.previous_threshold_mode, ctx.args.threshold or "valid-optimum"
    if previous and mode != previous:
        return (f"--threshold {mode} on exposure {ctx.exposure}{_after_eval(ctx)}, where "
                f"{ctx.last_look} chose its threshold as {previous}: switching between "
                "valid-optimum and 0.5 after a look is choosing the threshold knowing the answer. "
                f"A further look keeps the mode the last one used (--threshold {previous})")
    return None


def _r_slices_after_a_look(ctx):
    if ctx.args.slices:
        return (f"--slices on a further exposure{_after_eval(ctx)}: a slice named after a look "
                "is a hypothesis fitted to the test part. The slices are the chain's")
    return None


def _r_baselines_narrowed(ctx):
    previous = set(ctx.previous_baselines)
    if previous and not previous <= set(ctx.baselines):
        return (f"the baseline set is narrower than {ctx.last_look}'s "
                f"({', '.join(sorted(previous - set(ctx.baselines)))} dropped){_after_eval(ctx)}; "
                "dropping a baseline the network was already measured against is choosing the "
                "comparison after seeing it")
    added = set(ctx.baselines) - previous
    if previous and added:
        return (f"the baseline set is wider than {ctx.last_look}'s "
                f"({', '.join(sorted(added))} added){_after_eval(ctx)}; a baseline added after "
                "the look is choosing the comparison after seeing it — a further look measures "
                "against the baselines the last one did")
    return None


def _r_earlier_looks(ctx):
    """An after-eval campaign's record of the earlier looks must reconcile with the previous
    campaign's seal before a further look is built on it."""
    if ctx.earlier_problem:
        return (f"{ctx.earlier_problem}. A campaign /vega-train --after-eval started carries the "
                "count of the looks before it, and the seal of this look counts them too")
    return None


def _r_user_slices(ctx):
    """A user slice is first-exposure only and cannot be re-declared: one lost to a spelling is
    lost for good, so what can be checked without a test row is checked here."""
    if ctx.slice_problems:
        return ("; ".join(ctx.slice_problems) + ". A slice the user names is declared once, "
                "before the look, so it is checked here rather than lost after it")
    return None


_HAS_PRIOR = staticmethod(lambda ctx: ctx.prior is not None)
_LOOKED = staticmethod(lambda ctx: ctx.looked)
_FIRST = staticmethod(lambda ctx: not ctx.looked)

# eval-contract §1–§2, in the order a reader would ask them. Every one of these is refused with
# NO seal written: a choice made after a look is a leak whatever its label, so it has to be
# caught before the look rather than apologised for after it.
REFUSALS = (
    Refusal("B1", "eval-contract §4", _r_baseline_floor),
    Refusal("B2", "eval-contract §4", _r_baseline_contender),
    Refusal("B4", "eval-contract §4", _r_baselines_known),
    Refusal("L1", "run-contract §4", _r_run_in_ledger, stops=True),
    Refusal("S1", "eval-contract §1", _r_seal_removed, stops=True),
    Refusal("S2", "eval-contract §1", _r_again_required, when=_HAS_PRIOR, stops=True),
    Refusal("S3", "eval-contract §1", _r_earlier_looks, stops=True),
    Refusal("T1", "eval-contract §2", _r_threshold_after_a_look, when=_LOOKED),
    Refusal("T2", "eval-contract §2", _r_slices_after_a_look, when=_LOOKED),
    Refusal("T3", "eval-contract §2", _r_user_slices, when=_FIRST),
    Refusal("B3", "eval-contract §4", _r_baselines_narrowed, when=_LOOKED),
)


def pre_seal_checks(args, camp, ledger, run_id, seal_path, baselines, markers, *,
                    dtypes=None, features=()):
    """Everything that must be refused **before** the seal is written.

    Collect-only, like every other gate in the chain: it returns the refusals and the frozen
    pre-seal context (the prior seal, an after-eval campaign's earlier looks, the user slices
    declared and coerced), and says nothing. The one qualification it used to print — that the
    run being evaluated is not the one the ledger selected — is appended to `markers` instead,
    so it reaches markers.json, the notebook and the published layer the way every other
    qualification in this run does. `dtypes` ({column: dtype}, the shaper's output_schema then
    the train part's) and `features` (the columns the slices read) declare the user slices;
    neither reads a test row.
    """
    seal_path = Path(seal_path)
    prior, err = (Seal.load(seal_path) if seal_path.is_file() else (None, None))
    if err:
        return [err], None
    earlier, earlier_problem = earlier_looks.read(camp, seal_path.parent)
    declared, slice_problems = user_slices.declare(args.slices, dtypes or {}, features)

    ctx = PreSeal(args=args, campaign=camp, ledger=ledger, run_id=run_id,
                  seal_path=seal_path, baselines=tuple(baselines), prior=prior,
                  row=next((r for r in ledger.get("runs") or [] if r.get("id") == run_id),
                           None),
                  earlier=earlier, earlier_problem=earlier_problem,
                  user_slices=declared, slice_problems=slice_problems)

    selection = (ledger.get("selection") or {}).get("run")
    if selection and selection != run_id:
        markers.append(f"WARN: the ledger selected {selection!r} and this evaluation is of "
                       f"{run_id!r}; the layer records which run was looked at, and the "
                       "summary should say why")

    errs = []
    for rule in REFUSALS:
        if not rule.when(ctx):
            continue
        problem = rule.check(ctx)
        if problem is None:
            continue
        errs.append(problem)
        if rule.stops:
            # a rule the later ones depend on: no ledger row means no exposure question to
            # ask, and a removed seal means no prior to compare a baseline set against
            break
    return errs, ctx


def _previous_baselines(prior, seal_path):
    """What the last exposure measured against, read from its own record."""
    eval_dir = Path(seal_path).parent / "eval" / prior.run
    doc, err = read_json(eval_dir / "baselines.json", "the previous exposure's baselines.json")
    if err:
        return []
    return [b.get("name") for b in (doc.get("baselines") or []) if b.get("name")]


# --------------------------------------------------------------------------- scoring

def _numpy(values):
    """A tensor on any device, or anything array-like, as an ndarray (None stays None)."""
    import numpy as np
    if values is None:
        return None
    detach = getattr(values, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(values)


def _one_column(score):
    """A (N, 1) score as its one column; any other score as it is."""
    score = _numpy(score)
    if score is not None and score.ndim == 2 and score.shape[1] == 1:
        return score.reshape(-1)
    return score


def estimates_for(resampler, scoring, y, score):
    """Every metric of the task's row as an Estimate, on the shared resamples. `y` is None for
    an unlabelled part: a metric that reads no truth (`anomaly_score_mean`) is still measured,
    and one that needs a label is recorded as not computable."""
    import numpy as np
    y = None if y is None else np.asarray(y)
    score = np.asarray(score)
    n = len(score)
    out = {}
    for name in scoring.names:
        point = scoring.vector(name, y, score)
        if point is None:
            out[name] = Estimate.of(None, n=n, note="not computable on this part")
        elif isinstance(point, (list, tuple, dict)):
            out[name] = resampler.estimate_elements(
                list(point.values()) if isinstance(point, dict) else list(point),
                lambda idx, m=name: scoring.vector(m, at(y, idx), score[idx]))
        else:
            out[name] = resampler.estimate(
                point, lambda idx, m=name: scoring.point(m, at(y, idx), score[idx]))
    return out


# --------------------------------------------------------------------------- curves

def _svg_line(points_sets, title, xlabel, ylabel, width=420, height=300, pad=44, diagonal=False):
    colours = ("#1b3a73", "#9a3a2c", "#2e6b3f", "#7a3c77")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'role="img" aria-label="{title}"><title>{title}</title>',
           f'<rect width="{width}" height="{height}" fill="#fff"/>']
    x0, y0, x1, y1 = pad, height - pad, width - 10, 24

    def px(v):
        return x0 + (x1 - x0) * max(0.0, min(1.0, v))

    def py(v):
        return y0 + (y1 - y0) * max(0.0, min(1.0, v))

    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#ccc"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#ccc"/>')
    if diagonal:
        out.append(f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" '
                   'stroke="#bbb" stroke-dasharray="4 3"/>')
    for i, (label, pts) in enumerate(points_sets):
        if not pts:
            continue
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        colour = colours[i % len(colours)]
        out.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.8" points="{d}"/>')
        out.append(f'<text x="{x1}" y="{y1 + 12 * (i + 1)}" text-anchor="end" '
                   f'font-family="sans-serif" font-size="10" fill="{colour}">{label}</text>')
    out.append(f'<text x="{pad}" y="16" font-family="sans-serif" font-size="11" '
               f'fill="#333">{title}</text>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{height - 8}" text-anchor="middle" '
               f'font-family="sans-serif" font-size="9" fill="#777">{xlabel}</text>')
    out.append(f'<text x="10" y="{(y0 + y1) / 2:.0f}" font-family="sans-serif" font-size="9" '
               f'fill="#777" transform="rotate(-90 10 {(y0 + y1) / 2:.0f})">{ylabel}</text>')
    out.append("</svg>")
    return "\n".join(out)


def build_curves(kind, y, score, metrics_mod, threshold, out_dir, markers):
    """The curves of §3, as points and as SVG. The points are what the layer draws.

    `kind` is the effective kind the score is read as. A binary score is a logit: the curves
    rank and calibrate σ(score), always. An anomaly score is ranked on its own scale (it is no
    probability, so it has no reliability curve), and `threshold` — None when no threshold
    applies — is on that scale."""
    import numpy as np
    curves = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    s = _one_column(score)
    if y is None:
        if kind == "anomaly" and s is not None and s.ndim == 1:
            curves["scores"] = {"all": _hist(s), "x": "anomaly score", "y": "rows"}
        else:
            markers.append(f"WARN: no curve set is defined for an unlabelled {kind} part")
        return curves
    y = np.asarray(y)
    if kind == "anomaly" and s.ndim == 1 and len(np.unique(y.reshape(-1))) > 1:
        return _anomaly_curves(y.reshape(-1), s, threshold, out_dir, curves)
    if kind == "forecast":
        # one column per horizon step; H = 1 stores a (N, 1) head against a (N,) truth
        steps = np.asarray(s, dtype=np.float64).reshape(len(s), -1)
        err = np.abs(steps - np.asarray(y, dtype=np.float64).reshape(steps.shape))
        curves["per_horizon_error"] = {
            "points": [[h + 1, round(float(v), 6)] for h, v in enumerate(err.mean(axis=0))],
            "x": "horizon step", "y": "mean absolute error"}
        return curves
    y = y.reshape(-1) if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1) else y
    if kind == "binary" and s.ndim == 1 and len(np.unique(y)) > 1:
        p = metrics_mod.sigmoid(s)
        order = np.argsort(-p)
        y_sorted = y[order]
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        pos, neg = max(1, int(y.sum())), max(1, int((1 - y).sum()))
        # The output is 400 points however long the test part is, so the curves are thinned in
        # index space first and only the surviving points are ever boxed into Python floats.
        # Building the full n-point list and discarding all but 400 of it allocates three
        # objects per test row for a fixed-size result.
        ranks = np.arange(1, len(tps) + 1)
        curves["roc"] = {"points": _thin_arrays(fps / neg, tps / pos, lead=(0.0, 0.0)),
                         "x": "false positive rate", "y": "true positive rate"}
        curves["pr"] = {"points": _thin_arrays(tps / pos, tps / ranks),
                        "x": "recall", "y": "precision"}
        # the same bin count the ECE metric uses, so the curve is a picture OF the number
        # rather than a second, differently binned reading of the same calibration
        edges = np.linspace(0, 1, ECE_BINS + 1)
        rel = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p > lo) & (p <= hi)
            if m.sum():
                rel.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
        curves["reliability"] = {"points": [(a, b) for a, b, _ in rel],
                                 "counts": [c for _, _, c in rel], "bins": ECE_BINS,
                                 "x": "mean predicted", "y": "observed rate"}
        cut = 0.5 if threshold is None else float(threshold)
        pred = (p >= cut).astype(int)
        curves["confusion"] = {"threshold": cut, "matrix": [
            [int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())],
            [int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())]],
            "labels": ["negative", "positive"]}
        curves["scores"] = {"positive": _hist(p[y == 1]), "negative": _hist(p[y == 0]),
                            "x": "score", "y": "rows"}
        _write_svg(out_dir / "roc.svg", _svg_line([("ROC", curves["roc"]["points"])],
                                                  "ROC", "FPR", "TPR", diagonal=True), curves,
                   "roc")
        _write_svg(out_dir / "pr.svg", _svg_line([("PR", curves["pr"]["points"])],
                                                 "Precision–recall", "recall", "precision"),
                   curves, "pr")
        _write_svg(out_dir / "reliability.svg",
                   _svg_line([("reliability", curves["reliability"]["points"])],
                             "Calibration", "mean predicted", "observed", diagonal=True),
                   curves, "reliability")
    elif kind == "regression" and s.ndim == 1:
        pred = s.reshape(-1)
        lo, hi = float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))
        span = (hi - lo) or 1.0
        curves["predicted_vs_actual"] = {
            "points": _thin([(float((a - lo) / span), float((b - lo) / span))
                             for a, b in zip(y, pred)]), "x": "actual", "y": "predicted",
            "range": [lo, hi]}
        resid = pred - y
        curves["residuals"] = {"points": _thin([(float((a - lo) / span),
                                                 float(r)) for a, r in zip(pred, resid)]),
                               "x": "predicted", "y": "residual"}
        curves["residual_hist"] = _hist(resid)
        _write_svg(out_dir / "predicted_vs_actual.svg",
                   _svg_line([("rows", curves["predicted_vs_actual"]["points"])],
                             "Predicted vs actual", "actual", "predicted", diagonal=True),
                   curves, "predicted_vs_actual")
    elif kind in ("unsupervised",):
        err = np.mean((s - np.asarray(y)) ** 2, axis=1) if s.ndim > 1 else (s - y) ** 2
        curves["error_hist"] = _hist(err)
    else:
        markers.append(f"WARN: no curve set is defined for task kind {kind!r}")
    return curves


def _anomaly_curves(y, s, threshold, out_dir, curves):
    """ROC and PR ranked on the anomaly score's own scale, the score histograms by class, and
    the confusion at the threshold when one applies."""
    import numpy as np
    order = np.argsort(-s)
    y_sorted = y[order]
    tps, fps = np.cumsum(y_sorted), np.cumsum(1 - y_sorted)
    pos, neg = max(1, int(y.sum())), max(1, int((1 - y).sum()))
    ranks = np.arange(1, len(tps) + 1)
    curves["roc"] = {"points": _thin_arrays(fps / neg, tps / pos, lead=(0.0, 0.0)),
                     "x": "false positive rate", "y": "true positive rate"}
    curves["pr"] = {"points": _thin_arrays(tps / pos, tps / ranks),
                    "x": "recall", "y": "precision"}
    curves["scores"] = {"positive": _hist(s[y == 1]), "negative": _hist(s[y == 0]),
                        "x": "anomaly score", "y": "rows"}
    if threshold is not None:
        pred = (s >= float(threshold)).astype(int)
        curves["confusion"] = {"threshold": float(threshold), "matrix": [
            [int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())],
            [int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())]],
            "labels": ["normal", "anomalous"]}
    _write_svg(out_dir / "roc.svg", _svg_line([("ROC", curves["roc"]["points"])], "ROC", "FPR",
                                              "TPR", diagonal=True), curves, "roc")
    _write_svg(out_dir / "pr.svg", _svg_line([("PR", curves["pr"]["points"])],
                                             "Precision–recall", "recall", "precision"),
               curves, "pr")
    return curves


def _thin(points, limit=400):
    if len(points) <= limit:
        return [[round(float(a), 6), round(float(b), 6)] for a, b in points]
    step = len(points) / limit
    return [[round(float(points[int(i * step)][0]), 6),
             round(float(points[int(i * step)][1]), 6)] for i in range(limit)]


def _thin_arrays(xs, ys, limit=400, lead=None):
    """`_thin` over two numpy arrays, selecting indices before leaving the array domain.

    The index sequence is `_thin`'s exactly — `int(i * len/limit)` for i in range(limit) — so
    the recorded points are the same points, and a curve drawn before this change and one drawn
    after it are the same curve. `lead` prepends a fixed point (ROC's origin) and counts toward
    the length, as it did when it was a list element.
    """
    import numpy as np
    n = len(xs) + (1 if lead else 0)
    if n <= limit:
        out = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(xs, ys)]
        return ([[round(float(lead[0]), 6), round(float(lead[1]), 6)]] + out) if lead else out
    step = n / limit
    idx = np.array([int(i * step) for i in range(limit)])
    if lead:
        # index 0 was the lead point; every other index is one place into the arrays
        head = [[round(float(lead[0]), 6), round(float(lead[1]), 6)]] if idx[0] == 0 else []
        rest = idx[1:] - 1 if idx[0] == 0 else idx - 1
        xs_sel, ys_sel = xs[rest], ys[rest]
        return head + [[round(float(a), 6), round(float(b), 6)]
                       for a, b in zip(xs_sel, ys_sel)]
    xs_sel, ys_sel = xs[idx], ys[idx]
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(xs_sel, ys_sel)]


def _hist(values, bins=24):
    import numpy as np
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [round(float(e), 6) for e in edges],
            "counts": [int(c) for c in counts]}


def _write_svg(path, svg, curves, key):
    """Write the SVG and record where it went, as a path relative to the campaign.

    `curves.json` is spliced verbatim into the published layer, so a path recorded the way the
    invocation happened to spell it would put whoever ran `evaluate.py --campaign /home/…` into
    the model. The record names the file's place in the campaign; the campaign's own location
    is the reader's business.
    """
    path = Path(path)
    path.write_text(svg, encoding="utf-8")
    campaign = path.parent.parent.parent.parent          # <camp>/eval/<run>/curves/<file>
    try:
        curves[key]["svg"] = str(path.relative_to(campaign))
    except ValueError:                                   # not under the campaign after all
        curves[key]["svg"] = path.name


# --------------------------------------------------------------------------- slices

def build_slices(frame, spec, analysis, y, score, scoring, resampler, named, markers, *,
                 raw_test=None, not_cut=None):
    """The slices the chain named — never one invented after the look.

    The analysis's segments are cut from `raw_test`, the test part's raw rows in its order (the
    k-means read raw columns; `lens_segments`): a DataFrame, or the reason the rows could not be
    read. A slice the chain named that cannot be cut is appended to `not_cut` as `{source,
    why}` and said as a `WARN:` — never a silent omission.

    Every metric here is read through `scoring`, so a slice is measured at the same threshold
    as the headline. A slice whose labels are all one class is recorded with its `n` and marked
    `degenerate`: a ranking metric on it is not a low score, it is not a score at all. sklearn
    returns 0.0 for average precision with no positives, and reporting that as the worst slice
    would be a manufactured finding — so the ranking metrics are dropped for such a slice and
    the worst-slice ranking never sees it.
    """
    import numpy as np
    out = []
    primary, kind, names = scoring.primary, scoring.kind, scoring.names
    features = {f["name"]: f for f in (spec.get("data_contract") or {}).get("features") or []}
    y = None if y is None else np.asarray(y)
    score = np.asarray(score)

    def add(name, source, mask, note=None):
        n = int(mask.sum())
        if n < 30:
            markers.append(f"WARN: slice {name!r} has {n} test rows (< 30); it is recorded "
                           "with its n and no interval is claimed for it")
        idx = np.where(mask)[0]
        if len(idx) < 5:
            return
        degenerate = _single_class(kind, y, idx)
        if degenerate:
            markers.append(f"WARN: slice {name!r} (n = {n}) holds one class only; its ranking "
                           "metrics are undefined and are left out rather than reported as a "
                           "score of zero, and it is not eligible to be the worst slice")
            note = ((note + " ") if note else "") + ("every row in this slice carries the same "
                                                     "label, so a ranking metric has nothing "
                                                     "to rank")
        sub = Resampler(len(idx), resampler.n_boot, resampler.seed)
        est = {}
        for metric in [primary] + [m for m in names if m != primary][:4]:
            if degenerate and metric in RANKING:
                continue
            point = scoring.point(metric, at(y, idx), score[idx])
            if point is None:
                continue
            est[metric] = sub.estimate(
                point, lambda j, m=metric: scoring.point(m, at(at(y, idx), j), score[idx][j])
            ).as_json()
        out.append({"name": name, "source": source, "n": n, "metrics": est,
                    "degenerate": degenerate, "note": note})

    for why in _segment_slices(analysis, raw_test, len(frame), add, markers):
        markers.append(f"WARN: a segment slice was not cut: {why}")
        if not_cut is not None:
            not_cut.append({"source": lens_segments.SOURCE, "why": why})

    for column, feature in features.items():
        if feature.get("role") != "nominal" or column not in frame.columns:
            continue
        for value in sorted(frame[column].dropna().unique().tolist()):
            mask = (frame[column] == value).to_numpy()
            if mask.sum() >= 30:
                add(f"{column}={value}", "geometry:typing/" + column, mask)

    drift = ((analysis or {}).get("modules") or {}).get("drift") or {}
    for column in drifted_columns(drift):
        if column not in frame.columns:
            continue
        median = float(frame[column].median())
        add(f"{column}<=median", "analysis:modules/drift",
            (frame[column] <= median).to_numpy(),
            note=f"the train-median split of a column the analysis found drifting "
                 f"(median {median:.6g})")
        add(f"{column}>median", "analysis:modules/drift",
            (frame[column] > median).to_numpy())

    if ((analysis or {}).get("context") or {}).get("time"):
        n = len(frame)
        q = n // 4
        add("time:first-quartile", "analysis:context.time",
            np.array([i < q for i in range(n)]))
        add("time:last-quartile", "analysis:context.time",
            np.array([i >= n - q for i in range(n)]))

    for declared in _declared(named, frame, markers):
        column = declared.column
        if column not in frame.columns:
            markers.append(f"WARN: --slices {declared.raw!r}: no column {column!r} in the test "
                           "part's features; the slice was declared before the look and could "
                           "not be measured")
            continue
        # the literal was coerced to the column's dtype before the seal (user_slices.declare),
        # so the comparison is of values: 4 matches a code read back as 4.0
        mask = declared.mask(frame)
        # `add` drops a mask below five rows, and the user must hear about a slice they
        # declared before the look that turned out to name nothing — a marker saying the slice
        # was recorded when no row carries it is worse than no marker at all
        if int(mask.sum()) < 5:
            present = sorted({str(v) for v in frame[column].head(200).tolist()})[:8]
            markers.append(f"WARN: --slices {declared.raw!r}: {int(mask.sum())} test row(s) "
                           f"carry {column} = {declared.value!r} (the literal coerced to "
                           f"{declared.dtype}), too few to read; the column holds "
                           f"{', '.join(present)}{'…' if len(present) == 8 else ''}")
            continue
        add(declared.name, "user:turn", mask)
    return out


def _segment_slices(analysis, raw_test, n_rows, add, markers):
    """Cut one slice per segment the analysis's k-means placed test rows in (`add`), and
    return the reasons a segment slice was not cut (eval-contract §3, slices): the module did not
    run or found nothing, it recorded no assignment rule, the raw test rows could not be read, a
    segment holds fewer than five test rows."""
    import numpy as np
    rule, why = lens_segments.segment_plan(((analysis or {}).get("modules") or {}).get("segments"))
    if rule is None:
        return [why] if why else []
    if raw_test is None or isinstance(raw_test, str):
        return [raw_test or "the test part's raw rows were not available to place them"]
    if len(raw_test) != n_rows:
        return [f"{len(raw_test)} raw row(s) for {n_rows} test row(s); they are not the rows "
                "scored"]
    labels = rule.assign(raw_test)
    reasons = []
    missing = int((labels < 0).sum())
    if missing:
        markers.append(f"WARN: {missing} test row(s) miss one of the segment features "
                       f"({', '.join(rule.features)}) and belong to no segment slice")
    for seg in rule.segments:
        mask = labels == seg
        if int(mask.sum()) < 5:
            reasons.append(f"segment {seg} of {rule.k} holds {int(mask.sum())} test row(s), "
                           "too few to read")
            continue
        add(f"segment:{seg}", lens_segments.SOURCE, np.asarray(mask), note=rule.describe(seg))
    return reasons


def _declared(items, frame, markers):
    """The user slices as declared before the seal; an item still in its raw `COL=VALUE` form
    (a caller that did not declare it) is declared here against the frame's own dtypes."""
    raw = [i for i in items or () if isinstance(i, str)]
    out = [i for i in items or () if not isinstance(i, str)]
    if raw:
        declared, problems = user_slices.declare(
            raw, {str(c): str(frame[c].dtype) for c in frame.columns}, list(frame.columns))
        out += list(declared)
        markers.extend(f"WARN: {p}; the slice could not be measured" for p in problems)
    return out


def _single_class(kind, y, idx) -> bool:
    """Whether the rows `idx` of a labelled classification or anomaly part hold one class only
    (a ranking metric then has nothing to rank). An unlabelled part is never degenerate: its
    metrics read no label."""
    import numpy as np
    if y is None or (kind not in CLASSIFICATION and kind != "anomaly"):
        return False
    return len(np.unique(np.asarray(y)[idx].reshape(len(idx), -1)[:, 0])) < 2


# --------------------------------------------------------------------------- robustness

def build_robustness(frame, spec, analysis, y, score_fn, scoring, resampler, train_frame,
                     markers, base_score=None, perturbation_skipped=None):
    """Every entry a metric with its interval, and every skip said out loud.

    Each perturbed frame is scored **once** and the resamples index into that vector. Calling
    the network inside the bootstrap loop would be a thousand forward passes per check for the
    same thousand numbers, and the numbers would not differ. Every metric goes through
    `scoring`, so a robustness reading is the same reading as the headline.

    `score_fn` re-scores a perturbed copy of `frame`; None for a sample whose inputs are not one
    row of features (a window, a graph) — `base_score` is then the stored score, and the
    perturbation checks are skipped with `perturbation_skipped` as the reason, said out loud.
    """
    import numpy as np
    out = []
    primary, kind = scoring.primary, scoring.kind
    y = None if y is None else np.asarray(y)
    if score_fn is None and base_score is None:
        raise ValueError("build_robustness needs a score_fn or the base_score")
    base_score = np.asarray(score_fn(frame) if score_fn is not None else base_score)
    base_point = scoring.primary_of(y, base_score)

    def degenerate(idx):
        return _single_class(kind, y, idx)

    importance = ((analysis or {}).get("modules") or {}).get("importance") or {}
    ranked = lens_ranking(importance)
    features = {f["name"]: f for f in (spec.get("data_contract") or {}).get("features") or []}
    numeric = [c for c in ranked if features.get(c, {}).get("role") != "nominal"
               and c in frame.columns][:3]
    skipped = [c for c in ranked[:3] if c not in numeric]
    if score_fn is None:
        out.append({"check": "importance-perturbation", "status": "skipped",
                    "why": perturbation_skipped or "the inputs are not one row of features; a "
                           "one-standard-deviation shift of a column is not defined on them"})
        numeric, skipped = [], []
    elif not ranked:
        out.append({"check": "importance-perturbation", "status": "skipped",
                    "why": "the analysis recorded no importance ranking, so there is nothing "
                           "to perturb in the order it claimed mattered"})
    for column in numeric:
        sd = float(train_frame[column].std()) if train_frame is not None else \
            float(frame[column].std())
        shifted = frame.copy()
        shifted[column] = shifted[column] + sd
        shifted_score = score_fn(shifted)
        point = scoring.primary_of(y, shifted_score)
        if point is None:
            out.append({"check": f"perturb:{column}", "status": "skipped",
                        "why": f"{primary} is not computable on the perturbed copy"})
            continue
        delta = resampler.paired_difference(
            lambda idx, sc=shifted_score: scoring.primary_of(at(y, idx), sc[idx]),
            lambda idx: scoring.primary_of(at(y, idx), base_score[idx]),
            point, base_point)
        out.append({"check": f"perturb:{column}", "source": "analysis:modules/importance",
                    "shift": f"+1 train sd ({sd:.6g})",
                    "primary": resampler.estimate(
                        point,
                        lambda idx, sc=shifted_score: scoring.primary_of(
                            at(y, idx), sc[idx])).as_json(),
                    "delta": delta.as_json()})
    for column in skipped:
        out.append({"check": f"perturb:{column}", "status": "skipped",
                    "why": "a nominal feature, or a column the test part does not carry; a "
                           "one-standard-deviation shift is not defined for it"})

    drift = ((analysis or {}).get("modules") or {}).get("drift") or {}
    drifted = drifted_columns(drift)
    for column in drifted[:2]:
        if column not in frame.columns:
            continue
        median = float(frame[column].median())
        for label, mask in ((f"{column} stable (<= median)",
                             (frame[column] <= median).to_numpy()),
                            (f"{column} drifted (> median)",
                             (frame[column] > median).to_numpy())):
            idx = np.where(mask)[0]
            if len(idx) < 10:
                continue
            if degenerate(idx):
                markers.append(f"WARN: robustness check {label!r} (n = {len(idx)}) holds one "
                               f"class only; {primary} is undefined on it and the check is "
                               "recorded as skipped rather than as a score of zero")
                out.append({"check": label, "source": "analysis:modules/drift",
                            "status": "skipped", "n": int(len(idx)),
                            "why": f"every row on this side of the split carries the same "
                                   f"label, so {primary} has nothing to rank"})
                continue
            point = scoring.primary_of(at(y, idx), base_score[idx])
            if point is None:
                continue
            sub = Resampler(len(idx), resampler.n_boot, resampler.seed)
            out.append({"check": label, "source": "analysis:modules/drift",
                        "n": int(len(idx)),
                        "primary": sub.estimate(
                            point, lambda j, i=idx: scoring.primary_of(
                                at(at(y, i), j), base_score[i][j])).as_json()})
    if not drifted:
        out.append({"check": "drift-slices", "status": "skipped",
                    "why": "the analysis measured no drift, so there is no drifted/stable "
                           "split to read the primary on"})
    if not ((analysis or {}).get("context") or {}).get("time"):
        out.append({"check": "time-quartiles", "status": "skipped",
                    "why": "the analysis records no time context"})
    return out


# --------------------------------------------------------------------------- the run

def evaluate(args, out=print):
    markers = []
    camp_dir = Path(args.campaign)
    docs, err = load_campaign(camp_dir)
    if err:
        out(f"ERROR: {err}")
        return 2
    campaign, spec, ledger = docs["campaign"], docs["spec"], docs["experiments"]
    run_id = args.run or (ledger.get("selection") or {}).get("run")
    if not run_id:
        out("ERROR: no --run and the ledger records no selection; there is nothing to evaluate")
        return 2

    try:
        resolved = resolved_spec(camp_dir, run_id, spec)
    except (OSError, json.JSONDecodeError) as exc:
        out(f"ERROR: runs/{run_id}/netspec.resolved.json cannot be read ({exc})")
        return 2
    baselines = [b.strip() for b in (args.baselines or ",".join(default_baselines(resolved)))
                 .split(",") if b.strip()]
    seal_path = camp_dir / "test_seal.json"

    # the user slices are declared against the shaped output's dtypes — the shaper's manifest,
    # then the train part's schema; read here as records, the test part untouched
    early, _ = read_json((spec.get("input") or {}).get("manifest"), "the shaped manifest")
    train_entry = (((early or {}).get("outputs") or {}).get("files") or {}).get("train") or {}
    said = len(markers)
    errs, pre = pre_seal_checks(
        args, campaign, ledger, run_id, seal_path, baselines, markers,
        dtypes=user_slices.dtypes_of(early, train_entry.get("path")),
        features=[str(f.get("name")) for f in
                  ((resolved.get("data_contract") or {}).get("features") or [])
                  if isinstance(f, Mapping) and f.get("name")])
    for marker in markers[said:]:
        out(marker)
    if errs:
        for e in errs:
            out(f"ERROR: {e}")
        out("ERROR: refused before the seal was written — the test part has not been touched")
        return 2
    prior = pre.prior

    row = next(r for r in ledger["runs"] if r.get("id") == run_id)
    run_dir = camp_dir / "runs" / run_id
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        out(f"ERROR: {checkpoint} does not exist; the selected run has no checkpoint to "
            "evaluate")
        return 2
    got = digest_file(checkpoint)
    if row.get("checkpoint_digest") and got != row["checkpoint_digest"]:
        out(f"ERROR: {checkpoint}'s digest is {got} but the ledger records "
            f"{row['checkpoint_digest']}; the checkpoint changed since the run, and a metric "
            "computed against the wrong weights is worse than no metric")
        return 2

    manifest, merr = read_json((spec.get("input") or {}).get("manifest"), "the shaped manifest")
    if merr:
        out(f"ERROR: {merr}; the test part's address comes from the manifest, never the spec")
        return 2
    test_entry = ((manifest.get("outputs") or {}).get("files") or {}).get("test") or {}
    test_path = test_entry.get("path")
    if not test_path or not Path(test_path).is_file():
        out(f"ERROR: the manifest names the test part at {test_path!r}, which does not exist")
        return 2

    # the package is opened before the seal: one this stage cannot drive (an older generation,
    # no Evaluator port, a map that does not import) must not cost a look at the test part
    package, perr = open_package(camp_dir, spec, run_id, resolved,
                                 Path(getattr(args, "vega_forge_dir", None) or FORGE_DIR))
    if perr:
        out(f"ERROR: {perr}")
        out("ERROR: refused before the seal was written — the test part has not been touched")
        return 2
    try:
        effective = package.domain.Task.from_spec(package.domain.NetSpec.from_dict(resolved)) \
            .effective.value
    except (KeyError, TypeError, ValueError) as exc:
        out(f"ERROR: the resolved spec's task cannot be read ({exc})")
        return 2
    terr = check_threshold_arg(args.threshold, effective)
    if terr:
        out(f"ERROR: {terr}")
        out("ERROR: refused before the seal was written — the test part has not been touched")
        return 2

    gate = "unattended" if args.unattended else "user"
    earlier = pre.earlier
    reason = args.again or ("first evaluation" if earlier is None else
                            f"first look in an after-eval campaign; {earlier.previous_campaign} "
                            f"opened this test part {earlier.exposures} time(s) before it")
    try:
        seal = (prior.again(run_id, args.again, gate) if prior
                else Seal.create(camp_dir, run_id, test_path, test_entry.get("rows"), reason,
                                 gate, after_eval=earlier.record() if earlier else None))
    except SealError as exc:
        out(f"ERROR: {exc}")
        return 2

    # ---- the seal is written BEFORE the first byte of the test part is read
    seal.write(seal_path)
    out_ok(f"test_seal.json written — run {run_id}, exposure {seal.exposures}, gate {gate}"
           + (f", reason: {args.again}" if args.again else ""), markers)
    if seal.exposures > 1:
        out_warn(f"the test part has now been seen {seal.exposures} times"
                 + (f" ({seal.after_eval['exposures']} of them in the previous campaign "
                    f"{seal.after_eval['previous_campaign']})" if seal.after_eval else "")
                 + "; every number below carries that, and the layer, the notebook and the "
                 "summary all say so", markers)

    # Three states, not two. A manifest that recorded no digest for the test part is not a
    # passed check — and the seal outlives this run, so recording `true` there would leave a
    # claim on the record that nobody can re-derive. `Seal.create` already writes None for
    # "not checked"; that is what the third state is for.
    digest = digest_file(test_path)
    expected = test_entry.get("digest")
    matches = None if not expected else (digest == expected)
    seal = seal.with_digest(digest, matches)
    seal.write(seal_path)
    if matches is False:
        out(f"ERROR: the test part's digest is {digest} but the shaper's manifest recorded "
            f"{expected}; a changed test file is a new chain, not a new number")
        return 3
    if matches is None:
        out_warn(f"the shaper's manifest records no digest for the test part, so this file "
                 f"({digest[:19]}…) could not be checked against it; the seal records the "
                 "digest it found and says the comparison was not made", markers)
    else:
        out_ok(f"the test part's digest matches the shaper's manifest ({digest[:19]}…)",
               markers)

    campaign["test_seal"] = str(seal_path)
    (camp_dir / "campaign.json").write_text(
        json.dumps(campaign, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    try:
        return _evaluate_after_seal(args, package, camp_dir, spec, ledger, row, run_id, seal,
                                    seal_path, baselines, markers, out, pre.user_slices)
    except BrokenPipeError:
        # `evaluate.py … | head` closes stdout mid-run. That is the reader leaving, not the
        # evaluation failing, and reporting it as a post-seal failure would be a lie on the
        # record. The records under eval/ are written by the same call and stand.
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        out(f"ERROR: the evaluation failed after the seal was written: "
            f"{type(exc).__name__}: {exc}")
        out("ERROR: the seal stands and the exposure is recorded — a failed evaluation does "
            "not un-see the test part")
        (camp_dir / "eval" / run_id).mkdir(parents=True, exist_ok=True)
        (camp_dir / "eval" / run_id / "error.txt").write_text(traceback.format_exc(),
                                                              encoding="utf-8")
        return 3


def class_support(y, classes) -> dict | None:
    """The sealed test part's count of each class (a multilabel head: each label's positives),
    by class name: what each per-class reading rests on, so a per-class score of a class with a
    handful of rows is read as the anecdote it is. None for a reading without classes or a part
    without truth."""
    import numpy as np
    if y is None or not classes:
        return None
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == len(classes):
        counts = np.nansum(arr > 0.5, axis=0)
    elif arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
        flat = arr.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if len(flat) and (flat.min() < 0 or flat.max() >= len(classes) or np.any(flat % 1)):
            return None                                   # not class indices
        counts = np.bincount(flat.astype(np.int64), minlength=len(classes))[:len(classes)]
    else:
        return None
    return {str(c): int(n) for c, n in zip(classes, counts)}


# --------------------------------------------------------------------------- the one look
#
# Everything below the seal is a pipeline, not a set of rules: each step needs what the step
# before it produced, and the order is the contract's. What the steps share is a `Look` — the
# one opening of the test part, frozen the moment it is complete — so a step takes the look and
# what the earlier steps returned, rather than thirteen arguments in an order nothing states.
@dataclass(frozen=True)
class Parts:
    """The three parts as this run read them, and what the network gave them.

    The test part is here because it was opened once, under the seal; nothing downstream reopens
    it, and every later number is arithmetic on these arrays. `scored` and `scored_valid` are
    the package Evaluator's `ScoredPart`s (the score per the score contract, the truth keeping
    its dimension — None for an unlabelled part —, the row ids, the unscored samples, the per-row
    score); `head` is the network's raw head on the test part, what the predictions store.
    """

    test: Any
    train: Any
    valid: Any
    scored: Any
    scored_valid: Any
    head: Any
    frame_test: Any
    frame_train: Any

    @property
    def y(self):
        return _numpy(self.scored.truth)

    @property
    def score_net(self):
        return _numpy(self.scored.score)

    @property
    def y_valid(self):
        return _numpy(self.scored_valid.truth)

    @property
    def score_valid(self):
        return _numpy(self.scored_valid.score)

    @property
    def n_rows(self) -> int:
        return len(self.scored)


@dataclass(frozen=True)
class Look:
    """One look at the test part: what it opened, how it reads a metric, and where it writes.

    Frozen because none of it changes after the look — and because the alternative, threading
    these through five functions as positional arguments, is what let the chosen threshold go
    missing from the slice metrics without a single call site looking wrong.
    """

    camp_dir: Path
    run_id: str
    eval_dir: Path
    package: Package
    evaluator: Any                     # the package's Evaluator, at this run's MetricOptions
    spec_obj: Any
    resolved: Mapping
    model: Any
    parts: Parts
    scoring: Scoring
    resampler: Resampler
    threshold: ThresholdChoice
    analysis: Mapping | None
    shape: str                         # row | window | graph
    graph: Mapping | None = None       # the graph exposure record, for a graph sample
    gates: Mapping | None = None       # {metric: GateReading}: the data-dependent gates (§3)
    # how the network's threshold was chosen (the request, the classes, the anomaly rate): each
    # baseline's own threshold is chosen by the same rule on its own valid scores (§2)
    threshold_rule: Mapping | None = None
    manifest: str | None = None        # the shaper's manifest (the raw rows a segment reads)

    @property
    def adapters(self):
        return self.package.adapters

    @property
    def kind(self):
        return self.scoring.kind

    @property
    def effective(self):
        return self.scoring.task

    @property
    def primary(self):
        return self.scoring.primary

    @property
    def mode(self):
        return self.scoring.mode

    def score_frame(self, frame):
        """Re-score a perturbed copy of the test features, without touching the sealed file:
        the rows already read under the seal, as a part, through the Evaluator's one path."""
        return _score_frame(self, frame)


@dataclass(frozen=True)
class Measured:
    """Every model this run scored, and the record each baseline earned."""

    per_model: Mapping
    scorers: Mapping
    baselines: tuple
    scorings: Mapping = field(default_factory=dict)   # a baseline's Scoring at its own threshold

    @property
    def fitted(self) -> list:
        """The baselines fitted here: the verdict's candidates for the strongest."""
        return [b["name"] for b in self.baselines if b.get("fitted")]

    @property
    def compared(self) -> list:
        """Every model the network gets a paired difference against: the fitted baselines and
        the campaign's own R0 when it was predicted through the seal (a comparison row, never
        the verdict's strongest fitted baseline)."""
        return self.fitted + [b["name"] for b in self.baselines
                              if not b.get("fitted") and b.get("predicted")]


@dataclass(frozen=True)
class Comparison:
    """The paired differences and the machine verdict read off them."""

    differences: Mapping
    verdict: Mapping


@dataclass(frozen=True)
class Description:
    """What the run has to say beyond the headline: curves, slices, robustness."""

    curves: Mapping
    slices: tuple
    robustness: tuple
    worst: tuple | None
    not_cut: tuple = ()              # the slices the chain named that were not cut, and why


def _open_the_test_part(args, package, camp_dir, spec, run_id, seal, seal_path,
                        markers) -> Look:
    """Step 1 — open the sealed part once, score everything, and freeze what was found.

    This is the only function in the stage that touches the test file, and it does so through
    the package's one door (`ShapedParts.load_sealed_part`, with the seal's token). Every score
    comes from the package Evaluator's `scored` — the one scoring path — and the raw head from
    its `heads`. It ends by building the `Scoring` at the threshold chosen on **valid**, so
    every metric anywhere downstream is read the same way.
    """
    adapters, dom = package.adapters, package.domain
    resolved = package.resolved
    spec_obj = dom.NetSpec.from_dict(resolved)
    task = dom.Task.from_spec(spec_obj)
    kind, effective = task.kind.value, task.effective.value
    primary = (resolved.get("metrics") or {}).get("primary")
    mode = ((resolved.get("metrics") or {}).get("selection") or {}).get("mode", "max")
    names = [primary] + [m for m in SECONDARY.get(kind, []) if m != primary]
    shape = dom.SampleSpec.from_spec(spec_obj).shape.value

    eval_dir = camp_dir / "eval" / run_id
    eval_dir.mkdir(parents=True, exist_ok=True)

    model = adapters.store.load(camp_dir / "runs" / run_id / "checkpoints" / "best.pt",
                                spec=spec_obj, build=adapters.build,
                                run_dir=str(package.directory.parent))
    # a graph package's DataSource carries `exposure` (what training saw of the test part);
    # another shape's carries no such method (its port is the shape's, SOLID-I)
    graph = adapters.data.exposure(spec_obj) if shape == "graph" else None
    if graph is not None:
        graph = graph.as_dict()
        seal = seal.with_graph(graph)
        seal.write(seal_path)
        declared = sample_of(resolved).get("exposure")
        if isinstance(declared, Mapping):
            # the disclosure the spec declared for a transductive graph ({kind, rows, source}),
            # carried beside what the training view measured, so the layer states both
            graph = dict(graph, exposure=dict(declared))
    test = adapters.data.load_sealed_part(spec_obj, seal.token, str(seal_path))
    out_ok(f"the test part opened once through load_sealed_part — {len(test)} {shape} "
           "sample(s)", markers)
    if graph is not None:
        out_ok(f"graph protocol {graph['protocol']}: {graph['edges_into_test']} edge(s) into "
               "the test part in the training graph, test features "
               f"{'seen' if graph['features_seen_in_training'] else 'not seen'} in training — "
               "recorded in the seal and metrics.json", markers)

    base = adapters.evaluator
    scored = base.scored(spec_obj, test, model)
    head = _numpy(base.heads(spec_obj, test, model))
    train = adapters.data.load_part(spec_obj, dom.Part.TRAIN)
    valid = adapters.data.load_part(spec_obj, dom.Part.VALID)
    scored_valid = base.scored(spec_obj, valid, model)
    if scored.unscored:
        out_ok(f"{len(scored.unscored)} test sample(s) could not be scored (too little context, "
               "or a horizon past the part) — counted in unscored.json, never given a "
               "prediction", markers)

    # the metric gates that depend on the data (§3: mape when |y| > 0.1 on 99 % of the rows) are
    # decided on the train part's targets — never on the test part's — and said either way
    gates = metric_gates(kind, names, _numpy(getattr(train, "y", None)))
    for name, gate in gates.items():
        if gate.applies:
            out_ok(f"{name} is reported: {gate.why} (decided on the train part)", markers)
        elif name == primary:
            out_warn(f"{name} is the spec's primary and its gate would withhold it: {gate.why}; "
                     "it is reported because the spec names it primary, and read with that "
                     "caveat", markers)
        else:
            out_warn(f"{name} is withheld: {gate.why} (decided on the train part); every "
                     "model's record says so instead of carrying the number", markers)
    names = [n for n in names if n == primary or n not in gates or gates[n].applies]

    parts = Parts(test=test, train=train, valid=valid, scored=scored,
                  scored_valid=scored_valid, head=head,
                  frame_test=sample_frame(dom, adapters.data, spec_obj, test),
                  frame_train=(sample_frame(dom, adapters.data, spec_obj, train)
                               if shape == "row" else None))

    resampler = Resampler(parts.n_rows, args.n_boot, args.boot_seed)
    out_ok(f"{args.n_boot} bootstrap resamples at seed {args.boot_seed}, shared by every model "
           f"so the paired differences have intervals of their own ({CI_LEVEL:.0%} intervals)",
           markers)

    classes = task_classes(task)
    options = metric_options(dom, adapters.data, spec_obj, task, names)
    scoring = Scoring(adapters.metrics, kind, primary, mode, tuple(names), options, effective)
    rate, rate_source = (anomaly_rate(train, resolved) if effective == "anomaly"
                         else (None, None))
    threshold = choose_threshold(scoring, parts.y_valid, parts.score_valid, args.threshold,
                                 effective=effective, classes=classes, rate=rate,
                                 rate_source=rate_source, applies=kind in THRESHOLD_TASKS)
    if threshold.applies:
        # from here on there is one way to read a metric, and it carries the threshold. The
        # slices and the robustness checks take this same object, so they cannot be read at a
        # different threshold from the headline without someone building a second one.
        scoring = scoring.at_threshold(threshold.metric_threshold(classes))
        shown = (f"{threshold.value:.6g}" + (f" (per label: {len(threshold.per_label)})"
                                             if threshold.per_label else ""))
        out_ok(f"decision threshold {shown} on the {threshold.scale} scale from "
               f"{threshold.source} — {threshold.why}, and every metric below — the slices and "
               "the robustness checks included — is read at it", markers)
    elif kind in THRESHOLD_TASKS:
        out_warn(f"no decision threshold: {threshold.why}", markers)

    return Look(camp_dir=camp_dir, run_id=run_id, eval_dir=eval_dir, package=package,
                evaluator=base.with_options(scoring.options), spec_obj=spec_obj,
                resolved=resolved, model=model, parts=parts, scoring=scoring,
                resampler=resampler, threshold=threshold, analysis=_analysis(camp_dir, spec),
                threshold_rule={"requested": args.threshold, "classes": classes,
                                "rate": rate, "rate_source": rate_source},
                shape=shape, graph=graph, gates=gates,
                manifest=(spec.get("input") or {}).get("manifest"))


def _agree_with_the_port(look, per_model, markers) -> None:
    """The network's point metrics, read here through `Scoring`, against the package
    Evaluator's own `measure` of the same scored part at the same options: the one scoring path
    and this stage's bootstrap must be one reading. A disagreement is said, not smoothed."""
    dom = look.package.domain
    try:
        port = look.evaluator.measure(look.spec_obj, dom.Part.TEST, look.parts.scored).as_dict()
    except Exception as exc:  # noqa: BLE001
        out_warn(f"the package Evaluator could not measure the scored test part "
                 f"({type(exc).__name__}: {exc}); the numbers below are this stage's reading "
                 "alone", markers)
        return
    compared, differ = 0, []
    for name, estimate in per_model.items():
        mine, theirs = estimate.value, port.get(name)
        if mine is None or not isinstance(theirs, (int, float)) or isinstance(theirs, bool):
            continue
        compared += 1
        if abs(float(theirs) - float(mine)) > 1e-4 + 1e-9:     # the record rounds to 4 places
            differ.append(f"{name} {mine} vs {theirs:.6g}")
    if differ:
        out_warn("the package Evaluator's measure and this stage's reading disagree on "
                 + "; ".join(differ) + " — one scoring path should give one number", markers)
    elif compared:
        out_ok(f"the package Evaluator's own measure agrees on all {compared} metric(s) both "
               "compute — one scoring path, one reading", markers)


def _built_baselines(look, which, markers):
    """The baselines /vega-eval asked for, built by the package's baseline module with the
    context the spec states; (baselines, [records of the ones that could not be built])."""
    dom, adapters = look.package.domain, look.adapters
    wanted = [b for b in which if b != "campaign-baseline"]
    if not wanted:
        return [], []
    if adapters.baselines is None:
        why = ("the package's blueprint map gives no file the baseline role, so no baseline "
               "can be fitted")
        for name in wanted:
            out_warn(f"baseline {name!r} was not fitted: {why}", markers)
        return [], [{"name": n, "fitted": False, "why": why, "valid": None, "test": None}
                    for n in wanted]
    sample = dom.SampleSpec.from_spec(look.spec_obj)
    window = sample.window
    context = dom.BaselineContext(
        z=bottleneck(look.resolved, look.effective),
        # the protocol lens re-fits: the family and the recorded hyper-parameters (§4)
        protocol=lens_protocol(((look.analysis or {}).get("modules") or {}).get("importance")),
        season=(int(window.season) if window is not None and window.season else None),
        horizon=(int(window.horizon) if window is not None and window.horizon else 1))
    return adapters.baselines.build(look.spec_obj, wanted, context), []


def _fit_and_score(look, baseline):
    """(test ScoredPart, valid ScoredPart, None, the fit's outcome) of a baseline fitted on
    the train part's design in its own view, or (None, None, why it could not be, the outcome
    or the recipe) — a baseline the environment or the sample cannot fit keeps its row and says
    why. A fit changes nothing on the recipe: what it gave (a fitted baseline, or a refusal)
    is what is scored and what the row records."""
    adapters = look.adapters
    try:
        outcome = baseline.fit(baseline.design_of(adapters.data, look.parts.train))
    except Exception as exc:  # noqa: BLE001
        return None, None, f"fitting failed ({type(exc).__name__}: {exc})", baseline
    if not outcome.fitted:
        return None, None, outcome.why or "not fitted", outcome
    try:
        return (look.evaluator.scored(look.spec_obj, look.parts.test, outcome),
                look.evaluator.scored(look.spec_obj, look.parts.valid, outcome), None, outcome)
    except Exception as exc:  # noqa: BLE001
        return (None, None, f"fitted, but could not be scored ({type(exc).__name__}: {exc})",
                outcome)


def _measure(look, args, ledger, baselines, markers) -> Measured:
    """Step 2 — the network and every baseline, on the same rows and the same resamples.

    The baselines are fitted here and nowhere else: on the train part's design matrix in each
    baseline's own view (`design_of`: flat rows, a flattened window, a node's own features or
    its neighbours' mean, pooled graphs), on exactly the shaped features the network saw, and
    scored through the same Evaluator path as the network. A baseline the environment cannot fit
    keeps its row and says why, because a baseline that quietly disappeared reads as one the
    network beat.
    """
    parts, scoring, resampler = look.parts, look.scoring, look.resampler
    primary = look.primary
    per_model = {"network": _estimates(look, parts.score_net)}
    _agree_with_the_port(look, per_model["network"], markers)
    scorers, scorings = {"network": parts.score_net}, {}
    made, records = _built_baselines(look, baselines, markers)
    y_valid = parts.y_valid

    for baseline in made:
        test_scored, valid_scored, why, outcome = _fit_and_score(look, baseline)
        record = dict(outcome.record(), **_lens_fields(outcome, look, markers))
        if why is not None:
            out_warn(f"baseline {record['name']!r} was not fitted: {why} — the row says so and "
                     "is never silently dropped", markers)
            records.append(dict(record, fitted=False, why=why, valid=None, test=None))
            continue
        s_test, s_valid = _numpy(test_scored.score), _numpy(valid_scored.score)
        own, own_threshold = _own_threshold(look, s_valid)
        scorers[record["name"]], scorings[record["name"]] = s_test, own
        per_model[record["name"]] = _estimates(look, s_test, own)
        valid_sub = Resampler(len(s_valid), args.n_boot, args.boot_seed)
        valid_point = own.primary_of(y_valid, s_valid)
        records.append({
            "name": record["name"], "fitted": True,
            "definition": _definition(record["name"], look.kind, look),
            "view": record.get("view"),
            "protocol": "fitted on the train part, selected on valid, on exactly the shaped "
                        "features the network saw",
            "valid": valid_sub.estimate(
                valid_point,
                lambda idx, s=s_valid, o=own: o.primary_of(at(y_valid, idx), s[idx])).as_json(),
            "test": per_model[record["name"]][primary].as_json()})
        if own_threshold is not None:
            records[-1]["decision_threshold"] = own_threshold
        for key in ("delegate", "family", "hyper_parameters", "protocol_source"):
            if record.get(key) is not None:
                records[-1][key] = record[key]
        at_own = (f" at its own valid-chosen threshold {_g(own_threshold['at'])}"
                  f"{' (per label)' if own_threshold['per_label'] else ''}"
                  if own_threshold is not None and own_threshold.get("at") is not None else "")
        out_ok(f"baseline {record['name']}: valid {primary} "
               f"{_g(records[-1]['valid']['value'])}, test "
               f"{_g(records[-1]['test']['value'])}{at_own}", markers)

    _note_lens_twin(records, markers)
    _add_campaign_baseline(look, records, per_model, scorers, ledger, baselines, args, markers)
    return Measured(per_model=per_model, scorers=scorers, baselines=tuple(records),
                    scorings=scorings)


def _estimates(look, score, scoring=None):
    """Every metric of the task's row for one model's test score, on the shared resamples —
    and every metric a gate withheld (§3, decided on the train part), recorded with its reason
    in its place rather than missing. `scoring` is a baseline's own (its own threshold)."""
    out = estimates_for(look.resampler, scoring or look.scoring, look.parts.y, score)
    for name, gate in (look.gates or {}).items():
        if not gate.applies and name not in out:
            out[name] = Estimate.of(None, n=look.parts.n_rows, note=gate.note)
    return out


def _own_threshold(look, s_valid):
    """(the Scoring a baseline is read with, its `decision_threshold` record or None): a
    baseline's threshold is chosen on **its own** valid scores by the rule the network's was (the
    same request — valid-optimum, 0.5 or the user's first-exposure value —, the classes and the
    anomaly rate), never read at the network's: a threshold chosen for one model handicaps every
    other read at it (eval-contract §2). A reading without a threshold keeps the run's Scoring."""
    if not look.threshold.applies:
        return look.scoring, None
    rule = look.threshold_rule or {}
    classes = rule.get("classes") or ()
    own = choose_threshold(look.scoring, look.parts.y_valid, s_valid, rule.get("requested"),
                           effective=look.effective, classes=classes, rate=rule.get("rate"),
                           rate_source=rule.get("rate_source"))
    if not own.applies:
        return look.scoring, None
    return (look.scoring.at_threshold(own.metric_threshold(classes)),
            {"at": own.value, "per_label": dict(own.per_label) if own.per_label else None,
             "scale": own.scale, "source": own.source, "why": own.why})


def _lens_fields(baseline, look, markers):
    """The lens row's protocol as fitted: the family the analysis named, the hyper-parameters
    applied and those not carried (with why), and which record the protocol came from. A
    recorded hyper-parameter that did not carry is said, never silently dropped."""
    if baseline.record().get("name") != "lens":
        return {}
    record = baseline.record()
    protocol = lens_protocol(((look.analysis or {}).get("modules") or {}).get("importance"))
    hyper = record.get("hyper_parameters")
    for key, why in ((hyper or {}).get("not_carried") or {}).items():
        out_warn(f"baseline lens: the analysis's hyper-parameter {key!r} was not carried — "
                 f"{why}", markers)
    if protocol and protocol.get("hyper_parameters") and hyper is None:
        out_warn("baseline lens: the emitted baselines module does not record which of the "
                 "analysis's hyper-parameters it applied (re-emit the package with /vega-forge)",
                 markers)
    return {"family": record.get("family") or (protocol or {}).get("family"),
            "hyper_parameters": hyper,
            "protocol_source": (protocol or {}).get("source")}


def _note_lens_twin(records, markers):
    """A lens that re-fitted the same algorithm as another baseline is not a second opinion."""
    for record in records:
        if record["name"] != "lens" or not record.get("fitted"):
            continue
        twin = next((b["name"] for b in records
                     if b["name"] not in ("lens",) and b.get("fitted")
                     and b["test"] and record["test"]
                     and b["test"]["value"] == record["test"]["value"]), None)
        if twin:
            record["note"] = (f"the analysis's protocol re-fits to the same algorithm as "
                              f"{twin}, and the two scored identically — this row is a check "
                              f"that /data-lens and this stage agree, not an independent "
                              f"baseline to beat")
            out_warn(f"baseline lens re-fitted the same algorithm as {twin} and scored "
                     "identically; it is recorded as a check, not as a second opinion",
                     markers)


def _add_campaign_baseline(look, records, per_model, scorers, ledger, baselines, args,
                           markers):
    """A comparison row, not a fitted baseline, and labelled so (eval-contract §4): R0's valid
    number is the ledger's (upstream, no new interval), and its **test** number is measured here
    — R0's stored best.pt predicted through the seal, on the rows and the resamples every other
    model was read on — so it carries a CI and a paired difference against the network."""
    if "campaign-baseline" not in baselines:
        return
    r0 = next((r for r in ledger.get("runs") or [] if r.get("id") == campaign_baseline.R0),
              None)
    row = {"name": "campaign-baseline", "fitted": False, "predicted": False,
           "definition": _definition("campaign-baseline", look.kind, look),
           "protocol": "not fitted here: R0's stored best.pt, predicted through the seal on the "
                       "test rows the network was scored on; its valid number is the ledger's",
           "valid": Estimate.upstream_value(
               (r0 or {}).get("primary"),
               "the ledger's valid metric for R0; this stage measured nothing new for it"
           ).as_json(), "test": None, "_upstream": True}
    if look.run_id == campaign_baseline.R0:
        score, why = look.parts.score_net, None
        row["note"] = ("the evaluated run is R0 itself: the comparison row is the network, and "
                       "the paired difference is zero by construction")
    else:
        score, why = campaign_baseline.predict(
            look.camp_dir, look.camp_dir / "test_seal.json", ledger,
            look.parts.scored.row_ids,
            Path(getattr(args, "vega_forge_dir", None) or FORGE_DIR))
    if why is not None:
        out_warn(f"campaign-baseline was not predicted: {why} — the row keeps the ledger's "
                 "valid number and says why it has no test number", markers)
        records.append(dict(row, why=why))
        return
    scorers["campaign-baseline"] = score
    per_model["campaign-baseline"] = _estimates(look, score)
    row.update(predicted=True,
               test=per_model["campaign-baseline"][look.primary].as_json())
    row.setdefault("note", "R0, the spec as engineered: its valid number is the ledger's, and "
                           "its test number was measured here, through the seal, with an "
                           "interval like every other model's")
    records.append(row)
    out_ok(f"campaign-baseline (R0) predicted through the seal: test {look.primary} "
           f"{_g(row['test']['value'])} {row['test'].get('ci')} — a comparison row, not a "
           "fitted baseline; its valid number is the ledger's", markers)


def not_assessable(effective, primary):
    """Why no machine verdict can be read for this task and primary, or None when a paired
    verdict is a finding (eval-contract §5): a primary that lives on each model's own score
    scale (SCALE_BOUND_PRIMARIES) is not comparable across models without a label."""
    return NOT_ASSESSABLE if primary in SCALE_BOUND_PRIMARIES.get(effective, ()) else None


def unassessed_verdict(reason):
    """The machine verdict that is not assessable: `earned` null — neither a win nor a loss —,
    no baseline named, no difference and no interval, and the reason on the record."""
    return {"earned": None, "strongest_baseline": None, "diff": None, "ci": [],
            "beaten_by_baseline": None, "reason": reason}


def _compare(look, measured, markers) -> Comparison:
    """Step 3 — the paired differences, and the verdict read off the strongest one's interval.

    The verdict is a reading of an interval, not of two point estimates side by side. That is
    the whole reason the resamples are shared. A primary that lives on each model's own scale
    (`not_assessable`) has no interval worth reading: the verdict is then not assessable and no
    difference is computed.
    """
    resampler, scoring, parts = look.resampler, look.scoring, look.parts
    primary, mode = look.primary, look.mode
    y, score_net = parts.y, parts.score_net

    reason = not_assessable(look.effective, primary)
    if reason is not None:
        out_ok(f"no machine verdict: the verdict is not assessable — {reason}", markers)
        return Comparison(differences={}, verdict=unassessed_verdict(reason))

    # The network appears in every comparison, so it is scored on the draws once rather than
    # once per baseline: with the default 1000 resamples and four baselines that is 3000
    # full-vector metric evaluations of the same numbers.
    net_draws = resampler.per_draw(lambda idx: scoring.primary_of(at(y, idx), score_net[idx]))
    differences = {}
    for name in measured.compared:
        # each baseline at its own threshold (§2); the campaign's R0 at the network's
        s_other, own = measured.scorers[name], measured.scorings.get(name, scoring)
        differences[name] = resampler.difference_of(
            net_draws,
            resampler.per_draw(lambda idx, s=s_other, o=own: o.primary_of(at(y, idx), s[idx])),
            measured.per_model["network"][primary].value,
            measured.per_model[name][primary].value).as_json()

    scored = [(b, measured.per_model[b][primary].value) for b in measured.fitted
              if measured.per_model[b][primary].value is not None]
    if not scored:
        out_warn("no baseline could be fitted; the verdict has nothing to compare against and "
                 "says so", markers)
        return Comparison(differences=differences,
                          verdict={"earned": False, "strongest_baseline": None, "diff": None,
                                   "ci": []})

    strongest = (max if mode == "max" else min)(scored, key=lambda p: p[1])[0]
    diff = differences[strongest]
    estimate = Estimate.from_json(diff)
    earned = estimate.better_than_zero(mode)
    # Three outcomes, not two. An interval that excludes zero *against* the network is not the
    # same finding as one that covers zero: the first says the baseline is better and names
    # what to ship instead, the second says this test part cannot tell them apart. Reporting
    # both as "does not exclude 0" throws away the most actionable result an evaluation can
    # produce, and in the case that matters most it is also simply untrue.
    if earned:
        reading = "earned"
    elif estimate.excludes_zero:
        reading = (f"NOT earned, and worse: the interval excludes 0 in {strongest}'s favour, so "
                   f"this part says {strongest} beats the network rather than that it cannot "
                   "tell them apart")
    else:
        reading = ("NOT earned: the interval covers 0, so this part cannot separate the "
                   "network from the baseline")
    out_ok(f"the strongest fitted baseline is {strongest}; the network's paired difference "
           f"is {_g(diff.get('value'))} {diff.get('ci')} — {reading}", markers)
    return Comparison(differences=differences,
                      verdict={"earned": bool(earned), "strongest_baseline": strongest,
                               "diff": diff.get("value"), "ci": diff.get("ci") or [],
                               # the layer and the card need to tell "worse" from "no
                               # difference"; `earned` alone cannot say it
                               "beaten_by_baseline": bool(not earned
                                                          and estimate.excludes_zero)})


def _describe(look, declared, markers, out) -> Description:
    """Step 4 — the curves, the slices and the robustness checks, and the worst slice named."""
    parts, scoring = look.parts, look.scoring
    said = len(markers)
    curves = build_curves(look.effective, parts.y, parts.score_net, look.adapters.metrics,
                          look.threshold.value if look.threshold.applies else None,
                          look.eval_dir / "curves", markers)
    not_cut = []
    slices = build_slices(parts.frame_test, look.resolved, look.analysis, parts.y,
                          parts.score_net, scoring, look.resampler, declared, markers,
                          raw_test=_segment_rows(look), not_cut=not_cut)
    row = look.shape == "row"
    robustness = build_robustness(
        parts.frame_test, look.resolved, look.analysis, parts.y,
        look.score_frame if row else None, scoring, look.resampler, parts.frame_train, markers,
        base_score=parts.score_net,
        perturbation_skipped=None if row else (
            f"{'windowed' if look.shape == 'window' else 'graph'} inputs are skipped and said "
            "so: a sample reads more than one row, so a one-standard-deviation shift of a "
            "column is not one perturbation of it"))
    # The builders collect rather than print — but a marker nobody sees is a marker nobody
    # acts on, so everything they recorded is said here before the records are written.
    for marker in markers[said:]:
        out(marker)

    worst = None
    ranked = [(s["name"], (s["metrics"].get(look.primary) or {}).get("value"), s["n"])
              for s in slices if not s.get("degenerate")
              and (s["metrics"].get(look.primary) or {}).get("value") is not None]
    if ranked:
        worst = (min if look.mode == "max" else max)(ranked, key=lambda t: t[1])
        out_ok(f"the worst slice on {look.primary} is {worst[0]} at {_g(worst[1])} "
               f"(n = {worst[2]})", markers)
    return Description(curves=curves, slices=tuple(slices), robustness=tuple(robustness),
                       worst=worst, not_cut=tuple(not_cut))


def _segment_rows(look):
    """The test part's raw rows, for the analysis's segments (`lens_segments`), or why they are
    not read: only when the analysis recorded an assignment rule, only for a row sample (a
    window or a graph is not one raw row), and only the columns the rule reads."""
    rule, _ = lens_segments.segment_plan(((look.analysis or {}).get("modules") or {})
                                         .get("segments"))
    if rule is None:
        return None
    if look.shape != "row":
        return (f"a {look.shape} sample reads more than one raw row, and a segment places one "
                "row; the analysis's segments are not slices of this test part")
    rows, why = lens_segments.raw_test_rows(look.manifest, len(look.parts.frame_test),
                                            rule.features)
    return rows if why is None else why


def _decision(look):
    """The stored `prediction` of a threshold reading (R-C2): σ(logit) ≥ threshold for a binary
    head — the probability of task.classes[1], σ applied unconditionally —, row score ≥
    threshold for an anomaly score; None when the reading decides otherwise or no threshold
    applies (never a column of NaN)."""
    import numpy as np
    choice = look.threshold
    if not choice.applies or choice.value is None:
        return None
    if look.effective == "binary":
        score = _one_column(look.parts.score_net)
        return (look.adapters.metrics.sigmoid(score) >= float(choice.value)).astype(np.int64)
    if look.effective == "anomaly":
        row = look.parts.scored.row_score
        row = _one_column(look.parts.score_net if row is None else row)
        return (np.asarray(row, dtype=np.float64) >= float(choice.value)).astype(np.int64)
    return None


def _write_records(look, measured, comparison, description, row, args, markers) -> None:
    """Step 5 — the stored predictions and the records everything downstream reads."""
    parts, primary, kind = look.parts, look.primary, look.kind
    per_model = measured.per_model

    layout = PredictionsLayout.of(
        parts.scored, parts.head, look.effective, prediction=_decision(look),
        baselines=[(name, measured.scorers[name]) for name in measured.compared])
    layout.write(look.eval_dir / "predictions.test.parquet")
    unscored = unscored_record(parts.scored.unscored)
    _write(look.eval_dir / "unscored.json", unscored)

    options = look.scoring.options
    doc = {
        "schema": "vega/metrics@1", "run": look.run_id, "task": kind,
        "task_effective": look.effective, "primary": primary, "mode": look.mode,
        "n_test": parts.n_rows, "n_unscored": unscored["count"], "n_boot": args.n_boot,
        "boot_seed": args.boot_seed, "ci_level": look.resampler.ci_level,
        "predictions_layout": LAYOUT,
        # what the metrics read beyond the stored scores, so the notebook recomputes them
        # without opening a part (mase's scale is the train part's)
        "options": {"season": getattr(options, "season", None),
                    "mase_scale": getattr(options, "mase_scale", None)},
        "models": {name: {k: v.as_json() for k, v in est.items()}
                   for name, est in per_model.items()},
        "differences": comparison.differences, "verdict": comparison.verdict,
        "upstream": {"valid": {primary: Estimate.upstream_value(
            row.get("primary"), "the selected run's valid metric from the ledger").as_json()}},
    }
    classes = task_classes(look.package.domain.Task.from_spec(look.spec_obj))
    support = class_support(parts.y, classes)
    if support is not None:
        doc["class_support"] = support
    if look.graph is not None:
        doc["graph"] = dict(look.graph)
    if look.gates:
        doc["gates"] = {name: gate.as_json() for name, gate in look.gates.items()}
    _write(look.eval_dir / "metrics.json", doc)
    _write(look.eval_dir / "threshold.json", look.threshold.record(
        kind, look.effective,
        # which metrics move with the threshold is the catalog's answer per task kind, not a
        # list of the binary ones written out here
        {k: v.as_json() for k, v in per_model["network"].items()
         if look.threshold.applies
         and k in ((TASKS.get(kind) or {}).get("threshold_dependent") or [])}))
    _write(look.eval_dir / "baselines.json", {"schema": "vega/baselines@1",
                                              "baselines": list(measured.baselines)})
    worst = description.worst
    _write(look.eval_dir / "slices.json", {
        "schema": "vega/slices@1", "primary": primary,
        "worst": ({"name": worst[0], "value": worst[1], "n": worst[2]} if worst else None),
        "slices": list(description.slices),
        # a slice the chain named that this look could not cut, said with its reason
        "not_cut": list(description.not_cut)})
    _write(look.eval_dir / "robustness.json", {"schema": "vega/robustness@1",
                                               "primary": primary,
                                               "checks": list(description.robustness)})
    _write(look.eval_dir / "curves.json", {"schema": "vega/curves@1",
                                           "curves": description.curves})
    _write(look.eval_dir / "markers.json", {"markers": markers})


def _evaluate_after_seal(args, package, camp_dir, spec, ledger, row, run_id, seal, seal_path,
                         baselines, markers, out, declared=()):
    """The post-seal phase, as the five steps the contract fixes the order of.

    Each step needs what the one before it produced — there is no reading a slice before the
    test part is open, and no verdict before the baselines are scored — so this is a pipeline,
    not a registry. What it does share with the pre-seal gate is that nothing here reaches for
    a value it was not handed: the look is frozen the moment it is complete.
    """
    look = _open_the_test_part(args, package, camp_dir, spec, run_id, seal, seal_path,
                               markers)
    measured = _measure(look, args, ledger, baselines, markers)
    comparison = _compare(look, measured, markers)
    description = _describe(look, declared, markers, out)
    _write_records(look, measured, comparison, description, row, args, markers)

    net = measured.per_model["network"][look.primary]
    out_ok(f"the network on the test part: {look.primary} {_g(net.value)} "
           f"{list(net.ci) if net.ci else '(no interval)'} on {look.parts.n_rows} sample(s)",
           markers)
    out_ok(f"wrote {look.eval_dir} — predictions ({LAYOUT}), metrics, threshold, unscored, "
           "baselines, slices, robustness, curves", markers)
    return 0


def _score_frame(look, frame):
    """Re-score a perturbed copy of a row sample's test features, without touching the sealed
    file again: the matrix already in memory, as a part of the package's own `TensorPart`,
    through the Evaluator's `scored` — the score per the score contract, as the headline's.
    """
    import numpy as np
    import torch
    template = look.parts.test
    n_num = template.x_num.shape[-1]
    values = frame.to_numpy(dtype=np.float32)
    x_cat = (torch.tensor(values[:, n_num:], dtype=torch.long)
             if template.x_cat is not None else None)
    part = look.package.domain.TensorPart(
        torch.tensor(values[:, :n_num], dtype=torch.float32), x_cat, None,
        row_ids=np.asarray(template.row_ids), columns=tuple(template.columns))
    return _numpy(look.evaluator.scored(look.spec_obj, part, look.model).score)


def _analysis(camp_dir, spec):
    """The analysis layer, read as a record from the chain's HTML: a file of the model's name
    beside the campaign (a campaign moved with its model), else the model where the spec names
    it (`input.model`, a path relative to the project the stages run from, like every `input`
    path the spec carries — `data/<stem>.html` keeps its directory), else one of that name in
    the working directory."""
    import re
    model = (spec.get("input") or {}).get("model")
    if not model:
        return None
    named = Path(model)
    for path in (camp_dir / named.name, camp_dir.parent / named.name, named,
                 Path.cwd() / named.name):
        if not path.is_file():
            continue
        match = re.search(r'<script\s+id="layer-analysis-data"[^>]*>([\s\S]*?)</script>',
                          path.read_text(encoding="utf-8"))
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
    return None


def _definition(name, kind, look=None):
    """What a baseline is, in the catalog's words — including the task kinds where it is
    something else entirely (`linear` becomes a PCA floor for reconstruction; `tree` becomes
    an isolation forest for an unlabelled anomaly task) and the sample shapes where it reads
    something else (a flattened window, a node's own features, pooled graphs)."""
    row = (BASELINES["rows"] or {}).get(name)
    if not row:
        return name
    base = (row.get("by_task") or {}).get(kind) or row.get("definition") or name
    shape = _shape_key(look)
    extra = (row.get("by_shape") or {}).get(shape) if shape else None
    return f"{base} — {extra}" if extra else base


def _shape_key(look):
    """`window`, `graph-node`, `graph-level`, or None for a row sample (the catalog's
    `by_shape` keys)."""
    if look is None or look.shape == "row":
        return None
    if look.shape == "window":
        return "window"
    level = (sample_of(look.resolved).get("level") or "node")
    return f"graph-{'level' if level == 'graph' else 'node'}"


def _write(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=_default) + "\n",
                          encoding="utf-8")


def _default(value):
    if isinstance(value, Estimate):
        return value.as_json()
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return str(value)


def _g(value, places=5):
    return "·" if value is None else f"{float(value):.{places}g}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--run", default=None, help="default: the ledger's selection")
    ap.add_argument("--again", default=None, metavar="REASON",
                    help="a further exposure of the test part, with the reason on the record")
    ap.add_argument("--threshold", default=None,
                    help="valid-optimum (default) | 0.5 | a value (first exposure only)")
    ap.add_argument("--slices", nargs="*", default=None, metavar="COL=VALUE")
    ap.add_argument("--baselines", default=None,
                    help=f"default: {','.join(DEFAULT_BASELINES)}, plus what the task and the "
                         f"shape add ({', '.join(DEFAULT_WHEN)})")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=7)
    ap.add_argument("--unattended", action="store_true")
    ap.add_argument("--vega-forge-dir", default=None,
                    help=f"the vega-forge skill whose vega_core opens the package "
                         f"(default: {FORGE_DIR})")
    a = ap.parse_args(argv)
    # A bootstrap resample of a small slice will now and then hold a single class, and
    # sklearn says so once per draw. This script decides for itself what an undefined metric
    # means — `Estimate` records it as not computable and a degenerate slice is dropped from
    # the ranking — so the per-draw warning is noise over a decision already taken. Only
    # sklearn's own UserWarnings are quieted; everything else still reaches the terminal.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
    warnings.filterwarnings("ignore", message="No positive class found in y_true.*")
    try:
        return evaluate(a)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
