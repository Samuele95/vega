"""
diagnostics.py — the mechanical half of `training-catalog.md` §3.

A diagnostic is a **condition on the records**, nothing more: it reads `history.jsonl`,
`grad_stats.jsonl` and `run.json`, and says whether it fired and on what evidence. The
*reading* is the skill's to write and the *rung* is a proposal — this module produces neither,
which is what keeps "the numbers said so" separable from "and here is what I make of it".

Two rules the catalog states and this module keeps:

  * a diagnostic whose condition needs a record field the backend wrote as `null` is reported
    `unavailable`, never guessed — a run that could not be checked and a run that passed the
    check are different facts;
  * "primary rises / falls / improves" is read in the direction of `metrics.selection.mode`,
    and loss thresholds are relative to the run's best valid loss, so a regression loss in any
    unit gets the same treatment as a probability.

Pure: no I/O, no printing. `evaluate(...)` takes the records and returns `Finding`s.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = ["Finding", "evaluate", "CODES"]

CODES = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13")

CLASSIFICATION = ("binary", "multiclass", "multilabel", "sequence-classification")


@dataclass(frozen=True)
class Finding:
    """One diagnostic's verdict on one run."""

    code: str
    name: str
    fired: bool
    evidence: str = ""
    unavailable: bool = False
    why_unavailable: str = ""
    deferred: bool = False
    remedies: tuple = field(default_factory=tuple)

    def as_json(self) -> dict:
        out = {"code": self.code, "name": self.name, "fired": self.fired}
        if self.evidence:
            out["evidence"] = self.evidence
        if self.unavailable:
            out["unavailable"] = True
            out["why"] = self.why_unavailable
        if self.deferred:
            out["deferred"] = True
        if self.remedies:
            out["remedies"] = list(self.remedies)
        return out


# --------------------------------------------------------------------------- reading records

def _series(history, part, key):
    """One column of the epoch records, with None where the backend wrote null."""
    out = []
    for record in history:
        node = record.get(part) or {}
        value = node.get(key) if isinstance(node, Mapping) else None
        out.append(None if value is None else float(value))
    return out


def _clean(values):
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _better(mode):
    return (lambda a, b: a > b) if mode == "max" else (lambda a, b: a < b)


def _range(values):
    clean = _clean(values)
    return (max(clean) - min(clean)) if len(clean) > 1 else 0.0


def _per_block(history, key, sub):
    """[{block: value}] per epoch for `grad_norm.per_block` / `update_ratio`."""
    out = []
    for record in history:
        node = record.get(key) or {}
        if sub:
            node = (node or {}).get(sub) or {}
        out.append({k: (None if v is None else float(v))
                    for k, v in node.items()} if isinstance(node, Mapping) else {})
    return out


def _best_epoch_index(valid_primary, mode):
    clean = [(i, v) for i, v in enumerate(valid_primary) if v is not None]
    if not clean:
        return None
    pick = max if mode == "max" else min
    return pick(clean, key=lambda pair: pair[1])[0]


def _fmt(value, places=4):
    if value is None:
        return "·"
    if isinstance(value, float):
        return f"{value:.{places}g}"
    return str(value)


# --------------------------------------------------------------------------- the catalog

def evaluate(history: Sequence[Mapping[str, Any]], *, spec: Mapping[str, Any],
             run: Mapping[str, Any] | None = None,
             grad_stats: Sequence[Mapping[str, Any]] = (),
             floor_primary: float | None = None,
             catalog: Mapping[str, Any] | None = None) -> list:
    """Every diagnostic's verdict on one finished run.

    `floor_primary` is the floor run's valid primary, which D4 needs and which does not exist
    until the floor has run — before then D4 is `unavailable`, deferred, and re-evaluated
    once the floor exists (the catalog says so in as many words).
    """
    rows = (catalog or {}).get("diagnostics") or {}
    metrics = spec.get("metrics") or {}
    primary = metrics.get("primary")
    mode = (metrics.get("selection") or {}).get("mode", "max")
    task = (spec.get("task") or {}).get("kind")
    run = run or {}
    n = len(history)

    def row(code):
        entry = rows.get(code) or {}
        return entry.get("name", code), tuple(entry.get("remedies") or ())

    def finding(code, fired, evidence="", **kw):
        name, remedies = row(code)
        return Finding(code=code, name=name, fired=fired, evidence=evidence,
                       remedies=remedies if fired else (), **kw)

    def unavailable(code, why, deferred=False):
        name, _ = row(code)
        return Finding(code=code, name=name, fired=False, unavailable=True,
                       why_unavailable=why, deferred=deferred)

    if n == 0:
        return [unavailable(code, "the run wrote no epoch records") for code in CODES]

    train_loss = _series(history, "train", "loss")
    valid_loss = _series(history, "valid", "loss")
    valid_primary = _series(history, "valid", primary) if primary else [None] * n
    train_primary = _series(history, "train", primary) if primary else [None] * n
    ratios = _per_block(history, "update_ratio", None)
    grads = _per_block(history, "grad_norm", "per_block")
    nan_flags = [bool(r.get("nan")) for r in history]
    better = _better(mode)
    out = []

    # --- D1 divergence
    if not _clean(train_loss):
        out.append(unavailable("D1", "the records carry no train loss"))
    else:
        first = next((v for v in train_loss if v is not None), None)
        worst = max(_clean(train_loss))
        nan_at = next((i + 1 for i, flag in enumerate(nan_flags) if flag), None)
        blew_up = first not in (None, 0) and worst > 10 * abs(first)
        if nan_at:
            out.append(finding("D1", True, f"a nan record at epoch {nan_at}"))
        elif blew_up:
            out.append(finding("D1", True, f"train loss reached {_fmt(worst)} against "
                                           f"{_fmt(first)} at epoch 1 (more than 10x)"))
        else:
            out.append(finding("D1", False))

    # --- D2 not learning
    if not _clean(valid_primary):
        out.append(unavailable("D2", f"the records carry no valid {primary!r}"))
    elif not any(ratios):
        out.append(unavailable("D2", "the backend wrote no update_ratio"))
    else:
        cut = max(1, int(0.2 * n))
        first = next((v for v in valid_primary if v is not None), None)
        after = _clean(valid_primary[cut:])
        flat = bool(after) and first not in (None, 0) and all(
            abs(v - first) <= 0.01 * abs(first) for v in after)
        tiny = all(all((v is None or v < 1e-5) for v in epoch.values())
                   for epoch in ratios[cut:] if epoch)
        if flat and tiny:
            out.append(finding("D2", True,
                               f"valid {primary} stayed within 1 % of its epoch-1 value "
                               f"({_fmt(first)}) after epoch {cut}, and every block's "
                               "update ratio stayed below 1e-5"))
        else:
            out.append(finding("D2", False))

    # --- D3 overfitting after the best epoch
    best_i = _best_epoch_index(valid_primary, mode)
    if best_i is None or not _clean(valid_loss):
        out.append(unavailable("D3", "the records carry no valid loss or no best epoch"))
    else:
        best_loss = min(_clean(valid_loss))
        window = list(range(best_i, min(n, best_i + 11)))
        vl = [valid_loss[i] for i in window if valid_loss[i] is not None]
        tl = [train_loss[i] for i in window if train_loss[i] is not None]
        rose = len(vl) >= 2 and (vl[-1] - vl[0]) >= 0.05 * abs(best_loss)
        fell = len(tl) >= 2 and tl[-1] < tl[0]
        if rose and fell:
            out.append(finding("D3", True,
                               f"after the best epoch ({best_i + 1}) valid loss rose "
                               f"{_fmt(vl[-1] - vl[0])} over {len(vl)} epochs — "
                               f"{_fmt(100 * (vl[-1] - vl[0]) / abs(best_loss), 3)} % of its "
                               f"best value — while train loss fell {_fmt(tl[0] - tl[-1])}"))
        else:
            out.append(finding("D3", False))

    # --- D4 underfitting (needs the floor)
    if floor_primary is None:
        out.append(unavailable("D4", "the floor run has not run yet; D4 is re-evaluated once "
                                     "it exists", deferred=True))
    elif not _clean(train_loss) or not _clean(train_primary):
        out.append(unavailable("D4", "the records carry no train loss or train primary"))
    else:
        tail = max(1, int(0.2 * n))
        tl = _clean(train_loss[-tail:])
        flat = len(tl) >= 2 and tl[0] not in (0,) and \
            abs(tl[-1] - tl[0]) < 0.01 * abs(tl[0])
        tp = _clean(train_primary)
        margin = 0.05 * abs(floor_primary) if floor_primary else 0.0
        no_better = bool(tp) and not better(tp[-1], floor_primary + (margin if mode == "max"
                                                                    else -margin))
        vl_last = next((v for v in reversed(valid_loss) if v is not None), None)
        tl_last = tl[-1] if tl else None
        small_gap = (vl_last not in (None, 0) and tl_last is not None
                     and abs(vl_last - tl_last) < 0.05 * abs(vl_last))
        if flat and no_better and small_gap:
            out.append(finding("D4", True,
                               f"train loss moved {_fmt(abs(tl[-1] - tl[0]))} over the last "
                               f"{len(tl)} epochs, train {primary} ({_fmt(tp[-1])}) is no "
                               f"better than the floor's valid {primary} "
                               f"({_fmt(floor_primary)}), and the train/valid loss gap is "
                               f"{_fmt(abs(vl_last - tl_last))}"))
        else:
            out.append(finding("D4", False))

    # --- D5 thrashing
    if not _clean(valid_primary) or not any(ratios):
        out.append(unavailable("D5", "the records carry no valid primary or no update_ratio"))
    else:
        tail = _clean(valid_primary[-10:])
        span = _range(valid_primary)
        std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
        loud = span > 0 and std > 0.05 * span
        hot = any(any((v is not None and v > 1e-2) for v in epoch.values())
                  for epoch in ratios if epoch)
        if loud and hot:
            worst = max((max((v for v in epoch.values() if v is not None), default=0.0)
                         for epoch in ratios if epoch), default=0.0)
            out.append(finding("D5", True,
                               f"valid {primary} over the last {len(tail)} epochs has std "
                               f"{_fmt(std)} against a range of {_fmt(span)}, and an update "
                               f"ratio reached {_fmt(worst)}"))
        else:
            out.append(finding("D5", False))

    # --- D6 vanishing gradients
    ids = _block_order(history)
    if not any(grads) or len(ids) < 2:
        out.append(unavailable("D6", "the backend wrote no per-block gradient norms"))
    else:
        first_id, head_id = ids[0], ids[-1]
        streak, longest = 0, 0
        for epoch in grads:
            f, h = epoch.get(first_id), epoch.get(head_id)
            if f is not None and h not in (None, 0) and f < 1e-3 * h:
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
        if longest >= 5:
            out.append(finding("D6", True,
                               f"the first block ({first_id}) carried a gradient norm below "
                               f"1e-3 x the head's ({head_id}) for {longest} consecutive "
                               "epochs"))
        else:
            out.append(finding("D6", False))

    # --- D7 exploding gradients
    steps = [s.get("grad_norm") for s in grad_stats
             if isinstance(s, Mapping) and s.get("grad_norm") is not None]
    if not steps:
        out.append(unavailable("D7", "the run wrote no grad_stats.jsonl"))
    else:
        big = [v for v in steps if float(v) > 100]
        share = len(big) / len(steps)
        if share >= 0.01:
            out.append(finding("D7", True,
                               f"{len(big)} of {len(steps)} sampled steps "
                               f"({_fmt(100 * share, 3)} %) had a pre-clip global norm above "
                               f"100, peaking at {_fmt(max(float(v) for v in big))}"))
        else:
            out.append(finding("D7", False))

    # --- D8 stale early stop
    if best_i is None:
        out.append(unavailable("D8", f"the records carry no valid {primary!r}"))
    elif n < 5:
        out.append(unavailable("D8", f"{n} epochs is too few to call an early peak"))
    else:
        early = (best_i + 1) < 0.2 * n
        after = _clean(valid_primary[best_i + 1:])
        no_more = not after or not any(better(v, valid_primary[best_i]) for v in after)
        if early and no_more:
            out.append(finding("D8", True,
                               f"the best epoch was {best_i + 1} of {n} ({_fmt(100 * (best_i + 1) / n, 3)} % "
                               f"of the run) and valid {primary} never beat "
                               f"{_fmt(valid_primary[best_i])} afterwards"))
        else:
            out.append(finding("D8", False))

    # --- D9 budget cut
    stopped = run.get("stopped_by")
    if stopped is None:
        out.append(unavailable("D9", "run.json records no stopped_by"))
    elif stopped != "budget":
        out.append(finding("D9", False))
    else:
        tail = _clean(valid_primary[-5:])
        improving = len(tail) >= 2 and better(tail[-1], tail[0])
        out.append(finding("D9", True,
                           "the campaign's budget ended the run at epoch "
                           f"{n}; over the last {len(tail)} epochs valid {primary} was "
                           + ("still improving " if improving else "flat ")
                           + f"({_fmt(tail[0]) if tail else '·'} → "
                             f"{_fmt(tail[-1]) if tail else '·'}) — the budget, not the spec, "
                             "ended it, so this is not a result"))

    # --- D10 train/valid disagreement
    head_n = max(5, int(0.2 * n))
    if n < 5 or not _clean(train_primary) or not _clean(valid_primary):
        out.append(unavailable("D10", "too few epochs, or no train/valid primary recorded"))
    else:
        tp = _clean(train_primary[:head_n])
        vp = _clean(valid_primary[:head_n])
        tr_range, va_range = _range(train_primary), _range(valid_primary)
        gained = len(tp) >= 2 and tr_range > 0 and \
            (tp[-1] - tp[0] if mode == "max" else tp[0] - tp[-1]) >= 0.02 * tr_range
        lost = len(vp) >= 2 and va_range > 0 and \
            (vp[0] - vp[-1] if mode == "max" else vp[-1] - vp[0]) >= 0.02 * va_range
        if gained and lost:
            out.append(finding("D10", True,
                               f"over the first {len(tp)} epochs train {primary} moved "
                               f"{_fmt(tp[0])} → {_fmt(tp[-1])} while valid {primary} moved "
                               f"{_fmt(vp[0])} → {_fmt(vp[-1])} — they disagree from the start"))
        else:
            out.append(finding("D10", False))

    # --- D11 dead embedding
    embeddings = _embedding_ids(spec)
    if not embeddings:
        out.append(finding("D11", False, "the spec declares no embedding block"))
    elif not any(grads):
        out.append(unavailable("D11", "the backend wrote no per-block gradient norms"))
    else:
        dead = []
        for block_id in embeddings:
            streak, longest = 0, 0
            for epoch in grads:
                value = epoch.get(block_id)
                if value is not None and value == 0:
                    streak += 1
                    longest = max(longest, streak)
                else:
                    streak = 0
            if longest >= 3:
                dead.append((block_id, longest))
        if dead:
            out.append(finding("D11", True, "; ".join(
                f"{b}'s gradient norm was 0 for {k} consecutive epochs" for b, k in dead)))
        else:
            out.append(finding("D11", False))

    # --- D12 calibration drift
    if task not in CLASSIFICATION:
        out.append(finding("D12", False, f"not a classification task ({task})"))
    else:
        ll = _series(history, "valid", "log_loss")
        auc = _series(history, "valid", "roc_auc")
        if not _clean(ll) or not _clean(auc):
            out.append(unavailable("D12", "valid log_loss or roc_auc was not recorded"))
        else:
            best_ll = min(_clean(ll))
            tail_ll = _clean(ll[-10:])
            tail_auc = _clean(auc[-10:])
            sharper = len(tail_ll) >= 2 and (tail_ll[-1] - tail_ll[0]) >= 0.05 * abs(best_ll)
            ranks = len(tail_auc) >= 2 and (tail_auc[-1] - tail_auc[0]) >= 0.005
            if sharper and ranks:
                out.append(finding("D12", True,
                                   f"over the last {len(tail_ll)} epochs valid log-loss rose "
                                   f"{_fmt(tail_ll[-1] - tail_ll[0])} (best {_fmt(best_ll)}) "
                                   f"while ROC-AUC rose {_fmt(tail_auc[-1] - tail_auc[0])}"))
            else:
                out.append(finding("D12", False))

    out.append(_d13_stopped_early(history, run, spec, valid_primary, mode, finding, unavailable))
    return out


def _d13_stopped_early(history, run, spec, valid_primary, mode, finding, unavailable):
    """D13 — early stopping fired while the schedule had not decayed and the valid primary
    was still climbing through its own noise (the CIC-IDS2017 campaign C1: patience 10 stopped
    every run at epoch 22–58 of 200 with the lr at >= 80 % of its peak; patience 30 gained
    +0.048, so every ablation had been read at an under-trained baseline)."""
    stopped = run.get("stopped_by")
    if stopped is None:
        return unavailable("D13", "run.json records no stopped_by")
    if stopped != "early_stopping":
        return finding("D13", False)
    lr = _clean([r.get("lr") for r in history])          # top-level in the epoch record (backend-contract §1)
    patience = int(((spec.get("early_stopping") or {}).get("patience")) or 10)
    tail = _clean(valid_primary[-(patience + 1):])
    if not lr or len(tail) < 3:
        return unavailable("D13", "the records carry no lr, or too few valid primaries")
    peak = max(lr)
    lr_share = lr[-1] / peak if peak else 0.0
    sign = 1.0 if mode == "max" else -1.0
    diffs = [sign * (b - a) for a, b in zip(tail, tail[1:])]
    slope = sum(diffs) / len(diffs)
    mean = slope
    sd = (sum((d - mean) ** 2 for d in diffs) / max(1, len(diffs) - 1)) ** 0.5
    noise = sd / (len(diffs) ** 0.5)          # the slope's own standard error
    fired = lr_share >= 0.8 and slope > noise
    if fired:
        return finding("D13", True,
                       f"stopped at epoch {len(history)} with the lr at {lr_share:.0%} of its peak "
                       f"while the valid primary still rose {_fmt(slope)} per epoch over the last "
                       f"{len(tail)} epochs (noise {_fmt(noise)}) — patience {patience} read noise "
                       "as convergence")
    return finding("D13", False)


def _block_order(history):
    """Block ids in the order the network runs them, as the records name them."""
    for record in history:
        node = (record.get("grad_norm") or {}).get("per_block")
        if isinstance(node, Mapping) and node:
            return list(node)
    return []


def _embedding_ids(spec):
    return [b.get("id") for b in (spec.get("blocks") or [])
            if isinstance(b, Mapping) and b.get("type") == "embedding" and b.get("id")]


def fired_codes(findings) -> list:
    return [f.code for f in findings if f.fired]


def unavailable_codes(findings) -> list:
    return [f.code for f in findings if f.unavailable]
