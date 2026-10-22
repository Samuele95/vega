#!/usr/bin/env python3
"""
apply_evaluation_layer.py — validate a `layer-evaluation-data` document and inject the
/vega-eval `evaluation` layer into the model.

    python3 apply_evaluation_layer.py MODEL.html --data <dir>/evaluation-layer.json
                                      --out OUT.html [--notebook evaluate.ipynb]
                                      [--domain-forge-dir PATH]

The authoring file carries only the **judgement** — the five readings of eval-contract §5
(`verdict`, `calibration`, `slices`, `robustness`, `limits`) and the forks — plus
`"from_eval": "<dir>"`. Everything measured is spliced from that directory: `metrics.json`,
`threshold.json`, `baselines.json`, `slices.json`, `robustness.json`, `curves.json` and
`markers.json` verbatim, the seal from `test_seal.json` **minus its token**, and the notebook
record written here from `--notebook`.

What this script refuses to publish (eval-contract §6):

  * a metric anywhere in `metrics`, `baselines`, `slices`, `robustness` or
    `threshold.metrics_at_threshold` with no confidence interval — a point estimate on a
    held-out part of `n` rows is a number without a scale, and the whole stage exists to keep
    one from being read as a result. Entries marked `_upstream` are exempt, because they say in
    the record that this stage did not measure them — and only where the contract puts an
    upstream number (`metrics.upstream`, a baseline row's `valid`): a **test** number marked
    `_upstream` is refused, since no earlier stage ever saw the test part;
  * a metric its data-dependent gate withheld (`metrics.gates`, decided on the train part:
    `mape` when |y| > 0.1 on 99 % of the rows) published with a value anyway;
  * a threshold — the network's, or a baseline's own `decision_threshold` — whose `source` names
    the test part, and a `user:` threshold on any exposure
    after the first — both are a decision made knowing the answer;
  * a machine `verdict` that disagrees with its own record: `earned` that does not follow from
    the difference's interval in the mode's direction (or `beaten_by_baseline` that does not),
    an earned network with no fitted baseline named, and — the other way round — a verdict on
    a primary that lives on each model's own scale (an unlabelled anomaly task's
    `anomaly_score_mean`) that is not `earned: null` with its reason and no difference, or an
    `earned: null` anywhere else;
  * a `readings.verdict` that does not state the network's primary metric with its bounds and
    the paired difference against the strongest fitted baseline with its bounds (and against
    `linear` too, when `linear` was fitted and is not that baseline), at the rounding
    `metrics.json` holds, or whose *earned* statement contradicts `verdict.earned` — and, for a
    verdict that is not assessable, one that does not say so or claims earned or not earned,
    and for any other verdict, one that calls it not assessable;
  * a `readings.slices` that does not name the worst slice with its `n`;
  * **any** number in **any** reading that no spliced record supports at the rounding the
    reading used. A retyped digit is a verdict drifting from its evidence while still reading
    as evidence, and it is refused rather than warned about.

The block is written ONLY by `domain-forge/scripts/apply_layer.py`. The input is never modified.

Exit codes: 0 wrote OUT; 2 validation or input error; 3 platform writer missing or failed.
"""
from __future__ import annotations

import argparse
import html as _html
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_eval_core import (Estimate, digest_file,  # noqa: E402
                            reading_numbers, round_like, unsupported_numbers)

SKILL_DIR = HERE.parent
ASSETS = SKILL_DIR / "assets"
RENDER_JS = ASSETS / "evaluation-render.js"
STYLE_CSS = ASSETS / "evaluation-layer.css"

LAYER = "evaluation"
SCHEMA = "vega/evaluation@1"
MAX_RENDER = 60 * 1024
MAX_DATA = 400 * 1024

READINGS = ("verdict", "calibration", "slices", "robustness", "limits")
REQUIRED = ("schema", "seal", "run", "verdict", "metrics", "baselines", "slices",
            "robustness", "notebook", "readings")

# eval-contract §3–§4, read from the one machine copy. Restating these rows here would put a
# second copy in a module that does not own them, free to drift from the catalog and invisible
# to tests/test_catalog_sync.py — which compares the catalog against the prose, not against
# whatever a script happens to believe.
CATALOG = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
THRESHOLD_TASKS = tuple(CATALOG["threshold_tasks"])
REQUIRED_BASELINES = tuple(CATALOG["baselines"]["required"])
REQUIRED_ONE_OF = tuple(CATALOG["baselines"]["required_one_of"])

# eval-contract §5: the primaries that live on each model's own score scale, by task kind — a
# verdict on one of them is not assessable (the same table evaluate.py writes the verdict by)
SCALE_BOUND_PRIMARIES = {"anomaly": ("anomaly_score_mean",)}
NOT_ASSESSABLE = "not assessable"

# the records the readings are checked against; `readings` itself is excluded, so one reading
# can never be the evidence for a number in another
EVIDENCE = ("metrics", "threshold", "baselines", "slices", "robustness", "seal", "run",
            "verdict", "curves")

_SPLICE = (("metrics", "metrics.json"), ("threshold", "threshold.json"),
           ("baselines", "baselines.json"), ("slices", "slices.json"),
           ("robustness", "robustness.json"), ("curves", "curves.json"))

_NEGATIVE = re.compile(r"\b(not|no|does not|did not|hasn'?t|has not|fails? to|without)\b",
                       re.I)
# used only to notice that the input already carries this layer, so replacing it can be said;
# the block itself is written by domain-forge's apply_layer.py and never by this script
_BLOCK_RE = re.compile(r"[ \t]*<!--\s*@LAYER:start\s+" + LAYER +
                       r"\b[\s\S]*?@LAYER:end\s+" + LAYER + r"\s*-->[ \t]*\n?")


# --------------------------------------------------------------------------- splicing

def splice(doc, out, base_dir):
    """Replace `from_eval` with the measured records. The judgement stays the author's."""
    errs = []
    src = doc.get("from_eval")
    if not src:
        return doc, ["the authoring file carries no \"from_eval\": the measured records are "
                     "spliced from the evaluation directory, never retyped into the layer"]
    eval_dir = Path(src)
    if not eval_dir.is_absolute():
        # relative to the authoring document, then to the working directory — an author who
        # writes "." means "beside this file", which is not where the shell happens to be
        eval_dir = next((c for c in (base_dir / src, Path.cwd() / src) if c.is_dir()),
                        base_dir / src)
    if not eval_dir.is_dir():
        return doc, [f"from_eval {src!r} is not a directory (looked in {eval_dir})"]

    for key, name in _SPLICE:
        path = eval_dir / name
        if not path.is_file():
            if key == "threshold":
                continue
            errs.append(f"{name} is not in {eval_dir}; run evaluate.py before applying the "
                        "layer — the layer publishes what the evaluation measured")
            continue
        try:
            doc[key] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errs.append(f"{path} is not readable JSON: {exc}")
    if errs:
        return doc, errs

    markers_path = eval_dir / "markers.json"
    if markers_path.is_file():
        spliced = json.loads(markers_path.read_text(encoding="utf-8")).get("markers") or []
        doc["markers"] = list(spliced) + [m for m in (doc.get("markers") or [])
                                          if m not in spliced]

    seal_path = eval_dir.parent.parent / "test_seal.json"
    if not seal_path.is_file():
        seal_path = eval_dir / "test_seal.json"
    if not seal_path.is_file():
        return doc, [f"no test_seal.json beside the campaign (looked at {seal_path}); a layer "
                     "that publishes test numbers without the seal that recorded the look is "
                     "the exact claim this stage exists to make impossible"]
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal.pop("token", None)          # the token is the door key; it never travels in the model
    doc["seal"] = seal

    metrics = doc.get("metrics") or {}
    run_id = metrics.get("run") or seal.get("run")
    campaign = eval_dir.parent.parent
    row = _ledger_row(campaign, run_id)
    doc["run"] = {"id": run_id, "campaign": str(seal.get("campaign") or campaign),
                  "checkpoint_digest": (row or {}).get("checkpoint_digest"),
                  "spec_digest": (row or {}).get("spec_digest"),
                  "valid": ((metrics.get("upstream") or {}).get("valid") or {})}
    doc["verdict"] = metrics.get("verdict") or doc.get("verdict")
    doc.setdefault("schema", SCHEMA)
    out(f"OK: from_eval: the measured records spliced from {eval_dir}, the seal from "
        f"{seal_path.name} (token withheld), exposure {seal.get('exposures')}")
    return doc, []


def _ledger_row(campaign, run_id):
    path = Path(campaign) / "experiments.json"
    if not path.is_file():
        return None
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return next((r for r in ledger.get("runs") or [] if r.get("id") == run_id), None)


# --------------------------------------------------------------------------- validation

def _has_ci(entry):
    """Whether §6 lets this entry into the layer — asked of the type that owns the rule.

    Spelling the predicate out here as well would be a second copy of the one thing this
    stage is for, free to drift from the one the core enforces.
    """
    if not isinstance(entry, dict):
        return True
    try:
        return Estimate.from_json(entry).publishable
    except (TypeError, ValueError):
        return False


# pointers, not measurements: `slices.worst` repeats a number from the slice it names, and
# `threshold.value` is a decision the evaluation made rather than a quantity it measured. Both
# are checked where they belong — the slice row itself carries the interval, and the threshold
# has its own rule below — so requiring a CI here would be asking for an interval on a name.
_NOT_MEASUREMENTS = ("slices/worst", "threshold")
# ...except what the threshold record measured at that threshold: test metrics, each with its CI
_MEASURED_INSIDE = ("threshold/metrics_at_threshold",)


def _walk_metric_entries(node, path=""):
    """Every `{value: …}` record in a tree, with the path that names it."""
    inside = any(path == keep or path.startswith(keep + "/") for keep in _MEASURED_INSIDE)
    if not inside and any(path == skip or path.startswith(skip + "/")
                          for skip in _NOT_MEASUREMENTS):
        return
    if isinstance(node, dict):
        if "value" in node and not isinstance(node.get("value"), (dict, list)):
            yield path, node
            return
        for key, value in node.items():
            yield from _walk_metric_entries(value, f"{path}/{key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_metric_entries(value, f"{path}[{i}]")


def _measured(doc):
    """(path, record) of every tree whose `{value: …}` entries must carry an interval."""
    for key in ("metrics", "baselines", "slices", "robustness"):
        yield key, doc.get(key) or {}
    threshold = doc.get("threshold") or {}
    yield "threshold/metrics_at_threshold", threshold.get("metrics_at_threshold") or {}


# where §3 puts an upstream number: the selected run's valid metrics, a baseline row's valid
_UPSTREAM_PATHS = (re.compile(r"^metrics/upstream(/|$)"),
                   re.compile(r"^baselines/baselines\[\d+\]/valid$"))


def _upstream_here(path: str) -> bool:
    return any(p.search(path) for p in _UPSTREAM_PATHS)


def gate_problems(doc) -> list:
    """A metric `metrics.gates` withheld (its gate decided on the train part that it does not
    apply) that a model, a slice or a robustness check still publishes with a value — unless
    it is the spec's primary, which is reported with its caveat (eval-contract §2–§3)."""
    metrics = doc.get("metrics") or {}
    withheld = [name for name, gate in (metrics.get("gates") or {}).items()
                if isinstance(gate, dict) and gate.get("applies") is False
                and name != metrics.get("primary")]
    errs = []
    for name in withheld:
        for key, node in _measured(doc):
            for path, entry in _walk_metric_entries(node, key):
                if path.rsplit("/", 1)[-1] == name and entry.get("value") is not None:
                    errs.append(f"{path} publishes {name} ({entry.get('value')}), which its "
                                f"gate withheld on the train part "
                                f"({metrics['gates'][name].get('why')}); a withheld metric is "
                                "recorded with its reason, not with a number")
    return errs


def validate(doc, out):
    errs = []
    missing = [k for k in REQUIRED if k not in doc or doc.get(k) in (None, {}, [])]
    if missing:
        errs.append(f"the layer document is missing {', '.join(missing)}; eval-contract §6 "
                    "names them as required")
        return errs

    metrics = doc["metrics"]
    kind = metrics.get("task")
    primary = metrics.get("primary")
    if kind in THRESHOLD_TASKS and not doc.get("threshold"):
        errs.append(f"task kind {kind!r} decides with a threshold and the layer carries no "
                    "`threshold` record; a precision without the threshold it was read at is "
                    "not a number anyone can act on")

    # ---- every metric travels with an interval — the threshold record's metrics too: they
    # are test numbers read at the chosen threshold, the same estimates metrics.json holds
    for key, node in _measured(doc):
        for path, entry in _walk_metric_entries(node, key):
            if isinstance(entry, dict) and entry.get("_upstream") and not _upstream_here(path):
                errs.append(f"{path} is marked _upstream, and only the selected run's valid "
                            "metrics (metrics.upstream) and a baseline row's valid number are "
                            "upstream (§3): a test number is never carried from an earlier "
                            "stage, because no earlier stage saw the test part")
            elif not _has_ci(entry):
                errs.append(f"{path} carries a value ({entry.get('value')}) and no confidence "
                            "interval; a point estimate on a held-out part is a number "
                            "without a scale — measure it on the shared resamples, or mark "
                            "the row `_upstream` if this stage did not measure it")

    # ---- a metric its gate withheld is not published with a number
    errs += gate_problems(doc)

    # ---- the baseline floor is in the published record, not only in the run that made it
    named = [b.get("name") for b in (doc["baselines"].get("baselines") or [])]
    fitted = [b.get("name") for b in (doc["baselines"].get("baselines") or [])
              if b.get("fitted")]
    missing_floor = [b for b in REQUIRED_BASELINES if b not in named]
    if missing_floor:
        errs.append(f"the published baselines do not include {', '.join(missing_floor)}; it is "
                    "the floor every other number is read against")
    if not any(b in named for b in REQUIRED_ONE_OF):
        errs.append(f"the published baselines include none of {', '.join(REQUIRED_ONE_OF)}; "
                    "without one of them the layer cannot say what the network's non-linearity "
                    "bought")

    # ---- the threshold was not chosen on the test part
    threshold = doc.get("threshold") or {}
    source = str(threshold.get("source") or "")
    if "test" in source.lower():
        errs.append(f"the threshold's source is {source!r}, which names the test part; a "
                    "threshold optimised on the part it is then measured on is the answer "
                    "chosen after the look")
    # ---- each baseline's own threshold too (eval-contract §2): chosen on its valid scores
    for row in (doc.get("baselines") or {}).get("baselines") or ():
        own = str(((row or {}).get("decision_threshold") or {}).get("source") or "")
        if "test" in own.lower():
            errs.append(f"baseline {row.get('name')!r}'s threshold source is {own!r}, which names "
                        "the test part; a baseline's threshold is chosen on its valid scores")
    matched = ((doc.get("seal") or {}).get("test") or {}).get("digest_matches_manifest")
    if matched is False:
        errs.append("the seal records digest_matches_manifest false: the test part does not "
                    "hash to what the shaper's manifest recorded, so these numbers were "
                    "measured on a file that is not the one the chain sealed")
    exposures = int((doc.get("seal") or {}).get("exposures") or 1)
    if exposures > 1 and any(str(((r or {}).get("decision_threshold") or {}).get("source") or "")
                             .startswith("user:")
                             for r in (doc.get("baselines") or {}).get("baselines") or ()):
        errs.append(f"a baseline's threshold source is a user value on exposure {exposures}; "
                    "on a further exposure it is valid-optimum or 0.5, as the network's")
    if source.startswith("user:") and exposures > 1:
        errs.append(f"the threshold's source is {source!r} on exposure {exposures}; a value "
                    "the user gave after the test part had already been seen is a choice made "
                    "knowing the answer — on a further exposure it is valid-optimum or 0.5")

    # ---- the machine verdict agrees with its own record
    errs += verdict_problems(doc["verdict"], metrics)

    # ---- the readings exist, and each is prose
    readings = doc.get("readings") or {}
    for name in READINGS:
        text = readings.get(name)
        if not isinstance(text, str) or len(text.strip()) < 40:
            errs.append(f"readings.{name} is missing or too short to be a reading; "
                        "eval-contract §5 fixes what each one must say")
    if errs:
        return errs

    # ---- the verdict reading states the numbers the verdict object holds
    verdict = doc["verdict"]
    text = readings["verdict"]
    net = ((metrics.get("models") or {}).get("network") or {}).get(primary) or {}
    errs += _states(text, net, f"the network's {primary}", "readings.verdict")

    strongest = verdict.get("strongest_baseline")
    diffs = metrics.get("differences") or {}
    if strongest and strongest in diffs:
        errs += _states(text, diffs[strongest],
                        f"the paired difference against {strongest}", "readings.verdict")
    if "linear" in fitted and strongest != "linear" and "linear" in diffs:
        errs += _states(text, diffs["linear"],
                        "the paired difference against linear (fitted, and not the strongest "
                        "baseline)", "readings.verdict")

    # ---- and does not contradict it
    errs += _verdict_prose_problems(text, verdict)

    # ---- the slices reading names the worst slice with its n
    worst = (doc["slices"] or {}).get("worst")
    if worst:
        slices_text = readings["slices"]
        if str(worst.get("name")) not in slices_text:
            errs.append(f"readings.slices does not name the worst slice "
                        f"{worst.get('name')!r}; §5 asks for the slices where the network is "
                        "worst, by name")
        elif str(worst.get("n")) not in slices_text:
            errs.append(f"readings.slices names {worst.get('name')!r} without its n "
                        f"({worst.get('n')}); a slice reading without its n invites a "
                        "conclusion the sample cannot carry")

    # ---- a retyped digit is refused, in every reading
    evidence = {k: doc.get(k) for k in EVIDENCE if k in doc}
    for name in READINGS:
        bad = unsupported_numbers(readings[name], evidence)
        if bad:
            errs.append(f"readings.{name} states {', '.join(bad)}, which no spliced record "
                        "holds at that rounding; the readings summarise the records and a "
                        "number that drifted from them reads as evidence while being a "
                        "memory of evidence")

    notebook = doc.get("notebook") or {}
    if not notebook.get("executed"):
        if not any("notebook not executed" in str(m) for m in (doc.get("markers") or [])):
            errs.append("the notebook was not executed and `markers` does not say so; §6 "
                        "requires the marker beside an unexecuted notebook")
        else:
            out("WARN: the evaluation notebook was not executed — the layer says so in its "
                "markers, and the numbers below were not reproduced by it")
    return errs


def _states(text, entry, what, where):
    """The reading must contain the value and both bounds, at the record's own rounding."""
    if not isinstance(entry, dict) or entry.get("value") is None:
        return []
    wanted = [("the point estimate", entry.get("value"))]
    ci = entry.get("ci") or []
    if len(ci) == 2 and all(c is not None for c in ci):
        wanted += [("the lower bound", ci[0]), ("the upper bound", ci[1])]
    else:
        return [f"{where} must state {what} with its interval, and the record carries no "
                "interval for it"]
    # the reading is scanned once, not once per bound
    said = reading_numbers(text)
    missing = [label for label, value in wanted
               if not any(abs(round_like(value, shown) - said_value) < 1e-12
                          for shown, said_value in said)]
    if missing:
        return [f"{where} does not state {', '.join(missing)} of {what} "
                f"({entry.get('value')} {list(ci)}); §5 fixes the form of the verdict and the "
                "numbers in it are the record's, at the record's rounding"]
    return []


def _earned_claim(text, skip_not_assessable=False):
    """Whether the prose says the network earned its cost, or says it did not, or neither. With
    skip_not_assessable (a verdict that is not assessable), a sentence that says so claims
    neither, whatever else it names."""
    for sentence in re.split(r"(?<=[.;])\s+", text or ""):
        if "earn" not in sentence.lower() and "worth" not in sentence.lower():
            continue
        if skip_not_assessable and NOT_ASSESSABLE in sentence.lower():
            continue
        return not bool(_NEGATIVE.search(sentence))
    return None


def _verdict_prose_problems(text, verdict):
    """readings.verdict against the machine verdict (eval-contract §5): an assessable verdict is
    read as earned or not earned, as the record says, and never called not assessable — §5
    keeps that wording for a scale-bound primary; a not-assessable one says so in words and
    claims neither."""
    earned = verdict.get("earned")
    unassessed = earned is None
    claim = _earned_claim(text, skip_not_assessable=unassessed)
    said = NOT_ASSESSABLE in (text or "").lower()
    if unassessed:
        errs = [] if said else [
            "verdict.earned is null — the verdict is not assessable — and readings.verdict does "
            "not say so; §5 has it said in words, as neither a win nor a loss"]
        if claim is not None:
            errs.append(f"readings.verdict reads as {'earned' if claim else 'not earned'} but "
                        "the verdict is not assessable (verdict.earned is null); no paired "
                        "difference was computed, so neither can be claimed")
        return errs
    strongest = verdict.get("strongest_baseline")
    errs = [] if not said else [
        f"readings.verdict calls the verdict not assessable, but verdict.earned is {bool(earned)}"
        f"{f' against {strongest}' if strongest else ''}; §5 keeps that wording for a primary "
        "that lives on each model's own score scale (verdict.earned null), and this verdict is "
        "earned or not"]
    if claim is not None and claim != bool(earned):
        errs.append(f"readings.verdict reads as {'earned' if claim else 'not earned'} but "
                    f"verdict.earned is {bool(earned)}; the prose and the machine record are "
                    "the same finding and they disagree")
    return errs


def verdict_problems(verdict, metrics):
    """What is wrong with the machine verdict against its own record (eval-contract §2, §5).

    A verdict is `earned` true or false, read off the interval of the paired difference against
    the strongest fitted baseline — true exactly when that interval excludes 0 in the mode's
    direction, `beaten_by_baseline` exactly when it excludes 0 the other way — or, on a primary
    that lives on each model's own score scale (an unlabelled anomaly task), `earned` null: not
    assessable, with its reason, no baseline named, no difference, no interval, and no paired
    difference anywhere in the record. Anything else is two findings that disagree."""
    if not isinstance(verdict, dict):
        return [f"the verdict is {type(verdict).__name__}, not the machine verdict object"]
    kind = metrics.get("task_effective") or metrics.get("task")
    primary = metrics.get("primary")
    earned = verdict.get("earned")
    if earned is None:
        return _unassessed_problems(verdict, metrics, kind, primary)
    if not isinstance(earned, bool):
        return [f"verdict.earned is {earned!r}; it is true, false, or null for a verdict that "
                "is not assessable"]
    if primary in SCALE_BOUND_PRIMARIES.get(kind, ()):
        return [f"verdict.earned is {earned} on {primary}, which lives on each model's own score "
                "scale: without a label a paired difference compares scales, not models — the "
                "verdict is not assessable (earned null, with its reason)"]
    return _interval_problems(verdict, metrics.get("mode") or "max")


def _unassessed_problems(verdict, metrics, kind, primary):
    """A not-assessable verdict (`earned` null): only on a scale-bound primary, with its reason,
    and with nothing measured — no baseline, difference or interval, no paired differences."""
    errs = []
    if primary not in SCALE_BOUND_PRIMARIES.get(kind, ()):
        errs.append(f"verdict.earned is null (not assessable) on a {kind} task read on "
                    f"{primary}; only a primary that lives on each model's own score scale "
                    "(an unlabelled anomaly task's anomaly_score_mean) has no machine "
                    "verdict — every other verdict is earned or not, off its interval")
    if not str(verdict.get("reason") or "").strip():
        errs.append("the verdict is not assessable and carries no reason; the record says "
                    "why no paired difference was read")
    named = {k: verdict.get(k) for k in ("strongest_baseline", "diff", "beaten_by_baseline")
             if verdict.get(k) is not None}
    if named or verdict.get("ci") or metrics.get("differences"):
        errs.append(f"the verdict is not assessable, yet the record carries "
                    f"{named or 'an interval or paired differences'}; no paired difference "
                    "is computed on a primary each model reads on its own scale")
    return errs


def _interval_problems(verdict, mode):
    """An assessable verdict against its own interval: `earned` exactly when the interval of
    the difference excludes 0 in the mode's direction, `beaten_by_baseline` exactly when it
    excludes 0 the other way, and an earned network names the baseline it beat."""
    earned, strongest = verdict.get("earned"), verdict.get("strongest_baseline")
    if strongest is None:
        return (["verdict.earned is true with no fitted baseline named; a network earns its "
                 "cost only against a baseline it beat"] if earned else [])
    try:
        estimate = Estimate.from_json({"value": verdict.get("diff"), "ci": verdict.get("ci")})
    except (TypeError, ValueError):
        estimate = None
    if estimate is None or not estimate.has_interval:
        return [f"the verdict against {strongest} carries no interval ({verdict.get('ci')}); "
                "the verdict is read off the interval of the paired difference"]
    errs = []
    follows = bool(estimate.better_than_zero(mode))
    if follows != earned:
        errs.append(f"verdict.earned is {earned} but its interval {list(estimate.ci)} "
                    f"{'excludes' if follows else 'does not exclude'} 0 in the {mode} "
                    "direction; the verdict is its interval's reading and the two disagree")
    beaten = verdict.get("beaten_by_baseline")
    worse = bool(estimate.excludes_zero) and not follows
    if beaten is not None and bool(beaten) != worse:
        errs.append(f"verdict.beaten_by_baseline is {beaten} but its interval "
                    f"{list(estimate.ci)} {'excludes' if worse else 'does not exclude'} 0 in "
                    f"{strongest}'s favour")
    return errs


# --------------------------------------------------------------------------- noscript

def _esc(value):
    return _html.escape("" if value is None else str(value), quote=True)


def _cell(entry):
    if not isinstance(entry, dict) or entry.get("value") is None:
        return "·"
    ci = entry.get("ci") or []
    tail = f" [{ci[0]}, {ci[1]}]" if len(ci) == 2 else (" (upstream)" if entry.get("_upstream")
                                                        else "")
    return _esc(f"{entry['value']}{tail}")


def noscript_html(doc):
    metrics = doc["metrics"]
    primary = metrics.get("primary")
    seal = doc.get("seal") or {}
    verdict = doc.get("verdict") or {}
    rows = []
    for name, model in (metrics.get("models") or {}).items():
        rows.append(f"<tr><th scope=\"row\">{_esc(name)}</th><td>"
                    f"{_cell(model.get(primary))}</td></tr>")
    baselines = "".join(
        f"<tr><th scope=\"row\">{_esc(b.get('name'))}</th><td>"
        f"{_cell(b.get('test'))}</td><td>{_esc(b.get('definition'))}</td></tr>"
        for b in (doc["baselines"].get("baselines") or []))
    readings = "".join(f"<h4>{_esc(k)}</h4><p>{_esc((doc.get('readings') or {}).get(k))}</p>"
                       for k in READINGS)
    return (
        "<section class=\"layer-evaluation-noscript\">"
        f"<h3>Evaluation — {_esc(doc['run'].get('id'))} on the test part</h3>"
        f"<p>Sealed {_esc(seal.get('opened_at'))}; the test part has been opened "
        f"{_esc(seal.get('exposures'))} time(s). {_esc(metrics.get('n_test'))} rows, "
        f"{_esc(metrics.get('n_boot'))} bootstrap resamples at seed "
        f"{_esc(metrics.get('boot_seed'))}.</p>"
        + (f"<p><strong>Not assessable</strong>: {_esc(verdict.get('reason'))}</p>"
           if verdict.get("earned") is None else
           f"<p><strong>{'Earned' if verdict.get('earned') else 'Not earned'}</strong> against "
           f"{_esc(verdict.get('strongest_baseline'))}: difference {_esc(verdict.get('diff'))} "
           f"{_esc(verdict.get('ci'))}.</p>")
        + f"<table><caption>{_esc(primary)} on the test part</caption><tbody>{''.join(rows)}"
        "</tbody></table>"
        f"<table><caption>Baselines</caption><tbody>{baselines}</tbody></table>"
        f"{readings}</section>")


# --------------------------------------------------------------------------- the writer

def notebook_record(path, out):
    p = Path(path)
    if not p.is_file():
        return None, f"--notebook {p} not found; the stage order is evaluate → notebook → apply"
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{p} is not a readable notebook: {exc}"
    code = [c for c in nb.get("cells") or [] if c.get("cell_type") == "code"]
    outputs = [c.get("outputs") or [] for c in code]
    errored = any(o.get("output_type") == "error" for cell in outputs for o in cell)
    executed = bool(code) and all(cell for cell in outputs) and not errored
    # The notebook's guard cell prints `VEGA-EVAL: sealed` when it recomputed from the stored
    # predictions and `VEGA-EVAL: reopened` if it went to the test part itself. Reading the
    # marker rather than the prose is what makes the claim checkable.
    text = "\n".join("".join(o.get("text") or []) for cell in outputs for o in cell)
    marker = re.search(r"VEGA-EVAL:\s*(sealed|reopened)\b", text)
    reopened = None
    if marker:
        reopened = (marker.group(1) == "reopened")
    elif executed:
        out("WARN: the evaluation notebook carries no VEGA-EVAL marker; it was emitted by an "
            "older nb_emit.py, and whether it recomputed from the stored predictions or "
            "opened the test part again is unknown rather than proven")
    if not executed:
        out("WARN: notebook not executed" + (" — a cell ended in an error output" if errored
                                             else ""))
    return {"path": str(p), "executed": executed, "digest": digest_file(p),
            "reopened_test": reopened, "code_cells": len(code)}, None


def platform_apply(writer, html, data_text, render_js, style_css, produced_by, noscript,
                   tmpdir, model_path):
    notes = []
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("df_apply_layer", str(writer))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.apply_layer(html, LAYER, data_text, render_js, style_css=style_css,
                                 data_type="application/json", produced_by=produced_by,
                                 noscript_html=noscript)
        if isinstance(result, str) and f"@LAYER:start {LAYER}" in result:
            return result, "platform writer (import)"
        notes.append("import path returned no layer block")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"import path failed: {exc}")
    try:
        tmp = Path(tmpdir)
        files = {}
        for name, text in (("layer.json", data_text), ("evaluation-render.js", render_js),
                           ("evaluation-layer.css", style_css), ("noscript.html", noscript)):
            files[name] = tmp / name
            files[name].write_text(text, encoding="utf-8")
        out_file = tmp / "out.html"
        proc = subprocess.run(
            [sys.executable, str(writer), str(model_path), "--layer", LAYER,
             "--data", str(files["layer.json"]),
             "--render", str(files["evaluation-render.js"]),
             "--style", str(files["evaluation-layer.css"]),
             "--data-type", "application/json", "--produced-by", produced_by,
             "--noscript", str(files["noscript.html"]), "--out", str(out_file)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and out_file.is_file():
            return out_file.read_text(encoding="utf-8"), "platform writer (CLI)"
        notes.append(f"CLI path failed (exit {proc.returncode}): "
                     f"{(proc.stderr or proc.stdout).strip()[:300]}")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"CLI path failed: {exc}")
    return None, "; ".join(notes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="input MODEL.html (never modified)")
    ap.add_argument("--data", required=True, help="the evaluation's evaluation-layer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--notebook", default=None, help="evaluate.ipynb (executed)")
    ap.add_argument("--produced-by", default="/vega-eval")
    ap.add_argument("--domain-forge-dir", default=None)
    ap.add_argument("--render", default=str(RENDER_JS), help=argparse.SUPPRESS)
    ap.add_argument("--style", default=str(STYLE_CSS), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    model_path, out_path = Path(a.model), Path(a.out)
    if model_path.resolve() == out_path.resolve():
        print("ERROR: --out must differ from the input model (the input is never modified)")
        return 2
    try:
        html = model_path.read_text(encoding="utf-8")
        doc = json.loads(Path(a.data).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read the inputs: {exc}")
        return 2
    if not isinstance(doc, dict):
        print(f"ERROR: {a.data} parses to a {type(doc).__name__}, not an object; the authoring "
              "file is a JSON object carrying `from_eval` and `readings` (eval-contract §6)")
        return 2
    for asset in (Path(a.render), Path(a.style)):
        if not asset.is_file():
            print(f"ERROR: asset missing: {asset}")
            return 2
    render_js = Path(a.render).read_text(encoding="utf-8").rstrip()
    style_css = Path(a.style).read_text(encoding="utf-8").rstrip()
    if len(render_js.encode("utf-8")) > MAX_RENDER:
        print(f"ERROR: render script is {len(render_js.encode('utf-8'))} bytes; contract §6 "
              f"caps it at {MAX_RENDER // 1024} KB")
        return 2

    doc, errs = splice(doc, print, Path(a.data).resolve().parent)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2
    doc.pop("from_eval", None)

    if a.notebook:
        record, nb_err = notebook_record(a.notebook, print)
        if nb_err:
            print(f"ERROR: {nb_err}")
            return 2
        doc["notebook"] = record
        if not record["executed"]:
            markers = doc.setdefault("markers", [])
            if not any("notebook not executed" in str(m) for m in markers):
                markers.append("WARN: notebook not executed")

    errs = validate(doc, print)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2

    data_text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    if len(data_text.encode("utf-8")) > MAX_DATA:
        print(f"ERROR: the layer's data is {len(data_text.encode('utf-8')) // 1024} KB; "
              f"contract §6 caps it at {MAX_DATA // 1024} KB — the curves carry thinned "
              "points and the predictions stay on disk, so a document this large means "
              "something else was inlined")
        return 2

    seal, verdict = doc["seal"], doc["verdict"]
    print(f"OK: {a.data} validates against {SCHEMA} (run {doc['run'].get('id')}, exposure "
          f"{seal.get('exposures')}, {len(doc['baselines'].get('baselines') or [])} "
          f"baseline(s), {len(doc['slices'].get('slices') or [])} slice(s), "
          f"{len(doc['robustness'].get('checks') or [])} robustness check(s), "
          + (f"{NOT_ASSESSABLE} (no paired difference)" if verdict.get("earned") is None else
             f"{'earned' if verdict.get('earned') else 'NOT earned'} against "
             f"{verdict.get('strongest_baseline')}")
          + ", notebook executed "
          f"{doc['notebook'].get('executed')})")
    if int(seal.get("exposures") or 1) > 1:
        print(f"WARN: this layer publishes numbers from exposure {seal['exposures']} of the "
              "test part; the seal's history is in the document and the readings should say "
              "what the repeated look costs the verdict")

    noscript = noscript_html(doc)
    df_dir = (Path(a.domain_forge_dir) if a.domain_forge_dir
              else SKILL_DIR.parent / "domain-forge")
    writer = df_dir / "scripts" / "apply_layer.py"
    if not writer.is_file():
        print(f"ERROR: the platform writer is not at {writer}; the layer block is written "
              "only by domain-forge/scripts/apply_layer.py, never by this script")
        return 3
    with tempfile.TemporaryDirectory() as tmpdir:
        result, how = platform_apply(writer, html, data_text, render_js, style_css,
                                     a.produced_by, noscript, tmpdir, model_path)
    if result is None:
        print(f"ERROR: the platform writer did not produce a layer block: {how}")
        return 3
    if _BLOCK_RE.search(html):
        # In the sibling stages a replaced layer is a redo. Here it may be an earlier look at
        # the test part being written over by a later, higher-exposure one, and the reader of
        # the output would have no way to tell — so the replacement is said out loud.
        print("WARN: the input already carried an evaluation layer; it was replaced. If the "
              "two layers are different exposures of the test part, the earlier one is gone "
              "from this file — the seal's history is still the record of both")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"OK: wrote {out_path} via {how}")
    print(f"OK: render script {len(render_js.encode('utf-8'))} bytes, data "
          f"{len(data_text.encode('utf-8')) // 1024} KB of the {MAX_DATA // 1024} KB the "
          f"contract allows, output {len(result.encode('utf-8')) // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
