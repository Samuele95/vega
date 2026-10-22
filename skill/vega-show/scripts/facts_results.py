"""
facts_results — `ResultFacts`: what the sealed test said (contract §2.2–§2.4 and §4.5, the groups
`models`, `metrics`, `diffs`, `breakdown`, `verdict` and `limits`).

- **The reserved values.** `eval_network_<primary>` is the network's primary on the sealed test,
  `{value, ci, n}`, read at `records/evaluation.json#metrics.models.network.<primary>` (the story's
  `values.primary`); `eval_ci_level` is `metrics.ci_level`, the bootstrap level the evaluation used
  — absent before v2.3, which is no fact and a `WARN:`, and the page prints the interval without a
  level. An evaluation without the network's primary is refused (`PF2`: a pinned record lacks
  what the facts must read): there is nothing to show.
- **Models.** Every model the evaluation scored on the test part (`metrics.models`), by the
  catalog's result-model name (`campaign-baseline` is `campaign`), with its primary
  (`eval_<model>_<primary>`); the network also with the task's results facts that exist
  (`results_facts`, `results_emit`: a metric the evaluation did not compute is no fact and no
  warning). Each paired difference is `eval_diff_<model>`.
- **Breakdown.** The task's per-unit metric (`breakdown.by_task`, a `sequence-*` task taking its
  base's): the network's per-unit values, the trivial baseline's (what the resolver reads `weak`
  against — it computes `weak`, the extractor never does), and the unit names, each a fact:
  `eval_network_f1_per_class`, `eval_trivial_f1_per_class`, `class_names`. Names and values of
  different lengths are no breakdown, with a `WARN:`; a trivial vector of another length is no
  trivial side (`trivial: null`), with a `WARN:`, so `weak` is never read against a misaligned
  baseline.
- **Verdict.** The evaluation's machine verdict as recorded (`earned`, `beaten_by_baseline`, the
  strongest baseline by result-model name and its difference key, the reading's ref). A verdict
  that is not assessable (A30: `earned` null, an unlabelled anomaly primary on each model's own
  scale) also carries `reason`, the fact `eval_verdict_reason` (the record's `verdict.reason`,
  verbatim), and its limit `verdict_not_assessable`; it is never read as a win or a loss.
- **Limits.** `_Limits` below: one entry per limit kind whose condition holds (§2.4), mandatory
  as the catalog says, with the facts it rests on and the prose the model reads, and — where the
  record holds words of its own — `recorded`, quoted on the page beside the authored text: a
  warning's `WARN:` line, the MANIFEST stance a bundle ships by, and the classes whose sealed test
  support (`metrics.class_support`) is below the catalog's `small_support.below` with their counts
  (the `small_support` limit: a per-class score on a handful of rows is an anecdote). The words
  are the record's, verbatim, except a path in the machine's file system, cut to its file name
  (`…/pipeline.json`, `local_paths_cut`): the directories the chain ran in are no part of a
  warning, and a published page never names them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

from fact_sink import Catalogs, FactSink, Refusal, keys_in, peek, spellable
from show_ports import RecordStore
from vega_show_core import CATALOG, fact_key

__all__ = ["ResultFacts"]

EVAL, MANIFEST, NETSPEC = "records/evaluation.json", "MANIFEST.json", "netspec.json"
MET = f"{EVAL}#metrics"
_NOT_RAN = ("skipped", "not_run", "not run", "not-run")
_PATH = re.compile(r"(?<![\w.:/{])/(?:[^\s/(),;'\"{}]+/)+[^\s/(),;'\"{}]+")
_LOCAL_ROOTS = ("/home/", "/Users/", "/tmp/", "/root/", "/var/", "/mnt/", "/opt/", "/srv/",
                "/private/", "/media/", "/data/", "/scratch/", "/work/")


def local_paths_cut(words: str) -> str:
    """The recorded words with every path of the machine's file system cut to `…/<file name>`
    (a path under a local root, or one whose directory exists here); a route such as
    `/v1/network/tensor/{tensor_id}` is no file and stays."""
    def cut(m: re.Match) -> str:
        path = m.group(0)
        local = path.startswith(_LOCAL_ROOTS) or Path(path).parent.is_dir() and path.count("/") > 2
        return f"…/{Path(path).name}" if local else path
    return _PATH.sub(cut, str(words))
_UNIT_NAMES = {"class": (f"{NETSPEC}#task.classes",), "label": (f"{NETSPEC}#task.classes",),
               "horizon": (f"{MANIFEST}#output.horizons",),
               "feature": (f"{NETSPEC}#data_contract.features",), "bin": (), "k": ()}


class ResultFacts:
    """The evaluation's facts; `layer_markers` are the bundle layer's markers, whose `WARN:`
    lines are limits too."""

    def __init__(self, records: RecordStore, catalog: Catalogs,
                 layer_markers: Sequence[str] = ()) -> None:
        self._store, self._catalog, self._layer_markers = records, catalog, tuple(layer_markers)

    def collect(self, sink: FactSink) -> None:
        s = self._store
        kind, base = peek(s, f"{NETSPEC}#task.kind"), peek(s, f"{NETSPEC}#task.base_kind")
        primary = peek(s, f"{MET}.primary") or peek(s, f"{NETSPEC}#metrics.primary")
        direction = self._catalog.direction(primary)
        names = self._models(sink, primary, direction)
        if fact_key(("eval", "network", primary)) not in sink:
            raise Refusal("PF2", f"{MET}.models.network.{primary}", "the evaluation records no "
                          "network estimate of the task's primary (a finite value, with a finite "
                          "interval and n when present) — nothing to show; re-run /vega-eval")
        shown = self._results(sink, kind, primary)
        sink.name([primary, *shown])
        self._metrics(sink, primary, direction, shown)
        diffs = self._diffs(sink, names, direction)
        self._breakdown(sink, kind, base)
        verdict = self._verdict(sink, diffs)
        sink.group("limits", _Limits(s, self._catalog, self._layer_markers).collect(sink, verdict))

    def _models(self, sink: FactSink, primary: str, direction) -> dict:
        """{recorded model name: result-model name}; the models group and their primaries."""
        mapping = {"network": "network", **CATALOG["result_models"]["from_baseline"]}
        recorded = peek(self._store, f"{MET}.models", {}) or {}
        names, group = {}, []
        for model in recorded:
            name = mapping.get(model)
            if name is None or not spellable(model):
                sink.warn(f"{MET}.models.{model} is not a result model of the catalog; not shown")
                continue
            key = sink.take(self._store, ("eval", name, primary),
                            f"{MET}.models.{model}.{primary}", "ci_metric", direction)
            names[model] = name
            group.append({"name": name, "primary": key})
        sink.group("models", group)
        return names

    def _results(self, sink: FactSink, kind: str, primary: str) -> list[str]:
        shown = []
        for metric in CATALOG["results_facts"].get(kind, []):
            if metric != primary and sink.take(self._store, ("eval", "network", metric),
                                               f"{MET}.models.network.{metric}", "ci_metric",
                                               self._catalog.direction(metric)):
                shown.append(metric)
        return shown

    def _metrics(self, sink: FactSink, primary: str, direction, shown: list) -> None:
        s = self._store
        sink.take(s, "eval_n_test", f"{MET}.n_test", "count")
        sink.take(s, "seal_exposures", f"{EVAL}#seal.exposures", "count")
        if peek(s, f"{EVAL}#threshold.applies") is True:
            sink.take(s, "threshold_value", f"{EVAL}#threshold.value", "param")
        if not sink.take(s, "eval_ci_level", f"{MET}.ci_level", "param"):
            sink.warn(CATALOG["reserved_values"]["ci_level"]["absent"])
        sink.group("metrics", {"primary": primary, "direction": direction, "shown": shown})

    def _diffs(self, sink: FactSink, names: Mapping, direction) -> dict:
        out = {}
        for model in peek(self._store, f"{MET}.differences", {}) or {}:
            name = names.get(model)
            key = sink.take(self._store, ("eval", "diff", name),
                            f"{MET}.differences.{model}", "diff", direction) if name else None
            if key:
                out[name] = key
        sink.group("diffs", out)
        return out

    def _breakdown(self, sink: FactSink, kind: str, base: str | None) -> None:
        by_task = CATALOG["breakdown"]["by_task"]
        spec = by_task.get(kind)
        spec = by_task.get(base) if spec == "base" else spec
        sink.group("breakdown", self._per_unit(sink, spec) if spec else None)

    def _per_unit(self, sink: FactSink, spec: Mapping) -> dict | None:
        metric, unit = spec["metric"], spec["kind"][len("per_"):]
        stem = metric if f"per_{unit}" in metric else f"{metric}_per_{unit}"
        values = {m: sink.take(self._store, ("eval", m, stem),
                               (f"{MET}.models.{m}.{metric}.values", f"{MET}.models.{m}.{metric}"),
                               "list") for m in ("network", "trivial")}
        names = sink.take(self._store, (unit, "names"),
                          (f"{MET}.models.network.{metric}.names", *_UNIT_NAMES.get(unit, ())),
                          "labels")
        if not values["network"] or not names or \
                len(sink.value(names)) != len(sink.value(values["network"])):
            sink.warn(f"no {spec['kind']} breakdown: the evaluation's {metric} values and the "
                      f"{unit} names are missing or of different lengths")
            return None
        trivial = values["trivial"]
        if trivial and len(sink.value(trivial)) != len(sink.value(names)):
            sink.warn(f"the trivial baseline's {metric} values are not one per {unit} name; the "
                      f"{spec['kind']} breakdown has no trivial side")
            trivial = None
        return {"kind": spec["kind"], "metric": metric, "direction_of": spec["direction_of"],
                "direction": self._catalog.direction(spec["direction_of"]), "names": names,
                "values": values["network"], "trivial": trivial}

    def _verdict(self, sink: FactSink, diffs: Mapping) -> dict:
        at = f"{EVAL}#verdict" if peek(self._store, f"{EVAL}#verdict") else f"{MET}.verdict"
        verdict = peek(self._store, at) or {}
        recorded = verdict.get("strongest_baseline")
        strongest = CATALOG["result_models"]["from_baseline"].get(recorded, recorded)
        reading = f"{EVAL}#readings.verdict"
        group = {"earned": verdict.get("earned"),
                 "beaten_by_baseline": verdict.get("beaten_by_baseline"),
                 "strongest_baseline": strongest, "diff": diffs.get(strongest),
                 "reading_ref": reading if peek(self._store, reading) else None,
                 "reason": self._reason(sink, verdict, at)}
        sink.group("verdict", group)
        return group

    def _reason(self, sink: FactSink, verdict: Mapping, at: str) -> str | None:
        """A verdict that is not assessable (eval-contract §5, A30: `earned` recorded as null)
        carries its reason: the fact `eval_verdict_reason`, the record's words verbatim. Any
        other verdict has none; a null `earned` without a reason is no verdict (the resolver
        refuses it)."""
        if "earned" not in verdict or verdict.get("earned") is not None:
            return None
        return sink.take(self._store, "eval_verdict_reason", f"{at}.reason", "text")


class _Limits:
    """The limits the page must, or may, say (contract §2.4)."""

    def __init__(self, store: RecordStore, catalog: Catalogs, layer_markers: tuple) -> None:
        self._store, self._catalog, self._layer = store, catalog, layer_markers

    def collect(self, sink: FactSink, verdict: Mapping) -> list[dict]:
        """The limit entries, given the verdict group (its `diff` key is what an unearned or
        beaten verdict rests on)."""
        s, out, beaten = self._store, [], [verdict.get("diff")]
        for lid, holds, facts, prose in (
                ("split", True, ["train_rows", "valid_rows", "test_rows"], "readings.limits"),
                ("exposures", True, ["seal_exposures"], "readings.limits"),
                ("one_seed", True, [], "readings.limits"),
                ("robustness_not_run", self._robustness_not_run(), [], "readings.robustness"),
                ("verdict_not_earned", verdict.get("earned") is False, beaten, "readings.verdict"),
                ("verdict_beaten", verdict.get("beaten_by_baseline") is True, beaten,
                 "readings.verdict"),
                ("verdict_not_assessable", verdict.get("reason") is not None, [],
                 "readings.verdict"),
                ("calibration", self._calibration(), ["eval_network_ece"], "readings.calibration"),
                ("slices_empty", peek(s, f"{EVAL}#slices.slices", None) == [], [],
                 "readings.slices"),
                ("small_support", bool(self._small_support()), [], "metrics.class_support")):
            if holds:
                out.append(self._entry(lid, keys_in(sink, facts), f"{EVAL}#{prose}"))
        for entry in out:
            if entry["id"] == "small_support":
                entry["recorded"] = ", ".join(f"{c}: {n}" for c, n in self._small_support())
        out += self._manifest_limits(sink)
        if self._drift():
            out.append(self._entry("drift_suspected", [],
                                   "records/analysis.json#modules.drift.reading"))
        return out + self._warnings()

    def _entry(self, lid: str, facts: list, prose: str | None) -> dict:
        kind = "warn" if lid.startswith("warn_") else lid
        mandatory = self._catalog.show["limit_kinds"][kind]["mandatory"] != "never"
        ref = prose if prose and peek(self._store, prose) is not None else None
        return {"id": lid, "mandatory": mandatory, "facts": facts, "prose_ref": ref}

    def _warning(self, lid: str, marker: str) -> dict:
        """A bundle layer's `warn_pack_<n>` limit, with the marker it stands for: the model page
        is not a fact source, so the marker has no ref to read it through."""
        return {**self._entry(lid, [], None), "marker": marker, "recorded": local_paths_cut(marker)}

    def _small_support(self) -> list:
        """[(class, sealed test rows)] of the classes below the catalog's bound, in record order."""
        support = peek(self._store, f"{MET}.class_support", None)
        below = self._catalog.show["limit_kinds"]["small_support"].get("below", 0)
        if not isinstance(support, Mapping):
            return []
        return [(str(c), int(n)) for c, n in support.items()
                if isinstance(n, (int, float)) and not isinstance(n, bool) and n < below]

    def _manifest_limits(self, sink: FactSink) -> list[dict]:
        s, out = self._store, []
        stances = peek(s, f"{MANIFEST}#stances")
        if stances:
            words = "; ".join(str(x) for x in stances) if isinstance(stances, list) else str(stances)
            out.append({**self._entry("ship_stance", [], f"{MANIFEST}#stances"),
                        "recorded": local_paths_cut(words)})
        if isinstance(peek(s, f"{MANIFEST}#roundtrip.boundary"), Mapping):
            key = sink.take(s, "roundtrip_boundary_rows", f"{MANIFEST}#roundtrip.boundary.rows",
                            "count")
            out.append(self._entry("roundtrip_boundary", keys_in(sink, [key]),
                                   f"{MANIFEST}#roundtrip.boundary.reason"))
        if peek(s, f"{MANIFEST}#examples.kind") == "synthetic":
            out.append(self._entry("examples_synthetic", [], f"{MANIFEST}#examples.note"))
        if peek(s, f"{MANIFEST}#serve.forced_by") == "--serve":
            out.append(self._entry("serving_forced", [], f"{MANIFEST}#serve.note"))
        return out

    def _robustness_not_run(self) -> bool:
        checks = peek(self._store, f"{EVAL}#robustness.checks", None)
        if not isinstance(checks, list):
            return False
        return not any(isinstance(c, Mapping) and str(c.get("status", "ran")) not in _NOT_RAN
                       for c in checks)

    def _calibration(self) -> bool:
        kind = peek(self._store, f"{NETSPEC}#task.kind")
        metrics = ((self._catalog.eval.get("tasks") or {}).get(kind) or {}).get("secondary") or []
        return "ece" in metrics or peek(self._store, f"{MET}.models.network.ece") is not None

    def _drift(self) -> bool:
        drift = peek(self._store, "records/analysis.json#modules.drift", {}) or {}
        columns = drift.get("columns") or (drift.get("evidence") or {}).get("columns") or []
        return any((isinstance(c, str)) or (isinstance(c, Mapping) and c.get("drifted") is True)
                   for c in columns)

    def _warnings(self) -> list[dict]:
        out = []
        markers = peek(self._store, f"{EVAL}#markers", []) or []
        warned = [i for i, m in enumerate(markers) if isinstance(m, str) and m.startswith("WARN:")]
        for n, i in enumerate(warned, 1):
            out.append({**self._entry(f"warn_eval_{n}", [], f"{EVAL}#markers.{i}"),
                        "recorded": local_paths_cut(markers[i])})
        layer = [m for m in self._layer if isinstance(m, str) and m.startswith("WARN:")]
        out += [self._warning(f"warn_pack_{n}", m) for n, m in enumerate(layer, 1)]
        return out
