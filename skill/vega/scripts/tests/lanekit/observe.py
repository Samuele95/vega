"""
`lanes/<id>/observed.json` (`vega/lane-observed@1`): what a lane actually exercised.

Coverage is proved by observation, not declaration (D6 §0): the values under `observed` are
read from the lane's own artifacts —

- the network: the PACKED `netspec.json` (`bundle/*/netspec.json`); a lane that stops before
  pack reads the selected run's resolved spec, and one that stops before training the forge's
  `netspec.json`. `spec_source` says which;
- `blocks`: the flattened type list of that spec, composites' inner blocks included;
- `activations`: the set of `activation.params.name`, `dense.params.activation` and
  `head.params.activation`;
- the evaluation: `metrics.json` (finite network metrics, the baselines it holds, the verdict);
- the campaign: `experiments.json` (runs that trained, the selected run);
- the service: `MANIFEST.serve.endpoints`;
- inspection, showcase and page: the results the `inspect`, `show` and `page` stages wrote.

Nothing is copied from lanes.json. `validate(doc)` checks the document against
`observed.schema.json`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import contracts
from .registry import flatten_types

HERE = Path(__file__).resolve().parent
SCHEMA_FILE = HERE / "observed.schema.json"
TRAINED = {"done", "stopped_by_budget", "stopped_early", "early_stopped", "pruned", "diverged"}
GOALS = {"unsupervised": "reconstruct", "anomaly": "anomaly"}


def _load(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _first(paths: Sequence[Path]) -> Path | None:
    return next((p for p in sorted(paths) if p.is_file()), None)


# --------------------------------------------------------------------------- locating artifacts
class Artifacts:
    """Where a lane's records are, found from its work dir."""

    def __init__(self, lane_dir: Path, work: Path) -> None:
        self.lane_dir, self.work = lane_dir, work
        self.bundle = self._bundle()
        self.campaign = self._campaign()

    def _bundle(self) -> Path | None:
        manifest = _first(list(self.work.glob("bundle/*/MANIFEST.json")))
        return manifest.parent if manifest else None

    def _campaign(self) -> Path | None:
        seals = list(self.work.glob("vega-runs/*/test_seal.json"))
        if seals:
            return sorted(seals)[0].parent
        runs = [p.parent for p in self.work.glob("vega-runs/*/experiments.json")]
        runs = runs or [p.parent for p in self.work.glob("vega-runs/*/smoke.json")]
        if not runs and (self.work / "campaign" / "test_seal.json").is_file():
            runs = [self.work / "campaign"]     # the golden lane's migrated campaign (golden/cic.py)
        return sorted(runs)[0] if runs else None

    def selected_run(self) -> str | None:
        seal = _load(self.campaign / "test_seal.json") if self.campaign else None
        if seal and seal.get("run"):
            return seal["run"]
        exps = _load(self.campaign / "experiments.json") if self.campaign else None
        return ((exps or {}).get("selection") or {}).get("run")

    def spec(self) -> tuple[str | None, dict | None]:
        if self.bundle and (self.bundle / "netspec.json").is_file():
            return "bundle", _load(self.bundle / "netspec.json")
        run = self.selected_run()
        if self.campaign and run:
            resolved = _load(self.campaign / "runs" / run / "netspec.resolved.json")
            if resolved:
                return "selected-run", resolved
        forge = _load(self.work / "netspec.json")
        return ("forge", forge) if forge else (None, None)

    def model_html(self) -> str | None:
        layered = sorted(self.work.glob("*.html"), key=lambda p: (p.name.count("."), p.name))
        return str(layered[-1].relative_to(self.lane_dir)) if layered else None


# --------------------------------------------------------------------------- reading the network
def activations(blocks: Sequence[Mapping]) -> list[str]:
    found: set[str] = set()
    for block in blocks:
        params = block.get("params") or {}
        if block.get("type") == "activation" and params.get("name"):
            found.add(params["name"])
        if block.get("type") in ("dense", "head"):
            found.add(params.get("activation") or "none")
        found.update(activations(params.get("inner") or []))
    return sorted(found)


def _name(value: Any, key: str = "name") -> str | None:
    if isinstance(value, Mapping):
        return value.get(key) or value.get("name")
    return value if isinstance(value, str) else None


def network_facts(spec: Mapping | None) -> dict:
    """The network half of `observed` from a netspec."""
    if not spec:
        return {k: None for k in ("family", "task", "base", "goal", "shape", "window", "graph",
                                  "blocks", "activations", "loss", "optimizer", "schedule",
                                  "init", "protocol")}
    task = spec.get("task") or {}
    sample = (spec.get("data_contract") or {}).get("sample") or {}
    shape = sample.get("shape")
    blocks = spec.get("blocks") or []
    return {
        "family": _name(spec.get("family")), "task": task.get("kind"),
        "base": task.get("base_kind") or task.get("base"),
        "goal": GOALS.get(task.get("kind"), "predict") if task.get("kind") else None,
        "shape": shape,
        "window": ({"length": sample.get("length"), "stride": sample.get("stride", 1),
                    "horizon": sample.get("horizon")} if shape == "window" else None),
        "graph": ({"level": sample.get("level"), "protocol": sample.get("protocol")}
                  if shape == "graph" else None),
        "blocks": flatten_types(blocks), "activations": activations(blocks),
        "loss": _name(spec.get("loss")), "optimizer": _name(spec.get("optimizer")),
        "schedule": _name(spec.get("schedule")), "init": _name(spec.get("init"), "scheme"),
        "protocol": sample.get("protocol") if shape == "graph" else None,
    }


# --------------------------------------------------------------------------- reading the records
def evaluation_facts(art: Artifacts) -> dict:
    run = art.selected_run()
    metrics = _load(art.campaign / "eval" / run / "metrics.json") if art.campaign and run else None
    if not metrics:
        return {"metrics_finite": None, "baselines": None, "verdict": None}
    models = metrics.get("models") or {}
    network = models.get("network") or {}
    finite = sorted(name for name, cell in network.items() if _measured(cell))
    return {"metrics_finite": finite, "baselines": sorted(k for k in models if k != "network"),
            "verdict": verdict_word(metrics.get("verdict"))}


def _measured(cell: Any) -> bool:
    """A metric the evaluation measured: a finite point (`value`), or a per-element vector
    (`value` null, `values` a non-empty list whose elements are finite or null — an element the
    metric leaves undefined — with at least one finite). Gate C: the vectors (f1_per_class,
    per_horizon_error, per_feature_error, residual_quantiles) were reported as never recorded."""
    cell = cell if isinstance(cell, Mapping) else {}
    value = cell.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    values = cell.get("values")
    if not isinstance(values, list) or not values:
        return False
    nums = [v for v in values if v is not None]
    return bool(nums) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                              and math.isfinite(v) for v in nums)


def verdict_word(verdict: Any) -> str | None:
    """earned | beaten_by_baseline | not_earned | not_assessable, from metrics.json's verdict
    record. `earned` null is not assessable (A30: a primary each model reads on its own score
    scale, the unlabelled anomaly task), neither a win nor a loss."""
    if not isinstance(verdict, Mapping):
        return verdict if isinstance(verdict, str) else None
    if "earned" in verdict and verdict["earned"] is None:
        return "not_assessable"
    if verdict.get("earned"):
        return "earned"
    return "beaten_by_baseline" if verdict.get("beaten_by_baseline") else "not_earned"


def campaign_facts(art: Artifacts) -> dict:
    exps = _load(art.campaign / "experiments.json") if art.campaign else None
    if not exps:
        return {"runs": None, "selected_run": art.selected_run()}
    trained = [r for r in exps.get("runs") or [] if r.get("status") in TRAINED]
    return {"runs": len(trained), "selected_run": art.selected_run()}


def serving_facts(art: Artifacts) -> dict:
    manifest = _load(art.bundle / "MANIFEST.json") if art.bundle else None
    endpoints = ((manifest or {}).get("serve") or {}).get("endpoints")
    return {"endpoints": list(endpoints) if isinstance(endpoints, list) else None}


def result_facts(lane_dir: Path, art: Artifacts | None = None) -> dict:
    """Inspection, showcase and page, from the results the later stages wrote. A lane whose
    showcase no `show` stage checked (the golden lane builds its own) is read from the slot."""
    results = lane_dir / "results"
    inspect = _load(results / "inspect.json")
    page = _load(results / "page.json")
    show = _load(results / "show.json")
    slot = art.bundle / contracts.SHOWCASE_DIR if art and art.bundle else None
    if not isinstance(show, dict) and slot and (slot / contracts.SHOWCASE_MANIFEST).is_file():
        show = showcase_facts(slot)
    return {"inspection": _inspection(inspect), "showcase": show if isinstance(show, dict) else None,
            "page": _page(page)}


def showcase_facts(slot: Path) -> dict:
    """{motif, class_roles, report, locale} of a built showcase slot (results/show.json's shape)."""
    doc = _load(slot / contracts.SHOWCASE_MANIFEST) or {}
    story = _load(slot / ((doc.get("story") or {}).get("file") or "story.json")) or {}
    page = _load(slot / (doc.get("page") or {}).get("file")) if doc.get("page") else None
    motif = (page or {}).get("motif") or story.get("motif")
    motif = motif.get("id") if isinstance(motif, Mapping) else motif   # page.json's motif is {id, base, …}
    return {"motif": motif, "class_roles": sorted(((story.get("class_roles") or {}).get("roles") or {}).items()),
            "report": (doc.get("report") or {}).get("status"), "locale": story.get("locale", "en")}


def _inspection(doc: Any) -> dict | None:
    if not isinstance(doc, Mapping):
        return None
    keep = ("nodes", "explained_rows", "oracle_checks", "verdict")
    return {k: doc.get(k) for k in keep if k in doc} or dict(doc)


def _page(doc: Any) -> dict | None:
    if not isinstance(doc, Mapping):
        return None
    runs = doc.get("runs") or []
    views = sorted({v.get("id") for r in runs for v in r.get("views") or [] if v.get("visible", True)})
    return {"views": views, "themes": sorted({r.get("theme") for r in runs if r.get("theme")}),
            "widths": sorted({r.get("width") for r in runs if r.get("width")}),
            "console_errors": sum(len([c for c in r.get("console") or [] if c.get("level") == "error"])
                                  for r in runs), "verdict": doc.get("verdict")}


# --------------------------------------------------------------------------- the document
def build(lane_id: str, lane_dir: Path, work: Path, run: Mapping[str, Any]) -> dict:
    """observed.json for one lane. `run` carries what the runner knows: git, times, verdict,
    reason, stage results (dicts), flakes, tier."""
    art = Artifacts(lane_dir, work)
    source, spec = art.spec()
    observed = {"spec_source": source, **network_facts(spec), **evaluation_facts(art),
                **campaign_facts(art), **serving_facts(art), **result_facts(lane_dir, art)}
    shots = sorted(str(p.relative_to(lane_dir)) for p in (lane_dir / "shots").glob("*.png"))
    showcase = art.bundle / contracts.SHOWCASE_DIR if art.bundle else None
    return {
        "schema": contracts.LANE_OBSERVED, "lane": lane_id, "tier": run.get("tier"),
        "git": run["git"], "started_at": run["started_at"], "finished_at": run["finished_at"],
        "verdict": run["verdict"], "reason": run.get("reason"), "stages": list(run["stages"]),
        "observed": observed, "flakes": list(run.get("flakes") or []),
        "flake_calls": dict(run.get("flake_calls") or {}),
        "artifacts": {"work": str(work.relative_to(lane_dir)),
                      "bundle": str(art.bundle.relative_to(lane_dir)) if art.bundle else None,
                      "showcase": (str(showcase.relative_to(lane_dir))
                                   if showcase and showcase.is_dir() else None),
                      "model_html": art.model_html(), "shots": shots},
    }


def validate(doc: Mapping[str, Any]) -> list[str]:
    """Schema problems of an observed.json document ([] when it validates)."""
    import jsonschema
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '(root)'}: {e.message}"
            for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))]


def write(lane_dir: Path, doc: Mapping[str, Any]) -> Path:
    path = lane_dir / "observed.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
