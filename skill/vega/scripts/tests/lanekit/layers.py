"""
The three data layers of a lane's model: their payloads and the writer that puts them on the
base model in chain order (lanekit.fixture).

    table = RawTable(raw_path, frame)
    geo = geometry_payload(SCHEMA, lane, table, evidence)    # dataset-forge/geometry@1
    ana = analysis_payload(SCHEMA, lane, table, engine)      # data-lens/analysis@1
    shp = shape_payload(manifest, digest, verification, shaped, real)   # dataset-shaper/shape@1
    layers = LayerWriter(workspace, real)                    # geometry -> analysis -> shape

On every lane but the real-data one the geometry and analysis payloads are composed from the
generator's SCHEMA, the raw rows and the findings the generator planted (lanekit.recipes
.findings_for, and the generator's own FINDINGS: lanekit.planted), in the formats their skills'
contracts define, and the layers are written by domain-forge's platform writer. On the real
sub-chain (`expect.real_subchain`, L02) each layer goes through its own skill's validating
writer (`apply_geometry_layer.py` when dataset-forge ships it, else the platform writer, D6 §18
G9; `apply_analysis_layer.py`; `apply_shape_layer.py`), fed with the engines' evidence.

The geometry states every partition candidate its generator holds (lanekit.partition: on a
labelled lane the lane's label, chosen; on a lane that takes no label each label column of the
generator, and the choice abstained, amendment A12). The analysis context copies that
provenance, and carries the coordinate pair of a row lane that declares spatial data
(`expect.spatial`, the spatial-row modality of R-E4) as `context.spatial` (lanekit.planted).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chain import Workspace, platform_writer, run_tool
from .partition import candidate_task, partitions
from .planted import finding_evidence, planted_findings, spatial_context
from .recipes import ColumnPlan, Finding, effective_task, task_prose

__all__ = ["RawTable", "LayerWriter", "geometry_payload", "analysis_payload", "shape_payload", "NOOP",
           "GEOMETRY_ROLE", "FINAL_TYPE", "ANALYSIS_MODULES", "geometry_roles", "candidate_task",
           "spatial_context"]

ANALYSIS_MODULES = ("quality", "distributions", "relations", "inference", "segments", "importance",
                    "time_series", "spatial", "drift")
NOOP = ("(function(){ /* stand-in for the owning skill's render script */ "
        "var p=document.createElement('section'); p.className='tab-pane'; "
        "p.setAttribute('data-layer','%s'); p.setAttribute('data-tab','%s'); "
        "(document.querySelector('main')||document.body).appendChild(p); })();")
# A generator role's geometry role, in the closed set dataset-forge's report contract uses
# (report-contract §1 and its fixtures; SKILL.md step 0): every column no rule derives is a
# `dimension` of the basis — the features, the label and the columns the partition holds back
# for leakage alike, as the contract's orders layer keeps `delivered_days` — while an identifier
# is `identity` and a series or graph id with repeats a `foreign-key`, outside the basis. A
# timestamp is a dimension where a window lane orders by it, and an identity where the recipe
# drops it with the other identifiers (geometry_roles).
GEOMETRY_ROLE = {"numeric": "dimension", "nominal": "dimension", "time": "dimension", "key": "identity",
                 "label": "dimension", "drop": "dimension", "series": "foreign-key", "group": "foreign-key"}
FINAL_TYPE = {"numeric": "numeric", "nominal": "nominal", "time": "datetime", "key": "id",
              "drop": "numeric", "series": "nominal", "group": "nominal"}


def _raw(lane: Any) -> Mapping[str, Any]:
    return lane.raw if hasattr(lane, "raw") else lane


def geometry_roles(schema: Sequence[Any], lane: Any) -> dict[str, str]:
    """{column: geometry role} for a lane: GEOMETRY_ROLE, with a column the recipe drops as an
    identifier (lanekit.columns ColumnPlan.keys) outside the basis — a timestamp the lane does
    not order by is an `identity`, as a key is."""
    keys = set(ColumnPlan.of(schema, lane).keys)
    return {c.name: ("identity" if c.name in keys and GEOMETRY_ROLE[c.role] == "dimension"
                     else GEOMETRY_ROLE[c.role]) for c in schema}


@dataclass(frozen=True)
class RawTable:
    """The raw rows a lane's layers describe: the CSV the generator wrote and the frame as the
    shaper reads it back."""
    path: Path
    frame: Any

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def rows(self) -> int:
        return int(len(self.frame))

    @property
    def columns(self) -> int:
        return int(self.frame.shape[1])


# --------------------------------------------------------------------------- geometry
def _space(frame: Any, columns: Sequence[str]) -> dict:
    """The numeric space of the raw feature columns: rank, the dimensions holding 95 % / 99 %
    of the variance, and the participation ratio (standardised, as geometry.py computes it)."""
    import numpy as np

    cols = [c for c in columns if c in frame.columns]
    if len(cols) < 2:
        return {"ambient_dim": len(cols), "exact_rank": len(cols), "near_rank": len(cols),
                "dims_95": len(cols), "dims_99": len(cols), "participation_ratio": float(len(cols)),
                "intrinsic_dim_twonn": None, "reading": "fewer than two numeric columns: no space to read"}
    a = frame[cols].apply(lambda s: s.astype(float)).dropna().to_numpy()
    sd = a.std(0)
    sd[sd == 0] = 1.0
    s = np.linalg.svd((a - a.mean(0)) / sd, compute_uv=False)
    var = s ** 2 / max(float((s ** 2).sum()), 1e-12)
    rank = int((s > s[0] * 1e-9).sum()) if len(s) else 0
    pr = float((s ** 2).sum() ** 2 / max(float((s ** 4).sum()), 1e-12))
    cum = np.cumsum(var)
    dims95, dims99 = int(np.searchsorted(cum, 0.95) + 1), int(np.searchsorted(cum, 0.99) + 1)
    return {"ambient_dim": len(cols), "exact_rank": rank, "near_rank": int((var > 1e-6).sum()),
            "dims_95": min(dims95, len(cols)), "dims_99": min(dims99, len(cols)),
            "participation_ratio": round(pr, 4), "intrinsic_dim_twonn": None,
            "reading": f"{len(cols)} numeric columns span rank {rank}; {min(dims95, len(cols))} "
                       "dimensions hold 95 % of the standardised variance"}


def final_type(col: Any, lane: Any = None) -> str:
    """The final type a generator column has in the geometry layer (and, copied, the analysis
    context): its role's, else its dtype's — and `ordinal` for the label of an ordinal lane (an
    ordered grade is a category with an order, whatever digits spell it; the generator's recipe
    retypes it so, citing this typing)."""
    raw = _raw(lane) if lane is not None else {}
    target = raw.get("target") or {}
    if col.role == "label" and target.get("column") == col.name and effective_task(raw) == "ordinal":
        return "ordinal"
    return FINAL_TYPE.get(col.role) or ("nominal" if col.dtype == "str" else "numeric")


def _typing(col: Any, script_type: str | None, role: str, generator: str, final: str) -> dict:
    """One typing entry: the final type (final_type), the role, the reason — and, when the
    script typed the column otherwise, the retyping's consequence block (report-contract §3:
    every retyping is a decision)."""
    entry = {"column": col.name, "script_type": script_type or final, "final_type": final, "role": role,
             "reason": col.comment or f"a {col.role} column of the {generator} generator"}
    if script_type and script_type != final:
        entry["consequences"] = {
            "decision": f"{col.name} is {final}, not {script_type}: the generator declares it a {col.role} column",
            "alternatives": [f"keep the script's {script_type}"],
            "downstream": {"preprocessing": f"the recipe treats {col.name} as {final}",
                           "model_family": "none", "leakage": "none",
                           "dimensionality": "none: the column count is unchanged",
                           "interpretability": f"{col.name} reads as the generator means it"}}
    return entry


def geometry_payload(schema: Sequence[Any], lane: Any, table: RawTable,
                     evidence: Mapping[str, Any] | None = None) -> dict:
    """The `dataset-forge/geometry@1` layer (report-contract §1) for a lane: the typing from the
    generator's roles, the space of its numeric columns (geometry.py's own `space` when the
    evidence is given), the basis (every feature column), and the lane's partition."""
    raw, evidence = _raw(lane), evidence or {}
    columns = evidence.get("columns") or {}
    features = [c.name for c in schema if c.role in ("numeric", "nominal")]
    roles = geometry_roles(schema, lane)
    members = [c.name for c in schema if roles[c.name] == "dimension"]
    typing = [_typing(col, (columns.get(col.name) or {}).get("type"), roles[col.name], raw["generator"],
                      final_type(col, raw)) for col in schema]
    space = dict(evidence.get("space") or _space(table.frame, [c.name for c in schema if c.role == "numeric"]))
    space.setdefault("reading", f"the numeric space of the {raw['generator']} rows")
    return {"schema": "dataset-forge/geometry@1",
            "source": {"path": table.name, "rows": table.rows, "columns": table.columns},
            "markers": list(evidence.get("markers") or
                            [f"OK: geometry composed by lanekit.fixture for lane {raw.get('id')}"]),
            "typing": typing, "space": space,
            "basis": {"members": members, "size": len(members),
                      # geometry.py's TwoNN estimate when the script ran (its apply script splices
                      # that number in and refuses another), else the composed space's dims_95
                      "intrinsic_dim": space.get("intrinsic_dim_twonn") if evidence else space.get("dims_95"),
                      "reading": "the basis is every column the generator declares that no other column "
                                 "derives by construction — the features, the label and the columns the "
                                 "partition holds back; identifiers and keys stand outside it"},
            "derivations": [], "disagreements": [],
            "orthogonality": {"pairs": [], "reading": "no pair is dependent by construction"},
            "partitions": partitions(schema, lane, ColumnPlan.of(schema, lane), features, table),
            "stats": dict(evidence.get("stats") or {}),
            **({"explore": evidence["explore"]} if evidence.get("explore") else {}),
            "handoff": {"next": f"/data-lens {table.name}", "note": "the analysis reads this geometry"}}


# --------------------------------------------------------------------------- analysis
def _finding_record(f: Finding, generator: str, frame: Any = None) -> dict:
    """One admitted finding (analysis-contract §1); a transform finding also states what the raw
    rows say about it (lanekit.planted.finding_evidence: the skew before and after the log)."""
    measured = finding_evidence(f, frame)
    method = "the sample skewness of the raw column and of its log" if measured else "the generator's known formula"
    return {"id": f.id, "module": f.module, "severity": f.severity, "title": f.title, "columns": list(f.columns),
            "evidence": {"planted_by": f"lanekit.datasets.{generator}.make", **measured},
            "method": {"name": method, "assumptions_checked": "n/a"},
            "reading": f.reading, "so_what": f.so_what_record(),
            "transformation_candidates": [{"id": c.id, "op": c.op, "columns": list(c.columns),
                                           "rationale": c.rationale} for c in f.candidates]}


def _modules(generator: str, ran: set[str], engine: Path | None) -> dict:
    """Every analysis module's record: the engine's, a planted finding's, or why it did not run."""
    modules = {}
    for m in ANALYSIS_MODULES:
        if engine is not None:
            modules[m] = {"reading": f"the engine's {m} evidence; the admitted findings are the "
                                     "signal the generator planted"}
        elif m in ran:
            modules[m] = {"ran": True, "evidence": {"planted_by": f"lanekit.datasets.{generator}.make"},
                          "reading": f"the {m} finding is the generator's planted signal, stated by construction"}
        else:
            modules[m] = {"ran": False, "skipped_because": "a lane fixture admits the findings its "
                          "generator planted; the engine runs on the real data sub-chain only"}
    return modules


def analysis_payload(schema: Sequence[Any], lane: Any, table: RawTable, engine: Path | None = None) -> dict:
    """The `data-lens/analysis@1` layer (analysis-contract §1) for a lane: the findings the
    generator planted (lanekit.planted.planted_findings), the context the chain reads (the partition,
    the time context of a window lane), and every module's record — the engine's, spliced by
    `apply_analysis_layer.py` from `engine` on the real sub-chain, else a statement of why the
    module did not run."""
    raw = _raw(lane)
    plan = ColumnPlan.of(schema, lane)
    roles = geometry_roles(schema, lane)
    findings = planted_findings(schema, lane)
    time = ({"column": plan.order_by, "series_by": plan.series_by, "reading": "the rows are ordered "
             "in time within each series"} if raw["shape"] == "window" else None)
    features = [c.name for c in schema if c.role in ("numeric", "nominal")]
    spatial = spatial_context(schema, lane)
    doc: dict[str, Any] = {
        "schema": "data-lens/analysis@1",
        "source": {"path": table.name, "rows": table.rows, "columns": table.columns, "seed": int(raw["seed"]),
                   "geometry": "present"},
        "context": {"typing": [{"column": c.name, "type": final_type(c, raw), "role": roles[c.name]}
                               for c in schema],
                    "basis": [c.name for c in schema if roles[c.name] == "dimension"], "derivations": [],
                    "partition": {"label": plan.label, "task": task_prose(lane) if plan.label else None,
                                  "features": features, "leakage": list(plan.leak),
                                  "provenance": "single-candidate" if plan.label else "abstained"},
                    "time": time, "spatial": spatial,
                    "reading": "the analysis inherits the geometry's typing, basis and partition"},
        "modules": _modules(raw["generator"], {f.module for f in findings.values()}, engine),
        "findings": [_finding_record(f, raw["generator"], table.frame) for f in findings.values()],
        "transcript": [],
        "handoff": {"shaper_candidates": [c.id for f in findings.values() for c in f.candidates],
                    "next": f"/dataset-shaper {table.name}"},
        "stances": []}
    drift = findings.get("F4")
    if drift is not None and engine is None:
        # the column the drift finding names, where the evaluation reads the columns to slice
        # the test part by (vega-eval build_slices: modules.drift.columns), as the orders
        # fixture's analysis layer carries it
        doc["modules"]["drift"]["columns"] = list(drift.columns)
    if spatial and engine is None:
        pair = spatial["columns"][0]
        doc["modules"]["spatial"] = {
            "ran": True, "evidence": {"planted_by": f"lanekit.datasets.{raw['generator']}.make",
                                      "columns": [pair["column"], pair["pair"]], "crs": spatial["crs"]},
            "reading": f"{pair['column']} and {pair['pair']} are a latitude/longitude pair in {spatial['crs']}; "
                       "the generator planted a smooth field of the label over them, stated by construction"}
    if engine is not None:
        doc["from_analysis"] = str(engine)
    return doc


# --------------------------------------------------------------------------- shape
def shape_payload(manifest: Mapping[str, Any], digest: str, verification: Mapping[str, Any],
                  shaped: Path, real: bool) -> dict:
    """The `dataset-shaper/shape@1` layer: the recipe, the manifest and its digest (X-2: the
    sha256 of the manifest.json file), the verification record — spliced from the shaper's run
    directory by `apply_shape_layer.py` on the real sub-chain."""
    files = (manifest.get("outputs") or {}).get("files") or {}
    rows = sum(int(v.get("rows") or 0) for k, v in files.items() if k != "edges")
    doc: dict[str, Any] = {"schema": "dataset-shaper/shape@1", "manifest_digest": digest,
                           "readings": {"abstract": f"{(manifest.get('input') or {}).get('rows')} raw "
                                                    f"rows shaped into {rows} rows over "
                                                    f"{len(manifest.get('output_schema') or {})} columns"}}
    if real:
        doc["from_run"] = str(shaped)
    else:
        doc.update({"recipe": {"steps": manifest["steps"]}, "manifest": dict(manifest),
                    "verification": dict(verification)})
    return doc


# --------------------------------------------------------------------------- the writer
class LayerWriter:
    """Writes the three data layers onto the base model, in chain order, each to
    `<name>.domain<suffix>.html` in the work directory."""

    def __init__(self, workspace: Workspace, real: bool) -> None:
        self.ws, self.real = workspace, real
        self.writer = platform_writer(workspace.skills)

    def path(self, suffix: str) -> Path:
        return self.ws.work / f"{self.ws.name}.domain{suffix}.html"

    def platform(self, html: str, layer: str, payload: Mapping[str, Any], by: str) -> str:
        return self.writer.apply_layer(html, layer, json.dumps(payload), NOOP % (layer, layer),
                                       data_type="application/json", produced_by=by)

    def validating(self, script: str, model: Path, layer: str, payload: Mapping[str, Any], out: Path) -> str:
        data = self.ws.scratch / f"{layer}-layer.json"
        data.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        run_tool([sys.executable, self.ws.script(script), model, "--data", data, "--out", out], f"{layer} layer")
        return out.read_text(encoding="utf-8")

    def geometry(self, base: str, payload: Mapping[str, Any]) -> Path:
        out = self.path(".geometry")
        writer = self.ws.skills / "dataset-forge/scripts/apply_geometry_layer.py"
        if self.real and writer.is_file():
            model = self.path("")
            model.write_text(base, encoding="utf-8")
            run = self.ws.scratch / "geometry.json"          # lanekit.chain Workspace.geometry_evidence
            if run.is_file():
                # the script's run the layer's numbers are checked against and spliced from
                # (report-contract §1: "numbers come verbatim from geometry.json")
                payload = {**payload, "from_run": str(run)}
            self.validating("dataset-forge/scripts/apply_geometry_layer.py", model, "geometry", payload, out)
        else:
            out.write_text(self.platform(base, "geometry", payload, "/dataset-forge"), encoding="utf-8")
        return out

    def analysis(self, model: Path, payload: Mapping[str, Any]) -> Path:
        return self._next(model, payload, (".analysis", "analysis", "data-lens"))

    def shape(self, model: Path, payload: Mapping[str, Any]) -> Path:
        return self._next(model, payload, (".analysis.shaped", "shape", "dataset-shaper"))

    def _next(self, model: Path, payload: Mapping[str, Any], names: tuple[str, str, str]) -> Path:
        """The analysis or shape layer on `model`: (file suffix, layer, owning skill)."""
        suffix, layer, skill = names
        out = self.path(suffix)
        if self.real:
            self.validating(f"{skill}/scripts/apply_{layer}_layer.py", model, layer, payload, out)
        else:
            out.write_text(self.platform(model.read_text(encoding="utf-8"), layer, payload, f"/{skill}"),
                           encoding="utf-8")
        return out
