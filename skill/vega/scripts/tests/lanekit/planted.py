"""
What a row generator plants beyond the standard pattern (lanekit.recipes), as the layers and the
generator's own recipe read it:

    findings = planted_findings(SCHEMA, lane)    # findings_for(SCHEMA, lane) + the generator's FINDINGS
    F7 = skew_finding("F7", "T7", "area_m2", reading, rationale)     # a skew and its transform
    recipe = with_transform(standard_recipe(...), F7, "log")        # the step citing it, in phase 4
    evidence = finding_evidence(finding, frame)  # what the raw rows say about it (the skew)
    context = spatial_context(SCHEMA, lane)      # context.spatial of a spatial row lane

A generator may declare `FINDINGS`, a mapping {id: lanekit.recipes.Finding} of the findings it
plants beyond the six lanekit.recipes.findings_for numbers F1…F6 (D6 §4's "recipe beyond the
orders pattern": the log on tickets' `length_chars` and on housing's `area_m2`). The analysis
layer admits them beside the standard ones (lanekit.layers.analysis_payload), and the
generator's RECIPE cites their transformation candidates (`analysis:F<n>/T<m>`), which
`shape.py --model` resolves against that layer. They are read only for the generator's own
SCHEMA: a stand-in generator (lanekit.standin) or another table keeps the standard findings.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import datasets
from .recipes import Candidate, Finding, findings_for

__all__ = ["planted_findings", "skew_finding", "transform_step", "with_transform", "finding_evidence",
           "spatial_context", "LATITUDE", "LONGITUDE", "REPRESENTATION_OPS"]

LATITUDE, LONGITUDE = ("lat", "latitude"), ("lon", "lng", "longitude")
# the ops of the shaper's phases 5-7 (step-catalog.md): a phase-4 step goes before the first
REPRESENTATION_OPS = ("encode", "scale", "project", "select_features", "graph_edges")


def _raw(lane: Any) -> Mapping[str, Any]:
    return lane.raw if hasattr(lane, "raw") else lane


def _own_generator(schema: Sequence[Any], lane: Any) -> Any:
    """The lane's generator module when `schema` is its SCHEMA, else None."""
    name = _raw(lane).get("generator")
    if not name or name not in datasets.names():
        return None
    gen = datasets.load(name)
    return gen if tuple(gen.SCHEMA) == tuple(schema) else None


def planted_findings(schema: Sequence[Any], lane: Any) -> dict[str, Finding]:
    """findings_for(schema, lane), then the findings the lane's generator declares in FINDINGS
    (module docstring). A generator finding may not reuse a standard id, name a column its
    SCHEMA lacks, or propose a candidate id another finding holds."""
    out = findings_for(schema, lane)
    gen = _own_generator(schema, lane)
    extra: Mapping[str, Finding] = getattr(gen, "FINDINGS", None) or {}
    columns = {c.name for c in schema}
    taken = {c.id for f in out.values() for c in f.candidates}
    for key, finding in extra.items():
        problem = (f"reuses the standard id {key}" if key in out else
                   f"is keyed {key} but is {finding.id}" if key != finding.id else
                   f"names {sorted(set(finding.columns) - columns)}, not in its SCHEMA"
                   if not set(finding.columns) <= columns else
                   f"proposes {sorted(taken & {c.id for c in finding.candidates})}, which another finding holds"
                   if taken & {c.id for c in finding.candidates} else None)
        if problem:
            raise ValueError(f"generator {_raw(lane).get('generator')}: finding {key} {problem}")
        taken |= {c.id for c in finding.candidates}
        out[key] = finding
    return out


def skew_finding(fid: str, tid: str, column: str, reading: str, rationale: str) -> Finding:
    """A `distributions` finding that `column` is right-skewed, with the one candidate that
    answers it: a `transform` of the column (step-catalog.md phase 4, source `analysis:F<n>`)."""
    return Finding(fid, "distributions", "medium", f"{column} is right-skewed", (column,), reading,
                   (("preprocessing", f"a log brings the tail of {column} in before the scale"),
                    ("modeling", "the network reads the column on the scale its effect is linear in")),
                   (Candidate(tid, "transform", (column,), rationale),))


def transform_step(finding: Finding, kind: str) -> dict[str, Any]:
    """The recipe step (not yet numbered) that applies `finding`'s transform candidate."""
    cand = next(c for c in finding.candidates if c.op == "transform")
    return {"id": "", "op": "transform", "columns": list(cand.columns), "params": {"kind": kind},
            "source": finding.source(cand.id), "rationale": cand.rationale,
            "consequences": {"rows": "unchanged", "columns": f"unchanged ({len(cand.columns)} transformed)",
                             "downstream": f"the scale below standardises the {kind} of "
                                           f"{', '.join(cand.columns)}"},
            "alternatives": [{"op": "transform", "params": {"kind": "yeo-johnson"},
                              "when": "if the column could hold zeros or negatives"}]}


def with_transform(recipe: dict, finding: Finding, kind: str) -> dict:
    """`recipe` with `finding`'s transform in phase 4: after the split and any lag or impute,
    before the first representation step (REPRESENTATION_OPS), the steps renumbered S01…."""
    steps = recipe["steps"]
    at = next((i for i, s in enumerate(steps) if s["op"] in REPRESENTATION_OPS), len(steps))
    steps.insert(at, transform_step(finding, kind))
    for number, step in enumerate(steps, start=1):
        step["id"] = f"S{number:02d}"
    return recipe


def finding_evidence(finding: Finding, frame: Any) -> dict[str, Any]:
    """What the raw rows say about a transform finding: each column's sample skewness, and the
    skewness of its log (so the layer states the tail, not only that the generator planted it).
    Empty for a finding without a transform candidate or a column the frame lacks."""
    import numpy as np

    cols = [c for cand in finding.candidates if cand.op == "transform" for c in cand.columns
            if frame is not None and c in frame]
    if not cols:
        return {}

    def skew(v: Any) -> float:
        v = np.asarray(v, dtype=float)
        sd = v.std()
        return round(float(((v - v.mean()) ** 3).mean() / sd ** 3), 3) if sd > 0 else 0.0

    values = {c: frame[c].astype(float).dropna().to_numpy() for c in cols}
    return {"skew": {c: skew(v) for c, v in values.items()},
            "skew_of_log": {c: skew(np.log(v[v > 0])) for c, v in values.items()}}


def spatial_context(schema: Sequence[Any], lane: Any) -> dict | None:
    """`context.spatial` (analysis-contract §1) of a row lane that declares spatial data
    (`expect.spatial`, the spatial-row modality R-E4): the generator's latitude / longitude pair,
    WGS 84. None on any other lane — and a lane that declares spatial data without a coordinate
    pair among its numeric columns is refused, since its modality would then rest on nothing."""
    raw = _raw(lane)
    if not (raw.get("expect") or {}).get("spatial"):
        return None
    numeric = {c.name.lower(): c.name for c in schema if c.role == "numeric"}
    lat = next((numeric[n] for n in LATITUDE if n in numeric), None)
    lon = next((numeric[n] for n in LONGITUDE if n in numeric), None)
    if lat is None or lon is None:
        raise ValueError(f"lane {raw.get('id')} declares spatial data but its generator's numeric "
                         f"columns hold no latitude/longitude pair ({sorted(numeric.values())})")
    return {"columns": [{"column": lat, "pair": lon, "kind": "latlon"}], "crs": "EPSG:4326"}
