"""
rules_chain — the compiler's rules on provenance and on the chain around the spec.

`compile_spec.py` registers these in `RULES` beside the task, block and sample rules
(`rules_task.py`, `rules_blocks.py`, `rules_sample.py`). Each is a pure `Context -> markers`
function that cites the contract clause it enforces:

  P0  the schema and the required keys (netspec-contract §1)
  P1  every choice carries a source that resolves (netspec-contract §1)
  B1  no framework name outside `backend` (netspec-contract §3)
  S1  the test part is sealed (netspec-contract §1)
  D1a-D1d  the blueprint: required, its feature contract, serving.mode, a posture overruled
  D1e  the spec names the blueprint it was compiled against (input.pipeline_digest)
  K1  the ports this stage needs are drawn (code-contract.md §1)
  M2  the Network stances are honoured (netspec-contract §5)
"""
from __future__ import annotations

import json
import re
from typing import Mapping

from vega_core import Choice, canonical_digest, error, ok, warn

__all__ = ["SCHEMA", "schema", "provenance", "framework_leak", "sealed_test", "blueprint_present",
           "feature_contract", "pipeline_digest", "serving_mode", "posture",
           "ports_this_stage_needs", "stances"]

SCHEMA = "vega/netspec@1"

REQUIRED_KEYS = ("input", "seed", "task", "data_contract", "family", "blocks", "loss",
                 "metrics", "optimizer", "schedule", "regularization", "early_stopping",
                 "init", "training", "backend", "outputs")

FRAMEWORK_WORDS = ("pytorch", "torch", "tensorflow", "keras", "jax", "flax")


# -- P0 schema ---------------------------------------------------------------

def schema(ctx):
    out = []
    if ctx.spec.get("schema") != SCHEMA:
        out.append(error(f"schema must be {SCHEMA!r}, got {ctx.spec.get('schema')!r}",
                         rule="P0", clause="netspec-contract §1"))
    for key in REQUIRED_KEYS:
        if key not in ctx.spec:
            out.append(error(f"missing required top-level key {key!r}", rule="P0",
                             clause="netspec-contract §1"))
    return out


# -- P1 provenance -----------------------------------------------------------

def provenance(ctx):
    out = []
    for choice in ctx.choices:
        out.extend(_one_choice(ctx, choice))
    return out


def _one_choice(ctx, choice: Choice):
    where, node = choice.where, choice.node
    if not isinstance(node, Mapping):
        return [error("must be an object carrying source, rationale and alternatives",
                      rule="P1", clause="netspec-contract §1", where=where)]
    src = choice.source
    if not src:
        return [error("carries no source; an untraced choice is the one thing the compiler "
                      "may never pass", rule="P1", clause="netspec-contract §1", where=where)]
    if not ctx.resolver.well_formed(src):
        return [error(f"source {src!r} is not one of blueprint: / shape: / geometry: / "
                      "analysis: / user: / vega:default", rule="P1",
                      clause="netspec-contract §1 (the closed set)", where=where)]

    out = []
    resolved, detail = ctx.resolver.resolve(src)
    if not resolved:
        out.append(error(f"source {src!r} does not resolve ({detail})", rule="P1",
                         clause="netspec-contract §1", where=where))
    if not choice.rationale:
        out.append(error(f"source {src!r} carries no rationale", rule="P1",
                         clause="netspec-contract §1", where=where))

    structural = choice.structural
    if src == "vega:default":
        rows = choice.catalog_rows
        if not rows:
            out.append(error("a vega:default must name the catalog row it cites "
                             "(catalog: \"<section>/<row>\")", rule="P1",
                             clause="family-catalog.md §intro", where=where))
        for row in rows:
            if ctx.catalog.row(row) is None:
                out.append(error(f"catalog row {row!r} is not in catalog.json", rule="C1",
                                 clause="family-catalog.md §1-§4", where=where))
            elif ctx.catalog.row_is_structural(row):
                structural = True

    alts = choice.alternatives
    if alts is None:
        out.append(error("alternatives is required (an empty list only for a structural "
                         "choice)", rule="P1", clause="netspec-contract §1", where=where))
    elif not alts and not structural:
        out.append(error("alternatives is empty but the catalog does not mark this choice "
                         "structural; name the alternative that was weighed and the `when` "
                         "that separates it", rule="P1", clause="netspec-contract §1",
                         where=where))
    return out


# -- B1 the spec names no framework -----------------------------------------

def framework_leak(ctx):
    outside = {k: v for k, v in ctx.spec.items() if k not in ("backend", "outputs", "input")}
    blob = json.dumps(outside)
    out = []
    for word in FRAMEWORK_WORDS:
        if re.search(rf'"[^"]*\b{word}\b[^"]*"', blob, re.I):
            out.append(warn(f"the framework name {word!r} appears outside the `backend` "
                            "section; the spec is framework-agnostic and backend.hints is the "
                            "one place such settings live", rule="B1",
                            clause="netspec-contract §3"))
    return out


# -- S1 the sealed test part -------------------------------------------------

def sealed_test(ctx):
    out = []
    parts = ctx.data_contract.get("parts") or {}
    if parts.get("test") != "sealed":
        out.append(error(f"data_contract.parts.test is {parts.get('test')!r}; it must be the "
                         "literal \"sealed\" — the test part belongs to /vega-eval, which "
                         "reads its path from the manifest when it opens it", rule="S1",
                         clause="netspec-contract §1"))
    for hit in set(re.findall(r'"test\.[A-Za-z0-9_]+"', json.dumps(ctx.spec))):
        out.append(error(f"the spec monitors {hit} — nothing at this stage may name a metric "
                         "on the test part", rule="S1", clause="netspec-contract §1"))
    if not out:
        out.append(ok("the test part is named only as \"sealed\"", rule="S1",
                      clause="netspec-contract §1"))
    return out


# -- D1 the blueprint --------------------------------------------------------

def blueprint_present(ctx):
    if ctx.inputs.pipeline:
        return []
    return [error("no blueprint pipeline.json: /vega-forge does not run without one — the "
                  "emitted code lives in the blueprint's module map. Run "
                  "`/blueprint <model> --mode pipeline` first", rule="D1",
                  clause="code-contract.md §1")]


def feature_contract(ctx):
    pipeline = ctx.inputs.pipeline
    if not pipeline:
        return []
    contract = pipeline.get("feature_contract") or {}
    out = []
    if contract.get("input_dim") is not None and \
            contract["input_dim"] != ctx.data_contract.get("input_dim"):
        out.append(error(f"the blueprint's feature_contract.input_dim is "
                         f"{contract['input_dim']} but the spec's data_contract.input_dim is "
                         f"{ctx.data_contract.get('input_dim')}", rule="D1",
                         clause="code-contract.md §6"))
    label = contract.get("label")
    if label is not None and label != ctx.task.get("label"):
        out.append(error(f"the blueprint's feature_contract.label is {label!r} but the spec's "
                         f"task label is {ctx.task.get('label')!r}", rule="D1",
                         clause="code-contract.md §6"))
    return out + _manifest_digests(ctx, pipeline)


def _manifest_digests(ctx, pipeline):
    """One manifest digest (code-contract.md §6): the shape layer's `manifest_digest` — the
    sha256 of the shaped manifest.json FILE — is what the blueprint drew against
    (`pipeline.json.input.shape_manifest_digest`) and what the spec cites
    (`input.manifest_digest`). A mismatch is an ERROR: the emitted FeatureContract raises on it
    at its first load, so a warning here would only move the failure to /vega-train."""
    recorded = (ctx.inputs.layer("shape") or {}).get("manifest_digest")
    drawn = (pipeline.get("input") or {}).get("shape_manifest_digest")
    cited = (ctx.spec.get("input") or {}).get("manifest_digest")
    out = []
    if drawn and recorded and drawn != recorded:
        out.append(error(f"blueprint older than the shape — re-run /blueprint --mode pipeline: "
                         f"pipeline.json was drawn against manifest {drawn} and the shape layer "
                         f"records {recorded}", rule="D1b", clause="code-contract.md §6"))
    elif not recorded and drawn and cited and drawn != cited:
        out.append(error(f"blueprint older than the shape — re-run /blueprint --mode pipeline: "
                         f"pipeline.json was drawn against manifest {drawn} and the spec cites "
                         f"{cited}", rule="D1b", clause="code-contract.md §6"))
    if cited and recorded and cited != recorded:
        out.append(error(f"input.manifest_digest is {cited} but the shape layer records "
                         f"{recorded}: the spec cites another shaping — copy the shape layer's "
                         "manifest_digest", rule="D1b", clause="netspec-contract §1"))
    return out


# -- D1e the blueprint the spec was compiled against ------------------------------

def pipeline_digest(ctx):
    """`input.pipeline_digest` is the canonical digest of the pipeline.json compiled against
    (netspec-contract §1): the emitter's digest, stamped into blueprint-map.json, so the spec,
    the map and the bundle name one blueprint by one number."""
    pipeline = ctx.inputs.pipeline
    if not pipeline:
        return []
    actual = canonical_digest(pipeline)
    stamped = (ctx.spec.get("input") or {}).get("pipeline_digest")
    how = ("the sha256 of the parsed JSON with sorted keys and no whitespace, "
           "json.dumps(doc, sort_keys=True, separators=(',', ':')) — not of the file's bytes")
    if not stamped:
        return [error(f"input.pipeline_digest is missing: it names the blueprint this spec was "
                      f"compiled against, {actual} ({how})", rule="D1e",
                      clause="netspec-contract §1")]
    if stamped != actual:
        return [error(f"input.pipeline_digest is {stamped} but the pipeline.json compiled "
                      f"against digests to {actual} ({how}): the spec was written against "
                      "another blueprint, or its digest was computed another way",
                      rule="D1e", clause="netspec-contract §1")]
    return [ok(f"input.pipeline_digest names the blueprint compiled against ({actual[:19]}…)",
               rule="D1e", clause="netspec-contract §1")]


def serving_mode(ctx):
    pipeline = ctx.inputs.pipeline or {}
    serving = pipeline.get("serving") or {}
    mode = serving.get("mode")
    out = []
    if mode == "stream" and ctx.sample.get("shape") != "window":
        out.append(error("the blueprint's serving.mode is 'stream', which requires a windowed "
                         f"sample; data_contract.sample.shape is "
                         f"{ctx.sample.get('shape')!r}", rule="D1",
                         clause="code-contract.md §6"))
    latency = serving.get("latency_ms") or serving.get("latency")
    heavy = (ctx.family_row or {}).get("heavy")
    if mode == "online" and latency and heavy:
        cited = "blueprint:serving" in json.dumps(ctx.spec.get("family") or {})
        if not cited:
            out.append(error(f"serving.mode is 'online' with a stated latency ({latency}) and "
                             f"the family is {ctx.family_name!r}, which the catalog marks "
                             "heavy: that force must be quoted in family.rationale with a "
                             "blueprint:serving source, or answered by a FORK: — a heavy "
                             "family under a latency budget is not a silent default",
                             rule="D1", clause="family-catalog.md §2"))
    return out


def posture(ctx):
    pipeline = ctx.inputs.pipeline or {}
    posture = pipeline.get("posture")
    name = posture.get("family") if isinstance(posture, Mapping) else posture
    if not name or str(name).lower() not in ("tree", "none yet", "none", "linear"):
        return []
    family = ctx.spec.get("family") or {}
    source = str(family.get("source") or "")
    if source.startswith("user:"):
        return []
    if source == "vega:default" and "posture" in str(family.get("rationale") or "").lower():
        return []
    if source == "blueprint:posture":
        return [error(f"family.source is blueprint:posture but the posture is {name!r} and "
                      f"the spec's family is {ctx.family_name!r}: citing the posture as the "
                      "source for a family it did not name attributes the choice to the "
                      "blueprint. Overruling a posture is legitimate — it is a user: source, "
                      "or a vega:default whose rationale quotes the forces being overruled — "
                      "but it is not the blueprint's decision", rule="D1",
                      clause="family-catalog.md §2")]
    return [error(f"the blueprint's posture is {name!r} and the spec declares a network "
                  "anyway: family.source must be a user: source (the user overruled the "
                  "posture) or a vega:default whose rationale quotes the posture's forces",
                  rule="D1", clause="family-catalog.md §2")]


def ports_this_stage_needs(ctx):
    if not ctx.inputs.pipeline:
        return []
    roles = ctx.inputs.port_roles
    return [error(f"the blueprint draws no port with role {role!r}, which this stage needs. "
                  "The emitter does not invent a port — run `/blueprint <model> --mode "
                  f"pipeline --concern \"the {role} port\"`", rule="K1",
                  clause="code-contract.md §1")
            for role in ("DataSource", "Model", "Trainer", "ModelStore") if role not in roles]


# -- M2 memory stances -------------------------------------------------------

def stances(ctx):
    if not ctx.inputs.stances:
        return []
    blob = " ".join(ctx.inputs.stances).lower()
    family = str(ctx.family_name or "").lower()
    loss = str((ctx.spec.get("loss") or {}).get("name") or "")
    out = []
    # a stance is a `key: value` pair of a Network stances line (netspec-contract §5:
    # `declined: ft-transformer (...)`, `loss: bce_with_logits ...`); a word inside prose — an
    # eval line's `log_loss 0.2995`, a reading that says a fit "declined" — is not one
    for match in re.finditer(r"(?<![a-z0-9_])declin\w*\s*(?:family)?\s*:\s*([a-z0-9_\-]+)", blob):
        if match.group(1) == family and not str(
                (ctx.spec.get("family") or {}).get("source", "")).startswith("user:"):
            out.append(error(f"memory records that the family {ctx.family_name!r} was "
                             "declined; a spec that uses it anyway needs a user: source "
                             "saying the user changed their mind", rule="M2",
                             clause="netspec-contract §5"))
    for match in re.finditer(r"(?<![a-z0-9_])loss\s*:\s*([a-z0-9_]+)", blob):
        if loss and match.group(1) != loss.lower() and not str(
                (ctx.spec.get("loss") or {}).get("source", "")).startswith("user:"):
            out.append(error(f"memory records the declared loss {match.group(1)!r}; this spec "
                             f"uses {loss!r} without a user: source overriding the stance",
                             rule="M2", clause="netspec-contract §5"))
            break
    return out
