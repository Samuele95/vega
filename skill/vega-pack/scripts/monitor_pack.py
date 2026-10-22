"""
monitor_pack — the DriftWatch adapter, the reference it reads, and the smoke that holds it to
its promise (bundle-contract §7.5).

    monitor_pack.refuse_unfilled(options) -> None                      PackExit(2) before writing
    monitor_pack.render(ctx, predictor) -> {"monitor": …} or {}     PackExit(2) on a bad plan
    monitor_pack.validate_plan(plan, output, schema) -> [problem, …]    (monitor_plan's)
    monitor_pack.output_check(output) -> the automatic output check, or None

The checks are the packer's judgement — which measurement answers which concern the blueprint
wrote in prose — and arrive in `--monitor-plan`; every `pipeline.json` monitoring concern must
be answered by at least one check, or nothing is written. `monitor_plan.validate_plan` refuses a
plan the bundle cannot honour — a kind its reading does not admit, a column, field, class or
label that is not the bundle's, a parameter the monitor cannot read — with exit 2, before a
single reference is measured. Pack then adds one check of its own on the answer,
`[vega-pack:output-drift]`: an `output_psi` on `MANIFEST.output.score_field` where the reading
has one, else the decided classes' mix, else the labels' rates — so every reading gets a
task-aware check even when the blueprint's concerns are all input-side.

The numbers are `monitor_reference`'s: the input references on the train part, the unseen rate
and every answer reference on the valid part, never the test part. The adapter is the backend's
template (`templates/serving/package/monitor.py`) rendered into the Monitor port's file; the
smoke loads it through the bundle's own `load_monitor`. A blueprint whose Monitor port says
`"filled_by": "/vega-pack"` names this stage as the one that fills it, so without a plan pack
refuses before a file is written (`refuse_unfilled`, exit 2): a bundle that left the port its
blueprint gave it unfilled is the divergence the gate exists to catch, and it is caught here,
mechanically. Only a blueprint that does not say so (or no `--pipeline`) leaves the Monitor the
forge's pending stub without a plan, and the map says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bundle_manifest import refresh_stub
from monitor_plan import output_check, validate_plan
from monitor_reference import MonitorInputs, MonitorRender, category_text, resolve_check
from pack_context import PackExit, out_ok, out_warn
from vega_pack_core import read_json

__all__ = ["output_check", "refuse_unfilled", "render", "validate_plan"]

TEMPLATE = "package/monitor.py"
FILLER = "/vega-pack"


def refuse_unfilled(options) -> None:
    """Refuse, before anything is written, a pack that would leave unfilled a Monitor port the
    blueprint says this stage fills (`ports[].filled_by: /vega-pack`): the plan is the
    packer's judgement (SKILL step 1c) and the blueprint's monitoring concerns are what it
    answers, unattended too."""
    if options.monitor_plan or not options.pipeline:
        return
    pipeline, err = read_json(Path(options.pipeline), "the blueprint's pipeline.json")
    if err or not isinstance(pipeline, dict):
        return                       # the reading of the pipeline itself is refused elsewhere
    ports = [p for p in pipeline.get("ports") or [] if isinstance(p, dict)
             and p.get("role") == "Monitor" and p.get("filled_by") == FILLER]
    if not ports:
        return
    concerns = [str(m.get("source") or m.get("name")) for m in pipeline.get("monitoring") or []
                if isinstance(m, dict)]
    raise PackExit(2, (
        f"the blueprint's {ports[0].get('name') or 'Monitor'} port (role Monitor) says "
        f"filled_by {FILLER}, and no --monitor-plan was given: a bundle that leaves the port its "
        "blueprint gave this stage unfilled ships the stub the gate reads as a divergence",
        "write <bundle dir>/../monitor-plan.json — one check per pipeline.json monitoring "
        f"concern ({', '.join(concerns) or 'none named: at least one check on the answer'}), of a "
        "kind the bundle's reading admits (bundle-contract §7.5) — and pass it as "
        "--monitor-plan; refused before the bundle was written"))


def render(ctx, predictor) -> dict:
    reference = _render_monitor(ctx, predictor)
    if reference is None:
        return {}
    # the reference is pinned before the smoke reads it, so load_monitor verifies it as it
    # will in the finished bundle
    refresh_stub(ctx)
    smoke = monitor_smoke(ctx, predictor, reference["inputs"])
    document = reference["document"]
    return {"monitor": {"reference": "monitor-reference.json", "schema": document["schema"],
                        "checks": len(document["checks"]),
                        "kinds": sorted({c["kind"] for c in document["checks"]}),
                        "usage": "from predict import load_monitor; "
                                 "load_monitor().check(window, predictions)",
                        "smoke": smoke}}


def _render_monitor(ctx, predictor):
    """The reference written and the adapter rendered, or None when there is nothing to render;
    a plan the bundle cannot honour is a `PackExit`."""
    port = ctx.port("Monitor")
    if not port.get("file"):
        out_warn("the map names no Monitor adapter; nothing to render", ctx.markers)
        return None
    if not ctx.options.monitor_plan:
        out_warn("no --monitor-plan: the Monitor stays the forge's pending stub and the map "
                 "says so", ctx.markers)
        return None
    plan, err = read_json(Path(ctx.options.monitor_plan), "the monitor plan")
    if err or not isinstance(plan, dict):
        raise PackExit(2, err or "the monitor plan is not a JSON object with its checks")
    stub = json.loads(ctx.paths.manifest.read_text(encoding="utf-8"))
    output = stub["output"]
    problems = validate_plan(plan, output, ctx.schema)
    if problems:
        raise PackExit(2, tuple(f"monitor plan: {p}" for p in problems))
    concerns = _concerns(ctx, plan)
    if not len(ctx.campaign.part_rows("train")):
        raise PackExit(2, "fitted.json records no train rows; the references cannot be measured")
    inputs = MonitorInputs.of(ctx, predictor, stub.get("sample"))
    document = _measured(ctx, inputs, output, plan)
    document.update(concerns=concerns, plan=Path(ctx.options.monitor_plan).name)
    (ctx.paths.root / "monitor-reference.json").write_text(
        json.dumps(document, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    _render_adapter(ctx, port, TEMPLATE, len(document["checks"]), len(concerns))
    return {"document": document, "inputs": inputs}


def _measured(ctx, inputs, output, plan) -> dict:
    """The reference document: the plan's checks and pack's output check, measured."""
    checks = [resolve_check(c, output) for c in plan["checks"]]
    render = MonitorRender(output, ctx.schema, inputs)
    return render.render(dict(plan, checks=_with_output_check(checks, output, ctx.markers)))


def _concerns(ctx, plan) -> list:
    """The blueprint's monitoring concerns, each answered by a check of the plan."""
    concerns = [m.get("source") for m in (ctx.pipeline().get("monitoring") or [])]
    answered = {c.get("concern") for c in plan["checks"] if isinstance(c.get("concern"), str)}
    unanswered = [c for c in concerns if c not in answered]
    if unanswered:
        raise PackExit(2, "the monitor plan answers no check to the blueprint concern(s) "
                          + ", ".join(map(str, unanswered)) + "; a monitor that skips what the "
                          "analysis found fragile is not the blueprint's DriftWatch")
    return concerns


# --------------------------------------------------------------------------- the output check

def _with_output_check(checks, output, markers) -> list:
    auto = output_check(output)
    if auto is None:
        out_warn(f"the {output['reading']} reading admits no output check; the monitor watches "
                 "the plan's checks only", markers)
        return checks
    same = [c.get("id") for c in checks if c.get("kind") == auto["kind"]
            and c.get("field") == auto.get("field") and c.get("labels") == auto.get("labels")]
    if same:
        out_ok(f"the plan's check {same[0]} already watches the output ({auto['kind']}); pack "
               "adds no second one", markers)
        return checks
    return checks + [auto]


# --------------------------------------------------------------------------- the adapter

def _render_adapter(ctx, port, template, n_checks, n_concerns):
    path = ctx.templates / template
    if not path.is_file():
        raise PackExit(2, f"the {ctx.campaign.backend_name} backend ships no monitor template "
                          f"at {path}")
    text = path.read_text(encoding="utf-8")
    for key, value in (("{{PACKAGE}}", ctx.package),
                       ("{{MONITOR_MODULE}}", port.get("module") or "serving"),
                       ("{{PORT_MONITOR}}", port.get("name") or "Monitor"),
                       ("{{RESPONSIBILITY_MONITOR}}", port.get("responsibility") or ""),
                       ("{{SOURCE_MONITOR}}", f"blueprint:ports/{port.get('name') or 'Monitor'}")):
        text = text.replace(key, value)
    if "{{" in text:
        left = sorted({part.split("}}")[0] for part in text.split("{{")[1:]})
        raise PackExit(2, f"the monitor template keeps placeholder(s) {', '.join(left)} pack "
                          "does not fill")
    (ctx.paths.pkg / port["file"]).write_text(text, encoding="utf-8")
    out_ok(f"{port['file']} rendered from {template} — {port.get('name')} over {n_checks} "
           f"check(s) answering all {n_concerns} blueprint concern(s) and the output; references "
           "measured on train (the unseen rate and every answer reference on valid), never on "
           "test", ctx.markers)


# --------------------------------------------------------------------------- the smoke

def monitor_smoke(ctx, predictor, inputs, rows=5000, seed=7) -> dict:
    """The DriftWatch port promises an empty tuple when nothing moved. The pack holds the
    rendered monitor to it, through the bundle's own load_monitor, on the distributions its
    references were measured on: a window of the train part read on its inputs (the answer
    references are the valid part's, and a trained network answers its own training rows
    differently), and a window of the valid part with the truth the service takes, predicted by
    the bundle's own predictor. Random rows for a row sample; a contiguous run in the part's
    order otherwise, so windows and graphs stay whole. A reading that moves here is information
    about the plan's thresholds, and is said so."""
    import numpy as np
    module = sys.modules.get("predict")
    if module is None or not (ctx.paths.root / "monitor-reference.json").is_file():
        return {"run": False, "why": "no monitor, or no loaded bundle"}
    try:
        monitor = module.load_monitor(ctx.paths.root)
    except Exception as exc:                                          # noqa: BLE001
        raise PackExit(3, f"the bundle's load_monitor() does not build the monitor it ships: "
                          f"{type(exc).__name__}: {exc}") from exc
    rng, whole = np.random.default_rng(seed), (inputs.sample or {}).get("shape", "row") == "row"
    result = {"run": True, "rows": rows, "seed": seed, "moved": {}}
    for part, raw in (("train", inputs.train), ("valid", inputs.valid)):
        if not len(raw):
            continue
        n = min(rows, len(raw))
        pick = rng.choice(len(raw), n, replace=False) if whole else \
            range(start := int(rng.integers(0, len(raw) - n + 1)), start + n)
        window = raw.iloc[list(pick)].reset_index(drop=True)
        moved = _smoke_window(ctx, window, predictor if part == "valid" else None, monitor)
        result["moved"][part] = [f"{r.column}:{r.measure}={r.value} ({r.verdict})"
                                 for r in moved]
    _say_smoke(result, rows, ctx.markers)
    return result


def _smoke_window(ctx, window, predictor, monitor):
    """One window through `check`: its inputs alone, or (with a predictor) with the truth
    and the Predictor's frame."""
    stub = json.loads(ctx.paths.manifest.read_text(encoding="utf-8"))
    truth = stub["output"].get("truth") or {}
    features = window.drop(columns=[c for c in (ctx.schema.get("ignored") or [])
                                    if c in window.columns])
    try:
        if predictor is None:
            moved = monitor.check(features)
        else:
            preds = predictor.predict(features)
            moved = monitor.check(features.assign(**_truth_values(ctx, window, truth)), preds)
    except Exception as exc:                                          # noqa: BLE001
        raise PackExit(3, f"monitor smoke: {type(exc).__name__}: {exc}") from exc
    if moved is None or not isinstance(moved, tuple):
        raise PackExit(3, f"the monitor's check() returned {type(moved).__name__}, not a "
                          "tuple; the DriftWatch port promises an empty tuple")
    return moved


def _truth_values(ctx, window, truth) -> dict:
    """The truth column as the service passes it into a window: a class decoded as the schema
    decodes it, a number as a number, labels as sent."""
    column = truth.get("column")
    if not column or column not in window.columns:
        return {}
    if truth.get("type") == "class":
        decode = (ctx.schema.get("label") or {}).get("decode") or {}
        return {column: category_text(window[column]).map(lambda v: decode.get(v, v))}
    return {column: window[column]}


def _say_smoke(result, rows, markers):
    loud = {k: v for k, v in result["moved"].items() if v}
    if loud:
        out_warn("monitor smoke: readings move on in-distribution windows — "
                 + "; ".join(f"{k}: {', '.join(v)}" for k, v in loud.items())
                 + ". The plan's thresholds are stricter than the data's own variation", markers)
    else:
        out_ok(f"monitor smoke: check() is empty on a {rows}-row window of the train part (its "
               "inputs) and of the valid part (with the truth, predicted) — silent where "
               "nothing moved", markers)
