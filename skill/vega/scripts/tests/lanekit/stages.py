"""
A lane's stages: one function per stage invocation (D6 §6), `run_<stage>(lane, fx, ctx) -> str`.

Three drivers (lanes.json `driver`) choose the plan:

- `acceptance` (L01): forge, train, eval and pack are the four existing acceptance scripts, run
  on one work dir with `PACK_EXTRA=--serve` (D6 §2.3); the lane then continues like any other
  from `bundle_api`;
- `lanekit`: `fixture` builds the lane's data, model and netspec (lanekit.fixture, T02), and each
  later stage drives the same scripts the acceptance scripts drive, with the lane's own paths;
- `golden` (G00): `golden/cic.py` re-packs the recorded CIC campaign and builds its showcase
  (R2); the service stages then run as for any lane.

A stage fails on a non-zero exit, an `ERROR:` marker, a `WARN:` over the lane's budget, or a
failed assertion (StageFailure). The order, the stamps and `--from` are the driver's
(lanekit.lane); the assertions that read parquet run in their own process (lanekit.assertions).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import contracts, service as svc, stage_briefs
from .context import (SERVICE_STAGES, StageContext, StageFailure, assert_packed_blocks,
                      count_inner_calls, find_bundle, flake_record_path, markers_budget,
                      record_flakes_seen, record_inner_flakes, record_pin)
from .registry import Lane

PLANS = {"acceptance": ("forge", "train", "eval", "pack", "bundle_api", "report", "show",
                        "serve", "serve_check", "inspect", "page"),
         "lanekit": contracts.STAGES,
         "golden": (contracts.GOLDEN_STAGE,) + SERVICE_STAGES}
SHIP_STANCE = "lane {id}: the matrix ships every lane to prove the chain end to end (D6 §4)"


def stage_plan(lane: Lane) -> tuple[str, ...]:
    plan = PLANS[lane.driver]
    last = lane.last_stage
    return plan[: plan.index(last) + 1] if last in plan else plan


def missing_provider(stage: str, skills: Path, driver: str) -> tuple[str, str] | None:
    """(path, work package) of the first file the stage drives that is not in the tree."""
    if driver == "acceptance" and stage in contracts.ACCEPTANCE:
        needs = ((contracts.ACCEPTANCE[stage], "vega"),)
    else:
        needs = contracts.PROVIDERS.get(stage, ())
    return next(((rel, wp) for rel, wp in needs if not (skills / rel).exists()), None)


def shares_for(lane: Lane, defaults: Mapping[str, Any]) -> Mapping[str, float]:
    key = {"acceptance": "acceptance_shares", "golden": "golden_shares"}.get(lane.driver, "stage_shares")
    return defaults[key]


# --------------------------------------------------------------------------- acceptance (L01)
def run_acceptance(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """One of the four acceptance scripts, on the lane's work dir (D6 §2.3). The script is
    never re-run whole: its validate_model.py calls go through X1's known_flake.py, whose
    retries land in the record file and from there in observed.json."""
    script = ctx.path(contracts.ACCEPTANCE[ctx.stage])
    record = flake_record_path(ctx)
    extra = {contracts.FLAKE_RECORD_ENV: str(record), contracts.ACCEPTANCE_SUITES[0]: contracts.ACCEPTANCE_SUITES[1]}
    if ctx.stage == "pack":
        extra[contracts.PACK_EXTRA] = "--serve"
    count_inner_calls(ctx, script)
    try:
        result = ctx.run(["bash", script, ctx.work], retry=False, env_extra=extra)
    finally:
        recorded = record_inner_flakes(ctx, record)
    if "acceptance passed" not in result.output:
        raise StageFailure(f"{script.parent.parent.parent.name} acceptance did not reach 'acceptance passed'")
    if not recorded:
        record_flakes_seen(ctx, result)
    detail = f"{script.parent.parent.parent.name} acceptance passed ({result.markers.warn} WARN)"
    if ctx.stage == "pack":
        detail += "; " + assert_packed_blocks(lane, ctx.work)
        record_pin(ctx)
    return detail


# --------------------------------------------------------------------------- lanekit stages
def _fx(ctx: StageContext) -> dict:
    fx = ctx.state.fixture
    if not fx:
        raise StageFailure("no fixture.json for this lane (run the fixture stage first)")
    return fx


def _layered(fx: Mapping[str, Any], suffix: str) -> Path:
    base = Path(fx["model_html"])
    return base.with_name(base.name[: -len(".html")] + suffix + ".html")


def run_fixture(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """lanekit.fixture.make_lane (T02), then the fixture's assertions (D6 §6.1)."""
    out = ctx.lane_dir / "fixture.json"
    ctx.helper("fixture", ["--out", out])
    ctx.state.fixture = json.loads(out.read_text(encoding="utf-8"))
    ctx.helper("check-fixture")
    for model in sorted(ctx.work.glob("*.model-source/*.domain.html")) + [Path(ctx.state.fixture["model_html"])]:
        ctx.validate_layer(model)
    return f"fixture built: {Path(ctx.state.fixture['model_html']).name}"


def run_forge(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """compile, emit + smoke, conformance, notebook, layer, validate, smoke (D6 §6.2)."""
    fx = _fx(ctx)
    forge, run_dir, pipe = "vega-forge/scripts/", Path(fx["run_dir"]), fx["pipeline"]
    compile_out = ctx.run(ctx.py(forge + "compile_spec.py", ["--spec", fx["netspec"], "--model",
                                 fx["model_html"], "--pipeline", pipe, "--memory",
                                 ctx.work / ".claude/domain-forge-memory.md", "--check-only"]))
    markers_budget(ctx, compile_out)
    ctx.run(ctx.py(forge + "emit_backend.py", ["--spec", fx["netspec"], "--pipeline", pipe,
                   "--out", run_dir, "--smoke"]))
    ctx.run(ctx.py(forge + "conformance.py", ["--package", run_dir / fx["package"], "--pipeline", pipe]))
    shutil.copy2(fx["netspec"], run_dir / "netspec.json")
    ctx.helper("author", ["--kind", "forge", "--target", run_dir / "architecture-layer.json"])
    notebook = _notebook(ctx, "forge", run_dir, run_dir / "engineer.ipynb")
    arch = sorted((ctx.work / ".claude" / "architect-runs").glob("*/"))
    improve = sorted((ctx.work / ".claude" / "improve-runs").glob("*/"))
    out = _layered(fx, ".vega")
    ctx.run(ctx.py(forge + "apply_architecture_layer.py", [fx["model_html"], "--data",
                   run_dir / "architecture-layer.json", "--notebook", notebook,
                   *(["--architect-run", arch[-1]] if arch else []),
                   *(["--improve-run", improve[-1]] if improve else []), "--out", out]))
    ctx.validate_layer(out)
    ctx.run(ctx.py(forge + "smoke_architecture.py", [out, "--strict"]))
    ctx.helper("check-forge")
    return f"forged {fx['package']}; {out.name}"


def _notebook(ctx: StageContext, stage: str, source: Path, out: Path) -> Path:
    """nb_emit.py for the stage; executed in place when the lane's profile is full."""
    ctx.run(ctx.py("vega-forge/scripts/nb_emit.py", ["--stage", stage, "--from", source, "--out", out]))
    if ctx.lane.get("profile") == "full":
        ctx.run([ctx.menv.python, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                 "--ExecutePreprocessor.timeout=1800", out.name], cwd=out.parent)
    return out


def run_train(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """plan, rungs, study, selection, notebook, layer, validate, smoke (D6 §6.3)."""
    fx = _fx(ctx)
    tr, camp = "vega-train/scripts/", Path(fx["run_dir"])
    campaign = lane.campaign
    hpo = ["--hpo", campaign["hpo_trials"]] if campaign["hpo_trials"] else ["--no-hpo"]
    model = _layered(fx, ".vega")
    ctx.run(ctx.py(tr + "plan_ladder.py", ["--campaign", camp, "--model", model, "--budget", 2,
                   "--unattended", *hpo]))
    planned = {r.get("id") for r in json.loads((camp / "ladder.json").read_text(encoding="utf-8")).get("runs", [])}
    for rung in campaign["rungs"]:
        if rung in planned:
            ctx.run(ctx.py(tr + "trainer.py", ["--campaign", camp, "--run", rung, "--budget-min", 0.5]))
    if campaign["hpo_trials"]:
        ctx.run(ctx.py(tr + "hpo.py", ["--campaign", camp, "--study", "H1", "--budget",
                       f"trials={campaign['hpo_trials']},minutes=2"]))
    ctx.run(ctx.py(tr + "select.py", ["--campaign", camp]))
    ctx.helper("author", ["--kind", "train", "--target", camp / "training-layer.json"])
    notebook = _notebook(ctx, "train", camp, camp / "train.ipynb")
    out = _layered(fx, ".vega.train")
    ctx.run(ctx.py(tr + "apply_training_layer.py", [model, "--data", camp / "training-layer.json",
                   "--notebook", notebook, "--out", out]))
    ctx.validate_layer(out)
    ctx.run(ctx.py(tr + "smoke_training.py", [out, "--strict"]))
    ctx.helper("check-train")
    return f"trained {', '.join(r for r in campaign['rungs'] if r in planned)}; {out.name}"


def run_eval(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """the sealed look, authoring, notebook, layer, validate, smoke (D6 §6.4); a transductive
    lane then asserts pack's refusal (R-A10)."""
    fx = _fx(ctx)
    ev, camp = "vega-eval/scripts/", Path(fx["run_dir"])
    ctx.run(ctx.py(ev + "evaluate.py", ["--campaign", camp, "--n-boot", ctx.menv.defaults["n_boot"],
                   "--unattended"]))
    run = json.loads((camp / "test_seal.json").read_text(encoding="utf-8"))["run"]
    ctx.run(ctx.py(ev + "tests/write_authoring.py", [camp, run, camp / "evaluation-layer.json"]))
    notebook = _notebook(ctx, "eval", camp, camp / "evaluate.ipynb")
    out = _layered(fx, ".vega.train.eval")
    ctx.run(ctx.py(ev + "apply_evaluation_layer.py", [_layered(fx, ".vega.train"), "--data",
                   camp / "evaluation-layer.json", "--notebook", notebook, "--out", out]))
    ctx.validate_layer(out)
    ctx.run(ctx.py(ev + "smoke_evaluation.py", [out, "--strict"]))
    ctx.helper("check-eval")
    refusal = lane.expect.get("pack_refusal")
    return f"evaluated {run}; {out.name}" + ("; " + _assert_pack_refusal(ctx, fx, refusal) if refusal else "")


def _pack_argv(ctx: StageContext, fx: Mapping[str, Any], bundle_dir: Path, plan: Path | None = None) -> list[str]:
    lane = ctx.lane
    return ctx.py("vega-pack/scripts/pack.py", ["--campaign", fx["run_dir"], "--shaped", fx["shaped_dir"],
                  "--bundle-dir", bundle_dir, "--pipeline", fx["pipeline"], "--model",
                  _layered(fx, ".vega.train.eval"), "--ship-anyway", SHIP_STANCE.format(id=lane.id),
                  "--serve", "--unattended", "--row-noun", lane.lexicon["row"]["one"],
                  "--service-title", lane.get("service_title") or lane.title,
                  *(["--monitor-plan", plan] if plan else [])])


MONITOR_PLAN = "monitor-plan.json"


def _monitor_plan(ctx: StageContext, fx: Mapping[str, Any]) -> Path:
    """The plan the lane packs with (vega-pack's harness plan: the answer watched for every
    blueprint concern). The blueprint's Monitor port says filled_by /vega-pack, and pack refuses
    to leave it unfilled (bundle-contract §7.5)."""
    plan = ctx.work / MONITOR_PLAN
    ctx.run(ctx.py("vega-pack/scripts/tests/monitor_plans.py", ["--campaign", fx["run_dir"], "--pipeline",
                   fx["pipeline"], "--out", plan]))
    return plan


def _assert_pack_refusal(ctx: StageContext, fx: Mapping[str, Any], refusal: Mapping[str, Any]) -> str:
    result = ctx.run(_pack_argv(ctx, fx, ctx.work / "bundle-refused"), retry=False, check=False)
    if result.returncode != refusal["exit"] or refusal["needle"] not in result.output:
        raise StageFailure(f"pack.py exited {result.returncode} without {refusal['needle']!r}; "
                           f"the lane expects exit {refusal['exit']} and that refusal")
    # the refusal's own ERROR: lines are the outcome the lane expects, not a defect of the stage
    ctx.runner.markers.error -= result.markers.error
    ctx.note(f"OK: pack.py refused as expected (exit {result.returncode}, {refusal['needle']!r})")
    return f"pack refused (exit {result.returncode})"


def run_pack(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """pack, card, notebook, layer, validate, smoke (D6 §6.5); records the bundle pin."""
    fx = _fx(ctx)
    pk = "vega-pack/scripts/"
    ctx.run(_pack_argv(ctx, fx, ctx.work / "bundle", _monitor_plan(ctx, fx)), retry=False)
    bundle = find_bundle(ctx.work)
    ctx.helper("author", ["--kind", "pack", "--target", bundle / "bundle-layer.json"])
    ctx.run(ctx.py(pk + "model_card.py", ["--bundle", bundle, "--model", _layered(fx, ".vega.train.eval"),
                   "--intended-use", bundle / "bundle-layer.json", "--out", bundle / "model-card.html"]))
    notebook = _notebook(ctx, "pack", bundle, bundle / "predict.ipynb")
    out = _layered(fx, ".vega.train.eval.pack")
    ctx.run(ctx.py(pk + "apply_bundle_layer.py", [_layered(fx, ".vega.train.eval"), "--data",
                   bundle / "bundle-layer.json", "--notebook", notebook, "--out", out]))
    ctx.validate_layer(out)
    ctx.run(ctx.py(pk + "smoke_bundle.py", [out, "--strict"]))
    ctx.helper("check-pack")
    detail = assert_packed_blocks(lane, ctx.work)
    record_pin(ctx)
    return f"packed {bundle.name}; {detail}"


# --------------------------------------------------------------------------- every driver
SKIP_LINE = re.compile(r"^SKIPPED \[\d+\] [^:]+(?::\d+)?: (.*)$", re.M)


def run_bundle_api(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """`predict.py --check` and the bundle's own tests in an empty environment, with the skip
    policy: every skip reason allowed, none forbidden (D6 §6.6)."""
    bundle = find_bundle(ctx.work)
    home = Path(tempfile.mkdtemp(prefix="vega-home-"))
    env = ["env", "-i", f"PATH={Path(ctx.menv.python).parent}:/usr/bin:/bin", f"HOME={home}",
           "PYTHONDONTWRITEBYTECODE=1", "CUDA_VISIBLE_DEVICES="]
    try:
        ctx.run([*env, ctx.menv.python, "predict.py", "--check"], cwd=bundle)
        tests = ctx.run([*env, ctx.menv.python, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                         "tests/"], cwd=bundle, retry=False)
    finally:
        shutil.rmtree(home, ignore_errors=True)
    reasons = SKIP_LINE.findall(tests.output)
    bad = skips_outside_policy(reasons, ctx.menv.defaults, lane.expect.get("skips") or (),
                               packed_without_inspection(bundle))
    if bad:
        raise StageFailure(f"{len(bad)} skip(s) outside the policy: {bad[:3]}")
    return f"predict.py --check and tests/ green ({len(reasons)} allowed skip(s))"


def skips_outside_policy(reasons: Sequence[str], defaults: Mapping[str, Any],
                         lane_skips: Sequence[str], no_inspect: bool) -> list[str]:
    """The skip reasons the policy refuses (D6 §6.6): any with a forbidden pattern, any that no
    allowed reason (lanes.json allowed_skips, the lane's own `skips`) covers. The --no-inspect
    reason is allowed only on a bundle packed with --no-inspect."""
    allowed = [a for a in defaults["allowed_skips"] if a != contracts.NO_INSPECT_SKIP or no_inspect]
    allowed += list(lane_skips)
    bad = [r for r in reasons if any(p in r for p in defaults["forbidden_skip_patterns"])]
    return bad + [r for r in reasons if r not in bad and not any(a in r for a in allowed)]


def packed_without_inspection(bundle: Path) -> bool:
    """The bundle's MANIFEST says it was packed with --no-inspect (serve.inspection.enabled is
    false): only then may its tests skip the inspection routes."""
    node: Any = json.loads((bundle / contracts.BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    for key in contracts.INSPECTION_OFF["manifest"]:
        node = node.get(key) if isinstance(node, dict) else None
    return node is contracts.INSPECTION_OFF["manifest_off"]


def run_report(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """A stub report (main.aux, main.pdf, label-roles.json) or none (D6 §6.7, R-E7). With none,
    the stage briefs the earlier stages' models would have left (lanekit.stage_briefs), which the
    show's brief links need (PF6)."""
    if lane.get("report") != "stub":
        written = stage_briefs.write(ctx.work, find_bundle(ctx.work), lane.id)
        if not written:
            raise StageFailure("no report and no stage brief written: the show's brief links would have nothing to link")
        return (f"no report: the show stage takes its WARN path (brief links to {len(written)} stage brief(s): "
                f"{', '.join(p.name for p in written)})")
    ctx.helper("report", ["--docs", ctx.lane_dir / "docs"])
    return "stub report written to docs/"


def run_show(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """facts, draft, authoring, resolve, build (en and qps-ploc), layer, validate, smoke (D6
    §6.8), then the eight showcase assertions."""
    bundle = find_bundle(ctx.work)
    plan_dir = ctx.lane_dir / "show" / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    fields = _show_fields(ctx, bundle, plan_dir)
    for step in ("facts", "draft"):
        ctx.run(_cli(ctx, contracts.SHOW_CLI[step], fields))
    ctx.helper("author", ["--kind", "show", "--target", plan_dir / "story-plan.json",
               "--facts", plan_dir / "facts.json", "--draft", plan_dir / "story-plan.draft.json"])
    ctx.run(_cli(ctx, contracts.SHOW_CLI["resolve"], fields))
    ctx.run(_cli(ctx, contracts.SHOW_CLI["build"], fields))
    # the pseudo-locale build: build_showcase routes it into showcase/.work/qps/, never served (R-F1)
    ctx.run(_cli(ctx, contracts.SHOW_CLI["build"], dict(fields, locale=contracts.PSEUDO_LOCALE)))
    if not (bundle / contracts.SHOWCASE_DIR / contracts.SHOW_PSEUDO_DIR).is_dir():
        raise StageFailure(f"the {contracts.PSEUDO_LOCALE} build is not in showcase/{contracts.SHOW_PSEUDO_DIR}")
    ctx.run(_cli(ctx, contracts.SHOW_CLI["check"], fields))
    _layer_authoring(Path(fields["showcase"]))
    ctx.run(_cli(ctx, contracts.SHOW_CLI["layer"], fields))
    ctx.validate_layer(Path(fields["out"]))
    ctx.run(_cli(ctx, contracts.SHOW_CLI["smoke"], fields))
    ctx.helper("check-show", ["--pin", ctx.lane_dir / "results" / "bundle_pin.json"])
    return f"showcase built and checked in {bundle.name}/{contracts.SHOWCASE_DIR}"


def _layer_authoring(showcase: Path) -> None:
    """The layer's authoring file (SKILL step 8): the lane's abstract, and every WARN line of the
    stage's markers.json verbatim (the script prefix dropped), then the not-checked marker."""
    recorded = json.loads((showcase / ".work" / "markers.json").read_text(encoding="utf-8"))
    found = (re.fullmatch(r"(?:[\w.-]+\.py: )?(WARN: .*)", line, re.S) for line in recorded if isinstance(line, str))
    warns = [m.group(1) for m in found if m]
    path = Path(contracts.SHOW_LAYER_DATA.format(showcase=showcase))
    path.write_text(json.dumps({"from_showcase": str(showcase),
                                "readings": {"abstract": contracts.SHOW_LAYER_ABSTRACT},
                                "markers": warns + [contracts.SHOW_LAYER_NOT_CHECKED]}, indent=1) + "\n",
                    encoding="utf-8")


def _show_fields(ctx: StageContext, bundle: Path, plan_dir: Path) -> dict[str, str]:
    packed = sorted(ctx.work.glob("*.pack.html"))
    if not packed:
        raise StageFailure("no *.pack.html model in the work dir")
    docs = ctx.lane_dir / "docs"
    report = [a.format(docs=docs) for a in contracts.SHOW_REPORT_ARGS] if (docs / "label-roles.json").is_file() else []
    return {"model": str(packed[0]), "bundle": str(bundle), "plan_dir": str(plan_dir),
            "template": str(ctx.path(contracts.SHOW_TEMPLATE)), "memory": str(ctx.work / ".claude/domain-forge-memory.md"),
            "showcase": str(bundle / contracts.SHOWCASE_DIR), "locale": "en", "report_args": report,
            "dataset": ctx.lane.generator,
            "out": str(packed[0].with_name(packed[0].name[:-len(".html")] + ".show.html"))}


def _cli(ctx: StageContext, template: Sequence[str], fields: Mapping[str, Any]) -> list[str]:
    """An argv from a contracts CLI template; `{report_args}` expands to its list."""
    argv = [ctx.menv.python, str(ctx.path(template[0]))]
    for token in template[1:]:
        if token == "{report_args}":
            argv += list(fields["report_args"])
        else:
            argv.append(token.format(**{k: v for k, v in fields.items() if k != "report_args"}))
    return argv


def run_serve(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """Start the bundle's service; it is ready, its showcase verified, `/` is the showcase entry
    with no-cache, and the story endpoint serves the showcase's story bytes (D6 §6.9)."""
    bundle = find_bundle(ctx.work)
    log = ctx.lane_dir / "log" / "service.log"
    service = svc.start(bundle, threads=int(ctx.menv.defaults["threads"]),
                        launch=svc.Launch(ctx.menv.python, log, ctx.runner.env))
    ctx.state.service = service
    status, health = service.get_json(contracts.HEALTH_ROUTE)
    slot = (health or {}).get(contracts.HEALTH_SHOWCASE) or {}
    if not slot.get("verified"):
        raise StageFailure(f"/health {contracts.HEALTH_SHOWCASE} is not verified: {slot.get('reason')}")
    showcase = bundle / contracts.SHOWCASE_DIR
    code, headers, body = service.get("/")
    if code != 200 or body != contracts.showcase_entry(showcase).read_bytes():
        raise StageFailure(f"GET / is not the showcase entry (status {code})")
    if "no-cache" not in headers.get("cache-control", ""):
        raise StageFailure(f"GET / cache-control is {headers.get('cache-control')!r}, not no-cache")
    story = json.loads((showcase / contracts.SHOWCASE_MANIFEST).read_text(encoding="utf-8"))["story"]["file"]
    code, _, body = service.get(contracts.STORY_ROUTE)
    if code != 200 or body != (showcase / story).read_bytes():
        raise StageFailure(f"{contracts.STORY_ROUTE} does not serve showcase/{story} byte for byte")
    ctx.note(f"OK: service ready on {service.base_url}; showcase verified; / and the story served")
    return f"served on port {service.port}"


def _check(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """serve_check / inspect / page: the stage's check script against the live service."""
    if ctx.state.service is None:
        raise StageFailure("no live service (the serve stage did not run in this lane run)")
    out = ctx.lane_dir / "results" / f"{ctx.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = {"url": ctx.state.service.base_url, "bundle": str(find_bundle(ctx.work)),
              "lane": lane.id, "lanes_file": str(ctx.menv.lanes_file), "out": str(out),
              "shots": str(ctx.lane_dir / "shots")}
    ctx.run(_cli(ctx, contracts.CHECK_CLI[ctx.stage], dict(fields, report_args=[])), retry=False)
    return f"{Path(contracts.CHECK_CLI[ctx.stage][0]).name} passed"


def run_golden(lane: Lane, fx: Any, ctx: StageContext) -> str:
    """G00: golden/cic.py re-packs R7 with this release's pack.py and builds its showcase (R2)."""
    out = ctx.lane_dir / "results" / "golden.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = {"cids_root": os.environ["CIDS_ROOT"], "lane_dir": str(ctx.lane_dir), "work": str(ctx.work),
              "skills": str(ctx.menv.skills), "out": str(out), "report_args": []}
    ctx.run(_cli(ctx, contracts.GOLDEN_CLI, fields), retry=False)
    record_pin(ctx)
    return "golden lane re-packed and shown"


FUNCTIONS: Mapping[str, Callable[..., str]] = {
    "fixture": run_fixture, "forge": run_forge, "train": run_train, "eval": run_eval,
    "pack": run_pack, "bundle_api": run_bundle_api, "report": run_report, "show": run_show,
    "serve": run_serve, "serve_check": _check, "inspect": _check, "page": _check,
    contracts.GOLDEN_STAGE: run_golden}


def function_for(driver: str, stage: str) -> Callable[..., str]:
    if driver == "acceptance" and stage in contracts.ACCEPTANCE:
        return run_acceptance
    return FUNCTIONS[stage]
