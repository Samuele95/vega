#!/usr/bin/env python3
"""
cic.py — the golden CIC lane, G00 (BUILD-PLAN R2, D6 §11.5): the CIC-IDS2017 R7 network the v2.2
chain trained, evaluated under the seal and shipped, rebuilt as a bundle of this release and
given its showcase by /vega-show — with the facts a person bound by hand recovered by the script,
the page laid out as the hand-built one, and the attack classes drawn as alerts.

    python3 cic.py --cids-root DIR --lane-dir DIR --work DIR --skills DIR --json OUT
                   [--lanes-file F] [--lane G00] [--python EXE] [--chrome EXE]

`lanekit` runs it as the lane's `golden` stage (contracts.GOLDEN_CLI); the service stages (serve,
serve_check, inspect, page) then run on `<work>/bundle/<slug>` as on any lane. The steps:

 1. **inputs** — the lane's `recorded` paths under `$CIDS_ROOT` (the campaign, the run, the shaped
    parts, the v2.2 bundle, the hand-written story plan, the report); every file under them is
    snapshotted, and step 9 holds the lane to having changed none.
 2. **blueprint** — the v2.2 bundle's pipeline run with R-G2's two rules (cic_migrate).
 3. **migrate** — the campaign under `<work>/campaign`: the stored predictions re-encoded in
    `vega/predictions-layout@1`, R7's package re-emitted by this release (cic_migrate). The test
    part is never opened: the seal is copied as it stands (one exposure).
 4. **pack** — this release's `pack.py --serve` on the migrated campaign (run from `$CIDS_ROOT`,
    where the recipe's raw path is relative): provenance, the shaper's transform and twin check,
    the round trip over every test row against the stored predictions, the examples (the 50
    test rows the bundle ships), serving, inspection, the conformance gate; then the bundle
    layer's authoring (cic_authoring.pack_layer), the model card, the executed notebook, the
    layer, `validate_model.py` and `smoke_bundle.py --strict`.
 5. **report** — `docs/` in the lane dir: the real `main.aux`, `main.pdf`, `main.tex` and the
    role map golden/cic-report-roles.json binds (cic_authoring.label_roles).
 6. **show** — the user's stances in the work dir's memory (`show: motif lanes`, `show: roles
    alert = every class but BENIGN`), `extract_facts.py --report`, `draft_plan.py --carry` of
    the hand-written v1 plan, the plan completed (cic_authoring.story_plan), `resolve_story.py`,
    `build_showcase.py` (en and qps-ploc), `--check`, the layer, `validate_model.py`,
    `smoke_show.py --strict`.
 7. **facts** — fact recovery: every `values` entry of the v1 plan whose ref the extractor also
    reads (its v2 ref, `carry_v1.v2_ref`, is a fact's) has a fact equal to the value the v2.2
    pack resolved for it (the v2.2 bundle's `story.json` `values`, at the rounding it wrote).
 8. **served** — the v2.2 bundle's hand-built page and this showcase served side by side: AD7
    layout parity within 8 px in three passes (1440×900 night and day, 390×844 night), every view
    at rest and after each act (a preset, a source, each decision, each code step), AD7 content
    parity (the hand-built page's limits, results facts, choice titles and code references said
    again, or an allowed change named in CONTENT_ALLOW), and P-ROLE inverted — every attack
    class's preset drawn in `--alert`, BENIGN's never (cic_parity).
 9. **read-only** — nothing under `$CIDS_ROOT` that the lane read was created, changed or removed.

Each step logs to `<lane-dir>/log/golden.<step>.log`; the result is `--json`
(`vega/golden-result@1`: per step its status, seconds and detail; the migration record, the
fact recovery, the layout readings and the alert record). Exit 0: every step passed. 1: a step
failed (`ERROR:` naming it). 2: blocked — an input under `$CIDS_ROOT` is missing or the recorded
campaign cannot be migrated (D6 §11.5: a nightly finding, quoted).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
for _p in (HERE, TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cic_authoring as authoring  # noqa: E402
import cic_migrate as migrate  # noqa: E402
import cic_parity as parity  # noqa: E402

# The soft tokens of foreign-literals.json whose dataset sense is this lane's own (the CIC row
# noun, the traffic capture, its port columns, its flow extractor, its label column): the page
# smoke's P-LEAK lets the golden lane's page say them (page_smoke_checks reads this literal).
VOCAB_WITHHELD = ("flow", "flows", "traffic", "capture", "port", "extractor", "Label")
RESULT = "vega/golden-result@1"
STORY_PLAN_FIXTURE = "vega-show/scripts/tests/fixtures/cids/story-plan.v1.json"
NEUTRAL = "BENIGN"
DATASET = authoring.DATASET
ERROR_LINE = re.compile(r"^\s*(?:[\w.-]+\|\s*)?ERROR:", re.M)


# AD7 content parity: the hand-built page's words the showcase may say otherwise, each with why
# (cic_parity.content_problems); a change not listed here fails the lane
CONTENT_ALLOW: Mapping[str, Mapping[str, str]] = {
    "limits": {"rare classes are anecdotes":
               "R7's evaluation (v2.2, migrated, its test part never re-opened) records no class_support, "
               "so the small_support limit that carries it has no facts to key on; an evaluation of this "
               "release records it (vega-eval metrics.class_support) and the limit is kept"},
    "facts": {"test flows": "said as 'flows in the sealed test' (results.fact.n_test, the lexicon's words)"},
    "choices": {},
    "code_notes": {"the http route":
                   "the hand page cited the section by its title (“from a raw flow to an answer”, linked at "
                   "its serving figure's page); a showcase code note cites one numbered report label, the "
                   "section's serving figure"}}


class StepFailed(Exception):
    """A step's command or check failed (exit 1)."""


class Blocked(Exception):
    """The recorded inputs cannot be used as they stand (exit 2)."""


# --------------------------------------------------------------------------- the run
class Golden:
    def __init__(self, a: argparse.Namespace) -> None:
        self.cids, self.lane_dir = Path(a.cids_root).resolve(), Path(a.lane_dir).resolve()
        self.work, self.skills = Path(a.work).resolve(), Path(a.skills).resolve()
        self.python, self.out = a.python, Path(a.json)
        self.lanes_file = Path(a.lanes_file) if a.lanes_file else self.skills / "vega/scripts/tests/lanes.json"
        self.lane_id, self.chrome = a.lane, a.chrome
        self.logs = self.lane_dir / "log"
        self.result: dict[str, Any] = {"schema": RESULT, "lane": a.lane, "status": "running", "steps": []}
        self.state: dict[str, Any] = {}

    # ---------------------------------------------------------------- helpers
    def say(self, line: str) -> None:
        print(line, flush=True)

    def run(self, step: str, argv: Sequence[Any], cwd: Path | None = None, timeout: float = 1800,
            env: Mapping[str, str] | None = None) -> str:
        """A command, logged to golden.<step>.log; its output, or StepFailed on a non-zero exit
        or an ERROR: line."""
        argv = [str(a) for a in argv]
        log = self.logs / f"golden.{step}.log"
        merged = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
        t0 = time.time()
        proc = subprocess.run(argv, cwd=cwd or self.work, env=merged, capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
        output = (proc.stdout or "") + (proc.stderr or "")
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(argv)}\n{output}\n[exit {proc.returncode}, {time.time() - t0:.1f} s]\n\n")
        errors = [l.strip() for l in output.splitlines() if ERROR_LINE.match(l)]
        if proc.returncode != 0 or errors:
            name = next((Path(a).name for a in argv[1:3] if a.endswith(".py")), argv[0])
            tail = output.strip().splitlines()[-1:] or [""]
            raise StepFailed(f"{name} exited {proc.returncode}: {'; '.join(errors[:3]) or tail[0]} "
                             f"(log {log})")
        return output

    def py(self, script: str, *args: Any) -> list[Any]:
        return [self.python, self.skills / script, *args]

    # ---------------------------------------------------------------- 1. inputs
    def inputs(self) -> str:
        lanes = json.loads(self.lanes_file.read_text(encoding="utf-8"))
        lane = next((l for l in lanes["lanes"] if l["id"] == self.lane_id), None)
        if lane is None:
            raise Blocked(f"lane {self.lane_id} is not in {self.lanes_file}")
        rec = lane["recorded"]
        needed = {k: self.cids / v for k, v in rec.items() if k != "run"}
        missing = [f"{k} ({p})" for k, p in needed.items() if not p.exists()]
        if missing:
            raise Blocked("missing under $CIDS_ROOT: " + ", ".join(missing))
        old = needed["bundle"]
        manifest = json.loads((old / "MANIFEST.json").read_text(encoding="utf-8"))
        base = self.cids / Path(manifest["blueprint"]["run"]).parent
        evaluated = self._evaluated_model(needed["campaign"])
        watched = [needed["campaign"], needed["shaped"], old, needed["story_plan"], needed["aux"],
                   needed["pdf"], base, evaluated, self.cids / "dataset"]
        self.state.update(lane=lane, rec=rec, paths=needed, base=base, evaluated=evaluated,
                          watched=watched, snapshot=migrate.readonly_snapshot(watched))
        fixture = self.skills / STORY_PLAN_FIXTURE
        if not fixture.is_file() or fixture.read_bytes() != needed["story_plan"].read_bytes():
            self.say(f"WARN: {STORY_PLAN_FIXTURE} is not the hand-written plan at "
                     f"{rec['story_plan']}; the lane carries $CIDS_ROOT's")
        return (f"campaign {rec['campaign']} run {rec['run']}, shaped {rec['shaped']}, v2.2 bundle "
                f"{rec['bundle']}, blueprint {base.name}, model {evaluated.name}; "
                f"{len(self.state['snapshot'])} file(s) watched read-only")

    def _evaluated_model(self, campaign: Path) -> Path:
        """The evaluated model page the campaign's evaluation layer was written into: the
        `.train.eval.html` beside the campaign's model."""
        doc = json.loads((campaign / "campaign.json").read_text(encoding="utf-8"))
        model = self.cids / doc["input"]["model"]
        found = model.with_name(model.name[:-len(".html")] + ".train.eval.html")
        if not found.is_file():
            raise Blocked(f"the evaluated model page {found} is missing")
        return found

    # ---------------------------------------------------------------- 2-3. blueprint, migrate
    def blueprint(self) -> str:
        out = self.work / "blueprint-runs" / f"{self.state['base'].name}-r-g2"
        try:
            pipeline = migrate.amend_blueprint(self.state["base"], out)
        except migrate.MigrationError as exc:
            raise Blocked(str(exc)) from exc
        self.state["pipeline"] = pipeline
        changes = json.loads(pipeline.read_text(encoding="utf-8"))["amends"]["changes"]
        return f"{pipeline.relative_to(self.work)}: {len(changes)} change(s) (R-G2)"

    def migrate(self) -> str:
        try:
            m = migrate.migrate_campaign(self.cids, self.state["rec"], self.work, self.skills,
                                         self.state["pipeline"], self.python, self.logs / "golden.migrate.log")
        except migrate.MigrationError as exc:
            raise Blocked(f"the recorded campaign cannot be migrated: {exc}") from exc
        self.state["migration"] = m
        self.result["migration"] = m.record
        pred = m.record["predictions"]
        return (f"{pred['rows']} stored prediction(s) re-encoded ({pred['layout']}); {len(m.record['code']['to'])} "
                f"package file(s) re-emitted for {m.run}; the seal records {m.record['seal']['exposures']} exposure(s)")

    # ---------------------------------------------------------------- 4. pack
    def pack(self) -> str:
        m, lane = self.state["migration"], self.state["lane"]
        model = self.work / self.state["evaluated"].name
        shutil.copy2(self.state["evaluated"], model)
        plan = m.campaign / "monitor-plan.json"
        out = self.run("pack", self.py(
            "vega-pack/scripts/pack.py", "--campaign", m.campaign, "--shaped", self.state["paths"]["shaped"],
            "--bundle-dir", self.work / "bundle", "--pipeline", m.pipeline, "--model", model, "--serve",
            "--unattended", "--row-noun", lane["lexicon"]["row"]["one"], "--service-title", lane["service_title"],
            *(["--monitor-plan", plan] if plan.is_file() else [])), cwd=self.cids)
        trip = next((l for l in out.splitlines() if l.startswith("OK: round trip")), "")
        bundle = next(self.work.glob("bundle/*/MANIFEST.json")).parent
        self.state["bundle"] = bundle
        layer = authoring.pack_layer(bundle, self.state["paths"]["bundle"] / "bundle-layer.json",
                                     f"bundle/{bundle.name}")
        (bundle / "bundle-layer.json").write_text(json.dumps(layer, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
        pk = "vega-pack/scripts/"
        self.run("pack", self.py(pk + "model_card.py", "--bundle", bundle, "--model", model, "--intended-use",
                                 bundle / "bundle-layer.json", "--out", bundle / "model-card.html"))
        self.run("pack", self.py("vega-forge/scripts/nb_emit.py", "--stage", "pack", "--from", bundle,
                                 "--out", bundle / "predict.ipynb"))
        self.run("pack", [self.python, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                          "--ExecutePreprocessor.timeout=1800", "predict.ipynb"], cwd=bundle,
                 env={"CUDA_VISIBLE_DEVICES": ""})
        packed = model.with_name(model.name[:-len(".html")] + ".pack.html")
        self.run("pack", self.py(pk + "apply_bundle_layer.py", model, "--data", bundle / "bundle-layer.json",
                                 "--notebook", bundle / "predict.ipynb", "--out", packed))
        self.run("pack", self.py("domain-forge/scripts/validate_model.py", packed))
        self.run("pack", self.py(pk + "smoke_bundle.py", packed, "--strict"))
        self.state["packed"] = packed
        return f"{bundle.name}: {trip[4:].split(';')[0] if trip else 'packed'}; card, notebook, layer validated"

    # ---------------------------------------------------------------- 5. report
    def report(self) -> str:
        docs = self.lane_dir / "docs"
        doc = authoring.label_roles(self.cids, docs, authoring.REPORT_ROLES_FILE,
                                    self.skills / "vega-show/scripts", self.lane_id)
        self.state["docs"] = docs
        return f"docs/label-roles.json: {len(doc['roles'])} role(s) bound to the real report's labels"

    # ---------------------------------------------------------------- 6. show
    def show(self) -> str:
        from lanekit.registry import load_lanes
        from lanekit import stages as lane_stages
        lane, bundle, docs = load_lanes(self.lanes_file)[self.lane_id], self.state["bundle"], self.state["docs"]
        plan_dir = self.lane_dir / "show" / "plan"
        plan_dir.mkdir(parents=True, exist_ok=True)
        memory = self.work / ".claude" / "domain-forge-memory.md"
        authoring.write_stances(memory, lane)
        sh, tpl = "vega-show/scripts/", self.skills / "vega-show/assets/page/TEMPLATE.json"
        packed, showcase = self.state["packed"], bundle / "showcase"
        facts, draft, plan, story = (plan_dir / n for n in ("facts.json", "story-plan.draft.json",
                                                              "story-plan.json", "story.json"))
        # the steps before the build keep their markers where the build keeps them (<b>/showcase.building)
        markers = bundle / "showcase.building" / ".work" / "markers.json"
        self.run("show", self.py(sh + "extract_facts.py", "--model", packed, "--bundle", bundle,
                                 "--report", docs, "--out", facts, "--markers", markers))
        drafted = self.run("show", self.py(sh + "draft_plan.py", "--facts", facts, "--page-manifest", tpl, "--memory",
                                           memory, "--dataset", DATASET, "--carry", self.state["paths"]["story_plan"],
                                           "--out", draft, "--markers", markers))
        # every slot the carry could not keep, listed in the lane's result (and in the markers the
        # layer carries): a dropped limit or fact of the hand-built page is never silent
        self.result["carry_drops"] = [line.strip() for line in drafted.splitlines()
                                      if line.startswith("WARN: carry ")]
        authored, edits = authoring.story_plan(json.loads(draft.read_text(encoding="utf-8")),
                                               json.loads(facts.read_text(encoding="utf-8")), lane,
                                               out=self.say)
        problems = authoring.digit_problems(authored, "show")
        if problems:
            raise StepFailed("the story plan states a number: " + "; ".join(problems[:3]))
        plan.write_text(json.dumps(authored, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        self.result["carry_edits"] = edits
        self.run("show", self.py(sh + "resolve_story.py", "--plan", plan, "--facts", facts, "--page-manifest",
                                 tpl, "--report", docs, "--out", story, "--markers", markers))
        for locale in ("en", "qps-ploc"):
            self.run("show", self.py(sh + "build_showcase.py", "--bundle", bundle, "--model", packed, "--plan",
                                     plan, "--story", story, "--facts", facts, "--report", docs, "--out",
                                     showcase, "--unattended", "--locale", locale))
        self.run("show", self.py(sh + "build_showcase.py", "--check", showcase, "--report", docs))
        lane_stages._layer_authoring(showcase)
        shown = packed.with_name(packed.name[:-len(".html")] + ".show.html")
        self.run("show", self.py(sh + "apply_show_layer.py", packed, "--data",
                                 showcase / ".work" / "showcase-layer.json", "--out", shown))
        self.run("show", self.py("domain-forge/scripts/validate_model.py", shown))
        self.run("show", self.py(sh + "smoke_show.py", shown, "--strict"))
        self.state.update(facts=facts)
        return (f"showcase built in {bundle.name}/showcase from the carried plan "
                f"({len(edits)} carry edit(s)); {shown.name} validated")

    # ---------------------------------------------------------------- 7. fact recovery
    def facts(self) -> str:
        record = fact_recovery(json.loads(self.state["paths"]["story_plan"].read_text(encoding="utf-8")),
                               json.loads((self.state["paths"]["bundle"] / "story.json").read_text(encoding="utf-8")),
                               json.loads(self.state["facts"].read_text(encoding="utf-8")),
                               self.skills / "vega-show/scripts")
        self.result["fact_recovery"] = record
        if record["differ"]:
            raise StepFailed(f"{len(record['differ'])} hand-bound value(s) differ from the extracted fact: "
                             + "; ".join(d["name"] + " " + d["why"] for d in record["differ"][:3]))
        if not record["equal"]:
            raise StepFailed("no hand-bound value was compared: the extractor reads none of the v1 refs")
        return (f"{len(record['equal'])} hand-bound value(s) equal their extracted fact; "
                f"{len(record['unread'])} name a ref the extractor does not read")

    # ---------------------------------------------------------------- 8. served
    def served(self) -> str:
        sys.path.insert(0, str(self.skills / "vega-show/scripts"))
        import cdp
        chrome = cdp.find_chrome(self.chrome)
        if not chrome:
            raise StepFailed("no Chrome (set CHROME): layout parity and the alert colours cannot be read")
        show, shots = self.skills / "vega-show/scripts", self.lane_dir / "shots" / "golden"
        old = parity.serve(self.state["paths"]["bundle"], self.python, self.logs / "golden.served-hand.log",
                           ["--torch-threads", "2"])
        try:
            new = parity.serve(self.state["bundle"], self.python, self.logs / "golden.served-showcase.log",
                               ["--torch-threads", "2"])
            try:
                passes, problems = {}, []
                for name, scheme, viewport, mobile in parity.PASSES:
                    hand = parity.layout(old.url, chrome, show, parity.HAND_ALIASES, shots / name / "hand",
                                         scheme=scheme, viewport=viewport, mobile=mobile)
                    built = parity.layout(new.url, chrome, show, None, shots / name / "showcase",
                                          scheme=scheme, viewport=viewport, mobile=mobile)
                    found = parity.layout_problems(hand, built, frame_only=name != parity.PASSES[0][0])
                    passes[name] = {"hand": hand, "showcase": built, "problems": found}
                    problems += [f"{name}: {p}" for p in found]
                said = {"hand": parity.content(old.url, chrome, show, parity.HAND_ALIASES),
                        "showcase": parity.content(new.url, chrome, show)}
                lost, allowed = parity.content_problems(said["hand"], said["showcase"], CONTENT_ALLOW)
                alert, alert_problems = parity.alert_presets(new.url, chrome, show, NEUTRAL)
            finally:
                new.stop()
        finally:
            old.stop()
        self.result["layout"] = {"tolerance_px": 8, "passes": passes, "problems": problems}
        self.result["content"] = {**said, "problems": lost, "allowed": allowed}
        self.result["alert"] = {**alert, "problems": alert_problems}
        for line in allowed:
            self.say(f"WARN: AD7 content: an allowed change — {line}")
        if problems or lost or alert_problems:
            raise StepFailed("; ".join((problems + lost + alert_problems)[:6]))
        states = sum(1 for k in passes[parity.PASSES[0][0]]["showcase"] if ":" in k)
        return (f"AD7: {len(parity.PASSES)} pass(es) (night and day at 1440, night at 390) × "
                f"{len(parity.VIEWS)} view(s) and {states} state(s) within 8 px of the hand-built page; "
                f"its words kept ({len(allowed)} allowed change(s), each named); "
                f"{alert['alert_presets']} attack-class preset(s) drawn in --alert, {NEUTRAL} never")

    # ---------------------------------------------------------------- 9. read-only
    def readonly(self) -> str:
        changed = migrate.readonly_changes(self.state["snapshot"], self.state["watched"])
        self.result["readonly"] = {"watched": len(self.state["snapshot"]), "changed": changed}
        if changed:
            raise StepFailed(f"{len(changed)} file(s) under $CIDS_ROOT changed: {changed[:3]}")
        return f"{len(self.state['snapshot'])} file(s) under $CIDS_ROOT unchanged"

    # ---------------------------------------------------------------- the driver
    STEPS: tuple[tuple[str, str], ...] = (
        ("inputs", "inputs"), ("blueprint", "blueprint"), ("migrate", "migrate"), ("pack", "pack"),
        ("report", "report"), ("show", "show"), ("facts", "facts"), ("served", "served"),
        ("readonly", "readonly"))

    def main(self) -> int:
        self.logs.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        code = 0
        for step, method in self.STEPS:
            t0 = time.time()
            entry = {"id": step}
            try:
                entry.update(status="pass", detail=getattr(self, method)())
                self.say(f"OK: golden {step}: {entry['detail']}")
            except Blocked as exc:
                entry.update(status="blocked", detail=str(exc))
                self.say(f"ERROR: golden {step} blocked: {exc}")
                code = 2
            except (StepFailed, migrate.MigrationError, ValueError, RuntimeError, OSError,
                    subprocess.TimeoutExpired) as exc:
                entry.update(status="fail", detail=f"{type(exc).__name__}: {exc}")
                self.say(f"ERROR: golden {step} failed: {exc}")
                code = 1
            entry["seconds"] = round(time.time() - t0, 1)
            self.result["steps"].append(entry)
            self._write()
            if code:
                break
        self.result["status"] = {0: "pass", 1: "fail", 2: "blocked"}[code]
        self._write()
        return code

    def _write(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(json.dumps(self.result, indent=1, ensure_ascii=False, default=str) + "\n",
                            encoding="utf-8")


# --------------------------------------------------------------------------- fact recovery
def _same(old: Any, new: Any) -> bool:
    """`new` (the extracted fact's value) is `old` (the v2.2 pack's) at the rounding `old` was
    written at: a float to the decimals of its own repr, mappings and lists item by item."""
    if isinstance(old, bool) or isinstance(new, bool):
        return old is new
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if isinstance(old, int) and float(new).is_integer():
            return int(new) == old
        text = repr(float(old))
        places = len(text.split(".")[1]) if "." in text and "e" not in text else 12
        return round(float(new), places) == round(float(old), places)
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        return set(old) <= set(new) and all(_same(v, new[k]) for k, v in old.items())
    if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)):
        return len(old) == len(new) and all(_same(a, b) for a, b in zip(old, new))
    return str(old) == str(new)


def fact_recovery(plan_v1: Mapping, story_v1: Mapping, facts: Mapping, show_scripts: Path) -> dict:
    """D6 §11.5 'fact recovery' over the v1 plan's values (see the module doc)."""
    if str(show_scripts) not in sys.path:
        sys.path.insert(0, str(show_scripts))
    from carry_v1 import v2_ref
    roles = ((facts.get("groups") or {}).get("report") or {}).get("roles") or {}
    by_label = {str(b.get("label")): role for role, b in roles.items() if isinstance(b, Mapping)}
    by_ref: dict[str, list] = {}
    for key, fact in (facts.get("facts") or {}).items():
        by_ref.setdefault(str(fact.get("ref")), []).append((key, fact.get("value")))
    hand = story_v1.get("values") or {}
    record: dict[str, list] = {"equal": [], "differ": [], "unread": []}
    for name, ref in (plan_v1.get("values") or {}).items():
        found = by_ref.get(v2_ref(ref, by_label) or "")
        if not found:
            record["unread"].append({"name": name, "ref": ref})
            continue
        old = hand.get(name)
        match = next((key for key, value in found if _same(old, value)), None)
        if match:
            record["equal"].append({"name": name, "ref": ref, "fact": match})
        else:
            record["differ"].append({"name": name, "ref": ref, "facts": [k for k, _ in found],
                                     "why": f"hand-bound {old!r}, extracted "
                                            f"{[v for _, v in found]!r}"[:240]})
    return record


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cids-root", required=True)
    ap.add_argument("--lane-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--skills", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--lanes-file", default=None)
    ap.add_argument("--lane", default="G00")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--chrome", default=None)
    return Golden(ap.parse_args(argv)).main()


if __name__ == "__main__":
    sys.exit(main())
