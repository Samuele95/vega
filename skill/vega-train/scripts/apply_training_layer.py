#!/usr/bin/env python3
"""
apply_training_layer.py — validate a `layer-training-data` document and inject the /vega-train
`training` layer into the model.

    python3 apply_training_layer.py MODEL.html --data <dir>/training-layer.json --out OUT.html
                                    [--notebook train.ipynb] [--replay-epochs 3]
                                    [--domain-forge-dir PATH]

The authoring file carries only the **judgement** — `readings.abstract`, `readings.per_run`,
`readings.hpo`, `readings.selection`, each diagnostic's `reading`, `mechanics.one_step`, the
forks — plus `"from_campaign": "<dir>"`. Everything mechanical is spliced from the campaign:
`campaign.json`, `ladder.json`, `experiments.json` verbatim, the histories downsampled to ≤ 200
points each with the path of the full `history.jsonl` beside them, the loss and update-rule
strings from `spec_snapshot.json`, `mechanics.one_step` from the selected run's
`first_step.json`, and the notebook record written here from `--notebook`.

What this script refuses to publish (run-contract §5):

  * a selection whose run is **ineligible by the rule** with no `override` recorded — the rule
    exists to keep a diverged run or a pruned trial out of the verdict, and an override is the
    only legitimate way past it, in the open;
  * a layer whose notebook replay failed, or whose `--notebook` is missing — determinism is a
    claim the notebook tests, and a layer that skipped the test says nothing about it;
  * a layer with no reading for the selection, or with a run in the ledger and no history;
  * a fork carrying both an `answer` and an `or`;
  * a ledger whose `expect_held` is not what its `expect_check` judges — a rung (or a study)
    with no check, or a verdict that differs from the check's on the ledger it sits in: the
    verdict is the trainer's, judged by `expectations.py`, never written or edited by hand
    (run-contract §4). A trial inherits its study's expect and is judged on the study's record.

The block is written ONLY by `domain-forge/scripts/apply_layer.py`. The input is never modified.

Exit codes: 0 wrote OUT; 2 validation or input error; 3 platform writer missing or failed.
"""
from __future__ import annotations

import argparse
import hashlib
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

from vega_train_core import downsample, read_jsonl  # noqa: E402
from expectations import ExpectCheck, ExpectError, judge_row  # noqa: E402

SKILL_DIR = HERE.parent
ASSETS = SKILL_DIR / "assets"
RENDER_JS = ASSETS / "training-render.js"
STYLE_CSS = ASSETS / "training-layer.css"

LAYER = "training"
SCHEMA = "vega/training@1"
MAX_RENDER = 60 * 1024
MAX_DATA = 400 * 1024
MAX_POINTS = 200

_BLOCK_RE = re.compile(r"[ \t]*<!--\s*@LAYER:start\s+" + LAYER +
                       r"\b[\s\S]*?@LAYER:end\s+" + LAYER + r"\s*-->[ \t]*\n?")

REQUIRED = ("schema", "campaign", "ladder", "experiments", "histories", "mechanics",
            "notebook", "readings")


def _digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------- splicing

def splice(doc, out, base_dir):
    ref = doc.get("from_campaign")
    if not ref:
        return doc, []
    camp = Path(ref)
    if not camp.is_absolute():
        beside = Path(base_dir) / ref
        camp = beside if beside.is_dir() else camp
    if not camp.is_dir():
        return doc, [f"from_campaign {ref!r} is not a directory (looked beside {base_dir} and "
                     "in the working directory)"]

    for key, name in (("campaign", "campaign.json"), ("ladder", "ladder.json"),
                      ("experiments", "experiments.json")):
        if doc.get(key):
            continue
        path = camp / name
        if not path.is_file():
            return doc, [f"from_campaign: {path} not found — the campaign is planned by "
                         "plan_ladder.py and run by trainer.py before a layer is written"]
        try:
            doc[key] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return doc, [f"from_campaign: {path} is not valid JSON: {exc}"]

    spec, _ = _read(camp / "netspec.json")
    primary = ((spec or {}).get("metrics") or {}).get("primary") or "loss"

    if "histories" not in doc:
        doc["histories"] = {}
        runs = [r for r in (doc["experiments"].get("runs") or [])]
        # The 400 KB cap was sized on the default ladder (~8 runs). A campaign that carries more
        # rungs (a contract's mandatory ablations, a study) keeps every run's curve but draws each
        # with proportionally fewer points; the full record stays on disk and the notebook asserts
        # the numbers against it, so the layer's curve is a drawing, not the evidence.
        points = MAX_POINTS if len(runs) <= 8 else max(60, int(MAX_POINTS * 8 / len(runs)))
        for row in runs:
            run_id = row.get("id")
            path = camp / "runs" / run_id / "history.jsonl"
            records = read_jsonl(path)
            if not records:
                continue
            doc["histories"][run_id] = _series(records, primary, str(path), points)
        out(f"OK: from_campaign: {len(doc['histories'])} history/ies downsampled to "
            f"<= {points} points each, values to 6 significant digits; the full records stay "
            "on disk")

    # the mechanics are the campaign's (the loss and update-rule strings, the schedule, the
    # clip, the first step's record), spliced whatever the authoring file holds; its one piece
    # of judgement, `one_step` (SKILL step 5), is kept when written, else read off the run
    authored = doc.get("mechanics") or {}
    doc["mechanics"] = _mechanics(camp, spec, doc["experiments"])
    if authored.get("one_step"):
        doc["mechanics"]["one_step"] = authored["one_step"]
    out("OK: from_campaign: mechanics spliced from spec_snapshot.json and the selected run's "
        "first_step.json" + ("; the authoring file's one_step kept" if authored.get("one_step")
                             else ""))

    if "diagnostics" not in doc:
        doc["diagnostics"] = []
        for row in (doc["experiments"].get("runs") or []):
            for entry in row.get("diagnostics_detail") or []:
                if entry.get("fired"):
                    doc["diagnostics"].append(
                        {"run": row.get("id"), "code": entry.get("code"),
                         "name": entry.get("name"), "evidence": entry.get("evidence"),
                         "reading": "", "led_to": None})
    return doc, []


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _series(records, primary, path, points=MAX_POINTS):
    """One run's curves, downsampled and rounded to 6 significant digits (a drawing, not the
    record), with the full record's path beside them."""
    def ds(values):
        return [(float(f"{v:.6g}") if isinstance(v, float) else v)
                for v in downsample(values, points)]

    def col(part, key):
        return [((r.get(part) or {}).get(key)) for r in records]

    def nested(key, sub, block):
        out = []
        for r in records:
            node = r.get(key) or {}
            if sub:
                node = node.get(sub) or {}
            out.append(node.get(block) if isinstance(node, dict) else None)
        return out

    blocks = []
    for r in records:
        for b in ((r.get("grad_norm") or {}).get("per_block") or {}):
            if b not in blocks:
                blocks.append(b)
    series = {
        "path": path, "downsampled_from": len(records),
        "epochs": ds([r.get("epoch") for r in records]),
        "train.loss": ds(col("train", "loss")),
        "valid.loss": ds(col("valid", "loss")),
        f"train.{primary}": ds(col("train", primary)),
        f"valid.{primary}": ds(col("valid", primary)),
        "lr": ds([r.get("lr") for r in records]),
        "grad_norm.global": ds(
            [(r.get("grad_norm") or {}).get("global") for r in records]),
    }
    for block in blocks:
        series[f"grad_norm.{block}"] = ds(nested("grad_norm", "per_block", block))
        series[f"update_ratio.{block}"] = ds(nested("update_ratio", None, block))
    return series


def _mechanics(camp, spec, experiments):
    snapshot, _ = _read(camp / "spec_snapshot.json")
    resolved = (snapshot or {}).get("_resolved") or {}
    smoke, _ = _read(camp / "smoke.json")
    loss = (spec or {}).get("loss") or {}
    optimizer = (spec or {}).get("optimizer") or {}
    schedule = (spec or {}).get("schedule") or {}
    clip = next((r.get("value") for r in (spec or {}).get("regularization") or []
                 if r.get("kind") == "grad_clip"), None)
    selected = ((experiments.get("selection") or {}).get("run"))
    first, _ = _read(camp / "runs" / str(selected) / "first_step.json")
    return {
        "loss": (smoke or {}).get("loss_formula") or loss.get("formula"),
        "loss_params": loss.get("params"),
        "update_rule": optimizer.get("update_rule"),
        "optimizer": {"name": optimizer.get("name"), "params": optimizer.get("params")},
        "schedule": {"name": schedule.get("name"), **(schedule.get("params") or {}),
                     "lr_by_epoch": (resolved.get("schedule_lrs")
                                     or (smoke or {}).get("schedule_lrs") or [])},
        "clip": clip,
        "one_step": _one_step_text(first) if first else None,
        "one_step_record": first,
        "determinism": resolved.get("determinism"),
    }


def _one_step_text(first):
    """`mechanics.one_step` is read from first_step.json — never computed by hand (§5)."""
    if not first:
        return None
    block = first.get("block")
    g, d, r = first.get("grad_norm_head"), first.get("delta_norm_head"), first.get("ratio_head")
    return (f"at epoch 1, step 1 the head's weights ({block}) carried ‖g‖ = {g:.6g} and moved "
            f"‖Δθ‖ = {d:.6g}, a ratio of {r:.3g}" if None not in (g, d, r) else
            f"the head's first step was recorded for {block} but not every norm was available")


# --------------------------------------------------------------------------- validation

def validate(doc, out):
    errs = []
    if not isinstance(doc, dict):
        return ["layer document is not a JSON object"]
    for key in REQUIRED:
        if key not in doc:
            errs.append(f"missing required top-level key {key!r}")
    if errs:
        return errs
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA!r}, got {doc.get('schema')!r}")

    experiments = doc["experiments"] or {}
    rows = experiments.get("runs") or []
    if not rows:
        errs.append("experiments.runs is empty; a campaign with no run has no layer")
    selection = experiments.get("selection") or {}
    if not selection.get("run"):
        errs.append("experiments.selection.run is required — the layer is the campaign's "
                    "verdict, and a campaign without a selection has not reached one")
    else:
        by_id = {r.get("id"): r for r in rows}
        chosen = by_id.get(selection["run"])
        if chosen is None:
            errs.append(f"selection.run {selection['run']!r} is not a run in the ledger")
        elif chosen.get("eligible") is False and not selection.get("override"):
            errs.append(
                f"selection.run {selection['run']!r} is ineligible by the rule "
                f"({', '.join(chosen.get('ineligible_because') or ['—'])}) and no override is "
                "recorded. A diverged run, a run that never learned, a budget-cut run before "
                "its first improvement and a pruned trial are kept out of the verdict by the "
                "rule; the way past it is `select.py --override <id> --reason … --source "
                "user:<turn>`, in the open")
        if selection.get("override") and not selection["override"].get("reason"):
            errs.append("selection.override carries no reason; an override that overrules a "
                        "mechanical rule states why")

    histories = doc.get("histories") or {}
    for row in rows:
        run_id = row.get("id")
        if row.get("status") == "failed":
            continue
        if run_id not in histories:
            errs.append(f"histories has no entry for run {run_id!r}; every run in the ledger "
                        "carries its curves")
        elif not (histories[run_id] or {}).get("path"):
            errs.append(f"histories[{run_id!r}] carries no `path` to the full history.jsonl")

    notebook = doc.get("notebook") or {}
    if not notebook.get("path"):
        errs.append("notebook.path is required — pass --notebook; the stage order is "
                    "judgement → notebook → apply")
    else:
        if notebook.get("replay_ok") is False:
            errs.append("the notebook's replay did not match the records. Determinism is a "
                        "claim the notebook tests, and a campaign that failed the test is not "
                        "published — find the cause (a seed the backend did not set, a "
                        "nondeterministic op, a device change) first")
        if not notebook.get("executed") and not any("notebook not executed" in str(m)
                                                    for m in doc.get("markers") or []):
            errs.append("notebook.executed is false but markers carry no `WARN: notebook not "
                        "executed`; an unexecuted notebook is a WARN, never silently a report")

    mechanics = doc.get("mechanics") or {}
    if not mechanics.get("one_step"):
        out("WARN: mechanics.one_step is empty; §5 reads it from the selected run's "
            "first_step.json and it is what the reader sees of the first update")

    for fork in doc.get("forks") or []:
        if not isinstance(fork, dict):
            errs.append("every fork is an object")
            break
        if bool(fork.get("answer")) == bool(fork.get("or")):
            errs.append(f"fork {fork.get('asked')!r} carries "
                        + ("both `answer` and `or`" if fork.get("answer")
                           else "neither `answer` nor `or`")
                        + "; exactly one records what settled it")

    readings = doc.get("readings") or {}
    for key in ("abstract", "selection"):
        if not readings.get(key):
            errs.append(f"readings.{key} is required — the layer is a verdict, and a verdict "
                        "without its reasoning is a number")
    missing = [r.get("id") for r in rows
               if not (readings.get("per_run") or {}).get(r.get("id"))]
    if missing:
        out(f"WARN: no per-run reading for {', '.join(map(str, missing))}; "
            "training-catalog §5 says each run's reading answers whether it learned, "
            "converged and generalised")
    return errs


def _is_trial(row) -> bool:
    return row.get("kind") == "hpo" or str(row.get("source") or "").startswith("train:hpo/")


def exposure_problems(campaign) -> list:
    """The seal's one exception is disclosed or absent (run-contract §1): a campaign whose spec
    trains a `transductive` graph carries `campaign.exposure` with the spec's declared
    `exposure`; any other campaign carries none. The spec is read from the campaign's run
    directory; a campaign that names none is not judged here."""
    run_dir = Path(str((campaign or {}).get("run_dir") or ""))
    spec, _ = _read(run_dir / "netspec.json") if str(run_dir) not in ("", ".") else (None, None)
    if not isinstance(spec, dict):
        return []
    sample = (spec.get("data_contract") or {}).get("sample") or {}
    disclosed = (campaign or {}).get("exposure")
    if sample.get("shape") == "graph" and sample.get("protocol") == "transductive":
        declared = sample.get("exposure")
        if not isinstance(disclosed, dict) or disclosed.get("declared") != declared:
            return [f"the spec trains a transductive graph (the test nodes' features are in the "
                    f"training graph, exposure {declared!r}) and campaign.exposure "
                    f"{'is missing' if not disclosed else 'does not repeat it'}; the seal's one "
                    "exception is disclosed in the layer, or it is a leak"]
        return []
    if disclosed:
        return [f"campaign.exposure discloses {disclosed!r}, and the spec's protocol is not "
                "transductive: an exposure the training did not take is not on the record"]
    return []


def expect_problems(experiments) -> list:
    """Every ledger row and study record whose `expect_held` is not its `expect_check`'s
    verdict on this ledger (run-contract §4: scripts write the ledger, the check decides)."""
    rows = [r for r in (experiments or {}).get("runs") or [] if isinstance(r, dict)]
    mode = ((experiments or {}).get("ranking") or {}).get("mode")
    if mode not in ("max", "min"):
        return ["experiments.ranking carries no mode (select.py writes it); the expect "
                "verdicts cannot be checked in the metric's direction"]
    primary = next((r.get("primary_name") for r in rows if r.get("primary_name")), None)
    spec = {"metrics": {"primary": primary, "selection": {"mode": mode}}}
    errs = []
    targets = [(r.get("id"), r, r) for r in rows if not _is_trial(r)]
    targets += [(s.get("study"), s, {"id": s.get("study"), "primary": s.get("best_value")})
                for s in (experiments or {}).get("hpo") or [] if isinstance(s, dict)]
    for name, record, subject in targets:
        raw = record.get("expect_check")
        if raw is None:
            errs.append(f"ledger record {name!r} carries no expect_check — the ledger records "
                        "whether every rung's expect held, judged by trainer.py (a study by "
                        "hpo.py); a row without one was not written by them")
            continue
        try:
            check = ExpectCheck.parse(raw)
        except ExpectError as exc:
            errs.append(f"ledger record {name!r}: its expect_check is malformed: {exc}")
            continue
        held, evidence = judge_row(check, subject, rows, spec)
        if record.get("expect_held") != held:
            errs.append(f"ledger record {name!r} says expect_held "
                        f"{json.dumps(record.get('expect_held'))} but its expect_check judges "
                        f"{json.dumps(held)} on this ledger ({evidence}) — the verdict is the "
                        "check's, written by the scripts, never edited")
    return errs


# --------------------------------------------------------------------------- noscript

def _esc(value):
    return _html.escape("" if value is None else str(value), quote=True)


def noscript_html(doc):
    experiments = doc.get("experiments") or {}
    selection = experiments.get("selection") or {}
    readings = doc.get("readings") or {}
    parts = ['<section class="layer-training-noscript">', "<h2>Training</h2>",
             f"<p>{_esc(readings.get('abstract'))}</p>",
             "<h3>The ledger</h3>",
             "<table><thead><tr><th>run</th><th>hypothesis</th><th>delta</th>"
             "<th>valid</th><th>epochs</th><th>stopped by</th><th>diagnostics</th>"
             "</tr></thead><tbody>"]
    for row in experiments.get("runs") or []:
        mark = " (selected)" if row.get("id") == selection.get("run") else ""
        value = row.get("primary")
        parts.append("<tr>" + "".join(f"<td>{_esc(x)}</td>" for x in (
            f"{row.get('id')}{mark}", row.get("hypothesis"), row.get("delta_text"),
            "·" if value is None else f"{float(value):.5g}", row.get("epochs"),
            row.get("stopped_by"), ", ".join(row.get("diagnostics") or []))) + "</tr>")
    parts.append("</tbody></table>")
    parts += ["<h3>The selection</h3>",
              f"<p>{_esc(selection.get('run'))} — valid {_esc(selection.get('primary'))} "
              f"{_esc(selection.get('value'))}, margin over the baseline "
              f"{_esc(selection.get('margin_over_baseline'))}.</p>",
              f"<p>{_esc(selection.get('rule'))}</p>"]
    if selection.get("override"):
        parts.append(f"<p><strong>Override</strong> ({_esc(selection['override'].get('source'))}): "
                     f"{_esc(selection['override'].get('reason'))}</p>")
    parts.append(f"<p>{_esc(readings.get('selection'))}</p>")
    mechanics = doc.get("mechanics") or {}
    parts += ["<h3>Mechanics</h3>",
              f"<p>{_esc(mechanics.get('loss'))}</p>",
              f"<p>{_esc(mechanics.get('update_rule'))}</p>",
              f"<p>{_esc(mechanics.get('one_step'))}</p>"]
    if doc.get("diagnostics"):
        parts.append("<h3>Diagnostics</h3><ul>")
        for entry in doc["diagnostics"]:
            parts.append(f"<li>{_esc(entry.get('run'))} {_esc(entry.get('code'))} "
                         f"{_esc(entry.get('name'))}: {_esc(entry.get('evidence'))}</li>")
        parts.append("</ul>")
    parts.append("<p>Enable JavaScript for the curves, the study and the run selector."
                 "</p></section>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- the writer

def notebook_record(path, replay_epochs, out):
    p = Path(path)
    if not p.is_file():
        return None, f"--notebook {p} not found; the stage order is judgement → notebook → apply"
    try:
        nb = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"{p} is not a readable notebook: {exc}"
    code = [c for c in nb.get("cells") or [] if c.get("cell_type") == "code"]
    outputs = [c.get("outputs") or [] for c in code]
    errored = any(o.get("output_type") == "error" for cell in outputs for o in cell)
    executed = bool(code) and all(cell for cell in outputs) and not errored
    # The notebook's replay cell prints `VEGA-REPLAY: ok …` or `VEGA-REPLAY: mismatch …`.
    # Reading that marker rather than the prose around it is the difference between a verdict
    # and a guess: "does not match" and "do not match" are the same result and a phrase match
    # gets one of them wrong.
    text = "\n".join("".join(o.get("text") or []) for cell in outputs for o in cell)
    marker = re.search(r"VEGA-REPLAY:\s*(ok|mismatch)\b", text)
    replay_ok = None
    if marker:
        replay_ok = (marker.group(1) == "ok")
    elif executed:
        out("WARN: the notebook carries no VEGA-REPLAY marker; it was emitted by an older "
            "nb_emit.py, and whether the selected run replays is unknown rather than proven")
    if not executed:
        out("WARN: notebook not executed" + (" — a cell ended in an error output" if errored
                                             else ""))
    return {"path": str(p), "executed": executed, "digest": _digest_file(p),
            "replay_epochs": replay_epochs, "replay_ok": replay_ok,
            "code_cells": len(code)}, None


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
        for name, text in (("layer.json", data_text), ("training-render.js", render_js),
                           ("training-layer.css", style_css), ("noscript.html", noscript)):
            files[name] = tmp / name
            files[name].write_text(text, encoding="utf-8")
        out_file = tmp / "out.html"
        proc = subprocess.run(
            [sys.executable, str(writer), str(model_path), "--layer", LAYER,
             "--data", str(files["layer.json"]), "--render", str(files["training-render.js"]),
             "--style", str(files["training-layer.css"]), "--data-type", "application/json",
             "--produced-by", produced_by, "--noscript", str(files["noscript.html"]),
             "--out", str(out_file)],
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
    ap.add_argument("--data", required=True, help="the campaign's training-layer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--notebook", default=None, help="train.ipynb (executed)")
    ap.add_argument("--replay-epochs", type=int, default=3)
    ap.add_argument("--produced-by", default="/vega-train")
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
    for asset in (Path(a.render), Path(a.style)):
        if not asset.is_file():
            print(f"ERROR: asset missing: {asset}")
            return 2
    render_js = Path(a.render).read_text(encoding="utf-8").rstrip()
    style_css = Path(a.style).read_text(encoding="utf-8").rstrip()
    if len(render_js.encode("utf-8")) > MAX_RENDER:
        print(f"ERROR: render script is {len(render_js.encode('utf-8'))} bytes; contract §5 "
              f"caps it at {MAX_RENDER // 1024} KB")
        return 2

    doc, errs = splice(doc, print, Path(a.data).resolve().parent)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2
    doc.pop("from_campaign", None)

    if a.notebook:
        record, nb_err = notebook_record(a.notebook, a.replay_epochs, print)
        if nb_err:
            print(f"ERROR: {nb_err}")
            return 2
        doc["notebook"] = record
        if not record["executed"]:
            markers = doc.setdefault("markers", [])
            if not any("notebook not executed" in str(m) for m in markers):
                markers.append("WARN: notebook not executed")

    errs = validate(doc, print)
    errs += expect_problems(doc.get("experiments")) if isinstance(doc, dict) else []
    errs += exposure_problems(doc.get("campaign")) if isinstance(doc, dict) else []
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 2

    data_text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    if len(data_text.encode("utf-8")) > MAX_DATA:
        print(f"ERROR: the layer's data is {len(data_text.encode('utf-8')) // 1024} KB; "
              f"contract §5 caps it at {MAX_DATA // 1024} KB — the histories are downsampled "
              "to 200 points and the full records stay on disk, so a document this large "
              "means something else was inlined")
        return 2

    experiments = doc["experiments"]
    selection = experiments.get("selection") or {}
    print(f"OK: {a.data} validates against {SCHEMA} "
          f"({len(experiments.get('runs') or [])} run(s), "
          f"{len(doc.get('histories') or {})} history/ies, "
          f"{len(doc.get('diagnostics') or [])} diagnostic(s), selected "
          f"{selection.get('run')}"
          + (" by override" if selection.get("override") else " by the rule")
          + f", notebook executed {doc['notebook'].get('executed')}, replay "
            f"{doc['notebook'].get('replay_ok')})")

    noscript = noscript_html(doc)
    df_dir = (Path(a.domain_forge_dir) if a.domain_forge_dir
              else SKILL_DIR.parent / "domain-forge")
    writer = df_dir / "scripts" / "apply_layer.py"
    if not writer.is_file():
        print(f"ERROR: platform writer not found at {writer}; /vega-train writes layers only "
              "through domain-forge's apply_layer.py (pass --domain-forge-dir if the skill "
              "lives elsewhere)")
        return 3
    with tempfile.TemporaryDirectory() as td:
        result, how = platform_apply(writer, html, data_text, render_js, style_css,
                                     a.produced_by, noscript, td, model_path)
    if result is None:
        print(f"ERROR: platform writer {writer} failed ({how})")
        return 3
    if _BLOCK_RE.search(html):
        print("WARN: input already carried a training layer; it was replaced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result, encoding="utf-8")
    print(f"OK: wrote {out_path} via {how}")
    print(f"OK: render script {len(render_js.encode('utf-8'))} bytes, "
          f"data {len(data_text.encode('utf-8')) // 1024} KB of the "
          f"{MAX_DATA // 1024} KB the contract allows, "
          f"output {len(result.encode('utf-8')) // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
