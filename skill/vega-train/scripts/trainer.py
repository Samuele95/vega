#!/usr/bin/env python3
"""
trainer.py — run one run of a campaign, and record everything it did.

    python3 trainer.py --campaign <dir> --run R1
                       [--delta '<json>'] [--budget-min 5] [--device auto]
                       [--resume] [--replay <id> --epochs 3] [--vega-forge-dir PATH]

One run is one *delta on the spec*, and one rung of `ladder.json`: a run with no rung is
refused, `--delta` included — it carries a trial of a planned study (`H<study>-<n>`, which
`hpo.py` samples) or the planned rung's own delta, never a change nobody stated a hypothesis
for. A delta that lands on a value no emitted code reads (a dead delta, `live_paths.py`) is
refused: it would train its baseline's numbers. Every rung but a trial carries a checkable
`expect` (`expectations.py`: its own `expect_check`, else its rule's in catalog.json), and the
ledger row records `expect_held` — the check judged against the ledger, and judged again for
every pending row whenever another row lands.

The trainer applies the delta, recompiles through the forge's `compile_spec.py --check-only`
(an ablation that produces an invalid spec is refused **before** it runs, not after it wastes
the budget), re-emits the package when the delta touches the
architecture decision family (against the campaign root, where a custom block's `code_ref`
resolves) and passes `conformance.py` on the result, then trains through the emitted package's
adapters with callbacks that write the records the whole sub-chain reads:

    runs/<id>/netspec.resolved.json   the spec this run actually trained
    runs/<id>/history.jsonl           one epoch record per line (EpochRecord.as_dict())
    runs/<id>/grad_stats.jsonl        sampled per-step pre-clip gradient norms
    runs/<id>/first_step.json         the head's ‖g‖, ‖Δθ‖ and ratio at epoch 1, step 1
    runs/<id>/checkpoints/            best.pt and last.pt, digested in run.json
    runs/<id>/predictions.valid.parquet   the valid part scored from best.pt, in
                                      vega/predictions-layout@1 (R-C2), the test's layout
    runs/<id>/figures/                the curves as SVG
    runs/<id>/run.json                the delta, the digests, the timings, stopped_by
    experiments.json                  this run's ledger row, appended

The package is opened the way every stage opens it: `vega_core.PackageAdapters.load` resolves
each port's adapter by its blueprint-map role (D1 §12.1) — the `Network` builder, the
`ShapedParts` DataSource, the `Trainer`, the `ModelRegistry` store and, when the blueprint drew
one, the `Evaluator`. A run is `Trainer(store).fit(model, spec, data, callbacks,
run=RunContext(budget_min, checkpoint_dir, grad_sample_every))`; the valid predictions are the
Evaluator's `ScoredPart` (one scoring path with /vega-eval: the raw head, the truth in its own
dimension, the anchor's or node's raw position as the row id, the per-row score of the
reconstruction and anomaly readings), or `Trainer.predict` when no Evaluator port exists
(`stored_predictions.py`). A delta that moves a block param the spec leaves unstated is read
from the value the block is built with (`stated_defaults`), and `run.json.device` is
the Trainer module's `optim.device_for(spec)`.

A campaign is append-only: a finished run is never reopened. `--resume` continues a run whose
status is `running` under its own id; against a run that is `done` it creates a new id with
`supersedes` set, as the contract requires — `<id>b`, `<id>c`, … the first id no rung, ledger
row or run directory holds and no ladder rule owns (a resumed R1 is R1c: R1b is the recurrent
floor's) — warm-started from the superseded run's last.pt. The trainer refuses to start at all in a campaign
whose `test_seal.json` exists — training after the test was seen is a new campaign
(`--after-eval`), never a continuation of the one that saw it.

`--replay <id> --epochs k` re-runs the first k epochs and compares the records with the
recorded ones within the backend's determinism tolerance, writing nothing else. The spec is
replayed as the run resolved it, `max_epochs` included — the learning-rate schedule is a
function of it — and the replay stops itself after k epoch records. Determinism is a claim,
and this is what tests it.

Exit codes: 0 the run finished (including `stopped_by: budget`); 2 an input, a delta or a
recompile was refused; 3 the run failed while training; 4 a replay mismatch.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vega_train_core import (Delta, PathError, Rung, append_jsonl, digest_file,  # noqa: E402
                             digest_text, forge_scripts, import_forge, load_catalog,
                             normalise_path, read_jsonl, read_path, rung_source_ok)
import diagnostics as DX  # noqa: E402
import live_paths as LP  # noqa: E402
from expectations import check_for, rejudge  # noqa: E402
from stored_predictions import (PREDICTIONS_LAYOUT, StoredScores,  # noqa: E402,F401
                                plain as _plain, write_valid_predictions)

#: what a run id looks like in the catalog's `run_ids` (its other values are prose)
_RUN_ID = re.compile(r"[A-Z][0-9]+[a-z]*")
#: an HPO trial's id: its study's id, a dash, the trial's number (catalog `run_ids.hpo_trial`)
_TRIAL_ID = re.compile(r"(H[A-Za-z0-9_]*)-([0-9]+)")


# --------------------------------------------------------------------------- the campaign

@dataclass(frozen=True)
class Campaign:
    """The campaign directory, and the records that make it one."""

    root: Path
    campaign: Mapping[str, Any]
    spec: Mapping[str, Any]
    ladder: Mapping[str, Any]

    @staticmethod
    def read(root) -> tuple:
        root = Path(root)
        if not root.is_dir():
            return None, f"campaign directory not found: {root}"
        out = {}
        for key, name, required in (("campaign", "campaign.json", True),
                                    ("spec", "netspec.json", True),
                                    ("ladder", "ladder.json", False)):
            path = root / name
            if not path.is_file():
                if required:
                    return None, (f"{path} not found — the campaign is planned by "
                                  "plan_ladder.py in the run directory /vega-forge created")
                out[key] = {}
                continue
            try:
                out[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return None, f"{path} is not valid JSON: {exc}"
        return Campaign(root, out["campaign"], out["spec"], out["ladder"]), None

    # -- the seal

    def sealed(self) -> bool:
        return (self.root / "test_seal.json").is_file()

    # -- layout

    @property
    def package(self) -> str:
        return ((self.spec.get("outputs") or {}).get("package")
                or (self.campaign.get("input") or {}).get("package") or "pipeline")

    def run_dir(self, run_id) -> Path:
        return self.root / "runs" / run_id

    def rung(self, run_id):
        for raw in (self.ladder.get("runs") or []):
            if raw.get("id") == run_id:
                return Rung.from_json(raw)
        return None

    def planned_rung(self, run_id):
        """The ladder rung a run belongs to: its own, or — for an HPO trial `H<study>-<n>` —
        its study's, when the study is a planned `hpo` rung. None when the ladder holds
        neither: such a run would exist without a hypothesis."""
        own = self.rung(run_id)
        if own is not None:
            return own
        trial = _TRIAL_ID.fullmatch(str(run_id))
        study = self.rung(trial.group(1)) if trial else None
        return study if study is not None and study.kind == "hpo" else None

    def package_dir_for(self, run_id) -> Path:
        """A run that re-emitted the code runs its own package; otherwise the campaign's."""
        own = self.run_dir(run_id) / self.package
        return own if own.is_dir() else self.root / self.package

    def taken_ids(self) -> set:
        """Every run id the campaign holds or reserves: the ladder's rungs, the ledger's rows,
        the run directories, and the ids a ladder rule owns whether or not its rung was
        planned (catalog `run_ids`: R1 the floor, R1b the recurrent floor)."""
        taken = {str(r.get("id")) for r in (self.ladder.get("runs") or []) if r.get("id")}
        taken |= {str(r.get("id")) for r in (self.experiments().get("runs") or [])
                  if r.get("id")}
        runs = self.root / "runs"
        if runs.is_dir():
            taken |= {p.name for p in runs.iterdir()}
        catalog, _ = load_catalog()
        taken |= {str(v) for v in ((catalog or {}).get("run_ids") or {}).values()
                  if _RUN_ID.fullmatch(str(v))}
        return taken

    def superseding_id(self, run_id) -> str | None:
        """The id of a new run that supersedes `run_id`: `<id>b`, `<id>c`, … — the first no
        rung, ledger row or run directory holds and no ladder rule owns, so resuming R1 can
        never land in the recurrent floor's R1b. None when the letters are spent."""
        taken = self.taken_ids()
        return next((f"{run_id}{c}" for c in "bcdefghijklmnopqrstuvwxyz"
                     if f"{run_id}{c}" not in taken), None)

    def experiments(self) -> dict:
        path = self.root / "experiments.json"
        if not path.is_file():
            return {"schema": "vega/experiments@1", "runs": [], "ranking": {},
                    "selection": {}, "hpo": [],
                    "budget": dict(self.campaign.get("budget") or {})}
        return json.loads(path.read_text(encoding="utf-8"))

    def write_experiments(self, ledger) -> None:
        (self.root / "experiments.json").write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def spent_min(self) -> float:
        total = 0.0
        for row in self.experiments().get("runs") or []:
            total += float(row.get("time_s") or 0.0) / 60.0
        return total


# --------------------------------------------------------------------------- the forge's CLIs

def _run_cli(script, args, cwd=None, forge=None):
    cmd = [sys.executable, str(forge_scripts(forge) / script), *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd) if cwd else None,
                          timeout=1800)
    return proc.returncode, (proc.stdout + proc.stderr)


def recompile(spec_path, camp: Campaign, forge=None):
    """The forge's compiler, on the run's resolved spec. Never a second implementation."""
    model = (camp.campaign.get("input") or {}).get("model")
    if not model:
        # the compiler cannot run without the model it reads the layers from: say so here,
        # rather than let its usage error read as a spec that does not compile (F37)
        return 2, ("ERROR: campaign.json records no input.model — the campaign was planned "
                   "without the forge's .vega.html; plan it with plan_ladder.py --campaign "
                   "<dir> --model <the .vega.html>\n")
    pipeline = (camp.spec.get("input") or {}).get("pipeline")
    args = ["--spec", spec_path, "--check-only", "--model", str(_beside(camp.root, model))]
    if pipeline:
        args += ["--pipeline", pipeline]
    return _run_cli("compile_spec.py", args, forge=forge)


def re_emit(spec_path, out_dir, camp: Campaign, forge=None):
    """Re-emit with `--smoke`: the forward pass is the gate that a changed architecture builds
    at all, and its `smoke.json` is where this run's true parameter count comes from — which
    the selection rule uses as its tie-break, so a missing one is not cosmetic.

    The package lands in the run's own directory, but the campaign root stays what a custom
    block's relative `code_ref` resolves against (netspec-contract §2): `--campaign-root`
    names it, or the copy under `runs/<id>/` would look for the module beside the run."""
    pipeline = (camp.spec.get("input") or {}).get("pipeline")
    args = ["--spec", spec_path, "--out", out_dir, "--campaign-root", camp.root, "--smoke"]
    if pipeline:
        args += ["--pipeline", pipeline]
    return _run_cli("emit_backend.py", args, forge=forge)


def conformance(package_dir, camp: Campaign, forge=None):
    pipeline = (camp.spec.get("input") or {}).get("pipeline")
    return _run_cli("conformance.py", ["--package", package_dir, "--pipeline", pipeline],
                    forge=forge)


def _beside(root, name):
    """The chain's files sit beside the run directory, not inside it."""
    path = Path(name)
    if path.is_absolute() or path.is_file():
        return path
    for base in (root, root.parent, Path.cwd()):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return path


# --------------------------------------------------------------------------- figures

def svg_curve(series, title, ylabel, width=560, height=170, pad=34):
    """One SVG line chart per curve. No library: the layer draws its own and these are the
    on-disk copy a reader opens without a browser tab."""
    lines = [(name, [v for v in values]) for name, values in series if any(
        v is not None for v in values)]
    if not lines:
        return None
    flat = [v for _, values in lines for v in values if v is not None]
    lo, hi = min(flat), max(flat)
    if hi == lo:
        hi = lo + 1e-9
    n = max(len(values) for _, values in lines)
    colours = ("#1b3a73", "#9a3a2c", "#2e6b3f", "#7a3c77", "#8a6420")

    def x(i):
        return pad + (width - 2 * pad) * (i / max(1, n - 1))

    def y(v):
        return height - pad - (height - 2 * pad) * ((v - lo) / (hi - lo))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'role="img" aria-label="{title}">',
           f'<title>{title}</title>',
           f'<rect width="{width}" height="{height}" fill="#fff"/>',
           f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
           'stroke="#ccc"/>',
           f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#ccc"/>',
           f'<text x="{pad}" y="{pad - 12}" font-family="sans-serif" font-size="11" '
           f'fill="#333">{title}</text>',
           f'<text x="4" y="{pad - 2}" font-family="sans-serif" font-size="9" fill="#777">'
           f'{ylabel} {hi:.4g}</text>',
           f'<text x="4" y="{height - pad}" font-family="sans-serif" font-size="9" '
           f'fill="#777">{lo:.4g}</text>']
    for k, (name, values) in enumerate(lines):
        points = " ".join(f"{x(i):.1f},{y(v):.1f}"
                          for i, v in enumerate(values) if v is not None)
        colour = colours[k % len(colours)]
        out.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.6" '
                   f'points="{points}"/>')
        out.append(f'<text x="{width - pad - 4}" y="{pad + 4 + 12 * k}" text-anchor="end" '
                   f'font-family="sans-serif" font-size="10" fill="{colour}">{name}</text>')
    out.append(f'<text x="{width - pad}" y="{height - pad + 14}" text-anchor="end" '
               f'font-family="sans-serif" font-size="9" fill="#777">epoch {n}</text>')
    out.append("</svg>")
    return "\n".join(out)


def write_figures(run_dir, history, spec):
    figures = Path(run_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    primary = ((spec.get("metrics") or {}).get("primary")) or "loss"

    def col(part, key):
        return [((r.get(part) or {}).get(key)) for r in history]

    def per_block(key, sub=None):
        blocks, rows = [], []
        for r in history:
            node = r.get(key) or {}
            if sub:
                node = node.get(sub) or {}
            rows.append(node if isinstance(node, Mapping) else {})
            for b in rows[-1]:
                if b not in blocks:
                    blocks.append(b)
        return [(b, [row.get(b) for row in rows]) for b in blocks]

    charts = {
        "loss.svg": ([("train", col("train", "loss")), ("valid", col("valid", "loss"))],
                     "loss", "max"),
        f"{primary}.svg": ([("train", col("train", primary)),
                            ("valid", col("valid", primary))], primary, "max"),
        "lr.svg": ([("lr", [r.get("lr") for r in history])], "learning rate", "max"),
        "grad_norm.svg": ([("global", [(r.get("grad_norm") or {}).get("global")
                                       for r in history])] + per_block("grad_norm", "per_block"),
                          "gradient norm", "max"),
        "update_ratio.svg": (per_block("update_ratio"), "update ratio", "max"),
    }
    written = []
    for name, (series, title, _) in charts.items():
        svg = svg_curve(series, title, title)
        if svg:
            (figures / name).write_text(svg, encoding="utf-8")
            written.append(f"figures/{name}")
    return written


# --------------------------------------------------------------------------- the run

class Pruned(Exception):
    """Raised out of `on_epoch` to stop a trial the study has decided against.

    Pruning is a real stop, not a label applied afterwards: the exception unwinds the emitted
    `fit` at the epoch boundary, so a pruned trial has spent exactly the epochs its partial
    history shows and the budget it did not spend is still there for the next trial.
    """

    def __init__(self, epoch, value):
        super().__init__(f"pruned at epoch {epoch} (value {value})")
        self.epoch, self.value = epoch, value


class _Recorder:
    """The callbacks the emitted `fit` calls. Everything the campaign knows about a run comes
    through here — which is why the trainer writes no number it did not receive."""

    def __init__(self, run_dir, log, on_value=None):
        self.run_dir = Path(run_dir)
        self.history_path = self.run_dir / "history.jsonl"
        self.grad_path = self.run_dir / "grad_stats.jsonl"
        self.first_path = self.run_dir / "first_step.json"
        self.records = []
        self.log = log
        self.on_value = on_value

    def on_epoch(self, record):
        row = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        self.records.append(row)
        append_jsonl(self.history_path, row)
        primary = row.get("valid") or {}
        self.log(f"epoch {row.get('epoch'):>4}  train.loss {_g(row.get('train', {}).get('loss'))}"
                 f"  valid.loss {_g(primary.get('loss'))}"
                 f"  lr {_g(row.get('lr'))}"
                 f"  |g| {_g((row.get('grad_norm') or {}).get('global'))}")
        if self.on_value is not None:
            self.on_value(row)

    def on_first_step(self, record):
        Path(self.first_path).write_text(
            json.dumps(dict(record), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def on_step(self, record):
        append_jsonl(self.grad_path, dict(record))


def _g(value, places=5):
    return "·" if value is None else f"{float(value):.{places}g}"


def open_package(package_dir: Path, package: str, forge=None):
    """The emitted package this run trains with — the campaign root's, or the run's own when
    the delta re-emitted it — its adapters resolved by blueprint-map role through
    `vega_core.PackageAdapters.load`, the one resolver every stage uses (D1 §12.1). It imports
    the package fresh from its parent, so a re-emitted copy shadows nothing.

    Raises the resolver's `AdapterLoadError` (naming the file and what it should hold) when
    the package cannot be driven as its map says."""
    core = import_forge("vega_core", forge)
    return core.PackageAdapters.load(Path(package_dir), package)


def _device_of(adapters, spec_obj, device) -> str:
    """The device the run trained on, as D1 §12.2 records it: the Trainer module's `device_for`,
    the device the Trainer itself picks from the spec; the requested one when it cannot say."""
    try:
        return str(adapters.device_for(spec_obj))
    except Exception:  # noqa: BLE001 — a Trainer without device_for still trained
        return str(device)


class RunTarget(NamedTuple):
    """Where a run writes: its id and directory, the run it supersedes, the last.pt it is
    warm-started from."""

    run_id: str
    run_dir: Path
    supersedes: str | None = None
    warm_from: Path | None = None

    @staticmethod
    def of(camp: Campaign, run_id, resume) -> tuple:
        """(RunTarget, None), or (RunTarget, what happened) when a finished run is superseded,
        or (None, why not). A campaign is append-only (run-contract §1): a finished run is
        never reopened; `--resume` continues a `running` run under its own id from its own
        last.pt, and supersedes a finished one under a new id (`Campaign.superseding_id`),
        warm-started from the superseded run's last.pt."""
        run_dir = camp.run_dir(run_id)
        existing = run_dir / "run.json"
        if not existing.is_file():
            return RunTarget(run_id, run_dir), None
        status = json.loads(existing.read_text(encoding="utf-8")).get("status")
        finished = status in ("done", "stopped_by_budget")
        if not resume:
            return (None, f"run {run_id} is already {status}; a campaign is append-only. Use "
                          "--resume to start a new run that supersedes it.") if finished \
                else (RunTarget(run_id, run_dir), None)
        last = run_dir / "checkpoints" / "last.pt"
        if not finished:
            return RunTarget(run_id, run_dir, warm_from=last), None
        new_id = camp.superseding_id(run_id)
        if new_id is None:
            return None, (f"{run_id} has been superseded under every free id; resume one of "
                          "its successors instead")
        if camp.run_dir(new_id).exists():
            return None, (f"{camp.run_dir(new_id)} already exists; a campaign is append-only "
                          "and a superseding run never writes into another run's directory")
        return (RunTarget(new_id, camp.run_dir(new_id), run_id, last),
                f"{run_id} is {status}; --resume opens {new_id} with supersedes: {run_id}, "
                f"warm-started from {run_id}'s last.pt")


def stated_defaults(spec, delta, forge=None) -> tuple:
    """(spec, delta) with every block param the delta moves but the spec leaves unstated
    written in at the value the block is built with (`plan_ladder.unstated_param`: the older
    name it still reads, else the forge catalog's default) — a forge-composed `lstm` leaves
    `num_layers` at 1, and a trial that searches it moves it *from* 1. The delta's `from`
    is filled where it was left empty, so the checks read the same value the builder uses
    and the resolved spec states what trained. Unchanged when nothing is unstated."""
    import plan_ladder as PL
    base, edits, forge_cat = spec, [], None
    for edit in delta.as_json():
        try:
            read_path(spec, normalise_path(edit["path"]))
        except PathError:
            forge_cat = forge_cat if forge_cat is not None else PL.forge_catalog(forge)
            where, value = PL.unstated_param(spec, edit["path"], forge_cat)
            if where is not None:
                base = Delta.of([{"path": edit["path"], "to": value}]).apply(base)
                if edit.get("from") is None:
                    edit = dict(edit, **{"from": value})
        edits.append(edit)
    return base, (Delta.of(edits) if base is not spec else delta)


def run_one(camp: Campaign, run_id, *, delta_override=None, budget_min=None, device="auto",
            resume=False, forge=None, out=print, grad_every=10, on_epoch=None, rung=None):
    """Train one run and write its records. Returns (run.json record, ledger row, exit code)."""
    if camp.sealed():
        out("ERROR: this campaign's test_seal.json exists — the test part has been opened, "
            "and training never reopens the campaign that saw it. Start a new campaign with "
            "--after-eval, which records that you know.")
        return None, None, 2

    planned = camp.planned_rung(run_id)
    if planned is None:
        out(f"ERROR: run {run_id!r} has no rung in ladder.json — no run exists without a "
            "hypothesis, a source and an expect (run-contract §3): --delta carries a trial of "
            "a planned study (H<study>-<n>, sampled by hpo.py) or a planned rung's own delta. "
            "Add the rung to ladder.json first.")
        return None, None, 2
    if planned.kind == "hpo" and planned.id == run_id:
        out(f"ERROR: {run_id} is a study, not a run: hpo.py --study {run_id} runs its trials "
            f"({run_id}-1, {run_id}-2, …), each a point of the declared space")
        return None, None, 2
    if planned.id != run_id and delta_override is None:
        out(f"ERROR: {run_id} is a trial of {planned.id} and carries the point its study "
            "sampled: run it through hpo.py, or pass that point as --delta")
        return None, None, 2
    if rung is None:
        rung = trial_rung(planned, run_id, delta_override) if planned.id != run_id else planned
    if delta_override is not None and planned.kind != "hpo" and \
            _delta_signature(delta_override) != _delta_signature(planned.delta.as_json()):
        out(f"ERROR: {run_id}: --delta is not the rung's planned delta — the rung's hypothesis "
            "and expect are about its own change; a different change is a different rung, "
            "planned in ladder.json with its own")
        return None, None, 2
    delta = Delta.of(delta_override) if delta_override is not None else rung.delta
    if rung.source and not rung_source_ok(rung.source):
        out(f"ERROR: rung {run_id}'s source {rung.source!r} is not one the contract allows "
            "(netspec, netspec:<path>, vega:ladder/<rule>, train:diagnostic/<code>, "
            "train:hpo/<study>, analysis:F<n>, user:<turn>)")
        return None, None, 2

    # a trial samples a space that was declared and gated before the study ran, so it may
    # span decision families; an ablation may not (run-contract §3)
    is_trial = bool(rung.kind == "hpo" or str(rung.source or "").startswith("train:hpo/"))
    check = None
    if not is_trial:
        catalog, _ = load_catalog()
        check, why = check_for(rung.expect_check, rung.source, catalog, run_id)
        if check is None:
            out(f"ERROR: {run_id}: its expect is not checkable — {why}. The ledger records "
                "whether a rung's expect held (run-contract §3-§4): state it as the rung's "
                "expect_check in ladder.json")
            return None, None, 2
    base, delta = stated_defaults(camp.spec, delta, forge)
    problems = delta.problems(base, rung.family, multi_family_ok=is_trial)
    if problems:
        for problem in problems:
            out(f"ERROR: {run_id}: {problem}")
        return None, None, 2
    dead = LP.dead_edits(delta, base, LP.of_campaign(camp.root, forge))
    if dead:
        for _, why in dead:
            out(f"ERROR: {run_id}: dead delta — {why}; the run would train its baseline's "
                "numbers, so it is refused before it spends the budget")
        return None, None, 2

    target, why = RunTarget.of(camp, run_id, resume)
    if target is None:
        out(f"ERROR: {why}")
        return None, None, 2
    if target.supersedes:
        out(f"OK: {why}")
    run_id, run_dir, supersedes, warm_from = target
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    def log(line):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        out(f"    {line}")

    # --- the resolved spec
    resolved_delta = delta.resolved(base)
    resolved = delta.apply(base)
    spec_path = run_dir / "netspec.resolved.json"
    spec_text = json.dumps(resolved, indent=2, ensure_ascii=False) + "\n"
    spec_path.write_text(spec_text, encoding="utf-8")

    code, output = recompile(spec_path, camp, forge)
    if code != 0:
        out(f"ERROR: {run_id}: the delta produces a spec that does not compile — the rung is "
            "refused before it runs, not after it spends the budget")
        lines = output.strip().splitlines()
        errors = [line for line in lines if line.startswith("ERROR:")]
        # a compiler that failed without an ERROR: line (a usage error, a crash) still says
        # why: its last lines are the reason, and a refusal never hides it (F37)
        for line in errors or [line for line in lines if line.strip()][-3:]:
            out(f"    {line}")
        return None, None, 2
    out(f"OK: {run_id}: the resolved spec compiles ({resolved_delta.describe()})")

    # --- re-emit when the architecture moved
    package_dir = camp.root / camp.package
    if delta.touches_architecture(base) and len(delta):
        code, output = re_emit(spec_path, run_dir, camp, forge)
        if code != 0:
            out(f"ERROR: {run_id}: re-emission failed")
            for line in output.strip().splitlines()[-6:]:
                out(f"    {line}")
            return None, None, 2
        package_dir = run_dir / camp.package
        code, output = conformance(package_dir, camp, forge)
        if code != 0:
            out(f"ERROR: {run_id}: the re-emitted package fails the mechanical conformance "
                "gate — the fix goes to the backend's templates, not to this run's code")
            for line in output.strip().splitlines():
                if line.startswith("ERROR:"):
                    out(f"    {line}")
            return None, None, 2
        out(f"OK: {run_id}: architecture delta — package re-emitted into {package_dir.name}/ "
            "and it passes conformance")

    # --- open the package: its adapters, by blueprint-map role
    try:
        adapters = open_package(package_dir, camp.package, forge)
    except ImportError as exc:          # AdapterLoadError is an ImportError
        out(f"ERROR: {run_id}: the package at {package_dir} cannot be driven: {exc}")
        return None, None, 2
    domain = adapters.package.domain

    # --- train
    started = time.time()
    record = {"schema": "vega/run@1", "id": run_id, "kind": rung.kind,
              "delta": resolved_delta.as_json(), "status": "running",
              "spec_digest": digest_text(spec_text), "package_dir": str(package_dir),
              "supersedes": supersedes, "started_at": _now()}
    (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    recorder = _Recorder(run_dir, log, on_value=on_epoch)
    checkpoints = run_dir / "checkpoints"
    pruned_at, model = None, None
    try:
        spec_obj = domain.NetSpec.from_dict(resolved)
        model = adapters.build(spec_obj, run_dir=str(camp.root))
        if warm_from is not None and warm_from.is_file():
            model = adapters.store.load(warm_from, spec=spec_obj, build=adapters.build,
                                        run_dir=str(camp.root))
            log(f"warm-started from {warm_from.relative_to(camp.root).as_posix()}")
        elif warm_from is not None:
            log(f"WARN: {warm_from.relative_to(camp.root).as_posix()} does not exist; the run "
                "starts from the spec's init")
        result = adapters.trainer.fit(
            model, spec_obj, adapters.data, callbacks=recorder,
            run=domain.RunContext(budget_min=budget_min, checkpoint_dir=str(checkpoints),
                                  grad_sample_every=grad_every))
    except Pruned as stop:
        pruned_at = stop
        result = _PartialResult(len(recorder.records))
        log(str(stop))
    except Exception as exc:  # noqa: BLE001
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      time_s=time.time() - started, finished_at=_now())
        (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        log("TRACEBACK\n" + traceback.format_exc())
        out(f"ERROR: {run_id} failed while training: {type(exc).__name__}: {exc}")
        out(f"    the traceback is in {log_path}")
        return record, None, 3

    elapsed = time.time() - started
    history = recorder.records
    checkpoint = checkpoints / "best.pt"
    last = checkpoints / "last.pt"
    try:
        adapters.store.save(model, last, spec=spec_obj, epoch=result.epochs_run)
    except Exception as exc:  # noqa: BLE001
        log(f"WARN: last.pt not written: {type(exc).__name__}: {exc}")

    # --- predictions on valid, from best.pt
    predictions = None
    if checkpoint.is_file():
        try:
            predictions = write_valid_predictions(run_dir, adapters, spec_obj, checkpoint,
                                                  camp.root)
        except Exception as exc:  # noqa: BLE001
            log(f"WARN: predictions.valid.parquet not written: {type(exc).__name__}: {exc}")

    figures = write_figures(run_dir, history, resolved)

    stopped_by = ("pruned" if pruned_at else
                  ("divergence" if any(r.get("nan") for r in history)
                   else _plain(result.stopped_by)))
    record.update(
        status=("pruned" if pruned_at else
                ("stopped_by_budget" if stopped_by == "budget" else "done")),
        stopped_by=stopped_by, epochs=_plain(result.epochs_run),
        best_epoch=_plain(result.best_epoch), best_value=_plain(result.best_value),
        time_s=round(elapsed, 3), finished_at=_now(),
        checkpoint_digest=digest_file(checkpoint) if checkpoint.is_file() else None,
        checkpoint=str(checkpoint) if checkpoint.is_file() else None,
        code_digests={f"{camp.package}/{p.relative_to(package_dir).as_posix()}": digest_file(p)
                      for p in sorted(package_dir.rglob("*.py"))},
        predictions=(predictions or {}).get("path"),
        predictions_layout=PREDICTIONS_LAYOUT if predictions else None,
        predictions_unscored=(predictions or {}).get("unscored"),
        figures=figures, device=_device_of(adapters, spec_obj, device))
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8")

    # --- diagnostics and the ledger row
    catalog, cat_err = load_catalog()
    if cat_err:
        out(f"WARN: {cat_err}; diagnostics were not evaluated")
        findings = []
    else:
        findings = DX.evaluate(history, spec=resolved, run=record,
                               grad_stats=read_jsonl(run_dir / "grad_stats.jsonl"),
                               floor_primary=_floor_primary(camp),
                               catalog=catalog)
    row = ledger_row(run_id, rung, record, history, resolved, findings, check)
    row = _append_ledger(camp, row)

    fired = [f for f in findings if f.fired]
    out(f"OK: {run_id}: {record['epochs']} epoch(s) in {elapsed:.1f}s, stopped by "
        f"{stopped_by}, best epoch {record['best_epoch']}, "
        f"valid {row.get('primary_name')} {_g(row.get('primary'))}")
    if check is not None:
        verdict = {True: "held", False: "did not hold"}.get(row.get("expect_held"), "pending")
        out(f"OK: {run_id}: expect {verdict} — {row.get('expect_evidence')}")
    for f in fired:
        out(f"WARN: {run_id}: {f.code} {f.name} — {f.evidence}")
    unavailable = [f.code for f in findings if f.unavailable]
    if unavailable:
        out(f"OK: {run_id}: {', '.join(unavailable)} unavailable (a field the backend wrote as "
            "null, or a run that has not happened yet) — recorded, not guessed")
    return record, row, 0


class _PartialResult:
    """What a pruned trial returns in place of a FitResult: it never reached one."""

    def __init__(self, epochs):
        self.checkpoint = None
        self.best_epoch = None
        self.best_value = None
        self.epochs_run = epochs
        self.stopped_by = "pruned"
        self.history_path = None


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _floor_primary(camp: Campaign):
    """The floor run's valid primary, which D4 needs and which does not exist until it ran.
    The floor's ledger row is the one whose source is the ladder's floor rule (`kind` is
    `ablation` there; `floor` and `rule` are the older rows' spellings)."""
    for row in camp.experiments().get("runs") or []:
        if "vega:ladder/floor" in (row.get("source"), row.get("rule")) \
                or row.get("kind") == "floor":
            return row.get("primary")
    return None


# --------------------------------------------------------------------------- the ledger

def ledger_row(run_id, rung, record, history, spec, findings, check=None):
    """The run's ledger row (run-contract §4). `expect_held` is written pending (null) with
    the rung's `expect_check`; `_append_ledger` judges it against the ledger the row joins. A
    trial's expect is its study's, judged once on the study's record."""
    metrics = spec.get("metrics") or {}
    primary = metrics.get("primary")
    last = history[-1] if history else {}
    # The row must describe the epoch whose weights were SAVED, not the epoch with the best
    # number. Early stopping only counts an improvement that clears `min_delta`, so a later
    # epoch can hold a marginally better metric that never became a checkpoint — and reporting
    # that number beside `checkpoint_digest` would publish a metric belonging to a model
    # nobody kept. /vega-eval then loads best.pt and measures something else again.
    best_i = None
    saved = record.get("best_epoch")
    if saved is not None and history and 1 <= int(saved) <= len(history):
        best_i = int(saved) - 1
    elif primary and history:
        # no checkpoint epoch recorded (a run that never improved, or an older record): fall
        # back to the argmax and let the row say the two may differ
        mode = (metrics.get("selection") or {}).get("mode", "max")
        pick = max if mode == "max" else min
        scored = [(i, (r.get("valid") or {}).get(primary)) for i, r in enumerate(history)]
        scored = [(i, v) for i, v in scored if v is not None]
        best_i = pick(scored, key=lambda pair: pair[1])[0] if scored else None
    best = history[best_i] if best_i is not None else last
    valid = dict(best.get("valid") or {})
    train = dict(best.get("train") or {})
    gap = {}
    if train.get("loss") is not None and valid.get("loss") is not None:
        gap["loss"] = round(valid["loss"] - train["loss"], 6)
    return {
        "id": run_id,
        "kind": rung.kind if rung else "hpo",
        "family": rung.family if rung else None,
        "hypothesis": rung.hypothesis if rung else "",
        "source": rung.source if rung else "",
        "expect": rung.expect if rung else "",
        "delta": record.get("delta"),
        "delta_text": Delta.of(record.get("delta")).describe(),
        "status": record.get("status"),
        "epochs": record.get("epochs"),
        "stopped_by": record.get("stopped_by"),
        "best_epoch": record.get("best_epoch"),
        "time_s": record.get("time_s"),
        "primary_name": primary,
        "primary": valid.get(primary),
        "valid": valid,
        "train": train,
        "gap": gap,
        "params": _params_of(record),
        "diagnostics": [f.code for f in findings if f.fired],
        "diagnostics_detail": [f.as_json() for f in findings if f.fired or f.unavailable],
        "unavailable": [f.code for f in findings if f.unavailable],
        "expect_check": check.as_json(primary) if check is not None else None,
        "expect_held": None,
        "expect_evidence": ("" if check is not None else
                            f"inherited from {rung.study or 'its study'}: judged once on the "
                            "study's record (experiments.json hpo[])"
                            if rung and rung.kind == "hpo" else "no expect_check"),
        "spec_digest": record.get("spec_digest"),
        "checkpoint_digest": record.get("checkpoint_digest"),
        "code_digests": record.get("code_digests"),
        "reading": "",
    }


def _params_of(record):
    """This run's parameter count: its own smoke.json when it re-emitted, else the campaign's.

    The order matters — a trial that halved the width and read the campaign's count would
    report the baseline's parameters and win a tie it did not earn.
    """
    smoke = Path(record.get("package_dir", "")).parent / "smoke.json"
    if smoke.is_file():
        try:
            return json.loads(smoke.read_text(encoding="utf-8")).get("params_total")
        except json.JSONDecodeError:
            return None
    return None


def _append_ledger(camp: Campaign, row):
    """Append the row, judge every expect still pending (this row's, and the rows and studies
    that were waiting for the run this row records), write the ledger; the row as written."""
    ledger = camp.experiments()
    runs = [r for r in ledger.get("runs") or [] if r.get("id") != row["id"]]
    runs.append(row)
    ledger["runs"] = runs
    ledger["budget"] = dict(camp.campaign.get("budget") or {})
    ledger["budget"]["spent_min"] = round(
        sum(float(r.get("time_s") or 0) for r in runs) / 60.0, 3)
    rejudge(ledger, camp.spec)
    camp.write_experiments(ledger)
    return runs[-1]


def _delta_signature(raw) -> list:
    """What a delta changes — its normalised paths and targets — for comparing two deltas."""
    return sorted(json.dumps([normalise_path(e.path), e.to], sort_keys=True, default=str)
                  for e in Delta.of(raw))


def trial_rung(study: Rung, run_id, delta) -> Rung:
    """A trial's rung: its study's hypothesis, its expect inherited (judged once, on the
    study's record), its source `train:hpo/<study>` — what hpo.py hands every trial."""
    return Rung.from_json({"id": run_id, "kind": "hpo", "study": study.id,
                           "delta": list(delta or []), "hypothesis": study.hypothesis,
                           "expect": f"inherited from {study.id}",
                           "source": f"train:hpo/{study.id}"})


# --------------------------------------------------------------------------- replay

def replay(camp: Campaign, run_id, epochs, forge=None, out=print):
    """Re-run the first k epochs and compare with what was recorded.

    Determinism is a claim the backend makes (`spec_snapshot.json`'s tolerance); this is the
    only thing in the sub-chain that tests it, and the notebook's replay cell runs the same
    fit the same way.

    The spec is replayed **as the run resolved it**: shortening `max_epochs` would change the
    learning-rate schedule (a cosine or one-cycle curve is a function of it), so the replay
    stops itself at the epoch boundary once k records are in. No checkpoint is written: the
    run context carries no store.
    """
    run_dir = camp.run_dir(run_id)
    recorded = read_jsonl(run_dir / "history.jsonl")[:epochs]
    if not recorded:
        out(f"ERROR: {run_id} has no history.jsonl to replay against")
        return 2
    resolved = json.loads((run_dir / "netspec.resolved.json").read_text(encoding="utf-8"))
    tolerance = _tolerance(camp)

    package_dir = camp.package_dir_for(run_id)
    try:
        adapters = open_package(package_dir, camp.package, forge)
    except ImportError as exc:
        out(f"ERROR: the package {run_id} trained with cannot be driven: {exc}")
        return 2
    domain = adapters.package.domain
    spec_obj = domain.NetSpec.from_dict(resolved)
    model = adapters.build(spec_obj, run_dir=str(camp.root))

    fresh = []
    wanted = len(recorded)

    class _Enough(Exception):
        """Raised once the records the replay compares are in: the fit stops at an epoch
        boundary without the spec it trains under being changed."""

    class _Collect:
        def on_epoch(self, record):
            fresh.append(record.as_dict() if hasattr(record, "as_dict") else dict(record))
            if len(fresh) >= wanted:
                raise _Enough()

    try:
        adapters.trainer.fit(model, spec_obj, adapters.data, callbacks=_Collect(),
                             run=domain.RunContext())
    except _Enough:
        pass

    problems = []
    for i, (was, now) in enumerate(zip(recorded, fresh), start=1):
        for part in ("train", "valid"):
            for key, old in (was.get(part) or {}).items():
                new = (now.get(part) or {}).get(key)
                if old is None or new is None or not isinstance(old, (int, float)):
                    continue
                if abs(float(old) - float(new)) > tolerance + 1e-12:
                    problems.append(f"epoch {i} {part}.{key}: recorded {old!r}, replay {new!r} "
                                    f"(tolerance {tolerance})")
    if len(fresh) < len(recorded):
        problems.append(f"the replay produced {len(fresh)} epochs against {len(recorded)} "
                        "recorded")
    if problems:
        out(f"ERROR: the replay of {run_id} does not match its records — determinism is a "
            "claim, and this is the test of it. Find the cause (a seed the backend did not "
            "set, a nondeterministic op, a device change) before publishing.")
        for problem in problems[:10]:
            out(f"    {problem}")
        return 4
    out(f"OK: replay of {run_id}: {len(fresh)} epoch(s) match the records within the backend's "
        f"determinism tolerance ({tolerance})")
    return 0


def _tolerance(camp: Campaign):
    snapshot = camp.root / "spec_snapshot.json"
    device = str(camp.campaign.get("device") or "cpu").split(":")[0]
    if snapshot.is_file():
        try:
            resolved = json.loads(snapshot.read_text(encoding="utf-8")).get("_resolved") or {}
            row = (resolved.get("determinism") or {}).get(device) or {}
            if row.get("tolerance") is not None:
                return float(row["tolerance"])
        except json.JSONDecodeError:
            pass
    return float((camp.campaign.get("determinism") or {}).get("tolerance", 0.0))


# --------------------------------------------------------------------------- the shell

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--run", default=None, help="the run id from ladder.json")
    ap.add_argument("--delta", default=None,
                    help="a JSON delta: a trial of a planned study (H<study>-<n>), or the "
                         "planned rung's own delta; a run with no ladder rung is refused")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="this run's share of the campaign's wall clock")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--replay", default=None, metavar="RUN_ID")
    ap.add_argument("--epochs", type=int, default=3, help="epochs to replay")
    ap.add_argument("--grad-every", type=int, default=10,
                    help="sample the pre-clip gradient norm every k optimizer steps")
    ap.add_argument("--vega-forge-dir", default=None)
    a = ap.parse_args(argv)

    camp, err = Campaign.read(a.campaign)
    if err:
        print(f"ERROR: {err}")
        return 2

    if a.replay:
        return replay(camp, a.replay, a.epochs, forge=a.vega_forge_dir)

    if not a.run:
        print("ERROR: --run <id> (or --replay <id>) is required")
        return 2

    delta_override = None
    if a.delta:
        try:
            delta_override = json.loads(a.delta)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --delta is not valid JSON: {exc}")
            return 2

    try:
        _, _, code = run_one(camp, a.run, delta_override=delta_override,
                             budget_min=a.budget_min, device=a.device, resume=a.resume,
                             forge=a.vega_forge_dir, grad_every=a.grad_every)
    except PathError as exc:
        print(f"ERROR: {exc}")
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
