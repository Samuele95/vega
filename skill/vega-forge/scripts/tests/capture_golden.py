#!/usr/bin/env python3
"""
capture_golden.py — the golden captures the v2.3 backend refactor is measured against
(BUILD-PLAN B01; D1 §17.3, WP-B0).

    python3 capture_golden.py [--out DIR] [--work DIR] [--commit REV] [--cids-root DIR]
                              [--verify] [--keep-work]

The capture runs from a `git worktree` of the pre-refactor commit (`c849f8b`), so the
templates, the fixture builder and the shaper run **as they were**, beside a working tree whose
templates are being rewritten; of the working tree, only this script runs. It writes, under
`--out` (default `golden/` beside this script):

  orders-R0.history.jsonl        the orders fixture's spec (R0, no delta) trained on CPU for 3
                                 epochs by the old `fit` with its defaults: one record per line
                                 as /vega-train writes it, less the wall-clock `time_s`
  orders-valid-logit-range.json  per epoch, min/max of the valid logits and whether one lies
                                 outside [0, 1] (the old metrics read an all-[0, 1] score as a
                                 probability, E18), plus the provenance (commit, versions,
                                 part digests, spec overrides)
  orders-smoke.json              `emit_backend.py --smoke --device cpu` on the orders fixture
  cic-r7-smoke.json              the same on the CIC-IDS2017 R7 spec, from `--cids-root` (or
                                 `CIDS_ROOT`); `params_total` and `per_block` must equal the
                                 bundle's `records/architecture.json` before it is written

Two captures on one machine and environment are byte-identical; `--verify` is that test: it
captures into scratch and compares each file with `--out`, byte for byte. The history is fitted
twice, with the logit probe and without it, and the two must agree (the probe is inert).

Exit: 0 captured or identical; 1 `--verify` found a difference; 2 input error (no repository,
unknown commit, no R7 bundle); 3 a capture step failed (the scratch directory is kept).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
TESTS_REL = "vega-forge/scripts/tests"
COMMIT = "c849f8b"
EPOCHS = 3
#: applied under `training`: the replay's own override (`vega-train --replay`) plus the device
SPEC_OVERRIDES = {"max_epochs": EPOCHS, "device": "cpu"}
WALL_CLOCK = ("time_s",)
HISTORY = "orders-R0.history.jsonl"
LOGITS = "orders-valid-logit-range.json"
SMOKE = "orders-smoke.json"
CIC_SMOKE = "cic-r7-smoke.json"
CIC_BUNDLE = "bundle/cicids2017-label-R7"


class CaptureError(Exception):
    """A capture step failed; the message names the step and carries the output's tail."""


class InputError(Exception):
    """The capture cannot start: a missing repository, commit or bundle."""


# --------------------------------------------------------------------------- helpers

def _env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"   # never write into the tree or a linked skill
    return env


def _run(cmd, cwd, what):
    proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=_env(),
                          capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
        raise CaptureError(f"{what} exited {proc.returncode}:\n{tail}")
    return proc.stdout


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise InputError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _json_bytes(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _history_bytes(records):
    """The format /vega-train's `append_jsonl` writes, one record per line, less wall clock."""
    lines = []
    for record in records:
        row = {k: v for k, v in record.items() if k not in WALL_CLOCK}
        lines.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return "".join(lines).encode("utf-8")


def _environment():
    import numpy
    import pandas
    import pyarrow
    import torch
    return {"python": platform.python_version(), "machine": platform.machine(),
            "torch": torch.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__, "pyarrow": pyarrow.__version__,
            "torch_threads": int(torch.get_num_threads())}


def _provenance(sha, extra):
    base = {"schema": "vega/golden-capture@1", "commit": sha,
            "tool": f"{TESTS_REL}/capture_golden.py", "device": "cpu",
            "environment": _environment()}
    base.update(extra)
    return base


# --------------------------------------------------------------------------- the old tree

class PreRefactorTree:
    """A `git worktree` of the pre-refactor commit, with the external skills the working tree
    links (they are not in git) linked the same way, removed again on exit."""

    def __init__(self, repo: Path, sha: str, path: Path):
        self.repo, self.sha, self.path = repo, sha, path
        self._links = []

    def __enter__(self):
        _git(self.repo, "worktree", "add", "--detach", "--quiet", str(self.path), self.sha)
        try:
            for entry in sorted(self.repo.iterdir()):
                target = self.path / entry.name
                if entry.is_symlink() and not target.exists() and not target.is_symlink():
                    os.symlink(os.readlink(entry), target)
                    self._links.append(target)
        except OSError:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for link in self._links:
            os.unlink(link)
        proc = subprocess.run(["git", "-C", str(self.repo), "worktree", "remove", "--force",
                               str(self.path)], capture_output=True, text=True)
        if proc.returncode != 0:
            shutil.rmtree(self.path, ignore_errors=True)
            subprocess.run(["git", "-C", str(self.repo), "worktree", "prune"],
                           capture_output=True, text=True)
        return False

    def script(self, rel):
        return self.path / rel


def _emit_smoke(tree, spec, pipeline, out, cwd):
    """The pre-refactor `emit_backend.py --smoke` on CPU; returns smoke.json as a dict."""
    _run([sys.executable, tree.script("vega-forge/scripts/emit_backend.py"), "--spec", spec,
          "--pipeline", pipeline, "--out", out, "--smoke", "--device", "cpu"],
         cwd, f"emit_backend.py --smoke ({Path(spec).name})")
    return json.loads((Path(out) / "smoke.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- orders

def _build_orders(tree, work):
    fx = work / "orders"
    fx.mkdir(parents=True)
    shutil.copytree(tree.script(f"{TESTS_REL}/fixtures/dot-claude"), fx / ".claude")
    _run([sys.executable, tree.script(f"{TESTS_REL}/make_fixture.py"), fx], fx,
         "make_fixture.py")
    pipes = sorted(fx.glob("blueprint-runs/*/pipeline.json"))
    if len(pipes) != 1:
        raise CaptureError(f"the fixture holds {len(pipes)} pipeline.json, expected one")
    return fx, pipes[0]


def _fit(fx, run, probe):
    """One pre-refactor `fit` in a fresh interpreter (this script's `fit-driver`)."""
    result = run.parent / f"fit-{'probed' if probe else 'control'}.json"
    cmd = [sys.executable, Path(__file__).resolve(), "fit-driver", "--run", run,
           "--spec", fx / "netspec.json", "--result", result]
    _run(cmd + (["--probe"] if probe else []), fx, "the pre-refactor fit")
    return json.loads(result.read_text(encoding="utf-8"))


def _logit_record(probed, sha, parts):
    epochs = []
    for r in probed["ranges"]:
        outside = r["min"] < 0.0 or r["max"] > 1.0
        epochs.append({**r, "outside_unit_interval": outside})
    return {
        "schema": "vega/golden-logit-range@1",
        "rule": ("E18: at the capture commit, metrics.py read a binary score whose values all "
                 "lie in [0, 1] as a probability. The valid probability metrics (log_loss, "
                 "brier, ece, the threshold metrics) of an epoch whose logits all lie in [0, 1] "
                 "are expected to differ once logits are read as logits (E18 fixed); they are "
                 "compared only where outside_unit_interval is true (D1 §17.3)."),
        "epochs": epochs,
        "expected_to_differ_e18": [e["epoch"] for e in epochs if not e["outside_unit_interval"]],
        "fit": probed["fit"],
        "capture": _provenance(sha, {
            "spec": "the orders fixture's netspec.json (make_fixture.py), run R0: no delta",
            "spec_overrides": {f"training.{k}": v for k, v in SPEC_OVERRIDES.items()},
            "fit_call": "fit(model, spec, data, callbacks) with every other argument at its "
                        "default: no budget, no checkpoint, grad_sample_every 0",
            "dropped_fields": list(WALL_CLOCK),
            "parts": parts,
            "control": "a second fit without the logit probe wrote identical records"}),
    }


def capture_orders(tree, work, sha):
    fx, pipe = _build_orders(tree, work)
    run = fx / "run"
    smoke = _emit_smoke(tree, fx / "netspec.json", pipe, run, fx)
    probed = _fit(fx, run, probe=True)
    control = _fit(fx, run, probe=False)
    history = _history_bytes(probed["records"])
    if history != _history_bytes(control["records"]):
        raise CaptureError("the logit probe changed the training records: the fit with the "
                           "probe and the fit without it disagree, so the probe is not inert")
    if len(probed["records"]) != EPOCHS:
        raise CaptureError(f"the fit ran {len(probed['records'])} epochs, expected {EPOCHS}")
    spec = json.loads((fx / "netspec.json").read_text(encoding="utf-8"))
    parts = {p: _digest(fx / spec["data_contract"]["parts"][p]) for p in ("train", "valid")}
    smoke["_capture"] = _provenance(sha, {"spec": "the orders fixture's netspec.json",
                                          "command": "emit_backend.py --smoke --device cpu"})
    return {HISTORY: history, LOGITS: _json_bytes(_logit_record(probed, sha, parts)),
            SMOKE: _json_bytes(smoke)}


# --------------------------------------------------------------------------- CIC R7

def _cids_root(given):
    root = given or os.environ.get("CIDS_ROOT")
    if not root:
        return None
    root = Path(root).resolve()
    if not (root / CIC_BUNDLE / "netspec.json").is_file():
        raise InputError(f"{root} holds no {CIC_BUNDLE}/netspec.json (--cids-root/CIDS_ROOT "
                         "must name the tree the R7 bundle, its pipeline and parts live in)")
    return root


def capture_cic(tree, work, sha, root):
    bundle = root / CIC_BUNDLE
    spec_path = bundle / "netspec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    pipeline = root / spec["input"]["pipeline"]
    smoke = _emit_smoke(tree, spec_path, pipeline, work / "cic-r7", root)
    record = json.loads((bundle / "records" / "architecture.json").read_text("utf-8"))["smoke"]
    for key in ("params_total", "per_block"):
        if smoke[key] != record[key]:
            raise CaptureError(f"the pre-refactor tree does not reproduce the bundle's recorded "
                               f"{key}: {smoke[key]!r} against {record[key]!r}")
    smoke["_capture"] = _provenance(sha, {
        "spec": f"{CIC_BUNDLE}/netspec.json", "spec_digest": _digest(spec_path),
        "command": "emit_backend.py --smoke --device cpu, run from the CIDS root",
        "architecture_record": {"file": f"{CIC_BUNDLE}/records/architecture.json",
                                "params_total": record["params_total"],
                                "per_block_equal": True}})
    return {CIC_SMOKE: _json_bytes(smoke)}


# --------------------------------------------------------------------------- the driver

class _Collect:
    """The callbacks the pre-refactor `fit` calls: the records, and the logit probe."""

    def __init__(self, probe):
        self.records, self.ranges, self.probe = [], [], probe

    def on_epoch(self, record):
        row = record.as_dict() if hasattr(record, "as_dict") else dict(record)
        self.records.append(row)
        if self.probe is not None:
            self.ranges.append({"epoch": row["epoch"], **self.probe()})

    def on_first_step(self, record):
        pass


class _LogitProbe:
    """The valid logits the epoch was scored on, re-read in eval mode with the batching `fit`
    scores valid with. Every RNG is restored afterwards, so the next epoch cannot tell."""

    def __init__(self, model, spec, mods, part):
        self.model, self.spec, self.mods, self.part = model, spec, mods, part
        self.bs = int(spec.to_dict()["training"].get("batch_size") or 32)

    def __call__(self):
        import random

        import numpy as np
        import torch
        states = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        try:
            batches = self.mods["data"].iter_batches(self.spec, self.part, batch_size=self.bs)
            scores, _ = self.mods["loop"].predict(self.model, self.spec, batches)
        finally:
            torch.set_rng_state(states[0])
            np.random.set_state(states[1])
            random.setstate(states[2])
        s = np.asarray(scores, dtype=np.float64)
        return {"min": float(s.min()), "max": float(s.max()), "n": int(s.shape[0])}


def _import_emitted(run: Path):
    import importlib
    pkg_dir = next(p.parent for p in sorted(run.glob("*/blueprint-map.json")))
    sys.path.insert(0, str(run))
    pkg = importlib.import_module(pkg_dir.name)
    bmap = json.loads((pkg_dir / "blueprint-map.json").read_text(encoding="utf-8"))
    by_role = {e.get("role"): e["file"] for e in bmap["entries"] if e.get("role")}
    roles = {"network": "Model", "data": "DataSource", "loop": "Trainer"}
    return pkg, {k: importlib.import_module(f"{pkg_dir.name}." +
                                            by_role[r][:-3].replace("/", "."))
                 for k, r in roles.items()}


def fit_driver(a):
    run = Path(a.run).resolve()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    spec["training"] = {**(spec.get("training") or {}), **SPEC_OVERRIDES}
    pkg, mods = _import_emitted(run)
    ns = pkg.domain.NetSpec.from_dict(spec)
    model = mods["network"].build_model(ns, run_dir=str(run))
    probe = _LogitProbe(model, ns, mods, pkg.domain.Part.VALID) if a.probe else None
    collect = _Collect(probe)
    res = mods["loop"].fit(model, ns, mods["data"], callbacks=collect)
    out = {"records": collect.records, "ranges": collect.ranges,
           "fit": {"epochs_run": int(res.epochs_run), "best_epoch": res.best_epoch,
                   "best_value": res.best_value, "stopped_by": str(getattr(
                       res.stopped_by, "value", res.stopped_by))}}
    Path(a.result).write_text(json.dumps(out), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- the shell

def verify(captured, out):
    bad = 0
    for name, data in captured.items():
        path = out / name
        old = path.read_bytes() if path.is_file() else None
        if old == data:
            print(f"OK: {name} byte-identical ({len(data)} bytes)")
            continue
        bad += 1
        if old is None:
            print(f"ERROR: {name}: no committed golden at {path}")
            continue
        at = next((i for i, (x, y) in enumerate(zip(old, data)) if x != y),
                  min(len(old), len(data)))
        line = old[:at].count(b"\n") + 1
        print(f"ERROR: {name} differs from {path}: first difference at byte {at} (line {line}); "
              f"{len(old)} bytes committed, {len(data)} captured")
    return 1 if bad else 0


def _inputs(a):
    """(repository, full commit id, CIDS root or None) — all checked before anything is made."""
    repo = Path(_git(HERE, "rev-parse", "--show-toplevel"))
    sha = _git(repo, "rev-parse", "--verify", f"{a.commit}^{{commit}}")
    return repo, sha, _cids_root(a.cids_root)


def _capture(repo, sha, root, scratch):
    with PreRefactorTree(repo, sha, scratch / "tree") as tree:
        print(f"OK: worktree of {sha[:12]} at {tree.path}")
        captured = capture_orders(tree, scratch, sha)
        print(f"OK: orders captured — smoke, {EPOCHS} CPU epochs and their valid-logit ranges")
        if root is None:
            print(f"SKIP: {CIC_SMOKE} not captured: neither --cids-root nor CIDS_ROOT is set")
        else:
            captured.update(capture_cic(tree, scratch, sha, root))
            print(f"OK: CIC R7 smoke captured from {root}; params_total and per_block equal "
                  f"{CIC_BUNDLE}/records/architecture.json")
    return captured


def _publish(captured, a):
    out = Path(a.out).resolve()
    if a.verify:
        return verify(captured, out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in captured.items():
        (out / name).write_bytes(data)
        print(f"OK: wrote {out / name} ({len(data)} bytes)")
    return 0


def _parse(argv):
    if argv and argv[0] == "fit-driver":
        ap = argparse.ArgumentParser(prog="capture_golden.py fit-driver",
                                     description="internal: one pre-refactor fit")
        ap.add_argument("command")
        for flag in ("--run", "--spec", "--result"):
            ap.add_argument(flag, required=True)
        ap.add_argument("--probe", action="store_true")
        return ap.parse_args(argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(GOLDEN), help="the golden directory")
    ap.add_argument("--work", default=None, help="scratch parent (default: a temp directory)")
    ap.add_argument("--commit", default=COMMIT, help="the pre-refactor commit")
    ap.add_argument("--cids-root", default=None, help="the tree holding the CIC R7 bundle")
    ap.add_argument("--verify", action="store_true", help="compare, byte for byte, with --out")
    ap.add_argument("--keep-work", action="store_true", help="keep the scratch directory")
    a = ap.parse_args(argv)
    a.command = None
    return a


def main(argv=None):
    a = _parse(list(sys.argv[1:] if argv is None else argv))
    if a.command == "fit-driver":
        return fit_driver(a)
    try:
        repo, sha, root = _inputs(a)
    except InputError as e:
        print(f"ERROR: {e}")
        return 2
    parent = Path(a.work).resolve() if a.work else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="golden-", dir=parent))
    try:
        captured = _capture(repo, sha, root, scratch)
    except InputError as e:
        print(f"ERROR: {e}")
        shutil.rmtree(scratch, ignore_errors=True)
        return 2
    except CaptureError as e:
        print(f"ERROR: {e}")
        print(f"ERROR: the scratch directory is kept for inspection: {scratch}")
        return 3
    if not a.keep_work:
        shutil.rmtree(scratch, ignore_errors=True)
    return _publish(captured, a)


if __name__ == "__main__":
    sys.exit(main())
