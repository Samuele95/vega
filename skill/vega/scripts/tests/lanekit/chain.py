"""
The chain's tools a lane fixture drives, run from a skills tree (lanekit.fixture).

    run_tool([python, script, ...], "shaper")        # echoes OK:/WARN:/ERROR:, refuses a non-zero exit
    ws = Workspace(skills, work, "landcover")
    verification = ws.shape(recipe_path, model, seed) # shape.py, verify_shape.py, the twin, --check
    evidence = ws.geometry_evidence(raw_path, seed)  # dataset-forge/scripts/geometry.py
    report = ws.analysis_evidence(raw_path, model, series)   # data-lens/scripts/analysis.py
    writer = platform_writer(skills)                 # domain-forge's apply_layer module

Every refusal is a `FixtureError` whose message starts `ERROR:` and names the step. Standard
library only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

__all__ = ["FixtureError", "run_tool", "script", "platform_writer", "sha_file", "Workspace"]


class FixtureError(RuntimeError):
    """A step of the fixture refused; the message starts `ERROR:` and names the step."""


def run_tool(argv: Sequence[Any], what: str, cwd: Path | None = None, timeout: int = 900) -> str:
    """Run one tool of the chain; echo its markers; refuse on a non-zero exit."""
    proc = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout,
                          cwd=str(cwd) if cwd else None)
    for line in proc.stdout.splitlines():
        if line.startswith(("OK:", "WARN:", "ERROR:")):
            print(f"  {what}| {line}", flush=True)
    if proc.returncode != 0:
        tail = (proc.stdout[-2000:] + "\n" + proc.stderr[-2000:]).strip()
        message = f"ERROR: {what} exited {proc.returncode}; the fixture cannot be built on it\n{tail}"
        print(message.splitlines()[0], flush=True)
        raise FixtureError(message)
    return proc.stdout


def script(skills: Path, rel: str) -> Path:
    """`<skills>/<rel>`, refused when the tree does not hold it."""
    path = Path(skills) / rel
    if not path.is_file():
        raise FixtureError(f"ERROR: {rel} is not in the skills tree {skills}; the fixture drives it")
    return path


def platform_writer(skills: Path) -> Any:
    """domain-forge's apply_layer module (the platform writer and the domain digest)."""
    path = script(skills, "domain-forge/scripts/apply_layer.py")
    spec = importlib.util.spec_from_file_location("df_apply_layer", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Workspace:
    """A lane fixture's work directory and the skills tree whose tools it runs: `shaped/` holds
    the shaper's output, `.fixture/` the tools' scratch (reports, the twin module, evidence)."""
    skills: Path
    work: Path
    name: str

    @property
    def shaped(self) -> Path:
        return self.work / "shaped"

    @property
    def scratch(self) -> Path:
        return self.work / ".fixture"

    def prepare(self) -> "Workspace":
        for folder in (self.shaped, self.scratch):
            folder.mkdir(parents=True, exist_ok=True)
        return self

    def script(self, rel: str) -> Path:
        return script(self.skills, rel)

    def shape(self, recipe_path: Path, model: Path, seed: int) -> dict:
        """shape.py on the recipe (sources resolved against the model's layers), then its checks:
        verify_shape.py, the twin check of the emitted transform, and the determinism re-run.
        Returns the verification record (verify_shape.py's report plus `determinism` and
        `twin`), also written to `shaped/verification.json`."""
        shaper = self.script("dataset-shaper/scripts/shape.py")
        verify = self.script("dataset-shaper/scripts/verify_shape.py")
        py, shaped = sys.executable, self.shaped
        run_tool([py, shaper, "--recipe", recipe_path, "--model", model, "--out-dir", shaped,
                  "--format", "parquet", "--seed", seed], "shaper")
        report = self.scratch / "verify.json"
        run_tool([py, verify, "--recipe", recipe_path, "--out-dir", shaped, "--model", model, "--json", report],
                 "verify")
        twin = self.scratch / "transform"
        twin.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shaped / "fitted.json", twin / "fitted.json")      # the module reads it beside itself
        module = twin / "preprocess.py"
        run_tool([py, shaper, "--transform-only", "--recipe", recipe_path, "--fitted", twin / "fitted.json",
                  "--emit", module, "--out-dir", shaped], "transform")
        run_tool([py, verify, "--transform-only", "--recipe", recipe_path, "--out-dir", shaped,
                  "--module", module, "--part", "test"], "twin")
        # the verifier's report where the apply script reads it, then --check records its own
        # determinism verdict in it (F45); the twin check above passed or run_tool raised
        shutil.copy2(report, shaped / "verification.json")
        run_tool([py, shaper, "--check", "--out-dir", shaped], "determinism")
        verification = json.loads((shaped / "verification.json").read_text(encoding="utf-8"))
        if verification.get("determinism") != "pass":
            raise RuntimeError(f"shape.py --check recorded determinism {verification.get('determinism')!r}")
        verification["twin"] = "pass"
        (shaped / "verification.json").write_text(json.dumps(verification, indent=1) + "\n", encoding="utf-8")
        return verification

    def geometry_evidence(self, raw_path: Path, seed: int) -> dict:
        """geometry.py's evidence on the raw rows (the real data sub-chain)."""
        out = self.scratch / "geometry.json"
        run_tool([sys.executable, self.script("dataset-forge/scripts/geometry.py"), raw_path, "--out", out,
                  "--seed", seed], "geometry", cwd=self.scratch)
        return json.loads(out.read_text(encoding="utf-8"))

    def analysis_evidence(self, raw_path: Path, model: Path, series: str | None) -> Path:
        """analysis.py's evidence on the raw rows, read against the geometry model; returns the
        report `apply_analysis_layer.py` splices."""
        out = self.scratch / "analysis.json"
        run_tool([sys.executable, self.script("data-lens/scripts/analysis.py"), raw_path, "--model", model,
                  "--out", out, "--seed", 7, "--figures", self.scratch / "figures",
                  *(["--series-column", series] if series else [])], "analysis", cwd=self.scratch)
        return out
