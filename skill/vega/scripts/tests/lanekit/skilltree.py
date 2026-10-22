"""
The skills tree a matrix runs against: `<matrix>/skills/`, a COPY of the repository's tracked
files plus the external `domain-forge` skill (D6 §2.3).

The acceptance scripts compute `SKILLS` from their own location and `make_fixture.py` resolves
`parents[3]`, so the program never symlinks: it copies. What it copies is what git knows about
(`git ls-files` — tracked, plus untracked files that are not ignored, so a work package's new
files are exercised before they are committed), never bytecode, never an acceptance work dir.

domain-forge is not part of the package (D6 §18 G1). It is taken from `$DOMAIN_FORGE_DIR`,
else the sibling of the repository's skills, else `~/.claude/skills/domain-forge`, and the
digests of the four files the chain calls are pinned in `<matrix>/external-pins.json`. A run
whose pins differ from the last green run's prints a `WARN:` naming both digests; it does not
fail — domain-forge is external, and the chain's own checks decide whether it still works.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from . import contracts

EXCLUDED_PARTS = {"__pycache__", ".vega-acceptance", ".pytest_cache", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


@dataclass(frozen=True)
class SkillTree:
    root: Path                      # <matrix>/skills
    repo: Path                      # the repository it was copied from
    domain_forge: Path              # the external skill it was copied from
    files: int                      # files copied
    pins: Mapping[str, str]         # domain-forge relpath -> sha256


def repo_root(start: Path) -> Path:
    """The repository holding `start`; outside git (the matrix's own copy of the skills), the
    nearest ancestor holding the lane registry."""
    out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        return Path(out.stdout.strip())
    for parent in [start, *start.resolve().parents]:
        if (parent / contracts.TESTS_DIR / "lanes.json").is_file():
            return parent
    raise FileNotFoundError(f"{start} is in no repository and under no skills tree")


def git_state(repo: Path) -> dict:
    """{commit, dirty} of the repository the tree is copied from (observed.json `git`)."""
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or None
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def tracked_files(repo: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others",
                          "--exclude-standard"], capture_output=True, check=True).stdout
    names = sorted({n for n in out.decode("utf-8").split("\0") if n})
    return [n for n in names if _kept(Path(n)) and (repo / n).is_file()]


def _kept(rel: Path) -> bool:
    return not (EXCLUDED_PARTS & set(rel.parts)) and rel.suffix not in EXCLUDED_SUFFIXES


def find_domain_forge(repo: Path, env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    candidates = [env.get("DOMAIN_FORGE_DIR"), repo / "domain-forge",
                  Path.home() / ".claude" / "skills" / "domain-forge"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "scripts" / "validate_model.py").is_file():
            return Path(candidate).resolve()
    raise FileNotFoundError(
        "domain-forge not found (set DOMAIN_FORGE_DIR, or install it beside the skills): every "
        "lane's layers are written and validated by it (D6 §18 G1)")


def build(matrix: Path, repo: Path, env: Mapping[str, str] | None = None) -> SkillTree:
    """(Re)build `<matrix>/skills` from the repository and domain-forge. The copy is replaced
    whole, so a file deleted upstream does not survive in the tree."""
    root = matrix / "skills"
    staging = matrix / "skills.building"
    shutil.rmtree(staging, ignore_errors=True)
    names = tracked_files(repo)
    for name in names:
        _copy(repo / name, staging / name)
    df = find_domain_forge(repo, env)
    df_files = [p for p in sorted(df.rglob("*")) if p.is_file() and _kept(p.relative_to(df))]
    for path in df_files:
        _copy(path, staging / "domain-forge" / path.relative_to(df))
    shutil.rmtree(root, ignore_errors=True)
    staging.rename(root)
    pins = {rel: contracts.sha256_file(root / "domain-forge" / rel)
            for rel in contracts.DOMAIN_FORGE_PINNED if (root / "domain-forge" / rel).is_file()}
    return SkillTree(root, repo, df, len(names) + len(df_files), pins)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst, follow_symlinks=True)


# --------------------------------------------------------------------------- external pins
def pins_path(matrix: Path) -> Path:
    return matrix / "external-pins.json"


def check_pins(matrix: Path, tree: SkillTree) -> list[str]:
    """`WARN:` lines for every pinned domain-forge file whose digest differs from the last
    green run's, and record the current pins."""
    path = pins_path(matrix)
    doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    green = (doc.get("last_green") or {}).get("files") or {}
    warns = [f"WARN: domain-forge changed since the last green run: {rel} "
             f"{green[rel]} -> {digest}"
             for rel, digest in tree.pins.items() if rel in green and green[rel] != digest]
    doc.update({"schema": contracts.EXTERNAL_PINS, "domain_forge": str(tree.domain_forge),
                "current": {"files": dict(tree.pins), "at": _now()}})
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return warns


def record_green(matrix: Path) -> None:
    """After a green run, the current pins become the reference the next run compares with."""
    path = pins_path(matrix)
    if not path.is_file():
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["last_green"] = doc.get("current")
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- digests
def digest_paths(root: Path, rels: Iterable[str]) -> dict[str, str | None]:
    """{relpath: digest} over files or directories under root (None when absent). A directory
    digests its files' relative names and contents, bytecode and scratch excluded."""
    return {rel: digest_path(root / rel) for rel in rels}


def digest_path(path: Path) -> str | None:
    if path.is_file():
        return contracts.sha256_file(path)
    if not path.is_dir():
        return None
    h = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file() and _kept(p.relative_to(path))):
        h.update(str(file.relative_to(path)).encode("utf-8") + b"\0")
        h.update(contracts.sha256_file(file).encode("ascii") + b"\n")
    return "sha256:" + h.hexdigest()


def tree_digest(root: Path, exclude_top: Iterable[str] = ()) -> str:
    """The digest of every file under `root`, relative names included; top-level entries in
    `exclude_top` are left out (the bundle pin excludes `showcase/`, D6 §6.5)."""
    skip = set(exclude_top)
    h = hashlib.sha256()
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = file.relative_to(root)
        if rel.parts[0] in skip:
            continue
        h.update(str(rel).encode("utf-8") + b"\0" + contracts.sha256_file(file).encode("ascii") + b"\n")
    return "sha256:" + h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
