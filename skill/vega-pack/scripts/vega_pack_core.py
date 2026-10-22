"""
vega_pack_core — the bundle's value objects, its digests, and the tolerance the round trip is
held to.

`pack.py` and `apply_bundle_layer.py` both need these, and putting them here once is what keeps
the round-trip rule and the manifest's shape from being stated twice and drifting. Same
discipline as the rest of the chain's tooling: frozen value objects and pure functions, no
argv, no printing, no exits — the CLI shells decide what to say and what to return.

Three things are worth stating plainly, because they are the substance rather than the plumbing:

**The tolerance is a formula, not a knob.** `Tolerance` computes `1e-6 · max(1, max|score|)`
from the scores actually being compared, so it is absolute near zero and relative for large
regression values, and it carries the rule that produced it into the manifest. There is no
setter. A round trip that fails is a defect in the transformer or in a copied file, and
loosening the comparison would only move the defect into production.

**A digest is of a file, and the manifest says which file.** `FileDigest` pairs the path the
bundle knows a file by with the digest of what was actually written, so `MANIFEST.files` can be
checked from inside the bundle by a test that has no access to the chain that built it.

**What ships is what was evaluated.** `Provenance` carries the digests the upstream layers
recorded — the spec, the checkpoint, the recipe — beside the ones this stage computed, and
`disagreements()` is the whole of the check: a bundle whose weights are not the evaluated run's
weights is a different model wearing its numbers.

**The reading, not the task, drives serving** (bundle-contract.md §2.1). `readings()` is the
catalog's per-reading table (`vega/output@2`): link, decision, field patterns and their kinds,
the stored columns the round trip compares, the decided fields; `output_record.py` builds the
`vega/output@2` record from it, and the v2.2 per-kind field list has no copy left here.
`template_matches()` states once how a
`serving_templates` pattern matches a template file, so the renderer, the re-emit check and the
template tests cannot disagree about which files a pattern names.
"""
from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["Tolerance", "FileDigest", "Provenance", "RoundTrip", "BundlePaths",
           "digest_file", "digest_text", "now", "read_json", "read_parquet", "slugify",
           "readings", "field_kinds", "monitor_kinds_for",
           "serving_mode", "serving_templates", "template_matches", "matching_templates"]

HERE = Path(__file__).resolve().parent
CATALOG = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))


class PackError(RuntimeError):
    """A refusal that concerns what would ship. Never worked around — fixed upstream."""


# --------------------------------------------------------------------------- small helpers

def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def digest_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"{what} not found: {path}"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{what} could not be read ({path}): {exc}"


def read_parquet(path, columns=None):
    """A parquet file as a pandas frame, read through Arrow's own filesystem (A40).

    Given a bare local path, pandas opens a *Python* file object and hands it to Arrow, whose
    worker threads then call back into Python; one still doing so while the interpreter
    finalizes is ended mid-unwind and the process aborts at exit ('terminate called without an
    active exception') after its work is done. `pyarrow.fs.LocalFileSystem` keeps Python out of
    Arrow's threads (`vega/scripts/tests/exit_abort_probe.py` is the reproducer)."""
    import pandas as pd
    from pyarrow import fs
    return pd.read_parquet(str(Path(path).resolve()), columns=columns,
                           filesystem=fs.LocalFileSystem())


def slugify(*parts) -> str:
    """A directory name from the dataset and the label, safe on every filesystem."""
    out = []
    for part in parts:
        cleaned = "".join(c if (c.isalnum() or c in "-_") else "-" for c in str(part or ""))
        cleaned = "-".join(w for w in cleaned.split("-") if w)
        if cleaned:
            out.append(cleaned.lower())
    return "-".join(out) or "bundle"


# --------------------------------------------------------------------------- the tolerance

@dataclass(frozen=True)
class Tolerance:
    """How close the bundle's predictions must be to the ones the evaluation stored.

    The contract's formula, computed from the scores under comparison rather than chosen:
    `|Δ| ≤ 1e-6 · max(1, max|score|)`. Absolute near zero, so a probability of 0.0 cannot drift
    by a relative epsilon of nothing; relative above 1, so a regression predicting in the
    thousands is not held to a bound that floating point cannot meet.
    """

    value: float
    rule: str
    precision: str
    scale: float

    @staticmethod
    def for_scores(scores, precision: str = "fp32") -> "Tolerance":
        rule = CATALOG["roundtrip"]["rule"]
        base = float(CATALOG["roundtrip"]["value"])
        try:
            biggest = max(abs(float(s)) for s in _flatten(scores)) if len(scores) else 0.0
        except (TypeError, ValueError):
            biggest = 0.0
        if not math.isfinite(biggest):
            biggest = 0.0
        scale = max(1.0, biggest)
        return Tolerance(value=base * scale, rule=rule, precision=precision, scale=scale)

    def holds(self, a, b) -> bool:
        return abs(float(a) - float(b)) <= self.value

    def as_json(self) -> dict:
        return {"rule": self.rule, "value": self.value, "precision": self.precision,
                "scale": self.scale}


def _flatten(values):
    for v in values:
        if isinstance(v, (list, tuple)):
            yield from _flatten(v)
        else:
            yield v


# --------------------------------------------------------------------------- what shipped

@dataclass(frozen=True)
class FileDigest:
    """A file the bundle carries, by the name the bundle knows it by."""

    name: str
    digest: str
    bytes: int

    @staticmethod
    def of(bundle_dir, name) -> "FileDigest":
        path = Path(bundle_dir) / name
        return FileDigest(name=str(name), digest=digest_file(path),
                          bytes=path.stat().st_size)


@dataclass(frozen=True)
class Provenance:
    """What the chain said should ship, against what this bundle actually copied.

    Each entry is `(what, recorded upstream, found here)`. `recorded` is None where the chain
    keeps no digest for that file — `fitted.json` is the named case, and the shaper's twin
    check is its test instead.
    """

    entries: tuple = field(default_factory=tuple)

    def with_check(self, what, recorded, found) -> "Provenance":
        return Provenance(self.entries + ((what, recorded, found),))

    def disagreements(self) -> list:
        """The files whose contents are not the ones the chain recorded.

        This is the definition of shipping the wrong model, so it is a list of sentences a
        user can act on rather than a boolean.
        """
        out = []
        for what, recorded, found in self.entries:
            if recorded and found and recorded != found:
                out.append(f"{what}: the chain recorded {recorded} and the bundle carries "
                           f"{found} — this is not the file the evaluation measured, and a "
                           "bundle built from it would wear numbers it did not earn")
        return out

    def unverifiable(self) -> list:
        return [what for what, recorded, _ in self.entries if not recorded]


@dataclass(frozen=True)
class RoundTrip:
    """The result of running the bundle's own predict path over the evaluation's test rows."""

    rows: int
    source: str
    tolerance: Tolerance
    max_abs_diff: float
    status: str
    shaper_transform_check: str
    first_bad: Mapping | None = None
    columns: int = 1        # head outputs compared per row
    labels: int = 0         # v2.2: rows whose decoded class was compared with the stored argmax
    decisions: int = 0      # rows whose decided fields were compared with the evaluation's
    boundary: Mapping | None = None   # window + lag: {"rows", "reason"} excluded (§5)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def decided(self) -> int:
        """Rows whose decision was compared; the v2.2 `labels` count is the same thing."""
        return self.decisions or self.labels

    def as_json(self) -> dict:
        out = {"rows": self.rows, "source": self.source,
               "tolerance": self.tolerance.as_json(),
               "max_abs_diff": self.max_abs_diff, "status": self.status,
               "columns_compared": self.columns, "decisions_compared": self.decided,
               "labels_compared": self.decided,
               "shaper_transform_check": self.shaper_transform_check}
        if self.boundary:
            out["boundary"] = dict(self.boundary)
        if self.first_bad:
            out["first_disagreement"] = dict(self.first_bad)
        return out


@dataclass(frozen=True)
class BundlePaths:
    """Where everything in the bundle lives, so no caller spells a path twice."""

    root: Path
    package: str

    @property
    def manifest(self) -> Path:
        return self.root / "MANIFEST.json"

    @property
    def pkg(self) -> Path:
        return self.root / self.package

    @property
    def examples(self) -> Path:
        return self.root / "examples"

    @property
    def tests(self) -> Path:
        return self.root / "tests"

    def missing(self, required: Sequence[str]) -> list:
        return [name for name in required if not (self.root / name).exists()]


# --------------------------------------------------------------------------- the readings

def readings() -> dict:
    """The per-reading table of `vega/output@2` (bundle-contract.md §2.1), keyed by reading.

    A copy, so a caller filling a record from it cannot edit the catalog every later caller
    reads.
    """
    return copy.deepcopy(CATALOG["output_readings"]["readings"])


def field_kinds() -> dict:
    """Field kind -> its JSON key, shape and visibility in a `RowPrediction` (§7.1)."""
    return copy.deepcopy(CATALOG["field_kinds"])


def monitor_kinds_for(reading: str) -> list:
    """The monitor check kinds a plan may use for this reading (bundle-contract.md §7.5).

    A kind whose `readings` is null applies to every reading; an unknown reading has none.
    """
    if reading not in CATALOG["output_readings"]["readings"]:
        return []
    return [kind for kind, row in CATALOG["monitor_kinds"].items()
            if row.get("readings") is None or reading in row["readings"]]


def serving_mode(mode: str) -> dict:
    return dict((CATALOG["serving_modes"] or {}).get(mode or "batch")
                or CATALOG["serving_modes"]["batch"])


# --------------------------------------------------------------------------- serving templates

def serving_templates() -> dict:
    """Destination -> glob patterns over the backend's `templates/serving/` (R-D2)."""
    return {dest: list(patterns) for dest, patterns in CATALOG["serving_templates"].items()}


def template_matches(pattern: str, rel: str) -> bool:
    """Whether a `serving_templates` pattern names a template path (relative, `/`-separated).

    Segment by segment: the two must have the same number of segments and each segment must
    match under `fnmatch.fnmatchcase`. So `*` never crosses a `/` — `package/*.py` does not
    name `package/inspection/codec.py` — and it does match a leading dot (`bundle/*` names
    `bundle/.dockerignore`).
    """
    want = str(pattern).split("/")
    have = str(rel).replace("\\", "/").split("/")
    return len(want) == len(have) and all(
        fnmatch.fnmatchcase(part, glob) for part, glob in zip(have, want))


def matching_templates(root) -> dict:
    """Destination -> the sorted template files under `root` its patterns name.

    `root` is the backend's `templates/serving/` directory. A file named by no pattern is not
    rendered; a pattern naming no file is legal until the owning package lands its template.
    """
    root = Path(root)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts)
    return {dest: [rel for rel in files if any(template_matches(p, rel) for p in patterns)]
            for dest, patterns in serving_templates().items()}
