"""demo_pipeline.model.store — the Registry adapter (role: ModelStore)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["ModelRegistry"]


class ModelRegistry:
    """Checkpoints as versioned artifacts, by digest."""

    def save(self, model, path, spec=None, epoch=None) -> str:
        blob = json.dumps({"params": list(model.param_count()), "epoch": epoch}).encode()
        Path(path).write_bytes(blob)
        return _digest(blob)

    def load(self, ref, expected_digest=None, spec=None, build=None, run_dir=None):
        blob = Path(ref).read_bytes()
        if expected_digest is not None and _digest(blob) != expected_digest:
            raise ValueError(f"{ref}: the digest does not match {expected_digest}")
        return build(spec, run_dir) if build is not None else json.loads(blob)


def _digest(blob: bytes) -> str:
    return "sha256:" + hashlib.sha256(blob).hexdigest()
