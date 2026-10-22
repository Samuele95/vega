"""demo_pipeline.ingestion.data — the DataSource adapter (role: DataSource)."""
from __future__ import annotations

import json
from pathlib import Path

from ..domain import NetSpec, Part, SealedPartError

__all__ = ["ShapedParts"]

SEAL_FILE = "test_seal.json"


class ShapedParts:
    """The parts by name; the test part only through the seal."""

    def __init__(self, root: str = "."):
        self._root = Path(root)

    def load_part(self, spec: NetSpec, part: Part) -> tuple:
        if Part(part) is Part.TEST:
            raise SealedPartError("load_part refuses the test part; the evaluation opens it "
                                  "through load_sealed_part")
        return self._read(spec, Part(part))

    def load_sealed_part(self, spec: NetSpec, token: str, seal_path: str = SEAL_FILE) -> tuple:
        seal = json.loads((self._root / seal_path).read_text(encoding="utf-8"))
        if seal.get("token") != token:
            raise SealedPartError("the token does not match the seal")
        return self._read(spec, Part.TEST)

    def _read(self, spec: NetSpec, part: Part) -> tuple:
        return _rows(self._root / f"{spec.name}.{part.value}.json")


def _rows(path: Path) -> tuple:
    return tuple(json.loads(path.read_text(encoding="utf-8")))
