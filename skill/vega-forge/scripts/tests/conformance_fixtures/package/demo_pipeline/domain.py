"""demo_pipeline.domain — the artifacts as values (frozen dataclasses and enums)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Part(str, Enum):
    """The three parts the shaper wrote; the third is sealed."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class SealedPartError(RuntimeError):
    """Raised when anything but the sealed door asks for the test part."""


@dataclass(frozen=True)
class NetSpec:
    """The spec, as one value."""

    name: str
    seed: int = 0


@dataclass(frozen=True)
class RunContext:
    """What a fit needs beyond the model, the spec and the data: one value, not four
    parameters (C12)."""

    budget_min: float | None = None
    checkpoint_dir: str | None = None


@dataclass(frozen=True)
class Metrics:
    """The scores of one part."""

    part: Part
    values: tuple = ()
