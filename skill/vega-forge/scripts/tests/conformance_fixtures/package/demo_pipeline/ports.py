"""demo_pipeline.ports — one Protocol per port the blueprint drew."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSource(Protocol):
    """Yield the parts by name; the test part only through the seal."""

    def load_part(self, spec, part) -> Any: ...
    def load_sealed_part(self, spec, token: str, seal_path: str = "test_seal.json") -> Any: ...


@runtime_checkable
class Recipe(Protocol):
    """Raw -> shaped with the fitted recipe."""

    def transform(self, df_raw, history=None) -> Any: ...


@runtime_checkable
class Model(Protocol):
    """Features -> outputs."""

    def score(self, batch: dict) -> Any: ...
    def param_count(self) -> tuple: ...


@runtime_checkable
class Trainer(Protocol):
    """Fit a Model on a DataSource under a spec."""

    def fit(self, model, spec, data, callbacks=None, run=None) -> Any: ...
    def evaluate(self, model, spec, loader) -> Any: ...


@runtime_checkable
class Evaluator(Protocol):
    """One sealed look."""

    def score_part(self, spec, part, checkpoint=None, model=None) -> Any: ...
    def score_sealed_test(self, spec, token: str, checkpoint=None, model=None,
                          seal_path: str = "test_seal.json") -> Any: ...


@runtime_checkable
class Registry(Protocol):
    """Checkpoints as versioned artifacts, by digest."""

    def save(self, model, path, spec=None, epoch=None) -> str: ...
    def load(self, ref, expected_digest=None, spec=None, build=None, run_dir=None) -> Any: ...


@runtime_checkable
class Predictor(Protocol):
    """Raw rows -> predictions."""

    def predict(self, rows, history=None) -> Any: ...


@runtime_checkable
class Monitor(Protocol):
    """Watch what the analysis found fragile."""

    def check(self, batch, predictions=None) -> tuple: ...
