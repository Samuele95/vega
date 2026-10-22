"""
show_ports — the two ports `/vega-show` reads the world through.

The stage reads two kinds of thing it did not write: the bundle's pinned records, and the
compiled report's labels. Each is a port with one adapter today — `records.BundleRecords` and
`report_labels.AuxLabelSource` — so the fact extractor, the role proposer and the resolver
never touch a file format. A future non-LaTeX report skill adds one `LabelSource` adapter (a
`JsonLabelSource`, say) and nothing else changes; a test hands in a store built from a mapping.

Both protocols are `runtime_checkable`, so a composition root can assert what it wires. The
two adapters subclass their port explicitly (`BundleRecords(RecordStore)`,
`AuxLabelSource(LabelSource)`), so a type checker holds their signatures to the port's; a test
double may still conform structurally. The import runs adapter → port, never back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from vega_show_core import FactRef, LabelKind

__all__ = ["RecordStore", "LabelSource", "LabelRef"]


@dataclass(frozen=True)
class LabelRef:
    """One label of the report, as the report records it: number, page, title, anchor, kind
    (`other` when neither the anchor nor the name carries a kind prefix)."""

    label: str
    number: str
    page: int | str
    title: str
    kind: LabelKind
    anchor: str = ""


@runtime_checkable
class RecordStore(Protocol):
    """The bundle's pinned records, read by ref, each verified against `MANIFEST.files`.

    `read` resolves the contract's ref grammar (`file#path`); a ref that does not resolve
    raises `FactMissing`, and a pinned file whose bytes differ from its recorded digest raises
    `BundleAltered`. `manifest()` is a copy of `MANIFEST.json`; `manifest_digest()` is the
    sha256 of its bytes, the digest `SHOWCASE.json` pins."""

    def read(self, ref: FactRef) -> object: ...

    def manifest(self) -> dict: ...

    def manifest_digest(self) -> str: ...


@runtime_checkable
class LabelSource(Protocol):
    """The report's labels: `{label name: LabelRef}` and the digest of the file they came from."""

    def labels(self) -> Mapping[str, LabelRef]: ...

    def digest(self) -> str: ...
