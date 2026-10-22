"""
records — `BundleRecords`, the `RecordStore` adapter over a packed bundle's pinned files.

The stage reads facts only from what the bundle pins: `MANIFEST.json` itself, and the fact
sources of the catalog (`netspec.json`, `schema.json`, `records/<layer>.json`) that
`MANIFEST.files` lists. `open()` reads each of those files once, digests the bytes it read and
parses the same bytes, so the digest that is checked is the digest of what is used; a file
whose bytes differ from `MANIFEST.files`, or a pinned file that is gone, refuses the whole
bundle with `BundleAltered` (gate PF2). A file the manifest does not pin is never read: a ref
into it is `FactMissing`, as is a ref whose key, index or selector finds nothing.

The ref grammar is the contract's §2 (v2.2 pack's `_story_lookup`, made file-explicit):

    records/shape.json#manifest.steps[id=S4].rows_after.all
    netspec.json#task.classes.1
    MANIFEST.json#serve.identity.noun

`name[key=value]` picks the first list element whose `key` equals `value` (compared as text);
a segment applied to a list is an index. Nothing is written, and nothing but the fact
sources is opened.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from show_ports import RecordStore
from vega_show_core import CATALOG, FactMissing, FactRef, ShowError, digest_bytes

__all__ = ["BundleRecords", "BundleAltered"]

MANIFEST = "MANIFEST.json"
# the files a fact may be read from, besides MANIFEST.json itself (the catalog's fact_sources)
READABLE = tuple(s for s in CATALOG["fact_sources"] if s != MANIFEST and not s.endswith(":"))


class BundleAltered(ShowError):
    """A file the bundle pins differs from its `MANIFEST.files` digest, or is gone (gate PF2).
    `mismatches` holds `(file, recorded, on_disk)` triples; `on_disk` is None for a lost file."""

    def __init__(self, bundle: Path, mismatches) -> None:
        self.bundle, self.mismatches = Path(bundle), tuple(mismatches)
        lines = [f"{f}: {rec} recorded, {disk or 'missing'} on disk" for f, rec, disk in self.mismatches]
        super().__init__(f"the bundle {self.bundle} was altered after pack — " + "; ".join(lines))


class BundleRecords(RecordStore):
    """The pinned records of one bundle, verified against `MANIFEST.files` before use: the
    `RecordStore` port's adapter, declared as one, not only shaped like one."""

    def __init__(self, bundle: Path, manifest_files: Mapping[str, str]) -> None:
        self._bundle = Path(bundle)
        self._files = MappingProxyType(dict(manifest_files))
        self._docs: dict[str, object] = {}
        self._manifest: dict | None = None
        self._manifest_digest: str | None = None

    @classmethod
    def open(cls, bundle: Path) -> "BundleRecords":
        """Read `MANIFEST.json`, then verify and load every pinned fact source."""
        bundle = Path(bundle)
        raw = _read_bytes(bundle / MANIFEST, "the bundle's MANIFEST.json")
        manifest = _parse(raw, bundle / MANIFEST)
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict) or not all(isinstance(v, str) for v in files.values()):
            raise ShowError(f"{bundle / MANIFEST} has no MANIFEST.files mapping of digests")
        store = cls(bundle, files)
        store._manifest, store._manifest_digest = manifest, digest_bytes(raw)
        store.verify()
        return store

    @property
    def bundle(self) -> Path:
        return self._bundle

    def pinned(self) -> tuple[str, ...]:
        """The fact sources this bundle pins, in the catalog's order."""
        return tuple(f for f in READABLE if f in self._files)

    def verify(self) -> tuple[str, ...]:
        """Load every pinned fact source; `BundleAltered` names every file that differs."""
        mismatches = []
        for rel in self.pinned():
            if rel in self._docs:
                continue
            path = self._bundle / rel
            if not path.is_file():
                mismatches.append((rel, self._files[rel], None))
                continue
            raw = path.read_bytes()
            if digest_bytes(raw) != self._files[rel]:
                mismatches.append((rel, self._files[rel], digest_bytes(raw)))
                continue
            self._docs[rel] = _parse(raw, path)
        if mismatches:
            raise BundleAltered(self._bundle, mismatches)
        return self.pinned()

    def read(self, ref: FactRef) -> object:
        """The value a ref points at (a copy: the store's documents are never handed out)."""
        ref = FactRef.parse(ref) if isinstance(ref, str) else ref
        if ref.is_report:
            raise FactMissing(ref, "a report: ref is read through the report's labels, not the bundle")
        return copy.deepcopy(_resolve(self._document(ref), ref))

    def manifest(self) -> dict:
        if self._manifest is None:
            raw = _read_bytes(self._bundle / MANIFEST, "the bundle's MANIFEST.json")
            self._manifest = _parse(raw, self._bundle / MANIFEST)
            self._manifest_digest = digest_bytes(raw)
        return copy.deepcopy(self._manifest)

    def manifest_digest(self) -> str:
        if self._manifest_digest is None:
            self.manifest()
        return self._manifest_digest

    def _document(self, ref: FactRef) -> object:
        if ref.file == MANIFEST:
            self.manifest()
            return self._manifest
        if ref.file not in READABLE:
            raise FactMissing(ref, f"{ref.file} is not a fact source ({', '.join(READABLE)})")
        if ref.file not in self._files:
            raise FactMissing(ref, f"{ref.file} is not pinned by MANIFEST.files")
        if ref.file not in self._docs:
            self.verify()
        return self._docs[ref.file]


def _read_bytes(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ShowError(f"{what} could not be read ({path}): {exc}") from exc


def _parse(raw: bytes, path: Path) -> object:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ShowError(f"{path} is not JSON: {exc}") from exc


def _resolve(doc: object, ref: FactRef) -> object:
    node = doc
    for name, key, value in ref.segments():
        node = _step(node, name, ref)
        if key is not None:
            node = _select(node, key, value, ref)
    return node


def _step(node: object, name: str, ref: FactRef) -> object:
    if isinstance(node, list):
        if not (name.isascii() and name.isdigit()):
            raise FactMissing(ref, f"segment {name!r} applied to a list (an index is an integer)")
        if int(name) >= len(node):
            raise FactMissing(ref, f"index {name} is out of range (the list holds {len(node)})")
        return node[int(name)]
    if isinstance(node, dict):
        if name not in node:
            raise FactMissing(ref, f"no key {name!r}")
        return node[name]
    raise FactMissing(ref, f"segment {name!r} applied to a {type(node).__name__} value")


def _select(node: object, key: str, value: str, ref: FactRef) -> object:
    if not isinstance(node, list):
        raise FactMissing(ref, f"selector [{key}={value}] applied to a {type(node).__name__} value")
    for item in node:
        if isinstance(item, dict) and key in item and _text(item[key]) == value:
            return item
    raise FactMissing(ref, f"no element with {key} = {value!r}")


def _text(value: object) -> str:
    """A scalar as a selector compares it: JSON's spelling for true/false/null, else str()."""
    return json.dumps(value) if value is None or isinstance(value, bool) else str(value)
