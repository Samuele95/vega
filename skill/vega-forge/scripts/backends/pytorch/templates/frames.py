"""
{{PACKAGE}}.{{MODULE_INGESTION}}.frames — the shaped files, read and checked, and one frame's
rows as the tensors of the batch contract (support; used by the {{PORT_DATASOURCE}} adapter).

Emitted by /vega-forge from `backends/pytorch/templates/frames.py`. Fix defects in the template
and re-emit; never in this copy (code-contract §4).

`PartFrames` is the only code of the package that opens a shaped file. Every opening checks the
shaped manifest against the FeatureContract (the manifest file's digest is the contract's
identity), checks the file against the digest the manifest recorded for it, and pairs each row
with its **raw-file position** — `fitted.json.__split_raw__[part]`, which the shaper writes in
the order of the part's rows. The raw position is a row's identity everywhere downstream: the
row id of a stored prediction, the node an edge names, the anchor of a window (build plan R-C2).

The seal is a property of this reader too: `open(spec, "test")` raises `SealedPartError` unless
the reader came from `unseal(spec, token, seal_path)` — the DataSource's one door after the
token matched `test_seal.json`, read once (`reader.seal`: what the evaluation stamps). The one
other way into the test file is `open_features_only`, the transductive door (feature columns,
no label; a graph spec whose `sample.exposure` a `user:` source declared, D1 §9.4, §9.5).

`row_tensors` turns one frame into `(x_num, x_cat, y)`: the feature columns cut by the
contract's one rule (`FeatureContract.encode`, which the Predictor's assembler applies to served
rows too) and the label encoded by the task (`Task.encode_labels`, D1 §9.2). This file owns
reading the columns out of a frame, nothing about what they mean.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, NamedTuple

import numpy as np
import pandas as pd
import torch

from ..domain import (FeatureContract, NetSpec, Part, SampleSpec, SealedPartError, SealReading,
                      Task, stream_digest)
# >>> shape: graph
from ..domain import GraphProtocol
# <<< shape

_PARTS = (Part.TRAIN, Part.VALID, Part.TEST)
_FITTED = "fitted.json"


# --------------------------------------------------------------------------- files

def _file_digest(path) -> str:
    """The shaped file's digest, as the shaper recorded it (`domain.stream_digest`)."""
    with Path(path).open("rb") as fh:
        return stream_digest(fh)


def _read_table(path, columns=None) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        if columns:
            import pyarrow.parquet as pq
            missing = sorted(set(columns) - set(pq.read_schema(p).names))
            if missing:
                raise ValueError(f"{p.name} has no column(s) {missing}")
        from pyarrow import fs      # Arrow's own reader: no Python in its threads (A40)
        return pd.read_parquet(p, columns=list(columns) if columns else None,
                               filesystem=fs.LocalFileSystem())
    if suffix in (".csv", ".txt"):
        return pd.read_csv(p, usecols=list(columns) if columns else None)
    raise ValueError(f"unsupported part format {p.suffix!r} for {p}; the shaper writes parquet "
                     "or csv")


def _chunks(path, columns, rows: int) -> Iterator[pd.DataFrame]:
    """The file in chunks of `rows` rows, never held whole."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(p).iter_batches(batch_size=rows, columns=list(columns)):
            yield batch.to_pandas()
    elif p.suffix.lower() in (".csv", ".txt"):
        yield from pd.read_csv(p, chunksize=rows, usecols=list(columns))
    else:
        raise ValueError(f"unsupported part format {p.suffix!r} for {p}")


def _located(recorded, base: Path, what: str) -> Path:
    """The path as recorded, else the file of that name beside the manifest (the shaper writes
    both into one directory; a relative path recorded from elsewhere still finds it). The digest
    check that follows every opening is what proves it is the file meant."""
    if not recorded:
        raise ValueError(f"the spec and the shaped manifest name no path for the {what}")
    path = Path(recorded)
    if path.is_file():
        return path
    beside = base / path.name
    if beside.is_file():
        return beside
    raise ValueError(f"shaped {what} not found: {path} (nor {beside})")


def _stat_key(path: Path) -> tuple:
    st = path.stat()
    return str(path.resolve()), st.st_mtime_ns, st.st_size


@dataclass(frozen=True)
class _Manifest:
    """What the reader needs of the shaped manifest, kept small (the manifest itself can hold
    the split's index sets, hundreds of megabytes once parsed): its file digest — the
    FeatureContract's identity — each output's path and digest, and where `fitted.json` is."""

    path: Path
    digest: str
    files: Any
    fitted: Path

    @staticmethod
    def read(path: Path) -> "_Manifest":
        data = path.read_bytes()
        outputs = json.loads(data).get("outputs") or {}
        files = {name: (entry.get("path"), entry.get("digest"))
                 for name, entry in (outputs.get("files") or {}).items()
                 if isinstance(entry, Mapping)}
        return _Manifest(path=path, digest="sha256:" + hashlib.sha256(data).hexdigest(),
                         files=files, fitted=path.parent / (outputs.get("fitted") or _FITTED))

    def entry(self, name: str) -> tuple:
        if name not in self.files:
            raise ValueError(f"the shaped manifest records no outputs.files.{name}")
        return self.files[name]


class ShapedRows(NamedTuple):
    """One opened part: the frame, each row's raw-file position, the file's checked digest."""

    frame: Any
    raw_pos: Any
    digest: str


# --------------------------------------------------------------------------- the files

class _ShapedFiles:
    """The digest-checked access to the shaped files that both readers hold (private: it knows
    no part from another, so neither reader is a subtype of it or of the other). It memoises
    only what it parsed out of the manifest and `fitted.json`, keyed by each file's path, size
    and mtime."""

    def __init__(self, memo: dict | None = None):
        self._memo = {} if memo is None else memo

    def rows(self, spec, part: Part, columns) -> ShapedRows:
        spec = NetSpec.require(spec)
        path, want = self._address(spec, part)
        digest = self._checked(spec, path, want, part.value)
        frame = _read_table(path, columns)
        raw = self.raw_positions(spec, part)
        if len(raw) != len(frame):
            raise ValueError(f"fitted.json records {len(raw)} raw position(s) for the "
                             f"{part.value} part, which holds {len(frame)} row(s); these are not "
                             "the same part")
        return ShapedRows(frame, raw, digest)

    def chunks(self, spec, part: Part, columns, rows: int) -> Iterator[pd.DataFrame]:
        path, want = self._address(NetSpec.require(spec), part)
        self._checked(spec, path, want, part.value)
        return _chunks(path, columns, int(rows))

    def edges(self, spec: NetSpec) -> pd.DataFrame:
        spec = NetSpec.require(spec)
        man = self.manifest(spec)
        recorded, want = man.entry("edges")
        path = _located(spec.inputs.edges or recorded, man.path.parent, "edges part")
        self._checked(spec, path, want, "edges")
        return _read_table(path)

    def raw_positions(self, spec: NetSpec, part) -> np.ndarray:
        part = Part(part)
        found = self._split_raw(spec).get(part.value)
        if found is None:
            raise ValueError(f"fitted.json records no __split_raw__.{part.value}, so the "
                             f"{part.value} rows have no raw position; a row's id is its raw-file "
                             "position (R-C2) — re-run the shaper")
        return found

    def manifest(self, spec: NetSpec) -> _Manifest:
        spec = NetSpec.require(spec)
        recorded = spec.inputs.manifest
        if not recorded:
            raise ValueError("the spec names no shaped manifest (input.manifest), so no part can "
                             "be checked; a part is identified by the digest it records")
        path = Path(recorded)
        if not path.is_file():
            raise ValueError(f"the shaped manifest {path} (input.manifest) does not exist")
        return self._memoised(("manifest",) + _stat_key(path), lambda: _Manifest.read(path))

    def _address(self, spec: NetSpec, part: Part) -> tuple:
        man = self.manifest(spec)
        recorded, want = man.entry(part.value)
        if part is not Part.TEST:
            recorded = spec.inputs.part(part)
            if not recorded:
                raise ValueError(f"the spec names no {part.value} part in data_contract.parts")
        return _located(recorded, man.path.parent, f"{part.value} part"), want

    def _checked(self, spec, path: Path, want, what: str) -> str:
        """The contract admits the manifest, and the file is the one it recorded (its digest)."""
        spec = NetSpec.require(spec)
        spec.feature_contract.admits(self.manifest(spec).digest)
        if not want:
            raise ValueError(f"the shaped manifest records no digest for the {what} part; a part "
                             "is identified by its digest")
        got = self._memoised(("digest",) + _stat_key(path), lambda: _file_digest(path))
        if got != want:
            raise ValueError(f"the {what} part at {path} digests to {got}, but the shaped "
                             f"manifest recorded {want}; the part changed since the shaper "
                             "wrote it")
        return got

    def _split_raw(self, spec) -> dict:
        fitted = self.manifest(spec).fitted
        if not fitted.is_file():
            raise ValueError(f"no {fitted.name} beside the shaped manifest ({fitted}); the raw "
                             "positions of the parts' rows live there")
        return self._memoised(("fitted",) + _stat_key(fitted), lambda: _read_split_raw(fitted))

    def _memoised(self, key, make):
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]


# --------------------------------------------------------------------------- the readers

class PartFrames:
    """The sealed reader of the shaped files: train and valid, the graph's edges, the raw
    positions, never the test part — there is no code path to it here, no hook to override and
    no seal to carry (the seal is a type: `UnsealedFrames`, which only `unseal` makes)."""

    def __init__(self, memo: dict | None = None):
        self._files = _ShapedFiles(memo)

    def open(self, spec: NetSpec, part, columns=None) -> ShapedRows:
        """The part as a frame (every column, or `columns`), digest-checked, with its rows' raw
        positions. The test part is refused, whatever the spec."""
        return self._files.rows(spec, _unsealed_part(part, "open"), columns)

    def open_features_only(self, spec: NetSpec, part) -> ShapedRows:
        """The part's feature columns and nothing else — the label is never read. For the test
        part this is the transductive door (D1 §9.4): only a graph spec whose protocol is
        transductive, with `sample.exposure` declared by a `user:` source, may pass."""
        part = Part(part)
        if part is Part.TEST:
            _transductive_door(spec)
        return self._files.rows(spec, part, NetSpec.require(spec).feature_contract.columns)

    def chunks(self, spec: NetSpec, part, columns, rows: int) -> Iterator[pd.DataFrame]:
        """The part streamed `rows` at a time, checked once when the stream is opened (the
        streamed training pass never holds the part). Train and valid only."""
        return self._files.chunks(spec, _unsealed_part(part, "chunks"), columns, rows)

    def edges(self, spec: NetSpec) -> pd.DataFrame:
        """The graph's edges part (`sample.edges`), checked against `outputs.files.edges`."""
        return self._files.edges(spec)

    def raw_positions(self, spec: NetSpec, part) -> np.ndarray:
        """`fitted.json.__split_raw__[part]`: the raw-file position of each row of the part."""
        return self._files.raw_positions(spec, part)

    def part_sizes(self, spec: NetSpec) -> tuple:
        """(train, valid, test) row counts, as the split recorded them — the test part's count
        is public; its rows are not."""
        return tuple(len(self.raw_positions(spec, p)) for p in _PARTS)

    def manifest(self, spec: NetSpec) -> _Manifest:
        return self._files.manifest(spec)


class UnsealedFrames:
    """The reader that may also open the test part — of the one manifest the seal admitted it
    to — made only by `unseal`, whole, from the seal it checked (`seal`: what the evaluation
    stamps). Not a `PartFrames`: it holds one for every other part, and adds the one door."""

    def __init__(self, memo: dict, manifest: Path, seal: SealReading):
        self._files, self._sealed = _ShapedFiles(memo), PartFrames(memo)    # one memo, two doors
        self._manifest, self._seal = str(Path(manifest).resolve()), seal

    @property
    def seal(self) -> SealReading:
        return self._seal

    def open(self, spec: NetSpec, part, columns=None) -> ShapedRows:
        """As `PartFrames.open`, and the test part of the manifest the seal was checked for."""
        if Part(part) is not Part.TEST:
            return self._sealed.open(spec, part, columns)
        if self._manifest != str(self._files.manifest(spec).path.resolve()):
            raise SealedPartError("this reader was unsealed for another shaped manifest; that "
                                  "test part opens only through its own load_sealed_part")
        return self._files.rows(spec, Part.TEST, columns)

    def open_features_only(self, spec: NetSpec, part) -> ShapedRows:
        return self._sealed.open_features_only(spec, part)

    def chunks(self, spec: NetSpec, part, columns, rows: int) -> Iterator[pd.DataFrame]:
        return self._sealed.chunks(spec, part, columns, rows)

    def edges(self, spec: NetSpec) -> pd.DataFrame:
        return self._sealed.edges(spec)

    def raw_positions(self, spec: NetSpec, part) -> np.ndarray:
        return self._sealed.raw_positions(spec, part)

    def part_sizes(self, spec: NetSpec) -> tuple:
        return self._sealed.part_sizes(spec)

    def manifest(self, spec: NetSpec) -> _Manifest:
        return self._sealed.manifest(spec)


def unseal(spec: NetSpec, token: str, seal_path="test_seal.json") -> UnsealedFrames:
    """The sealed door: a reader that may open this spec's test part, once the token matches
    the seal file /vega-eval wrote. Nothing else makes one — a module function, not a method of
    a reader, so holding (or injecting) a reader grants nothing."""
    seal = Path(seal_path)
    if not seal.is_file():
        raise SealedPartError(f"no seal file at {seal}: the test part is opened only under a "
                              "seal /vega-eval writes, and no seal exists yet")
    record = json.loads(seal.read_text(encoding="utf-8"))
    if not token or token != record.get("token"):
        raise SealedPartError("the token does not match the campaign's test_seal.json; the "
                              "test part stays closed")
    memo: dict = {}
    return UnsealedFrames(memo, _ShapedFiles(memo).manifest(spec).path, SealReading.of(record))


def _unsealed_part(part, door: str) -> Part:
    """`part`, unless it is the test part, which no door of the sealed reader opens."""
    part = Part(part)
    if part is Part.TEST:
        how = "is never streamed" if door == "chunks" else f"is refused by {door}"
        raise SealedPartError(f"the test part {how}; /vega-eval opens it once, through "
                              "load_sealed_part(spec, token), under the campaign's test_seal.json")
    return part


def _transductive_door(spec) -> None:
    graph = SampleSpec.from_spec(NetSpec.require(spec)).graph   # refuses an undeclared exposure
    # >>> shape: graph
    if graph is not None and graph.protocol is GraphProtocol.TRANSDUCTIVE:
        return
    # <<< shape
    raise SealedPartError("the test part's features are read before the seal only by a "
                          "transductive graph spec with a declared exposure; this spec "
                          "is not one")


def _read_split_raw(fitted: Path) -> dict:
    blob = json.loads(fitted.read_text(encoding="utf-8")).get("__split_raw__") or {}
    return {str(k): np.asarray(v, dtype=np.int64) for k, v in blob.items()}


# --------------------------------------------------------------------------- tensors

def feature_tensors(frame, contract: FeatureContract, what: str = "shaped") -> tuple:
    """`(x_num (N, F) float32, x_cat (N, C) int64 | None)` of the frame's rows: the contract's
    columns read out of the frame and cut by the contract's one rule (`encode`). A column the
    contract names and the frame lacks is refused by name."""
    missing = [c for c in contract.columns if c not in frame.columns]
    if missing:
        raise ValueError(f"the {what} part is missing the feature column(s) "
                         f"{', '.join(missing)} the spec's FeatureContract names")
    numeric = frame[list(contract.numeric_columns)].to_numpy(dtype=np.float32)
    nominal = (frame[list(contract.nominal_columns)].astype("float64").to_numpy()
               if contract.nominal_columns else None)
    x_num, x_cat = contract.encode(numeric, nominal)
    return (torch.tensor(x_num, dtype=torch.float32),
            None if x_cat is None else torch.tensor(x_cat, dtype=torch.long))


def label_tensor(frame, task: Task):
    """The label as the loss reads it (`Task.encode_labels`), or None: a kind trained on no
    label, no label named, or a frame without the label column (a features-only read)."""
    if not task.label or task.label not in frame.columns:
        return None
    y = task.encode_labels(frame[task.label].to_numpy())
    return None if y is None else torch.as_tensor(y)


def row_tensors(frame, contract: FeatureContract, task: Task, what: str = "shaped") -> tuple:
    """`(x_num, x_cat, y)` of the frame's rows (D1 §3.3, §9.2)."""
    x_num, x_cat = feature_tensors(frame, contract, what)
    return x_num, x_cat, label_tensor(frame, task)
