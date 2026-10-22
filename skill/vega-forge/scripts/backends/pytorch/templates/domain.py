"""
{{PACKAGE}}.domain — the blueprint's artifacts as immutable value objects, and the index plans
every adapter shares.

Emitted by /vega-forge from `backends/pytorch/templates/domain.py`. A design defect here is
fixed in that template and the package re-emitted; never in this copy (code-contract §4).

Two kinds of thing live here, and nothing else:

  * **values** — the spec, the task, the sample, a checkpoint, a metric record, a prediction,
    the parts the DataSource hands out, the parameter objects that keep every port signature
    at five parameters or fewer (C12);
# >>> shape: row
  * **plans** — none: a row sample is the row itself, so the DataSource
    (`{{MODULE_INGESTION}}`) and the Predictor (`{{MODULE_SERVING}}`) have no index logic to
    agree on beyond the feature contract.
# <<< shape
# >>> shape: window
  * **plans** — the pure index logic that `{{MODULE_INGESTION}}` (training, evaluation) and
    serving must agree on: which rows form a window and which rows it predicts
    (`WindowSpec.plan`). `{{MODULE_INGESTION}}` may depend only on `domain` in the module map,
    and the Predictor lives in another module, so the one copy of the logic that can be wrong
    is kept here, numpy only; the tensorising halves live in the adapters (`windows.py`,
    `assemble.py`).
# <<< shape
# >>> shape: graph
  * **plans** — the pure index logic that `{{MODULE_INGESTION}}` (training, evaluation) and
    serving must agree on: which nodes and edges a graph view holds (`GraphSpec.view`), how a
    served query attaches to the reference graph (`GraphReference.attach`).
    `{{MODULE_INGESTION}}` may depend only on `domain` in the module map, and the Predictor
    lives in another module, so the one copy of the logic that can be wrong is kept here,
    numpy only; the tensorising halves live in the adapters (`graphs.py`, `assemble.py`).
# <<< shape

`domain` imports no module of its package (it is the module every other depends on and that
depends on nothing: C4). Tensors enter only as duck-typed values: a part's methods call tensor
methods (`index_select`, `new_tensor`, `to`), never `torch` itself.

Effective Java rules this file carries (the anchors are the blueprint's; the corpus is the
`effective-java` skill, and no item's text is restated here):
  EJ-17 minimize mutability — every artifact is a frozen dataclass or an enum (C6); the one
        view, `NetworkAnatomy`, holds nothing but the network it views.
  EJ-1  static factories — `NetSpec.from_dict`, `Task.from_spec`, `Metrics.from_predictions`:
        the name says what the factory validates.
  EJ-2  builders / parameter objects — `RunContext`, `BatchPlan`, `MetricOptions`,
        `BaselineContext`, `Versions` keep every signature within five parameters (EJ-51).
  EJ-10/EJ-11 equals and hashCode — value objects compare by value; where the artifact's
        identity is a digest, the digest is what is compared.
  EJ-34 enums instead of constants — a string task kind, batch key or stage kind never
        crosses a port.
  EJ-49 check parameters for validity — construction validates and names the offender.
"""
from __future__ import annotations

import hashlib
import json
# >>> shape: window graph
import math
# <<< shape
# >>> shape: row
# >>> task: multilabel
import math
# <<< task
# <<< shape
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence, Union

import numpy as np


# --------------------------------------------------------------------------- enums

class TaskKind(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    REGRESSION = "regression"
    ORDINAL = "ordinal"
    UNSUPERVISED = "unsupervised"
    ANOMALY = "anomaly"
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    SEQUENCE_REGRESSION = "sequence-regression"
    FORECAST = "forecast"


class Part(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Family(str, Enum):
    """The catalog's 21 named compositions (`catalog.json.families`). A family the catalog
    does not name is legal when every block is known (netspec-contract §2), and arrives as
    `Family.USER_DEFINED`."""

    MLP_PLAIN = "mlp-plain"
    MLP_RESIDUAL = "mlp-residual"
    FT_TRANSFORMER = "ft-transformer"
    CNN1D = "cnn1d"
    GRU = "gru"
    LSTM = "lstm"
    TRANSFORMER_SEQ = "transformer-seq"
    AUTOENCODER = "autoencoder"
    VAE = "vae"
    DEEP_SVDD = "deep-svdd"
    LINEAR = "linear"
    PCA_FLOOR = "pca-floor"
    RNN = "rnn"
    BI_LSTM = "bi-lstm"
    BI_GRU = "bi-gru"
    LSTM_ATTENTION = "lstm-attention"
    GRU_ATTENTION = "gru-attention"
    GCN = "gcn"
    GRAPHSAGE = "graphsage"
    GAT = "gat"
    GIN = "gin"
    USER_DEFINED = "user-defined"

    @classmethod
    def of(cls, name: str) -> "Family":
        try:
            return cls(name)
        except ValueError:
            return cls.USER_DEFINED


class Objective(str, Enum):
    """What the head is trained to do, read from `loss.name`: predict the label, reconstruct
    the input (plain or variational), or pull the embedding to a centre (one-class)."""

    PREDICT = "predict"
    RECONSTRUCTION = "reconstruction"
    VAE = "vae"
    SVDD = "svdd"

    @classmethod
    def of_loss(cls, name) -> "Objective":
        by_loss = {"reconstruction": cls.RECONSTRUCTION, "vae": cls.VAE, "svdd": cls.SVDD}
        return by_loss.get(str(name or ""), cls.PREDICT)


class LabelFormatKind(str, Enum):
    INDEX = "index"
    NAMES = "names"
    DELIMITED = "delimited"


class SampleShape(str, Enum):
    ROW = "row"
    WINDOW = "window"
    GRAPH = "graph"


#: The sample shapes this package carries: its spec's own. /vega-forge emits the code of one
#: shape (the shape sections of the templates it keeps — this file's values and plans of a
#: shape included — and that shape's support files), so a spec of any other shape is refused
#: by name where a dict becomes a spec (`NetSpec.from_dict`, `SampleSpec.from_spec`:
#: `UnsupportedShapeError`), never met as a missing module or value inside an adapter.
CARRIED_SHAPES = frozenset({
    # >>> shape: row
    SampleShape.ROW,
    # <<< shape
    # >>> shape: window
    SampleShape.WINDOW,
    # <<< shape
    # >>> shape: graph
    SampleShape.GRAPH,
    # <<< shape
})


# >>> shape: window
class WindowContext(str, Enum):
    PREVIOUS_PART = "previous-part"
    NONE = "none"
# <<< shape


# >>> shape: graph
class GraphProtocol(str, Enum):
    """Whether a graph's test nodes were seen in training (a graph package's: the DataSource
    port's `exposure` record names it)."""

    INDUCTIVE = "inductive"
    TRANSDUCTIVE = "transductive"
# <<< shape


# >>> shape: graph
class GraphLevel(str, Enum):
    NODE = "node"
    GRAPH = "graph"


class GraphBatching(str, Enum):
    FULL = "full"
    SAMPLED = "sampled"
    GRAPHS = "graphs"
# <<< shape


class BatchKey(str, Enum):
    """The batch contract every DataSource yields and every Model reads (backend-contract §1).
    A `str` enum hashes as its value, so `batch["x_num"]` and `batch[BatchKey.X_NUM]` are the
    same lookup; the parts below build their dicts with the plain values."""

    X_NUM = "x_num"
    X_CAT = "x_cat"
    Y = "y"
    EDGE_INDEX = "edge_index"
    EDGE_WEIGHT = "edge_weight"
    TARGET_INDEX = "target_index"
    GRAPH_INDEX = "graph_index"
    NUM_GRAPHS = "num_graphs"


class StageKind(str, Enum):
    """How a stage of the Network reads the running state (one entry per spec block)."""

    EMB = "emb"              # m(x_cat) -> emb; the running tensor h is untouched
    CONCAT = "concat"        # h = m(x_num, emb)
    TOKENS = "tokens"        # h = m(x_num, x_cat)                 (feature_tokenizer)
    TENSOR = "tensor"        # h = m(h)
    GRAPH = "graph"          # h = m(h, edge_index, edge_weight)   (gcn/sage/gat/gin_conv)
    GRAPH_DROP = "gdrop"     # edge_index, edge_weight = m(edge_index, edge_weight)
    GRAPH_POOL = "gpool"     # h = m(h, graph_index, num_graphs); node rows -> graph rows


class DesignView(str, Enum):
    """The flat design matrix a baseline reads from a part (`ShapedParts.design_matrix`)."""

    FLAT = "flat"                        # rows; windows flattened to (W, L*F' + L*C)
    NEIGHBOUR_MEAN = "neighbour-mean"    # graph node: [x | mean of the neighbours' x]
    GRAPH_POOLED = "graph-pooled"        # graph level: [mean | max | sum] over each graph


class RowIdKind(str, Enum):
    RAW_POSITION = "raw_position"
    GRAPH_ID = "graph_id"


# >>> shape: window
class UnscoredReason(str, Enum):
    INSUFFICIENT_CONTEXT = "insufficient_context"
    HORIZON_BEYOND_PART = "horizon_beyond_part"
# <<< shape


class DriftMeasure(str, Enum):
    """What a monitor reading measured: the population stability of a column or an answer,
    the recall of a class, the error of an answer against its target, the rate of a decision
    against its reference. Named apart from any blueprint word; `str` keeps the JSON."""

    PSI = "psi"
    RECALL = "recall"
    MAE = "mae"
    RATE = "rate"

    def __str__(self) -> str:
        return self.value


class DriftVerdict(str, Enum):
    """A monitor reading's outcome against its limits — `DriftVerdict`, not `Verdict`, which
    is /vega-eval's word for whether the network earned its cost. `unmeasured` says the
    window could not be measured (too few rows, a column missing), never that it held."""

    OK = "ok"
    WATCH = "watch"
    ALERT = "alert"
    UNMEASURED = "unmeasured"

    def __str__(self) -> str:
        return self.value


class BatchForm(str, Enum):
    """The positional form of an exported batch: the sample shape, a graph's by its level."""

    ROW = "row"
    WINDOW = "window"
    NODE = "node"
    GRAPH = "graph"

    def __str__(self) -> str:
        return self.value


class ExportFormat(str, Enum):
    """The runtimes a trained network is exported to (the ArtifactExporter's `fmt`)."""

    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"

    def __str__(self) -> str:
        return self.value


class SelectionDirection(str, Enum):
    """Which way the monitored metric improves: the early-stopping and selection `mode`."""

    MAX = "max"
    MIN = "min"

    def __str__(self) -> str:
        return self.value


class StopReason(str, Enum):
    """What ended a fit. A closed four-variant set, so it is an enum and not a bare string
    (EJ-34); subclassing `str` keeps every JSON line and comparison working unchanged."""

    MAX_EPOCHS = "max_epochs"
    NAN = "nan"
    EARLY_STOPPING = "early_stopping"
    BUDGET = "budget"


_SEQUENCE_BASES = {"sequence-classification": ("binary", "multiclass", "multilabel"),
                   "sequence-regression": ("regression", "ordinal")}
_CLASS_KINDS = frozenset({"binary", "multiclass", "multilabel", "ordinal"})
_PART_ORDER = (Part.TRAIN, Part.VALID, Part.TEST)
_EPOCH_SCALARS = ("epoch", "time_s", "lr", "nan")
_EPOCH_MAPS = ("train", "valid", "grad_norm", "weight_norm", "update_ratio", "early_stopping",
               "shuffle")


# --------------------------------------------------------------------------- helpers

def _digest_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _listify(value):
    """A framework tensor or array becomes plain numbers before it enters a value object: a
    value object holds values, and a field that is a live buffer is neither hashable nor
    safe to hand out (EJ-17, EJ-50)."""
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


class _FrozenMap(tuple):
    """A frozen mapping that still knows it is one.

    Freezing a mapping to a bare tuple of pairs loses the distinction between `{}` and `[]`,
    and between a mapping and a list that happens to hold pairs — so thawing guesses, and a
    spec that goes through `to_dict()` and back does not compare equal to itself. Identity is
    the whole point of these value objects (EJ-10/EJ-11), so the type carries the tag rather
    than the shape carrying a hint — and it carries it in `__eq__` and `__hash__` too, or two
    values that serialise differently would still compare equal, which is the same defect one
    layer down (EJ-10: equals is an equivalence over what the value *is*).
    """

    __slots__ = ()

    def __eq__(self, other):
        if isinstance(other, _FrozenMap):
            return tuple(self) == tuple(other)
        if isinstance(other, tuple):
            return NotImplemented if type(other) is not tuple else False
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    def __hash__(self):
        return hash(("_FrozenMap", tuple(self)))


def _freeze(value):
    """Recursively make a JSON value hashable, so a value object built from a spec compares
    and hashes by value (EJ-10/EJ-11) without copying the spec into typed fields it would
    then have to keep in sync."""
    if isinstance(value, Mapping):
        return _FrozenMap(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value):
    """The exact inverse of `_freeze`: a frozen mapping thaws to a mapping and a frozen list
    to a list, whether or not either is empty and whatever the list holds."""
    if isinstance(value, _FrozenMap):
        return {k: _thaw(v) for k, v in value}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def _spec_dict(spec) -> dict:
    """The spec as a mapping, whether a `NetSpec` or the raw dict an entry point holds."""
    to_dict = getattr(spec, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(spec or {})


def _sample_dict(spec) -> dict:
    return dict((_spec_dict(spec).get("data_contract") or {}).get("sample") or {})


def _carried(shape: SampleShape) -> SampleShape:
    """`shape`, when this package carries it; else `UnsupportedShapeError` naming it."""
    if shape not in CARRIED_SHAPES:
        raise UnsupportedShapeError(shape)
    return shape


def _py(value):
    """A numpy scalar as the plain Python value it holds (a dict key, a message, JSON)."""
    item = getattr(value, "item", None)
    return item() if callable(item) and getattr(value, "ndim", 1) == 0 else value


# >>> shape: window graph
def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
# <<< shape


# >>> shape: graph
def _index_array(values, name: str):
    arr = np.asarray(values if values is not None else (), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr
# <<< shape


# >>> shape: window graph
def _missing(arr):
    """Which entries of a key column are missing (NaN, None, NaT)."""
    if arr.dtype.kind in "fmM":
        return np.isnan(arr)
    if arr.dtype.kind == "O":
        return np.fromiter((_is_missing(v) for v in arr), dtype=bool, count=len(arr))
    return np.zeros(len(arr), dtype=bool)


def _factorize(values, where, column: str):
    """Codes by first appearance (so one group keeps its stored order) and the group values.
    A missing key is refused and located: a row without its key has no group to be in."""
    arr = np.asarray(values)
    missing = _missing(arr)
    if missing.any():
        at = _py(where[int(np.argmax(missing))])
        raise ValueError(f"the {column} column is missing at row {at}; every row needs the key "
                         "that groups it")
    if arr.dtype.kind == "O":
        index: dict = {}
        codes = np.fromiter((index.setdefault(_py(v), len(index)) for v in arr),
                            dtype=np.int64, count=len(arr))
        return codes, [k for k in index]
    uniq, first, inverse = np.unique(arr, return_index=True, return_inverse=True)
    rank = np.argsort(first, kind="stable")
    remap = np.empty(len(rank), dtype=np.int64)
    remap[rank] = np.arange(len(rank))
    return remap[np.asarray(inverse).reshape(-1)], [_py(v) for v in uniq[rank]]
# <<< shape


def _take(values, idx):
    """Rows `idx` of a tensor (a device gather, safe inside a captured CUDA graph) or an
    array; None stays None."""
    if values is None:
        return None
    select = getattr(values, "index_select", None)
    return select(0, idx) if callable(select) else values[idx]


# >>> shape: window
def _rows(values, pos):
    """Rows at a 2-D position table: `(B, L)` positions give `(B, L, ...)` rows."""
    select = getattr(values, "index_select", None)
    if not callable(select):
        return values[pos]
    flat = select(0, pos.reshape(-1))
    return flat.reshape(tuple(pos.shape) + tuple(values.shape[1:]))
# <<< shape


# ------------------------------------------------------------------ the shared primitives
# Duck-typed (this file imports no framework), defined once for every module of the package:
# the adapters and their support files read a result, place a batch, find a module's device
# and digest a file through these, never through a copy of their own.

def host_array(values):
    """A tensor (on any device) or an array as a host ndarray; None stays None."""
    if values is None:
        return None
    detach = getattr(values, "detach", None)
    return detach().cpu().numpy() if callable(detach) else np.asarray(values)


def _move(values, device):
    to = getattr(values, "to", None)
    return to(device) if callable(to) and values is not None else values


def on_device(batch: Mapping, device) -> dict:
    """The batch with every value that moves (`.to`) on `device`, the rest as it is."""
    return {k: _move(v, device) for k, v in batch.items()}


def device_of(module):
    """Where a module's tensors live: its first parameter's device, else its first buffer's,
    else the CPU (`"cpu"`, which every `.to` reads as the CPU device)."""
    for tensors in (module.parameters(), module.buffers()):
        first = next(iter(tensors), None)
        if first is not None:
            return first.device
    return "cpu"


def stream_digest(stream) -> str:
    """`sha256:<hex>` of a binary stream read in 1 MiB chunks: the one hashing of a file the
    package's identities use (a shaped part, a checkpoint) — the caller opens the file."""
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        h.update(chunk)
    return "sha256:" + h.hexdigest()


# >>> shape: graph
def _arange_like(ref, n: int):
    """`0 … n-1` on the same device and in the same library as `ref`."""
    if isinstance(ref, np.ndarray):
        return np.arange(n, dtype=np.int64)
    return ref.new_ones(n).cumsum(0) - 1


def _full_like(ref, n: int, value: int):
    if isinstance(ref, np.ndarray):
        return np.full(n, value, dtype=np.int64)
    return ref.new_full((n,), value)


def _nonzero(mask):
    if isinstance(mask, np.ndarray):
        return np.flatnonzero(mask)
    return mask.nonzero().reshape(-1)
# <<< shape


# >>> shape: window
def _steps_like(anchors, length: int):
    """The window's relative positions `-(L-1) … 0`, built once beside the anchors so a
    gather creates no tensor (a captured CUDA graph cannot contain a host copy)."""
    offsets = list(range(1 - int(length), 1))
    new_tensor = getattr(anchors, "new_tensor", None)
    return new_tensor(offsets) if callable(new_tensor) else np.asarray(offsets, dtype=np.int64)
# <<< shape


# --------------------------------------------------------------------------- the spec

class FeatureRole(str, Enum):
    """What a feature column is to the network (`data_contract.features[].role`): a numeric
    input, a nominal id an embedding reads, or a key that groups and orders rows and is never
    an input. The role decides which tensor a column fills, nothing finer: a column the shaper
    marked otherwise (a flag, a scaled count) is a numeric input."""

    NUMERIC = "numeric"
    NOMINAL = "nominal"
    KEY = "key"

    @classmethod
    def of(cls, role) -> "FeatureRole":
        return cls(role) if role in (cls.NOMINAL.value, cls.KEY.value) else cls.NUMERIC


@dataclass(frozen=True)
class Feature:
    """One column of the feature contract: its name and role and, for a nominal, the
    embedding's row count and the reserved row an unseen or missing id reads as
    (`unseen_category_id`; the last row when the contract names none)."""

    name: str
    role: FeatureRole = FeatureRole.NUMERIC
    cardinality: int = 0
    unseen_id: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "role", FeatureRole(self.role))
        object.__setattr__(self, "cardinality", int(self.cardinality or 0))
        if not self.name:
            raise ValueError("a feature of the contract needs its column name")
        if self.role is not FeatureRole.NOMINAL:
            return
        if self.cardinality < 1:
            raise ValueError(f"nominal feature {self.name!r} has no cardinality: an embedding is "
                             "sized by data_contract.features[].cardinality")
        unseen = self.cardinality - 1 if self.unseen_id is None else int(self.unseen_id)
        if not 0 <= unseen < self.cardinality:
            raise ValueError(f"nominal feature {self.name!r}: unseen_category_id {unseen} is not "
                             f"a row of its {self.cardinality}-row embedding")
        object.__setattr__(self, "unseen_id", unseen)

    @staticmethod
    def of(entry: Mapping[str, Any]) -> "Feature":
        """A `data_contract.features[]` entry as a value (its other keys are provenance)."""
        return Feature(name=str(entry.get("name") or ""), role=FeatureRole.of(entry.get("role")),
                       cardinality=entry.get("cardinality") or 0,
                       unseen_id=entry.get("unseen_category_id"))

    def ids(self, values) -> np.ndarray:
        """A nominal's codes as the int64 rows of its embedding: a code outside
        `[0, cardinality)`, a missing one (NaN) included, reads as `unseen_id`."""
        codes = np.array(values, dtype=np.float64, copy=True)
        codes[np.isnan(codes)] = -1
        codes[(codes < 0) | (codes >= self.cardinality)] = self.unseen_id
        return codes.astype(np.int64)


@dataclass(frozen=True, eq=False)
class FeatureContract:
    """What the network is allowed to see. Identity: the shaped manifest's digest.

    The features are `Feature` values in contract order; the contract is the one place the
    features boundary is cut into tensors (`encode`), so the DataSource's training rows and
    the Predictor's served rows are tensorised by the same rule. `admits` is what makes the
    digest identity: the DataSource hands it the digest of the manifest on disk before it
    opens a part, so a contract built from one manifest never reads the parts of another.
    """

    input_dim: int
    label: str | None
    features: tuple
    manifest_digest: str

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(
            f if isinstance(f, Feature) else Feature.of(dict(f)) for f in self.features))
        if int(self.input_dim) <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim!r}")
        if not str(self.manifest_digest or "").startswith("sha256:"):
            raise ValueError("a FeatureContract is identified by the shaped manifest's digest "
                             f"('sha256:<hex>', the spec's input.manifest_digest); got "
                             f"{self.manifest_digest!r}")
        named = len(self.numeric_columns) + len(self.nominal_columns)
        if not named:
            raise ValueError("the FeatureContract lists no features (data_contract.features): "
                             "a network reads at least one numeric or nominal column")
        if named != int(self.input_dim):
            raise ValueError(f"the FeatureContract declares input_dim {self.input_dim} but names "
                             f"{named} feature column(s); the contract is the authority and it "
                             "disagrees with itself")

    def __eq__(self, other):
        if not isinstance(other, FeatureContract):
            return NotImplemented
        return self.manifest_digest == other.manifest_digest

    def __hash__(self):
        return hash(self.manifest_digest)

    @staticmethod
    def from_spec(spec_dict: Mapping[str, Any]) -> "FeatureContract":
        dc = spec_dict.get("data_contract") or {}
        return FeatureContract(
            input_dim=int(dc.get("input_dim") or 0),
            label=(spec_dict.get("task") or {}).get("label"),
            features=tuple(Feature.of(f) for f in dc.get("features") or ()),
            manifest_digest=(spec_dict.get("input") or {}).get("manifest_digest"))

    def admits(self, manifest_digest: str) -> None:
        """Refuse a manifest whose digest is not the one this contract was built from (EJ-49).

        The value object compares digests; it does not read files. The DataSource adapter
        hashes the manifest on disk and hands the digest here, which is the split the
        blueprint's hexagon draws: IO in the adapter, the identity rule in the value."""
        if manifest_digest != self.manifest_digest:
            raise ValueError(
                f"the shaped manifest digests to {manifest_digest}, but the FeatureContract was "
                f"built from {self.manifest_digest}; the parts this contract describes are "
                "not the parts on disk")

    def _named(self, role: FeatureRole) -> tuple:
        return tuple(f.name for f in self.features if f.role is role)

    @property
    def numeric_columns(self) -> tuple:
        """The numeric feature columns, in contract order — what fills `x_num`. A `key` column
        groups rows (a series, a graph) and is never a feature, so it is not one of them."""
        return self._named(FeatureRole.NUMERIC)

    @property
    def key_columns(self) -> tuple:
        return self._named(FeatureRole.KEY)

    @property
    def nominal_columns(self) -> tuple:
        return self._named(FeatureRole.NOMINAL)

    @property
    def nominal_features(self) -> tuple:
        return tuple(f for f in self.features if f.role is FeatureRole.NOMINAL)

    @property
    def cardinalities(self) -> tuple:
        """The nominal features' embedding sizes, in contract order (`x_cat`'s columns)."""
        return tuple(f.cardinality for f in self.nominal_features)

    @property
    def columns(self) -> tuple:
        """The feature columns in tensor order: the numeric block, then the nominal ids."""
        return self.numeric_columns + self.nominal_columns

    def encode(self, numeric, nominal=None) -> tuple:
        """The features boundary's one tensorising rule, for training and serving alike:
        `(numeric_block(numeric), nominal_ids(nominal))`."""
        return self.numeric_block(numeric), self.nominal_ids(nominal)

    def numeric_block(self, numeric) -> np.ndarray:
        """`numeric` (n, F), in `numeric_columns` order, as the float32 `x_num` block."""
        x_num = np.asarray(numeric, dtype=np.float32)
        if x_num.ndim != 2 or x_num.shape[1] != len(self.numeric_columns):
            raise ValueError(f"the numeric block is {x_num.shape}; the contract reads "
                             f"{len(self.numeric_columns)} numeric column(s) per row")
        return x_num

    def nominal_ids(self, nominal):
        """`nominal` (n, C) codes in `nominal_columns` order (float, NaN for a missing code) as
        the int64 `x_cat` rows of their embeddings — an unseen or missing code read as the
        feature's `unseen_id` — or None for a contract without nominals."""
        nominals = self.nominal_features
        if not nominals:
            return None
        codes = np.asarray(nominal, dtype=np.float64).reshape(-1, len(nominals))
        return np.stack([f.ids(codes[:, j]) for j, f in enumerate(nominals)], axis=1)


@dataclass(frozen=True)
class BatchSize:
    """`training.batch_size`: a number of samples per batch, or `full` — every sample of the
    part in one batch (`rows` None). Read once, here: a reader asks for the size of a held
    part's batches (`for_part`) or of a stream's (`streamed`, which has no `full`)."""

    rows: int | None = 32

    def __post_init__(self):
        if self.rows is not None and int(self.rows) < 1:
            raise ValueError("training.batch_size is a positive count or 'full', got "
                             f"{self.rows!r}")

    @staticmethod
    def of(value) -> "BatchSize":
        if value == "full":
            return BatchSize(None)
        return BatchSize(int(value) if value else 32)

    @property
    def full(self) -> bool:
        return self.rows is None

    def for_part(self, samples: int) -> int:
        """The batch size over a held part of `samples` samples."""
        return max(1, int(samples)) if self.full else int(self.rows)

    def streamed(self) -> int:
        """The batch size of a streamed part, which is never held whole: `full` is refused."""
        if self.full:
            raise ValueError("a streamed row part has no 'full' batch; training.batch_size is an "
                             "integer on a row sample (R1)")
        return int(self.rows)


@dataclass(frozen=True)
class TrainingPlan:
    """`training`, read once: the batch size, the epoch cap, the streamed shuffle's buffer, the
    device, the live gradient-clipping bound (the regularization declaration wins over it:
    `Regularization`) and the wall-clock budget."""

    batch_size: BatchSize = field(default_factory=BatchSize)
    batch_size_stated: bool = False
    max_epochs: int = 100
    shuffle_buffer: int | None = None
    device: str = "auto"
    grad_clip: float | None = None
    budget_min: float | None = None

    @staticmethod
    def of(node) -> "TrainingPlan":
        node = node if isinstance(node, Mapping) else {}
        budget = node.get("budget") if isinstance(node.get("budget"), Mapping) else {}
        return TrainingPlan(batch_size=BatchSize.of(node.get("batch_size")),
                            batch_size_stated=node.get("batch_size") is not None,
                            max_epochs=int(node.get("max_epochs") or 100),
                            shuffle_buffer=int(node["shuffle_buffer"]) if node.get(
                                "shuffle_buffer") else None,
                            device=str(node.get("device") or "auto"),
                            grad_clip=node.get("grad_clip"),
                            budget_min=budget.get("wall_clock_min"))


@dataclass(frozen=True)
class SpecComponent:
    """A named component of the spec with its parameters — the optimiser, the schedule:
    `{"name": ..., "params": {...}}`, read once, the parameters frozen."""

    name: str | None = None
    params: _FrozenMap = field(default_factory=_FrozenMap)

    @staticmethod
    def of(node, default_name: str | None = None) -> "SpecComponent":
        node = node if isinstance(node, Mapping) else {}
        return SpecComponent(node.get("name") or default_name, _freeze(dict(node.get("params")
                                                                            or {})))

    def params_dict(self) -> dict:
        """The parameters as a fresh mutable copy (a caller adds what the contract fixes)."""
        return dict(_thaw(self.params))


@dataclass(frozen=True)
class LossSpec:
    """`loss`, read once: its name, its parameters, the weighed alternatives a ladder merged
    beside it, and — a quantile loss's — the levels its head is laid out by."""

    name: str | None = None
    params: _FrozenMap = field(default_factory=_FrozenMap)
    alternatives: tuple = ()

    @staticmethod
    def of(node) -> "LossSpec":
        node = node if isinstance(node, Mapping) else {}
        alts = tuple(_freeze(dict(a)) for a in node.get("alternatives") or ()
                     if isinstance(a, Mapping))
        return LossSpec(node.get("name"), _freeze(dict(node.get("params") or {})), alts)

    def params_dict(self) -> dict:
        return dict(_thaw(self.params))

    def alternatives_list(self) -> list:
        return [dict(_thaw(a)) for a in self.alternatives]

    @property
    def quantiles(self) -> tuple:
        """The quantile levels of a `quantile` loss (default the median), else none."""
        if self.name != "quantile":
            return ()
        raw = self.params_dict().get("quantiles", (0.5,))
        return tuple(float(q) for q in (raw if isinstance(raw, (list, tuple)) else [raw]))


@dataclass(frozen=True)
class SelectionRule:
    """How a fit selects its epoch: `early_stopping`, falling back to `metrics.selection` — the
    monitored valid metric (`key`, the monitor without its `valid.` prefix), its direction,
    the patience and the improvement that counts."""

    monitor: str | None = None
    mode: SelectionDirection = SelectionDirection.MAX
    patience: int = 10
    min_delta: float = 1e-3

    @staticmethod
    def of(early_stopping, selection) -> "SelectionRule":
        es = early_stopping if isinstance(early_stopping, Mapping) else {}
        sel = selection if isinstance(selection, Mapping) else {}
        mode = es.get("mode") or sel.get("mode", "max")
        if mode not in tuple(d.value for d in SelectionDirection):
            raise ValueError(f"the early-stopping mode {mode!r} is not 'max' or 'min'; the "
                             "selection's direction is never guessed")
        return SelectionRule(monitor=es.get("monitor") or sel.get("monitor"),
                             mode=SelectionDirection(mode), patience=int(es.get("patience", 10)),
                             min_delta=float(es.get("min_delta", 1e-3)))

    @property
    def key(self) -> str:
        return str(self.monitor or "valid.loss").split(".", 1)[-1]


@dataclass(frozen=True)
class SpecInputs:
    """Where the spec's data is: the shaped manifest (its path and the digest the contract is
    identified by), the train and valid parts' paths, and a graph's edges part."""

    manifest: str | None = None
    manifest_digest: str | None = None
    parts: _FrozenMap = field(default_factory=_FrozenMap)
    edges: str | None = None

    def part(self, part: "Part") -> str | None:
        return dict(self.parts).get(Part(part).value)


@dataclass(frozen=True)
class NetSpec:
    """The network specification. Identity: the digest of its canonical JSON.

    `NetSpec.from_dict(d)` is the one place a mapping becomes a spec (EJ-1): every other
    signature in this package takes a NetSpec, never a raw mapping (EJ-64), and every section
    an adapter reads is read here once, as a value (`training`, `loss`, `optimizer`,
    `schedule`, `selection`, `inputs`, `hints`, `metric_names`), never by string key at the
    point of use (EJ-62). The domain does no IO: an entry point reads the file and converts.
    """

    task: TaskKind
    family: Family
    label: str | None
    seed: int
    _data: _FrozenMap = field(repr=False, default_factory=_FrozenMap)

    @property
    def digest(self) -> str:
        """Derived, never stored. The digest is what `ModelStore.load` refuses a checkpoint
        over, so it has to *be* the content: a settable field lets two specs with identical
        content compare unequal and lets a caller stamp any identity it likes on a value
        (EJ-10 — equality is over what the value is, EJ-49 — the invariant is enforced by
        construction, not by convention)."""
        return _digest_text(json.dumps(_thaw(self._data), sort_keys=True,
                                       separators=(",", ":")))

    @staticmethod
    def require(spec) -> "NetSpec":
        """The spec as it must cross a port: a `NetSpec`, never a raw mapping (EJ-64 — the
        ports are typed with the value object, and a dict is refused here by name rather than
        converted silently inside every adapter). Entry points convert once, with
        `NetSpec.from_dict`."""
        if isinstance(spec, NetSpec):
            return spec
        raise TypeError(f"a {type(spec).__name__} was passed where the ports take a NetSpec; "
                        "convert once at the entry point with NetSpec.from_dict(...)")

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "NetSpec":
        """Identity is the spec's *content*: the digest is the canonical JSON's, so a spec read
        from a file and the same spec passed as a mapping are the same value (EJ-10/EJ-11)."""
        for key in ("task", "data_contract", "family", "blocks", "loss", "metrics",
                    "optimizer", "training"):
            if key not in d:
                raise ValueError(f"netspec is missing the required key {key!r}")
        kind = (d.get("task") or {}).get("kind")
        try:
            task = TaskKind(kind)
        except ValueError as exc:
            raise ValueError(f"unknown task kind {kind!r}") from exc
        shape = _sample_dict(d).get("shape") or SampleShape.ROW.value
        if shape in {s.value for s in SampleShape}:
            _carried(SampleShape(shape))
        return NetSpec(task=task,
                       family=Family.of((d.get("family") or {}).get("name")),
                       label=(d.get("task") or {}).get("label"),
                       seed=int(d.get("seed", 7)),
                       _data=_freeze(d))

    def to_dict(self) -> dict:
        return _thaw(self._data)

    def _node(self, name: str):
        return dict(self._data).get(name)

    @property
    def feature_contract(self) -> FeatureContract:
        return FeatureContract.from_spec(self.to_dict())

    @property
    def training(self) -> TrainingPlan:
        return TrainingPlan.of(_thaw(self._node("training")))

    @property
    def loss(self) -> LossSpec:
        return LossSpec.of(_thaw(self._node("loss")))

    @property
    def optimizer(self) -> SpecComponent:
        return SpecComponent.of(_thaw(self._node("optimizer")))

    @property
    def schedule(self) -> SpecComponent:
        """`schedule`; a spec that names none has the schedule `none`."""
        return SpecComponent.of(_thaw(self._node("schedule")), "none")

    @property
    def selection(self) -> SelectionRule:
        metrics = _thaw(self._node("metrics")) or {}
        return SelectionRule.of(_thaw(self._node("early_stopping")),
                                metrics.get("selection") if isinstance(metrics, Mapping) else None)

    @property
    def metric_names(self) -> tuple:
        """The metric names the spec asks for, primary first."""
        node = _thaw(self._node("metrics")) or {}
        return tuple(n for n in [node.get("primary")] + list(node.get("secondary") or []) if n)

    @property
    def inputs(self) -> SpecInputs:
        given = _thaw(self._node("input")) or {}
        dc = _thaw(self._node("data_contract")) or {}
        return SpecInputs(manifest=given.get("manifest"),
                          manifest_digest=given.get("manifest_digest"),
                          parts=_freeze(dict(dc.get("parts") or {})),
                          edges=(dc.get("sample") or {}).get("edges"))

    @property
    def hints(self) -> Mapping:
        """`backend.hints`, the measured substrate posture the blueprint binds (read-only)."""
        return MappingProxyType(dict(((_thaw(self._node("backend")) or {}).get("hints") or {})))

    @property
    def blocks(self) -> list:
        """A copy of `blocks`, in spec order (the builder writes regularization into it)."""
        return list(_thaw(self._node("blocks")) or [])

    @property
    def init(self) -> dict:
        return dict(_thaw(self._node("init")) or {})

    def __repr__(self) -> str:
        """The digest is derived, so the dataclass repr no longer carries it — and it is the
        identity `ModelStore.load` refuses a checkpoint over, which makes it the one field a
        traceback must show (EJ-12)."""
        return (f"NetSpec(task={self.task!r}, family={self.family!r}, label={self.label!r}, "
                f"seed={self.seed!r}, digest={self.digest!r})")


#: the block param a dropout declaration sets, per block type (netspec-contract §3, R2)
_DROPOUT_PARAM = (("dropout", "p"), ("rnn", "dropout"), ("gru", "dropout"), ("lstm", "dropout"),
                  ("transformer_encoder", "dropout"))


def _dropout_key(block: Mapping) -> str | None:
    """The param of `block` a dropout rate lands on, or None: a recurrent block of one layer
    reads none (its rate applies between layers)."""
    kind, params = block.get("type"), block.get("params") or {}
    if kind in ("rnn", "gru", "lstm") and int(
            params.get("num_layers", params.get("layers", 1)) or 1) < 2:
        return None
    return dict(_DROPOUT_PARAM).get(kind)


def _agree(where: str, held, declared) -> None:
    """A live copy beside its declaration must say the same (the compiler's R2 refuses it
    first; a spec that skipped the compiler is refused here rather than trained silently)."""
    if held is not None and held != declared:
        raise ValueError(f"{where} is {held!r} but regularization[] declares {declared!r}; "
                         "state it once, in regularization[] (netspec-contract §3, R2)")


def _declared_dropout(blocks, drops: tuple, ancestors: tuple) -> None:
    """Write the first covering declaration's rate into every block that reads one (in place,
    on a thawed copy): a declaration covers the blocks `where` names and their `inner`."""
    for block in blocks or ():
        if not isinstance(block, dict):
            continue
        ids = ancestors + ((str(block["id"]),) if block.get("id") is not None else ())
        key = _dropout_key(block)
        rate = next((v for v, where in drops if where is None or set(where) & set(ids)), None)
        if key is not None and rate is not None:
            _agree(f"block {'.'.join(ids) or block.get('type')} params.{key}",
                   (block.get("params") or {}).get(key), rate)
            block["params"] = {**(block.get("params") or {}), key: float(rate)}
        _declared_dropout((block.get("params") or {}).get("inner"), drops, ids)


@dataclass(frozen=True)
class Regularization:
    """`regularization[]`, the one declaration of dropout, weight decay, label smoothing and
    gradient clipping (netspec-contract §3, rule R2), read once.

    Every reader of those settings reads them here — the optimiser's weight decay, the loss's
    label smoothing, the clipping bound and each dropout rate the builder sets — so a ladder
    delta on `regularization[kind=…]/value` moves the weights. A setting no declaration covers
    keeps its live path (`optimizer.params.weight_decay`, a block's own `p`). The compiler's
    twin, `scripts/vega_regularization.py`, derives the same live paths for
    `spec_snapshot.json` (`read_paths`) and refuses a live copy that disagrees (EJ-49).
    """

    entries: tuple = ()        # (kind, value, where: tuple of block ids | None), spec order

    @staticmethod
    def of(spec: "NetSpec") -> "Regularization":
        rows = _thaw(NetSpec.require(spec)._node("regularization")) or ()
        return Regularization(tuple(
            (str(r.get("kind")), r.get("value"), None if r.get("where") is None else
             tuple(str(w) for w in (r["where"] if isinstance(r["where"], (list, tuple))
                                    else [r["where"]])))
            for r in rows if isinstance(r, Mapping)))

    def declares(self, kind: str) -> bool:
        return any(k == kind for k, _, _ in self.entries)

    def value(self, kind: str, default=None):
        """The first `kind` declaration's value; `default` (the live path's) when none."""
        return next((v for k, v, _ in self.entries if k == kind), default)

    def loss_params(self, takes: tuple, params: Mapping) -> dict:
        """`params` with the declared label smoothing, for a loss that `takes` it; a
        `loss.params` copy that disagrees is refused, as the compiler's R2 refuses it."""
        return self._declared(params, "label_smoothing", "loss.params") \
            if "label_smoothing" in takes else dict(params)

    def optimizer_params(self, params: Mapping) -> dict:
        """`params` with the declared weight decay (a disagreeing copy refused)."""
        out = self._declared(params, "weight_decay", "optimizer.params")
        if self.declares("weight_decay"):
            out["weight_decay"] = float(out["weight_decay"])
        return out

    def _declared(self, params: Mapping, kind: str, where: str) -> dict:
        out = dict(params)
        if self.declares(kind):
            _agree(f"{where}.{kind}", out.get(kind), self.value(kind))
            out[kind] = self.value(kind)
        return out

    def blocks(self, blocks) -> list:
        """A copy of `blocks` with each declared dropout rate written where a block reads one."""
        out = json.loads(json.dumps(list(blocks or ())))      # a copy, key order kept
        drops = tuple((v, where) for k, v, where in self.entries if k == "dropout")
        if any(v is None for v, _ in drops):
            raise ValueError("regularization[kind=dropout] declares no value; a rate is a "
                             "number in [0, 1)")
        _declared_dropout(out, drops, ())
        return out


# --------------------------------------------------------------------------- the task

def _label_format_kind(task: Mapping) -> str | None:
    declared = task.get("label_format")
    return declared.get("kind") if isinstance(declared, Mapping) else declared


def _base_of(kind: TaskKind, task: Mapping) -> TaskKind | None:
    """`task.base_kind` for a sequence task. A v2.2 spec carries none; its base is read from
    the classes and the label format, the fallback the readers keep for such specs only."""
    if kind.value not in _SEQUENCE_BASES:
        return None
    declared = task.get("base_kind")
    if declared:
        return TaskKind(declared)
    classes = tuple(task.get("classes") or ())
    if kind is TaskKind.SEQUENCE_REGRESSION:
        return TaskKind.ORDINAL if classes else TaskKind.REGRESSION
    if _label_format_kind(task) == LabelFormatKind.DELIMITED.value:
        return TaskKind.MULTILABEL
    return TaskKind.BINARY if len(classes) == 2 else TaskKind.MULTICLASS


@dataclass(frozen=True)
class LabelFormat:
    """How the one label column stores the answer: class indices, class names, or (the only
    form a multilabel takes) a delimited set of names, `"a|c"`, the empty string for none."""

    kind: LabelFormatKind
    sep: str = "|"

    def __post_init__(self):
        object.__setattr__(self, "kind", LabelFormatKind(self.kind))
        if self.kind is LabelFormatKind.DELIMITED and not self.sep:
            raise ValueError("a delimited label format needs a non-empty separator")

    @staticmethod
    def from_spec(spec) -> "LabelFormat":
        """`task.label_format`, or its default: delimited for a multilabel, index when the
        classes are integers, names otherwise."""
        task = _spec_dict(spec).get("task") or {}
        declared = task.get("label_format")
        if isinstance(declared, Mapping):
            return LabelFormat(kind=declared.get("kind"), sep=declared.get("sep") or "|")
        if declared:
            return LabelFormat(kind=declared)
        kind = TaskKind(task.get("kind"))
        effective = _base_of(kind, task) or kind
        if effective is TaskKind.MULTILABEL:
            return LabelFormat(LabelFormatKind.DELIMITED)
        classes = tuple(task.get("classes") or ())
        ints = bool(classes) and all(isinstance(c, int) for c in classes)
        return LabelFormat(LabelFormatKind.INDEX if ints else LabelFormatKind.NAMES)


@dataclass(frozen=True)
class Task:
    """The task as a value: its kind, the base a sequence task reads through, the label set the
    head's outputs are ordered by, how the label column stores it, which class means
    "positive", and what the loss trains the head to do.

    `labels` is the spec's `LabelSet` — built once, from `task.classes` in the spec's order —
    and every reader of the classes reads it: the head's width and the weights' arity are
    `len(task.labels)`, a class name is `task.labels.at(i)`, a label column is encoded through
    it. A tuple or list given here becomes that `LabelSet`; nothing keeps the bare tuple.

    `effective` is the kind every output reader keys on (a `sequence-*` task is its base's
    reading over a window; every other kind is itself).
    """

    kind: TaskKind
    base: TaskKind | None = None
    labels: LabelSet = ()
    label: str | None = None
    label_format: LabelFormat | None = None
    positive: str | None = None
    objective: Objective = Objective.PREDICT

    def __post_init__(self):
        object.__setattr__(self, "kind", TaskKind(self.kind))
        object.__setattr__(self, "base", TaskKind(self.base) if self.base else None)
        if not isinstance(self.labels, LabelSet):
            object.__setattr__(self, "labels", LabelSet(tuple(self.labels or ())))
        object.__setattr__(self, "objective", Objective(self.objective))
        if isinstance(self.label_format, str):
            object.__setattr__(self, "label_format", LabelFormat(self.label_format))
        if self.label_format is None:
            fmt = (LabelFormatKind.DELIMITED if self.effective is TaskKind.MULTILABEL
                   else LabelFormatKind.NAMES)
            object.__setattr__(self, "label_format", LabelFormat(fmt))
        self._check_base()
        self._check_classes()

    def _check_base(self):
        bases = _SEQUENCE_BASES.get(self.kind.value)
        if bases is None and self.base is not None:
            raise ValueError(f"task.base_kind {self.base.value!r} is only read by a sequence "
                             f"task; {self.kind.value!r} is its own reading")
        if bases is not None and (self.base is None or self.base.value not in bases):
            got = self.base.value if self.base else None
            raise ValueError(f"a {self.kind.value} task reads through task.base_kind, one of "
                             f"{list(bases)} (catalog sequence_map); got {got!r}")

    def _check_classes(self):
        kind = self.effective.value
        if kind in _CLASS_KINDS and len(self.labels) < 2:
            raise ValueError(f"a {kind} head is ordered by task.classes, and the spec lists "
                             f"{len(self.labels)}")
        if kind == "binary" and len(self.labels) != 2:
            raise ValueError(f"a binary task has two classes, the spec lists {len(self.labels)}")
        if kind == "multilabel" and self.label_format.kind is not LabelFormatKind.DELIMITED:
            raise ValueError("a multilabel label is one column of delimited class names; "
                             "task.label_format.kind must be 'delimited'")
        names = {str(c) for c in self.labels.classes}
        if self.positive is not None and names and str(self.positive) not in names:
            raise ValueError(f"task.positive {self.positive!r} is not one of task.classes")

    @staticmethod
    def from_spec(spec) -> "Task":
        d = _spec_dict(spec)
        t = d.get("task") or {}
        try:
            kind = TaskKind(t.get("kind"))
        except ValueError as exc:
            raise ValueError(f"unknown task kind {t.get('kind')!r}") from exc
        return Task(kind=kind, base=_base_of(kind, t), labels=LabelSet.from_spec(d),
                    label=t.get("label"), label_format=LabelFormat.from_spec(d),
                    positive=t.get("positive"),
                    objective=Objective.of_loss((d.get("loss") or {}).get("name")))

    @property
    def effective(self) -> TaskKind:
        return self.base if self.base is not None else self.kind

    @property
    def is_sequence(self) -> bool:
        return self.kind.value in _SEQUENCE_BASES

    def encode_labels(self, values):
        """The label column as the loss's target, by the effective kind (D1 §9.2) — class
        indices (binary, multiclass, ordinal), a k-hot (multilabel), the value (regression,
        forecast), 0/1 (anomaly) — or None for a kind trained on no label. The one reading of
        a label column: the DataSource's parts and every baseline's truth come from it."""
        values = np.asarray(values)
        kind = self.effective
        # >>> task: binary multiclass ordinal
        if kind in (TaskKind.BINARY, TaskKind.MULTICLASS, TaskKind.ORDINAL):
            return self._class_ids(values)
        # <<< task
        # >>> task: multilabel
        if kind is TaskKind.MULTILABEL:
            return self.labels.encode_multi(values.tolist(), self.label_format.sep)
        # <<< task
        # >>> task: regression forecast
        if kind in (TaskKind.REGRESSION, TaskKind.FORECAST):
            return values.astype(np.float32)
        # <<< task
        # >>> task: anomaly
        if kind is TaskKind.ANOMALY:
            return self._anomalous(values)
        # <<< task
        return None

    # >>> task: binary multiclass ordinal
    def _class_ids(self, values):
        """Class names through the spec's LabelSet, or indices checked against the classes."""
        k = len(self.labels)
        if self.label_format.kind is LabelFormatKind.NAMES or values.dtype.kind in "OUS":
            return np.asarray(self.labels.encode(values.tolist()), dtype=np.int64)
        bad = ~np.isfinite(values.astype(np.float64)) | (values < 0) | (values >= k) \
            | (values != np.floor(values))
        if bad.any():
            raise ValueError(f"label row {int(np.argmax(bad))} holds {values[bad][0]!r}, which "
                             f"is not a class index of the spec's {k} classes")
        return values.astype(np.int64)
    # <<< task

    # >>> task: anomaly
    def _anomalous(self, values):
        """1 where the row is of the class `positive` names, else 0 (a 0/1 label as it is)."""
        if self.positive is None:
            if values.dtype.kind in "OUS" or not np.isin(values, (0, 1)).all():
                raise ValueError("the anomaly label holds values other than 0/1 and "
                                 "task.positive names no anomalous class")
            return values.astype(np.float32)
        try:
            hit = values.astype(np.float64) == float(self.positive)
        except (TypeError, ValueError):
            hit = np.asarray([str(v) == str(self.positive) for v in values.tolist()])
        return hit.astype(np.float32)
    # <<< task


# >>> task: regression forecast
#: a train target whose spread is below this share of its size (at least 1e-6) is held at scale
#: 1e-6·max(1, |mean|): the affine divides nothing and never multiplies the network's number by 0
_TARGET_SCALE_FLOOR = 1e-6


@dataclass(frozen=True)
class TargetAffine:
    """The fixed train-fitted target affine of a regression-family head (A37): `shift` the TRAIN
    target's mean and `scale` its standard deviation (floored away from 0), one value per head
    output channel — per output, per horizon step, and repeated over a quantile head's quantiles,
    horizon-major (channel h·Q + q reads step h, the readout's layout). The head's last
    operation is `y = z · scale + shift`: the network learns in standardised units while its
    output — what the loss, the metrics, the stored predictions and serving read — is in the
    target's own units. Computed from the train part's targets alone; `record()` is what
    `spec_snapshot._resolved.init_notes` carries."""

    shift: tuple
    scale: tuple
    per: str                        # what one channel stands for: output, horizon step, quantile
    quantiles: tuple = ()
    samples: int = 0
    block: str | None = None

    @staticmethod
    def applies(task: Task) -> bool:
        """The regression family — regression, sequence-regression and forecast, a quantile
        head among them — trained to predict."""
        return (task.effective in (TaskKind.REGRESSION, TaskKind.FORECAST)
                and task.objective is Objective.PREDICT)

    @staticmethod
    def of(spec, targets, block: str | None = None) -> "TargetAffine | None":
        """From the train part's targets, `(N,)` or `(N, k)` (k the horizon, or the outputs);
        None off the regression family, without targets, or with a column holding no finite
        value (nothing is invented for it)."""
        d = _spec_dict(spec)
        task = Task.from_spec(d)
        if targets is None or not TargetAffine.applies(task):
            return None
        y = np.asarray(targets, dtype=np.float64)
        columns = [c[np.isfinite(c)] for c in y.reshape(len(y), -1).T]
        if not len(y) or not columns or any(not len(c) for c in columns):
            return None
        qs = LossSpec.of(d.get("loss")).quantiles
        mean = [float(c.mean()) for c in columns]
        std = [max(float(c.std()), _TARGET_SCALE_FLOOR * max(1.0, abs(m)))
               for c, m in zip(columns, mean)]
        repeat = max(1, len(qs))
        steps = "horizon step" if task.effective is TaskKind.FORECAST else "output"
        return TargetAffine(tuple(m for m in mean for _ in range(repeat)),
                            tuple(s for s in std for _ in range(repeat)),
                            f"{steps} and quantile" if qs else steps, qs, int(len(y)), block)

    def record(self) -> dict:
        """The init note: the numbers, what each channel stands for, where they came from."""
        return {"block": self.block, "shift": list(self.shift), "scale": list(self.scale),
                "statistic": "train mean and standard deviation", "per": self.per,
                "quantiles": list(self.quantiles), "samples": self.samples, "part": "train",
                "source": "A37"}

    @staticmethod
    def init_notes(affine: "TargetAffine | None", spec) -> dict:
        """`spec_snapshot._resolved.init_notes` (D1 P12): the init scheme the weights keep and,
        for a regression-family head, the target affine its last operation applies."""
        return {"scheme": (_spec_dict(spec).get("init") or {}).get("scheme"),
                "target_affine": affine.record() if affine else None,
                "notes": [affine.note()] if affine else []}

    def note(self) -> str:
        pairs = "; ".join(f"× {s:.6g} + {m:.6g}" for m, s in zip(self.shift, self.scale))
        return (f"the head{' ' + self.block if self.block else ''} ends on the train target's "
                f"affine per {self.per}, y = z · scale + shift ({pairs}; {self.samples} training "
                "samples, the train part only): the network learns in standardised units and "
                "answers in the target's own")
# <<< task


# --------------------------------------------------------------------------- the sample

# >>> shape: window
@dataclass(frozen=True)
class UnscoredRow:
    """A row of a part that gets no prediction, and why. Never given a fabricated answer."""

    raw_pos: Any
    reason: UnscoredReason

    def __post_init__(self):
        object.__setattr__(self, "raw_pos", _py(self.raw_pos))
        object.__setattr__(self, "reason", UnscoredReason(self.reason))


@dataclass(frozen=True)
class SeriesRows:
    """The key columns of one frame of shaped rows, as a window plan reads them: the raw
    position of each row (its identity), and its `series_by` and `order_by` values."""

    raw_pos: Any
    series: Any = None
    order: Any = None

    def __post_init__(self):
        raw = np.asarray(self.raw_pos if self.raw_pos is not None else ())
        object.__setattr__(self, "raw_pos", raw)
        for name in ("series", "order"):
            values = getattr(self, name)
            if values is not None:
                values = np.asarray(values)
                if len(values) != len(raw):
                    raise ValueError(f"{len(values)} {name} value(s) for {len(raw)} row(s)")
                object.__setattr__(self, name, values)

    def __len__(self):
        return len(self.raw_pos)


@dataclass(frozen=True)
class SeriesBlock:
    """One series inside one assembled flat array: its context rows (borrowed from the
    previous part(s), inputs only) first, then the rows of this part."""

    start: int
    n_context: int
    n_rows: int
    raw_pos: Any
    key: Any = None

    def __post_init__(self):
        raw = np.asarray(self.raw_pos if self.raw_pos is not None else ())
        object.__setattr__(self, "raw_pos", raw)
        if min(int(self.start), int(self.n_context), int(self.n_rows)) < 0:
            raise ValueError("a series block has non-negative start, context and row counts")
        if len(raw) != int(self.n_rows):
            raise ValueError(f"a series block of {self.n_rows} row(s) carries {len(raw)} raw "
                             "position(s)")

    @property
    def stop(self) -> int:
        return int(self.start) + int(self.n_context) + int(self.n_rows)


@dataclass(frozen=True)
class WindowLayout:
    """The flat order a window plan runs over. `rows[i]` indexes the concatenation
    `[context rows ; part rows]` (the context frame first, `n_context_frame` rows)."""

    rows: Any
    blocks: tuple
    n_context_frame: int = 0

    def take(self, context_values, part_values):
        """The flat array for one column block: context rows then part rows, in plan order.
        The frames are the ones the layout was cut from; another context frame is refused."""
        ctx = np.asarray(context_values) if context_values is not None else None
        if (0 if ctx is None else len(ctx)) != int(self.n_context_frame):
            raise ValueError(f"the layout was cut with {self.n_context_frame} context row(s); "
                             "take() needs the same context frame")
        part = np.asarray(part_values)
        stacked = part if ctx is None or not len(ctx) else np.concatenate([ctx, part])
        return stacked[self.rows]


@dataclass(frozen=True)
class WindowPlan:
    """Which flat rows every window reads and predicts. `anchors[w]` is the flat position of
    window w's last input row, `targets[w]` the flat positions of the rows it predicts,
    `row_ids[w]` the anchor's raw position (the row the answer belongs to)."""

    anchors: Any
    targets: Any
    row_ids: Any
    unscored: tuple
    length: int
    series_index: Any = None

    def __len__(self):
        return len(self.anchors)

    def inputs(self):
        """`(W, L)` flat positions each window reads: `anchor-(L-1) … anchor`, never later."""
        return self.anchors[:, None] + np.arange(1 - int(self.length), 1, dtype=np.int64)[None, :]


def _series_groups(rows: SeriesRows, spec: "WindowSpec", label: str) -> list:
    """`[(series key, row indices ordered by order_by)]`, series by first appearance."""
    if spec.series_by and rows.series is None:
        raise ValueError(f"sample.series_by is {spec.series_by!r} but the {label} rows carry "
                         "no series values")
    if spec.order_by and rows.order is None:
        raise ValueError(f"sample.order_by is {spec.order_by!r} but the {label} rows carry no "
                         "order values")
    if spec.series_by:
        codes, keys = _factorize(rows.series, rows.raw_pos, spec.series_by)
    else:
        codes, keys = np.zeros(len(rows), dtype=np.int64), [None]
    by_code = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes, minlength=len(keys)))
    groups, lo = [], 0
    for g, hi in enumerate(bounds):
        idx = by_code[lo:hi]
        if spec.order_by and len(idx):
            gaps = _missing(rows.order[idx])
            if gaps.any():
                raise ValueError(f"the {spec.order_by} column is missing at row "
                                 f"{_py(rows.raw_pos[idx][gaps][0])}; a window is ordered by it")
            idx = idx[np.argsort(rows.order[idx], kind="stable")]
        if len(idx):
            groups.append((keys[g], idx.astype(np.int64)))
        lo = hi
    return groups


def _block_windows(spec: "WindowSpec", block: SeriesBlock, stride: int) -> tuple:
    """Anchors, row ids and unscored rows of one series (rules 3–5 of the plan)."""
    i = np.arange(int(block.n_rows))
    has_context = int(block.n_context) + i >= spec.length - 1
    fits = i + spec.horizon <= int(block.n_rows) - 1
    chosen = np.flatnonzero(has_context & fits)
    if stride > 1 and len(chosen):
        chosen = chosen[(len(chosen) - 1 - np.arange(len(chosen))) % stride == 0]
    unscored = [UnscoredRow(block.raw_pos[k], UnscoredReason.INSUFFICIENT_CONTEXT)
                for k in np.flatnonzero(~has_context)]
    unscored += [UnscoredRow(block.raw_pos[k], UnscoredReason.HORIZON_BEYOND_PART)
                 for k in np.flatnonzero(has_context & ~fits)]
    anchors = int(block.start) + int(block.n_context) + chosen
    return anchors.astype(np.int64), block.raw_pos[chosen], unscored


@dataclass(frozen=True)
class WindowSpec:
    """`data_contract.sample` for `shape: window`, and the plan both training and serving cut
    windows by (rules 1–6 of the backend design, §9.3):

    1. a window never holds rows of two `series_by` values; rows are ordered by `order_by`
       (stable), else by stored order;
    2. `context: previous-part` prefixes each series with the last L−1 rows of the same series
       in the previous part(s); context rows are inputs only, never anchors;
    3. an anchor is a row whose window has L rows within its series (context included);
    4. `stride` thins the training anchors, counted from the series end (the most recent
       anchor is always kept); scoring is dense, one prediction per scorable row;
    5. a sequence task predicts the anchor's own label; a forecast predicts rows
       anchor+1 … anchor+H of the same part and series, never past the part's end;
    6. the autoregressive channel (forecast only) is the label at input rows, all ≤ anchor.

    `horizon` is the forecast's H, and 0 for a sequence task (its target is offset 0).
    """

    length: int
    stride: int = 1
    horizon: int = 0
    series_by: str | None = None
    order_by: str | None = None
    context: WindowContext = WindowContext.PREVIOUS_PART
    autoregressive: bool = False
    season: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "context", WindowContext(self.context))
        if int(self.length) < 2:
            raise ValueError(f"sample.length must be >= 2, got {self.length!r}")
        if int(self.stride) < 1:
            raise ValueError(f"sample.stride must be >= 1, got {self.stride!r}")
        if int(self.horizon) < 0:
            raise ValueError(f"sample.horizon must be >= 1 on a forecast, got {self.horizon!r}")
        if self.autoregressive and not self.is_forecast:
            raise ValueError("sample.autoregressive is forecast-only: on a sequence task the "
                             "anchor's own label is the target, and the channel would carry it")
        if self.season is not None and int(self.season) < 2:
            raise ValueError(f"sample.season must be >= 2 when set, got {self.season!r}")

    def laid_out(self) -> "WindowSpec":
        """The order the assembler lays a series out in, one window per row: this window with
        no horizon, no stride, no autoregressive channel and the previous part as context (a
        served request's per-row notes rank its rows by it)."""
        return replace(self, horizon=0, stride=1, autoregressive=False,
                       context=WindowContext.PREVIOUS_PART)

    @staticmethod
    def from_spec(spec) -> "WindowSpec":
        d = _spec_dict(spec)
        s = _sample_dict(d)
        forecast = (d.get("task") or {}).get("kind") == TaskKind.FORECAST.value
        horizon = int(s.get("horizon") or 0) if forecast else 0
        if forecast and horizon < 1:
            raise ValueError("a forecast declares sample.horizon >= 1")
        return WindowSpec(length=int(s.get("length") or 0), stride=int(s.get("stride") or 1),
                          horizon=horizon, series_by=s.get("series_by"),
                          order_by=s.get("order_by"),
                          context=s.get("context") or WindowContext.PREVIOUS_PART.value,
                          autoregressive=bool(s.get("autoregressive", forecast)),
                          season=s.get("season"))

    @property
    def is_forecast(self) -> bool:
        return int(self.horizon) > 0

    @property
    def offsets(self) -> tuple:
        """The target rows relative to the anchor: `(1 … H)` for a forecast, `(0,)` else."""
        return tuple(range(1, int(self.horizon) + 1)) if self.is_forecast else (0,)

    def history_rows(self, lag_depth: int = 0) -> int:
        """The rows of history a served request sends per series for a fully determined
        answer: the window's L−1 context rows plus the recipe's lag depth."""
        return int(self.length) - 1 + int(lag_depth)

    def layout(self, part: SeriesRows, context: SeriesRows | None = None) -> WindowLayout:
        """Group `part` into series (rule 1) and borrow each series' context (rule 2)."""
        borrowed = {}
        if context is not None and self.context is WindowContext.PREVIOUS_PART:
            borrowed = dict(_series_groups(context, self, "context"))
        n_ctx = len(context) if context is not None else 0
        rows, blocks, start = [], [], 0
        for key, idx in _series_groups(part, self, "part"):
            tail = borrowed.get(key, np.zeros(0, dtype=np.int64))[-(int(self.length) - 1):]
            rows += [tail, idx + n_ctx]
            blocks.append(SeriesBlock(start=start, n_context=len(tail), n_rows=len(idx),
                                      raw_pos=part.raw_pos[idx], key=key))
            start += len(tail) + len(idx)
        flat = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        return WindowLayout(rows=flat.astype(np.int64), blocks=tuple(blocks),
                            n_context_frame=n_ctx)

    def plan(self, blocks: Sequence[SeriesBlock], dense: bool) -> WindowPlan:
        """The windows over `blocks` (rules 3–5): every scorable row when `dense`, the
        stride-thinned training anchors otherwise."""
        stride = 1 if dense else int(self.stride)
        per = [_block_windows(self, b, stride) for b in blocks]
        anchors = (np.concatenate([p[0] for p in per]) if per
                   else np.zeros(0, dtype=np.int64))
        row_ids = np.concatenate([p[1] for p in per]) if per else np.zeros(0, dtype=np.int64)
        series = np.concatenate([np.full(len(p[0]), b, dtype=np.int64)
                                 for b, p in enumerate(per)]) if per else np.zeros(0, np.int64)
        targets = anchors[:, None] + np.asarray(self.offsets, dtype=np.int64)[None, :]
        unscored = tuple(u for p in per for u in p[2])
        return WindowPlan(anchors=anchors, targets=targets, row_ids=row_ids, unscored=unscored,
                          length=int(self.length), series_index=series)


# <<< shape


# >>> shape: graph
@dataclass(frozen=True)
class GraphExposure:
    """What the training graph saw of the test part: for the eval record (`metrics.graph`).
    Zero and false for inductive by construction; disclosed for transductive. A graph
    package's: the DataSource port's `exposure` returns it."""

    protocol: GraphProtocol
    edges_into_test: int = 0
    features_seen_in_training: bool = False

    def __post_init__(self):
        object.__setattr__(self, "protocol", GraphProtocol(self.protocol))

    def as_dict(self) -> dict:
        return {"protocol": self.protocol.value, "edges_into_test": int(self.edges_into_test),
                "features_seen_in_training": bool(self.features_seen_in_training)}
# <<< shape


# >>> shape: graph
# --------------------------------------------------------------------------- graphs

def _part_sizes(value) -> tuple:
    sizes = tuple(int(v) for v in value)
    if len(sizes) != 3 or min(sizes) < 0:
        raise ValueError("part sizes are (train, valid, test) non-negative counts, got "
                         f"{value!r}")
    return sizes


def _part_of(ids, sizes: tuple):
    """0 train, 1 valid, 2 test for global node ids over `[train | valid | test]`."""
    return np.searchsorted(np.cumsum(sizes), ids, side="right")


def _attach(ref_end, query_end, rows) -> tuple:
    """The inference-graph rule: every (reference, query) pair once, as a reference → query
    edge, in first-seen order (a symmetric edges part lists each pair twice; one message)."""
    if not len(ref_end):
        return ref_end, query_end, rows
    width = int(max(ref_end.max(), query_end.max())) + 1
    _, first = np.unique(ref_end * width + query_end, return_index=True)
    keep = np.sort(first)
    return ref_end[keep], query_end[keep], rows[keep]


@dataclass(frozen=True)
class GraphEdges:
    """The edges part resolved to global node ids over `[train | valid | test]`: row 0 of an
    edge is its source, row 1 its destination (a message goes from source to destination)."""

    src: Any
    dst: Any
    weight: Any = None

    def __post_init__(self):
        src, dst = _index_array(self.src, "src"), _index_array(self.dst, "dst")
        if len(src) != len(dst):
            raise ValueError(f"{len(src)} edge source(s) for {len(dst)} destination(s)")
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dst", dst)
        if self.weight is not None:
            weight = np.asarray(self.weight, dtype=np.float32)
            if weight.shape != src.shape:
                raise ValueError(f"{len(weight)} edge weight(s) for {len(src)} edge(s)")
            object.__setattr__(self, "weight", weight)

    def __len__(self):
        return len(self.src)

    def check(self, n_nodes: int) -> None:
        bad = np.flatnonzero((self.src < 0) | (self.src >= n_nodes) |
                             (self.dst < 0) | (self.dst >= n_nodes))
        if len(bad):
            k = int(bad[0])
            raise ValueError(f"edge {k} ({int(self.src[k])} → {int(self.dst[k])}) names a node "
                             f"outside the {n_nodes} node(s) of the parts")


@dataclass(frozen=True)
class QueryEdges:
    """A served query's links to the reference graph (the shaper's `edges_for`): query row
    `query[k]` of the request is linked to reference node `ref[k]`."""

    query: Any
    ref: Any
    weight: Any = None

    def __post_init__(self):
        query, ref = _index_array(self.query, "query"), _index_array(self.ref, "ref")
        if len(query) != len(ref):
            raise ValueError(f"{len(query)} query position(s) for {len(ref)} reference id(s)")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "ref", ref)
        if self.weight is not None:
            object.__setattr__(self, "weight", np.asarray(self.weight, dtype=np.float32))


@dataclass(frozen=True)
class GraphView:
    """The nodes and edges one graph pass reads. `node_ids` are global ids (reference nodes
    first, then query nodes); `edge_index` and `target_index` are positions in `node_ids`;
    `edge_rows` index the edges part (for weights and audit). `target_index` is None at graph
    level, where the targets are graphs (`GraphSpec.graphs`)."""

    part: Part
    node_ids: Any
    n_reference: int
    edge_index: Any
    edge_rows: Any
    edge_weight: Any
    target_index: Any
    feature_parts: tuple
    label_parts: tuple
    test_edges_withheld: int = 0

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def reference_ids(self):
        return self.node_ids[:int(self.n_reference)]

    @property
    def query_ids(self):
        return self.node_ids[int(self.n_reference):]


def _view(part, nodes, edge_parts: tuple, extra: tuple) -> GraphView:
    """Assemble a view from `nodes = (node_ids, n_reference, target_index)`, `edge_parts =
    (src, dst, rows, weights)` and `extra = (feature_parts, label_parts, withheld)`."""
    node_ids, n_ref, targets = nodes
    src, dst, rows, weights = edge_parts
    return GraphView(part=part, node_ids=np.asarray(node_ids, dtype=np.int64),
                     n_reference=int(n_ref),
                     edge_index=np.stack([src, dst]).astype(np.int64).reshape(2, -1),
                     edge_rows=np.asarray(rows, dtype=np.int64),
                     edge_weight=None if weights is None else weights[rows],
                     target_index=targets, feature_parts=extra[0], label_parts=extra[1],
                     test_edges_withheld=int(extra[2]))


def _inductive_view(sizes: tuple, edges: GraphEdges, part: Part) -> GraphView:
    """Reference = the parts before `part` (train for valid, train ∪ valid for test; the
    train view is the train graph alone); queries = `part`, attached reference → query."""
    q = _PART_ORDER.index(part)
    ps, pd = _part_of(edges.src, sizes), _part_of(edges.dst, sizes)
    n_ref = sizes[0] if q < 2 else sizes[0] + sizes[1]
    n_nodes = n_ref + (sizes[q] if q else 0)
    ref_rows = np.flatnonzero((edges.src < n_ref) & (edges.dst < n_ref))
    src, dst = edges.src[ref_rows], edges.dst[ref_rows]
    if q:
        forward, backward = (edges.src < n_ref) & (pd == q), (edges.dst < n_ref) & (ps == q)
        rows = np.flatnonzero(forward | backward)
        ref_end = np.where(forward, edges.src, edges.dst)[rows]
        query_end = np.where(forward, edges.dst, edges.src)[rows]
        a_src, a_dst, a_rows = _attach(ref_end, query_end, rows)
        src, dst = np.concatenate([src, a_src]), np.concatenate([dst, a_dst])
        ref_rows = np.concatenate([ref_rows, a_rows])
    withheld = int(((ps == 2) | (pd == 2)).sum()) if q < 2 else 0
    targets = np.arange(n_ref if q else 0, n_nodes, dtype=np.int64)
    return _view(part, (np.arange(n_nodes), n_ref, targets),
                 (src, dst, ref_rows, edges.weight),
                 (_PART_ORDER[:q + 1], (part,), withheld))


def _transductive_view(sizes: tuple, edges: GraphEdges, part: Part) -> GraphView:
    """Every node and every edge; the targets are `part`'s nodes (the disclosed exposure)."""
    q = _PART_ORDER.index(part)
    lo = sum(sizes[:q])
    total = sum(sizes)
    rows = np.arange(len(edges), dtype=np.int64)
    return _view(part, (np.arange(total), total, np.arange(lo, lo + sizes[q], dtype=np.int64)),
                 (edges.src, edges.dst, rows, edges.weight), (_PART_ORDER, (part,), 0))


def _graph_level_view(sizes: tuple, edges: GraphEdges, part: Part) -> GraphView:
    """The part's own graphs: its nodes and the edges among them. An edge with one end in
    the part joins two graphs (a graph lies in one part), which is a data defect."""
    q = _PART_ORDER.index(part)
    lo = sum(sizes[:q])
    hi = lo + sizes[q]
    in_src = (edges.src >= lo) & (edges.src < hi)
    in_dst = (edges.dst >= lo) & (edges.dst < hi)
    crossing = np.flatnonzero(in_src ^ in_dst)
    if len(crossing):
        k = int(crossing[0])
        a, b = (_PART_ORDER[int(p)].value for p in _part_of(
            np.array([edges.src[k], edges.dst[k]]), sizes))
        raise ValueError(f"edge {k} ({int(edges.src[k])} → {int(edges.dst[k])}) joins parts "
                         f"{a}/{b}: at level graph an edge stays inside one graph, and a graph "
                         "inside one part (a group split by graph_id)")
    rows = np.flatnonzero(in_src & in_dst)
    touches_test = (_part_of(edges.src, sizes) == 2) | (_part_of(edges.dst, sizes) == 2)
    withheld = int(touches_test.sum()) if q < 2 else 0
    return _view(part, (np.arange(lo, hi), hi - lo, None),
                 (edges.src[rows] - lo, edges.dst[rows] - lo, rows, edges.weight),
                 ((part,), (part,), withheld))


@dataclass(frozen=True)
class GraphGroups:
    """Graph-level grouping of a view's nodes: `graph_index[i]` is node i's graph (0…G−1, by
    first appearance), `values` the graph ids, `first_rows` one node per graph (its label)."""

    values: tuple
    graph_index: Any
    first_rows: Any

    @property
    def num_graphs(self) -> int:
        return len(self.values)

    def labels_of(self, labels):
        """One label per graph, from the rows `GraphSpec.graphs` checked agree."""
        return None if labels is None else labels[self.first_rows]


def _rows_differ(labels, reference):
    a, b = np.asarray(labels), np.asarray(reference)
    same = a == b
    if a.dtype.kind == "f":
        same = same | (np.isnan(a) & np.isnan(b))
    return ~same.reshape(len(a), -1).all(axis=1)


@dataclass(frozen=True)
class GraphSpec:
    """`data_contract.sample` for `shape: graph`, and the views every graph pass reads.

    The inference-graph rule (build plan R-A3), for training, evaluation and serving alike:
    reference nodes are computed over the reference graph (every edge between two reference
    nodes); query nodes receive messages from reference nodes only — no query → reference and
    no query ↔ query edge — so an answer never depends on the other rows being answered.
    """

    edges: str | None = None
    level: GraphLevel = GraphLevel.NODE
    graph_id: str | None = None
    directed: bool = False
    self_loops: bool = False
    protocol: GraphProtocol = GraphProtocol.INDUCTIVE
    batching: GraphBatching = GraphBatching.FULL
    neighbours: tuple | None = None
    declared_exposure: Any = None

    def __post_init__(self):
        for name, kind in (("level", GraphLevel), ("protocol", GraphProtocol),
                           ("batching", GraphBatching)):
            object.__setattr__(self, name, kind(getattr(self, name)))
        if self.neighbours is not None:
            object.__setattr__(self, "neighbours", tuple(int(k) for k in self.neighbours))
        object.__setattr__(self, "declared_exposure", _freeze(self.declared_exposure) or None)
        if (self.level is GraphLevel.GRAPH) != bool(self.graph_id):
            raise ValueError("sample.graph_id is required iff sample.level is 'graph'")
        if (self.batching is GraphBatching.GRAPHS) != (self.level is GraphLevel.GRAPH):
            raise ValueError("sample.batching is 'graphs' iff sample.level is 'graph'")
        sampled = self.batching is GraphBatching.SAMPLED
        if sampled != bool(self.neighbours) or (sampled and min(self.neighbours) < 1):
            raise ValueError("sample.neighbours (a positive fan-out per conv block) is "
                             "required iff sample.batching is 'sampled'")
        self._check_exposure()

    def _check_exposure(self):
        source = str(dict(_thaw(self.declared_exposure) or {}).get("source") or "")
        if self.protocol is GraphProtocol.TRANSDUCTIVE and not source.startswith("user:"):
            raise SealedPartError("a transductive graph reads the test part's features during "
                                  "training; that exposure needs sample.exposure with a "
                                  "'user:' source, and none was declared")
        if self.protocol is GraphProtocol.INDUCTIVE and self.declared_exposure:
            raise ValueError("sample.exposure is declared only by a transductive protocol")

    @staticmethod
    def from_spec(spec) -> "GraphSpec":
        s = _sample_dict(spec)
        level = GraphLevel(s.get("level") or GraphLevel.NODE.value)
        batching = s.get("batching") or ("graphs" if level is GraphLevel.GRAPH else "full")
        return GraphSpec(edges=s.get("edges"), level=level, graph_id=s.get("graph_id"),
                         directed=bool(s.get("directed", False)),
                         self_loops=bool(s.get("self_loops", False)),
                         protocol=s.get("protocol") or GraphProtocol.INDUCTIVE.value,
                         batching=batching, neighbours=s.get("neighbours") or None,
                         declared_exposure=_freeze(s["exposure"]) if s.get("exposure") else None)

    def view(self, part_sizes, edges: GraphEdges, part) -> GraphView:
        """The view of `part` over the parts `(train, valid, test)` sized `part_sizes`."""
        sizes, part = _part_sizes(part_sizes), Part(part)
        edges.check(sum(sizes))
        if self.level is GraphLevel.GRAPH:
            return _graph_level_view(sizes, edges, part)
        if self.protocol is GraphProtocol.TRANSDUCTIVE:
            return _transductive_view(sizes, edges, part)
        return _inductive_view(sizes, edges, part)

    def exposure(self, part_sizes, edges: GraphEdges) -> GraphExposure:
        """Computed from the training view: messages into test nodes and whether test
        features were in the training graph."""
        if self.protocol is GraphProtocol.INDUCTIVE or self.level is GraphLevel.GRAPH:
            return GraphExposure(self.protocol)
        sizes = _part_sizes(part_sizes)
        train = self.view(sizes, edges, Part.TRAIN)
        into_test = int((train.node_ids[train.edge_index[1]] >= sizes[0] + sizes[1]).sum())
        return GraphExposure(self.protocol, into_test, sizes[2] > 0)

    @staticmethod
    def graphs(graph_ids, edge_index=None, labels=None) -> GraphGroups:
        """Group a view's nodes by `graph_id`, refusing an edge between two graphs and a graph
        whose rows disagree on the label (each named)."""
        codes, values = _factorize(graph_ids, np.arange(len(graph_ids)), "graph_id")
        first = np.unique(codes, return_index=True)[1].astype(np.int64)
        if edge_index is not None and np.asarray(edge_index).size:
            src, dst = np.asarray(edge_index)
            cross = np.flatnonzero(codes[src] != codes[dst])
            if len(cross):
                k = int(cross[0])
                raise ValueError(f"edge {int(src[k])} → {int(dst[k])} joins graph "
                                 f"{values[codes[src[k]]]!r} and graph {values[codes[dst[k]]]!r}; "
                                 "an edge stays inside one graph")
        if labels is not None:
            differ = np.flatnonzero(_rows_differ(labels, np.asarray(labels)[first[codes]]))
            if len(differ):
                g = values[codes[int(differ[0])]]
                raise ValueError(f"graph {g!r} has rows with different labels; a graph's label "
                                 "is the common label of its rows")
        return GraphGroups(values=tuple(values), graph_index=codes, first_rows=first)

    @staticmethod
    def check_split(ids_by_part: Mapping) -> None:
        """Refuse a graph whose rows lie in two parts, naming it and the parts."""
        seen: dict = {}
        for part, ids in ids_by_part.items():
            name = Part(part).value
            for value in sorted({_py(v) for v in np.asarray(ids).tolist()}, key=str):
                if seen.setdefault(value, name) != name:
                    raise ValueError(f"graph {value!r} straddles parts {seen[value]}/{name} — the "
                                     "shaper's split must be a group split by graph_id")


@dataclass(frozen=True)
class InferenceGraph:
    """A served node-level batch graph: the reference nodes `0…R−1`, then the query nodes
    `R…R+Q−1` (the targets), with reference edges plus reference → query attachments."""

    edge_index: Any
    edge_weight: Any
    target_index: Any
    n_nodes: int


@dataclass(frozen=True)
class GraphReference:
    """The bundle's reference graph (`vega/graph-reference@1`) as values: the shaped features
    of the train ∪ valid nodes in dense `ref` order and the edges among them. It carries no
    raw position — those would reveal the test part's membership by complement."""

    x_num: Any
    x_cat: Any
    edge_index: Any
    edge_weight: Any = None

    def __post_init__(self):
        n = int(np.asarray(self.x_num).shape[0])
        if self.x_cat is not None and int(np.asarray(self.x_cat).shape[0]) != n:
            raise ValueError("the reference nominal block and numeric block differ in rows")
        edges = np.asarray(self.edge_index, dtype=np.int64).reshape(2, -1)
        if edges.size and (edges.min() < 0 or edges.max() >= n):
            raise ValueError(f"a reference edge names a ref id outside 0…{n - 1}")
        if self.edge_weight is not None and len(self.edge_weight) != edges.shape[1]:
            raise ValueError("the reference edge weights and edges differ in length")
        object.__setattr__(self, "edge_index", edges)

    @staticmethod
    def from_frames(nodes_frame, edges_frame, contract: FeatureContract) -> "GraphReference":
        """`reference-nodes.parquet` (`ref` + the contract's shaped features) and
        `reference-edges.parquet` (`src_ref, dst_ref, weight`) as a reference graph."""
        _require_columns(nodes_frame, ("ref",) + contract.numeric_columns
                         + contract.nominal_columns, "reference nodes")
        _require_columns(edges_frame, ("src_ref", "dst_ref"), "reference edges")
        ref = np.asarray(nodes_frame["ref"].to_numpy(), dtype=np.int64)
        order = np.argsort(ref, kind="stable")
        if not np.array_equal(ref[order], np.arange(len(ref))):
            raise ValueError("reference nodes: the ref ids must be the dense ids 0…R−1, once "
                             "each")
        x_num = _column_block(nodes_frame, contract.numeric_columns, "float32")[order]
        x_cat = (_column_block(nodes_frame, contract.nominal_columns, "int64")[order]
                 if contract.nominal_columns else None)
        edges = np.stack([np.asarray(edges_frame["src_ref"].to_numpy(), dtype=np.int64),
                          np.asarray(edges_frame["dst_ref"].to_numpy(), dtype=np.int64)])
        return GraphReference(x_num=x_num, x_cat=x_cat, edge_index=edges,
                              edge_weight=_edge_weights(edges_frame))

    @property
    def n_nodes(self) -> int:
        return int(np.asarray(self.x_num).shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def attach(self, pairs: QueryEdges, n_queries: int) -> InferenceGraph:
        """Reference edges plus each (reference, query) link once as reference → query: the
        rule `GraphSpec.view` applies to the sealed test part, applied to a request."""
        r, q = self.n_nodes, int(n_queries)
        if len(pairs.query) and (pairs.query.min() < 0 or pairs.query.max() >= q
                                 or pairs.ref.min() < 0 or pairs.ref.max() >= r):
            raise ValueError(f"a query link leaves the {q} query row(s) or the {r} reference "
                             "node(s)")
        if (self.edge_weight is None) != (pairs.weight is None) and len(pairs.query):
            raise ValueError("the reference edges and the query links disagree on weights")
        rows = np.arange(len(pairs.query), dtype=np.int64)
        src, dst, rows = _attach(pairs.ref, pairs.query + r, rows)
        index = np.concatenate([self.edge_index, np.stack([src, dst]).reshape(2, -1)], axis=1)
        weight = None if self.edge_weight is None else np.concatenate(
            [np.asarray(self.edge_weight, dtype=np.float32),
             np.asarray(pairs.weight if pairs.weight is not None else (), np.float32)[rows]])
        return InferenceGraph(edge_index=index, edge_weight=weight,
                              target_index=np.arange(r, r + q, dtype=np.int64), n_nodes=r + q)


def _require_columns(frame, columns, what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"the {what} frame lacks the column(s) {missing}")


def _column_block(frame, columns, dtype: str):
    if not columns:
        return np.zeros((len(frame), 0), dtype=dtype)
    try:
        return np.asarray(frame[list(columns)].to_numpy(dtype=dtype))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the reference columns {list(columns)} do not all read as {dtype} "
                         f"(a missing value?): {exc}") from exc


def _edge_weights(edges_frame):
    if "weight" not in edges_frame.columns:
        return None
    weight = np.asarray(edges_frame["weight"].to_numpy(dtype="float64", na_value=np.nan))
    missing = np.isnan(weight)
    if missing.all():
        return None
    if missing.any():
        raise ValueError(f"reference edge {int(np.argmax(missing))} has no weight while others "
                         "do; an edges part is weighted throughout or not at all")
    return weight.astype(np.float32)
# <<< shape


@dataclass(frozen=True)
class SampleSpec:
    """`data_contract.sample` as a value. The key columns (`series_by`, `order_by`,
    `graph_id`) are derived from it: they group and order rows and are never features."""

    shape: SampleShape = SampleShape.ROW
    # >>> shape: row graph
    window: None = None                  # this package carries no window sample
    # <<< shape
    # >>> shape: window
    window: WindowSpec | None = None
    # <<< shape
    # >>> shape: row window
    graph: None = None                   # this package carries no graph sample
    # <<< shape
    # >>> shape: graph
    graph: GraphSpec | None = None
    # <<< shape

    def __post_init__(self):
        object.__setattr__(self, "shape", SampleShape(self.shape))
        if (self.window is not None) != (self.shape is SampleShape.WINDOW):
            raise ValueError("a window sample carries its WindowSpec, and only a window sample")
        if (self.graph is not None) != (self.shape is SampleShape.GRAPH):
            raise ValueError("a graph sample carries its GraphSpec, and only a graph sample")

    @staticmethod
    def from_spec(spec) -> "SampleSpec":
        d = _spec_dict(spec)
        shape = _carried(SampleShape(_sample_dict(d).get("shape") or SampleShape.ROW.value))
        return SampleSpec(shape=shape,
                          # >>> shape: window
                          window=WindowSpec.from_spec(d) if shape is SampleShape.WINDOW else None,
                          # <<< shape
                          # >>> shape: graph
                          graph=GraphSpec.from_spec(d) if shape is SampleShape.GRAPH else None,
                          # <<< shape
                          )

    @property
    def keys(self) -> tuple:
        named: tuple = ()
        # >>> shape: window
        if self.window is not None:
            named += (self.window.series_by, self.window.order_by)
        # <<< shape
        # >>> shape: graph
        if self.graph is not None:
            named += (self.graph.graph_id,)
        # <<< shape
        return tuple(k for k in named if k)

    @property
    def is_node_level(self) -> bool:
        # >>> shape: graph
        if self.graph is not None:
            return self.graph.level is GraphLevel.NODE
        # <<< shape
        return False

    @property
    def is_graph_level(self) -> bool:
        # >>> shape: graph
        if self.graph is not None:
            return self.graph.level is GraphLevel.GRAPH
        # <<< shape
        return False


@dataclass(frozen=True)
class SampleIndex:
    """Which row of the request each sample's answer belongs to: `rows[i]` is sample i's row
    position (−1 for a whole graph); `spans[i]` the (first, last) row a window read;
    `graphs[i]` the graph id a graph-level sample answers."""

    shape: SampleShape
    rows: tuple
    spans: tuple | None = None
    graphs: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "shape", SampleShape(self.shape))
        object.__setattr__(self, "rows", tuple(int(r) for r in _listify(self.rows)))
        if self.spans is not None:
            spans = tuple((int(a), int(b)) for a, b in _listify(self.spans))
            object.__setattr__(self, "spans", spans)
        if self.graphs is not None:
            object.__setattr__(self, "graphs", tuple(_py(g) for g in _listify(self.graphs)))
        for name in ("spans", "graphs"):
            values = getattr(self, name)
            if values is not None and len(values) != len(self.rows):
                raise ValueError(f"{len(values)} {name} for {len(self.rows)} sample(s)")

    @staticmethod
    def of_rows(n: int) -> "SampleIndex":
        return SampleIndex(SampleShape.ROW, tuple(range(int(n))))

    def __len__(self):
        return len(self.rows)

    def sample_of(self, row: int) -> int:
        """The sample whose answer belongs to request row `row` (an explained node's
        position inside `target_index`)."""
        try:
            return self.rows.index(int(row))
        except ValueError as exc:
            raise ValueError(f"row {row} has no sample of its own in this batch") from exc


# --------------------------------------------------------------------------- artifacts

@dataclass(frozen=True)
class Checkpoint:
    """A saved model, by one name: the store saves one, a bundle pins one, a fit returns one.

    Identity: `weights_digest`, the sha256 of the weights themselves (every tensor's name,
    dtype, shape and bytes, in key order). It is written *inside* the file when the store
    saves it and recomputed by the store on every load, so it travels with the weights under
    any file name — a bundle renames `best.pt` to `weights.pt`, and a digest kept beside the
    file would not follow it. `digest` is the file's own sha256, for a caller that pins a
    specific file. Equality and the hash read the identity — the weights' digest and the spec's
    they were trained against — never the path, the file digest or the epoch, which describe
    where and when the weights were written.
    """

    path: str = field(compare=False)
    digest: str = field(compare=False)
    epoch: int | None = field(default=None, compare=False)
    spec_digest: str | None = None
    weights_digest: str | None = None

    @staticmethod
    def of(path, digest: str, spec_digest: str | None = None, epoch: int | None = None,
           weights_digest: str | None = None) -> "Checkpoint":
        """The Registry adapter hashes the file and builds the value from the digest; the
        value object holds identity and does no IO. The identity is whole or refused: the
        file's digest, the weights' and the spec's it was trained against."""
        for name, value in (("digest", digest), ("spec_digest", spec_digest),
                            ("weights_digest", weights_digest)):
            if not str(value or "").startswith("sha256:"):
                raise ValueError(f"a checkpoint's {name} is 'sha256:<hex>', got {value!r}")
        return Checkpoint(path=str(path), digest=str(digest), epoch=epoch,
                          spec_digest=spec_digest, weights_digest=weights_digest)


# >>> compat: ModelVersion
#: the v2.3 RC name of `Checkpoint`, kept for one release so a blueprint that names the artifact
#: that way still finds it (emitted only for such a blueprint); new code says `Checkpoint`
ModelVersion = Checkpoint
# <<< compat


@dataclass(frozen=True)
class MetricOptions:
    """The knobs a metric reads, as one value (`metric(name, y_true, y_score, options)`).
    `threshold` is in probability space; binary and multilabel scores are logits."""

    threshold: float = 0.5
    k: int | None = None
    rate: float | None = None
    bins: int = 15
    season: int = 1
    mase_scale: float | None = None
    quantiles: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles or ()))

    def stated(self, spec: NetSpec) -> MetricOptions:
        """These options with what the spec states where they keep the default: a quantile
        loss's levels
        # >>> shape: window
        and a window's season (`sample.season`)
        # <<< shape
        are the spec's, not choices. The Trainer and the Evaluator resolve their options here
        alike."""
        options = replace(self, quantiles=self.quantiles or spec.loss.quantiles)
        # >>> shape: window
        window = SampleSpec.from_spec(spec).window
        if window is not None and options.season == 1 and window.season:
            options = replace(options, season=int(window.season))
        # <<< shape
        return options

    # >>> shape: window
    def scaled(self, spec: NetSpec, names, data) -> MetricOptions:
        """These options with a forecast window's `mase_scale` when `mase` is among `names`
        and no scale was given: `data.target_scale(spec, season)`, the DataSource's train
        target (never the part being scored). A window package's method only: no other
        shape has a target scale to read."""
        if self.mase_scale is not None or "mase" not in names or data is None:
            return self
        return replace(self, mase_scale=data.target_scale(spec, self.season))
    # <<< shape


@dataclass(frozen=True)
class Metrics:
    """A metric record. Identity: the run plus, for the test part, the seal exposure — the
    sealed test's record as the Evaluator port returns it (`score_sealed_test`) carries both,
    read from the seal it was opened under."""

    part: Part
    values: _FrozenMap
    run: str | None = None
    exposure: int | None = None

    @staticmethod
    def from_predictions(part: Part, scored: "ScoredPart", names: Sequence[str], metric,
                         options: MetricOptions | None = None) -> "Metrics":
        """`metric` is the metric function — `{{MODULE_MODEL}}/metrics.py`'s `metric(name,
        y_true, y_score, options)` — passed in, not imported.

        `domain` is the module every other module depends on and that depends on nothing
        (the blueprint's map gives it no arrows), so a value object may not reach into an
        adapter's module to build itself. Taking the function as a parameter keeps the named
        static factory of EJ-1 without the arrow C4 forbids. The run and the seal exposure are
        provenance, not inputs: `with_provenance` stamps them.
        """
        if not callable(metric):
            raise ValueError("from_predictions needs the metric function as `metric`; domain "
                             "imports no adapter module")
        return Metrics(part=Part(part),
                       values=_freeze({n: metric(n, scored.truth, scored.score, options)
                                       for n in names}))

    @staticmethod
    def of_values(part: Part, values: Mapping[str, Any], run: str | None = None,
                  exposure: int | None = None) -> "Metrics":
        """Metrics already computed — by the Trainer's evaluation of the valid part, through
        the same `metrics.py` the Evaluator calls — frozen into the value the port returns."""
        return Metrics(part=Part(part), values=_freeze(dict(values)), run=run, exposure=exposure)

    def with_provenance(self, run: str | None, exposure: int | None) -> "Metrics":
        """The same values, stamped with the run that produced them and the seal exposure."""
        return replace(self, run=run, exposure=exposure)

    def as_dict(self) -> dict:
        return _thaw(self.values)

    def __getitem__(self, name):
        return self.as_dict()[name]


@dataclass(frozen=True)
class ClassWeights:
    """The per-class loss weights, as the blueprint's artifact rather than a raw list.

    Identity: the tuple of values, in the order of the spec's `task.classes` — the order the
    head's logits are in, so a weight can only ever meet the logit it was computed for. The
    spec computed them (inverse frequency on the deduplicated train part, clipped); this type
    carries them across the boundary and refuses a list that could not be a weighting: the
    wrong length for the head, a non-finite value, a value that is not positive (EJ-49).

    Why it is a type and not a list in the loss's params: the loss used to turn the list into
    a fresh tensor on every optimizer step. That is a host-to-device copy per step on the one
    path the substrate measurement says is launch-bound, and it is also a tensor created
    *inside* the step, which is exactly what a captured CUDA graph cannot contain. The loss
    now asks this value for its device tensor once and keeps it (EJ-6: do not create an object
    that can be reused).
    """

    values: tuple

    def __post_init__(self):
        vals = tuple(float(v) for v in self.values)
        if not vals:
            raise ValueError("ClassWeights needs at least one weight")
        bad = [i for i, v in enumerate(vals) if not (v > 0.0 and v < float("inf"))]
        if bad:
            raise ValueError(f"class weight(s) at index {bad[:5]} are not finite and positive: "
                             f"{[vals[i] for i in bad[:5]]}")
        object.__setattr__(self, "values", vals)

    @staticmethod
    def of(values, n_classes: int | None = None) -> "ClassWeights":
        w = ClassWeights(tuple(values))
        if n_classes is not None and len(w) != int(n_classes):
            raise ValueError(f"{len(w)} class weight(s) for a head of {n_classes} classes; the "
                             "weights and task.classes must be in one-to-one order")
        return w

    @staticmethod
    def or_scalar(value):
        """A list of per-class or per-label values as `ClassWeights`; a number or None as is
        (a loss parameter that is one weight or one per output: `pos_weight`, focal's `alpha`)."""
        return ClassWeights.of(value) if isinstance(value, (list, tuple)) else value

    def __len__(self):
        return len(self.values)


# >>> task: multilabel
def _split_labels(cell, sep: str) -> tuple:
    """One delimited label cell as its names: `"a|c"` → `("a", "c")`; the empty string, a
    missing cell and an empty list are the empty set."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return ()
    if isinstance(cell, str):
        return tuple(p.strip() for p in cell.split(sep) if p.strip())
    return tuple(str(p).strip() for p in _listify(cell) if str(p).strip())
# <<< task


@dataclass(frozen=True)
class LabelSet:
    """The classes the head's outputs are defined against, in the order the spec lists them.

    The blueprint models a class as a value, not a string, precisely so a row claiming two of
    them is a contradiction rather than a mismatch — but the class list is the spec's, not the
    backend's, so it cannot be an `Enum` written into this file. `LabelSet` is that enum made
    from data: `of(name)` and `at(index)` are the only constructors of a class value, and both
    refuse a member the spec does not list, naming the offender (EJ-34 in the form the data
    allows, EJ-49).
    """

    classes: tuple

    def __post_init__(self):
        object.__setattr__(self, "classes", tuple(self.classes or ()))
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("the spec's task.classes repeats a class name")

    @staticmethod
    def from_spec(spec_dict: Mapping[str, Any]) -> "LabelSet":
        """The spec's `task.classes`, in its order: the one factory (`Task.from_spec` calls it,
        and the served Predictor reads the same set through it)."""
        return LabelSet(tuple((_spec_dict(spec_dict).get("task") or {}).get("classes") or ()))

    def __len__(self):
        return len(self.classes)

    def of(self, name) -> "ClassValue":
        try:
            return ClassValue(name=name, index=self.classes.index(name),
                              _classes=self.classes)
        except ValueError as exc:
            raise ValueError(f"{name!r} is not one of the spec's task.classes") from exc

    def at(self, index: int) -> "ClassValue":
        if not 0 <= int(index) < len(self.classes):
            raise ValueError(f"class index {index!r} is outside the spec's "
                             f"{len(self.classes)} classes")
        return ClassValue(name=self.classes[int(index)], index=int(index),
                          _classes=self.classes)

    def encode(self, names) -> tuple:
        """Names to output indices, refusing any name the spec does not list."""
        index = {c: i for i, c in enumerate(self.classes)}
        unknown = sorted({str(v) for v in names if v not in index})
        if unknown:
            raise ValueError("the label column holds value(s) the spec's task.classes does "
                             f"not list: {unknown[:5]}")
        return tuple(index[v] for v in names)

    # >>> task: multilabel
    def encode_multi(self, values, sep: str = "|"):
        """Delimited label cells to a `(N, K)` float32 k-hot matrix in class order, refusing
        (and naming) any name the spec does not list."""
        index = {str(c): i for i, c in enumerate(self.classes)}
        out = np.zeros((len(values), len(self.classes)), dtype=np.float32)
        unknown: set = set()
        for row, cell in enumerate(values):
            for name in _split_labels(cell, sep):
                if name in index:
                    out[row, index[name]] = 1.0
                else:
                    unknown.add(name)
        if unknown:
            raise ValueError("the label column holds name(s) the spec's task.classes does not "
                             f"list: {sorted(unknown)[:5]}")
        return out
    # <<< task


@dataclass(frozen=True)
class ClassValue:
    """One class of the label, as a value. Built only through `LabelSet` (`of`, `at`), so an
    instance is by construction a member of the set the head was trained against: the set
    hands its classes in as a witness, and an instance built without it — or one whose name
    is not the witness's class at its index — is refused (equality stays name and index)."""

    name: str
    index: int
    _classes: tuple = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        classes = self._classes
        if not isinstance(classes, tuple) or not 0 <= self.index < len(classes) or \
                classes[self.index] != self.name:
            raise ValueError(f"ClassValue({self.name!r}, {self.index!r}) is built through "
                             "LabelSet.of(name) or LabelSet.at(index), which check it is a "
                             "class of the spec's task.classes")


# >>> compat: TrafficClass
#: the v2.2 name of `ClassValue`, kept for one release so a blueprint that names the artifact
#: that way still finds it (emitted only for such a blueprint); new code says `ClassValue`
TrafficClass = ClassValue
# <<< compat


@dataclass(frozen=True)
class ReadingProvenance:
    """Where a monitor reading comes from: the check that measured it (its id, the blueprint
    concern it answers, its title), the rows of the window it read, and what the measurement
    noted — typed, so a consumer reads fields and a renamed one fails where it is built."""

    check: str
    concern: str | None
    title: str | None
    window_rows: int
    note: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "window_rows", int(self.window_rows))


@dataclass(frozen=True)
class DriftReading:
    """One monitor reading: a column, a measure, a value and a verdict, for one window.

    The blueprint's Monitor reports and never gates, so a reading is a value with no
    behaviour — the verdict is a statement about the window, not an action on the detector.
    The measure and the verdict are closed sets (`DriftMeasure`, `DriftVerdict`): a reading
    built with any other word is refused here, not met by a consumer (EJ-34); where it comes
    from is a `ReadingProvenance`, never an association list.
    """

    column: str
    measure: DriftMeasure
    value: Any
    verdict: DriftVerdict
    window: ReadingProvenance | None = None

    def __post_init__(self):
        object.__setattr__(self, "measure", DriftMeasure(self.measure))
        object.__setattr__(self, "verdict", DriftVerdict(self.verdict))
        if self.window is not None and not isinstance(self.window, ReadingProvenance):
            raise TypeError("DriftReading.window is a ReadingProvenance (the check, its concern "
                            "and title, the window's rows, the note), never its parts")


@dataclass(frozen=True)
class Versions:
    """Which feature contract and which model produced an answer, and what it warns about."""

    contract: str | None = None
    model: str | None = None
    warnings: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "warnings", tuple(self.warnings or ()))


def _read_label(score, labels: LabelSet) -> ClassValue:
    """The default reading of raw head outputs (logits): K outputs over K classes → argmax;
    K−1 cumulative outputs → the count of logits ≥ 0 (a binary head is K = 2)."""
    values = _listify(score)
    values = [float(v) for v in (values if isinstance(values, (list, tuple)) else [values])]
    if len(values) == len(labels) and len(values) > 1:
        return labels.at(max(range(len(values)), key=values.__getitem__))
    if len(values) == len(labels) - 1:
        return labels.at(sum(1 for v in values if v >= 0.0))
    raise ValueError(f"{len(values)} output(s) cannot be read against {len(labels)} classes; "
                     "the head gives K logits or K−1 cumulative logits")


@dataclass(frozen=True)
class Prediction:
    """One scored row: the class and its probabilities — the predictions boundary's contract —
    with the raw head they were read from and the versions that produced them. No identity of
    its own: it belongs to the batch it came from.

    `probabilities` are what the reading's head means, in the spec's class order: one per
    class for a single-label reading (they sum to one, and `label`, when there is one, is one
    of their classes), one per label for a multilabel reading; none for a value reading. They
    are computed by the reading (`TaskReadout`) and handed in — this value validates them and
    never re-derives them. `score` stays exactly what the head produced (the round trip
    compares it); `label` is the reading's decision (its threshold, its task). The serving
    Predictor builds the value with its own reading's decision; `Prediction.of` is the default
    reading of the label from the logits.
    """

    row_id: Any
    score: Any
    label: ClassValue | None = None
    probabilities: tuple = ()
    versions: Versions = field(default_factory=Versions)

    def __post_init__(self):
        object.__setattr__(self, "score", _freeze(_listify(self.score)))
        if not isinstance(self.versions, Versions):
            raise TypeError("Prediction.versions is a Versions value (which contract and which "
                            "model produced the answer, and its warnings), never its parts")
        probs = tuple(float(p) for p in _listify(self.probabilities) or ())
        if any(not (0.0 <= p <= 1.0) for p in probs):
            raise ValueError(f"a prediction's probabilities lie in [0, 1]; got {list(probs)}")
        if probs and self.label is not None:
            classes = self.label._classes
            if len(probs) != len(classes) or abs(sum(probs) - 1.0) > 1e-4:
                raise ValueError(f"a single-label prediction carries one probability per class "
                                 f"of its {len(classes)}, summing to one; got {list(probs)}")
        object.__setattr__(self, "probabilities", probs)

    @property
    def contract_version(self) -> str | None:
        return self.versions.contract

    @property
    def model_version(self) -> str | None:
        return self.versions.model

    @property
    def warnings(self) -> tuple:
        return self.versions.warnings

    @staticmethod
    def of(row_id, score, labels: LabelSet | None = None, versions: Versions | None = None,
           probabilities=()) -> "Prediction":
        """The named factory (EJ-1): the class is read off the raw head outputs (logits, so a
        binary logit ≥ 0 is the positive class, probability 0.5) through the `LabelSet` the
        head was trained against, so the label is by construction a member of that set and
        never a bare string the caller assembled (EJ-62); the probabilities are the reading's
        (`TaskReadout.proba`), handed in. A multilabel answer has no single class: pass no
        labels."""
        label = _read_label(score, labels) if labels is not None and len(labels) else None
        return Prediction(row_id=row_id, score=score, label=label, probabilities=probabilities,
                          versions=versions or Versions())


# --------------------------------------------------------------------------- parts

@dataclass(frozen=True)
class SealReading:
    """What the seal /vega-eval wrote says about the one look it admitted: the campaign run
    and the test part's exposure count. Read once, at the DataSource's door that checked the
    token, and carried by the part that door opened — so the provenance the evaluation stamps
    is the seal that admitted the read, never a second read of a file that can change."""

    run: str | None = None
    exposures: int | None = None

    @staticmethod
    def of(record: Mapping[str, Any]) -> "SealReading":
        return SealReading(record.get("run"), int(record.get("exposures") or 0) or None)


@dataclass(frozen=True, eq=False)
class TensorPart:
    """One shaped part of rows as tensors: the numeric block, the nominal ids and the label.

    Deliberately NOT called `Dataset`: the blueprint's `Dataset` is an entity identified by
    the input digest, and two files with that digest are the same dataset. This is a view of
    one part of one of them, held in memory, with no identity beyond `digest` (the part
    file's sha256, when the DataSource checked it). `to(device)` is the resident form the
    Trainer's resident pass reads: moved once, then every minibatch is an `index_select` on
    the device, which a captured CUDA graph can contain.
    """

    x_num: Any
    x_cat: Any
    y: Any
    row_ids: Any = None
    columns: tuple = ()
    unscored: tuple = ()
    device: str | None = None
    digest: str | None = None
    seal: SealReading | None = None       # the test part's, from the door that opened it

    def __eq__(self, other):
        """The part's identity is its file's digest (the same shaped part, wherever it is
        held); a part with no digest is only itself. Never a comparison of tensor payloads."""
        if not isinstance(other, TensorPart):
            return NotImplemented
        return self is other or (self.digest is not None and self.digest == other.digest)

    def __hash__(self):
        return hash(self.digest) if self.digest is not None else id(self)

    def __len__(self):
        return int(self.x_num.shape[0]) if self.x_num is not None else int(len(self.y))

    def __getitem__(self, idx):
        out = {"x_num": self.x_num[idx] if self.x_num is not None else None,
               "x_cat": self.x_cat[idx] if self.x_cat is not None else None,
               "y": self.y[idx] if self.y is not None else None}
        return {k: v for k, v in out.items() if v is not None}

    def gather(self, idx) -> dict:
        """The rows at (device) indices `idx`, in that order, as the batch dict."""
        out = {BatchKey.X_NUM.value: _take(self.x_num, idx),
               BatchKey.X_CAT.value: _take(self.x_cat, idx),
               BatchKey.Y.value: _take(self.y, idx)}
        return {k: v for k, v in out.items() if v is not None}

    def slices(self, batch_size) -> Iterator[dict]:
        """The part in stored order, `batch_size` rows at a time — views, not copies."""
        n, bs = len(self), int(batch_size)
        for start in range(0, n, bs):
            yield self[slice(start, min(n, start + bs))]

    def to(self, device) -> "TensorPart":
        return replace(self, x_num=_move(self.x_num, device), x_cat=_move(self.x_cat, device),
                       y=_move(self.y, device), device=str(device))

    @property
    def row_id_kind(self) -> RowIdKind:
        return RowIdKind.RAW_POSITION


# >>> compat: ResidentPart
#: the v2.2 name of the resident part, folded into `TensorPart.to(device)`; kept one release
#: (emitted only for a blueprint that still names it)
ResidentPart = TensorPart
# <<< compat


# >>> shape: window
@dataclass(frozen=True, eq=False)
class WindowedPart:
    """One part of windows, held flat: `x_num (M, F')`, `x_cat (M, C)`, `y (M, ...)` are the
    part's rows (and the context rows before them, in `WindowLayout` order); each sample is
    an anchor. Windows are cut on the fly, `(B, L, F')` per batch, so memory is O(rows·F) and
    the gather is static-shaped (a resident CUDA-graph capture can hold it).
    Equality is identity: the payload is arrays, which compare entry by entry, so a
    generated `==` would raise rather than answer."""

    x_num: Any
    x_cat: Any
    y: Any
    anchors: Any
    targets: Any
    row_ids: Any
    length: int
    unscored: tuple = ()
    columns: tuple = ()
    seal: SealReading | None = None       # the test part's, from the door that opened it
    steps: Any = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        if len(self.targets) != len(self.anchors):
            raise ValueError(f"{len(self.targets)} target row(s) for {len(self.anchors)} window(s)")
        object.__setattr__(self, "steps", _steps_like(self.anchors, self.length))

    def __len__(self):
        return int(len(self.anchors))

    def gather(self, idx) -> dict:
        return self._batch(_take(self.anchors, idx), _take(self.targets, idx))

    def __getitem__(self, idx) -> dict:
        return self.gather(idx)

    def slices(self, batch_size) -> Iterator[dict]:
        n, bs = len(self), int(batch_size)
        for start in range(0, n, bs):
            stop = min(n, start + bs)
            yield self._batch(self.anchors[start:stop], self.targets[start:stop])

    def to(self, device) -> "WindowedPart":
        return replace(self, x_num=_move(self.x_num, device), x_cat=_move(self.x_cat, device),
                       y=_move(self.y, device), anchors=_move(self.anchors, device),
                       targets=_move(self.targets, device))

    @property
    def row_id_kind(self) -> RowIdKind:
        return RowIdKind.RAW_POSITION

    def _batch(self, anchors, targets) -> dict:
        pos = anchors[:, None] + self.steps[None, :]
        out = {BatchKey.X_NUM.value: _rows(self.x_num, pos)}
        if self.x_cat is not None:
            out[BatchKey.X_CAT.value] = _rows(self.x_cat, pos)
        if self.y is not None:
            y = _rows(self.y, targets)
            out[BatchKey.Y.value] = y[:, 0] if int(targets.shape[1]) == 1 else y
        return out


# <<< shape


# >>> shape: graph
@dataclass(frozen=True, eq=False)
class GraphPart:
    """One graph view as tensors. At node level the samples are the target nodes: every
    batch holds the whole view's graph and a slice of `target_index` (full-batch message
    passing, loss on the targets). At graph level the samples are graphs: a batch is the
    selected graphs' nodes and edges, relabelled, with `graph_index` and `num_graphs`.
    Equality is identity: the payload is arrays, which compare entry by entry, so a
    generated `==` would raise rather than answer."""

    x_num: Any
    x_cat: Any
    edge_index: Any
    edge_weight: Any
    target_index: Any
    y: Any
    row_ids: Any
    graph_index: Any = None
    num_graphs: int | None = None
    unscored: tuple = ()
    columns: tuple = ()
    exposure: GraphExposure | None = None
    seal: SealReading | None = None       # the test part's, from the door that opened it

    def __post_init__(self):
        if (self.graph_index is None) == (self.target_index is None):
            raise ValueError("a graph part has target_index (node level) or graph_index "
                             "(graph level), not both and not neither")
        if self.graph_index is not None and self.num_graphs is None:
            raise ValueError("a graph-level part states num_graphs")

    @property
    def level(self) -> GraphLevel:
        return GraphLevel.GRAPH if self.graph_index is not None else GraphLevel.NODE

    @property
    def row_id_kind(self) -> RowIdKind:
        return RowIdKind.GRAPH_ID if self.level is GraphLevel.GRAPH else RowIdKind.RAW_POSITION

    def __len__(self):
        return int(self.num_graphs) if self.level is GraphLevel.GRAPH else len(self.target_index)

    def as_batch(self) -> dict:
        if self.level is GraphLevel.GRAPH:
            return self.gather(_arange_like(self.graph_index, int(self.num_graphs)))
        return self._node_batch(self.target_index, self.y)

    def gather(self, idx) -> dict:
        if self.level is GraphLevel.GRAPH:
            return self._graph_batch(idx)
        return self._node_batch(_take(self.target_index, idx), _take(self.y, idx))

    def __getitem__(self, idx) -> dict:
        return self.gather(idx)

    def slices(self, batch_size) -> Iterator[dict]:
        n, bs = len(self), int(batch_size)
        ref = self.graph_index if self.level is GraphLevel.GRAPH else self.target_index
        for start in range(0, n, bs):
            yield self.gather(_arange_like(ref, min(n, start + bs) - start) + start)

    def to(self, device) -> "GraphPart":
        return replace(self, x_num=_move(self.x_num, device), x_cat=_move(self.x_cat, device),
                       edge_index=_move(self.edge_index, device),
                       edge_weight=_move(self.edge_weight, device),
                       target_index=_move(self.target_index, device), y=_move(self.y, device),
                       graph_index=_move(self.graph_index, device))

    def _node_batch(self, target_index, y) -> dict:
        out = {BatchKey.X_NUM.value: self.x_num, BatchKey.X_CAT.value: self.x_cat,
               BatchKey.EDGE_INDEX.value: self.edge_index,
               BatchKey.EDGE_WEIGHT.value: self.edge_weight,
               BatchKey.TARGET_INDEX.value: target_index, BatchKey.Y.value: y}
        return {k: v for k, v in out.items() if v is not None}

    def _graph_batch(self, idx) -> dict:
        slot = _full_like(self.graph_index, int(self.num_graphs), -1)
        slot[idx] = _arange_like(self.graph_index, len(idx))
        node_graph = slot[self.graph_index]
        keep = node_graph >= 0
        nodes = _nonzero(keep)
        position = (keep * 1).cumsum(0) - 1
        edges = _nonzero(keep[self.edge_index[0]])
        out = {BatchKey.X_NUM.value: _take(self.x_num, nodes),
               BatchKey.X_CAT.value: _take(self.x_cat, nodes),
               BatchKey.EDGE_INDEX.value: position[self.edge_index[:, edges]],
               BatchKey.EDGE_WEIGHT.value: _take(self.edge_weight, edges),
               BatchKey.GRAPH_INDEX.value: node_graph[nodes],
               BatchKey.NUM_GRAPHS.value: int(len(idx)), BatchKey.Y.value: _take(self.y, idx)}
        return {k: v for k, v in out.items() if v is not None}
# <<< shape


_SAMPLE_PARTS: tuple = ()
# >>> shape: row
_SAMPLE_PARTS += (TensorPart,)
# <<< shape
# >>> shape: window
_SAMPLE_PARTS += (WindowedPart,)
# <<< shape
# >>> shape: graph
_SAMPLE_PARTS += (GraphPart,)
# <<< shape
#: the part `load_part` returns: the part of the sample shape this package carries
SamplePart = Union[_SAMPLE_PARTS]


# --------------------------------------------------------------------------- parameter objects

@dataclass(frozen=True)
class BatchPlan:
    """How `iter_batches` orders a streamed part: the shuffle, its seed and its buffer.
    `buffer_rows` is the one default of the buffer — the DataSource shuffles with it and the
    epoch source records it, so the record cannot describe a shuffle that did not happen."""

    shuffle: bool = False
    seed: int | None = None
    shuffle_buffer: int | None = None

    #: a streamed shuffle holds this many batches when `shuffle_buffer` is not set
    BUFFER_BATCHES = 64

    def buffer_rows(self, batch_size: int) -> int:
        """The rows one shuffle holds: `shuffle_buffer`, else `BUFFER_BATCHES` batches."""
        return int(self.shuffle_buffer or int(batch_size) * self.BUFFER_BATCHES)

    def rng(self, default_seed: int):
        """The shuffle's generator: `seed`, else the spec's `default_seed` — one definition for
        the stream that shuffles a buffer and the source that holds a part one buffer holds."""
        return np.random.default_rng(int(self.seed if self.seed is not None else default_seed))


@dataclass(frozen=True)
class RunContext:
    """What one fit needs beyond the model, the spec and the data: the wall-clock budget,
    where its checkpoints go, and the gradient sampling rate. The ModelStore that writes the
    checkpoints is the Trainer's collaborator, given to it, not a field of a run's value."""

    budget_min: float | None = None
    checkpoint_dir: str | None = None
    grad_sample_every: int = 0


@dataclass(frozen=True)
class BaselineContext:
    """What a baseline needs beyond the spec: the bottleneck width, the protocol record, the
    season (seasonal-naive, mase) and the forecast horizon."""

    z: int | None = None
    protocol: Any = None
    season: int | None = None
    horizon: int = 1

    def __post_init__(self):
        object.__setattr__(self, "protocol", _freeze(self.protocol))

    def protocol_dict(self) -> dict:
        return dict(_thaw(self.protocol) or {})


@dataclass(frozen=True, eq=False)
class DesignMatrix:
    """A part as the flat matrix a baseline reads (`DesignView`), with its truth and ids.
    Equality is identity: the payload is arrays, which compare entry by entry, so a
    generated `==` would raise rather than answer.

    `rows_of(batch)` is the design's one layout — `[x_num | x_cat]`, each sample flattened (a
    window: L·F′ + L·C) — so a baseline is fitted on (`of_part`, the DataSource's) and scored
    on (a batch, the baseline's) the same columns by construction, not by two copies kept
    equal."""

    x: Any
    y: Any
    row_ids: Any

    @staticmethod
    def rows_of(batch: Mapping) -> np.ndarray:
        """One batch's samples as design rows, float64 (tensors on any device, or arrays)."""
        blocks = [host_array(batch["x_num"])]
        if batch.get("x_cat") is not None:
            blocks.append(host_array(batch["x_cat"]))
        return np.concatenate([b.reshape(len(b), -1).astype(np.float64) for b in blocks],
                              axis=1)

    @staticmethod
    def of_part(part, rows: int = 4096) -> "DesignMatrix":
        """A held row or window part, flat, float32: `rows` samples at a time, so a window
        part is never expanded whole."""
        xs, ys = [], []
        for batch in part.slices(rows):
            xs.append(DesignMatrix.rows_of(batch).astype(np.float32))
            ys.append(host_array(batch.get("y")))
        y = None if not ys or any(v is None for v in ys) else np.concatenate(ys)
        x = np.concatenate(xs) if xs else np.zeros((0, 0), dtype=np.float32)
        return DesignMatrix(x=x, y=y, row_ids=np.asarray(part.row_ids))
    # >>> shape: graph

    @staticmethod
    def neighbour_mean(x, edge_index) -> np.ndarray:
        """Row i: the mean of `x` over the sources of the edges into i (zeros without any) —
        the second half of a node's neighbour-mean design row."""
        src, dst = (np.asarray(host_array(e), dtype=np.int64) for e in edge_index)
        x = np.asarray(x)
        total = np.zeros_like(x)
        np.add.at(total, dst, x[src])
        count = np.maximum(np.bincount(dst, minlength=len(x)), 1).astype(x.dtype)
        return total / count[:, None]

    @staticmethod
    def pooled(x, graph_index, num_graphs: int) -> np.ndarray:
        """`[mean | max | sum]` of each graph's node rows, graphs in index order — a graph's
        design row at graph level."""
        graph_index = np.asarray(host_array(graph_index), dtype=np.int64)
        order = np.argsort(graph_index, kind="stable")
        counts = np.bincount(graph_index, minlength=num_graphs)
        if (counts == 0).any():
            raise ValueError(f"graph {int(np.argmin(counts))} of the part has no node")
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        rows = np.asarray(x)[order]
        total = np.add.reduceat(rows, starts, axis=0)
        return np.concatenate([total / counts[:, None], np.maximum.reduceat(rows, starts, axis=0),
                               total], axis=1)
    # <<< shape


@dataclass(frozen=True, eq=False)
class ScoredPart:
    """A part scored by one model, as every consumer of predictions reads it: the score per
    the task's score contract (raw head outputs), the truth, the row ids and their kind, the
    rows that got no prediction, the per-row score of the reconstruction and anomaly
    readings, and the graph ids at graph level.
    Equality is identity: the payload is arrays, which compare entry by entry, so a
    generated `==` would raise rather than answer."""

    score: Any
    truth: Any
    row_ids: Any
    row_id_kind: RowIdKind = RowIdKind.RAW_POSITION
    unscored: tuple = ()
    row_score: Any = None
    graph_ids: Any = None

    def __post_init__(self):
        object.__setattr__(self, "row_id_kind", RowIdKind(self.row_id_kind))
        object.__setattr__(self, "unscored", tuple(self.unscored or ()))
        n = len(self.row_ids)
        for name in ("score", "truth", "row_score", "graph_ids"):
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(f"{len(values)} {name} value(s) for {n} row id(s)")

    def __len__(self):
        return len(self.row_ids)


# --------------------------------------------------------------------------- training records

@dataclass(frozen=True)
class EpochRecord:
    """One line of history.jsonl (backend-contract §1). Written by the Trainer, read by the
    training layer, the diagnostics catalog and the notebooks."""

    epoch: int
    time_s: float
    lr: float
    train: _FrozenMap
    valid: _FrozenMap
    grad_norm: _FrozenMap = field(default_factory=_FrozenMap)
    weight_norm: _FrozenMap = field(default_factory=_FrozenMap)
    update_ratio: _FrozenMap = field(default_factory=_FrozenMap)
    nan: bool = False
    early_stopping: _FrozenMap = field(default_factory=_FrozenMap)
    shuffle: _FrozenMap = field(default_factory=_FrozenMap)

    @staticmethod
    def of(fields: Mapping[str, Any]) -> "EpochRecord":
        """The record from one mapping of the contract's fields; an unknown field is refused
        by name (a misspelt key would otherwise vanish from the history)."""
        unknown = sorted(set(fields) - set(_EPOCH_SCALARS) - set(_EPOCH_MAPS))
        if unknown:
            raise ValueError(f"an epoch record has no field(s) {unknown}; the fields are "
                             f"{list(_EPOCH_SCALARS + _EPOCH_MAPS)}")
        if fields.get("epoch") is None:
            raise ValueError("an epoch record needs its epoch")
        frozen = {f: _freeze(fields.get(f) or {}) for f in _EPOCH_MAPS}
        return EpochRecord(epoch=int(fields["epoch"]), time_s=float(fields.get("time_s") or 0.0),
                           lr=float(fields.get("lr") or 0.0), nan=bool(fields.get("nan")),
                           **frozen)

    def as_dict(self) -> dict:
        return {"epoch": self.epoch, "time_s": self.time_s, "lr": self.lr,
                "train": _thaw(self.train), "valid": _thaw(self.valid),
                "grad_norm": _thaw(self.grad_norm), "weight_norm": _thaw(self.weight_norm),
                "update_ratio": _thaw(self.update_ratio), "nan": self.nan,
                "early_stopping": _thaw(self.early_stopping),
                "shuffle": _thaw(self.shuffle)}


@dataclass(frozen=True)
class FitResult:
    """What the Trainer returns: where the best checkpoint is and what ended the run."""

    checkpoint: Checkpoint | None
    best_epoch: int | None
    best_value: float | None
    epochs_run: int
    stopped_by: StopReason
    history_path: str | None = None
    #: the valid part's metrics at the best epoch — the numbers selection compares, returned
    #: as the blueprint's `Metrics` value rather than left for the caller to re-derive
    best_valid: "Metrics | None" = None
    #: the intra-op thread count a CPU fit ran on (its substrate posture, part of what the
    #: run is: a float reduction is split by it); None on another device
    threads: int | None = None


# --------------------------------------------------------------------------- the network, viewed

@dataclass(frozen=True, eq=False)
class NetworkAnatomy:
    """The Model port's `anatomy()`: a read-only view of one live staged network — the way the
    serving and inspection code read the network beside the port's methods (code-contract §2).

    A view, not a value: its identity is the network it views (two views of one network are
    equal), and the state it reports — `training`, `device`, the one-class `centre` and
    whether the loss wrote it (`centre_set`) — is the network's when asked, never a snapshot.
    What the network *is* was fixed when it was built, and the view reads it from the network:
    the stages, aligned one to one with the spec's blocks (`ids`; `kinds`, the `StageKind` of
    the call the forward makes; `types`; `shapes`, per stage, the batch axis dropped), the
    reading (`readout`, the spec's `TaskReadout`), the `sample` and the spec's identity
    (`spec_digest`).

    How the inspector reaches the network: `stages`, the block modules, read through the
    inspection surface their type declares (R-B3), and `forward(batch)`, the network's own pass
    over an assembled batch with gradients as the caller enables them — it fires the stage
    modules' hooks and never switches the mode. The view writes nothing on the network, and
    its readers are held to the same rule: a served network is never mutated by a request.
    The network is duck-typed (the domain imports no framework): it carries `ids`, `kinds`,
    `types`, `shapes`, `stages`, `readout`, `sample`, `spec_digest`, `training`, `device`, and
    under the svdd objective the buffers `svdd_center` / `svdd_center_set`.
    """

    _network: Any

    def __post_init__(self):
        widths = {name: len(getattr(self._network, name))
                  for name in ("ids", "kinds", "types", "shapes", "stages")}
        if len(set(widths.values())) != 1:
            raise ValueError(f"a network's stages are aligned one per block; got {widths}")

    def __eq__(self, other):
        if not isinstance(other, NetworkAnatomy):
            return NotImplemented
        return self._network is other._network

    def __hash__(self):
        return id(self._network)

    def __repr__(self) -> str:
        return (f"NetworkAnatomy(spec_digest={self.spec_digest!r}, stages={len(self.ids)}, "
                f"training={self.training!r})")

    @property
    def spec_digest(self) -> str:
        return self._network.spec_digest

    @property
    def ids(self) -> tuple:
        return tuple(self._network.ids)

    @property
    def kinds(self) -> tuple:
        return tuple(StageKind(k) for k in self._network.kinds)

    @property
    def types(self) -> tuple:
        return tuple(self._network.types)

    @property
    def shapes(self) -> tuple:
        return tuple(self._network.shapes)

    @property
    def stages(self) -> tuple:
        return tuple(self._network.stages)

    @property
    def readout(self):
        return self._network.readout

    @property
    def sample(self) -> "SampleSpec":
        return self._network.sample

    @property
    def training(self) -> bool:
        """The network's mode now (a served network is in eval mode)."""
        return bool(self._network.training)

    @property
    def device(self):
        """Where the network's tensors are now."""
        return self._network.device

    @property
    def centre(self):
        """The one-class centre a svdd reading measures from (the buffer), else None."""
        return getattr(self._network, "svdd_center", None)

    @property
    def centre_set(self) -> bool:
        """Whether the loss wrote the centre (a svdd network's), else False."""
        return self.centre is not None and bool(self._network.svdd_center_set)

    def objective_center(self):
        """The one-class centre a svdd reading measures from (`TaskReadout`), else None."""
        return self.centre

    def forward(self, batch):
        """The network's own pass over an assembled batch (its hooks fire; its mode is
        untouched; gradients as the caller enables them)."""
        return self._network(batch)


class UnsupportedShapeError(ValueError):
    """Raised for a spec whose sample shape this package does not carry (`CARRIED_SHAPES`).

    /vega-forge emits a package for its spec's sample shape: the code of the other shapes, and
    their support files, are not in it. A spec of another shape is refused where a dict becomes
    a spec, naming the shape and the ones the package carries; the fix is a package emitted
    for that spec, never a module copied in beside this one.
    """

    def __init__(self, shape):
        self.shape = SampleShape(shape)
        carried = ", ".join(s.value for s in SampleShape if s in CARRIED_SHAPES)
        super().__init__(
            f"the spec's sample shape is {self.shape.value!r}, and this package carries the "
            f"{carried} sample shape only: /vega-forge emits the code of one sample shape, its "
            "spec's (data_contract.sample.shape) — emit a package for this spec with "
            "/vega-forge rather than run it on this one")


class SealedPartError(RuntimeError):
    """Raised by the DataSource adapter for any request for the test part.

    The test part belongs to /vega-eval, which opens it once through `load_sealed_part` with
    the token only its seal file carries.

    This type lives in `domain` because `domain` is the module every adapter may import and
    that imports nothing, so the exception can be raised at any boundary and caught anywhere
    without an arrow the module map forbids. What makes the seal a property of the code rather
    than of anyone's discipline is where it is *raised*: the DataSource adapter's `part_path`
    and `load_part` raise it before a file is opened, and `load_sealed_part` — the one door —
    demands a token from a seal file that does not exist until the evaluation writes it.
    A graph spec whose protocol would read the test features without a declared exposure
    raises it too, at construction.
    """
