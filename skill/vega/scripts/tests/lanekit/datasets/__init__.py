"""
The dataset generators of the lane program (D6 §4), discovered from this package.

A generator is one module `lanekit/datasets/<name>.py`; `<name>` is what a lane's `generator`
names. It exposes one function and three constants:

    def make(seed: int, rows: int) -> pandas.DataFrame   # the RAW table: own dtypes, own scales
    SCHEMA: tuple[ColumnSpec, ...]                        # every raw column, in order
    RECIPE: Callable[[Path, Path, Lane], dict]            # (raw_path, shaped_dir, lane) -> dataset-shaper/recipe@1
    VOCAB: frozenset[str]                                 # the domain words the dataset brings

`make` is deterministic in `(seed, rows)` and plants a learnable signal. `RECIPE` returns the
recipe `shape.py` executes; most generators return `lanekit.recipes.standard_recipe(SCHEMA,
raw_path, shaped_dir, lane)`, which builds the recipe every lane shares from the schema's roles
and the lane's `recipe_options`. `VOCAB` is what the literal lint and the leak checks read.

A generator may also carry the words its base model is written in: `ENTITY_COMMENT` (one
sentence about one row), `COMPONENT` (the entity's component) and `OVERVIEW` (the model's prose
summary). lanekit.fixture writes neutral ones from the lane when they are absent.

`GENERATORS` maps every generator name to its module, imported on first use and checked
against the protocol above: a module that does not keep it is refused by name, not used.
Discovery is by file (`*.py` here, `_`-prefixed files excluded) — the same rule the registry's
validation applies — so adding a generator is adding a file.

`PLANNED` names the generators BUILD-PLAN §2.9 schedules and the work package that provides
each (T05 rows, T06 windows, T07 graphs); `pending()` lists the planned ones not in the tree
yet, so a reader can tell "not written yet" from "unknown".
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

__all__ = ["ColumnSpec", "GENERATORS", "PLANNED", "ROLES", "DTYPES", "GeneratorError",
           "names", "load", "pending", "check_schema"]

HERE = Path(__file__).resolve().parent

# A column's role says what the shaper does with it (lanekit.recipes.standard_recipe):
#   numeric  a quantity: scaled on train
#   nominal  a category: retyped nominal, then encoded (ordinal when the network embeds it or a
#            feature_tokenizer reads it, one-hot otherwise; recipe_options.encode decides first)
#   time     a timestamp: parsed; the time split's column; a window lane's order_by key
#   key      an identifier: dropped before any fitted step (a graph's key_pair reads it raw)
#   label    a label candidate: the lane's target is selected, every other one is leakage
#   drop     a column the analysis found leaking or empty: dropped by name
#   series   a window's series key: kept, never a feature (windows never cross it)
#   group    a graph's graph_id: kept, the group split's column (graphs never straddle parts)
ROLES = ("numeric", "nominal", "time", "key", "label", "drop", "series", "group")
DTYPES = ("float", "int", "str", "bool", "datetime")
PROTOCOL = ("make", "SCHEMA", "RECIPE", "VOCAB")

# BUILD-PLAN §2.9: the generators the lanes of §4.8 name, and the package that writes each
PLANNED: Mapping[str, str] = {
    "landcover": "T05", "tickets": "T05", "wine": "T05", "housing": "T05", "meters": "T05",
    "turbine": "T05", "bikeshare": "T06", "vibration": "T06", "tank": "T06", "grid": "T07",
    "papers": "T07", "roads": "T07", "molecules": "T07"}


class GeneratorError(LookupError):
    """A generator that does not exist or does not keep the protocol."""


@dataclass(frozen=True)
class ColumnSpec:
    """One raw column of a generator: its name, dtype, role, unit and a one-line comment."""
    name: str
    dtype: str
    role: str
    unit: str | None = None
    comment: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"a column needs a name, got {self.name!r}")
        if self.dtype not in DTYPES:
            raise ValueError(f"column {self.name!r}: dtype {self.dtype!r} is not one of {DTYPES}")
        if self.role not in ROLES:
            raise ValueError(f"column {self.name!r}: role {self.role!r} is not one of {ROLES}")


def check_schema(schema: Any) -> list[str]:
    """What is wrong with a SCHEMA: not ColumnSpecs, a duplicate name, or no feature at all."""
    if not isinstance(schema, tuple) or not all(isinstance(c, ColumnSpec) for c in schema):
        return ["SCHEMA must be a tuple of ColumnSpec"]
    names_ = [c.name for c in schema]
    problems = [f"column {n!r} is declared twice" for n in sorted({n for n in names_ if names_.count(n) > 1})]
    if not any(c.role in ("numeric", "nominal") for c in schema):
        problems.append("SCHEMA declares no feature column (numeric or nominal)")
    return problems


def names() -> tuple[str, ...]:
    """The generator names in this package, sorted: every `*.py` not `_`-prefixed."""
    return tuple(sorted(info.name for info in pkgutil.iter_modules([str(HERE)])
                        if not info.name.startswith("_") and not info.ispkg))


def load(name: str) -> Any:
    """The generator module `name`, imported and checked against the protocol."""
    if name not in names():
        planned = f" (planned: {PLANNED[name]} provides it)" if name in PLANNED else ""
        raise GeneratorError(f"no generator {name!r} in lanekit.datasets{planned}; "
                             f"the package holds {list(names()) or 'none yet'}")
    module = importlib.import_module(f"{__name__}.{name}")
    missing = [attr for attr in PROTOCOL if not hasattr(module, attr)]
    if missing:
        raise GeneratorError(f"generator {name!r} lacks {', '.join(missing)} "
                             "(make, SCHEMA, RECIPE, VOCAB: lanekit.datasets)")
    if not callable(module.make) or not callable(module.RECIPE):
        raise GeneratorError(f"generator {name!r}: make and RECIPE must be callable")
    problems = check_schema(module.SCHEMA)
    if problems:
        raise GeneratorError(f"generator {name!r}: " + "; ".join(problems))
    if not isinstance(module.VOCAB, frozenset):
        raise GeneratorError(f"generator {name!r}: VOCAB must be a frozenset of words")
    return module


def pending() -> dict[str, str]:
    """{name: providing work package} of the planned generators not in the tree yet."""
    present = set(names())
    return {name: wp for name, wp in PLANNED.items() if name not in present}


class _Generators(Mapping[str, Any]):
    """name -> generator module, discovered by file and imported on first use."""

    def __getitem__(self, name: str) -> Any:
        try:
            return load(name)
        except GeneratorError as exc:
            raise KeyError(str(exc)) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(names())

    def __len__(self) -> int:
        return len(names())

    def __contains__(self, name: object) -> bool:
        return name in names()


GENERATORS: Mapping[str, Any] = _Generators()
