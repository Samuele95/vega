"""
The lane registry: `lanes.json` (`vega/test-lanes@1`) read into frozen `Lane` values (D6 §3.1,
BUILD-PLAN §4.8), and the catalogs a lane is checked against.

    lanes = load_lanes()                                    # {id: Lane}
    report = validate(doc, Catalogs.load(skills_root))      # lanekit.validation
    affected_lanes(lanes, changed_paths, repo, rev)         # lanekit.routing (--changed)

A `Lane` is what the registry declares; nothing in it is an observation. What a lane proved is
read from its artifacts (lanekit.observe).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import contracts

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
LANES_FILE = TESTS / "lanes.json"
LANES_SCHEMA = TESTS / "lanes.schema.json"
FLAKES_FILE = HERE / "flakes.json"
STAGE_ORDER = contracts.STAGES
TIERS = ("pr", "release", "nightly")


# --------------------------------------------------------------------------- values
@dataclass(frozen=True)
class Lane:
    """One lane of the registry, as declared. Nothing in it is an observation."""
    id: str
    raw: Mapping[str, Any] = field(repr=False, compare=False, hash=False)

    def __getattr__(self, name: str) -> Any:        # the declared fields, read-only
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return self.raw[name]
        except KeyError as exc:
            raise AttributeError(f"lane {self.id} declares no {name!r}") from exc

    def get(self, name: str, default: Any = None) -> Any:
        return self.raw.get(name, default)

    @property
    def driver(self) -> str:
        return self.raw.get("driver", "lanekit")

    @property
    def expect(self) -> Mapping[str, Any]:
        return self.raw.get("expect") or {}

    @property
    def target(self) -> Mapping[str, Any]:
        return self.raw.get("target") or {}

    @property
    def classes(self) -> tuple[str, ...] | None:
        classes = self.target.get("classes")
        return tuple(str(c) for c in classes) if classes else None

    @property
    def graph_level(self) -> str | None:
        return (self.raw.get("graph") or {}).get("level")

    @property
    def last_stage(self) -> str:
        return self.expect.get("last_stage") or "page"

    @property
    def modality(self) -> str:
        """The motif modality (R-E4): spatial-row for a row lane that declares spatial data."""
        if self.raw["shape"] == "row" and self.expect.get("spatial"):
            return "spatial-row"
        return self.raw["shape"]

    def flat_block_types(self) -> list[str]:
        return flatten_types(self.raw.get("blocks") or [])

    def canonical(self) -> str:
        return json.dumps(self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def flatten_types(blocks: Sequence[Mapping[str, Any]]) -> list[str]:
    """The flattened type list of a block list, composites' inner blocks included once (D6
    §6.13: the observed `blocks`)."""
    out: list[str] = []
    for block in blocks:
        out.append(block["type"])
        out.extend(flatten_types((block.get("params") or {}).get("inner") or []))
    return out


def load_document(path: Path = LANES_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_lanes(path: Path = LANES_FILE) -> dict[str, Lane]:
    doc = load_document(path)
    return {entry["id"]: Lane(entry["id"], entry) for entry in doc["lanes"]}


def lanes_for(lanes: Mapping[str, Lane], ids: Iterable[str] | None = None,
              tier: str | None = None) -> list[Lane]:
    """The lanes a run names: explicit ids (in registry order), else a tier's members."""
    if ids:
        wanted = [i.strip() for i in ids if i.strip()]
        unknown = [i for i in wanted if i not in lanes]
        if unknown:
            raise KeyError(f"unknown lane(s): {', '.join(unknown)}")
        return [lanes[i] for i in lanes if i in wanted]
    return [lane for lane in lanes.values() if tier is None or tier in lane.tiers]


# --------------------------------------------------------------------------- catalogs
@dataclass(frozen=True)
class Catalogs:
    """The catalogs the registry is checked against, read from a skills root."""
    forge: Mapping[str, Any]
    coverage: Mapping[str, Any]
    show: Mapping[str, Any] | None
    generators: frozenset[str] | None

    @classmethod
    def load(cls, skills: Path) -> "Catalogs":
        forge = json.loads((skills / contracts.FORGE_CATALOG).read_text(encoding="utf-8"))
        coverage = json.loads((skills / contracts.BACKEND_COVERAGE).read_text(encoding="utf-8"))
        show_path = skills / contracts.SHOW_CATALOG
        show = json.loads(show_path.read_text(encoding="utf-8")) if show_path.is_file() else None
        return cls(forge, coverage, show, _generators(skills))


def _generators(skills: Path) -> frozenset[str] | None:
    """The generator names `lanekit.datasets` provides (T02), plus the external ones; None
    while the datasets package does not exist."""
    pkg = skills / contracts.LANEKIT_DIR / "datasets"
    if not (pkg / "__init__.py").is_file():
        return None
    names = {p.stem for p in pkg.glob("*.py") if not p.stem.startswith("_")}
    return frozenset(names | set(contracts.EXTERNAL_GENERATORS))


def choice_name(value: Any) -> str | None:
    """An optimizer / schedule / init choice is a name or {name, params}."""
    if value is None or isinstance(value, str):
        return value
    return value.get("name")


def required_lexicon(show_catalog: Mapping | None, modality: str) -> tuple[str, ...]:
    """The required lexicon keys: the vega-show catalog's `lexicon.required` when it exists
    in a form this reads (graph levels keyed `graph:node`, `graph-node` or `graph_node`), else
    the contract's copy of R-E6."""
    table = ((show_catalog or {}).get("lexicon") or {}).get("required")
    if isinstance(table, Mapping):
        for key in (modality, modality.replace("-", ":"), modality.replace("-", "_")):
            if isinstance(table.get(key), list):
                return tuple(table[key])
    return contracts.LEXICON_REQUIRED[modality]


def lexicon_articles(show_catalog: Mapping | None, key: str) -> tuple[str, ...]:
    """The articles a lexicon key may take: the vega-show catalog's `lexicon.articles` (a
    `default` list and per-key lists) when it exists, else the contract's copy of R-E6."""
    table = ((show_catalog or {}).get("lexicon") or {}).get("articles")
    if not isinstance(table, Mapping):
        table = contracts.LEXICON_ARTICLES
    chosen = table.get(key, table.get("default"))
    return tuple(chosen or ())


def motifs_by_modality(show_catalog: Mapping | None) -> Mapping[str, Mapping]:
    table = ((show_catalog or {}).get("motifs") or {}).get("by_modality")
    return table if isinstance(table, Mapping) else contracts.MOTIFS_BY_MODALITY


def validate(doc: Mapping[str, Any], catalogs: Catalogs, flakes: Mapping | None = None) -> Any:
    """The registry against its schema, the catalogs and itself (lanekit.validation)."""
    from .validation import validate as _validate
    return _validate(doc, catalogs, flakes)


def affected_lanes(lanes: Mapping[str, Lane], changed: Iterable[str], repo: Path,
                   rev_range: str | None = None) -> dict[str, list[str]]:
    """{lane id: [the changed paths that route it]} (lanekit.routing, D6 §3.4)."""
    from .routing import affected_lanes as _affected
    return _affected(lanes, changed, repo, rev_range)
