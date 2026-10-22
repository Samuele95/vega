"""
vega_chain — the chain around a spec, read as values: the model's layers, the shaped manifest,
the blueprint's pipeline, and the closed set of `source` values resolved against them.

Split from `vega_core.py` (which re-exports everything here, so callers keep one import):
`ChainInputs` is the one port-shaped value a compiler rule reads the chain through (HEX,
EJ-20), and `SourceResolver` answers whether a `source` is well formed and whether it
resolves — two questions that deserve different sentences. Pure: nothing here prints, exits or
writes; the only reading is of values already in memory.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:                       # vega_core re-exports this module; no import cycle
    from vega_core import Catalog

__all__ = ["ChainInputs", "SourceResolver", "model_layers", "dig", "find_in_list",
           "canonical_digest"]

_LAYER_DATA_RE = re.compile(
    r'<script\s+id="layer-([A-Za-z0-9_-]+)-data"[^>]*>([\s\S]*?)</script>')


def model_layers(html: str) -> dict:
    """{layer name: parsed data} for every `layer-<name>-data` script in a domain-forge model.

    A layer whose JSON does not parse maps to None rather than being dropped: "the analysis
    layer is corrupt" and "there is no analysis layer" are different failures and a source
    that cites one should not be told the other.
    """
    out = {}
    for m in _LAYER_DATA_RE.finditer(html or ""):
        try:
            out[m.group(1)] = json.loads(m.group(2))
        except json.JSONDecodeError:
            out[m.group(1)] = None
    return out


def dig(obj, path, sep="/"):
    """Walk a slash path through nested dicts and lists; None when it does not resolve.

    A list segment matches by index or by an element's `id` / `name` / `label` / `key`, which
    is how the layers address their own elements.
    """
    cur = obj
    for part in [p for p in str(path).split(sep) if p != ""]:
        if isinstance(cur, Mapping):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, Sequence) and not isinstance(cur, (str, bytes)):
            if part.isdigit() and int(part) < len(cur):
                cur = cur[int(part)]
                continue
            hit = next((x for x in cur if isinstance(x, Mapping) and part in
                        (x.get("id"), x.get("name"), x.get("label"), x.get("key"))), None)
            if hit is None:
                return None
            cur = hit
        else:
            return None
    return cur


def canonical_digest(doc) -> str:
    """`sha256:<hex>` of a JSON document in canonical form: keys sorted, no whitespace
    (`separators=(",", ":")`), UTF-8 with non-ASCII escaped. It is the emitter's digest of
    `pipeline.json` (`blueprint-map.json.pipeline_digest`) and so the one definition of the
    spec's `input.pipeline_digest` (netspec-contract §1) — a digest of the parsed document, so
    re-indenting the file does not change it and editing a value does."""
    return "sha256:" + hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def find_in_list(items, key, value):
    return next((x for x in (items or []) if isinstance(x, Mapping) and x.get(key) == value),
                None)


# --------------------------------------------------------------------------- the chain

@dataclass(frozen=True)
class ChainInputs:
    """Everything upstream a rule may consult, behind one value.

    A rule takes `ChainInputs` and never a path, so the same rule runs against a fixture, a
    test double or a real chain — the hexagonal seam of this script (HEX, EJ-20). Any of the
    three may be absent; a rule that needs one says so in its own marker rather than finding
    a None halfway through.
    """

    layers: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, Any] | None = None
    pipeline: Mapping[str, Any] | None = None
    stances: tuple = field(default_factory=tuple)

    @staticmethod
    def of(model_html=None, manifest=None, pipeline=None, stances=()) -> "ChainInputs":
        return ChainInputs(layers=model_layers(model_html or ""), manifest=manifest,
                           pipeline=pipeline, stances=tuple(stances or ()))

    # -- what the rules ask

    def layer(self, name):
        return self.layers.get(name)

    def manifest_steps(self) -> tuple:
        return tuple((self.manifest or {}).get("steps") or ())

    def encode_step_for(self, column):
        """The encode step that produced this column's codes.

        An executed manifest records what the step *did* (`parameters_fitted.fitted` is keyed
        by column) rather than what it was asked to do, so a step's own `columns` list may not
        be there to match on.
        """
        def touches(step):
            if column in (step.get("columns") or []):
                return True
            fitted = (step.get("parameters_fitted") or {}).get("fitted")
            return isinstance(fitted, Mapping) and column in fitted

        return next((s for s in self.manifest_steps()
                     if s.get("op") == "encode" and touches(s)), None)

    def unseen_category_id(self, column):
        """The code a category the training data never saw maps to, for one column.

        An executed manifest records the encode step's fitted parameters per column
        (`parameters_fitted.fitted.<column>.unseen_category_id`); a hand-written one put a
        single id on the step. Reading both here is what keeps the embedding-width rule an
        error rather than the warning it silently became when only the flat shape was
        understood — and a warning is not a gate.
        """
        step = self.encode_step_for(column)
        if not step:
            return None
        fitted = step.get("parameters_fitted") or {}
        per_column = fitted.get("fitted")
        if isinstance(per_column, Mapping):
            entry = per_column.get(column)
            if isinstance(entry, Mapping) and entry.get("unseen_category_id") is not None:
                return entry["unseen_category_id"]
        return fitted.get("unseen_category_id")

    def output_columns(self) -> tuple | None:
        """The shaped columns, in order, however the shaper wrote them down.

        `shape.py` writes `output_schema` as a mapping keyed by column name; earlier hand-made
        manifests used a `columns` list. Both are read here rather than in each caller,
        because which of the two a manifest carries is the shaper's business and not a
        distinction anything downstream should have to know about.
        """
        node = dig(self.manifest, "output_schema")
        if isinstance(node, Mapping):
            if isinstance(node.get("columns"), (list, tuple)):
                node = node["columns"]
            else:                       # {name: {dtype, role, origin}, …}
                return tuple(node.keys())
        if not isinstance(node, (list, tuple)):
            return None
        return tuple(c.get("name") if isinstance(c, Mapping) else c for c in node)

    def split_kind(self):
        """The split's kind, wherever the manifest records it.

        An executed manifest records what the split step *did* under `parameters_fitted`
        (`kind`, `by`, `time_column`); a hand-made one may carry `split.kind` or the step's
        `params`. All are read here: missing the executed record would downgrade the time-split
        check of rule W1 to a warning on every real manifest.
        """
        for path in ("split/kind", "split/strategy"):
            v = dig(self.manifest, path)
            if v:
                return v
        return self._split_param("kind")

    def split_by(self):
        """The column the split grouped or stratified by, or None."""
        return dig(self.manifest, "split/by") or self._split_param("by")

    def _split_param(self, key):
        for step in self.manifest_steps():
            if step.get("op") != "split":
                continue
            for holder in ("parameters_fitted", "params", "parameters"):
                value = (step.get(holder) or {}).get(key)
                if value:
                    return value
        return None

    def rows_of(self, part):
        node = dig(self.manifest, f"outputs/files/{part}")
        return node.get("rows") if isinstance(node, Mapping) else None

    # -- the v2.3 sample: keys, the graph block, the series key

    def output_roles(self) -> Mapping[str, Any] | None:
        """{column: role} when the output schema is the shaper's mapping, else None."""
        node = dig(self.manifest, "output_schema")
        if isinstance(node, Mapping) and not isinstance(node.get("columns"), (list, tuple)):
            return {k: (v or {}).get("role") for k, v in node.items() if isinstance(v, Mapping)}
        return None

    @property
    def graph(self) -> Mapping[str, Any] | None:
        """The manifest's `graph` block (`shape:manifest/graph`), written by a `graph_edges`
        step; None when no such step ran."""
        node = dig(self.manifest, "graph")
        return node if isinstance(node, Mapping) else None

    def edges_path(self):
        node = dig(self.manifest, "outputs/files/edges")
        return node.get("path") if isinstance(node, Mapping) else None

    def graphs_of(self, part):
        """The number of graphs in a part at level graph, when the graph block records it."""
        count = (self.graph or {}).get("n_graphs")
        if isinstance(count, Mapping):
            return count.get(part)
        return None

    def series_key(self) -> tuple:
        """(column, source) of the series key the shaping record names — the `lag` step's
        `group_by`, else a time split's `by` (`defaults/window_series`) — or (None, None).

        The step's own parameters are looked up in the manifest and, since an executed
        manifest records only what a lag step fitted, in the shape layer's recipe as well.
        """
        recipe = {s.get("id"): s for s in ((self.layer("shape") or {}).get("recipe") or {})
                  .get("steps") or () if isinstance(s, Mapping)}
        for step in self.manifest_steps():
            if step.get("op") != "lag":
                continue
            for node in (step, recipe.get(step.get("id")) or {}):
                for holder in ("params", "parameters", "parameters_fitted"):
                    column = (node.get(holder) or {}).get("group_by")
                    if column:
                        return column, f"shape:{step.get('id')}"
        if self.split_kind() == "time" and self.split_by():
            step = next((s for s in self.manifest_steps() if s.get("op") == "split"), {})
            return self.split_by(), (f"shape:{step['id']}" if step.get("id")
                                     else "shape:manifest/split")
        return None, None

    def port_with_role(self, role):
        return next((p for p in ((self.pipeline or {}).get("ports") or [])
                     if isinstance(p, Mapping) and p.get("role") == role), None)

    @property
    def port_roles(self) -> frozenset:
        return frozenset(p.get("role") for p in ((self.pipeline or {}).get("ports") or [])
                         if isinstance(p, Mapping))


# --------------------------------------------------------------------------- sources

@dataclass(frozen=True)
class SourceResolver:
    """The closed set of `source` values, and what each one resolves against.

    Well-formedness and resolution are deliberately separate questions: `analysis:F9` is a
    well-formed source that does not resolve when the analysis layer has no F9, and the two
    failures deserve different sentences.
    """

    inputs: ChainInputs
    catalog: Catalog

    def well_formed(self, src) -> bool:
        return any(re.match(p, str(src)) for p in self.catalog.source_patterns.values())

    def resolve(self, src) -> tuple:
        """(resolved, detail)."""
        s = str(src)
        if s == "vega:default":
            return True, "the family catalog"
        if s.startswith("user:"):
            return True, "a user turn or a recorded Network stance"
        if s.startswith("blueprint:"):
            return self._blueprint(s[len("blueprint:"):])
        if s.startswith("shape:"):
            return self._shape(s[len("shape:"):])
        if s.startswith("geometry:"):
            return self._geometry(s[len("geometry:"):])
        if s.startswith("analysis:"):
            return self._analysis(s[len("analysis:"):])
        return False, ("not one of blueprint: / shape: / geometry: / analysis: / user: / "
                       "vega:default")

    def _blueprint(self, rest) -> tuple:
        pipeline = self.inputs.pipeline
        if not pipeline:
            return False, "no blueprint pipeline.json was given, and the blueprint is required"
        if rest == "posture":
            return "posture" in pipeline, "pipeline.json.posture"
        if rest == "artifacts":
            return bool(pipeline.get("artifacts")), "pipeline.json.artifacts"
        for prefix, key, ident in (("bets/", "bets", "id"), ("ports/", "ports", "name"),
                                   ("modules/", "modules", "name"),
                                   ("boundaries/", "boundaries", "name")):
            if rest.startswith(prefix):
                name = rest[len(prefix):]
                found = find_in_list(pipeline.get(key), ident, name)
                return found is not None, f"pipeline.json.{key}[{ident}={name}]"
        return rest in pipeline, f"pipeline.json.{rest}"

    def _shape(self, rest) -> tuple:
        if re.fullmatch(r"S\d+", rest):
            steps = list(self.inputs.manifest_steps())
            if not steps:
                shape_layer = self.inputs.layer("shape") or {}
                steps = ((shape_layer.get("recipe") or {}).get("steps")) or []
            return find_in_list(steps, "id", rest) is not None, f"the shaper's step {rest}"
        if rest.startswith("manifest/"):
            if not self.inputs.manifest:
                return False, "no shaped manifest.json was found"
            key = rest[len("manifest/"):]
            return dig(self.inputs.manifest, key) is not None, f"manifest.{key}"
        return False, "shape: takes manifest/<key> or S<n>"

    def _geometry(self, rest) -> tuple:
        layer = self.inputs.layer("geometry")
        if layer is None:
            return False, "the model carries no geometry layer"
        head, _, name = str(rest).partition("/")
        if head == "partitions" and name:
            return self._partition(layer, name)
        if head == "typing" and name:          # typing[] is keyed by `column`, as the shaper reads it
            found = find_in_list(layer.get("typing"), "column", name) or dig(layer, rest)
            return found is not None, f"the geometry layer's typing of {name!r}"
        return dig(layer, rest) is not None, f"the geometry layer's {rest}"

    @staticmethod
    def _partition(layer, name) -> tuple:
        """`partitions/<label>` and `partitions/chosen`, resolved as `/dataset-shaper` resolves
        them (netspec-contract §1, the sources table): the geometry layer's `partitions` holds
        `chosen` (a label) and `candidates[]` by `label`, so `<label>` is the candidate whose
        `label` it is and `chosen` the candidate `chosen` names. A key after the label — `/key`
        or the shaper's `.key` (`partitions/late.dropped_for_leakage`) — must be the
        candidate's."""
        partitions = layer.get("partitions")
        if not isinstance(partitions, Mapping):
            return False, "the geometry layer has no partitions"
        label, sep, tail = name.partition("/")
        if not sep:
            label, _, tail = name.partition(".")
        if label == "chosen":
            label = partitions.get("chosen")
            if not label:
                return False, ("the geometry layer chose no partition "
                               f"(provenance {partitions.get('provenance')!r})")
        candidate = find_in_list(partitions.get("candidates"), "label", label)
        labels = [c.get("label") for c in partitions.get("candidates") or []
                  if isinstance(c, Mapping)]
        if candidate is None:
            return False, (f"the geometry layer has no partition candidate labelled {label!r} "
                           f"(candidates: {labels})")
        if tail and dig(candidate, tail.replace(".", "/")) is None:
            return False, f"the partition candidate {label!r} has no {tail!r}"
        return True, f"the geometry layer's partition candidate {label!r}"

    def _analysis(self, rest) -> tuple:
        layer = self.inputs.layer("analysis")
        if layer is None:
            return False, "the model carries no analysis layer"
        if re.fullmatch(r"F\d+", rest):
            return (find_in_list(layer.get("findings"), "id", rest) is not None,
                    f"the analysis finding {rest}")
        if rest == "importance":
            rest = "modules/importance"
        if rest.startswith("context."):
            return (dig(layer.get("context") or {}, rest.split(".", 1)[1]) is not None,
                    f"the analysis context {rest}")
        return dig(layer, rest) is not None, f"the analysis layer's {rest}"
