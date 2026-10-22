"""
floors.py — the ladder's floors, the rungs a campaign reads its network against (R-A12,
D1 §12.3). `plan_ladder.py` plans them; this module builds their deltas.

    rung_floor(spec, catalog, dims_95=None)       R1: the floor of the sample's shape
    rung_recurrent_floor(spec, forge_cat)         R1b: the gated recurrent block as `rnn`

**R1** is `linear` for a supervised task and `pca-floor` for reconstruction and anomaly. On a
window `linear` reads the window through a `flatten` inserted before the head, at graph level
through `graph_pool(mean)`, at node level it is structure-blind (each node's own features);
the inserts are the catalog's (`ladder_rules.floor.inserts`). `pca-floor` is the linear
autoencoder of the forge's composition (D1 §6, `[embedding?] -> concat -> dense(z, none) ->
head(recon_dim, none)`): the features through `dense(z, none)` before the task's head, z the
spec's own bottleneck or the geometry layer's dims_95 (`Latent`). A spec whose family already
is the floor gets none, nor does a node-level sample under neighbour sampling, nor a PCA
floor with no z to read — each with the reason.

**R1b** applies to the families the forge catalog's `rnn` row is the ablation floor of
(`families.rnn.ablation_floor_for`): the gated block becomes `rnn` with the same `hidden` and
`num_layers`, `return: last`, `bidirectional: false`, and every block the rnn composition does
not hold is removed; the head is kept.

Every block a floor keeps carries the source it already had (a rung changes values; the
spec's `source` belongs to netspec-contract §1's closed set); every block it inserts cites the
family's composition, says why and names the alternative that was weighed.
"""
from __future__ import annotations

import json
from typing import Mapping, NamedTuple

SUPERVISED = ("binary", "multiclass", "multilabel", "regression", "ordinal",
              "sequence-classification", "sequence-regression", "forecast")


# --------------------------------------------------------------------------- the floor (R1)

def _copy(value):
    return json.loads(json.dumps(value))


def sample_of(spec) -> tuple:
    """(shape, level) of the spec's `data_contract.sample`: `row` when it names none; the
    level (`node` | `graph`) only for a graph sample."""
    sample = ((spec.get("data_contract") or {}).get("sample")) or {}
    shape = str(sample.get("shape") or "row")
    return shape, (str(sample.get("level") or "node") if shape == "graph" else None)


def _fresh_id(taken, stem):
    """A block id no block of the spec uses: `stem`, else `stem2`, `stem3`, …"""
    if stem not in taken:
        return stem
    n = 2
    while f"{stem}{n}" in taken:
        n += 1
    return f"{stem}{n}"


# What the linear floor inserts before the head per shape is the catalog's (`ladder_rules.floor.
# inserts`, keyed `row`, `window`, `graph/node`, `graph/graph`; R-A12, family-catalog.md
# §Floors): the head of a window reads the flattened window, a graph-level head one row per
# graph, and at node level nothing is inserted — the floor is structure-blind. Each inserted
# block cites the linear composition and says why, with the alternative that was weighed.
_INSERT_PROSE = {
    "flatten": {
        "stem": "floor_flatten",
        "rationale": "the linear floor on a window: the head reads the whole window flattened "
                     "(families/linear, window composition) — one weight per step and "
                     "feature, which is what a linear model over lags is",
        "alternatives": [{"type": "pool", "params": {"kind": "mean"},
                          "when": "averaging the steps would lose which step a value came "
                                  "from; flatten keeps it, so the floor is the honest linear "
                                  "reading of the window"}]},
    "graph_pool": {
        "stem": "floor_pool",
        "rationale": "the linear floor at graph level: the node rows are pooled per graph "
                     "before the head (families/linear, level-graph composition) — the label "
                     "is one per graph",
        "alternatives": [{"params": {"readout": "sum"},
                          "when": "if the graph's size carries the signal (a sum grows with "
                                  "it); mean is the catalog's level-graph floor"}]},
}


def floor_inserts(spec, catalog, family="linear"):
    """The blocks the floor inserts before the head, from the catalog: for `linear` the
    sample's (`ladder_rules.floor.inserts`: [] on rows and at node level), for `pca-floor`
    its bottleneck (`ladder_rules.floor.pca_inserts`: `dense(z, none)`)."""
    rule = ((catalog or {}).get("ladder_rules") or {}).get("floor") or {}
    if family == "pca-floor":
        return [_copy(b) for b in rule.get("pca_inserts") or []]
    shape, level = sample_of(spec)
    key = f"graph/{level}" if shape == "graph" else shape
    return [_copy(b) for b in (rule.get("inserts") or {}).get(key) or []]


def positive_width(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 \
        else None


class Latent(NamedTuple):
    """The PCA floor's latent width z, where it came from, and the geometry's dims_95."""

    z: int
    where: str
    dims_95: int | None = None

    @staticmethod
    def of(spec, dims_95=None) -> tuple:
        """(Latent, None) or (None, why) — families/pca-floor's `dense(z, none)`.

        The spec's own bottleneck first, so R0 and the floor compress to the same width and
        the comparison is about the non-linearity alone: the narrowest top-level `dense` (the
        `<bottleneck id>` rule) or a `reparameterize` block's z, whichever is narrower (a
        VAE's dense is 2z, its mean and log-variance). Else the geometry layer's dims_95, the
        catalog's default for z."""
        widths = []
        for block in spec.get("blocks") or []:
            if not isinstance(block, Mapping):
                continue
            key = {"dense": "units", "reparameterize": "z"}.get(block.get("type"))
            width = positive_width((block.get("params") or {}).get(key)) if key else None
            if width:
                widths.append((width, f"{block.get('type')} {block.get('id')}"))
        if widths:
            z, where = min(widths)
            return Latent(z, f"the spec's bottleneck ({where})", positive_width(dims_95)), None
        if positive_width(dims_95):
            return Latent(dims_95, "the geometry layer's dims_95", dims_95), None
        return None, ("the PCA floor needs its latent width z (families/pca-floor: "
                      "dense(z, none)) and neither the spec's bottleneck (a dense or "
                      "reparameterize block) nor the geometry layer's dims_95 gives one — "
                      "pass --model with the geometry layer")

    def block(self, insert, block_id) -> dict:
        """The inserted `dense(z, none)`, citing the PCA floor's composition."""
        params = {k: (self.z if v == "z" else v) for k, v in (insert.get("params") or {}).items()}
        other = self.dims_95 if self.dims_95 and self.dims_95 != self.z else None
        alternative = ({"params": {"units": other},
                        "when": "the catalog's default z (dims_95 from geometry:space), when "
                                "the spec's bottleneck differs from it"} if other else
                       {"params": {"units": self.z * 2},
                        "when": "a wider z keeps more variance; the floor keeps the width the "
                                "network itself compresses to"})
        return {"id": block_id, "type": insert.get("type"), "params": params,
                "source": "vega:default", "catalog": "families/pca-floor",
                "rationale": f"the PCA floor's linear bottleneck, z = {self.z} from "
                             f"{self.where} (families/pca-floor: dense(z, none) before the "
                             "head) — without it the floor is a full-rank linear map that can "
                             "learn the identity, and it would not be a PCA floor",
                "alternatives": [alternative]}


def _insert_block(insert, block_id, latent=None):
    """One inserted floor block, citing the family's composition and saying why."""
    if latent is not None:
        return latent.block(insert, block_id)
    prose = _INSERT_PROSE.get(insert.get("type")) or {}
    return {"id": block_id, "type": insert.get("type"),
            "params": dict(insert.get("params") or {}), "source": "vega:default",
            "catalog": "families/linear",
            "rationale": prose.get("rationale") or
            f"the linear floor's {insert.get('type')} for this sample shape (families/linear)",
            "alternatives": _copy(prose.get("alternatives") or [])}


def _floor_blocks(spec, family, catalog, latent=None):
    """The floor's blocks: the spec's embedding and concat, the family's inserts (the linear
    floor's on a window or at graph level, the PCA floor's `dense(z, none)` at `latent`), the
    task's head — or (None, why)."""
    blocks = [_copy(b) for b in spec.get("blocks") or []
              if isinstance(b, Mapping) and b.get("type") in ("embedding", "concat")]
    head = next((b for b in reversed(spec.get("blocks") or [])
                 if isinstance(b, Mapping) and b.get("type") == "head"), None)
    if head is None:
        return None, "the spec declares no head block, so no floor can be built from it"
    # The blocks the floor keeps carry the sources they already had: a rung changes VALUES,
    # and a spec's `source` belongs to netspec-contract §1's closed set, which has no room for
    # `vega:ladder/…`. The rung's own provenance lives in ladder.json, where it belongs.
    for block in blocks:
        block["rationale"] = (f"{block.get('rationale', '')} (kept by the floor: the question "
                              "is whether the network's depth buys anything, not whether "
                              "nominal ids can be read as numbers)").strip()
    taken = {str(b.get("id")) for b in spec.get("blocks") or [] if isinstance(b, Mapping)}
    for insert in floor_inserts(spec, catalog, family):
        stem = ((_INSERT_PROSE.get(insert.get("type")) or {}).get("stem")
                or ("floor_z" if family == "pca-floor" else f"floor_{insert.get('type')}"))
        block_id = _fresh_id(taken, stem)
        taken.add(block_id)
        blocks.append(_insert_block(insert, block_id, latent))
    if family == "pca-floor" and ((spec.get("loss") or {}).get("name")) == "vae":
        blocks[-1:] = _variational(blocks[-1], latent, taken)
    floor_head = _copy(head)
    floor_head["rationale"] = (f"{head.get('rationale', '')} (the floor keeps the task's head "
                               "unchanged; only what feeds it is removed, so the comparison is "
                               "about the non-linearity and nothing else)").strip()
    return blocks + [floor_head], None


def _variational(bottleneck, latent, taken):
    """A `vae` spec's PCA floor keeps its objective, so it keeps the variational bottleneck the
    objective reads: `dense(2z, none)` (the mean and the log-variance) into `reparameterize(z)`
    — a linear VAE, probabilistic PCA fitted the way the network is. Changing the loss instead
    would be a second decision in one rung (run-contract §3), and a vae loss without a
    reparameterize block has no KL term to read."""
    z = latent.z
    dense = dict(bottleneck, params={**(bottleneck.get("params") or {}), "units": 2 * z})
    dense["rationale"] = (f"the PCA floor's linear bottleneck for a variational objective: 2z = "
                          f"{2 * z} (the mean and the log-variance of z = {z}, {latent.where}), "
                          "linear, into the reparameterize block the vae loss reads")
    block_id = _fresh_id(taken, "floor_sample")
    taken.add(block_id)
    sample = {"id": block_id, "type": "reparameterize", "params": {"z": z}, "source": "vega:default",
              "catalog": "families/pca-floor",
              "rationale": "the vae objective reads the KL term of a reparameterize block; the "
                           "floor keeps one, so it trains on the network's own objective and "
                           "only the non-linearity is taken away",
              "alternatives": [{"params": {"z": z * 2}, "when": "a wider z keeps more variance"}]}
    return [dense, sample]


def _floor_reading(family, spec):
    """What the floor reads, per shape — the family node's rationale says it."""
    shape, level = sample_of(spec)
    if family != "linear":
        return (f"the catalog's {family} row is the reconstruction floor: a linear "
                "autoencoder through dense(z, none), PCA by another name")
    if shape == "window":
        return "the catalog's linear row on a window: the flattened window into the head"
    if shape == "graph" and level == "graph":
        return "the catalog's linear row at graph level: graph_pool(mean) into the head"
    if shape == "graph":
        return ("the catalog's linear row at node level, structure-blind by design: the floor "
                "reads each node's own features; whether the edges buy anything is what the "
                "graph family must show against it")
    return "the catalog's linear row on rows"


def rung_floor(spec, catalog, dims_95=None):
    """The linear floor (supervised) or the PCA floor (reconstruction, anomaly), per shape:
    `linear` with `flatten` before the head on a window, with `graph_pool(mean)` at graph
    level, structure-blind at node level (R-A12); `pca-floor` through `dense(z, none)` before
    the head (D1 §6), z from the spec's bottleneck or `dims_95`. A spec whose family already
    is the floor gets no floor rung: it would re-run the baseline."""
    kind = (spec.get("task") or {}).get("kind")
    family = "linear" if kind in SUPERVISED else "pca-floor"
    old = spec.get("family") or {}
    if old.get("name") == family:
        return None, (f"the spec's family is already {family}; the floor would re-run the "
                      "baseline, so R1 is not planned")
    sample = ((spec.get("data_contract") or {}).get("sample")) or {}
    if sample_of(spec) == ("graph", "node") and sample.get("batching") == "sampled":
        return None, (f"the sample draws neighbours per convolution (batching sampled, "
                      f"neighbours {sample.get('neighbours')}); the structure-blind floor has "
                      "no convolution to sample for, and a rung does not change the sample — "
                      "R1 is not planned (a floor needs the spec's batching full)")
    latent = None
    if family == "pca-floor":
        latent, why = Latent.of(spec, dims_95)
        if latent is None:
            return None, f"{why}; R1 is not planned"
    blocks, why = _floor_blocks(spec, family, catalog, latent)
    if blocks is None:
        return None, why
    floor_node = {"name": family, "source": "vega:default",
                  "catalog": f"families/{family}",
                  "rationale": f"{_floor_reading(family, spec)} — the floor every campaign "
                               "includes as an ablation; it is a default the catalog names, "
                               "not a decision this run invented",
                  "alternatives": [{"name": old.get("name"),
                                    "when": "the baseline this floor is read against"}]}
    shape, level = sample_of(spec)
    blind = shape == "graph" and level == "node"
    return {"id": "R1", "kind": "ablation", "family": "architecture",
            "delta": [{"path": "family", "from": old, "to": floor_node},
                      {"path": "blocks", "to": blocks}],
            "hypothesis": (f"the structure the graph family reads is real: the {family} floor, "
                           "blind to the edges, must sit clearly below the baseline" if blind
                           else f"the non-linear structure the analysis and the posture claim "
                                f"is real: the {family} floor must sit clearly below the "
                                "baseline"),
            "source": "vega:ladder/floor",
            "expect": "valid primary(floor) clearly below valid primary(R0); if it is not, the "
                      "network has not earned its parameters and the summary says so"}, None


# ------------------------------------------------------------------ the recurrent floor (R1b)

_GATED = ("gru", "lstm")


def _rnn_block(block, last):
    """A gated recurrent block as the plain `rnn` of the same size: `hidden` and
    `num_layers` kept, `bidirectional: false`, `return: last` when it feeds the head."""
    params = dict(block.get("params") or {})
    layers = params.get("num_layers", params.get("layers", 1))
    new = {"hidden": params.get("hidden"), "num_layers": layers, "nonlinearity": "tanh",
           "bidirectional": False,
           "return": "last" if last else params.get("return", "all")}
    if params.get("dropout") is not None:
        new["dropout"] = params["dropout"]
    was = {k: v for k, v in params.items() if k in ("bidirectional", "return")}
    return {"id": block.get("id"), "type": "rnn", "params": new, "source": "vega:default",
            "catalog": "families/rnn",
            "rationale": f"the recurrent floor (families/rnn, ablation_floor_for): the "
                         f"{block.get('type')} block {block.get('id')} as a plain tanh rnn of "
                         f"the same hidden size and depth — the gate is the only change",
            "alternatives": [{"type": block.get("type"), **({"params": was} if was else {}),
                              "when": "the gated cell this floor is read against"}]}


def rung_recurrent_floor(spec, forge_cat):
    """R1b (R-A12, D1 §12.3): for the families the catalog's `rnn` row is the ablation floor
    of (`families.rnn.ablation_floor_for`: gru, lstm, bi-*, *-attention), the spec with its
    gated recurrent block swapped to `rnn` — same `hidden` and `num_layers`, `return: last`,
    `bidirectional: false` — every block the rnn composition does not hold (the
    attention_pool, a pool) removed, the head kept. Returns (rung, why) with rung None and
    why None when the rule does not apply to this family."""
    rnn = ((forge_cat or {}).get("families") or {}).get("rnn") or {}
    old = spec.get("family") or {}
    if old.get("name") not in (rnn.get("ablation_floor_for") or ()):
        return None, None
    blocks = [b for b in spec.get("blocks") or [] if isinstance(b, Mapping)]
    gated = [i for i, b in enumerate(blocks) if b.get("type") in _GATED]
    if not gated:
        return None, (f"the {old.get('name')} network holds no gru or lstm block to swap to "
                      "rnn")
    keep = set(rnn.get("block_types") or ("embedding", "concat", "rnn", "dropout", "head"))
    out, removed = [], []
    for i, block in enumerate(blocks):
        if i in gated:
            out.append(_rnn_block(block, last=(i == gated[-1])))
        elif block.get("type") in keep:
            out.append(_copy(block))
        else:
            removed.append(f"{block.get('id')} ({block.get('type')})")
    node = {"name": "rnn", "source": "vega:default", "catalog": "families/rnn",
            "rationale": f"the catalog's rnn row is the ablation floor of {old.get('name')} "
                         "(families/rnn ablation_floor_for): the same window read by a plain "
                         "recurrent cell" + (f"; {', '.join(removed)} removed, the rnn "
                                             "composition holds no such block"
                                             if removed else ""),
            "alternatives": [{"name": old.get("name"),
                              "when": "the gated network this floor is read against"}]}
    return {"id": "R1b", "kind": "ablation", "family": "architecture",
            "delta": [{"path": "family", "from": old, "to": node},
                      {"path": "blocks", "to": out}],
            "hypothesis": "the gated cell buys long-range memory: rnn must sit below the "
                          "baseline",
            "source": "vega:ladder/recurrent-floor",
            "expect": "valid primary(R1b) below valid primary(R0); if it is not, the gate "
                      "buys nothing on windows of this length and the summary says so"}, None
