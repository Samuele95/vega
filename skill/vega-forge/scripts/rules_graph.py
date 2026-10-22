"""
rules_graph — the compiler's rule W2 on a graph sample (netspec-contract §1 and §3, D1 §14.1,
build plan R-A3, R-A10, R-A11).

  W2  a graph family, or a graph block, needs a `graph` sample; on one: the sample's source
      resolves (`shape:manifest/graph`); `level` is node or graph, `graph_id` iff level graph and
      a key column (never a feature); the manifest has the `edges` part the sample names and the
      `graph` block whose `level`, `graph_id` and `directed` the sample copies; a non-graph
      family is a WARN (its edges are ignored) unless the family's `floor_shapes` name graph;
      a family with a catalog `level` (gin) runs only there; `graph_pool` iff level graph; the
      `protocol` is inductive or transductive, transductive only from a `user:` source and with
      its disclosed `exposure {kind: features, rows: test, source: user:…}`, and no exposure on
      an inductive protocol; the `batching` is full, sampled or graphs — graphs iff level graph,
      `neighbours` (one positive fan-out per conv block) iff sampled.

The runtime half — `training.batch_size: "full"` iff batching full, an integer batch at most
the seed nodes or graphs — is R1's (`rules_sample.py`).
"""
from __future__ import annotations

import posixpath
from typing import Mapping

from rules_sample import int_at_least, key_columns, rule_markers, sample_source, sub_source
from vega_core import Level, ok

__all__ = ["graphs"]

_PROTOCOLS = ("inductive", "transductive")
_BATCHING = ("full", "sampled", "graphs")
_CONVS = ("gcn_conv", "sage_conv", "gat_conv", "gin_conv")


def _same_path(a, b) -> bool:
    """Equal paths (netspec-contract §3): the sample copies the manifest's own spelling of the
    edges part (§1, `"edges": "<manifest.outputs.files.edges path>"`), so the two normalised
    spellings must be equal — `./shaped/x` is `shaped/x`, but a bare file name, the same file
    name in another directory, or an absolute spelling of a relative one is not the manifest's
    path. The compiler reads no file system to reconcile spellings (a rule takes values)."""
    if not a or not b:
        return False
    return posixpath.normpath(str(a)) == posixpath.normpath(str(b))


def _graph_blocks(ctx):
    return [(path, b) for path, b, _ in ctx.blocks if isinstance(b, Mapping)
            and (ctx.catalog.block(b.get("type")) or {}).get("graph_only")]


def _off_graph(ctx):
    err, _ = rule_markers("W2")
    out = []
    if (ctx.family_row or {}).get("sample_shape") == "graph":
        out.append(err(f"family {ctx.family_name!r} is a graph family and needs "
                       "data_contract.sample.shape graph over the shaper's edges part; the "
                       f"sample is {ctx.sample_shape!r}"))
    for path, block in _graph_blocks(ctx):
        out.append(err(f"{block.get('type')!r} is a graph block and the sample is "
                       f"{ctx.sample_shape!r}: there are no edges to read", where=path))
    return out


def _graph_family(ctx):
    err, wrn = rule_markers("W2")
    level, row, out = ctx.sample.get("level"), ctx.family_row or {}, []
    if row and row.get("sample_shape") != "graph" and not ctx.catalog.is_floor_on(
            ctx.family_name, "graph"):
        out.append(wrn(f"family {ctx.family_name!r} is not a graph family: on a graph sample "
                       "the edges are ignored and every node is read by its own features"))
    if row.get("level") and level != row.get("level"):
        out.append(err(f"family {ctx.family_name!r} requires level {row.get('level')!r} "
                       f"(catalog families/{ctx.family_name}), the sample is level {level!r}"))
    pools = [path for path, b in _graph_blocks(ctx) if b.get("type") == "graph_pool"]
    if level == "graph" and not pools:
        out.append(err("level graph needs a graph_pool block before the head: the label is "
                       "one per graph, and the head must read graph rows"))
    if level == "node" and pools:
        out.append(err("graph_pool at level node reduces over the nodes, and at node level "
                       "every node is a sample — drop it", where=pools[0]))
    return out


def _manifest_graph(ctx):
    err, wrn = rule_markers("W2", "netspec-contract §3 (the edges part)")
    if ctx.inputs.manifest is None:
        return [wrn("the shaped manifest was not read; the edges part and the graph block "
                    "were not cross-checked")]
    edges, block, s = ctx.inputs.edges_path(), ctx.inputs.graph, ctx.sample
    if not edges or block is None:
        return [err("the shaped manifest has no edges part (outputs.files.edges) and no graph "
                    "block: a graph sample needs the shaper's graph_edges step")]
    out = []
    if not _same_path(s.get("edges"), edges):
        out.append(err(f"sample.edges {s.get('edges')!r} is not the manifest's edges part "
                       f"{edges!r} (manifest.outputs.files.edges.path)"))
    for key in ("level", "graph_id", "directed"):
        if s.get(key) != block.get(key):
            out.append(err(f"sample.{key} is {s.get(key)!r} but shape:manifest/graph records "
                           f"{block.get(key)!r} — the sample copies the shaper's graph block"))
    return out


def _graph_level(ctx):
    err, _ = rule_markers("W2")
    level, gid = ctx.sample.get("level"), ctx.sample.get("graph_id")
    if level not in ("node", "graph"):
        return [err(f"sample.level is {level!r}; it is node or graph")]
    if level == "graph" and not gid:
        return [err("level graph needs sample.graph_id, the key column that groups the rows "
                    "into graphs")]
    if level == "node" and gid:
        return [err(f"sample.graph_id {gid!r} is set at level node; it is a level-graph key")]
    return key_columns(ctx, ("graph_id",), "W2")


def _protocol(ctx):
    err, _ = rule_markers("W2", "netspec-contract §1 (protocol)")
    s = ctx.sample
    protocol, exposure = s.get("protocol"), s.get("exposure")
    if protocol not in _PROTOCOLS:
        return [err(f"sample.protocol is {protocol!r}; declare inductive (the default, "
                    "defaults/graph_protocol) or transductive (a user: turn only)")]
    out = sub_source(ctx, "protocol")
    if protocol == "inductive":
        if exposure:
            out.append(err("sample.exposure is set on an inductive protocol; the exposure is "
                           "the transductive protocol's disclosure, and nothing is exposed here"))
        return out
    if not str(s.get("protocol_source") or "").startswith("user:"):
        out.append(err("protocol transductive without a user: source — the test nodes' "
                       "features would be seen in training without anyone having said so"))
    if not isinstance(exposure, Mapping) or exposure.get("kind") != "features" or \
            exposure.get("rows") != "test" or \
            not str(exposure.get("source") or "").startswith("user:"):
        out.append(err("protocol transductive needs sample.exposure {kind: features, rows: "
                       f"test, source: user:<turn>}}, the disclosure the layer shows; got "
                       f"{exposure!r}"))
    return out


def _batching(ctx):
    err, _ = rule_markers("W2", "netspec-contract §1 (batching)")
    s, level = ctx.sample, ctx.sample.get("level")
    batching, neighbours = s.get("batching"), s.get("neighbours")
    if batching not in _BATCHING:
        return [err(f"sample.batching is {batching!r}; it is full, sampled or graphs "
                    "(defaults/graph_batching)")]
    out = sub_source(ctx, "batching")
    if (batching == "graphs") != (level == "graph"):
        out.append(err(f"sample.batching {batching!r} at level {level!r}: graphs (mini-batches "
                       "of whole graphs) is the level-graph batching, and only it"))
    convs = sum(1 for _, b, _ in ctx.blocks if isinstance(b, Mapping) and b.get("type") in _CONVS)
    if batching == "sampled":
        if not isinstance(neighbours, list) or len(neighbours) != convs or \
                not all(int_at_least(n, 1) for n in neighbours):
            out.append(err(f"batching sampled needs sample.neighbours, one positive fan-out per "
                           f"conv block ({convs}); got {neighbours!r}"))
    elif neighbours is not None:
        out.append(err(f"sample.neighbours {neighbours!r} is set with batching {batching!r}; "
                       "it is the sampled batching's fan-out, and only its"))
    return out


def graphs(ctx):
    if ctx.sample_shape != "graph":
        return _off_graph(ctx)
    out = (sample_source(ctx, "graph") + _graph_level(ctx) + _manifest_graph(ctx)
           + _graph_family(ctx) + _protocol(ctx) + _batching(ctx))
    if not any(m.level is Level.ERROR for m in out):
        out.append(ok(f"the graph sample (level {ctx.sample.get('level')}, protocol "
                      f"{ctx.sample.get('protocol')}, batching {ctx.sample.get('batching')}) "
                      "agrees with the shaper's graph block", rule="W2",
                      clause="netspec-contract §3"))
    return out
