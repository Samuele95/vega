"""
inspection.readers_graph — the readers of the graph blocks: `gcn_conv`, `sage_conv`, `gat_conv`,
`gin_conv` (stage kind `graph`, called `m(h, edge_index, edge_weight)`), `graph_pool` (`gpool`,
`m(h, graph_index, num_graphs)`) and `edge_dropout` (`gdrop`, `m(edge_index, edge_weight)`)
(inspect-contract §3.1, §5.3; build plan R-B2, R-B3).

Each conv is read through the surface R-B3 fixes — GCN `.lin .bias .normalize .self_loops`,
SAGE `.lin_root .lin_neigh .aggregator .normalize`, GAT `.lin .att_src .att_dst .heads .concat
.negative_slope`, GIN `.mlp .eps` — and its payload names the tensors a message is formed from
(the probes of the explanation read the same tensors through `coefficients`). A trainable GIN
ε is a parameter and is published as the tensor `<node>.eps`; a fixed one is a number only.
`edge_dropout` is the identity at inference, which its payload says. What an explanation reads
of each — the messages into the explained targets, the pool's node shares, their checks — is
`probes_graph.py`'s.
"""
from __future__ import annotations

from torch import nn

from . import probes_graph as PR
from .codec import num
from .readers import _Reader

__all__ = ["GRAPH"]


class _GCNConv(_Reader):
    TYPE, VIEW, CALL = "gcn_conv", "graph", "graph"
    PROBE = PR.GCN
    FORMULA = "h'_i = Σ_{j∈N(i)∪{i}} Â_ij·W h_j + b"
    PAYLOAD = ("weight", "bias", "normalize", "self_loops")

    def config(self, block, module):
        return {"out": int(module.lin.out_features), "normalize": str(module.normalize),
                "self_loops": bool(module.self_loops), "bias": module.bias is not None}

    def params(self, module, site):
        return {"weight": site.tensor(module.lin.weight, "weight", "weight", ("out", "in")),
                "bias": site.tensor(module.bias, "bias", "bias", ("out",)),
                "normalize": str(module.normalize), "self_loops": bool(module.self_loops)}


class _SAGEConv(_Reader):
    TYPE, VIEW, CALL = "sage_conv", "graph", "graph"
    PROBE = PR.SAGE
    FORMULA = "h'_i = W_r h_i + W_n·agg_{j∈N(i)} h_j"
    PAYLOAD = ("root", "neigh", "bias", "aggregator", "normalize")

    def config(self, block, module):
        return {"out": int(module.lin_neigh.out_features), "aggregator": str(module.aggregator),
                "root_weight": module.lin_root is not None,
                "normalize": bool(module.normalize), "bias": module.lin_neigh.bias is not None}

    def params(self, module, site):
        root = module.lin_root
        return {"root": None if root is None else
                site.tensor(root.weight, "root", "weight", ("out", "in")),
                "neigh": site.tensor(module.lin_neigh.weight, "neigh", "weight", ("out", "in")),
                "bias": site.tensor(module.lin_neigh.bias, "bias", "bias", ("out",)),
                "aggregator": str(module.aggregator), "normalize": bool(module.normalize)}


class _GATConv(_Reader):
    TYPE, VIEW, CALL = "gat_conv", "graph", "graph"
    PROBE = PR.GAT
    FORMULA = ("α_ij = softmax_j(LeakyReLU(a_srcᵀ W h_j + a_dstᵀ W h_i)) ; "
               "h'_i = ‖_h Σ_j α^h_ij W^h h_j")
    PAYLOAD = ("weight", "bias", "att_src", "att_dst", "heads", "concat", "negative_slope")

    def config(self, block, module):
        return {"out": int(module.att_src.shape[-1]), "heads": int(module.heads),
                "concat": bool(module.concat), "negative_slope": float(module.negative_slope),
                "bias": module.bias is not None}

    def params(self, module, site):
        return {"weight": site.tensor(module.lin.weight, "weight", "weight", ("head_unit", "in")),
                "bias": site.tensor(module.bias, "bias", "bias", ("out",)),
                "att_src": site.tensor(module.att_src, "att_src", "weight", ("head", "unit")),
                "att_dst": site.tensor(module.att_dst, "att_dst", "weight", ("head", "unit")),
                "heads": int(module.heads), "concat": bool(module.concat),
                "negative_slope": num(module.negative_slope)}


class _GINConv(_Reader):
    TYPE, VIEW, CALL = "gin_conv", "graph", "graph"
    PROBE = PR.GIN
    FORMULA = "h'_i = MLP((1+ε)·h_i + Σ_{j∈N(i)} h_j)"
    PAYLOAD = ("mlp", "eps", "train_eps")

    def config(self, block, module):
        lins = [m for m in module.mlp if isinstance(m, nn.Linear)]
        held = {"out": int(lins[-1].out_features), "mlp_layers": len(lins),
                "train_eps": isinstance(module.eps, nn.Parameter)}
        if not held["train_eps"]:     # a trained ε is no longer the spec's initial value
            held["eps"] = float(module.eps.detach().reshape(-1)[0])
        return held

    def params(self, module, site):
        mlp = [{"weight": site.tensor(m.weight, f"mlp.{j}.weight", "weight", ("out", "in")),
                "bias": site.tensor(m.bias, f"mlp.{j}.bias", "bias", ("out",))}
               for j, m in enumerate(module.mlp) if isinstance(m, nn.Linear)]
        trained = isinstance(module.eps, nn.Parameter)
        if trained:
            site.tensor(module.eps, "eps", "weight", ("one",))
        return {"mlp": mlp, "eps": num(module.eps.detach().reshape(-1)[0]), "train_eps": trained}


class _GraphPool(_Reader):
    TYPE, VIEW, CALL, FORMULA, PAYLOAD = "graph_pool", "pool", "gpool", \
        "g = mean|max|sum_{i∈G} h_i", ("readout",)
    PROBE = PR.GRAPH_POOL

    def config(self, block, module):
        return {"readout": str(module.readout)}


class _EdgeDropout(_Reader):
    TYPE, VIEW, CALL, FORMULA, PAYLOAD = "edge_dropout", "edges", "gdrop", \
        "E' = E (inactive at inference)", ("p", "active_at_inference")
    PROBE = PR.EDGE_DROPOUT

    def config(self, block, module):
        return {"p": float(module.p)}

    def params(self, module, site):
        return {"p": num(module.p), "active_at_inference": False}


GRAPH = (_GCNConv(), _SAGEConv(), _GATConv(), _GINConv(), _GraphPool(), _EdgeDropout())
