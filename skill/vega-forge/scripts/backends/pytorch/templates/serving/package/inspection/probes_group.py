"""
inspection.probes_group — the probes of the latent block and of the synthetic nodes
(inspect-contract §3.1; build plan R-B4, R-B7): `reparameterize`, `input`, `output`,
`residual_add`, `parallel_merge`. The composites (`sequential`, `residual`, `parallel`,
`decoder`, `transformer_encoder`) and `custom` have no probe and no check (`probes.NO_PROBE`):
a composite is read through its children, user code is opaque.

- **reparameterize**: `mu` and `sigma = exp(½·logvar)` from the block's own pure `split`
  (logvar clamped as the block clamps it) and `kl`, the sample's KL term recomputed from its
  input — never read from the block's `.kl`, which an eval pass does not write (R-B4); the
  check is out = mu.
- **input**: `values`, the sample's shaped inputs (numeric values; nominal ids); the edge stream
  has no per-sample values (its relevance is the attribution's).
- **output**: `readout` — the link, the head's raw values for the sample and what
  `TaskReadout.proba` reads from them (R-B7: the one source of output semantics).
- **residual_add**: `skip_norm` and `branch_norm` of the two terms it adds (per step on a
  sequence); the check is skip + branch = out.
- **parallel_merge**: `branch_norms`, each branch output's norm; the check is their
  concatenation or sum = out.
"""
from __future__ import annotations

import torch

from .probes import Probe, f64, gap, norm_last, reduced, sample_io

__all__ = ["INPUT", "OUTPUT", "PARALLEL_MERGE", "REPARAMETERIZE", "RESIDUAL_ADD"]


class _Reparameterize(Probe):
    KEYS = ("mu", "sigma", "kl")
    CHECKS = True

    def _halves(self, cap, node) -> tuple:
        """(mu, logvar, out) through the block's own pure `split` (it writes no attribute)."""
        x, y = sample_io(cap, node)
        with torch.no_grad():
            mu, logvar = cap.module(node.id).split(x)
        return mu, logvar, y

    def probe(self, cap, node):
        mu, logvar, _ = self._halves(cap, node)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return {"mu": reduced(cap, mu), "sigma": reduced(cap, torch.exp(0.5 * logvar)),
                "kl": reduced(cap, kl)}

    def recompute(self, cap, node):
        mu, _, y = self._halves(cap, node)
        return gap(mu, y)


class _Input(Probe):
    KEYS = ("values",)

    def probe(self, cap, node):
        return {"values": reduced(cap, f64(cap.take(cap.value(node.id), node)))}


class _Output(Probe):
    KEYS = ("readout",)

    def probe(self, cap, node):
        anatomy, head = cap.anatomy, cap.value(node.id)
        with torch.no_grad():
            proba = anatomy.readout.proba(head.detach(), cap.batch, anatomy)
        return {"readout": {"link": str(anatomy.readout.link),
                            "raw": reduced(cap, f64(cap.take(head, node))),
                            "proba": None if proba is None else
                            reduced(cap, f64(cap.take(proba, node)))}}


class _ResidualAdd(Probe):
    KEYS = ("skip_norm", "branch_norm")
    CHECKS = True

    def _terms(self, cap, node) -> tuple:
        feeds = {kind: t for _, kind, t in cap.feeds(node.id)}
        return (f64(cap.take(feeds["skip"], node)), f64(cap.take(feeds["data"], node)),
                f64(cap.take(cap.value(node.id), node)))

    def probe(self, cap, node):
        skip, branch, _ = self._terms(cap, node)
        return {"skip_norm": reduced(cap, norm_last(skip)),
                "branch_norm": reduced(cap, norm_last(branch))}

    def recompute(self, cap, node):
        skip, branch, out = self._terms(cap, node)
        return gap(skip + branch, out)


class _ParallelMerge(Probe):
    KEYS = ("branch_norms",)
    CHECKS = True

    def _branches(self, cap, node) -> list:
        return [f64(cap.take(t, node)) for _, kind, t in cap.feeds(node.id) if kind == "merge"]

    def probe(self, cap, node):
        return {"branch_norms": [reduced(cap, norm_last(b)) for b in self._branches(cap, node)]}

    def recompute(self, cap, node):
        branches = self._branches(cap, node)
        out = f64(cap.take(cap.value(node.id), node))
        if str(cap.module(node.parent).merge) == "sum":
            return gap(torch.stack(branches).sum(dim=0), out)
        return gap(torch.cat(branches, dim=-1), out)


REPARAMETERIZE = _Reparameterize()
INPUT = _Input()
OUTPUT = _Output()
RESIDUAL_ADD = _ResidualAdd()
PARALLEL_MERGE = _ParallelMerge()
