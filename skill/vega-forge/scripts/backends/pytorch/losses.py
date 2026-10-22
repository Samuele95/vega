"""PyTorch losses: the splice source of the Trainer's `<training>/losses.py` (D1 §4.1, §8) — the
`prelude` and `losses` segments, the task kind's `task` sections. `from_params` refuses a parameter
the loss does not take; constants are made once per device; no loss holds model state (R-B4/B6)."""
from __future__ import annotations

# --8<-- prelude
import warnings
from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch.nn import functional as F
# -->8-- prelude


# --8<-- losses
class _Loss:
    """A loss: a callable over (outputs, batch, model), its formula, and `from_params`. A loss
    sets `name` and `PARAMS`, calls `super().__init__(**its params)` (which makes the constant
    cache `_const` reads), implements `__call__`, may override `formula`, is named in `LOSSES`."""

    name, FORMULA = "loss", ""
    PARAMS: tuple = ()          # the loss.params keys this loss takes
    REFUSED = ""                # why a key a reader might expect is not one of them

    def __init__(self, **params):
        self.params = {k: v for k, v in params.items() if v is not None}
        self._consts = {}

    @classmethod
    def from_params(cls, params: Mapping):
        """`None` is the default; a key this loss does not take is refused, never dropped."""
        given = {k: v for k, v in dict(params or {}).items() if v is not None}
        unknown = sorted(set(given) - set(cls.PARAMS))
        if unknown:
            raise ValueError(f"the {cls.name} loss takes no parameter "
                             f"{', '.join(map(repr, unknown))}; it takes "
                             f"{', '.join(cls.PARAMS) or 'none'}{cls.REFUSED} (loss.params)")
        return cls(**given)

    def _const(self, key, values, device, dtype=None):
        """Made on first use, then the same tensor every call: a captured graph holds it."""
        slot = (key, str(device), str(dtype))
        if slot not in self._consts:
            self._consts[slot] = torch.tensor(values, dtype=dtype, device=device)
        return self._consts[slot]

    def _per_output(self, key, weights, outputs, dtype=None):
        """One value per class or label, checked against the head, as a device constant."""
        if len(weights) != outputs.shape[-1]:
            raise ValueError(f"{len(weights)} {key} values for a head of {outputs.shape[-1]} "
                             "outputs; they are one per class or label, in task.classes order")
        return self._const(key, weights.values, outputs.device, dtype)

    def formula(self):
        return self.FORMULA

    def __repr__(self):
        return f"{self.name}({self.params})"


# >>> task: binary multiclass multilabel
class BCEWithLogits(_Loss):
    """Binary (B,1) vs y (B,), multilabel (B,K) vs y (B,K); `pos_weight` a number or a K-vector."""

    name, PARAMS = "bce_with_logits", ("pos_weight",)

    def __init__(self, pos_weight=None):
        super().__init__(pos_weight=pos_weight)
        self.pos_weight = ClassWeights.or_scalar(pos_weight)

    def __call__(self, outputs, batch, model=None):
        w = self.pos_weight
        pw = (self._per_output("pos_weight", w, outputs) if isinstance(w, ClassWeights)
              else self._const("pos_weight", float(w), outputs.device) if w else None)
        y = batch["y"].to(outputs.dtype).reshape(outputs.shape)
        return F.binary_cross_entropy_with_logits(outputs, y, pos_weight=pw)

    def formula(self):
        w = self.pos_weight
        if isinstance(w, ClassWeights):
            return ("L = -mean_{i,k}[ w_k·y_ik·log σ(z_ik) + (1-y_ik)·log(1-σ(z_ik)) ],  "
                    f"w = pos_weight per label, K = {len(w)}")
        return (f"L = -mean[ {w}·y·log σ(z) + (1-y)·log(1-σ(z)) ]" if w else
                "L = -mean[ y·log σ(z) + (1-y)·log(1-σ(z)) ]")


class CrossEntropy(_Loss):
    """Cross-entropy, class-weighted (torch's 'mean' written out) and label-smoothed (c849f8b)."""

    name, PARAMS = "cross_entropy", ("weight", "label_smoothing")

    def __init__(self, weight=None, label_smoothing=0.0):
        super().__init__(weight=weight, label_smoothing=label_smoothing)
        self.weight = ClassWeights.or_scalar(weight)
        self.label_smoothing = float(label_smoothing or 0.0)

    def __call__(self, outputs, batch, model=None):
        y = batch["y"].reshape(-1).long()
        if self.weight is None:
            return F.cross_entropy(outputs, y, label_smoothing=self.label_smoothing)
        w = self._per_output("weight", self.weight, outputs, outputs.dtype)
        logp = F.log_softmax(outputs, dim=-1)
        wy = w.index_select(0, y)
        nll = -(logp.gather(1, y.unsqueeze(1)).squeeze(1) * wy).sum()
        if not self.label_smoothing:
            return nll / wy.sum()
        eps, k = self.label_smoothing, outputs.shape[-1]
        smooth = -(logp * w.unsqueeze(0)).sum()
        return ((1.0 - eps) * nll + (eps / k) * smooth) / wy.sum()

    def formula(self):
        eps = self.label_smoothing
        if self.weight is None:
            return "L = -mean[ log softmax(z)_y ]" + (f", label smoothing {eps}" if eps else "")
        if not eps:
            return "L = -Σ_i w[y_i]·log softmax(z_i)_{y_i} / Σ_i w[y_i]"
        return (f"L = Σ_i [ (1-{eps})·w[y_i]·(-log p_i,y_i) + ({eps}/K)·Σ_c w[c]·(-log p_i,c) ]"
                f" / Σ_i w[y_i],  K = {len(self.weight)}, class-weighted")


class Focal(_Loss):
    """CE scaled by (1-p_t)^γ, sigmoid per output or softmax (`multiclass`); `alpha` per class."""

    name, PARAMS = "focal", ("gamma", "alpha", "multiclass")

    def __init__(self, gamma=2.0, alpha=None, multiclass=False):
        super().__init__(gamma=gamma, alpha=alpha, multiclass=multiclass or None)
        self.gamma, self.multiclass = float(gamma), bool(multiclass)
        a = ClassWeights.or_scalar(alpha)
        self.alpha = a if a is None or isinstance(a, ClassWeights) else float(a)
        values = () if a is None else getattr(self.alpha, "values", (self.alpha,))
        if (self.multiclass and isinstance(self.alpha, float)) or \
                (not self.multiclass and not all(0.0 <= v <= 1.0 for v in values)):
            raise ValueError(f"focal alpha {alpha!r}: a number or one per label in [0, 1] "
                             "(sigmoid), or one weight per class (multiclass, softmax)")

    def __call__(self, outputs, batch, model=None):
        a = self.alpha
        if isinstance(a, ClassWeights):
            a = self._per_output("alpha", a, outputs, outputs.dtype)
        if self.multiclass:
            y = batch["y"].reshape(-1).long()
            logp = F.log_softmax(outputs, dim=-1).gather(1, y.unsqueeze(1)).squeeze(1)
            loss = -(1 - logp.exp()).pow(self.gamma) * logp
            return (loss if a is None else loss * a.index_select(0, y)).mean()
        y = batch["y"].to(outputs.dtype).reshape(outputs.shape)
        p = torch.sigmoid(outputs)
        ce = F.binary_cross_entropy_with_logits(outputs, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        loss = ce * (1 - pt).pow(self.gamma)
        if a is not None:
            loss = loss * (a * y + (1 - a) * (1 - y))
        return loss.mean()

    def formula(self):
        g, a = self.gamma, self.alpha
        if self.multiclass:
            return f"L = -mean[ {'α[y]·' if a else ''}(1-p_y)^{g} · log softmax(z)_y ]"
        if a is None:
            return f"L = -mean[ (1-p_t)^{g} · log p_t ]"
        what = f"α per label, K = {len(a)}" if isinstance(a, ClassWeights) else f"α = {a}"
        return f"L = -mean[ α_t·(1-p_t)^{g} · log p_t ],  α_t = α·y + (1-α)·(1-y), {what}"
# <<< task


# >>> task: regression forecast
class MSE(_Loss):
    name, FORMULA = "mse", "L = mean[ (ŷ - y)² ]"

    def __call__(self, outputs, batch, model=None):
        return F.mse_loss(outputs, batch["y"].to(outputs.dtype).reshape(outputs.shape))


class MAE(_Loss):
    name, FORMULA = "mae", "L = mean[ |ŷ - y| ]"

    def __call__(self, outputs, batch, model=None):
        return F.l1_loss(outputs, batch["y"].to(outputs.dtype).reshape(outputs.shape))


class Huber(_Loss):
    name, PARAMS = "huber", ("delta",)

    def __init__(self, delta=1.0):
        super().__init__(delta=delta)
        self.delta = float(delta)

    def __call__(self, outputs, batch, model=None):
        return F.huber_loss(outputs, batch["y"].to(outputs.dtype).reshape(outputs.shape),
                            delta=self.delta)

    def formula(self):
        d = self.delta
        return f"L = mean[ ½(ŷ-y)² if |ŷ-y| ≤ {d}, else {d}(|ŷ-y| - ½·{d}) ]"


class Quantile(_Loss):
    """Pinball over rows, horizons, levels; a horizon-major head (B,Q) or (B,H·Q), col h·Q + q."""

    name, PARAMS = "quantile", ("quantiles", "horizon")

    def __init__(self, quantiles=(0.5,), horizon=None):
        qs = list(quantiles) if isinstance(quantiles, (list, tuple)) else [float(quantiles)]
        super().__init__(quantiles=qs, horizon=horizon)
        self.quantiles = [float(q) for q in qs]
        self.horizon = None if horizon is None else int(horizon)
        if not self.quantiles or not all(0.0 < q < 1.0 for q in self.quantiles) \
                or (self.horizon is not None and self.horizon < 1):
            raise ValueError(f"quantile levels lie in (0, 1) and a horizon is at least 1; got "
                             f"quantiles {qs}, horizon {horizon!r}")

    def __call__(self, outputs, batch, model=None):
        n_q, lead = len(self.quantiles), (self.horizon,) if self.horizon else ()
        if outputs.shape[-1] != (self.horizon or 1) * n_q:
            raise ValueError(f"the quantile loss reads H·Q = {self.horizon or 1}·{n_q} head "
                             f"outputs; the head has {outputs.shape[-1]}")
        q = self._const("levels", self.quantiles, outputs.device, outputs.dtype)
        pred = outputs.reshape(outputs.shape[0], *lead, n_q)
        diff = batch["y"].to(outputs.dtype).reshape(outputs.shape[0], *lead, 1) - pred
        return torch.maximum(q * diff, (q - 1) * diff).mean()

    def formula(self):
        h, i = self.horizon, "ih" if self.horizon else "i"
        tail = f", h = 1…H (H = {h}); head column h·Q + q (horizon-major)" if h else ""
        return (f"L = mean_{{{i},q}} max[ q(y_{i} - ŷ_{i}q), (q-1)(y_{i} - ŷ_{i}q) ] over q ∈ "
                f"{self.quantiles} (Q = {len(self.quantiles)}){tail}")
# <<< task


# >>> task: ordinal
class CumulativeLink(_Loss):
    """Ordinal: the k-1 cumulative logits of the rank-consistent head, each 'class > j'."""

    name = "cumulative_link"
    FORMULA = ("L = -mean_j[ 1{y>j}·log σ(z_j) + 1{y≤j}·log(1-σ(z_j)) ]  (j = 0…k-2; "
               "rank-consistent thresholds, z_0 ≥ z_1 ≥ …)")

    def __call__(self, outputs, batch, model=None):
        y = batch["y"].reshape(-1).long()
        thresholds = torch.arange(outputs.shape[-1], device=outputs.device).unsqueeze(0)
        target = (y.unsqueeze(1) > thresholds).to(outputs.dtype)
        return F.binary_cross_entropy_with_logits(outputs, target)
# <<< task


# >>> task: unsupervised anomaly
class Reconstruction(_Loss):
    """Numeric mse + nominal mean CE over `[n_num numeric | card_1 … card_c logits]` (D1 §3.4)."""

    name, PARAMS = "reconstruction", ("n_numeric", "cat_cardinalities")

    def __init__(self, n_numeric=None, cat_cardinalities=None):
        super().__init__(n_numeric=n_numeric, cat_cardinalities=cat_cardinalities)
        self.n_numeric = n_numeric
        if any(c is None for c in cat_cardinalities or ()):
            raise ValueError("an embedded nominal has no cardinality: the reconstruction head's "
                             "logits are sized by data_contract.features[].cardinality")
        self.cat_cardinalities = [int(c) for c in cat_cardinalities or ()]

    def __call__(self, outputs, batch, model=None):
        x_num = batch["x_num"]
        n = self.n_numeric if self.n_numeric is not None else x_num.shape[-1]
        if outputs.shape[-1] != n + sum(self.cat_cardinalities):
            raise ValueError(f"the reconstruction head has {outputs.shape[-1]} outputs; recon_dim "
                             f"= {n} numeric + {sum(self.cat_cardinalities)} nominal logits")
        num = F.mse_loss(outputs[..., :n], x_num[..., :n].to(outputs.dtype))
        if not self.cat_cardinalities or batch.get("x_cat") is None:
            return num
        logits = outputs[..., n:].split(self.cat_cardinalities, dim=-1)
        terms = [F.cross_entropy(z, batch["x_cat"][..., i].long()) for i, z in enumerate(logits)]
        return num + torch.stack(terms).mean()

    def formula(self):
        return ("L = mean[ (x̂_num - x_num)² ] + mean_c[ -log softmax(x̂_c)_{x_c} ]"
                if self.cat_cardinalities else "L = mean[ (x̂ - x)² ]")


def _eval_divergence(model, batch):
    """`Reparameterize.divergence(x)` on this batch, its input caught by a one-forward pre-hook."""
    block = next((m for m in model.modules() if callable(getattr(m, "divergence", None))), None)
    if block is None:
        return None
    seen = []
    handle = block.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    try:
        model(batch)
    finally:
        handle.remove()
    return block.divergence(seen[-1]) if seen else None


class VAE(_Loss):
    """Reconstruction + β·KL: training reads its forward's `model.kl()`; eval recomputes it."""

    name, PARAMS = "vae", ("beta", "n_numeric", "cat_cardinalities")

    def __init__(self, beta=1.0, n_numeric=None, cat_cardinalities=None):
        super().__init__(beta=beta, n_numeric=n_numeric, cat_cardinalities=cat_cardinalities)
        self.beta = float(beta)
        self.rec = Reconstruction(n_numeric=n_numeric, cat_cardinalities=cat_cardinalities)

    def __call__(self, outputs, batch, model=None):
        kl = (None if model is None else _eval_divergence(model, batch) if not model.training
              else model.kl())
        if kl is None:
            raise ValueError("the vae loss needs the KL term of a reparameterize block on this "
                             "forward pass; the spec's family says vae but its blocks have none")
        return self.rec(outputs, batch, model) + self.beta * kl

    def formula(self):
        return f"L = reconstruction + {self.beta}·KL(q(z|x) ‖ N(0, I))"


class SVDD(_Loss):
    """Deep SVDD: embeddings pulled to c, the first train batch's mean (|c_j| ≥ 0.1), a buffer."""

    name, EPS = "svdd", 0.1
    FORMULA = ("L = mean[ ‖f(x) - c‖² ]   (c fixed from the first training batch, stored with "
               "the weights)")
    REFUSED = "; the centre is the network's buffer, set from the first training batch (R-B6)"

    def __call__(self, outputs, batch, model=None):
        center = model.objective_center() if hasattr(model, "objective_center") else None
        is_set = None if center is None else model.svdd_center_set    # registered with the centre
        if center is None or is_set is None or center.shape[-1:] != outputs.shape[-1:]:
            raise ValueError("the svdd centre is the network's buffers svdd_center (through "
                             "objective_center()) and svdd_center_set, z wide (R-B6); this model: "
                             f"{getattr(center, 'shape', None)}, head z = {outputs.shape[-1]}")
        first = outputs.detach().mean(dim=0)
        pushed = torch.full_like(first, self.EPS).copysign(first)
        first = torch.where(first.abs() < self.EPS, pushed, first).to(center.dtype)
        c = torch.where(is_set, center, first)
        if model.training:
            center.copy_(c)
            is_set.fill_(True)
        return torch.mean(torch.sum((outputs - c) ** 2, dim=-1))
# <<< task


LOSSES = MappingProxyType({loss.name: loss for loss in (
    # >>> task: binary multiclass multilabel
    BCEWithLogits, CrossEntropy, Focal,
    # <<< task
    # >>> task: regression forecast
    MSE, MAE, Huber, Quantile,
    # <<< task
    # >>> task: ordinal
    CumulativeLink,
    # <<< task
    # >>> task: unsupervised anomaly
    Reconstruction, VAE, SVDD,
    # <<< task
)})     # the task kind's losses, by name: written, not discovered (a subclass is no entry)
# what the contract fixes: a loss.params value that disagrees is refused, not left to the step
_FIXED = {"multiclass": "the effective task's", "horizon": "a forecast's sample.horizon, or none"}


def _resolved(name, spec):
    """What the contract fixes: the reconstruction layout, a forecast's horizon, focal's link."""
    if name in ("reconstruction", "vae"):
        fc = spec.feature_contract
        return {"n_numeric": len(fc.numeric_columns), "cat_cardinalities": list(fc.cardinalities)}
    if name == "quantile":
        window = SampleSpec.from_spec(spec).window if spec.task is TaskKind.FORECAST else None
        return {"horizon": int(window.horizon) if window is not None else None}
    if name == "focal":
        return {"multiclass": Task.from_spec(spec).effective is TaskKind.MULTICLASS}
    return {}


def _own_params(node, cls):
    """loss.params less another loss's keys a weighed alternative holds (ladder merges): warned."""
    params, carried = node.params_dict(), set()
    for alt in node.alternatives_list():
        other = LOSSES.get(alt.get("name")) if alt.get("name") != cls.name else None
        carried |= {k for k, v in (alt.get("params") or {}).items() if other is not None
                    and k in other.PARAMS and k not in cls.PARAMS and params.get(k) == v}
    if carried:
        warnings.warn(f"loss.params {', '.join(sorted(carried))}: not {cls.name} parameters but "
                      "the weighed alternative's (loss.alternatives); dropped", stacklevel=3)
    return {k: v for k, v in params.items() if k not in carried}


def build_loss(spec):
    """The spec's `loss` block: its parameters and what the contract fixes, nothing else."""
    spec = NetSpec.require(spec)
    node = spec.loss
    cls = LOSSES.get(node.name)
    if cls is None:
        raise ValueError(f"unknown loss {node.name!r}; this backend implements {sorted(LOSSES)}")
    params = Regularization.of(spec).loss_params(cls.PARAMS, _own_params(node, cls))   # R2
    for key, value in _resolved(cls.name, spec).items():
        given = params.get(key)
        if given is not None and key in _FIXED and given != value:
            raise ValueError(f"loss.params.{key} is {given!r}, but the contract fixes it at "
                             f"{value!r} ({_FIXED[key]}; task {spec.task.value}); leave it unset")
        params[key] = value if given is None else given
    return cls.from_params(params)
# -->8-- losses
