"""
The choices of a composed netspec (lanekit.spec): each one a node carrying its `source`, its
`rationale` and the alternatives weighed (netspec-contract §1).

    family_choice(raw)                     blocks(raw, contract, target, task_source)
    loss(raw, target, balance_source)      metrics(target, labelled)
    optimizer(raw)   schedule(raw, budget, lr)   init(raw)       update_rule(name, params)
    regularization(blocks, optimizer)      early_stopping(metrics, budget)
    declared_once(blocks, optimizer, regularization)            training(contract, sample, budget)

One policy decides every source (the catalog's own reading): a choice equal to the catalog's
default for its context is `vega:default` with the catalog row it cites; a choice the lane
declares against the default is the lane's stance, `user:stance/<key>` (lanekit.fixture writes
the matching memory line); the head cites the task's source. Every non-structural choice names
one alternative with its `when`. The optimizer states its `update_rule` (netspec-contract §1),
which the architecture and training layers show beside the loss's formula.

`training` carries the contract's source for its batch size (netspec-contract §1, amendment A8):
`vega:default` citing `defaults/batch_size` — clip(training targets / 8, 16, 512) to a power of
two, the rule's arithmetic in the rationale — with the runtime rows it bundles, or, under full
batching, `"full"` sourced as the sample's batching is (`defaults/graph_batching`, or the lane's
stance). No field outside the contract's: the lane budget (`max_epochs`, `budget.wall_clock_min`,
`user:stance/lane-budget`, which early_stopping cites) is stated in the rationale. D6 §5.3 fixed
64 (128 for L21); the rule sizes the batch to the targets the lane's shaper actually leaves.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from .spec_data import (CATALOG, CONV_TYPES, INIT_DEFAULTS, OPTIMIZER_PARAMS, TASK_ROWS, Budget,
                        Contract, Target, batch_rule, head_out)

__all__ = ["choice", "family_choice", "block_alternative", "blocks", "loss", "metrics", "optimizer",
           "schedule", "init", "regularization", "declared_once", "early_stopping", "training",
           "FLOOR_ALTERNATIVE",
           "UPDATE_RULES", "update_rule"]

# a floor family's catalog row names no alternative: it is weighed against the network it floors
FLOOR_ALTERNATIVE = {"linear": "mlp-plain", "pca-floor": "autoencoder"}
UNDERFITS = "if the run underfits"


def choice(source: str, rationale: str, alternatives: Sequence[Mapping[str, Any]],
           catalog: str | Sequence[str] | None = None) -> dict:
    """{source, [catalog], rationale, alternatives}: the catalog row only on a vega:default."""
    node: dict[str, Any] = {"source": source}
    if catalog is not None and source == "vega:default":
        node["catalog"] = catalog if isinstance(catalog, str) else list(catalog)
    node["rationale"] = rationale
    node["alternatives"] = [dict(a) for a in alternatives]
    return node


# --------------------------------------------------------------------------- the family
def family_choice(raw: Mapping[str, Any]) -> dict:
    name = raw["family"]
    row = CATALOG["families"][name]
    return {"name": name, **choice("user:stance/family", f"the lane declares {name}: {row.get('when')}",
                                   _family_alternatives(name))}


def _family_alternatives(name: str) -> list[dict]:
    """The catalog's alternatives of a family, each read as {name, when}; a family whose row
    names none (the floors) is weighed against the first network of its shape."""
    families = CATALOG["families"]
    out = []
    for text in families[name].get("alternatives") or []:
        hit = next((f for f in sorted(families, key=len, reverse=True) if str(text).startswith(f)), None)
        if hit and hit != name:
            rest = str(text)[len(hit):].strip(" ()-") or "as the ablation the catalog names"
            out.append({"name": hit, "when": rest})
    if not out:
        shape = families[name].get("sample_shape")
        other = FLOOR_ALTERNATIVE.get(name) or next(
            f for f, row in families.items() if row.get("sample_shape") == shape and f != name)
        out.append({"name": other, "when": f"when the floor {name} is beaten and capacity is the question"})
    return out[:2]


# --------------------------------------------------------------------------- blocks
# a capacity rung: the parameter a block's width lives in, and its default
_DOUBLED = {"dense": ("units", 32), "rnn": ("hidden", 32), "gru": ("hidden", 32), "lstm": ("hidden", 32),
            "conv1d": ("channels", 16), **{kind: ("out", 32) for kind in CONV_TYPES}}
# a reduction rung: the parameter that names the reduction, and the one it is swapped for
_REDUCED = {"pool": "kind", "pool1d": "kind", "graph_pool": "readout", "positional": "kind"}
_SWAP = {"max": "mean", "mean": "max", "sum": "mean", "avg": "max", "cls": "mean",
         "sinusoidal": "learned", "learned": "sinusoidal"}


def _more_dropout(p: Mapping[str, Any]) -> dict:
    return {"params": {"p": round(min(0.6, float(p.get("p", 0.1)) + 0.1), 2)}, "when": "if train is far above valid"}


def _smaller_embedding(p: Mapping[str, Any]) -> dict:
    dims = {c: max(2, int(d) // 2) for c, d in (p.get("dims") or {}).items()}
    return {"params": {"dims": dims} if dims else {"dim": 4}, "when": "if the embedding overfits"}


def _other_norm(kind: str) -> dict:
    return {"type": "layernorm" if kind == "batchnorm" else "batchnorm", "when": "if the batch statistics are unreliable"}


_ALTERNATIVE = {
    "dropout": _more_dropout, "edge_dropout": _more_dropout, "embedding": _smaller_embedding,
    "activation": lambda p: {"params": {"name": "relu" if p.get("name") != "relu" else "gelu"},
                             "when": "as an ablation of the non-linearity"},
    "residual": lambda p: {"params": {"repeat": int(p.get("repeat", 1)) + 1},
                           "when": "if valid loss still falls at the last epoch"},
    "transformer_encoder": lambda p: {"params": {"layers": int(p.get("layers", 1)) + 1}, "when": UNDERFITS},
    "feature_tokenizer": lambda p: {"params": {"d": max(8, int(p.get("d", 32)) // 2)}, "when": "if the tokens overfit"},
    "attention": lambda p: {"type": "pool", "params": {"kind": "mean"}, "when": "if the attention adds nothing over a mean"},
    "attention_pool": lambda p: {"type": "pool", "params": {"kind": "mean"}, "when": "if every step matters equally"},
    "reparameterize": lambda p: {"params": {"z": int(p.get("z", 4)) * 2}, "when": "if the reconstruction is too coarse"},
    "decoder": lambda p: {"params": {"widths": [int(w) * 2 for w in p.get("widths") or [16]]},
                          "when": "if the reconstruction is too coarse"},
    "batchnorm": lambda p: _other_norm("batchnorm"), "layernorm": lambda p: _other_norm("layernorm"),
}


def block_alternative(block: Mapping[str, Any]) -> dict:
    """The alternative a block's own parameters suggest (a capacity, reduction or kind rung)."""
    kind, p = block.get("type"), dict(block.get("params") or {})
    if kind in _DOUBLED:
        key, default = _DOUBLED[kind]
        return {"params": {key: int(p.get(key, default)) * 2}, "when": UNDERFITS}
    if kind in _REDUCED:
        key = _REDUCED[kind]
        return {"params": {key: _SWAP.get(str(p.get(key)), "mean")}, "when": "as an ablation of the reduction"}
    rule = _ALTERNATIVE.get(kind)
    return rule(p) if rule else {"type": "dense", "when": f"if the {kind} block adds nothing over one dense layer"}


def blocks(raw: Mapping[str, Any], contract: Contract, target: Target, task_source: str) -> list[dict]:
    """The lane's blocks, each with its choice: the head sized by the task row, concat as
    structure, the custom block as the lane's, every other block the family's or the lane's."""
    family = raw["family"]
    allowed = set(CATALOG["families"][family].get("block_types") or ())
    out = copy.deepcopy(list(raw.get("blocks") or ()))
    for block in out:
        kind, params = block["type"], block.setdefault("params", {})
        if kind == "head":
            block.update(_head_choice(block, out, target, contract, task_source))
        elif kind == "concat":
            block.update(choice("vega:default", "structural: the numeric features are joined to "
                                "the embeddings before the first block that reads them", [], "blocks/concat"))
        elif kind == "custom":
            block.setdefault("source", "user:stance/custom-gain")
            block.update({"rationale": f"the lane's custom block ({params.get('code_ref')}): "
                          "the custom path proved end to end", "alternatives": [block_alternative(block)]})
        elif kind in allowed:
            block.update(choice("vega:default", f"the {family} composition "
                                f"({CATALOG['families'][family]['composition']}) at the lane's sizes",
                                [block_alternative(block)], f"families/{family}"))
        else:
            block.update(choice("user:stance/blocks", f"the lane composes {kind} beyond the "
                                f"{family} family's canonical blocks, to prove the block",
                                [block_alternative(block)]))
    return out


def _head_choice(head: dict, all_blocks: Sequence[Mapping[str, Any]], target: Target, contract: Contract,
                 task_source: str) -> dict:
    params = head.setdefault("params", {})
    z = next((b["params"].get("out") for b in all_blocks if b.get("type") == "head"), None)
    want = head_out(target, contract, z if isinstance(z, int) else None)
    if params.get("out") in ("recon_dim", None):
        params["out"] = want
    return choice(task_source, f"out {params['out']} for the {target.effective} task "
                  f"({CATALOG['tasks'][target.effective]['head']['out']}); the loss "
                  "applies the link, never the head", [])


def _first_dropout(block: Mapping[str, Any]) -> float | None:
    if block.get("type") == "dropout":
        return float((block.get("params") or {}).get("p", 0.1))
    for inner in (block.get("params") or {}).get("inner") or []:
        p = _first_dropout(inner)
        if p is not None:
            return p
    return None


def _dropout_blocks(all_blocks: Sequence[Mapping[str, Any]]) -> list[tuple[str, float]]:
    """(top-level block id, p) of every block that is or holds a dropout."""
    found = ((block["id"], _first_dropout(block)) for block in all_blocks)
    return [(bid, p) for bid, p in found if p is not None]


# --------------------------------------------------------------------------- objective
def loss(raw: Mapping[str, Any], target: Target, balance_source: str | None) -> dict:
    spec_loss = raw["loss"]
    name, params = spec_loss["name"], dict(spec_loss.get("params") or {})
    effective = target.effective
    row = TASK_ROWS[effective]
    family_loss = CATALOG["families"][raw["family"]].get("loss")
    if params.get("pos_weight") is not None and balance_source:
        source, rows = balance_source, None
        why = "the positive class is rare on train: its errors are weighted by the inverse ratio"
    elif name == row.default_loss or name == family_loss:
        source = "vega:default"
        rows = [f"tasks/{effective}"] if name == row.default_loss else [f"families/{raw['family']}"]
        rows += ["defaults/label_smoothing"] if "label_smoothing" in params else []
        why = f"the {effective} task's loss" + (" with label smoothing" if "label_smoothing" in params else "")
    else:
        source, rows = "user:stance/loss", None
        why = f"the lane declares {name} for the {effective} task"
    others = [n for n in TASK_ROWS[target.kind].losses_allowed if n != name] or \
        [n for n in row.losses_allowed if n != name]
    alternative = ({"name": others[0], "when": "as an ablation of the objective"} if others else
                   {"name": name, "params": {}, "when": "without the weighting, as an ablation"})
    return {"name": name, "params": params, **choice(source, why, [alternative], rows)}


def metrics(target: Target, labelled: bool) -> dict:
    effective = target.effective
    row = TASK_ROWS[effective]
    primary, secondary = row.primary, list(row.secondary)
    if effective == "anomaly" and not labelled:
        primary, secondary = (row.primary_alt or {}).get("metric", "anomaly_score_mean"), []
    mode = CATALOG["metric_direction"].get(primary, "max")
    alt = (row.primary_alt or {}).get("metric") if labelled else None
    alt = alt if alt and alt != primary else (secondary[0] if secondary else None)
    alternatives = [{"primary": alt, "when": "if the task's balance or reading changes"}] if alt else \
        [{"primary": primary, "secondary": [], "when": "as the only reading without a label"}]
    return {"primary": primary, "secondary": secondary,
            "selection": {"monitor": f"valid.{primary}", "mode": mode},
            **choice("vega:default", f"the {effective} task row's primary and secondary metrics; "
                     "selection reads the primary on valid", alternatives, f"tasks/{effective}")}


# --------------------------------------------------------------------------- optimisation
def _named(value: Any) -> tuple[str, dict]:
    if isinstance(value, str):
        return value, {}
    return value["name"], dict(value.get("params") or {})


# the step each optimiser takes (netspec-contract §1 `optimizer.update_rule`), in the contract's
# notation and in the form the backend's torch.optim classes implement: g the gradient, η the
# learning rate, λ the weight decay, m̂ and v̂ the bias-corrected moments
_MOMENTS = "m ← β₁m + (1−β₁)g; v ← β₂v + (1−β₂)g²"
UPDATE_RULES = {
    "adamw": _MOMENTS + "; θ ← θ − η·(m̂/(√v̂+ε) + λθ)",
    "adam": _MOMENTS + "; θ ← θ − η·m̂/(√v̂+ε)",
    "radam": _MOMENTS + "; θ ← θ − η·r·m̂/(√v̂+ε) once the variance of v̂ is tractable (ρ > 5), "
                        "θ ← θ − η·m̂ before",
    "rmsprop": "v ← αv + (1−α)g²; θ ← θ − η·g/(√v+ε)",
    "sgd": "θ ← θ − η·g",
}


def update_rule(name: str, params: Mapping[str, Any]) -> str:
    """The update rule of optimiser `name` with its params: a weight decay other optimisers than
    adamw fold into the gradient (g ← g + λθ), and sgd's momentum (b ← μb + g; θ ← θ − η·b)."""
    if name == "sgd" and float(params.get("momentum") or 0.0):
        rule = "b ← μb + g; θ ← θ − η·b"
    else:
        rule = UPDATE_RULES[name]
    if name != "adamw" and float(params.get("weight_decay") or 0.0):
        rule = "g ← g + λθ; " + rule
    return rule


def optimizer(raw: Mapping[str, Any]) -> dict:
    name, params = _named(raw["optimizer"])
    params = {**OPTIMIZER_PARAMS.get(name, {}), **params}
    if name == "adamw":
        node = choice("vega:default", "adamw: robust across widths without tuning, with decoupled decay",
                      [{"name": "sgd", "params": {"lr": 0.01, "momentum": 0.9},
                        "when": "if adamw generalises worse (an ablation)"}], "defaults/optimizer")
    else:
        node = choice("user:stance/optimizer", f"the lane declares {name}",
                      [{"name": "adamw", "params": OPTIMIZER_PARAMS["adamw"],
                        "when": "the catalog's default, as the comparison"}])
    return {"name": name, "params": params, "update_rule": update_rule(name, params), **node}


def schedule(raw: Mapping[str, Any], budget: Budget, lr: float) -> dict:
    name, params = _named(raw["schedule"])
    default = "cosine" if budget.max_epochs >= 30 else "plateau"
    catalog_params = {"cosine": {"warmup_epochs": 3} if budget.max_epochs >= 30 else {"warmup_epochs": 1},
                      "plateau": {"factor": 0.5, "patience": 5},
                      "onecycle": {"max_lr": round(lr * 10, 6)}, "none": {}}
    params = {**catalog_params.get(name, {}), **params}
    if name == default:
        node = choice("vega:default", f"{name} for a run of {budget.max_epochs} epochs",
                      [{"name": "none", "when": "as an ablation"}], "defaults/schedule")
    else:
        node = choice("user:stance/schedule", f"the lane declares {name} for its {budget.max_epochs} epochs",
                      [{"name": default, "when": f"the catalog's default for {budget.max_epochs} epochs"}])
    return {"name": name, "params": params, **node}


def init(raw: Mapping[str, Any]) -> dict:
    scheme = raw["init"] if isinstance(raw["init"], str) else raw["init"]["name"]
    other = "xavier_uniform" if scheme != "xavier_uniform" else "kaiming_uniform"
    if scheme in INIT_DEFAULTS:
        node = choice("vega:default", f"{scheme} keeps the activations' variance at depth",
                      [{"scheme": other, "when": "if the non-linearity changes family"}], "defaults/init")
    else:
        node = choice("user:stance/init", f"the lane declares {scheme}",
                      [{"scheme": other, "when": "the catalog's default, as the comparison"}])
    return {"scheme": scheme, "bias": "zeros", **node}


def regularization(all_blocks: Sequence[Mapping[str, Any]], opt: Mapping[str, Any]) -> list[dict]:
    wd = float((opt.get("params") or {}).get("weight_decay", CATALOG["defaults"]["weight_decay"]["value"]))
    out = [{"kind": "weight_decay", "value": wd,
            **choice("vega:default", "mild shrinkage; nothing in the analysis asks for more",
                     [{"value": wd * 10, "when": "if train is far above valid"}], "defaults/weight_decay")}]
    drop = _dropout_blocks(all_blocks)
    if drop:
        out.append({"kind": "dropout", "value": drop[0][1], "where": [b for b, _ in drop],
                    **choice("vega:default", "the dropout the lane's blocks carry, declared once",
                             [{"value": round(min(0.6, drop[0][1] + 0.2), 2),
                               "when": "if the gap between train and valid stays wide"}], "defaults/dropout")})
    out.append({"kind": "grad_clip", "value": float(CATALOG["defaults"]["grad_clip"]["value"]),
                **choice("vega:default", "a cheap guard against the non-finite epoch",
                         [{"value": None, "when": "as an ablation"}], "defaults/grad_clip")})
    return out


#: the block param a dropout declaration sets (vega-forge vega_regularization.DROPOUT_PARAM)
_RATE_PARAM = {"dropout": "p", "transformer_encoder": "dropout", "rnn": "dropout", "gru": "dropout",
               "lstm": "dropout"}


def _drop_rates(block: dict) -> None:
    params = block.get("params") or {}
    key = _RATE_PARAM.get(block.get("type"))
    layers = int(params.get("num_layers", params.get("layers", 1)) or 1)
    if key and (block.get("type") not in ("rnn", "gru", "lstm") or layers >= 2):
        params.pop(key, None)
    for inner in params.get("inner") or []:
        _drop_rates(inner)


def declared_once(all_blocks: Sequence[Mapping[str, Any]], opt: Mapping[str, Any],
                  regs: Sequence[Mapping[str, Any]]) -> tuple[list, dict]:
    """(blocks, optimizer) without the copies the declaration covers (netspec-contract §3, R2):
    the emitted code reads `regularization[]`, so the optimizer's `weight_decay` and the rate of
    every block a dropout declaration covers are stated there once — a copy beside them would be
    refused the moment a rung moved the declaration."""
    blocks, opt = copy.deepcopy(list(all_blocks)), copy.deepcopy(dict(opt))
    kinds = {r["kind"]: r for r in regs}
    if "weight_decay" in kinds:
        opt["params"].pop("weight_decay", None)
    where = set((kinds.get("dropout") or {}).get("where") or ())
    for block in blocks:
        if block.get("id") in where:
            _drop_rates(block)
    return blocks, opt


def early_stopping(chosen: Mapping[str, Any], budget: Budget) -> dict:
    return {"monitor": chosen["selection"]["monitor"], "mode": chosen["selection"]["mode"],
            "patience": budget.patience, "min_delta": 0.001,
            **choice("user:stance/lane-budget", f"{budget.phrase()}: patience {budget.patience} "
                     f"within {budget.max_epochs} epochs, on the primary metric selection reads",
                     [{"params": {"patience": 10}, "when": "on a full-length campaign"}])}


RUNTIME_ROWS = ("defaults/device", "defaults/precision")


def _batch(contract: Contract, sample: Mapping[str, Any]) -> tuple[int | str, str, str | None, str]:
    """(batch_size, source, catalog row, why): `"full"` under full batching, sourced as the
    sample's batching is; else the catalog's rule over the training targets (defaults/batch_size)."""
    if sample.get("batching") == "full":
        return ("full", sample.get("batching_source"), sample.get("batching_catalog"),
                "one batch is the whole graph view: full-batch gradients are exact while it fits "
                f"({sample.get('batching_catalog') or sample.get('batching_source')})")
    unit = {"sampled": "seed nodes", "graphs": "graphs"}.get(sample.get("batching"),
                                                             "windows" if sample["shape"] == "window" else "rows")
    return (batch_rule(contract.targets), "vega:default", "defaults/batch_size",
            f"clip({contract.targets} training {unit} / 8, 16, 512) rounded to a power of two "
            "(defaults/batch_size)")


def training(contract: Contract, sample: Mapping[str, Any], budget: Budget) -> dict:
    batch, source, row, why = _batch(contract, sample)
    alternative = ({"batch_size": 256, "when": "if the view outgrows memory (then batching sampled)"}
                   if batch == "full" else {"batch_size": batch * 2, "when": "if the loss thrashes between batches"})
    rationale = (f"the batch: {why}; device and precision the catalog's; {budget.max_epochs} epochs and "
                 f"{budget.wall_clock_min} minutes a run are {budget.phrase()} (user:stance/lane-budget)")
    node = choice(source, rationale, [alternative], [r for r in (row, *RUNTIME_ROWS) if r])
    return {"batch_size": batch, "max_epochs": budget.max_epochs, "device": "auto", "precision": "fp32",
            "num_workers": 0, "budget": {"wall_clock_min": budget.wall_clock_min}, **node}
