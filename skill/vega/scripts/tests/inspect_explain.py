"""
inspect_explain.py — the explanation invariants of D6 §8.2 that read one explanation alone:
EI5 (the explained quantity's gradient at the head is the analytic one), EI6 (the attribution's
shape), EI7 (the attribution's properties) and EI9 (the readout's probabilities). Pure
functions over `vega/explanation@2` and `vega/network-view@2` payloads; each returns a list of
`(ok, detail)` observations, so `inspect_check.py` records them and the unit test asserts them.

    ei5(view, exact, explanation)          the head's exact grad_out against the target's
                                           analytic gradient (one-hot, the sign of a binary
                                           logit, σ(z)(1−σ(z)) of an expected rank, the
                                           reconstruction error's, 2(z − c) of an SVDD distance)
    ei6(network, explanation)              one attribution entry per input column (row), a
                                           steps × channels matrix plus the nominal columns per
                                           step (window), the target's features, its
                                           neighbours and its edges named (graph)
    ei7(network, explanation)              every number finite; grad × input: attribution =
                                           value · grad, and 0 where the shaped value is 0
    ei9(explanation)                       read = link(raw): softmax sums to one, σ per label
                                           never renormalised, the served probabilities agree

The reconstruction error is inspect-contract §4.4's: the mean squared error over the numeric
columns plus the mean cross-entropy over the embedded nominals, `recon_dim` laying the numeric
columns first, then each nominal's `cardinality` logits in contract order.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from check_kit import finite

__all__ = ["GRAD_ATOL", "GRAD_RTOL", "analytic_gradient", "ei5", "ei6", "ei7", "ei9",
           "head_of", "numbers"]

GRAD_RTOL, GRAD_ATOL = 1e-4, 1e-5
PROB_TOL = 1e-6
ONE_HOT = frozenset({"logit", "label_logit", "threshold_logit", "forecast_step", "quantile",
                     "value"})


def head_of(view) -> str | None:
    """The node the readout reads: the network's last node before `output`."""
    return next((s for s, k in view.into.get("output", ()) if k == "readout"), None)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def _within(got, want, rtol: float = GRAD_RTOL, atol: float = GRAD_ATOL) -> tuple[bool, float]:
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    if got.shape != want.shape:
        return False, math.inf
    gap = np.abs(got - want)
    scale = max(1.0, float(np.abs(want).max())) if want.size else 1.0
    ok = bool(np.isfinite(gap).all() and (gap <= atol * scale + rtol * np.abs(want)).all())
    return ok, float(gap.max()) if gap.size else 0.0


# --------------------------------------------------------------------------- EI5
def _recon_parts(network: Mapping, explanation: Mapping) -> tuple:
    """(numeric x, nominal [(id, cardinality)]) of the explained row, from its shaped inputs
    (`inputs.order` is numeric, then nominal)."""
    inputs = network.get("inputs") or {}
    numeric, nominal = list(inputs.get("numeric") or ()), list(inputs.get("nominal") or ())
    shaped = np.asarray((explanation.get("inputs") or {}).get("shaped"), dtype=np.float64)
    x = shaped[:len(numeric)]
    ids = [(int(shaped[len(numeric) + j]), int(n["cardinality"])) for j, n in enumerate(nominal)]
    return x, ids


def _recon_gradient(z, network, explanation, feature: int | None) -> np.ndarray:
    x, nominal = _recon_parts(network, explanation)
    n_num, g = len(x), np.zeros_like(z)
    if feature is None and n_num:
        g[:n_num] = 2.0 * (z[:n_num] - x) / n_num
    elif feature is not None and feature < n_num:
        g[feature] = 2.0 * (z[feature] - x[feature])
    at = n_num
    for j, (ident, card) in enumerate(nominal):
        if feature is None or feature == n_num + j:
            ce = _softmax(z[at:at + card])
            ce[ident] -= 1.0
            g[at:at + card] = ce / (len(nominal) if feature is None else 1)
        at += card
    return g


def analytic_gradient(view, exact, explanation: Mapping) -> np.ndarray | None:
    """∂s/∂(head output) of the explanation's target, from the head's own output, or None
    when the target kind is not one D6 §8.2 names."""
    head = head_of(view)
    z = exact.get(head, "out")
    if z is None:
        return None
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    target = explanation.get("target") or {}
    kind, index = target.get("kind"), int(target.get("index") or 0)
    sign = float(target.get("sign") or 1)
    if kind in ONE_HOT or kind == "binary_logit":
        g = np.zeros_like(z)                 # a binary head has one column: s = sign · z
        g[0 if kind == "binary_logit" else index] = sign
        return g
    if kind == "expected_rank":
        p = _sigmoid(z)
        return p * (1.0 - p)
    if kind == "svdd_distance":
        centre = view.tensors_of(head).get("centre")
        return None if centre is None else 2.0 * (z - np.asarray(centre).reshape(-1))
    if kind in ("reconstruction_error", "feature_error"):
        return _recon_gradient(z, view.network, explanation,
                               index if kind == "feature_error" else None)
    return None


def ei5(view, exact, explanation: Mapping) -> list:
    head = head_of(view)
    got = exact.get(head, "grad_out") if head else None
    want = analytic_gradient(view, exact, explanation)
    kind = (explanation.get("target") or {}).get("kind")
    if got is None or want is None:
        return [(False, f"target {kind}: no exact grad_out at the head, or no analytic "
                        "gradient for it")]
    ok, gap = _within(np.asarray(got).reshape(-1), want)
    return [(ok, f"target {kind}: |∂s/∂out − analytic| = {gap:.3g} at the head {head}")]


# --------------------------------------------------------------------------- EI6
def _channel_names(network: Mapping) -> list:
    labels = ((network.get("inputs") or {}).get("axis_labels") or {}).get("feature") or ()
    return [lab["name"] if isinstance(lab, Mapping) else str(lab) for lab in labels]


def ei6(network: Mapping, explanation: Mapping) -> list:
    att, inputs = explanation.get("attribution") or {}, network.get("inputs") or {}
    shape = att.get("shape")
    names = list((att.get("features") or {}).get("names") or ())
    nominal = [n["name"] for n in inputs.get("nominal") or ()]
    if shape == "row":
        feats = att.get("features") or {}
        lengths = {len(feats.get(k) or ()) for k in ("names", "values", "attribution", "grad")}
        order = [c for c in inputs.get("order") or () if c != inputs.get("past_target")]
        return [(names == order and lengths == {len(order)},
                 f"row: {len(names)} attribution entries for {len(order)} input columns "
                 "(a nominal is one entry)")]
    if shape == "window":
        steps = int(att.get("steps") or 0)
        matrix = np.asarray(att.get("matrix") or [], dtype=np.float64)
        nom = att.get("nominal") or {}
        nom_att = np.asarray(nom.get("attribution") or [], dtype=np.float64)
        length = int((network.get("sample") or {}).get("length") or 0)
        nom_ok = list(nom.get("names") or ()) == nominal and \
            (not nominal or nom_att.shape == (len(nominal), length))
        ok = (steps == length and names == _channel_names(network) and nom_ok and
              matrix.shape == (length, len(names)) and len(att.get("per_step") or ()) == length
              and len(att.get("per_feature") or ()) == len(names))
        return [(ok, f"window: matrix {matrix.shape} for {length} steps × {len(names)} "
                     f"channels, {len(nominal)} nominal column(s) per step")]
    if shape == "graph":
        return _graph_shape(att, names, inputs)
    return [(False, f"attribution shape {shape!r} is not row, window or graph")]


def _graph_shape(att: Mapping, names: list, inputs: Mapping) -> list:
    order = list(inputs.get("order") or ())
    edges = att.get("edges") or []
    neighbours = att.get("neighbours") or []
    named = all({"src", "dst"} <= set(e) for e in edges)
    placed = all({"node", "hop"} <= set(n) for n in neighbours)
    if att.get("level") == "node":
        self_ok = len(att.get("self") or ()) == len(names)
    else:
        self_ok = att.get("self") is None
    ok = names == order and self_ok and named and placed and bool(neighbours)
    return [(ok, f"graph ({att.get('level')}): {len(names)} features, {len(neighbours)} "
                 f"neighbour(s), {len(edges)} edge(s) each naming (src, dst)")]


# --------------------------------------------------------------------------- EI7
def numbers(value: Any):
    """Every number of a JSON value (bools and ids skipped by the caller's choice of keys)."""
    if isinstance(value, Mapping):
        for v in value.values():
            yield from numbers(v)
    elif isinstance(value, list):
        for v in value:
            yield from numbers(v)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield value


def ei7(network: Mapping, explanation: Mapping) -> list:
    att = explanation.get("attribution") or {}
    out = [(all(finite(v) for v in numbers(att)), "every attribution number is finite")]
    if att.get("method") != "grad_x_input":
        return out
    shape, inputs = att.get("shape"), network.get("inputs") or {}
    if shape == "row":
        out.append(_row_product(att, len(inputs.get("numeric") or ())))
    elif shape == "window":
        out.append(_window_zeros(att, explanation))
    elif shape == "graph" and att.get("level") == "node":
        shaped = np.asarray((explanation.get("inputs") or {}).get("shaped"), dtype=np.float64)
        own = np.asarray(att.get("self") or [], dtype=np.float64)
        zero = shaped.reshape(-1)[:own.size] == 0.0
        out.append((bool((own[zero] == 0.0).all()),
                    f"graph: {int(zero.sum())} zero-valued feature(s) of the target attribute 0"))
    return out


def _row_product(att: Mapping, n_numeric: int) -> tuple:
    feats = att.get("features") or {}
    values = np.asarray(feats.get("values") or [], dtype=np.float64)[:n_numeric]
    grad = np.asarray(feats.get("grad") or [], dtype=np.float64)[:n_numeric]
    attr = np.asarray(feats.get("attribution") or [], dtype=np.float64)[:n_numeric]
    ok, gap = _within(attr, values * grad, 1e-5, 1e-6)
    zero = values == 0.0
    ok = ok and bool((attr[zero] == 0.0).all())
    return ok, (f"row: attribution = value · grad on {n_numeric} numeric input(s) (|Δ| "
                f"{gap:.3g}); {int(zero.sum())} zero-valued input(s) attribute 0")


def _window_zeros(att: Mapping, explanation: Mapping) -> tuple:
    inputs = explanation.get("inputs") or {}
    order = list(inputs.get("order") or ())
    shaped = np.asarray(inputs.get("shaped") or [], dtype=np.float64)
    matrix = np.asarray(att.get("matrix") or [], dtype=np.float64)
    names = list((att.get("features") or {}).get("names") or ())
    zeros = 0
    for f, name in enumerate(names):
        if name not in order or shaped.ndim != 2:
            continue
        col = shaped[:, order.index(name)]
        hit = col == 0.0
        zeros += int(hit.sum())
        if matrix.shape[0] != col.shape[0] or not (matrix[hit, f] == 0.0).all():
            return False, f"window: channel {name!r} has a zero-valued step with a non-zero " \
                          "attribution"
    return True, f"window: {zeros} zero-valued cell(s) attribute 0"


# --------------------------------------------------------------------------- EI9
def _reconstruction(raw, network: Mapping, explanation: Mapping) -> tuple:
    """(read, error) of a reconstruction head: x̂ for the numeric columns and the argmax
    category of each nominal's logits; the error of inspect-contract §4.4."""
    x, nominal = _recon_parts(network, explanation)
    n_num, read = len(x), list(raw[:len(x)])
    error = float(np.mean((raw[:n_num] - x) ** 2)) if n_num else 0.0
    at, ce = n_num, []
    for ident, card in nominal:
        logits = raw[at:at + card]
        read.append(float(np.argmax(logits)))
        ce.append(-math.log(max(float(_softmax(logits)[ident]), 1e-300)))
        at += card
    return np.asarray(read), error + (float(np.mean(ce)) if ce else 0.0)


def ei9(explanation: Mapping, positive: str | None = None, network: Mapping | None = None) -> list:
    outputs = explanation.get("outputs") or {}
    link = outputs.get("link")
    raw = np.asarray(outputs.get("raw") or [], dtype=np.float64)
    read = outputs.get("read")
    if link == "distance":
        return [(read is None, "distance: no read (a distance is not a probability)")]
    read = np.asarray(read or [], dtype=np.float64)
    if link == "reconstruction":
        return _ei9_reconstruction(explanation, outputs, raw, read, network or {})
    want = {"softmax": lambda: _softmax(raw), "sigmoid": lambda: _sigmoid(raw),
            "sigmoid_each": lambda: _sigmoid(raw), "cumulative": lambda: _sigmoid(raw)
            }.get(link, lambda: raw)()
    ok, gap = _within(read, want, PROB_TOL, PROB_TOL)
    out = [(ok, f"{link}: read = link(raw) (|Δ| {gap:.3g})")]
    if link == "softmax":
        out.append((abs(float(read.sum()) - 1.0) <= PROB_TOL,
                    f"softmax: Σ read = {read.sum():.9g}"))
    if link in ("sigmoid_each", "sigmoid", "cumulative"):
        out.append((bool(((read >= 0) & (read <= 1)).all()),
                    f"{link}: each read in [0, 1], none renormalised"))
    out += _served_probabilities(explanation, outputs, raw, link, positive)
    return out


def _ei9_reconstruction(explanation, outputs, raw, read, network) -> list:
    want, error = _reconstruction(raw, network, explanation)
    ok, gap = _within(read, want, PROB_TOL, PROB_TOL)
    out = [(ok, f"reconstruction: read = x̂ and each nominal's argmax (|Δ| {gap:.3g})")]
    told = (outputs.get("answer") or {}).get("error")
    if told is not None:
        out.append((abs(float(told) - error) <= GRAD_ATOL * max(1.0, error) + GRAD_RTOL * error,
                    f"reconstruction: the error {told} = mean squared error + mean "
                    f"cross-entropy {error:.9g}"))
    return out


def _served_probabilities(explanation, outputs, raw, link, positive) -> list:
    """The explained row's served probabilities say what the head says: σ(z) for a binary's
    positive class, σ(z_l) per label (no sum constraint), the softmax for multiclass."""
    probs = (explanation.get("prediction") or {}).get("probabilities")
    if not isinstance(probs, Mapping) or not raw.size:
        return []
    names = [str(n) for n in outputs.get("names") or ()]
    if link == "sigmoid" and positive is not None and positive in probs:
        return [(abs(float(probs[positive]) - float(_sigmoid(raw)[0])) <= PROB_TOL,
                 f"binary: served p({positive}) = σ(raw)")]
    if link in ("sigmoid_each", "softmax") and set(names) <= set(probs):
        want = _sigmoid(raw) if link == "sigmoid_each" else _softmax(raw)
        got = np.asarray([float(probs[n]) for n in names])
        ok, gap = _within(got, want, PROB_TOL, PROB_TOL)
        return [(ok, f"{link}: served probabilities = the head's (|Δ| {gap:.3g})")]
    return []
