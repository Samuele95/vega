"""
architecture_sample — the `architecture` layer's sample record (D1 §16, E20): what one
sample is by the spec's declaration, and what the smoke pass measured of it.

The shape and its parameters (a window's length, stride, horizon, context, autoregressive flag;
a graph's level, protocol, batching, fan-outs) are the spec's; every count (the series, the
windows of train and valid, the valid rows left unscored; a graph's training and valid views,
the test edges withheld, the edges into test) is the smoke record's `sample` block, which
`emit_backend.py --smoke` writes. Nothing here computes a count.

Two scripts share it, so the rule is stated once:

  * `apply_architecture_layer.py` refuses a `sample` block measured on another spec
    (`sample_errors`) and writes the sample into the <noscript> fallback (`sample_line`);
  * `smoke_architecture.py` derives what the render's sample panel must show
    (`expectations`) and holds the panel to it (`panel_checks`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

# the sample parameters a spec declares that the smoke pass's `sample` block measures again
# (D1 §16), per shape; and the unit `madds_per_sample` counts per (emit_backend._units)
SAMPLE_DECLARED = {"window": ("length", "stride", "horizon", "context", "autoregressive"),
                   "graph": ("level", "protocol", "batching")}
SAMPLE_COUNTS = {"window": ("series", "windows", "unscored"),
                 "graph": ("train_view", "valid_view", "test_edges_withheld")}
MADDS_UNIT = {"row": "row", "window": "anchor", "node": "target_node", "graph": "graph"}
# the panel's counts, by the data-k the render gives them: only a smoke record puts them there
PANEL_COUNTS = {"window": ("series", "windows_train", "windows_valid", "unscored_valid"),
                "graph": ("train_nodes", "train_edges", "train_targets", "valid_nodes",
                          "valid_edges", "valid_targets", "test_edges_withheld",
                          "edges_into_test")}
# what the smoke panel says `madds_per_sample` counts per (the render's UNIT)
UNIT_WORDS = {"row": "row", "anchor": "anchor (one window)", "target_node": "target node",
              "graph": "graph"}


def declared_sample(spec: dict) -> dict:
    """The sample the spec declares (`data_contract.sample`; a bare `sample` is read too)."""
    sample = spec.get("sample") or (spec.get("data_contract") or {}).get("sample")
    return sample if isinstance(sample, dict) else {}


def measured_sample(smoke: dict) -> dict | None:
    """The smoke record's `sample` block, or None when it has none."""
    sample = smoke.get("sample")
    return sample if isinstance(sample, dict) else None


def _value(declared: dict, measured: dict | None) -> Callable[[str], object]:
    """A parameter the way the render picks it: the smoke record's when measured, else the
    spec's declaration."""
    def value(key: str):
        got = (measured or {}).get(key)
        return got if got is not None else declared.get(key)
    return value


# ================================================================ apply: one sample, not two

def _madds_unit(shape: str, level) -> str | None:
    if shape == "graph":
        return MADDS_UNIT["graph" if level == "graph" else "node"]
    return MADDS_UNIT.get(shape)


def sample_errors(spec: dict, smoke: dict) -> list:
    """The sample panel shows the smoke pass's `sample` counts beside the spec's declaration, so
    the two must describe one sample: a record measured on another spec would put another
    network's windows or views on this page. A record without the block is no error here (the
    panel says the counts are missing; apply's `main` says it as a WARN)."""
    declared = declared_sample(spec)
    shape = declared.get("shape") or "row"
    measured = smoke.get("sample")
    if measured is None:
        return []
    if not isinstance(measured, dict):
        return ["smoke.sample is not an object; emit_backend.py --smoke writes it (D1 §16)"]
    fix = " — the smoke record was not measured on this spec; re-run emit_backend.py --smoke"
    if measured.get("shape") != shape:
        return [f"smoke.sample.shape is {measured.get('shape')!r} but the spec declares a "
                f"{shape!r} sample{fix}"]
    errs = [f"smoke.sample.{key} is {measured.get(key)!r} but the spec declares "
            f"{declared[key]!r}{fix}" for key in SAMPLE_DECLARED.get(shape, ())
            if declared.get(key) is not None and measured.get(key) != declared[key]]
    missing = [key for key in SAMPLE_COUNTS.get(shape, ()) if key not in measured]
    if missing:
        errs.append(f"smoke.sample lacks {', '.join(missing)}: the {shape} panel's counts are the "
                    "smoke pass's (D1 §16); re-run emit_backend.py --smoke")
    level = declared.get("level") or measured.get("level")
    unit = _madds_unit(shape, level)
    if smoke.get("madds_unit") is not None and smoke.get("madds_unit") != unit:
        errs.append(f"smoke.madds_unit is {smoke.get('madds_unit')!r} but a {shape} sample"
                    + (f" at {level} level" if shape == "graph" else "")
                    + f" counts per {unit!r}{fix}")
    return errs


def _window_line(value, measured: dict) -> str:
    horizon = value("horizon") or 0
    line = (f"a window of {value('length')} rows, stride {value('stride')}"
            + (f", horizon {horizon}" if horizon else "")
            + f", context {value('context')}, autoregressive "
            + ("yes" if value("autoregressive") else "no"))
    if measured:
        windows = measured.get("windows") or {}
        line += (f"; {measured.get('series')} series, {windows.get('train')} training and "
                 f"{windows.get('valid')} valid windows")
    return line


def _graph_line(value, measured: dict) -> str:
    line = (f"a graph sample at {value('level')} level, protocol {value('protocol')}, "
            f"batching {value('batching')}")
    view = measured.get("train_view") or {}
    if view:
        what = "graphs" if value("level") == "graph" else "targets"
        line += (f"; the training view holds {view.get('nodes')} nodes, {view.get('edges')} "
                 f"edges and {view.get('targets')} {what}, "
                 f"{measured.get('test_edges_withheld')} test edges withheld")
    if value("protocol") == "transductive":
        line += "; exposure disclosed: the test nodes' features were seen in training"
    return line


def sample_line(spec: dict, smoke: dict) -> str:
    """The sample in one sentence, for the <noscript> fallback."""
    declared, measured = declared_sample(spec), measured_sample(smoke) or {}
    shape = declared.get("shape") or measured.get("shape") or "row"
    value = _value(declared, measured)
    if shape == "window":
        return _window_line(value, measured)
    if shape == "graph":
        return _graph_line(value, measured)
    return "one row of the shaped part"


# ================================================================ smoke: what the panel shows

def shown(value) -> str:
    """A declared value the way the render writes it (its fmt): absent is a middle dot."""
    if value is None or value == "":
        return "·"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def flat(text) -> str:
    """Text with every blank removed: the render groups digits with spaces."""
    return re.sub(r"\s+", "", str(text or ""))


def unscored_note(context, horizon) -> str:
    """The caption under the valid part's unscored count (the render's unscoredNote). A valid
    row gets no window for too little context before it or, on a forecast, a horizon past the
    part's end (D1 §9.3); the record counts those rows but not the reason, so the caption names
    every reason the declared context and horizon allow and claims none of them."""
    if context == "none":
        return "too little context before them" + (
            ", or a horizon past the part’s end" if horizon else " for a window")
    if context == "previous-part":
        return ("a horizon past the part’s end, or too little history in their series"
                if horizon else "too little history in their series for a window")
    return "no scorable window"


@dataclass
class Want:
    """What the sample panel must show: its shape, the text of each data-k (values) and of
    each stat's caption (notes), the exposure chips and the edges into test they name."""
    shape: str
    values: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    exposure: int = 0
    edges_into_test: object = None


def _window_want(want: Want, value, measured: dict | None) -> None:
    want.values.update(length=shown(value("length")), stride=shown(value("stride")),
                       context=f"context {shown(value('context'))}",
                       autoregressive="autoregressive " + ("yes" if value("autoregressive")
                                                           else "no"))
    if value("horizon"):
        want.values["horizon"] = shown(value("horizon"))
    if not measured:
        return
    windows, unscored = measured.get("windows") or {}, measured.get("unscored") or {}
    want.values.update(series=shown(measured.get("series")),
                       windows_train=shown(windows.get("train")),
                       windows_valid=shown(windows.get("valid")),
                       unscored_valid=shown(unscored.get("valid")))
    want.notes["unscored_valid"] = unscored_note(value("context"), value("horizon") or 0)


def _graph_want(want: Want, value, declared: dict, measured: dict | None) -> None:
    want.values.update(level=f"level {shown(value('level'))}",
                       protocol=f"protocol {shown(value('protocol'))}",
                       batching=f"batching {shown(value('batching'))}")
    fan_outs = declared.get("neighbours")
    if isinstance(fan_outs, list) and fan_outs:
        want.values["neighbours"] = "fan-out " + " × ".join(str(k) for k in fan_outs)
    if measured:
        for view in ("train", "valid"):
            counts = measured.get(f"{view}_view") or {}
            for key in ("nodes", "edges", "targets"):
                want.values[f"{view}_{key}"] = shown(counts.get(key))
        want.values["test_edges_withheld"] = shown(measured.get("test_edges_withheld"))
    if value("protocol") == "transductive":
        want.exposure = 1
        want.edges_into_test = ((measured or {}).get("exposure") or {}).get("edges_into_test")
        if want.edges_into_test is not None:
            want.values["edges_into_test"] = shown(want.edges_into_test)


def expectations(spec: dict, smoke: dict) -> Want:
    """What the sample panel must show for this spec and smoke record. Declared values are the
    smoke record's when it measured them, else the spec's; counts only ever the smoke
    record's."""
    declared, measured = declared_sample(spec), measured_sample(smoke)
    shape = declared.get("shape") or (measured or {}).get("shape") or "row"
    value = _value(declared, measured)
    want = Want(shape, {"shape": f"shape {shape}"})
    if shape == "window":
        _window_want(want, value, measured)
    elif shape == "graph":
        _graph_want(want, value, declared, measured)
    elif smoke.get("rows_train") is not None:
        want.values["rows_train"] = shown(smoke.get("rows_train"))
    return want


@dataclass(frozen=True)
class Case:
    """One smoke pass over the layered file: its label, the layer's spec and smoke record, and
    whether this pass's copy lost its measurements (the `sample` block removed)."""
    label: str
    spec: dict
    smoke: dict
    degraded: bool = False

    def record(self) -> dict:
        """The smoke record this pass's copy of the file carries."""
        if self.degraded:
            return {k: v for k, v in self.smoke.items() if k != "sample"}
        return self.smoke


def _shown_checks(check, c: dict, case: Case, want: Want) -> None:
    """Each value and caption the record calls for, exactly; and without a record, no count."""
    label, got, notes = case.label, c.get("sample_values") or {}, c.get("sample_notes") or {}
    for key, text in want.values.items():
        check(flat(got.get(key)) == flat(text), f"{label}:sample-{key}",
              f"shows {got.get(key)!r}, the record says {text!r}")
    for key, text in want.notes.items():
        check(notes.get(key) == text, f"{label}:sample-{key}-caption",
              f"captions it {notes.get(key)!r}; the declaration allows only {text!r}")
    if want.shape == "window" and "horizon" not in want.values:
        check("horizon" not in got, f"{label}:sample-no-horizon-off-forecast",
              f"shows horizon {got.get('horizon')!r}")
    if measured_sample(case.record()) is None and want.shape in PANEL_COUNTS:
        stray = sorted(k for k in PANEL_COUNTS[want.shape] if k in got)
        check(not stray, f"{label}:sample-no-counts-without-a-record",
              "shows " + ", ".join(f"{k} {got[k]!r}" for k in stray))


def _hook_check(check, c: dict, case: Case) -> None:
    hooked = c.get("hook_sample")
    same = (isinstance(hooked, dict)
            and json.dumps(hooked.get("measured"), sort_keys=True)
            == json.dumps(measured_sample(case.record()), sort_keys=True)
            and json.dumps(hooked.get("declared"), sort_keys=True)
            == json.dumps(declared_sample(case.spec), sort_keys=True))
    check(same, f"{case.label}:hook-sample", str(hooked)[:200])


def panel_checks(check, c: dict, case: Case) -> None:
    """The sample panel (E20): mounted beside the diagram, the spec's shape, exactly the smoke
    record's sample (or, degraded, the spec's declaration and a statement of what is missing,
    and no count). `c` is the probe's checks for this pass."""
    label, smoke = case.label, case.record()
    want = expectations(case.spec, smoke)
    check(c.get("sample_mounted"), f"{label}:sample-panel-mounted")
    check(c.get("sample_beside"), f"{label}:sample-panel-beside-the-diagram")
    check(c.get("sample_shape") == want.shape, f"{label}:sample-panel-shape",
          f"{c.get('sample_shape')} != {want.shape}")
    _shown_checks(check, c, case, want)
    if want.shape == "window":
        check(c.get("sample_strip") == 1, f"{label}:sample-window-strip",
              str(c.get("sample_strip")))
    check(c.get("sample_exposure") == want.exposure, f"{label}:sample-exposure-chip",
          f"{c.get('sample_exposure')} chip(s), {want.exposure} expected (transductive only)")
    if want.exposure and want.edges_into_test is not None:
        check(flat(want.edges_into_test) in flat(c.get("sample_text")),
              f"{label}:sample-exposure-names-the-edges-into-test")
    missing = measured_sample(smoke) is None
    check(bool(c.get("sample_missing")) == missing, f"{label}:sample-missing-said-iff-missing",
          f"missing statement {'absent' if missing else 'shown'}")
    unit = smoke.get("madds_unit")
    if unit:
        check(f"mult-adds / {UNIT_WORDS.get(unit, unit)}" in (c.get("smoke_text") or ""),
              f"{label}:madds-unit-named", unit)
    _hook_check(check, c, case)
