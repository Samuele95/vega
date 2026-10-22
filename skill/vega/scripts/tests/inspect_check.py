#!/usr/bin/env python3
"""
inspect_check.py — the `/v1/network` and `/v1/explain` invariants (D6 §8: NI1–NI8, EI1–EI12,
BUILD-PLAN R-F5 `detail: "exact"`, R-F8, R-H4) against a bundle's live service.

    python3 inspect_check.py --base-url URL --bundle DIR [--lane ID] [--lanes-file FILE]
                             [--json OUT] [--python PY]

The network is read once (`GET /v1/network`, twice for NI7) and held to the bundle it
describes: its netspec, its architecture record's smoke, its weights.pt (NI1–NI7,
inspect_network.py). Then every preset of `/v1/examples`, the first 5 demo items and the first
5 shipped example rows — at most 20 samples — are explained at `detail: "exact"` and each
explanation is held to:

  EI1   its `prediction` is /v1/predict's answer for the same row
  EI2   every node with an oracle (`oracles.py`, the one float64 oracle module) recomputes from
        the published weights and the node's captured input (dense/head: W·x + b = pre,
        act(pre) = post), within rtol 1e-4, atol 1e-5·max(1, |v|)
  EI3   every node reads its predecessor's captured output (joins and skips included)
  EI4   the last node's output is the model's raw output (and the served raw fields)
  EI5   the explained quantity's gradient at the head is the analytic one (every selectable
        target of the first sample, every class of a classification)
  EI6   the attribution's shape per sample shape
  EI7   the attribution is finite, grad × input holds, a zero input attributes 0; on a linear
        network the attributions sum to s(x) − s(0)
  EI8   the same request twice gives the same bytes
  EI9   the readout: read = link(raw), softmax sums to one, σ per label never renormalised
  EI10  an empty request, an empty row and an unknown target answer 422 naming the field
  EI11  the explained window / receptive field is what the request sent (inspect_receptive.py)
  EI12  every warning /v1/predict gives the row is carried by the explanation

NI8 needs inspection off: on a bundle packed with `--no-inspect` the given service answers it;
otherwise a second service of the same bundle is started with `<ENV>_INSPECT=off` and must
answer the three routes 404 naming inspection.

Exit 0 when every invariant holds or does not apply, 1 when one fails, 2 when the check cannot
run. `--json` writes `vega/inspect-check@1`, with `nodes`, `explained_rows` and
`oracle_checks` (type -> recomputed nodes) for observed.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import inspect_explain as X  # noqa: E402
import inspect_network as N  # noqa: E402
import inspect_receptive as RF  # noqa: E402
import oracles as O  # noqa: E402
from check_kit import (BundleRecords, CheckError, Client, Report, close, dig,  # noqa: E402
                       write_result)
from inspect_context import ReferenceGraph, ServedBatches, StoredWeights  # noqa: E402
from inspect_recompute import ExactPass, NetworkView, Recompute, perturbed  # noqa: E402
from lanekit import contracts as C  # noqa: E402
from lanekit import service as svc  # noqa: E402

RAW_TARGETS = ("logit", "binary_logit", "label_logit", "value", "forecast_step",
               "threshold_logit", "quantile")
COMPLETENESS_TOL = 1e-5
MAX_CLASS_TARGETS = 5
UNKNOWN = "vega-check-unknown"


class InspectCheck:
    """One run of NI1–NI8 and EI1–EI12 against a live service (see the module docstring)."""

    def __init__(self, bundle: BundleRecords, client: Client, report: Report, python: str):
        self.bundle, self.client, self.report, self.python = bundle, client, report, python
        self.oracle_checks: Counter = Counter()
        self.explained = 0
        self.view = None
        self.batches = self.reference = None

    # -- the network (NI1–NI7) ---------------------------------------------------------------
    def network(self) -> None:
        r = self.report
        first, second = self.client.get(C.NETWORK_ROUTE), self.client.get(C.NETWORK_ROUTE)
        for ok, detail in N.ni1(first.status, first.json):
            r.record("NI1", ok, detail)
        if first.status != 200:
            raise CheckError(f"GET {C.NETWORK_ROUTE} answered {first.status}: {first.body[:300]!r}")
        network = first.json
        self.view = NetworkView(network, self._fetch_tensor)
        status = (self.client.json_of(self.client.get(C.HEALTH_ROUTE), C.HEALTH_ROUTE)
                  .get(C.HEALTH_INSPECTION) or {})
        smoke, where = self.bundle.packed_smoke()
        match = self._weights(network)
        for check_id, observations in (
                ("NI2", N.ni2(self.view, self.bundle.netspec)),
                ("NI3", N.ni3(self.view, smoke, where)),
                ("NI4", N.ni4(self.view, smoke, match, where)), ("NI5", N.ni5(self.view, match)),
                ("NI6", N.ni6(network, self.bundle.netspec, self.bundle.sample,
                              self.bundle.output)),
                ("NI7", N.ni7(first.body, second.body, dig(self.bundle.manifest, C.NETWORK_PIN),
                              status))):
            for ok, detail in observations:
                r.record(check_id, ok, detail)

    def _fetch_tensor(self, tensor_id: str) -> Mapping:
        path = C.NETWORK_TENSOR_ROUTE.replace("{id}", tensor_id)
        return self.client.json_of(self.client.get(path), f"GET {path}")

    def _weights(self, network: Mapping):
        published, roles = {}, {}
        for tid, payload in (network.get("tensors") or {}).items():
            if payload.get("encoding") != "withheld":
                published[tid], roles[tid] = self.view.tensor(tid), payload.get("role")
        for node in network.get("nodes") or ():           # an untrained GIN ε travels inline
            params = node.get("params") or {}
            if node["type"] == "gin_conv" and not params.get("train_eps") and "eps" in params:
                published[f"{node['id']}.eps"] = np.asarray([float(params["eps"])])
                roles[f"{node['id']}.eps"] = "buffer"
        return StoredWeights.load(self.bundle.root).match(published, roles)

    # -- the samples ---------------------------------------------------------------------------
    def samples(self, examples: Mapping) -> list:
        """(name, request) of every sample D6 §8 explains, at most EXPLAIN_MAX_SAMPLES."""
        out = []
        for preset in examples.get("presets") or ():
            out.append((f"preset {preset.get('id')}", self._request(preset)))
        demo = (examples.get("demo") or {}).get("items") or ()
        for k, item in enumerate(demo[:C.EXPLAIN_DEMO_ROWS]):
            out.append((f"demo {k}", self._request(item)))
        out += self._shipped((examples.get("test") or {}).get("rows") or [])
        return out[:C.EXPLAIN_MAX_SAMPLES]

    def _request(self, item: Mapping) -> dict:
        rows = list(item.get("rows") or ([item["row"]] if item.get("row") else []))
        req = {"rows": rows} if self.bundle.shape != "row" else {"row": rows[0]}
        if item.get("history") and self.bundle.shape != "graph":
            req["history"] = list(item["history"])
        return req

    def _shipped(self, rows: list) -> list:
        n, shape = C.EXPLAIN_SHIPPED_ROWS, self.bundle.shape
        if shape == "row":
            return [(f"example row {k}", {"row": rows[k]}) for k in range(min(n, len(rows)))]
        if shape == "window":
            answers = self._predict({"rows": rows})
            done = [k for k, a in enumerate(answers) if a.get("answered", True)][:n]
            return [(f"example row {k}", {"rows": rows, "at": k}) for k in done]
        if self.bundle.level == "node":
            return [(f"example row {k}", {"rows": rows, "at": k}) for k in range(min(n, len(rows)))]
        group, firsts = self.bundle.sample.get("graph_id"), {}
        for k, row in enumerate(rows):
            firsts.setdefault(row.get(group), k)
        return [(f"example graph {g}", {"rows": rows, "at": k})
                for g, k in list(firsts.items())[:n]]

    # -- one sample ----------------------------------------------------------------------------
    def includes(self) -> tuple:
        kinds = {self.bundle.field_kind(f)[0] for f in self.bundle.output.get("fields") or ()}
        return tuple(C.FIELD_KIND_INCLUDE[k] for k in C.FIELD_KIND_INCLUDE if k in kinds)

    def _query(self) -> str:
        return "".join(f"&include={i}" for i in self.includes()).lstrip("&")

    def _predict(self, req: Mapping) -> list:
        rows = req.get("rows") or [req["row"]]
        doc = {"rows": rows, **({"history": req["history"]} if req.get("history") else {})}
        path = C.PREDICT_ROUTE + (f"?{self._query()}" if self._query() else "")
        return self.client.json_of(self.client.post(path, doc), f"POST {path}")["predictions"]

    def explain(self, req: Mapping, **fields):
        path = C.EXPLAIN_ROUTE + (f"?{self._query()}" if self._query() else "")
        return self.client.post(path, {**req, "detail": C.EXPLAIN_DETAIL, **fields})

    def sensitivity(self, name: str, req: Mapping, explanation: Mapping) -> None:
        """EI2's own proof: the head's weight that meets the sample's largest input moved in a
        copy of the served payload, EI2 must fail at the head — and only there."""
        head = X.head_of(self.view)
        exact = ExactPass.of(explanation)
        x = exact.get(head, "in")
        tensor_id = (self.view.node(head).get("params") or {}).get("weight")
        if x is None or not tensor_id:
            self.report.record("EI2", False, f"no exact input or weight at the head {head} to "
                               "perturb", f"{name} (perturbed copy)")
            return
        column = int(np.argmax(np.abs(np.asarray(x, dtype=np.float64)).reshape(-1)[
            -self.view.tensor(tensor_id).shape[-1]:]))
        weight = self.view.tensor(tensor_id)
        delta = 0.5 * max(1.0, float(np.abs(weight).max()))
        copy_view = NetworkView(perturbed(self.view.network, self.view, tensor_id, (0, column),
                                          delta), self._fetch_tensor)
        context = None
        if self.bundle.shape == "graph":
            context = self.batches.context(req.get("rows"), req.get("history"),
                                           explanation.get("sample") or {})
        failed = Recompute(copy_view, exact, context,
                           (explanation.get("outputs") or {}).get("raw")).run().failures("ei2")
        where = sorted({c.node for c in failed})
        self.report.record("EI2", where == [head],
                           f"{tensor_id}[0, {column}] moved by {delta:.3g} in a copy of the "
                           f"payload: EI2 fails at {where} (expected exactly [{head}])",
                           f"{name} (perturbed copy)")

    def sample(self, name: str, req: Mapping) -> dict | None:
        answer = self.explain(req)
        if answer.status != 200:
            self.report.record("EI1", False, f"POST {C.EXPLAIN_ROUTE} answered {answer.status}: "
                               f"{answer.body[:300]!r}", name)
            return None
        explanation = answer.json["explanation"]
        self.explained += 1
        row = int((explanation.get("sample") or {}).get("row") or 0)
        served = self._predict(req)[row]
        self.ei1(name, explanation, served)
        exact = ExactPass.of(explanation)
        self.recompute(name, req, explanation, exact)
        self.ei4_served(name, explanation, served)
        for check_id, observations in (
                ("EI5", X.ei5(self.view, exact, explanation)),
                ("EI6", X.ei6(self.view.network, explanation)),
                ("EI7", X.ei7(self.view.network, explanation)),
                ("EI9", X.ei9(explanation, self.bundle.output.get("positive"),
                              self.view.network)),
                ("EI11", self.ei11(explanation, req))):
            for ok, detail in observations:
                self.report.record(check_id, ok, detail, name)
        self.ei12(name, explanation, served)
        return explanation

    def ei1(self, name: str, explanation: Mapping, served: Mapping) -> None:
        mine = explanation.get("prediction") or {}
        differ = _differences(mine, served)
        self.report.record("EI1", not differ, f"the explanation's prediction = /v1/predict's "
                           f"answer (differ: {differ[:3]})", name)

    def recompute(self, name: str, req: Mapping, explanation: Mapping, exact: ExactPass) -> None:
        context = None
        if self.bundle.shape == "graph":
            context = self.batches.context(req.get("rows"), req.get("history"),
                                           explanation.get("sample") or {})
        raw = (explanation.get("outputs") or {}).get("raw")
        result = Recompute(self.view, exact, context, raw).run()
        unchecked = [i for i in self.view.ids if self.view.node(i)["type"] in O.ORACLES and
                     exact.levels.get(i) != "exact"]
        self.report.record("EI2", not unchecked and not explanation.get("degraded"),
                           f"every oracle node at level exact (summarised: {unchecked[:4]}, "
                           f"degraded {explanation.get('degraded')})", name)
        for which, check_id in (("ei2", "EI2"), ("ei3", "EI3"), ("ei4", "EI4")):
            for c in getattr(result, which):
                self.report.record(check_id, c.ok, f"{c.node} ({c.type}) {c.detail}: |Δ| "
                                   f"{c.gap:.3g} (bound {c.bound:.3g})", name)
        for node_id in sorted({c.node for c in result.ei2}):
            self.oracle_checks[self.view.node(node_id)["type"]] += 1
        if not result.ei4:
            self.report.record("EI4", False, "no last node with an exact output", name)

    def ei4_served(self, name: str, explanation: Mapping, served: Mapping) -> None:
        """The served raw fields (logits, value, horizon steps, quantiles, embedding) are the
        head's raw output under its names, within the bundle's round-trip tolerance (the
        served forward may take the encoder's no-grad fast path, the explanation the captured
        one: the inspector's own fidelity rule, inspect-contract §5.4)."""
        tolerance = self.bundle.tolerance
        outputs = explanation.get("outputs") or {}
        names = [str(n) for n in outputs.get("names") or ()]
        raw = list(outputs.get("raw") or ())
        for field in self.bundle.output.get("fields") or ():
            kind, of = self.bundle.field_kind(field)
            index = _raw_index(kind, of, names, outputs.get("kind"))
            if index is None:
                continue
            got = _served_raw(served, kind, of)
            if got is None:
                continue
            self.report.record("EI4", abs(float(got) - float(raw[index])) <= tolerance,
                               f"served {field} = raw[{names[index]}] ({got} vs {raw[index]}, "
                               f"round-trip tolerance {tolerance:.3g})", name)

    def ei11(self, explanation: Mapping, req: Mapping) -> list:
        shape = self.bundle.shape
        if shape == "row":
            return [(True, "a row sample reads its one row")] if \
                (explanation.get("sample") or {}).get("span") is None else \
                [(False, "a row sample names a span")]
        if shape == "window":
            return RF.ei11_window(explanation, req, self.bundle.sample,
                                  int(self.bundle.sample.get("length")))
        if self.bundle.level == "node":
            return RF.ei11_node(explanation, req, self.reference)
        return RF.ei11_graph(explanation, req, self.bundle.sample, self.reference)

    def ei12(self, name: str, explanation: Mapping, served: Mapping) -> None:
        carried = set((explanation.get("prediction") or {}).get("warnings") or ()) | \
            set(explanation.get("warnings") or ())
        lost = [w for w in served.get("warnings") or () if w not in carried]
        self.report.record("EI12", not lost, f"{len(served.get('warnings') or ())} served "
                           f"warning(s) carried (lost: {lost[:2]})", name)

    # -- once per lane -------------------------------------------------------------------------
    def ei8(self, req: Mapping) -> None:
        a, b = self.explain(req), self.explain(req)
        self.report.record("EI8", a.status == b.status == 200 and a.body == b.body,
                           f"the same request twice: {len(a.body)} and {len(b.body)} bytes, "
                           "identical")

    def ei10(self, req: Mapping) -> None:
        empty_row = {"rows": [{}]} if "rows" in req else {"row": {}}
        cases = [("no rows", {"rows": []}, "rows", None), ("an empty row", empty_row, None, None),
                 ("an unknown target kind", {**req, "target": {"kind": UNKNOWN}}, "target.kind",
                  UNKNOWN)]
        selector = {"binary": "class", "multiclass": "class",
                    "multilabel": "label"}.get(self.bundle.reading)
        if selector:
            cases.append((f"an unknown {selector}", {**req, "target": {selector: UNKNOWN}},
                          f"target.{selector}", UNKNOWN))
        for what, doc, field, value in cases:
            answer = self.explain({}, **doc)
            body = answer.json or {}
            fields = [e.get("field") for e in body.get("errors") or ()]
            named = field is None or (field in fields and
                                      (value is None or value in json.dumps(body)))
            self.report.record("EI10", answer.status == 422 and named,
                               f"{what}: {answer.status}, errors naming {fields}", what)

    def targets(self, name: str, req: Mapping) -> None:
        """EI5 on every selectable target kind of the first sample, and every class."""
        task = self.view.network.get("task") or {}
        asks = [{"kind": k} for k in task.get("targets") or ()]
        classes = [str(c) for c in self.bundle.output.get("classes") or ()][:MAX_CLASS_TARGETS]
        if self.bundle.reading in ("binary", "multiclass"):
            asks += [{"class": c} for c in classes]
        elif self.bundle.reading == "multilabel":
            asks += [{"label": c} for c in classes]
        for target in asks:
            answer = self.explain(req, target=target)
            if answer.status != 200:
                self.report.record("EI5", False, f"target {target}: {answer.status} "
                                   f"{answer.body[:200]!r}", name)
                continue
            body = answer.json["explanation"]
            for ok, detail in X.ei5(self.view, ExactPass.of(body), body):
                self.report.record("EI5", ok, f"{target}: {detail}", name)

    def completeness(self, name: str, req: Mapping) -> None:
        """EI7 on a linear network: Σ attribution = s(x) − s(0) for a raw-score target."""
        family = (self.bundle.netspec.get("family") or {}).get("name")
        kinds = [k for k in (self.view.network.get("task") or {}).get("targets") or ()
                 if k in RAW_TARGETS]
        if family != "linear" or not kinds:
            return
        answer = self.explain(req, target={"kind": kinds[0]})
        body = self.client.json_of(answer, f"explain target {kinds[0]}")["explanation"]
        att, target = body.get("attribution") or {}, body.get("target") or {}
        s0 = _score_at_zero(self.view, target)
        total = float(att.get("total"))
        want = float(target.get("value")) - s0
        self.report.record("EI7", abs(total - want) <= COMPLETENESS_TOL * max(1.0, abs(want)),
                           f"linear {kinds[0]}: Σ attribution {total:.9g} = s − s(0) "
                           f"{want:.9g}", name)

    # -- NI8 -----------------------------------------------------------------------------------
    def ni8(self, client: Client, where: str) -> None:
        sample = self._any_request()
        for method, path, doc in (("GET", C.NETWORK_ROUTE, None),
                                  ("GET", C.NETWORK_TENSOR_ROUTE.replace("{id}", "any"), None),
                                  ("POST", C.EXPLAIN_ROUTE, sample)):
            answer = client.get(path) if method == "GET" else client.post(path, doc)
            needle = C.INSPECTION_OFF["needle"]
            self.report.record("NI8", answer.status == C.INSPECTION_OFF["status"] and
                               needle in (answer.body or b"").decode("utf-8", "replace"),
                               f"{method} {path}: {answer.status} {answer.body[:120]!r}", where)
        status = client.json_of(client.get(C.HEALTH_ROUTE), C.HEALTH_ROUTE).get(
            C.HEALTH_INSPECTION) or {}
        self.report.record("NI8", status.get("ready") is False and status.get("why"),
                           f"/health.inspection: ready {status.get('ready')}, why "
                           f"{status.get('why')!r}", where)

    def _any_request(self) -> dict:
        examples = self.client.json_of(self.client.get(C.EXAMPLES_ROUTE), C.EXAMPLES_ROUTE)
        rows = (examples.get("test") or {}).get("rows") or []
        return {"row": rows[0]} if self.bundle.shape == "row" else {"rows": rows}

    def ni8_switched_off(self) -> None:
        """A second service of the bundle with <ENV>_INSPECT=off answers NI8."""
        env_name = dig(self.bundle.manifest, C.SERVE_ENV_KEY)
        if not env_name:
            raise CheckError("MANIFEST.serve.identity.env is missing: the switch has no name")
        env = dict(os.environ, **{f"{env_name}{C.INSPECTION_OFF['env_suffix']}":
                                  C.INSPECTION_OFF["env_off"]})
        log = Path(tempfile.mkdtemp(prefix="vega-inspect-off-")) / "service.log"
        service = svc.start(self.bundle.root, threads=1,
                            launch=svc.Launch(self.python, log, env))
        try:
            self.ni8(Client(service.base_url), f"{env_name}{C.INSPECTION_OFF['env_suffix']}=off")
        finally:
            svc.stop(service)

    # -- the run -------------------------------------------------------------------------------
    def run(self) -> None:
        if not self.bundle.inspection_enabled():
            self.ni8(self.client, "packed with --no-inspect")
            for check_id in C.INSPECT_CHECK_IDS:
                if check_id != "NI8":
                    self.report.not_applicable(check_id, "the bundle ships without inspection")
            return
        self.network()
        if self.bundle.shape == "graph":
            self.batches = ServedBatches(self.bundle.root)
            self.reference = ReferenceGraph.load(self.bundle.root)
        examples = self.client.json_of(self.client.get(C.EXAMPLES_ROUTE), C.EXAMPLES_ROUTE)
        samples = self.samples(examples)
        if not samples:
            raise CheckError("/v1/examples gives no sample to explain")
        explained = [(name, req, self.sample(name, req)) for name, req in samples]
        name, first, body = explained[0]
        if body is not None:
            self.sensitivity(name, first, body)
        self.ei8(first)
        self.ei10(first)
        self.targets(name, first)
        self.completeness(name, first)
        self.ni8_switched_off()


# --------------------------------------------------------------------------- helpers
def _differences(a: Any, b: Any, path: str = "") -> list:
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = [f"{path}.{k}: only one side" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += _differences(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} vs {len(b)} entries"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _differences(x, y, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
            not isinstance(a, bool) and not isinstance(b, bool):
        return [] if close(a, b, 1e-6, 1e-9) else [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def _raw_index(kind: str, of: Any, names: list, outputs_kind: str) -> int | None:
    """The head column a served raw-kind field reads (inspect-contract §4.5's names)."""
    if kind == "raw":
        if outputs_kind == "thresholds":
            return None                  # an ordinal's raw fields are named by threshold
        return names.index(str(of)) if str(of) in names else None
    if kind == "value" and outputs_kind == "value":
        return 0
    if kind == "horizon":
        step, _, q = str(of).partition("/")
        name = f"t+{step}" + (f"/q{q}" if q else "")
        return names.index(name) if name in names else None
    if kind == "quantile":
        name = f"q{of}"
        return names.index(name) if name in names else None
    if kind == "embedding":
        return int(of)
    return None


def _served_raw(served: Mapping, kind: str, of: Any):
    key = C.FIELD_KIND_KEYS.get(kind)
    value = served.get(key) if key else None
    if value is None:
        return None
    if kind == "raw" or kind == "quantile":
        return value.get(str(of))
    if kind == "horizon":
        step, _, q = str(of).partition("/")
        cell = value.get(f"h{step}")
        return cell.get(q) if q and isinstance(cell, Mapping) else cell
    if kind == "embedding":
        return value[int(of)] if int(of) < len(value) else None
    return value


def _score_at_zero(view: NetworkView, target: Mapping) -> float:
    """s(0) of a linear network: its head over a zero input (and zero embedding outputs),
    through the head's affine, read as the target reads the head."""
    head = X.head_of(view)
    _, weights, params = O.view_block(view, head)
    width = int(view.node(head).get("in_shape")[-1])
    z = np.asarray(O.ORACLES["head"](weights, np.zeros((1, width)), params))[0]
    if weights.get("target_scale") is not None:
        z = z * weights["target_scale"] + weights["target_shift"]
    kind, index = target.get("kind"), int(target.get("index") or 0)
    return float(target.get("sign") or 1) * float(z[0 if kind == "binary_logit" else index])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--lane", default="")
    ap.add_argument("--lanes-file", default=None, help="accepted for the stage CLI; unused")
    ap.add_argument("--json", default=None)
    ap.add_argument("--python", default=sys.executable,
                    help="the interpreter of the second (inspection-off) service for NI8")
    args = ap.parse_args(argv)
    report = Report(C.INSPECT_CHECK, args.lane, args.base_url, str(args.bundle),
                    C.INSPECT_CHECK_IDS)
    try:
        bundle = BundleRecords.load(args.bundle)
        check = InspectCheck(bundle, Client(args.base_url.rstrip("/")), report, args.python)
        check.run()
    except (CheckError, svc.ServiceError) as exc:
        print(f"ERROR: inspect_check cannot run: {exc}")
        write_result(args.json, {**report.as_dict(), "verdict": "error", "error": str(exc)})
        return 2
    report.print_lines()
    nodes = len(check.view.ids) if check.view is not None else 0
    write_result(args.json, report.as_dict(nodes=nodes, explained_rows=check.explained,
                                           oracle_checks=dict(sorted(check.oracle_checks.items()))))
    print(("OK: " if report.verdict == "pass" else "ERROR: ") +
          f"inspect_check {report.verdict} on {args.lane or args.bundle.name}: "
          f"{check.explained} sample(s) explained exactly")
    return 0 if report.verdict == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
