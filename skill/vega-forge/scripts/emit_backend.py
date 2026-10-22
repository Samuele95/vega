#!/usr/bin/env python3
"""
emit_backend.py — compile the spec into the blueprint's package, and prove it builds.

    python3 emit_backend.py --spec netspec.json --backend pytorch
                            --pipeline blueprint-runs/<ts>/pipeline.json
                            --out vega-runs/<ts>/ [--campaign-root <dir>] [--smoke]
                            [--device auto]

The emitter writes the package `code-contract.md` §2 describes — one sub-package per blueprint
module, `ports.py`, `domain.py`, the adapters and their support files (D1 §4.1),
`blueprint-map.json`, `spec_snapshot.json`, `requirements.txt`. It compiles **exactly the
spec**: a change to the network is a change to the spec and this script runs again. A `custom`
block's relative `code_ref` resolves against `--campaign-root` (default: `--out`, which is the
campaign root when the forge stage emits; a rung that re-emits into its own directory passes
the campaign's — netspec-contract §2).

With `--smoke` it then opens the package the way every later stage does — through
`vega_core.PackageAdapters.load` (the one resolver by blueprint-map role) — builds the network
with the Model port's builder (`Network.build`), runs one forward pass on a real batch of the
train part (`ShapedParts.load_part` + `make_loader`), and writes `smoke.json`:

- `params_total`, `params_trainable`, `per_block` (id, params, out_shape), `forward_ok`,
  `out_shape`, `batch_size` (the samples of the smoke batch), `rows_train` (the samples of the
  train part), `device`, `backend_version`;
- `madds_per_sample` and `madds_unit` — multiply-adds per `row`, per `anchor` (a window), per
  `target_node` (a node-level graph: the view's total over its targets) or per `graph`. A block
  that counts itself (`madds(n_rows, n_edges, length)`: the recurrent, attention-pool and
  graph blocks, the tokenizer, the cumulative head) is asked, and nothing inside it is counted
  again; `nn.Linear`, `nn.Conv1d`, `nn.MultiheadAttention` and bare recurrent layers are
  counted from their outputs; a module that holds parameters and is none of these makes the
  number a stated lower bound (`madds_lower_bound_only`, `madds_unknown_modules`) — never a
  silent zero (D1 E19);
- `schedule_lrs` and `loss_formula`, from the Trainer's own modules (`optim.resolved_lrs`,
  `losses.build_loss(spec).formula()`, through `PackageAdapters.lrs_for` / `loss_for`);
- `init_notes` (A37; spec_snapshot.json's `_resolved` too): a regression head's target affine;
- `sample` (D1 §16): `{"shape": "row"}`; for a window its length, stride, horizon, series
  count, window counts of train and valid, the valid part's unscored rows, context and
  autoregressive flag; for a graph its level, protocol, batching, the train and valid views'
  node, edge and target counts, the exposure (null when inductive) and the test edges the
  training view withheld.

A shape mismatch, a NaN on the first pass or an unknown block type is `ERROR:` — and the fix is
the spec, never the emitted code.

Exit codes: 0 emitted (and smoked, when asked); 2 input or emission error; 3 the smoke pass
failed (the package is on disk, but no layer may be written from it).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backends  # noqa: E402


def read_json(path, what):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: {what} not found: {path}")
        raise SystemExit(2)
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: {what} could not be read ({path}): {e}")
        raise SystemExit(2)


# --------------------------------------------------------------------------- MAdds

_GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}


def _madds_args(inp):
    """(n_rows, n_edges, length) of a block's positional inputs: the rows (windows, nodes) and
    steps of its first floating tensor, the edges of an `edge_index` (2, E) among them."""
    import torch
    tensors = [t for t in inp if torch.is_tensor(t)]
    x = next((t for t in tensors if t.is_floating_point()), None)
    if x is None:
        x = next((t for t in tensors if t.dim() >= 1 and not (t.dim() == 2 and t.shape[0] == 2)),
                 None)
    edges = next((t for t in tensors if not t.is_floating_point() and t.dim() == 2
                  and int(t.shape[0]) == 2), None)
    n_rows = int(x.shape[0]) if x is not None and x.dim() >= 1 else 0
    length = int(x.shape[1]) if x is not None and x.dim() == 3 else 1
    return n_rows, (int(edges.shape[1]) if edges is not None else 0), length


def _recurrent_step(mod):
    """The multiply-adds of one step of one sequence through a torch RNN / GRU / LSTM, summed
    per layer: layer 0 reads `input_size` features, every later layer reads the previous
    layer's `hidden * dirs` (the projection's `proj_size * dirs` for a projected LSTM)."""
    dirs = 2 if mod.bidirectional else 1
    hidden = mod.hidden_size
    out = getattr(mod, "proj_size", 0) or hidden        # what a step emits and feeds back
    per_dir = 0
    for layer in range(mod.num_layers):
        reads = mod.input_size if layer == 0 else out * dirs
        per_dir += _GATES[type(mod).__name__] * hidden * (reads + out)
        if out != hidden:
            per_dir += hidden * out                     # the LSTM's hidden -> proj_size
    return dirs * per_dir


def _counted(mod, inp, out):
    """The multiply-adds of one call of a torch module the counter knows, over the whole batch;
    None for a module it does not know."""
    from torch import nn
    if isinstance(mod, nn.Linear):
        return int(out.numel()) * mod.in_features
    if isinstance(mod, nn.Conv1d):
        return int(out.numel()) * (mod.in_channels // mod.groups) * mod.kernel_size[0]
    if isinstance(mod, (nn.RNN, nn.GRU, nn.LSTM)):
        x = inp[0]
        rows, steps = (x.shape[0], x.shape[1]) if mod.batch_first else (x.shape[1], x.shape[0])
        return int(rows) * int(steps) * _recurrent_step(mod)
    if isinstance(mod, nn.MultiheadAttention):
        q = inp[0]
        rows, steps = (q.shape[0], q.shape[1]) if mod.batch_first else (q.shape[1], q.shape[0])
        d = int(q.shape[2])
        return int(rows) * (4 * int(steps) * d * d + 2 * int(steps) * int(steps) * d)
    if isinstance(mod, (nn.Embedding, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        return 0                     # lookups and normalisations: no multiply-add worth a count
    return None


class _MaddsCounter:
    """Forward hooks over a model: the blocks that count themselves, the torch modules the
    counter knows, and the parameter-holding modules it does not (a lower bound, named)."""

    def __init__(self, model):
        self.total, self.unknown, self._handles = 0, set(), []
        owners = [m for m in model.modules() if callable(getattr(m, "madds", None))]
        inside = {id(sub) for m in owners for sub in m.modules() if sub is not m}
        for m in model.modules():
            if id(m) not in inside:
                self._handles.append(m.register_forward_hook(self._hook))

    def _hook(self, mod, inp, out):
        if callable(getattr(mod, "madds", None)):
            self.total += int(mod.madds(*_madds_args(inp)))
            return
        n = _counted(mod, inp, out)
        if n is not None:
            self.total += n
        elif any(True for _ in mod.parameters(recurse=False)):
            self.unknown.add(type(mod).__name__)

    def close(self):
        for h in self._handles:
            h.remove()


def _madds(model, batch):
    """(multiply-adds of one forward pass over `batch`, sorted unknown module names). Counted
    with autograd on, so a fused inference fast path cannot bypass the hooks."""
    import torch
    counter = _MaddsCounter(model)
    try:
        with torch.enable_grad():
            model(batch)
    finally:
        counter.close()
    return counter.total, sorted(counter.unknown)


# --------------------------------------------------------------------------- the sample block

def _units(batch):
    """(samples in a batch, the unit a sample is) — rows, anchors, target nodes or graphs."""
    if batch.get("num_graphs") is not None:
        return int(batch["num_graphs"]), "graph"
    if batch.get("target_index") is not None:
        return int(batch["target_index"].shape[0]), "target_node"
    x = batch["x_num"] if batch.get("x_num") is not None else batch["x_cat"]
    return int(x.shape[0]), ("anchor" if x.dim() == 3 else "row")


def _ingestion(adapters, name):
    """A support module beside the DataSource adapter (`frames`, `graphs`)."""
    return importlib.import_module(type(adapters.data).__module__.rsplit(".", 1)[0] + "." + name)


def _window_sample(adapters, ns, sample, train):
    window = sample.window
    valid = adapters.data.load_part(ns, "valid")
    series = 1
    if window.series_by:
        frames = _ingestion(adapters, "frames").PartFrames()
        frame = frames.open(ns, "train", columns=[window.series_by]).frame
        series = int(frame[window.series_by].nunique())
    return {"shape": "window", "length": int(window.length), "stride": int(window.stride),
            "horizon": int(window.horizon), "series": series,
            "windows": {"train": len(train), "valid": len(valid)},
            "unscored": {"valid": len(valid.unscored)}, "context": window.context.value,
            "autoregressive": bool(window.autoregressive)}


def _view_counts(part):
    return {"nodes": int(part.x_num.shape[0]) if part.x_num is not None else int(
                part.x_cat.shape[0]),
            "edges": int(part.edge_index.shape[1]), "targets": len(part)}


def _graph_sample(adapters, ns, sample, train):
    graph, domain = sample.graph, adapters.package.domain
    valid = adapters.data.load_part(ns, "valid")
    table = _ingestion(adapters, "graphs").EdgeTable.load(
        ns, _ingestion(adapters, "frames").PartFrames())
    withheld = graph.view(table.sizes, table.edges, domain.Part.TRAIN).test_edges_withheld
    exposure = adapters.data.exposure(ns)
    transductive = exposure is not None and exposure.protocol is \
        domain.GraphProtocol.TRANSDUCTIVE
    return {"shape": "graph", "level": graph.level.value, "protocol": graph.protocol.value,
            "batching": graph.batching.value, "train_view": _view_counts(train),
            "valid_view": _view_counts(valid),
            "exposure": exposure.as_dict() if transductive else None,
            "test_edges_withheld": int(withheld)}


def _sample_block(adapters, ns, train):
    sample = adapters.package.domain.SampleSpec.from_spec(ns)
    if sample.window is not None:
        return _window_sample(adapters, ns, sample, train)
    if sample.graph is not None:
        return _graph_sample(adapters, ns, sample, train)
    return {"shape": "row"}


# --------------------------------------------------------------------------- smoke

def _on(batch, device):
    import torch
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _trainer_numbers(adapters, ns, n_samples, batch_size):
    """The schedule's per-epoch learning rates and the loss formula, from the Trainer's own
    modules (its optimiser's and its losses'); a failure is recorded beside the number, never
    raised (the forward pass is the gate; compile_spec's B2 already refused a loss parameter
    the backend's loss does not take)."""
    rec = {}
    try:
        rec["schedule_lrs"] = adapters.lrs_for(ns, max(1, -(-int(n_samples) // max(1, batch_size))))
    except Exception as exc:  # noqa: BLE001
        rec["schedule_lrs"], rec["schedule_lrs_error"] = [], str(exc)[:160]
    try:
        rec["loss_formula"] = adapters.loss_for(ns).formula()
    except Exception as exc:  # noqa: BLE001
        rec["loss_formula"], rec["loss_formula_error"] = None, str(exc)[:160]
    return rec


def _forward(model, batch):
    import torch
    model.eval()
    with torch.no_grad():
        out = model(batch)
    if not torch.isfinite(out).all():
        raise RuntimeError("the first forward pass produced a non-finite value — the spec "
                           "builds a network that cannot be trained; fix the spec (widths, "
                           "init, normalisation), never the emitted code")
    return out


def smoke(spec, out_dir, package, device_want="auto"):
    """Open the package by role, build the network, run one forward pass on a real train
    batch; returns smoke.json as a dict."""
    import torch
    from vega_core import PackageAdapters

    adapters = PackageAdapters.load(Path(out_dir) / package, package)
    ns = adapters.package.domain.NetSpec.from_dict(spec)
    device = (adapters.device_for(ns) if device_want == "auto"
              else torch.device(device_want))
    model = adapters.build(ns, run_dir=str(out_dir)).to(device)
    train = adapters.data.load_part(ns, "train")
    batch = _on(next(iter(adapters.data.make_loader(train, ns, shuffle=False))), device)
    out = _forward(model, batch)
    total, trainable = model.param_count()
    n, unit = _units(batch)
    madds, unknown = _madds(model, batch)
    rec = {"params_total": int(total), "params_trainable": int(trainable),
           "per_block": model.block_shapes(batch),
           "madds_per_sample": int(madds // max(1, n)), "madds_unit": unit,
           "madds_lower_bound_only": bool(unknown), "forward_ok": True,
           "out_shape": [int(s) for s in out.shape[1:]], "batch_size": n,
           "device": str(device), "backend_version": torch.__version__,
           "rows_train": len(train)}
    if unknown:
        rec["madds_unknown_modules"] = unknown
    rec.update(_trainer_numbers(adapters, ns, len(train), n))
    rec["sample"] = _sample_block(adapters, ns, train)
    # A37: what the Trainer will fit — a regression-family package's domain carries the affine
    affine = getattr(adapters.package.domain, "TargetAffine", None)
    fitted = (affine.of(ns, adapters.data.train_targets(ns), model.ids[-1])
              if affine is not None and getattr(model, "target_affine_pending", False) else None)
    rec["init_notes"] = (affine.init_notes(fitted, ns) if affine is not None else
                         {"scheme": ns.init.get("scheme"), "target_affine": None, "notes": []})
    return rec


# --------------------------------------------------------------------------- CLI

def _parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--pipeline", default=None, help="the blueprint run's pipeline.json (required)")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--out", required=True, help="the run directory")
    ap.add_argument("--campaign-root", default=None,
                    help="what a custom block's relative code_ref resolves against "
                         "(netspec-contract §2); default: --out, the forge stage's campaign root")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    return ap


def _emit(a, spec, pipeline):
    """The backend's emit(); returns (record, exit code or None)."""
    name = a.backend or (spec.get("backend") or {}).get("name") or "pytorch"
    try:
        backend = backends.load(name)
    except ValueError as e:
        print(f"ERROR: {e}")
        return None, 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        return backend.emit(spec, pipeline, out, run_dir=Path(a.campaign_root or out)), None
    except backend.EmitError as e:
        print(f"ERROR: {e}")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: the {name} backend failed to emit: {type(e).__name__}: {e}")
    return None, 2


def _report_emission(rec, out):
    pkg = rec["package"]
    print(f"OK: emitted {pkg} into {out} — {len(rec['files'])} file(s) across "
          f"{len(rec['modules'])} blueprint module(s) ({', '.join(rec['modules'])})")
    print(f"OK: blueprint-map.json written, unmapped: [] , "
          f"{sum(1 for e in rec['blueprint_map']['entries'] if e.get('role'))} port entr(ies)")
    for e in rec["blueprint_map"]["entries"]:
        if str(e.get("status", "")).startswith("pending"):
            print(f"OK: {e['file']} is {e['status']} — the map says what the code says")
    for m in rec.get("markers") or []:
        print(m)


def _run_smoke(spec, out, pkg, device):
    try:
        s = smoke(spec, out, pkg, device)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: the smoke forward pass failed: {type(e).__name__}: {e}")
        print("ERROR: no architecture layer may be written from a spec that does not build — "
              "fix the spec and re-run from the compiler")
        (out / "smoke.json").write_text(
            json.dumps({"forward_ok": False, "error": f"{type(e).__name__}: {e}"}, indent=2),
            encoding="utf-8")
        return 3
    (out / "smoke.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    snap = out / "spec_snapshot.json"      # the smoke reads the train part; the emitter does not
    doc = json.loads(snap.read_text(encoding="utf-8"))
    doc.setdefault("_resolved", {})["init_notes"] = s["init_notes"]
    snap.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lb = " (lower bound)" if s.get("madds_lower_bound_only") else ""
    print(f"OK: smoke forward pass on a real train batch — {s['params_total']:,} parameters "
          f"({s['params_trainable']:,} trainable), {len(s['per_block'])} block(s), "
          f"{s['madds_per_sample']:,} MAdds per {s['madds_unit'].replace('_', ' ')}{lb}, "
          f"out {s['out_shape']}, sample {s['sample']['shape']}, device {s['device']}, "
          f"torch {s['backend_version']}")
    print(f"OK: wrote {out / 'smoke.json'}")
    return 0


def main(argv=None):
    a = _parser().parse_args(argv)
    spec = read_json(a.spec, "spec")
    pipe_path = a.pipeline or (spec.get("input") or {}).get("pipeline")
    if not pipe_path:
        print("ERROR: no blueprint pipeline.json: the emitted code lives in the blueprint's "
              "module map. Run `/blueprint <model> --mode pipeline` and pass --pipeline")
        return 2
    pipeline = read_json(pipe_path, "pipeline.json")
    rec, code = _emit(a, spec, pipeline)
    if rec is None:
        return code
    _report_emission(rec, Path(a.out))
    if not a.smoke:
        return 0
    return _run_smoke(spec, Path(a.out), rec["package"], a.device)


if __name__ == "__main__":
    sys.exit(main())
