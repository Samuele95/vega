#!/usr/bin/env python3
"""
test_emit_smoke.py — `emit_backend.py --smoke` on generation v2.3 (WP B14; D1 §16, E19).

    python3 -m pytest -q vega-forge/scripts/tests/test_emit_smoke.py

The smoke opens the emitted package the way every later stage does (`PackageAdapters.load`),
builds the network with the Model port's builder and runs one forward pass on a real batch of
the train part. On small shaped datasets of each sample shape (written the way the shaper
writes them, `data_fixtures.write_shaped`), it proves:

1. **smoke.json per shape**: the forward pass runs; `params_total` is the network's; `madds_unit`
   is `row`, `anchor`, `target_node` or `graph`; the `sample` block carries D1 §16's fields with
   the counts the DataSource itself gives (windows per part, the valid part's unscored rows,
   the series count; the train and valid views' nodes, edges and targets, the test edges the
   training view withheld, no exposure when inductive; when transductive, every node and edge
   in both views, nothing withheld and the exposure disclosed — refused without a `user:`
   exposure).
2. **MAdds (E19)**: per row and per anchor they equal a hand count (the dense layers; the GRU's
   `gates · L · H · (in + H)`, summed per layer of a stacked bare torch RNN / GRU / LSTM, whose
   later layers read `H · dirs`); a block that counts itself (`madds()`) is asked and nothing
   inside it is counted twice; a module that holds parameters and that the counter does not
   know makes the number a lower bound, named — never a silent zero.
3. **The CLI**: `--smoke` writes smoke.json and prints the unit; `--campaign-root` is what a
   custom block's relative `code_ref` resolves against.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCRIPTS))

import data_fixtures as fx  # noqa: E402
import emit_backend  # noqa: E402
import render_templates as rt  # noqa: E402

EMIT = rt.load_emit()
_KEEP = []


def _work(prefix):
    work = tempfile.TemporaryDirectory(prefix=prefix)
    _KEEP.append(work)
    return Path(work.name)


def _emitted(spec, package):
    """The package of `spec` emitted by the real emitter into a fresh run directory."""
    out = _work(f"b14-smoke-{package}-")
    pipeline = rt.scoped_pipeline(rt.reference_pipeline(), spec)
    pipeline["system"]["package"] = package
    EMIT.emit(spec, pipeline, out, run_dir=out)
    return out


def _cpu(spec):
    spec["training"]["device"] = "cpu"
    return spec


# ================================================================ shaped datasets per shape

def row_spec():
    rng = np.random.default_rng(11)
    n = 120
    frame = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    frame["zone"] = rng.integers(0, 5, size=n)
    frame["y"] = (frame.x0 - frame.x1 > 0).astype(np.int64)
    perm = rng.permutation(n)
    parts = {"train": perm[:70], "valid": perm[70:95], "test": perm[95:]}
    man = fx.write_shaped(_work("b14-rows-"), {k: frame.iloc[v] for k, v in parts.items()},
                          parts)
    spec = rt.reference_spec("row")
    spec["data_contract"]["features"] = [fx.numeric(f"x{i}") for i in range(4)] + \
        [fx.nominal("zone", 5)]
    spec["data_contract"]["input_dim"] = 5
    spec["training"]["batch_size"] = 16
    return _cpu(fx.spec_on(spec, man))


def window_spec():
    """Three series × 50 steps, split 30/10/10 in time: L = 6, H = 2, the autoregressive
    channel, previous-part context."""
    rng = np.random.default_rng(5)
    rows = [{"series": f"s{s}", "ts": t, "x0": math.sin(t / 4 + s), "x1": float(s),
             "target": math.sin(t / 4 + s) + rng.normal(0, 0.05)}
            for t in range(50) for s in range(3)]
    frame = pd.DataFrame(rows)
    pos = np.arange(len(frame))
    parts = {"train": pos[frame.ts < 30], "valid": pos[(frame.ts >= 30) & (frame.ts < 40)],
             "test": pos[frame.ts >= 40]}
    man = fx.write_shaped(_work("b14-windows-"), {k: frame.iloc[v] for k, v in parts.items()},
                          parts)
    spec = rt.reference_spec("window")
    spec["data_contract"]["sample"].update({"length": 6, "stride": 1, "horizon": 2,
                                            "season": 4})
    spec["task"]["horizon"] = 2
    spec["blocks"] = [{"id": "G", "type": "gru", "params": {"hidden": 8, "return": "last"}},
                      {"id": "H", "type": "head", "params": {"out": 2, "activation": "none"}}]
    spec["training"]["batch_size"] = 16
    return _cpu(fx.spec_on(spec, man))


USER_EXPOSURE = {"kind": "features", "rows": "test", "source": "user:b14-smoke"}


def node_spec(protocol="inductive", exposure=None):
    """60 nodes of three communities, 150 undirected edges (both directions); a transductive
    protocol with its declared `user:` exposure when asked."""
    rng = np.random.default_rng(2)
    n = 60
    community = np.arange(n) % 3
    frame = pd.DataFrame({f"x{i}": rng.normal(community * 0.5, 1.0) for i in range(3)})
    frame["community"] = np.array(["a", "b", "c", "d"])[community]
    perm = rng.permutation(n)
    parts = {"train": perm[:36], "valid": perm[36:48], "test": perm[48:]}
    part_of = {int(p): name for name, pos in parts.items() for p in pos}
    pairs = set()
    while len(pairs) < 75:
        a, b = (int(v) for v in rng.integers(0, n, size=2))
        if a != b:
            pairs.add((min(a, b), max(a, b)))
    edges = pd.DataFrame([{"src_pos": s, "dst_pos": d, "src_part": part_of[s],
                           "dst_part": part_of[d], "weight": None}
                          for a, b in sorted(pairs) for s, d in ((a, b), (b, a))])
    man = fx.write_shaped(_work("b14-nodes-"), {k: frame.iloc[v] for k, v in parts.items()},
                          parts, edges)
    spec = rt.reference_spec("graph")
    spec["data_contract"]["features"] = [fx.numeric(f"x{i}") for i in range(3)]
    spec["data_contract"]["input_dim"] = 3
    if protocol != "inductive":
        spec["data_contract"]["sample"].update({"protocol": protocol, "exposure": exposure})
    spec["training"]["batch_size"] = "full"
    return _cpu(fx.spec_on(spec, man)), edges, parts


def level_spec():
    """30 graphs of 4–7 nodes (a cycle or a chain), split by whole graphs 18/6/6."""
    rng = np.random.default_rng(8)
    rows, edges, parts = [], [], {"train": [], "valid": [], "test": []}
    for g in range(30):
        size, label = int(rng.integers(4, 8)), g % 2
        part = "train" if g < 18 else "valid" if g < 24 else "test"
        first = len(rows)
        for k in range(size):
            rows.append({"molecule": f"m{g}", "x0": rng.normal(), "x1": float(k) / size,
                         "x2": rng.normal(label, 0.5), "y": label})
            parts[part].append(first + k)
        links = [(first + k, first + k + 1) for k in range(size - 1)]
        links += [(first + size - 1, first)] if label else []
        edges += [{"src_pos": s, "dst_pos": d, "src_part": part, "dst_part": part,
                   "weight": None} for a, b in links for s, d in ((a, b), (b, a))]
    frame = pd.DataFrame(rows)
    parts = {k: np.asarray(v) for k, v in parts.items()}
    edges = pd.DataFrame(edges)
    man = fx.write_shaped(_work("b14-graphs-"), {k: frame.iloc[v] for k, v in parts.items()},
                          parts, edges)
    spec = rt.reference_spec("graph-level")
    spec["data_contract"]["features"] = [fx.numeric(f"x{i}") for i in range(3)]
    spec["data_contract"]["input_dim"] = 3
    spec["training"]["batch_size"] = 8
    return _cpu(fx.spec_on(spec, man)), edges


# ================================================================ 1. smoke.json per shape

def _smoke(spec, package):
    out = _emitted(spec, package)
    return emit_backend.smoke(spec, out, package, "cpu"), out


def _common(rec, unit):
    assert rec["forward_ok"] is True and rec["madds_unit"] == unit
    assert rec["params_total"] == rec["params_trainable"] > 0
    assert rec["madds_per_sample"] > 0 and rec["madds_lower_bound_only"] is False, rec
    assert rec["device"] == "cpu" and rec["loss_formula"] and len(rec["schedule_lrs"]) == 3
    json.dumps(rec, allow_nan=False)


def test_smoke_row():
    rec, _ = _smoke(row_spec(), "b14_smoke_row")
    _common(rec, "row")
    assert rec["sample"] == {"shape": "row"}
    assert rec["batch_size"] == 16 and rec["rows_train"] == 70
    # embedding 0 · dense (4 + 3 → 16) · head (16 → 1): per row
    assert rec["madds_per_sample"] == (4 + 3) * 16 + 16 * 1
    assert [b["id"] for b in rec["per_block"]] == ["B1", "B2", "B3", "B4", "B5"]


def test_smoke_window():
    spec = window_spec()
    rec, _ = _smoke(spec, "b14_smoke_window")
    _common(rec, "anchor")
    s = rec["sample"]
    assert s["shape"] == "window" and (s["length"], s["stride"], s["horizon"]) == (6, 1, 2)
    assert s["series"] == 3 and s["context"] == "previous-part" and s["autoregressive"] is True
    # train: 3 series × 30 rows, 5 context rows each, and the last 2 cannot see H = 2 ahead
    assert s["windows"]["train"] == rec["rows_train"] == 3 * (30 - 5 - 2)
    assert s["windows"]["valid"] > 0 and s["unscored"]["valid"] >= 0
    # the GRU over L = 6 steps of F' = 2 + 1 (the autoregressive channel), H = 8: 3 gates;
    # then the head 8 -> 2
    assert rec["madds_per_sample"] == 3 * 6 * 8 * (3 + 8) + 8 * 2
    assert rec["out_shape"] == [2]


def test_smoke_graph_node():
    spec, edges, parts = node_spec()
    rec, _ = _smoke(spec, "b14_smoke_node")
    _common(rec, "target_node")
    s = rec["sample"]
    assert (s["shape"], s["level"], s["protocol"], s["batching"]) == \
        ("graph", "node", "inductive", "full")
    assert s["exposure"] is None
    assert s["train_view"] == {"nodes": 36, "targets": 36,
                               "edges": int(((edges.src_part == "train")
                                             & (edges.dst_part == "train")).sum())}
    assert s["valid_view"]["targets"] == 12 and s["valid_view"]["nodes"] == 48
    touching_test = int(((edges.src_part == "test") | (edges.dst_part == "test")).sum())
    assert s["test_edges_withheld"] == touching_test > 0
    assert rec["batch_size"] == rec["rows_train"] == 36


def test_smoke_graph_node_transductive():
    """The transductive door (D1 §9.4, §16): both views hold every node and every edge, the
    targets are the part's own nodes, nothing is withheld, and the exposure the training
    graph had of the test part is disclosed — the messages into test nodes, the features."""
    spec, edges, parts = node_spec("transductive", USER_EXPOSURE)
    rec, _ = _smoke(spec, "b14_smoke_node_tr")
    _common(rec, "target_node")
    s = rec["sample"]
    assert (s["shape"], s["level"], s["protocol"], s["batching"]) == \
        ("graph", "node", "transductive", "full")
    assert s["train_view"] == {"nodes": 60, "edges": len(edges), "targets": 36}
    assert s["valid_view"] == {"nodes": 60, "edges": len(edges), "targets": 12}
    assert s["test_edges_withheld"] == 0
    into_test = int((edges.dst_part == "test").sum())
    assert s["exposure"] == {"protocol": "transductive", "edges_into_test": into_test,
                             "features_seen_in_training": True} and into_test > 0
    assert rec["batch_size"] == rec["rows_train"] == 36


def test_a_transductive_spec_without_its_exposure_is_refused():
    """No `user:` exposure, no smoke: the seal holds before the package reads a file."""
    spec, _, _ = node_spec("transductive", None)
    with pytest.raises(Exception, match="user:") as info:
        _smoke(spec, "b14_smoke_node_undeclared")
    assert type(info.value).__name__ == "SealedPartError", repr(info.value)


def test_smoke_graph_level():
    spec, edges = level_spec()
    rec, _ = _smoke(spec, "b14_smoke_level")
    _common(rec, "graph")
    s = rec["sample"]
    assert (s["level"], s["batching"]) == ("graph", "graphs") and s["exposure"] is None
    assert s["train_view"]["targets"] == 18 and s["valid_view"]["targets"] == 6
    assert s["test_edges_withheld"] == int((edges.src_part == "test").sum()) > 0
    assert rec["batch_size"] == 8 and rec["rows_train"] == 18


# ================================================================ 2. the MAdds counter (E19)

class _Scale(torch.nn.Module):
    """Holds a parameter the counter has no rule for."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(3))

    def forward(self, x):
        return x * self.w


class _Counts(torch.nn.Module):
    """Counts itself; its inner Linear must not be counted again."""

    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.inner(x)

    def madds(self, n_rows, n_edges=0, length=1):
        return 1000 * int(n_rows)


class _Net(torch.nn.Module):
    def __init__(self, *mods):
        super().__init__()
        self.mods = torch.nn.ModuleList(mods)

    def forward(self, batch):
        x = batch["x_num"]
        for m in self.mods:
            x = m(x)
        return x


def test_the_counter_names_what_it_does_not_know():
    batch = {"x_num": torch.randn(4, 3)}
    total, unknown = emit_backend._madds(_Net(torch.nn.Linear(3, 3), _Scale()), batch)
    assert total == 4 * 3 * 3 and unknown == ["_Scale"]
    total, unknown = emit_backend._madds(_Net(_Counts(), torch.nn.ReLU()), batch)
    assert total == 4 * 1000 and unknown == []


def test_a_bare_recurrent_layer_is_counted():
    """E19: an nn.RNN is counted (1 gate), not dropped as unknown."""
    rnn = torch.nn.RNN(3, 5, batch_first=True)

    class Seq(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn

        def forward(self, batch):
            return self.rnn(batch["x_num"])[0]

    total, unknown = emit_backend._madds(Seq(), {"x_num": torch.randn(2, 7, 3)})
    assert unknown == [] and total == 2 * 1 * 7 * 5 * (3 + 5)


@pytest.mark.parametrize("make", [
    lambda: torch.nn.RNN(3, 5, num_layers=2, batch_first=True),
    lambda: torch.nn.GRU(3, 5, num_layers=3, bidirectional=True, batch_first=True),
    lambda: torch.nn.LSTM(4, 6, num_layers=2, batch_first=False),
    lambda: torch.nn.LSTM(4, 6, num_layers=2, bidirectional=True, proj_size=2,
                          batch_first=True),
], ids=["rnn-2", "bigru-3", "lstm-2-seq-first", "bilstm-2-proj"])
def test_a_bare_stacked_recurrent_layer_is_counted_per_layer(make):
    """E19: layer 0 reads `input_size` features, every later layer `hidden · dirs` (a projected
    LSTM's `proj_size · dirs`). One step of one sequence uses each recurrent weight element
    once, so the oracle is the weight count times rows and steps."""
    rnn = make()

    class Seq(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn

        def forward(self, batch):
            x = batch["x_num"]
            return self.rnn(x if rnn.batch_first else x.transpose(0, 1))[0]

    rows, steps = 3, 7
    total, unknown = emit_backend._madds(Seq(), {"x_num": torch.randn(rows, steps,
                                                                      rnn.input_size)})
    weights = sum(p.numel() for name, p in rnn.named_parameters() if name.startswith("weight"))
    assert unknown == [] and total == rows * steps * weights


def test_a_stacked_gru_matches_the_hand_count():
    """GRU(3 -> 5, 2 layers, bidirectional): layer 0 reads 3, layer 1 reads 5 · 2 = 10."""
    gru = torch.nn.GRU(3, 5, num_layers=2, bidirectional=True, batch_first=True)
    per_step = 2 * 3 * 5 * ((3 + 5) + (10 + 5))
    assert emit_backend._recurrent_step(gru) == per_step


def test_the_units_of_a_batch():
    assert emit_backend._units({"x_num": torch.zeros(5, 3)}) == (5, "row")
    assert emit_backend._units({"x_num": torch.zeros(5, 6, 3)}) == (5, "anchor")
    assert emit_backend._units({"x_num": torch.zeros(9, 3),
                                "target_index": torch.arange(4)}) == (4, "target_node")
    assert emit_backend._units({"x_num": torch.zeros(9, 3), "num_graphs": 3}) == (3, "graph")


# ================================================================ 3. the CLI

def test_the_cli_smokes_and_resolves_custom_code_against_the_campaign_root():
    spec = row_spec()
    root = _work("b14-campaign-")
    (root / "custom").mkdir()
    (root / "custom" / "gain.py").write_text(
        "import torch\n\n\nclass _Gain(torch.nn.Module):\n    def __init__(self, n):\n"
        "        super().__init__()\n        self.g = torch.nn.Linear(n, n)\n\n"
        "    def forward(self, x):\n        return self.g(x)\n\n\n"
        "def build(in_shape, params):\n    return _Gain(int(in_shape[-1]))\n", encoding="utf-8")
    spec["blocks"].insert(3, {"id": "BC", "type": "custom", "source": "user:test",
                              "params": {"code_ref": "custom/gain.py", "out_shape": [16]}})
    pipeline = rt.scoped_pipeline(rt.reference_pipeline(), spec)
    pipeline["system"]["package"] = "b14_smoke_cli"
    work = _work("b14-cli-")
    (work / "netspec.json").write_text(json.dumps(spec), encoding="utf-8")
    (work / "pipeline.json").write_text(json.dumps(pipeline), encoding="utf-8")
    base = [sys.executable, str(SCRIPTS / "emit_backend.py"), "--spec",
            str(work / "netspec.json"), "--pipeline", str(work / "pipeline.json"), "--smoke",
            "--device", "cpu"]
    r = subprocess.run(base + ["--out", str(work / "rung")], capture_output=True, text=True)
    assert r.returncode == 2 and "code_ref" in r.stdout, r.stdout + r.stderr
    r = subprocess.run(base + ["--out", str(work / "rung"), "--campaign-root", str(root)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MAdds per row" in r.stdout and "sample row" in r.stdout
    smoke = json.loads((work / "rung" / "smoke.json").read_text(encoding="utf-8"))
    assert smoke["forward_ok"] and "BC" in [b["id"] for b in smoke["per_block"]]
    assert (work / "rung" / "b14_smoke_cli" / "model" / "custom" / "gain.py").is_file()


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
