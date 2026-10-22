#!/usr/bin/env python3
"""
run_negatives_shapes.py — the compiler's shape gates W2 (graphs), W3 (the future), B2
(backend coverage), G2 (the shape walk) and T1e (the task's fields) must refuse what they say
they refuse, on the lanes' own fixtures (D6 §12.2, BUILD-PLAN T10; G2 and T1e, v2.3's other new
rules, so that the coverage report finds a seeded case for each, D6 §12.8).

    python3 run_negatives_shapes.py <matrix-dir> [-v] [--json <path>] [--lanes-file F]

Each case compiles a mutated copy of a lane's `netspec.json` (and, where stated, of its
manifest, its pipeline or its backend) with `compile_spec.py --check-only`, and asserts the exit
code and the words of the refusal. The lane's fixture is read from the matrix
(`<matrix>/lanes/<id>/fixture.json`, written by the lane's fixture stage); a lane the matrix does
not hold is built here with `lanekit.fixture` into `<matrix>/negatives/fixtures/<id>/` (and
reused on the next run) — the fixture stage alone, no training.

| id  | lane | seed                                                              | expect                        |
|-----|------|-------------------------------------------------------------------|-------------------------------|
| W2a | L19  | `sample.shape = "row"`                                            | exit 2; W2, gcn, graph        |
| W2b | L02  | `gcn_conv(16)` inserted before the head                          | exit 2; W2, gcn_conv, no edges|
| W2c | L19  | `graph_pool(mean)` added before the head                         | exit 2; W2, graph_pool, node  |
| W2d | L22  | `graph_pool` removed                                              | exit 2; W2, graph_pool, level graph |
| W2e | L22  | `sample.graph_id = null`                                          | exit 2; W2, graph_id          |
| W2f | L20  | `protocol: transductive` from `vega:default`                     | exit 2; W2, transductive, user: |
| W2g | L19  | `sample.edges` names a path the manifest does not                 | exit 2; W2, edges             |
| W2h | L19  | a manifest copy without `outputs.files.edges`                     | exit 2; W2, graph_edges       |
| W2i | L19  | family mlp-plain, blocks dense → activation → head, graph shape   | exit 0; a W2 WARN, edges are ignored |
| W3a | L10  | `gru.params.bidirectional = true` on the forecast                 | exit 2; W3, bidirectional, forecast |
| W3b | L13  | the pipeline's `serving.mode = "stream"`                          | exit 2; W3, stream            |
| W3c | L13  | as packed (a bidirectional sequence-classification)               | exit 0                        |
| W3d | L16  | `lstm.params.bidirectional = true` on a sequence-regression       | exit 0                        |
| B2a | L14  | a copied backend whose `coverage.json` lacks `rnn` (`--backend pytorch_minus`) | exit 2; B2, rnn, not covered |
| G2a | L16  | the reducer after the lstm removed: its steps reach the head      | exit 2; G2, unreduced         |
| G2b | L02  | a `conv1d` inserted before the head of a row network              | exit 2; G2, sequence block    |
| T1e | L13  | `task.base_kind` removed from a sequence-classification           | exit 2; T1e, base_kind        |

D6 wrote W3a's needle `B1` for the recurrent block's id; the lanes' ids are the composer's
(`B3` on L10), so the case names the block by the id its spec gives it. W2i, W3c and W3d are the
boundary positives: the gate must stay quiet there.

Exit 0 when every case behaved as stated; `--json` writes `vega/negatives-result@1`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

S = Path(__file__).resolve().parent.parent
SKILLS = S.parent.parent
TESTS = SKILLS / "vega" / "scripts" / "tests"
sys.path.insert(0, str(TESTS / "negatives"))

import negkit  # noqa: E402

COMPILER = S / "compile_spec.py"
LANES = ("L02", "L10", "L13", "L14", "L16", "L19", "L20", "L22")

_ap = argparse.ArgumentParser(description="the compiler's shape gates on the lanes' fixtures")
_ap.add_argument("matrix", help="the matrix dir (lanes/<id>/fixture.json)")
_ap.add_argument("-v", dest="verbose", action="store_true")
_ap.add_argument("--json", default=None, metavar="PATH")
_ap.add_argument("--lanes-file", default=str(TESTS / "lanes.json"))
ARGS = _ap.parse_args()
MATRIX = Path(ARGS.matrix).resolve()
CASES = negkit.Cases(__file__, MATRIX, ARGS.verbose, head=True)


# --------------------------------------------------------------------------- the fixtures

def _built(lane_id: str):
    """A lane's fixture: the matrix's own, else one built here once (the fixture stage)."""
    found = negkit.lane(MATRIX, lane_id)
    if found is not None:
        return found, "matrix"
    root = MATRIX / "negatives" / "fixtures" / lane_id
    cached = negkit.lane_in(root, lane_id)
    if cached is not None:
        return cached, "built earlier"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    proc = subprocess.run([sys.executable, "-m", "lanekit.fixture", "--lane", lane_id, "--work",
                           str(root / "work"), "--skills", str(SKILLS), "--lanes-file",
                           ARGS.lanes_file], capture_output=True, text=True, cwd=str(TESTS),
                          timeout=1800)
    text = proc.stdout
    start = text.rfind("\n{")
    try:
        doc = json.loads(text[start + 1:] if start >= 0 else text)
    except json.JSONDecodeError:
        return None, (f"its fixture could not be built (exit {proc.returncode}): "
                      f"{(proc.stdout + proc.stderr).strip().splitlines()[-1:]}")
    (root / "fixture.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    how = "built here"
    if proc.returncode != 0:
        # the fixture said it was written, then the interpreter did not exit cleanly (the
        # exit-time abort PLAN-AMENDMENTS A40 gives T08): said, and the files are used
        how += f" (WARN: the builder printed its record, then exited {proc.returncode})"
    return negkit.lane_in(root, lane_id), how


def fixtures() -> dict:
    """{lane id: Lane | why not}; the missing ones are built side by side."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        made = dict(zip(LANES, pool.map(_built, LANES)))
    out = {}
    for lane_id, (lane, how) in made.items():
        out[lane_id] = lane if lane is not None else how
        print(f"  {lane_id}: " + (f"fixture ({how}) at {lane.work}" if lane is not None else
                                  f"no fixture — {how}"))
    return out


# --------------------------------------------------------------------------- one compile

def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(path: Path) -> str:
    """The spec's pipeline_digest (netspec-contract §1, D1e): the parsed pipeline.json with
    sorted keys and no whitespace, as the emitter digests it — not the file's bytes."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return "sha256:" + hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"))
                                      .encode("utf-8")).hexdigest()


def compile_seeded(lane, td: Path, spec_seed=None, manifest_seed=None, pipeline_seed=None,
                   backend_seed=None):
    """compile_spec.py --check-only on the lane's spec with the seeds applied to copies: the
    spec's `input` rebound to a seeded manifest or pipeline (path and digest), as a spec written
    against that file would be. Returns (exit code, output)."""
    spec = json.loads(lane.netspec.read_text(encoding="utf-8"))
    pipeline = lane.pipeline
    if manifest_seed is not None:
        shaped = td / "shaped"
        shaped.mkdir()
        doc = json.loads((lane.shaped / "manifest.json").read_text(encoding="utf-8"))
        manifest_seed(doc)
        (shaped / "manifest.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
        spec["input"]["manifest"] = str(shaped / "manifest.json")
        spec["input"]["manifest_digest"] = _digest(shaped / "manifest.json")
        pdoc = json.loads(pipeline.read_text(encoding="utf-8"))
        if isinstance(pdoc.get("input"), dict) and "shape_manifest_digest" in pdoc["input"]:
            pdoc["input"]["shape_manifest_digest"] = spec["input"]["manifest_digest"]
            pipeline = td / "pipeline.json"
            pipeline.write_text(json.dumps(pdoc, indent=2), encoding="utf-8")
    if pipeline_seed is not None:
        pdoc = json.loads(Path(pipeline).read_text(encoding="utf-8"))
        pipeline_seed(pdoc)
        pipeline = td / "pipeline.json"
        pipeline.write_text(json.dumps(pdoc, indent=2), encoding="utf-8")
    if Path(pipeline) != lane.pipeline:
        spec["input"]["pipeline"] = str(pipeline)
        spec["input"]["pipeline_digest"] = _canonical(pipeline)
    if spec_seed is not None:
        spec_seed(spec)
    path = td / "netspec.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    compiler, extra = COMPILER, []
    if backend_seed is not None:
        compiler, extra = backend_seed(td)
    proc = subprocess.run([sys.executable, str(compiler), "--spec", str(path), "--model",
                           str(lane.model), "--pipeline", str(pipeline), "--memory",
                           str(lane.memory), "--check-only", "--explain", *extra],
                          capture_output=True, text=True, cwd=str(lane.work), timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


# --------------------------------------------------------------------------- the seeds

def _blocks(spec, kind):
    return [b for b in spec["blocks"] if isinstance(b, dict) and b.get("type") == kind]


def _before_head(spec, block):
    head = next(i for i, b in enumerate(spec["blocks"]) if b.get("type") == "head")
    spec["blocks"].insert(head, block)


def _block(bid, kind, params, why, alternatives=()):
    return {"id": bid, "type": kind, "params": params, "source": "user:seeded negative",
            "rationale": why, "alternatives": list(alternatives)}


def _sample(spec):
    return spec["data_contract"]["sample"]


def to_row(spec):
    _sample(spec)["shape"] = "row"


def gcn_before_head(spec):
    _before_head(spec, _block("BX", "gcn_conv", {"out": 16}, "a graph block on rows"))


def pool_before_head(spec):
    _before_head(spec, _block("BX", "graph_pool", {"readout": "mean"}, "a pool at node level"))


def drop_pool(spec):
    spec["blocks"] = [b for b in spec["blocks"] if b.get("type") != "graph_pool"]


def no_graph_id(spec):
    _sample(spec)["graph_id"] = None


def transductive_by_default(spec):
    _sample(spec).update(protocol="transductive", protocol_source="vega:default")


def foreign_edges(spec):
    _sample(spec)["edges"] = str(Path(_sample(spec)["edges"]).with_name("elsewhere.parquet"))


def no_edges_part(manifest):
    manifest["outputs"]["files"].pop("edges", None)


def plain_on_graph(spec):
    """mlp-plain on the graph sample: dense -> activation -> head, sourced as the user's."""
    head = copy.deepcopy(_blocks(spec, "head")[0])
    spec["family"] = {"name": "mlp-plain", "source": "user:seeded negative",
                      "rationale": "a non-graph family on a graph sample",
                      "alternatives": [{"name": "gcn", "when": "when the edges carry signal"}]}
    spec["blocks"] = [_block("B1", "dense", {"units": 16, "activation": "none"}, "the width",
                             [{"params": {"units": 32}, "when": "if the run underfits"}]),
                      _block("B2", "activation", {"name": "relu"}, "the non-linearity",
                             [{"params": {"name": "gelu"}, "when": "as an ablation"}]), head]


def bidirectional(kind):
    def seed(spec):
        blocks = _blocks(spec, kind)
        assert blocks, f"the lane's spec has no {kind} block to seed"
        blocks[0].setdefault("params", {})["bidirectional"] = True
    return seed


def drop_reducer(spec):
    spec["blocks"] = [b for b in spec["blocks"] if b.get("type") not in ("attention", "attention_pool", "pool")]


def conv_on_rows(spec):
    _before_head(spec, _block("BX", "conv1d", {"channels": 8, "kernel": 3}, "a sequence block on rows"))


def no_base_kind(spec):
    spec["task"].pop("base_kind")


def stream_mode(pipeline):
    pipeline.setdefault("serving", {})["mode"] = "stream"


def backend_without(block):
    """A copied scripts/ (the compiler and its rule modules, the catalog) whose backends/ holds
    `pytorch_minus`: the pytorch coverage.json without `block`."""
    def seed(td: Path):
        scripts = td / "scripts"
        scripts.mkdir()
        for path in list(S.glob("*.py")) + [S / "catalog.json"]:
            shutil.copy2(path, scripts / path.name)
        cov = json.loads((S / "backends" / "pytorch" / "coverage.json").read_text("utf-8"))
        assert block in cov["blocks"], f"{block} is not covered to begin with"
        cov["blocks"] = [b for b in cov["blocks"] if b != block]
        cov["name"] = "pytorch_minus"
        target = scripts / "backends" / "pytorch_minus"
        target.mkdir(parents=True)
        (target / "coverage.json").write_text(json.dumps(cov, indent=2), encoding="utf-8")
        return scripts / "compile_spec.py", ["--backend", "pytorch_minus"]
    return seed


# id, lane, seeds, expected exit, needles, sentence
TABLE = (
    ("W2a", "L19", dict(spec_seed=to_row), 2, ("W2", "gcn", "graph"),
     "a graph family on sample.shape row is refused"),
    ("W2b", "L02", dict(spec_seed=gcn_before_head), 2, ("W2", "gcn_conv", "no edges"),
     "a gcn_conv on a row sample has no edges to read"),
    ("W2c", "L19", dict(spec_seed=pool_before_head), 2, ("W2", "graph_pool", "node"),
     "a graph_pool at level node is refused"),
    ("W2d", "L22", dict(spec_seed=drop_pool), 2, ("W2", "graph_pool", "level graph"),
     "level graph without a graph_pool is refused"),
    ("W2e", "L22", dict(spec_seed=no_graph_id), 2, ("W2", "graph_id"),
     "level graph without sample.graph_id is refused"),
    ("W2f", "L20", dict(spec_seed=transductive_by_default), 2, ("W2", "transductive", "user:"),
     "a transductive protocol nobody chose is refused"),
    ("W2g", "L19", dict(spec_seed=foreign_edges), 2, ("W2", "edges"),
     "sample.edges naming a part the manifest does not is refused"),
    ("W2h", "L19", dict(manifest_seed=no_edges_part), 2, ("W2", "graph_edges"),
     "a manifest without the edges part is refused"),
    ("W2i", "L19", dict(spec_seed=plain_on_graph), 0, ("W2", "edges are ignored"),
     "a non-graph family on a graph sample passes with a WARN"),
    ("W3a", "L10", dict(spec_seed=bidirectional("gru")), 2, ("W3", "bidirectional", "forecast"),
     "a bidirectional gru on a forecast is refused"),
    ("W3b", "L13", dict(pipeline_seed=stream_mode), 2, ("W3", "stream"),
     "a bidirectional block under a stream blueprint is refused"),
    ("W3c", "L13", {}, 0, (), "a bidirectional sequence-classification compiles as packed"),
    ("W3d", "L16", dict(spec_seed=bidirectional("lstm")), 0, (),
     "a bidirectional lstm on a sequence-regression compiles"),
    ("B2a", "L14", dict(backend_seed=backend_without("rnn")), 2, ("B2", "rnn", "not covered"),
     "a backend whose coverage lacks rnn refuses the rnn lane"),
    ("G2a", "L16", dict(spec_seed=drop_reducer), 2, ("G2", "unreduced"),
     "an lstm's steps reaching the head unreduced are refused"),
    ("G2b", "L02", dict(spec_seed=conv_on_rows), 2, ("G2", "sequence block"),
     "a conv1d on a row sample is refused"),
    ("T1e", "L13", dict(spec_seed=no_base_kind), 2, ("T1e", "base_kind"),
     "a sequence-classification without its base kind is refused"),
)


def run_case(case, lanes):
    cid, lane_id, seeds, code, needles, sentence = case
    name = f"{cid}  {lane_id}: {sentence}"
    lane = lanes.get(lane_id)
    if not isinstance(lane, negkit.Lane):
        CASES.not_run(name, f"lane {lane_id}: {lane}")
        return
    with tempfile.TemporaryDirectory(prefix=f"neg-{cid}-") as td:
        rc, out = compile_seeded(lane, Path(td), **seeds)
    # the needles are read on the refusal's own lines (ERROR: for a refusal, WARN: for a
    # positive that must warn): the rule's OK line cites its id too, and must not count
    level = "ERROR:" if code else "WARN:"
    said = "\n".join(ln for ln in out.splitlines() if ln.startswith(level) and "[" in ln)
    ok_ = CASES.expect(name, rc, said if needles else out, code=code, contains=needles)
    if not ok_ and not ARGS.verbose:
        print("        " + "\n        ".join(out.strip().splitlines()[:8]))
    if ok_ and code == 0 and cid == "W3c":
        spec = json.loads(lane.netspec.read_text(encoding="utf-8"))
        bidir = [b for b in spec["blocks"] if (b.get("params") or {}).get("bidirectional")]
        CASES.check(f"W3c  {lane_id}: the lane's packed spec is the bidirectional one", bool(bidir),
                    "the L13 spec carries no bidirectional block, so W3c proves nothing")


def main() -> int:
    CASES.section("compile_spec.py — the lanes' fixtures", gate="vega-forge/compile_spec.py")
    lanes = fixtures()
    CASES.section("compile_spec.py W2 — a graph sample reads the shaper's graph, or nothing does")
    for case in TABLE[:9]:
        run_case(case, lanes)
    CASES.section("compile_spec.py W3 — no bidirectional block reads a future it will not have")
    for case in TABLE[9:13]:
        run_case(case, lanes)
    CASES.section("compile_spec.py B2 — the backend covers what the spec uses")
    for case in TABLE[13:14]:
        run_case(case, lanes)
    CASES.section("compile_spec.py G2, T1e — the shape walk reaches the head; the task carries its fields")
    for case in TABLE[14:]:
        run_case(case, lanes)
    return CASES.finish(ARGS.json)


if __name__ == "__main__":
    sys.exit(main())
