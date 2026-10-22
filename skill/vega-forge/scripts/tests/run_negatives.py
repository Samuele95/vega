#!/usr/bin/env python3
"""
run_negatives.py — the other half of the acceptance run: every gate must refuse what it says
it refuses.

`run_chain.sh` proves the happy path works. That is the easy half. A gate that never fires is
indistinguishable from no gate at all, so this script seeds one defect at a time into a copy
of the fixture and asserts that the right script refuses it, with the right rule and a message
that names the fix.

    python3 run_negatives.py <work-dir> [-v] [--json <path>]

Exit 0 when every seeded defect was caught by the rule that should catch it. With `--json`,
the cases are also written to <path> as `vega/negatives-result@1` (id, name, ok, detail per
case; passed and failed counts), whatever the outcome.

The size discipline's cases (C0a, C11a–C14e) are planted by `conformance_fixtures/seeds.py`,
the same seeds `test_conformance_sizes.py` plants in its clean package, and are judged under
`--size-severity error`.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

S = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "conformance_fixtures"))

import seeds  # noqa: E402

_ap = argparse.ArgumentParser(description="seed one defect at a time; every gate must refuse it")
_ap.add_argument("work", nargs="?", default=None, help="the acceptance work dir (default: .)")
_ap.add_argument("-v", dest="verbose", action="store_true", help="print every case's output")
_ap.add_argument("--json", default=None, metavar="PATH",
                 help="also write the cases as vega/negatives-result@1 to PATH")
ARGS = _ap.parse_args()

HERE = Path(ARGS.work).resolve() if ARGS.work else Path.cwd()
MODEL = HERE / "orders.domain.analysis.shaped.html"
PIPE = HERE / "blueprint-runs" / "2026-09-05T10-12Z" / "pipeline.json"
RUN = HERE / "vega-runs" / "2026-09-05T11-03Z"
ARCH = HERE / ".claude" / "architect-runs" / "2026-09-05T11-20Z"
VERBOSE = ARGS.verbose
CASE_ID = re.compile(r"^([A-Z]{1,3}\d+[a-z]?)\s{2}")

SPEC = json.loads((HERE / "netspec.json").read_text())
PIPELINE = json.loads(PIPE.read_text())
LAYER = json.loads((RUN / "architecture-layer.json").read_text())

results = []


def run(cmd, cwd=None):
    proc = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True,
                          cwd=str(cwd or HERE), timeout=600)
    return proc.returncode, (proc.stdout + proc.stderr)


def expect(name, rc, out, *, code=2, contains=()):
    """A negative passes when the script refused AND said why in the words that name the fix."""
    problems = []
    if rc != code:
        problems.append(f"exit {rc}, expected {code}")
    for needle in contains:
        if needle.lower() not in out.lower():
            problems.append(f"message does not mention {needle!r}")
    record(name, problems, out)


def record(name, problems, out):
    """One case's verdict, printed and kept for --json: it passed when `problems` is empty."""
    results.append((name, not problems, "; ".join(problems)))
    mark = "ok  " if not problems else "FAIL"
    line = f"  {mark}  {name}"
    if problems:
        line += "  — " + "; ".join(problems)
    print(line)
    if VERBOSE or problems:
        print("        " + "\n        ".join(out.strip().splitlines()[:6]))


# --------------------------------------------------------------------------- spec negatives

def _canonical(doc):
    """The spec's input.pipeline_digest (netspec-contract §1, D1e): the emitter's digest."""
    return "sha256:" + hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"))
                                      .encode("utf-8")).hexdigest()


def compile_with(mutate, name, contains, pipeline=None, model=None):
    """A seeded spec (and pipeline) compiled. A seeded pipeline is a blueprint the spec was
    written against, so the spec names it by its digest — the seed is the only defect — unless
    `mutate` seeds the digest itself (it runs after)."""
    spec = copy.deepcopy(SPEC)
    pl = None
    if pipeline not in (None, False):
        pl = copy.deepcopy(PIPELINE)
        pipeline(pl)
        spec["input"]["pipeline_digest"] = _canonical(pl)
    mutate(spec)
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "netspec.json"
        sp.write_text(json.dumps(spec))
        pipe = PIPE
        if pl is not None:
            pipe = Path(td) / "pipeline.json"
            pipe.write_text(json.dumps(pl))
        if model is not None:
            seeded_model = Path(td) / MODEL.name
            seeded_model.write_text(model(MODEL.read_text(encoding="utf-8")), encoding="utf-8")
        cmd = [S / "compile_spec.py", "--spec", sp, "--model",
               seeded_model if model is not None else MODEL, "--check-only",
               "--memory", HERE / ".claude" / "domain-forge-memory.md"]
        if pipeline is not False:
            cmd += ["--pipeline", pipe]
        rc, out = run(cmd)
    expect(name, rc, out, contains=contains)


def spec_negatives():
    print("\ncompile_spec.py — the spec's own gates")

    compile_with(lambda s: s["optimizer"].pop("source"),
                 "P1  an untraced choice is refused",
                 ("carries no source", "untraced"))

    compile_with(lambda s: s["optimizer"].update(source="analysis:F99"),
                 "P1  a source that does not resolve is refused",
                 ("does not resolve", "F99"))

    compile_with(lambda s: s["schedule"].update(alternatives=[]),
                 "P1  a non-structural choice with no alternative is refused",
                 ("alternatives is empty", "structural"))

    compile_with(lambda s: s["optimizer"].update(catalog="defaults/nonesuch"),
                 "C1  a vega:default citing a catalog row that does not exist",
                 ("catalog row", "nonesuch"))

    compile_with(lambda s: s["data_contract"]["parts"].update(test="shaped/orders.test.parquet"),
                 "S1  a named test part is refused",
                 ("sealed", "/vega-eval"))

    compile_with(lambda s: s["metrics"]["selection"].update(monitor="test.pr_auc"),
                 "S1  a metric monitored on the test part is refused",
                 ("test part", "monitor"))

    compile_with(lambda s: s["metrics"]["selection"].update(monitor="train.pr_auc"),
                 "T1  a monitor on train is refused",
                 ("valid", "overfits the selection"))

    compile_with(lambda s: s["metrics"]["selection"].update(monitor="valid.mase"),
                 "T1  a monitor naming an undeclared metric is refused",
                 ("not among the declared metrics",))

    compile_with(lambda s: s["loss"].update(name="mse"),
                 "T1  mse on a binary label is refused",
                 ("not one the binary row allows", "bce_with_logits"))

    compile_with(lambda s: s["blocks"][-1]["params"].update(activation="sigmoid"),
                 "T1  a sigmoid head under bce_with_logits is refused",
                 ("double sigmoid", "the loss applies the link function"))

    compile_with(lambda s: s["blocks"][-1]["params"].update(out=2),
                 "T1  a head whose width contradicts the task is refused",
                 ("requires 1",))

    compile_with(lambda s: s["data_contract"].update(input_dim=7),
                 "D2  an input_dim that disagrees with the manifest is refused",
                 ("input_dim is 7", "output schema"))

    compile_with(lambda s: s["blocks"][0]["params"]["columns"].remove("region"),
                 "D2  a nominal neither embedded nor one-hot is refused",
                 ("invents an order that does not exist",))

    # the seeded value must actually disagree with the manifest, so it is derived from the
    # fixture rather than written down: a literal here silently stops being a defect the day
    # the fixture's vocabulary changes, and the gate then passes for the wrong reason
    def _wrong_cardinality(s):
        f = s["data_contract"]["features"][-1]
        f.update(cardinality=int(f.get("cardinality") or 2) + 3)

    compile_with(_wrong_cardinality,
                 "D2  an embedded cardinality that is not unseen_id + 1 is refused",
                 ("unseen_category_id", "what an unseen category maps to"))

    compile_with(lambda s: s["blocks"][2].update(type="densee"),
                 "G1  an unknown block type is refused, with the nearest known named",
                 ("not in the catalog", "nearest known type: 'dense'"))

    def _widen(s):
        s["blocks"][3]["params"]["projection"] = "none"
        s["blocks"][3]["params"]["inner"][0]["params"]["units"] = 128

    compile_with(_widen,
                 "G1  a residual that changes width without a projection is refused",
                 ("enters at width 64", "projection: auto"))

    compile_with(lambda s: s["metrics"]["secondary"].append("bleu"),
                 "B2  a metric the backend does not implement is refused",
                 ("not implemented by the pytorch backend",))

    compile_with(lambda s: s["training"].update(batch_size=4096),
                 "R1  a batch larger than the training rows is refused",
                 ("exceeds the", "training rows"))

    compile_with(lambda s: s["training"].pop("budget"),
                 "R1  a spec with no wall-clock budget is refused",
                 ("wall_clock_min", "must still end"))

    compile_with(lambda s: s["optimizer"]["params"].update(weight_decay=0.01),
                 "R2  a live weight decay beside a declaration that disagrees is refused",
                 ("optimizer/params/weight_decay is 0.01", "declares weight_decay 0.0001",
                  "set the value once"))

    def _b3_rate(s):
        b3 = next(b for b in s["blocks"] if b["id"] == "B3")
        b3["params"]["inner"][3]["params"]["p"] = 0.5
    compile_with(_b3_rate,
                 "R2  a dropout rate beside the declaration covering it that disagrees is refused",
                 ("blocks[id=B3]/params/inner/3/params/p is 0.5", "declares dropout 0.2"))

    compile_with(lambda s: s["regularization"].append(
                     dict(copy.deepcopy(s["regularization"][1]), value=0.4)),
                 "R2  two dropout declarations over one block that disagree are refused",
                 ("conflicting declarations", "blocks[id=B3]"))

    compile_with(lambda s: s["family"].update(name="gru"),
                 "W1  a sequence family on a row sample is refused",
                 ("needs windows",))

    # no --pipeline flag AND no input.pipeline: there is no blueprint anywhere to fall back on
    compile_with(lambda s: s["input"].pop("pipeline"),
                 "D1  a run with no blueprint is refused",
                 ("no blueprint", "/blueprint"), pipeline=False)

    compile_with(lambda s: s["input"].pop("pipeline_digest"),
                 "D1e  a spec with no input.pipeline_digest is refused",
                 ("input.pipeline_digest is missing", _canonical(PIPELINE)))

    compile_with(lambda s: s["input"].update(
                     pipeline_digest="sha256:" + hashlib.sha256(PIPE.read_bytes()).hexdigest()),
                 "D1e  a pipeline_digest computed over the file's bytes is refused",
                 ("not of the file's bytes", _canonical(PIPELINE)))

    compile_with(lambda s: s["input"].update(pipeline_digest=_canonical(PIPELINE)),
                 "D1e  a spec naming another blueprint than the one compiled against is refused",
                 ("written against another blueprint",),
                 pipeline=lambda p: p["serving"].update(latency_ms=50))

    compile_with(lambda s: s["input"].update(manifest_digest="sha256:" + "0" * 64),
                 "D1b  a spec citing another manifest digest than the shape layer's is refused",
                 ("the shape layer records", "copy the shape layer's manifest_digest"))

    compile_with(lambda s: None,
                 "D1b  a blueprint drawn against another manifest is an error, not a warning",
                 ("ERROR: blueprint older than the shape", "re-run /blueprint --mode pipeline"),
                 pipeline=lambda p: p["input"].update(shape_manifest_digest="sha256:" + "0" * 64))

    compile_with(lambda s: next(b for b in s["blocks"] if b["type"] == "head").update(
                     source="geometry:partitions/overloaded"),
                 "P1  a geometry partition label that is no candidate does not resolve",
                 ("does not resolve", "no partition candidate labelled 'overloaded'"))

    compile_with(lambda s: next(b for b in s["blocks"] if b["type"] == "head").update(
                     source="geometry:partitions/chosen"),
                 "P1  geometry:partitions/chosen does not resolve when the geometry chose none",
                 ("does not resolve", "chose no partition"),
                 model=lambda html: html.replace('"chosen": "late"', '"chosen": null'))

    compile_with(lambda s: None, "D1  a feature contract that disagrees is refused",
                 ("feature_contract.input_dim",),
                 pipeline=lambda p: p["feature_contract"].update(input_dim=11))

    compile_with(lambda s: None, "K1  a blueprint missing a port this stage needs",
                 ("role 'Trainer'", "--concern"),
                 pipeline=lambda p: p.__setitem__(
                     "ports", [x for x in p["ports"] if x["role"] != "Trainer"]))

    compile_with(lambda s: None, "D1  a posture of 'tree' overruled without saying so",
                 ("posture", "user: source"),
                 pipeline=lambda p: p.__setitem__("posture", {"family": "tree"}))

    compile_with(lambda s: None, "D1  serving.mode 'stream' without a windowed sample",
                 ("stream", "windowed sample"),
                 pipeline=lambda p: p["serving"].update(mode="stream"))

    compile_with(lambda s: s["family"].update(name="ft-transformer",
                                              source="vega:default",
                                              catalog="families/ft-transformer"),
                 "D1  a heavy family under an online latency, unquoted",
                 ("latency", "FORK"),
                 pipeline=lambda p: p["serving"].update(mode="online", latency_ms=15))

    compile_with(lambda s: s["loss"].update(name="focal", params={"gamma": 2}),
                 "M2  a loss contradicting a declared Network stance",
                 ("stance", "user: source"))

    compile_with(lambda s: s["family"].update(name="ft-transformer", source="vega:default",
                                              catalog="families/ft-transformer"),
                 "M2  a family the user declined, used without saying they changed their mind",
                 ("declined", "user: source"))


# --------------------------------------------------------------------------- gate negatives

def conformance_negatives():
    print("\nconformance.py — the mechanical gate")
    pkg = RUN / "orders_pipeline"

    with tempfile.TemporaryDirectory() as td:
        copy_ = Path(td) / "orders_pipeline"
        shutil.copytree(pkg, copy_)
        (copy_ / "model" / "helper.py").write_text("# a file nobody mapped\nX = 1\n")
        rc, out = run([S / "conformance.py", "--package", copy_, "--pipeline", PIPE])
        expect("C3  a file with no map entry is refused", rc, out,
               contains=("helper.py", "designer's choice"))

    with tempfile.TemporaryDirectory() as td:
        copy_ = Path(td) / "orders_pipeline"
        shutil.copytree(pkg, copy_)
        loop = copy_ / "training" / "loop.py"
        # seed the defect before the adapter's own `from ..domain import …` line, whatever
        # names that line imports today (the template's import list changes with the domain)
        loop.write_text(loop.read_text().replace(
            "from ..domain import ",
            "from ..ingestion.data import load_part  # an adapter importing an adapter\n"
            "from ..domain import ", 1))
        rc, out = run([S / "conformance.py", "--package", copy_, "--pipeline", PIPE])
        expect("C5  an adapter importing another adapter is refused", rc, out,
               contains=("never on another adapter", "pass it in as a port"))

    with tempfile.TemporaryDirectory() as td:
        copy_ = Path(td) / "orders_pipeline"
        shutil.copytree(pkg, copy_)
        data = copy_ / "ingestion" / "data.py"
        # the row package's old shape dispatch put back: a lazy import of the window module,
        # which a row package does not hold (it fails only when its branch runs)
        data.write_text(data.read_text() + (
            "\n\ndef _windows(spec, part, frames):\n"
            "    from .windows import WindowParts\n"
            "    return WindowParts(frames).part(spec, part)\n"))
        rc, out = run([S / "conformance.py", "--package", copy_, "--pipeline", PIPE])
        expect("C4a a lazy import of a module the package does not hold is refused", rc, out,
               contains=("'ingestion.windows'", "does not hold", "shape"))

    with tempfile.TemporaryDirectory() as td:
        copy_ = Path(td) / "orders_pipeline"
        shutil.copytree(pkg, copy_)
        data = copy_ / "ingestion" / "data.py"
        # the seal removed from load_part (on the adapter class or at module level), whatever
        # the template's wording: its `raise SealedPartError(` becomes a ValueError
        text = data.read_text()
        head, sep, tail = text.partition("def load_part(")
        end = tail.find("\n    def ") if "\n    def " in tail else len(tail)
        seeded = head + sep + tail[:end].replace("raise SealedPartError(", "raise ValueError(",
                                                 1) + tail[end:]
        assert seeded != text, "the C8 seed found no `raise SealedPartError(` in load_part"
        data.write_text(seeded)
        rc, out = run([S / "conformance.py", "--package", copy_, "--pipeline", PIPE])
        expect("C8  a DataSource that does not raise on the test part is refused", rc, out,
               contains=("SealedPartError", "not of anyone's discipline"))

    with tempfile.TemporaryDirectory() as td:
        copy_ = Path(td) / "orders_pipeline"
        shutil.copytree(pkg, copy_)
        dom = copy_ / "domain.py"
        dom.write_text(dom.read_text().replace(
            "@dataclass(frozen=True)\nclass Prediction:", "@dataclass\nclass Prediction:"))
        rc, out = run([S / "conformance.py", "--package", copy_, "--pipeline", PIPE])
        expect("C6  a mutable artifact in domain.py is refused", rc, out,
               contains=("Prediction", "frozen dataclass"))

    with tempfile.TemporaryDirectory() as td:
        copy_ = Path(td) / "orders_pipeline"
        shutil.copytree(pkg, copy_)
        bmap = json.loads((copy_ / "blueprint-map.json").read_text())
        # the blueprint's external campaign forgotten by the map: an artifact with no home
        assert bmap.get("external_artifacts"), "the C6 roster seed found no external artifact"
        bmap["external_artifacts"] = []
        (copy_ / "blueprint-map.json").write_text(json.dumps(bmap))
        rc, out = run([S / "conformance.py", "--package", copy_, "--pipeline", PIPE])
        expect("C6  a blueprint artifact the map does not account for is refused", rc, out,
               contains=("Campaign", "external_artifacts"))

    with tempfile.TemporaryDirectory() as td:
        copy_ = Path(td) / "orders_pipeline"
        shutil.copytree(pkg, copy_)
        bmap = json.loads((copy_ / "blueprint-map.json").read_text())
        bmap["unmapped"] = ["model/somewhere.py"]
        (copy_ / "blueprint-map.json").write_text(json.dumps(bmap))
        rc, out = run([S / "conformance.py", "--package", copy_, "--pipeline", PIPE])
        expect("C3  a non-empty `unmapped` is refused", rc, out,
               contains=("declares 1 unmapped file", "must be empty"))


_SIZE_LINE = re.compile(r"^(ERROR|WARN): C1[1-4] ")


def _size_lines(out) -> int:
    return sum(1 for ln in out.splitlines() if _SIZE_LINE.match(ln))


def _bundle_package():
    """The package of a bundle /vega-pack wrote in this work dir, when there is one: C14e is
    then seeded in the real serving slice rather than in the forge package marked as one."""
    for manifest in sorted((HERE / "bundle").glob("*/MANIFEST.json")):
        try:
            package = manifest.parent / json.loads(manifest.read_text())["package"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (package / "blueprint-map.json").is_file():
            return package
    return None


def conformance_size_negatives():
    print("\nconformance.py — the size discipline (C0, C11–C14), under --size-severity error")
    pkg = RUN / "orders_pipeline"
    bundle_pkg = _bundle_package()
    # The unseeded package's own verdict at this severity: today's templates carry findings,
    # so a boundary seed must leave the exit where it was AND add no finding of its own, and a
    # refused seed must add at least one. Measured once per package the seeds are planted in.
    bases = {}
    for seed in seeds.SEEDS:
        source = bundle_pkg if (seed.wants_bundle and bundle_pkg is not None) else pkg
        if source not in bases:
            extra = ["--bundle"] if source == bundle_pkg else []
            rc0, out0 = run([S / "conformance.py", "--package", source, "--pipeline", PIPE,
                             "--size-severity", "error", *extra])
            bases[source] = (rc0, _size_lines(out0))
        base_rc, base = bases[source]
        with tempfile.TemporaryDirectory() as td:
            try:
                seeded, planted = seeds.plant(seed, source, PIPE, Path(td))
            except seeds.SeedError as exc:
                record(seed.name, [f"the seed could not be planted: {exc}"], "")
                continue
            rc, out = run(seeds.command(seeded, PIPE, planted)[1:])
        problems = seeds.problems(seed, planted, rc, out, base_rc)
        record(seed.name, problems + _count_problems(seed, _size_lines(out), base), out)


def _count_problems(seed, n, base):
    """A refused seed adds at least one size finding; a boundary seed adds none (C0a stops the
    gate before any is measured)."""
    if seed.id != "C0a" and seed.refused and n <= base:
        return [f"{n} size finding(s), no more than the unseeded package's {base}"]
    if not seed.refused and n != base:
        return [f"{n} size finding(s) where the unseeded package has {base}"]
    return []


def emit_negatives():
    print("\nemit_backend.py — the emitter refuses what it may not invent")
    with tempfile.TemporaryDirectory() as td:
        pl = copy.deepcopy(PIPELINE)
        pl["ports"] = [p for p in pl["ports"] if p["role"] != "ModelStore"]
        pipe = Path(td) / "pipeline.json"
        pipe.write_text(json.dumps(pl))
        rc, out = run([S / "emit_backend.py", "--spec", HERE / "netspec.json",
                       "--pipeline", pipe, "--out", Path(td) / "run"])
        expect("K1  a port the blueprint did not draw is not invented", rc, out,
               contains=("does not invent a port", "--concern"))

    with tempfile.TemporaryDirectory() as td:
        pl = copy.deepcopy(PIPELINE)
        undeclared = [a for a in pl.get("artifacts") or [] if a.pop("external", None)]
        assert undeclared, "the roster seed found no external artifact in the fixture's blueprint"
        pipe = Path(td) / "pipeline.json"
        pipe.write_text(json.dumps(pl))
        rc, out = run([S / "emit_backend.py", "--spec", HERE / "netspec.json",
                       "--pipeline", pipe, "--out", Path(td) / "run"])
        expect("K1  a blueprint artifact with no home is refused, never dropped from the roster",
               rc, out, contains=(repr(undeclared[0]["name"]), "not declared external",
                                  "--concern"))

    with tempfile.TemporaryDirectory() as td:
        spec = copy.deepcopy(SPEC)
        spec["blocks"][2]["params"]["units"] = 64
        spec["blocks"][3]["params"]["inner"][0]["params"]["units"] = 999
        spec["blocks"][3]["params"]["projection"] = "none"
        sp = Path(td) / "netspec.json"
        sp.write_text(json.dumps(spec))
        rc, out = run([S / "emit_backend.py", "--spec", sp, "--pipeline", PIPE,
                       "--out", Path(td) / "run", "--smoke"])
        expect("V1  a spec that does not build fails the smoke pass", rc, out, code=3,
               contains=("smoke forward pass failed", "no architecture layer may be written"))


_SHAPE_PROBE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from orders_pipeline import domain
spec = json.loads(open(sys.argv[2]).read())
spec["data_contract"]["sample"] = {"shape": "window", "length": 4, "stride": 1, "horizon": 1,
                                   "series_by": None, "order_by": None, "context": "none"}
try:
    domain.NetSpec.from_dict(spec)
except domain.UnsupportedShapeError as exc:
    print(f"ERROR: {exc}")
    sys.exit(2)
print("the row package accepted a window spec")
"""


def shape_negatives():
    print("\nthe emitted package — one sample shape, its spec's")
    with tempfile.TemporaryDirectory() as td:
        probe = Path(td) / "probe.py"
        probe.write_text(_SHAPE_PROBE)
        rc, out = run([probe, RUN, HERE / "netspec.json"])
        expect("K2  a row package refuses a window spec by name", rc, out,
               contains=("'window'", "carries the row sample shape only", "/vega-forge"))


def _layer_data(html_path):
    """The `layer-architecture-data` document an apply wrote, or {}."""
    text = Path(html_path).read_text(encoding="utf-8") if Path(html_path).is_file() else ""
    m = re.search(r'<script id="layer-architecture-data"[^>]*>([\s\S]*?)</script>', text)
    return json.loads(m.group(1)) if m else {}


def apply_negatives():
    print("\napply_architecture_layer.py — what may not be published")

    def apply_with(mutate, name, contains, code=2):
        doc = copy.deepcopy(LAYER)
        mutate(doc)
        with tempfile.TemporaryDirectory() as td:
            dp = Path(td) / "architecture-layer.json"
            doc["from_run"] = str(RUN)
            dp.write_text(json.dumps(doc))
            rc, out = run([S / "apply_architecture_layer.py", MODEL, "--data", dp,
                           "--notebook", RUN / "engineer.ipynb", "--architect-run", ARCH,
                           "--out", Path(td) / "out.html"])
        expect(name, rc, out, code=code, contains=contains)

    apply_with(lambda d: d.__setitem__("smoke", {"forward_ok": False, "params_total": 1}),
               "V1  a layer whose smoke failed is refused",
               ("forward_ok is not true", "never published"))

    apply_with(lambda d: d.__setitem__("conformance", {"mechanical": "fail"}),
               "K1  a layer whose mechanical gate did not pass is refused",
               ("mechanical", "conformance.py"))

    apply_with(lambda d: d["forks"][0].update(**{"or": "default applied (unattended)"}),
               "F1  a fork carrying both an answer and an `or` is refused",
               ("both `answer` and `or`",))

    apply_with(lambda d: d["readings"].pop("mechanics"),
               "R1  a layer with no mechanics reading is refused",
               ("readings.mechanics",))

    # F40: a defaults_to_test entry is a rung's delta — the trainer's grammar, on a path the
    # emitted code reads; the rehearsal's `regularization[dropout]` was published and then
    # dropped by the ladder, so the dropout hypothesis was never trained
    apply_with(lambda d: d["defaults_to_test"].append(
                   {"path": "regularization[dropout]",
                    "observation": "a train/valid gap under 0.05 would mean dropout is not needed"}),
               "DT1  a defaults_to_test path no rung can move is refused",
               ("regularization[dropout]", "regularization[kind=dropout]/value"))

    def _unseal(d):
        # the layer normally splices its netspec from the run; here it carries one that no
        # longer compiles, which is the case the check exists for
        spec = json.loads((RUN / "netspec.json").read_text())
        spec["data_contract"]["parts"]["test"] = "shaped/orders.test.parquet"
        d["netspec"] = spec

    apply_with(_unseal, "S1  a layer whose spec no longer compiles is refused",
               ("netspec", "sealed"))

    # the judgement half (code-contract §5): a structural or divergence finding is confirmed
    # when two of three independent fresh audits report it, and a confirmed finding is advisory
    # — unattended a WARN and an open item (the chain continues), attended the user's recorded
    # FORK answer; a first audit that reports one is never published alone; the mechanical
    # half, a malformed record and a duplicated run id still refuse
    def audits(td, *mutations, tool="architect"):
        """One audit run dir per mutation of the recorded fixture run, the first audit first;
        `None` is an audit that found nothing."""
        runs, root = [], Path(td) / f"{tool}-runs"
        for mutate in mutations:
            n = len(list(root.glob("audit-*"))) + 1 if root.is_dir() else 1
            run = root / f"audit-{n}"
            run.mkdir(parents=True)
            found = json.loads((ARCH / "findings.json").read_text())
            if mutate is None:
                found["findings"] = []
            else:
                mutate(found["findings"])
            found["run"] = f"audit-{n}"
            (run / "findings.json").write_text(json.dumps(found))
            runs.append(run)
        return runs

    def gate_with(td, architect=(), improve=(), mutate=None, unattended=False):
        doc = copy.deepcopy(LAYER)
        doc["from_run"] = str(RUN)
        if mutate:
            mutate(doc)
        dp = Path(td) / "architecture-layer.json"
        dp.write_text(json.dumps(doc))
        out_html = Path(td) / "out.html"
        cmd = [S / "apply_architecture_layer.py", MODEL, "--data", dp, "--notebook",
               RUN / "engineer.ipynb", "--out", out_html, *(["--unattended"] if unattended else [])]
        for flag, runs in (("--architect-run", architect), ("--improve-run", improve)):
            for r in runs:
                cmd += [flag, r]
        rc, out = run(cmd)
        return rc, out, out_html

    def structural(found):
        found[0]["depth"] = "structural"

    def elsewhere(found):
        found[0].update(file="model/builder.py", anchors=["EJ-51"], axis="Model")

    with tempfile.TemporaryDirectory() as td:
        rc, out, _ = gate_with(td, architect=audits(td, structural))
        # a single audit's structural finding is not published until two fresh audits judge it
        expect("K1  a structural architect finding may not be published", rc, out,
               contains=("first audit reported 1 structural", "fresh context",
                         "never published unjudged"))

    # nor an improve one (the T5 round 1 rehearsal: --improve-run was documented and not
    # accepted, so an improve finding never reached the gate)
    with tempfile.TemporaryDirectory() as td:
        rc, out, _ = gate_with(td, architect=[ARCH], improve=audits(td, structural, tool="improve"))
        expect("K1b a single improve audit's structural finding is not published unjudged", rc,
               out, contains=("conformance.improve", "first audit reported 1 structural"))

    with tempfile.TemporaryDirectory() as td:
        rc, out, _ = gate_with(td, architect=audits(td, structural, lambda f: None, None))
        expect("K1c attended, a finding two of three audits confirm is not published without the "
               "user's recorded FORK answer", rc, out,
               contains=("confirmed by audits 1, 2 of 3", "no answer is recorded", "FORK:",
                         "fix now", "accept and record", "stop"))

    def confirmed_published(rc, out, html, tool, mode):
        """The layer was written with the confirmed finding as a WARN marker, an open item in
        the tool's advisory block and the finding's `open_item` — the chain continues."""
        layer = _layer_data(html) if rc == 0 else {}
        rec = (layer.get("conformance") or {}).get(tool) or {}
        advisory = rec.get("advisory") or {}
        items = advisory.get("open_items") or []
        rows = [f for f in layer.get("code_findings") or [] if f.get("open_item")]
        warns = [m for m in layer.get("markers") or [] if str(m).startswith(f"WARN: {tool} 1:")]
        problems = [] if rc == 0 else [f"exit {rc}, expected 0 (published, the chain continues)"]
        if rc == 0 and not (rec.get("structural") == 1 and advisory.get("mode") == mode
                            and advisory.get("confirmed") == ["1:A-1"] and len(items) == 1
                            and "open item for /architect and /improve" in items[0]["text"]
                            and len(rows) == 1 and rows[0]["tool"] == tool and len(warns) == 1):
            problems.append(f"the layer does not carry the advisory finding: {json.dumps(rec)[:400]}")
        if rc == 0 and "WARN: " + tool not in out:
            problems.append("the apply output says no WARN")
        return problems

    with tempfile.TemporaryDirectory() as td:
        rc, out, html = gate_with(td, architect=[ARCH], improve=audits(
            td, structural, lambda f: f.__setitem__(slice(None), f[:1]), None, tool="improve"),
            unattended=True)
        record("K1d unattended, a structural improve finding two of three audits confirm is a WARN "
               "and an open item, and the layer is written",
               confirmed_published(rc, out, html, "improve", "unattended"), out)

    with tempfile.TemporaryDirectory() as td:
        rc, out, html = gate_with(td, architect=audits(td, structural, lambda f: None, None),
                                  unattended=True)
        problems = confirmed_published(rc, out, html, "architect", "unattended")
        if rc == 0 and "advisory (unattended)" not in out:
            problems.append("the WARN does not say the finding is advisory (unattended)")
        record("K1j unattended, a confirmed architect finding never stops the chain: WARN, open "
               "item, layer written", problems, out)

    def gate_fork(answer):
        return lambda d: d["forks"].append({"family": "conformance", "gate": "architect",
                                            "findings": ["1:A-1"], "asked": "FORK: architect 1:A-1",
                                            "answer": answer})

    with tempfile.TemporaryDirectory() as td:
        rc, out, html = gate_with(td, architect=audits(td, structural, lambda f: None, None),
                                  mutate=gate_fork("accept and record — the user keeps it"))
        problems = confirmed_published(rc, out, html, "architect", "attended")
        if rc == 0 and (_layer_data(html)["conformance"]["architect"]["advisory"]["outcome"]
                        != "accepted"):
            problems.append("the advisory outcome is not `accepted`")
        record("K1k attended, the user's recorded `accept and record` publishes the confirmed "
               "finding as a WARN and an open item", problems, out)

    for answer, said in (("fix now", "answered fix now"), ("stop", "answered stop")):
        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = gate_with(td, architect=audits(td, structural, lambda f: None, None),
                                   mutate=gate_fork(answer))
            expect(f"K1l attended, the answer `{answer}` publishes nothing", rc, out,
                   contains=(said,))

    with tempfile.TemporaryDirectory() as td:
        rc, out, _ = gate_with(td, architect=audits(td, structural, lambda f: None, None),
                               mutate=lambda d: d["forks"].append({
                                   "family": "conformance", "gate": "architect",
                                   "findings": ["1:A-1"], "asked": "FORK: architect 1:A-1",
                                   "or": "default applied (unattended)"}))
        expect("K1m attended, a default `or` is not the user's answer", rc, out,
               contains=("carries `or`", "the answer is the user's"))

    with tempfile.TemporaryDirectory() as td:
        rc, out, html = gate_with(td, architect=audits(td, structural, elsewhere, None))
        layer = _layer_data(html) if rc == 0 else {}
        arch = (layer.get("conformance") or {}).get("architect") or {}
        rows = [f for f in layer.get("code_findings") or [] if f.get("quorum")]
        problems = [] if rc == 0 else [f"exit {rc}, expected 0 (published)"]
        if rc == 0 and not (arch.get("structural") == 0 and arch.get("unconfirmed", {}).get(
                "structural") == 1 and len(arch.get("audits") or []) == 3
                and [r["reported_by"] for r in arch.get("matching") or []] == [[1]]
                and rows and rows[0]["quorum"] == {"reported_by": [1], "confirmed": False}):
            problems.append(f"the layer does not record the unconfirmed finding and the three "
                            f"audits: {json.dumps(arch)[:400]}")
        if "unconfirmed" not in out:
            problems.append("the apply output does not say the finding is unconfirmed")
        record("K1e an unconfirmed structural finding is recorded with its audits and does not "
               "refuse", problems, out)

    with tempfile.TemporaryDirectory() as td:
        rc, out, _ = gate_with(td, architect=audits(td, structural, lambda f: None, None),
                               mutate=lambda d: d.__setitem__("conformance", {"mechanical": "fail"}),
                               unattended=True)
        expect("K1f the mechanical half still refuses whatever the judgement half found, "
               "unattended too", rc, out,
               contains=("conformance.mechanical is 'fail'", "conformance.py"))

    with tempfile.TemporaryDirectory() as td:
        rc, out, _ = gate_with(td, architect=audits(td, structural, elsewhere))
        expect("K1g two audits are not a judgement: the first and two fresh ones are", rc, out,
               contains=("2 architect audits were given", "never another number"))

    with tempfile.TemporaryDirectory() as td:
        first = audits(td, structural)[0]
        rc, out, _ = gate_with(td, architect=[first, first, audits(td, None)[0]])
        expect("K1h one audit given twice is not an independent audit", rc, out,
               contains=("are the same run", "its own fresh run"))

    with tempfile.TemporaryDirectory() as td:
        first = audits(td, structural)[0]
        twin = Path(td) / "copy" / ".claude" / "architect-runs" / first.name
        twin.mkdir(parents=True)
        found = json.loads((first / "findings.json").read_text())
        found["findings"][0]["title"] = "another audit, started in the same second"
        (twin / "findings.json").write_text(json.dumps(found))
        rc, out, _ = gate_with(td, architect=[first, twin, audits(td, None)[0]], unattended=True)
        expect("K1n two audits that share a run id are refused, unattended too (F31)", rc, out,
               contains=("share the run id", "one after another"))

    with tempfile.TemporaryDirectory() as td:
        def typed(d):
            d["conformance"]["architect"] = {"ran": True, "structural": 0, "divergence": 0,
                                             "recorded": 0}
        rc, out, _ = gate_with(td, mutate=typed)
        expect("K1i a judgement typed into the layer instead of spliced from audits is refused",
               rc, out, contains=("never typed into the layer", "--architect-run"))

    # an unexecuted notebook must be a WARN in the markers, not a silence
    with tempfile.TemporaryDirectory() as td:
        nb = json.loads((RUN / "engineer.ipynb").read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = []
        nbp = Path(td) / "engineer.ipynb"
        nbp.write_text(json.dumps(nb))
        doc = copy.deepcopy(LAYER)
        doc["from_run"] = str(RUN)
        dp = Path(td) / "architecture-layer.json"
        dp.write_text(json.dumps(doc))
        rc, out = run([S / "apply_architecture_layer.py", MODEL, "--data", dp,
                       "--notebook", nbp, "--architect-run", ARCH,
                       "--out", Path(td) / "out.html"])
        ok = rc == 0 and "WARN: notebook not executed" in out
        results.append(("N1  an unexecuted notebook is published with its WARN", ok,
                        "" if ok else f"exit {rc}, no WARN in output"))
        print(f"  {'ok  ' if ok else 'FAIL'}  "
              "N1  an unexecuted notebook is published with its WARN")


def nb_negatives():
    print("\nnb_emit.py — a stage whose records do not exist")
    # every stage is implemented now, and each refuses for its own reason: a directory that
    # is not a bundle has no manifest to document
    with tempfile.TemporaryDirectory() as td:
        rc, out = run([S / "nb_emit.py", "--stage", "pack", "--from", td,
                       "--out", Path(td) / "x.ipynb"])
        expect("N1  --stage pack on a directory that is not a bundle is refused", rc, out,
               contains=("MANIFEST.json", "pack.py"))

    # `--stage train` and `--stage eval` are implemented, and refuse for their own reasons: a
    # run directory that is not a campaign has nothing to write a campaign notebook from, and
    # an evaluation notebook documents a look that has already happened. The forge's own run
    # directory becomes a campaign once /vega-train has planned in it, so these two are asked
    # about a directory that is a run and nothing more.
    with tempfile.TemporaryDirectory() as td:
        bare = Path(td) / "not-a-campaign"
        bare.mkdir()
        (bare / "netspec.json").write_text((RUN / "netspec.json").read_text(encoding="utf-8"),
                                           encoding="utf-8")
        rc, out = run([S / "nb_emit.py", "--stage", "train", "--from", bare,
                       "--out", Path(td) / "x.ipynb"])
        expect("N1  --stage train on a run directory that is not a campaign is refused",
               rc, out, contains=("campaign.json", "plan_ladder.py"))

        rc, out = run([S / "nb_emit.py", "--stage", "eval", "--from", bare,
                       "--out", Path(td) / "x.ipynb"])
        expect("N1  --stage eval on a directory that is not a campaign is refused", rc, out,
               contains=("campaign.json",))


def write_json(path) -> None:
    """`vega/negatives-result@1`: one entry per case, in the order the cases ran."""
    cases = []
    for name, ok_, detail in results:
        m = CASE_ID.match(name)
        cases.append({"id": m.group(1) if m else name.split()[0], "name": name, "ok": ok_,
                      "detail": detail})
    passed = sum(1 for c in cases if c["ok"])
    doc = {"schema": "vega/negatives-result@1",
           "script": Path(__file__).resolve().relative_to(S.parent.parent).as_posix(),
           "work": str(HERE), "cases": cases, "passed": passed, "failed": len(cases) - passed}
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def main():
    spec_negatives()
    conformance_negatives()
    conformance_size_negatives()
    emit_negatives()
    shape_negatives()
    apply_negatives()
    nb_negatives()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} of {len(results)} seeded defects were caught by "
          "the gate that should catch them")
    if ARGS.json:
        write_json(ARGS.json)
        print(f"wrote {ARGS.json} (vega/negatives-result@1)")
    if failed:
        print("\nnot caught:")
        for name, _, why in failed:
            print(f"  - {name}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
