"""
demo_pack — the demonstration sample and its presets of a served bundle: `examples/demo.json`
(`vega/demo@2`) and `examples/presets.json` (`vega/presets@2`), which the service assembles into
`GET /v1/examples` (`vega/examples@2`, bundle-contract §7.4; build plan R-D8).

    demo_pack.write(ctx, predictor) -> {"examples.demo": …, "serve.demo_sample": …}  or {}
    demo_pack.build(inputs, predictor) -> DemoDocuments            the documents, nothing written

Drawn from the **training part only**, never the test part (seed 7), `--demo-sample` items per
stratum; strata per reading (`demo_strata`), one typical preset per stratum and the edge presets
the schema and the Recipe imply (`demo_presets`), each item and preset in the request form of
the bundle's sample shape (`demo_samples`): a row carries `row`; a window its last row as `rows`
with its `history`; a graph `rows` only (R-A3). **No English ships in the payload**: a stratum
has the data's own name or none, a preset a `kind` and `facts`, and the page renders
`preset.<kind>.name` and `preset.<kind>.text` from its own strings table. The demonstration's
attributions are not the bundle's (R-D9: `/vega-show` computes them through the inspector).

Every answer that decides a stratum or a preset is the bundle's own Predictor, over the raw rows
exactly as they are sent, read through `MANIFEST.output`'s field kinds — never a field name or a
class typed here. Row and graph bundles also get `examples/demo.csv` (the items' rows with their
truth column: a table a caller can upload); a window is a sequence and ships in demo.json only.

Nothing is written unless the service is (`--serve`, or a blueprint whose mode serves) and
`--demo-sample` is above zero. Both documents are checked against
`references/schemas/examples-2.schema.json` before they ship (`#/$defs/demo`,
`#/$defs/presets_file`); a document that fails is a defect here, `PackExit(3)` (without
`jsonschema` installed the check cannot run, and a WARN says so). An answer that does not carry
the output record's fields is refused by `build` (exit 3); `write` meets it only with a serving
adapter older than the record (v2.2, until the serving cutover), and then writes no
demonstration and says so with a WARN.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import demo_presets as dp
import demo_strata as ds
from demo_samples import DemoInputs, samples_for
from pack_context import PackExit, out_ok, out_warn
from vega_pack_core import CATALOG, digest_file

__all__ = ["DemoDocuments", "DemoInputs", "build", "check_documents", "recipe_module", "write"]

HERE = Path(__file__).resolve().parent
EXAMPLES = CATALOG["examples"]
DEMO_SCHEMA = EXAMPLES["files"]["demo"]["schema"]
PRESETS_SCHEMA = EXAMPLES["files"]["presets"]["schema"]
JSON_SCHEMA = HERE.parent / EXAMPLES["json_schema"]
CLASS_TRUTH = "class"


@dataclass(frozen=True)
class DemoDocuments:
    """What a demonstration is: `demo.json` and `presets.json` as documents, the demo.csv table
    (row and graph shapes; None for a window), what went wrong drawing it (`notes`, WARN lines)
    and what is merely so (`remarks`, OK lines)."""

    demo: dict
    presets: dict
    table: object
    notes: tuple
    remarks: tuple = ()

    @property
    def items(self) -> int:
        return len(self.demo["items"])


def write(ctx, predictor) -> dict:
    """The demonstration files of a served bundle and their manifest records; {} when none."""
    if not (ctx.serving and ctx.options.demo_sample):
        return {}
    per_stratum = int(ctx.options.demo_sample)
    if per_stratum < 0:
        raise PackExit(2, f"--demo-sample {per_stratum}: the items per stratum are a count, 0 "
                          "for none")
    inputs = _inputs(ctx, predictor, per_stratum)
    if inputs is None:
        out_warn("no demonstration sample: the shaper recorded no train part", ctx.markers)
        return {}
    try:
        docs = build(inputs, predictor)
    except dp.UnreadAnswer as unread:
        # Transitional (build plan §2.12(3)): until the serving cutover renders the v2.3
        # Predictor, a served bundle carries the v2.2 adapter, whose answer is not the output
        # record's; the pack goes on without a demonstration rather than stopping here.
        out_warn(f"no demonstration sample: {unread.lines[0]} (a serving adapter older than "
                 "the output record)", ctx.markers)
        return {}
    for note in docs.notes:
        out_warn(f"demonstration sample: {note}", ctx.markers)
    for remark in docs.remarks:
        out_ok(f"demonstration sample: {remark}", ctx.markers)
    if not docs.items:
        out_warn("no demonstration sample: no stratum of the train part holds an item",
                 ctx.markers)
        return {}
    if not check_documents(docs):
        out_warn("check not run: the demonstration documents against examples-2.schema.json "
                 "(jsonschema is not installed)", ctx.markers)
    record = _write_files(ctx.paths.examples, docs, per_stratum, inputs.seed)
    out_ok(f"demonstration sample: {docs.items} item(s) in {len(docs.demo['strata'])} "
           f"strata of the train part ({per_stratum} per stratum, seed {inputs.seed}), "
           f"{record['presets']['count']} preset(s) — "
           f"{record['presets']['typical']} typical, {record['presets']['edge']} edge; kinds "
           "and facts only, the page words them", ctx.markers)
    return {"examples.demo": record,
            "serve.demo_sample": {"per_stratum": per_stratum, "file": record["file"],
                                  "presets": record["presets"]["file"]}}


def _inputs(ctx, predictor, per_stratum):
    """The demo's inputs from the context: the train part's raw rows, the records the Predictor
    was built from (its manifest), the schema and the bundle's own Recipe module."""
    if not ctx.campaign.split.get("train"):
        return None
    info = predictor.info()
    transformer = ctx.port("FeatureTransformer").get("file") or "features/preprocess.py"
    return DemoInputs(train=ctx.campaign.part_rows("train").reset_index(drop=True),
                      output=dict(info.get("output") or {}),
                      sample=dict(info.get("sample") or {"shape": "row"}),
                      schema=ctx.schema, per_stratum=per_stratum,
                      recipe=recipe_module(ctx.paths.pkg / transformer))


def recipe_module(path):
    """The bundle's emitted Recipe adapter, imported by file so it reads the bundle's fitted
    parameters (the bundle's own `load` pointed VEGA_FITTED at them)."""
    spec = importlib.util.spec_from_file_location("_bundle_recipe_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- the documents

def build(inputs: DemoInputs, predictor) -> DemoDocuments:
    """The demonstration of `inputs`, drawn and answered by `predictor`; nothing is written."""
    samples = samples_for(inputs)
    reader = dp.AnswerReader(inputs.output)
    truths = ds.Truths.of(inputs, samples)
    drawn = ds.draw(inputs, samples, truths, lambda keys: reader.score(
        _answers(samples, predictor, reader, keys)))
    members = sorted({k for s in drawn.strata for k in s.members})
    ctx = dp.PresetContext(inputs, samples, truths,
                           _answers(samples, predictor, reader, members), reader)
    presets, base = [], None
    for stratum in drawn.strata:
        got = dp.typical(stratum, ctx)
        if got is not None:
            presets.append(got[1])
            base = got[0] if stratum.id == drawn.neutral else base
    edge = dp.edges(ctx, drawn.neutral, base) if base is not None else dp.Edges(())
    presets.sort(key=lambda p: p["stratum"] != drawn.neutral)
    return DemoDocuments(demo=_demo(inputs, samples, truths, drawn),
                         presets={"schema": PRESETS_SCHEMA, "presets": presets
                                  + list(edge.presets)},
                         table=_table(samples, drawn), notes=drawn.notes + edge.notes,
                         remarks=edge.remarks)


def _answers(samples, predictor, reader, keys):
    """The Predictor's answers to the candidates `keys`, held to the output record's fields."""
    got = samples.answers(predictor, keys)
    if len(keys):
        reader.check(got)
    return got


def _demo(inputs, samples, truths, drawn) -> dict:
    items = [{"stratum": s.id, **samples.item(k), "truth": truths.values.get(k)}
             for s in drawn.strata for k in s.members]
    demo = {"source": "train", "schema": DEMO_SCHEMA,
            "strata": [s.as_json() for s in drawn.strata], "items": items}
    if samples.shape == "row" and truths.kind == CLASS_TRUTH:
        # the console page's two lists (row class readings only): each item's row and class
        demo["rows"] = [it["rows"][0] for it in items]
        demo["labels"] = [it["truth"] for it in items]
    return demo


def _table(samples, drawn):
    keys = [k for s in drawn.strata for k in s.members]
    return samples.table(keys)


# --------------------------------------------------------------------------- checked, written

def check_documents(docs: DemoDocuments) -> bool:
    """Both documents against examples-2.schema.json (draft 2020-12): True when they validate,
    PackExit(3) naming the first error when one does not, False when `jsonschema` is not
    installed and the check could not run."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return False
    schema = json.loads(JSON_SCHEMA.read_text(encoding="utf-8"))
    for name, doc, file in (("demo", docs.demo, "demo.json"),
                            ("presets_file", docs.presets, "presets.json")):
        validator = Draft202012Validator({"$schema": schema["$schema"], "$defs": schema["$defs"],
                                          "$ref": f"#/$defs/{name}"})
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
        if errors:
            where = "/".join(map(str, errors[0].absolute_path)) or "(root)"
            raise PackExit(3, f"examples/{file} does not validate against {JSON_SCHEMA.name} "
                              f"#/$defs/{name} at {where}: {errors[0].message}")
    return True


def _write_files(examples: Path, docs: DemoDocuments, per_stratum, seed) -> dict:
    examples.mkdir(parents=True, exist_ok=True)
    demo_path, presets_path = examples / "demo.json", examples / "presets.json"
    _dump(demo_path, docs.demo)
    _dump(presets_path, docs.presets)
    table = None
    if docs.table is not None:
        docs.table.to_csv(examples / "demo.csv", index=False)
        table = {"file": "examples/demo.csv", "rows": int(len(docs.table)),
                 "digest": digest_file(examples / "demo.csv")}
    kinds = [p["group"] for p in docs.presets["presets"]]
    return {"schema": DEMO_SCHEMA, "file": "examples/demo.json", "source_part": "train",
            "seed": seed, "per_stratum": per_stratum,
            "strata": {s["id"]: s["n"] for s in docs.demo["strata"]},
            "items": docs.items, "digest": digest_file(demo_path), "csv": table,
            "presets": {"schema": PRESETS_SCHEMA, "file": "examples/presets.json",
                        "count": len(kinds), "typical": kinds.count("typical"),
                        "edge": kinds.count("edge"), "digest": digest_file(presets_path)},
            "note": "training rows — an illustration of every stratum, not evidence"}


def _dump(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
