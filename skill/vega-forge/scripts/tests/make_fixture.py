#!/usr/bin/env python3
"""Build the `orders` fixture the vega-forge scripts are verified against.

    python3 make_fixture.py <work-dir>

The fixture is a real domain-forge model (its geometry, analysis and shape layers
written through the platform writer, not pasted in), real parquet parts with a real
shaped manifest, a real blueprint pipeline, and a netspec that should compile. A
fixture that cut corners on any of those would exercise a file no chain produces.

Writes, under the directory given as argv[1] (default `.`):

  orders.csv                                 the RAW dataset, in the recipe's input schema
  orders.recipe.json                         the recipe /dataset-shaper executes
  shaped/orders.{train,valid,test}.parquet   the shaped parts — written by shape.py, not here
  shaped/manifest.json  fitted.json  recipe.json  reproduce_orders.py
                                             the shaper's own outputs, so the raw rows behind
                                             every shaped row can be recovered (which is what
                                             /vega-pack's round trip needs)
  blueprint-runs/<ts>/pipeline.json          the blueprint's world model in machine form
  blueprint-runs/<ts>/blueprint.md           the baseline /architect --baseline reads
  orders.model-source/                       the domain-forge model source folder (meta,
                                             overview, entities/Order.html) and the base
                                             model compose_model.py composes from it
  orders.domain.analysis.shaped.html         a domain-forge model with geometry/analysis/shape
  netspec.json                               a spec that should compile
  .claude/domain-forge-memory.md             a memory file with a Network stance
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SKILLS = HERE.parent.parent.parent
# The model source writer, the base-model composer, the world model and the chain's tool runner
# are the lane program's (lanekit, D6 §5.2): every lane fixture builds its base model, its world
# model and its layers with the same functions this fixture calls.
sys.path.insert(0, str(SKILLS / "vega" / "scripts" / "tests"))
from lanekit.chain import FixtureError, platform_writer, run_tool, script, sha_file  # noqa: E402
from lanekit.layers import NOOP  # noqa: E402
from lanekit.model_source import ModelEntity, ModelSourceError, compose_base_model, write_model_source  # noqa: E402
from lanekit.pipeline import WorldModel  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
TS = "2026-09-05T10-12Z"
RNG = np.random.default_rng(7)

NUMERIC = ["unit_price", "qty", "discount", "weekday", "hour", "dist_km", "promo"]
NOMINAL = {"zip": 42, "region": 6}          # cardinality = unseen_category_id + 1
LABEL = "late"
RAW_ROWS = 740


def make_raw(n):
    """The dataset before anything was done to it.

    On their own scales and with their own dtypes, because the point of shaping through the
    real `shape.py` is that the recipe's steps have something to do: a `scale` step over
    columns that are already standard normal proves nothing, and a `zip` that is already an
    integer never exercises the encoder /vega-pack's bundle has to reproduce.
    """
    df = pd.DataFrame({
        "order_id": [f"ORD-{i:06d}" for i in range(n)],
        "unit_price": np.round(RNG.gamma(3.0, 8.0, n) + 0.5, 2),
        "qty": RNG.integers(1, 25, n),
        "discount": np.round(RNG.beta(1.6, 9.0, n) * 60.0, 1),
        "weekday": RNG.integers(0, 7, n),
        "hour": RNG.integers(0, 24, n),
        "dist_km": np.round(RNG.gamma(2.0, 30.0, n) + 1.0, 1),
        "promo": RNG.integers(0, 2, n),
        "zip": [f"{60000 + int(z):05d}" for z in RNG.integers(0, 41, n)],
        "region": RNG.choice(["north", "south", "east", "west", "central"], n),
        "note": "n/a",                                       # constant: the recipe drops it
    })
    z = pd.to_numeric(df["zip"]) - 60000
    logit = (0.030 * (df.dist_km - 60.0) - 0.055 * (df.discount - 10.0)
             + 0.045 * (df.qty - 12.0) + 0.020 * z - 1.6 + RNG.normal(0, 0.5, n))
    p = 1 / (1 + np.exp(-logit))
    df[LABEL] = (p > 0.62).astype("int64")
    # a leakage column: known only after the fact, and the recipe drops it by name
    df["delivered_days"] = np.where(df[LABEL] == 1, RNG.integers(8, 20, n),
                                    RNG.integers(1, 8, n))
    return df


def recipe_for(raw_path, shaped_dir):
    """The recipe that turns those raw rows into the parts the rest of the chain reads.

    Every step carries the `source` the contract requires. The steps are the ones a real
    analysis would have justified on this dataset, and between them they exercise each kind
    of fitted parameter the bundle's transformer has to reproduce: an ordinal encoding with a
    reserved id for an unseen category, and a standard scaling fitted on train alone.
    """
    return {
        "schema": "dataset-shaper/recipe@1",
        "input": {"path": str(raw_path), "digest": sha_file(raw_path),
                  "rows": RAW_ROWS, "columns": 13},
        "seed": 7, "goal": "train-ready", "target": LABEL,
        "outputs": {"dir": str(shaped_dir), "format": "parquet", "files": []},
        "steps": [
            {"id": "S01", "op": "retype", "columns": ["zip", "region"],
             "params": {"to": "nominal"}, "rationale": "zip is a postal code and region a name; both arrive as text and neither is a quantity \u2014 the geometry layer types them nominal, and an untyped zip would be scaled as if 60614 were sixty thousand of something",
             "consequences": {"rows": "unchanged", "columns": "unchanged (two retyped)", "downstream": "zip and region become encodable"},
             "source": "geometry:typing"},
            {"id": "S02", "op": "drop_identity", "columns": ["order_id"],
             "rationale": "order_id is unique per row: it identifies, it does not predict",
             "consequences": {"rows": "unchanged", "columns": "-1", "downstream": "no identifier reaches the model"},
             "source": "geometry:basis"},
            {"id": "S03", "op": "drop_constant", "params": {"tolerance": 0.0},
             "rationale": "a column with one value carries no information and would only add a weight",
             "consequences": {"rows": "unchanged", "columns": "-1 (note)", "downstream": "no constant weight"},
             "source": "shaper:default"},
            {"id": "S04", "op": "select_partition",
             "params": {"label": LABEL, "task": "binary classification"},
             "rationale": "late is the question this dataset is being asked; the rest are the features",
             "consequences": {"rows": "unchanged", "columns": "unchanged", "downstream": "late is named as the label"},
             "source": "geometry:partition"},
            {"id": "S05", "op": "drop_leakage", "columns": ["delivered_days"],
             "rationale": "delivered_days is known only once the order has arrived, so a model trained on it would score perfectly and predict nothing",
             "consequences": {"rows": "unchanged", "columns": "-1", "downstream": "the leak is gone before the split"},
             "source": "analysis:F3/T1"},
            {"id": "S06", "op": "split",
             "params": {"kind": "stratified", "by": LABEL, "test": 0.16, "valid": 0.16},
             "rationale": "the rows are independent orders with no time ordering to respect, so a seeded random split is the honest one",
             "consequences": {"rows": "split 68/16/16", "columns": "unchanged", "downstream": "every fitted step below is fitted on train alone"},
             "source": "shaper:default"},
            {"id": "S07", "op": "encode", "columns": ["zip", "region"],
             "params": {"strategy": "ordinal"}, "rationale": "zip and region are nominal with a bounded vocabulary; an ordinal code with a reserved id for a category the training data never saw is what the embedding blocks consume",
             "consequences": {"rows": "unchanged", "columns": "unchanged (two encoded)", "downstream": "two integer codes with a reserved unseen id"},
             "source": "geometry:typing"},
            {"id": "S08", "op": "scale", "columns": NUMERIC,
             # op_scale takes its column set from params, not from the step's `columns`;
             # without this it scales every numeric column, the ordinal codes included, and
             # the embedding blocks are handed floats where they expect category ids
             "params": {"kind": "standard", "columns": NUMERIC}, "rationale": "the numeric columns span three orders of magnitude \u2014 cents against kilometres \u2014 and a network trained on them unscaled spends its first epochs undoing that",
             "consequences": {"rows": "unchanged", "columns": "unchanged", "downstream": "the seven numeric columns are standard on train"},
             "source": "shaper:default"},
        ],
    }


def shape_the_dataset(raw_path, shaped):
    """Run the real /dataset-shaper on the raw rows, rather than writing parts by hand.

    The fixture used to fabricate the shaped parquets and a manifest describing steps that had
    never run. Everything downstream of the shaper was then being verified against an artefact
    no shaper had produced — and /vega-pack, whose whole job is to reproduce the evaluation's
    predictions *from raw rows*, had no raw rows to start from and no fitted parameters to
    apply. Running the executor here costs a few seconds and makes the chain's first link real.
    """
    recipe_path = OUT / "orders.recipe.json"
    try:                            # /dataset-shaper is a sibling this chain depends on
        shaper = script(SKILLS, "dataset-shaper/scripts/shape.py")
        recipe_path.write_text(json.dumps(recipe_for(raw_path, shaped), indent=2), encoding="utf-8")
        run_tool([sys.executable, shaper, "--recipe", recipe_path, "--out-dir", shaped,
                  "--format", "parquet", "--seed", "7"], "shaper", timeout=600)
    except FixtureError as exc:     # no shaper, or a recipe it refuses: no fixture is built on it
        raise SystemExit(str(exc)) from exc
    return json.loads((shaped / "manifest.json").read_text(encoding="utf-8"))


MODEL_IRI = "http://example.org/orders#"
OVERVIEW = """# orders

An order is late or it is not. `late` is the attribute the chain partitions on, and every
other property describes the order at the moment it was placed — before anything is known
about how it will be delivered.

The domain has one class. That is deliberate: an order line is the unit of everything
downstream, and nothing in this problem needs a second entity to be modelled before the
data is shaped.

`distKm` and `region` are the two properties the analysis found fragile: the first drifts
between the parts, the second is a low-cardinality nominal that the network embeds rather
than reads as a number.
"""
# the Order class: its data properties (local name, xsd range, comment) and design rationale
ORDER = ModelEntity(
    base_iri=MODEL_IRI, name="Order", component="ordering",
    properties=[("unitPrice", "xsd:float", None), ("qty", "xsd:float", None), ("distKm", "xsd:float", None),
                ("zip", "xsd:string", None), ("region", "xsd:string", None),
                ("late", "xsd:boolean", "Whether the order arrived after its promised date.")],
    overview=OVERVIEW, comment="One order line as it was placed.",
    rationale={"choice": "One class, Order: an order line is the unit of everything downstream, and late is "
                         "one of its properties.",
               "alternatives": ["Order and Delivery as two classes, late on Delivery"],
               "tradeoffs": "Nothing is known about the delivery when the order is placed; a Delivery class would "
                            "hold only the label and the leakage column the recipe drops.", "anchors": []})


def write_blueprint(man_digest):
    """The blueprint's world model (lanekit.pipeline, shared with every lane fixture): each port
    carries its own interface (`signature[]`, blueprint §3) within the size discipline, and a
    module that owns a port is OOP; only `domain`, which owns none, is FP (code-contract §2,
    C14). Returns (pipeline.json, its canonical digest)."""
    runs = OUT / "blueprint-runs" / TS
    runs.mkdir(parents=True, exist_ok=True)
    world = WorldModel(name="orders-late-pipeline", package="orders_pipeline",
                       manifest_digest=man_digest, serving_mode="batch", input_dim=9,
                       label=LABEL, partition_source="[geometry:partitions/late]",
                       monitoring=(("dist_km drift", "[analysis:F3]"),), shape="row",
                       task="binary")
    pipeline = world.pipeline()
    (runs / "pipeline.json").write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    (runs / "blueprint.md").write_text(world.blueprint_md(), encoding="utf-8")
    # the canonical digest (netspec-contract §1, compiler rule D1e): the parsed document with
    # sorted keys and no whitespace — the emitter's blueprint-map digest, not the file's bytes
    digest = hashlib.sha256(json.dumps(pipeline, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    return runs / "pipeline.json", "sha256:" + digest


def layer_payloads(manifest, manifest_digest):
    """The geometry, analysis and shape layers, each with the skill that writes it. The
    geometry's partitions have the report contract's shape (a chosen label and candidates by
    label, dataset-forge report-contract §1), which `geometry:partitions/late` resolves against;
    the shape layer records the manifest file's digest as `apply_shape_layer.py` writes it."""
    geometry = {"schema": "dataset-forge/geometry@1",
                "partitions": {"chosen": LABEL, "provenance": "user-chosen",
                               "candidates": [{"label": LABEL, "task": "binary classification",
                                               "features": NUMERIC + list(NOMINAL),
                                               "input_dim": 9}]},
                "typing": {c: {"role": "numeric"} for c in NUMERIC} |
                          {c: {"role": "nominal", "cardinality": v - 1} for c, v in NOMINAL.items()},
                "space": {"dims_95": 6, "intrinsic": 5}}
    analysis = {"schema": "data-lens/analysis@1",
                "findings": [{"id": "F2", "title": "the label is imbalanced", "detail": "17 % positives on train"},
                             {"id": "F3", "title": "dist_km drifts between parts"},
                             {"id": "F4", "title": "420 training rows against wide layers"},
                             {"id": "F5", "title": "importance is non-linear"}],
                "modules": {"importance": {"family": "HistGradientBoosting", "auc": 0.79},
                            "drift": {"columns": ["dist_km"]}, "segments": {}},
                "context": {"time": None, "spatial": None}}
    shape = {"schema": "dataset-shaper/shape@1", "recipe": {"steps": manifest["steps"]}, "manifest": manifest,
             "manifest_digest": manifest_digest}
    return (("geometry", geometry, "/dataset-forge"), ("analysis", analysis, "/data-lens"),
            ("shape", shape, "/dataset-shaper"))


def write_model(manifest, manifest_digest):
    """The layered model: returns (its path, its domain digest).

    The base model is composed by domain-forge itself: a model source folder (one entity
    fragment carrying its data properties as siblings) run through compose_model.py into
    domain-forge's own template, so it carries everything a forged model carries — the canonical
    Turtle and its JSON-LD mirror, the projectors, the runtime — and passes every invariant
    validate_model.py holds a real model to. A base written by hand passed only the invariants
    its author knew about. The three predecessor layers are then written through the platform
    writer, so they sit in well-formed @LAYER blocks and strip_layer.py can remove them — a
    fixture whose layers were pasted in by hand would exercise a file no chain ever produces.
    T-box only, deliberately: an A-box would make validate_model.py require the @KG_RUNTIME
    block (invariants 17-19) — the SPARQL and reasoner runners that belong to domain-forge's own
    template. This fixture stands in for the chain's model, and what it has to exercise is
    invariants 13-16, the layer invariants /vega-forge is responsible for.
    """
    try:
        html = compose_base_model(write_model_source(OUT / "orders.model-source", ORDER), SKILLS)
    except ModelSourceError as exc:
        raise SystemExit(str(exc)) from exc
    writer = platform_writer(SKILLS)
    for name, payload, by in layer_payloads(manifest, manifest_digest):
        html = writer.apply_layer(html, name, json.dumps(payload), NOOP % (name, name),
                                  data_type="application/json", produced_by=by)
    model_path = OUT / "orders.domain.analysis.shaped.html"
    model_path.write_text(html, encoding="utf-8")
    return model_path, writer.domain_digest(html)


ORDERS_BLOCKS = [
    {"id": "B1", "type": "embedding", "params": {"columns": ["zip", "region"], "dims": {"zip": 12, "region": 4}},
     "source": "vega:default", "catalog": "defaults/embedding_dim",
     "rationale": "zip is ordinal-encoded per shape:S10 (41 categories + the reserved unseen id); an embedding "
                  "beats 42 one-hot inputs; dim 12 by the catalog's rule for cardinality 42, 4 for region's 6",
     "alternatives": [{"params": {"dims": {"zip": 4, "region": 2}}, "when": "if the embedding overfits (train >> valid)"}]},
    {"id": "B2", "type": "concat", "params": {}, "source": "vega:default", "catalog": "blocks/concat",
     "rationale": "structural: the numeric features must be joined to the embeddings before the first dense",
     "alternatives": []},
    {"id": "B2b", "type": "dense", "params": {"units": 64, "activation": "none"},
     "source": "vega:default", "catalog": "defaults/sizing",
     "rationale": "the sizing rule: clip(2*9, 32, 256) = 32, raised one step to 64 because F5 found non-linear "
                  "importance; depth 2 for 500 rows",
     "alternatives": [{"params": {"units": 32}, "when": "if train >> valid after R0"}]},
    {"id": "B3", "type": "residual",
     "params": {"repeat": 2, "projection": "auto", "inner": [
         {"type": "dense", "params": {"units": 64, "activation": "none"}}, {"type": "batchnorm", "params": {}},
         {"type": "activation", "params": {"name": "gelu"}}, {"type": "dropout", "params": {}}]},
     "source": "analysis:F5",
     "rationale": "importance found non-linear structure; two residual blocks of width 64 for 500 training rows",
     "alternatives": [{"params": {"repeat": 3}, "when": "if valid loss is still falling at the last epoch"}]},
    {"id": "B4", "type": "head", "params": {"out": 1, "activation": "none"}, "source": "geometry:partitions/late",
     "rationale": "one logit for a binary task; the loss applies the sigmoid", "alternatives": []}]
# three stances, because the compiler's M2 rule has three things to enforce: a family the user
# chose, a family the user declined, and a loss the user declared
MEMORY = ("# domain-forge memory\n\n## Network stances\n\n"
          "- 2026-09-05 | orders | family: mlp-residual (user); dropout B3: 0.2\n"
          "- 2026-09-05 | orders | declined family: ft-transformer — the serving budget will not carry it\n"
          "- 2026-09-05 | orders | loss: bce_with_logits (declared; the imbalance is the reason)\n")


def orders_spec(inputs, files):
    """The netspec over the orders parts, `input` as main digested it: every choice sourced."""
    def choice(source, rationale, alternatives, catalog=None):
        return {"source": source, **({"catalog": catalog} if catalog else {}), "rationale": rationale,
                "alternatives": alternatives}

    feature = [{"name": c, "dtype": "float32", "role": "numeric", "origin": "S11"} for c in NUMERIC]
    for name, card, step in (("zip", 42, "S10"), ("region", 6, "S09")):
        feature.append({"name": name, "dtype": "int64", "role": "nominal", "cardinality": card,
                        "unseen_category_id": card - 1, "origin": step,
                        "role_source": [f"geometry:typing/{name}", f"shape:{step}"]})
    return {
        "schema": "vega/netspec@1", "input": inputs, "seed": 7,
        "task": {"kind": "binary", "label": LABEL, "classes": [0, 1], "positive": 1,
                 "source": "geometry:partitions/late"},
        "data_contract": {
            "source": "shape:manifest/output_schema",
            "parts": {"train": files["train"]["path"], "valid": files["valid"]["path"], "test": "sealed"},
            "features": feature, "input_dim": 9, "sample": {"shape": "row"},
            "class_balance": {"train": {"0": 0.83, "1": 0.17}, "source": "analysis:F2"}},
        "family": {"name": "mlp-residual", **choice(
            "blueprint:posture", "tabular, input_dim 9, non-linear importance (F5) — the posture the blueprint bet on",
            [{"name": "mlp-plain", "when": "if depth <= 3 makes residual connections redundant"},
             {"name": "ft-transformer", "when": "if input_dim > 50 with many nominals"}])},
        "blocks": ORDERS_BLOCKS,
        "loss": {"name": "bce_with_logits", "params": {"pos_weight": 4.9}, "source": "analysis:F2",
                 "rationale": "17 % positives: weight the positive class by the inverse ratio",
                 "formula": "L = -[w.y.log s(z) + (1-y).log(1-s(z))]",
                 "alternatives": [{"name": "focal", "params": {"gamma": 2},
                                   "when": "if the easy negatives still dominate the gradient"}]},
        "metrics": {"primary": "pr_auc", "secondary": ["roc_auc", "f1", "log_loss", "brier"],
                    "selection": {"monitor": "valid.pr_auc", "mode": "max"}, **choice(
                        "analysis:F2", "imbalanced binary task: PR-AUC ranks the positive class; accuracy "
                        "would reward the majority",
                        [{"primary": "roc_auc", "when": "if the positive share on train is above 0.3"}])},
        # the weight decay and B3's dropout rate are declared once, in regularization[] (R2): the
        # emitted code reads them there, and a live copy beside them would be refused
        "optimizer": {"name": "adamw", "params": {"lr": 0.001, "betas": [0.9, 0.999]},
                      "source": "vega:default", "catalog": "defaults/optimizer",
                      "rationale": "the family catalog's default for tabular MLPs",
                      "update_rule": "m <- b1.m + (1-b1).g; v <- b2.v + (1-b2).g^2; "
                                     "th <- th - eta.(m^/(sqrt(v^)+eps) + lam.th)",
                      "alternatives": [{"name": "sgd", "params": {"lr": 0.01, "momentum": 0.9},
                                        "when": "if AdamW generalises worse (an ablation)"}]},
        "schedule": {"name": "cosine", "params": {"warmup_epochs": 3}, **choice(
            "vega:default", "warmup steadies AdamW's first steps; max_epochs 200 gives cosine a known horizon",
            [{"name": "plateau", "when": "if the run is shorter than 30 epochs"}], "defaults/schedule")},
        "regularization": [
            {"kind": "weight_decay", "value": 0.0001, **choice(
                "vega:default", "mild shrinkage; nothing in the analysis asks for more",
                [{"value": 0.001, "when": "if train >> valid"}], "defaults/weight_decay")},
            {"kind": "dropout", "value": 0.2, "where": ["B3"], **choice(
                "analysis:F4", "F4: 500 rows against 64-wide layers overfit without it",
                [{"value": 0.4, "when": "if the gap is still wide at the best epoch"}])},
            {"kind": "grad_clip", "value": 1.0, **choice(
                "vega:default", "a cheap guard against the NaN epoch",
                [{"value": None, "when": "as an ablation"}], "defaults/grad_clip")}],
        "early_stopping": {"monitor": "valid.pr_auc", "mode": "max", "patience": 10, "min_delta": 0.001, **choice(
            "vega:default", "the primary metric is what selection uses",
            [{"params": {"patience": 20}, "when": "if the valid metric is noisy"}], "defaults/early_stopping")},
        "init": {"scheme": "kaiming_uniform", "bias": "zeros", **choice(
            "vega:default", "gelu is relu-family; keeps activation variance at depth",
            [{"scheme": "xavier_uniform", "when": "if the activations move to tanh"}], "defaults/init")},
        "training": {"batch_size": 64, "max_epochs": 200, "device": "auto", "precision": "fp32",
                     "num_workers": 0, "budget": {"wall_clock_min": 20}, **choice(
                         "vega:default", "500 training rows: 8 batches per epoch, an epoch is cheap",
                         [{"batch_size": 128, "when": "if D5 (thrashing) fires"}],
                         ["defaults/batch_size", "defaults/max_epochs", "defaults/device", "defaults/precision"])},
        "backend": {"name": "pytorch", "min_version": "2.2", "hints": {}},
        "decision_families": {"architecture": ["family", "blocks"], "objective": ["loss", "metrics"],
                              "optimization": ["optimizer", "schedule", "training", "init"],
                              "regularization": ["regularization", "early_stopping"]},
        "outputs": {"run_dir": "vega-runs/2026-09-05T11-03Z/", "package": "orders_pipeline",
                    "code": ["orders_pipeline/model/network.py", "orders_pipeline/ingestion/data.py",
                             "orders_pipeline/training/loop.py", "orders_pipeline/model/metrics.py"],
                    "notebook": "engineer.ipynb"},
    }


def main():
    shaped = OUT / "shaped"
    shaped.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "orders.csv"
    make_raw(RAW_ROWS).to_csv(raw_path, index=False)
    manifest = shape_the_dataset(raw_path, shaped)
    files = manifest["outputs"]["files"]
    # One definition of the manifest digest across the chain (X-2): the sha256 of the
    # manifest.json FILE as the shaper wrote it — what the shape layer records as
    # `manifest_digest`, what the blueprint copies into pipeline.json, what the compiler
    # compares and what the emitted FeatureContract admits. Never the canonical-JSON digest
    # of the object: that was a second definition, and the emitted code refused it at load.
    man_digest = sha_file(shaped / "manifest.json")
    pipeline, pipe_digest = write_blueprint(man_digest)
    model_path, model_digest = write_model(manifest, man_digest)
    inputs = {"model": model_path.name, "model_digest": model_digest, "layers_read": ["geometry", "analysis", "shape"],
              "shaped_dir": str(shaped), "manifest": str(shaped / "manifest.json"), "manifest_digest": man_digest,
              "pipeline": str(pipeline), "pipeline_digest": pipe_digest}
    (OUT / "netspec.json").write_text(json.dumps(orders_spec(inputs, files), indent=2), encoding="utf-8")
    (OUT / ".claude").mkdir(exist_ok=True)
    (OUT / ".claude" / "domain-forge-memory.md").write_text(MEMORY, encoding="utf-8")
    print(f"OK: fixture written to {OUT}")
    summary = ", ".join(f"{k} {v['rows']} rows" for k, v in files.items())
    print(f"OK: parts {summary}")


if __name__ == "__main__":
    main()
