"""
The ONE place every name this program consumes from another subsystem lives.

The verification program adds no product behaviour: it drives the chain's scripts and reads
what they wrote. Every file name, record key, route, schema id, closed set and CLI shape it
relies on is written down here, as BUILD-PLAN §1 fixes it, so that a rename upstream is one
edit in one file and a disagreement is found by reading one table (D6 §17.2).

A name here is data. Code that needs a consumed name imports it from this module; nothing in
`stages.py`, `observe.py` or the checks spells a foreign path or key on its own.

Sources, by section of BUILD-PLAN.md:
  R-A1 batch keys · R-A4 graph reference files · R-B3 inspection surface · R-C1 links ·
  R-C2 predictions layout · R-C4 answer keys · R-D1/R-D4/R-D6 serving routes and the slot ·
  R-E1 SHOWCASE.json · R-E4 motifs · R-E5 roles · R-E6 lexicon · R-E7 report roles ·
  R-F1 probe · R-F4 views · R-H3 page driver · R-H6 negatives · §4.8 lanes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

# --------------------------------------------------------------------------- schema ids
TEST_LANES = "vega/test-lanes@1"
LANE_OBSERVED = "vega/lane-observed@1"
KNOWN_FLAKES = "vega/known-flakes@1"
LANE_STAMP = "vega/lane-stamp@1"
MATRIX_SUMMARY = "vega/matrix-summary@1"
MATRIX_TIMING = "vega/matrix-timing@1"
EXTERNAL_PINS = "vega/external-pins@1"
NEGATIVES_RESULT = "vega/negatives-result@1"
PAGE_SMOKE = "vega/page-smoke@1"
PREDICTIONS_LAYOUT = "vega/predictions-layout@1"
NETWORK_VIEW = "vega/network-view@2"
EXPLANATION = "vega/explanation@2"
EXAMPLES = "vega/examples@2"
GRAPH_REFERENCE = "vega/graph-reference@1"
CODE_VIEW = "vega/code-view@2"
LABEL_ROLES = "vega/label-roles@1"

# --------------------------------------------------------------------------- stages (D6 §6)
# The ids a lane's stages carry in observed.json, stamps and timing. The acceptance driver (L01)
# runs forge…pack as the four acceptance scripts; the golden driver (G00) replaces them with
# golden/cic.py.
STAGES = ("fixture", "forge", "train", "eval", "pack", "bundle_api", "report", "show",
          "serve", "serve_check", "inspect", "page")
GOLDEN_STAGE = "golden"

# The stage at which each coverage dimension is proved (D6 §0; BUILD-PLAN §4 rule).
PROVED_AT = {"families": "page", "blocks": "page", "tasks": "page", "goals": "page",
             "sample_shapes": "page", "graph_levels": "page", "graph_protocols": "eval",
             "activations": "page", "losses": "train", "optimizers": "train",
             "schedules": "train", "init_schemes": "train", "metrics": "eval",
             "motifs": "page", "class_roles": "page", "views": "page", "report_paths": "show"}

# --------------------------------------------------------------------------- providers
# A stage is runnable once the files it drives exist in the skills tree. Until then the stage is
# `absent`, named with the work package that provides it (BUILD-PLAN §2): the lane stops there,
# which a non-strict run reports and a strict run (release, nightly, pr) fails.
TESTS_DIR = "vega/scripts/tests"
LANEKIT_DIR = TESTS_DIR + "/lanekit"
SERVING_TEMPLATES = "vega-forge/scripts/backends/pytorch/templates/serving"
SHOW_SCRIPTS = "vega-show/scripts"
SHOW_CATALOG = SHOW_SCRIPTS + "/catalog.json"
FORGE_CATALOG = "vega-forge/scripts/catalog.json"
PACK_CATALOG = "vega-pack/scripts/catalog.json"
BACKEND_COVERAGE = "vega-forge/scripts/backends/pytorch/coverage.json"

PROVIDERS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "fixture": ((LANEKIT_DIR + "/fixture.py", "T02"), (LANEKIT_DIR + "/model_source.py", "T02"),
                (LANEKIT_DIR + "/spec.py", "T02"), (LANEKIT_DIR + "/datasets/__init__.py", "T02")),
    "forge": ((LANEKIT_DIR + "/authoring.py", "T03"),),
    "train": ((LANEKIT_DIR + "/authoring.py", "T03"),),
    "eval": (("vega-eval/scripts/tests/write_authoring.py", "vega-eval"),),
    "pack": ((LANEKIT_DIR + "/authoring.py", "T03"),),
    "bundle_api": ((SERVING_TEMPLATES + "/tests/_oracle.py", "S11"),),
    "report": ((LANEKIT_DIR + "/report_stub.py", "T03"), (SHOW_CATALOG, "K3")),
    "show": ((SHOW_SCRIPTS + "/extract_facts.py", "H03"), (SHOW_SCRIPTS + "/draft_plan.py", "H04"),
             (SHOW_SCRIPTS + "/resolve_story.py", "H05"), (SHOW_SCRIPTS + "/build_showcase.py", "H06"),
             (SHOW_SCRIPTS + "/apply_show_layer.py", "H07"), (SHOW_SCRIPTS + "/smoke_show.py", "H07"),
             (LANEKIT_DIR + "/authoring.py", "T03")),
    "serve": ((SERVING_TEMPLATES + "/serve/routes_showcase.py", "I07"),),
    "serve_check": ((TESTS_DIR + "/serving_check.py", "T08"), (TESTS_DIR + "/check_kit.py", "T08")),
    "inspect": ((TESTS_DIR + "/inspect_check.py", "T08"), (TESTS_DIR + "/oracles.py", "T08"),
                (TESTS_DIR + "/check_kit.py", "T08")),
    "page": ((TESTS_DIR + "/page_smoke.py", "T09"), ("vega-show/scripts/smoke_page.py", "W15"),
             ("vega-show/scripts/cdp.py", "W15")),
    GOLDEN_STAGE: ((TESTS_DIR + "/golden/cic.py", "R2"), (TESTS_DIR + "/golden/cic_authoring.py", "R2")),
}

# The acceptance scripts L01 runs as its forge…pack stages (D6 §2.3). PACK_EXTRA is appended to
# both pack.py calls of the pack acceptance (steps 1 and 9).
ACCEPTANCE = {"forge": "vega-forge/scripts/tests/acceptance.sh",
              "train": "vega-train/scripts/tests/acceptance.sh",
              "eval": "vega-eval/scripts/tests/acceptance.sh",
              "pack": "vega-pack/scripts/tests/acceptance.sh"}
PACK_EXTRA = "PACK_EXTRA"
# The acceptance scripts' own suites (the coverage matrices, the zoo, the page fixtures, the
# showcase negatives) read no work dir; L01 runs the scripts with this variable set to "lane" and
# they say what they leave to the standalone run and to the matrix's unit and negatives tiers.
ACCEPTANCE_SUITES = ("VEGA_ACCEPTANCE_SUITES", "lane")
# X1's wrapper inside the acceptance scripts: `known_flake.py run --stage <s> -- <cmd…>` retries
# once on a flakes.json signature, announces it on stderr ("WARN: known flake <id> …") and
# appends one observed.json.flakes[] record per retry, as a JSON line, to $VEGA_FLAKE_RECORD.
KNOWN_FLAKE_WRAPPER = TESTS_DIR + "/known_flake.py"
FLAKE_RECORD_ENV = "VEGA_FLAKE_RECORD"
FLAKE_RECORD_KEYS = ("id", "stage", "command", "retried", "second_attempt")

# The skill directories whose code each stage runs. A stamp records their digests; a change
# makes the stamp stale, so `--from` cannot resume past it.
STAGE_CODE: Mapping[str, tuple[str, ...]] = {
    # the whole lanekit tree: a fixture runs its builder, composer, recipes, layers, stand-ins and
    # generators (T02's modules and every datasets/ file), so any of them changing stales it
    "fixture": ("dataset-shaper", "dataset-forge", "data-lens", LANEKIT_DIR),
    "forge": ("vega-forge", "dataset-shaper"),
    "train": ("vega-train", "vega-forge"),
    "eval": ("vega-eval", "vega-forge", "vega-train"),
    "pack": ("vega-pack", "vega-forge", "vega-eval", "dataset-shaper"),
    "bundle_api": (SERVING_TEMPLATES,),
    "report": (LANEKIT_DIR + "/report_stub.py", LANEKIT_DIR + "/stage_briefs.py"),
    "show": ("vega-show",),
    "serve": (SERVING_TEMPLATES,),
    "serve_check": (TESTS_DIR + "/serving_check.py", TESTS_DIR + "/check_kit.py"),
    # the check, its support modules (T08) and the one oracle module they re-export (R-H4)
    "inspect": (TESTS_DIR + "/inspect_check.py", TESTS_DIR + "/oracles.py",
                TESTS_DIR + "/check_kit.py", TESTS_DIR + "/inspect_recompute.py",
                TESTS_DIR + "/inspect_explain.py", TESTS_DIR + "/inspect_network.py",
                TESTS_DIR + "/inspect_context.py", TESTS_DIR + "/inspect_receptive.py",
                "vega-forge/scripts/tests/block_oracles.py"),
    # the stage's wrapper, the page and W15's driver (vega-show/scripts: smoke_page, its checks,
    # cdp, the qps reader and the parity server), the leak sets and the style lock (T04)
    "page": (TESTS_DIR + "/page_smoke.py", TESTS_DIR + "/page_smoke_checks.py", "vega-show/assets/page", "vega-show/scripts",
             TESTS_DIR + "/lint_templates.py", TESTS_DIR + "/foreign-literals.json",
             TESTS_DIR + "/style_lock.py", TESTS_DIR + "/style-lock.json", TESTS_DIR + "/style_lock_probe.js"),
    GOLDEN_STAGE: (TESTS_DIR + "/golden", "vega-pack", "vega-show"),
}

# The files a stage produces that later stages consume (relative to the lane's work dir, glob
# patterns). Their digests go into the stamp; `--from` refuses to resume when one changed.
STAGE_OUTPUTS: Mapping[str, tuple[str, ...]] = {
    "fixture": ("*.domain.analysis.shaped.html", "netspec.json", "shaped/manifest.json"),
    "forge": ("*.domain.analysis.shaped.vega.html", "vega-runs/*/smoke.json"),
    "train": ("*.domain.analysis.shaped.vega.train.html", "vega-runs/*/experiments.json"),
    "eval": ("*.domain.analysis.shaped.vega.train.eval.html", "vega-runs/*/test_seal.json",
             "vega-runs/*/eval/*/metrics.json", "vega-runs/*/eval/*/predictions.test.parquet"),
    "pack": ("*.vega.train.eval.pack.html", "bundle/*/MANIFEST.json"),
    "bundle_api": (),
    "report": ("../docs/label-roles.json", "../docs/main.aux", "../docs/main.pdf"),
    "show": ("*.pack.show.html", "bundle/*/showcase/SHOWCASE.json"),
    "serve": (), "serve_check": (), "inspect": (), "page": (),
    GOLDEN_STAGE: ("bundle/*/MANIFEST.json", "bundle/*/showcase/SHOWCASE.json"),
}

# --------------------------------------------------------------------------- data (R-A1, R-A4)
BATCH_KEYS = ("x_num", "x_cat", "y", "edge_index", "edge_weight", "target_index",
              "graph_index", "num_graphs")
# D6 wrote these names before D1 fixed the batch; they read as the D1 keys.
BATCH_KEY_ALIASES = {"graph_id": "graph_index", "target_mask": "target_index",
                     "n_graphs": "num_graphs"}
GRAPH_REFERENCE_FILES = {"nodes": "graph/reference-nodes.parquet",
                         "edges": "graph/reference-edges.parquet", "meta": "graph/graph.json"}
GRAPH_REFERENCE_FORBIDDEN_COLUMNS = ("pos", "row", "row_id", "raw_pos")   # no raw positions (R-A4)
SHAPED_EDGES_KEY = ("outputs", "files", "edges")                           # manifest path
SHAPED_GRAPH_KEY = ("graph",)                                             # manifest.graph
UNSCORED = "unscored.json"

# --------------------------------------------------------------------------- outputs (R-C1–C5)
LINKS = ("softmax", "sigmoid", "sigmoid_each", "identity", "cumulative", "reconstruction",
         "distance", "horizon")
READING_LINK = {"binary": "sigmoid", "multiclass": "softmax", "multilabel": "sigmoid_each",
                "ordinal": "cumulative", "regression": "identity", "quantile": "identity",
                "forecast": "horizon", "reconstruction": "reconstruction",
                "anomaly-reconstruction": "reconstruction", "anomaly-svdd": "distance"}
# MANIFEST.output.field_kinds kind -> the RowPrediction JSON key (D2 §2.3). `None` is CSV only.
FIELD_KIND_KEYS = {"class": "label", "flag": "labels", "alarm": "anomalous",
                   "probability": "probabilities", "raw": "logits", "value": "value",
                   "quantile": "quantiles", "horizon": "values", "score": "score",
                   "row_score": "reconstruction_error", "cumulative": "cumulative",
                   "recon": "reconstruction", "embedding": "embedding", "decision": None,
                   "group": "graph_id", "answered": "answered", "warnings": "warnings"}
# kinds answered only on ?include=<name>
FIELD_KIND_INCLUDE = {"raw": "logits", "recon": "reconstruction", "embedding": "embedding"}
# R-C4: the RowPrediction keys every answer of a reading carries (include-only keys excluded)
READING_KEYS = {"binary": ("label", "probabilities"), "multiclass": ("label", "probabilities"),
                "multilabel": ("labels", "probabilities"),
                "ordinal": ("label", "probabilities", "cumulative"), "regression": ("value",),
                "quantile": ("value", "quantiles"), "forecast": ("values",),
                "reconstruction": ("reconstruction_error",),
                "anomaly-reconstruction": ("score", "anomalous"),
                "anomaly-svdd": ("score", "anomalous")}
THRESHOLD_SCALES = ("probability", "score")


def OUTPUT_FIELDS(manifest: Mapping, include: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The JSON keys `/v1/predict` answers for this bundle, read through
    MANIFEST.output.field_kinds (R-C4/R-C5). `warnings` is always present. A v2.2 manifest
    without field_kinds answers its `output.fields` names."""
    output = manifest.get("output") or {}
    kinds = output.get("field_kinds")
    if not kinds:
        return tuple(dict.fromkeys(list(output.get("fields") or []) + ["warnings"]))
    keys = []
    for entry in kinds.values():
        kind = (entry or {}).get("kind")
        key = FIELD_KIND_KEYS.get(kind)
        if key is None or (kind in FIELD_KIND_INCLUDE and FIELD_KIND_INCLUDE[kind] not in include):
            continue
        keys.append(key)
    return tuple(dict.fromkeys(keys + ["warnings"]))


def output_record_problems(manifest: Mapping) -> list[str]:
    """What is wrong with MANIFEST.output as a field record: an unknown field kind, a field
    without a kind, or a compared/decided field outside `fields` (D2 §3.1)."""
    output = manifest.get("output") or {}
    fields = list(output.get("fields") or [])
    kinds = output.get("field_kinds")
    if not fields:
        return ["MANIFEST.output.fields is empty"]
    if kinds is None:
        return []                                           # v2.2 record: names only
    problems = [f"field {f!r} has no field_kinds entry" for f in fields if f not in kinds]
    problems += [f"field {f!r} has unknown kind {v.get('kind')!r}" for f, v in kinds.items()
                 if (v or {}).get("kind") not in FIELD_KIND_KEYS]
    for pair in list(output.get("compared") or []) + list(output.get("decided") or []):
        name = pair[0] if isinstance(pair, (list, tuple)) else pair
        if name not in fields:
            problems.append(f"compared/decided field {name!r} is not in fields")
    return problems


# --------------------------------------------------------------------------- serving (R-D1…R-D14)
SERVE_MODULE = "serve.app"
SERVE_ARGS = ("--port", "{port}", "--torch-threads", "{threads}")
HEALTH_ROUTE = "/health"
HEALTH_READY = "ready"
HEALTH_SHOWCASE = "showcase"                # /health.showcase == GET /v1/showcase (R-D4)
HEALTH_INSPECTION = "inspection"
SHOWCASE_ROUTE = "/v1/showcase"
STORY_ROUTE = "/v1/story"
CONSOLE_ROUTE = "/console"
NETWORK_ROUTE = "/v1/network"
EXPLAIN_ROUTE = "/v1/explain"
REQUIRED_ENDPOINTS = ("GET /v1/network", "POST /v1/explain")      # MANIFEST.serve.endpoints
SERVE_ENDPOINTS_KEY = ("serve", "endpoints")
SERVE_ENV_KEY = ("serve", "identity", "env")                        # the <ENV> prefix
SHOWCASE_SLOT_KEY = ("serve", "showcase_slot")                      # R-D6
NETWORK_PIN = ("serve", "inspection", "network_digest", "full")     # MANIFEST path (D3 §13.2)
NETWORK_DIGEST_FIELD = "network_digest"                             # in the /v1/network body
INSPECTION_OFF = {"pack_flag": "--no-inspect",
                  "manifest": ("serve", "inspection", "enabled"), "manifest_off": False,
                  "env_suffix": "_INSPECT", "env_off": "off", "status": 404, "needle": "inspection"}
# the bundle tests' skip reason when no inspection routes ship: allowed (lanes.json
# allowed_skips) only on a bundle whose MANIFEST says it was packed with --no-inspect
NO_INSPECT_SKIP = "packed with --no-inspect"
BUNDLE_FORBIDDEN_DIRS = ("ingestion", "training", "evaluation")
ROUNDTRIP_KEY = "roundtrip"

# --------------------------------------------------------------------------- the live checks (T08)
# What serving_check.py (D6 §7, SA1–SA14) and inspect_check.py (D6 §8, NI1–NI8, EI1–EI12) read
# from a bundle and ask its service. Routes and keys are the serving templates' (S02, S07,
# I08); the record files are pack's (S00/S01/S10); the result schemas are this program's.
SERVING_CHECK = "vega/serving-check@1"
INSPECT_CHECK = "vega/inspect-check@1"
SERVING_CHECK_IDS = tuple(f"SA{i}" for i in range(1, 15))
INSPECT_CHECK_IDS = tuple(f"NI{i}" for i in range(1, 9)) + tuple(f"EI{i}" for i in range(1, 13))
BUNDLE_MANIFEST = "MANIFEST.json"
BUNDLE_SCHEMA = "schema.json"
BUNDLE_NETSPEC = "netspec.json"
BUNDLE_WEIGHTS = "weights.pt"
WEIGHTS_STATE = "state_dict"                                   # weights.pt {state_dict, …}
ARCHITECTURE_RECORD = "records/architecture.json"              # its `smoke` (per_block, params_total)
ARCHITECTURE_SMOKE = ("smoke",)
# a lane's work dir holds <work>/bundle/<slug> and the campaign it trained; a selected run that
# is not the forged spec (a rung, a trial) smoked its own re-emitted package there
LANE_RUN_SMOKE = "vega-runs/{campaign}/runs/{run}/smoke.json"
EXAMPLE_ROWS = "examples/rows.csv"
EXAMPLE_REQUEST = "examples/request.json"
EXAMPLE_EXPECTED = "examples/expected.parquet"
EXAMPLE_ROW_KEY = "__row__"                                    # expected.parquet -> rows.csv position
ROUNDTRIP_TOLERANCE = ("roundtrip", "tolerance", "value")
PREDICT_ROUTE = "/v1/predict"
PREDICT_CSV_ROUTE = "/v1/predict/csv"
EXAMPLES_ROUTE = "/v1/examples"
INFO_ROUTE = "/v1/info"
NETWORK_TENSOR_ROUTE = "/v1/network/tensor/{id}"
CSV_ACCEPT = "text/csv"
# SA13: the versions every answer carries — JSON body keys, and the CSV answer's headers
VERSION_KEYS = ("api_version", "model_version", "contract_version", "feature_contract_version")
VERSION_HEADERS = {"api_version": "x-api-version", "model_version": "x-model-version",
                   "contract_version": "x-contract-version",
                   "feature_contract_version": "x-feature-contract-version"}
ANSWER_TAIL = ("answered", "warnings")                         # every RowPrediction carries them
TRUTH_PREFIX = "true_"                                         # the CSV answer echoes the truth
WARNING_SEP = "; "                                             # a CSV answer joins a row's warnings
HISTORY_NEEDLE = "history"                                     # SA12: the unanswered window's warning
# the reading -> the per-task invariant of D6 §7 (R-C3/R-C4 rewrite); a sequence task reads as
# its base (MANIFEST.output.reading is already the base's)
READING_CHECK = {"binary": "SA4", "multiclass": "SA5", "multilabel": "SA6", "regression": "SA7",
                 "quantile": "SA7", "ordinal": "SA8", "forecast": "SA9", "reconstruction": "SA10",
                 "anomaly-reconstruction": "SA11", "anomaly-svdd": "SA11"}
# D6 §8: what inspect_check explains — every preset, the first demo items, the first shipped
# example rows, at most this many samples per lane, each at detail exact
EXPLAIN_DETAIL = "exact"
EXPLAIN_DEMO_ROWS = 5
EXPLAIN_SHIPPED_ROWS = 5
EXPLAIN_MAX_SAMPLES = 20
NODE_ID = r"^[A-Za-z0-9_.@:-]{1,128}$"                          # inspect-contract §5.2 (R-F8)
GRAPH_REFERENCE_META_KEYS = {"nodes": "n_nodes", "method": "method", "params": "params"}
GRAPH_REFERENCE_EDGE_COLUMNS = ("src_ref", "dst_ref", "weight")
RAW_PREFIX = "raw:"                                            # reference-node raw columns (B05)

# --------------------------------------------------------------------------- showcase (R-E1…R-E11)
SHOWCASE_DIR = "showcase"
# the stage's own siblings of showcase/ (show-contract §5.1): the previous showcase one rebuild
# keeps and the transient build directory — never bundle files, never served, so the bundle pin
# leaves them out with showcase/ itself (a lane whose show stage re-runs keeps its pin)
SHOWCASE_SIBLINGS = ("showcase.prev", "showcase.building")
SHOWCASE_MANIFEST = "SHOWCASE.json"
SHOWCASE_SCHEMA = "vega/showcase@1"
SHOWCASE_BUNDLE_DIGEST = ("bundle", "manifest_digest")
SHOWCASE_FILES = "files"
SHOWCASE_SERVED = "served"
SHOWCASE_UNPINNED = (".work/", ".smoke/", "docs/")        # scratch the closed world ignores (R-D5)
# How GET / identifies the showcase page: its bytes are the entry file SHOWCASE.json names.
SHOWCASE_MARKER = "entry"
SHOWCASE_STORY = ("story", "file")


def showcase_entry(showcase_dir: Path) -> Path:
    """The file GET / must serve byte for byte when the showcase is verified."""
    doc = json.loads((showcase_dir / SHOWCASE_MANIFEST).read_text(encoding="utf-8"))
    return showcase_dir / doc[SHOWCASE_MARKER]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------- vega-show
# D5 §7 CLIs, with R-E2 (TEMPLATE.json replaces page-manifest.json) and R-E7 (label roles).
# Placeholders are filled by stages.run_show; a CLI change upstream is an edit here.
# the steps before the build keep their markers where the build keeps them (SKILL: <s> is
# <b>/showcase.building), so a carry's dropped slot, an unearned verdict or a fact the extractor
# warned about reaches showcase/.work/markers.json and the layer (gate C: they were only printed)
SHOW_MARKERS = "{bundle}/showcase.building/.work/markers.json"
SHOW_CLI = {
    "facts": (SHOW_SCRIPTS + "/extract_facts.py", "--model", "{model}", "--bundle", "{bundle}",
              "{report_args}", "--out", "{plan_dir}/facts.json", "--markers", SHOW_MARKERS),
    "draft": (SHOW_SCRIPTS + "/draft_plan.py", "--facts", "{plan_dir}/facts.json",
              "--page-manifest", "{template}", "--memory", "{memory}", "--dataset", "{dataset}",
              "--out", "{plan_dir}/story-plan.draft.json", "--markers", SHOW_MARKERS),
    "resolve": (SHOW_SCRIPTS + "/resolve_story.py", "--plan", "{plan_dir}/story-plan.json",
                "--facts", "{plan_dir}/facts.json", "--page-manifest", "{template}",
                "{report_args}", "--out", "{plan_dir}/story.json", "--markers", SHOW_MARKERS),
    "build": (SHOW_SCRIPTS + "/build_showcase.py", "--bundle", "{bundle}", "--model", "{model}",
              "--plan", "{plan_dir}/story-plan.json", "--story", "{plan_dir}/story.json",
              "--facts", "{plan_dir}/facts.json", "{report_args}", "--out", "{showcase}",
              "--unattended", "--locale", "{locale}"),
    "check": (SHOW_SCRIPTS + "/build_showcase.py", "--check", "{showcase}", "{report_args}"),
    "layer": (SHOW_SCRIPTS + "/apply_show_layer.py", "{model}", "--data",
              "{showcase}/.work/showcase-layer.json", "--out", "{out}"),
    "smoke": (SHOW_SCRIPTS + "/smoke_show.py", "{out}", "--strict"),
}
SHOW_TEMPLATE = "vega-show/assets/page/TEMPLATE.json"
# H07: the layer's authoring file the lane writes in the model's place (SKILL step 8): a fixed,
# digit-free abstract, and the run's WARN lines verbatim. The served walk is the lane's page
# stage, so at the layer it has not run and the not-checked marker says so (gate LY2).
SHOW_LAYER_DATA = "{showcase}/.work/showcase-layer.json"
SHOW_LAYER_ABSTRACT = ("The page shows a reader outside the chain what this lane's network reads, "
                       "how the chain chose it, and how it did on the sealed test part.")
SHOW_LAYER_NOT_CHECKED = "WARN: served page not checked — the lane walks the served page in its page stage"
# R-E10: the one digit rule (H01): check_prose(text, names=names) -> [problem] (keyword: the
# second positional parameter is `known`, the fact keys), placeholders removed
# first; `names` is facts.json `names`, the whitelist of tokens that carry digits legitimately
DIGIT_RULE_API = (SHOW_SCRIPTS + "/digit_rule.py", "check_prose")
FACTS_NAMES = "names"
# A44(c): the role map is the one in the report's directory (extract_facts, resolve_story and
# build_showcase all read docs/label-roles.json beside --report); resolve_story.py has no
# --label-roles, so the shared report arguments name the report directory only
SHOW_REPORT_ARGS = ("--report", "{docs}")
SHOW_PSEUDO_DIR = ".work/qps"                               # qps-ploc builds (R-F1), never served
SHOW_LAYER_OUT = "{stem}.show.html"

# R-E4 motifs, keyed by modality (the vega-show catalog holds the machine copy)
MOTIFS = ("grid", "points", "traces", "lanes", "links", "contours")
MOTIFS_BY_MODALITY = {
    "row": {"base": "points", "admissible": ("points", "lanes", "grid")},
    "spatial-row": {"base": "contours", "admissible": ("contours", "points", "lanes", "grid")},
    "window": {"base": "traces", "admissible": ("traces", "lanes", "grid")},
    "graph": {"base": "links", "admissible": ("links", "grid")}}
ROLE_VALUES = ("neutral", "alert", "plain")                 # R-E5

# R-E6 lexicon@1: closed keys, required per shape (and graph level)
LEXICON_KEYS = ("row", "feature", "source", "step", "window", "node", "edge", "graph", "target")
LEXICON_REQUIRED = {"row": ("row", "feature", "source"),
                    "window": ("row", "feature", "source", "step", "window"),
                    "graph-node": ("row", "feature", "source", "node", "edge"),
                    "graph-graph": ("row", "feature", "source", "node", "edge", "graph")}
LEXICON_WORD = r"^[a-z][a-z' -]{0,30}$"
LEXICON_ENTRY = ("one", "many", "article")                  # every key, source included
LEXICON_ARTICLES = {"default": ("a", "an"), "source": ("a", "an", "the")}


def lexicon_modality(shape: str, level: str | None) -> str:
    """The LEXICON_REQUIRED key of a lane: its shape, and for graphs its level."""
    return f"graph-{level or 'node'}" if shape == "graph" else shape


SKILLS_ROOT = Path(__file__).resolve().parents[4]           # the tree this lanekit sits in


def REPORT_ROLES(skills: Path | None = None) -> tuple[str, ...]:
    """The report roles a stub report binds (R-E7): the vega-show catalog's `report_roles`, read
    from `skills` (default: the tree this module belongs to)."""
    catalog = json.loads(((skills or SKILLS_ROOT) / SHOW_CATALOG).read_text(encoding="utf-8"))
    roles = catalog["report_roles"]
    return tuple(roles if isinstance(roles, list) else roles.keys())


# R-F4 view ids; `tried` and `weights` are sub-views of `network`; `#window` aliases `many`
VIEWS = ("start", "context", "network", "try", "many", "results", "code")
SUB_VIEWS = {"network": ("tried", "weights")}
VIEW_ALIASES = {"window": "many"}


def expected_views(runs: int) -> tuple[str, ...]:
    """Every view on every lane; the network's sub-views when the campaign ran ≥ 2 runs (§4.7)."""
    views = list(VIEWS)
    if runs >= 2:
        at = views.index("network") + 1
        views[at:at] = list(SUB_VIEWS["network"])
    return tuple(views)


# R-F1 probe
PROBE_HOOK = "window.__showcase"
PROBE_QUERY = "probe"
PROBE_RESULT_ID = "showcase-probe"
PSEUDO_LOCALE = "qps-ploc"

# --------------------------------------------------------------------------- checks and gates
CHECK_CLI = {
    "serve_check": (TESTS_DIR + "/serving_check.py", "--base-url", "{url}", "--bundle", "{bundle}",
                    "--lane", "{lane}", "--lanes-file", "{lanes_file}", "--json", "{out}"),
    "inspect": (TESTS_DIR + "/inspect_check.py", "--base-url", "{url}", "--bundle", "{bundle}",
                "--lane", "{lane}", "--lanes-file", "{lanes_file}", "--json", "{out}"),
    "page": (TESTS_DIR + "/page_smoke.py", "--base-url", "{url}", "--bundle", "{bundle}",
             "--lane", "{lane}", "--lanes-file", "{lanes_file}", "--out", "{out}",
             "--shots", "{shots}"),
}
# G00: re-pack R7 into <work>/bundle/<slug> with this release's pack.py and build its showcase
GOLDEN_CLI = (TESTS_DIR + "/golden/cic.py", "--cids-root", "{cids_root}", "--lane-dir", "{lane_dir}",
              "--work", "{work}", "--skills", "{skills}", "--json", "{out}")
CONFORMANCE_IDS = tuple(f"C{i}" for i in range(15))
CONFORMANCE_MEASURES = ("vega-forge/scripts/conformance.py", "measure_module")   # R-G1
SHAPER_GRAPH_GATES = tuple(f"SG{i}" for i in range(1, 8))           # SG8 moved to pack as N-S9
SLOT_GATES = tuple(f"SS{i}" for i in range(1, 10)) + tuple(f"IN{i}" for i in range(1, 8))
# D3's own negative ids for the same slot cases (BUILD-PLAN T10: "D3's NS1/NS2 are SS3/SS2")
SLOT_GATE_ALIASES = {"NS1": "SS3", "NS2": "SS2"}
PACK_GRAPH_GATES = ("N-S3", "N-S9")                                 # transductive; tampered reference
PACK_REFUSAL_TRANSDUCTIVE = "inductive graphs only"                 # S04's refusal needle

# --------------------------------------------------------------------------- lanekit modules
FIXTURE_API = ("lanekit.fixture", "make_lane")            # (lane, work, skills, composer=) -> LaneFixture
AUTHORING_API = ("lanekit.authoring", ("forge_authoring", "train_authoring",
                                       "pack_authoring", "show_authoring"))
REPORT_STUB_API = ("lanekit.report_stub", "write")        # (docs_dir, roles) -> dict
GENERATORS_API = ("lanekit.datasets", "GENERATORS")
ORACLES_API = ("oracles", "shape_rule")                   # T08 re-export of block_oracles.py:
                                                          # (type, params, per-sample in_shape, lane)
EXTERNAL_GENERATORS = {"orders": "vega-forge/scripts/tests/make_fixture.py",
                       "cic": TESTS_DIR + "/golden/cic.py"}

# domain-forge files whose digests the matrix pins (D6 §2.3); domain-forge is external
DOMAIN_FORGE_PINNED = ("scripts/compose_model.py", "scripts/apply_layer.py",
                       "scripts/validate_model.py", "assets/template.html")
