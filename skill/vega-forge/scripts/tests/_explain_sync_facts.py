"""
_explain_sync_facts.py — what the design fixes and `test_explain_catalog_sync.py` holds the
explain catalog to: the page's stage kinds (D4 §7.2), the non-computing kinds and the block
explainers (D4 §8.1), the fixed walk and R-E8's anchors, the package paths each walk step must
find, the silencing activations (D4 §8.2), R-B8's attention pool, the regular-expression
constructs JavaScript does not read, and the damaged catalogs the schema must refuse.

The leading underscore keeps the unit tier's `test_*.py` discovery from collecting it.
"""
from __future__ import annotations

import re

DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "vega/explain-catalog@1"

#: D4 §7.2: block type -> the page's stage kind (net/kinds.js KIND_OF); `input` is the 27th kind
KIND_OF = {
    "dense": "dense", "embedding": "embedding", "concat": "concat", "batchnorm": "norm",
    "layernorm": "norm", "dropout": "dropout", "activation": "activation",
    "residual": "residual", "sequential": "sequential", "parallel": "parallel",
    "conv1d": "conv", "pool1d": "pool1d", "rnn": "recurrent", "gru": "recurrent",
    "lstm": "recurrent", "attention": "attention", "attention_pool": "attention_pool",
    "transformer_encoder": "transformer", "feature_tokenizer": "tokenizer",
    "positional": "positional", "flatten": "reshape", "reshape": "reshape", "pool": "pool",
    "head": "head", "decoder": "decoder", "reparameterize": "latent",
    "gcn_conv": "graph_conv", "sage_conv": "graph_conv", "gat_conv": "graph_conv",
    "gin_conv": "graph_conv", "graph_pool": "graph_pool", "edge_dropout": "edge_dropout",
    "custom": "custom",
}
RENDERER_KINDS = 27

#: D4 §8.1: the stage kinds that are a one-line "then …" note at answer time, not a step
NON_COMPUTING = ("dropout", "edge_dropout", "reshape", "concat", "positional")
#: kinds that hold other blocks or user code: their own step recomputes nothing
NO_RECOMPUTE = NON_COMPUTING + ("sequential", "custom")
#: D4 §8.1: the block explainers of js/how/explainers/ (input and link explain no block)
EXPLAINERS = ("dense", "embedding", "norm", "residual", "conv", "recurrent", "attention",
              "graph", "latent", "pool", "generic")

#: D4 §14.2 and R-E8: the fixed walk steps, in the page's order
WALK_ORDER = ("root", "store", "build", "forward", "link", "shape", "run", "explain", "route")
#: R-E8: the anchors fixed verbatim for the v2.3 layout
RE8_AT = {"build": "^build_model = ", "store": r"^\s+def load\(", "route": r"^def register\("}
#: the package paths each fixed step's file pattern must find, and ones it must not
WALK_PATHS = {
    "root": (["predict.py"], ["pkg/serving/predict.py", "serve/predict.py"]),
    "store": (["pkg/model/store.py"], ["pkg/model/network.py"]),
    "build": (["pkg/model/network.py"], ["pkg/model/graph_network.py", "pkg/model/builder.py"]),
    "forward": (["pkg/model/network.py"], ["pkg/model/blocks_core.py"]),
    "link": (["pkg/model/network.py"], ["pkg/model/readout.py"]),
    "shape": (["pkg/features/preprocess.py"], ["pkg/features/transform.py"]),
    "run": (["pkg/serving/predict.py"], ["predict.py"]),
    "explain": (["pkg/serving/inspection/inspector.py"], ["pkg/serving/predict.py"]),
    "route": (["serve/routes_predict.py"], ["serve/routes_monitor.py", "serve/app.py"]),
}
#: R-B9: where a custom block's module is copied, and the text its `build` anchor must find
CUSTOM_PATH = "pkg/model/custom/my_block.py"
CUSTOM_TEXT = "import torch\n\n\ndef build(in_shape, params):\n"
CUSTOM_BUILD_LINE = 3

#: D4 §8.2: the activations whose negative side is (almost) silenced — the "quiet units" note
SILENCING = {"relu", "gelu", "silu", "leaky_relu"}
#: R-B8: attention_pool is explained from its score layer and its query
ATTENTION_POOL_READS = ["score", "query"]

#: constructs Python accepts and JavaScript's RegExp does not (the page compiles the anchors)
PY_ONLY = (r"(?P<", r"(?P=", r"(?#", r"\A", r"\Z", r"(?>", r"*+", r"++", r"?+", r"}+")
INLINE_FLAGS = re.compile(r"\(\?[aiLmsux]")


def js_unreadable(pattern):
    """The constructs of `pattern` JavaScript's RegExp does not read (empty when none)."""
    bad = [tok for tok in PY_ONLY if tok in pattern]
    if INLINE_FLAGS.search(pattern):
        bad.append("inline flags")
    return bad


def damages(explain):
    """(what, damage) pairs: each damage breaks a copy of the catalog `explain` in a way the
    schema must refuse."""
    def drop(path, key):
        def go(cat):
            node = cat
            for p in path:
                node = node[p]
            node.pop(key, None)
        return go

    def put(path, key, value):
        def go(cat):
            node = cat
            for p in path:
                node = node[p]
            node[key] = value
        return go

    def swap_walk(cat):
        steps = cat["walk_fixed"]
        steps[0], steps[1] = steps[1], steps[0]

    return (
        ("a block without its formula", drop(["blocks", "dense"], "formula")),
        ("a stage kind outside the page's", put(["blocks", "dense"], "stage_kind", "tokens")),
        ("a code anchor without `after`", drop(["blocks", "dense", "code"], "after")),
        ("an empty `at` list", put(["blocks", "dense", "code"], "at", [])),
        ("the fixed walk out of order", swap_walk),
        ("a tenth fixed walk step", lambda cat: cat["walk_fixed"].append(cat["walk_fixed"][0])),
        ("a link outside R-C1", put(["links"], "sigmoid-per-label", explain["links"]["sigmoid"])),
        ("a missing link", drop(["links"], "horizon")),
        ("an explainer that explains no block", put(["recompute", "none"], "explainer", "input")),
        ("an unknown recompute method", put(["blocks", "dense"], "recompute", "matmul")),
        ("an unknown activation", put(["activations"], "elu", explain["activations"]["relu"])),
        ("an unknown top-level key", put([], "blocks_extra", {})),
    )
