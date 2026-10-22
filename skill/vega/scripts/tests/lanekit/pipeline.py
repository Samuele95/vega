"""
The blueprint's world model a fixture carries: `blueprint-runs/<ts>/pipeline.json` and the
`blueprint.md` baseline `/architect --baseline` reads (D6 §5.1, BUILD-PLAN R-G2).

    world = WorldModel(name="landcover-cover-pipeline", package="landcover_pipeline",
                       manifest_digest=digest, serving_mode="online", input_dim=16,
                       label="cover", partition_source="[geometry:partitions/chosen]",
                       monitoring=(("b04 drift", "[analysis:F4]"),))
    pipeline, text = world.pipeline(), world.blueprint_md()

One world model for every fixture: the orders fixture (`make_fixture.py`) and every lane
(lanekit.fixture) render it with their own names, so the ports, their `signature[]` and the
module map are written once. Every port-owning module is `OOP` (the adapter is an object
guarding a state — the seal, the fitted parameters, the device); only `domain`, which owns
none, is `FP` (code-contract §2, C14). Each port carries its own interface within the size
discipline: at most 5 parameters beyond `self` per method (C12). They are the v2.3 port
interfaces in full: DataSource D1 §9.1, Trainer §10.1, Evaluator §11.1, Model §4.1/§7.2,
Predictor D2 §4.6, Monitor D2 §7.1; the Registry keeps today's two methods. The Model port
also carries the serving seam — the raw head of an assembled batch and the network's
`anatomy()`, all the Predictor and the inspection read of it (code-contract §1). The v2.3 domain
types appear in annotations only, which ports.py never evaluates; a default that names one is
written `...` (the stub convention), because a default is evaluated when the Protocol class is
defined.

Standard library only: `make_fixture.py` imports this before anything heavy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["WorldModel", "PORTS", "MODULES", "ARTIFACTS", "EXTERNAL_ARTIFACTS", "ANCHORS",
           "port_signatures", "FILLED_BY"]

# (name, role, responsibility, source, signature) — the Model port's source is the partition
# and the Monitor's the first monitored finding, both filled in per fixture (None here)
PORTS: tuple[tuple[str, str, str, str | None, tuple[str, ...]], ...] = (
    ("DataSource", "DataSource",
     "yield the parts by name; the test part only through the seal", "[shape:phase/split]",
     ("def load_part(self, spec: NetSpec, part: Part) -> SamplePart: ...",
      "def iter_batches(self, spec: NetSpec, part: Part, batch_size: int | None = "
      "None, columns=None, plan: BatchPlan | None = None) -> Iterator[dict]: ...",
      "def batch_count(self, spec: NetSpec, part: Part, batch_size: int | None = None, "
      "plan: BatchPlan | None = None) -> int: ...",
      "def load_resident(self, spec: NetSpec, part: Part, device) -> SamplePart: "
      "...",
      "def make_loader(self, dataset: SamplePart, spec: NetSpec, shuffle: bool = "
      "False, seed: int | None = None, batch_size: int | None = None) -> "
      "Iterator[dict]: ...",
      "def load_sealed_part(self, spec: NetSpec, token: str, seal_path: str = "
      "\"test_seal.json\") -> SamplePart: ...",
      "def design_matrix(self, spec: NetSpec, dataset: SamplePart, view: DesignView "
      "= ...) -> DesignMatrix: ...",
      "def exposure(self, spec: NetSpec) -> GraphExposure | None: ...",
      "def target_scale(self, spec: NetSpec, season: int = 1) -> float | None: "
      "...",
      "def train_targets(self, spec: NetSpec) -> Any: ...")),
    ("Recipe", "FeatureTransformer",
     "raw -> shaped with the fitted recipe; no fitting at inference", "[shape:manifest]",
     ("def transform(self, df_raw, history=None, with_unseen: bool = False) -> Any: "
      "...",)),
    ("Model", "Model", "features -> outputs; parameters as one value", None,
     ("def score(self, batch: dict) -> Any: ...",
      "def predict_proba(self, batch: dict) -> Any: ...",
      "def param_count(self) -> tuple[int, int]: ...",
      "def block_shapes(self, batch: dict) -> list[dict]: ...",
      "def head(self, batch: dict) -> Any: ...",
      "def anatomy(self) -> NetworkAnatomy | None: ...")),
    ("Trainer", "Trainer",
     "fit a Model on a DataSource under a spec; emit an epoch record", "[analysis:importance]",
     ("def fit(self, model: TrainedNetwork, spec: NetSpec, data: TrainingData, "
      "callbacks: Callbacks | None = None, run: RunContext | None = None) -> FitResult: ...",
      "def evaluate(self, model: TrainedNetwork, spec: NetSpec, loader, loss_fn=None) "
      "-> dict: ...",
      "def predict(self, model: TrainedNetwork, spec: NetSpec, part: SamplePart) "
      "-> ScoredPart: ...")),
    ("Evaluator", "Evaluator",
     "one sealed look; metrics with intervals; baselines under one protocol",
     "[analysis:modules/drift]",
     ("def scored(self, spec: NetSpec, dataset: SamplePart, model: ScoredModel) "
      "-> ScoredPart: ...",
      "def score_part(self, spec: NetSpec, part: Part, checkpoint=None, "
      "model: ScoredModel | None = None) -> Metrics: ...",
      "def score_sealed_test(self, spec: NetSpec, token: str, checkpoint=None, "
      "model: ScoredModel | None = None, seal_path: str = \"test_seal.json\") -> Metrics: ...",
      "def measure(self, spec: NetSpec, part: Part, scored: ScoredPart) -> Metrics: ...",
      "def heads(self, spec: NetSpec, dataset: SamplePart, model: ScoredModel) -> Any: ...",
      "def metric_names(self, spec: NetSpec) -> list: ...",
      "def with_options(self, options: MetricOptions) -> Any: ...")),
    ("Registry", "ModelStore",
     "checkpoints and specs as versioned artifacts, by digest",
     "[DESIGNER'S CHOICE — reproducibility]",
     ("def save(self, model: TrainedNetwork, path, spec: NetSpec, *, "
      "epoch: int | None = None) -> Checkpoint: ...",
      "def load(self, ref: Checkpoint | str, spec: NetSpec, build: ModelBuilder, *, "
      "expected_digest: str | None = None, run_dir=None) -> TrainedNetwork: ...",
      "def version(self, path) -> Checkpoint: ...")),
    ("Predictor", "Predictor",
     "raw rows -> predictions through the Recipe and the Model", "[DESIGNER'S CHOICE — serving]",
     ("def predict(self, df_raw, history=None) -> Any: ...",
      "def predictions(self, df_raw, history=None) -> tuple[Prediction, ...]: ...",
      "def predict_one(self, row: dict, history=None) -> dict: ...")),
    ("Monitor", "Monitor", "watch what the analysis found fragile", None,
     ("def check(self, batch, predictions=None) -> tuple[DriftReading, ...]: ...",
      "def report(self, batch, predictions=None) -> tuple[DriftReading, ...]: ...")),
)

# (name, paradigm, ports owned, depends on): the arrows the package's code takes (vega-forge
# `emit.module_arrows`, what `port_surface.py` names): the Trainer, the Evaluator and the
# Predictor read the Model module's readout and metrics; a graph Trainer also reads the
# DataSource module's graph views (GRAPH_ARROWS) — C4 refuses an arrow no import takes
MODULES: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("domain", "FP", (), ()),
    ("ingestion", "OOP", ("DataSource",), ("domain",)),
    ("features", "OOP", ("Recipe",), ("domain",)),
    ("model", "OOP", ("Model", "Registry"), ("domain",)),
    ("training", "OOP", ("Trainer",), ("domain", "model")),
    ("evaluation", "OOP", ("Evaluator",), ("domain", "model")),
    ("serving", "OOP", ("Predictor", "Monitor"), ("domain", "model")),
)
GRAPH_ARROWS = {"training": ("ingestion",)}
# the ports whose adapter a later stage fills, and the stage (vega-forge emit.ROLE_FILES): the
# blueprint says so on the port (`filled_by`), so §3 states as pending what the map records
FILLED_BY = {"FeatureTransformer": "/vega-pack", "Predictor": "/vega-pack", "Monitor": "/vega-pack"}

ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("NetSpec", "value", "digest"), ("FeatureContract", "value", "manifest digest"),
    ("Checkpoint", "value", "digest"), ("Metrics", "value", "run + seal exposure"),
    ("Prediction", "value", "none"), ("Campaign", "entity", "run directory"))

# The artifacts held outside the emitted package (code-contract §3): name -> (owner, where).
# The campaign is /vega-train's directory, so domain.py does not define it.
EXTERNAL_ARTIFACTS: dict[str, tuple[str, str]] = {
    "Campaign": ("/vega-train", "the campaign directory (campaign.json, runs/<id>/)")}

ANCHORS = ("EJ-17", "EJ-20", "EJ-2", "EJ-1", "EJ-10", "EJ-15", "EJ-18", "EJ-34", "EJ-64")

# the DataSource methods only some packages carry (vega-forge emit.SCOPED_METHODS): a graph's
# exposure, a window's seasonal scale, a regression-family head's train targets — a port
# declares them where its adapter has them, and nowhere else (SOLID-I)
SCOPED = {"exposure": (("graph",), None), "target_scale": (("window",), None),
          "train_targets": (None, ("regression", "forecast"))}


def port_signatures(shape: str | None = None, task: str | None = None) -> dict[str, list[str]]:
    """{role: signature[]} of the v2.3 ports for a package of `shape` and `task` (None: any)."""
    return {role: [m for m in sig if role != "DataSource" or _carried(m, shape, task)]
            for _name, role, _resp, _src, sig in PORTS}


def _carried(line: str, shape: str | None, task: str | None) -> bool:
    name = line.split("(", 1)[0].removeprefix("def ").strip()
    shapes, tasks = SCOPED.get(name, (None, None))
    return (shape is None or shapes is None or shape in shapes) and \
        (task is None or tasks is None or task in tasks)


def _artifact(name: str, kind: str, identity: str) -> dict[str, Any]:
    """One `artifacts[]` entry; an external one says which stage holds it and where."""
    entry: dict[str, Any] = {"name": name, "kind": kind, "identity": identity}
    if name in EXTERNAL_ARTIFACTS:
        owner, where = EXTERNAL_ARTIFACTS[name]
        entry["external"] = {"owner": owner, "where": where}
    return entry


@dataclass(frozen=True)
class WorldModel:
    """What varies between fixtures' world models: the system's name and package, the shaped
    manifest it was drawn against, the serving mode, the feature contract, the partition the
    Model port answers to, and what the Monitor watches (name, source) — the first entry is
    the Monitor port's source."""
    name: str
    package: str
    manifest_digest: str
    serving_mode: str
    input_dim: int
    label: str | None
    partition_source: str
    monitoring: tuple[tuple[str, str], ...]
    shape: str | None = None            # the sample shape the package carries (None: any)
    task: str | None = None             # the effective task kind (None: any)

    def __post_init__(self) -> None:
        if not self.monitoring:
            raise ValueError("a world model watches at least one thing (monitoring)")
        if self.serving_mode not in ("batch", "online", "stream"):
            raise ValueError(f"serving mode {self.serving_mode!r} is not batch, online or stream")

    def pipeline(self) -> dict[str, Any]:
        """pipeline.json: the blueprint's world model in machine form (code-contract §1)."""
        sources = {"Model": self.partition_source, "Monitor": self.monitoring[0][1]}
        return {
            "system": {"name": self.name, "package": self.package, "source": "[spec §1]"},
            "input": {"shape_manifest_digest": self.manifest_digest},
            "posture": {"family": "neural", "source": "[analysis:importance]"},
            "serving": {"mode": self.serving_mode},
            "training": {"protocol": "fixed split (the shaper's)"},
            "monitoring": [{"name": n, "source": s} for n, s in self.monitoring],
            "feature_contract": {"input_dim": self.input_dim, "label": self.label},
            "boundaries": [
                {"name": "raw", "contract": "the input file's columns",
                 "source": "[geometry:typing]"},
                {"name": "shaped", "contract": "shape.manifest.output_schema",
                 "source": "[shape:manifest]"},
                {"name": "features", "contract": "the network's data_contract",
                 "source": "[shape:manifest]"},
                {"name": "predictions", "contract": "score, prediction, warnings",
                 "source": "[DESIGNER'S CHOICE — serving]"}],
            "ports": [{"name": name, "role": role, "responsibility": resp,
                       **({"filled_by": FILLED_BY[role]} if role in FILLED_BY else {}),
                       "source": source if source is not None else sources[name],
                       "signature": [m for m in signature if role != "DataSource"
                                     or _carried(m, self.shape, self.task)]}
                      for name, role, resp, source, signature in PORTS],
            "modules": [{"name": name, "paradigm": paradigm, "ports_owned": list(owned),
                         "depends_on": list(deps) + list(GRAPH_ARROWS.get(name, ())
                                                         if self.shape == "graph" else ())}
                        for name, paradigm, owned, deps in MODULES],
            "artifacts": [_artifact(n, k, i) for n, k, i in ARTIFACTS],
            "anchors": list(ANCHORS),
        }

    def blueprint_md(self) -> str:
        """blueprint.md: the baseline /architect --baseline reads."""
        return (f"# Blueprint — {self.name}\n\n## §0 Style\nHexagonal, ports and adapters.\n\n"
                "## §2 World model\nThe artifacts are values; the campaign is the one entity.\n\n"
                "## §3 Ports\nDataSource, Recipe, Model, Trainer, Evaluator, Registry, Predictor, "
                "Monitor.\n\n## §4 Module map\ndomain <- ingestion, features, model; training, "
                "evaluation and serving <- domain, model"
                + ("; training <- ingestion (the graph views)" if self.shape == "graph" else "")
                + ".\n")
