---
name: blueprint
description: "First-architecture proposal pass for a greenfield project — ingest a spec-analysis HTML or a domain-forge model, derive a proposed architectural style and world model plus ranked design findings, every claim traced to the spec or stamped as a designer's choice. With a DATASET model (/dataset-forge, /data-lens, /dataset-shaper HTML) it proposes the data / ML pipeline the geometry, analysis and recipe imply (--mode pipeline). Trigger when the user runs /blueprint, asks to propose a first architecture from a spec, choose between monolithic / microservices / layered / event-driven / distributed styles for a new system, design something not yet built, turn a spec, domain model or dataset model into a buildable blueprint, or scaffold the module / boundary / paradigm shape of a new system before code exists. Anchored to Bloch's Effective Java plus SOLID, GoF, FP, DDD and Hexagonal Architecture. Sibling to /architect, which reshapes an EXISTING codebase; /blueprint proposes the FIRST one."
---

# /blueprint

A first-architecture proposal pass for a system that does **not yet have code**. Same three axes as `/architect` — **Model / Abstraction / Composition** — anchored to *Effective Java* (Bloch, 3rd ed.) supplemented by SOLID, GoF, FP, DDD, and Hexagonal Architecture. Where `/architect` audits what the code *is*, `/blueprint` proposes what the code *should be*, derived from a prose spec or a domain model.

## Anchor corpus (single source of truth)

The EJ anchor set is **deliberately scoped** to the same 43 items as `/architect` — the items whose frontmatter `sets:` contains `architect` in the `effective-java` skill's corpus (`items/EJ-N.md`; verify: `grep -lE 'sets:.*architect' items/EJ-*.md` → 43). `blueprint` owns no corpus of its own — it inherits the architect set. That `effective-java` directory is the single source of truth (full per-item guidance + verbatim titles + the per-item "anchor must justify" clause); do **not** maintain a second copy here (drift between two copies is a defect). When you need the corpus for a breadth read, dispatch a sub-agent that loads the relevant `items/EJ-N.md` files — do not inline them into your own context. **Do not anchor a finding to an EJ item outside that 43-item set**; if no listed item fits, anchor to SOLID / GoF / FP / DDD / HEX / CQRS / MEASURED-PERF and say so in the `rationale`. Source PDF: `~/Documenti/Effective Java (2017, Addison-Wesley).pdf`.

The procedure is human-in-the-loop and multi-turn: **ingest → derive the architectural style → derive the world model → gate the structural bets → emit architect-compatible artifacts.** Big structural decisions (the architectural style, the decomposition, the paradigm per subsystem, the bounded contexts) are surfaced one per turn for your approval. Tactical calls — naming, field types, which GoF pattern inside a settled boundary — the skill makes itself.

Because there is no code to verify against, the skill's discipline is the **trace-to-spec invariant**: every design claim either cites the spec/model section it derives from, or is stamped `[DESIGNER'S CHOICE — pressure]` with the design pressure that justifies it. A claim that is neither is a hallucination and must not ship.

## Relationship to the family

- `/spec-analysis` produces the prose spec (`spec-analysis.html`) — a common **input** to this skill (for a `data-project`, the spec of the code around the data).
- `/domain-forge` produces the formal domain model (`model.html`) — the other accepted input.
- `/dataset-forge`, `/data-lens` and `/dataset-shaper` produce a **dataset model** (`ex:sourceKind "dataset"` with `geometry`, `analysis`, `shape` layers) — the third accepted input, handled in `--mode pipeline` (see *Dataset models* below).
- `/blueprint` (this skill) turns any of them into a **proposed** architecture: `blueprint.md` + `findings.json` + a self-contained `blueprint.html`.
- `/architect` later reshapes the **built** code — and can diff it against this skill's `blueprint.md` as the intended baseline.
- `/vega` (the orchestrator of the Vega package, formerly `/noctua`) schedules this skill as the `blueprint` stage of every lane, passes `--mode`, and checks — like every package stage — that `blueprint.html` was published in the conversation and that the run left its stage brief (`docs/blueprint-brief.md`, per `vega/references/brief-contract.md`).
- `/vega-forge` (the network sub-chain of the Vega package) consumes `pipeline.json` from a `--mode pipeline` run: the model-family posture it must honour or overrule with a stated source, and the world model (§2–§4 in machine form) it must build the network *in* — its emitted code is the adapters of this blueprint's ports, and `/architect --baseline` on that code diffs it against this `blueprint.md`.
- `/vega-pack` (the shipping stage of the same sub-chain) calls `/blueprint --extends` from its blueprint declaration fork when the bundle serves endpoints the blueprint never declared — inspection, the showcase slot, a demonstration service on a `batch` / `stream` blueprint — so `/architect --baseline` on the bundle reads them as bets (see *Extending a pipeline blueprint*).

Position in the chain: `/spec-analysis → /domain-forge → /blueprint → (build) → /architect`, and on the data lane `/dataset-forge → /data-lens → /dataset-shaper → /blueprint --mode pipeline → (build) → /architect`.

## Trigger

User invokes:

- `/blueprint` — auto-detect the input artifact in the cwd (spec-analysis HTML or domain-forge model); propose a whole-system architecture.
- `/blueprint <path>` — point at a specific spec-analysis HTML or domain-forge model file.
- `/blueprint --concern "<text>"` — focused lens (e.g., `--concern "propose the audio-pipeline decomposition only"`).
- `/blueprint --axis model|abstraction|composition` — single-axis proposal.
- `/blueprint --blueprint-only` — world-model blueprint, skip the ranked findings.
- `/blueprint --findings-only` — ranked design findings, skip the blueprint prose.
- `/blueprint --report-only` — derive + render artifacts, skip the interactive gating turns (one-shot; you review the finished blueprint).
- `/blueprint --mode system|pipeline` — `system` (default for prose specs and software models) proposes the software system; `pipeline` (default for a dataset model, allowed for a `data-project` spec) proposes the data / ML pipeline that consumes the dataset.
- `/blueprint --extends serving.api | serving.inspection | serving.showcase | serving.demo --pipeline <run>/pipeline.json --entries '<json>' --because "<pressure>"` — amend a finished `--mode pipeline` run with the serving surface a later stage adds (served endpoints, the inspection endpoints, the showcase slot, a demonstration service on a `batch` / `stream` blueprint) **as gated bets**, without re-deriving anything else. See *Extending a pipeline blueprint* below.

Args may combine (`--extends` combines only with `--report-only` and its own options).

## Procedure

### Step 1 — Ingest and classify the input

Locate the input artifact (from `<path>` or by scanning the cwd for `spec-analysis*.html`, `model.html`, or a domain-forge HTML). Classify it:

- **Prose spec** (`spec-analysis.html` or any prose description) → you derive entities, boundaries, and flows from prose. Lower structural fidelity; more designer inference.
- **Domain model** (`domain-forge model.html`, with embedded Turtle / `data-*` nodes) → entities, relationships, and constraints are already formalized. Higher fidelity; your job is the *architecture over* the model, not re-deriving it.
- **Both present** → prefer the domain model for entity/relationship grounding, use the prose spec for behavior, flows, and business rules. Note which artifact grounds which claim.
- **Dataset model** (a domain-forge HTML whose Turtle carries `ex:sourceKind "dataset"`, listed by `python3 <domain-forge>/scripts/strip_layer.py <file> --list` as carrying `geometry`, and possibly `analysis` and `shape`) → `--mode pipeline`. The entities are *not* the software's: the `Record` class and its properties describe the data; the system to design is the one that ingests, shapes, trains on and serves that data. See *Dataset models* below. A `data-project` spec-analysis alongside it grounds the existing flows; the dataset model grounds the data.

If no recognized artifact is found, say so and ask for a path or a prose description. Do **not** invent a spec.

#### Dataset models (`--mode pipeline`)

Read, and cite by tag, the three layers' data scripts — never the rendered prose alone:

- `geometry` (`layer-geometry-data`): `typing` (roles), `basis`, `derivations` and `cycles` with the active orientation, the chosen partition (`partitions.chosen` is a label; `features`, `dropped_for_leakage`, `input_dim`, `task` are on the `partitions.candidates[]` entry with that label); `disagreements`; `space` and `source.rows`. Tags use the layer's key names: `[geometry:basis]`, `[geometry:derivations/total]`, `[geometry:partitions/chosen]`, `[geometry:cycles/price-qty-total]`.
- `analysis` (`layer-analysis-data`, when present): `findings[]` with `severity` and `so_what`, `context.time`, `context.spatial`, `modules.drift`, `modules.importance` (learnability, importance, leakage probe). Tags: `[analysis:F3]`, `[analysis:importance]`, `[analysis:context.time]`.
- `shape` (`layer-shape-data`, when present): `recipe.steps[]` by phase, `manifest` (`fit_on`, fitted parameters), `verification`. Tags: `[shape:S6]`, `[shape:phase/split]`, `[shape:manifest]`.

The world model of a pipeline (Step 2) is: the **data contract** at each boundary (raw → typed → shaped → features → predictions; the shaped schema in `shape.manifest` *is* the feature contract), the **stages** (ingestion, validation against the quality findings, the recipe's phases as transformation stages, training on the partition, evaluation with the CV protocol the analysis used, serving, monitoring for the drift the analysis measured), and the **artifacts** (the recipe, the fitted parameters, the model, the metrics) as value objects with versions. The style axes (Step 2.5) read: *distribution* — a script, a scheduled pipeline, a service (forces: data volume from `geometry.source.rows`, latency of the use the user states); *internal* — ports for the data source, the feature transformer (the recipe as the one implementation), the model, the trainer, the evaluator, the store, the predictor and the monitor (the eight roles the network sub-chain implements); *interaction* — batch, streaming, online inference (forces: `context.time` regularity, the split kind `shape:phase/split`); *data* — files, warehouse, feature store, model registry (forces: reproducibility from `shape.manifest`, the drift findings). The **model-family posture** (linear / tree / neural / none yet) is a Structural bet derived from `[geometry:partitions/<label>]` (task, `input_dim`), `[analysis:importance]` (learnability, which features carry it) and `[analysis:*]` on distributions — it is a *posture with forces*, never a training decision or a performance claim. **Monitoring** derives from the analysis's drift and quality findings: what to watch is what was found fragile.

In `--mode pipeline` the run directory also receives **`pipeline.json`** — the machine-readable hand-off the network sub-chain of the Vega package (`/vega-forge`) reads instead of parsing prose: `{ "schema": "blueprint/pipeline@1", "input": { "model": <path>, "model_digest": <canonical Turtle digest>, "shape_manifest_digest": <the shape layer's `manifest_digest` — the sha256 of the shaped manifest.json FILE; the one definition /vega-forge's compiler and the emitted FeatureContract compare against, so a stale blueprint is refused, not warned about> }, "bets": [ { "id": "bet0", "title": "architectural style", "resolution": "user|auto-resolved", "chosen": "<one line>" }, { "id": "bet2", "title": "model-family posture", "resolution": …, "chosen": … } ], "posture": { "family_class": "linear|tree|neural|none-yet", "family": "<a vega family name when the bet named one, else null>", "forces": ["[geometry:partitions/chosen]", "[analysis:importance]"], "alternatives": [{ "family_class": …, "when": … }], "bet": "bet2", "resolution": "user|auto-resolved" }, "feature_contract": { "source": "[shape:manifest]", "input_dim": …, "label": … }, "serving": { "mode": "batch|online|stream", "latency": …, "source": …, "api": { "errors": ["413", "503 busy + Retry-After", "404/405 problem+json", "404 /v1/monitor without reference"], "endpoints": [ { "method": "POST", "path": "/v1/predict", "capability": "core|code|inspection|showcase", "calls": "<a ports[] name>", "target": "serving.api|serving.inspection|serving.showcase|serving.demo", "bet": "bet<n>", "source": …, "fallback": { "capability": "core", "target": …, "bet": … } } ] }, "concurrency": { "max_inflight": 2, "inflight_wait": "0", "max_body": "…", "admission": "before body" }, "inspection": { "enabled": true, "weights": "full|summary", "bet": …, "source": … }, "showcase": { "slot": "showcase/", "bet": …, "source": … }, "demo": { "adapter": "online", "forced_by": "--serve", "production_mode": "batch|stream", "scope": "demonstration", "bet": …, "source": … } }   ← `api` and `concurrency` only for `online`/`stream` (or with `demo`): the bets the served bundle is audited against; `api.endpoints`, `inspection`, `showcase` and `demo` are written only by `--extends`, "training": { "protocol": "<the CV / split protocol the analysis used>", "source": "[analysis:importance]" }, "monitoring": [ { "watch": …, "source": "[analysis:F<n>]" } ], "system": { "name": …, "package": "<a Python package slug>", "source": … }, "boundaries": [ { "name": "raw|typed|shaped|features|predictions", "contract": …, "source": … } ], "ports": [ { "name": "<§3 port>", "role": "DataSource|FeatureTransformer|Model|Trainer|Evaluator|ModelStore|Predictor|Monitor", "responsibility": "<one line>", "signature": ["def <method>(self, <at most 5 parameters>) -> <§2 type>: ..."], "filled_by": "/vega-pack", "source": … } ]   ← `filled_by` on the FeatureTransformer, Predictor and Monitor ports only, "modules": [ { "name": "<§4 module>", "paradigm": "OOP|FP — OOP for every module that owns a port", "ports_owned": […], "depends_on": […] } ], "artifacts": [ { "name": "<§2 value object or entity>", "kind": "value|entity", "identity": …, "external": { "owner": "/vega-train|/vega-eval|/vega-pack|/dataset-shaper|…", "where": "<where it lives>" } } ]   ← `external` only on an artifact a later stage holds outside the emitted package, "anchors": ["EJ-17", "EJ-20", …], "extends": [ { "target": "serving.<…>", "base": "<base run>/pipeline.json", "base_digest": "sha256:…", "bet": "bet<n>", "by": "<the stage or user that asked>", "at": "<UTC>" } ]   ← `extends` only on an amended run }`. Every entry carries the same tags as §2 (`/vega-forge` strips the brackets when it cites them); `bets[]` lists every structural bet by id so a downstream source `blueprint:bets/<id>` resolves; `system`, `boundaries`, `ports`, `modules`, `artifacts` are §2–§4 **in machine form** — the same names, nothing invented for the file — and `anchors` are the EJ items the findings cite, from the corpus. This is what makes the blueprint the model the network is *built in*: the network sub-chain emits its code as the adapters of these ports inside this module map (`vega-forge/references/code-contract.md`), and `/architect --baseline <run>/blueprint.md` is how conformance is later proved. Each port carries a `role` from the Vega vocabulary beside its own name (the name is the blueprint's — `Recipe`, `Registry` — the role is what the emitter keys on); a pipeline blueprint draws all eight roles (`DataSource`, `FeatureTransformer`, `Model`, `Trainer`, `Evaluator`, `ModelStore`, `Predictor`, `Monitor`) — a role left out is a designer's choice the ledger must show, and the Vega stage that needs it will refuse to invent it. A pipeline blueprint that leaves `ports` or `modules` empty has no world model to build in and the summary says so. **Every artifact has a home**: the network sub-chain emits the artifacts its package holds as the value objects of `domain.py`, and an artifact that lives outside that package — the campaign, which is `/vega-train`'s campaign directory; the recipe, its fitted parameters and the shaped parts, `/dataset-shaper`'s; the seal and the verdict, `/vega-eval`'s; the served bundle's API values and the network's exports (TorchScript, ONNX and their parity record, under the bundle's `exports/`; the package's `ArtifactExporter` writes them, it does not hold them), `/vega-pack`'s — is drawn in §2 like any other and declared in `pipeline.json` as `"external": { "owner": "<the stage that holds it>", "where": "<where it lives>" }`. `owner` is one of `/dataset-forge`, `/data-lens`, `/dataset-shaper`, `/blueprint`, `/vega-train`, `/vega-eval`, `/vega-pack`, `/vega-show` (never `/vega-forge`: what it holds is in the package), and `where` names the directory or file. `/vega-forge` records each external artifact with its owner in `blueprint-map.json` (`vega-forge/references/code-contract.md` §3) and refuses, by name, an artifact that its `domain.py` does not define and §2 does not declare external, or one declared external that `domain.py` defines — an artifact with no home is the divergence `/architect --baseline` would otherwise find, so §2 says for each entity and value object whether the package holds it or which stage does. **A held artifact takes the package's name for it**: before drawing §2, run `python3 <vega-forge>/scripts/held_artifacts.py --shape <row|window|graph>` (the vega-forge skill's directory, beside this one; the shape is the shaped data's sample: independent rows, series cut into windows, or nodes and edges — the package carries that shape's values only) — it lists every name the backend's `domain.py` defines for that shape that an artifact can be held under (value objects, enums, and aliases), each with what it is — and name each package-held artifact by one of those names; a domain word for it (the parcels' `TensorPart`, say) is prose beside the name, never a second artifact. An artifact that is not on the list is declared external with the stage that holds it, or it is not an artifact of the package. **A port a later stage fills says so**: the network sub-chain implements the DataSource, Model, ModelStore, Trainer and Evaluator adapters when it emits the package, and `/vega-pack` fills the FeatureTransformer (the shaper's fitted recipe), the Predictor and the Monitor when it builds the bundle — until then the package carries them as stubs that say so. Each of those three ports carries `"filled_by": "/vega-pack"` in `pipeline.json` and no other port does (the emitter refuses either drawing otherwise); §3 says of each that its adapter is filled by `/vega-pack`, and §4 says that the arrows of a module whose ports are all filled later (serving's, to the model module) are taken when that stage fills them — a baseline that states a pending adapter or arrow as present is the divergence `/architect --baseline` reports. **The package is the library the processes wire, not the processes**: the training run is `/vega-train`'s campaign (its trainer wires the DataSource, the Model, the ModelStore and the Trainer per run) and the serving process is `/vega-pack`'s bundle (`predict.py`, `serve/app.py`), both entry points outside the emitted package; §0's distribution axis says so (one package, wired by two stage-owned entry points), never that the package contains a training script or a serving app. **A module that owns a port is `OOP`**: a port is a type and its adapter is the object behind it — most guard a state (the data adapter its seal, the store its digests, the predictor its loaded network), and the package's `Trainer` is a stateless driver whose fit's state lives in the fit it runs (a fold over epochs, its resolved collaborators frozen), so §4's *because* says the adapter is an object behind a port and does not claim that every adapter holds state — so the paradigm bet may assign `FP` only to modules with no port — `domain` (value objects), a metrics library; the Vega conformance gate (`vega-forge/references/code-contract.md` §5, C14) refuses a package whose port adapter is not a class or whose module declares `FP` while owning a port, and the bet's *because* in §4 says which it is. **A port's methods are its adapter's**: the code behind each port is the backend's adapter of the port's role, a fixed class the blueprint does not write (`ShapedParts`, `Network`, `ModelRegistry`, `Trainer`, `Evaluator`, `Predictor`, `Monitor`; the recipe's transform-only module for the `FeatureTransformer`), so before drawing §3 run `python3 <vega-forge>/scripts/port_surface.py --shape <row|window|graph> --task <kind>` (the kind is the partition's task, read through a sequence task to its base: `binary`, `multiclass`, `regression`, `forecast`, …) — it prints, per role, the **port methods** (the declarations a port of that role carries, which the chain's stages call through it), the adapter's other public methods, and the package's own interfaces `ports.py` declares beside the ports — and draw each port's `signature[]` from it: every port method of the role, each with exactly the parameters printed (names, order, defaults; the annotations may name any type `held_artifacts.py` lists), plus any of the adapter's other methods the port should expose, with theirs. The Model port's port methods include the **serving seam** — `head` (the raw head of an assembled batch) and `anatomy` (a read-only view of the staged network, the domain's `NetworkAnatomy`: a view whose identity is the network's and whose state is read when asked, so §2 draws it as such and never as "the network as data") — which is what the bundle's Predictor and inspector read of the network object and all they read, so §3 draws them beside the scoring methods and says so (a Model port without them is refused like any other omission); beside the network object, the Predictor reads two support files of the Model module, its readout (what an output means) and its sample assembler (the network's own cut of a served batch from the shaped rows), and §3 says so rather than claim the seam is all the Predictor reads of the module. §3 names the port for the domain and says, beside it, which adapter class implements it (*the `Fitter` port, implemented by the package's `Trainer`*), so the class name the code publishes is a gated fact rather than a drift; and §3 lists the package's own interfaces (`ModelBuilder`, `TrainedNetwork`, `TrainingData`, `Callbacks`, `ScoredModel`, `ArtifactExporter`) as the backend's, not as ports — the Trainer, Evaluator and ModelStore port methods are typed with them as `port_surface.py` prints them (the network a fit trains, the DataSource as a fit reads it — without its sealed door —, a model the Evaluator scores, the builder a checkpoint is restored through); and the evaluation's baselines are scored models (`ScoredModel`), never further implementations of the Model port, so §0 and §3 may say each port has one adapter. The same listing ends with the module arrows the backend's code needs, role to role (the Trainer's and the Evaluator's code read the Model module's readout and metrics): §4's `depends_on` draws each of them between the modules owning those roles, and no arrow the listing does not name — the conformance gate (C4) refuses a package whose imports run along an arrow §4 did not draw, and an arrow no import can take (one adapter's module to another's, say) is a dependency the code cannot honour. The emitter refuses, by port and method, a port that omits a port method, declares a method its adapter does not have, or changes a parameter — a call written against such a port binds its arguments to the wrong parameters. Likewise §2: every name `held_artifacts.py` lists is a value object the package holds whether §2 draws it or not (the map lists the ones §2 does not under `backend_artifacts`), so §2 never strikes one from the world model — it draws the ones the domain names and leaves the rest to the backend. And §2 names the source of a value the package reads from the spec instead of restating it in another order or form: the label set is the spec's `task.classes`, in that order (the geometry layer's listing — not the analysis's frequency order), and the features are the shaped manifest's output schema. And a boundary's `contract` states what the package enforces at it, no more: raw rows carry every input column (the recipe drops what the shape step dropped, a leakage column included), the shaped and features boundaries are the shaped manifest's output schema. **A port method takes at most 5 parameters**: each `ports[].signature[]` entry is one method of the port as a Python declaration (`def fit(self, model, spec: NetSpec, data, callbacks=None, run: RunContext | None = None) -> FitResult: ...`), rendered verbatim into the emitted `ports.py` and implemented by the adapter, so it is held to the gate's C12 — **at most 5 parameters beyond `self` / `cls`**, keyword-only ones counted, no `*args` / `**kwargs` (a port is an interface, and a variadic hides it). When an operation needs more inputs, they travel as a **parameter object**: a §2 value object listed in `artifacts[]` with `kind: value` and tagged like any §2 entry — and, the package holding it, one of the names `held_artifacts.py` lists (`RunContext` carries a fit's wall-clock budget, its checkpoint directory and its gradient sampling rate — the ModelStore the checkpoints go through is the Trainer's injected collaborator, `Trainer(store)`, never a field of a run's value, so §2 does not give `RunContext` a store; `BatchPlan` a part's batching; `MetricOptions` a metric pass's knobs), never a second name for one of them; §3 and the §7 ledger name the parameter objects §2 draws, and no other — EJ-51 (*design method signatures carefully*: long parameter lists are the defect; a helper class holding the group is the remedy) and EJ-2 (a builder when a group has many optional members). A signature with a sixth parameter is not a style note: the emitter renders it, C12 refuses the package, and the fix is a re-run of this skill, so §3 draws every port method within the bound and says which value object carries the rest. **A signature annotates only with names the emitted `ports.py` can import**: it imports no code (its annotations sit under `TYPE_CHECKING`), so a `ports[].signature[]` entry names only builtins (`dict`, `list`, `tuple`, `str`, `int`, `float`, `bool`, `None`), the `typing` names `Any`, `Iterator`, `Iterable`, `Sequence`, `Mapping`, `Callable`, `Optional`, `Union`, `Literal`, the ports §3 draws (by their names) and the names `held_artifacts.py --shape <…>` lists (`BatchKey`, `EpochRecord`, `MetricOptions`, `Versions` among them); a framework or library type — a pandas `DataFrame`, a torch `Tensor`, a numpy array — stays unannotated (`def predict(self, rows) -> list[Prediction]: ...`), and every method is an instance method whose first parameter is `self`. A name outside that vocabulary is refused by the emitter, by port and name, and the fix is a re-run of this skill. `pipeline.json` is the fourth artifact of a `--mode pipeline` run — written with the other three, listed in the summary and required by the stop criteria. The posture stays what it is in the prose — a bet with forces, and `resolution` says whether a human gated it. The summary's *suggested next step* for a pipeline blueprint is `/vega-forge <model> --pipeline <run>/pipeline.json`.

Anchors keep their corpus: the EJ items apply to the pipeline's code (immutability of the feature contract, the recipe as a value object, builders for configuration, no leaking of fitted state), and where no item fits, SOLID / HEX / CQRS / MEASURED-PERF as today. `location` accepts the three new tag families beside `spec §` and `model:`.

State in your first status line: which artifact(s) you ingested, their classification, and (for prose) the inherent inference risk.

### Step 2 — Derive the candidate world model (internal)

Read the spec/model and build, in your own working notes, a first-pass model along the three axes:

- **Model** — entities (identity + lifecycle), value objects (equality by value), aggregates, bounded contexts. For a domain-forge input, lift these from the model; for prose, extract them and mark each with its source span.
- **Abstraction** — the ports/interfaces the system needs, each with its method signatures (at most 5 parameters per method beyond `self`; a longer list becomes a parameter object in the Model axis), where the abstraction surface should be widest, where concrete implementation grounds execution.
- **Composition** — module map, dependency direction, paradigm per subsystem (OOP for stateful identity and for every module that owns a port — the adapter is an object behind the port; FP for pure transformation with no port), where hexagonal/ports-and-adapters earns its indirection.

Every line you write here carries a tag: `[spec §X]`, `[model:Entity]`, or `[DESIGNER'S CHOICE — pressure]`. This tagging is not cosmetic; it is the input to the gate in Step 4 and the final artifact's honesty.

For a heavy spec, dispatch an `Explore` or `general-purpose` sub-agent scoped to reading the artifact and returning the extracted entity/flow inventory — keep only its conclusion in your context.

### Step 2.5 — Derive the architectural style (multi-axis)

Before any decomposition, fix the system's architectural style — but **not** as a single label off a menu. A style is a **point across independent axes**; pick a coherent combination, each axis driven by a force the spec actually exhibits. "We'll use microservices" with no per-axis force is the cardinal failure of this step.

The axes (vocabulary, not a rigid checklist — add an axis a spec demands, drop one it doesn't exercise):

- **Distribution** — single process → modular monolith → service-oriented → microservices. Selected by: independent scaling needs, deploy-boundary / team-boundary pressure, failure-isolation requirements.
- **Internal dependency** — layered (n-tier) / hexagonal (ports-and-adapters) / clean / pipes-and-filters. Selected by: how many implementations of a port the spec implies (swappable backends → hexagonal earns its indirection), testability demands.
- **Interaction** — request-response / event-driven / streaming / batch. Selected by: latency tolerance, long-running async work, fan-out, ordering and delivery guarantees the spec states.
- **Data** — shared database / database-per-service / CQRS / event-sourced. Selected by: consistency-vs-availability needs, multi-tenancy, audit/history requirements, read/write asymmetry.

For each axis, write the chosen point and either the **spec force** that drives it (`[spec §X]` / `[model:Y]`), or `[DESIGNER'S CHOICE — pressure]` when the spec is silent and you're committing on a design pressure (e.g. "no multi-tenant requirement stated yet → shared DB, revisit if tenancy lands").

Then **check coherence**: the combination must hold together. Flag tensions explicitly — e.g. "microservices distribution + shared-DB data" is the distributed-monolith anti-pattern; surface it rather than emit it silently. The output is one coherent style-point plus the forces that justify each axis.

For a heavy spec, this can ride the same scoped sub-agent read as Step 2.

### Step 3 — Identify the structural bets

If `--axis model|abstraction|composition` was passed, scope Steps 2–5 to that single axis — derive, gate, and emit findings only for it (style bet 0 still runs, since every axis lives inside the chosen style). If `--concern "<text>"` was passed, bias the lens of Steps 2–5 toward that concern and note it in §1.

From the candidate model and the style-point, list the **structural decisions** — the choices that, if wrong, invalidate large parts of the blueprint. The architectural style from Step 2.5 is **Structural bet 0** and is gated **first**: §3 ports, §4 module map, and §6 perf posture all derive from it, and the decomposition only has meaning *inside* the chosen style. The rest typically:

- **Bet 0** — the architectural style (the multi-axis point).
- **Bet 1** — the top-level decomposition (services / modules / bounded contexts), *inside* the chosen style. In `--mode pipeline`: the stage decomposition and where the recipe's phases fall.
- **Bet 2** — the paradigm assignment per subsystem. In `--mode pipeline`: also the model-family posture.
- **Bet 3** — the hard boundaries (anti-corruption layers, ports). In `--mode pipeline`: the feature contract (the shaped schema) and the monitoring boundary.
- plus any decision the spec underdetermines and you had to invent.

These are what you gate. Everything else (naming, field types, intra-boundary pattern choice) you decide yourself and note as tactical.

### Step 4 — Gate the structural bets, one per turn

This is the human-in-the-loop core. For each structural bet, in dependency order **starting with bet 0**:

1. Surface it as a **decision**, not a fait accompli: state the proposal, the spec/model evidence behind it, the designer's-choice elements, and 2–3 alternatives with trade-offs.
2. Recommend one. Lead with the evidence, not the conclusion.
3. Wait for approval, redirection, or a redraw of the boundary.
4. Record the resolved decision (and any stance the user expressed) before moving to the next bet.

Gate bet 0 per-axis, because that is where the leverage and the trade-offs live:

> **Structural bet 0 — architectural style.** I propose: distribution = `<X>` [spec §A]; internal = `<Y>` [spec §B]; interaction = `<Z>` [DESIGNER'S CHOICE — pressure]; data = `<W>` [model:C]. Coherence: `<holds / named tension>`. Alternatives on the axis that matters most: `<A>` (trade-off), `<B>` (trade-off). I recommend this combination because `<evidence>`. Approve, change an axis, or redraw?

Subsequent bets are framed *inside* the approved style:

> **Structural bet N — <name>.** I propose <X>, derived from <spec §Y / model:Z>; the <W> part is a designer's choice driven by <pressure>. Alternatives: <A> (trade-off), <B> (trade-off). I recommend <X> because <evidence>. Approve, pick another, or redraw?

If `--report-only` was passed, skip the gating turns: make each call yourself, stamp it, surface all of them together in the final artifact for post-hoc review, and note in the summary that gates were auto-resolved.

One structural decision per turn. Do not batch the big bets — batching is what produces a whole blueprint built on a wrong early call that the user then has to reject wholesale.

### Step 5 — Compose the blueprint and findings

Once the structural bets are resolved, write the two artifacts to a run directory `blueprint-runs/<UTC-timestamp>/`:

**`blueprint.md`** — same section shape as the `/architect` auditor's output, so a later `/architect` run can diff against it. Sections, in order:

- **§0 Architectural style** — the multi-axis style-point, each axis with its trace tag and the coherence check. Precedes everything; it is the topology the eventual code was *supposed* to inhabit.
- **§1 Domain & spec basis** — what the system is, which input artifact(s) ground it, the inherent inference risk. (Replaces architect's "current world-model assessment" — there is no current code, so this grounds in the spec instead.)
- **§2 Proposed world model** — entities, value objects, aggregates, bounded contexts. **Every item carries its trace tag.** This is the load-bearing section and the design baseline.
- **§3 Proposed interfaces / ports** — the abstraction surface: each port with its methods (at most 5 parameters each beyond `self`, the parameter objects named) — in `--mode pipeline` the same declarations as `pipeline.json.ports[].signature`; after an `--extends` amendment, also the *served endpoints* table (method, path, capability, the port each calls, the bet that declared it).
- **§4 Module map** — modules, dependency direction, paradigm per subsystem (the style of §0 made concrete).
- **§5 Abstraction inventory** — where abstraction is widest, where concrete grounds it, the concrete-to-abstract intent.
- **§6 Performance posture** — the cost of each proposed abstraction, flagged explicitly; in-process vs network-hop costs follow from §0's distribution axis.
- **§7 Designer's-choice ledger** — every `[DESIGNER'S CHOICE]` collected in one place with its justifying pressure, so the user sees exactly what was invented beyond the spec.
- **§8 Risk register & open questions** — what the spec couldn't resolve.

**`findings.json`** — ranked **design proposals** (not audit findings), using the **same field names** architect-auditor emits so a later `/architect` run can diff against them. Each finding carries, with names matching the auditor exactly:

- `id`, `axis`, `anchors`, `anchor_justification` (one clause per anchor — see the load-bearing rule below), `title`, `rationale` (the prose argument — this is where each anchor's principle does its load-bearing work, same field the auditor uses), `proposed_change` (NOT `proposal`), `abstraction_cost`, `performance_implication`, `impact`, `risk`, `effort_estimate_min` (NOT `effort`), `leverage_score`, `blueprint_anchor` (the `§N.section` it links to — the same field the auditor uses).
- `depth` — `structural | architectural` only. **Documented divergence:** the auditor also allows `surface`; `/blueprint` omits it because surface-depth findings require existing code to be tactical against.

Two greenfield-specific field semantics, **documented as intentional divergences** (not silent drift) so the downstream diff still resolves:

- `location` → carries the **spec/model span** the finding derives from (e.g. `spec §3.2` or `model:Segment`) instead of a `file:line`, since no code exists. This doubles as the trace tag: it is either a spec/model citation **or** the literal `DESIGNER'S CHOICE: <pressure>`.
- `concrete_reduction_loc` → set to `null` (no code to reduce). The auditor treats it as an integer; `null` is the explicit "not applicable pre-code" signal, and `leverage_score` must be computed without it.

Rank by must / should / could.

**The anchor must be load-bearing, not decorative** (inherited hard from `/architect`'s auditor): the EJ anchor must come from the 43-item grounded corpus (see **Anchor corpus** above), and each anchor's `anchor_justification` must name what *that specific item's principle predicts here* — ideally echoing the corpus table's "anchor must justify" clause for that item — such that swapping a different anchor would change the reasoning. Test: if the proposal would read identically with the anchor deleted, the anchor is decoration — find the one that actually drove the design, or drop it. A bare anchor tag, or one citing an EJ item outside the corpus, is an incomplete finding.

Restate §0, §1, and §2 inline in chat **verbatim** after writing — they are the load-bearing framing the user must see.

### Step 6 — Render the HTML

Produce a single self-contained `blueprint.html` in the run directory, matching the prose-first, scrollable, offline style of the existing `spec-analysis.html` artifacts: no external scripts, no network deps. It contains the blueprint sections as connected prose, the ranked-findings scorecard (anchors rendered **with** their justification clause, never bare), the architectural-style axis table, and the designer's-choice ledger as a callout. End with a MD + JSON export of the underlying artifacts.

**Publish it.** The HTML is the product, and the user sees it before reading about it: open `blueprint.html` in the conversation — through the runtime's artifact or file tool when it has one (self-contained, it renders as it is), otherwise by delivering the file so it renders inline — and say in one line what it shows. There is no summary of an unpublished blueprint.

### Step 6b — Write the stage brief

When the run is part of a Vega lane (invoked by `/vega`, or a dataset model with a `shape` layer, or the user asks), write `<run>/docs/blueprint-brief.md` per `vega/references/brief-contract.md`: the frontmatter with the records every number resolves to (`blueprint.md`, `findings.json`, `pipeline.json`), then *what this stage was for*, *what was decided* (one paragraph per structural bet: the choice, its resolution — gated or auto-resolved — the forces cited, the alternative; the posture; the eight port roles and the paradigm per module), *what ran*, *what came out* (the findings by depth with their anchors, the trace and anchor integrity results), *what the reader should check*, *open* (designer's choices, concerns for a re-run), *next* (the `/vega-forge` invocation). Every number with its record key; nothing the artifacts do not hold.

### Step 7 — Final summary

```
## /blueprint summary (run <UTC-timestamp>)

Input ingested: <artifact path(s)> (<prose | model | dataset model | both>) — mode <system | pipeline>
Artifacts: blueprint.md • findings.json • blueprint.html (published) • pipeline.json (--mode pipeline) • docs/blueprint-brief.md (Vega lane)  (paths)

Architectural style: <distribution / internal / interaction / data> — coherence <holds | tension noted>
Structural bets resolved (N): list with the chosen option
Designer's choices (M): count — full ledger in §7
Open questions (K): count — see §8

Trace integrity: every §2 entity, every style axis, and every finding carries a
spec/model citation or a DESIGNER'S-CHOICE stamp. <PASS | list any unstamped>
Anchor integrity: every finding anchor carries a load-bearing justification clause. <PASS | list bare anchors>

Suggested next step: build against this baseline, then run /architect on the
code to diff it against blueprint.md §0 (style) and §2 (world model)
— in --mode pipeline: /vega-forge <model> --pipeline <run>/pipeline.json to
engineer the network the posture bet on (pipeline.json written: yes/no).
```

## Extending a pipeline blueprint (`--extends`)

A pipeline blueprint is gated before any code exists, so the serving surface a later stage adds — the inspection endpoints, the showcase slot, a demonstration service packed with `--serve` on a `batch` or `stream` blueprint — was never a bet. Shipped undeclared, `/architect --baseline` on the bundle reads it as a divergence. `--extends` amends a finished `--mode pipeline` run with that surface **as bets**, gated like the others, and changes nothing else: the style, the decomposition, the ports, the modules and the posture stay what the base run gated. It is the mechanism `/vega-pack`'s blueprint declaration fork applies (`vega-pack/references/bundle-contract.md`).

```
/blueprint --extends <target>[,<target>…] --pipeline <base run>/pipeline.json --entries '<json>'
           --because "<pressure>" [--by <caller>] [--weights full|summary] [--slot <dir>] [--report-only]
```

`--because` is required: it is the force of every bet the call gates, and a call without it is refused before any bet is surfaced. `--by` names who asked, recorded in `extends[].by`; it defaults to `user`.

**The four targets** — exactly these; any other is refused with this list:

| target | records | accepts entries with `capability` |
|---|---|---|
| `serving.api` | the endpoints, in `serving.api.endpoints[]` | `core`, `code` |
| `serving.inspection` | `serving.inspection = { "enabled": true, "weights": "full" \| "summary" }` (`--weights`, default `full`) and its endpoints (today `GET /v1/network`, `GET /v1/network/tensor/{tensor_id}`, `POST /v1/explain`) | `inspection` |
| `serving.showcase` | `serving.showcase = { "slot": "<bundle-relative dir>" }` (`--slot`, default `showcase/`) and its endpoints (the slot's routes as the bundle's `MANIFEST.serve.showcase_slot.routes` lists them; today `GET /`, `GET /console`, `GET /showcase/{path}`, `GET /v1/story`, `GET /report.pdf`, `GET /v1/showcase`; `GET /` and `GET /console` override the core console fallback: see the fallback rule below) | `showcase` |
| `serving.demo` | `serving.demo = { "adapter": "online", "forced_by": "--serve", "production_mode": "<serving.mode>", "scope": "demonstration" }` — an online **demonstration adapter beside** a `batch` or `stream` blueprint, never a change of `serving.mode` (§0's interaction axis stays what it was gated as) — and the endpoints that adapter answers | `core`, `code` |

The entries are the caller's, never typed here: `/vega-pack` passes the `ENDPOINTS` and `CAPABILITY` its route modules declare (`serve/routes_*.py`) that `serving.api.endpoints[]` lacks, grouped by capability — `inspection` to `serving.inspection`, `showcase` to `serving.showcase`, `core` / `code` to `serving.demo` when the blueprint's mode is `batch` or `stream` and the bundle was packed with `--serve`, else to `serving.api`.

**The entry shape**, one for every target: `{ "method": "GET|POST|PUT|PATCH|DELETE", "path": "/…", "capability": "core|code|inspection|showcase", "calls": "<port>" }`. `--entries` is a JSON list when one target is named, and an object `{ "<target>": [entries…] }` when several are. The rules:

- `method` is an upper-case HTTP method; `path` starts with `/`, path parameters in braces (`/v1/network/tensor/{tensor_id}`).
- `calls` names a port of `ports[]` by its **name** (`OrderScorer`); a caller that passes a **role** (`Predictor`, the vocabulary of the serving templates) gets it resolved to the one port carrying that role. An endpoint that calls no port (`/health`, a static page) omits `calls`. A `calls` that resolves to no port is **refused**: the amendment cannot invent a port — ports are §3's, and a new one is a re-run of the whole blueprint.
- `capability` must be one the target accepts (the table); an `inspection` or `showcase` entry through `serving.api` is refused and names the target to use.
- `(method, path)` is unique across `serving.api.endpoints[]`: an entry already present with the same fields is a no-op; with different fields it is refused — an endpoint changes meaning only through a re-gated bet.
- **The fallback rule.** One pair is not a conflict: a `core` entry and a `showcase` entry with the same `(method, path)` and the same `calls`. The bundle serves such a route from the core module as a fallback, and from the showcase slot when a slot is present. Today these are `GET /` and `GET /console`, which `routes_meta` serves as the console page. They are one endpoint, declared once. The entry keeps `capability: showcase` with the `serving.showcase` target and bet, and records the core declaration as `"fallback": { "capability": "core", "target": "<its target>", "bet": "<its bet>" }`. The order does not matter: `serving.demo` and `serving.api` are gated before `serving.showcase` in one call, and a base run may already hold either entry, but the result is the same one entry and neither is refused. A caller may therefore pass the fallback routes under both capabilities, as its route modules declare them.
- Every accepted entry, whatever its target, lands in the **one** list `serving.api.endpoints[]` with `target` and `bet` added — the list `/vega-pack` compares, as a set of `(method, path)`, with the bundle's `MANIFEST.serve.endpoints`.
- `serving.demo` on an `online` blueprint is refused as redundant: the served app is already the production adapter.
- `serving.api.errors` and `serving.concurrency` (above) are written with the first endpoint on a blueprint that had none (`batch`, or `stream` without them): the served surface is audited against them.

**One bet per target, gated one per turn** (Step 4's format; order `serving.demo`, `serving.api`, `serving.inspection`, `serving.showcase`). The proposal is the entries; the force is `--because` (the stage and the flag that need them — `/vega-pack` over the bundle's route modules, `pack --serve` on a batch blueprint), stamped `[DESIGNER'S CHOICE — <pressure>]` because no spec or layer asked for a served endpoint; the alternatives are the caller's other options (pack without the inspection and page routes; ship with a recorded divergence). The bet takes the next free id (`bet<n>`, one past the highest), `title` = the target, `chosen` = one line naming the entries, `resolution` `user` when gated in the conversation or `auto-resolved` under `--report-only` — which is how a stage running `--unattended` calls it, passing `--by "<stage> (vega:default)"`. Each entry's and each block's `source` is that stamp, and each carries `bet`.

**What an amendment writes.** A new run directory `blueprint-runs/<UTC-timestamp>/`; the base run is never edited — its `pipeline.json` stays the one `/vega-forge` emitted against, and `/vega-pack` pins the amended one by digest, keeping the base as `emitted_against` in `blueprint-map.json`:

- `pipeline.json` — the base's content, the new `serving` blocks and endpoints, the new `bets[]`, and one `extends[]` record per target: `{ "target", "base": "<base run>/pipeline.json", "base_digest", "bet", "by", "at" }`. `base_digest` is the canonical digest every Vega stage computes — `"sha256:" + sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")))` — so the chain of amendments is checkable back to the gated base; a second amendment's base is the first amendment's run, and `extends[]` accumulates.
- `blueprint.md` — the base's, sections kept in number and order (diff-compatible with `/architect`): §3 gains the *served endpoints* table, §0's interaction axis notes the demonstration adapter (`serving.demo` only), §7 lists each amendment bet as a designer's choice with its pressure, §8 anything the amendment left open.
- `findings.json` — the base's, unchanged: an amendment declares, it proposes no new finding.
- `blueprint.html` — re-rendered and **published**, like any run; `docs/blueprint-brief.md` (Vega lane) gains the amendment's bets under *what was decided*.

The base must be a `blueprint/pipeline@1` document with `ports` and `modules`; anything else is refused before any bet is surfaced. The summary is Step 7's with `Input ingested: <base run>/pipeline.json (amendment: <targets>)` and the amended run's paths.

## Failure modes — DO NOT

- **Don't pick an architectural style by fashion or single label.** Style is a coherent multi-axis point, each axis traced to a spec force or stamped a designer's choice. A flat "we'll use microservices" with no per-axis force is the cardinal failure of Step 2.5. And never emit an incoherent combination (e.g. distributed monolith) without flagging it.
- **Don't present a designer's choice as a spec derivation.** If the spec doesn't say it, it is `[DESIGNER'S CHOICE]` with a named pressure — never silently promoted to "the spec requires." Greenfield's whole failure mode is confident invention.
- **Don't ship a decorative anchor.** Each finding anchor must carry a justification clause naming what its principle predicts here. If the proposal reads identically with the anchor removed, the anchor is decoration — fix or drop it.
- **Don't batch the structural bets.** One per turn, gated, starting with bet 0. A blueprint built on a wrong unreviewed early call wastes the run.
- **Don't invent a spec.** No recognized input artifact → ask, don't fabricate.
- **Don't re-derive a domain-forge model from scratch.** When a formal model is the input, lift its entities and relationships; your job is the architecture over them, not re-modeling.
- **Don't drift the artifact schema.** `blueprint.md` and `findings.json` must stay diff-compatible with `/architect`'s output, or the downstream chain breaks.
- **Don't load the Bloch corpus into your own context for breadth-reads — dispatch a sub-agent.** Keep the orchestrator lean, same as `/architect`.
- **Don't one-shot the HTML before §0/§1/§2 are approved** (except under `--report-only`).
- **Don't claim performance numbers.** With no code, perf posture is *intent and cost-flagging* derived from the §0 distribution axis, not measurement. Label it as such.
- **Don't draw a port method with more than 5 parameters, and don't give a port-owning module `FP`.** Both are refused by the Vega conformance gate (C12, C14) in code this blueprint's signatures and paradigms decide; the remedy is a parameter object in §2 and `OOP` on the module, written here, not patched downstream.
- **Don't let `--extends` re-open the design.** It declares the serving surface a later stage added, against the ports the base gated; it never adds a port or a module, never changes `serving.mode` (`serving.demo` is a demonstration adapter beside it), and never edits the base run.
- **Don't re-analyse the dataset in `--mode pipeline`.** The basis, the partition, the findings and the recipe are the layers' — cite them by tag. A blueprint that recomputes a correlation or picks a label the geometry did not is inventing a spec. And don't turn the model-family posture into a promise: it is a bet with forces, gated like the others, and the first training run is what settles it.

## Stop criteria

The command is done when:
1. All structural bets (style + decomposition + paradigm + boundaries) are resolved (gated, or auto-resolved under `--report-only`), AND
2. The three artifacts are written (four in `--mode pipeline`: `pipeline.json` beside them), `blueprint.html` was published in the conversation, the stage brief was written when the run is part of a Vega lane, and §0/§1/§2 restated verbatim in chat, AND
3. Final summary emitted with trace-integrity and anchor-integrity checks.

For `--blueprint-only`: done after the blueprint + HTML, no findings phase.
For `--findings-only`: done after ranked findings, no blueprint prose.
For `--report-only`: done after artifacts rendered + a note that gates were auto-resolved.
For `--axis`: done when the single named axis is derived, gated (after bet 0), and emitted; other axes are explicitly skipped in the summary.
For `--concern`: the full flow runs, lensed to the concern, which is named in §1 and the summary.
For `--extends`: done when every named target's bet is resolved (gated, or auto-resolved under `--report-only`), the amended run directory holds the four artifacts with `extends[]` recording each target against the base's digest, `blueprint.html` was published, and the summary names the base and the targets.
