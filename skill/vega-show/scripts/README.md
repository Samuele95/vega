# `/vega-show` scripts

The network, shown to people who were not in the chain. The page says nothing the chain did not
measure, and touches nothing the chain shipped: every number is a fact a script read from the
bundle's pinned records, the model writes only words, and the bundle is read, verified and served
— never written. `references/show-contract.md` is the contract; `catalog.json` is its machine
copy, and every script reads the closed sets from it rather than restating them.

## The files, by step

| file | what it is |
|---|---|
| `catalog.json` | `vega/show-catalog@1`: the contract's closed sets, bounds, forks S1–S3 with their unattended answers and markers, the `show:` stance lines, the report roles and walk roles, the showcase layout and slot, and every gate id. `tests/test_catalog_sync.py` fails when it and the contract diverge. |
| **core** | |
| `vega_show_core.py` | The value objects (`FactRef`, `Fact`, `FactSet`, `TaskShape`, `ShowPaths`, the enums built from the catalog), digests and the JSON edges. Pure: no argv, no printing, no exits; every script imports it and it imports none of them. `frozen_json.py` is its hashable JSON (`freeze`/`thaw`). |
| `show_ports.py` | The two ports the stage reads the world through: `RecordStore` (the bundle) and `LabelSource` (the report). |
| `records.py` | `BundleRecords(RecordStore)`: reads only what `MANIFEST.files` pins, digests the bytes it parses, and refuses the whole bundle (`BundleAltered`, gate PF2) on one differing file. |
| `report_labels.py` | `AuxLabelSource(LabelSource)`: the LaTeX `.aux` adapter — number, page, title and kind of every `\newlabel`, `@cref` skipped. |
| `role_map.py` | `read_role_map()`: `vega/label-roles@1` (`docs/label-roles.json`), checked against the `.aux` it names (an older map is `ERROR:` PF6, "recompile /document-project"). |
| `digit_rule.py` | The one digit rule (Unicode-aware, `facts.names` whitelist) and `PLACEHOLDER_RE`, used by the resolver (V2, V3), the page lint (PG5) and the layer (LY3). |
| **1 · facts** | |
| `extract_facts.py` | `--model M.pack.html --bundle B [--report R] [--label-roles J] --out plan/facts.json [--markers PATH] [--json]`. The fact set (`vega/show-facts@1`) and the stage's preflight: PF1 (the bundle layer is this bundle), PF2 (pinned records), PF3 (`predict.py --check` from a clean shell), PF4 (a service with `MANIFEST.serve.showcase_slot`), PF6 (the report's labels current), PF7 (catalog coverage: an unknown task, family, block type or shape is `ERROR:`), PF8 (the round trip passed). |
| `fact_sink.py` | `FactSink`: the key grammar, the kind checks, collisions and groups; `Catalogs` and `Refusal`. A fact is a value read at a ref, never computed (`count_of` is the one derivation). |
| `facts_data.py` · `facts_network.py` · `facts_campaign.py` · `facts_results.py` · `facts_report.py` | One collector each: the data and the recipe (`sizes`, `recipe`, `split`, `sample`, `inputs`, `class_balance`); the shipped network and its decisions; the ladder (`runs`, `selection`, `hpo`); the sealed test (`models`, `metrics`, `diffs`, `breakdown`, `verdict`, `limits`, the reserved `primary` and `ci_level`); the report roles or the stage briefs. |
| **2 · draft** | |
| `draft_plan.py` | `--facts plan/facts.json [--page-manifest assets/page/TEMPLATE.json] [--memory .claude/domain-forge-memory.md] [--dataset NAME ...] [--carry OLD-PLAN] [--bundle B] [--unattended] --out .work/story-plan.draft.json [--markers PATH] [--json]`. `PlanDrafter` writes `proposals` and every `authored` slot null (`status: draft`), pins the facts and the template (V1), and prints the forks left open. `--unattended` settles S2 and S3 with their unattended answers; `--carry` fills authored slots from a `vega/story-plan@1` or `@2`; `--dataset` adds a name the `show:` stance lines may use; `--bundle` names the bundle whose `netspec.json` gives the walk's inner block types. |
| `draft_io.py` | The draft's IO edge: the fact set, the template digests, the pinned `.aux` of a report without a role map (PF6), the block types, the markers file. |
| `memory_stances.py` | `MemoryStances.read(path, datasets)`: the `show:` lines of § Network stances for this dataset — a recorded stance is applied and cited, never re-asked. |
| `propose_lexicon.py` · `propose_motif.py` · `propose_roles.py` · `propose_families.py` · `propose_labels.py` | One owner per proposal (R-E9): the lexicon (the row noun from `MANIFEST.serve.identity`); the motif (the modality's base, fork S3); the class roles (the structural neutral, `alert` only with a basis, fork S2); the input groups (a partition of the shaped inputs, 3–9); the report roles proposed from a hand-authored report's `.aux` and the code-walk steps. |
| `carry.py` · `carry_v1.py` | `PlanCarrier`: an earlier plan's authored text carried into a new draft, every slot it cannot carry a `WARN:`; `V1Carry` translates the hand-authored v2.2 `vega/story-plan@1` (record refs become fact keys). |
| **3 · resolve** | |
| `resolve_story.py` | `--plan plan/story-plan.json --facts plan/facts.json [--page-manifest …] [--report R] --out story.json [--check] [--markers PATH] [--json]`. Validates the authored plan (V1–V15, each refusal with its rule id) and writes `vega/story@2`; `--check` resolves again and compares byte for byte (exit 3 when it differs). |
| `plan_rules.py` · `plan_roles.py` · `plan_texts.py` · `plan_claims.py` | `PlanValidator` and the block checks (lexicon, input groups, choices and focus); the class-role check (V6); the text rules over every authored string (V2 digit rule, V3 placeholders, V4 no record path, V12 lengths); the claims the facts contradict (V15). |
| `story_sections.py` · `story_report.py` | The story's values table and sections; the code notes' report binding through roles (emitted, or proposed and confirmed against the `.aux`), or brief links. |
| **4 · build and check** | |
| `build_showcase.py` | `--bundle B [--model M] [--plan P] [--story S] [--facts F] [--report R] [--label-roles J] [--out B/showcase] [--locale en|qps-ploc] [--unattended | --brief-links] [--template DIR] [--json]`, and `--check B/showcase [--report R]`. Copies the template byte for byte (BD1), writes `page.json` (BD2) and the attribution, copies the report or the briefs (fork S1: `--unattended` or `--brief-links`, else `FORK:` exit 2 before anything is written), pins everything in `SHOWCASE.json` (BD3–BD6), proves the bundle unchanged (BD8), and swaps `showcase.building/` → `showcase/` (the previous kept as `showcase.prev/`). `--locale qps-ploc` builds into `showcase/.work/qps/` only (BD7), never published. |
| `showcase_inputs.py` · `showcase_manifest.py` | What one build reads, read once and checked before anything is written (PF1, PF2, PF4–PF6, PF8, PF9, V1–V15 again); `SHOWCASE.json` (`vega/showcase@1`), the build options and fork S1, the file states BD7 and BD8 compare. |
| `attribution_build.py` | Run by the build in a clean child process inside the bundle: `predict.load_inspector(bundle).attribute(...)` over the demo strata of `examples/demo.json` (the training part, never the test part) → `data/attribution.json` (`vega/attribution@2`), the demo sample's row scores and the network view's lane. |
| `showcase_check.py` | `ShowcaseChecker` (CK1–CK10): pass (0), broken (2) or stale (3, a report, a brief or a re-emit moved), written to `showcase/.work/checks.json`. |
| **the page's configuration and template** | |
| `page_config.py` · `page_derive.py` · `page_check.py` | `build_page_config` assembles `page.json` (`vega/page@1`) from the chain's records — it proposes nothing — and `build_walk` computes the walk ids once; `Catalogs` and the derivations; `check_page_config` is gate BD2 (`page_check.py <page.json> [--network N]`). |
| `page_template.py` | The template's identity: `--write` / `--check` (PG1) / `--verify PAGE` (BD1) / `--digests` over `assets/page/TEMPLATE.json`, and the endpoint lists the page requires (PF5). |
| `page_lint.py` · `page_checks.py` · `page_words.py` · `js_tokens.py` · `js_shape.py` | The page template's prose gate and size discipline (PG2, PG4–PG7, PG9) over a small ES tokenizer; the vocabulary a literal may use outside the strings table. |
| **5 · served check, layer** | |
| `smoke_show.py` · `smoke_show_layer.py` | `--serve B --strict [--python …] [--port …]`: the bundle's own service started with the skills off the path, the slot verified, every served file and every report or brief link at its pinned digest, every view walked through the page's probe by the one page driver (`smoke_page.drive`/`judge`), no unresolved placeholder, the alert colour iff an alert role (SV1–SV5), `showcase/.smoke/smoke.json` (`vega/show-smoke@1`) with `start.png` and `results.png`; `<out> --strict`: the *Showcase* tab's nine checks (`smoke_show_layer.py`, D5 §9.2), a degraded copy included; `--mock CASE --showcase S`: a page fixture case served with a showcase's story and page config, walked and judged by the page driver. |
| `smoke_page.py` · `smoke_page_checks.py` · `cdp.py` | The one page driver: the showcase page opened in a headless Chrome over the DevTools protocol (`cdp.py`, standard library only), walked by its own probe and judged by its checks (P-PROBE … P-LINKS, P-AXIS among them); `--fixture CASE|all` serves the page fixture matrix, `--bundle B` the bundle's own service. `smoke_show.py`, the lanes' page stage and the tests drive the page through it. |
| `serve_probe.py` | `ServedPageProbe`: the served walk `smoke_show.py --serve` runs (gates SV1–SV5 beside the page driver's P-checks); a slot that is not verified is not walked. |
| `apply_show_layer.py` | `<model>.pack.html --data <bundle>/showcase/.work/showcase-layer.json --out <out>`: validates the layer data (LY1–LY7) and writes the `showcase` layer through `domain-forge/scripts/apply_layer.py`. |
| `show_layer_data.py` · `show_layer_story.py` | `ShowLayer`: the layer data (`vega/showcase-layer@1`) spliced from the showcase — `SHOWCASE.json`, the story summary with the headline and lead rendered as the page renders them, the bindings (every fact key, its value, record and uses), the report binding, the build check and the served walk read from `.work/checks.json` and `.smoke/smoke.json`, the markers, the usage — and the gates LY1–LY6. |

Assets beside this directory: `assets/page/` (the page template, `TEMPLATE.json` its identity),
`assets/showcase-render.js` and `assets/showcase-layer.css` (the layer's renderer, at most
40 KB). None is rewritten per run.

## The order, and why it is the order

```
extract → draft → (forks) → author → resolve → build → check → served walk → layer → validate → publish → brief
```

**Facts first, and only once.** Every number the page can show is read by one script from the
files the bundle pins, each checked against its `MANIFEST.files` digest before it is parsed. The
draft, the validator and the resolver read the fact set and nothing else, so a number cannot
enter the story by any other door — and a bundle altered after packing stops the stage before a
word is written.

**The draft proposes what the records settle.** The lexicon, the motif, the class roles, the
input groups, the choices, the runs, the limits and the walk are proposed by rule, each with its
source. What is left to the model is what only a writer can do: the words. The forks are the
three places where the records cannot decide and a person should — the report mode, an `alert`
colour, a motif — and each has an unattended answer that claims nothing (brief links, `plain`, the
modality's base).

**The resolver writes the story, not the model.** The plan is validated rule by rule (V1–V15),
and only a plan that passes becomes `story.json`, written once and pinned. A story edited by hand
is refused by the check and by the slot, because its digest no longer matches.

**The build proves the bundle untouched.** The showcase is built beside the bundle
(`showcase.building/`), pinned by its own `SHOWCASE.json` — the bundle's `MANIFEST.json` digest
included — and swapped into place only after the build has shown that no file of the bundle
changed. The previous showcase is kept, which is why nothing in this stage needs a `GATE:`.

**The served walk runs on the real service.** The page is only what the bundle's own service
serves through its slot, so the walk starts that service and reads the page as a visitor would;
a page that renders from files but not from the service is not a page anyone can open.

**The layer and the brief come last.** They describe a showcase that passed its check and its
walk; the brief is post-report, never a chapter, so the report stays stable when the page changes.

## Running it

```bash
S=<bundle>/showcase.building
python3 extract_facts.py --model <m>.pack.html --bundle <bundle> [--report docs/] --out $S/plan/facts.json
python3 draft_plan.py --facts $S/plan/facts.json --page-manifest ../assets/page/TEMPLATE.json \
        --memory .claude/domain-forge-memory.md [--unattended] --out $S/.work/story-plan.draft.json
cp $S/.work/story-plan.draft.json $S/plan/story-plan.json      # then author `authored` (SKILL.md step 4)
python3 resolve_story.py --plan $S/plan/story-plan.json --facts $S/plan/facts.json --out $S/story.json
python3 build_showcase.py --bundle <bundle> --model <m>.pack.html [--unattended | --brief-links] --out <bundle>/showcase
python3 build_showcase.py --check <bundle>/showcase
python3 smoke_show.py --serve <bundle> --strict
python3 apply_show_layer.py <m>.pack.html --data <bundle>/showcase/.work/showcase-layer.json --out <m>.pack.show.html
python3 smoke_show.py <m>.pack.show.html --strict
```

`tests/autoauthor.py --plan … --facts … --out …` is the mechanical author the tests and the lanes
use in place of the model; it is a test harness, never the stage's author.

## Tests

| file | what it proves |
|---|---|
| `tests/test_catalog_sync.py` (with `_sync_common.py`, `_sync_contract.py`, `_sync_catalogs.py`, `_sync_schemas.py`, `_sync_patch.py`) | CS1–CS3: the catalog equals the contract, agrees with the vega-forge and vega-eval catalogs and the backend's `METRICS`, every schema and its examples validate, and the document-project patch names every report role. |
| `tests/test_show_core.py` · `test_digit_rule.py` · `test_aux_adapter.py` | The value objects and the bundle adapter; the digit rule and the placeholder grammar; the `.aux` adapter and the role-map reader. |
| `tests/test_facts_cids.py` · `coverage_fixtures.py` | The extractor on the CIDS bundle (skipped when absent) and a fact-level check over every admitted (family, task, shape, goal) cell. |
| `tests/test_draft.py` · `test_carry.py` · `_draft_fixtures.py` | The draft and its proposers; the carry of `@1` and `@2` plans. |
| `tests/test_story_rules.py` · `autoauthor.py` | V1–V15, the resolved story, and the mechanical author. |
| `tests/test_build_showcase.py` | The build and the check: a byte-identical template copy, `SHOWCASE.json`, fork S1, the swap, the pseudo-locale build, broken and stale transitions, a re-emitted bundle. |
| `tests/test_show_layer.py` | The layer and its gates LY1–LY7 (D5 negatives 33–35, VS21), the story summary and bindings, the written page (validate_model 13–16, the tab's nine checks, a seeded render caught), the walk not run, the mock walk; with `VEGA_SHOW_LANES` the served walk, the layer and the tab on L01 and L02, and a story edited after the build refused by the walk (SV1). |
| `tests/test_page_config.py` · `test_page_template.py` · `test_page_size.py` · `test_page_math.py` · `test_code_room.py` · `_page_fixtures.py` | `page.json` and BD2; the template's identity and literal gates (PG1–PG8); its size discipline; the page's arithmetic against torch; the code room's walk; `_page_fixtures.py` builds `test_page_config.py`'s synthetic chain records (one netspec per view key and sample shape). |
| `tests/make_page_fixtures.py` · `page-matrix.json` · `page_fixture_story.py` · `serve_page_fixture.py` · `test_smoke_page.py` · `test_walk_anchors.py` · `probe_latent.py` | The page fixture matrix (35 tiny real networks of the backend, explained through the inspector, served under the real routes with no training); the driver's checks on cases with defects seeded; every walk step's code anchor resolving in a rendered package; the latent renderers on real bundles. |
| `tests/test_coverage_matrix.py` | Complete coverage through the stage: every admitted (family, task, sample shape, goal) cell of the vega-forge catalog through extract → draft → `autoauthor` → resolve → build → check, its `page.json` lane the cell's; with the page fixtures (`--fixtures DIR`, `$VEGA_SHOW_FIXTURES`), `smoke_show.py --mock` over every case with its own showcase, and every lane the cells build walked by a case. |
| `tests/run_negatives.py` (with `_negatives_kit.py`, `_negatives_plan.py`, `_negatives_bundle.py`, `_negatives_served.py`, `_negatives_static.py`) | `<matrix> [--json OUT]`: every gate of the contract's §10 against a seeded defect — D5's forty, D6's VS01–VS27, the slot's SS1–SS9 and D4's PN1–PN12 and gate C's GC-1–GC-4 in one table de-duplicated by gate id, each case named by the gate that must catch it; the table is held to `catalog.json.gates` first (a gate without a case fails the run). Seeds are planted on matrix lanes L02, L03 and L05, staged once through the stage and put back before each case, and on copies of the skill's own files (`vega/negatives-result@1`). |
| `tests/acceptance.sh` · `_acceptance_lanes.py` · `fixtures/report/` · `showcase-layer.authoring.json` | `[matrix-dir]`: the template and catalog gates and `conformance.py --size-only` over these scripts; lane A (L01: no report, then the report fixture, then the staleness transitions, the bundle untouched, its round trip still passing) and lane B (L02, the stub report), each through the served walk, the layer, invariants 13–16 and the tab; the page fixtures and their driver; the coverage matrix; `run_negatives.py`; the static verifier's recorded SHIP. |
| `tests/capture_parity_golden.py` · `cic_fixture.py` · `parity_server.py` · `parity_browser.py` · `check_pseudo_locale.py` | The page template's parity with the hand-built CIC page and its pseudo-locale check (`CIDS_ROOT` required). |
