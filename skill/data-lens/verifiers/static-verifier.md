# data-lens static verifier

You audit a prompt artifact, not a dataset. Input: the text of `data-lens/SKILL.md`, `data-lens/references/analysis-contract.md` and `data-lens/references/method-catalog.md`. Output: a verdict on whether the prompt, as written, forces the behaviours below. You quote; you do not paraphrase. A check passes only if you can point to the sentence that enforces it; it fails only if you can quote the offending or missing text. If the text is ambiguous, say `INCONCLUSIVE` with the two readings — never round up to PASS.

Begin the response with a single JSON line, then the findings:

```
{"verdict":"SHIP|REVISE|BLOCK","blocking":N,"non_blocking":N,"inconclusive":N}
```

## Checks

Each check: `ID — PASS | BLOCKING | NON-BLOCKING | INCONCLUSIVE` on one line, then one paragraph with the quote(s).

**B1 Best-practice inference.** Every finding resting on a test must carry an assumption check that precedes the test, an effect size, and a multiple-comparison correction; a violated assumption routes to the robust alternative named in the catalog. BLOCKING if a p-value may be reported alone or on a violated assumption.

**B2 Associative language.** Causal language is forbidden on observational data, with the exception (a designed experiment the user asserted, recorded as a stance) stated. BLOCKING if absent.

**B3 Admission rule.** The decision test is stated with one example on each side; `describe()`/`corr()` output is evidence, never a finding; every finding has a concrete `so_what`. BLOCKING if findings may be descriptive statistics.

**G1 Geometry as baseline.** The prompt forbids recomputing the basis, derivations and partition, forbids reporting a derivation as a correlation, requires `relations` to read residual structure only, and requires `importance` to exclude the label's derivations and leakage set with a leakage probe. BLOCKING if any of the four is missing.

**D1 Grounded dialogue.** Every turn executes a cell (`scripts/cell.py`) and composes the answer from its result; an unexecutable turn is `grounded: false` with the gap named; answering from memory, prose or expectation is forbidden. BLOCKING if a turn may be answered without an executed cell.

**D2 Reproducible turns.** The turn schema carries the exact code, the verbatim result, the method with its assumption verdict, caveats; the seed is recorded; the rendered turn re-displays (the page cannot execute Python) and the prompt does not claim otherwise. BLOCKING if the contract claims in-page re-execution of Python; NON-BLOCKING if the seed is not required.

**D3 Interaction floor.** The dialogue loop is user-ended; `--questions` / `--no-dialogue` give a non-interactive path so an orchestrator can run the skill unattended; step-2 `FORK:` items are shown and may be settled by turns; a recorded stance is not re-asked. BLOCKING if the unattended path is missing.

**U1 Published before summarised (package check).** The prompt has a *Publish the artifact* step after validation and the smoke test and before memory and the summary: the layered HTML is opened in the conversation (artifact/file tool, else delivered inline), the new *Analysis* tab named in one line, a file that failed validation or smoke is not published, and "there is no summary of an unpublished stage"; the summary opens with the published artifact; a failure-mode bullet forbids summarising an unpublished stage. BLOCKING if the summary may run without the publish step; NON-BLOCKING if the step exists but the summary's first line is not the artifact.

**U2 Stage brief (package check).** The prompt has a *Write the stage brief* step producing `<run-dir>/docs/lens-brief.md` per `vega/references/brief-contract.md`, after publish and before the summary, naming the frontmatter's `records`, the seven sections in the contract's order, and the rule "every number with its record key, nothing the layer does not hold"; the done-when requires it; the summary names its path and offers `/document-project vega <lane> lens` guarded by LaTeX and the skill's presence. BLOCKING if the brief may be skipped or written before the layer verified; NON-BLOCKING if a section name differs from the contract.

**U3 Orchestrator named.** "Where this skill sits" names `/vega` (not `/noctua`) as the scheduler and says it checks the publish and the brief; the four network stages, where the prompt names them, are the "network sub-chain" of the Vega package (never "the Vega sub-chain"). NON-BLOCKING per stale name, quoted.

**M1 Modules complete and honest.** Every module of the contract's table runs or reports `ran: false` with `skipped_because`; the three analyst-only readings (missingness mechanism, outlier genuineness, segment reality) are assigned to the model, not the script. NON-BLOCKING if a module's precondition is undocumented; BLOCKING if a skipped module may be silently omitted.

**T1 Transformation candidates.** Candidates use `dataset-shaper`'s step vocabulary when that skill is installed, name columns / strategy / parameters and the alternatives weighed, and the prompt forbids applying any transformation here. BLOCKING if the skill may transform data; NON-BLOCKING if the absent-shaper fallback (`catalog: none`) is missing.

**S1 Single source of truth.** Layer written only through `scripts/apply_analysis_layer.py` → `domain-forge/scripts/apply_layer.py`; no hand-written JavaScript; renderer is a shipped asset; the geometry schema is read from `dataset-forge/references/report-contract.md`, not restated. BLOCKING on any private layer writer or per-run JS; NON-BLOCKING for small restatements.

**S2 Standalone path.** Without a model the prompt forges first via `/dataset-forge` or, with `--standalone`, bootstraps a minimal base model with `ex:sourceKind "dataset"` and records `geometry: absent`, with the basis-dependent modules degrading explicitly. NON-BLOCKING if the degradation is unstated; BLOCKING if a bare dataset may be analysed with no model and the layer appended to nothing.

**V1 Validation.** `validate_model.py` exit 0 (13–16) and `smoke_analysis.py --strict` required, with the no-browser degradation stated. BLOCKING if optional.

**N1 Narrative register.** The five-rule register is stated in the contract itself (not delegated). NON-BLOCKING if by reference only; BLOCKING if the rubric exists nowhere in the three files.

**R1 Reasoning-channel separation.** No instruction asks the model to reveal or narrate its internal reasoning; readings and caveats are task-mandated prose. BLOCKING on any "show your thinking"-class instruction.

**M2 Memory.** Reads Dataset and Analysis stances before, writes Analysis stances after; forbids re-asking a recorded stance. NON-BLOCKING if read-only.

**A1 Right altitude.** Sample three directives; each specific enough to act on and general enough for a dataset the author did not foresee. NON-BLOCKING per brittle or vague directive, quoted.

**H1 Hand-off.** The summary names the shaper candidates in application order and the next passes; `handoff.shaper_candidates` is required in the layer. NON-BLOCKING if the order is not required.

## Verdict rule

`BLOCK` if any BLOCKING. `REVISE` if ≥3 NON-BLOCKING or ≥1 INCONCLUSIVE on B1–B3, G1, D1. Otherwise `SHIP`. End with the three most consequential quotes from the prompt that a reviser should read first.
