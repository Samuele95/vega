# verifier-runs — the static verifiers' last runs, pinned to the prompts they read

Every skill of the package ships `verifiers/static-verifier.md`: a prompt that audits the
skill's own prompt (its `SKILL.md` and the contracts its *Input:* line names) and begins its
answer with one JSON line, `{"verdict":"SHIP|REVISE|BLOCK","blocking":N,"non_blocking":N,"inconclusive":N}`.
The verifier is LLM work: a subagent in a fresh context reads the verifier and the prompt text
and answers. This directory records the last such run of each skill, so a script can tell
whether the prompt a release ships is the prompt a verifier passed.

## One file per skill: `<skill>.json` (`vega/verifier-run@1`)

```json
{ "schema": "vega/verifier-run@1", "skill": "vega-show", "verdict": "SHIP",
  "blocking": 0, "non_blocking": 2, "inconclusive": 0,
  "audited": { "SKILL.md": "sha256:…", "references/show-contract.md": "sha256:…",
               "verifiers/static-verifier.md": "sha256:…" },
  "run_at": "2026-10-02T09:00:00Z", "session": "<id>" }
```

`audited` maps each file the verifier read, relative to the skill's directory, to its sha256 at
the time of the run. It holds at least `SKILL.md` and every file the verifier's *Input:* line
names; the verifier itself is recorded too, so a check added to it later also asks for a new run.

## Writing a record

Run the verifier on the skill in a fresh context, save its answer, then:

```bash
python3 vega/scripts/tests/verifier_gate.py record --skill vega-show --session <session-id> \
        --response <answer.md>
# or, without the answer file:
python3 vega/scripts/tests/verifier_gate.py record --skill vega-show --session <session-id> \
        --verdict SHIP --blocking 0 --non-blocking 2
```

`record` reads the verdict from the answer's first JSON line (or the flags), digests `SKILL.md`,
the verifier and its *Input:* files (or the paths given with `--audited`), and writes
`<skill>.json` here. A record is written only for a run that happened: a verdict typed without a
verifier run is the one thing this directory exists to prevent.

## Checking

```bash
python3 vega/scripts/tests/verifier_gate.py              # every skill of the package
python3 vega/scripts/tests/verifier_gate.py --skill vega-show
python3 vega/scripts/tests/verifier_gate.py --release    # also release/rehearsal.json (T5)
```

For each skill (vega-forge, vega-train, vega-eval, vega-pack, vega-show, vega, dataset-forge,
data-lens, dataset-shaper) the gate fails when:

- `verifiers/static-verifier.md` is missing, or lacks the package checks — for a stage,
  *Published before summarised*, *Stage brief* and *Orchestrator named* (U1–U3; vega-eval numbers
  them U4–U6); for the orchestrator `vega`, *Published and briefed*;
- `<skill>.json` is missing, is not a `vega/verifier-run@1` run of that skill, or its verdict is
  not `SHIP` with zero blocking findings;
- the run did not audit `SKILL.md` or a file the *Input:* line names;
- an audited file's sha256 differs from the file today: the prompt changed after the verifier
  read it, so the verdict no longer describes it. Re-run the verifier and record again.

With `--release` it also requires `release/rehearsal.json` (`vega/rehearsal@1`, the T5 LLM
rehearsal of D6 §11.4, written by the release run): a `session` and a non-empty `checks` list
whose every entry has `verdict: "pass"`:

```json
{ "schema": "vega/rehearsal@1", "session": "<id>", "run_at": "…",
  "checks": [ { "id": "check_brief", "verdict": "pass" }, { "id": "page_smoke", "verdict": "pass" } ] }
```

## Owners

The gate and this README are T04's. The records are written by the packages that run the
verifiers: `vega-pack`, `vega-forge`, `vega-train`, `vega-eval` and `dataset-shaper` by R1,
`vega-show` by H08 then H09, `vega` by H10 then R3. `dataset-forge.json` and `data-lens.json` have
no owner in the build plan's ledger yet (the gate covers them, as D6 §15.7 lists them). Until a
skill's record exists, `verifier_gate.py` fails on it by design: `matrix.sh static` is red on the
verifier gate until release (D6 §16.4, checklist item 3).
