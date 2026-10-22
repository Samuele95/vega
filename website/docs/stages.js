// Vega docs content. A new stage page needs only an entry in `pages` (and a one-line page file that imports the template).
// Every string here comes from content/stages.md.
window.VEGA_DOCS = {
  order: [
    { id: 'vega', cmd: '/vega', group: 'Orchestrator' },
    { id: 'dataset-forge', cmd: '/dataset-forge', group: 'The data lane' },
    { id: 'data-lens', cmd: '/data-lens', group: 'The data lane' },
    { id: 'dataset-shaper', cmd: '/dataset-shaper', group: 'The data lane' },
    { id: 'blueprint', cmd: '/blueprint --mode pipeline', group: 'Architecture' },
    { id: 'vega-forge', cmd: '/vega-forge', group: 'The network sub-chain' },
    { id: 'vega-train', cmd: '/vega-train', group: 'The network sub-chain' },
    { id: 'vega-eval', cmd: '/vega-eval', group: 'The network sub-chain' },
    { id: 'vega-pack', cmd: '/vega-pack', group: 'The network sub-chain' },
    { id: 'document', cmd: '/document-project vega <lane>', group: 'Telling it' },
    { id: 'vega-show', cmd: '/vega-show', group: 'Telling it' }
  ],
  pages: {
    '_template': {
      title: 'stage-name',
      tagline: 'A one-line promise for the stage.',
      cmd: '/stage-name',
      role: 'Role. Who the stage is, in one sentence.',
      ruleLead: 'The rule it never bends, in one sentence.',
      ruleBody: 'Optional: why the rule holds, in two or three sentences.',
      sits: 'Where it sits: after which stage, before which.',
      takes: 'What it takes: the products of earlier stages it reads.',
      flags: [
        { f: '--flag <value>', d: 'What the flag does.' },
        { f: '--unattended', d: 'What the stage does when no user is present.' }
      ],
      produces: ['Each product the stage writes, one per line.', 'The layer it publishes.'],
      done: ['Each condition that must hold before the stage is done.', 'The layer is published and the brief written.']
    },
    'vega-eval': {
      title: 'vega-eval',
      scene: 'seal',
      tagline: 'One look, with intervals.',
      cmd: '/vega-eval',
      role: 'The evaluator who gets one look at the test part and makes it count.',
      ruleLead: 'The test is opened once, under a seal, and every number that comes out of it travels with an interval and a baseline.',
      ruleBody: 'A metric without a confidence interval is a guess about n rows; a metric without the strongest baseline beside it is a number without a meaning; a threshold picked on the test is a leak; a second look is disclosed, never hidden.',
      sits: 'After vega-train (the selected run); before vega-pack.',
      takes: "The training layer, the campaign directory with the selected run's checkpoint, and the sealed test part.",
      flags: [
        { f: '--run <id>', d: 'Evaluate a run other than the selection (recorded as a user decision).' },
        { f: '--again "<reason>"', d: 'A further exposure of the test; refused without a reason once the seal exists.' },
        { f: '--baselines trivial,linear,tree,lens,campaign-baseline', d: 'The honest baselines (every requested one is attempted; one that cannot be fitted is reported, never dropped).' },
        { f: '--unattended', d: 'No user present: the seal gate is passed and stamped, defaults applied.' }
      ],
      produces: [
        'The test seal (the exposure recorded)',
        'Stored predictions',
        "The task's full metric set with bootstrap confidence intervals",
        'Paired differences against each baseline',
        'A decision threshold chosen on the validation part',
        'Calibration, slices, robustness readings',
        'A verdict in a fixed form (earned / not earned / beaten by a baseline / not assessable)',
        'An executed notebook that recomputes every metric from the stored predictions',
        'An evaluation layer'
      ],
      done: [
        'Every metric is reported for the network and every fitted baseline with intervals',
        'The threshold comes from validation',
        'The verdict agrees with the machine verdict',
        "The notebook's assertions pass",
        'The layer is published and the stage brief written'
      ]
    },
    'vega-show': {
      title: 'vega-show',
      tagline: 'Show it, honestly.',
      cmd: '/vega-show',
      role: 'The curator of one exhibit: the shipped network, shown to people who were not in the chain.',
      ruleLead: 'The page says nothing the chain did not measure, and touches nothing the chain shipped.',
      ruleBody: "Every number is a fact a script extracted from the bundle's pinned records; the story text may contain no digit of its own. The bundle is read, verified and served — never written.",
      sits: 'After vega-pack, and after the report when one is written.',
      takes: 'The packed model and its bundle; optionally the compiled report.',
      flags: [
        { f: '--report <docs dir | main.pdf>', d: 'Link each code note into the written report.' },
        { f: '--plan <edited story-plan.json>', d: 'Re-run for wording only: no new facts, the plan is re-validated and rebuilt.' },
        { f: '--unattended', d: 'Defaults taken and marked (with no report, notes link to the stage briefs).' }
      ],
      produces: [
        "A showcase directory the bundle's service verifies and serves beside its API",
        'The guided page: the problem and the data, the network with its weights and decisions, what was tried, one row through the network, many rows, results against baselines, the code with report notes',
        'Its story',
        'The served-walk check',
        'A showcase layer'
      ],
      done: [
        'The story plan passes its rules',
        'Every served file is pinned',
        'The served walk visits every view with zero page errors and zero unresolved placeholders',
        'The bundle is unchanged',
        'The layer is published and the brief written'
      ]
    }
  }
};
