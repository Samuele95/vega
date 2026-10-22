
This bundle was built with `--include full`: it carries the whole emitted package, not only
the serving slice, and two more entry points.

```bash
python3 train.py --shaped path/to/shaped --out runs/mine     # fit this spec once
python3 evaluate.py --part valid --shaped path/to/shaped     # score a part
```

Neither is the stage it echoes. `train.py` runs one spec with no ladder, no diagnostics and
no ledger, so its checkpoint is a network no model card describes. `evaluate.py` prints point
metrics with no intervals, no baselines and no verdict. The test part stays sealed in both:
`load_part` refuses it, and the one door wants the campaign's `test_seal.json` and its token.
The shaped parts do not travel with the bundle — `recipe.json` is what makes new ones.
