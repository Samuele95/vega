# {slug}

The network `{run}` of campaign `{campaign}`, packaged to run on its own.

```bash
pip install -r requirements.txt
python3 predict.py --input rows.csv --output preds.csv   # raw rows in, predictions out
python3 predict.py --check                               # it still reproduces itself
```

`rows.csv` holds the columns `schema.json` lists as input. `model-card.html` is the whole
story: what this is, what it was measured at, and what it must not be used for.
{include_lines}
**{verdict_line}**
