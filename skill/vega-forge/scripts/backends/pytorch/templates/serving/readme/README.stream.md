

## The stream consumer

The blueprint serves this bundle as a stream: `serve/consumer.py` reads raw rows as JSON lines,
answers them batch by batch through the same predictor the CLI uses, and writes one JSON line
per batch — the answer envelope of the bundle's contract, with the monitor's readings when the
bundle carries a monitor reference. It keeps the last `MANIFEST.sample.history_rows` rows of each
series as the next batch's history, so a window completes across batches.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # if torch is not installed
pip install -r requirements-serve.txt
python -m serve.consumer --bundle . < rows.jsonl > answers.jsonl
```

Before the first row is read it checks every file against `MANIFEST.json` and answers the
shipped examples again; a bundle that does not reproduce them does not consume. Its source is
standard input: replace `stdin_source()` in `serve/consumer.py`'s copy of the template with the
queue client of the deployment, or wire the process's standard streams to it. With Docker:

```bash
docker build -t {dist} .
docker run --rm -i {dist} < rows.jsonl > answers.jsonl
```

A row that breaks the input schema is not answered: its batch's problem line names every
offending column, and the batch's other rows are answered. The bundle's variables are prefixed
`{env}_` (`{env}_BUNDLE` names the bundle directory).
