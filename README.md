# Vega

The reference star for neural network engineering and dataset analysis.
A raw dataset in; a trained, evaluated, shipped network out.

**Site:** https://samuele95.github.io/vega/

## What's here

| Folder | Contents |
|---|---|
| `skill/` | The Vega skills, release v2.3.0: the `/vega` orchestrator, eight stage skills and `patches/` for three external skills |
| `website/` | The landing site: hand-built canvas, WebGL, CSS and synthesized sound. Deployed to GitHub Pages by `.github/workflows/pages.yml` |

## The chain

`/vega` drives the chain. It runs each stage in order, checks every product and keeps the ledger. Attended, it asks at real decisions; unattended, it takes the documented default and marks it.

- **The data lane**
  - `dataset-forge` maps the data's shape.
  - `data-lens` keeps only findings that change a decision.
  - `dataset-shaper` turns decisions into a traced, verified recipe.
- **The network**
  - `blueprint` sets the architecture.
  - `vega-forge` writes the spec, every choice cited.
  - `vega-train` changes one thing at a time, each change with a hypothesis.
  - `vega-eval` opens the sealed test once.
  - `vega-pack` ships a bundle proven by a round trip.
- **Telling it**
  - `document-project` writes the LaTeX report (optional).
  - `vega-show` builds a showcase that says only what was measured.

## Install

Five steps to the first run.

1. Replace the installed skill folders with the nine in `skill/`:
   ```
   vega/  dataset-forge/  data-lens/  dataset-shaper/  vega-forge/  vega-train/  vega-eval/  vega-pack/  vega-show/
   ```
2. Apply `skill/patches/` to blueprint, document-project and domain-forge.
3. Install the external skills: domain-forge, spec-analysis, inferred-questions, model-chat, blueprint, document-project, architect, improve, effective-java.
4. Check the environment:
   ```
   python3 vega/scripts/env_check.py --env
   ```
5. Point the orchestrator at your dataset:
   ```
   /vega <your dataset>
   ```

## The website locally

The pages load React and Babel from unpkg, so they need a network connection. To serve the site from a local folder:

```
cd website && python3 -m http.server 8173
```

Then open `http://localhost:8173/site/index.dc.html`.

## License

MIT — see `LICENSE`.
