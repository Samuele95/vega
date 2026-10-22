"""
wine — the laboratory analysis of red and white wines and the grade a tasting panel gave them
(lane L04: a linear network with an ordinal head, the lane whose declared verdict is
`beaten_by_baseline`).

    frame = make(seed, rows)        # eleven physico-chemical measurements, color, quality

`quality` is the panel's grade, 3 to 8. The grade is a latent score cut at fixed shares of the
rows (few 3s and 8s, most 5s and 6s), and the score is on purpose mostly NOT linear in the
measurements — the lane exists to show a verdict the network does not earn, against the
gradient-boosted trees its evaluation fits on the same features:

- alcohol raises the score (the one linear term a linear model can read);
- sulphates help up to a point and hurt beyond it: the score falls with the squared distance
  from the middle of their range;
- residual sugar pleases in a white wine and spoils a red one (an interaction with `color`);
- a volatile acidity past 0.65 g/L tastes of vinegar: a step down, not a slope;
- too little free sulfur dioxide lets the wine oxidise: another step.

The measurements follow the published ranges of red and white wines (acidity, sugar, sulfur,
alcohol), and density follows alcohol and sugar with its own noise, so no column derives
another exactly. Deterministic in (seed, rows).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..recipes import standard_recipe

__all__ = ["SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "GRADES", "make"]

GRADES = (3, 4, 5, 6, 7, 8)
# the share of the rows each grade holds, lowest first (the cut points of the latent score)
GRADE_SHARES = (0.04, 0.08, 0.32, 0.36, 0.15, 0.05)
RED_SHARE = 0.3
# per colour: (mean, sd) of each measurement drawn from a normal; the lognormal ones are below
NORMAL = {"fixed_acidity": ((8.3, 1.6), (6.9, 0.8)), "citric_acid": ((0.27, 0.19), (0.33, 0.12)),
          "ph": ((3.31, 0.15), (3.19, 0.15)), "sulphates": ((0.66, 0.16), (0.49, 0.11)),
          "alcohol": ((10.4, 1.05), (10.5, 1.2))}
LOGNORMAL = {"volatile_acidity": ((0.52, 0.30), (0.28, 0.32)), "residual_sugar": ((2.5, 0.45), (5.5, 0.75)),
             "chlorides": ((0.085, 0.35), (0.045, 0.35)), "free_sulfur_dioxide": ((14.0, 0.55), (34.0, 0.45))}

SCHEMA = (
    ColumnSpec("fixed_acidity", "float", "numeric", "g/L", "Tartaric acid, the acidity that does not evaporate."),
    ColumnSpec("volatile_acidity", "float", "numeric", "g/L", "Acetic acid, the acidity that smells of vinegar."),
    ColumnSpec("citric_acid", "float", "numeric", "g/L", "Citric acid."),
    ColumnSpec("residual_sugar", "float", "numeric", "g/L", "The sugar left after fermentation."),
    ColumnSpec("chlorides", "float", "numeric", "g/L", "Sodium chloride, the salt in the wine."),
    ColumnSpec("free_sulfur_dioxide", "float", "numeric", "mg/L", "Free sulfur dioxide, the part that protects the wine."),
    ColumnSpec("total_sulfur_dioxide", "float", "numeric", "mg/L", "Free and bound sulfur dioxide together."),
    ColumnSpec("density", "float", "numeric", "g/cm3", "The wine's density."),
    ColumnSpec("ph", "float", "numeric", None, "The wine's pH."),
    ColumnSpec("sulphates", "float", "numeric", "g/L", "Potassium sulphate."),
    ColumnSpec("alcohol", "float", "numeric", "% vol", "The alcohol content by volume."),
    ColumnSpec("color", "str", "nominal", comment="Whether the wine is red or white."),
    ColumnSpec("quality", "int", "label", comment="The grade a tasting panel gave the wine, 3 to 8."),
)

# the words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6); the grades
# 3 to 8 are digits, which no leak check reads
VOCAB = frozenset({"wine", "wines", "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                   "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "sulphates",
                   "alcohol", "acidity", "sulfur", "tasting"})
# D6 §4's words VOCAB leaves out. A shared template uses these in a sense of its own, so in VOCAB
# each would fail the literal gate there: `color` (CSS, 396 lines), `red` and `white` (CSS and the
# page fixtures' multilabel classes), `grade` (the page fixtures' ordinal column; the tank lane
# withholds it too, and it names the label here). Each joins VOCAB with a sense-scoped
# lint-allow.json entry per line (T04's file). `quality` hits no template, but data-lens names its
# data-quality module `quality` on every lane's analysis layer: in VOCAB it would be a
# cross-lane leak on every other lane's page (D6 §10.6), so it stays out.
VOCAB_WITHHELD = frozenset({"color", "red", "white", "grade", "quality"})
COMPONENT = "cellar"
ENTITY_COMMENT = "One wine as the laboratory measured it and the tasting panel graded it."
OVERVIEW = """# wine

A wine is analysed in a laboratory and graded by a tasting panel, from 3 to 8. `quality` is
the attribute the chain partitions on; every other property is a laboratory measurement taken
before the tasting, plus the wine's colour.

The domain has one class, Wine. The grades are ordered, so the question is an ordinal one:
a 7 mistaken for a 6 is a smaller error than a 7 mistaken for a 3.

The panel's taste is not a straight line through the measurements: sulphates help up to a
point, sugar pleases in a white and spoils a red, a vinegar note is a step down. The chain
states that a linear network reads only the straight part of it.
"""


def make(seed: int, rows: int):
    """`rows` wines, deterministic in (seed, rows): the colour, the measurements, the grade."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = int(rows)
    red = rng.uniform(size=n) < RED_SHARE
    cols: dict[str, Any] = {}
    for name, ((mr, sr), (mw, sw)) in NORMAL.items():
        cols[name] = rng.normal(np.where(red, mr, mw), np.where(red, sr, sw))
    for name, ((mr, sr), (mw, sw)) in LOGNORMAL.items():
        cols[name] = rng.lognormal(np.log(np.where(red, mr, mw)), np.where(red, sr, sw))
    cols["citric_acid"] = np.clip(cols["citric_acid"], 0.0, 1.2)
    cols["residual_sugar"] = np.clip(cols["residual_sugar"], 0.6, 65.0)
    cols["alcohol"] = np.clip(cols["alcohol"], 8.0, 14.9)
    bound = rng.lognormal(np.log(np.where(red, 30.0, 100.0)), 0.4)
    cols["total_sulfur_dioxide"] = cols["free_sulfur_dioxide"] + bound
    cols["density"] = (0.9920 - 0.0011 * (cols["alcohol"] - 10.5) + 0.00037 * cols["residual_sugar"]
                       + 0.0025 * red + rng.normal(0.0, 0.0009, n))
    score = _score(rng, cols, red, n)
    cuts = np.quantile(score, np.cumsum(GRADE_SHARES)[:-1])
    quality = np.array(GRADES)[np.searchsorted(cuts, score)]
    frame = pd.DataFrame({name: np.round(cols[name], 5 if name == "density" else 3)
                          for name in [c.name for c in SCHEMA if c.role == "numeric"]})
    frame["color"] = np.where(red, "red", "white")
    frame["quality"] = quality.astype(int)
    return frame[[c.name for c in SCHEMA]]


def _score(rng: Any, cols: dict, red: Any, n: int):
    """The panel's latent score (module docstring): one linear term, a hump, an interaction,
    two steps, and the panel's own noise."""
    import numpy as np

    alcohol = (cols["alcohol"] - 10.5) / 1.15
    sulphates = (cols["sulphates"] - 0.55) / 0.13
    sugar = (np.log(cols["residual_sugar"]) - 1.3) / 0.6
    return (0.9 * alcohol - 0.9 * sulphates ** 2 + 0.9 * sugar * np.where(red, -1.0, 1.0)
            - 1.4 * (cols["volatile_acidity"] > 0.65) - 0.8 * (cols["free_sulfur_dioxide"] < 10.0)
            + rng.normal(0.0, 0.35, n))


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe (lanekit.recipes) — `quality` selected, a stratified split, `color`
    encoded ordinal (the network embeds it), the measurements scaled on train — with the grade
    retyped, in phase 1, to an ordered category in the lane's class order.

    The grade is spelled in digits but it is not a quantity, and it is not a class index either
    (a 3 is the lowest of six grades, not the fourth class): the geometry types it `ordinal`
    (lanekit.layers.final_type), and the retype makes the shaped label hold the grade names the
    lane's `task.classes` lists, so every reader maps a grade to its position by name."""
    recipe = standard_recipe(SCHEMA, raw_path, shaped_dir, lane)
    raw = lane.raw if hasattr(lane, "raw") else lane
    target = raw.get("target") or {}
    if target.get("column") != "quality":
        return recipe
    order = [str(c) for c in target.get("classes") or GRADES]
    retype = {"id": "", "op": "retype", "columns": ["quality"], "params": {"to": "ordinal", "order": order},
              "source": "geometry:typing/quality",
              "rationale": "the grade is an ordered category spelled in digits: an ordinal type in the "
                           "lane's grade order, so no reader takes a grade for a quantity or for an index",
              "consequences": {"rows": "unchanged", "columns": "unchanged (retyped)",
                               "downstream": "the label holds the grade names task.classes lists, in order"}}
    steps = recipe["steps"]
    at = next((i for i, s in enumerate(steps) if s["op"] not in ("retype",)), len(steps))
    steps.insert(at, retype)
    for number, step in enumerate(steps, start=1):
        step["id"] = f"S{number:02d}"
    return recipe
