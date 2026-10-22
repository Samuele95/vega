"""
The `papers` generator (lanes L20 and L23, D6 §4): a citation graph, one row a paper.

    frame = make(seed, rows)        # `rows` papers in publication order, deterministic in (seed, rows)

Every paper has one of five topics and cites one earlier paper (`cites` names that paper's
`paper_key`; the earliest papers cite none). The citation is **homophilous at 0.8**: four
citations in five stay within the citing paper's topic, the fifth goes to another topic — the
relation the lanes' `graph_edges` step records as a symmetric key relation on
`[paper_key, cites]`. L20 reads it inductively; L23 reads the same data (seed 2) under a
transductive protocol the user chose (`user:` source), and stops at `eval` (R-A10).

The planted signal: twelve bag-of-terms shares (`term_01` … `term_12`, one paper's shares sum to
1) of which each topic favours two, with enough noise that a paper's own terms name its topic
only part of the time; its neighbours' terms, read along the citations, name it far more
often — the reason a graph network has something to learn that a structure-blind one does not.
`n_authors` and `abstract_len` carry a weak topic effect each (systems papers have more
authors, theory abstracts run longer); `year` carries none.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..recipes import standard_recipe

__all__ = ["make", "SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "TOPICS", "HOMOPHILY", "TERMS"]

TOPICS = ("theory", "systems", "learning", "vision", "language")
TOPIC_SHARE = (0.18, 0.20, 0.26, 0.18, 0.18)
HOMOPHILY = 0.8
TERMS = tuple(f"term_{i:02d}" for i in range(1, 13))
FIRST_CITING = 12            # the papers before this one cite nothing: too little came before them
RECENT = 150                 # a citation goes to one of the most recent candidates

SCHEMA = (
    ColumnSpec("paper_key", "str", "key", comment="The paper's identifier."),
    ColumnSpec("cites", "str", "key", comment="The key of the one earlier paper this paper cites; empty when none."),
    ColumnSpec("year", "int", "numeric", "year", "The year the paper was published."),
    ColumnSpec("n_authors", "int", "numeric", None, "How many authors signed the paper."),
    ColumnSpec("abstract_len", "int", "numeric", "words", "The length of the paper's abstract."),
    *(ColumnSpec(t, "float", "numeric", None, f"The paper's share of bag-of-terms group {t[-2:]}.")
      for t in TERMS),
    ColumnSpec("topic", "str", "label", comment="The paper's topic, one of five."),
)

# The words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6). The column
# `year` is one of foreign-literals.json's generic words, which are never a lane's.
VOCAB = frozenset({"paper_key", "n_authors", "abstract_len", *TERMS, "topic", "topics", "theory",
                   "vision", "papers", "citation", "citations"})
# D6 §4's words VOCAB withholds for now. The shared templates use each in a sense of their own:
# `cites` (a column: a recipe step, spec choice or rung that cites its decision, a payload that
# cites a tensor), `paper` (the entity and row noun: a model card's printed page), `systems` (a
# class: the code-view fixtures' "record systems"), `learning` (a class: the learning rate) and
# `language` (a class: the code room's programming language, the page's locale). lint-allow.json
# has no sense-scoped entry for them yet, so in VOCAB they would fail the literal gate there.
# They join VOCAB once lint-allow.json holds test_lanekit_graph_generators.SENSE_ALLOW (a need of
# T04, its owner), unless the narrower VOCAB is ratified instead; that test checks the entries
# cover every line that uses them, and that no word is withheld that no template uses.
VOCAB_WITHHELD = frozenset({"cites", "paper", "systems", "learning", "language"})

ENTITY_COMMENT = "One paper of a citation index: its metadata, its bag-of-terms shares and the paper it cites."
COMPONENT = "bibliography"


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe over this SCHEMA: the keys dropped (the edges read them raw), every
    number scaled, a stratified split on `topic`, and `graph_edges` key_pair on
    `[paper_key, cites]`, symmetric, citing the relation finding (lanekit.recipes). The
    protocol (inductive for L20, transductive for L23) is the network stage's decision."""
    return standard_recipe(SCHEMA, raw_path, shaped_dir, lane)


def _citations(rng, topic):
    """The row each paper cites (-1 for none): an earlier paper of its own topic with
    probability HOMOPHILY, else an earlier paper of another topic; a recent one either way."""
    import numpy as np

    rows = len(topic)
    cited = np.full(rows, -1, dtype=np.int64)
    for i in range(FIRST_CITING, rows):
        same = np.flatnonzero(topic[:i] == topic[i])
        other = np.flatnonzero(topic[:i] != topic[i])
        pool = same if (rng.random() < HOMOPHILY and len(same)) or not len(other) else other
        cited[i] = int(rng.choice(pool[-RECENT:]))
    return cited


def _terms(rng, topic):
    """(rows, 12) bag-of-terms shares: gamma noise with each topic's two home terms raised."""
    import numpy as np

    raw = rng.gamma(1.0, 1.0, (len(topic), len(TERMS)))
    raw[np.arange(len(topic)), 2 * topic] += 1.1
    raw[np.arange(len(topic)), 2 * topic + 1] += 0.9
    return raw / raw.sum(axis=1, keepdims=True)


def make(seed: int, rows: int):
    """`rows` papers (module docstring), in publication order, deterministic in (seed, rows)."""
    import numpy as np
    import pandas as pd

    if rows <= FIRST_CITING:
        raise ValueError(f"papers needs more than {FIRST_CITING} papers for any to cite another, got {rows}")
    rng = np.random.default_rng(seed)
    topic = rng.choice(len(TOPICS), rows, p=TOPIC_SHARE)
    year = np.sort(rng.integers(1995, 2026, rows))
    cited = _citations(rng, topic)
    authors = np.maximum(1, rng.poisson(3.0, rows) + 1 + np.where(topic == 1, 2, 0) - np.where(topic == 0, 1, 0))
    abstract = np.round(rng.normal(170.0, 35.0, rows) + np.where(topic == 0, 30.0, 0.0)
                        - np.where(topic == 3, 15.0, 0.0)).astype(np.int64)
    keys = np.array([f"P{i + 1:05d}" for i in range(rows)], dtype=object)
    frame = pd.DataFrame({"paper_key": keys, "cites": np.where(cited < 0, "", keys[np.maximum(cited, 0)]),
                          "year": year, "n_authors": authors, "abstract_len": np.maximum(abstract, 40)})
    for name, column in zip(TERMS, _terms(rng, topic).T):
        frame[name] = np.round(column, 5)
    frame["topic"] = np.array(TOPICS, dtype=object)[topic]
    return frame[[c.name for c in SCHEMA]]
