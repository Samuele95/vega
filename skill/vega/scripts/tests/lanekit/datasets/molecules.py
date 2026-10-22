"""
The `molecules` generator (lane L22, D6 §4): small molecules, one row an atom, one graph a
molecule.

    frame = make(seed, rows)        # `rows` MOLECULES (8-16 atoms each), deterministic in (seed, rows)

At graph level the lane's `rows` count graphs (lanes.json `rows_unit: graph`), so `make` returns
`rows` molecules and about twelve times as many atom rows. Each molecule is a spanning tree of
bonds: its first atom is a carbon, and every later atom bonds to an earlier atom of the same
molecule that still has a free valence (C 4, N 3, O 2, S 2, Cl 1). `bonded_to` names that
atom's `atom_key` (empty on the first atom); `molecule_id` is the graph id. The lane's
`graph_edges` step reads `[atom_key, bonded_to]` as a symmetric key relation inside each
molecule, and its split is by group on `molecule_id` — a molecule never straddles two parts,
so no bond crosses a part.

The planted signal: `soluble` (one value per molecule, repeated on its atoms) reads the polar
atoms **on the tree** — a terminal N or O (one bond: a solvent-exposed group) counts more than
a buried one — plus the charged atoms, against the hydrophobic carbons, aromatic atoms, sulphur
and chlorine; about half the molecules are soluble. A sum over atoms of what each atom and its
bonds say is exactly what a GIN with a sum readout computes. `degree` is the atom's bond count
in the tree; `charge` is formal (N may carry +1, O -1); `aromatic` is 0 or 1 (C and N atoms of a
molecule with an aromatic system).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import ColumnSpec
from ..recipes import standard_recipe

__all__ = ["make", "SCHEMA", "RECIPE", "VOCAB", "VOCAB_WITHHELD", "ELEMENTS", "VALENCE",
           "ATOMS_PER_MOLECULE"]

ELEMENTS = ("C", "N", "O", "S", "Cl")
VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "Cl": 1}
ATOMS_PER_MOLECULE = (8, 16)

SCHEMA = (
    ColumnSpec("atom_key", "str", "key", comment="The atom's identifier, unique across molecules."),
    ColumnSpec("bonded_to", "str", "key",
               comment="The key of the earlier atom this atom is bonded to; empty on a molecule's first atom."),
    ColumnSpec("molecule_id", "str", "group", comment="The molecule the atom belongs to."),
    ColumnSpec("element", "str", "nominal", comment="The atom's element: C, N, O, S or Cl."),
    ColumnSpec("charge", "int", "numeric", "e", "The atom's formal charge."),
    ColumnSpec("aromatic", "int", "numeric", None, "1 when the atom is part of an aromatic system, else 0."),
    ColumnSpec("degree", "int", "numeric", None, "The number of bonds the atom takes part in."),
    ColumnSpec("soluble", "int", "label", comment="1 when the atom's molecule is water-soluble, else 0."),
)

# The words this dataset brings to a page (D6 §4: column names, class names, entity, row noun),
# read by the literal lint and the page leak check (lint_templates.py, D6 §10.6). The classes `0`
# and `1` and the element symbols are too short to be tokens.
VOCAB = frozenset({"atom", "atoms", "atom_key", "bonded_to", "molecule_id", "charge", "aromatic",
                   "soluble", "molecule", "molecules", "bond", "bonds", "solubility"})
# D6 §4's words VOCAB withholds for now. The shared templates use each in a sense of their own:
# `element` (a column: a DOM or SVG element, a tensor's or a tuple's element, a per-element metric)
# and `degree` (a column: a node's degree in the graph template's neighbour sampler).
# lint-allow.json has no sense-scoped entry for them yet, so in VOCAB they would fail the literal
# gate there.
# They join VOCAB once lint-allow.json holds test_lanekit_graph_generators.SENSE_ALLOW (a need of
# T04, its owner), unless the narrower VOCAB is ratified instead; that test checks the entries
# cover every line that uses them, and that no word is withheld that no template uses.
VOCAB_WITHHELD = frozenset({"element", "degree"})

ENTITY_COMMENT = "One atom of a small molecule: its element, charge, bonds and the molecule it belongs to."
COMPONENT = "chemistry"


def RECIPE(raw_path: Path, shaped_dir: Path, lane: Any) -> dict:  # noqa: N802 (the protocol's name)
    """The standard recipe over this SCHEMA: the keys dropped (the edges read them raw),
    `element` encoded, the numbers scaled, a group split on `molecule_id`, and `graph_edges`
    key_pair on `[atom_key, bonded_to]` with `graph_id: molecule_id` (lanekit.recipes)."""
    return standard_recipe(SCHEMA, raw_path, shaped_dir, lane)


def _molecule(rng, size: int):
    """(elements, parent index or -1, degree) of one molecule's spanning tree."""
    polar = rng.beta(2.0, 4.0)
    weights = [max(0.1, 1.0 - polar - 0.12), polar * 0.45, polar * 0.55, 0.05, 0.07]
    elements, parent, degree = ["C"], [-1], [0]
    for _ in range(1, size):
        element = ELEMENTS[int(rng.choice(len(ELEMENTS), p=[w / sum(weights) for w in weights]))]
        free = [k for k, e in enumerate(elements) if degree[k] < VALENCE[e]]
        if element == "Cl" and sum(VALENCE[elements[k]] - degree[k] for k in free) <= 1:
            element = "C"                      # a chlorine would close the last free valence
        p = free[-1 - min(int(rng.exponential(2.0)), len(free) - 1)]
        elements.append(element)
        parent.append(p)
        degree.append(1)
        degree[p] += 1
    return elements, parent, degree


def _soluble_score(elements, degree, charge, aromatic) -> float:
    """What makes the molecule soluble (module docstring), before the noise."""
    polar = [e in ("N", "O") for e in elements]
    exposed = sum(1 for p, d in zip(polar, degree) if p and d == 1)
    buried = sum(polar) - exposed
    carbons = sum(1 for e, a in zip(elements, aromatic) if e == "C" and not a)
    return (1.3 * exposed + 0.7 * buried + 1.0 * sum(1 for c in charge if c)
            - 0.45 * carbons - 0.6 * sum(aromatic) - 0.8 * elements.count("S") - 1.1 * elements.count("Cl"))


def make(seed: int, rows: int):
    """`rows` molecules (module docstring), their atoms in molecule order, deterministic in
    (seed, rows)."""
    import numpy as np
    import pandas as pd

    if rows < 2:
        raise ValueError(f"molecules needs at least 2 molecules, got {rows}")
    rng = np.random.default_rng(seed)
    lo, hi = ATOMS_PER_MOLECULE
    records, scores, start = [], [], 0
    for m, size in enumerate(rng.integers(lo, hi + 1, rows)):
        elements, parent, degree = _molecule(rng, int(size))
        ring = rng.random() < 0.5
        charge = [int(rng.random() < 0.15) if e == "N" else -int(rng.random() < 0.15) if e == "O" else 0
                  for e in elements]
        aromatic = [int(ring and e in ("C", "N") and rng.random() < 0.6) for e in elements]
        scores.append(_soluble_score(elements, degree, charge, aromatic) + rng.normal(0.0, 0.6))
        for k, element in enumerate(elements):
            records.append((f"A{start + k + 1:06d}", "" if parent[k] < 0 else f"A{start + parent[k] + 1:06d}",
                            f"M{m + 1:04d}", element, charge[k], aromatic[k], degree[k], m))
        start += len(elements)
    cut = float(np.median(scores))
    frame = pd.DataFrame(records, columns=["atom_key", "bonded_to", "molecule_id", "element", "charge",
                                           "aromatic", "degree", "_molecule"])
    frame["soluble"] = (np.asarray(scores)[frame.pop("_molecule").to_numpy()] > cut).astype(np.int64)
    return frame[[c.name for c in SCHEMA]]
