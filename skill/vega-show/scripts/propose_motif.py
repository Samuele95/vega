"""
propose_motif — `MotifProposer`: the backdrop motif a draft proposes (contract §3.4, R-E4; D4
§9.3's choice, which R-E9 makes the draft's).

The motif set is closed and keyed by modality (`catalog.motifs.by_modality`). The modality is the
fact set's (`task.modality`: `spatial-row` when the sample shape is `row` and the analysis records
a spatial context, else the sample shape), so a spatial row lane is based on `contours` and every
other lane on its shape's base (`points`, `traces`, `links`). The proposal is

    {modality, base, admissible, stance, fork}

- **a `show: motif <id> (user)` stance** of this dataset (the latest) whose id is admissible for
  the modality is applied: `stance` is the memory line (its date and `show: motif <id>`, which
  `page_config` cites as `show:<date>`), and fork `S3` is not asked;
- **otherwise** fork `S3` is open when the modality admits more than one motif (the base is
  offered first, the variants after it); a stance naming a motif outside the admissible list is
  a `WARN:` and not applied — a motif outside the list is refused downstream (`V5`, `BD2`).

`answer(proposal)` is the motif the draft settles without asking: the stance's, else the base —
fork S3's unattended answer (`catalog.forks.S3`).
"""
from __future__ import annotations

import re
from typing import Mapping

from memory_stances import USER_SUFFIX, MemoryStances
from vega_show_core import CATALOG

__all__ = ["MotifProposer"]

_ID = re.compile(r"^([a-z][a-z-]*)$")


class MotifProposer:
    """The motif proposal of one modality, with the stance that settles it when one exists."""

    def __init__(self, modality: str, stances: MemoryStances, catalog: Mapping = CATALOG) -> None:
        table = catalog["motifs"]["by_modality"]
        if modality not in table:
            raise ValueError(f"modality {modality!r} is not one of the catalog's {sorted(table)}")
        self._modality, self._spec, self._stances = modality, table[modality], stances
        self._fork = catalog["motifs"]["fork"]
        self._markers: list[str] = []

    @property
    def markers(self) -> tuple[str, ...]:
        return tuple(self._markers)

    def propose(self) -> dict:
        self._markers = []
        admissible = list(self._spec["admissible"])
        stance = self._stance(admissible)
        fork = None if stance is not None or len(admissible) < 2 else self._fork
        return {"modality": self._modality, "base": self._spec["base"], "admissible": admissible,
                "stance": stance.line if stance else None, "fork": fork}

    def answer(self, proposal: Mapping) -> str:
        """The stance's motif when the proposal carries one, else the modality's base."""
        stance = proposal.get("stance")
        if stance:
            named = _motif_of(re.sub(r"^.*?show:\s*motif\s+", "", str(stance)))
            if named in proposal.get("admissible", ()):
                return named
        return str(proposal["base"])

    def _stance(self, admissible: list[str]):
        stance = self._stances.latest("motif")
        if stance is None:
            return None
        named = _motif_of(stance.body)
        if named not in admissible:
            self._markers.append(f"WARN: the stance {stance.line!r} names the motif {named!r}, "
                                 f"which the {self._modality} modality does not admit "
                                 f"({', '.join(admissible)}); not applied — fork {self._fork} "
                                 "stays open")
            return None
        return stance


def _motif_of(body: str) -> str | None:
    word = USER_SUFFIX.sub("", body).strip().split(" ")[0] if body else ""
    return word if _ID.fullmatch(word) else None
