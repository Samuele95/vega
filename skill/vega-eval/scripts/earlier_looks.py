"""
earlier_looks — the looks an earlier campaign took at this test part (`campaign.json.after_eval`).

`/vega-train --after-eval` starts a **new** campaign after the test was seen: a new directory,
no seal of its own, and `campaign.json.after_eval = {previous_campaign, exposures}` (run-contract
§1). The test part is the same file, so the first look in such a campaign is a **further**
exposure (eval-contract §1): `evaluate.py` holds it to the further-exposure rules before the seal
— no `--threshold <value>`, no `--slices`, no baseline set narrower than the previous campaign's
evaluation measured — and its seal counts the earlier looks (`exposures = after_eval.exposures +
1`), so the layer, the notebook, the summary and the bundle all disclose the total.

`read` reconciles the record with the previous campaign's own seal before trusting it: a
previous campaign that cannot be found, has no seal, or whose seal counts a different number of
looks is refused — a count that cannot be checked is not one to build a further look on.

Pure reads of JSON records; no argv, no printing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from threshold_choice import requested_mode

__all__ = ["EarlierLooks", "read"]


@dataclass(frozen=True)
class EarlierLooks:
    """What the previous campaign saw: where it is, how many looks, and what it measured."""

    previous_campaign: str
    exposures: int
    run: str | None
    baselines: tuple
    threshold_mode: str | None = None     # the --threshold mode its last look chose under

    def record(self) -> dict:
        """The seal's `after_eval` entry (and campaign.json's), as run-contract §1 spells it."""
        return {"previous_campaign": self.previous_campaign, "exposures": self.exposures}


def _locate(previous: str, camp_dir: Path) -> Path | None:
    """The previous campaign's directory: as written (absolute, or from the working directory),
    then beside this campaign, then from the work dir that holds `vega-runs/`."""
    raw = Path(previous)
    for candidate in (raw, camp_dir.parent / raw.name, camp_dir.parent / raw,
                      camp_dir.parent.parent / raw):
        if candidate.is_dir() and (candidate / "campaign.json").is_file():
            return candidate
    return None


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path} cannot be read ({exc})"


def read(campaign: Mapping, camp_dir) -> tuple:
    """(EarlierLooks, None), (None, None) for a campaign of its own, or (None, why the record
    cannot be trusted)."""
    camp_dir = Path(camp_dir)
    raw = campaign.get("after_eval")
    if raw in (None, {}, False):
        return None, None
    if not isinstance(raw, Mapping):
        return None, (f"campaign.json's after_eval is {raw!r}, not the record "
                      "{previous_campaign, exposures} /vega-train --after-eval writes")
    previous, count = raw.get("previous_campaign"), raw.get("exposures")
    if not previous:
        return None, ("campaign.json's after_eval names no previous_campaign; the earlier looks "
                      "cannot be reconciled with a seal nobody can find")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        return None, (f"campaign.json's after_eval records exposures {count!r}; an after-eval "
                      "campaign exists because the test part was seen at least once")
    where = _locate(str(previous), camp_dir)
    if where is None:
        return None, (f"after_eval names the previous campaign {previous!r}, which is not "
                      f"found (looked as written, beside {camp_dir} and above it); the "
                      f"{count} earlier look(s) cannot be reconciled with its seal")
    if where.resolve() == camp_dir.resolve():
        return None, "after_eval names this campaign as its own predecessor"
    seal, err = _json(where / "test_seal.json")
    if err:
        return None, (f"the previous campaign {where} has no readable test_seal.json ({err}); "
                      f"after_eval's {count} earlier look(s) cannot be reconciled with it")
    seen = seal.get("exposures")
    if seen != count:
        return None, (f"after_eval records {count} earlier look(s) and the previous campaign's "
                      f"seal ({where / 'test_seal.json'}) records {seen}; the count is the "
                      "seal's, and one that does not reconcile is not a count to build on")
    run = seal.get("run")
    names, mode = (), None
    if run:
        doc, _ = _json(where / "eval" / str(run) / "baselines.json")
        names = tuple(b.get("name") for b in ((doc or {}).get("baselines") or ())
                      if b.get("name"))
        threshold, _ = _json(where / "eval" / str(run) / "threshold.json")
        mode = requested_mode(threshold)
    return EarlierLooks(previous_campaign=str(previous), exposures=int(count), run=run,
                        baselines=names, threshold_mode=mode), None
