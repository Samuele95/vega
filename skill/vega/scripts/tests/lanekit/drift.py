"""
Drift: where the copies lanekit.contracts keeps disagree with the catalogs that own the names.

    from lanekit import drift
    drift.drift(skills)       # [] when no copy drifted; run_matrix.py validate reports each as an ERROR

One helper per owning catalog: the forge catalog's `inspection.links` (K1), the pack catalog's
`field_kinds` (K2), and the vega-show catalog's motifs, roles, lexicon, views, slot and probe
(K3). A catalog or key that does not exist yet is not compared: its package has not landed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import contracts as c


def drift(skills: Path | None = None) -> list[str]:
    """Every disagreement between contracts.py and the catalogs under `skills` (default: the
    tree this lanekit sits in)."""
    root = skills or c.SKILLS_ROOT
    show = _catalog(root, c.SHOW_CATALOG)
    return (forge_drift(_catalog(root, c.FORGE_CATALOG)) + pack_drift(_catalog(root, c.PACK_CATALOG))
            + show_names_drift(show) + show_places_drift(show))


def forge_drift(forge: Mapping[str, Any]) -> list[str]:
    """`inspection.links` and the reading each link serves (R-C1)."""
    links = (forge.get("inspection") or {}).get("links")
    if not isinstance(links, Mapping):
        return []
    derived = {reading: link for link, row in links.items() for reading in (row.get("readings") or {})}
    return (_differs("inspection.links", sorted(c.LINKS), sorted(links))
            + _differs("reading -> link", dict(sorted(c.READING_LINK.items())), dict(sorted(derived.items()))))


def pack_drift(pack: Mapping[str, Any]) -> list[str]:
    """`field_kinds`: each kind's answer key, and the include that shows it (R-C4)."""
    kinds = pack.get("field_kinds")
    if not isinstance(kinds, Mapping):
        return []
    shown = {k: str((v or {}).get("shown") or "") for k, v in kinds.items()}
    return (_differs("field_kinds kind -> key", dict(sorted(c.FIELD_KIND_KEYS.items())),
                     {k: (v or {}).get("key") for k, v in sorted(kinds.items())})
            + _differs("field_kinds shown on include", dict(sorted(c.FIELD_KIND_INCLUDE.items())),
                       {k: s.split("=", 1)[1] for k, s in sorted(shown.items()) if s.startswith("include=")}))


def show_names_drift(show: Mapping[str, Any]) -> list[str]:
    """The vega-show names a lane declares: motifs (R-E4), roles (R-E5), the lexicon (R-E6)."""
    out: list[str] = []
    motifs = show.get("motifs") or {}
    if isinstance(motifs.get("by_modality"), Mapping):
        out += _differs("motifs.set", list(c.MOTIFS), list(motifs.get("set") or []))
        out += _differs("motifs.by_modality",
                        {m: {"base": r["base"], "admissible": list(r["admissible"])}
                         for m, r in c.MOTIFS_BY_MODALITY.items()},
                        {m: {"base": r.get("base"), "admissible": list(r.get("admissible") or [])}
                         for m, r in motifs["by_modality"].items()})
    roles = (show.get("roles") or {}).get("set")
    if roles:
        out += _differs("roles.set", list(c.ROLE_VALUES), list(roles))
    lexicon = show.get("lexicon") or {}
    if lexicon.get("keys"):
        out += _differs("lexicon.keys", list(c.LEXICON_KEYS), list(lexicon["keys"]))
    if isinstance(lexicon.get("required"), Mapping):
        theirs = {k.replace(":", "-").replace("_", "-"): tuple(v) for k, v in lexicon["required"].items()}
        out += _differs("lexicon.required", dict(c.LEXICON_REQUIRED), theirs)
    return out


def show_places_drift(show: Mapping[str, Any]) -> list[str]:
    """Where the page and the slot live: views (R-F4), the showcase slot (R-E1), the probe (R-F1)."""
    out: list[str] = []
    views = show.get("views") or {}
    if views.get("ids"):
        out += _differs("views.ids", list(c.VIEWS), list(views["ids"]))
        out += _differs("views.sub_views", {k: list(v) for k, v in c.SUB_VIEWS.items()},
                        {k: list(v) for k, v in (views.get("sub_views") or {}).items()})
        out += _differs("views.aliases", dict(c.VIEW_ALIASES), dict(views.get("aliases") or {}))
    slot = show.get("slot") or {}
    if slot.get("manifest"):
        out += _differs("slot.manifest", c.SHOWCASE_MANIFEST, slot["manifest"])
        out += _differs("slot.dir", c.SHOWCASE_DIR, slot.get("dir"))
        out += _differs("slot.schema", c.SHOWCASE_SCHEMA, slot.get("schema"))
    probe = show.get("probe") or {}
    if probe.get("hook"):
        out += _differs("probe.hook", c.PROBE_HOOK, probe["hook"])
        out += _differs("probe.query", c.PROBE_QUERY, probe.get("query"))
        out += _differs("probe.result_id", c.PROBE_RESULT_ID, probe.get("result_id"))
        out += _differs("probe.qps_dir", c.SHOW_PSEUDO_DIR, str(probe.get("qps_dir") or "").rstrip("/"))
    return out


def _catalog(skills: Path, rel: str) -> Mapping[str, Any]:
    path = skills / rel
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _differs(what: str, mine: Any, theirs: Any) -> list[str]:
    return [] if mine == theirs else [f"{what}: contracts.py has {mine!r}, the catalog {theirs!r}"]
