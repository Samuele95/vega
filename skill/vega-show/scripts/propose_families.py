"""
propose_families — `FamilyProposer`: the input groups a draft proposes (`input_groups@2`,
contract §3.7; D5 §6.3.4's family rules). The page draws the network's inputs by group, so a
group is a partition cell of the shaped inputs (`groups.inputs.shaped`, in
`netspec.data_contract.features[].name` order), each member exactly once.

The candidates, in order, each with the evidence that made it:

1. **made by the recipe** — the inputs the recipe created (absent from the raw columns,
   `inputs.created_by_recipe`), one candidate per creating step (the first recipe step whose
   `columns_added` names it) with two or more of them (`{kind: created, step, op}`); the
   created inputs of single-input steps pooled when there are two or more (`{kind: created,
   steps}`). An input an `encode` step created is an encoded one;
2. **encoded** — the encoded nominals (`inputs.encoded`, and the inputs an `encode` step
   created): one candidate; a single one stands alone (`{kind: encoded}`);
3. **moving together** — the remaining inputs joined by union-find over the analysis relations
   (`inputs.relations`, `{a, b, r}`) with |r| at or above `input_groups.corr_threshold`; each
   component of two or more is a candidate (`{kind: correlation, pairs, threshold}`);
4. **sharing a name** — the inputs still alone, grouped greedily by the most frequent name token
   they share (names split on non-alphanumerics and camelCase; a token of two or more letters
   carried by two or more of them; `{kind: name-token, token}`);
5. every input still alone is a candidate of one (no evidence).

Then the count bounds (`input_groups.min`/`max` when there are `min_applies_from_inputs` or
more inputs, else one to the input count): while there are too many, the smallest candidate
is merged into its nearest — the largest |r| between their members, else the most name tokens
shared, else the smallest other — recorded as `{kind: merged, from, by}` on the one that
absorbs it; while there are too few, the largest is split by its most frequent token, else in
two halves in input order (`{kind: split, from, by}`).

Ids are the evidence's words (`custom_s2`, `encoded`, the shared token), made to match
`input_groups.id_pattern` and unique. The groups come out ordered by their first member's
position, each group's members in input order; the model relabels and rewrites them (`V7`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from vega_show_core import CATALOG

__all__ = ["FamilyProposer", "name_tokens"]

_CHUNK = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_STOP = frozenset({"of", "the", "and", "per", "to", "in", "on", "by", "for", "at", "a", "an"})


def name_tokens(name: str) -> list[str]:
    """The lower-case word tokens of an input name (`Wind Speed Mean` → wind, speed, mean;
    `max_rpmDelta` → max, rpm, delta; `HTTPStatus` → http, status), letters only, two or more
    of them, stop words dropped."""
    out = []
    for chunk in _CHUNK.findall(str(name)):
        out += [t.lower() for t in _CAMEL.findall(chunk)]
    return [t for t in dict.fromkeys(out) if len(t) >= 2 and t.isalpha() and t not in _STOP]


@dataclass
class _Group:
    gid: str
    members: list
    evidence: list = field(default_factory=list)


class FamilyProposer:
    """The input-group partition of one network's shaped inputs."""

    def __init__(self, inputs: Mapping, recipe: Sequence[Mapping],
                 catalog: Mapping = CATALOG) -> None:
        self._shaped = [str(n) for n in inputs.get("shaped") or ()]
        self._order = {n: i for i, n in enumerate(self._shaped)}
        self._created = [n for n in self._shaped if n in set(inputs.get("created_by_recipe") or ())]
        self._encoded = set(inputs.get("encoded") or ())
        self._relations = [r for r in inputs.get("relations") or ()
                           if r.get("a") in self._order and r.get("b") in self._order
                           and isinstance(r.get("r"), (int, float))]
        self._recipe, self._spec = list(recipe or ()), catalog["input_groups"]
        self._id_re = re.compile(self._spec["id_pattern"])
        self._groups: list[_Group] = []

    def propose(self) -> list[dict]:
        self._groups = []
        pool = list(self._shaped)
        self._made_by_recipe(pool)
        self._encoded_group(pool)
        self._correlated(pool)
        self._by_token(pool)
        for name in list(pool):
            self._add(name, [name], [], pool)
        self._bound()
        members = [m for g in self._groups for m in g.members]
        if sorted(members) != sorted(self._shaped) or len(members) != len(set(members)):
            raise ValueError("the input groups do not partition the shaped inputs")
        ordered = sorted(self._groups, key=lambda g: self._order[g.members[0]])
        return [{"suggested_id": g.gid, "members": g.members, "evidence": g.evidence}
                for g in ordered]

    # ---------------------------------------------------------------- the candidates
    def _add(self, gid: str, members: list, evidence: list, pool: list) -> _Group:
        group = _Group(self._unique(gid), sorted(members, key=self._order.get), list(evidence))
        self._groups.append(group)
        for name in members:
            pool.remove(name)
        return group

    def _made_by_recipe(self, pool: list) -> None:
        by_step: dict[str, list] = {}
        ops = {}
        for name in self._created:
            step = next((s for s in self._recipe if name in (s.get("columns_added") or ())), None)
            if step is not None and step.get("op") == "encode":
                self._encoded.add(name)
                continue
            key = str(step.get("step")) if step else ""
            by_step.setdefault(key, []).append(name)
            ops[key] = step.get("op") if step else None
        pooled, steps = [], []
        for key, names in by_step.items():
            if key and len(names) >= 2:
                self._add(f"{ops[key] or 'recipe'}_{key}", names,
                          [{"kind": "created", "step": key, "op": ops[key]}], pool)
            else:
                pooled += names
                steps += [key] if key else []
        if len(pooled) >= 2:
            self._add("created", pooled, [{"kind": "created", "steps": steps}], pool)

    def _encoded_group(self, pool: list) -> None:
        names = [n for n in pool if n in self._encoded]
        if names:
            self._add("encoded" if len(names) > 1 else names[0], names, [{"kind": "encoded"}], pool)

    def _correlated(self, pool: list) -> None:
        parent = {n: n for n in pool}

        def root(n):
            while parent[n] != n:
                parent[n] = parent[parent[n]]
                n = parent[n]
            return n
        strong = [r for r in self._relations if r["a"] in parent and r["b"] in parent
                  and abs(r["r"]) >= self._spec["corr_threshold"]]
        for rel in strong:
            parent[root(rel["a"])] = root(rel["b"])
        components: dict[str, list] = {}
        for name in pool:
            components.setdefault(root(name), []).append(name)
        for names in components.values():
            if len(names) >= 2:
                pairs = sum(1 for r in strong if r["a"] in names and r["b"] in names)
                evidence = {"kind": "correlation", "pairs": pairs,
                            "threshold": self._spec["corr_threshold"]}
                self._add(_shared_token(names) or "related", names, [evidence], pool)

    def _by_token(self, pool: list) -> None:
        while True:
            counts: dict[str, list] = {}
            for name in pool:
                for token in name_tokens(name):
                    counts.setdefault(token, []).append(name)
            best = max((t for t, ns in counts.items() if len(ns) >= 2),
                       key=lambda t: len(counts[t]), default=None)
            if best is None:
                return
            self._add(best, counts[best], [{"kind": "name-token", "token": best}], pool)

    # ---------------------------------------------------------------- the count bounds
    def _bound(self) -> None:
        many = len(self._shaped) >= self._spec["min_applies_from_inputs"]
        low, high = (self._spec["min"], self._spec["max"]) if many else (1, len(self._shaped))
        while len(self._groups) > high:
            self._merge_smallest()
        while len(self._groups) < low and self._split_largest():
            pass

    def _merge_smallest(self) -> None:
        small = min(reversed(self._groups), key=lambda g: len(g.members))
        others = [g for g in self._groups if g is not small]
        target, how = self._nearest(small, others)
        target.members = sorted(target.members + small.members, key=self._order.get)
        target.evidence.append({"kind": "merged", "from": small.gid, "by": how})
        self._groups.remove(small)

    def _nearest(self, small: _Group, others: list) -> tuple[_Group, str]:
        mine = set(small.members)
        best_r = {}
        for rel in self._relations:
            for a, b in ((rel["a"], rel["b"]), (rel["b"], rel["a"])):
                if a in mine and b not in mine:
                    best_r[b] = max(best_r.get(b, 0.0), abs(rel["r"]))
        scored = [(max((best_r.get(m, 0.0) for m in g.members), default=0.0), g) for g in others]
        top = max(scored, key=lambda s: s[0])
        if top[0] > 0:
            return top[1], "correlation"
        tokens = {t for m in small.members for t in name_tokens(m)}
        shared = [(len(tokens & {t for m in g.members for t in name_tokens(m)}), g) for g in others]
        top = max(shared, key=lambda s: s[0])
        if top[0] > 0:
            return top[1], "name-token"
        return min(others, key=lambda g: len(g.members)), "size"

    def _split_largest(self) -> bool:
        large = max(self._groups, key=lambda g: len(g.members))
        if len(large.members) < 2:
            return False
        counts: dict[str, list] = {}
        for name in large.members:
            for token in name_tokens(name):
                counts.setdefault(token, []).append(name)
        cuts = [t for t, ns in counts.items() if 1 <= len(ns) < len(large.members)]
        token = max(cuts, key=lambda t: len(counts[t]), default=None)
        part = counts[token] if token else large.members[len(large.members) // 2:]
        large.members = [m for m in large.members if m not in part]
        new = _Group(self._unique(token or f"{large.gid}_part"), list(part),
                     [{"kind": "split", "from": large.gid,
                       "by": "name-token" if token else "order"}])
        self._groups.insert(self._groups.index(large) + 1, new)
        return True

    def _unique(self, word: str) -> str:
        base = re.sub(r"[^a-z0-9]+", "_", str(word).lower()).strip("_")
        base = base if base[:1].isalpha() else f"g_{base}".rstrip("_")
        base, taken = base[:28], {g.gid for g in self._groups}
        gid, n = base, 2
        while gid in taken or not self._id_re.fullmatch(gid):
            gid, n = f"{base}_{n}", n + 1
        return gid


def _shared_token(names: list) -> str | None:
    """The name token most of `names` share (two or more of them), else None."""
    counts: dict[str, int] = {}
    for name in names:
        for token in name_tokens(name):
            counts[token] = counts.get(token, 0) + 1
    best = max(counts, key=counts.get, default=None)
    return best if best is not None and counts[best] >= 2 else None
