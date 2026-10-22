#!/usr/bin/env python3
"""
bootstrap_base.py — the minimal dataset model /data-lens uses under `--standalone`.

    python3 bootstrap_base.py DATASET --out BASE.html [--abox N] [--seed 7]
                              [--domain-forge-dir PATH] [--name SLUG]

Writes a domain-forge model HTML from domain-forge's `assets/template.html`: one
`ex:Record` class with one data property per column, an `owl:Ontology` node carrying
`ex:sourceKind "dataset"` (so a later /domain-forge refine recognises a dataset ontology),
an optional A-box row sample, and an Abstract that says in plain words what this model is
and — the honest part — that its roles were GUESSED from the data, not modelled. It writes
no rules and no `geometry` layer: it is the floor under an analysis, not a forge.

The types are heuristics; the column whose values are all distinct is recorded as the
identity. Anything richer — nominal classes with individuals, lookup hierarchies,
derivation rules — is /dataset-forge's job, and the Abstract says so.

Exit codes: 0 wrote the model; 2 input error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SKILL_DIR = HERE.parent

XSD = {"integer": "xsd:integer", "decimal": "xsd:decimal", "boolean": "xsd:boolean",
       "date": "xsd:date", "dateTime": "xsd:dateTime", "string": "xsd:string"}


def slug(s):
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")
    return s or "col"


def prop_name(c):
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", str(c)) if p]
    if not parts:
        return "hasValue"
    head = parts[0][0].lower() + parts[0][1:]
    return head + "".join(p[:1].upper() + p[1:] for p in parts[1:])


def xsd_for(s):
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "decimal"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "dateTime"
    txt = s.dropna().astype(str)
    if len(txt):
        if txt.str.match(r"^\d{4}-\d{2}-\d{2}$").mean() > 0.9:
            return "date"
        if txt.str.match(r"^\d{4}-\d{2}-\d{2}[T ]").mean() > 0.9:
            return "dateTime"
    return "string"


def ttl_literal(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if kind == "boolean":
        return "true" if bool(v) else "false"
    if kind == "integer":
        try:
            return str(int(v))
        except Exception:
            return None
    if kind == "decimal":
        try:
            f = float(v)
            if not np.isfinite(f):
                return None
            return repr(round(f, 10))
        except Exception:
            return None
    if kind in ("date", "dateTime"):
        return '"%s"^^%s' % (str(v)[:19].replace(" ", "T") if kind == "dateTime" else str(v)[:10],
                             XSD[kind])
    return json.dumps(str(v), ensure_ascii=False)


def build(df, name, abox_n, seed):
    cols = list(df.columns)
    kinds = {c: xsd_for(df[c]) for c in cols}
    props = {}
    for c in cols:
        p = prop_name(c)
        base, i = p, 2
        while p in props.values():
            p, i = f"{base}{i}", i + 1
        props[c] = p
    identities = [c for c in cols if df[c].nunique(dropna=True) == len(df) and len(df) > 1]
    ttl = [f"@prefix ex:   <http://example.org/{name}#> .",
           "@prefix owl:  <http://www.w3.org/2002/07/owl#> .",
           "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
           "@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .", "",
           f"<http://example.org/{name}> a owl:Ontology ;",
           '    ex:sourceKind "dataset" ;',
           f'    ex:sourcePath "{name}" ;',
           '    rdfs:comment "Bootstrapped by /data-lens --standalone: one class, one data '
           'property per column, types and roles guessed from the data." .', "",
           "# record",
           'ex:Record a owl:Class ; ex:component "record" ; rdfs:label "Record" ; '
           'rdfs:comment "One row of the dataset." .']
    for c in cols:
        note = " # identity (all values distinct)" if c in identities else ""
        ttl.append(f"ex:{props[c]} a owl:DatatypeProperty ; rdfs:domain ex:Record ; "
                   f'rdfs:range {XSD[kinds[c]]} ; rdfs:label "{c}" .{note}')
    graph = [{"@id": "ex:Record", "@type": "owl:Class", "component": "record", "label": "Record",
              "comment": "One row of the dataset."}]
    for c in cols:
        graph.append({"@id": f"ex:{props[c]}", "@type": "owl:DatatypeProperty",
                      "domain": "ex:Record", "range": XSD[kinds[c]], "label": str(c)})
    if abox_n and len(df):
        rng = np.random.default_rng(seed)
        n = min(int(abox_n), len(df))
        idx = sorted(rng.choice(len(df), n, replace=False).tolist())
        ttl += ["", "# individuals (a seeded row sample; the A-box of this dataset)"]
        for i in idx:
            row = df.iloc[i]
            pairs = []
            for c in cols:
                lit = ttl_literal(row[c], kinds[c])
                if lit is not None:
                    pairs.append(f"ex:{props[c]} {lit}")
            ttl.append(f"ex:Record-{i} a owl:NamedIndividual , ex:Record"
                       + ((" ; " + " ; ".join(pairs)) if pairs else "") + " .")
            node = {"@id": f"ex:Record-{i}", "@type": ["owl:NamedIndividual", "ex:Record"]}
            for c in cols:
                v = row[c]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                if kinds[c] in ("integer",):
                    node[f"ex:{props[c]}"] = int(v)
                elif kinds[c] == "decimal":
                    f = float(v)
                    if np.isfinite(f):
                        node[f"ex:{props[c]}"] = round(f, 10)
                elif kinds[c] == "boolean":
                    node[f"ex:{props[c]}"] = bool(v)
                else:
                    node[f"ex:{props[c]}"] = str(v)
            graph.append(node)
    md = f"""# {name} — dataset record model (bootstrapped)

**Purpose & scope.** This model is the floor under a `/data-lens` analysis of `{name}`, not a
forged domain model. It says exactly what a table says: there are rows, and each row carries
one value per column. One class, `Record`, holds {len(cols)} data properties — one per column —
with the datatypes a parser guessed from the values.

**Components.** A single component, `record`. There are no bounded contexts here because a
table has none: the concepts a dataset speaks about live in its columns' meaning, and reading
that meaning out is `/dataset-forge`'s work, not this file's.

**Modeling decisions.** Types are heuristics ({', '.join(sorted(set(kinds.values())))}) and the
roles behind them are GUESSED, not modelled: {('the column(s) ' + ', '.join(identities) + ' are all-distinct and are treated as the identity') if identities else 'no column is all-distinct, so no identity was assumed'}.
Nothing is derived from anything: no column is expressed as a rule over the others, no lookup
hierarchy is declared, no partition is proposed. A downstream reader must treat every property
as primitive until a forge says otherwise.

**Knowledge layers.** Ontology only. `model-dmn`, `model-horn` and `model-swrl` are empty on
purpose: a rule this file cannot justify would be a claim no engine could check.

**Open questions.** Which columns are definable from the others, which is a label and which
its leakage set, and what the data actually means — all of it. Run `/dataset-forge {name}` to
answer them, and this bootstrap becomes unnecessary.
"""
    return "\n".join(ttl) + "\n", graph, md


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--abox", type=int, default=0, help="rows in the A-box sample (0 = none)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--name", default=None, help="ontology slug (default: the dataset stem)")
    ap.add_argument("--domain-forge-dir", default=None)
    ap.add_argument("--sep", default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(HERE))
    from analysis import read_dataset
    try:
        df = read_dataset(a.dataset, a.sep)
    except Exception as e:
        print(f"ERROR: cannot read {a.dataset}: {e}")
        return 2
    df_dir = Path(a.domain_forge_dir) if a.domain_forge_dir else SKILL_DIR.parent / "domain-forge"
    tpl = df_dir / "assets" / "template.html"
    if not tpl.is_file():
        print(f"ERROR: domain-forge template not found at {tpl} (pass --domain-forge-dir)")
        return 2
    html = tpl.read_text(encoding="utf-8")
    name = a.name or Path(a.dataset).stem
    ttl, graph, md = build(df, name, a.abox, a.seed)

    def replace(block_id, new_inner):
        nonlocal html
        pat = re.compile(r'(<script id="' + block_id + r'"[^>]*>)([\s\S]*?)(</script>)')
        if not pat.search(html):
            raise SystemExit(f"ERROR: template has no <script id={block_id}>")
        html = pat.sub(lambda m: m.group(1) + new_inner + m.group(3), html, count=1)

    jsonld = json.loads(re.search(r'<script id="model-jsonld"[^>]*>([\s\S]*?)</script>',
                                  html).group(1))
    jsonld["@context"]["ex"] = f"http://example.org/{name}#"
    jsonld["@graph"] = graph
    replace("domain-model", "\n" + ttl)
    replace("model-jsonld", "\n" + json.dumps(jsonld, indent=2, ensure_ascii=False) + "\n")
    replace("model-markdown", "\n" + md)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"OK: wrote {out} — 1 class, {len(df.columns)} data properties, "
          f"{a.abox if a.abox else 'no'} A-box rows, ex:sourceKind \"dataset\"")
    print(f"OK: validate it with {df_dir}/scripts/validate_model.py {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
