#!/usr/bin/env python3
"""
page_template — the page template's identity (BUILD-PLAN R-E1, R-E2; D4 §3.2, §17.1 check 1).

    python3 page_template.py [--template DIR] --write          write TEMPLATE.json (W15 first)
    python3 page_template.py [--template DIR] --check          PG1: TEMPLATE.json == write_manifest()
    python3 page_template.py [--template DIR] --verify PAGE    BD1: PAGE is a byte copy of it
    python3 page_template.py [--template DIR] --digests        SHOWCASE.json.template, as JSON

The template (`vega-show/assets/page/`) is copied byte for byte into `<bundle>/showcase/page/`
and nothing in it is substituted. `TEMPLATE.json` (`vega/page-template@1`) pins it:

- `files`: every file under the template except the catalog's `template.excluded`
  (`TEMPLATE.json`, `package.json`, `tests/`), with its sha256. A symlink, or a file whose
  extension the schema does not admit (html css js json svg woff2), is refused: the manifest
  would pin what the slot will not serve.
- `requires_endpoints`: `REQUIRED_ENDPOINTS`, the one declaration of the service routes the
  page needs (D5 §2.2's list, which R-E2 moves here, and `GET /health`, which the page boots
  on); PF5 checks it against `MANIFEST.serve.endpoints`. `OPTIONAL_ENDPOINTS` are the routes
  the page reads when the service has them (`POST /v1/monitor` only when a monitor ships, D5
  §2.2). The modules are read only to hold the two lists to what the page calls: every call of
  `api`, `post`, `upload` or `fetch` on a `URLS.<name>` of `js/core/api.js` (also as the head
  of a template, `` `${URLS.x}?…` ``) whose URL is a service route (`health`, `v<n>/…`) must
  be on one of them, `METHOD /path` (`post` and `upload` are POST, `api` and `fetch` GET unless
  the call names its `method`; the path without its query). A slot URL (`showcase/…`,
  `report.pdf`) is a file, not a route. A service path typed outside the `URLS` table — a
  string or a template, with or without a leading `/` or `./` — is refused.
- `strings_keys_digest`: the digest of `strings.keys.json` (the generated key list).

`write_manifest()` computes the document; `--write` writes it with `vega_show_core.write_json`,
and so does `digests()` into a scratch directory while the file does not exist yet: its
`manifest_digest` is the digest of the bytes a fresh `--write` gives (kept per template root
and content, so a process asks it once per template state).
`verify_template_copy(page_dir)` compares a copy with `TEMPLATE.json` (with `write_manifest()`
while the file does not exist yet) and returns one line per missing, unlisted or altered file;
`[]` only for an exact copy.

Exit 0: done / identical / an exact copy. 1: PG1 or BD1 found a difference. 2: the template or
the copy cannot be read.
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import tempfile
from pathlib import Path

from js_tokens import TokenError, flat, significant, tokenize_file
from vega_show_core import CATALOG, digest_file, write_json

__all__ = ["TEMPLATE_DIR", "VERSION", "REQUIRED_ENDPOINTS", "OPTIONAL_ENDPOINTS", "TemplateError",
           "template_files", "endpoint_calls", "endpoint_problems", "requires_endpoints",
           "write_manifest", "read_manifest", "identity_problems", "verify_template_copy", "digests",
           "main"]

SKILL = Path(__file__).resolve().parent.parent
SPEC = CATALOG["template"]
TEMPLATE_DIR = SKILL / SPEC["dir"]
MANIFEST = SPEC["manifest"]
SCHEMA = SPEC["schema"]
STRINGS_KEYS = Path(SPEC["strings_keys"]).relative_to(SPEC["dir"]).as_posix()
VERSION = "2.3.0"                     # the template's release: SHOWCASE.json.template.version
EXTENSIONS = frozenset({".html", ".css", ".js", ".json", ".svg", ".woff2"})
URL_TABLE = "js/core/api.js"
CALLERS = {"api": "GET", "fetch": "GET", "post": "POST", "upload": "POST"}
SERVICE_PATH = re.compile(r"(?:\.?/)?(?:health(?=$|[/?#])|v\d+/)")      # a call's URL argument
ROUTE_LITERAL = re.compile(r"(?:\.?/)?v\d+/|\.?/health(?=$|[/?#])")       # any other literal
STRINGS_TABLE = re.compile(r"^js/core/strings(?:\.en|/[a-z-]+\.en)\.js$")   # prose, never a URL
# BUILD-PLAN R-E2: TEMPLATE.json's requires_endpoints (PF5: ⊆ MANIFEST.serve.endpoints) —
# D5 §2.2's list and the readiness route the page boots on (D4 §4.2)
REQUIRED_ENDPOINTS = ("GET /health", "GET /v1/schema", "GET /v1/examples", "GET /v1/network",
                      "POST /v1/explain", "GET /v1/code", "POST /v1/predict", "POST /v1/predict/csv")
# read when the service has them: the monitor ships only with one (D5 §2.2); info and the
# slot's story route are enrichments
OPTIONAL_ENDPOINTS = ("GET /v1/info", "POST /v1/monitor", "GET /v1/story", "GET /v1/showcase")


class TemplateError(RuntimeError):
    """The template cannot be pinned: unreadable, a symlink, a file the schema does not admit."""


def _excluded(rel: str) -> bool:
    return any(rel.startswith(x) if x.endswith("/") else rel == x for x in SPEC["excluded"])


def template_files(template: Path = TEMPLATE_DIR) -> list[str]:
    """The template's pinned files, relative and sorted (the catalog's exclusions left out)."""
    template = Path(template)
    if not template.is_dir():
        raise TemplateError(f"the page template {template} is not a directory")
    out = []
    for path in sorted(template.rglob("*")):
        rel = path.relative_to(template).as_posix()
        if _excluded(rel) or (path.is_dir() and not path.is_symlink()):
            continue
        if path.is_symlink():
            raise TemplateError(f"{rel}: a symlink in the template (the slot refuses symlinks)")
        if path.suffix not in EXTENSIONS:
            raise TemplateError(f"{rel}: a {path.suffix or 'extensionless'} file; the template "
                                f"holds only {' '.join(sorted(EXTENSIONS))}")
        out.append(rel)
    return out


# ---- the endpoints the modules call ------------------------------------------------------------
def _url_table(path: Path) -> tuple[dict, tuple[int, int]]:
    """({name: url}, (first line, last line)) of `export const URLS = Object.freeze({…})`."""
    toks = flat(tokenize_file(path))
    start = next((i for i, t in enumerate(toks) if t.is_("ident", "URLS") and i + 1 < len(toks)
                  and toks[i + 1].is_("punct", "=")), None)
    if start is None:
        raise TemplateError(f"{path}: no URLS table")
    out, i = {}, start
    while i + 2 < len(toks) and not toks[i].is_("punct", "}"):
        if toks[i].kind in ("ident", "keyword", "string") and toks[i + 1].is_("punct", ":") \
                and toks[i + 2].kind == "string":
            out[toks[i].value if toks[i].kind == "string" else toks[i].text] = toks[i + 2].value
            i += 3
        else:
            i += 1
    return out, (toks[start].line, toks[min(i, len(toks) - 1)].line)


def _call_end(toks: list, open_: int) -> int:
    depth = 0
    for i in range(open_, len(toks)):
        if toks[i].kind == "punct" and toks[i].text in "([{":
            depth += 1
        elif toks[i].kind == "punct" and toks[i].text in ")]}":
            depth -= 1
            if depth == 0:
                return i
    return len(toks) - 1


def _method(toks: list, open_: int, close: int, default: str) -> str:
    for i in range(open_, close - 1):
        if toks[i].is_("ident", "method") and toks[i + 1].is_("punct", ":") and toks[i + 2].kind == "string":
            return toks[i + 2].value.upper()
    return default


def _urls_name(toks: list, i: int) -> str | None:
    """The `<name>` of `URLS.<name>` at toks[i], or of a template that starts with `${URLS.<name>}`."""
    head = toks[i:i + 3]
    if head and head[0].kind == "template" and head[0].inner and not head[0].parts[0]:
        head = significant(head[0].inner[0])[:3]
    if len(head) == 3 and head[0].is_("ident", "URLS") and head[1].is_("punct", "."):
        return head[2].text
    return None


def _typed_paths(toks: list, rel: str, table: tuple) -> None:
    """Refuse a service path typed outside the URLS table: a call's URL argument that names a
    route (`"health"`, `"/v1/x"`, `` `v1/x/${id}` ``), or any literal that starts with a
    versioned route or `/health`."""
    for i, t in enumerate(toks):
        text = t.value if t.kind == "string" else t.parts[0] if t.kind == "template" else None
        if text is None or (rel == URL_TABLE and table[0] <= t.line <= table[1]):
            continue
        argument = i >= 2 and toks[i - 1].is_("punct", "(") and toks[i - 2].text in CALLERS
        if (SERVICE_PATH if argument else ROUTE_LITERAL).match(text):
            raise TemplateError(f"{rel}:{t.line}: {t.text[:60]} types a service URL outside the URLS "
                                f"table of {URL_TABLE}")


def _calls(toks: list, rel: str, urls: dict) -> dict:
    """{`METHOD /path`: [rel:line]} of the service routes one module's calls read."""
    out = {}
    for i, t in enumerate(toks[:-2]):
        if t.kind != "ident" or t.text not in CALLERS or not toks[i + 1].is_("punct", "("):
            continue
        name = _urls_name(toks, i + 2)
        if name is None:
            continue
        if name not in urls:
            raise TemplateError(f"{rel}:{t.line}: URLS.{name} is not in the URLS table of {URL_TABLE}")
        url, close = urls[name].split("?")[0], _call_end(toks, i + 1)
        if SERVICE_PATH.match(url):
            route = f"{_method(toks, i + 1, close, CALLERS[t.text])} /{url.lstrip('./')}"
            out.setdefault(route, []).append(f"{rel}:{t.line}")
    return out


def endpoint_calls(template: Path = TEMPLATE_DIR) -> dict:
    """{`METHOD /path`: [file:line, …]} of every service route the template's modules call."""
    template = Path(template)
    try:
        urls, table = _url_table(template / URL_TABLE)
        found = {}
        for rel in template_files(template):
            if rel.endswith(".js") and not STRINGS_TABLE.match(rel):
                toks = flat(tokenize_file(template / rel))
                _typed_paths(toks, rel, table)
                for route, where in _calls(toks, rel, urls).items():
                    found.setdefault(route, []).extend(where)
    except (OSError, TokenError) as exc:
        raise TemplateError(f"the template's modules cannot be read: {exc}") from exc
    return found


def endpoint_problems(template: Path = TEMPLATE_DIR) -> list[str]:
    """One line per route a module calls that neither list declares."""
    known = set(REQUIRED_ENDPOINTS) | set(OPTIONAL_ENDPOINTS)
    return [f"{where[0]}: {route} is called but is neither a required nor an optional endpoint "
            f"(REQUIRED_ENDPOINTS / OPTIONAL_ENDPOINTS of page_template.py)"
            for route, where in sorted(endpoint_calls(template).items()) if route not in known]


def requires_endpoints(template: Path = TEMPLATE_DIR) -> list[str]:
    """TEMPLATE.json's requires_endpoints: REQUIRED_ENDPOINTS, once every route the modules call
    is on one of the two lists."""
    problems = endpoint_problems(template)
    if problems:
        raise TemplateError("; ".join(problems))
    return list(REQUIRED_ENDPOINTS)


# ---- the manifest ------------------------------------------------------------------------------------
def write_manifest(template: Path = TEMPLATE_DIR) -> dict:
    """The `vega/page-template@1` document of the template as it is on disk now."""
    template = Path(template)
    files = template_files(template)
    keys = template / STRINGS_KEYS
    if STRINGS_KEYS not in files:
        raise TemplateError(f"{keys} is missing: run node tests/strings-keys.js --write")
    return {"schema": SCHEMA, "version": VERSION,
            "files": {rel: digest_file(template / rel) for rel in files},
            "requires_endpoints": requires_endpoints(template),
            "strings_keys_digest": digest_file(keys)}


def read_manifest(template: Path = TEMPLATE_DIR) -> dict | None:
    """TEMPLATE.json as a document; None while it does not exist (W15 writes it first)."""
    path = Path(template) / MANIFEST
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TemplateError(f"{path} cannot be read: {exc}") from exc
    files = doc.get("files") if isinstance(doc, dict) else None
    if doc.get("schema") != SCHEMA or not isinstance(files, dict) or not files:
        raise TemplateError(f"{path} is not a {SCHEMA} document with its files")
    return doc


def identity_problems(template: Path = TEMPLATE_DIR) -> list[str]:
    """PG1: how TEMPLATE.json differs from a fresh write_manifest(); [] when it is absent."""
    pinned = read_manifest(template)
    if pinned is None:
        return []
    fresh = write_manifest(template)
    out = [f"{rel}: {why}" for rel, why in _file_diff(pinned["files"], fresh["files"], "the template")]
    out += [f"{key}: TEMPLATE.json holds {pinned.get(key)!r}, the template gives {fresh[key]!r}"
            for key in ("schema", "version", "requires_endpoints", "strings_keys_digest")
            if pinned.get(key) != fresh[key]]
    extra = sorted(set(pinned) - set(fresh))
    return out + ([f"TEMPLATE.json carries keys the schema does not: {', '.join(extra)}"] if extra else [])


def _file_diff(pinned: dict, found: dict, where: str) -> list[tuple[str, str]]:
    out = [(rel, f"pinned but missing from {where}") for rel in sorted(set(pinned) - set(found))]
    out += [(rel, f"in {where} but not pinned") for rel in sorted(set(found) - set(pinned))]
    out += [(rel, f"{where} has {found[rel]}, TEMPLATE.json pins {pinned[rel]}")
            for rel in sorted(set(pinned) & set(found)) if pinned[rel] != found[rel]]
    return out


def verify_template_copy(page_dir: Path, template: Path = TEMPLATE_DIR) -> list[str]:
    """BD1: one line per file the copy lacks, adds or alters; [] only for an exact copy of the
    TEMPLATE.json files (of write_manifest()'s while TEMPLATE.json does not exist)."""
    page_dir = Path(page_dir)
    if not page_dir.is_dir():
        return [f"{page_dir}: not a directory"]
    pinned = (read_manifest(template) or write_manifest(template))["files"]
    found, out = {}, []
    for path in sorted(page_dir.rglob("*")):
        rel = path.relative_to(page_dir).as_posix()
        if path.is_symlink():
            out.append(f"{rel}: a symlink in the copy")
        elif path.is_file():
            found[rel] = digest_file(path)
    return out + [f"{rel}: {why}" for rel, why in _file_diff(pinned, found, "the copy")]


def digests(template: Path = TEMPLATE_DIR) -> dict:
    """SHOWCASE.json.template: {name, version, manifest_digest} (the TEMPLATE.json digest).
    While TEMPLATE.json does not exist, the digest a fresh `--write` gives is memoised per
    template root and content (see `_fresh_manifest_digest`)."""
    template = Path(template)
    pinned = read_manifest(template)
    if pinned is not None:
        return {"name": SPEC["name"], "version": pinned["version"],
                "manifest_digest": digest_file(template / MANIFEST)}
    content = "\n".join(f"{rel}\t{digest_file(template / rel)}" for rel in template_files(template))
    digest = _fresh_manifest_digest(str(template.resolve()), content, VERSION,
                                    REQUIRED_ENDPOINTS, OPTIONAL_ENDPOINTS)
    return {"name": SPEC["name"], "version": VERSION, "manifest_digest": digest}


@functools.lru_cache(maxsize=16)
def _fresh_manifest_digest(root: str, content: str, *_declared) -> str:
    """The digest of the TEMPLATE.json a fresh `--write` of `root` gives. The manifest is a
    function of the pinned files' bytes (and of the declared version and endpoint lists), so
    `content` — every pinned file with its sha256 — and `_declared` are the key: an edit of any
    file, or another template, computes it again. It is kept because computing it tokenises
    every module of the page (`endpoint_calls`) and the draft asks it once per plan; a refusal
    (TemplateError) is raised every time, never kept."""
    with tempfile.TemporaryDirectory() as scratch:
        return write_json(Path(scratch) / MANIFEST, write_manifest(Path(root)))


# ---- the command line --------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--template", type=Path, default=TEMPLATE_DIR)
    act = ap.add_mutually_exclusive_group(required=True)
    act.add_argument("--write", action="store_true")
    act.add_argument("--check", action="store_true")
    act.add_argument("--verify", type=Path, metavar="PAGE_DIR")
    act.add_argument("--digests", action="store_true")
    a = ap.parse_args(argv)
    try:
        return _run(a)
    except TemplateError as exc:
        print(f"ERROR: {exc}")
        return 2


def _run(a: argparse.Namespace) -> int:
    if a.write:
        doc = write_manifest(a.template)
        digest = write_json(a.template / MANIFEST, doc)
        print(f"OK: wrote {a.template / MANIFEST} ({len(doc['files'])} files, {digest})")
        return 0
    if a.digests:
        print(json.dumps(digests(a.template), indent=1))
        return 0
    if a.check:
        if read_manifest(a.template) is None:
            print(f"WARN: {a.template / MANIFEST} does not exist yet; identity not checked (W15 writes it)")
            return 0
        return _report(identity_problems(a.template), "PG1", "TEMPLATE.json equals write_manifest()")
    return _report(verify_template_copy(a.verify, a.template), "BD1",
                   f"{a.verify} is a byte copy of the template")


def _report(problems: list[str], gate: str, ok: str) -> int:
    for line in problems:
        print(f"ERROR: {gate} {line}")
    print(f"FAIL: {gate} {len(problems)} difference(s)" if problems else f"OK: {gate} {ok}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
