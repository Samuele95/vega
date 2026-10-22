#!/usr/bin/env python3
"""
style_lock.py — the page's visual style, locked (D6 §10.7, §15.4; BUILD-PLAN R-F3, R-F10).

    python3 style_lock.py extract [REFERENCE] [--out FILE] [--chrome EXE]
    python3 style_lock.py check [TARGET] [--lock FILE] [--chrome EXE] [--static-only] [--strict]

REFERENCE and TARGET are a page (an HTML file with its <style> and <link>ed sheets), a page
template directory (its index.html and the sheets it links, `showcase/page/...` resolved inside
it), or `git:<commit>:<path>` (a page read from git: this repository, else $VEGA_SOURCE_REPO).
The reference defaults to the hand-built page of 0d06aea; the target to vega-show/assets/page.

The lock (`vega/style-lock@1`, style-lock.json) holds two layers. static: --sans, --mono, every
custom property of [data-theme="night"] and [data-theme="day"] (the --str blocks included), the
keyframes' bodies, the reduced-motion rule, the breakpoints — read from the CSS text. computed:
one headless-Chrome run at 1440 px wide, full motion, both themes, reading the computed font,
line-height, letter-spacing, colours, border-left-color, padding, margin, max-width (in ch when
a whole number of ch) and animation of D4's selectors (§13.3), each on a specimen element built
from the selector at the top of the body (style_lock_probe.js): the reading depends on the
stylesheets alone, never on the data a page shows. A target may add tokens, rules and keyframes;
it may not change or drop a locked value, and it keeps the locked breakpoints (a responsive
rule at a new width is an addition, not a change). `probe_js()` and
`compare()` give a page driver (T09's page smoke) the same reading on a live page.

Exit 0: the target holds the lock (or extract wrote it). 1: it does not. 2: the lock, the
target or the reference could not be read, or (with --strict, and always for extract) no browser.
"""
from __future__ import annotations

import argparse
import hashlib
import html as htmllib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LOCK = HERE / "style-lock.json"
TEMPLATE = REPO / "vega-show" / "assets" / "page"
REFERENCE = "git:0d06aea:vega-forge/scripts/backends/pytorch/templates/serving/static/index.html"
VIEWPORT = (1440, 900)
CHROME_CANDIDATES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
SELECTORS = ("body", "header", ".crumb", "h1.big", "h2", ".lead", ".caps", ".num", ".hint", "button.action",
             ".disclose", "nav.index button", "nav.index button.on", ".next", ".fact .v", "kbd", ".step p",
             ".rail button.on", ".guide", ".guide .n", ".card", ".choices button", ".choices button.on",
             ".wmap .rh", ".bar .name", ".bar .track", ".bar .fill", ".bar .val", ".bar.sel .fill",
             ".presets h3", ".verdict", ".agree.yes", ".agree.no", ".warn", ".formula", ".soft", ".form input",
             "table th", ".limits b", ".overlay .sheet", ".err", ".src", ".walk li button.on", ".tk-k", ".tk-s")
PROPERTIES = ("font-family", "font-size", "font-weight", "line-height", "letter-spacing", "color",
              "background-color", "border-left-color", "padding", "margin", "max-width", "animation-name",
              "animation-duration")
THEMES = ("night", "day")
SCOPES = {"root": re.compile(r"(?:html)?:root"),
          "night": re.compile(r"(?::root|html)?\[data-theme=(?:\"night\"|'night'|night)\]"),
          "day": re.compile(r"(?::root|html)?\[data-theme=(?:\"day\"|'day'|day)\]")}
PROBE_JS = (HERE / "style_lock_probe.js").read_text(encoding="utf-8")


class LockError(Exception):
    """The lock, the target or the reference cannot be read, or the browser cannot run (exit 2)."""


# ---- the CSS ----------------------------------------------------------------------------------------
def norm(text: str) -> str:
    """A CSS fragment with its insignificant whitespace removed, so equal CSS compares equal."""
    text = re.sub(r"\s+", " ", text.strip())
    return re.sub(r"\s*([{}:;,()>])\s*", r"\1", text)


def _split_top(css: str) -> list[tuple[str, str | None]]:
    """The statements of a stylesheet at one nesting level: (prelude, block or None)."""
    out, depth, quote, start, open_at = [], 0, None, 0, None
    for i, ch in enumerate(css):
        if quote:
            quote = None if ch == quote and css[i - 1] != "\\" else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "{":
            open_at = i if depth == 0 else open_at
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append((css[start:open_at].strip(), css[open_at + 1:i]))
                start = i + 1
        elif ch == ";" and depth == 0:
            out.append((css[start:i].strip(), None))
            start = i + 1
    return out


def declarations(block: str) -> list[tuple[str, str]]:
    parts, depth, cur = [], 0, []
    for ch in block:
        depth += (ch == "(") - (ch == ")")
        if ch == ";" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [(p.split(":", 1)[0].strip(), norm(p.split(":", 1)[1])) for p in parts if ":" in p]


@dataclass
class Sheet:
    """What the static layer reads from a stylesheet."""
    tokens: dict = field(default_factory=lambda: {s: {} for s in SCOPES})
    keyframes: dict = field(default_factory=dict)
    reduced_motion: list = field(default_factory=list)
    breakpoints: set = field(default_factory=set)

    @classmethod
    def parse(cls, css: str) -> "Sheet":
        sheet = cls()
        sheet._walk(re.sub(r"/\*.*?\*/", "", css, flags=re.S), "")
        return sheet

    def _walk(self, css: str, media: str) -> None:
        for prelude, block in _split_top(css):
            if block is None:
                continue
            head = norm(prelude)
            if head.startswith("@keyframes"):
                self.keyframes[head.split(None, 1)[1] if " " in head else head[10:]] = norm(block)
            elif head.startswith(("@media", "@supports")):
                self.breakpoints |= {int(float(v)) if float(v).is_integer() else float(v)
                                     for v in re.findall(r"(?:max|min)-width:(\d+(?:\.\d+)?)px", head)}
                self._walk(block, head)
            elif not head.startswith("@"):
                self._rule(head, block, media)

    def _rule(self, selector: str, block: str, media: str) -> None:
        if "prefers-reduced-motion:reduce" in media:
            self.reduced_motion.append(f"{selector}{{{norm(block)}}}")
        if media:
            return
        for sel in selector.split(","):
            for scope, rx in SCOPES.items():
                if rx.fullmatch(sel):
                    self.tokens[scope].update((n, v) for n, v in declarations(block) if n.startswith("--"))


# ---- pages: files, template directories, git ------------------------------------------------------------
def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def read_page(spec: str) -> tuple[str, Path | None, dict]:
    """(html, the directory its links resolve in, source facts) of a page, a template or git:C:P."""
    if spec.startswith("git:"):
        _, commit, path = spec.split(":", 2)
        for repo in [str(REPO)] + [r for r in [os.environ.get("VEGA_SOURCE_REPO")] if r]:
            try:
                return (_git(repo, "show", f"{commit}:{path}"), None,
                        {"commit": commit, "path": path, "blob": _git(repo, "rev-parse", f"{commit}:{path}").strip()})
            except (OSError, subprocess.CalledProcessError):
                continue
        raise LockError(f"cannot read {commit}:{path} from git (this tree, $VEGA_SOURCE_REPO)")
    target = Path(spec)
    page = target / "index.html" if target.is_dir() else target
    try:
        data = page.read_bytes()
    except OSError as exc:
        raise LockError(f"cannot read the page {page}: {exc}") from exc
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return data.decode("utf-8"), page.parent, {"path": str(page), "blob": blob}


def _resolve(base: Path, href: str) -> Path:
    parts = [p for p in href.split("?")[0].split("/") if p not in ("", ".")]
    for i in range(len(parts)):
        candidate = base.joinpath(*parts[i:])
        if candidate.is_file():
            return candidate
    raise LockError(f"the stylesheet {href} is not under {base}")


def _sheet_text(path: Path, seen: frozenset = frozenset()) -> str:
    """A stylesheet with its @imports inlined where they stand (their rules come first)."""
    if path in seen:
        raise LockError(f"@import cycle at {path}")
    text = re.sub(r"/\*.*?\*/", "", path.read_text(encoding="utf-8"), flags=re.S)
    rx = re.compile(r"@import\s+(?:url\()?[\"']?([^\"')]+)[\"']?\)?[^;]*;")
    return rx.sub(lambda m: _sheet_text(_resolve(path.parent, m.group(1)), seen | {path}), text)


def page_css(page: str, base: Path | None) -> str:
    """The page's CSS in cascade order: its <style> blocks and <link rel=stylesheet> sheets."""
    parts = []
    for m in re.finditer(r"<link\b[^>]*>|<style\b[^>]*>(.*?)</style>", page, re.S | re.I):
        tag = m.group(0)
        if not tag.lower().startswith("<link"):
            parts.append(m.group(1))
            continue
        href = re.search(r"href=[\"']([^\"']+)", tag)
        if re.search(r"rel=[\"']?stylesheet", tag, re.I) and href:
            if base is None:
                raise LockError(f"a page read from git links {href.group(1)}, which cannot be resolved")
            parts.append(_sheet_text(_resolve(base, href.group(1))))
    return "\n".join(parts)


# ---- the computed layer: one headless Chrome run -----------------------------------------------------------
def find_chrome(explicit: str | None) -> str | None:
    for c in (explicit, os.environ.get("CHROME"), *CHROME_CANDIDATES):
        if c and (Path(c).is_file() or shutil.which(c)):
            return c if Path(c).is_file() else shutil.which(c)
    return None


def probe_js(lock: dict | None = None) -> str:
    """The reading as a JS expression over a live page: the object `__styleLock` returns."""
    spec = probe_spec(lock)
    return f"(function(){{{PROBE_JS}\nreturn __styleLock({json.dumps(spec)});}})()"


def probe_spec(lock: dict | None) -> dict:
    tokens = sorted({t for scope in (lock or {}).get("tokens", {}).values() for t in scope})
    computed = (lock or {}).get("computed") or {}
    return {"selectors": list(computed.get("selectors") or SELECTORS), "themes": list(THEMES),
            "properties": list(computed.get("properties") or PROPERTIES), "tokens": tokens}


def specimen_document(page: str, css: str, spec: dict) -> str:
    """The page's stylesheets over an empty body with its <html>/<body> attributes, and the probe."""
    html_attrs, body_attrs = (next(iter(re.findall(rf"<{t}\b([^>]*)>", page, re.I)), "") for t in ("html", "body"))
    call = f"document.getElementById('style-lock-out').textContent=JSON.stringify(__styleLock({json.dumps(spec)}));"
    return (f"<!doctype html>\n<html{html_attrs}><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<style>\n{css}\n</style>"
            f"</head><body{body_attrs}><pre id=\"style-lock-out\" hidden></pre>\n<script>{PROBE_JS}\n{call}</script>"
            f"</body></html>\n")


def run_probe(page: str, css: str, spec: dict, chrome: str) -> dict:
    work = Path(tempfile.mkdtemp(prefix="style-lock-", dir="/tmp" if Path("/tmp").is_dir() else None))
    try:
        (work / "specimen.html").write_text(specimen_document(page, css, spec), encoding="utf-8")
        cmd = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run", "--hide-scrollbars",
               "--no-default-browser-check", f"--user-data-dir={work / 'profile'}",
               f"--window-size={VIEWPORT[0]},{VIEWPORT[1]}", "--virtual-time-budget=5000", "--dump-dom",
               (work / "specimen.html").as_uri()]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LockError(f"the browser did not run: {exc}") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)
    m = re.search(r'<pre id="style-lock-out"[^>]*>(.*?)</pre>', proc.stdout, re.S)
    if not m or not m.group(1).strip():
        raise LockError(f"no style reading from the browser (exit {proc.returncode}): {proc.stderr[-600:]}")
    reading = json.loads(htmllib.unescape(m.group(1)))
    if reading["viewport"][0] != VIEWPORT[0]:
        raise LockError(f"the browser's viewport is {reading['viewport']}, not {list(VIEWPORT)}")
    return reading


# ---- the comparison -------------------------------------------------------------------------------------
def check_static(lock: dict, sheet: Sheet) -> list[str]:
    out = []
    for scope, names in lock["tokens"].items():
        for name, value in names.items():
            got = sheet.tokens[scope].get(name)
            if got != value:
                out.append(f"token {name} [{scope}]: the lock holds {value}, the page "
                           + ("has none" if got is None else f"has {got}"))
    for name, body in lock["keyframes"].items():
        if sheet.keyframes.get(name) != body:
            out.append(f"@keyframes {name}: " + ("missing" if name not in sheet.keyframes else
                                                 f"the lock holds {body}, the page has {sheet.keyframes[name]}"))
    out += [f"reduced-motion rule missing: {r}" for r in lock["reduced_motion"] if r not in sheet.reduced_motion]
    missing = sorted(set(lock["breakpoints"]) - sheet.breakpoints)
    if missing:
        out.append(f"breakpoints: the lock holds {sorted(lock['breakpoints'])} px, the page has "
                   f"{sorted(sheet.breakpoints)} px (missing {missing})")
    return out


def compare(lock: dict, reading: dict, themes=THEMES) -> list[str]:
    """A probe reading (probe_js) against the lock: the computed styles and the resolved tokens."""
    want, out = lock["computed"], []
    if reading["viewport"][0] != want["captured"]["viewport"][0]:
        return [f"read at {reading['viewport'][0]} px wide; the lock was captured at {want['captured']['viewport'][0]}"]
    skip = {"animation-duration"} if reading.get("reduced_motion") != want["captured"]["reduced_motion"] else set()
    for theme in themes:
        got = reading["themes"][theme]
        for sel, props in want["themes"][theme].items():
            out += [f"{theme} {sel} {p}: the lock holds {v!r}, the page {got['styles'].get(sel, {}).get(p)!r}"
                    for p, v in props.items() if p not in skip and got["styles"].get(sel, {}).get(p) != v]
        for scope in ("root", theme):
            out += [f"{theme} token {n}: the lock holds {v}, the page resolves {norm(got['tokens'].get(n, ''))!r}"
                    for n, v in lock["tokens"][scope].items() if norm(got["tokens"].get(n, "")) != v]
    return out


# ---- the two commands ---------------------------------------------------------------------------------------
def extract(reference: str, chrome: str) -> dict:
    page, base, source = read_page(reference)
    sheet = Sheet.parse(page_css(page, base))
    lock = {"schema": "vega/style-lock@1",
            "about": "The page's locked style (D6 §10.7, D4 §13.1-13.3, BUILD-PLAN R-F10): written by "
                     "vega/scripts/tests/style_lock.py extract, checked by style_lock.py check.",
            "source": source, "tokens": sheet.tokens, "keyframes": sheet.keyframes,
            "reduced_motion": sheet.reduced_motion, "breakpoints": sorted(sheet.breakpoints)}
    reading = run_probe(page, page_css(page, base), probe_spec(lock), chrome)
    version = subprocess.run([chrome, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    captured = {"chrome": version, "viewport": reading["viewport"], "reduced_motion": reading["reduced_motion"],
                "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "specimen": "top of the body"}
    lock["computed"] = {"captured": captured, "selectors": list(SELECTORS), "properties": list(PROPERTIES),
                        "themes": {t: reading["themes"][t]["styles"] for t in THEMES}}
    drift = compare(lock, reading)
    if drift:
        raise LockError("the reference's resolved tokens differ from its CSS text: " + "; ".join(drift[:5]))
    return lock


def check(target: str, lock: dict, chrome: str | None) -> list[str]:
    """Every way the target departs from the lock (the computed layer only with a browser)."""
    page, base = read_page(target)[:2]
    css = page_css(page, base)
    problems = check_static(lock, Sheet.parse(css))
    if chrome:
        problems += compare(lock, run_probe(page, css, probe_spec(lock), chrome))
    return problems


def load_lock(path: Path = LOCK) -> dict:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LockError(f"cannot read the lock {path}: {exc}") from exc
    if lock.get("schema") != "vega/style-lock@1":
        raise LockError(f"{path} is not vega/style-lock@1")
    return lock


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="write the lock from the reference page")
    ex.add_argument("reference", nargs="?", default=REFERENCE)
    ex.add_argument("--out", type=Path, default=None, help="the lock file (default: stdout)")
    ck = sub.add_parser("check", help="compare a page or the page template with the lock")
    ck.add_argument("target", nargs="?", default=str(TEMPLATE))
    ck.add_argument("--lock", type=Path, default=LOCK)
    ck.add_argument("--static-only", action="store_true", help="the static layer only (no browser)")
    ck.add_argument("--strict", action="store_true", help="a missing browser is an error, not a WARN")
    for p in (ex, ck):
        p.add_argument("--chrome", default=None, help="the headless browser (default: $CHROME, google-chrome, chromium)")
    a = ap.parse_args(argv)
    try:
        return _extract(a) if a.cmd == "extract" else _check(a)
    except LockError as exc:
        print(f"ERROR: {exc}")
        return 2


def _extract(a: argparse.Namespace) -> int:
    chrome = find_chrome(a.chrome)
    if not chrome:
        raise LockError("extract needs a headless browser for the computed layer ($CHROME, google-chrome, chromium)")
    text = json.dumps(extract(a.reference, chrome), indent=1, ensure_ascii=False) + "\n"
    if a.out is None:
        sys.stdout.write(text)
    else:
        a.out.write_text(text, encoding="utf-8")
        print(f"OK: the style lock written to {a.out}")
    return 0


def _check(a: argparse.Namespace) -> int:
    lock, chrome = load_lock(a.lock), None if a.static_only else find_chrome(a.chrome)
    if not a.static_only and not chrome:
        if a.strict:
            raise LockError("no headless browser for the computed layer ($CHROME, google-chrome, chromium)")
        print("WARN: no headless browser: the computed layer is not checked, the static layer is")
    problems = check(a.target, lock, chrome)
    for p in problems:
        print(f"FAIL: {p}")
    layers = "static and computed" if chrome else "static"
    if problems:
        print(f"FAIL: {a.target} departs from the style lock in {len(problems)} value(s) ({layers} layers)")
        return 1
    n = sum(len(v) for v in lock["tokens"].values())
    print(f"OK: {a.target} holds the style lock ({layers} layers: {n} tokens, {len(lock['keyframes'])} keyframes"
          + (f", {len(lock['computed']['selectors'])} selectors x {len(THEMES)} themes)" if chrome else ")"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
