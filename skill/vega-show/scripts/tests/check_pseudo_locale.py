#!/usr/bin/env python3
"""
check_pseudo_locale.py — every visible chrome word of the page template comes from its strings
table: the qps-ploc pseudo-locale on the CIC R7 bundle.

    python3 check_pseudo_locale.py [--template DIR] [--upstream URL | --bundle DIR]
                                   [--python EXE] [--chrome EXE] [--strict] [--keep]

Serves the template (vega-show/assets/page by default) through parity_server.py in front of R7's
own service, drives it in headless Chrome at `?probe=all&locale=qps-ploc` (the probe walks every
view, opens the sub-views, answers every preset and steps through "how the network computed it"),
and reads the DOM the walk leaves behind. Under qps-ploc the page wraps each run of a strings-table
template's own text ⟦…⟧, and each lexicon word and each string the dataset, the service or the
bundle brings ‹…›; a parameter keeps the marks it came with, so a word passed into a table string
unwrapped is seen too (core/text.js). The
story's own prose is wrapped ⟪…⟫ by whoever builds the showcase; R7 serves a story@1 that no
showcase build wrapped, so this check plays the builder: the story reaches the page through a small
proxy that wraps each authored string ⟪…⟫ (ids, value keys, patterns and report file fields are not
prose and stay as they are).

Every text node of the body, in document order, is read as one stream; the wrapped runs are taken
out (nested runs included); what is left must hold no letter. Two exemptions, as D6 §10.5 has them
and one more: the code view's source panel (#src, files shown verbatim and verified by digest),
text drawn on canvases (the DOM does not hold it), and the tour's keycaps (a .k element holding one
character: core/tour.js draws the index's and the tabs' keys there, digits and the off-tour view's
key letter, which main.js names; D6 §10.5 does not grant this one, and the page stage's qps check,
T09's, settles it: a key to press is not a word to translate, so a .k element holding one
character stays exempt, and two or more letters there are still read). A <kbd> keycap is not
exempt: its letter comes from the strings table.
The page must also be clean (booted, every view rendered, no page error, no HTTP error) and its
probe must report no missing key (`missingKeys`) and no unresolved placeholder
(`unresolvedPlaceholders`). Each word left over is reported with the CSS path of the node that
holds it.

What it cannot see: only the DOM the probe's walk leaves behind is read. Text the page shows on
demand and the walk never asks for (the network view's unit, class and family cards, the hover tips
of the weights map and the layer maps, a refusal's problem text) is not drawn, so a literal there
passes this check. The static half covers those paths: D6 §15.3's chrome-string check and W07's
page_lint literals (D4 §17.1(2)) refuse a string literal with words outside the strings table.
The page lint excludes js/legacy/** while it exists (build plan §2.7), so in the legacy views these
paths are guarded only by their review until W05, W08, W09 and W13 replace them.

This check stays R7's (`cic_fixture.py --qps`); the lanes' page stage (T09,
vega/scripts/tests/page_smoke.py, check P-QPS, VS26) reads each lane's own qps-ploc build, the
story wrapped by build_showcase, through `wrapped_runs`.

The upstream: --upstream when R7's service runs already, otherwise the bundle in --bundle (default
$CIDS_ROOT/bundle/cicids2017-label-R7) started read-only on a free port with --python.

Exit 0: every chrome word wrapped (or, without --strict, no browser or no bundle: WARN). 1: a word
unwrapped, a missing key or an unclean run. 2: the check could not run (under --strict).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unicodedata
import urllib.error
import urllib.request
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capture_parity_golden as golden  # noqa: E402
import parity_browser  # noqa: E402
import parity_server  # noqa: E402

LOCALE = "qps-ploc"
# the runs and their brackets: chrome (the strings table), data (dataset, service, bundle), prose
# (the story, as the showcase build wraps it)
RUNS = {"⟦": "⟧", "‹": "›", "⟪": "⟫"}
CLOSERS = {v: k for k, v in RUNS.items()}
STORY_WRAP = ("⟪", "⟫")
# story@1 fields that are not prose: ids and value keys, input-group patterns, the neutral class (a
# class name the page compares with), the report's file fields, a code note's page expression
STORY_KEEP = frozenset({"schema", "id", "value", "values", "names", "focus", "match", "neutral_class",
                        "page", "report", "file", "pdf_digest", "aux_digest", "selected", "reference"})
# a number in scientific notation (5.33e+4, 2.4E-05): its exponent's letter is notation, not a word
SCIENTIFIC = re.compile(r"(?<![\w.])[-+\u2212]?\d+(?:\.\d+)?[eE][-+\u2212]?\d+(?!\w|\.\d)")
SKIP_TAGS = frozenset({"script", "style", "template", "head", "title", "canvas"})
VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"})


def say(level: str, msg: str):
    print(f"{level}: {msg}", flush=True)


# ---- the builder's part: the story's prose wrapped ⟪…⟫ -------------------------------------------
def wrap_story(node, key=None):
    """The story with every authored string wrapped; the fields that are not prose kept."""
    if key in STORY_KEEP:
        return node
    if isinstance(node, str):
        return f"{STORY_WRAP[0]}{node}{STORY_WRAP[1]}" if node else node
    if isinstance(node, list):
        return [wrap_story(v) for v in node]
    if isinstance(node, dict):
        return {k: wrap_story(v, k) for k, v in node.items()}
    return node


class _StoryProxy(BaseHTTPRequestHandler):
    """R7's service, with GET /v1/story answered by the wrapped story; every other request forwarded."""

    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._forward()

    def do_POST(self):
        self._forward()

    def _forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(self.server.upstream + self.path, data=body, method=self.command)
        for h in ("content-type", "accept"):
            if self.headers.get(h):
                req.add_header(h, self.headers[h])
        try:
            with parity_server._OPENER.open(req, timeout=parity_server.TIMEOUT) as r:
                status, ctype, out = r.status, r.headers.get("Content-Type", "application/json"), r.read()
        except (urllib.error.URLError, OSError) as exc:
            status, ctype, out = 502, "application/json", json.dumps({"title": "upstream unreachable", "detail": str(exc)}).encode()
        if urlsplit(self.path).path == "/v1/story" and status == 200:
            out = json.dumps(wrap_story(json.loads(out)), ensure_ascii=False).encode("utf-8")
            self.server.story_wrapped = True
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, fmt, *args):
        pass


class StoryProxy:
    def __init__(self, upstream: str):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StoryProxy)
        self.httpd.daemon_threads = True
        self.httpd.upstream = upstream.rstrip("/")
        self.httpd.story_wrapped = False
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="story-proxy", daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]

    @property
    def story_wrapped(self) -> bool:
        return self.httpd.story_wrapped

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=10)


# ---- the browser: the probe's output and the DOM it leaves --------------------------------------
def drive(chrome: str, url: str, workdir: Path) -> tuple[dict, str]:
    profile = Path(tempfile.mkdtemp(prefix="profile-", dir=workdir))
    cmd = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run",
           "--no-default-browser-check", f"--user-data-dir={profile}", "--force-prefers-reduced-motion",
           f"--window-size={parity_browser.VIEWPORT}", f"--virtual-time-budget={parity_browser.BUDGET_MS}",
           "--enable-logging=stderr", "--v=0", "--dump-dom", f"{url}/?probe=all&locale={LOCALE}"]
    env = dict(os.environ)
    if len(tempfile.gettempdir()) > 60 and Path("/tmp").is_dir():
        env["TMPDIR"] = "/tmp"      # Chrome's singleton socket path must stay under the 108-byte limit
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    console = parity_browser.CONSOLE_ERROR_RE.findall(proc.stderr)
    m = parity_browser.PROBE_RE.search(proc.stdout)
    if not m:
        raise SystemExit(f"ERROR: no probe output in the page (chrome exit {proc.returncode}):\n"
                         + ("\n".join(console) or proc.stderr[-2000:]))
    probe = json.loads(html.unescape(m.group(1)))
    probe["browser_console_errors"] = console
    (workdir / "dom.html").write_text(proc.stdout, encoding="utf-8")
    return probe, proc.stdout


# ---- the DOM: text nodes in order, with the path of the element that holds each -----------------
class _Texts(HTMLParser):
    """Collects (text, css path) of every text node of the body outside the exempt subtrees."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.out = [], []

    def _exempt(self) -> bool:
        for tag, attrs in self.stack:
            if tag in SKIP_TAGS or attrs.get("id") in ("src", "showcase-probe"):
                return True
        return not any(tag == "body" for tag, _ in self.stack)

    def handle_starttag(self, tag, attrs):
        if tag in VOID:
            return
        self.stack.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if not data or self._exempt():
            return
        _, attrs = self.stack[-1]
        if "k" in (attrs.get("class") or "").split() and len(data.strip()) == 1:
            data = " "                                   # the tour's keycap: the key to press
        self.out.append((data, self.path()))

    def path(self) -> str:
        parts = []
        for tag, attrs in self.stack:
            if attrs.get("id"):
                parts = [f"{tag}#{attrs['id']}"]
            else:
                cls = ".".join((attrs.get("class") or "").split())
                parts.append(tag + (f".{cls}" if cls else ""))
        return " > ".join(parts)


def unwrapped(segments: list[tuple[str, str]]) -> tuple[list[dict], dict]:
    """The letters no run covers, grouped by the node that holds them; and the runs counted."""
    stream = "".join(text for text, _ in segments)
    covered = bytearray(len(stream))
    stack, counts = [], {k: 0 for k in RUNS}
    for i, ch in enumerate(stream):
        if ch in RUNS:
            stack.append((ch, i))
        elif ch in CLOSERS:
            for depth in range(len(stack) - 1, -1, -1):          # the innermost open run of that kind
                if stack[depth][0] == CLOSERS[ch]:
                    _, start = stack[depth]
                    del stack[depth:]
                    covered[start:i + 1] = b"\x01" * (i + 1 - start)
                    counts[CLOSERS[ch]] += 1
                    break
    found, at = [], 0
    for text, path in segments:
        left = "".join(ch if not covered[at + k] else " " for k, ch in enumerate(text))
        at += len(text)
        words = re.findall(r"\S*[^\W\d_]\S*", SCIENTIFIC.sub(" ", left))
        words = [w for w in words if any(unicodedata.category(c).startswith("L") for c in w)]
        if words:
            found.append({"path": path, "words": words, "text": text.strip()[:120]})
    return found, counts


def wrapped_runs(dom: str) -> tuple[list[dict], dict]:
    """The words of a walked page's DOM no run covers ({path, words, text} per node, the exempt
    subtrees left out), and the runs counted by kind. The lanes' page stage reads its pseudo-locale
    run through this too (vega/scripts/tests/page_smoke.py, P-QPS)."""
    parser = _Texts()
    parser.feed(dom)
    return unwrapped(parser.out)


def check(probe: dict, dom: str, requests: list, origin: str, story_wrapped: bool) -> tuple[list[str], dict]:
    """The problems of one run, and the wrapped runs counted by kind."""
    problems = golden.clean(probe, requests, origin)
    ext = probe.get("extensions") or {}
    for name in ("missingKeys", "unresolvedPlaceholders"):
        if name not in ext:
            problems.append(f"the probe does not report {name} (the page did not register it)")
        elif ext[name]:
            problems.append(f"{name}: {ext[name]}")
    if not story_wrapped:
        problems.append("the page never read v1/story: the story's prose was not wrapped")
    found, counts = wrapped_runs(dom)
    for run, n in counts.items():
        if not n:
            problems.append(f"no {run}…{RUNS[run]} run in the page: the pseudo-locale did not apply")
    for f in found:
        problems.append(f"unwrapped text in {f['path']}: {f['words']} (in {f['text']!r})")
    return problems, counts


def serve_and_drive(a: argparse.Namespace, workdir: Path) -> tuple[dict, str, list, str, bool]:
    """R7's service (reached or started), the story proxy and the template server; the page driven
    once. Returns the probe, the DOM, the requests served, the page's origin and whether the story
    was wrapped."""
    chrome = parity_browser.find_chrome(a.chrome)
    if not chrome:
        raise golden.Unavailable(f"no headless browser (checked $CHROME, {', '.join(parity_browser.CHROME_CANDIDATES)})")
    upstream = golden.Upstream(a.upstream, a.bundle, a.python, workdir)
    proxy = srv = None
    try:
        upstream.wait_ready()
        proxy = StoryProxy(upstream.url)
        srv = parity_server.start(a.template, proxy.url)
        probe, dom = drive(chrome, srv.url, workdir)
        return probe, dom, srv.requests, srv.url, proxy.story_wrapped
    finally:
        for part in (srv, proxy, upstream):
            if part is not None:
                part.stop()


def report(problems: list[str], counts: dict, views: int, workdir: Path) -> int:
    for p in problems:
        say("FAIL", p)
    if problems:
        say("FAIL", f"pseudo-locale on R7: {len(problems)} problem(s) (DOM and probe kept in {workdir})")
        return 1
    say("OK", f"pseudo-locale on R7: every visible chrome word of the walked DOM wrapped ({counts['⟦']} chrome, "
              f"{counts['‹']} data, {counts['⟪']} story runs), {views} views rendered, no missing key, no unresolved "
              "placeholder, no page error")
    return 0


def run(a: argparse.Namespace) -> int:
    workdir = Path(tempfile.mkdtemp(prefix="vega-qps-"))
    try:
        probe, dom, requests, origin, story_wrapped = serve_and_drive(a, workdir)
        (workdir / "probe.json").write_text(json.dumps(probe, indent=1, ensure_ascii=False), encoding="utf-8")
        problems, counts = check(probe, dom, requests, origin, story_wrapped)
        rc = report(problems, counts, len(probe.get("views") or {}), workdir)
        a.keep = a.keep or rc != 0
        return rc
    except golden.Unavailable as exc:
        say("ERROR" if a.strict else "WARN", f"{exc}; pseudo-locale not checked")
        return 2 if a.strict else 0
    finally:
        if a.keep:
            say("INFO", f"work directory kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--template", type=Path, default=golden.TEMPLATE)
    ap.add_argument("--upstream", default=None, help="R7's service, when it runs already")
    ap.add_argument("--bundle", type=Path, default=golden.BUNDLE,
                    help="R7's bundle, started when --upstream is absent (default: $CIDS_ROOT/bundle/cicids2017-label-R7)")
    ap.add_argument("--python", default=sys.executable, help="the interpreter that runs the bundle's service")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--keep", action="store_true", help="keep the work directory (DOM, probe output, service log)")
    ap.add_argument("--strict", action="store_true", help="a missing browser, bundle or CIDS_ROOT is an error, not a WARN")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
