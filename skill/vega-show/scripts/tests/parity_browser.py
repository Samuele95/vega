#!/usr/bin/env python3
"""
parity_browser.py — the headless browser of the parity probe: drive a served page under
`?probe=all` and read the probe's output back (capture_parity_golden.py's driver).

    find_chrome(explicit) -> str | None       $CHROME, else google-chrome, chromium (or None)
    chrome_version(chrome) -> str
    hook_probe(page_html) -> str              a page that does not carry the probe, with it hooked in
    drive(chrome, url, workdir) -> dict       the probe's vega/page-probe@1 document, plus
                                              `browser_console_errors` read from Chrome's stderr

The browser runs with a fresh profile, reduced motion (the drawings finish at once), a 1440x900
window and a virtual-time budget of 180 s, after which Chrome dumps the DOM. The probe keeps
listening once its walk is written: an error raised later, until the budget ends, is appended to
the output's `late_errors` (js/core/probe.js).
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CHROME_CANDIDATES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
VIEWPORT = "1440,900"
BUDGET_MS = 180000
PROBE_RE = re.compile(r'<pre id="showcase-probe"[^>]*>(.*?)</pre>', re.S)
# what Chrome logs of the page's console on stderr: an uncaught error, a failed load, a refusal
CONSOLE_ERROR_RE = re.compile(r':CONSOLE[:(][^\]]*\] "((?:Uncaught|Failed to load|Refused to)[^"]*)"')
# the early hook put before a hooked page's own script: its errors are heard from the start
EARLY = """<script>
(function () {
  var t = { errors: [], started: false };
  window.__probeTrace = t;
  addEventListener("error", function (e) {
    var n = e.target;
    t.errors.push(n && n !== window && (n.src || n.href) ? { kind: "resource", text: n.src || n.href }
                                                         : { kind: "error", text: String(e.message || e.error) });
  }, true);
  addEventListener("unhandledrejection", function (e) { t.errors.push({ kind: "rejection", text: String(e.reason) }); });
  addEventListener("securitypolicyviolation", function (e) {
    t.errors.push({ kind: "csp", text: e.violatedDirective + " " + e.blockedURI });
  });
  var ce = console.error.bind(console);
  console.error = function () {
    t.errors.push({ kind: "console", text: [].slice.call(arguments).map(String).join(" ") });
    ce.apply(null, arguments);
  };
})();
</script>
"""
# the late hook: the template's probe module, served next to the hooked page at showcase/page/
LATE = ('<script type="module">import { startProbe } from "./showcase/page/js/core/probe.js"; '
        'startProbe();</script>\n')
CHARSET = '<meta charset="utf-8">\n'


def hook_probe(page: str) -> str:
    """The page with EARLY right after its <meta charset> and LATE right before </body>; the
    directory that serves it must hold js/core/probe.js."""
    if CHARSET not in page or "</body>" not in page:
        raise ValueError("the page has no <meta charset> or </body> to hook the probe to")
    page = page.replace(CHARSET, CHARSET + EARLY, 1)
    return page.replace("</body>", LATE + "</body>", 1)


def find_chrome(explicit: str | None) -> str | None:
    for c in (explicit, os.environ.get("CHROME")):
        if c and (Path(c).is_file() or shutil.which(c)):
            return c if Path(c).is_file() else shutil.which(c)
    for c in CHROME_CANDIDATES:
        if shutil.which(c):
            return shutil.which(c)
    return None


def chrome_version(chrome: str) -> str:
    try:
        return subprocess.run([chrome, "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def drive(chrome: str, url: str, workdir: Path) -> dict:
    """The page at url?probe=all in headless Chrome; the probe's vega/page-probe@1 document."""
    profile = Path(tempfile.mkdtemp(prefix="profile-", dir=workdir))
    cmd = [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-first-run",
           "--no-default-browser-check", f"--user-data-dir={profile}", "--force-prefers-reduced-motion",
           f"--window-size={VIEWPORT}", f"--virtual-time-budget={BUDGET_MS}", "--enable-logging=stderr",
           "--v=0", "--dump-dom", url + "/?probe=all"]
    env = dict(os.environ)
    if len(tempfile.gettempdir()) > 60 and Path("/tmp").is_dir():
        env["TMPDIR"] = "/tmp"      # Chrome's singleton socket path must stay under the 108-byte limit
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    console = CONSOLE_ERROR_RE.findall(proc.stderr)
    m = PROBE_RE.search(proc.stdout)
    if not m:
        shown = "\n".join(console) or proc.stderr[-2000:]
        raise SystemExit(f"ERROR: no probe output in the page (chrome exit {proc.returncode}):\n{shown}")
    probe = json.loads(html.unescape(m.group(1)))
    probe["browser_console_errors"] = console
    return probe
