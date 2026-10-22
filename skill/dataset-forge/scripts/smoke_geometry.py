#!/usr/bin/env python3
"""
smoke_geometry.py — headless-Chromium check that the /dataset-forge `geometry` layer renders
and that its explorer moves.

    python3 smoke_geometry.py OUT.html [--strict] [--json REPORT.json]

The render script runs only in a browser, so a copy of OUT.html is opened in headless Chromium
with a probe appended that drives the page the way an analyst would and writes what it saw into
a <pre id="geometry-smoke">; the DOM dump is then parsed and held against the layer's own data.
Checks (report-contract §6):

  1. the tab button (data-tab="geometry") sits in the host's nav.tabs / nav[role=tablist], the
     pane section.tab-pane[data-layer="geometry"] is mounted, hidden until the tab is clicked;
  2. the four views exist: data-view = space | graph | orthogonality | partition;
  3. View A draws one scree bar per singular value and one point per sample row; the PCA
     toggle relabels the axes as rotated and shows the loadings, and toggles back; hovering a
     point fills the row panel with its derived values and their rules; brushing a rectangle
     selects some rows (not none, not all), the selection banner appears and clears;
  4. View B draws one node per column, one edge per kept rule's body column, the cycles, and
     a basis-member count equal to the report's; a derived node's card shows the provenance
     triple; in a cycle, "make this the basis member" re-orients it — the basis gains the
     column and loses the orientation's derived members, the orientation's own pre-written
     consequence block is shown, the partition view's rule heads update — and "restore the
     report's orientation" brings the report's basis back;
  5. View C draws one cell per ordered pair of measured basis members; clicking a cell makes
     the pair View A's axes;
  6. View D draws one card per candidate; selecting one strikes exactly its leakage set on the
     graph, marks exactly its features and its label, shows its input_dim and colours View A
     by the label;
  7. no uncaught JS error (window.onerror, console.error, Chromium stderr);
  8. on a copy without the explore payload, Views A and C render their empty state and
     Views B and D still work.

Exit 0 on success, 1 on a failed check (or, with --strict, when no browser is available);
without --strict a missing browser prints WARN and exits 0 — the render is then unverified,
and the summary says so. Markers: OK: / WARN: / ERROR:.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True               # the siblings below leave no __pycache__ in the skill
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from smoke_geometry_checks import Checks, check_copy, expectations  # noqa: E402
from smoke_geometry_probe import PROBE  # noqa: E402

CHROME_CANDIDATES = ("google-chrome", "chromium", "chromium-browser", "chrome")
DATA_RE = re.compile(r'(<script\s+id="layer-geometry-data"[^>]*>)([\s\S]*?)(</script>)', re.I)
PRE_RE = re.compile(r'<pre id="geometry-smoke">([\s\S]*?)</pre>')


def find_chrome():
    env = os.environ.get("CHROME")
    if env and (Path(env).is_file() or shutil.which(env)):
        return env if Path(env).is_file() else shutil.which(env)
    for c in CHROME_CANDIDATES:
        if shutil.which(c):
            return shutil.which(c)
    return None


def dump_dom(chrome, path, budget_ms=8000):
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.run(
            [chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--window-size=1280,2400",
             f"--user-data-dir={profile}", f"--virtual-time-budget={budget_ms}",
             "--enable-logging=stderr", "--v=0", "--dump-dom", f"file://{path.resolve()}"],
            capture_output=True, text=True, timeout=120)
    return proc.stdout, proc.stderr


def layer_data(html):
    m = DATA_RE.search(html)
    if not m:
        return None, None
    try:
        return json.loads(m.group(2)), m
    except json.JSONDecodeError:
        return None, m


def probe_copy(html, tmpdir, name, target, strip_explore=False):
    if strip_explore:
        doc, m = layer_data(html)
        if doc is not None:
            doc.pop("explore", None)
            text = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
            html = html[:m.start(2)] + "\n" + text + "\n" + html[m.end(2):]
    probe = PROBE.replace("__TARGET__", json.dumps(target))
    idx = html.rfind("</body>")
    html = html[:idx] + probe + html[idx:] if idx >= 0 else html + probe
    p = Path(tmpdir) / name
    p.write_text(html, encoding="utf-8")
    return p


def parse_results(dom):
    m = PRE_RE.search(dom)
    if not m:
        return None
    txt = (m.group(1).replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")
           .replace("&amp;", "&"))
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return None


def _probe(chrome, html, tmpdir, label, target):
    """(the probe's report, Chromium's stderr, None) for one copy — or (None, None, why)."""
    p = probe_copy(html, tmpdir, f"smoke-{label}.html", target, strip_explore=label != "full")
    try:
        dom, err = dump_dom(chrome, p)
    except subprocess.TimeoutExpired:
        return None, None, "render: chromium timed out"
    R = parse_results(dom)
    return (R, err, None) if R is not None else (None, None, "probe: no probe output captured")


def _load(path):
    """(html, layer data) or an exit code."""
    if not path.is_file():
        print(f"ERROR: {path} not found")
        return 1
    html = path.read_text(encoding="utf-8")
    doc, _ = layer_data(html)
    if doc is None:
        print("ERROR: no parseable layer-geometry-data script in the file")
        return 1
    return html, doc


def _verdict(check, path, X):
    if check.failures:
        print(f"ERROR: geometry layer smoke test failed for {path} "
              f"({len(check.failures)} of {check.count} checks)")
        return 1
    moved = ("re-orienting a cycle updates the basis and the candidates" if X["cycles"]
             else "no cycle to re-orient")
    print(f"OK: geometry layer smoke test passed for {path} ({check.count} checks; the four views "
          f"mount, {moved}, no JS error{'' if X['explore'] else '; no explore payload: A and C empty'})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("html", help="layered HTML output to check")
    ap.add_argument("--strict", action="store_true", help="exit 1 when no browser is available")
    ap.add_argument("--json", default=None, help="write the probe's observations here")
    a = ap.parse_args(argv)
    path = Path(a.html)
    got = _load(path)
    if isinstance(got, int):
        return got
    html, doc = got
    chrome = find_chrome()
    if not chrome:
        msg = f"no headless browser available (checked $CHROME and {', '.join(CHROME_CANDIDATES)})"
        print(f"ERROR: {msg} — failing under --strict" if a.strict
              else f"WARN: {msg}; the geometry explorer is unverified")
        return 1 if a.strict else 0
    X = expectations(doc)
    target = {"derived": X["target"], "candidate": X["candidate"]}
    check, report = Checks(), {}
    with tempfile.TemporaryDirectory() as td:
        for label in ("full", "no-explore"):
            R, err, why = _probe(chrome, html, td, label, target)
            if R is None:
                check(False, f"{label}:{why.split(':')[0]}", why.split(": ", 1)[1])
                continue
            report[label] = R
            check_copy(check, label, R, err, X)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return _verdict(check, path, X)


if __name__ == "__main__":
    sys.exit(main())
