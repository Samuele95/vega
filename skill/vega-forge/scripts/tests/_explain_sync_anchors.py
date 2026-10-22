"""
_explain_sync_anchors.py — the code-anchor side of `test_explain_catalog_sync.py`.

An anchor is resolved the way the page's code walk resolves it (`resolveWalk` in
vega-show/assets/page/js/views/code.js): from the first line matching `after` (or the top),
the first alternative of `at` that matches a line. The block anchors are resolved against the
segment files `blocks.py` is cut into (`# --8<-- name` markers, `SEGMENT_FILE`), as a rendered
package holds them; the fixed walk's file patterns against the package paths each step must
find. Each `check_*` function records its findings on the group's `Check`.

The leading underscore keeps the unit tier's `test_*.py` discovery from collecting it.
"""
from __future__ import annotations

import re

from _explain_sync_common import BLOCKS_PY, blocks_module
from _explain_sync_facts import CUSTOM_BUILD_LINE, CUSTOM_PATH, CUSTOM_TEXT, WALK_PATHS


def resolve(code, text):
    """The line index the page's code walk lands on in `text` for this anchor (resolveWalk:
    from the first `after` line, the first `at` alternative that matches), or None."""
    lines = text.split("\n")
    start = 0
    if code.get("after"):
        after = re.compile(code["after"])
        start = next((i for i, line in enumerate(lines) if after.search(line)), -1)
        if start < 0:
            return None
    for pattern in code["at"]:
        rx = re.compile(pattern)
        hit = next((i for i in range(start, len(lines)) if rx.search(lines[i])), None)
        if hit is not None:
            return hit
    return None


def segment_texts():
    """{segment: source text} of blocks.py, cut at its `# --8<-- name` / `# -->8-- name`
    markers exactly as the emitter splices them."""
    src = BLOCKS_PY.read_text(encoding="utf-8")
    found = re.findall(r"^# --8<-- (\w+)\n(.*?)^# -->8-- \1$", src, re.M | re.S)
    return {name: body for name, body in found}


def segment_files(segments, segment_file):
    """({file stem: its text}, {file stem: its package path}): the segments joined into the
    files `segment_file` (blocks.SEGMENT_FILE) sends them to, as a rendered package holds them."""
    parts = {}
    for name, body in segments.items():
        parts.setdefault(segment_file.get(name, "?"), []).append(body)
    files = {f: "\n".join(bodies) for f, bodies in parts.items()}
    return files, {f: f"pkg/model/{f}.py" for f in files}


def segment_of(segments, cls_name):
    """The segment of blocks.py that defines class `cls_name`, or None."""
    rx = re.compile(rf"^class {re.escape(cls_name)}\b", re.M)
    return next((n for n, body in segments.items() if rx.search(body)), None)


def check_walk_anchors(c, by_id, anchors):
    """The fixed steps `by_id` agree with the anchors the vega-show catalog states
    (`walk.anchors`: "<file> <pattern>" per step id; `blocks` is the per-type rule)."""
    for sid, stated in anchors.items():
        if sid == "blocks":
            continue
        filename, _, pattern = stated.partition(" ")
        step = by_id.get(sid)
        if not c.expect(step is not None, f"vega-show catalog walk.anchors names {sid!r}, "
                                          "which is no fixed step"):
            continue
        c.expect(re.search(step["code"]["file"], f"pkg/model/{filename}")
                 or re.search(step["code"]["file"], filename),
                 f"walk_fixed.{sid}.code.file {step['code']['file']!r} misses "
                 f"{filename!r} (vega-show catalog walk.anchors)")
        c.expect(step["code"]["at"][0] == pattern,
                 f"walk_fixed.{sid}.code.at[0] {step['code']['at'][0]!r} != vega-show catalog "
                 f"walk.anchors {pattern!r}")


def check_walk_paths(c, by_id):
    """Each fixed step's file pattern finds the package paths it must and none it must not."""
    for sid, (hits, misses) in WALK_PATHS.items():
        step = by_id.get(sid)
        if step is None:
            continue
        rx = re.compile(step["code"]["file"])
        for path in hits:
            c.expect(rx.search(path), f"walk_fixed.{sid}.code.file misses {path}")
        for path in misses:
            c.expect(not rx.search(path), f"walk_fixed.{sid}.code.file also finds {path}")


def check_block_anchor(c, where, code, seg, target):
    """One block anchor: its file pattern finds the file of segment `seg` and no other, and
    resolves there on the class's own line. `target` is (class name, {file stem: text},
    {file stem: package path})."""
    cls_name, files, paths = target
    if not c.expect(seg is not None, f"{where}: class {cls_name} is in no blocks.py segment"):
        return
    home = blocks_module().SEGMENT_FILE[seg]
    rx = re.compile(code["file"])
    c.expect(rx.search(paths[home]),
             f"{where}.code.file {code['file']!r} misses {paths[home]} (segment {seg!r}, "
             "SEGMENT_FILE)")
    for other, path in paths.items():
        if other != home:
            c.expect(not rx.search(path), f"{where}.code.file {code['file']!r} also finds {path}")
    line = resolve(code, files[home])
    if not c.expect(line is not None, f"{where}.code finds nothing in {home}.py"):
        return
    found = files[home].split("\n")[line]
    c.expect(re.match(rf"class {re.escape(cls_name)}\b", found),
             f"{where}.code lands on {found.strip()!r} in {home}.py, not on class {cls_name}")


def check_custom_anchor(c, code, paths):
    """A custom block's anchor finds its copied module and no block file, and lands on the
    module's `build` (R-B9, the entry point netspec-contract names)."""
    rx = re.compile(code["file"])
    c.expect(rx.search(CUSTOM_PATH), f"blocks.custom.code.file misses the copied module "
                                     f"{CUSTOM_PATH} (R-B9)")
    for path in paths.values():
        c.expect(not rx.search(path), f"blocks.custom.code.file also finds {path}")
    c.expect(resolve(code, CUSTOM_TEXT) == CUSTOM_BUILD_LINE,
             "blocks.custom.code.at misses `def build(` — the entry point netspec-contract "
             "names for a custom module")
