"""
_negatives_bundle — `run_negatives.py`'s cases on what the stage reads and what it builds: the
preflight (PF1–PF9: the bundle, its MANIFEST, the model page's bundle layer, the report), the
build (BD1–BD8) and the check (CK1–CK10), each seeded into the staged lane and run through
`extract_facts.py` or `build_showcase.py` as the stage runs them. A gate that guards the build's
own invariants (BD5, BD6, BD8) is seeded in the build's code path, in a child that runs the real
command line with one function made faulty (`seeded_build`).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from _negatives_kit import (Check, Expect, case, edit_json, edit_text, py, read_json, refused, sha, show,
                            template_copy)

BASE = "L02"


def off_by_one(digest: str) -> str:
    return digest[:-1] + ("0" if digest[-1] != "0" else "1")


def restage(w) -> None:
    """The stage again from the facts on (a MANIFEST edit the preflight lets pass changes the
    bundle the fact set must describe)."""
    w.stage(steps=("facts", "draft", "author", "resolve"))


def seeded_build(w, patch: str):
    """`build_showcase.py` run as the stage runs it, in a child whose build has one seeded fault:
    `patch` is Python run after `import build_showcase as b, showcase_manifest as m`."""
    argv = ["--bundle", w.bundle, "--model", w.model, "--plan", w.plan / "story-plan.json",
            "--story", w.plan / "story.json", "--facts", w.plan / "facts.json", *w.report(),
            "--out", w.showcase, "--unattended"]
    code = ("import build_showcase as b, showcase_manifest as m\nfrom pathlib import Path\n" + patch +
            "\nsys.exit(b.main(sys.argv[1:]))\n")
    return py(code, *argv, cwd=w.work)


# =============================================================================== preflight

@case("PF1", "D5-23", BASE, "the model page's bundle layer describes another manifest than MANIFEST.json")
def pf1_layer(w):
    w.edit_layer(lambda data: data["manifest"].update(note="another bundle"))
    return refused(w.facts(), "PF1", "bundle layer")


@case("PF1", "VS18", BASE, "the bundle's MANIFEST.json altered after pack (not the pack layer's)")
def pf1_manifest(w):
    edit_json(w.manifest, lambda m: m.update(note="edited after pack"))
    return refused(w.facts(), "PF1", "MANIFEST", "pack")


@case("PF2", "D5-22", BASE, "a pinned record altered in the bundle")
def pf2(w):
    record = w.bundle / "records" / "evaluation.json"
    record.write_bytes(record.read_bytes() + b"\n")
    return refused(w.facts(), "PF2", "records/evaluation.json")


@case("PF3", "", BASE, "a predict.py whose --check fails from a clean shell")
def pf3(w):
    script = w.bundle / "predict.py"
    script.write_text('raise SystemExit("seeded: the check fails")\n' + script.read_text(encoding="utf-8"),
                      encoding="utf-8")
    return refused(w.facts(), "PF3", "--check")


@case("PF4", "D5-24", BASE, "a bundle without serve.showcase_slot")
def pf4_slot(w):
    w.relayer(lambda m: m["serve"].pop("showcase_slot"))
    return refused(w.facts(), "PF4", "--serve")


@case("PF4", "VS24", BASE, "a bundle packed without a service (serve.present false)")
def pf4_no_service(w):
    w.relayer(lambda m: m["serve"].update(present=False))
    return refused(w.facts(), "PF4", "--serve")


@case("PF5", "", BASE, "a bundle whose service lacks an endpoint the page requires")
def pf5(w):
    w.relayer(lambda m: m["serve"].update(endpoints=[e for e in m["serve"]["endpoints"] if e != "GET /v1/network"]))
    restage(w)
    return refused(w.build(), "PF5", "GET /v1/network")


@case("PF6", "D5-29", BASE, "a label-role map older than its .aux")
def pf6_map(w):
    edit_json(w.docs / "label-roles.json", lambda d: d["report"].update(aux_digest="sha256:" + "0" * 64))
    return refused(w.facts(), "PF6", "recompile /document-project")


@case("PF6", "VS15", BASE, "the report recompiled after its label-role map (the .aux moved on)")
def pf6_aux(w):
    aux = w.docs / "main.aux"
    aux.write_text(aux.read_text(encoding="utf-8") + "\\relax\n", encoding="utf-8")
    return refused(w.facts(), "PF6", "label-roles")


def extract_past_pf3(w):
    """`extract_facts.py` in a child whose predict check (PF3) is held as passed: a netspec
    seeded with what the catalog does not know also breaks the bundle's own predictor, which
    PF3 would refuse first; PF7 must refuse it on its own (an ERROR, never a skipped fact)."""
    argv = ["--model", w.model, "--bundle", w.bundle, *w.report(), "--out", w.plan / "facts.json"]
    return py("import extract_facts as e\n"
              "e._predict_check = lambda bundle: 'OK: PF3 held aside (the seed is the netspec)'\n"
              "sys.exit(e.main(sys.argv[1:]))\n", *argv, cwd=w.work)


@case("PF7", "D5-36", BASE, "an unknown family in the netspec")
def pf7_family(w):
    edit_json(w.bundle / "netspec.json", lambda n: n["family"].update(name="hypernet"))
    w.repin_bundle("netspec.json")
    return refused(extract_past_pf3(w), "PF7", "hypernet")


@case("PF7", "D5-37", BASE, "an unknown task kind in the netspec")
def pf7_task(w):
    edit_json(w.bundle / "netspec.json", lambda n: n["task"].update(kind="clustering"))
    w.repin_bundle("netspec.json")
    return refused(extract_past_pf3(w), "PF7", "clustering")


@case("PF8", "VS25", BASE, "a bundle whose round trip failed")
def pf8(w):
    w.relayer(lambda m: m["roundtrip"].update(status="fail"))
    return refused(w.facts(), "PF8", "round trip")


@case("PF9", "VS27", BASE, "building into a bundle whose MANIFEST.files pins a showcase/ path")
def pf9(w):
    w.relayer(lambda m: m["files"].update({"showcase/story.json": "sha256:" + "0" * 64}))
    restage(w)
    return refused(w.build(), "PF9", "two owners")


# =============================================================================== build

@case("BD1", "PN4", BASE, "a template whose fmt.js differs by one byte from what TEMPLATE.json pins")
def bd1(w):
    def seed(page):
        fmt = page / "js" / "core" / "fmt.js"
        fmt.write_bytes(fmt.read_bytes().replace(b"f", b"g", 1))
    page = template_copy(w.root / "tpl", seed, pin=False)
    return refused(w.build("--template", page), "BD1", "fmt.js")


@case("BD2", "PN7", BASE, "page.json's motif 'packets'")
def bd2(w):
    edit_json(w.showcase / "page.json", lambda d: d["motif"].update(id="packets"))
    return refused(show("page_check.py", w.showcase / "page.json"), "BD2", "packets")


@case("BD3", "D5-25", BASE, "a .py file in the staged showcase")
def bd3(w):
    work = w.bundle / "showcase.building" / ".work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "hook.py").write_text("print('no')\n", encoding="utf-8")
    return refused(w.build(), "BD3", "hook.py")


@case("BD4", "", BASE, "a symlink in the staged showcase")
def bd4(w):
    work = w.bundle / "showcase.building" / ".work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "link.json").symlink_to(w.manifest)
    return refused(w.build(), "BD4", "symlink")


@case("BD5", "D5-26", BASE, "a build whose SHOWCASE.json serves a path it does not pin")
def bd5(w):
    return refused(seeded_build(w, (
        "real = m.ShowcaseDocument.assemble\n"
        "def assemble(self, *a, **k):\n"
        "    doc = real(self, *a, **k)\n"
        "    doc['served'].append('absent.json')\n"
        "    return doc\n"
        "m.ShowcaseDocument.assemble = assemble")), "BD5", "absent.json")


@case("BD6", "", BASE, "a build whose SHOWCASE.json does not validate vega/showcase@1")
def bd6(w):
    return refused(seeded_build(w, (
        "real = m.ShowcaseDocument._bundle\n"
        "def bundle(self):\n"
        "    doc = real(self)\n"
        "    doc.pop('manifest_digest')\n"
        "    return doc\n"
        "m.ShowcaseDocument._bundle = bundle")), "BD6", "manifest_digest")


@case("BD7", "", BASE, "a build locale that is neither en nor qps-ploc")
def bd7(w):
    return refused(w.build("--locale", "xx-yy"), "BD7", "xx-yy")


@case("BD8", "", BASE, "a build that writes into the bundle outside showcase*/")
def bd8(w):
    return refused(seeded_build(w, (
        "real = b.bundle_run\n"
        "def bundle_run(bundle, scratch):\n"
        "    out = real(bundle, scratch)\n"
        "    (Path(bundle) / 'stray.txt').write_text('x', encoding='utf-8')\n"
        "    return out\n"
        "b.bundle_run = bundle_run")), "BD8", "stray.txt")


# =============================================================================== check

def checked(w, code: int, gate: str, *words, extra=()) -> Expect:
    rc, out = w.check(*extra)
    return Expect(rc, out, code, (f"{gate} ", *words))


@case("CK1", "", BASE, "a SHOWCASE.json that does not read")
def ck1(w):
    path = w.showcase / "SHOWCASE.json"
    path.write_bytes(path.read_bytes()[:200])
    return checked(w, 2, "CK1", "broken")


@case("CK2", "D5-32", BASE, "story.json edited after the build (a value removed)")
def ck2_story(w):
    def drop(story):
        story["values"].pop(next(k for k in story["values"] if k != "primary"))
    edit_json(w.showcase / "story.json", drop)
    return checked(w, 2, "CK2", "story.json", "broken")


@case("CK2", "PN4", BASE, "a byte changed in the showcase's page/js/core/fmt.js")
def ck2_fmt(w):
    fmt = w.showcase / "page" / "js" / "core" / "fmt.js"
    fmt.write_bytes(fmt.read_bytes().replace(b"f", b"g", 1))
    return checked(w, 2, "CK2", "fmt.js")


@case("CK3", "D5-25", BASE, "a .py file dropped into the showcase")
def ck3(w):
    (w.showcase / "hook.py").write_text("print('no')\n", encoding="utf-8")
    return checked(w, 2, "CK3", "hook.py")


@case("CK4", "D5-26", BASE, "a served path that files does not pin")
def ck4(w):
    edit_json(w.showcase / "SHOWCASE.json", lambda d: d["served"].append("extra.json"))
    return checked(w, 2, "CK4", "extra.json")


@case("CK5", "D5-27", BASE, "SHOWCASE.json's bundle.manifest_digest off by one hex")
def ck5(w):
    edit_json(w.showcase / "SHOWCASE.json",
              lambda d: d["bundle"].update(manifest_digest=off_by_one(d["bundle"]["manifest_digest"])))
    return checked(w, 2, "CK5", "manifest_digest")


def reemit(w) -> None:
    """MANIFEST.json as `pack.py --re-emit-serving` leaves it: a reemits entry naming the
    digest the showcase was built for."""
    previous = sha(w.manifest.read_bytes())
    edit_json(w.manifest, lambda m: m.setdefault("reemits", []).append(
        {"at": "2026-10-09T00:00:00Z", "reason": "templates", "previous_manifest_digest": previous,
         "changed": [], "added": [], "removed": []}))


@case("CK6", "", BASE, "the bundle re-emitted after the showcase was built")
def ck6(w):
    reemit(w)
    return checked(w, 3, "CK6", "stale")


@case("CK7", "D5-28", BASE, "the report's PDF changed")
def ck7(w):
    pdf = w.docs / "main.pdf"
    pdf.write_bytes(pdf.read_bytes() + b" ")
    return checked(w, 3, "CK7", "stale")


@case("CK8", "", "L03", "a stage brief edited after a brief-link build")
def ck8(w):
    briefs = read_json(w.showcase / "SHOWCASE.json")["report"]["briefs"]
    sources = [w.work / e["source"] if not Path(e["source"]).is_absolute() else Path(e["source"])
               for e in briefs.values()]
    mine = next(s for s in sources if w.root in s.resolve().parents)
    mine.write_text(mine.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8")
    return checked(w, 3, "CK8", "stale")


@case("CK9", "", "L03", "a compiled report that appears after a brief-link build")
def ck9(w):
    shutil.copytree(Path(__file__).resolve().parent / "fixtures" / "report", w.docs)
    return checked(w, 3, "CK9", "stale")


@case("CK10", "", BASE, "a template updated since the build: WARN, not stale")
def ck10(w):
    page = template_copy(w.root / "tpl", lambda p: edit_text(p / "page.css", "--accent:", "--accent: "))
    return checked(w, 0, "CK10", "template updated since build", extra=("--template", page))


CASES = [v for v in list(globals().values()) if type(v).__name__ == "Case"]
