"""
serve.routes_showcase — the showcase slot of {{PACKAGE}}'s service: the demonstration page
/vega-show built for this bundle, served on the service's own origin from a directory the
bundle does not pin and the service never writes.

    GET /                  the verified showcase's entry (`page/index.html`), through the page
                           override routes_meta answers `/` with; else the console page
    GET /console           always the console page (routes_meta registers it)
    GET /showcase/{path}   a file the verified showcase serves, else 404
    GET /v1/story          the showcase's story (`story.file`)
    GET /report.pdf        the report the showcase links (`report.pdf.file`), else 404
    GET /v1/showcase       the slot's status, the object /health and /v1/info carry as `showcase`

A route module (R-D1, R-D3), capability `showcase`: nothing here reaches the model, and the
page's files answer while the model is not ready. The directory is {{ENV}}_SHOWCASE (`--showcase
DIR` sets it), else `<bundle>/showcase`; its manifest is SHOWCASE.json (vega/showcase@1,
/vega-show's contract). The bundle pins no showcase file: a showcase changes nothing it verifies.

Opening (`ShowcaseSlot.open`, once, in `register`, before the self-test). No directory, or no
SHOWCASE.json in it, is an absent slot, the normal state. Otherwise the first failure refuses
the showcase (gates SL1-SL11, in `_Opening`'s order): another schema; a `showcase/` path the
bundle's MANIFEST.json pins; a showcase made for another bundle (`bundle.manifest_digest` is
not the sha256 of this MANIFEST.json's bytes, or `bundle.model_version` not this model's) or
stale (that digest is a `MANIFEST.reemits[].previous_manifest_digest`); a pinned path that is
not relative and normalised or has no servable extension; `served` not a subset of `files`,
or the entry (an .html page), the story or the linked report not served; a symlink anywhere,
or a servable file `files` does not pin outside `.work/`, `.smoke/` and `docs/`; a pinned file
outside the directory or not a regular file, over the byte cap (64 MiB in all), or not its
digest. A refused showcase is reported (`verified: false`, the reason) and logged; it never
makes the model unready, and nothing of it is served.

Serving. The served files are read into memory at opening and served from there alone (a file
altered afterwards is never served, no traversal reaches a file), under `/showcase/` only (no
service route is shadowed), with `ETag` (the digest; `If-None-Match` answers 304), `no-cache`,
`nosniff`, and for HTML the page's content security policy.
"""
# No postponed annotations here: FastAPI must read the route signatures as objects.
import hashlib
import json
import logging
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .routes_meta import CSP

__all__ = ["CAPABILITY", "ENDPOINTS", "MODEL_ROUTES", "OPEN_WHILE_UNREADY", "ShowcaseSlot",
           "register"]

# the file route as FastAPI registers it (`{path:path}` spans segments): MANIFEST.serve.endpoints
# equals the app's routes (MANIFEST.serve.showcase_slot writes `GET /showcase/{path}`)
ENDPOINTS = (("GET", "/"), ("GET", "/console"), ("GET", "/showcase/{path:path}"),
             ("GET", "/v1/story"), ("GET", "/report.pdf"), ("GET", "/v1/showcase"))
MODEL_ROUTES = ()
OPEN_WHILE_UNREADY = ("/", "/console", "/showcase/", "/v1/showcase")
CAPABILITY = "showcase"

SCHEMA = "vega/showcase@1"
MANIFEST = "SHOWCASE.json"
MEDIA = MappingProxyType({
    ".html": "text/html; charset=utf-8", ".json": "application/json",
    ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp",
    ".woff2": "font/woff2", ".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8"})
SLOT_MAX_BYTES = 67_108_864
UNPINNED = (".work/", ".smoke/", "docs/")        # scratch: neither pinned nor served
UNLISTED_SHOWN = 20                               # the unpinned files the status names, at most
RELPATH = re.compile(r"^(?!/)(?!.*(^|/)\.\.(/|$))[A-Za-z0-9._/-]{1,200}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
STALE = "the bundle was re-emitted after this showcase; re-run /vega-show"
OTHER_BUNDLE = "the showcase was made for another bundle"
NO_SHOWCASE = "this service has no showcase"
REFUSED = "this service's showcase was refused; GET /v1/showcase says why"
NOT_SERVED = "no such file in this service's showcase"
NO_REPORT = "this showcase links no report (its notes link the stage briefs)"
log = logging.getLogger("{{CMD}}.serve")


def register(app: FastAPI, state) -> None:
    slot = _opened(state)
    state.report("showcase", slot.status())
    if slot.verified:
        state.page_override = slot.page

    @app.get("/showcase/{path:path}", include_in_schema=False)
    def showcase_file(path: str, request: Request):
        return slot.answer(path, request.headers.get("if-none-match"))

    @app.get("/v1/story")
    def story(request: Request):
        return slot.answer(slot.story, request.headers.get("if-none-match"))

    @app.get("/report.pdf", include_in_schema=False)
    def report(request: Request):
        return slot.answer(slot.report, request.headers.get("if-none-match"), NO_REPORT)

    @app.get("/v1/showcase")
    def showcase_status():
        return JSONResponse(slot.status())


def _opened(state) -> "ShowcaseSlot":
    """The service's slot; failing to open it refuses the showcase, never the model."""
    try:
        manifest = (Path(state.root) / "MANIFEST.json").read_bytes()
        return ShowcaseSlot.open(state.showcase_dir(), manifest,
                                 state.manifest.get("model_version"))
    except Exception as exc:
        log.exception("the showcase could not be opened")
        return ShowcaseSlot(True, refusal=SlotRefused(
            f"the showcase could not be opened ({type(exc).__name__}); the server log records "
            "the cause"))


class SlotRefused(Exception):
    """Why the showcase is not served, and the unlisted files that made it so (if any)."""

    def __init__(self, reason: str, unlisted=()):
        super().__init__(reason)
        self.reason, self.unlisted = reason, tuple(unlisted)


@dataclass(frozen=True)
class Served:
    """One served file: its bytes (read once), its media type, its pinned digest."""
    body: bytes
    media: str
    digest: str

    def answer(self, if_none_match: Optional[str]) -> Response:
        headers = {"ETag": f'"{self.digest}"', "Cache-Control": "no-cache",
                   "X-Content-Type-Options": "nosniff"}
        if self.media.startswith("text/html"):
            headers["Content-Security-Policy"] = CSP
        if _matches(if_none_match, self.digest):
            return Response(status_code=304, headers=headers)
        return Response(self.body, media_type=self.media, headers=headers)


class ShowcaseSlot:
    """The showcase directory, opened once: verified or why not, its served files, its status."""

    def __init__(self, present: bool, digest: Optional[str] = None,
                 doc: Optional[Mapping] = None, served: Optional[Mapping] = None,
                 refusal: Optional[SlotRefused] = None):
        self.present, self.digest = present, digest
        self.reason = refusal.reason if refusal is not None else None
        self.unlisted = refusal.unlisted if refusal is not None else ()
        self.verified = bool(present and refusal is None and served is not None)
        self.doc = doc if isinstance(doc, Mapping) else {}
        self.served = MappingProxyType(dict(served or {}) if self.verified else {})
        self.entry = _text(self.doc.get("entry"))
        self.story = _text(_at(self.doc, "story", "file"))
        self.report = _text(_at(self.doc, "report", "pdf", "file"))

    @staticmethod
    def open(directory, bundle_manifest: bytes, model_version) -> "ShowcaseSlot":
        """`directory`'s slot for this bundle: absent, refused (and why), or verified."""
        if directory is None or not Path(directory).is_dir():
            return ShowcaseSlot(False)
        root = Path(directory).resolve()
        path = root / MANIFEST
        if not (path.is_file() or path.is_symlink()):
            return ShowcaseSlot(False)
        digest, doc = None, None
        try:
            raw = _read(path, MANIFEST)
            digest, doc = _sha(raw), _document(raw)
            served = _Opening(root, doc, bundle_manifest, model_version).verify()
        except (SlotRefused, OSError) as failure:
            refusal = failure if isinstance(failure, SlotRefused) else SlotRefused(
                f"the showcase could not be read ({type(failure).__name__}: {failure.strerror})")
            log.warning("the showcase at %s is refused: %s", root, refusal.reason)
            return ShowcaseSlot(True, digest, doc, refusal=refusal)
        log.info("the showcase at %s is verified: %d file(s) served", root, len(served))
        return ShowcaseSlot(True, digest, doc, served)

    def page(self) -> Optional[Response]:
        """`GET /`: the verified entry (the page override), or None (the console page)."""
        served = self.served.get(self.entry)
        return served.answer(None) if served is not None else None

    def answer(self, rel: Optional[str], if_none_match: Optional[str],
               missing: str = NOT_SERVED) -> Response:
        """The served file `rel` (304 when the caller holds its ETag), else a 404 problem."""
        if not self.verified:
            raise StarletteHTTPException(404, REFUSED if self.present else NO_SHOWCASE)
        served = self.served.get(rel) if rel else None
        if served is None:
            raise StarletteHTTPException(404, missing)
        return served.answer(if_none_match)

    def status(self) -> dict:
        """The slot's status (R-D4): /v1/showcase answers it, /health carries it."""
        files, template = self.doc.get("files"), self.doc.get("template")
        return {"present": self.present, "verified": self.verified, "reason": self.reason,
                "digest": self.digest, "entry": self.entry,
                "template": {"version": _text(template.get("version")),
                             "manifest_digest": _text(template.get("manifest_digest"))}
                if isinstance(template, Mapping) else None,
                "report_status": _text(_at(self.doc, "report", "status")),
                "created_at": _text(self.doc.get("created_at")),
                "files": len(files) if isinstance(files, Mapping) else 0,
                "unlisted": list(self.unlisted)}


class _Opening:
    """One opening's checks, in order: `verify()` -> {path: Served}, or SlotRefused."""

    def __init__(self, root: Path, doc: Mapping, bundle_manifest: bytes, model_version):
        self.root, self.doc, self.model_version = root, doc, model_version
        self.bundle_digest = _sha(bundle_manifest)
        try:
            self.bundle = json.loads(bundle_manifest)
        except ValueError:
            self.bundle = {}

    def verify(self) -> dict:
        self._made_for_this_bundle()
        files = self._files()
        served = self._served(files)
        self._closed_world(files)
        return self._read_all(files, served)

    def _made_for_this_bundle(self) -> None:
        if self.doc.get("schema") != SCHEMA:
            raise SlotRefused(f"{MANIFEST} is not {SCHEMA} (its schema is "
                              f"{self.doc.get('schema')!r})")
        pinned = _at(self.bundle, "files")
        owned = sorted(r for r in (pinned if isinstance(pinned, Mapping) else ())
                       if str(r).startswith("showcase/"))
        if owned:
            raise SlotRefused(f"the bundle's MANIFEST.json pins {owned[0]}: a showcase file "
                              f"would have two owners (MANIFEST.json and {MANIFEST})")
        digest, reemits = _text(_at(self.doc, "bundle", "manifest_digest")), \
            _at(self.bundle, "reemits")
        if digest != self.bundle_digest:
            if digest is not None and any(_at(r, "previous_manifest_digest") == digest for r in (
                    reemits if isinstance(reemits, list) else ())):
                raise SlotRefused(STALE)
            raise SlotRefused(f"{OTHER_BUNDLE}: its bundle.manifest_digest is not the sha256 of "
                              "this bundle's MANIFEST.json")
        if _at(self.doc, "bundle", "model_version") != self.model_version:
            raise SlotRefused(f"{OTHER_BUNDLE}: its bundle.model_version is not this bundle's "
                              "model_version")

    def _files(self) -> dict:
        files = self.doc.get("files")
        if not isinstance(files, Mapping) or not files:
            raise SlotRefused(f"{MANIFEST} pins no file (`files` is empty or not an object)")
        for rel, digest in files.items():
            if not _relpath(rel):
                raise SlotRefused(f"{MANIFEST} lists {rel!r}, a path outside the showcase "
                                  "directory (relative, normalised, no '..' segment)")
            if rel == MANIFEST:
                raise SlotRefused(f"{MANIFEST} lists itself among its files")
            if _media(rel) is None:
                raise SlotRefused(f"{rel}: not a file the slot can serve (the servable "
                                  f"extensions are {' '.join(MEDIA)})")
            if not isinstance(digest, str) or not DIGEST.match(digest):
                raise SlotRefused(f"{rel}: its digest in {MANIFEST} is not sha256:<64 hex>")
        return dict(files)

    def _served(self, files: Mapping) -> frozenset:
        served = self.doc.get("served")
        if not isinstance(served, list) or not all(isinstance(s, str) for s in served):
            raise SlotRefused(f"{MANIFEST}: `served` is not a list of paths")
        extra = sorted(set(served) - set(files))
        if extra:
            raise SlotRefused(f"`served` lists {extra[0]}, which `files` does not pin")
        pdf = _at(self.doc, "report", "pdf")
        wanted = {"entry": self.doc.get("entry"), "story.file": _at(self.doc, "story", "file")}
        if pdf is not None:
            wanted["report.pdf.file"] = _at(pdf, "file")
        for name, rel in wanted.items():
            if not isinstance(rel, str) or rel not in served:
                raise SlotRefused(f"{name} ({rel!r}) is not a file the showcase serves")
        if not wanted["entry"].endswith(".html"):
            raise SlotRefused(f"the entry {wanted['entry']!r} is not an .html page")
        if pdf is not None and _at(pdf, "digest") != files[wanted["report.pdf.file"]]:
            raise SlotRefused("report.pdf.digest is not the digest `files` pins for "
                              f"{wanted['report.pdf.file']}")
        return frozenset(served)

    def _closed_world(self, files: Mapping) -> None:
        """No symlink anywhere; no unpinned servable file outside the scratch directories."""
        unlisted = []
        for rel, link in _walk(self.root):
            if link:
                raise SlotRefused(f"{rel} is a symlink: a showcase holds none")
            if rel != MANIFEST and rel not in files and not rel.startswith(UNPINNED) \
                    and _media(rel) is not None:
                unlisted.append(rel)
        if unlisted:
            raise SlotRefused(f"{unlisted[0]} is not pinned in {MANIFEST} ({len(unlisted)} "
                              "unpinned servable file(s) outside .work/, .smoke/ and docs/)",
                              unlisted[:UNLISTED_SHOWN])

    def _read_all(self, files: Mapping, served: frozenset) -> dict:
        """Every pinned file held to its digest within the byte cap; the served ones kept."""
        total, kept = 0, {}
        for rel in sorted(files):
            path = self._inside(rel)
            total += path.lstat().st_size
            if total > SLOT_MAX_BYTES:
                raise SlotRefused(f"the showcase's files exceed the slot's cap of "
                                  f"{SLOT_MAX_BYTES} bytes (at {rel})")
            body = _read(path, rel)
            if _sha(body) != files[rel]:
                raise SlotRefused(f"{rel} does not match its digest in {MANIFEST}: it changed "
                                  "after the showcase was built")
            if rel in served:
                kept[rel] = Served(body, _media(rel), files[rel])
        return kept

    def _inside(self, rel: str) -> Path:
        """The pinned file's path, inside the directory once resolved and a regular file."""
        path = self.root / rel
        if self.root not in path.resolve().parents:
            raise SlotRefused(f"{rel} lies outside the showcase directory")
        try:
            mode = path.lstat().st_mode
        except OSError:
            raise SlotRefused(f"{rel} is pinned in {MANIFEST} but missing or unreadable") from None
        if not stat.S_ISREG(mode):
            raise SlotRefused(f"{rel} is not a regular file")
        return path


def _walk(root: Path):
    """(relative path, is a symlink) of every file, and every symlinked directory, below root."""
    for top, dirs, names in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(dirs + names):
            path = Path(top) / name
            if path.is_symlink() or name in names:
                yield path.relative_to(root).as_posix(), path.is_symlink()


def _read(path: Path, rel: str) -> bytes:
    """The file's bytes, never through a symlink (the last component is opened O_NOFOLLOW)."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SlotRefused(f"{rel} could not be read ({exc.strerror or type(exc).__name__}; "
                          "a symlink is refused)") from None
    with os.fdopen(fd, "rb") as fh:
        return fh.read()


def _document(raw: bytes) -> dict:
    try:
        doc = json.loads(raw)
    except ValueError:
        raise SlotRefused(f"{MANIFEST} does not parse as JSON") from None
    if not isinstance(doc, dict):
        raise SlotRefused(f"{MANIFEST} is not a JSON object")
    return doc


def _relpath(rel) -> bool:
    return isinstance(rel, str) and RELPATH.match(rel) is not None and \
        all(part not in ("", ".", "..") for part in rel.split("/"))


def _media(rel: str) -> Optional[str]:
    return MEDIA.get(PurePosixPath(rel).suffix.lower())


def _matches(if_none_match: Optional[str], digest: str) -> bool:
    """RFC 9110 If-None-Match, weak comparison: `*`, or any listed tag is this digest's."""
    tags = [t.strip() for t in (if_none_match or "").split(",") if t.strip()]
    return any(t == "*" or t.removeprefix("W/") == f'"{digest}"' for t in tags)


def _sha(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _at(doc, *keys):
    """doc[k1][k2]…, or None where a step is not a mapping."""
    for key in keys:
        if not isinstance(doc, Mapping):
            return None
        doc = doc.get(key)
    return doc


def _text(value) -> Optional[str]:
    return value if isinstance(value, str) else None
