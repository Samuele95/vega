"""
check_kit.py — what the live lane checks share (D6 §7, §8; BUILD-PLAN T08): the records of the
bundle under test, a small HTTP client for its running service, the tensor decoder, and the
report every check id writes into.

    bundle = BundleRecords.load(bundle_dir)        MANIFEST.json, schema.json, netspec.json,
                                                   records/architecture.json, the examples
    client = Client(base_url)                      get/post JSON, post a CSV upload
    report = Report(schema, lane, base_url, bundle)
    report.record(check_id, ok, detail, case=None) one observation of one invariant
    report.not_applicable(check_id, why)           an invariant the bundle's task does not have
    decode(tensor_payload) -> numpy array          json | base64-le (float32 or int64)

Not a test module and not a check of its own: `serving_check.py` (SA1–SA14) and
`inspect_check.py` (NI1–NI8, EI1–EI12) import it. Every name read from another subsystem comes
from `lanekit.contracts`, the one place such names live (D6 §17.2). The helpers raise
`CheckError` for a precondition the check cannot run without (no service, no examples); an
invariant that fails is a `record(..., ok=False)`, never an exception.
"""
from __future__ import annotations

import base64
import json
import math
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lanekit import contracts as C  # noqa: E402
from lanekit.service import request  # noqa: E402

__all__ = ["Answer", "BundleRecords", "CheckError", "Client", "Report", "close", "decode", "dig",
           "finite", "read_parquet", "write_result"]

TIMEOUT_S = 120.0


class CheckError(RuntimeError):
    """A precondition of the check is missing: the service does not answer, a record is
    absent. The check exits 2 naming it; no invariant is reported as passed."""


def dig(doc: Any, path: Iterable[str]) -> Any:
    """doc[path[0]][path[1]]…, None where a key is missing."""
    for key in path:
        if not isinstance(doc, Mapping):
            return None
        doc = doc.get(key)
    return doc


def finite(value: Any) -> bool:
    """A JSON number that is finite (a bool is not a number)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(float(value))


def close(a: Any, b: Any, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    """Two numbers (or two nulls) agree within `atol + rtol·|b|`."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


def decode(payload: Mapping | None) -> np.ndarray | None:
    """A tensor payload (network-view@2 §5.2, or an exact tensor of explanation@2) as an
    array: `json` lists, `base64-le` float32 or int64, row-major, in its declared shape."""
    if payload is None:
        return None
    shape = tuple(int(s) for s in payload.get("shape") or ())
    encoding, data = payload.get("encoding"), payload.get("data")
    if encoding == "json":
        return np.asarray(data, dtype=np.float64).reshape(shape)
    if encoding == "base64-le":
        dtype = "<i8" if payload.get("dtype") == "int64" else "<f4"
        raw = np.frombuffer(base64.b64decode(data), dtype=dtype)
        return raw.reshape(shape).astype(np.float64 if dtype == "<f4" else np.int64)
    raise ValueError(f"tensor {payload.get('id')!r}: encoding {encoding!r} carries no data")


# --------------------------------------------------------------------------- the bundle
@dataclass(frozen=True)
class BundleRecords:
    """The records of the bundle the live service serves, read once."""
    root: Path
    manifest: Mapping
    schema: Mapping
    netspec: Mapping
    architecture: Mapping

    @staticmethod
    def load(root: Path) -> "BundleRecords":
        root = Path(root)
        manifest = _read(root / C.BUNDLE_MANIFEST)
        schema = _read(root / (manifest.get("input_schema") or C.BUNDLE_SCHEMA))
        arch = root / C.ARCHITECTURE_RECORD
        return BundleRecords(root, manifest, schema, _read(root / C.BUNDLE_NETSPEC),
                             _read(arch) if arch.is_file() else {})

    @property
    def output(self) -> Mapping:
        return self.manifest.get("output") or {}

    @property
    def sample(self) -> Mapping:
        return self.manifest.get("sample") or {"shape": "row"}

    @property
    def shape(self) -> str:
        return str(self.sample.get("shape") or "row")

    @property
    def level(self) -> str | None:
        return self.sample.get("level") if self.shape == "graph" else None

    @property
    def reading(self) -> str:
        return str(self.output.get("reading"))

    @property
    def tolerance(self) -> float:
        value = dig(self.manifest, C.ROUNDTRIP_TOLERANCE)
        if value is None:
            raise CheckError("MANIFEST.json records no roundtrip.tolerance.value")
        return float(value)

    def field_kind(self, name: str) -> tuple[str | None, Any]:
        entry = (self.output.get("field_kinds") or {}).get(name) or {}
        return entry.get("kind"), entry.get("of")

    def fields_of(self, *kinds: str) -> list[str]:
        return [f for f in self.output.get("fields") or () if self.field_kind(f)[0] in kinds]

    def expected(self):
        """examples/expected.parquet (pandas), the frame the round trip wrote."""
        return read_parquet(self.root / C.EXAMPLE_EXPECTED)

    def packed_smoke(self) -> tuple[Mapping | None, str]:
        """(smoke, where it was read) of the network this bundle packs: the architecture
        record's when the packed netspec is the forged one, or when pack put the selected
        run's own smoke there (`smoke_of` naming the manifest's run, A44(b)); else that run's
        smoke.json in the campaign the lane trained (a ladder rung or a trial re-emits its
        package and smokes it), found beside the bundle in the lane's work dir."""
        arch_spec = self.architecture.get("netspec") or {}
        smoke = dig(self.architecture, C.ARCHITECTURE_SMOKE)
        shipped = (self.architecture.get("smoke_of") or {}).get("run")
        if smoke and (arch_spec.get("blocks") == self.netspec.get("blocks")
                      or (shipped and shipped == self.manifest.get("run"))):
            return smoke, C.ARCHITECTURE_RECORD
        campaign, run = self.manifest.get("campaign"), self.manifest.get("run")
        if campaign and run:
            path = self.root.parents[1] / C.LANE_RUN_SMOKE.format(campaign=campaign, run=run)
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8")), str(path)
        return None, (f"the packed network (run {run}) is not the forged one, and no smoke of "
                      "it is found")

    def inspection_enabled(self) -> bool:
        off = C.INSPECTION_OFF
        return dig(self.manifest, off["manifest"]) is not off["manifest_off"]


def read_parquet(path: Path, columns=None):
    """A parquet file as a pandas frame, read by Arrow's own filesystem. pandas hands Arrow a
    Python file object for a local path, and Arrow's worker threads then call back into Python:
    one still doing so when the interpreter finalizes is ended mid-unwind, and the process
    aborts at exit ('terminate called without an active exception', A40) — after its work is
    done. A native filesystem keeps Python out of Arrow's threads."""
    import pandas as pd
    from pyarrow import fs
    return pd.read_parquet(path, columns=columns, filesystem=fs.LocalFileSystem())


def _read(path: Path) -> dict:
    if not path.is_file():
        raise CheckError(f"the bundle has no {path.name} ({path})")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- HTTP
@dataclass(frozen=True)
class Answer:
    """One HTTP answer: status, lower-cased headers, raw body, and the JSON (None if not)."""
    status: int
    headers: Mapping[str, str]
    body: bytes

    @property
    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None


@dataclass
class Client:
    """The live service at `base_url`."""
    base_url: str
    seen: list = field(default_factory=list)      # every JSON answer body, for SA13

    def get(self, path: str) -> Answer:
        return self._send("GET", path)

    def post(self, path: str, doc: Any, headers: Mapping[str, str] | None = None) -> Answer:
        body = json.dumps(doc, allow_nan=False).encode("utf-8")
        return self._send("POST", path, body, {"content-type": "application/json",
                                               **dict(headers or {})})

    def upload(self, path: str, csv_bytes: bytes, accept: str | None = None) -> Answer:
        """POST a CSV as the multipart `file` field, as a browser form sends it."""
        boundary = "----vega-check-" + uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"rows.csv\"\r\nContent-Type: text/csv\r\n\r\n").encode("utf-8")
        body += csv_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
        headers = {"content-type": f"multipart/form-data; boundary={boundary}"}
        if accept:
            headers["accept"] = accept
        return self._send("POST", path, body, headers)

    def json_of(self, answer: Answer, what: str) -> Any:
        """The JSON of a 200 answer, or CheckError naming what was asked."""
        if answer.status != 200 or answer.json is None:
            raise CheckError(f"{what} answered {answer.status}: {answer.body[:300]!r}")
        return answer.json

    def _send(self, method: str, path: str, body: bytes | None = None,
              headers: Mapping[str, str] | None = None) -> Answer:
        try:
            status, got, raw = request(self.base_url + path, method=method, body=body,
                                       headers=headers, timeout=TIMEOUT_S)
        except OSError as exc:
            raise CheckError(f"{method} {path}: the service at {self.base_url} does not "
                             f"answer ({exc})") from None
        answer = Answer(status, got, raw)
        self.seen.append((method, path, answer))
        return answer


# --------------------------------------------------------------------------- the report
@dataclass
class Report:
    """What one check run observed: per invariant id, every case with its verdict."""
    schema: str
    lane: str
    base_url: str
    bundle: str
    ids: tuple
    cases: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)

    def record(self, check_id: str, ok: bool, detail: str, case: str | None = None) -> bool:
        self._known(check_id)
        self.cases.setdefault(check_id, []).append(
            {"case": case, "ok": bool(ok), "detail": str(detail)[:600]})
        return bool(ok)

    def not_applicable(self, check_id: str, why: str) -> None:
        self._known(check_id)
        self.skipped.setdefault(check_id, why)

    def _known(self, check_id: str) -> None:
        if check_id not in self.ids:
            raise KeyError(f"{check_id} is not one of this check's ids {self.ids}")

    def status(self, check_id: str) -> str:
        cases = self.cases.get(check_id) or []
        if any(not c["ok"] for c in cases):
            return "fail"
        if cases:
            return "pass"
        return "not-applicable" if check_id in self.skipped else "not-run"

    @property
    def verdict(self) -> str:
        return "pass" if all(self.status(i) in ("pass", "not-applicable") for i in self.ids) \
            else "fail"

    def as_dict(self, **extra: Any) -> dict:
        checks = []
        for check_id in self.ids:
            cases = self.cases.get(check_id) or []
            checks.append({"id": check_id, "status": self.status(check_id),
                           "cases": len(cases), "failed": [c for c in cases if not c["ok"]][:12],
                           "why": self.skipped.get(check_id)})
        return {"schema": self.schema, "lane": self.lane, "base_url": self.base_url,
                "bundle": self.bundle, **extra, "checks": checks, "verdict": self.verdict}

    def print_lines(self) -> None:
        for check in self.as_dict()["checks"]:
            status = check["status"]
            tag = {"pass": "OK:", "not-applicable": "OK:", "fail": "ERROR:"}.get(status, "ERROR:")
            what = f"{check['cases']} case(s)" if status != "not-applicable" else check["why"]
            print(f"{tag} {check['id']} {status} — {what}")
            for case in check["failed"][:3]:
                print(f"  {check['id']} [{case['case']}] {case['detail']}")


def write_result(path: str | None, doc: Mapping) -> None:
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")
