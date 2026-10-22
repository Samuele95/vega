"""
Start and stop a bundle's service on a free port (D6 §6.9).

    svc = start(bundle, threads=2)          # polls /health until it answers 200, 30 s at most
    svc = start(bundle, launch=Launch(python, log, env))
    status, headers, body = svc.get("/health")
    stop(svc)

The service is the bundle's own `python -m serve.app --port <p> --torch-threads <n>`, run in the
bundle directory, in its own session (stopping it kills the whole process group), with its
output in a log beside the lane's other logs. `/health` answers 503 until the service's
self-test passed and 200 after (the serving templates' contract): `start` waits for the 200,
and on a timeout stops the process and raises `ServiceError` with the last status and the tail
of the log, so a service that never became ready is a stage failure that says why.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from . import contracts
from .runner import kill_group

READY_TIMEOUT_S = 30.0
POLL_S = 0.25


class ServiceError(RuntimeError):
    """The service did not become ready, or died."""


@dataclass
class Service:
    bundle: Path
    port: int
    process: subprocess.Popen
    log: Path

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def get(self, path: str, headers: Mapping[str, str] | None = None,
            timeout: float = 10.0) -> tuple[int, dict, bytes]:
        return request(self.base_url + path, headers=headers, timeout=timeout)

    def get_json(self, path: str) -> tuple[int, dict]:
        status, _, body = self.get(path)
        return status, json.loads(body.decode("utf-8") or "null")

    def alive(self) -> bool:
        return self.process.poll() is None


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(url: str, method: str = "GET", body: bytes | None = None,
            headers: Mapping[str, str] | None = None, timeout: float = 10.0) -> tuple[int, dict, bytes]:
    """(status, lower-cased headers, body) — an HTTP error status is an answer, not an error."""
    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


@dataclass(frozen=True)
class Launch:
    """How the service process is launched: the interpreter, the log, the environment."""
    python: str = sys.executable
    log: Path | None = None
    env: Mapping[str, str] | None = None


def start(bundle: Path, port: int | None = None, threads: int = 2, launch: Launch = Launch()) -> Service:
    """Start the bundle's service and wait until /health answers 200."""
    port = port or free_port()
    log = launch.log or bundle.parent / f"serve-{port}.log"
    args = [a.format(port=port, threads=threads) for a in contracts.SERVE_ARGS]
    argv: Sequence[str] = [launch.python, "-m", contracts.SERVE_MODULE, *args]
    run_env = dict(os.environ if launch.env is None else launch.env, PYTHONDONTWRITEBYTECODE="1")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        out.write(f"\n$ {' '.join(argv)}   (cwd {bundle})\n".encode())
        out.flush()
        process = subprocess.Popen(argv, cwd=bundle, env=run_env, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    service = Service(Path(bundle), port, process, log)
    wait_ready(service)
    return service


def wait_ready(service: Service, timeout: float = READY_TIMEOUT_S) -> dict:
    """Poll /health until 200; the body is returned. Raises ServiceError on death or timeout."""
    deadline, last = time.monotonic() + timeout, "no answer"
    while time.monotonic() < deadline:
        if not service.alive():
            raise ServiceError(f"the service exited with {service.process.returncode} before it "
                               f"was ready:\n{_tail(service.log)}")
        try:
            status, _, body = service.get(contracts.HEALTH_ROUTE, timeout=2.0)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(POLL_S)
            continue
        if status == 200:
            return json.loads(body.decode("utf-8") or "{}")
        last = f"{status} {body[:300].decode('utf-8', errors='replace')}"
        time.sleep(POLL_S)
    stop(service)
    raise ServiceError(f"/health did not answer 200 within {timeout:.0f} s (last: {last})\n"
                       f"{_tail(service.log)}")


def stop(service: Service) -> int | None:
    """Stop the service and its process group; returns its exit status."""
    if service.alive():
        kill_group(service.process.pid)
    try:
        return service.process.wait(timeout=10)
    except subprocess.TimeoutExpired:                              # pragma: no cover
        return None


def _tail(log: Path, lines: int = 15) -> str:
    try:
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no service log)"
