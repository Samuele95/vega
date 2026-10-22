"""
{{PACKAGE}}.{{MODULE_STORE}}.store — the {{PORT_MODELSTORE}} adapter (role: ModelStore).

{{RESPONSIBILITY_MODELSTORE}}
Source: {{SOURCE_MODELSTORE}}

Emitted by /vega-forge from `backends/pytorch/templates/store.py`. A defect is fixed in the
template and re-emitted, never in this copy (code-contract §4).

`ModelRegistry` is the one class here (C14). A stored model is the domain's `Checkpoint`, and
its identity is the digest of its weights, written inside the file by `save`
and recomputed by `load` on every load — not an option a caller may skip. A caller may also pin
the file's own digest (`expected_digest`, or a `Checkpoint` passed as `ref`). A metric computed
against the wrong weights is worse than no metric (EJ-49 — check parameters, name the offender).

What is stored is the network's whole `state_dict`: the parameters **and the registered
buffers** — the batch-norm running statistics, the one-class objective's centre
(`svdd_center`, `svdd_center_set`, R-B6). The weights digest covers the buffers too, so a centre
changed after training is a different model; `load` restores strictly, and a buffer the built
network declares but the file lacks (or the reverse) is refused by name, never left at its
initial value.

`load` reads with `weights_only=True`: the file holds a state dict, two strings and an int, so
nothing in it needs pickle's arbitrary-object path — and the serving path loads it. It takes
the network's builder as a parameter rather than importing the Model adapter: this file depends
on the ports and on `domain`, never on a sibling adapter (EJ-20, EJ-64).

Exporting to another runtime is not the store's job: `export.py`'s `TorchExporter` (the
`ArtifactExporter`, beside the network) does it, so a bundle can carry the store without the
exporter (EJ-21, SOLID-I).

    ModelRegistry().save(model, path, spec, *, epoch=None) -> Checkpoint
    ModelRegistry().load(ref, spec, build, *, expected_digest=None, run_dir=None) -> network
    ModelRegistry().version(path) -> Checkpoint     (read and verify, build nothing)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ..domain import Checkpoint, NetSpec, stream_digest
from ..ports import ModelBuilder as _ModelBuilder, TrainedNetwork as _Trained

__all__ = ["ModelRegistry"]


def _file_digest(path) -> str:
    """The checkpoint file's digest — the Registry adapter's IO (opening the file), the
    hashing the domain's one (`stream_digest`)."""
    with Path(path).open("rb") as fh:
        return stream_digest(fh)


def _weights_digest(state_dict) -> str:
    """sha256 over every tensor's name, dtype, shape and bytes, in key order — the weights'
    identity (buffers included), independent of the file's name, container format or pickle
    protocol."""
    h = hashlib.sha256()
    for name in sorted(state_dict):
        t = state_dict[name]
        if not torch.is_tensor(t):
            continue
        t = t.detach().to("cpu").contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return "sha256:" + h.hexdigest()


def _target(ref, expected_digest):
    """(path, the file digest to pin, the weights digest to pin) of a path or a Checkpoint."""
    if isinstance(ref, Checkpoint):
        return ref.path, expected_digest or ref.digest, ref.weights_digest
    return ref, expected_digest, None


def _read(path, expected_digest, want_weights) -> tuple:
    """(the file's blob, its file digest), after the file digest (when pinned) and the weights
    digest (always) were checked: the digest is read once, and returned for the caller."""
    if not Path(path).is_file():
        raise ValueError(f"checkpoint not found: {path}")
    found = _file_digest(path)
    if expected_digest and found != expected_digest:
        raise ValueError(f"checkpoint digest mismatch for {path}: expected {expected_digest}, "
                         f"found {found} — refusing to load")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    carried = blob.get("weights_digest") if isinstance(blob, dict) else None
    if not carried:
        raise ValueError(f"{path} carries no weights digest, so its identity cannot be "
                         "verified; it was saved by an older store — re-save it through save()")
    got = _weights_digest(blob["state_dict"])
    if got != carried or (want_weights and got != want_weights):
        raise ValueError(f"the weights in {path} digest to {got}, but the file records "
                         f"{carried}" + (f" and the caller pinned {want_weights}"
                                         if want_weights else "") + " — refusing to load")
    return blob, found


def _restore(model, state_dict, path) -> None:
    """Load strictly, naming what does not match: a parameter or buffer the network declares
    and the file lacks would otherwise keep its initial value (a centre never set)."""
    declared, stored = set(model.state_dict()), set(state_dict)
    missing, unexpected = sorted(declared - stored), sorted(stored - declared)
    if missing or unexpected:
        raise ValueError(f"{path} does not hold this network's state: "
                         + "; ".join(p for p in (
                             f"missing {', '.join(missing)}" if missing else "",
                             f"unexpected {', '.join(unexpected)}" if unexpected else "") if p)
                         + " — the spec and the checkpoint describe different networks")
    model.load_state_dict(state_dict, strict=True)


class ModelRegistry:
    """The ModelStore port's adapter: a model put by the digest of its weights, got back
    verified."""

    def save(self, model: _Trained, path, spec: NetSpec, *,
             epoch: int | None = None) -> Checkpoint:
        """Write the whole state (parameters and buffers) with its digest and the digest of
        the spec it was trained against inside, and return the Checkpoint that addresses it.
        The spec is required — a checkpoint without its spec could later be loaded under any
        spec that builds the same shapes (a permuted class order, another manifest) — and the
        epoch is the one option."""
        spec_digest = NetSpec.require(spec).digest        # refused before anything is written
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = model.state_dict()
        wd = _weights_digest(state)
        torch.save({"state_dict": state, "spec_digest": spec_digest, "epoch": epoch,
                    "weights_digest": wd}, p)
        return Checkpoint.of(p, _file_digest(p), spec_digest=spec_digest, epoch=epoch,
                             weights_digest=wd)

    def version(self, path) -> Checkpoint:
        """The Checkpoint of a stored file, its weights digest verified — what a bundle
        stamps as `model_version` without building the network."""
        blob, digest = _read(path, None, None)
        return Checkpoint.of(path, digest, spec_digest=blob.get("spec_digest"),
                             epoch=blob.get("epoch"), weights_digest=blob["weights_digest"])

    def load(self, ref: Checkpoint | str, spec: NetSpec, build: _ModelBuilder, *,
             expected_digest: str | None = None, run_dir=None) -> _Trained:
        """Rebuild the network through the given builder and load the verified state into it.

        `ref` is a path or a `Checkpoint`; a Checkpoint pins its file digest and its weights
        digest, a path is verified against the weights digest the file carries. `spec` (the
        NetSpec the checkpoint was trained under: the network is built from it, never guessed
        from the weights) and `build` (the Model port's `build_model(spec, run_dir=...)` — a
        parameter, not an import, so the store verifies and restores weights without knowing
        what a network is) are required; a file digest to pin and the run directory a custom
        block reads are the options. The model comes back in eval mode.
        """
        path, pinned, want_weights = _target(ref, expected_digest)
        blob, _digest = _read(path, pinned, want_weights)
        spec = NetSpec.require(spec)
        if not blob.get("spec_digest"):
            raise ValueError(f"{path} records no spec digest, so the spec it was trained under "
                             "cannot be checked; re-save it through save(model, path, spec)")
        if blob["spec_digest"] != spec.digest:
            raise ValueError(f"the checkpoint was trained under spec {blob['spec_digest']} but "
                             f"{spec.digest} was given")
        model = build(spec, run_dir=run_dir)
        _restore(model, blob["state_dict"], path)
        model.eval()
        return model
