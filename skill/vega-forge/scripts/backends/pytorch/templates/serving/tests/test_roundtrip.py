"""
tests/test_roundtrip.py — the bundle reproduces itself, from inside the bundle.

    python3 -m pytest tests/test_roundtrip.py        # or: python3 tests/test_roundtrip.py

Two claims, both checkable without the chain that built this: every file `MANIFEST.files`
names still hashes to what it hashed to when the bundle was written, and the predictor's answers
to `examples/rows.csv` still equal `examples/expected.parquet` — the comparison is the bundle's
own `check_examples` (the one `predict.py --check`, the service's self-test and the notebook
make): at each `__row__`, every field MANIFEST.output compares within the round-trip tolerance
the manifest records, every field it decides exactly, no compared row unanswered. The first
claim is what makes the second mean anything — a bundle whose weights were swapped would
reproduce its own new answers perfectly.

The pack-time round trip covered every test row against /vega-eval's stored predictions; this
one is what ships, so it runs on the examples and takes seconds.

A third claim is the serving seam: the Predictor and the inspector hold the network as the
blueprint's Model port (`ports.py`, the port `blueprint-map.json` gives the role `Model`) and
read it through that port alone, so a stand-in carrying the port's methods and nothing else
reproduces the examples, and opens the inspector, exactly as the network does.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
TOL = float(MANIFEST["roundtrip"]["tolerance"]["value"])


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def test_files_are_the_ones_that_were_packed():
    bad = []
    for name, recorded in (MANIFEST.get("files") or {}).items():
        path = ROOT / name
        if not path.is_file():
            bad.append(f"{name}: missing")
        elif _digest(path) != recorded:
            bad.append(f"{name}: {_digest(path)} != {recorded}")
    assert not bad, "files changed since the bundle was packed: " + "; ".join(bad)


def _model_port():
    """The blueprint's Model port, as this bundle's ports.py declares it."""
    package = MANIFEST["package"]
    entries = json.loads((ROOT / package / "blueprint-map.json").read_text(encoding="utf-8"))
    name = next(e["port"] for e in entries["entries"] if e.get("role") == "Model")
    return getattr(importlib.import_module(f"{package}.ports"), name)


def _port_only(network, port):
    """`network` behind `port`: one forwarding method per port method, no other attribute."""
    names = [n for n, v in vars(port).items() if callable(v) and not n.startswith("_")]
    forward = {n: (lambda n: lambda self, *a, **k: getattr(self._network, n)(*a, **k))(n)
               for n in names}
    stand_in = type("PortOnly", (), {"__slots__": ("_network",), **forward})()
    stand_in._network = network
    return stand_in


def test_the_network_is_reached_through_its_port_alone():
    import predict as cli
    port, restore = _model_port(), cli._restore
    cli._restore = lambda *args: _port_only(restore(*args), port)
    try:
        predictor = cli.load(ROOT)
        assert type(predictor.model).__name__ == "PortOnly"
        # the stand-in carries the port's methods and no other attribute (the Protocols are
        # static interfaces: an isinstance test would check names only)
        assert set(type(predictor.model).__dict__) - {"__slots__", "_network", "__module__",
                                                      "__dict__", "__weakref__", "__doc__"} \
            == {n for n, v in vars(port).items() if callable(v) and not n.startswith("_")}
        result = cli.check_examples(ROOT, predictor)
        assert result["passed"], "; ".join(result["problems"])
        try:
            status = cli.load_inspector(ROOT).status()
        except LookupError:                  # packed with --no-inspect: nothing to open
            return
        assert status["ready"], status.get("why")
    finally:
        cli._restore = restore


def test_predictions_reproduce():
    from predict import check_examples
    result = check_examples(ROOT)
    assert result["passed"], (
        "the bundle no longer reproduces its own examples within "
        f"{TOL} ({MANIFEST['roundtrip']['tolerance']['rule']}): "
        + "; ".join(result["problems"]))
    # the rows compared are every row the manifest says carries an expected answer
    recorded = (MANIFEST.get("examples") or {}).get("compared_rows")
    assert recorded is None or result["rows"] == recorded, (
        f"{result['rows']} example row(s) compared; MANIFEST.examples records {recorded}")


if __name__ == "__main__":
    test_files_are_the_ones_that_were_packed()
    print("ok  every file hashes to what the manifest records")
    test_predictions_reproduce()
    print(f"ok  the predictor reproduces examples/expected.parquet within {TOL}")
    test_the_network_is_reached_through_its_port_alone()
    print("ok  the predictor and the inspector read the network through its Model port alone")
    print("3 checks passed")
