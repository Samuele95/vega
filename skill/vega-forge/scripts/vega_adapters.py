"""
vega_adapters — one adapter resolver for every stage that drives an emitted package (D1 §12.1).

`/vega-train`'s trainer, `/vega-eval`'s evaluator, the emitter's smoke pass, the notebooks and
`/vega-pack` all open the package `/vega-forge` emitted and reach its adapters by the role the
blueprint map gives each file. Before v2.3 each carried a private copy of that lookup
(`trainer.py:_import_package`, `vega-eval/scripts/evaluate.py:import_package`, the smoke in
`emit_backend.py`), and the copies had already drifted (one knew the Evaluator, one did not).
`PackageAdapters.load(package_dir, package)` is the one copy:

  * it imports the package **fresh** from the directory's parent, so a re-emitted copy of the
    same package name (a rung that re-emitted code) shadows nothing it should not;
  * it reads `blueprint-map.json` and finds each port's module by its **role** — never by a file
    name the emitter happened to choose, since the module names are the blueprint's;
  * on a `v2.3` map it **instantiates the adapter classes** whose names the backend contract
    fixes (`Network`, `ShapedParts`, `Trainer`, `Evaluator`, `ModelRegistry`; D1 §4.2), wiring
    the Evaluator's ports by its factory (`Evaluator.restoring(data, store, build)`), and it
    finds `TaskReadout` in `<model>/readout.py`;
  * on a map without a generation (the v2.2 layout, the default until gate A) the adapter
    modules *are* the adapters — module-level functions — and there is no readout: the value is
    the same, so a consumer switched to it keeps working across the cutover.

A missing role, class or support module is an `AdapterLoadError` that names the file and what it
should have held; nothing is guessed.
"""
from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = ["PackageAdapters", "AdapterLoadError", "ADAPTER_CLASSES"]

#: the adapter class each port role's module holds in a v2.3 package (D1 §4.2: the names are fixed)
ADAPTER_CLASSES: Mapping[str, str] = {"Model": "Network", "DataSource": "ShapedParts",
                                      "Trainer": "Trainer", "Evaluator": "Evaluator",
                                      "ModelStore": "ModelRegistry"}

_REQUIRED_ROLES = ("Model", "DataSource", "Trainer", "ModelStore")


class AdapterLoadError(ImportError):
    """The package cannot be driven as its map says: the message names what is missing."""


@dataclass(frozen=True)
class _PackageTree:
    """The imported package and its map, asked by role. Private: `PackageAdapters` is the
    surface."""

    root: Path
    name: str
    module: Any
    entries: tuple
    generation: str

    @staticmethod
    def open(package_dir, package: str) -> "_PackageTree":
        root = Path(package_dir)
        map_path = root / "blueprint-map.json"
        try:
            doc = json.loads(map_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterLoadError(f"{map_path} cannot be read ({exc}); the adapters are found "
                                   "by the roles the blueprint map gives each file") from exc
        module = _fresh_import(root, package)
        return _PackageTree(root=root, name=package, module=module,
                            entries=tuple(e for e in doc.get("entries") or ()
                                          if isinstance(e, Mapping)),
                            generation=str(doc.get("generation") or "v2.2"))

    def entry(self, role: str) -> Mapping | None:
        return next((e for e in self.entries if e.get("role") == role), None)

    def import_file(self, rel: str) -> Any:
        dotted = self.name + "." + rel[:-3].replace("/", ".")
        try:
            return importlib.import_module(dotted)
        except ImportError as exc:
            raise AdapterLoadError(f"{self.root / rel} does not import ({exc})") from exc

    def role_module(self, role: str, required: bool = True) -> Any:
        entry = self.entry(role)
        if entry is None:
            if required:
                raise AdapterLoadError(f"{self.root / 'blueprint-map.json'} maps no file to the "
                                       f"{role} role; the package cannot be driven without it")
            return None
        return self.import_file(str(entry["file"]))

    @property
    def model_dir(self) -> str:
        return str(Path(str(self.entry("Model")["file"])).parent.as_posix())

    def support(self, stem: str, required: bool = True) -> Any:
        rel = f"{self.model_dir}/{stem}.py"
        if not (self.root / rel).is_file():
            if required:
                raise AdapterLoadError(f"{self.root / rel} is missing; a {self.generation} "
                                       f"package carries it beside the Model adapter")
            return None
        return self.import_file(rel)

    def trainer_support(self, stem: str) -> Any:
        """A support module of the Trainer's module (`optim`, `losses`), or None (v2.2)."""
        rel = f"{Path(str(self.entry('Trainer')['file'])).parent.as_posix()}/{stem}.py"
        return self.import_file(rel) if (self.root / rel).is_file() else None

    def baselines(self) -> Any:
        """The baseline module that holds `build`, the family's one entry (v2.3:
        `baselines_extra.py`, which extends `baselines.py`; a package where `baselines.py`
        holds it is read the same way); None without any baseline row."""
        rows = [e for e in self.entries if e.get("role") == "baseline"]
        if not rows:
            return None
        modules = [self.import_file(str(e["file"])) for e in rows]
        return next((m for m in modules if callable(getattr(m, "build", None))), modules[0])


def _fresh_import(root: Path, package: str) -> Any:
    """Import `package` from `root`'s parent, dropping any copy already imported."""
    parent = str(root.parent.resolve())
    if parent in sys.path:
        sys.path.remove(parent)
    sys.path.insert(0, parent)
    for name in list(sys.modules):
        if name == package or name.startswith(package + "."):
            del sys.modules[name]
    importlib.invalidate_caches()
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise AdapterLoadError(f"the package {package!r} does not import from {parent} "
                               f"({exc})") from exc


def _adapter_class(tree: _PackageTree, role: str, module: Any) -> Any:
    name = ADAPTER_CLASSES[role]
    cls = getattr(module, name, None)
    if not isinstance(cls, type):
        raise AdapterLoadError(f"{tree.root / tree.entry(role)['file']} holds no class {name!r}; "
                               f"a v2.3 {role} adapter is that class (backend-contract §1)")
    return cls


@dataclass(frozen=True)
class PackageAdapters:
    """The emitted package's adapters, reached by role and ready to call.

    `data`, `trainer`, `store` and `evaluator` are adapter instances on a v2.3 package and the
    adapter modules on a v2.2 one; `build` is the `ModelBuilder` (`Network.build`, or the v2.2
    module's `build_model`); `readout` is the `TaskReadout` class (None on v2.2); `evaluator` and
    `baselines` are None when the blueprint drew no Evaluator port.
    """

    package: Any
    data: Any
    trainer: Any
    evaluator: Any
    store: Any
    build: Callable
    network_module: Any
    metrics: Any
    baselines: Any
    readout: Any
    generation: str = "v2.2"
    optim: Any = None
    losses: Any = None

    @staticmethod
    def load(package_dir, package: str) -> "PackageAdapters":
        """Import `package` from `package_dir` (the package's own directory) and resolve every
        adapter by its blueprint-map role."""
        tree = _PackageTree.open(package_dir, package)
        mods = {role: tree.role_module(role) for role in _REQUIRED_ROLES}
        mods["Evaluator"] = tree.role_module("Evaluator", required=False)
        common = {"package": tree.module, "network_module": mods["Model"],
                  "metrics": tree.support("metrics"), "baselines": tree.baselines(),
                  "generation": tree.generation}
        if tree.generation == "v2.2":
            return PackageAdapters(data=mods["DataSource"], trainer=mods["Trainer"],
                                   evaluator=mods["Evaluator"], store=mods["ModelStore"],
                                   build=_v22_build(tree, mods["Model"]), readout=None,
                                   **common)
        return PackageAdapters._instantiated(tree, mods, common)

    @staticmethod
    def _instantiated(tree: _PackageTree, mods: Mapping, common: Mapping) -> "PackageAdapters":
        build = _adapter_class(tree, "Model", mods["Model"]).build
        data = _adapter_class(tree, "DataSource", mods["DataSource"])()
        store = _adapter_class(tree, "ModelStore", mods["ModelStore"])()
        evaluator = None
        if mods["Evaluator"] is not None:
            evaluator = _adapter_class(tree, "Evaluator", mods["Evaluator"]).restoring(data, store,
                                                                                       build)
        readout = getattr(tree.support("readout"), "TaskReadout", None)
        if not isinstance(readout, type):
            raise AdapterLoadError(f"{tree.root / tree.model_dir / 'readout.py'} holds no class "
                                   "'TaskReadout' (R-B7: the only source of output semantics)")
        return PackageAdapters(data=data, trainer=_adapter_class(tree, "Trainer",
                                                                 mods["Trainer"])(store),
                               evaluator=evaluator, store=store, build=build, readout=readout,
                               optim=tree.trainer_support("optim"),
                               losses=tree.trainer_support("losses"), **common)

    def device_for(self, spec) -> Any:
        """The device the package's Trainer trains `spec` on: its optimiser module's
        `device_for` (a v2.2 Trainer module's own function)."""
        return (self.optim or self.trainer).device_for(spec)

    def lrs_for(self, spec, steps_per_epoch: int = 1) -> list:
        """The per-epoch learning rates the fit will see (the optimiser module's
        `resolved_lrs`), resolved before any run."""
        if self.optim is None:
            return self.trainer.lrs_for(spec, steps_per_epoch)
        return self.optim.resolved_lrs(spec, steps_per_epoch)

    def loss_for(self, spec) -> Any:
        """The loss the fit trains with (`build_loss`), with its `formula()`."""
        if self.losses is None:
            return self.trainer.loss_for(spec)
        return self.losses.build_loss(spec)

    @property
    def roles(self) -> Mapping[str, Any]:
        """The port adapters by blueprint role (the Evaluator only when drawn)."""
        out = {"Model": self.build, "DataSource": self.data, "Trainer": self.trainer,
               "ModelStore": self.store}
        if self.evaluator is not None:
            out["Evaluator"] = self.evaluator
        return out


def _v22_build(tree: _PackageTree, network_module: Any) -> Callable:
    build = getattr(network_module, "build_model", None)
    if not callable(build):
        raise AdapterLoadError(f"{tree.root / tree.entry('Model')['file']} has no build_model; "
                               "the Model adapter must expose it (backend-contract §1)")
    return build
