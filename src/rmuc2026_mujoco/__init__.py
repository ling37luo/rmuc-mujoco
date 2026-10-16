"""Deprecated 1.x import name for :mod:`rmuc_mujoco`.

``rmuc2026_mujoco`` and every ``rmuc2026_mujoco.<module>`` resolve to the same
module objects as the renamed package, so ``isinstance`` checks, monkeypatching
and ``python -m rmuc2026_mujoco.<module>`` keep working.  New code should import
``rmuc_mujoco``; this alias will be removed in a later major release.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from types import ModuleType

import rmuc_mujoco as _target
from rmuc_mujoco import *  # noqa: F401,F403

__all__ = list(_target.__all__)
__version__ = _target.__version__

_TARGET = _target.__name__
_PREFIX = __name__ + "."

# Old flat module names whose code now lives elsewhere in ``rmuc_mujoco``.
_MOVED: dict[str, str] = {
    name: f"{_TARGET}.seasons.rmuc2026.{name}"
    for name in (
        "fly_batch",
        "fly_demo",
        "fly_progress",
        "fly_routes",
        "fly_runtime",
        "fly_view",
        "jump_probe",
        "ramp_audit",
        "ramp_source_audit",
        "wheel_probe",
    )
}


def _target_name(fullname: str) -> str:
    tail = fullname[len(_PREFIX) :]
    head, _, rest = tail.partition(".")
    moved = _MOVED.get(head)
    if moved is None:
        return f"{_TARGET}.{tail}"
    return f"{moved}.{rest}" if rest else moved


class _AliasLoader(importlib.abc.Loader):
    """Bind an old module name to the already-imported new module object."""

    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        module = importlib.import_module(self._target)
        self._target_spec = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The import system sets ``__spec__`` on whatever create_module
        # returned; keep the real module's own spec so its relative imports
        # and reloads still resolve under the new package.
        module.__spec__ = self._target_spec

    # ``python -m`` asks for code to run as ``__main__``.  Run the target
    # module under its own name so relative imports resolve from its real
    # package, including modules that moved into ``seasons``.
    def get_code(self, fullname: str):
        source = f"import runpy\nrunpy.run_module({self._target!r}, run_name='__main__', alter_sys=True)\n"
        return compile(source, f"<alias of {self._target}>", "exec")


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        name = _target_name(fullname)
        target_spec = importlib.util.find_spec(name)
        if target_spec is None:
            return None
        # ``runpy`` reports ``origin`` as ``sys.argv[0]``.
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(name),
            origin=target_spec.origin,
            is_package=target_spec.submodule_search_locations is not None,
        )


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
