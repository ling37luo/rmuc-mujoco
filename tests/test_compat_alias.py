from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import rmuc_mujoco


def test_old_package_name_resolves_to_the_same_modules() -> None:
    import rmuc2026_mujoco
    import rmuc2026_mujoco.query as old_query
    from rmuc2026_mujoco.manifest import FieldAsset

    import rmuc_mujoco.query as new_query

    assert old_query is new_query
    assert FieldAsset is rmuc_mujoco.FieldAsset
    assert rmuc2026_mujoco.FieldAsset is rmuc_mujoco.FieldAsset
    assert set(rmuc2026_mujoco.__all__) == set(rmuc_mujoco.__all__)


def test_old_package_name_supports_python_dash_m() -> None:
    src = Path(rmuc_mujoco.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(src)}
    for module in ("cli", "wheel_probe", "acceptance"):
        result = subprocess.run(
            [sys.executable, "-m", f"rmuc2026_mujoco.{module}", "--help"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_old_package_name_rejects_missing_modules() -> None:
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rmuc2026_mujoco.does_not_exist")
