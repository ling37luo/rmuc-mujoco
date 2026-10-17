"""The source-build gate binds a retained build to one registered STEP release."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from rmuc_mujoco import FieldAsset, regression, source_build
from rmuc_mujoco.manifest import sha256_file
from rmuc_mujoco.seasons import RMUC2026, RMUC2027
from rmuc_mujoco.source_build import verified_source_glb
from rmuc_mujoco.stairs import build_stair_catalog


def _build(tmp_path: Path, source) -> Path:
    """A minimal passing build of ``source`` whose GLB is a small stand-in file."""

    root = tmp_path / "build"
    root.mkdir()
    (root / "field.glb").write_bytes(b"stand-in glb")
    manifest = {
        "artifact_type": "rmuc2026_official_field_mujoco_asset",
        "status": "PASS",
        "source": {"sha256": source.step.sha256, "size_bytes": source.step.size_bytes},
        "conversion": {
            "colored_intermediate_glb": {
                "file": "field.glb",
                "sha256": sha256_file(root / "field.glb"),
            },
            "main_floor_height_before_shift_m": 0.1,
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _pin_stand_in(monkeypatch, manifest: Path) -> dict:
    """Register both releases with the stand-in GLB, as each pins its own build."""

    glb = sha256_file(manifest.parent / "field.glb")
    releases = {
        source.step.sha256: replace(source, source_glb_sha256=glb)
        for source in (RMUC2026.default_source, RMUC2027.default_source)
    }
    monkeypatch.setattr(source_build, "source_for_step", lambda sha, size=None: releases.get(sha))
    return releases


def test_a_build_is_accepted_for_its_own_release_only(tmp_path: Path, monkeypatch) -> None:
    manifest = _build(tmp_path, RMUC2027.default_source)
    releases = _pin_stand_in(monkeypatch, manifest)
    rmuc2027 = releases[RMUC2027.default_source.step.sha256]
    rmuc2026 = releases[RMUC2026.default_source.step.sha256]

    glb, shift, manifest_sha256 = verified_source_glb(manifest, rmuc2027)
    assert glb == (manifest.parent / "field.glb").resolve()
    assert shift == pytest.approx(-0.1)
    assert manifest_sha256 == sha256_file(manifest)
    # Naming no release accepts any registered one.
    assert verified_source_glb(manifest)[0] == glb
    with pytest.raises(ValueError, match="is not rmuc2026 2.0.0"):
        verified_source_glb(manifest, rmuc2026)


def test_a_build_of_an_unregistered_step_or_another_glb_is_refused(tmp_path: Path) -> None:
    manifest = _build(tmp_path, RMUC2027.default_source)

    # The registry pins the real conversion's GLB, which the stand-in is not.
    with pytest.raises(ValueError, match="audited unsimplified"):
        verified_source_glb(manifest, RMUC2027.default_source)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["source"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="registered official STEP"):
        verified_source_glb(manifest)


def test_the_stair_check_refuses_a_release_without_reviewed_parts(
    field_asset_dir: Path, monkeypatch
) -> None:
    unreviewed = replace(RMUC2026.default_source, stair_parts=())
    monkeypatch.setattr(FieldAsset, "field_source", property(lambda self: unreviewed))

    # Refused before any source build is read, so none has to exist.
    with pytest.raises(ValueError, match="no reviewed stair parts"):
        build_stair_catalog(field_asset_dir, field_asset_dir / "absent.json")


def test_check_reports_stairs_unavailable_for_a_release_without_parts(
    field_asset_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    for name in (
        "check_public_interfaces",
        "check_surface_routes",
        "check_example_episodes",
        "run_tunnel_checks",
        "run_high_point_checks",
    ):
        monkeypatch.setattr(regression, name, lambda *_args, **_kwargs: {"status": "PASS"})
    stair_calls: list = []
    monkeypatch.setattr(
        regression,
        "run_stair_checks",
        lambda *args, **_kwargs: stair_calls.append(args) or {"status": "PASS"},
    )
    unreviewed = replace(RMUC2026.default_source, stair_parts=())
    monkeypatch.setattr(FieldAsset, "field_source", property(lambda self: unreviewed))

    report = regression.run_regression(
        field_asset_dir, "source.json", output=tmp_path / "suite.json"
    )

    section = report["sections"]["stairs"]
    assert section["status"] == "NOT_AVAILABLE"
    assert section["reason"] == "source release has no reviewed stair parts"
    assert not stair_calls
