"""schema/runtime-pack.schema.json describes the manifests FieldAsset accepts.

conftest.py checks every manifest a test opens against the schema; these tests
check the schema itself and that it refuses manifests that break the contract.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from rmuc_mujoco.pack import export_runtime_asset_pack

from test_pack_profiles import _synthetic_source_build

jsonschema = pytest.importorskip("jsonschema")

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema/runtime-pack.schema.json"


def _validator():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture
def livery_manifest(tmp_path: Path) -> dict[str, Any]:
    source = _synthetic_source_build(tmp_path / "source", include_surface_guide=True)
    export_runtime_asset_pack(source, tmp_path / "pack", include_surface_guide=True)
    return json.loads((tmp_path / "pack" / "manifest.json").read_text(encoding="utf-8"))


def _profile(name: str) -> dict[str, object]:
    return dict(
        entrypoint=f"rmuc2026_field_{name}.xml",
        includes_visual_meshes=True,
        visual_mesh_count=1,
        intended_use="maximum_fidelity_visualization",
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda m: m.update(schema_version=5), id="unknown schema version"),
        pytest.param(lambda m: m.update(schema_version=1), id="schema 1 with a livery"),
        pytest.param(
            lambda m: m["runtime_profiles"]["profiles"].update(pro=_profile("pro")),
            id="pro profile without visual_pro",
        ),
        pytest.param(
            lambda m: m["runtime_profiles"]["profiles"].update(fast=_profile("fast")),
            id="unknown profile",
        ),
        pytest.param(
            lambda m: m["visual_layers"]["livery"].update(
                mesh_chunks=[dict(file="visual/x.obj", sha256="0" * 64, faces=1, lod=0)]
            ),
            id="livery piece with an unknown field",
        ),
        pytest.param(
            lambda m: m["collision"].update(kind="exact_brep"), id="unknown collision kind"
        ),
        pytest.param(
            lambda m: m["distribution"].update(
                safe_to_publish_without_rightsholder_permission=True
            ),
            id="publishable assets",
        ),
    ],
)
def test_the_schema_refuses_manifests_that_break_the_contract(
    livery_manifest: dict[str, Any], mutate: Callable[[dict[str, Any]], None]
) -> None:
    validator = _validator()
    assert not list(validator.iter_errors(livery_manifest))
    manifest = copy.deepcopy(livery_manifest)
    mutate(manifest)

    assert list(validator.iter_errors(manifest))
