from __future__ import annotations

import json
from pathlib import Path

import pytest

import rmuc_mujoco.download as download
from rmuc_mujoco import FieldAsset
from rmuc_mujoco.errors import ManifestError
from rmuc_mujoco.seasons import (
    DEFAULT_SEASON_ID,
    FLY_RAMPS,
    RMUC2026,
    RMUC2027,
    SEASONS,
    get_season,
    season_of,
    source_for_step,
)


def _rewrite_source_identity(root: Path, **fields: object) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source_identity"].update(fields)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_registry_resolves_seasons_and_step_identities() -> None:
    assert DEFAULT_SEASON_ID == "rmuc2026"
    assert get_season("rmuc2026") is RMUC2026
    with pytest.raises(ValueError, match="known seasons: rmuc2026, rmuc2027"):
        get_season("rmuc1999")

    source = RMUC2026.default_source
    assert source_for_step(source.step.sha256) is source
    assert source_for_step(source.step.sha256, source.step.size_bytes) is source
    assert source_for_step(source.step.sha256, source.step.size_bytes + 1) is None
    assert source_for_step("0" * 64) is None
    assert season_of(source) is RMUC2026


def test_rmuc2027_is_registered_without_2026_features() -> None:
    from rmuc_mujoco.scenarios import BASE_SCENARIOS, scenarios_for

    source = RMUC2027.default_source
    assert get_season("rmuc2027") is RMUC2027
    assert source.version == "1.0.0" and source.step_product == "00_RMUC20270001_ASM"
    assert source_for_step(source.step.sha256, source.step.size_bytes) is source
    assert not RMUC2027.has(FLY_RAMPS)
    assert RMUC2027.surface_guide is None
    # The 2027 table keeps every core scenario and adds the high point and tunnels.
    table = scenarios_for(RMUC2027)
    assert set(table) == set(BASE_SCENARIOS) | {"high_point_basic", "tunnel_basic"}
    assert all(table[name] is spec for name, spec in BASE_SCENARIOS.items())


def test_every_registered_source_has_a_unique_step_identity() -> None:
    sources = [source for season in SEASONS.values() for source in season.sources]
    assert len({source.step.sha256 for source in sources}) == len(sources)
    for season_id, season in SEASONS.items():
        assert season.id == season_id
        assert season.sources
        assert all(source.season_id == season_id for source in season.sources)


def test_1x_download_constants_still_name_the_2026_pins() -> None:
    source = RMUC2026.default_source
    assert download.OFFICIAL_STEP_SHA256 == source.step.sha256
    assert download.OFFICIAL_STEP_SIZE == source.step.size_bytes
    assert download.OFFICIAL_STEP_PRODUCT == source.step_product
    assert download.OFFICIAL_RULEBOOK_V2_0_0_SHA256 == RMUC2026.surface_guide.file.sha256


def test_runtime_pack_schema_lists_exactly_the_registered_steps() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "schema/runtime-pack.schema.json"
    identity = json.loads(schema_path.read_text(encoding="utf-8"))["properties"]["source_identity"][
        "properties"
    ]
    sources = [source for season in SEASONS.values() for source in season.sources]
    assert set(identity["official_step_sha256"]["enum"]) == {s.step.sha256 for s in sources}
    assert set(identity["season"]["enum"]) == set(SEASONS)


def test_pack_without_season_fields_is_read_from_its_step(field_asset_dir: Path) -> None:
    asset = FieldAsset.open(field_asset_dir)

    assert "season" not in asset.manifest["source_identity"]
    assert asset.season is RMUC2026
    assert asset.field_source is RMUC2026.default_source


def test_recorded_season_must_agree_with_the_step(field_asset_dir: Path) -> None:
    _rewrite_source_identity(field_asset_dir, season="rmuc2026", official_step_version="2.0.0")
    assert FieldAsset.open(field_asset_dir).season is RMUC2026

    _rewrite_source_identity(field_asset_dir, season="rmuc2027")
    with pytest.raises(ManifestError, match="season disagrees"):
        FieldAsset.open(field_asset_dir)

    _rewrite_source_identity(field_asset_dir, season="rmuc2026", official_step_version="9.9.9")
    with pytest.raises(ManifestError, match="STEP version disagrees"):
        FieldAsset.open(field_asset_dir)


def test_unregistered_step_is_rejected(field_asset_dir: Path) -> None:
    _rewrite_source_identity(field_asset_dir, official_step_sha256="0" * 64)
    with pytest.raises(ManifestError, match="registered official STEP"):
        FieldAsset.open(field_asset_dir)


@pytest.fixture
def season_without_fly_ramps(monkeypatch: pytest.MonkeyPatch):
    import dataclasses

    season = dataclasses.replace(
        RMUC2026, id="no_fly", title="No-fly season", features=frozenset(), scenario_table=None
    )
    monkeypatch.setattr(FieldAsset, "season", property(lambda _self: season))
    return season


def test_season_without_fly_ramps_offers_only_core_scenarios(
    field_asset_dir: Path, season_without_fly_ramps
) -> None:
    from rmuc_mujoco.scenarios import BASE_SCENARIOS, scenario_descriptor, scenarios_for

    asset = FieldAsset.open(field_asset_dir)
    assert scenarios_for(season_without_fly_ramps) is BASE_SCENARIOS
    assert "fly_ramp_north" in scenarios_for(RMUC2026)
    with pytest.raises(ValueError, match="unknown scenario 'fly_ramp_north'"):
        scenario_descriptor(asset, "fly_ramp_north")
    slope = scenario_descriptor(asset, "slope_basic", profile="collision_only")
    assert "fly_ramps_excluded" not in slope["scenario"]["region"]


def test_fly_ramp_entry_points_report_the_missing_feature(
    field_asset_dir: Path, tmp_path: Path, season_without_fly_ramps
) -> None:
    from rmuc_mujoco import acceptance_routes, regression
    from rmuc_mujoco.training_region import export_training_region

    asset = FieldAsset.open(field_asset_dir)
    assert acceptance_routes._fly_ramp_scenarios(asset) == ()
    assert regression._fly_ramp_scenarios(asset) == ()
    with pytest.raises(ValueError, match="No-fly season fields have no fly ramps"):
        export_training_region(asset, tmp_path / "region", scenario_id="fly_ramp_north")
