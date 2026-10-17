"""The 2027 high point: measurement, contract, probes and route wiring.

Every field here is synthetic, at 1 cm.  The official pack is measured by the
``high-point-check`` command against the real build, not by this suite.
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rmuc_mujoco import FieldAsset, regression, slope_runtime
from rmuc_mujoco.cli import main
from rmuc_mujoco.high_point import (
    PATCH_STEP_M,
    _cad_offset,
    _patch,
    _platform_plane,
    _route,
    _runs,
    build_high_point_catalog,
    high_points_for,
    load_high_point_catalog,
    measure_flight,
    measure_high_point,
)
from rmuc_mujoco.high_point_check import (
    BLOCK_TOLERANCE_M,
    CONTACT_M,
    check_levels,
    check_riser,
    run_high_point_checks,
)
from rmuc_mujoco.manifest import sha256_file
from rmuc_mujoco.pack import export_runtime_asset_pack
from rmuc_mujoco.query import load_heightfield
from rmuc_mujoco.seasons import HIGH_POINTS, RMUC2026, RMUC2027, HighPoint
from rmuc_mujoco.slope_batch import run_slope_episode
from rmuc_mujoco.slope_demo import STEP_ROVER_WHEEL_RADIUS_M
from rmuc_mujoco.slope_runtime import CATALOG_LABELS, SlopeSession, load_route_catalog
from test_pack_profiles import _heightfield_bootstrap_png, _synthetic_source_build

# The highland flight of figure 4-17 as the drawing defines it: a platform and
# three 200 mm risers with 300 mm treads, descending along +x.
PLATFORM_TOP_M = 0.80
RISE_M = 0.20
TREAD_M = 0.30
FACES_M = (0.00, 0.30, 0.60)
# What a riser looks like when the face is missing: the rise spread over a run
# instead of standing in one cell.  The probe's sled can climb this.
RAMP_RUN_M = 0.30
AXIS = np.round(np.arange(-2.0, 3.0 + 1e-9, 0.01), 6)
LATERAL = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.01), 6)
# The platform has to be the widest band inside the box, as it is on the field.
SEARCH_BOX_M = (-1.0, 0.9, -0.5, 0.5)

FLIGHT = HighPoint(
    high_point_id="synthetic",
    search_box_xy_m=SEARCH_BOX_M,
    platform_top_m=PLATFORM_TOP_M,
    step_rise_m=RISE_M,
    step_tread_m=TREAD_M,
    highland_level_m=RISE_M,
    highland_step_count=3,
    floor_step_count=4,
    highland_flight_xy=(1.0, 0.0),
    floor_flight_xy=(0.0, -1.0),
)


def _flight_height(x: np.ndarray, *, ramped: bool, drop: float) -> np.ndarray:
    """One row of the flight: platform, three risers and the highland landing."""

    platform = PLATFORM_TOP_M - drop
    height = np.full(x.shape, platform)
    for index, face in enumerate(FACES_M):
        after = platform - (index + 1) * RISE_M
        if ramped:
            fraction = np.clip((x - face) / RAMP_RUN_M, 0.0, 1.0)
            target = after + (1.0 - fraction) * RISE_M
        else:
            target = np.full(x.shape, after)
        height = np.where(x >= face, np.minimum(height, target), height)
    return height


def _pack(tmp_path: Path, name: str, *, ramped: bool = False, drop: float = 0.0) -> Path:
    """A 1 cm pack whose highland flight is either stepped or a ramp."""

    source = _synthetic_source_build(tmp_path / f"{name}_source")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # height_m is indexed [y, x], so the profile runs across the columns.
    profile = _flight_height(AXIS, ramped=ramped, drop=drop)
    height = np.repeat(profile[None, :], len(LATERAL), axis=0)
    collision = manifest["collision"]
    np.savez_compressed(source / collision["samples_file"], x_m=AXIS, y_m=LATERAL, height_m=height)
    (source / collision["image_file"]).write_bytes(
        _heightfield_bootstrap_png(height, float(height.max()))
    )
    collision.update(
        rows_y=len(LATERAL),
        columns_x=len(AXIS),
        half_size_xy_m=[2.5, 1.0],
        resolution_m=0.01,
        maximum_height_m=float(height.max()),
        minimum_height_m=float(height.min()),
        geom_center_after_translation_m=[0.5, 0.0, 0.0],
        samples_sha256=sha256_file(source / collision["samples_file"]),
        image_sha256=sha256_file(source / collision["image_file"]),
    )
    manifest["dimensions"] = {
        "official_core_battlefield_m": [5.0, 2.0],
        "cad_assembly_outer_bounds_after_translation_m": [
            [-2.0, -1.0, float(height.min())],
            [3.0, 1.0, float(height.max())],
        ],
        "cad_assembly_outer_extents_m": [5.0, 2.0, 1.0],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pack = tmp_path / name
    export_runtime_asset_pack(source, pack)
    return pack


@pytest.fixture
def stepped_pack(tmp_path: Path) -> Path:
    return _pack(tmp_path, "stepped")


def _flight(asset: FieldAsset) -> dict:
    """Measure the highland flight the way the catalog does."""

    grid = load_heightfield(asset)
    mesh_x, mesh_y, mesh_z = _patch(grid, _cad_offset(asset), SEARCH_BOX_M, PATCH_STEP_M)
    plane = _platform_plane(mesh_x, mesh_y, mesh_z)
    return measure_flight(grid, FLIGHT, plane, flight="highland")


def _catalog(asset: FieldAsset, **overrides) -> dict:
    document = {
        "scenario_id": "high_point_basic",
        "source_manifest_sha256": asset.manifest_sha256,
        "heightfield_samples_sha256": asset.collision["samples_sha256"],
        "patches": [{"patch_id": "high_point_synthetic_highland", "route": {}}],
    }
    document.update(overrides)
    return document


def test_the_level_detector_reads_a_step_flight_in_order() -> None:
    """The detector sees four levels and three steps, not one long ramp."""

    along = np.round(np.arange(0.0, 2.0 + 1e-9, 0.005), 6)
    heights = _flight_height(along - 0.5, ramped=False, drop=0.0)
    runs = _runs(along, heights)

    assert [round(run["level_m"], 4) for run in runs] == [0.8, 0.6, 0.4, 0.2]
    assert [round(run["length_m"], 4) for run in runs[:3]] == [0.5, TREAD_M, TREAD_M]
    assert runs[-1]["length_m"] == pytest.approx(0.9, abs=0.01)


def test_the_measured_flight_matches_the_record(stepped_pack: Path) -> None:
    flight = _flight(FieldAsset.open(stepped_pack, verify=True))

    assert [round(value, 4) for value in flight["risers_m"]] == [RISE_M] * 3
    assert [round(value, 4) for value in flight["treads_m"]] == [TREAD_M] * 2
    assert flight["platform_level_m"] == pytest.approx(PLATFORM_TOP_M, abs=1e-6)
    assert flight["landing_level_m"] == pytest.approx(PLATFORM_TOP_M - 3 * RISE_M, abs=1e-6)
    assert flight["direction_xy"] == [1.0, 0.0]
    # The half-rise crossing sits within a couple of cells of the drawn face.
    for measured, face in zip(flight["riser_positions_m"], (0.5, 0.8, 1.1)):
        assert measured == pytest.approx(face, abs=0.02)


def test_a_field_whose_risers_are_ramps_fails_the_riser_probe(tmp_path: Path) -> None:
    """The gate this check exists for: a 200 mm slope is not a 200 mm step."""

    asset = FieldAsset.open(_pack(tmp_path, "reference"), verify=True)
    flight = _flight(asset)
    stepped = check_riser(asset, "synthetic", flight, "collision_only", load_heightfield(asset))

    assert stepped["status"] == "PASS", stepped
    assert stepped["reached_face"] and stepped["blocked_by_face"]
    assert stepped["climbed_m"] == pytest.approx(CONTACT_M, abs=BLOCK_TOLERANCE_M)
    assert stepped["highest_gain_m"] <= 0.05

    ramped = FieldAsset.open(_pack(tmp_path, "ramped", ramped=True), verify=True)
    probe = check_riser(ramped, "synthetic", flight, "collision_only", load_heightfield(ramped))

    assert probe["status"] == "FAIL"
    assert probe["reached_face"] is True
    assert probe["blocked_by_face"] is False
    assert probe["climbed_m"] > CONTACT_M + BLOCK_TOLERANCE_M
    assert probe["highest_gain_m"] > 0.05


def test_a_field_whose_levels_are_low_fails_the_level_probe(tmp_path: Path) -> None:
    asset = FieldAsset.open(_pack(tmp_path, "reference_2"), verify=True)
    flight = _flight(asset)
    rows = check_levels(asset, "synthetic", flight, "collision_only", load_heightfield(asset))

    assert rows and all(row["status"] == "PASS" for row in rows)
    assert all(abs(row["support_error_m"]) <= 0.005 for row in rows)

    lowered = FieldAsset.open(_pack(tmp_path, "lowered", drop=0.05), verify=True)
    low_rows = check_levels(
        lowered, "synthetic", flight, "collision_only", load_heightfield(lowered)
    )

    assert low_rows and all(row["status"] == "FAIL" for row in low_rows)
    # Each level is still a level and still supports the block; it simply is not
    # at the height the record declares.
    assert all(abs(row["support_error_m"]) <= 0.005 for row in low_rows)
    assert all(row["level_error_m"] == pytest.approx(-0.05, abs=0.005) for row in low_rows)


def test_the_route_runs_from_the_landing_up_to_the_platform(stepped_pack: Path) -> None:
    """The runtime spawns a climb at low_xyz_m facing uphill_unit_xy.

    Both stair and slope routes name their ends by height, so a route that
    swapped them would send the robot away from the steps: on this flight it
    would drive along the platform top and still report a completed traversal.
    """

    asset = FieldAsset.open(stepped_pack, verify=True)
    flight = _flight(asset)
    route = _route(load_heightfield(asset), FLIGHT, flight, None)

    low = np.array(route["low_xyz_m"])
    high = np.array(route["high_xyz_m"])
    uphill = np.array(route["uphill_unit_xy"])
    assert high[2] > low[2]
    assert high[2] - low[2] == pytest.approx(route["height_gain_m"], abs=1e-9)
    # Height grows along the named uphill direction, which is also the heading.
    assert (high[:2] - low[:2]) @ uphill == pytest.approx(route["length_m"], abs=1e-6)
    assert math.atan2(uphill[1], uphill[0]) == pytest.approx(route["heading_yaw_rad"])
    # The high end is the platform, the low end the landing below the flight.
    assert high[2] == pytest.approx(flight["platform_level_m"], abs=0.01)
    assert low[2] == pytest.approx(flight["landing_level_m"], abs=0.01)
    # Every waypoint climbs, so no sampled point sits above the platform.
    waypoints = np.array(route["waypoints_xyz_m"])
    assert np.all(np.diff(waypoints[:, 2]) >= -1e-9)
    assert waypoints[:, 2].max() <= high[2] + 1e-9


def _climb_session(asset: FieldAsset) -> SlopeSession:
    """A session on the synthetic flight's route, with the built-in rover."""

    route = _route(load_heightfield(asset), FLIGHT, _flight(asset), None)
    catalog = _catalog(asset, patches=[{"patch_id": route["route_id"], "route": route}])
    return SlopeSession(asset, mode="policy", route_catalog={"catalog": catalog})


def _climb(session: SlopeSession, direction: str) -> dict:
    case = {
        "case_id": f"climb_{direction}",
        "case_index": 0,
        "patch": session.route["route_id"],
        "direction": direction,
        "speed_mps": 0.3,
        "repeat": 0,
        "seed": 0,
        "timeout_s": 20.0,
    }
    return run_slope_episode(session, case)


def test_the_step_rover_climbs_a_flight_the_four_wheel_rover_cannot(
    stepped_pack: Path, monkeypatch
) -> None:
    """The drawn flight is climbable by a rover whose wheels stand taller than a riser."""

    # The synthetic pack carries the 2026 STEP identity; the scenario is 2027's.
    monkeypatch.setattr(FieldAsset, "season", property(lambda self: RMUC2027))
    asset = FieldAsset.open(stepped_pack, verify=True)
    session = _climb_session(asset)
    assert session.robot_kind == "example_step_rover"
    # The route end keeps the front wheels clear of the first riser, so the rover
    # spawns standing on the landing instead of propped up on a step.
    assert session.initial_pose[2] == pytest.approx(
        session.route["low_xyz_m"][2] + STEP_ROVER_WHEEL_RADIUS_M, abs=0.01
    )
    for direction in ("uphill", "downhill"):
        episode = _climb(session, direction)
        assert episode["outcome"] == "PASS", (direction, episode["failure_reason"])
        assert episode["physics_status"] == "PASS"

    # The original rover's 160 mm wheels meet a 200 mm face.
    monkeypatch.setattr(slope_runtime, "EXAMPLE_ROVERS", {})
    original = _climb_session(asset)
    assert original.robot_kind == "example_four_wheel_rover"
    assert _climb(original, "uphill")["outcome"] != "PASS"


def test_a_box_without_the_flight_in_it_is_rejected(stepped_pack: Path) -> None:
    asset = FieldAsset.open(stepped_pack, verify=True)
    moved = replace(FLIGHT, search_box_xy_m=(2.0, 2.6, -0.5, 0.5))

    with pytest.raises(ValueError, match="leaves the heightfield"):
        measure_high_point(asset, moved)


def test_the_2027_records_describe_two_perpendicular_flights() -> None:
    assert RMUC2027.has(HIGH_POINTS) and not RMUC2026.has(HIGH_POINTS)
    high_points = RMUC2027.default_source.high_points
    assert len(high_points) == 2
    for high_point in high_points:
        assert (high_point.highland_step_count, high_point.floor_step_count) == (3, 4)
        assert high_point.step_rise_m == RISE_M and high_point.step_tread_m == TREAD_M
        highland = np.array(high_point.highland_flight_xy)
        floor = np.array(high_point.floor_flight_xy)
        # Both flights leave the platform at a right angle, so one high point
        # offers the highland route and the base-floor route from one corner.
        assert np.isclose(np.linalg.norm(highland), 1.0)
        assert np.isclose(np.linalg.norm(floor), 1.0)
        assert abs(float(np.dot(highland, floor))) < 1e-9


def test_the_checks_refuse_a_field_without_high_points(stepped_pack: Path) -> None:
    with pytest.raises(ValueError, match="no high points"):
        run_high_point_checks(stepped_pack, profiles=("collision_only",))
    with pytest.raises(ValueError, match="no high points"):
        build_high_point_catalog(stepped_pack)
    with pytest.raises(ValueError, match="no high points"):
        high_points_for(FieldAsset.open(stepped_pack))


def test_the_high_point_check_command_reports_the_missing_feature(
    stepped_pack: Path, tmp_path: Path, capsys
) -> None:
    code = main(
        [
            "high-point-check",
            str(stepped_pack),
            "--output",
            str(tmp_path / "check.json"),
        ]
    )

    assert code == 2
    assert "no high points" in capsys.readouterr().err
    assert not (tmp_path / "check.json").exists()


def test_a_route_catalog_is_only_read_for_the_pack_it_measured(stepped_pack: Path) -> None:
    asset = FieldAsset.open(stepped_pack, verify=True)

    loaded = load_high_point_catalog(asset, {"catalog": _catalog(asset)})
    assert loaded["scenario_id"] == "high_point_basic"
    for overrides, message in (
        ({"scenario_id": "stairs_basic"}, "does not match"),
        ({"source_manifest_sha256": "0" * 64}, "does not match"),
        ({"heightfield_samples_sha256": "0" * 64}, "does not match"),
        ({"patches": []}, "no routes"),
    ):
        with pytest.raises(ValueError, match=message):
            load_high_point_catalog(asset, {"catalog": _catalog(asset, **overrides)})
    with pytest.raises(ValueError, match="requires --route-catalog"):
        load_high_point_catalog(asset, None)


def test_the_route_loader_picks_the_feature_from_the_catalog(
    stepped_pack: Path, monkeypatch
) -> None:
    """One session drives whichever catalog it is handed."""

    asset = FieldAsset.open(stepped_pack, verify=True)
    assert set(CATALOG_LABELS) == {"stairs_basic", "high_point_basic"}
    assert load_route_catalog(asset, {"catalog": _catalog(asset)})["scenario_id"] == (
        "high_point_basic"
    )

    seen = []
    monkeypatch.setattr(
        "rmuc_mujoco.stairs.load_stair_catalog",
        lambda field, document: seen.append(document) or {"scenario_id": "stairs_basic"},
    )
    unnamed = {"catalog": _catalog(asset, scenario_id=None)}
    assert load_route_catalog(asset, unnamed)["scenario_id"] == "stairs_basic"
    assert seen == [unnamed]


def _regression(pack: Path, tmp_path: Path, monkeypatch, season, name: str):
    """Run the suite with the field stages stubbed and the season chosen."""

    calls: list = []
    examples: list = []

    def stage(*_args, **_kwargs):
        return {"status": "PASS"}

    for attribute in (
        "check_public_interfaces",
        "check_surface_routes",
        "run_stair_checks",
        "run_tunnel_checks",
    ):
        monkeypatch.setattr(regression, attribute, stage)
    monkeypatch.setattr(
        regression,
        "check_example_episodes",
        lambda *args: examples.append(args) or {"status": "PASS"},
    )
    monkeypatch.setattr(
        regression,
        "run_high_point_checks",
        lambda asset, **kwargs: calls.append(kwargs)
        or {"status": "PASS", "catalog": {"patches": ["climb"]}},
    )
    if season is not None:
        monkeypatch.setattr(regression.FieldAsset, "season", property(lambda self: season))
    report = regression.run_regression(pack, "source.json", output=tmp_path / f"{name}.json")
    return report, calls, examples


def test_the_regression_runs_the_high_points_stage_on_a_2027_field(
    stepped_pack: Path, tmp_path: Path, monkeypatch
) -> None:
    report, calls, examples = _regression(
        stepped_pack, tmp_path, monkeypatch, RMUC2027, "suite_2027"
    )

    assert report["sections"]["high_points"]["status"] == "PASS"
    assert calls and calls[0]["profiles"] and calls[0]["source_manifest"] == "source.json"
    # The published climbs reach the example robots, which run after the stage.
    assert examples and examples[0][-1] == {"patches": ["climb"]}


def test_the_regression_reports_no_high_points_stage_on_a_2026_field(
    stepped_pack: Path, tmp_path: Path, monkeypatch
) -> None:
    report, calls, examples = _regression(stepped_pack, tmp_path, monkeypatch, None, "suite_2026")

    section = report["sections"]["high_points"]
    assert section["status"] == "NOT_AVAILABLE"
    assert section["reason"] == "season has no high points"
    assert not calls
    assert "high_points" in report["unavailable_sections"]
    assert examples and examples[0][-1] is None
