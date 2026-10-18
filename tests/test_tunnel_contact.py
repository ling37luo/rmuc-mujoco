"""Tunnel roof layer: geometry extraction, pack contract and probe checks."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rmuc_mujoco import FieldAsset, ManifestError, regression, source_contact_pack
from rmuc_mujoco import tunnel_contact
from rmuc_mujoco.manifest import sha256_file
from rmuc_mujoco.pack import export_runtime_asset_pack
from rmuc_mujoco.query import HeightFieldData, find_spawn_candidates, load_heightfield
from rmuc_mujoco.seasons import RMUC2026, TunnelRoof
from rmuc_mujoco.slope_batch import run_slope_episode
from rmuc_mujoco.slope_runtime import SlopeSession
from rmuc_mujoco.tunnel_check import run_tunnel_checks
from rmuc_mujoco.tunnel_contact import RoofGeometry, export_tunnel_pack, roof_geometry
from rmuc_mujoco.tunnel_routes import ROUTE_MARGIN_M, build_tunnel_catalog
from test_pack_profiles import _heightfield_bootstrap_png, _synthetic_source_build

UNDERSIDE_M = 0.25
TOP_M = 0.27
# One roof per pinned 2026 tunnel id: (centre x, centre y, travel axis).
ROOFS = {
    "a_red": (-0.8, -0.8, "y"),
    "a_blue": (0.8, 0.8, "y"),
    "b_red": (-0.8, 0.8, "x"),
    "b_blue": (0.8, -0.8, "x"),
}


def _span(centre_x: float, centre_y: float, axis: str) -> tuple[float, float, float, float]:
    half_x, half_y = (0.175, 0.40) if axis == "x" else (0.40, 0.175)
    return (centre_x - half_x, centre_x + half_x, centre_y - half_y, centre_y + half_y)


def _geometry(tunnel_id: str) -> RoofGeometry:
    roof = next(r for r in RMUC2026.default_source.tunnel_roofs if r.tunnel_id == tunnel_id)
    return RoofGeometry(roof, _span(*ROOFS[tunnel_id]), (UNDERSIDE_M, 0.0, 0.0), (TOP_M, 0.0, 0.0))


def _roofed_pack(tmp_path: Path, half_size_m: float) -> Path:
    """A flat square field at 1 cm whose four roofs are still solid blocks."""

    source = _synthetic_source_build(tmp_path / "source")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    axis = np.round(np.arange(-half_size_m, half_size_m + 1e-9, 0.01), 6)
    height = np.zeros((len(axis), len(axis)))
    gx, gy = np.meshgrid(axis, axis)
    for tunnel_id in ROOFS:
        x0, x1, y0, y1 = _span(*ROOFS[tunnel_id])
        height[(gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)] = TOP_M
    collision = manifest["collision"]
    np.savez_compressed(source / collision["samples_file"], x_m=axis, y_m=axis, height_m=height)
    (source / collision["image_file"]).write_bytes(_heightfield_bootstrap_png(height, 1.0))
    collision.update(
        rows_y=len(axis),
        columns_x=len(axis),
        half_size_xy_m=[half_size_m, half_size_m],
        resolution_m=0.01,
        samples_sha256=sha256_file(source / collision["samples_file"]),
        image_sha256=sha256_file(source / collision["image_file"]),
    )
    size = 2 * half_size_m
    manifest["dimensions"] = {
        "official_core_battlefield_m": [size, size],
        "cad_assembly_outer_bounds_after_translation_m": [
            [-half_size_m, -half_size_m, 0.0],
            [half_size_m, half_size_m, 1.0],
        ],
        "cad_assembly_outer_extents_m": [size, size, 1.0],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pack = tmp_path / "pack"
    export_runtime_asset_pack(source, pack)
    return pack


@pytest.fixture
def roofed_pack(tmp_path: Path) -> Path:
    """A flat 3 m field at 1 cm whose four roofs are still solid blocks."""

    return _roofed_pack(tmp_path, 1.5)


@pytest.fixture
def stub_replacement(monkeypatch):
    """Replace source-GLB reading with the fixture's exact roof geometry."""

    def replacement(_source_build, runtime_pack):
        asset = (
            runtime_pack if isinstance(runtime_pack, FieldAsset) else FieldAsset.open(runtime_pack)
        )
        field = load_heightfield(asset)
        updated = field.height_m.copy()
        gx, gy = np.meshgrid(field.x_m, field.y_m)
        roofs, records = {}, []
        for roof in RMUC2026.default_source.tunnel_roofs:
            geometry = _geometry(roof.tunnel_id)
            x0, x1, y0, y1 = geometry.span_xy_m
            inside = (
                (gx >= x0 - 0.002) & (gx <= x1 + 0.002) & (gy >= y0 - 0.002) & (gy <= y1 + 0.002)
            )
            updated[inside] = 0.0
            rows, cols = np.nonzero(inside)
            roofs[roof.tunnel_id] = geometry.vertices()
            records.append(
                {
                    "tunnel_id": roof.tunnel_id,
                    "source_part_index": roof.source_part_index,
                    "span_world_xy_m": [[x0, y0], [x1, y1]],
                    "travel_axis": geometry.travel_axis,
                    "underside_m": [UNDERSIDE_M, UNDERSIDE_M],
                    "top_m": [TOP_M, TOP_M],
                    "floor_m": [0.0, 0.0],
                    "clearance_m": [UNDERSIDE_M, UNDERSIDE_M, UNDERSIDE_M],
                    "hfield_nodes_lowered": int(inside.sum()),
                    "changed_grid_rectangle_yx": [
                        int(rows.min()),
                        int(rows.max()),
                        int(cols.min()),
                        int(cols.max()),
                    ],
                }
            )
        report = {
            "source_manifest_sha256": asset.manifest["source_identity"]["field_manifest_sha256"],
            "source_glb_sha256": RMUC2026.default_source.source_glb_sha256,
            "roofs": records,
        }
        return HeightFieldData(field.x_m, field.y_m, updated), roofs, report

    monkeypatch.setattr(tunnel_contact, "build_tunnel_replacement", replacement)


@pytest.fixture
def tunnel_pack(roofed_pack: Path, stub_replacement, tmp_path: Path) -> Path:
    output = tmp_path / "tunnel_pack"
    export_tunnel_pack(tmp_path / "unused_source", roofed_pack, output)
    return output


@pytest.fixture
def wide_tunnel_pack(tmp_path: Path, stub_replacement) -> Path:
    """The same four roofs on a 6 m field, which leaves room for their routes."""

    wide = tmp_path / "wide"
    wide.mkdir()
    output = tmp_path / "wide_tunnel_pack"
    export_tunnel_pack(tmp_path / "unused_source", _roofed_pack(wide, 3.0), output)
    return output


def _rewrite(pack: Path, mutate) -> None:
    path = pack / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


# --- geometry extraction ---------------------------------------------------


def _bridge_part(trimesh):
    """A B-style roof: a 0.8 m channel span whose ends rest higher on rails."""

    middle = trimesh.creation.box(extents=(0.35, 0.80, 0.18))
    middle.apply_translation((0.0, 0.0, 0.22 + 0.09))
    ends = []
    for y in (-0.475, 0.475):
        end = trimesh.creation.box(extents=(0.35, 0.15, 0.09))
        end.apply_translation((0.0, y, 0.31 + 0.045))
        ends.append(end)
    return trimesh.util.concatenate([middle, *ends])


def test_roof_geometry_takes_the_lowest_underside_as_its_span() -> None:
    trimesh = pytest.importorskip("trimesh")
    roof = TunnelRoof("b_test", 0, (-0.5, 0.5, -0.7, 0.7), (0.35, 0.80))
    geometry = roof_geometry(_bridge_part(trimesh), roof, floor_z_m=0.0)

    assert np.allclose(geometry.span_xy_m, (-0.175, 0.175, -0.40, 0.40))
    assert np.isclose(geometry.underside(np.array(0.0), np.array(0.0)), 0.22)
    assert np.isclose(geometry.top(np.array(0.0), np.array(0.0)), 0.40)
    assert geometry.travel_axis == "x"
    with pytest.raises(ValueError, match="differs from"):
        roof_geometry(
            _bridge_part(trimesh),
            TunnelRoof("b", 0, roof.search_box_xy_m, (0.35, 1.1)),
            floor_z_m=0.0,
        )


def test_lower_span_replaces_only_roof_samples_and_rejects_a_blocked_channel() -> None:
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("rtree")
    geometry = _geometry("b_red")
    x0, x1, y0, y1 = geometry.span_xy_m
    floor = trimesh.creation.box(extents=(3.0, 3.0, 0.1))
    floor.apply_translation((0.0, 0.0, -0.05))
    slab = trimesh.creation.box(extents=(x1 - x0, y1 - y0, TOP_M - UNDERSIDE_M))
    slab.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (TOP_M + UNDERSIDE_M) / 2))
    axis = np.round(np.arange(-1.5, 1.5 + 1e-9, 0.01), 6)
    gx, gy = np.meshgrid(axis, axis)
    covered = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
    field = HeightFieldData(axis, axis, np.where(covered, TOP_M, 0.0))
    updated = field.height_m.copy()

    record = tunnel_contact._lower_span(
        geometry, [floor, slab], field, updated, np.zeros(3), floor_minimum_m=0.0
    )

    assert np.all(updated[covered] == 0.0)
    assert np.array_equal(updated[~covered], field.height_m[~covered])
    assert record["hfield_nodes_lowered"] == int(covered.sum())
    assert np.allclose(record["clearance_m"], UNDERSIDE_M, atol=2e-3)

    block = trimesh.creation.box(extents=(0.05, 0.05, 0.2))
    block.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, 0.1))
    with pytest.raises(ValueError, match="inside the channel"):
        tunnel_contact._lower_span(
            geometry,
            [floor, slab, block],
            field,
            field.height_m.copy(),
            np.zeros(3),
            floor_minimum_m=0.0,
        )


# --- pack contract ----------------------------------------------------------


def test_tunnel_pack_verifies_and_adds_every_roof_to_every_profile(tunnel_pack: Path) -> None:
    asset = FieldAsset.open(tunnel_pack, verify=True)
    layer = asset.collision["tunnel_contact_layer"]

    assert [roof["tunnel_id"] for roof in layer["roofs"]] == list(ROOFS)
    assert layer["season"] == "rmuc2026"
    for profile in asset.available_runtime_profiles:
        root = ET.parse(asset.entrypoint_for(profile)).getroot()
        names = {geom.get("name") for geom in root.iter("geom")}
        assert {f"rmuc_tunnel_roof_{tunnel_id}" for tunnel_id in ROOFS} <= names
    grid = load_heightfield(asset)
    assert float(grid.height_m.max()) == 0.0


def test_tunnel_layer_rejects_tampering(tunnel_pack: Path) -> None:
    profile_xml = FieldAsset.open(tunnel_pack).entrypoint_for("full")
    original = profile_xml.read_text(encoding="utf-8")
    profile_xml.write_text(
        original.replace(
            'name="rmuc_tunnel_roof_a_red" type="mesh" mesh="rmuc_tunnel_roof_a_red" contype="2"',
            'name="rmuc_tunnel_roof_a_red" type="mesh" mesh="rmuc_tunnel_roof_a_red" contype="0"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        FieldAsset.open(tunnel_pack, verify=True)
    profile_xml.write_text(original, encoding="utf-8")

    def drop_roof(manifest):
        manifest["collision"]["tunnel_contact_layer"]["roofs"].pop()

    _rewrite(tunnel_pack, drop_roof)
    with pytest.raises(ManifestError, match="every pinned roof"):
        FieldAsset.open(tunnel_pack)

    def wrong_season(manifest):
        layer = manifest["collision"]["tunnel_contact_layer"]
        layer["roofs"].append(layer["roofs"][0])
        layer["season"] = "rmuc2027"

    _rewrite(tunnel_pack, wrong_season)
    with pytest.raises(ManifestError, match="source or status contract"):
        FieldAsset.open(tunnel_pack)


def test_walls_must_precede_tunnels_and_spawns_avoid_roofs(
    tunnel_pack: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="before the tunnel layer"):
        source_contact_pack.export_source_wall_pack(tmp_path, tunnel_pack, tmp_path / "walls")
    asset = FieldAsset.open(tunnel_pack)
    spawns = find_spawn_candidates(
        asset, count=64, footprint_radius_m=0.1, boundary_margin_m=0.05, minimum_separation_m=0.1
    )
    assert spawns
    for spawn in spawns:
        for roof in asset.collision["tunnel_contact_layer"]["roofs"]:
            (x0, y0), (x1, y1) = roof["span_world_xy_m"]
            assert not (x0 - 0.1 <= spawn.x_m <= x1 + 0.1 and y0 - 0.1 <= spawn.y_m <= y1 + 0.1)


# --- probes -----------------------------------------------------------------


def test_probes_pass_under_are_blocked_above_and_rest_on_each_roof(tunnel_pack: Path) -> None:
    report = run_tunnel_checks(tunnel_pack, profiles=("collision_only",))

    assert report["status"] == "PASS", [t for t in report["trials"] if t["status"] != "PASS"]
    assert report["summary"] == {"roofs": 4, "trials": 12, "passed": 12, "failed": 0}


def test_check_runs_the_tunnel_section_only_for_tunnel_packs(
    tunnel_pack: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        regression,
        "run_tunnel_checks",
        lambda asset, profiles: calls.append(profiles) or {"status": "PASS"},
    )
    for name in (
        "check_public_interfaces",
        "check_surface_routes",
        "run_stair_checks",
        "check_example_episodes",
    ):
        monkeypatch.setattr(regression, name, lambda *a, **k: {"status": "PASS"})
    report = regression.run_regression(
        tunnel_pack, "source.json", output=tunnel_pack.parent / "r.json"
    )

    assert report["sections"]["tunnels"]["status"] == "PASS"
    assert calls


def test_slope_patches_under_a_roof_are_not_ordinary_slopes() -> None:
    from rmuc_mujoco.slope_catalog import catalog_slope_patches

    axis = np.round(np.arange(-2.0, 2.0 + 1e-9, 0.02), 6)
    gx, _ = np.meshgrid(axis, axis)
    ramp = HeightFieldData(axis, axis, np.clip(gx, -1.0, 1.0) * np.tan(np.radians(12.0)))
    open_sky = catalog_slope_patches(ramp, max_per_band=8, minimum_separation_m=0.5)
    assert open_sky
    roofs = [
        [
            [p.center_xyz_m[0] - 0.2, p.center_xyz_m[1] - 0.2],
            [p.center_xyz_m[0] + 0.2, p.center_xyz_m[1] + 0.2],
        ]
        for p in open_sky
    ]
    covered = catalog_slope_patches(
        ramp, max_per_band=8, minimum_separation_m=0.5, excluded_xy_m=roofs
    )
    centres = {tuple(np.round(p.center_xyz_m[:2], 3)) for p in open_sky}
    assert not centres & {tuple(np.round(p.center_xyz_m[:2], 3)) for p in covered}


def test_each_roof_publishes_a_through_and_an_over_route(wide_tunnel_pack: Path) -> None:
    catalog = build_tunnel_catalog(wide_tunnel_pack)

    assert catalog["status"] == "PASS_STATIC" and not catalog["rejected"]
    routes = {patch["patch_id"]: patch["route"] for patch in catalog["patches"]}
    assert set(routes) == {
        f"tunnel_{tunnel_id}_{kind}" for tunnel_id in ROOFS for kind in ("through", "over")
    }
    # b_red runs along x at (-0.8, 0.8); each end clears the 0.35 m roof by the margin.
    through = routes["tunnel_b_red_through"]
    reach = 0.175 + ROUTE_MARGIN_M
    assert through["low_xyz_m"][:2] == pytest.approx([-0.8 - reach, 0.8])
    assert through["high_xyz_m"][:2] == pytest.approx([-0.8 + reach, 0.8])
    assert through["roof_station_m"] == pytest.approx(reach)
    assert (through["roof_underside_m"], through["roof_top_m"]) == (UNDERSIDE_M, TOP_M)
    assert through["width_m"] == pytest.approx(0.8)
    # The over route shares the line and ends and passes above the roof top.
    over = routes["tunnel_b_red_over"]
    assert over["low_xyz_m"] == through["low_xyz_m"] and over["tunnel_pass"] == "over"
    assert max(point[2] for point in over["waypoints_xyz_m"]) == pytest.approx(TOP_M)


def test_a_route_that_leaves_the_field_is_rejected(tunnel_pack: Path) -> None:
    """On the 3 m field every route end would lie off the heightfield."""

    catalog = build_tunnel_catalog(tunnel_pack)

    assert catalog["status"] == "NO_TUNNEL_ROUTES" and not catalog["patches"]
    assert {row["reason"] for row in catalog["rejected"]} == {"route_outside_heightfield"}


def test_the_rover_drives_through_and_cannot_pass_for_over(wide_tunnel_pack: Path) -> None:
    """Both routes share their ends; the roof crossing tells them apart."""

    catalog = build_tunnel_catalog(wide_tunnel_pack)
    session = SlopeSession(wide_tunnel_pack, mode="policy", route_catalog={"catalog": catalog})

    def drive(patch: str) -> dict:
        case = {
            "case_id": patch,
            "case_index": 0,
            "patch": patch,
            "direction": "uphill",
            "speed_mps": 0.3,
            "repeat": 0,
            "seed": 0,
            "timeout_s": 20.0,
        }
        return run_slope_episode(session, case)

    through = drive("tunnel_b_red_through")
    assert through["outcome"] == "PASS", through["failure_reason"]
    over = drive("tunnel_b_red_over")
    assert (over["outcome"], over["failure_reason"]) == ("FAIL", "passed_under_roof")


def test_check_hands_the_tunnel_routes_to_the_example_robots(
    tunnel_pack: Path, monkeypatch
) -> None:
    examples = []
    monkeypatch.setattr(
        regression,
        "run_tunnel_checks",
        lambda asset, profiles: {"status": "PASS", "catalog": {"patches": ["route"]}},
    )
    monkeypatch.setattr(
        regression,
        "check_example_episodes",
        lambda *args: examples.append(args) or {"status": "PASS"},
    )
    for name in ("check_public_interfaces", "check_surface_routes", "run_stair_checks"):
        monkeypatch.setattr(regression, name, lambda *a, **k: {"status": "PASS"})

    regression.run_regression(tunnel_pack, "source.json", output=tunnel_pack.parent / "r.json")

    assert examples and examples[0][5] == {"patches": ["route"]}
