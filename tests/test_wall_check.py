import numpy as np
import pytest
import trimesh

from rmuc_mujoco.wall_check import audit_wall_ownership, run_corner_route
from rmuc_mujoco.perimeter_check import run_wheel_route
from rmuc_mujoco.query import HeightFieldData
from test_perimeter_check import _flat_probe


def test_ownership_checks_actual_samples_and_actual_mesh():
    x = np.linspace(-0.3, 0.3, 61)
    y = np.linspace(-0.3, 0.3, 61)
    source = trimesh.creation.box(extents=(0.2, 0.2, 1))
    source.apply_translation([0, 0, 0.5])
    heights = np.zeros((61, 61))
    grid = HeightFieldData(x, y, heights)
    region = {"source_part_index": 402, "changed_grid_rectangle_yx": [21, 39, 21, 39]}
    assert audit_wall_ownership(grid, source, source.copy(), region)["status"] == "PASS"
    rounded = source.copy()
    rounded.apply_translation([0, 5e-9, 0])
    assert audit_wall_ownership(grid, source, rounded, region)["mesh_matches_source"] is True
    heights[30, 30] = 1  # A stale hfield roof under the new mesh is double support.
    assert audit_wall_ownership(grid, source, source.copy(), region)["status"] == "FAIL"
    heights[30, 30] = 0
    changed = source.copy()
    changed.apply_translation([0.01, 0, 0])
    assert audit_wall_ownership(grid, source, changed, region)["mesh_matches_source"] is False
    changed = source.copy()
    changed.faces = changed.faces[:-1]  # Identical vertices cannot hide a missing face.
    assert audit_wall_ownership(grid, source, changed, region)["mesh_matches_source"] is False


def test_two_wall_cycles_do_not_reset_state_between_contacts(tmp_path, monkeypatch):
    model, route = _flat_probe(tmp_path, monkeypatch)
    report, trace = run_wheel_route(model, route, 0.5, 0, cycles=2, slide=False)
    assert report["status"] == "PASS"
    assert [p["phase"] for p in report["phases"]] == [
        "settle",
        "approach",
        "retreat",
        "approach",
        "retreat",
    ]
    qpos = np.asarray(trace["qpos"])
    assert np.max(np.abs(np.diff(qpos[:, :3], axis=0))) < 0.01


@pytest.mark.parametrize("wall", [True, False])
def test_corner_requires_a_loaded_blocking_contact_and_return(tmp_path, monkeypatch, wall):
    model, _ = _flat_probe(tmp_path, monkeypatch, wall=wall)
    corner = {
        "id": "synthetic_corner",
        "start_xy_m": [0.3, 0.1],
        "target_xy_m": [-0.15, -0.1],
        "initial_height_m": 0,
    }
    report, _ = run_corner_route(model, corner, 0.5)
    assert report["status"] == ("PASS" if wall else "FAIL")
    if wall:
        assert len(report["blocking_contacts"]) == 2
        assert all(hit["opposing_force_n"] > 1 for hit in report["blocking_contacts"])
        assert report["phases"][-1]["reached"]
    else:
        assert report["blocking_contacts"] == []  # A ground contact is not a wall hit.


def test_an_obstructed_fence_corner_skips_the_local_probe(monkeypatch):
    """A corner filled by a structure has no fence corner for a local probe."""

    from rmuc_mujoco import wall_check

    corners = [
        {"id": "left_bottom", "xy_m": [0.0, 0.0], "panel_join_status": "PASS_STATIC"},
        {"id": "right_top", "xy_m": [2.0, 2.0], "panel_join_status": "PASS_STATIC"},
    ]

    def heights(grid, xy):
        # Open floor near left_bottom; a 1.7 m block over everything near right_top.
        return np.where(np.hypot(xy[..., 0] - 2.0, xy[..., 1] - 2.0) < 1.9, 1.7, 0.0)

    monkeypatch.setattr(wall_check, "fence_corners", lambda fence: [dict(c) for c in corners])
    monkeypatch.setattr(wall_check, "_height", heights)
    catalog = wall_check._corner_catalog({"perimeter_fence": {}}, None, [], np.zeros(3))

    by_id = {corner["id"]: corner for corner in catalog}
    assert "start_xy_m" in by_id["left_bottom"]
    assert by_id["right_top"]["corner_access"] == "SOURCE_OBSTRUCTED"
    assert by_id["right_top"]["local_probe"] == "NOT_RUN"
    assert "start_xy_m" not in by_id["right_top"]


def test_only_a_derived_wall_standing_on_the_road_skips_its_end_cap():
    from rmuc_mujoco.wall_check import free_end_stands_on_road
    from test_derived_walls import BOX, Y1, _merged_part
    from rmuc_mujoco.wall_active_contact import derive_wall

    exact = trimesh.creation.box(extents=(0.2, 1.0, 1.0))
    exact.apply_translation([0.2, 0.6, 0.5])
    assert not free_end_stands_on_road(exact, 1.1, 1)
    derived, _ = derive_wall(_merged_part(), BOX)
    assert free_end_stands_on_road(derived, Y1, 1)
