"""Walls derived from merged source parts: hull, record gates, roof transfer, audit."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rmuc_mujoco.query import HeightFieldData
from rmuc_mujoco.seasons import SourceWall
from rmuc_mujoco.wall_active_contact import (
    _replace_derived_wall_roof,
    derive_wall,
    verified_derived_wall,
)
from rmuc_mujoco.wall_check import audit_wall_ownership

trimesh = pytest.importorskip("trimesh")

# A 0.2 m wall standing on the floor at x = 0.1 whose inner face meets a road
# 0.2 m up at x = 0.3, drawn as RMUC 2027 draws its road-area walls: wall and
# road are one part, so the wall has no bottom and no inner face below the road.
OUTER, INNER, ROAD, TOP, Y0, Y1 = 0.10, 0.30, 0.20, 1.00, 0.10, 1.10
BOX = (OUTER, INNER, Y0, Y1)
# Trapezoidal section (1.0 m tall outside, 0.8 m above the road) times 1 m.
VOLUME_M3 = 0.18


def _merged_part(*, drop_end: bool = False, dent: bool = False):
    vertices = [
        (OUTER, Y0, 0.0),
        (OUTER, Y1, 0.0),
        (OUTER, Y0, TOP),
        (OUTER, Y1, TOP),
        (INNER, Y0, ROAD),
        (INNER, Y1, ROAD),
        (INNER, Y0, TOP),
        (INNER, Y1, TOP),
        (0.80, Y0, ROAD),  # far edge of the road, outside the wall box
        (0.80, Y1, ROAD),
        (0.20, 0.60, 0.90),  # used only by the dented top
    ]
    top = [[2, 6, 10], [6, 7, 10], [7, 3, 10], [3, 2, 10]] if dent else [[2, 6, 7], [2, 7, 3]]
    faces = [
        [0, 2, 3],
        [0, 3, 1],  # outer face
        [4, 5, 7],
        [4, 7, 6],  # inner face above the road
        *top,
        *([] if drop_end else [[0, 4, 6], [0, 6, 2]]),  # end at y = Y0
        [1, 3, 7],
        [1, 7, 5],  # end at y = Y1
        [4, 8, 9],
        [4, 9, 5],  # road
    ]
    used = vertices if dent else vertices[:10]
    return trimesh.Trimesh(np.array(used), np.array(faces), process=False)


def _record(**changes) -> SourceWall:
    return replace(SourceWall(7, "high", 0, 8, 12, VOLUME_M3, BOX), **changes)


def _field() -> tuple[HeightFieldData, np.ndarray, np.ndarray]:
    """Floor outside the wall, its roof over it and the road inside it, at 1 cm."""

    x = np.round(np.arange(0.0, 0.601, 0.01), 6)
    y = np.round(np.arange(0.0, 1.201, 0.01), 6)
    grid_x, grid_y = np.meshgrid(x, y)
    on_wall = (grid_x >= OUTER) & (grid_x <= INNER) & (grid_y >= Y0) & (grid_y <= Y1)
    height = np.where(grid_x > INNER, ROAD, 0.0)
    height[on_wall] = TOP
    ix = np.flatnonzero((x >= OUTER) & (x <= INNER))
    iy = np.flatnonzero((y >= Y0) & (y <= Y1))
    return HeightFieldData(x, y, height), ix, iy


def test_a_merged_wall_is_hulled_from_its_box_and_measured_against_the_part() -> None:
    hull, measured = derive_wall(_merged_part(), BOX)

    assert measured["box_vertices"] == measured["hull_vertices"] == 8
    assert measured["hull_triangles"] == 12 and measured["watertight"]
    assert measured["hull_volume_m3"] == pytest.approx(VOLUME_M3)
    # Every part triangle in the box is on the hull, and the only hull face the
    # part does not draw is the underside, which the terrain hides.
    assert measured["part_triangles_in_box"] == 10
    assert measured["part_triangles_off_hull"] == 0
    assert measured["uncovered_side_area_m2"] == pytest.approx(0.0, abs=1e-12)
    assert hull.volume == pytest.approx(VOLUME_M3)


def test_a_missing_side_or_a_dent_is_measured_not_hidden() -> None:
    _, open_end = derive_wall(_merged_part(drop_end=True), BOX)
    assert open_end["uncovered_side_area_m2"] == pytest.approx(VOLUME_M3)  # the end's area
    _, dented = derive_wall(_merged_part(dent=True), BOX)
    assert (dented["box_vertices"], dented["hull_vertices"]) == (9, 8)
    assert dented["part_triangles_off_hull"] == 4


def test_a_derived_wall_must_match_its_record() -> None:
    hull, _ = verified_derived_wall(_merged_part(), _record())
    assert hull.is_watertight
    with pytest.raises(ValueError, match="hull changed"):
        verified_derived_wall(_merged_part(), _record(hull_volume_m3=VOLUME_M3 + 0.01))
    with pytest.raises(ValueError, match="hull changed"):
        verified_derived_wall(_merged_part(dent=True), _record())
    with pytest.raises(ValueError, match="side no longer matches"):
        verified_derived_wall(_merged_part(drop_end=True), _record())


def test_a_derived_roof_is_handed_over_and_filled_to_the_underside() -> None:
    hull, _ = derive_wall(_merged_part(), BOX)
    field, ix, iy = _field()
    # Two samples at one end of the wall, lost by the builder's spike filter.
    field.height_m[iy[0], ix[5]] = field.height_m[iy[0], ix[10]] = 0.0
    wall = _record(roof_nodes=len(ix) * len(iy), spike_filtered_end_nodes=2)
    output = field.height_m.copy()

    report = _replace_derived_wall_roof(hull, field, output, wall=wall)

    assert report["roof_nodes_transferred"] == len(ix) * len(iy)
    assert report["spike_filtered_end_nodes"] == 2
    assert report["maximum_gap_beneath_source_wall_m"] == 0.0
    # The underside rises from the floor outside to the road inside, so the
    # heightfield under the wall follows it instead of dropping to the floor.
    underside = np.broadcast_to(field.x_m[ix] - OUTER, (len(iy), len(ix)))
    np.testing.assert_allclose(output[np.ix_(iy, ix)], underside, atol=1e-9)
    mask = np.ones(output.shape, dtype=bool)
    mask[np.ix_(iy, ix)] = False
    np.testing.assert_array_equal(output[mask], field.height_m[mask])


def test_a_changed_derived_roof_is_refused_without_mutation() -> None:
    hull, _ = derive_wall(_merged_part(), BOX)
    field, ix, iy = _field()
    wall = _record(roof_nodes=len(ix) * len(iy))
    field.height_m[iy[50], ix[10]] = 0.0  # low in mid-wall: not a spike-filtered end
    output = field.height_m.copy()
    with pytest.raises(ValueError, match="end samples changed"):
        _replace_derived_wall_roof(hull, field, output, wall=wall)
    np.testing.assert_array_equal(output, field.height_m)

    field.height_m[iy[50], ix[10]] = TOP - 0.01
    output = field.height_m.copy()
    with pytest.raises(ValueError, match="roof is not the derived source wall"):
        _replace_derived_wall_roof(hull, field, output, wall=wall)
    np.testing.assert_array_equal(output, field.height_m)


def test_the_audit_expects_a_derived_wall_filled_to_its_underside() -> None:
    hull, _ = derive_wall(_merged_part(), BOX)
    field, ix, iy = _field()
    output = field.height_m.copy()
    _replace_derived_wall_roof(hull, field, output, wall=_record(roof_nodes=len(ix) * len(iy)))
    region = {"source_part_index": 7, "changed_grid_rectangle_yx": [iy[0], iy[-1], ix[0], ix[-1]]}

    filled = HeightFieldData(field.x_m, field.y_m, output)
    report = audit_wall_ownership(filled, hull, hull.copy(), region, derived=True)
    assert report["status"] == "PASS" and report["transferred_samples"] == len(ix) * len(iy)

    # Lowered to the flank as an exact wall is, it would leave a void beneath.
    lowered = output.copy()
    lowered[np.ix_(iy, ix)] = 0.0
    flat = HeightFieldData(field.x_m, field.y_m, lowered)
    assert audit_wall_ownership(flat, hull, hull.copy(), region, derived=True)["status"] == "FAIL"


def test_the_hull_keeps_only_whole_facets_of_noisy_points() -> None:
    """A point 9 um off its facet, inside the tolerance, as float32 GLB points are.

    The plane through it and two corners leaves the far corners out; that
    partial facet must not be triangulated next to the whole one.
    """

    top = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0, 9e-6)]
    bottom = [(0, 0, -1), (1, 0, -1), (1, 1, -1), (0, 1, -1)]
    points = np.array(top + bottom, dtype=float)
    # Any triangles will do as long as they use every point: merging drops the rest.
    part = trimesh.Trimesh(
        points, np.array([[0, 1, 4], [1, 2, 3], [5, 6, 7], [7, 8, 5]]), process=False
    )

    hull, measured = derive_wall(part, (-0.1, 1.1, -0.1, 1.1))

    # Two pentagons (top and front, which hold the extra point) and four quads.
    assert measured["hull_vertices"] == 9
    assert measured["hull_triangles"] == 3 + 3 + 4 * 2
    assert measured["watertight"] and hull.volume == pytest.approx(1.0, abs=1e-5)
