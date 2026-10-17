"""The RMUC 2027 high point (制高点): measured steps, contract check and routes.

Rulebook 4.2.4 puts a high point in each base area and joins it to the trapezoid
highland with three steps that a robot can climb.  Figure 4-17 dimensions the two
flights of every high point:

* section A-A, the highland flight: 200 mm risers with 300 mm treads, reference
  total (600);
* section B-B, the floor flight: four 200 mm risers down to the base-area floor,
  reference total (800).

Both flights reach their reference totals exactly on the drawn surface, and the
heightfield reproduces that surface's treads to a millimetre.  The treads
themselves differ from the figure's reference dimension on one flight only: the
drawn B-B treads are 305.1 mm, not 300 mm, which the 800 mm total absorbs.

The platform and its steps are drawn in the official source and the collision
heightfield already represents them, so nothing here changes field geometry.  It
measures the built field, binds the measurement to the source surface and
publishes route records the existing controllers and viewer already accept.

Every level is compared after removing a locally fitted platform plane.  The
recovered battlefield carries a shallow camber (about 2.6 % in y, with opposite
signs at the two ends), so raw absolute heights drift by tens of millimetres
across one high point and a raw riser read along the tilt loses about 8 mm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import FieldAsset
from .source_build import verified_source_glb
from .query import HeightFieldData, load_heightfield, sample_mujoco_heightfield
from .seasons import HIGH_POINTS, HighPoint

SAMPLE_STEP_M = 0.005
PATCH_STEP_M = 0.02
# Height change that ends a level run, and the shortest run worth reporting.
LEVEL_TOLERANCE_M = 0.006
MIN_RUN_M = 0.05
# The platform is the widest band of the search box; the band must also cover
# the camber across the platform itself.
PLATFORM_BAND_M = 0.10
PLATFORM_HALF_BAND_M = 0.08
PLANE_RESIDUAL_M = 0.02
PLANE_MAX_TILT = 0.10
MAX_REACH_M = 3.0
# Route records: approach on the low level, exit on the platform.  A route end
# is where a robot spawns, so it sits far enough from the nearest riser for a
# robot about a metre long to start clear of the flight; the step rover's
# 500 mm wheels reach 0.53 m ahead of its centre.  Both levels have the room.
ROUTE_MARGIN_M = 0.60
# The corridor the runtime session holds a robot inside.  It matches the stair
# routes and is narrower than either flight, so it never constrains a climb.
ROUTE_WIDTH_M = 0.60
# The band the source comparison samples across, kept well inside the treads so
# no sample lands on a level boundary.
SOURCE_BAND_HALF_WIDTH_M = 0.15
# Samples closer than this to a level boundary sit on a riser's heightfield ramp.
SOURCE_EDGE_MARGIN_M = 0.05
# Risers are the load-bearing dimension and the field reproduces them to 1.2 mm.
STEP_RISE_TOLERANCE_M = 0.005
# The figure marks its 300 mm treads as reference.  The drawing's own surface
# gives exactly 300 mm on section A-A but 305.1 mm on section B-B, and reading
# a riser ramp out of a 10 mm heightfield adds a few millimetres on top, so the
# tread gate is looser than the riser gate while both totals stay exact.
STEP_TREAD_TOLERANCE_M = 0.015
PLATFORM_RISE_TOLERANCE_M = 0.005
HIGHLAND_RISE_TOLERANCE_M = 0.005
# Tread interiors of the built field reproduce the source surface: the 2027
# V1.0.0 pack agrees to 1.0 mm at worst with a zero mean.  3 mm leaves room for
# ordinary grid rounding while staying far below the 200 mm riser it certifies.
SOURCE_AGREEMENT_M = 0.003
SOURCE_PART_MARGIN_M = 0.30


def _cad_offset(asset: FieldAsset) -> np.ndarray:
    """The translation the builder applied to the drawing to reach the pack."""

    spawn = asset.recommended_spawn
    return np.array(
        [
            float(spawn["x_before_translation_m"]),
            float(spawn["y_before_translation_m"]),
            float(spawn["terrain_height_m"]),
        ]
    )


def _patch(grid: HeightFieldData, offset: np.ndarray, box, step: float):
    x = np.arange(box[0], box[1] + 1e-9, step)
    y = np.arange(box[2], box[3] + 1e-9, step)
    mesh_x, mesh_y = np.meshgrid(x - offset[0], y - offset[1], indexing="ij")
    return mesh_x, mesh_y, sample_mujoco_heightfield(grid, mesh_x, mesh_y)


def _platform_plane(mesh_x, mesh_y, mesh_z):
    """Fit the platform top: the widest height band of the search box.

    The platform is the largest single flat area a high point's box contains, so
    the mode of a coarse height histogram selects it without relying on the
    record's nominal levels.  Residual clipping then removes the livery grooves
    and the few neighbouring faces that share the band.
    """

    low, high = float(mesh_z.min()), float(mesh_z.max())
    # The last bin is half-open, so the top edge has to clear the highest sample:
    # on a flat platform the platform *is* the highest surface in the box, and a
    # rounding-tight top edge would drop every one of its samples and hand the
    # mode to a level below it.  The extra band guarantees the clear margin.
    edges = np.arange(low, high + 2 * PLATFORM_BAND_M, PLATFORM_BAND_M)
    counts, _ = np.histogram(mesh_z, bins=edges)
    index = int(np.argmax(counts))
    centre = 0.5 * (edges[index] + edges[index + 1])
    inside = np.abs(mesh_z - centre) <= PLATFORM_HALF_BAND_M
    if inside.sum() < 9:
        raise ValueError("no platform-height band found in the high point box")
    design_x, design_y, design_z = mesh_x[inside], mesh_y[inside], mesh_z[inside]
    for _ in range(4):
        matrix = np.column_stack((design_x, design_y, np.ones_like(design_x)))
        coefficients, *_ = np.linalg.lstsq(matrix, design_z, rcond=None)
        residual = design_z - matrix @ coefficients
        keep = np.abs(residual - np.median(residual)) <= PLANE_RESIDUAL_M
        if keep.all():
            break
        design_x, design_y, design_z = design_x[keep], design_y[keep], design_z[keep]
    if design_x.size < 9:
        raise ValueError("platform plane fit lost its inliers")
    along_x, along_y = float(coefficients[0]), float(coefficients[1])
    if math.hypot(along_x, along_y) > PLANE_MAX_TILT:
        raise ValueError("platform-height band is not a single surface")
    level = design_z - (along_x * design_x + along_y * design_y)
    return {
        "coefficients_xyz": [along_x, along_y, float(coefficients[2])],
        "tilt_percent": 100 * math.hypot(along_x, along_y),
        "inlier_count": int(design_x.size),
        "center_xy_m": [
            0.5 * (float(design_x.min()) + float(design_x.max())),
            0.5 * (float(design_y.min()) + float(design_y.max())),
        ],
        "bounds_xy_m": [
            [float(design_x.min()), float(design_y.min())],
            [float(design_x.max()), float(design_y.max())],
        ],
        "design_level_m": float(np.median(level)),
    }


def _runs(along: np.ndarray, design: np.ndarray) -> list[dict[str, float]]:
    """Collapse one design-level profile into level runs."""

    runs, current, start = [], float(design[0]), float(along[0])
    for value, position in zip(design[1:], along[1:]):
        if abs(float(value) - current) > LEVEL_TOLERANCE_M:
            runs.append({"start_m": start, "stop_m": float(position), "level_m": current})
            current, start = float(value), float(position)
    runs.append({"start_m": start, "stop_m": float(along[-1]), "level_m": current})
    for run in runs:
        run["length_m"] = run["stop_m"] - run["start_m"]
    return [run for run in runs if run["length_m"] > MIN_RUN_M]


def _crossing(along: np.ndarray, design: np.ndarray, level: float) -> float:
    """Where a monotonically descending riser profile passes its half-rise level.

    The heightfield meets a vertical riser with a ramp about one cell wide, so
    the half-rise crossing is within a few millimetres of the drawn face and is
    not quantised to the sampling step the way a run boundary is.
    """

    above = design > level
    turns = np.flatnonzero(above[:-1] != above[1:])
    if len(turns) != 1:
        raise ValueError("a riser profile must cross its half-rise level exactly once")
    index = int(turns[0])
    start, stop = float(along[index]), float(along[index + 1])
    low, high = float(design[index]), float(design[index + 1])
    if high == low:
        return 0.5 * (start + stop)
    return start + (level - low) * (stop - start) / (high - low)


def _riser_positions(
    along: np.ndarray, design: np.ndarray, runs: list[dict[str, float]]
) -> list[float]:
    """Each riser sits between two runs, at that pair's half-rise crossing."""

    positions = []
    for index in range(len(runs) - 1):
        middle = 0.5 * (runs[index]["level_m"] + runs[index + 1]["level_m"])
        window = (along >= runs[index]["stop_m"] - 2 * SAMPLE_STEP_M) & (
            along <= runs[index + 1]["start_m"] + 2 * SAMPLE_STEP_M
        )
        positions.append(_crossing(along[window], design[window], middle))
    return positions


def measure_flight(
    grid: HeightFieldData,
    high_point: HighPoint,
    plane: dict[str, Any],
    *,
    flight: str,
) -> dict[str, Any]:
    """Walk outward from the platform centre down one flight and read its steps.

    The platform centre is taken from the fitted plane's inliers rather than
    from the record's search box, so an off-centre box cannot bias the run the
    rays cross.
    """

    direction = np.array(
        high_point.highland_flight_xy if flight == "highland" else high_point.floor_flight_xy,
        dtype=float,
    )
    if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-9):
        raise ValueError(
            f"{high_point.high_point_id} {flight} flight direction is not a unit vector"
        )
    center = np.asarray(plane["center_xy_m"], dtype=float)
    along = np.arange(0.0, MAX_REACH_M + 1e-9, SAMPLE_STEP_M)
    world = center + along[:, None] * direction
    if np.any(world < np.asarray(grid.bounds_xy_m[0])) or np.any(
        world > np.asarray(grid.bounds_xy_m[1])
    ):
        raise ValueError(f"{high_point.high_point_id} {flight} flight leaves the heightfield")
    height = sample_mujoco_heightfield(grid, world[:, 0], world[:, 1])
    coefficients = plane["coefficients_xyz"]
    design = height - (coefficients[0] * world[:, 0] + coefficients[1] * world[:, 1])
    runs = _runs(along, design)
    if len(runs) < 2:
        raise ValueError(f"{high_point.high_point_id} {flight} flight has no steps")
    risers = _riser_positions(along, design, runs)
    levels = [run["level_m"] for run in runs]
    return {
        "flight": flight,
        "direction_xy": [float(v) for v in direction],
        "start_xy_m": [float(v) for v in center],
        # Levels are tilt-removed coordinates, so a physical height is compared
        # against its level only after removing the same plane.
        "plane_coefficients_xy": [float(coefficients[0]), float(coefficients[1])],
        "runs": runs,
        "riser_positions_m": risers,
        "risers_m": [levels[i] - levels[i + 1] for i in range(len(levels) - 1)],
        "treads_m": [risers[i + 1] - risers[i] for i in range(len(risers) - 1)],
        "platform_level_m": levels[0],
        "landing_level_m": levels[-1],
        "landing_length_m": runs[-1]["length_m"],
    }


def _check(name: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"check": name, "status": "PASS" if passed else "FAIL", **detail}


def _flight_checks(high_point: HighPoint, flight: dict[str, Any], expected_steps: int) -> list:
    label = flight["flight"]
    risers, treads = flight["risers_m"], flight["treads_m"]
    rise_error = max((abs(value - high_point.step_rise_m) for value in risers), default=math.inf)
    tread_error = max((abs(value - high_point.step_tread_m) for value in treads), default=math.inf)
    return [
        _check(
            f"{label}_flight_steps",
            len(risers) == expected_steps
            and all(
                abs(value - high_point.step_rise_m) <= STEP_RISE_TOLERANCE_M for value in risers
            ),
            {
                "measured_steps": len(risers),
                "expected_steps": expected_steps,
                "measured_rises_m": [round(value, 4) for value in risers],
                "expected_rise_m": high_point.step_rise_m,
                "rise_tolerance_m": STEP_RISE_TOLERANCE_M,
                "max_rise_error_m": round(rise_error, 5),
            },
        ),
        _check(
            f"{label}_flight_treads",
            bool(treads)
            and all(
                abs(value - high_point.step_tread_m) <= STEP_TREAD_TOLERANCE_M for value in treads
            ),
            {
                "measured_treads_m": [round(value, 4) for value in treads],
                "expected_tread_m": high_point.step_tread_m,
                "tread_tolerance_m": STEP_TREAD_TOLERANCE_M,
                "max_tread_error_m": round(tread_error, 5),
            },
        ),
    ]


def measure_high_point(
    asset: FieldAsset | str | Path, high_point: HighPoint, *, grid: HeightFieldData | None = None
) -> dict[str, Any]:
    """Measure one high point in a verified pack and check the drawing contract."""

    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    heightfield = load_heightfield(field) if grid is None else grid
    offset = _cad_offset(field)
    mesh_x, mesh_y, mesh_z = _patch(heightfield, offset, high_point.search_box_xy_m, PATCH_STEP_M)
    plane = _platform_plane(mesh_x, mesh_y, mesh_z)
    flights = {
        name: measure_flight(heightfield, high_point, plane, flight=name)
        for name in ("highland", "floor")
    }
    highland, floor = flights["highland"], flights["floor"]
    floor_level = floor["landing_level_m"]
    platform_top = plane["design_level_m"] - floor_level
    highland_level = highland["landing_level_m"] - floor_level
    checks = (
        _flight_checks(high_point, highland, high_point.highland_step_count)
        + _flight_checks(high_point, floor, high_point.floor_step_count)
        + [
            _check(
                "platform_height_above_base_floor",
                abs(platform_top - high_point.platform_top_m) <= PLATFORM_RISE_TOLERANCE_M,
                {
                    "measured_m": round(platform_top, 4),
                    "expected_m": high_point.platform_top_m,
                    "tolerance_m": PLATFORM_RISE_TOLERANCE_M,
                },
            ),
            _check(
                "highland_height_above_base_floor",
                abs(highland_level - high_point.highland_level_m) <= HIGHLAND_RISE_TOLERANCE_M,
                {
                    "measured_m": round(highland_level, 4),
                    "expected_m": high_point.highland_level_m,
                    "tolerance_m": HIGHLAND_RISE_TOLERANCE_M,
                },
            ),
        ]
    )
    return {
        "high_point_id": high_point.high_point_id,
        "search_box_xy_m": list(high_point.search_box_xy_m),
        "platform": plane,
        "flights": flights,
        "platform_top_m": platform_top,
        "highland_level_m": highland_level,
        "checks": checks,
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
    }


# --- source binding ---------------------------------------------------------


def world_z_shift(asset: FieldAsset, source_manifest: str | Path) -> float:
    """The CAD-to-world z offset of a pack's heightfield.

    A pack's NPZ x/y axes are the drawing's x/y, but its z has been moved
    twice: the builder drops the drawing so the main floor reads
    ``main_floor_height_before_shift_m``, and the pack's world frame then
    subtracts ``recommended_spawn.terrain_height_m``.  So a source ray hit at
    CAD height ``z`` is at world ``z - main_floor_height_before_shift_m -
    terrain_height_m``.
    """

    _, shift, _ = verified_source_glb(source_manifest, asset.field_source)
    return shift - float(asset.recommended_spawn["terrain_height_m"])


def source_top(meshes, points, z_shift_m: float, box) -> np.ndarray:
    """Highest source surface over CAD-frame points, in world z.

    The heightfield builder used the same selection rule (its structural audit
    records "highest downward ray hit at each xy sample"), so a difference at a
    sample is discretisation of one shared surface, not two different surfaces.
    """

    points = np.asarray(points, dtype=float)
    origins = np.column_stack((points, np.full(len(points), 5.0)))
    directions = np.tile((0.0, 0.0, -1.0), (len(points), 1))
    top = np.full(len(points), -np.inf)
    lower = np.array([box[0] - SOURCE_PART_MARGIN_M, box[2] - SOURCE_PART_MARGIN_M])
    upper = np.array([box[1] + SOURCE_PART_MARGIN_M, box[3] + SOURCE_PART_MARGIN_M])
    for mesh in meshes:
        if np.any(mesh.bounds[1, :2] < lower) or np.any(mesh.bounds[0, :2] > upper):
            continue
        locations, rays, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
        if len(rays):
            np.maximum.at(top, rays, locations[:, 2] + z_shift_m)
    return top


def interior_points(
    high_point: HighPoint,
    flight: dict[str, Any],
    *,
    half_width_m: float = SOURCE_BAND_HALF_WIDTH_M,
    margin_m: float = SOURCE_EDGE_MARGIN_M,
) -> np.ndarray:
    """World-frame samples well inside each measured level of one flight.

    A single-valued heightfield cannot hold a vertical riser: the cell that
    straddles one is filled with the higher surface, and the triangles either
    side of it meet it along a ramp.  Samples within about a grid cell of an
    edge are therefore expected to differ from the source, so only the tread
    interiors carry the claim that the two surfaces are the same surface.
    """

    direction = np.array(flight["direction_xy"], dtype=float)
    across = np.array([-direction[1], direction[0]])
    center = np.asarray(flight["start_xy_m"], dtype=float)
    lateral = np.arange(-half_width_m, half_width_m + 1e-9, 0.05)
    columns = []
    for run in flight["runs"]:
        low, high = run["start_m"] + margin_m, run["stop_m"] - margin_m
        if high <= low:
            continue
        interior = np.arange(low, high + 1e-9, SAMPLE_STEP_M)
        mesh_along, mesh_across = np.meshgrid(interior, lateral, indexing="ij")
        columns.append(center + mesh_along[..., None] * direction + mesh_across[..., None] * across)
    if not columns:
        raise ValueError(f"{high_point.high_point_id} {flight['flight']}: no tread interior")
    return np.concatenate(columns).reshape(-1, 2)


def compare_with_source(
    asset: FieldAsset,
    high_point: HighPoint,
    source_manifest: str | Path,
    *,
    grid: HeightFieldData | None = None,
    measured: dict[str, Any],
) -> dict[str, Any]:
    """Check the field's step surfaces against the official source surface.

    ``measured`` is the result of :func:`measure_high_point`, which supplies the
    levels to compare.  Only tread interiors are compared, per
    :func:`interior_points`; the report states how many samples were taken, how
    far the two surfaces diverge, and where the worst sample is.
    """

    import trimesh

    source = asset.season.default_source
    glb, _, _ = verified_source_glb(source_manifest, source)
    z_shift = world_z_shift(asset, source_manifest)
    meshes = list(trimesh.load(glb, force="scene", process=False).dump(concatenate=False))
    heightfield = load_heightfield(asset) if grid is None else grid
    offset = _cad_offset(asset)
    flights = measured["flights"]
    report: dict[str, Any] = {}
    worst = (-math.inf, None)
    total = 0
    for name, flight in flights.items():
        world = interior_points(high_point, flight)
        cad = world + offset[:2]
        runtime = sample_mujoco_heightfield(heightfield, world[:, 0], world[:, 1])
        source_z = source_top(meshes, cad, z_shift, high_point.search_box_xy_m)
        covered = np.isfinite(source_z)
        if covered.sum() < 9:
            raise ValueError(f"{high_point.high_point_id} {name}: no source surface on the treads")
        difference = np.abs(runtime[covered] - source_z[covered])
        index = int(np.argmax(difference))
        total += int(covered.sum())
        if difference[index] > worst[0]:
            worst = (float(difference[index]), [float(v) for v in cad[covered][index]])
        report[name] = {
            "samples": int(covered.sum()),
            "uncovered_samples": int((~covered).sum()),
            "max_difference_m": float(difference.max()),
            "mean_difference_m": float(difference.mean()),
            "tread_rises_m": [round(value, 4) for value in flight["risers_m"]],
        }
    return {
        "source_glb_sha256": source.source_glb_sha256,
        "source_step_sha256": source.step.sha256,
        "edge_margin_m": SOURCE_EDGE_MARGIN_M,
        "samples": total,
        "flights": report,
        "max_difference_m": worst[0],
        "worst_xy_m": worst[1],
        "status": "PASS" if worst[0] <= SOURCE_AGREEMENT_M else "FAIL",
    }


# --- routes -----------------------------------------------------------------


def _route(
    grid: HeightFieldData,
    high_point: HighPoint,
    flight: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """One climbable route: from the low level, over the flight, onto the platform.

    The route axis is the measured descent direction, so its coordinate grows as
    the flight descends: the platform sits at the smaller coordinate and the
    landing at the larger one.  The two ends are named by height, as the stair
    and slope routes name them, so ``low_xyz_m`` is the landing the climb starts
    from and ``high_xyz_m`` is the platform it reaches.  The runtime spawns a
    climb at the low end facing ``uphill_unit_xy``, so naming these the other way
    round would drive the robot across the platform top instead of the steps.
    """

    direction = np.array(flight["direction_xy"], dtype=float)
    climb = -direction
    center = np.asarray(flight["start_xy_m"], dtype=float)
    risers = flight["riser_positions_m"]
    low_s = risers[-1] + ROUTE_MARGIN_M
    high_s = risers[0] - ROUTE_MARGIN_M

    def world(along_m):
        return center + along_m * direction

    def xyz(along_m):
        xy = world(along_m)
        z = sample_mujoco_heightfield(grid, np.array([xy[0]]), np.array([xy[1]]))[0]
        return [float(xy[0]), float(xy[1]), float(z)]

    route_id = f"high_point_{high_point.high_point_id}_{flight['flight']}"
    return {
        "route_id": route_id,
        "status": "HIGHPOINT_HEIGHTFIELD_SCREENED",
        "source_audit": source,
        "low_xyz_m": xyz(low_s),
        "high_xyz_m": xyz(high_s),
        "uphill_unit_xy": [float(v) for v in climb],
        "heading_yaw_rad": math.atan2(climb[1], climb[0]),
        "length_m": float(low_s - high_s),
        "width_m": ROUTE_WIDTH_M,
        "height_gain_m": float(flight["platform_level_m"] - flight["landing_level_m"]),
        "steps": len(flight["risers_m"]),
        "step_rises_m": [round(value, 4) for value in flight["risers_m"]],
        "waypoints_xyz_m": [xyz(value) for value in np.linspace(low_s, high_s, 17)],
        "dynamic_status": "NOT_RUN",
    }


def build_high_point_catalog(
    asset: FieldAsset | str | Path, source_manifest=None
) -> dict[str, Any]:
    """Measure every high point of a pack and publish one route per flight."""

    field = asset if isinstance(asset, FieldAsset) else FieldAsset.open(asset, verify=True)
    season = field.season
    season.require(HIGH_POINTS)
    grid = load_heightfield(field)
    source = season.default_source
    measurements, patches, rejected = [], [], []
    for high_point in source.high_points:
        try:
            measured = measure_high_point(field, high_point, grid=grid)
        except ValueError as exc:
            rejected.append({"high_point_id": high_point.high_point_id, "reason": str(exc)})
            continue
        if source_manifest is not None:
            measured["source_audit"] = compare_with_source(
                field, high_point, source_manifest, grid=grid, measured=measured
            )
        else:
            measured["source_audit"] = None
        measurements.append(measured)
        for flight in ("highland", "floor"):
            patches.append(
                {
                    "patch_id": f"high_point_{high_point.high_point_id}_{flight}",
                    "route": _route(
                        grid, high_point, measured["flights"][flight], measured["source_audit"]
                    ),
                }
            )
    audits = [m["source_audit"] for m in measurements if m["source_audit"]]
    # Every recorded high point has to be measured.  A stair screen may reject a
    # source face and stay passing, because it screens every face it can find;
    # this record names the high points that must exist, so one that cannot be
    # measured at all is a defect in the field or in the record, not a candidate
    # that failed to qualify.
    passed = (
        bool(patches)
        and len(measurements) == len(source.high_points)
        and all(m["status"] == "PASS" for m in measurements)
        and all(audit["status"] == "PASS" for audit in audits)
    )
    return {
        "schema_version": 1,
        "scenario_id": "high_point_basic",
        "status": "PASS_STATIC" if passed else "FAIL",
        "source_manifest_sha256": field.manifest_sha256,
        "heightfield_samples_sha256": field.collision["samples_sha256"],
        "season": season.id,
        "source_step_sha256": source.step.sha256,
        "source_glb_sha256": source.source_glb_sha256,
        "high_points": measurements,
        "patches": patches,
        "rejected": rejected,
        "scope": (
            "Measured 2027 high points: platform level, both step flights and the "
            "collision heightfield's agreement with the official source surface. "
            "Robot climbing is a separate result."
        ),
        "geometry_modified": False,
    }


def load_high_point_catalog(asset: FieldAsset, path_or_dict) -> dict[str, Any]:
    """Read a catalog (or a high-point-check report) for this exact field pack."""

    if path_or_dict is None:
        raise ValueError("high_point_basic requires --route-catalog from high-point-check")
    document = (
        path_or_dict
        if isinstance(path_or_dict, dict)
        else json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    )
    catalog = document.get("catalog", document)
    if (
        catalog.get("scenario_id") != "high_point_basic"
        or catalog.get("source_manifest_sha256") != asset.manifest_sha256
        or catalog.get("heightfield_samples_sha256") != asset.collision["samples_sha256"]
    ):
        raise ValueError("high point catalog does not match the selected field pack")
    if not catalog.get("patches"):
        raise ValueError("high point catalog has no routes")
    return catalog


def high_points_for(asset: FieldAsset) -> tuple[HighPoint, ...]:
    """The recorded high points of the pack's season."""

    season = asset.season
    season.require(HIGH_POINTS)
    return season.default_source.high_points


__all__ = [
    "build_high_point_catalog",
    "compare_with_source",
    "high_points_for",
    "load_high_point_catalog",
    "measure_flight",
    "measure_high_point",
]
