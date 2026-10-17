"""Build an opt-in, source-exact contact replacement for the official walls.

This is deliberately not a runtime-pack writer.  It creates a locally hashed
collision experiment so the exact convex wall can replace, rather than overlap,
the single-heightfield roof.  The candidate stays blocked until robot and wall
tip contact gates are satisfied by the runtime-pack integration.  RMUC 2026's
walls are closed convex parts 402/403; RMUC 2027 merges the same walls into
larger parts, so each is derived as the hull of its part's vertices in a box.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from .collision_candidate import CONVEX_WALL_EVIDENCE
from .manifest import FieldAsset, sha256_file
from .query import HeightFieldData, load_heightfield
from .seasons import SourceWall
from .wall_tip_repair import END_BAND_M, MIN_MISSING_ROOF_M, _projected_top


ROOF_ERROR_LIMIT_M = 1.0e-5
BOTTOM_ERROR_LIMIT_M = 1.0e-5
WALL_CONTACT_SOLREF = "0.005 1"
CONVEX_PLANE_TOLERANCE_M = 2.0e-6
# Derived walls: vertices this close to the recorded box belong to the wall,
# points this close to a facet plane lie on it (the GLB stores float32), and a
# facet facing further down than this is the underside the terrain hides.
WALL_BOX_MARGIN_M = 1.0e-3
HULL_PLANE_TOLERANCE_M = 1.0e-5
UNDERSIDE_NORMAL_Z = -0.3
HULL_VOLUME_TOLERANCE_M3 = 1.0e-5
UNCOVERED_AREA_TOLERANCE_M2 = 1.0e-5
EXACT_WALL_BOUNDARY = (
    "local exact-source physics candidate only; correctly spawned 120 mm sphere "
    "tip trials reached at most 4.67 mm wall penetration without warnings; "
    "matched frozen Fudan 0.5 m/s near-wall trials had no solver warning, but "
    "downhill hfield pairs can reach MuJoCo's 50-contact cap and broader robot "
    "routes are unverified"
)
DERIVED_WALL_BOUNDARY = (
    "local derived-source physics candidate only; each wall is the convex hull of "
    "its source part's vertices in the recorded box, every part triangle there lies "
    "on the hull and only the recorded side area and the buried underside are not "
    "source faces; wall-check reports the contact trials and broader robot routes "
    "are unverified"
)


def _exact_source_convex_mesh(source_mesh: Any, *, source_part_index: int) -> Any:
    """Use the audited source facets as a convex mesh without a SciPy hull.

    The source GLB duplicates vertices at material/normal seams.  Welding those
    coincident vertices restores the closed 20-triangle polyhedron.  Checking
    every source vertex against every outward facet rejects a non-convex edit;
    asking trimesh for ``convex_hull`` would silently require optional SciPy.
    """

    evidence = CONVEX_WALL_EVIDENCE[source_part_index]
    mesh = source_mesh.copy()
    mesh.merge_vertices()
    if (
        len(mesh.vertices) != evidence["unique_vertices"]
        or len(mesh.faces) != evidence["hull_facets"]
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or mesh.euler_number != 2
        or mesh.volume <= 0.0
        or not np.isclose(mesh.volume, evidence["hull_volume_m3"], atol=1.0e-5)
    ):
        raise ValueError(f"source part {source_part_index} closed convex topology changed")
    signed_distance = np.einsum(
        "fvc,fc->fv",
        mesh.vertices[None, :, :] - mesh.triangles_center[:, None, :],
        mesh.face_normals,
    )
    if float(np.max(signed_distance)) > CONVEX_PLANE_TOLERANCE_M:
        raise ValueError(f"source part {source_part_index} is no longer convex")
    return mesh


def _replace_wall_roof(
    wall_mesh: Any,
    field: HeightFieldData,
    output_height_m: np.ndarray,
    *,
    expected_nodes: int,
) -> dict[str, Any]:
    """Transfer one exact wall footprint from hfield roof to source mesh.

    Every replaced hfield sample must be the source roof, and the replacement
    must be the source wall's lower face and the existing lower flank.  These
    gates stop a same-looking but geometrically different wall from carving
    floor or other official structure.
    """

    bounds = np.asarray(wall_mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("source wall bounds are invalid")
    x, y = field.x_m, field.y_m
    if (
        x.ndim != 1
        or y.ndim != 1
        or output_height_m.shape != (len(y), len(x))
        or field.height_m.shape != output_height_m.shape
        or not np.allclose(np.diff(x), 0.01, rtol=0.0, atol=1.0e-6)
        or not np.allclose(np.diff(y), 0.01, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("wall replacement requires the verified 1 cm world grid")

    ix = np.flatnonzero((x >= bounds[0, 0]) & (x <= bounds[1, 0]))
    iy = np.flatnonzero((y >= bounds[0, 1]) & (y <= bounds[1, 1]))
    if len(ix) < 3 or len(iy) < 3 or ix[0] < 2 or ix[-1] > len(x) - 3:
        raise ValueError("wall replacement footprint or flank is incomplete")
    top = _projected_top(wall_mesh, x[ix], y[iy])
    covered = np.isfinite(top)
    if int(covered.sum()) != expected_nodes or not np.all(covered):
        raise ValueError("source wall roof footprint changed")
    original = field.height_m[np.ix_(iy, ix)]
    top_error = float(np.max(np.abs(original - top)))
    if top_error > ROOF_ERROR_LIMIT_M:
        raise ValueError(f"heightfield roof is not the exact source wall: {top_error}")

    inverted = wall_mesh.copy()
    inverted.vertices[:, 2] *= -1.0
    bottom = -_projected_top(inverted, x[ix], y[iy])
    lower_flank = np.minimum(field.height_m[iy, ix[0] - 2], field.height_m[iy, ix[-1] + 2])
    bottom_gap = float(np.max(bottom - lower_flank[:, None]))
    bottom_burial = float(np.max(lower_flank[:, None] - bottom))
    if not np.isfinite(bottom).all() or bottom_gap > BOTTOM_ERROR_LIMIT_M:
        raise ValueError(f"lower flank would leave a gap beneath the source wall: {bottom_gap}")

    # Assign only after all geometry/ownership checks pass.  The exact wall
    # mesh owns the roof and sides, and the hfield still owns the source floor.
    output_height_m[np.ix_(iy, ix)] = lower_flank[:, None]
    return {
        "roof_nodes_transferred": expected_nodes,
        "roof_source_abs_error_max_m": top_error,
        "maximum_gap_beneath_source_wall_m": max(0.0, bottom_gap),
        "maximum_source_wall_burial_beneath_floor_m": bottom_burial,
        "source_bounds_world_m": bounds.tolist(),
        "lower_flank_height_range_m": [float(lower_flank.min()), float(lower_flank.max())],
        "maximum_removed_roof_height_m": float(np.max(original - lower_flank[:, None])),
        "changed_grid_rectangle_yx": [int(iy[0]), int(iy[-1]), int(ix[0]), int(ix[-1])],
    }


def _exact_walls(release) -> bool:
    return all(wall.hull_box_xy_m is None for wall in release.source_walls)


def _hull_facets(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Facets of the convex hull of a small point set, without SciPy.

    A plane through three points with every point on one side supports a facet;
    the facet is the set of points on that plane.  Facets are keyed by that set,
    not by the plane, because a plane fitted from a skinny triple of float32
    points far from the origin can tilt enough to move its offset by a
    millimetre.  Only maximal sets are kept, and each facet's plane is the least
    squares fit of its own points.  The cost grows with the cube of the point
    count, which suits the dozen vertices of a wall.
    """

    sets = set()
    for i, j, k in combinations(range(len(points)), 3):
        normal = np.cross(points[j] - points[i], points[k] - points[i])
        length = float(np.linalg.norm(normal))
        if length < 1.0e-12:
            continue
        side = (points - points[i]) @ (normal / length)
        if np.all(side >= -HULL_PLANE_TOLERANCE_M):
            side = -side
        elif not np.all(side <= HULL_PLANE_TOLERANCE_M):
            continue
        sets.add(frozenset(np.flatnonzero(side >= -HULL_PLANE_TOLERANCE_M).tolist()))
    facets = []
    for members in sets:
        if any(members < other for other in sets):
            continue
        index = np.array(sorted(members))
        centre = points[index].mean(axis=0)
        normal = np.linalg.svd(points[index] - centre)[2][2]
        if np.max((points - centre) @ normal) > HULL_PLANE_TOLERANCE_M:
            normal = -normal
        if np.max((points - centre) @ normal) > HULL_PLANE_TOLERANCE_M:
            raise ValueError("derived wall points are not convex")
        facets.append((index, normal, centre))
    return facets


def derive_wall(part_mesh: Any, box_xy_m: tuple[float, float, float, float]):
    """Hull a wall out of a merged source part and measure it against the part.

    Returns the hull in the part's frame and its measurements: vertex and
    triangle counts, volume, how many part triangles inside the box are not on
    the hull surface, and the hull side area no part triangle covers.  Downward
    facets are the underside, which the terrain hides, and are not counted.
    """

    import trimesh

    part = part_mesh.copy()
    part.merge_vertices()
    vertices = np.asarray(part.vertices, dtype=np.float64)
    x0, x1, y0, y1 = box_xy_m
    inside = (
        (vertices[:, 0] >= x0 - WALL_BOX_MARGIN_M)
        & (vertices[:, 0] <= x1 + WALL_BOX_MARGIN_M)
        & (vertices[:, 1] >= y0 - WALL_BOX_MARGIN_M)
        & (vertices[:, 1] <= y1 + WALL_BOX_MARGIN_M)
    )
    points = vertices[inside]
    if len(points) < 4:
        raise ValueError("derived wall box holds no wall")
    facets = _hull_facets(points)
    faces = []
    for index, normal, centre in facets:
        u = points[index[0]] - centre
        u /= np.linalg.norm(u)
        w = np.cross(normal, u)
        offsets = points[index] - centre
        order = index[np.argsort(np.arctan2(offsets @ w, offsets @ u))]
        faces.append([[order[0], order[t], order[t + 1]] for t in range(1, len(order) - 1)])
    hull = trimesh.Trimesh(points, np.concatenate(faces), process=False)
    # Part triangles inside the box, as indices into the hull's points.
    local = np.full(len(vertices), -1)
    local[inside] = np.arange(len(points))
    own = local[part.faces[np.all(inside[part.faces], axis=1)]]

    def area(triangles: np.ndarray) -> float:
        corners = points[triangles]
        return float(
            np.sum(
                np.linalg.norm(
                    np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
                )
            )
            / 2
        )

    on_facet = np.zeros(len(own), dtype=bool)
    uncovered = 0.0
    for (index, normal, _), triangles in zip(facets, faces):
        mine = np.all(np.isin(own, index), axis=1)
        on_facet |= mine
        if normal[2] >= UNDERSIDE_NORMAL_Z:
            uncovered += area(np.asarray(triangles)) - area(own[mine])
    covered_points = set().union(*(set(index.tolist()) for index, _, _ in facets))
    return hull, {
        "box_vertices": int(len(points)),
        "hull_vertices": len(covered_points),
        "hull_triangles": int(len(hull.faces)),
        "hull_volume_m3": float(hull.volume),
        "watertight": bool(hull.is_watertight),
        "part_triangles_in_box": int(len(own)),
        "part_triangles_off_hull": int(np.sum(~on_facet)),
        "uncovered_side_area_m2": float(uncovered),
    }


def verified_derived_wall(part_mesh: Any, wall: SourceWall):
    """Derive one recorded wall and refuse any change from its record."""

    hull, measured = derive_wall(part_mesh, wall.hull_box_xy_m)
    if (
        measured["box_vertices"] != wall.hull_vertices
        or measured["hull_vertices"] != wall.hull_vertices
        or measured["hull_triangles"] != wall.hull_triangles
        or not measured["watertight"]
        or abs(measured["hull_volume_m3"] - wall.hull_volume_m3) > HULL_VOLUME_TOLERANCE_M3
    ):
        raise ValueError(f"derived wall {wall.source_part_index} hull changed")
    if measured["part_triangles_off_hull"]:
        raise ValueError(
            f"source part {wall.source_part_index} has triangles inside its derived wall"
        )
    if (
        abs(measured["uncovered_side_area_m2"] - wall.uncovered_side_area_m2)
        > UNCOVERED_AREA_TOLERANCE_M2
    ):
        raise ValueError(f"derived wall {wall.source_part_index} side no longer matches the source")
    return hull, measured


def _replace_derived_wall_roof(
    wall_mesh: Any,
    field: HeightFieldData,
    output_height_m: np.ndarray,
    *,
    wall: SourceWall,
) -> dict[str, Any]:
    """Hand one derived wall's roof to its hull and fill up to its underside.

    As for an exact wall, every covered sample must already be the source roof,
    except the recorded end samples that the builder's spike filter left low.
    A derived wall's underside meets the road along the wall's inner foot rather
    than lying on the floor, so each covered sample becomes the higher of the
    lower flank and the hull underside: no void is left beneath the wall and no
    heightfield rises into it.  Samples the hull does not cover are unchanged.
    """

    bounds = np.asarray(wall_mesh.bounds, dtype=np.float64)
    x, y = field.x_m, field.y_m
    if (
        bounds.shape != (2, 3)
        or not np.isfinite(bounds).all()
        or output_height_m.shape != (len(y), len(x))
        or field.height_m.shape != output_height_m.shape
        or not np.allclose(np.diff(x), 0.01, rtol=0.0, atol=1.0e-6)
        or not np.allclose(np.diff(y), 0.01, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("wall replacement requires the verified 1 cm world grid")
    ix = np.flatnonzero((x >= bounds[0, 0]) & (x <= bounds[1, 0]))
    iy = np.flatnonzero((y >= bounds[0, 1]) & (y <= bounds[1, 1]))
    if len(ix) < 3 or len(iy) < 3 or ix[0] < 2 or ix[-1] > len(x) - 3:
        raise ValueError("wall replacement footprint or flank is incomplete")
    top = _projected_top(wall_mesh, x[ix], y[iy])
    covered = np.isfinite(top)
    if int(covered.sum()) != wall.roof_nodes:
        raise ValueError("derived wall roof footprint changed")
    original = field.height_m[np.ix_(iy, ix)]
    error = np.where(covered, original - top, 0.0)
    low = covered & (error < -MIN_MISSING_ROOF_M)
    near_end = np.minimum(y[iy] - bounds[0, 1], bounds[1, 1] - y[iy]) <= END_BAND_M
    if int(low.sum()) != wall.spike_filtered_end_nodes or not np.all(
        np.broadcast_to(near_end[:, None], low.shape)[low]
    ):
        raise ValueError("derived wall end samples changed")
    top_error = float(np.max(np.abs(error[covered & ~low])))
    if top_error > ROOF_ERROR_LIMIT_M:
        raise ValueError(f"heightfield roof is not the derived source wall: {top_error}")
    inverted = wall_mesh.copy()
    inverted.vertices[:, 2] *= -1.0
    bottom = -_projected_top(inverted, x[ix], y[iy])
    if not np.isfinite(bottom[covered]).all():
        raise ValueError("derived wall underside does not cover its roof")
    lower_flank = np.minimum(field.height_m[iy, ix[0] - 2], field.height_m[iy, ix[-1] + 2])
    floor = np.maximum(lower_flank[:, None], bottom)
    output_height_m[np.ix_(iy, ix)] = np.where(covered, floor, original)
    return {
        "roof_nodes_transferred": wall.roof_nodes,
        "spike_filtered_end_nodes": wall.spike_filtered_end_nodes,
        "roof_source_abs_error_max_m": top_error,
        "maximum_gap_beneath_source_wall_m": 0.0,
        "maximum_source_wall_burial_beneath_floor_m": float(np.max((floor - bottom)[covered])),
        "source_bounds_world_m": bounds.tolist(),
        "lower_flank_height_range_m": [float(lower_flank.min()), float(lower_flank.max())],
        "maximum_removed_roof_height_m": float(np.max((original - floor)[covered])),
        "changed_grid_rectangle_yx": [int(iy[0]), int(iy[-1]), int(ix[0]), int(ix[-1])],
        "underside_owner": "heightfield_follows_derived_wall_underside",
    }


def _verified_source_scene(
    source_root: Path, asset: FieldAsset
) -> tuple[Any, np.ndarray, dict[str, str]]:
    """Load the original GLB in the runtime frame after source/hash checks."""

    import trimesh

    release = asset.field_source
    source_manifest_path = source_root / "manifest.json"
    source_sha = sha256_file(source_manifest_path)
    if source_sha != asset.manifest["source_identity"]["field_manifest_sha256"]:
        raise ValueError("source build does not match this runtime pack")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source["source"]["sha256"] != release.step.sha256:
        raise ValueError("source build is not the audited official STEP")
    if source["collision"]["samples_sha256"] != asset.collision["samples_sha256"]:
        raise ValueError("source and runtime collision samples disagree")
    # Exact walls need the build's wall-tip repair for their roof gate; derived
    # walls account for the same few spike-filtered end samples themselves.
    if (
        _exact_walls(release)
        and source["collision"]["verified_wall_tip_repair"]["status"] != "PASS"
    ):
        raise ValueError("candidate requires the verified seven-node wall-tip repair")
    glb_record = source["conversion"]["colored_intermediate_glb"]
    if glb_record["sha256"] != release.source_glb_sha256:
        raise ValueError("source GLB identity changed")
    glb_path = source_root / glb_record["file"]
    if sha256_file(glb_path) != release.source_glb_sha256:
        raise ValueError("source GLB file hash changed")

    scene = trimesh.load(glb_path, force="scene", process=False)
    spawn = asset.recommended_spawn
    translation = np.asarray(
        [
            -float(spawn["x_before_translation_m"]),
            -float(spawn["y_before_translation_m"]),
            -float(source["conversion"]["main_floor_height_before_shift_m"])
            - float(spawn["terrain_height_m"]),
        ],
        dtype=np.float64,
    )
    return (
        scene,
        translation,
        {
            "source_manifest_sha256": source_sha,
            "source_glb_sha256": release.source_glb_sha256,
        },
    )


def _source_walls(source_root: Path, asset: FieldAsset) -> tuple[dict[int, Any], dict[str, str]]:
    """Load the release's walls only after the source/runtime hashes agree."""

    release = asset.field_source
    scene, translation, source_identity = _verified_source_scene(source_root, asset)
    nodes = list(scene.graph.nodes_geometry)
    if len(nodes) <= max(wall.source_part_index for wall in release.source_walls):
        raise ValueError("source GLB part order is incomplete")
    walls: dict[int, Any] = {}
    for wall in release.source_walls:
        part = wall.source_part_index
        transform, geometry_name = scene.graph[nodes[part]]
        original = scene.geometry[geometry_name].copy()
        original.apply_transform(transform)
        if wall.hull_box_xy_m is not None:
            hull, _ = verified_derived_wall(original, wall)
        else:
            if len(original.faces) != 20 or len(np.unique(original.vertices, axis=0)) != 12:
                raise ValueError(f"source part {part} topology changed")
            hull = _exact_source_convex_mesh(original, source_part_index=part)
        hull.apply_translation(translation)
        walls[part] = hull
    return walls, source_identity


def build_verified_wall_replacement(
    source_build: Path, runtime_pack: Path
) -> tuple[HeightFieldData, dict[int, Any], dict[str, Any]]:
    """Return a local candidate without modifying either verified input."""

    asset = FieldAsset.open(runtime_pack)
    release = asset.field_source
    if not release.source_walls:
        raise ValueError(f"{release.season_id} {release.version} records no source walls")
    source_root = Path(source_build).expanduser().resolve()
    field = load_heightfield(asset)
    walls, source_identity = _source_walls(source_root, asset)
    updated = field.height_m.copy()
    reports = []
    regions = []
    for wall in release.source_walls:
        part = wall.source_part_index
        if wall.hull_box_xy_m is None:
            record = _replace_wall_roof(walls[part], field, updated, expected_nodes=wall.roof_nodes)
        else:
            record = _replace_derived_wall_roof(walls[part], field, updated, wall=wall)
        record["source_part_index"] = part
        reports.append(record)
        regions.append(record["changed_grid_rectangle_yx"])
    for first, second in combinations(regions, 2):
        # The present walls are far apart.  Reject a changed source geometry
        # before one wall's floor can be overwritten by the other.
        if not (first[1] < second[0] or second[1] < first[0]):
            if not (first[3] < second[2] or second[3] < first[2]):
                raise ValueError("exact wall ownership rectangles overlap")
    exact = _exact_walls(release)
    report = {
        "status": "EXPERIMENTAL_BLOCKED",
        "source_identity": source_identity,
        "base_runtime_manifest_sha256": asset.manifest_sha256,
        "base_collision_samples_sha256": asset.collision["samples_sha256"],
        "source_part_indices": [wall.source_part_index for wall in release.source_walls],
        "contact_ownership": "source_convex_mesh_roof_and_sides; existing_hfield_source_bottom",
        "wall_contact_solref": WALL_CONTACT_SOLREF,
        "wall_friction": [1.0, 0.005, 0.0001],
        "walls": reports,
        "outside_wall_rectangles_bitwise_unchanged": True,
        "validation_boundary": EXACT_WALL_BOUNDARY if exact else DERIVED_WALL_BOUNDARY,
    }
    return HeightFieldData(field.x_m, field.y_m, updated), walls, report


def write_local_wall_contact_candidate(
    source_build: Path, runtime_pack: Path, output_directory: Path
) -> Path:
    """Write only experimental local assets, never a runtime-schema manifest."""

    from PIL import Image

    asset = FieldAsset.open(runtime_pack)
    field, walls, report = build_verified_wall_replacement(source_build, runtime_pack)
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {output}")
    collision_dir = output / "collision"
    collision_dir.mkdir(parents=True)

    spawn = asset.recommended_spawn
    source_height = field.height_m + float(spawn["terrain_height_m"])
    maximum = float(asset.collision["maximum_height_m"])
    if float(np.min(source_height)) < 0.0 or float(np.max(source_height)) > maximum:
        raise ValueError("candidate source height exceeds the unchanged hfield range")
    npz_path = collision_dir / "rmuc2026_heightfield.npz"
    np.savez_compressed(
        npz_path,
        x_m=field.x_m + float(spawn["x_before_translation_m"]),
        y_m=field.y_m + float(spawn["y_before_translation_m"]),
        height_m=source_height,
    )
    pixels = np.rint(np.clip(source_height / maximum, 0.0, 1.0) * 65535.0).astype(np.uint16)
    png_path = collision_dir / "rmuc2026_heightfield.png"
    Image.fromarray(pixels[::-1]).save(png_path)

    xml_root = ET.parse(asset.entrypoint_for("collision_only")).getroot()
    mesh_assets = xml_root.find("./asset")
    worldbody = xml_root.find("./worldbody")
    assert mesh_assets is not None and worldbody is not None
    for part in sorted(walls):
        relative = f"collision/official_wall_{part}.obj"
        walls[part].export(output / relative, file_type="obj")
        name = f"{asset.field_source.season_id}_official_wall_{part}"
        ET.SubElement(mesh_assets, "mesh", name=name, file=relative)
        ET.SubElement(
            worldbody,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            contype="2",
            conaffinity="1",
            friction="1 0.005 0.0001",
            solref=WALL_CONTACT_SOLREF,
            group="0",
        )
    xml_path = output / "rmuc2026_wall_contact_candidate.xml"
    ET.ElementTree(xml_root).write(xml_path, encoding="utf-8", xml_declaration=True)

    files = []
    for path in (
        npz_path,
        png_path,
        xml_path,
        *(output / f"collision/official_wall_{i}.obj" for i in sorted(walls)),
    ):
        files.append(
            {
                "file": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    report["artifact_type"] = "rmuc2026_wall_contact_local_candidate"
    report["candidate_collision_samples_sha256"] = sha256_file(npz_path)
    report["files"] = files
    report["runtime_schema_pack"] = False
    manifest_path = output / "candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_build", type=Path)
    parser.add_argument("runtime_pack", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args(argv)
    print(
        write_local_wall_contact_candidate(
            args.source_build, args.runtime_pack, args.output_directory
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
