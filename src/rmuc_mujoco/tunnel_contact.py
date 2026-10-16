"""Tunnel roofs as source meshes over a lowered heightfield.

A single-height heightfield records a tunnel roof as a solid block, so nothing
can drive through.  For every roof its season pins, this layer lowers the
heightfield under the roof's covered span to the channel floor and adds the
roof as a convex collision mesh.  Both come from the build's unsimplified GLB,
and every replaced sample must equal that roof's top, so a changed source is
rejected instead of carving the wrong structure.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from .manifest import FieldAsset, sha256_file
from .query import HeightFieldData, load_heightfield
from .seasons import TUNNELS, FieldSource, TunnelRoof
from .source_contact_pack import _append_wall_geoms, _update_file_record
from .training_region import _heightfield_png
from .wall_active_contact import WALL_CONTACT_SOLREF

LAYER_KIND = "official_source_tunnel_roofs_v1"
ROOF_FRICTION = (1.0, 0.005, 0.0001)
ROOF_CONTYPE = 2
ROOF_CONAFFINITY = 1
ROOF_MESH_ROLE = "tunnel_roof_mesh"
# Underside/top faces are the roof part's near-horizontal faces.
FACE_NORMAL_Z = 0.99
# An underside sits above the channel floor; the frame's feet do not.
MIN_UNDERSIDE_ABOVE_FLOOR_M = 0.15
SAME_LEVEL_M = 0.03
SPAN_TOLERANCE_M = 0.005
PLANE_RESIDUAL_LIMIT_M = 5.0e-4
ROOF_TOP_ERROR_LIMIT_M = 1.0e-3
MIN_CLEARANCE_M = 0.15
EDGE_BAND_M = 0.002
# Side-rail lips may reach this far under a roof edge; they stay solid.
RAIL_BAND_M = 0.02
RAY_INSET_M = 0.001

# Vertices 0-3: underside corners (x0y0, x1y0, x1y1, x0y1); 4-7: the top ones.
HEXAHEDRON_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int64,
)


def roof_mesh_name(tunnel_id: str) -> str:
    return f"rmuc_tunnel_roof_{tunnel_id}"


def roof_mesh_file(tunnel_id: str) -> str:
    return f"collision/tunnel_roof_{tunnel_id}.obj"


@dataclass(frozen=True)
class RoofGeometry:
    """A planar roof slab over one rectangular covered span (CAD frame)."""

    roof: TunnelRoof
    span_xy_m: tuple[float, float, float, float]
    underside_plane: tuple[float, float, float]
    top_plane: tuple[float, float, float]

    @staticmethod
    def _z(plane: Sequence[float], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return plane[0] + plane[1] * x + plane[2] * y

    def underside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._z(self.underside_plane, x, y)

    def top(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._z(self.top_plane, x, y)

    def vertices(self) -> np.ndarray:
        x0, x1, y0, y1 = self.span_xy_m
        x = np.array([x0, x1, x1, x0])
        y = np.array([y0, y0, y1, y1])
        return np.vstack(
            (
                np.column_stack((x, y, self.underside(x, y))),
                np.column_stack((x, y, self.top(x, y))),
            )
        )

    @property
    def travel_axis(self) -> str:
        """The channel runs across the roof's short side."""

        x0, x1, y0, y1 = self.span_xy_m
        return "x" if x1 - x0 < y1 - y0 else "y"


def _fit_plane(points: np.ndarray) -> tuple[tuple[float, float, float], float]:
    design = np.column_stack((np.ones(len(points)), points[:, 0], points[:, 1]))
    coefficients, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    residual = float(np.max(np.abs(design @ coefficients - points[:, 2])))
    return (float(coefficients[0]), float(coefficients[1]), float(coefficients[2])), residual


def roof_geometry(part_mesh: Any, roof: TunnelRoof, *, floor_z_m: float) -> RoofGeometry:
    """Extract one roof from its source part.

    ``part_mesh`` is in the CAD frame and ``floor_z_m`` is the CAD height of the
    main floor.  The covered span is the part's lowest underside inside the
    search box; its top is the part's upward faces over that span.
    """

    triangles = np.asarray(part_mesh.triangles, dtype=np.float64)
    normals = np.asarray(part_mesh.face_normals, dtype=np.float64)
    centers = triangles.mean(axis=1)
    x0, x1, y0, y1 = roof.search_box_xy_m
    inside = (
        (centers[:, 0] >= x0)
        & (centers[:, 0] <= x1)
        & (centers[:, 1] >= y0)
        & (centers[:, 1] <= y1)
    )
    down = (
        inside
        & (normals[:, 2] < -FACE_NORMAL_Z)
        & (centers[:, 2] - floor_z_m > MIN_UNDERSIDE_ABOVE_FLOOR_M)
    )
    if not down.any():
        raise ValueError(f"tunnel {roof.tunnel_id}: no roof underside in the search box")
    lowest = float(centers[down, 2].min())
    underside = triangles[down & (centers[:, 2] <= lowest + SAME_LEVEL_M)].reshape(-1, 3)
    span = (
        float(underside[:, 0].min()),
        float(underside[:, 0].max()),
        float(underside[:, 1].min()),
        float(underside[:, 1].max()),
    )
    size = (span[1] - span[0], span[3] - span[2])
    if any(abs(a - b) > SPAN_TOLERANCE_M for a, b in zip(size, roof.covered_span_m)):
        raise ValueError(
            f"tunnel {roof.tunnel_id}: covered span {size[0]:.3f} x {size[1]:.3f} m "
            f"differs from {roof.covered_span_m}"
        )
    underside_plane, underside_residual = _fit_plane(underside)
    over_span = (
        inside
        & (normals[:, 2] > FACE_NORMAL_Z)
        & (centers[:, 0] > span[0])
        & (centers[:, 0] < span[1])
        & (centers[:, 1] > span[2])
        & (centers[:, 1] < span[3])
    )
    if not over_span.any():
        raise ValueError(f"tunnel {roof.tunnel_id}: no roof top over the covered span")
    top = triangles[over_span].reshape(-1, 3)
    covers = (
        top[:, 0].min() <= span[0] + SPAN_TOLERANCE_M
        and top[:, 0].max() >= span[1] - SPAN_TOLERANCE_M
        and top[:, 1].min() <= span[2] + SPAN_TOLERANCE_M
        and top[:, 1].max() >= span[3] - SPAN_TOLERANCE_M
    )
    top_plane, top_residual = _fit_plane(top)
    if (
        not covers
        or underside_residual > PLANE_RESIDUAL_LIMIT_M
        or top_residual > PLANE_RESIDUAL_LIMIT_M
    ):
        raise ValueError(
            f"tunnel {roof.tunnel_id}: roof is not one planar slab over its span "
            f"(underside {underside_residual:.2e} m, top {top_residual:.2e} m)"
        )
    geometry = RoofGeometry(roof, span, underside_plane, top_plane)
    corners = geometry.vertices()
    if float(np.min(corners[4:, 2] - corners[:4, 2])) <= 0.005:
        raise ValueError(f"tunnel {roof.tunnel_id}: roof slab has no thickness")
    return geometry


def _verified_source(source_root: Path, asset: FieldAsset) -> tuple[dict, Path, str]:
    """Bind the source build and its GLB to this pack before reading geometry."""

    manifest_path = source_root / "manifest.json"
    source_manifest_sha = sha256_file(manifest_path)
    if source_manifest_sha != asset.manifest["source_identity"]["field_manifest_sha256"]:
        raise ValueError("source build does not match this runtime pack")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    field_source = asset.field_source
    if source["source"]["sha256"] != field_source.step.sha256:
        raise ValueError("source build is not the pack's official STEP")
    glb_record = source["conversion"]["colored_intermediate_glb"]
    glb_path = (source_root / str(glb_record["file"])).resolve()
    if source_root not in glb_path.parents or not glb_path.is_file():
        raise ValueError("source GLB is missing or outside the build")
    if (
        glb_record["sha256"] != field_source.source_glb_sha256
        or sha256_file(glb_path) != field_source.source_glb_sha256
    ):
        raise ValueError("source GLB is not the pinned conversion of this STEP")
    base_samples = source["collision"]["samples_sha256"]
    current = asset.collision["samples_sha256"]
    walls = asset.collision.get("source_contact_layer")
    chained = (
        walls is not None
        and walls["base_collision_samples_sha256"] == base_samples
        and walls["updated_collision_samples_sha256"] == current
    )
    if current != base_samples and not chained:
        raise ValueError("runtime heightfield is neither the source build's nor its wall layer's")
    return source, glb_path, source_manifest_sha


def _first_hit_below(meshes: Sequence[Any], origins: np.ndarray) -> np.ndarray:
    """Height of the first surface straight below each origin (NaN on a miss)."""

    import trimesh

    combined = trimesh.util.concatenate(list(meshes))
    directions = np.tile((0.0, 0.0, -1.0), (len(origins), 1))
    locations, rays, _ = combined.ray.intersects_location(origins, directions, multiple_hits=True)
    floor = np.full(len(origins), -np.inf)
    np.maximum.at(floor, rays, locations[:, 2])
    floor[~np.isfinite(floor)] = np.nan
    return floor


def _lower_span(
    geometry: RoofGeometry,
    meshes: Sequence[Any],
    field: HeightFieldData,
    updated: np.ndarray,
    translation: np.ndarray,
    floor_minimum_m: float,
) -> dict[str, Any]:
    """Replace the roof samples in one span with the channel floor.

    Like the builder, heights below the main floor (``floor_minimum_m`` in the
    world frame) are raised to it; clearance is reported after that.
    """

    x0, x1, y0, y1 = geometry.span_xy_m
    x_cad = field.x_m - translation[0]
    y_cad = field.y_m - translation[1]
    ix = np.flatnonzero((x_cad >= x0 - EDGE_BAND_M) & (x_cad <= x1 + EDGE_BAND_M))
    iy = np.flatnonzero((y_cad >= y0 - EDGE_BAND_M) & (y_cad <= y1 + EDGE_BAND_M))
    if len(ix) < 3 or len(iy) < 3:
        raise ValueError(f"tunnel {geometry.roof.tunnel_id}: span misses the heightfield grid")
    gx, gy = np.meshgrid(x_cad[ix], y_cad[iy])
    original = field.height_m[np.ix_(iy, ix)]
    top = geometry.top(gx, gy) + translation[2]
    interior = (
        (gx > x0 + EDGE_BAND_M)
        & (gx < x1 - EDGE_BAND_M)
        & (gy > y0 + EDGE_BAND_M)
        & (gy < y1 - EDGE_BAND_M)
    )
    error = np.abs(original - top)
    interior_error = float(np.max(error[interior]))
    if interior_error > ROOF_TOP_ERROR_LIMIT_M:
        raise ValueError(
            f"tunnel {geometry.roof.tunnel_id}: heightfield is not this roof's top "
            f"({interior_error:.4f} m)"
        )
    # Interior samples are the roof top (checked above).  At the span edge a
    # sample may land on the slab's chamfer; any height inside the slab is
    # roof material that the roof mesh now owns.
    underside = geometry.underside(gx, gy) + translation[2]
    in_slab = (original >= underside - ROOF_TOP_ERROR_LIMIT_M) & (
        original <= top + ROOF_TOP_ERROR_LIMIT_M
    )
    roof_samples = (error <= ROOF_TOP_ERROR_LIMIT_M) | (~interior & in_slab)
    nearby = [
        mesh
        for mesh in meshes
        if not (
            mesh.bounds[1, 0] < x0
            or mesh.bounds[0, 0] > x1
            or mesh.bounds[1, 1] < y0
            or mesh.bounds[0, 1] > y1
        )
    ]
    # Cast from just under the roof, 1 mm inside the span: a ray on the span
    # boundary would graze the slab's own side face instead of the floor.
    rx = np.clip(gx[roof_samples], x0 + RAY_INSET_M, x1 - RAY_INSET_M)
    ry = np.clip(gy[roof_samples], y0 + RAY_INSET_M, y1 - RAY_INSET_M)
    origins = np.column_stack((rx, ry, geometry.underside(rx, ry) - 1.0e-3))
    floor_cad = _first_hit_below(nearby, origins)
    if not np.isfinite(floor_cad).all():
        raise ValueError(f"tunnel {geometry.roof.tunnel_id}: no floor under part of the roof")
    clearance = origins[:, 2] + 1.0e-3 - floor_cad
    # Samples over a side rail stay solid; only the open channel is lowered.
    # A low sample away from the span edge means the source changed.
    open_channel = clearance >= MIN_CLEARANCE_M
    sx, sy = gx[roof_samples], gy[roof_samples]
    edge_distance = np.minimum.reduce((sx - x0, x1 - sx, sy - y0, y1 - sy))
    misplaced = ~open_channel & (edge_distance > RAIL_BAND_M)
    if misplaced.any():
        raise ValueError(
            f"tunnel {geometry.roof.tunnel_id}: roof clearance {clearance[misplaced].min():.3f} m "
            f"inside the channel is below {MIN_CLEARANCE_M} m"
        )
    lowered = np.zeros_like(roof_samples)
    lowered[roof_samples] = open_channel
    floor_world = np.maximum(floor_cad[open_channel] + translation[2], floor_minimum_m)
    clearance = origins[open_channel, 2] + 1.0e-3 + translation[2] - floor_world
    block = updated[np.ix_(iy, ix)]
    block[lowered] = floor_world
    updated[np.ix_(iy, ix)] = block
    corners = geometry.vertices() + translation
    return {
        "tunnel_id": geometry.roof.tunnel_id,
        "source_part_index": geometry.roof.source_part_index,
        "span_world_xy_m": [
            [round(x0 + translation[0], 6), round(y0 + translation[1], 6)],
            [round(x1 + translation[0], 6), round(y1 + translation[1], 6)],
        ],
        "travel_axis": geometry.travel_axis,
        "underside_m": [
            round(float(corners[:4, 2].min()), 6),
            round(float(corners[:4, 2].max()), 6),
        ],
        "top_m": [round(float(corners[4:, 2].min()), 6), round(float(corners[4:, 2].max()), 6)],
        "floor_m": [round(float(floor_world.min()), 6), round(float(floor_world.max()), 6)],
        "clearance_m": [
            round(float(clearance.min()), 6),
            round(float(np.median(clearance)), 6),
            round(float(clearance.max()), 6),
        ],
        "hfield_nodes_lowered": int(lowered.sum()),
        "changed_grid_rectangle_yx": [int(iy[0]), int(iy[-1]), int(ix[0]), int(ix[-1])],
    }


def build_tunnel_replacement(
    source_build: str | Path, runtime_pack: str | Path | FieldAsset
) -> tuple[HeightFieldData, dict[str, np.ndarray], dict[str, Any]]:
    """Lower every pinned roof span and return the roof slabs in the world frame."""

    import trimesh

    asset = runtime_pack if isinstance(runtime_pack, FieldAsset) else FieldAsset.open(runtime_pack)
    asset.season.require(TUNNELS)
    field_source: FieldSource = asset.field_source
    if not field_source.tunnel_roofs:
        raise ValueError(f"{asset.season.title} {field_source.version} pins no tunnel roofs")
    if asset.collision.get("tunnel_contact_layer") is not None:
        raise ValueError("runtime pack already has a tunnel layer")
    source_root = Path(source_build).expanduser().resolve()
    source, glb_path, source_manifest_sha = _verified_source(source_root, asset)
    floor_z = float(source["conversion"]["main_floor_height_before_shift_m"])
    spawn = asset.recommended_spawn
    translation = np.array(
        [
            -float(spawn["x_before_translation_m"]),
            -float(spawn["y_before_translation_m"]),
            -floor_z - float(spawn["terrain_height_m"]),
        ]
    )
    meshes = list(trimesh.load(glb_path, force="scene", process=False).dump(concatenate=False))
    field = load_heightfield(asset)
    updated = field.height_m.copy()
    roofs: dict[str, np.ndarray] = {}
    records = []
    for roof in field_source.tunnel_roofs:
        if roof.source_part_index >= len(meshes):
            raise ValueError(
                f"tunnel {roof.tunnel_id}: source part {roof.source_part_index} missing"
            )
        geometry = roof_geometry(meshes[roof.source_part_index], roof, floor_z_m=floor_z)
        records.append(
            _lower_span(
                geometry,
                meshes,
                field,
                updated,
                translation,
                floor_minimum_m=-float(spawn["terrain_height_m"]),
            )
        )
        roofs[roof.tunnel_id] = geometry.vertices() + translation
    changed = updated != field.height_m
    allowed = np.zeros_like(changed)
    for record in records:
        r0, r1, c0, c1 = record["changed_grid_rectangle_yx"]
        allowed[r0 : r1 + 1, c0 : c1 + 1] = True
    if np.any(changed & ~allowed):
        raise ValueError("tunnel layer changed samples outside the roof spans")
    report = {
        "source_manifest_sha256": source_manifest_sha,
        "source_glb_sha256": field_source.source_glb_sha256,
        "roofs": records,
    }
    return HeightFieldData(field.x_m, field.y_m, updated), roofs, report


def _write_roof_obj(path: Path, vertices: np.ndarray) -> None:
    lines = [f"v {x:.9f} {y:.9f} {z:.9f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in HEXAHEDRON_FACES]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


CLAIM_BOUNDARY = (
    "Roof slabs and the lowered channel floor come from the pinned source GLB; "
    "tunnel-check probes passage under, blocking above the clearance and support "
    "on the roof. Robot traversal and official material properties are unverified."
)


def export_tunnel_pack(
    source_build: str | Path,
    runtime_pack: str | Path,
    output: str | Path,
) -> dict:
    """Derive a pack whose tunnels can be driven through and over.

    Apply an optional source-wall layer first; this layer is applied last.
    Every runtime profile gets the same heightfield and roof meshes.
    """

    asset = FieldAsset.open(runtime_pack, verify=True)
    field, roofs, report = build_tunnel_replacement(source_build, asset)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for original in asset.verified_files:
            target = staging / original.relative_to(asset.root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        manifest = copy.deepcopy(dict(asset.manifest))
        collision = manifest["collision"]
        base_samples = collision["samples_sha256"]
        spawn = asset.recommended_spawn
        source_height = field.height_m + float(spawn["terrain_height_m"])
        maximum = float(collision["maximum_height_m"])
        minimum = float(collision["minimum_height_m"])
        if (
            float(source_height.min()) < minimum - 1e-8
            or float(source_height.max()) > maximum + 1e-8
        ):
            raise ValueError("tunnel floor exceeds the declared heightfield range")
        samples_relative = str(collision["samples_file"])
        image_relative = str(collision["image_file"])
        np.savez_compressed(
            staging / samples_relative,
            x_m=field.x_m + float(spawn["x_before_translation_m"]),
            y_m=field.y_m + float(spawn["y_before_translation_m"]),
            height_m=source_height,
        )
        normalized = (source_height - minimum) / (maximum - minimum)
        (staging / image_relative).write_bytes(_heightfield_png(normalized))
        _update_file_record(manifest, staging, samples_relative)
        _update_file_record(manifest, staging, image_relative)
        collision["samples_sha256"] = sha256_file(staging / samples_relative)
        collision["image_sha256"] = sha256_file(staging / image_relative)

        mesh_records = []
        for record in report["roofs"]:
            tunnel_id = record["tunnel_id"]
            relative = roof_mesh_file(tunnel_id)
            _write_roof_obj(staging / relative, roofs[tunnel_id])
            _update_file_record(manifest, staging, relative, role=ROOF_MESH_ROLE)
            mesh_records.append(
                {
                    "name": roof_mesh_name(tunnel_id),
                    "file": relative,
                    "sha256": sha256_file(staging / relative),
                    "source_part_index": record["source_part_index"],
                    "contype": ROOF_CONTYPE,
                    "conaffinity": ROOF_CONAFFINITY,
                    "friction": list(ROOF_FRICTION),
                    "solref": [float(value) for value in WALL_CONTACT_SOLREF.split()],
                }
            )
        for profile in asset.available_runtime_profiles:
            relative = str(asset.runtime_profiles[profile]["entrypoint"])
            _append_wall_geoms(staging / relative, mesh_records)
            _update_file_record(manifest, staging, relative)
        field_source = asset.field_source
        collision["tunnel_contact_layer"] = {
            "kind": LAYER_KIND,
            "status": "EXPERIMENTAL_BLOCKED",
            "season": field_source.season_id,
            "official_step_version": field_source.version,
            "source_manifest_sha256": report["source_manifest_sha256"],
            "source_glb_sha256": report["source_glb_sha256"],
            "base_collision_samples_sha256": base_samples,
            "updated_collision_samples_sha256": collision["samples_sha256"],
            "roofs": report["roofs"],
            "mesh_geoms": mesh_records,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        manifest["contents"]["file_count_excluding_manifest"] = len(manifest["contents"]["files"])
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        FieldAsset.open(staging, verify=True)
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rmuc_mujoco.tunnel_contact")
    parser.add_argument("source_build", type=Path)
    parser.add_argument("runtime_pack", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    manifest = export_tunnel_pack(args.source_build, args.runtime_pack, args.output)
    print(json.dumps(manifest["collision"]["tunnel_contact_layer"]["roofs"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LAYER_KIND",
    "RoofGeometry",
    "build_tunnel_replacement",
    "export_tunnel_pack",
    "roof_geometry",
    "roof_mesh_file",
    "roof_mesh_name",
]
