"""Published tunnel routes: through the channel under a roof, or over the roof.

Rulebook 4.4.2 puts the tunnels on the road area: a robot may pass through a
tunnel (穿越) or over it (跨越), and table 5-27 tells the two apart by whether
the robot was detected inside on its way from one end to the other.  Each roof
of a pack's tunnel layer gives one route of each kind along its travel axis.
The two share their line and ends; the runtime tells them apart by where the
robot's body passes the roof span.  The tunnel probes judge the collision
layer, so publishing a route claims nothing about which robots can drive it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import FieldAsset
from .query import load_heightfield, sample_mujoco_heightfield
from .scenarios import scenario_descriptor

# A route end is where a robot spawns, so it clears the roof span by this much:
# room for a robot about a metre long to start outside the tunnel.
ROUTE_MARGIN_M = 0.60
PASSES = ("through", "over")


def _height(field, point: np.ndarray) -> float:
    return float(sample_mujoco_heightfield(field, point[:1], point[1:])[0])


def _route(field, roof: dict[str, Any], kind: str) -> dict[str, Any]:
    (x0, y0), (x1, y1) = roof["span_world_xy_m"]
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    along_x = roof["travel_axis"] == "x"
    axis = np.array([1.0, 0.0]) if along_x else np.array([0.0, 1.0])
    half_along = ((x1 - x0) if along_x else (y1 - y0)) / 2
    reach = half_along + ROUTE_MARGIN_M
    ends = [centre - reach * axis, centre + reach * axis]
    (bx0, by0), (bx1, by1) = field.bounds_xy_m
    if any(not (bx0 <= p[0] <= bx1 and by0 <= p[1] <= by1) for p in ends):
        raise ValueError("route_outside_heightfield")
    heights = [_height(field, point) for point in ends]
    # Ends are named by height, as the stair and slope routes name theirs, so
    # uphill climbs a sloping channel; a level road runs along the travel axis.
    if heights[1] < heights[0] - 0.01:
        ends.reverse()
        heights.reverse()
    low, high = ends
    uphill = (high - low) / np.linalg.norm(high - low)
    top = float(max(roof["top_m"]))

    def xyz(along: float) -> list[float]:
        point = low + along * uphill
        z = _height(field, point)
        if kind == "over" and abs(along - reach) <= half_along:
            z = max(z, top)
        return [float(point[0]), float(point[1]), z]

    return {
        "route_id": f"tunnel_{roof['tunnel_id']}_{kind}",
        "status": "TUNNEL_LAYER_SCREENED",
        "tunnel_id": roof["tunnel_id"],
        "tunnel_pass": kind,
        "low_xyz_m": [float(low[0]), float(low[1]), heights[0]],
        "high_xyz_m": [float(high[0]), float(high[1]), heights[1]],
        "uphill_unit_xy": [float(value) for value in uphill],
        "heading_yaw_rad": math.atan2(uphill[1], uphill[0]),
        "length_m": float(2 * reach),
        # The corridor is the channel's own width; walls and rails bound it.
        "width_m": float((y1 - y0) if along_x else (x1 - x0)),
        "height_gain_m": heights[1] - heights[0],
        "roof_station_m": float(reach),
        "roof_half_length_m": float(half_along),
        "roof_underside_m": float(min(roof["underside_m"])),
        "roof_top_m": top,
        "clearance_m": float(min(roof["clearance_m"])),
        "waypoints_xyz_m": [xyz(along) for along in np.linspace(0.0, 2 * reach, 17)],
        "dynamic_status": "NOT_RUN",
    }


def build_tunnel_catalog(pack: str | Path | FieldAsset) -> dict[str, Any]:
    """A through and an over route for every roof of the pack's tunnel layer."""

    asset = pack if isinstance(pack, FieldAsset) else FieldAsset.open(pack, verify=True)
    layer = asset.collision.get("tunnel_contact_layer")
    if layer is None:
        raise ValueError("pack has no tunnel layer; build one with tunnel-pack")
    field = load_heightfield(asset)
    patches, rejected = [], []
    for roof in layer["roofs"]:
        for kind in PASSES:
            route_id = f"tunnel_{roof['tunnel_id']}_{kind}"
            try:
                patches.append({"patch_id": route_id, "route": _route(field, roof, kind)})
            except ValueError as exc:
                rejected.append({"id": route_id, "reason": str(exc)})
    return {
        "schema_version": 1,
        "scenario_id": "tunnel_basic",
        "status": "PASS_STATIC" if patches else "NO_TUNNEL_ROUTES",
        "source_manifest_sha256": asset.manifest_sha256,
        "profile_hash": scenario_descriptor(asset, "tunnel_basic")["profile_hash"],
        "heightfield_samples_sha256": asset.collision["samples_sha256"],
        "patches": patches,
        "rejected": rejected,
        "scope": (
            "through and over routes along each roof's travel axis; the tunnel probes "
            "judge the collision layer, not which robots can drive a route"
        ),
        "validation_status": asset.manifest.get("validation_status"),
    }


def load_tunnel_catalog(asset: FieldAsset, path_or_dict) -> dict[str, Any]:
    """Read a catalog (or a tunnel-check report) for this exact field pack."""

    if path_or_dict is None:
        raise ValueError("tunnel_basic requires --route-catalog from tunnel-check")
    document = (
        path_or_dict
        if isinstance(path_or_dict, dict)
        else json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    )
    catalog = document.get("catalog", document)
    if (
        catalog.get("scenario_id") != "tunnel_basic"
        or catalog.get("source_manifest_sha256") != asset.manifest_sha256
        or catalog.get("heightfield_samples_sha256") != asset.collision["samples_sha256"]
    ):
        raise ValueError("tunnel catalog does not match the selected field pack")
    if not catalog.get("patches"):
        raise ValueError("tunnel catalog has no routes")
    return catalog


__all__ = ["PASSES", "ROUTE_MARGIN_M", "build_tunnel_catalog", "load_tunnel_catalog"]
