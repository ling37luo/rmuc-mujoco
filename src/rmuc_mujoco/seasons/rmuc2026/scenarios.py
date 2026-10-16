"""RMUC 2026 scenario table: the core scenarios plus the two fly ramps."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from ...scenarios import BASE_SCENARIOS, ScenarioSpec
from .ramp_audit import FIXED_FLY_RAMPS


def _ramp_route(route: Any) -> dict[str, Any]:
    return {
        "route_id": route.route_id,
        "source_part_index": route.source_part_index,
        "low_edge_center_xyz_m": list(route.low_edge_center_xyz_m),
        "high_edge_center_xyz_m": list(route.high_edge_center_xyz_m),
        "uphill_unit_xy": list(route.uphill_unit_xy),
        "horizontal_run_m": route.horizontal_run_m,
        "surface_width_m": route.surface_width_m,
        "slope_angle_degrees": route.slope_angle_degrees,
        "cad_low_seam_along_m": route.cad_low_seam_along_m,
        "cad_high_seam_along_m": route.cad_high_seam_along_m,
    }


_RAMP_NORTH, _RAMP_SOUTH = tuple(_ramp_route(route) for route in FIXED_FLY_RAMPS)


def _slope_basic() -> ScenarioSpec:
    base = BASE_SCENARIOS["slope_basic"]
    region: dict[str, Any] = {}
    for key, value in base.region.items():
        if key == "route_screen":
            region["fly_ramps_excluded"] = ["fly_ramp_north", "fly_ramp_south"]
        region[key] = value
    return dataclasses.replace(
        base,
        purpose="ordinary traversable slopes below the dedicated fly ramps",
        region=region,
    )


def _fly_ramp(scenario_id: str, side: str, route: dict[str, Any]) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id,
        "collision_only",
        f"audited {side} fly ramp",
        {"selector": "route", "route_id": route["route_id"], "visual": "omitted"},
        {"strategy": "route_endpoints", "wheel_diameter_m": 0.12},
        (route,),
        ("human_view", "mujoco", "isaac"),
    )


SCENARIOS: Mapping[str, ScenarioSpec] = {
    "full_eval": BASE_SCENARIOS["full_eval"],
    "turn_basic": BASE_SCENARIOS["turn_basic"],
    "stairs_basic": BASE_SCENARIOS["stairs_basic"],
    "slope_basic": _slope_basic(),
    "fly_ramp_north": _fly_ramp("fly_ramp_north", "north", _RAMP_NORTH),
    "fly_ramp_south": _fly_ramp("fly_ramp_south", "south", _RAMP_SOUTH),
    "boundary_contact": BASE_SCENARIOS["boundary_contact"],
}
