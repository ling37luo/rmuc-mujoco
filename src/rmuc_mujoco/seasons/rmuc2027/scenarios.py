"""RMUC 2027 scenario table: the core scenarios plus the high-point climbs.

The high point replaces the 2026 fly ramps, so this table has no ramp entries
and ``slope_basic`` keeps the whole core table: 2027 has no ramps to exclude
from the ordinary-slope search.
"""

from __future__ import annotations

from typing import Mapping

from ...scenarios import BASE_SCENARIOS, ScenarioSpec


def _high_point() -> ScenarioSpec:
    """Both base-area high points, one route per step flight."""

    return ScenarioSpec(
        "high_point_basic",
        "collision_only",
        "800 mm base-area high points and both 200 mm step flights",
        {
            "selector": "source_bound_high_point_catalog",
            "visual": "omitted",
            "catalog_id": "high_point_basic",
            "catalog_command": "high-point-check",
        },
        {
            "strategy": "catalog_patch_spawn",
            "controller_owns_gear_and_action_mapping": True,
            "goal": "independent_highland_or_floor_flight_climb",
            "default_direction": "uphill",
            "directions": ["uphill", "downhill"],
            "robot_climbing_separate_from_contact_validation": True,
        },
        (),
        ("human_view", "mujoco", "isaac"),
    )


SCENARIOS: Mapping[str, ScenarioSpec] = {
    **BASE_SCENARIOS,
    "high_point_basic": _high_point(),
}
