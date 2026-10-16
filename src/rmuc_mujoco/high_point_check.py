"""Physical checks for a season's high points (制高点).

For every high point, in every selected profile:

* a rigid block dropped on each measured level must come to rest on the surface
  the pack declares there, and at that level's own design height;
* a sled driven at the lowest riser from the level below must reach the face,
  be held there, and stay at that level instead of riding up a ramp.

A high point is climbed, not driven up, so no wheel probe is used: a 120 mm
wheel cannot mount a 200 mm riser with a vertical face, and the rulebook's
question is whether the steps exist and are level, not whether a particular
robot can climb them.  Probes are rigid test bodies, not robots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

from .acceptance import _contact_sample, _warning_counts
from .high_point import build_high_point_catalog
from .manifest import FieldAsset
from .mjcf import HFIELD_NAME, inject_exact_heightfield
from .query import load_heightfield, sample_mujoco_heightfield
from .seasons import HIGH_POINTS

# The block is short enough to sit inside one 305 mm tread clear of both ramps.
BLOCK_LENGTH_M = 0.20
BLOCK_WIDTH_M = 0.24
BLOCK_MASS_KG = 2.0
BLOCK_HEIGHT_M = 0.10
RUNNER_RADIUS_M = 0.03
DROP_CLEARANCE_M = 0.02
SETTLE_S = 1.0
# A dropped block's rest height must match the surface the pack declares at its
# own footprint, and that surface must be the level the run detector measured.
SUPPORT_TOLERANCE_M = 0.005
LEVEL_TOLERANCE_M = 0.02
MAX_PENETRATION_M = 0.01
# Drive the sled at the outermost riser from the level below it.  The sled's
# front edge starts START_M from the face, so the body must travel CONTACT_M to
# touch it.  DRIVE_M takes it well past that, so a field whose riser is a ramp
# lets the sled keep going while a vertical face stops it at CONTACT_M.
START_M = 0.55
CONTACT_M = START_M - BLOCK_LENGTH_M / 2
DRIVE_M = 0.60
# Contact softness and the feet's rounded noses move the stop point a little.
BLOCK_TOLERANCE_M = 0.05
SPEED_M_S = 0.3
# The sled's body must not gain height; a positive gain means it rode the face
# instead of being stopped by it.
RISE_TOLERANCE_M = 0.05
CONTACT = dict(friction=[0.2, 0.005, 0.0001], priority=1)


def _level_probe_points(flight: dict[str, Any]) -> list[dict[str, Any]]:
    """One probe position per measured level, kept clear of both level ends."""

    direction = np.array(flight["direction_xy"], dtype=float)
    center = np.asarray(flight["start_xy_m"], dtype=float)
    limit = BLOCK_LENGTH_M / 2 + 0.02
    points = []
    for index, run in enumerate(flight["runs"]):
        low, high = run["start_m"] + limit, run["stop_m"] - limit
        if high <= low:
            continue
        along = min(max(0.5 * (run["start_m"] + run["stop_m"]), low), high)
        points.append(
            {
                "level_index": index,
                "along_m": float(along),
                "xy_m": [float(v) for v in center + along * direction],
                "design_level_m": float(run["level_m"]),
                "length_m": float(run["length_m"]),
            }
        )
    return points


def _drop_block(asset: FieldAsset, profile: str, xy, drop_z: float):
    spec = mujoco.MjSpec.from_file(str(asset.entrypoint_for(profile)))
    body = spec.worldbody.add_body(name="rmuc_high_point_probe", pos=[*xy, 0.0])
    body.add_joint(name="rmuc_high_point_probe_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    body.add_geom(
        name="rmuc_high_point_probe_block",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, BLOCK_HEIGHT_M / 2],
        size=[BLOCK_LENGTH_M / 2, BLOCK_WIDTH_M / 2, BLOCK_HEIGHT_M / 2],
        mass=BLOCK_MASS_KG,
        **CONTACT,
    )
    model = spec.compile()
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    data = mujoco.MjData(model)
    data.qpos[0] = drop_z
    mujoco.mj_forward(model, data)
    return model, data


def _footprint_floor(field, start: np.ndarray, unit: np.ndarray) -> float:
    """Highest surface under the sled's footprint.

    The sled is rigid and cannot pitch, and the field carries a camber, so
    starting it at the height under its centre would bury the uphill foot by
    a few millimetres.  A buried foot ploughs through the heightfield's cell
    walls instead of rolling, which reads as a blocked probe.  Start above the
    whole footprint and let the sled settle onto its higher foot.
    """

    lateral = np.array([-unit[1], unit[0]])
    along, across = np.meshgrid(
        np.linspace(-BLOCK_LENGTH_M / 2, BLOCK_LENGTH_M / 2, 9),
        np.linspace(-BLOCK_WIDTH_M / 2, BLOCK_WIDTH_M / 2, 9),
    )
    points = start + along.ravel()[:, None] * unit + across.ravel()[:, None] * lateral
    return float(np.max(sample_mujoco_heightfield(field, points[:, 0], points[:, 1])))


def _sled_model(asset: FieldAsset, profile: str, start, direction: np.ndarray, field):
    """A sled: four rounded feet under a block, free only along the flight.

    The feet are spheres, not the cylinders a runner suggests.  A rigid body
    free only to slide has no rolling DOF, so either shape slides rather than
    rolls; but a cylinder lying on a heightfield contacts a whole row of cells
    at once (~50 contacts here) and the resulting chatter stalls the sled on
    the field's 2.6% camber, while a sphere's point contact slides cleanly.
    """

    spec = mujoco.MjSpec.from_file(str(asset.entrypoint_for(profile)))
    body = spec.worldbody.add_body(name="rmuc_high_point_probe", pos=[*start, 0.0])
    body.add_joint(
        name="rmuc_high_point_probe_path",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[*direction, 0],
    )
    body.add_joint(name="rmuc_high_point_probe_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    lateral = np.array([-direction[1], direction[0]])
    for index, along in enumerate((-1, 1)):
        centre = along * (BLOCK_LENGTH_M / 2 - RUNNER_RADIUS_M) * direction
        half_width = BLOCK_WIDTH_M / 2 * lateral
        for side, offset in enumerate((-1, 1)):
            body.add_geom(
                name=f"rmuc_high_point_probe_foot_{index}{side}",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[*(centre + offset * half_width), RUNNER_RADIUS_M],
                size=[RUNNER_RADIUS_M],
                mass=BLOCK_MASS_KG / 8,
                **CONTACT,
            )
    body.add_geom(
        name="rmuc_high_point_probe_block",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 2 * RUNNER_RADIUS_M + BLOCK_HEIGHT_M / 2],
        size=[BLOCK_LENGTH_M / 2, BLOCK_WIDTH_M / 2, BLOCK_HEIGHT_M / 2],
        mass=BLOCK_MASS_KG / 2,
        **CONTACT,
    )
    model = spec.compile()
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    data = mujoco.MjData(model)
    data.qpos[1] = _footprint_floor(field, start, direction) + 0.002
    mujoco.mj_forward(model, data)
    return model, data


def check_levels(
    asset: FieldAsset, high_point_id: str, flight: dict[str, Any], profile: str, field
) -> list[dict[str, Any]]:
    """Drop the block on every measured level of one flight."""

    rows = []
    for point in _level_probe_points(flight):
        sample = point["xy_m"]
        surface = float(
            sample_mujoco_heightfield(field, np.array([sample[0]]), np.array([sample[1]]))[0]
        )
        model, data = _drop_block(
            asset, profile, sample, surface + BLOCK_HEIGHT_M / 2 + DROP_CLEARANCE_M
        )
        contacts = 0
        max_penetration = 0.0
        for _ in range(math.ceil(SETTLE_S / model.opt.timestep)):
            mujoco.mj_step(model, data)
            _, penetration, _finite = _contact_sample(model, data)
            contacts += 1 if int(data.ncon) else 0
            max_penetration = max(max_penetration, penetration)
        rest = float(data.qpos[0])
        support_error = rest - surface
        # The design level is a tilt-removed coordinate, so compare like with
        # like by removing the same plane from the rest height.
        coefficients = flight["plane_coefficients_xy"]
        level_error = (rest - (coefficients[0] * sample[0] + coefficients[1] * sample[1])) - point[
            "design_level_m"
        ]
        passed = (
            abs(support_error) <= SUPPORT_TOLERANCE_M
            and abs(level_error) <= LEVEL_TOLERANCE_M
            and contacts > 0
            and max_penetration <= MAX_PENETRATION_M
            and not _warning_counts(data)
        )
        rows.append(
            {
                "high_point_id": high_point_id,
                "profile": profile,
                "probe": "rests_on_level",
                "flight": flight["flight"],
                "level_index": point["level_index"],
                "probe_xy_m": sample,
                "level_length_m": point["length_m"],
                "surface_height_m": surface,
                "rest_height_m": rest,
                "support_error_m": round(support_error, 5),
                "level_error_m": round(level_error, 5),
                "contact_steps": contacts,
                "max_penetration_m": round(max_penetration, 6),
                "warnings": _warning_counts(data),
                "final_qpos": [float(v) for v in data.qpos],
                "status": "PASS" if passed else "FAIL",
            }
        )
    return rows


def check_riser(
    asset: FieldAsset, high_point_id: str, flight: dict[str, Any], profile: str, field
) -> dict[str, Any]:
    """Drive the sled at the outermost riser from the level below it."""

    direction = np.array(flight["direction_xy"], dtype=float)
    center = np.asarray(flight["start_xy_m"], dtype=float)
    riser = float(flight["riser_positions_m"][-1])
    lower_level = float(flight["runs"][-1]["level_m"])
    start_along = riser + START_M
    start = center + start_along * direction
    climb = -direction
    model, data = _sled_model(asset, profile, start, climb, field)
    # Let the sled settle onto the level before measuring, so a later rise is a
    # gain over its resting height and not slack in the initial drop.
    for _ in range(math.ceil(SETTLE_S / model.opt.timestep)):
        mujoco.mj_step(model, data)
    start_z = float(data.qpos[1])
    highest = start_z
    max_penetration = 0.0
    drive_steps = math.ceil((DRIVE_M / SPEED_M_S + 1.2) / model.opt.timestep)
    for _ in range(drive_steps):
        target = min(DRIVE_M, SPEED_M_S * data.time)
        velocity = float(np.clip(4.0 * (target - data.qpos[0]) + SPEED_M_S, -SPEED_M_S, SPEED_M_S))
        data.qfrc_applied[0] = float(np.clip(160.0 * (velocity - data.qvel[0]), -100.0, 100.0))
        mujoco.mj_step(model, data)
        _, penetration, _finite = _contact_sample(model, data)
        max_penetration = max(max_penetration, penetration)
        highest = max(highest, float(data.qpos[1]))
    climbed = float(data.qpos[0])
    rise = highest - start_z
    # Getting to the face at all is what makes the result about the riser: a
    # sled that stopped short, or that never left its start, proves nothing.
    reached = climbed >= CONTACT_M - BLOCK_TOLERANCE_M
    # A face holds the sled at the face; carrying on past it means the sled rode
    # up a ramp, and a positive body rise says the same thing directly.
    blocked = climbed <= CONTACT_M + BLOCK_TOLERANCE_M
    passed = (
        reached
        and blocked
        and rise <= RISE_TOLERANCE_M
        and max_penetration <= MAX_PENETRATION_M
        and not _warning_counts(data)
    )
    return {
        "high_point_id": high_point_id,
        "profile": profile,
        "probe": "blocked_by_riser",
        "flight": flight["flight"],
        "riser_along_m": riser,
        "probe_start_xy_m": [float(v) for v in start],
        "riser_rise_m": round(float(flight["risers_m"][-1]), 4),
        "lower_level_m": lower_level,
        "contact_m": round(CONTACT_M, 4),
        "drive_m": DRIVE_M,
        "start_height_m": start_z,
        "climbed_m": climbed,
        "reached_face": reached,
        "blocked_by_face": blocked,
        "highest_gain_m": rise,
        "max_penetration_m": round(max_penetration, 6),
        "warnings": _warning_counts(data),
        "final_qpos": [float(v) for v in data.qpos],
        "status": "PASS" if passed else "FAIL",
    }


def run_high_point_checks(
    pack: str | Path | FieldAsset,
    *,
    source_manifest: str | Path | None = None,
    profiles: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Measure every high point, probe it physically, and publish its routes.

    The report carries the measured catalog, so ``high-point-check`` output is
    also what ``--route-catalog`` consumes, exactly as ``stairs-check`` is.
    """

    asset = pack if isinstance(pack, FieldAsset) else FieldAsset.open(pack, verify=True)
    asset.season.require(HIGH_POINTS)
    selected = tuple(profiles) if profiles is not None else asset.available_runtime_profiles
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("profiles must be nonempty and unique")
    field = load_heightfield(asset)
    catalog = build_high_point_catalog(asset, source_manifest)
    trials, parity = [], []
    for measured in catalog["high_points"]:
        high_point_id = measured["high_point_id"]
        reference = None
        for profile in selected:
            rows = [
                *check_levels(
                    asset, high_point_id, measured["flights"]["highland"], profile, field
                ),
                *check_levels(asset, high_point_id, measured["flights"]["floor"], profile, field),
                check_riser(asset, high_point_id, measured["flights"]["highland"], profile, field),
                check_riser(asset, high_point_id, measured["flights"]["floor"], profile, field),
            ]
            trials.extend(rows)
            states = np.concatenate([np.asarray(row["final_qpos"]) for row in rows])
            if reference is None:
                reference = states
            else:
                difference = float(np.max(np.abs(states - reference)))
                parity.append(
                    {
                        "high_point_id": high_point_id,
                        "profile": profile,
                        "max_final_state_difference": difference,
                        "status": "PASS" if difference <= 1e-9 else "FAIL",
                    }
                )
    passed = catalog["status"] == "PASS_STATIC" and all(
        row["status"] == "PASS" for row in trials + parity
    )
    return {
        "schema_version": 1,
        "scenario_id": "high_point_basic",
        "status": "PASS" if passed else "FAIL",
        "catalog": catalog,
        "season": asset.season.id,
        "source_manifest_sha256": asset.manifest_sha256,
        "heightfield_samples_sha256": asset.collision["samples_sha256"],
        "trials": trials,
        "profile_parity": parity,
        "summary": {
            "high_points": len(catalog["high_points"]),
            "trials": len(trials),
            "passed": sum(row["status"] == "PASS" for row in trials),
            "failed": sum(row["status"] != "PASS" for row in trials),
            "catalog_status": catalog["status"],
        },
        "mujoco_version": mujoco.__version__,
        "geometry_modified": False,
        "scope": (
            "Rigid probes check that every measured high point level supports a body and "
            "that each riser's face reaches and blocks one.  The catalog is the same "
            "measurement, published as climb routes.  Robots and gaits are not evaluated."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rmuc_mujoco.high_point_check")
    parser.add_argument("pack", type=Path)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="source build manifest; adds the audit against the official GLB",
    )
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_high_point_checks(
        args.pack, source_manifest=args.source_manifest, profiles=args.profiles
    )
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["summary"]}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["check_levels", "check_riser", "run_high_point_checks"]
