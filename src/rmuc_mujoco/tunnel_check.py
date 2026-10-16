"""Physical checks for a pack's tunnel layer.

For every roof, in every selected profile:

* a block 30 mm lower than the roof's minimum clearance slides along the
  channel, in both directions, and must clear the roof at least once without
  ever touching it;
* a block 30 mm higher than its maximum clearance must be stopped by the roof
  at least once and must never pass under it;
* a 50 mm ball dropped on the roof centre must come to rest on the roof top.

Probes are rigid test bodies, not robots; they check the collision layer only.
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
from .manifest import FieldAsset
from .mjcf import HFIELD_NAME, inject_exact_heightfield
from .query import load_heightfield, sample_mujoco_heightfield

PROBE_MARGIN_M = 0.03
PROBE_LENGTH_M = 0.20
PROBE_WIDTH_M = 0.30
PROBE_MASS_KG = 2.0
RUNNER_RADIUS_M = 0.03
LEAD_M = 0.05
SPEED_M_S = 0.3
BALL_RADIUS_M = 0.05
SUPPORT_TOLERANCE_M = 0.005
MAX_PENETRATION_M = 0.01


def _roof_frame(roof: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    (x0, y0), (x1, y1) = roof["span_world_xy_m"]
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    unit = np.array([1.0, 0.0]) if roof["travel_axis"] == "x" else np.array([0.0, 1.0])
    half_along = (x1 - x0) / 2 if roof["travel_axis"] == "x" else (y1 - y0) / 2
    return centre, unit, half_along


def _footprint_floor(field, start: np.ndarray, unit: np.ndarray) -> float:
    lateral = np.array([-unit[1], unit[0]])
    s, w = np.meshgrid(
        np.linspace(-PROBE_LENGTH_M / 2, PROBE_LENGTH_M / 2, 9),
        np.linspace(-PROBE_WIDTH_M / 2, PROBE_WIDTH_M / 2, 9),
    )
    points = start + s.ravel()[:, None] * unit + w.ravel()[:, None] * lateral
    return float(np.max(sample_mujoco_heightfield(field, points[:, 0], points[:, 1])))


def _block_model(
    asset: FieldAsset, field, profile: str, start: np.ndarray, unit: np.ndarray, height: float
):
    """A sled: two rounded runners under a block, ``height`` tall overall."""

    spec = mujoco.MjSpec.from_file(str(asset.entrypoint_for(profile)))
    body = spec.worldbody.add_body(name="rmuc_tunnel_probe", pos=[*start, 0.0])
    body.add_joint(name="rmuc_tunnel_probe_path", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[*unit, 0])
    body.add_joint(name="rmuc_tunnel_probe_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    lateral = np.array([-unit[1], unit[0]])
    # Priority makes the probe's low friction govern its contacts, so it
    # slides at the commanded speed instead of testing field friction.
    contact = dict(friction=[0.2, 0.005, 0.0001], priority=1)
    for index, along in enumerate((-1, 1)):
        centre = along * (PROBE_LENGTH_M / 2 - RUNNER_RADIUS_M) * unit
        half_width = PROBE_WIDTH_M / 2 * lateral
        body.add_geom(
            name=f"rmuc_tunnel_probe_runner_{index}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            fromto=[
                *(centre - half_width),
                RUNNER_RADIUS_M,
                *(centre + half_width),
                RUNNER_RADIUS_M,
            ],
            size=[RUNNER_RADIUS_M],
            mass=PROBE_MASS_KG / 4,
            **contact,
        )
    block_height = height - 2 * RUNNER_RADIUS_M
    half = [PROBE_LENGTH_M / 2, PROBE_WIDTH_M / 2]
    if unit[1] != 0.0:
        half.reverse()
    body.add_geom(
        name="rmuc_tunnel_probe_block",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 2 * RUNNER_RADIUS_M + block_height / 2],
        size=[*half, block_height / 2],
        mass=PROBE_MASS_KG / 2,
        **contact,
    )
    model = spec.compile()
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    data = mujoco.MjData(model)
    data.qpos[1] = _footprint_floor(field, start, unit) + 0.002
    mujoco.mj_forward(model, data)
    return model, data


def _ball_model(asset: FieldAsset, profile: str, centre: np.ndarray, drop_z: float):
    spec = mujoco.MjSpec.from_file(str(asset.entrypoint_for(profile)))
    body = spec.worldbody.add_body(name="rmuc_tunnel_probe", pos=[*centre, 0.0])
    body.add_joint(name="rmuc_tunnel_probe_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    body.add_geom(
        name="rmuc_tunnel_probe_ball",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS_M],
        mass=0.5,
    )
    model = spec.compile()
    inject_exact_heightfield(model, asset, hfield_name=HFIELD_NAME)
    data = mujoco.MjData(model)
    data.qpos[0] = drop_z
    mujoco.mj_forward(model, data)
    return model, data


def _roof_contacts(pairs, roof_geom: str) -> int:
    return sum(
        count
        for pair, count in pairs.items()
        if roof_geom in (name.strip() for name in pair.split("|"))
    )


def _roof_penetration(model, data, roof_id: int) -> float:
    deepest = 0.0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if roof_id in (int(contact.geom1), int(contact.geom2)):
            deepest = max(deepest, -float(contact.dist))
    return deepest


def _slide(model, data, *, distance: float, roof_geom: str) -> dict[str, Any]:
    steps = math.ceil((1.3 * distance / SPEED_M_S + 1.5) / model.opt.timestep)
    roof_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, roof_geom)
    if roof_id < 0:
        raise ValueError(f"profile lacks roof geom {roof_geom}")
    roof_hits = max_contacts = 0
    max_penetration = max_roof_penetration = 0.0
    finite = True
    for _ in range(steps):
        target = min(distance, SPEED_M_S * data.time)
        velocity = np.clip(4.0 * (target - data.qpos[0]) + SPEED_M_S, -SPEED_M_S, SPEED_M_S)
        data.qfrc_applied[0] = np.clip(160.0 * (velocity - data.qvel[0]), -100.0, 100.0)
        mujoco.mj_step(model, data)
        pairs, penetration, contact_finite = _contact_sample(model, data)
        roof_hits += _roof_contacts(pairs, roof_geom)
        max_penetration = max(max_penetration, penetration)
        max_roof_penetration = max(max_roof_penetration, _roof_penetration(model, data, roof_id))
        max_contacts = max(max_contacts, int(data.ncon))
        finite = contact_finite and bool(
            np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        )
        if not finite or _warning_counts(data):
            break
    return {
        "travelled_m": float(data.qpos[0]),
        "roof_contact_steps": roof_hits,
        # Terrain contact is reported; only roof penetration judges the roof.
        "max_penetration_m": max_penetration,
        "max_roof_penetration_m": max_roof_penetration,
        "max_contacts": max_contacts,
        "warnings": _warning_counts(data),
        "finite": finite,
        "final_qpos": [float(v) for v in data.qpos],
    }


def check_roof(
    asset: FieldAsset, roof: dict[str, Any], profile: str, field=None
) -> list[dict[str, Any]]:
    field = load_heightfield(asset) if field is None else field
    centre, unit, half_along = _roof_frame(roof)
    roof_geom = f"rmuc_tunnel_roof_{roof['tunnel_id']}"
    clearance = roof["clearance_m"]
    rows = []
    # Start with the probe's front LEAD_M before the near span edge and stop
    # LEAD_M past the far edge, in both directions.  Terrain outside the roof
    # may stop a sliding probe; only roof contact decides these checks.
    offset = half_along + LEAD_M + PROBE_LENGTH_M / 2
    distance = 2 * offset
    clear_under = 2 * half_along + LEAD_M + PROBE_LENGTH_M
    for kind, height in (
        ("pass_under", clearance[0] - PROBE_MARGIN_M),
        ("blocked_above", clearance[2] + PROBE_MARGIN_M),
    ):
        runs = []
        for sign in (1.0, -1.0):
            direction = sign * unit
            model, data = _block_model(
                asset, field, profile, centre - offset * direction, direction, height
            )
            result = _slide(model, data, distance=distance, roof_geom=roof_geom)
            runs.append(
                {
                    "direction_xy": [float(v) for v in direction],
                    "cleared_roof": result["travelled_m"] >= clear_under,
                    "reached_roof": result["roof_contact_steps"] > 0,
                    **result,
                }
            )
        sound = all(
            run["finite"]
            and not run["warnings"]
            and run["max_roof_penetration_m"] <= MAX_PENETRATION_M
            for run in runs
        )
        if kind == "pass_under":
            passed = any(run["cleared_roof"] for run in runs) and not any(
                run["reached_roof"] for run in runs
            )
        else:
            passed = any(run["reached_roof"] for run in runs) and not any(
                run["cleared_roof"] for run in runs
            )
        rows.append(
            {
                "tunnel_id": roof["tunnel_id"],
                "profile": profile,
                "probe": kind,
                "probe_height_m": round(height, 4),
                "path_length_m": round(distance, 4),
                "directions": runs,
                "final_qpos": [v for run in runs for v in run["final_qpos"]],
                "status": "PASS" if passed and sound else "FAIL",
            }
        )
    top_centre = 0.5 * (roof["top_m"][0] + roof["top_m"][1])
    model, data = _ball_model(asset, profile, centre, top_centre + BALL_RADIUS_M + 0.05)
    roof_hits = 0
    for _ in range(math.ceil(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        pairs, _penetration, _finite = _contact_sample(model, data)
        roof_hits += _roof_contacts(pairs, roof_geom)
    rest_error = float(data.qpos[0]) - (top_centre + BALL_RADIUS_M)
    passed = (
        abs(rest_error) <= SUPPORT_TOLERANCE_M
        and roof_hits > 0
        and not _warning_counts(data)
        and abs(float(data.qvel[0])) < 0.01
    )
    rows.append(
        {
            "tunnel_id": roof["tunnel_id"],
            "profile": profile,
            "probe": "rests_on_roof",
            "rest_height_error_m": rest_error,
            "roof_contact_steps": roof_hits,
            "final_qpos": [float(v) for v in data.qpos],
            "warnings": _warning_counts(data),
            "status": "PASS" if passed else "FAIL",
        }
    )
    return rows


def run_tunnel_checks(
    pack: str | Path | FieldAsset, *, profiles: Sequence[str] | None = None
) -> dict[str, Any]:
    asset = pack if isinstance(pack, FieldAsset) else FieldAsset.open(pack, verify=True)
    layer = asset.collision.get("tunnel_contact_layer")
    if layer is None:
        raise ValueError("pack has no tunnel layer; build one with tunnel-pack")
    selected = tuple(profiles) if profiles is not None else asset.available_runtime_profiles
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("profiles must be nonempty and unique")
    field = load_heightfield(asset)
    trials, parity = [], []
    for roof in layer["roofs"]:
        reference = None
        for profile in selected:
            rows = check_roof(asset, roof, profile, field)
            trials.extend(rows)
            states = np.concatenate([np.asarray(row["final_qpos"]) for row in rows])
            if reference is None:
                reference = states
            else:
                difference = float(np.max(np.abs(states - reference)))
                parity.append(
                    {
                        "tunnel_id": roof["tunnel_id"],
                        "profile": profile,
                        "max_final_state_difference": difference,
                        "status": "PASS" if difference <= 1e-9 else "FAIL",
                    }
                )
    passed = all(row["status"] == "PASS" for row in trials + parity)
    return {
        "status": "PASS" if passed else "FAIL",
        "season": asset.season.id,
        "source_manifest_sha256": asset.manifest_sha256,
        "roofs": [
            {k: roof[k] for k in ("tunnel_id", "travel_axis", "clearance_m", "top_m", "floor_m")}
            for roof in layer["roofs"]
        ],
        "trials": trials,
        "profile_parity": parity,
        "summary": {
            "roofs": len(layer["roofs"]),
            "trials": len(trials),
            "passed": sum(row["status"] == "PASS" for row in trials),
            "failed": sum(row["status"] != "PASS" for row in trials),
        },
        "mujoco_version": mujoco.__version__,
        "geometry_modified": False,
        "scope": (
            "Rigid probes check the tunnel collision layer: passage below the roof, "
            "blocking above it and support on it. Robots and gaits are not evaluated."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rmuc_mujoco.tunnel_check")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_tunnel_checks(args.pack, profiles=args.profiles)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["summary"]}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["check_roof", "run_tunnel_checks"]
