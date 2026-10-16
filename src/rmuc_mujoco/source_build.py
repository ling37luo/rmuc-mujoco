"""Access to a retained local source build for source-bound contact checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .collision_candidate import SOURCE_GLB_SHA256
from .download import OFFICIAL_STEP_SHA256
from .manifest import sha256_file


def verified_source_glb(source_manifest_path: Path) -> tuple[Path, float, str]:
    """Return the audited unsimplified GLB, its ground shift and manifest hash."""

    root = source_manifest_path.expanduser().resolve().parent
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if (
        source.get("artifact_type") != "rmuc2026_official_field_mujoco_asset"
        or source.get("status") != "PASS"
        or source["source"]["sha256"] != OFFICIAL_STEP_SHA256
    ):
        raise ValueError("source build is not the pinned passing official field")
    record = source["conversion"]["colored_intermediate_glb"]
    if record["sha256"] != SOURCE_GLB_SHA256:
        raise ValueError("source GLB is not the audited unsimplified official geometry")
    relative = Path(str(record["file"]))
    path = (root / relative).resolve()
    if relative.is_absolute() or root not in path.parents or not path.is_file():
        raise ValueError("source GLB is missing or outside the build")
    if sha256_file(path) != SOURCE_GLB_SHA256:
        raise ValueError("source GLB SHA-256 mismatch")
    shift = -float(source["conversion"]["main_floor_height_before_shift_m"])
    if not np.isfinite(shift):
        raise ValueError("source ground shift is non-finite")
    return path, shift, sha256_file(source_manifest_path)
