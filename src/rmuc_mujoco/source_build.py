"""Access to a retained local source build for source-bound contact checks."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .manifest import sha256_file
from .seasons import FieldSource, source_for_step


def verified_source_glb(
    source_manifest_path: str | Path, source: FieldSource | None = None
) -> tuple[Path, float, str]:
    """Return the audited unsimplified GLB, its ground shift and manifest hash.

    ``source`` is the STEP release the build has to be a conversion of, normally
    the field pack's own; without it any registered release is accepted.  Each
    release pins the GLB the builder writes for it, so a check that reads parts
    by index reads the parts of the release it was written for.
    """

    manifest_path = Path(source_manifest_path).expanduser().resolve()
    root = manifest_path.parent
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        document.get("artifact_type") != "rmuc2026_official_field_mujoco_asset"
        or document.get("status") != "PASS"
    ):
        raise ValueError("source build is not a passing official field")
    step = document.get("source") or {}
    registered = source_for_step(step.get("sha256"), step.get("size_bytes"))
    if registered is None:
        raise ValueError("source build is not a registered official STEP release")
    if source is not None and registered != source:
        raise ValueError(f"source build is not {source.season_id} {source.version}")
    try:
        record = document["conversion"]["colored_intermediate_glb"]
        relative = Path(str(record["file"]))
        shift = -float(document["conversion"]["main_floor_height_before_shift_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"source build has no usable conversion record: {exc}") from exc
    if record.get("sha256") != registered.source_glb_sha256:
        raise ValueError("source GLB is not the audited unsimplified official geometry")
    glb = (root / relative).resolve()
    if relative.is_absolute() or root not in glb.parents or not glb.is_file():
        raise ValueError("source GLB is missing or outside the build")
    if sha256_file(glb) != registered.source_glb_sha256:
        raise ValueError("source GLB SHA-256 mismatch")
    if not math.isfinite(shift):
        raise ValueError("source ground shift is non-finite")
    return glb, shift, sha256_file(manifest_path)
