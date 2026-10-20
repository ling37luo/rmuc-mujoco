"""Add a maximum-fidelity visual profile from a less simplified source build.

``pro`` shares every collision asset, world geom and display colour with
``full``; only the CAD meshes change.  They come from a source build of the
same STEP release made with a larger visual face budget; a budget above the
drawing's face count keeps the whole official tessellation.  Large groups are
split into compact pieces, each its own mesh asset and geom, and pro keeps
full's scene centre and extent.  Apply it after the other layers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco

from .lite_profile import _mesh_file_record
from .manifest import FieldAsset, sha256_file
from .mesh_chunks import MESH_CHUNK_FACES, MeshChunkError, obj_face_count, split_obj
from .pack import EXPECTED_ARTIFACT_TYPE, ExportBlocked, _source_file


PRO_XML = "rmuc2026_field_pro.xml"
PRO_METHOD = "official_source_build_at_larger_visual_budget_v1"
PRO_CHUNK_FACES = MESH_CHUNK_FACES


def _source_build_contract(build: Path, source: FieldAsset) -> dict[str, object]:
    """The source build's manifest, checked against the pack it extends."""

    try:
        manifest = json.loads((build / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportBlocked(f"source build manifest is unreadable: {exc}") from exc
    if manifest.get("artifact_type") != EXPECTED_ARTIFACT_TYPE or manifest.get("status") != "PASS":
        raise ExportBlocked("source build is not a passing official field build")
    step = (manifest.get("source") or {}).get("sha256")
    if step != source.manifest["source_identity"].get("official_step_sha256"):
        raise ExportBlocked("source build and pack come from different STEP files")
    # Both builds translate their meshes by the spawn the heightfield fixes.
    spawn = source.manifest["coordinate_frame"]["recommended_spawn"]
    build_spawn = manifest.get("recommended_spawn") or {}
    for key in ("x_before_translation_m", "y_before_translation_m", "terrain_height_m"):
        if not math.isclose(
            float(build_spawn.get(key, math.nan)), float(spawn[key]), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ExportBlocked(f"source build world frame differs from the pack ({key})")
    visuals = manifest.get("visual_meshes")
    if not isinstance(visuals, list) or len(visuals) != len(source.visual_meshes):
        raise ExportBlocked("source build has a different number of CAD visual meshes")
    for index, (full, pro) in enumerate(zip(source.visual_meshes, visuals)):
        if (
            not isinstance(pro, dict)
            or Path(str(pro.get("file"))).name != Path(str(full["file"])).name
            or pro.get("material_id") != full.get("material_id")
            or pro.get("rgba") != full.get("rgba")
            or pro.get("visual_role", "cad_structure") != full.get("visual_role", "cad_structure")
        ):
            raise ExportBlocked(f"source build visual mesh {index} does not match {full['file']}")
    return manifest


def export_pro_pack(
    source_build: Path,
    source_pack: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create a new schema-4 pack whose ``pro`` profile shows the build's CAD meshes.

    ``source_build`` is a retained source build of the pack's STEP release
    with a larger ``--target-visual-faces``.  The pack and the build are
    verified first; the build's meshes are copied into the new pack and bound
    by its manifest.
    """

    source = FieldAsset.open(source_pack, verify=True)
    if int(source.manifest["schema_version"]) < 2:
        raise ExportBlocked("pro-pack requires a runtime pack with named lighting (schema 2+)")
    if "pro" in source.runtime_profiles:
        raise ExportBlocked("source pack already contains pro")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ExportBlocked(f"output already exists: {output}")
    if output == source.root or source.root in output.parents:
        raise ExportBlocked("output cannot be inside the source pack")
    build = Path(source_build).expanduser().resolve()
    build_manifest = _source_build_contract(build, source)

    full_visuals = list(source.visual_meshes)
    pro_sources = [
        _source_file(build, record.get("file"), record.get("sha256"), suffix=".obj")
        for record in build_manifest["visual_meshes"]
    ]
    full_faces = [obj_face_count(source.root / str(record["file"])) for record in full_visuals]
    pro_faces = [obj_face_count(path) for _relative, path, _digest in pro_sources]
    if any(count == 0 for count in pro_faces):
        raise ExportBlocked("source build contains an empty visual mesh")
    if sum(pro_faces) <= sum(full_faces):
        raise ExportBlocked(
            f"source build has {sum(pro_faces)} visual faces, no more than full's {sum(full_faces)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for record in source.manifest["contents"]["files"]:
            relative = str(record["file"])
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.root / relative, destination)
        manifest = json.loads((source.root / "manifest.json").read_text(encoding="utf-8"))
        pro_records: list[dict[str, object]] = []
        file_records: list[dict[str, object]] = []
        replacements: dict[str, list[str]] = {}
        (staging / "visual_pro").mkdir(exist_ok=True)
        for full_record, (_relative, path, _digest), source_faces, faces in zip(
            full_visuals, pro_sources, full_faces, pro_faces
        ):
            full_relative = str(full_record["file"])
            stem = Path(full_relative).stem
            if faces <= PRO_CHUNK_FACES:
                pieces = [(None, faces)]
                names = [f"visual_pro/{stem}.obj"]
                shutil.copyfile(path, staging / names[0])
            else:
                try:
                    pieces = split_obj(path, PRO_CHUNK_FACES)
                except MeshChunkError as exc:
                    raise ExportBlocked(str(exc)) from exc
                names = [f"visual_pro/{stem}__chunk{index}.obj" for index in range(len(pieces))]
                for name, (text, _count) in zip(names, pieces):
                    (staging / name).write_text(text, encoding="utf-8")
            chunks = []
            for name, (_text, count) in zip(names, pieces):
                file_record = _mesh_file_record(staging, name, "cad_visual_mesh_pro")
                file_records.append(file_record)
                chunks.append({"file": name, "sha256": file_record["sha256"], "faces": count})
            replacements[full_relative] = names
            pro_records.append(
                {
                    "source_file": full_relative,
                    "source_faces": source_faces,
                    "faces": faces,
                    "chunks": chunks,
                }
            )

        # A split group gets one mesh asset and one geom per chunk; the first
        # keeps full's names, the others add "__chunk<n>".
        root = ET.parse(source.entrypoint_for("full")).getroot()
        parents = {child: parent for parent in root.iter() for child in parent}
        geoms_by_mesh = {geom.get("mesh"): geom for geom in root.iter("geom") if geom.get("mesh")}
        changed = 0
        for mesh_asset in root.findall("./asset/mesh"):
            names = replacements.get(mesh_asset.get("file"))
            if names is None:
                continue
            changed += 1
            mesh_asset.set("file", names[0])
            mesh_name = str(mesh_asset.get("name"))
            geom = geoms_by_mesh[mesh_name]
            for index, name in reversed(list(enumerate(names[1:], start=1))):
                extra_mesh = ET.Element("mesh", dict(mesh_asset.attrib))
                extra_mesh.set("name", f"{mesh_name}__chunk{index}")
                extra_mesh.set("file", name)
                asset_parent = parents[mesh_asset]
                asset_parent.insert(list(asset_parent).index(mesh_asset) + 1, extra_mesh)
                extra_geom = ET.Element("geom", dict(geom.attrib))
                extra_geom.set("name", f"{geom.get('name')}__chunk{index}")
                extra_geom.set("mesh", f"{mesh_name}__chunk{index}")
                geom_parent = parents[geom]
                geom_parent.insert(list(geom_parent).index(geom) + 1, extra_geom)
        if changed != len(full_visuals):
            raise ExportBlocked("full MJCF visual meshes disagree with the source manifest")
        # MuJoCo sets the scene centre and extent, which place the shadow map
        # and the free camera, from each geom's bounding sphere.  Chunks of
        # groups with long triangles keep large spheres off centre (RMUC 2026:
        # extent 43.2 to 47.9 m), so pro states full's values.
        full_model = mujoco.MjModel.from_xml_path(str(source.entrypoint_for("full")))
        statistic = ET.Element(
            "statistic",
            {
                "center": " ".join(f"{value:.17g}" for value in full_model.stat.center),
                "extent": f"{full_model.stat.extent:.17g}",
            },
        )
        root.insert(list(root).index(root.find("./asset")), statistic)
        ET.indent(root, space="  ")
        (staging / PRO_XML).write_bytes(
            ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
        )
        manifest["schema_version"] = 4
        manifest["visual_pro"] = {
            "source_profile": "full",
            "method": PRO_METHOD,
            "source_build": {
                "manifest_sha256": sha256_file(build / "manifest.json"),
                "target_visual_faces": int(build_manifest["conversion"]["target_visual_faces"]),
            },
            "source_visual_faces": sum(full_faces),
            "output_visual_faces": sum(pro_faces),
            "max_chunk_faces": PRO_CHUNK_FACES,
            "physics_changed": False,
            "meshes": pro_records,
        }
        manifest["runtime_profiles"]["profiles"]["pro"] = {
            "entrypoint": PRO_XML,
            "includes_visual_meshes": True,
            "visual_mesh_count": sum(len(record["chunks"]) for record in pro_records),
            "intended_use": "maximum_fidelity_visualization",
        }
        file_records.insert(0, _mesh_file_record(staging, PRO_XML, "field_mjcf_pro"))
        manifest["contents"]["files"].extend(file_records)
        manifest["contents"]["file_count_excluding_manifest"] = len(manifest["contents"]["files"])
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        FieldAsset.open(staging, verify=True)
        staging.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["PRO_XML", "export_pro_pack"]
