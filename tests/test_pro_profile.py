"""The pro profile: full's pack with a larger-budget build's CAD meshes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import pytest

from rmuc_mujoco import FieldAsset, ManifestError, pro_profile
from rmuc_mujoco.lite_profile import export_interactive_lite_pack
from rmuc_mujoco.pack import ExportBlocked, export_runtime_asset_pack
from rmuc_mujoco.pro_profile import PRO_XML, export_pro_pack

from test_pack_profiles import _synthetic_source_build


# Eight faces where the synthetic full meshes have four.
OCTAHEDRON = "\n".join(
    [
        "o octahedron",
        "v 1 0 0",
        "v -1 0 0",
        "v 0 1 0",
        "v 0 -1 0",
        "v 0 0 1",
        "v 0 0 -1",
        "f 1 3 5",
        "f 3 2 5",
        "f 2 4 5",
        "f 4 1 5",
        "f 3 1 6",
        "f 2 3 6",
        "f 4 2 6",
        "f 1 4 6",
        "",
    ]
)
# Two octahedra, the second 3 m higher: sixteen faces.
_VERTICES = [line for line in OCTAHEDRON.splitlines() if line.startswith("v ")]
_FACES = [line for line in OCTAHEDRON.splitlines() if line.startswith("f ")]
DOUBLE_OCTAHEDRON = "\n".join(
    [
        "o double_octahedron",
        *_VERTICES,
        *(f"v {x} {y} {float(z) + 3}" for _v, x, y, z in (line.split() for line in _VERTICES)),
        *_FACES,
        *(f"f {' '.join(str(int(i) + 6) for i in line.split()[1:])}" for line in _FACES),
        "",
    ]
)


def _rewrite_meshes(build: Path, mesh: str) -> dict:
    """Give every visual group of ``build`` the mesh ``mesh``; return its manifest."""

    manifest_path = build / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["visual_meshes"]:
        path = build / record["file"]
        path.write_text(mesh, encoding="ascii")
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _pro_build(tmp_path: Path, source: Path, *, mesh: str = OCTAHEDRON, mutate=None) -> Path:
    """The synthetic source build again, with denser meshes in every group."""

    build = tmp_path / "pro_source"
    shutil.copytree(source, build)
    manifest = _rewrite_meshes(build, mesh)
    manifest["conversion"] = {"target_visual_faces": 2_300_000}
    if mutate is not None:
        mutate(manifest)
    (build / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return build


@pytest.fixture
def full_pack(tmp_path: Path) -> tuple[Path, Path]:
    source = _synthetic_source_build(tmp_path / "source")
    pack = tmp_path / "pack"
    export_runtime_asset_pack(source, pack)
    return source, pack


def test_pro_swaps_only_the_cad_mesh_files(tmp_path: Path, full_pack) -> None:
    source, pack = full_pack

    result = export_pro_pack(_pro_build(tmp_path, source), pack, tmp_path / "pro")

    asset = FieldAsset.open(tmp_path / "pro", verify=True)
    assert asset.available_runtime_profiles == ("pro", "full", "collision_only")
    assert asset.runtime_profiles["pro"]["entrypoint"] == PRO_XML
    pro = result["visual_pro"]
    assert (pro["source_visual_faces"], pro["output_visual_faces"]) == (38 * 4, 38 * 8)
    assert pro["source_build"]["target_visual_faces"] == 2_300_000
    meshes = {mesh.get("file") for mesh in ET.parse(asset.entrypoint_for("pro")).findall(".//mesh")}
    assert meshes == {row["chunks"][0]["file"] for row in pro["meshes"]}
    assert all(name.startswith("visual_pro/") for name in meshes)
    # Collision, the default profile and full itself are untouched.
    assert asset.manifest["runtime_profiles"]["default"] == "full"
    for name in ("rmuc2026_field.xml", "rmuc2026_field_collision_only.xml"):
        assert (tmp_path / "pro" / name).read_bytes() == (pack / name).read_bytes()
    model = mujoco.MjModel.from_xml_path(str(asset.entrypoint_for("pro")))
    full = mujoco.MjModel.from_xml_path(str(asset.entrypoint_for("full")))
    assert (model.nmeshface, model.ngeom) == (full.nmeshface * 2, full.ngeom)


def test_large_groups_are_split_into_chunks_with_the_same_triangles(
    tmp_path: Path, full_pack, monkeypatch
) -> None:
    source, pack = full_pack
    monkeypatch.setattr(pro_profile, "PRO_CHUNK_FACES", 5)

    export_pro_pack(_pro_build(tmp_path, source, mesh=DOUBLE_OCTAHEDRON), pack, tmp_path / "pro")

    asset = FieldAsset.open(tmp_path / "pro", verify=True)
    row = asset.manifest["visual_pro"]["meshes"][0]
    assert [chunk["faces"] for chunk in row["chunks"]] == [4, 4, 4, 4]
    assert asset.runtime_profiles["pro"]["visual_mesh_count"] == 38 * 4
    model = mujoco.MjModel.from_xml_path(str(asset.entrypoint_for("pro")))
    assert model.nmeshface == 38 * 16
    # Chunk 1 copies chunk 0's geom apart from its name and mesh.
    root = ET.parse(asset.entrypoint_for("pro")).getroot()
    first = root.find(".//geom[@mesh='rmuc2026_visual_mesh_0']")
    extra = root.find(".//geom[@mesh='rmuc2026_visual_mesh_0__chunk1']")
    assert {**first.attrib, "name": extra.get("name"), "mesh": extra.get("mesh")} == extra.attrib
    # The chunks hold the source triangles: same vertex positions per face.
    whole = _triangles(DOUBLE_OCTAHEDRON)
    split = []
    for chunk in row["chunks"]:
        split += _triangles((tmp_path / "pro" / chunk["file"]).read_text(encoding="utf-8"))
    assert split == whole


def test_a_tampered_chunk_geom_is_refused(tmp_path: Path, full_pack, monkeypatch) -> None:
    source, pack = full_pack
    monkeypatch.setattr(pro_profile, "PRO_CHUNK_FACES", 5)
    export_pro_pack(_pro_build(tmp_path, source, mesh=DOUBLE_OCTAHEDRON), pack, tmp_path / "pro")
    xml = tmp_path / "pro" / PRO_XML
    text = xml.read_text(encoding="utf-8")
    # Give chunk 1 of the first group another group's material.
    lines = text.splitlines(keepends=True)
    [index] = [i for i, line in enumerate(lines) if 'mesh="rmuc2026_visual_mesh_0__chunk1"' in line]
    assert 'material="rmuc2026_material_0"' in lines[index]
    lines[index] = lines[index].replace("rmuc2026_material_0", "rmuc2026_material_1")
    _rewrite_bound_file(tmp_path / "pro", PRO_XML, "".join(lines))

    with pytest.raises(ManifestError, match="not a copy of its first chunk"):
        FieldAsset.open(tmp_path / "pro", verify=True)


def _triangles(obj: str) -> list[tuple[tuple[str, ...], ...]]:
    vertices = [tuple(line.split()[1:]) for line in obj.splitlines() if line.startswith("v ")]
    return [
        tuple(vertices[int(index) - 1] for index in line.split()[1:])
        for line in obj.splitlines()
        if line.startswith("f ")
    ]


def _rewrite_bound_file(pack: Path, relative: str, text: str) -> None:
    """Replace a pack file and keep its manifest hash in step."""

    (pack / relative).write_text(text, encoding="utf-8")
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["contents"]["files"]:
        if record["file"] == relative:
            record["sha256"] = hashlib.sha256((pack / relative).read_bytes()).hexdigest()
            record["size_bytes"] = (pack / relative).stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_a_lite_profile_can_follow_pro(tmp_path: Path) -> None:
    # Octahedra in full leave lite room to reduce; pro has two per group.
    source = _synthetic_source_build(tmp_path / "source")
    _rewrite_meshes(source, OCTAHEDRON)
    export_runtime_asset_pack(source, tmp_path / "pack")
    build = _pro_build(tmp_path, source, mesh=DOUBLE_OCTAHEDRON)
    export_pro_pack(build, tmp_path / "pack", tmp_path / "pro")

    export_interactive_lite_pack(tmp_path / "pro", tmp_path / "lite", target_visual_faces=200)

    asset = FieldAsset.open(tmp_path / "lite", verify=True)
    assert asset.available_runtime_profiles == ("pro", "full", "interactive_lite", "collision_only")
    assert asset.manifest["visual_pro"]["output_visual_faces"] == 38 * 16
    assert asset.manifest["visual_lite"]["output_visual_faces"] <= 200


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda m: m["visual_meshes"][3].__setitem__("material_id", "other"), "visual mesh 3"),
        (lambda m: m["visual_meshes"].pop(), "number of CAD visual meshes"),
        (lambda m: m["recommended_spawn"].__setitem__("x_before_translation_m", 0.5), "frame"),
        (lambda m: m["source"].__setitem__("sha256", "0" * 64), "different STEP"),
    ],
)
def test_a_build_that_does_not_line_up_with_the_pack_is_refused(
    tmp_path: Path, full_pack, mutate, error
) -> None:
    source, pack = full_pack

    with pytest.raises(ExportBlocked, match=error):
        export_pro_pack(_pro_build(tmp_path, source, mutate=mutate), pack, tmp_path / "pro")
    assert not (tmp_path / "pro").exists()


def test_a_build_without_more_detail_is_refused(tmp_path: Path, full_pack) -> None:
    source, pack = full_pack
    same = (source / "visual" / "mesh_00.obj").read_text(encoding="ascii")

    with pytest.raises(ExportBlocked, match="no more than full"):
        export_pro_pack(_pro_build(tmp_path, source, mesh=same), pack, tmp_path / "pro")


def test_pro_is_added_once(tmp_path: Path, full_pack) -> None:
    source, pack = full_pack
    build = _pro_build(tmp_path, source)
    export_pro_pack(build, pack, tmp_path / "pro")

    with pytest.raises(ExportBlocked, match="already contains pro"):
        export_pro_pack(build, tmp_path / "pro", tmp_path / "pro_again")


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda pro: pro.__setitem__("output_visual_faces", pro["source_visual_faces"]), "detail"),
        (lambda pro: pro.__setitem__("physics_changed", True), "physics contract"),
        (lambda pro: pro["meshes"][0].__setitem__("faces", 9), "chunk faces disagree"),
        (lambda pro: pro.__setitem__("max_chunk_faces", 0), "max_chunk_faces"),
    ],
)
def test_the_pro_manifest_contract_fails_closed(tmp_path: Path, full_pack, mutate, error) -> None:
    source, pack = full_pack
    export_pro_pack(_pro_build(tmp_path, source), pack, tmp_path / "pro")
    manifest_path = tmp_path / "pro" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest["visual_pro"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ManifestError, match=error):
        FieldAsset.open(tmp_path / "pro", verify=False)
