"""Splitting OBJ meshes into pieces keeps every vertex, texture coordinate and face."""

from __future__ import annotations

from pathlib import Path

import pytest

from rmuc_mujoco.mesh_chunks import MeshChunkError, obj_face_count, split_obj


def _grid_obj(path: Path, cells: int, *, uv: bool) -> Path:
    side = cells + 1
    lines = ["o grid"]
    lines += [f"v {x} {y} 0" for y in range(side) for x in range(side)]
    if uv:
        lines += [f"vt {x / cells} {y / cells}" for y in range(side) for x in range(side)]
    for y in range(cells):
        for x in range(cells):
            a = y * side + x + 1
            for triangle in ((a, a + 1, a + side + 1), (a, a + side + 1, a + side)):
                lines.append("f " + " ".join(f"{i}/{i}" if uv else str(i) for i in triangle))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def triangles(text: str) -> list[tuple[object, ...]]:
    """Each face as its corners' vertex and texture-coordinate lines."""

    rows = text.splitlines()
    vertices = [row for row in rows if row.startswith("v ")]
    texcoords = [row for row in rows if row.startswith("vt ")]
    faces = []
    for row in rows:
        if row.startswith("f "):
            corners = [[int(part) for part in token.split("/")] for token in row.split()[1:]]
            faces.append(
                tuple(
                    (vertices[c[0] - 1], texcoords[c[1] - 1] if len(c) > 1 else None)
                    for c in corners
                )
            )
    return faces


@pytest.mark.parametrize("uv", [False, True])
def test_pieces_are_compact_and_keep_every_face(tmp_path: Path, uv: bool) -> None:
    path = _grid_obj(tmp_path / "grid.obj", 10, uv=uv)
    whole = triangles(path.read_text(encoding="ascii"))

    pieces = split_obj(path, 60)

    assert obj_face_count(path) == 200
    assert [faces for _text, faces in pieces] == [50, 50, 50, 50]
    split = [triangles(text) for text, _faces in pieces]
    assert sorted(face for piece in split for face in piece) == sorted(whole)
    # Each piece keeps file order, and the pieces are the grid's four quarters.
    boxes = []
    for piece in split:
        positions = [whole.index(face) for face in piece]
        assert positions == sorted(positions)
        centres = [
            [sum(float(corner[0].split()[axis]) for corner in face) / 3 for axis in (1, 2)]
            for face in piece
        ]
        boxes.append([(min(c[a] for c in centres), max(c[a] for c in centres)) for a in (0, 1)])
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            assert any(box[a][1] <= other[a][0] or other[a][1] <= box[a][0] for a in (0, 1))


def test_a_mesh_within_the_limit_is_one_piece(tmp_path: Path) -> None:
    path = _grid_obj(tmp_path / "grid.obj", 3, uv=True)

    [(text, faces)] = split_obj(path, 100)

    assert faces == 18 and triangles(text) == triangles(path.read_text(encoding="ascii"))


def test_a_mesh_with_normals_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "normals.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n", encoding="ascii")

    with pytest.raises(MeshChunkError, match="normals"):
        split_obj(path, 1)
