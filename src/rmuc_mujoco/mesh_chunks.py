"""Split large OBJ meshes into pieces MuJoCo draws quickly.

MuJoCo draws one mesh of several hundred thousand faces far slower than the
same triangles in pieces: the RMUC 2027 pro profile rendered at 104 fps with
its 825k-face group whole and at 1289 fps in pieces of at most 200k faces, and
the livery at 190 and 1282 fps.  Pieces keep every vertex line, texture
coordinate and face of the source.

Pieces are compact: the faces are halved at the median along their longest
axis until each half fits.  MuJoCo bounds each geom by a sphere when it sets
the scene extent, which sizes the shadow map.  Runs in file order spread over
the field and enlarged the RMUC 2026 extent from 43.2 to 47.1 m; slabs along
one axis still enlarged its pro extent to 48.4 m.
"""

from __future__ import annotations

from pathlib import Path


MESH_CHUNK_FACES = 150_000


class MeshChunkError(ValueError):
    """Raised when an OBJ cannot be split without changing it."""


def obj_face_count(path: Path) -> int:
    with Path(path).open("rb") as stream:
        return sum(line.startswith(b"f ") for line in stream)


def split_obj(path: Path, limit: int) -> list[tuple[str, int]]:
    """OBJ texts of compact pieces of at most ``limit`` faces.

    Each piece keeps its faces in file order.  Faces may reference texture
    coordinates (``v/vt``).  Median halving keeps every piece above half the
    limit, well clear of MuJoCo's four-vertex minimum.
    """

    vertices: list[str] = []
    texcoords: list[str] = []
    faces: list[list[tuple[int, ...]]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("vn "):
            raise MeshChunkError(f"cannot split {Path(path).name}: it has vertex normals")
        if line.startswith("v "):
            vertices.append(line)
        elif line.startswith("vt "):
            texcoords.append(line)
        elif line.startswith("f "):
            corners = [tuple(int(part) for part in token.split("/")) for token in line.split()[1:]]
            if any(len(corner) != (2 if texcoords else 1) for corner in corners):
                raise MeshChunkError(f"cannot split {Path(path).name}: unsupported face {line!r}")
            faces.append(corners)
    points = [[float(value) for value in line.split()[1:4]] for line in vertices]
    centroids = [
        [sum(points[corner[0] - 1][axis] for corner in face) / len(face) for axis in range(3)]
        for face in faces
    ]
    pieces = []
    for group in _halve(centroids, list(range(len(faces))), limit):
        run = [faces[index] for index in sorted(group)]
        used = sorted({corner[0] for face in run for corner in face})
        new_vertex = {old: new for new, old in enumerate(used, start=1)}
        lines = [vertices[index - 1] for index in used]
        if texcoords:
            used_uv = sorted({corner[1] for face in run for corner in face})
            new_uv = {old: new for new, old in enumerate(used_uv, start=1)}
            lines += [texcoords[index - 1] for index in used_uv]
            lines += [
                "f " + " ".join(f"{new_vertex[v]}/{new_uv[t]}" for v, t in face) for face in run
            ]
        else:
            lines += ["f " + " ".join(str(new_vertex[v]) for (v,) in face) for face in run]
        pieces.append(("\n".join(lines) + "\n", len(run)))
    return pieces


def _halve(centroids: list[list[float]], group: list[int], limit: int) -> list[list[int]]:
    """Split face indices at the median of their longest centroid axis until each fits."""

    if len(group) <= limit:
        return [group]
    spans = [
        max(centroids[i][axis] for i in group) - min(centroids[i][axis] for i in group)
        for axis in range(3)
    ]
    axis = spans.index(max(spans))
    ordered = sorted(group, key=lambda index: (centroids[index][axis], index))
    middle = len(ordered) // 2
    return _halve(centroids, ordered[:middle], limit) + _halve(centroids, ordered[middle:], limit)


__all__ = ["MESH_CHUNK_FACES", "MeshChunkError", "obj_face_count", "split_obj"]
