# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Validated mesh readers used by the public CAD importer."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

MeshFormat = Literal["stl", "obj", "3mf"]


class MeshData(TypedDict):
    """Triangulated mesh returned by mesh-based CAD imports."""

    vertices: NDArray[np.float64]
    faces: NDArray[np.int32]
    normals: NDArray[np.float64]
    format: MeshFormat


_STL_TRIANGLE_DTYPE: np.dtype[Any] = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)


def _face_normals(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> NDArray[np.float64]:
    edges_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edges_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    normals = np.cross(edges_a, edges_b)
    lengths = np.linalg.norm(normals, axis=1)
    return np.divide(
        normals,
        lengths[:, None],
        out=np.zeros_like(normals),
        where=lengths[:, None] > 0.0,
    )


def _validated_mesh(
    vertices: object,
    faces: object,
    mesh_format: MeshFormat,
    normals: object | None = None,
) -> MeshData:
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array_64 = np.asarray(faces, dtype=np.int64)

    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,):
        raise ValueError("Mesh vertices must have shape (n, 3).")
    if face_array_64.ndim != 2 or face_array_64.shape[1:] != (3,):
        raise ValueError("Mesh faces must have shape (n, 3).")
    if not len(vertex_array) or not len(face_array_64):
        raise ValueError("The mesh does not contain any triangles.")
    if not np.isfinite(vertex_array).all():
        raise ValueError("Mesh vertices contain non-finite coordinates.")
    if face_array_64.min() < 0 or face_array_64.max() >= len(vertex_array):
        raise ValueError("A mesh face references a vertex outside the file.")
    if face_array_64.max() > np.iinfo(np.int32).max:
        raise ValueError("The mesh has too many vertices for 32-bit face indices.")

    face_array = face_array_64.astype(np.int32, copy=False)
    if normals is None:
        normal_array = _face_normals(vertex_array, face_array)
    else:
        normal_array = np.asarray(normals, dtype=np.float64)
        if normal_array.shape != (len(face_array), 3):
            raise ValueError("Mesh face normals must have shape (n_faces, 3).")
        if not np.isfinite(normal_array).all():
            raise ValueError("Mesh normals contain non-finite values.")

    return {
        "vertices": vertex_array,
        "faces": face_array,
        "normals": normal_array,
        "format": mesh_format,
    }


def load_stl(path: Path) -> MeshData:
    """Read a binary or UTF-8 ASCII STL file."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        prefix = handle.read(84)

    if len(prefix) >= 84:
        triangle_count = struct.unpack_from("<I", prefix, 80)[0]
        expected_size = 84 + triangle_count * _STL_TRIANGLE_DTYPE.itemsize
        if expected_size == file_size:
            return _load_binary_stl(path, triangle_count)

    return _load_ascii_stl(path)


def _load_binary_stl(path: Path, triangle_count: int) -> MeshData:
    if not triangle_count:
        raise ValueError(f"{path.name} does not contain any STL triangles.")
    if triangle_count > np.iinfo(np.int32).max // 3:
        raise ValueError(f"{path.name} contains too many STL triangles.")

    with path.open("rb") as handle:
        handle.seek(84)
        records = np.fromfile(
            handle,
            dtype=_STL_TRIANGLE_DTYPE,
            count=triangle_count,
        )
    if len(records) != triangle_count:
        raise ValueError(f"{path.name} is a truncated binary STL file.")

    vertices = records["vertices"].astype(np.float64).reshape((-1, 3))
    faces: NDArray[np.int32] = np.arange(
        triangle_count * 3,
        dtype=np.int32,
    ).reshape((-1, 3))
    normals = records["normal"].astype(np.float64)
    return _validated_mesh(vertices, faces, "stl", normals)


def _load_ascii_stl(path: Path) -> MeshData:
    vertices: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float] | None] = []
    facet_vertices: list[tuple[float, float, float]] = []
    current_normal: tuple[float, float, float] | None = None
    inside_facet = False

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path.name} is neither a valid binary nor UTF-8 ASCII STL file."
        ) from exc

    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().split()
        if not parts:
            continue
        keyword = parts[0].lower()
        try:
            if keyword == "facet":
                if len(parts) != 5 or parts[1].lower() != "normal":
                    raise ValueError("expected 'facet normal x y z'")
                if inside_facet:
                    raise ValueError("started a facet before ending the previous one")
                inside_facet = True
                current_normal = (
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            elif keyword == "vertex":
                if not inside_facet:
                    raise ValueError("found a vertex outside an STL facet")
                if len(parts) != 4:
                    raise ValueError("expected 'vertex x y z'")
                facet_vertices.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]))
                )
                if len(facet_vertices) > 3:
                    raise ValueError("a facet contains more than three vertices")
            elif keyword == "endfacet":
                if not inside_facet:
                    raise ValueError("found 'endfacet' without a matching facet")
                if len(facet_vertices) != 3:
                    raise ValueError("a facet must contain exactly three vertices")
                vertices.extend(facet_vertices)
                normals.append(current_normal)
                facet_vertices = []
                current_normal = None
                inside_facet = False
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc

    if inside_facet:
        raise ValueError(f"{path.name} ends before its final STL facet is complete.")
    if not vertices:
        raise ValueError(f"{path.name} does not contain any STL triangles.")

    face_count = len(vertices) // 3
    faces: NDArray[np.int32] = np.arange(
        face_count * 3,
        dtype=np.int32,
    ).reshape((-1, 3))
    stored_normals = normals if all(normal is not None for normal in normals) else None
    return _validated_mesh(vertices, faces, "stl", stored_normals)


def _obj_vertex_index(token: str, current_vertex_count: int) -> int:
    raw_index = token.split("/", maxsplit=1)[0]
    if not raw_index:
        raise ValueError("a face has an empty vertex index")
    index = int(raw_index)
    if index == 0:
        raise ValueError("OBJ vertex indices cannot be zero")
    return index - 1 if index > 0 else current_vertex_count + index


def load_obj(path: Path) -> MeshData:
    """Read OBJ vertex positions and triangulate polygon faces."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path.name} is not a UTF-8 OBJ file.") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        parts = raw_line.partition("#")[0].strip().split()
        if not parts:
            continue
        try:
            if parts[0] == "v":
                if len(parts) < 4:
                    raise ValueError("a vertex requires three coordinates")
                vertices.append(
                    (float(parts[1]), float(parts[2]), float(parts[3]))
                )
            elif parts[0] == "f":
                if len(parts) < 4:
                    raise ValueError("a face requires at least three vertices")
                polygon = [
                    _obj_vertex_index(token, len(vertices)) for token in parts[1:]
                ]
                faces.extend(
                    (polygon[0], polygon[index], polygon[index + 1])
                    for index in range(1, len(polygon) - 1)
                )
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc

    return _validated_mesh(vertices, faces, "obj")


def load_3mf(path: Path) -> MeshData:
    """Read a 3MF package with trimesh and combine its transformed geometry."""
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError("trimesh is required for 3MF import.") from exc

    geometry = trimesh.load(path, file_type="3mf", force="mesh", process=False)
    if isinstance(geometry, trimesh.Scene):
        geometry = geometry.to_geometry()
    if not isinstance(geometry, trimesh.Trimesh):
        raise ValueError(f"{path.name} does not contain a triangle mesh.")
    return _validated_mesh(geometry.vertices, geometry.faces, "3mf")
