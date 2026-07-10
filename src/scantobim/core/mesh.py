"""Triangle mesh container, polygon triangulation and mesh assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """Indexed triangle mesh.

    Attributes
    ----------
    vertices:
        ``(V, 3)`` float64.
    faces:
        ``(F, 3)`` int64 vertex indices, CCW seen from the face normal side.
    vertex_colors:
        Optional ``(V, 3)`` uint8.
    face_groups:
        Optional ``(F,)`` int — id of the source surface (plane) per face,
        exported as OBJ groups so CAD/BIM tools can address single surfaces.
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None = None
    face_groups: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64).reshape(-1, 3)

    def face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, lengths, out=np.zeros_like(n), where=lengths > 0)

    def vertex_normals(self) -> np.ndarray:
        """Area-weighted vertex normals."""
        tri = self.vertices[self.faces]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
        vn = np.zeros_like(self.vertices)
        for c in range(3):
            np.add.at(vn, self.faces[:, c], fn)
        lengths = np.linalg.norm(vn, axis=1, keepdims=True)
        return np.divide(vn, lengths, out=np.zeros_like(vn), where=lengths > 0)

    def area(self) -> float:
        tri = self.vertices[self.faces]
        return float(
            0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        )

    def edge_stats(self) -> dict:
        """Boundary/manifold statistics for the quality report."""
        edges = np.vstack(
            [self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]
        )
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return {
            "edges": int(len(counts)),
            "boundary_edges": int(np.sum(counts == 1)),
            "manifold_edges": int(np.sum(counts == 2)),
            "non_manifold_edges": int(np.sum(counts > 2)),
        }


def triangulate_polygon(poly: np.ndarray) -> np.ndarray:
    """Ear-clipping triangulation of a simple 2D polygon (CCW).

    Returns ``(T, 3)`` indices into ``poly``. Robust to collinear vertices;
    falls back to a fan if clipping stalls on degenerate input.
    """
    n = len(poly)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)
    if n == 3:
        return np.array([[0, 1, 2]], dtype=np.int64)

    indices = list(range(n))
    triangles: list[list[int]] = []
    guard = 0
    while len(indices) > 3 and guard < 4 * n * n:
        guard += 1
        m = len(indices)
        ear_found = False
        for pos in range(m):
            i_prev, i_cur, i_next = (
                indices[pos - 1],
                indices[pos],
                indices[(pos + 1) % m],
            )
            a, b, c = poly[i_prev], poly[i_cur], poly[i_next]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-14:  # reflex or degenerate
                continue
            # No other polygon vertex inside the candidate ear.
            others = [k for k in indices if k not in (i_prev, i_cur, i_next)]
            if others and _any_point_in_triangle(poly[others], a, b, c):
                continue
            triangles.append([i_prev, i_cur, i_next])
            indices.pop(pos)
            ear_found = True
            break
        if not ear_found:
            # Degenerate remainder (all collinear or numerically stuck):
            # drop the most collinear vertex and continue.
            best_pos, best_val = 0, np.inf
            for pos in range(len(indices)):
                a = poly[indices[pos - 1]]
                b = poly[indices[pos]]
                c = poly[indices[(pos + 1) % len(indices)]]
                val = abs(
                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                )
                if val < best_val:
                    best_val, best_pos = val, pos
            indices.pop(best_pos)
    if len(indices) == 3:
        triangles.append(list(indices))
    return np.array(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), dtype=np.int64)


def _any_point_in_triangle(pts: np.ndarray, a, b, c) -> bool:
    v0, v1 = c - a, b - a
    v2 = pts - a
    dot00 = v0 @ v0
    dot01 = v0 @ v1
    dot11 = v1 @ v1
    dot02 = v2 @ v0
    dot12 = v2 @ v1
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-18:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    # Inclusive: a vertex exactly on the ear boundary blocks the ear too —
    # clipping across it would create a triangle crossing the polygon edge
    # that continues from that vertex (classic concave-polygon failure).
    return bool(np.any((u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)))


def merge_meshes(parts: list[Mesh]) -> Mesh:
    """Concatenate meshes, preserving per-part colors/groups."""
    if not parts:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    vertices = []
    faces = []
    colors = []
    groups = []
    offset = 0
    has_colors = all(p.vertex_colors is not None for p in parts)
    for p in parts:
        vertices.append(p.vertices)
        faces.append(p.faces + offset)
        if has_colors:
            colors.append(p.vertex_colors)
        if p.face_groups is not None:
            groups.append(p.face_groups)
        else:
            groups.append(np.zeros(len(p.faces), dtype=np.int64))
        offset += len(p.vertices)
    return Mesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(faces),
        vertex_colors=np.vstack(colors) if has_colors else None,
        face_groups=np.concatenate(groups),
    )


def weld_vertices(mesh: Mesh, tolerance: float) -> Mesh:
    """Merge vertices closer than ``tolerance``.

    After corner snapping, the polygons of adjacent planes carry vertices at
    *identical* corner coordinates; welding stitches them into shared mesh
    vertices so edges become true creases instead of cracks.
    """
    if len(mesh.vertices) == 0 or tolerance <= 0:
        return mesh
    quant = np.round(mesh.vertices / tolerance).astype(np.int64)
    _, first_idx, inverse = np.unique(
        quant, axis=0, return_index=True, return_inverse=True
    )
    new_vertices = mesh.vertices[first_idx]
    new_colors = mesh.vertex_colors[first_idx] if mesh.vertex_colors is not None else None
    new_faces = inverse[mesh.faces]
    # Drop faces that collapsed.
    ok = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    return Mesh(
        vertices=new_vertices,
        faces=new_faces[ok],
        vertex_colors=new_colors,
        face_groups=None if mesh.face_groups is None else mesh.face_groups[ok],
    )
