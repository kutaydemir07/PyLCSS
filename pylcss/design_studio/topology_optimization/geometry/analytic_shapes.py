# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Analytic signed-distance shapes used during surface recovery."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .recovery_grid import BoxRegion, CylinderRegion, _split_cylinder_region


def smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Smooth minimum of two fields a and b with blending radius k."""
    if k <= 1e-9:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * h * k / 6.0


def smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Smooth maximum of two fields a and b with blending radius k."""
    if k <= 1e-9:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * h * k / 6.0


class AnalyticShape:
    """Base class for robust, analytical geometric shapes in physical space."""

    def __init__(self, is_solid: bool = True) -> None:
        self.is_solid = is_solid

    def sdf(self, p: np.ndarray) -> np.ndarray:
        """Evaluate physical signed distance at points p (..., 3).
        Negative is inside the shape, positive is outside.
        """
        raise NotImplementedError

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        """Project physical points p (N, 3) onto the boundary if within tolerance."""
        raise NotImplementedError

    def get_normal(self, p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute normal vectors at physical coordinates p (N, 3) using central difference."""
        p_arr = np.asarray(p, dtype=float)
        grad = np.zeros_like(p_arr)
        for i in range(3):
            p_plus = p_arr.copy()
            p_plus[..., i] += eps
            p_minus = p_arr.copy()
            p_minus[..., i] -= eps
            grad[..., i] = (self.sdf(p_plus) - self.sdf(p_minus)) / (2.0 * eps)
        norm = np.linalg.norm(grad, axis=-1, keepdims=True)
        return grad / np.maximum(norm, 1e-12)


class BoxShape(AnalyticShape):
    """An analytical box in physical space, supporting arbitrary orientation."""

    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 0.0,
        y_min: float = 0.0,
        y_max: float = 0.0,
        z_min: float = 0.0,
        z_max: float = 0.0,
        center: Optional[np.ndarray] = None,
        extents: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        is_solid: bool = True,
    ) -> None:
        super().__init__(is_solid)
        if center is not None and extents is not None:
            self.center = np.asarray(center, dtype=float)
            self.extents = np.asarray(extents, dtype=float)
            self.rotation = (
                np.asarray(rotation, dtype=float) if rotation is not None else np.eye(3)
            )
        else:
            xmin, xmax = sorted((float(x_min), float(x_max)))
            ymin, ymax = sorted((float(y_min), float(y_max)))
            zmin, zmax = sorted((float(z_min), float(z_max)))
            self.center = np.array([xmin + xmax, ymin + ymax, zmin + zmax]) * 0.5
            self.extents = np.array([xmax - xmin, ymax - ymin, zmax - zmin]) * 0.5
            self.rotation = np.eye(3)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        d = p_arr - self.center
        d_local = d @ self.rotation
        q = np.abs(d_local) - self.extents
        out_dist = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        in_dist = np.minimum(np.max(q, axis=-1), 0.0)
        return out_dist + in_dist

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) == 0:
            return p_arr
        d = p_arr - self.center
        d_local = d @ self.rotation

        # Outer clamping
        q = np.clip(d_local, -self.extents, self.extents)

        # Inner projection to closest face
        inside_mask = np.all(np.abs(d_local) <= self.extents, axis=-1)
        if np.any(inside_mask):
            sub_d = d_local[inside_mask]
            dist_to_faces = self.extents[None, :] - np.abs(sub_d)
            face_idx = np.argmin(dist_to_faces, axis=-1)

            row_idx = np.arange(len(face_idx))
            sgn = np.sign(sub_d[row_idx, face_idx])
            sgn[sgn == 0] = 1.0

            q_sub = q[inside_mask]
            q_sub[row_idx, face_idx] = sgn * self.extents[face_idx]
            q[inside_mask] = q_sub

        p_proj = self.center + q @ self.rotation.T
        dist = np.linalg.norm(p_arr - p_proj, axis=-1)
        snap_mask = dist <= tolerance

        out = p_arr.copy()
        out[snap_mask] = p_proj[snap_mask]
        return out


class CylinderShape(AnalyticShape):
    """An analytical cylinder in physical space, supporting arbitrary orientation."""

    def __init__(
        self,
        center: np.ndarray,
        axis: np.ndarray,
        half_height: float,
        r_a: float,
        r_b: float,
        t_a: np.ndarray,
        t_b: np.ndarray,
        is_solid: bool = True,
    ) -> None:
        super().__init__(is_solid)
        self.center = np.asarray(center, dtype=float)
        axis_norm = np.asarray(axis, dtype=float)
        self.axis = axis_norm / (np.linalg.norm(axis_norm) or 1.0)
        self.half_height = float(half_height)
        self.r_a = float(r_a)
        self.r_b = float(r_b)
        self.t_a = np.asarray(t_a, dtype=float)
        self.t_b = np.asarray(t_b, dtype=float)

    @classmethod
    def from_legacy(
        cls,
        axis_name: str,
        c0: float,
        c1: float,
        lo: float,
        hi: float,
        radius_a: float,
        radius_b: float,
        mins: np.ndarray,
        span: np.ndarray,
        is_solid: bool = True,
    ) -> CylinderShape:
        axis_name = str(axis_name or "z").lower().strip()
        lo, hi = sorted((float(lo), float(hi)))

        if axis_name == "x":
            axis_vec = np.array([1.0, 0.0, 0.0])
            t_a = np.array([0.0, 1.0, 0.0])
            t_b = np.array([0.0, 0.0, 1.0])
            cy = mins[1] + c0 * span[1]
            cz = mins[2] + c1 * span[2]
            xmin = mins[0] + lo * span[0]
            xmax = mins[0] + hi * span[0]
            center = np.array([0.5 * (xmin + xmax), cy, cz])
            half_height = 0.5 * (xmax - xmin)
            r_a = radius_a * span[1]
            r_b = radius_b * span[2]
        elif axis_name == "y":
            axis_vec = np.array([0.0, 1.0, 0.0])
            t_a = np.array([1.0, 0.0, 0.0])
            t_b = np.array([0.0, 0.0, 1.0])
            cx = mins[0] + c0 * span[0]
            cz = mins[2] + c1 * span[2]
            ymin = mins[1] + lo * span[1]
            ymax = mins[1] + hi * span[1]
            center = np.array([cx, 0.5 * (ymin + ymax), cz])
            half_height = 0.5 * (ymax - ymin)
            r_a = radius_a * span[0]
            r_b = radius_b * span[2]
        else:  # 'z'
            axis_vec = np.array([0.0, 0.0, 1.0])
            t_a = np.array([1.0, 0.0, 0.0])
            t_b = np.array([0.0, 1.0, 0.0])
            cx = mins[0] + c0 * span[0]
            cy = mins[1] + c1 * span[1]
            zmin = mins[2] + lo * span[2]
            zmax = mins[2] + hi * span[2]
            center = np.array([cx, cy, 0.5 * (zmin + zmax)])
            half_height = 0.5 * (zmax - zmin)
            r_a = radius_a * span[0]
            r_b = radius_b * span[1]

        return cls(center, axis_vec, half_height, r_a, r_b, t_a, t_b, is_solid)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        d = p_arr - self.center
        z_local = d @ self.axis
        x_local = d @ self.t_a
        y_local = d @ self.t_b

        ellipse_val = np.sqrt((x_local / self.r_a) ** 2 + (y_local / self.r_b) ** 2)
        d_r = (ellipse_val - 1.0) * min(self.r_a, self.r_b)
        d_a = np.abs(z_local) - self.half_height

        sdf = np.maximum(d_r, d_a)
        outer_corners = (d_r > 0.0) & (d_a > 0.0)
        sdf = np.where(
            outer_corners,
            np.sqrt(np.maximum(d_r, 0.0) ** 2 + np.maximum(d_a, 0.0) ** 2),
            sdf,
        )
        return sdf

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) == 0:
            return p_arr
        d = p_arr - self.center
        z_local = d @ self.axis
        x_local = d @ self.t_a
        y_local = d @ self.t_b

        ellipse_val = np.sqrt((x_local / self.r_a) ** 2 + (y_local / self.r_b) ** 2)
        ellipse_val = np.maximum(ellipse_val, 1e-12)

        x_proj = x_local.copy()
        y_proj = y_local.copy()
        z_proj = z_local.copy()

        dist_to_sides = (1.0 - ellipse_val) * min(self.r_a, self.r_b)
        dist_to_ends = self.half_height - np.abs(z_local)

        inside_mask = (ellipse_val <= 1.0) & (np.abs(z_local) <= self.half_height)
        outside_mask = ~inside_mask

        # 1. Project points inside cylinder
        project_to_ends = inside_mask & (dist_to_ends < dist_to_sides)
        project_to_sides = inside_mask & (dist_to_ends >= dist_to_sides)

        z_proj[project_to_ends] = np.sign(z_local[project_to_ends]) * self.half_height

        x_proj[project_to_sides] = (
            x_local[project_to_sides] / ellipse_val[project_to_sides]
        )
        y_proj[project_to_sides] = (
            y_local[project_to_sides] / ellipse_val[project_to_sides]
        )

        # 2. Project points outside cylinder
        x_proj[outside_mask] = x_local[outside_mask] / ellipse_val[outside_mask]
        y_proj[outside_mask] = y_local[outside_mask] / ellipse_val[outside_mask]
        z_proj[outside_mask] = np.clip(
            z_local[outside_mask], -self.half_height, self.half_height
        )

        # Reconstruct coordinates
        p_proj = (
            self.center
            + z_proj[:, None] * self.axis
            + x_proj[:, None] * self.t_a
            + y_proj[:, None] * self.t_b
        )
        dist = np.linalg.norm(p_arr - p_proj, axis=-1)
        snap_mask = dist <= tolerance

        out = p_arr.copy()
        out[snap_mask] = p_proj[snap_mask]
        return out


class SphereShape(AnalyticShape):
    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        is_solid: bool = True,
    ) -> None:
        super().__init__(is_solid)
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        return np.linalg.norm(p_arr - self.center, axis=-1) - self.radius

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) == 0:
            return p_arr
        d = p_arr - self.center
        dist = np.linalg.norm(d, axis=-1)
        dist_val = np.maximum(dist, 1e-12)
        p_proj = self.center + d * (self.radius / dist_val[:, None])
        snap_mask = np.abs(dist - self.radius) <= tolerance

        out = p_arr.copy()
        out[snap_mask] = p_proj[snap_mask]
        return out


class CapsuleShape(AnalyticShape):
    def __init__(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        radius: float,
        is_solid: bool = True,
    ) -> None:
        super().__init__(is_solid)
        self.p1 = np.asarray(p1, dtype=float)
        self.p2 = np.asarray(p2, dtype=float)
        self.radius = float(radius)

    def _closest_points(self, p_arr: np.ndarray) -> np.ndarray:
        v = self.p2 - self.p1
        len2 = np.sum(v**2)
        if len2 < 1e-12:
            return np.broadcast_to(self.p1, p_arr.shape)
        t = np.dot(p_arr - self.p1, v) / len2
        t = np.clip(t, 0.0, 1.0)
        return self.p1 + t[:, None] * v

    def sdf(self, p: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        orig_shape = p_arr.shape
        p_flat = p_arr.reshape(-1, 3)
        q_flat = self._closest_points(p_flat)
        sdf_flat = np.linalg.norm(p_flat - q_flat, axis=-1) - self.radius
        return sdf_flat.reshape(orig_shape[:-1])

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) == 0:
            return p_arr
        q = self._closest_points(p_arr)
        d = p_arr - q
        dist = np.linalg.norm(d, axis=-1)
        dist_val = np.maximum(dist, 1e-12)
        p_proj = q + d * (self.radius / dist_val[:, None])
        snap_mask = np.abs(dist - self.radius) <= tolerance

        out = p_arr.copy()
        out[snap_mask] = p_proj[snap_mask]
        return out


class TorusShape(AnalyticShape):
    def __init__(
        self,
        center: np.ndarray,
        normal: np.ndarray,
        r_major: float,
        r_minor: float,
        is_solid: bool = True,
    ) -> None:
        super().__init__(is_solid)
        self.center = np.asarray(center, dtype=float)
        norm_arr = np.asarray(normal, dtype=float)
        self.normal = norm_arr / (np.linalg.norm(norm_arr) or 1.0)
        self.r_major = float(r_major)
        self.r_minor = float(r_minor)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        d = p_arr - self.center
        h = d @ self.normal
        d_plane = d - h[..., None] * self.normal
        dist_plane = np.linalg.norm(d_plane, axis=-1)
        return np.sqrt((dist_plane - self.r_major) ** 2 + h**2) - self.r_minor

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) == 0:
            return p_arr
        d = p_arr - self.center
        h = d @ self.normal
        d_plane = d - h[:, None] * self.normal
        dist_plane = np.linalg.norm(d_plane, axis=-1)
        dist_plane_val = np.maximum(dist_plane, 1e-12)

        q = self.center + d_plane * (self.r_major / dist_plane_val[:, None])
        d_tube = p_arr - q
        dist_tube = np.linalg.norm(d_tube, axis=-1)
        dist_tube_val = np.maximum(dist_tube, 1e-12)

        p_proj = q + d_tube * (self.r_minor / dist_tube_val[:, None])
        snap_mask = np.abs(dist_tube - self.r_minor) <= tolerance

        out = p_arr.copy()
        out[snap_mask] = p_proj[snap_mask]
        return out


class ConeShape(AnalyticShape):
    def __init__(
        self,
        base: np.ndarray,
        apex: np.ndarray,
        radius: float,
        is_solid: bool = True,
    ) -> None:
        super().__init__(is_solid)
        self.base = np.asarray(base, dtype=float)
        self.apex = np.asarray(apex, dtype=float)
        self.radius = float(radius)
        v = self.apex - self.base
        self.height = np.linalg.norm(v)
        self.axis = v / (self.height if self.height > 1e-12 else 1.0)
        self.cos_theta = self.height / np.sqrt(self.height**2 + self.radius**2)
        self.sin_theta = self.radius / np.sqrt(self.height**2 + self.radius**2)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        d = p_arr - self.base
        z_local = d @ self.axis
        p_trans = d - z_local[..., None] * self.axis
        r_local = np.linalg.norm(p_trans, axis=-1)

        z_prime = self.height - z_local
        d_s = r_local * self.cos_theta - z_prime * self.sin_theta

        sdf = np.maximum(d_s, -z_local)
        sdf = np.maximum(sdf, z_local - self.height)
        return sdf

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) == 0:
            return p_arr
        d = p_arr - self.base
        z_local = d @ self.axis
        p_trans = d - z_local[:, None] * self.axis
        r_local = np.linalg.norm(p_trans, axis=-1)
        r_local_val = np.maximum(r_local, 1e-12)

        p_proj = p_arr.copy()

        # Side projection
        z_prime = self.height - z_local
        k = self.radius / self.height
        factor = 1.0 / (k**2 + 1.0)
        proj_z_prime = np.clip((r_local * k + z_prime) * factor, 0.0, self.height)
        proj_r = proj_z_prime * k
        proj_z_local = self.height - proj_z_prime

        p_side = (
            self.base
            + proj_z_local[:, None] * self.axis
            + p_trans * (proj_r / r_local_val)[:, None]
        )

        # Base projection
        p_base = (
            self.base
            + p_trans * (np.clip(r_local, 0.0, self.radius) / r_local_val)[:, None]
        )

        dist_side = np.linalg.norm(p_arr - p_side, axis=-1)
        dist_base = np.linalg.norm(p_arr - p_base, axis=-1)

        use_base = dist_base < dist_side
        p_proj[use_base] = p_base[use_base]
        p_proj[~use_base] = p_side[~use_base]

        min_dist = np.minimum(dist_side, dist_base)
        snap_mask = min_dist <= tolerance

        out = p_arr.copy()
        out[snap_mask] = p_proj[snap_mask]
        return out


class PlaneShape(AnalyticShape):
    def __init__(
        self,
        point: np.ndarray,
        normal: np.ndarray,
        is_solid: bool = True,
    ) -> None:
        super().__init__(is_solid)
        self.point = np.asarray(point, dtype=float)
        norm_arr = np.asarray(normal, dtype=float)
        self.normal = norm_arr / (np.linalg.norm(norm_arr) or 1.0)

    def sdf(self, p: np.ndarray) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        return (p_arr - self.point) @ self.normal

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) == 0:
            return p_arr
        dist = (p_arr - self.point) @ self.normal
        p_proj = p_arr - dist[:, None] * self.normal
        snap_mask = np.abs(dist) <= tolerance

        out = p_arr.copy()
        out[snap_mask] = p_proj[snap_mask]
        return out


def _orthogonal_vectors(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors t_a, t_b such that (v, t_a, t_b) form an orthonormal basis."""
    v = np.asarray(v, dtype=float)
    v = v / (np.linalg.norm(v) or 1.0)
    if np.abs(v[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0])
    else:
        other = np.array([0.0, 1.0, 0.0])
    t_a = np.cross(v, other)
    t_a = t_a / np.linalg.norm(t_a)
    t_b = np.cross(v, t_a)
    t_b = t_b / np.linalg.norm(t_b)
    return t_a, t_b


def _parse_shape_from_dict(
    d: dict, bounds: Optional[tuple[np.ndarray, np.ndarray]]
) -> Optional[AnalyticShape]:
    """Parse a shape dictionary into an AnalyticShape object."""
    shape_type = str(d.get("type") or d.get("shape") or "").strip().lower()
    is_solid = bool(d.get("is_solid", True))

    if shape_type in {"sphere", "spherical"}:
        center = np.asarray(d.get("center", [0.0, 0.0, 0.0]), dtype=float)
        radius = float(d.get("radius", d.get("r", 1.0)))
        return SphereShape(center=center, radius=radius, is_solid=is_solid)

    elif shape_type in {"cylinder", "cylindrical", "hole"}:
        if "p1" in d and "p2" in d:
            p1 = np.asarray(d["p1"], dtype=float)
            p2 = np.asarray(d["p2"], dtype=float)
            radius = float(d.get("radius", d.get("r", 1.0)))
            v = p2 - p1
            h = np.linalg.norm(v)
            axis = v / (h if h > 1e-12 else 1.0)
            center = 0.5 * (p1 + p2)
            t_a, t_b = _orthogonal_vectors(axis)
            return CylinderShape(
                center=center,
                axis=axis,
                half_height=0.5 * h,
                r_a=radius,
                r_b=radius,
                t_a=t_a,
                t_b=t_b,
                is_solid=is_solid,
            )
        elif "axis" in d:
            axis_name = str(d.get("axis", "z")).strip().lower()
            radius = float(d.get("radius", d.get("r", 1.0)))
            center = np.asarray(d.get("center", [0.0, 0.0, 0.0]), dtype=float)
            height = float(d.get("height", d.get("h", 1.0)))

            axis_vec = np.array([0.0, 0.0, 1.0])
            if axis_name == "x":
                axis_vec = np.array([1.0, 0.0, 0.0])
            elif axis_name == "y":
                axis_vec = np.array([0.0, 1.0, 0.0])

            t_a, t_b = _orthogonal_vectors(axis_vec)
            return CylinderShape(
                center=center,
                axis=axis_vec,
                half_height=0.5 * height,
                r_a=radius,
                r_b=radius,
                t_a=t_a,
                t_b=t_b,
                is_solid=is_solid,
            )

    elif shape_type in {"box", "rectangular", "cuboid"}:
        if "center" in d and "size" in d:
            center = np.asarray(d["center"], dtype=float)
            size = np.asarray(d["size"], dtype=float)
            extents = 0.5 * size
            rot_mat = np.eye(3)
            if "rotation" in d:
                rot_mat = np.asarray(d["rotation"], dtype=float).reshape(3, 3)
            return BoxShape(
                center=center, extents=extents, rotation=rot_mat, is_solid=is_solid
            )
        elif "xmin" in d and "xmax" in d:
            return BoxShape(
                x_min=float(d["xmin"]),
                x_max=float(d["xmax"]),
                y_min=float(d.get("ymin", 0.0)),
                y_max=float(d.get("ymax", 1.0)),
                z_min=float(d.get("zmin", 0.0)),
                z_max=float(d.get("zmax", 1.0)),
                is_solid=is_solid,
            )

    elif shape_type in {"capsule"}:
        p1 = np.asarray(d.get("p1", [0.0, 0.0, 0.0]), dtype=float)
        p2 = np.asarray(d.get("p2", [0.0, 0.0, 1.0]), dtype=float)
        radius = float(d.get("radius", d.get("r", 0.5)))
        return CapsuleShape(p1=p1, p2=p2, radius=radius, is_solid=is_solid)

    elif shape_type in {"torus"}:
        center = np.asarray(d.get("center", [0.0, 0.0, 0.0]), dtype=float)
        normal = np.asarray(d.get("normal", [0.0, 0.0, 1.0]), dtype=float)
        normal = normal / (np.linalg.norm(normal) or 1.0)
        r_major = float(d.get("r_major", d.get("R", 1.0)))
        r_minor = float(d.get("r_minor", d.get("r", 0.2)))
        return TorusShape(
            center=center,
            normal=normal,
            r_major=r_major,
            r_minor=r_minor,
            is_solid=is_solid,
        )

    elif shape_type in {"cone"}:
        base = np.asarray(d.get("base", [0.0, 0.0, 0.0]), dtype=float)
        apex = np.asarray(d.get("apex", [0.0, 0.0, 1.0]), dtype=float)
        radius = float(d.get("radius", d.get("r", 0.5)))
        return ConeShape(base=base, apex=apex, radius=radius, is_solid=is_solid)

    elif shape_type in {"plane", "halfspace"}:
        point = np.asarray(d.get("point", [0.0, 0.0, 0.0]), dtype=float)
        normal = np.asarray(d.get("normal", [0.0, 0.0, 1.0]), dtype=float)
        normal = normal / (np.linalg.norm(normal) or 1.0)
        return PlaneShape(point=point, normal=normal, is_solid=is_solid)

    return None


def _convert_legacy_to_physical_shapes(
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
) -> list[AnalyticShape]:
    """Convert legacy fractional boxes/cylinders or general dictionaries to physical shapes."""
    if bounds is None:
        return []
    mins, maxs = bounds
    mins = np.asarray(mins, dtype=float)[:3]
    maxs = np.asarray(maxs, dtype=float)[:3]
    span = np.maximum(maxs - mins, 1e-12)

    shapes = []

    # Parse solid boxes
    for box in solid_boxes or ():
        if isinstance(box, dict):
            shape = _parse_shape_from_dict(box, bounds)
            if shape is not None:
                shapes.append(shape)
        elif len(box) >= 6:
            x0, x1, y0, y1, z0, z1 = [float(v) for v in box[:6]]
            shapes.append(
                BoxShape(
                    x_min=mins[0] + min(x0, x1) * span[0],
                    x_max=mins[0] + max(x0, x1) * span[0],
                    y_min=mins[1] + min(y0, y1) * span[1],
                    y_max=mins[1] + max(y0, y1) * span[1],
                    z_min=mins[2] + min(z0, z1) * span[2],
                    z_max=mins[2] + max(z0, z1) * span[2],
                    is_solid=True,
                )
            )

    # Parse void boxes
    for box in void_boxes or ():
        if isinstance(box, dict):
            shape = _parse_shape_from_dict(box, bounds)
            if shape is not None:
                shapes.append(shape)
        elif len(box) >= 6:
            x0, x1, y0, y1, z0, z1 = [float(v) for v in box[:6]]
            shapes.append(
                BoxShape(
                    x_min=mins[0] + min(x0, x1) * span[0],
                    x_max=mins[0] + max(x0, x1) * span[0],
                    y_min=mins[1] + min(y0, y1) * span[1],
                    y_max=mins[1] + max(y0, y1) * span[1],
                    z_min=mins[2] + min(z0, z1) * span[2],
                    z_max=mins[2] + max(z0, z1) * span[2],
                    is_solid=False,
                )
            )

    # Parse solid cylinders
    for cyl in solid_cylinders or ():
        if isinstance(cyl, dict):
            shape = _parse_shape_from_dict(cyl, bounds)
            if shape is not None:
                shapes.append(shape)
        else:
            parsed = _split_cylinder_region(cyl)
            if parsed is not None:
                axis_name, c0, c1, lo, hi, radius_a, radius_b = parsed
                shapes.append(
                    CylinderShape.from_legacy(
                        axis_name=axis_name,
                        c0=c0,
                        c1=c1,
                        lo=lo,
                        hi=hi,
                        radius_a=radius_a,
                        radius_b=radius_b,
                        mins=mins,
                        span=span,
                        is_solid=True,
                    )
                )

    # Parse void cylinders
    for cyl in void_cylinders or ():
        if isinstance(cyl, dict):
            shape = _parse_shape_from_dict(cyl, bounds)
            if shape is not None:
                shapes.append(shape)
        else:
            parsed = _split_cylinder_region(cyl)
            if parsed is not None:
                axis_name, c0, c1, lo, hi, radius_a, radius_b = parsed
                shapes.append(
                    CylinderShape.from_legacy(
                        axis_name=axis_name,
                        c0=c0,
                        c1=c1,
                        lo=lo,
                        hi=hi,
                        radius_a=radius_a,
                        radius_b=radius_b,
                        mins=mins,
                        span=span,
                        is_solid=False,
                    )
                )

    return shapes


def _project_passive_shapes_surfaces(
    vertices: np.ndarray,
    shapes: list[AnalyticShape],
    tolerance: float,
) -> np.ndarray:
    """Project physical vertices near shapes onto their exact analytical boundaries."""
    if len(vertices) == 0 or not shapes:
        return vertices

    out = np.asarray(vertices, dtype=float).copy()
    for shape in shapes:
        out = shape.project(out, tolerance)
    return out
