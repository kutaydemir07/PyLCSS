# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Marching-cubes surface recovery and smoothing for the voxel-optimised field."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CylinderRegion = Tuple[str, float, float, float, float, float]
BoxRegion = Tuple[float, float, float, float, float, float]


def _split_cylinder_region(cylinder) -> Optional[Tuple[str, float, float, float, float, float, float]]:
    if cylinder is None or len(cylinder) < 6:
        return None
    axis, c0, c1, lo, hi, radius_a = cylinder[:6]
    radius_b = cylinder[6] if len(cylinder) > 6 else radius_a
    radius_a = float(radius_a)
    radius_b = float(radius_b)
    if radius_a <= 0.0 or radius_b <= 0.0:
        return None
    return (
        str(axis or 'z').lower(),
        float(c0),
        float(c1),
        float(lo),
        float(hi),
        radius_a,
        radius_b,
    )


def _voxel_origin_cell(
    shape: Tuple[int, int, int],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return physical origin and cell size for a structured voxel grid."""
    shape_arr = np.maximum(np.asarray(shape, dtype=float), 1.0)
    if bounds is not None:
        mins, maxs = bounds
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)
        if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
            return mins[:3], (maxs[:3] - mins[:3]) / shape_arr
    return -0.5 * shape_arr, np.ones(3, dtype=float)


def _regularize_extruded_density(grid: np.ndarray, axis: str) -> np.ndarray:
    """Force a density field to be constant along an extrusion axis."""
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax = axis_map.get(str(axis or '').strip().lower())
    if ax is None or grid.ndim != 3:
        return grid
    profile = np.mean(grid, axis=ax, keepdims=True)
    return np.broadcast_to(profile, grid.shape).copy()


def _project_extruded_planes(
    vertices: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    axis: str,
    tolerance: float,
) -> np.ndarray:
    """Snap top/bottom vertices of an extruded result back onto exact planes."""
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax = axis_map.get(str(axis or '').strip().lower())
    if ax is None or bounds is None or len(vertices) == 0:
        return vertices

    mins, maxs = bounds
    lo = float(np.asarray(mins, dtype=float)[ax])
    hi = float(np.asarray(maxs, dtype=float)[ax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vertices

    out = np.asarray(vertices, dtype=float).copy()
    tol = max(float(tolerance), 1e-9)
    out[out[:, ax] <= lo + tol, ax] = lo
    out[out[:, ax] >= hi - tol, ax] = hi
    return out


# =========================================================================
# Robust 3D Physical Analytic Shape Engine
# =========================================================================

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
    def __init__(self, is_solid: bool = True):
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
        x_min: float = 0.0, x_max: float = 0.0,
        y_min: float = 0.0, y_max: float = 0.0,
        z_min: float = 0.0, z_max: float = 0.0,
        center: Optional[np.ndarray] = None,
        extents: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
        is_solid: bool = True
    ):
        super().__init__(is_solid)
        if center is not None and extents is not None:
            self.center = np.asarray(center, dtype=float)
            self.extents = np.asarray(extents, dtype=float)
            self.rotation = np.asarray(rotation, dtype=float) if rotation is not None else np.eye(3)
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
        is_solid: bool = True
    ):
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
        c0: float, c1: float,
        lo: float, hi: float,
        radius_a: float,
        radius_b: float,
        mins: np.ndarray,
        span: np.ndarray,
        is_solid: bool = True
    ) -> CylinderShape:
        axis_name = str(axis_name or 'z').lower().strip()
        lo, hi = sorted((float(lo), float(hi)))
        
        if axis_name == 'x':
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
        elif axis_name == 'y':
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
        else: # 'z'
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
        
        ellipse_val = np.sqrt((x_local / self.r_a)**2 + (y_local / self.r_b)**2)
        d_r = (ellipse_val - 1.0) * min(self.r_a, self.r_b)
        d_a = np.abs(z_local) - self.half_height
        
        sdf = np.maximum(d_r, d_a)
        outer_corners = (d_r > 0.0) & (d_a > 0.0)
        sdf = np.where(outer_corners, np.sqrt(np.maximum(d_r, 0.0)**2 + np.maximum(d_a, 0.0)**2), sdf)
        return sdf

    def project(self, p: np.ndarray, tolerance: float) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if len(p_arr) == 0:
            return p_arr
        d = p_arr - self.center
        z_local = d @ self.axis
        x_local = d @ self.t_a
        y_local = d @ self.t_b
        
        ellipse_val = np.sqrt((x_local / self.r_a)**2 + (y_local / self.r_b)**2)
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
        
        x_proj[project_to_sides] = x_local[project_to_sides] / ellipse_val[project_to_sides]
        y_proj[project_to_sides] = y_local[project_to_sides] / ellipse_val[project_to_sides]
        
        # 2. Project points outside cylinder
        x_proj[outside_mask] = x_local[outside_mask] / ellipse_val[outside_mask]
        y_proj[outside_mask] = y_local[outside_mask] / ellipse_val[outside_mask]
        z_proj[outside_mask] = np.clip(z_local[outside_mask], -self.half_height, self.half_height)
        
        # Reconstruct coordinates
        p_proj = self.center + z_proj[:, None] * self.axis + x_proj[:, None] * self.t_a + y_proj[:, None] * self.t_b
        dist = np.linalg.norm(p_arr - p_proj, axis=-1)
        snap_mask = dist <= tolerance
        
        out = p_arr.copy()
        out[snap_mask] = p_proj[snap_mask]
        return out


class SphereShape(AnalyticShape):
    def __init__(self, center: np.ndarray, radius: float, is_solid: bool = True):
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
    def __init__(self, p1: np.ndarray, p2: np.ndarray, radius: float, is_solid: bool = True):
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
    def __init__(self, center: np.ndarray, normal: np.ndarray, r_major: float, r_minor: float, is_solid: bool = True):
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
        return np.sqrt((dist_plane - self.r_major)**2 + h**2) - self.r_minor

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
    def __init__(self, base: np.ndarray, apex: np.ndarray, radius: float, is_solid: bool = True):
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
        
        p_side = self.base + proj_z_local[:, None] * self.axis + p_trans * (proj_r / r_local_val)[:, None]
        
        # Base projection
        p_base = self.base + p_trans * (np.clip(r_local, 0.0, self.radius) / r_local_val)[:, None]
        
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
    def __init__(self, point: np.ndarray, normal: np.ndarray, is_solid: bool = True):
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


def _orthogonal_vectors(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def _parse_shape_from_dict(d: dict, bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[AnalyticShape]:
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
            return CylinderShape(center=center, axis=axis, half_height=0.5*h, r_a=radius, r_b=radius, t_a=t_a, t_b=t_b, is_solid=is_solid)
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
            return CylinderShape(center=center, axis=axis_vec, half_height=0.5*height, r_a=radius, r_b=radius, t_a=t_a, t_b=t_b, is_solid=is_solid)
            
    elif shape_type in {"box", "rectangular", "cuboid"}:
        if "center" in d and "size" in d:
            center = np.asarray(d["center"], dtype=float)
            size = np.asarray(d["size"], dtype=float)
            extents = 0.5 * size
            rot_mat = np.eye(3)
            if "rotation" in d:
                rot_mat = np.asarray(d["rotation"], dtype=float).reshape(3, 3)
            return BoxShape(center=center, extents=extents, rotation=rot_mat, is_solid=is_solid)
        elif "xmin" in d and "xmax" in d:
            return BoxShape(
                x_min=float(d["xmin"]), x_max=float(d["xmax"]),
                y_min=float(d.get("ymin", 0.0)), y_max=float(d.get("ymax", 1.0)),
                z_min=float(d.get("zmin", 0.0)), z_max=float(d.get("zmax", 1.0)),
                is_solid=is_solid
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
        return TorusShape(center=center, normal=normal, r_major=r_major, r_minor=r_minor, is_solid=is_solid)
        
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
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
) -> List[AnalyticShape]:
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
            shapes.append(BoxShape(
                x_min=mins[0] + min(x0, x1) * span[0],
                x_max=mins[0] + max(x0, x1) * span[0],
                y_min=mins[1] + min(y0, y1) * span[1],
                y_max=mins[1] + max(y0, y1) * span[1],
                z_min=mins[2] + min(z0, z1) * span[2],
                z_max=mins[2] + max(z0, z1) * span[2],
                is_solid=True
            ))

    # Parse void boxes
    for box in void_boxes or ():
        if isinstance(box, dict):
            shape = _parse_shape_from_dict(box, bounds)
            if shape is not None:
                shapes.append(shape)
        elif len(box) >= 6:
            x0, x1, y0, y1, z0, z1 = [float(v) for v in box[:6]]
            shapes.append(BoxShape(
                x_min=mins[0] + min(x0, x1) * span[0],
                x_max=mins[0] + max(x0, x1) * span[0],
                y_min=mins[1] + min(y0, y1) * span[1],
                y_max=mins[1] + max(y0, y1) * span[1],
                z_min=mins[2] + min(z0, z1) * span[2],
                z_max=mins[2] + max(z0, z1) * span[2],
                is_solid=False
            ))

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
                shapes.append(CylinderShape.from_legacy(
                    axis_name=axis_name,
                    c0=c0, c1=c1,
                    lo=lo, hi=hi,
                    radius_a=radius_a,
                    radius_b=radius_b,
                    mins=mins,
                    span=span,
                    is_solid=True
                ))

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
                shapes.append(CylinderShape.from_legacy(
                    axis_name=axis_name,
                    c0=c0, c1=c1,
                    lo=lo, hi=hi,
                    radius_a=radius_a,
                    radius_b=radius_b,
                    mins=mins,
                    span=span,
                    is_solid=False
                ))

    return shapes


def _project_passive_shapes_surfaces(
    vertices: np.ndarray,
    shapes: List[AnalyticShape],
    tolerance: float,
) -> np.ndarray:
    """Project physical vertices near shapes onto their exact analytical boundaries."""
    if len(vertices) == 0 or not shapes:
        return vertices

    out = np.asarray(vertices, dtype=float).copy()
    for shape in shapes:
        out = shape.project(out, tolerance)
    return out


def _enhanced_mesh_postprocess(
    verts: np.ndarray,
    faces: np.ndarray,
    decimate_ratio: float = 1.0,
    smoothing_iterations: int = 2,
) -> Optional[Dict[str, np.ndarray]]:
    """Run a print-ready post-processing pipeline using trimesh.

    Pipeline:
        1. Split into connected components, keep ones above 1% of max volume.
        2. Fill small holes.
        3. Light Humphrey smoothing (volume-preserving alternative to Laplacian).
        4. Optional quadric decimation (requires `fast_simplification`).

    Returns None if trimesh is unavailable so the caller can fall back to the
    legacy Taubin path.
    """
    try:
        import trimesh
        import trimesh.smoothing
    except ImportError:
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None

    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    # 1. Keep only meaningful connected components.
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        components = [mesh]
    if components:
        volumes = [abs(float(c.volume)) for c in components]
        if volumes:
            max_vol = max(volumes)
            kept = [c for c, v in zip(components, volumes)
                    if v >= 0.01 * max_vol]
            if kept:
                mesh = trimesh.util.concatenate(kept)

    # 2. Close small holes.
    try:
        mesh.fill_holes()
    except Exception:
        logger.debug("trimesh.fill_holes failed; continuing")

    # 3. Volume-preserving Humphrey smoothing.
    try:
        trimesh.smoothing.filter_humphrey(
            mesh,
            alpha=0.1,
            beta=0.5,
            iterations=max(1, int(smoothing_iterations)),
        )
    except Exception:
        logger.debug("Humphrey smoothing failed; falling back to Taubin")
        try:
            trimesh.smoothing.filter_taubin(
                mesh, iterations=max(1, int(smoothing_iterations))
            )
        except Exception:
            pass

    # 4. Optional decimation (requires `fast_simplification`).
    if 0.0 < float(decimate_ratio) < 1.0 and len(mesh.faces) > 0:
        target = max(64, int(len(mesh.faces) * float(decimate_ratio)))
        try:
            decimated = mesh.simplify_quadric_decimation(face_count=target)
            if decimated is not None and len(decimated.faces) > 0:
                mesh = decimated
        except ImportError:
            logger.info(
                "STL decimation skipped — install `fast_simplification` for "
                "quadric decimation support."
            )
        except Exception:
            logger.debug("Decimation failed; keeping un-decimated mesh")

    return {
        'vertices': np.asarray(mesh.vertices, dtype=float),
        'faces':    np.asarray(mesh.faces,    dtype=int),
    }


def _taubin_smooth_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 6,
    shapes: Sequence[AnalyticShape] = (),
    tolerance: float = 0.0,
) -> np.ndarray:
    """Light volume-preserving smoothing for marching-cubes output."""
    if len(verts) == 0 or len(faces) == 0 or iterations <= 0:
        return verts

    verts = np.asarray(verts, dtype=float).copy()
    faces = np.asarray(faces, dtype=int)
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])

    for _ in range(max(0, int(iterations))):
        for factor in (0.5, -0.53):
            neighbor_sum = np.zeros_like(verts)
            neighbor_count = np.zeros(len(verts), dtype=float)
            np.add.at(neighbor_sum, edges[:, 0], verts[edges[:, 1]])
            np.add.at(neighbor_count, edges[:, 0], 1.0)
            np.add.at(neighbor_sum, edges[:, 1], verts[edges[:, 0]])
            np.add.at(neighbor_count, edges[:, 1], 1.0)
            mask = neighbor_count > 0
            avg = neighbor_sum[mask] / neighbor_count[mask, None]
            
            displacement = avg - verts[mask]
            
            if shapes and tolerance > 0.0:
                for shape in shapes:
                    dist = shape.sdf(verts[mask])
                    near_mask = np.abs(dist) <= tolerance
                    if np.any(near_mask):
                        n = shape.get_normal(verts[mask][near_mask])
                        disp_sub = displacement[near_mask]
                        proj_dot = np.sum(disp_sub * n, axis=-1, keepdims=True)
                        displacement[near_mask] = disp_sub - proj_dot * n

            verts[mask] += factor * displacement
    return verts


def _apply_passive_density_regions(
    field: np.ndarray,
    *,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cell: Optional[np.ndarray] = None,
    spacing: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clamp passive regions on the high-resolution scalar field.

    The optimiser stores circular non-design regions analytically, but the
    density array itself is voxelised. Re-applying the analytic regions here
    prevents circular holes from becoming low-sided polygons during export.
    """
    if field.ndim != 3:
        return field

    if bounds is None:
        shape_arr = np.maximum(np.asarray(field.shape, dtype=float), 1.0)
        mins = -0.5 * shape_arr
        maxs = 0.5 * shape_arr
        bounds = (mins, maxs)

    shapes = _convert_legacy_to_physical_shapes(
        bounds,
        solid_boxes=solid_boxes,
        void_boxes=void_boxes,
        solid_cylinders=solid_cylinders,
        void_cylinders=void_cylinders,
    )
    if not shapes:
        return field

    mins, maxs = bounds
    mins = np.asarray(mins, dtype=float)[:3]
    maxs = np.asarray(maxs, dtype=float)[:3]
    span = np.maximum(maxs - mins, 1e-12)

    nx, ny, nz = field.shape

    if cell is None or spacing is None:
        cell = span / np.maximum(np.asarray(field.shape, dtype=float), 1.0)
        spacing = cell

    x_phys = mins[0] + 0.5 * spacing[0] + np.arange(nx, dtype=float) * spacing[0]
    y_phys = mins[1] + 0.5 * spacing[1] + np.arange(ny, dtype=float) * spacing[1]
    z_phys = mins[2] + 0.5 * spacing[2] + np.arange(nz, dtype=float) * spacing[2]

    # Create coordinate grid in physical space
    X, Y, Z = np.meshgrid(x_phys, y_phys, z_phys, indexing='ij')
    pts_grid = np.stack([X, Y, Z], axis=-1)

    solid_shapes = [shape for shape in shapes if shape.is_solid]
    void_shapes = [shape for shape in shapes if not shape.is_solid]

    for shape in solid_shapes:
        shape_sdf = shape.sdf(pts_grid)
        field[shape_sdf <= 0.0] = 1.0
    for shape in void_shapes:
        shape_sdf = shape.sdf(pts_grid)
        field[shape_sdf <= 0.0] = 0.0

    return field


def _apply_passive_cylinder_sdf(
    signed_distance: np.ndarray,
    *,
    pad: int,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cell: Optional[np.ndarray] = None,
    spacing: Optional[np.ndarray] = None,
    blend_radius: float = 0.0,
) -> np.ndarray:
    """Re-impose analytic shape boundaries on the SDF.

    `signed_distance` uses negative values for material and positive values for
    void. A solid cylinder/shape is therefore a min operation; a through-hole/void
    is a max operation. Applying this after SDF smoothing keeps holes round instead of
    preserving the voxel-grid polygon.
    """
    if signed_distance.ndim != 3:
        return signed_distance

    if bounds is None:
        shape_arr = np.maximum(np.asarray(signed_distance.shape, dtype=float) - 2.0 * float(pad), 1.0)
        mins = -0.5 * shape_arr
        maxs = 0.5 * shape_arr
        bounds = (mins, maxs)

    shapes = _convert_legacy_to_physical_shapes(
        bounds,
        solid_boxes=solid_boxes,
        void_boxes=void_boxes,
        solid_cylinders=solid_cylinders,
        void_cylinders=void_cylinders,
    )
    if not shapes:
        return signed_distance

    mins, maxs = bounds
    mins = np.asarray(mins, dtype=float)[:3]
    maxs = np.asarray(maxs, dtype=float)[:3]
    span = np.maximum(maxs - mins, 1e-12)

    nx, ny, nz = signed_distance.shape

    if cell is None or spacing is None:
        shape_unpadded = np.maximum(np.asarray(signed_distance.shape, dtype=float) - 2.0 * float(pad), 1.0)
        cell = span / shape_unpadded
        spacing = cell

    x_phys = mins[0] + 0.5 * spacing[0] + (np.arange(nx, dtype=float) - float(pad)) * spacing[0]
    y_phys = mins[1] + 0.5 * spacing[1] + (np.arange(ny, dtype=float) - float(pad)) * spacing[1]
    z_phys = mins[2] + 0.5 * spacing[2] + (np.arange(nz, dtype=float) - float(pad)) * spacing[2]

    # Create coordinate grid in physical space
    X, Y, Z = np.meshgrid(x_phys, y_phys, z_phys, indexing='ij')
    pts_grid = np.stack([X, Y, Z], axis=-1)

    # Determine if signed_distance is in physical units or density units
    is_physical = (np.max(signed_distance) > 2.0) or (np.min(signed_distance) < -2.0)
    grid_spacing = 1.0
    if not is_physical:
        # Scale physical SDF to match unitless density field
        # A grid voxel has spacing. We can use the mean spacing as the scale factor.
        grid_spacing = np.mean(spacing)

    k = blend_radius if is_physical else (blend_radius / grid_spacing)

    solid_shapes = [shape for shape in shapes if shape.is_solid]
    void_shapes = [shape for shape in shapes if not shape.is_solid]

    for shape in solid_shapes:
        shape_sdf = shape.sdf(pts_grid)
        if not is_physical:
            shape_sdf = shape_sdf / grid_spacing

        if k > 0.0:
            signed_distance = smin(signed_distance, shape_sdf, k)
        else:
            np.minimum(signed_distance, shape_sdf, out=signed_distance)

    for shape in void_shapes:
        shape_sdf = shape.sdf(pts_grid)
        if not is_physical:
            shape_sdf = shape_sdf / grid_spacing

        if k > 0.0:
            signed_distance = smax(signed_distance, -shape_sdf, k)
        else:
            np.maximum(signed_distance, -shape_sdf, out=signed_distance)

    return signed_distance


def _resample_source_mask(
    source_mask: Optional[np.ndarray],
    target_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    if source_mask is None:
        return None
    source = np.asarray(source_mask, dtype=bool)
    if source.ndim != 3 or min(source.shape) < 1:
        return None
    if tuple(source.shape) == tuple(target_shape):
        return source
    try:
        import scipy.ndimage as ndi

        zoom = tuple(float(t) / float(s) for t, s in zip(target_shape, source.shape))
        resized = ndi.zoom(source.astype(float), zoom=zoom, order=0, mode='nearest', grid_mode=True)
        out = resized >= 0.5
        if out.shape != tuple(target_shape):
            cropped = np.zeros(target_shape, dtype=bool)
            common = tuple(slice(0, min(a, b)) for a, b in zip(cropped.shape, out.shape))
            cropped[common] = out[common]
            return cropped
        return out
    except Exception:
        logger.debug("Failed to resample topology source mask", exc_info=True)
        return None


def _recover_voxel_shape(
    density: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    cutoff: float,
    print_ready: bool = False,
    decimate_ratio: float = 1.0,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    extrusion_axis: str = 'none',
    source_mask: Optional[np.ndarray] = None,
    blend_radius: float = 0.0,
) -> Optional[Dict[str, np.ndarray]]:
    """Extract a recovered surface from a structured voxel density field.

    When `print_ready=True`, runs an additional trimesh pipeline after the
    marching-cubes + Taubin pass: connected-component filtering, hole-filling,
    light Humphrey smoothing, and (if `fast_simplification` is
    available) optional quadric decimation when `decimate_ratio < 1.0`.
    """
    try:
        from skimage import measure
        import scipy.ndimage as ndi
    except ImportError:
        return None

    try:
        grid = np.asarray(density, dtype=float)
        if grid.ndim != 3 or min(grid.shape) < 1:
            return None

        grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(grid)) <= 0.0:
            return None
        grid = _regularize_extruded_density(grid, extrusion_axis)

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        extrusion_ax = axis_map.get(str(extrusion_axis or '').strip().lower())
        if extrusion_ax is not None:
            in_plane = [i for i in range(3) if i != extrusion_ax]
            min_plane_dim = max(1, min(int(grid.shape[i]) for i in in_plane))
            upsample = int(np.clip(np.ceil(240.0 / min_plane_dim), 1, 8))
            zoom_factors = np.ones(3, dtype=float)
            zoom_factors[in_plane] = float(upsample)
        else:
            min_dim = max(1, min(grid.shape))
            upsample = int(np.clip(np.ceil(48.0 / min_dim), 1, 12))
            zoom_factors = np.full(3, float(upsample), dtype=float)

        while np.any(zoom_factors > 1.0) and np.prod(np.asarray(grid.shape) * zoom_factors) > 2_500_000:
            largest = int(np.argmax(zoom_factors))
            zoom_factors[largest] = max(1.0, zoom_factors[largest] - 1.0)

        if np.any(zoom_factors > 1.0):
            field = ndi.zoom(
                grid,
                zoom=tuple(float(v) for v in zoom_factors),
                order=3,
                mode='nearest',
                grid_mode=True,
            )
        else:
            field = grid.copy()

        origin, cell = _voxel_origin_cell(tuple(grid.shape), bounds)
        spacing = cell / zoom_factors

        field = np.clip(field, 0.0, 1.0)
        source_field = _resample_source_mask(source_mask, tuple(field.shape))
        sigma = 0.20 if float(np.max(zoom_factors)) <= 1.0 else 0.35
        field = ndi.gaussian_filter(field, sigma=sigma)
        if source_field is not None:
            field[~source_field] = 0.0
        # Source masks are voxel approximations of the input body; passive keep
        # and cut regions are analytic constraints and must be re-imposed after
        # that coarse clipping so round holes/bosses stay round in recovery.
        field = _apply_passive_density_regions(
            field,
            solid_boxes=solid_boxes,
            void_boxes=void_boxes,
            solid_cylinders=solid_cylinders,
            void_cylinders=void_cylinders,
            bounds=bounds,
            cell=cell,
            spacing=spacing,
        )
        pad = max(3, min(10, int(np.ceil(float(np.max(zoom_factors))))))
        field = np.pad(field, pad_width=pad, mode='constant', constant_values=0.0)

        level = float(np.clip(cutoff, 1e-6, 0.999999))
        mask = field >= level
        if not np.any(mask):
            nonzero = field[field > 0.0]
            if nonzero.size == 0:
                return None
            mask = field >= float(np.percentile(nonzero, 75.0))
        if np.all(mask):
            return None

        # Build a signed iso-field directly from the filtered physical density:
        # negative is material, positive is void.  Passive shapes are applied
        # before marching cubes so circular holes do not need post-hoc vertex
        # snapping, which can fold triangles near tight bolt holes.
        iso_field = level - field
        mc_level = 0.0
        passive_shapes_present = bool(solid_cylinders or void_cylinders or solid_boxes or void_boxes)
        analytic_cylinder_iso = False
        if passive_shapes_present:
            iso_field = _apply_passive_cylinder_sdf(
                iso_field,
                pad=pad,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
                blend_radius=blend_radius,
            )
            analytic_cylinder_iso = True

        # Prefer the filtered physical density field itself for the visible
        # boundary.  The previous binary-mask -> distance-transform SDF route
        # made thin topology webs look like terraced contour plots because the
        # signed-distance bands were smoothed after thresholding.
        if not (float(np.min(iso_field)) < mc_level < float(np.max(iso_field))):
            outside = ndi.distance_transform_edt(~mask, sampling=tuple(float(v) for v in spacing))
            inside = ndi.distance_transform_edt(mask, sampling=tuple(float(v) for v in spacing))
            iso_field = outside - inside
            iso_field = ndi.gaussian_filter(iso_field, sigma=0.35)
            if passive_shapes_present:
                iso_field = _apply_passive_cylinder_sdf(
                    iso_field,
                    pad=pad,
                    solid_boxes=solid_boxes,
                    void_boxes=void_boxes,
                    solid_cylinders=solid_cylinders,
                    void_cylinders=void_cylinders,
                    bounds=bounds,
                    cell=cell,
                    spacing=spacing,
                    blend_radius=blend_radius,
                )
                analytic_cylinder_iso = True
            mc_level = 0.0
            if not (float(np.min(iso_field)) < mc_level < float(np.max(iso_field))):
                return None

        verts, faces, _, _ = measure.marching_cubes(
            iso_field,
            level=float(mc_level),
            spacing=tuple(float(v) for v in spacing),
            gradient_direction='ascent',
        )
        if len(verts) == 0 or len(faces) == 0:
            return None

        surface_origin = origin + 0.5 * spacing - float(pad) * spacing
        verts = verts + surface_origin
        active_shapes = ()
        if passive_shapes_present:
            active_shapes = _convert_legacy_to_physical_shapes(
                bounds,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
            )
        verts = _taubin_smooth_surface(
            verts, faces,
            iterations=2,
            shapes=active_shapes,
            tolerance=float(np.max(spacing)) * 3.0,
        )
        verts = _project_extruded_planes(
            verts, bounds, extrusion_axis,
            tolerance=float(np.max(spacing)) * 2.5,
        )
        if passive_shapes_present and not analytic_cylinder_iso:
            shapes = _convert_legacy_to_physical_shapes(
                bounds,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
            )
            verts = _project_passive_shapes_surfaces(
                verts,
                shapes,
                tolerance=float(np.max(spacing)) * 3.0,
            )
        if bounds is not None:
            bound_min = np.asarray(bounds[0], dtype=float)[:3]
            bound_max = np.asarray(bounds[1], dtype=float)[:3]
            verts = np.clip(verts, bound_min, bound_max)


        # Print-ready: trimesh pipeline (hole-fill, Humphrey, optional decimate).
        if print_ready:
            improved = _enhanced_mesh_postprocess(
                verts, faces,
                decimate_ratio=float(decimate_ratio),
                smoothing_iterations=2,
            )
            if improved is not None and len(improved.get('faces', [])) > 0:
                improved['vertices'] = _project_extruded_planes(
                    improved['vertices'], bounds, extrusion_axis,
                    tolerance=float(np.max(spacing)) * 2.5,
                )
                if passive_shapes_present and not analytic_cylinder_iso:
                    shapes = _convert_legacy_to_physical_shapes(
                        bounds,
                        solid_boxes=solid_boxes,
                        void_boxes=void_boxes,
                        solid_cylinders=solid_cylinders,
                        void_cylinders=void_cylinders,
                    )
                    improved['vertices'] = _project_passive_shapes_surfaces(
                        improved['vertices'],
                        shapes,
                        tolerance=float(np.max(spacing)) * 3.0,
                    )
                if bounds is not None:
                    improved['vertices'] = np.clip(
                        improved['vertices'], bound_min, bound_max
                    )
                return improved

        return {
            'vertices': np.asarray(verts, dtype=float),
            'faces': np.asarray(faces, dtype=int),
        }
    except Exception:
        logger.exception("Voxel shape recovery failed")
        return None


