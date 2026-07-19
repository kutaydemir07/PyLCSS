# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Boundary conditions, load cases and manufacturing constraints for the voxel
topology optimiser, plus the parsers that build them from node properties."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

_SUPPORT_TO_DOFS: Dict[str, List[int]] = {
    'None':    [],
    'Fix X':   [0],
    'Fix Y':   [1],
    'Fix Z':   [2],
    'Fix XY':  [0, 1],
    'Fix YZ':  [1, 2],
    'Fix XZ':  [0, 2],
    'Fix XYZ': [0, 1, 2],
}


@dataclass
class LoadCase:
    """One independent loading scenario contributing weight·compliance to the objective."""
    name:               str   = "LC1"
    weight:             float = 1.0
    # Point forces: list of (ix_frac, iy_frac, iz_frac, fx, fy, fz), fracs ∈ [0,1]
    point_forces:       List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    # Localized patch forces over a fractional node box:
    # (x0, x1, y0, y1, z0, z1, fx_total, fy_total, fz_total)
    box_forces:         List[Tuple[float, float, float, float, float, float, float, float, float]] = field(default_factory=list)
    # Distributed face force: list of (face, fx_per, fy_per, fz_per)
    distributed_forces: List[Tuple[str, float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = str(self.name or "Load case")
        self.weight = float(self.weight)
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(f"Load case {self.name!r} weight must be finite and positive.")

        has_nonzero_force = False
        for point in self.point_forces:
            if len(point) != 6 or not all(math.isfinite(float(v)) for v in point):
                raise ValueError(f"Load case {self.name!r} has an invalid point force.")
            if not all(0.0 <= float(v) <= 1.0 for v in point[:3]):
                raise ValueError(f"Load case {self.name!r} point coordinates must be in [0, 1].")
            has_nonzero_force |= any(abs(float(v)) > 1e-15 for v in point[3:])
        for box in self.box_forces:
            if len(box) != 9 or not all(math.isfinite(float(v)) for v in box):
                raise ValueError(f"Load case {self.name!r} has an invalid box force.")
            if not all(0.0 <= float(v) <= 1.0 for v in box[:6]):
                raise ValueError(f"Load case {self.name!r} box coordinates must be in [0, 1].")
            if any(float(box[i]) > float(box[i + 1]) for i in (0, 2, 4)):
                raise ValueError(f"Load case {self.name!r} box intervals must be ordered.")
            has_nonzero_force |= any(abs(float(v)) > 1e-15 for v in box[6:])
        allowed_faces = {"left", "right", "top", "bottom", "front", "back"}
        for face, fx, fy, fz in self.distributed_forces:
            values = (fx, fy, fz)
            if str(face).lower() not in allowed_faces or not all(
                math.isfinite(float(v)) for v in values
            ):
                raise ValueError(f"Load case {self.name!r} has an invalid distributed force.")
            has_nonzero_force |= any(abs(float(v)) > 1e-15 for v in values)
        if not has_nonzero_force:
            raise ValueError(f"Load case {self.name!r} must contain a non-zero force.")


@dataclass
class VoxelBC:
    """Fully configurable boundary conditions for a 3-D voxel domain.

    ix/iy/iz coordinates are INTEGER node indices in the domain grid:
        ix : 0 … nelx   (left  → right,  X)
        iy : 0 … nely   (bottom → top,   Y)
        iz : 0 … nelz   (front → back,   Z)
    """
    fixed_left_face_dofs:   List[int] = field(default_factory=list)  # ix=0 face
    fixed_right_face_dofs:  List[int] = field(default_factory=list)  # ix=nelx face
    fixed_top_face_dofs:    List[int] = field(default_factory=list)  # iy=nely face
    fixed_bottom_face_dofs: List[int] = field(default_factory=list)  # iy=0 face
    fixed_front_face_dofs:  List[int] = field(default_factory=list)  # iz=0 face
    fixed_back_face_dofs:   List[int] = field(default_factory=list)  # iz=nelz face
    # Localized support boxes in fractional node coordinates:
    # (x_min, x_max, y_min, y_max, z_min, z_max, dofs)
    fixed_boxes: List[Tuple[float, float, float, float, float, float, List[int]]] = field(default_factory=list)
    # Multi-load-case objective: weighted sum of compliances.  If empty, legacy
    # point_forces / distributed_forces below are normalised into a single LC.
    load_cases: List[LoadCase] = field(default_factory=list)
    # Non-design regions in fractional coordinates (x0, x1, y0, y1, z0, z1).
    # Voxels inside solid_boxes are clamped to ρ=1, voxels inside void_boxes to ρ≈0.
    solid_boxes: List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    void_boxes:  List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    # Cylindrical passive regions:
    # (axis, center_a, center_b, axis_min, axis_max, radius), all fractional.
    solid_cylinders: List[Tuple[str, float, float, float, float, float]] = field(default_factory=list)
    void_cylinders:  List[Tuple[str, float, float, float, float, float]] = field(default_factory=list)
    # Legacy single-LC fields — used only when load_cases is empty.
    point_forces: List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    box_forces:   List[Tuple[float, float, float, float, float, float, float, float, float]] = field(default_factory=list)
    distributed_forces: List[Tuple[str, float, float, float]] = field(default_factory=list)


@dataclass
class ManufacturingConstraints:
    """Geometry projections applied after every OC/MMA update.

    Each constraint is a *projection* on the density field, applied in order
    (symmetry → extrusion → overhang → max_member_size → pattern_repeat) and
    followed by passive re-clamping.  Sensitivities are not back-propagated
    through the projections; this is the pragmatic compromise most density-
    based industrial codes make and is sufficient for engineering use.
    """
    # Symmetry planes through the domain centre. Subset of {'x','y','z'} as a
    # single string (e.g. 'y', 'xy', 'xyz'); '' or 'none' disables.
    symmetry:            str = 'none'
    # Force the density to be uniform along this axis (extruded part).  One of
    # {'x','y','z'} or 'none'.
    extrusion:           str = 'none'
    # AM build direction; voxels can only have material where they are
    # supported by a 3×3 neighbourhood of denser voxels in the layer below.
    # One of {'+x','-x','+y','-y','+z','-z'} or 'none'.
    overhang_build_axis: str = 'none'
    # Maximum member size as a radius in *voxels*.  Where the local mean
    # density in a (2r+1)³ window exceeds `max_member_threshold`, voxels are
    # scaled down so the local mean equals the threshold.  0 disables.
    max_member_size_voxels: float = 0.0
    max_member_threshold:   float = 0.6
    # N-fold rotational pattern around an axis through the domain centre.
    # 1 disables.  N=2 and N=4 are common (180°/90° rotations); N=3 and N=6
    # work too but use bilinear interpolation under the hood.
    pattern_repeat: int = 1
    pattern_axis:   str = 'y'  # 'x' | 'y' | 'z'

    def __post_init__(self) -> None:
        self.symmetry = str(self.symmetry or "none").lower()
        self.extrusion = str(self.extrusion or "none").lower()
        self.overhang_build_axis = str(self.overhang_build_axis or "none").lower()
        self.pattern_axis = str(self.pattern_axis or "y").lower()
        if self.symmetry != "none" and not set(self.symmetry) <= {"x", "y", "z"}:
            raise ValueError("Symmetry must be None or a combination of X, Y, and Z.")
        if self.extrusion not in {"none", "x", "y", "z"}:
            raise ValueError("Extrusion axis must be None, X, Y, or Z.")
        if self.overhang_build_axis not in {
            "none", "+x", "-x", "+y", "-y", "+z", "-z",
        }:
            raise ValueError("Overhang build axis is invalid.")
        self.max_member_size_voxels = float(self.max_member_size_voxels)
        self.max_member_threshold = float(self.max_member_threshold)
        if not math.isfinite(self.max_member_size_voxels) or self.max_member_size_voxels < 0.0:
            raise ValueError("Maximum member size must be finite and non-negative.")
        if not math.isfinite(self.max_member_threshold) or not 0.0 < self.max_member_threshold <= 1.0:
            raise ValueError("Maximum member threshold must be in (0, 1].")
        self.pattern_repeat = int(self.pattern_repeat)
        if self.pattern_repeat < 1 or self.pattern_axis not in {"x", "y", "z"}:
            raise ValueError("Pattern repeat must be at least 1 with axis X, Y, or Z.")




def _parse_support(label: str) -> List[int]:
    return _SUPPORT_TO_DOFS.get(label, [])


def _parse_support_region_dofs(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value if int(v) in (0, 1, 2)]

    text = str(value or "").strip()
    if not text:
        return []
    if text in _SUPPORT_TO_DOFS:
        return _SUPPORT_TO_DOFS[text]

    axes = text.upper().replace("FIX", "").replace(" ", "")
    return [idx for axis, idx in (("X", 0), ("Y", 1), ("Z", 2)) if axis in axes]


def _parse_region_boxes(
    value: Any,
    field_name: str,
) -> List[Tuple[float, float, float, float, float, float]]:
    """Parse a JSON list of {"x":[a,b],"y":[a,b],"z":[a,b]} into fractional boxes."""
    text = str(value or "").strip()
    if not text:
        return []
    try:
        regions = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {field_name} JSON: {exc}") from exc
    if not isinstance(regions, list):
        raise ValueError(f"{field_name} must be a JSON list")

    boxes: List[Tuple[float, float, float, float, float, float]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_type = str(region.get('type') or region.get('shape') or '').strip().lower()
        if region_type in {'cylinder', 'cylindrical', 'circle', 'circular', 'hole'}:
            continue
        try:
            x0, x1 = sorted(float(v) for v in region.get('x', [0.0, 0.0]))
            y0, y1 = sorted(float(v) for v in region.get('y', [0.0, 0.0]))
            z0, z1 = sorted(float(v) for v in region.get('z', [0.0, 1.0]))
        except Exception as exc:
            raise ValueError(f"{field_name} contains an invalid box interval") from exc
        values = (x0, x1, y0, y1, z0, z1)
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"{field_name} box coordinates must be finite fractions in [0, 1]")
        boxes.append(values)
    return boxes


def _parse_region_cylinders(
    value: Any,
    field_name: str,
) -> List[Tuple[str, float, float, float, float, float]]:
    """Parse cylindrical passive regions from the same JSON region list.

    Supported examples:
        {"type":"cylinder","axis":"z","center":[0.5,0.5],"radius":0.2,"z":[0,1]}
        {"type":"hole","center":[0.85,0.15],"r":0.04}

    All coordinates are fractional in the voxel domain. For axis='z', center is
    [x, y] and z gives the through-axis span; axis='x' uses center [y, z], and
    axis='y' uses center [x, z].
    """
    text = str(value or "").strip()
    if not text:
        return []
    try:
        regions = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {field_name} JSON: {exc}") from exc
    if not isinstance(regions, list):
        raise ValueError(f"{field_name} must be a JSON list")

    cylinders: List[Tuple[str, float, float, float, float, float]] = []
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_type = str(region.get('type') or region.get('shape') or '').strip().lower()
        if region_type not in {'cylinder', 'cylindrical', 'circle', 'circular', 'hole'}:
            continue

        axis = str(region.get('axis') or 'z').strip().lower()
        if axis not in {'x', 'y', 'z'}:
            raise ValueError(f"{field_name} cylinder axis must be X, Y, or Z")

        try:
            radius = float(region.get('radius', region.get('r', region.get('diameter', 0.0))))
            if 'diameter' in region and 'radius' not in region and 'r' not in region:
                radius *= 0.5
        except Exception:
            radius = 0.0
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"{field_name} cylinder radius must be finite and positive")

        center = region.get('center', [0.5, 0.5])
        if not isinstance(center, (list, tuple)):
            center = [0.5, 0.5]

        def _interval(name: str) -> Tuple[float, float]:
            raw = region.get(name, [0.0, 1.0])
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                return float(raw[0]), float(raw[1])
            return 0.0, 1.0

        if axis == 'z':
            c0 = float(region.get('cx', center[0] if len(center) > 0 else 0.5))
            c1 = float(region.get('cy', center[1] if len(center) > 1 else 0.5))
            lo, hi = _interval('z')
        elif axis == 'x':
            c0 = float(region.get('cy', center[0] if len(center) > 0 else 0.5))
            c1 = float(region.get('cz', center[1] if len(center) > 1 else 0.5))
            lo, hi = _interval('x')
        else:
            c0 = float(region.get('cx', center[0] if len(center) > 0 else 0.5))
            c1 = float(region.get('cz', center[1] if len(center) > 1 else 0.5))
            lo, hi = _interval('y')

        lo, hi = sorted((lo, hi))
        values = (c0, c1, lo, hi)
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
            raise ValueError(
                f"{field_name} cylinder center and axial range must be fractions in [0, 1]"
            )
        cylinders.append((axis, c0, c1, lo, hi, radius))
    return cylinders


def _parse_load_cases(value: Any) -> List["LoadCase"]:
    """Parse a JSON list of load-case descriptors into LoadCase instances.

    Accepted schema per entry:
        {
          "name":   "LC1",          # optional
          "weight": 1.0,            # optional, default 1.0
          "point_forces": [
            {"x":1.0,"y":0.5,"z":0.5,"fx":0,"fy":-1,"fz":0}
          ],
          "box_forces": [
            {"x":[0.96,1.0],"y":[0.45,0.55],"z":[0,1],"fx":0,"fy":-1,"fz":0}
          ],
          "distributed_forces": [
            {"face":"right","fx":0,"fy":-1,"fz":0}
          ]
        }
    """
    text = str(value or "").strip()
    if not text:
        return []
    try:
        items = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid load_cases JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("load_cases must be a JSON list")

    result: List[LoadCase] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        name   = str(item.get('name') or f"LC{i + 1}")
        weight = float(item.get('weight', 1.0))

        pts: List[Tuple[float, float, float, float, float, float]] = []
        for pf in item.get('point_forces') or []:
            if not isinstance(pf, dict):
                continue
            pts.append((
                float(pf.get('x',  pf.get('ix_frac', 0.5))),
                float(pf.get('y',  pf.get('iy_frac', 0.5))),
                float(pf.get('z',  pf.get('iz_frac', 0.5))),
                float(pf.get('fx', 0.0)),
                float(pf.get('fy', 0.0)),
                float(pf.get('fz', 0.0)),
            ))

        boxes: List[Tuple[float, float, float, float, float, float, float, float, float]] = []
        for bf in item.get('box_forces') or []:
            if not isinstance(bf, dict):
                continue
            x0, x1 = bf.get('x', [0.0, 1.0])
            y0, y1 = bf.get('y', [0.0, 1.0])
            z0, z1 = bf.get('z', [0.0, 1.0])
            boxes.append((
                float(x0), float(x1),
                float(y0), float(y1),
                float(z0), float(z1),
                float(bf.get('fx', 0.0)),
                float(bf.get('fy', 0.0)),
                float(bf.get('fz', 0.0)),
            ))

        dists: List[Tuple[str, float, float, float]] = []
        for df in item.get('distributed_forces') or []:
            if not isinstance(df, dict):
                continue
            dists.append((
                str(df.get('face', 'right')).lower(),
                float(df.get('fx', 0.0)),
                float(df.get('fy', 0.0)),
                float(df.get('fz', 0.0)),
            ))

        result.append(LoadCase(
            name=name,
            weight=weight,
            point_forces=pts,
            box_forces=boxes,
            distributed_forces=dists,
        ))
    return result


