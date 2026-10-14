# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Typed, validated study inputs for voxel topology optimization."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np


def _finite_tuple(values: Any, length: int, label: str) -> tuple[float, ...]:
    """Normalize numeric inputs before the solver uses them in arithmetic."""
    try:
        if isinstance(values, (str, bytes)):
            raise ValueError
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain {length} finite numbers.") from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain {length} finite numbers.")
    return result


def _integer(value: Any, label: str) -> int:
    """Accept integral values without silently truncating fractional values."""
    try:
        result = int(value)
        if isinstance(value, (bool, np.bool_)) or float(value) != result:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    return result


def _support_dofs(values: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    try:
        result = [_integer(value, label) for value in values]
    except TypeError as exc:
        raise ValueError(f"{label} must be a list of X, Y, and/or Z DOFs.") from exc
    if (not result and not allow_empty) or any(
        value not in (0, 1, 2) for value in result
    ):
        raise ValueError(f"{label} must constrain X, Y, and/or Z.")
    return result


def _box_coordinates(values: Any, label: str) -> tuple[float, ...]:
    result = _finite_tuple(values, 6, label)
    if not all(0.0 <= value <= 1.0 for value in result):
        raise ValueError(f"{label} coordinates must be in [0, 1].")
    if any(result[index] > result[index + 1] for index in (0, 2, 4)):
        raise ValueError(f"{label} intervals must be ordered.")
    return result


@dataclass
class JointDefinition:
    """Penalty coupling between two solid-body anchor locations.

    Voxel solids expose translational nodal DOFs. Joint centre translations
    are coupled, while bearing-spider directions encode the released rotation:
    revolute spiders retain radial/axial stiffness but release tangential
    motion, spherical spiders retain radial stiffness, and prismatic spiders
    release motion along the declared axis.
    """

    name: str = "Joint"
    kind: str = "spherical"
    anchor_a: Tuple[float, float, float] = (0.25, 0.5, 0.5)
    anchor_b: Tuple[float, float, float] = (0.75, 0.5, 0.5)
    axis: str = "x"
    relative_stiffness: float = 100.0

    def __post_init__(self) -> None:
        self.name = str(self.name or "Joint")
        self.kind = str(self.kind or "spherical").strip().lower()
        aliases = {"pin": "revolute", "ball": "spherical", "slider": "prismatic"}
        self.kind = aliases.get(self.kind, self.kind)
        if self.kind not in {"fixed", "revolute", "spherical", "prismatic"}:
            raise ValueError(
                f"Joint {self.name!r} type must be fixed, revolute, spherical, or prismatic."
            )
        self.anchor_a = _finite_tuple(self.anchor_a, 3, f"Joint {self.name!r} anchor A")
        self.anchor_b = _finite_tuple(self.anchor_b, 3, f"Joint {self.name!r} anchor B")
        if not all(0.0 <= v <= 1.0 for v in (*self.anchor_a, *self.anchor_b)):
            raise ValueError(
                f"Joint {self.name!r} anchors must contain three fractions in [0, 1]."
            )
        self.axis = str(self.axis or "x").strip().lower()
        if self.axis not in {"x", "y", "z"}:
            if self.kind in {"revolute", "prismatic"}:
                raise ValueError(f"Joint {self.name!r} axis must be X, Y, or Z.")
            # Fixed and spherical joints have no preferred axis. Keep a valid
            # placeholder for consumers that construct directional vectors.
            self.axis = "x"
        self.relative_stiffness = float(self.relative_stiffness)
        if not math.isfinite(self.relative_stiffness) or self.relative_stiffness <= 0.0:
            raise ValueError(
                f"Joint {self.name!r} relative stiffness must be finite and positive."
            )


@dataclass
class BoundaryMesh:
    """Selected CAD surface/line in fractional coordinates, with no thickness.

    Cells contain one vertex, two edge vertices, or three triangle vertices.
    Forces are integrated resultants per cell; supports constrain sampled
    boundary nodes. Neither changes the material design variables.
    """

    vertices: Any
    cells: Any
    cell_forces: Any = None
    fixed_dofs: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate geometry and conditions, including changes made by graph nodes."""
        try:
            vertices = np.asarray(self.vertices, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Boundary vertices must contain finite XYZ coordinates."
            ) from exc
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) == 0
            or not np.all(np.isfinite(vertices))
            or np.any(vertices < 0.0)
            or np.any(vertices > 1.0)
        ):
            raise ValueError(
                "Boundary vertices must be a non-empty (N, 3) array in [0, 1]."
            )
        try:
            cells = np.asarray(self.cells, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Boundary cells must contain vertex index rows.") from exc
        if (
            cells.ndim != 2
            or cells.shape[1] not in (1, 2, 3)
            or len(cells) == 0
            or not np.all(np.isfinite(cells))
            or np.any(cells != np.floor(cells))
            or np.any(cells < 0)
            or np.any(cells >= len(vertices))
        ):
            raise ValueError(
                "Boundary cells must contain 1, 2, or 3 valid integer vertex "
                "indices per row."
            )
        forces = None
        if self.cell_forces is not None:
            try:
                forces = np.asarray(self.cell_forces, dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Boundary forces must contain finite XYZ vectors."
                ) from exc
            if forces.shape != (len(cells), 3) or not np.all(np.isfinite(forces)):
                raise ValueError(
                    "Boundary forces must contain one finite XYZ vector per cell."
                )
        dofs = _support_dofs(self.fixed_dofs, "Boundary support", allow_empty=True)
        self.vertices = vertices
        self.cells = cells.astype(np.intp)
        self.cell_forces = forces
        self.fixed_dofs = dofs


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
    # Total face force: list of (face, fx_total, fy_total, fz_total).
    distributed_forces: List[Tuple[str, float, float, float]] = field(default_factory=list)
    boundary_forces: List[BoundaryMesh] = field(default_factory=list)
    boundary_supports: List[BoundaryMesh] = field(default_factory=list)
    # Optional supports that are active only in this configuration.  This makes
    # one density field serve several multibody poses without
    # pretending that all pose-dependent joints are fixed simultaneously.
    fixed_boxes: List[
        Tuple[float, float, float, float, float, float, List[int]]
    ] = field(default_factory=list)
    replace_supports: bool = False
    joints: List[JointDefinition] = field(default_factory=list)
    replace_joints: bool = False
    # Separate assembly hardware inferred from coaxial cylindrical joint
    # anchors. It is displayed/exported with the recovered assembly but is not
    # part of the SIMP design variables or material-volume constraint.
    joint_pin_cylinders: List[
        Tuple[str, float, float, float, float, float, float]
    ] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = str(self.name or "Load case")
        self.weight = float(self.weight)
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(f"Load case {self.name!r} weight must be finite and positive.")

        # Existence checks use exact zero so their meaning does not depend on
        # force units. Solver convergence tolerances are handled separately.
        has_nonzero_force = False
        for boundary in self.boundary_forces:
            boundary.validate()
            if boundary.cell_forces is None:
                raise ValueError(f"Load case {self.name!r} has a boundary without forces.")
            has_nonzero_force |= bool(np.any(boundary.cell_forces != 0.0))
        for boundary in self.boundary_supports:
            boundary.validate()
        points = []
        for point in self.point_forces:
            point = _finite_tuple(point, 6, f"Load case {self.name!r} point force")
            if not all(0.0 <= v <= 1.0 for v in point[:3]):
                raise ValueError(f"Load case {self.name!r} point coordinates must be in [0, 1].")
            points.append(point)
            has_nonzero_force |= any(v != 0.0 for v in point[3:])
        boxes = []
        for box in self.box_forces:
            box = _finite_tuple(box, 9, f"Load case {self.name!r} box force")
            _box_coordinates(box[:6], f"Load case {self.name!r} box")
            boxes.append(box)
            has_nonzero_force |= any(v != 0.0 for v in box[6:])
        allowed_faces = {"left", "right", "top", "bottom", "front", "back"}
        distributed = []
        for force in self.distributed_forces:
            if len(force) != 4:
                raise ValueError(f"Load case {self.name!r} has an invalid distributed force.")
            face = str(force[0]).strip().lower()
            values = _finite_tuple(force[1:], 3, f"Load case {self.name!r} face force")
            if face not in allowed_faces:
                raise ValueError(f"Load case {self.name!r} has an invalid distributed force.")
            distributed.append((face, *values))
            has_nonzero_force |= any(v != 0.0 for v in values)
        if not has_nonzero_force:
            raise ValueError(f"Load case {self.name!r} must contain a non-zero force.")

        supports = []
        for box in self.fixed_boxes:
            if len(box) != 7:
                raise ValueError(f"Load case {self.name!r} has an invalid support box.")
            label = f"Load case {self.name!r} support"
            values = _box_coordinates(box[:6], label)
            supports.append((*values, _support_dofs(box[6], label)))
        self.point_forces = points
        self.box_forces = boxes
        self.distributed_forces = distributed
        self.fixed_boxes = supports


@dataclass
class VoxelBC:
    """Fully configurable boundary conditions for a 3-D voxel domain.

    Positions and region bounds are fractions of the domain extent in [0, 1].
    The solver maps them to integer node indices: X runs left to right,
    Y bottom to top, and Z front to back. DOFs 0, 1, and 2 are translations
    along X, Y, and Z respectively.
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
    boundary_supports: List[BoundaryMesh] = field(default_factory=list)
    boundary_forces: List[BoundaryMesh] = field(default_factory=list)
    # Explicit operating scenarios. If empty, the global force fields below
    # are assembled into a single load case.
    load_cases: List[LoadCase] = field(default_factory=list)
    # Non-design regions in fractional coordinates (x0, x1, y0, y1, z0, z1).
    # Voxels inside solid_boxes are clamped to ρ=1, voxels inside void_boxes to ρ≈0.
    solid_boxes: List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    void_boxes:  List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    # Cylindrical passive regions:
    # (axis, center_a, center_b, axis_min, axis_max, radius), all fractional.
    solid_cylinders: List[Tuple[str, float, float, float, float, float]] = field(default_factory=list)
    void_cylinders:  List[Tuple[str, float, float, float, float, float]] = field(default_factory=list)
    # Explicit pin/shaft hardware for graph joints. These cylinders are kept
    # out of passive-solid masks so they cannot alter the topology objective or
    # material budget. Shape recovery and STEP export add them after bore cuts.
    joint_pin_cylinders: List[
        Tuple[str, float, float, float, float, float, float]
    ] = field(default_factory=list)
    # Global forces for a single scenario, including graph-connected loads.
    # Used when no explicit load cases are assembled.
    point_forces: List[Tuple[float, float, float, float, float, float]] = field(default_factory=list)
    box_forces:   List[Tuple[float, float, float, float, float, float, float, float, float]] = field(default_factory=list)
    distributed_forces: List[Tuple[str, float, float, float]] = field(default_factory=list)
    joints: List[JointDefinition] = field(default_factory=list)


@dataclass
class ManufacturingConstraints:
    """Manufacturing mappings assembled in the differentiable solver graph.

    The mappings are evaluated before finite-element assembly and their
    adjoints are included in the optimizer sensitivity. This keeps the
    optimized, analyzed, previewed, and recovered density fields consistent.
    """
    # Symmetry planes through the domain centre. Subset of {'x','y','z'} as a
    # single string (e.g. 'y', 'xy', 'xyz'); '' or 'none' disables.
    symmetry:            str = 'none'
    # Force the density to be uniform along this axis (extruded part).  One of
    # {'x','y','z'} or 'none'.
    extrusion:           str = 'none'
    # AM build direction; support is propagated through a neighborhood
    # determined by the build direction, overhang angle, and analysis grid.
    # One of {'+x','-x','+y','-y','+z','-z'} or 'none'.
    overhang_build_axis: str = 'none'
    overhang_angle_deg: float = 45.0
    # Mould withdrawal direction for a cast or moulded part. Material may not
    # be shadowed along this axis, so the design stays free of undercuts and
    # a one-sided tool can be pulled off. One of {'+x','-x','+y','-y','+z',
    # '-z'} or 'none'. This is independent of the additive overhang rule.
    pull_out_axis: str = 'none'
    # Maximum member size as a radius in *voxels*.  Where the local mean
    # density in a (2r+1)³ window exceeds `max_member_threshold`, voxel
    # densities are scaled down to discourage thick members. 0 disables.
    max_member_size_voxels: float = 0.0
    # Preferred physical maximum diameter. Zero retains the legacy voxel radius.
    max_member_size_physical: float = 0.0
    max_member_threshold:   float = 0.6
    # N-fold rotational pattern around an axis through the domain centre.
    # 1 disables.  N=2 and N=4 are common (180°/90° rotations); N=3 and N=6
    # work too but use bilinear interpolation under the hood.
    pattern_repeat: int = 1
    pattern_axis:   str = 'y'  # 'x' | 'y' | 'z'

    def __post_init__(self) -> None:
        self.symmetry = str(self.symmetry or "none").strip().lower() or "none"
        self.extrusion = str(self.extrusion or "none").strip().lower() or "none"
        self.overhang_build_axis = (
            str(self.overhang_build_axis or "none").strip().lower() or "none"
        )
        self.pull_out_axis = str(self.pull_out_axis or "none").strip().lower() or "none"
        self.pattern_axis = str(self.pattern_axis or "y").strip().lower() or "y"
        if self.symmetry != "none" and not set(self.symmetry) <= {"x", "y", "z"}:
            raise ValueError("Symmetry must be None or a combination of X, Y, and Z.")
        if self.extrusion not in {"none", "x", "y", "z"}:
            raise ValueError("Extrusion axis must be None, X, Y, or Z.")
        if self.overhang_build_axis not in {
            "none", "+x", "-x", "+y", "-y", "+z", "-z",
        }:
            raise ValueError("Overhang build axis is invalid.")
        if self.pull_out_axis not in {
            "none", "+x", "-x", "+y", "-y", "+z", "-z",
        }:
            raise ValueError("Pull-out direction is invalid.")
        self.max_member_size_voxels = float(self.max_member_size_voxels)
        self.max_member_size_physical = float(self.max_member_size_physical)
        self.max_member_threshold = float(self.max_member_threshold)
        self.overhang_angle_deg = float(self.overhang_angle_deg)
        if not math.isfinite(self.max_member_size_voxels) or self.max_member_size_voxels < 0.0:
            raise ValueError("Maximum member size must be finite and non-negative.")
        if (
            not math.isfinite(self.max_member_size_physical)
            or self.max_member_size_physical < 0.0
        ):
            raise ValueError(
                "Physical maximum member size must be finite and non-negative."
            )
        if not math.isfinite(self.max_member_threshold):
            raise ValueError("Maximum member threshold must be finite.")
        if (
            (self.max_member_size_voxels > 0.0 or self.max_member_size_physical > 0.0)
            and not 0.0 < self.max_member_threshold <= 1.0
        ):
            raise ValueError("Maximum member threshold must be in (0, 1].")
        if not math.isfinite(self.overhang_angle_deg):
            raise ValueError("Overhang angle must be finite.")
        if (
            self.overhang_build_axis != "none"
            and not 0.0 < self.overhang_angle_deg < 90.0
        ):
            raise ValueError("Overhang angle must be between 0 and 90 degrees.")
        self.pattern_repeat = _integer(self.pattern_repeat, "Pattern repeat")
        if self.pattern_repeat < 1:
            raise ValueError("Pattern repeat must be at least 1.")
        if self.pattern_repeat > 1 and self.pattern_axis not in {"x", "y", "z"}:
            raise ValueError("Pattern repeat must be at least 1 with axis X, Y, or Z.")
