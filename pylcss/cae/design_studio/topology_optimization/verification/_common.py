# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Shared geometry, boundary definitions and meshing for CAD verification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def finite_nonnegative(value, *, name, positive=False):
    """Validate a setting without replacing explicit zero with defaults."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(number) or number < 0 or (positive and number == 0):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {requirement}.")
    return number


def validation_integer(value, *, name, minimum=1):
    """Require an exact integer with no arbitrary upper ceiling."""
    number = finite_nonnegative(value, name=name)
    if not number.is_integer() or number < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(number)


def flatten_records(values):
    """Flatten connected lists without unpacking boundary dictionaries."""
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [item for value in values for item in flatten_records(value)]
    return [values]


def validate_boundary_definitions(material, supports, loads, *, context):
    """Require study definitions that can be mapped onto a newly generated mesh."""
    from pylcss.cae.solver_backends.selection import dict_geometries
    from pylcss.cae.solver_backends.validation import record_list

    if not isinstance(material, Mapping) or not material:
        raise ValueError(f"{context} requires a material definition.")
    supports = record_list(flatten_records(supports), label="Supports")
    loads = record_list(flatten_records(loads), label="Loads")
    if not supports or not loads:
        raise ValueError(f"{context} requires supports and loads.")
    pending = [*supports, *loads]
    visited = set()
    while pending:
        record = pending.pop()
        if isinstance(record, Mapping) and id(record) not in visited:
            visited.add(id(record))
            if record.get("node_ids") is not None:
                raise ValueError(
                    f"{context} requires geometric supports and loads or coordinate "
                    "conditions; old mesh node IDs cannot be reused after remeshing."
                )
            pending.extend(dict_geometries(record))
    return dict(material), supports, loads


def validated_cad_shape(shape):
    """Accept actual solid geometry, independently of reconstruction metadata.

    Multiple disjoint solids are allowed. Overlapping members must be fused so
    that a volume mesh cannot double-count material or omit their connectivity.
    Bounding boxes discard disjoint pairs before computing exact intersections.
    """
    shape = shape.val() if hasattr(shape, "val") else shape
    if shape is None or not shape.isValid() or not shape.Solids():
        raise ValueError("CAD analysis requires a valid solid CAD body.")
    volume = float(shape.Volume())
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("CAD analysis requires finite positive material volume.")
    solids = sorted(
        (
            (solid.BoundingBox(), solid, float(solid.Volume()))
            for solid in shape.Solids()
        ),
        key=lambda entry: entry[0].xmin,
    )
    for index, (bounds, solid, solid_volume) in enumerate(solids):
        for other_index in range(index + 1, len(solids)):
            other_bounds, other, other_volume = solids[other_index]
            if other_bounds.xmin >= bounds.xmax:
                break
            if (
                other_bounds.ymin >= bounds.ymax
                or bounds.ymin >= other_bounds.ymax
                or other_bounds.zmin >= bounds.zmax
                or bounds.zmin >= other_bounds.zmax
            ):
                continue
            overlap = float(solid.intersect(other).Volume())
            if not np.isfinite(overlap) or overlap > 1e-9 * min(
                solid_volume, other_volume
            ):
                raise ValueError(
                    "CAD analysis requires non-overlapping material solids. "
                    "Fuse overlapping members before analysis."
                )
    return shape


def tet_mesh_from_geometry(
    geometry,
    element_size,
    directory,
    *,
    name="analysis",
    minimum_size=0.0,
    grading=0.3,
    element_order="C3D10",
    max_elements=500_000,
):
    """Generate and quality-check a Netgen volume mesh for either STL or CAD."""
    from skfem import Mesh, MeshTet2
    from pylcss.cae.design_studio.fem._helpers import suppress_output
    from pylcss.cae.design_studio.fem.quality import attach_mesh_quality

    mesh_path = Path(directory) / f"{name}.msh"
    with suppress_output():
        generated = geometry.GenerateMesh(
            maxh=element_size,
            minh=minimum_size,
            grading=grading,
        )
        if generated.ne > max_elements:
            raise ValueError(
                f"Validation mesh has {generated.ne:,} elements, exceeding the "
                f"requested limit of {max_elements:,}. Increase element size "
                "or the element limit."
            )
        generated.Export(str(mesh_path), "Gmsh2 Format")
    mesh = Mesh.load(str(mesh_path))
    if element_order == "C3D10":
        mesh = MeshTet2.from_mesh(mesh)
    if not attach_mesh_quality(mesh)["solver_ready"]:
        raise ValueError("Validation mesh contains invalid or collapsed elements.")
    return mesh


def cad_tet_mesh(shape, element_size, directory, **settings):
    """Export a solid CAD boundary and mesh it with the shared volume mesher."""
    from netgen.occ import OCCGeometry

    step_path = Path(directory) / "analysis.step"
    shape.exportStep(str(step_path))
    return tet_mesh_from_geometry(
        OCCGeometry(str(step_path)),
        element_size,
        directory,
        **settings,
    )
