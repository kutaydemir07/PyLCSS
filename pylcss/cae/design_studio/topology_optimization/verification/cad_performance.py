# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Compare recovered and reconstructed geometry with the same structural solve.

Both geometries are independently volume-meshed and analyzed with quadratic
tetrahedra, the same material and the same boundary definitions. A geometric
fit or a successful voxel solve does not establish CAD performance preservation.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from ._common import (
    finite_nonnegative,
    tet_mesh_from_geometry,
    validate_boundary_definitions,
    validated_cad_shape,
    validation_integer,
)


def compare_compliance_levels(
    reference,
    reconstructed,
    *,
    maximum_relative_change=0.05,
    convergence_tolerance=0.03,
):
    """Require finite solved responses and convergence on both geometries."""
    maximum_relative_change = finite_nonnegative(
        maximum_relative_change, name="maximum_relative_change"
    )
    convergence_tolerance = finite_nonnegative(
        convergence_tolerance, name="convergence_tolerance"
    )

    def valid(levels):
        try:
            return len(levels) >= 2 and all(
                np.isfinite(level["compliance"]) and level["compliance"] > 0
                for level in levels
            )
        except (KeyError, TypeError, ValueError):
            return False

    if not valid(reference) or not valid(reconstructed):
        return {
            "passed": False,
            "reason": "Two solved positive-compliance mesh levels are required for each geometry",
        }
    changes = [
        abs(levels[-1]["compliance"] - levels[-2]["compliance"])
        / levels[-1]["compliance"]
        for levels in (reference, reconstructed)
    ]
    deviation = (
        abs(reconstructed[-1]["compliance"] - reference[-1]["compliance"])
        / reference[-1]["compliance"]
    )
    converged = all(change <= convergence_tolerance for change in changes)
    passed = converged and deviation <= maximum_relative_change
    return {
        "passed": bool(passed),
        "relative_compliance_change": float(deviation),
        "reference_mesh_relative_change": float(changes[0]),
        "cad_mesh_relative_change": float(changes[1]),
        "mesh_converged": bool(converged),
        "reason": (
            "Passed"
            if passed
            else "Structural meshes did not converge"
            if not converged
            else f"Compliance changed by {deviation:.1%}, exceeding {maximum_relative_change:.1%}"
        ),
    }


def validate_cad_performance(
    reference_mesh,
    shape,
    *,
    material,
    constraints,
    loads,
    mesh_size=None,
    convergence_levels=2,
    maximum_relative_compliance_change=0.05,
    convergence_tolerance=0.03,
    maximum_stress=None,
    maximum_displacement=None,
    max_validation_elements=500000,
    external_solver_path="",
    external_timeout_s=900.0,
    cancel_callback=None,
):
    """Run real CalculiX comparisons; unavailable solves cannot yield a pass."""
    from netgen.stl import STLGeometry
    from netgen.occ import OCCGeometry
    import trimesh
    from pylcss.cae.solver_backends import ExternalRunConfig, run_calculix_static
    from ..geometry.amrto_volume import measure_cad_volume

    material, constraints, loads = validate_boundary_definitions(
        material, constraints, loads, context="CAD performance comparison"
    )
    convergence_levels = validation_integer(
        convergence_levels, name="convergence_levels", minimum=2
    )
    max_validation_elements = validation_integer(
        max_validation_elements, name="max_validation_elements"
    )
    maximum_relative_compliance_change = finite_nonnegative(
        maximum_relative_compliance_change, name="maximum_relative_compliance_change"
    )
    convergence_tolerance = finite_nonnegative(
        convergence_tolerance, name="convergence_tolerance"
    )
    external_timeout_s = finite_nonnegative(
        external_timeout_s, name="external_timeout_s", positive=True
    )
    if maximum_stress is not None:
        maximum_stress = finite_nonnegative(maximum_stress, name="maximum_stress")
    if maximum_displacement is not None:
        maximum_displacement = finite_nonnegative(
            maximum_displacement, name="maximum_displacement"
        )
    shape = validated_cad_shape(shape)
    cad_volume, _ = measure_cad_volume(shape)
    vertices, faces = reference_mesh
    source = trimesh.Trimesh(vertices, faces, process=False)
    if (
        not source.is_watertight
        or not source.is_winding_consistent
        or source.volume <= 0
    ):
        raise ValueError(
            "CAD performance comparison requires a closed oriented reference surface"
        )
    size = (
        float(mesh_size)
        if mesh_size is not None
        else float(np.linalg.norm(source.extents)) / 10
    )
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Mechanical mesh size must be finite and positive")
    report = {
        "status": "failed",
        "method": "Recovered STL and reconstructed STEP, Netgen C3D10 / CalculiX",
        "maximum_relative_compliance_change": maximum_relative_compliance_change,
        "convergence_tolerance": convergence_tolerance,
        "same_material_and_boundary_conditions": True,
        "reference_levels": [],
        "cad_levels": [],
        "passed": False,
    }
    with TemporaryDirectory(prefix="pylcss_cad_validation_") as directory:
        folder = Path(directory)
        source.export(folder / "reference.stl")
        shape.exportStep(str(folder / "reconstructed.step"))
        for label, geometry in (
            ("reference", STLGeometry(str(folder / "reference.stl"))),
            ("cad", OCCGeometry(str(folder / "reconstructed.step"))),
        ):
            for level in range(convergence_levels):
                if cancel_callback is not None and cancel_callback():
                    raise RuntimeError("CAD performance comparison cancelled")
                length = size / (2**level)
                try:
                    mesh = tet_mesh_from_geometry(
                        geometry,
                        length,
                        folder,
                        name=f"{label}_{level}",
                        max_elements=max_validation_elements,
                        reference_volume=cad_volume if label == "cad" else None,
                    )
                except ValueError as exc:
                    report["reason"] = f"{label} mesh level {level + 1}: {exc}"
                    return report
                if cancel_callback is not None and cancel_callback():
                    raise RuntimeError("CAD performance comparison cancelled")
                result = run_calculix_static(
                    mesh,
                    material,
                    constraints,
                    loads,
                    ExternalRunConfig(
                        executable=external_solver_path or None,
                        work_dir=str(folder / f"{label}_{level}"),
                        run_solver=True,
                        timeout_s=external_timeout_s,
                        job_name=f"cad_{label}_L{level + 1}",
                        cancel_callback=cancel_callback,
                    ),
                )
                if cancel_callback is not None and cancel_callback():
                    raise RuntimeError("CAD performance comparison cancelled")
                compliance = result.get("compliance")
                if compliance is None or not np.isfinite(compliance) or compliance <= 0:
                    report["reason"] = (
                        f"{label} validation did not produce a finite positive compliance"
                    )
                    return report
                report[f"{label}_levels"].append(
                    {
                        "level": level + 1,
                        "maximum_element_size": length,
                        "elements": int(mesh.t.shape[1]),
                        "compliance": float(compliance),
                        "maximum_displacement": result.get("peak_displacement"),
                        "maximum_stress": result.get(
                            "peak_stress_nodal", result.get("max_stress_gauss")
                        ),
                    }
                )
        report.update(
            compare_compliance_levels(
                report["reference_levels"],
                report["cad_levels"],
                maximum_relative_change=maximum_relative_compliance_change,
                convergence_tolerance=convergence_tolerance,
            )
        )
        report["maximum_stress_limit"] = maximum_stress
        report["maximum_displacement_limit"] = maximum_displacement
        for metric, limit in (
            ("maximum_stress", maximum_stress),
            ("maximum_displacement", maximum_displacement),
        ):
            if limit is not None:
                value = report["cad_levels"][-1][metric]
                if value is None or not np.isfinite(value) or value > limit:
                    report.update(
                        passed=False,
                        reason=f"Reconstructed CAD exceeds or has no solved {metric} limit ({limit:g})",
                    )
    report["status"] = "passed" if report["passed"] else "failed"
    return report
