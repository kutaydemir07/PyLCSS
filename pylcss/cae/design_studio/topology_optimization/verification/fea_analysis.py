# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""On-demand CalculiX analysis of the completed TopOpt CAD solid."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ..geometry.lattice_cad import lattice_cad_strategy, topopt_export_formats
from ..configuration.fea_meshing import CadFeaMeshOptions
from .calculix_validation import _flatten


def topopt_fea_unavailable_reason(result):
    """Share the geometry eligibility check between the action and the solver."""
    if not isinstance(result, dict) or result.get("type") != "topopt_voxel":
        return "Run TopOpt"
    if str(result.get("physics_mode") or "structural").lower() != "structural":
        return "Unsupported study physics"
    if (result.get("multibody") or {}).get("global_joints"):
        return "Connector analysis required"
    options = result.get("structure_options")
    if "STEP" not in topopt_export_formats(result):
        return "STL lattice"
    if result.get("cad_shape") is None and result.get("shape") is None:
        return "Build CAD"
    if lattice_cad_strategy(options) == "beam" and not (
        result.get("cad_reconstruction") or {}
    ).get("fused"):
        return "Unfused beam CAD"
    return ""


def _cad_tet_mesh(
    shape,
    element_size,
    directory,
    *,
    minimum_size=0.0,
    grading=0.3,
    element_order="C3D10",
    max_elements=500_000,
):
    """Mesh the native CAD boundary with the requested tetrahedral settings."""
    from netgen.occ import OCCGeometry
    from skfem import Mesh, MeshTet2
    from pylcss.cae.design_studio.fem.quality import attach_mesh_quality
    from pylcss.cae.design_studio.fem._helpers import suppress_output

    step_path = Path(directory) / "analysis.step"
    mesh_path = Path(directory) / "analysis.msh"
    shape.exportStep(str(step_path))
    with suppress_output():
        generated = OCCGeometry(str(step_path)).GenerateMesh(
            maxh=element_size, minh=minimum_size, grading=grading
        )
        if generated.ne > max_elements:
            raise ValueError(
                f"CAD FEA mesh has {generated.ne:,} elements, exceeding the "
                f"requested limit of {max_elements:,}. Increase element size "
                "or the FEA element limit."
            )
        generated.Export(str(mesh_path), "Gmsh2 Format")
    mesh = Mesh.load(str(mesh_path))
    if element_order == "C3D10":
        mesh = MeshTet2.from_mesh(mesh)
    if not attach_mesh_quality(mesh)["solver_ready"]:
        raise ValueError("CAD FEA mesh contains invalid or collapsed elements.")
    return mesh


def run_topopt_fea(
    result,
    material,
    supports,
    loads,
    *,
    quality="Standard",
    mesh_options=None,
    progress_callback=None,
    cancel_callback=None,
):
    """Re-mesh the CAD solid and solve with the connected study definitions."""
    from pylcss.cae.solver_backends import ExternalRunConfig, run_calculix_static
    from pylcss.cae.solver_backends.selection import dict_geometries

    reason = topopt_fea_unavailable_reason(result)
    if reason:
        raise ValueError(f"CAD FEA unavailable: {reason}.")
    if not isinstance(material, dict) or not material:
        raise ValueError("Connect a Material to TopOpt before analysing FEA.")
    supports, loads = _flatten(supports), _flatten(loads)
    if not supports or not loads:
        raise ValueError("Connect supports and loads to TopOpt before analysing FEA.")
    # Node IDs belong to the old mesh and cannot identify the new CAD mesh.
    pending = [*supports, *loads]
    while pending:
        record = pending.pop()
        if isinstance(record, dict):
            if record.get("node_ids") is not None:
                raise ValueError("CAD FEA requires geometric supports and loads.")
            pending.extend(dict_geometries(record))
    if quality not in {"Standard", "Mesh Convergence"}:
        raise ValueError("Choose Standard or Mesh Convergence FEA quality.")
    options = CadFeaMeshOptions(**(mesh_options or {}))

    def cancelled():
        return cancel_callback is not None and cancel_callback()

    if cancelled():
        return None
    shape = result.get("cad_shape")
    if shape is None:
        shape = result.get("shape")
    shape = shape.val() if hasattr(shape, "val") else shape
    if not shape.isValid() or not shape.Solids() or shape.Volume() <= 0:
        raise ValueError("CAD FEA requires a valid solid CAD body.")
    bounds = shape.BoundingBox()
    spans = np.array([bounds.xlen, bounds.ylen, bounds.zlen], dtype=float)
    size, minimum_size = options.resolve_sizes(spans)
    levels = []
    analysis = None
    with TemporaryDirectory(prefix="pylcss_topopt_cad_fea_") as directory:
        for level in range(2 if quality == "Mesh Convergence" else 1):
            if cancelled():
                return None
            level_size = size / (2**level)
            level_minimum = minimum_size / (2**level)
            if progress_callback is not None:
                progress_callback(
                    f"Meshing CAD, level {level + 1} (maximum size {level_size:.4g} mm)"
                )
            mesh = _cad_tet_mesh(
                shape,
                level_size,
                directory,
                minimum_size=level_minimum,
                grading=options.mesh_grading,
                element_order=options.element_order,
                max_elements=options.max_elements,
            )
            if cancelled():
                return None
            if progress_callback is not None:
                progress_callback(
                    f"Solving CalculiX, level {level + 1} "
                    f"({mesh.p.shape[1]:,} nodes, {mesh.t.shape[1]:,} elements)"
                )
            analysis = run_calculix_static(
                mesh,
                material,
                supports,
                loads,
                ExternalRunConfig(
                    run_solver=True,
                    job_name=f"pylcss_topopt_cad_fea_L{level + 1}",
                    cancel_callback=cancel_callback,
                ),
            )
            if cancelled():
                return None
            if analysis.get("mesh") is None or any(
                analysis.get(key) is None
                or np.asarray(analysis[key]).size == 0
                or not np.all(np.isfinite(analysis[key]))
                for key in ("stress", "displacement")
            ):
                raise RuntimeError(
                    analysis.get("message")
                    or "CalculiX produced no stress/displacement results."
                )
            levels.append(
                {
                    "refine": level + 1,
                    "n_nodes": int(mesh.p.shape[1]),
                    "n_elements": int(mesh.t.shape[1]),
                    "maximum_element_size": size / (2**level),
                    "minimum_element_size": level_minimum,
                    "compliance": analysis.get("compliance"),
                    "peak_von_mises": analysis.get(
                        "peak_stress_nodal", analysis.get("max_stress_gauss")
                    ),
                    "solved": True,
                }
            )
    change = None
    if len(levels) == 2:
        previous, current = (entry["compliance"] for entry in levels)
        if (
            previous is not None
            and current is not None
            and np.isfinite([previous, current]).all()
            and current > 0
        ):
            change = abs(current - previous) / current
    analysis.update(
        analysis_target="CAD solid",
        source="topopt_cad_fea",
        element_order=options.element_order,
        mesh_settings={**options.as_dict(), "resolved_element_size_mm": size},
        convergence_study={
            "element_order": options.element_order,
            "levels": levels,
            "tolerance": 0.05,
            "compliance_rel_change": change,
            "converged": change <= 0.05 if change is not None else None,
        },
    )
    return analysis
