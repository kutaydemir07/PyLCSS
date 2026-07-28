# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CalculiX input-deck generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from pylcss.solver_backends.base import SolverBackendError
from pylcss.solver_backends.mesh import (
    id_lines,
    load_vector,
    mesh_to_tet4,
    mesh_to_tet10,
    tet10_connectivity,
)
from pylcss.solver_backends.selection import (
    dict_geometries,
    nodes_matching_condition,
    nodes_matching_geometries,
    tet_face_sets_for_geometries,
)
from pylcss.solver_backends.validation import (
    finite_float,
    integer,
    nonnegative_float,
    positive_float,
    validate_isotropic_material,
)


def _surface_area_weights_from_mesh_selection(
    geometries: list[Any],
) -> dict[int, float]:
    """Return tributary nodal areas from mesh-selection surface triangles."""
    weights: dict[int, float] = {}
    for geom in geometries or []:
        if not isinstance(geom, dict):
            continue
        node_ids = geom.get("surface_node_ids")
        vertices = geom.get("surface_vertices")
        triangles = geom.get("surface_triangles")
        if node_ids is None or vertices is None or triangles is None:
            continue
        try:
            node_ids_arr = np.asarray(node_ids, dtype=int).reshape(-1)
            verts = np.asarray(vertices, dtype=float)
            tris = np.asarray(triangles, dtype=int)
        except (TypeError, ValueError):
            continue
        if (
            verts.ndim != 2
            or verts.shape[1] < 3
            or tris.ndim != 2
            or tris.shape[1] < 3
            or node_ids_arr.size != verts.shape[0]
        ):
            continue
        for tri in tris[:, :3]:
            if np.any(tri < 0) or np.any(tri >= len(verts)):
                continue
            a, b, c = (int(v) for v in tri)
            pa, pb, pc = verts[a, :3], verts[b, :3], verts[c, :3]
            area = 0.5 * float(np.linalg.norm(np.cross(pb - pa, pc - pa)))
            if area <= 1e-16:
                continue
            share = area / 3.0
            for local_idx in (a, b, c):
                weights[int(node_ids_arr[local_idx])] = (
                    weights.get(int(node_ids_arr[local_idx]), 0.0) + share
                )
    return weights


def _build_sets_and_step(
    mesh: Any,
    constraints: list[dict],
    loads: list[dict],
    warnings: list[str],
) -> tuple[list[str], list[str]]:
    """Build CalculiX node sets, surface sets, and step records."""
    model_lines: list[str] = []
    step_lines: list[str] = ["*STEP", "*STATIC"]

    boundary_lines: list[str] = []
    for idx, constraint in enumerate(constraints, start=1):
        geoms = dict_geometries(constraint)
        condition = str(constraint.get("condition", "") or "").strip()
        if geoms:
            node_ids = nodes_matching_geometries(mesh, geoms) + 1
        elif condition:
            node_ids = (
                nodes_matching_condition(
                    mesh, condition, warnings, label=f"Constraint {idx}"
                )
                + 1
            )
        else:
            warnings.append(f"Constraint {idx} has no selected geometry or condition.")
            continue
        if len(node_ids) == 0:
            warnings.append(f"Constraint {idx} did not match any mesh nodes.")
            continue

        set_name = f"BC_{idx}"
        model_lines.append(f"*NSET, NSET={set_name}")
        model_lines.extend(id_lines(node_ids))

        try:
            fixed_dofs = [
                integer(value, label=f"Constraint {idx} DOF")
                for value in constraint.get("fixed_dofs", [0, 1, 2])
            ]
        except TypeError as exc:
            raise SolverBackendError(
                f"Constraint {idx} fixed_dofs must be a sequence."
            ) from exc
        if not set(fixed_dofs).issubset({0, 1, 2}):
            raise SolverBackendError(
                f"Constraint {idx} fixed_dofs may only contain 0, 1, or 2."
            )
        disp = constraint.get("displacement", None)
        displacement: np.ndarray | None = None
        if disp is not None:
            try:
                displacement = np.asarray(disp, dtype=float).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise SolverBackendError(
                    f"Constraint {idx} displacement must contain three numbers."
                ) from exc
            if displacement.size != 3 or not np.all(np.isfinite(displacement)):
                raise SolverBackendError(
                    f"Constraint {idx} displacement must contain three finite numbers."
                )
        for dof_idx in fixed_dofs:
            value = 0.0 if displacement is None else float(displacement[dof_idx])
            ccx_dof = dof_idx + 1
            boundary_lines.append(f"{set_name}, {ccx_dof}, {ccx_dof}, {value:.12g}")

    if boundary_lines:
        step_lines.append("*BOUNDARY")
        step_lines.extend(boundary_lines)
    else:
        warnings.append("No boundary constraints were exported to the CalculiX deck.")

    cload_lines: list[str] = []
    dload_lines: list[str] = []
    for idx, load in enumerate(loads, start=1):
        ltype = str(load.get("type", "force") or "force").strip().lower()
        if ltype == "force":
            geoms = dict_geometries(load)
            condition = str(load.get("condition", "") or "").strip()
            if geoms:
                node_ids = nodes_matching_geometries(mesh, geoms) + 1
            elif condition:
                node_ids = (
                    nodes_matching_condition(
                        mesh, condition, warnings, label=f"Force load {idx}"
                    )
                    + 1
                )
            else:
                warnings.append(
                    f"Force load {idx} has no selected geometry or condition."
                )
                continue
            if len(node_ids) == 0:
                warnings.append(f"Force load {idx} did not match any mesh nodes.")
                continue
            force = load_vector(load)
            area_weights = _surface_area_weights_from_mesh_selection(geoms)
            if area_weights:
                total_area = float(sum(area_weights.values()))
                valid_nodes = set(int(v) for v in (node_ids - 1).tolist())
                if total_area > 1e-16:
                    for node_idx0, area in sorted(area_weights.items()):
                        if node_idx0 not in valid_nodes:
                            continue
                        nodal_force = force * (float(area) / total_area)
                        for dof_idx, value in enumerate(nodal_force, start=1):
                            if abs(float(value)) > 1e-16:
                                cload_lines.append(
                                    f"{int(node_idx0) + 1}, {dof_idx}, {float(value):.12g}"
                                )
                    warnings.append(
                        f"Force load {idx} distributed by tributary surface area "
                        f"over {len(area_weights)} selected mesh nodes."
                    )
                    continue
            nodal_force = force / max(len(node_ids), 1)
            for node_id in node_ids:
                for dof_idx, value in enumerate(nodal_force, start=1):
                    if abs(float(value)) > 1e-16:
                        cload_lines.append(
                            f"{int(node_id)}, {dof_idx}, {float(value):.12g}"
                        )
        elif ltype == "gravity":
            direction = str(load.get("direction", "-Y") or "-Y").strip().upper()
            dir_map = {
                "-X": (-1.0, 0.0, 0.0),
                "+X": (1.0, 0.0, 0.0),
                "-Y": (0.0, -1.0, 0.0),
                "+Y": (0.0, 1.0, 0.0),
                "-Z": (0.0, 0.0, -1.0),
                "+Z": (0.0, 0.0, 1.0),
            }
            if direction not in dir_map:
                raise SolverBackendError(
                    f"Unsupported gravity direction: {direction!r}."
                )
            dx, dy, dz = dir_map[direction]
            acceleration = nonnegative_float(
                load.get("accel", 9810.0),
                label="Gravity acceleration",
            )
            dload_lines.append(
                f"EALL, GRAV, {acceleration:.12g}, {dx:.1f}, {dy:.1f}, {dz:.1f}"
            )
        elif ltype == "pressure":
            geoms = dict_geometries(load)
            if not geoms:
                warnings.append(
                    f"Pressure load {idx} has no selected face geometry; skipped."
                )
                continue
            faces = tet_face_sets_for_geometries(mesh, geoms)
            if not faces:
                warnings.append(
                    f"Pressure load {idx}: no external tet faces matched the selected geometry. "
                    "Check that the selected face is a boundary of the mesh."
                )
                continue
            set_name = f"PRESS_{idx}"
            model_lines.append(f"*SURFACE, NAME={set_name}, TYPE=ELEMENT")
            for elem_id, face_id in faces:
                model_lines.append(f"{int(elem_id)}, S{int(face_id)}")
            pressure = finite_float(
                load.get("pressure", load.get("magnitude", 0.0)),
                label=f"Pressure load {idx} magnitude",
            )
            dload_lines.append(f"{set_name}, P, {pressure:.12g}")
        else:
            warnings.append(f"Unsupported CalculiX load type: {ltype}")

    if cload_lines:
        step_lines.append("*CLOAD")
        step_lines.extend(cload_lines)
    if dload_lines:
        step_lines.append("*DLOAD")
        step_lines.extend(dload_lines)
    if not cload_lines and not dload_lines:
        warnings.append("No external loads were exported to the CalculiX deck.")

    step_lines.extend(
        [
            "*NODE FILE",
            "U, RF",
            "*EL FILE",
            "S, E, ENER",
            "*END STEP",
        ]
    )
    return model_lines, step_lines


def _material_block(material: dict, *, include_plasticity: bool = False) -> list[str]:
    """Write the material model selected explicitly by the solver study.

    Yield strength remains available as an allowable in a linear study.  When
    ``include_plasticity`` is true, it instead defines this bilinear
    isotropic-hardening *PLASTIC table:

        σ_y at εp = 0      → ``yield_strength``
        σ_y at εp = ``ε*`` → ``yield_strength + tangent_modulus · ε*``

    where ``ε* = 0.10`` is a representative plastic-strain anchor.  When
    ``include_plasticity`` is false, only the elastic law is emitted,
    irrespective of the allowable yield-strength value.
    """
    validate_isotropic_material(material, require_yield=include_plasticity)
    e = positive_float(
        material.get("E", 210000.0),
        label="Material Young's modulus",
    )
    nu = finite_float(
        material.get("nu", material.get("poissons_ratio", 0.3)),
        label="Material Poisson's ratio",
    )
    rho = positive_float(
        material.get("rho", material.get("density", 7.85e-9)),
        label="Material density",
    )
    sigma_y = nonnegative_float(
        material.get("yield_strength", 0.0) or 0.0,
        label="Material yield strength",
    )
    et = nonnegative_float(
        material.get("tangent_modulus", 0.0) or 0.0,
        label="Material tangent modulus",
    )

    lines: list[str] = [
        "*MATERIAL, NAME=MAT1",
        "*ELASTIC",
        f"{e:.12g}, {nu:.12g}",
        "*DENSITY",
        f"{rho:.12g}",
    ]
    if include_plasticity:
        if sigma_y <= 0.0:
            raise SolverBackendError(
                "Nonlinear (Plastic) requires Material.yield_strength greater than zero."
            )
        eps_anchor = 0.10
        sigma_anchor = sigma_y + et * eps_anchor
        lines.extend(
            [
                "*PLASTIC, HARDENING=ISOTROPIC",
                f"{sigma_y:.12g}, 0.0",
                f"{sigma_anchor:.12g}, {eps_anchor:.6g}",
            ]
        )
    lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1")
    lines.append("")
    return lines


def _step_header(analysis_type: str) -> list[str]:
    """Return the *STEP / *STATIC header lines for the requested analysis.

    ``analysis_type``:
      - ``'Linear'``                 → ``*STEP`` + bare ``*STATIC``
      - ``'Nonlinear (Geometric)'``  → ``*STEP, NLGEOM`` + incremented ``*STATIC``
      - ``'Nonlinear (Plastic)'``    → same as Geometric (CalculiX auto-enables
                                       NLGEOM when *PLASTIC is present, but
                                       writing it explicitly keeps intent
                                       readable in the deck)
    """
    if analysis_type == "Linear":
        return ["*STEP", "*STATIC"]
    # Incremented static — 10 increments by default, with adaptive sub-stepping
    # on convergence trouble.  Init/total/min/max increment lengths follow the
    # *STATIC card spec: initial_inc, total_step, min_inc, max_inc.
    return [
        "*STEP, NLGEOM, INC=200",
        "*STATIC",
        "0.1, 1.0, 1e-5, 1.0",
    ]


def _build_input_deck(
    mesh: Any,
    material: dict,
    constraints: list[dict],
    loads: list[dict],
    warnings: list[str],
    analysis_type: str = "Linear",
) -> str:
    """Create a CalculiX/Abaqus-style input deck.

    ``analysis_type`` explicitly selects linear, geometric-nonlinear, or
    plastic-nonlinear static behavior.  Merely defining a yield strength must
    not change the constitutive model.
    """
    # A 10-row connectivity is a quadratic (C3D10) mesh — emit it verbatim;
    # anything else is treated as linear C3D4 (mesh_to_tet4 downgrades higher
    # orders to their corner nodes, preserving the prior behaviour).
    quadratic = tet10_connectivity(mesh) is not None
    if quadratic:
        points, tets = mesh_to_tet10(mesh, warnings)
        elem_type = "C3D10"
    else:
        points, tets = mesh_to_tet4(mesh, warnings)
        elem_type = "C3D4"
    step_header = _step_header(analysis_type)
    # Note: _build_sets_and_step prepends a hard-coded ``*STEP``/``*STATIC``;
    # we override it below by replacing the leading two entries with the
    # analysis-specific header.
    model_lines, step_lines = _build_sets_and_step(mesh, constraints, loads, warnings)
    if step_lines[:2] == ["*STEP", "*STATIC"]:
        step_lines = step_header + step_lines[2:]

    lines: list[str] = [
        "*HEADING",
        f"PyLCSS CalculiX deck ({analysis_type})",
        "*NODE",
    ]
    for idx, xyz in enumerate(points, start=1):
        lines.append(f"{idx}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}")

    lines.append(f"*ELEMENT, TYPE={elem_type}, ELSET=EALL")
    for idx, conn in enumerate(tets, start=1):
        node_ids = ", ".join(str(int(v) + 1) for v in conn)
        lines.append(f"{idx}, {node_ids}")

    lines.extend(
        _material_block(
            material,
            include_plasticity=analysis_type == "Nonlinear (Plastic)",
        )
    )
    lines.extend(model_lines)
    lines.extend(step_lines)
    return "\n".join(lines) + "\n"
