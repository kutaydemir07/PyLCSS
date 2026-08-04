# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CalculiX deck preparation for bonded multi-component solid assemblies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np

from pylcss.solver_backends.base import SolverBackendError
from pylcss.solver_backends.calculix_deck import (
    _material_block,
    _step_header,
    _surface_area_weights_from_mesh_selection,
)
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
    validate_isotropic_material,
)


class CombinedMesh:
    """Minimal mesh container consumed by result import and the VTK viewer."""

    def __init__(self, points: np.ndarray, connectivity: np.ndarray) -> None:
        self.p = np.ascontiguousarray(points, dtype=float)
        self.t = np.ascontiguousarray(connectivity, dtype=int)


@dataclass(frozen=True)
class PartLayout:
    """Global numbering assigned to one FEA component."""

    name: str
    token: str
    mesh: Any
    material: dict[str, Any]
    node_start: int
    node_count: int
    element_start: int
    element_count: int
    element_set: str
    surface_name: str


@dataclass(frozen=True)
class PreparedAssembly:
    """Deck text and numbering metadata required after the solver run."""

    deck: str
    mesh: CombinedMesh
    parts: tuple[PartLayout, ...]
    connections: tuple[dict[str, Any], ...]
    constrained_node_ids: np.ndarray


def _safe_token(value: str, index: int) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_")
    if not token:
        token = f"COMPONENT_{index}"
    if token[0].isdigit():
        token = "C_" + token
    return token.upper()[:48]


def _part_for_record(record: dict[str, Any], parts: list[PartLayout], label: str):
    source_mesh = record.get("mesh")
    if source_mesh is None:
        raise SolverBackendError(
            f"{label} is not associated with a component mesh. Connect the "
            "component's mesh to the support/load node."
        )
    for part in parts:
        if source_mesh is part.mesh:
            return part
    raise SolverBackendError(
        f"{label} references a mesh that is not connected through an FEA Component."
    )


def _external_tet_faces(mesh: Any) -> list[tuple[int, int]]:
    """Return zero-based element and one-based CalculiX face ids."""
    points, cells = mesh_to_tet4(mesh, [])
    del points
    local_faces = np.asarray(
        ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)),
        dtype=int,
    )
    owners: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for element_index, cell in enumerate(cells):
        for face_index, local_nodes in enumerate(local_faces, start=1):
            key = tuple(sorted(int(value) for value in cell[local_nodes]))
            owners.setdefault(key, []).append((element_index, face_index))
    return [
        owner[0]
        for owner in owners.values()
        if len(owner) == 1
    ]


def _bbox_gap(first: PartLayout, second: PartLayout) -> float:
    a = np.asarray(first.mesh.p, dtype=float)
    b = np.asarray(second.mesh.p, dtype=float)
    a_min, a_max = np.min(a, axis=1), np.max(a, axis=1)
    b_min, b_max = np.min(b, axis=1), np.max(b, axis=1)
    separation = np.maximum(np.maximum(b_min - a_max, a_min - b_max), 0.0)
    return float(np.linalg.norm(separation))


def _build_connections(
    parts: list[PartLayout],
    *,
    tolerance: float,
    connection_mode: str,
    warnings: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    if len(parts) < 2:
        return [], []
    if connection_mode == "None (Disconnected)":
        warnings.append(
            "The assembly contains no interfaces. Disconnected components can "
            "produce rigid-body modes unless every component is independently supported."
        )
        return [], []
    if connection_mode != "Automatic Bonded":
        raise SolverBackendError(
            f"Unsupported assembly connection mode: {connection_mode!r}."
        )

    surface_lines: list[str] = []
    for part in parts:
        faces = _external_tet_faces(part.mesh)
        if not faces:
            raise SolverBackendError(
                f"Component {part.name!r} has no external tetrahedron faces."
            )
        surface_lines.append(
            f"*SURFACE, NAME={part.surface_name}, TYPE=ELEMENT"
        )
        surface_lines.extend(
            f"{part.element_start + local_element + 1}, S{face_id}"
            for local_element, face_id in faces
        )

    tie_lines: list[str] = []
    reports: list[dict[str, Any]] = []
    for first_index, first in enumerate(parts):
        for second in parts[first_index + 1 :]:
            gap = _bbox_gap(first, second)
            if gap > tolerance:
                continue
            tie_name = f"BOND_{first.token}_{second.token}"[:72]
            tie_lines.append(
                f"*TIE, NAME={tie_name}, POSITION TOLERANCE={tolerance:.12g}, ADJUST=NO"
            )
            tie_lines.append(f"{second.surface_name}, {first.surface_name}")
            reports.append(
                {
                    "name": tie_name,
                    "component_a": first.name,
                    "component_b": second.name,
                    "type": "bonded_tie",
                    "search_tolerance_mm": tolerance,
                    "bounding_box_gap_mm": gap,
                    "review_status": "automatic_requires_review",
                }
            )
    if not reports:
        raise SolverBackendError(
            "Automatic bonded-interface search found no component pair within "
            f"{tolerance:g} mm. Move mating faces together or increase the "
            "connection tolerance after checking the intended geometry."
        )
    warnings.append(
        f"Created {len(reports)} automatic bonded interface(s). Review the "
        "connection report; friction, separation, and bolt preload are not represented."
    )
    return surface_lines + tie_lines, reports


def _build_assembly_sets_and_step(
    parts: list[PartLayout],
    constraints: list[dict[str, Any]],
    loads: list[dict[str, Any]],
    warnings: list[str],
    analysis_type: str,
) -> tuple[list[str], list[str], np.ndarray]:
    model_lines: list[str] = []
    step_lines: list[str] = _step_header(analysis_type)

    boundary_lines: list[str] = []
    constrained_nodes: set[int] = set()
    for index, constraint in enumerate(constraints, start=1):
        part = _part_for_record(constraint, parts, f"Constraint {index}")
        geometries = dict_geometries(constraint)
        condition = str(constraint.get("condition", "") or "").strip()
        if geometries:
            local_nodes = nodes_matching_geometries(part.mesh, geometries)
        elif condition:
            local_nodes = nodes_matching_condition(
                part.mesh,
                condition,
                warnings,
                label=f"Constraint {index}",
            )
        else:
            raise SolverBackendError(
                f"Constraint {index} has no selected geometry or condition."
            )
        if local_nodes.size == 0:
            raise SolverBackendError(
                f"Constraint {index} did not match any mesh nodes. Review the "
                "named selection and component mesh."
            )

        set_name = f"BC_{index}_{part.token}"[:72]
        global_nodes = local_nodes + part.node_start + 1
        constrained_nodes.update(int(value) - 1 for value in global_nodes)
        model_lines.append(f"*NSET, NSET={set_name}")
        model_lines.extend(id_lines(global_nodes))
        try:
            fixed_dofs = [
                integer(value, label=f"Constraint {index} DOF")
                for value in constraint.get("fixed_dofs", [0, 1, 2])
            ]
        except TypeError as exc:
            raise SolverBackendError(
                f"Constraint {index} fixed_dofs must be a sequence."
            ) from exc
        if not set(fixed_dofs).issubset({0, 1, 2}):
            raise SolverBackendError(
                f"Constraint {index} fixed_dofs may only contain 0, 1, or 2."
            )
        displacement = constraint.get("displacement")
        values = None
        if displacement is not None:
            try:
                values = np.asarray(displacement, dtype=float).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise SolverBackendError(
                    f"Constraint {index} displacement must contain three numbers."
                ) from exc
            if values.size != 3 or not np.all(np.isfinite(values)):
                raise SolverBackendError(
                    f"Constraint {index} displacement must contain three finite numbers."
                )
        for dof in fixed_dofs:
            value = 0.0 if values is None else float(values[dof])
            boundary_lines.append(
                f"{set_name}, {dof + 1}, {dof + 1}, {value:.12g}"
            )

    if boundary_lines:
        step_lines.append("*BOUNDARY")
        step_lines.extend(boundary_lines)
    else:
        raise SolverBackendError(
            "No boundary constraints were exported to the assembly deck."
        )

    cload_lines: list[str] = []
    dload_lines: list[str] = []
    for index, load in enumerate(loads, start=1):
        part = _part_for_record(load, parts, f"Load {index}")
        load_type = str(load.get("type", "force") or "force").strip().lower()
        if load_type == "force":
            geometries = dict_geometries(load)
            condition = str(load.get("condition", "") or "").strip()
            if geometries:
                local_nodes = nodes_matching_geometries(part.mesh, geometries)
            elif condition:
                local_nodes = nodes_matching_condition(
                    part.mesh,
                    condition,
                    warnings,
                    label=f"Force load {index}",
                )
            else:
                raise SolverBackendError(
                    f"Force load {index} has no selected geometry or condition."
                )
            if local_nodes.size == 0:
                raise SolverBackendError(
                    f"Force load {index} did not match any mesh nodes. Review the "
                    "named selection and component mesh."
                )
            force = load_vector(load)
            area_weights = _surface_area_weights_from_mesh_selection(geometries)
            total_area = float(sum(area_weights.values()))
            if total_area > 1.0e-16:
                selected = set(int(value) for value in local_nodes)
                for local_node, area in sorted(area_weights.items()):
                    if local_node not in selected:
                        continue
                    nodal_force = force * float(area) / total_area
                    for dof, value in enumerate(nodal_force, start=1):
                        if abs(float(value)) > 1.0e-16:
                            cload_lines.append(
                                f"{part.node_start + local_node + 1}, "
                                f"{dof}, {float(value):.12g}"
                            )
            else:
                nodal_force = force / max(local_nodes.size, 1)
                for local_node in local_nodes:
                    for dof, value in enumerate(nodal_force, start=1):
                        if abs(float(value)) > 1.0e-16:
                            cload_lines.append(
                                f"{part.node_start + int(local_node) + 1}, "
                                f"{dof}, {float(value):.12g}"
                            )
        elif load_type == "gravity":
            direction = str(load.get("direction", "-Y") or "-Y").strip().upper()
            direction_map = {
                "-X": (-1.0, 0.0, 0.0),
                "+X": (1.0, 0.0, 0.0),
                "-Y": (0.0, -1.0, 0.0),
                "+Y": (0.0, 1.0, 0.0),
                "-Z": (0.0, 0.0, -1.0),
                "+Z": (0.0, 0.0, 1.0),
            }
            if direction not in direction_map:
                raise SolverBackendError(
                    f"Unsupported gravity direction: {direction!r}."
                )
            acceleration = nonnegative_float(
                load.get("accel", 9810.0),
                label="Gravity acceleration",
            )
            dx, dy, dz = direction_map[direction]
            dload_lines.append(
                f"{part.element_set}, GRAV, {acceleration:.12g}, "
                f"{dx:.1f}, {dy:.1f}, {dz:.1f}"
            )
        elif load_type == "pressure":
            geometries = dict_geometries(load)
            faces = tet_face_sets_for_geometries(part.mesh, geometries)
            if not faces:
                raise SolverBackendError(
                    f"Pressure load {index} did not match an external face on "
                    f"component {part.name!r}."
                )
            set_name = f"PRESS_{index}_{part.token}"[:72]
            model_lines.append(f"*SURFACE, NAME={set_name}, TYPE=ELEMENT")
            model_lines.extend(
                f"{part.element_start + element_id}, S{face_id}"
                for element_id, face_id in faces
            )
            pressure = finite_float(
                load.get("pressure", load.get("magnitude", 0.0)),
                label=f"Pressure load {index} magnitude",
            )
            dload_lines.append(f"{set_name}, P, {pressure:.12g}")
        else:
            warnings.append(f"Unsupported CalculiX load type: {load_type}")

    if cload_lines:
        step_lines.append("*CLOAD")
        step_lines.extend(cload_lines)
    if dload_lines:
        step_lines.append("*DLOAD")
        step_lines.extend(dload_lines)
    if not cload_lines and not dload_lines:
        driven = any(
            constraint.get("displacement") is not None
            and any(
                abs(float(constraint["displacement"][dof])) > 1.0e-15
                for dof in constraint.get("fixed_dofs", (0, 1, 2))
            )
            for constraint in constraints
        )
        if not driven:
            raise SolverBackendError(
                "No external loads or non-zero prescribed displacements were "
                "exported to the assembly deck."
            )
    step_lines.extend(
        [
            "*NODE FILE",
            "U, RF",
            "*EL FILE",
            "S, E, ENER",
            "*END STEP",
        ]
    )
    return (
        model_lines,
        step_lines,
        np.asarray(sorted(constrained_nodes), dtype=int),
    )


def prepare_calculix_assembly(
    components: list[dict[str, Any]],
    constraints: list[dict[str, Any]],
    loads: list[dict[str, Any]],
    warnings: list[str],
    *,
    analysis_type: str,
    connection_mode: str,
    connection_tolerance_mm: float,
) -> PreparedAssembly:
    """Validate components and build a bonded CalculiX assembly deck."""
    if len(components) < 2:
        raise SolverBackendError(
            "A multi-component study requires at least two FEA Component inputs."
        )
    try:
        tolerance = float(connection_tolerance_mm)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError(
            "Connection tolerance must be numeric."
        ) from exc
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise SolverBackendError(
            "Connection tolerance must be finite and greater than zero."
        )

    layouts: list[PartLayout] = []
    point_blocks: list[np.ndarray] = []
    cell_blocks: list[np.ndarray] = []
    names: set[str] = set()
    tokens: set[str] = set()
    expected_order: int | None = None
    node_start = 0
    element_start = 0
    for index, component in enumerate(components, start=1):
        if not isinstance(component, dict) or component.get("type") != "fea_component":
            raise SolverBackendError(
                f"Component input {index} is not an FEA Component result."
            )
        name = str(component.get("name") or "").strip()
        mesh = component.get("mesh")
        material = component.get("material")
        if not name:
            raise SolverBackendError(f"Component input {index} has no name.")
        if name.casefold() in names:
            raise SolverBackendError(f"Duplicate component name: {name!r}.")
        if not isinstance(material, dict):
            raise SolverBackendError(f"Component {name!r} has no valid material.")
        validate_isotropic_material(
            material,
            require_yield=analysis_type == "Nonlinear (Plastic)",
        )
        quadratic = tet10_connectivity(mesh) is not None
        if quadratic:
            points, cells = mesh_to_tet10(mesh, warnings)
            order = 10
        else:
            points, cells = mesh_to_tet4(mesh, warnings)
            order = 4
        if expected_order is None:
            expected_order = order
        elif order != expected_order:
            raise SolverBackendError(
                "All components in one assembly must use the same tetrahedron "
                "order (all Tet or all Tet10)."
            )
        token = _safe_token(name, index)
        while token in tokens:
            token = f"{token}_{index}"[:48]
        tokens.add(token)
        names.add(name.casefold())
        point_blocks.append(points.T)
        cell_blocks.append(cells.T + node_start)
        layouts.append(
            PartLayout(
                name=name,
                token=token,
                mesh=mesh,
                material=material,
                node_start=node_start,
                node_count=points.shape[0],
                element_start=element_start,
                element_count=cells.shape[0],
                element_set=f"EL_{token}"[:72],
                surface_name=f"SURF_{token}"[:72],
            )
        )
        node_start += points.shape[0]
        element_start += cells.shape[0]

    combined = CombinedMesh(
        np.hstack(point_blocks),
        np.hstack(cell_blocks),
    )
    connection_lines, connection_reports = _build_connections(
        layouts,
        tolerance=tolerance,
        connection_mode=connection_mode,
        warnings=warnings,
    )
    model_lines, step_lines, constrained_node_ids = _build_assembly_sets_and_step(
        layouts,
        constraints,
        loads,
        warnings,
        analysis_type,
    )

    element_type = "C3D10" if expected_order == 10 else "C3D4"
    lines = [
        "*HEADING",
        f"PyLCSS CalculiX bonded assembly ({analysis_type})",
        "*NODE",
    ]
    for node_id, xyz in enumerate(combined.p.T, start=1):
        lines.append(
            f"{node_id}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}"
        )
    for part, cells in zip(layouts, cell_blocks):
        lines.append(
            f"*ELEMENT, TYPE={element_type}, ELSET={part.element_set}"
        )
        for local_index, connectivity in enumerate(cells.T, start=1):
            element_id = part.element_start + local_index
            node_ids = ", ".join(str(int(value) + 1) for value in connectivity)
            lines.append(f"{element_id}, {node_ids}")
        lines.extend(
            _material_block(
                part.material,
                include_plasticity=analysis_type == "Nonlinear (Plastic)",
                material_name=f"MAT_{part.token}"[:72],
                element_set=part.element_set,
            )
        )
    lines.extend(connection_lines)
    lines.extend(model_lines)
    lines.extend(step_lines)
    return PreparedAssembly(
        deck="\n".join(lines) + "\n",
        mesh=combined,
        parts=tuple(layouts),
        connections=tuple(connection_reports),
        constrained_node_ids=constrained_node_ids,
    )


def assembly_result_summary(
    result: dict[str, Any],
    prepared: PreparedAssembly,
) -> dict[str, Any]:
    """Add per-component mass and peak-response summaries to an FRD result."""
    stress = np.asarray(result.get("stress", []), dtype=float)
    displacement = np.asarray(result.get("displacement", []), dtype=float)
    if displacement.size == 3 * prepared.mesh.p.shape[1]:
        displacement = displacement.reshape((-1, 3))
    else:
        displacement = np.zeros((prepared.mesh.p.shape[1], 3), dtype=float)
    element_volumes = np.asarray(result.get("element_volumes", []), dtype=float)
    nodal_force = np.asarray(result.get("nodal_force", []), dtype=float)
    constrained = np.asarray(prepared.constrained_node_ids, dtype=int)
    if (
        nodal_force.ndim == 2
        and nodal_force.shape[1] >= 3
        and constrained.size
    ):
        reaction_force = np.sum(nodal_force[constrained, :3], axis=0)
    else:
        reaction_force = np.zeros(3, dtype=float)

    summaries: list[dict[str, Any]] = []
    total_mass = 0.0
    minimum_safety_factor: float | None = None
    for part in prepared.parts:
        node_slice = slice(part.node_start, part.node_start + part.node_count)
        element_slice = slice(
            part.element_start,
            part.element_start + part.element_count,
        )
        volume = (
            float(np.sum(element_volumes[element_slice]))
            if element_volumes.size >= part.element_start + part.element_count
            else 0.0
        )
        density = float(
            part.material.get("rho", part.material.get("density", 0.0)) or 0.0
        )
        mass = volume * density
        total_mass += mass
        peak_stress = (
            float(np.max(stress[node_slice]))
            if stress.size >= part.node_start + part.node_count
            else 0.0
        )
        peak_displacement = float(
            np.max(np.linalg.norm(displacement[node_slice], axis=1))
        )
        yield_strength = float(part.material.get("yield_strength", 0.0) or 0.0)
        safety_factor = (
            yield_strength / peak_stress
            if yield_strength > 0.0 and peak_stress > 0.0
            else None
        )
        if safety_factor is not None:
            minimum_safety_factor = (
                safety_factor
                if minimum_safety_factor is None
                else min(minimum_safety_factor, safety_factor)
            )
        summaries.append(
            {
                "name": part.name,
                "material": str(part.material.get("name") or "Material"),
                "nodes": part.node_count,
                "elements": part.element_count,
                "volume_mm3": volume,
                "mass_tonne": mass,
                "peak_displacement_mm": peak_displacement,
                "peak_stress_mpa": peak_stress,
                "yield_safety_factor": safety_factor,
            }
        )
    return {
        "mass": total_mass,
        "reaction_force": reaction_force,
        "reaction_magnitude": float(np.linalg.norm(reaction_force)),
        "reaction_node_ids": constrained,
        "components": summaries,
        "connections": list(prepared.connections),
        "component_count": len(prepared.parts),
        "connection_count": len(prepared.connections),
        "minimum_yield_safety_factor": minimum_safety_factor,
        "assembly_model": "automatic_bonded_tie",
    }


__all__ = [
    "CombinedMesh",
    "PartLayout",
    "PreparedAssembly",
    "assembly_result_summary",
    "prepare_calculix_assembly",
]
