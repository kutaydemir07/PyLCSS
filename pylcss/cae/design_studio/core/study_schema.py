# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Lightweight metadata for saved-study controls and input files.

Discovery and override validation use the same node property declarations.
Paths, selection payloads, display choices and backend switches are not numeric
coupling controls. No Qt objects or solver backends are imported here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StudyNodeSchema:
    group: str
    numeric_controls: Mapping[str, str] = field(default_factory=dict)
    input_files: tuple[str, ...] = ()


def _controls(*names: str, **labels: str) -> dict[str, str]:
    return {name: labels.get(name, name.replace("_", " ").title()) for name in names}


STUDY_NODE_SCHEMAS = {
    "com.cad.sim.material": StudyNodeSchema(
        "Material",
        _controls(
            "youngs_modulus",
            "poissons_ratio",
            "density",
            "tangent_modulus",
            "failure_strain",
            "enable_fracture",
            "yield_strength",
            "strain_rate_sensitive",
            "strain_rate_c",
            "strain_rate_p",
            youngs_modulus="Young's modulus",
            poissons_ratio="Poisson's ratio",
        ),
    ),
    "com.cad.sim.mesh": StudyNodeSchema(
        "Mesh",
        _controls(
            "element_size",
            "refinement_size",
            "shell_thickness",
            "shell_nip",
            shell_nip="Shell integration points",
        ),
        ("mesh_path",),
    ),
    "com.cad.sim.constraint": StudyNodeSchema(
        "Boundary condition",
        _controls(
            "displacement_x",
            "displacement_y",
            "displacement_z",
            "displacement_x_enabled",
            "displacement_y_enabled",
            "displacement_z_enabled",
        ),
    ),
    "com.cad.sim.load": StudyNodeSchema(
        "Load",
        _controls(
            "force_x",
            "force_y",
            "force_z",
            "pressure",
            "gravity_accel",
            "case_weight",
            case_weight="Case weight",
        ),
    ),
    "com.cad.topopt.load": StudyNodeSchema(
        "Topology load",
        _controls(
            "force_x",
            "force_y",
            "force_z",
        ),
    ),
    "com.cad.sim.pressure_load": StudyNodeSchema("Load", _controls("pressure")),
    "com.cad.sim.impact": StudyNodeSchema(
        "Impact",
        _controls(
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "node_tolerance",
            "wall_friction",
            "wall_gap_mm",
            "impactor_mass_kg",
            "self_contact_enabled",
            "self_contact_friction",
            impactor_mass_kg="Impactor mass",
        ),
    ),
    "com.cad.sim.crash_solver": StudyNodeSchema(
        "Impact solver",
        _controls(
            "end_time",
            "n_frames",
            "history_samples",
            "time_steps",
            "enable_mass_scaling",
            "mass_scaling_target_dt",
            "time_step_scale",
            n_frames="Result frames",
        ),
    ),
    "com.cad.sim.topopt_voxel": StudyNodeSchema(
        "Topology optimization",
        _controls(
            "automatic_grid",
            "voxel_size",
            "max_voxel_elements",
            "automatic_mesh_refinement",
            "automatic_filter_radius",
            "topology_minimum_solid_size_mm",
            "topology_minimum_void_size_mm",
            "voxel_ram_fraction",
            "topology_objective_tolerance",
            "topology_overlap_tolerance",
            "nelx",
            "nely",
            "nelz",
            "volfrac",
            "rmin",
            "penal",
            "density_cutoff",
            "max_iter",
            "tol",
            "convergence_patience",
            "stress_constraint",
            "load_pnorm_p",
            "topology_convergence_enabled",
            "topology_convergence_levels",
            volfrac="Target material fraction",
            rmin="Filter radius",
            penal="SIMP penalty",
            max_iter="Maximum iterations",
            tol="Convergence tolerance",
        ),
    ),
    "com.cad.import_step": StudyNodeSchema("Geometry", input_files=("filepath",)),
    "com.cad.import_stl": StudyNodeSchema("Geometry", input_files=("filepath",)),
    "com.cad.freecad_part": StudyNodeSchema(
        "Geometry", input_files=("fcstd_filename",)
    ),
}

MATERIAL_CUSTOM_PROPERTIES = frozenset(
    {
        "youngs_modulus",
        "poissons_ratio",
        "density",
        "yield_strength",
        "tangent_modulus",
        "failure_strain",
        "strain_rate_c",
        "strain_rate_p",
    }
)


def study_node_identifier(type_name: str) -> str | None:
    """Match a saved identifier, optionally followed by its class name."""
    text = str(type_name or "")
    for identifier in sorted(STUDY_NODE_SCHEMAS, key=len, reverse=True):
        if text == identifier or text.startswith(identifier + "."):
            return identifier
    return None
