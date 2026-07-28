# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""LS-DYNA keyword and Engine deck generation for OpenRadioss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pylcss.input_values import as_bool
from pylcss.solver_backends.base import SolverBackendError
from pylcss.solver_backends.mesh import (
    is_shell_mesh,
    mesh_to_shell,
    mesh_to_tet4,
)
from pylcss.solver_backends.openradioss_impact import append_crash_event_cards
from pylcss.solver_backends.validation import (
    finite_float,
    integer,
    nonnegative_float,
    positive_float,
    validate_isotropic_material,
)


_MPA_TO_TONNE_MM_MS2 = 1.0e-6


@dataclass(frozen=True, slots=True)
class _PlasticMaterial:
    density: float
    youngs_modulus: float
    poissons_ratio: float
    yield_strength: float
    tangent_modulus: float
    failure_strain: float
    strain_rate_c: float
    strain_rate_p: float


def _plastic_material(material: dict[str, Any]) -> _PlasticMaterial:
    """Validate and convert MPa fields to the tonne-mm-ms deck unit system."""
    validate_isotropic_material(material, validate_strain_rate=True)
    youngs_modulus_mpa = positive_float(
        material.get("E", 210000.0),
        label="Material Young's modulus",
    )
    poissons_ratio = finite_float(
        material.get("nu", material.get("poissons_ratio", 0.3)),
        label="Material Poisson's ratio",
    )
    density = positive_float(
        material.get("rho", material.get("density", 7.85e-9)),
        label="Material density",
    )
    yield_strength_mpa = nonnegative_float(
        material.get("yield_strength", 250.0),
        label="Material yield strength",
    )
    tangent_modulus_mpa = nonnegative_float(
        material.get("tangent_modulus", 2000.0),
        label="Material tangent modulus",
    )
    failure_strain = nonnegative_float(
        material.get("failure_strain", 0.0),
        label="Material failure strain",
    )
    if not as_bool(material.get("enable_fracture", True)):
        failure_strain = 0.0

    # OpenRadioss documents SRC in 1/s and does not auto-scale it to the
    # deck's millisecond time unit. Zero for both values disables hardening.
    strain_rate_c = nonnegative_float(
        material.get("strain_rate_c", 0.0) or 0.0,
        label="Material Cowper-Symonds C",
    )
    strain_rate_p = nonnegative_float(
        material.get("strain_rate_p", 0.0) or 0.0,
        label="Material Cowper-Symonds P",
    )
    return _PlasticMaterial(
        density=density,
        youngs_modulus=youngs_modulus_mpa * _MPA_TO_TONNE_MM_MS2,
        poissons_ratio=poissons_ratio,
        yield_strength=yield_strength_mpa * _MPA_TO_TONNE_MM_MS2,
        tangent_modulus=tangent_modulus_mpa * _MPA_TO_TONNE_MM_MS2,
        failure_strain=failure_strain,
        strain_rate_c=strain_rate_c,
        strain_rate_p=strain_rate_p,
    )


def _append_shell_part(
    lines: list[str],
    *,
    triangles: np.ndarray,
    thickness: float,
    integration_points: int,
    hourglass_ihq: int,
    hourglass_coefficient: float,
) -> None:
    # A triangular *ELEMENT_SHELL repeats n3 in the n4 slot.
    lines.extend(["*ELEMENT_SHELL", "$#   eid     pid      n1      n2      n3      n4"])
    for element_id, connectivity in enumerate(triangles, start=1):
        node_1, node_2, node_3 = (int(value) + 1 for value in connectivity)
        lines.append(f"{element_id}, 1, {node_1}, {node_2}, {node_3}, {node_3}")

    hourglass_id = integer(
        hourglass_ihq,
        label="Hourglass formulation",
        minimum=0,
    )
    coefficient = nonnegative_float(
        hourglass_coefficient,
        label="Hourglass coefficient",
    )
    lines.extend(
        [
            "*PART",
            "PyLCSS shell part",
            "$#     pid     secid       mid     eosid     hgid",
            f"1, 1, 1, 0, {hourglass_id}",
            "*SECTION_SHELL",
            "$#   secid    elform      shrf       nip     propt   qr/irid     icomp     setyp",
            f"1, 2, 0.833333, {integration_points}, 1.0, 0, 0, 1",
            "$#      t1        t2        t3        t4",
            f"{thickness:.12g}, {thickness:.12g}, {thickness:.12g}, {thickness:.12g}",
        ]
    )
    if hourglass_id > 0:
        lines.extend(
            [
                "*HOURGLASS",
                "$#    hgid       ihq        qm       ibq        q1        q2     qb/vdc        qw",
                (
                    f"{hourglass_id}, {hourglass_id}, {coefficient:.6g}, "
                    f"0, 1.5, 0.06, {coefficient:.6g}, {coefficient:.6g}"
                ),
            ]
        )


def _append_solid_part(lines: list[str], tetrahedra: np.ndarray) -> None:
    lines.extend(["*ELEMENT_SOLID", "$#   eid     pid      n1      n2      n3      n4"])
    for element_id, connectivity in enumerate(tetrahedra, start=1):
        node_ids = [int(value) + 1 for value in connectivity]
        lines.append(
            f"{element_id}, 1, {node_ids[0]}, {node_ids[1]}, "
            f"{node_ids[2]}, {node_ids[3]}"
        )
    lines.extend(
        [
            "*PART",
            "PyLCSS solid part",
            "$#     pid     secid       mid",
            "1, 1, 1",
            "*SECTION_SOLID",
            "$#   secid    elform",
            "1, 10",
        ]
    )


def _append_material(lines: list[str], material: _PlasticMaterial) -> None:
    # The second *MAT_PLASTIC_KINEMATIC card is mandatory for OpenRadioss.
    lines.extend(
        [
            "*MAT_PLASTIC_KINEMATIC",
            "$#     mid        ro         e        pr      sigy      etan      beta",
            (
                f"1, {material.density:.12g}, {material.youngs_modulus:.12g}, "
                f"{material.poissons_ratio:.12g}, "
                f"{material.yield_strength:.12g}, "
                f"{material.tangent_modulus:.12g}, 0.0"
            ),
            "$#     src       srp        fs        vp",
            (
                f"{material.strain_rate_c:.6g}, "
                f"{material.strain_rate_p:.6g}, "
                f"{material.failure_strain:.6g}, 0.0"
            ),
        ]
    )


def _model_cards(
    mesh: Any,
    material: dict[str, Any],
    *,
    end_time: float,
    output_dt: float,
    warnings: list[str],
    hourglass_ihq: int,
    hourglass_coefficient: float,
) -> tuple[list[str], np.ndarray]:
    shell_mode = is_shell_mesh(mesh)
    tetrahedra: np.ndarray = np.empty((0, 4), dtype=int)
    triangles: np.ndarray = np.empty((0, 3), dtype=int)
    if shell_mode:
        points, triangles = mesh_to_shell(mesh, warnings)
    else:
        points, tetrahedra = mesh_to_tet4(mesh, warnings)

    lines = [
        "*KEYWORD",
        "*CONTROL_UNITS",
        "$#  length      time      mass",
        "mm ms mtrc_ton",
        "*TITLE",
        "PyLCSS OpenRadioss crash deck",
        "*CONTROL_TERMINATION",
        f"{end_time:.12g}",
        "*DATABASE_BINARY_D3PLOT",
        f"{output_dt:.12g}",
        "*NODE",
    ]
    for node_id, coordinates in enumerate(points, start=1):
        lines.append(
            f"{node_id}, {coordinates[0]:.12g}, "
            f"{coordinates[1]:.12g}, {coordinates[2]:.12g}"
        )

    if shell_mode:
        _append_shell_part(
            lines,
            triangles=triangles,
            thickness=positive_float(
                getattr(mesh, "shell_thickness", 1.5),
                label="Shell thickness",
            ),
            integration_points=integer(
                getattr(mesh, "shell_nip", 5),
                label="Shell integration-point count",
                minimum=1,
            ),
            hourglass_ihq=hourglass_ihq,
            hourglass_coefficient=hourglass_coefficient,
        )
    else:
        _append_solid_part(lines, tetrahedra)
    _append_material(lines, _plastic_material(material))
    return lines, points


def _build_keyword_deck(
    mesh: Any,
    material: dict[str, Any],
    constraints: list[dict[str, Any]],
    impact: dict[str, Any],
    end_time: float,
    output_dt: float,
    gravity: dict[str, Any] | None,
    warnings: list[str],
    impactor_mass: float = 0.0,
    out_meta: dict[str, Any] | None = None,
    hourglass_ihq: int = 4,
    hourglass_coefficient: float = 0.10,
) -> str:
    """Create an LS-DYNA keyword deck accepted by OpenRadioss Starter."""
    end_time = positive_float(end_time, label="Simulation end time")
    output_dt = positive_float(output_dt, label="Animation output interval")
    lines, points = _model_cards(
        mesh,
        material,
        end_time=end_time,
        output_dt=output_dt,
        warnings=warnings,
        hourglass_ihq=hourglass_ihq,
        hourglass_coefficient=hourglass_coefficient,
    )
    append_crash_event_cards(
        lines,
        mesh=mesh,
        points=points,
        constraints=constraints,
        impact=impact,
        end_time=end_time,
        gravity=gravity,
        warnings=warnings,
        impactor_mass=impactor_mass,
        out_meta=out_meta,
    )

    # OpenRadioss ignores LS-DYNA *DATABASE_GLSTAT/_MATSUM/_RWFORC cards.
    # Energy and wall-force histories therefore come from the Engine T01 file.
    lines.append("*END")
    return "\n".join(lines) + "\n"


def _build_engine_deck(
    job_name: str,
    end_time: float,
    output_dt: float,
    history_dt: float | None = None,
    mass_scaling_dt: float = 0.0,
    mass_scaling_scale: float = 0.9,
    time_step_scale: float = 0.9,
) -> str:
    """Create the OpenRadioss ``<job>_0001.rad`` Engine deck."""
    end_time = positive_float(end_time, label="Simulation end time")
    output_dt = positive_float(output_dt, label="Animation output interval")
    mass_scaling_dt = nonnegative_float(
        mass_scaling_dt,
        label="Mass-scaling target time step",
    )
    mass_scaling_scale = positive_float(
        mass_scaling_scale,
        label="Mass-scaling safety factor",
    )
    if mass_scaling_scale > 1.0:
        raise SolverBackendError("Mass-scaling safety factor must not exceed 1.0.")
    time_step_scale = positive_float(
        time_step_scale,
        label="Time-step safety factor",
    )
    if time_step_scale > 0.9:
        raise SolverBackendError("Time-step safety factor must not exceed 0.9.")
    history_dt = positive_float(
        output_dt if history_dt is None else history_dt,
        label="Time-history output interval",
    )

    lines = [
        "#RADIOSS ENGINE INPUT",
        f"/RUN/{job_name}/1",
        f"{end_time:.6g}",
    ]
    if mass_scaling_dt > 0.0:
        lines.extend(
            [
                "/DT/NODA/CST/0",
                f"{mass_scaling_scale:.6g}  {mass_scaling_dt:.6g}",
            ]
        )
    else:
        lines.extend(
            [
                "/DT/NODA/STOP/0",
                f"{time_step_scale:.6g}  0.0",
            ]
        )
    lines.extend(
        [
            "/ANIM/DT",
            f"0.  {output_dt:.6g}",
            "/ANIM/VECT/DISP",
            "/ANIM/VECT/VEL",
            "/ANIM/VECT/ACC",
            "/ANIM/ELEM/VONM",
            "/ANIM/ELEM/EPSP",
            "/ANIM/ELEM/ENER",
            "/ANIM/BRICK/TENS/STRESS",
            "/TH/TITLE",
            "/TFILE",
            f"{history_dt:.6g}",
            "",
        ]
    )
    return "\n".join(lines)
