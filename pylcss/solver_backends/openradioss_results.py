# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Result alignment and energy summaries for the OpenRadioss adapter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pylcss.solver_backends.base import SolverBackendError
from pylcss.solver_backends.mesh import (
    is_shell_mesh,
    mesh_to_shell,
    mesh_to_tet4,
)
from pylcss.solver_backends.validation import finite_float
from pylcss.design_studio.crash.provenance import (
    build_crash_provenance,
    write_crash_manifest,
)
from pylcss.design_studio.crash.quality import evaluate_crash_quality
from pylcss.design_studio.crash.signals import build_crash_measurements


Frame: TypeAlias = dict[str, Any]
FloatArray: TypeAlias = NDArray[np.float64]
_STRESS_TO_MPA = 1.0e6


def wrap_deck_result(
    status: str,
    work_dir: Path,
    deck: Path,
    engine_deck: Path | None,
    starter_executable: str | None,
    engine_executable: str | None,
    solver_log: str,
    warnings: list[str],
    visualization_mode: str,
    displacement_scale: float,
) -> dict[str, Any]:
    """Build the common result returned when no animation could be imported."""
    return {
        "type": "external_solver",
        "backend": "OpenRadioss",
        "external_status": status,
        "mesh": None,
        "visualization_mode": visualization_mode,
        "disp_scale": displacement_scale,
        "input_file": str(deck),
        "engine_file": str(engine_deck) if engine_deck else None,
        "work_dir": str(work_dir),
        "solver_executable": starter_executable,
        "secondary_solver_executable": engine_executable,
        "solver_log": solver_log,
        "warnings": warnings,
        "message": "OpenRadioss external deck run finished without animation import.",
    }


def _vector_field(
    values: Any,
    source_ids: NDArray[np.int_],
    target_count: int,
) -> FloatArray:
    try:
        raw = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError(
            "OpenRadioss animation vector field is not numeric."
        ) from exc
    if raw.ndim == 1 and raw.size % 3 == 0:
        raw = raw.reshape((-1, 3))
    if raw.ndim != 2 or raw.shape[1] != 3:
        if raw.size:
            raise SolverBackendError(
                "OpenRadioss animation vector field must have three components."
            )
        raw = np.zeros((0, 3), dtype=float)
    if not np.all(np.isfinite(raw)):
        raise SolverBackendError(
            "OpenRadioss animation contains non-finite vector values."
        )

    target: FloatArray = np.zeros((target_count, 3), dtype=float)
    if source_ids.size == raw.shape[0]:
        valid = (source_ids >= 1) & (source_ids <= target_count)
        target[source_ids[valid] - 1] = raw[valid]
    elif raw.shape[0]:
        count = min(raw.shape[0], target_count)
        target[:count] = raw[:count]
    return target


def _scalar_field(
    values: Any,
    source_ids: NDArray[np.int_],
    target_count: int,
) -> FloatArray:
    try:
        raw = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError(
            "OpenRadioss animation scalar field is not numeric."
        ) from exc
    if not np.all(np.isfinite(raw)):
        raise SolverBackendError(
            "OpenRadioss animation contains non-finite scalar values."
        )
    target: FloatArray = np.zeros(target_count, dtype=float)
    if source_ids.size == raw.size:
        valid = (source_ids >= 1) & (source_ids <= target_count)
        target[source_ids[valid] - 1] = raw[valid]
    elif raw.size:
        count = min(raw.size, target_count)
        target[:count] = raw[:count]
    return target


def _optional_cell_field(
    frame: Frame,
    name: str,
    element_ids: NDArray[np.int_],
    element_count: int,
) -> FloatArray | None:
    raw_value = frame.get(name)
    if raw_value is None:
        return None
    return _scalar_field(raw_value, element_ids, element_count)


def align_animation_frames(mesh: Any, frames: list[Frame]) -> list[Frame]:
    """Align converted frame arrays with the original PyLCSS mesh numbering."""
    points = np.asarray(mesh.p)
    elements = np.asarray(mesh.t)
    if points.ndim != 2 or elements.ndim != 2:
        raise SolverBackendError(
            "Animation alignment requires two-dimensional mesh arrays."
        )
    point_count = int(points.shape[1])
    element_count = int(elements.shape[1])

    aligned: list[Frame] = []
    for frame in frames:
        node_ids = np.asarray(frame.get("node_ids", []), dtype=int).reshape(-1)
        element_ids = np.asarray(
            frame.get("element_ids", []),
            dtype=int,
        ).reshape(-1)

        rigid_wall_reference = None
        wall_candidates = np.flatnonzero(node_ids > point_count)
        if wall_candidates.size:
            wall_index = int(wall_candidates[np.argmax(node_ids[wall_candidates])])
            raw_displacement = np.asarray(
                frame.get("displacement", []), dtype=float
            ).reshape((-1, 3))
            raw_velocity = np.asarray(frame.get("velocity", []), dtype=float)
            raw_acceleration = np.asarray(
                frame.get("acceleration", []), dtype=float
            )
            if wall_index < raw_displacement.shape[0]:
                rigid_wall_reference = {
                    "node_id": int(node_ids[wall_index]),
                    "displacement": raw_displacement[wall_index].tolist(),
                    "velocity": (
                        raw_velocity[wall_index].tolist()
                        if raw_velocity.ndim == 2
                        and wall_index < raw_velocity.shape[0]
                        else [0.0, 0.0, 0.0]
                    ),
                    "acceleration": (
                        raw_acceleration[wall_index].tolist()
                        if raw_acceleration.ndim == 2
                        and wall_index < raw_acceleration.shape[0]
                        else [0.0, 0.0, 0.0]
                    ),
                }

        displacement_xyz = _vector_field(
            frame.get("displacement", []),
            node_ids,
            point_count,
        )
        displacement = displacement_xyz.reshape(-1)
        stress = (
            _scalar_field(
                frame.get("stress_vm", []),
                node_ids,
                point_count,
            )
            * _STRESS_TO_MPA
        )
        stress_cell = _optional_cell_field(
            frame,
            "stress_vm_cell",
            element_ids,
            element_count,
        )
        if stress_cell is not None:
            stress_cell *= _STRESS_TO_MPA

        aligned.append(
            {
                "displacement": displacement,
                "stress_vm": stress,
                "stress_vm_cell": stress_cell,
                "velocity": _vector_field(
                    frame.get("velocity", []),
                    node_ids,
                    point_count,
                ),
                "acceleration": _vector_field(
                    frame.get("acceleration", []),
                    node_ids,
                    point_count,
                ),
                "ener_cell": _optional_cell_field(
                    frame,
                    "ener_cell",
                    element_ids,
                    element_count,
                ),
                "eps_p": _scalar_field(
                    frame.get("eps_p", []),
                    node_ids,
                    point_count,
                ),
                "failed": _scalar_field(
                    frame.get("failed", []),
                    node_ids,
                    point_count,
                ),
                "eps_p_cell": _optional_cell_field(
                    frame,
                    "eps_p_cell",
                    element_ids,
                    element_count,
                ),
                "failed_cell": _optional_cell_field(
                    frame,
                    "failed_cell",
                    element_ids,
                    element_count,
                ),
                "rigid_wall_reference": rigid_wall_reference,
                "time": finite_float(
                    frame.get("time", 0.0),
                    label="OpenRadioss frame time",
                ),
                "time_is_normalized": bool(frame.get("time_is_normalized", False)),
            }
        )
    return aligned


def animation_event_peaks(frames: list[Frame]) -> tuple[float, float]:
    """Return event-wide displacement and Von Mises maxima."""
    peak_displacement = 0.0
    peak_stress = 0.0
    for frame in frames:
        displacement = np.asarray(
            frame.get("displacement", []),
            dtype=float,
        ).reshape(-1)
        if displacement.size and displacement.size % 3 == 0:
            peak_displacement = max(
                peak_displacement,
                float(
                    np.max(
                        np.linalg.norm(
                            displacement.reshape((-1, 3)),
                            axis=1,
                        )
                    )
                ),
            )
        stress_value = frame.get("stress_vm_cell")
        if stress_value is None:
            stress_value = frame.get("stress_vm", [])
        stress = np.asarray(stress_value, dtype=float).reshape(-1)
        if stress.size:
            peak_stress = max(peak_stress, float(np.max(stress)))
    return peak_displacement, peak_stress


def compute_time_history(
    mesh: Any,
    material: dict[str, Any],
    frames: list[Frame],
    end_time: float,
) -> dict[str, Any]:
    """Reconstruct kinetic and internal energy from animation fields."""
    del end_time  # Frame timestamps are authoritative.
    if not frames:
        return {"t_ms": [], "ke_kj": [], "ie_kj": []}

    shell_mode = is_shell_mesh(mesh)
    if shell_mode:
        points, connectivity = mesh_to_shell(mesh, [])
        nodes_per_element = 3
    else:
        points, connectivity = mesh_to_tet4(mesh, [])
        nodes_per_element = 4

    density = float(material.get("rho", material.get("density", 7.85e-9)))
    point_count = points.shape[0]
    element_count = connectivity.shape[0]

    v0 = points[connectivity[:, 0]]
    edge_1 = points[connectivity[:, 1]] - v0
    edge_2 = points[connectivity[:, 2]] - v0
    if shell_mode:
        thickness = float(getattr(mesh, "shell_thickness", 1.5))
        element_volume = (
            0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1) * thickness
        )
    else:
        edge_3 = points[connectivity[:, 3]] - v0
        element_volume = (
            np.abs(
                np.einsum(
                    "ij,ij->i",
                    np.cross(edge_1, edge_2),
                    edge_3,
                )
            )
            / 6.0
        )

    node_mass = np.zeros(point_count, dtype=float)
    mass_share = density * element_volume / nodes_per_element
    for local_node in range(nodes_per_element):
        np.add.at(node_mass, connectivity[:, local_node], mass_share)

    times: list[float] = []
    kinetic_energy: list[float] = []
    internal_energy: list[float] = []
    for frame in frames:
        times.append(float(frame.get("time", 0.0)))

        velocity = np.asarray(frame.get("velocity", []), dtype=float)
        if velocity.shape == (point_count, 3):
            speed_squared = np.einsum("ij,ij->i", velocity, velocity)
            kinetic_energy.append(float(0.5 * np.sum(node_mass * speed_squared)))
        else:
            kinetic_energy.append(0.0)

        specific_energy = np.asarray(
            frame.get("ener_cell", []),
            dtype=float,
        ).reshape(-1)
        count = min(specific_energy.size, element_count)
        internal_energy.append(
            float(np.sum(specific_energy[:count] * density * element_volume[:count]))
            if count
            else 0.0
        )

    total_volume = float(np.sum(element_volume))
    return {
        "t_ms": times,
        "ke_kj": kinetic_energy,
        "ie_kj": internal_energy,
        "total_volume_mm3": total_volume,
        "total_mass_kg": float(density * total_volume * 1.0e3),
    }


def compute_structural_mass_kg(mesh: Any, material: dict[str, Any]) -> float:
    """Return deformable mesh mass, excluding added sled/impactor mass."""
    shell_mode = is_shell_mesh(mesh)
    if shell_mode:
        points, connectivity = mesh_to_shell(mesh, [])
    else:
        points, connectivity = mesh_to_tet4(mesh, [])
    if connectivity.size == 0:
        return 0.0

    v0 = points[connectivity[:, 0]]
    edge_1 = points[connectivity[:, 1]] - v0
    edge_2 = points[connectivity[:, 2]] - v0
    if shell_mode:
        thickness = float(getattr(mesh, "shell_thickness", 1.5))
        element_volume = (
            0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1) * thickness
        )
    else:
        edge_3 = points[connectivity[:, 3]] - v0
        element_volume = (
            np.abs(
                np.einsum(
                    "ij,ij->i",
                    np.cross(edge_1, edge_2),
                    edge_3,
                )
            )
            / 6.0
        )
    density = float(material.get("rho", material.get("density", 7.85e-9)))
    return float(density * np.sum(element_volume) * 1.0e3)


def read_engine_energy_history(output_path: Path | str) -> dict[str, Any]:
    """Parse the authoritative global energy table from an Engine output log."""
    path = Path(output_path)
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    rows: list[
        tuple[
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 13 or not fields[0].isdigit() or not fields[5].endswith("%"):
            continue
        try:
            values = [
                float(value.replace("D", "E").replace("d", "e"))
                for value in fields[1:3] + [fields[5][:-1]] + fields[6:13]
            ]
            if len(values) != 10:
                continue
            row = (
                int(fields[0]),
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                values[8],
                values[9],
            )
        except ValueError:
            continue
        if all(math.isfinite(value) for value in row[1:]):
            rows.append(row)

    if not rows:
        return {}
    by_cycle = {row[0]: row for row in rows}
    ordered = [by_cycle[cycle] for cycle in sorted(by_cycle)]
    return {
        "source": "OpenRadioss Engine global energy table",
        "cycle": [row[0] for row in ordered],
        "t_ms": [row[1] for row in ordered],
        "time_step_ms": [row[2] for row in ordered],
        "energy_error_pct": [row[3] for row in ordered],
        "ie_kj": [row[4] for row in ordered],
        "ke_kj": [row[5] + row[6] for row in ordered],
        "ke_translational_kj": [row[5] for row in ordered],
        "ke_rotational_kj": [row[6] for row in ordered],
        "external_work_kj": [row[7] for row in ordered],
        "mass_error": [row[8] for row in ordered],
        "total_mass_tonne": [row[9] for row in ordered],
        "mass_added_tonne": [row[10] for row in ordered],
    }


def _primary_cell_field(
    frame: Frame,
    viewer_mesh: Any,
    name: str,
) -> NDArray[np.float64] | None:
    value = frame.get(name)
    if value is None:
        return None
    field = np.asarray(value, dtype=float).reshape(-1)
    indices = np.asarray(
        getattr(viewer_mesh, "primary_cell_indices", []),
        dtype=int,
    ).reshape(-1)
    if indices.size and int(np.max(indices)) < field.size:
        return field[indices]
    return field


def build_existing_deck_result(
    *,
    status: str,
    work_dir: Path,
    deck_path: Path,
    engine_deck_path: Path,
    starter_executable: str,
    engine_executable: str,
    solver_log: str,
    warnings: list[str],
    visualization_mode: str,
    displacement_scale: float,
    frames: list[Frame],
    end_time: float | None,
    source_name: str,
) -> dict[str, Any]:
    """Build the viewer result for a successfully imported user deck."""
    viewer_mesh = frames[0].get("mesh")
    last = frames[-1]
    displacement = np.asarray(
        last.get("displacement", []),
        dtype=float,
    ).reshape(-1)
    point_count = displacement.size // 3 if displacement.size else 0
    final_peak_displacement = (
        float(
            np.max(
                np.linalg.norm(
                    displacement.reshape(point_count, 3),
                    axis=1,
                )
            )
        )
        if point_count
        else 0.0
    )
    stress = np.asarray(last.get("stress_vm", []), dtype=float)
    plastic_strain = _primary_cell_field(last, viewer_mesh, "eps_p_cell")
    failed_elements = _primary_cell_field(last, viewer_mesh, "failed_cell")
    element_stress = _primary_cell_field(last, viewer_mesh, "stress_vm_cell")
    final_peak_stress = (
        float(np.max(element_stress))
        if element_stress is not None and element_stress.size
        else float(stress.max())
        if stress.size
        else 0.0
    )
    peak_displacement, peak_stress = animation_event_peaks(frames)
    return {
        "type": "crash",
        "backend": "OpenRadioss",
        "external_status": status,
        "mesh": viewer_mesh,
        "displacement": displacement,
        "stress": stress,
        "element_stress": element_stress,
        "visualization_mode": visualization_mode,
        "disp_scale": displacement_scale,
        "frames": frames,
        "peak_displacement": peak_displacement,
        "peak_stress": peak_stress,
        "final_frame_displacement": final_peak_displacement,
        "final_frame_stress": final_peak_stress,
        "plastic_strain": plastic_strain,
        "failed_elements": failed_elements,
        "n_failed": int(
            np.count_nonzero(
                np.asarray(
                    failed_elements if failed_elements is not None else [],
                    dtype=float,
                )
                >= 0.5
            )
        ),
        "wall": None,
        "end_time": (
            end_time
            if end_time is not None
            else finite_float(last.get("time", 1.0), label="Final frame time")
        ),
        "input_file": str(deck_path),
        "engine_file": str(engine_deck_path),
        "work_dir": str(work_dir),
        "solver_executable": starter_executable,
        "secondary_solver_executable": engine_executable,
        "solver_log": solver_log,
        "warnings": warnings,
        "message": (
            f"OpenRadioss completed on user deck `{source_name}`; "
            f"{len(frames)} animation frames imported."
        ),
    }


def _energy_balance_summary(
    mesh: Any,
    material: dict[str, Any],
    frames: list[Frame],
    *,
    end_time: float,
    engine_output: Path,
    warnings: list[str],
) -> tuple[dict[str, Any], float | None, float | None, float | None]:
    reconstructed = compute_time_history(mesh, material, frames, end_time)
    engine_history = read_engine_energy_history(engine_output)
    if engine_history:
        history = {
            **{
                key: value
                for key, value in reconstructed.items()
                if key not in {"t_ms", "ke_kj", "ie_kj"}
            },
            **engine_history,
        }
    else:
        history = {
            **reconstructed,
            "source": "animation-field reconstruction",
        }

    raw_energy_errors = history.get("energy_error_pct")
    energy_errors = np.asarray(
        raw_energy_errors if raw_energy_errors is not None else [],
        dtype=float,
    )
    raw_mass_errors = history.get("mass_error")
    mass_errors = np.asarray(
        raw_mass_errors if raw_mass_errors is not None else [],
        dtype=float,
    )
    maximum_energy_error = (
        float(np.max(np.abs(energy_errors))) / 100.0 if energy_errors.size else None
    )
    final_energy_error = (
        float(energy_errors[-1]) / 100.0 if energy_errors.size else None
    )
    maximum_mass_error = (
        float(np.max(np.abs(mass_errors))) / 100.0 if mass_errors.size else None
    )
    if energy_errors.size:
        final_energy_percent = float(energy_errors[-1])
        if final_energy_percent > 2.0:
            warnings.append(
                "OpenRadioss reports positive final energy creation of "
                f"{final_energy_percent:.2f}%; investigate contacts, kinematic "
                "conditions, and element stability before using the result."
            )
        elif final_energy_percent < -15.0:
            warnings.append(
                "OpenRadioss reports final energy loss of "
                f"{final_energy_percent:.2f}%, beyond the usual preliminary "
                "15% check; inspect hourglass/contact energy and mesh quality."
            )
    if mass_errors.size and float(np.max(np.abs(mass_errors))) > 1.0:
        warnings.append(
            "OpenRadioss reports more than 1% mass change. Reduce or disable "
            "constant-time-step mass scaling and inspect local added mass."
        )
    return (
        history,
        maximum_energy_error,
        final_energy_error,
        maximum_mass_error,
    )


def build_generated_crash_result(
    *,
    mesh: Any,
    material: dict[str, Any],
    frames: list[Frame],
    status: str,
    visualization_mode: str,
    displacement_scale: float,
    wall: Any,
    end_time: float,
    deck_path: Path,
    engine_deck_path: Path,
    work_dir: Path,
    job_name: str,
    starter_executable: str | None,
    engine_executable: str | None,
    solver_log: str,
    warnings: list[str],
    solver_history: dict[str, Any],
    measurement: dict[str, Any],
    impact: dict[str, Any],
    constraints: list[dict[str, Any]],
    solver_settings: dict[str, Any],
    acceleration_cfc: int,
    force_cfc: int,
) -> dict[str, Any]:
    """Build a crash result for a PyLCSS-generated and completed simulation."""
    point_count = int(np.asarray(mesh.p).shape[1])
    last = frames[-1]
    displacement = np.asarray(last["displacement"], dtype=float)
    stress = np.asarray(last["stress_vm"], dtype=float)
    final_peak_displacement = float(
        np.max(
            np.linalg.norm(
                displacement.reshape(point_count, 3),
                axis=1,
            )
        )
    )
    element_stress = last.get("stress_vm_cell")
    peak_field = (
        np.asarray(element_stress, dtype=float)
        if element_stress is not None
        else stress
    )
    final_peak_stress = float(np.max(peak_field)) if peak_field.size else 0.0
    peak_displacement, peak_stress = animation_event_peaks(frames)
    (
        time_history,
        maximum_energy_error,
        final_energy_error,
        maximum_mass_error,
    ) = _energy_balance_summary(
        mesh,
        material,
        frames,
        end_time=end_time,
        engine_output=work_dir / f"{job_name}_0001.out",
        warnings=warnings,
    )
    failed_elements = last.get("failed_cell")
    raw_internal_energy = time_history.get("ie_kj")
    internal_energy_kj = np.asarray(
        raw_internal_energy if raw_internal_energy is not None else [],
        dtype=float,
    ).reshape(-1)
    absorbed_energy_kj = (
        float(internal_energy_kj[-1]) if internal_energy_kj.size else 0.0
    )
    measurement = dict(measurement)
    measurement["source_point_count"] = point_count
    measurement["material_validation"] = dict(material.get("validation") or {})
    measurement["structural_mass_kg"] = compute_structural_mass_kg(mesh, material)
    measurements = build_crash_measurements(
        solver_history=solver_history,
        frames=frames,
        measurement=measurement,
        acceleration_cfc=acceleration_cfc,
        force_cfc=force_cfc,
    )
    quality = evaluate_crash_quality(
        measurements=measurements,
        external_status=status,
        end_time_ms=end_time,
    )
    prescribed_wall = (
        str(measurement.get("scenario") or "") == "prescribed_moving_wall"
    )
    if prescribed_wall:
        # The OpenRadioss global-energy history does not include work supplied
        # by a massless prescribed wall.  Its raw final-energy percentage is
        # therefore not an applicable conservation check for a controlled
        # platen test.  Keep it below as a traceable solver diagnostic, but do
        # not present its generic preliminary warning as a model-quality issue.
        warnings[:] = [
            warning
            for warning in warnings
            if not warning.startswith("OpenRadioss reports final energy loss")
        ]
    qualified_energy_error = quality.get("energy_balance_max_error")
    if qualified_energy_error is None:
        qualified_energy_error = maximum_energy_error
    metrics = dict(measurements.get("metrics") or {})
    provenance = build_crash_provenance(
        mesh=mesh,
        material=material,
        impact=impact,
        constraints=constraints,
        solver_settings=solver_settings,
        deck_path=deck_path,
        engine_path=engine_deck_path,
        starter_executable=starter_executable,
        engine_executable=engine_executable,
        time_history_converter=solver_history.get("converter"),
        result_artifacts={
            "starter_output": work_dir / f"{job_name}_0000.out",
            "engine_output": work_dir / f"{job_name}_0001.out",
            "time_history_binary": solver_history.get("source_file"),
            "time_history_csv": solver_history.get("source_csv"),
        },
    )
    manifest_path = write_crash_manifest(
        work_dir,
        provenance=provenance,
        quality=quality,
        metrics=metrics,
    )
    return {
        "type": "crash",
        "backend": "OpenRadioss",
        "external_status": status,
        "mesh": mesh,
        "displacement": displacement,
        "stress": stress,
        "element_stress": element_stress,
        "visualization_mode": visualization_mode,
        "disp_scale": displacement_scale,
        "frames": frames,
        "peak_displacement": peak_displacement,
        "peak_stress": peak_stress,
        "final_frame_displacement": final_peak_displacement,
        "final_frame_stress": final_peak_stress,
        "plastic_strain": last.get("eps_p_cell"),
        "failed_elements": failed_elements,
        "n_failed": int(
            np.count_nonzero(
                np.asarray(
                    failed_elements if failed_elements is not None else [],
                    dtype=float,
                )
                >= 0.5
            )
        ),
        "absorbed_energy": absorbed_energy_kj * 1.0e6,
        "absorbed_energy_kj": absorbed_energy_kj,
        "peak_force": float(metrics.get("peak_crushing_force_kN") or 0.0),
        "mean_force": float(metrics.get("mean_crushing_force_kN") or 0.0),
        "crush_force_efficiency": float(
            metrics.get("crush_force_efficiency") or 0.0
        ),
        "specific_energy_absorption": float(
            metrics.get("specific_energy_absorption_kJ_kg") or 0.0
        ),
        "structural_mass_kg": float(metrics.get("structural_mass_kg") or 0.0),
        "crush_distance": float(metrics.get("useful_crush_stroke_mm") or 0.0),
        "peak_acceleration_g": float(metrics.get("peak_acceleration_g") or 0.0),
        "delta_v": float(metrics.get("delta_v_m_s") or 0.0),
        "energy_balance_max_error": qualified_energy_error,
        "energy_balance_final_error": (
            qualified_energy_error if prescribed_wall else final_energy_error
        ),
        "solver_energy_balance_max_error": maximum_energy_error,
        "solver_energy_balance_final_error": final_energy_error,
        "mass_balance_max_error": maximum_mass_error,
        "wall": wall,
        "end_time": end_time,
        "time_history": solver_history or time_history,
        "histories": measurements,
        "crash_metrics": metrics,
        "quality": quality,
        "quality_status": quality.get("status"),
        "numerical_status": quality.get("numerical_status"),
        "physical_validation_status": quality.get("physical_validation_status"),
        "ml_eligible": bool(quality.get("ml_eligible")),
        "provenance": provenance,
        "manifest_file": str(manifest_path),
        "input_file": str(deck_path),
        "engine_file": str(engine_deck_path),
        "work_dir": str(work_dir),
        "solver_executable": starter_executable,
        "secondary_solver_executable": engine_executable,
        "solver_log": solver_log,
        "warnings": warnings,
        "message": "OpenRadioss solve complete; animation frames imported.",
    }


def build_generated_fallback_result(
    *,
    mesh: Any,
    status: str,
    visualization_mode: str,
    displacement_scale: float,
    wall: Any,
    end_time: float,
    deck_path: Path,
    engine_deck_path: Path,
    work_dir: Path,
    starter_executable: str | None,
    engine_executable: str | None,
    solver_log: str,
    warnings: list[str],
    material: dict[str, Any],
    impact: dict[str, Any],
    constraints: list[dict[str, Any]],
    solver_settings: dict[str, Any],
    solver_history: dict[str, Any],
    measurement: dict[str, Any],
    job_name: str,
    acceleration_cfc: int,
    force_cfc: int,
) -> dict[str, Any]:
    """Build the deck-only or no-animation result for a generated model."""
    measurement = dict(measurement)
    measurement["source_point_count"] = int(np.asarray(mesh.p).shape[1])
    measurement["material_validation"] = dict(material.get("validation") or {})
    measurement["structural_mass_kg"] = compute_structural_mass_kg(mesh, material)
    measurements = build_crash_measurements(
        solver_history=solver_history,
        frames=[],
        measurement=measurement,
        acceleration_cfc=acceleration_cfc,
        force_cfc=force_cfc,
    )
    quality = evaluate_crash_quality(
        measurements=measurements,
        external_status=status,
        end_time_ms=end_time,
    )
    metrics = dict(measurements.get("metrics") or {})
    provenance = build_crash_provenance(
        mesh=mesh,
        material=material,
        impact=impact,
        constraints=constraints,
        solver_settings=solver_settings,
        deck_path=deck_path,
        engine_path=engine_deck_path,
        starter_executable=starter_executable,
        engine_executable=engine_executable,
        time_history_converter=solver_history.get("converter"),
        result_artifacts={
            "starter_output": work_dir / f"{job_name}_0000.out",
            "engine_output": work_dir / f"{job_name}_0001.out",
            "time_history_binary": solver_history.get("source_file"),
            "time_history_csv": solver_history.get("source_csv"),
        },
    )
    manifest_path = write_crash_manifest(
        work_dir,
        provenance=provenance,
        quality=quality,
        metrics=metrics,
    )
    return {
        "type": "external_solver",
        "backend": "OpenRadioss",
        "external_status": status,
        "mesh": mesh,
        "visualization_mode": visualization_mode,
        "disp_scale": displacement_scale,
        "wall": wall,
        "end_time": end_time,
        "time_history": solver_history,
        "histories": measurements,
        "crash_metrics": metrics,
        "quality": quality,
        "quality_status": quality.get("status"),
        "numerical_status": quality.get("numerical_status"),
        "physical_validation_status": quality.get("physical_validation_status"),
        "ml_eligible": bool(quality.get("ml_eligible")),
        "provenance": provenance,
        "manifest_file": str(manifest_path),
        "input_file": str(deck_path),
        "engine_file": str(engine_deck_path),
        "work_dir": str(work_dir),
        "solver_executable": starter_executable,
        "secondary_solver_executable": engine_executable,
        "solver_log": solver_log,
        "warnings": warnings,
        "message": (
            "OpenRadioss-compatible keyword deck generated. Enable external "
            "execution and configure starter/engine to run the solve from PyLCSS."
            if status == "deck_written"
            else "OpenRadioss run finished but no animation frames could be imported."
        ),
    }
