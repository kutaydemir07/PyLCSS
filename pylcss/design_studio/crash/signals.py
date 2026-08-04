# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash measurement contract and deterministic signal processing.

The routines here keep raw solver channels immutable and create a separately
labelled processed view.  Acceleration and force filters use zero-phase,
four-pole-equivalent Butterworth profiles at the conventional SAE J211 /
ISO 6487 Channel Frequency Class cut-offs.  This is a post-processing
implementation; it does not certify the upstream acquisition hardware.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, sosfiltfilt


STANDARD_GRAVITY_M_S2 = 9.80665
FORCE_DECK_UNIT_TO_KN = 1000.0

# Nominal -3 dB cut-offs associated with the common Channel Frequency Classes.
CFC_CUTOFF_HZ = {
    60: 100.0,
    180: 300.0,
    600: 1000.0,
    1000: 1650.0,
}


def _array(value: object) -> np.ndarray:
    try:
        result = np.asarray(value if value is not None else [], dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return result


def _serialise(value: np.ndarray | Sequence[float]) -> list[float]:
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def _clean_time_series(
    time_ms: Iterable[float],
    values: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    t = _array(time_ms)
    y = _array(values)
    n = min(t.size, y.size)
    if n == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    t = t[:n]
    y = y[:n]
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    if t.size == 0:
        return t, y
    order = np.argsort(t, kind="stable")
    t = t[order]
    y = y[order]
    unique_t, first = np.unique(t, return_index=True)
    return unique_t, y[first]


def resample_channel(
    source_time_ms: Iterable[float],
    source_values: Iterable[float],
    target_time_ms: Iterable[float],
) -> np.ndarray:
    """Linearly resample a finite signal and hold its end values."""
    t, y = _clean_time_series(source_time_ms, source_values)
    target = _array(target_time_ms)
    if target.size == 0:
        return np.asarray([], dtype=float)
    if t.size == 0:
        return np.zeros(target.size, dtype=float)
    if t.size == 1:
        return np.full(target.size, y[0], dtype=float)
    return np.interp(target, t, y, left=y[0], right=y[-1])


def cfc_filter(
    time_ms: Iterable[float],
    values: Iterable[float],
    cfc: int,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Apply a phase-neutral Channel Frequency Class low-pass profile.

    A two-pole Butterworth section is run forward and backward, yielding a
    zero-phase, four-pole-equivalent response.  Irregular histories are first
    resampled to their median cadence and returned on the original time grid.
    """
    t, y = _clean_time_series(time_ms, values)
    metadata: Dict[str, object] = {
        "cfc": int(cfc),
        "cutoff_hz": CFC_CUTOFF_HZ.get(int(cfc)),
        "method": "zero_phase_two_pole_forward_reverse_butterworth",
        "applied": False,
    }
    if t.size < 8 or int(cfc) not in CFC_CUTOFF_HZ:
        metadata["reason"] = "insufficient_samples_or_unknown_cfc"
        return y.copy(), metadata

    delta_ms = np.diff(t)
    delta_ms = delta_ms[np.isfinite(delta_ms) & (delta_ms > 0.0)]
    if delta_ms.size == 0:
        metadata["reason"] = "invalid_time_axis"
        return y.copy(), metadata
    dt_ms = float(np.median(delta_ms))
    sample_rate_hz = 1000.0 / dt_ms
    cutoff_hz = float(CFC_CUTOFF_HZ[int(cfc)])
    metadata["sample_rate_hz"] = sample_rate_hz
    if cutoff_hz >= 0.95 * 0.5 * sample_rate_hz:
        metadata["reason"] = "sample_rate_below_filter_requirement"
        return y.copy(), metadata

    uniform_t = np.arange(t[0], t[-1] + 0.25 * dt_ms, dt_ms)
    if uniform_t.size < 8:
        metadata["reason"] = "insufficient_uniform_samples"
        return y.copy(), metadata
    uniform_y = np.interp(uniform_t, t, y)
    sos = butter(2, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    try:
        filtered_uniform = sosfiltfilt(sos, uniform_y)
    except ValueError:
        metadata["reason"] = "signal_too_short_for_zero_phase_padding"
        return y.copy(), metadata
    metadata["applied"] = True
    return np.interp(t, uniform_t, filtered_uniform), metadata


def _integrate(time_ms: np.ndarray, values: np.ndarray) -> np.ndarray:
    if time_ms.size < 2 or values.size != time_ms.size:
        return np.zeros(time_ms.size, dtype=float)
    return cumulative_trapezoid(values, time_ms * 1.0e-3, initial=0.0)


def _differentiate(time_ms: np.ndarray, values: np.ndarray) -> np.ndarray:
    if time_ms.size < 3 or values.size != time_ms.size:
        return np.zeros(time_ms.size, dtype=float)
    return np.gradient(values, time_ms * 1.0e-3, edge_order=2)


def _frame_reference_channels(
    frames: Sequence[Mapping[str, object]],
    node_ids_1based: Sequence[int],
    support_ids_1based: Sequence[int],
    axis: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Average selected animation nodes into scalar reference channels."""
    times = []
    displacements = []
    velocities = []
    accelerations = []
    selection = np.asarray(node_ids_1based, dtype=int).reshape(-1) - 1
    support = np.asarray(support_ids_1based, dtype=int).reshape(-1) - 1

    for frame in frames:
        flat = _array(frame.get("displacement"))
        n_points = flat.size // 3
        if n_points <= 0:
            continue
        frame_node_ids = np.asarray(
            frame.get("node_ids", []), dtype=int
        ).reshape(-1)
        if frame_node_ids.size == n_points:
            requested_ids = selection + 1
            support_ids = support + 1
            valid = np.flatnonzero(
                np.isin(frame_node_ids, requested_ids)
            )
            valid_support = np.flatnonzero(
                np.isin(frame_node_ids, support_ids)
            )
        else:
            valid = selection[(selection >= 0) & (selection < n_points)]
            valid_support = support[
                (support >= 0) & (support < n_points)
            ]
        if valid.size == 0:
            valid = np.arange(n_points, dtype=int)
        disp = flat[: 3 * n_points].reshape(n_points, 3)
        reference_disp = float(np.mean(disp[valid] @ axis))
        if valid_support.size:
            reference_disp -= float(np.mean(disp[valid_support] @ axis))

        vel = np.asarray(frame.get("velocity", []), dtype=float)
        if vel.shape != (n_points, 3):
            vel = np.zeros((n_points, 3), dtype=float)
        acc = np.asarray(frame.get("acceleration", []), dtype=float)
        if acc.shape != (n_points, 3):
            acc = np.zeros((n_points, 3), dtype=float)

        times.append(float(frame.get("time", 0.0)))
        displacements.append(reference_disp)
        velocities.append(float(np.mean(vel[valid] @ axis)))
        accelerations.append(float(np.mean(acc[valid] @ axis)))

    t = _array(times)
    disp = _array(displacements)
    vel = _array(velocities)
    acc = _array(accelerations)
    if acc.size and np.max(np.abs(acc)) <= 1.0e-15 and vel.size == t.size:
        # Animation converters/releases do not all expose /ANIM/VECT/ACC.
        # Differentiating the selected velocity is the documented fallback.
        acc = _differentiate(t, vel) / 1000.0  # m/s² -> mm/ms²
    return {
        "time_ms": t,
        "displacement_mm": disp,
        "velocity_m_s": vel,
        "acceleration_mm_ms2": acc,
    }


def _frame_rigid_wall_channels(
    frames: Sequence[Mapping[str, object]],
    axis: np.ndarray,
    source_point_count: int = 0,
) -> Dict[str, np.ndarray]:
    """Extract independent motion of the native rigid-wall main node."""
    times = []
    displacements = []
    velocities = []
    accelerations = []
    node_id = None
    for frame in frames:
        wall = frame.get("rigid_wall_reference")
        if not isinstance(wall, Mapping) and source_point_count > 0:
            frame_node_ids = np.asarray(
                frame.get("node_ids", []), dtype=int
            ).reshape(-1)
            candidates = np.flatnonzero(
                frame_node_ids > int(source_point_count)
            )
            flat_displacement = _array(frame.get("displacement"))
            frame_velocity = np.asarray(
                frame.get("velocity", []), dtype=float
            )
            frame_acceleration = np.asarray(
                frame.get("acceleration", []), dtype=float
            )
            if candidates.size and flat_displacement.size % 3 == 0:
                index = int(
                    candidates[np.argmax(frame_node_ids[candidates])]
                )
                displacement_vectors = flat_displacement.reshape(-1, 3)
                if index < displacement_vectors.shape[0]:
                    wall = {
                        "node_id": int(frame_node_ids[index]),
                        "displacement": displacement_vectors[index],
                        "velocity": (
                            frame_velocity[index]
                            if frame_velocity.ndim == 2
                            and index < frame_velocity.shape[0]
                            else np.zeros(3)
                        ),
                        "acceleration": (
                            frame_acceleration[index]
                            if frame_acceleration.ndim == 2
                            and index < frame_acceleration.shape[0]
                            else np.zeros(3)
                        ),
                    }
        if not isinstance(wall, Mapping):
            continue
        displacement = _array(wall.get("displacement"))
        velocity = _array(wall.get("velocity"))
        acceleration = _array(wall.get("acceleration"))
        if (
            displacement.size != 3
            or velocity.size != 3
            or acceleration.size != 3
        ):
            continue
        node_id = wall.get("node_id", node_id)
        times.append(float(frame.get("time", 0.0)))
        displacements.append(float(displacement @ axis))
        velocities.append(float(velocity @ axis))
        accelerations.append(float(acceleration @ axis))
    return {
        "time_ms": _array(times),
        "displacement_mm": _array(displacements),
        "velocity_m_s": _array(velocities),
        "acceleration_mm_ms2": _array(accelerations),
        "node_id": node_id,
    }


def _project_wall_force(history: Mapping[str, object], axis: np.ndarray) -> np.ndarray:
    force_components = []
    for key in (
        "rigid_wall_force_x_raw",
        "rigid_wall_force_y_raw",
        "rigid_wall_force_z_raw",
    ):
        force_components.append(_array(history.get(key)))
    n_force = min(
        (arr.size for arr in force_components),
        default=0,
    )
    if n_force > 0:
        vector = np.column_stack(
            [arr[:n_force] for arr in force_components]
        )
        return np.abs(vector @ axis) * FORCE_DECK_UNIT_TO_KN

    impulse_components = [
        _array(history.get(key))
        for key in (
            "rigid_wall_impulse_x_raw",
            "rigid_wall_impulse_y_raw",
            "rigid_wall_impulse_z_raw",
        )
    ]
    time_ms = _array(history.get("time_ms", history.get("t_ms")))
    n_impulse = min(
        [arr.size for arr in impulse_components] + [time_ms.size],
        default=0,
    )
    if n_impulse >= 3:
        impulse = np.column_stack(
            [arr[:n_impulse] for arr in impulse_components]
        )
        projected_impulse = impulse @ axis
        # /TFILE stores impulse in tonne*mm/ms. Differentiation against the
        # physical millisecond axis yields tonne*mm/ms², equal to 1000 kN.
        projected_force = np.gradient(
            projected_impulse,
            time_ms[:n_impulse],
            edge_order=2,
        )
        return np.abs(projected_force) * FORCE_DECK_UNIT_TO_KN

    # Requalification can start from the preserved measurement contract,
    # where the projected force is already stored in kN.
    return _array(history.get("rigid_wall_force_kN"))


def _project_wall_impulse(
    history: Mapping[str, object],
    axis: np.ndarray,
) -> np.ndarray:
    components = [
        _array(history.get(key))
        for key in (
            "rigid_wall_impulse_x_raw",
            "rigid_wall_impulse_y_raw",
            "rigid_wall_impulse_z_raw",
        )
    ]
    n = min((arr.size for arr in components), default=0)
    if n <= 0:
        return np.asarray([], dtype=float)
    vector = np.column_stack([arr[:n] for arr in components])
    # tonne*mm/ms = 1000 N*s.
    return np.abs(vector @ axis) * 1000.0


def _copy_global_channels(
    solver_history: Mapping[str, object],
    target_time: np.ndarray,
) -> Dict[str, list[float]]:
    source_time = _array(
        solver_history.get("time_ms", solver_history.get("t_ms"))
    )
    mapping = {
        "kinetic_energy_kj": ("kinetic_energy_kj", "ke_kj"),
        "internal_energy_kj": ("internal_energy_kj", "ie_kj"),
        "total_energy_kj": ("total_energy_kj",),
        "translational_total_energy_kj": (
            "translational_total_energy_kj",
        ),
        "delta_total_energy_kj": ("delta_total_energy_kj",),
        "delta_total_energy_relative": (
            "delta_total_energy_relative",
        ),
        "rotational_kinetic_energy_kj": (
            "rotational_kinetic_energy_kj",
        ),
        "contact_energy_kj": ("contact_energy_kj",),
        "contact_elastic_energy_kj": ("contact_elastic_energy_kj",),
        "contact_friction_energy_kj": ("contact_friction_energy_kj",),
        "contact_damping_energy_kj": ("contact_damping_energy_kj",),
        "hourglass_energy_kj": ("hourglass_energy_kj",),
        "external_work_kj": ("external_work_kj",),
        "timestep_ms": ("timestep_ms",),
        "momentum_x": ("momentum_x",),
        "momentum_y": ("momentum_y",),
        "momentum_z": ("momentum_z",),
        "global_velocity_x": ("global_velocity_x",),
        "global_velocity_y": ("global_velocity_y",),
        "global_velocity_z": ("global_velocity_z",),
    }
    output: Dict[str, list[float]] = {}
    for target, aliases in mapping.items():
        values = np.asarray([], dtype=float)
        for alias in aliases:
            values = _array(solver_history.get(alias))
            if values.size:
                break
        if values.size and source_time.size:
            output[target] = _serialise(
                resample_channel(source_time, values, target_time)
            )

    mass_tonne = _array(solver_history.get("mass_tonne"))
    if mass_tonne.size and source_time.size:
        output["mass_kg"] = _serialise(
            1000.0 * resample_channel(source_time, mass_tonne, target_time)
        )
    elif _array(solver_history.get("mass_kg")).size and source_time.size:
        output["mass_kg"] = _serialise(
            resample_channel(
                source_time,
                solver_history.get("mass_kg"),
                target_time,
            )
        )
    elif solver_history.get("total_mass_kg") is not None:
        output["mass_kg"] = _serialise(
            np.full(target_time.size, float(solver_history["total_mass_kg"]))
        )
    return output


def _global_body_kinematics(
    solver_history: Mapping[str, object],
    target_time: np.ndarray,
    axis: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Derive independent centre-of-mass motion from global momentum and mass.

    In the PyLCSS moving-body setup the added impactor mass is distributed over
    the deformable body, so there is no rigid impactor reference node.  Radioss
    global momentum divided by global mass is the physically meaningful
    centre-of-mass velocity and is independent of the rigid-wall force history.
    """
    source_time = _array(
        solver_history.get("time_ms", solver_history.get("t_ms"))
    )
    momentum_components = [
        _array(solver_history.get(key))
        for key in ("momentum_x", "momentum_y", "momentum_z")
    ]
    mass_tonne = _array(solver_history.get("mass_tonne"))
    n = min(
        [source_time.size, mass_tonne.size]
        + [component.size for component in momentum_components],
        default=0,
    )
    if n < 2:
        return {
            "time_ms": np.asarray([], dtype=float),
            "velocity_m_s": np.asarray([], dtype=float),
            "displacement_mm": np.asarray([], dtype=float),
            "acceleration_m_s2": np.asarray([], dtype=float),
        }
    source_time = source_time[:n]
    momentum = np.column_stack(
        [component[:n] for component in momentum_components]
    )
    mass = mass_tonne[:n]
    valid = np.isfinite(source_time) & np.isfinite(mass) & (mass > 0.0)
    valid &= np.all(np.isfinite(momentum), axis=1)
    if np.count_nonzero(valid) < 2:
        return {
            "time_ms": np.asarray([], dtype=float),
            "velocity_m_s": np.asarray([], dtype=float),
            "displacement_mm": np.asarray([], dtype=float),
            "acceleration_m_s2": np.asarray([], dtype=float),
        }
    source_time = source_time[valid]
    projected_momentum = momentum[valid] @ axis
    # tonne*mm/ms divided by tonne equals mm/ms, numerically equal to m/s.
    source_velocity = projected_momentum / mass[valid]
    velocity = resample_channel(source_time, source_velocity, target_time)
    displacement = _integrate(target_time, velocity) * 1000.0
    acceleration = _differentiate(target_time, velocity)
    return {
        "time_ms": target_time.copy(),
        "velocity_m_s": velocity,
        "displacement_mm": displacement,
        "acceleration_m_s2": acceleration,
    }


def build_crash_measurements(
    solver_history: Mapping[str, object],
    frames: Sequence[Mapping[str, object]],
    measurement: Mapping[str, object],
    acceleration_cfc: int = 60,
    force_cfc: int = 600,
) -> Dict[str, object]:
    """Build raw/processed histories and standard crashworthiness metrics."""
    axis = _array(measurement.get("impact_axis"))
    if axis.size != 3 or np.linalg.norm(axis) <= 0.0:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    axis = axis / np.linalg.norm(axis)

    frame_channels = _frame_reference_channels(
        frames,
        measurement.get("reference_node_ids", []),
        measurement.get("support_node_ids", []),
        axis,
    )
    wall_channels = _frame_rigid_wall_channels(
        frames,
        axis,
        source_point_count=int(
            measurement.get("source_point_count") or 0
        ),
    )
    history_time = _array(
        solver_history.get("time_ms", solver_history.get("t_ms"))
    )
    frame_time = frame_channels["time_ms"]
    if history_time.size >= 2:
        time_ms = history_time
    elif frame_time.size:
        time_ms = frame_time
    else:
        time_ms = np.asarray([], dtype=float)

    wall_force_kn = _project_wall_force(solver_history, axis)
    wall_impulse_ns = _project_wall_impulse(solver_history, axis)
    if wall_force_kn.size and history_time.size:
        force_kn = resample_channel(history_time, wall_force_kn, time_ms)
        force_source = (
            "OpenRadioss /TH/RWALL impulse differentiated on physical time"
            if _array(
                solver_history.get("rigid_wall_impulse_x_raw")
            ).size
            else "direct rigid-wall force channel"
        )
    else:
        force_kn = np.zeros(time_ms.size, dtype=float)
        force_source = "unavailable"

    scenario = str(measurement.get("scenario") or "")
    impactor_mass_kg = max(float(measurement.get("impactor_mass_kg") or 0.0), 0.0)
    initial_speed = abs(float(measurement.get("initial_speed_m_s") or 0.0))
    reference_disp = resample_channel(
        frame_time, frame_channels["displacement_mm"], time_ms
    )
    reference_vel = resample_channel(
        frame_time, frame_channels["velocity_m_s"], time_ms
    )
    reference_acc_mm_ms2 = resample_channel(
        frame_time, frame_channels["acceleration_mm_ms2"], time_ms
    )
    wall_time = wall_channels["time_ms"]
    independent_wall_kinematics = wall_time.size >= 2
    if independent_wall_kinematics:
        wall_displacement_mm = resample_channel(
            wall_time,
            wall_channels["displacement_mm"],
            time_ms,
        )
        wall_displacement_mm = (
            wall_displacement_mm - wall_displacement_mm[0]
        )
        wall_velocity_m_s = resample_channel(
            wall_time,
            wall_channels["velocity_m_s"],
            time_ms,
        )
        wall_acceleration_mm_ms2 = resample_channel(
            wall_time,
            wall_channels["acceleration_mm_ms2"],
            time_ms,
        )
    else:
        wall_displacement_mm = np.zeros(time_ms.size, dtype=float)
        wall_velocity_m_s = np.zeros(time_ms.size, dtype=float)
        wall_acceleration_mm_ms2 = np.zeros(time_ms.size, dtype=float)

    body_channels = _global_body_kinematics(
        solver_history,
        time_ms,
        axis,
    )
    independent_body_kinematics = body_channels["time_ms"].size >= 2

    if scenario == "moving_body_fixed_wall" and time_ms.size:
        if independent_body_kinematics:
            body_velocity_m_s = body_channels["velocity_m_s"]
            body_acceleration_m_s2 = body_channels["acceleration_m_s2"]
            crush_mm = body_channels["displacement_mm"]
            pulse_g = np.maximum(
                -body_acceleration_m_s2 / STANDARD_GRAVITY_M_S2,
                0.0,
            )
            pulse_source = (
                "independent OpenRadioss global momentum / global mass"
            )
            displacement_source = (
                "integrated OpenRadioss centre-of-mass velocity"
            )
        elif impactor_mass_kg > 0.0 and force_kn.size:
            deceleration_m_s2 = force_kn * 1000.0 / impactor_mass_kg
            body_velocity_m_s = np.maximum(
                initial_speed - _integrate(time_ms, deceleration_m_s2),
                0.0,
            )
            crush_mm = _integrate(time_ms, body_velocity_m_s) * 1000.0
            pulse_g = deceleration_m_s2 / STANDARD_GRAVITY_M_S2
            pulse_source = "rigid-wall force / moving-body mass fallback"
            displacement_source = "integrated force-derived body velocity fallback"
        else:
            body_velocity_m_s = reference_vel
            crush_mm = np.abs(reference_disp)
            pulse_g = (
                np.abs(reference_acc_mm_ms2)
                * 1000.0
                / STANDARD_GRAVITY_M_S2
            )
            pulse_source = "impact-face structural acceleration fallback"
            displacement_source = "impact-face structural displacement fallback"
    elif scenario == "fixed_specimen_moving_impactor" and time_ms.size:
        if independent_wall_kinematics:
            crush_mm = wall_displacement_mm
            pulse_g = np.maximum(
                -wall_acceleration_mm_ms2
                * 1000.0
                / STANDARD_GRAVITY_M_S2,
                0.0,
            )
            pulse_source = (
                "independent OpenRadioss rigid-wall main-node acceleration"
            )
            displacement_source = (
                "independent OpenRadioss rigid-wall main-node displacement"
            )
        elif impactor_mass_kg > 0.0 and force_kn.size:
            deceleration_m_s2 = force_kn * 1000.0 / impactor_mass_kg
            impactor_speed = np.maximum(
                initial_speed - _integrate(time_ms, deceleration_m_s2),
                0.0,
            )
            crush_mm = _integrate(time_ms, impactor_speed) * 1000.0
            pulse_g = deceleration_m_s2 / STANDARD_GRAVITY_M_S2
            pulse_source = "finite-mass rigid-wall force / impactor mass"
            displacement_source = "integrated finite-mass rigid-wall velocity"
        else:
            crush_mm = initial_speed * time_ms
            pulse_g = np.abs(reference_acc_mm_ms2) * 1000.0 / STANDARD_GRAVITY_M_S2
            pulse_source = "impact-face structural acceleration"
            displacement_source = "prescribed rigid-wall travel"
    elif scenario == "prescribed_moving_wall" and time_ms.size:
        crush_mm = initial_speed * time_ms
        pulse_g = np.abs(reference_acc_mm_ms2) * 1000.0 / STANDARD_GRAVITY_M_S2
        pulse_source = "impact-face structural acceleration"
        displacement_source = "prescribed rigid-wall travel"
    else:
        crush_mm = np.abs(reference_disp)
        pulse_g = np.abs(reference_acc_mm_ms2) * 1000.0 / STANDARD_GRAVITY_M_S2
        pulse_source = "reference-node structural acceleration"
        displacement_source = "reference-to-support relative displacement"

    crush_mm = np.maximum.accumulate(np.maximum(crush_mm, 0.0))
    if not np.any(force_kn) and impactor_mass_kg > 0.0 and pulse_g.size:
        force_kn = pulse_g * STANDARD_GRAVITY_M_S2 * impactor_mass_kg / 1000.0
        force_source = (
            "independent rigid-wall acceleration × measured impactor mass"
            if independent_wall_kinematics
            else "reference acceleration × impactor mass fallback"
        )

    filtered_force, force_filter = cfc_filter(time_ms, force_kn, force_cfc)
    filtered_pulse, pulse_filter = cfc_filter(time_ms, pulse_g, acceleration_cfc)
    filtered_force = np.maximum(filtered_force, 0.0)
    filtered_pulse = np.maximum(filtered_pulse, 0.0)

    global_channels = _copy_global_channels(solver_history, time_ms)
    raw = {
        "time_ms": _serialise(time_ms),
        "rigid_wall_force_kN": _serialise(force_kn),
        "crush_displacement_mm": _serialise(crush_mm),
        "reference_velocity_m_s": _serialise(reference_vel),
        "acceleration_g": _serialise(pulse_g),
        **global_channels,
    }
    if wall_impulse_ns.size and history_time.size:
        raw["rigid_wall_impulse_N_s"] = _serialise(
            resample_channel(
                history_time,
                wall_impulse_ns,
                time_ms,
            )
        )
    if independent_wall_kinematics:
        raw.update(
            {
                "rigid_wall_displacement_mm": _serialise(
                    wall_displacement_mm
                ),
                "rigid_wall_velocity_m_s": _serialise(
                    wall_velocity_m_s
                ),
                "rigid_wall_acceleration_g": _serialise(
                    -wall_acceleration_mm_ms2
                    * 1000.0
                    / STANDARD_GRAVITY_M_S2
                ),
            }
        )
    processed = {
        **raw,
        "rigid_wall_force_kN": _serialise(filtered_force),
        "acceleration_g": _serialise(filtered_pulse),
    }

    useful_crush = float(crush_mm[-1]) if crush_mm.size else 0.0
    peak_force = float(np.max(filtered_force)) if filtered_force.size else 0.0
    peak_acceleration = (
        float(np.max(filtered_pulse)) if filtered_pulse.size else 0.0
    )
    force_energy_kj = 0.0
    if time_ms.size >= 2 and useful_crush > 0.0:
        force_energy_kj = float(
            np.trapezoid(force_kn, crush_mm) / 1000.0
        )
    ie = _array(global_channels.get("internal_energy_kj"))
    absorbed_energy_kj = float(ie[-1]) if ie.size else force_energy_kj
    mean_force = (
        force_energy_kj * 1000.0 / useful_crush if useful_crush > 0.0 else 0.0
    )
    cfe = mean_force / peak_force if peak_force > 0.0 else 0.0
    mass = _array(global_channels.get("mass_kg"))
    structural_mass_kg = float(
        measurement.get("structural_mass_kg")
        or (mass[0] if mass.size else 0.0)
    )
    sea = (
        absorbed_energy_kj / structural_mass_kg
        if structural_mass_kg > 0.0
        else 0.0
    )
    # Integral checks use raw channels. Applying different CFC profiles to
    # structural force (normally CFC 600) and occupant pulse (normally CFC 60)
    # is correct for peak reporting but can change short-record edge integrals.
    if scenario == "moving_body_fixed_wall" and independent_body_kinematics:
        body_velocity_m_s = body_channels["velocity_m_s"]
        delta_v = float(
            abs(body_velocity_m_s[0] - body_velocity_m_s[-1])
        )
        final_speed = abs(float(body_velocity_m_s[-1]))
    elif independent_wall_kinematics and wall_velocity_m_s.size:
        delta_v = float(
            abs(wall_velocity_m_s[0] - wall_velocity_m_s[-1])
        )
        final_speed = abs(float(wall_velocity_m_s[-1]))
    else:
        delta_v = (
            float(
                _integrate(
                    time_ms,
                    pulse_g * STANDARD_GRAVITY_M_S2,
                )[-1]
            )
            if time_ms.size
            else 0.0
        )
        final_speed = max(initial_speed - delta_v, 0.0)
    if wall_impulse_ns.size:
        impulse_ns = float(
            abs(wall_impulse_ns[-1] - wall_impulse_ns[0])
        )
    else:
        impulse_ns = (
            float(np.trapezoid(force_kn, time_ms))
            if time_ms.size >= 2
            else 0.0
        )
    threshold = 0.05 * peak_acceleration
    active = np.flatnonzero(filtered_pulse >= threshold) if threshold > 0.0 else []
    pulse_duration = (
        float(time_ms[active[-1]] - time_ms[active[0]])
        if len(active) >= 2
        else 0.0
    )
    impactor_ke_loss_kj = (
        0.5
        * impactor_mass_kg
        * (initial_speed ** 2 - final_speed ** 2)
        / 1000.0
        if impactor_mass_kg > 0.0
        else 0.0
    )

    metrics = {
        "peak_crushing_force_kN": peak_force,
        "mean_crushing_force_kN": mean_force,
        "crush_force_efficiency": cfe,
        "absorbed_energy_kJ": absorbed_energy_kj,
        "force_displacement_energy_kJ": force_energy_kj,
        "impactor_kinetic_energy_loss_kJ": impactor_ke_loss_kj,
        "specific_energy_absorption_kJ_kg": sea,
        "structural_mass_kg": structural_mass_kg,
        "useful_crush_stroke_mm": useful_crush,
        "peak_acceleration_g": peak_acceleration,
        "delta_v_m_s": delta_v,
        "pulse_duration_ms": pulse_duration,
        "force_impulse_N_s": impulse_ns,
    }
    units = {
        "time_ms": "ms",
        "rigid_wall_force_kN": "kN",
        "crush_displacement_mm": "mm",
        "reference_velocity_m_s": "m/s",
        "acceleration_g": "g",
        "rigid_wall_displacement_mm": "mm",
        "rigid_wall_velocity_m_s": "m/s",
        "rigid_wall_acceleration_g": "g",
        "rigid_wall_impulse_N_s": "N·s",
        "kinetic_energy_kj": "kJ",
        "internal_energy_kj": "kJ",
        "total_energy_kj": "kJ",
        "contact_energy_kj": "kJ",
        "hourglass_energy_kj": "kJ",
        "external_work_kj": "kJ",
        "mass_kg": "kg",
        "timestep_ms": "ms",
    }
    return {
        "schema": "pylcss.crash.measurements",
        "schema_version": "1.0.0",
        "standard_basis": [
            "SAE J211/1 electronic impact instrumentation",
            "ISO 6487 road-vehicle impact instrumentation",
        ],
        "measurement": {
            **dict(measurement),
            "impact_axis": _serialise(axis),
            "positive_force_convention": "compression_positive",
            "positive_acceleration_convention": "deceleration_positive",
            "force_source": force_source,
            "displacement_source": displacement_source,
            "pulse_source": pulse_source,
            "rigid_wall_reference_node_id": wall_channels.get("node_id"),
            "independent_wall_kinematics": independent_wall_kinematics,
            "independent_body_kinematics": independent_body_kinematics,
        },
        "units": units,
        "raw": raw,
        "processed": processed,
        "processing": {
            "raw_preserved": True,
            "force_filter": force_filter,
            "acceleration_filter": pulse_filter,
            "integration": "trapezoidal_on_physical_time",
            "crush_displacement": "monotonic_envelope",
            "independent_wall_kinematics": independent_wall_kinematics,
            "independent_body_kinematics": independent_body_kinematics,
            "independent_impactor_kinematics": (
                independent_wall_kinematics
                if scenario == "fixed_specimen_moving_impactor"
                else independent_body_kinematics
            ),
        },
        "metrics": metrics,
    }
