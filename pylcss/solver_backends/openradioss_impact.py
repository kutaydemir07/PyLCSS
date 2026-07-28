# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash-event cards for PyLCSS-generated OpenRadioss keyword decks."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from pylcss.solver_backends.base import SolverBackendError
from pylcss.solver_backends.mesh import id_lines
from pylcss.solver_backends.selection import (
    dict_geometries,
    nodes_matching_condition,
    nodes_matching_geometries,
    normalize_geometries,
)
from pylcss.solver_backends.validation import (
    integer,
    nonnegative_float,
)


logger = logging.getLogger(__name__)


def _keyword_field(value: object) -> str:
    """Format one LS-DYNA keyword field without exceeding 10 columns."""

    if value is None:
        return " " * 10
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    else:
        number = float(value)
        text = f"{number:.6g}"
        if len(text) > 10:
            text = f"{number:.3e}"
    if len(text) > 10:
        raise ValueError(
            f"LS-DYNA keyword value does not fit a 10-column field: {value!r}"
        )
    return text.rjust(10)


def _keyword_card(*values: object) -> str:
    """Build one conventional 8-by-10-column LS-DYNA card."""

    if len(values) > 8:
        raise ValueError("An LS-DYNA keyword card can contain at most 8 fields.")
    return "".join(_keyword_field(value) for value in values)


@dataclass(frozen=True, slots=True)
class _ImpactState:
    velocity: np.ndarray
    speed: float
    direction: np.ndarray
    scenario: str
    scenario_label: str
    moving_body: bool
    prescribed_wall: bool
    wall_friction: float
    wall_gap: float


def _node_set(lines: list[str], set_id: int, nodes_1based: np.ndarray) -> None:
    lines.extend(["*SET_NODE_LIST", "$#     sid"])
    lines.append(f"{set_id}")
    lines.extend(id_lines(nodes_1based, per_line=8))


def _normalise_crash_scenario(value: Any) -> str:
    """Map current GUI labels and legacy saved values to backend scenario IDs."""
    text = str(value or "").strip().lower().replace("_", " ")
    if text.startswith("moving body") or text == "moving":
        return "moving_body_fixed_wall"
    if text.startswith("prescribed"):
        return "prescribed_moving_wall"
    return "fixed_specimen_moving_impactor"


def _scenario_label(scenario: str) -> str:
    if scenario == "moving_body_fixed_wall":
        return "Moving body + fixed wall"
    if scenario == "prescribed_moving_wall":
        return "Prescribed moving wall"
    return "Fixed specimen + moving impactor"


def _wall_friction(raw_value: Any, *, moving_body: bool) -> float:
    """Return rigid-wall Coulomb friction, preserving legacy defaults."""
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = -1.0
    if value >= 0.0:
        return value
    return 0.0 if moving_body else 0.08


def _impact_state(impact: dict[str, Any]) -> _ImpactState:
    try:
        velocity = np.asarray(
            impact.get("velocity", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise SolverBackendError(
            "Impact velocity must contain three finite numbers."
        ) from exc
    if velocity.size != 3 or not np.all(np.isfinite(velocity)):
        raise SolverBackendError("Impact velocity must contain three finite numbers.")

    speed = float(np.linalg.norm(velocity))
    direction = velocity / speed if speed > 0.0 else np.array([1.0, 0.0, 0.0])
    scenario = _normalise_crash_scenario(impact.get("application_scope"))
    moving_body = scenario == "moving_body_fixed_wall"
    try:
        wall_gap = max(float(impact.get("wall_gap_mm", 0.0) or 0.0), 0.0)
    except (TypeError, ValueError):
        wall_gap = 0.0

    return _ImpactState(
        velocity=velocity,
        speed=speed,
        direction=direction,
        scenario=scenario,
        scenario_label=_scenario_label(scenario),
        moving_body=moving_body,
        prescribed_wall=scenario == "prescribed_moving_wall",
        wall_friction=_wall_friction(
            impact.get("wall_friction"),
            moving_body=moving_body,
        ),
        wall_gap=wall_gap,
    )


def _append_constraints(
    lines: list[str],
    *,
    mesh: Any,
    constraints: list[dict[str, Any]],
    moving_body: bool,
    next_set_id: int,
    warnings: list[str],
) -> tuple[int, list[int]]:
    constrained_node_ids: list[int] = []
    if moving_body:
        if constraints:
            warnings.append(
                "Moving body + fixed wall crash: SPC constraints are ignored in this "
                "scope because the whole structure is a free-flying projectile and "
                "the rigid wall provides the reaction. To model a fixed-rear "
                "laboratory test, switch the ImpactCondition to "
                "'Fixed specimen + moving impactor'."
            )
        return next_set_id, constrained_node_ids

    for idx, constraint in enumerate(constraints, start=1):
        geometries = dict_geometries(constraint)
        condition = str(constraint.get("condition") or "").strip()
        if geometries:
            node_ids = nodes_matching_geometries(mesh, geometries) + 1
        elif condition:
            node_ids = (
                nodes_matching_condition(
                    mesh,
                    condition,
                    warnings=warnings,
                    label=f"Crash constraint {idx}",
                )
                + 1
            )
        else:
            warnings.append(
                f"Crash constraint {idx} has no face geometry or condition; skipped."
            )
            continue
        if len(node_ids) == 0:
            warnings.append(f"Crash constraint {idx} did not match any mesh nodes.")
            continue

        constrained_node_ids.extend(int(value) for value in node_ids)
        set_id = next_set_id
        next_set_id += 1
        _node_set(lines, set_id, node_ids)
        try:
            fixed_dofs = {
                integer(value, label=f"Crash constraint {idx} DOF")
                for value in constraint.get("fixed_dofs", [0, 1, 2])
            }
        except TypeError as exc:
            raise SolverBackendError(
                f"Crash constraint {idx} fixed_dofs must be a sequence."
            ) from exc
        if not fixed_dofs.issubset({0, 1, 2}):
            raise SolverBackendError(
                f"Crash constraint {idx} fixed_dofs may only contain 0, 1, or 2."
            )
        lines.extend(
            [
                "*BOUNDARY_SPC_SET",
                "$#     nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz",
                (
                    f"{set_id}, 0, {int(0 in fixed_dofs)}, "
                    f"{int(1 in fixed_dofs)}, {int(2 in fixed_dofs)}, 0, 0, 0"
                ),
            ]
        )

    return next_set_id, constrained_node_ids


def _impact_nodes(
    mesh: Any,
    points: np.ndarray,
    impact: dict[str, Any],
    state: _ImpactState,
) -> tuple[list[Any], np.ndarray]:
    impact_faces = normalize_geometries(impact.get("face_list", []))
    node_ids: np.ndarray
    if state.moving_body:
        node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    elif impact_faces:
        node_ids = (
            nodes_matching_geometries(
                mesh,
                impact_faces,
                tolerance=nonnegative_float(
                    impact.get("node_tolerance", 2.0),
                    label="Impact node-selection tolerance",
                ),
            )
            + 1
        )
    else:
        node_ids = np.arange(1, points.shape[0] + 1, dtype=int)
    return impact_faces, node_ids


def _warn_about_requested_travel(
    points: np.ndarray,
    impact_nodes: np.ndarray,
    constrained_node_ids: list[int],
    *,
    direction: np.ndarray,
    travel: float,
    warnings: list[str],
) -> None:
    if constrained_node_ids:
        impact_projection = points[np.asarray(impact_nodes, dtype=int) - 1] @ direction
        constrained_idx = np.asarray(constrained_node_ids, dtype=int) - 1
        constrained_idx = constrained_idx[
            (constrained_idx >= 0) & (constrained_idx < points.shape[0])
        ]
        if constrained_idx.size:
            support_projection = points[constrained_idx] @ direction
            downstream = (
                support_projection[:, None] - impact_projection[None, :]
            ).reshape(-1)
            downstream = downstream[downstream > 0.0]
            if downstream.size:
                available = float(np.min(downstream))
                if travel > 0.85 * available:
                    warnings.append(
                        "Fixed specimen + moving impactor: |velocity| * end_time is "
                        f"{travel:.1f} mm, but the nearest constrained "
                        f"support is only {available:.1f} mm along the "
                        "impact direction.  The moving wall can overrun "
                        "the supported end in the animation.  "
                        "Reduce end_time, velocity, or sled mass, or use "
                        "Moving body + fixed wall for a free-body barrier impact."
                    )
        return

    bbox_span = float(np.ptp(points @ direction))
    if bbox_span > 0.0 and travel > 0.85 * bbox_span:
        warnings.append(
            "Fixed specimen + moving impactor has no active constraints and the "
            f"requested stroke ({travel:.1f} mm) is close to or "
            "larger than the part length in the impact direction.  "
            "For a wall/barrier event, use Moving body + fixed wall."
        )


def _append_velocity(
    lines: list[str],
    *,
    points: np.ndarray,
    impact_nodes: np.ndarray,
    constrained_node_ids: list[int],
    state: _ImpactState,
    end_time: float,
    moving_rigid_wall: bool,
    warnings: list[str],
) -> None:
    if len(impact_nodes) == 0:
        warnings.append(
            "Impact condition did not match any nodes; no initial velocity exported."
        )
        return
    if state.speed <= 0.0:
        return

    if not state.moving_body:
        travel = state.speed * end_time
        if travel > 0.0:
            _warn_about_requested_travel(
                points,
                impact_nodes,
                constrained_node_ids,
                direction=state.direction,
                travel=travel,
                warnings=warnings,
            )
    if moving_rigid_wall:
        logger.info(
            "OpenRadioss deck: %s uses a moving rigid wall; mesh starts at rest",
            state.scenario_label,
        )
        return

    logger.info(
        "OpenRadioss deck: applying initial velocity to %d node(s), scenario=%s",
        len(impact_nodes),
        state.scenario_label,
    )
    lines.extend(
        [
            "*INITIAL_VELOCITY_NODE",
            "$#     nid        vx        vy        vz       vxr       vyr       vzr",
        ]
    )
    for node_id in impact_nodes:
        lines.append(
            f"{int(node_id)}, "
            f"{state.velocity[0]:.12g}, {state.velocity[1]:.12g}, "
            f"{state.velocity[2]:.12g}, 0, 0, 0"
        )


def _append_self_contact(lines: list[str]) -> None:
    lines.extend(
        [
            "*CONTACT_AUTOMATIC_SINGLE_SURFACE",
            "$#    ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr",
            "0",
            "$#      fs        fd        dc        vc       vdc    penchk        bt        dt",
            "0.08, 0.08",
            "$#     sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf",
            "1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0",
        ]
    )


def _append_stationary_wall(
    lines: list[str],
    *,
    mesh: Any,
    points: np.ndarray,
    impact_faces: list[Any],
    impact_nodes: np.ndarray,
    state: _ImpactState,
    next_set_id: int,
    out_meta: dict[str, Any] | None,
    warnings: list[str],
) -> int:
    if not state.moving_body or len(impact_nodes) == 0 or state.speed <= 0.0:
        return next_set_id

    projections = points @ state.direction
    leading_projection = float(np.max(projections))
    bbox_diagonal = float(
        np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    )
    automatic_gap = max(bbox_diagonal * 0.005, 0.1)
    gap = state.wall_gap if state.wall_gap > 0.0 else automatic_gap
    centroid = np.mean(points, axis=0)
    wall_point = centroid + state.direction * (
        leading_projection + gap - float(centroid @ state.direction)
    )
    wall_normal = -state.direction
    wall_head = wall_point + wall_normal
    wall_slave_set_id = next_set_id
    next_set_id += 1
    _node_set(
        lines,
        wall_slave_set_id,
        np.arange(1, points.shape[0] + 1, dtype=int),
    )
    lines.extend(
        [
            "*RIGIDWALL_PLANAR",
            "$#    nsid     nsedx   dsearch",
            f"{wall_slave_set_id}, 0, 0.0",
            "$#      xt        yt        zt        xh        yh        zh      fric",
            _keyword_card(
                wall_point[0], wall_point[1], wall_point[2],
                wall_head[0], wall_head[1], wall_head[2],
                state.wall_friction,
            ),
        ]
    )
    logger.info(
        "OpenRadioss deck: fixed planar wall point=%s normal=%s "
        "gap=%.3f mm friction=%.3g",
        wall_point.tolist(),
        wall_normal.tolist(),
        gap,
        state.wall_friction,
    )
    if out_meta is not None:
        out_meta["wall"] = {
            "type": "stationary",
            "pt": [float(value) for value in wall_point],
            "normal": [float(value) for value in wall_normal],
            "half_extent": float(0.6 * bbox_diagonal),
            "v0_mm_per_ms": 0.0,
            "velocity_dir": [0.0, 0.0, 0.0],
        }

    if impact_faces:
        face_node_indices = nodes_matching_geometries(mesh, impact_faces)
        if len(face_node_indices) > 0:
            face_projection = float(np.max(points[face_node_indices] @ state.direction))
            if leading_projection - face_projection > bbox_diagonal * 0.2:
                warnings.append(
                    "Moving body + fixed wall: the named impact face is at the TRAILING "
                    "edge in the velocity direction - the rear of the body will hit "
                    "the wall first, not the intended impact face.  For a frontal "
                    "crash where the +X face hits the barrier first, set "
                    "velocity_x to a POSITIVE value (e.g. +20 mm/ms)."
                )
    return next_set_id


def _append_moving_wall(
    lines: list[str],
    *,
    points: np.ndarray,
    impact_nodes: np.ndarray,
    state: _ImpactState,
    moving_rigid_wall: bool,
    constrained_node_ids: list[int],
    impactor_mass: float,
    next_set_id: int,
    out_meta: dict[str, Any] | None,
    warnings: list[str],
) -> int:
    if not moving_rigid_wall:
        return next_set_id

    face_node_indices = np.asarray(impact_nodes, dtype=int) - 1
    face_node_indices = face_node_indices[
        (face_node_indices >= 0) & (face_node_indices < points.shape[0])
    ]
    face_positions = points[face_node_indices] if face_node_indices.size else points
    bbox_diagonal = float(
        np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0))
    )
    automatic_gap = max(bbox_diagonal * 0.003, 0.05)
    gap = state.wall_gap if state.wall_gap > 0.0 else automatic_gap
    face_centroid = np.mean(face_positions, axis=0)
    wall_projection = float(np.min(face_positions @ state.direction)) - gap
    wall_point = face_centroid + state.direction * (
        wall_projection - float(face_centroid @ state.direction)
    )
    wall_normal = state.direction
    wall_head = wall_point + wall_normal
    wall_mass = 0.0 if state.prescribed_wall else max(float(impactor_mass), 0.0) * 1e-3
    wall_slave_set_id = next_set_id
    next_set_id += 1
    _node_set(
        lines,
        wall_slave_set_id,
        np.arange(1, points.shape[0] + 1, dtype=int),
    )
    lines.extend(
        [
            "*RIGIDWALL_PLANAR_MOVING",
            "$#    nsid     nsedx   dsearch",
            f"{wall_slave_set_id}, 0, 0.0",
            "$#      xt        yt        zt        xh        yh        zh      fric",
            _keyword_card(
                wall_point[0], wall_point[1], wall_point[2],
                wall_head[0], wall_head[1], wall_head[2],
                state.wall_friction,
            ),
            "$#    mass        v0",
            f"{wall_mass:.12g}, {state.speed:.12g}",
        ]
    )
    if state.prescribed_wall and float(impactor_mass) > 0.0:
        warnings.append(
            "Prescribed moving wall scenario ignores impactor_mass_kg. "
            "OpenRadioss uses Mass=0 so V0 is an imposed velocity."
        )
    if not constrained_node_ids and not state.prescribed_wall:
        warnings.append(
            "Fixed specimen + moving impactor has no active constraints. "
            "The specimen can translate after contact; add a Constraint node "
            "for a fixed-rear crush test or use Moving body + fixed wall."
        )
    if wall_mass <= 0.0 and not state.prescribed_wall:
        warnings.append(
            "Fixed specimen + moving impactor uses a moving rigid wall with "
            "Mass=0. OpenRadioss treats V0 as imposed velocity, not an "
            "initial velocity of a finite-mass impactor. Set "
            "impactor_mass_kg on the Crash Solver for an inertial sled impact."
        )
    logger.info(
        "OpenRadioss deck: moving planar wall point=%s normal=%s "
        "velocity=%.3g mm/ms mass=%.3g tonne gap=%.3f mm friction=%.3g scenario=%s",
        wall_point.tolist(),
        wall_normal.tolist(),
        state.speed,
        wall_mass,
        gap,
        state.wall_friction,
        state.scenario_label,
    )
    if out_meta is not None:
        out_meta["wall"] = {
            "type": "prescribed" if state.prescribed_wall else "moving",
            "pt": [float(value) for value in wall_point],
            "normal": [float(value) for value in wall_normal],
            "half_extent": float(0.6 * bbox_diagonal),
            "v0_mm_per_ms": state.speed,
            "velocity_dir": [float(value) for value in state.direction],
        }
    return next_set_id


def _append_gravity(
    lines: list[str],
    *,
    gravity: dict[str, Any] | None,
    end_time: float,
    next_set_id: int,
) -> int:
    acceleration = (
        nonnegative_float(
            gravity.get("accel", 0.0),
            label="Gravity acceleration",
        )
        if gravity
        else 0.0
    )
    if not gravity or acceleration == 0.0:
        return next_set_id

    direction = str(gravity.get("direction", "-Y") or "-Y").strip().upper()
    direction_map = {
        "-X": ("X", +acceleration),
        "+X": ("X", -acceleration),
        "-Y": ("Y", +acceleration),
        "+Y": ("Y", -acceleration),
        "-Z": ("Z", +acceleration),
        "+Z": ("Z", -acceleration),
    }
    if direction not in direction_map:
        raise SolverBackendError(f"Unsupported gravity direction: {direction!r}.")
    axis, signed_acceleration = direction_map[direction]
    curve_id = next_set_id
    lines.extend(
        [
            "*DEFINE_CURVE",
            f"{curve_id}, 0, 1.0, 1.0",
            "0.0, 1.0",
            f"{end_time * 10.0:.6g}, 1.0",
            f"*LOAD_BODY_{axis}",
            f"{curve_id}, {signed_acceleration:.12g}",
        ]
    )
    return next_set_id + 1


def _append_impactor_mass(
    lines: list[str],
    *,
    points: np.ndarray,
    state: _ImpactState,
    impactor_mass: float,
) -> None:
    if not state.moving_body or float(impactor_mass) <= 0.0:
        return

    added_mass_tonnes = float(impactor_mass) * 1e-3
    if state.speed > 0.0:
        projections = points @ state.direction
        minimum_projection = float(np.min(projections))
        mass_nodes = np.where(projections < minimum_projection + 5.0)[0] + 1
        mass_label = "trailing node(s)"
    else:
        mass_nodes = np.arange(1, points.shape[0] + 1, dtype=int)
        mass_label = "node(s)"
    if len(mass_nodes) == 0:
        return

    mass_per_node = added_mass_tonnes / len(mass_nodes)
    lines.extend(["*ELEMENT_MASS", "$#   eid     nid    mass"])
    start_element_id = points.shape[0] * 10 + 1000000
    for offset, node_id in enumerate(mass_nodes):
        lines.append(f"{start_element_id + offset}, {node_id}, {mass_per_node:.12g}")
    logger.info(
        "OpenRadioss deck: distributed %.1f kg sled mass over %d %s",
        float(impactor_mass),
        len(mass_nodes),
        mass_label,
    )


def append_crash_event_cards(
    lines: list[str],
    *,
    mesh: Any,
    points: np.ndarray,
    constraints: list[dict[str, Any]],
    impact: dict[str, Any],
    end_time: float,
    gravity: dict[str, Any] | None,
    warnings: list[str],
    impactor_mass: float,
    out_meta: dict[str, Any] | None,
) -> None:
    """Append boundary, contact, wall, gravity, and sled-mass cards."""
    state = _impact_state(impact)
    logger.info(
        "OpenRadioss deck: velocity=%r mm/ms, magnitude=%.3f m/s, scenario=%s",
        state.velocity.tolist(),
        state.speed,
        state.scenario_label,
    )
    next_set_id, constrained_node_ids = _append_constraints(
        lines,
        mesh=mesh,
        constraints=constraints,
        moving_body=state.moving_body,
        next_set_id=100,
        warnings=warnings,
    )
    impact_faces, impact_nodes = _impact_nodes(mesh, points, impact, state)
    moving_rigid_wall = (
        not state.moving_body
        and bool(impact_faces)
        and len(impact_nodes) > 0
        and state.speed > 0.0
    )
    _append_velocity(
        lines,
        points=points,
        impact_nodes=impact_nodes,
        constrained_node_ids=constrained_node_ids,
        state=state,
        end_time=end_time,
        moving_rigid_wall=moving_rigid_wall,
        warnings=warnings,
    )
    _append_self_contact(lines)
    next_set_id = _append_stationary_wall(
        lines,
        mesh=mesh,
        points=points,
        impact_faces=impact_faces,
        impact_nodes=impact_nodes,
        state=state,
        next_set_id=next_set_id,
        out_meta=out_meta,
        warnings=warnings,
    )
    next_set_id = _append_moving_wall(
        lines,
        points=points,
        impact_nodes=impact_nodes,
        state=state,
        moving_rigid_wall=moving_rigid_wall,
        constrained_node_ids=constrained_node_ids,
        impactor_mass=impactor_mass,
        next_set_id=next_set_id,
        out_meta=out_meta,
        warnings=warnings,
    )
    _append_gravity(
        lines,
        gravity=gravity,
        end_time=end_time,
        next_set_id=next_set_id,
    )
    _append_impactor_mass(
        lines,
        points=points,
        state=state,
        impactor_mass=impactor_mass,
    )
    if out_meta is not None:
        reference_nodes = (
            constrained_node_ids if state.prescribed_wall else impact_nodes
        )
        out_meta["measurement"] = {
            "scenario": state.scenario,
            "scenario_label": state.scenario_label,
            "impact_axis": [float(value) for value in state.direction],
            "initial_speed_m_s": float(state.speed),
            "impactor_mass_kg": float(impactor_mass),
            "reference_name": (
                "constrained support" if state.prescribed_wall else "impact face"
            ),
            "reference_node_ids": [int(value) for value in reference_nodes],
            "support_node_ids": [int(value) for value in constrained_node_ids],
            "coordinate_system": "global Cartesian",
        }
