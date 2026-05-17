# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""OpenRadioss crash backend adapter.

OpenRadioss reads LS-DYNA-style ``*`` keyword input, so this adapter writes an
LS-DYNA keyword deck and then runs Starter + Engine.  Animation results are
imported via :mod:`pylcss.solver_backends.radioss_reader` (which delegates to
the ``anim_to_vtk`` converter that ships with OpenRadioss tools).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np

from pylcss.solver_backends.common import (
    ExternalRunConfig,
    SolverBackendError,
    dict_geometries,
    id_lines,
    make_work_dir,
    mesh_to_tet4,
    nodes_matching_condition,
    nodes_matching_geometries,
    resolve_executable,
    run_process,
    tail,
)
from pylcss.solver_backends.radioss_reader import read_animation_frames, resolve_anim_to_vtk
from typing import Optional


# PyLCSS crash material nodes expose stiffness and strength in MPa while the
# generated Radioss deck uses the consistent tonne-mm-ms unit system.
# In that system 1 MPa = 1 N/mm^2 = 1e-6 tonne/(mm*ms^2).
_MPA_TO_TONNE_MM_MS2 = 1.0e-6
_TONNE_MM_MS2_TO_MPA = 1.0 / _MPA_TO_TONNE_MM_MS2


def _radioss_runtime_env(binary_path: str) -> tuple:
    """Build PATH + env-var additions so Starter / Engine find their runtime DLLs.

    The official OpenRadioss Windows launcher prepends three sibling folders
    under ``OpenRadioss/extlib/`` to ``PATH`` (Intel OneAPI runtime, HyperMesh
    reader, and the H3D writer), plus sets ``RAD_CFG_PATH`` to ``hm_cfg_files``
    and ``RAD_H3D_PATH`` to the H3D writer dir.  Without those, the .exe fails
    to load on Windows (0xC0000135 STATUS_DLL_NOT_FOUND) — exactly the symptom
    we hit.

    Returns
    -------
    (extra_path_dirs: list[str], extra_env: dict)
    """
    bin_path = Path(binary_path).resolve()
    extra_dirs: List[str] = [str(bin_path.parent)]
    extra_env: dict = {}

    # Walk up looking for the OpenRadioss install root — distinguished by the
    # presence of an ``extlib/`` sibling to ``exec/``.
    root: Optional[Path] = None
    for parent in (bin_path.parent, *bin_path.parent.parents):
        if (parent / "extlib").is_dir():
            root = parent
            break
    if root is None:
        return extra_dirs, extra_env

    # Platform-specific subfolder used inside extlib/*/.
    plat = "win64" if bin_path.suffix.lower() == ".exe" else "linux64"
    candidates = [
        root / "extlib" / "intelOneAPI_runtime" / plat,
        root / "extlib" / "hm_reader" / plat,
        root / "extlib" / "h3d" / "lib" / plat,
    ]
    for c in candidates:
        if c.is_dir():
            extra_dirs.append(str(c))

    cfg = root / "hm_cfg_files"
    if cfg.is_dir():
        extra_env["RAD_CFG_PATH"] = str(cfg)
    h3d = root / "extlib" / "h3d" / "lib" / plat
    if h3d.is_dir():
        extra_env["RAD_H3D_PATH"] = str(h3d)

    return extra_dirs, extra_env


def _collect_radioss_failure_context(
    work_dir: Path,
    job_name: str,
    returncode: int,
    executable: str,
    stage: str,
) -> str:
    """Build a diagnostic block when Starter or Engine exits non-zero.

    Radioss writes its real error messages to ``<job>_0000.out`` (Starter) or
    ``<job>_0001.out`` (Engine) — stdout is usually empty on failure.  This
    helper reads those companion logs and decodes the exit code so the user
    actually sees what went wrong instead of an empty traceback tail.
    """
    parts: List[str] = []
    parts.append(f"Stage: {stage}")
    parts.append(f"Exit code: {returncode} (0x{(returncode & 0xFFFFFFFF):08X})")
    parts.append(f"Executable: {executable}")

    if returncode in (-1073741515, 3221225781):
        parts.append(
            "Windows STATUS_DLL_NOT_FOUND (0xC0000135). A runtime DLL is missing "
            "next to the binary — verify the OpenRadioss install is intact, or "
            "rerun  scripts/install_solvers.py --only radioss --force."
        )

    candidate_logs = [
        f"{job_name}_0000.out", f"{job_name}_0001.out",
        f"{job_name}_0000.txt", f"{job_name}_0001.txt",
        f"{job_name}.out",      f"{job_name}.log",
    ]
    for name in candidate_logs:
        path = work_dir / name
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not content:
            continue
        parts.append(f"--- {path.name} (last 2500 chars) ---\n{tail(content, 2500)}")
    return "\n".join(parts)


def _node_set(lines: List[str], set_id: int, nodes_1based: np.ndarray) -> None:
    lines.extend(["*SET_NODE_LIST", "$#     sid"])
    lines.append(f"{set_id}")
    lines.extend(id_lines(nodes_1based, per_line=8))


def _build_keyword_deck(
    mesh: Any,
    material: dict,
    constraints: List[dict],
    impact: dict,
    end_time: float,
    output_dt: float,
    gravity: dict | None,
    warnings: List[str],
    impactor_mass: float = 0.0,
    out_meta: dict | None = None,
) -> str:
    """Create a minimal LS-DYNA keyword deck for OpenRadioss Starter.

    If ``out_meta`` is provided, it is populated with wall geometry data
    (``wall``: dict or None) so the viewer can render the wall.  The keyword
    deck only contains the wall coordinates as text; without this side channel
    the viewer has no way to know where the wall was placed.
    """
    points, tets = mesh_to_tet4(mesh, warnings)

    e_mpa = float(material.get("E", 210000.0))
    nu = float(material.get("nu", material.get("poissons_ratio", 0.3)))
    rho = float(material.get("rho", material.get("density", 7.85e-9)))
    yield_strength_mpa = float(material.get("yield_strength", 250.0))
    tangent_modulus_mpa = float(material.get("tangent_modulus", 2000.0))
    failure_strain = float(material.get("failure_strain", 0.0))
    if not material.get("enable_fracture", True):
        failure_strain = 0.0

    # Cowper-Symonds strain-rate parameters.  Per the Altair *MAT_003 /
    # *MAT_PLASTIC_KINEMATIC reference, SRC is documented with the unit
    # annotation [1/s] — the LS-DYNA reader does NOT auto-scale this
    # field to the deck's consistent time unit.  So the value is written
    # to the deck as-is; supply C in 1/s regardless of whether the rest
    # of the deck is in mm-ms-tonne.  Default 0/0 disables rate hardening.
    # Mild steel: strain_rate_c ≈ 40 s⁻¹, strain_rate_p ≈ 5.
    # Aluminum 6061: strain_rate_c ≈ 6500, strain_rate_p ≈ 4.
    # Rate-hardened yield is σ_y(ε̇) = σ_y0 · (1 + (ε̇/C)^(1/p)); at
    # crash rates (100–1000 s⁻¹) this raises flow stress ~30–80 %.
    src = float(material.get("strain_rate_c", 0.0) or 0.0)
    srp = float(material.get("strain_rate_p", 0.0) or 0.0)

    e = e_mpa * _MPA_TO_TONNE_MM_MS2
    yield_strength = yield_strength_mpa * _MPA_TO_TONNE_MM_MS2
    tangent_modulus = tangent_modulus_mpa * _MPA_TO_TONNE_MM_MS2

    lines: List[str] = [
        "*KEYWORD",
        "*TITLE",
        "PyLCSS OpenRadioss crash deck",
        "*CONTROL_TERMINATION",
        f"{float(end_time):.12g}",
        "*DATABASE_BINARY_D3PLOT",
        f"{float(max(output_dt, 1e-12)):.12g}",
        "*NODE",
    ]

    for idx, xyz in enumerate(points, start=1):
        lines.append(f"{idx}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}")

    lines.extend(["*ELEMENT_SOLID", "$#   eid     pid      n1      n2      n3      n4"])
    for idx, conn in enumerate(tets, start=1):
        node_ids = [int(v) + 1 for v in conn]
        lines.append(
            f"{idx}, 1, {node_ids[0]}, {node_ids[1]}, {node_ids[2]}, {node_ids[3]}"
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
            "*MAT_PLASTIC_KINEMATIC",
            # Card 1: MID, RO, E, PR, SIGY, ETAN, BETA
            "$#     mid        ro         e        pr      sigy      etan      beta",
            f"1, {rho:.12g}, {e:.12g}, {nu:.12g}, {yield_strength:.12g}, {tangent_modulus:.12g}, 0.0",
            # Card 2: SRC, SRP, FS, VP  (Cowper-Symonds strain-rate params,
            # failure strain, viscoplastic flag).  This second card is
            # MANDATORY in LS-DYNA / OpenRadioss; omitting it makes the parser
            # eat the next keyword as the missing data line.
            "$#     src       srp        fs        vp",
            f"{src:.6g}, {srp:.6g}, {failure_strain:.6g}, 0.0",
        ]
    )

    next_set_id = 100

    velocity = np.asarray(impact.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    v_mag = float(np.linalg.norm(velocity))
    v_hat = velocity / v_mag if v_mag > 0.0 else np.array([1.0, 0.0, 0.0])
    scope = str(impact.get("application_scope") or "Impact Face").strip().lower()
    moving_body = scope.replace("_", " ").startswith("moving")

    # ── SPC constraints (Impact Face scope only) ────────────────────────────
    # For the Moving Body scope the entire structure is a free-flying projectile:
    # there must be NO kinematic constraints — the rigid wall provides the only
    # reaction force.  Adding SPCs here would turn the free-flight crash into a
    # "hammer blow on a fixed-end bar", which gives elastic oscillations at
    # ~0.7 mm amplitude instead of the expected 30–70 mm of progressive crush.
    constrained_node_ids: List[int] = []
    if not moving_body:
        for idx, constraint in enumerate(constraints, start=1):
            geoms = dict_geometries(constraint)
            condition = str(constraint.get("condition") or "").strip()
            if geoms:
                node_ids = nodes_matching_geometries(mesh, geoms) + 1
            elif condition:
                node_ids = nodes_matching_condition(
                    mesh,
                    condition,
                    warnings=warnings,
                    label=f"Crash constraint {idx}",
                ) + 1
            else:
                warnings.append(f"Crash constraint {idx} has no face geometry or condition; skipped.")
                continue
            if len(node_ids) == 0:
                warnings.append(f"Crash constraint {idx} did not match any mesh nodes.")
                continue
            constrained_node_ids.extend(int(v) for v in node_ids)
            set_id = next_set_id
            next_set_id += 1
            _node_set(lines, set_id, node_ids)

            fixed = set(int(v) for v in constraint.get("fixed_dofs", [0, 1, 2]))
            dofx = 1 if 0 in fixed else 0
            dofy = 1 if 1 in fixed else 0
            dofz = 1 if 2 in fixed else 0
            lines.extend(
                [
                    "*BOUNDARY_SPC_SET",
                    "$#     nsid       cid      dofx      dofy      dofz     dofrx     dofry     dofrz",
                    f"{set_id}, 0, {dofx}, {dofy}, {dofz}, 0, 0, 0",
                ]
            )
    elif constraints:
        warnings.append(
            "Moving Body crash: SPC constraints are ignored in this scope — the "
            "entire body is a free-flying projectile and the rigid wall provides "
            "the only reaction.  To model a fixed-rear laboratory test, switch "
            "the ImpactCondition to 'Impact Face' scope and apply the velocity "
            "only to the front face nodes."
        )

    # ── Impact node set / moving-wall contact ────────────────────────────────
    print(f"OpenRadioss deck: impact velocity (mm/ms) = {velocity.tolist()!r}, "
          f"|v|={v_mag:.3f} mm/ms "
          f"(= {v_mag:.1f} m/s)")
    impact_faces = impact.get("face_list", [])
    if moving_body:
        # ALL nodes get the initial velocity — the body is a free-flying mass.
        impact_nodes = np.arange(1, points.shape[0] + 1, dtype=int)
    elif impact_faces:
        impact_nodes = nodes_matching_geometries(
            mesh,
            impact_faces,
            tolerance=float(impact.get("node_tolerance", 2.0)),
        ) + 1
    else:
        impact_nodes = np.arange(1, points.shape[0] + 1, dtype=int)

    moving_rigid_wall = (
        (not moving_body)
        and bool(impact_faces)
        and len(impact_nodes) > 0
        and v_mag > 0.0
    )

    if len(impact_nodes) == 0:
        warnings.append("Impact condition did not match any nodes; no initial velocity exported.")
    elif v_mag > 0.0:
        if not moving_body:
            travel = v_mag * float(end_time)
            if travel > 0.0:
                if constrained_node_ids:
                    impact_proj = points[np.asarray(impact_nodes, dtype=int) - 1] @ v_hat
                    constrained_idx = np.asarray(constrained_node_ids, dtype=int) - 1
                    constrained_idx = constrained_idx[
                        (constrained_idx >= 0) & (constrained_idx < points.shape[0])
                    ]
                    if constrained_idx.size:
                        support_proj = points[constrained_idx] @ v_hat
                        downstream = (
                            support_proj[:, None] - impact_proj[None, :]
                        ).reshape(-1)
                        downstream = downstream[downstream > 0.0]
                        if downstream.size:
                            available = float(np.min(downstream))
                            if travel > 0.85 * available:
                                warnings.append(
                                    "Impact Face crash: |velocity| * end_time is "
                                    f"{travel:.1f} mm, but the nearest constrained "
                                    f"support is only {available:.1f} mm along the "
                                    "impact direction.  The moving wall can overrun "
                                    "the supported end in the animation.  "
                                    "Reduce end_time, velocity, or sled mass, or use "
                                    "Moving Body scope for a free-body wall impact."
                                )
                else:
                    bbox_span = float(np.ptp(points @ v_hat))
                    if bbox_span > 0.0 and travel > 0.85 * bbox_span:
                        warnings.append(
                            "Impact Face crash has no active constraints and the "
                            f"requested stroke ({travel:.1f} mm) is close to or "
                            "larger than the part length in the impact direction.  "
                            "For a wall/barrier event, use Moving Body scope."
                        )
        if moving_rigid_wall:
            print(
                "OpenRadioss deck: Impact Face scope uses a moving rigid wall; "
                "the deformable mesh starts at rest."
            )
        else:
            print(
                "OpenRadioss deck: applying initial velocity to "
                f"{len(impact_nodes)} node(s) with scope="
                f"{'Moving Body' if moving_body else 'Impact Face'}"
            )
            # OpenRadioss's LS-DYNA reader does not implement
            # ``*INITIAL_VELOCITY_NODE_SET`` — only the per-node
            # ``*INITIAL_VELOCITY_NODE`` form.  Emit one row per impact node.
            lines.append("*INITIAL_VELOCITY_NODE")
            lines.append("$#     nid        vx        vy        vz       vxr       vyr       vzr")
            for node_id in impact_nodes:
                lines.append(
                    f"{int(node_id)}, "
                    f"{velocity[0]:.12g}, {velocity[1]:.12g}, {velocity[2]:.12g}, "
                    "0, 0, 0"
                )

    # Standard tube-crush decks use automatic single-surface contact so folds
    # do not pass through each other as the structure collapses.  Keep it on
    # for all crash decks; SSID=0 means "all external segments".  Card 3
    # (penalty scale factors) is emitted with defaults so the deck is
    # portable to native LS-DYNA preprocessors which require it to be
    # physically present, even if blank.
    lines.extend([
        "*CONTACT_AUTOMATIC_SINGLE_SURFACE",
        "$#    ssid      msid     sstyp     mstyp    sboxid    mboxid       spr       mpr",
        "0",
        "$#      fs        fd        dc        vc       vdc    penchk        bt        dt",
        "0.08, 0.08",
        "$#     sfs       sfm       sst       mst      sfst      sfmt       fsf       vsf",
        "1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0",
    ])

    # ── Rigid wall (Moving Body scope only) ──────────────────────────────────
    # The wall is placed just in front of the leading face of the body in the
    # velocity direction.  The wall normal points back toward the body so that
    # nodes moving in the impact direction are pushed back when they make contact.
    if moving_body and len(impact_nodes) > 0 and v_mag > 0.0:
        all_pos = points                                  # (N_nodes, 3)
        projs = all_pos @ v_hat                           # scalar projection per node
        max_proj = float(np.max(projs))                   # leading-edge projection
        bbox_diag = float(np.linalg.norm(
            np.max(all_pos, axis=0) - np.min(all_pos, axis=0)
        ))
        gap = max(bbox_diag * 0.005, 0.1)                 # 0.5 % of extent, min 0.1 mm
        # Wall anchor point: on the plane perpendicular to v_hat passing through
        # the leading edge + gap.  Centroid keeps it in the middle of the body.
        centroid = np.mean(all_pos, axis=0)
        wall_pt = centroid + v_hat * (max_proj + gap - float(centroid @ v_hat))
        # Wall normal opposes the impact direction so the body bounces off.
        wall_normal = -v_hat
        # LS-DYNA *RIGIDWALL_PLANAR: Card 2 is XT,YT,ZT (tail = point on wall)
        # followed by XH,YH,ZH (head = second point; tail→head = outward normal).
        wall_head = wall_pt + wall_normal
        lines.extend([
            "*RIGIDWALL_PLANAR",
            "$#    nsid    nsidex     boxid    offset     birth     death     rwksf",
            "0, 0, 0, 0.0, 0.0",
            "$#      xt        yt        zt        xh        yh        zh      fric",
            (f"{wall_pt[0]:.12g}, {wall_pt[1]:.12g}, {wall_pt[2]:.12g}, "
             f"{wall_head[0]:.12g}, {wall_head[1]:.12g}, {wall_head[2]:.12g}, 0.0"),
        ])
        print(
            f"OpenRadioss deck: added RIGIDWALL_PLANAR at "
            f"[{wall_pt[0]:.3g}, {wall_pt[1]:.3g}, {wall_pt[2]:.3g}] "
            f"normal=[{wall_normal[0]:.3g}, {wall_normal[1]:.3g}, {wall_normal[2]:.3g}] "
            f"(Moving Body barrier, gap={gap:.3f} mm)"
        )
        if out_meta is not None:
            out_meta["wall"] = {
                "type": "stationary",
                "pt": [float(wall_pt[0]), float(wall_pt[1]), float(wall_pt[2])],
                "normal": [float(wall_normal[0]), float(wall_normal[1]), float(wall_normal[2])],
                "half_extent": float(0.6 * bbox_diag),
                "v0_mm_per_ms": 0.0,
                "velocity_dir": [0.0, 0.0, 0.0],
            }
        # Warn if the named impact face is at the trailing edge — common sign
        # that the velocity direction is wrong (e.g. -20 instead of +20 mm/ms
        # for a frontal-crash geometry where +X is the impact face).
        if impact_faces:
            face_nodes_idx = nodes_matching_geometries(mesh, impact_faces)
            if len(face_nodes_idx) > 0:
                face_max_proj = float(np.max(points[face_nodes_idx] @ v_hat))
                if max_proj - face_max_proj > bbox_diag * 0.2:
                    warnings.append(
                        "Moving Body crash: the named impact face is at the TRAILING "
                        "edge in the velocity direction — the rear of the body will hit "
                        "the wall first, not the intended impact face.  For a frontal "
                        "crash where the +X face hits the barrier first, set "
                        "velocity_x to a POSITIVE value (e.g. +20 mm/ms)."
                    )

    # ── Moving rigid wall (fixed-rear Impact Face scope) ─────────────────────
    # This is the industry-standard tube/crashbox crush setup: fixed support via
    # SPCs, structure initially at rest, and a massive rigid wall moving into
    # the selected impact face.
    if moving_rigid_wall:
        face_nodes_idx = np.asarray(impact_nodes, dtype=int) - 1
        face_nodes_idx = face_nodes_idx[
            (face_nodes_idx >= 0) & (face_nodes_idx < points.shape[0])
        ]
        face_pos = points[face_nodes_idx] if face_nodes_idx.size else points
        bbox_diag = float(np.linalg.norm(
            np.max(points, axis=0) - np.min(points, axis=0)
        ))
        gap = max(bbox_diag * 0.003, 0.05)
        face_centroid = np.mean(face_pos, axis=0)
        face_proj = face_pos @ v_hat
        # Put the wall just outside the selected face on the side opposite the
        # velocity vector; then move it along v_hat into the part.
        wall_proj = float(np.min(face_proj)) - gap
        wall_pt = face_centroid + v_hat * (wall_proj - float(face_centroid @ v_hat))
        wall_normal = v_hat
        wall_head = wall_pt + wall_normal
        wall_mass = max(float(impactor_mass), 0.0) * 1e-3  # kg -> tonne
        lines.extend([
            "*RIGIDWALL_PLANAR_MOVING",
            "$#    nsid    nsidex     boxid",
            "0, 0, 0",
            "$#      xt        yt        zt        xh        yh        zh      fric",
            (f"{wall_pt[0]:.12g}, {wall_pt[1]:.12g}, {wall_pt[2]:.12g}, "
             f"{wall_head[0]:.12g}, {wall_head[1]:.12g}, {wall_head[2]:.12g}, 0.08"),
            "$#    mass        v0",
            f"{wall_mass:.12g}, {v_mag:.12g}",
        ])
        if wall_mass <= 0.0:
            warnings.append(
                "Impact Face crash uses a moving rigid wall with Mass=0, so "
                "V0 is an imposed velocity.  Set impactor_mass_kg on the Crash "
                "Solver for an inertial sled impact."
            )
        print(
            f"OpenRadioss deck: added moving RIGIDWALL_PLANAR at "
            f"[{wall_pt[0]:.3g}, {wall_pt[1]:.3g}, {wall_pt[2]:.3g}] "
            f"normal=[{wall_normal[0]:.3g}, {wall_normal[1]:.3g}, {wall_normal[2]:.3g}], "
            f"v0={v_mag:.3g} mm/ms, mass={wall_mass:.3g} tonne"
        )
        if out_meta is not None:
            out_meta["wall"] = {
                "type": "moving",
                "pt": [float(wall_pt[0]), float(wall_pt[1]), float(wall_pt[2])],
                "normal": [float(wall_normal[0]), float(wall_normal[1]), float(wall_normal[2])],
                "half_extent": float(0.6 * bbox_diag),
                "v0_mm_per_ms": float(v_mag),
                "velocity_dir": [float(v_hat[0]), float(v_hat[1]), float(v_hat[2])],
            }

    # Gravity body force via *LOAD_BODY_X/Y/Z (LS-DYNA syntax accepted by Radioss).
    # *LOAD_BODY_* is a BASE ACCELERATION of the reference frame — a body in that
    # frame experiences a fictitious inertial force in the OPPOSITE direction
    # (D'Alembert).  So to make objects fall in -Y, the supplied SF must be +9810,
    # not -9810.  (See dynasupport.com/howtos/general/gravity-load: "A positive
    # gravity constant is used to make objects drop in the negative direction.")
    if gravity and float(gravity.get("accel", 0.0)) != 0.0:
        direction = gravity.get("direction", "-Y")
        accel = float(gravity.get("accel", 9810.0))
        dir_map = {
            "-X": ("X", +accel),
            "+X": ("X", -accel),
            "-Y": ("Y", +accel),
            "+Y": ("Y", -accel),
            "-Z": ("Z", +accel),
            "+Z": ("Z", -accel),
        }
        axis, signed = dir_map.get(direction, ("Y", +accel))
        # *DEFINE_CURVE: a unit constant curve so LOAD_BODY can scale by it.
        curve_id = next_set_id
        next_set_id += 1
        lines.extend(
            [
                "*DEFINE_CURVE",
                f"{curve_id}, 0, 1.0, 1.0",
                "0.0, 1.0",
                f"{float(end_time) * 10.0:.6g}, 1.0",
                f"*LOAD_BODY_{axis}",
                f"{curve_id}, {signed:.12g}",
            ]
        )

    # ── Impactor Mass (Sled, Moving Body scope only) ─────────────────────────
    if moving_body and float(impactor_mass) > 0.0:
        added_mass_tonnes = float(impactor_mass) * 1e-3
        mass_nodes = np.array([], dtype=int)
        mass_label = "node(s)"
        if v_mag > 0.0:
            # Moving-body wall impact: add sled inertia to the trailing edge,
            # opposite to the direction of travel.
            projs = points @ v_hat
            min_proj = float(np.min(projs))
            # Find nodes within 5 mm of the trailing edge
            mass_nodes = np.where(projs < min_proj + 5.0)[0] + 1
            mass_label = "trailing node(s)"
        else:
            # Fallback if no velocity: distribute over all nodes
            mass_nodes = np.arange(1, points.shape[0] + 1, dtype=int)
            mass_label = "node(s)"
            
        if len(mass_nodes) > 0:
            mass_per_node = added_mass_tonnes / len(mass_nodes)
            lines.append("*ELEMENT_MASS")
            lines.append("$#   eid     nid    mass")
            # Start mass element IDs high to avoid clash with solid elements
            start_eid = points.shape[0] * 10 + 1000000
            for i, nid in enumerate(mass_nodes):
                lines.append(f"{start_eid + i}, {nid}, {mass_per_node:.12g}")
            print(
                f"OpenRadioss deck: Added {float(impactor_mass):.1f} kg sled mass "
                f"distributed over {len(mass_nodes)} {mass_label}."
            )

    # Time-history outputs (*DATABASE_GLSTAT / _MATSUM / _RWFORC) are NOT
    # emitted here.  OpenRadioss's LS-DYNA reader does not implement those
    # keywords — they are silently dropped during translation to native
    # Radioss, and the engine produces no glstat/matsum/rwforc files.
    # Energy-balance and rigid-wall force histories therefore have to come
    # from the T01 file produced via the engine /TFILE cadence card (see
    # _build_engine_deck) plus the auto-included rigid-wall TH group that
    # *RIGIDWALL_PLANAR creates internally.  Native LS-DYNA users running
    # this deck through LS-PrePost will get full *DATABASE_* support, but
    # for OpenRadioss the T01 file is the only TH output available.
    lines.append("*END")
    return "\n".join(lines) + "\n"


def _build_engine_deck(
    job_name: str,
    end_time: float,
    output_dt: float,
    mass_scaling_dt: float = 0.0,
    mass_scaling_scale: float = 0.9,
) -> str:
    """Write the OpenRadioss ``<job>_0001.rad`` engine deck.

    Beyond the bare ``/RUN`` + termination time, three blocks materially affect
    how the run reports progress and what comes back as animation:

    1.  ``/ANIM/DT`` + ``/ANIM/ELEM/*`` - without these the Engine writes no
        animation files at the user-requested frequency, so the viewer ends up
        with one or two frames and the "expected remaining time" line in the
        log is the only signal of progress.

    2.  ``/DT/NODA/CST`` - explicit dynamics' default time step shrinks as
        contact engages or stiff regions activate, which makes the Engine's
        per-cycle ``REMAINING TIME ESTIMATE`` grow upward over the run.  When
        ``mass_scaling_dt > 0`` we ask Radioss to hold the time step at that
        value by adding mass to nodes whose CFL bound drops below it.  This is
        the standard industry trick for keeping crash sims wall-clock-bounded.

    Parameters
    ----------
    mass_scaling_dt
        Target nodal time step.  ``0`` disables ``/DT/NODA/CST`` entirely so
        the run uses pure CFL - physics is unchanged, but dt may drift and
        the estimated-remaining-time line will grow during the run.
    mass_scaling_scale
        Safety factor on the critical time step.  ``0.9`` is the
        Altair-documented default.
    """
    out_dt = float(max(output_dt, 1e-12))
    lines: List[str] = [
        "#RADIOSS ENGINE INPUT",
        f"/RUN/{job_name}/1",
        f"{float(end_time):.6g}",
    ]

    # Optional nodal-mass time-step control (keeps dt approximately constant).
    if float(mass_scaling_dt) > 0.0:
        lines.extend(
            [
                "/DT/NODA/CST/0",
                f"{float(mass_scaling_scale):.6g}  {float(mass_scaling_dt):.6g}",
            ]
        )

    # Animation output: explicit frequency + the fields the viewer renders.
    lines.extend(
        [
            "/ANIM/DT",
            f"0.  {out_dt:.6g}",
            "/ANIM/VECT/DISP",
            "/ANIM/VECT/VEL",
            "/ANIM/VECT/ACC",
            "/ANIM/ELEM/VONM",   # Von Mises stress field
            "/ANIM/ELEM/EPSP",   # equivalent plastic strain
            "/ANIM/ELEM/ENER",
            "/ANIM/BRICK/TENS/STRESS",
        ]
    )

    # Time-history file cadence.  The correct OpenRadioss engine keyword
    # for T01 sampling rate is /TFILE — *not* /TH/DT (which is a Starter
    # keyword for TH-group definitions, not an engine cadence card).
    # Sampling at output_dt matches the animation cadence so any TH
    # curves line up with the visualization timeline.
    lines.extend(
        [
            "/TFILE",
            f"{out_dt:.6g}",
            "",
        ]
    )
    return "\n".join(lines)


def run_openradioss_existing_deck(
    deck_path: str | Path,
    config: ExternalRunConfig,
    engine_deck_path: Optional[str | Path] = None,
    end_time: Optional[float] = None,
    visualization_mode: str = "Von Mises Stress",
    disp_scale: float = 1.0,
) -> dict:
    """Run an OpenRadioss / LS-DYNA-style deck the user already has on disk.

    This is the "run the real benchmark" path: no PyLCSS preprocessing, no
    parametric geometry — point at an existing ``.k`` / ``.rad`` (such as the
    Chrysler Neon HPC benchmark) and the function will:

    1. Stage the deck files into a fresh work directory.
    2. Run **Starter** with ``-i <deck>`` to produce the model + engine ``.rad``.
    3. Run **Engine** on the resulting ``<job>_0001.rad``.
    4. Convert the ``A###`` animation files via ``anim_to_vtk`` and surface
       them as the same ``frames`` list the crash viewer already animates.

    Notes
    -----
    * If the user supplies a Radioss engine file directly (``.rad``), Starter
      is skipped — Engine is run on that file as-is.
    * Materials, contacts, and section properties live inside the user's deck
      verbatim — we never rewrite them.
    """
    deck_path = Path(deck_path).resolve()
    if not deck_path.is_file():
        raise SolverBackendError(f"Deck file not found: {deck_path}")

    warnings: List[str] = []
    work_dir = make_work_dir("pylcss_radioss_deck_", config.work_dir)
    # Stage the deck (so we don't pollute the user's source directory).
    import shutil

    def _stage(src: Path) -> Path:
        """Copy ``src`` into ``work_dir`` unless it's already there."""
        dest = work_dir / src.name
        try:
            same = dest.resolve() == src.resolve()
        except OSError:
            same = False
        if not same:
            shutil.copy2(src, dest)
        return dest

    staged_deck = _stage(deck_path)

    # If the user also pointed at a separate Radioss engine file, stage it.
    staged_engine: Optional[Path] = None
    if engine_deck_path:
        engine_src = Path(engine_deck_path).resolve()
        if engine_src.is_file():
            staged_engine = _stage(engine_src)

    starter = resolve_executable(
        config.executable,
        env_vars=("PYLCSS_OPENRADIOSS_STARTER", "OPENRADIOSS_STARTER"),
        candidates=(
            "starter_win64.exe", "starter_win64",
            "starter_linux64_gf", "starter_linux64_gf_sp", "starter_linux64_gf_dp",
        ),
    )
    engine = resolve_executable(
        config.secondary_executable,
        env_vars=("PYLCSS_OPENRADIOSS_ENGINE", "OPENRADIOSS_ENGINE"),
        candidates=(
            "engine_win64.exe", "engine_win64",
            "engine_linux64_gf", "engine_linux64_gf_sp", "engine_linux64_gf_dp",
        ),
    )

    solver_log = ""
    status = "deck_staged"

    # Starter must always run on the model deck — it produces the ``.rst``
    # restart file that Engine reads to initialise the model state.  Having a
    # pre-built ``_0001.rad`` (engine control file) does NOT mean Starter can
    # be skipped: those two files describe different things (model vs. run
    # parameters).  Only skip Starter when the user already has both the
    # ``_0001.rad`` AND a matching ``_0000.rst`` file already in the work
    # directory.
    rst_candidate = work_dir / (staged_deck.stem + ".rst")
    skip_starter = (
        staged_engine is not None
        and staged_deck.suffix.lower() == ".rad"
        and rst_candidate.is_file()
    )

    if not config.run_solver:
        return {
            "type": "external_solver",
            "backend": "OpenRadioss",
            "external_status": status,
            "mesh": None,
            "visualization_mode": visualization_mode,
            "disp_scale": disp_scale,
            "input_file": str(staged_deck),
            "engine_file": str(staged_engine) if staged_engine else None,
            "work_dir": str(work_dir),
            "solver_executable": starter,
            "secondary_solver_executable": engine,
            "solver_log": "",
            "warnings": warnings + [
                "deck_only=True - the deck was staged but neither Starter nor "
                "Engine was run.  Uncheck deck_only on the node to launch."
            ],
            "message": (
                f"Deck staged in {work_dir}. Toggle off `deck_only` to run "
                "Starter + Engine on this deck."
            ),
        }

    if not skip_starter:
        if starter is None:
            raise SolverBackendError(
                "OpenRadioss Starter executable not found.  Run "
                "scripts/install_solvers.py --only radioss or set "
                "PYLCSS_OPENRADIOSS_STARTER."
            )
        path_dirs, env_extra = _radioss_runtime_env(starter)
        import time as _time
        _t0 = _time.time()
        print(f"OpenRadioss Starter: launching on {staged_deck.name}...")
        # Force single SPMD domain — see comment in run_openradioss_crash.
        proc = run_process(
            [starter, "-i", str(staged_deck), "-nspmd", "1"],
            cwd=work_dir,
            timeout_s=config.timeout_s,
            extra_path_dirs=path_dirs,
            extra_env=env_extra,
            stdout_file=work_dir / "_pylcss_starter.log",
        )
        print(f"OpenRadioss Starter: completed in {_time.time() - _t0:.1f}s "
              f"(exit={proc.returncode}).")
        solver_log = tail(proc.stdout or "")
        if proc.returncode != 0:
            aux = _collect_radioss_failure_context(
                work_dir, staged_deck.stem, proc.returncode, starter, stage="Starter"
            )
            raise SolverBackendError(
                "OpenRadioss Starter failed on the user-supplied deck.  Last "
                "solver output:\n" + (solver_log or "(stdout was empty)\n") + "\n" + aux
            )
        status = "starter_completed"

    # Locate the engine .rad file Starter generated.
    if staged_engine is None:
        candidates = sorted(work_dir.glob("*_0001.rad"))
        if not candidates:
            warnings.append(
                "Starter completed but no `_0001.rad` engine file was produced; "
                "cannot continue to Engine run."
            )
            return _wrap_deck_result(
                status, work_dir, staged_deck, None, starter, engine, solver_log, warnings,
                visualization_mode, disp_scale,
            )
        staged_engine = candidates[0]

    if engine is None:
        warnings.append("Engine executable not found; skipping dynamic solve.")
        return _wrap_deck_result(
            status, work_dir, staged_deck, staged_engine, starter, engine,
            solver_log, warnings, visualization_mode, disp_scale,
        )

    path_dirs, env_extra = _radioss_runtime_env(engine)
    import time as _time
    _t0 = _time.time()
    import os as _os
    nthread = max(1, _os.cpu_count() or 1)
    print(f"OpenRadioss Engine: launching on {staged_engine.name} "
          f"(SMP -nthread {nthread})... "
          "(this is where most of the wall-clock time goes)")
    proc_eng = run_process(
        [engine, "-i", str(staged_engine), "-nthread", str(nthread)],
        cwd=work_dir,
        timeout_s=config.timeout_s,
        extra_path_dirs=path_dirs,
        extra_env=env_extra,
        stdout_file=work_dir / "_pylcss_engine.log",
    )
    print(f"OpenRadioss Engine: completed in {_time.time() - _t0:.1f}s "
          f"(exit={proc_eng.returncode}).")
    solver_log = tail((proc_eng.stdout or "") + "\n" + solver_log)
    if proc_eng.returncode != 0:
        aux = _collect_radioss_failure_context(
            work_dir, staged_engine.stem.replace("_0001", ""),
            proc_eng.returncode, engine, stage="Engine"
        )
        raise SolverBackendError(
            "OpenRadioss Engine failed on the user-supplied deck.  Last solver "
            "output:\n" + (solver_log or "(stdout was empty)\n") + "\n" + aux
        )
    status = "engine_completed"

    # Read animation frames.  Job name is the stem of the engine file minus the
    # trailing ``_0001`` suffix (Radioss naming convention).
    stem = staged_engine.stem
    if stem.endswith("_0001"):
        job_name = stem[: -len("_0001")]
    else:
        job_name = stem
    converter = resolve_anim_to_vtk()
    raw_frames, anim_warnings = read_animation_frames(
        work_dir, job_name, converter=converter, timeout_s=config.timeout_s
    )
    warnings.extend(anim_warnings)

    if not raw_frames:
        return _wrap_deck_result(
            status, work_dir, staged_deck, staged_engine, starter, engine,
            solver_log, warnings, visualization_mode, disp_scale,
        )

    # The frames already carry node-count-matched arrays from
    # ``read_animation_frames``; we just expose them in the viewer's contract.
    last = raw_frames[-1]
    flat_disp = np.asarray(last.get("displacement", []), dtype=float).reshape(-1)
    n_points = flat_disp.size // 3 if flat_disp.size else 0
    peak_disp = (
        float(np.max(np.linalg.norm(flat_disp.reshape(n_points, 3), axis=1)))
        if n_points
        else 0.0
    )
    stress_vm = np.asarray(last.get("stress_vm", []), dtype=float)
    peak_vm = float(stress_vm.max()) if stress_vm.size else 0.0

    return {
        "type": "crash",
        "backend": "OpenRadioss",
        "external_status": status,
        # mesh is None here — we never built one in PyLCSS; the viewer uses
        # the per-frame point data emitted by anim_to_vtk.
        "mesh": None,
        "displacement": flat_disp,
        "stress": stress_vm,
        "visualization_mode": visualization_mode,
        "disp_scale": disp_scale,
        "frames": raw_frames,
        "peak_displacement": peak_disp,
        "peak_stress": peak_vm,
        # User-supplied decks: we don't know the wall geometry (it lives
        # inside their deck text) and we don't have a mesh/density to compute
        # KE/IE from, so both are left out.  end_time defaults to 1.0 so the
        # viewer's time label still produces useful normalized values.
        "wall": None,
        "end_time": float(end_time) if end_time is not None else 1.0,
        "input_file": str(staged_deck),
        "engine_file": str(staged_engine),
        "work_dir": str(work_dir),
        "solver_executable": starter,
        "secondary_solver_executable": engine,
        "solver_log": solver_log,
        "warnings": warnings,
        "message": (
            f"OpenRadioss completed on user deck `{deck_path.name}`; "
            f"{len(raw_frames)} animation frames imported."
        ),
    }


def _wrap_deck_result(
    status, work_dir, deck, engine_deck, starter_exe, engine_exe,
    solver_log, warnings, visualization_mode, disp_scale,
) -> dict:
    return {
        "type": "external_solver",
        "backend": "OpenRadioss",
        "external_status": status,
        "mesh": None,
        "visualization_mode": visualization_mode,
        "disp_scale": disp_scale,
        "input_file": str(deck),
        "engine_file": str(engine_deck) if engine_deck else None,
        "work_dir": str(work_dir),
        "solver_executable": starter_exe,
        "secondary_solver_executable": engine_exe,
        "solver_log": solver_log,
        "warnings": warnings,
        "message": (
            "OpenRadioss external deck run finished without animation import."
        ),
    }


def _build_animation_frames_with_mesh(mesh: Any, frames: list) -> list:
    """Pad / truncate animation frame arrays to match the source mesh point count.

    Preserves the per-node velocity field and per-element internal-energy
    density when available, so the time-history builder downstream can
    compute KE(t) and IE(t).
    """
    n_points = int(np.asarray(mesh.p).shape[1])
    fixed: list = []
    for frame in frames:
        flat = np.asarray(frame.get("displacement", []), dtype=float).reshape(-1)
        vm = (
            np.asarray(frame.get("stress_vm", []), dtype=float).reshape(-1)
            * _TONNE_MM_MS2_TO_MPA
        )
        node_ids = np.asarray(frame.get("node_ids", []), dtype=int).reshape(-1)
        vel_raw = np.asarray(frame.get("velocity", []), dtype=float)
        if vel_raw.ndim != 2 or vel_raw.shape[1] != 3:
            vel_raw = np.zeros((0, 3), dtype=float)

        # Ensure length 3*N and N respectively. anim_to_vtk may reorder points,
        # but it emits NODE_ID, so scatter back to the source mesh ids first.
        target_disp_3 = np.zeros((n_points, 3), dtype=float)
        disp_3 = flat.reshape((-1, 3)) if flat.size % 3 == 0 else np.zeros((0, 3))
        if node_ids.size == disp_3.shape[0]:
            valid = (node_ids >= 1) & (node_ids <= n_points)
            target_disp_3[node_ids[valid] - 1] = disp_3[valid]
        elif disp_3.size:
            n_copy = min(disp_3.shape[0], n_points)
            target_disp_3[:n_copy] = disp_3[:n_copy]
        target_disp = np.zeros(3 * n_points, dtype=float)
        target_disp[0::3] = target_disp_3[:, 0]
        target_disp[1::3] = target_disp_3[:, 1]
        target_disp[2::3] = target_disp_3[:, 2]

        target_vm = np.zeros(n_points, dtype=float)
        if node_ids.size == vm.size:
            valid = (node_ids >= 1) & (node_ids <= n_points)
            target_vm[node_ids[valid] - 1] = vm[valid]
        else:
            target_vm[: min(vm.size, target_vm.size)] = vm[: target_vm.size]

        target_vel = np.zeros((n_points, 3), dtype=float)
        if node_ids.size == vel_raw.shape[0] and vel_raw.shape[0] > 0:
            valid = (node_ids >= 1) & (node_ids <= n_points)
            target_vel[node_ids[valid] - 1] = vel_raw[valid]
        elif vel_raw.shape[0] > 0:
            n_copy = min(vel_raw.shape[0], n_points)
            target_vel[:n_copy] = vel_raw[:n_copy]

        ener_cell = frame.get("ener_cell")
        if ener_cell is not None:
            ener_cell = np.asarray(ener_cell, dtype=float).reshape(-1)

        fixed.append(
            {
                "displacement": target_disp,
                "stress_vm": target_vm,
                "velocity": target_vel,
                "ener_cell": ener_cell,
                "eps_p": np.zeros(n_points, dtype=float),
                "failed": np.zeros(n_points, dtype=float),
                "time": float(frame.get("time", 0.0)),
            }
        )
    return fixed


def _compute_time_history(
    mesh: Any,
    material: dict,
    frames: list,
    end_time: float,
) -> dict:
    """KE(t) [kJ] and IE(t) [kJ] from per-frame velocity + cell ENER fields.

    Lumped node masses are derived from the tet mesh: each node receives
    1/4 of every adjacent element's mass (ρ_material × element_volume).
    """
    if not frames:
        return {"t_ms": [], "ke_kj": [], "ie_kj": []}

    try:
        points, tets = mesh_to_tet4(mesh, [])
    except Exception:
        return {"t_ms": [float(f.get("time", 0.0)) * float(end_time) for f in frames],
                "ke_kj": [0.0] * len(frames),
                "ie_kj": [0.0] * len(frames)}

    rho_consistent = float(material.get("rho", material.get("density", 7.85e-9)))
    n_points = points.shape[0]
    n_elem = tets.shape[0]

    # Element volumes (always positive thanks to mesh_to_tet4's orientation flip).
    v0 = points[tets[:, 0]]
    e1 = points[tets[:, 1]] - v0
    e2 = points[tets[:, 2]] - v0
    e3 = points[tets[:, 3]] - v0
    elem_vol = np.abs(np.einsum("ij,ij->i", np.cross(e1, e2), e3)) / 6.0
    total_vol = float(np.sum(elem_vol))

    # Lumped node mass: 1/4 of each adjacent element's mass.
    node_mass = np.zeros(n_points, dtype=float)
    for k in range(4):
        np.add.at(node_mass, tets[:, k], 0.25 * rho_consistent * elem_vol)

    t_ms: list = []
    ke_kj: list = []
    ie_kj: list = []
    for frame in frames:
        t_ms.append(float(frame.get("time", 0.0)) * float(end_time))

        vel = frame.get("velocity")
        if vel is None or not isinstance(vel, np.ndarray) or vel.shape != (n_points, 3):
            ke_kj.append(0.0)
        else:
            speed_sq = np.einsum("ij,ij->i", vel, vel)
            # 0.5 · m · v² → in tonne-mm-ms units this evaluates directly in kJ.
            ke_kj.append(float(0.5 * np.sum(node_mass * speed_sq)))

        ener = frame.get("ener_cell")
        if ener is None or ener.size == 0:
            ie_kj.append(0.0)
        else:
            arr = np.asarray(ener, dtype=float).reshape(-1)
            n = min(arr.size, n_elem)
            ie_kj.append(float(np.sum(arr[:n] * elem_vol[:n])))

    return {"t_ms": t_ms, "ke_kj": ke_kj, "ie_kj": ie_kj,
            "total_volume_mm3": total_vol,
            "total_mass_kg": float(rho_consistent * total_vol * 1e3)}


def run_openradioss_crash(
    mesh: Any,
    material: dict,
    constraints: List[dict],
    impact: dict,
    config: ExternalRunConfig,
    end_time: float,
    output_dt: float,
    visualization_mode: str = "Von Mises Stress",
    disp_scale: float = 1.0,
    gravity: dict | None = None,
    mass_scaling_dt: float = 0.0,
    mass_scaling_scale: float = 0.9,
    impactor_mass: float = 0.0,
) -> dict:
    """Write deck, run Starter + Engine, then import animation frames."""
    if impact is None:
        raise SolverBackendError("OpenRadioss backend requires an impact condition.")

    warnings: List[str] = []
    work_dir = make_work_dir("pylcss_openradioss_", config.work_dir)
    job_name = config.job_name or "pylcss_openradioss"
    deck_path = work_dir / f"{job_name}.k"
    deck_meta: dict = {}
    deck_path.write_text(
        _build_keyword_deck(
            mesh=mesh,
            material=material,
            constraints=constraints,
            impact=impact,
            end_time=end_time,
            output_dt=output_dt,
            gravity=gravity,
            warnings=warnings,
            impactor_mass=impactor_mass,
            out_meta=deck_meta,
        ),
        encoding="utf-8",
    )
    wall_info = deck_meta.get("wall")

    engine_deck_path = work_dir / f"{job_name}_0001.rad"

    def _write_engine_deck() -> None:
        engine_deck_path.write_text(
            _build_engine_deck(
                job_name,
                end_time,
                output_dt,
                mass_scaling_dt=mass_scaling_dt,
                mass_scaling_scale=mass_scaling_scale,
            ),
            encoding="utf-8",
        )

    # Starter rewrites <job>_0001.rad. Keep a useful deck-only artifact, then
    # rewrite the engine controls after Starter succeeds and before Engine runs.
    _write_engine_deck()

    starter = resolve_executable(
        config.executable,
        env_vars=("PYLCSS_OPENRADIOSS_STARTER", "OPENRADIOSS_STARTER"),
        candidates=(
            "starter_win64.exe",
            "starter_win64",
            "starter_linux64_gf",
            "starter_linux64_gf_sp",
            "starter_linux64_gf_dp",
        ),
    )
    engine = resolve_executable(
        config.secondary_executable,
        env_vars=("PYLCSS_OPENRADIOSS_ENGINE", "OPENRADIOSS_ENGINE"),
        candidates=(
            "engine_win64.exe",
            "engine_win64",
            "engine_linux64_gf",
            "engine_linux64_gf_sp",
            "engine_linux64_gf_dp",
        ),
    )

    status = "deck_written"
    solver_log = ""
    frames: list = []

    if config.run_solver:
        if starter is None:
            warnings.append(
                "OpenRadioss Starter executable not found. Set the node path, "
                "add starter_* to PATH, define PYLCSS_OPENRADIOSS_STARTER, or "
                "run scripts/install_solvers.py."
            )
        else:
            path_dirs, env_extra = _radioss_runtime_env(starter)
            import time as _time
            _t0 = _time.time()
            print(f"OpenRadioss Starter: launching on {Path(deck_path).name}...")
            # ``-nspmd 1`` forces a single domain decomposition so the bundled
            # non-hybrid ``engine_win64`` (which only handles 1 SPMD domain)
            # can read the restart file Starter writes.  Without this the
            # default 4-domain Starter output trips Engine with
            # "NON HYBRID EXECUTABLE ONLY SUPPORTS ONE SPMD DOMAIN".
            proc = run_process(
                [starter, "-i", str(deck_path), "-nspmd", "1"],
                cwd=work_dir,
                timeout_s=config.timeout_s,
                extra_path_dirs=path_dirs,
                extra_env=env_extra,
                # Spool stdout to disk — Radioss prints per-cycle progress
                # which overflows Windows' 4 KB pipe buffer and deadlocks
                # the subprocess on long runs.
                stdout_file=work_dir / "_pylcss_starter.log",
            )
            print(f"OpenRadioss Starter: completed in {_time.time() - _t0:.1f}s "
                  f"(exit={proc.returncode}).")
            solver_log = tail(proc.stdout or "")
            if proc.returncode != 0:
                aux = _collect_radioss_failure_context(
                    work_dir, job_name, proc.returncode, starter, stage="Starter"
                )
                raise SolverBackendError(
                    "OpenRadioss Starter failed. Last solver output:\n"
                    + (solver_log or "(stdout was empty)\n")
                    + "\n"
                    + aux
                )
            status = "starter_completed"
            if engine is None:
                warnings.append(
                    "Starter completed but no Engine executable was found; "
                    "skipping the dynamic solve."
                )
            else:
                _write_engine_deck()
                path_dirs, env_extra = _radioss_runtime_env(engine)
                import time as _time
                _t0 = _time.time()
                print(f"OpenRadioss Engine: launching on {Path(engine_deck_path).name}... "
                      "(this is where most of the wall-clock time goes)")
                import os as _os
                nthread = max(1, (_os.cpu_count() or 1) // 1)
                proc_eng = run_process(
                    [engine, "-i", str(engine_deck_path),
                     "-nthread", str(nthread)],
                    cwd=work_dir,
                    timeout_s=config.timeout_s,
                    extra_path_dirs=path_dirs,
                    extra_env=env_extra,
                    stdout_file=work_dir / "_pylcss_engine.log",
                )
                print(f"OpenRadioss Engine: completed in {_time.time() - _t0:.1f}s "
                      f"(exit={proc_eng.returncode}).")
                solver_log = tail((proc_eng.stdout or "") + "\n" + solver_log)
                if proc_eng.returncode != 0:
                    aux = _collect_radioss_failure_context(
                        work_dir, job_name, proc_eng.returncode, engine, stage="Engine"
                    )
                    raise SolverBackendError(
                        "OpenRadioss Engine failed. Last solver output:\n"
                        + (solver_log or "(stdout was empty)\n")
                        + "\n"
                        + aux
                    )
                status = "engine_completed"
                converter = resolve_anim_to_vtk()
                raw_frames, anim_warnings = read_animation_frames(
                    work_dir, job_name, converter=converter, timeout_s=config.timeout_s
                )
                warnings.extend(anim_warnings)
                frames = _build_animation_frames_with_mesh(mesh, raw_frames)
                if frames:
                    max_disp = 0.0
                    max_vm = 0.0
                    for frame in frames:
                        disp = np.asarray(frame.get("displacement", []), dtype=float)
                        if disp.size:
                            max_disp = max(max_disp, float(np.max(np.abs(disp))))
                        vm = np.asarray(frame.get("stress_vm", []), dtype=float)
                        if vm.size:
                            max_vm = max(max_vm, float(np.max(vm)))
                    print(
                        "OpenRadioss import: viewer fields ready, "
                        f"global max |u| = {max_disp:.4e} mm, "
                        f"global max |VM| = {max_vm:.4e} MPa"
                    )

    # Choose result type based on what we managed to import.
    if frames:
        n_points = int(np.asarray(mesh.p).shape[1])
        last = frames[-1]
        displacement_flat = last["displacement"]
        stress_field = last["stress_vm"]
        peak_disp = float(np.max(np.linalg.norm(displacement_flat.reshape(n_points, 3), axis=1)))
        peak_vm = float(np.max(stress_field)) if stress_field.size else 0.0
        time_history = _compute_time_history(mesh, material, frames, end_time)
        return {
            "type": "crash",
            "backend": "OpenRadioss",
            "external_status": status,
            "mesh": mesh,
            "displacement": displacement_flat,
            "stress": stress_field,
            "visualization_mode": visualization_mode,
            "disp_scale": disp_scale,
            "frames": frames,
            "peak_displacement": peak_disp,
            "peak_stress": peak_vm,
            "wall": wall_info,
            "end_time": float(end_time),
            "time_history": time_history,
            "input_file": str(deck_path),
            "engine_file": str(engine_deck_path),
            "work_dir": str(work_dir),
            "solver_executable": starter,
            "secondary_solver_executable": engine,
            "solver_log": solver_log,
            "warnings": warnings,
            "message": "OpenRadioss solve complete; animation frames imported.",
        }

    return {
        "type": "external_solver",
        "backend": "OpenRadioss",
        "external_status": status,
        "mesh": mesh,
        "visualization_mode": visualization_mode,
        "disp_scale": disp_scale,
        "wall": wall_info,
        "end_time": float(end_time),
        "input_file": str(deck_path),
        "engine_file": str(engine_deck_path),
        "work_dir": str(work_dir),
        "solver_executable": starter,
        "secondary_solver_executable": engine,
        "solver_log": solver_log,
        "warnings": warnings,
        "message": (
            "OpenRadioss-compatible keyword deck generated. Enable external "
            "execution and configure starter/engine to run the solve from PyLCSS."
            if status == "deck_written"
            else "OpenRadioss run finished but no animation frames could be imported."
        ),
    }
