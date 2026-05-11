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
    nodes_matching_geometries,
    resolve_executable,
    run_process,
    tail,
)
from pylcss.solver_backends.radioss_reader import read_animation_frames, resolve_anim_to_vtk
from typing import Optional


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
) -> str:
    """Create a minimal LS-DYNA keyword deck for OpenRadioss Starter."""
    points, tets = mesh_to_tet4(mesh, warnings)

    e = float(material.get("E", 210000.0))
    nu = float(material.get("nu", material.get("poissons_ratio", 0.3)))
    rho = float(material.get("rho", material.get("density", 7.85e-9)))
    yield_strength = float(material.get("yield_strength", 250.0))
    tangent_modulus = float(material.get("tangent_modulus", 2000.0))
    failure_strain = float(material.get("failure_strain", 0.0))

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
            f"0.0, 0.0, {failure_strain:.6g}, 0.0",
        ]
    )

    next_set_id = 100
    for idx, constraint in enumerate(constraints, start=1):
        geoms = dict_geometries(constraint)
        if not geoms:
            warnings.append(f"Crash constraint {idx} has no face geometry; skipped.")
            continue
        node_ids = nodes_matching_geometries(mesh, geoms) + 1
        if len(node_ids) == 0:
            warnings.append(f"Crash constraint {idx} did not match any mesh nodes.")
            continue
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

    velocity = np.asarray(impact.get("velocity", [0.0, 0.0, 0.0]), dtype=float)
    impact_faces = impact.get("face_list", [])
    if impact_faces:
        impact_nodes = nodes_matching_geometries(
            mesh,
            impact_faces,
            tolerance=float(impact.get("node_tolerance", 2.0)),
        ) + 1
    else:
        impact_nodes = np.arange(1, points.shape[0] + 1, dtype=int)

    if len(impact_nodes) == 0:
        warnings.append("Impact condition did not match any nodes; no initial velocity exported.")
    elif np.linalg.norm(velocity) > 0.0:
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

    # Gravity body force via *LOAD_BODY_X/Y/Z (LS-DYNA syntax accepted by Radioss).
    if gravity and float(gravity.get("accel", 0.0)) != 0.0:
        direction = gravity.get("direction", "-Y")
        accel = float(gravity.get("accel", 9810.0))
        dir_map = {
            "-X": ("X", -accel),
            "+X": ("X", +accel),
            "-Y": ("Y", -accel),
            "+Y": ("Y", +accel),
            "-Z": ("Z", -accel),
            "+Z": ("Z", +accel),
        }
        axis, signed = dir_map.get(direction, ("Y", -accel))
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

    lines.append("*END")
    return "\n".join(lines) + "\n"


def _build_engine_deck(job_name: str, end_time: float) -> str:
    """Write a minimal OpenRadioss ``<job>_0001.rad`` engine deck."""
    return "\n".join(
        [
            "#RADIOSS ENGINE INPUT",
            f"/RUN/{job_name}/1",
            f"{float(end_time):.6g}",
            "/STOP",
            "",
        ]
    )


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
    print(f"OpenRadioss Engine: launching on {staged_engine.name}... "
          "(this is where most of the wall-clock time goes)")
    proc_eng = run_process(
        [engine, "-i", str(staged_engine)],
        cwd=work_dir,
        timeout_s=config.timeout_s,
        extra_path_dirs=path_dirs,
        extra_env=env_extra,
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
    """Pad / truncate animation frame arrays to match the source mesh point count."""
    n_points = int(np.asarray(mesh.p).shape[1])
    fixed: list = []
    for frame in frames:
        flat = np.asarray(frame.get("displacement", []), dtype=float).reshape(-1)
        vm = np.asarray(frame.get("stress_vm", []), dtype=float).reshape(-1)
        # Ensure length 3*N and N respectively.
        target_disp = np.zeros(3 * n_points, dtype=float)
        target_disp[: min(flat.size, target_disp.size)] = flat[: target_disp.size]
        target_vm = np.zeros(n_points, dtype=float)
        target_vm[: min(vm.size, target_vm.size)] = vm[: target_vm.size]
        fixed.append(
            {
                "displacement": target_disp,
                "stress_vm": target_vm,
                "eps_p": np.zeros(n_points, dtype=float),
                "failed": np.zeros(n_points, dtype=float),
                "time": float(frame.get("time", 0.0)),
            }
        )
    return fixed


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
) -> dict:
    """Write deck, run Starter + Engine, then import animation frames."""
    if impact is None:
        raise SolverBackendError("OpenRadioss backend requires an impact condition.")

    warnings: List[str] = []
    work_dir = make_work_dir("pylcss_openradioss_", config.work_dir)
    job_name = config.job_name or "pylcss_openradioss"
    deck_path = work_dir / f"{job_name}.k"
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
        ),
        encoding="utf-8",
    )

    engine_deck_path = work_dir / f"{job_name}_0001.rad"
    engine_deck_path.write_text(_build_engine_deck(job_name, end_time), encoding="utf-8")

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
                path_dirs, env_extra = _radioss_runtime_env(engine)
                import time as _time
                _t0 = _time.time()
                print(f"OpenRadioss Engine: launching on {Path(engine_deck_path).name}... "
                      "(this is where most of the wall-clock time goes)")
                proc_eng = run_process(
                    [engine, "-i", str(engine_deck_path)],
                    cwd=work_dir,
                    timeout_s=config.timeout_s,
                    extra_path_dirs=path_dirs,
                    extra_env=env_extra,
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

    # Choose result type based on what we managed to import.
    if frames:
        n_points = int(np.asarray(mesh.p).shape[1])
        last = frames[-1]
        displacement_flat = last["displacement"]
        stress_field = last["stress_vm"]
        peak_disp = float(np.max(np.linalg.norm(displacement_flat.reshape(n_points, 3), axis=1)))
        peak_vm = float(np.max(stress_field)) if stress_field.size else 0.0
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
