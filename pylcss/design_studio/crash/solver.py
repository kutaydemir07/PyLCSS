# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash solver node — runs an external OpenRadioss explicit transient analysis.

The previous in-house CPU/GPU explicit solvers (central-difference, J2
plasticity, penalty contact, Taichi kernels) were removed: they were not
validated against an industry reference for crash physics.  OpenRadioss
is the single supported backend.
"""
import logging
import math
from pathlib import Path

from pylcss.design_studio.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)

INDUSTRIAL_HOURGLASS_IHQ = 4
INDUSTRIAL_HOURGLASS_COEFFICIENT = 0.10


class CrashSolverNode(CadQueryNode):
    """Explicit transient crash solver — dispatches to OpenRadioss."""
    __identifier__ = 'com.cad.sim.crash_solver'
    NODE_NAME = 'Crash Solver'

    def __init__(self):
        super().__init__()
        self.add_input('mesh',           color=(200, 100, 200))
        self.add_input('crash_material', color=(255, 150,  50))
        self.add_input('constraints',    color=(255, 100, 100), multi_input=True)
        self.add_input('impact',         color=(255, 200,   0))
        self.add_output('crash_results', color=(  0, 220, 255))

        self.create_property('openradioss_starter_path', '', widget_type='text')
        self.create_property('openradioss_engine_path',  '', widget_type='text')
        self.create_property('external_work_dir',        '', widget_type='text')
        # Write the OpenRadioss deck without launching the starter/engine.
        self.create_property('deck_only', False, widget_type='checkbox')
        self.create_property('external_timeout_s', 1800.0, widget_type='float')

        # Simulation duration (ms) and result-frame count.  Keep new nodes in
        # preview territory; users can raise both for final verification runs.
        self.create_property('end_time',   0.5, widget_type='float')
        self.create_property('n_frames',   30,  widget_type='int')

        # Viewer-only — picks which field colours the deformed mesh.
        self.create_property(
            'visualization', 'Von Mises Stress',
            widget_type='combo',
            items=['Von Mises Stress', 'Displacement', 'Plastic Strain', 'Failed Elements'],
        )
        # Multiplies displayed displacement; does not affect physics.
        self.create_property('disp_scale', 3.0, widget_type='float')

        # Hidden compatibility fields for pre-OpenRadioss-only sessions.
        self.create_property('solver_backend', 'OpenRadioss', widget_type='combo',
                             items=['OpenRadioss'])
        self.create_property('run_external_solver', True, widget_type='checkbox')

        # OpenRadioss mass-scaling target.  When enable_mass_scaling is true,
        # the engine deck requests /DT/NODA/CST with dt = end_time / time_steps.
        self.create_property('time_steps', 500, widget_type='int')
        self.create_property('enable_mass_scaling', False, widget_type='checkbox')

        self.create_property('damping_alpha', 0.0, widget_type='float')
        self.create_property('damping_beta', 0.0, widget_type='float')
        self.create_property('enable_corotation', True, widget_type='checkbox')
        self.create_property('enable_contact', False, widget_type='checkbox')
        self.create_property('contact_stiffness', 0.1, widget_type='float')
        self.create_property('contact_thickness', 0.2, widget_type='float')
        self.create_property('contact_update_interval', 5, widget_type='int')
        self.create_property('mass_scaling_threshold', 0.0, widget_type='float')

        # Impactor (sled) mass in kg.  In "Fixed specimen + moving impactor"
        # this becomes the moving rigid wall mass; in "Moving body + fixed wall"
        # it is lumped onto the projectile's trailing edge.
        # Without a sled mass the crashbox (~250 g) has too little KE to crush
        # plastically at realistic velocities — it bounces elastically.
        # Set to 0 to disable; typical component test values: 25–200 kg.
        self.create_property('impactor_mass_kg', 0.0, widget_type='float')

    def run(self, cancel_callback=None):
        logger.info("Crash Solver: routing to OpenRadioss backend")
        from pylcss.solver_backends import (
            ExternalRunConfig,
            SolverBackendError,
            run_openradioss_crash,
        )
        from pylcss.solver_backends.common import as_bool, flatten_inputs

        mesh = self.get_input_value('mesh', None)
        material = self.get_input_value('crash_material', None)
        impact = self.get_input_value('impact', None)
        constraints = flatten_inputs(self.get_input_list('constraints'))
        logger.debug(
            f"Crash Solver: mesh={mesh is not None}, "
            f"material={material is not None}, impact={impact is not None}, "
            f"constraints={len(constraints)}"
        )

        missing = []
        if mesh is None:
            missing.append('mesh')
        if material is None:
            missing.append('crash_material')
        if impact is None:
            missing.append('impact')
        if missing:
            msg = "OpenRadioss backend requires " + ", ".join(missing) + "."
            self.set_error(msg)
            return None

        scope = str(impact.get('application_scope') or '').strip().lower().replace('_', ' ')
        if not scope.startswith('moving body'):
            if not (impact.get('face_list') or []):
                msg = "Fixed-specimen crash requires a selected impact face."
                self.set_error(msg)
                return None
            if not constraints and not scope.startswith('prescribed'):
                msg = "Fixed-specimen moving-impactor crash requires a rear support constraint."
                self.set_error(msg)
                return None

        try:
            n_frames = int(self.get_property('n_frames') or 0)
            end_time = float(self.get_property('end_time') or 0.0)
            timeout_s = float(self.get_property('external_timeout_s') or 0.0)
            impactor_mass = float(self.get_property('impactor_mass_kg') or 0.0)
            disp_scale = float(self.get_property('disp_scale') or 0.0)
        except (TypeError, ValueError):
            self.set_error("Crash duration, frames, timeout, mass, and display scale must be numeric.")
            return None
        if n_frames < 1 or not math.isfinite(end_time) or end_time <= 0.0:
            self.set_error("Crash end time must be positive and result frames must be at least 1.")
            return None
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            self.set_error("OpenRadioss timeout must be finite and greater than zero.")
            return None
        if not math.isfinite(impactor_mass) or impactor_mass < 0.0:
            self.set_error("Impactor mass must be finite and non-negative.")
            return None
        if not math.isfinite(disp_scale) or disp_scale <= 0.0:
            self.set_error("Crash display scale must be finite and greater than zero.")
            return None
        output_dt = end_time / n_frames

        deck_only = as_bool(self.get_property('deck_only'))
        legacy_run = self.get_property('run_external_solver')
        if legacy_run is not None and not as_bool(legacy_run):
            deck_only = True
        run_flag = not deck_only
        logger.info("Crash Solver: deck_only=%s, run_solver=%s", deck_only, run_flag)

        try:
            starter_path = str(
                self.get_property('openradioss_starter_path') or ''
            ).strip() or None
            engine_path = str(
                self.get_property('openradioss_engine_path') or ''
            ).strip() or None
            work_dir = str(self.get_property('external_work_dir') or '').strip() or None
            project_dir = getattr(self, '_project_dir', None)
            if project_dir:
                if starter_path and not Path(starter_path).is_absolute():
                    starter_path = str(Path(project_dir) / starter_path)
                if engine_path and not Path(engine_path).is_absolute():
                    engine_path = str(Path(project_dir) / engine_path)
                if work_dir and not Path(work_dir).is_absolute():
                    work_dir = str(Path(project_dir) / work_dir)
            config = ExternalRunConfig(
                executable=starter_path,
                secondary_executable=engine_path,
                work_dir=work_dir,
                run_solver=run_flag,
                timeout_s=timeout_s,
                job_name='pylcss_openradioss',
                cancel_callback=cancel_callback,
            )
            # Mass scaling (/DT/NODA/CST): add nodal mass when element dt drops
            # below the target, preventing timestep collapse during element
            # distortion in the crush zone.
            # Safe with *ELEMENT_MASS impactor masses: those rear nodes have
            # enormous effective dt (~100 ms) so /DT/NODA/CST never touches them;
            # only distorting crush-zone elements (dt → μs) get mass added.
            ms_enabled = as_bool(self.get_property('enable_mass_scaling'))
            ms_dt_target = 0.0
            if ms_enabled:
                steps_req = max(int(self.get_property('time_steps') or 500), 1)
                ms_dt_target = float(end_time) / steps_req

            # Hourglass control is solver policy, not a study knob.
            hg_ihq = INDUSTRIAL_HOURGLASS_IHQ
            hg_coef = INDUSTRIAL_HOURGLASS_COEFFICIENT

            result = run_openradioss_crash(
                mesh=mesh,
                material=material,
                constraints=constraints,
                impact=impact,
                config=config,
                end_time=end_time,
                output_dt=output_dt,
                visualization_mode=self.get_property('visualization'),
                disp_scale=disp_scale,
                mass_scaling_dt=ms_dt_target,
                mass_scaling_scale=0.67,
                impactor_mass=impactor_mass,
                hourglass_ihq=hg_ihq,
                hourglass_coefficient=hg_coef,
            )
            warnings = result.get('warnings') or []
            if warnings:
                logger.warning("OpenRadioss backend warnings: %s", "; ".join(warnings))
            logger.info(
                "Crash Solver: status=%s, type=%s, work_dir=%s",
                result.get('external_status'), result.get('type'), result.get('work_dir'),
            )
            return result
        except SolverBackendError as exc:
            logger.warning("Crash Solver: OpenRadioss backend error: %s", exc)
            self.set_error(str(exc))
            return None
        except Exception as exc:
            logger.exception("Crash Solver: external backend crashed")
            self.set_error(f"OpenRadioss backend crashed: {exc}")
            return None
