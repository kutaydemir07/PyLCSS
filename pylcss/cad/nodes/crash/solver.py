# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Crash solver node — runs an external OpenRadioss explicit transient analysis.

The previous in-house CPU/GPU explicit solvers (central-difference, J2
plasticity, penalty contact, Taichi kernels) were removed: they were not
validated against an industry reference for crash physics.  OpenRadioss
is the single supported backend.
"""
import logging

from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


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

        # Deprecated — kept so projects saved before the OpenRadioss-only cut
        # load cleanly. The in-house CPU/GPU explicit solver no longer exists;
        # OpenRadioss is the only backend and is always used.
        self.create_property('solver_backend', 'OpenRadioss', widget_type='combo',
                             items=['OpenRadioss'])
        self.create_property('run_external_solver', True, widget_type='checkbox')
        self.create_property('time_steps', 500, widget_type='int')
        self.create_property('damping_alpha', 10.0, widget_type='float')
        self.create_property('enable_corotation', True, widget_type='checkbox')
        self.create_property('enable_contact', False, widget_type='checkbox')
        self.create_property('contact_stiffness', 0.1, widget_type='float')
        self.create_property('contact_thickness', 0.2, widget_type='float')
        self.create_property('contact_update_interval', 10, widget_type='int')
        self.create_property('enable_mass_scaling', False, widget_type='checkbox')
        self.create_property('mass_scaling_threshold', 0.05, widget_type='float')
        # Impactor (sled) mass in kg added to the crashbox inertia.
        # Without a sled mass the crashbox (~250 g) has too little KE to crush
        # plastically at realistic velocities — it bounces elastically.
        # Set to 0 to disable; typical component test values: 25–200 kg.
        self.create_property('impactor_mass_kg', 0.0, widget_type='float')

    def run(self):
        print("Crash Solver: routing to OpenRadioss backend.")
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
        print(
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
            print(f"Crash Solver: {msg}")
            self.set_error(msg)
            return None

        n_frames = max(1, int(self.get_property('n_frames') or 1))
        end_time = float(self.get_property('end_time') or 0.0)
        output_dt = end_time / n_frames if end_time > 0 else 1.0

        deck_only = as_bool(self.get_property('deck_only'))
        legacy_run = self.get_property('run_external_solver')
        if legacy_run is not None and not as_bool(legacy_run):
            deck_only = True
        run_flag = not deck_only
        print(f"Crash Solver: deck_only={deck_only}, run_solver={run_flag}")

        try:
            config = ExternalRunConfig(
                executable=(self.get_property('openradioss_starter_path') or None),
                secondary_executable=(self.get_property('openradioss_engine_path') or None),
                work_dir=(self.get_property('external_work_dir') or None),
                keep_files=True,
                run_solver=run_flag,
                timeout_s=float(self.get_property('external_timeout_s') or 1800.0),
                job_name='pylcss_openradioss',
            )
            impactor_mass = float(self.get_property('impactor_mass_kg') or 0.0)
            # Mass scaling (/DT/NODA/CST): add nodal mass when element dt drops
            # below the target, preventing timestep collapse during element
            # distortion in the crush zone.
            # Safe with *ELEMENT_MASS impactor masses: those rear nodes have
            # enormous effective dt (~100 ms) so /DT/NODA/CST never touches them;
            # only distorting crush-zone elements (dt → μs) get mass added.
            ms_enabled = bool(self.get_property('enable_mass_scaling'))
            ms_dt_target = 0.0
            if ms_enabled:
                steps_req = max(int(self.get_property('time_steps') or 500), 1)
                ms_dt_target = float(end_time) / steps_req

            result = run_openradioss_crash(
                mesh=mesh,
                material=material,
                constraints=constraints,
                impact=impact,
                config=config,
                end_time=end_time,
                output_dt=output_dt,
                visualization_mode=self.get_property('visualization'),
                disp_scale=float(self.get_property('disp_scale') or 1.0),
                mass_scaling_dt=ms_dt_target,
                mass_scaling_scale=0.67,
                impactor_mass=impactor_mass,
            )
            warnings = result.get('warnings') or []
            if warnings:
                print("OpenRadioss backend warnings:\n  " + "\n  ".join(warnings))
            print(
                f"Crash Solver: status={result.get('external_status')}, "
                f"type={result.get('type')}, "
                f"work_dir={result.get('work_dir')}, "
                f"starter_exe={result.get('solver_executable')}, "
                f"engine_exe={result.get('secondary_solver_executable')}"
            )
            return result
        except SolverBackendError as exc:
            print(f"Crash Solver: OpenRadioss backend error: {exc}")
            self.set_error(str(exc))
            return None
        except Exception as exc:
            import traceback
            print(f"Crash Solver: External backend raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self.set_error(f"OpenRadioss backend crashed: {exc}")
            return None
