# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEA solver node — runs an external CalculiX (ccx) static linear analysis.

The previous in-house scikit-fem implementation was removed: it was not
validated against an industry reference and produced silent correctness
risks (volumetric locking, stress under-smoothing, sign mistakes in
boundary-condition assembly).  CalculiX is the single supported backend.
"""
import logging

from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


class SolverNode(CadQueryNode):
    """Static linear FEA solver — dispatches to CalculiX (ccx)."""
    __identifier__ = 'com.cad.sim.solver'
    NODE_NAME = 'FEA Solver'

    def __init__(self):
        super().__init__()
        self.add_input('mesh',        color=(200, 100, 200))
        self.add_input('material',    color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads',       color=(255, 255,   0), multi_input=True)
        self.add_output('results',    color=(  0, 255, 255))

        self.create_property('external_solver_path', '',  widget_type='text')
        self.create_property('external_work_dir',    '',  widget_type='text')
        # Write the .inp deck without launching ccx (for inspection).
        self.create_property('deck_only',            False, widget_type='checkbox')
        self.create_property('external_timeout_s',   3600.0, widget_type='float')
        self.create_property('visualization', 'Von Mises Stress', widget_type='combo',
                             items=['Von Mises Stress', 'Displacement'])
        self.create_property('deformation_scale', 'Auto', widget_type='combo',
                             items=['Auto', '1x', '5x', '10x', '50x', '100x', '200x'])
        # Linear vs nonlinear static.  'Nonlinear (Plastic)' is auto-enabled
        # whenever the connected material has yield_strength > 0 — picking it
        # here just makes the intent explicit in the deck header.
        self.create_property('analysis_type', 'Linear', widget_type='combo',
                             items=['Linear',
                                    'Nonlinear (Geometric)',
                                    'Nonlinear (Plastic)'])

        # Deprecated — kept so projects saved before the CalculiX-only cut load
        # cleanly. The in-house scikit-fem path no longer exists; ``CalculiX`` is
        # the only backend and is always used regardless of this property.
        self.create_property('solver_backend', 'CalculiX', widget_type='combo',
                             items=['CalculiX'])
        self.create_property('run_external_solver', True, widget_type='checkbox')

    def run(self):
        print("FEA Solver: routing to CalculiX backend.")
        from pylcss.solver_backends import (
            ExternalRunConfig,
            SolverBackendError,
            run_calculix_static,
        )
        from pylcss.solver_backends.common import as_bool, flatten_inputs

        mesh = self.get_input_value('mesh', None)
        material = self.get_input_value('material', None)
        constraint_list = flatten_inputs(self.get_input_list('constraints'))
        load_list = flatten_inputs(self.get_input_list('loads'))
        print(
            f"FEA Solver: mesh={mesh is not None}, material={material is not None}, "
            f"constraints={len(constraint_list)}, loads={len(load_list)}"
        )

        missing = []
        if mesh is None:
            missing.append('mesh')
        if material is None:
            missing.append('material')
        if not constraint_list:
            missing.append('at least one constraint')
        if not load_list:
            missing.append('at least one load')
        if missing:
            msg = "CalculiX backend requires " + ", ".join(missing) + "."
            print(f"FEA Solver: {msg}")
            self.set_error(msg)
            return None

        # Selecting the solver implicitly means "run it"; `deck_only` is the
        # explicit opt-out for users who only want the input deck.  We still
        # honour the legacy `run_external_solver` property on projects saved
        # before this change.
        deck_only = as_bool(self.get_property('deck_only'))
        legacy_run = self.get_property('run_external_solver')
        if legacy_run is not None and not as_bool(legacy_run):
            deck_only = True
        run_flag = not deck_only
        print(f"FEA Solver: deck_only={deck_only}, run_solver={run_flag}")

        try:
            config = ExternalRunConfig(
                executable=(self.get_property('external_solver_path') or None),
                work_dir=(self.get_property('external_work_dir') or None),
                keep_files=True,
                run_solver=run_flag,
                timeout_s=float(self.get_property('external_timeout_s') or 3600.0),
                job_name='pylcss_calculix',
            )
            result = run_calculix_static(
                mesh=mesh,
                material=material,
                constraints=constraint_list,
                loads=load_list,
                config=config,
                visualization_mode=self.get_property('visualization'),
                analysis_type=(self.get_property('analysis_type') or 'Linear'),
            )
            warnings = result.get('warnings') or []
            if warnings:
                print("CalculiX backend warnings:\n  " + "\n  ".join(warnings))
            print(
                f"FEA Solver: status={result.get('external_status')}, "
                f"type={result.get('type')}, "
                f"work_dir={result.get('work_dir')}, "
                f"solver_exe={result.get('solver_executable')}"
            )
            return result
        except SolverBackendError as exc:
            print(f"FEA Solver: CalculiX backend error: {exc}")
            self.set_error(str(exc))
            return None
        except Exception as exc:
            import traceback
            print(f"FEA Solver: External backend raised {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self.set_error(f"CalculiX backend crashed: {exc}")
            return None
