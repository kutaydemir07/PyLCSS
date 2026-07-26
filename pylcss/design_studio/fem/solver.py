# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEA solver node — runs an external CalculiX (ccx) static linear analysis.

The previous in-house scikit-fem implementation was removed: it was not
validated against an industry reference and produced silent correctness
risks (volumetric locking, stress under-smoothing, sign mistakes in
boundary-condition assembly).  CalculiX is the single supported backend.
"""
import logging
import math
from pathlib import Path

from pylcss.design_studio.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


def _has_nonzero_prescribed_displacement(constraints, tol=1e-15):
    """True when a constraint drives at least one DOF to a non-zero value."""
    for constraint in constraints or []:
        if not isinstance(constraint, dict):
            continue
        displacement = constraint.get('displacement')
        if displacement is None:
            continue
        for dof in constraint.get('fixed_dofs', range(len(displacement))):
            try:
                if abs(float(displacement[int(dof)])) > tol:
                    return True
            except (IndexError, TypeError, ValueError):
                continue
    return False


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
        # The constitutive model is explicit. A yield strength is retained as
        # an allowable for linear safety-factor reporting and never silently
        # promotes a study to nonlinear plasticity.
        self.create_property('analysis_type', 'Linear', widget_type='combo',
                             items=['Linear',
                                    'Nonlinear (Geometric)',
                                    'Nonlinear (Plastic)'])

        # Hidden compatibility properties: required so older .cad sessions
        # deserialize. They are excluded from the inspector.
        self.create_property('solver_backend', 'CalculiX', widget_type='combo',
                             items=['CalculiX'])
        self.create_property('run_external_solver', True, widget_type='checkbox')

    def run(self, cancel_callback=None):
        logger.debug("FEA Solver: routing to CalculiX backend.")
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
        logger.debug(
            "FEA Solver: mesh=%s, material=%s, constraints=%d, loads=%d",
            mesh is not None, material is not None,
            len(constraint_list), len(load_list),
        )

        if mesh is not None and getattr(getattr(mesh, 't', None), 'shape', (0,))[0] == 3:
            msg = (
                "The CalculiX FEA solver supports Tet and Tet10 solid meshes only. "
                "Shell meshes are supported by the OpenRadioss Crash Solver."
            )
            self.set_error(msg)
            return None
        missing = []
        if mesh is None:
            missing.append('mesh')
        if material is None:
            missing.append('material')
        if not constraint_list:
            missing.append('at least one constraint')
        driven_by_displacement = _has_nonzero_prescribed_displacement(constraint_list)
        if not load_list and not driven_by_displacement:
            missing.append('at least one load')
        if missing:
            msg = "CalculiX backend requires " + ", ".join(missing) + "."
            logger.warning("FEA Solver: %s", msg)
            self.set_error(msg)
            return None

        # Selecting the solver implicitly means "run it"; `deck_only` is the
        # explicit opt-out for users who only want the input deck.
        deck_only = as_bool(self.get_property('deck_only'))
        legacy_run = self.get_property('run_external_solver')
        if legacy_run is not None and not as_bool(legacy_run):
            deck_only = True
        run_flag = not deck_only
        logger.debug("FEA Solver: deck_only=%s, run_solver=%s", deck_only, run_flag)

        try:
            timeout_s = float(self.get_property('external_timeout_s') or 0.0)
            if not math.isfinite(timeout_s) or timeout_s <= 0.0:
                self.set_error("CalculiX timeout must be finite and greater than zero.")
                return None
            solver_path = str(self.get_property('external_solver_path') or '').strip() or None
            work_dir = str(self.get_property('external_work_dir') or '').strip() or None
            project_dir = getattr(self, '_project_dir', None)
            if project_dir:
                if solver_path and not Path(solver_path).is_absolute():
                    solver_path = str(Path(project_dir) / solver_path)
                if work_dir and not Path(work_dir).is_absolute():
                    work_dir = str(Path(project_dir) / work_dir)
            config = ExternalRunConfig(
                executable=solver_path,
                work_dir=work_dir,
                run_solver=run_flag,
                timeout_s=timeout_s,
                job_name='pylcss_calculix',
                cancel_callback=cancel_callback,
            )
            result = run_calculix_static(
                mesh=mesh,
                material=material,
                constraints=constraint_list,
                loads=load_list,
                config=config,
                visualization_mode=self.get_property('visualization'),
                analysis_type=(self.get_property('analysis_type') or 'Linear'),
                deformation_scale=self.get_property('deformation_scale'),
            )
            warnings = result.get('warnings') or []
            if warnings:
                logger.debug("CalculiX backend warnings:\n  %s", "\n  ".join(warnings))
            logger.debug(
                "FEA Solver: status=%s, type=%s, work_dir=%s, solver_exe=%s",
                result.get('external_status'), result.get('type'),
                result.get('work_dir'), result.get('solver_executable'),
            )
            return result
        except SolverBackendError as exc:
            logger.error("FEA Solver: CalculiX backend error: %s", exc)
            self.set_error(str(exc))
            return None
        except Exception as exc:
            logger.exception("FEA Solver: External backend raised %s", type(exc).__name__)
            self.set_error(f"CalculiX backend crashed: {exc}")
            return None
