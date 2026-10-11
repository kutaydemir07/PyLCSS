# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Modal FEA solver node — extracts natural frequencies and mode shape eigenvectors."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from pylcss.cae.design_studio.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


class ModalSolverNode(CadQueryNode):
    """CalculiX natural frequency extraction and mode shape vibration analysis."""

    __identifier__ = "com.cad.sim.modal_solver"
    NODE_NAME = "Modal Solver"

    def __init__(self):
        super().__init__()
        self.add_input("mesh", color=(200, 100, 200))
        self.add_input("material", color=(200, 200, 200))
        self.add_input("constraints", color=(255, 100, 100), multi_input=True)
        self.add_output("results", color=(0, 255, 255))

        self.create_property("num_modes", 6, widget_type="int")
        self.create_property("active_mode", 1, widget_type="int")
        self.create_property(
            "deformation_scale",
            "Auto",
            widget_type="combo",
            items=["Auto", "1x", "5x", "10x", "50x", "100x"],
        )
        self.create_property("external_solver_path", "", widget_type="text")
        self.create_property("external_work_dir", "", widget_type="text")
        self.create_property("deck_only", False, widget_type="checkbox")
        self.create_property("external_timeout_s", 3600.0, widget_type="float")

    def run(self, cancel_callback=None):
        logger.debug("Modal Solver: routing to CalculiX frequency backend.")
        from pylcss.cae.solver_backends import ExternalRunConfig, SolverBackendError
        from pylcss.cae.solver_backends.calculix_modal import run_calculix_modal
        from pylcss.input_values import as_bool, flatten_inputs

        self.clear_error()
        mesh = self.get_input_value("mesh", None)
        material = self.get_input_value("material", None)
        constraint_list = flatten_inputs(self.get_input_list("constraints"))

        if mesh is None:
            self.set_error("Modal Solver requires an FE solid mesh.")
            return None
        if material is None:
            self.set_error("Modal Solver requires an assigned material.")
            return None

        try:
            rho = float(material.get("rho", material.get("density", 0.0)) or 0.0)
        except (AttributeError, TypeError, ValueError):
            rho = 0.0
        if not math.isfinite(rho) or rho <= 0.0:
            self.set_error(
                "Modal analysis requires a non-zero material density (Material.density > 0)."
            )
            return None

        try:
            num_modes = int(self.get_property("num_modes") or 6)
            num_modes = max(1, min(100, num_modes))
        except (TypeError, ValueError):
            num_modes = 6

        try:
            active_mode = int(self.get_property("active_mode") or 1)
            active_mode = max(1, active_mode)
        except (TypeError, ValueError):
            active_mode = 1

        deck_only = as_bool(self.get_property("deck_only"))
        run_flag = not deck_only

        try:
            timeout_s = float(self.get_property("external_timeout_s") or 3600.0)
            if not math.isfinite(timeout_s) or timeout_s <= 0.0:
                self.set_error("Solver timeout must be finite and greater than zero.")
                return None

            solver_path = (
                str(self.get_property("external_solver_path") or "").strip() or None
            )
            work_dir = (
                str(self.get_property("external_work_dir") or "").strip() or None
            )
            project_dir = getattr(self, "_project_dir", None)
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
                job_name="pylcss_modal",
                cancel_callback=cancel_callback,
            )

            result = run_calculix_modal(
                mesh=mesh,
                material=material,
                constraints=constraint_list,
                num_modes=num_modes,
                active_mode=active_mode,
                config=config,
                deformation_scale=self.get_property("deformation_scale"),
            )
            return result
        except SolverBackendError as exc:
            logger.error("Modal Solver error: %s", exc)
            self.set_error(str(exc))
            return None
        except Exception as exc:
            logger.exception("Modal Solver backend crashed: %s", exc)
            self.set_error(f"Modal solver crashed: {exc}")
            return None
