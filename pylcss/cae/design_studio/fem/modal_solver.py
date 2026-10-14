# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Modal FEA solver node — extracts natural frequencies and mode shape eigenvectors."""

from __future__ import annotations

import logging
import math

from pylcss.cae.design_studio.core.base_node import CadQueryNode
from pylcss.cae.design_studio.fem._solver_config import (
    add_external_solver_properties,
    external_run_config,
)

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
        add_external_solver_properties(self)

    def run(self, cancel_callback=None):
        logger.debug("Modal Solver: routing to CalculiX frequency backend.")
        from pylcss.cae.solver_backends import SolverBackendError
        from pylcss.cae.solver_backends.calculix_modal import run_calculix_modal
        from pylcss.cae.solver_backends.validation import integer
        from pylcss.input_values import flatten_inputs

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
            num_modes = integer(
                self.get_property("num_modes"), label="Number of modes", minimum=1
            )
            active_mode = integer(
                self.get_property("active_mode"), label="Active mode", minimum=1
            )
            config = external_run_config(
                self,
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
