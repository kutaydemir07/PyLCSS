# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Automatic preview of the inspector's preserved face layers."""

from pylcss.cae.design_studio.topology_optimization.geometry.non_design import (
    cached_non_design_payloads,
    design_domain_shape,
    non_design_payloads,
)


class NonDesignWorkbenchMixin:
    @staticmethod
    def _cached_topopt_design_domain(node):
        return node.get_cached_input_value("design_domain")

    def _preview_topopt_non_design(self, node, *, render_domain=False, cached_only=False):
        if self.properties.current_node is not node or not node.has_property(
            "non_design_regions"
        ):
            return
        if self._execution_is_active() or getattr(self.viewer, "_picking_mode", False):
            return
        geometry = self._cached_topopt_design_domain(node)
        if geometry is None:
            self.viewer.render_non_design_layers([])
            return
        try:
            payloads = cached_non_design_payloads(node, geometry)
            if payloads is None:
                payloads = non_design_payloads(node, geometry)
            if payloads is None:
                self.viewer.render_non_design_layers([])
                status = getattr(self.properties, "_non_design_status", None)
                if status is not None:
                    status.setText("Run to update the preserved-region preview.")
                return
            if render_domain and payloads:
                self._render_result_in_viewer(design_domain_shape(geometry))
            self.viewer.render_non_design_layers(
                [payload["geometry"] for payload in payloads]
            )
            message = (
                f"{len(payloads)} preserved region(s) · cyan preview"
                if payloads
                else "No active non-design face layers"
            )
        except (ValueError, RuntimeError) as exc:
            self.viewer.render_non_design_layers([])
            message = str(exc)
        status = getattr(self.properties, "_non_design_status", None)
        if status is not None:
            status.setText(message)
