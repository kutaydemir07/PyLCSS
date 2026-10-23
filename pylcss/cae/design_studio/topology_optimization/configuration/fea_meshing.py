# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Mesh controls for independent FEA of the recovered CAD solid."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


FEA_MESH_DEFAULTS = {
    "fea_element_size_mm": 0.0,
    "fea_min_element_size_mm": 0.0,
    "fea_mesh_grading": 0.3,
    "fea_element_order": "C3D10",
    "fea_max_elements": 500_000,
    "fea_convergence_tolerance": 0.05,
    "fea_surface_tolerance_mm": 0.0,
    "fea_solver_backend": 'Automatic',
}
FEA_QUALITY_OPTIONS = ("Standard", "Mesh Convergence")
FEA_ELEMENT_ORDERS = {
    "Quadratic tetrahedra (C3D10)": "C3D10",
    "Linear tetrahedra (C3D4)": "C3D4",
}


def fea_mesh_options_from_node(node):
    """Read FEA controls and strip their node-property prefix."""
    options = {}
    for property_name, default in FEA_MESH_DEFAULTS.items():
        value = node.get_property(property_name)
        options[property_name.removeprefix("fea_")] = (
            default if value is None else value
        )
    return options


def is_topopt_fea_setting(node, name):
    """FEA edits prepare the next analysis without invalidating the topology."""
    return getattr(
        node, "__identifier__", ""
    ) == "com.cad.sim.topopt_voxel" and name in {
        *FEA_MESH_DEFAULTS,
        "fea_quality",
        "validation_yield_safety_factor",
        "validation_displacement_limit_mm",
    }


@dataclass(frozen=True)
class CadFeaMeshOptions:
    element_size_mm: float = 0.0
    min_element_size_mm: float = 0.0
    mesh_grading: float = 0.3
    element_order: str = "C3D10"
    max_elements: int = 500_000
    convergence_tolerance: float = 0.05
    surface_tolerance_mm: float = 0.0
    solver_backend: str = 'Automatic'

    def __post_init__(self):
        if self.solver_backend not in ('Automatic', 'Sparse Direct', 'CalculiX'):
            raise ValueError('FEA solver backend must be Automatic, Sparse Direct, or CalculiX.')
        for name in ("element_size_mm", "min_element_size_mm", "mesh_grading", "convergence_tolerance", "surface_tolerance_mm"):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError(
                    f"FEA {name.replace('_', ' ')} must be numeric."
                )
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"FEA {name.replace('_', ' ')} must be numeric."
                ) from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"FEA {name.replace('_', ' ')} must be finite and nonnegative."
                )
            object.__setattr__(self, name, value)
        if self.element_size_mm > 0 and self.min_element_size_mm > self.element_size_mm:
            raise ValueError(
                "FEA minimum element size cannot exceed maximum element size."
            )
        if not 0 < self.mesh_grading <= 1:
            raise ValueError(
                "FEA mesh grading must be greater than zero and at most 1."
            )
        if self.element_order not in FEA_ELEMENT_ORDERS.values():
            raise ValueError("Choose C3D4 or C3D10 FEA tetrahedra.")
        if isinstance(self.max_elements, bool):
            raise ValueError("FEA maximum elements must be a positive integer.")
        try:
            limit = float(self.max_elements)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "FEA maximum elements must be a positive integer."
            ) from exc
        if not math.isfinite(limit) or limit < 1 or not limit.is_integer():
            raise ValueError("FEA maximum elements must be a positive integer.")
        object.__setattr__(self, "max_elements", int(limit))

    def resolve_sizes(self, spans):
        """Resolve automatic sizing without silently coarsening a request."""
        try:
            if isinstance(spans, (str, bytes)):
                raise ValueError
            spans = tuple(
                float(span) if not isinstance(span, bool) else math.nan
                for span in spans
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("CAD FEA requires finite positive body dimensions.") from exc
        if len(spans) != 3 or any(not math.isfinite(s) or s <= 0 for s in spans):
            raise ValueError("CAD FEA requires finite positive body dimensions.")
        maximum = self.element_size_mm or min(max(spans) / 20, min(spans) / 2)
        if self.min_element_size_mm > maximum:
            raise ValueError(
                "FEA minimum element size cannot exceed maximum element size."
            )
        return maximum, self.min_element_size_mm

    def as_dict(self):
        return asdict(self)
