# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Public orchestration for recovered topology mesh to STEP reconstruction."""

from __future__ import annotations

import logging
from typing import Any, Optional

from . import cad_features as _cad_features_module
from . import cad_mesh_repair as _cad_mesh_repair_module
from . import faceted_brep as _faceted_brep_module
from . import occ_shapes as _occ_shapes_module
from .cad_features import (
    _apply_passive_regions_to_step,
)
from .cad_mesh_repair import (
    _extract_recovered_mesh,
)
from .faceted_brep import _recovered_mesh_to_faceted_brep_solid
from .occ_shapes import (
    _assert_valid_occ_shape,
    _unify_same_domain_shape,
)

logger = logging.getLogger(__name__)

_RECOVERED_MODE = "Recovered Shape"
_COMPATIBILITY_MODULES = (
    _cad_mesh_repair_module,
    _occ_shapes_module,
    _cad_features_module,
    _faceted_brep_module,
)


def __getattr__(name: str) -> object:
    """Resolve helpers moved to focused reconstruction modules."""
    for module in _COMPATIBILITY_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include moved compatibility names in interactive discovery."""
    names = set(globals())
    for module in _COMPATIBILITY_MODULES:
        names.update(dir(module))
    return sorted(names)


def reconstruct_topopt_cad(
    payload: Any,
    *,
    source_geometry: str = _RECOVERED_MODE,
    sew_tolerance: float = 1e-4,
    relative_sew_tolerance: float = 1e-6,
    merge_angle_deg: float = 0.0,
    max_faces: Optional[int] = 1500,
    repair_mesh: bool = True,
    validate_watertight: bool = True,
    validate_brep: bool = True,
    min_component_faces: int = 8,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
    **_ignored_legacy_kwargs: Any,
) -> Any:
    """Build a CadQuery workplane from the topology result's Recovered Shape."""
    mode = str(source_geometry or _RECOVERED_MODE).strip()
    if mode and mode.lower() != _RECOVERED_MODE.lower():
        logger.info(
            "CAD reconstruction source_geometry=%r ignored; using %s.",
            mode,
            _RECOVERED_MODE,
        )

    recovered = _extract_recovered_mesh(payload)
    if recovered is None:
        raise RuntimeError(
            "Recovered Shape STEP export needs topology_result['recovered_shape'] "
            "with vertices and faces."
        )

    vertices, faces = recovered
    solid = _recovered_mesh_to_faceted_brep_solid(
        vertices,
        faces,
        sew_tolerance=float(sew_tolerance),
        relative_sew_tolerance=float(relative_sew_tolerance or 0.0),
        merge_angle_deg=float(merge_angle_deg or 0.0),
        max_faces=max_faces,
        repair_mesh=bool(repair_mesh),
        validate_watertight=bool(validate_watertight),
        validate_brep=bool(validate_brep),
        min_component_faces=int(min_component_faces or 0),
    )
    solid = _apply_passive_regions_to_step(
        solid,
        payload,
        void_axial_margin=float(void_axial_margin or 0.0),
        void_radial_margin=float(void_radial_margin or 0.0),
        validate_after_boolean=bool(validate_brep),
    )
    if merge_angle_deg and float(merge_angle_deg) > 0.0:
        solid = _unify_same_domain_shape(solid, merge_angle_deg=float(merge_angle_deg))
    if validate_brep:
        _assert_valid_occ_shape(solid, label="final Recovered Shape STEP body")

    import cadquery as cq

    return cq.Workplane(obj=solid)


__all__ = ["reconstruct_topopt_cad"]
