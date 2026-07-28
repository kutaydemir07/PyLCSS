# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""OpenCASCADE shell, solid, validation, and unification helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _shape_to_shells(shape: Any) -> list[Any]:
    """Extract TopoDS shells from an OCC shape or compound."""
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shells: list[Any] = []
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        try:
            shells.append(TopoDS.Shell_s(exp.Current()))
        except Exception:
            pass
        exp.Next()
    return shells


def _shell_to_solid(shell: Any) -> Any:
    """Fix a sewn shell and close it into an OCC solid."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid

    try:
        shell_fixer = ShapeFix_Shell(shell)
        shell_fixer.Perform()
        fixed_shell = shell_fixer.Shell()
    except Exception:
        fixed_shell = shell

    builder = BRepBuilderAPI_MakeSolid()
    builder.Add(fixed_shell)
    if not builder.IsDone():
        raise RuntimeError("Sewed recovered mesh did not close into a solid.")

    solid = builder.Solid()
    try:
        solid_fixer = ShapeFix_Solid(solid)
        solid_fixer.Perform()
        fixed_solid = solid_fixer.Solid()
        if fixed_solid is not None:
            solid = fixed_solid
    except Exception:
        logger.debug("ShapeFix_Solid failed for recovered mesh; using raw solid")
    return solid


def _assert_valid_occ_shape(shape: Any, *, label: str = "STEP body") -> None:
    """Raise a useful error if OpenCASCADE says the B-rep is invalid."""
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer

        occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape
        analyzer = BRepCheck_Analyzer(occ_shape)
        if not analyzer.IsValid():
            raise RuntimeError(f"{label} is not a valid OpenCASCADE B-rep.")
    except RuntimeError:
        raise
    except Exception:
        logger.debug("BRepCheck_Analyzer unavailable or failed; skipped validation")


def _unify_same_domain_shape(shape: Any, merge_angle_deg: float = 0.0) -> Any:
    """Merge same-domain faces after sewing and analytic feature booleans."""
    if not merge_angle_deg or float(merge_angle_deg) <= 0.0:
        return shape
    try:
        import cadquery as cq
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

        occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape
        up = ShapeUpgrade_UnifySameDomain(occ_shape, True, True, True)
        try:
            up.SetLinearTolerance(1e-4)
            up.SetAngularTolerance(float(np.radians(merge_angle_deg)))
        except Exception:
            pass
        up.Build()
        merged = cq.Shape.cast(up.Shape())
        if merged is not None and merged.isValid():
            return merged
    except Exception:
        logger.debug("ShapeUpgrade_UnifySameDomain failed; keeping recovered B-rep")
    return shape


def _solid_volume(shape: Any) -> Optional[float]:
    try:
        return float(shape.Volume())
    except Exception:
        return None


def _single_solid_if_possible(shape: Any) -> Any:
    """Return the only child solid from a compound when OCC keeps a wrapper."""
    try:
        solids = shape.Solids()
    except Exception:
        return shape
    if solids and len(solids) == 1:
        return solids[0]
    return shape


def _shells_to_cq_shape(shells: list[Any]) -> Any:
    """Convert one or more sewn shells into a CadQuery Solid or Compound."""
    import cadquery as cq

    solids = []
    errors = []
    for shell in shells:
        try:
            solid = cq.Solid(_shell_to_solid(shell))
            volume = _solid_volume(solid)
            if solid.isValid() and (volume is None or abs(volume) > 1e-12):
                solids.append(solid)
        except Exception as exc:
            errors.append(str(exc))

    if not solids:
        detail = f" ({'; '.join(errors[:3])})" if errors else ""
        raise RuntimeError(f"Recovered mesh sewing produced no valid solid{detail}.")
    if len(solids) == 1:
        return solids[0]
    return cq.Compound.makeCompound(solids)


# ---------------------------------------------------------------------------
# Passive feature preservation for STEP
# ---------------------------------------------------------------------------


def _compound_or_single(tools: list[Any]) -> Any:
    """Combine multiple OCC tools into a single compound for batch boolean."""
    import cadquery as cq

    tools = [t for t in tools if t is not None]
    if not tools:
        return None
    if len(tools) == 1:
        return tools[0]
    return cq.Compound.makeCompound(tools)
