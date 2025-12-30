# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Export helpers for shapes and assemblies.
"""
import os
from typing import Any

try:
    import cadquery as cq
    from cadquery import exporters
except ImportError:
    cq = None
    exporters = None


def ensure_out(path: str) -> None:
    """Ensure the output directory exists."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def export_shape(shape: Any, path: str) -> None:
    """
    Export a CadQuery object to a file (STEP, STL, GLTF, etc.).

    Supports:
      - Workplane (exports content)
      - Shape/Solid
      - Assembly (preserves colors & hierarchy for STEP/GLTF)
      - SimpleAssembly (automatically unwraps)
    """
    if exporters is None:
        raise RuntimeError("CadQuery exporters not available. Install cadquery.")

    ensure_out(path)

    candidate = shape

    # 1. Unwrap your SimpleAssembly class if passed directly
    if hasattr(candidate, "assembly"):
        candidate = candidate.assembly

    # 2. Unwrap generic wrappers (like if you have a custom Part class)
    # We check for 'val' (Workplane) but NOT 'toCompound' (Assembly)
    # to avoid accidentally flattening Assemblies.
    if hasattr(candidate, "val") and not isinstance(candidate, (cq.Assembly, cq.Workplane)):
         try:
             candidate = candidate.val()
         except Exception:
             pass

    # 3. Export
    # CadQuery's exporters.export() handles Workplanes and Assemblies natively.
    # Passing an Assembly object here (instead of a compound) preserves colors/names in STEP.
    exporters.export(candidate, path)