# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Parametric parts implemented with CadQuery.

Each function returns a CadQuery Workplane or solid that can be used
directly or combined into assemblies.
"""
from typing import Tuple

try:
    import cadquery as cq
except ImportError:
    cq = None


def plate(width: float = 100.0, length: float = 60.0, thickness: float = 5.0,
          hole_diameter: float = 5.0, hole_margin: float = 10.0) -> "cq.Workplane":
    """
    Create a rectangular plate with four corner holes.
    Units: millimeters
    """
    if cq is None:
        raise RuntimeError("CadQuery not available. Install cadquery to use parts.")

    w = width
    l = length
    t = thickness
    hd = hole_diameter
    hm = hole_margin

    # Create the main block
    wp = (cq.Workplane("XY")
          .box(w, l, t)
          .faces(">Z")
          .workplane(centerOption="CenterOfBoundBox")
          )

    # Calculate hole positions relative to center
    hole_x = (w / 2.0) - hm
    hole_y = (l / 2.0) - hm

    # Cut holes in all 4 corners
    for sx in (-1, 1):
        for sy in (-1, 1):
            wp = wp.pushPoints([(sx * hole_x, sy * hole_y)])
            wp = wp.circle(hd / 2.0).cutThruAll()

    return wp


def peg(radius: float = 3.0, height: float = 12.0) -> "cq.Workplane":
    """
    A simple cylindrical peg.
    """
    if cq is None:
        raise RuntimeError("CadQuery not available. Install cadquery to use parts.")

    return cq.Workplane("XY").circle(radius).extrude(height)


def bracket(width: float = 40.0, height: float = 40.0, thickness: float = 3.0,
            cutout: Tuple[float, float] = (20.0, 10.0)) -> "cq.Workplane":
    """
    L-shaped bracket with a rectangular cutout for weight reduction.
    cutout: (width, height) of the internal cutout.
    """
    if cq is None:
        raise RuntimeError("CadQuery not available. Install cadquery to use parts.")

    w = width
    h = height
    t = thickness
    cu_w, cu_h = cutout

    # 1. Build the "L" shape by fusing two boxes
    # Base part (horizontal)
    base = cq.Workplane("XY").box(w, t, t)
    
    # Upright part (vertical) - shifted to sit on top/side correctly
    # We calculate offset to place the vertical bar correctly relative to origin
    upright = (cq.Workplane("XY")
               .transformed(offset=(0, (h - t) / 2.0, (t) / 2.0))
               .box(t, h - t, t))

    s = base.union(upright)

    # 2. Cut the rectangular hole
    # We select the bottom face (<Z is usually bottom, but depends on orientation logic)
    # Here we just target the main face to cut through.
    s = s.faces("<Z").workplane(centerOption="CenterOfBoundBox")
    
    # Position the cut
    s = s.pushPoints([( (w - cu_w) / 2.0, (h - cu_h) / 2.0 )])
    s = s.rect(cu_w, cu_h).cutThruAll()

    return s