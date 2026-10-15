# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Content identities for mutable CAD geometry, excluding display meshes."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cadquery as cq


def cad_shape_fingerprint(shape: cq.Shape) -> str:
    """Hash current geometry, topology, placement, orientation, and tolerances.

    OCC's shape hash identifies its TShape and location, so an in-place pole
    or tolerance edit can leave that hash unchanged. Serialize on each request
    instead; excluding triangulation keeps viewer tessellation from changing
    the key or copying a potentially large display mesh into the stream.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepTools import BRepTools
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_FormatVersion_VERSION_1, TopTools_IndexedMapOfShape

    # Tessellation clears OCC's transient Checked flag even when geometry and
    # tolerances are unchanged. Normalize it on private topology; sharing the
    # read-only geometry avoids copying surfaces or touching the input owner.
    isolated = BRepBuilderAPI_Copy(shape.wrapped, False, False).Shape()
    parts = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(isolated, parts)
    for index in range(1, parts.Extent() + 1):
        parts.FindKey(index).Checked(False)
    with BytesIO() as stream:
        BRepTools.Write_s(isolated, stream, False, False,
                         TopTools_FormatVersion_VERSION_1)
        return sha256(stream.getbuffer()).hexdigest()
