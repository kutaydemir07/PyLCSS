# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Keep CAD boundary conditions in place when a cubic voxel box is padded."""

from copy import deepcopy

import numpy as np


def rebase_voxel_boundary_conditions(bc, old_bounds, new_bounds):
    """Change fractional coordinates without moving physical selections/forces."""
    result = deepcopy(bc)
    old_min, old_max = (np.asarray(value, dtype=float) for value in old_bounds)
    new_min, new_max = (np.asarray(value, dtype=float) for value in new_bounds)
    scale = (old_max - old_min) / (new_max - new_min)
    offset = (old_min - new_min) / (new_max - new_min)

    def point(values):
        return tuple((np.asarray(values[:3]) * scale + offset).tolist()) + tuple(values[3:])

    def box(values):
        coords = np.asarray(values[:6], dtype=float).reshape(3, 2)
        coords = coords * scale[:, None] + offset[:, None]
        return tuple(coords.ravel().tolist()) + tuple(values[6:])

    def cylinder(values):
        axis = {"x": 0, "y": 1, "z": 2}[str(values[0]).lower()]
        a, b = [index for index in range(3) if index != axis]
        radius_b = values[6] if len(values) > 6 else values[5]
        return (values[0], values[1] * scale[a] + offset[a], values[2] * scale[b] + offset[b],
                values[3] * scale[axis] + offset[axis], values[4] * scale[axis] + offset[axis],
                values[5] * scale[a], radius_b * scale[b])

    def rebase(item):
        for name in ("fixed_boxes", "box_forces", "solid_boxes", "void_boxes"):
            if hasattr(item, name):
                setattr(item, name, [box(values) for values in getattr(item, name)])
        item.point_forces = [point(values) for values in item.point_forces]
        for name in ("boundary_supports", "boundary_forces"):
            for boundary in getattr(item, name):
                boundary.vertices = np.asarray(boundary.vertices) * scale + offset
        for name in ("solid_cylinders", "void_cylinders", "joint_pin_cylinders"):
            if hasattr(item, name):
                setattr(item, name, [cylinder(values) for values in getattr(item, name)])
        for joint in item.joints:
            joint.anchor_a = point(joint.anchor_a)
            joint.anchor_b = point(joint.anchor_b)

    rebase(result)
    for case in result.load_cases:
        rebase(case)
    return result
