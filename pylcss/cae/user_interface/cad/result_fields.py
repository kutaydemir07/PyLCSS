# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Shared display fields supported by cached solver results."""

import numpy as np


def has_field(data, name):
    value = data.get(name)
    return value is not None and np.asarray(value).size > 0


def available_result_fields(data):
    if not isinstance(data, dict):
        return ["Default View"]
    kind = data.get("type")
    fields = []
    if kind in {"fea", "external_solver"}:
        if has_field(data, "stress"):
            fields.append("Von Mises Stress")
        if has_field(data, "displacement"):
            fields.append("Displacement")
        if has_field(data, "safety_factor") or (
            has_field(data, "stress") and float(data.get("yield_strength") or 0) > 0
        ):
            fields.append("Safety Factor")
        tensor = np.asarray(data.get("stress_tensor", []))
        if tensor.ndim == 2 and tensor.shape[0] and tensor.shape[1] >= 6:
            fields.extend([
                "Maximum Principal Stress", "Minimum Principal Stress",
                "Stress XX", "Stress YY", "Stress ZZ", "Stress XY", "Stress YZ", "Stress ZX",
            ])
    elif kind == "topopt_voxel":
        if any(data.get(key) is not None for key in ("cad_shape", "shape", "recovered_shape")):
            fields.append("CAD")
        if has_field(data, "density"):
            fields.append("Density")
        validation = data.get("validation")
        if isinstance(validation, dict):
            if validation.get("type") in {"fea", "external_solver"}:
                fields.extend("FEA " + field for field in available_result_fields(validation)
                              if field != "Default View")
    elif kind == "modal":
        shapes = data.get("mode_displacements_flat") or {}
        for mode in data.get("modes") or []:
            number = int(mode["mode"])
            if number in shapes or str(number) in shapes:
                fields.append(f"Mode {number} ({float(mode['frequency_hz']):.1f} Hz)")
    elif kind == "crash":
        frames = data.get("frames") or []
        if frames:
            fields.append("Deformed Animation")
            for key, label in (("stress_vm", "Von Mises Stress"), ("displacement", "Displacement"), ("eps_p", "Plastic Strain"), ("failed", "Failed Elements")):
                if any(has_field(frame, key) for frame in frames):
                    fields.append(label)
    return fields or ["Default View"]


def activate_modal_mode(data, field):
    """Update the selected eigenvector and its visual scale without a solve."""
    if data.get("type") != "modal" or not str(field).startswith("Mode "):
        return
    number = int(str(field).split()[1])
    shapes = data.get("mode_displacements_flat") or {}
    displacement = shapes.get(number, shapes.get(str(number)))
    if displacement is None:
        return
    displacement = np.asarray(displacement, dtype=float).reshape(-1)
    peak = float(np.max(np.linalg.norm(displacement.reshape(-1, 3), axis=1)))
    mesh = data.get("mesh")
    length = float(np.max(np.ptp(mesh.p, axis=1))) if mesh is not None else 1.0
    auto_scale = 0.05 * length / peak if peak > 0 else 1.0
    was_auto = np.isclose(data.get("deformation_scale", 1.0), data.get("auto_deformation_scale", 1.0))
    data.update(displacement=displacement, active_mode=number, peak_displacement=peak, auto_deformation_scale=auto_scale)
    if was_auto:
        data["deformation_scale"] = auto_scale
    for mode in data.get("modes") or []:
        if int(mode["mode"]) == number:
            data["frequency_hz"] = mode["frequency_hz"]
            break
