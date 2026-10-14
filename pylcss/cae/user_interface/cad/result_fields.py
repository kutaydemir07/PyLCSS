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
        source = data.get("topology_source") or data
        generated = bool(data.get("generated_lattice"))
        if has_field(source, "density"):
            fields.append("Density")
        recovered = source.get("recovered_shape")
        if isinstance(recovered, dict) and has_field(recovered, "faces"):
            fields.append("STL Shape")
        if any(body.get(key) is not None for body in (source, data) for key in ("cad_shape", "shape")):
            fields.append("CAD")
        analyses = [("FEA ", data)]
        if generated:
            analyses = [("FEA Solid ", source), ("FEA Lattice ", data)]
        for prefix, body in analyses:
            validation = body.get("validation")
            if isinstance(validation, dict) and validation.get("type") in {"fea", "external_solver"}:
                fields.extend(prefix + field for field in available_result_fields(validation)
                              if field != "Default View")
        if (generated or data.get("study_kind") == "lattice_infill") and isinstance(data.get("recovered_shape"), dict) and has_field(data["recovered_shape"], "faces"):
            fields.append("Lattice")
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


def topopt_view_mode(data, mode):
    """Map saved view names onto available, already computed results."""
    mesh_mode = "Lattice" if data.get("generated_lattice") or data.get("study_kind") == "lattice_infill" else "STL Shape"
    mode = {
        "Surface": mesh_mode, "Recovered Shape": mesh_mode,
        "Recovered Surface (Mesh)": mesh_mode, "Manufactured Mesh": mesh_mode,
        "Reconstructed CAD (B-rep)": "CAD", "Voxel Density": "Density",
    }.get(str(mode), str(mode))
    if mode == "CAD" and "CAD" not in available_result_fields(data):
        return mesh_mode
    if data.get("generated_lattice") and mode.startswith("FEA ") and not mode.startswith(("FEA Solid ", "FEA Lattice ")):
        prefix = "FEA Lattice " if data.get("validation") else "FEA Solid "
        return prefix + mode[len("FEA "):]
    return mode


def topopt_view_result(data, mode):
    """Keep the original solve viewable after generating a lattice."""
    source = data.get("topology_source") or data
    if mode in {"Density", "STL Shape"} or str(mode).startswith("FEA Solid "):
        return source
    if mode == "CAD" and data.get("cad_shape") is None and data.get("shape") is None:
        return source
    return data


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
