# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Public orchestration for recovered topology mesh to STEP reconstruction."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from . import cad_features as _cad_features_module
from . import cad_mesh_repair as _cad_mesh_repair_module
from . import occ_shapes as _occ_shapes_module
from .cad_features import (
    _apply_passive_regions_to_step,
    _attach_passive_regions_to_lattice,
)
from .cad_mesh_repair import (
    _effective_tolerance,
    _extract_recovered_mesh,
    _mesh_repair_and_validate,
)
from .lattice_cad import (
    beam_lattice_from_member_plan,
    beam_lattice_from_structure,
    beam_lattice_solid,
    lattice_cad_strategy,
)
from .occ_shapes import (
    _assert_valid_occ_shape,
    _solid_volume,
    _unify_same_domain_shape,
)
from .draw_brep import _draw_axis, _draw_direction_brep
from .spline_brep import _extruded_spline_brep, _extrusion_axis_index

logger = logging.getLogger(__name__)

_RECOVERED_MODE = "Recovered Shape"

# Triangles above which per-triangle B-rep sewing stops being a reasonable
# thing to ask for. OCC's sewing cost is strongly superlinear in the face
# count: measured on this machine at 0.9 s for 2 000 triangles, 14 s for 8 000
# and 197 s for 20 000. A minimal-surface lattice recovers 150 000 to several
# million triangles even on a coarse grid, so the honest options are to stop
# here with a usable message or to spend hours producing a file no CAD system
# will open. Solid envelopes instead pass through compact profile or freeform
# spline reconstruction. A lattice does not have an equivalent compact boundary
# representation, which is exactly why it needs its own limit.
DEFAULT_MAXIMUM_SEWN_TRIANGLES = 24_000
_COMPATIBILITY_MODULES = (
    _cad_mesh_repair_module,
    _occ_shapes_module,
    _cad_features_module,
)


def _payload_pull_direction(payload: Any) -> Optional[str]:
    """Read the mould withdrawal direction a cast/moulded study was solved with."""
    if not isinstance(payload, dict):
        return None
    for key in ("pull_out_direction", "pull_out_axis", "draw_direction"):
        value = payload.get(key)
        if value:
            return str(value)
    manufacturing = payload.get("manufacturing")
    if isinstance(manufacturing, dict):
        constraints = manufacturing.get("constraints")
        if isinstance(constraints, dict):
            value = constraints.get("pull_out_direction")
            if value:
                return str(value)
    return None


def _payload_recovery_field(
    payload: Any,
) -> tuple[np.ndarray, float, Optional[tuple[np.ndarray, np.ndarray]]]:
    """Read the density field and level the recovered surface was built from."""
    if not isinstance(payload, dict):
        raise RuntimeError("Draw-direction CAD needs a topology result payload.")

    field = payload.get("recovery_density")
    level = payload.get("recovery_cutoff")
    if field is None:
        field = payload.get("manufactured_density")
        level = payload.get("density_cutoff")
    if field is None:
        field = payload.get("density")
        level = payload.get("density_cutoff")
    grid = np.asarray(field, dtype=float) if field is not None else np.zeros(0)
    if grid.ndim != 3 or min(grid.shape, default=0) < 2:
        raise RuntimeError(
            "Draw-direction CAD needs the three-dimensional recovery field."
        )
    recovered = payload.get("recovered_shape")
    if isinstance(recovered, dict) and recovered.get("effective_recovery_cutoff"):
        level = recovered.get("effective_recovery_cutoff")

    raw_bounds = payload.get("bounds")
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
    if isinstance(raw_bounds, dict):
        lower = raw_bounds.get("min")
        upper = raw_bounds.get("max")
        if lower is not None and upper is not None:
            bounds = (
                np.asarray(lower, dtype=float).reshape(-1)[:3],
                np.asarray(upper, dtype=float).reshape(-1)[:3],
            )
    elif isinstance(raw_bounds, (tuple, list)) and len(raw_bounds) == 2:
        bounds = (
            np.asarray(raw_bounds[0], dtype=float).reshape(-1)[:3],
            np.asarray(raw_bounds[1], dtype=float).reshape(-1)[:3],
        )
    return grid, float(level if level is not None else 0.5), bounds


class SmoothCadUnavailable(RuntimeError):
    """Raised when a result has no exact B-rep representation.

    Three topology results have one. An extrusion-constrained body is a profile
    swept along an axis. A cast or moulded body is a height field over the
    parting plane, because the pull-out constraint takes a running minimum of
    the density along the withdrawal direction and no column can turn solid
    again once it has turned to void. A strut lattice is already an exact set
    of sized members.

    An unrestricted, additive or symmetry-only result is none of these: its
    surface is a general two-manifold with no exact CAD form. Freeform
    subdivision fitting used to cover that case, and a general 3-D load path is
    not a smooth blob -- on a 3.6 m press crown it produced 3000 Bezier
    patches, no planar or cylindrical face at all, and 23 mm of surface
    deviation. Callers are expected to fall back to the recovered triangle
    surface, which is the honest geometry.
    """


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
    reconstruction_strategy: str = "auto",
    sew_tolerance: float = 1e-4,
    relative_sew_tolerance: float = 1e-6,
    fit_tolerance: float = 0.0,
    # The section profile is simplified by Ramer-Douglas-Peucker at this
    # fraction of the model diagonal before spline fitting, so it sets how much
    # material the CAD body loses against the recovered mesh it is fitted to.
    # The previous 2e-3 gave a 101-face body that was 1.38% light -- a mass
    # error large enough to matter in any downstream stress or cost check, and
    # invisible because the deviation gate (1.8 mm here) passes it comfortably.
    # Measured on the reference cantilever: 2e-3 -> 101 faces, -1.381%;
    # 1e-3 -> 229 faces, +0.090%; 5e-4 -> 304 faces, +0.033%; 1e-4 -> 902
    # faces, +0.015%. 1e-3 is the useful knee: it removes voxel-pitch contour
    # oscillations while keeping the prismatic body's area within one tenth of
    # one percent. The tighter 5e-4 setting merely made a spline trace the
    # staircase more faithfully.
    relative_fit_tolerance: float = 1e-3,
    # Voxel stair steps turn by 90 degrees. Treating them as CAD creases makes
    # a formally spline-based body look blocky, so only near-reversal corners
    # survive profile fairing; analytic passive features are restored later.
    crease_angle_deg: float = 135.0,
    maximum_volume_delta: float = 0.05,
    maximum_relative_deviation: float = 0.01,
    merge_angle_deg: float = 0.0,
    max_faces: Optional[int] = 1500,
    repair_mesh: bool = True,
    validate_watertight: bool = True,
    validate_brep: bool = True,
    min_component_faces: int = 0,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
    return_report: bool = False,
    structure_options: Any = None,
    member_plan: Any = None,
    maximum_sewn_triangles: int = DEFAULT_MAXIMUM_SEWN_TRIANGLES,
    **_ignored_legacy_kwargs: Any,
) -> Any:
    """Build a validated CAD body from the topology result.

    A strut lattice is reconstructed from its centrelines: the sized members
    are already an exact description of the body, so it becomes cylinders,
    cones and joint balls directly and never touches the isosurface. See
    :mod:`.lattice_cad`.

    Everything else needs a manufacturing constraint that makes the body
    exactly representable. An extrusion makes it a faired profile swept along
    one axis (:mod:`.spline_brep`); a pull-out direction makes it a height
    field over the parting plane (:mod:`.draw_brep`). Either way the result
    must pass topology, volume, sampled-deviation, above-cutoff point coverage
    and B-rep validity checks. A general 3-D result raises
    :class:`SmoothCadUnavailable`; the caller shows and exports its recovered
    surface instead. There is deliberately no faceted, freeform, or
    alternate-shape fallback: approximating a load path with smooth patches
    reported a CAD body that no longer described the optimized structure.
    """
    mode = str(source_geometry or _RECOVERED_MODE).strip()
    if mode and mode.lower() != _RECOVERED_MODE.lower():
        logger.info(
            "CAD reconstruction source_geometry=%r ignored; using %s.",
            mode,
            _RECOVERED_MODE,
        )

    # The payload carries what was built, so every caller — the solve, the
    # viewer preview, the STEP export worker — routes the same way without
    # each having to remember to pass it.
    if isinstance(payload, dict):
        if structure_options is None:
            structure_options = payload.get("structure_options")
        if member_plan is None:
            member_plan = payload.get("member_plan")

    strategy_kind = lattice_cad_strategy(structure_options)
    if strategy_kind == "beam":
        return _reconstruct_beam_lattice_cad(
            payload,
            structure_options,
            member_plan,
            void_axial_margin=float(void_axial_margin or 0.0),
            void_radial_margin=float(void_radial_margin or 0.0),
            validate_brep=bool(validate_brep),
            return_report=return_report,
        )

    recovered = _extract_recovered_mesh(payload)
    if recovered is None:
        raise RuntimeError(
            "Recovered Shape STEP export needs topology_result['recovered_shape'] "
            "with vertices and faces."
        )

    vertices, faces = recovered
    if strategy_kind == "isosurface" and len(faces) > int(maximum_sewn_triangles):
        display = getattr(structure_options, "display_name", "This lattice")
        raise RuntimeError(
            f"{display} recovered {len(faces):,} triangles. A minimal-surface "
            "or prismatic lattice has no compact B-rep, and OCC's per-triangle "
            "sewing cost grows far faster than the triangle count "
            f"(over {int(maximum_sewn_triangles):,} it runs into hours), so "
            "this body cannot be delivered as STEP. Export it as STL, which "
            "carries this geometry directly. Native beam-lattice 3MF is also "
            "available for strut families such as BCC and Octet. Automatic "
            "STEP reconstruction is disabled for every lattice family."
        )
    strategy = str(reconstruction_strategy or "auto").strip().lower()
    strategy = {
        "automatic": "auto",
        "recovered shape": "auto",
        "smooth": "spline",
        "smooth spline": "spline",
    }.get(strategy, strategy)
    if strategy in {"faceted", "faceted b-rep", "facet"}:
        raise ValueError(
            "Faceted CAD reconstruction has been removed. Solid topology must "
            "pass the validated smooth B-rep reconstruction."
        )
    if strategy in {"freeform", "subdivision", "subdivision spline"}:
        raise ValueError(
            "Freeform subdivision CAD reconstruction has been removed. An "
            "extrusion-constrained result uses its exact profile-spline B-rep; "
            "every other solid result is delivered as its recovered surface."
        )
    if strategy not in {"auto", "spline", "draw"}:
        raise ValueError(
            "reconstruction_strategy must be 'auto', 'spline', or 'draw'."
        )

    extrusion_axis = (
        payload.get("extrusion_axis") if isinstance(payload, dict) else None
    )
    has_extrusion = _extrusion_axis_index(extrusion_axis) is not None
    pull_direction = _payload_pull_direction(payload)
    has_draw = _draw_axis(pull_direction) is not None
    if strategy == "spline" and not has_extrusion:
        raise RuntimeError(
            "Profile-spline CAD requires an explicit X, Y, or Z extrusion."
        )
    if strategy == "draw" and not has_draw:
        raise RuntimeError(
            "Draw-direction CAD requires an explicit pull-out axis."
        )
    if not has_extrusion and not has_draw:
        # Checked before the mesh repair: there is nothing to repair the mesh
        # for, and the caller only needs to know it should keep the surface.
        raise SmoothCadUnavailable(
            "This result has no exact B-rep. A CAD body is produced for an "
            "extrusion-constrained topology, whose surface is a profile swept "
            "along one axis, for a cast or moulded topology, whose pull-out "
            "constraint makes it a height field over the parting plane, and "
            "for a strut lattice. Set the manufacturing process to Extruded or "
            "Cast / Moulded (or give an explicit extrusion or pull-out axis) "
            "to get an editable solid; otherwise use the recovered surface, "
            "which is exportable as STL."
        )

    if repair_mesh:
        smooth_vertices, smooth_faces = _mesh_repair_and_validate(
            vertices,
            faces,
            weld_tolerance=_effective_tolerance(
                vertices,
                float(sew_tolerance),
                float(relative_sew_tolerance or 0.0),
            ),
            min_component_faces=int(min_component_faces or 0),
            validate_watertight=bool(validate_watertight),
        )
    else:
        smooth_vertices, smooth_faces = vertices, faces

    recovered_payload = (
        payload.get("recovered_shape") if isinstance(payload, dict) else None
    )
    protected_profile_points = (
        recovered_payload.get("protected_profile_points")
        if isinstance(recovered_payload, dict)
        else None
    )
    if has_extrusion and strategy != "draw":
        # An extrusion is the tighter description of the two: a pull-out
        # constraint that happens to accompany it is already satisfied by the
        # swept profile, and the profile has the smaller control net.
        solid, report = _extruded_spline_brep(
            smooth_vertices,
            smooth_faces,
            extrusion_axis=extrusion_axis,
            absolute_fit_tolerance=float(fit_tolerance or 0.0),
            relative_fit_tolerance=float(relative_fit_tolerance or 0.0),
            crease_angle_deg=float(crease_angle_deg),
            maximum_volume_delta=float(maximum_volume_delta),
            maximum_relative_deviation=float(maximum_relative_deviation),
            protected_points=protected_profile_points,
        )
    else:
        field, level, bounds = _payload_recovery_field(payload)
        solid, report = _draw_direction_brep(
            smooth_vertices,
            smooth_faces,
            density=field,
            level=level,
            bounds=bounds,
            pull_direction=pull_direction,
            absolute_fit_tolerance=float(fit_tolerance or 0.0),
            relative_fit_tolerance=float(relative_fit_tolerance or 0.0),
            maximum_volume_delta=float(maximum_volume_delta),
            maximum_relative_deviation=float(maximum_relative_deviation),
            protected_points=protected_profile_points,
        )
    report["fallback_used"] = False

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

    report["cad_face_count_after_feature_healing"] = (
        int(len(solid.Faces())) if hasattr(solid, "Faces") else None
    )
    report["cad_volume"] = _solid_volume(solid)
    report["requested_strategy"] = strategy
    workplane = cq.Workplane(obj=solid)
    if return_report:
        return workplane, report
    return workplane


def _payload_bounds(payload: Any) -> tuple[Any, Any]:
    """Read the study bounds out of a topology output payload."""
    import numpy as np

    bounds = payload.get("bounds") if isinstance(payload, dict) else None
    if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
        mins = np.asarray(bounds["min"], dtype=float)
        maxs = np.asarray(bounds["max"], dtype=float)
        if mins.size >= 3 and maxs.size >= 3 and bool((maxs[:3] > mins[:3]).all()):
            return mins[:3], maxs[:3]
    raise RuntimeError(
        "An analytic lattice body needs the study bounds; the topology result "
        "does not carry them."
    )


def build_topopt_beam_lattice(
    payload: Any,
    structure_options: Any,
    member_plan: Any = None,
) -> Any:
    """Return the :class:`~.lattice_cad.BeamLattice` for a solved strut study.

    Prefers the sized member plan, because those radii were set against stress
    and buckling; falls back to de-homogenizing the optimizer's density field
    through the same map the voxel rasterizer uses, so the two agree.
    """
    import numpy as np

    bounds = _payload_bounds(payload)
    if member_plan is not None:
        return beam_lattice_from_member_plan(member_plan, bounds)

    density = payload.get("density") if isinstance(payload, dict) else None
    if density is None:
        raise RuntimeError(
            "An analytic lattice body needs the optimized density field."
        )
    cutoff = float(payload.get("density_cutoff") or 0.5)
    return beam_lattice_from_structure(
        np.asarray(density, dtype=float),
        cutoff,
        structure_options,
        bounds,
    )


def _reconstruct_beam_lattice_cad(
    payload: Any,
    structure_options: Any,
    member_plan: Any,
    *,
    void_axial_margin: float,
    void_radial_margin: float,
    validate_brep: bool,
    return_report: bool,
) -> Any:
    """Build the CAD body of a strut lattice from its centrelines."""
    import numpy as np

    lattice = build_topopt_beam_lattice(payload, structure_options, member_plan)
    solid, report = beam_lattice_solid(lattice, return_report=True)
    report["beam_lattice"] = lattice.diagnostics()
    report["requested_strategy"] = "beam lattice"

    # An analytic member reaches its full radius past the outermost node, where
    # the voxel field was truncated at the envelope instead. State the overhang
    # so a fit check against neighbouring parts is made on the real number.
    try:
        study_min, study_max = _payload_bounds(payload)
        lower, upper = lattice.solid_bounds()
        overrun = float(
            np.max(np.concatenate([study_min - lower, upper - study_max]))
        )
        report["bounds_overrun"] = max(0.0, overrun)
        if overrun > 0.0:
            report["bounds_overrun_note"] = (
                f"Members reach {overrun:.3g} beyond the study bounding box, "
                "because a member centred on the outermost node carries its "
                "full radius outward while the voxel field was cut off at the "
                "envelope. Check the fit against neighbouring parts, and trim "
                "in CAD if the envelope is hard."
            )
    except Exception:
        logger.debug("Could not measure the lattice bounds overrun.", exc_info=True)
    report["guidance"] = report.get("guidance", "") or (
        "Exported as exact analytic solids. The 3MF beam lattice export is "
        "smaller still and is what the AM tool chain reads natively."
    )

    shape = solid.val() if hasattr(solid, "val") else solid
    shape, passive_counts = _attach_passive_regions_to_lattice(
        shape,
        payload,
        void_axial_margin=void_axial_margin,
        void_radial_margin=void_radial_margin,
    )
    report["passive_features"] = passive_counts
    if validate_brep:
        # A compound of overlapping members is expected here and is not a
        # defect, so only the individual solids are worth asserting on.
        _assert_valid_occ_shape(shape, label="analytic lattice STEP body")
    report["cad_volume"] = _solid_volume(shape)

    import cadquery as cq

    workplane = cq.Workplane(obj=shape)
    if return_report:
        return workplane, report
    return workplane


__all__ = [
    "SmoothCadUnavailable",
    "build_topopt_beam_lattice",
    "reconstruct_topopt_cad",
]
