# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Density-field to manufacturing-surface recovery orchestration."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from ..manufacturing.structures import (
    ManufacturingStructureOptions,
    build_manufacturing_field,
)
from .analytic_shapes import (
    _convert_legacy_to_physical_shapes,
    _project_passive_shapes_surfaces,
)
from .mesh_postprocess import _enhanced_mesh_postprocess, _taubin_smooth_surface
from .passive_regions import (
    _apply_passive_cylinder_sdf,
    _apply_passive_density_regions,
    _resample_source_mask,
)
from .print_surface import (
    extract_flying_edges_surface,
    volume_preserving_level_field,
)
from .recovery_grid import (
    BoxRegion,
    CylinderRegion,
    _project_extruded_planes,
    _regularize_extruded_density,
    _voxel_origin_cell,
)
from . import analytic_shapes as _analytic_shapes
from . import mesh_postprocess as _mesh_postprocess_module
from . import passive_regions as _passive_regions
from . import recovery_grid as _recovery_grid

logger = logging.getLogger(__name__)

_HELPER_MODULES = (
    _recovery_grid,
    _analytic_shapes,
    _mesh_postprocess_module,
    _passive_regions,
)


def __getattr__(name: str) -> object:
    """Resolve helpers moved out of this formerly monolithic module."""
    for module in _HELPER_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    helper_names = {
        name
        for module in _HELPER_MODULES
        for name in dir(module)
        if name.startswith("_") or not name.startswith("__")
    }
    return sorted(set(globals()) | helper_names)


def _recover_voxel_shape(
    density: np.ndarray,
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    cutoff: float,
    print_ready: bool = False,
    decimate_ratio: float = 1.0,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    joint_pin_cylinders: Sequence[CylinderRegion] = (),
    extrusion_axis: str = "none",
    source_mask: Optional[np.ndarray] = None,
    passive_solid_mask: Optional[np.ndarray] = None,
    passive_void_mask: Optional[np.ndarray] = None,
    blend_radius: float = 0.0,
    structure_options: Optional[ManufacturingStructureOptions] = None,
    surface_backend: str = "vtk_sdf",
) -> Optional[dict[str, np.ndarray]]:
    """Extract a recovered surface from a structured voxel density field.

    With ``print_ready=True``, the preferred path uses a volume-preserving
    signed-distance field and VTK Flying Edges, followed by mesh repair and
    optional decimation. ``surface_backend="legacy"`` retains marching cubes.

    ``structure_options`` can replace the solid envelope with a topology-
    following rib network or TPMS lattice. This is manufacturing geometry and
    must be verified independently from the continuous SIMP result.
    """
    try:
        from skimage import measure
        import scipy.ndimage as ndi
    except ImportError:
        return None

    try:
        grid = np.asarray(density, dtype=float)
        if grid.ndim != 3 or min(grid.shape) < 1:
            return None

        grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(grid)) <= 0.0:
            return None
        grid = _regularize_extruded_density(grid, extrusion_axis)

        axis_map = {"x": 0, "y": 1, "z": 2}
        extrusion_ax = axis_map.get(str(extrusion_axis or "").strip().lower())
        if extrusion_ax is not None:
            in_plane = [i for i in range(3) if i != extrusion_ax]
            min_plane_dim = max(1, min(int(grid.shape[i]) for i in in_plane))
            upsample = int(np.clip(np.ceil(240.0 / min_plane_dim), 1, 8))
            zoom_factors = np.ones(3, dtype=float)
            zoom_factors[in_plane] = float(upsample)
        else:
            min_dim = max(1, min(grid.shape))
            # Cubic interpolation beyond roughly 36 samples on the thinnest
            # axis added tens of thousands of nearly coplanar triangles but no
            # recoverable design information. Print-ready smoothing and
            # quadric decimation provide the useful finish at much lower cost.
            upsample = int(np.clip(np.ceil(36.0 / min_dim), 1, 10))
            zoom_factors = np.full(3, float(upsample), dtype=float)

        while (
            np.any(zoom_factors > 1.0)
            and np.prod(np.asarray(grid.shape) * zoom_factors) > 2_500_000
        ):
            largest = int(np.argmax(zoom_factors))
            zoom_factors[largest] = max(1.0, zoom_factors[largest] - 1.0)

        if np.any(zoom_factors > 1.0):
            field = ndi.zoom(
                grid,
                zoom=tuple(float(v) for v in zoom_factors),
                order=3,
                mode="nearest",
                grid_mode=True,
            )
        else:
            field = grid.copy()

        origin, cell = _voxel_origin_cell(tuple(grid.shape), bounds)
        spacing = cell / zoom_factors

        field = np.clip(field, 0.0, 1.0)
        source_field = _resample_source_mask(source_mask, tuple(field.shape))
        explicit_solid_field = _resample_source_mask(
            passive_solid_mask, tuple(field.shape)
        )
        explicit_void_field = _resample_source_mask(
            passive_void_mask, tuple(field.shape)
        )
        sigma = 0.20 if float(np.max(zoom_factors)) <= 1.0 else 0.35
        field = ndi.gaussian_filter(field, sigma=sigma)
        if source_field is not None:
            field[~source_field] = 0.0
        if explicit_solid_field is not None:
            field[explicit_solid_field] = 1.0
        if explicit_void_field is not None:
            field[explicit_void_field] = 0.0
        # Source masks are voxel approximations of the input body; passive keep
        # and cut regions are analytic constraints and must be re-imposed after
        # that coarse clipping so round holes/bosses stay round in recovery.
        field = _apply_passive_density_regions(
            field,
            solid_boxes=solid_boxes,
            void_boxes=void_boxes,
            solid_cylinders=solid_cylinders,
            void_cylinders=void_cylinders,
            bounds=bounds,
            cell=cell,
            spacing=spacing,
        )
        # Joint pins are separate assembly hardware, not passive optimizer
        # material. Add them only to the manufactured/recovered field and only
        # after through-bores have been re-imposed.
        field = _apply_passive_density_regions(
            field,
            solid_cylinders=joint_pin_cylinders,
            bounds=bounds,
            cell=cell,
            spacing=spacing,
        )
        if structure_options is not None and structure_options.mode != "solid":
            field = build_manufacturing_field(
                field,
                cutoff,
                structure_options,
                resolution_scale=float(np.mean(zoom_factors)),
                passive_solid_mask=explicit_solid_field,
                passive_void_mask=explicit_void_field,
            )
            # Contact pads, bores, and other passive engineering features have
            # priority over the generated infill representation.
            field = _apply_passive_density_regions(
                field,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
            )
            field = _apply_passive_density_regions(
                field,
                solid_cylinders=joint_pin_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
            )
        pad = max(3, min(10, int(np.ceil(float(np.max(zoom_factors))))))
        field = np.pad(field, pad_width=pad, mode="constant", constant_values=0.0)

        level = float(np.clip(cutoff, 1e-6, 0.999999))
        mask = field >= level
        if not np.any(mask):
            nonzero = field[field > 0.0]
            if nonzero.size == 0:
                return None
            mask = field >= float(np.percentile(nonzero, 75.0))
        if np.all(mask):
            return None

        # Build a signed iso-field directly from the filtered physical density:
        # negative is material, positive is void.  Passive shapes are applied
        # before marching cubes so circular holes do not need post-hoc vertex
        # snapping, which can fold triangles near tight bolt holes.
        iso_field = level - field
        mc_level = 0.0
        passive_shapes_present = bool(
            solid_cylinders
            or void_cylinders
            or solid_boxes
            or void_boxes
            or joint_pin_cylinders
        )
        analytic_cylinder_iso = False
        if passive_shapes_present:
            iso_field = _apply_passive_cylinder_sdf(
                iso_field,
                pad=pad,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
                blend_radius=blend_radius,
            )
            iso_field = _apply_passive_cylinder_sdf(
                iso_field,
                pad=pad,
                solid_cylinders=joint_pin_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
                blend_radius=blend_radius,
            )
            analytic_cylinder_iso = True

        # Prefer the filtered physical density field itself for the visible
        # boundary.  The previous binary-mask -> distance-transform SDF route
        # made thin topology webs look like terraced contour plots because the
        # signed-distance bands were smoothed after thresholding.
        if not (float(np.min(iso_field)) < mc_level < float(np.max(iso_field))):
            outside = ndi.distance_transform_edt(
                ~mask, sampling=tuple(float(v) for v in spacing)
            )
            inside = ndi.distance_transform_edt(
                mask, sampling=tuple(float(v) for v in spacing)
            )
            iso_field = outside - inside
            iso_field = ndi.gaussian_filter(iso_field, sigma=0.35)
            if passive_shapes_present:
                iso_field = _apply_passive_cylinder_sdf(
                    iso_field,
                    pad=pad,
                    solid_boxes=solid_boxes,
                    void_boxes=void_boxes,
                    solid_cylinders=solid_cylinders,
                    void_cylinders=void_cylinders,
                    bounds=bounds,
                    cell=cell,
                    spacing=spacing,
                    blend_radius=blend_radius,
                )
                iso_field = _apply_passive_cylinder_sdf(
                    iso_field,
                    pad=pad,
                    solid_cylinders=joint_pin_cylinders,
                    bounds=bounds,
                    cell=cell,
                    spacing=spacing,
                    blend_radius=blend_radius,
                )
                analytic_cylinder_iso = True
            mc_level = 0.0
            if not (float(np.min(iso_field)) < mc_level < float(np.max(iso_field))):
                return None

        surface_origin = origin + 0.5 * spacing - float(pad) * spacing
        backend_name = "scikit-image marching cubes"
        vtk_surface = None
        use_vtk_sdf = str(surface_backend or "").strip().lower() not in {
            "legacy",
            "marching_cubes",
            "skimage",
        }
        if use_vtk_sdf:
            material_mask = iso_field <= float(mc_level)
            try:
                print_sdf, _ = volume_preserving_level_field(
                    iso_field,
                    material_mask,
                )
                if passive_shapes_present:
                    print_sdf = _apply_passive_cylinder_sdf(
                        print_sdf,
                        pad=pad,
                        solid_boxes=solid_boxes,
                        void_boxes=void_boxes,
                        solid_cylinders=solid_cylinders,
                        void_cylinders=void_cylinders,
                        bounds=bounds,
                        cell=cell,
                        spacing=spacing,
                        blend_radius=blend_radius,
                    )
                    print_sdf = _apply_passive_cylinder_sdf(
                        print_sdf,
                        pad=pad,
                        solid_cylinders=joint_pin_cylinders,
                        bounds=bounds,
                        cell=cell,
                        spacing=spacing,
                        blend_radius=blend_radius,
                    )
                vtk_surface = extract_flying_edges_surface(
                    print_sdf,
                    spacing,
                    surface_origin,
                )
            except Exception:
                logger.exception(
                    "Signed-distance print recovery failed; using marching cubes"
                )

        if vtk_surface is not None:
            verts = np.asarray(vtk_surface["vertices"], dtype=float)
            faces = np.asarray(vtk_surface["faces"], dtype=int)
            backend_name = str(vtk_surface["surface_backend"])
        else:
            verts, faces, _, _ = measure.marching_cubes(
                iso_field,
                level=float(mc_level),
                spacing=tuple(float(v) for v in spacing),
                gradient_direction="ascent",
            )
            verts = verts + surface_origin
        if len(verts) == 0 or len(faces) == 0:
            return None

        active_shapes = ()
        if passive_shapes_present:
            active_shapes = _convert_legacy_to_physical_shapes(
                bounds,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
                # The pin is a separate solid inside the bore. Including it in
                # the projection set keeps its analytic circular surface
                # protected if a non-SDF fallback is used.
                # It is deliberately appended after the bore definition.
                # `_convert_legacy_to_physical_shapes` is order independent for
                # projection, while scalar composition above enforces order.
            )
            active_shapes = tuple(active_shapes) + tuple(
                _convert_legacy_to_physical_shapes(
                    bounds,
                    solid_cylinders=joint_pin_cylinders,
                )
            )
        if vtk_surface is None:
            verts = _taubin_smooth_surface(
                verts,
                faces,
                iterations=2,
                shapes=active_shapes,
                tolerance=float(np.max(spacing)) * 3.0,
            )
        verts = _project_extruded_planes(
            verts,
            bounds,
            extrusion_axis,
            tolerance=float(np.max(spacing)) * 2.5,
        )
        if passive_shapes_present and not analytic_cylinder_iso:
            shapes = _convert_legacy_to_physical_shapes(
                bounds,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
            )
            shapes = tuple(shapes) + tuple(
                _convert_legacy_to_physical_shapes(
                    bounds,
                    solid_cylinders=joint_pin_cylinders,
                )
            )
            verts = _project_passive_shapes_surfaces(
                verts,
                shapes,
                tolerance=float(np.max(spacing)) * 3.0,
            )
        if bounds is not None:
            bound_min = np.asarray(bounds[0], dtype=float)[:3]
            bound_max = np.asarray(bounds[1], dtype=float)[:3]
            verts = np.clip(verts, bound_min, bound_max)

        # Print-ready: trimesh pipeline (hole-fill, Humphrey, optional decimate).
        if print_ready:
            improved = _enhanced_mesh_postprocess(
                verts,
                faces,
                decimate_ratio=float(decimate_ratio),
                smoothing_iterations=0 if vtk_surface is not None else 2,
            )
            if improved is not None and len(improved.get("faces", [])) > 0:
                improved["vertices"] = _project_extruded_planes(
                    improved["vertices"],
                    bounds,
                    extrusion_axis,
                    tolerance=float(np.max(spacing)) * 2.5,
                )
                if passive_shapes_present and not analytic_cylinder_iso:
                    shapes = _convert_legacy_to_physical_shapes(
                        bounds,
                        solid_boxes=solid_boxes,
                        void_boxes=void_boxes,
                        solid_cylinders=solid_cylinders,
                        void_cylinders=void_cylinders,
                    )
                    shapes = tuple(shapes) + tuple(
                        _convert_legacy_to_physical_shapes(
                            bounds,
                            solid_cylinders=joint_pin_cylinders,
                        )
                    )
                    improved["vertices"] = _project_passive_shapes_surfaces(
                        improved["vertices"],
                        shapes,
                        tolerance=float(np.max(spacing)) * 3.0,
                    )
                if bounds is not None:
                    improved["vertices"] = np.clip(
                        improved["vertices"], bound_min, bound_max
                    )
                improved["surface_backend"] = backend_name
                improved["manufacturing_structure"] = (
                    structure_options.display_name
                    if structure_options is not None
                    else "Solid Envelope"
                )
                return improved

        return {
            "vertices": np.asarray(verts, dtype=float),
            "faces": np.asarray(faces, dtype=int),
            "surface_backend": backend_name,
            "manufacturing_structure": (
                structure_options.display_name
                if structure_options is not None
                else "Solid Envelope"
            ),
        }
    except Exception:
        logger.exception("Voxel shape recovery failed")
        return None
