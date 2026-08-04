# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Print-oriented continuous-field surface extraction.

The preferred pipeline retains the interpolated optimizer field, smooths its
implicit boundary, recalibrates the zero level to retain material volume, and
extracts it with VTK Flying Edges. A binary-mask SDF helper remains available
for callers that do not have a continuous field.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _closed_mesh_volume(surface: dict) -> Optional[float]:
    """Return the volume a extracted surface encloses, or None if it is open."""
    try:
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray(surface["vertices"], dtype=float),
            faces=np.asarray(surface["faces"], dtype=np.int64),
            process=False,
        )
        if not mesh.is_watertight:
            return None
        volume = float(abs(mesh.volume))
        return volume if np.isfinite(volume) and volume > 0.0 else None
    except Exception:
        return None


def volume_preserving_level_field(
    signed_field: np.ndarray,
    material_mask: np.ndarray,
    *,
    smoothing_sigma: float = 0.35,
    target_material_count: float | None = None,
) -> tuple[np.ndarray, float]:
    """Fair a continuous topology level field without discarding sub-voxel data.

    ``signed_field`` is negative in material and positive in void.  The prior
    recovery path thresholded this smooth field to a Boolean mask before
    building an SDF, which reintroduced voxel terraces.  This variant retains
    the optimizer's interpolated density boundary and shifts the zero level
    after smoothing so the thresholded material count is unchanged.
    """
    from scipy import ndimage as ndi

    field = np.asarray(signed_field, dtype=float)
    mask = np.asarray(material_mask, dtype=bool)
    if field.ndim != 3 or mask.shape != field.shape:
        raise ValueError("Level field and material mask must be matching 3-D arrays.")
    if not np.any(mask) or np.all(mask):
        raise ValueError("Level-field recovery needs both material and void voxels.")
    if not np.all(np.isfinite(field)):
        raise ValueError("Level field must contain only finite values.")

    smooth = (
        ndi.gaussian_filter(field, sigma=float(smoothing_sigma), mode="nearest")
        if smoothing_sigma > 0.0
        else field.copy()
    )
    target_count = int(
        round(
            float(target_material_count)
            if target_material_count is not None
            else float(np.count_nonzero(mask))
        )
    )
    target_count = min(max(target_count, 1), smooth.size - 1)
    flat = smooth.ravel()
    kth = min(max(target_count - 1, 0), flat.size - 1)
    iso_offset = float(np.partition(flat, kth)[kth])
    smooth -= iso_offset
    return smooth, iso_offset


def extract_volume_calibrated_surface(
    build_surface: Callable[[float], Optional[dict]],
    *,
    target_volume: float,
    relative_tolerance: float = 1e-3,
    max_extractions: int = 6,
    initial_level_step: float = 0.02,
) -> Optional[dict[str, np.ndarray | str]]:
    """Solve for the level whose *finished* surface encloses the analyzed volume.

    :func:`volume_preserving_level_field` matches a sample *count*: it shifts the
    level until the requested number of grid points falls inside. That is not the
    same as matching the enclosed volume, because the isosurface cuts through the
    boundary voxels rather than taking or leaving them whole. On the bundled
    cantilever the count was right to 0.17% while the recovered solid came out
    0.75% heavy -- a mass error the user never asked for and cannot see the
    cause of.

    ``build_surface(shift)`` must return the surface as the caller will deliver
    it, with every vertex-moving step already applied. Measuring the bare
    isosurface instead is not good enough: snapping an extruded result onto its
    exact end-cap planes afterwards moves the volume by another 1.35%, so a level
    calibrated before that step lands 1.5% heavy even though the extraction hit
    its own target to 0.0007%. It must also re-impose any analytic passive shape
    after the shift, or calibration quietly resizes the bores along with the
    optimizer's own boundary.

    The recovery field is a continuous density level set, not necessarily a
    unit-gradient physical-distance field.  An area-based Newton step therefore
    has incompatible units and can jump beyond the entire scalar range when a
    support gate removed a noticeable voxel layer.  The first correction is a
    small probe in level-set units; secant iterations then infer the actual
    volume sensitivity.  Typically three builds are enough.  The surface is
    returned uncalibrated rather than wrong if the volume cannot be measured,
    which happens when a trial surface is not closed.
    """
    target = float(target_volume)
    trials: list[tuple[float, float]] = []
    best: Optional[tuple[float, dict]] = None
    surface = None
    shift = 0.0

    for attempt in range(max(1, int(max_extractions))):
        surface = build_surface(shift)
        if surface is None:
            break
        volume = _closed_mesh_volume(surface)
        if volume is None or target <= 0.0:
            # Nothing to steer by; the uncalibrated extraction stands.
            break
        error = abs(volume - target) / target
        if best is None or error < best[0]:
            best = (error, surface)
        if error <= float(relative_tolerance):
            break
        trials.append((shift, volume))
        if attempt + 1 >= max(1, int(max_extractions)):
            break

        if len(trials) == 1:
            direction = 1.0 if target > trials[0][1] else -1.0
            shift = trials[0][0] + direction * max(
                abs(float(initial_level_step)), 1.0e-6
            )
        else:
            (shift_a, volume_a), (shift_b, volume_b) = trials[-2], trials[-1]
            if abs(volume_b - volume_a) <= 1e-30:
                break
            shift = shift_b + (target - volume_b) * (shift_b - shift_a) / (
                volume_b - volume_a
            )
        if not np.isfinite(shift):
            break

    if best is not None:
        final_volume = _closed_mesh_volume(best[1])
        logger.info(
            "Recovery level calibrated to enclosed volume in %d extraction(s): "
            "%.6g vs %.6g target (%+.4f%%).",
            len(trials) + 1,
            final_volume if final_volume else float("nan"),
            target,
            100.0 * best[0] * (1.0 if final_volume and final_volume > target else -1.0),
        )
        return best[1]
    return surface


def extract_surface_nets_surface(
    signed_distance: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    *,
    constraint_scale: float = 0.5,
    smoothing_iterations: int = 30,
) -> Optional[dict[str, np.ndarray | str]]:
    """Extract the zero set with VTK's Surface Nets, a dual-vertex method.

    Offered because a dual method places its vertices inside the cell rather
    than on grid edges, so it resolves a crease more sharply than Flying Edges
    and produced a noticeably tamer worst-case dihedral on the reference
    cantilever (98 degrees against 128).

    It is not the default, and the reason is worth stating rather than
    discovering. ``vtkSurfaceNets3D`` consumes a *label map*, so the continuous
    optimizer field this pipeline goes to some trouble to preserve is thrown
    away at the door and the boundary is recovered by constrained smoothing of a
    binary mask instead. Measured against Flying Edges on the same field that
    costs 0.8% to 5.2% of the enclosed volume depending on ``constraint_scale``,
    and moves the surface 0.31 to 0.47 mm -- more than a recovery voxel. It is
    also not adaptive here: 244,336 triangles against 244,332, so none of the
    triangle-budget saving a dual method is usually chosen for materialises.

    The volume loss is recoverable, because
    :func:`extract_volume_calibrated_surface` solves for the level that hits the
    analyzed volume and the level enters this function as the binarization
    threshold. The lost sub-voxel detail is not recoverable. Prefer the default
    unless a specific result needs the sharper crease behaviour.
    """
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    except ImportError:
        return None

    field = np.asarray(signed_distance, dtype=float)
    if field.ndim != 3 or min(field.shape) < 2:
        return None
    labels = (field <= 0.0).astype(np.uint8)
    if not labels.any() or labels.all():
        return None

    try:
        image = vtk.vtkImageData()
        image.SetDimensions(*(int(v) for v in labels.shape))
        image.SetSpacing(*(float(v) for v in np.asarray(spacing, dtype=float)[:3]))
        image.SetOrigin(*(float(v) for v in np.asarray(origin, dtype=float)[:3]))
        scalars = numpy_to_vtk(
            np.ascontiguousarray(labels.ravel(order="F")),
            deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR,
        )
        scalars.SetName("material")
        image.GetPointData().SetScalars(scalars)

        nets = vtk.vtkSurfaceNets3D()
        nets.SetInputData(image)
        nets.SetBackgroundLabel(0)
        nets.SetLabel(0, 1)
        nets.SetOutputMeshTypeToTriangles()
        nets.SmoothingOn()
        nets.SetConstraintScale(float(constraint_scale))
        smoother = nets.GetSmoother()
        if smoother is not None:
            smoother.SetNumberOfIterations(int(smoothing_iterations))

        # Surface Nets emits quads with no consistent winding. Without this the
        # signed volume cancels against itself and reads 35-46% light.
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(nets.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        poly = normals.GetOutput()
        if poly is None or poly.GetNumberOfPolys() == 0:
            return None

        vertices = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=float)
        raw_faces = np.asarray(vtk_to_numpy(poly.GetPolys().GetData()), dtype=np.int64)
        if raw_faces.size % 4 != 0:
            return None
        packed = raw_faces.reshape((-1, 4))
        if not np.all(packed[:, 0] == 3):
            return None
        return {
            "vertices": vertices,
            "faces": np.asarray(packed[:, 1:4], dtype=int),
            "surface_backend": "VTK Surface Nets (dual, label-map)",
        }
    except Exception:
        logger.exception("VTK Surface Nets extraction failed")
        return None


def extract_flying_edges_surface(
    signed_distance: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    *,
    smoothing_iterations: int = 20,
    pass_band: float = 0.08,
) -> Optional[dict[str, np.ndarray | str]]:
    """Extract and volume-conservatively fair a signed-distance zero set.

    Twenty windowed-sinc iterations remove visible voxel stair-stepping without
    the shrinkage associated with repeated Laplacian smoothing. The implicit
    surface has already been volume-calibrated before this geometric fairing.
    """
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    except ImportError:
        return None

    field = np.asarray(signed_distance, dtype=np.float32)
    if field.ndim != 3 or min(field.shape) < 2:
        return None
    if not (float(np.min(field)) < 0.0 < float(np.max(field))):
        return None

    try:
        image = vtk.vtkImageData()
        image.SetDimensions(*(int(v) for v in field.shape))
        image.SetSpacing(*(float(v) for v in np.asarray(spacing, dtype=float)[:3]))
        image.SetOrigin(*(float(v) for v in np.asarray(origin, dtype=float)[:3]))
        scalars = numpy_to_vtk(
            np.ascontiguousarray(field.ravel(order="F")),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        scalars.SetName("signed_distance")
        image.GetPointData().SetScalars(scalars)

        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, 0.0)
        contour.ComputeNormalsOff()
        contour.ComputeGradientsOff()
        contour.Update()

        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(contour.GetOutputPort())
        triangles.PassLinesOff()
        triangles.PassVertsOff()
        triangles.Update()

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(triangles.GetOutputPort())
        clean.ToleranceIsAbsoluteOff()
        clean.SetTolerance(1e-8)
        clean.Update()

        current_port = clean.GetOutputPort()
        if int(smoothing_iterations) > 0:
            fair = vtk.vtkWindowedSincPolyDataFilter()
            fair.SetInputConnection(current_port)
            fair.SetNumberOfIterations(int(smoothing_iterations))
            fair.SetPassBand(float(np.clip(pass_band, 0.01, 1.5)))
            fair.NormalizeCoordinatesOn()
            fair.BoundarySmoothingOff()
            fair.FeatureEdgeSmoothingOff()
            fair.NonManifoldSmoothingOff()
            fair.Update()
            current_port = fair.GetOutputPort()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(current_port)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        poly = normals.GetOutput()
        if (
            poly is None
            or poly.GetNumberOfPoints() == 0
            or poly.GetNumberOfPolys() == 0
        ):
            return None

        vertices = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=float)
        raw_faces = np.asarray(vtk_to_numpy(poly.GetPolys().GetData()), dtype=np.int64)
        if raw_faces.size % 4 != 0:
            logger.warning(
                "VTK returned non-triangular cells after triangle filtering."
            )
            return None
        packed = raw_faces.reshape((-1, 4))
        if not np.all(packed[:, 0] == 3):
            return None
        faces = np.asarray(packed[:, 1:4], dtype=int)
        return {
            "vertices": vertices,
            "faces": faces,
            "surface_backend": (
                "VTK Flying Edges + volume-preserving continuous field"
            ),
        }
    except Exception:
        logger.exception("VTK print-surface extraction failed")
        return None
