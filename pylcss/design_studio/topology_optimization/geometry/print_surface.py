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
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def volume_preserving_signed_distance(
    material_mask: np.ndarray,
    spacing: np.ndarray,
    *,
    smoothing_sigma: float = 0.8,
) -> tuple[np.ndarray, float]:
    """Return a smoothed SDF whose zero set preserves material voxel count."""
    from scipy import ndimage as ndi

    mask = np.asarray(material_mask, dtype=bool)
    if mask.ndim != 3 or not np.any(mask) or np.all(mask):
        raise ValueError(
            "Signed-distance recovery needs both material and void voxels."
        )
    sampling = tuple(float(v) for v in np.asarray(spacing, dtype=float)[:3])
    outside = ndi.distance_transform_edt(~mask, sampling=sampling)
    inside = ndi.distance_transform_edt(mask, sampling=sampling)
    sdf = outside - inside
    if smoothing_sigma > 0.0:
        sdf = ndi.gaussian_filter(
            sdf,
            sigma=float(smoothing_sigma),
            mode="nearest",
        )

    target_count = int(np.count_nonzero(mask))
    flat = np.asarray(sdf, dtype=float).ravel()
    kth = min(max(target_count - 1, 0), flat.size - 1)
    iso_offset = float(np.partition(flat, kth)[kth])
    sdf -= iso_offset
    return sdf, iso_offset


def volume_preserving_level_field(
    signed_field: np.ndarray,
    material_mask: np.ndarray,
    *,
    smoothing_sigma: float = 0.35,
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
    target_count = int(np.count_nonzero(mask))
    flat = smooth.ravel()
    kth = min(max(target_count - 1, 0), flat.size - 1)
    iso_offset = float(np.partition(flat, kth)[kth])
    smooth -= iso_offset
    return smooth, iso_offset


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
