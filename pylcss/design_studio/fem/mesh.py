# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM mesh node — tetrahedral *and* shell-surface mesh generation via Netgen-OCC.

For thin-walled crash parts (crashboxes, tubes, automotive structures) industry
practice is shell elements on the mid-surface with an assigned thickness, not
solid Tet4.  ``mesh_type='Shell'`` makes Netgen stop after the SURFACE meshing
pass — the result is a triangle mesh whose nodes live in R^3 — and tags it
with ``shell_thickness`` so the OpenRadioss writer emits ``*SECTION_SHELL``
instead of ``*SECTION_SOLID``. For a solid input this is its boundary surface,
not an automatically extracted sheet-metal midsurface.
"""

from __future__ import annotations

import os
import tempfile
import logging
import threading
from os import PathLike
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from skfem import Mesh, MeshTet, MeshTet2
from pylcss.design_studio.core.base_node import CadQueryNode
from pylcss.design_studio.fem._helpers import suppress_output, OCCGeometry
from pylcss.design_studio.fem.quality import attach_mesh_quality

logger = logging.getLogger(__name__)
_GMSH_LOCK = threading.Lock()


class _GmshOccModel(Protocol):
    def getMass(self, dimension: int, tag: int) -> float: ...


class _GmshModel(Protocol):
    occ: _GmshOccModel

    def getEntities(self, dimension: int) -> list[tuple[int, int]]: ...

    def getBoundingBox(
        self,
        dimension: int,
        tag: int,
    ) -> tuple[float, float, float, float, float, float]: ...


class _GmshApi(Protocol):
    model: _GmshModel


def _selected_cad_face_indices(
    shape: object,
    selection: object | None,
) -> list[int]:
    """Map selected CadQuery faces to zero-based face indices on ``shape``."""
    if selection is None:
        return []
    if isinstance(selection, dict):
        selected = list(selection.get("faces") or [])
        if not selected and selection.get("face") is not None:
            selected = [selection["face"]]
    elif hasattr(selection, "vals"):
        selected = list(selection.vals())
    else:
        selected = [selection]

    base = shape.val() if hasattr(shape, "val") else shape
    try:
        all_faces = list(base.Faces())
    except Exception:
        return []

    indices = []
    for chosen in selected:
        if chosen is None or isinstance(chosen, dict):
            continue
        for index, candidate in enumerate(all_faces):
            same = False
            try:
                same = bool(candidate.isSame(chosen))
            except Exception:
                try:
                    same = bool(candidate.wrapped.IsSame(chosen.wrapped))
                except Exception:
                    same = False
            if same:
                indices.append(index)
                break
    return sorted(set(indices))


def _selected_cad_entities(
    shape: object,
    selection: object | None,
) -> tuple[int | None, list[object]]:
    """Return ``(dimension, selected entities)`` for a CAD refinement payload."""
    if selection is None:
        return None, []
    entity_type = "Face"
    if isinstance(selection, dict):
        entity_type = str(selection.get("entity_type") or "Face").title()
        selected = list(selection.get("entities") or [])
        if not selected:
            selected = list(selection.get("faces") or [])
        if not selected and selection.get("entity") is not None:
            selected = [selection["entity"]]
        if not selected and selection.get("face") is not None:
            selected = [selection["face"]]
    elif hasattr(selection, "vals"):
        selected = list(selection.vals())
    else:
        selected = [selection]
    dimensions = {"Vertex": 0, "Edge": 1, "Face": 2}
    return dimensions.get(entity_type, 2), [
        item for item in selected if item is not None and not isinstance(item, dict)
    ]


def _entity_bounds(entity: object) -> NDArray[np.float64] | None:
    """Return a CAD entity's ``[xmin, ymin, zmin, xmax, ymax, zmax]`` bounds."""
    wrapped = getattr(entity, "wrapped", entity)
    try:
        from OCP.BRepBndLib import BRepBndLib
        from OCP.Bnd import Bnd_Box

        bounds = Bnd_Box()
        BRepBndLib.Add_s(wrapped, bounds)
        return np.asarray(bounds.Get(), dtype=float)
    except Exception:
        try:
            center = entity.Center()
            xyz = np.asarray([center.x, center.y, center.z], dtype=float)
            return np.r_[xyz, xyz]
        except Exception:
            return None


def _entity_measure(entity: object, dimension: int | None) -> float:
    """Return edge length or face area when CadQuery exposes it."""
    try:
        if dimension == 1:
            return float(entity.Length())
        if dimension == 2:
            return float(entity.Area())
    except Exception:
        pass
    return 0.0


class _GmshVolumeMesh:
    """Minimal solver/viewer mesh retaining native Gmsh C3D10 connectivity."""

    def __init__(self, points: ArrayLike, connectivity: ArrayLike) -> None:
        point_array = np.asarray(points, dtype=float)
        connectivity_array = np.asarray(connectivity, dtype=int)
        self.p = np.ascontiguousarray(point_array.T)
        self.t = np.ascontiguousarray(connectivity_array.T)


def _gmsh_entity_tags_for_selection(
    shape: object,
    selection: object | None,
    gmsh: _GmshApi,
) -> tuple[int | None, list[int]]:
    """Map selected OCC subshapes to imported Gmsh entities geometrically.

    Entity centres alone are ambiguous for concentric edges and symmetric
    faces. Compare both bounding-box position and extent, with length/area as
    a tie-breaker, so an edge or vertex refinement remains attached to the
    entity the engineer actually selected after the STEP round trip.
    """
    dimension, selected = _selected_cad_entities(shape, selection)
    if not selected:
        return dimension, []
    candidates = list(gmsh.model.getEntities(dimension))
    if not candidates:
        return dimension, []
    candidate_bounds = []
    candidate_measures = []
    for _, tag in candidates:
        candidate_bounds.append(
            np.asarray(gmsh.model.getBoundingBox(dimension, tag), dtype=float)
        )
        try:
            candidate_measures.append(
                float(gmsh.model.occ.getMass(dimension, tag)) if dimension > 0 else 0.0
            )
        except Exception:
            candidate_measures.append(0.0)
    candidate_bounds = np.asarray(candidate_bounds, dtype=float)
    candidate_measures = np.asarray(candidate_measures, dtype=float)
    all_min = np.min(candidate_bounds[:, :3], axis=0)
    all_max = np.max(candidate_bounds[:, 3:], axis=0)
    scale = max(float(np.linalg.norm(all_max - all_min)), 1.0e-12)
    tags = []
    for entity in selected:
        bounds = _entity_bounds(entity)
        if bounds is None:
            continue
        center = 0.5 * (bounds[:3] + bounds[3:])
        extent = bounds[3:] - bounds[:3]
        candidate_centers = 0.5 * (candidate_bounds[:, :3] + candidate_bounds[:, 3:])
        candidate_extents = candidate_bounds[:, 3:] - candidate_bounds[:, :3]
        score = (
            np.linalg.norm(candidate_centers - center, axis=1) / scale
            + np.linalg.norm(candidate_extents - extent, axis=1) / scale
        )
        measure = _entity_measure(entity, dimension)
        if measure > 0.0:
            positive = candidate_measures > 0.0
            score[positive] += 0.05 * np.abs(
                np.log(candidate_measures[positive] / measure)
            )
        index = int(np.argmin(score))
        tags.append(int(candidates[index][1]))
    return dimension, sorted(set(tags))


def _generate_gmsh_mesh(
    step_path: str | PathLike[str],
    msh_path: str | PathLike[str],
    *,
    shape: object,
    mesh_type: str,
    element_size: float,
    refinement: object | None,
    refinement_size: float,
    algorithm: str,
    curvature_elements: int,
) -> None:
    """Generate a CAD-conforming mesh with Gmsh's OpenCASCADE importer."""
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "Gmsh is not installed. Install the pinned `gmsh` dependency or "
            "select Netgen."
        ) from exc

    is_shell = mesh_type == "Shell"
    is_quadratic = mesh_type == "Tet10"
    with _GMSH_LOCK:
        # Graph execution normally runs in a QThread.  Gmsh's Python wrapper
        # installs SIGINT/SIGTERM handlers when ``interruptible`` is left at
        # its default, which Python only permits on the main interpreter
        # thread.  Gmsh itself is serialized by _GMSH_LOCK, so disabling those
        # process-global handlers is both thread-safe and avoids the otherwise
        # deterministic "signal only works in main thread" failure.
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("pylcss_mesh")
            gmsh.model.occ.importShapes(str(step_path), highestDimOnly=False)
            gmsh.model.occ.synchronize()
            if not gmsh.model.getEntities(2):
                raise RuntimeError("Gmsh imported no CAD surfaces from the STEP shape.")
            if not is_shell and not gmsh.model.getEntities(3):
                raise RuntimeError(
                    "Gmsh imported no closed CAD volume. Use Shell for an "
                    "intentional surface model or repair the solid."
                )

            gmsh.option.setNumber("Mesh.MeshSizeMax", float(element_size))
            gmsh.option.setNumber(
                "Mesh.MeshSizeMin",
                min(float(element_size), float(refinement_size))
                if refinement is not None
                else 0.0,
            )
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", int(curvature_elements))
            gmsh.option.setNumber("Mesh.Optimize", 1)

            algorithm_3d = {
                "HXT": 10,
                "Delaunay": 1,
                "Frontal (Netgen)": 4,
            }.get(str(algorithm), 10)
            gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)
            gmsh.option.setNumber(
                "Mesh.Algorithm",
                5 if str(algorithm) == "Delaunay" else 6,
            )

            if refinement is not None:
                dimension, tags = _gmsh_entity_tags_for_selection(
                    shape, refinement, gmsh
                )
                if not tags:
                    raise RuntimeError(
                        "The refinement selection could not be mapped to the "
                        "imported CAD face, edge, or vertex."
                    )
                distance = gmsh.model.mesh.field.add("Distance")
                list_name = {
                    0: "PointsList",
                    1: "CurvesList",
                    2: "SurfacesList",
                }[dimension]
                gmsh.model.mesh.field.setNumbers(distance, list_name, tags)
                gmsh.model.mesh.field.setNumber(distance, "Sampling", 60)
                threshold = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
                gmsh.model.mesh.field.setNumber(
                    threshold, "SizeMin", float(refinement_size)
                )
                gmsh.model.mesh.field.setNumber(
                    threshold, "SizeMax", float(element_size)
                )
                gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "DistMax",
                    max(float(element_size), 3.0 * float(refinement_size)),
                )
                gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

            if is_quadratic:
                gmsh.option.setNumber("Mesh.ElementOrder", 2)
                # Elastic+optimization untangling for curved C3D10 elements.
                gmsh.option.setNumber("Mesh.HighOrderOptimize", 2)
            gmsh.model.mesh.generate(2 if is_shell else 3)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.Binary", 0)
            gmsh.write(str(msh_path))
        finally:
            gmsh.finalize()


def _load_gmsh_volume_mesh(
    msh_path: str | PathLike[str],
    wants_tet10: bool,
) -> MeshTet | _GmshVolumeMesh:
    import meshio

    mio = meshio.read(msh_path)
    wanted = "tetra10" if wants_tet10 else "tetra"
    blocks = [
        np.asarray(block.data, dtype=int) for block in mio.cells if block.type == wanted
    ]
    if not blocks:
        raise ValueError(f"Gmsh produced no {wanted} volume elements.")
    connectivity = np.vstack(blocks)
    points = np.asarray(mio.points, dtype=float)[:, :3]
    if wants_tet10:
        return _GmshVolumeMesh(points, connectivity)
    return MeshTet(points.T, connectivity[:, :4].T)


class _ShellSurfaceMesh:
    """Lightweight skfem-compatible wrapper for a 3D triangle (shell) mesh.

    skfem.MeshTri is strictly 2D (``p.shape == (2, N)``), so we cannot use it
    for a surface embedded in R^3.  Downstream code only touches ``mesh.p``
    and ``mesh.t``, so this duck-typed wrapper is enough for the OpenRadioss
    deck writer, the boundary-condition matchers, and the VTK viewer.
    """

    def __init__(
        self,
        points_3d: ArrayLike,
        triangles: ArrayLike,
        *,
        shell_thickness: float,
        shell_nip: int,
    ) -> None:
        point_array = np.asarray(points_3d, dtype=float)
        triangle_array = np.asarray(triangles, dtype=int)
        self.p = np.ascontiguousarray(point_array.T)  # (3, N_nodes)
        self.t = np.ascontiguousarray(triangle_array.T)  # (3, N_elem)
        self.shell_thickness = float(shell_thickness)
        self.shell_nip = int(shell_nip)


class MeshNode(CadQueryNode):
    """Generate a CAD-conforming tetrahedral or explicit shell mesh."""

    __identifier__ = "com.cad.sim.mesh"
    NODE_NAME = "Generate Mesh"

    def __init__(self) -> None:
        super().__init__()
        self.add_input("shape", color=(100, 255, 100))
        self.add_input("element_size", color=(180, 180, 0))
        # NEW: Local refinement inputs
        self.add_input(
            "refinement_faces", color=(255, 100, 100)
        )  # List of faces for refinement
        self.add_input(
            "refinement_size", color=(255, 100, 100)
        )  # Smaller element size for refinement
        self.add_output("mesh", color=(200, 100, 200))

        # Mesh type selection.  'Shell' produces a 3-node triangle surface
        # mesh suitable for OpenRadioss *ELEMENT_SHELL crash decks; the
        # writer reads `shell_thickness` (mm) off the returned mesh object.
        self.create_property(
            "mesh_type", "Tet", widget_type="combo", items=["Tet", "Tet10", "Shell"]
        )
        self.create_property(
            "mesher",
            "Gmsh (Recommended)",
            widget_type="combo",
            items=["Gmsh (Recommended)", "Netgen"],
        )
        self.create_property(
            "gmsh_algorithm",
            "HXT",
            widget_type="combo",
            items=["HXT", "Delaunay", "Frontal (Netgen)"],
        )
        self.create_property("curvature_elements", 20, widget_type="int")
        self.create_property("element_size", 2.0, widget_type="float")
        self.create_property(
            "refinement_size", 0.5, widget_type="float"
        )  # Finer mesh for critical areas
        # Shell-only: through-thickness wall (mm) written to *SECTION_SHELL,
        # and number of through-thickness integration points for elasto-plastic
        # stress recovery.  Industry crash practice: 3–5 NIP for thin sheet metal.
        self.create_property("shell_thickness", 1.5, widget_type="float")
        self.create_property("shell_nip", 5, widget_type="int")

    def run(self) -> Mesh | _GmshVolumeMesh | _ShellSurfaceMesh | None:
        self.clear_error()
        mesher = str(self.get_property("mesher") or "Gmsh (Recommended)")
        use_gmsh = mesher.startswith("Gmsh")
        if not use_gmsh and OCCGeometry is None:
            self.set_error("Netgen-occ is not installed")
            return None

        shape = self.get_input_shape("shape")
        # Resolve element size input with fallback to property
        size = self.get_input_value("element_size", "element_size")

        # NEW: Get refinement parameters
        refinement_faces = self.get_input_value("refinement_faces", None)
        refinement_size = self.get_input_value("refinement_size", "refinement_size")
        try:
            size = float(size)
            refinement_size = float(refinement_size)
            curvature_elements = int(self.get_property("curvature_elements") or 0)
        except (TypeError, ValueError):
            self.set_error(
                "Element/refinement sizes and curvature resolution must be numeric."
            )
            return None

        if not np.isfinite(size) or size <= 0.0:
            self.set_error("Element size must be a finite value greater than zero.")
            return None
        if not np.isfinite(refinement_size) or refinement_size <= 0.0:
            self.set_error("Refinement size must be a finite value greater than zero.")
            return None
        if refinement_faces is not None and refinement_size >= size:
            self.set_error(
                "Local refinement size must be smaller than the global element size."
            )
            return None
        if curvature_elements < 0 or curvature_elements > 200:
            self.set_error("Curvature resolution must be between 0 and 200.")
            return None

        if not shape:
            self.set_error("Connect a CAD solid to the mesh node's shape input.")
            return None

        # Handle assemblies by converting to compound
        if hasattr(shape, "toCompound"):
            try:
                shape = shape.toCompound()
            except Exception as exc:
                self.set_error(f"Could not convert the assembly to a compound: {exc}")
                return None

        # Optimized temporary file handling for performance
        # Try to use RAM disk if available (significant speedup for optimization loops)
        try:
            # os.access() can report writable for a Windows junction or sandbox
            # path that still rejects file creation.  Verify candidates with a
            # real create/delete probe before giving Netgen a directory.
            temp_base = None
            candidates = ["R:\\", "Z:\\", "/tmp/", "/dev/shm/", tempfile.gettempdir()]
            for path in candidates:
                if not os.path.isdir(path):
                    continue
                try:
                    with tempfile.NamedTemporaryFile(
                        prefix=".pylcss_probe_", dir=path, delete=True
                    ):
                        pass
                    temp_base = path
                    break
                except OSError:
                    continue
            if temp_base is None:
                raise OSError(
                    "No writable temporary directory is available for Netgen."
                )

            # Initialise paths before try so the finally block can safely
            # reference them even if the NamedTemporaryFile call fails.
            step_path = None
            msh_path = None

            # Create temporary files in optimized location
            with tempfile.NamedTemporaryFile(
                suffix=".step", dir=temp_base, delete=False
            ) as step_file:
                step_path = step_file.name

            msh_path = step_path.replace(".step", ".msh")

            try:
                # 1. Export CadQuery shape to STEP
                if hasattr(shape, "val"):
                    shape.val().exportStep(step_path)
                else:
                    shape.exportStep(step_path)

                mesh_type = (self.get_property("mesh_type") or "Tet").strip()
                is_shell = mesh_type.lower() == "shell"
                wants_tet10 = mesh_type.lower() == "tet10"
                if mesh_type not in {"Tet", "Tet10", "Shell"}:
                    raise ValueError(f"Unsupported mesh type {mesh_type!r}.")
                if is_shell:
                    thickness = float(self.get_property("shell_thickness") or 0.0)
                    nip = int(self.get_property("shell_nip") or 0)
                    if not np.isfinite(thickness) or thickness <= 0.0:
                        raise ValueError("Shell thickness must be greater than zero.")
                    if nip < 1:
                        raise ValueError("Shell integration points must be at least 1.")

                if use_gmsh:
                    _generate_gmsh_mesh(
                        step_path,
                        msh_path,
                        shape=shape,
                        mesh_type=mesh_type,
                        element_size=size,
                        refinement=refinement_faces,
                        refinement_size=refinement_size,
                        algorithm=(self.get_property("gmsh_algorithm") or "HXT"),
                        curvature_elements=curvature_elements,
                    )
                else:
                    # Load Geometry with Netgen and generate a mesh.
                    with suppress_output():
                        geo = OCCGeometry(step_path)

                        if refinement_faces is not None:
                            dimension, selected = _selected_cad_entities(
                                shape, refinement_faces
                            )
                            if dimension != 2:
                                raise ValueError(
                                    "Netgen local refinement accepts CAD faces only. "
                                    "Select Gmsh to refine an edge or vertex."
                                )
                            face_indices = _selected_cad_face_indices(
                                shape, refinement_faces
                            )
                            if not face_indices:
                                raise ValueError(
                                    "Local refinement is connected, but its selected "
                                    "CAD faces do not belong to the meshed shape."
                                )
                            for face_index in face_indices:
                                geo.SetFaceMeshsize(int(face_index), refinement_size)

                        if is_shell:
                            import netgen.meshing as ngmeshing

                            ng_mesh = geo.GenerateMesh(
                                maxh=size,
                                perfstepsend=ngmeshing.MeshingStep.MESHSURFACE,
                            )
                        else:
                            ng_mesh = geo.GenerateMesh(maxh=size)
                        ng_mesh.Export(msh_path, "Gmsh2 Format")

                # 5. Load into skfem (Tet) or meshio + ShellSurfaceMesh wrapper (Shell)
                if is_shell:
                    import meshio

                    mio = meshio.read(msh_path)
                    triangles = None
                    for cell_block in mio.cells:
                        if cell_block.type == "triangle":
                            triangles = np.asarray(cell_block.data, dtype=int)
                            break
                    if triangles is None or triangles.size == 0:
                        logger.error(
                            "FEA Mesh: mesher produced no surface triangles; "
                            "is the input a solid (use Tet) or a shell/face (use Shell)?"
                        )
                        self.set_error(
                            "The mesher produced no surface triangles. Use Shell for a valid "
                            "face/shell or Tet for a closed solid."
                        )
                        return None
                    points_3d = np.asarray(mio.points, dtype=float)
                    mesh = _ShellSurfaceMesh(
                        points_3d,
                        triangles,
                        shell_thickness=thickness,
                        shell_nip=nip,
                    )
                    logger.debug(
                        "FEA Mesh: Shell mesh ready. Nodes: %d, Tris: %d, t=%.4g mm, NIP=%d",
                        mesh.p.shape[1],
                        mesh.t.shape[1],
                        thickness,
                        nip,
                    )
                else:
                    logger.debug("FEA Mesh: Loading tetrahedral mesh...")
                    mesh = (
                        _load_gmsh_volume_mesh(msh_path, wants_tet10)
                        if use_gmsh
                        else Mesh.load(msh_path)
                    )
                    if np.asarray(mesh.t).ndim != 2 or np.asarray(mesh.t).shape[0] < 4:
                        raise ValueError(
                            "The mesher did not produce a tetrahedral volume mesh."
                        )
                    if wants_tet10 and not use_gmsh:
                        # MeshTet2 stores its six midside nodes in
                        # dofs.element_dofs while keeping corner topology in t.
                        # The CalculiX/VTK adapters expand that to true C3D10
                        # connectivity when exporting or rendering.
                        mesh = MeshTet2.from_mesh(mesh)
                    logger.debug(
                        "FEA Mesh: Load complete. Nodes: %d, Tets: %d, order=%s",
                        mesh.p.shape[1],
                        mesh.t.shape[1],
                        "quadratic C3D10" if wants_tet10 else "linear C3D4",
                    )

                quality_report = attach_mesh_quality(mesh)
                if not quality_report["solver_ready"]:
                    raise ValueError(
                        "Generated mesh contains collapsed or invalid elements. "
                        "Change the element size or repair the CAD geometry."
                    )
                logger.info(
                    "FEA Mesh quality: min=%.4f, p05=%.4f, mean=%.4f; %s",
                    quality_report["min_mean_ratio"],
                    quality_report["p05_mean_ratio"],
                    quality_report["mean_mean_ratio"],
                    quality_report["assessment"],
                )

            except Exception as e:
                logger.error("FEA Mesh: ERROR loading mesh: %s", e)
                self.set_error(f"Mesh generation failed: {e}")
                return None

            finally:
                # Clean up temporary files immediately.
                # Guard against step_path / msh_path being None when the
                # NamedTemporaryFile call itself failed (UnboundLocalError fix).
                try:
                    if step_path and os.path.exists(step_path):
                        os.remove(step_path)
                    if msh_path and os.path.exists(msh_path):
                        os.remove(msh_path)
                except OSError:
                    pass  # Ignore cleanup errors

        except Exception as exc:
            logger.exception("FEA Mesh: unexpected meshing failure")
            self.set_error(f"Mesh generation failed: {exc}")
            return None

        return mesh
