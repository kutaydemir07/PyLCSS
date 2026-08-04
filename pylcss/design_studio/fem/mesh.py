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
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from skfem import Mesh, MeshTet2
from pylcss.design_studio.core.base_node import CadQueryNode
from pylcss.design_studio.fem._helpers import suppress_output, OCCGeometry
from pylcss.design_studio.fem.quality import attach_mesh_quality

logger = logging.getLogger(__name__)


def migrate_removed_mesher_properties(session_data: object) -> bool:
    """Drop retired backend controls before older projects are deserialized."""
    if not isinstance(session_data, dict):
        return False
    nodes = session_data.get("nodes")
    if not isinstance(nodes, dict):
        return False

    changed = False
    retired = ("mesher", "gmsh_algorithm", "curvature_elements")
    for node_data in nodes.values():
        if not isinstance(node_data, dict):
            continue
        if not str(node_data.get("type_", "")).endswith(".MeshNode"):
            continue
        custom = node_data.get("custom")
        if not isinstance(custom, dict):
            continue
        for name in retired:
            if name in custom:
                custom.pop(name)
                changed = True
    return changed


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
    """Generate a CAD mesh or load a deterministic shell mesh artifact."""

    __identifier__ = "com.cad.sim.mesh"
    NODE_NAME = "Mesh"

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
            "mesh_source",
            "CAD",
            widget_type="combo",
            items=["CAD", "Mesh NPZ"],
        )
        self.create_property("mesh_path", "", widget_type="text")
        self.create_property(
            "mesh_type", "Tet", widget_type="combo", items=["Tet", "Tet10", "Shell"]
        )
        self.create_property("element_size", 2.0, widget_type="float")
        self.create_property(
            "refinement_size", 0.5, widget_type="float"
        )  # Finer mesh for critical areas
        # Shell-only: through-thickness wall (mm) written to *SECTION_SHELL,
        # and number of through-thickness integration points for elasto-plastic
        # stress recovery.  Industry crash practice: 3–5 NIP for thin sheet metal.
        self.create_property("shell_thickness", 1.5, widget_type="float")
        self.create_property("shell_nip", 5, widget_type="int")

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _resolve_mesh_path(
        cls,
        value: object,
        project_dir: object | None = None,
    ) -> Path | None:
        """Resolve absolute, project-relative, repo-relative, or cwd paths."""
        raw = str(value or "").strip()
        if not raw:
            return None
        path = Path(raw)
        candidates = [path]
        if project_dir:
            candidates.append(Path(str(project_dir)) / path)
        candidates.extend((cls._repo_root() / path, Path.cwd() / path))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _load_shell_mesh_npz(self) -> _ShellSurfaceMesh | None:
        raw_path = self.get_property("mesh_path")
        mesh_path = self._resolve_mesh_path(
            raw_path,
            getattr(self, "_project_dir", None),
        )
        if mesh_path is None:
            self.set_error(
                "Mesh NPZ path could not be resolved. Select a .npz file "
                f"containing points and triangles (checked {raw_path!r})."
            )
            return None
        if mesh_path.suffix.lower() != ".npz":
            self.set_error("Imported deterministic meshes must use the .npz format.")
            return None

        try:
            with np.load(mesh_path, allow_pickle=False) as archive:
                if "points" not in archive or "triangles" not in archive:
                    raise ValueError(
                        "The mesh archive must contain 'points' and 'triangles'."
                    )
                points = np.asarray(archive["points"], dtype=float)
                triangles_raw = np.asarray(archive["triangles"])
                thickness = float(
                    np.asarray(
                        archive["shell_thickness"]
                        if "shell_thickness" in archive
                        else self.get_property("shell_thickness")
                    ).reshape(-1)[0]
                )
                nip = int(
                    np.asarray(
                        archive["shell_nip"]
                        if "shell_nip" in archive
                        else self.get_property("shell_nip")
                    ).reshape(-1)[0]
                )

            if points.ndim != 2:
                raise ValueError("Mesh points must be a two-dimensional array.")
            if points.shape[1] == 3:
                pass
            elif points.shape[0] == 3:
                points = points.T
            else:
                raise ValueError("Mesh points must have shape (N, 3) or (3, N).")
            if points.shape[0] < 3 or not np.all(np.isfinite(points)):
                raise ValueError("Mesh points must contain at least three finite nodes.")

            if triangles_raw.ndim != 2:
                raise ValueError("Shell triangles must be a two-dimensional array.")
            if triangles_raw.shape[1] == 3:
                pass
            elif triangles_raw.shape[0] == 3:
                triangles_raw = triangles_raw.T
            else:
                raise ValueError(
                    "Shell triangles must have shape (M, 3) or (3, M)."
                )
            if triangles_raw.shape[0] < 1:
                raise ValueError("The mesh archive contains no shell triangles.")
            if not np.issubdtype(triangles_raw.dtype, np.integer):
                if not np.all(np.isfinite(triangles_raw)):
                    raise ValueError("Shell connectivity contains non-finite values.")
                rounded = np.rint(triangles_raw)
                if not np.allclose(triangles_raw, rounded):
                    raise ValueError("Shell connectivity must contain integer node ids.")
                triangles_raw = rounded
            triangles = np.asarray(triangles_raw, dtype=int)
            if np.any(triangles < 0) or np.any(triangles >= points.shape[0]):
                raise ValueError("Shell connectivity references a missing mesh node.")
            if np.any(
                (triangles[:, 0] == triangles[:, 1])
                | (triangles[:, 1] == triangles[:, 2])
                | (triangles[:, 0] == triangles[:, 2])
            ):
                raise ValueError("Shell connectivity contains repeated-node triangles.")
            if not np.isfinite(thickness) or thickness <= 0.0:
                raise ValueError("Shell thickness must be greater than zero.")
            if nip < 1:
                raise ValueError("Shell integration points must be at least 1.")

            mesh = _ShellSurfaceMesh(
                points,
                triangles,
                shell_thickness=thickness,
                shell_nip=nip,
            )
            quality_report = attach_mesh_quality(mesh)
            if not quality_report["solver_ready"]:
                raise ValueError(
                    "Imported mesh contains collapsed or invalid shell elements."
                )
            logger.info(
                "FEA Mesh: loaded deterministic shell mesh %s "
                "(%d nodes, %d triangles, t=%.4g mm, NIP=%d)",
                mesh_path,
                mesh.p.shape[1],
                mesh.t.shape[1],
                thickness,
                nip,
            )
            return mesh
        except Exception as exc:
            logger.error("FEA Mesh: could not load %s: %s", mesh_path, exc)
            self.set_error(f"Mesh import failed: {exc}")
            return None

    def run(self) -> Mesh | _ShellSurfaceMesh | None:
        self.clear_error()
        source = str(self.get_property("mesh_source") or "CAD").strip()
        if source == "Mesh NPZ":
            return self._load_shell_mesh_npz()
        if source != "CAD":
            self.set_error(f"Unsupported mesh source {source!r}.")
            return None
        if OCCGeometry is None:
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
        except (TypeError, ValueError):
            self.set_error("Element and refinement sizes must be numeric.")
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

                # Load Geometry with Netgen and generate a mesh.
                with suppress_output():
                    geo = OCCGeometry(step_path)

                    if refinement_faces is not None:
                        dimension, _ = _selected_cad_entities(
                            shape, refinement_faces
                        )
                        if dimension != 2:
                            raise ValueError(
                                "Netgen local refinement accepts CAD faces only."
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
                    mesh = Mesh.load(msh_path)
                    if np.asarray(mesh.t).ndim != 2 or np.asarray(mesh.t).shape[0] < 4:
                        raise ValueError(
                            "The mesher did not produce a tetrahedral volume mesh."
                        )
                    if wants_tet10:
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
