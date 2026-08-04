# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Background execution and export workers for Design Studio."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from PySide6 import QtCore

logger = logging.getLogger(__name__)

__all__ = [
    "GraphExecutionWorker",
    "TopOptCadPreviewWorker",
    "TopOptMeshExportWorker",
    "TopOptStepExportWorker",
]


class GraphExecutionWorker(QtCore.QThread):
    """Background worker to run the node graph without freezing the UI."""

    computation_finished = QtCore.Signal(object)  # Emits results dict
    computation_cancelled = QtCore.Signal(object)  # Emits safe partial results
    computation_error = QtCore.Signal(str)
    optimization_step = QtCore.Signal(
        object, object, int, int
    )  # preview/status payload, optional density, step, total

    def __init__(self, nodes, skip_simulation=False, skip_meshing=False, parent=None):
        super().__init__(parent)
        self.nodes = nodes
        self.skip_simulation = skip_simulation
        # Opening a project previews geometry only; meshing waits for Run.
        self.skip_meshing = skip_meshing
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            from pylcss.design_studio.engine import (
                GraphExecutionCancelled,
                execute_graph,
            )

            # The TopOpt node sends one design-domain preview and then tiny
            # iteration-status payloads. Final rendering is owned by the
            # normal computation_finished path.
            def progress_cb(mesh, densities, step, total):
                if self._is_running:
                    self.optimization_step.emit(mesh, densities, step, total)

            # Pass skip_simulation and callback to engine.  A preview is an
            # automatic background refresh the user never asked for, so a node
            # that is still being wired up must not interrupt them with a
            # dialog; its diagnosis stays on the node for the panel to show.
            results = execute_graph(
                self.nodes,
                skip_simulation=self.skip_simulation,
                skip_meshing=self.skip_meshing,
                cancel_callback=lambda: not self._is_running,
                progress_callback=progress_cb,
                raise_on_error=not self.skip_simulation,
            )

            if self._is_running:
                self.computation_finished.emit(results)
            else:
                self.computation_cancelled.emit(results)
        except GraphExecutionCancelled as exc:
            self.computation_cancelled.emit(exc.results)
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.computation_error.emit(str(e))
        finally:
            self._is_running = False

    def cancel(self):
        """Ask long-running simulation nodes to stop cleanly."""
        self._is_running = False
        self.requestInterruption()
        for node in self.nodes:
            request_stop = getattr(node, "request_stop", None)
            if callable(request_stop):
                request_stop()


#: STEP application protocol. AP242 is the current MBD schema and supersedes
#: AP214; OCCT still defaults to AP214IS, which is what this file used to
#: inherit silently.
STEP_SCHEMA = "AP242DIS"
#: Model units. Everything in PyLCSS is millimetres, and a STEP file that does
#: not say so is only correct by the receiving system's assumption.
STEP_UNIT = "MM"


def _write_step_shape(shape, path, product_name):
    """Write STEP with declared units, schema and part name, and verify it.

    CadQuery's exporter sets only the pcurve and precision statics, so
    everything else fell through to OCCT's defaults: AP214IS, and a product
    name of "Open CASCADE STEP translator 7.8". That name is what the receiving
    CAD or PDM system shows as the part, so every body PyLCSS ever exported
    arrived labelled as the translator that wrote it.

    OCCT reads these statics when the writer is constructed, and 7.6 onward
    needs a writer to have existed before they take effect, hence the discarded
    one below. The write status is checked because ``Write`` reports failure by
    return code rather than by raising.

    The statics are process-global and shared with every other OCCT STEP write
    in this interpreter — the assistant's CAD export and the temporary body the
    mesher hands to CalculiX among them — so they are restored afterwards. A
    topology export must not leave the next unrelated file carrying this part's
    name.
    """
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_Writer

    solid = shape.val() if hasattr(shape, "val") else shape
    if not hasattr(solid, "exportStep"):
        raise RuntimeError(
            "The topology result did not produce a CAD body to write as STEP."
        )

    name = str(product_name or "").strip() or "PyLCSS topology result"
    settings = {
        "write.step.unit": STEP_UNIT,
        "write.step.schema": STEP_SCHEMA,
        "write.step.product.name": name,
    }
    STEPControl_Writer()
    previous = {key: Interface_Static.CVal_s(key) for key in settings}
    try:
        for key, value in settings.items():
            Interface_Static.SetCVal_s(key, value)
        status = solid.exportStep(str(path))
    finally:
        for key, value in previous.items():
            Interface_Static.SetCVal_s(key, value)
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP writer rejected the body (status {status}).")
    return True


def _external_write_cad_step(
    payload,
    path,
    reconstruction_strategy="auto",
    product_name="",
):
    from pylcss.design_studio.topology_optimization.geometry.cad_reconstruction import (
        reconstruct_topopt_cad,
    )

    shape = reconstruct_topopt_cad(
        payload,
        source_geometry="Recovered Shape",
        reconstruction_strategy=str(reconstruction_strategy or "auto"),
        sew_tolerance=1e-4,
        max_faces=1500,
    )
    # The name comes from the caller's destination, not from ``path``: the
    # export writes to a temporary beside it and commits atomically, so
    # ``path`` here is ".bracket_a8f31c.step".
    return _write_step_shape(shape, path, product_name)


def _external_write_cad_brep(payload, path, reconstruction_strategy="auto"):
    """Build preview CAD in a process so OCC cannot stall the Qt event loop."""
    import cadquery as cq
    from pylcss.design_studio.topology_optimization.geometry.cad_reconstruction import (
        reconstruct_topopt_cad,
    )

    shape, report = reconstruct_topopt_cad(
        payload,
        source_geometry="Recovered Shape",
        reconstruction_strategy=str(reconstruction_strategy or "auto"),
        sew_tolerance=1e-4,
        max_faces=1500,
        return_report=True,
    )
    cq.exporters.export(shape, str(path), exportType="BREP")
    return report


#: Chordal deviation as a fraction of the body's bounding-box diagonal. The
#: previous fixed 0.01 mm was calibrated for a part around 100 mm across and
#: silently changed meaning with size: on a 500 mm bracket it emitted five
#: times the triangles that deviation buys, and on a 10 mm fitting it was
#: coarser than the printer's own resolution.
STL_RELATIVE_TOLERANCE = 1.0e-4
#: Clamps in model units, so a tiny part still gets a usable mesh and a huge
#: one cannot ask for an unbounded triangle count.
STL_MINIMUM_TOLERANCE = 5.0e-4
STL_MAXIMUM_TOLERANCE = 0.25
#: Radians. Independent of size, so it is left fixed.
STL_ANGULAR_TOLERANCE = 0.1


def _tessellation_tolerance(shape) -> float:
    """Chordal tolerance scaled to the body being exported."""
    try:
        box = shape.BoundingBox()
        diagonal = float(
            np.linalg.norm([box.xlen, box.ylen, box.zlen])
        )
    except Exception:
        logger.debug("Could not measure the body; using the default STL tolerance.")
        return 0.01
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        return 0.01
    return float(
        np.clip(
            diagonal * STL_RELATIVE_TOLERANCE,
            STL_MINIMUM_TOLERANCE,
            STL_MAXIMUM_TOLERANCE,
        )
    )


def _temporary_export_path(path: str | os.PathLike[str]) -> str:
    """Return a closed temporary file beside an export destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.stem}_",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    return temporary


def _commit_export(temporary: str, destination: str) -> None:
    """Atomically replace ``destination`` with a verified non-empty file."""
    if not os.path.isfile(temporary) or os.path.getsize(temporary) <= 0:
        raise RuntimeError("The exporter did not create a non-empty file.")
    os.replace(temporary, destination)


def _write_binary_stl(path: str, vertices: object, faces: object) -> int:
    """Write a validated binary STL and return its triangle count."""
    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int64)
    if vertices_array.ndim != 2 or vertices_array.shape[1:] != (3,):
        raise ValueError("STL vertices must have shape (n, 3).")
    if faces_array.ndim != 2 or faces_array.shape[1:] != (3,):
        raise ValueError("STL faces must have shape (n, 3).")
    if not len(vertices_array) or not len(faces_array):
        raise ValueError("The STL mesh does not contain any triangles.")
    if not np.isfinite(vertices_array).all():
        raise ValueError("STL vertices contain NaN or infinite coordinates.")
    if faces_array.min() < 0 or faces_array.max() >= len(vertices_array):
        raise ValueError("STL faces reference vertices outside the mesh.")

    triangles = vertices_array[faces_array]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("STL contains a degenerate zero-area triangle.")
    normals /= lengths[:, None]

    records = np.empty(
        len(faces_array),
        dtype=np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        ),
    )
    records["normal"] = normals
    records["vertices"] = triangles
    records["attribute"] = 0
    with open(path, "wb") as stream:
        stream.write(b"Binary STL exported by PyLCSS".ljust(80, b"\0"))
        stream.write(np.asarray([len(records)], dtype="<u4").tobytes())
        records.tofile(stream)
    return int(len(records))


def _topopt_cad_payload(
    topo_output,
    *,
    density_cutoff=0.30,
    extrusion_axis="none",
    passive_regions=None,
):
    """Return the self-contained payload shared by preview and STEP export."""
    payload = dict(topo_output or {})
    payload["density_cutoff"] = float(density_cutoff or 0.30)
    regions = dict(passive_regions or {})
    if regions:
        payload["passive_regions"] = regions
    else:
        payload.setdefault("passive_regions", {})
    payload["extrusion_axis"] = str(extrusion_axis or "none").strip().lower()
    return payload


class TopOptCadPreviewWorker(QtCore.QThread):
    """Build the real TopOpt CAD body without blocking the viewer."""

    preview_finished = QtCore.Signal(object, object)
    preview_error = QtCore.Signal(str)

    def __init__(
        self,
        topo_output,
        *,
        density_cutoff=0.30,
        extrusion_axis="none",
        passive_regions=None,
        reconstruction_strategy="auto",
        parent=None,
    ):
        super().__init__(parent)
        self.topo_output = dict(topo_output or {})
        self.density_cutoff = float(density_cutoff or 0.30)
        self.extrusion_axis = str(extrusion_axis or "none").strip().lower()
        self.passive_regions = dict(passive_regions or {})
        self.reconstruction_strategy = str(
            reconstruction_strategy or "auto"
        ).strip().lower()

    def run(self):
        temporary = ""
        try:
            payload = _topopt_cad_payload(
                self.topo_output,
                density_cutoff=self.density_cutoff,
                extrusion_axis=self.extrusion_axis,
                passive_regions=self.passive_regions,
            )
            descriptor, temporary = tempfile.mkstemp(
                prefix="pylcss_topopt_preview_",
                suffix=".brep",
            )
            os.close(descriptor)

            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _external_write_cad_brep,
                    payload,
                    temporary,
                    self.reconstruction_strategy,
                )
                report = future.result()

            import cadquery as cq

            shape = cq.importers.importBrep(temporary)
            self.preview_finished.emit(shape, report)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.preview_error.emit(str(exc))
        finally:
            if temporary and os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass


class TopOptMeshExportWorker(QtCore.QThread):
    """Export topology STL or beam-lattice 3MF without blocking Qt."""

    export_finished = QtCore.Signal(object)
    export_error = QtCore.Signal(str)

    def __init__(
        self,
        path,
        *,
        cad_shape=None,
        recovered_shape=None,
        topology_result=None,
        structure_options=None,
        member_plan=None,
        export_format="stl",
        parent=None,
    ):
        super().__init__(parent)
        self.path = str(path)
        self.cad_shape = cad_shape
        self.recovered_shape = recovered_shape
        self.topology_result = dict(topology_result or {})
        self.structure_options = structure_options
        self.member_plan = member_plan
        self.export_format = str(export_format).lower()

    def _beam_lattice_surface(self):
        """Tessellate the analytic members, or None if this is not a strut lattice.

        A strut lattice has exact centrelines and radii, so its STL is written
        from those rather than from the recovered isosurface. The isosurface is
        a rasterization of the same members on the recovery grid: it carries
        that grid's terracing, and it is not what the viewer, STEP or 3MF show.
        One geometry, four outputs.
        """
        from pylcss.design_studio.topology_optimization.geometry.cad_reconstruction import (
            build_topopt_beam_lattice,
        )
        from pylcss.design_studio.topology_optimization.geometry.lattice_cad import (
            beam_lattice_triangles,
            lattice_cad_strategy,
        )

        if lattice_cad_strategy(self.structure_options) != "beam":
            return None
        if not self.topology_result:
            return None
        lattice = build_topopt_beam_lattice(
            self.topology_result,
            self.structure_options,
            self.member_plan,
        )
        vertices, faces = beam_lattice_triangles(lattice)
        if not len(faces):
            return None
        return vertices, faces, lattice

    def _export_stl(self, temporary: str) -> dict[str, object]:
        try:
            beam_surface = self._beam_lattice_surface()
        except Exception:
            # The rasterized recovery is still the optimized geometry; fall
            # back to it rather than failing an export the user asked for.
            logger.warning(
                "Analytic lattice STL unavailable; exporting the recovered "
                "surface instead.",
                exc_info=True,
            )
            beam_surface = None
        if beam_surface is not None:
            vertices, faces, lattice = beam_surface
            triangle_count = _write_binary_stl(temporary, vertices, faces)
            return {
                "format": "stl",
                "path": self.path,
                "triangles": triangle_count,
                "source": "analytic beam lattice",
                "beams": int(len(lattice.beams)),
                "recovered_shape": None,
            }

        if self.cad_shape is not None:
            import cadquery as cq

            shape = (
                self.cad_shape.val()
                if hasattr(self.cad_shape, "val")
                else self.cad_shape
            )
            tolerance = _tessellation_tolerance(shape)
            _, triangles = shape.tessellate(
                tolerance=tolerance,
                angularTolerance=STL_ANGULAR_TOLERANCE,
            )
            cq.exporters.export(
                shape,
                temporary,
                exportType="STL",
                tolerance=tolerance,
                angularTolerance=STL_ANGULAR_TOLERANCE,
            )
            return {
                "format": "stl",
                "path": self.path,
                "triangles": int(len(triangles)),
                "chordal_tolerance": tolerance,
                "recovered_shape": None,
            }

        recovered = self.recovered_shape
        if not isinstance(recovered, dict):
            raise ValueError("The topology result has no recovered mesh to export.")
        vertices = np.asarray(recovered.get("vertices"), dtype=float)
        faces = np.asarray(recovered.get("faces"), dtype=int)

        triangle_count = 0
        try:
            import trimesh

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            if not len(mesh.faces):
                raise ValueError("The recovered topology mesh has no triangles.")
            mesh.export(temporary, file_type="stl")
            triangle_count = int(len(mesh.faces))
        except Exception:
            logger.debug(
                "trimesh STL export failed; using the built-in writer",
                exc_info=True,
            )
            triangle_count = _write_binary_stl(temporary, vertices, faces)
        return {
            "format": "stl",
            "path": self.path,
            "triangles": triangle_count,
            "recovered_shape": recovered,
        }

    def _export_3mf(self, temporary: str) -> dict[str, object]:
        from pylcss.design_studio.topology_optimization.geometry.cad_reconstruction import (
            build_topopt_beam_lattice,
        )
        from pylcss.design_studio.topology_optimization.geometry.lattice_cad import (
            write_beam_lattice_3mf,
        )

        lattice = build_topopt_beam_lattice(
            self.topology_result,
            self.structure_options,
            self.member_plan,
        )
        write_beam_lattice_3mf(temporary, lattice)
        return {
            "format": "3mf",
            "path": self.path,
            "beams": int(len(lattice.beams)),
            "nodes": int(len(lattice.nodes)),
        }

    def run(self):
        temporary = ""
        try:
            temporary = _temporary_export_path(self.path)
            if self.export_format == "stl":
                details = self._export_stl(temporary)
            elif self.export_format == "3mf":
                details = self._export_3mf(temporary)
            else:
                raise ValueError(f"Unsupported topology export: {self.export_format}.")
            _commit_export(temporary, self.path)
            temporary = ""
            self.export_finished.emit(details)
        except Exception as exc:
            logger.exception("Topology %s export failed", self.export_format.upper())
            self.export_error.emit(str(exc))
        finally:
            if temporary and os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass


class TopOptStepExportWorker(QtCore.QThread):
    """Background worker for TopOpt STEP export."""

    export_finished = QtCore.Signal(str)
    export_error = QtCore.Signal(str)

    def __init__(
        self,
        topo_output,
        path,
        *,
        density_cutoff=0.30,
        extrusion_axis="none",
        passive_regions=None,
        reconstruction_strategy="auto",
        cached_shape=None,
        parent=None,
    ):
        super().__init__(parent)
        self.topo_output = dict(topo_output or {})
        self.path = str(path)
        self.density_cutoff = float(density_cutoff or 0.30)
        self.extrusion_axis = str(extrusion_axis or "none").strip().lower()
        self.passive_regions = dict(passive_regions or {})
        self.reconstruction_strategy = str(
            reconstruction_strategy or "auto"
        ).strip().lower()
        self.cached_shape = cached_shape

    @staticmethod
    def _bounds_tuple(bounds_payload):
        if (
            isinstance(bounds_payload, dict)
            and "min" in bounds_payload
            and "max" in bounds_payload
        ):
            mins = np.asarray(bounds_payload["min"], dtype=float)
            maxs = np.asarray(bounds_payload["max"], dtype=float)
            if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
                return mins[:3], maxs[:3]
        return None

    def run(self):
        temporary = ""
        try:
            temporary = _temporary_export_path(self.path)
            product_name = Path(self.path).stem
            if self.cached_shape is not None:
                _write_step_shape(self.cached_shape, temporary, product_name)
            else:
                payload = _topopt_cad_payload(
                    self.topo_output,
                    density_cutoff=self.density_cutoff,
                    extrusion_axis=self.extrusion_axis,
                    passive_regions=self.passive_regions,
                )

                from concurrent.futures import ProcessPoolExecutor

                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        _external_write_cad_step,
                        payload,
                        temporary,
                        self.reconstruction_strategy,
                        product_name,
                    )
                    future.result()

            _commit_export(temporary, self.path)
            temporary = ""
            self.export_finished.emit(self.path)
        except Exception as exc:
            logger.exception("Topology STEP export failed")
            self.export_error.emit(str(exc))
        finally:
            if temporary and os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
