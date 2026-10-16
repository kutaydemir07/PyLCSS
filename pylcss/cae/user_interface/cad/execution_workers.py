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
    "TopOptCadDisplayWorker",
    "TopOptCadPreviewWorker",
    "TopOptLatticeDisplayWorker",
    "TopOptLatticeWorker",
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
        self._is_running = not self.isInterruptionRequested()
        try:
            from pylcss.cae.design_studio.engine import (
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
                cancel_callback=lambda: (
                    not self._is_running or self.isInterruptionRequested()
                ),
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


def _write_step_shape(shape, path, product_name, require_solid=True):
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
    in this interpreter — CAD export and the temporary body the
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
        from pylcss.cae.design_studio.topology_optimization.geometry.step_export import export_cad_model
        export_cad_model(solid, export_path=path, product_name=name,
                         require_solid=require_solid)
    finally:
        for key, value in previous.items():
            Interface_Static.SetCVal_s(key, value)
    return True


def _external_write_cad_step(
    payload,
    path,
    reconstruction_strategy="auto",
    product_name="",
):
    from pylcss.cae.design_studio.topology_optimization.geometry.cad_reconstruction import (
        reconstruct_topopt_cad,
    )

    shape, report = reconstruct_topopt_cad(
        payload,
        source_geometry="Recovered Shape",
        reconstruction_strategy=str(reconstruction_strategy or "auto"),
        sew_tolerance=1e-4,
        return_report=True,
    )
    from pylcss.cae.design_studio.topology_optimization.geometry.cad_config import require_step_export_ready
    require_step_export_ready(report)
    # The name comes from the caller's destination, not from ``path``: the
    # export writes to a temporary beside it and commits atomically, so
    # ``path`` here is ".bracket_a8f31c.step".
    return _write_step_shape(shape, path, product_name, require_solid=True)


def _external_write_cad_brep(payload, path, reconstruction_strategy="auto",
                             reconstruction_options=None, *, progress_callback=None):
    """Build the CAD body shared by viewing and export in an isolated process."""
    return _external_build_cad_brep(
        payload, path, reconstruction_strategy, reconstruction_options,
        progress_callback=progress_callback,
    )


def _external_build_cad_brep(payload, path, reconstruction_strategy,
                            reconstruction_options, *, progress_callback):
    from pylcss.cae.design_studio.topology_optimization.geometry.cad_reconstruction import (
        reconstruct_topopt_cad,
    )

    options = dict(reconstruction_options or {})
    options.pop("preview_only", None)
    options.update(source_geometry="Recovered Shape",
                   reconstruction_strategy=str(reconstruction_strategy or "auto"),
                   return_report=True, progress_callback=progress_callback)
    options.setdefault("sew_tolerance", 1e-4)
    shape, report = reconstruct_topopt_cad(payload, **options)
    if progress_callback:
        progress_callback({"type": "topopt_progress", "status_only": True,
                           "stage": "CAD reconstruction — Preparing display mesh"}, None, 0, 0)
    body = shape.val() if hasattr(shape, "val") else shape
    report["_preview_mesh"] = _external_prepare_cad_display(body)
    report["worker_pid"] = os.getpid()
    # Keep the transfer small: the prepared display mesh travels separately,
    # and importing BREP must not reimport OCC's triangulation as well.
    from OCP.BRepTools import BRepTools
    from OCP.TopTools import TopTools_FormatVersion_VERSION_1
    if not BRepTools.Write_s(body.wrapped, str(path), False, False,
                            TopTools_FormatVersion_VERSION_1):
        raise RuntimeError("Could not transfer the reconstructed CAD body")
    return report


def _external_prepare_cad_display(shape):
    """Prepare native bounds and tessellation only in the isolated CAD process."""
    from pylcss.cae.design_studio.topology_optimization.geometry.cad_tessellation import (
        tessellate_cad_absolute,
    )
    body = shape.val() if hasattr(shape, "val") else shape
    box = body.BoundingBox()
    tolerance = max(float(np.linalg.norm([box.xlen, box.ylen, box.zlen])) * 1e-3, 1e-7)
    vertices, faces, face_ranges = tessellate_cad_absolute(body, tolerance, .3, return_face_ranges=True)
    return {"vertices": vertices, "faces": faces, "face_ranges": face_ranges}


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


_LATTICE_INPUT_REFERENCE = "__pylcss_lattice_input_reference__"


def _lattice_job_input(result):
    """Send generation inputs, leaving analysis and viewer caches in Qt.

    Keep omitted keys as placeholders so the child's result delta still
    removes obsolete analysis fields when it is applied to the original solve.
    Unchanged metadata is restored from the parent, never from placeholders.
    """
    from pylcss.cae.design_studio.topology_optimization.integration.post_lattice import topology_source

    source = topology_source(result)
    required = {
        "type", "density", "recovery_density", "design_density", "bounds",
        "recovery_cutoff", "density_cutoff", "passive_solid_mask",
        "passive_void_mask", "passive_regions", "cad_shape", "shape",
        "source_cad_shape", "mass", "volume", "source_volume",
    }
    payload = {key: value if key in required else None for key, value in source.items()}
    recovered = source.get("recovered_shape")
    if isinstance(recovered, dict):
        payload["recovered_shape"] = _lattice_display_input(recovered)
    return payload


def _lattice_display_input(recovered):
    """Display preparation consumes triangles, never CAD or analysis caches."""
    return {key: recovered[key] for key in ("vertices", "faces") if key in recovered}


def _lattice_native_reference_paths(value, path=(), references=None):
    """Identify inherited CAD objects without transferring them back to Qt."""
    import cadquery as cq
    references = {} if references is None else references
    if isinstance(value, (cq.Shape, cq.Workplane)):
        references.setdefault(id(value), path)
    elif isinstance(value, dict):
        for key, item in value.items():
            _lattice_native_reference_paths(item, path + (key,), references)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _lattice_native_reference_paths(item, path + (index,), references)
    return references


def _encode_lattice_references(value, references):
    import cadquery as cq
    if isinstance(value, (cq.Shape, cq.Workplane)):
        if id(value) not in references:
            raise RuntimeError("Lattice generation returned an unexpected native CAD object")
        return {_LATTICE_INPUT_REFERENCE: references[id(value)]}
    if isinstance(value, dict):
        return {key: _encode_lattice_references(item, references) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_lattice_references(item, references) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_lattice_references(item, references) for item in value)
    return value


def _decode_lattice_references(value, source):
    if isinstance(value, dict):
        if set(value) == {_LATTICE_INPUT_REFERENCE}:
            original = source
            for key in value[_LATTICE_INPUT_REFERENCE]:
                original = original[key]
            return original
        return {key: _decode_lattice_references(item, source) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_lattice_references(item, source) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_lattice_references(item, source) for item in value)
    return value


def _external_generate_lattice(result, settings, *, progress_callback=None):
    """Generate physical cells in the isolated process; return a data delta."""
    from pylcss.cae.design_studio.topology_optimization.integration.post_lattice import (
        generate_post_topopt_lattice, topology_source,
    )
    source = topology_source(result)
    output = generate_post_topopt_lattice(result, settings, progress_callback=progress_callback)
    # Prepare the bounded preview before the full manufacturing mesh leaves
    # the generation process. Keep native display filters in their own child:
    # a reducer crash must not discard the completed manufacturing mesh.
    from .cad_background import run_cad_job
    try:
        output["lattice_preview_mesh"] = run_cad_job(
            "lattice_display",
            _lattice_display_input(output["recovered_shape"]),
            progress_callback=progress_callback,
        )
    except Exception as exc:
        # Preview failure must not discard a valid manufacturing/export mesh.
        output["_lattice_display_error"] = str(exc)
        logger.exception("Could not prepare lattice display")
    # Retain the application's original CAD context. Reimporting an unchanged
    # fitted body through OCC's binary pickle is unnecessary and can fail.
    updates = {key: value for key, value in output.items()
               if key != "topology_source" and (key not in source or value is not source[key])}
    return {
        "updates": _encode_lattice_references(updates, _lattice_native_reference_paths(source)),
        "removed_keys": tuple(key for key in source if key not in output),
        "worker_pid": os.getpid(),
    }


def _external_prepare_lattice_display(recovered, *, progress_callback=None):
    from pylcss.cae.design_studio.topology_optimization.geometry.lattice_display import prepare_lattice_display
    if progress_callback:
        progress_callback({'status_only': True, 'stage': 'Preparing lattice display'}, None, 0, 0)
    return prepare_lattice_display(recovered)


def _restore_lattice_output(result, transfer):
    from pylcss.cae.design_studio.topology_optimization.integration.post_lattice import (
        _copy_containers, topology_source,
    )
    source = topology_source(result)
    output = dict(source)
    for key in transfer["removed_keys"]:
        output.pop(key, None)
    output.update(_decode_lattice_references(transfer["updates"], source))
    archived = _copy_containers(source)
    output["topology_source"] = archived
    output["generated_lattice"]["worker_pid"] = transfer["worker_pid"]
    return output


class TopOptLatticeWorker(QtCore.QThread):
    """Generate cells from a saved solve while the inspector stays responsive."""

    lattice_finished = QtCore.Signal(object)
    lattice_error = QtCore.Signal(str)
    lattice_cancelled = QtCore.Signal()
    lattice_progress = QtCore.Signal(object, object, int, int)

    def __init__(self, result, settings, parent=None):
        super().__init__(parent)
        self.result = result
        self.settings = dict(settings)

    def run(self):
        from .cad_background import CadJobCancelled, run_cad_job
        try:
            if self.isInterruptionRequested():
                raise CadJobCancelled("Lattice generation cancelled")
            transfer = run_cad_job(
                "lattice", _lattice_job_input(self.result), self.settings,
                cancel_callback=self.isInterruptionRequested,
                progress_callback=self.lattice_progress.emit,
            )
            if self.isInterruptionRequested():
                raise CadJobCancelled("Lattice generation cancelled")
            output = _restore_lattice_output(self.result, transfer)
            if self.isInterruptionRequested():
                raise CadJobCancelled("Lattice generation cancelled")
            self.lattice_finished.emit(output)
        except (InterruptedError, CadJobCancelled):
            self.lattice_cancelled.emit()
        except Exception as exc:
            logger.exception("TopOpt lattice generation failed")
            self.lattice_error.emit(str(exc))


class TopOptLatticeDisplayWorker(QtCore.QThread):
    """Prepare a saved lattice's missing display cache outside Qt/native state."""

    display_finished = QtCore.Signal(object)
    display_error = QtCore.Signal(str)
    display_cancelled = QtCore.Signal()

    def __init__(self, recovered, *, parent=None):
        super().__init__(parent)
        self.recovered = recovered

    def run(self):
        from .cad_background import CadJobCancelled, run_cad_job
        try:
            if self.isInterruptionRequested():
                raise CadJobCancelled('Lattice display cancelled')
            display = run_cad_job('lattice_display', _lattice_display_input(self.recovered),
                                  cancel_callback=self.isInterruptionRequested)
            if self.isInterruptionRequested():
                raise CadJobCancelled('Lattice display cancelled')
            self.display_finished.emit(display)
        except CadJobCancelled:
            self.display_cancelled.emit()
        except Exception as exc:
            self.display_error.emit(str(exc))


class TopOptCadDisplayWorker(QtCore.QThread):
    """Restore a missing display cache without rebuilding or meshing on Qt."""

    display_finished = QtCore.Signal(object)
    display_error = QtCore.Signal(str)
    display_cancelled = QtCore.Signal()
    display_progress = QtCore.Signal(object, object, int, int)

    def __init__(self, shape, *, parent=None):
        super().__init__(parent)
        self.shape = shape
        self.preview_mesh = None

    def run(self):
        from .cad_background import CadJobCancelled, run_cad_job
        try:
            if self.isInterruptionRequested():
                raise CadJobCancelled("CAD display cancelled")
            self.display_progress.emit(
                {"type": "topopt_progress", "status_only": True,
                 "stage": "Preparing CAD display in the background"}, None, 0, 0)
            self.preview_mesh = run_cad_job(
                "display_mesh", self.shape, cancel_callback=self.isInterruptionRequested)
            self.display_finished.emit(self.preview_mesh)
        except CadJobCancelled:
            self.display_cancelled.emit()
        except Exception as exc:
            self.display_error.emit(str(exc))


class TopOptCadPreviewWorker(QtCore.QThread):
    """Build the real TopOpt CAD body without blocking the viewer."""

    preview_finished = QtCore.Signal(object, object)
    preview_error = QtCore.Signal(str)
    preview_cancelled = QtCore.Signal()
    preview_progress = QtCore.Signal(object, object, int, int)

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
        self.preview_mesh = None

    def run(self):
        from .cad_background import CadJobCancelled, reconstruct_cad_in_background
        try:
            payload = _topopt_cad_payload(
                self.topo_output,
                density_cutoff=self.density_cutoff,
                extrusion_axis=self.extrusion_axis,
                passive_regions=self.passive_regions,
            )
            shape, report, self.preview_mesh = reconstruct_cad_in_background(
                payload, reconstruction_strategy=self.reconstruction_strategy,
                progress_callback=self.preview_progress.emit,
                cancel_callback=self.isInterruptionRequested,
            )
            self.preview_finished.emit(shape, report)
        except CadJobCancelled:
            self.preview_cancelled.emit()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.preview_error.emit(str(exc))


class TopOptMeshExportWorker(QtCore.QThread):
    """Coordinate isolated topology STL/3MF export without blocking Qt."""

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

    def _export_stl(self, temporary: str) -> dict[str, object]:
        # The manufacturing boundary includes clipped members, joined material,
        # skins and passive interfaces. Export it directly instead of replacing
        # it with intersecting capped cylinders or refitting it through STEP.
        recovered = self.recovered_shape
        if not isinstance(recovered, dict):
            recovered = self.topology_result.get("recovered_shape")
        if isinstance(recovered, dict):
            self.recovered_shape = recovered
        if self.cad_shape is not None and not isinstance(recovered, dict) and not self.topology_result:
            from pylcss.cae.design_studio.topology_optimization.geometry.cad_tessellation import (
                tessellate_cad_absolute,
            )

            shape = (
                self.cad_shape.val()
                if hasattr(self.cad_shape, "val")
                else self.cad_shape
            )
            tolerance = _tessellation_tolerance(shape)
            vertices, triangles = tessellate_cad_absolute(shape, tolerance, STL_ANGULAR_TOLERANCE)
            # OCC emits zero-area pole triangles on otherwise valid spheres
            # and other singular parameterizations. Remove only triangles that
            # collapse at STL precision, then require a closed oriented mesh.
            import trimesh

            vertices = np.asarray(vertices, dtype=np.float32).astype(float)
            sites = vertices[triangles]
            keep = np.linalg.norm(np.cross(sites[:, 1] - sites[:, 0],
                                           sites[:, 2] - sites[:, 0]), axis=1) > 1e-12
            removed = int(np.count_nonzero(~keep))
            mesh = trimesh.Trimesh(vertices, triangles[keep], process=True)
            if not mesh.is_watertight or not mesh.is_winding_consistent:
                raise ValueError("CAD tessellation did not produce a closed, consistently oriented STL mesh.")
            vertices, triangles = mesh.vertices, mesh.faces
            triangle_count = _write_binary_stl(temporary, vertices, triangles)
            return {
                "format": "stl",
                "path": self.path,
                "triangles": triangle_count,
                "chordal_tolerance": tolerance,
                "relative_occ_deflection": False,
                "removed_degenerate_triangles": removed,
                "recovered_shape": None,
            }

        recovered = self.recovered_shape
        if not isinstance(recovered, dict) and self.topology_result:
            field = (
                self.topology_result.get("recovery_density")
                if self.topology_result.get("recovery_density") is not None
                else self.topology_result.get("density")
            )
            if field is not None:
                from pylcss.cae.design_studio.topology_optimization.geometry.cad_features import _payload_bounds
                from pylcss.cae.design_studio.topology_optimization.geometry.cad_reconstruction import _structure_options

                bounds = _payload_bounds(self.topology_result)
                cutoff = float(
                    self.topology_result.get("recovery_cutoff")
                    or self.topology_result.get("density_cutoff")
                    or 0.5
                )
                try:
                    from pylcss.cae.design_studio.topology_optimization.geometry.surface_recovery import (
                        _recover_voxel_shape,
                    )
                    from pylcss.cae.design_studio.topology_optimization.geometry.recovery_payload import (
                        surface_recovery_constraints,
                    )
                    recovered = _recover_voxel_shape(
                        np.asarray(field, dtype=float),
                        bounds,
                        cutoff,
                        **surface_recovery_constraints(self.topology_result),
                        extrusion_axis=self.topology_result.get("extrusion_axis") or "none",
                        structure_options=_structure_options(self.topology_result, self.structure_options),
                        member_plan=self.member_plan,
                    )
                    if isinstance(recovered, dict):
                        self.recovered_shape = recovered
                except Exception:
                    logger.debug("Failed on-demand surface recovery in export worker", exc_info=True)
        if not isinstance(recovered, dict):
            raise ValueError("The topology result has no recovered mesh to export.")
        vertices = np.asarray(recovered.get("vertices"), dtype=float)
        faces = np.asarray(recovered.get("faces"), dtype=int)

        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError("The recovered manufacturing boundary must be closed and consistently oriented.")
        reduction = recovered.get("mesh_reduction")
        if reduction is None:
            from pylcss.cae.design_studio.topology_optimization.geometry.mesh_reduction import compact_mesh
            vertices, faces, reduction = compact_mesh(
                mesh.vertices, mesh.faces, tolerance=float(np.linalg.norm(mesh.extents))*1e-3)
        else:
            vertices, faces = mesh.vertices, mesh.faces
        triangle_count = _write_binary_stl(temporary, vertices, faces)
        return {
            "format": "stl", "path": self.path, "triangles": triangle_count,
            "source": "Recovered manufacturing boundary", "mesh_reduction": reduction,
            "recovered_shape": recovered,
        }

    def _export_3mf(self, temporary: str) -> dict[str, object]:
        from pylcss.cae.design_studio.topology_optimization.geometry.cad_reconstruction import (
            build_topopt_beam_lattice,
        )
        from pylcss.cae.design_studio.topology_optimization.geometry.lattice_cad import (
            write_beam_lattice_3mf,
            topopt_export_formats,
        )

        if "3MF" not in topopt_export_formats(self.topology_result):
            raise ValueError("3MF members would omit preserved material. Export the complete body as STEP or STL.")
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
            from .cad_background import run_cad_job

            # OCC can retain the GIL in tessellation even inside QThread, and
            # cleaning a shared shape also invalidates the viewer's mesh.
            # Pickling gives the child its own CAD body and recovery arrays.
            details = run_cad_job("mesh", {
                "path": self.path,
                "cad_shape": self.cad_shape,
                "recovered_shape": self.recovered_shape,
                "topology_result": self.topology_result,
                "structure_options": self.structure_options,
                "member_plan": self.member_plan,
                "export_format": self.export_format,
            }, temporary, cancel_callback=self.isInterruptionRequested)
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
        self.reconstruction_report = None
        self.preview_mesh = None

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
            from pylcss.cae.design_studio.topology_optimization.geometry.lattice_cad import topopt_export_formats
            from pylcss.cae.design_studio.topology_optimization.geometry.cad_config import require_step_export_ready
            if "STEP" not in topopt_export_formats(self.topo_output):
                raise ValueError("Export this complete lattice body as STL. A supported CAD representation including all its material is not available.")
            if self.cached_shape is not None:
                require_step_export_ready(self.topo_output.get("cad_reconstruction"))
            temporary = _temporary_export_path(self.path)
            product_name = Path(self.path).stem
            from .cad_background import run_cad_job

            if self.cached_shape is not None:
                # Cached exports need the same isolation as a fresh fit: STEP
                # transfer, validation and OCC's global settings can stall Qt.
                run_cad_job(
                    "step_shape", self.cached_shape, temporary, product_name,
                    True,
                    cancel_callback=self.isInterruptionRequested,
                )
            else:
                payload = _topopt_cad_payload(
                    self.topo_output,
                    density_cutoff=self.density_cutoff,
                    extrusion_axis=self.extrusion_axis,
                    passive_regions=self.passive_regions,
                )

                # An export can be the first request for CAD. Retain that
                # constructed body so the next export uses the same cache as a
                # solve or CAD preview, instead of fitting it again.
                from .cad_background import reconstruct_cad_in_background
                self.cached_shape, self.reconstruction_report, self.preview_mesh = reconstruct_cad_in_background(
                    payload, reconstruction_strategy=self.reconstruction_strategy,
                    cancel_callback=self.isInterruptionRequested,
                )
                require_step_export_ready(self.reconstruction_report)
                run_cad_job(
                    "step_shape", self.cached_shape, temporary, product_name,
                    True,
                    cancel_callback=self.isInterruptionRequested,
                )

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


def _external_write_cad_mesh(options, temporary):
    """Run mesh recovery/tessellation only in the private CAD subprocess."""
    exporter = TopOptMeshExportWorker(**options)
    if exporter.export_format == "stl":
        details = exporter._export_stl(temporary)
    elif exporter.export_format == "3mf":
        details = exporter._export_3mf(temporary)
    else:
        raise ValueError(f"Unsupported topology export: {exporter.export_format}.")
    details["worker_pid"] = os.getpid()
    return details
