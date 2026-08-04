# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchProjectMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime

from PySide6 import QtCore, QtWidgets


logger = logging.getLogger(__name__)

__all__ = ["WorkbenchProjectMixin"]


class WorkbenchProjectMixin:
    @staticmethod
    def _yield_during_project_io():
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ExcludeUserInputEvents
        )

    def _open_example(self):
        """Choose a bundled engineering workflow with its scope visible."""
        if not self._ensure_idle_for_io("opening an example"):
            return
        from pylcss.user_interface.example_browser import choose_bundled_example

        path = choose_bundled_example("design", self)
        if path is not None:
            self.load_project_file(path, preview=True, trusted=True)

    def _clear_project_caches(self):
        """Forget every UI/runtime reference owned by the previous project."""
        self._last_rendered_node = None
        self._last_rendered_geom = None
        self._last_topopt_preview_payload = None
        self._prefer_topopt_after_run = False
        self._pending_execute = None
        self.undo_stack.clear()
        self.redo_stack.clear()

        try:
            self.graph.clear_selection()
        except Exception:
            logger.debug("Could not clear Design Studio graph selection", exc_info=True)
        try:
            self.viewer.clear()
            self.viewer.clear_cached_results()
        except Exception:
            logger.debug("Could not clear Design Studio viewer state", exc_info=True)
        try:
            self.properties.display_node(None)
        except Exception:
            logger.debug("Could not clear Design Studio inspector state", exc_info=True)
        try:
            self.results.clear_results()
        except Exception:
            logger.debug("Could not clear Design Studio result state", exc_info=True)

        # The public runtime keeps a process-level cache for evaluations of
        # saved .cad files.  A project boundary is the one unambiguous point at
        # which those references should be released as well.
        try:
            from pylcss.design_studio.runtime import clear_cache

            clear_cache()
        except Exception:
            logger.debug("Could not clear Design Studio runtime cache", exc_info=True)

    def _show_restored_project_result(self):
        """Display the best numerical result restored from the sidecar."""
        node = self._find_renderable_simulation_node()
        if node is None:
            return False
        result = getattr(node, "_last_result", None)
        if not self._is_simulation_render_result(result):
            return False

        try:
            self.graph.clear_selection()
            node.set_selected(True)
        except Exception:
            logger.debug("Could not select the restored result node", exc_info=True)
        # Keep restoration deterministic if NodeGraphQt coalesces the selection
        # signal while the project-load event is still unwinding.
        self._on_node_selected(node)
        return True

    def _new_project(self):
        """Create a new project."""
        if not self._ensure_idle_for_io("creating a new project"):
            return

        self.graph.clear_session()
        self.current_file = None
        self._project_code_trusted = True
        self._clear_project_caches()
        self.timeline.add_event("New project created")
        self.statusBar().showMessage("New project")

    def _open_project(self):
        """Open a project file."""
        if not self._ensure_idle_for_io("opening a project"):
            return

        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Project", "", "Design Projects (*.cad);;All Files (*)"
        )
        if fname:
            try:
                self.load_project_file(fname, preview=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Failed to open project: {e}"
                )

    def load_project_file(self, project_file, *, preview=False, trusted=False):
        """Load a Design Studio graph and its safe numerical result sidecar.

        This non-dialog entry point is also used by the application-level
        project loader.  Loading is transactional: an invalid file leaves the
        current editable graph in place.
        """
        if not self._ensure_idle_for_io("opening a project"):
            raise RuntimeError("A Design Studio computation is still running.")

        fname = os.path.abspath(os.fspath(project_file))
        previous_file = self.current_file
        previous_trust = self._project_code_trusted
        try:
            backup_session = self.graph.serialize_session()
        except Exception:
            backup_session = None

        self._is_loading = True
        loaded_successfully = False
        restored_results = 0
        try:
            from pylcss.design_studio.session_persistence import (
                parse_design_studio_session,
            )

            with open(fname, "r", encoding="utf-8") as handle:
                session_data = parse_design_studio_session(handle.read())
            self._project_code_trusted = bool(
                trusted or not self._session_contains_executable_code(session_data)
            )
            from pylcss.design_studio.crash.conditions import (
                migrate_impact_scenario_properties,
            )
            from pylcss.design_studio.fem.mesh import (
                migrate_removed_mesher_properties,
            )

            migrate_impact_scenario_properties(session_data)
            migrate_removed_mesher_properties(session_data)
            from pylcss.design_studio.session_persistence import (
                expand_guided_topology_session,
                migrate_lattice_topology_nodes,
            )

            # Retype before expansion: which guided defaults a study is
            # hydrated with depends on whether it is a lattice study.
            migrate_lattice_topology_nodes(session_data)
            expand_guided_topology_session(session_data)

            self.graph.clear_session()
            self.graph.deserialize_session(session_data)
            # NodeGraphQt restores each node with its own stored dark palette
            # and ignores the QApplication theme, so nodes loaded from a file
            # stayed black-on-black in light mode until the user toggled the
            # theme. Re-theme immediately after deserialization.
            self._retheme_loaded_graph()
            self.current_file = fname
            self._set_project_context(fname)
            restored_results = self._restore_project_results(fname, session_data)
            self._clear_project_caches()
            try:
                self._fit_all()
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)

            self.timeline.add_event(f"Opened project: {fname}")
            result_suffix = (
                f" ({restored_results} saved result set(s) restored)"
                if restored_results
                else ""
            )
            self.statusBar().showMessage(f"Opened: {fname}{result_suffix}")
            loaded_successfully = True
        except Exception:
            self.current_file = previous_file
            self._project_code_trusted = previous_trust
            if backup_session is not None:
                try:
                    self.graph.clear_session()
                    self.graph.deserialize_session(backup_session)
                    self._retheme_loaded_graph()
                    self._set_project_context(previous_file)
                    self._fit_all()
                except Exception:
                    logger.exception("Could not restore graph after a failed load")
            raise
        finally:
            self._is_loading = False
        # Display a persisted result without executing any project code or
        # starting a solver. Scheduling for the next event-loop turn lets
        # NodeGraphQt finish its queued connection notifications first.
        if loaded_successfully and restored_results:
            QtCore.QTimer.singleShot(0, self._show_restored_project_result)

        # Preview only after deserialisation has fully completed.  Scheduling
        # it for the next event-loop turn also lets NodeGraphQt finish its
        # queued port/selection notifications before graph execution starts.
        #
        # This refresh is not something the user asked for, and nothing is
        # cached yet on a cold open, so it stays geometry-only: meshing an
        # entire model from scratch here is what made simply opening a project
        # look like it had started a solve.  Mesh and remesh run on Run, or on
        # the next interactive preview once the user edits something.
        if loaded_successfully and preview and not restored_results:
            QtCore.QTimer.singleShot(
                0,
                lambda: self._execute_graph(
                    skip_simulation=True, skip_meshing=True
                ),
            )
        return restored_results

    @staticmethod
    def _session_contains_executable_code(session_data):
        """Return whether a serialized graph contains a Python Code node."""
        nodes = session_data.get("nodes", {})
        if not isinstance(nodes, dict):
            return False
        return any(
            isinstance(node, dict)
            and str(node.get("type_", "")).endswith(".CadQueryCodeNode")
            for node in nodes.values()
        )

    def _confirm_project_code_execution(self):
        """Ask once before executing Python embedded in an external project."""
        if self._project_code_trusted:
            return True
        if not self.isVisible():
            logger.warning(
                "Blocked executable project code until the project is trusted"
            )
            return False

        reply = QtWidgets.QMessageBox.warning(
            self,
            "Project Contains Executable Python",
            "This Design Studio project contains one or more CadQuery Code "
            "nodes. Their Python code can access files, start processes, and "
            "perform other actions with your user permissions when the graph "
            "is previewed or solved.\n\nInspect the Code nodes first and execute "
            "only if you trust the project and its author.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            self.statusBar().showMessage("Project code was not executed")
            return False
        self._project_code_trusted = True
        return True

    def _retheme_loaded_graph(self):
        """Apply the active theme to nodes restored by ``deserialize_session``."""
        try:
            from pylcss.design_studio.core.port_schema import (
                apply_display_port_labels,
            )
            from pylcss.user_interface.common.theme_manager import (
                current_theme,
                retheme_node_graph,
            )

            retheme_node_graph(self.graph, current_theme())
            for node in self.graph.all_nodes():
                apply_display_port_labels(node)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

    def save_to_folder(self, folder_path):
        """Save the Design Studio graph/results into an application project."""
        folder = os.path.abspath(os.fspath(folder_path))
        os.makedirs(folder, exist_ok=True)
        return self.save_project_file(os.path.join(folder, "design_studio.cad"))

    def load_from_folder(self, folder_path):
        """Load Design Studio state from an application project folder."""
        project_file = os.path.join(
            os.path.abspath(os.fspath(folder_path)), "design_studio.cad"
        )
        if not os.path.isfile(project_file):
            # Older application projects did not include Design Studio.
            self.graph.clear_session()
            self.current_file = None
            self._clear_project_caches()
            self.statusBar().showMessage(
                "Legacy project: no Design Studio data."
            )
            return 0
        return self.load_project_file(project_file, preview=False)

    def _execution_is_active(self):
        return bool(self.worker and self.worker.isRunning())

    def _set_project_context(self, project_file=None):
        """Attach the saved-project directory used by relative-path nodes."""
        project_dir = (
            os.path.dirname(os.path.abspath(project_file)) if project_file else None
        )
        for node in self.graph.all_nodes():
            node._project_dir = project_dir

    def _ensure_idle_for_io(self, action_name):
        if not self._execution_is_active():
            return True

        message = f"Wait for the run to finish before {action_name}."
        self.statusBar().showMessage(message)
        QtWidgets.QMessageBox.information(self, "Run Active", message)
        return False

    @staticmethod
    def _results_sidecar_path(project_file):
        return os.path.abspath(str(project_file) + ".results.h5")

    def _restore_project_results(self, project_file, session_data):
        """Restore safe numerical results referenced by a .cad graph."""
        metadata = session_data.get("_result_store")
        if not isinstance(metadata, dict):
            return 0
        filename = str(metadata.get("file") or "")
        if not filename or os.path.basename(filename) != filename:
            logger.warning("Ignored unsafe project result-store path: %r", filename)
            return 0
        sidecar = os.path.join(os.path.dirname(os.path.abspath(project_file)), filename)
        if not os.path.isfile(sidecar):
            logger.warning("Project result sidecar is missing: %s", sidecar)
            return 0
        try:
            from pylcss.design_studio.result_store import load_results

            restored = load_results(
                sidecar,
                self.graph.all_nodes(),
                progress_callback=self._yield_during_project_io,
            )
            self._restore_project_cad_breps(project_file)
            return restored
        except Exception:
            # A damaged or newer sidecar must not make the editable graph
            # impossible to open.
            logger.exception("Could not restore project result sidecar")
            return 0

    TOPOPT_BREP_SUFFIX = ".topopt.brep"

    @staticmethod
    def _brep_node_token(node):
        """Filename-safe, save-stable token identifying one TopOpt node."""
        token = str(getattr(node, "id", "") or "")
        return re.sub(r"[^A-Za-z0-9_-]", "_", token) or "topopt"

    def _export_project_cad_breps(self, project_file):
        """Write each reconstructed TopOpt solid beside the project file.

        The result sidecar is HDF5 and cannot carry a live OpenCASCADE shape,
        so the solid travels as its own B-rep and the result keeps only its
        file name for ``_restore_project_cad_breps`` to read back.
        """
        target = os.path.abspath(os.fspath(project_file))
        project_dir = os.path.dirname(target)
        base = os.path.basename(target)
        exported = 0
        for node in self.graph.all_nodes():
            result = getattr(node, "_last_result", None)
            if not isinstance(result, dict) or result.get("type") != "topopt_voxel":
                continue
            shape = result.get("cad_shape")
            if shape is None:
                shape = result.get("shape")
            if shape is None:
                # Density-only results stay valid; drop any name left over from
                # an earlier save so the loader never chases a missing file.
                result.pop("cad_brep_file", None)
                continue

            filename = f"{base}.{self._brep_node_token(node)}{self.TOPOPT_BREP_SUFFIX}"
            destination = os.path.join(project_dir, filename)
            fd, temporary = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=project_dir
            )
            os.close(fd)
            try:
                solid = shape.val() if hasattr(shape, "val") else shape
                solid.exportBrep(temporary)
                os.replace(temporary, destination)
            except Exception:
                # A B-rep that cannot be written must not stop the project from
                # being saved; the density and recovered surface still persist.
                logger.exception("Could not export the TopOpt CAD B-rep")
                result.pop("cad_brep_file", None)
                continue
            finally:
                if os.path.exists(temporary):
                    try:
                        os.remove(temporary)
                    except OSError:
                        pass
            result["cad_brep_file"] = filename
            exported += 1
            self._yield_during_project_io()
        return exported

    def _restore_project_cad_breps(self, project_file):
        """Restore optional project-local B-reps referenced by TopOpt results."""
        project_dir = os.path.dirname(os.path.abspath(project_file))
        restored = 0
        for node in self.graph.all_nodes():
            result = getattr(node, "_last_result", None)
            if (
                not isinstance(result, dict)
                or result.get("type") != "topopt_voxel"
                or result.get("cad_shape") is not None
            ):
                continue
            filename = str(result.get("cad_brep_file") or "")
            if (
                not filename
                or os.path.basename(filename) != filename
                or not filename.lower().endswith(".brep")
            ):
                continue
            brep_path = os.path.join(project_dir, filename)
            if not os.path.isfile(brep_path):
                logger.warning("Saved TopOpt CAD B-rep is missing: %s", brep_path)
                continue
            try:
                import cadquery as cq

                shape = cq.importers.importBrep(brep_path)
                if not shape.solids().vals():
                    raise ValueError("the saved B-rep contains no solid")
                result["cad_shape"] = shape
                result["shape"] = shape
                result["cad_brep_restored"] = True
                restored += 1
            except Exception:
                # The density and recovered surface remain usable if a CAD
                # sidecar was deleted, damaged, or written by a newer OCC.
                logger.exception("Could not restore saved TopOpt CAD B-rep")
        return restored

    def _push_undo(self, action):
        try:
            self.undo_stack.append(action)
            # clear redo on new action
            self.redo_stack.clear()
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

    def _on_property_changed(self, node, prop_name, old, new):
        try:
            self._push_undo(
                {
                    "type": "prop_change",
                    "node": node,
                    "prop": prop_name,
                    "old": old,
                    "new": new,
                }
            )
            self.timeline.add_event(
                f"Property changed: {prop_name} = {new} ({getattr(node, 'name', '')})"
            )

            # IMPORTANT: When a property changes, the modified node should be rendered
            # Store it as last_rendered_node so _on_execution_finished renders the right node
            self._last_rendered_node = node

            # OPTIMIZATION: Check if this is a visualization-only property change
            # These properties don't need a re-run, just a re-render
            visualization_only_props = [
                "visualization",
                "deformation_scale",
                "disp_scale",
                "density_cutoff",
                "element_type",
            ]

            if prop_name in {"description", "tags", "notes"}:
                return

            if prop_name in visualization_only_props:
                # Cached-result updates and rendering are owned by the graph
                # property handler for both inspector and on-canvas edits.
                return
            # Auto-execute if enabled (for non-visualization properties)
            if True:
                # Property edits are previews.  Never launch a long CalculiX,
                # OpenRadioss, or TopOpt run implicitly; the explicit Run action
                # owns expensive engineering computation.
                self._execute_graph(skip_simulation=True)

        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

    def _save_project(self):
        """Save current project."""
        if not self._ensure_idle_for_io("saving the project"):
            return

        if not self.current_file:
            self._save_as_project()
            return

        try:
            self.save_project_file(self.current_file)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to save project: {e}"
            )

    def save_project_file(self, project_file):
        """Atomically save a graph and its compressed simulation-result sidecar."""
        if not self._ensure_idle_for_io("saving the project"):
            raise RuntimeError("A Design Studio computation is still running.")

        target = os.path.abspath(os.fspath(project_file))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        previous_file = self.current_file
        self._set_project_context(target)
        from pylcss.design_studio.result_store import save_results

        try:
            # Runs first: the exported file name is written into the result,
            # and the result is what the HDF5 sidecar below persists.
            self._export_project_cad_breps(target)
            result_sidecar = self._results_sidecar_path(target)
            result_count = save_results(
                result_sidecar,
                self.graph.all_nodes(),
                progress_callback=self._yield_during_project_io,
            )
            project_data = self.graph.serialize_session()
            from pylcss.design_studio.session_persistence import (
                compact_design_studio_session,
            )

            project_data = compact_design_studio_session(project_data)
            project_data = {
                **project_data,
                "_copyright": "Copyright (c) 2026 Kutay Demir.",
                "_license": (
                    "Licensed under the PolyForm Shield License 1.0.0. "
                    "See LICENSE file for details."
                ),
                "_result_store": {
                    "format": "HDF5",
                    "version": 1,
                    "file": os.path.basename(result_sidecar),
                    "result_count": result_count,
                    "saved_at": datetime.now().isoformat(),
                },
            }

            target_dir = os.path.dirname(target)
            fd, temp_path = tempfile.mkstemp(
                prefix="pylcss_cad_", suffix=".tmp", dir=target_dir
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(project_data, handle, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, target)
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
        except Exception:
            self.current_file = previous_file
            self._set_project_context(previous_file)
            raise

        self.current_file = target
        self.timeline.add_event(f"Saved project: {target}")
        self.statusBar().showMessage(
            f"Saved: {target} ({result_count} results)"
        )
        return result_count

    def _save_as_project(self):
        """Save project with a new name."""
        if not self._ensure_idle_for_io("saving the project"):
            return

        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Design Projects (*.cad);;All Files (*)"
        )
        if fname:
            # Ensure .cad extension
            if not fname.endswith(".cad"):
                fname += ".cad"
            self.current_file = fname
            self._set_project_context(fname)
            self._save_project()
