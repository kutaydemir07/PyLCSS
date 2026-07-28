# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchProjectMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime

from PySide6 import QtCore, QtWidgets


logger = logging.getLogger(__name__)

__all__ = ["WorkbenchProjectMixin"]


class WorkbenchProjectMixin:
    def _new_project(self):
        """Create a new project."""
        if not self._ensure_idle_for_io("creating a new project"):
            return

        self.graph.clear_session()
        self.current_file = None
        self._last_rendered_node = None
        try:
            self.viewer.clear()
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
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

    def load_project_file(self, project_file, *, preview=False):
        """Load a Design Studio graph and its safe numerical result sidecar.

        This non-dialog entry point is also used by the application-level
        project loader.  Loading is transactional: an invalid file leaves the
        current editable graph in place.
        """
        if not self._ensure_idle_for_io("opening a project"):
            raise RuntimeError("A Design Studio computation is still running.")

        fname = os.path.abspath(os.fspath(project_file))
        previous_file = self.current_file
        try:
            backup_session = self.graph.serialize_session()
        except Exception:
            backup_session = None

        self._is_loading = True
        loaded_successfully = False
        restored_results = 0
        try:
            with open(fname, "r", encoding="utf-8") as handle:
                session_data = json.load(handle)
            if not isinstance(session_data, dict) or "nodes" not in session_data:
                raise ValueError("The file is not a valid Design Studio graph.")

            self.graph.clear_session()
            self.graph.deserialize_session(session_data)
            self.current_file = fname
            self._set_project_context(fname)
            restored_results = self._restore_project_results(fname, session_data)
            self._last_rendered_node = None
            try:
                self.viewer.clear()
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
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
            if backup_session is not None:
                try:
                    self.graph.clear_session()
                    self.graph.deserialize_session(backup_session)
                    self._set_project_context(previous_file)
                    self._fit_all()
                except Exception:
                    logger.exception("Could not restore graph after a failed load")
            raise
        finally:
            self._is_loading = False
        # Preview only after deserialisation has fully completed.  Scheduling
        # it for the next event-loop turn also lets NodeGraphQt finish its
        # queued port/selection notifications before graph execution starts.
        if loaded_successfully and preview:
            QtCore.QTimer.singleShot(
                0, lambda: self._execute_graph(skip_simulation=True)
            )
        return restored_results

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
            self._last_rendered_node = None
            try:
                self.viewer.clear()
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
            self.statusBar().showMessage(
                "Loaded legacy project (no Design Studio graph was stored)."
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

        message = f"Wait for the current computation to finish before {action_name}."
        self.statusBar().showMessage(message)
        QtWidgets.QMessageBox.information(self, "Computation In Progress", message)
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

            return load_results(sidecar, self.graph.all_nodes())
        except Exception:
            # A damaged or newer sidecar must not make the editable graph
            # impossible to open.
            logger.exception("Could not restore project result sidecar")
            return 0

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
            result_sidecar = self._results_sidecar_path(target)
            result_count = save_results(result_sidecar, self.graph.all_nodes())
            project_data = self.graph.serialize_session()
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
            f"Saved graph and {result_count} result set(s): {target}"
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
