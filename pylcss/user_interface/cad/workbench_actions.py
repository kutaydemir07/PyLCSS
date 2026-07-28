# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""WorkbenchActionMixin behavior for the Design Studio workbench."""

from __future__ import annotations

import logging
from datetime import datetime

from PySide6 import QtWidgets


logger = logging.getLogger(__name__)

__all__ = ["WorkbenchActionMixin"]


class WorkbenchActionMixin:
    def _undo(self):
        """Undo last action."""
        if not self.undo_stack:
            self.statusBar().showMessage("Nothing to undo")
            return
        action = self.undo_stack.pop()
        try:
            typ = action.get("type")
            if typ == "add_node":
                node = action.get("node")
                # remove node
                try:
                    self.graph.remove_node(node)
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
                # push redo
                self.redo_stack.append({"type": "add_node", "node": node})
                self.timeline.add_event(f"Undid add node {getattr(node, 'name', '')}")
            elif typ == "remove_nodes":
                nodes = action.get("nodes", [])
                for n in nodes:
                    try:
                        self.graph.add_node(n)
                        # restore position if available
                        try:
                            pos = action.get("positions", {}).get(id(n))
                            if pos:
                                n.set_pos(pos[0], pos[1])
                        except Exception:
                            logger.debug("Optional UI operation failed.", exc_info=True)
                    except Exception:
                        logger.debug("Optional UI operation failed.", exc_info=True)
                self.redo_stack.append(action)
                self.timeline.add_event(f"Undid delete of {len(nodes)} node(s)")
            elif typ == "prop_change":
                node = action.get("node")
                prop = action.get("prop")
                old = action.get("old")
                new = action.get("new")
                try:
                    node.set_property(prop, old)
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
                # push redo
                self.redo_stack.append(
                    {
                        "type": "prop_change",
                        "node": node,
                        "prop": prop,
                        "old": new,
                        "new": old,
                    }
                )
                self.timeline.add_event(
                    f"Undid property {prop} on {getattr(node, 'name', 'node')}"
                )
            else:
                self.timeline.add_event("Unknown undo action")
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        self.statusBar().showMessage("Undo")

    def _redo(self):
        """Redo last action."""
        if not self.redo_stack:
            self.statusBar().showMessage("Nothing to redo")
            return
        action = self.redo_stack.pop()
        try:
            typ = action.get("type")
            if typ == "add_node":
                node = action.get("node")
                try:
                    self.graph.add_node(node)
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
                self.undo_stack.append(action)
                self.timeline.add_event(f"Redid add node {getattr(node, 'name', '')}")
            elif typ == "remove_nodes":
                nodes = action.get("nodes", [])
                action.get("positions", {})
                for n in nodes:
                    try:
                        self.graph.remove_node(n)
                    except Exception:
                        logger.debug("Optional UI operation failed.", exc_info=True)
                self.undo_stack.append(action)
                self.timeline.add_event(f"Redid delete of {len(nodes)} node(s)")
            elif typ == "prop_change":
                node = action.get("node")
                prop = action.get("prop")
                new = action.get("new")
                try:
                    node.set_property(prop, new)
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
                self.undo_stack.append(action)
                self.timeline.add_event(
                    f"Redid property {prop} on {getattr(node, 'name', 'node')}"
                )
            else:
                self.timeline.add_event("Unknown redo action")
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        self.statusBar().showMessage("Redo")

    def _delete_selected(self):
        """Delete selected nodes."""
        selected = list(self.graph.selected_nodes())
        if not selected:
            return

        # record positions so undo can restore
        positions = {id(n): n.pos() for n in selected}
        try:
            self._push_undo(
                {"type": "remove_nodes", "nodes": selected, "positions": positions}
            )
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

        for node in selected:
            try:
                self.graph.remove_node(node)
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
        self.timeline.add_event(f"Deleted {len(selected)} selected nodes")

    def _clear_graph(self):
        """Clear entire graph."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear All",
            "Remove all nodes?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            # record all nodes for undo
            all_nodes = list(self.graph.all_nodes())
            positions = {id(n): n.pos() for n in all_nodes}
            try:
                self._push_undo(
                    {"type": "remove_nodes", "nodes": all_nodes, "positions": positions}
                )
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
            self.graph.clear_session()
            self._last_rendered_node = None
            try:
                self.viewer.clear()
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
            self.timeline.add_event("Cleared graph")

    def _fit_all(self):
        """Fit all nodes in view."""
        try:
            # Fit the node graph view
            self.graph.fit_to_selection()
            # If no selection, center on all nodes
            if not self.graph.selected_nodes():
                self.graph.center_on_nodes(self.graph.all_nodes())
            self.statusBar().showMessage("Fit to view")
        except Exception:
            # Fallback - try basic centering
            try:
                self.graph.center_selection()
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)
            self.statusBar().showMessage("View adjusted")

    def _reset_view(self):
        """Reset the 3D viewer to default orientation."""
        try:
            # Reset the 3D viewer camera
            if hasattr(self.viewer, "renderer") and self.viewer.renderer:
                self.viewer.renderer.ResetCamera()
                if hasattr(self.viewer, "iren") and self.viewer.iren:
                    self.viewer.iren.GetRenderWindow().Render()
            self.statusBar().showMessage("3D view reset")
            self.timeline.add_event("3D view reset to default")
        except Exception:
            self.statusBar().showMessage("View reset")

    def _generate_report(self):
        """Generate a report from the model with node information."""
        self.statusBar().showMessage("Generating report...")
        self.timeline.add_event("Report generation started")

        # Collect model information
        all_nodes = list(self.graph.all_nodes())
        if not all_nodes:
            QtWidgets.QMessageBox.information(
                self, "Empty Model", "No nodes in the graph to report on."
            )
            self.statusBar().showMessage("No nodes to report")
            return

        # Build report content
        report_lines = [
            "=" * 60,
            "CAD MODEL REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Project: {self.current_file or 'Unsaved'}",
            "",
            f"Total Nodes: {len(all_nodes)}",
            "",
            "NODE SUMMARY:",
            "-" * 40,
        ]

        # Categorize nodes
        node_types = {}
        for node in all_nodes:
            node_class = node.__class__.__name__
            node_types[node_class] = node_types.get(node_class, 0) + 1

        for node_type, count in sorted(node_types.items()):
            report_lines.append(f"  {node_type}: {count}")

        report_lines.extend(
            [
                "",
                "NODE DETAILS:",
                "-" * 40,
            ]
        )

        for node in all_nodes:
            report_lines.append(f"  [{node.__class__.__name__}] {node.name()}")
            # Add key properties
            try:
                props = node.model.properties
                for key, val in list(props.items())[:5]:  # First 5 properties
                    if not key.startswith("_"):
                        report_lines.append(f"      {key}: {val}")
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)

        report_lines.append("\n" + "=" * 60)
        report_text = "\n".join(report_lines)

        # Show in a dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Model Report")
        dialog.resize(600, 500)
        layout = QtWidgets.QVBoxLayout(dialog)

        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(report_text)
        text_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        layout.addWidget(text_edit)

        # Save button
        btn_layout = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Save Report...")
        close_btn = QtWidgets.QPushButton("Close")

        def save_report():
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(
                dialog, "Save Report", "model_report.txt", "Text Files (*.txt)"
            )
            if fname:
                with open(fname, "w") as f:
                    f.write(report_text)
                self.statusBar().showMessage(f"Report saved to {fname}")

        save_btn.clicked.connect(save_report)
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()
        self.statusBar().showMessage("Report generated")
        self.timeline.add_event("Report generated")

    def _validate_model(self):
        """Validate the current model for issues."""
        self.statusBar().showMessage("Validating model...")
        self.timeline.add_event("Model validation started")

        issues = []
        warnings = []

        all_nodes = list(self.graph.all_nodes())

        if not all_nodes:
            issues.append("Model is empty - no nodes found")
        else:
            # Check for disconnected nodes
            for node in all_nodes:
                has_input = False
                has_output = False

                for port in node.input_ports():
                    if port.connected_ports():
                        has_input = True
                        break

                for port in node.output_ports():
                    if port.connected_ports():
                        has_output = True
                        break

                # Code-first and scalar nodes don't need inputs
                node_class = node.__class__.__name__
                is_primitive = node_class in [
                    "CadQueryCodeNode",
                    "NumberNode",
                    "VariableNode",
                    "MaterialNode",
                    "CrashMaterialNode",
                    "RunRadiossDeckNode",
                ]
                is_export = "Export" in node_class

                if not is_primitive and not has_input:
                    warnings.append(f"{node.name()} has no connected inputs")

                if not is_export and not has_output:
                    # Check if it's a terminal node (not an issue)
                    if node_class not in ["SolverNode"]:
                        pass  # Non-terminal nodes without outputs are fine

            # Check for simulation setup
            has_mesh = any(
                n.__class__.__name__ in ("MeshNode", "RemeshNode") for n in all_nodes
            )
            has_solver = any(n.__class__.__name__ == "SolverNode" for n in all_nodes)
            has_material = any(
                n.__class__.__name__ == "MaterialNode" for n in all_nodes
            )
            has_constraint = any(
                n.__class__.__name__ == "ConstraintNode" for n in all_nodes
            )
            has_load = any(
                n.__class__.__name__ in ("LoadNode", "PressureLoadNode")
                for n in all_nodes
            )
            has_prescribed_displacement = any(
                n.__class__.__name__ == "ConstraintNode"
                and n.get_property("constraint_type") == "Displacement"
                and any(
                    n.get_property(f"displacement_{axis}_enabled") is not False
                    and abs(float(n.get_property(f"displacement_{axis}") or 0.0))
                    > 1e-15
                    for axis in ("x", "y", "z")
                )
                for n in all_nodes
            )

            if has_solver:
                if not has_mesh:
                    issues.append("Solver requires a Mesh node")
                if not has_material:
                    issues.append("Solver requires a Material node")
                if not has_constraint:
                    warnings.append("Solver may need constraint nodes (fixed supports)")
                if not has_load and not has_prescribed_displacement:
                    warnings.append("Solver may need load nodes")

        # Show results
        if not issues and not warnings:
            QtWidgets.QMessageBox.information(
                self,
                "Validation Complete",
                f"Model is valid.\n\nTotal nodes: {len(all_nodes)}",
            )
            self.statusBar().showMessage("Model valid")
        else:
            msg = ""
            if issues:
                msg += "ERRORS:\n" + "\n".join(f"  - {i}" for i in issues) + "\n\n"
            if warnings:
                msg += "WARNINGS:\n" + "\n".join(f"  - {w}" for w in warnings)

            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Validation Results")
            box.setText(f"Found {len(issues)} errors and {len(warnings)} warnings")
            box.setDetailedText(msg)
            box.setIcon(
                QtWidgets.QMessageBox.Warning
                if issues
                else QtWidgets.QMessageBox.Information
            )
            box.exec()

            if issues:
                self.statusBar().showMessage(f"{len(issues)} validation errors")
            else:
                self.statusBar().showMessage(f"Valid with {len(warnings)} warnings")

        self.timeline.add_event(
            f"Validation: {len(issues)} errors, {len(warnings)} warnings"
        )
