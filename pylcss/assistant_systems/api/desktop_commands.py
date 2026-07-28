# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Desktop navigation and application workflow commands."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pylcss.assistant_systems.services.input import MouseController

logger = logging.getLogger(__name__)


class DesktopCommandsMixin:
    """Route navigation, project, analysis, and desktop-control actions."""

    main_window: Any
    mouse: MouseController | None
    _on_pause: Callable[[], None] | None
    _on_resume: Callable[[], None] | None
    _run_sync: Callable[..., Any]

    def _require_mouse(self) -> MouseController:
        """Return input control or fail with an actionable message."""
        if self.mouse is None:
            raise RuntimeError(
                "Desktop input control is unavailable; graph tools remain usable"
            )
        return self.mouse

    def _handle_mouse_click(self, data: dict[str, Any]) -> None:
        """Handle mouse click command."""
        button = data.get("button", "left")
        self._require_mouse().click(button=button)

    def _handle_double_click(self, data: dict[str, Any]) -> None:
        """Handle double click command."""
        self._require_mouse().double_click()

    def _handle_drag_toggle(self, data: dict[str, Any]) -> None:
        """Handle drag/drop toggle."""
        self._require_mouse().toggle_drag()

    def _handle_scroll(self, data: dict[str, Any]) -> None:
        """Handle scroll command."""
        direction = data.get("direction", 3)
        self._require_mouse().scroll(direction)

    def _handle_scroll_horizontal(self, data: dict[str, Any]) -> None:
        """Handle horizontal scroll command."""
        direction = data.get("direction", 3)
        self._require_mouse().scroll_horizontal(direction)

    def _handle_switch_tab(self, data: dict[str, Any]) -> None:
        """Handle switch to specific tab."""
        tab_index = data.get("tab", 0)

        if self.main_window:

            def switch_tab() -> None:
                tab_widget = self._get_tab_widget()
                if tab_widget and 0 <= tab_index < tab_widget.count():
                    tab_widget.setCurrentIndex(tab_index)
                    logger.info(f"Switched to tab {tab_index}")

            self._run_sync(switch_tab)
        else:
            # Use keyboard shortcut as fallback
            # Ctrl+1 through Ctrl+9 for tabs
            if 0 <= tab_index <= 8:
                self._require_mouse().hotkey("ctrl", str(tab_index + 1))

    def _handle_next_tab(self, data: dict[str, Any]) -> None:
        """Handle next tab command."""
        if self.main_window:

            def next_tab() -> None:
                tab_widget = self._get_tab_widget()
                if tab_widget:
                    current = tab_widget.currentIndex()
                    next_idx = (current + 1) % tab_widget.count()
                    tab_widget.setCurrentIndex(next_idx)

            self._run_sync(next_tab)
        else:
            self._require_mouse().hotkey("ctrl", "tab")

    def _handle_previous_tab(self, data: dict[str, Any]) -> None:
        """Handle previous tab command."""
        if self.main_window:

            def previous_tab() -> None:
                tab_widget = self._get_tab_widget()
                if tab_widget:
                    current = tab_widget.currentIndex()
                    prev_idx = (current - 1) % tab_widget.count()
                    tab_widget.setCurrentIndex(prev_idx)

            self._run_sync(previous_tab)
        else:
            self._require_mouse().hotkey("ctrl", "shift", "tab")

    def _get_tab_widget(self) -> Any | None:
        """Get the main tab widget from PyLCSS window."""
        if not self.main_window:
            return None
        # MainWindow uses 'tabs', fallback to 'tab_widget'
        tab_widget = getattr(self.main_window, "tabs", None)
        if tab_widget is None:
            tab_widget = getattr(self.main_window, "tab_widget", None)
        return tab_widget

    def _handle_keyboard(self, data: dict[str, Any]) -> None:
        """Handle keyboard shortcut command."""
        keys = data.get("keys", [])
        if keys:
            self._require_mouse().hotkey(*keys)

    def _handle_pylcss_action(self, data: dict[str, Any]) -> None:
        """Handle PyLCSS-specific actions."""
        command = data.get("command", "")

        if not self.main_window:
            logger.warning("PyLCSS action requires main window reference")
            return

        action_map = {
            # Core actions
            "run_optimization": self._run_optimization,
            "stop_optimization": self._stop_optimization,
            "generate_samples": self._generate_samples,
            "train_surrogate": self._train_surrogate,
            "run_sensitivity": self._run_sensitivity,
            "new_project": self._new_project,
            "open_project": self._open_project,
            "export_results": self._export_results,
            "build_model": self._build_model,
            "build_node_graph": self._build_node_graph,
            "build_system_graph": self._build_system_graph,
            # Modeling environment - nodes
            "add_input": self._add_modeling_node,
            "add_output": self._add_modeling_node,
            "add_function": self._add_modeling_node,
            "add_intermediate": self._add_modeling_node,
            "validate_graph": self._validate_graph,
            # Modeling environment - system management
            "add_system": self._add_system,
            "remove_system": self._remove_system,
            "rename_system": self._rename_system,
            "next_system": self._next_system,
            "previous_system": self._previous_system,
            # Modeling environment - graph operations
            "auto_connect": self._auto_connect,
            "clear_graph": self._clear_graph,
            "select_all_nodes": self._select_all_nodes,
            "delete_selected": self._delete_selected,
            # Design Studio environment
            "cad_add_box": self._add_cad_node,
            "cad_add_cylinder": self._add_cad_node,
            "cad_add_sphere": self._add_cad_node,
            "cad_add_cone": self._add_cad_node,
            "cad_add_torus": self._add_cad_node,
            "cad_add_extrude": self._add_cad_node,
            "cad_add_fillet": self._add_cad_node,
            "cad_add_chamfer": self._add_cad_node,
            "cad_add_boolean": self._add_cad_node,
            "cad_add_union": self._add_cad_node,
            "cad_add_cut": self._add_cad_node,
            "cad_add_revolve": self._add_cad_node,
            "cad_execute": self._cad_execute_scoped,
            "cad_export": self._cad_export,
            # Solution space
            "resample": self._resample,
            "add_plot": self._add_plot,
            "clear_plots": self._clear_plots,
            "save_plots": self._save_plots,
            "configure_colors": self._configure_colors,
            "view_code": self._view_code,
            "compute_family": self._compute_family,
            "add_variant": self._add_variant,
            "remove_variant": self._remove_variant,
            "edit_variant": self._edit_variant,
            "compute_adg": self._compute_adg,
            # Surrogate training
            "refresh_nodes": self._refresh_nodes,
            "generate_training_data": self._generate_training_data,
            "browse_data_file": self._browse_data_file,
            "save_surrogate": self._save_surrogate,
            "stop_training": self._stop_training,
            "adaptive_training": self._adaptive_training,
            # Optimization
            "optimization_settings": self._optimization_settings,
            # Sensitivity
            "refresh_outputs": self._refresh_outputs,
            "export_sensitivity": self._export_sensitivity,
            "get_sensitivity": self._get_sensitivity_results,
            # Surrogate (extended)
            "train_surrogate_node": self._train_surrogate_node,
            # Granular Control
            "connect_nodes": self._connect_nodes,
            "set_property": self._set_property,
        }

        # Special handling for node creation commands that need the command name
        # Note: add_system, remove_system, rename_system are NOT node creation commands
        node_creation_commands = [
            "add_input",
            "add_output",
            "add_function",
            "add_intermediate",
        ]
        cad_node_commands = [c for c in action_map.keys() if c.startswith("cad_add_")]

        if command in node_creation_commands:
            self._add_modeling_node(command)
            return
        elif command in cad_node_commands:
            self._add_cad_node(command)
            return

        handler = action_map.get(command)
        if handler:
            # Special case for commands needing data
            if command == "build_node_graph":
                handler(data)
            elif command == "build_system_graph":
                handler(data)
            elif command == "train_surrogate_node":
                handler(data)
            elif command == "connect_nodes":
                handler(data)
            elif command == "set_property":
                handler(data)
            else:
                handler()
        else:
            logger.warning(f"Unknown PyLCSS action: {command}")

    def _run_optimization(self) -> None:
        """Trigger optimization run."""
        if not self.main_window or not hasattr(self.main_window, "optimization_widget"):
            return
        widget = self.main_window.optimization_widget
        if hasattr(widget, "btn_run"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_run, "click", Qt.QueuedConnection)
            logger.info("Assistant: Running optimization")
        elif hasattr(widget, "start_optimization"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget, "start_optimization", Qt.QueuedConnection)
            logger.info("Assistant: Starting optimization")

    def _stop_optimization(self) -> None:
        """Stop optimization."""
        if not self.main_window or not hasattr(self.main_window, "optimization_widget"):
            return
        widget = self.main_window.optimization_widget
        if hasattr(widget, "btn_stop"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_stop, "click", Qt.QueuedConnection)
            logger.info("Assistant: Stopping optimization")
        elif hasattr(widget, "stop_optimization"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget, "stop_optimization", Qt.QueuedConnection)

    def _generate_samples(self) -> None:
        """Generate samples in solution space."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_compute_feasible"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.btn_compute_feasible, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Computing solution space")
        elif hasattr(widget, "run_computation"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget, "run_computation", Qt.QueuedConnection)

    def _train_surrogate(self) -> None:
        """Train surrogate model."""
        if not self.main_window or not hasattr(self.main_window, "surrogate_widget"):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, "btn_train"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_train, "click", Qt.QueuedConnection)
            logger.info("Assistant: Training surrogate model")
        elif hasattr(widget, "start_training"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget, "start_training", Qt.QueuedConnection)

    def _run_sensitivity(self) -> None:
        """Run sensitivity analysis."""
        if not self.main_window or not hasattr(self.main_window, "sensitivity_widget"):
            return
        widget = self.main_window.sensitivity_widget
        if hasattr(widget, "btn_analyze"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_analyze, "click", Qt.QueuedConnection)
            logger.info("Assistant: Running sensitivity analysis")
        elif hasattr(widget, "run_analysis"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget, "run_analysis", Qt.QueuedConnection)

    def _new_project(self) -> None:
        """Create new project."""
        self._require_mouse().hotkey("ctrl", "n")

    def _open_project(self) -> None:
        """Open project dialog."""
        self._require_mouse().hotkey("ctrl", "o")

    def _export_results(self) -> None:
        """Export results."""
        self._require_mouse().hotkey("ctrl", "e")

    def _save_project(self) -> None:
        """Save the active project."""
        self._require_mouse().hotkey("ctrl", "s")

    def _build_model(self) -> None:
        """Build/transfer model from modeling environment."""
        if self.main_window:
            # Use QMetaObject.invokeMethod to call on main thread
            from PySide6.QtCore import QMetaObject, Qt

            if hasattr(self.main_window, "transfer_model"):
                QMetaObject.invokeMethod(
                    self.main_window, "transfer_model", Qt.QueuedConnection
                )

    def _train_surrogate_node(
        self, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Trigger surrogate training for a specific node."""
        if not self.main_window or not hasattr(self.main_window, "modeling_widget"):
            raise RuntimeError("The modeling workspace is unavailable")

        params = data.get("params", {}) if data else {}
        node_name = params.get("node_name")

        if not node_name:
            raise ValueError("A node name is required for surrogate training")

        def trigger_training() -> dict[str, Any]:
            widget = self.main_window.modeling_widget
            graph = widget.current_graph
            nodes = graph.all_nodes()
            node = next((item for item in nodes if item.name() == node_name), None)
            if node is None or not hasattr(node, "surrogate_widget"):
                raise ValueError(
                    f"Node {node_name!r} was not found or cannot train a surrogate"
                )
            node.surrogate_widget.btn_train.click()
            logger.info("Triggered training for node %s", node_name)
            return {"success": True, "node": node_name}

        return self._run_sync(trigger_training)

    def _get_sensitivity_results(self) -> str:
        """
        Retrieve sensitivity analysis results and inject them into LLM context.
        """
        if not self.main_window or not hasattr(self.main_window, "sensitivity_widget"):
            raise RuntimeError("Sensitivity results are unavailable")

        def fetch_results() -> str:
            widget = self.main_window.sensitivity_widget
            results = getattr(widget, "last_results", None)

            if not results:
                message = "No sensitivity results available. Please run analysis first."
            else:
                lines = ["**Sensitivity Analysis Results:**"]
                variables = results.get("variable_names", [])
                totals = results.get("total_order", [])
                combined = sorted(
                    zip(variables, totals), key=lambda item: item[1], reverse=True
                )
                lines.extend(
                    f"- {variable}: {total:.4f}" for variable, total in combined
                )
                message = "\n".join(lines)

            dialog = getattr(self.main_window, "_llm_dialog", None)
            if dialog is not None:
                if hasattr(dialog, "add_system_message"):
                    dialog.add_system_message(message)
                elif hasattr(dialog, "chat_widget") and hasattr(
                    dialog.chat_widget, "add_message"
                ):
                    dialog.chat_widget.add_message("System", message)

            logger.info("Sensitivity results retrieved")
            return message

        return self._run_sync(fetch_results)

    def _resample(self) -> None:
        """Resample the solution space."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_resample"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_resample, "click", Qt.QueuedConnection)
            logger.info("Assistant: Resampling")

    def _add_plot(self) -> None:
        """Add a new plot to the solution space."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_add_plot"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_add_plot, "click", Qt.QueuedConnection)
            logger.info("Assistant: Adding plot")

    def _clear_plots(self) -> None:
        """Clear all plots."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_clear_plots"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.btn_clear_plots, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Clearing plots")

    def _save_plots(self) -> None:
        """Save all plots."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_save_all"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_save_all, "click", Qt.QueuedConnection)
            logger.info("Assistant: Saving plots")

    def _configure_colors(self) -> None:
        """Open color configuration dialog."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_colors"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_colors, "click", Qt.QueuedConnection)
            logger.info("Assistant: Configuring colors")

    def _view_code(self) -> None:
        """View the generated code."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_view_code"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_view_code, "click", Qt.QueuedConnection)
            logger.info("Assistant: Viewing code")

    def _compute_family(self) -> None:
        """Compute product family solution space."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_compute_family"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.btn_compute_family, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Computing product family")

    def _add_variant(self) -> None:
        """Add a product variant."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_add_variant"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.btn_add_variant, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Adding variant")

    def _remove_variant(self) -> None:
        """Remove a product variant."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_remove_variant"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.btn_remove_variant, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Removing variant")

    def _edit_variant(self) -> None:
        """Edit variant requirements."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_edit_variant"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.btn_edit_variant, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Editing variant")

    def _compute_adg(self) -> None:
        """Generate Attribute Dependency Graph."""
        if not self.main_window or not hasattr(self.main_window, "sol_space_widget"):
            return
        widget = self.main_window.sol_space_widget
        if hasattr(widget, "btn_compute_adg"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.btn_compute_adg, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Computing ADG")

    def _refresh_nodes(self) -> None:
        """Refresh the node list in surrogate training."""
        if not self.main_window or not hasattr(self.main_window, "surrogate_widget"):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, "btn_refresh"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_refresh, "click", Qt.QueuedConnection)
            logger.info("Assistant: Refreshing nodes")

    def _generate_training_data(self) -> None:
        """Generate training data for surrogate model."""
        if not self.main_window or not hasattr(self.main_window, "surrogate_widget"):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, "btn_generate"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_generate, "click", Qt.QueuedConnection)
            logger.info("Assistant: Generating training data")

    def _browse_data_file(self) -> None:
        """Browse for data file."""
        if not self.main_window or not hasattr(self.main_window, "surrogate_widget"):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, "btn_browse"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_browse, "click", Qt.QueuedConnection)
            logger.info("Assistant: Browsing for data file")

    def _save_surrogate(self) -> None:
        """Save and attach surrogate model to node."""
        if not self.main_window or not hasattr(self.main_window, "surrogate_widget"):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, "btn_save"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_save, "click", Qt.QueuedConnection)
            logger.info("Assistant: Saving surrogate model")

    def _stop_training(self) -> None:
        """Stop surrogate model training."""
        if not self.main_window or not hasattr(self.main_window, "surrogate_widget"):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, "btn_stop"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_stop, "click", Qt.QueuedConnection)
            logger.info("Assistant: Stopping training")

    def _adaptive_training(self) -> None:
        """Start adaptive/active learning training."""
        if not self.main_window or not hasattr(self.main_window, "surrogate_widget"):
            return
        widget = self.main_window.surrogate_widget
        if hasattr(widget, "btn_adaptive"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_adaptive, "click", Qt.QueuedConnection)
            logger.info("Assistant: Starting adaptive training")

    def _optimization_settings(self) -> None:
        """Open optimization settings dialog."""
        if not self.main_window or not hasattr(self.main_window, "optimization_widget"):
            return
        widget = self.main_window.optimization_widget
        if hasattr(widget, "settings_widget") and hasattr(
            widget.settings_widget, "btn_settings"
        ):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.settings_widget.btn_settings, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Opening optimization settings")

    def _refresh_outputs(self) -> None:
        """Refresh outputs in sensitivity analysis."""
        if not self.main_window or not hasattr(self.main_window, "sensitivity_widget"):
            return
        widget = self.main_window.sensitivity_widget
        if hasattr(widget, "btn_refresh_outputs"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(
                widget.btn_refresh_outputs, "click", Qt.QueuedConnection
            )
            logger.info("Assistant: Refreshing outputs")

    def _export_sensitivity(self) -> None:
        """Export sensitivity analysis results."""
        if not self.main_window or not hasattr(self.main_window, "sensitivity_widget"):
            return
        widget = self.main_window.sensitivity_widget
        if hasattr(widget, "btn_export"):
            from PySide6.QtCore import QMetaObject, Qt

            QMetaObject.invokeMethod(widget.btn_export, "click", Qt.QueuedConnection)
            logger.info("Assistant: Exporting sensitivity results")

    def _handle_control(self, data: dict[str, Any]) -> None:
        """Handle control commands (pause, resume, etc.)."""
        command = data.get("command", "")

        callbacks = {
            "pause_tracking": self._on_pause,
            "resume_tracking": self._on_resume,
            "center_cursor": lambda: self._require_mouse().center_cursor(),
        }

        callback = callbacks.get(command)
        if callback:
            callback()
        else:
            logger.warning(f"Unknown control command: {command}")

    def _handle_window(self, data: dict[str, Any]) -> None:
        """Handle window control commands."""
        command = data.get("command", "")

        if not self.main_window:
            raise RuntimeError("Window control requires the PyLCSS main window")

        def control_window() -> None:
            if command == "minimize":
                self.main_window.showMinimized()
            elif command == "maximize":
                if self.main_window.isMaximized():
                    self.main_window.showNormal()
                else:
                    self.main_window.showMaximized()
            elif command == "close":
                self.main_window.close()
            else:
                raise ValueError(f"Unknown window command: {command!r}")

        self._run_sync(control_window)
