# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""NodeGraphQt node types for the system-modeling editor."""

from __future__ import annotations

import logging

from NodeGraphQt import BaseNode
from PySide6 import QtWidgets

from pylcss.user_interface.common.theme_manager import current_theme

from .editor import CodeEditorDialog, PortManagerDialog

__all__ = [
    "CustomBlockNode",
    "InputNode",
    "IntermediateNode",
    "OutputNode",
    "apply_system_node_style",
]


class CustomBlockNode(BaseNode):
    """
    Custom function block node for user-defined mathematical operations.

    Represents a black box function where users can define custom Python code
    to compute outputs from inputs. Supports dynamic port configuration and
    includes a built-in code editor with syntax highlighting.

    Node Properties:
        - num_inputs: Number of input ports (dynamically adjustable)
        - num_outputs: Number of output ports (dynamically adjustable)
        - code_content: Python code defining the function logic
        - func_name: Generated function name for code compilation
    """

    __identifier__ = "com.pfd.custom_block"
    NODE_NAME = "Function / Discipline"

    def __init__(self) -> None:
        """Initialize the custom block node with default ports and code editor."""
        super().__init__()
        # Role colour — neutral gray body with violet border for
        # compute/function blocks.
        self.set_color(35, 35, 35)
        self.set_property("border_color", (148, 107, 220, 255), push_undo=False)
        self.set_property("text_color", (242, 238, 250, 255), push_undo=False)
        self.set_port_deletion_allowed(True)
        # Start with default ports
        self.add_input("in_1")
        self.add_output("out_1")
        if not self.has_property("func_name"):
            self.create_property("func_name", "")
        # Create properties for number of inputs/outputs
        self.create_property("num_inputs", "1")
        self.create_property("num_outputs", "1")

        self.create_property("surrogate_model_path", "")
        self.create_property("surrogate_status", "Not Trained")
        self.create_property("surrogate_train_trigger", 0.0)
        self.create_property("surrogate_controls", False)
        self.create_property("code_content", "# out_1 = in_1 * 2\n")

        self.add_text_input("interface_summary", "Interface", text="1 input → 1 output")
        self.add_text_input("execution_summary", "Execution", text="Python function")
        for widget_name in ("interface_summary", "execution_summary"):
            widget = self.get_widget(widget_name)
            if widget:
                widget.get_custom_widget().setReadOnly(True)

        self.add_button(
            "edit_function",
            text="Edit function…",
            tooltip="Open the function editor, variables, and simulation coupling panel",
        )
        self.add_button(
            "manage_ports",
            text="Manage ports…",
            tooltip="Add or remove named function inputs and outputs",
        )
        self.add_checkbox(
            "use_surrogate",
            "",
            text="Use trained surrogate",
            state=False,
            tooltip="Replace direct evaluation with the trained surrogate model",
        )
        self.add_button(
            "train_surrogate",
            text="Train surrogate",
            tooltip="Train a surrogate model for this function block",
            tab="Surrogate",
        )
        self.get_widget("edit_function").get_custom_widget().clicked.connect(
            self._open_code_editor
        )
        self.get_widget("manage_ports").get_custom_widget().clicked.connect(
            self._open_port_manager
        )
        self.get_widget("train_surrogate").get_custom_widget().clicked.connect(
            self._train_surrogate
        )
        self.get_widget("train_surrogate").setVisible(False)
        self._refresh_function_summary()

    def _open_code_editor(self):
        parent = self.graph.widget if self.graph else None
        dialog = CodeEditorDialog(
            self.get_property("code_content") or "", node=self, parent=parent
        )
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.set_property("code_content", dialog.get_code())

    def _open_port_manager(self):
        parent = self.graph.widget if self.graph else None
        dialog = PortManagerDialog(self, parent)
        dialog.exec_()
        self._refresh_function_summary()

    def _train_surrogate(self):
        import time

        self.set_property("surrogate_train_trigger", time.time())

    def _refresh_function_summary(self):
        if not self.has_property("interface_summary"):
            return
        num_inputs = len(self.input_ports())
        num_outputs = len(self.output_ports())
        input_word = "input" if num_inputs == 1 else "inputs"
        output_word = "output" if num_outputs == 1 else "outputs"
        summary = f"{num_inputs} {input_word} → {num_outputs} {output_word}"
        super().set_property("interface_summary", summary, push_undo=False)

        code = str(self.get_property("code_content") or "")
        if "cad.crash(" in code:
            execution = "Design Studio · Crash"
        elif "cad.fea(" in code:
            execution = "Design Studio · Static FEA"
        elif "cad.topopt(" in code:
            execution = "Design Studio · Topology optimization"
        else:
            execution = "Python function"
        if self.get_property("use_surrogate"):
            status = str(self.get_property("surrogate_status") or "Not trained")
            execution = f"Surrogate · {status}"
        super().set_property("execution_summary", execution, push_undo=False)
        train_widget = self.get_widget("train_surrogate")
        if train_widget is not None:
            should_show = bool(self.get_property("use_surrogate"))
            if train_widget.isVisible() != should_show:
                train_widget.setVisible(should_show)
                try:
                    self.view.draw_node()
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )

    def set_property(self, name, value, push_undo=True):
        """
        Handle property changes with special logic for port management.

        Dynamically adjusts input/output ports when num_inputs/num_outputs
        properties change, and synchronizes code widget content.

        Args:
            name: Property name being changed
            value: New property value
            push_undo: Whether to push change to undo stack
        """
        self.get_property(name) if self.has_property(name) else None

        if name == "num_inputs":
            try:
                num = max(1, int(value))  # Ensure at least 1
                self._update_input_ports(num)
                super().set_property(name, str(num), push_undo)
            except (ValueError, TypeError):
                pass  # Ignore invalid values
        elif name == "num_outputs":
            try:
                num = max(1, int(value))  # Ensure at least 1
                self._update_output_ports(num)
                super().set_property(name, str(num), push_undo)
            except (ValueError, TypeError):
                pass  # Ignore invalid values
        else:
            super().set_property(name, value, push_undo)

        if name in ("code_content", "use_surrogate", "surrogate_status"):
            self._refresh_function_summary()

    def _update_input_ports(self, num_inputs):
        """
        Dynamically adjust input ports to match specified count.

        Intelligently adds or removes ports while preserving renamed ports
        and avoiding conflicts with existing port names.

        Args:
            num_inputs: Target number of input ports
        """
        current_ports = self.input_ports()
        current_count = len(current_ports)

        if num_inputs > current_count:
            # Add new ports
            for i in range(current_count, num_inputs):
                name = f"in_{i + 1}"
                # Ensure unique name if in_{i+1} is already taken by a renamed port
                existing_names = [p.name() for p in self.input_ports()]
                idx = i + 1
                while name in existing_names:
                    idx += 1
                    name = f"in_{idx}"
                self.add_input(name)

        elif num_inputs < current_count:
            # Remove ports from the end
            ports_to_remove = current_ports[num_inputs:]
            for p in ports_to_remove:
                for cp in p.connected_ports():
                    p.disconnect_from(cp)
                self.delete_input(p.name())
        self._refresh_function_summary()

    def _update_output_ports(self, num_outputs):
        """
        Dynamically adjust output ports to match specified count.

        Intelligently adds or removes ports while preserving renamed ports
        and avoiding conflicts with existing port names.

        Args:
            num_outputs: Target number of output ports
        """
        current_ports = self.output_ports()
        current_count = len(current_ports)

        if num_outputs > current_count:
            # Add new ports
            for i in range(current_count, num_outputs):
                name = f"out_{i + 1}"
                # Ensure unique name
                existing_names = [p.name() for p in self.output_ports()]
                idx = i + 1
                while name in existing_names:
                    idx += 1
                    name = f"out_{idx}"
                self.add_output(name)

        elif num_outputs < current_count:
            # Remove ports from the end
            ports_to_remove = current_ports[num_outputs:]
            for p in ports_to_remove:
                for cp in p.connected_ports():
                    p.disconnect_from(cp)
                self.delete_output(p.name())
        self._refresh_function_summary()


class InputNode(BaseNode):
    """
    Design variable input node for system models.

    Represents an input parameter to the system with configurable bounds,
    units, and variable naming. These nodes provide the interface between
    design variables and the computational graph.

    Node Properties:
        - var_name: Variable name used in generated code
        - unit: Physical unit of the variable
        - min: Minimum allowed value in design space
        - max: Maximum allowed value in design space
    """

    __identifier__ = "com.pfd.input"
    NODE_NAME = "Design Variable"

    def __init__(self) -> None:
        """Initialize the input node with default design variable properties."""
        super().__init__()
        # Role colour — neutral gray body with cyan border for
        # design variables.
        self.set_color(35, 35, 35)
        self.set_property("border_color", (55, 177, 224, 255), push_undo=False)
        self.set_property("text_color", (235, 247, 252, 255), push_undo=False)
        self.set_port_deletion_allowed(True)
        self.add_output("x")
        self.add_text_input(
            "var_name",
            "Name",
            text="x",
            tooltip="Variable name used in equations and function ports",
        )
        self.add_text_input(
            "unit", "Unit", text="-", tooltip="Engineering unit displayed in results"
        )
        self.add_text_input(
            "min",
            "Lower bound",
            text="0.0",
            tooltip="Smallest value the optimizer may use",
        )
        self.add_text_input(
            "max",
            "Upper bound",
            text="10.0",
            tooltip="Largest value the optimizer may use",
        )
        self.set_name("x")

    def set_property(self, name, value, push_undo=True):
        """
        Handle property changes for input node configuration.

        Args:
            name: Property name being changed
            value: New property value
            push_undo: Whether to push change to undo stack
        """
        super().set_property(name, value, push_undo)


class OutputNode(BaseNode):
    """
    Quantity of interest output node for system models.

    Represents an output parameter from the system with optimization objectives,
    requirements bounds, and visualization settings. These nodes define the
    system's objectives and constraints.

    Node Properties:
        - var_name: Variable name used in generated code
        - unit: Physical unit of the variable
        - req_min: Required minimum value (constraint)
        - req_max: Required maximum value (constraint)
        - minimize: Whether this output should be minimized
        - maximize: Whether this output should be maximized
    """

    __identifier__ = "com.pfd.output"
    NODE_NAME = "Quantity of Interest"

    def __init__(self) -> None:
        """Initialize the output node with default objective properties."""
        super().__init__()
        # Role colour — neutral gray body with teal-green border for
        # quantities of interest (graph outputs).
        self.set_color(35, 35, 35)
        self.set_property("border_color", (63, 190, 143, 255), push_undo=False)
        self.set_property("text_color", (235, 250, 244, 255), push_undo=False)
        self.set_port_deletion_allowed(True)
        self.add_input("y")
        self.add_text_input(
            "var_name",
            "Name",
            text="y",
            tooltip="Result name used by downstream analysis and optimization",
        )
        self.add_text_input(
            "unit", "Unit", text="-", tooltip="Engineering unit for this result"
        )
        self.add_text_input(
            "req_min",
            "Allowed min",
            text="-1e9",
            tooltip="Feasibility lower limit; use -inf for no lower limit",
        )
        self.add_text_input(
            "req_max",
            "Allowed max",
            text="1e9",
            tooltip="Feasibility upper limit; use inf for no upper limit",
        )
        self.create_property("minimize", False)
        self.create_property("maximize", False)
        self.create_property("show_in_legend", True)
        self.add_combo_menu(
            "objective_mode",
            "Optimization role",
            items=["Constraint only", "Minimize", "Maximize"],
            tooltip="Use as a feasibility constraint or as an optimization objective",
        )
        self.set_name("y")

    def set_property(self, name, value, push_undo=True):
        """
        Handle property changes with optimization objective logic.

        When minimize/maximize is enabled, automatically disables the other
        and sets bounds to (-inf, inf) since objectives don't have constraints.

        Args:
            name: Property name being changed
            value: New property value
            push_undo: Whether to push change to undo stack
        """
        super().set_property(name, value, push_undo)

        if name == "objective_mode":
            self.set_property("minimize", value == "Minimize", push_undo)
            self.set_property("maximize", value == "Maximize", push_undo)
        elif name == "minimize" and value:
            # Uncheck maximize
            self.set_property("maximize", False, push_undo)
            # Set bounds to -inf inf
            self.set_property("req_min", "-inf", push_undo)
            self.set_property("req_max", "inf", push_undo)
        elif name == "maximize" and value:
            # Uncheck minimize
            self.set_property("minimize", False, push_undo)
            # Set bounds to -inf inf
            self.set_property("req_min", "-inf", push_undo)
            self.set_property("req_max", "inf", push_undo)
        if name in ("minimize", "maximize") and hasattr(self, "view"):
            objective = "Constraint only"
            if self.get_property("minimize"):
                objective = "Minimize"
            elif self.get_property("maximize"):
                objective = "Maximize"
            if self.get_property("objective_mode") != objective:
                super().set_property("objective_mode", objective, push_undo)


class IntermediateNode(BaseNode):
    """
    Intermediate variable node for signal routing and renaming.

    Acts as a pass-through node that can rename variables and change units
    between different parts of the system graph. Useful for organizing complex
    graphs and providing semantic meaning to intermediate calculations.

    Node Properties:
        - var_name: Variable name for this intermediate value
        - unit: Physical unit of the intermediate variable
    """

    __identifier__ = "com.pfd.intermediate"
    NODE_NAME = "Intermediate Variable"

    def __init__(self) -> None:
        """Initialize the intermediate node with pass-through ports."""
        super().__init__()
        # Role colour — neutral gray body with slate border for
        # pass-through/intermediate variables.
        self.set_color(35, 35, 35)
        self.set_property("border_color", (118, 130, 150, 255), push_undo=False)
        self.set_property("text_color", (238, 241, 246, 255), push_undo=False)
        self.set_port_deletion_allowed(True)
        self.add_input("z")
        self.add_output("z")
        self.add_text_input("var_name", "Name", text="z")
        self.add_text_input("unit", "Unit", text="-")


_SYSTEM_NODE_STYLES = {
    "com.pfd.input": ((35, 35, 35, 255), (55, 177, 224, 255), (235, 247, 252, 255)),
    "com.pfd.custom_block": (
        (35, 35, 35, 255),
        (148, 107, 220, 255),
        (242, 238, 250, 255),
    ),
    "com.pfd.output": ((35, 35, 35, 255), (63, 190, 143, 255), (235, 250, 244, 255)),
    "com.pfd.intermediate": (
        (35, 35, 35, 255),
        (118, 130, 150, 255),
        (238, 241, 246, 255),
    ),
}

_SYSTEM_NODE_TOOLTIPS = {
    "com.pfd.input": (
        "<b>Design Variable</b><br/>An optimizer-controlled input. "
        "Define its engineering unit and lower/upper design bounds."
    ),
    "com.pfd.custom_block": (
        "<b>Function / Discipline</b><br/>Transforms inputs into outputs using Python "
        "or a linked Design Studio FEA, crash, or TopOpt study. Double-click to edit."
    ),
    "com.pfd.output": (
        "<b>Quantity of Interest</b><br/>A model result used as a feasibility "
        "constraint, minimization objective, or maximization objective."
    ),
    "com.pfd.intermediate": (
        "<b>Intermediate Variable</b><br/>Routes and renames a value between disciplines."
    ),
}


def apply_system_node_style(node) -> None:
    """Apply role colours and refresh compact widgets after deserialization."""
    identifier = getattr(node, "__identifier__", "")
    style = _SYSTEM_NODE_STYLES.get(identifier)
    if not style:
        return
    color, border, text = style
    node._pylcss_dark_node_palette = (color, border, text)
    if current_theme() == "light":
        color = (248, 250, 252, 255)
        text = (31, 35, 40, 255)
    node.set_property("color", color, push_undo=False)
    node.set_property("border_color", border, push_undo=False)
    node.set_property("text_color", text, push_undo=False)
    if identifier == "com.pfd.output" and node.has_property("objective_mode"):
        objective = "Constraint only"
        if node.get_property("minimize"):
            objective = "Minimize"
        elif node.get_property("maximize"):
            objective = "Maximize"
        node.set_property("objective_mode", objective, push_undo=False)
    controls = getattr(node, "controls_widget", None)
    if controls is not None:
        controls.sync_all()
    code_widget = getattr(node, "code_widget", None)
    if code_widget is not None:
        code_widget.refresh_summary()
    refresh_function = getattr(node, "_refresh_function_summary", None)
    if callable(refresh_function):
        refresh_function()
    try:
        node.view.draw_node()
        node.view.setToolTip(_SYSTEM_NODE_TOOLTIPS.get(identifier, ""))
    except Exception:
        logging.getLogger(__name__).debug(
            "Optional UI operation failed.", exc_info=True
        )
