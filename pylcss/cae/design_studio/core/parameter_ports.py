# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared dynamic parameter sockets for script and FreeCAD part nodes."""

import re

from NodeGraphQt.base.model import NodeModel


_PARAMETER_COLOR = (180, 180, 0)
_SHAPE_COLOR = (100, 255, 100)


class _ParameterNodeModel(NodeModel):
    """Register saved dynamic fields before NodeGraphQt restores the sockets."""

    def set_property(self, name, value):
        if re.fullmatch(
            r"param_[1-9]\d*_(name|value)", name
        ) and not self.is_custom_property(name):
            is_name = name.endswith("_name")
            self.add_property(
                name, "" if is_name else 0.0, widget_type="text" if is_name else "float"
            )
        super().set_property(name, value)


class ParameterPortsMixin:
    """Manage named scalar sockets while preserving values across resizing."""

    INITIAL_PARAMS = 0

    def _initialize_parameter_ports(self, defaults):
        # Preserve the base node's metadata and port state in the specialized model.
        model = _ParameterNodeModel()
        model.__dict__.update(self.model.__dict__)
        self.set_model(model)
        self.set_port_deletion_allowed(True)
        self.create_property("param_count", len(defaults), widget_type="int")
        self._sync_param_ports(len(defaults))
        for index, (name, value) in enumerate(defaults, start=1):
            self.set_property(f"param_{index}_name", name)
            self.set_property(f"param_{index}_value", value)

    def get_param_count(self) -> int:
        """Return the saved count, falling back to the existing sockets."""
        try:
            return max(0, int(self.get_property("param_count")))
        except (ValueError, TypeError, OverflowError):
            return (
                sum(name.startswith("param_") for name in self.inputs())
                or self.INITIAL_PARAMS
            )

    def set_param_count(self, count: int) -> None:
        """Resize sockets, honoring an optional user-configured port limit."""
        count = max(0, int(count))
        if self.has_property("parameter_limit"):
            limit = int(self.get_property("parameter_limit"))
            if limit < 1:
                raise ValueError("Parameter port limit must be greater than zero.")
            if count > limit:
                raise ValueError(
                    f"Parameter count exceeds {limit}; increase the parameter port limit."
                )
        if count == self.get_param_count() and len(self.inputs()) == count:
            return
        self._sync_param_ports(count)
        self.set_property("param_count", count)

    def _sync_param_ports(self, count: int) -> None:
        """Add missing sockets and disconnect removed ones before deletion."""
        self.set_port_deletion_allowed(True)
        for index in range(1, count + 1):
            name = f"param_{index}"
            if name not in self.inputs():
                self.add_input(name, color=_PARAMETER_COLOR)
            for suffix, default, widget in (
                ("name", "", "text"),
                ("value", 0.0, "float"),
            ):
                prop = f"{name}_{suffix}"
                if not self.has_property(prop):
                    self.create_property(prop, default, widget_type=widget)

        removed = [
            port
            for name, port in self.inputs().items()
            if name.startswith("param_")
            and name[6:].isdigit()
            and int(name[6:]) > count
        ]
        temporary_scene = None
        if removed and self.view.scene() is None:
            # NodeGraphQt's deletion API assumes a scene even for an unattached node.
            from PySide6.QtWidgets import QGraphicsScene

            temporary_scene = QGraphicsScene()
            temporary_scene.addItem(self.view)
        try:
            for port in removed:
                port.clear_connections()
                self.delete_input(port)
        finally:
            if temporary_scene is not None:
                temporary_scene.removeItem(self.view)

        draw = getattr(getattr(self, "view", None), "draw_node", None)
        if callable(draw):
            draw()

    def set_ports(self, port_data):
        """Restore dynamic sockets and their colors from a saved graph."""
        super().set_ports(port_data)
        for port in self.input_ports():
            if port.name().startswith("param_"):
                port.color = (*_PARAMETER_COLOR, 255)
        shape = self.get_output("shape")
        if shape is not None:
            shape.color = (*_SHAPE_COLOR, 255)
