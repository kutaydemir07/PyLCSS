# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Code-based parametric geometry node.

This node is intentionally a modeling block, not a hidden manual CAD escape
hatch: parameters remain visible as node inputs/properties while the geometry
definition can live in one readable CadQuery script.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

import cadquery as cq
import numpy as np

from pylcss.cae.design_studio.core.base_node import CadQueryNode, is_shape, resolve_numeric_input


DEFAULT_CODE = """# Available names:
#   cq, math, np, params
#   L, W, H, hole_d, fillet_r, clearance
#
# Set result = <CadQuery Workplane/Shape/Assembly>.
body = cq.Workplane("XY").box(L, W, H)

if hole_d > 0:
    body = body.faces(">Z").workplane().hole(hole_d)

if fillet_r > 0:
    body = body.edges("|Z").fillet(fillet_r)

result = body
"""


_PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_scalar(raw: str) -> Any:
    text = str(raw).strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_parameter_text(text: str) -> dict[str, Any]:
    """Parse a Python dict or simple ``name=value`` lines from the editor."""
    raw = (text or "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Parameter text must evaluate to a dict.")
        return {str(k): v for k, v in parsed.items()}

    params: dict[str, Any] = {}
    for line_no, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Parameter line {line_no} must use name=value.")
        name, value = stripped.split("=", 1)
        name = name.strip()
        if not _PARAM_NAME_RE.match(name):
            raise ValueError(f"Invalid parameter name on line {line_no}: {name!r}.")
        params[name] = _parse_scalar(value)
    return params


class CadQueryCodeNode(CadQueryNode):
    """Create a part or assembly from a parameterized CadQuery script."""

    __identifier__ = "com.cad.code_part"
    NODE_NAME = "Code Part"

    def __init__(self):
        super().__init__()
        self.set_port_deletion_allowed(True)
        self.create_property("param_count", 6, widget_type="int")
        for idx in range(1, 7):
            self.add_input(f"param_{idx}", color=(180, 180, 0))
        self.add_output("shape", color=(100, 255, 100))

        self.create_property("code", DEFAULT_CODE, widget_type="text")
        self.create_property("parameters", "", widget_type="text")

        defaults = [
            ("L", 40.0),
            ("W", 20.0),
            ("H", 8.0),
            ("hole_d", 6.0),
            ("fillet_r", 1.0),
            ("clearance", 0.2),
        ]
        for idx, (name, value) in enumerate(defaults, start=1):
            self.create_property(f"param_{idx}_name", name, widget_type="text")
            self.create_property(f"param_{idx}_value", value, widget_type="float")

    def get_param_count(self) -> int:
        """Return the current number of active parameter ports."""
        val = self.get_property("param_count")
        if val is not None:
            try:
                return max(0, min(int(val), 32))
            except (ValueError, TypeError):
                pass
        param_ports = [name for name in self.inputs().keys() if name.startswith("param_")]
        return len(param_ports) if param_ports else 6

    def set_param_count(self, count: int) -> None:
        """Dynamically scale the number of parameter ports (0 to 32)."""
        count = max(0, min(int(count), 32))
        current = self.get_param_count()
        if count == current and len([k for k in self.inputs().keys() if k.startswith("param_")]) == count:
            return
        self.set_property("param_count", count)
        self._sync_param_ports(count)

    def _sync_param_ports(self, count: int) -> None:
        """Synchronize node input sockets and property slots with *count*."""
        self.set_port_deletion_allowed(True)
        # Add new ports if needed
        for idx in range(1, count + 1):
            port_name = f"param_{idx}"
            if port_name not in self.inputs():
                self.add_input(port_name, color=(180, 180, 0))
            if not self.has_property(f"param_{idx}_name"):
                self.create_property(f"param_{idx}_name", "", widget_type="text")
            if not self.has_property(f"param_{idx}_value"):
                self.create_property(f"param_{idx}_value", 0.0, widget_type="float")

        # Remove excess ports if needed
        for idx in range(count + 1, 33):
            port_name = f"param_{idx}"
            port = self.get_input(port_name)
            if port is not None:
                try:
                    port.clear_connections()
                except Exception:
                    pass
                try:
                    self.delete_input(port)
                except Exception:
                    pass

        if hasattr(self, "view") and self.view and hasattr(self.view, "draw_node"):
            try:
                self.view.draw_node()
            except Exception:
                pass

    def set_ports(self, port_data):
        super().set_ports(port_data)
        for port in self.input_ports():
            if port.name().startswith("param_"):
                try:
                    port.color = (180, 180, 0, 255)
                except Exception:
                    pass
        shape_port = self.get_output("shape")
        if shape_port:
            try:
                shape_port.color = (100, 255, 100, 255)
            except Exception:
                pass

    def _collect_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        count = self.get_param_count()
        for idx in range(1, count + 1):
            name = str(self.get_property(f"param_{idx}_name") or "").strip()
            if not name:
                continue
            if not _PARAM_NAME_RE.match(name):
                raise ValueError(f"Invalid parameter name: {name!r}.")
            fallback = self.get_property(f"param_{idx}_value")
            value = resolve_numeric_input(self.get_input(f"param_{idx}"), fallback)
            if value is None:
                raise ValueError(f"Connected parameter {name!r} did not produce a number.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"Parameter {name!r} must be finite.")
            params[name] = value

        params.update(_parse_parameter_text(str(self.get_property("parameters") or "")))
        return params

    def run(self):
        try:
            self.clear_error()
            params = self._collect_parameters()
            safe_builtins = {
                "abs": abs,
                "all": all,
                "any": any,
                "bool": bool,
                "dict": dict,
                "enumerate": enumerate,
                "float": float,
                "int": int,
                "len": len,
                "list": list,
                "max": max,
                "min": min,
                "range": range,
                "round": round,
                "set": set,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
            }
            globals_dict = {
                "__builtins__": safe_builtins,
                "cq": cq,
                "math": math,
                "np": np,
                "params": params,
            }
            locals_dict = dict(params)
            code = str(self.get_property("code") or "")
            exec(compile(code, "<pylcss-cadquery-code-node>", "exec"), globals_dict, locals_dict)

            missing = object()
            result = locals_dict.get("result", missing)
            if result is missing:
                result = locals_dict.get("shape", missing)
            if result is missing:
                result = locals_dict.get("assembly", None)
            if isinstance(result, dict):
                for key in ("shape", "assembly", "result"):
                    if key in result and result[key] is not None:
                        result = result[key]
                        break

            if not is_shape(result):
                raise ValueError("Code must assign result to a CadQuery Workplane, Shape, or Assembly.")
            return result
        except Exception as exc:
            self.set_error(f"Code part error: {exc}")
            return None
