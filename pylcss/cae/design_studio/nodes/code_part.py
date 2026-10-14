# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Code-based parametric geometry node.

This node is intentionally a modeling block, not a hidden manual CAD escape
hatch: parameters remain visible as node inputs/properties while the geometry
definition can live in one readable CadQuery script.
"""

from __future__ import annotations

import math
from typing import Any

import cadquery as cq
import numpy as np

from pylcss.cae.design_studio.core.base_node import CadQueryNode, is_shape, resolve_numeric_input
from pylcss.cae.design_studio.core.parameter_ports import ParameterPortsMixin
from pylcss.cae.design_studio.core.parameters import (
    PARAMETER_NAME_RE as _PARAM_NAME_RE,
    parse_parameter_text as _parse_parameter_text,
)


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


class CadQueryCodeNode(ParameterPortsMixin, CadQueryNode):
    """Create a part or assembly from a parameterized CadQuery script."""

    __identifier__ = "com.cad.code_part"
    NODE_NAME = "Code Part"
    INITIAL_PARAMS = 6

    def __init__(self):
        super().__init__()
        self.create_property("parameter_limit", 32, widget_type="int")
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
        self._initialize_parameter_ports(defaults)


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
                "filter": filter,
                "frozenset": frozenset,
                "getattr": getattr,
                "hasattr": hasattr,
                "isinstance": isinstance,
                "issubclass": issubclass,
                "iter": iter,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "next": next,
                "pow": pow,
                "range": range,
                "reversed": reversed,
                "slice": slice,
                "sorted": sorted,
                "round": round,
                "set": set,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "zip": zip,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "RuntimeError": RuntimeError,
            }
            namespace = {
                **params,
                "__builtins__": safe_builtins,
                "cq": cq,
                "math": math,
                "np": np,
                "params": params,
            }
            code = str(self.get_property("code") or "")
            exec(compile(code, "<pylcss-cadquery-code-node>", "exec"), namespace, namespace)

            missing = object()
            result = namespace.get("result", missing)
            if result is missing:
                result = namespace.get("shape", missing)
            if result is missing:
                result = namespace.get("assembly", None)
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
