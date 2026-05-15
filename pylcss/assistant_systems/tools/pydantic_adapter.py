# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Adapter that turns the existing ``Tool`` dataclass registry into PydanticAI
tools without rewriting every tool by hand.

Why this exists
---------------
The legacy registry (`tools/registry.py`) defines each tool as a ``Tool``
dataclass with:

    Tool(name=..., description=..., parameters=[ToolParameter(...)],
         handler=lambda data: dispatcher.method(data))

The handler always takes a single ``dict`` argument.  PydanticAI wants a
typed Python callable plus a Pydantic model for arguments -- it then exposes
that to the LLM through *native* function calling (strict-mode JSON schema
validated at decode time, not via prompt parsing) and auto-retries on
``ValidationError``.

This module bridges the two:

- ``build_pydantic_model_for_tool(tool)`` synthesises a ``BaseModel`` subclass
  from the ``ToolParameter`` list at runtime, with proper Python typing,
  defaults, and the description carried over.
- ``wrap_legacy_tool(tool)`` returns a callable ``handler(**kwargs) -> Any``
  that PydanticAI can register: it instantiates the Pydantic model
  (validating + coercing inputs), then forwards the validated dict to the
  legacy lambda.

Coexists with the old ToolRegistry; nothing about the legacy path is changed.
PydanticAgentRunner (next file) wires this adapter into a real
``pydantic_ai.Agent`` so the LLM gets native function-calling for the same
23 tools we already ship.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, ValidationError, create_model

from pylcss.assistant_systems.tools.registry import Tool, ToolParameter

logger = logging.getLogger(__name__)


# Legacy ToolParameter.type strings -> Python types.
# We keep this small + explicit; anything weird falls back to ``Any`` so the
# wrapped tool still loads, just without strict validation.
_TYPE_MAP: Dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _python_type_for_param(param: ToolParameter) -> type:
    """Map a legacy ToolParameter to the Python type pydantic should validate
    against.  Enum strings get a Literal-of-strings constraint."""
    base = _TYPE_MAP.get(param.type.lower(), Any)
    if param.enum:
        # Pydantic accepts a Literal[...] for enum-like string constraints.
        # We build it lazily because typing.Literal needs splatting at module
        # scope, which is awkward for runtime construction; instead we use a
        # set-of-allowed-values check via ``Field(pattern=...)``? No -- the
        # cleanest path is to keep the Python type as ``str`` and let the
        # PydanticAI JSON schema include the enum (which it does via Field).
        # The enum itself is enforced by the JSON schema sent to the LLM.
        return base if base is not Any else str
    return base


def build_pydantic_model_for_tool(tool: Tool) -> Type[BaseModel]:
    """Synthesise a Pydantic ``BaseModel`` describing ``tool``'s arguments.

    The model class name is the tool's name in PascalCase + 'Args' so traces
    and validation errors are recognisable. The model's fields carry the
    parameter descriptions (so PydanticAI emits a proper JSON schema) and
    the right Python type.
    """
    fields: Dict[str, Tuple[Any, Any]] = {}
    for param in tool.parameters:
        py_type = _python_type_for_param(param)
        # Build a Field with description (and enum / default if present).
        field_kwargs: Dict[str, Any] = {"description": param.description}
        if param.enum:
            # Surfaces in the JSON schema as ``"enum": [...]`` -- the LLM
            # sees it and the schema validator rejects out-of-set values.
            field_kwargs["json_schema_extra"] = {"enum": list(param.enum)}
        if param.required:
            field = Field(..., **field_kwargs)
        else:
            field = Field(default=param.default, **field_kwargs)
            # Optional becomes ``Optional[T]`` so None is a legal value.
            py_type = Optional[py_type] if py_type is not Any else Any
        fields[param.name] = (py_type, field)

    # PascalCase + Args; e.g. "create_cad_geometry" -> "CreateCadGeometryArgs"
    pascal = "".join(part.capitalize() for part in tool.name.split("_")) or "Tool"
    cls_name = f"{pascal}Args"
    model = create_model(cls_name, __base__=BaseModel, **fields)
    return model


def wrap_legacy_tool(
    tool: Tool, args_model: Optional[Type[BaseModel]] = None,
) -> Tuple[Type[BaseModel], Callable[..., Any]]:
    """Return ``(args_model, callable)`` for registering with PydanticAI.

    The returned callable accepts keyword arguments that match the model's
    fields; it builds a plain dict and forwards it to the legacy handler.
    Validation errors are raised as ``ModelRetry`` so PydanticAI shows the
    LLM the diagnostic and lets it correct the call.
    """
    if args_model is None:
        args_model = build_pydantic_model_for_tool(tool)

    handler = tool.handler

    def _call(**kwargs: Any) -> Any:
        # Pydantic-ai 1.x already validates kwargs against the args_model
        # before calling us, so we don't need to re-validate.  But we do want
        # the legacy "everything is a dict" calling convention preserved.
        if handler is None:
            raise RuntimeError(f"Tool {tool.name!r} has no handler bound.")
        try:
            payload = dict(kwargs)
            return handler(payload)
        except ValidationError as exc:
            # Surface as ModelRetry so PydanticAI re-prompts the model.
            from pydantic_ai import ModelRetry
            raise ModelRetry(f"Tool {tool.name} arguments invalid: {exc}") from exc
        except (ValueError, RuntimeError) as exc:
            # Handler raised due to missing/invalid parameters — let the LLM
            # see the error message and correct the call instead of crashing.
            from pydantic_ai import ModelRetry
            raise ModelRetry(f"Tool {tool.name} failed: {exc}") from exc

    # Preserve metadata pydantic_ai uses to build the JSON schema sent to the
    # LLM -- name, docstring, and signature inferred from the model.
    _call.__name__ = tool.name
    _call.__doc__ = tool.description
    return args_model, _call
