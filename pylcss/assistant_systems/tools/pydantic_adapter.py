# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Adapt PyLCSS tool metadata to validated PydanticAI tools."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic_ai import ModelRetry
from pydantic_ai import Tool as PydanticTool

from pylcss.assistant_systems.tools.tool_types import Tool, ToolParameter

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[str, type[Any]] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _python_type_for_param(param: ToolParameter) -> Any:
    """Map tool JSON types and enum values to runtime Python annotations."""
    try:
        base_type = _TYPE_MAP[param.type]
    except KeyError as exc:
        raise ValueError(f"Unsupported tool parameter type: {param.type!r}") from exc
    if param.enum:
        return Literal[tuple(param.enum)]
    return base_type


def build_pydantic_model_for_tool(tool: Tool) -> type[BaseModel]:
    """Build a strict Pydantic model for a tool's arguments."""
    fields: dict[str, tuple[Any, Any]] = {}
    for param in tool.parameters:
        python_type = _python_type_for_param(param)
        if param.required:
            field = Field(..., description=param.description)
        else:
            field = Field(default=param.default, description=param.description)
            python_type = python_type | None if python_type is not Any else Any
        fields[param.name] = (python_type, field)

    class_name = (
        "".join(part.capitalize() for part in tool.name.split("_")) or "Tool"
    ) + "Args"
    return create_model(
        class_name,
        __config__=ConfigDict(extra="forbid", validate_default=True),
        **fields,
    )


def wrap_legacy_tool(
    tool: Tool,
    args_model: type[BaseModel] | None = None,
) -> tuple[type[BaseModel], Callable[..., Any]]:
    """Bind a dict-based PyLCSS handler to validated keyword arguments."""
    model = args_model or build_pydantic_model_for_tool(tool)
    handler = tool.handler

    def call(**kwargs: Any) -> Any:
        if handler is None:
            raise ModelRetry(f"Tool {tool.name!r} has no handler bound.")
        try:
            payload = model.model_validate(kwargs).model_dump(exclude_none=True)
            if tool.validator is not None and not tool.validator(payload):
                raise ValueError("tool-specific validation rejected the arguments")
            return handler(payload)
        except ValidationError as exc:
            raise ModelRetry(f"Tool {tool.name} arguments are invalid: {exc}") from exc
        except ModelRetry:
            raise
        except Exception as exc:
            logger.exception("Assistant tool %r failed", tool.name)
            raise ModelRetry(f"Tool {tool.name} failed: {exc}") from exc

    call.__name__ = tool.name
    call.__doc__ = tool.description
    return model, call


def build_pydantic_tool(
    tool: Tool,
    *,
    auto_approve_confirmation: bool = False,
) -> tuple[PydanticTool[Any], Callable[..., Any]]:
    """Build a serial, schema-driven PydanticAI tool and its bound handler."""
    args_model, handler = wrap_legacy_tool(tool)
    wrapped = PydanticTool.from_schema(
        function=handler,
        name=tool.name,
        description=tool.description,
        json_schema=args_model.model_json_schema(),
        sequential=True,
    )
    wrapped.requires_approval = (
        tool.requires_confirmation and not auto_approve_confirmation
    )
    return wrapped, handler
