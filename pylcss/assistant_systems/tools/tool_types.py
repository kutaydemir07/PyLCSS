# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Typed tool definitions and the in-process assistant tool registry."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


ParameterType = Literal["string", "number", "integer", "boolean", "array", "object"]
ToolCategory = Literal[
    "general",
    "cad",
    "modeling",
    "analysis",
    "navigation",
    "project",
]


@dataclass(slots=True)
class ToolParameter:
    """A parameter for a tool."""

    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass(slots=True)
class Tool:
    """A tool that agents can invoke."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[[dict[str, Any]], Any] | None = None
    category: ToolCategory = "general"
    requires_confirmation: bool = False
    validator: Callable[[dict[str, Any]], bool] | None = None

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def get_description_for_prompt(self) -> str:
        """Get a description suitable for inclusion in prompts."""
        params_desc = ", ".join(
            [
                f"{p.name}: {p.type}" + (" (optional)" if not p.required else "")
                for p in self.parameters
            ]
        )
        return f"- **{self.name}**({params_desc}): {self.description}"


class ToolRegistry:
    """Registry of all available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if not tool.name or not tool.name.replace("_", "").isalnum():
            raise ValueError(f"Invalid tool name: {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        parameter_names = [parameter.name for parameter in tool.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError(f"Tool {tool.name!r} has duplicate parameter names")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_by_category(self, category: str) -> list[Tool]:
        """Get all tools in a category."""
        return [t for t in self._tools.values() if t.category == category]

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI schemas for all tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def get_category_schemas(self, category: str) -> list[dict[str, Any]]:
        """Get schemas for tools in a category."""
        return [t.to_openai_schema() for t in self.list_by_category(category)]

    def get_tools_description(self, categories: Iterable[str] | None = None) -> str:
        """Get a text description of tools for prompts."""
        tools: Iterable[Tool] = self._tools.values()
        if categories:
            selected_categories = set(categories)
            tools = (t for t in tools if t.category in selected_categories)

        by_category: dict[str, list[Tool]] = {}
        for tool in tools:
            by_category.setdefault(tool.category, []).append(tool)

        lines = []
        for category, category_tools in sorted(by_category.items()):
            lines.append(f"\n### {category.upper()} Tools")
            for tool in category_tools:
                lines.append(tool.get_description_for_prompt())

        return "\n".join(lines)

    @property
    def all_tools(self) -> list[Tool]:
        return list(self._tools.values())
