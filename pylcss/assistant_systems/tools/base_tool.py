from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from dataclasses import dataclass

@dataclass
class ToolSchema:
    """Base schema for tools."""
    pass

class BaseTool(ABC):
    """
    Abstract base class for all LLM tools.
    """
    name: str = "base_tool"
    description: str = "Base tool description"
    args_schema: Type[ToolSchema] = ToolSchema

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Execute the tool logic.
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema for the tool arguments.
        """
        # This is a placeholder since we removed pydantic.
        # If schema generation is needed, we should implement a converter for dataclasses
        # or just return a static dict if provided.
        return {}
