import logging
from typing import Optional
from pydantic import BaseModel, Field
from .base_tool import BaseTool

logger = logging.getLogger(__name__)

class ExecuteSchema(BaseModel):
    environment: str = Field(..., description="Target environment: 'cad' or 'optimization' or 'surrogate'")
    mode: str = Field(default="full", description="Execution mode: 'full' or 'single_node'")
    node_id: Optional[str] = Field(None, description="Specific node ID to execute (if mode is single_node)")

class ExecuteTool(BaseTool):
    name = "execute_graph"
    description = "Execute the current graph or a specific node."
    args_schema = ExecuteSchema

    def __init__(self, main_window=None):
        self.main_window = main_window

    def run(self, environment: str, mode: str = "full", node_id: Optional[str] = None) -> str:
        if not self.main_window:
            return "Error: Main window not available."

        from PySide6.QtCore import QMetaObject, Qt

        if environment.lower() == 'cad':
             if hasattr(self.main_window, 'cad_widget'):
                 # Need to invoke on main thread safely
                 QMetaObject.invokeMethod(self.main_window.cad_widget, "execute_graph", Qt.QueuedConnection)
                 return "CAD execution triggered."
        
        elif environment.lower() == 'optimization':
             if hasattr(self.main_window, 'optimization_widget'):
                 QMetaObject.invokeMethod(self.main_window.optimization_widget, "start_optimization", Qt.QueuedConnection)
                 return "Optimization started."

        return f"Execution for {environment} not fully implemented yet."
