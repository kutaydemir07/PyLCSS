from abc import ABC, abstractmethod
from typing import Callable, List
from ..core import OptimizationResult
from ..evaluator import ModelEvaluator

class BaseSolver(ABC):
    def __init__(self, settings: dict):
        self.settings = settings
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    @abstractmethod
    def solve(self, evaluator: ModelEvaluator, x0: list, 
              callback: Callable) -> OptimizationResult:
        """
        evaluator: The wrapper that runs the model
        x0: Initial guess
        callback: Function to call on every iteration (for GUI updates)
        """
        pass
