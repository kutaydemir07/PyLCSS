# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
import numpy as np

@dataclass
class Variable:
    name: str
    min_val: float
    max_val: float
    value: float = 0.0

@dataclass
class Objective:
    name: str
    weight: float = 1.0
    minimize: bool = True  # True=Min, False=Max

@dataclass
class Constraint:
    name: str
    min_val: float = float('-inf')
    max_val: float = float('inf')

@dataclass
class OptimizationResult:
    x: np.ndarray
    cost: float
    objectives: Dict[str, float]
    constraints: Dict[str, float]
    max_violation: float
    message: str
    success: bool
