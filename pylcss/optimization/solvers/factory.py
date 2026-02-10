# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from .scipy_solver import ScipySolver
from .global_opt import GlobalSolver
from .pareto import ParetoSolver, MultiStartSolver

def get_solver(method: str, settings: dict):
    """
    Factory to return the appropriate solver instance based on the method name.
    """
    scipy_methods = ['SLSQP', 'COBYLA', 'trust-constr']
    
    if method in scipy_methods:
        return ScipySolver(settings)
    
    if method in ['Nevergrad', 'Differential Evolution']:
        return GlobalSolver(settings)

    if method == 'NSGA-II':
        return ParetoSolver(settings)

    if method == 'Multi-Start':
        return MultiStartSolver(settings)
    
    # Default to ScipySolver (SLSQP) if unknown
    return ScipySolver(settings)
