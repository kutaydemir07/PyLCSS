# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""User-facing optimization method names."""

SCIPY_METHODS = ("SLSQP", "COBYLA", "trust-constr")
GLOBAL_METHODS = ("Nevergrad", "Differential Evolution")
SUPPORTED_METHODS = SCIPY_METHODS + GLOBAL_METHODS + ("NSGA-II", "Multi-Start")

__all__ = ["GLOBAL_METHODS", "SCIPY_METHODS", "SUPPORTED_METHODS"]
