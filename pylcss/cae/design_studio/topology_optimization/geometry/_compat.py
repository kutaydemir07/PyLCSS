# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Warn saved callers about options retired from geometry reconstruction."""
import warnings


def warn_retired_options(options, *, operation):
    if options:
        names = ', '.join(sorted(options))
        warnings.warn(f'{operation}: retired option(s) {names} have no effect',
                      DeprecationWarning, stacklevel=3)
