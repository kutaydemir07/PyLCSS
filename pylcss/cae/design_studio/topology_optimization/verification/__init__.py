# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Independent engineering verification backends."""

from pylcss.cae.design_studio._lazy_imports import load_attribute, public_names

__all__ = ["run_topopt_fea", "validate_cad_performance"]

_LAZY_EXPORTS = {
    "run_topopt_fea": (f"{__name__}.fea_analysis", "run_topopt_fea"),
    "validate_cad_performance": (
        f"{__name__}.cad_performance",
        "validate_cad_performance",
    ),
}


def __getattr__(name):
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__():
    return public_names(_LAZY_EXPORTS, globals())
