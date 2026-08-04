# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Small, safe text-formatting helpers used by Qt and pyqtgraph."""

from __future__ import annotations

from html import escape

__all__ = ["format_html"]

_GREEK_HTML_MAP: dict[str, str] = {
    "alpha": "&alpha;",
    "beta": "&beta;",
    "gamma": "&gamma;",
    "delta": "&delta;",
    "epsilon": "&epsilon;",
    "zeta": "&zeta;",
    "eta": "&eta;",
    "theta": "&theta;",
    "iota": "&iota;",
    "kappa": "&kappa;",
    "lambda": "&lambda;",
    "mu": "&mu;",
    "nu": "&nu;",
    "xi": "&xi;",
    "omicron": "&omicron;",
    "pi": "&pi;",
    "rho": "&rho;",
    "sigma": "&sigma;",
    "tau": "&tau;",
    "upsilon": "&upsilon;",
    "phi": "&phi;",
    "chi": "&chi;",
    "psi": "&psi;",
    "omega": "&omega;",
    "Alpha": "&Alpha;",
    "Beta": "&Beta;",
    "Gamma": "&Gamma;",
    "Delta": "&Delta;",
    "Epsilon": "&Epsilon;",
    "Zeta": "&Zeta;",
    "Eta": "&Eta;",
    "Theta": "&Theta;",
    "Iota": "&Iota;",
    "Kappa": "&Kappa;",
    "Lambda": "&Lambda;",
    "Mu": "&Mu;",
    "Nu": "&Nu;",
    "Xi": "&Xi;",
    "Omicron": "&Omicron;",
    "Pi": "&Pi;",
    "Rho": "&Rho;",
    "Sigma": "&Sigma;",
    "Tau": "&Tau;",
    "Upsilon": "&Upsilon;",
    "Phi": "&Phi;",
    "Chi": "&Chi;",
    "Psi": "&Psi;",
    "Omega": "&Omega;",
}


def format_html(text: str) -> str:
    """Format a variable name as escaped HTML with Greek names/subscripts."""
    if not text:
        return text

    base, separator, subscript = str(text).partition("_")
    safe_base = _GREEK_HTML_MAP.get(base, escape(base))
    if not separator:
        return safe_base
    return f"{safe_base}<sub>{escape(subscript)}</sub>"
