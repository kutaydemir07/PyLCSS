# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Text formatting utilities for PyLCSS user interface.

This module provides functions for formatting mathematical text in different
output formats (LaTeX, HTML) for proper display in plots and UI widgets.
"""

from typing import Dict, List


# Greek letters supported for formatting
GREEK_LETTERS: List[str] = [
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa',
    'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon',
    'phi', 'chi', 'psi', 'omega',
    'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa',
    'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho', 'Sigma', 'Tau', 'Upsilon',
    'Phi', 'Chi', 'Psi', 'Omega'
]

# HTML entity mapping for Greek letters
GREEK_HTML_MAP: Dict[str, str] = {
    'alpha': '&alpha;', 'beta': '&beta;', 'gamma': '&gamma;', 'delta': '&delta;',
    'epsilon': '&epsilon;', 'zeta': '&zeta;', 'eta': '&eta;', 'theta': '&theta;',
    'iota': '&iota;', 'kappa': '&kappa;', 'lambda': '&lambda;', 'mu': '&mu;',
    'nu': '&nu;', 'xi': '&xi;', 'omicron': '&omicron;', 'pi': '&pi;',
    'rho': '&rho;', 'sigma': '&sigma;', 'tau': '&tau;', 'upsilon': '&upsilon;',
    'phi': '&phi;', 'chi': '&chi;', 'psi': '&psi;', 'omega': '&omega;',
    'Alpha': '&Alpha;', 'Beta': '&Beta;', 'Gamma': '&Gamma;', 'Delta': '&Delta;',
    'Epsilon': '&Epsilon;', 'Zeta': '&Zeta;', 'Eta': '&Eta;', 'Theta': '&Theta;',
    'Iota': '&Iota;', 'Kappa': '&Kappa;', 'Lambda': '&Lambda;', 'Mu': '&Mu;',
    'Nu': '&Nu;', 'Xi': '&Xi;', 'Omicron': '&Omicron;', 'Pi': '&Pi;',
    'Rho': '&Rho;', 'Sigma': '&Sigma;', 'Tau': '&Tau;', 'Upsilon': '&Upsilon;',
    'Phi': '&Phi;', 'Chi': '&Chi;', 'Psi': '&Psi;', 'Omega': '&Omega;'
}


def format_latex(text: str) -> str:
    """
    Convert variable names to LaTeX format for mathematical display.

    Handles Greek letters and subscripts for proper mathematical notation
    in plots and documentation. Used for rendering variable names in
    plots and other LaTeX-compatible displays.

    Args:
        text: Variable name string to format

    Returns:
        str: LaTeX-formatted string with $ delimiters

    Examples:
        'alpha' -> '$\\alpha$'
        'z_a' -> '$z_{a}$'
        'sigma_1' -> '$\\sigma_{1}$'
    """
    if not text:
        return text

    # Check if the whole word is a greek letter
    if text in GREEK_LETTERS:
        return f"$\\{text}$"

    # Handle subscripts (underscores)
    if '_' in text:
        parts = text.split('_')
        base = parts[0]
        sub = '_'.join(parts[1:])

        # Check if base is greek
        if base in GREEK_LETTERS:
            base = f"\\{base}"

        return f"${base}_{{{sub}}}$"

    return text


def format_html(text: str) -> str:
    """
    Convert variable names to HTML format for Qt display widgets.

    Handles Greek letters and subscripts for proper mathematical notation
    in Qt labels and other HTML-compatible displays.

    Args:
        text: Variable name string to format

    Returns:
        str: HTML-formatted string with entities and tags

    Examples:
        'alpha' -> '&alpha;'
        'z_a' -> 'z<sub>a</sub>'
        'sigma_1' -> '&sigma;<sub>1</sub>'
    """
    if not text:
        return text

    # Check if the whole word is a greek letter
    if text in GREEK_HTML_MAP:
        return GREEK_HTML_MAP[text]

    # Handle subscripts (underscores)
    if '_' in text:
        parts = text.split('_')
        base = parts[0]
        sub = '_'.join(parts[1:])

        # Check if base is greek
        if base in GREEK_HTML_MAP:
            base = GREEK_HTML_MAP[base]

        return f"{base}<sub>{sub}</sub>"

    return text






