# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Design goals mapped to structural topology optimizer defaults."""

from __future__ import annotations

from typing import Any


INDUSTRIAL_DESIGN_GOALS = (
    "Lightweight Stiffness",
    "Minimum Mass Under Stress",
)


def normalize_design_goal(value: object = None) -> str:
    """Return a current goal name; reject unsupported goals."""
    text = str(value or INDUSTRIAL_DESIGN_GOALS[0]).strip().lower()
    text = " ".join(text.replace("_", " ").replace("-", " ").split())
    for goal in INDUSTRIAL_DESIGN_GOALS:
        if text == goal.lower():
            return goal
    raise ValueError("Design goal must be " + " or ".join(INDUSTRIAL_DESIGN_GOALS) + ".")


def industrial_topopt_defaults(
    design_goal: object = "Lightweight Stiffness",
) -> dict[str, Any]:
    """Return solver defaults without changing mesh, filter, or result-view controls."""
    goal = normalize_design_goal(design_goal)
    settings: dict[str, Any] = {
        "penal": 3.0,
        "formulation": "Density (SIMP)",
        "optimizer": "Auto",
        "physics_mode": "Structural",
        "stress_constraint": False,
        "max_iter": 120,
        "tol": 0.0035,
        "density_cutoff": 0.30,
    }
    if goal == "Minimum Mass Under Stress":
        settings.update(optimizer="GCMMA", stress_constraint=True)
    return settings
