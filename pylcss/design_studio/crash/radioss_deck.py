# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""``Run Radioss Deck`` node — execute an existing ``.k`` / ``.rad`` deck.

This is the "run the real benchmark" path: the user supplies an already-prepared
OpenRadioss / LS-DYNA-style input deck (e.g. the Chrysler Neon HPC benchmark)
and PyLCSS just runs Starter + Engine on it, then plays the animation frames
in the existing crash viewer.  No PyLCSS preprocessing, no parametric geometry.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from pylcss.design_studio.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


class RunRadiossDeckNode(CadQueryNode):
    """Execute a user-supplied OpenRadioss / LS-DYNA deck and import animation."""

    __identifier__ = "com.cad.sim.radioss_deck"
    NODE_NAME = "Run Radioss Deck"

    def __init__(self):
        super().__init__()
        # Outputs the same dict shape the in-process crash solver produces, so
        # the viewer picks up frames automatically.
        self.add_output("crash_results", color=(0, 220, 255))

        # Path to the deck (.k or .rad). Blank = error.
        self.create_property("deck_path", "", widget_type="text")
        # Optional explicit Radioss engine file. When blank, Starter is run
        # on the deck first and produces it.
        self.create_property("engine_path", "", widget_type="text")

        # Solver paths — empty means "use solver_paths.json / PATH discovery".
        self.create_property("starter_path", "", widget_type="text")
        self.create_property("engine_executable_path", "", widget_type="text")
        self.create_property("work_dir", "", widget_type="text")

        # Selecting this node implies "run it"; deck_only writes only the
        # staging copy and skips Starter+Engine for inspection.
        self.create_property("deck_only", False, widget_type="checkbox")
        # Generous default — the Neon benchmark on commodity hardware easily
        # takes 30+ minutes.
        self.create_property("timeout_s", 7200.0, widget_type="float")
        self.create_property(
            "visualization", "Von Mises Stress", widget_type="combo",
            items=[
                "Von Mises Stress", "Displacement", "Plastic Strain",
                "Failed Elements",
            ],
        )
        self.create_property("disp_scale", 1.0, widget_type="float")
        # anim_to_vtk preserves the deck's native stress unit.  The normal
        # PyLCSS/OpenRadioss convention is tonne-mm-ms, where native stress
        # multiplied by 1e6 gives MPa.  Set this to 1 for MPa-native decks.
        self.create_property("stress_scale_to_mpa", 1.0e6, widget_type="float")

    @staticmethod
    def _repo_root():
        """Return the PyLCSS install root (parent of the ``pylcss`` package).

        File lives at ``<repo>/pylcss/design_studio/crash/radioss_deck.py`` →
        ``parents[3]`` is ``<repo>``.
        """
        from pathlib import Path
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _resolve_deck_path(cls, value, project_dir=None):
        """Accept absolute paths AND repo-relative paths like ``data/benchmarks/x.k``.

        Resolution order:
            1. Path as written (if it exists).
            2. ``<project_dir>/<path>`` when the node belongs to a saved project.
            3. ``<repo_root>/<path>``.
            4. ``<cwd>/<path>``.
        Returns the first hit or ``None``.
        """
        from pathlib import Path
        if not value:
            return None
        p = Path(value)
        if p.is_file():
            return str(p.resolve())
        if project_dir:
            project = Path(project_dir) / value
            if project.is_file():
                return str(project.resolve())
        repo = cls._repo_root() / value
        if repo.is_file():
            return str(repo.resolve())
        cwd = Path.cwd() / value
        if cwd.is_file():
            return str(cwd.resolve())
        return None

    def run(self, cancel_callback=None):
        self.clear_error()
        from pylcss.solver_backends import (
            ExternalRunConfig,
            SolverBackendError,
            run_openradioss_existing_deck,
        )
        from pylcss.input_values import as_bool

        raw = (self.get_property("deck_path") or "").strip()
        deck_path = self._resolve_deck_path(raw, getattr(self, "_project_dir", None))
        if not deck_path:
            msg = (
                f"Run Radioss Deck: no valid `deck_path` set "
                f"(checked '{raw}' against the repo root and cwd).  "
                "Point this property at an OpenRadioss `.rad` or LS-DYNA `.k` "
                "input deck, or use one of the bundled decks under data/benchmarks/."
            )
            self.set_error(msg)
            return None
        logger.info("Run Radioss Deck: resolved %s", deck_path)

        raw_engine = (self.get_property("engine_path") or "").strip()
        engine_path = (
            self._resolve_deck_path(raw_engine, getattr(self, "_project_dir", None))
            if raw_engine else None
        )
        if raw_engine and not engine_path:
            self.set_error(f"Engine deck path could not be resolved: {raw_engine}")
            return None

        deck_only = as_bool(self.get_property("deck_only"))
        run_flag = not deck_only

        try:
            timeout_s = float(self.get_property("timeout_s") or 0.0)
            disp_scale = float(self.get_property("disp_scale") or 0.0)
            stress_scale = float(self.get_property("stress_scale_to_mpa") or 0.0)
        except (TypeError, ValueError):
            self.set_error("Deck timeout and display scales must be numeric.")
            return None
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            self.set_error("Deck solver timeout must be finite and greater than zero.")
            return None
        if not math.isfinite(disp_scale) or disp_scale <= 0.0:
            self.set_error("Deck displacement scale must be finite and greater than zero.")
            return None
        if not math.isfinite(stress_scale) or stress_scale <= 0.0:
            self.set_error("Deck stress conversion scale must be finite and greater than zero.")
            return None

        work_dir = str(self.get_property("work_dir") or "").strip() or None
        starter_path = str(self.get_property("starter_path") or "").strip() or None
        engine_executable = str(
            self.get_property("engine_executable_path") or ""
        ).strip() or None
        project_dir = getattr(self, "_project_dir", None)
        if project_dir:
            if work_dir and not Path(work_dir).is_absolute():
                work_dir = str(Path(project_dir) / work_dir)
            if starter_path and not Path(starter_path).is_absolute():
                starter_path = str(Path(project_dir) / starter_path)
            if engine_executable and not Path(engine_executable).is_absolute():
                engine_executable = str(Path(project_dir) / engine_executable)

        config = ExternalRunConfig(
            executable=starter_path,
            secondary_executable=engine_executable,
            work_dir=work_dir,
            run_solver=run_flag,
            timeout_s=timeout_s,
            job_name=Path(deck_path).stem,
            cancel_callback=cancel_callback,
        )
        logger.info(
            "Run Radioss Deck: deck=%r, engine=%r, run_solver=%s, timeout=%s",
            deck_path, engine_path, run_flag, config.timeout_s,
        )

        try:
            result = run_openradioss_existing_deck(
                deck_path=deck_path,
                config=config,
                engine_deck_path=engine_path,
                visualization_mode=self.get_property("visualization"),
                disp_scale=disp_scale,
                stress_scale_to_mpa=stress_scale,
            )
            warnings = result.get("warnings") or []
            if warnings:
                logger.warning("Run Radioss Deck warnings: %s", "; ".join(warnings))
            logger.info(
                "Run Radioss Deck: status=%s, type=%s, work_dir=%s, frames=%s",
                result.get('external_status'), result.get('type'), result.get('work_dir'),
                len(result.get('frames', []) or []),
            )
            return result
        except SolverBackendError as exc:
            logger.warning("Run Radioss Deck backend error: %s", exc)
            self.set_error(str(exc))
            return None
        except Exception as exc:
            logger.exception("Run Radioss Deck crashed")
            self.set_error(f"Run Radioss Deck crashed: {exc}")
            return None
