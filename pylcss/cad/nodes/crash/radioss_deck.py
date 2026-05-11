# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""``Run Radioss Deck`` node — execute an existing ``.k`` / ``.rad`` deck.

This is the "run the real benchmark" path: the user supplies an already-prepared
OpenRadioss / LS-DYNA-style input deck (e.g. the Chrysler Neon HPC benchmark)
and PyLCSS just runs Starter + Engine on it, then plays the animation frames
in the existing crash viewer.  No PyLCSS preprocessing, no parametric geometry.
"""

from __future__ import annotations

import os
from pathlib import Path

from pylcss.cad.core.base_node import CadQueryNode


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
            items=["Von Mises Stress", "Displacement", "Plastic Strain"],
        )
        self.create_property("disp_scale", 1.0, widget_type="float")

    @staticmethod
    def _repo_root():
        """Return the PyLCSS install root (parent of the ``pylcss`` package)."""
        from pathlib import Path
        return Path(__file__).resolve().parents[4]

    @classmethod
    def _resolve_deck_path(cls, value):
        """Accept absolute paths AND repo-relative paths like ``data/benchmarks/x.k``.

        Resolution order:
            1. Path as written (if it exists).
            2. ``<repo_root>/<path>``.
            3. ``<cwd>/<path>``.
        Returns the first hit or ``None``.
        """
        from pathlib import Path
        if not value:
            return None
        p = Path(value)
        if p.is_file():
            return str(p.resolve())
        repo = cls._repo_root() / value
        if repo.is_file():
            return str(repo.resolve())
        cwd = Path.cwd() / value
        if cwd.is_file():
            return str(cwd.resolve())
        return None

    def run(self):
        print("Run Radioss Deck: node executed.")
        from pylcss.solver_backends import (
            ExternalRunConfig,
            SolverBackendError,
            run_openradioss_existing_deck,
        )
        from pylcss.solver_backends.common import as_bool

        raw = (self.get_property("deck_path") or "").strip()
        deck_path = self._resolve_deck_path(raw)
        if not deck_path:
            msg = (
                f"Run Radioss Deck: no valid `deck_path` set "
                f"(checked '{raw}' against the repo root and cwd).  "
                "Point this property at an OpenRadioss `.rad` or LS-DYNA `.k` "
                "input deck, or use one of the bundled decks under data/benchmarks/."
            )
            print(msg)
            self.set_error(msg)
            return None
        print(f"Run Radioss Deck: resolved deck_path -> {deck_path}")

        raw_engine = (self.get_property("engine_path") or "").strip()
        engine_path = self._resolve_deck_path(raw_engine) if raw_engine else None
        if raw_engine and not engine_path:
            print(f"Run Radioss Deck: engine_path '{raw_engine}' could not be resolved; ignoring.")

        deck_only = as_bool(self.get_property("deck_only"))
        run_flag = not deck_only

        config = ExternalRunConfig(
            executable=(self.get_property("starter_path") or None),
            secondary_executable=(self.get_property("engine_executable_path") or None),
            work_dir=(self.get_property("work_dir") or None),
            keep_files=True,
            run_solver=run_flag,
            timeout_s=float(self.get_property("timeout_s") or 7200.0),
            job_name=Path(deck_path).stem,
        )
        print(
            f"Run Radioss Deck: deck={deck_path!r}, engine={engine_path!r}, "
            f"run_solver={run_flag}, timeout_s={config.timeout_s}"
        )

        try:
            result = run_openradioss_existing_deck(
                deck_path=deck_path,
                config=config,
                engine_deck_path=engine_path,
                visualization_mode=self.get_property("visualization"),
                disp_scale=float(self.get_property("disp_scale") or 1.0),
            )
            warnings = result.get("warnings") or []
            if warnings:
                print("Run Radioss Deck warnings:\n  " + "\n  ".join(warnings))
            print(
                f"Run Radioss Deck: status={result.get('external_status')}, "
                f"type={result.get('type')}, "
                f"work_dir={result.get('work_dir')}, frames={len(result.get('frames', []) or [])}"
            )
            return result
        except SolverBackendError as exc:
            print(f"Run Radioss Deck: backend error:\n{exc}")
            self.set_error(str(exc))
            return None
        except Exception as exc:
            import traceback
            print(f"Run Radioss Deck: crashed {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self.set_error(f"Run Radioss Deck crashed: {exc}")
            return None
