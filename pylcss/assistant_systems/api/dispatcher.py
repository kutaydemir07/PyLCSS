# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Thread-safe dispatch from assistant tools to PyLCSS commands."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot

from pylcss.assistant_systems.api.cad_commands import CadCommandsMixin
from pylcss.assistant_systems.api.desktop_commands import DesktopCommandsMixin
from pylcss.assistant_systems.api.system_commands import SystemCommandsMixin
from pylcss.assistant_systems.services.input import MouseController

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

logger = logging.getLogger(__name__)


class MainThreadExecutor(QObject):
    """Execute a callable on the object's Qt thread."""

    execute_signal = Signal(object, object, object)

    def __init__(self) -> None:
        super().__init__()
        self.execute_signal.connect(self.run_func, Qt.BlockingQueuedConnection)

    @Slot(object, object, object)
    def run_func(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        result_container: dict[str, Any],
    ) -> None:
        """Run function and store result."""
        try:
            result_container["value"] = func(*args)
        except Exception as exc:
            result_container["error"] = exc


class CommandDispatcher(
    CadCommandsMixin,
    SystemCommandsMixin,
    DesktopCommandsMixin,
):
    """Dispatch assistant actions to the appropriate PyLCSS command surface."""

    def __init__(
        self,
        main_window: QMainWindow | None = None,
        mouse_controller: MouseController | None = None,
        *,
        enable_desktop_input: bool = True,
    ) -> None:
        """Initialize dispatch for an optional application window."""
        self.main_window = main_window
        self.mouse = mouse_controller
        if self.mouse is None and enable_desktop_input:
            try:
                self.mouse = MouseController()
            except Exception as exc:
                logger.warning(
                    "Desktop input control is unavailable; "
                    "graph tools remain enabled: %s",
                    exc,
                )

        self._executor = MainThreadExecutor()
        if main_window is not None:
            self._executor.moveToThread(main_window.thread())

        self._on_pause: Callable[[], None] | None = None
        self._on_resume: Callable[[], None] | None = None

        self._action_handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "mouse_click": self._handle_mouse_click,
            "mouse_double_click": self._handle_double_click,
            "mouse_drag_toggle": self._handle_drag_toggle,
            "scroll": self._handle_scroll,
            "scroll_horizontal": self._handle_scroll_horizontal,
            "switch_tab": self._handle_switch_tab,
            "next_tab": self._handle_next_tab,
            "previous_tab": self._handle_previous_tab,
            "keyboard": self._handle_keyboard,
            "pylcss_action": self._handle_pylcss_action,
            "control": self._handle_control,
            "window": self._handle_window,
        }

    def set_control_callbacks(
        self,
        on_pause: Callable[[], None] | None = None,
        on_resume: Callable[[], None] | None = None,
    ) -> None:
        """Set callbacks for control commands."""
        self._on_pause = on_pause
        self._on_resume = on_resume

    def dispatch(self, command_name: str, command_data: dict[str, Any]) -> bool:
        """Dispatch a named command and report whether it completed."""
        action = command_data.get("action")

        if not action:
            logger.warning("Command %r has no action defined", command_name)
            return False

        handler = self._action_handlers.get(action)

        if handler is None:
            logger.warning(f"Unknown action type: {action}")
            return False

        try:
            handler(command_data)
        except Exception:
            logger.exception("Failed to execute command %r", command_name)
            return False
        logger.info("Executed command: %s", command_name)
        return True

    def _run_sync(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run a callable on the owning Qt thread and return its result."""
        if QThread.currentThread() == self._executor.thread():
            return func(*args, **kwargs)

        if kwargs:
            from functools import partial

            func = partial(func, **kwargs)

        result: dict[str, Any] = {"value": None, "error": None}
        self._executor.execute_signal.emit(func, args, result)

        if result["error"] is not None:
            raise result["error"]
        return result["value"]
