# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Mouse Controller Module using PyAutoGUI.

Provides system-level mouse and keyboard control for hands-free operation.
"""

import logging
import time
from typing import Tuple, Optional, List

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    # Safety settings
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    pyautogui.PAUSE = 0.01  # Small pause between actions
except ImportError:
    PYAUTOGUI_AVAILABLE = False

logger = logging.getLogger(__name__)


class MouseController:
    """
    Controls mouse cursor and keyboard for hands-free operation.
    
    Wraps PyAutoGUI to provide safe, controlled access to system input.
    """
    
    def __init__(self):
        """Initialize the mouse controller."""
        if not PYAUTOGUI_AVAILABLE:
            raise ImportError("PyAutoGUI is required. Install with: pip install pyautogui")
            
        self._screen_width, self._screen_height = pyautogui.size()
        self._dragging = False
        self._last_move_time = 0.0
        self._move_throttle = 0.016  # ~60fps
        
    def get_screen_size(self) -> Tuple[int, int]:
        """Get the screen dimensions."""
        return self._screen_width, self._screen_height
        
    def get_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        return pyautogui.position()
        
    def move_cursor(self, dx: int, dy: int) -> None:
        """
        Move cursor by relative delta.
        
        Args:
            dx: Horizontal movement (positive = right)
            dy: Vertical movement (positive = down)
        """
        # Throttle moves to prevent overwhelming the system
        current_time = time.time()
        if current_time - self._last_move_time < self._move_throttle:
            return
        self._last_move_time = current_time
        
        if dx == 0 and dy == 0:
            return
            
        try:
            # Get current position
            x, y = pyautogui.position()
            
            # Calculate new position with bounds checking
            # Use 50px margin from edges to prevent PyAutoGUI failsafe corner detection
            margin = 50
            new_x = max(margin, min(self._screen_width - margin, x + dx))
            new_y = max(margin, min(self._screen_height - margin, y + dy))
            
            # Move cursor
            pyautogui.moveTo(new_x, new_y, _pause=False)
            
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe triggered - moved to corner")
            raise
            
    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        """
        Move cursor to absolute position.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Time to move (0 for instant)
        """
        try:
            pyautogui.moveTo(x, y, duration=duration)
        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI failsafe triggered")
            raise
            
    def center_cursor(self) -> None:
        """Move cursor to center of screen."""
        self.move_to(self._screen_width // 2, self._screen_height // 2)
        
    def click(self, button: str = 'left', clicks: int = 1) -> None:
        """
        Perform a mouse click.
        
        Args:
            button: 'left', 'right', or 'middle'
            clicks: Number of clicks (2 for double-click)
        """
        try:
            pyautogui.click(button=button, clicks=clicks)
            logger.debug(f"Clicked {button} x{clicks}")
        except pyautogui.FailSafeException:
            logger.warning("Click aborted - failsafe")
            raise
            
    def double_click(self, button: str = 'left') -> None:
        """Perform a double click."""
        self.click(button=button, clicks=2)
        
    def right_click(self) -> None:
        """Perform a right click."""
        self.click(button='right')
        
    def toggle_drag(self) -> bool:
        """
        Toggle drag mode.
        
        Returns:
            True if now dragging, False if released.
        """
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False
            logger.debug("Drag released")
        else:
            pyautogui.mouseDown()
            self._dragging = True
            logger.debug("Drag started")
        return self._dragging
        
    def scroll(self, clicks: int) -> None:
        """
        Scroll the mouse wheel.
        
        Args:
            clicks: Positive for up, negative for down
        """
        try:
            pyautogui.scroll(clicks)
            logger.debug(f"Scrolled {clicks}")
        except pyautogui.FailSafeException:
            logger.warning("Scroll aborted - failsafe")
            raise
            
    def scroll_horizontal(self, clicks: int) -> None:
        """
        Scroll horizontally (if supported).
        
        Args:
            clicks: Positive for right, negative for left
        """
        try:
            pyautogui.hscroll(clicks)
        except Exception as e:
            logger.warning(f"Horizontal scroll not supported: {e}")
            
    def press_key(self, key: str) -> None:
        """
        Press and release a single key.
        
        Args:
            key: Key name (e.g., 'enter', 'escape', 'a')
        """
        try:
            pyautogui.press(key)
            logger.debug(f"Pressed key: {key}")
        except Exception as e:
            logger.warning(f"Failed to press key {key}: {e}")
            
    def hotkey(self, *keys: str) -> None:
        """
        Press a keyboard shortcut.
        
        Args:
            keys: Keys to press together (e.g., 'ctrl', 's')
        """
        try:
            pyautogui.hotkey(*keys)
            logger.debug(f"Hotkey: {'+'.join(keys)}")
        except Exception as e:
            logger.warning(f"Failed hotkey {'+'.join(keys)}: {e}")
            
    def type_text(self, text: str, interval: float = 0.02) -> None:
        """
        Type text character by character.
        
        Args:
            text: Text to type
            interval: Delay between characters
        """
        try:
            pyautogui.typewrite(text, interval=interval)
            logger.debug(f"Typed: {text[:20]}...")
        except Exception as e:
            logger.warning(f"Failed to type text: {e}")
            
    def write_text(self, text: str) -> None:
        """
        Write text (supports unicode, uses clipboard).
        
        Args:
            text: Text to write
        """
        try:
            import pyperclip
            # Save current clipboard
            old_clipboard = pyperclip.paste()
            # Copy text to clipboard
            pyperclip.copy(text)
            # Paste
            pyautogui.hotkey('ctrl', 'v')
            # Restore clipboard
            time.sleep(0.1)
            pyperclip.copy(old_clipboard)
        except ImportError:
            # Fallback to typewrite
            self.type_text(text)
        except Exception as e:
            logger.warning(f"Failed to write text: {e}")
