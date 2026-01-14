# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Voice Controller Module using Vosk for offline speech recognition.

Provides continuous voice command recognition for hands-free control
with improved fuzzy matching and audio feedback support.
"""

import os
import json
import queue
import logging
import threading
import time
import re
from typing import Optional, Callable, Dict, Any, List, Tuple
from pathlib import Path
from difflib import SequenceMatcher

try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from pylcss.hands_free.config import VoiceControlConfig, VOICE_COMMANDS, COMMAND_ALIASES, VOSK_MODEL_PATH

logger = logging.getLogger(__name__)


def _play_beep(frequency: int = 800, duration_ms: int = 100) -> None:
    """
    Play a system beep for audio feedback.
    
    Uses winsound on Windows, falls back to print bell on other platforms.
    """
    try:
        import winsound
        # Run in thread to not block
        threading.Thread(
            target=lambda: winsound.Beep(frequency, duration_ms),
            daemon=True
        ).start()
    except ImportError:
        # Non-Windows fallback
        print('\a', end='', flush=True)
    except Exception as e:
        logger.debug(f"Beep failed: {e}")


def _phonetic_key(word: str) -> str:
    """
    Generate a simple phonetic key for a word (simplified Soundex).
    
    This helps match words that sound similar but are spelled differently.
    """
    if not word:
        return ""
    
    word = word.lower().strip()
    
    # Common phonetic substitutions
    replacements = [
        (r'ph', 'f'),
        (r'ck', 'k'),
        (r'gh', 'g'),
        (r'wh', 'w'),
        (r'wr', 'r'),
        (r'kn', 'n'),
        (r'mb', 'm'),
        (r'ng', 'n'),
        (r'qu', 'kw'),
        (r'x', 'ks'),
        (r'ce', 'se'),
        (r'ci', 'si'),
        (r'cy', 'sy'),
        (r'tion', 'shun'),
        (r'sion', 'shun'),
    ]
    
    for pattern, replacement in replacements:
        word = re.sub(pattern, replacement, word)
    
    # Remove consecutive duplicate letters
    result = []
    prev = None
    for char in word:
        if char != prev:
            result.append(char)
            prev = char
    
    return ''.join(result)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def _similarity_score(s1: str, s2: str) -> float:
    """
    Calculate similarity score between two strings (0.0 to 1.0).
    
    Combines multiple matching strategies for robust recognition.
    """
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    
    if s1 == s2:
        return 1.0
    
    # Sequence matcher score
    seq_score = SequenceMatcher(None, s1, s2).ratio()
    
    # Levenshtein-based score
    max_len = max(len(s1), len(s2))
    if max_len > 0:
        lev_score = 1.0 - (_levenshtein_distance(s1, s2) / max_len)
    else:
        lev_score = 0.0
    
    # Phonetic similarity
    p1 = _phonetic_key(s1)
    p2 = _phonetic_key(s2)
    phonetic_score = SequenceMatcher(None, p1, p2).ratio() if p1 and p2 else 0.0
    
    # Word overlap score
    words1 = set(s1.split())
    words2 = set(s2.split())
    if words1 and words2:
        overlap = len(words1 & words2)
        word_score = overlap / max(len(words1), len(words2))
    else:
        word_score = 0.0
    
    # Weighted combination
    return (seq_score * 0.3 + lev_score * 0.3 + phonetic_score * 0.2 + word_score * 0.2)


class VoiceController:
    """
    Offline voice command recognition using Vosk.
    
    Continuously listens for voice commands and triggers callbacks
    when recognized commands are detected.
    
    Features:
    - Fuzzy command matching with phonetic similarity
    - Audio feedback on command recognition
    - Dictation mode for typing text
    - Hotword activation
    """
    
    def __init__(self, config: Optional[VoiceControlConfig] = None):
        """
        Initialize the voice controller.
        
        Args:
            config: Voice control configuration. Uses defaults if None.
        """
        self.config = config or VoiceControlConfig()
        
        self._model: Optional[vosk.Model] = None
        self._recognizer: Optional[vosk.KaldiRecognizer] = None
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream = None
        
        # State
        self._running = False
        self._paused = False
        self._listening = True  # False when waiting for hotword
        self._dictation_mode = False
        self._thread: Optional[threading.Thread] = None
        
        # Audio queue for processing
        self._audio_queue: queue.Queue = queue.Queue()
        
        # Callbacks
        self._on_command: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._on_text: Optional[Callable[[str], None]] = None
        self._on_partial: Optional[Callable[[str], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        
        # Command vocabulary with aliases
        self._commands = VOICE_COMMANDS.copy()
        self._aliases = COMMAND_ALIASES.copy()
        
        # Build command index for fast matching
        self._build_command_index()
        
    def _build_command_index(self) -> None:
        """Build phonetic index for commands for faster matching."""
        self._command_phonetics = {}
        self._command_words = {}
        
        for cmd in self._commands:
            self._command_phonetics[cmd] = _phonetic_key(cmd)
            self._command_words[cmd] = set(cmd.lower().split())
        
    def set_callbacks(
        self,
        on_command: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Set callback functions for recognition events.
        
        Args:
            on_command: Called when a command is recognized (command_text, command_data)
            on_text: Called for any recognized text (dictation mode)
            on_partial: Called with partial recognition results
            on_status: Called for status updates
        """
        self._on_command = on_command
        self._on_text = on_text
        self._on_partial = on_partial
        self._on_status = on_status
        
    def is_model_available(self) -> bool:
        """Check if the Vosk model is available."""
        model_path = Path(self.config.model_path)
        return model_path.exists() and model_path.is_dir()
        
    def get_model_download_info(self) -> Dict[str, str]:
        """Get information for downloading the Vosk model."""
        return {
            "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "name": "vosk-model-small-en-us-0.15",
            "size": "40 MB",
            "path": str(VOSK_MODEL_PATH),
        }
        
    def start(self) -> bool:
        """
        Start the voice recognition system.
        
        Returns:
            True if started successfully, False otherwise.
        """
        if self._running:
            logger.warning("Voice controller already running")
            return True
            
        if not VOSK_AVAILABLE:
            logger.error("Vosk is not installed. Install with: pip install vosk")
            return False
            
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio is not installed. Install with: pip install pyaudio")
            return False
            
        if not self.is_model_available():
            logger.error(f"Vosk model not found at {self.config.model_path}")
            if self._on_status:
                self._on_status("Model not found. Please download the Vosk model.")
            return False
            
        try:
            # Load Vosk model
            vosk.SetLogLevel(-1)  # Suppress Vosk logging
            self._model = vosk.Model(self.config.model_path)
            self._recognizer = vosk.KaldiRecognizer(self._model, self.config.sample_rate)
            self._recognizer.SetWords(True)
            
            # Initialize PyAudio
            self._audio = pyaudio.PyAudio()
            
            # Find the default input device
            try:
                default_device = self._audio.get_default_input_device_info()
                logger.info(f"Using audio device: {default_device['name']}")
            except OSError:
                logger.warning("No default audio input device found")
            
            self._stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=4000,
                stream_callback=self._audio_callback
            )
            
            # Start processing thread
            self._running = True
            self._listening = not self.config.hotword_enabled
            self._thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._thread.start()
            
            logger.info("Voice controller started")
            if self._on_status:
                status = "Listening..." if self._listening else f"Say '{self.config.hotword}' to activate"
                self._on_status(status)
            
            # Play startup beep
            if self.config.feedback_enabled:
                _play_beep(600, 100)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice controller: {e}")
            self.stop()
            return False
            
    def stop(self) -> None:
        """Stop the voice recognition system."""
        self._running = False
        
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            
        if self._audio:
            try:
                self._audio.terminate()
            except Exception:
                pass
            self._audio = None
            
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
            
        self._model = None
        self._recognizer = None
        
        logger.info("Voice controller stopped")
        
    def pause(self) -> None:
        """Pause voice recognition."""
        self._paused = True
        if self._on_status:
            self._on_status("Voice recognition paused")
        logger.info("Voice recognition paused")
        
    def resume(self) -> None:
        """Resume voice recognition."""
        self._paused = False
        if self._on_status:
            self._on_status("Listening...")
        logger.info("Voice recognition resumed")
        
    def start_dictation(self) -> None:
        """Enable dictation mode (types all recognized text)."""
        self._dictation_mode = True
        if self._on_status:
            self._on_status("Dictation mode ON")
        if self.config.feedback_enabled:
            _play_beep(1000, 100)
        logger.info("Dictation mode started")
        
    def stop_dictation(self) -> None:
        """Disable dictation mode."""
        self._dictation_mode = False
        if self._on_status:
            self._on_status("Dictation mode OFF - Command mode")
        if self.config.feedback_enabled:
            _play_beep(500, 100)
        logger.info("Dictation mode stopped")
        
    def is_running(self) -> bool:
        """Check if voice controller is running."""
        return self._running
        
    def is_paused(self) -> bool:
        """Check if voice controller is paused."""
        return self._paused
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - adds audio data to queue."""
        if self._running and not self._paused:
            self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
        
    def _processing_loop(self) -> None:
        """Main processing loop running in background thread."""
        while self._running:
            try:
                # Get audio data from queue
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if not self._recognizer:
                continue
                
            # Process audio
            if self._recognizer.AcceptWaveform(data):
                result = json.loads(self._recognizer.Result())
                text = result.get('text', '').strip().lower()
                
                if text:
                    self._process_text(text)
            else:
                # Partial result
                partial = json.loads(self._recognizer.PartialResult())
                partial_text = partial.get('partial', '').strip()
                
                if partial_text and self._on_partial:
                    self._on_partial(partial_text)
                    
    def _process_text(self, text: str) -> None:
        """
        Process recognized text and dispatch commands.
        
        Args:
            text: The recognized text to process.
        """
        logger.info(f"Voice recognized: '{text}'")
        
        # Check for hotword if enabled
        if self.config.hotword_enabled and not self._listening:
            if self.config.hotword.lower() in text.lower():
                self._listening = True
                if self._on_status:
                    self._on_status("Listening for command...")
                if self.config.feedback_enabled:
                    _play_beep(800, 100)
                # Start timeout
                threading.Timer(self.config.command_timeout, self._hotword_timeout).start()
            return
            
        # Dictation mode - type everything
        if self._dictation_mode:
            if self._on_text:
                self._on_text(text)
            return
            
        # Try to match a command using fuzzy matching
        matched_command, command_data, confidence = self._match_command_fuzzy(text)
        
        if matched_command and command_data:
            if self._on_command:
                self._on_command(matched_command, command_data)
            if self._on_status:
                self._on_status(f"✓ {matched_command}")
                
            # Audio feedback for successful command
            if self.config.feedback_enabled:
                _play_beep(1000, 50)  # Short high beep for success
                
            logger.info(f"Matched command: '{text}' -> '{matched_command}' (confidence: {confidence:.2f})")
        else:
            # Not a recognized command
            logger.info(f"No command match for: '{text}'")
            if self._on_text:
                self._on_text(text)
                
        # Reset listening state if hotword mode
        if self.config.hotword_enabled:
            self._listening = False
            if self._on_status:
                self._on_status(f"Say '{self.config.hotword}' to activate")
                
    def _match_command_fuzzy(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], float]:
        """
        Match text to a command using advanced fuzzy matching.
        
        Returns:
            Tuple of (matched_command_name, command_data, confidence_score)
        """
        text = text.strip().lower()
        
        # Direct match - highest confidence
        if text in self._commands:
            return text, self._commands[text], 1.0
            
        # Check aliases
        if text in self._aliases:
            alias_target = self._aliases[text]
            if alias_target in self._commands:
                return alias_target, self._commands[alias_target], 0.95
        
        # Fuzzy matching with all commands
        best_match = None
        best_score = 0.0
        best_data = None
        
        for cmd, data in self._commands.items():
            score = _similarity_score(text, cmd)
            
            # Boost score for partial word matches
            text_words = set(text.split())
            cmd_words = self._command_words.get(cmd, set())
            
            # If any word from text matches command word exactly, boost score
            for word in text_words:
                if word in cmd_words:
                    score += 0.15
                elif len(word) > 2:
                    # Check if word is a prefix of any command word
                    for cword in cmd_words:
                        if cword.startswith(word) or word.startswith(cword):
                            score += 0.1
                            break
            
            if score > best_score:
                best_score = score
                best_match = cmd
                best_data = data
        
        # Extended keyword matching for common patterns
        keyword_matches = self._keyword_match(text)
        if keyword_matches:
            kw_cmd, kw_score = keyword_matches
            if kw_score > best_score and kw_cmd in self._commands:
                best_score = kw_score
                best_match = kw_cmd
                best_data = self._commands[kw_cmd]
        
        # Require minimum confidence to avoid false positives
        min_confidence = 0.50  # Raised from 0.40 to reduce false matches
        if best_score >= min_confidence:
            return best_match, best_data, best_score
            
        return None, None, 0.0
        
    def _keyword_match(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Keyword-based matching for common commands.
        
        Returns tuple of (command, score) or None.
        """
        text = text.lower()
        
        # Define keyword patterns with associated commands and scores
        KEYWORD_PATTERNS = [
            # Mouse actions
            (["click", "select", "press", "tap", "pick"], "click", 0.85),
            (["right click", "right-click", "context", "menu"], "right click", 0.9),
            (["double click", "double-click", "double", "twice"], "double click", 0.9),
            
            # Scrolling
            (["scroll up", "up scroll"], "scroll up", 0.9),
            (["scroll down", "down scroll"], "scroll down", 0.9),
            
            # Tab navigation
            (["modeling", "model tab"], "go to modeling", 0.85),
            (["cad", "design"], "go to cad", 0.85),
            (["solution", "sample", "space"], "go to solution space", 0.8),
            (["surrogate", "surrogate tab"], "go to surrogate", 0.85),
            (["optimization", "optim", "optimize"], "go to optimization", 0.85),
            (["sensitivity", "analysis"], "go to sensitivity", 0.85),
            (["next tab", "next"], "next tab", 0.8),
            (["previous tab", "prev", "back"], "previous tab", 0.8),
            
            # Keyboard shortcuts
            (["save", "save project"], "save", 0.9),
            (["undo", "go back"], "undo", 0.9),
            (["redo", "redo that"], "redo", 0.9),
            (["copy", "copy that"], "copy", 0.9),
            (["paste"], "paste", 0.9),
            (["delete", "remove"], "delete", 0.9),
            (["escape", "cancel", "close"], "escape", 0.85),
            (["enter", "confirm", "okay", "ok"], "enter", 0.85),
            
            # Control commands
            (["pause", "stop", "halt", "wait", "hold"], "pause", 0.85),
            (["resume", "continue", "start", "go"], "resume", 0.8),
            (["calibrate", "recenter", "center"], "calibrate", 0.9),
        ]
        
        best_match = None
        best_score = 0.0
        
        for keywords, command, base_score in KEYWORD_PATTERNS:
            for keyword in keywords:
                if keyword in text:
                    # Longer keyword matches are more confident
                    length_bonus = min(0.1, len(keyword) / 50)
                    score = base_score + length_bonus
                    
                    # Exact match bonus
                    if text == keyword:
                        score += 0.1
                        
                    if score > best_score:
                        best_score = score
                        best_match = command
        
        if best_match:
            return best_match, best_score
        return None
        
    def _hotword_timeout(self) -> None:
        """Called when hotword timeout expires."""
        if self._listening and self.config.hotword_enabled:
            self._listening = False
            if self._on_status:
                self._on_status(f"Say '{self.config.hotword}' to activate")
