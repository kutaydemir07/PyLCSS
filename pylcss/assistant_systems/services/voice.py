# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Voice Controller Module using Faster-Whisper for high-quality speech recognition.

Provides continuous voice command recognition for hands-free control
using local Whisper models and energy-based VAD.
"""

import os
import json
import queue
import logging
import threading
import time
import re
import tempfile
import wave
import struct
import math
import numpy as np
from typing import Optional, Callable, Dict, Any, List, Tuple
from pathlib import Path
from difflib import SequenceMatcher

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from pylcss.assistant_systems.config import VoiceControlConfig, VOICE_COMMANDS, COMMAND_ALIASES

logger = logging.getLogger(__name__)


def _play_beep(frequency: int = 800, duration_ms: int = 100) -> None:
    """Play a system beep for audio feedback."""
    try:
        import winsound
        threading.Thread(
            target=lambda: winsound.Beep(frequency, duration_ms),
            daemon=True
        ).start()
    except ImportError:
        print('\a', end='', flush=True)
    except Exception as e:
        logger.debug(f"Beep failed: {e}")


class VoiceController:
    """
    Voice command recognition using Faster-Whisper.
    
    Implementes energy-based VAD to detect speech segments and transcribes them
    using a local Whisper model.
    """
    
    def __init__(self, config: Optional[VoiceControlConfig] = None):
        self.config = config or VoiceControlConfig()
        
        self._model = None
        self._audio: Optional[pyaudio.PyAudio] = None
        self._stream = None
        
        # Audio params
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper expects 16kHz
        self.chunk = 1024
        
        # VAD params
        self.energy_threshold = 300  # Default, will be calibrated
        self.silence_timeout = 1.0    # Seconds of silence to mark end of phrase
        self.min_phrase_length = 0.5  # Minimum phrase duration
        
        # State
        self._running = False
        self._paused = False
        self._listening = True
        self._dictation_mode = False
        self._llm_mode = False
        self._thread: Optional[threading.Thread] = None
        
        self._audio_queue: queue.Queue = queue.Queue()
        
        # Callbacks
        self._on_command: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._on_text: Optional[Callable[[str], None]] = None
        self._on_partial: Optional[Callable[[str], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        self._on_llm_request: Optional[Callable[[str], None]] = None
        
        # Commands
        self._commands = VOICE_COMMANDS.copy()
        
        # Load model lazily
        
    def calibrate_energy(self, duration: float = 1.0) -> None:
        """Measure ambient noise and set energy threshold."""
        if not self._audio:
            return
            
        logger.info("Calibrating microphone energy...")
        if self._on_status:
            self._on_status("Calibrating microphone...")
            
        stream = self._audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        energies = []
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                data = stream.read(self.chunk, exception_on_overflow=False)
                shorts = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                energy = float(np.sqrt(np.mean(shorts**2)))
                energies.append(energy)
        except Exception as e:
            logger.warning(f"Calibration error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            
        if energies:
            avg_energy = np.mean(energies)
            max_energy = np.max(energies)
            logger.info(f"Calibration: Avg={avg_energy:.2f}, Max={max_energy:.2f}")
            
            # Set threshold slightly above valid max noise, but ensure minimum sensitivity
            # If silence is ~10, threshold ~50 is good.
            # If silence is ~1000 (noisy), threshold ~1500.
            # Updated: Lowered floor to 15.0 and multiplier to 1.3 for better sensitivity
            self.energy_threshold = max(15.0, max_energy * 1.3)
            logger.info(f"Set energy threshold to: {self.energy_threshold:.2f}")
        
    def set_callbacks(
        self,
        on_command: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_llm_request: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._on_command = on_command
        self._on_text = on_text
        self._on_partial = on_partial
        self._on_status = on_status
        self._on_llm_request = on_llm_request
        
    def is_model_available(self) -> bool:
        return True # Whisper downloads automatically if needed
        
    def get_model_download_info(self) -> Dict[str, str]:
        return {"url": "Automatic", "path": "Cache"}
        
    def start(self) -> bool:
        if self._running:
            return True
            
        if not WHISPER_AVAILABLE:
            logger.error("Faster-Whisper not installed")
            return False
            
        try:
            # Init Audio first for calibration
            self._audio = pyaudio.PyAudio()
            
            # Use fixed LOW threshold for reliable detection
            # Skip calibration which can set threshold too high in noisy environments
            self.energy_threshold = 200  # Low, sensitive setting
            logger.info(f"Using fixed energy threshold: {self.energy_threshold}")
            
            # Initialize Model
            if self._on_status:
                self._on_status("Loading Whisper model...")
            
            # Use 'tiny' or 'base' for CPU speed, 'small'/'medium' for accuracy
            # 'tiny.en' is very fast on CPU
            model_size = "base.en" 
            run_on_gpu = False # Auto-detect? Safe to assume CPU for broader compatibility first
            
            self._model = WhisperModel(model_size, device="cpu", compute_type="int8")
            logger.info(f"Loaded Faster-Whisper model: {model_size}")
            
            self._stream = self._audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback
            )
            
            self._running = True
            self._thread = threading.Thread(target=self._processing_loop, daemon=True)
            self._thread.start()
            
            if self._on_status:
                self._on_status(f"Listening (Threshold: {int(self.energy_threshold)})...")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice controller: {e}")
            return False
            
    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio:
            self._audio.terminate()
            
    def pause(self) -> None:
        self._paused = True
    
    def resume(self) -> None:
        self._paused = False
        
    def start_llm_mode(self) -> None:
        self._llm_mode = True
        self._dictation_mode = False
        if self._on_status:
             self._on_status("🤖 Speak to AI...")
        _play_beep(800, 150)
        
    def stop_llm_mode(self) -> None:
        self._llm_mode = False
        if self._on_status:
            self._on_status("Command mode")
        _play_beep(400, 100)
        
    def is_llm_mode(self) -> bool:
        return self._llm_mode
        
    def is_running(self) -> bool:
        return self._running
        
    # --- Internal ---
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self._running and not self._paused:
            self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
        
    def _calculate_energy(self, data) -> float:
        """Calculate RMS energy of audio chunk."""
        shorts = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        if len(shorts) == 0:
            return 0
        return float(np.sqrt(np.mean(shorts**2)))
        
    def _processing_loop(self) -> None:
        """Process audio for VAD and transcription."""
        
        frames = []
        is_speaking = False
        silence_start_time = 0.0
        
        logger.info("Voice processing loop started")
        
        while self._running:
            try:
                data = self._audio_queue.get(timeout=0.1)
                
                # Check energy
                energy = self._calculate_energy(data)
                
                if energy > self.energy_threshold:
                    if not is_speaking:
                        is_speaking = True
                        logger.debug("Speech detected")
                        frames = [] # Start buffering
                    
                    frames.append(data)
                    silence_start_time = 0.0 # Reset silence timer
                    
                else:
                    # Silence
                    if is_speaking:
                        frames.append(data)
                        
                        if silence_start_time == 0.0:
                            silence_start_time = time.time()
                            
                        # Check silence timeout
                        if (time.time() - silence_start_time) > self.silence_timeout:
                            is_speaking = False
                            duration = len(frames) * self.chunk / self.rate
                            
                            if duration >= self.min_phrase_length:
                                # Process phrase
                                logger.info(f"Processing phrase ({duration:.2f}s)")
                                self._transcribe_and_process(frames)
                            else:
                                logger.debug("Phrase too short, ignored")
                                
                            frames = [] # Clear buffer
                            silence_start_time = 0.0
                            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                
    def _transcribe_and_process(self, frames: List[bytes]) -> None:
        if not self._model:
            return
            
        try:
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self._audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
                
            # Transcribe
            segments, info = self._model.transcribe(temp_path, beam_size=5, language="en")
            
            text = " ".join([s.text for s in segments]).strip()
            
            os.remove(temp_path)
            
            if text:
                logger.info(f"Transcribed: '{text}' (prob: {info.language_probability:.2f})")
                self._process_text(text)
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")

    def _process_text(self, text: str) -> None:
        """Process recognized text (Hybrid: Command First, LLM Fallback)."""
        # Clean text
        text = re.sub(r'[^\w\s]', '', text).lower().strip()
        
        if not text:
            return
            
        # 1. ALWAYS Try to match a predefined command FIRST
        # This ensures fast execution for known commands ("Rotate Left", "Stop")
        matched_cmd, cmd_data = self._match_command(text)
        if matched_cmd:
            if self._on_command:
                self._on_command(matched_cmd, cmd_data)
            if self._on_status:
                self._on_status(f"✓ {matched_cmd}")
            _play_beep(1000, 50)
            return

        # 2. If valid command NOT found, Fallback to LLM ("Always-on Assistant")
        # If the manager has hooked up the LLM request callback, use it.
        if self._on_llm_request:
            # Handle control commands locally if needed, but usually LLM can handle "exit" too
            # or we map "exit" to a predefined command in config.
            
            # Pause immediately to prevent self-hearing during processing/TTS
            self.pause()
            
            if self._on_status:
                 self._on_status(f"🤖 AI: {text}")
                 
            self._on_llm_request(text)
            _play_beep(600, 100)
            return
             
        # 3. No command and no LLM handler
        # Just show what was heard
        if self._on_status:
             self._on_status(f"? {text}")

    def _match_command(self, text: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Simple exact or fuzzy match."""
        # 1. Exact match
        if text in self._commands:
            return text, self._commands[text]
            
        # 2. Fuzzy match
        best_match = None
        best_score = 0.0
        
        for cmd in self._commands:
            score = SequenceMatcher(None, text, cmd).ratio()
            if score > best_score:
                best_score = score
                best_match = cmd
                
        if best_score > 0.6: # Configurable threshold
            return best_match, self._commands[best_match]
            
        return None, None

    # Callbacks for dictation, etc can be added back if needed
    def start_dictation(self): pass
    def stop_dictation(self): pass

