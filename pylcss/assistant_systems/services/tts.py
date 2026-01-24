# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Text-to-Speech Module using Edge-TTS.

Provides high-quality neural speech synthesis using Microsoft Edge's online TTS
and pygame for playback.
"""

import logging
import threading
import tempfile
import asyncio
import os
from typing import Optional, Callable

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextToSpeech:
    """
    Text-to-speech engine using Edge-TTS (Neural Voices).
    """
    
    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self._speaking = False
        self._lock = threading.Lock()
        
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
            except Exception as e:
                logger.error(f"Failed to init pygame mixer: {e}")
                
    def is_available(self) -> bool:
        return EDGE_TTS_AVAILABLE and PYGAME_AVAILABLE
        
    def speak(self, text: str, async_: bool = True, on_complete: Optional[Callable] = None) -> None:
        """Speak text using Edge-TTS."""
        if not self.is_available():
            logger.warning("TTS dependencies missing (edge-tts or pygame)")
            if on_complete:
                on_complete()
            return
            
        if async_:
            threading.Thread(target=self._run_speech, args=(text, on_complete), daemon=True).start()
        else:
            self._run_speech(text, on_complete)
            
    def _run_speech(self, text: str, on_complete: Optional[Callable] = None) -> None:
        """Run the async speech generation in a thread."""
        with self._lock:
            self._speaking = True
            try:
                asyncio.run(self._generate_and_play(text))
            except Exception as e:
                logger.error(f"TTS Error: {e}")
            finally:
                self._speaking = False
                if on_complete:
                    on_complete()
                
    async def _generate_and_play(self, text: str) -> None:
        """Generate mp3 and play it."""
        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_path = f.name
                
            await communicate.save(temp_path)
            
            # Play
            if PYGAME_AVAILABLE:
                try:
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        if not self._speaking: # Check for stop
                            pygame.mixer.music.stop()
                            break
                        await asyncio.sleep(0.1)
                        
                    pygame.mixer.music.unload()
                except Exception as e:
                    logger.error(f"Playback error: {e}")
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Generation error: {e}")

    def stop(self) -> None:
        """Stop speaking."""
        self._speaking = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except:
                pass
            
    def is_speaking(self) -> bool:
        return self._speaking
        
    def set_rate(self, rate: int) -> None:
        # edge-tts uses percentage string like "+10%"
        # Map nice int to string? For now ignore or simplified map
        self.rate = f"+0%" # Default
            
    def set_volume(self, volume: float) -> None:
        # edge-tts volume is also string "+0%"
        pass


# Singleton
_tts_instance: Optional[TextToSpeech] = None

def get_tts() -> TextToSpeech:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TextToSpeech()
    return _tts_instance

def speak(text: str, async_: bool = True) -> None:
    get_tts().speak(text, async_=async_)
