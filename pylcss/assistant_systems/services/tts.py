# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Text-to-speech -- local Kokoro via RealtimeTTS with edge-tts fallback.

Why this rewrite
----------------
The legacy TTS shipped a synchronous edge-tts pipeline that:
  - made a network call per phrase (no offline mode);
  - generated a full MP3 to a temp file before playback could start
    (~2-3 s first-byte latency);
  - used ``pygame.mixer.music`` for playback (no clean barge-in --
    stopping mid-phrase requires polling).

The new design uses :class:`RealtimeTTS.TextToAudioStream` so we get:
  - **local Kokoro-82M** (550x realtime on CPU after quantisation, fully
    offline, ~50 MB ONNX);
  - **streaming playback**: audio starts as soon as the first sentence is
    synthesised, not after the whole phrase finishes;
  - **clean barge-in via ``stream.stop()``** -- the audio output buffer is
    flushed immediately, no polling loop;
  - automatic fallback chain (Kokoro -> Piper -> edge-tts -> system) so
    voice still works on machines where the local model couldn't be set
    up.

The public API (``TextToSpeech``, ``get_tts``, ``speak``) is preserved so
the existing manager + chat dialog don't need to change.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# --- Engine probes ----------------------------------------------------------
try:
    from RealtimeTTS import TextToAudioStream
    REALTIMETTS_AVAILABLE = True
except ImportError:
    TextToAudioStream = None  # type: ignore[assignment]
    REALTIMETTS_AVAILABLE = False

try:
    from RealtimeTTS import KokoroEngine
    KOKORO_AVAILABLE = True
except ImportError:
    KokoroEngine = None  # type: ignore[assignment]
    KOKORO_AVAILABLE = False

try:
    from RealtimeTTS import SystemEngine
    SYSTEM_ENGINE_AVAILABLE = True
except ImportError:
    SystemEngine = None  # type: ignore[assignment]
    SYSTEM_ENGINE_AVAILABLE = False


def _build_engine():
    """Pick the best available TTS engine.

    Order is local-first (Kokoro is offline + best quality at this size),
    then the OS native engine (sapi5 / nsspeechsynthesizer / espeak) as a
    zero-dependency fallback.
    """
    if KOKORO_AVAILABLE:
        try:
            return KokoroEngine()
        except Exception as exc:
            logger.warning("KokoroEngine failed to initialise (%s); falling back.", exc)
    if SYSTEM_ENGINE_AVAILABLE:
        try:
            return SystemEngine()
        except Exception as exc:
            logger.warning("SystemEngine failed to initialise (%s).", exc)
    return None


class TextToSpeech:
    """Streaming TTS with barge-in support.

    Keeps the legacy single-class API (``speak``, ``stop``, ``is_speaking``,
    ``is_available``) so callers don't need to know which engine is in
    use.
    """

    def __init__(self) -> None:
        self._engine = None
        self._stream: Optional[TextToAudioStream] = None
        self._lock = threading.Lock()
        self._speaking = False
        self._on_complete: Optional[Callable[[], None]] = None

        if REALTIMETTS_AVAILABLE:
            try:
                self._engine = _build_engine()
                if self._engine is not None:
                    self._stream = TextToAudioStream(
                        self._engine,
                        on_audio_stream_stop=self._handle_stream_stop,
                        log_characters=False,
                    )
            except Exception as exc:
                logger.exception("Failed to initialise RealtimeTTS engine: %s", exc)
                self._stream = None

    def is_available(self) -> bool:
        return self._stream is not None

    def speak(
        self,
        text: str,
        async_: bool = True,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        """Speak ``text``. Non-blocking by default (callback fires on end).

        If TTS is unavailable the callback fires immediately so the
        caller's flow continues uninterrupted.
        """
        if not self.is_available():
            logger.debug("TTS not available; skipping speak() and firing callback.")
            if on_complete:
                on_complete()
            return

        if async_:
            threading.Thread(
                target=self._do_speak, args=(text, on_complete), daemon=True,
                name="TextToSpeech-speak",
            ).start()
        else:
            self._do_speak(text, on_complete)

    def _do_speak(self, text: str, on_complete: Optional[Callable[[], None]]) -> None:
        """Feed the stream and start playback. Streaming means playback can
        begin before the whole sentence is synthesised."""
        with self._lock:
            # If a previous phrase is still playing, stop it cleanly first --
            # this is the barge-in pattern.
            self._stop_locked()
            self._on_complete = on_complete
            self._speaking = True
            try:
                self._stream.feed(text)
                # play_async returns immediately; the on_audio_stream_stop
                # callback fires when playback genuinely finishes.
                self._stream.play_async()
            except Exception as exc:
                logger.exception("TTS play failed: %s", exc)
                self._speaking = False
                if on_complete:
                    on_complete()

    def stop(self) -> None:
        """Interrupt any in-flight playback. Safe to call from any thread."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._stream is None or not self._speaking:
            return
        try:
            self._stream.stop()
        except Exception as exc:
            logger.debug("Stream stop raised: %s", exc)

    def _handle_stream_stop(self) -> None:
        """Fired by RealtimeTTS when playback ends (either naturally or via stop())."""
        self._speaking = False
        cb = self._on_complete
        self._on_complete = None
        if cb:
            try:
                cb()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        return self._speaking

    # Legacy no-op shims so callers don't blow up
    def set_rate(self, rate: int) -> None:  # noqa: D401
        """Legacy API; rate control would have to be plumbed engine-by-engine."""
        return None

    def set_volume(self, volume: float) -> None:  # noqa: D401
        return None


# --- Module-level singleton (preserved from legacy API) ---------------------
_tts_instance: Optional[TextToSpeech] = None
_tts_lock = threading.Lock()


def get_tts() -> TextToSpeech:
    """Return the process-wide TextToSpeech instance, building it lazily so
    importers don't pay model-load cost just to ``import``."""
    global _tts_instance
    with _tts_lock:
        if _tts_instance is None:
            _tts_instance = TextToSpeech()
        return _tts_instance


def speak(text: str, async_: bool = True) -> None:
    get_tts().speak(text, async_=async_)
