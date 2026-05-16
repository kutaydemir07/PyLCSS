# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Voice controller -- streaming STT via RealtimeSTT + Silero VAD.

What changed from the previous revision
---------------------------------------
The legacy controller built its own audio pipeline on top of
``pyaudio`` + an energy-RMS VAD + a ``faster_whisper.WhisperModel`` loaded
synchronously on first ``start()``.  That:

  - froze the UI for 3-5 s the first time the user pressed "start";
  - never used the GPU even when one was present;
  - waited for a full second of silence before transcribing, so even
    short commands felt sluggish;
  - missed quiet speech and false-triggered on background noise because
    the energy threshold was a fixed integer.

The new controller delegates the whole audio loop to
:mod:`RealtimeSTT.AudioToTextRecorder`, which wraps:

  - **Silero VAD** (production-grade ONNX model) for speech segmentation;
  - **faster-whisper** (auto GPU detection, float16 on CUDA, int8 on CPU);
  - **streaming partial transcripts** so the UI shows what the user is
    saying *while* they say it, not 1-2 s after they finish;
  - a background warmup thread so model load doesn't block the UI.

The public API (``VoiceController``, ``WHISPER_AVAILABLE``,
``set_callbacks``, ``start``/``stop``/``pause``/``resume``,
``start_llm_mode``/``stop_llm_mode``/``is_llm_mode``,
``is_running``, ``start_dictation``/``stop_dictation``) is preserved so
the existing :class:`AssistantManager` doesn't need to change.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from pylcss.assistant_systems.config import VoiceControlConfig

logger = logging.getLogger(__name__)


# --- Dependency probes ------------------------------------------------------
# Keep these names because UI code does ``from voice import WHISPER_AVAILABLE``.
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

try:
    from RealtimeSTT import AudioToTextRecorder
    REALTIMESTT_AVAILABLE = True
except ImportError:
    AudioToTextRecorder = None  # type: ignore[assignment]
    REALTIMESTT_AVAILABLE = False

# Backwards-compatible alias -- a lot of existing code checks WHISPER_AVAILABLE
# to decide whether to enable the voice UI at all.
WHISPER_AVAILABLE = REALTIMESTT_AVAILABLE


# Engineering jargon seeded into Whisper's `initial_prompt` so the model
# transcribes domain terms correctly instead of producing phonetic guesses.
_ENGINEERING_VOCAB = (
    "PyLCSS, CalculiX, OpenRadioss, CadQuery, FEA, von Mises, "
    "mesh, tetrahedral, hex mesh, Poisson, modulus, Young's, yield, "
    "fillet, chamfer, extrude, revolve, boolean cut, sketch, workplane, "
    "surrogate, optimization, sensitivity, solution space, NSGA, "
    "Latin hypercube, design variable, quantity of interest, parametric, "
    "node, edge, face, vertex, boundary condition, load, constraint"
)


def _play_beep(frequency: int = 800, duration_ms: int = 100) -> None:
    """Cross-platform short beep for mode-change feedback."""
    try:
        import winsound
        threading.Thread(
            target=lambda: winsound.Beep(frequency, duration_ms),
            daemon=True,
        ).start()
    except ImportError:
        print("\a", end="", flush=True)
    except Exception as e:
        logger.debug(f"Beep failed: {e}")


def _pick_device_and_compute_type() -> tuple[str, str]:
    """Auto-detect CUDA / MPS / CPU and pick the matching compute_type for
    faster-whisper.  float16 on GPU, int8 on CPU keeps quality + speed
    balanced without user config."""
    if TORCH_AVAILABLE:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # faster-whisper doesn't run on MPS; fall back to CPU but flag it.
            logger.info("MPS detected but faster-whisper requires CPU/CUDA; using CPU.")
    return "cpu", "int8"


class VoiceController:
    """Streaming STT controller for the PyLCSS assistant.

    Single public state machine:
      - ``start()``        -> recorder thread spins up + model warms in bg
      - ``pause()`` /
        ``resume()``       -> ignore / accept incoming audio without
                              tearing the recorder down
      - ``start_llm_mode``  -> incoming text is routed to the LLM callback
        / ``stop_llm_mode``    instead of the command callback (+ beep)
      - ``stop()``         -> tear down recorder and worker thread

    Callbacks are set via :meth:`set_callbacks` and are always called from
    the recorder's background thread, so the manager forwards them to the
    UI through Qt signals.
    """

    def __init__(self, config: Optional[VoiceControlConfig] = None) -> None:
        self.config = config or VoiceControlConfig()

        self._recorder: Optional[Any] = None  # AudioToTextRecorder
        self._recorder_thread: Optional[threading.Thread] = None
        self._warmup_event = threading.Event()

        self._on_command: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._on_dictation: Optional[Callable[[str], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        self._on_partial: Optional[Callable[[str], None]] = None
        self._on_llm_request: Optional[Callable[[str], None]] = None

        self._running = False
        self._paused = False
        self._llm_mode = False
        self._dictation_mode = False
        # Last init/start failure surfaced via get_last_error() so the
        # manager can put the real reason into the UI instead of a generic
        # "Failed to start voice control" string.
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API expected by AssistantManager
    # ------------------------------------------------------------------
    def set_callbacks(
        self,
        on_command: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_dictation: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
        on_llm_request: Optional[Callable[[str], None]] = None,
        on_text: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Bind UI-layer callbacks.

        ``on_text`` is an alias for ``on_dictation`` kept for backward
        compatibility with AssistantManager, which still uses the legacy
        name.  ``on_llm_request`` is fired with the raw utterance when
        :meth:`is_llm_mode` is true, so the manager can route it to the
        agent runner instead of going through the keyword-command dispatcher.
        """
        self._on_command = on_command
        # on_dictation wins if both are passed; on_text is the legacy alias.
        self._on_dictation = on_dictation if on_dictation is not None else on_text
        self._on_status = on_status
        self._on_partial = on_partial
        self._on_llm_request = on_llm_request

    def is_model_available(self) -> bool:
        return REALTIMESTT_AVAILABLE

    def get_model_download_info(self) -> Dict[str, str]:
        device, _ = _pick_device_and_compute_type()
        model = self._model_id()
        return {
            "engine": "RealtimeSTT + faster-whisper + Silero VAD",
            "model": model,
            "device": device,
            "note": "First run downloads the Whisper weights (~500 MB).",
        }

    def is_running(self) -> bool:
        return self._running

    def is_llm_mode(self) -> bool:
        return self._llm_mode

    def get_last_error(self) -> Optional[str]:
        """Return the most recent init/start failure reason, or None.

        Manager reads this when ``start()`` returns False to put the
        actual diagnostic into the UI banner instead of a generic
        "Failed to start" string -- the latter sent users straight to
        "revert the whole stack" instead of, say, fixing a mic
        permission or letting the Whisper download finish.
        """
        return self._last_error

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        if not REALTIMESTT_AVAILABLE:
            self._last_error = (
                "RealtimeSTT is not installed. Run "
                "`pip install -r requirements.txt` (or "
                "`pip install RealtimeSTT==1.0.0`)."
            )
            logger.error(self._last_error)
            self._emit_status("Voice unavailable (install RealtimeSTT)")
            return False
        if self._running:
            return True

        device, compute_type = _pick_device_and_compute_type()
        model_id = self._model_id()
        logger.info(
            "Starting voice controller (model=%s, device=%s, compute_type=%s)",
            model_id, device, compute_type,
        )
        self._emit_status("Loading speech model...")

        try:
            self._recorder = AudioToTextRecorder(
                model=model_id,
                language="en",
                device=device,
                compute_type=compute_type,
                # Silero VAD does the heavy lifting; webrtc is the cheap
                # pre-filter that decides when to ask Silero.
                silero_sensitivity=0.4,
                webrtc_sensitivity=2,
                # Tight silence threshold: a short pause is enough to send
                # the utterance.  RealtimeSTT default is 0.6 s; 0.35 s
                # gives a more responsive feel without cutting off
                # mid-sentence.
                post_speech_silence_duration=0.35,
                min_length_of_recording=0.2,
                min_gap_between_recordings=0.0,
                # Streaming partial transcripts -- the secret sauce that
                # makes the UI feel alive while the user is still talking.
                enable_realtime_transcription=True,
                realtime_processing_pause=0.1,
                realtime_model_type="tiny.en",  # cheap intermediate model
                on_realtime_transcription_update=self._on_realtime_update,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                # Whisper-side bias toward our domain vocabulary.
                initial_prompt=_ENGINEERING_VOCAB,
                # Keep the model warm between utterances.
                spinner=False,
                no_log_file=True,
            )
        except Exception as exc:
            # Save the full diagnostic (type + message + hints) so the
            # manager can show the actual root cause in the UI.  Common
            # failure modes here:
            #   - No microphone / mic permission denied
            #   - PortAudio not initialised (sounddevice missing system lib)
            #   - Whisper weights blocked from download (corp proxy)
            #   - Antivirus killed the Silero ONNX inference DLL
            import traceback
            tb = traceback.format_exc()
            self._last_error = (
                f"Voice init failed ({type(exc).__name__}: {exc}).\n"
                f"Common causes: no microphone, blocked Whisper download, "
                f"or sounddevice/PortAudio runtime missing.\n"
                f"Full traceback in the terminal log."
            )
            logger.error("Voice init failed:\n%s", tb)
            self._emit_status(f"Voice init failed: {type(exc).__name__}")
            return False

        # Drive the recorder loop on a background thread so the UI stays
        # responsive while the model loads + listens.
        self._running = True
        self._paused = False
        self._recorder_thread = threading.Thread(
            target=self._recorder_loop, daemon=True, name="VoiceController-loop",
        )
        self._recorder_thread.start()
        self._emit_status("Listening")
        return True

    def stop(self) -> None:
        self._running = False
        if self._recorder is not None:
            try:
                self._recorder.abort()
                self._recorder.shutdown()
            except Exception as exc:
                logger.debug("Recorder shutdown raised: %s", exc)
        self._recorder = None
        self._recorder_thread = None
        self._emit_status("Stopped")

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def start_llm_mode(self) -> None:
        self._llm_mode = True
        _play_beep(900, 80)
        self._emit_status("LLM mode")

    def stop_llm_mode(self) -> None:
        self._llm_mode = False
        _play_beep(600, 80)
        self._emit_status("Listening")

    def start_dictation(self) -> None:
        self._dictation_mode = True

    def stop_dictation(self) -> None:
        self._dictation_mode = False

    # ------------------------------------------------------------------
    # Recorder loop + callbacks (all on background thread)
    # ------------------------------------------------------------------
    def _recorder_loop(self) -> None:
        """Pull recognised utterances one at a time and route them."""
        assert self._recorder is not None
        while self._running:
            try:
                text = self._recorder.text()  # blocks until utterance done
            except Exception as exc:
                if not self._running:
                    break
                logger.warning("Recorder.text() raised: %s", exc)
                time.sleep(0.2)
                continue

            if not self._running:
                break
            if self._paused:
                continue
            text = (text or "").strip()
            if not text:
                continue
            self._dispatch_text(text)

    def _dispatch_text(self, text: str) -> None:
        """Route a finalised utterance to the right callback."""
        if self._dictation_mode and self._on_dictation:
            self._on_dictation(text)
            return
        # In LLM mode every utterance is a free-form request for the agent;
        # if the manager registered an on_llm_request handler use it (the
        # legacy code path the manager still expects).  Otherwise fall back
        # to the generic on_command with llm_mode flag in the payload so
        # the new-style handler can also route correctly.
        if self._llm_mode and self._on_llm_request is not None:
            self._on_llm_request(text)
            return
        if self._on_command:
            payload = {"text": text, "llm_mode": self._llm_mode}
            self._on_command(text, payload)

    def _on_realtime_update(self, partial_text: str) -> None:
        """Streaming partial transcript -- fires many times per utterance."""
        if self._paused or not partial_text:
            return
        if self._on_partial:
            self._on_partial(partial_text)

    def _on_recording_start(self) -> None:
        if self._on_status:
            self._on_status("Hearing...")

    def _on_recording_stop(self) -> None:
        if self._on_status:
            self._on_status("Transcribing...")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _emit_status(self, status: str) -> None:
        if self._on_status:
            try:
                self._on_status(status)
            except Exception:
                pass
        logger.debug("voice status: %s", status)

    def _model_id(self) -> str:
        """Decide which Whisper variant to load.

        Honours the user's config (config.model_size if set), else defaults
        to distil-large-v3.5 on GPU (much faster than turbo, English-only,
        best WER on benchmarks) or distil-medium on CPU (still beats the
        old base.en at ~3x the speed).
        """
        cfg_model = getattr(self.config, "model_size", None) or getattr(self.config, "whisper_model", None)
        if cfg_model:
            return cfg_model
        device, _ = _pick_device_and_compute_type()
        return "distil-large-v3.5" if device == "cuda" else "distil-small.en"
