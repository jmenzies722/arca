"""
Wake Word Detection — always-on passive listening for "hey arca".

Uses openwakeword (open source, no API key, runs on CPU).
Runs in a background daemon thread, fires a callback when the wake word
is detected so the VoicePipeline can immediately begin recording.

How it works:
  1. Background thread opens mic at 16kHz
  2. Audio fed in chunks to openwakeword model
  3. Model outputs confidence score per wake word
  4. When score exceeds threshold → callback fired → full recording begins
  5. Thread continues listening for next activation

openwakeword models (~1MB each):
  - hey_jarvis         → closest to "hey arca" phonetically (use as default)
  - alexa              → shorter wake word pattern
  - Custom models can be trained at openWakeWord GitHub

Phase 2 note: We use "hey_jarvis" as a phonetic proxy. Phase 3 will train
a custom "hey arca" model using the openWakeWord training pipeline.

Alternative: pvporcupine (Pico Voice) — higher accuracy, requires API key.
Install: pip install pvporcupine
Config: set wakeword.engine = "porcupine" + wakeword.porcupine_key
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import numpy as np

# Chunk size for openwakeword — 80ms at 16kHz
_CHUNK_SIZE = 1280
_SAMPLE_RATE = 16000


class WakeWordDetector:
    """Passive background listener. Fires callback when wake word detected."""

    def __init__(
        self,
        on_detected: Callable,
        model: str = "hey_jarvis",
        threshold: float = 0.5,
        cooldown_s: float = 2.0,
        engine: str = "openwakeword",
        porcupine_key: str = "",
    ):
        """
        Args:
            on_detected: Callback fired when wake word is detected.
            model: openwakeword model name or path. Default: 'hey_jarvis'.
            threshold: Confidence threshold (0.0–1.0). Default: 0.5.
            cooldown_s: Seconds to ignore detections after one fires (prevents repeats).
            engine: 'openwakeword' or 'porcupine'.
            porcupine_key: Porcupine API key (only needed if engine='porcupine').
        """
        self.on_detected = on_detected
        self.model = model
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.engine = engine
        self.porcupine_key = porcupine_key

        self._running = False
        self._thread: threading.Thread | None = None
        self._last_detection = 0.0
        self._oww = None
        self._porcupine = None

    def start(self) -> None:
        """Start background listening thread."""
        if self._running:
            return

        if self.engine == "openwakeword":
            self._init_openwakeword()
        elif self.engine == "porcupine":
            self._init_porcupine()
        else:
            raise ValueError(f"Unknown wake word engine: {self.engine}")

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="arca-wakeword",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background listener."""
        self._running = False
        if self._porcupine:
            self._porcupine.delete()
            self._porcupine = None

    def _init_openwakeword(self) -> None:
        try:
            from openwakeword.model import Model as OWWModel

            self._oww = OWWModel(
                wakeword_models=[self.model],
                enable_speex_noise_suppression=False,
                vad_threshold=0.0,  # Disable internal VAD, we handle activation ourselves
            )
        except ImportError:
            raise RuntimeError(
                "openwakeword not installed.\n"
                "Install: pip install openwakeword\n"
                "Then:    python -m openwakeword --download_models"
            )

    def _init_porcupine(self) -> None:
        try:
            import pvporcupine

            keywords = ["hey siri"]  # Closest phonetically available; replace with custom
            self._porcupine = pvporcupine.create(
                access_key=self.porcupine_key,
                keywords=keywords,
            )
        except ImportError:
            raise RuntimeError(
                "pvporcupine not installed.\n"
                "Install: pip install pvporcupine\n"
                "Requires a Porcupine API key from console.picovoice.ai"
            )

    def _listen_loop(self) -> None:
        import sounddevice as sd

        audio_buffer: list[float] = []

        def callback(indata: np.ndarray, frames: int, time_info, status):
            audio_buffer.extend(indata[:, 0].tolist())

        with sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=_CHUNK_SIZE // 4,
            callback=callback,
        ):
            while self._running:
                if len(audio_buffer) >= _CHUNK_SIZE:
                    chunk = np.array(audio_buffer[:_CHUNK_SIZE], dtype=np.float32)
                    audio_buffer = audio_buffer[_CHUNK_SIZE:]
                    self._process_chunk(chunk)
                else:
                    time.sleep(0.01)

    def _process_chunk(self, chunk: np.ndarray) -> None:
        now = time.time()
        if now - self._last_detection < self.cooldown_s:
            return  # Cooldown active

        if self.engine == "openwakeword" and self._oww:
            self._oww.predict(chunk)
            scores = self._oww.prediction_buffer.get(self.model, [0.0])
            score = scores[-1] if scores else 0.0
            if score >= self.threshold:
                self._last_detection = now
                self.on_detected()

        elif self.engine == "porcupine" and self._porcupine:
            # Porcupine needs int16
            chunk_int16 = (chunk * 32767).astype(np.int16)
            keyword_idx = self._porcupine.process(chunk_int16[:self._porcupine.frame_length])
            if keyword_idx >= 0:
                self._last_detection = now
                self.on_detected()
