"""
Voice Activity Detection using Silero VAD.
Detects when speech starts and stops so we know when to begin/end recording.

Flow:
  1. Continuously read mic chunks
  2. Run each chunk through Silero VAD model
  3. When speech detected → signal recording start
  4. When silence detected for vad_silence_ms → signal recording end
"""

from __future__ import annotations

import collections
import threading
from typing import Callable

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Silero VAD requires 16kHz mono audio, 512 samples per chunk (32ms)
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms per chunk


class VAD:
    """Silero VAD wrapper for real-time speech detection."""

    def __init__(self, threshold: float = 0.5, silence_ms: int = 800):
        """
        Args:
            threshold: Speech probability threshold (0.0–1.0). Higher = less sensitive.
            silence_ms: How many ms of silence before we consider speech ended.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("silero-vad requires PyTorch. Install: pip install torch silero-vad")

        self.threshold = threshold
        self.silence_chunks = int(silence_ms / 32)  # Each chunk is 32ms

        # Load Silero VAD model (downloads on first run, ~2MB)
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._model.eval()
        self._get_speech_ts = self._utils[0]

    def is_speech(self, audio_chunk: np.ndarray) -> float:
        """
        Returns speech probability for a single 512-sample chunk.
        Call this on each 32ms frame of audio.
        """
        tensor = torch.from_numpy(audio_chunk.astype(np.float32))
        with torch.no_grad():
            prob = self._model(tensor, SAMPLE_RATE).item()
        return prob

    def record_until_silence(
        self,
        on_speech_start: Callable | None = None,
        on_speech_end: Callable | None = None,
    ) -> np.ndarray:
        """
        Record from mic until speech starts, then until silence is detected.

        Returns:
            Concatenated audio array (float32, 16kHz mono) of the speech segment.
        """
        import sounddevice as sd

        audio_buffer: list[np.ndarray] = []
        pre_roll: collections.deque[np.ndarray] = collections.deque(maxlen=10)  # 320ms pre-roll
        silence_count = 0
        speech_started = False
        stop_event = threading.Event()

        def callback(indata: np.ndarray, frames: int, time, status):
            nonlocal silence_count, speech_started

            chunk = indata[:, 0].copy()  # mono
            prob = self.is_speech(chunk)

            if prob >= self.threshold:
                # Speech detected
                if not speech_started:
                    speech_started = True
                    # Include pre-roll to catch the start of the utterance
                    audio_buffer.extend(pre_roll)
                    pre_roll.clear()
                    if on_speech_start:
                        on_speech_start()
                silence_count = 0
                audio_buffer.append(chunk)
            else:
                # Silence
                if speech_started:
                    audio_buffer.append(chunk)
                    silence_count += 1
                    if silence_count >= self.silence_chunks:
                        if on_speech_end:
                            on_speech_end()
                        stop_event.set()
                else:
                    pre_roll.append(chunk)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=callback,
        ):
            stop_event.wait(timeout=30)  # Max 30s recording

        if not audio_buffer:
            return np.array([], dtype=np.float32)

        return np.concatenate(audio_buffer)
