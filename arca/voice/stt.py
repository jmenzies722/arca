"""
Speech-to-Text using faster-whisper (local, Metal GPU on Apple Silicon).

faster-whisper is a reimplementation of OpenAI Whisper using CTranslate2,
optimized for speed. On Apple Silicon with Metal, base.en transcribes in ~300ms.

Models (auto-downloaded on first run):
  tiny.en   (~75MB)  — fastest, lower accuracy
  base.en   (~145MB) — recommended default
  small.en  (~466MB) — better accuracy, still fast
  medium.en (~1.4GB) — high accuracy
  large-v3  (~3GB)   — best accuracy, slower

Flow:
  1. Receive audio as numpy float32 array (16kHz mono)
  2. Pass to WhisperModel.transcribe()
  3. Stream segments back as they're decoded
  4. Join all segments → final text string
"""

from __future__ import annotations

from typing import Generator, Iterator

import numpy as np

try:
    from faster_whisper import WhisperModel as _WhisperModel

    _FW_AVAILABLE = True
except ImportError:
    _FW_AVAILABLE = False


class STT:
    """Local Whisper STT via faster-whisper."""

    def __init__(self, model_size: str = "base.en", device: str = "auto", compute_type: str = "auto"):
        """
        Args:
            model_size: Whisper model variant. 'base.en' is a good default.
            device: 'auto' detects CUDA/Metal. Falls back to CPU.
            compute_type: 'auto' picks best dtype for device. Options: float16, int8, float32.
        """
        if not _FW_AVAILABLE:
            raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

        # On Apple Silicon, faster-whisper uses CoreML/Metal via 'auto'
        # On CUDA machines, it uses float16
        # On CPU-only, it uses int8 (quantized)
        self._model = _WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Float32 numpy array at 16kHz mono.
            language: Language code. 'en' enables English-only models.

        Returns:
            Transcribed text string.
        """
        if len(audio) == 0:
            return ""

        segments, _info = self._model.transcribe(
            audio,
            beam_size=5,
            language=language,
            vad_filter=True,               # Built-in VAD to skip silent chunks
            vad_parameters={"min_silence_duration_ms": 300},
        )

        return " ".join(seg.text.strip() for seg in segments).strip()

    def transcribe_stream(self, audio: np.ndarray, language: str = "en") -> Iterator[str]:
        """
        Transcribe audio, yielding text as each segment is decoded.
        Use this to show partial transcription in the UI while still processing.
        """
        if len(audio) == 0:
            return

        segments, _ = self._model.transcribe(
            audio,
            beam_size=5,
            language=language,
            vad_filter=True,
        )

        for segment in segments:
            yield segment.text.strip()
