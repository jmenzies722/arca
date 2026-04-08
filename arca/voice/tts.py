"""
Text-to-Speech — supports Kokoro (local) and OpenAI TTS (cloud).

Kokoro-82M is a local ONNX TTS model:
  - #1 on HuggingFace TTS Arena
  - Runs on CPU (~200-500ms with persistent server)
  - 26 voices across American/British English + more
  - Download: pip install kokoro-onnx

OpenAI TTS is a fallback cloud option (~500ms API latency).

Flow:
  1. Receive text string
  2. Run through selected TTS engine
  3. Play audio through system speakers via sounddevice
  4. Return when audio finishes playing (blocking by default)

Kokoro Voice IDs (selection):
  af_heart   — warm American female (default)
  af_nova    — energetic American female
  am_echo    — deep American male
  am_onyx    — neutral American male
  bf_emma    — British female
  bm_george  — British male
"""

from __future__ import annotations

import io
import os
from typing import Literal

import numpy as np


class TTS:
    """Text-to-Speech — local Kokoro or cloud OpenAI."""

    def __init__(
        self,
        engine: Literal["kokoro", "openai", "none"] = "kokoro",
        voice: str = "af_heart",
        speed: float = 1.1,
        openai_voice: str = "nova",
        openai_api_key: str | None = None,
    ):
        self.engine = engine
        self.voice = voice
        self.speed = speed
        self.openai_voice = openai_voice
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._kokoro = None

        if engine == "kokoro":
            self._init_kokoro()

    _KOKORO_DIR = os.path.expanduser("~/.arca/kokoro")
    _MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    _VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    def _init_kokoro(self) -> None:
        """Load Kokoro model from ~/.arca/kokoro/, downloading if needed."""
        try:
            from kokoro_onnx import Kokoro

            os.makedirs(self._KOKORO_DIR, exist_ok=True)
            model_path = os.path.join(self._KOKORO_DIR, "kokoro-v1.0.onnx")
            voices_path = os.path.join(self._KOKORO_DIR, "voices-v1.0.bin")

            if not os.path.exists(model_path):
                print("[arca/tts] Downloading Kokoro ONNX model (~82MB)...")
                self._download(self._MODEL_URL, model_path)

            if not os.path.exists(voices_path):
                print("[arca/tts] Downloading Kokoro voices (~11MB)...")
                self._download(self._VOICES_URL, voices_path)

            self._kokoro = Kokoro(model_path, voices_path)

        except ImportError:
            print("[arca/tts] kokoro-onnx not installed. Install: pip install kokoro-onnx")
            print("[arca/tts] Falling back to no TTS output.")
            self.engine = "none"
        except Exception as e:
            print(f"[arca/tts] Failed to initialize Kokoro: {e}")
            print("[arca/tts] Falling back to no TTS output.")
            self.engine = "none"

    @staticmethod
    def _download(url: str, dest: str) -> None:
        import urllib.request
        urllib.request.urlretrieve(url, dest)

    def speak(self, text: str, blocking: bool = True) -> None:
        """
        Synthesize text and play through speakers.

        Args:
            text: Text to speak.
            blocking: If True, waits for audio to finish. Set False for background play.
        """
        if not text or self.engine == "none":
            return

        if self.engine == "kokoro":
            self._speak_kokoro(text, blocking)
        elif self.engine == "openai":
            self._speak_openai(text, blocking)

    def _speak_kokoro(self, text: str, blocking: bool) -> None:
        import sounddevice as sd

        if self._kokoro is None:
            return

        samples, sample_rate = self._kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang="en-us",
        )

        sd.play(samples, sample_rate)
        if blocking:
            sd.wait()

    def _speak_openai(self, text: str, blocking: bool) -> None:
        import sounddevice as sd
        import soundfile as sf

        try:
            import openai

            client = openai.OpenAI(api_key=self._openai_api_key)
            response = client.audio.speech.create(
                model="tts-1",
                voice=self.openai_voice,
                input=text,
            )

            audio_bytes = io.BytesIO(response.content)
            data, samplerate = sf.read(audio_bytes)
            sd.play(data, samplerate)
            if blocking:
                sd.wait()
        except Exception as e:
            print(f"[arca/tts] OpenAI TTS error: {e}")

    def stop(self) -> None:
        """Stop currently playing audio."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
