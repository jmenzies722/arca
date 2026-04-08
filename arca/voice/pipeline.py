"""
Voice Pipeline — orchestrates VAD + STT into a single "listen until done" call.

This is the main interface for capturing voice input:
  1. VAD opens mic and waits for speech
  2. Speech detected → STT begins buffering audio
  3. Silence detected → audio finalized → STT transcribes
  4. Text returned to caller

Two modes:
  - push_to_talk: Recording only while key held (Ctrl+Space)
  - vad_auto: Always listening, VAD determines start/end

The pipeline is designed to be called from a background thread so the
UI remains responsive during recording.
"""

from __future__ import annotations

import threading
from typing import Callable, Literal

import numpy as np

from .stt import STT
from .tts import TTS
from .vad import VAD


class VoicePipeline:
    """End-to-end: mic → VAD → STT → text."""

    def __init__(
        self,
        stt: STT,
        tts: TTS,
        vad: VAD,
        mode: Literal["push_to_talk", "vad_auto"] = "push_to_talk",
    ):
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.mode = mode

        self._recording = False
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._recording

    def listen(
        self,
        on_recording_start: Callable | None = None,
        on_recording_end: Callable | None = None,
        on_transcribing: Callable | None = None,
    ) -> str:
        """
        Listen for one utterance and return transcribed text.

        Callbacks fire at each stage so the UI can show live feedback:
          on_recording_start → show mic indicator
          on_recording_end   → show "transcribing..." indicator
          on_transcribing    → hide mic indicator
        """
        with self._lock:
            self._recording = True

        try:
            # Step 1: Record audio via VAD
            audio = self.vad.record_until_silence(
                on_speech_start=on_recording_start,
                on_speech_end=lambda: (
                    self._set_recording(False),
                    on_recording_end() if on_recording_end else None,
                ),
            )

            if len(audio) == 0:
                return ""

            # Step 2: Transcribe
            if on_transcribing:
                on_transcribing()

            text = self.stt.transcribe(audio)
            return text.strip()

        finally:
            with self._lock:
                self._recording = False

    def listen_async(
        self,
        on_result: Callable[[str], None],
        on_recording_start: Callable | None = None,
        on_recording_end: Callable | None = None,
        on_transcribing: Callable | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> threading.Thread:
        """
        Non-blocking listen — runs in background thread.
        Calls on_result(text) when transcription is ready.
        Returns the thread for callers that need to join/cancel.
        """
        def _run():
            try:
                text = self.listen(
                    on_recording_start=on_recording_start,
                    on_recording_end=on_recording_end,
                    on_transcribing=on_transcribing,
                )
                if text:
                    on_result(text)
            except Exception as e:
                if on_error:
                    on_error(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def _set_recording(self, value: bool) -> None:
        with self._lock:
            self._recording = value
