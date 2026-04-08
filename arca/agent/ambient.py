"""
Ambient Mode — Arca watches your terminal and speaks up proactively.

When ambient mode is enabled, Arca runs a lightweight background monitor:
  1. Shell commands and their output are piped through a queue
  2. An error classifier checks each result for anomalies
  3. When an error is detected, a mini Claude query generates a suggestion
  4. Suggestion is spoken via TTS and shown in the TUI

This is the "pair programmer looking over your shoulder" experience.
It's opt-in (ambient_mode = true in arca.toml) and intentionally low-volume:
  - Rate limited: max 1 suggestion per 30 seconds
  - Only fires on clear error signals (non-zero exit, exception traces, build failures)
  - Suggestion is brief (1-2 sentences max)

The suggestion is generated with a lean prompt to minimize latency:
  - No conversation history (fresh context per error)
  - Max 150 tokens (forces conciseness)
  - Uses claude-haiku-4-5 for speed (fast + cheap)

How commands reach the ambient monitor:
  The agent's shell_exec tool calls AmbientMonitor.feed() after each execution.
  In Phase 3 (PTY integration), the live shell output will also feed the monitor.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable


# Patterns that indicate an error worth flagging
ERROR_PATTERNS = [
    re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r"error:", re.IGNORECASE),
    re.compile(r"exception:", re.IGNORECASE),
    re.compile(r"failed to", re.IGNORECASE),
    re.compile(r"cannot find module", re.IGNORECASE),
    re.compile(r"modulenotfounderror", re.IGNORECASE),
    re.compile(r"syntaxerror", re.IGNORECASE),
    re.compile(r"permission denied", re.IGNORECASE),
    re.compile(r"command not found", re.IGNORECASE),
    re.compile(r"no such file or directory", re.IGNORECASE),
    re.compile(r"build failed", re.IGNORECASE),
    re.compile(r"compilation failed", re.IGNORECASE),
    re.compile(r"tests? failed", re.IGNORECASE),
    re.compile(r"npm err", re.IGNORECASE),
]


@dataclass
class ShellEvent:
    command: str
    stdout: str
    stderr: str
    exit_code: int


class AmbientMonitor:
    """
    Background monitor that watches shell output and generates proactive suggestions.
    """

    def __init__(
        self,
        on_suggestion: Callable[[str], None],
        anthropic_api_key: str = "",
        model: str = "claude-haiku-4-5-20251001",
        min_interval_s: float = 30.0,
    ):
        """
        Args:
            on_suggestion: Callback fired with suggestion text.
            anthropic_api_key: Claude API key.
            model: Fast model for ambient suggestions. Haiku is ideal.
            min_interval_s: Minimum seconds between suggestions (rate limit).
        """
        self.on_suggestion = on_suggestion
        self._api_key = anthropic_api_key
        self._model = model
        self._min_interval = min_interval_s

        self._queue: queue.Queue[ShellEvent] = queue.Queue(maxsize=50)
        self._last_suggestion = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._client = None

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        except ImportError:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="arca-ambient",
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def feed(self, command: str, stdout: str, stderr: str, exit_code: int) -> None:
        """
        Feed a completed shell command result to the monitor.
        Called by the agent's shell_exec tool after each command.
        Non-blocking — drops events if queue is full.
        """
        if not self._running:
            return

        event = ShellEvent(
            command=command,
            stdout=stdout[-2000:],   # Last 2000 chars
            stderr=stderr[-2000:],
            exit_code=exit_code,
        )
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass  # Drop silently — queue full means we're busy

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                event = self._queue.get(timeout=1.0)
                self._process_event(event)
            except queue.Empty:
                continue

    def _process_event(self, event: ShellEvent) -> None:
        """Check if event warrants a suggestion. Generate if so."""
        # Rate limit
        now = time.time()
        if now - self._last_suggestion < self._min_interval:
            return

        # Only react to failures
        if not self._is_error(event):
            return

        self._last_suggestion = now
        suggestion = self._generate_suggestion(event)
        if suggestion:
            self.on_suggestion(suggestion)

    def _is_error(self, event: ShellEvent) -> bool:
        """Return True if this event looks like an error worth flagging."""
        if event.exit_code != 0:
            return True

        combined = (event.stdout + event.stderr).lower()
        return any(p.search(combined) for p in ERROR_PATTERNS)

    def _generate_suggestion(self, event: ShellEvent) -> str:
        """Call Claude for a brief helpful suggestion."""
        if not self._client:
            return ""

        output = (event.stderr or event.stdout)[:800].strip()
        prompt = (
            f"Command: {event.command}\n"
            f"Exit code: {event.exit_code}\n"
            f"Output:\n{output}\n\n"
            "In one or two sentences, what's the likely cause and fix? "
            "Be direct. No preamble."
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            return ""
