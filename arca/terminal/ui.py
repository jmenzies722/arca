"""
Arca Terminal UI — built with Textual.

Layout:
  ┌─────────────────────────────────────────┐
  │  ARCA  v0.1.0          [●] Recording   │  ← Header
  ├─────────────────────────────────────────┤
  │                                         │
  │  [output log — scrollable]              │  ← RichLog (main output area)
  │  Shows: AI responses, tool output,      │
  │         shell results, system messages  │
  │                                         │
  ├─────────────────────────────────────────┤
  │  > Type or Ctrl+Space to speak...       │  ← Input bar
  └─────────────────────────────────────────┘
  Ctrl+Space: voice | Ctrl+R: reset | Ctrl+C: quit

Key bindings:
  Ctrl+Space  — toggle voice recording
  Ctrl+R      — reset conversation
  Ctrl+L      — clear display
  Ctrl+C      — quit
  Enter       — submit text input

The UI is event-driven. The agent loop runs in a background worker thread
and posts events back to the UI thread for display updates.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Callable

from rich.markup import escape
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static


# ─── Custom Widgets ────────────────────────────────────────────────────────────

class StatusBar(Static):
    """Shows recording/thinking status."""

    DEFAULT_CSS = """
    StatusBar {
        background: $surface;
        color: $text-muted;
        height: 1;
        padding: 0 1;
    }
    StatusBar.recording {
        color: $error;
    }
    StatusBar.thinking {
        color: $warning;
    }
    """

    def set_idle(self) -> None:
        self.remove_class("recording", "thinking")
        self.update("  Ctrl+Space: voice  |  Ctrl+R: reset  |  Ctrl+L: clear  |  Ctrl+C: quit")

    def set_recording(self) -> None:
        self.add_class("recording")
        self.remove_class("thinking")
        self.update("  ● Recording... (release Ctrl+Space or wait for silence)")

    def set_thinking(self, message: str = "Thinking...") -> None:
        self.remove_class("recording")
        self.add_class("thinking")
        self.update(f"  ◐ {message}")


# ─── Main App ──────────────────────────────────────────────────────────────────

class ArcaApp(App):
    """The main Arca terminal application."""

    TITLE = "Arca"
    SUB_TITLE = "AI Voice Terminal  v0.1.0"

    CSS = """
    Screen {
        background: $background;
    }

    #main-container {
        height: 1fr;
        border: none;
    }

    #output {
        height: 1fr;
        border: none;
        padding: 0 1;
        scrollbar-gutter: stable;
    }

    #input-bar {
        height: 3;
        padding: 0 1;
        border-top: solid $surface;
    }

    #user-input {
        border: none;
        background: $surface;
    }

    StatusBar {
        height: 1;
        border-top: solid $surface;
    }
    """

    BINDINGS = [
        Binding("ctrl+space", "toggle_voice", "Voice", show=True),
        Binding("ctrl+r", "reset_conversation", "Reset", show=True),
        Binding("ctrl+l", "clear_output", "Clear", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    def __init__(
        self,
        agent_runner: Callable[[str], None],
        voice_trigger: Callable | None = None,
        **kwargs,
    ):
        """
        Args:
            agent_runner: Callable that takes user text and runs the agent.
                          Should post results back via self.post_message or callbacks.
            voice_trigger: Optional callable to trigger voice recording.
        """
        super().__init__(**kwargs)
        self._agent_runner = agent_runner
        self._voice_trigger = voice_trigger
        self._is_recording = False
        self._is_thinking = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main-container"):
            yield RichLog(id="output", markup=True, highlight=True, wrap=True)
            with Container(id="input-bar"):
                yield Input(
                    placeholder="Type a message or press Ctrl+Space to speak...",
                    id="user-input",
                )
        yield StatusBar(id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#status", StatusBar).set_idle()
        self._print_welcome()
        self.query_one("#user-input", Input).focus()

    def _print_welcome(self) -> None:
        log = self.query_one("#output", RichLog)
        log.write(Text.from_markup(
            "[bold cyan]Arca[/] [dim]v0.1.0[/] — AI Voice Terminal\n"
            "[dim]Speak or type. Ctrl+Space for voice. Ctrl+R to reset.[/]\n"
        ))

    # ─── Input Handling ─────────────────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """User pressed Enter in the text input."""
        text = event.value.strip()
        if not text:
            return

        event.input.clear()
        self._handle_user_input(text)

    def action_toggle_voice(self) -> None:
        """Ctrl+Space — start/stop voice recording."""
        if self._is_recording or self._is_thinking:
            return

        if self._voice_trigger:
            self._voice_trigger()
        else:
            self._print_system("Voice not configured. Set voice_enabled=true in arca.toml")

    def action_reset_conversation(self) -> None:
        """Ctrl+R — clear conversation history."""
        self._print_system("Conversation reset.")

    def action_clear_output(self) -> None:
        """Ctrl+L — clear the display."""
        self.query_one("#output", RichLog).clear()

    # ─── Output Rendering ───────────────────────────────────────────────────────

    def print_user(self, text: str) -> None:
        """Render a user message."""
        log = self.query_one("#output", RichLog)
        log.write(Text.from_markup(f"\n[bold green]You[/]  {escape(text)}"))

    def print_assistant(self, text: str) -> None:
        """Render a complete assistant response."""
        log = self.query_one("#output", RichLog)
        log.write(Text.from_markup(f"\n[bold cyan]Arca[/]  {escape(text)}"))

    def print_assistant_chunk(self, chunk: str) -> None:
        """Append a streaming text chunk to the last line (no newline prefix)."""
        log = self.query_one("#output", RichLog)
        log.write(Text(chunk), end="")  # type: ignore[call-arg]

    def print_tool(self, name: str, inputs: dict, result: dict | None = None) -> None:
        """Render a tool call + result."""
        log = self.query_one("#output", RichLog)
        input_summary = ", ".join(f"{k}={repr(v)[:40]}" for k, v in inputs.items())
        log.write(Text.from_markup(f"\n[dim]  → {name}({input_summary})[/]"))
        if result:
            if "error" in result:
                log.write(Text.from_markup(f"  [red]✗ {escape(result['error'])}[/]"))
            elif "stdout" in result and result["stdout"]:
                # Show first 10 lines of shell output
                lines = result["stdout"].strip().splitlines()[:10]
                for line in lines:
                    log.write(Text.from_markup(f"  [dim]{escape(line)}[/]"))
                if len(result["stdout"].splitlines()) > 10:
                    log.write(Text.from_markup("  [dim]... (truncated)[/]"))

    def _print_system(self, message: str) -> None:
        log = self.query_one("#output", RichLog)
        log.write(Text.from_markup(f"\n[dim italic]{escape(message)}[/]"))

    # ─── Agent Coordination ─────────────────────────────────────────────────────

    def _handle_user_input(self, text: str) -> None:
        """Route user text through the agent in a background thread."""
        self.print_user(text)
        self._set_thinking(True)

        def run():
            try:
                self._agent_runner(text)
            finally:
                self.call_from_thread(self._set_thinking, False)

        threading.Thread(target=run, daemon=True).start()

    def set_recording_state(self, recording: bool) -> None:
        """Called from voice pipeline thread."""
        self._is_recording = recording
        status = self.query_one("#status", StatusBar)
        if recording:
            self.call_from_thread(status.set_recording)
        else:
            self.call_from_thread(status.set_idle)

    def _set_thinking(self, thinking: bool) -> None:
        self._is_thinking = thinking
        status = self.query_one("#status", StatusBar)
        if thinking:
            status.set_thinking()
        else:
            status.set_idle()
