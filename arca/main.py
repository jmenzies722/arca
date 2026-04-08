"""
Arca — entry point.

Usage:
  arca                 # Launch full TUI with voice
  arca --no-voice      # Text-only mode (no mic/speaker)
  arca --text "query"  # One-shot: run query and exit
  arca --setup         # Interactive first-run setup

Boot sequence:
  1. Load config (arca.toml + ~/.arca/config.toml)
  2. Initialize memory (SQLite session)
  3. Initialize voice pipeline (STT + VAD + TTS) if enabled
  4. Initialize agent loop (Claude API)
  5. Launch Textual TUI
  6. Wire callbacks: voice → agent → UI → TTS
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Optional

import typer

from .agent.loop import AgentLoop
from .agent.memory import Memory
from .config import load_config
from .terminal.ui import ArcaApp

app = typer.Typer(
    name="arca",
    help="AI voice agentic terminal — speak naturally, execute intelligently",
    add_completion=False,
)


def _check_api_key(config) -> bool:
    if not config.anthropic_api_key:
        typer.echo(
            "Error: ANTHROPIC_API_KEY not set.\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Or add it to ~/.arca/config.toml",
            err=True,
        )
        return False
    return True


@app.command()
def main(
    no_voice: bool = typer.Option(False, "--no-voice", help="Disable voice input/output"),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="One-shot text query, then exit"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model (e.g. claude-opus-4-6)"),
    reset: bool = typer.Option(False, "--reset", help="Clear conversation history and start fresh"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """Arca — AI voice agentic terminal."""

    if version:
        from . import __version__
        typer.echo(f"arca {__version__}")
        return

    # ── Config ─────────────────────────────────────────────────────────────────
    config = load_config()

    if model:
        config.agent.model = model

    if no_voice:
        config.voice_enabled = False

    if not _check_api_key(config):
        raise typer.Exit(1)

    # ── Memory ─────────────────────────────────────────────────────────────────
    memory = Memory(
        db_path=config.memory.db_path,
        max_history=config.memory.session_history,
    )
    if reset:
        memory.start_session()
    else:
        memory.start_session()  # Always start a new session per run

    # ── Agent ──────────────────────────────────────────────────────────────────
    agent = AgentLoop(config=config, memory=memory)

    # ── One-shot mode ──────────────────────────────────────────────────────────
    if text:
        _run_one_shot(agent, text)
        return

    # ── TUI + Voice mode ───────────────────────────────────────────────────────
    _run_tui(config, agent)


def _run_one_shot(agent: AgentLoop, text: str) -> None:
    """Run a single query and print the result to stdout."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()

    def on_chunk(chunk: str):
        print(chunk, end="", flush=True)

    def on_tool_start(name: str, inputs: dict):
        console.print(f"\n[dim]→ {name}({', '.join(f'{k}={repr(v)[:30]}' for k, v in inputs.items())})[/]")

    result = agent.run(
        text,
        on_text_chunk=on_chunk,
        on_tool_start=on_tool_start,
    )
    print()  # newline after streaming


def _run_tui(config, agent: AgentLoop) -> None:
    """Launch the full interactive TUI."""

    # We'll wire voice pipeline if enabled
    tts = None
    voice_pipeline = None

    if config.voice_enabled:
        try:
            tts, voice_pipeline = _init_voice(config)
        except Exception as e:
            typer.echo(f"[voice] Failed to initialize: {e}", err=True)
            typer.echo("[voice] Continuing without voice. Use --no-voice to suppress this.", err=True)

    # Callback: agent runner (called from UI thread → runs in background thread)
    def agent_runner(user_text: str):
        """Run agent and push results back to TUI."""

        def on_chunk(chunk: str):
            ui_app.call_from_thread(
                lambda: ui_app.query_one("#output").write(chunk, end="")
            )

        def on_tool_start(name: str, inputs: dict):
            ui_app.call_from_thread(ui_app.print_tool, name, inputs, None)

        def on_tool_end(name: str, result: dict):
            ui_app.call_from_thread(ui_app.print_tool, name, {}, result)

        # Print "Arca  " prefix before streaming starts
        ui_app.call_from_thread(
            lambda: ui_app.query_one("#output").write("\n[bold cyan]Arca[/]  ", end="")
        )

        response = agent.run(
            user_text,
            on_text_chunk=on_chunk,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )

        # Speak response via TTS
        if tts and response:
            threading.Thread(target=tts.speak, args=(response,), daemon=True).start()

    # Voice trigger callback (called when user presses Ctrl+Space)
    def voice_trigger():
        if voice_pipeline is None:
            return

        def _listen():
            ui_app.call_from_thread(ui_app.set_recording_state, True)

            text = voice_pipeline.listen(
                on_recording_start=lambda: ui_app.call_from_thread(ui_app.set_recording_state, True),
                on_recording_end=lambda: ui_app.call_from_thread(ui_app.set_recording_state, False),
            )

            ui_app.call_from_thread(ui_app.set_recording_state, False)

            if text:
                # Show transcribed text in input, then submit
                ui_app.call_from_thread(ui_app.print_user, text)
                threading.Thread(target=agent_runner, args=(text,), daemon=True).start()

        threading.Thread(target=_listen, daemon=True).start()

    ui_app = ArcaApp(
        agent_runner=agent_runner,
        voice_trigger=voice_trigger if voice_pipeline else None,
    )
    ui_app.run()


def _init_voice(config):
    """Initialize STT + VAD + TTS. Returns (tts, pipeline)."""
    from .voice.pipeline import VoicePipeline
    from .voice.stt import STT
    from .voice.tts import TTS
    from .voice.vad import VAD

    stt = STT(
        model_size=config.voice.stt_model,
        device=config.voice.stt_device,
    )
    vad = VAD(
        threshold=config.voice.vad_threshold,
        silence_ms=config.voice.vad_silence_ms,
    )
    tts = TTS(
        engine=config.tts.engine,
        voice=config.tts.voice,
        speed=config.tts.speed,
        openai_voice=config.tts.openai_voice,
        openai_api_key=config.openai_api_key,
    )
    pipeline = VoicePipeline(stt=stt, tts=tts, vad=vad)
    return tts, pipeline


if __name__ == "__main__":
    app()
