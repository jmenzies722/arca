"""
Arca — entry point.

Phase 2 boot sequence:
  1. Load config
  2. Detect project context (git, README, CLAUDE.md)
  3. Initialize memory (SQLite + chromadb)
  4. Start MCP client (connects to configured servers)
  5. Initialize voice pipeline (STT + VAD + TTS)
  6. Start wake word detector (if enabled)
  7. Start ambient monitor (if enabled)
  8. Initialize agent loop (Claude API + all tools)
  9. Launch Textual TUI
  10. Wire all callbacks
"""

from __future__ import annotations

import os
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
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model"),
    reset: bool = typer.Option(False, "--reset", help="Clear conversation history"),
    ambient: bool = typer.Option(False, "--ambient", help="Enable ambient monitoring mode"),
    no_context: bool = typer.Option(False, "--no-context", help="Skip project context detection"),
    serve: bool = typer.Option(False, "--serve", help="Run as WebSocket sidecar for Tauri app"),
    port: int = typer.Option(8765, "--port", "-p", help="WebSocket server port (with --serve)"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
) -> None:
    """Arca — AI voice agentic terminal."""

    if version:
        from . import __version__
        typer.echo(f"arca {__version__}")
        return

    config = load_config()

    if model:
        config.agent.model = model
    if no_voice:
        config.voice_enabled = False
    if ambient:
        config.ambient.enabled = True

    if not _check_api_key(config):
        raise typer.Exit(1)

    # ── Reset flag: clear history before starting ───────────────────────────────
    if reset:
        _db = Memory(db_path=config.memory.db_path, semantic_search=False)
        _db.clear_all_sessions()
        _db.close()
        typer.echo("[memory] Conversation history cleared.")

    # ── Project Context (Phase 2) ───────────────────────────────────────────────
    project_context = ""
    if config.context.enabled and not no_context:
        try:
            from .agent.context import detect_project_context
            project_context = detect_project_context()
            if project_context:
                typer.echo(f"[context] Project context loaded ({len(project_context)} chars)")
        except Exception as e:
            typer.echo(f"[context] Warning: {e}", err=True)

    # ── Memory ──────────────────────────────────────────────────────────────────
    memory = Memory(
        db_path=config.memory.db_path,
        max_history=config.memory.session_history,
        semantic_search=config.memory.semantic_search,
        semantic_db_path=config.memory.semantic_db_path,
        semantic_top_k=config.memory.semantic_top_k,
    )
    memory.start_session()

    # ── MCP Client (Phase 2) ────────────────────────────────────────────────────
    mcp_client = None
    if config.tools.mcp_passthrough and config.mcp.servers:
        try:
            from .agent.mcp_client import build_mcp_client_from_config
            mcp_client = build_mcp_client_from_config(config)
            if mcp_client:
                typer.echo(f"[mcp] Connecting to {len(config.mcp.servers)} server(s)...")
                mcp_client.start()  # Blocks until connected (30s timeout)
        except Exception as e:
            typer.echo(f"[mcp] Warning: {e}", err=True)

    # ── Ambient Monitor (Phase 2) ───────────────────────────────────────────────
    ambient_monitor = None
    if config.ambient.enabled:
        try:
            from .agent.ambient import AmbientMonitor

            def _on_ambient_suggestion(text: str):
                typer.echo(f"\n[Arca] {text}")  # Will be replaced by TUI callback

            ambient_monitor = AmbientMonitor(
                on_suggestion=_on_ambient_suggestion,
                anthropic_api_key=config.anthropic_api_key,
                model=config.ambient.model,
                min_interval_s=config.ambient.min_interval_s,
            )
            ambient_monitor.start()
            typer.echo("[ambient] Monitoring enabled")
        except Exception as e:
            typer.echo(f"[ambient] Warning: {e}", err=True)

    # ── Agent Loop ──────────────────────────────────────────────────────────────
    agent = AgentLoop(
        config=config,
        memory=memory,
        mcp_client=mcp_client,
        ambient_monitor=ambient_monitor,
        project_context=project_context,
    )

    # Rebuild tool list now that MCP is connected
    if mcp_client:
        agent.refresh_mcp_tools()

    # ── One-shot mode ───────────────────────────────────────────────────────────
    if text:
        _run_one_shot(agent, text)
        return

    # ── WebSocket sidecar mode (for Tauri app) ──────────────────────────────────
    if serve:
        _run_serve(config, agent, ambient_monitor, port)
        return

    # ── TUI + Voice mode ────────────────────────────────────────────────────────
    _run_tui(config, agent, ambient_monitor)


def _run_serve(config, agent: AgentLoop, ambient_monitor=None, port: int = 8765) -> None:
    """Run Arca as a WebSocket sidecar for the Tauri desktop app."""
    import signal
    from .server import ArcaServer

    tts = None
    voice_pipeline = None

    if config.voice_enabled:
        try:
            tts, voice_pipeline = _init_voice(config)
        except Exception as e:
            typer.echo(f"[voice] Failed to initialize: {e}", err=True)

    server = ArcaServer(
        agent=agent,
        tts=tts,
        voice_pipeline=voice_pipeline,
        port=port,
    )

    # Wire ambient monitor to push suggestions to the frontend
    if ambient_monitor:
        ambient_monitor.on_suggestion = server.emit_ambient

    server.start()
    typer.echo(f"[arca] Sidecar running on ws://127.0.0.1:{port}")
    typer.echo("[arca] Waiting for Tauri app to connect. Ctrl+C to stop.")

    try:
        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        pass  # AttributeError on Windows where signal.pause() doesn't exist


def _run_one_shot(agent: AgentLoop, text: str) -> None:
    """Run a single query and print to stdout."""
    from rich.console import Console
    console = Console()

    def on_chunk(chunk: str):
        print(chunk, end="", flush=True)

    def on_tool_start(name: str, inputs: dict):
        console.print(
            f"\n[dim]→ {name}({', '.join(f'{k}={repr(v)[:30]}' for k, v in inputs.items())})[/]"
        )

    agent.run(text, on_text_chunk=on_chunk, on_tool_start=on_tool_start)
    print()


def _run_tui(config, agent: AgentLoop, ambient_monitor=None) -> None:
    """Launch the full interactive TUI with all Phase 2 components wired."""

    tts = None
    voice_pipeline = None
    wakeword_detector = None

    if config.voice_enabled:
        try:
            tts, voice_pipeline = _init_voice(config)
        except Exception as e:
            typer.echo(f"[voice] Failed to initialize: {e}", err=True)

    def agent_runner(user_text: str):
        """Run agent and stream results to TUI."""

        def on_chunk(chunk: str):
            ui_app.call_from_thread(
                lambda: ui_app.query_one("#output").write(chunk, end="")
            )

        def on_tool_start(name: str, inputs: dict):
            ui_app.call_from_thread(ui_app.print_tool, name, inputs, None)

        def on_tool_end(name: str, result: dict):
            ui_app.call_from_thread(ui_app.print_tool, name, {}, result)

        ui_app.call_from_thread(
            lambda: ui_app.query_one("#output").write("\n[bold cyan]Arca[/]  ", end="")
        )

        response = agent.run(
            user_text,
            on_text_chunk=on_chunk,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )

        if tts and response:
            threading.Thread(target=tts.speak, args=(response,), daemon=True).start()

    def voice_trigger():
        if voice_pipeline is None:
            return

        def _listen():
            text = voice_pipeline.listen(
                on_recording_start=lambda: ui_app.call_from_thread(
                    ui_app.set_recording_state, True
                ),
                on_recording_end=lambda: ui_app.call_from_thread(
                    ui_app.set_recording_state, False
                ),
            )
            ui_app.call_from_thread(ui_app.set_recording_state, False)
            if text:
                ui_app.call_from_thread(ui_app.print_user, text)
                threading.Thread(target=agent_runner, args=(text,), daemon=True).start()

        threading.Thread(target=_listen, daemon=True).start()

    ui_app = ArcaApp(
        agent_runner=agent_runner,
        voice_trigger=voice_trigger if voice_pipeline else None,
        reset_callback=agent.reset,
    )

    # Phase 2: Wire ambient monitor callback to TUI
    if ambient_monitor:
        def on_ambient(suggestion: str):
            ui_app.call_from_thread(ui_app.print_ambient, suggestion)
            if tts and config.ambient.speak_suggestions:
                threading.Thread(target=tts.speak, args=(suggestion,), daemon=True).start()
        ambient_monitor.on_suggestion = on_ambient

    # Phase 2: Start wake word detector
    if config.voice.wake_word_enabled and voice_pipeline:
        try:
            from .voice.wakeword import WakeWordDetector
            wakeword_detector = WakeWordDetector(
                on_detected=voice_trigger,
                model=config.voice.wake_word_model,
                threshold=config.voice.wake_word_threshold,
                engine=config.voice.wake_word_engine,
                porcupine_key=config.voice.porcupine_key,
            )
            wakeword_detector.start()
            typer.echo(f"[wake word] Listening for '{config.voice.wake_word}'")
        except Exception as e:
            typer.echo(f"[wake word] Failed to start: {e}", err=True)

    try:
        ui_app.run()
    finally:
        if wakeword_detector:
            wakeword_detector.stop()
        if ambient_monitor:
            ambient_monitor.stop()


def _init_voice(config):
    """Initialize STT + VAD + TTS. Returns (tts, pipeline)."""
    from .voice.pipeline import VoicePipeline
    from .voice.stt import STT
    from .voice.tts import TTS
    from .voice.vad import VAD

    stt = STT(model_size=config.voice.stt_model, device=config.voice.stt_device)
    vad = VAD(threshold=config.voice.vad_threshold, silence_ms=config.voice.vad_silence_ms)
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
