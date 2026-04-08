"""
Arca WebSocket server — bridges the Tauri frontend to the Python agent backend.

The Tauri app connects to ws://127.0.0.1:8765 and sends/receives JSON messages.

Inbound (frontend → server):
  { "type": "text",        "content": "..." }
  { "type": "voice_start" }
  { "type": "voice_stop"  }
  { "type": "reset"       }

Outbound (server → frontend):
  { "type": "user_message",  "text": "..." }
  { "type": "assistant_start" }
  { "type": "text_chunk",    "text": "..." }
  { "type": "tool_start",    "name": "...", "inputs": {...} }
  { "type": "tool_end",      "name": "...", "result": {...} }
  { "type": "status",        "state": "idle" | "recording" | "thinking" }
  { "type": "ambient",       "text": "..." }
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import TYPE_CHECKING

try:
    import websockets
    from websockets.asyncio.server import serve, ServerConnection
except ImportError:
    raise RuntimeError("websockets not installed. Run: pip install websockets")

if TYPE_CHECKING:
    from .agent.loop import AgentLoop
    from .agent.memory import Memory
    from .voice.pipeline import VoicePipeline
    from .voice.tts import TTS

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class ArcaServer:
    """
    WebSocket server that wires the Tauri frontend to the Arca agent backend.
    One connection at a time (the desktop app). Additional connections are rejected.
    """

    def __init__(
        self,
        agent: "AgentLoop",
        tts: "TTS | None" = None,
        voice_pipeline: "VoicePipeline | None" = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        self.agent = agent
        self.tts = tts
        self.voice_pipeline = voice_pipeline
        self.host = host
        self.port = port

        self._conn: ServerConnection | None = None
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the server in a background thread. Returns immediately."""
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with serve(self._handler, self.host, self.port):
            print(f"[arca/server] Listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    # ── Connection handler ─────────────────────────────────────────────────────

    async def _handler(self, ws: ServerConnection) -> None:
        async with self._lock:
            if self._conn is not None:
                await ws.close(1008, "Already connected")
                return
            self._conn = ws

        try:
            await self._emit({"type": "status", "state": "idle"})
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(msg)
        finally:
            self._conn = None

    # ── Message dispatch ───────────────────────────────────────────────────────

    async def _dispatch(self, msg: dict) -> None:
        t = msg.get("type")
        if t == "text":
            await self._handle_text(msg.get("content", ""))
        elif t == "voice_start":
            await self._handle_voice_start()
        elif t == "voice_stop":
            await self._handle_voice_stop()
        elif t == "reset":
            self.agent.reset()
            await self._emit({"type": "status", "state": "idle"})

    async def _handle_text(self, text: str) -> None:
        if not text.strip():
            return

        await self._emit({"type": "user_message", "text": text})
        await self._emit({"type": "status", "state": "thinking"})
        await self._emit({"type": "assistant_start"})

        loop = asyncio.get_event_loop()

        def on_chunk(chunk: str):
            asyncio.run_coroutine_threadsafe(
                self._emit({"type": "text_chunk", "text": chunk}), loop
            )

        def on_tool_start(name: str, inputs: dict):
            asyncio.run_coroutine_threadsafe(
                self._emit({"type": "tool_start", "name": name, "inputs": inputs}), loop
            )

        def on_tool_end(name: str, result: dict):
            asyncio.run_coroutine_threadsafe(
                self._emit({"type": "tool_end", "name": name, "result": result}), loop
            )

        response = await loop.run_in_executor(
            None,
            lambda: self.agent.run(
                text,
                on_text_chunk=on_chunk,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            ),
        )

        await self._emit({"type": "status", "state": "idle"})

        if self.tts and response:
            threading.Thread(target=self.tts.speak, args=(response,), daemon=True).start()

    async def _handle_voice_start(self) -> None:
        if self.voice_pipeline is None:
            await self._emit({"type": "status", "state": "idle"})
            return

        await self._emit({"type": "status", "state": "recording"})
        loop = asyncio.get_event_loop()

        def _listen():
            text = self.voice_pipeline.listen()
            asyncio.run_coroutine_threadsafe(self._on_voice_done(text), loop)

        threading.Thread(target=_listen, daemon=True).start()

    async def _handle_voice_stop(self) -> None:
        # Voice stop is handled by VAD silence detection — nothing to do explicitly
        pass

    async def _on_voice_done(self, text: str | None) -> None:
        await self._emit({"type": "status", "state": "idle"})
        if text:
            await self._handle_text(text)

    # ── Emit ───────────────────────────────────────────────────────────────────

    async def _emit(self, payload: dict) -> None:
        if self._conn is None:
            return
        try:
            await self._conn.send(json.dumps(payload))
        except Exception:
            pass

    def emit_ambient(self, text: str) -> None:
        """Thread-safe: push an ambient suggestion to the frontend."""
        if self._loop and self._conn:
            asyncio.run_coroutine_threadsafe(
                self._emit({"type": "ambient", "text": text}),
                self._loop,
            )
