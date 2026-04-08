"""
MCP Client — connects Arca to any MCP server and exposes their tools
to the Claude agent loop.

MCP (Model Context Protocol) is Anthropic's open standard for connecting
AI models to external tools and data sources. Claude Code, Cursor, and
GitHub Copilot all speak MCP. By implementing an MCP client, Arca gets
instant access to the entire MCP ecosystem:
  - GitHub (PRs, issues, code search)
  - Notion, Linear, Jira
  - Database clients
  - File systems
  - Custom internal tools

How MCP works:
  1. MCP servers are local processes (Node/Python/Go) that expose tools
  2. Client connects via stdio (most common) or SSE (remote servers)
  3. Client calls initialize() → gets server name + capabilities
  4. Client calls list_tools() → gets all available tool schemas
  5. Client calls call_tool(name, args) → gets result

Architecture in Arca:
  - MCPClient manages connections to all configured servers
  - On startup: spawns server processes, initializes sessions
  - Exposes list_tools() → merged list of all server tools (prefixed: "mcp__github__list_prs")
  - Exposes call_tool(name, args) → routes to correct server
  - Agent loop adds MCP tools to Claude's tool list
  - dispatch_tool() routes "mcp__*" calls here

Config example (arca.toml):
  [[mcp.servers]]
  name = "github"
  command = "npx"
  args = ["-y", "@modelcontextprotocol/server-github"]
  env = { GITHUB_PERSONAL_ACCESS_TOKEN = "$GITHUB_TOKEN" }

  [[mcp.servers]]
  name = "filesystem"
  command = "npx"
  args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/josh"]

The async MCP SDK is run in a dedicated thread with its own event loop.
Synchronous call() interface exposed for the agent loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def resolved_env(self) -> dict[str, str]:
        """Resolve $VAR references in env values."""
        resolved = {}
        for k, v in self.env.items():
            if v.startswith("$"):
                resolved[k] = os.environ.get(v[1:], "")
            else:
                resolved[k] = v
        return {**os.environ, **resolved}


class MCPClient:
    """
    Manages connections to all configured MCP servers.
    Exposes a unified synchronous tool interface to the agent loop.
    """

    def __init__(self, servers: list[MCPServerConfig]):
        self._servers = servers
        self._tools: dict[str, dict] = {}          # tool_key → schema
        self._tool_server: dict[str, str] = {}     # tool_key → server name
        self._sessions: dict[str, Any] = {}         # server name → session (in async loop)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        """
        Start the async event loop thread and connect to all MCP servers.
        Blocks until all connections are established (or timeout).
        """
        if not _MCP_AVAILABLE:
            print("[arca/mcp] mcp package not installed. MCP tools unavailable.")
            print("[arca/mcp] Install: pip install mcp")
            return

        if not self._servers:
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="arca-mcp",
        )
        self._thread.start()
        self._ready.wait(timeout=30)

    def stop(self) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def list_tools(self) -> list[dict]:
        """
        Return Claude-compatible tool schemas for all MCP tools.
        Tool names are prefixed: "mcp__{server_name}__{original_name}"
        """
        return list(self._tools.values())

    def call_tool(self, prefixed_name: str, arguments: dict) -> Any:
        """
        Call an MCP tool synchronously.
        prefixed_name format: "mcp__{server}__{tool}"
        """
        if not self._loop or prefixed_name not in self._tool_server:
            return {"error": f"MCP tool not available: {prefixed_name}"}

        server_name = self._tool_server[prefixed_name]
        # Extract original tool name (strip prefix)
        original_name = prefixed_name.replace(f"mcp__{server_name}__", "", 1)

        future = asyncio.run_coroutine_threadsafe(
            self._async_call_tool(server_name, original_name, arguments),
            self._loop,
        )
        try:
            result = future.result(timeout=30)
            return result
        except Exception as e:
            return {"error": str(e)}

    # ─── Async internals ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Entry point for the async thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_all())
        except Exception as e:
            print(f"[arca/mcp] Event loop error: {e}")
        finally:
            self._ready.set()

    async def _connect_all(self) -> None:
        """Connect to all configured MCP servers concurrently."""
        tasks = [self._connect_server(s) for s in self._servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for server, result in zip(self._servers, results):
            if isinstance(result, Exception):
                print(f"[arca/mcp] Failed to connect to '{server.name}': {result}")

        self._ready.set()

        # Keep the loop alive (sessions stay open)
        while True:
            await asyncio.sleep(60)

    async def _connect_server(self, server: MCPServerConfig) -> None:
        """Connect to one MCP server, list its tools, store the session."""
        params = StdioServerParameters(
            command=server.command,
            args=server.args,
            env=server.resolved_env(),
        )

        # Note: we use the context manager but keep the session alive
        # by running it in a background task
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()

                for tool in tools_response.tools:
                    key = f"mcp__{server.name}__{tool.name}"
                    self._tools[key] = {
                        "name": key,
                        "description": f"[{server.name}] {tool.description or tool.name}",
                        "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
                    }
                    self._tool_server[key] = server.name

                print(f"[arca/mcp] Connected to '{server.name}' — {len(tools_response.tools)} tools")

                # Keep session alive by storing reference and waiting
                self._sessions[server.name] = session
                # Wait forever (until loop stops)
                await asyncio.Event().wait()

    async def _async_call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """Async tool call — runs in the MCP event loop thread."""
        session = self._sessions.get(server_name)
        if session is None:
            return {"error": f"No active session for server: {server_name}"}

        try:
            result = await session.call_tool(tool_name, arguments)
            # MCP returns CallToolResult with content list
            if result.content:
                # Extract text content
                texts = [c.text for c in result.content if hasattr(c, "text")]
                return {"result": "\n".join(texts), "is_error": result.isError}
            return {"result": "", "is_error": result.isError}
        except Exception as e:
            return {"error": str(e)}


def build_mcp_client_from_config(config) -> MCPClient | None:
    """Build an MCPClient from arca config. Returns None if no servers configured."""
    mcp_cfg = getattr(config, "mcp", None)
    if mcp_cfg is None:
        return None

    servers_raw = getattr(mcp_cfg, "servers", [])
    if not servers_raw:
        return None

    servers = [
        MCPServerConfig(
            name=s.get("name", f"server_{i}"),
            command=s.get("command", ""),
            args=s.get("args", []),
            env=s.get("env", {}),
        )
        for i, s in enumerate(servers_raw)
        if s.get("command")
    ]

    return MCPClient(servers) if servers else None
