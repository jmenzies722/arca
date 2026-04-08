"""
Claude Agentic Loop — the brain of Arca.

Phase 2 additions:
  - Project context injected into system prompt at startup
  - Semantic memory search: relevant past exchanges prepended to history
  - MCP tool passthrough: Claude can call any connected MCP server tool
  - Ambient monitor: shell results fed to background error detector

The core "think → act → observe → repeat" cycle is unchanged:
  1. User sends a message (text or transcribed voice)
  2. Claude processes it with context + history + semantic memory
  3. Claude may call tools (shell, file, web, MCP) — we execute them
  4. Tool results go back to Claude as "tool" role messages
  5. Claude continues until it produces a final text response
"""

from __future__ import annotations

import json
import os
from typing import Callable

import anthropic

from .memory import Memory
from .tools import TOOL_SCHEMAS, dispatch_tool

BASE_SYSTEM_PROMPT = """You are Arca, an AI terminal companion. You run locally on the user's machine with full tool access.

Your role: Help the user get development work done through natural voice and text commands. Think step by step, use tools liberally, and explain what you're doing concisely.

Personality: Capable, direct, friendly. Like a senior engineer pair-programming with you. Not overly formal, not overly casual.

Rules:
- When you use a tool, briefly say what you're doing before calling it
- After tool results, summarize what happened in plain English
- If a command fails, diagnose why and suggest a fix
- Keep responses concise — the user will hear them via TTS
- For voice responses: write naturally spoken sentences, not markdown lists
- For terminal display: you can use markdown for code blocks and formatting

Context: The user's current directory is provided in each message. Read it to understand what project you're working in."""


def _build_system(config, project_context: str = "", extra: str = "") -> str:
    persona = getattr(config, "persona", "Arca") if config else "Arca"
    system = BASE_SYSTEM_PROMPT.replace("Arca", persona, 1)
    if project_context:
        system += f"\n\n{project_context}"
    if extra:
        system += f"\n\n{extra}"
    return system


class AgentLoop:
    """
    The main Claude agentic loop.
    Phase 2: project context + semantic memory + MCP tools + ambient feeding.
    """

    def __init__(
        self,
        config=None,
        memory: Memory | None = None,
        mcp_client=None,
        ambient_monitor=None,
        project_context: str = "",
    ):
        self.config = config
        self.memory = memory or Memory()
        self.mcp_client = mcp_client
        self.ambient_monitor = ambient_monitor

        api_key = (
            getattr(config, "anthropic_api_key", None)
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._client = anthropic.Anthropic(api_key=api_key)

        agent_cfg = getattr(config, "agent", None)
        self._model = getattr(agent_cfg, "model", "claude-sonnet-4-6")
        self._max_tokens = getattr(agent_cfg, "max_tokens", 8096)
        extra_prompt = getattr(agent_cfg, "system_prompt_extra", "")

        self._system = _build_system(config, project_context, extra_prompt)
        self._tools = self._build_tool_list()

    def _build_tool_list(self) -> list[dict]:
        """Merge built-in tools + MCP tools, filtered by config."""
        tools_cfg = getattr(self.config, "tools", None)

        enabled = []
        for schema in TOOL_SCHEMAS:
            name = schema["name"]
            if name == "shell_exec" and not getattr(tools_cfg, "shell", True):
                continue
            if name in ("file_read", "file_write", "file_edit", "list_dir") \
                    and not getattr(tools_cfg, "file_ops", True):
                continue
            if name == "web_search" and not getattr(tools_cfg, "web_search", True):
                continue
            enabled.append(schema)

        # Add MCP tools if client is connected and passthrough is enabled
        if self.mcp_client and getattr(tools_cfg, "mcp_passthrough", True):
            mcp_tools = self.mcp_client.list_tools()
            enabled.extend(mcp_tools)

        return enabled

    def refresh_mcp_tools(self) -> None:
        """Rebuild tool list after MCP servers connect (call after mcp_client.start())."""
        self._tools = self._build_tool_list()

    def run(
        self,
        user_input: str,
        on_text_chunk: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end: Callable[[str, dict], None] | None = None,
    ) -> str:
        """
        Process one user turn through the full agentic loop.

        Phase 2: semantic context is prepended before the conversation history.
        """
        augmented_input = f"[CWD: {os.getcwd()}]\n\n{user_input}"

        # Phase 2: Semantic search for relevant past context
        semantic_context = self.memory.semantic_search(user_input)

        self.memory.add_message("user", augmented_input)
        history = self.memory.load_history()

        # Prepend semantic context (if any) before current history
        # These are "ghost" messages — help Claude without cluttering main history
        messages = semantic_context + history if semantic_context else history

        final_text = ""

        while True:
            response_text, tool_calls = self._call_claude(
                messages=messages,
                on_text_chunk=on_text_chunk,
            )

            if response_text:
                final_text = response_text

            if not tool_calls:
                self.memory.add_message("assistant", response_text)
                break

            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                tool_use_id = tool_call["id"]

                if on_tool_start:
                    on_tool_start(tool_name, tool_input)

                # Route: MCP tool or built-in
                if tool_name.startswith("mcp__") and self.mcp_client:
                    result = self.mcp_client.call_tool(tool_name, tool_input)
                else:
                    result = dispatch_tool(tool_name, tool_input, self.config)

                # Feed shell results to ambient monitor
                if tool_name == "shell_exec" and self.ambient_monitor:
                    self.ambient_monitor.feed(
                        command=tool_input.get("command", ""),
                        stdout=result.get("stdout", ""),
                        stderr=result.get("stderr", ""),
                        exit_code=result.get("exit_code", 0),
                    )

                if on_tool_end:
                    on_tool_end(tool_name, result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result),
                })

            assistant_content = []
            if response_text:
                assistant_content.append({"type": "text", "text": response_text})
            for tc in tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })

            self.memory.add_message("assistant", assistant_content)
            self.memory.add_message("user", tool_results)

            messages = self.memory.load_history()

        return final_text

    def _call_claude(
        self,
        messages: list[dict],
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict]]:
        """Single streaming Claude API call. Returns (text, tool_calls)."""
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        current_tool: dict | None = None
        tool_input_json: str = ""

        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=self._tools,
            messages=messages,
        ) as stream:
            for event in stream:
                event_type = type(event).__name__

                if event_type == "RawContentBlockStartEvent":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {"id": block.id, "name": block.name, "input": {}}
                        tool_input_json = ""

                elif event_type == "RawContentBlockDeltaEvent":
                    delta = event.delta
                    if delta.type == "text_delta":
                        text_parts.append(delta.text)
                        if on_text_chunk:
                            on_text_chunk(delta.text)
                    elif delta.type == "input_json_delta":
                        tool_input_json += delta.partial_json

                elif event_type == "RawContentBlockStopEvent":
                    if current_tool is not None:
                        try:
                            current_tool["input"] = json.loads(tool_input_json) if tool_input_json else {}
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_calls.append(current_tool)
                        current_tool = None
                        tool_input_json = ""

        return "".join(text_parts), tool_calls

    def reset(self) -> None:
        """Clear conversation history and start fresh."""
        self.memory.clear_session()
        self.memory.start_session()
