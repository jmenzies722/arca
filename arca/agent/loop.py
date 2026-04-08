"""
Claude Agentic Loop — the brain of Arca.

This is the core "think → act → observe → repeat" cycle:

  1. User sends a message (text or transcribed voice)
  2. Claude processes it with full conversation history
  3. Claude may call tools (shell, file, web) — we execute them
  4. Tool results go back to Claude as "tool" role messages
  5. Claude continues until it produces a final text response
  6. Final response returned to the UI and TTS

This loop is identical to how Claude Code works internally.
Claude is given the tool schemas and autonomously decides
which tools to call, in what order, with what parameters.

Key design decisions:
  - Streaming: We stream Claude's text tokens for low-latency display
  - Tool approval: Phase 1 auto-approves all tools. Phase 2 adds confirmation UI.
  - System prompt: Informs Claude of its persona, the user's context, and rules
"""

from __future__ import annotations

import json
import os
from typing import Callable, Generator, Iterator

import anthropic

from .memory import Memory
from .tools import TOOL_SCHEMAS, dispatch_tool

# ─── System Prompt ─────────────────────────────────────────────────────────────

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


def _build_system(config, extra: str = "") -> str:
    persona = getattr(config, "persona", "Arca") if config else "Arca"
    system = BASE_SYSTEM_PROMPT.replace("Arca", persona, 1)
    if extra:
        system += f"\n\n{extra}"
    return system


# ─── Agent Loop ────────────────────────────────────────────────────────────────

class AgentLoop:
    """
    The main Claude agentic loop.
    Handles multi-turn conversation with tool use.
    """

    def __init__(self, config=None, memory: Memory | None = None):
        self.config = config
        self.memory = memory or Memory()

        api_key = (
            getattr(config, "anthropic_api_key", None)
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._client = anthropic.Anthropic(api_key=api_key)

        model = getattr(getattr(config, "agent", None), "model", "claude-sonnet-4-6")
        max_tokens = getattr(getattr(config, "agent", None), "max_tokens", 8096)
        extra_prompt = getattr(getattr(config, "agent", None), "system_prompt_extra", "")

        self._model = model
        self._max_tokens = max_tokens
        self._system = _build_system(config, extra_prompt)

        # Active tool schemas (filtered by config)
        self._tools = self._enabled_tools()

    def _enabled_tools(self) -> list[dict]:
        tools_cfg = getattr(self.config, "tools", None)
        if tools_cfg is None:
            return TOOL_SCHEMAS

        enabled = []
        for schema in TOOL_SCHEMAS:
            name = schema["name"]
            if name in ("shell_exec",) and not getattr(tools_cfg, "shell", True):
                continue
            if name in ("file_read", "file_write", "file_edit", "list_dir") and not getattr(
                tools_cfg, "file_ops", True
            ):
                continue
            if name in ("web_search",) and not getattr(tools_cfg, "web_search", True):
                continue
            enabled.append(schema)
        return enabled

    def run(
        self,
        user_input: str,
        on_text_chunk: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end: Callable[[str, dict], None] | None = None,
    ) -> str:
        """
        Process one user turn through the full agentic loop.

        Args:
            user_input: The user's message (text or transcribed voice).
            on_text_chunk: Called with each streamed text token for live display.
            on_tool_start: Called when Claude invokes a tool (name, inputs).
            on_tool_end: Called when a tool finishes (name, result).

        Returns:
            The final assistant text response.
        """
        # Add context: current directory
        augmented_input = f"[CWD: {os.getcwd()}]\n\n{user_input}"

        # Add to memory and get history
        self.memory.add_message("user", augmented_input)
        messages = self.memory.load_history()

        final_text = ""

        # Agentic loop — keeps going until Claude stops requesting tools
        while True:
            response_text, tool_calls = self._call_claude(
                messages=messages,
                on_text_chunk=on_text_chunk,
            )

            if response_text:
                final_text = response_text

            if not tool_calls:
                # No more tools — Claude is done
                self.memory.add_message("assistant", response_text)
                break

            # Execute tool calls, collect results
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                tool_use_id = tool_call["id"]

                if on_tool_start:
                    on_tool_start(tool_name, tool_input)

                result = dispatch_tool(tool_name, tool_input, self.config)

                if on_tool_end:
                    on_tool_end(tool_name, result)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result),
                })

            # Add Claude's response (with tool_use blocks) to history
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

            # Refresh message history for next iteration
            messages = self.memory.load_history()

        return final_text

    def _call_claude(
        self,
        messages: list[dict],
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Make one streaming API call to Claude.

        Returns:
            (text_response, list_of_tool_calls)
            tool_calls is empty if Claude is done.
        """
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
                        current_tool = {
                            "id": block.id,
                            "name": block.name,
                            "input": {},
                        }
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
