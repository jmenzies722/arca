"""
Tool definitions for Claude's tool_use API.

Each tool has:
  1. A schema (passed to Claude so it knows what tools exist and their signatures)
  2. A handler (Python function that actually executes the tool call)

Tools available in Phase 1:
  - shell_exec: Run any shell command, return stdout/stderr/exit_code
  - file_read:  Read a file's contents
  - file_write: Write or overwrite a file
  - file_edit:  Replace a specific string in a file
  - web_search: Search the web via Brave or Tavily API
  - list_dir:   List contents of a directory

How tool_use works in Claude's API:
  1. Claude's response has stop_reason="tool_use"
  2. Response content contains ToolUseBlock(s) with tool name + input
  3. We execute the tool locally → get result
  4. We send a "tool" role message back with the result
  5. Claude continues generating (may call more tools or finish)
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


# ─── Tool Schemas (sent to Claude) ────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "shell_exec",
        "description": (
            "Execute a shell command in the user's terminal. "
            "Returns stdout, stderr, and exit code. "
            "Use for: running scripts, git commands, npm/pip, file operations, "
            "checking system state, compiling code, running tests. "
            "Working directory is the user's current project directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute. Use full paths when needed.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max seconds to wait. Default 30. Use higher for builds/tests.",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "file_read",
        "description": "Read the contents of a file. Returns the file text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative file path.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional: first line to read (1-indexed).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional: last line to read (1-indexed, inclusive).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "file_write",
        "description": (
            "Write content to a file. Creates the file if it doesn't exist. "
            "Overwrites existing content. Use file_edit for targeted changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "file_edit",
        "description": (
            "Replace a specific string in a file with new content. "
            "The old_string must be unique in the file. "
            "Use for targeted edits without rewriting the whole file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit."},
                "old_string": {"type": "string", "description": "Exact text to find and replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_dir",
        "description": "List files and directories at a given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to current directory.",
                    "default": ".",
                },
            },
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current information. "
            "Use for: documentation, package info, error messages, recent news, "
            "anything not in training data or that might have changed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
]


# ─── Tool Handlers (execute the actual operations) ─────────────────────────────

def handle_shell_exec(command: str, timeout: int = 30, cwd: str | None = None) -> dict:
    """Run a shell command. Returns stdout, stderr, exit_code."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or os.getcwd(),
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "exit_code": -1,
            "command": command,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "command": command,
        }


def handle_file_read(path: str, start_line: int | None = None, end_line: int | None = None) -> dict:
    """Read a file, optionally a line range."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return {"error": f"File not found: {path}"}

        lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

        if start_line is not None or end_line is not None:
            s = (start_line or 1) - 1
            e = end_line or len(lines)
            lines = lines[s:e]

        content = "".join(lines)
        return {
            "path": str(p),
            "content": content,
            "lines": len(lines),
        }
    except Exception as ex:
        return {"error": str(ex)}


def handle_file_write(path: str, content: str) -> dict:
    """Write content to file. Creates directories if needed."""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p), "bytes_written": len(content.encode())}
    except Exception as ex:
        return {"error": str(ex)}


def handle_file_edit(path: str, old_string: str, new_string: str) -> dict:
    """Replace old_string with new_string in a file."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return {"error": f"File not found: {path}"}

        content = p.read_text(encoding="utf-8")
        count = content.count(old_string)

        if count == 0:
            return {"error": f"old_string not found in {path}"}
        if count > 1:
            return {"error": f"old_string is ambiguous — found {count} occurrences. Make it more specific."}

        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return {"path": str(p), "replaced": 1}
    except Exception as ex:
        return {"error": str(ex)}


def handle_list_dir(path: str = ".") -> dict:
    """List directory contents."""
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return {"error": f"Path not found: {path}"}
        if not p.is_dir():
            return {"error": f"Not a directory: {path}"}

        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        items = []
        for entry in entries:
            items.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size": entry.stat().st_size if entry.is_file() else None,
            })
        return {"path": str(p), "entries": items}
    except Exception as ex:
        return {"error": str(ex)}


def handle_web_search(query: str, num_results: int = 5, brave_api_key: str = "") -> dict:
    """Search the web via Brave Search API."""
    import httpx

    key = brave_api_key or os.environ.get("BRAVE_API_KEY", "")
    if not key:
        return {"error": "BRAVE_API_KEY not set. Add to environment or arca.toml."}

    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": key},
            params={"q": query, "count": num_results},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "description": r.get("description"),
            }
            for r in data.get("web", {}).get("results", [])
        ]
        return {"query": query, "results": results}
    except Exception as ex:
        return {"error": str(ex)}


# ─── Dispatch ──────────────────────────────────────────────────────────────────

def dispatch_tool(name: str, inputs: dict[str, Any], config=None) -> Any:
    """Route a tool call from Claude to the correct handler."""
    handlers = {
        "shell_exec": lambda: handle_shell_exec(
            inputs["command"],
            inputs.get("timeout", 30),
            cwd=getattr(config, "shell", None) and config.shell.working_dir,
        ),
        "file_read": lambda: handle_file_read(
            inputs["path"],
            inputs.get("start_line"),
            inputs.get("end_line"),
        ),
        "file_write": lambda: handle_file_write(inputs["path"], inputs["content"]),
        "file_edit": lambda: handle_file_edit(
            inputs["path"], inputs["old_string"], inputs["new_string"]
        ),
        "list_dir": lambda: handle_list_dir(inputs.get("path", ".")),
        "web_search": lambda: handle_web_search(
            inputs["query"],
            inputs.get("num_results", 5),
            brave_api_key=getattr(config, "brave_api_key", ""),
        ),
    }

    handler = handlers.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}

    return handler()
