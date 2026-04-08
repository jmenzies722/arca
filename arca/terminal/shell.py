"""
PTY Shell Runner — runs a real interactive shell underneath Arca.

The user can drop to raw shell at any time. The AI layer sits on top;
the shell is always live underneath.

Two modes:
  1. Command capture: run a command, capture stdout/stderr (for tool use)
  2. Interactive PTY: pass stdin/stdout directly to shell (for the user's raw shell)

Phase 1 uses subprocess for tool commands (simpler, reliable).
Phase 2 will add a full PTY widget so the user sees a live interactive shell.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator


class Shell:
    """Manages a shell subprocess for interactive use and command capture."""

    def __init__(self, shell_path: str = "/bin/zsh", cwd: str | None = None):
        self.shell_path = shell_path
        self.cwd = cwd or os.getcwd()
        self._env = {**os.environ}

    def run(self, command: str, timeout: int = 30) -> tuple[str, str, int]:
        """
        Run a command and capture its output.
        Returns (stdout, stderr, exit_code).

        This is used by the agent's shell_exec tool.
        Streams output line by line internally (shows live progress in logs).
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable=self.shell_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
                env=self._env,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout}s", -1
        except Exception as e:
            return "", str(e), -1

    def run_streaming(self, command: str, timeout: int = 30) -> Iterator[str]:
        """
        Run a command and yield output lines as they arrive.
        Use this for long-running commands (builds, tests) where you want live feedback.
        """
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                executable=self.shell_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.cwd,
                env=self._env,
            )

            for line in proc.stdout:
                yield line

            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            yield f"\n[timed out after {timeout}s]\n"
        except Exception as e:
            yield f"\n[error: {e}]\n"

    def set_cwd(self, path: str) -> bool:
        """Change the shell's working directory."""
        p = Path(path).expanduser().resolve()
        if p.is_dir():
            self.cwd = str(p)
            os.chdir(self.cwd)
            return True
        return False
