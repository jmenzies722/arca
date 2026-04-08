"""
Project Context Detector — auto-discovers what project the user is in
and injects that context into Claude's system prompt.

Why this matters:
  Without context, Claude has to re-discover the project every conversation.
  With context, Claude immediately knows the stack, structure, and purpose —
  responses are smarter from the first message.

What gets detected:
  - Git repo root, branch, recent commits
  - Project type (Python, Node, Go, Rust, etc.)
  - Key files: CLAUDE.md, README.md, package.json, pyproject.toml, go.mod, etc.
  - Top-level directory structure
  - Active branch + uncommitted changes summary

Context is injected as a section of Claude's system prompt:
  "Project Context: [context string]"

This is rebuilt each time arca launches (or on Ctrl+R reset).
It's intentionally cheap — no embeddings, just file reads.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


# Files to read if they exist (truncated to avoid token bloat)
CONTEXT_FILES = [
    ("CLAUDE.md", 3000),       # Full AI instructions for this project
    ("README.md", 1500),       # Project overview
    (".cursorrules", 1000),    # Cursor AI rules (often has project context)
    ("CONTRIBUTING.md", 500),  # Dev workflow hints
]

# Files to read key metadata from (don't need full content)
META_FILES = [
    "pyproject.toml",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    "Gemfile",
    "composer.json",
]


def detect_project_context(cwd: str | None = None) -> str:
    """
    Detect project context from the current directory.

    Returns a context string ready to inject into Claude's system prompt.
    Returns empty string if no meaningful context found.
    """
    root = _find_git_root(cwd)
    lines: list[str] = []

    # Git info
    git_info = _get_git_info(root)
    if git_info:
        lines.append(f"## Project: {root.name}")
        lines.append(git_info)

    # Project type detection
    project_type = _detect_type(root)
    if project_type:
        lines.append(f"Stack: {project_type}")

    # Directory structure (top-level, excluding common noise)
    structure = _get_structure(root)
    if structure:
        lines.append(f"Structure:\n{structure}")

    # Key files
    for filename, max_chars in CONTEXT_FILES:
        path = root / filename
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... (truncated)"
            lines.append(f"\n### {filename}\n{content}")

    if not lines:
        return ""

    return "## Project Context (auto-detected)\n\n" + "\n".join(lines)


def _find_git_root(cwd: str | None = None) -> Path:
    """Walk up from cwd to find git root. Falls back to cwd."""
    start = Path(cwd or ".").resolve()
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return start


def _get_git_info(root: Path) -> str:
    """Get branch, recent commits, dirty status."""
    parts: list[str] = []

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        parts.append(f"Branch: {branch}")
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if status:
            changed = len(status.splitlines())
            parts.append(f"Uncommitted changes: {changed} file(s)")
    except Exception:
        pass

    try:
        log = subprocess.check_output(
            ["git", "log", "--oneline", "-5"],
            cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        if log:
            parts.append(f"Recent commits:\n{log}")
    except Exception:
        pass

    return "\n".join(parts)


def _detect_type(root: Path) -> str:
    """Detect project tech stack from manifest files."""
    indicators: list[str] = []

    checks = [
        ("pyproject.toml", "Python"),
        ("requirements.txt", "Python"),
        ("setup.py", "Python"),
        ("package.json", "Node.js"),
        ("go.mod", "Go"),
        ("Cargo.toml", "Rust"),
        ("pom.xml", "Java/Maven"),
        ("build.gradle", "Java/Gradle"),
        ("Gemfile", "Ruby"),
        ("composer.json", "PHP"),
        ("*.tf", "Terraform"),
        ("docker-compose.yml", "Docker Compose"),
        ("Dockerfile", "Docker"),
    ]

    for filename, label in checks:
        if "*" in filename:
            if list(root.glob(filename)):
                indicators.append(label)
        elif (root / filename).exists():
            indicators.append(label)

    return " + ".join(dict.fromkeys(indicators))  # Deduplicated, order-preserved


def _get_structure(root: Path, max_entries: int = 30) -> str:
    """Get top-level directory structure (skipping noise)."""
    skip = {
        ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
        "dist", "build", ".next", ".nuxt", "target", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", "coverage", ".DS_Store",
    }

    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = []
        for entry in entries[:max_entries]:
            if entry.name in skip or entry.name.startswith("."):
                continue
            icon = "/" if entry.is_dir() else ""
            lines.append(f"  {entry.name}{icon}")
        return "\n".join(lines)
    except Exception:
        return ""
