"""
Config loader — reads arca.toml from project root, ~/.arca/config.toml, or defaults.
Priority: project root > ~/.arca/config.toml > defaults
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VoiceConfig:
    stt_model: str = "base.en"
    stt_device: str = "auto"
    push_to_talk_key: str = "ctrl+space"
    wake_word: str = "hey arca"
    vad_threshold: float = 0.5
    vad_silence_ms: int = 800


@dataclass
class TTSConfig:
    enabled: bool = True
    engine: str = "kokoro"      # kokoro | openai | none
    voice: str = "af_heart"
    speed: float = 1.1
    openai_voice: str = "nova"


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 8096
    temperature: float = 1.0
    system_prompt_extra: str = ""


@dataclass
class ToolsConfig:
    shell: bool = True
    file_ops: bool = True
    web_search: bool = True
    mcp_passthrough: bool = True


@dataclass
class MemoryConfig:
    session_history: int = 50
    db_path: str = "~/.arca/memory.db"


@dataclass
class ShellConfig:
    default_shell: str = "/bin/zsh"
    working_dir: str = "."
    timeout: int = 30


@dataclass
class ArcaConfig:
    persona: str = "Arca"
    voice_enabled: bool = True
    ambient_mode: bool = False

    voice: VoiceConfig = field(default_factory=VoiceConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    shell: ShellConfig = field(default_factory=ShellConfig)

    # Runtime — loaded from environment
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    brave_api_key: str = field(default_factory=lambda: os.environ.get("BRAVE_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))


def _merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> ArcaConfig:
    """Load and merge config from all sources."""
    raw: dict = {}

    # 1. User global config
    user_cfg = Path.home() / ".arca" / "config.toml"
    if user_cfg.exists():
        with open(user_cfg, "rb") as f:
            raw = _merge(raw, tomllib.load(f))

    # 2. Project-level config (current dir or parents)
    project_cfg = _find_project_config()
    if project_cfg:
        with open(project_cfg, "rb") as f:
            raw = _merge(raw, tomllib.load(f))

    # 3. Build config object from merged dict
    cfg = ArcaConfig()

    arca_raw = raw.get("arca", {})
    cfg.persona = arca_raw.get("persona", cfg.persona)
    cfg.voice_enabled = arca_raw.get("voice_enabled", cfg.voice_enabled)
    cfg.ambient_mode = arca_raw.get("ambient_mode", cfg.ambient_mode)

    if v := raw.get("voice"):
        cfg.voice = VoiceConfig(**{k: v[k] for k in VoiceConfig.__dataclass_fields__ if k in v})

    if t := raw.get("tts"):
        cfg.tts = TTSConfig(**{k: t[k] for k in TTSConfig.__dataclass_fields__ if k in t})

    if a := raw.get("agent"):
        cfg.agent = AgentConfig(**{k: a[k] for k in AgentConfig.__dataclass_fields__ if k in a})

    if tl := raw.get("tools"):
        cfg.tools = ToolsConfig(**{k: tl[k] for k in ToolsConfig.__dataclass_fields__ if k in tl})

    if m := raw.get("memory"):
        cfg.memory = MemoryConfig(**{k: m[k] for k in MemoryConfig.__dataclass_fields__ if k in m})

    if s := raw.get("shell"):
        cfg.shell = ShellConfig(**{k: s[k] for k in ShellConfig.__dataclass_fields__ if k in s})

    return cfg


def _find_project_config() -> Path | None:
    """Walk up from CWD looking for arca.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / "arca.toml"
        if candidate.exists():
            return candidate
        if (parent / ".git").exists():
            break  # Don't cross git root
    return None
