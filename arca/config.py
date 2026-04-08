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
    wake_word_enabled: bool = False         # Phase 2: always-on wake word
    wake_word_engine: str = "openwakeword"  # openwakeword | porcupine
    wake_word_model: str = "hey_jarvis"     # Phonetically close to "hey arca"
    wake_word_threshold: float = 0.5
    porcupine_key: str = ""
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
    semantic_search: bool = True            # Phase 2: chromadb similarity search
    semantic_db_path: str = "~/.arca/vectors"
    semantic_top_k: int = 5                 # How many similar past messages to include


@dataclass
class ShellConfig:
    default_shell: str = "/bin/zsh"
    working_dir: str = "."
    timeout: int = 30


@dataclass
class ContextConfig:
    """Phase 2: Project context auto-detection."""
    enabled: bool = True
    read_claude_md: bool = True
    read_readme: bool = True
    read_git_info: bool = True
    max_context_chars: int = 6000           # Limit to avoid token bloat


@dataclass
class AmbientConfig:
    """Phase 2: Ambient monitoring mode."""
    enabled: bool = False
    model: str = "claude-haiku-4-5-20251001"  # Fast cheap model for suggestions
    min_interval_s: float = 30.0             # Minimum seconds between suggestions
    speak_suggestions: bool = True           # TTS for ambient suggestions


@dataclass
class MCPServerEntry:
    name: str = ""
    command: str = ""
    args: list = field(default_factory=list)
    env: dict = field(default_factory=dict)


@dataclass
class MCPConfig:
    """Phase 2: MCP server connections."""
    servers: list[dict] = field(default_factory=list)


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
    context: ContextConfig = field(default_factory=ContextConfig)
    ambient: AmbientConfig = field(default_factory=AmbientConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

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


def _apply_dataclass(cls, raw: dict):
    """Safely build a dataclass from a dict, ignoring unknown keys."""
    known = {k for k in cls.__dataclass_fields__}
    return cls(**{k: raw[k] for k in known if k in raw})


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

    cfg = ArcaConfig()

    arca_raw = raw.get("arca", {})
    cfg.persona = arca_raw.get("persona", cfg.persona)
    cfg.voice_enabled = arca_raw.get("voice_enabled", cfg.voice_enabled)
    cfg.ambient_mode = arca_raw.get("ambient_mode", cfg.ambient_mode)

    if v := raw.get("voice"):
        cfg.voice = _apply_dataclass(VoiceConfig, v)
    if t := raw.get("tts"):
        cfg.tts = _apply_dataclass(TTSConfig, t)
    if a := raw.get("agent"):
        cfg.agent = _apply_dataclass(AgentConfig, a)
    if tl := raw.get("tools"):
        cfg.tools = _apply_dataclass(ToolsConfig, tl)
    if m := raw.get("memory"):
        cfg.memory = _apply_dataclass(MemoryConfig, m)
    if s := raw.get("shell"):
        cfg.shell = _apply_dataclass(ShellConfig, s)
    if c := raw.get("context"):
        cfg.context = _apply_dataclass(ContextConfig, c)
    if amb := raw.get("ambient"):
        cfg.ambient = _apply_dataclass(AmbientConfig, amb)
    if mcp := raw.get("mcp"):
        cfg.mcp = MCPConfig(servers=mcp.get("servers", []))

    return cfg


def _find_project_config() -> Path | None:
    """Walk up from CWD looking for arca.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / "arca.toml"
        if candidate.exists():
            return candidate
        if (parent / ".git").exists():
            break
    return None
