# Arca

**AI voice agentic terminal.** Speak naturally. Execute intelligently.

Arca is a locally-running terminal where voice and keyboard are first-class peers. Say what you want done — it reasons, runs tools, and responds. Like Claude Code and Warp had a baby, with a voice.

```
You:   "refactor the auth module to use JWT tokens"

Arca:  → reads src/auth.py
       → plans changes across 3 files
       → "Ready to edit auth.py, middleware.py, and tests. Approve?"
       → executes → "Done. Tests passing."
```

---

## Why Arca

| Tool | Gap |
|------|-----|
| Warp | Voice is a feature, not the core. Cloud-dependent. |
| Claude Code | No voice. No persistent companion. |
| Open Interpreter | No voice. Limited terminal UX. |
| Talon Voice | Voice without AI reasoning. |

Arca is the only tool that combines: real terminal + voice input + agentic AI reasoning + local-first execution.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/jmenzies722/arca
cd arca

# 2. Install + download models
bash scripts/setup.sh

# 3. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Launch
arca
```

### Usage

```bash
arca                    # Full TUI with voice
arca --no-voice         # Text-only mode
arca --text "query"     # One-shot query, then exit
arca --model claude-opus-4-6  # Override model
arca --reset            # Fresh conversation
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Space` | Start/stop voice recording |
| `Enter` | Submit text input |
| `Ctrl+R` | Reset conversation |
| `Ctrl+L` | Clear display |
| `Ctrl+C` | Quit |

---

## How It Works

Arca has 5 layers. See [PHASES.md](PHASES.md) for deep technical detail.

```
Voice Input  →  STT (Whisper)  →  Claude Agent Loop  →  Tools  →  TTS + Display
   mic           local, fast       tool use API         shell       Kokoro, local
```

### Voice Pipeline
- **STT**: `faster-whisper` with Metal GPU acceleration (Apple Silicon) — 300-500ms
- **VAD**: Silero VAD — detects when you start/stop speaking
- **TTS**: Kokoro-82M ONNX — #1 TTS model, runs fully local, 200-500ms

### Agent Loop
Claude uses tool use to autonomously:
- Execute shell commands
- Read/write/edit files
- Search the web
- Call MCP servers

### Tools Available (Phase 1)
| Tool | What it does |
|------|-------------|
| `shell_exec` | Run any shell command |
| `file_read` | Read a file |
| `file_write` | Create/overwrite a file |
| `file_edit` | Targeted string replacement |
| `list_dir` | List directory contents |
| `web_search` | Brave Search API |

---

## Configuration

Arca reads `arca.toml` from your project root or `~/.arca/config.toml`.

```toml
[voice]
stt_model = "base.en"       # tiny.en / base.en / small.en / medium.en / large-v3
vad_silence_ms = 800        # How long to wait after you stop talking

[tts]
engine = "kokoro"           # kokoro (local) | openai (cloud) | none
voice = "af_heart"          # Kokoro voice ID

[agent]
model = "claude-sonnet-4-6"
max_tokens = 8096

[tools]
shell = true
web_search = true           # Requires BRAVE_API_KEY
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...   # Required
BRAVE_API_KEY=BSA...           # Optional: enables web_search tool
OPENAI_API_KEY=sk-...          # Optional: enables OpenAI TTS
```

---

## Architecture

```
arca/
├── arca/
│   ├── voice/
│   │   ├── vad.py          # Silero VAD — detects speech start/end
│   │   ├── stt.py          # faster-whisper — audio → text
│   │   ├── tts.py          # Kokoro/OpenAI — text → speech
│   │   └── pipeline.py     # Orchestrates VAD + STT
│   ├── agent/
│   │   ├── loop.py         # Claude agentic loop (think → act → observe)
│   │   ├── tools.py        # Tool schemas + handlers
│   │   └── memory.py       # SQLite conversation history
│   ├── terminal/
│   │   ├── ui.py           # Textual TUI
│   │   └── shell.py        # PTY shell runner
│   ├── config.py           # Config loader (arca.toml)
│   └── main.py             # Entry point + wiring
├── scripts/
│   └── setup.sh            # First-run setup
├── arca.toml               # Default config
└── PHASES.md               # Technical build documentation
```

---

## Roadmap

See [PHASES.md](PHASES.md) for the full build plan.

- **Phase 1** (now): Python TUI, voice pipeline, Claude tool use, SQLite memory
- **Phase 2**: Wake word, MCP integration, ambient mode, project context detection
- **Phase 3**: Tauri + xterm.js production app, plugin system, Ollama local model fallback
- **Phase 4**: Open source launch, Pro tier, team features

---

## Requirements

- Python 3.11+
- macOS (Apple Silicon recommended for Metal GPU acceleration)
- `ANTHROPIC_API_KEY`
- Microphone + speakers (for voice mode)

---

## License

MIT
