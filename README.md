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

# 2. Setup (creates .venv, downloads Whisper + Kokoro models)
bash scripts/setup.sh

# 3. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Launch
source .venv/bin/activate
arca
```

### Usage

```bash
arca                    # Full TUI with voice
arca --no-voice         # Text-only mode
arca --text "query"     # One-shot query, then exit
arca --serve            # WebSocket sidecar for Tauri desktop app
arca --model claude-opus-4-6  # Override model
arca --reset            # Clear all conversation history
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

### Memory
- **SQLite**: rolling window of last 50 messages (short-term)
- **chromadb**: semantic search across all sessions (long-term) — Claude remembers relevant past context automatically

### Phase 2 Features
- **Project Context**: reads CLAUDE.md, README, git status — Claude knows your project before you say a word
- **MCP Integration**: connect any MCP server (GitHub, Notion, etc.) via `arca.toml`
- **Ambient Monitor**: background error detection — if a shell command fails, Claude explains why unprompted
- **Wake Word**: always-on "hey arca" trigger (openwakeword)
- **Semantic Memory**: chromadb retrieves relevant past exchanges from any session

### Tools Available
| Tool | What it does |
|------|-------------|
| `shell_exec` | Run any shell command |
| `file_read` | Read a file |
| `file_write` | Create/overwrite a file |
| `file_edit` | Targeted string replacement |
| `list_dir` | List directory contents |
| `web_search` | Brave Search API |
| `mcp__*` | Any tool from connected MCP servers |

---

## Tauri Desktop App (Phase 3)

The `app/` directory contains the Tauri + React desktop app — a native Mac window with xterm.js terminal and Apple dark glass UI.

```bash
# Start Python sidecar first
source .venv/bin/activate
arca --serve

# Then in another terminal, run the Tauri app
cd app
npm run tauri dev
```

The Tauri frontend connects to `ws://127.0.0.1:8765`.

---

## Configuration

Arca reads `arca.toml` from your project root or `~/.arca/config.toml`.

```toml
[voice]
stt_model = "base.en"       # tiny.en / base.en / small.en / medium.en / large-v3
vad_silence_ms = 800

[tts]
engine = "kokoro"           # kokoro (local) | openai (cloud) | none
voice = "af_heart"

[agent]
model = "claude-sonnet-4-6"
max_tokens = 8096

[tools]
shell = true
web_search = true           # Requires BRAVE_API_KEY

[context]
enabled = true              # Auto-detect project stack, read CLAUDE.md/README

[ambient]
enabled = false             # Proactive error suggestions

# Add MCP servers
[[mcp.servers]]
name = "github"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "$GITHUB_TOKEN" }
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
│   │   ├── vad.py          # Silero VAD
│   │   ├── stt.py          # faster-whisper
│   │   ├── tts.py          # Kokoro/OpenAI TTS
│   │   ├── pipeline.py     # VAD + STT orchestration
│   │   └── wakeword.py     # Always-on wake word
│   ├── agent/
│   │   ├── loop.py         # Claude agentic loop
│   │   ├── tools.py        # Tool schemas + handlers
│   │   ├── memory.py       # SQLite + chromadb memory
│   │   ├── context.py      # Project context detection
│   │   ├── mcp_client.py   # MCP server connection
│   │   └── ambient.py      # Background error monitor
│   ├── terminal/
│   │   ├── ui.py           # Textual TUI
│   │   └── shell.py        # PTY shell
│   ├── server.py           # WebSocket sidecar for Tauri
│   ├── config.py           # Config loader
│   └── main.py             # Entry point
├── app/                    # Tauri desktop app (Phase 3)
│   ├── src/                # React + xterm.js frontend
│   └── src-tauri/          # Rust shell
├── scripts/
│   └── setup.sh
└── arca.toml
```

---

## Roadmap

- **Phase 1** ✅: Python TUI, voice pipeline, Claude tool use, SQLite memory
- **Phase 2** ✅: Wake word, MCP integration, ambient mode, semantic memory, project context
- **Phase 3** 🔧: Tauri desktop app, xterm.js, WebSocket sidecar (in progress)
- **Phase 4**: Open source launch, Pro tier, Homebrew Cask

---

## Requirements

- Python 3.11+
- macOS (Apple Silicon recommended for Metal GPU acceleration)
- `ANTHROPIC_API_KEY`
- Microphone + speakers (for voice mode)

---

## License

MIT
