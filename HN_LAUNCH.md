# HN Launch Post — Draft

**Title:** Show HN: Arca – AI voice agentic terminal (Claude + local Whisper + Kokoro TTS)

---

**Body:**

I built Arca because I wanted to talk to my terminal the way I talk to a senior engineer.

Not just "run this command" — but "the auth tests are failing after the refactor, figure out why and fix it." Then watch it read the files, run the tests, diagnose the issue, patch it, and confirm it's green. All while narrating what it's doing in a natural voice.

**What it is:**
Arca is a locally-running AI terminal that treats voice and keyboard as equals. You speak (or type), it reasons with Claude's tool use API, executes shell commands / reads files / searches the web, and responds via text-to-speech. Everything runs on your machine — no cloud dependencies except the Claude API.

**The stack:**
- Claude Sonnet (tool use API) — the brain
- faster-whisper + Silero VAD — local STT, ~300ms on Apple Silicon
- Kokoro-82M ONNX — local TTS, #1 on HF TTS Arena, ~200ms
- SQLite + chromadb — rolling conversation history + semantic search across sessions
- Textual TUI (Python) for terminal mode
- Tauri + xterm.js for the desktop app (Phase 3, just landed)

**Phase 2 features already in:**
- Project context detection: reads your CLAUDE.md/README/git status at boot, so Claude already knows what project you're in
- MCP integration: connect GitHub, Notion, or any MCP server via config
- Ambient monitor: background process watches shell failures and explains them unprompted
- Wake word: "hey arca" always-on trigger

**What makes it different from Claude Code / Open Interpreter / Warp:**
None of those three have voice as a first-class input. Talon Voice has voice but no AI reasoning. Arca is the only tool that does all of: real terminal execution + voice I/O + agentic reasoning + local-first models.

**Repo:** https://github.com/jmenzies722/arca

Setup is one script (`bash scripts/setup.sh`) — it creates a venv, downloads Whisper base.en (~145MB) and Kokoro models (~93MB), and you're running in a few minutes.

Would love feedback on the voice latency (Apple Silicon is ~500ms end-to-end), the wake word accuracy, and what MCP servers people would want connected out of the box.

---

**Tags to consider:** Show HN, voice, AI, terminal, Claude, open-source
