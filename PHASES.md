# Arca — Build Phases & Technical Documentation

This document explains exactly what is happening in each phase of Arca's development — what components exist, how they connect, and what's happening under the hood. This is the source of truth for the architecture as it evolves.

---

## Phase 1 — Python TUI Prototype

**Goal:** Prove the core voice-to-agent loop works. Ship something usable daily.
**Stack:** Python + Textual TUI + faster-whisper + Kokoro TTS + Claude API
**Status:** In progress

---

### What's Built in Phase 1

```
arca/
├── arca/
│   ├── voice/
│   │   ├── vad.py          ✓
│   │   ├── stt.py          ✓
│   │   ├── tts.py          ✓
│   │   └── pipeline.py     ✓
│   ├── agent/
│   │   ├── loop.py         ✓
│   │   ├── tools.py        ✓
│   │   └── memory.py       ✓
│   ├── terminal/
│   │   ├── ui.py           ✓
│   │   └── shell.py        ✓
│   ├── config.py           ✓
│   └── main.py             ✓
└── scripts/setup.sh        ✓
```

---

### Data Flow: How a Voice Command Works

This is the complete end-to-end journey of a voice command through Phase 1 Arca.

```
1. User presses Ctrl+Space
   └─ ArcaApp.action_toggle_voice() called
      └─ voice_trigger() callback fires
         └─ New background thread spawned (_listen)

2. VAD opens mic (sounddevice.InputStream)
   └─ Audio read in 512-sample chunks (32ms each)
   └─ Each chunk → Silero VAD model → speech probability (0.0–1.0)
   └─ Pre-roll buffer (320ms) kept in case speech started mid-chunk
   └─ When probability ≥ threshold:
        - Pre-roll added to recording buffer
        - UI status bar → "● Recording..."
   └─ When silence detected for vad_silence_ms:
        - Recording stops
        - Audio buffer finalized as numpy float32 array

3. Audio → faster-whisper (STT)
   └─ WhisperModel.transcribe(audio, beam_size=5, language="en")
   └─ Returns segments → joined into text string
   └─ On Apple Silicon: ~300ms for base.en with Metal GPU
   └─ UI: show transcribed text

4. Text → AgentLoop.run()
   └─ Augmented with [CWD: /path/to/project]
   └─ Added to Memory (SQLite)
   └─ Memory.load_history() → last N messages as Claude API format

5. Claude API call (streaming)
   └─ client.messages.stream(
        model="claude-sonnet-4-6",
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        messages=conversation_history
      )
   └─ Events streamed as they arrive:
        RawContentBlockStartEvent  → new block (text or tool_use)
        RawContentBlockDeltaEvent  → text chunk or tool input JSON delta
        RawContentBlockStopEvent   → block complete

6a. If Claude returns text (stop_reason="end_turn"):
    └─ Text displayed in TUI in real-time via on_text_chunk callback
    └─ Full response stored in Memory
    └─ Response passed to TTS

6b. If Claude calls a tool (stop_reason="tool_use"):
    └─ Tool name + input extracted from response
    └─ dispatch_tool(name, inputs) called:
         shell_exec  → subprocess.run(command, shell=True)
         file_read   → Path(path).read_text()
         file_write  → Path(path).write_text(content)
         file_edit   → str.replace(old, new)
         list_dir    → Path(path).iterdir()
         web_search  → httpx.get(Brave API)
    └─ Result serialized as JSON
    └─ Tool result sent back as "user" role message with type="tool_result"
    └─ Loop continues → Claude sees result → generates next response or calls more tools
    └─ This loop repeats until Claude is done (no more tool_use in response)

7. TTS
   └─ Final text response → Kokoro ONNX model
   └─ kokoro.create(text, voice="af_heart", speed=1.1)
   └─ Returns (samples: np.ndarray, sample_rate: int)
   └─ sounddevice.play(samples, sample_rate) → speakers
   └─ Runs in daemon thread (non-blocking)
```

---

### Component Deep Dives

#### VAD (arca/voice/vad.py)

Silero VAD is a lightweight neural network (~1MB) that outputs a single float (speech probability) for each 512-sample chunk of 16kHz audio. It runs in ~1ms per chunk.

Key parameters:
- `threshold`: Default 0.5. Lower = more sensitive (picks up whispers). Higher = less false positives.
- `silence_ms`: Default 800ms. How long silence must persist before recording ends.
- Pre-roll: 320ms of audio buffered before speech starts, so the first syllable isn't clipped.

```python
# Internal loop inside VAD.record_until_silence()
while True:
    chunk = mic.read(512)           # 32ms of audio
    prob = silero_model(chunk)      # 0.0–1.0 speech probability
    if prob >= 0.5:
        # Speech — buffer it
        audio_buffer.append(chunk)
        silence_count = 0
    elif speech_started:
        # Silence after speech — count down
        audio_buffer.append(chunk)
        silence_count += 1
        if silence_count >= 25:    # 800ms / 32ms = 25 chunks
            break                  # Done
```

#### STT (arca/voice/stt.py)

faster-whisper wraps OpenAI Whisper using CTranslate2 for optimized inference. On Apple Silicon, it uses CoreML/Metal via the `auto` device setting.

Model selection tradeoffs:
```
tiny.en    75MB   ~150ms   Good enough for commands
base.en   145MB   ~300ms   Recommended — fast + accurate
small.en  466MB   ~600ms   Better for long sentences
medium.en 1.4GB  ~1200ms   High accuracy
large-v3  3.0GB  ~3000ms   Best accuracy (use for transcription tasks)
```

The built-in VAD filter (`vad_filter=True`) skips silent segments, which is redundant with our Silero VAD but provides an extra layer of filtering for edge cases.

#### Agent Loop (arca/agent/loop.py)

The loop implements the Claude tool use protocol exactly:

```
User message
    ↓
Claude API call (streaming)
    ↓
Response contains:
    - text blocks    → stream to UI
    - tool_use blocks → collect (may be multiple in one response)
    ↓
If tool_use blocks present:
    - Execute each tool
    - Assemble tool_result blocks
    - Append [assistant: {text + tool_use}] to messages
    - Append [user: {tool_results}] to messages
    - Call Claude API again with full updated history
    ↓
Repeat until stop_reason = "end_turn" (no tool calls)
    ↓
Return final text
```

Important: The "user" role is used for both actual user messages AND tool results. This is the Claude API spec — tool results are always sent as user-role messages.

#### Memory (arca/agent/memory.py)

SQLite stores every message in the current session. On each API call, the last `max_history` messages are loaded and sent to Claude. This keeps the conversation coherent without hitting token limits.

Schema:
```sql
sessions(id, created_at, project_path)
messages(id, session_id, role, content_json, created_at)
```

`content_json` stores either:
- A plain string (for simple user/assistant text messages)
- A JSON array (for structured content blocks — tool_use, tool_result)

Future (Phase 2): Add semantic search via chromadb to retrieve relevant past context even when it's outside the rolling window.

#### TUI (arca/terminal/ui.py)

Textual is a Python TUI framework with a React-like component model. The app runs a single-threaded event loop. All agent/voice work runs in daemon threads and uses `call_from_thread()` to post updates back to the UI thread safely.

```
Main thread:     Textual event loop
                 ├─ Keyboard events
                 ├─ Input submission
                 └─ UI updates (from call_from_thread)

Background threads:
  voice_thread:  VAD → STT → text result
  agent_thread:  Claude API → tools → response
  tts_thread:    Kokoro synthesis → speakers
```

---

### Configuration System (arca/config.py)

Config is loaded in priority order:
1. `~/.arca/config.toml` — user global defaults
2. `arca.toml` in project root (or any parent dir up to git root) — project overrides

Both are merged (deep merge), with project config winning. This means you can have a global `base.en` Whisper model but a specific project that uses `large-v3` for better accuracy.

---

### Phase 1 Week-by-Week Plan

**Week 1: Voice loop**
- [x] faster-whisper STT
- [x] Silero VAD
- [x] Kokoro TTS
- [x] VoicePipeline: listen() → text
- [ ] Manual integration test: speak → transcribe → print → speak back

**Week 2: Agent loop**
- [x] Claude tool use protocol
- [x] shell_exec, file_read/write/edit, list_dir, web_search tools
- [x] SQLite memory
- [ ] Integration test: voice → Claude → shell command → result → TTS

**Week 3: TUI**
- [x] Textual app skeleton
- [x] RichLog output area
- [x] Input bar + status bar
- [x] Ctrl+Space binding
- [ ] Wire agent callbacks → UI updates
- [ ] End-to-end dogfood session

**Week 4: Polish + stability**
- [ ] Error handling for missing models / API errors
- [ ] `arca --text` one-shot mode working
- [ ] Config loading tested (arca.toml in project vs global)
- [ ] `bash scripts/setup.sh` works clean on fresh machine
- [ ] Push to GitHub, tag v0.1.0

---

## Phase 2 — Features & MCP Integration

**Goal:** Daily driver for real development work.
**Status:** Complete — v0.2.0

### What's Built in Phase 2

```
arca/
├── arca/
│   ├── voice/
│   │   └── wakeword.py         ✓  Always-on wake word (openwakeword / porcupine)
│   ├── agent/
│   │   ├── context.py          ✓  Project context auto-detection
│   │   ├── mcp_client.py       ✓  MCP server connection manager
│   │   └── ambient.py          ✓  Background error monitor + proactive suggestions
│   │   ├── loop.py             ✓  Updated: context + semantic memory + MCP tools
│   │   └── memory.py           ✓  Updated: chromadb semantic search layer
│   ├── terminal/
│   │   └── ui.py               ✓  Updated: ambient suggestion rendering
│   ├── config.py               ✓  Updated: ContextConfig, AmbientConfig, MCPConfig
│   └── main.py                 ✓  Updated: full Phase 2 boot sequence
└── arca.toml                   ✓  Phase 2 config sections
```

---

### Data Flow: How Phase 2 Changes Everything

#### Boot Sequence (new in Phase 2)

```
1. load_config()
   └─ Reads ~/.arca/config.toml + project arca.toml
   └─ Merges deep: project config wins over global

2. detect_project_context()
   └─ Finds git root (walks up from CWD)
   └─ Reads: CLAUDE.md (3000 chars), README.md (1500 chars), .cursorrules
   └─ Runs: git branch, git log -5, git status --short
   └─ Detects stack: pyproject.toml→Python, package.json→Node, etc.
   └─ Returns context string → injected into Claude system prompt
   └─ Claude now KNOWS the project before the first message

3. Memory(semantic_search=True)
   └─ Opens SQLite at ~/.arca/memory.db
   └─ Opens chromadb at ~/.arca/vectors/ (persistent PersistentClient)
   └─ Collection: "arca_messages" with cosine similarity index
   └─ On each add_message(): indexes text in chromadb (role≠"tool")

4. MCPClient.start() [if servers configured]
   └─ Spawns asyncio event loop in daemon thread
   └─ For each configured server:
        subprocess(command, args, env) → stdio pipes
        ClientSession.initialize() → handshake
        list_tools() → get all tool schemas
        Prefixes: "mcp__{server_name}__{tool_name}"
   └─ Blocks main thread until all connect (30s timeout)
   └─ Sessions stay alive indefinitely in the async loop

5. AmbientMonitor.start() [if enabled]
   └─ Daemon thread with queue.Queue(maxsize=50)
   └─ Waits for ShellEvent objects from shell_exec tool calls
   └─ Rate limited: 1 suggestion per 30s minimum

6. AgentLoop(project_context=..., mcp_client=..., ambient_monitor=...)
   └─ system = BASE_PROMPT + project_context + extra
   └─ tools = built-in tools + mcp_client.list_tools()

7. WakeWordDetector.start() [if enabled]
   └─ Daemon thread with sounddevice.InputStream at 16kHz
   └─ Feeds 1280-sample chunks to openwakeword model
   └─ When score ≥ threshold → calls voice_trigger() callback
   └─ Cooldown: 2s between detections
```

#### Per-Turn Flow (updated in Phase 2)

```
User speaks/types "fix the failing test"
    ↓
AgentLoop.run("fix the failing test")
    ↓
Step 1: Semantic search
  chromadb.query("fix the failing test", n_results=5)
  └─ Returns up to 5 similar past messages from other sessions
  └─ These become "ghost" messages prepended to history
  └─ Claude sees: "Past context: ..." before the conversation

Step 2: memory.load_history()
  └─ Last 50 messages from current SQLite session

Step 3: messages = semantic_context + current_history
  └─ Combined list sent to Claude API

Step 4: Claude API call (streaming)
  └─ System prompt includes project context
  └─ Tool list includes both built-in + MCP tools
  └─ Claude might call: shell_exec("pytest") or mcp__github__list_prs(...)

Step 5: Tool dispatch
  if tool_name.startswith("mcp__"):
      mcp_client.call_tool(name, args)
      └─ asyncio.run_coroutine_threadsafe → async loop thread
      └─ session.call_tool(original_name, args)
      └─ Returns within 30s timeout
  else:
      dispatch_tool(name, args)  ← built-in handlers

Step 6: Ambient feed
  if tool_name == "shell_exec":
      ambient_monitor.feed(cmd, stdout, stderr, exit_code)
      └─ Non-blocking: queue.put_nowait()
      └─ Background thread checks: exit_code≠0 or error pattern?
      └─ If yes AND rate limit OK → Claude haiku generates 1-2 sentence hint
      └─ Callback fires → TUI shows yellow "◉ Arca (ambient)" suggestion
```

---

### Component Deep Dives (Phase 2)

#### Project Context (arca/agent/context.py)

Detects project type by checking manifest files in git root:
```
pyproject.toml  → Python
package.json    → Node.js
go.mod          → Go
Cargo.toml      → Rust
pom.xml         → Java/Maven
Dockerfile      → Docker
*.tf files      → Terraform
```

Git info: branch, last 5 commits, count of uncommitted changes.

Key files read and truncated:
- CLAUDE.md: 3000 chars (AI-specific project instructions — most valuable)
- README.md: 1500 chars (project overview)
- .cursorrules: 1000 chars (often has project conventions)

This context is injected once into Claude's system prompt at boot.
It costs ~2K tokens per session but makes Claude immediately useful.

#### MCP Client (arca/agent/mcp_client.py)

Architecture: async sessions in a dedicated thread, sync interface for the agent loop.

```
Main thread (sync):          Async thread (event loop):
  mcp_client.call_tool()  →  asyncio.run_coroutine_threadsafe()
  future.result(30s)      ←  session.call_tool() completes
```

Tool naming convention:
  Server name: "github"
  Original tool: "list_pull_requests"
  Arca tool key: "mcp__github__list_pull_requests"

Claude sees this prefixed name and calls it naturally.
We strip the prefix when routing to the correct server session.

Session lifecycle:
- Sessions opened once at startup, kept alive indefinitely
- If a server crashes, the async task exits silently
- Phase 3 will add reconnection logic

#### Semantic Memory (arca/agent/memory.py)

Two-layer memory:
```
SQLite (short-term):     Last N messages, exact order, current session
chromadb (long-term):    All sessions, semantic similarity, any time

Query: "how do I run the tests"
SQLite: "...3 messages ago you ran: npm test"
chroma: "...2 weeks ago you and Arca discussed: pytest -v --cov=src"
Both injected → Claude has full context without re-asking
```

chromadb uses default embeddings (all-MiniLM-L6-v2, ~80MB, auto-downloaded).
Tool results are NOT indexed (too noisy; raw JSON pollutes the vector space).
Only user and assistant text messages are indexed.

#### Ambient Monitor (arca/agent/ambient.py)

Error classification uses regex patterns — intentionally simple and fast:
- Non-zero exit code (catch-all)
- Specific patterns: "Error:", "Traceback", "failed to", "command not found", etc.

When triggered:
```python
prompt = f"Command: {cmd}\nExit code: {exit_code}\nOutput:\n{output[:800]}\n\nIn one or two sentences, what's the likely cause and fix? Be direct. No preamble."
```
Uses claude-haiku-4-5 with max_tokens=150 for fast, cheap, focused responses.

Rate limit: 1 suggestion per 30s to avoid being annoying.

#### Wake Word (arca/voice/wakeword.py)

openwakeword processes 1280-sample chunks (80ms) through an ONNX model.
Models are small (~1MB each), run on CPU in real-time.

"hey_jarvis" is used as a phonetic proxy for "hey arca" in Phase 2.
Phase 3 will train a custom "hey arca" model using openWakeWord's
training pipeline (requires ~30 minutes of wake word audio samples).

Porcupine alternative: Higher accuracy, commercial API key required.
Good for production, not ideal for open source distribution.

---

### Phase 2 Config Reference

```toml
# Voice: wake word (new)
[voice]
wake_word_enabled = false        # Set true to enable always-on mode
wake_word_engine = "openwakeword"
wake_word_model = "hey_jarvis"
wake_word_threshold = 0.5

# Project context (new)
[context]
enabled = true
read_claude_md = true
read_readme = true
read_git_info = true
max_context_chars = 6000

# Ambient monitoring (new)
[ambient]
enabled = false                  # Set true for proactive suggestions
model = "claude-haiku-4-5-20251001"
min_interval_s = 30.0
speak_suggestions = true

# Semantic memory (updated)
[memory]
semantic_search = true
semantic_db_path = "~/.arca/vectors"
semantic_top_k = 5

# MCP servers (new)
[[mcp.servers]]
name = "github"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "$GITHUB_TOKEN" }
```

---

## Phase 3 — Production App (Tauri + xterm.js)

**Goal:** Installable Mac app with full terminal emulation.
**Why Tauri over Electron:**
- Native performance (Rust backend, no Node/V8 overhead)
- ~10MB bundle vs ~200MB Electron
- xterm.js for real PTY terminal (colors, interactivity, scrollback)
- node-pty for shell process management

**Architecture changes:**
- Python voice/agent backend runs as local sidecar process
- Tauri frontend connects via local WebSocket or Unix socket
- xterm.js renders the PTY output
- React + Tailwind UI layer (Apple-inspired dark theme)

**New capabilities:**
- Full PTY: `vim`, `htop`, `git diff --color` work correctly
- Ollama integration: local model fallback (llama3, qwen2.5-coder)
- Plugin system: drop TOML configs to add custom skills/tools
- `.dmg` installer + Homebrew Cask

---

## Phase 4 — Distribution

**Goal:** Public product, revenue, community.

**Open source:**
- MIT license core
- GitHub: `jmenzies722/arca`
- Docs: mintlify.com
- Launch: Hacker News, ProductHunt, Twitter/X

**Pro tier ($20/mo):**
- Premium voices (ElevenLabs, custom voice cloning)
- Cloud conversation sync across machines
- Team MCP configs (shared tool access)
- Priority support

**Positioning:**
- "The terminal for developers who talk to their computer"
- "Claude Code + Warp + a voice, unified"
- "Local-first — your code never leaves your machine"

---

## Technical Decisions Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| faster-whisper over Whisper.cpp | Python-native, easier to integrate with Python stack | whisper.cpp is faster but needs FFI |
| Silero VAD over WebRTC VAD | Neural model, better accuracy in noisy environments | WebRTC VAD is simpler but rule-based |
| Kokoro over ElevenLabs | Local-first, no API cost, #1 quality rating | ElevenLabs has better emotional range |
| Textual over Blessed/Urwid | Modern, React-like, active development | Blessed more established but older |
| SQLite over JSON files | ACID transactions, queryable history | JSON simpler but no query capability |
| Claude tool use over function calling | Native protocol, better multi-step reasoning | LangChain/LangGraph heavier abstraction |
| Tauri (Phase 3) over Electron | 10x smaller bundle, native performance | Electron more mature ecosystem |
