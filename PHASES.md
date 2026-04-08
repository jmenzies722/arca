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
**Planned additions:**
- Wake word detection (always-on mode via Porcupine)
- MCP passthrough — connect any existing MCP server
- Project context detection: auto-reads git root, README, CLAUDE.md
- Ambient mode: Arca watches your terminal and speaks up proactively
- Chromadb semantic search over conversation history
- `arca.toml` per-project tool permissions

**Architecture changes:**
- Add MCP client layer in `arca/agent/mcp_client.py`
- Add project context loader in `arca/agent/context.py`
- Add wake word thread in `arca/voice/wakeword.py`

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
