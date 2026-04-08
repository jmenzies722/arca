import { useEffect, useRef, useState, useCallback } from "react";
import { Terminal } from "xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";
import "xterm/css/xterm.css";
import "./App.css";

const WS_URL = "ws://127.0.0.1:8765";
const RECONNECT_DELAY_MS = 2000;

// ── xterm theme — Apple dark glass ──────────────────────────────────────────
const TERMINAL_THEME = {
  background:    "transparent",
  foreground:    "#e8e8ed",
  cursor:        "#60a5fa",
  cursorAccent:  "#0a0a0a",
  selectionBackground: "rgba(96,165,250,0.25)",
  black:         "#1c1c1e",
  red:           "#ff453a",
  green:         "#32d74b",
  yellow:        "#ffd60a",
  blue:          "#0a84ff",
  magenta:       "#bf5af2",
  cyan:          "#5ac8fa",
  white:         "#e8e8ed",
  brightBlack:   "#636366",
  brightRed:     "#ff6961",
  brightGreen:   "#4cd964",
  brightYellow:  "#ffe066",
  brightBlue:    "#409cff",
  brightMagenta: "#da8fff",
  brightCyan:    "#70d7ff",
  brightWhite:   "#ffffff",
};

// ── Status config ────────────────────────────────────────────────────────────
const STATUS = {
  connecting: { label: "Connecting…",  dot: "bg-yellow-400 animate-pulse", text: "text-yellow-300" },
  idle:       { label: "Ready",        dot: "bg-green-400",                 text: "text-green-300"  },
  recording:  { label: "Recording…",   dot: "bg-red-400 animate-pulse",     text: "text-red-300"   },
  thinking:   { label: "Thinking…",    dot: "bg-blue-400 animate-pulse",    text: "text-blue-300"  },
  error:      { label: "Disconnected", dot: "bg-zinc-500",                  text: "text-zinc-400"  },
};

export default function App() {
  const termRef       = useRef(null);   // DOM node
  const xtermRef      = useRef(null);   // Terminal instance
  const fitRef        = useRef(null);   // FitAddon
  const wsRef         = useRef(null);   // WebSocket
  const inputRef      = useRef(null);   // Text input DOM
  const reconnectRef  = useRef(null);   // Reconnect timer

  const [status, setStatus]   = useState("connecting");
  const [input, setInput]     = useState("");
  const [voiceHeld, setVoice] = useState(false);

  // ── Terminal init ──────────────────────────────────────────────────────────
  useEffect(() => {
    const term = new Terminal({
      theme: TERMINAL_THEME,
      fontFamily: '"SF Mono", "JetBrains Mono", "Fira Code", monospace',
      fontSize: 13,
      lineHeight: 1.5,
      cursorBlink: true,
      cursorStyle: "bar",
      allowTransparency: true,
      scrollback: 5000,
    });

    const fit = new FitAddon();
    term.loadAddon(fit);
    term.loadAddon(new WebLinksAddon());
    term.open(termRef.current);
    fit.fit();
    xtermRef.current = term;
    fitRef.current   = fit;

    printWelcome(term);

    const onResize = () => fit.fit();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      term.dispose();
    };
  }, []);

  // ── WebSocket connection ───────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    setStatus("connecting");

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("idle");
      clearTimeout(reconnectRef.current);
      writeLine(xtermRef.current, "\x1b[2m[arca] Connected to backend\x1b[0m");
    };

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      const term = xtermRef.current;
      if (!term) return;

      switch (msg.type) {
        case "user_message":
          writeLine(term, `\r\n\x1b[1;32mYou\x1b[0m  ${msg.text}`);
          break;
        case "assistant_start":
          term.write("\r\n\x1b[1;36mArca\x1b[0m  ");
          break;
        case "text_chunk":
          term.write(msg.text);
          break;
        case "tool_start":
          term.write(`\r\n\x1b[2m  → ${msg.name}(${formatInputs(msg.inputs)})\x1b[0m`);
          break;
        case "tool_end":
          if (msg.result?.error) {
            term.write(`\r\n\x1b[31m  ✗ ${msg.result.error}\x1b[0m`);
          } else if (msg.result?.stdout?.trim()) {
            const lines = msg.result.stdout.trim().split("\n").slice(0, 8);
            lines.forEach(l => term.write(`\r\n\x1b[2m  ${l}\x1b[0m`));
          }
          break;
        case "status":
          setStatus(msg.state);
          break;
        case "ambient":
          writeLine(term, `\r\n\x1b[1;33m◉ Arca\x1b[2m (ambient)\x1b[0m  ${msg.text}`);
          break;
      }
    };

    ws.onclose = () => {
      setStatus("error");
      reconnectRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // ── Send message ───────────────────────────────────────────────────────────
  const send = useCallback((text) => {
    if (!text.trim() || wsRef.current?.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: "text", content: text.trim() }));
    setInput("");
  }, []);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  // ── Voice PTT ──────────────────────────────────────────────────────────────
  const voiceStart = useCallback(() => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    setVoice(true);
    wsRef.current.send(JSON.stringify({ type: "voice_start" }));
  }, []);

  const voiceStop = useCallback(() => {
    if (!voiceHeld) return;
    setVoice(false);
    wsRef.current?.send(JSON.stringify({ type: "voice_stop" }));
  }, [voiceHeld]);

  // Ctrl+Space global shortcut
  useEffect(() => {
    const down = (e) => { if (e.ctrlKey && e.code === "Space") { e.preventDefault(); voiceStart(); } };
    const up   = (e) => { if (e.code === "Space") voiceStop(); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, [voiceStart, voiceStop]);

  // Reset (Ctrl+R)
  useEffect(() => {
    const handler = (e) => {
      if (e.ctrlKey && e.key === "r") {
        e.preventDefault();
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: "reset" }));
          xtermRef.current?.clear();
          printWelcome(xtermRef.current);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const st = STATUS[status] ?? STATUS.error;

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] select-none">

      {/* ── Title bar / status ── */}
      <div className="flex items-center justify-between px-4 py-2
                      bg-[#111112]/80 backdrop-blur border-b border-white/[0.06]
                      shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-white/80 font-semibold text-sm tracking-wide">Arca</span>
          <span className="text-white/20 text-xs">v0.2.0</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${st.dot}`} />
          <span className={`text-xs ${st.text}`}>{st.label}</span>
        </div>
        <div className="text-white/20 text-xs hidden sm:block">
          Ctrl+Space: voice  ·  Ctrl+R: reset  ·  Ctrl+C: quit
        </div>
      </div>

      {/* ── Terminal ── */}
      <div className="flex-1 overflow-hidden px-1" ref={termRef} />

      {/* ── Input bar ── */}
      <div className="flex items-center gap-2 px-3 py-2
                      bg-[#111112]/80 backdrop-blur border-t border-white/[0.06]
                      shrink-0">
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a message or hold Ctrl+Space to speak…"
          className="flex-1 bg-white/[0.05] text-white/90 placeholder-white/20
                     text-sm px-3 py-1.5 rounded-lg border border-white/[0.08]
                     focus:outline-none focus:border-blue-500/50 focus:bg-white/[0.07]
                     transition-colors font-mono"
          autoFocus
        />

        {/* Voice button */}
        <button
          onMouseDown={voiceStart}
          onMouseUp={voiceStop}
          onTouchStart={voiceStart}
          onTouchEnd={voiceStop}
          className={`w-8 h-8 rounded-full flex items-center justify-center transition-all
                      ${voiceHeld
                        ? "bg-red-500 shadow-[0_0_12px_rgba(239,68,68,0.6)] scale-110"
                        : "bg-white/[0.08] hover:bg-white/[0.14] border border-white/[0.1]"
                      }`}
          title="Hold to speak (Ctrl+Space)"
        >
          <MicIcon active={voiceHeld} />
        </button>

        {/* Send button */}
        <button
          onClick={() => send(input)}
          disabled={!input.trim()}
          className="w-8 h-8 rounded-full flex items-center justify-center
                     bg-blue-600 hover:bg-blue-500 disabled:opacity-30
                     disabled:cursor-not-allowed transition-all"
          title="Send (Enter)"
        >
          <SendIcon />
        </button>
      </div>
    </div>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function printWelcome(term) {
  if (!term) return;
  term.writeln("\x1b[1;36mArca\x1b[0m \x1b[2mv0.2.0\x1b[0m — AI Voice Terminal");
  term.writeln("\x1b[2mSpeak or type. Ctrl+Space for voice. Ctrl+R to reset.\x1b[0m");
}

function writeLine(term, text) {
  term?.writeln(text);
}

function formatInputs(inputs = {}) {
  return Object.entries(inputs)
    .map(([k, v]) => `${k}=${JSON.stringify(String(v).slice(0, 30))}`)
    .join(", ");
}

function MicIcon({ active }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke={active ? "white" : "rgba(255,255,255,0.6)"} strokeWidth="2"
         strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="2" width="6" height="12" rx="3"/>
      <path d="M19 10a7 7 0 01-14 0"/>
      <line x1="12" y1="19" x2="12" y2="22"/>
      <line x1="8" y1="22" x2="16" y2="22"/>
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13"/>
      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  );
}
