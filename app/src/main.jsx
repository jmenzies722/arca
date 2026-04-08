import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./global.css";

// StrictMode removed — it double-invokes effects in dev, creating two WebSocket
// connections. The server rejects the second, triggering an instant reconnect loop.
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
