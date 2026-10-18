# SignBridge frontend

A thin display + input layer for SignBridge's `backend/server.py` — it owns
no recognition logic, only rendering and keyboard/click events over one
WebSocket. See `../docs/ARCHITECTURE.md` for the full wire protocol and
`../docs/REWORK_ROADMAP.md` for how this fits into the larger rework.

Styled from `DESIGN.md` (Linear's marketing-site design system, fetched
from [awesome-design-md](https://github.com/VoltAgent/awesome-design-md),
adapted here for a product/tool UI). Dark-only by deliberate choice — this
app is used over a live camera feed in all kinds of lighting.

## Develop

```bash
npm install
npm run dev          # http://localhost:5173, proxies /ws to the backend
```

Run the backend alongside it (from the project root):

```bash
.venv/bin/python -m backend.server    # ws://127.0.0.1:8765/ws
```

## Build

```bash
npm run build         # -> dist/, served by backend/server.py as static files
```

Once `dist/` exists, `python -m backend.server` alone serves the whole app
at `http://127.0.0.1:8765` — no separate frontend server needed.

## Structure

| Path | Purpose |
|---|---|
| `src/types.ts` | TypeScript types mirroring the backend's wire protocol |
| `src/lib/ws.ts` | WebSocket connection, auto-reconnect, message dispatch |
| `src/lib/keyboard.ts` | Keydown → named action, mirroring each mode's `KEYMAP` in `../modes.py` |
| `src/state/store.ts` | Zustand store holding the latest server state |
| `src/components/VideoOverlay.tsx` | Streamed camera frame + animated landmark overlay |
| `src/components/modes/` | One view per mode (Communicator/Coach/Control/Pointer) |
| `src/components/PromptModal.tsx` | Generic text-entry modal for the `prompt` message type |
