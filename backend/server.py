"""SignBridge backend server - serves the Engine over one local WebSocket.

Run from the project root so the flat sibling modules (tracking.py,
modes.py, ...) resolve on sys.path:

    .venv/bin/python -m backend.server

Then connect to ws://127.0.0.1:8765/ws. Wire protocol (JSON, tagged by
``type``) is documented in docs/ARCHITECTURE.md:

    landmarks | view_state | prompt | recording   (backend -> frontend)
    event                                          (frontend -> backend)

An ``event`` message looks like ``{"type": "event", "action": "...", ...}``;
the action names match ``Mode.KEYMAP`` in modes.py plus the globals handled
in ``Engine.handle_action`` (``mode.next``, ``mode.prev``,
``recording.cancel``), plus the two prompt actions ``prompt.submit``
(with a ``value``) and ``prompt.cancel``.

Camera I/O is blocking, so the engine loop runs in a background thread (not
on the asyncio event loop) and fans each frame's messages out to one
asyncio.Queue per connected browser tab.

frontend/dist, once Phase 3 builds it, gets served as static files at "/".
Until then this process is WebSocket-API-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.runtime import Engine, EngineConfig

app = FastAPI(title="SignBridge")

_engine: Engine | None = None
_engine_lock = threading.Lock()
_engine_config = EngineConfig()


def _get_engine() -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = Engine(_engine_config)
    return _engine


class ConnectionHub:
    """Runs Engine.step() in a background thread and fans messages out to
    every connected client's queue."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self._queues: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            try:
                result = self.engine.step()
            except Exception as exc:  # keep the loop alive, surface nothing fatal
                print(f"[backend] engine.step() failed: {exc!r}")
                time.sleep(0.2)
                continue
            if not result.get("ok"):
                time.sleep(0.05)
                continue

            messages = [
                {"type": "landmarks", "t": time.time(), **result["landmarks"]},
                {"type": "view_state", "mode": self.engine.mode_name, "payload": result["view_state"]},
            ]
            if result["prompt"] is not None:
                messages.append({"type": "prompt", **result["prompt"]})
            if result["recording"] is not None:
                messages.append({"type": "recording", **result["recording"]})
            for msg in messages:
                self._broadcast(msg)

    def _broadcast(self, msg: dict) -> None:
        if self._loop is None:
            return
        with self._lock:
            queues = list(self._queues)
        for q in queues:
            self._loop.call_soon_threadsafe(_put_nowait_safe, q, msg)

    def add_queue(self) -> "asyncio.Queue":
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            self._queues.append(q)
        return q

    def remove_queue(self, q: "asyncio.Queue") -> None:
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)


def _put_nowait_safe(q: "asyncio.Queue", msg: dict) -> None:
    try:
        q.put_nowait(msg)
    except asyncio.QueueFull:
        pass  # a slow client drops frames rather than backing up the whole hub


_hub: ConnectionHub | None = None
_hub_lock = threading.Lock()


def _get_hub(loop: asyncio.AbstractEventLoop) -> ConnectionHub:
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = ConnectionHub(_get_engine())
            _hub.start(loop)
        return _hub


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    engine = _get_engine()
    hub = _get_hub(asyncio.get_running_loop())
    queue = hub.add_queue()

    async def sender() -> None:
        while True:
            msg = await queue.get()
            await ws.send_text(json.dumps(msg))

    sender_task = asyncio.create_task(sender())
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "event":
                continue
            action = msg.get("action")
            if action == "prompt.submit":
                engine.handle_prompt_submit(msg.get("value", ""))
            elif action == "prompt.cancel":
                engine.handle_prompt_cancel()
            elif action:
                kw = {k: v for k, v in msg.items() if k not in ("type", "action")}
                engine.handle_action(action, **kw)
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        hub.remove_queue(queue)


# Serve the built frontend, once Phase 3 produces frontend/dist. Mounted
# last and at "/" so it never shadows the /ws route above - Starlette
# matches routes in registration order.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--tracker", choices=("holistic", "hands"), default="holistic")
    ap.add_argument("--mode", default="communicator")
    ap.add_argument("--swap-hands", action="store_true")
    args = ap.parse_args()

    global _engine_config
    _engine_config = EngineConfig(
        camera=args.camera, tracker=args.tracker, mode=args.mode, swap_hands=args.swap_hands)

    if not _FRONTEND_DIST.exists():
        print(f"[backend] note: {_FRONTEND_DIST} not built yet - serving the "
              f"WebSocket API only (no frontend/dist to serve as static files).")
    print(f"[backend] SignBridge server on http://{args.host}:{args.port} "
          f"(WebSocket at /ws)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
