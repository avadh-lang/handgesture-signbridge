# SignBridge — System Architecture

This document describes SignBridge's current architecture and the target
production architecture for the planned rework (see `docs/REWORK_ROADMAP.md`
for the phased implementation plan). Written to support the project's
research paper.

## Current architecture (as of 2026-10-18)

A single Python process, one OpenCV window, synchronous loop:

```
Webcam
  -> cv2.VideoCapture (app.py)
  -> tracking.py (MediaPipe Holistic/Hands)
  -> skeleton.Pose  (pure-numpy landmark representation)
  -> Mode.update(ctx)              [modes.py: Communicator/Coach/Control/Pointer]
       - computes recognition state (DTW match, fingerspell classify, Leitner, pointer FSM)
       - AND draws directly onto the frame via hud.py in the same call
  -> cv2.imshow + cv2.waitKey(1)   [app.py: the entire UI + the entire input/event loop]
```

Everything — recognition, application logic, and rendering — lives in one
process and one thread, synchronized to the camera's frame rate. Keyboard
input is raw `cv2.waitKey` integer keycodes, consumed directly by `app.py`
and by each mode's `on_key(code, ctx)`.

**Strengths:** simple, zero network/IPC overhead, fully offline, easy to
reason about end-to-end.

**Limitation driving this rework:** the UI is inseparable from OpenCV's
window — there is no way to present a modern interface without either (a)
replacing OpenCV's rendering entirely in-place, or (b) separating
computation from rendering so a different frontend can consume the same
computed state. The codebase already keeps `tracking.py`/`skeleton.py`
framework-free (pure numpy), so the recognition core is reusable as-is;
only the mode layer currently conflates computation with drawing.

## Target architecture: local inference server + thin web client

Chosen pattern for the next phase, and the one documented for the paper:

```
┌─────────────────────────────── one machine, no cloud ───────────────────────────────┐
│                                                                                        │
│  Webcam ──► backend/runtime.py Engine                        Browser (localhost tab)  │
│             ├─ tracking.py (MediaPipe) ──► skeleton.Pose                              │
│             ├─ signbank.py + dtw.py        (record-your-own-sign match)               │
│             ├─ fingerspell.py              (rule-based, + future ML classifier)       │
│             ├─ leitner.py                  (spaced-repetition state)                  │
│             ├─ pointer.py                  (cursor state machine)                     │
│             └─ Mode.compute(ctx) ──► ViewState (plain data, no drawing)               │
│                        │                                                              │
│                        ▼                                                              │
│             backend/server.py (FastAPI, one WebSocket route)                          │
│                        │  JSON, localhost only                                        │
│        ┌───────────────┼────────────────────────────────┐                            │
│        ▼               ▼                                ▼                            │
│   `landmarks`     `view_state`                      `prompt` / `recording`            │
│   (~30fps)         (~10-15Hz)                        (as needed)                      │
│        │               │                                │                             │
│        └───────────────┴────────────────────────────────┘                            │
│                        ▼                                                              │
│             React frontend (frontend/)                                                │
│             ├─ getUserMedia (own camera grab, for the visible video only)             │
│             ├─ <canvas> landmark overlay, interpolated for fluid motion               │
│             └─ per-mode views (Communicator/Coach/Control/Pointer)                    │
│                        │                                                              │
│                        ▼  `event` (named actions, e.g. "backspace", "mode.next")      │
│             back to backend/server.py -> Mode.on_action(...)                          │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### Why this pattern

- **Not a native desktop GUI (PyQt/Tk):** would fix "it's a raw OpenCV window"
  but still look and feel dated, and doesn't produce a "production system
  architecture" story worth a paper section — it stays a single monolithic
  app.
- **Not a browser-only app with recognition reimplemented in JS:** would
  duplicate the DTW matcher, fingerspelling classifier, Leitner scheduler,
  and pointer state machine in a second language, doubling maintenance and
  risking the two implementations drifting apart. Rejected.
- **Local inference server + thin client (chosen):** 100% of recognition
  and application logic stays in Python, server-side, unchanged in
  substance — the browser is a display and input layer only. This keeps the
  offline/no-cloud guarantee (the WebSocket never leaves `127.0.0.1`), lets
  the existing `selftest.py` suite keep testing the same logic it tests
  today, and gives a clean "edge inference server" architecture to describe
  in the paper: perception layer (MediaPipe) → recognition layer (DTW/ML/
  rules) → application layer (modes/state) → presentation layer (web UI),
  with a well-defined message-passing boundary between the last two.

### Wire protocol (WebSocket, `ws://127.0.0.1:8765/ws`, JSON frames tagged by `type`)

| Type | Direction | Rate | Purpose |
|---|---|---|---|
| `landmarks` | backend → frontend | ~30fps | Raw hand/body landmark coordinates only (no JPEG — the browser captures its own camera feed via `getUserMedia` for the visible video; Python's camera read only drives recognition) |
| `view_state` | backend → frontend | ~10-15Hz | Per-mode application state (text, scores, bound commands, pointer status) — one payload shape per mode |
| `event` | frontend → backend | on input | Named action strings (`"backspace"`, `"mode.next"`, `"communicator.accept_suggestion"`) replacing raw keycodes |
| `prompt` | backend → frontend | as needed | Async text-entry request (e.g. "command sign name"), replacing today's synchronous `ctx.request_prompt()` |
| `recording` | backend → frontend | during capture | Countdown/progress while teaching a new sign (3-rep recorder state) |

Full per-mode `view_state` payload shapes and the `on_key`→`on_action`
migration plan are specified in `docs/REWORK_ROADMAP.md` (Phase 2/3).

### What stays unchanged

- `tracking.py`, `skeleton.py`, `signbank.py`, `dtw.py`, `leitner.py`,
  `pointer.py`, `filters.py`, `recorder.py`, `speller.py`, `glosser.py`,
  `commands.py`, `output_sink.py` — all pure logic, reused as-is.
- `app.py` stays as a working CLI/debug fallback (OpenCV window), not
  deleted — useful for quick testing without a browser.
- `selftest.py`'s existing 77 tests keep testing the same recognition
  logic; new tests are additive, not replacements.

### Known open questions (tracked in `docs/REWORK_ROADMAP.md`'s risk list)

- Whether the OS allows the browser and Python to read the same physical
  webcam concurrently (unproven, needs validation before frontend work
  depends on it; has a documented fallback).
- Whether landmark smoothing (One-Euro filter) should apply before
  recognition (risk: blurs fast legitimate transitions) or only to the
  display overlay — an open design decision, not yet made.
