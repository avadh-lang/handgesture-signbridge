# SignBridge Rework Roadmap (Phases 2–5)

Written 2026-10-18 as the detailed, ready-to-execute plan for the actual
UI/backend/ML rebuild, following the Phase 1 documentation pass
(`docs/ARCHITECTURE.md`, `docs/competitor_analysis.md`). This is the plan a
future session should pick up and execute, phase by phase, without needing
to re-derive any of this research.

See `docs/ARCHITECTURE.md` for the target architecture diagram and wire
protocol this roadmap implements.

## Status (updated 2026-10-18, same day)

**Phase 2 is built and verified** — see its section below for exactly what
landed and one deliberate deviation from the original plan. Phases 3–5 are
still just this written plan, not built.

- `modes.py`: every `Mode` now has `compute(ctx) -> dict` (pure state, no
  `hud`/cv2 calls) and `on_action(action, ctx, **kw)` + a per-mode `KEYMAP`
  (`on_key` is now a thin shim on the base `Mode` class translating codes
  to actions via `KEYMAP`). `update()` calls `compute()` then draws via
  `hud.*`, so `app.py`'s cv2 window is behaviorally unchanged.
- New `backend/` package: `runtime.py` (`Engine` — owns the camera, calls
  `Mode.compute()`, exposes `handle_action`/`handle_prompt_submit`/
  `handle_prompt_cancel`) and `server.py` (FastAPI, one `/ws` WebSocket
  route implementing the full wire protocol from `docs/ARCHITECTURE.md`:
  `landmarks`/`view_state`/`prompt`/`recording` out, `event` in). Mounts
  `frontend/dist` as static files once Phase 3 produces it; API-only until
  then.
- **Deviation from the original plan**: `app.py` was *not* rewired to
  build its loop on top of `Engine`. `Engine` is a separate, fresh
  implementation of the same loop shape, built on the new `Mode.compute()`.
  This was a deliberate risk call — making `app.py` depend on `Engine` would
  have meant also reintroducing a `render_cv2()` step to turn `ViewState`
  back into `hud.*` calls, which is more surface area to get wrong in the
  same change that introduces the server. Net effect: `app.py`'s loop body
  and `Engine.step()` are structurally similar but not DRY — low risk now,
  worth DRYing up in a later pass once the web frontend (Phase 3) is real
  and both paths have proven stable.
- Verified: `.venv/bin/python selftest.py` is 88/88 (77 original + 11 new
  `test_modes`/`test_backend_runtime` checks exercising `compute()`, the
  `on_key`→`KEYMAP`→`on_action` shim, and the `landmarks` JSON serializer).
  `app.py` launches cleanly post-refactor. A raw WebSocket client against
  `python -m backend.server` confirmed `landmarks`/`view_state` arrive and
  a `mode.next` event actually switches the active mode.
- New deps (`fastapi`, `uvicorn[standard]`, `websockets`) added to
  `requirements.txt` and installed into `.venv` — not needed for the plain
  cv2 CLI, only for `python -m backend.server`.
- **Not yet validated**: risk #1 below (dual camera access from both the
  browser and Python at once) — still open, matters once Phase 3 starts.

## Target architecture specifics (recap)

- **Pattern**: local inference server + thin web client, single process
  tree, single machine, no containers/cloud.
- New `backend/` package wrapping the existing modules:
  - `backend/runtime.py` — `Engine` class: the refactored camera→tracker→
    mode loop, extracted from `app.py`'s current `while True:` body, shared
    by both the CLI and the server.
  - `backend/server.py` — FastAPI app, one `/ws` WebSocket route, serves
    the built `frontend/dist` as static files so `python -m backend.server`
    boots the whole app at `http://127.0.0.1:8765`.
- `app.py` stays alive as the cv2 CLI/debug fallback — not deleted.
- Wire protocol: see the table in `docs/ARCHITECTURE.md`. Full per-mode
  `view_state` payload shapes:
  - **Communicator**: `{word, sentence_text, suggestions:[str,...], detecting_letter, detecting_conf, hold_fraction, flash, type_out, reverse, reverse_cards:[{text, active}], bank_count}`
  - **Coach**: `{target_kind, target_name, score, pass, msg, session_ok, session_n, box_counts:[int x6]}`
  - **Control**: `{commands:[{name, action_type, action_value}], last_fired, last_name, bound_count}`
  - **Pointer**: `{enabled, status_lines:[{text, kind}], pinch_ratio, state}`

## Phase 2 — backend refactor (no visible behavior change yet)

1. `modes.py`:
   - Split every `Mode.update(ctx)` into `compute(ctx) -> ViewState` (pure
     state, matching the shapes above) + a separate `render_cv2(frame,
     state)` wrapper that makes the existing `hud.top_bar`/`hud.panel`/
     `hud.cards`/etc. calls — used only by `app.py`'s cv2 fallback path.
   - Rename `on_key(code, ctx)` → `on_action(action: str, ctx: Ctx, **kw) ->
     bool` on the `Mode` base class and all 4 modes. Add a `KEYMAP` dict
     (e.g. `{9: "mode.next", 96: "mode.prev", ord(' '): "communicator.space",
     8: "backspace", 127: "backspace", 13: "submit", ...}`) so `app.py`'s
     existing cv2 loop translates `cv2.waitKey` codes into the same named
     actions before calling `on_action` — guarantees the keyboard path and
     the web path share one action vocabulary and can never drift apart.
     This is a wide, mechanical refactor touching all 4 modes at once — do
     it as one atomic, carefully-tested change (see risk #4 below), not
     piecemeal.
2. `backend/runtime.py` (new): extract `app.py`'s loop body into an
   `Engine` class with a `step()` method returning `(ViewState,
   landmarks_dict, prompt_request|None, recording_state)`, parameterized by
   the camera/tracker/bank/live/recorder/mode-list it owns. Both `app.py`
   (sync, cv2 `imshow`) and `backend/server.py` (async, WebSocket) drive
   this one `Engine` — no duplicated loop logic.
3. `backend/server.py` (new): FastAPI app, one `/ws` route. Camera I/O is
   blocking, so `Engine.step()` must run in a background thread (not on the
   asyncio event loop), pushing `landmarks`/`view_state`/`prompt`/
   `recording` messages onto an `asyncio.Queue` that the WebSocket handler
   drains and forwards; the WebSocket's receive side parses incoming
   `event` JSON and calls `Engine.handle_action(...)`. Also serves
   `frontend/dist` as static files.
4. `app.py`: trimmed to build an `Engine`, call `.step()` in the existing
   loop, and keep cv2-specific rendering (`hud.*`, `cv2.imshow`,
   `cv2.waitKey`) as the dev/debug fallback — explicitly not deleted.
5. `requirements.txt`: add `fastapi`, `uvicorn[standard]`, `websockets` —
   **install into `.venv` explicitly** (`.venv/bin/pip install ...`). The
   project is pinned to Python 3.12 for MediaPipe; the system Python is
   newer (3.14) and won't see these installs if they go to system pip.

**Verification:**
- `selftest.py` stays green (77/77) — the pure-compute function signatures
  (`classify_letter`, `SignBank.distances`, etc.) don't change, only how
  `Mode.update()` is split.
- Add a `test_runtime` section (or new test file) that feeds a synthetic
  `Pose` sequence through `Engine.step()`/`Mode.compute()` without a camera
  (mirrors `selftest.py`'s existing synthetic-landmark style) and asserts
  `ViewState` fields change as expected (e.g. Coach's `score` rises when
  the synthetic pose matches the target).
- Manual regression: run `python app.py` unchanged, confirm the cv2 window
  still works exactly as before — full keyboard walkthrough of all 4 modes,
  since `on_key`→`on_action` touches every mode's input handling at once.
- Start `uvicorn backend.server:app`, confirm `landmarks`/`view_state` JSON
  arrives over a raw WebSocket test client and an `event` round-trips to a
  visible mode change.

## Phase 3 — `frontend/` (Vite + React + Tailwind)

Follow `~/.claude/DESIGN_TOOLKIT.md`'s documented workflow exactly (this
machine's global instructions require it for frontend work): install Taste
Skill + Web Design Guidelines skills, pick a `DESIGN.md` from
awesome-design-md, build with Taste Skill active, verify with Playwright
CLI, audit with Web Design Guidelines before calling a component done.

1. Scaffold: `npm create vite@latest frontend -- --template react-ts` +
   Tailwind.
2. `frontend/src/lib/ws.ts` — WebSocket client: connect, auto-reconnect,
   typed message dispatch.
3. `frontend/src/state/` — a small store (Zustand or plain context) holding
   the latest `landmarks`, `view_state`, `prompt`, `recording`.
4. `frontend/src/components/VideoOverlay.tsx` — `getUserMedia` video
   element + an absolutely-positioned `<canvas>` drawing the hand/body
   points from the `landmarks` stream, redrawn on `requestAnimationFrame`
   with interpolation between the last two received landmark frames for a
   fluid feel (not jittery at sub-30fps-over-websocket jitter) — this is
   the direct implementation of the "smooth, low-latency input" goal.
5. Mode components, each subscribing to `view_state.payload` for its mode:
   - `CommunicatorView.tsx` — caption bar, suggestion chips (clickable →
     `communicator.accept_suggestion`), live letter-detect hold-progress
     indicator, reverse-channel gloss cards.
   - `CoachView.tsx` — target sign/letter display, animated score meter,
     pass/fail toast, Leitner box-count bars.
   - `ControlView.tsx` — bound-commands table, "last fired" toast,
     learn-new-command flow driving the `PromptModal`.
   - `PointerView.tsx` — status line, pinch-ratio meter, enabled/disabled
     banner (e.g. pyautogui missing, or no macOS Accessibility permission).
6. `PromptModal.tsx` — generic modal for any `{type:"prompt", kind, label,
   initial}` message, submit/cancel wired to `prompt.submit`/`prompt.cancel`
   events.
7. `ModeSwitcher.tsx` + a global keydown handler mapping browser keys (Tab,
   backtick, letters) to the same named `action` strings used server-side —
   old muscle memory ports over without the server knowing about browser
   key codes.
8. Build: `npm run build` → `frontend/dist`, served by `backend/server.py`
   as static files (Vite dev server used only during active frontend
   development, proxying `/ws` to the FastAPI backend).

**Verification:**
- Playwright CLI: launch headed, assert the four mode views render against
  a mocked/replayed WebSocket fixture — worth adding a `backend/server.py
  --mock` mode or a standalone fixture server so UI work doesn't need a
  live camera.
- Web Design Guidelines audit pass on each new component before merging.
- Manual latency smoke test: wave a hand, eyeball overlay lag against the
  raw camera preview.

## Phase 4 — trained fingerspelling classifier

**Do not claim "fixed" in the paper until genuinely multi-person data has
been collected and a held-out-person evaluation shows improvement** — a
single-person-trained model would just reproduce today's overfitting
problem under a different name.

1. `tools/collect_fingerspell.py` (new) — adapts `recorder.py`'s "demonstrate
   3×" flow into a guided A–Z walkthrough (skip J/Z — motion-resolved today,
   don't contradict that design). Saves each rep's single-frame
   `skeleton.hand_feature` vector (not the DTW `sequence_feature` clip — a
   classifier training example should be single-frame geometry matching
   `fingerspell.py`'s existing per-frame approach) tagged with a
   `person_id` and `handedness`, to
   `data/fingerspell_dataset/<person_id>/<LETTER>_<rep>.npz`.
2. `tools/train_fingerspell.py` (new) — loads all collected examples,
   **person-held-out** train/test split (never random-split within a
   person — that would hide the exact overfitting problem being fixed),
   trains a small `sklearn.ensemble.RandomForestClassifier` or
   `MLPClassifier` (CPU-only, offline, consistent with the no-cloud ethos —
   add `scikit-learn` to `requirements.txt`), serializes to
   `data/models/fingerspell_rf.joblib` + a `fingerspell_rf.meta.json`
   (feature version, letter list, training accuracy, person count).
3. `fingerspell.py`: add `classify_letter_ml(hand, handedness)` — lazily
   loads the serialized model (module-level cache), falls back silently to
   the existing rule-based `classify_letter` if no model file is present.
   Add a single dispatcher `classify_letter_best(hand, handedness)` used by
   both call sites (`modes.py` `Communicator.compute`/`Coach._score`) so
   neither needs to know which backend is active — preserves the
   zero-dependency fallback.
4. `tools/eval_fingerspell.py` (new) — the paper's "before vs. after"
   harness: runs both `classify_letter` and the trained model over a
   held-out test split (ideally a held-out *person*) and reports
   accuracy/confusion matrix for both, written to
   `data/eval/fingerspell_comparison.json` + a plot. This produces the
   actual number for the paper.
5. `selftest.py`: add a section that (a) trains a throwaway tiny model on
   synthetic letters (reusing the existing synthetic-handshape generator
   already in `selftest.py`) and asserts `classify_letter_ml` runs and
   returns a valid `Letter`, and (b) asserts the fallback path works when
   no model file exists.

## Phase 5 — broaden landmark smoothing + measure latency

- `filters.py` needs no change (already a generic `OneEuroFilter`/
  `OneEuroPoint`/`Hysteresis`).
- New `backend/smoothing.py` (or extend `tracking.py`): wrap the tracker's
  output so each hand's landmarks pass through a per-landmark (or
  per-centroid + per-fingertip) `OneEuroPoint`, with its **own tuning
  preset** separate from Pointer mode's (fingerspelling needs fast response
  to a changed handshape; Pointer needs slow, buttery drift — don't reuse
  Pointer's cutoff/beta as-is).
- **Open design decision, resolve with a measured experiment, not a
  default**: smooth only the *display* overlay sent to the frontend, or
  also the *recognition* input (DTW/classifier)? Smoothing before
  recognition risks blurring legitimate fast transitions (e.g.
  fingerspelling J/Z's motion stroke). Run the Phase 4 eval harness's
  pattern (before/after accuracy) on this question too before deciding.
- `backend/runtime.py`: timestamp each pipeline stage (`t_capture`,
  `t_process`, `t_view_state`, `t_sent`), log to `data/latency_log.csv`.
  Add `tools/report_latency.py` for p50/p90 numbers — compare against the
  ~15–35ms gesture-to-feedback bar found in competitor research (see
  `docs/competitor_analysis.md` sources).

## Risks / sequencing dependencies

1. **Dual camera access is unproven.** The "no video frames over the wire"
   design assumes the browser (`getUserMedia`) and Python
   (`cv2.VideoCapture`) can both read the same webcam concurrently — OS/
   driver-dependent, not guaranteed (especially on macOS). Validate with a
   throwaway script at the start of Phase 2, before frontend work depends
   on it. Fallback (already in the plan): Python-side throttled JPEG
   streaming for the visible video only; landmarks are sent either way.
2. **Phase 4 data collection is a logistics dependency**, independent of
   coding time — `tools/collect_fingerspell.py` has to actually be run by
   several real volunteers. Can start in parallel with Phase 2/3 so data is
   ready by the time Phase 4 code lands. The paper cannot claim the fix
   works until this has happened with genuinely different people.
3. **Python version mismatch risk.** System Python is 3.14 (as of last
   check), `.venv` is 3.12 (MediaPipe requirement). Every new pip install
   for Phases 2/4 (FastAPI, uvicorn, websockets, scikit-learn) must
   explicitly target `.venv/bin/pip` — a system-level `pip install` will
   silently not be visible to the app.
4. **The `on_key`→`on_action` rename touches every mode and `app.py`'s
   keycode dispatch simultaneously.** Do it in one atomic Phase 2 change
   with `selftest.py` plus a manual full keyboard walkthrough of all 4
   modes in the cv2 fallback before starting Phase 3, so the frontend is
   built against a confirmed-stable action vocabulary.
5. **Smoothing-before-recognition (Phase 5) is an open question, not a
   default.** Treat it as a measured experiment (before/after accuracy),
   same pattern as the Phase 4 ML evaluation.
6. **A stray `Real Time Sign Language Translator 2/` folder** (2 PDFs,
   likely lit-review source material) sits in the project root, unrelated
   to the app, gitignored. Flagged for the user to decide whether to
   relocate it — out of scope for this rework.
7. **Frontend tooling**: confirm `node`/`npm` are available before starting
   Phase 3 (already confirmed present at planning time: node v26.10.0, npm
   v11.19.1 — re-check if executing this roadmap much later).

## Suggested execution order

Phase 2 and the start of Phase 4's data collection (item 1: build +
distribute `tools/collect_fingerspell.py` to volunteers) can run in
parallel — data collection doesn't block backend work. Phase 3 should not
start until Phase 2's `ViewState`/`on_action` contracts are stable (risk
#4). Phase 5 can start any time after Phase 2 (it only needs
`backend/runtime.py` to exist for the latency logging) but its "smooth
recognition input" question should be evaluated using the same held-out
methodology as Phase 4, so it's reasonable to do them back-to-back.
