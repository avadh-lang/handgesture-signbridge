# SignBridge

A real-time sign-language studio that runs on one webcam, fully offline.

Built on [MediaPipe](https://developers.google.com/mediapipe) Holistic (both
hands **+** a body pose) and OpenCV. Three modes over one tracking loop:

| Mode | What it does |
|---|---|
| **1 Communicator** | Fingerspell or sign → live text. Speak it aloud (`say`) or type it straight into the focused app (Slack, Notes, a search bar). `v` flips to the **reverse channel**: type a phrase, hold up big ASL-gloss cards for the other person. |
| **2 Coach** | Practise a target sign or letter, get a live accuracy score, and spaced-repetition drills (Leitner boxes) that persist across sessions. Every attempt is logged to `data/practice_log.csv`. |
| **3 Control** | A sign you invented fires a macro — open an app, send a keystroke, type a snippet, run a shell command. |
| **4 Pointer** | Hands-free mouse: point to move the cursor, pinch to click (deep pinch = right-click, hold = drag), two fingers to scroll. So you can navigate *and* communicate with no keyboard, mouse or voice. Needs `pyautogui` + Accessibility permission. |

### The part that isn't available anywhere else

**Teach it your own signs in ~10 seconds.** Demonstrate a sign three times and
it's in your personal dictionary — a moving, two-handed, body-relative sign, or
a still handshape, it doesn't matter. No training, no dataset, no network:

* each demo becomes a fixed-length **feature clip** — de-rotated, palm-scaled
  handshape descriptors + where the hands sit *relative to your torso* + how
  they move;
* recognition is nearest-template with [dynamic time warping](dtw.py), so a sign
  done a little faster or slower still matches;
* a rejection rule means an unknown gesture is ignored, not mis-fired.

```bash
python learn_sign.py "pizza"                  # 3 demos, saved to signs/
python learn_sign.py "mom" --reps 4
python learn_sign.py "flat spin" --bind app:Safari    # also bind it for Control
python learn_sign.py --list
python learn_sign.py --check "pizza"          # live distance readout for tuning
```

Your custom vocabulary works in every mode: whole-word signs drop into the
Communicator sentence, become drill targets in Coach, and trigger macros in
Control.

## Setup

Cross-platform (macOS / Windows / Linux). **Python 3.12** is recommended
everywhere — it's the newest version the pinned MediaPipe release has
verified wheels for (confirmed broken on macOS arm64 for 3.13/3.14; use 3.12
there, and fall back to it on Windows/Linux too if you hit the same).

**macOS / Linux**
```bash
cd "handgesture project"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Windows** (PowerShell)
```powershell
cd "handgesture project"
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

**Linux extra step** — voice output shells out to `espeak`/`espeak-ng`, which
isn't installed by pip:
```bash
sudo apt install espeak-ng   # Debian/Ubuntu; use your distro's package manager
```

### Permissions

| OS | Camera | Typing into other apps / Control macros |
|---|---|---|
| macOS | System Settings → Privacy & Security → Camera, allow your terminal | …→ Accessibility, allow your terminal |
| Windows | Settings → Privacy & security → Camera, allow desktop apps | works out of the box |
| Linux | your user needs to be in the `video` group | **X11 only** — Wayland blocks synthetic keystrokes/mouse input from other apps by design, so typing into other apps and Pointer mode won't work there |

## Run

Examples use the macOS/Linux venv path (`.venv/bin/python`); on Windows use
`.venv\Scripts\python` instead.

```bash
.venv/bin/python app.py                    # Communicator, Holistic tracker
.venv/bin/python app.py --mode coach
.venv/bin/python app.py --mode control
.venv/bin/python app.py --mode pointer      # hands-free mouse
.venv/bin/python app.py --tracker hands    # lighter; no body reference
.venv/bin/python app.py --camera 1 --no-flip
.venv/bin/python app.py --swap-hands       # if left/right come through mirrored
```

**Keys**

| Scope | Key | Action |
|---|---|---|
| global | `Tab` / `` ` `` | next / previous mode |
| global | `s` · `h` · `q`/`Esc` | snapshot · toggle help · quit |
| while recording a sign | `Esc` | cancel |
| Communicator | `space` · `⌫` | commit word · delete |
| Communicator | `1` `2` `3` | accept the numbered word suggestion |
| Communicator | `Enter` | speak the sentence, then clear |
| Communicator | `t` · `v` · `r` | type-into-app on/off · reverse channel · teach a word |
| Coach | `space` · `r` · `n` · `a` | submit attempt · retry · skip · add a drill |
| Control | `l` · `e` · `u` | learn + bind a command · reload `commands.json` · unbind last |
| Pointer | — | drop your hand out of frame to release the cursor |

## Web backend (in progress)

SignBridge is being reworked into a local inference server + web UI (see
`docs/ARCHITECTURE.md` and `docs/REWORK_ROADMAP.md`). The backend half is
up — the camera/MediaPipe/recognition loop is now also reachable over a
local WebSocket, independent of the `app.py` cv2 window above (which still
works exactly as before and isn't going away):

```bash
.venv/bin/python -m backend.server          # ws://127.0.0.1:8765/ws
```

There's no web frontend yet (that's Phase 3 of the roadmap) — this serves
the WebSocket API only for now. `backend/runtime.py`'s `Engine` class is
what both `app.py` and this server are meant to eventually share.

## How fingerspelling works

`fingerspell.py` classifies the ASL A–Z from a single hand geometrically. It's
solid on the well-separated letters and less sure on the genuine look-alikes
(A/S/T/M/N are all near-fists); **J** and **Z** are air-drawn and handled by the
motion layer. For rock-solid fingerspelling, record your own A–Z into the bank —
the bank is always checked first.

`glosser.py` (reverse channel) is a fast gloss, not a grammatical ASL
translator: it drops articles and the copula, points pronouns, and marks unknown
names for fingerspelling.

## No webcam? 

```bash
.venv/bin/python demo_image.py path/to/hand.jpg      # still-image gesture + letter
.venv/bin/python selftest.py                         # 77 camera-free checks
```

## Files

| File | Purpose |
|---|---|
| `skeleton.py` | Landmark normalisation, body-relative location, per-frame & sequence features |
| `dtw.py` | Banded dynamic time warping |
| `signbank.py` | Personal sign dictionary: record → match → reject |
| `live.py` | Rolling-window sign matcher with debounce |
| `fingerspell.py` | Rule-based ASL A–Z from one hand |
| `predict.py` + `data/words_en.txt` | Offline prefix word prediction |
| `speller.py` | Fingerspelling → word → sentence state machine |
| `glosser.py` | English → ASL-gloss cards |
| `leitner.py` · `sessionlog.py` | Spaced-repetition scheduler · practice CSV log |
| `tracking.py` | Unified MediaPipe Holistic / Hands tracker → `skeleton.Pose` |
| `filters.py` | One-Euro filter + hysteresis debounce |
| `pointer.py` | Hands-free mouse: move / click / drag / scroll |
| `hud.py` · `textinput.py` | OpenCV overlays · in-window text field |
| `recorder.py` | Non-blocking "demonstrate a sign N times" capture |
| `modes.py` | Communicator / Coach / Control / Pointer |
| `commands.py` + `data/commands.json` | Sign → app/key/text/shell bindings |
| `app.py` · `learn_sign.py` | Mode host · teach/list/delete/check CLI |
| `gestures.py` · `hand_tracker.py` · `actions.py` · `demo_image.py` | The original single-frame gesture recogniser (still used by `demo_image.py`) |

## Credits

The **Pointer** mode — One-Euro cursor smoothing, active-zone → screen mapping,
the pinch click/drag state machine, two-finger scroll — and the CSV practice log
are adapted from
[Prathamesh-2803/Real-time-hand-gesture-control-system](https://github.com/Prathamesh-2803/Real-time-hand-gesture-control-system),
rebuilt on top of SignBridge's `skeleton.Pose` so they share one tracker.
